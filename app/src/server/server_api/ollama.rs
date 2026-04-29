use std::{
    collections::HashMap,
    sync::OnceLock,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use ::settings::Setting;
use anyhow::{Context, Result as AnyhowResult};
use async_trait::async_trait;
use chrono::Utc;
use futures::StreamExt;
use reqwest::Url;
use serde::{Deserialize, Serialize};
use warp_graphql::scalars::time::ServerTimestamp;
use warp_multi_agent_api as maa_api;

use crate::{
    ai::{
        agent::api::ServerConversationToken,
        agent::conversation::{AIAgentConversationFormat, ServerAIConversationMetadata},
        ambient_agents::{
            AmbientAgentTask, AmbientAgentTaskId, AmbientAgentTaskState, TaskStatusMessage,
        },
        generate_code_review_content::api::{
            GenerateCodeReviewContentRequest, GenerateCodeReviewContentResponse,
        },
        llms::{
            AvailableLLMs, LLMInfo, LLMModelHost, LLMProvider, LLMSpec, LLMUsageMetadata,
            ModelsByFeature,
        },
        request_usage_model::{RequestLimitInfo, RequestLimitRefreshDuration, RequestUsageInfo},
    },
    ai_assistant::{
        AIGeneratedCommand, AIGeneratedCommandParameter, GenerateCommandsFromNaturalLanguageError,
        execution_context::WarpAiExecutionContext, requests::GenerateDialogueResult,
        utils::TranscriptPart,
    },
    drive::workflows::ai_assist::{GeneratedCommandMetadata, GeneratedCommandMetadataError},
    server::server_api::ai::{
        AIClient, AgentMessageHeader, AgentRunEvent, ArtifactDownloadResponse, AttachmentFileInfo,
        CreateFileArtifactUploadRequest, CreateFileArtifactUploadResponse, FileArtifactRecord,
        ListAgentMessagesRequest, PrepareAttachmentUploadsResponse, ReadAgentMessageResponse,
        ReportAgentEventRequest, ReportAgentEventResponse, SendAgentMessageRequest,
        SendAgentMessageResponse, SpawnAgentRequest, SpawnAgentResponse, TaskAttachment,
        TaskListFilter, TaskStatusUpdate,
    },
    settings::{self, AISettings, LocalLLMProvider},
    terminal::{model::block::SerializedBlock, view::ambient_agent::OLLAMA_COMPLETED_PREFIX},
};
use ai::index::full_source_code_embedding::{
    self, ContentHash, EmbeddingConfig, NodeHash, RepoMetadata, store_client::IntermediateNode,
};
use session_sharing_protocol::common::SessionId;
use uuid::Uuid;
use warp_graphql::ai::AgentTaskState;
use warp_graphql::queries::get_scheduled_agent_history::ScheduledAgentHistory;
use warp_multi_agent_api::ConversationData;

// ---- Ollama config ----

#[derive(Clone, Debug)]
pub(crate) struct OllamaConfig {
    pub(crate) base_url: String,
    pub(crate) model: String,
}

impl OllamaConfig {
    pub(crate) fn from_ai_settings(ai_settings: &AISettings) -> Option<Self> {
        if *ai_settings.local_llm_provider.value() != LocalLLMProvider::Ollama {
            return None;
        }

        let model = ai_settings.ollama_model.value().trim();
        if model.is_empty() {
            return None;
        }

        let base_url = ai_settings
            .ollama_base_url
            .value()
            .trim()
            .trim_end_matches('/');

        Some(Self {
            base_url: if base_url.is_empty() {
                "http://127.0.0.1:11434".to_string()
            } else {
                base_url.to_string()
            },
            model: model.to_string(),
        })
    }

    fn from_env() -> Option<Self> {
        let model = std::env::var("WARP_OLLAMA_MODEL")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())?;
        let base_url = std::env::var("WARP_OLLAMA_BASE_URL")
            .ok()
            .map(|value| value.trim().trim_end_matches('/').to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "http://127.0.0.1:11434".to_string());

        Some(Self { base_url, model })
    }

    #[cfg(not(target_family = "wasm"))]
    fn from_settings_file() -> Option<Option<Self>> {
        let path = settings::user_preferences_toml_file_path();
        let contents = std::fs::read_to_string(path).ok()?;
        let root = contents.parse::<toml::Value>().ok()?;

        let provider = root
            .get("agents")
            .and_then(|value| value.get("warp_agent"))
            .and_then(|value| value.get("models"))
            .and_then(|value| value.get("local_llm_provider"))
            .and_then(|value| value.as_str());

        match provider {
            Some("warp") => Some(None),
            Some("ollama") => {
                let ollama = root
                    .get("agents")
                    .and_then(|value| value.get("warp_agent"))
                    .and_then(|value| value.get("models"))
                    .and_then(|value| value.get("ollama"));

                let model = ollama
                    .and_then(|value| value.get("model"))
                    .and_then(|value| value.as_str())
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_string);
                let base_url = ollama
                    .and_then(|value| value.get("base_url"))
                    .and_then(|value| value.as_str())
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .unwrap_or("http://127.0.0.1:11434")
                    .trim_end_matches('/')
                    .to_string();

                Some(model.map(|model| Self { base_url, model }))
            }
            Some(other) => {
                log::warn!("Unknown local_llm_provider setting value: {other}");
                Some(None)
            }
            None => None,
        }
    }

    #[cfg(target_family = "wasm")]
    fn from_settings_file() -> Option<Option<Self>> {
        None
    }

    pub(crate) fn resolve() -> Option<Self> {
        match Self::from_settings_file() {
            Some(config) => config,
            None => Self::from_env(),
        }
    }

    fn from_model_id(model_id: &str) -> Option<Self> {
        let model = model_id.strip_prefix("ollama:")?.trim();
        if model.is_empty() {
            return None;
        }

        let base_url = Self::resolve()
            .map(|config| config.base_url)
            .unwrap_or_else(|| "http://127.0.0.1:11434".to_string());

        Some(Self {
            base_url,
            model: model.to_string(),
        })
    }
}

// ---- Local agent state ----

#[derive(Clone, Debug)]
enum LocalAgentStatus {
    Running { current_action: String },
    Done { final_output: String },
    Failed { error: String },
    Cancelled,
}

struct LocalAgentState {
    task_id: AmbientAgentTaskId,
    prompt: String,
    created_at: chrono::DateTime<chrono::Utc>,
    status: LocalAgentStatus,
    cancel_flag: Arc<AtomicBool>,
}

type LocalRunsInner = HashMap<AmbientAgentTaskId, Arc<parking_lot::Mutex<LocalAgentState>>>;
type LocalRuns = Arc<parking_lot::Mutex<LocalRunsInner>>;

fn live_ollama_config() -> &'static parking_lot::RwLock<Option<OllamaConfig>> {
    static LIVE_OLLAMA_CONFIG: OnceLock<parking_lot::RwLock<Option<OllamaConfig>>> =
        OnceLock::new();
    LIVE_OLLAMA_CONFIG.get_or_init(|| parking_lot::RwLock::new(OllamaConfig::resolve()))
}

// ---- OllamaAIClient ----

pub(crate) struct OllamaAIClient {
    server_api: Arc<super::ServerApi>,
    http_client: reqwest::Client,
    local_runs: LocalRuns,
    active_config: Arc<parking_lot::RwLock<Option<OllamaConfig>>>,
}

impl OllamaAIClient {
    pub(crate) fn new(server_api: Arc<super::ServerApi>) -> AnyhowResult<Self> {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(90))
            .build()
            .context("Failed to create Ollama HTTP client")?;

        Ok(Self {
            server_api,
            http_client,
            local_runs: Arc::new(parking_lot::Mutex::new(HashMap::new())),
            active_config: Arc::new(parking_lot::RwLock::new(
                live_ollama_config().read().clone(),
            )),
        })
    }

    pub(crate) fn config_from_settings(ai_settings: &AISettings) -> Option<OllamaConfig> {
        OllamaConfig::from_ai_settings(ai_settings)
    }

    fn active_config(&self) -> Option<OllamaConfig> {
        self.active_config.read().clone()
    }

    pub(crate) fn set_active_config(&self, config: Option<OllamaConfig>) {
        *self.active_config.write() = config.clone();
        *live_ollama_config().write() = config;
    }

    fn model_choices(&self, config: &OllamaConfig) -> AnyhowResult<ModelsByFeature> {
        let local_info = || LLMInfo {
            display_name: format!("{} (Ollama)", config.model),
            base_model_name: config.model.clone(),
            id: format!("ollama:{}", config.model).into(),
            reasoning_level: None,
            usage_metadata: LLMUsageMetadata {
                request_multiplier: 1,
                credit_multiplier: None,
            },
            description: Some("Local Ollama model".to_string()),
            disable_reason: None,
            vision_supported: false,
            spec: Some(LLMSpec {
                cost: 0.0,
                quality: 0.5,
                speed: 0.5,
            }),
            provider: LLMProvider::Unknown,
            host_configs: HashMap::from([(
                LLMModelHost::DirectApi,
                crate::ai::llms::RoutingHostConfig {
                    enabled: true,
                    model_routing_host: LLMModelHost::DirectApi,
                },
            )]),
            discount_percentage: None,
        };

        let available = AvailableLLMs::new(
            format!("ollama:{}", config.model).into(),
            [local_info()],
            None,
        )?;

        Ok(ModelsByFeature {
            agent_mode: available.clone(),
            coding: available.clone(),
            cli_agent: Some(available.clone()),
            computer_use: Some(available),
        })
    }

    fn unlimited_request_usage(&self) -> RequestUsageInfo {
        RequestUsageInfo {
            request_limit_info: RequestLimitInfo {
                limit: 999_999,
                num_requests_used_since_refresh: 0,
                next_refresh_time: ServerTimestamp::new(Utc::now() + chrono::Duration::days(3650)),
                is_unlimited: true,
                request_limit_refresh_duration: RequestLimitRefreshDuration::Monthly,
                is_unlimited_voice: true,
                voice_request_limit: 999_999,
                voice_requests_used_since_last_refresh: 0,
                is_unlimited_codebase_indices: true,
                max_codebase_indices: 999_999,
                max_files_per_repo: 999_999,
                embedding_generation_batch_size: 100,
            },
            bonus_grants: vec![],
        }
    }

    async fn chat(
        &self,
        config: &OllamaConfig,
        messages: Vec<OllamaChatMessage>,
    ) -> AnyhowResult<String> {
        chat_with_config(&self.http_client, config, messages).await
    }

    fn strip_code_fences(text: &str) -> &str {
        let trimmed = text.trim();
        if let Some(stripped) = trimmed.strip_prefix("```") {
            let without_lang = stripped
                .split_once('\n')
                .map(|(_, rest)| rest)
                .unwrap_or(stripped);
            return without_lang
                .strip_suffix("```")
                .unwrap_or(without_lang)
                .trim();
        }
        trimmed
    }

    fn parse_generated_commands(&self, text: &str) -> AnyhowResult<Vec<AIGeneratedCommand>> {
        #[derive(Deserialize)]
        struct CommandsEnvelope {
            commands: Vec<GeneratedCommandWire>,
        }

        #[derive(Deserialize)]
        struct GeneratedCommandWire {
            command: String,
            description: String,
            #[serde(default)]
            parameters: Vec<GeneratedCommandParameterWire>,
        }

        #[derive(Deserialize)]
        struct GeneratedCommandParameterWire {
            id: String,
            description: String,
        }

        let payload = Self::strip_code_fences(text);
        let commands = serde_json::from_str::<Vec<GeneratedCommandWire>>(payload)
            .or_else(|_| serde_json::from_str::<CommandsEnvelope>(payload).map(|env| env.commands))
            .context("Failed to parse Ollama command suggestions JSON")?;

        Ok(commands
            .into_iter()
            .map(|command| {
                AIGeneratedCommand::new(
                    command.command,
                    command.description,
                    command
                        .parameters
                        .into_iter()
                        .map(|parameter| {
                            AIGeneratedCommandParameter::new(parameter.id, parameter.description)
                        })
                        .collect(),
                )
            })
            .collect())
    }

    fn make_local_ambient_task(state: &LocalAgentState) -> AmbientAgentTask {
        let (task_state, status_message, is_sandbox) = match &state.status {
            LocalAgentStatus::Running { current_action } => (
                AmbientAgentTaskState::InProgress,
                Some(TaskStatusMessage {
                    message: current_action.clone(),
                }),
                true,
            ),
            LocalAgentStatus::Done { final_output } => (
                AmbientAgentTaskState::Error,
                Some(TaskStatusMessage {
                    message: format!("{OLLAMA_COMPLETED_PREFIX}{final_output}"),
                }),
                false,
            ),
            LocalAgentStatus::Failed { error } => (
                AmbientAgentTaskState::Error,
                Some(TaskStatusMessage {
                    message: format!("Ollama agent error: {}", error),
                }),
                false,
            ),
            LocalAgentStatus::Cancelled => (AmbientAgentTaskState::Cancelled, None, false),
        };

        AmbientAgentTask {
            task_id: state.task_id,
            parent_run_id: None,
            title: format!(
                "Ollama: {}",
                state.prompt.chars().take(60).collect::<String>()
            ),
            state: task_state,
            prompt: state.prompt.clone(),
            created_at: state.created_at,
            started_at: Some(state.created_at),
            updated_at: chrono::Utc::now(),
            status_message,
            source: None,
            session_id: None,
            session_link: None,
            creator: None,
            conversation_id: None,
            request_usage: None,
            is_sandbox_running: is_sandbox,
            agent_config_snapshot: None,
            artifacts: vec![],
        }
    }
}

// ---- Chat wire types (non-tool) ----

#[derive(Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaChatMessage>,
    stream: bool,
}

#[derive(Serialize)]
struct OllamaChatMessage {
    role: &'static str,
    content: String,
}

#[derive(Deserialize)]
struct OllamaChatResponse {
    message: OllamaChatResponseMessage,
}

#[derive(Deserialize)]
struct OllamaChatResponseMessage {
    content: String,
}

// ---- Tool-calling wire types ----

#[derive(Serialize, Deserialize, Clone)]
struct OllamaAgentMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<OllamaToolCall>,
}

impl OllamaAgentMessage {
    fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(content.into()),
            tool_calls: vec![],
        }
    }
    fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(content.into()),
            tool_calls: vec![],
        }
    }
    fn tool_result(content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.into()),
            tool_calls: vec![],
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct OllamaToolCall {
    function: OllamaToolCallFunction,
}

#[derive(Serialize, Deserialize, Clone)]
struct OllamaToolCallFunction {
    name: String,
    arguments: serde_json::Value,
}

#[derive(Serialize)]
struct OllamaAgentChatRequest {
    model: String,
    messages: Vec<OllamaAgentMessage>,
    tools: Vec<serde_json::Value>,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Serialize)]
struct OllamaOptions {
    num_predict: i32,
}

#[derive(Deserialize)]
struct OllamaAgentChatResponse {
    message: OllamaAgentMessage,
}

fn normalize_ollama_assistant_message(message: OllamaAgentMessage) -> OllamaAgentMessage {
    if !message.tool_calls.is_empty() {
        return message;
    }

    let Some(content) = message.content.as_deref() else {
        return message;
    };

    let trimmed = content.trim();
    let Some(tool_call) = parse_tool_call_from_content(trimmed) else {
        return message;
    };

    OllamaAgentMessage {
        tool_calls: vec![tool_call],
        content: None,
        ..message
    }
}

fn parse_tool_call_from_content(content: &str) -> Option<OllamaToolCall> {
    let parsed = parse_tool_call_json(content)
        .or_else(|| extract_json_object(content).and_then(|json| parse_tool_call_json(&json)))
        .or_else(|| parse_function_style_tool_call(content))?;

    Some(parsed)
}

fn parse_function_style_tool_call(content: &str) -> Option<OllamaToolCall> {
    let open_paren = content.find('(')?;
    let close_paren = content.rfind(')')?;
    if close_paren <= open_paren {
        return None;
    }

    let name = content[..open_paren].trim();
    if name.is_empty() {
        return None;
    }

    let args_src = content[open_paren + 1..close_paren].trim();
    let arguments = if args_src.is_empty() {
        serde_json::Value::Object(serde_json::Map::new())
    } else {
        parse_function_style_arguments(args_src)?
    };

    Some(OllamaToolCall {
        function: OllamaToolCallFunction {
            name: name.to_string(),
            arguments,
        },
    })
}

fn parse_function_style_arguments(args_src: &str) -> Option<serde_json::Value> {
    let mut args = serde_json::Map::new();
    let mut current = String::new();
    let mut parts = Vec::new();
    let mut in_quotes = false;
    let mut quote_char = '\0';
    let mut escape = false;

    for ch in args_src.chars() {
        if escape {
            current.push(ch);
            escape = false;
            continue;
        }

        if ch == '\\' && in_quotes {
            current.push(ch);
            escape = true;
            continue;
        }

        if in_quotes {
            if ch == quote_char {
                in_quotes = false;
            }
            current.push(ch);
            continue;
        }

        match ch {
            '"' | '\'' => {
                in_quotes = true;
                quote_char = ch;
                current.push(ch);
            }
            ',' => {
                if !current.trim().is_empty() {
                    parts.push(current.trim().to_string());
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }

    if !current.trim().is_empty() {
        parts.push(current.trim().to_string());
    }

    for part in parts {
        let (key, raw_value) = part.split_once('=')?;
        let key = key.trim();
        let raw_value = raw_value.trim();
        if key.is_empty() {
            return None;
        }

        let value = if let Some(unquoted) = raw_value
            .strip_prefix('"')
            .and_then(|value| value.strip_suffix('"'))
            .or_else(|| raw_value.strip_prefix('\'').and_then(|value| value.strip_suffix('\'')))
        {
            serde_json::Value::String(unquoted.replace("\\\"", "\"").replace("\\'", "'"))
        } else if let Ok(boolean) = raw_value.parse::<bool>() {
            serde_json::Value::Bool(boolean)
        } else if let Ok(integer) = raw_value.parse::<i64>() {
            serde_json::Value::Number(integer.into())
        } else {
            serde_json::Value::String(raw_value.to_string())
        };

        args.insert(key.to_string(), value);
    }

    Some(serde_json::Value::Object(args))
}

fn try_solve_simple_arithmetic(prompt: &str) -> Option<String> {
    let normalized = prompt.trim().trim_end_matches('?').trim();
    let expression = normalized
        .strip_prefix("What is ")
        .or_else(|| normalized.strip_prefix("what is "))
        .unwrap_or(normalized)
        .trim();

    let tokens: Vec<&str> = expression.split_whitespace().collect();
    if tokens.len() != 3 {
        return None;
    }

    let lhs = tokens[0].parse::<i64>().ok()?;
    let rhs = tokens[2].parse::<i64>().ok()?;
    let value = match tokens[1] {
        "+" => lhs.checked_add(rhs)?,
        "-" => lhs.checked_sub(rhs)?,
        "*" | "x" | "X" => lhs.checked_mul(rhs)?,
        "/" => {
            if rhs == 0 {
                return None;
            }
            lhs.checked_div(rhs)?
        }
        _ => return None,
    };

    Some(value.to_string())
}

fn parse_tool_call_json(content: &str) -> Option<OllamaToolCall> {
    let value: serde_json::Value = serde_json::from_str(content).ok()?;
    tool_call_from_value(&value)
}

fn extract_json_object(content: &str) -> Option<String> {
    let start = content.find('{')?;
    let end = content.rfind('}')?;
    (start < end).then(|| content[start..=end].to_string())
}

fn tool_call_from_value(value: &serde_json::Value) -> Option<OllamaToolCall> {
    let object = value.as_object()?;

    if let Some(function) = object.get("function") {
        let function = function.as_object()?;
        let name = function.get("name")?.as_str()?.to_string();
        let arguments = function
            .get("arguments")
            .cloned()
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
        return Some(OllamaToolCall {
            function: OllamaToolCallFunction { name, arguments },
        });
    }

    let name = object
        .get("name")
        .or_else(|| object.get("tool"))
        .and_then(|value| value.as_str())?
        .to_string();
    let arguments = object
        .get("parameters")
        .or_else(|| object.get("arguments"))
        .cloned()
        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

    Some(OllamaToolCall {
        function: OllamaToolCallFunction { name, arguments },
    })
}

async fn chat_with_config(
    http_client: &reqwest::Client,
    config: &OllamaConfig,
    messages: Vec<OllamaChatMessage>,
) -> AnyhowResult<String> {
    let url =
        Url::parse(&format!("{}/api/chat", config.base_url)).context("Invalid Ollama base URL")?;

    let response = http_client
        .post(url)
        .json(&OllamaChatRequest {
            model: config.model.clone(),
            messages,
            stream: false,
        })
        .send()
        .await
        .context("Failed to call Ollama")?
        .error_for_status()
        .context("Ollama returned an error response")?
        .json::<OllamaChatResponse>()
        .await
        .context("Failed to decode Ollama response")?;

    Ok(response.message.content)
}

async fn run_ollama_tool_loop(
    config: &OllamaConfig,
    http_client: &reqwest::Client,
    mut messages: Vec<OllamaAgentMessage>,
) -> AnyhowResult<String> {
    let tools = agent_tools();
    let url =
        Url::parse(&format!("{}/api/chat", config.base_url)).context("Invalid Ollama base URL")?;

    const MAX_ITERATIONS: usize = 30;
    let mut iterations = 0;

    loop {
        iterations += 1;
        if iterations > MAX_ITERATIONS {
            anyhow::bail!(
                "Agent exceeded maximum iterations (30 tool calls). Please try a simpler task."
            );
        }

        let request = OllamaAgentChatRequest {
            model: config.model.clone(),
            messages: messages.clone(),
            tools: tools.clone(),
            stream: false,
            options: OllamaOptions { num_predict: 4096 },
        };

        let response = http_client
            .post(url.clone())
            .timeout(Duration::from_secs(120))
            .json(&request)
            .send()
            .await
            .context("Failed to reach Ollama")?;

        let parsed: OllamaAgentChatResponse = response
            .error_for_status()
            .context("Ollama returned an error response")?
            .json()
            .await
            .context("Failed to parse Ollama response")?;

        let assistant_msg = normalize_ollama_assistant_message(parsed.message.clone());
        if assistant_msg.tool_calls.is_empty() {
            return Ok(assistant_msg.content.unwrap_or_default());
        }

        messages.push(assistant_msg.clone());
        for tool_call in &assistant_msg.tool_calls {
            let result =
                execute_tool(&tool_call.function.name, &tool_call.function.arguments).await;
            messages.push(OllamaAgentMessage::tool_result(result));
        }
    }
}

pub(crate) async fn generate_local_multi_agent_output(
    request: &maa_api::Request,
) -> Result<Option<super::AIOutputStream<maa_api::ResponseEvent>>, Arc<super::AIApiError>> {
    let request_model_id = request
        .settings
        .as_ref()
        .and_then(|settings| settings.model_config.as_ref())
        .map(|config| config.base.clone());

    let config = live_ollama_config().read().clone().or_else(|| {
        request_model_id
            .as_deref()
            .and_then(OllamaConfig::from_model_id)
    });
    let Some(config) = config else {
        return Ok(None);
    };
    let model_id = format!("ollama:{}", config.model);

    let messages = build_multi_agent_agent_messages(request);
    if messages
        .iter()
        .all(|message| message.content.as_deref().unwrap_or("").trim().is_empty())
    {
        return Ok(None);
    }

    let http_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(90))
        .build()
        .map_err(|error| Arc::new(super::AIApiError::Other(error.into())))?;

    let answer = run_ollama_tool_loop(&config, &http_client, messages)
        .await
        .map_err(|error| Arc::new(super::AIApiError::Other(error)))?;

    let request_id = Uuid::new_v4().to_string();
    let conversation_id = request
        .metadata
        .as_ref()
        .map(|metadata| metadata.conversation_id.clone())
        .filter(|id| !id.is_empty())
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    let existing_task_id = request
        .task_context
        .as_ref()
        .and_then(|context| context.tasks.first())
        .map(|task| task.id.clone());
    let task_id = existing_task_id
        .clone()
        .unwrap_or_else(|| "root-task".to_string());

    let mut events = vec![Ok(maa_api::ResponseEvent {
        r#type: Some(maa_api::response_event::Type::Init(
            maa_api::response_event::StreamInit {
                request_id: request_id.clone(),
                conversation_id,
                run_id: String::new(),
            },
        )),
    })];

    if existing_task_id.is_none() {
        events.push(Ok(wrap_action_event(
            maa_api::client_action::Action::CreateTask(maa_api::client_action::CreateTask {
                task: Some(maa_api::Task {
                    id: task_id.clone(),
                    messages: vec![],
                    dependencies: None,
                    description: String::new(),
                    summary: String::new(),
                    server_data: String::new(),
                }),
            }),
        )));
    }

    events.push(Ok(wrap_action_event(
        maa_api::client_action::Action::AddMessagesToTask(
            maa_api::client_action::AddMessagesToTask {
                task_id: task_id.clone(),
                messages: vec![
                    maa_api::Message {
                        id: format!("{request_id}-output"),
                        task_id: task_id.clone(),
                        server_message_data: String::new(),
                        citations: vec![],
                        message: Some(maa_api::message::Message::AgentOutput(
                            maa_api::message::AgentOutput {
                                text: answer.trim().to_string(),
                            },
                        )),
                        request_id: request_id.clone(),
                        timestamp: None,
                    },
                    maa_api::Message {
                        id: format!("{request_id}-model"),
                        task_id,
                        server_message_data: String::new(),
                        citations: vec![],
                        message: Some(maa_api::message::Message::ModelUsed(
                            maa_api::message::ModelUsed {
                                model_id: model_id.clone(),
                                model_display_name: format!("{} (Ollama)", config.model),
                                is_fallback: false,
                            },
                        )),
                        request_id,
                        timestamp: None,
                    },
                ],
            },
        ),
    )));

    events.push(Ok(maa_api::ResponseEvent {
        r#type: Some(maa_api::response_event::Type::Finished(
            maa_api::response_event::StreamFinished {
                reason: Some(maa_api::response_event::stream_finished::Reason::Done(
                    maa_api::response_event::stream_finished::Done {},
                )),
                conversation_usage_metadata: None,
                token_usage: vec![],
                should_refresh_model_config: false,
                request_cost: None,
            },
        )),
    }));

    Ok(Some(futures::stream::iter(events).boxed()))
}

fn wrap_action_event(action: maa_api::client_action::Action) -> maa_api::ResponseEvent {
    maa_api::ResponseEvent {
        r#type: Some(maa_api::response_event::Type::ClientActions(
            maa_api::response_event::ClientActions {
                actions: vec![maa_api::ClientAction {
                    action: Some(action),
                }],
            },
        )),
    }
}

fn build_multi_agent_agent_messages(request: &maa_api::Request) -> Vec<OllamaAgentMessage> {
    let mut messages = vec![OllamaAgentMessage::system(
        "You are a helpful AI assistant with access to tools. \
Use the available tools to inspect local files, run shell commands, and gather context when needed. \
Do not claim you lack tool access when tools would help. \
Answer directly when the user asks for simple reasoning or general knowledge that does not require local context or command execution. \
Only call tools when they are actually needed. \
When you decide to call a tool, return the tool call itself instead of describing it in prose or wrapping it in markdown. \
After completing the task, provide a clear summary of what you did and the results.",
    )];

    if let Some(task_context) = &request.task_context {
        for task in &task_context.tasks {
            for message in &task.messages {
                match &message.message {
                    Some(maa_api::message::Message::UserQuery(user_query)) => {
                        messages.push(OllamaAgentMessage::user(user_query.query.clone()));
                    }
                    Some(maa_api::message::Message::AgentOutput(output)) => {
                        messages.push(OllamaAgentMessage {
                            role: "assistant".to_string(),
                            content: Some(output.text.clone()),
                            tool_calls: vec![],
                        });
                    }
                    _ => {}
                }
            }
        }
    }

    append_request_input_messages(&mut messages, request.input.as_ref());
    messages
}

fn append_request_input_messages(
    messages: &mut Vec<OllamaAgentMessage>,
    input: Option<&maa_api::request::Input>,
) {
    let Some(input) = input else {
        return;
    };

    match input.r#type.as_ref() {
        Some(maa_api::request::input::Type::UserInputs(user_inputs)) => {
            for input in &user_inputs.inputs {
                match input.input.as_ref() {
                    Some(maa_api::request::input::user_inputs::user_input::Input::UserQuery(
                        user_query,
                    )) => messages.push(OllamaAgentMessage::user(user_query.query.clone())),
                    Some(
                        maa_api::request::input::user_inputs::user_input::Input::CliAgentUserQuery(
                            user_query,
                        ),
                    ) => {
                        if let Some(user_query) = &user_query.user_query {
                            messages.push(OllamaAgentMessage::user(user_query.query.clone()));
                        }
                    }
                    Some(maa_api::request::input::user_inputs::user_input::Input::MessagesReceivedFromAgents(
                        received,
                    )) => {
                        for message in &received.messages {
                            messages.push(OllamaAgentMessage::user(message.message_body.clone()));
                        }
                    }
                    _ => {}
                }
            }
        }
        Some(maa_api::request::input::Type::QueryWithCannedResponse(query)) => {
            messages.push(OllamaAgentMessage::user(query.query.clone()));
        }
        Some(maa_api::request::input::Type::AutoCodeDiffQuery(query)) => {
            messages.push(OllamaAgentMessage::user(query.query.clone()));
        }
        Some(maa_api::request::input::Type::CreateNewProject(query)) => {
            messages.push(OllamaAgentMessage::user(query.query.clone()));
        }
        _ => {}
    }
}

// ---- Local agent loop ----

fn agent_tools() -> Vec<serde_json::Value> {
    serde_json::from_str(r#"[
      {
        "type": "function",
        "function": {
          "name": "run_command",
          "description": "Execute a shell command and return its stdout and stderr. Use this for running scripts, compiling code, checking output, etc.",
          "parameters": {
            "type": "object",
            "properties": {
              "command": {"type": "string", "description": "The shell command to run"},
              "working_dir": {"type": "string", "description": "Optional working directory (absolute path)"}
            },
            "required": ["command"]
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "read_file",
          "description": "Read the full contents of a file at a given path",
          "parameters": {
            "type": "object",
            "properties": {
              "path": {"type": "string", "description": "Absolute or relative path to the file"}
            },
            "required": ["path"]
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "write_file",
          "description": "Write or overwrite a file with given content",
          "parameters": {
            "type": "object",
            "properties": {
              "path": {"type": "string", "description": "Absolute or relative path to the file"},
              "content": {"type": "string", "description": "Content to write into the file"}
            },
            "required": ["path", "content"]
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "list_files",
          "description": "List files in a directory, with optional glob pattern",
          "parameters": {
            "type": "object",
            "properties": {
              "path": {"type": "string", "description": "Directory path to list"},
              "pattern": {"type": "string", "description": "Optional glob pattern to filter files (e.g. '*.rs')"}
            },
            "required": ["path"]
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "search_in_files",
          "description": "Search for a text pattern inside files using grep-like search",
          "parameters": {
            "type": "object",
            "properties": {
              "pattern": {"type": "string", "description": "Text or regex pattern to search for"},
              "path": {"type": "string", "description": "Directory or file path to search in"},
              "case_sensitive": {"type": "boolean", "description": "Whether the search is case-sensitive (default true)"}
            },
            "required": ["pattern", "path"]
          }
        }
      }
    ]"#).unwrap_or_default()
}

async fn execute_tool(name: &str, args: &serde_json::Value) -> String {
    execute_tool_with_defaults(name, args, None).await
}

async fn execute_tool_with_defaults(
    name: &str,
    args: &serde_json::Value,
    default_working_directory: Option<&str>,
) -> String {
    match name {
        "run_command" => {
            let command = match args.get("command").and_then(|v| v.as_str()) {
                Some(c) => c.to_string(),
                None => return "Error: 'command' argument is required".to_string(),
            };
            let working_dir = args
                .get("working_dir")
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .or_else(|| default_working_directory.map(str::to_string));
            run_shell_command(command, working_dir).await
        }

        "read_file" => {
            let path = match args.get("path").and_then(|v| v.as_str()) {
                Some(p) => p.to_string(),
                None => return "Error: 'path' argument is required".to_string(),
            };
            let default_working_directory = default_working_directory.map(str::to_string);
            match tokio::task::spawn_blocking(move || {
                let resolved = resolve_tool_path(&path, default_working_directory.as_deref());
                std::fs::read_to_string(&resolved)
                    .map(|content| truncate_str(&content, 16000).to_string())
                    .unwrap_or_else(|err| {
                        format!("Error reading file '{}': {err}", resolved.display())
                    })
            })
            .await
            {
                Ok(result) => result,
                Err(err) => format!("Error: {err}"),
            }
        }

        "write_file" => {
            let path = match args.get("path").and_then(|v| v.as_str()) {
                Some(p) => p.to_string(),
                None => return "Error: 'path' argument is required".to_string(),
            };
            let content = match args.get("content").and_then(|v| v.as_str()) {
                Some(c) => c.to_string(),
                None => return "Error: 'content' argument is required".to_string(),
            };
            let default_working_directory = default_working_directory.map(str::to_string);
            match tokio::task::spawn_blocking(move || {
                let resolved = resolve_tool_path(&path, default_working_directory.as_deref());
                if let Some(parent) = resolved.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let len = content.len();
                std::fs::write(&resolved, &content)
                    .map(|()| {
                        format!(
                            "Successfully wrote {len} bytes to '{}'",
                            resolved.display()
                        )
                    })
                    .unwrap_or_else(|err| {
                        format!("Error writing file '{}': {err}", resolved.display())
                    })
            })
            .await
            {
                Ok(result) => result,
                Err(err) => format!("Error: {err}"),
            }
        }

        "list_files" => {
            let path = match args.get("path").and_then(|v| v.as_str()) {
                Some(p) => p.to_string(),
                None => return "Error: 'path' argument is required".to_string(),
            };
            let pattern = args
                .get("pattern")
                .and_then(|v| v.as_str())
                .unwrap_or("*")
                .to_string();
            // Use run_command: let the shell handle the listing cross-platform
            let cmd = if cfg!(target_os = "windows") {
                format!("dir /B /S \"{path}\"")
            } else {
                format!("find '{path}' -name '{pattern}' 2>/dev/null | head -200")
            };
            run_shell_command(cmd, default_working_directory.map(str::to_string)).await
        }

        "search_in_files" => {
            let pattern = match args.get("pattern").and_then(|v| v.as_str()) {
                Some(p) => p.to_string(),
                None => return "Error: 'pattern' argument is required".to_string(),
            };
            let path = match args.get("path").and_then(|v| v.as_str()) {
                Some(p) => p.to_string(),
                None => return "Error: 'path' argument is required".to_string(),
            };
            let case_insensitive = !args
                .get("case_sensitive")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let cmd = if cfg!(target_os = "windows") {
                let flag = if case_insensitive { "/I" } else { "" };
                format!("findstr /R/S {flag} \"{pattern}\" \"{path}\"")
            } else {
                let flag = if case_insensitive { "-i" } else { "" };
                format!("grep -rn {flag} '{pattern}' '{path}' 2>/dev/null | head -100")
            };
            run_shell_command(cmd, default_working_directory.map(str::to_string)).await
        }

        other => format!("Error: unknown tool '{other}'"),
    }
}

fn resolve_tool_path(path: &str, default_working_directory: Option<&str>) -> std::path::PathBuf {
    let candidate = std::path::PathBuf::from(path);
    if candidate.is_absolute() {
        candidate
    } else if let Some(working_directory) = default_working_directory {
        std::path::Path::new(working_directory).join(candidate)
    } else {
        candidate
    }
}

async fn run_shell_command(command: String, working_dir: Option<String>) -> String {
    match tokio::task::spawn_blocking(move || {
        let mut cmd = if cfg!(target_os = "windows") {
            let mut c = std::process::Command::new("cmd");
            c.args(["/C", &command]);
            c
        } else {
            let mut c = std::process::Command::new("sh");
            c.args(["-c", &command]);
            c
        };

        if let Some(dir) = working_dir {
            cmd.current_dir(dir);
        }

        cmd.output()
    })
    .await
    {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let status = output.status.code().unwrap_or(-1);
            let mut result = format!("Exit code: {status}");
            if !stdout.is_empty() {
                result.push_str(&format!("\nSTDOUT:\n{}", truncate_str(&stdout, 8000)));
            }
            if !stderr.is_empty() {
                result.push_str(&format!("\nSTDERR:\n{}", truncate_str(&stderr, 4000)));
            }
            result
        }
        Ok(Err(err)) => format!("Error running command: {err}"),
        Err(err) => format!("Error spawning command task: {err}"),
    }
}

fn truncate_str<'a>(s: &'a str, max_bytes: usize) -> &'a str {
    if s.len() <= max_bytes {
        return s;
    }
    // Truncate at a char boundary
    let mut boundary = max_bytes;
    while !s.is_char_boundary(boundary) {
        boundary -= 1;
    }
    &s[..boundary]
}

async fn run_ollama_agent_loop(
    config: OllamaConfig,
    http_client: reqwest::Client,
    task_id: AmbientAgentTaskId,
    prompt: String,
    working_directory: Option<String>,
    local_runs: LocalRuns,
) {
    if let Some(answer) = try_solve_simple_arithmetic(&prompt) {
        update_local_run_status(
            &local_runs,
            &task_id,
            LocalAgentStatus::Done {
                final_output: answer,
            },
        );
        return;
    }

    let tools = agent_tools();
    let base_url = config.base_url.clone();
    let model = config.model.clone();

    let cwd_context = working_directory
        .as_deref()
        .map(|cwd| format!("Current working directory: {cwd}. "))
        .unwrap_or_default();
    let system_prompt = format!(
        "\
You are a helpful AI assistant with access to tools. \
{cwd_context}\
Use the available tools to complete the user's task when local context, file inspection, or command execution is required. \
Assume file paths and shell commands should use the current working directory unless the user specifies otherwise. \
Answer directly when the task can be completed from reasoning alone, such as simple math or explanation requests. \
Only call tools when they are actually needed. \
Do not claim you lack filesystem or terminal access. \
When you decide to call a tool, return the tool call itself instead of describing it in prose or wrapping it in markdown. \
After completing the task, provide a clear summary of what you did and the results. \
If you encounter errors, explain what went wrong and what you tried."
    );

    let mut messages: Vec<OllamaAgentMessage> = vec![
        OllamaAgentMessage::system(system_prompt),
        OllamaAgentMessage::user(&prompt),
    ];

    let url = match Url::parse(&format!("{base_url}/api/chat")) {
        Ok(u) => u,
        Err(err) => {
            update_local_run_status(
                &local_runs,
                &task_id,
                LocalAgentStatus::Failed {
                    error: format!("Invalid Ollama base URL: {err}"),
                },
            );
            return;
        }
    };

    const MAX_ITERATIONS: usize = 30;
    let mut iterations = 0;

    loop {
        iterations += 1;
        if iterations > MAX_ITERATIONS {
            update_local_run_status(&local_runs, &task_id, LocalAgentStatus::Failed {
                error: "Agent exceeded maximum iterations (30 tool calls). Please try a simpler task.".to_string(),
            });
            return;
        }

        // Check for cancellation
        if is_cancelled(&local_runs, &task_id) {
            return;
        }

        let request = OllamaAgentChatRequest {
            model: model.clone(),
            messages: messages.clone(),
            tools: tools.clone(),
            stream: false,
            options: OllamaOptions { num_predict: 4096 },
        };

        let response_result = http_client
            .post(url.clone())
            .timeout(Duration::from_secs(120))
            .json(&request)
            .send()
            .await;

        let response = match response_result {
            Ok(r) => r,
            Err(err) => {
                update_local_run_status(
                    &local_runs,
                    &task_id,
                    LocalAgentStatus::Failed {
                        error: format!("Failed to reach Ollama: {err}"),
                    },
                );
                return;
            }
        };

        let parsed: OllamaAgentChatResponse = match response.json().await {
            Ok(p) => p,
            Err(err) => {
                update_local_run_status(
                    &local_runs,
                    &task_id,
                    LocalAgentStatus::Failed {
                        error: format!("Failed to parse Ollama response: {err}"),
                    },
                );
                return;
            }
        };

        let assistant_msg = normalize_ollama_assistant_message(parsed.message.clone());

        if assistant_msg.tool_calls.is_empty() {
            // No tool calls — this is the final answer
            let final_output = assistant_msg.content.unwrap_or_default();
            update_local_run_status(
                &local_runs,
                &task_id,
                LocalAgentStatus::Done { final_output },
            );
            return;
        }

        // Execute each tool call
        messages.push(assistant_msg.clone());

        for tool_call in &assistant_msg.tool_calls {
            if is_cancelled(&local_runs, &task_id) {
                return;
            }

            let fn_name = &tool_call.function.name;
            let fn_args = &tool_call.function.arguments;

            // Update status to show current action
            let action_desc = match fn_name.as_str() {
                "run_command" => {
                    let cmd = fn_args
                        .get("command")
                        .and_then(|v| v.as_str())
                        .unwrap_or("...");
                    format!("Running: {}", cmd.chars().take(80).collect::<String>())
                }
                "read_file" => {
                    let path = fn_args
                        .get("path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("...");
                    format!("Reading: {path}")
                }
                "write_file" => {
                    let path = fn_args
                        .get("path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("...");
                    format!("Writing: {path}")
                }
                "list_files" => {
                    let path = fn_args
                        .get("path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("...");
                    format!("Listing: {path}")
                }
                "search_in_files" => {
                    let pattern = fn_args
                        .get("pattern")
                        .and_then(|v| v.as_str())
                        .unwrap_or("...");
                    format!("Searching for: {pattern}")
                }
                other => format!("Calling: {other}"),
            };
            update_local_run_status(
                &local_runs,
                &task_id,
                LocalAgentStatus::Running {
                    current_action: action_desc,
                },
            );

            let result =
                execute_tool_with_defaults(fn_name, fn_args, working_directory.as_deref()).await;
            messages.push(OllamaAgentMessage::tool_result(result));
        }
    }
}

fn update_local_run_status(
    local_runs: &LocalRuns,
    task_id: &AmbientAgentTaskId,
    status: LocalAgentStatus,
) {
    let guard = local_runs.lock();
    if let Some(run) = guard.get(task_id) {
        run.lock().status = status;
    }
}

fn is_cancelled(local_runs: &LocalRuns, task_id: &AmbientAgentTaskId) -> bool {
    let guard = local_runs.lock();
    guard
        .get(task_id)
        .map(|run| run.lock().cancel_flag.load(Ordering::Relaxed))
        .unwrap_or(false)
}

// ---- AIClient impl ----

#[async_trait]
impl AIClient for OllamaAIClient {
    async fn generate_commands_from_natural_language(
        &self,
        prompt: String,
        _ai_execution_context: Option<WarpAiExecutionContext>,
    ) -> std::result::Result<Vec<AIGeneratedCommand>, GenerateCommandsFromNaturalLanguageError>
    {
        let Some(config) = self.active_config() else {
            return self
                .server_api
                .generate_commands_from_natural_language(prompt, _ai_execution_context)
                .await;
        };
        let content = self
            .chat(&config, vec![
                OllamaChatMessage {
                    role: "system",
                    content: "Convert user requests into shell commands. Return only valid JSON as either an array or an object with a `commands` array. Each item must contain `command`, `description`, and `parameters` where `parameters` is an array of `{ \"id\": string, \"description\": string }`.".to_string(),
                },
                OllamaChatMessage {
                    role: "user",
                    content: prompt,
                },
            ])
            .await
            .map_err(|error| {
                log::warn!("Ollama command generation failed: {error:#}");
                GenerateCommandsFromNaturalLanguageError::AiProviderError
            })?;

        self.parse_generated_commands(&content).map_err(|error| {
            log::warn!("Failed to parse Ollama command response: {error:#}");
            GenerateCommandsFromNaturalLanguageError::Other
        })
    }

    async fn generate_dialogue_answer(
        &self,
        transcript: Vec<TranscriptPart>,
        prompt: String,
        _ai_execution_context: Option<WarpAiExecutionContext>,
    ) -> AnyhowResult<GenerateDialogueResult> {
        let Some(config) = self.active_config() else {
            return self
                .server_api
                .generate_dialogue_answer(transcript, prompt, _ai_execution_context)
                .await;
        };
        let mut messages = vec![OllamaChatMessage {
            role: "system",
            content: "You are Warp AI running locally through Ollama. Provide concise, helpful answers for terminal and developer questions. Prefer directly actionable guidance.".to_string(),
        }];

        for part in transcript {
            messages.push(OllamaChatMessage {
                role: "user",
                content: part.raw_user_prompt().to_string(),
            });
            messages.push(OllamaChatMessage {
                role: "assistant",
                content: part.raw_assistant_answer().to_string(),
            });
        }

        messages.push(OllamaChatMessage {
            role: "user",
            content: prompt,
        });

        let answer = self.chat(&config, messages).await?;

        Ok(GenerateDialogueResult::Success {
            answer,
            truncated: false,
            request_limit_info: self.unlimited_request_usage().request_limit_info,
            transcript_summarized: false,
        })
    }

    async fn generate_metadata_for_command(
        &self,
        command: String,
    ) -> std::result::Result<GeneratedCommandMetadata, GeneratedCommandMetadataError> {
        self.server_api.generate_metadata_for_command(command).await
    }

    async fn get_request_limit_info(&self) -> AnyhowResult<RequestUsageInfo> {
        match self.active_config() {
            Some(_) => Ok(self.unlimited_request_usage()),
            None => self.server_api.get_request_limit_info().await,
        }
    }

    async fn get_feature_model_choices(&self) -> AnyhowResult<ModelsByFeature> {
        match self.active_config() {
            Some(config) => self.model_choices(&config),
            None => self.server_api.get_feature_model_choices().await,
        }
    }

    async fn get_free_available_models(
        &self,
        _referrer: Option<String>,
    ) -> AnyhowResult<ModelsByFeature> {
        match self.active_config() {
            Some(config) => self.model_choices(&config),
            None => self.server_api.get_free_available_models(_referrer).await,
        }
    }

    async fn update_merkle_tree(
        &self,
        embedding_config: EmbeddingConfig,
        nodes: Vec<IntermediateNode>,
    ) -> AnyhowResult<HashMap<NodeHash, bool>> {
        self.server_api
            .update_merkle_tree(embedding_config, nodes)
            .await
    }

    async fn generate_code_embeddings(
        &self,
        embedding_config: EmbeddingConfig,
        fragments: Vec<full_source_code_embedding::Fragment>,
        root_hash: NodeHash,
        repo_metadata: RepoMetadata,
    ) -> AnyhowResult<HashMap<ContentHash, bool>> {
        self.server_api
            .generate_code_embeddings(embedding_config, fragments, root_hash, repo_metadata)
            .await
    }

    async fn provide_negative_feedback_response_for_ai_conversation(
        &self,
        conversation_id: String,
        request_ids: Vec<String>,
    ) -> AnyhowResult<i32> {
        self.server_api
            .provide_negative_feedback_response_for_ai_conversation(conversation_id, request_ids)
            .await
    }

    async fn create_agent_task(
        &self,
        prompt: String,
        environment_uid: Option<String>,
        parent_run_id: Option<String>,
        config: Option<crate::server::server_api::ai::AgentConfigSnapshot>,
    ) -> AnyhowResult<crate::ai::ambient_agents::AmbientAgentTaskId> {
        self.server_api
            .create_agent_task(prompt, environment_uid, parent_run_id, config)
            .await
    }

    async fn update_agent_task(
        &self,
        task_id: crate::ai::ambient_agents::AmbientAgentTaskId,
        task_state: Option<AgentTaskState>,
        session_id: Option<SessionId>,
        conversation_id: Option<String>,
        status_message: Option<TaskStatusUpdate>,
    ) -> AnyhowResult<()> {
        self.server_api
            .update_agent_task(
                task_id,
                task_state,
                session_id,
                conversation_id,
                status_message,
            )
            .await
    }

    async fn spawn_agent(&self, request: SpawnAgentRequest) -> AnyhowResult<SpawnAgentResponse> {
        let Some(config) = self.active_config() else {
            return self.server_api.spawn_agent(request).await;
        };

        let task_uuid = Uuid::new_v4();
        let run_uuid = Uuid::new_v4();
        let task_id: AmbientAgentTaskId = task_uuid
            .to_string()
            .parse()
            .context("Failed to create local task ID")?;
        let run_id = run_uuid.to_string();
        let cancel_flag = Arc::new(AtomicBool::new(false));

        let state = LocalAgentState {
            task_id,
            prompt: request.prompt.clone(),
            created_at: chrono::Utc::now(),
            status: LocalAgentStatus::Running {
                current_action: "Starting Ollama agent...".to_string(),
            },
            cancel_flag: cancel_flag.clone(),
        };

        self.local_runs
            .lock()
            .insert(task_id, Arc::new(parking_lot::Mutex::new(state)));

        let http_client = self.http_client.clone();
        let local_runs = self.local_runs.clone();
        let prompt = request.prompt.clone();
        let working_directory = request.working_directory.clone();

        tokio::spawn(async move {
            run_ollama_agent_loop(
                config,
                http_client,
                task_id,
                prompt,
                working_directory,
                local_runs,
            )
            .await;
        });

        log::info!("Spawned local Ollama agent: task_id={task_id}, run_id={run_id}");

        Ok(SpawnAgentResponse {
            task_id,
            run_id,
            at_capacity: false,
        })
    }

    async fn list_ambient_agent_tasks(
        &self,
        limit: i32,
        filter: TaskListFilter,
    ) -> AnyhowResult<Vec<crate::server::server_api::ai::AmbientAgentTask>> {
        self.server_api
            .list_ambient_agent_tasks(limit, filter)
            .await
    }

    async fn list_agent_runs_raw(
        &self,
        limit: i32,
        filter: TaskListFilter,
    ) -> AnyhowResult<serde_json::Value> {
        self.server_api.list_agent_runs_raw(limit, filter).await
    }

    async fn get_ambient_agent_task(
        &self,
        task_id: &crate::ai::ambient_agents::AmbientAgentTaskId,
    ) -> AnyhowResult<crate::server::server_api::ai::AmbientAgentTask> {
        // Check local runs first
        let local_state = self.local_runs.lock().get(task_id).cloned();
        if let Some(state_arc) = local_state {
            let state = state_arc.lock();
            return Ok(Self::make_local_ambient_task(&state));
        }
        self.server_api.get_ambient_agent_task(task_id).await
    }

    async fn get_agent_run_raw(
        &self,
        task_id: &crate::ai::ambient_agents::AmbientAgentTaskId,
    ) -> AnyhowResult<serde_json::Value> {
        self.server_api.get_agent_run_raw(task_id).await
    }

    async fn get_scheduled_agent_history(
        &self,
        schedule_id: &str,
    ) -> AnyhowResult<ScheduledAgentHistory> {
        self.server_api
            .get_scheduled_agent_history(schedule_id)
            .await
    }

    async fn get_ai_conversation(
        &self,
        server_conversation_token: ServerConversationToken,
    ) -> AnyhowResult<(ConversationData, ServerAIConversationMetadata)> {
        self.server_api
            .get_ai_conversation(server_conversation_token)
            .await
    }

    async fn list_ai_conversation_metadata(
        &self,
        conversation_ids: Option<Vec<String>>,
    ) -> AnyhowResult<Vec<ServerAIConversationMetadata>> {
        self.server_api
            .list_ai_conversation_metadata(conversation_ids)
            .await
    }

    async fn get_ai_conversation_format(
        &self,
        server_conversation_token: ServerConversationToken,
    ) -> AnyhowResult<AIAgentConversationFormat> {
        self.server_api
            .get_ai_conversation_format(server_conversation_token)
            .await
    }

    async fn get_block_snapshot(
        &self,
        server_conversation_token: ServerConversationToken,
    ) -> AnyhowResult<SerializedBlock> {
        self.server_api
            .get_block_snapshot(server_conversation_token)
            .await
    }

    async fn delete_ai_conversation(&self, server_conversation_token: String) -> AnyhowResult<()> {
        self.server_api
            .delete_ai_conversation(server_conversation_token)
            .await
    }

    async fn list_agents(
        &self,
        repo: Option<String>,
    ) -> AnyhowResult<Vec<crate::server::server_api::ai::AgentListItem>> {
        self.server_api.list_agents(repo).await
    }

    async fn cancel_ambient_agent_task(
        &self,
        task_id: &crate::ai::ambient_agents::AmbientAgentTaskId,
    ) -> AnyhowResult<()> {
        let local_state = self.local_runs.lock().get(task_id).cloned();
        if let Some(state_arc) = local_state {
            let state = state_arc.lock();
            state.cancel_flag.store(true, Ordering::Relaxed);
            drop(state);
            update_local_run_status(&self.local_runs, task_id, LocalAgentStatus::Cancelled);
            return Ok(());
        }
        self.server_api.cancel_ambient_agent_task(task_id).await
    }

    async fn get_task_attachments(&self, task_id: String) -> AnyhowResult<Vec<TaskAttachment>> {
        self.server_api.get_task_attachments(task_id).await
    }

    async fn create_file_artifact_upload_target(
        &self,
        request: CreateFileArtifactUploadRequest,
    ) -> AnyhowResult<CreateFileArtifactUploadResponse> {
        self.server_api
            .create_file_artifact_upload_target(request)
            .await
    }

    async fn confirm_file_artifact_upload(
        &self,
        artifact_uid: String,
        checksum: String,
    ) -> AnyhowResult<FileArtifactRecord> {
        self.server_api
            .confirm_file_artifact_upload(artifact_uid, checksum)
            .await
    }

    async fn get_artifact_download(
        &self,
        artifact_uid: &str,
    ) -> AnyhowResult<ArtifactDownloadResponse> {
        self.server_api.get_artifact_download(artifact_uid).await
    }

    async fn prepare_attachments_for_upload(
        &self,
        task_id: &crate::ai::ambient_agents::AmbientAgentTaskId,
        files: &[AttachmentFileInfo],
    ) -> AnyhowResult<PrepareAttachmentUploadsResponse> {
        self.server_api
            .prepare_attachments_for_upload(task_id, files)
            .await
    }

    async fn download_task_attachments(
        &self,
        task_id: &crate::ai::ambient_agents::AmbientAgentTaskId,
        attachment_ids: &[String],
    ) -> AnyhowResult<crate::server::server_api::ai::DownloadAttachmentsResponse> {
        self.server_api
            .download_task_attachments(task_id, attachment_ids)
            .await
    }

    async fn get_handoff_snapshot_attachments(
        &self,
        task_id: &crate::ai::ambient_agents::AmbientAgentTaskId,
    ) -> AnyhowResult<Vec<TaskAttachment>> {
        self.server_api
            .get_handoff_snapshot_attachments(task_id)
            .await
    }

    async fn send_agent_message(
        &self,
        request: SendAgentMessageRequest,
    ) -> AnyhowResult<SendAgentMessageResponse> {
        self.server_api.send_agent_message(request).await
    }

    async fn list_agent_messages(
        &self,
        run_id: &str,
        request: ListAgentMessagesRequest,
    ) -> AnyhowResult<Vec<AgentMessageHeader>> {
        self.server_api.list_agent_messages(run_id, request).await
    }

    async fn poll_agent_events(
        &self,
        run_ids: &[String],
        since_sequence: i64,
        limit: i32,
    ) -> AnyhowResult<Vec<AgentRunEvent>> {
        self.server_api
            .poll_agent_events(run_ids, since_sequence, limit)
            .await
    }

    async fn report_agent_event(
        &self,
        run_id: &str,
        request: ReportAgentEventRequest,
    ) -> AnyhowResult<ReportAgentEventResponse> {
        self.server_api.report_agent_event(run_id, request).await
    }

    async fn mark_message_delivered(&self, message_id: &str) -> AnyhowResult<()> {
        self.server_api.mark_message_delivered(message_id).await
    }

    async fn read_agent_message(&self, message_id: &str) -> AnyhowResult<ReadAgentMessageResponse> {
        self.server_api.read_agent_message(message_id).await
    }

    async fn get_public_conversation(
        &self,
        conversation_id: &str,
    ) -> AnyhowResult<serde_json::Value> {
        self.server_api
            .get_public_conversation(conversation_id)
            .await
    }

    async fn get_run_conversation(&self, run_id: &str) -> AnyhowResult<serde_json::Value> {
        self.server_api.get_run_conversation(run_id).await
    }

    async fn generate_code_review_content(
        &self,
        request: GenerateCodeReviewContentRequest,
    ) -> AnyhowResult<GenerateCodeReviewContentResponse> {
        self.server_api.generate_code_review_content(request).await
    }
}
