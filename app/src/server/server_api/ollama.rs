use std::{
    collections::HashMap,
    path::Path,
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
use chrono::{Local, Utc};
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
    util::git,
};
use ai::index::full_source_code_embedding::{
    self, ContentHash, EmbeddingConfig, NodeHash, RepoMetadata, store_client::IntermediateNode,
};
use session_sharing_protocol::common::SessionId;
use uuid::Uuid;
use warp_graphql::ai::AgentTaskState;
use warp_graphql::queries::get_scheduled_agent_history::ScheduledAgentHistory;
use warp_multi_agent_api::ConversationData;

mod parsing;
mod prompts;
mod tools;

use self::{
    parsing::{looks_like_tool_attempt, normalize_assistant_message, parse_tool_calls_from_content},
    prompts::{
        build_local_agent_system_prompt, build_local_dialogue_system_prompt,
        build_multi_agent_system_prompt,
        build_tool_repair_messages,
    },
    tools::{agent_tools, execute_tool, is_supported_tool_name},
};

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

fn message_content<'a>(message: &'a OllamaAgentMessage) -> &'a str {
    message.content.as_deref().unwrap_or_default()
}

fn combined_user_text(messages: &[OllamaAgentMessage]) -> String {
    messages
        .iter()
        .filter(|message| message.role == "user")
        .map(message_content)
        .collect::<Vec<_>>()
        .join("\n")
        .to_lowercase()
}

fn runtime_context_has_current_time(messages: &[OllamaAgentMessage]) -> bool {
    messages.iter().any(|message| {
        message.role == "system" && message_content(message).contains("Current local time:")
    })
}

fn is_time_or_date_request(user_text: &str) -> bool {
    [
        "what time",
        "current time",
        "time is it",
        "what's the time",
        "date is it",
        "current date",
        "today's date",
        "today date",
    ]
    .iter()
    .any(|pattern| user_text.contains(pattern))
}

fn is_git_related_request(user_text: &str) -> bool {
    [
        "git",
        "commit",
        "push",
        "stage",
        "staging",
        "branch",
        "diff",
        "status",
        "repository",
        "repo",
    ]
    .iter()
    .any(|pattern| user_text.contains(pattern))
}

fn contains_shell_command_chaining(command: &str) -> bool {
    command.contains("&&")
        || command.contains("||")
        || command.contains(';')
        || command.contains('\n')
}

fn is_git_inspection_command(command: &str) -> bool {
    let command = command.trim().to_lowercase();
    command.starts_with("git status")
        || command.starts_with("git diff")
        || command.starts_with("git log")
        || command.starts_with("git show")
}

fn is_git_mutation_command(command: &str) -> bool {
    let command = command.trim().to_lowercase();
    command.starts_with("git add")
        || command.starts_with("git commit")
        || command.starts_with("git push")
        || command.starts_with("git rm")
        || command.starts_with("git mv")
        || command.starts_with("git reset")
        || command.starts_with("git checkout")
        || command.starts_with("git cherry-pick")
        || command.starts_with("git rebase")
        || command.starts_with("git merge")
}

fn has_prior_git_inspection(messages: &[OllamaAgentMessage]) -> bool {
    has_prior_matching_command(messages, is_git_inspection_command)
}

fn has_prior_matching_command(
    messages: &[OllamaAgentMessage],
    predicate: impl Fn(&str) -> bool,
) -> bool {
    messages.iter().any(|message| {
        message.tool_calls.iter().any(|call| {
            call.function.name == "run_command"
                && call
                    .function
                    .arguments
                    .get("command")
                    .and_then(|value| value.as_str())
                    .is_some_and(&predicate)
        })
    })
}

fn validate_tool_call(
    messages: &[OllamaAgentMessage],
    tool_name: &str,
    arguments: &serde_json::Value,
) -> Option<String> {
    let user_text = combined_user_text(messages);
    let wants_stage = user_text.contains("stage");
    let wants_commit = user_text.contains("commit");
    let wants_push = user_text.contains("push");

    if runtime_context_has_current_time(messages)
        && is_time_or_date_request(&user_text)
        && !user_text.contains("verify")
        && !user_text.contains("double-check")
    {
        return Some(
            "Policy: Warp runtime context already includes the current local time/date. Do not \
call any tool. Respond in plain language using the `Current local time:` value from runtime \
context."
                .to_string(),
        );
    }

    if is_git_related_request(&user_text) && !has_prior_git_inspection(messages) {
        if tool_name != "run_command" {
            return Some(
                "Policy: For git workflows, inspect repository state first. Your next tool call \
should be `run_command` with `git status --short --branch`."
                    .to_string(),
            );
        }

        let command = arguments.get("command").and_then(|value| value.as_str())?;
        if contains_shell_command_chaining(command) {
            return Some(
                "Policy: Use one shell command per tool call. Do not chain git commands with \
`&&`, `||`, `;`, or newlines. First call `run_command` with `git status --short --branch`."
                    .to_string(),
            );
        }

        if is_git_mutation_command(command) && !is_git_inspection_command(command) {
            return Some(
                "Policy: Inspect repository state first before staging, committing, or pushing. \
Your next tool call should be `run_command` with `git status --short --branch`."
                    .to_string(),
            );
        }

        if wants_stage && command.trim().to_lowercase().starts_with("git diff --cached") {
            return Some(
                "Policy: Stage the intended files first. Your next tool call should be \
`run_command` with a `git add ...` command for the intended files."
                    .to_string(),
            );
        }

        if wants_commit && command.trim().to_lowercase().starts_with("git commit") {
            return Some(
                "Policy: Before committing, stage the intended files and inspect the staged diff \
summary. If files are not staged yet, your next tool call should be `run_command` with a \
`git add ...` command."
                    .to_string(),
            );
        }

        if wants_push && command.trim().to_lowercase().starts_with("git push") {
            return Some(
                "Policy: Inspect, stage, and commit the changes before pushing. Your next tool \
call should be `run_command` with `git status --short --branch`."
                    .to_string(),
            );
        }
    }

    if tool_name == "run_command" {
        let command = arguments.get("command").and_then(|value| value.as_str())?;
        let command_lower = command.trim().to_lowercase();
        if contains_shell_command_chaining(command) {
            return Some(
                "Policy: Use one shell command per tool call. Do not combine multiple shell \
commands with `&&`, `||`, `;`, or newlines."
                    .to_string(),
            );
        }

        let has_prior_git_add =
            has_prior_matching_command(messages, |cmd| cmd.trim().to_lowercase().starts_with("git add"));
        let has_prior_staged_diff = has_prior_matching_command(messages, |cmd| {
            cmd.trim().to_lowercase().starts_with("git diff --cached")
        });
        let has_prior_git_commit = has_prior_matching_command(messages, |cmd| {
            cmd.trim().to_lowercase().starts_with("git commit")
        });

        if wants_stage && command_lower.starts_with("git diff --cached") && !has_prior_git_add {
            return Some(
                "Policy: Stage the intended files first. Your next tool call should be \
`run_command` with a `git add ...` command for the intended files."
                    .to_string(),
            );
        }

        if wants_commit && command_lower.starts_with("git commit")
            && (!has_prior_git_add || !has_prior_staged_diff)
        {
            return Some(
                "Policy: Before committing, stage the intended files and inspect the staged diff \
summary with `git diff --cached --stat`."
                    .to_string(),
            );
        }

        if wants_push && command_lower.starts_with("git push") && !has_prior_git_commit {
            return Some(
                "Policy: Create the commit before pushing the branch. Your next tool call should \
be `run_command` with `git commit -m ...` after the staged diff has been inspected."
                    .to_string(),
            );
        }
    }

    None
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

fn all_tool_calls_supported(tool_calls: &[OllamaToolCall]) -> bool {
    tool_calls
        .iter()
        .all(|tool_call| is_supported_tool_name(&tool_call.function.name))
}

async fn repair_tool_calls_from_content(
    http_client: &reqwest::Client,
    config: &OllamaConfig,
    raw_content: &str,
) -> AnyhowResult<Vec<OllamaToolCall>> {
    let repair_output =
        chat_with_config(http_client, config, build_tool_repair_messages(raw_content)).await?;
    Ok(parse_tool_calls_from_content(&repair_output)
        .into_iter()
        .filter(|tool_call| is_supported_tool_name(&tool_call.function.name))
        .collect())
}

async fn maybe_repair_assistant_message(
    http_client: &reqwest::Client,
    config: &OllamaConfig,
    message: OllamaAgentMessage,
) -> AnyhowResult<OllamaAgentMessage> {
    let normalized = normalize_assistant_message(message);
    if !normalized.tool_calls.is_empty() && all_tool_calls_supported(&normalized.tool_calls) {
        return Ok(normalized);
    }

    let raw_content = normalized
        .content
        .clone()
        .filter(|content| looks_like_tool_attempt(content))
        .or_else(|| {
            (!normalized.tool_calls.is_empty())
                .then(|| serde_json::to_string(&normalized.tool_calls).ok())
                .flatten()
        });

    let Some(raw_content) = raw_content else {
        return Ok(normalized);
    };

    let repaired_tool_calls = repair_tool_calls_from_content(http_client, config, &raw_content).await?;
    if repaired_tool_calls.is_empty() {
        return Ok(normalized);
    }

    Ok(OllamaAgentMessage {
        tool_calls: repaired_tool_calls,
        content: None,
        ..normalized
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
    let mut policy_history = messages.clone();
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

        let assistant_msg =
            maybe_repair_assistant_message(http_client, config, parsed.message).await?;
        if assistant_msg.tool_calls.is_empty() {
            return Ok(assistant_msg.content.unwrap_or_default());
        }

        messages.push(assistant_msg.clone());
        for tool_call in &assistant_msg.tool_calls {
            let result = if let Some(policy_error) = validate_tool_call(
                &policy_history,
                &tool_call.function.name,
                &tool_call.function.arguments,
            ) {
                policy_error
            } else {
                let result = execute_tool(
                    &tool_call.function.name,
                    &tool_call.function.arguments,
                    None,
                )
                .await;
                policy_history.push(OllamaAgentMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: vec![tool_call.clone()],
                });
                policy_history.push(OllamaAgentMessage::tool_result(result.clone()));
                result
            };
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
    let mut messages = vec![OllamaAgentMessage::system(build_multi_agent_system_prompt())];
    messages.push(OllamaAgentMessage::system(format_runtime_context(
        Local::now(),
        None,
        None,
        None,
    )));

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

#[derive(Debug, Clone, Default)]
struct GitRuntimeContext {
    repo_root: Option<String>,
    branch: Option<String>,
    head: Option<String>,
    status_summary: Option<String>,
    staged_diff_stat: Option<String>,
    unstaged_diff_stat: Option<String>,
}

async fn collect_git_runtime_context(working_directory: Option<&str>) -> Option<GitRuntimeContext> {
    let working_directory = working_directory?.trim();
    if working_directory.is_empty() {
        return None;
    }

    let repo_path = Path::new(working_directory);
    let repo_root = git::run_git_command(repo_path, &["rev-parse", "--show-toplevel"])
        .await
        .ok()
        .map(|output| output.trim().to_string())
        .filter(|output| !output.is_empty())?;

    let branch = git::detect_current_branch_display(repo_path)
        .await
        .ok()
        .filter(|output| !output.is_empty());
    let head = git::run_git_command(repo_path, &["rev-parse", "--short", "HEAD"])
        .await
        .ok()
        .map(|output| output.trim().to_string())
        .filter(|output| !output.is_empty());
    let status_summary = git::run_git_command(repo_path, &["status", "--short", "--branch"])
        .await
        .ok()
        .map(|output| output.trim().to_string())
        .filter(|output| !output.is_empty())
        .map(|output| truncate_owned(output, 4000));
    let staged_diff_stat = git::run_git_command(repo_path, &["diff", "--cached", "--stat"])
        .await
        .ok()
        .map(|output| output.trim().to_string())
        .filter(|output| !output.is_empty())
        .map(|output| truncate_owned(output, 2000));
    let unstaged_diff_stat = git::run_git_command(repo_path, &["diff", "--stat"])
        .await
        .ok()
        .map(|output| output.trim().to_string())
        .filter(|output| !output.is_empty())
        .map(|output| truncate_owned(output, 2000));

    Some(GitRuntimeContext {
        repo_root: Some(repo_root),
        branch,
        head,
        status_summary,
        staged_diff_stat,
        unstaged_diff_stat,
    })
}

fn format_runtime_context(
    current_time: chrono::DateTime<Local>,
    working_directory: Option<&str>,
    execution_context: Option<&WarpAiExecutionContext>,
    git_context: Option<&GitRuntimeContext>,
) -> String {
    let mut lines = vec![
        "Warp runtime context (authoritative, already known facts; do not re-check these facts with tools unless the user explicitly asks you to verify them):".to_string(),
        format!("Current local time: {}", current_time.to_rfc3339()),
    ];

    if let Some(working_directory) = working_directory.filter(|cwd| !cwd.trim().is_empty()) {
        lines.push(format!("Current working directory: {working_directory}"));
    }

    if let Some(execution_context) = execution_context {
        if let Some(os_category) = execution_context.os.category.as_deref() {
            lines.push(format!("Operating system: {os_category}"));
        }

        if let Some(distribution) = execution_context.os.distribution.as_deref() {
            lines.push(format!("OS distribution: {distribution}"));
        }

        if !execution_context.shell_name.trim().is_empty() {
            lines.push(format!("Shell: {}", execution_context.shell_name));
        }

        if let Some(shell_version) = execution_context
            .shell_version
            .as_deref()
            .filter(|version| !version.trim().is_empty())
        {
            lines.push(format!("Shell version: {shell_version}"));
        }
    }

    if let Some(git_context) = git_context {
        lines.push("Git repository context:".to_string());

        if let Some(repo_root) = git_context.repo_root.as_deref() {
            lines.push(format!("Git root: {repo_root}"));
        }
        if let Some(branch) = git_context.branch.as_deref() {
            lines.push(format!("Git branch: {branch}"));
        }
        if let Some(head) = git_context.head.as_deref() {
            lines.push(format!("Git HEAD: {head}"));
        }
        if let Some(status_summary) = git_context.status_summary.as_deref() {
            lines.push("Git status summary:".to_string());
            lines.push(status_summary.to_string());
        }
        if let Some(staged_diff_stat) = git_context.staged_diff_stat.as_deref() {
            lines.push("Staged diff summary:".to_string());
            lines.push(staged_diff_stat.to_string());
        }
        if let Some(unstaged_diff_stat) = git_context.unstaged_diff_stat.as_deref() {
            lines.push("Unstaged diff summary:".to_string());
            lines.push(unstaged_diff_stat.to_string());
        }
    }

    lines.join("\n")
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

fn truncate_owned(s: String, max_bytes: usize) -> String {
    truncate_str(&s, max_bytes).to_string()
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
    let system_prompt = build_local_agent_system_prompt(working_directory.as_deref());
    let git_context = collect_git_runtime_context(working_directory.as_deref()).await;
    let runtime_context = format_runtime_context(
        Local::now(),
        working_directory.as_deref(),
        None,
        git_context.as_ref(),
    );

    let mut messages: Vec<OllamaAgentMessage> = vec![
        OllamaAgentMessage::system(system_prompt),
        OllamaAgentMessage::system(runtime_context),
        OllamaAgentMessage::user(&prompt),
    ];
    let mut policy_history = messages.clone();

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

        let parsed: OllamaAgentChatResponse = match response.error_for_status() {
            Ok(response) => match response.json().await {
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
            },
            Err(err) => {
                update_local_run_status(
                    &local_runs,
                    &task_id,
                    LocalAgentStatus::Failed {
                        error: format!("Ollama returned an error response: {err}"),
                    },
                );
                return;
            }
        };

        let assistant_msg = match maybe_repair_assistant_message(&http_client, &config, parsed.message)
            .await
        {
            Ok(p) => p,
            Err(err) => {
                update_local_run_status(
                    &local_runs,
                    &task_id,
                    LocalAgentStatus::Failed {
                        error: format!("Failed to normalize Ollama tool response: {err}"),
                    },
                );
                return;
            }
        };

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

            let result = if let Some(policy_error) =
                validate_tool_call(&policy_history, fn_name, fn_args)
            {
                policy_error
            } else {
                let result = execute_tool(fn_name, fn_args, working_directory.as_deref()).await;
                policy_history.push(OllamaAgentMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: vec![tool_call.clone()],
                });
                policy_history.push(OllamaAgentMessage::tool_result(result.clone()));
                result
            };
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
            content: build_local_dialogue_system_prompt(),
        }];
        messages.push(OllamaChatMessage {
            role: "system",
            content: format_runtime_context(
                Local::now(),
                None,
                _ai_execution_context.as_ref(),
                None,
            ),
        });

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

#[cfg(test)]
mod tests {
    use super::format_runtime_context;
    use crate::ai_assistant::execution_context::{WarpAiExecutionContext, WarpAiOsContext};
    use chrono::{FixedOffset, TimeZone};

    #[test]
    fn format_runtime_context_includes_known_runtime_facts() {
        let current_time = FixedOffset::west_opt(4 * 3600)
            .unwrap()
            .with_ymd_and_hms(2026, 4, 30, 8, 37, 6)
            .single()
            .unwrap()
            .with_timezone(&chrono::Local);
        let execution_context = WarpAiExecutionContext {
            os: WarpAiOsContext {
                category: Some("windows".to_string()),
                distribution: Some("Windows 11".to_string()),
            },
            shell_name: "powershell".to_string(),
            shell_version: Some("7.5".to_string()),
        };

        let formatted = format_runtime_context(
            current_time,
            Some("C:\\Repos\\warp"),
            Some(&execution_context),
            Some(&GitRuntimeContext {
                repo_root: Some("C:\\Repos\\warp".to_string()),
                branch: Some("codex/git-awareness".to_string()),
                head: Some("abc1234".to_string()),
                status_summary: Some("## codex/git-awareness\n M app/src/server/server_api/ollama.rs".to_string()),
                staged_diff_stat: Some(" app/src/server/server_api/ollama.rs | 42 ++++++++++++++++++++++".to_string()),
                unstaged_diff_stat: Some(" app/src/server/server_api/ollama/prompts.rs | 6 +++".to_string()),
            }),
        );

        assert!(formatted.contains("Warp runtime context (authoritative, already known facts; do not re-check these facts with tools unless the user explicitly asks you to verify them):"));
        assert!(formatted.contains("Current local time: 2026-04-30T"));
        assert!(formatted.contains("Current working directory: C:\\Repos\\warp"));
        assert!(formatted.contains("Operating system: windows"));
        assert!(formatted.contains("OS distribution: Windows 11"));
        assert!(formatted.contains("Shell: powershell"));
        assert!(formatted.contains("Shell version: 7.5"));
        assert!(formatted.contains("Git repository context:"));
        assert!(formatted.contains("Git root: C:\\Repos\\warp"));
        assert!(formatted.contains("Git branch: codex/git-awareness"));
        assert!(formatted.contains("Git HEAD: abc1234"));
        assert!(formatted.contains("Git status summary:"));
        assert!(formatted.contains("Staged diff summary:"));
        assert!(formatted.contains("Unstaged diff summary:"));
    }
}
