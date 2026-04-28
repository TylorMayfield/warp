use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use anyhow::{Context, Result as AnyhowResult};
use async_trait::async_trait;
use chrono::Utc;
use reqwest::Url;
use serde::{Deserialize, Serialize};
use warp_graphql::scalars::time::ServerTimestamp;

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
        execution_context::WarpAiExecutionContext, requests::GenerateDialogueResult,
        utils::TranscriptPart, AIGeneratedCommand, AIGeneratedCommandParameter,
        GenerateCommandsFromNaturalLanguageError,
    },
    drive::workflows::ai_assist::{GeneratedCommandMetadata, GeneratedCommandMetadataError},
    server::server_api::ai::{
        AIClient, AgentMessageHeader, AgentRunEvent,
        ArtifactDownloadResponse, AttachmentFileInfo, CreateFileArtifactUploadRequest,
        CreateFileArtifactUploadResponse, FileArtifactRecord, ListAgentMessagesRequest,
        PrepareAttachmentUploadsResponse, ReadAgentMessageResponse, ReportAgentEventRequest,
        ReportAgentEventResponse, SendAgentMessageRequest, SendAgentMessageResponse,
        SpawnAgentRequest, SpawnAgentResponse, TaskAttachment, TaskListFilter, TaskStatusUpdate,
    },
    settings,
    terminal::model::block::SerializedBlock,
};
use ai::index::full_source_code_embedding::{
    self,
    store_client::IntermediateNode,
    ContentHash, EmbeddingConfig, NodeHash, RepoMetadata,
};
use session_sharing_protocol::common::SessionId;
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
    run_id: String,
    prompt: String,
    created_at: chrono::DateTime<chrono::Utc>,
    status: LocalAgentStatus,
    cancel_flag: Arc<AtomicBool>,
}

type LocalRunsInner = HashMap<AmbientAgentTaskId, Arc<parking_lot::Mutex<LocalAgentState>>>;
type LocalRuns = Arc<parking_lot::Mutex<LocalRunsInner>>;

// ---- OllamaAIClient ----

pub(crate) struct OllamaAIClient {
    server_api: Arc<super::ServerApi>,
    http_client: reqwest::Client,
    local_runs: LocalRuns,
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
        })
    }

    fn active_config(&self) -> Option<OllamaConfig> {
        OllamaConfig::resolve()
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

        let available = AvailableLLMs::new(format!("ollama:{}", config.model).into(), [local_info()], None)?;

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

    async fn chat(&self, config: &OllamaConfig, messages: Vec<OllamaChatMessage>) -> AnyhowResult<String> {
        let url = Url::parse(&format!("{}/api/chat", config.base_url))
            .context("Invalid Ollama base URL")?;

        let response = self
            .http_client
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

    fn strip_code_fences(text: &str) -> &str {
        let trimmed = text.trim();
        if let Some(stripped) = trimmed.strip_prefix("```") {
            let without_lang = stripped
                .split_once('\n')
                .map(|(_, rest)| rest)
                .unwrap_or(stripped);
            return without_lang.strip_suffix("```").unwrap_or(without_lang).trim();
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
                            AIGeneratedCommandParameter::new(
                                parameter.id,
                                parameter.description,
                            )
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
                Some(TaskStatusMessage { message: current_action.clone() }),
                true,
            ),
            LocalAgentStatus::Done { final_output } => (
                AmbientAgentTaskState::Error,
                Some(TaskStatusMessage { message: format!("Ollama agent completed.\n\n{}", final_output) }),
                false,
            ),
            LocalAgentStatus::Failed { error } => (
                AmbientAgentTaskState::Error,
                Some(TaskStatusMessage { message: format!("Ollama agent error: {}", error) }),
                false,
            ),
            LocalAgentStatus::Cancelled => (
                AmbientAgentTaskState::Cancelled,
                None,
                false,
            ),
        };

        AmbientAgentTask {
            task_id: state.task_id,
            parent_run_id: None,
            title: format!("Ollama: {}", state.prompt.chars().take(60).collect::<String>()),
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
        Self { role: "user".to_string(), content: Some(content.into()), tool_calls: vec![] }
    }
    fn system(content: impl Into<String>) -> Self {
        Self { role: "system".to_string(), content: Some(content.into()), tool_calls: vec![] }
    }
    fn tool_result(content: impl Into<String>) -> Self {
        Self { role: "tool".to_string(), content: Some(content.into()), tool_calls: vec![] }
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
    match name {
        "run_command" => {
            let command = match args.get("command").and_then(|v| v.as_str()) {
                Some(c) => c.to_string(),
                None => return "Error: 'command' argument is required".to_string(),
            };
            let working_dir = args.get("working_dir").and_then(|v| v.as_str()).map(str::to_string);

            #[cfg(target_os = "windows")]
            let mut cmd = {
                let mut c = tokio::process::Command::new("cmd");
                c.args(["/C", &command]);
                c
            };
            #[cfg(not(target_os = "windows"))]
            let mut cmd = {
                let mut c = tokio::process::Command::new("sh");
                c.args(["-c", &command]);
                c
            };

            if let Some(dir) = working_dir {
                cmd.current_dir(dir);
            }

            match tokio::time::timeout(
                Duration::from_secs(60),
                cmd.output(),
            ).await {
                Ok(Ok(output)) => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let status = output.status.code().unwrap_or(-1);
                    let mut result = format!("Exit code: {status}");
                    if !stdout.is_empty() {
                        result.push_str(&format!("\nSTDOUT:\n{}", truncate_output(&stdout, 8000)));
                    }
                    if !stderr.is_empty() {
                        result.push_str(&format!("\nSTDERR:\n{}", truncate_output(&stderr, 4000)));
                    }
                    result
                }
                Ok(Err(err)) => format!("Error running command: {err}"),
                Err(_) => "Error: command timed out after 60 seconds".to_string(),
            }
        }

        "read_file" => {
            let path = match args.get("path").and_then(|v| v.as_str()) {
                Some(p) => p,
                None => return "Error: 'path' argument is required".to_string(),
            };
            match tokio::fs::read_to_string(path).await {
                Ok(content) => truncate_output(&content, 16000).to_string(),
                Err(err) => format!("Error reading file '{path}': {err}"),
            }
        }

        "write_file" => {
            let path = match args.get("path").and_then(|v| v.as_str()) {
                Some(p) => p,
                None => return "Error: 'path' argument is required".to_string(),
            };
            let content = match args.get("content").and_then(|v| v.as_str()) {
                Some(c) => c,
                None => return "Error: 'content' argument is required".to_string(),
            };
            if let Some(parent) = std::path::Path::new(path).parent() {
                let _ = tokio::fs::create_dir_all(parent).await;
            }
            match tokio::fs::write(path, content).await {
                Ok(()) => format!("Successfully wrote {} bytes to '{path}'", content.len()),
                Err(err) => format!("Error writing file '{path}': {err}"),
            }
        }

        "list_files" => {
            let path = match args.get("path").and_then(|v| v.as_str()) {
                Some(p) => p,
                None => return "Error: 'path' argument is required".to_string(),
            };
            let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("*");

            #[cfg(target_os = "windows")]
            let command = format!("dir /B /S \"{}\"", path);
            #[cfg(not(target_os = "windows"))]
            let command = format!("find {} -name '{}' 2>/dev/null | head -200", path, pattern);

            #[cfg(target_os = "windows")]
            let mut cmd = tokio::process::Command::new("cmd");
            #[cfg(target_os = "windows")]
            cmd.args(["/C", &command]);
            #[cfg(not(target_os = "windows"))]
            let mut cmd = tokio::process::Command::new("sh");
            #[cfg(not(target_os = "windows"))]
            cmd.args(["-c", &command]);

            match tokio::time::timeout(Duration::from_secs(15), cmd.output()).await {
                Ok(Ok(output)) => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    if stdout.is_empty() { "No files found".to_string() }
                    else { truncate_output(&stdout, 8000).to_string() }
                }
                Ok(Err(err)) => format!("Error listing files: {err}"),
                Err(_) => "Error: listing timed out".to_string(),
            }
        }

        "search_in_files" => {
            let pattern = match args.get("pattern").and_then(|v| v.as_str()) {
                Some(p) => p,
                None => return "Error: 'pattern' argument is required".to_string(),
            };
            let path = match args.get("path").and_then(|v| v.as_str()) {
                Some(p) => p,
                None => return "Error: 'path' argument is required".to_string(),
            };
            let case_flag = if args.get("case_sensitive").and_then(|v| v.as_bool()).unwrap_or(true) {
                ""
            } else {
                " -i"
            };

            #[cfg(target_os = "windows")]
            let command = format!("findstr /R{} /S \"{}\" \"{}\"", if case_flag.is_empty() { "" } else { "/I" }, pattern, path);
            #[cfg(not(target_os = "windows"))]
            let command = format!("grep -rn{} '{}' '{}' 2>/dev/null | head -100", case_flag, pattern, path);

            #[cfg(target_os = "windows")]
            let mut cmd = tokio::process::Command::new("cmd");
            #[cfg(target_os = "windows")]
            cmd.args(["/C", &command]);
            #[cfg(not(target_os = "windows"))]
            let mut cmd = tokio::process::Command::new("sh");
            #[cfg(not(target_os = "windows"))]
            cmd.args(["-c", &command]);

            match tokio::time::timeout(Duration::from_secs(30), cmd.output()).await {
                Ok(Ok(output)) => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    if stdout.is_empty() { format!("No matches found for '{pattern}' in '{path}'") }
                    else { truncate_output(&stdout, 8000).to_string() }
                }
                Ok(Err(err)) => format!("Error searching: {err}"),
                Err(_) => "Error: search timed out".to_string(),
            }
        }

        other => format!("Error: unknown tool '{other}'"),
    }
}

fn truncate_output<'a>(s: &'a str, max_bytes: usize) -> &'a str {
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
    local_runs: LocalRuns,
) {
    let tools = agent_tools();
    let base_url = config.base_url.clone();
    let model = config.model.clone();

    let system_prompt = "\
You are a helpful AI assistant with access to tools. \
Use the available tools to complete the user's task. \
After completing the task, provide a clear summary of what you did and the results. \
If you encounter errors, explain what went wrong and what you tried.";

    let mut messages: Vec<OllamaAgentMessage> = vec![
        OllamaAgentMessage::system(system_prompt),
        OllamaAgentMessage::user(&prompt),
    ];

    let url = match Url::parse(&format!("{base_url}/api/chat")) {
        Ok(u) => u,
        Err(err) => {
            update_local_run_status(&local_runs, &task_id, LocalAgentStatus::Failed {
                error: format!("Invalid Ollama base URL: {err}"),
            });
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
                update_local_run_status(&local_runs, &task_id, LocalAgentStatus::Failed {
                    error: format!("Failed to reach Ollama: {err}"),
                });
                return;
            }
        };

        let parsed: OllamaAgentChatResponse = match response.json().await {
            Ok(p) => p,
            Err(err) => {
                update_local_run_status(&local_runs, &task_id, LocalAgentStatus::Failed {
                    error: format!("Failed to parse Ollama response: {err}"),
                });
                return;
            }
        };

        let assistant_msg = parsed.message.clone();

        if assistant_msg.tool_calls.is_empty() {
            // No tool calls — this is the final answer
            let final_output = assistant_msg.content.unwrap_or_default();
            update_local_run_status(&local_runs, &task_id, LocalAgentStatus::Done { final_output });
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
                    let cmd = fn_args.get("command").and_then(|v| v.as_str()).unwrap_or("...");
                    format!("Running: {}", cmd.chars().take(80).collect::<String>())
                }
                "read_file" => {
                    let path = fn_args.get("path").and_then(|v| v.as_str()).unwrap_or("...");
                    format!("Reading: {path}")
                }
                "write_file" => {
                    let path = fn_args.get("path").and_then(|v| v.as_str()).unwrap_or("...");
                    format!("Writing: {path}")
                }
                "list_files" => {
                    let path = fn_args.get("path").and_then(|v| v.as_str()).unwrap_or("...");
                    format!("Listing: {path}")
                }
                "search_in_files" => {
                    let pattern = fn_args.get("pattern").and_then(|v| v.as_str()).unwrap_or("...");
                    format!("Searching for: {pattern}")
                }
                other => format!("Calling: {other}"),
            };
            update_local_run_status(&local_runs, &task_id, LocalAgentStatus::Running {
                current_action: action_desc,
            });

            let result = execute_tool(fn_name, fn_args).await;
            messages.push(OllamaAgentMessage::tool_result(result));
        }
    }
}

fn update_local_run_status(local_runs: &LocalRuns, task_id: &AmbientAgentTaskId, status: LocalAgentStatus) {
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
    ) -> std::result::Result<Vec<AIGeneratedCommand>, GenerateCommandsFromNaturalLanguageError> {
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
        self.server_api
            .generate_metadata_for_command(command)
            .await
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
        self.server_api.update_merkle_tree(embedding_config, nodes).await
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
            .update_agent_task(task_id, task_state, session_id, conversation_id, status_message)
            .await
    }

    async fn spawn_agent(&self, request: SpawnAgentRequest) -> AnyhowResult<SpawnAgentResponse> {
        let Some(config) = self.active_config() else {
            return self.server_api.spawn_agent(request).await;
        };

        let task_uuid = uuid::Uuid::new_v4();
        let run_uuid = uuid::Uuid::new_v4();
        let task_id: AmbientAgentTaskId = task_uuid.to_string().parse()
            .context("Failed to create local task ID")?;
        let run_id = run_uuid.to_string();
        let cancel_flag = Arc::new(AtomicBool::new(false));

        let state = LocalAgentState {
            task_id,
            run_id: run_id.clone(),
            prompt: request.prompt.clone(),
            created_at: chrono::Utc::now(),
            status: LocalAgentStatus::Running { current_action: "Starting Ollama agent...".to_string() },
            cancel_flag: cancel_flag.clone(),
        };

        self.local_runs.lock().insert(task_id, Arc::new(parking_lot::Mutex::new(state)));

        let http_client = self.http_client.clone();
        let local_runs = self.local_runs.clone();
        let prompt = request.prompt.clone();

        tokio::spawn(async move {
            run_ollama_agent_loop(config, http_client, task_id, prompt, local_runs).await;
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
        self.server_api.list_ambient_agent_tasks(limit, filter).await
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
        self.server_api.get_scheduled_agent_history(schedule_id).await
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
        self.server_api.get_block_snapshot(server_conversation_token).await
    }

    async fn delete_ai_conversation(
        &self,
        server_conversation_token: String,
    ) -> AnyhowResult<()> {
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

    async fn get_public_conversation(&self, conversation_id: &str) -> AnyhowResult<serde_json::Value> {
        self.server_api.get_public_conversation(conversation_id).await
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
