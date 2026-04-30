pub(super) fn agent_tools() -> Vec<serde_json::Value> {
    serde_json::from_str(r#"[
      {
        "type": "function",
        "function": {
          "name": "run_command",
          "description": "Execute a shell command and return its stdout and stderr. Use this for running scripts, compiling code, checking output, and performing git operations such as status, add, commit, and push.",
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

pub(super) const SUPPORTED_TOOL_NAMES: &[&str] = &[
    "run_command",
    "read_file",
    "write_file",
    "list_files",
    "search_in_files",
];

pub(super) fn is_supported_tool_name(name: &str) -> bool {
    SUPPORTED_TOOL_NAMES.contains(&name)
}

pub(super) async fn execute_tool(
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
            super::run_shell_command(command, working_dir).await
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
                    .map(|content| super::truncate_str(&content, 16000).to_string())
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
            let cmd = if cfg!(target_os = "windows") {
                format!("dir /B /S \"{path}\"")
            } else {
                format!("find '{path}' -name '{pattern}' 2>/dev/null | head -200")
            };
            super::run_shell_command(cmd, default_working_directory.map(str::to_string)).await
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
            super::run_shell_command(cmd, default_working_directory.map(str::to_string)).await
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
