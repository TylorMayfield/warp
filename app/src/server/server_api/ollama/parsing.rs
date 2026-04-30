fn split_top_level(content: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut quote_char = '\0';
    let mut escape = false;
    let mut brace_depth = 0usize;
    let mut bracket_depth = 0usize;
    let mut paren_depth = 0usize;

    for ch in content.chars() {
        if escape {
            current.push(ch);
            escape = false;
            continue;
        }

        if in_quotes {
            current.push(ch);
            if ch == '\\' {
                escape = true;
            } else if ch == quote_char {
                in_quotes = false;
            }
            continue;
        }

        match ch {
            '"' | '\'' => {
                in_quotes = true;
                quote_char = ch;
                current.push(ch);
            }
            '{' => {
                brace_depth += 1;
                current.push(ch);
            }
            '}' => {
                brace_depth = brace_depth.saturating_sub(1);
                current.push(ch);
            }
            '[' => {
                bracket_depth += 1;
                current.push(ch);
            }
            ']' => {
                bracket_depth = bracket_depth.saturating_sub(1);
                current.push(ch);
            }
            '(' => {
                paren_depth += 1;
                current.push(ch);
            }
            ')' => {
                paren_depth = paren_depth.saturating_sub(1);
                current.push(ch);
            }
            ';' | '\n'
                if brace_depth == 0 && bracket_depth == 0 && paren_depth == 0 =>
            {
                let trimmed = current.trim();
                if !trimmed.is_empty() {
                    parts.push(trimmed.to_string());
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }

    let trimmed = current.trim();
    if !trimmed.is_empty() {
        parts.push(trimmed.to_string());
    }

    parts
}

pub(super) fn normalize_assistant_message(
    message: super::OllamaAgentMessage,
) -> super::OllamaAgentMessage {
    if !message.tool_calls.is_empty() {
        return message;
    }

    let Some(content) = message.content.as_deref() else {
        return message;
    };

    let tool_calls = parse_tool_calls_from_content(content.trim());
    if tool_calls.is_empty() {
        return message;
    }

    super::OllamaAgentMessage {
        tool_calls,
        content: None,
        ..message
    }
}

pub(super) fn parse_tool_calls_from_content(content: &str) -> Vec<super::OllamaToolCall> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return vec![];
    }

    for candidate in candidate_fragments(trimmed) {
        let calls = parse_candidate(&candidate);
        if !calls.is_empty() {
            return calls;
        }
    }

    vec![]
}

pub(super) fn looks_like_tool_attempt(content: &str) -> bool {
    let content = content.trim();
    content.starts_with('{')
        || content.starts_with('[')
        || content.contains("tool")
        || content.contains("function")
        || content.contains("run_command")
        || content.contains("read_file")
        || content.contains("write_file")
        || content.contains("list_files")
        || content.contains("search_in_files")
        || content.contains("import ")
        || (content.contains('(') && content.contains(')'))
}

fn candidate_fragments(content: &str) -> Vec<String> {
    let mut candidates = vec![content.to_string()];

    if let Some(extracted) = extract_json_block(content) {
        candidates.push(extracted);
    }

    candidates.extend(split_top_level(content));
    candidates
}

fn parse_candidate(candidate: &str) -> Vec<super::OllamaToolCall> {
    parse_tool_call_json(candidate)
        .or_else(|| parse_function_style_tool_call(candidate).map(|call| vec![call]))
        .unwrap_or_default()
}

fn parse_tool_call_json(content: &str) -> Option<Vec<super::OllamaToolCall>> {
    let value: serde_json::Value = serde_json::from_str(content).ok()?;
    tool_calls_from_value(&value)
}

fn extract_json_block(content: &str) -> Option<String> {
    let start = content.find(|ch| ['{', '['].contains(&ch))?;
    let end = content.rfind(|ch| ['}', ']'].contains(&ch))?;
    (start < end).then(|| content[start..=end].to_string())
}

fn tool_calls_from_value(value: &serde_json::Value) -> Option<Vec<super::OllamaToolCall>> {
    match value {
        serde_json::Value::Array(items) => {
            let mut calls = Vec::new();
            for item in items {
                calls.extend(tool_calls_from_value(item)?);
            }
            Some(calls)
        }
        serde_json::Value::Object(object) => {
            if let Some(function) = object.get("function") {
                let function = function.as_object()?;
                let name = function.get("name")?.as_str()?.to_string();
                let arguments = function
                    .get("arguments")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                return Some(vec![super::OllamaToolCall {
                    function: super::OllamaToolCallFunction { name, arguments },
                }]);
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

            Some(vec![super::OllamaToolCall {
                function: super::OllamaToolCallFunction { name, arguments },
            }])
        }
        _ => None,
    }
}

fn parse_function_style_tool_call(content: &str) -> Option<super::OllamaToolCall> {
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

    Some(super::OllamaToolCall {
        function: super::OllamaToolCallFunction {
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
