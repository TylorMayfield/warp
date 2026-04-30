#!/usr/bin/env python3
"""
Runs lightweight workflow checks against a real Ollama model without building Warp.

The harness simulates Warp's local tool-calling loop:
- sends the real system prompt + runtime context to Ollama
- lets the model choose tool calls
- returns fixture outputs for those tool calls
- verifies the tool sequence against scenario expectations

Usage:
  python script/ollama_workflow_test.py --list
  python script/ollama_workflow_test.py --scenario git_stage_commit_push
  python script/ollama_workflow_test.py --model llama3.1:8b --scenario all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.1:8b"
MAX_ITERATIONS = 20

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Execute a shell command and return its stdout and stderr. "
                "Use this for running scripts, compiling code, checking output, "
                "and performing git operations such as status, add, commit, and push."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run",
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Optional working directory (absolute path)",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file at a given path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite a file with given content",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write into the file",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory, with optional glob pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Optional glob pattern to filter files (e.g. '*.rs')",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_files",
            "description": "Search for a text pattern inside files using grep-like search",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text or regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file path to search in",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether the search is case-sensitive (default true)",
                    },
                },
                "required": ["pattern", "path"],
            },
        },
    },
]


def build_local_agent_system_prompt(working_directory: str | None) -> str:
    cwd_context = f"Current working directory: {working_directory}. " if working_directory else ""
    return (
        "You are a helpful AI assistant with access to tools. "
        f"{cwd_context}"
        "Treat runtime context provided by Warp as authoritative when answering questions "
        "about the local machine or session. "
        "Use the available tools to complete the user's task when local context, file "
        "inspection, or command execution is required. "
        "You may use shell commands for git workflows when the user asks, including "
        "inspecting repository state, staging files, writing commits, and pushing branches. "
        "Assume file paths and shell commands should use the current working directory unless "
        "the user specifies otherwise. "
        "If runtime context already contains the answer, answer from that context directly "
        "instead of using a tool. "
        "Answer directly when the task can be completed from reasoning alone, such as simple "
        "math or explanation requests. "
        "Only call tools when they are actually needed. "
        "Do not claim you lack filesystem or terminal access. "
        "Available tool names are: run_command, read_file, write_file, list_files, search_in_files. "
        "Do not invent Python, bash, or JSON-RPC tools that are not listed. "
        "Do not invent files, paths, commands, or machine details that were not provided by "
        "the user, runtime context, or tool output. "
        "If a user asks about the current machine state and the answer is not already present "
        "in the runtime context, use a tool to inspect it instead of guessing or claiming you "
        "cannot access it. "
        "Use repository context from Warp plus git inspection commands to understand changed "
        "files before writing commit messages. "
        "For git workflows that change repository state, inspect first, then take actions step "
        "by step, using one tool call per shell command. "
        "Before staging, committing, or pushing, inspect repository state with git status or "
        "diff commands unless the required detail is already present from earlier tool results "
        "in the same interaction. "
        "Never combine multiple shell commands with &&, ;, or similar chaining inside one tool call. "
        "When asked to stage, commit, or push, prefer this order unless the user requests otherwise: "
        "inspect status, stage the intended files, inspect staged diff summary, commit, then push. "
        "Examples: "
        "If runtime context includes `Current local time: 2026-04-30T08:37:06-04:00` and the user asks "
        "`What time is it right now?`, answer with that time directly and do not call any tool. "
        "If the user asks to stage, commit, and push changes, first call `run_command` with "
        "`git status --short --branch`, then stage files with a separate `git add ...` command, "
        "then inspect staged changes with `git diff --cached --stat`, then commit with "
        "`git commit -m ...`, then push with `git push ...`. "
        "If the user asks what changed on the branch, inspect with `git status --short --branch` "
        "or `git diff --stat` before summarizing. "
        "When you decide to call a tool, return the tool call itself instead of describing it "
        "in prose or wrapping it in markdown. "
        "After completing the task, provide a clear summary of what you did and the results. "
        "If you encounter errors, explain what went wrong and what you tried."
    )


def format_runtime_context(scenario: dict[str, Any], cwd: str | None) -> str:
    runtime = scenario.get("runtime_context", {})
    current_time = runtime.get("current_time") or datetime.now().astimezone().isoformat()
    lines = [
        "Warp runtime context (authoritative, already known facts; do not re-check these facts with tools unless the user explicitly asks you to verify them):",
        f"Current local time: {current_time}",
    ]
    if cwd:
        lines.append(f"Current working directory: {cwd}")

    execution = runtime.get("execution_context") or {}
    os_ctx = execution.get("os") or {}
    if os_ctx.get("category"):
        lines.append(f"Operating system: {os_ctx['category']}")
    if os_ctx.get("distribution"):
        lines.append(f"OS distribution: {os_ctx['distribution']}")
    if execution.get("shell_name"):
        lines.append(f"Shell: {execution['shell_name']}")
    if execution.get("shell_version"):
        lines.append(f"Shell version: {execution['shell_version']}")

    git_ctx = runtime.get("git") or {}
    if git_ctx:
        lines.append("Git repository context:")
        if git_ctx.get("repo_root"):
            lines.append(f"Git root: {git_ctx['repo_root']}")
        if git_ctx.get("branch"):
            lines.append(f"Git branch: {git_ctx['branch']}")
        if git_ctx.get("head"):
            lines.append(f"Git HEAD: {git_ctx['head']}")
        if git_ctx.get("status_summary"):
            lines.append("Git status summary:")
            lines.append(git_ctx["status_summary"])
        if git_ctx.get("staged_diff_stat"):
            lines.append("Staged diff summary:")
            lines.append(git_ctx["staged_diff_stat"])
        if git_ctx.get("unstaged_diff_stat"):
            lines.append("Unstaged diff summary:")
            lines.append(git_ctx["unstaged_diff_stat"])

    return "\n".join(lines)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


def candidate_fragments(content: str) -> list[str]:
    fragments = [content.strip()]
    first_json = min(
        [idx for idx in [content.find("{"), content.find("[")] if idx != -1],
        default=-1,
    )
    last_json = max(content.rfind("}"), content.rfind("]"))
    if first_json != -1 and last_json > first_json:
        fragments.append(content[first_json : last_json + 1].strip())
    fragments.extend(split_top_level(content))
    seen: list[str] = []
    for fragment in fragments:
        if fragment and fragment not in seen:
            seen.append(fragment)
    return seen


def split_top_level(content: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    in_quotes = False
    quote_char = ""
    escape = False
    brace_depth = 0
    bracket_depth = 0
    paren_depth = 0

    for ch in content:
        if escape:
            current.append(ch)
            escape = False
            continue
        if in_quotes:
            current.append(ch)
            if ch == "\\":
                escape = True
            elif ch == quote_char:
                in_quotes = False
            continue

        if ch in ('"', "'"):
            in_quotes = True
            quote_char = ch
            current.append(ch)
        elif ch == "{":
            brace_depth += 1
            current.append(ch)
        elif ch == "}":
            brace_depth = max(0, brace_depth - 1)
            current.append(ch)
        elif ch == "[":
            bracket_depth += 1
            current.append(ch)
        elif ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
            current.append(ch)
        elif ch == "(":
            paren_depth += 1
            current.append(ch)
        elif ch == ")":
            paren_depth = max(0, paren_depth - 1)
            current.append(ch)
        elif ch in (";", "\n") and brace_depth == bracket_depth == paren_depth == 0:
            piece = "".join(current).strip()
            if piece:
                parts.append(piece)
            current = []
        else:
            current.append(ch)

    piece = "".join(current).strip()
    if piece:
        parts.append(piece)
    return parts


def tool_calls_from_value(value: Any) -> list[ToolCall] | None:
    if isinstance(value, list):
        calls: list[ToolCall] = []
        for item in value:
            parsed = tool_calls_from_value(item)
            if parsed is None:
                return None
            calls.extend(parsed)
        return calls

    if not isinstance(value, dict):
        return None

    if isinstance(value.get("function"), dict):
        function = value["function"]
        name = function.get("name")
        if not isinstance(name, str):
            return None
        arguments = function.get("arguments") or {}
        if not isinstance(arguments, dict):
            return None
        return [ToolCall(name=name, arguments=arguments)]

    name = value.get("name") or value.get("tool")
    if not isinstance(name, str):
        return None
    arguments = value.get("parameters")
    if arguments is None:
        arguments = value.get("arguments") or {}
    if not isinstance(arguments, dict):
        return None
    return [ToolCall(name=name, arguments=arguments)]


def parse_function_style_tool_call(content: str) -> ToolCall | None:
    open_paren = content.find("(")
    close_paren = content.rfind(")")
    if open_paren == -1 or close_paren <= open_paren:
        return None
    name = content[:open_paren].strip()
    if not name:
        return None

    args_src = content[open_paren + 1 : close_paren].strip()
    arguments: dict[str, Any] = {}
    if args_src:
        current: list[str] = []
        parts: list[str] = []
        in_quotes = False
        quote_char = ""
        escape = False
        for ch in args_src:
            if escape:
                current.append(ch)
                escape = False
                continue
            if ch == "\\" and in_quotes:
                current.append(ch)
                escape = True
                continue
            if in_quotes:
                if ch == quote_char:
                    in_quotes = False
                current.append(ch)
                continue
            if ch in ('"', "'"):
                in_quotes = True
                quote_char = ch
                current.append(ch)
            elif ch == ",":
                piece = "".join(current).strip()
                if piece:
                    parts.append(piece)
                current = []
            else:
                current.append(ch)
        piece = "".join(current).strip()
        if piece:
            parts.append(piece)

        for part in parts:
            if "=" not in part:
                return None
            key, raw_value = part.split("=", 1)
            key = key.strip()
            raw_value = raw_value.strip()
            if not key:
                return None
            if (
                (raw_value.startswith('"') and raw_value.endswith('"'))
                or (raw_value.startswith("'") and raw_value.endswith("'"))
            ):
                value: Any = raw_value[1:-1].replace('\\"', '"').replace("\\'", "'")
            elif raw_value in ("true", "false"):
                value = raw_value == "true"
            else:
                try:
                    value = int(raw_value)
                except ValueError:
                    value = raw_value
            arguments[key] = value

    return ToolCall(name=name, arguments=arguments)


def normalize_tool_calls(message: dict[str, Any]) -> tuple[list[ToolCall], str]:
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        parsed: list[ToolCall] = []
        for item in tool_calls:
            fn = item.get("function") or {}
            name = fn.get("name")
            arguments = fn.get("arguments") or {}
            if isinstance(name, str) and isinstance(arguments, dict):
                parsed.append(ToolCall(name=name, arguments=arguments))
        return parsed, ""

    content = (message.get("content") or "").strip()
    if not content:
        return [], ""

    for fragment in candidate_fragments(content):
        try:
            value = json.loads(fragment)
        except json.JSONDecodeError:
            value = None
        if value is not None:
            parsed = tool_calls_from_value(value)
            if parsed:
                return parsed, ""

        parsed_fn = parse_function_style_tool_call(fragment)
        if parsed_fn:
            return [parsed_fn], ""

    return [], content


def combined_user_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "user"
    ).lower()


def runtime_context_has_current_time(messages: list[dict[str, Any]]) -> bool:
    return any(
        message.get("role") == "system"
        and "Current local time:" in str(message.get("content") or "")
        for message in messages
    )


def is_time_or_date_request(user_text: str) -> bool:
    return any(
        pattern in user_text
        for pattern in [
            "what time",
            "current time",
            "time is it",
            "what's the time",
            "date is it",
            "current date",
            "today's date",
            "today date",
        ]
    )


def is_git_related_request(user_text: str) -> bool:
    return any(
        pattern in user_text
        for pattern in [
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
    )


def contains_shell_command_chaining(command: str) -> bool:
    return any(token in command for token in ["&&", "||", ";", "\n"])


def is_git_inspection_command(command: str) -> bool:
    lowered = command.strip().lower()
    return lowered.startswith(("git status", "git diff", "git log", "git show"))


def is_git_mutation_command(command: str) -> bool:
    lowered = command.strip().lower()
    return lowered.startswith(
        (
            "git add",
            "git commit",
            "git push",
            "git rm",
            "git mv",
            "git reset",
            "git checkout",
            "git cherry-pick",
            "git rebase",
            "git merge",
        )
    )


def has_prior_git_inspection(messages: list[dict[str, Any]]) -> bool:
    return has_prior_matching_command(messages, is_git_inspection_command)


def has_prior_matching_command(
    messages: list[dict[str, Any]], predicate
) -> bool:
    for message in messages:
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            if function.get("name") != "run_command":
                continue
            command = (function.get("arguments") or {}).get("command")
            if isinstance(command, str) and predicate(command):
                return True
    return False


def validate_tool_call(messages: list[dict[str, Any]], call: ToolCall) -> str | None:
    user_text = combined_user_text(messages)
    wants_stage = "stage" in user_text
    wants_commit = "commit" in user_text
    wants_push = "push" in user_text

    if (
        runtime_context_has_current_time(messages)
        and is_time_or_date_request(user_text)
        and "verify" not in user_text
        and "double-check" not in user_text
    ):
        return (
            "Policy: Warp runtime context already includes the current local time/date. Do not "
            "call any tool. Respond in plain language using the `Current local time:` value from "
            "runtime context."
        )

    if is_git_related_request(user_text) and not has_prior_git_inspection(messages):
        if call.name != "run_command":
            return (
                "Policy: For git workflows, inspect repository state first. Your next tool call "
                "should be `run_command` with `git status --short --branch`."
            )

        command = str(call.arguments.get("command", ""))
        if contains_shell_command_chaining(command):
            return (
                "Policy: Use one shell command per tool call. Do not chain git commands with "
                "`&&`, `||`, `;`, or newlines. First call `run_command` with "
                "`git status --short --branch`."
            )

        if is_git_mutation_command(command) and not is_git_inspection_command(command):
            return (
                "Policy: Inspect repository state first before staging, committing, or pushing. "
                "Your next tool call should be `run_command` with `git status --short --branch`."
            )

        if wants_stage and command.strip().lower().startswith("git diff --cached"):
            return (
                "Policy: Stage the intended files first. Your next tool call should be "
                "`run_command` with a `git add ...` command for the intended files."
            )

        if wants_commit and command.strip().lower().startswith("git commit"):
            return (
                "Policy: Before committing, stage the intended files and inspect the staged diff "
                "summary. If files are not staged yet, your next tool call should be "
                "`run_command` with a `git add ...` command."
            )

        if wants_push and command.strip().lower().startswith("git push"):
            return (
                "Policy: Inspect, stage, and commit the changes before pushing. Your next tool "
                "call should be `run_command` with `git status --short --branch`."
            )

    if call.name == "run_command":
        command = str(call.arguments.get("command", ""))
        lowered = command.strip().lower()
        if contains_shell_command_chaining(command):
            return (
                "Policy: Use one shell command per tool call. Do not combine multiple shell "
                "commands with `&&`, `||`, `;`, or newlines."
            )

        has_prior_git_add = has_prior_matching_command(
            messages, lambda cmd: cmd.strip().lower().startswith("git add")
        )
        has_prior_staged_diff = has_prior_matching_command(
            messages, lambda cmd: cmd.strip().lower().startswith("git diff --cached")
        )
        has_prior_git_commit = has_prior_matching_command(
            messages, lambda cmd: cmd.strip().lower().startswith("git commit")
        )

        if wants_stage and lowered.startswith("git diff --cached") and not has_prior_git_add:
            return (
                "Policy: Stage the intended files first. Your next tool call should be "
                "`run_command` with a `git add ...` command for the intended files."
            )

        if wants_commit and lowered.startswith("git commit") and (
            not has_prior_git_add or not has_prior_staged_diff
        ):
            return (
                "Policy: Before committing, stage the intended files and inspect the staged diff "
                "summary with `git diff --cached --stat`."
            )

        if wants_push and lowered.startswith("git push") and not has_prior_git_commit:
            return (
                "Policy: Create the commit before pushing the branch. Your next tool call should "
                "be `run_command` with `git commit -m ...` after the staged diff has been "
                "inspected."
            )

    return None


def post_chat(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/chat"
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Failed to reach Ollama at {url}: {exc}") from exc


def find_fixture_result(scenario: dict[str, Any], call: ToolCall) -> str:
    fixtures = scenario.get("fixtures", [])
    for fixture in fixtures:
        if fixture.get("tool") != call.name:
            continue
        key = fixture.get("argument", "command")
        actual = str(call.arguments.get(key, ""))
        pattern = fixture.get("matches")
        if pattern and not re.search(pattern, actual, re.IGNORECASE):
            continue
        return fixture["result"]
    raise AssertionError(
        f"No fixture matched tool call {call.name} with arguments {json.dumps(call.arguments)}"
    )


def validate_expected_step(step: dict[str, Any], call: ToolCall, index: int) -> None:
    if step.get("tool") != call.name:
        raise AssertionError(
            f"Step {index + 1} expected tool '{step.get('tool')}', got '{call.name}'"
        )
    argument = step.get("argument", "command")
    pattern = step.get("matches")
    if pattern is None:
        return
    actual = str(call.arguments.get(argument, ""))
    if not re.search(pattern, actual, re.IGNORECASE):
        raise AssertionError(
            f"Step {index + 1} expected {argument} to match /{pattern}/, got: {actual}"
        )


def call_matches_pattern(call: ToolCall, pattern_spec: dict[str, Any]) -> bool:
    if pattern_spec.get("tool") != call.name:
        return False
    argument = pattern_spec.get("argument", "command")
    pattern = pattern_spec.get("matches")
    if pattern is None:
        return True
    actual = str(call.arguments.get(argument, ""))
    return re.search(pattern, actual, re.IGNORECASE) is not None


def run_scenario(
    base_url: str,
    model: str,
    scenario: dict[str, Any],
    cwd_override: str | None,
    verbose: bool,
) -> tuple[bool, str]:
    cwd = cwd_override or scenario.get("working_directory")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_local_agent_system_prompt(cwd)},
        {"role": "system", "content": format_runtime_context(scenario, cwd)},
        {"role": "user", "content": scenario["prompt"]},
    ]
    policy_history = list(messages)
    expected_steps = scenario.get("expected_tool_calls", [])
    allowed_extra_steps = scenario.get("allowed_extra_tool_calls", [])
    seen_calls: list[ToolCall] = []
    final_answer = ""
    policy_violations = 0
    max_policy_violations = int(scenario.get("max_policy_violations", 0))

    for _ in range(MAX_ITERATIONS):
        payload = {
            "model": model,
            "messages": messages,
            "tools": TOOLS,
            "stream": False,
            "options": {"num_predict": 4096},
        }
        response = post_chat(base_url, payload)
        message = response.get("message") or {}
        tool_calls, content = normalize_tool_calls(message)

        if verbose:
            print(f"\n[{scenario['name']}] assistant raw:")
            print(json.dumps(message, indent=2))

        if not tool_calls:
            final_answer = content
            break

        messages.append(
            {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": [
                    {"function": {"name": c.name, "arguments": c.arguments}} for c in tool_calls
                ],
            }
        )

        for call in tool_calls:
            policy_error = validate_tool_call(policy_history, call)
            if policy_error:
                policy_violations += 1
                messages.append({"role": "tool", "content": policy_error})
                continue

            step_index = len(seen_calls)
            if step_index >= len(expected_steps):
                if any(call_matches_pattern(call, spec) for spec in allowed_extra_steps):
                    seen_calls.append(call)
                    result = find_fixture_result(scenario, call)
                    policy_history.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {"function": {"name": call.name, "arguments": call.arguments}}
                            ],
                        }
                    )
                    policy_history.append({"role": "tool", "content": result})
                    messages.append({"role": "tool", "content": result})
                    continue
                raise AssertionError(
                    f"Unexpected extra tool call {call.name}: {json.dumps(call.arguments)}"
                )
            validate_expected_step(expected_steps[step_index], call, step_index)
            seen_calls.append(call)
            result = find_fixture_result(scenario, call)
            policy_history.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": call.name, "arguments": call.arguments}}
                    ],
                }
            )
            policy_history.append({"role": "tool", "content": result})
            messages.append({"role": "tool", "content": result})

    else:
        raise AssertionError("Exceeded maximum iterations without a final answer")

    minimum_tool_calls = int(scenario.get("minimum_tool_calls", len(expected_steps)))
    maximum_tool_calls = int(scenario.get("maximum_tool_calls", len(expected_steps)))

    if len(seen_calls) < minimum_tool_calls or len(seen_calls) > maximum_tool_calls:
        raise AssertionError(
            f"Expected between {minimum_tool_calls} and {maximum_tool_calls} tool calls, observed {len(seen_calls)}"
        )

    if policy_violations > max_policy_violations:
        raise AssertionError(
            f"Observed {policy_violations} policy violations, expected at most {max_policy_violations}"
        )

    final_answer_pattern = scenario.get("final_answer_matches")
    if final_answer_pattern and not re.search(final_answer_pattern, final_answer, re.IGNORECASE | re.DOTALL):
        raise AssertionError(
            f"Final answer did not match /{final_answer_pattern}/.\nActual:\n{final_answer}"
        )

    summary = [
        f"scenario={scenario['name']}",
        f"tool_calls={len(seen_calls)}",
        f"policy_violations={policy_violations}",
    ]
    for idx, call in enumerate(seen_calls, 1):
        key = "command" if "command" in call.arguments else next(iter(call.arguments), "")
        snippet = str(call.arguments.get(key, ""))[:120]
        summary.append(f"{idx}. {call.name}: {snippet}")
    if final_answer:
        summary.append(f"final={final_answer[:160]}")
    return True, "\n".join(summary)


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    scenarios = data.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError(f"Invalid scenario file: {path}")
    return scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--scenario-file",
        default=str(Path(__file__).with_name("ollama_workflow_cases.json")),
    )
    parser.add_argument("--scenario", default="all")
    parser.add_argument("--cwd")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenarios = load_scenarios(Path(args.scenario_file))

    if args.list:
        for scenario in scenarios:
            print(scenario["name"])
        return 0

    selected = scenarios
    if args.scenario != "all":
        selected = [scenario for scenario in scenarios if scenario["name"] == args.scenario]
        if not selected:
            print(f"No scenario named '{args.scenario}'", file=sys.stderr)
            return 2

    failures = 0
    for scenario in selected:
        print(f"Running {scenario['name']} against model {args.model}...")
        try:
            _, summary = run_scenario(
                args.base_url,
                args.model,
                scenario,
                args.cwd,
                args.verbose,
            )
            print(f"PASS\n{summary}\n")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL\nscenario={scenario['name']}\nerror={exc}\n")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
