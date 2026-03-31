use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;
use wait_timeout::ChildExt;

#[derive(Clone, Debug, Serialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub args: HashMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PluginSpec {
    name: String,
    description: String,
    command: Vec<String>,
    #[serde(default)]
    args_schema: HashMap<String, String>,
}

#[derive(Clone, Debug)]
pub struct ToolExecutor {
    workspace_root: PathBuf,
    auto_mode: bool,
    shell_timeout_seconds: u64,
    memory_file: PathBuf,
    plugins_dir: PathBuf,
    plugins: HashMap<String, PluginSpec>,
}

#[derive(Debug)]
struct ProcessResult {
    exit_code: i32,
    stdout: String,
    stderr: String,
    timed_out: bool,
}

#[derive(Debug)]
struct PythonSymbol {
    name: String,
    kind: String,
    start_line: usize,
    end_line: usize,
}

impl ToolExecutor {
    pub fn new(
        workspace_root: PathBuf,
        auto_mode: bool,
        shell_timeout_seconds: u64,
        memory_file: PathBuf,
        plugins_dir: PathBuf,
    ) -> Self {
        let root = workspace_root
            .canonicalize()
            .unwrap_or_else(|_| workspace_root.clone());
        let mut instance = Self {
            workspace_root: root,
            auto_mode,
            shell_timeout_seconds,
            memory_file,
            plugins_dir,
            plugins: HashMap::new(),
        };
        instance.plugins = instance.load_plugins();
        instance
    }

    pub fn list_specs(&self) -> Vec<ToolSpec> {
        let mut specs = vec![
            tool_spec(
                "list_files",
                "Recursively list files and directories under a path.",
                &[
                    "path:string path relative to workspace, default '.'",
                    "max_depth:integer depth limit, default 3",
                ],
            ),
            tool_spec(
                "read_file",
                "Read a text file with line numbers.",
                &[
                    "path:string path relative to workspace",
                    "start_line:1-based start line, default 1",
                    "end_line:1-based end line inclusive, default 200",
                ],
            ),
            tool_spec(
                "write_file",
                "Write text content to a file (overwrites existing).",
                &[
                    "path:string path relative to workspace",
                    "content:full file content as string",
                ],
            ),
            tool_spec(
                "replace_in_file",
                "Replace text in a file with exact string matching.",
                &[
                    "path:string path relative to workspace",
                    "old:text to replace",
                    "new:replacement text",
                    "count:optional max replacements, default all",
                ],
            ),
            tool_spec(
                "structured_patch",
                "Apply structured edit operations to files.",
                &["operations:list of operations: replace_text, replace_lines, append_text"],
            ),
            tool_spec(
                "python_symbol_overview",
                "List top-level Python symbols (functions/classes) in a file.",
                &["path:python file path relative to workspace"],
            ),
            tool_spec(
                "replace_python_symbol",
                "Replace one top-level Python function/class by symbol name.",
                &[
                    "path:python file path relative to workspace",
                    "symbol:function or class name",
                    "symbol_type:optional 'function' or 'class'",
                    "code:replacement code block",
                ],
            ),
            tool_spec(
                "run_shell",
                "Run a shell command in the workspace root.",
                &["command:shell command string"],
            ),
            tool_spec(
                "open_calculator",
                "Open a desktop calculator application using platform-aware commands.",
                &[],
            ),
            tool_spec(
                "git_status",
                "Show git status (short + branch) for the workspace repository.",
                &[],
            ),
            tool_spec(
                "git_diff",
                "Show git diff for workspace, optionally for a specific path.",
                &[
                    "path:optional path relative to workspace",
                    "staged:optional boolean, default false",
                    "unified:optional context lines, default 3",
                ],
            ),
            tool_spec(
                "git_log",
                "Show recent commit log lines.",
                &["count:optional number of commits, default 10"],
            ),
            tool_spec(
                "git_commit_plan",
                "Summarize pending changes and propose a commit message.",
                &[],
            ),
            tool_spec(
                "git_commit",
                "Create a git commit in workspace repository.",
                &[
                    "message:commit message",
                    "add_all:optional boolean, default false",
                ],
            ),
            tool_spec(
                "apply_patch",
                "Apply a unified diff patch using git apply.",
                &["patch:unified diff text"],
            ),
            tool_spec(
                "memory_get",
                "Read persistent memory key/value entries.",
                &["key:optional key; omit to return full memory"],
            ),
            tool_spec(
                "memory_set",
                "Set a persistent memory key/value entry.",
                &["key:memory key", "value:memory value"],
            ),
            tool_spec(
                "memory_delete",
                "Delete a persistent memory key.",
                &["key:memory key"],
            ),
            tool_spec(
                "memory_search",
                "Search persistent memory by text query.",
                &[
                    "query:search string",
                    "limit:optional max results, default 10",
                ],
            ),
            tool_spec(
                "parallel_tools",
                "Execute multiple tool calls concurrently and return all results.",
                &[
                    "calls:list of tool calls [{tool, args}]",
                    "max_workers:optional worker count, default 4",
                ],
            ),
            tool_spec(
                "reload_plugins",
                "Reload plugin tools from plugins directory.",
                &[],
            ),
            tool_spec(
                "model_routing",
                "Return current CLI model routing configuration.",
                &[],
            ),
        ];

        let mut plugin_names: Vec<&String> = self.plugins.keys().collect();
        plugin_names.sort();
        for name in plugin_names {
            if let Some(plugin) = self.plugins.get(name) {
                specs.push(ToolSpec {
                    name: format!("plugin.{}", plugin.name),
                    description: plugin.description.clone(),
                    args: plugin.args_schema.clone(),
                });
            }
        }

        specs
    }

    pub fn render_specs_for_prompt(&self) -> String {
        serde_json::to_string_pretty(&self.list_specs()).unwrap_or_else(|_| "[]".to_string())
    }

    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    pub fn set_workspace(
        &mut self,
        workspace_root: PathBuf,
        memory_file: PathBuf,
        plugins_dir: PathBuf,
    ) -> Result<PathBuf, String> {
        let root = workspace_root
            .canonicalize()
            .map_err(|err| format!("failed to resolve workspace: {err}"))?;
        if !root.is_dir() {
            return Err(format!("workspace is not a directory: {}", root.display()));
        }

        self.workspace_root = root.clone();
        self.memory_file = memory_file;
        self.plugins_dir = plugins_dir;
        self.plugins = self.load_plugins();
        Ok(root)
    }

    pub fn execute(&mut self, tool_name: &str, args: &Value, model_routing: &Value) -> String {
        if tool_name == "reload_plugins" {
            self.plugins = self.load_plugins();
            let names: Vec<String> = self.plugins.keys().cloned().collect();
            return json!({"loaded_plugins": names}).to_string();
        }
        if tool_name == "model_routing" {
            return model_routing.to_string();
        }

        self.execute_readonly(tool_name, args, model_routing)
    }

    fn execute_readonly(&self, tool_name: &str, args: &Value, model_routing: &Value) -> String {
        let Some(map) = args.as_object() else {
            return "ERROR: args must be an object".to_string();
        };

        if let Some(plugin_name) = tool_name.strip_prefix("plugin.") {
            return self.run_plugin_tool(plugin_name, map);
        }

        match tool_name {
            "list_files" => {
                let path = map
                    .get("path")
                    .and_then(Value::as_str)
                    .unwrap_or(".")
                    .to_string();
                let max_depth = map.get("max_depth").and_then(Value::as_i64).unwrap_or(3);
                self.list_files(&path, max_depth as usize)
            }
            "read_file" => {
                let Some(path) = map.get("path").and_then(Value::as_str) else {
                    return "ERROR: missing argument 'path'".to_string();
                };
                let start_line = map.get("start_line").and_then(Value::as_i64).unwrap_or(1);
                let end_line = map.get("end_line").and_then(Value::as_i64).unwrap_or(200);
                self.read_file(path, start_line as usize, end_line as usize)
            }
            "write_file" => {
                let Some(path) = map.get("path").and_then(Value::as_str) else {
                    return "ERROR: missing argument 'path'".to_string();
                };
                let content = map.get("content").and_then(Value::as_str).unwrap_or("");
                self.write_file(path, content)
            }
            "replace_in_file" => {
                let Some(path) = map.get("path").and_then(Value::as_str) else {
                    return "ERROR: missing argument 'path'".to_string();
                };
                let Some(old) = map.get("old").and_then(Value::as_str) else {
                    return "ERROR: missing argument 'old'".to_string();
                };
                let new = map.get("new").and_then(Value::as_str).unwrap_or("");
                let count = map.get("count").and_then(Value::as_i64).map(|v| v as usize);
                self.replace_in_file(path, old, new, count)
            }
            "structured_patch" => self.structured_patch(map.get("operations")),
            "python_symbol_overview" => {
                let Some(path) = map.get("path").and_then(Value::as_str) else {
                    return "ERROR: missing argument 'path'".to_string();
                };
                self.python_symbol_overview(path)
            }
            "replace_python_symbol" => {
                let Some(path) = map.get("path").and_then(Value::as_str) else {
                    return "ERROR: missing argument 'path'".to_string();
                };
                let Some(symbol) = map.get("symbol").and_then(Value::as_str) else {
                    return "ERROR: missing argument 'symbol'".to_string();
                };
                let Some(code) = map.get("code").and_then(Value::as_str) else {
                    return "ERROR: missing argument 'code'".to_string();
                };
                let symbol_type = map.get("symbol_type").and_then(Value::as_str);
                self.replace_python_symbol(path, symbol, symbol_type, code)
            }
            "run_shell" => {
                let Some(command) = map.get("command").and_then(Value::as_str) else {
                    return "ERROR: missing argument 'command'".to_string();
                };
                self.run_shell(command)
            }
            "open_calculator" => self.open_calculator(),
            "git_status" => self.git_status(),
            "git_diff" => {
                let path = map.get("path").and_then(Value::as_str);
                let staged = map.get("staged").and_then(Value::as_bool).unwrap_or(false);
                let unified = map.get("unified").and_then(Value::as_i64).unwrap_or(3);
                self.git_diff(path, staged, unified as usize)
            }
            "git_log" => {
                let count = map.get("count").and_then(Value::as_i64).unwrap_or(10);
                self.git_log(count as usize)
            }
            "git_commit_plan" => self.git_commit_plan(),
            "git_commit" => {
                let Some(message) = map.get("message").and_then(Value::as_str) else {
                    return "ERROR: missing argument 'message'".to_string();
                };
                let add_all = map.get("add_all").and_then(Value::as_bool).unwrap_or(false);
                self.git_commit(message, add_all)
            }
            "apply_patch" => {
                let Some(patch) = map.get("patch").and_then(Value::as_str) else {
                    return "ERROR: missing argument 'patch'".to_string();
                };
                self.apply_patch(patch)
            }
            "memory_get" => {
                let key = map.get("key").and_then(Value::as_str);
                self.memory_get(key)
            }
            "memory_set" => {
                let Some(key) = map.get("key").and_then(Value::as_str) else {
                    return "ERROR: missing argument 'key'".to_string();
                };
                let value = map.get("value").cloned().unwrap_or(Value::Null);
                self.memory_set(key, value)
            }
            "memory_delete" => {
                let Some(key) = map.get("key").and_then(Value::as_str) else {
                    return "ERROR: missing argument 'key'".to_string();
                };
                self.memory_delete(key)
            }
            "memory_search" => {
                let Some(query) = map.get("query").and_then(Value::as_str) else {
                    return "ERROR: missing argument 'query'".to_string();
                };
                let limit = map.get("limit").and_then(Value::as_i64).unwrap_or(10);
                self.memory_search(query, limit as usize)
            }
            "parallel_tools" => {
                self.parallel_tools(map.get("calls"), map.get("max_workers"), model_routing)
            }
            "reload_plugins" => {
                "ERROR: reload_plugins must be called via mutable execute".to_string()
            }
            "model_routing" => model_routing.to_string(),
            other => format!("ERROR: unknown tool '{}'", other),
        }
    }

    fn list_files(&self, path: &str, max_depth: usize) -> String {
        let resolved = match self.resolve_existing_path(path) {
            Ok(value) => value,
            Err(err) => return format!("ERROR: {err}"),
        };

        let safe_depth = max_depth.clamp(0, 8);
        let mut lines: Vec<String> = Vec::new();
        self.walk_tree(&resolved, safe_depth, &mut lines);
        if lines.len() > 1500 {
            lines.truncate(1500);
            lines.push("... output truncated ...".to_string());
        }
        lines.join("\n")
    }

    fn walk_tree(&self, node: &Path, depth_left: usize, lines: &mut Vec<String>) {
        if lines.len() >= 1500 {
            return;
        }

        let rel = self.to_workspace_rel(node);
        if node.is_file() {
            lines.push(rel);
            return;
        }

        let label = if rel == "." {
            ".".to_string()
        } else {
            format!("{rel}/")
        };
        lines.push(label.clone());

        if depth_left == 0 {
            return;
        }

        let mut children = match fs::read_dir(node) {
            Ok(items) => items.filter_map(Result::ok).collect::<Vec<_>>(),
            Err(_) => {
                lines.push(format!("{} [permission denied]", label));
                return;
            }
        };

        children.sort_by_key(|entry| {
            let is_file = entry.path().is_file();
            let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
            (is_file, name)
        });

        for child in children {
            if lines.len() >= 1500 {
                break;
            }
            self.walk_tree(&child.path(), depth_left - 1, lines);
        }
    }

    fn read_file(&self, path: &str, start_line: usize, end_line: usize) -> String {
        let resolved = match self.resolve_existing_file(path) {
            Ok(value) => value,
            Err(err) => return format!("ERROR: {err}"),
        };

        let bytes = match fs::read(&resolved) {
            Ok(value) => value,
            Err(err) => return format!("ERROR: failed reading file: {err}"),
        };
        let text = String::from_utf8_lossy(&bytes).to_string();
        let lines: Vec<&str> = text.lines().collect();

        let start = if start_line == 0 { 1 } else { start_line };
        let mut end = if end_line < start { start } else { end_line };
        end = end.min(start + 999);

        let start_idx = start.saturating_sub(1);
        let end_idx = end.min(lines.len());

        let mut numbered = Vec::new();
        for (idx, line) in lines
            .iter()
            .enumerate()
            .skip(start_idx)
            .take(end_idx.saturating_sub(start_idx))
        {
            numbered.push(format!("{:5}: {}", idx + 1, line));
        }

        let header = format!("FILE: {path} (lines {start}-{end_idx} of {})", lines.len());
        if numbered.is_empty() {
            format!("{header}\n<no content in selected range>")
        } else {
            format!("{header}\n{}", numbered.join("\n"))
        }
    }

    fn write_file(&self, path: &str, content: &str) -> String {
        let resolved = match self.resolve_writable_path(path) {
            Ok(value) => value,
            Err(err) => return format!("ERROR: {err}"),
        };

        if let Some(parent) = resolved.parent() {
            if let Err(err) = fs::create_dir_all(parent) {
                return format!("ERROR: failed creating parent directory: {err}");
            }
        }

        match fs::write(&resolved, content.as_bytes()) {
            Ok(_) => format!("WROTE: {path} ({} bytes)", content.len()),
            Err(err) => format!("ERROR: failed writing file: {err}"),
        }
    }

    fn replace_in_file(&self, path: &str, old: &str, new: &str, count: Option<usize>) -> String {
        if old.is_empty() {
            return "ERROR: old text cannot be empty".to_string();
        }

        let resolved = match self.resolve_existing_file(path) {
            Ok(value) => value,
            Err(err) => return format!("ERROR: {err}"),
        };

        let bytes = match fs::read(&resolved) {
            Ok(value) => value,
            Err(err) => return format!("ERROR: failed reading file: {err}"),
        };
        let text = String::from_utf8_lossy(&bytes).to_string();
        let occurrences = text.matches(old).count();
        if occurrences == 0 {
            return format!("NO_CHANGE: '{}' not found in {path}", old);
        }

        let (updated, applied) = if let Some(limit) = count {
            (text.replacen(old, new, limit), occurrences.min(limit))
        } else {
            (text.replace(old, new), occurrences)
        };

        match fs::write(&resolved, updated.as_bytes()) {
            Ok(_) => format!("REPLACED: {path} occurrences={occurrences} applied={applied}"),
            Err(err) => format!("ERROR: failed writing file: {err}"),
        }
    }

    fn structured_patch(&self, operations: Option<&Value>) -> String {
        let Some(list) = operations.and_then(Value::as_array) else {
            return "ERROR: operations must be a non-empty list".to_string();
        };
        if list.is_empty() {
            return "ERROR: operations must be a non-empty list".to_string();
        }

        let mut results = Vec::new();
        for (index, op) in list.iter().enumerate() {
            let Some(map) = op.as_object() else {
                results.push(format!("op#{index}: ERROR operation must be an object"));
                continue;
            };
            let op_type = map.get("type").and_then(Value::as_str).unwrap_or("");
            match op_type {
                "replace_text" => {
                    let path = map.get("path").and_then(Value::as_str);
                    let old = map.get("old").and_then(Value::as_str);
                    let new = map.get("new").and_then(Value::as_str).unwrap_or("");
                    let count = map.get("count").and_then(Value::as_i64).map(|v| v as usize);
                    let result = match (path, old) {
                        (Some(path), Some(old)) => self.replace_in_file(path, old, new, count),
                        _ => "ERROR: replace_text needs path and old".to_string(),
                    };
                    results.push(format!("op#{index} replace_text: {result}"));
                }
                "replace_lines" => {
                    let path = map.get("path").and_then(Value::as_str);
                    let start = map.get("start_line").and_then(Value::as_i64);
                    let end = map.get("end_line").and_then(Value::as_i64);
                    let new_text = map.get("new_text").and_then(Value::as_str).unwrap_or("");
                    let result = match (path, start, end) {
                        (Some(path), Some(start), Some(end)) => {
                            self.replace_lines(path, start as usize, end as usize, new_text)
                        }
                        _ => "ERROR: replace_lines needs path/start_line/end_line".to_string(),
                    };
                    results.push(format!("op#{index} replace_lines: {result}"));
                }
                "append_text" => {
                    let path = map.get("path").and_then(Value::as_str);
                    let text = map.get("text").and_then(Value::as_str).unwrap_or("");
                    let result = match path {
                        Some(path) => self.append_text(path, text),
                        None => "ERROR: append_text needs path".to_string(),
                    };
                    results.push(format!("op#{index} append_text: {result}"));
                }
                other => results.push(format!(
                    "op#{index}: ERROR unknown operation type '{other}'"
                )),
            }
        }

        results.join("\n")
    }

    fn replace_lines(
        &self,
        path: &str,
        start_line: usize,
        end_line: usize,
        new_text: &str,
    ) -> String {
        let resolved = match self.resolve_existing_file(path) {
            Ok(value) => value,
            Err(err) => return format!("ERROR: {err}"),
        };
        if start_line < 1 || end_line < start_line {
            return "ERROR: invalid line range".to_string();
        }

        let bytes = match fs::read(&resolved) {
            Ok(value) => value,
            Err(err) => return format!("ERROR: failed reading file: {err}"),
        };
        let text = String::from_utf8_lossy(&bytes).to_string();
        let had_newline = text.ends_with('\n');
        let mut lines: Vec<String> = text.lines().map(|v| v.to_string()).collect();

        if start_line > lines.len() + 1 {
            return format!("ERROR: start_line out of range for {path}");
        }

        let start_idx = start_line - 1;
        let end_idx = end_line.min(lines.len());
        let replacement: Vec<String> = new_text.lines().map(|v| v.to_string()).collect();
        lines.splice(start_idx..end_idx, replacement);

        let mut updated = lines.join("\n");
        if had_newline {
            updated.push('\n');
        }

        match fs::write(&resolved, updated.as_bytes()) {
            Ok(_) => format!("WROTE: {path} lines {start_line}-{end_line}"),
            Err(err) => format!("ERROR: failed writing file: {err}"),
        }
    }

    fn append_text(&self, path: &str, text: &str) -> String {
        let resolved = match self.resolve_writable_path(path) {
            Ok(value) => value,
            Err(err) => return format!("ERROR: {err}"),
        };

        if let Some(parent) = resolved.parent() {
            if let Err(err) = fs::create_dir_all(parent) {
                return format!("ERROR: failed creating parent directory: {err}");
            }
        }

        match fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&resolved)
        {
            Ok(mut handle) => match handle.write_all(text.as_bytes()) {
                Ok(_) => format!("APPENDED: {path} ({} bytes)", text.len()),
                Err(err) => format!("ERROR: failed appending to file: {err}"),
            },
            Err(err) => format!("ERROR: failed opening file for append: {err}"),
        }
    }

    fn python_symbol_overview(&self, path: &str) -> String {
        let resolved = match self.resolve_existing_file(path) {
            Ok(value) => value,
            Err(err) => return format!("ERROR: {err}"),
        };
        if resolved.extension().and_then(|v| v.to_str()) != Some("py") {
            return "ERROR: python_symbol_overview only supports .py files".to_string();
        }

        let source = match fs::read_to_string(&resolved) {
            Ok(value) => value,
            Err(err) => return format!("ERROR: failed reading Python file: {err}"),
        };

        let symbols = self.parse_python_symbols(&source);
        json!({"path": path, "symbols": symbols.iter().map(|s| json!({
            "name": s.name,
            "type": s.kind,
            "start_line": s.start_line,
            "end_line": s.end_line,
        })).collect::<Vec<_>>()})
        .to_string()
    }

    fn replace_python_symbol(
        &self,
        path: &str,
        symbol: &str,
        symbol_type: Option<&str>,
        code: &str,
    ) -> String {
        let resolved = match self.resolve_existing_file(path) {
            Ok(value) => value,
            Err(err) => return format!("ERROR: {err}"),
        };
        if resolved.extension().and_then(|v| v.to_str()) != Some("py") {
            return "ERROR: replace_python_symbol only supports .py files".to_string();
        }

        let source = match fs::read_to_string(&resolved) {
            Ok(value) => value,
            Err(err) => return format!("ERROR: failed reading Python file: {err}"),
        };

        let symbols = self.parse_python_symbols(&source);
        let wanted = symbol_type.map(|v| v.trim().to_ascii_lowercase());
        let target = symbols.iter().find(|s| {
            if s.name != symbol {
                return false;
            }
            match &wanted {
                Some(kind) if kind == "function" => s.kind == "function",
                Some(kind) if kind == "class" => s.kind == "class",
                Some(_) => false,
                None => true,
            }
        });

        let Some(target) = target else {
            return format!("ERROR: symbol not found: {symbol}");
        };

        self.replace_lines(path, target.start_line, target.end_line, code)
    }

    fn parse_python_symbols(&self, source: &str) -> Vec<PythonSymbol> {
        let pattern = Regex::new(r"^(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)").unwrap();
        let lines: Vec<&str> = source.lines().collect();

        let mut starts: Vec<(usize, String, String)> = Vec::new();
        for (idx, line) in lines.iter().enumerate() {
            let trimmed = line.trim_end();
            if trimmed.is_empty() {
                continue;
            }
            if line.starts_with(' ') || line.starts_with('\t') {
                continue;
            }
            if let Some(caps) = pattern.captures(line) {
                let kind = caps
                    .get(1)
                    .map(|v| {
                        if v.as_str() == "def" {
                            "function".to_string()
                        } else {
                            "class".to_string()
                        }
                    })
                    .unwrap_or_else(|| "unknown".to_string());
                let name = caps
                    .get(2)
                    .map(|v| v.as_str().to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                starts.push((idx + 1, name, kind));
            }
        }

        let mut symbols = Vec::new();
        for index in 0..starts.len() {
            let (start_line, name, kind) = &starts[index];
            let end_line = if let Some((next_start, _, _)) = starts.get(index + 1) {
                next_start.saturating_sub(1)
            } else {
                lines.len().max(*start_line)
            };
            symbols.push(PythonSymbol {
                name: name.clone(),
                kind: kind.clone(),
                start_line: *start_line,
                end_line,
            });
        }
        symbols
    }

    fn run_shell(&self, command: &str) -> String {
        let normalized = command.trim();
        if normalized.eq_ignore_ascii_case("calc") || normalized.eq_ignore_ascii_case("calculator")
        {
            return self.open_calculator();
        }

        if !self.confirm_prompt(&format!("Approve shell command? [y/N]\n$ {command}")) {
            return "DENIED: shell command not approved by user".to_string();
        }

        let mut cmd = Command::new("bash");
        cmd.arg("-lc").arg(command);
        cmd.current_dir(&self.workspace_root);

        match self.run_process(&mut cmd, None) {
            Ok(result) => self.format_process_output(result),
            Err(err) => format!("ERROR: {err}"),
        }
    }

    fn open_calculator(&self) -> String {
        if !self.confirm_prompt("Approve opening calculator? [y/N]") {
            return "DENIED: calculator launch not approved by user".to_string();
        }

        let candidates: Vec<Vec<&str>> = if cfg!(target_os = "windows") {
            vec![vec!["calc.exe"]]
        } else if cfg!(target_os = "macos") {
            vec![vec!["open", "-a", "Calculator"]]
        } else {
            vec![
                vec!["gnome-calculator"],
                vec!["kcalc"],
                vec!["mate-calc"],
                vec!["qalculate-gtk"],
                vec!["galculator"],
                vec!["xcalc"],
                vec!["calc"],
            ]
        };

        let mut attempts = Vec::new();
        for candidate in candidates {
            if candidate.is_empty() {
                continue;
            }
            let cmd = candidate[0];
            let args = &candidate[1..];
            match self.launch_detached(cmd, args) {
                Ok(_) => {
                    let launched = std::iter::once(cmd)
                        .chain(args.iter().copied())
                        .collect::<Vec<_>>()
                        .join(" ");
                    return format!("CALCULATOR_OPENED: {launched}");
                }
                Err(err) => attempts.push(format!(
                    "{}: {}",
                    std::iter::once(cmd)
                        .chain(args.iter().copied())
                        .collect::<Vec<_>>()
                        .join(" "),
                    err
                )),
            }
        }

        format!(
            "ERROR: no calculator command succeeded. Attempts:\n{}",
            attempts.join("\n")
        )
    }

    fn launch_detached(&self, program: &str, args: &[&str]) -> io::Result<()> {
        let mut command = Command::new(program);
        command
            .args(args)
            .current_dir(&self.workspace_root)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        let mut child = command.spawn()?;
        thread::spawn(move || {
            let _ = child.wait();
        });
        Ok(())
    }

    fn git_status(&self) -> String {
        if let Some(err) = self.require_git_repo() {
            return err;
        }

        let mut cmd = Command::new("git");
        cmd.arg("-C")
            .arg(&self.workspace_root)
            .arg("status")
            .arg("--short")
            .arg("--branch")
            .arg("--")
            .arg(".");

        match self.run_process(&mut cmd, None) {
            Ok(result) => self.format_process_output(result),
            Err(err) => format!("ERROR: {err}"),
        }
    }

    fn git_diff(&self, path: Option<&str>, staged: bool, unified: usize) -> String {
        if let Some(err) = self.require_git_repo() {
            return err;
        }

        let mut cmd = Command::new("git");
        cmd.arg("-C")
            .arg(&self.workspace_root)
            .arg("diff")
            .arg("--no-color")
            .arg(format!("--unified={}", unified.clamp(0, 20)));

        if staged {
            cmd.arg("--staged");
        }

        if let Some(path) = path {
            let resolved = match self.resolve_existing_or_virtual(path) {
                Ok(value) => value,
                Err(err) => return format!("ERROR: {err}"),
            };
            let rel = self.to_workspace_rel_path(&resolved);
            cmd.arg("--").arg(rel);
        } else {
            cmd.arg("--").arg(".");
        }

        match self.run_process(&mut cmd, None) {
            Ok(result) => {
                if result.exit_code == 0 && result.stdout.trim().is_empty() {
                    "NO_DIFF: no changes for selected scope".to_string()
                } else {
                    self.format_process_output(result)
                }
            }
            Err(err) => format!("ERROR: {err}"),
        }
    }

    fn git_log(&self, count: usize) -> String {
        if let Some(err) = self.require_git_repo() {
            return err;
        }

        let safe_count = count.clamp(1, 100);
        let mut cmd = Command::new("git");
        cmd.arg("-C")
            .arg(&self.workspace_root)
            .arg("log")
            .arg("--oneline")
            .arg("--decorate")
            .arg("-n")
            .arg(safe_count.to_string())
            .arg("--")
            .arg(".");

        match self.run_process(&mut cmd, None) {
            Ok(result) => self.format_process_output(result),
            Err(err) => format!("ERROR: {err}"),
        }
    }

    fn git_commit_plan(&self) -> String {
        if let Some(err) = self.require_git_repo() {
            return err;
        }

        let mut cmd = Command::new("git");
        cmd.arg("-C")
            .arg(&self.workspace_root)
            .arg("status")
            .arg("--porcelain=1")
            .arg("--")
            .arg(".");

        let result = match self.run_process(&mut cmd, None) {
            Ok(value) => value,
            Err(err) => return format!("ERROR: {err}"),
        };

        if result.exit_code != 0 {
            return self.format_process_output(result);
        }

        let mut staged = Vec::new();
        let mut unstaged = Vec::new();
        let mut untracked = Vec::new();

        for line in result.stdout.lines() {
            if line.len() < 4 {
                continue;
            }
            let status = &line[..2];
            let path = line[3..].to_string();

            if status == "??" {
                untracked.push(path);
                continue;
            }
            let mut chars = status.chars();
            let left = chars.next().unwrap_or(' ');
            let right = chars.next().unwrap_or(' ');
            if left != ' ' && left != '?' {
                staged.push(path.clone());
            }
            if right != ' ' {
                unstaged.push(path);
            }
        }

        let mut changed_paths = staged.clone();
        changed_paths.extend(unstaged.clone());
        changed_paths.extend(untracked.clone());
        changed_paths.sort();
        changed_paths.dedup();

        let payload = json!({
            "staged": staged,
            "unstaged": unstaged,
            "untracked": untracked,
            "changed_paths": changed_paths.clone(),
            "suggested_commit_message": self.suggest_commit_message(&changed_paths),
        });
        payload.to_string()
    }

    fn suggest_commit_message(&self, changed_paths: &[String]) -> String {
        if changed_paths.is_empty() {
            return "chore: no pending changes".to_string();
        }

        let mut scopes = changed_paths
            .iter()
            .map(|path| path.split('/').next().unwrap_or(path).to_string())
            .collect::<Vec<_>>();
        scopes.sort();
        scopes.dedup();

        let mut scope_summary = scopes
            .iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        if scopes.len() > 3 {
            scope_summary.push_str(", ...");
        }

        let prefix = if changed_paths.iter().any(|path| path.starts_with("tests/")) {
            "test"
        } else if changed_paths.iter().any(|path| path.starts_with("src/")) {
            "feat"
        } else {
            "chore"
        };

        format!(
            "{prefix}: update {} files ({scope_summary})",
            changed_paths.len()
        )
    }

    fn git_commit(&self, message: &str, add_all: bool) -> String {
        if message.trim().is_empty() {
            return "ERROR: commit message cannot be empty".to_string();
        }

        if let Some(err) = self.require_git_repo() {
            return err;
        }

        if add_all {
            let mut add_cmd = Command::new("git");
            add_cmd
                .arg("-C")
                .arg(&self.workspace_root)
                .arg("add")
                .arg("-A")
                .arg("--")
                .arg(".");
            let add_result = match self.run_process(&mut add_cmd, None) {
                Ok(value) => value,
                Err(err) => return format!("ERROR: {err}"),
            };
            if add_result.exit_code != 0 {
                return format!(
                    "ERROR: git add failed\n{}",
                    self.format_process_output(add_result)
                );
            }
        }

        if !self.confirm_prompt(&format!("Approve git commit? [y/N]\nmessage: {message}")) {
            return "DENIED: git commit not approved by user".to_string();
        }

        let mut commit_cmd = Command::new("git");
        commit_cmd
            .arg("-C")
            .arg(&self.workspace_root)
            .arg("commit")
            .arg("-m")
            .arg(message)
            .arg("--")
            .arg(".");

        match self.run_process(&mut commit_cmd, None) {
            Ok(result) => self.format_process_output(result),
            Err(err) => format!("ERROR: {err}"),
        }
    }

    fn apply_patch(&self, patch: &str) -> String {
        if patch.trim().is_empty() {
            return "ERROR: patch is empty".to_string();
        }

        if let Some(err) = self.require_git_repo() {
            return err;
        }

        if let Some(bad_path) = self.find_unsafe_patch_path(patch) {
            return format!("ERROR: unsafe patch path '{bad_path}'");
        }

        let mut apply_cmd = Command::new("git");
        apply_cmd
            .arg("-C")
            .arg(&self.workspace_root)
            .arg("apply")
            .arg("--whitespace=nowarn")
            .arg("-");

        let apply_result = match self.run_process(&mut apply_cmd, Some(patch.to_string())) {
            Ok(value) => value,
            Err(err) => return format!("ERROR: {err}"),
        };

        if apply_result.exit_code != 0 {
            return format!(
                "ERROR: git apply failed\n{}",
                self.format_process_output(apply_result)
            );
        }

        let mut status_cmd = Command::new("git");
        status_cmd
            .arg("-C")
            .arg(&self.workspace_root)
            .arg("status")
            .arg("--short");

        let status_result = match self.run_process(&mut status_cmd, None) {
            Ok(value) => value,
            Err(err) => return format!("ERROR: {err}"),
        };

        let mut output = status_result.stdout.trim().to_string();
        if output.is_empty() {
            output = "<clean>".to_string();
        }
        if output.len() > 4000 {
            output.truncate(4000);
            output.push_str("\n... status truncated ...");
        }

        format!("PATCH APPLIED\nGIT_STATUS:\n{output}")
    }

    fn memory_get(&self, key: Option<&str>) -> String {
        let data = self.load_memory();
        match key {
            Some(key) => match data.get(key) {
                Some(value) => json!({"key": key, "value": value}).to_string(),
                None => format!("NOT_FOUND: memory key '{key}'"),
            },
            None => Value::Object(data).to_string(),
        }
    }

    fn memory_set(&self, key: &str, value: Value) -> String {
        if key.trim().is_empty() {
            return "ERROR: memory key cannot be empty".to_string();
        }

        let mut data = self.load_memory();
        data.insert(key.to_string(), value);
        match self.save_memory(&data) {
            Ok(_) => format!("MEMORY_SET: {key}"),
            Err(err) => format!("ERROR: failed saving memory: {err}"),
        }
    }

    fn memory_delete(&self, key: &str) -> String {
        let mut data = self.load_memory();
        if data.remove(key).is_none() {
            return format!("NOT_FOUND: memory key '{key}'");
        }

        match self.save_memory(&data) {
            Ok(_) => format!("MEMORY_DELETED: {key}"),
            Err(err) => format!("ERROR: failed saving memory: {err}"),
        }
    }

    fn memory_search(&self, query: &str, limit: usize) -> String {
        if query.trim().is_empty() {
            return "ERROR: query cannot be empty".to_string();
        }

        let lowered = query.to_ascii_lowercase();
        let safe_limit = limit.clamp(1, 100);
        let data = self.load_memory();
        let mut matches = Vec::new();

        for (key, value) in data {
            let haystack = format!("{key}\n{}", value).to_ascii_lowercase();
            if haystack.contains(&lowered) {
                matches.push(json!({"key": key, "value": value}));
                if matches.len() >= safe_limit {
                    break;
                }
            }
        }

        Value::Array(matches).to_string()
    }

    fn load_memory(&self) -> Map<String, Value> {
        if !self.memory_file.exists() {
            return Map::new();
        }

        let text = match fs::read_to_string(&self.memory_file) {
            Ok(value) => value,
            Err(_) => return Map::new(),
        };

        match serde_json::from_str::<Value>(&text) {
            Ok(Value::Object(map)) => map,
            _ => Map::new(),
        }
    }

    fn save_memory(&self, data: &Map<String, Value>) -> io::Result<()> {
        if let Some(parent) = self.memory_file.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(
            &self.memory_file,
            serde_json::to_string_pretty(data).unwrap_or_else(|_| "{}".to_string()),
        )
    }

    fn parallel_tools(
        &self,
        calls: Option<&Value>,
        max_workers: Option<&Value>,
        model_routing: &Value,
    ) -> String {
        let Some(calls) = calls.and_then(Value::as_array) else {
            return "ERROR: calls must be a non-empty list".to_string();
        };
        if calls.is_empty() {
            return "ERROR: calls must be a non-empty list".to_string();
        }

        let worker_count = max_workers
            .and_then(Value::as_i64)
            .map(|v| v.clamp(1, 8) as usize)
            .unwrap_or(4);

        let mut normalized = Vec::new();
        for (index, call) in calls.iter().enumerate() {
            let Some(obj) = call.as_object() else {
                return format!("ERROR: calls[{index}] must be an object");
            };
            let Some(tool) = obj.get("tool").and_then(Value::as_str) else {
                return format!("ERROR: calls[{index}].tool must be a string");
            };
            if tool == "parallel_tools" {
                return "ERROR: parallel_tools cannot call itself".to_string();
            }
            if tool == "reload_plugins" {
                return "ERROR: parallel_tools cannot call reload_plugins".to_string();
            }
            let args = obj.get("args").cloned().unwrap_or_else(|| json!({}));
            if !args.is_object() {
                return format!("ERROR: calls[{index}].args must be an object");
            }
            normalized.push((index, tool.to_string(), args));
        }

        // Interactive mode should avoid concurrent permission prompts.
        if !self.auto_mode {
            let mut results = Vec::new();
            for (index, tool, args) in normalized {
                let result = self.execute_readonly(&tool, &args, model_routing);
                results.push(json!({
                    "index": index,
                    "tool": tool,
                    "args": args,
                    "result": result,
                }));
            }
            return json!({"parallel_results": results}).to_string();
        }

        let mut results = Vec::new();
        let bounded_workers = worker_count.max(1).min(normalized.len().max(1));
        for chunk in normalized.chunks(bounded_workers) {
            let mut chunk_handles = Vec::new();
            for (index, tool, args) in chunk.iter().cloned() {
                let this = self.clone();
                let routing = model_routing.clone();
                chunk_handles.push(thread::spawn(move || {
                    let result = this.execute_readonly(&tool, &args, &routing);
                    json!({
                        "index": index,
                        "tool": tool,
                        "args": args,
                        "result": result,
                    })
                }));
            }

            for handle in chunk_handles {
                match handle.join() {
                    Ok(value) => results.push(value),
                    Err(_) => results.push(json!({
                        "index": -1,
                        "tool": "<thread>" ,
                        "args": {},
                        "result": "ERROR: parallel call panicked",
                    })),
                }
            }
        }

        results.sort_by_key(|item| {
            item.get("index")
                .and_then(Value::as_i64)
                .unwrap_or(i64::MAX)
        });
        json!({"parallel_results": results}).to_string()
    }

    fn load_plugins(&self) -> HashMap<String, PluginSpec> {
        let mut plugins = HashMap::new();
        if !self.plugins_dir.exists() || !self.plugins_dir.is_dir() {
            return plugins;
        }

        let name_regex = Regex::new(r"^[A-Za-z0-9_-]{1,64}$").unwrap();

        let entries = match fs::read_dir(&self.plugins_dir) {
            Ok(value) => value,
            Err(_) => return plugins,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|v| v.to_str()) != Some("json") {
                continue;
            }

            let text = match fs::read_to_string(&path) {
                Ok(value) => value,
                Err(_) => continue,
            };
            let spec: PluginSpec = match serde_json::from_str(&text) {
                Ok(value) => value,
                Err(_) => continue,
            };

            if !name_regex.is_match(&spec.name) {
                continue;
            }
            if spec.description.trim().is_empty() {
                continue;
            }
            if spec.command.is_empty() || spec.command.iter().any(|v| v.trim().is_empty()) {
                continue;
            }

            plugins.insert(spec.name.clone(), spec);
        }

        plugins
    }

    fn run_plugin_tool(&self, plugin_name: &str, args: &Map<String, Value>) -> String {
        let Some(plugin) = self.plugins.get(plugin_name) else {
            return format!("ERROR: plugin tool not found: plugin.{plugin_name}");
        };

        if !self.confirm_prompt(&format!(
            "Approve plugin command? [y/N]\nplugin: {plugin_name}\ncommand: {}",
            plugin.command.join(" ")
        )) {
            return "DENIED: plugin command not approved by user".to_string();
        }

        let mut cmd = Command::new(&plugin.command[0]);
        if plugin.command.len() > 1 {
            cmd.args(&plugin.command[1..]);
        }
        cmd.current_dir(&self.workspace_root);
        cmd.env("LOCAL_CODEX_WORKSPACE", &self.workspace_root);

        let payload = Value::Object(args.clone()).to_string();
        match self.run_process(&mut cmd, Some(payload)) {
            Ok(result) => format!(
                "PLUGIN: {plugin_name}\n{}",
                self.format_process_output(result)
            ),
            Err(err) => format!("ERROR: {err}"),
        }
    }

    fn require_git_repo(&self) -> Option<String> {
        let mut cmd = Command::new("git");
        cmd.arg("-C")
            .arg(&self.workspace_root)
            .arg("rev-parse")
            .arg("--is-inside-work-tree");
        let result = self.run_process(&mut cmd, None).ok()?;
        if result.exit_code == 0 && result.stdout.trim() == "true" {
            None
        } else {
            Some("ERROR: workspace is not a git repository".to_string())
        }
    }

    fn find_unsafe_patch_path(&self, patch: &str) -> Option<String> {
        for line in patch.lines() {
            if let Some(rest) = line.strip_prefix("diff --git ") {
                let parts: Vec<&str> = rest.split_whitespace().collect();
                if parts.len() >= 2 {
                    for raw in [parts[0], parts[1]] {
                        if let Some(normalized) = self.normalize_patch_path(raw) {
                            if self.is_unsafe_patch_path(&normalized) {
                                return Some(normalized);
                            }
                        }
                    }
                }
                continue;
            }

            let candidate = if let Some(value) = line.strip_prefix("--- ") {
                Some(value.split('\t').next().unwrap_or(value).trim())
            } else if let Some(value) = line.strip_prefix("+++ ") {
                Some(value.split('\t').next().unwrap_or(value).trim())
            } else if let Some(value) = line.strip_prefix("rename from ") {
                Some(value.trim())
            } else if let Some(value) = line.strip_prefix("rename to ") {
                Some(value.trim())
            } else {
                None
            };

            if let Some(candidate) = candidate {
                if let Some(normalized) = self.normalize_patch_path(candidate) {
                    if self.is_unsafe_patch_path(&normalized) {
                        return Some(normalized);
                    }
                }
            }
        }

        None
    }

    fn normalize_patch_path(&self, raw: &str) -> Option<String> {
        let mut value = raw.trim().trim_matches('"').trim_matches('\'').to_string();
        if value.is_empty() {
            return None;
        }
        if value == "/dev/null" || value == "dev/null" {
            return None;
        }
        if let Some(stripped) = value.strip_prefix("a/") {
            value = stripped.to_string();
        } else if let Some(stripped) = value.strip_prefix("b/") {
            value = stripped.to_string();
        }
        if value == "dev/null" {
            return None;
        }
        Some(value)
    }

    fn is_unsafe_patch_path(&self, value: &str) -> bool {
        if value.contains('\\') || value.starts_with('/') {
            return true;
        }
        let path = Path::new(value);
        if path.components().any(|c| matches!(c, Component::ParentDir)) {
            return true;
        }
        if let Some(first) = path.components().next() {
            if let Component::Normal(part) = first {
                if part.to_string_lossy().contains(':') {
                    return true;
                }
            }
        }

        match self.resolve_existing_or_virtual(value) {
            Ok(_) => false,
            Err(_) => true,
        }
    }

    fn run_process(
        &self,
        command: &mut Command,
        stdin_payload: Option<String>,
    ) -> io::Result<ProcessResult> {
        command.stdout(Stdio::piped()).stderr(Stdio::piped());
        if stdin_payload.is_some() {
            command.stdin(Stdio::piped());
        }

        let mut child = command.spawn()?;

        if let Some(payload) = stdin_payload {
            if let Some(mut stdin) = child.stdin.take() {
                stdin.write_all(payload.as_bytes())?;
            }
        }

        let timeout = Duration::from_secs(self.shell_timeout_seconds.max(1));
        let status_opt = child.wait_timeout(timeout)?;

        if status_opt.is_none() {
            let _ = child.kill();
            let output = child.wait_with_output()?;
            return Ok(ProcessResult {
                exit_code: -1,
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                timed_out: true,
            });
        }

        let output = child.wait_with_output()?;
        Ok(ProcessResult {
            exit_code: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            timed_out: false,
        })
    }

    fn format_process_output(&self, result: ProcessResult) -> String {
        let mut stdout = result.stdout.trim().to_string();
        let mut stderr = result.stderr.trim().to_string();
        if stdout.len() > 12_000 {
            stdout.truncate(12_000);
            stdout.push_str("\n... stdout truncated ...");
        }
        if stderr.len() > 12_000 {
            stderr.truncate(12_000);
            stderr.push_str("\n... stderr truncated ...");
        }

        let exit_code = if result.timed_out {
            -1
        } else {
            result.exit_code
        };
        let timeout_note = if result.timed_out {
            "\nNOTE: process timed out"
        } else {
            ""
        };

        format!(
            "EXIT_CODE: {exit_code}\nSTDOUT:\n{}\nSTDERR:\n{}{}",
            if stdout.is_empty() {
                "<empty>"
            } else {
                stdout.as_str()
            },
            if stderr.is_empty() {
                "<empty>"
            } else {
                stderr.as_str()
            },
            timeout_note
        )
    }

    fn confirm_prompt(&self, prompt: &str) -> bool {
        if self.auto_mode {
            return true;
        }

        print!("{prompt}\n> ");
        let _ = io::stdout().flush();
        let mut line = String::new();
        match io::stdin().read_line(&mut line) {
            Ok(_) => {
                let value = line.trim().to_ascii_lowercase();
                value == "y" || value == "yes"
            }
            Err(_) => false,
        }
    }

    fn resolve_existing_path(&self, path: &str) -> Result<PathBuf, String> {
        let resolved = self.resolve_existing_or_virtual(path)?;
        if !resolved.exists() {
            return Err(format!("path does not exist: {path}"));
        }
        Ok(resolved)
    }

    fn resolve_existing_file(&self, path: &str) -> Result<PathBuf, String> {
        let resolved = self.resolve_existing_path(path)?;
        if !resolved.is_file() {
            return Err(format!("file not found: {path}"));
        }
        Ok(resolved)
    }

    fn resolve_writable_path(&self, path: &str) -> Result<PathBuf, String> {
        self.resolve_existing_or_virtual(path)
    }

    fn resolve_existing_or_virtual(&self, path: &str) -> Result<PathBuf, String> {
        let rel = Path::new(path);
        if rel.is_absolute() {
            return Err("path escapes workspace root".to_string());
        }
        if rel
            .components()
            .any(|component| matches!(component, Component::ParentDir))
        {
            return Err("path escapes workspace root".to_string());
        }

        let candidate = self.workspace_root.join(rel);
        if candidate.exists() {
            let canonical = candidate
                .canonicalize()
                .map_err(|err| format!("failed to canonicalize path: {err}"))?;
            if !canonical.starts_with(&self.workspace_root) {
                return Err("path escapes workspace root".to_string());
            }
            return Ok(canonical);
        }

        let parent = candidate
            .parent()
            .ok_or_else(|| "path escapes workspace root".to_string())?;
        let canonical_parent = self.canonicalize_nearest_existing(parent)?;
        if !canonical_parent.starts_with(&self.workspace_root) {
            return Err("path escapes workspace root".to_string());
        }

        Ok(candidate)
    }

    fn canonicalize_nearest_existing(&self, path: &Path) -> Result<PathBuf, String> {
        let mut cursor = path.to_path_buf();
        loop {
            if cursor.exists() {
                return cursor
                    .canonicalize()
                    .map_err(|err| format!("failed to canonicalize path: {err}"));
            }
            let Some(parent) = cursor.parent() else {
                return Err("path escapes workspace root".to_string());
            };
            cursor = parent.to_path_buf();
        }
    }

    fn to_workspace_rel(&self, path: &Path) -> String {
        self.to_workspace_rel_path(path)
            .to_string_lossy()
            .to_string()
    }

    fn to_workspace_rel_path(&self, path: &Path) -> PathBuf {
        match path.strip_prefix(&self.workspace_root) {
            Ok(stripped) if stripped.as_os_str().is_empty() => PathBuf::from("."),
            Ok(stripped) => stripped.to_path_buf(),
            Err(_) => PathBuf::from("."),
        }
    }
}

fn tool_spec(name: &str, description: &str, args: &[&str]) -> ToolSpec {
    let mut parsed = HashMap::new();
    for arg in args {
        if let Some((k, v)) = arg.split_once(':') {
            parsed.insert(k.to_string(), v.to_string());
        }
    }
    ToolSpec {
        name: name.to_string(),
        description: description.to_string(),
        args: parsed,
    }
}
