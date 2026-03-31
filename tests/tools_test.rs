use local_codex::tools::ToolExecutor;
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;

fn run_checked(args: &[&str], cwd: &Path) {
    let output = Command::new(args[0])
        .args(&args[1..])
        .current_dir(cwd)
        .output()
        .expect("failed to run command");
    assert!(
        output.status.success(),
        "command failed: {:?}\nstdout:\n{}\nstderr:\n{}",
        args,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn routing_json() -> serde_json::Value {
    json!({
        "text_model": "qwen3:8b",
        "planner_model": null,
        "coder_model": null,
        "image_model": null,
    })
}

struct Harness {
    _tmp: TempDir,
    workspace: PathBuf,
    executor: ToolExecutor,
}

impl Harness {
    fn new() -> Self {
        let tmp = TempDir::new().expect("failed creating temp dir");
        let workspace = tmp.path().to_path_buf();

        run_checked(&["git", "init", "-b", "main"], &workspace);
        run_checked(
            &["git", "config", "user.email", "test@example.com"],
            &workspace,
        );
        run_checked(&["git", "config", "user.name", "Test User"], &workspace);

        fs::write(workspace.join("hello.txt"), "one\n").expect("failed writing hello.txt");
        run_checked(&["git", "add", "hello.txt"], &workspace);
        run_checked(&["git", "commit", "-m", "init"], &workspace);

        let plugins_dir = workspace.join(".local_codex").join("plugins");
        fs::create_dir_all(&plugins_dir).expect("failed creating plugins dir");
        let plugin_spec = json!({
            "name": "echo",
            "description": "Echo input value",
            "command": [
                "python3",
                "-c",
                "import json,sys; payload=json.load(sys.stdin); print('ECHO:' + str(payload.get('value', '')))"
            ],
            "args_schema": {"value": "text to echo"}
        });
        fs::write(
            plugins_dir.join("echo.json"),
            serde_json::to_string(&plugin_spec).expect("failed serializing plugin"),
        )
        .expect("failed writing plugin spec");

        let memory_file = workspace.join(".local_codex").join("memory.json");
        let executor = ToolExecutor::new(workspace.clone(), true, 30, memory_file, plugins_dir);

        Self {
            _tmp: tmp,
            workspace,
            executor,
        }
    }
}

#[test]
fn read_file_blocks_workspace_escape() {
    let mut harness = Harness::new();
    let result = harness.executor.execute(
        "read_file",
        &json!({"path": "../secret.txt"}),
        &routing_json(),
    );
    assert!(result.contains("path escapes workspace root"));
}

#[test]
fn write_file_blocks_workspace_escape() {
    let mut harness = Harness::new();
    let result = harness.executor.execute(
        "write_file",
        &json!({"path": "../secret.txt", "content": "x"}),
        &routing_json(),
    );
    assert!(result.contains("path escapes workspace root"));
}

#[test]
fn apply_patch_rejects_unsafe_path() {
    let mut harness = Harness::new();
    let patch = [
        "diff --git a/../evil.txt b/../evil.txt",
        "--- a/../evil.txt",
        "+++ b/../evil.txt",
        "@@ -0,0 +1 @@",
        "+pwnd",
    ]
    .join("\n");

    let result = harness
        .executor
        .execute("apply_patch", &json!({"patch": patch}), &routing_json());
    assert!(result.contains("unsafe patch path"));
}

#[test]
fn git_tools_and_patch_workflow() {
    let mut harness = Harness::new();
    let patch = [
        "diff --git a/hello.txt b/hello.txt",
        "--- a/hello.txt",
        "+++ b/hello.txt",
        "@@ -1 +1 @@",
        "-one",
        "+two",
        "",
    ]
    .join("\n");

    let apply_result =
        harness
            .executor
            .execute("apply_patch", &json!({"patch": patch}), &routing_json());
    assert!(apply_result.contains("PATCH APPLIED"));

    let file_value = fs::read_to_string(harness.workspace.join("hello.txt")).expect("failed read");
    assert_eq!(file_value, "two\n");

    let diff_result =
        harness
            .executor
            .execute("git_diff", &json!({"path": "hello.txt"}), &routing_json());
    assert!(diff_result.contains("+two"));

    let status_result = harness
        .executor
        .execute("git_status", &json!({}), &routing_json());
    assert!(status_result.contains("hello.txt"));

    let plan_result = harness
        .executor
        .execute("git_commit_plan", &json!({}), &routing_json());
    assert!(plan_result.contains("suggested_commit_message"));

    let commit_result = harness.executor.execute(
        "git_commit",
        &json!({"message": "test: update hello", "add_all": true}),
        &routing_json(),
    );
    assert!(commit_result.contains("EXIT_CODE: 0"));
}

#[test]
fn git_tools_respect_workspace_scope() {
    let tmp = TempDir::new().expect("failed creating temp dir");
    let repo = tmp.path().join("repo");
    let workspace = repo.join("app");
    fs::create_dir_all(&workspace).expect("failed creating workspace");

    run_checked(&["git", "init", "-b", "main"], &repo);
    run_checked(&["git", "config", "user.email", "test@example.com"], &repo);
    run_checked(&["git", "config", "user.name", "Test User"], &repo);

    fs::write(repo.join("root.txt"), "root\n").expect("failed writing root.txt");
    fs::write(workspace.join("local.txt"), "local\n").expect("failed writing local.txt");
    run_checked(&["git", "add", "."], &repo);
    run_checked(&["git", "commit", "-m", "init"], &repo);

    fs::write(repo.join("root.txt"), "root changed\n").expect("failed updating root.txt");
    fs::write(workspace.join("local.txt"), "local changed\n").expect("failed updating local.txt");

    let plugins_dir = workspace.join(".local_codex").join("plugins");
    fs::create_dir_all(&plugins_dir).expect("failed creating plugins dir");
    let memory_file = workspace.join(".local_codex").join("memory.json");
    let mut executor = ToolExecutor::new(workspace.clone(), true, 30, memory_file, plugins_dir);

    let status = executor.execute("git_status", &json!({}), &routing_json());
    assert!(status.contains("local.txt"));
    assert!(!status.contains("root.txt"));

    let diff = executor.execute("git_diff", &json!({}), &routing_json());
    assert!(diff.contains("local changed"));
    assert!(!diff.contains("root changed"));

    let plan = executor.execute("git_commit_plan", &json!({}), &routing_json());
    assert!(plan.contains("local.txt"));
    assert!(!plan.contains("root.txt"));
}

#[test]
fn memory_lifecycle() {
    let mut harness = Harness::new();

    let set_result = harness.executor.execute(
        "memory_set",
        &json!({"key": "project_goal", "value": "build local codex"}),
        &routing_json(),
    );
    assert!(set_result.contains("MEMORY_SET"));

    let get_result = harness.executor.execute(
        "memory_get",
        &json!({"key": "project_goal"}),
        &routing_json(),
    );
    assert!(get_result.contains("build local codex"));

    let search_result =
        harness
            .executor
            .execute("memory_search", &json!({"query": "codex"}), &routing_json());
    assert!(search_result.contains("project_goal"));

    let delete_result = harness.executor.execute(
        "memory_delete",
        &json!({"key": "project_goal"}),
        &routing_json(),
    );
    assert!(delete_result.contains("MEMORY_DELETED"));
}

#[test]
fn structured_patch_and_symbol_aware_edits() {
    let mut harness = Harness::new();
    let source_file = harness.workspace.join("sample.py");
    fs::write(
        &source_file,
        "def greet():\n    return 'hi'\n\nclass A:\n    pass\n",
    )
    .expect("failed writing sample.py");

    let overview = harness.executor.execute(
        "python_symbol_overview",
        &json!({"path": "sample.py"}),
        &routing_json(),
    );
    assert!(overview.contains("\"name\":\"greet\"") || overview.contains("\"name\": \"greet\""));
    assert!(overview.contains("\"name\":\"A\"") || overview.contains("\"name\": \"A\""));

    let replace_symbol = harness.executor.execute(
        "replace_python_symbol",
        &json!({
            "path": "sample.py",
            "symbol": "greet",
            "symbol_type": "function",
            "code": "def greet():\n    return 'hello'"
        }),
        &routing_json(),
    );
    assert!(replace_symbol.contains("WROTE:") || replace_symbol.contains("REPLACED_SYMBOL"));

    let updated_after_symbol = fs::read_to_string(&source_file).expect("failed reading sample.py");
    assert!(updated_after_symbol.contains("return 'hello'"));

    let structured = harness.executor.execute(
        "structured_patch",
        &json!({
            "operations": [
                {
                    "type": "replace_text",
                    "path": "sample.py",
                    "old": "class A",
                    "new": "class B"
                },
                {
                    "type": "append_text",
                    "path": "sample.py",
                    "text": "\n# end\n"
                }
            ]
        }),
        &routing_json(),
    );
    assert!(structured.contains("replace_text"));

    let updated = fs::read_to_string(&source_file).expect("failed reading sample.py");
    assert!(updated.contains("class B"));
    assert!(updated.contains("# end"));
}

#[test]
fn plugin_and_parallel_tools() {
    let mut harness = Harness::new();

    let specs = harness.executor.list_specs();
    let tool_names = specs
        .iter()
        .map(|spec| spec.name.as_str())
        .collect::<Vec<_>>();
    assert!(tool_names.contains(&"plugin.echo"));

    let plugin_result =
        harness
            .executor
            .execute("plugin.echo", &json!({"value": "ok"}), &routing_json());
    assert!(plugin_result.contains("ECHO:ok"));

    let parallel_result = harness.executor.execute(
        "parallel_tools",
        &json!({
            "calls": [
                {"tool": "read_file", "args": {"path": "hello.txt"}},
                {"tool": "memory_set", "args": {"key": "k", "value": "v"}}
            ],
            "max_workers": 2
        }),
        &routing_json(),
    );
    assert!(parallel_result.contains("parallel_results"));
    assert!(
        parallel_result.contains("\"tool\":\"read_file\"")
            || parallel_result.contains("\"tool\": \"read_file\"")
    );
}
