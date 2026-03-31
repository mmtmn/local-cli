use anyhow::Result;
use local_codex::agent::Orchestrator;
use local_codex::config::ModelRouting;
use local_codex::llm::{ChatMessage, LlmClient};
use local_codex::tools::ToolExecutor;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

#[derive(Clone)]
struct StubLlm {
    responses: Arc<Mutex<VecDeque<String>>>,
}

impl StubLlm {
    fn new(responses: Vec<&str>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(
                responses
                    .into_iter()
                    .map(|value| value.to_string())
                    .collect::<VecDeque<_>>(),
            )),
        }
    }
}

impl LlmClient for StubLlm {
    fn chat(&self, _model: &str, _messages: &[ChatMessage], _temperature: f32) -> Result<String> {
        let mut lock = self.responses.lock().expect("lock poisoned");
        Ok(lock
            .pop_front()
            .unwrap_or_else(|| "{\"type\":\"final\",\"message\":\"stub done\"}".to_string()))
    }
}

fn build_orchestrator(tmp: &TempDir, llm: StubLlm) -> Orchestrator {
    let workspace = tmp.path().to_path_buf();
    let memory_file = workspace.join(".local_codex").join("memory.json");
    let plugins_dir = workspace.join(".local_codex").join("plugins");

    let tools = ToolExecutor::new(workspace.clone(), true, 30, memory_file, plugins_dir);

    let routing = ModelRouting {
        text_model: "qwen3:8b".to_string(),
        planner_model: None,
        coder_model: None,
        image_model: None,
    };

    Orchestrator::new(
        Box::new(llm),
        tools,
        workspace.display().to_string(),
        6,
        0.1,
        2,
        routing,
        false,
    )
}

#[test]
fn parses_embedded_json_action() {
    let tmp = TempDir::new().expect("failed creating temp dir");
    let llm = StubLlm::new(vec![
        "I will do this now. {\"type\":\"plan\",\"needs_plan\":false,\"summary\":\"finish fast\",\"reason\":\"simple request\",\"steps\":[\"answer directly\"]}",
        "{\"type\":\"final\",\"message\":\"done\",\"completed\":[\"answered the request\"],\"verified\":[\"response was produced\"],\"missed\":[]}",
    ]);

    let mut orchestrator = build_orchestrator(&tmp, llm);
    let result = orchestrator.run_active("do it");
    assert_eq!(result, "done");

    let summary = orchestrator
        .last_run_summary()
        .expect("missing last run summary");
    assert_eq!(summary.agent_name, "main");
    assert_eq!(
        summary
            .plan
            .as_ref()
            .expect("missing recorded plan")
            .summary,
        "finish fast"
    );
}

#[test]
fn executes_parallel_action_and_records_history() {
    let tmp = TempDir::new().expect("failed creating temp dir");
    std::fs::write(tmp.path().join("a.txt"), "hello\n").expect("failed writing a.txt");

    let llm = StubLlm::new(vec![
        "{\"type\":\"plan\",\"needs_plan\":true,\"summary\":\"inspect files\",\"reason\":\"multiple reads\",\"steps\":[\"read the file\",\"list the tree\"]}",
        "```json\n{\"type\":\"parallel_tools\",\"calls\":[{\"tool\":\"read_file\",\"args\":{\"path\":\"a.txt\"}},{\"tool\":\"list_files\",\"args\":{\"path\":\".\",\"max_depth\":1}}]}\n```",
        "{\"type\":\"final\",\"message\":\"done\",\"completed\":[\"read the file\",\"listed the tree\"],\"verified\":[\"tool history recorded\"],\"missed\":[]}",
    ]);

    let mut orchestrator = build_orchestrator(&tmp, llm);
    let result = orchestrator.run_active("inspect files in parallel");
    assert_eq!(result, "done");

    let snapshot = orchestrator
        .agent_snapshot("main")
        .expect("missing main snapshot");
    assert_eq!(snapshot.tool_history.len(), 1);
    assert_eq!(snapshot.tool_history[0]["tool"], "parallel_tools");
}

#[test]
fn delegate_named_agent_and_session_round_trip() {
    let tmp = TempDir::new().expect("failed creating temp dir");
    let llm = StubLlm::new(vec![
        "{\"type\":\"plan\",\"needs_plan\":true,\"summary\":\"use delegation\",\"reason\":\"split the task\",\"steps\":[\"delegate subtask\",\"combine result\"]}",
        "{\"type\":\"delegate\",\"agent_name\":\"research\",\"prompt\":\"do subtask\",\"max_steps\":2}",
        "{\"type\":\"plan\",\"needs_plan\":false,\"summary\":\"handle delegated subtask\",\"reason\":\"small task\",\"steps\":[\"complete subtask\"]}",
        "{\"type\":\"final\",\"message\":\"subtask done\",\"completed\":[\"finished the subtask\"],\"verified\":[\"delegate responded\"],\"missed\":[]}",
        "{\"type\":\"final\",\"message\":\"all done\",\"completed\":[\"delegated work\",\"combined result\"],\"verified\":[\"subtask finished\"],\"missed\":[]}",
    ]);

    let mut orchestrator = build_orchestrator(&tmp, llm);
    let result = orchestrator.run_active("solve this with delegation");
    assert_eq!(result, "all done");
    assert!(orchestrator.has_agent("research"));

    let parent = orchestrator
        .agent_snapshot("main")
        .expect("missing main snapshot");
    assert_eq!(parent.delegate_history.len(), 1);
    assert_eq!(parent.delegate_history[0]["agent_name"], "research");

    let session_file = tmp.path().join("session.json");
    orchestrator
        .save_session(&session_file)
        .expect("failed saving session");

    let llm_after_load = StubLlm::new(vec!["{\"type\":\"final\",\"message\":\"ok\"}"]);
    let mut restored = build_orchestrator(&tmp, llm_after_load);
    restored
        .load_session(&session_file)
        .expect("failed loading session");

    assert!(restored.has_agent("research"));
    assert_eq!(restored.active_agent(), "main");

    let restored_parent = restored
        .agent_snapshot("main")
        .expect("missing restored main snapshot");
    assert_eq!(restored_parent.delegate_history.len(), 1);
}
