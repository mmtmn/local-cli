use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use local_codex::agent::Orchestrator;
use local_codex::config::{ModelRouting, Settings};
use local_codex::llm::{LlmClient, OllamaClient, OpenAiCompatibleClient};
use local_codex::tools::ToolExecutor;
use local_codex::ui::{self, Tone};
use std::io;
use std::path::{Path, PathBuf};

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Provider {
    Ollama,
    Openai,
}

#[derive(Parser, Debug)]
#[command(
    name = "local-cli",
    about = "Local open-source coding orchestrator for local/open models"
)]
struct Cli {
    #[arg(long, value_enum, default_value = "ollama")]
    provider: Provider,

    #[arg(
        long,
        default_value = "qwen3:8b",
        help = "Fallback base model; used as --text-model when --text-model is unset"
    )]
    model: String,

    #[arg(long, help = "Primary text/chat model")]
    text_model: Option<String>,

    #[arg(long, help = "Planning model (defaults to text model)")]
    planner_model: Option<String>,

    #[arg(long, help = "Coding/editing model (defaults to text model)")]
    coder_model: Option<String>,

    #[arg(long, help = "Image model identifier (for future image tools)")]
    image_model: Option<String>,

    #[arg(long)]
    endpoint: Option<String>,

    #[arg(long)]
    api_key: Option<String>,

    #[arg(long, default_value = ".")]
    workspace: PathBuf,

    #[arg(long, default_value_t = 12)]
    max_steps: u32,

    #[arg(long, default_value_t = 45)]
    shell_timeout: u64,

    #[arg(long, default_value_t = 0.1)]
    temperature: f32,

    #[arg(
        long,
        alias = "auto-approve",
        help = "Force auto mode on (auto mode is enabled by default)"
    )]
    auto_mode: bool,

    #[arg(long, help = "Require approval prompts (disables auto mode)")]
    ask_permissions: bool,

    #[arg(long)]
    memory_file: Option<PathBuf>,

    #[arg(long)]
    plugins_dir: Option<PathBuf>,

    #[arg(long, default_value_t = 2)]
    max_delegation_depth: u32,

    #[arg(long)]
    session_file: Option<PathBuf>,

    #[arg(long)]
    fresh_session: bool,

    #[arg(long, help = "Run one prompt and exit")]
    prompt: Option<String>,

    #[arg(long, help = "Hide [tool] execution logs")]
    no_tool_logs: bool,
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum ReplControl {
    Continue,
    Exit,
}

struct RuntimeSettings {
    settings: Settings,
    prompt: Option<String>,
    fresh_session: bool,
    session_file: Option<PathBuf>,
    show_tool_logs: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut runtime = build_settings(cli)?;
    let workspace = runtime.settings.workspace_resolved();

    let tools = ToolExecutor::new(
        workspace.clone(),
        runtime.settings.auto_mode,
        runtime.settings.shell_timeout_seconds,
        runtime.settings.memory_file_resolved(),
        runtime.settings.plugins_dir_resolved(),
    );

    let llm = build_llm(&runtime.settings)?;

    let mut orchestrator = Orchestrator::new(
        llm,
        tools,
        workspace.display().to_string(),
        runtime.settings.max_steps,
        runtime.settings.temperature,
        runtime.settings.max_delegation_depth,
        runtime.settings.routing.clone(),
        runtime.show_tool_logs,
    );

    let session_file = runtime
        .session_file
        .as_ref()
        .map(|value| resolve_path(&workspace, value.clone()));

    if let Some(path) = session_file.as_ref() {
        if path.exists() && !runtime.fresh_session {
            match orchestrator.load_session(path) {
                Ok(loaded) => ui::print_notice(&format!("loaded session: {}", loaded.display())),
                Err(err) => {
                    ui::print_warning(&format!("failed to load session {}: {err}", path.display()))
                }
            }
        } else if path.exists() && runtime.fresh_session {
            ui::print_notice(&format!(
                "starting fresh session and ignoring existing file: {}",
                path.display()
            ));
        }
    }

    if let Some(prompt) = runtime.prompt.as_deref() {
        run_once(&mut orchestrator, &runtime, prompt, session_file.as_deref())?;
        return Ok(());
    }

    repl(&mut orchestrator, &mut runtime, session_file.as_deref())
}

fn build_settings(cli: Cli) -> Result<RuntimeSettings> {
    let endpoint = cli
        .endpoint
        .unwrap_or_else(|| default_endpoint(cli.provider).to_string());
    let api_key = cli.api_key.or_else(|| std::env::var("OPENAI_API_KEY").ok());

    let routing = ModelRouting {
        text_model: cli.text_model.unwrap_or(cli.model),
        planner_model: cli.planner_model,
        coder_model: cli.coder_model,
        image_model: cli.image_model,
    };

    let workspace = if cli.workspace.is_absolute() {
        cli.workspace
    } else {
        std::env::current_dir()
            .context("failed to resolve current directory")?
            .join(cli.workspace)
    };

    let auto_mode = cli.auto_mode || !cli.ask_permissions;

    let settings = Settings {
        provider: match cli.provider {
            Provider::Ollama => "ollama".to_string(),
            Provider::Openai => "openai".to_string(),
        },
        endpoint,
        api_key,
        workspace,
        routing,
        max_steps: cli.max_steps.max(1),
        shell_timeout_seconds: cli.shell_timeout.max(1),
        auto_mode,
        temperature: cli.temperature,
        memory_file: cli.memory_file,
        plugins_dir: cli.plugins_dir,
        max_delegation_depth: cli.max_delegation_depth,
    };

    Ok(RuntimeSettings {
        settings,
        prompt: cli.prompt,
        fresh_session: cli.fresh_session,
        session_file: cli.session_file,
        show_tool_logs: !cli.no_tool_logs,
    })
}

fn build_llm(settings: &Settings) -> Result<Box<dyn LlmClient>> {
    // Keep network timeout generous for local models under heavy load.
    let timeout_seconds = 600;

    match settings.provider.as_str() {
        "ollama" => {
            let client = OllamaClient::new(settings.endpoint.clone(), timeout_seconds)?;
            Ok(Box::new(client))
        }
        "openai" => {
            let client = OpenAiCompatibleClient::new(
                settings.endpoint.clone(),
                settings.api_key.clone(),
                timeout_seconds,
            )?;
            Ok(Box::new(client))
        }
        other => Err(anyhow!("unsupported provider: {other}")),
    }
}

fn run_once(
    orchestrator: &mut Orchestrator,
    runtime: &RuntimeSettings,
    prompt: &str,
    session_file: Option<&Path>,
) -> Result<()> {
    let response = orchestrator.run_active(prompt);
    render_response(orchestrator, &response);
    if let Some(path) = session_file {
        orchestrator.save_session(path)?;
    }
    let _ = runtime;
    Ok(())
}

fn repl(
    orchestrator: &mut Orchestrator,
    runtime: &mut RuntimeSettings,
    session_file: Option<&Path>,
) -> Result<()> {
    let workspace = runtime.settings.workspace_resolved();
    ui::print_welcome(
        &workspace,
        &runtime.settings.provider,
        &runtime.settings.routing,
        runtime.settings.auto_mode,
        session_file,
    );
    ui::print_help();

    loop {
        print!(
            "{}",
            ui::prompt(
                orchestrator.active_agent(),
                &runtime.settings.workspace_resolved(),
                &runtime.settings.routing,
                runtime.settings.auto_mode
            )
        );
        ui::flush_stdout();

        let mut line = String::new();
        let bytes = io::stdin().read_line(&mut line)?;
        if bytes == 0 {
            ui::print_notice("exiting");
            return Ok(());
        }

        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        if input.starts_with('/') {
            let action = handle_command(input, orchestrator, runtime, session_file)?;
            if action == ReplControl::Exit {
                return Ok(());
            }
            maybe_save_session(orchestrator, session_file);
            continue;
        }

        let response = orchestrator.run_active(input);
        render_response(orchestrator, &response);
        maybe_save_session(orchestrator, session_file);
    }
}

fn handle_command(
    input: &str,
    orchestrator: &mut Orchestrator,
    runtime: &mut RuntimeSettings,
    session_file: Option<&Path>,
) -> Result<ReplControl> {
    let (command, rest) = split_command(input);

    match command {
        "/help" => ui::print_help(),
        "/exit" | "/quit" => return Ok(ReplControl::Exit),
        "/reset" => {
            orchestrator.reset_active()?;
            ui::print_notice(&format!("reset agent '{}'", orchestrator.active_agent()));
        }
        "/save" => {
            let target = command_path_or_default(rest, session_file)?;
            let saved = orchestrator.save_session(target)?;
            ui::print_notice(&format!("session saved: {}", saved.display()));
        }
        "/load" => {
            let target = command_path_or_default(rest, session_file)?;
            let loaded = orchestrator.load_session(target)?;
            ui::print_notice(&format!("session loaded: {}", loaded.display()));
        }
        "/reload_plugins" => {
            let result = orchestrator.reload_plugins();
            ui::print_assistant_message(orchestrator.active_agent(), &result);
        }
        "/status" => {
            print_status(orchestrator, runtime, session_file);
        }
        "/plan" => {
            if let Some(plan) = orchestrator
                .last_run_summary()
                .and_then(|summary| summary.plan.as_ref())
            {
                ui::print_last_plan(plan);
            } else {
                ui::print_warning("no recorded plan for this session yet");
            }
        }
        "/scope" => {
            handle_scope_command(rest, orchestrator, runtime)?;
        }
        "/models" | "/routing" => {
            let payload = serde_json::to_string_pretty(&orchestrator.model_routing())
                .unwrap_or_else(|_| "{}".to_string());
            ui::print_panel("ROUTING", Tone::Info, &[payload]);
        }
        "/agents" => {
            print_agent_list(orchestrator);
        }
        "/agent" => {
            handle_agent_command(rest, orchestrator, session_file)?;
        }
        _ => {
            ui::print_warning(&format!("unknown command {command}"));
        }
    }

    Ok(ReplControl::Continue)
}

fn handle_agent_command(
    rest: &str,
    orchestrator: &mut Orchestrator,
    session_file: Option<&Path>,
) -> Result<()> {
    let trimmed = rest.trim();
    if trimmed.is_empty() || trimmed == "list" {
        print_agent_list(orchestrator);
        return Ok(());
    }

    if let Some(name) = trimmed.strip_prefix("new ") {
        orchestrator.create_agent(name.trim())?;
        ui::print_notice(&format!("created agent '{}'", name.trim()));
        return Ok(());
    }

    if let Some(name) = trimmed.strip_prefix("use ") {
        orchestrator.set_active_agent(name.trim())?;
        ui::print_notice(&format!("active agent: {}", orchestrator.active_agent()));
        return Ok(());
    }

    if let Some(name) = trimmed.strip_prefix("reset ") {
        orchestrator.reset_agent(name.trim())?;
        ui::print_notice(&format!("reset agent '{}'", name.trim()));
        return Ok(());
    }

    if let Some(run_part) = trimmed.strip_prefix("run ") {
        let mut split = run_part.trim().splitn(2, ' ');
        let name = split.next().unwrap_or("").trim();
        let prompt = split.next().unwrap_or("").trim();

        if name.is_empty() || prompt.is_empty() {
            ui::print_warning("usage: /agent run <name> <prompt>");
            return Ok(());
        }

        let response = orchestrator.run_with_agent(name, prompt)?;
        render_response(orchestrator, &response);
        maybe_save_session(orchestrator, session_file);
        return Ok(());
    }

    ui::print_warning("unknown /agent command");
    Ok(())
}

fn print_agent_list(orchestrator: &Orchestrator) {
    let active = orchestrator.active_agent().to_string();
    let names = orchestrator.list_agents();

    if names.is_empty() {
        ui::print_warning("no agents");
        return;
    }

    ui::print_agents(&active, &names);
}

fn maybe_save_session(orchestrator: &mut Orchestrator, session_file: Option<&Path>) {
    let Some(path) = session_file else {
        return;
    };

    if let Err(err) = orchestrator.save_session(path) {
        ui::print_warning(&format!("failed to save session {}: {err}", path.display()));
    }
}

fn render_response(orchestrator: &Orchestrator, fallback: &str) {
    if let Some(summary) = orchestrator.last_run_summary() {
        if let Some(handoff) = summary.handoff.as_ref() {
            ui::print_handoff(&summary.agent_name, handoff);
            return;
        }
    }

    ui::print_assistant_message(orchestrator.active_agent(), fallback);
}

fn print_status(
    orchestrator: &Orchestrator,
    runtime: &RuntimeSettings,
    session_file: Option<&Path>,
) {
    ui::print_status(
        &runtime.settings.workspace_resolved(),
        &runtime.settings.provider,
        &runtime.settings.routing,
        orchestrator.active_agent(),
        runtime.settings.auto_mode,
        session_file,
        orchestrator.last_run_summary(),
    );
}

fn handle_scope_command(
    rest: &str,
    orchestrator: &mut Orchestrator,
    runtime: &mut RuntimeSettings,
) -> Result<()> {
    let trimmed = rest.trim();
    if trimmed.is_empty() {
        ui::print_scope(&runtime.settings.workspace_resolved());
        return Ok(());
    }

    if let Some(path) = trimmed.strip_prefix("use ") {
        let current = runtime.settings.workspace_resolved();
        let target = resolve_scope_path(&current, path.trim())?;
        if !target.is_dir() {
            return Err(anyhow!(
                "scope target is not a directory: {}",
                target.display()
            ));
        }

        runtime.settings.workspace = target.clone();
        let new_workspace = runtime.settings.workspace_resolved();
        let updated = orchestrator.set_workspace(
            new_workspace.clone(),
            runtime.settings.memory_file_resolved(),
            runtime.settings.plugins_dir_resolved(),
        )?;
        ui::print_notice(&format!("scope changed to {}", updated.display()));
        ui::print_scope(&updated);
        return Ok(());
    }

    ui::print_warning("usage: /scope or /scope use <path>");
    Ok(())
}

fn split_command(input: &str) -> (&str, &str) {
    let mut pieces = input.splitn(2, ' ');
    let command = pieces.next().unwrap_or("");
    let rest = pieces.next().unwrap_or("");
    (command, rest)
}

fn command_path_or_default(rest: &str, session_file: Option<&Path>) -> Result<PathBuf> {
    if !rest.trim().is_empty() {
        return Ok(expand_user_path(rest.trim()));
    }

    if let Some(path) = session_file {
        return Ok(path.to_path_buf());
    }

    Err(anyhow!("no session path provided"))
}

fn expand_user_path(raw: &str) -> PathBuf {
    if raw == "~" {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home);
        }
    }

    if let Some(stripped) = raw.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(stripped);
        }
    }

    PathBuf::from(raw)
}

fn resolve_path(base: &Path, value: PathBuf) -> PathBuf {
    if value.is_absolute() {
        value
    } else {
        base.join(value)
    }
}

fn resolve_scope_path(base: &Path, raw: &str) -> Result<PathBuf> {
    let expanded = expand_user_path(raw);
    let candidate = if expanded.is_absolute() {
        expanded
    } else {
        base.join(expanded)
    };
    candidate
        .canonicalize()
        .with_context(|| format!("failed to resolve scope path {}", candidate.display()))
}

fn default_endpoint(provider: Provider) -> &'static str {
    match provider {
        Provider::Ollama => "http://127.0.0.1:11434/api/chat",
        Provider::Openai => "http://127.0.0.1:8000/v1/chat/completions",
    }
}
