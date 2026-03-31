use crate::agent::{RunSummary, TaskHandoff, TaskPlan};
use crate::config::ModelRouting;
use std::env;
use std::io::{self, IsTerminal, Write};
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const SPINNER_FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

#[derive(Clone, Copy)]
pub enum Tone {
    Accent,
    Info,
    Success,
    Warning,
    Danger,
    Muted,
}

pub struct Activity {
    live: bool,
    label: String,
    started_at: Instant,
    stop: Option<Arc<AtomicBool>>,
    handle: Option<JoinHandle<()>>,
}

impl Activity {
    pub fn finish(mut self, message: &str, tone: Tone) {
        let elapsed = format_elapsed(self.started_at.elapsed());
        if self.live {
            if let Some(stop) = self.stop.take() {
                stop.store(true, Ordering::SeqCst);
            }
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
            clear_live_line();
        }

        let suffix = if message.trim().is_empty() {
            elapsed
        } else {
            format!("{message} · {elapsed}")
        };
        print_event(&self.label, tone, &suffix);
    }
}

pub fn start_activity(label: &str, message: &str, tone: Tone) -> Activity {
    let started_at = Instant::now();
    if !supports_live_output() {
        return Activity {
            live: false,
            label: label.to_string(),
            started_at,
            stop: None,
            handle: None,
        };
    }

    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = Arc::clone(&stop);
    let label_text = label.to_string();
    let message_text = single_line(message);
    let label_for_thread = label_text.clone();
    let handle = thread::spawn(move || {
        let mut frame_index = 0usize;
        while !stop_thread.load(Ordering::SeqCst) {
            let frame = SPINNER_FRAMES[frame_index % SPINNER_FRAMES.len()];
            frame_index += 1;
            let elapsed = format_elapsed(started_at.elapsed());
            let line = format!(
                "{} {} {} {}",
                paint(frame, tone, true),
                paint(&label_for_thread, tone, true),
                paint(&message_text, Tone::Muted, false),
                paint(&elapsed, Tone::Muted, false)
            );
            print!("\r\x1b[2K{line}");
            let _ = io::stdout().flush();
            thread::sleep(Duration::from_millis(90));
        }
    });

    Activity {
        live: true,
        label: label_text,
        started_at,
        stop: Some(stop),
        handle: Some(handle),
    }
}

pub fn print_welcome(
    workspace: &Path,
    provider: &str,
    routing: &ModelRouting,
    auto_mode: bool,
    session_file: Option<&Path>,
) {
    let session_value = session_file
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "disabled".to_string());

    print_panel(
        "Local CLI",
        Tone::Accent,
        &[
            format!("scope {}", scope_label(workspace)),
            "default boundary current launch directory only".to_string(),
            "widen scope explicitly with /scope use <path> or --workspace <path>".to_string(),
            format!(
                "models text={} planner={} coder={}",
                routing.text_model,
                routing
                    .planner_model
                    .as_deref()
                    .unwrap_or(routing.text_model.as_str()),
                routing
                    .coder_model
                    .as_deref()
                    .unwrap_or(routing.text_model.as_str())
            ),
            format!(
                "provider {}   approvals {}   session {}",
                provider,
                if auto_mode { "auto" } else { "ask" },
                session_value
            ),
        ],
    );
}

pub fn print_help() {
    print_panel(
        "Commands",
        Tone::Info,
        &[
            "/help                  show this help".to_string(),
            "/status                show scope, model, and last task state".to_string(),
            "/plan                  show the last recorded plan".to_string(),
            "/scope                 show current scope".to_string(),
            "/scope use <path>      change scope explicitly".to_string(),
            "/agents                list agents".to_string(),
            "/agent new <name>      create an agent".to_string(),
            "/agent use <name>      switch active agent".to_string(),
            "/agent run <name> <prompt>  run a prompt on a specific agent".to_string(),
            "/agent reset <name>    reset an agent session".to_string(),
            "/reset                 reset the active agent".to_string(),
            "/save [path]           save session".to_string(),
            "/load [path]           load session".to_string(),
            "/reload_plugins        reload plugin tools".to_string(),
            "/models                show routing".to_string(),
            "/exit                  quit".to_string(),
        ],
    );
}

pub fn print_status(
    workspace: &Path,
    provider: &str,
    routing: &ModelRouting,
    active_agent: &str,
    auto_mode: bool,
    session_file: Option<&Path>,
    last_summary: Option<&RunSummary>,
) {
    let mut lines = vec![
        format!("agent {active_agent}"),
        format!("scope {}", scope_label(workspace)),
        "boundary current launch directory".to_string(),
        format!("provider {provider}"),
        format!("text model {}", routing.text_model),
        format!(
            "planner model {}",
            routing
                .planner_model
                .as_deref()
                .unwrap_or(routing.text_model.as_str())
        ),
        format!(
            "coder model {}",
            routing
                .coder_model
                .as_deref()
                .unwrap_or(routing.text_model.as_str())
        ),
        format!("approvals {}", if auto_mode { "auto" } else { "ask" }),
        format!(
            "session file {}",
            session_file
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "disabled".to_string())
        ),
    ];

    if let Some(summary) = last_summary {
        lines.push(String::new());
        lines.push(format!("last prompt {}", summary.user_prompt));
        if let Some(plan) = summary.plan.as_ref() {
            lines.push(format!(
                "last plan {} ({})",
                plan.summary,
                if plan.needs_plan {
                    "detailed"
                } else {
                    "direct"
                }
            ));
        }
        if let Some(handoff) = summary.handoff.as_ref() {
            lines.push(format!("last handoff {}", handoff.summary_or_message()));
        }
    }

    print_panel("Status", Tone::Info, &lines);
}

pub fn print_scope(workspace: &Path) {
    print_panel(
        "Scope",
        Tone::Accent,
        &[
            format!("current {}", scope_label(workspace)),
            "boundary current launch directory only".to_string(),
            "use /scope use <path> to widen or narrow scope explicitly".to_string(),
        ],
    );
}

pub fn prompt(
    active_agent: &str,
    workspace: &Path,
    routing: &ModelRouting,
    auto_mode: bool,
) -> String {
    let mode = if auto_mode { "auto" } else { "ask" };
    let label = format!(
        "✦ {}  {}  {}  {}",
        active_agent,
        workspace_name(workspace),
        compact(&routing.text_model, 22),
        mode
    );
    if supports_color() {
        format!("{} › ", paint(&label, Tone::Accent, true))
    } else {
        format!("{label} > ")
    }
}

pub fn print_plan(agent_name: &str, plan: &TaskPlan) {
    let mut lines = vec![
        format!("agent {agent_name}"),
        format!(
            "{} {}",
            glyph_for_tone(Tone::Accent),
            if plan.needs_plan {
                "detailed plan required"
            } else {
                "direct execution is enough"
            }
        ),
        format!("summary {}", plan.summary),
    ];

    if !plan.reason.is_empty() {
        lines.push(format!("reason {}", plan.reason));
    }

    if !plan.focus.is_empty() {
        lines.push("focus".to_string());
        for item in &plan.focus {
            lines.push(format!("• {item}"));
        }
    }

    if !plan.steps.is_empty() {
        lines.push("steps".to_string());
        for (index, step) in plan.steps.iter().enumerate() {
            lines.push(format!("{}. {}", index + 1, step));
        }
    }

    print_panel("Plan", Tone::Accent, &lines);
}

pub fn print_last_plan(plan: &TaskPlan) {
    print_plan("last-task", plan);
}

pub fn print_event(label: &str, tone: Tone, message: &str) {
    clear_live_line();
    let width = inner_width().saturating_sub(14).max(24);
    let wrapped = wrap_text(message, width);
    let prefix = format!(
        "{} {}",
        paint(glyph_for_tone(tone), tone, true),
        paint(label, tone, true)
    );
    for (index, line) in wrapped.iter().enumerate() {
        if index == 0 {
            println!("{prefix} {}", paint(line, Tone::Muted, false));
        } else {
            println!(
                "{} {}",
                " ".repeat(display_width(&prefix)),
                paint(line, Tone::Muted, false)
            );
        }
    }
}

pub fn print_agents(active_agent: &str, names: &[String]) {
    let mut lines = Vec::new();
    for name in names {
        if name == active_agent {
            lines.push(format!("• {}  active", name));
        } else {
            lines.push(format!("• {name}"));
        }
    }
    print_panel("Agents", Tone::Info, &lines);
}

pub fn print_handoff(agent_name: &str, handoff: &TaskHandoff) {
    let mut lines = vec![format!("agent {agent_name}")];

    if !handoff.summary.is_empty() {
        lines.push(format!("summary {}", handoff.summary));
    }

    if !handoff.completed.is_empty() {
        lines.push("completed".to_string());
        for item in &handoff.completed {
            lines.push(format!("• {item}"));
        }
    }

    if !handoff.verified.is_empty() {
        lines.push("verified".to_string());
        for item in &handoff.verified {
            lines.push(format!("• {item}"));
        }
    }

    lines.push("remaining gaps".to_string());
    if handoff.missed.is_empty() {
        lines.push("• none noted".to_string());
    } else {
        for item in &handoff.missed {
            lines.push(format!("• {item}"));
        }
    }

    if !handoff.next_steps.is_empty() {
        lines.push("next steps".to_string());
        for item in &handoff.next_steps {
            lines.push(format!("• {item}"));
        }
    }

    if !handoff.message.trim().is_empty() {
        lines.push("handoff".to_string());
        for line in handoff.message.lines() {
            lines.push(line.to_string());
        }
    }

    print_panel("Handoff", Tone::Success, &lines);
}

pub fn print_assistant_message(agent_name: &str, message: &str) {
    let mut lines = vec![format!("agent {agent_name}")];
    for line in message.lines() {
        lines.push(line.to_string());
    }
    print_panel("Assistant", Tone::Success, &lines);
}

pub fn print_notice(message: &str) {
    print_event("info", Tone::Info, message);
}

pub fn print_warning(message: &str) {
    print_event("warn", Tone::Warning, message);
}

pub fn print_error(message: &str) {
    print_event("error", Tone::Danger, message);
}

pub fn flush_stdout() {
    let _ = io::stdout().flush();
}

pub fn print_panel(title: &str, tone: Tone, lines: &[String]) {
    clear_live_line();
    let width = panel_width();
    let inner = width.saturating_sub(4);
    let title_text = format!(" {} ", title);
    let border_fill = "─".repeat(inner.saturating_sub(display_width(&title_text)).max(2));
    println!(
        "{}",
        paint(&format!("╭{}{}╮", title_text, border_fill), tone, true)
    );
    for line in expand_lines(lines, inner) {
        println!(
            "{} {:<inner$} {}",
            paint("│", tone, true),
            line,
            paint("│", tone, true)
        );
    }
    println!(
        "{}",
        paint(&format!("╰{}╯", "─".repeat(inner + 2)), tone, true)
    );
}

fn panel_width() -> usize {
    env::var("COLUMNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .map(|value| value.clamp(72, 118))
        .unwrap_or(96)
}

fn inner_width() -> usize {
    panel_width().saturating_sub(4)
}

fn expand_lines(lines: &[String], width: usize) -> Vec<String> {
    let mut expanded = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            expanded.push(String::new());
            continue;
        }
        expanded.extend(wrap_text(line, width));
    }
    expanded
}

fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let safe_width = width.max(8);
    let mut wrapped = Vec::new();

    for raw_line in text.lines() {
        if raw_line.is_empty() {
            wrapped.push(String::new());
            continue;
        }

        let indent_size = raw_line
            .chars()
            .take_while(|ch| ch.is_ascii_whitespace())
            .count()
            .min(safe_width.saturating_sub(1));
        let indent = " ".repeat(indent_size);
        let content = raw_line.trim();
        let available = safe_width.saturating_sub(indent_size).max(8);

        let mut current = String::new();
        for word in content.split_whitespace() {
            if current.is_empty() {
                if word.chars().count() > available {
                    for chunk in chunk_word(word, available) {
                        wrapped.push(format!("{indent}{chunk}"));
                    }
                } else {
                    current.push_str(word);
                }
                continue;
            }

            let candidate_len = current.chars().count() + 1 + word.chars().count();
            if candidate_len <= available {
                current.push(' ');
                current.push_str(word);
            } else {
                wrapped.push(format!("{indent}{current}"));
                if word.chars().count() > available {
                    for chunk in chunk_word(word, available) {
                        wrapped.push(format!("{indent}{chunk}"));
                    }
                    current.clear();
                } else {
                    current = word.to_string();
                }
            }
        }

        if !current.is_empty() {
            wrapped.push(format!("{indent}{current}"));
        }
    }

    if wrapped.is_empty() {
        vec![String::new()]
    } else {
        wrapped
    }
}

fn chunk_word(word: &str, width: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for ch in word.chars() {
        current.push(ch);
        if current.chars().count() >= width {
            chunks.push(current);
            current = String::new();
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn workspace_name(workspace: &Path) -> String {
    workspace
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("/")
        .to_string()
}

fn scope_label(workspace: &Path) -> String {
    format!("{}  {}", workspace_name(workspace), workspace.display())
}

fn compact(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }

    let head_len = (max_chars / 2).saturating_sub(1);
    let tail_len = max_chars.saturating_sub(head_len + 3);
    let head = value.chars().take(head_len).collect::<String>();
    let tail = value
        .chars()
        .rev()
        .take(tail_len)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    format!("{head}...{tail}")
}

fn single_line(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn glyph_for_tone(tone: Tone) -> &'static str {
    match tone {
        Tone::Accent => "✦",
        Tone::Info => "●",
        Tone::Success => "✓",
        Tone::Warning => "▲",
        Tone::Danger => "✕",
        Tone::Muted => "·",
    }
}

fn display_width(value: &str) -> usize {
    value.chars().count()
}

fn format_elapsed(duration: Duration) -> String {
    if duration.as_secs() >= 60 {
        let minutes = duration.as_secs() / 60;
        let seconds = duration.as_secs() % 60;
        format!("{minutes}m{seconds:02}s")
    } else if duration.as_millis() >= 1000 {
        format!("{:.1}s", duration.as_secs_f32())
    } else {
        format!("{}ms", duration.as_millis())
    }
}

fn clear_live_line() {
    if supports_live_output() {
        print!("\r\x1b[2K");
        let _ = io::stdout().flush();
    }
}

fn supports_live_output() -> bool {
    io::stdout().is_terminal()
        && env::var("TERM")
            .map(|value| value != "dumb")
            .unwrap_or(true)
}

fn supports_color() -> bool {
    supports_live_output() && env::var_os("NO_COLOR").is_none()
}

fn paint(value: &str, tone: Tone, bold: bool) -> String {
    if !supports_color() {
        return value.to_string();
    }

    let color = match tone {
        Tone::Accent => "38;5;81",
        Tone::Info => "38;5;111",
        Tone::Success => "38;5;114",
        Tone::Warning => "38;5;214",
        Tone::Danger => "38;5;203",
        Tone::Muted => "38;5;246",
    };
    let prefix = if bold {
        format!("\x1b[1;{color}m")
    } else {
        format!("\x1b[{color}m")
    };
    format!("{prefix}{value}\x1b[0m")
}
