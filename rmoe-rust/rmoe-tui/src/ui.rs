//! UI rendering functions.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span, Text},
    widgets::{
        Block, Borders, Clear, Gauge, List, ListItem, Padding, Paragraph, Scrollbar,
        ScrollbarOrientation, ScrollbarState, Tabs, Wrap,
    },
    Frame,
};

use crate::app::{App, AppMode, DiagnosticPhase, Focus};

/// Main UI drawing function
pub fn draw(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(20),  // Sidebar
            Constraint::Min(60),     // Main content
        ])
        .split(f.size());

    draw_sidebar(f, app, chunks[0]);
    draw_main_content(f, app, chunks[1]);
}

/// Draw the sidebar navigation
fn draw_sidebar(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .title(" R-MoE ")
        .title_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .borders(Borders::ALL)
        .border_style(if app.focus == Focus::Sidebar {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default().fg(Color::DarkGray)
        });

    let items: Vec<ListItem> = app
        .get_sidebar_items()
        .iter()
        .enumerate()
        .map(|(i, (icon, label))| {
            let style = if i == app.sidebar_index {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            let prefix = if i == app.sidebar_index { "▸ " } else { "  " };
            ListItem::new(format!("{}{} {}", prefix, icon, label)).style(style)
        })
        .collect();

    let list = List::new(items)
        .block(block)
        .highlight_style(Style::default().fg(Color::Cyan));

    f.render_widget(list, area);

    // Draw status at bottom of sidebar
    let status_area = Rect {
        x: area.x + 1,
        y: area.y + area.height - 3,
        width: area.width - 2,
        height: 2,
    };
    
    let status_text = if app.is_loading {
        Span::styled("⟳ Loading...", Style::default().fg(Color::Yellow))
    } else {
        Span::styled("● Ready", Style::default().fg(Color::Green))
    };
    
    f.render_widget(Paragraph::new(status_text), status_area);
}

/// Draw the main content area
fn draw_main_content(f: &mut Frame, app: &App, area: Rect) {
    match app.mode {
        AppMode::Dashboard => draw_dashboard(f, app, area),
        AppMode::Chat => draw_chat(f, app, area),
        AppMode::Diagnose => draw_diagnose(f, app, area),
        AppMode::Models => draw_models(f, app, area),
        AppMode::Settings => draw_settings(f, app, area),
        AppMode::Help => draw_help(f, app, area),
    }
}

/// Draw dashboard screen
fn draw_dashboard(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),   // Header
            Constraint::Length(10),  // Stats
            Constraint::Min(10),     // Logs
            Constraint::Length(3),   // Status bar
        ])
        .split(area);

    // Header
    let header_block = Block::default()
        .title(" Dashboard ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Blue));

    let header_text = vec![
        Line::from(vec![
            Span::styled("🧠 R-MoE Engine", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Recursive Multi-Agent Mixture-of-Experts"),
        ]),
        Line::from(vec![
            Span::styled("for Clinical Diagnostics", Style::default().fg(Color::Yellow)),
        ]),
    ];

    let header = Paragraph::new(header_text)
        .block(header_block)
        .alignment(Alignment::Center);

    f.render_widget(header, chunks[0]);

    // Stats
    let stats_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(34),
            Constraint::Percentage(33),
        ])
        .split(chunks[1]);

    // Provider stats
    let configured = app.providers.iter().filter(|p| p.configured).count();
    let provider_block = Block::default()
        .title(" Providers ")
        .borders(Borders::ALL);
    let provider_text = format!("{}/{} configured", configured, app.providers.len());
    f.render_widget(
        Paragraph::new(provider_text)
            .block(provider_block)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Green)),
        stats_chunks[0],
    );

    // Pipeline stats
    let pipeline_block = Block::default()
        .title(" Pipeline ")
        .borders(Borders::ALL);
    f.render_widget(
        Paragraph::new("MPE → ARLL → CSR")
            .block(pipeline_block)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Cyan)),
        stats_chunks[1],
    );

    // Confidence threshold
    let threshold_block = Block::default()
        .title(" Threshold ")
        .borders(Borders::ALL);
    f.render_widget(
        Paragraph::new("θ = 0.90")
            .block(threshold_block)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Yellow)),
        stats_chunks[2],
    );

    // Logs
    let logs_block = Block::default()
        .title(" Activity Log ")
        .borders(Borders::ALL);
    
    let log_items: Vec<ListItem> = app.logs
        .iter()
        .rev()
        .take(10)
        .map(|log| ListItem::new(log.as_str()).style(Style::default().fg(Color::DarkGray)))
        .collect();

    let logs_list = List::new(log_items).block(logs_block);
    f.render_widget(logs_list, chunks[2]);

    // Status bar
    draw_status_bar(f, app, chunks[3]);
}

/// Draw chat interface
fn draw_chat(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(10),     // Messages
            Constraint::Length(3),   // Input
            Constraint::Length(3),   // Status
        ])
        .split(area);

    // Messages
    let messages_block = Block::default()
        .title(" Chat ")
        .borders(Borders::ALL)
        .border_style(if app.focus == Focus::Content {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default().fg(Color::DarkGray)
        });

    let messages: Vec<ListItem> = app.chat_messages
        .iter()
        .map(|msg| {
            let style = match msg.role.as_str() {
                "user" => Style::default().fg(Color::Green),
                "assistant" => Style::default().fg(Color::Cyan),
                "system" => Style::default().fg(Color::Yellow),
                _ => Style::default(),
            };
            let prefix = match msg.role.as_str() {
                "user" => "You",
                "assistant" => "R-MoE",
                "system" => "System",
                _ => "Unknown",
            };
            ListItem::new(vec![
                Line::from(vec![
                    Span::styled(format!("[{}] ", msg.timestamp), Style::default().fg(Color::DarkGray)),
                    Span::styled(format!("{}:", prefix), style.add_modifier(Modifier::BOLD)),
                ]),
                Line::from(format!("  {}", msg.content)),
                Line::from(""),
            ])
        })
        .collect();

    let messages_list = List::new(messages).block(messages_block);
    f.render_widget(messages_list, chunks[0]);

    // Input
    let input_block = Block::default()
        .title(" Message (Enter to send) ")
        .borders(Borders::ALL)
        .border_style(if app.focus == Focus::Input {
            Style::default().fg(Color::Green)
        } else {
            Style::default().fg(Color::DarkGray)
        });

    let input = Paragraph::new(app.chat_input.as_str())
        .block(input_block)
        .style(Style::default().fg(Color::White));
    
    f.render_widget(input, chunks[1]);

    // Show cursor in input
    if app.focus == Focus::Input {
        f.set_cursor(
            chunks[1].x + app.chat_input.len() as u16 + 1,
            chunks[1].y + 1,
        );
    }

    // Status
    draw_status_bar(f, app, chunks[2]);
}

/// Draw diagnostic interface
fn draw_diagnose(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),   // Input
            Constraint::Length(5),   // Progress
            Constraint::Min(10),     // Output
            Constraint::Length(3),   // Status
        ])
        .split(area);

    // Symptoms input
    let input_block = Block::default()
        .title(" Symptoms / Clinical Notes (Enter to diagnose) ")
        .borders(Borders::ALL)
        .border_style(if app.focus == Focus::Input {
            Style::default().fg(Color::Green)
        } else {
            Style::default().fg(Color::DarkGray)
        });

    let input = Paragraph::new(app.symptoms_input.as_str())
        .block(input_block);
    f.render_widget(input, chunks[0]);

    if app.focus == Focus::Input {
        f.set_cursor(
            chunks[0].x + app.symptoms_input.len() as u16 + 1,
            chunks[0].y + 1,
        );
    }

    // Progress
    let progress_block = Block::default()
        .title(" Pipeline Progress ")
        .borders(Borders::ALL);

    let phase_label = match app.diagnostic_phase {
        DiagnosticPhase::Idle => "Idle",
        DiagnosticPhase::MPE => "Phase 1: MPE (Perception)",
        DiagnosticPhase::ARLL => "Phase 2: ARLL (Reasoning)",
        DiagnosticPhase::CSR => "Phase 3: CSR (Clinical)",
        DiagnosticPhase::Complete => "Complete",
    };

    let progress_color = match app.diagnostic_phase {
        DiagnosticPhase::Idle => Color::DarkGray,
        DiagnosticPhase::MPE => Color::Blue,
        DiagnosticPhase::ARLL => Color::Yellow,
        DiagnosticPhase::CSR => Color::Magenta,
        DiagnosticPhase::Complete => Color::Green,
    };

    let gauge = Gauge::default()
        .block(progress_block)
        .gauge_style(Style::default().fg(progress_color))
        .percent((app.diagnostic_progress * 100.0) as u16)
        .label(format!("{} | Sc: {:.0}% | Iter: {}", 
            phase_label, 
            app.diagnostic_confidence * 100.0,
            app.diagnostic_iterations
        ));

    f.render_widget(gauge, chunks[1]);

    // Output
    let output_block = Block::default()
        .title(" Diagnostic Output ")
        .borders(Borders::ALL)
        .border_style(if app.focus == Focus::Content {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default().fg(Color::DarkGray)
        });

    let output = Paragraph::new(app.diagnostic_output.as_str())
        .block(output_block)
        .wrap(Wrap { trim: true });

    f.render_widget(output, chunks[2]);

    // Status
    draw_status_bar(f, app, chunks[3]);
}

/// Draw models screen
fn draw_models(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(12),  // Providers list
            Constraint::Min(8),      // Selected models
            Constraint::Length(3),   // Status
        ])
        .split(area);

    // Providers
    let providers_block = Block::default()
        .title(" API Providers ")
        .borders(Borders::ALL)
        .border_style(if app.focus == Focus::Content {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default().fg(Color::DarkGray)
        });

    let provider_items: Vec<ListItem> = app.providers
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let status = if p.configured { "✓" } else { "✗" };
            let status_color = if p.configured { Color::Green } else { Color::Red };
            let selected = if i == app.selected_provider { "▸ " } else { "  " };
            
            ListItem::new(Line::from(vec![
                Span::raw(selected),
                Span::styled(status, Style::default().fg(status_color)),
                Span::raw(format!(" {} ", p.name)),
                Span::styled(format!("({})", p.model), Style::default().fg(Color::DarkGray)),
            ]))
        })
        .collect();

    let providers_list = List::new(provider_items).block(providers_block);
    f.render_widget(providers_list, chunks[0]);

    // Selected models
    let models_block = Block::default()
        .title(" Pipeline Models ")
        .borders(Borders::ALL);

    let models_text = vec![
        Line::from(vec![
            Span::styled("Vision (MPE):    ", Style::default().fg(Color::Blue)),
            Span::raw(&app.selected_vision_model),
        ]),
        Line::from(vec![
            Span::styled("Reasoning (ARLL): ", Style::default().fg(Color::Yellow)),
            Span::raw(&app.selected_reasoning_model),
        ]),
        Line::from(vec![
            Span::styled("Clinical (CSR):  ", Style::default().fg(Color::Magenta)),
            Span::raw(&app.selected_clinical_model),
        ]),
    ];

    let models_para = Paragraph::new(models_text).block(models_block);
    f.render_widget(models_para, chunks[1]);

    // Status
    draw_status_bar(f, app, chunks[2]);
}

/// Draw settings screen
fn draw_settings(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .title(" Settings ")
        .borders(Borders::ALL);

    let settings_text = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Confidence Threshold: ", Style::default().fg(Color::Cyan)),
            Span::raw("0.90"),
        ]),
        Line::from(vec![
            Span::styled("  Max Iterations:       ", Style::default().fg(Color::Cyan)),
            Span::raw("3"),
        ]),
        Line::from(vec![
            Span::styled("  Temperature:          ", Style::default().fg(Color::Cyan)),
            Span::raw("0.2"),
        ]),
        Line::from(vec![
            Span::styled("  Max Tokens:           ", Style::default().fg(Color::Cyan)),
            Span::raw("512"),
        ]),
        Line::from(""),
        Line::from(Span::styled("  (Settings editing coming soon)", Style::default().fg(Color::DarkGray))),
    ];

    let para = Paragraph::new(settings_text).block(block);
    f.render_widget(para, area);
}

/// Draw help screen
fn draw_help(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .title(" Help ")
        .borders(Borders::ALL);

    let help_text = vec![
        Line::from(""),
        Line::from(Span::styled("  Navigation", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(""),
        Line::from("  ↑/k, ↓/j    Move up/down"),
        Line::from("  ←/h, →/l    Switch panels"),
        Line::from("  Tab         Cycle focus"),
        Line::from("  Enter       Select / Submit"),
        Line::from("  Esc         Back / Cancel"),
        Line::from("  i           Focus input"),
        Line::from("  q           Quit"),
        Line::from("  Ctrl+C      Force quit"),
        Line::from(""),
        Line::from(Span::styled("  Screens", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(""),
        Line::from("  Dashboard   Overview and logs"),
        Line::from("  Chat        Interactive chat"),
        Line::from("  Diagnose    Clinical pipeline"),
        Line::from("  Models      Provider config"),
        Line::from("  Settings    Configuration"),
        Line::from(""),
        Line::from(Span::styled("  Pipeline: MPE → ARLL → CSR", Style::default().fg(Color::Yellow))),
        Line::from(Span::styled("  #wanna# Protocol: Sc ≥ 0.90", Style::default().fg(Color::Green))),
    ];

    let para = Paragraph::new(help_text).block(block);
    f.render_widget(para, area);
}

/// Draw status bar
fn draw_status_bar(f: &mut Frame, app: &App, area: Rect) {
    let status_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));

    let mode_name = match app.mode {
        AppMode::Dashboard => "Dashboard",
        AppMode::Chat => "Chat",
        AppMode::Diagnose => "Diagnose",
        AppMode::Models => "Models",
        AppMode::Settings => "Settings",
        AppMode::Help => "Help",
    };

    let focus_name = match app.focus {
        Focus::Sidebar => "Sidebar",
        Focus::Content => "Content",
        Focus::Input => "Input",
    };

    let status = Line::from(vec![
        Span::styled(format!(" {} ", mode_name), Style::default().fg(Color::Cyan)),
        Span::raw(" | "),
        Span::styled(format!("Focus: {} ", focus_name), Style::default().fg(Color::DarkGray)),
        Span::raw(" | "),
        Span::raw(&app.status_message),
        Span::raw(" | "),
        Span::styled("q:Quit  ?:Help", Style::default().fg(Color::DarkGray)),
    ]);

    let status_para = Paragraph::new(status).block(status_block);
    f.render_widget(status_para, area);
}
