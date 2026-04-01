//! Application state and logic.

use crossterm::event::{KeyCode, KeyEvent};
use std::collections::VecDeque;

/// Application mode/screen
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppMode {
    /// Main dashboard
    Dashboard,
    /// Chat interface
    Chat,
    /// Diagnostic pipeline
    Diagnose,
    /// Model selection
    Models,
    /// Settings
    Settings,
    /// Help screen
    Help,
}

/// Input focus area
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    /// Sidebar navigation
    Sidebar,
    /// Main content area
    Content,
    /// Input field
    Input,
}

/// A chat message
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub timestamp: String,
}

/// Diagnostic phase
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticPhase {
    Idle,
    MPE,
    ARLL,
    CSR,
    Complete,
}

/// Provider configuration
#[derive(Debug, Clone)]
pub struct ProviderStatus {
    pub name: String,
    pub configured: bool,
    pub model: String,
}

/// Application state
pub struct App {
    /// Current mode/screen
    pub mode: AppMode,
    /// Current focus
    pub focus: Focus,
    /// Should quit
    pub should_quit: bool,
    /// Sidebar selection index
    pub sidebar_index: usize,
    
    // Chat state
    pub chat_messages: Vec<ChatMessage>,
    pub chat_input: String,
    pub chat_scroll: usize,
    
    // Diagnostic state
    pub diagnostic_phase: DiagnosticPhase,
    pub diagnostic_progress: f64,
    pub diagnostic_confidence: f64,
    pub diagnostic_iterations: usize,
    pub diagnostic_output: String,
    pub symptoms_input: String,
    
    // Model state
    pub providers: Vec<ProviderStatus>,
    pub selected_provider: usize,
    pub selected_vision_model: String,
    pub selected_reasoning_model: String,
    pub selected_clinical_model: String,
    
    // Status
    pub status_message: String,
    pub is_loading: bool,
    pub streaming_text: String,
    
    // Logs
    pub logs: VecDeque<String>,
}

impl App {
    pub fn new() -> Self {
        Self {
            mode: AppMode::Dashboard,
            focus: Focus::Sidebar,
            should_quit: false,
            sidebar_index: 0,
            
            chat_messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: "Welcome to R-MoE Clinical Diagnostics. How can I help you today?".to_string(),
                    timestamp: chrono::Local::now().format("%H:%M").to_string(),
                },
            ],
            chat_input: String::new(),
            chat_scroll: 0,
            
            diagnostic_phase: DiagnosticPhase::Idle,
            diagnostic_progress: 0.0,
            diagnostic_confidence: 0.0,
            diagnostic_iterations: 0,
            diagnostic_output: String::new(),
            symptoms_input: String::new(),
            
            providers: vec![
                ProviderStatus { name: "OpenAI".to_string(), configured: std::env::var("OPENAI_API_KEY").is_ok(), model: "gpt-4o".to_string() },
                ProviderStatus { name: "Anthropic".to_string(), configured: std::env::var("ANTHROPIC_API_KEY").is_ok(), model: "claude-sonnet-4-20250514".to_string() },
                ProviderStatus { name: "Google".to_string(), configured: std::env::var("GOOGLE_API_KEY").is_ok(), model: "gemini-1.5-pro".to_string() },
                ProviderStatus { name: "Groq".to_string(), configured: std::env::var("GROQ_API_KEY").is_ok(), model: "llama-3.1-70b".to_string() },
                ProviderStatus { name: "Ollama".to_string(), configured: true, model: "llama3.1".to_string() },
            ],
            selected_provider: 0,
            selected_vision_model: "openai:gpt-4o".to_string(),
            selected_reasoning_model: "anthropic:claude-sonnet-4-20250514".to_string(),
            selected_clinical_model: "openai:gpt-4o".to_string(),
            
            status_message: "Ready".to_string(),
            is_loading: false,
            streaming_text: String::new(),
            
            logs: VecDeque::with_capacity(100),
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) {
        match self.focus {
            Focus::Sidebar => self.handle_sidebar_key(key),
            Focus::Content => self.handle_content_key(key),
            Focus::Input => self.handle_input_key(key),
        }
    }

    fn handle_sidebar_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                if self.sidebar_index > 0 {
                    self.sidebar_index -= 1;
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.sidebar_index < 5 {
                    self.sidebar_index += 1;
                }
            }
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                self.mode = match self.sidebar_index {
                    0 => AppMode::Dashboard,
                    1 => AppMode::Chat,
                    2 => AppMode::Diagnose,
                    3 => AppMode::Models,
                    4 => AppMode::Settings,
                    5 => AppMode::Help,
                    _ => AppMode::Dashboard,
                };
                self.focus = Focus::Content;
            }
            KeyCode::Char('q') => {
                self.should_quit = true;
            }
            KeyCode::Tab => {
                self.focus = Focus::Content;
            }
            _ => {}
        }
    }

    fn handle_content_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Left | KeyCode::Char('h') | KeyCode::Esc => {
                self.focus = Focus::Sidebar;
            }
            KeyCode::Tab => {
                if matches!(self.mode, AppMode::Chat | AppMode::Diagnose) {
                    self.focus = Focus::Input;
                } else {
                    self.focus = Focus::Sidebar;
                }
            }
            KeyCode::Up | KeyCode::Char('k') => {
                match self.mode {
                    AppMode::Chat => {
                        if self.chat_scroll > 0 {
                            self.chat_scroll -= 1;
                        }
                    }
                    AppMode::Models => {
                        if self.selected_provider > 0 {
                            self.selected_provider -= 1;
                        }
                    }
                    _ => {}
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                match self.mode {
                    AppMode::Chat => {
                        self.chat_scroll += 1;
                    }
                    AppMode::Models => {
                        if self.selected_provider < self.providers.len() - 1 {
                            self.selected_provider += 1;
                        }
                    }
                    _ => {}
                }
            }
            KeyCode::Char('i') => {
                if matches!(self.mode, AppMode::Chat | AppMode::Diagnose) {
                    self.focus = Focus::Input;
                }
            }
            KeyCode::Char('q') => {
                self.should_quit = true;
            }
            _ => {}
        }
    }

    fn handle_input_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                self.focus = Focus::Content;
            }
            KeyCode::Enter => {
                match self.mode {
                    AppMode::Chat => {
                        if !self.chat_input.is_empty() {
                            self.send_chat_message();
                        }
                    }
                    AppMode::Diagnose => {
                        if !self.symptoms_input.is_empty() {
                            self.start_diagnosis();
                        }
                    }
                    _ => {}
                }
            }
            KeyCode::Char(c) => {
                match self.mode {
                    AppMode::Chat => self.chat_input.push(c),
                    AppMode::Diagnose => self.symptoms_input.push(c),
                    _ => {}
                }
            }
            KeyCode::Backspace => {
                match self.mode {
                    AppMode::Chat => { self.chat_input.pop(); }
                    AppMode::Diagnose => { self.symptoms_input.pop(); }
                    _ => {}
                }
            }
            KeyCode::Tab => {
                self.focus = Focus::Content;
            }
            _ => {}
        }
    }

    fn send_chat_message(&mut self) {
        let content = std::mem::take(&mut self.chat_input);
        self.chat_messages.push(ChatMessage {
            role: "user".to_string(),
            content: content.clone(),
            timestamp: chrono::Local::now().format("%H:%M").to_string(),
        });
        
        // Simulate response (would call actual API)
        self.is_loading = true;
        self.status_message = "Generating response...".to_string();
        self.log(&format!("User: {}", content));
        
        // Placeholder response
        self.chat_messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: format!("I received your message: \"{}\". This is a placeholder response. Connect API keys to enable real responses.", content),
            timestamp: chrono::Local::now().format("%H:%M").to_string(),
        });
        self.is_loading = false;
        self.status_message = "Ready".to_string();
    }

    fn start_diagnosis(&mut self) {
        let symptoms = std::mem::take(&mut self.symptoms_input);
        self.diagnostic_phase = DiagnosticPhase::MPE;
        self.diagnostic_progress = 0.0;
        self.diagnostic_iterations = 1;
        self.is_loading = true;
        self.status_message = "Running diagnostic pipeline...".to_string();
        self.log(&format!("Starting diagnosis: {}", symptoms));
        self.diagnostic_output = format!("Symptoms: {}\n\nAnalyzing...", symptoms);
    }

    pub fn tick(&mut self) {
        // Simulate diagnostic progress
        if self.is_loading && matches!(self.diagnostic_phase, DiagnosticPhase::MPE | DiagnosticPhase::ARLL | DiagnosticPhase::CSR) {
            self.diagnostic_progress += 0.02;
            
            if self.diagnostic_progress >= 0.33 && self.diagnostic_phase == DiagnosticPhase::MPE {
                self.diagnostic_phase = DiagnosticPhase::ARLL;
                self.diagnostic_confidence = 0.75;
                self.log("Phase 1 (MPE) complete - moving to ARLL");
            } else if self.diagnostic_progress >= 0.66 && self.diagnostic_phase == DiagnosticPhase::ARLL {
                self.diagnostic_phase = DiagnosticPhase::CSR;
                self.diagnostic_confidence = 0.92;
                self.log("Phase 2 (ARLL) complete - moving to CSR");
            } else if self.diagnostic_progress >= 1.0 {
                self.diagnostic_phase = DiagnosticPhase::Complete;
                self.diagnostic_progress = 1.0;
                self.is_loading = false;
                self.status_message = "Diagnosis complete".to_string();
                self.diagnostic_output = "# Diagnostic Report\n\n## Primary Diagnosis\nCommunity-acquired Pneumonia (ICD-11: CA40)\n\n## Confidence\n92%\n\n## Findings\n- Bilateral infiltrates\n- Consolidation in RLL\n\n## Recommendations\n- Chest CT for confirmation\n- Sputum culture\n- Start empiric antibiotics".to_string();
                self.log("Diagnosis complete - confidence: 92%");
            }
        }
    }

    pub fn log(&mut self, message: &str) {
        let timestamp = chrono::Local::now().format("%H:%M:%S").to_string();
        self.logs.push_back(format!("[{}] {}", timestamp, message));
        if self.logs.len() > 100 {
            self.logs.pop_front();
        }
    }

    pub fn get_sidebar_items(&self) -> Vec<(&str, &str)> {
        vec![
            ("󰋜", "Dashboard"),
            ("󰭹", "Chat"),
            ("󰺕", "Diagnose"),
            ("󰘚", "Models"),
            ("󰒓", "Settings"),
            ("󰋖", "Help"),
        ]
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}
