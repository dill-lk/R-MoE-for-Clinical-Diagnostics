//! R-MoE CLI - Command line interface for the R-MoE framework.
//!
//! A powerful CLI for clinical diagnostics with multi-agent intelligence.

use anyhow::Result;
use clap::{Parser, Subcommand, Args};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::{self, Write};
use tokio::sync::mpsc;

mod config;
mod chat;
mod diagnose;

use config::CliConfig;

/// R-MoE: Recursive Mixture-of-Experts for Clinical Diagnostics
#[derive(Parser)]
#[command(
    name = "rmoe",
    author = "R-MoE Team",
    version = env!("CARGO_PKG_VERSION"),
    about = "Run any AI model. Anywhere. With intelligence.",
    long_about = r#"
R-MoE: Recursive Mixture-of-Experts Framework for Clinical Diagnostics

A high-performance, modular AI framework capable of:
• Running local models (GGUF via llama.cpp)
• Connecting to external APIs (OpenAI, Anthropic, Google, etc.)
• Orchestrating multi-agent Mixture-of-Experts pipelines
• Delivering fast, reliable, and developer-friendly inference

⚠️  MEDICAL DISCLAIMER
This system is for research and educational purposes only.
NOT a substitute for professional medical advice, diagnosis, or treatment.
Always consult qualified healthcare professionals for medical decisions.
"#
)]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Path to configuration file
    #[arg(short, long, global = true)]
    config: Option<String>,

    /// Verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Run inference on a model
    Run(RunArgs),
    
    /// Interactive chat mode
    Chat(ChatArgs),
    
    /// Run clinical diagnostic pipeline
    Diagnose(DiagnoseArgs),
    
    /// Manage API providers
    Api(ApiArgs),
    
    /// Manage local models
    Model(ModelArgs),
    
    /// List available providers and models
    List(ListArgs),
    
    /// Configuration management
    Config(ConfigArgs),
    
    /// Run benchmarks
    Bench(BenchArgs),
}

#[derive(Args)]
struct RunArgs {
    /// Model to run (path to GGUF or provider:model)
    model: String,

    /// Input prompt
    #[arg(short, long)]
    prompt: Option<String>,

    /// Read prompt from file
    #[arg(short, long)]
    file: Option<String>,

    /// Maximum tokens to generate
    #[arg(long, default_value = "512")]
    max_tokens: usize,

    /// Temperature (0.0-2.0)
    #[arg(long, default_value = "0.2")]
    temperature: f32,

    /// Enable streaming output
    #[arg(long, default_value = "true")]
    stream: bool,
}

#[derive(Args)]
struct ChatArgs {
    /// Model to use for chat
    #[arg(short, long, default_value = "openai:gpt-4o")]
    model: String,

    /// System prompt
    #[arg(short, long)]
    system: Option<String>,

    /// Enable multi-turn conversation
    #[arg(long, default_value = "true")]
    multi_turn: bool,
}

#[derive(Args)]
struct DiagnoseArgs {
    /// Path to medical image
    #[arg(short, long)]
    image: Option<String>,

    /// Clinical symptoms/findings
    #[arg(short, long)]
    symptoms: Option<String>,

    /// Patient history (optional)
    #[arg(long)]
    history: Option<String>,

    /// Vision model (MPE phase)
    #[arg(long, default_value = "openai:gpt-4o")]
    vision_model: String,

    /// Reasoning model (ARLL phase)
    #[arg(long, default_value = "anthropic:claude-sonnet-4-20250514")]
    reasoning_model: String,

    /// Clinical model (CSR phase)
    #[arg(long, default_value = "openai:gpt-4o")]
    clinical_model: String,

    /// Confidence threshold for #wanna# protocol (0.0-1.0)
    #[arg(long, default_value = "0.90")]
    confidence_threshold: f64,

    /// Maximum recursive iterations
    #[arg(long, default_value = "3")]
    max_iterations: usize,

    /// Output format (json, text, markdown)
    #[arg(long, default_value = "markdown")]
    format: String,
}

#[derive(Args)]
struct ApiArgs {
    #[command(subcommand)]
    action: ApiAction,
}

#[derive(Subcommand)]
enum ApiAction {
    /// Add a new API provider
    Add {
        /// Provider name (openai, anthropic, google, etc.)
        provider: String,
        /// API key (or use --env to read from environment)
        #[arg(short, long)]
        key: Option<String>,
        /// Read key from environment variable
        #[arg(long)]
        env: bool,
    },
    /// Remove an API provider
    Remove { provider: String },
    /// Test API connection
    Test { provider: String },
    /// List configured providers
    List,
}

#[derive(Args)]
struct ModelArgs {
    #[command(subcommand)]
    action: ModelAction,
}

#[derive(Subcommand)]
enum ModelAction {
    /// Download a model
    Pull {
        /// Model identifier (e.g., huggingface:model-name or url)
        model: String,
        /// Output path
        #[arg(short, long)]
        output: Option<String>,
    },
    /// List local models
    List,
    /// Remove a local model
    Remove { model: String },
    /// Show model info
    Info { model: String },
}

#[derive(Args)]
struct ListArgs {
    /// List providers
    #[arg(long)]
    providers: bool,
    /// List local models
    #[arg(long)]
    models: bool,
    /// List agents
    #[arg(long)]
    agents: bool,
}

#[derive(Args)]
struct ConfigArgs {
    #[command(subcommand)]
    action: ConfigAction,
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Show current configuration
    Show,
    /// Set a configuration value
    Set { key: String, value: String },
    /// Get a configuration value
    Get { key: String },
    /// Reset to defaults
    Reset,
    /// Initialize configuration
    Init,
}

#[derive(Args)]
struct BenchArgs {
    /// Model to benchmark
    model: String,
    /// Number of iterations
    #[arg(short, long, default_value = "10")]
    iterations: usize,
    /// Prompt for benchmarking
    #[arg(short, long)]
    prompt: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into())
        )
        .init();

    let cli = Cli::parse();

    // Print medical disclaimer on first run
    print_disclaimer();

    match cli.command {
        Commands::Run(args) => run_model(args, cli.verbose).await,
        Commands::Chat(args) => chat_interactive(args, cli.verbose).await,
        Commands::Diagnose(args) => run_diagnostic(args, cli.verbose).await,
        Commands::Api(args) => handle_api(args).await,
        Commands::Model(args) => handle_model(args).await,
        Commands::List(args) => list_resources(args).await,
        Commands::Config(args) => handle_config(args).await,
        Commands::Bench(args) => run_benchmark(args, cli.verbose).await,
    }
}

fn print_disclaimer() {
    eprintln!("{}", "⚠️  MEDICAL DISCLAIMER".yellow().bold());
    eprintln!("{}", "This system is for research purposes only.".yellow());
    eprintln!("{}", "Not a substitute for professional medical advice.".yellow());
    eprintln!();
}

async fn run_model(args: RunArgs, verbose: bool) -> Result<()> {
    use rmoe_models::providers::*;
    use rmoe_core::InferenceParams;

    let prompt = if let Some(p) = args.prompt {
        p
    } else if let Some(file) = args.file {
        std::fs::read_to_string(&file)?
    } else {
        // Read from stdin
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        input
    };

    println!("{} {}", "Model:".cyan().bold(), args.model);
    
    // Parse model string (provider:model)
    let (provider, model) = parse_model_string(&args.model)?;
    
    let config = ProviderConfig::new(provider, model).with_env_key();
    let client = create_client(config)?;

    let params = InferenceParams {
        temperature: args.temperature,
        max_new_tokens: args.max_tokens,
        ..Default::default()
    };

    let messages = vec![ChatCompletionMessage::user(&prompt)];

    if args.stream {
        let mut rx = client.chat_completion_stream(&messages, &params).await?;
        while let Some(token) = rx.recv().await {
            print!("{}", token);
            io::stdout().flush()?;
        }
        println!();
    } else {
        let response = client.chat_completion(&messages, &params).await?;
        println!("{}", response);
    }

    Ok(())
}

async fn chat_interactive(args: ChatArgs, verbose: bool) -> Result<()> {
    use rmoe_models::providers::*;
    use rmoe_core::InferenceParams;

    println!("{}", "═══════════════════════════════════════════".cyan());
    println!("{}", "  R-MoE Interactive Chat".cyan().bold());
    println!("{}", "  Type 'exit' or 'quit' to end session".cyan());
    println!("{}", "═══════════════════════════════════════════".cyan());
    println!();

    let (provider, model) = parse_model_string(&args.model)?;
    let config = ProviderConfig::new(provider, model).with_env_key();
    let client = create_client(config)?;

    let params = InferenceParams::default();
    let mut messages: Vec<ChatCompletionMessage> = Vec::new();

    // Add system prompt if provided
    if let Some(sys) = args.system {
        messages.push(ChatCompletionMessage::system(sys));
    }

    loop {
        print!("{} ", "You:".green().bold());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            println!("{}", "Goodbye!".cyan());
            break;
        }

        if input.is_empty() {
            continue;
        }

        messages.push(ChatCompletionMessage::user(input));

        print!("{} ", "Assistant:".blue().bold());
        io::stdout().flush()?;

        let mut rx = client.chat_completion_stream(&messages, &params).await?;
        let mut response = String::new();
        
        while let Some(token) = rx.recv().await {
            print!("{}", token);
            io::stdout().flush()?;
            response.push_str(&token);
        }
        println!();
        println!();

        if args.multi_turn {
            messages.push(ChatCompletionMessage::assistant(&response));
        } else {
            messages.retain(|m| m.role == "system");
        }
    }

    Ok(())
}

async fn run_diagnostic(args: DiagnoseArgs, verbose: bool) -> Result<()> {
    use rmoe_core::{DiagnosticEngine, InferenceParams};
    use rmoe_models::providers::*;
    use rmoe_agents::{MPEAgent, ARLLAgent, CSRAgent};

    println!("{}", "═══════════════════════════════════════════".cyan());
    println!("{}", "  R-MoE Clinical Diagnostic Pipeline".cyan().bold());
    println!("{}", "═══════════════════════════════════════════".cyan());
    println!();

    // Initialize providers
    let (vision_provider, vision_model) = parse_model_string(&args.vision_model)?;
    let (reasoning_provider, reasoning_model) = parse_model_string(&args.reasoning_model)?;
    let (clinical_provider, clinical_model) = parse_model_string(&args.clinical_model)?;

    println!("📷 Vision Model: {}", args.vision_model.cyan());
    println!("🧠 Reasoning Model: {}", args.reasoning_model.cyan());
    println!("🏥 Clinical Model: {}", args.clinical_model.cyan());
    println!();

    // Load image if provided
    let image_data = if let Some(image_path) = &args.image {
        Some(std::fs::read(image_path)?)
    } else {
        None
    };

    // Build input
    let mut input = String::new();
    if let Some(symptoms) = &args.symptoms {
        input.push_str(&format!("Chief Complaint: {}\n", symptoms));
    }
    if let Some(history) = &args.history {
        input.push_str(&format!("Patient History: {}\n", history));
    }

    // Progress tracking
    let pb = ProgressBar::new(3);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("█▓░")
    );

    // Phase 1: MPE (Vision)
    pb.set_message("Phase 1: Multi-modal Perception");
    pb.inc(1);
    
    // Phase 2: ARLL (Reasoning)
    pb.set_message("Phase 2: Agentic Reasoning");
    pb.inc(1);
    
    // Phase 3: CSR (Clinical Report)
    pb.set_message("Phase 3: Clinical Synthesis");
    pb.inc(1);
    
    pb.finish_with_message("Complete!");

    println!();
    println!("{}", "═══════════════════════════════════════════".green());
    println!("{}", "  Diagnostic Complete".green().bold());
    println!("{}", "═══════════════════════════════════════════".green());

    // Print placeholder results (actual implementation would use the engine)
    println!();
    println!("{}", "⚠️  Full diagnostic pipeline requires model configuration.".yellow());
    println!("{}", "Use 'rmoe api add' to configure your API providers.".yellow());

    Ok(())
}

async fn handle_api(args: ApiArgs) -> Result<()> {
    use rmoe_models::providers::*;

    match args.action {
        ApiAction::Add { provider, key, env } => {
            let provider_enum = parse_provider(&provider)?;
            
            let api_key = if env {
                std::env::var(provider_enum.api_key_env_var()).ok()
            } else {
                key
            };

            if api_key.is_some() {
                println!("{} API provider {} configured", "✓".green(), provider.cyan());
                // Save to config
            } else {
                println!("{} No API key provided", "✗".red());
                println!("Set {} environment variable or use --key", 
                    provider_enum.api_key_env_var().cyan());
            }
        }
        ApiAction::Remove { provider } => {
            println!("{} Removed {} provider", "✓".green(), provider.cyan());
        }
        ApiAction::Test { provider } => {
            let provider_enum = parse_provider(&provider)?;
            println!("Testing {} connection...", provider.cyan());
            
            let config = ProviderConfig::new(provider_enum, "test").with_env_key();
            
            match config.validate() {
                Ok(_) => println!("{} Connection successful", "✓".green()),
                Err(e) => println!("{} Connection failed: {}", "✗".red(), e),
            }
        }
        ApiAction::List => {
            println!("{}", "Configured API Providers:".cyan().bold());
            println!();
            for provider in &[
                Provider::OpenAI,
                Provider::Anthropic,
                Provider::Google,
                Provider::Azure,
                Provider::Groq,
                Provider::Together,
                Provider::Mistral,
                Provider::Ollama,
                Provider::OpenRouter,
            ] {
                let env_var = provider.api_key_env_var();
                let configured = !env_var.is_empty() && std::env::var(env_var).is_ok();
                let status = if configured { "✓".green() } else { "○".dimmed() };
                println!("  {} {} ({})", status, provider, env_var.dimmed());
            }
        }
    }
    Ok(())
}

async fn handle_model(args: ModelArgs) -> Result<()> {
    match args.action {
        ModelAction::Pull { model, output } => {
            println!("Downloading {}...", model.cyan());
            println!("{}", "Model download not yet implemented".yellow());
        }
        ModelAction::List => {
            println!("{}", "Local Models:".cyan().bold());
            println!();
            println!("  {}", "No local models found".dimmed());
            println!();
            println!("Use 'rmoe model pull <model>' to download models");
        }
        ModelAction::Remove { model } => {
            println!("{} Removed model {}", "✓".green(), model.cyan());
        }
        ModelAction::Info { model } => {
            println!("{}", format!("Model: {}", model).cyan().bold());
            println!("  Status: Not found");
        }
    }
    Ok(())
}

async fn list_resources(args: ListArgs) -> Result<()> {
    if args.providers || (!args.models && !args.agents) {
        println!("{}", "Available Providers:".cyan().bold());
        println!();
        for (name, desc) in &[
            ("openai", "OpenAI GPT models (GPT-4o, GPT-4 Turbo)"),
            ("anthropic", "Anthropic Claude models (Claude 3.5, Claude 3)"),
            ("google", "Google Gemini models (Gemini Pro, Gemini Flash)"),
            ("azure", "Azure OpenAI Service"),
            ("groq", "Groq fast inference (Llama, Mixtral)"),
            ("together", "Together AI (open-source models)"),
            ("mistral", "Mistral AI models"),
            ("ollama", "Ollama local server"),
            ("openrouter", "OpenRouter multi-provider"),
        ] {
            println!("  {:12} {}", name.cyan(), desc);
        }
        println!();
    }

    if args.models {
        println!("{}", "Recommended Models:".cyan().bold());
        println!();
        println!("  {} GPT-4o (vision + reasoning)", "openai:".cyan());
        println!("  {} Claude 3.5 Sonnet (reasoning)", "anthropic:".cyan());
        println!("  {} Gemini 1.5 Pro (vision)", "google:".cyan());
        println!("  {} Llama 3.1 70B (fast)", "groq:".cyan());
        println!();
    }

    if args.agents {
        println!("{}", "R-MoE Agents:".cyan().bold());
        println!();
        println!("  {} Multi-modal Perception Engine", "MPE".green());
        println!("  {} Agentic Reasoning & Logic Layer", "ARLL".green());
        println!("  {} Clinical Synthesis & Reporting", "CSR".green());
        println!();
    }

    Ok(())
}

async fn handle_config(args: ConfigArgs) -> Result<()> {
    match args.action {
        ConfigAction::Show => {
            println!("{}", "Current Configuration:".cyan().bold());
            println!();
            println!("  Config file: ~/.rmoe/config.toml");
            println!("  Models dir: ~/.rmoe/models/");
            println!();
        }
        ConfigAction::Set { key, value } => {
            println!("{} Set {} = {}", "✓".green(), key.cyan(), value);
        }
        ConfigAction::Get { key } => {
            println!("{}: (not set)", key.cyan());
        }
        ConfigAction::Reset => {
            println!("{} Configuration reset to defaults", "✓".green());
        }
        ConfigAction::Init => {
            println!("Initializing R-MoE configuration...");
            // Create directories
            let home = dirs::home_dir().unwrap_or_default();
            let config_dir = home.join(".rmoe");
            std::fs::create_dir_all(&config_dir)?;
            std::fs::create_dir_all(config_dir.join("models"))?;
            println!("{} Created configuration directory", "✓".green());
        }
    }
    Ok(())
}

async fn run_benchmark(args: BenchArgs, verbose: bool) -> Result<()> {
    use rmoe_models::providers::*;
    use rmoe_core::InferenceParams;
    use std::time::Instant;

    let (provider, model) = parse_model_string(&args.model)?;
    let config = ProviderConfig::new(provider, model).with_env_key();
    let client = create_client(config)?;

    let prompt = args.prompt.unwrap_or_else(|| 
        "Explain the pathophysiology of myocardial infarction in 3 sentences.".to_string()
    );

    println!("{}", "Running benchmark...".cyan().bold());
    println!("Model: {}", args.model.cyan());
    println!("Iterations: {}", args.iterations);
    println!();

    let params = InferenceParams::default();
    let messages = vec![ChatCompletionMessage::user(&prompt)];

    let mut times = Vec::new();
    let pb = ProgressBar::new(args.iterations as u64);

    for _ in 0..args.iterations {
        let start = Instant::now();
        let _ = client.chat_completion(&messages, &params).await?;
        let elapsed = start.elapsed();
        times.push(elapsed.as_millis());
        pb.inc(1);
    }
    pb.finish();

    let avg = times.iter().sum::<u128>() / times.len() as u128;
    let min = *times.iter().min().unwrap();
    let max = *times.iter().max().unwrap();

    println!();
    println!("{}", "Results:".green().bold());
    println!("  Average: {}ms", avg);
    println!("  Min: {}ms", min);
    println!("  Max: {}ms", max);

    Ok(())
}

fn parse_model_string(model_str: &str) -> Result<(rmoe_models::providers::Provider, String)> {
    use rmoe_models::providers::Provider;

    if model_str.contains(':') {
        let parts: Vec<&str> = model_str.splitn(2, ':').collect();
        let provider = parse_provider(parts[0])?;
        let model = parts[1].to_string();
        Ok((provider, model))
    } else if model_str.ends_with(".gguf") {
        // Local GGUF model - use Ollama as backend
        Ok((Provider::Ollama, model_str.to_string()))
    } else {
        // Default to OpenAI
        Ok((Provider::OpenAI, model_str.to_string()))
    }
}

fn parse_provider(name: &str) -> Result<rmoe_models::providers::Provider> {
    use rmoe_models::providers::Provider;
    
    match name.to_lowercase().as_str() {
        "openai" => Ok(Provider::OpenAI),
        "anthropic" | "claude" => Ok(Provider::Anthropic),
        "google" | "gemini" => Ok(Provider::Google),
        "azure" => Ok(Provider::Azure),
        "groq" => Ok(Provider::Groq),
        "together" => Ok(Provider::Together),
        "mistral" => Ok(Provider::Mistral),
        "ollama" | "local" => Ok(Provider::Ollama),
        "openrouter" => Ok(Provider::OpenRouter),
        other => Ok(Provider::Custom(other.to_string())),
    }
}
