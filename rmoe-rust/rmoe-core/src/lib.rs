//! # rmoe-core
//!
//! Core types, traits, and engine for the R-MoE (Recursive Multi-Agent
//! Mixture-of-Experts) framework.
//!
//! This crate provides:
//! - Data structures for differential diagnosis (DDx), confidence scoring
//! - The WannaStateMachine for confidence-gated recursion
//! - Traits for models, agents, and routing
//! - The DiagnosticEngine orchestrator

pub mod models;
pub mod engine;
pub mod traits;
pub mod error;
pub mod config;

pub use models::*;
pub use engine::*;
pub use traits::*;
pub use error::*;
pub use config::*;
