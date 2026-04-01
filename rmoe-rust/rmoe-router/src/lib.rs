//! # rmoe-router
//!
//! Smart routing for Mixture-of-Experts model selection.
//!
//! Features:
//! - Keyword-based routing
//! - Confidence-based routing
//! - Load balancing
//! - Fallback logic (local ↔ cloud)

use std::collections::HashMap;
use std::sync::Arc;
use rmoe_core::RMoEResult;
use serde::{Deserialize, Serialize};

pub mod keyword;
pub mod confidence;
pub mod load_balancer;

pub use keyword::*;
pub use confidence::*;
pub use load_balancer::*;

/// Route decision from the router.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteDecision {
    /// Selected model/agent
    pub target: String,
    /// Confidence in the decision
    pub confidence: f64,
    /// Reasoning for the decision
    pub reasoning: String,
    /// Alternative options
    pub alternatives: Vec<String>,
}

/// Router configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Routing strategy
    pub strategy: RoutingStrategy,
    /// Default model when no match
    pub default_model: String,
    /// Enable fallback to local models
    pub enable_local_fallback: bool,
    /// Custom routing rules
    pub rules: Vec<RoutingRule>,
}

/// Routing strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Route based on keywords
    Keyword,
    /// Route based on confidence scores
    Confidence,
    /// Round-robin load balancing
    RoundRobin,
    /// Route based on cost optimization
    CostOptimized,
    /// Route based on latency requirements
    LatencyOptimized,
    /// Use ML-based routing
    Learned,
}

/// A routing rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingRule {
    /// Rule name
    pub name: String,
    /// Condition (e.g., "contains:pain")
    pub condition: String,
    /// Target model/agent
    pub target: String,
    /// Priority (higher = more important)
    pub priority: i32,
}

/// Trait for routing implementations.
pub trait Router: Send + Sync {
    /// Route a query to the appropriate model/agent.
    fn route(&self, input: &str, context: &RoutingContext) -> RouteDecision;
}

/// Context for routing decisions.
#[derive(Debug, Clone, Default)]
pub struct RoutingContext {
    /// Previous model used
    pub previous_model: Option<String>,
    /// Session type (e.g., "diagnostic", "chat")
    pub session_type: String,
    /// User preferences
    pub preferences: HashMap<String, String>,
    /// Available models
    pub available_models: Vec<String>,
    /// Current load per model
    pub model_load: HashMap<String, f64>,
}

/// Smart router combining multiple strategies.
pub struct SmartRouter {
    config: RouterConfig,
    keyword_router: KeywordRouter,
    confidence_router: ConfidenceRouter,
    load_balancer: LoadBalancer,
}

impl SmartRouter {
    pub fn new(config: RouterConfig) -> Self {
        Self {
            keyword_router: KeywordRouter::new(config.rules.clone()),
            confidence_router: ConfidenceRouter::new(),
            load_balancer: LoadBalancer::new(),
            config,
        }
    }

    pub fn with_default_rules() -> Self {
        let rules = vec![
            RoutingRule {
                name: "medical_vision".to_string(),
                condition: "contains:image,x-ray,ct,mri,scan,radiology".to_string(),
                target: "vision".to_string(),
                priority: 10,
            },
            RoutingRule {
                name: "cardiology".to_string(),
                condition: "contains:heart,chest,cardiac,cardio,ecg,ekg".to_string(),
                target: "cardiology".to_string(),
                priority: 8,
            },
            RoutingRule {
                name: "neurology".to_string(),
                condition: "contains:brain,neuro,headache,seizure,stroke".to_string(),
                target: "neurology".to_string(),
                priority: 8,
            },
            RoutingRule {
                name: "general_reasoning".to_string(),
                condition: "contains:diagnose,differential,analyze".to_string(),
                target: "reasoning".to_string(),
                priority: 5,
            },
        ];

        let config = RouterConfig {
            strategy: RoutingStrategy::Keyword,
            default_model: "general".to_string(),
            enable_local_fallback: true,
            rules,
        };

        Self::new(config)
    }
}

impl Router for SmartRouter {
    fn route(&self, input: &str, context: &RoutingContext) -> RouteDecision {
        match self.config.strategy {
            RoutingStrategy::Keyword => self.keyword_router.route(input, context),
            RoutingStrategy::Confidence => self.confidence_router.route(input, context),
            RoutingStrategy::RoundRobin => self.load_balancer.route(input, context),
            _ => {
                // Default to keyword routing
                self.keyword_router.route(input, context)
            }
        }
    }
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            strategy: RoutingStrategy::Keyword,
            default_model: "general".to_string(),
            enable_local_fallback: true,
            rules: vec![],
        }
    }
}
