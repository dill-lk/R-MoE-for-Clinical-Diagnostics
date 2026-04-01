//! Load balancing for model routing.

use std::sync::atomic::{AtomicUsize, Ordering};
use super::{RouteDecision, Router, RoutingContext};

/// Round-robin load balancer.
pub struct LoadBalancer {
    /// Current index for round-robin
    current_index: AtomicUsize,
    /// Available models
    models: Vec<String>,
}

impl LoadBalancer {
    pub fn new() -> Self {
        Self {
            current_index: AtomicUsize::new(0),
            models: vec![
                "openai:gpt-4o".to_string(),
                "anthropic:claude-sonnet-4-20250514".to_string(),
                "google:gemini-1.5-pro".to_string(),
            ],
        }
    }

    pub fn with_models(models: Vec<String>) -> Self {
        Self {
            current_index: AtomicUsize::new(0),
            models,
        }
    }

    /// Add a model to the pool.
    pub fn add_model(&mut self, model: String) {
        if !self.models.contains(&model) {
            self.models.push(model);
        }
    }

    /// Remove a model from the pool.
    pub fn remove_model(&mut self, model: &str) {
        self.models.retain(|m| m != model);
    }
}

impl Router for LoadBalancer {
    fn route(&self, _input: &str, context: &RoutingContext) -> RouteDecision {
        let models = if context.available_models.is_empty() {
            &self.models
        } else {
            &context.available_models
        };

        if models.is_empty() {
            return RouteDecision {
                target: "default".to_string(),
                confidence: 0.5,
                reasoning: "No models available".to_string(),
                alternatives: vec![],
            };
        }

        let index = self.current_index.fetch_add(1, Ordering::SeqCst) % models.len();
        let target = models[index].clone();

        let alternatives: Vec<String> = models
            .iter()
            .filter(|m| **m != target)
            .take(2)
            .cloned()
            .collect();

        RouteDecision {
            target,
            confidence: 1.0 / models.len() as f64,
            reasoning: "Round-robin load balancing".to_string(),
            alternatives,
        }
    }
}

impl Default for LoadBalancer {
    fn default() -> Self {
        Self::new()
    }
}
