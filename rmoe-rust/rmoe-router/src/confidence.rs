//! Confidence-based routing.

use super::{RouteDecision, Router, RoutingContext};

/// Confidence-based router that uses #wanna# protocol scores.
pub struct ConfidenceRouter {
    /// Confidence threshold for accepting a route
    threshold: f64,
}

impl ConfidenceRouter {
    pub fn new() -> Self {
        Self { threshold: 0.90 }
    }

    pub fn with_threshold(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Calculate route confidence based on input characteristics.
    fn calculate_confidence(&self, input: &str) -> f64 {
        // Longer, more specific inputs get higher confidence
        let length_factor = (input.len() as f64 / 500.0).min(1.0);
        
        // Medical terminology increases confidence
        let medical_terms = [
            "pain", "symptom", "diagnosis", "treatment", "patient",
            "medical", "clinical", "history", "examination", "finding",
        ];
        let medical_factor = medical_terms
            .iter()
            .filter(|term| input.to_lowercase().contains(*term))
            .count() as f64 / 5.0;

        // Combine factors
        (length_factor * 0.3 + medical_factor * 0.7).min(1.0)
    }
}

impl Router for ConfidenceRouter {
    fn route(&self, input: &str, context: &RoutingContext) -> RouteDecision {
        let confidence = self.calculate_confidence(input);
        
        let target = if confidence >= self.threshold {
            // High confidence - use specialized model
            "specialist".to_string()
        } else if confidence >= 0.7 {
            // Medium confidence - use reasoning model
            "reasoning".to_string()
        } else {
            // Low confidence - use general model
            "general".to_string()
        };

        RouteDecision {
            target,
            confidence,
            reasoning: format!(
                "Confidence score: {:.2} (threshold: {:.2})",
                confidence, self.threshold
            ),
            alternatives: vec!["general".to_string(), "reasoning".to_string()],
        }
    }
}

impl Default for ConfidenceRouter {
    fn default() -> Self {
        Self::new()
    }
}
