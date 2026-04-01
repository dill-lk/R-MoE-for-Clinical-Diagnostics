//! Keyword-based routing.

use super::{RouteDecision, Router, RoutingContext, RoutingRule};

/// Keyword-based router.
pub struct KeywordRouter {
    rules: Vec<RoutingRule>,
}

impl KeywordRouter {
    pub fn new(rules: Vec<RoutingRule>) -> Self {
        let mut sorted_rules = rules;
        sorted_rules.sort_by(|a, b| b.priority.cmp(&a.priority));
        Self { rules: sorted_rules }
    }

    fn parse_condition(&self, condition: &str) -> Option<Vec<String>> {
        if condition.starts_with("contains:") {
            let keywords = condition["contains:".len()..]
                .split(',')
                .map(|s| s.trim().to_lowercase())
                .collect();
            Some(keywords)
        } else {
            None
        }
    }

    fn matches_rule(&self, input: &str, rule: &RoutingRule) -> bool {
        if let Some(keywords) = self.parse_condition(&rule.condition) {
            let input_lower = input.to_lowercase();
            keywords.iter().any(|kw| input_lower.contains(kw))
        } else {
            false
        }
    }
}

impl Router for KeywordRouter {
    fn route(&self, input: &str, _context: &RoutingContext) -> RouteDecision {
        let mut matched_rules: Vec<&RoutingRule> = self.rules
            .iter()
            .filter(|rule| self.matches_rule(input, rule))
            .collect();

        if let Some(best_rule) = matched_rules.first() {
            let alternatives: Vec<String> = matched_rules
                .iter()
                .skip(1)
                .take(3)
                .map(|r| r.target.clone())
                .collect();

            RouteDecision {
                target: best_rule.target.clone(),
                confidence: 0.8,
                reasoning: format!("Matched rule: {}", best_rule.name),
                alternatives,
            }
        } else {
            RouteDecision {
                target: "general".to_string(),
                confidence: 0.5,
                reasoning: "No specific rules matched, using default".to_string(),
                alternatives: vec![],
            }
        }
    }
}

impl Default for KeywordRouter {
    fn default() -> Self {
        Self::new(vec![])
    }
}
