//! ARLL (Agentic Reasoning & Logic Layer) Agent.
//!
//! Phase 2 of R-MoE pipeline: Clinical reasoning with DeepSeek-R1.

use async_trait::async_trait;
use rmoe_core::{
    Agent, AgentInput, AgentOutput, RMoEError, RMoEResult,
    ReasoningOutput, DDxEnsemble, TextModel, InferenceParams,
};
use tracing::{info, debug};

/// System prompt for ARLL agent.
pub const ARLL_SYSTEM_PROMPT: &str = r#"You are ARLL (Agentic Reasoning & Logic Layer), Phase 2 of the R-MoE pipeline.
Model: DeepSeek-R1 (Chain-of-Thought Reasoning Agent)

Role: Apply deep clinical reasoning to MPE visual evidence using step-by-step CoT logic.

Capabilities:
- Chain-of-Thought (CoT): Reason step-by-step through the patient's condition
- Differential Diagnosis (DDx): Generate probability distribution over candidates
- Confidence Scoring: Compute Sc = 1 - sigma^2, where sigma^2 is DDx variance
- Vector RAG: Cross-reference against MIMIC-CXR, RSNA, clinical guidelines

#wanna# Protocol:
If Sc < 0.90, do NOT guess. Set "wanna" to true and specify feedback request:
- "High-Res Crop": region=<anatomical_region>;zoom=<factor>
- "Alternate View": region=<anatomical_region>;angle=<projection>

CRITICAL: In "ddx" array, "diagnosis" MUST be a medical condition name.

Output Format - respond ONLY with this JSON:
{
  "cot": "<full step-by-step reasoning trace>",
  "ddx": [
    {"diagnosis": "<primary diagnosis>", "probability": <0.0-1.0>, "evidence": "<supporting findings>"},
    {"diagnosis": "<alternate 1>", "probability": <0.0-1.0>, "evidence": "<findings>"},
    {"diagnosis": "<alternate 2>", "probability": <0.0-1.0>, "evidence": "<findings>"}
  ],
  "sigma2": <variance>,
  "sc": <1 - sigma2>,
  "wanna": <true if Sc < 0.90>,
  "feedback_request": "<High-Res Crop|Alternate View|null>",
  "feedback_payload": "<region=...;zoom=...|null>",
  "rag_references": ["<reference 1>", "<reference 2>"],
  "temporal_note": "<interval change vs prior, or null>"
}"#;

/// ARLL (Reasoning) Agent.
pub struct ARLLAgent<T: TextModel> {
    model: T,
    params: InferenceParams,
    iteration: usize,
    threshold: f64,
}

impl<T: TextModel> ARLLAgent<T> {
    pub fn new(model: T, params: InferenceParams, threshold: f64) -> Self {
        Self {
            model,
            params,
            iteration: 1,
            threshold,
        }
    }

    pub fn with_iteration(mut self, iteration: usize) -> Self {
        self.iteration = iteration;
        self
    }

    /// Execute reasoning on perception evidence.
    pub async fn execute_reasoning(
        &self,
        mpe_evidence: &str,
        prior_context: Option<&str>,
        rag_refs: &[String],
    ) -> RMoEResult<ReasoningOutput> {
        let mut user_prompt = format!(
            "MPE Evidence (Iteration {}):\n{}\n",
            self.iteration, mpe_evidence
        );

        if let Some(prior) = prior_context {
            user_prompt.push_str(&format!("\nPrior Context:\n{}\n", prior));
        }

        if !rag_refs.is_empty() {
            user_prompt.push_str("\nRAG References:\n");
            for ref_str in rag_refs {
                user_prompt.push_str(&format!("- {}\n", ref_str));
            }
        }

        user_prompt.push_str(&format!(
            "\nConfidence threshold θ = {:.2}. Compute Sc and decide if #wanna# needed.",
            self.threshold
        ));

        info!(iteration = self.iteration, "ARLL executing reasoning");

        let raw_output = self.model
            .generate(ARLL_SYSTEM_PROMPT, &user_prompt, &self.params)
            .await?;

        debug!(output_len = raw_output.len(), "ARLL raw output received");

        crate::parser::parse_arll_output(&raw_output)
    }
}

#[async_trait]
impl<T: TextModel + Send + Sync> Agent for ARLLAgent<T> {
    fn name(&self) -> &str {
        "ARLL (Agentic Reasoning & Logic Layer)"
    }

    fn role(&self) -> &str {
        "Clinical reasoning and differential diagnosis"
    }

    async fn execute(&self, input: AgentInput) -> RMoEResult<AgentOutput> {
        let evidence = &input.context;
        let prior = input.prior_context.as_deref();
        
        let reasoning = self.execute_reasoning(evidence, prior, &input.rag_references).await?;
        Ok(AgentOutput::Reasoning(reasoning))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmoe_models::MockModel;

    #[tokio::test]
    async fn test_arll_agent() {
        let model = MockModel::reasoning();
        let agent = ARLLAgent::new(model, InferenceParams::default(), 0.90);
        
        let input = AgentInput {
            context: "Mock perception evidence".to_string(),
            ..Default::default()
        };

        let output = agent.execute(input).await;
        assert!(output.is_ok());
    }
}
