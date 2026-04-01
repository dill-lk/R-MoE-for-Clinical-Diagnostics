//! CSR (Clinical Synthesis & Reporting) Agent.
//!
//! Phase 3 of R-MoE pipeline: Report generation with MedGemma-2B.

use async_trait::async_trait;
use rmoe_core::{
    Agent, AgentInput, AgentOutput, RMoEError, RMoEResult,
    ClinicalReport, ReasoningOutput, TextModel, InferenceParams,
};
use tracing::{info, debug};

/// System prompt for CSR agent.
pub const CSR_SYSTEM_PROMPT: &str = r#"You are CSR (Clinical Synthesis & Reporting), Phase 3 of the R-MoE pipeline.

Role: Convert validated ARLL reasoning into a professional, standards-compliant clinical report.

Capabilities:
- ICD-11 & SNOMED CT Classification: Apply standardized diagnostic coding
- Risk Stratification: Apply Lung-RADS, TIRADS, BI-RADS, LI-RADS, PI-RADS as appropriate
- Professional Clinical Narrative: Generate report-ready prose in formal radiological terminology
- Treatment Recommendations: Provide evidence-based follow-up protocols

Output Format - respond ONLY with this JSON:
{
  "standard": "<ICD-11 code>",
  "snomed_ct": "<SNOMED CT concept ID>",
  "risk_stratification": {
    "scale": "<Lung-RADS|TIRADS|BI-RADS|LI-RADS|N/A>",
    "score": "<score value>",
    "interpretation": "<brief interpretation>",
    "action": "<recommended clinical action>"
  },
  "narrative": "<full structured report: Clinical History / Technique / Findings / Impression>",
  "summary": "<one-sentence impression>",
  "treatment_recommendations": "<evidence-based follow-up and intervention guidance>",
  "hitl_review_required": <true|false>,
  "hitl_reason": "<reason if required, else empty string>"
}

Constraint: If evidence insufficient after all iterations, set hitl_review_required=true."#;

/// CSR (Clinical Reporting) Agent.
pub struct CSRAgent<T: TextModel> {
    model: T,
    params: InferenceParams,
}

impl<T: TextModel> CSRAgent<T> {
    pub fn new(model: T, params: InferenceParams) -> Self {
        Self { model, params }
    }

    /// Generate clinical report from reasoning output.
    pub async fn generate_report(
        &self,
        reasoning: &ReasoningOutput,
        iterations_used: usize,
    ) -> RMoEResult<ClinicalReport> {
        let user_prompt = format!(
            "ARLL Reasoning Output (after {} iterations):\n\
            Chain-of-Thought: {}\n\n\
            DDx Ensemble:\n{}\n\n\
            Confidence (Sc): {:.4}\n\
            RAG References: {:?}\n\
            Temporal Note: {}\n\n\
            Generate the final clinical report.",
            iterations_used,
            reasoning.cot,
            serde_json::to_string_pretty(&reasoning.ensemble).unwrap_or_default(),
            reasoning.ensemble.sc(),
            reasoning.rag_references,
            reasoning.temporal_note
        );

        info!(iterations = iterations_used, "CSR generating clinical report");

        let raw_output = self.model
            .generate(CSR_SYSTEM_PROMPT, &user_prompt, &self.params)
            .await?;

        debug!(output_len = raw_output.len(), "CSR raw output received");

        crate::parser::parse_csr_output(&raw_output)
    }
}

#[async_trait]
impl<T: TextModel + Send + Sync> Agent for CSRAgent<T> {
    fn name(&self) -> &str {
        "CSR (Clinical Synthesis & Reporting)"
    }

    fn role(&self) -> &str {
        "Clinical report generation with ICD-11/SNOMED coding"
    }

    async fn execute(&self, input: AgentInput) -> RMoEResult<AgentOutput> {
        // Parse reasoning from context (assumes JSON)
        let reasoning: ReasoningOutput = serde_json::from_str(&input.context)
            .map_err(|e| RMoEError::ParseError(format!("Failed to parse reasoning: {}", e)))?;

        let iterations = input.metadata.get("iterations")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        let report = self.generate_report(&reasoning, iterations).await?;
        Ok(AgentOutput::Report(report))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmoe_models::MockModel;

    #[tokio::test]
    async fn test_csr_agent() {
        let model = MockModel::clinical();
        let agent = CSRAgent::new(model, InferenceParams::default());
        
        let mock_reasoning = MockModel::reasoning().mock_reasoning(3);
        let report = agent.generate_report(&mock_reasoning, 3).await;
        assert!(report.is_ok());
    }
}
