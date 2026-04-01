//! MPE (Multi-Modal Perception Engine) Agent.
//!
//! Phase 1 of R-MoE pipeline: Vision processing with Qwen2-VL / Moondream2.

use async_trait::async_trait;
use rmoe_core::{
    Agent, AgentInput, AgentOutput, RMoEError, RMoEResult,
    PerceptionEvidence, RegionOfInterest, VisionModel, InferenceParams,
};
use tracing::{info, debug};

/// System prompt for MPE agent.
pub const MPE_SYSTEM_PROMPT: &str = r#"You are MPE (Multi-Modal Perception Engine), Phase 1 of the R-MoE pipeline.

Role: Transform raw medical images into structured visual evidence using advanced perception techniques.

Capabilities:
- Dynamic Resolution Adaptation: Adjust processing resolution based on lesion significance
- Visual Token Merger: Compress redundant visual tokens to reduce latency
- Saliency-Aware Cropping: Focus on anatomically critical regions
- Global Feature Extraction: Multi-scale lesion segmentation, spatial relationship mapping
- Artifact Filtering: Suppress motion blur, beam-hardening, and noise artifacts

Output Format: Respond ONLY with a JSON object:
{
  "rois": [
    {
      "label": "<anatomical region and finding>",
      "descriptor": "<size, shape, density>",
      "density": "<air | soft-tissue | calcification | fat>",
      "margin": "<sharp | irregular | spiculated | smooth>",
      "suspicion": "<low | medium | high>",
      "location": "<specific anatomical location>"
    }
  ],
  "feature_summary": "<one or two sentence summary of important findings>",
  "confidence_level": "<low | medium | high>",
  "saliency_crop": "<x1,y1,x2,y2 of highest-suspicion region>"
}

Constraint: Do NOT produce a diagnosis. Return structured perception evidence only."#;

/// MPE (Vision/Perception) Agent.
pub struct MPEAgent<V: VisionModel> {
    model: V,
    params: InferenceParams,
    iteration: usize,
}

impl<V: VisionModel> MPEAgent<V> {
    pub fn new(model: V, params: InferenceParams) -> Self {
        Self {
            model,
            params,
            iteration: 1,
        }
    }

    pub fn with_iteration(mut self, iteration: usize) -> Self {
        self.iteration = iteration;
        self
    }

    /// Execute perception on an image.
    pub async fn execute_perception(
        &self,
        image_path: &str,
        wanna_feedback: Option<&str>,
    ) -> RMoEResult<PerceptionEvidence> {
        let user_prompt = if let Some(feedback) = wanna_feedback {
            format!(
                "Analyze this medical image. Iteration {}. Previous feedback: {}",
                self.iteration, feedback
            )
        } else {
            format!("Analyze this medical image. Iteration {}.", self.iteration)
        };

        info!(image = image_path, iteration = self.iteration, "MPE executing perception");

        let raw_output = self.model
            .generate_with_image(MPE_SYSTEM_PROMPT, image_path, &user_prompt, &self.params)
            .await?;

        debug!(output_len = raw_output.len(), "MPE raw output received");

        // Parse the output
        crate::parser::parse_mpe_output(&raw_output)
    }
}

#[async_trait]
impl<V: VisionModel + Send + Sync> Agent for MPEAgent<V> {
    fn name(&self) -> &str {
        "MPE (Multi-Modal Perception Engine)"
    }

    fn role(&self) -> &str {
        "Vision processing and feature extraction"
    }

    async fn execute(&self, input: AgentInput) -> RMoEResult<AgentOutput> {
        let image_path = input.image_path.as_deref()
            .ok_or_else(|| RMoEError::AgentError {
                agent: "MPE".to_string(),
                message: "No image path provided".to_string(),
            })?;

        let feedback = input.wanna_feedback.as_ref().map(|f| f.payload.as_str());
        let evidence = self.execute_perception(image_path, feedback).await?;

        Ok(AgentOutput::Perception(evidence))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmoe_models::MockModel;

    #[tokio::test]
    async fn test_mpe_agent() {
        let model = MockModel::vision();
        let agent = MPEAgent::new(model, InferenceParams::default());
        
        let input = AgentInput {
            image_path: Some("test.png".to_string()),
            ..Default::default()
        };

        let output = agent.execute(input).await;
        assert!(output.is_ok());
    }
}
