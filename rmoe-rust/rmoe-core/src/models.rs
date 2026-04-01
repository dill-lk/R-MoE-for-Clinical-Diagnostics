//! Core data models for R-MoE framework.
//!
//! Ported from Python `rmoe/models.py` with Rust idioms.
//! All structures support serialization for audit trails and API responses.

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

// ═══════════════════════════════════════════════════════════════════════════════
//  Enumerations
// ═══════════════════════════════════════════════════════════════════════════════

/// Possible outcomes of the ARLL confidence gate (#wanna# protocol).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum WannaState {
    /// Sc ≥ θ → proceed to CSR phase
    ProceedToReport,
    /// Sc < θ, iter < limit → request high-resolution crop
    RequestHighResCrop,
    /// Sc < θ, iter < limit → request alternate imaging view
    RequestAlternateView,
    /// Sc < θ, iter < limit → request different modality (CXR→CT→MRI)
    RequestModalityEscalation,
    /// iter == hard limit → escalate to human radiologist
    EscalateToHuman,
}

impl Default for WannaState {
    fn default() -> Self {
        Self::ProceedToReport
    }
}

/// Human-in-the-Loop interaction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HITLMode {
    /// Always prompt the clinician
    Interactive,
    /// Prompt only in TTY sessions
    Auto,
    /// Fully autonomous (no prompts)
    Disabled,
}

impl Default for HITLMode {
    fn default() -> Self {
        Self::Auto
    }
}

/// Which expert to query in post-diagnosis Q&A.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExpertTarget {
    /// ARLL / DeepSeek-R1-Distill (reasoning)
    Reasoning,
    /// CSR / MedGemma-2B (clinical synthesis)
    Clinical,
    /// MPE / Qwen2-VL (vision/perception)
    Vision,
}

/// Imaging modality for escalation routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Modality {
    CXR,    // Chest X-Ray
    CT,     // Computed Tomography
    MRI,    // Magnetic Resonance Imaging
    PET,    // Positron Emission Tomography
    US,     // Ultrasound
    XRay,   // General X-Ray
    DEXA,   // Bone Density Scan
}

impl std::fmt::Display for Modality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Modality::CXR => write!(f, "CXR"),
            Modality::CT => write!(f, "CT"),
            Modality::MRI => write!(f, "MRI"),
            Modality::PET => write!(f, "PET-CT"),
            Modality::US => write!(f, "Ultrasound"),
            Modality::XRay => write!(f, "X-Ray"),
            Modality::DEXA => write!(f, "DEXA"),
        }
    }
}

/// Escalation urgency level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EscalationUrgency {
    Routine,
    Urgent,
    Emergent,
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Inference configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Hyper-parameters for model inference.
/// Defaults tuned for T4 (16 GB) with 2B-class quantised models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceParams {
    /// KV-cache context window
    pub n_ctx: usize,
    /// CPU decode threads
    pub n_threads: usize,
    /// CPU prompt-eval threads
    pub n_threads_batch: usize,
    /// Generation budget per inference call
    pub max_new_tokens: usize,
    /// Sampling temperature (paper §4.2: 0.2 for clinical precision)
    pub temperature: f32,
    /// Top-K sampling
    pub top_k: usize,
    /// Nucleus sampling threshold
    pub top_p: f32,
    /// Repetition penalty
    pub repeat_penalty: f32,
    /// Penalty context window
    pub penalty_last_n: usize,
    /// GPU layers to offload (-1 = all)
    pub n_gpu_layers: i32,
}

impl Default for InferenceParams {
    fn default() -> Self {
        Self {
            n_ctx: 2048,
            n_threads: 4,
            n_threads_batch: 4,
            max_new_tokens: 512,
            temperature: 0.2,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.1,
            penalty_last_n: 64,
            n_gpu_layers: -1,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Differential Diagnosis (DDx) types
// ═══════════════════════════════════════════════════════════════════════════════

/// Single candidate diagnosis with probability and evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DDxHypothesis {
    /// Medical condition name (e.g., "Pulmonary adenocarcinoma")
    pub diagnosis: String,
    /// Probability mass p ∈ [0, 1]
    pub probability: f64,
    /// Supporting evidence from imaging findings
    pub evidence: String,
    /// ICD-11 code if available
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icd11: Option<String>,
    /// SNOMED CT concept ID if available
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snomed_ct: Option<String>,
}

impl DDxHypothesis {
    pub fn new(diagnosis: impl Into<String>, probability: f64, evidence: impl Into<String>) -> Self {
        Self {
            diagnosis: diagnosis.into(),
            probability,
            evidence: evidence.into(),
            icd11: None,
            snomed_ct: None,
        }
    }

    pub fn with_codes(mut self, icd11: impl Into<String>, snomed_ct: impl Into<String>) -> Self {
        self.icd11 = Some(icd11.into());
        self.snomed_ct = Some(snomed_ct.into());
        self
    }
}

/// Collection of DDx hypotheses from the ARLL agent.
///
/// Confidence score (paper §3.1):
///   Sc = 1 − σ²
///   σ² = Var(p₁ … pₙ) over the DDx probability distribution.
///
/// When the model is highly confident, all probability mass sits on one
/// diagnosis → σ² ≈ 0 → Sc ≈ 1.
/// When mass is spread across many diagnoses → σ² large → Sc small.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DDxEnsemble {
    pub hypotheses: Vec<DDxHypothesis>,
}

impl DDxEnsemble {
    pub fn new(hypotheses: Vec<DDxHypothesis>) -> Self {
        Self { hypotheses }
    }

    /// Get all probabilities as a vector.
    pub fn probabilities(&self) -> Vec<f64> {
        self.hypotheses.iter().map(|h| h.probability).collect()
    }

    /// Variance σ² of the DDx probability distribution.
    pub fn sigma2(&self) -> f64 {
        let probs = self.probabilities();
        if probs.is_empty() {
            return 1.0;
        }
        let mu: f64 = probs.iter().sum::<f64>() / probs.len() as f64;
        probs.iter().map(|p| (p - mu).powi(2)).sum::<f64>() / probs.len() as f64
    }

    /// Confidence score Sc = 1 − σ² ∈ [0, 1].
    pub fn sc(&self) -> f64 {
        (1.0 - self.sigma2()).clamp(0.0, 1.0)
    }

    /// Hypothesis with highest probability mass.
    pub fn primary(&self) -> Option<&DDxHypothesis> {
        self.hypotheses.iter().max_by(|a, b| {
            a.probability.partial_cmp(&b.probability).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Check if confidence exceeds threshold.
    pub fn is_confident(&self, threshold: f64) -> bool {
        self.sc() >= threshold
    }

    /// Shannon entropy H(P) of the DDx distribution (nats).
    pub fn entropy(&self) -> f64 {
        self.probabilities()
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum()
    }

    /// Top N hypotheses by probability.
    pub fn top_n(&self, n: usize) -> Vec<&DDxHypothesis> {
        let mut sorted: Vec<_> = self.hypotheses.iter().collect();
        sorted.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap_or(std::cmp::Ordering::Equal));
        sorted.into_iter().take(n).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Per-phase outputs
// ═══════════════════════════════════════════════════════════════════════════════

/// Region of interest detected by MPE.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionOfInterest {
    /// Anatomical region and finding label
    pub label: String,
    /// Size, shape, density description
    pub descriptor: String,
    /// Density type: air | soft-tissue | calcification | fat
    pub density: String,
    /// Margin type: sharp | irregular | spiculated | smooth
    pub margin: String,
    /// Suspicion level: low | medium | high
    pub suspicion: String,
    /// Specific anatomical location
    pub location: String,
}

/// Structured output from MPE Phase 1 (Qwen2-VL / Moondream2).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerceptionEvidence {
    /// Detected regions of interest
    pub rois: Vec<RegionOfInterest>,
    /// Plain-language summary of findings
    pub feature_summary: String,
    /// Confidence level: "low" | "medium" | "high"
    pub confidence_level: String,
    /// Bounding box of highest-suspicion region ("x1,y1,x2,y2")
    pub saliency_crop: String,
    /// Full model output (for audit)
    pub raw_summary: String,
}

/// Structured output from ARLL Phase 2 (DeepSeek-R1).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReasoningOutput {
    /// Full chain-of-thought trace
    pub cot: String,
    /// DDx ensemble with probabilities
    pub ensemble: DDxEnsemble,
    /// True if #wanna# protocol triggered
    pub wanna: bool,
    /// Feedback request type: "High-Res Crop" | "Alternate View" | "none"
    pub feedback_request: String,
    /// Feedback payload: "region=...;zoom=..." etc.
    pub feedback_payload: String,
    /// RAG references from knowledge base
    pub rag_references: Vec<String>,
    /// Interval change notes vs prior scan
    pub temporal_note: String,
    /// Full model output (for audit)
    pub raw_output: String,
}

/// Compact feedback returned to MPE by the #wanna# protocol.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FeedbackTensor {
    pub request_type: String,
    pub payload: String,
}

/// Risk stratification result.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RiskScore {
    /// Scale name: Lung-RADS | TIRADS | BI-RADS | LI-RADS | PI-RADS
    pub scale: String,
    /// Score value: e.g., "4X", "TR5", "6"
    pub score: String,
    /// Brief interpretation
    pub interpretation: String,
    /// Recommended clinical action
    pub action: String,
}

/// Clinical entity extracted from text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalEntity {
    /// Entity type: diagnosis | measurement | risk_factor | finding
    pub entity_type: String,
    /// Raw text
    pub text: String,
    /// ICD-11 code
    pub icd11: String,
    /// SNOMED CT concept ID
    pub snomed_ct: String,
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Uncertainty metrics
// ═══════════════════════════════════════════════════════════════════════════════

/// All uncertainty quantities for a single iteration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UncertaintyMetrics {
    /// Confidence score Sc = 1 - σ²
    pub confidence: f64,
    /// Uncertainty = 1 - Sc
    pub uncertainty: f64,
    /// H(Sc) binary entropy approximation
    pub predictive_entropy: f64,
    /// DDx variance σ²
    pub ddx_variance: f64,
    /// Shannon entropy of DDx distribution
    pub ddx_entropy: f64,
}

impl UncertaintyMetrics {
    /// Compute uncertainty metrics from confidence score and DDx probabilities.
    pub fn compute(sc: f64, probs: &[f64]) -> Self {
        let ddx_variance = if probs.is_empty() {
            1.0
        } else {
            let mu: f64 = probs.iter().sum::<f64>() / probs.len() as f64;
            probs.iter().map(|p| (p - mu).powi(2)).sum::<f64>() / probs.len() as f64
        };

        let ddx_entropy: f64 = probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();

        // Binary entropy approximation for predictive entropy
        let predictive_entropy = if sc > 0.0 && sc < 1.0 {
            -sc * sc.ln() - (1.0 - sc) * (1.0 - sc).ln()
        } else {
            0.0
        };

        Self {
            confidence: sc,
            uncertainty: 1.0 - sc,
            predictive_entropy,
            ddx_variance,
            ddx_entropy,
        }
    }
}

/// One bin of the ECE reliability diagram.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CalibrationBin {
    pub lower: f64,
    pub upper: f64,
    pub mean_conf: f64,
    pub mean_acc: f64,
    pub count: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Pipeline trace & summary
// ═══════════════════════════════════════════════════════════════════════════════

/// Full record of a single pipeline iteration (for audit & visualization).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationTrace {
    pub iteration: usize,
    pub perception_summary: String,
    pub reasoning_summary: String,
    pub decision: String,
    pub metrics: UncertaintyMetrics,
    pub ddx_ensemble: DDxEnsemble,
    pub rag_references: Vec<String>,
    pub temporal_note: String,
    pub doctor_feedback: String,
    #[serde(with = "chrono::serde::ts_seconds")]
    pub timestamp: DateTime<Utc>,
    pub elapsed_ms: u64,
}

impl Default for IterationTrace {
    fn default() -> Self {
        Self {
            iteration: 1,
            perception_summary: String::new(),
            reasoning_summary: String::new(),
            decision: String::new(),
            metrics: UncertaintyMetrics::default(),
            ddx_ensemble: DDxEnsemble::default(),
            rag_references: Vec::new(),
            temporal_note: String::new(),
            doctor_feedback: String::new(),
            timestamp: Utc::now(),
            elapsed_ms: 0,
        }
    }
}

/// Complete record of a diagnostic run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    /// Unique session identifier
    pub session_id: String,
    /// Diagnosis completed successfully
    pub success: bool,
    /// Escalated to human radiologist
    pub escalated_to_human: bool,
    /// Number of #wanna# iterations executed
    pub iterations_executed: usize,
    /// Final clinical report (JSON string)
    pub final_report_json: String,
    /// Full iteration trace
    pub trace: Vec<IterationTrace>,
    /// Total elapsed time in milliseconds
    pub total_elapsed_ms: u64,
    /// ECE calibration bins
    pub calibration_bins: Vec<CalibrationBin>,
    /// Input image path
    pub image_path: String,
    /// Prior image path (if temporal comparison)
    pub prior_image_path: String,
    /// Vision model used
    pub model_vision: String,
    /// Reasoning model used
    pub model_reasoning: String,
    /// Clinical model used
    pub model_clinical: String,
    /// Session start timestamp
    #[serde(with = "chrono::serde::ts_seconds")]
    pub started_at: DateTime<Utc>,
}

impl Default for RunSummary {
    fn default() -> Self {
        Self {
            session_id: Uuid::new_v4().to_string()[..8].to_string(),
            success: false,
            escalated_to_human: false,
            iterations_executed: 0,
            final_report_json: String::new(),
            trace: Vec::new(),
            total_elapsed_ms: 0,
            calibration_bins: Vec::new(),
            image_path: String::new(),
            prior_image_path: String::new(),
            model_vision: String::new(),
            model_reasoning: String::new(),
            model_clinical: String::new(),
            started_at: Utc::now(),
        }
    }
}

/// Decision emitted by the #wanna# state machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WannaDecision {
    pub state: WannaState,
    pub iteration: usize,
    pub feedback: FeedbackTensor,
}

impl Default for WannaDecision {
    fn default() -> Self {
        Self {
            state: WannaState::ProceedToReport,
            iteration: 1,
            feedback: FeedbackTensor::default(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Doctor-in-the-loop
// ═══════════════════════════════════════════════════════════════════════════════

/// Input received from the clinician during HITL interaction.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DoctorFeedback {
    /// Clinician's message
    pub message: String,
    /// Zoom region specification
    pub zoom_region: String,
    /// True if this is a zoom command
    pub is_zoom_command: bool,
    /// Raw input string
    pub raw_input: String,
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Model configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Paths to all model files and inference hyper-parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSettings {
    /// CLIP mmproj file path
    pub vision_projection_model: String,
    /// Vision text backbone path (Moondream2 / Qwen2-VL)
    pub vision_text_model: String,
    /// Reasoning model path (DeepSeek-R1-Distill)
    pub reasoning_model: String,
    /// Clinical model path (MedGemma-2B)
    pub clinical_model: String,
    /// Current imaging modality
    pub modality: Modality,
    /// Inference parameters
    pub inference: InferenceParams,
}

impl Default for ModelSettings {
    fn default() -> Self {
        Self {
            vision_projection_model: "models/vision_proj.gguf".to_string(),
            vision_text_model: "models/vision_text.gguf".to_string(),
            reasoning_model: "models/reasoning_expert.gguf".to_string(),
            clinical_model: "models/clinical_expert.gguf".to_string(),
            modality: Modality::CXR,
            inference: InferenceParams::default(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Clinical Report
// ═══════════════════════════════════════════════════════════════════════════════

/// Structured clinical report from CSR phase.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClinicalReport {
    /// ICD-11 code
    pub standard: String,
    /// SNOMED CT concept ID
    pub snomed_ct: String,
    /// Risk stratification details
    pub risk_stratification: RiskScore,
    /// Full structured narrative
    pub narrative: String,
    /// One-sentence impression
    pub summary: String,
    /// Treatment recommendations
    pub treatment_recommendations: String,
    /// Whether HITL review is required
    pub hitl_review_required: bool,
    /// Reason for HITL review (if required)
    pub hitl_reason: String,
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Chat / Q&A types
// ═══════════════════════════════════════════════════════════════════════════════

/// Role in a chat conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

/// Single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
            image_url: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
            image_url: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
            image_url: None,
        }
    }

    pub fn user_with_image(content: impl Into<String>, image_url: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
            image_url: Some(image_url.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ddx_ensemble_confidence() {
        // High confidence: all mass on one diagnosis
        let ensemble = DDxEnsemble::new(vec![
            DDxHypothesis::new("Diagnosis A", 0.95, "Strong evidence"),
            DDxHypothesis::new("Diagnosis B", 0.03, "Weak evidence"),
            DDxHypothesis::new("Diagnosis C", 0.02, "Minimal evidence"),
        ]);
        assert!(ensemble.sc() > 0.8);
        assert!(ensemble.is_confident(0.8));

        // Low confidence: spread across diagnoses
        let ensemble = DDxEnsemble::new(vec![
            DDxHypothesis::new("Diagnosis A", 0.35, "Some evidence"),
            DDxHypothesis::new("Diagnosis B", 0.35, "Some evidence"),
            DDxHypothesis::new("Diagnosis C", 0.30, "Some evidence"),
        ]);
        assert!(ensemble.sc() < 0.9);
        assert!(!ensemble.is_confident(0.9));
    }

    #[test]
    fn test_ddx_primary() {
        let ensemble = DDxEnsemble::new(vec![
            DDxHypothesis::new("Pneumonia", 0.60, "Consolidation"),
            DDxHypothesis::new("TB", 0.25, "Upper lobe"),
            DDxHypothesis::new("Cancer", 0.15, "Mass"),
        ]);
        let primary = ensemble.primary().unwrap();
        assert_eq!(primary.diagnosis, "Pneumonia");
        assert_eq!(primary.probability, 0.60);
    }

    #[test]
    fn test_uncertainty_metrics() {
        let metrics = UncertaintyMetrics::compute(0.85, &[0.7, 0.2, 0.1]);
        assert!(metrics.confidence > 0.8);
        assert!(metrics.uncertainty < 0.2);
        assert!(metrics.ddx_entropy > 0.0);
    }
}
