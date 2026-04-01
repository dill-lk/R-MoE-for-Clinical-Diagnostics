//! DiagnosticEngine and WannaStateMachine implementation.
//!
//! This module is the core of R-MoE, implementing:
//! - WannaStateMachine: Confidence-gated recursion (#wanna# protocol)
//! - DiagnosticEngine: Full pipeline orchestration (3-phase + HITL)

use crate::models::*;
use crate::config::RMoEConfig;
use crate::error::RMoEResult;

use std::time::Instant;
use tracing::{info, warn, debug, instrument};

// ═══════════════════════════════════════════════════════════════════════════════
//  WannaStateMachine
// ═══════════════════════════════════════════════════════════════════════════════

/// Implements the #wanna# protocol (paper §3.2).
///
/// State transitions:
/// - Sc ≥ θ → ProceedToReport
/// - Sc < θ, iter ≤ limit, crop → RequestHighResCrop
/// - Sc < θ, iter ≤ limit, alt → RequestAlternateView
/// - Sc < θ, iter > limit → EscalateToHuman
#[derive(Debug, Clone)]
pub struct WannaStateMachine {
    /// Maximum recursive iterations
    pub hard_limit: usize,
    /// Confidence threshold θ
    pub threshold: f64,
}

impl Default for WannaStateMachine {
    fn default() -> Self {
        Self {
            hard_limit: 3,
            threshold: 0.90,
        }
    }
}

impl WannaStateMachine {
    /// Create a new state machine with custom parameters.
    pub fn new(hard_limit: usize, threshold: f64) -> Self {
        Self { hard_limit, threshold }
    }

    /// Decide the next action based on confidence score and iteration.
    #[instrument(skip(self, reasoning), fields(sc = %sc, iteration = %iteration))]
    pub fn decide(
        &self,
        sc: f64,
        iteration: usize,
        reasoning: Option<&ReasoningOutput>,
    ) -> WannaDecision {
        // Gate passed - proceed to CSR
        if sc >= self.threshold {
            info!(sc, threshold = self.threshold, "Confidence gate PASSED");
            return WannaDecision {
                state: WannaState::ProceedToReport,
                iteration,
                feedback: FeedbackTensor::default(),
            };
        }

        // Hard limit reached - escalate to human
        if iteration >= self.hard_limit {
            warn!(
                sc,
                iteration,
                hard_limit = self.hard_limit,
                "Max iterations reached, escalating to human"
            );
            return WannaDecision {
                state: WannaState::EscalateToHuman,
                iteration,
                feedback: FeedbackTensor::default(),
            };
        }

        // Determine feedback type from ARLL output
        let (state, request_type, payload) = if let Some(r) = reasoning {
            let req = r.feedback_request.to_lowercase();
            if req.contains("alternate") || req.contains("view") {
                (
                    WannaState::RequestAlternateView,
                    r.feedback_request.clone(),
                    r.feedback_payload.clone(),
                )
            } else if req.contains("modality") || req.contains("ct") || req.contains("mri") {
                (
                    WannaState::RequestModalityEscalation,
                    r.feedback_request.clone(),
                    r.feedback_payload.clone(),
                )
            } else {
                (
                    WannaState::RequestHighResCrop,
                    "High-Res Crop".to_string(),
                    r.feedback_payload.clone(),
                )
            }
        } else {
            (
                WannaState::RequestHighResCrop,
                "High-Res Crop".to_string(),
                format!("region=suspicious;zoom=2.0;iteration={}", iteration),
            )
        };

        debug!(
            ?state,
            sc,
            threshold = self.threshold,
            iteration,
            "Confidence gate FAILED, requesting feedback"
        );

        WannaDecision {
            state,
            iteration,
            feedback: FeedbackTensor {
                request_type,
                payload,
            },
        }
    }

    /// Check if a confidence score would pass the gate.
    pub fn would_pass(&self, sc: f64) -> bool {
        sc >= self.threshold
    }

    /// Check if we've exhausted iterations.
    pub fn is_exhausted(&self, iteration: usize) -> bool {
        iteration >= self.hard_limit
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  MPE Confidence Gate
// ═══════════════════════════════════════════════════════════════════════════════

/// Phase-1 pre-filter: Early #wanna# if MPE reports low confidence with no ROIs.
#[derive(Debug, Clone, Default)]
pub struct MPEConfidenceGate;

impl MPEConfidenceGate {
    /// Check if perception evidence passes the pre-filter.
    ///
    /// Returns false if confidence is low AND no ROIs detected.
    pub fn passes(&self, evidence: &PerceptionEvidence) -> bool {
        let level = evidence.confidence_level.to_lowercase();
        if level == "low" && evidence.rois.is_empty() {
            debug!("MPE gate FAILED: low confidence with no ROIs");
            return false;
        }
        true
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Pipeline Orchestrator (Synchronous skeleton)
// ═══════════════════════════════════════════════════════════════════════════════

/// Orchestrates the full R-MoE diagnostic pipeline.
///
/// Pipeline flow:
/// ```text
/// INPUT → MPE → [MPE Gate] → ARLL → [ARLL Gate] → CSR
///                   ↑                    |
///                   └──── #wanna# ←──────┘  (max 3 iter)
///                                ↓
///                           [HITL prompt]
/// ```
pub struct DiagnosticEngine {
    config: RMoEConfig,
    state_machine: WannaStateMachine,
    mpe_gate: MPEConfidenceGate,
}

impl DiagnosticEngine {
    /// Create a new engine with configuration.
    pub fn new(config: RMoEConfig) -> Self {
        let state_machine = WannaStateMachine::new(
            config.pipeline.max_iterations,
            config.pipeline.confidence_threshold,
        );
        Self {
            config,
            state_machine,
            mpe_gate: MPEConfidenceGate,
        }
    }

    /// Get the state machine reference.
    pub fn state_machine(&self) -> &WannaStateMachine {
        &self.state_machine
    }

    /// Get the configuration reference.
    pub fn config(&self) -> &RMoEConfig {
        &self.config
    }

    /// Run the diagnostic pipeline (mock implementation for structure).
    ///
    /// This is a skeleton that shows the pipeline structure.
    /// Real implementation requires agent instances.
    #[instrument(skip(self), fields(image = %image_path))]
    pub async fn run(
        &self,
        image_path: &str,
        prior_image: Option<&str>,
    ) -> RMoEResult<RunSummary> {
        let start = Instant::now();
        let mut summary = RunSummary {
            image_path: image_path.to_string(),
            prior_image_path: prior_image.unwrap_or_default().to_string(),
            model_vision: self.config.models.vision_text.clone(),
            model_reasoning: self.config.models.reasoning.clone(),
            model_clinical: self.config.models.clinical.clone(),
            ..Default::default()
        };

        info!(
            image = image_path,
            threshold = self.config.pipeline.confidence_threshold,
            max_iter = self.config.pipeline.max_iterations,
            "Starting diagnostic pipeline"
        );

        let mut last_reasoning: Option<ReasoningOutput> = None;
        let mut _wanna_feedback = String::new();

        for iteration in 1..=self.state_machine.hard_limit {
            let iter_start = Instant::now();
            info!(iteration, "Starting iteration");

            // Phase 1: MPE (Mock)
            let perception = PerceptionEvidence {
                feature_summary: format!("Mock perception for iteration {}", iteration),
                confidence_level: if iteration == 1 { "medium" } else { "high" }.to_string(),
                ..Default::default()
            };

            // MPE gate check
            if !self.mpe_gate.passes(&perception) && iteration < self.state_machine.hard_limit {
                let trace = IterationTrace {
                    iteration,
                    perception_summary: perception.feature_summary.clone(),
                    decision: "MPEGateFail".to_string(),
                    metrics: UncertaintyMetrics::compute(0.5, &[]),
                    elapsed_ms: iter_start.elapsed().as_millis() as u64,
                    ..Default::default()
                };
                summary.trace.push(trace);
                summary.iterations_executed = iteration;
                _wanna_feedback = format!("High-Res Crop|{}|zoom=2.0", iteration);
                continue;
            }

            // Phase 2: ARLL (Mock)
            let sc = match iteration {
                1 => 0.65,
                2 => 0.82,
                _ => 0.94,
            };
            let ensemble = DDxEnsemble::new(vec![
                DDxHypothesis::new("Pulmonary adenocarcinoma", 0.42 + (iteration as f64 * 0.15), "Spiculated mass"),
                DDxHypothesis::new("Community-acquired pneumonia", 0.31 - (iteration as f64 * 0.08), "Consolidation"),
                DDxHypothesis::new("Pulmonary sarcoidosis", 0.15 - (iteration as f64 * 0.03), "Hilar changes"),
            ]);

            let reasoning = ReasoningOutput {
                cot: format!("Chain-of-thought reasoning for iteration {}", iteration),
                ensemble: ensemble.clone(),
                wanna: sc < self.state_machine.threshold,
                feedback_request: if sc < self.state_machine.threshold {
                    "High-Res Crop".to_string()
                } else {
                    "none".to_string()
                },
                feedback_payload: format!("region=LUL;zoom=2.5;iteration={}", iteration),
                ..Default::default()
            };

            let metrics = UncertaintyMetrics::compute(sc, &ensemble.probabilities());

            let trace = IterationTrace {
                iteration,
                perception_summary: perception.feature_summary.clone(),
                reasoning_summary: reasoning.cot.clone(),
                decision: String::new(), // Will be set below
                metrics,
                ddx_ensemble: ensemble.clone(),
                elapsed_ms: iter_start.elapsed().as_millis() as u64,
                ..Default::default()
            };

            summary.trace.push(trace);
            summary.iterations_executed = iteration;
            last_reasoning = Some(reasoning.clone());

            // State machine decision
            let decision = self.state_machine.decide(sc, iteration, Some(&reasoning));

            // Update trace decision
            if let Some(trace) = summary.trace.last_mut() {
                trace.decision = format!("{:?}", decision.state);
            }

            match decision.state {
                WannaState::ProceedToReport => {
                    info!(sc, iteration, "Gate passed, proceeding to CSR");
                    summary.success = true;
                    break;
                }
                WannaState::EscalateToHuman => {
                    warn!(sc, iteration, "Escalating to human radiologist");
                    summary.escalated_to_human = true;
                    break;
                }
                _ => {
                    debug!(?decision.state, "#wanna# triggered, continuing to next iteration");
                    _wanna_feedback = format!("{}|{}", decision.feedback.request_type, decision.feedback.payload);
                }
            }
        }

        // Phase 3: CSR (Mock)
        if summary.success {
            if let Some(_reasoning) = last_reasoning {
                let report = ClinicalReport {
                    standard: "2C25.0".to_string(),
                    snomed_ct: "254637007".to_string(),
                    risk_stratification: RiskScore {
                        scale: "Lung-RADS".to_string(),
                        score: "4X".to_string(),
                        interpretation: "Suspicious for malignancy".to_string(),
                        action: "Tissue sampling recommended".to_string(),
                    },
                    narrative: "FINDINGS: A 3.2 cm spiculated mass in the left upper lobe...".to_string(),
                    summary: "Suspicious pulmonary nodule, Lung-RADS 4X.".to_string(),
                    treatment_recommendations: "CT-guided biopsy within 1-3 months.".to_string(),
                    hitl_review_required: false,
                    hitl_reason: String::new(),
                };
                summary.final_report_json = serde_json::to_string(&report)?;
            }
        }

        summary.total_elapsed_ms = start.elapsed().as_millis() as u64;

        info!(
            success = summary.success,
            escalated = summary.escalated_to_human,
            iterations = summary.iterations_executed,
            elapsed_ms = summary.total_elapsed_ms,
            "Pipeline completed"
        );

        Ok(summary)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Calibration Tracker
// ═══════════════════════════════════════════════════════════════════════════════

/// Tracks calibration metrics for ECE (Expected Calibration Error).
#[derive(Debug, Clone, Default)]
pub struct CalibrationTracker {
    /// (confidence, was_correct) pairs
    observations: Vec<(f64, bool)>,
}

impl CalibrationTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an observation.
    pub fn update(&mut self, confidence: f64, was_correct: bool) {
        self.observations.push((confidence, was_correct));
    }

    /// Compute ECE with specified number of bins.
    pub fn ece(&self, n_bins: usize) -> f64 {
        if self.observations.is_empty() {
            return 0.0;
        }

        let mut bins: Vec<Vec<(f64, bool)>> = vec![Vec::new(); n_bins];
        
        for &(conf, correct) in &self.observations {
            let bin_idx = ((conf * n_bins as f64) as usize).min(n_bins - 1);
            bins[bin_idx].push((conf, correct));
        }

        let n = self.observations.len() as f64;
        let mut ece = 0.0;

        for bin in &bins {
            if bin.is_empty() {
                continue;
            }
            let bin_size = bin.len() as f64;
            let avg_conf: f64 = bin.iter().map(|(c, _)| c).sum::<f64>() / bin_size;
            let avg_acc: f64 = bin.iter().filter(|(_, c)| *c).count() as f64 / bin_size;
            ece += (bin_size / n) * (avg_conf - avg_acc).abs();
        }

        ece
    }

    /// Get reliability diagram bins.
    pub fn reliability_bins(&self, n_bins: usize) -> Vec<CalibrationBin> {
        let mut bins: Vec<Vec<(f64, bool)>> = vec![Vec::new(); n_bins];
        
        for &(conf, correct) in &self.observations {
            let bin_idx = ((conf * n_bins as f64) as usize).min(n_bins - 1);
            bins[bin_idx].push((conf, correct));
        }

        bins.iter()
            .enumerate()
            .map(|(i, bin)| {
                let lower = i as f64 / n_bins as f64;
                let upper = (i + 1) as f64 / n_bins as f64;
                if bin.is_empty() {
                    CalibrationBin {
                        lower,
                        upper,
                        mean_conf: 0.0,
                        mean_acc: 0.0,
                        count: 0,
                    }
                } else {
                    let count = bin.len();
                    let mean_conf = bin.iter().map(|(c, _)| c).sum::<f64>() / count as f64;
                    let mean_acc = bin.iter().filter(|(_, c)| *c).count() as f64 / count as f64;
                    CalibrationBin {
                        lower,
                        upper,
                        mean_conf,
                        mean_acc,
                        count,
                    }
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wanna_state_machine_pass() {
        let sm = WannaStateMachine::new(3, 0.90);
        let decision = sm.decide(0.95, 1, None);
        assert_eq!(decision.state, WannaState::ProceedToReport);
    }

    #[test]
    fn test_wanna_state_machine_request_crop() {
        let sm = WannaStateMachine::new(3, 0.90);
        let decision = sm.decide(0.75, 1, None);
        assert_eq!(decision.state, WannaState::RequestHighResCrop);
    }

    #[test]
    fn test_wanna_state_machine_escalate() {
        let sm = WannaStateMachine::new(3, 0.90);
        let decision = sm.decide(0.75, 3, None);
        assert_eq!(decision.state, WannaState::EscalateToHuman);
    }

    #[test]
    fn test_mpe_gate_pass() {
        let gate = MPEConfidenceGate;
        let evidence = PerceptionEvidence {
            confidence_level: "medium".to_string(),
            ..Default::default()
        };
        assert!(gate.passes(&evidence));
    }

    #[test]
    fn test_mpe_gate_fail() {
        let gate = MPEConfidenceGate;
        let evidence = PerceptionEvidence {
            confidence_level: "low".to_string(),
            rois: Vec::new(),
            ..Default::default()
        };
        assert!(!gate.passes(&evidence));
    }

    #[test]
    fn test_calibration_ece() {
        let mut tracker = CalibrationTracker::new();
        // Perfect calibration: 90% confident predictions are correct 90% of the time
        for i in 0..10 {
            tracker.update(0.9, i < 9);
        }
        let ece = tracker.ece(10);
        assert!(ece < 0.1, "ECE should be low for well-calibrated predictions");
    }
}
