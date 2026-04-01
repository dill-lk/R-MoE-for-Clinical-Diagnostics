//! Output parsers for agent responses.
//!
//! Handles JSON extraction and fallback regex parsing.

use rmoe_core::{
    RMoEError, RMoEResult,
    PerceptionEvidence, RegionOfInterest,
    ReasoningOutput, DDxEnsemble, DDxHypothesis,
    ClinicalReport, RiskScore,
};
use regex::Regex;
use tracing::{debug, warn};

/// Extract the first complete JSON object from text.
pub fn extract_json_block(text: &str) -> Option<serde_json::Value> {
    let mut depth = 0;
    let mut start: Option<usize> = None;

    for (i, ch) in text.char_indices() {
        match ch {
            '{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start {
                        if let Ok(val) = serde_json::from_str(&text[s..=i]) {
                            return Some(val);
                        }
                        start = None;
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// Parse MPE (perception) output.
pub fn parse_mpe_output(raw: &str) -> RMoEResult<PerceptionEvidence> {
    // Try JSON extraction first
    if let Some(json) = extract_json_block(raw) {
        if let Ok(evidence) = serde_json::from_value::<PerceptionEvidence>(json.clone()) {
            debug!("MPE output parsed from JSON");
            return Ok(evidence);
        }

        // Try partial extraction
        let mut evidence = PerceptionEvidence {
            raw_summary: raw.to_string(),
            ..Default::default()
        };

        if let Some(summary) = json.get("feature_summary").and_then(|v| v.as_str()) {
            evidence.feature_summary = summary.to_string();
        }
        if let Some(level) = json.get("confidence_level").and_then(|v| v.as_str()) {
            evidence.confidence_level = level.to_string();
        }
        if let Some(crop) = json.get("saliency_crop").and_then(|v| v.as_str()) {
            evidence.saliency_crop = crop.to_string();
        }
        if let Some(rois) = json.get("rois").and_then(|v| v.as_array()) {
            evidence.rois = rois.iter()
                .filter_map(|r| serde_json::from_value(r.clone()).ok())
                .collect();
        }

        if !evidence.feature_summary.is_empty() {
            return Ok(evidence);
        }
    }

    // Fallback: extract summary from prose
    warn!("MPE JSON parse failed, using fallback extraction");
    let evidence = PerceptionEvidence {
        feature_summary: extract_summary_fallback(raw),
        confidence_level: extract_confidence_level(raw),
        raw_summary: raw.to_string(),
        ..Default::default()
    };

    Ok(evidence)
}

/// Parse ARLL (reasoning) output.
pub fn parse_arll_output(raw: &str) -> RMoEResult<ReasoningOutput> {
    let mut output = ReasoningOutput {
        raw_output: raw.to_string(),
        ..Default::default()
    };

    // Try JSON extraction
    if let Some(json) = extract_json_block(raw) {
        if let Some(cot) = json.get("cot").and_then(|v| v.as_str()) {
            output.cot = cot.to_string();
        }
        if let Some(wanna) = json.get("wanna").and_then(|v| v.as_bool()) {
            output.wanna = wanna;
        }
        if let Some(req) = json.get("feedback_request").and_then(|v| v.as_str()) {
            output.feedback_request = req.to_string();
        }
        if let Some(payload) = json.get("feedback_payload").and_then(|v| v.as_str()) {
            output.feedback_payload = payload.to_string();
        }
        if let Some(note) = json.get("temporal_note").and_then(|v| v.as_str()) {
            output.temporal_note = note.to_string();
        }

        // Parse DDx
        if let Some(ddx) = json.get("ddx").and_then(|v| v.as_array()) {
            let hypotheses: Vec<DDxHypothesis> = ddx.iter()
                .filter_map(|item| {
                    let diagnosis = item.get("diagnosis")?.as_str()?.to_string();
                    if !is_clinical_hypothesis(&diagnosis) {
                        return None;
                    }
                    let probability = item.get("probability")?.as_f64()?;
                    let evidence = item.get("evidence")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    Some(DDxHypothesis::new(diagnosis, probability, evidence))
                })
                .collect();

            if !hypotheses.is_empty() {
                output.ensemble = DDxEnsemble::new(hypotheses);
            }
        }

        // Parse RAG references
        if let Some(refs) = json.get("rag_references").and_then(|v| v.as_array()) {
            output.rag_references = refs.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
        }

        debug!("ARLL output parsed from JSON");
        return Ok(output);
    }

    // Fallback: regex extraction
    warn!("ARLL JSON parse failed, using regex fallback");
    output.cot = raw.to_string();
    output.ensemble = parse_ddx_fallback(raw);

    // Check for wanna keywords
    let lower = raw.to_lowercase();
    output.wanna = lower.contains("#wanna#") || lower.contains("wanna");
    if output.wanna {
        if lower.contains("alternate") {
            output.feedback_request = "Alternate View".to_string();
        } else {
            output.feedback_request = "High-Res Crop".to_string();
        }
    }

    Ok(output)
}

/// Parse CSR (clinical report) output.
pub fn parse_csr_output(raw: &str) -> RMoEResult<ClinicalReport> {
    // Try JSON extraction
    if let Some(json) = extract_json_block(raw) {
        if let Ok(report) = serde_json::from_value::<ClinicalReport>(json.clone()) {
            debug!("CSR output parsed from JSON");
            return Ok(report);
        }

        // Partial extraction
        let mut report = ClinicalReport::default();

        if let Some(s) = json.get("standard").and_then(|v| v.as_str()) {
            report.standard = s.to_string();
        }
        if let Some(s) = json.get("snomed_ct").and_then(|v| v.as_str()) {
            report.snomed_ct = s.to_string();
        }
        if let Some(s) = json.get("narrative").and_then(|v| v.as_str()) {
            report.narrative = s.to_string();
        }
        if let Some(s) = json.get("summary").and_then(|v| v.as_str()) {
            report.summary = s.to_string();
        }
        if let Some(s) = json.get("treatment_recommendations").and_then(|v| v.as_str()) {
            report.treatment_recommendations = s.to_string();
        }
        if let Some(b) = json.get("hitl_review_required").and_then(|v| v.as_bool()) {
            report.hitl_review_required = b;
        }
        if let Some(s) = json.get("hitl_reason").and_then(|v| v.as_str()) {
            report.hitl_reason = s.to_string();
        }

        // Risk stratification
        if let Some(rs) = json.get("risk_stratification") {
            report.risk_stratification = RiskScore {
                scale: rs.get("scale").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                score: rs.get("score").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                interpretation: rs.get("interpretation").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                action: rs.get("action").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            };
        }

        if !report.narrative.is_empty() || !report.summary.is_empty() {
            return Ok(report);
        }
    }

    // Fallback: use raw as narrative
    warn!("CSR JSON parse failed, using raw text as narrative");
    Ok(ClinicalReport {
        narrative: raw.to_string(),
        ..Default::default()
    })
}

/// Check if a string looks like a clinical diagnosis name.
fn is_clinical_hypothesis(name: &str) -> bool {
    let non_clinical = [
        "sigma", " sc ", "sc is", "let me", "want me", "break down",
        "iteration", "phase", "protocol", "metric", "wanna", "shows a primary",
        "it seems", "attn score", "roi", "attention map", "arll", "probability",
        "how to", "the model", "approach this",
    ];

    let stripped = name.trim();
    if stripped.len() < 4 {
        return false;
    }
    // Medical diagnoses start with uppercase
    if !stripped.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
        return false;
    }

    let lower = stripped.to_lowercase();
    !non_clinical.iter().any(|s| lower.contains(s))
}

/// Extract DDx from prose using regex.
fn parse_ddx_fallback(raw: &str) -> DDxEnsemble {
    let re = Regex::new(r"([A-Z][A-Za-z ]{3,40})[:\-–]?\s*([0-9]+(?:\.[0-9]+)?)\s*%?").unwrap();
    
    let mut hypotheses = Vec::new();
    for cap in re.captures_iter(raw).take(6) {
        let name = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        let prob_str = cap.get(2).map(|m| m.as_str()).unwrap_or("0");

        if !is_clinical_hypothesis(name) {
            continue;
        }

        let mut prob: f64 = prob_str.parse().unwrap_or(0.0);
        if prob > 1.0 {
            prob /= 100.0;
        }
        if prob > 0.0 && prob <= 1.0 {
            hypotheses.push(DDxHypothesis::new(name.to_string(), prob, "regex fallback"));
        }
    }

    DDxEnsemble::new(hypotheses)
}

/// Extract confidence level from text.
fn extract_confidence_level(text: &str) -> String {
    let lower = text.to_lowercase();
    if lower.contains("high confidence") || lower.contains("confident") {
        "high".to_string()
    } else if lower.contains("low confidence") || lower.contains("uncertain") {
        "low".to_string()
    } else {
        "medium".to_string()
    }
}

/// Extract summary from prose.
fn extract_summary_fallback(text: &str) -> String {
    // Take first sentence or first 200 chars
    let first_sentence = text.split(['.', '!', '?'])
        .next()
        .unwrap_or(text)
        .trim();

    if first_sentence.len() > 200 {
        format!("{}...", &first_sentence[..200])
    } else {
        first_sentence.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_block() {
        let text = r#"Some text before {"key": "value", "num": 42} and after"#;
        let json = extract_json_block(text);
        assert!(json.is_some());
        assert_eq!(json.unwrap()["key"], "value");
    }

    #[test]
    fn test_is_clinical_hypothesis() {
        assert!(is_clinical_hypothesis("Pulmonary adenocarcinoma"));
        assert!(is_clinical_hypothesis("Community-acquired pneumonia"));
        assert!(!is_clinical_hypothesis("let me think"));
        assert!(!is_clinical_hypothesis("sigma is 0.2"));
        assert!(!is_clinical_hypothesis("ABC")); // Too short
    }

    #[test]
    fn test_parse_ddx_fallback() {
        let text = "Pneumonia: 0.65, Tuberculosis: 0.25, Cancer: 0.10";
        let ensemble = parse_ddx_fallback(text);
        assert!(!ensemble.hypotheses.is_empty());
    }
}
