//! Diagnostic pipeline module.

use anyhow::Result;

pub struct DiagnosticSession {
    pub patient_id: Option<String>,
    pub image_path: Option<String>,
    pub symptoms: Vec<String>,
    pub history: Option<String>,
}

impl DiagnosticSession {
    pub fn new() -> Self {
        Self {
            patient_id: None,
            image_path: None,
            symptoms: vec![],
            history: None,
        }
    }

    pub fn with_image(mut self, path: impl Into<String>) -> Self {
        self.image_path = Some(path.into());
        self
    }

    pub fn with_symptoms(mut self, symptoms: Vec<String>) -> Self {
        self.symptoms = symptoms;
        self
    }

    pub fn with_history(mut self, history: impl Into<String>) -> Self {
        self.history = Some(history.into());
        self
    }
}
