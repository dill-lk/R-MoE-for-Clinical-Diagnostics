//! Interactive chat module.

use anyhow::Result;

pub struct ChatSession {
    pub history: Vec<Message>,
}

pub struct Message {
    pub role: String,
    pub content: String,
}

impl ChatSession {
    pub fn new() -> Self {
        Self { history: vec![] }
    }

    pub fn add_message(&mut self, role: impl Into<String>, content: impl Into<String>) {
        self.history.push(Message {
            role: role.into(),
            content: content.into(),
        });
    }

    pub fn clear(&mut self) {
        self.history.clear();
    }
}
