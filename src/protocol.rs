use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct Request {
    pub protocol: u32,
    pub id: String,
    #[serde(flatten)]
    pub command: Command,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Command {
    Say {
        text: String,
        speed: f32,
        voice: crate::voices::VoiceSelection,
        output: Option<String>,
        no_play: bool,
    },
    Cancel {
        request_id: Option<String>,
    },
    Status,
    Shutdown,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Response {
    pub protocol: u32,
    pub id: String,
    #[serde(flatten)]
    pub result: ResultPayload,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResultPayload {
    Synthesis {
        output: String,
        sample_rate: i32,
        samples: usize,
        audio_seconds: f64,
        load_milliseconds: u64,
        synthesis_milliseconds: u64,
    },
    Status {
        running: bool,
        pid: u32,
        model: String,
        sample_rate: i32,
        backend: serde_json::Value,
    },
    Cancelled {
        request_id: Option<String>,
        count: usize,
    },
    Shutdown,
    Error {
        code: String,
        message: String,
    },
}

impl Response {
    pub fn error(id: impl Into<String>, code: &str, error: impl std::fmt::Display) -> Self {
        Self {
            protocol: 1,
            id: id.into(),
            result: ResultPayload::Error {
                code: code.into(),
                message: error.to_string(),
            },
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/protocol.rs"]
mod tests;
