use super::*;

#[test]
fn request_variants_and_responses_round_trip() {
    for command in [
        Command::Say {
            text: "hello".into(),
            speed: 1.0,
            voice: crate::voices::VoiceSelection::Legacy(0),
            output: Some("out.wav".into()),
            no_play: true,
        },
        Command::Status,
        Command::Shutdown,
    ] {
        let request = Request {
            protocol: 1,
            id: "id".into(),
            command,
        };
        let encoded = serde_json::to_string(&request).unwrap();
        let decoded: Request = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.protocol, 1);
        assert_eq!(decoded.id, "id");
    }

    let response = Response::error("request", "bad_request", "broken");
    let encoded = serde_json::to_string(&response).unwrap();
    let decoded: Response = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded.protocol, 1);
    assert_eq!(decoded.id, "request");
    assert!(matches!(decoded.result, ResultPayload::Error { .. }));
}
