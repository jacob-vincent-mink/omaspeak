use super::*;
use std::io::{self, Read};
struct Fragmented {
    bytes: Vec<u8>,
    offset: usize,
}
impl Read for Fragmented {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.offset == self.bytes.len() {
            return Ok(0);
        }
        out[0] = self.bytes[self.offset];
        self.offset += 1;
        Ok(1)
    }
}
#[test]
fn fragmented_pcm_reassembles_signed_samples() {
    let bytes = [i16::MIN, 0, i16::MAX]
        .into_iter()
        .flat_map(i16::to_le_bytes)
        .collect();
    let mut pcm = Vec::new();
    decode_pcm(Fragmented { bytes, offset: 0 }, 3, &mut |p| {
        pcm.extend_from_slice(p);
        Ok(())
    })
    .unwrap();
    assert_eq!(pcm, vec![-1.0, 0.0, 32767.0 / 32768.0]);
}
#[test]
fn malformed_empty_and_oversize_pcm_fail() {
    for bytes in [vec![], vec![0], vec![0; 6]] {
        assert!(decode_pcm(bytes.as_slice(), 2, &mut |_| Ok(())).is_err());
    }
}
#[test]
fn failed_sink_stops_reading() {
    let mut calls = 0;
    assert!(
        decode_pcm(&[0u8; 16000][..], 8000, &mut |_| {
            calls += 1;
            bail!("cancel")
        })
        .is_err()
    );
    assert_eq!(calls, 1);
}
#[test]
fn voice_aliases_have_stable_indices_and_resolve_provider_ids() {
    let mut c = Config::default();
    c.backend.kind = "elevenlabs".into();
    c.backend
        .cloud
        .voices
        .insert("reader".into(), "voice-id".into());
    assert_eq!(
        voices(&c).unwrap(),
        vec![Voice {
            id: 0,
            name: "reader".into()
        }]
    );
    assert!(
        crate::voices::VoiceSelection::Name("missing".into())
            .resolve(&voices(&c).unwrap())
            .is_err()
    );
}
#[test]
fn invalid_urls_cannot_embed_keys_or_redirect_to_cleartext_remote_hosts() {
    for base_url in [
        "https://key@example.com",
        "http://example.com",
        "https://example.com?key=secret",
        "https://example.com/#secret",
    ] {
        let c = CloudConfig {
            base_url: base_url.into(),
            ..Default::default()
        };
        assert!(cloud_http::base_url(&c, "").is_err());
    }
    let c = CloudConfig {
        base_url: "http://127.0.0.1:1234/v1".into(),
        ..Default::default()
    };
    assert!(cloud_http::base_url(&c, "").is_ok());
}

#[test]
fn buffered_cloud_generation_requires_mono_pcm_at_requested_rate() {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    for (mime, valid) in [
        ("audio/pcm; rate=\"24000\"; channels=1", true),
        ("audio/pcm; sample_rate=16000", false),
        ("audio/pcm; channels=2", false),
        ("audio/pcm; ignored", true),
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut c = Config::default();
        c.backend.kind = "openai-compatible".into();
        c.backend.cloud.base_url = format!("http://{}/v1", listener.local_addr().unwrap());
        let backend = CloudBackend::create(&c).unwrap();
        assert_eq!(backend.kind(), "openai-compatible");
        assert_eq!(backend.sample_rate(), 24000);
        assert_eq!(backend.num_voices(), 1);
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(socket.try_clone().unwrap());
            let mut length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
                if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = value.trim().parse().unwrap();
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: {mime}\r\nContent-Length: 4\r\nConnection: close\r\n\r\n").as_bytes()).unwrap();
            socket.write_all(&[0, 0, 0, 64]).unwrap();
        });
        let result = backend.generate("hello", 1.0, 0);
        server.join().unwrap();
        if valid {
            assert_eq!(result.unwrap(), vec![0.0, 0.5]);
        } else {
            assert!(result.is_err());
        }
    }
    let mut config = Config::default();
    config.backend.kind = "deepgram".into();
    config.backend.cloud.model = "custom-voice".into();
    assert_eq!(voices(&config).unwrap()[0].name, "custom-voice");
    assert_eq!(model_name(&config), "custom-voice");
    config.backend.kind = "unknown".into();
    assert!(CloudBackend::create(&config).is_err());
}
