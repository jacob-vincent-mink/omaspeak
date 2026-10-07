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
