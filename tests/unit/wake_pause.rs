use super::*;

fn listener(label: &str) -> (PathBuf, UnixListener) {
    let directory = std::env::temp_dir().join(format!("omaspeak-pause-{label}-{}", request_id()));
    fs::create_dir_all(&directory).unwrap();
    let listener = UnixListener::bind(directory.join("control.sock")).unwrap();
    (directory, listener)
}

#[test]
fn holds_connection_and_player_inherits_it_until_exit() {
    let (directory, listener) = listener("inherit");
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let mut request = String::new();
        BufReader::new(&mut stream).read_line(&mut request).unwrap();
        let request: serde_json::Value = serde_json::from_str(&request).unwrap();
        assert_eq!(request["type"], "hold_pause");
        let mut response = serde_json::to_vec(
            &serde_json::json!({"protocol":1,"id":request["id"],"type":"state","state":"paused"}),
        )
        .unwrap();
        response.push(b'\n');
        stream.write_all(&response).unwrap();
        stream
    });
    let pause = WakePause::acquire_at(&directory.join("control.sock"), &mut || false)
        .unwrap()
        .unwrap();
    let mut stream = server.join().unwrap();
    assert!(!pause.disconnected());
    // No audio: a pipe-blocked child models a player that is still alive.
    let mut command = ProcessCommand::new("cat");
    command.stdin(Stdio::piped()).stdout(Stdio::null());
    pause.retain_in_player(&mut command);
    let mut child = command.spawn().unwrap();
    drop(pause);
    stream
        .set_read_timeout(Some(Duration::from_millis(100)))
        .unwrap();
    assert!(stream.read(&mut [0]).is_err());
    drop(child.stdin.take());
    assert!(child.wait().unwrap().success());
    assert_eq!(stream.read(&mut [0]).unwrap(), 0);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn absent_daemon_is_allowed_and_cancelled_acquisition_does_not_connect() {
    let path = std::env::temp_dir().join(format!("absent-{}", request_id()));
    assert!(
        WakePause::acquire_at(&path, &mut || false)
            .unwrap()
            .is_none()
    );
    let error = WakePause::acquire_at(&path, &mut || true).err().unwrap();
    assert!(error.downcast_ref::<PlaybackCancelled>().is_some());
}

#[test]
fn rejects_unowned_pause_acknowledgement() {
    let (directory, listener) = listener("invalid");
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = String::new();
        BufReader::new(&mut stream).read_line(&mut request).unwrap();
        stream.write_all(b"{\"protocol\":1,\"id\":\"wrong-owner\",\"type\":\"state\",\"state\":\"paused\"}\n").unwrap();
    });
    assert!(WakePause::acquire_at(&directory.join("control.sock"), &mut || false).is_err());
    server.join().unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn lost_malformed_and_oversized_pause_replies_fail_without_an_owner() {
    for payload in [b"".to_vec(), b"not-json\n".to_vec(), vec![b'x'; 65_537]] {
        let (directory, listener) = listener("bad-reply");
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = String::new();
            BufReader::new(&mut stream).read_line(&mut request).unwrap();
            let _ = stream.write_all(&payload);
        });
        assert!(WakePause::acquire_at(&directory.join("control.sock"), &mut || false).is_err());
        server.join().unwrap();
        fs::remove_dir_all(directory).unwrap();
    }
    let (directory, _listener) = listener("not-socket");
    assert!(WakePause::acquire_at(&directory.join("x".repeat(200)), &mut || false).is_err());
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn waiting_for_pause_is_cancellable_and_closes_the_pending_connection() {
    let (directory, listener) = listener("cancel-wait");
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut request = String::new();
        BufReader::new(&mut stream).read_line(&mut request).unwrap();
        assert_eq!(stream.read(&mut [0]).unwrap(), 0);
    });
    let started = Instant::now();
    let error = WakePause::acquire_at(&directory.join("control.sock"), &mut || {
        started.elapsed() > Duration::from_millis(50)
    })
    .err()
    .unwrap();
    assert!(error.downcast_ref::<PlaybackCancelled>().is_some());
    assert!(started.elapsed() < Duration::from_secs(1));
    server.join().unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn a_silent_wake_daemon_cannot_hold_playback_preparation_forever() {
    let (directory, listener) = listener("deadline");
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut request = String::new();
        BufReader::new(&mut stream).read_line(&mut request).unwrap();
        assert_eq!(stream.read(&mut [0]).unwrap(), 0);
    });
    let started = Instant::now();
    let error = WakePause::acquire_at(&directory.join("control.sock"), &mut || false)
        .err()
        .unwrap();
    assert!(error.to_string().contains("within 3 seconds"));
    assert!(started.elapsed() < Duration::from_secs(5));
    server.join().unwrap();
    fs::remove_dir_all(directory).unwrap();
}
