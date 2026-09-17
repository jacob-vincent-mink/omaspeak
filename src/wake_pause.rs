//! Playback holds a connection to Omawake; absence is allowed, unsafe replies are not.
use super::*;

pub(super) struct WakePause(UnixStream);

impl WakePause {
    pub(super) fn acquire(mut cancelled: impl FnMut() -> bool) -> Result<Option<Self>> {
        let runtime = std::env::var_os("XDG_RUNTIME_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let user = std::env::var_os("USER").unwrap_or_else(|| "unknown".into());
                std::env::temp_dir().join(format!("omavoice-{}", user.to_string_lossy()))
            });
        Self::acquire_at(&runtime.join("omawake/control.sock"), &mut cancelled)
    }

    fn acquire_at(path: &Path, cancelled: &mut impl FnMut() -> bool) -> Result<Option<Self>> {
        if cancelled() {
            return Err(PlaybackCancelled.into());
        }
        let mut stream = match UnixStream::connect(path) {
            Ok(stream) => stream,
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
                ) =>
            {
                return Ok(None);
            }
            Err(error) => return Err(error).context("connect to Omawake for playback pause"),
        };
        stream.set_write_timeout(Some(Duration::from_millis(500)))?;
        let id = request_id();
        let mut request =
            serde_json::to_vec(&serde_json::json!({"protocol":1,"id":id,"type":"hold_pause"}))?;
        request.push(b'\n');
        stream.write_all(&request)?;
        stream.set_nonblocking(true)?;
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut bytes = Vec::new();
        loop {
            if cancelled() {
                return Err(PlaybackCancelled.into());
            }
            anyhow::ensure!(
                Instant::now() < deadline,
                "Omawake did not acknowledge playback pause within 3 seconds"
            );
            let mut byte = [0];
            match stream.read(&mut byte) {
                Ok(0) => bail!("Omawake closed the playback pause connection"),
                Ok(_) => {
                    if byte[0] == b'\n' {
                        break;
                    }
                    bytes.push(byte[0]);
                    anyhow::ensure!(
                        bytes.len() <= 65_536,
                        "Omawake pause response exceeds 65536 bytes"
                    );
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(10))
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => {
                    return Err(error).context("read Omawake playback pause acknowledgement");
                }
            }
        }
        let response: serde_json::Value = serde_json::from_slice(&bytes)?;
        anyhow::ensure!(
            response["protocol"] == 1
                && response["id"] == id
                && response["type"] == "state"
                && response["state"] == "paused",
            "Omawake cannot hold playback pause; update Omawake before playing speech: {response}"
        );
        Ok(Some(Self(stream)))
    }

    pub(super) fn disconnected(&self) -> bool {
        socket_peer_disconnected(self.0.as_raw_fd())
    }

    pub(super) fn retain_in_player(&self, command: &mut ProcessCommand) {
        let fd = self.0.as_raw_fd();
        // Keep this descriptor in the player as well: if Omaspeak is killed,
        // the hold survives until the actual audio process has stopped.
        unsafe {
            command.pre_exec(move || {
                let flags = libc::fcntl(fd, libc::F_GETFD);
                if flags < 0 || libc::fcntl(fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/wake_pause.rs"]
mod tests;
