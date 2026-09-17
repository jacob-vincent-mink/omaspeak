//! Responsive bounded admission around one persistent, supervised synthesis worker.
use super::*;
use std::collections::VecDeque;
use std::os::unix::process::CommandExt;
use std::process::{ChildStdin, ChildStdout};
use std::thread;

pub(crate) const QUEUE_CAPACITY: usize = 8;
const READING_CAPACITY: usize = 16;
const READ_TIMEOUT: Duration = Duration::from_secs(2);
const WORK_TIMEOUT: Duration = Duration::from_secs(300);
const MAX_MESSAGE: usize = 1024 * 1024 + 16_384;

#[derive(serde::Serialize, serde::Deserialize)]
struct WorkerConfig {
    config: Config,
    paths: AppPaths,
}

pub(super) fn worker(spec: &str) -> Result<()> {
    let spec: WorkerConfig =
        serde_json::from_str(spec).context("decode frozen worker configuration")?;
    let config = spec.config;
    let paths = &spec.paths;
    let engine = Engine::load(&config, paths)?;
    let mut output = std::io::stdout().lock();
    write_response(
        &mut output,
        &Response {
            protocol: 1,
            id: "ready".into(),
            result: status_payload(&engine, &config),
        },
    )?;
    output.flush()?;
    let mut input = std::io::stdin().lock();
    loop {
        let request = read_request(&mut input, MAX_MESSAGE)?;
        let response = match request.command {
            Command::Say { no_play: true, .. } => handle_request(&engine, &config, paths, request),
            Command::Shutdown => return Ok(()),
            _ => Response::error(
                request.id,
                "invalid_request",
                "worker only accepts file synthesis",
            ),
        };
        write_response(&mut output, &response)?;
        output.flush()?;
    }
}

struct ProcessWorker {
    child: Child,
    input: ChildStdin,
    output: ChildStdout,
    incoming: Vec<u8>,
    outgoing: Vec<u8>,
    written: usize,
    ready: bool,
    started: Instant,
}

fn nonblocking(fd: RawFd) -> Result<()> {
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(std::io::Error::last_os_error()).context("make worker pipe nonblocking");
    }
    Ok(())
}

impl ProcessWorker {
    fn start(config: &Config, paths: &AppPaths) -> Result<Self> {
        let mut command = ProcessCommand::new(std::env::current_exe()?);
        command
            .arg("__request-worker")
            .arg(serde_json::to_string(&WorkerConfig {
                config: config.clone(),
                paths: paths.clone(),
            })?)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .process_group(0);
        if config.backend.kind == "supertonic" {
            let report = omaspeak::runtime::discover(&config.backend, &paths.config_file);
            if let Some(loader_path) = omaspeak::runtime::reexec_loader_path(&report)? {
                command.env("LD_LIBRARY_PATH", loader_path);
                command.env(omaspeak::runtime::REEXEC_SENTINEL, "1");
            }
        }
        configure_child_parent_death(&mut command);
        let mut child = command.spawn().context("start synthesis worker")?;
        let input = child.stdin.take().context("worker stdin")?;
        let output = child.stdout.take().context("worker stdout")?;
        let process = Self {
            child,
            input,
            output,
            incoming: Vec::new(),
            outgoing: Vec::new(),
            written: 0,
            ready: false,
            started: Instant::now(),
        };
        nonblocking(process.input.as_raw_fd())?;
        nonblocking(process.output.as_raw_fd())?;
        Ok(process)
    }

    fn submit(&mut self, request: &Request) -> Result<()> {
        anyhow::ensure!(
            self.outgoing.is_empty(),
            "worker still has a pending request"
        );
        self.outgoing = serde_json::to_vec(request)?;
        self.outgoing.push(b'\n');
        self.written = 0;
        Ok(())
    }

    fn poll(&mut self) -> Result<Option<Response>> {
        if !self.ready && self.started.elapsed() > WORK_TIMEOUT {
            bail!("synthesis worker startup timed out");
        }
        if !self.outgoing.is_empty() {
            match self.input.write(&self.outgoing[self.written..]) {
                Ok(0) => bail!("synthesis worker input closed"),
                Ok(n) => {
                    self.written += n;
                    if self.written == self.outgoing.len() {
                        self.outgoing.clear();
                    }
                }
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
                    ) => {}
                Err(error) => return Err(error.into()),
            }
        }
        if let Some(line) = read_frame(&mut self.output, &mut self.incoming, MAX_MESSAGE)? {
            return Ok(Some(
                serde_json::from_slice(&line).context("parse synthesis worker response")?,
            ));
        }
        if let Some(status) = self.child.try_wait()? {
            bail!("synthesis worker exited: {status}");
        }
        Ok(None)
    }
}

impl Drop for ProcessWorker {
    fn drop(&mut self) {
        kill_group(&mut self.child);
    }
}

fn kill_group(child: &mut Child) {
    // Descendant native workers share this group; a replacement must not overlap.
    unsafe {
        libc::kill(-(child.id() as i32), libc::SIGKILL);
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn read_frame(
    reader: &mut impl Read,
    pending: &mut Vec<u8>,
    limit: usize,
) -> Result<Option<Vec<u8>>> {
    // Read at most one bounded chunk per poll; slow/large clients cannot monopolize service.
    let mut buffer = [0_u8; 8192];
    match reader.read(&mut buffer) {
        Ok(0) => bail!("connection closed before a complete response"),
        Ok(n) => pending.extend_from_slice(&buffer[..n]),
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
            ) =>
        {
            return Ok(None);
        }
        Err(error) => return Err(error.into()),
    }
    anyhow::ensure!(pending.len() <= limit, "message exceeds {limit} bytes");
    if let Some(end) = pending.iter().position(|byte| *byte == b'\n') {
        anyhow::ensure!(
            end + 1 == pending.len(),
            "only one request per connection is allowed"
        );
        return Ok(Some(std::mem::take(pending)));
    }
    Ok(None)
}

struct Reading {
    stream: UnixStream,
    bytes: Vec<u8>,
    started: Instant,
}

struct Job {
    stream: UnixStream,
    request: Request,
    stage: Option<PathBuf>,
    destination: Option<PathBuf>,
    player: Option<Child>,
    response: Option<Response>,
    started: Instant,
}

impl Job {
    fn new(stream: UnixStream, request: Request) -> Self {
        Self {
            stream,
            request,
            stage: None,
            destination: None,
            player: None,
            response: None,
            started: Instant::now(),
        }
    }
    fn reply(&mut self, result: ResultPayload) {
        write_response_best_effort(
            &mut self.stream,
            &Response {
                protocol: 1,
                id: self.request.id.clone(),
                result,
            },
        );
    }
    fn error(&mut self, code: &str, message: impl std::fmt::Display) {
        write_response_best_effort(
            &mut self.stream,
            &Response::error(&self.request.id, code, message),
        );
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        if let Some(mut player) = self.player.take() {
            kill_group(&mut player);
        }
        if let Some(stage) = &self.stage {
            let _ = fs::remove_file(stage);
        }
    }
}

pub(super) fn serve(
    config_path: &Path,
    paths: &AppPaths,
    interrupted: Arc<AtomicBool>,
) -> Result<()> {
    let config = Config::load(config_path)?;
    anyhow::ensure!(
        matches!(config.backend.kind.as_str(), "audiocpp" | "supertonic"),
        "unsupported TTS backend"
    );
    let (socket, _startup_lock) = prepare_daemon_socket(paths)?;
    let listener = UnixListener::bind(&socket)?;
    fs::set_permissions(&socket, fs::Permissions::from_mode(0o600))?;
    let metadata = fs::symlink_metadata(&socket)?;
    listener.set_nonblocking(true)?;
    let result = serve_loop(&config, paths, &listener, &interrupted);
    drop(listener);
    finish_daemon(&socket, &metadata, result)
}

fn serve_loop(
    config: &Config,
    paths: &AppPaths,
    listener: &UnixListener,
    interrupted: &AtomicBool,
) -> Result<()> {
    let mut worker = Some(ProcessWorker::start(config, paths)?);
    let mut metadata = json!({"kind":config.backend.kind,"requests":{"loading":true}});
    let mut sample_rate = 0;
    let mut reading = Vec::<Reading>::new();
    let mut queue = VecDeque::<Job>::new();
    let mut active: Option<Job> = None;
    let mut shutting_down = false;
    while !shutting_down && !interrupted.load(Ordering::Relaxed) {
        for _ in 0..READING_CAPACITY {
            let stream = match listener.accept() {
                Ok((stream, _)) => stream,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(error) => return Err(error.into()),
            };
            stream.set_nonblocking(true)?;
            if reading.len() >= READING_CAPACITY {
                let mut stream = stream;
                write_response_best_effort(
                    &mut stream,
                    &Response::error("unknown", "busy", "request admission limit reached"),
                );
            } else {
                reading.push(Reading {
                    stream,
                    bytes: Vec::new(),
                    started: Instant::now(),
                });
            }
        }
        let mut index = 0;
        while index < reading.len() {
            let peer = &mut reading[index];
            let result = if peer.started.elapsed() > READ_TIMEOUT {
                Err(anyhow!("request read timed out"))
            } else {
                read_frame(
                    &mut peer.stream,
                    &mut peer.bytes,
                    config
                        .daemon
                        .max_text_bytes
                        .saturating_add(16_384)
                        .min(MAX_MESSAGE),
                )
            };
            match result {
                Ok(None) => {
                    index += 1;
                    continue;
                }
                Err(error) => {
                    let mut peer = reading.remove(index);
                    write_response_best_effort(
                        &mut peer.stream,
                        &Response::error("unknown", "invalid_request", error),
                    );
                }
                Ok(Some(bytes)) => {
                    let peer = reading.remove(index);
                    let request = read_request(&mut &bytes[..], MAX_MESSAGE);
                    match request {
                        Err(error) => {
                            let mut stream = peer.stream;
                            write_response_best_effort(
                                &mut stream,
                                &Response::error("unknown", "invalid_request", error),
                            );
                        }
                        Ok(request) => {
                            let mut job = Job::new(peer.stream, request);
                            if job.request.id.is_empty() || job.request.id.len() > 128 {
                                job.error(
                                    "invalid_request",
                                    "request ID must contain 1 to 128 bytes",
                                );
                                continue;
                            }
                            match &job.request.command {
                                Command::Say { text, .. } => {
                                    if text.len() > config.daemon.max_text_bytes {
                                        job.error(
                                            "invalid_request",
                                            "text exceeds configured limit",
                                        );
                                    } else if active
                                        .as_ref()
                                        .is_some_and(|other| other.request.id == job.request.id)
                                        || queue
                                            .iter()
                                            .any(|other| other.request.id == job.request.id)
                                    {
                                        job.error(
                                            "duplicate_request",
                                            "request ID is already active or queued",
                                        );
                                    } else if queue.len() >= QUEUE_CAPACITY {
                                        job.error("busy", "speech queue is full");
                                    } else {
                                        queue.push_back(job);
                                    }
                                }
                                Command::Status => {
                                    let mut backend = metadata.clone();
                                    backend["requests"] = json!({"active":active.as_ref().map(|job| &job.request.id),"queued":queue.iter().map(|job| &job.request.id).collect::<Vec<_>>(),"capacity":QUEUE_CAPACITY,"loading":worker.as_ref().is_some_and(|worker| !worker.ready),"worker_ready":worker.as_ref().is_some_and(|worker| worker.ready)});
                                    job.reply(ResultPayload::Status {
                                        running: true,
                                        pid: std::process::id(),
                                        model: config.model.name.clone(),
                                        language: config.model.language.clone(),
                                        sample_rate,
                                        backend,
                                        audio: omaspeak::audio_devices::status(
                                            &config.audio.device,
                                            None,
                                            None,
                                        ),
                                    });
                                }
                                Command::Cancel { request_id } => {
                                    let target = request_id.clone();
                                    let mut count = 0;
                                    if active.as_ref().is_some_and(|current| {
                                        target.as_ref().is_none_or(|id| id == &current.request.id)
                                    }) {
                                        let mut current = active.take().unwrap();
                                        if current.player.is_none() {
                                            worker.take();
                                        }
                                        current.error("cancelled", "speech request cancelled");
                                        drop(current);
                                        count += 1;
                                    }
                                    if let Some(id) = &target
                                        && let Some(position) =
                                            queue.iter().position(|queued| &queued.request.id == id)
                                    {
                                        let mut queued = queue.remove(position).unwrap();
                                        queued
                                            .error("cancelled", "queued speech request cancelled");
                                        count += 1;
                                    }
                                    job.reply(ResultPayload::Cancelled {
                                        request_id: target,
                                        count,
                                    });
                                }
                                Command::Shutdown => {
                                    job.reply(ResultPayload::Shutdown);
                                    shutting_down = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        if shutting_down {
            break;
        }
        queue.retain(|job| !socket_peer_disconnected(job.stream.as_raw_fd()));
        if active.as_ref().is_some_and(|job| {
            socket_peer_disconnected(job.stream.as_raw_fd()) || job.started.elapsed() > WORK_TIMEOUT
        }) {
            let mut job = active.take().unwrap();
            if job.player.is_none() {
                worker.take();
            }
            job.error("cancelled", "request disconnected or timed out");
        }
        if let Some(job) = &mut active
            && let Some(player) = &mut job.player
            && let Some(status) = player.try_wait()?
        {
            // Reaped player has exited; retain Job cleanup for any descendants.
            if status.success() {
                if let Some(response) = job.response.take() {
                    job.reply(response.result);
                }
            } else {
                job.error("playback", format!("playback worker exited: {status}"));
            }
            active.take();
        }
        let response = worker.as_mut().map(ProcessWorker::poll).transpose();
        match response {
            Ok(Some(Some(response))) => {
                if !worker.as_ref().unwrap().ready {
                    if response.id != "ready"
                        || !matches!(response.result, ResultPayload::Status { .. })
                    {
                        bail!("invalid synthesis worker startup response");
                    }
                    if let ResultPayload::Status {
                        backend,
                        sample_rate: rate,
                        ..
                    } = response.result
                    {
                        metadata = backend;
                        sample_rate = rate;
                    }
                    worker.as_mut().unwrap().ready = true;
                } else if let Some(mut job) = active.take() {
                    if response.id != job.request.id {
                        job.error("runtime", "synthesis worker response ID mismatch");
                        worker.take();
                    } else {
                        match complete_synthesis(&mut job, response) {
                            Ok(true) => active = Some(job),
                            Ok(false) => {}
                            Err(error) => job.error("runtime", error),
                        }
                    }
                } else {
                    bail!("unsolicited synthesis worker response");
                }
            }
            Err(error) => {
                let starting = worker.as_ref().is_some_and(|worker| !worker.ready);
                worker.take();
                if let Some(mut job) = active.take() {
                    job.error("runtime", &error);
                }
                // Fail the bounded batch explicitly; never replay a request after a worker crash.
                for mut job in queue.drain(..) {
                    job.error("runtime", &error);
                }
                if starting {
                    return Err(error).context("initialize synthesis worker");
                }
            }
            _ => {}
        }
        if worker.is_none() && !queue.is_empty() {
            worker = Some(ProcessWorker::start(config, paths)?);
        }
        if active.is_none()
            && worker.as_ref().is_some_and(|worker| worker.ready)
            && let Some(mut job) = queue.pop_front()
        {
            match start_synthesis(&mut job, paths, worker.as_mut().unwrap()) {
                Ok(()) => active = Some(job),
                Err(error) => job.error("runtime", error),
            }
        }
        thread::sleep(Duration::from_millis(10));
    }
    worker.take();
    if let Some(mut job) = active.take() {
        job.error("cancelled", "daemon stopped");
    }
    for mut job in queue {
        job.error("cancelled", "daemon stopped");
    }
    Ok(())
}

fn start_synthesis(job: &mut Job, paths: &AppPaths, worker: &mut ProcessWorker) -> Result<()> {
    let Command::Say {
        text,
        speed,
        voice,
        output,
        ..
    } = &job.request.command
    else {
        bail!("not a speech request");
    };
    let destination = output
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| paths.state_dir.join("last.wav"));
    let parent = destination
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let stage = parent.join(format!(".omaspeak-{}.wav", request_id()));
    use std::os::unix::fs::OpenOptionsExt;
    fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(&stage)?;
    job.stage = Some(stage.clone());
    job.destination = Some(destination);
    job.started = Instant::now();
    worker.submit(&Request {
        protocol: 1,
        id: job.request.id.clone(),
        command: Command::Say {
            text: text.clone(),
            speed: *speed,
            voice: voice.clone(),
            output: Some(stage.to_string_lossy().into_owned()),
            no_play: true,
        },
    })
}

fn complete_synthesis(job: &mut Job, mut response: Response) -> Result<bool> {
    if let ResultPayload::Synthesis { output, .. } = &mut response.result {
        let destination = job
            .destination
            .as_ref()
            .context("missing request destination")?;
        fs::rename(
            job.stage.as_ref().context("missing staged output")?,
            destination,
        )?;
        job.stage = None;
        *output = destination.to_string_lossy().into_owned();
        if matches!(job.request.command, Command::Say { no_play: false, .. }) {
            let mut command = ProcessCommand::new(std::env::current_exe()?);
            command
                .arg("__voice-playback")
                .arg(destination)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::inherit())
                .process_group(0);
            configure_child_parent_death(&mut command);
            job.player = Some(command.spawn()?);
            job.response = Some(response);
            return Ok(true);
        }
    }
    job.reply(response.result);
    Ok(false)
}
