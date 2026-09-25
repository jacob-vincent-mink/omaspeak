use super::*;

#[test]
fn horizontal_setup_choice_requires_a_direction_before_enter() {
    let items = [
        MenuItem::available("Accept setup", "Begin model installation"),
        MenuItem::available("Back", "Return without changes"),
    ];
    let mut keys =
        std::collections::VecDeque::from([KeyCode::Enter, KeyCode::Left, KeyCode::Enter]);
    let mut output = Vec::new();
    let choice = run_choice(
        &mut output,
        "Accept setup",
        "Model: Supertonic",
        &items,
        || {
            Ok(Event::Key(KeyEvent::new(
                keys.pop_front().unwrap(),
                KeyModifiers::NONE,
            )))
        },
    )
    .unwrap();
    assert_eq!(choice, Some(0));
    assert!(keys.is_empty());
    let rendered = String::from_utf8_lossy(&output);
    assert!(rendered.contains("Choose an option to continue"));
    assert!(rendered.contains("Model: Supertonic"));
}

#[test]
fn horizontal_setup_choice_skips_unavailable_option_and_can_go_back() {
    let items = [
        MenuItem::unavailable("Recommended", "Provider missing"),
        MenuItem::available("Customize", "Select a provider"),
    ];
    let mut keys = std::collections::VecDeque::from([
        KeyCode::Left,
        KeyCode::Enter,
        KeyCode::Right,
        KeyCode::Enter,
    ]);
    let mut output = Vec::new();
    let choice = run_choice(
        &mut output,
        "Select setup",
        "Model: Supertonic",
        &items,
        || {
            Ok(Event::Key(KeyEvent::new(
                keys.pop_front().unwrap(),
                KeyModifiers::NONE,
            )))
        },
    )
    .unwrap();
    assert_eq!(choice, Some(1));
    assert!(keys.is_empty());
    assert!(String::from_utf8_lossy(&output).contains("unavailable: Recommended"));

    let mut keys = std::collections::VecDeque::from([KeyCode::Esc]);
    assert_eq!(
        run_choice(&mut Vec::new(), "Accept setup", "", &items, || {
            Ok(Event::Key(KeyEvent::new(
                keys.pop_front().unwrap(),
                KeyModifiers::NONE,
            )))
        })
        .unwrap(),
        None
    );
    assert!(
        select_choice(
            "Select setup",
            "",
            &[
                MenuItem::unavailable("A", ""),
                MenuItem::unavailable("B", "")
            ]
        )
        .is_err()
    );
}

#[test]
fn rows_wrap_unicode_at_narrow_and_normal_widths() {
    use unicode_width::UnicodeWidthStr;
    for width in [24, 80] {
        for indent in [0, 4, 6] {
            let text = wrap(
                "OpenVINO / GPU — /opt/运行时/lib/intel64/Release\nprovider registration failed; configure external libraries",
                width,
                indent,
            );
            for (index, line) in text.split("\r\n").enumerate() {
                assert!(line.width() + if index == 0 { indent } else { 0 } < width);
            }
        }
        let mut output = Vec::new();
        render_at_width(
            &mut output,
            "Runtime",
            "Choose an external installation",
            &[MenuItem::available(
                "OpenVINO",
                "/opt/runtime/lib/intel64/Release",
            )],
            0,
            width,
        )
        .unwrap();
        assert!(!output.is_empty());
    }
    assert!(wrap("\x1bhello\tworld", 80, 0).contains("helloworld"));

    let prose = wrap(
        "This never installs or starts a service; an active daemon restarts after Apply.",
        32,
        6,
    );
    assert!(!prose.contains("\r\n      his"));
    assert!(!prose.contains("\r\n      pply"));
    assert_eq!(
        prose.replace("\r\n      ", " "),
        "This never installs or starts a service; an active daemon restarts after Apply."
    );
}
use std::collections::VecDeque;

fn key(code: KeyCode) -> Event {
    Event::Key(KeyEvent::new(code, KeyModifiers::NONE))
}

fn assert_no_bare_line_feeds(output: &[u8]) {
    assert!(
        output
            .iter()
            .enumerate()
            .all(|(index, byte)| *byte != b'\n' || index > 0 && output[index - 1] == b'\r'),
        "raw-mode rendering must return to column zero before every line feed"
    );
}

#[test]
fn item_constructors_preserve_metadata() {
    assert_eq!(
        MenuItem::available("CPU", "portable"),
        MenuItem {
            label: "CPU".into(),
            detail: "portable".into(),
            enabled: true,
        }
    );
    assert!(!MenuItem::unavailable("NPU", "not built").enabled);
}

#[test]
fn state_uses_available_preference_and_skips_disabled_rows() {
    let items = [
        MenuItem::unavailable("zero", "disabled"),
        MenuItem::available("one", "enabled"),
        MenuItem::unavailable("two", "disabled"),
        MenuItem::available("three", "enabled"),
    ];
    let mut state = MenuState::new(&items, 0).unwrap();
    assert_eq!(state.selected, 1);
    assert_eq!(state.apply(Action::Down, &items), None);
    assert_eq!(state.selected, 3);
    assert_eq!(state.apply(Action::Down, &items), None);
    assert_eq!(state.selected, 1);
    assert_eq!(state.apply(Action::Up, &items), None);
    assert_eq!(state.selected, 3);
    assert_eq!(state.apply(Action::Ignore, &items), None);
    assert_eq!(state.apply(Action::Accept, &items), Some(Some(3)));
    assert_eq!(state.apply(Action::Cancel, &items), Some(None));

    assert!(MenuState::new(&[], 0).is_err());
    assert!(MenuState::new(&[MenuItem::unavailable("x", "x")], 0).is_err());
}

#[test]
fn keyboard_mapping_covers_navigation_selection_and_cancel() {
    for code in [KeyCode::Up, KeyCode::Left, KeyCode::Char('k')] {
        assert_eq!(action(KeyEvent::new(code, KeyModifiers::NONE)), Action::Up);
    }
    for code in [KeyCode::Down, KeyCode::Right, KeyCode::Char('j')] {
        assert_eq!(
            action(KeyEvent::new(code, KeyModifiers::NONE)),
            Action::Down
        );
    }
    assert_eq!(
        action(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        Action::Accept
    );
    assert_eq!(
        action(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
        Action::Cancel
    );
    assert_eq!(
        action(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
        Action::Cancel
    );
}

#[test]
fn menu_renders_metadata_and_processes_arrow_enter_and_cancel() {
    let items = [
        MenuItem::available("CPU", "Built in"),
        MenuItem::unavailable("CUDA", "Unavailable in this build"),
        MenuItem::available("OpenVINO", "Intel CPU, GPU, and NPU"),
    ];
    let mut events = VecDeque::from([
        Event::Resize(80, 24),
        Event::Key(KeyEvent::new_with_kind(
            KeyCode::Down,
            KeyModifiers::NONE,
            KeyEventKind::Release,
        )),
        key(KeyCode::Down),
        key(KeyCode::Enter),
    ]);
    let mut output = Vec::new();
    let selected = run_menu(
        &mut output,
        "Runtime",
        "Choose a runtime.",
        &items,
        0,
        || Ok(events.pop_front().unwrap()),
    )
    .unwrap();
    assert_eq!(selected, Some(2));
    let rendered = String::from_utf8(output).unwrap();
    assert!(rendered.contains("Runtime"));
    assert!(rendered.contains("CUDA"));
    assert!(rendered.contains("Unavailable in this build"));
    assert!(rendered.contains("Esc/q back"));
    assert!(rendered.contains("OMASPEAK  /  RUNTIME"));
    assert!(rendered.contains("SELECTED  /  OpenVINO"));
    assert!(rendered.contains("Intel CPU, GPU, and NPU"));
    assert_no_bare_line_feeds(rendered.as_bytes());

    let mut events = VecDeque::from([key(KeyCode::Char('q'))]);
    assert_eq!(
        run_menu(&mut Vec::new(), "x", "y", &items, 0, || {
            Ok(events.pop_front().unwrap())
        })
        .unwrap(),
        None
    );
}

fn plain_terminal_output(output: &[u8]) -> String {
    let mut text = String::new();
    let mut escape = false;
    for character in String::from_utf8_lossy(output).chars() {
        if character == '\x1b' {
            escape = true;
        } else if escape {
            if character.is_ascii_alphabetic() {
                escape = false;
            }
        } else {
            text.push(character);
        }
    }
    text
}

#[test]
fn viewport_keeps_selection_visible_without_overflow_on_resize() {
    use unicode_width::UnicodeWidthStr;
    let items = (0..100)
        .map(|n| {
            MenuItem::available(
                format!("Choice {n}"),
                "Long details with Unicode 运行时 and many words to wrap across a narrow terminal.",
            )
        })
        .collect::<Vec<_>>();
    for (width, height) in [(24, 8), (80, 24), (12, 4), (4, 2), (1, 1)] {
        for selected in [0, 50, 99] {
            let mut output = Vec::new();
            render_at_size(
                &mut output,
                "Choose a model",
                "Navigate the installed and downloadable catalog.",
                &items,
                selected,
                width,
                height,
            )
            .unwrap();
            let text = plain_terminal_output(&output);
            assert!(
                text.split("\r\n").count() <= height,
                "{width}x{height}: {text:?}"
            );
            assert!(text.split("\r\n").all(|line| line.width() < width));
            if width >= 24 {
                assert!(text.contains(&format!("› Choice {selected}")), "{text:?}");
            }
            assert_no_bare_line_feeds(&output);
        }
    }
}

#[derive(Default)]
struct FakePreview {
    toggles: Vec<usize>,
    stops: usize,
    fail: bool,
    message: Option<String>,
}
impl Preview for FakePreview {
    fn toggle(&mut self, index: usize) -> Result<String> {
        self.toggles.push(index);
        if self.fail {
            bail!("Install this model first");
        }
        Ok("Playing sample".into())
    }
    fn poll(&mut self) -> Result<Option<String>> {
        Ok(self.message.take())
    }
    fn stop(&mut self) {
        self.stops += 1;
    }
}

#[test]
fn voice_preview_plays_highlighted_voice_and_stops_on_navigation_and_exit() {
    let items = [
        MenuItem::available("M1", "one"),
        MenuItem::available("F1", "two"),
    ];
    let mut events = VecDeque::from([
        Some(key(KeyCode::Char(' '))),
        Some(Event::Key(KeyEvent::new_with_kind(
            KeyCode::Char(' '),
            KeyModifiers::NONE,
            KeyEventKind::Repeat,
        ))),
        None,
        Some(key(KeyCode::Down)),
        Some(key(KeyCode::Char(' '))),
        Some(key(KeyCode::Enter)),
    ]);
    let mut preview = FakePreview::default();
    let mut output = Vec::new();
    let result = run_preview_menu(
        &mut output,
        "Voices",
        "Space play/stop",
        &items,
        0,
        &mut preview,
        || Ok(events.pop_front().unwrap()),
    )
    .unwrap();
    assert_eq!(result, Some(1));
    assert_eq!(preview.toggles, [0, 1]);
    assert_eq!(preview.stops, 2);
    assert!(
        String::from_utf8(output)
            .unwrap()
            .contains("Playing sample")
    );
}

#[test]
fn preview_error_is_inline_and_does_not_select_or_exit() {
    let items = [MenuItem::available("M1", "one")];
    let mut events = VecDeque::from([Some(key(KeyCode::Char(' '))), Some(key(KeyCode::Esc))]);
    let mut preview = FakePreview {
        fail: true,
        message: Some("Sample finished".into()),
        ..Default::default()
    };
    let mut output = Vec::new();
    assert_eq!(
        run_preview_menu(
            &mut output,
            "Voices",
            "Space play/stop",
            &items,
            0,
            &mut preview,
            || Ok(events.pop_front().unwrap())
        )
        .unwrap(),
        None
    );
    assert_eq!(preview.stops, 1);
    let text = String::from_utf8(output).unwrap();
    assert!(text.contains("Install this model first"));
    assert!(text.contains("Sample finished"));
}
