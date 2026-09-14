use super::*;
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
    for code in [KeyCode::Up, KeyCode::Char('k')] {
        assert_eq!(action(KeyEvent::new(code, KeyModifiers::NONE)), Action::Up);
    }
    for code in [KeyCode::Down, KeyCode::Char('j')] {
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
    assert_eq!(
        action(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE)),
        Action::Ignore
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
    assert!(rendered.contains("Esc cancel"));
    assert!(rendered.contains("\r\n      "));
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
