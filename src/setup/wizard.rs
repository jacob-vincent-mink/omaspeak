use std::io::{self, Write};

use anyhow::{Context, Result, bail};
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    execute, queue,
    style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor},
    terminal::{self, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
};

/// One row in an interactive setup selector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MenuItem {
    pub label: String,
    pub detail: String,
    pub enabled: bool,
}

impl MenuItem {
    pub fn available(label: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            detail: detail.into(),
            enabled: true,
        }
    }

    pub fn unavailable(label: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            detail: detail.into(),
            enabled: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Action {
    Up,
    Down,
    Accept,
    Cancel,
    Ignore,
}

#[derive(Debug)]
struct MenuState {
    selected: usize,
}

impl MenuState {
    fn new(items: &[MenuItem], preferred: usize) -> Result<Self> {
        if items.is_empty() {
            bail!("interactive menu has no choices");
        }
        if !items.iter().any(|item| item.enabled) {
            bail!("interactive menu has no available choices");
        }
        let selected = if items.get(preferred).is_some_and(|item| item.enabled) {
            preferred
        } else {
            items
                .iter()
                .position(|item| item.enabled)
                .unwrap_or_default()
        };
        Ok(Self { selected })
    }

    fn apply(&mut self, action: Action, items: &[MenuItem]) -> Option<Option<usize>> {
        match action {
            Action::Up => self.move_by(items, -1),
            Action::Down => self.move_by(items, 1),
            Action::Accept => return Some(Some(self.selected)),
            Action::Cancel => return Some(None),
            Action::Ignore => {}
        }
        None
    }

    fn move_by(&mut self, items: &[MenuItem], direction: isize) {
        let mut next = self.selected;
        loop {
            next = (next as isize + direction).rem_euclid(items.len() as isize) as usize;
            if items[next].enabled {
                self.selected = next;
                return;
            }
        }
    }
}

/// Show an arrow-key selector. Enter accepts; Escape or q cancels.
pub fn select(
    title: &str,
    help: &str,
    items: &[MenuItem],
    preferred: usize,
) -> Result<Option<usize>> {
    let mut stdout = io::stdout();
    let _terminal = TerminalSession::enter(&mut stdout)?;
    run_menu(&mut stdout, title, help, items, preferred, || {
        event::read().context("read terminal input")
    })
}

struct TerminalSession;

impl TerminalSession {
    fn enter(output: &mut impl Write) -> Result<Self> {
        terminal::enable_raw_mode().context("enable terminal raw mode")?;
        if let Err(error) = execute!(output, EnterAlternateScreen, cursor::Hide) {
            let _ = terminal::disable_raw_mode();
            return Err(error).context("open interactive setup screen");
        }
        Ok(Self)
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        let _ = execute!(io::stdout(), cursor::Show, LeaveAlternateScreen);
        let _ = terminal::disable_raw_mode();
    }
}

fn run_menu(
    output: &mut impl Write,
    title: &str,
    help: &str,
    items: &[MenuItem],
    preferred: usize,
    mut read: impl FnMut() -> Result<Event>,
) -> Result<Option<usize>> {
    let mut state = MenuState::new(items, preferred)?;
    loop {
        render(output, title, help, items, state.selected)?;
        let Event::Key(key) = read()? else {
            continue;
        };
        if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            continue;
        }
        if let Some(result) = state.apply(action(key), items) {
            return Ok(result);
        }
    }
}

fn action(key: KeyEvent) -> Action {
    match (key.code, key.modifiers) {
        (KeyCode::Up | KeyCode::Char('k'), KeyModifiers::NONE) => Action::Up,
        (KeyCode::Down | KeyCode::Char('j'), KeyModifiers::NONE) => Action::Down,
        (KeyCode::Enter, KeyModifiers::NONE) => Action::Accept,
        (KeyCode::Esc | KeyCode::Char('q'), KeyModifiers::NONE)
        | (KeyCode::Char('c'), KeyModifiers::CONTROL) => Action::Cancel,
        _ => Action::Ignore,
    }
}

fn render(
    output: &mut impl Write,
    title: &str,
    help: &str,
    items: &[MenuItem],
    selected: usize,
) -> Result<()> {
    queue!(
        output,
        terminal::Clear(ClearType::All),
        cursor::MoveTo(0, 0),
        SetAttribute(Attribute::Bold),
        Print(title),
        SetAttribute(Attribute::Reset),
        Print("\n\n"),
        Print(help),
        Print("\n"),
        SetForegroundColor(Color::DarkGrey),
        Print("↑↓ navigate · Enter select · Esc cancel"),
        ResetColor,
        Print("\n\n")
    )?;

    for (index, item) in items.iter().enumerate() {
        if !item.enabled {
            queue!(output, SetForegroundColor(Color::DarkGrey))?;
        } else if index == selected {
            queue!(
                output,
                SetForegroundColor(Color::Cyan),
                SetAttribute(Attribute::Bold)
            )?;
        }
        queue!(
            output,
            Print(if index == selected { "  › " } else { "    " }),
            Print(&item.label),
            SetAttribute(Attribute::Reset),
            ResetColor,
            Print("\n      "),
            SetForegroundColor(Color::DarkGrey),
            Print(&item.detail),
            ResetColor,
            Print("\n\n")
        )?;
    }
    output.flush()?;
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/setup_wizard.rs"]
mod tests;
