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

/// A cancellable audition owned by one voice selector.
pub trait Preview {
    fn toggle(&mut self, index: usize) -> Result<String>;
    fn poll(&mut self) -> Result<Option<String>>;
    fn stop(&mut self);
}

pub fn select_with_preview(
    title: &str,
    help: &str,
    items: &[MenuItem],
    preferred: usize,
    preview: &mut impl Preview,
) -> Result<Option<usize>> {
    let mut stdout = io::stdout();
    let _terminal = TerminalSession::enter(&mut stdout)?;
    let result = run_preview_menu(&mut stdout, title, help, items, preferred, preview, || {
        if event::poll(std::time::Duration::from_millis(100))? {
            Ok(Some(event::read()?))
        } else {
            Ok(None)
        }
    });
    preview.stop();
    result
}

#[allow(clippy::too_many_arguments)]
fn run_preview_menu(
    output: &mut impl Write,
    title: &str,
    help: &str,
    items: &[MenuItem],
    preferred: usize,
    preview: &mut impl Preview,
    mut read: impl FnMut() -> Result<Option<Event>>,
) -> Result<Option<usize>> {
    let mut state = MenuState::new(items, preferred)?;
    let mut status = help.to_owned();
    let mut rendered = None;
    loop {
        match preview.poll() {
            Ok(Some(message)) => status = message,
            Err(error) => status = format!("{error:#}"),
            Ok(None) => {}
        }
        let frame = (state.selected, status.clone());
        if rendered.as_ref() != Some(&frame) {
            render(output, title, &status, items, state.selected)?;
            rendered = Some(frame);
        }
        let event = read()?;
        if matches!(event, Some(Event::Resize(_, _))) {
            rendered = None;
        }
        let Some(Event::Key(key)) = event else {
            continue;
        };
        if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            continue;
        }
        if key.code == KeyCode::Char(' ') && key.modifiers == KeyModifiers::NONE {
            if key.kind == KeyEventKind::Press {
                status = preview
                    .toggle(state.selected)
                    .unwrap_or_else(|error| format!("{error:#}"));
            }
            continue;
        }
        let action = action(key);
        if matches!(
            action,
            Action::Up | Action::Down | Action::Accept | Action::Cancel
        ) {
            preview.stop();
            status = help.to_owned();
        }
        if let Some(result) = state.apply(action, items) {
            return Ok(result);
        }
    }
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
    let (width, height) = terminal::size()
        .ok()
        .filter(|(w, h)| *w > 0 && *h > 0)
        .unwrap_or((80, 24));
    render_at_size(
        output,
        title,
        help,
        items,
        selected,
        width.max(1) as usize,
        height.max(1) as usize,
    )
}

#[cfg(test)]
fn render_at_width(
    output: &mut impl Write,
    title: &str,
    help: &str,
    items: &[MenuItem],
    selected: usize,
    width: usize,
) -> Result<()> {
    render_at_size(output, title, help, items, selected, width, 24)
}

#[allow(clippy::too_many_arguments)]
fn render_at_size(
    output: &mut impl Write,
    title: &str,
    help: &str,
    items: &[MenuItem],
    selected: usize,
    width: usize,
    height: usize,
) -> Result<()> {
    let height = height.max(1);
    let mut header = vec![clip(title, width)];
    if height >= 8 {
        header.extend(
            wrap(help, width, 0)
                .split("\r\n")
                .take(height.saturating_sub(5))
                .map(str::to_owned),
        );
    }
    // Reserve the last row and never emit a trailing newline: raw terminals
    // otherwise scroll the selection off-screen at the bottom edge.
    header.truncate(height.saturating_sub(2));
    let budget = height.saturating_sub(header.len() + 1).max(1);
    let mut lines = Vec::new();
    let mut selected_line = 0;
    for (index, item) in items.iter().enumerate() {
        if index == selected {
            selected_line = lines.len();
        }
        let prefix = if index == selected { "  › " } else { "    " };
        lines.push((index, clip(&format!("{prefix}{}", item.label), width)));
        let detail = wrap(&item.detail, width, 6);
        for (line, value) in detail.split("\r\n").enumerate() {
            let value = if line == 0 {
                format!("      {value}")
            } else {
                value.to_owned()
            };
            lines.push((index, clip(&value, width)));
        }
    }
    let start = selected_line
        .saturating_sub(budget / 3)
        .min(lines.len().saturating_sub(budget));
    queue!(
        output,
        terminal::Clear(ClearType::All),
        cursor::MoveTo(0, 0)
    )?;
    for line in header {
        queue!(output, Print(line), Print("\r\n"))?;
    }
    let visible = lines.iter().skip(start).take(budget);
    for (offset, (index, line)) in visible.enumerate() {
        let color = if !items[*index].enabled {
            Color::DarkGrey
        } else if *index == selected {
            Color::Cyan
        } else {
            Color::Reset
        };
        queue!(output, SetForegroundColor(color), Print(line), ResetColor)?;
        if offset + 1 < budget || height > 1 {
            queue!(output, Print("\r\n"))?;
        }
    }
    if height > 1 {
        let footer = format!(
            "{}/{} · ↑↓ navigate · Enter select · Esc cancel",
            selected + 1,
            items.len()
        );
        queue!(
            output,
            SetAttribute(Attribute::Dim),
            Print(clip(&footer, width)),
            SetAttribute(Attribute::Reset)
        )?;
    }
    output.flush()?;
    Ok(())
}

fn clip(text: &str, width: usize) -> String {
    use unicode_width::UnicodeWidthChar;
    let mut remaining = width.saturating_sub(1);
    text.chars()
        .filter(|c| !c.is_control())
        .take_while(|c| {
            let size = c.width().unwrap_or(0);
            if size > remaining {
                false
            } else {
                remaining -= size;
                true
            }
        })
        .collect()
}

fn wrap(text: &str, width: usize, indent: usize) -> String {
    use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

    let limit = width.saturating_sub(indent + 1).max(1);
    let text: String = text
        .chars()
        .filter(|character| *character == '\n' || !character.is_control())
        .collect();
    let mut output = String::new();
    for (paragraph_index, paragraph) in text.split('\n').enumerate() {
        if paragraph_index > 0 {
            wrapped_newline(&mut output, indent);
        }
        let mut column = 0;
        for word in paragraph.split_whitespace() {
            let word_width = word.width();
            if column > 0 && column + 1 + word_width <= limit {
                output.push(' ');
                output.push_str(word);
                column += 1 + word_width;
                continue;
            }
            if column > 0 {
                wrapped_newline(&mut output, indent);
                column = 0;
            }
            for character in word.chars() {
                let character_width = character.width().unwrap_or(0);
                if column > 0 && column + character_width > limit {
                    wrapped_newline(&mut output, indent);
                    column = 0;
                }
                output.push(character);
                column += character_width;
            }
        }
    }
    output
}

fn wrapped_newline(output: &mut String, indent: usize) {
    output.push_str("\r\n");
    output.push_str(&" ".repeat(indent));
}

#[cfg(test)]
#[path = "../../tests/unit/setup_wizard.rs"]
mod tests;
