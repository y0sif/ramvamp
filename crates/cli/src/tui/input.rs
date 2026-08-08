//! The panel's one-line editor.
//!
//! Deliberately small: printable characters, backspace, the four motions,
//! `Ctrl-U`, and paste. It is one *row* rather than one *line* — a bracketed
//! paste can put newlines in the buffer, and [`LineEditor::take`] hands the
//! whole thing over as a single turn. That is the fix for the recorded
//! limitation that a pasted paragraph used to submit one line per turn.
//!
//! Everything here is a total function over an owned `String`, so all of it
//! is testable without a terminal.

/// A single-row editor over a `String`, with a byte-index cursor kept on
/// character boundaries.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct LineEditor {
    buf: String,
    /// Byte offset into `buf`, always at a character boundary.
    cursor: usize,
}

impl LineEditor {
    pub(crate) fn text(&self) -> &str {
        &self.buf
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub(crate) fn insert_char(&mut self, c: char) {
        self.buf.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    pub(crate) fn insert_str(&mut self, text: &str) {
        self.buf.insert_str(self.cursor, text);
        self.cursor += text.len();
    }

    /// Insert a bracketed paste, normalised.
    ///
    /// iTerm2 sends a bare CR for a line break and other terminals send
    /// CRLF; both become a single LF, so the submitted turn has the line
    /// endings the tokenizer expects. Other C0 controls are dropped rather
    /// than stored: they would render as garbage and there is nothing
    /// sensible for a one-row editor to do with them.
    pub(crate) fn insert_paste(&mut self, text: &str) {
        self.insert_str(&normalise_paste(text));
    }

    pub(crate) fn backspace(&mut self) {
        if let Some((offset, _)) = self.buf[..self.cursor].char_indices().next_back() {
            self.buf.remove(offset);
            self.cursor = offset;
        }
    }

    pub(crate) fn clear(&mut self) {
        self.buf.clear();
        self.cursor = 0;
    }

    pub(crate) fn left(&mut self) {
        if let Some((offset, _)) = self.buf[..self.cursor].char_indices().next_back() {
            self.cursor = offset;
        }
    }

    pub(crate) fn right(&mut self) {
        if let Some(c) = self.buf[self.cursor..].chars().next() {
            self.cursor += c.len_utf8();
        }
    }

    pub(crate) fn home(&mut self) {
        self.cursor = 0;
    }

    pub(crate) fn end(&mut self) {
        self.cursor = self.buf.len();
    }

    /// Take the line and reset the editor.
    pub(crate) fn take(&mut self) -> String {
        self.cursor = 0;
        std::mem::take(&mut self.buf)
    }

    /// The visible slice of the line for a field `width` columns wide, plus
    /// the cursor's column within that slice.
    ///
    /// Newlines from a paste render as `⏎` so a multi-line turn is visibly
    /// multi-line without the panel growing. Windowing counts characters,
    /// which is the column count for everything but full-width scripts;
    /// a CJK paste renders correctly but may sit one column right of where
    /// this puts the cursor.
    pub(crate) fn view(&self, width: usize) -> (String, usize) {
        if width == 0 {
            return (String::new(), 0);
        }
        let display: Vec<char> = self
            .buf
            .chars()
            .map(|c| if c == '\n' { '⏎' } else { c })
            .collect();
        let cursor = self.buf[..self.cursor].chars().count();

        let mut start = 0;
        if display.len() >= width {
            if cursor >= width {
                start = cursor + 1 - width;
            }
            // One column past the end is a legal cursor resting place, so
            // the window may stop one character short of the text.
            start = start.min(display.len() + 1 - width);
        }
        let end = (start + width).min(display.len());
        (
            display[start..end].iter().collect(),
            cursor.saturating_sub(start),
        )
    }
}

/// CRLF and bare CR become LF; other C0 controls are dropped.
pub(crate) fn normalise_paste(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                out.push('\n');
            }
            '\n' => out.push('\n'),
            '\t' => out.push(' '),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn editor(text: &str) -> LineEditor {
        let mut editor = LineEditor::default();
        editor.insert_str(text);
        editor
    }

    #[test]
    fn typing_and_backspace_track_the_cursor() {
        let mut editor = LineEditor::default();
        for c in "abc".chars() {
            editor.insert_char(c);
        }
        assert_eq!(editor.text(), "abc");
        editor.backspace();
        assert_eq!(editor.text(), "ab");
        editor.backspace();
        editor.backspace();
        assert_eq!(editor.text(), "");
        // Backspace on an empty line is a no-op, not an underflow.
        editor.backspace();
        assert!(editor.is_empty());
    }

    #[test]
    fn motions_stop_at_the_ends() {
        let mut editor = editor("abc");
        editor.home();
        editor.left();
        editor.insert_char('!');
        assert_eq!(editor.text(), "!abc");
        editor.end();
        editor.right();
        editor.insert_char('?');
        assert_eq!(editor.text(), "!abc?");
    }

    #[test]
    fn motions_step_over_whole_characters() {
        let mut editor = editor("héllo");
        editor.home();
        editor.right();
        editor.right();
        // Cursor sits after the two-byte 'é', so this is a boundary insert
        // rather than a panic.
        editor.insert_char('-');
        assert_eq!(editor.text(), "hé-llo");
        editor.left();
        editor.backspace();
        assert_eq!(editor.text(), "h-llo");
    }

    #[test]
    fn ctrl_u_clears_and_take_resets() {
        let mut editor = editor("throw this away");
        editor.clear();
        assert!(editor.is_empty());
        editor.insert_str("keep this");
        assert_eq!(editor.take(), "keep this");
        assert!(editor.is_empty());
        assert_eq!(editor.view(20), (String::new(), 0));
    }

    #[test]
    fn paste_normalises_line_endings() {
        // iTerm2 sends a bare CR; other terminals send CRLF. Both are one
        // newline.
        assert_eq!(normalise_paste("a\r\nb"), "a\nb");
        assert_eq!(normalise_paste("a\rb"), "a\nb");
        assert_eq!(normalise_paste("a\nb"), "a\nb");
        assert_eq!(normalise_paste("a\r\n\r\nb"), "a\n\nb");
        // Tabs survive as spaces; other controls are dropped.
        assert_eq!(normalise_paste("a\tb\x07c\x1bd"), "a bcd");
    }

    #[test]
    fn a_multi_line_paste_is_one_turn() {
        let mut editor = LineEditor::default();
        editor.insert_paste("first line\r\nsecond line");
        assert_eq!(editor.take(), "first line\nsecond line");
    }

    #[test]
    fn view_shows_the_whole_line_when_it_fits() {
        let editor = editor("hello");
        assert_eq!(editor.view(20), ("hello".to_string(), 5));
    }

    #[test]
    fn view_scrolls_to_keep_the_cursor_visible() {
        let editor = editor("abcdefghij");
        // Cursor is at the end, so the window is the tail with room for it.
        assert_eq!(editor.view(5), ("ghij".to_string(), 4));
        let mut editor = editor;
        editor.home();
        assert_eq!(editor.view(5), ("abcde".to_string(), 0));
    }

    #[test]
    fn view_renders_pasted_newlines_and_never_panics_when_tiny() {
        let editor = editor("one\ntwo");
        assert_eq!(editor.view(10).0, "one⏎two");
        for width in 0..12 {
            let (text, cursor) = editor.view(width);
            assert!(text.chars().count() <= width);
            assert!(cursor <= width);
        }
    }
}
