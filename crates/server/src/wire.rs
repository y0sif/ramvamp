//! Hand-written HTTP/1.1 chunked framing for the streaming path.
//!
//! # Why this exists at all
//!
//! `tiny_http::Response` is correct and convenient, and it is *unusable* for
//! server-sent events. Measured, not read in the docs:
//!
//! * `Response::raw_print` wraps the body in a `chunked_transfer::Encoder`
//!   built with `chunks_size = 8192` and `flush_after_write = false`, which
//!   then writes into the connection's 1 KiB `BufWriter`. Nothing reaches the
//!   socket until 8 KiB accumulates or the encoder is dropped, so a
//!   twenty-event stream paced at 500 ms arrived **all at once at t = 10.0 s**.
//!   A client cannot tell that apart from a server that thought for ten
//!   seconds and then answered instantly, which is exactly the property SSE
//!   exists to provide.
//! * `Request::respond` funnels the result through
//!   `ignore_client_closing_errors`, which maps `BrokenPipe`,
//!   `ConnectionReset`, `ConnectionAborted` and `ConnectionRefused` to
//!   `Ok(())`. On that path **a client hanging up is invisible**, so a
//!   generation that nobody is reading runs to completion and occupies the
//!   single session for minutes.
//!
//! `Request::into_writer` hands over the connection's writer directly. It is a
//! `SequentialWriter<BufWriter<RefinedTcpStream>>`, whose `flush` takes the
//! lock and flushes the `BufWriter` through to the socket, so an explicit
//! flush per event does reach the wire — verified: events then arrive at the
//! real interval, and the first write after a hangup returns
//! `ErrorKind::BrokenPipe` rather than success.
//!
//! What that costs is the framing, which is this module: the response head and
//! `{hexlen}\r\n{payload}\r\n` per event, terminated by a zero-length chunk.
//!
//! # The two framing rules that bite
//!
//! * **The hex length counts the payload only**, not the `\r\n` that follows
//!   it. Counting the terminator shifts every subsequent chunk boundary by
//!   two bytes and the stream desynchronizes at event two.
//! * **`Content-Length` is never sent beside `Transfer-Encoding`.** Sending
//!   both is a request-smuggling primitive: two intermediaries can disagree
//!   about where the message ends. `Response` would add one, which is the
//!   second reason the streaming path does not use it.

use std::io::Write;
use std::time::{Duration, Instant};

use crate::engine::StreamError;

/// The streaming response head, byte for byte.
///
/// `X-Accel-Buffering: no` is for reverse proxies that buffer by default
/// (nginx above all); it costs nothing on a direct connection and turns a
/// silently-batched stream back into a live one when someone puts this behind
/// a proxy anyway. `Connection: keep-alive` is HTTP/1.1's default and is
/// stated rather than implied because some clients look for it before they
/// will hold a stream open.
pub const SSE_HEAD: &str = "HTTP/1.1 200 OK\r\n\
     Content-Type: text/event-stream\r\n\
     Cache-Control: no-cache\r\n\
     Connection: keep-alive\r\n\
     X-Accel-Buffering: no\r\n\
     Transfer-Encoding: chunked\r\n\
     \r\n";

/// The zero-length chunk that ends a chunked body.
pub const TERMINATOR: &str = "0\r\n\r\n";

/// An SSE *comment*: a line beginning with `:`, which every EventSource
/// implementation reads and discards.
///
/// This is the keep-alive. It is a comment rather than an empty `data:` event
/// on purpose — a `data:` event would reach the client's chunk handler and a
/// strict one would try to parse it as JSON.
pub const KEEPALIVE: &str = ": keep-alive\n\n";

/// A writer that frames whatever it is given as one chunk per call and
/// flushes it to the socket immediately.
///
/// Deliberately not a `Write` implementation: `write` has no natural event
/// boundary, and the whole point of this type is that one call is one chunk
/// is one flush.
#[derive(Debug)]
pub struct ChunkedWriter<W: Write> {
    inner: W,
    head_sent: bool,
    terminated: bool,
}

impl<W: Write> ChunkedWriter<W> {
    /// Wrap a connection writer. Nothing is written until [`Self::head`].
    pub fn new(inner: W) -> Self {
        ChunkedWriter {
            inner,
            head_sent: false,
            terminated: false,
        }
    }

    /// Whether the response head has gone out.
    ///
    /// Once it has, the status code is committed and a later failure can only
    /// be reported inside the stream, never as an HTTP status.
    pub fn head_sent(&self) -> bool {
        self.head_sent
    }

    /// Write [`SSE_HEAD`] and flush it.
    ///
    /// Called *before* the model does any work: a 4K prompt takes about six
    /// minutes to prefill, and every client we care about gives up on silence
    /// well before that (undici's `bodyTimeout`, the `openai` provider's
    /// `headerTimeout`, `CLAUDE_CODE_BYTE_STREAM_IDLE_TIMEOUT_MS` — all
    /// 300 s). The head is the first of the bytes that keep the clock reset.
    pub fn head(&mut self) -> Result<(), StreamError> {
        self.inner.write_all(SSE_HEAD.as_bytes())?;
        self.inner.flush()?;
        self.head_sent = true;
        Ok(())
    }

    /// Frame `payload` as one chunk and flush.
    ///
    /// An empty payload is a no-op rather than a chunk: a zero-length chunk
    /// *is* the end-of-body marker, so framing one here would truncate the
    /// stream in a way no client could distinguish from a clean end.
    pub fn event(&mut self, payload: &str) -> Result<(), StreamError> {
        if payload.is_empty() {
            return Ok(());
        }
        let bytes = payload.as_bytes();
        // Length of the payload only. The trailing CRLF is framing, not
        // content, and counting it desynchronizes the stream at chunk two.
        write!(self.inner, "{:x}\r\n", bytes.len())?;
        self.inner.write_all(bytes)?;
        self.inner.write_all(b"\r\n")?;
        self.inner.flush()?;
        Ok(())
    }

    /// Write the terminating zero-length chunk and flush. Idempotent.
    pub fn finish(&mut self) -> Result<(), StreamError> {
        if self.terminated {
            return Ok(());
        }
        self.terminated = true;
        self.inner.write_all(TERMINATOR.as_bytes())?;
        self.inner.flush()?;
        Ok(())
    }
}

/// One streaming response's writer plus the two facts every writer of it
/// needs to agree on.
///
/// Shared behind a mutex between the thread running the model and the
/// keep-alive thread (see [`KeepAlive`]), which is the only reason this is a
/// struct and not three locals.
#[derive(Debug)]
pub struct StreamState<W: Write> {
    writer: ChunkedWriter<W>,
    /// The first write failure seen, from either writer. Sticky: once the
    /// socket is gone every later write would fail the same way, and the
    /// generation is about to be aborted anyway.
    failed: Option<StreamError>,
    /// When the last byte went out. The keep-alive thread reads this rather
    /// than keeping its own clock, so a stream that is already producing
    /// content chunks does not also get comments interleaved into it.
    last_write: Instant,
}

impl<W: Write> StreamState<W> {
    /// Wrap a connection writer.
    pub fn new(writer: W) -> Self {
        StreamState {
            writer: ChunkedWriter::new(writer),
            failed: None,
            last_write: Instant::now(),
        }
    }

    /// The first write failure, if there has been one.
    pub fn failure(&self) -> Option<StreamError> {
        self.failed
    }

    /// Whether the head has gone out; see [`ChunkedWriter::head_sent`].
    pub fn head_sent(&self) -> bool {
        self.writer.head_sent()
    }

    /// How long since anything was written.
    pub fn idle_for(&self) -> Duration {
        self.last_write.elapsed()
    }

    /// Send the response head, recording any failure.
    pub fn head(&mut self) -> Result<(), StreamError> {
        self.record(|w| w.head())
    }

    /// Send one framed event, recording any failure.
    pub fn emit(&mut self, payload: &str) -> Result<(), StreamError> {
        self.record(|w| w.event(payload))
    }

    /// Send the body terminator, recording any failure.
    pub fn finish(&mut self) -> Result<(), StreamError> {
        self.record(|w| w.finish())
    }

    fn record(
        &mut self,
        write: impl FnOnce(&mut ChunkedWriter<W>) -> Result<(), StreamError>,
    ) -> Result<(), StreamError> {
        if let Some(failed) = self.failed {
            return Err(failed);
        }
        match write(&mut self.writer) {
            Ok(()) => {
                self.last_write = Instant::now();
                Ok(())
            }
            Err(e) => {
                self.failed = Some(e);
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The head is a wire contract, so it is pinned byte for byte rather than
    /// described. A reordering or a lost header is a behaviour change.
    #[test]
    fn the_response_head_is_byte_exact() {
        assert_eq!(
            SSE_HEAD,
            "HTTP/1.1 200 OK\r\n\
             Content-Type: text/event-stream\r\n\
             Cache-Control: no-cache\r\n\
             Connection: keep-alive\r\n\
             X-Accel-Buffering: no\r\n\
             Transfer-Encoding: chunked\r\n\
             \r\n"
        );
        // Sending both framing headers is a request-smuggling primitive.
        assert!(!SSE_HEAD.to_ascii_lowercase().contains("content-length"));
        assert!(SSE_HEAD.ends_with("\r\n\r\n"));
    }

    #[test]
    fn a_chunk_is_hex_length_then_payload_then_crlf() -> Result<(), StreamError> {
        let mut writer = ChunkedWriter::new(Vec::new());
        writer.event("data: {}\n\n")?;
        // 10 payload bytes -> "a", and the trailing CRLF is *not* counted.
        assert_eq!(writer.inner, b"a\r\ndata: {}\n\n\r\n");
        Ok(())
    }

    /// A payload over 15 bytes is where a decimal length would still parse as
    /// hex and silently mean something else, so the base is pinned here.
    #[test]
    fn the_chunk_length_is_lowercase_hex_of_the_payload_only() -> Result<(), StreamError> {
        let payload = "x".repeat(255);
        let mut writer = ChunkedWriter::new(Vec::new());
        writer.event(&payload)?;
        let out = String::from_utf8(writer.inner).expect("ascii");
        assert!(out.starts_with("ff\r\n"), "{:?}", &out[..8]);
        assert!(out.ends_with("\r\n"));
        assert_eq!(out.len(), 4 + 255 + 2);
        Ok(())
    }

    /// Multi-byte characters are counted in bytes, not in `char`s — a length
    /// in characters would leave the stream short and desynchronized.
    #[test]
    fn the_chunk_length_counts_bytes_not_characters() -> Result<(), StreamError> {
        let payload = "data: \"héllo→\"\n\n";
        let mut writer = ChunkedWriter::new(Vec::new());
        writer.event(payload)?;
        let out = writer.inner;
        let head = format!("{:x}\r\n", payload.len());
        assert!(out.starts_with(head.as_bytes()));
        assert_ne!(payload.len(), payload.chars().count());
        assert_eq!(out.len(), head.len() + payload.len() + 2);
        Ok(())
    }

    #[test]
    fn the_terminator_is_a_zero_length_chunk_and_is_idempotent() -> Result<(), StreamError> {
        let mut writer = ChunkedWriter::new(Vec::new());
        writer.finish()?;
        writer.finish()?;
        assert_eq!(writer.inner, b"0\r\n\r\n");
        Ok(())
    }

    /// An empty payload would frame as `0\r\n\r\n`, which is the end of the
    /// body. Framing one mid-stream truncates the response.
    #[test]
    fn an_empty_payload_is_never_framed() -> Result<(), StreamError> {
        let mut writer = ChunkedWriter::new(Vec::new());
        writer.event("")?;
        assert!(writer.inner.is_empty());
        Ok(())
    }

    #[test]
    fn a_whole_stream_frames_and_unframes() -> Result<(), StreamError> {
        let mut writer = ChunkedWriter::new(Vec::new());
        writer.head()?;
        writer.event(KEEPALIVE)?;
        writer.event("data: {\"a\":1}\n\n")?;
        writer.finish()?;
        let out = String::from_utf8(writer.inner).expect("ascii");
        let body = out.strip_prefix(SSE_HEAD).expect("head first");
        assert_eq!(
            body,
            format!(
                "{:x}\r\n{KEEPALIVE}\r\n{:x}\r\ndata: {{\"a\":1}}\n\n\r\n0\r\n\r\n",
                KEEPALIVE.len(),
                15
            )
        );
        Ok(())
    }

    /// The keep-alive is a comment line, which EventSource discards. If it
    /// ever became a `data:` event a strict client would try to parse it.
    #[test]
    fn the_keep_alive_is_an_sse_comment() {
        assert!(KEEPALIVE.starts_with(':'));
        assert!(!KEEPALIVE.starts_with("data:"));
        assert!(KEEPALIVE.ends_with("\n\n"));
        assert!(!KEEPALIVE.contains('\r'));
    }

    /// A failed write is sticky and is reported to every later caller, so the
    /// keep-alive thread and the generation loop reach the same conclusion
    /// about a socket that has gone away.
    #[test]
    fn a_write_failure_sticks() {
        struct Dead;
        impl Write for Dead {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::from(std::io::ErrorKind::BrokenPipe))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut state = StreamState::new(Dead);
        assert_eq!(state.head(), Err(StreamError::Disconnected));
        assert_eq!(state.failure(), Some(StreamError::Disconnected));
        assert_eq!(state.emit("data: x\n\n"), Err(StreamError::Disconnected));
    }
}
