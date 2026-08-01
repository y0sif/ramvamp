//! Byte-range access to a model source.
//!
//! [`RangeRead`] is the one primitive the repacker needs from a source: an
//! exact-fill positioned read out of a large read-only blob. A local GGUF
//! file implements it ([`LocalFile`]), and a remote HTTP source does the
//! same over `Range` requests ([`RemoteFile`]) behind the same object-safe
//! trait. Offsets and lengths ultimately come from untrusted file headers,
//! so implementations bounds-check every request and return typed errors
//! instead of panicking.

use std::fs::File;
use std::io::{self, Read};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use ureq::http::{Response, StatusCode, header};

/// Error reading a byte range out of a source.
#[derive(Debug, thiserror::Error)]
pub enum SourceError {
    /// The source could not be opened or stat'd.
    #[error("failed to open source {path:?}")]
    Open {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    /// The requested range does not lie inside the source.
    #[error("range at offset {offset} of {len} bytes out of bounds for {source_len}-byte source")]
    OutOfBounds {
        offset: u64,
        len: u64,
        source_len: u64,
    },
    /// The source ended before the buffer was filled (exact-fill violation).
    #[error("short read at offset {offset}: wanted {wanted} bytes")]
    ShortRead { offset: u64, wanted: usize },
    /// Underlying I/O failure.
    #[error("read failed at offset {offset}")]
    Io {
        offset: u64,
        #[source]
        source: io::Error,
    },
    /// HTTP transport failure talking to a remote source.
    #[error("http request to {url} failed")]
    Http {
        url: String,
        #[source]
        source: Box<ureq::Error>,
    },
    /// The server answered with an HTTP status we cannot use.
    #[error("unexpected HTTP status {status} from {url} (expected {expected})")]
    HttpStatus {
        url: String,
        status: u16,
        expected: &'static str,
    },
    /// The redirect chain did not settle within the hop budget.
    #[error("too many redirects resolving {url}")]
    TooManyRedirects { url: String },
    /// A redirect response had a missing or unusable `Location` header.
    #[error("unusable redirect Location {location:?} from {url}")]
    BadRedirect { url: String, location: String },
    /// The total length of the remote source could not be determined.
    #[error("cannot determine total length of {url}: {detail}")]
    BadLength { url: String, detail: String },
    /// A ranged response carried a body length other than the one requested.
    #[error("range response at offset {offset}: got {got} bytes, wanted {wanted}")]
    RangeLenMismatch { offset: u64, got: u64, wanted: u64 },
}

/// A read-only source of bytes addressable by absolute offset.
///
/// `read_at` has exact-fill semantics: on `Ok(())` the whole buffer was
/// filled from `offset`; a short read is an error, never a partial fill.
/// Implementations must tolerate arbitrary offsets/lengths (they come from
/// untrusted headers) and answer with [`SourceError::OutOfBounds`] rather
/// than panicking or truncating.
///
/// The trait is object-safe by design: parsers take `&dyn RangeRead` so
/// local files and (later) remote ranged-HTTP sources are interchangeable.
pub trait RangeRead {
    /// Total size of the source in bytes.
    fn len(&self) -> u64;

    /// Whether the source is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fill `buf` exactly with the bytes at `offset..offset + buf.len()`.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), SourceError>;
}

/// A local file source using positioned reads (`pread` via
/// [`FileExt::read_exact_at`]). Linux-first, like the rest of the crate.
///
/// The length is captured at open time; if the file shrinks underneath us,
/// reads fail with [`SourceError::ShortRead`] instead of blocking or lying.
#[derive(Debug)]
pub struct LocalFile {
    file: File,
    len: u64,
}

impl LocalFile {
    /// Open `path` read-only and capture its current length.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SourceError> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|source| SourceError::Open {
            path: path.to_owned(),
            source,
        })?;
        let len = file
            .metadata()
            .map_err(|source| SourceError::Open {
                path: path.to_owned(),
                source,
            })?
            .len();
        Ok(Self { file, len })
    }
}

impl RangeRead for LocalFile {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), SourceError> {
        let want = buf.len() as u64;
        let in_bounds = offset.checked_add(want).is_some_and(|end| end <= self.len);
        if !in_bounds {
            return Err(SourceError::OutOfBounds {
                offset,
                len: want,
                source_len: self.len,
            });
        }
        self.file.read_exact_at(buf, offset).map_err(|e| {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                SourceError::ShortRead {
                    offset,
                    wanted: buf.len(),
                }
            } else {
                SourceError::Io { offset, source: e }
            }
        })
    }
}

/// In-memory source over a borrowed byte slice. Used by tests and small
/// fixtures; never for whole models (see the no-materialization hard rule).
impl RangeRead for &[u8] {
    fn len(&self) -> u64 {
        <[u8]>::len(self) as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), SourceError> {
        let source_len = <[u8]>::len(self) as u64;
        let want = buf.len() as u64;
        let in_bounds = offset
            .checked_add(want)
            .is_some_and(|end| end <= source_len);
        if !in_bounds {
            return Err(SourceError::OutOfBounds {
                offset,
                len: want,
                source_len,
            });
        }
        // `offset` and `offset + want` fit in usize: both are <= source_len,
        // which is itself a usize.
        let start = offset as usize;
        buf.copy_from_slice(&self[start..start + buf.len()]);
        Ok(())
    }
}

/// User-Agent sent with every remote request.
const HTTP_USER_AGENT: &str = "ramvamp-repack/0.1";
/// Attempts per range request: one initial try plus retries.
const HTTP_ATTEMPTS: u32 = 5;
/// Backoff before the first retry; doubles per subsequent retry. With
/// [`HTTP_ATTEMPTS`] = 5 the sleeps are 1+2+4+8 s, so a read tolerates
/// roughly 15 s of accumulated outage before giving up.
const HTTP_BACKOFF: Duration = Duration::from_secs(1);
/// Max duration for establishing a connection (socket + TLS handshake).
/// A healthy CDN connects in well under a second; 15 s absorbs slow DNS
/// or congested links without hanging forever on a dead host.
const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
/// Max duration for the response headers to arrive once the request is
/// sent. Header turnaround is size-independent, so 60 s is generous.
const HTTP_RECV_RESPONSE_TIMEOUT: Duration = Duration::from_secs(60);
/// Max duration for receiving one response body. Bodies are at most one
/// transfer window (32 MiB by default, 1 GiB cap), so 600 s still admits
/// a slow-but-live ~56 KiB/s link on the default window while cutting a
/// stalled connection instead of blocking `read_exact` indefinitely.
const HTTP_RECV_BODY_TIMEOUT: Duration = Duration::from_secs(600);
/// Redirect hops followed when resolving a URL. Hugging Face `resolve`
/// URLs 302 once to a CDN; a few extra hops are tolerated.
const MAX_REDIRECT_HOPS: usize = 5;

/// A remote HTTP(S) source read through `Range: bytes=start-end` requests.
///
/// Redirects are followed manually (rather than relying on ureq's
/// automatic redirect handling), re-sending the `Range` header on every
/// hop, so the range is guaranteed to reach the host that actually serves
/// bytes. The total length is resolved once at open time from a 1-byte
/// range probe's `Content-Range`.
///
/// The redirect target cannot always be cached: Hugging Face's Xet bridge
/// signs the *exact* `Range` header into the redirect URL's policy
/// (`ByteRange.ExpectedHeader`), so a URL resolved for one range answers
/// 403 for every other range. `open` probes the resolved URL with a
/// different range once: if the CDN accepts it (S3-style presigned URLs
/// do), the URL is cached and reads hit it directly; otherwise every read
/// re-resolves from the original URL with its own range (two requests per
/// read). A cached URL that later goes stale (signed URLs expire) demotes
/// to per-read resolution.
///
/// `read_at` expects `206 Partial Content` with an exact-length body and
/// retries transient failures ([`HTTP_ATTEMPTS`] attempts, exponential
/// backoff from [`HTTP_BACKOFF`]). Public repos only; no authentication.
#[derive(Debug)]
pub struct RemoteFile {
    agent: ureq::Agent,
    /// URL as given by the caller; the fallback start of every resolution.
    original_url: String,
    /// Redirect-resolved URL proven to accept arbitrary ranges, or `None`
    /// when the CDN pins signed URLs to one exact range.
    reusable_url: Mutex<Option<String>>,
    len: u64,
}

impl RemoteFile {
    /// Resolve `url` (following redirects with the `Range` header intact),
    /// determine the total length via a 1-byte range probe, and return a
    /// ready-to-read source.
    pub fn open(url: &str) -> Result<Self, SourceError> {
        let config = ureq::Agent::config_builder()
            // Statuses are inspected by hand: probes must see 3xx/206.
            .http_status_as_error(false)
            // With 0, ureq returns redirect responses as-is (never errors),
            // letting `ranged_get` follow them with Range intact.
            .max_redirects(0)
            // ureq 3.x defaults every timeout to None, so a stalled
            // connection would hang `read_exact` forever without these.
            // Per-phase timeouts only: a global or per-call deadline
            // would also kill legitimately slow window-sized reads.
            .timeout_connect(Some(HTTP_CONNECT_TIMEOUT))
            .timeout_recv_response(Some(HTTP_RECV_RESPONSE_TIMEOUT))
            .timeout_recv_body(Some(HTTP_RECV_BODY_TIMEOUT))
            .user_agent(HTTP_USER_AGENT)
            .build();
        let agent = ureq::Agent::new_with_config(config);
        let (final_url, probe) = ranged_get(&agent, url, "bytes=0-0")?;
        let len = total_len_from_probe(&final_url, &probe)?;
        let reusable = if final_url == url {
            // No redirect happened; the origin serves ranges itself.
            Some(final_url)
        } else if len >= 2 && accepts_other_ranges(&agent, &final_url) {
            Some(final_url)
        } else {
            None
        };
        tracing::debug!(
            url,
            len,
            reusable = reusable.is_some(),
            "opened remote source"
        );
        Ok(RemoteFile {
            agent,
            original_url: url.to_owned(),
            reusable_url: Mutex::new(reusable),
            len,
        })
    }

    /// One ranged GET for `buf.len()` bytes at `offset`, starting from
    /// `url` and following redirects, expecting 206 and an exact-length
    /// body.
    fn fetch_range(&self, url: &str, offset: u64, buf: &mut [u8]) -> Result<(), SourceError> {
        let want = buf.len() as u64;
        // Callers bounds-check first, and `buf` is non-empty here, so the
        // inclusive end cannot underflow or overflow.
        let range = format!("bytes={}-{}", offset, offset + want - 1);
        let (landed_url, mut resp) = ranged_get(&self.agent, url, &range)?;
        let status = resp.status();
        if status != StatusCode::PARTIAL_CONTENT {
            return Err(SourceError::HttpStatus {
                url: landed_url,
                status: status.as_u16(),
                expected: "206 Partial Content",
            });
        }
        // Exact fill means exact length: a longer body would silently
        // desynchronize the caller, a shorter one is a short read.
        let content_length = resp.body().content_length();
        if let Some(got) = content_length
            && got != want
        {
            return Err(SourceError::RangeLenMismatch {
                offset,
                got,
                wanted: want,
            });
        }
        let mut reader = resp.body_mut().as_reader();
        reader.read_exact(buf).map_err(|e| {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                SourceError::ShortRead {
                    offset,
                    wanted: buf.len(),
                }
            } else {
                SourceError::Io { offset, source: e }
            }
        })?;
        if content_length.is_none() {
            // Chunked/decompressed body: no length to check up front, so
            // verify the body ends exactly where the range does.
            let mut probe = [0u8; 1];
            let n = reader
                .read(&mut probe)
                .map_err(|e| SourceError::Io { offset, source: e })?;
            if n != 0 {
                return Err(SourceError::RangeLenMismatch {
                    offset,
                    got: want + 1,
                    wanted: want,
                });
            }
        }
        Ok(())
    }
}

impl RangeRead for RemoteFile {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), SourceError> {
        let want = buf.len() as u64;
        let in_bounds = offset.checked_add(want).is_some_and(|end| end <= self.len);
        if !in_bounds {
            return Err(SourceError::OutOfBounds {
                offset,
                len: want,
                source_len: self.len,
            });
        }
        if buf.is_empty() {
            return Ok(());
        }
        let mut backoff = HTTP_BACKOFF;
        let mut last_err = None;
        for attempt in 0..HTTP_ATTEMPTS {
            if attempt > 0 {
                thread::sleep(backoff);
                backoff *= 2;
            }
            let cached = lock_ignore_poison(&self.reusable_url).clone();
            let url = cached.clone().unwrap_or_else(|| self.original_url.clone());
            match self.fetch_range(&url, offset, buf) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    if cached.is_some() && url_went_stale(&e) {
                        // Expired signed CDN URL: demote to per-read
                        // resolution from the original URL.
                        tracing::warn!(offset, error = %e, "cached remote URL stale; dropping it");
                        *lock_ignore_poison(&self.reusable_url) = None;
                    } else if is_transient(&e) {
                        tracing::warn!(offset, attempt, error = %e, "retrying remote read");
                    } else {
                        return Err(e);
                    }
                    last_err = Some(e);
                }
            }
        }
        // The loop ran at least once, so an error was recorded.
        Err(last_err.unwrap_or(SourceError::ShortRead {
            offset,
            wanted: buf.len(),
        }))
    }
}

/// GET `url` with the given `Range` header, following redirects by hand
/// and re-sending the header on every hop so it always reaches the host
/// that serves bytes. Returns the settled URL and its (non-redirect)
/// response; the caller judges the status.
fn ranged_get(
    agent: &ureq::Agent,
    url: &str,
    range: &str,
) -> Result<(String, Response<ureq::Body>), SourceError> {
    let mut current = url.to_owned();
    for _ in 0..=MAX_REDIRECT_HOPS {
        let resp = agent
            .get(&current)
            .header("Range", range)
            .call()
            .map_err(|e| SourceError::Http {
                url: current.clone(),
                source: Box::new(e),
            })?;
        if !resp.status().is_redirection() {
            return Ok((current, resp));
        }
        let location = resp
            .headers()
            .get(header::LOCATION)
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| SourceError::BadRedirect {
                url: current.clone(),
                location: "<missing>".to_owned(),
            })?;
        current = absolutize(&current, location)?;
    }
    Err(SourceError::TooManyRedirects {
        url: url.to_owned(),
    })
}

/// Whether a redirect-resolved URL accepts a `Range` other than the one it
/// was resolved with. Hugging Face's Xet bridge signs the exact expected
/// `Range` header into the URL policy and answers 403 to anything else;
/// S3-style presigned URLs accept any range.
fn accepts_other_ranges(agent: &ureq::Agent, url: &str) -> bool {
    match agent.get(url).header("Range", "bytes=1-1").call() {
        Ok(resp) => resp.status() == StatusCode::PARTIAL_CONTENT,
        Err(e) => {
            tracing::debug!(url, error = %e, "range-reuse probe failed");
            false
        }
    }
}

/// Resolve a redirect `Location` against the URL that produced it.
/// Absolute URLs pass through; absolute paths reuse the base's
/// scheme+authority. Relative-path redirects are rejected (never produced
/// by the hosts we talk to, and not worth a URL library).
fn absolutize(base: &str, location: &str) -> Result<String, SourceError> {
    if location.starts_with("https://") || location.starts_with("http://") {
        return Ok(location.to_owned());
    }
    if location.starts_with('/')
        && let Some(scheme_end) = base.find("://")
    {
        let after_scheme = &base[scheme_end + 3..];
        let authority_end = after_scheme
            .find('/')
            .map_or(base.len(), |i| scheme_end + 3 + i);
        return Ok(format!("{}{}", &base[..authority_end], location));
    }
    Err(SourceError::BadRedirect {
        url: base.to_owned(),
        location: location.to_owned(),
    })
}

/// Extract the total length from a 1-byte range probe: the response must be
/// `206 Partial Content` with a parseable `Content-Range: bytes 0-0/total`.
fn total_len_from_probe(url: &str, resp: &Response<ureq::Body>) -> Result<u64, SourceError> {
    let status = resp.status();
    if status != StatusCode::PARTIAL_CONTENT {
        return Err(SourceError::HttpStatus {
            url: url.to_owned(),
            status: status.as_u16(),
            expected: "206 Partial Content (server must support range requests)",
        });
    }
    let value = resp
        .headers()
        .get(header::CONTENT_RANGE)
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| SourceError::BadLength {
            url: url.to_owned(),
            detail: "missing Content-Range header".to_owned(),
        })?;
    parse_content_range_total(value).ok_or_else(|| SourceError::BadLength {
        url: url.to_owned(),
        detail: format!("unparseable Content-Range {value:?}"),
    })
}

/// Parse the total out of `bytes <start>-<end>/<total>`. Returns `None`
/// for the `*` unknown-total form or anything malformed.
fn parse_content_range_total(value: &str) -> Option<u64> {
    let rest = value.trim().strip_prefix("bytes")?.trim_start();
    let (_, total) = rest.rsplit_once('/')?;
    total.trim().parse::<u64>().ok()
}

/// Whether the cached CDN URL should be re-resolved: signed URLs expire
/// (403) or the host starts redirecting again.
fn url_went_stale(e: &SourceError) -> bool {
    matches!(
        e,
        SourceError::HttpStatus { status, .. }
            if *status == 403 || (300..400).contains(status)
    )
}

/// Whether a failure is worth retrying: transport errors, truncated
/// bodies, and 5xx responses. Everything else (404, 416, ...) fails fast.
fn is_transient(e: &SourceError) -> bool {
    match e {
        SourceError::Http { .. } | SourceError::ShortRead { .. } | SourceError::Io { .. } => true,
        SourceError::RangeLenMismatch { .. } => true,
        SourceError::HttpStatus { status, .. } => *status >= 500,
        _ => false,
    }
}

/// Lock a mutex, treating poison as harmless: the guarded value is a plain
/// `String`, always left in a valid state.
fn lock_ignore_poison<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_reads_exact_ranges() {
        let data: &[u8] = &[10, 20, 30, 40, 50];
        assert_eq!(RangeRead::len(&data), 5);
        assert!(!RangeRead::is_empty(&data));
        let mut buf = [0u8; 3];
        data.read_at(1, &mut buf).unwrap();
        assert_eq!(buf, [20, 30, 40]);
    }

    #[test]
    fn slice_rejects_out_of_bounds() {
        let data: &[u8] = &[1, 2, 3];
        let mut buf = [0u8; 2];
        let err = data.read_at(2, &mut buf).unwrap_err();
        assert!(matches!(
            err,
            SourceError::OutOfBounds {
                offset: 2,
                len: 2,
                source_len: 3
            }
        ));
        // Offset + len overflowing u64 must not panic.
        let err = data.read_at(u64::MAX, &mut buf).unwrap_err();
        assert!(matches!(err, SourceError::OutOfBounds { .. }));
    }

    #[test]
    fn empty_slice_is_empty() {
        let data: &[u8] = &[];
        assert!(RangeRead::is_empty(&data));
        let mut buf = [0u8; 1];
        assert!(data.read_at(0, &mut buf).is_err());
    }

    #[test]
    fn local_file_open_missing_is_typed_error() {
        let err = LocalFile::open("/nonexistent/ramvamp-test-no-such-file").unwrap_err();
        assert!(matches!(err, SourceError::Open { .. }));
    }

    #[test]
    fn content_range_total_parses() {
        assert_eq!(parse_content_range_total("bytes 0-0/12345"), Some(12345));
        assert_eq!(
            parse_content_range_total("bytes 100-199/31284617216"),
            Some(31284617216)
        );
        assert_eq!(parse_content_range_total("bytes 0-0/*"), None);
        assert_eq!(parse_content_range_total("pages 0-0/5"), None);
        assert_eq!(parse_content_range_total("bytes 0-0"), None);
        assert_eq!(parse_content_range_total(""), None);
    }

    #[test]
    fn absolutize_handles_absolute_and_rooted_locations() {
        assert_eq!(
            absolutize("https://a.example/x", "https://cdn.example/y?sig=1").unwrap(),
            "https://cdn.example/y?sig=1"
        );
        assert_eq!(
            absolutize("https://a.example/deep/path?q=1", "/other/file").unwrap(),
            "https://a.example/other/file"
        );
        let err = absolutize("https://a.example/x", "relative/path").unwrap_err();
        assert!(matches!(err, SourceError::BadRedirect { .. }));
    }

    #[test]
    fn stale_and_transient_classification() {
        let stale = SourceError::HttpStatus {
            url: "u".into(),
            status: 403,
            expected: "206",
        };
        assert!(url_went_stale(&stale));
        let redirect = SourceError::HttpStatus {
            url: "u".into(),
            status: 302,
            expected: "206",
        };
        assert!(url_went_stale(&redirect));
        let server_err = SourceError::HttpStatus {
            url: "u".into(),
            status: 503,
            expected: "206",
        };
        assert!(is_transient(&server_err) && !url_went_stale(&server_err));
        let not_found = SourceError::HttpStatus {
            url: "u".into(),
            status: 404,
            expected: "206",
        };
        assert!(!is_transient(&not_found) && !url_went_stale(&not_found));
    }
}
