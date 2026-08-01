//! Byte-range access to a model source.
//!
//! [`RangeRead`] is the one primitive the repacker needs from a source: an
//! exact-fill positioned read out of a large read-only blob. A local GGUF
//! file implements it today ([`LocalFile`]); a remote HTTP range-request
//! source plugs in later behind the same object-safe trait. Offsets and
//! lengths ultimately come from untrusted file headers, so implementations
//! bounds-check every request and return typed errors instead of panicking.

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

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
}
