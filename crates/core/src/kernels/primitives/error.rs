//! Typed errors for the primitive kernels.
//!
//! The dedicated `PrimitiveError` enum that used to live here has been
//! unified with the quants error into the shared
//! [`crate::kernels::KernelError`] (see `kernels/error.rs`), so the backend
//! trait surfaces one error type. The old name stays as an alias; variant
//! names and semantics are unchanged.

pub use crate::kernels::KernelError as PrimitiveError;
