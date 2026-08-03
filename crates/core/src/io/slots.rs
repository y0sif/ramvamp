//! Page-aligned expert slot pool.
//!
//! One allocation of `n_layers * slots_per_layer` slots, each sized to its
//! layer's blob stride and based at a 4096-aligned address, faulted in at
//! construction, handed out through an explicit free list. These are the
//! destination buffers that io_uring + `O_DIRECT` reads land in and that the
//! expert GEMVs then read from.
//!
//! Two hard requirements, both measured rather than assumed:
//!
//! - The backing pages must be touched before any O_DIRECT read targets
//!   them. btrfs runs direct reads with page faults disabled
//!   (`fs/btrfs/direct-io.c`), and when it cannot fault the destination it
//!   silently completes the transfer through the buffered path instead —
//!   no error, full byte count, and the page cache grows inside our cgroup.
//! - A slot backing an in-flight read is never handed to a second reader.
//!   Aliasing two concurrent O_DIRECT reads onto one buffer makes btrfs
//!   fail checksum verification (measured 13-27% spurious EIO, and it
//!   increments the filesystem's persistent `corruption_errs`). Index
//!   arithmetic is not sufficient, because completions arrive out of order.
//!
//! # Alignment contract
//!
//! btrfs requires 4096 on the file offset, the transfer length **and** the
//! destination address. The device's 512-byte logical block size is not
//! sufficient, and `statx(STATX_DIOALIGN)` is unsupported on btrfs so the
//! real requirement cannot be probed at runtime — it is hard-coded as
//! [`SLOT_ALIGN`] and asserted in the tests.
//!
//! The slab is allocated once with `Layout::from_size_align(_, SLOT_ALIGN)`
//! and freed in `Drop` with that identical layout. Every slot's *pitch* (the
//! distance to the next slot) is the layer's stride rounded up to
//! [`SLOT_ALIGN`], so each slot base is `base + k * 4096` and therefore
//! 4096-aligned too. The real installed geometry needs no rounding — both
//! Qwen3-30B-A3B strides (3,059,712 B and 2,654,208 B) are exact multiples
//! of 4096 — but the pool stays correct for a layout that is not, at a cost
//! of under one page per slot. A slot's *length* is the declared stride, not
//! the pitch; a caller issuing O_DIRECT reads still owes the filesystem a
//! 4096-multiple length, which is the layout's job, not the pool's.
//!
//! # Aliasing contract
//!
//! [`SlotPool::acquire`] is the only way to obtain a slot and it yields an
//! owning [`SlotGuard`]. A guard is the unique lease on its slot: the slot
//! index leaves the layer's free stack on acquire and only returns on the
//! guard's `Drop`. There is no by-index accessor, so two concurrent readers
//! aliasing one buffer is not expressible. Guards are `Send`, so the io_uring
//! path parks them in its in-flight table (keyed by `user_data`) for the
//! lifetime of the read and drops them after the completion is consumed.
//!
//! # Allocation
//!
//! Everything is allocated in [`SlotPool::new`]: the slab, the per-slot
//! occupancy flags, and the per-layer free stacks (each at exactly
//! `slots_per_layer` capacity, which they never exceed). `acquire` is a
//! `Vec::pop` and release is a `Vec::push` within that capacity, so the
//! decode loop never allocates.


use std::alloc::{Layout, alloc, dealloc};
use std::ptr::NonNull;
use std::slice;
use std::sync::{Mutex, MutexGuard, PoisonError};

use thiserror::Error;

/// Required alignment of every slot base address, in bytes.
///
/// btrfs's O_DIRECT path rejects anything coarser than this on the buffer
/// address, and the requirement is not probeable (`STATX_DIOALIGN` is
/// unimplemented there), so it is fixed rather than discovered.
pub const SLOT_ALIGN: usize = 4096;

/// Largest slab [`SlotPool::new`] will attempt, in bytes (64 GiB).
///
/// The design target is 12 slots/layer over 48 layers = 1569 MiB. The cap
/// exists so that a corrupt or hostile layout turns into a typed error
/// instead of an allocator abort.
pub const MAX_POOL_BYTES: u64 = 64 << 30;

/// Failures from sizing, allocating, or leasing expert slots.
///
/// Geometry reaches this module from `experts/layout.json`, which is
/// untrusted until validated, so every rejection is a recoverable error and
/// never a panic or an OOM abort.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SlotError {
    /// `slots_per_layer` was zero; a pool with no slots can never serve a read.
    #[error("slot pool needs at least one slot per layer")]
    ZeroSlotsPerLayer,

    /// The stride list was empty.
    #[error("slot pool needs at least one layer")]
    NoLayers,

    /// More layers than a `u32` index can address.
    #[error("slot pool layer count {n_layers} exceeds u32")]
    TooManyLayers {
        /// Number of strides supplied.
        n_layers: usize,
    },

    /// A layer declared a zero-byte blob stride. Zero-length slots would all
    /// share one address, which is exactly the aliasing this pool prevents.
    #[error("layer {layer}: blob stride is zero")]
    ZeroStride {
        /// Offending layer index.
        layer: u32,
    },

    /// The requested geometry overflowed or exceeded [`MAX_POOL_BYTES`].
    #[error("slot pool needs {requested} bytes, over the {limit}-byte cap")]
    PoolTooLarge {
        /// Bytes the geometry asked for, computed in 128-bit to survive overflow.
        requested: u128,
        /// The cap that was exceeded.
        limit: u128,
    },

    /// The allocator refused the slab.
    #[error("slot pool allocation of {bytes} bytes failed")]
    AllocFailed {
        /// Size of the refused request.
        bytes: usize,
    },

    /// A layer index is at or past the pool's layer count.
    #[error("layer {layer} out of range ({n_layers} layers)")]
    LayerOutOfRange {
        /// Requested layer index.
        layer: u32,
        /// Layers in the pool.
        n_layers: u32,
    },

    /// Every slot for the layer is already leased.
    #[error("layer {layer}: all {slots_per_layer} slots are in use")]
    Exhausted {
        /// Layer whose free stack is empty.
        layer: u32,
        /// Slots the layer owns.
        slots_per_layer: u32,
    },
}

/// Resolved geometry of one layer's slot run, fixed at construction.
#[derive(Debug, Clone, Copy)]
struct LayerGeom {
    /// Byte offset of the layer's first slot from the slab base. A multiple
    /// of [`SLOT_ALIGN`], because every pitch is.
    base_offset: usize,
    /// Usable bytes per slot: the layer's declared blob stride.
    stride: usize,
    /// Distance between consecutive slot bases: `stride` rounded up to
    /// [`SLOT_ALIGN`].
    pitch: usize,
}

/// Mutable pool bookkeeping. Both halves are sized at construction and never
/// grow.
#[derive(Debug)]
struct PoolState {
    /// Per-slot lease flag, flat-indexed `layer * slots_per_layer + slot`.
    /// Redundant with the free stacks by construction; kept as the explicit
    /// per-slot state that makes a double release detectable rather than
    /// silently corrupting the free list.
    leased: Box<[bool]>,
    /// Per-layer stack of free slot indices, capacity `slots_per_layer`.
    free: Box<[Vec<u32>]>,
}

/// A pool of page-aligned, pre-faulted expert blob buffers.
///
/// One owning allocation is carved into `slots_per_layer` slots for each
/// layer, each slot sized to that layer's blob stride. See the module docs
/// for the alignment, aliasing, and allocation contracts.
///
/// Usage sketch (not a doctest until the module is re-exported from
/// `io/mod.rs`):
///
/// ```text
/// // 48 layers with the real Qwen3-30B-A3B strides, 12 slots each.
/// let pool = SlotPool::new(12, &layer_strides)?;
/// let mut slot = pool.acquire(layer)?;
/// let dst: *mut u8 = slot.as_mut_ptr();   // io_uring read destination
/// // ... submit, park `slot` in the in-flight table, wait for the CQE ...
/// let filled: &[u8] = slot.as_slice();    // compute reads the filled blob
/// // dropping `slot` returns it to that layer's free stack
/// ```
#[derive(Debug)]
pub struct SlotPool {
    /// Base of the single owning allocation, aligned to [`SLOT_ALIGN`].
    base: NonNull<u8>,
    /// The exact layout `base` was allocated with; `Drop` frees with it.
    alloc_layout: Layout,
    /// Total slab size in bytes.
    total_bytes: usize,
    /// Slots held for every layer.
    slots_per_layer: u32,
    /// Per-layer geometry, indexed by layer.
    layers: Box<[LayerGeom]>,
    /// Free stacks and lease flags.
    state: Mutex<PoolState>,
}

// SAFETY: `SlotPool` owns its allocation exclusively for its whole lifetime
// and never dereferences `base` itself — it only computes offsets from it.
// Every byte handed out is handed out through a `SlotGuard`, and which slot a
// guard covers is decided under `state`'s mutex, so two threads can never
// obtain overlapping views. The raw pointer is the only reason the automatic
// impls do not apply.
unsafe impl Send for SlotPool {}
// SAFETY: see the `Send` impl. `&SlotPool` exposes only `acquire`, whose
// mutation goes through the mutex, and immutable geometry accessors.
unsafe impl Sync for SlotPool {}

impl SlotPool {
    /// Allocate, pre-fault, and index a pool of `slots_per_layer` slots for
    /// each stride in `layer_strides`.
    ///
    /// Sizing arithmetic runs in 128-bit, so an absurd geometry is reported
    /// rather than wrapped, and the slab is allocated with a null check so a
    /// refusal is an error rather than an abort. Construction touches every
    /// page, which for the 1569 MiB target configuration is the dominant
    /// cost of this call.
    ///
    /// # Errors
    ///
    /// [`SlotError::ZeroSlotsPerLayer`], [`SlotError::NoLayers`],
    /// [`SlotError::TooManyLayers`], or [`SlotError::ZeroStride`] for a
    /// degenerate geometry; [`SlotError::PoolTooLarge`] when the slab would
    /// exceed [`MAX_POOL_BYTES`] (or this platform's `isize::MAX`);
    /// [`SlotError::AllocFailed`] when the allocator returns null.
    pub fn new(slots_per_layer: u32, layer_strides: &[u64]) -> Result<Self, SlotError> {
        if slots_per_layer == 0 {
            return Err(SlotError::ZeroSlotsPerLayer);
        }
        if layer_strides.is_empty() {
            return Err(SlotError::NoLayers);
        }
        let n_layers =
            u32::try_from(layer_strides.len()).map_err(|_| SlotError::TooManyLayers {
                n_layers: layer_strides.len(),
            })?;

        // Cap at the smaller of the policy limit and what this platform can
        // even address, so the `usize` narrowing below cannot fail.
        let cap = u128::from(MAX_POOL_BYTES).min(isize::MAX as u128);
        let align = SLOT_ALIGN as u64;

        let mut layers = Vec::with_capacity(layer_strides.len());
        let mut total: u128 = 0;
        for (index, &stride) in layer_strides.iter().enumerate() {
            let layer = index as u32;
            if stride == 0 {
                return Err(SlotError::ZeroStride { layer });
            }
            // Rounding up keeps every slot base a multiple of SLOT_ALIGN even
            // when a layout's stride is not.
            let pitch = stride
                .checked_next_multiple_of(align)
                .ok_or(SlotError::PoolTooLarge {
                    requested: u128::from(stride),
                    limit: cap,
                })?;
            let base_offset = total;
            total += u128::from(pitch) * u128::from(slots_per_layer);
            if total > cap {
                return Err(SlotError::PoolTooLarge {
                    requested: total,
                    limit: cap,
                });
            }
            layers.push(LayerGeom {
                // In range: `total <= cap <= isize::MAX` was just checked, and
                // `base_offset < total`.
                base_offset: base_offset as usize,
                stride: stride as usize,
                pitch: pitch as usize,
            });
        }
        // In range for the same reason.
        let total_bytes = total as usize;

        let alloc_layout = Layout::from_size_align(total_bytes, SLOT_ALIGN).map_err(|_| {
            SlotError::PoolTooLarge {
                requested: total,
                limit: cap,
            }
        })?;
        // SAFETY: `total_bytes` is non-zero — there is at least one layer and
        // one slot, and every pitch is at least SLOT_ALIGN — which is the only
        // precondition of `alloc` beyond a valid layout.
        let raw = unsafe { alloc(alloc_layout) };
        let base = NonNull::new(raw).ok_or(SlotError::AllocFailed { bytes: total_bytes })?;

        // SAFETY: `base` is the start of `total_bytes` freshly allocated,
        // exclusively owned bytes.
        unsafe { prefault(base, total_bytes) };

        let n_slots = layers.len() * slots_per_layer as usize;
        let free = (0..layers.len())
            .map(|_| {
                let mut stack = Vec::with_capacity(slots_per_layer as usize);
                // Reversed so `pop` hands out ascending slot indices, which
                // keeps the early-life access pattern sequential.
                stack.extend((0..slots_per_layer).rev());
                stack
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        tracing::debug!(
            n_layers,
            slots_per_layer,
            total_bytes,
            align = SLOT_ALIGN,
            "expert slot pool allocated and pre-faulted"
        );

        Ok(Self {
            base,
            alloc_layout,
            total_bytes,
            slots_per_layer,
            layers: layers.into_boxed_slice(),
            state: Mutex::new(PoolState {
                leased: vec![false; n_slots].into_boxed_slice(),
                free,
            }),
        })
    }

    /// Slots held for each layer.
    pub fn slots_per_layer(&self) -> u32 {
        self.slots_per_layer
    }

    /// Number of layers the pool covers.
    pub fn n_layers(&self) -> u32 {
        // Bounded by the `u32::try_from` in `new`.
        self.layers.len() as u32
    }

    /// Usable bytes in one of `layer`'s slots: the layer's declared blob
    /// stride, not the padded pitch.
    ///
    /// # Panics
    ///
    /// If `layer >= self.n_layers()`. Layer indices are program state, not
    /// untrusted input; use [`SlotPool::acquire`] when the index may be out
    /// of range, which reports [`SlotError::LayerOutOfRange`] instead.
    #[track_caller]
    pub fn stride(&self, layer: u32) -> usize {
        self.layers[layer as usize].stride
    }

    /// Distance between consecutive slot bases in `layer`: [`SlotPool::stride`]
    /// rounded up to [`SLOT_ALIGN`]. Equal to the stride for any
    /// page-aligned layout, including the installed Qwen3 one.
    ///
    /// # Panics
    ///
    /// If `layer >= self.n_layers()`.
    #[track_caller]
    pub fn pitch(&self, layer: u32) -> usize {
        self.layers[layer as usize].pitch
    }

    /// Largest per-slot stride across all layers — the size a scratch buffer
    /// must have to hold any expert blob.
    pub fn max_stride(&self) -> usize {
        self.layers.iter().map(|l| l.stride).max().unwrap_or(0)
    }

    /// Total resident bytes owned by the pool.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Slots currently free in `layer`, or `None` past the last layer.
    ///
    /// Advisory only: with concurrent readers the value can change before the
    /// caller acts on it. Use it for accounting, not for deciding whether
    /// [`SlotPool::acquire`] will succeed.
    pub fn free_slots(&self, layer: u32) -> Option<u32> {
        if layer >= self.n_layers() {
            return None;
        }
        let state = self.lock();
        // In range: checked above, and `free` has one entry per layer.
        Some(state.free[layer as usize].len() as u32)
    }

    /// Take the unique lease on a free slot of `layer`.
    ///
    /// The returned guard is the only handle to those bytes until it is
    /// dropped, which is what makes concurrent O_DIRECT reads onto one buffer
    /// unrepresentable.
    ///
    /// # Errors
    ///
    /// [`SlotError::LayerOutOfRange`] for an unknown layer;
    /// [`SlotError::Exhausted`] when every slot of the layer is leased. The
    /// caller decides what exhaustion means — wait for a completion, or fall
    /// back to a synchronous read.
    pub fn acquire(&self, layer: u32) -> Result<SlotGuard<'_>, SlotError> {
        let geom = *self
            .layers
            .get(layer as usize)
            .ok_or(SlotError::LayerOutOfRange {
                layer,
                n_layers: self.n_layers(),
            })?;
        let slot = {
            let mut state = self.lock();
            // In range: `layer` indexed `layers` successfully above.
            let slot = state.free[layer as usize]
                .pop()
                .ok_or(SlotError::Exhausted {
                    layer,
                    slots_per_layer: self.slots_per_layer,
                })?;
            let flat = self.flat_index(layer, slot);
            debug_assert!(!state.leased[flat], "free stack yielded a leased slot");
            state.leased[flat] = true;
            slot
        };

        // No overflow and in bounds: `pitch * slots_per_layer` was summed into
        // `total_bytes` in `new`, and `slot < slots_per_layer`.
        let offset = geom.base_offset + geom.pitch * slot as usize;
        debug_assert!(offset + geom.stride <= self.total_bytes);
        // SAFETY: `offset + stride <= total_bytes`, so the result stays within
        // the one allocation `base` points into, and a pointer one past the
        // end would still be well-defined. Non-null because `base` is.
        let ptr = unsafe { NonNull::new_unchecked(self.base.as_ptr().add(offset)) };
        Ok(SlotGuard {
            pool: self,
            layer,
            slot,
            ptr,
            len: geom.stride,
        })
    }

    /// Return a slot to its layer's free stack. Called only by
    /// [`SlotGuard::drop`], which guarantees one call per lease.
    fn release(&self, layer: u32, slot: u32) {
        let flat = self.flat_index(layer, slot);
        let mut state = self.lock();
        if !state.leased[flat] {
            // Unreachable while `SlotGuard` is the only lease handle. Logged
            // rather than panicked so a bug here degrades into a leaked slot
            // instead of unwinding through a drop.
            tracing::error!(layer, slot, "slot released without an active lease");
            return;
        }
        state.leased[flat] = false;
        let stack = &mut state.free[layer as usize];
        debug_assert!(
            stack.len() < stack.capacity(),
            "free stack would reallocate"
        );
        stack.push(slot);
    }

    /// Flat index of `(layer, slot)` into the lease-flag array.
    fn flat_index(&self, layer: u32, slot: u32) -> usize {
        layer as usize * self.slots_per_layer as usize + slot as usize
    }

    /// Lock the bookkeeping, recovering from a poisoned mutex.
    ///
    /// A panic elsewhere while the lock was held cannot leave the free stacks
    /// inconsistent — every mutation under the lock is a single push or pop —
    /// so refusing to hand out slots for the rest of the process would be a
    /// worse outcome than continuing.
    fn lock(&self) -> MutexGuard<'_, PoolState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Drop for SlotPool {
    fn drop(&mut self) {
        // SAFETY: `base` came from `alloc(self.alloc_layout)` in `new` and was
        // never reallocated, so this is the same block with the identical
        // layout. Every `SlotGuard` borrows the pool, so none can outlive it
        // and no live view into the slab exists here.
        unsafe { dealloc(self.base.as_ptr(), self.alloc_layout) };
    }
}

/// Exclusive lease on one expert slot.
///
/// While this value exists it is the only handle to its slot's bytes: no
/// other `acquire` can return the same slot, and there is no by-index
/// accessor on [`SlotPool`]. Dropping it returns the slot to the free stack,
/// so it must be held for the entire time a read is in flight — for io_uring
/// that means parking it in the in-flight table, not dropping it at
/// submission.
#[derive(Debug)]
pub struct SlotGuard<'pool> {
    pool: &'pool SlotPool,
    layer: u32,
    slot: u32,
    /// Base of this slot inside the pool slab; 4096-aligned.
    ptr: NonNull<u8>,
    /// Usable bytes: the layer's blob stride.
    len: usize,
}

// SAFETY: the guard is the unique lease on a disjoint sub-range of the slab,
// so moving it to another thread cannot create aliasing. It is only not
// automatically `Send`/`Sync` because of the raw pointer; `&SlotGuard` grants
// shared immutable reads and `&mut SlotGuard` is required for every write.
unsafe impl Send for SlotGuard<'_> {}
// SAFETY: see the `Send` impl.
unsafe impl Sync for SlotGuard<'_> {}

impl SlotGuard<'_> {
    /// Layer this slot belongs to.
    pub fn layer(&self) -> u32 {
        self.layer
    }

    /// Index of this slot within its layer.
    pub fn index(&self) -> u32 {
        self.slot
    }

    /// Usable bytes in the slot: the layer's blob stride.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Always `false`; a zero-stride layer is rejected at construction.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Read-only view of the slot, for compute after a read completes.
    ///
    /// Every byte was initialized to zero in [`SlotPool::new`], so this is
    /// defined even before the slot has been filled; it just reads zeros.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` starts `len` initialized bytes inside the pool slab,
        // which outlives `self` via `pool`. This guard is the unique lease on
        // that range, and `&self` rules out a concurrent `&mut` view of it.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Mutable view of the slot, for a synchronous read destination
    /// (`read_exact_at`) or for tests.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as `as_slice`, plus `&mut self` proves no other reference
        // into this range exists. Ranges of distinct guards are disjoint by
        // construction.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Write destination for an asynchronous read, e.g. an io_uring `Read`
    /// SQE.
    ///
    /// The pointer is valid for `len()` bytes for as long as this guard is
    /// alive, and it is 4096-aligned as btrfs's O_DIRECT path requires.
    ///
    /// The kernel writing through it while no Rust reference exists is sound;
    /// the caller's obligation is only to keep the guard alive until the
    /// completion has been reaped, and not to hand the same pointer to two
    /// concurrent reads (which would require duplicating it by hand — the
    /// pool will not do it).
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Read-only raw pointer to the slot base, 4096-aligned.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr().cast_const()
    }
}

impl Drop for SlotGuard<'_> {
    fn drop(&mut self) {
        self.pool.release(self.layer, self.slot);
    }
}

/// Make every page of `base[..len]` resident and its bytes initialized.
///
/// This is the single most important thing in this file. btrfs issues
/// O_DIRECT reads with page faults disabled (`fs/btrfs/direct-io.c:1107`):
/// when it cannot fault the destination it does not fail, it silently
/// completes the transfer through the *buffered* path — the read returns the
/// full byte count with no error, and the page cache grows inside our 3 GB
/// cgroup, which is the one thing the whole design exists to avoid. Freshly
/// allocated pages are untouched, so without this the very first read into
/// every slot takes that path.
///
/// Two passes, for two different reasons:
///
/// 1. `write_bytes` zeroes the slab, which makes every byte an initialized
///    `u8` so [`SlotGuard::as_slice`] is defined before the first fill.
/// 2. One `write_volatile` per 4096 bytes. A volatile store cannot be elided
///    or sunk by the optimizer, so the write fault that makes each page a
///    private resident anon page provably happens here rather than inside a
///    fault-disabled O_DIRECT read. The step is [`SLOT_ALIGN`] rather than
///    the running page size because 4096 divides every Linux page size, so
///    stepping by it touches every page on 16K- and 64K-page hosts too.
///
/// The kernel serves *read* faults from the shared zero page but always
/// allocates a private page on a *write* fault, which is why both passes
/// write.
///
/// # Safety
///
/// `base` must point to `len` writable bytes that the caller owns
/// exclusively.
unsafe fn prefault(base: NonNull<u8>, len: usize) {
    let ptr = base.as_ptr();
    // SAFETY: the caller guarantees `len` writable, exclusively owned bytes
    // at `ptr`.
    unsafe { ptr.write_bytes(0, len) };
    let mut offset = 0;
    while offset < len {
        // SAFETY: `offset < len`, so `ptr.add(offset)` is a writable byte in
        // the same allocation.
        unsafe { ptr.add(offset).write_volatile(0) };
        offset += SLOT_ALIGN;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Arc;

    /// The two real Qwen3-30B-A3B per-layer blob strides.
    const REAL_STRIDES: [u64; 2] = [3_059_712, 2_654_208];

    /// Small page-aligned geometry: three layers, two distinct strides.
    fn small_pool(slots_per_layer: u32) -> SlotPool {
        SlotPool::new(slots_per_layer, &[8192, 4096, 8192]).unwrap()
    }

    /// Acquire every slot of every layer at once.
    fn acquire_all(pool: &SlotPool) -> Vec<SlotGuard<'_>> {
        let mut held = Vec::new();
        for layer in 0..pool.n_layers() {
            for _ in 0..pool.slots_per_layer() {
                held.push(pool.acquire(layer).unwrap());
            }
        }
        held
    }

    #[test]
    fn pool_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SlotPool>();
        assert_send_sync::<SlotGuard<'_>>();
        assert_send_sync::<SlotError>();
    }

    #[test]
    fn every_slot_base_is_page_aligned() {
        // Includes a deliberately unaligned stride: alignment must survive it.
        let pool = SlotPool::new(3, &[8192, 4096, 100, 3_059_712]).unwrap();
        let held = acquire_all(&pool);
        assert_eq!(held.len(), 12);
        for guard in &held {
            assert_eq!(
                guard.as_ptr() as usize % SLOT_ALIGN,
                0,
                "layer {} slot {} base {:p} is not {SLOT_ALIGN}-aligned",
                guard.layer(),
                guard.index(),
                guard.as_ptr()
            );
        }
    }

    #[test]
    fn strides_and_sizing_match_the_real_layout() {
        // 48 layers, 24 of each real stride, but one slot each so the test
        // does not allocate the 1569 MiB the target configuration would.
        let strides: Vec<u64> = (0..48).map(|i| REAL_STRIDES[i % 2]).collect();
        for &stride in &REAL_STRIDES {
            assert_eq!(stride % SLOT_ALIGN as u64, 0, "real stride is page-sized");
        }
        let pool = SlotPool::new(1, &strides).unwrap();
        assert_eq!(pool.n_layers(), 48);
        assert_eq!(pool.slots_per_layer(), 1);
        assert_eq!(pool.max_stride(), 3_059_712);
        for layer in 0..pool.n_layers() {
            let expected = REAL_STRIDES[layer as usize % 2] as usize;
            assert_eq!(pool.stride(layer), expected, "layer {layer} stride");
            // Already page-aligned, so no padding.
            assert_eq!(pool.pitch(layer), expected, "layer {layer} pitch");
            assert_eq!(pool.acquire(layer).unwrap().len(), expected);
        }
        let expected_total: usize = strides.iter().map(|&s| s as usize).sum();
        assert_eq!(pool.total_bytes(), expected_total);

        // The target configuration's accounting, checked without allocating
        // it: 24 layers of each stride, 12 slots each.
        let target_bytes = 24 * 12 * (3_059_712usize + 2_654_208);
        assert_eq!(target_bytes, 1_645_608_960);
        assert_eq!(
            target_bytes / (1024 * 1024),
            1569,
            "target pool is 1569 MiB"
        );
    }

    #[test]
    fn unaligned_stride_is_padded_up_to_a_page_pitch() {
        let pool = SlotPool::new(2, &[100, 4097]).unwrap();
        assert_eq!(pool.stride(0), 100);
        assert_eq!(pool.pitch(0), 4096);
        assert_eq!(pool.stride(1), 4097);
        assert_eq!(pool.pitch(1), 8192);
        assert_eq!(pool.total_bytes(), 2 * 4096 + 2 * 8192);

        let a = pool.acquire(1).unwrap();
        let b = pool.acquire(1).unwrap();
        assert_eq!(a.len(), 4097);
        let gap = (b.as_ptr() as usize).abs_diff(a.as_ptr() as usize);
        assert_eq!(gap, 8192, "consecutive slots are a pitch apart");
        assert_eq!(b.as_ptr() as usize % SLOT_ALIGN, 0);
    }

    #[test]
    fn slots_are_zeroed_and_writable_over_their_whole_length() {
        let pool = SlotPool::new(1, &[3_059_712]).unwrap();
        let mut guard = pool.acquire(0).unwrap();
        assert!(
            guard.as_slice().iter().all(|&b| b == 0),
            "slab is zero-initialized at construction"
        );
        // Touch the first and last byte of every page, then read back.
        let len = guard.len();
        let bytes = guard.as_mut_slice();
        let mut offset = 0;
        while offset < len {
            bytes[offset] = 0xa5;
            let end = (offset + SLOT_ALIGN - 1).min(len - 1);
            bytes[end] = 0x5a;
            offset += SLOT_ALIGN;
        }
        let bytes = guard.as_slice();
        let mut offset = 0;
        while offset < len {
            assert_eq!(bytes[offset], 0xa5, "page at {offset} lost its write");
            offset += SLOT_ALIGN;
        }
    }

    /// Count resident pages of `[base, base+len)` via `/proc/self/pagemap`.
    ///
    /// Bit 63 of each 64-bit entry is "page present". Unprivileged readers
    /// have the PFN masked out since Linux 4.2 but keep the flags, so this
    /// works without capabilities. Returns `None` if pagemap is unreadable.
    #[cfg(target_os = "linux")]
    fn resident_pages(base: *const u8, len: usize) -> Option<usize> {
        use std::io::{Read, Seek, SeekFrom};

        let pages = len / SLOT_ALIGN;
        let mut file = std::fs::File::open("/proc/self/pagemap").ok()?;
        let index = base as u64 / SLOT_ALIGN as u64;
        file.seek(SeekFrom::Start(index * 8)).ok()?;
        let mut buf = vec![0u8; pages * 8];
        file.read_exact(&mut buf).ok()?;
        Some(
            buf.chunks_exact(8)
                .filter(|e| u64::from_le_bytes((*e).try_into().unwrap()) >> 63 & 1 == 1)
                .count(),
        )
    }

    /// The load-bearing test: prove the slab is resident after `new`, and
    /// prove the measurement can tell, by checking an equally sized fresh
    /// allocation that was *not* pre-faulted.
    #[test]
    #[cfg(target_os = "linux")]
    fn construction_faults_in_every_page() {
        const SLAB: usize = 512 * 1024;

        // Control first, before any pool allocation churns the allocator:
        // a same-sized, same-aligned block that nobody touched.
        let control_layout = Layout::from_size_align(SLAB, SLOT_ALIGN).unwrap();
        // SAFETY: non-zero size, valid layout.
        let control = unsafe { alloc(control_layout) };
        assert!(!control.is_null());
        let control_resident = resident_pages(control, SLAB);
        // SAFETY: same pointer and layout as the allocation above.
        unsafe { dealloc(control, control_layout) };

        // 4 layers * 8 slots * 16 KiB == SLAB.
        let pool = SlotPool::new(8, &[16384; 4]).unwrap();
        assert_eq!(pool.total_bytes(), SLAB);
        let first = pool.acquire(0).unwrap();
        let pool_resident = resident_pages(first.as_ptr(), SLAB);
        drop(first);

        let (Some(control_resident), Some(pool_resident)) = (control_resident, pool_resident)
        else {
            // pagemap is unreadable (hardened kernel, unusual container).
            // The write/read-back coverage in the sibling test still stands.
            eprintln!("skipping residency assertions: /proc/self/pagemap unreadable");
            return;
        };

        let pages = SLAB / SLOT_ALIGN;
        assert_eq!(
            pool_resident, pages,
            "every page of the slab must be resident after new(); \
             an untouched page makes btrfs silently fall back to buffered I/O"
        );
        assert!(
            control_resident * 2 < pages,
            "residency check is not meaningful: an untouched {SLAB}-byte block \
             already reported {control_resident}/{pages} pages resident"
        );
    }

    #[test]
    fn simultaneously_held_slots_never_overlap() {
        let pool = SlotPool::new(4, &[8192, 4096, 12288]).unwrap();
        let held = acquire_all(&pool);
        assert_eq!(held.len(), 12);
        let ranges: Vec<(usize, usize)> = held
            .iter()
            .map(|g| {
                let start = g.as_ptr() as usize;
                (start, start + g.len())
            })
            .collect();
        for (i, a) in ranges.iter().enumerate() {
            for (j, b) in ranges.iter().enumerate().skip(i + 1) {
                assert!(
                    a.1 <= b.0 || b.1 <= a.0,
                    "slots {i} {a:?} and {j} {b:?} overlap"
                );
            }
        }
        // And every range lies inside the slab.
        let base = ranges.iter().map(|r| r.0).min().unwrap();
        let end = ranges.iter().map(|r| r.1).max().unwrap();
        assert!(end - base <= pool.total_bytes());
    }

    #[test]
    fn writes_through_one_slot_do_not_disturb_another() {
        let pool = small_pool(2);
        let mut held = acquire_all(&pool);
        for (i, guard) in held.iter_mut().enumerate() {
            guard.as_mut_slice().fill(i as u8 + 1);
        }
        for (i, guard) in held.iter().enumerate() {
            assert!(
                guard.as_slice().iter().all(|&b| b == i as u8 + 1),
                "slot {i} was clobbered by a neighbour"
            );
        }
    }

    #[test]
    fn exhaustion_is_a_typed_error_and_a_release_makes_a_slot_available() {
        let pool = small_pool(2);
        let a = pool.acquire(1).unwrap();
        let b = pool.acquire(1).unwrap();
        assert_ne!(a.index(), b.index(), "the same slot was leased twice");
        assert_eq!(pool.free_slots(1), Some(0));
        assert_eq!(
            pool.acquire(1).unwrap_err(),
            SlotError::Exhausted {
                layer: 1,
                slots_per_layer: 2
            }
        );
        // Other layers are unaffected.
        assert_eq!(pool.free_slots(0), Some(2));
        pool.acquire(0).unwrap();

        let freed = a.index();
        drop(a);
        assert_eq!(pool.free_slots(1), Some(1));
        let c = pool.acquire(1).unwrap();
        assert_eq!(c.index(), freed, "the released slot came back");
        drop((b, c));
        assert_eq!(pool.free_slots(1), Some(2));
    }

    #[test]
    fn acquire_rejects_an_out_of_range_layer() {
        let pool = small_pool(1);
        assert_eq!(
            pool.acquire(3).unwrap_err(),
            SlotError::LayerOutOfRange {
                layer: 3,
                n_layers: 3
            }
        );
        assert_eq!(pool.acquire(u32::MAX).unwrap_err(), {
            SlotError::LayerOutOfRange {
                layer: u32::MAX,
                n_layers: 3,
            }
        });
        assert_eq!(pool.free_slots(3), None);
    }

    #[test]
    fn rejects_degenerate_geometry() {
        assert_eq!(
            SlotPool::new(0, &[4096]).unwrap_err(),
            SlotError::ZeroSlotsPerLayer
        );
        assert_eq!(SlotPool::new(4, &[]).unwrap_err(), SlotError::NoLayers);
        assert_eq!(
            SlotPool::new(4, &[4096, 0, 4096]).unwrap_err(),
            SlotError::ZeroStride { layer: 1 }
        );
    }

    #[test]
    fn rejects_oversized_and_overflowing_geometry() {
        // Over the cap but nowhere near overflowing.
        let err = SlotPool::new(12, &[u64::from(u32::MAX); 48]).unwrap_err();
        assert!(
            matches!(err, SlotError::PoolTooLarge { .. }),
            "expected PoolTooLarge, got {err:?}"
        );
        // Would overflow a u64 product: 2^32 slots of ~2^63 bytes each.
        let err = SlotPool::new(u32::MAX, &[u64::MAX / 2; 4]).unwrap_err();
        assert!(
            matches!(err, SlotError::PoolTooLarge { .. }),
            "expected PoolTooLarge, got {err:?}"
        );
        // Rounding the stride up to a page is itself checked for overflow.
        let err = SlotPool::new(1, &[u64::MAX]).unwrap_err();
        assert!(
            matches!(err, SlotError::PoolTooLarge { .. }),
            "expected PoolTooLarge, got {err:?}"
        );
        // Right at the cap in one layer, still rejected rather than attempted.
        let err = SlotPool::new(1, &[MAX_POOL_BYTES + SLOT_ALIGN as u64]).unwrap_err();
        assert!(
            matches!(err, SlotError::PoolTooLarge { requested, limit }
                if requested > limit),
            "expected PoolTooLarge, got {err:?}"
        );
    }

    #[test]
    fn steady_state_acquire_and_release_do_not_allocate() {
        let pool = small_pool(3);
        let capacities: Vec<usize> = {
            let state = pool.lock();
            state.free.iter().map(Vec::capacity).collect()
        };
        for _ in 0..64 {
            let held = acquire_all(&pool);
            drop(held);
        }
        let after: Vec<usize> = {
            let state = pool.lock();
            state.free.iter().map(Vec::capacity).collect()
        };
        assert_eq!(
            capacities, after,
            "free stacks reallocated during steady state"
        );
        assert!(capacities.iter().all(|&c| c == 3));
    }

    /// Requirement 3, directly: hammer the pool from many threads and assert
    /// that no base address is ever leased twice at the same time. Modulo
    /// indexing over a queue depth would fail this; the free list does not.
    #[test]
    fn concurrent_acquire_never_hands_out_a_live_slot_twice() {
        let pool = Arc::new(SlotPool::new(2, &[8192, 4096, 8192]).unwrap());
        let live: Arc<Mutex<HashSet<usize>>> = Arc::new(Mutex::new(HashSet::new()));
        let mut handles = Vec::new();
        for t in 0..8u32 {
            let pool = Arc::clone(&pool);
            let live = Arc::clone(&live);
            handles.push(std::thread::spawn(move || {
                for i in 0..500u32 {
                    let layer = (t + i) % pool.n_layers();
                    let Ok(mut guard) = pool.acquire(layer) else {
                        continue; // legitimate exhaustion
                    };
                    let addr = guard.as_ptr() as usize;
                    assert!(
                        live.lock().unwrap().insert(addr),
                        "address {addr:#x} was leased twice concurrently"
                    );
                    // Write through the lease so a real aliasing bug would
                    // also show up as data corruption under a sanitizer.
                    guard.as_mut_slice()[0] = t as u8;
                    assert!(live.lock().unwrap().remove(&addr));
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
        assert!(live.lock().unwrap().is_empty());
        for layer in 0..pool.n_layers() {
            assert_eq!(
                pool.free_slots(layer),
                Some(2),
                "layer {layer} leaked slots"
            );
        }
    }

    #[test]
    fn error_messages_name_the_problem() {
        assert_eq!(
            SlotError::Exhausted {
                layer: 7,
                slots_per_layer: 12
            }
            .to_string(),
            "layer 7: all 12 slots are in use"
        );
        assert_eq!(
            SlotError::ZeroStride { layer: 3 }.to_string(),
            "layer 3: blob stride is zero"
        );
    }
}
