//! Installer tests against a synthetic in-crate GGUF fixture: two MoE
//! layers, four experts, qwen3moe metadata injected post-parse (the same
//! pattern the planner tests use). Every install here is tiny and local;
//! the real 17 GiB remote pin is never touched.

use std::fs;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use ramvamp_core::format::{self, COMMON_FILE, ProjectionName};

use super::executor::{WindowGeometry, for_each_overlap};
use super::lock::{InstallLock, lock_path};
use super::state::{load_state, write_state};
use super::*;
use crate::gguf::testutil::{FixtureBuilder, TensorSpec};
use crate::gguf::{GgmlType, GgufFile, MetaValue};
use crate::plan::RepackPlan;

/// Unique directory under the system temp dir, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!("ramvamp-install-{tag}-{pid}-{n}-{nanos}"));
        fs::create_dir_all(&path).expect("create test temp dir");
        TempDir(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Two MoE layers, four experts; layer 0's down projection is Q6_K.
/// Large f16 tensors (64 KiB each) guarantee ops far larger than the
/// 4 KiB test window, exercising span-by-span multi-window writes.
fn fixture_bytes() -> Vec<u8> {
    FixtureBuilder {
        tensors: vec![
            TensorSpec::new("token_embd.weight", &[64, 512], GgmlType::F16),
            TensorSpec::new("blk.0.attn_k_norm.weight", &[8], GgmlType::F32),
            TensorSpec::new("blk.0.attn_norm.weight", &[64], GgmlType::F32),
            TensorSpec::new("blk.0.ffn_gate_exps.weight", &[256, 8, 4], GgmlType::Q4_K),
            TensorSpec::new("blk.0.ffn_up_exps.weight", &[256, 8, 4], GgmlType::Q4_K),
            TensorSpec::new("blk.0.ffn_down_exps.weight", &[256, 8, 4], GgmlType::Q6_K),
            TensorSpec::new("blk.1.ffn_gate_exps.weight", &[256, 8, 4], GgmlType::Q4_K),
            TensorSpec::new("blk.1.ffn_up_exps.weight", &[256, 8, 4], GgmlType::Q4_K),
            TensorSpec::new("blk.1.ffn_down_exps.weight", &[256, 8, 4], GgmlType::Q4_K),
            TensorSpec::new("output_norm.weight", &[64], GgmlType::F32),
            TensorSpec::new("output.weight", &[64, 512], GgmlType::F16),
        ],
        ..FixtureBuilder::default()
    }
    .build()
}

/// The qwen3moe.* metadata the planner requires, injected post-parse
/// (same technique as the planner's own tests).
fn inject_meta(gguf: &mut GgufFile) {
    let m = &mut gguf.metadata;
    m.insert(
        "general.architecture".to_owned(),
        MetaValue::String("qwen3moe".to_owned()),
    );
    m.insert("qwen3moe.block_count".to_owned(), MetaValue::U32(2));
    m.insert("qwen3moe.embedding_length".to_owned(), MetaValue::U32(64));
    m.insert("qwen3moe.expert_count".to_owned(), MetaValue::U32(4));
    m.insert("qwen3moe.expert_used_count".to_owned(), MetaValue::U32(2));
    m.insert(
        "qwen3moe.expert_feed_forward_length".to_owned(),
        MetaValue::U32(8),
    );
    m.insert(
        "qwen3moe.attention.head_count".to_owned(),
        MetaValue::U32(8),
    );
    m.insert(
        "qwen3moe.attention.head_count_kv".to_owned(),
        MetaValue::U32(2),
    );
    m.insert(
        "qwen3moe.attention.key_length".to_owned(),
        MetaValue::U32(8),
    );
    m.insert("qwen3moe.rope.freq_base".to_owned(), MetaValue::F32(1e7));
    m.insert(
        "qwen3moe.attention.layer_norm_rms_epsilon".to_owned(),
        MetaValue::F32(1e-6),
    );
    m.insert("qwen3moe.context_length".to_owned(), MetaValue::U64(4096));
}

fn plan_for(bytes: &[u8]) -> (GgufFile, RepackPlan) {
    let slice: &[u8] = bytes;
    let mut gguf = GgufFile::parse(&slice).expect("fixture parses");
    inject_meta(&mut gguf);
    let plan = RepackPlan::from_gguf(&gguf).expect("fixture plans");
    (gguf, plan)
}

fn pin(tag: &str) -> SourcePin {
    SourcePin {
        url: tag.to_owned(),
        hf_repo: "test/repo".to_owned(),
        revision: "deadbeef".to_owned(),
        file: "tiny-Q4_K_M.gguf".to_owned(),
    }
}

fn opts(window_bytes: u64, fsync_every: u64) -> InstallOptions {
    InstallOptions {
        window_bytes,
        fsync_every_windows: fsync_every,
        ..InstallOptions::default()
    }
}

fn proj_tensor_name(layer: usize, name: ProjectionName) -> String {
    let stem = match name {
        ProjectionName::Gate => "gate",
        ProjectionName::Up => "up",
        ProjectionName::Down => "down",
    };
    format!("blk.{layer}.ffn_{stem}_exps.weight")
}

#[test]
fn full_install_is_byte_identical_and_verifies() {
    let tmp = TempDir::new("full");
    let bytes = fixture_bytes();
    let gguf_path = tmp.path().join("tiny.gguf");
    fs::write(&gguf_path, &bytes).unwrap();
    let local = crate::source::LocalFile::open(&gguf_path).unwrap();
    let mut gguf = GgufFile::parse(&local).unwrap();
    inject_meta(&mut gguf);
    let plan = RepackPlan::from_gguf(&gguf).unwrap();

    let final_dir = tmp.path().join("model.rvmp");
    let report = install(
        &local,
        &plan,
        &pin("file://tiny"),
        &final_dir,
        &opts(4096, 4),
    )
    .unwrap();
    assert!(report.verified);
    assert_eq!(report.windows_resumed, 0);
    assert!(
        report.windows_total > 4,
        "test wants a multi-window install"
    );
    assert_eq!(report.bytes_copied, plan.totals.download_bytes);
    assert!(format::is_complete(&final_dir));
    assert!(!format::partial_dir(&final_dir).exists());
    assert!(!final_dir.join(STATE_FILE).exists());

    let manifest = format::load_manifest(&final_dir).unwrap();
    manifest.validate().unwrap();
    assert_eq!(manifest.arch, plan.arch);
    assert_eq!(manifest.quant, plan.quant);
    assert_eq!(manifest.common_tensors, plan.common_tensors);
    assert_eq!(manifest.source.hf_repo, "test/repo");
    assert_eq!(manifest.source.sha256, report.source_sha256);
    assert_eq!(manifest.model_id, "tiny-q4_k_m");
    format::verify_files(&final_dir, &manifest).unwrap();
    let layout = format::load_layout(&final_dir).unwrap();
    layout.validate_against(&manifest).unwrap();
    assert_eq!(layout, plan.layout);

    // Gate 1 of the validation protocol: every repacked expert slab is
    // byte-identical to its GGUF slice.
    for (layer_idx, layer) in plan.layout.layers.iter().enumerate() {
        let file_bytes = fs::read(final_dir.join(&layer.file)).unwrap();
        for proj in &layer.projections {
            let tensor = gguf
                .tensor(&proj_tensor_name(layer_idx, proj.name))
                .unwrap();
            for expert in 0..u64::from(layer.n_experts) {
                let src = gguf.expert_slab(tensor, expert).unwrap();
                let dst_start = (expert * layer.stride + proj.offset_in_blob) as usize;
                assert_eq!(
                    &file_bytes[dst_start..dst_start + proj.len as usize],
                    &bytes[src.start as usize..src.end as usize],
                    "layer {layer_idx} expert {expert} {:?} differs",
                    proj.name
                );
            }
        }
    }

    // Common tensors are byte-identical too.
    let common = fs::read(final_dir.join(COMMON_FILE)).unwrap();
    for (name, tensor_entry) in &plan.common_tensors {
        let tensor = gguf.tensor(name).unwrap();
        let src_start = gguf.data_offset(tensor) as usize;
        assert_eq!(
            &common
                [tensor_entry.offset as usize..(tensor_entry.offset + tensor_entry.len) as usize],
            &bytes[src_start..src_start + tensor_entry.len as usize],
            "common tensor {name} differs"
        );
    }
}

#[test]
fn interrupted_install_resumes_to_identical_result() {
    let tmp = TempDir::new("resume");
    let bytes = fixture_bytes();
    let slice: &[u8] = &bytes;
    let (_, plan) = plan_for(&bytes);

    // Reference: uninterrupted install.
    let dir_a = tmp.path().join("a.rvmp");
    let report_a = install(&slice, &plan, &pin("mem://tiny"), &dir_a, &opts(4096, 1)).unwrap();

    // Interrupted install: abort after 3 windows (fsync every window, so
    // exactly 3 are durable), then resume to completion.
    let dir_b = tmp.path().join("b.rvmp");
    let mut aborting = opts(4096, 1);
    aborting.fail_after_windows = Some(3);
    let err = install(&slice, &plan, &pin("mem://tiny"), &dir_b, &aborting).unwrap_err();
    assert!(matches!(err, InstallError::TestAbort { windows: 3 }));
    let partial = format::partial_dir(&dir_b);
    assert!(partial.exists());
    let st = load_state(&partial).unwrap();
    assert_eq!(st.durable_windows, 3);
    assert_eq!(st.window_digests.len(), 3);

    let mut resuming = opts(4096, 1);
    resuming.resume = true;
    let report_b = install(&slice, &plan, &pin("mem://tiny"), &dir_b, &resuming).unwrap();
    assert_eq!(report_b.windows_resumed, 3);
    assert!(report_b.verified);
    assert!(!partial.exists());

    // Same bytes, same hashes, same digest-of-digests either way.
    let manifest_a = format::load_manifest(&dir_a).unwrap();
    let manifest_b = format::load_manifest(&dir_b).unwrap();
    assert_eq!(manifest_a.files, manifest_b.files);
    assert_eq!(manifest_a.source.sha256, manifest_b.source.sha256);
    assert_eq!(report_a.source_sha256, report_b.source_sha256);
}

#[test]
fn tampered_destination_is_detected_on_resume() {
    let tmp = TempDir::new("tamper");
    let bytes = fixture_bytes();
    let slice: &[u8] = &bytes;
    let (_, plan) = plan_for(&bytes);

    let final_dir = tmp.path().join("model.rvmp");
    let mut aborting = opts(4096, 1);
    aborting.fail_after_windows = Some(3);
    let err = install(&slice, &plan, &pin("mem://tiny"), &final_dir, &aborting).unwrap_err();
    assert!(matches!(err, InstallError::TestAbort { .. }));

    let partial = format::partial_dir(&final_dir);
    let st = load_state(&partial).unwrap();
    assert_eq!(st.durable_windows, 3);

    // Flip one destination byte in every durable window, so whichever
    // window the resume spot-check picks, it must notice.
    let mut ops = plan.copy_ops.clone();
    ops.sort_by_key(|op| op.src.start);
    let geom = WindowGeometry::from_ops(&ops, st.window_bytes).unwrap();
    for window in 0..st.durable_windows {
        let range = geom.range(window);
        let mut tampered = false;
        for_each_overlap(&ops, &range, |op, overlap_start, _| {
            if !tampered {
                let path = partial.join(&op.dst_file);
                let offset = op.dst_offset + (overlap_start - op.src.start);
                let file = fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&path)
                    .unwrap();
                let mut byte = [0u8; 1];
                file.read_exact_at(&mut byte, offset).unwrap();
                byte[0] ^= 0xff;
                file.write_all_at(&byte, offset).unwrap();
                file.sync_all().unwrap();
                tampered = true;
            }
            Ok(())
        })
        .unwrap();
        assert!(tampered, "window {window} had no copied bytes to tamper");
    }

    let mut resuming = opts(4096, 1);
    resuming.resume = true;
    let err = install(&slice, &plan, &pin("mem://tiny"), &final_dir, &resuming).unwrap_err();
    assert!(
        matches!(err, InstallError::WindowDigestMismatch { .. }),
        "expected WindowDigestMismatch, got {err:?}"
    );
}

#[test]
fn overwrite_and_partial_guard_semantics() {
    let tmp = TempDir::new("overwrite");
    let bytes = fixture_bytes();
    let slice: &[u8] = &bytes;
    let (_, plan) = plan_for(&bytes);
    let source_pin = pin("mem://tiny");

    // Existing complete install: refused without --overwrite, replaced
    // with it.
    let final_dir = tmp.path().join("model.rvmp");
    install(&slice, &plan, &source_pin, &final_dir, &opts(4096, 4)).unwrap();
    let err = install(&slice, &plan, &source_pin, &final_dir, &opts(4096, 4)).unwrap_err();
    assert!(matches!(err, InstallError::TargetExists(_)));
    let mut overwriting = opts(4096, 4);
    overwriting.overwrite = true;
    install(&slice, &plan, &source_pin, &final_dir, &overwriting).unwrap();
    assert!(format::is_complete(&final_dir));

    // Existing partial: refused without --resume/--overwrite, replaced
    // with --overwrite.
    let dir2 = tmp.path().join("model2.rvmp");
    let partial2 = format::partial_dir(&dir2);
    fs::create_dir_all(&partial2).unwrap();
    fs::write(partial2.join("junk"), b"leftover").unwrap();
    let err = install(&slice, &plan, &source_pin, &dir2, &opts(4096, 4)).unwrap_err();
    assert!(matches!(err, InstallError::PartialExists(_)));
    let mut overwriting2 = opts(4096, 4);
    overwriting2.overwrite = true;
    install(&slice, &plan, &source_pin, &dir2, &overwriting2).unwrap();
    assert!(format::is_complete(&dir2));

    // A target that is not an install and not empty is never deleted.
    let dir3 = tmp.path().join("precious");
    fs::create_dir_all(&dir3).unwrap();
    fs::write(dir3.join("do-not-delete.txt"), b"user data").unwrap();
    let mut overwriting3 = opts(4096, 4);
    overwriting3.overwrite = true;
    let err = install(&slice, &plan, &source_pin, &dir3, &overwriting3).unwrap_err();
    assert!(matches!(err, InstallError::OverwriteRefused(_)));
    assert!(dir3.join("do-not-delete.txt").exists());
}

#[test]
fn resume_with_nothing_to_resume_starts_fresh() {
    let tmp = TempDir::new("resume-fresh");
    let bytes = fixture_bytes();
    let slice: &[u8] = &bytes;
    let (_, plan) = plan_for(&bytes);
    let final_dir = tmp.path().join("model.rvmp");
    let mut resuming = opts(4096, 4);
    resuming.resume = true;
    let report = install(&slice, &plan, &pin("mem://tiny"), &final_dir, &resuming).unwrap();
    assert_eq!(report.windows_resumed, 0);
    assert!(format::is_complete(&final_dir));
}

#[test]
fn lock_blocks_concurrent_install_and_stale_locks_break() {
    let tmp = TempDir::new("lock");
    let bytes = fixture_bytes();
    let slice: &[u8] = &bytes;
    let (_, plan) = plan_for(&bytes);
    let final_dir = tmp.path().join("model.rvmp");

    // A thread holds the lock; install must fail with Locked.
    let (locked_tx, locked_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let lock_dir = final_dir.clone();
    let holder = thread::spawn(move || {
        let lock = InstallLock::acquire(&lock_dir).unwrap();
        locked_tx.send(()).unwrap();
        release_rx.recv().unwrap();
        drop(lock);
    });
    locked_rx.recv().unwrap();
    let err = install(
        &slice,
        &plan,
        &pin("mem://tiny"),
        &final_dir,
        &opts(4096, 4),
    )
    .unwrap_err();
    match err {
        InstallError::Locked { pid, .. } => assert_eq!(pid, std::process::id()),
        other => panic!("expected Locked, got {other:?}"),
    }
    release_tx.send(()).unwrap();
    holder.join().unwrap();

    // Released: install proceeds.
    install(
        &slice,
        &plan,
        &pin("mem://tiny"),
        &final_dir,
        &opts(4096, 4),
    )
    .unwrap();

    // A lock whose pid no longer exists is stale and gets broken.
    let dir2 = tmp.path().join("model2.rvmp");
    fs::write(lock_path(&dir2), b"4294900000").unwrap();
    install(&slice, &plan, &pin("mem://tiny"), &dir2, &opts(4096, 4)).unwrap();
    assert!(format::is_complete(&dir2));
}

#[test]
fn resume_rejects_identity_mismatches() {
    let tmp = TempDir::new("mismatch");
    let bytes = fixture_bytes();
    let slice: &[u8] = &bytes;
    let (_, plan) = plan_for(&bytes);
    let final_dir = tmp.path().join("model.rvmp");

    let mut aborting = opts(4096, 1);
    aborting.fail_after_windows = Some(2);
    install(&slice, &plan, &pin("mem://tiny"), &final_dir, &aborting).unwrap_err();
    let partial = format::partial_dir(&final_dir);

    // Different source URL.
    let mut resuming = opts(4096, 1);
    resuming.resume = true;
    let err = install(&slice, &plan, &pin("mem://other"), &final_dir, &resuming).unwrap_err();
    assert!(matches!(err, InstallError::StateMismatch(_)));

    // Corrupted plan fingerprint.
    let mut st = load_state(&partial).unwrap();
    st.plan_fingerprint = "0".repeat(64);
    write_state(&partial, &st).unwrap();
    let err = install(&slice, &plan, &pin("mem://tiny"), &final_dir, &resuming).unwrap_err();
    assert!(matches!(err, InstallError::StateMismatch(_)));
}

#[test]
fn discard_partial_removes_only_the_partial() {
    let tmp = TempDir::new("discard");
    let bytes = fixture_bytes();
    let slice: &[u8] = &bytes;
    let (_, plan) = plan_for(&bytes);
    let final_dir = tmp.path().join("model.rvmp");

    let mut aborting = opts(4096, 1);
    aborting.fail_after_windows = Some(2);
    install(&slice, &plan, &pin("mem://tiny"), &final_dir, &aborting).unwrap_err();
    let partial = format::partial_dir(&final_dir);
    assert!(partial.exists());

    let removed = discard_partial(&final_dir).unwrap();
    assert_eq!(removed, partial);
    assert!(!partial.exists());

    let err = discard_partial(&final_dir).unwrap_err();
    assert!(matches!(err, InstallError::NothingToDiscard(_)));

    // Discard never touches a completed install.
    install(
        &slice,
        &plan,
        &pin("mem://tiny"),
        &final_dir,
        &opts(4096, 4),
    )
    .unwrap();
    let err = discard_partial(&final_dir).unwrap_err();
    assert!(matches!(err, InstallError::NothingToDiscard(_)));
    assert!(format::is_complete(&final_dir));
}

#[test]
fn window_geometry_math() {
    let ops = vec![
        crate::plan::CopyOp {
            src: 100..150,
            dst_file: "common.bin".to_owned(),
            dst_offset: 0,
        },
        crate::plan::CopyOp {
            src: 200..300,
            dst_file: "common.bin".to_owned(),
            dst_offset: 64,
        },
    ];
    let geom = WindowGeometry::from_ops(&ops, 64).unwrap();
    assert_eq!(geom.region_start, 100);
    assert_eq!(geom.region_end, 300);
    // 200 bytes / 64 = 3.125 -> 4 windows, last one short.
    assert_eq!(geom.n_windows(), 4);
    assert_eq!(geom.range(0), 100..164);
    assert_eq!(geom.range(3), 292..300);

    // Overlap visiting: window 1 (164..228) sees the tail gap of op 0?
    // No: op 0 ends at 150. It sees only op 1's head.
    let mut seen = Vec::new();
    for_each_overlap(&ops, &geom.range(1), |op, s, e| {
        seen.push((op.dst_offset, s, e));
        Ok(())
    })
    .unwrap();
    assert_eq!(seen, vec![(64, 200, 228)]);
}
