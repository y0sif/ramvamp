//! Arm B: the same batched Q4_K x Q8_K GEMV as a Vulkan compute dispatch.
//!
//! The heap on this device is unified (DEVICE_LOCAL and HOST_VISIBLE on the
//! same memory type), so every buffer is mapped once and written in place:
//! there is no staging buffer and no copy command.
//!
//! Two timed regions are offered, and they bracket the truth rather than
//! picking a flattering point inside it. [`GpuGemv::run`] times submit to
//! fence with the weights already resident, which is the best case a backend
//! ever sees. [`GpuGemv::run_streamed`] puts the whole weight slab back
//! inside the timed region, which is what a backend actually pays, since a
//! prefill chunk dispatches a different expert on every submit.

use std::io::Cursor;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use ash::vk;
use ramvamp_core::kernels::quants::{BlockQ8K, QuantFormat};

use crate::vk::VkContext;

/// Compiled by `scripts/build-shaders.sh`; committed next to its source so
/// a build never needs `glslc`.
const SPV: &[u8] = include_bytes!("../shaders/gemv_q4k_q8k.spv");

/// Workgroup size the shader declares.
const LOCAL_SIZE: u32 = 256;
/// Subgroup width the activation-minor packing is built for.
///
/// [`GpuActs`] lays the activations out so that the 32 lanes sharing one
/// super-block read 32 consecutive uints. That is a claim about the lane
/// count, and `VkPhysicalDeviceSubgroupProperties::subgroupSize` does not
/// establish it: it is a device *default*, while Mesa's Intel backend
/// compiles each compute pipeline at SIMD8, SIMD16 or SIMD32 from that
/// pipeline's register pressure. So [`GpuGemv::new`] pins the width with
/// `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` where the device
/// supports it, and where it does not it says so instead of claiming a
/// guarantee it never had.
pub const ASSUMED_SUBGROUP: u32 = 32;
/// The one packed weight format this shader can read.
///
/// Not merely "the format the arithmetic implements". The weight buffer is
/// declared `uint w[]` and every weight read indexes it in 4-byte units, so
/// the read pattern needs `block_bytes % 4 == 0`. Q4_K's 144 B satisfies
/// that; **Q6_K's 210 B does not**, and Q6_K is `ffn_down_exps` on 24 of 48
/// layers, so this indexing cannot be reused for half the expert FFN without
/// a byte-granular path. See [`check_shader_format`].
pub const SHADER_FORMAT: QuantFormat = QuantFormat::Q4_K;
/// Uints per packed Q4_K super-block (144 bytes).
const UINTS_PER_BLOCK: usize = 36;
// The shader's `UINTS_PER_BLOCK` and the format's packed block size are the
// same fact stated twice, and the shader indexes `uint w[]` with the first.
const _: () = assert!(UINTS_PER_BLOCK * 4 == SHADER_FORMAT.block_bytes());
/// 32-weight sub-blocks per super-block.
const SUBBLOCKS: usize = 8;
/// 16-element bsum groups per super-block.
const BSUM_GROUPS: usize = 16;
/// Caps the shader's shared arrays are sized for.
const MAX_BLOCKS: usize = 8;
/// Caps the shader's shared arrays are sized for.
const MAX_ACTS: usize = 32;

/// Reject any packed weight format the shader's `uint`-indexed reads cannot
/// address.
///
/// The gate is `block_bytes % 4 == 0`, and it is a real constraint rather
/// than a formality: the expert FFN's `down` projection is Q6_K on 24 of 48
/// layers of the v0 model (`docs/architecture.md`), and Q6_K packs 256
/// weights into 210 B. Block `b` of a Q6_K row starts at byte `210 * b`,
/// which is only 2-byte aligned on odd `b`, so neither the f16 `d` pair nor
/// the `qs` run lands on a uint boundary. Half the expert FFN therefore
/// needs a byte-granular shader, not a re-tuned version of this one, and
/// this function is where that is enforced instead of assumed.
///
/// # Errors
///
/// Any format other than [`SHADER_FORMAT`], with the reason stated.
pub fn check_shader_format(format: QuantFormat) -> Result<()> {
    if !format.block_bytes().is_multiple_of(4) {
        bail!(
            "{format:?} packs a block into {} B, which is not a whole number of uints; the \
             shader indexes its weight buffer as `uint w[]` and cannot address this format \
             at all without a byte-granular read path",
            format.block_bytes()
        );
    }
    if format != SHADER_FORMAT {
        bail!(
            "{format:?} is uint-addressable ({} B/block) but this shader implements \
             {SHADER_FORMAT:?} arithmetic only",
            format.block_bytes()
        );
    }
    Ok(())
}

/// One mapped, device-local, host-visible buffer.
///
/// Owns its allocation and its mapping: the `Drop` below frees all three
/// handles, so a failure part-way through [`GpuGemv::new`] unwinds cleanly
/// instead of leaking a buffer, its device memory and its mapping.
struct Buffer<'a> {
    ctx: &'a VkContext,
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
    ptr: *mut u8,
    size: u64,
    /// Property flags of the memory type this landed in. Recorded because
    /// they decide what a host read of the mapping costs, and the choice is
    /// "first type that matches", not a considered one.
    props: vk::MemoryPropertyFlags,
}

impl<'a> Buffer<'a> {
    fn new(ctx: &'a VkContext, size: u64) -> Result<Self> {
        let ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `ci` outlives the call; the device is live.
        let buf = unsafe { ctx.device.create_buffer(&ci, None) }.context("creating a buffer")?;
        // Every fallible step below owns the handles taken so far, so each
        // `?` runs this value's `Drop` and releases them. Null handles are
        // the "not acquired" state and `Drop` skips them.
        let mut this = Self {
            ctx,
            buf,
            mem: vk::DeviceMemory::null(),
            ptr: std::ptr::null_mut(),
            size,
            props: vk::MemoryPropertyFlags::empty(),
        };
        // SAFETY: `buf` was just created on this device.
        let req = unsafe { ctx.device.get_buffer_memory_requirements(buf) };
        let want = vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT;
        let index = ctx.memory_type(req.memory_type_bits, want).or_else(|_| {
            ctx.memory_type(
                req.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
        })?;
        this.props = ctx.memory.memory_types[index as usize].property_flags;
        let ai = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(index);
        // SAFETY: `ai` outlives the call.
        this.mem = unsafe { ctx.device.allocate_memory(&ai, None) }.context("allocating memory")?;
        // SAFETY: `this.mem` was allocated for `buf`'s requirements and
        // nothing is bound to either yet.
        unsafe { ctx.device.bind_buffer_memory(buf, this.mem, 0) }.context("binding memory")?;
        // SAFETY: the memory type is HOST_VISIBLE, and the mapping lives as
        // long as the `Buffer`.
        this.ptr = unsafe {
            ctx.device
                .map_memory(this.mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }
        .context("mapping memory")?
        .cast::<u8>();
        Ok(this)
    }

    /// Copy `src` into the mapping.
    fn write<T: Copy>(&self, src: &[T]) -> Result<()> {
        let bytes = std::mem::size_of_val(src) as u64;
        if bytes > self.size {
            bail!("writing {bytes} B into a {} B buffer", self.size);
        }
        // SAFETY: the mapping covers `self.size >= bytes`, is host-coherent,
        // and no dispatch is in flight (the caller writes between submits).
        // `copy_nonoverlapping` on a byte pointer needs no alignment beyond
        // 1, which is why the cast is to `u8`.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr().cast::<u8>(), self.ptr, bytes as usize);
        }
        Ok(())
    }

    /// Copy the mapping into `dst`.
    fn read<T: Copy>(&self, dst: &mut [T]) -> Result<()> {
        let bytes = std::mem::size_of_val(dst) as u64;
        if bytes > self.size {
            bail!("reading {bytes} B from a {} B buffer", self.size);
        }
        // SAFETY: as `write`, and the caller has waited on the fence that
        // signalled the dispatch's completion before calling.
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr, dst.as_mut_ptr().cast::<u8>(), bytes as usize);
        }
        Ok(())
    }
}

impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        let d = &self.ctx.device;
        // SAFETY: the device is drained first, so nothing in flight still
        // references these handles. Each handle is released only if it was
        // actually acquired, which is what the null checks test: a `Buffer`
        // that failed part-way through `new` holds nulls for the rest.
        unsafe {
            let _ = d.device_wait_idle();
            if !self.ptr.is_null() {
                d.unmap_memory(self.mem);
            }
            if self.buf != vk::Buffer::null() {
                d.destroy_buffer(self.buf, None);
            }
            if self.mem != vk::DeviceMemory::null() {
                d.free_memory(self.mem, None);
            }
        }
    }
}

/// Activations repacked for the GPU: activation-minor, so the 32 lanes of a
/// subgroup read 32 consecutive uints.
///
/// This is a layout change, not an arithmetic one: the same int8 values and
/// the same f32 scales, reindexed. The runtime produces activations itself,
/// so a transposed pack is a real option rather than a benchmark-only trick.
struct GpuActs {
    /// `[b][j][k][a]`, four int8 per uint.
    qs: Vec<u32>,
    /// `[b][a]`: Q8_K carries one scale per 256-weight block, so a row has
    /// `n_blocks` of them, not one.
    d: Vec<f32>,
    /// `[b][g][a]`.
    bsums: Vec<i32>,
}

fn repack_acts(acts: &[BlockQ8K], n_acts: usize, n_blocks: usize) -> Result<GpuActs> {
    if acts.len() != n_acts * n_blocks {
        bail!(
            "acts has {} blocks, expected n_acts * n_blocks = {}",
            acts.len(),
            n_acts * n_blocks
        );
    }
    let mut qs = vec![0u32; n_blocks * SUBBLOCKS * 8 * n_acts];
    let mut d = vec![0f32; n_blocks * n_acts];
    let mut bsums = vec![0i32; n_blocks * BSUM_GROUPS * n_acts];
    for a in 0..n_acts {
        for b in 0..n_blocks {
            let block = &acts[a * n_blocks + b];
            d[b * n_acts + a] = block.d;
            for j in 0..SUBBLOCKS {
                for k in 0..8 {
                    let base = 32 * j + 4 * k;
                    let word = u32::from_le_bytes([
                        block.qs[base] as u8,
                        block.qs[base + 1] as u8,
                        block.qs[base + 2] as u8,
                        block.qs[base + 3] as u8,
                    ]);
                    qs[((b * SUBBLOCKS + j) * 8 + k) * n_acts + a] = word;
                }
            }
            for (g, &bs) in block.bsums.iter().enumerate() {
                bsums[(b * BSUM_GROUPS + g) * n_acts + a] = i32::from(bs);
            }
        }
    }
    Ok(GpuActs { qs, d, bsums })
}

/// The whole GPU arm: buffers, pipeline, and a pre-recorded command buffer.
pub struct GpuGemv<'a> {
    ctx: &'a VkContext,
    weights: Buffer<'a>,
    act_qs: Buffer<'a>,
    act_d: Buffer<'a>,
    act_bsums: Buffer<'a>,
    out: Buffer<'a>,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    shader: vk::ShaderModule,
    cmd_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    query_pool: vk::QueryPool,
    timestamps_valid: bool,
    /// Whether the pipeline was created with a *required* subgroup size, as
    /// opposed to being allowed to run at whatever width the driver picked.
    /// Reported, because it is the difference between a measured access
    /// pattern and an assumed one.
    subgroup_pinned: bool,
    out_len: usize,
}

impl<'a> GpuGemv<'a> {
    /// Build every Vulkan object and record the dispatch once.
    ///
    /// `format` is the packed weight format the caller intends to upload; it
    /// is gated by [`check_shader_format`] rather than trusted, because the
    /// shader's `uint`-indexed reads are specific to [`SHADER_FORMAT`].
    pub fn new(
        ctx: &'a VkContext,
        format: QuantFormat,
        in_dim: usize,
        out_dim: usize,
        n_acts: usize,
    ) -> Result<Self> {
        check_shader_format(format)?;
        // Device limits the shader's declared geometry rests on. Both are
        // printed by the capability probe; neither was checked, which meant
        // the spike relied on the reference part happening to satisfy them.
        if ctx.caps.max_workgroup_invocations < LOCAL_SIZE {
            bail!(
                "the shader declares local_size_x = {LOCAL_SIZE} but this device allows only \
                 {} invocations per workgroup",
                ctx.caps.max_workgroup_invocations
            );
        }
        if ctx.caps.max_workgroup_size[0] < LOCAL_SIZE {
            bail!(
                "the shader declares local_size_x = {LOCAL_SIZE} but this device's \
                 maxComputeWorkGroupSize[0] is {}",
                ctx.caps.max_workgroup_size[0]
            );
        }
        // Whether the subgroup width can be *required* of the pipeline, or
        // only observed. `subgroupSize` alone settles nothing: it is the
        // device default, and the Intel backend picks SIMD8/16/32 per
        // pipeline from register pressure, so a pipeline can run 16 lanes on
        // a device whose default reads 32.
        let subgroup_pinned = ctx
            .caps
            .can_require_subgroup_size(ASSUMED_SUBGROUP, LOCAL_SIZE);
        if !subgroup_pinned && ctx.caps.subgroup_size != ASSUMED_SUBGROUP {
            bail!(
                "the activation-minor packing is built for a {ASSUMED_SUBGROUP}-lane subgroup \
                 (the lanes sharing one super-block read {ASSUMED_SUBGROUP} consecutive uints). \
                 This device cannot pin a compute subgroup size (subgroupSizeControl enabled \
                 {}, requiredSubgroupSizeStages {:?}, range {}..{}, maxComputeWorkgroupSubgroups \
                 {}) and its default subgroupSize is {}, so the pipeline will not run at \
                 {ASSUMED_SUBGROUP}. The arithmetic would still be correct, but the timing \
                 would not measure the access pattern the shader was written for, and a timing \
                 is the only thing this spike produces",
                ctx.caps.subgroup_size_control_enabled,
                ctx.caps.required_subgroup_size_stages,
                ctx.caps.subgroup_size_range.0,
                ctx.caps.subgroup_size_range.1,
                ctx.caps.max_compute_workgroup_subgroups,
                ctx.caps.subgroup_size,
            );
        }
        let n_blocks = in_dim / 256;
        if n_blocks == 0 || n_blocks > MAX_BLOCKS {
            bail!("in_dim {in_dim} needs 1..={MAX_BLOCKS} super-blocks per row");
        }
        if n_acts == 0 || n_acts > MAX_ACTS {
            bail!("n_acts {n_acts} is outside the shader's 1..={MAX_ACTS}");
        }
        if (n_acts * n_blocks) as u32 > LOCAL_SIZE {
            bail!("n_acts * n_blocks must fit the {LOCAL_SIZE}-invocation workgroup");
        }

        let weight_bytes = (out_dim * n_blocks * UINTS_PER_BLOCK * 4) as u64;
        let weights = Buffer::new(ctx, weight_bytes)?;
        let act_qs = Buffer::new(ctx, (n_blocks * SUBBLOCKS * 8 * n_acts * 4) as u64)?;
        let act_d = Buffer::new(ctx, (n_blocks * n_acts * 4) as u64)?;
        let act_bsums = Buffer::new(ctx, (n_blocks * BSUM_GROUPS * n_acts * 4) as u64)?;
        let out_len = out_dim * n_acts;
        let out = Buffer::new(ctx, (out_len * 4) as u64)?;

        // Both halves matter and neither implies the other: a zero period
        // gives nothing to convert ticks with, and a zero
        // `timestampValidBits` on the family this spike submits on means
        // `cmd_write_timestamp` writes undefined values there. Reporting a
        // GFLOP/s off either would be fabricating one.
        let timestamps_valid = ctx.caps.timestamps_usable();

        // From here on every handle is a field of `this` the moment it
        // exists, in the same style as `Buffer::new`: null is the "not
        // acquired" state, `Drop` releases whatever was acquired and skips
        // the rest, and each `?` below therefore unwinds cleanly instead of
        // leaking the handles taken before it. `create_compute_pipelines`
        // failing used to leak four of them.
        let mut this = Self {
            ctx,
            weights,
            act_qs,
            act_d,
            act_bsums,
            out,
            set_layout: vk::DescriptorSetLayout::null(),
            pool: vk::DescriptorPool::null(),
            // Freed with its pool, so it needs no null check in `Drop`.
            set: vk::DescriptorSet::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            shader: vk::ShaderModule::null(),
            cmd_pool: vk::CommandPool::null(),
            // As `set`: freed with `cmd_pool`.
            cmd: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            query_pool: vk::QueryPool::null(),
            timestamps_valid,
            subgroup_pinned,
            out_len,
        };

        let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = (0..5)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let sl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: `sl_ci` outlives the call.
        this.set_layout = unsafe { ctx.device.create_descriptor_set_layout(&sl_ci, None) }
            .context("creating the descriptor set layout")?;

        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(5)];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&sizes);
        // SAFETY: `pool_ci` outlives the call.
        this.pool = unsafe { ctx.device.create_descriptor_pool(&pool_ci, None) }
            .context("creating the descriptor pool")?;
        let layouts = [this.set_layout];
        let alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(this.pool)
            .set_layouts(&layouts);
        // SAFETY: `alloc` outlives the call and the pool has room for one set.
        this.set = unsafe { ctx.device.allocate_descriptor_sets(&alloc) }
            .context("allocating the descriptor set")?[0];

        let infos: Vec<[vk::DescriptorBufferInfo; 1]> = [
            &this.weights,
            &this.act_qs,
            &this.act_d,
            &this.act_bsums,
            &this.out,
        ]
        .iter()
        .map(|b| {
            [vk::DescriptorBufferInfo::default()
                .buffer(b.buf)
                .offset(0)
                .range(vk::WHOLE_SIZE)]
        })
        .collect();
        let writes: Vec<vk::WriteDescriptorSet<'_>> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(this.set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        // SAFETY: every buffer referenced outlives the descriptor set.
        unsafe { ctx.device.update_descriptor_sets(&writes, &[]) };

        let code = ash::util::read_spv(&mut Cursor::new(SPV)).context("parsing the SPIR-V")?;
        let sm_ci = vk::ShaderModuleCreateInfo::default().code(&code);
        // SAFETY: `sm_ci` outlives the call.
        this.shader = unsafe { ctx.device.create_shader_module(&sm_ci, None) }
            .context("creating the shader module")?;

        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(12)];
        let pl_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&ranges);
        // SAFETY: `pl_ci` outlives the call.
        this.pipeline_layout = unsafe { ctx.device.create_pipeline_layout(&pl_ci, None) }
            .context("creating the pipeline layout")?;

        // Where the device allows it, the subgroup width is *required* of
        // this pipeline rather than left to the compiler, so the 32-lane
        // access pattern the activation packing is built around is the one
        // that actually runs and therefore the one that is timed.
        let mut required_size = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(ASSUMED_SUBGROUP);
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(this.shader)
            .name(c"main");
        if subgroup_pinned {
            stage = stage.push_next(&mut required_size);
        }
        let cp_ci = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(this.pipeline_layout)];
        // SAFETY: `cp_ci` and its pNext chain outlive the call.
        this.pipeline = unsafe {
            ctx.device
                .create_compute_pipelines(vk::PipelineCache::null(), &cp_ci, None)
        }
        .map_err(|(_, e)| anyhow!("creating the compute pipeline: {e}"))?[0];

        let cmd_pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(ctx.queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: `cmd_pool_ci` outlives the call.
        this.cmd_pool = unsafe { ctx.device.create_command_pool(&cmd_pool_ci, None) }
            .context("creating the command pool")?;
        let cb_ai = vk::CommandBufferAllocateInfo::default()
            .command_pool(this.cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: `cb_ai` outlives the call.
        this.cmd = unsafe { ctx.device.allocate_command_buffers(&cb_ai) }
            .context("allocating the command buffer")?[0];

        // SAFETY: nothing is in flight yet.
        this.fence = unsafe {
            ctx.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .context("creating the fence")?;

        let qp_ci = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(2);
        // SAFETY: `qp_ci` outlives the call.
        this.query_pool = unsafe { ctx.device.create_query_pool(&qp_ci, None) }
            .context("creating the timestamp query pool")?;

        this.record(out_dim as u32, n_acts as u32, n_blocks as u32)?;
        Ok(this)
    }

    /// Whether the pipeline runs at a subgroup size this spike *required*, or
    /// merely at the one the driver chose.
    ///
    /// `false` does not mean the arithmetic is wrong; it means the
    /// activation-minor packing's 32-lane premise is unverified for this
    /// pipeline, so the timing may be of a different access pattern than the
    /// one the shader was written for.
    pub fn subgroup_pinned(&self) -> bool {
        self.subgroup_pinned
    }

    fn record(&self, out_dim: u32, n_acts: u32, n_blocks: u32) -> Result<()> {
        let d = &self.ctx.device;
        let begin = vk::CommandBufferBeginInfo::default();
        // SAFETY: the command buffer is not in flight (nothing submitted yet)
        // and every handle recorded below outlives it.
        unsafe {
            d.begin_command_buffer(self.cmd, &begin)
                .context("beginning the command buffer")?;
            d.cmd_reset_query_pool(self.cmd, self.query_pool, 0, 2);
            d.cmd_write_timestamp(
                self.cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pool,
                0,
            );
            d.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            d.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.set],
                &[],
            );
            let push = [out_dim, n_acts, n_blocks];
            d.cmd_push_constants(
                self.cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(push.as_ptr().cast::<u8>(), 12),
            );
            d.cmd_dispatch(self.cmd, out_dim, 1, 1);
            d.cmd_write_timestamp(
                self.cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                1,
            );
            let barrier = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            d.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            );
            d.end_command_buffer(self.cmd)
                .context("ending the command buffer")?;
        }
        Ok(())
    }

    /// Repack the packed weight matrix into the storage buffer.
    ///
    /// The bytes are copied verbatim: the shader reads the same packed Q4_K
    /// super-blocks the CPU kernel reads, as uints.
    pub fn upload_weights(&self, weight: &[u8]) -> Result<()> {
        self.weights.write(weight)
    }

    /// Repack and upload the activation batch.
    pub fn upload_acts(&self, acts: &[BlockQ8K], n_acts: usize, n_blocks: usize) -> Result<()> {
        let packed = repack_acts(acts, n_acts, n_blocks)?;
        self.act_qs.write(&packed.qs)?;
        self.act_d.write(&packed.d)?;
        self.act_bsums.write(&packed.bsums)
    }

    /// Property flags of the memory type every buffer landed in.
    ///
    /// [`Buffer::new`] takes the first type matching `DEVICE_LOCAL |
    /// HOST_VISIBLE | HOST_COHERENT`, so this is a report of what the driver
    /// happened to list first and not a tuned choice. It matters to the
    /// timings: such memory is typically write-combined, which makes host
    /// writes into the mapping stream and host reads out of it slow.
    pub fn memory_properties(&self) -> vk::MemoryPropertyFlags {
        self.weights.props
    }

    /// Submit the recorded dispatch and wait for it, with the weights already
    /// resident in device-visible memory.
    ///
    /// Returns `(host wall seconds, GPU timestamp seconds)`. The host figure
    /// covers exactly `reset_fences` + `queue_submit` + `wait_for_fences`:
    /// no transfer of any kind is inside it, which is only what a backend
    /// pays when it dispatches the same expert again. [`GpuGemv::run_streamed`]
    /// measures the other end. The GPU figure is the dispatch alone.
    pub fn run(&self) -> Result<(f64, f64)> {
        let start = Instant::now();
        self.submit_and_wait()?;
        let host = start.elapsed().as_secs_f64();
        Ok((host, self.device_time()))
    }

    /// As [`GpuGemv::run`], with the whole weight slab written into the
    /// mapped buffer *inside* the timed region.
    ///
    /// This is the transfer a real backend cannot avoid: a prefill chunk
    /// dispatches a different expert every time, so those bytes have to land
    /// in device-visible memory once per dispatch. The CPU arm reads the same
    /// bytes straight from the heap and copies nothing, so quoting the
    /// resident number against it compares an arm that transfers with an arm
    /// that does not.
    pub fn run_streamed(&self, weight: &[u8]) -> Result<(f64, f64)> {
        let start = Instant::now();
        self.weights.write(weight)?;
        self.submit_and_wait()?;
        let host = start.elapsed().as_secs_f64();
        Ok((host, self.device_time()))
    }

    fn submit_and_wait(&self) -> Result<()> {
        let d = &self.ctx.device;
        let cmds = [self.cmd];
        let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
        // SAFETY: the command buffer is not already in flight (the previous
        // call waited on `fence` before returning), and `submit` outlives
        // the call.
        unsafe {
            d.reset_fences(&[self.fence])
                .context("resetting the fence")?;
            d.queue_submit(self.ctx.queue, &submit, self.fence)
                .context("submitting the dispatch")?;
            d.wait_for_fences(&[self.fence], true, u64::MAX)
                .context("waiting on the dispatch")?;
        }
        Ok(())
    }

    /// Seconds between the two timestamps of the dispatch that just finished,
    /// or 0 if the device does not carry usable timestamps.
    fn device_time(&self) -> f64 {
        if !self.timestamps_valid {
            return 0.0;
        }
        let mut ticks = [0u64; 2];
        // SAFETY: both queries were written by the command buffer that has
        // now completed (the caller waited on its fence).
        let got = unsafe {
            self.ctx.device.get_query_pool_results(
                self.query_pool,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        };
        if got.is_err() {
            return 0.0;
        }
        // Only the low `timestampValidBits` of each tick count are defined,
        // so the difference is taken modulo that width rather than as a
        // 64-bit subtraction: on a family with fewer than 64 valid bits the
        // raw values carry undefined high bits and can appear to run
        // backwards across a wrap.
        let mask = self.ctx.caps.timestamp_mask();
        let delta = (ticks[1].wrapping_sub(ticks[0]) & mask) as f64;
        delta * f64::from(self.ctx.caps.timestamp_period) / 1e9
    }

    /// Copy the output buffer out. Call after [`GpuGemv::run`].
    pub fn read_out(&self, dst: &mut [f32]) -> Result<()> {
        if dst.len() != self.out_len {
            bail!(
                "out slice has {} floats, expected {}",
                dst.len(),
                self.out_len
            );
        }
        self.out.read(dst)
    }
}

impl Drop for GpuGemv<'_> {
    fn drop(&mut self) {
        let d = &self.ctx.device;
        // SAFETY: `device_wait_idle` drains everything this struct submitted
        // before any handle is destroyed, and no handle is shared. Each
        // handle is released only if it was actually acquired, which is what
        // the null checks test: a `GpuGemv` that failed part-way through
        // `new` holds nulls for the rest, exactly as a part-built `Buffer`
        // does. `set` and `cmd` are freed with their pools and are not
        // released here. The five buffers are not touched either: they own
        // their own handles and their fields are dropped straight after this
        // body returns.
        unsafe {
            let _ = d.device_wait_idle();
            if self.query_pool != vk::QueryPool::null() {
                d.destroy_query_pool(self.query_pool, None);
            }
            if self.fence != vk::Fence::null() {
                d.destroy_fence(self.fence, None);
            }
            if self.cmd_pool != vk::CommandPool::null() {
                d.destroy_command_pool(self.cmd_pool, None);
            }
            if self.pipeline != vk::Pipeline::null() {
                d.destroy_pipeline(self.pipeline, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                d.destroy_pipeline_layout(self.pipeline_layout, None);
            }
            if self.shader != vk::ShaderModule::null() {
                d.destroy_shader_module(self.shader, None);
            }
            if self.pool != vk::DescriptorPool::null() {
                d.destroy_descriptor_pool(self.pool, None);
            }
            if self.set_layout != vk::DescriptorSetLayout::null() {
                d.destroy_descriptor_set_layout(self.set_layout, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The alignment the shader's weight reads assume, pinned.
    ///
    /// `crates/core`'s kernel rule is that a vectorized path states and tests
    /// the alignment it relies on. The shader's is coarser than the CPU
    /// kernels': it indexes the whole weight buffer as `uint w[]`, so it
    /// needs the packed block size to be a whole number of 4-byte words, not
    /// merely the 2-byte alignment a packed sub-tensor offset guarantees.
    #[test]
    fn the_shader_format_is_uint_addressable() {
        assert_eq!(SHADER_FORMAT, QuantFormat::Q4_K);
        assert_eq!(SHADER_FORMAT.block_bytes(), 144);
        assert!(
            SHADER_FORMAT.block_bytes().is_multiple_of(4),
            "the shader indexes `uint w[]`, so its format's block must be a whole number of uints"
        );
        assert_eq!(UINTS_PER_BLOCK * 4, SHADER_FORMAT.block_bytes());
        check_shader_format(SHADER_FORMAT).expect("the shader's own format must be accepted");
    }

    /// And the format that breaks it, named, because it is not hypothetical:
    /// Q6_K is the expert `down` projection on 24 of the v0 model's 48
    /// layers, so this is half the expert FFN and not an edge case.
    #[test]
    fn q6_k_is_refused_because_210_is_not_a_multiple_of_four() {
        assert_eq!(QuantFormat::Q6_K.block_bytes(), 210);
        assert!(
            !QuantFormat::Q6_K.block_bytes().is_multiple_of(4),
            "210 B is the whole reason this shader does not generalize"
        );
        let err = check_shader_format(QuantFormat::Q6_K)
            .expect_err("Q6_K must be refused, not silently misread")
            .to_string();
        assert!(
            err.contains("210") && err.contains("uint"),
            "the refusal must name the size and the reason, got: {err}"
        );
    }

    /// A format that is uint-addressable but is still not this shader's
    /// arithmetic must also be refused, so the gate cannot be read as "any
    /// 4-byte-aligned k-quant works".
    #[test]
    fn a_uint_addressable_but_wrong_format_is_still_refused() {
        assert!(QuantFormat::Q5_K.block_bytes().is_multiple_of(4));
        assert!(check_shader_format(QuantFormat::Q5_K).is_err());
    }

    /// The activation repack is a pure reindex: every int8, every bsum, and
    /// every scale must come back out of the GPU-side arrays unchanged. A
    /// transposition bug here would show up as a wrong answer, not a crash,
    /// so it is worth a test that needs no device.
    #[test]
    fn repack_is_a_pure_reindex() {
        let n_acts = 4;
        let n_blocks = 3;
        let mut acts = vec![BlockQ8K::default(); n_acts * n_blocks];
        for (i, block) in acts.iter_mut().enumerate() {
            block.d = 0.5 + i as f32;
            for (j, q) in block.qs.iter_mut().enumerate() {
                *q = ((i * 7 + j * 13) as i32 % 255 - 127) as i8;
            }
            for (g, s) in block.bsums.iter_mut().enumerate() {
                *s = (i * 31 + g) as i16;
            }
        }
        let packed = repack_acts(&acts, n_acts, n_blocks).expect("repack");
        for a in 0..n_acts {
            for b in 0..n_blocks {
                let src = &acts[a * n_blocks + b];
                assert_eq!(packed.d[b * n_acts + a], src.d, "d at ({a}, {b})");
                for g in 0..BSUM_GROUPS {
                    assert_eq!(
                        packed.bsums[(b * BSUM_GROUPS + g) * n_acts + a],
                        i32::from(src.bsums[g]),
                        "bsum at ({a}, {b}, {g})"
                    );
                }
                for j in 0..SUBBLOCKS {
                    for k in 0..8 {
                        let word = packed.qs[((b * SUBBLOCKS + j) * 8 + k) * n_acts + a];
                        for t in 0..4 {
                            let got = ((word >> (8 * t)) & 0xFF) as u8 as i8;
                            assert_eq!(
                                got,
                                src.qs[32 * j + 4 * k + t],
                                "qs at ({a}, {b}, {j}, {k}, {t})"
                            );
                        }
                    }
                }
            }
        }
    }
}
