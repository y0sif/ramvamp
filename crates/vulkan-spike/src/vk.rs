//! Vulkan instance/device bring-up and the capability probe.
//!
//! Everything the shader design depends on is read off the physical device
//! here and printed, because two of the answers change the design: whether
//! the 4x8-bit packed integer dot product is *hardware* accelerated (DP4a),
//! and whether device-local memory is host visible (an iGPU's unified heap
//! removes the staging copy entirely).

use std::ffi::{CStr, c_char};

use anyhow::{Context, Result, anyhow, bail};
use ash::{Device, Entry, Instance, vk};

/// What the probe found, in the shape the report wants.
pub struct Caps {
    /// `deviceName`.
    pub device_name: String,
    /// Device type, e.g. integrated GPU.
    pub device_type: String,
    /// `apiVersion`, formatted.
    pub api_version: String,
    /// `VK_KHR_driver_properties` name and info.
    pub driver: String,
    /// Whether the device advertises `shaderIntegerDotProduct`.
    pub dot_product_feature: bool,
    /// `integerDotProduct4x8BitPackedSignedAccelerated`.
    pub dp4a_signed_accelerated: bool,
    /// `integerDotProduct4x8BitPackedUnsignedAccelerated`.
    pub dp4a_unsigned_accelerated: bool,
    /// `integerDotProduct4x8BitPackedMixedSignednessAccelerated`.
    pub dp4a_mixed_accelerated: bool,
    /// `subgroupSize`, the device's *default* subgroup size.
    ///
    /// Not the width a given pipeline runs at. Mesa's Intel backend compiles
    /// a compute shader at SIMD8, SIMD16 or SIMD32 from register pressure and
    /// picks per pipeline, so this value is a default and not a guarantee.
    /// [`Caps::can_require_subgroup_size`] says whether the size can be
    /// pinned instead of merely observed.
    pub subgroup_size: u32,
    /// `minSubgroupSize` / `maxSubgroupSize` from subgroup size control.
    pub subgroup_size_range: (u32, u32),
    /// Whether the device advertises the `subgroupSizeControl` *feature*.
    pub subgroup_size_control_feature: bool,
    /// Whether the feature was actually enabled on the logical device.
    ///
    /// Separate from the advertisement: passing
    /// `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` without the
    /// enabled feature is a validation error, so the pipeline path gates on
    /// this and not on what the physical device offered.
    pub subgroup_size_control_enabled: bool,
    /// `requiredSubgroupSizeStages`: the stages for which a subgroup size may
    /// be pinned at pipeline creation.
    pub required_subgroup_size_stages: vk::ShaderStageFlags,
    /// `maxComputeWorkgroupSubgroups`: the cap on
    /// `local_size / requiredSubgroupSize` when a size is pinned.
    pub max_compute_workgroup_subgroups: u32,
    /// `maxComputeWorkGroupInvocations`.
    pub max_workgroup_invocations: u32,
    /// `maxComputeWorkGroupSize`.
    pub max_workgroup_size: [u32; 3],
    /// `maxComputeSharedMemorySize`, bytes.
    pub max_shared_memory: u32,
    /// Nanoseconds per timestamp tick.
    pub timestamp_period: f32,
    /// `timestampValidBits` of the queue family the spike submits on.
    ///
    /// This is the per-family limit, not a device-wide one, and it is what
    /// makes `vkCmdWriteTimestamp` legal at all: where it is 0 the family
    /// supports no timestamps and the values written are undefined, so a
    /// GFLOP/s computed from them is fabricated. Where it is between 1 and
    /// 63 only that many low bits of each tick count are defined, which is
    /// why [`GpuGemv::device_time`](crate::gpu::GpuGemv) masks the delta to
    /// this width instead of subtracting the raw 64-bit values.
    pub timestamp_valid_bits: u32,
    /// One line per memory heap, with the property flags of its types.
    pub heaps: Vec<String>,
    /// Whether some memory type is DEVICE_LOCAL and HOST_VISIBLE at once.
    pub unified_memory: bool,
    /// Whether some host-visible memory type is also HOST_CACHED.
    ///
    /// Relevant to any timing that includes a readback: the spike's buffers
    /// go in the first `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` type,
    /// which is write-combined, and reading an output buffer back out of
    /// write-combined memory is slow. A cached type, where one exists, is
    /// where an output buffer would belong instead.
    pub host_cached_available: bool,
}

/// An owned Vulkan device plus the queue and pool the spike submits on.
pub struct VkContext {
    /// Kept alive for the instance's sake; the loader must outlive it.
    _entry: Entry,
    instance: Instance,
    /// Logical device.
    pub device: Device,
    /// Compute queue.
    pub queue: vk::Queue,
    /// Family the queue came from.
    pub queue_family: u32,
    /// Memory properties, cached.
    pub memory: vk::PhysicalDeviceMemoryProperties,
    /// Probe results.
    pub caps: Caps,
}

/// Read a fixed-size `char[]` device string.
fn c_str(bytes: &[c_char]) -> String {
    // SAFETY: Vulkan guarantees these arrays are NUL-terminated within
    // their declared length.
    let s = unsafe { CStr::from_ptr(bytes.as_ptr()) };
    s.to_string_lossy().into_owned()
}

fn format_version(v: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(v),
        vk::api_version_minor(v),
        vk::api_version_patch(v)
    )
}

impl VkContext {
    /// Bring up an instance, pick a device, and probe it.
    ///
    /// `prefer_integrated` picks the first integrated GPU when one exists;
    /// otherwise the first device with a compute queue wins.
    pub fn new(prefer_integrated: bool) -> Result<Self> {
        // SAFETY: `Entry::load` dlopen's the loader; nothing here outlives
        // the returned `Entry`, which is stored in the struct.
        let entry = unsafe { Entry::load() }.context("loading the Vulkan loader")?;

        let app = vk::ApplicationInfo::default()
            .application_name(c"ramvamp-vulkan-spike")
            .api_version(vk::make_api_version(0, 1, 3, 0));
        let ci = vk::InstanceCreateInfo::default().application_info(&app);
        // SAFETY: `ci` and everything it borrows outlive the call.
        let instance = unsafe { entry.create_instance(&ci, None) }
            .context("creating the Vulkan instance (is a 1.3 driver installed?)")?;

        // SAFETY: `instance` is live.
        let devices =
            unsafe { instance.enumerate_physical_devices() }.context("enumerating devices")?;
        if devices.is_empty() {
            bail!("no Vulkan physical devices");
        }
        let mut chosen = None;
        for &pd in &devices {
            // SAFETY: `pd` came from this instance.
            let props = unsafe { instance.get_physical_device_properties(pd) };
            let integrated = props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU;
            if !prefer_integrated || integrated {
                chosen = Some(pd);
                break;
            }
            chosen.get_or_insert(pd);
        }
        let physical = chosen.ok_or_else(|| anyhow!("no usable physical device"))?;

        // The family is chosen before the probe, because `timestampValidBits`
        // is a property of *this* family and not of the device: probing it
        // against any other family would report a limit the spike never
        // submits under.
        // SAFETY: `physical` came from this instance.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let queue_family = families
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or_else(|| anyhow!("no compute queue family"))? as u32;

        let mut caps = probe(
            &instance,
            physical,
            families[queue_family as usize].timestamp_valid_bits,
        );

        let priorities = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        // `subgroupSizeControl` is requested whenever the device advertises
        // it, because it is what turns the shader's 32-lane assumption from
        // something observed into something required: with it enabled the
        // pipeline carries a
        // `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` and the
        // driver may not pick another width. Without it the width is the
        // compiler's choice per pipeline and the spike can only report that.
        let want_size_control = caps.subgroup_size_control_feature;
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .shader_integer_dot_product(true)
            .subgroup_size_control(want_size_control);
        let dci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qci)
            .push_next(&mut features13);
        // SAFETY: `dci` and its pNext chain outlive the call.
        let device = unsafe { instance.create_device(physical, &dci, None) }
            .context("creating the logical device")?;
        caps.subgroup_size_control_enabled = want_size_control;
        // SAFETY: the family index came from this device's own list.
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        // SAFETY: `physical` came from this instance.
        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };

        Ok(Self {
            _entry: entry,
            instance,
            device,
            queue,
            queue_family,
            memory,
            caps,
        })
    }

    /// Index of a memory type in `bits` carrying every flag in `want`.
    pub fn memory_type(&self, bits: u32, want: vk::MemoryPropertyFlags) -> Result<u32> {
        (0..self.memory.memory_type_count)
            .find(|&i| {
                bits & (1 << i) != 0
                    && self.memory.memory_types[i as usize]
                        .property_flags
                        .contains(want)
            })
            .ok_or_else(|| anyhow!("no memory type with {want:?} among the allowed types"))
    }
}

impl Drop for VkContext {
    fn drop(&mut self) {
        // SAFETY: every child object the spike creates is destroyed before
        // the context is dropped, and the queue is idle by then.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn probe(instance: &Instance, physical: vk::PhysicalDevice, timestamp_valid_bits: u32) -> Caps {
    let mut driver_props = vk::PhysicalDeviceDriverProperties::default();
    let mut dot_props = vk::PhysicalDeviceShaderIntegerDotProductProperties::default();
    let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::default();
    let mut size_control = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default()
        .push_next(&mut driver_props)
        .push_next(&mut dot_props)
        .push_next(&mut subgroup_props)
        .push_next(&mut size_control);
    // SAFETY: `physical` came from `instance`; the pNext chain outlives the
    // call and every struct in it is default-initialized with its sType.
    unsafe { instance.get_physical_device_properties2(physical, &mut props2) };

    let mut dot_feature = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
    let mut size_control_feature = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut dot_feature)
        .push_next(&mut size_control_feature);
    // SAFETY: as above.
    unsafe { instance.get_physical_device_features2(physical, &mut features2) };

    let props = props2.properties;
    let limits = props.limits;

    let mut heaps = Vec::new();
    // SAFETY: `physical` came from `instance`.
    let mem = unsafe { instance.get_physical_device_memory_properties(physical) };
    let mut unified = false;
    let mut host_cached = false;
    for h in 0..mem.memory_heap_count as usize {
        let heap = mem.memory_heaps[h];
        let mut kinds = Vec::new();
        for t in 0..mem.memory_type_count as usize {
            let ty = mem.memory_types[t];
            if ty.heap_index as usize != h {
                continue;
            }
            let f = ty.property_flags;
            if f.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                && f.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            {
                unified = true;
            }
            if f.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                && f.contains(vk::MemoryPropertyFlags::HOST_CACHED)
            {
                host_cached = true;
            }
            // Each type is numbered and comma-separated: joining the flag
            // sets with " | " ran them together into one unreadable string
            // in which the type boundaries were invisible, and which type a
            // buffer landed in is exactly the question this line has to
            // answer.
            kinds.push(format!("#{t} {f:?}"));
        }
        heaps.push(format!(
            "heap {h}: {:.2} GiB, flags {:?}, types [{}]",
            heap.size as f64 / (1u64 << 30) as f64,
            heap.flags,
            kinds.join(", ")
        ));
    }

    Caps {
        device_name: c_str(&props.device_name),
        device_type: format!("{:?}", props.device_type),
        api_version: format_version(props.api_version),
        driver: format!(
            "{:?} / {} / {}",
            driver_props.driver_id,
            c_str(&driver_props.driver_name),
            c_str(&driver_props.driver_info)
        ),
        dot_product_feature: dot_feature.shader_integer_dot_product != 0,
        dp4a_signed_accelerated: dot_props.integer_dot_product4x8_bit_packed_signed_accelerated
            != 0,
        dp4a_unsigned_accelerated: dot_props.integer_dot_product4x8_bit_packed_unsigned_accelerated
            != 0,
        dp4a_mixed_accelerated: dot_props
            .integer_dot_product4x8_bit_packed_mixed_signedness_accelerated
            != 0,
        subgroup_size: subgroup_props.subgroup_size,
        subgroup_size_range: (
            size_control.min_subgroup_size,
            size_control.max_subgroup_size,
        ),
        subgroup_size_control_feature: size_control_feature.subgroup_size_control != 0,
        // Set by `VkContext::new` once the logical device exists; the probe
        // runs before it and cannot know what was enabled.
        subgroup_size_control_enabled: false,
        required_subgroup_size_stages: size_control.required_subgroup_size_stages,
        max_compute_workgroup_subgroups: size_control.max_compute_workgroup_subgroups,
        max_workgroup_invocations: limits.max_compute_work_group_invocations,
        max_workgroup_size: limits.max_compute_work_group_size,
        max_shared_memory: limits.max_compute_shared_memory_size,
        timestamp_period: limits.timestamp_period,
        timestamp_valid_bits,
        heaps,
        unified_memory: unified,
        host_cached_available: host_cached,
    }
}

/// Mask of the defined low bits of a timestamp tick count at `valid_bits`.
///
/// A free function rather than only a method on [`Caps`], because it is the
/// one piece of the device-timing path that can be wrong silently: an
/// off-by-one here does not fail, it corrupts every reported dispatch time
/// and every GFLOP/s derived from one. Testing it needs no device.
///
/// `valid_bits` of 0 yields 0, which is the honest answer: the family
/// supports no timestamps, so no bit of a tick count is defined. That case
/// never reaches the arithmetic, because [`Caps::timestamps_usable`] already
/// suppresses the series, but the shift below would be a no-op at 0 rather
/// than a mask and the distinction is worth pinning.
pub fn timestamp_mask_for(valid_bits: u32) -> u64 {
    if valid_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << valid_bits) - 1
    }
}

impl Caps {
    /// Whether a timestamp read off this device's compute queue means
    /// anything.
    ///
    /// Both halves are required and neither implies the other: a zero period
    /// gives no seconds-per-tick to convert with, and a zero
    /// `timestampValidBits` says the *family* the spike submits on does not
    /// support `vkCmdWriteTimestamp`, so the ticks themselves are undefined.
    pub fn timestamps_usable(&self) -> bool {
        self.timestamp_period > 0.0 && self.timestamp_valid_bits > 0
    }

    /// Mask of the defined low bits of a timestamp tick count on this family.
    pub fn timestamp_mask(&self) -> u64 {
        timestamp_mask_for(self.timestamp_valid_bits)
    }

    /// Whether a compute pipeline on this device may *require* a subgroup
    /// size of `size` rather than accept whatever the compiler picks.
    ///
    /// Three conditions, all of them real. The `subgroupSizeControl` feature
    /// has to be enabled on the logical device (passing the required-size
    /// struct without it is invalid usage), `requiredSubgroupSizeStages` has
    /// to include COMPUTE, and `size` has to be a power of two inside
    /// `minSubgroupSize..=maxSubgroupSize` with the workgroup fitting in
    /// `maxComputeWorkgroupSubgroups` subgroups at that width.
    pub fn can_require_subgroup_size(&self, size: u32, local_size: u32) -> bool {
        let (min, max) = self.subgroup_size_range;
        self.subgroup_size_control_enabled
            && self
                .required_subgroup_size_stages
                .contains(vk::ShaderStageFlags::COMPUTE)
            && size.is_power_of_two()
            && size >= min
            && size <= max
            && local_size.div_ceil(size) <= self.max_compute_workgroup_subgroups
    }

    /// Print the probe, loudly where it changes the design.
    pub fn report(&self) {
        println!("== Vulkan device ==");
        println!("  name              : {}", self.device_name);
        println!("  type              : {}", self.device_type);
        println!("  api version       : {}", self.api_version);
        println!("  driver            : {}", self.driver);
        println!(
            "  shaderIntegerDotProduct feature : {}",
            self.dot_product_feature
        );
        println!(
            "  DP4a 4x8 packed signed   accelerated : {}",
            self.dp4a_signed_accelerated
        );
        println!(
            "  DP4a 4x8 packed unsigned accelerated : {}",
            self.dp4a_unsigned_accelerated
        );
        println!(
            "  DP4a 4x8 packed mixed    accelerated : {}",
            self.dp4a_mixed_accelerated
        );
        println!(
            "  subgroup size     : {} (device DEFAULT, not the width any one pipeline runs at)",
            self.subgroup_size
        );
        println!(
            "  subgroup size range: {}..{}",
            self.subgroup_size_range.0, self.subgroup_size_range.1
        );
        println!(
            "  subgroupSizeControl: feature {}, enabled {}, requiredSubgroupSizeStages {:?}, \
             maxComputeWorkgroupSubgroups {}",
            self.subgroup_size_control_feature,
            self.subgroup_size_control_enabled,
            self.required_subgroup_size_stages,
            self.max_compute_workgroup_subgroups
        );
        println!(
            "  max workgroup     : {} invocations, size {:?}",
            self.max_workgroup_invocations, self.max_workgroup_size
        );
        println!("  max shared memory : {} B", self.max_shared_memory);
        println!("  timestamp period  : {} ns/tick", self.timestamp_period);
        println!(
            "  timestampValidBits (compute family): {}",
            self.timestamp_valid_bits
        );
        if !self.timestamps_usable() {
            println!(
                "  !! WARNING: this queue family carries NO usable timestamps \
                 (timestampValidBits {}, period {} ns). The device-timestamp series is \
                 suppressed rather than reported, because a GFLOP/s computed from \
                 undefined tick values is a fabricated number.",
                self.timestamp_valid_bits, self.timestamp_period
            );
        }
        for h in &self.heaps {
            println!("  {h}");
        }
        println!(
            "  unified memory (DEVICE_LOCAL | HOST_VISIBLE): {}",
            self.unified_memory
        );
        if !self.dp4a_signed_accelerated {
            println!(
                "  !! WARNING: 4x8-bit packed SIGNED integer dot product is NOT hardware \
                 accelerated on this device. The int8 path below is EMULATED; treat the \
                 GPU number as a lower bound and do not read it as a DP4a result."
            );
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::timestamp_mask_for;

    /// A family that carries no timestamps defines no bits of a tick count.
    /// The series is suppressed before the mask is ever applied, but a naive
    /// `(1 << 0) - 1` and a "no bits" answer agree only by luck, so it is
    /// pinned.
    #[test]
    fn zero_valid_bits_masks_everything_away() {
        assert_eq!(timestamp_mask_for(0), 0);
    }

    /// One valid bit keeps one bit.
    #[test]
    fn one_valid_bit_keeps_one_bit() {
        assert_eq!(timestamp_mask_for(1), 0b1);
    }

    /// The reference part's own answer, and the case every published device
    /// timing on this box passes through: 36 bits, so a delta must be taken
    /// modulo 2^36 and not as a 64-bit subtraction.
    #[test]
    fn thirty_six_valid_bits_is_the_reference_part() {
        assert_eq!(timestamp_mask_for(36), (1u64 << 36) - 1);
        assert_eq!(timestamp_mask_for(36), 0x0000_000F_FFFF_FFFF);
        // The property the mask exists for: a tick count that wrapped inside
        // the valid width still yields the true positive delta, where a raw
        // 64-bit subtraction of values carrying undefined high bits would
        // not.
        let mask = timestamp_mask_for(36);
        let start = (1u64 << 36) - 10;
        let end = 5u64;
        assert_eq!((end.wrapping_sub(start)) & mask, 15);
    }

    /// 63 is the largest width that still needs a shift, and the off-by-one
    /// most likely to be written as `1 << 64`, which panics in debug and is
    /// undefined in release.
    #[test]
    fn sixty_three_valid_bits_keeps_all_but_the_top() {
        assert_eq!(timestamp_mask_for(63), u64::MAX >> 1);
        assert_eq!(timestamp_mask_for(63), 0x7FFF_FFFF_FFFF_FFFF);
    }

    /// 64 and anything above it is the whole word, taken by the branch that
    /// avoids shifting by the width of the type.
    #[test]
    fn sixty_four_or_more_valid_bits_is_the_whole_word() {
        assert_eq!(timestamp_mask_for(64), u64::MAX);
        assert_eq!(timestamp_mask_for(65), u64::MAX);
        assert_eq!(timestamp_mask_for(u32::MAX), u64::MAX);
    }
}
