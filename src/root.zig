const bandit = @import("bandit.zig");

pub const ArmPosterior = bandit.ArmPosterior;
pub const CapacityAllocator = bandit.CapacityAllocator;
pub const rngFromSeed = bandit.rngFromSeed;

pub const alloc_guard = @import("alloc_guard.zig");
pub const CountingAllocator = alloc_guard.CountingAllocator;
pub const expectNoAllocations = alloc_guard.expectNoAllocations;
pub const expectAtMostAllocations = alloc_guard.expectAtMostAllocations;

test {
    @import("std").testing.refAllDecls(@This());
}
