const bandit = @import("bandit.zig");

pub const ArmPosterior = bandit.ArmPosterior;
pub const CapacityAllocator = bandit.CapacityAllocator;
pub const rngFromSeed = bandit.rngFromSeed;

test {
    @import("std").testing.refAllDecls(@This());
}
