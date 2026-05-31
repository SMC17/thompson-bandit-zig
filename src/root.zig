const bandit = @import("bandit.zig");

pub const ArmPosterior = bandit.ArmPosterior;
pub const CapacityAllocator = bandit.CapacityAllocator;
pub const rngFromSeed = bandit.rngFromSeed;

pub const alloc_guard = @import("alloc_guard.zig");
pub const CountingAllocator = alloc_guard.CountingAllocator;
pub const expectNoAllocations = alloc_guard.expectNoAllocations;
pub const expectAtMostAllocations = alloc_guard.expectAtMostAllocations;

pub const wcet = @import("wcet.zig");
pub const BudgetedArm = wcet.BudgetedArm;
pub const Pick = wcet.Pick;
pub const pickWithinBudget = wcet.pickWithinBudget;
pub const anyArmFits = wcet.anyArmFits;
pub const totalBudgetUs = wcet.totalBudgetUs;

pub const pcg64 = @import("pcg64.zig");
pub const Pcg64 = pcg64.Pcg64;

pub const samplers = @import("samplers.zig");
pub const gaussian = samplers.gaussian;
pub const exponential = samplers.exponential;
pub const poisson = samplers.poisson;
pub const integrateMC = samplers.integrateMC;
pub const IntegrateMcResult = samplers.IntegrateMcResult;

pub const sliding_reward = @import("sliding_reward.zig");
pub const SlidingReward = sliding_reward.SlidingReward;
pub const SlidingRewardError = sliding_reward.Error;

pub const decaying_reward = @import("decaying_reward.zig");
pub const DecayingReward = decaying_reward.DecayingReward;
pub const DecayingRewardError = decaying_reward.Error;

pub const per_arm_sliding = @import("per_arm_sliding.zig");
pub const PerArmSlidingStore = per_arm_sliding.PerArmSlidingStore;
pub const ArmId = per_arm_sliding.ArmId;
pub const PerArmSlidingError = per_arm_sliding.Error;

test {
    @import("std").testing.refAllDecls(@This());
}
