//! wcet: v0.0.5 substrate from L91 fathom-fc.
//!
//! L91 mining of mac:~/drones/firmware/fathom-fc/crate.spec.yaml
//! pulled six WCET budgets verbatim from production source:
//!
//!   imu_read_pair      200 us at 1000 Hz
//!   rate_controller    150 us at 1000 Hz
//!   attitude_controller 200 us at 250 Hz
//!   allocation_quad_x   50 us at 1000 Hz
//!   gnss_decode       5000 us at  5 Hz
//!   mavlink_uart_tx    500 us at 50 Hz
//!
//! The fathom-fc orchestrator refuses to dispatch a task whose
//! budget would not fit the remaining cycle slice. This is the
//! "arms with hard ceilings" shape from L82 P1492. The existing
//! thompson-bandit primitive picks the arm with the highest
//! sampled mean and ignores time. This module ports the budget
//! gate: pick the highest-sampled arm WHOSE budget fits the
//! remaining time slice, else fail closed.
//!
//! Scope of this revision:
//!   BudgetedArm wraps an ArmPosterior with a budget_us cap.
//!   pickWithinBudget(arms, remaining_us, rng) returns the
//!     highest-sample arm whose budget fits, or null if none fit.
//!   anyArmFits(arms, remaining_us) inspection helper for callers
//!     who want to surface budget exhaustion as their own error.
//!
//! Out of scope: deadline-driven rebudgeting (caller policy), per-
//! arm budget tracking across calls (caller can wrap), measured-
//! vs-budgeted reconciliation (caller composes with the existing
//! recordSuccess / recordFailure surface).

const std = @import("std");
const bandit = @import("bandit.zig");

pub const ArmPosterior = bandit.ArmPosterior;

pub const BudgetedArm = struct {
    posterior: ArmPosterior,
    budget_us: u32,
};

pub const Pick = struct {
    arm_index: usize,
    sampled: f64,
};

/// Returns the highest-sampled arm whose budget_us is less than or
/// equal to remaining_us, or null if no arm fits.
pub fn pickWithinBudget(
    arms: []const BudgetedArm,
    remaining_us: u32,
    rng: *std.Random.DefaultPrng,
) ?Pick {
    var best: ?Pick = null;
    for (arms, 0..) |a, i| {
        if (a.budget_us > remaining_us) continue;
        const s = a.posterior.sample(rng);
        if (best == null or s > best.?.sampled) {
            best = .{ .arm_index = i, .sampled = s };
        }
    }
    return best;
}

/// True if any arm in the slice has a budget that fits remaining_us.
/// Callers compose this with their own deadline policy.
pub fn anyArmFits(arms: []const BudgetedArm, remaining_us: u32) bool {
    for (arms) |a| {
        if (a.budget_us <= remaining_us) return true;
    }
    return false;
}

/// Total budget across a slice of arms. Useful for sanity-checking
/// a slate against a worst-case dispatch.
pub fn totalBudgetUs(arms: []const BudgetedArm) u64 {
    var total: u64 = 0;
    for (arms) |a| total += a.budget_us;
    return total;
}

// --- Tests -----------------------------------------------------------------

const testing = std.testing;

test "pickWithinBudget returns the only fitting arm" {
    const arms = [_]BudgetedArm{
        .{ .posterior = .uniform(), .budget_us = 5000 },
        .{ .posterior = .uniform(), .budget_us = 200 },
    };
    var rng = bandit.rngFromSeed(1);
    const pick = pickWithinBudget(&arms, 300, &rng) orelse return error.TestExpectedPick;
    try testing.expectEqual(@as(usize, 1), pick.arm_index);
}

test "pickWithinBudget returns null when no arm fits" {
    const arms = [_]BudgetedArm{
        .{ .posterior = .uniform(), .budget_us = 1000 },
        .{ .posterior = .uniform(), .budget_us = 500 },
    };
    var rng = bandit.rngFromSeed(1);
    try testing.expectEqual(@as(?Pick, null), pickWithinBudget(&arms, 100, &rng));
}

test "pickWithinBudget picks from posterior when many arms fit" {
    // Seed two arms with different posteriors so the bandit prefers one.
    var hot: ArmPosterior = .uniform();
    var i: usize = 0;
    while (i < 100) : (i += 1) hot.recordSuccess();
    var cold: ArmPosterior = .uniform();
    i = 0;
    while (i < 100) : (i += 1) cold.recordFailure();
    const arms = [_]BudgetedArm{
        .{ .posterior = cold, .budget_us = 100 },
        .{ .posterior = hot, .budget_us = 100 },
    };
    var rng = bandit.rngFromSeed(42);
    var hot_picks: usize = 0;
    var trial: usize = 0;
    while (trial < 200) : (trial += 1) {
        const pick = pickWithinBudget(&arms, 200, &rng).?;
        if (pick.arm_index == 1) hot_picks += 1;
    }
    // hot arm dominates posterior, should get picked > 75% of the time
    try testing.expect(hot_picks > 150);
}

test "pickWithinBudget honors per-arm budget within the slate" {
    var hot: ArmPosterior = .uniform();
    var i: usize = 0;
    while (i < 50) : (i += 1) hot.recordSuccess();
    const arms = [_]BudgetedArm{
        .{ .posterior = hot, .budget_us = 5000 }, // hot arm too expensive
        .{ .posterior = .uniform(), .budget_us = 100 },
    };
    var rng = bandit.rngFromSeed(1);
    const pick = pickWithinBudget(&arms, 200, &rng).?;
    // Even though hot has higher posterior, it does not fit; arm 1 wins.
    try testing.expectEqual(@as(usize, 1), pick.arm_index);
}

test "anyArmFits true when at least one budget fits" {
    const arms = [_]BudgetedArm{
        .{ .posterior = .uniform(), .budget_us = 5000 },
        .{ .posterior = .uniform(), .budget_us = 100 },
    };
    try testing.expect(anyArmFits(&arms, 200));
    try testing.expect(!anyArmFits(&arms, 50));
}

test "anyArmFits on empty slice is false" {
    try testing.expect(!anyArmFits(&[_]BudgetedArm{}, 100));
}

test "totalBudgetUs sums every arm" {
    const arms = [_]BudgetedArm{
        .{ .posterior = .uniform(), .budget_us = 200 },
        .{ .posterior = .uniform(), .budget_us = 150 },
        .{ .posterior = .uniform(), .budget_us = 50 },
    };
    try testing.expectEqual(@as(u64, 400), totalBudgetUs(&arms));
}

test "pickWithinBudget at exact budget boundary still fits" {
    const arms = [_]BudgetedArm{
        .{ .posterior = .uniform(), .budget_us = 100 },
    };
    var rng = bandit.rngFromSeed(1);
    // budget_us = 100, remaining_us = 100 should fit
    const pick = pickWithinBudget(&arms, 100, &rng);
    try testing.expect(pick != null);
}

test "fathom-fc 1 kHz control loop slate fits the 1000us cycle" {
    // Verbatim budgets from L91 (mac:~/drones/firmware/fathom-fc/
    // crate.spec.yaml). At 1 kHz the cycle is 1000 us. The four
    // control-loop tasks should fit together.
    const arms = [_]BudgetedArm{
        .{ .posterior = .uniform(), .budget_us = 200 }, // imu_read_pair
        .{ .posterior = .uniform(), .budget_us = 150 }, // rate_controller
        .{ .posterior = .uniform(), .budget_us = 200 }, // attitude_controller
        .{ .posterior = .uniform(), .budget_us = 50 }, // allocation_quad_x
    };
    try testing.expectEqual(@as(u64, 600), totalBudgetUs(&arms));
    try testing.expect(anyArmFits(&arms, 1000));
    // Confirm: remaining_us = 50 only fits allocation_quad_x
    var rng = bandit.rngFromSeed(1);
    const pick = pickWithinBudget(&arms, 50, &rng).?;
    try testing.expectEqual(@as(usize, 3), pick.arm_index);
}
