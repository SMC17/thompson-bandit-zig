//! per_arm_sliding: v0.0.10 — per-arm SlidingReward registry.
//!
//! v0.0.8 SlidingReward tracks one arm's sliding window. Real
//! bandits have N arms each with their own window. Port a small
//! registry keyed on ArmId so callers iterate "current mean
//! reward per arm in the last window."
//!
//! Scope of this revision:
//!   ArmId = u32.
//!   PerArmSlidingStore { allocator, by_arm: ArmId ->
//!     SlidingReward, window_size }.
//!   init(allocator, window_size) errors on window_size 0.
//!   record(arm_id, reward) auto-creates the arm's reward
//!     tracker on first sight and folds in the sample.
//!   meanReward(arm_id) returns the arm's current mean (0 for
//!     unknown arms).
//!   bestArm() returns the arm with the highest mean among
//!     known arms; null for empty store.
//!
//! Out of scope: per-arm window size (uniform across all arms;
//! caller composes one PerArmSlidingStore per tier if needed),
//! Thompson sampling against these means (caller composes with
//! ArmPosterior).

const std = @import("std");
const sliding_reward = @import("sliding_reward.zig");

pub const SlidingReward = sliding_reward.SlidingReward;
pub const ArmId = u32;

pub const Error = error{
    OutOfMemory,
    InvalidWindow,
};

pub const PerArmSlidingStore = struct {
    allocator: std.mem.Allocator,
    by_arm: std.AutoHashMap(ArmId, SlidingReward),
    window_size: usize,

    pub fn init(allocator: std.mem.Allocator, window_size: usize) Error!PerArmSlidingStore {
        if (window_size == 0) return Error.InvalidWindow;
        return .{
            .allocator = allocator,
            .by_arm = std.AutoHashMap(ArmId, SlidingReward).init(allocator),
            .window_size = window_size,
        };
    }

    pub fn deinit(self: *PerArmSlidingStore) void {
        var it = self.by_arm.valueIterator();
        while (it.next()) |r| r.deinit();
        self.by_arm.deinit();
    }

    pub fn record(self: *PerArmSlidingStore, arm_id: ArmId, reward: u8) Error!void {
        const entry = self.by_arm.getPtr(arm_id);
        if (entry) |e| {
            e.record(reward);
            return;
        }
        var fresh = SlidingReward.init(self.allocator, self.window_size) catch return Error.OutOfMemory;
        fresh.record(reward);
        self.by_arm.put(arm_id, fresh) catch {
            fresh.deinit();
            return Error.OutOfMemory;
        };
    }

    pub fn meanReward(self: *const PerArmSlidingStore, arm_id: ArmId) f64 {
        const entry = self.by_arm.get(arm_id) orelse return 0.0;
        return entry.meanReward();
    }

    pub fn nKnownArms(self: PerArmSlidingStore) usize {
        return self.by_arm.count();
    }

    pub fn bestArm(self: *const PerArmSlidingStore) ?ArmId {
        var best_id: ?ArmId = null;
        var best_mean: f64 = -1.0;
        var it = self.by_arm.iterator();
        while (it.next()) |kv| {
            const m = kv.value_ptr.*.meanReward();
            if (m > best_mean) {
                best_mean = m;
                best_id = kv.key_ptr.*;
            }
        }
        return best_id;
    }
};

// --- Tests -----------------------------------------------------------------

const testing = std.testing;

test "init refuses window_size 0" {
    try testing.expectError(Error.InvalidWindow, PerArmSlidingStore.init(testing.allocator, 0));
}

test "record auto-creates arm on first sight" {
    var s: PerArmSlidingStore = try .init(testing.allocator, 5);
    defer s.deinit();
    try testing.expectEqual(@as(usize, 0), s.nKnownArms());
    try s.record(1, 1);
    try testing.expectEqual(@as(usize, 1), s.nKnownArms());
}

test "meanReward on unknown arm is 0" {
    var s: PerArmSlidingStore = try .init(testing.allocator, 5);
    defer s.deinit();
    try testing.expectEqual(@as(f64, 0.0), s.meanReward(99));
}

test "two arms tracked independently" {
    var s: PerArmSlidingStore = try .init(testing.allocator, 5);
    defer s.deinit();
    try s.record(1, 1);
    try s.record(1, 1);
    try s.record(2, 0);
    try s.record(2, 0);
    try testing.expectApproxEqAbs(@as(f64, 1.0), s.meanReward(1), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), s.meanReward(2), 1e-12);
}

test "bestArm returns highest-mean arm" {
    var s: PerArmSlidingStore = try .init(testing.allocator, 5);
    defer s.deinit();
    try testing.expectEqual(@as(?ArmId, null), s.bestArm());
    try s.record(1, 0);
    try s.record(2, 1);
    try s.record(3, 1);
    try s.record(3, 1);
    try testing.expectEqual(@as(?ArmId, 3), s.bestArm());
}

test "window-size 1 keeps only most recent per-arm" {
    var s: PerArmSlidingStore = try .init(testing.allocator, 1);
    defer s.deinit();
    try s.record(1, 1);
    try s.record(1, 0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), s.meanReward(1), 1e-12);
}
