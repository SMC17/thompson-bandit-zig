//! sliding_reward: v0.0.8 — sliding-window reward tracker.
//!
//! ArmPosterior (v0.0.1) integrates reward over the arm's full
//! lifetime; a noisy-but-stale arm and a fresh arm with the same
//! posterior look identical even when the noisy one's recent
//! reward is much lower. Production bandits want a sliding
//! window so the posterior reflects RECENT behavior.
//!
//! Port a circular-buffer-backed reward tracker. Per-arm fixed
//! window of binary reward samples; running sum + window_size
//! gives running mean. O(1) push (replaces oldest); O(1) mean.
//!
//! Scope of this revision:
//!   SlidingReward fixed-capacity window of binary rewards.
//!   init(allocator, window_size) errors on window_size == 0.
//!   record(reward: u32) appends 0 or 1; replaces oldest at
//!     capacity. Caller passes 0 for failure, 1 for success.
//!   meanReward() returns the current window mean (0.0 to 1.0).
//!   nSamples() returns the current sample count (<= window).
//!
//! Out of scope: non-binary rewards (caller can extend by
//! changing the buffer type), per-window-frame timestamping
//! (caller composes with a sibling ts buffer if needed),
//! exponential decay window (Thompson sampling literature has a
//! sliding-Bernoulli variant; this primitive is the fixed-window
//! companion).

const std = @import("std");

pub const Error = error{
    OutOfMemory,
    InvalidWindow,
};

pub const SlidingReward = struct {
    allocator: std.mem.Allocator,
    window: []u8,
    head: usize,
    n: usize,
    cap: usize,
    sum: u64,

    pub fn init(allocator: std.mem.Allocator, window_size: usize) Error!SlidingReward {
        if (window_size == 0) return Error.InvalidWindow;
        const w = allocator.alloc(u8, window_size) catch return Error.OutOfMemory;
        @memset(w, 0);
        return .{
            .allocator = allocator,
            .window = w,
            .head = 0,
            .n = 0,
            .cap = window_size,
            .sum = 0,
        };
    }

    pub fn deinit(self: *SlidingReward) void {
        self.allocator.free(self.window);
    }

    pub fn record(self: *SlidingReward, reward: u8) void {
        const r: u8 = if (reward != 0) 1 else 0;
        if (self.n < self.cap) {
            self.window[(self.head + self.n) % self.cap] = r;
            self.sum += r;
            self.n += 1;
        } else {
            const old = self.window[self.head];
            self.sum -= old;
            self.window[self.head] = r;
            self.sum += r;
            self.head = (self.head + 1) % self.cap;
        }
    }

    pub fn meanReward(self: SlidingReward) f64 {
        if (self.n == 0) return 0.0;
        return @as(f64, @floatFromInt(self.sum)) / @as(f64, @floatFromInt(self.n));
    }

    pub fn nSamples(self: SlidingReward) usize {
        return self.n;
    }

    pub fn isFull(self: SlidingReward) bool {
        return self.n == self.cap;
    }

    pub fn clear(self: *SlidingReward) void {
        @memset(self.window, 0);
        self.head = 0;
        self.n = 0;
        self.sum = 0;
    }
};

// --- Tests -----------------------------------------------------------------

const testing = std.testing;

test "init refuses window_size 0" {
    try testing.expectError(Error.InvalidWindow, SlidingReward.init(testing.allocator, 0));
}

test "empty window has mean 0 and 0 samples" {
    var r = try SlidingReward.init(testing.allocator, 10);
    defer r.deinit();
    try testing.expectEqual(@as(f64, 0.0), r.meanReward());
    try testing.expectEqual(@as(usize, 0), r.nSamples());
}

test "record clamps non-zero to 1" {
    var r = try SlidingReward.init(testing.allocator, 5);
    defer r.deinit();
    r.record(99);
    try testing.expectEqual(@as(f64, 1.0), r.meanReward());
}

test "mean reward over five samples" {
    var r = try SlidingReward.init(testing.allocator, 5);
    defer r.deinit();
    r.record(1);
    r.record(0);
    r.record(1);
    r.record(1);
    r.record(0);
    try testing.expectApproxEqAbs(@as(f64, 0.6), r.meanReward(), 1e-12);
}

test "wrap-around drops oldest and updates running sum" {
    var r = try SlidingReward.init(testing.allocator, 3);
    defer r.deinit();
    r.record(1);
    r.record(1);
    r.record(1);
    try testing.expectApproxEqAbs(@as(f64, 1.0), r.meanReward(), 1e-12);
    r.record(0);
    try testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), r.meanReward(), 1e-12);
    r.record(0);
    try testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), r.meanReward(), 1e-12);
    r.record(0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), r.meanReward(), 1e-12);
}

test "isFull transitions at capacity" {
    var r = try SlidingReward.init(testing.allocator, 3);
    defer r.deinit();
    try testing.expect(!r.isFull());
    r.record(1);
    r.record(1);
    try testing.expect(!r.isFull());
    r.record(1);
    try testing.expect(r.isFull());
}

test "clear resets to empty" {
    var r = try SlidingReward.init(testing.allocator, 5);
    defer r.deinit();
    r.record(1);
    r.record(1);
    r.clear();
    try testing.expectEqual(@as(f64, 0.0), r.meanReward());
    try testing.expectEqual(@as(usize, 0), r.nSamples());
    try testing.expect(!r.isFull());
}

test "window_size 1 is a degenerate but valid latest-reward tracker" {
    var r = try SlidingReward.init(testing.allocator, 1);
    defer r.deinit();
    r.record(1);
    try testing.expectApproxEqAbs(@as(f64, 1.0), r.meanReward(), 1e-12);
    r.record(0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), r.meanReward(), 1e-12);
    r.record(1);
    try testing.expectApproxEqAbs(@as(f64, 1.0), r.meanReward(), 1e-12);
}
