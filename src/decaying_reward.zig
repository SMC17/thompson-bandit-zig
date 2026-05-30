//! decaying_reward: v0.0.9 — exponentially-decayed reward tracker.
//!
//! Companion to v0.0.8 SlidingReward (fixed-window). The
//! exponentially-decayed-Bernoulli variant gives each reward an
//! age-weighted contribution: recent rewards count fully, older
//! rewards fade with a caller-supplied half-life. Constant
//! memory (single accumulator + sample count); no buffer of
//! historical rewards.
//!
//! Scope of this revision:
//!   DecayingReward { decay (per-update multiplier in (0, 1]),
//!     numerator (weighted reward sum), denominator (weighted
//!     count) }.
//!   init(decay) errors on decay outside (0, 1].
//!   record(reward) folds in one sample, decaying the prior
//!     accumulators first.
//!   meanReward() returns numerator/denominator (0 when no
//!     samples).
//!
//! Out of scope: rolling decay-window resets, per-arm decaying
//! posterior (caller composes one DecayingReward per arm), time-
//! based decay (caller scales the decay factor by elapsed dt at
//! record time).

const std = @import("std");

pub const Error = error{
    InvalidDecay,
};

pub const DecayingReward = struct {
    decay: f64,
    numerator: f64,
    denominator: f64,
    n: u64,

    pub fn init(decay: f64) Error!DecayingReward {
        if (!std.math.isFinite(decay) or decay <= 0.0 or decay > 1.0) return Error.InvalidDecay;
        return .{
            .decay = decay,
            .numerator = 0.0,
            .denominator = 0.0,
            .n = 0,
        };
    }

    pub fn record(self: *DecayingReward, reward: f64) void {
        if (!std.math.isFinite(reward)) return;
        self.numerator = self.numerator * self.decay + reward;
        self.denominator = self.denominator * self.decay + 1.0;
        self.n += 1;
    }

    pub fn meanReward(self: DecayingReward) f64 {
        if (self.denominator <= 0.0) return 0.0;
        return self.numerator / self.denominator;
    }

    pub fn nSamples(self: DecayingReward) u64 {
        return self.n;
    }

    pub fn reset(self: *DecayingReward) void {
        self.numerator = 0.0;
        self.denominator = 0.0;
        self.n = 0;
    }
};

// --- Tests -----------------------------------------------------------------

const testing = std.testing;

test "init refuses decay outside (0, 1]" {
    try testing.expectError(Error.InvalidDecay, DecayingReward.init(0.0));
    try testing.expectError(Error.InvalidDecay, DecayingReward.init(-0.5));
    try testing.expectError(Error.InvalidDecay, DecayingReward.init(1.5));
    try testing.expectError(Error.InvalidDecay, DecayingReward.init(std.math.nan(f64)));
}

test "init accepts decay = 1.0 (no decay) and small decay" {
    _ = try DecayingReward.init(1.0);
    _ = try DecayingReward.init(0.001);
}

test "empty tracker reports mean 0 and n 0" {
    var d = try DecayingReward.init(0.9);
    try testing.expectEqual(@as(f64, 0.0), d.meanReward());
    try testing.expectEqual(@as(u64, 0), d.nSamples());
}

test "single sample returns the sample" {
    var d = try DecayingReward.init(0.9);
    d.record(0.7);
    try testing.expectApproxEqAbs(@as(f64, 0.7), d.meanReward(), 1e-12);
}

test "constant stream yields constant mean" {
    var d = try DecayingReward.init(0.9);
    var i: usize = 0;
    while (i < 100) : (i += 1) d.record(0.5);
    try testing.expectApproxEqAbs(@as(f64, 0.5), d.meanReward(), 1e-9);
}

test "recent samples weight more than older ones (decay 0.5)" {
    var d = try DecayingReward.init(0.5);
    d.record(1.0);
    d.record(1.0);
    d.record(0.0);
    const m = d.meanReward();
    try testing.expect(m < 0.5);
}

test "decay 1.0 reduces to running mean" {
    var d = try DecayingReward.init(1.0);
    d.record(1.0);
    d.record(0.0);
    d.record(1.0);
    d.record(0.0);
    try testing.expectApproxEqAbs(@as(f64, 0.5), d.meanReward(), 1e-12);
}

test "reset zeros accumulators and sample count" {
    var d = try DecayingReward.init(0.9);
    d.record(1.0);
    d.record(0.5);
    d.reset();
    try testing.expectEqual(@as(f64, 0.0), d.meanReward());
    try testing.expectEqual(@as(u64, 0), d.nSamples());
}

test "non-finite reward is dropped" {
    var d = try DecayingReward.init(0.9);
    d.record(1.0);
    const n_before = d.nSamples();
    d.record(std.math.nan(f64));
    d.record(std.math.inf(f64));
    try testing.expectEqual(n_before, d.nSamples());
}
