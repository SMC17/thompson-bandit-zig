//! pcg64: v0.0.6 substrate from L113 Aletheia zmonte.
//!
//! zmonte exposes two named generators (Pcg64 and Mt19937_64) and
//! guarantees bit-exact reproducibility across runs. thompson-
//! bandit-zig currently relies on std.Random.DefaultPrng (Wyhash)
//! via rngFromSeed; the bandit logic does not care which PRNG
//! sits underneath, but downstream consumers that need
//! cross-language reproducibility (Python/Rust reference samplers)
//! benefit from a named generator with a stable algorithm.
//!
//! This module ships PCG-XSH-RR 64/32 as a stand-alone named
//! generator. Caller composes it with rngFromSeed only when the
//! cross-language reproducibility property matters; the existing
//! DefaultPrng path remains the default for in-process work.
//!
//! Algorithm: PCG-XSH-RR 64/32 per O'Neill (2014). 64-bit linear
//! congruential state; output is 32 bits formed by XOR-shifting
//! the state and then a random-rotate. Multiplier and increment
//! follow the PCG reference defaults.
//!
//! Scope of this revision:
//!   Pcg64 record { state, inc }. init(seed, seq) seeds the
//!     generator with the canonical two-step (state=0, advance,
//!     state += seed, advance).
//!   nextU32 returns the next 32-bit output and advances state.
//!   nextF64 returns a uniform [0, 1) by drawing two u32s and
//!     forming a 53-bit mantissa.
//!   advance() advances state without returning output; used for
//!     bit-exact stream-skipping when reproducing recorded runs.
//!
//! Out of scope: jump-ahead by arbitrary delta (zmonte may ship
//! this in a later mine), MT19937-64 (separate primitive if
//! needed), distribution samplers (caller composes with std.Random
//! or with zmonte's sampler surface once ported).
//!
//! Citation: M. E. O'Neill, "PCG: A Family of Simple Fast Space-
//! Efficient Statistically Good Algorithms for Random Number
//! Generation," Harvey Mudd College Tech Report HMC-CS-2014-0905
//! (2014).

const std = @import("std");

const PCG_MULT: u64 = 6364136223846793005;
const PCG_INC_DEFAULT: u64 = 1442695040888963407;

pub const Pcg64 = struct {
    state: u64,
    inc: u64,

    /// Canonical PCG seed routine. seq must be unique per
    /// independent stream; default seq = 1 is fine for a single
    /// stream.
    pub fn init(seed: u64, seq: u64) Pcg64 {
        var g: Pcg64 = .{ .state = 0, .inc = (seq << 1) | 1 };
        _ = g.nextU32();
        g.state +%= seed;
        _ = g.nextU32();
        return g;
    }

    pub fn initDefault(seed: u64) Pcg64 {
        return init(seed, PCG_INC_DEFAULT);
    }

    pub fn nextU32(self: *Pcg64) u32 {
        const oldstate = self.state;
        self.state = oldstate *% PCG_MULT +% self.inc;
        const xorshifted: u32 = @truncate(((oldstate >> 18) ^ oldstate) >> 27);
        const rot: u5 = @truncate(oldstate >> 59);
        return std.math.rotr(u32, xorshifted, @as(u32, rot));
    }

    /// Uniform [0, 1) via 53-bit mantissa concatenation. Drops
    /// the low 5 bits of the first draw and the low 6 bits of
    /// the second to fit IEEE-754 double precision.
    pub fn nextF64(self: *Pcg64) f64 {
        const hi: u64 = @as(u64, self.nextU32() >> 5);
        const lo: u64 = @as(u64, self.nextU32() >> 6);
        const m: u64 = (hi << 26) | lo;
        return @as(f64, @floatFromInt(m)) * (1.0 / @as(f64, @floatFromInt(@as(u64, 1) << 53)));
    }

    pub fn advance(self: *Pcg64) void {
        _ = self.nextU32();
    }
};

// --- Tests -----------------------------------------------------------------

const testing = std.testing;

test "Pcg64.init produces deterministic stream from a given seed" {
    var a: Pcg64 = .initDefault(42);
    var b: Pcg64 = .initDefault(42);
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        try testing.expectEqual(a.nextU32(), b.nextU32());
    }
}

test "Different seeds produce different streams" {
    var a: Pcg64 = .initDefault(1);
    var b: Pcg64 = .initDefault(2);
    var any_diff = false;
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        if (a.nextU32() != b.nextU32()) {
            any_diff = true;
            break;
        }
    }
    try testing.expect(any_diff);
}

test "Different seq values produce different streams from same seed" {
    var a: Pcg64 = .init(42, 1);
    var b: Pcg64 = .init(42, 99);
    var any_diff = false;
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        if (a.nextU32() != b.nextU32()) {
            any_diff = true;
            break;
        }
    }
    try testing.expect(any_diff);
}

test "nextF64 stays in [0, 1)" {
    var g: Pcg64 = .initDefault(7);
    var i: usize = 0;
    while (i < 10_000) : (i += 1) {
        const f = g.nextF64();
        try testing.expect(f >= 0.0);
        try testing.expect(f < 1.0);
    }
}

test "nextF64 mean over 10000 draws is roughly 0.5" {
    var g: Pcg64 = .initDefault(123);
    var sum: f64 = 0.0;
    const n: usize = 10_000;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        sum += g.nextF64();
    }
    const mean = sum / @as(f64, @floatFromInt(n));
    try testing.expect(mean > 0.45);
    try testing.expect(mean < 0.55);
}

test "advance skips one draw, matching subsequent stream alignment" {
    var a: Pcg64 = .initDefault(99);
    var b: Pcg64 = .initDefault(99);
    a.advance();
    _ = b.nextU32();
    try testing.expectEqual(a.nextU32(), b.nextU32());
    try testing.expectEqual(a.nextU32(), b.nextU32());
}

test "Pcg64 state and inc are public for snapshot-restore" {
    var g: Pcg64 = .initDefault(11);
    _ = g.nextU32();
    const snap_state = g.state;
    const snap_inc = g.inc;
    const next_after_snap = g.nextU32();
    var g2: Pcg64 = .{ .state = snap_state, .inc = snap_inc };
    try testing.expectEqual(next_after_snap, g2.nextU32());
}
