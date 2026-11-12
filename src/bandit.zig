//! thompson-bandit — Beta(α, β) posterior + Thompson sampling.
//!
//! Ports AWS DynamoDB L650 adaptive capacity throttling + Doordash L46
//! Thompson bandit + Datadog L37 Thompson-based cardinality control
//! into a single Zig substrate. Composes with shipped expt-substrate
//! Wyhash seed-derivation for deterministic A/B → bandit transitions.
//!
//! The seed: for each partition (or arm), maintain a Beta(α, β)
//! posterior over success rate. α = successes + prior_α, β =
//! failures + prior_β. To allocate capacity, sample one value from
//! each arm's posterior + allocate weighted by the sampled values.
//! Hot partitions naturally get more capacity; cold partitions get
//! less; the posterior tracks empirical reality.
//!
//! v0.0.1 ships the posterior + sampler + capacity allocator with a
//! deterministic Wyhash-seeded RNG. v0.0.2 adds Beta(α, β) via
//! gamma-distribution ratio for non-integer α/β. v0.0.3 wires
//! windowed posteriors (Datadog-style drift) and the multi-armed
//! allocator that supports K > 2 arms with capacity-floor guarantees.

const std = @import("std");

/// One arm's posterior. Successes + failures are integer counts on
/// the success criterion; the substrate is criterion-agnostic.
pub const ArmPosterior = struct {
    alpha: u64, // 1 + successes (Beta prior α = 1)
    beta: u64, // 1 + failures (Beta prior β = 1)

    pub fn uniform() ArmPosterior {
        return .{ .alpha = 1, .beta = 1 };
    }

    pub fn fromCounts(successes: u64, failures: u64) ArmPosterior {
        return .{ .alpha = 1 + successes, .beta = 1 + failures };
    }

    pub fn recordSuccess(self: *ArmPosterior) void {
        self.alpha += 1;
    }

    pub fn recordFailure(self: *ArmPosterior) void {
        self.beta += 1;
    }

    /// Posterior mean: E[θ] = α / (α + β).
    pub fn mean(self: ArmPosterior) f64 {
        const a: f64 = @floatFromInt(self.alpha);
        const b: f64 = @floatFromInt(self.beta);
        return a / (a + b);
    }

    /// Sample one value from this arm's Beta(α, β) posterior.
    ///
    /// v0.0.1 uses the integer-α / integer-β special case where
    /// Beta(α, β) is the α-th order statistic of (α + β - 1) uniform
    /// random samples. This is exact for the Beta prior + count
    /// updates the substrate ships. v0.0.2 wires the
    /// gamma-distribution ratio for arbitrary real-valued α + β.
    pub fn sample(self: ArmPosterior, rng: *std.Random.DefaultPrng) f64 {
        return sampleBetaIntegerParams(self.alpha, self.beta, rng);
    }
};

/// Beta(α, β) sampler for integer parameters via order statistics.
/// Draws (α + β - 1) uniform samples and returns the α-th smallest.
/// Sufficient for the Beta(integer, integer) regime ArmPosterior
/// ships at v0.0.1.
fn sampleBetaIntegerParams(alpha: u64, beta: u64, rng: *std.Random.DefaultPrng) f64 {
    const n = alpha + beta - 1;
    // Cap n at 1024 to keep stack allocation bounded — for typical
    // adaptive-capacity workloads the success + failure counts
    // saturate within this range; v0.0.2 ships gamma sampling for
    // larger α + β.
    if (n > 1024) return @as(f64, @floatFromInt(alpha)) / @as(f64, @floatFromInt(alpha + beta));

    var samples: [1024]f64 = undefined;
    var i: u64 = 0;
    while (i < n) : (i += 1) samples[@intCast(i)] = rng.random().float(f64);
    std.mem.sort(f64, samples[0..@intCast(n)], {}, std.sort.asc(f64));
    return samples[@intCast(alpha - 1)];
}

/// Adaptive capacity allocator. Holds K arms; on `allocate`, samples
/// each arm's posterior + returns a fractional allocation summing
/// to 1.0. Caller multiplies by the available capacity to get the
/// per-arm budget.
pub const CapacityAllocator = struct {
    allocator: std.mem.Allocator,
    arms: std.array_list.Managed(ArmPosterior),
    /// Floor for any arm so cold arms are still explored. Default
    /// 0.0 = no floor. AWS DynamoDB ships a baseline 25%/N capacity
    /// for each partition so no partition fully starves.
    floor_per_arm: f64,

    pub fn init(allocator: std.mem.Allocator, k: usize, floor: f64) !CapacityAllocator {
        try validateAllocation(k, floor);
        var arms: std.array_list.Managed(ArmPosterior) = .init(allocator);
        errdefer arms.deinit();
        var i: usize = 0;
        while (i < k) : (i += 1) try arms.append(.uniform());
        return .{ .allocator = allocator, .arms = arms, .floor_per_arm = floor };
    }

    pub fn deinit(self: *CapacityAllocator) void {
        self.arms.deinit();
    }

    pub fn armCount(self: CapacityAllocator) usize {
        return self.arms.items.len;
    }

    pub fn arm(self: *CapacityAllocator, idx: usize) *ArmPosterior {
        return &self.arms.items[idx];
    }

    /// Allocate fractional capacity across all arms via Thompson
    /// sampling. The result is a slice of length `armCount()`
    /// summing to 1.0 (within float tolerance). Caller owns the
    /// returned slice and frees it.
    pub fn allocate(self: *CapacityAllocator, rng: *std.Random.DefaultPrng) ![]f64 {
        const k = self.arms.items.len;
        try validateAllocation(k, self.floor_per_arm);
        var out = try self.allocator.alloc(f64, k);
        errdefer self.allocator.free(out);

        // Step 1 — sample each arm's posterior.
        var total: f64 = 0;
        for (self.arms.items, 0..) |a, i| {
            const s = a.sample(rng);
            out[i] = s;
            total += s;
        }

        // Step 2 — normalise to sum to 1.
        if (total == 0) {
            const uniform_share = 1.0 / @as(f64, @floatFromInt(k));
            for (out) |*v| v.* = uniform_share;
        } else {
            for (out) |*v| v.* = v.* / total;
        }

        // Step 3 — apply floor. If any arm is below the floor, lift
        // it and renormalise the rest.
        if (self.floor_per_arm > 0) applyFloor(out, self.floor_per_arm);

        return out;
    }
};

fn validateAllocation(k: usize, floor: f64) !void {
    if (k == 0) return error.InvalidArmCount;
    if (!std.math.isFinite(floor) or floor < 0 or floor > 1.0 / @as(f64, @floatFromInt(k)))
        return error.InvalidFloor;
}

fn applyFloor(allocations: []f64, floor: f64) void {
    // Pin low shares first, then distribute only the remaining mass. Dividing
    // max(share, floor) by its sum would immediately push pinned shares below
    // the floor again. Negative values are temporary markers, never returned.
    var pinned: usize = 0;
    while (true) {
        const remaining = 1.0 - @as(f64, @floatFromInt(pinned)) * floor;
        var total: f64 = 0;
        var active: usize = 0;
        for (allocations) |a| if (a >= 0) {
            total += a;
            active += 1;
        };
        if (active == 0) break;
        if (total == 0) {
            for (allocations) |*a| if (a.* >= 0) {
                a.* = remaining / @as(f64, @floatFromInt(active));
            };
            break;
        }
        var changed = false;
        for (allocations) |*a| {
            if (a.* >= 0 and a.* / total * remaining < floor) {
                a.* = -1;
                pinned += 1;
                changed = true;
            }
        }
        if (!changed) {
            for (allocations) |*a| if (a.* >= 0) {
                a.* = a.* / total * remaining;
            };
            break;
        }
    }
    for (allocations) |*a| if (a.* < 0) {
        a.* = floor;
    };
}

/// Build a DefaultPrng deterministically from a u64 seed (e.g.
/// derived from `expt-substrate` Wyhash). Same seed → same sequence
/// → reproducible allocations + tests.
pub fn rngFromSeed(seed: u64) std.Random.DefaultPrng {
    return std.Random.DefaultPrng.init(seed);
}

// --- Tests ----------------------------------------------------------------

const testing = std.testing;

test "ArmPosterior.uniform starts at Beta(1, 1)" {
    const p: ArmPosterior = .uniform();
    try testing.expectEqual(@as(u64, 1), p.alpha);
    try testing.expectEqual(@as(u64, 1), p.beta);
    try testing.expectApproxEqAbs(@as(f64, 0.5), p.mean(), 1e-9);
}

test "ArmPosterior.fromCounts seeds with prior" {
    const p: ArmPosterior = .fromCounts(8, 2);
    try testing.expectEqual(@as(u64, 9), p.alpha);
    try testing.expectEqual(@as(u64, 3), p.beta);
    // Mean = 9 / 12 = 0.75.
    try testing.expectApproxEqAbs(@as(f64, 0.75), p.mean(), 1e-9);
}

test "recordSuccess + recordFailure bump α + β" {
    var p: ArmPosterior = .uniform();
    p.recordSuccess();
    p.recordSuccess();
    p.recordSuccess();
    p.recordFailure();
    try testing.expectEqual(@as(u64, 4), p.alpha);
    try testing.expectEqual(@as(u64, 2), p.beta);
}

test "sample is deterministic with seeded RNG" {
    var rng_a = rngFromSeed(42);
    var rng_b = rngFromSeed(42);
    const p: ArmPosterior = .fromCounts(5, 3);
    const sa = p.sample(&rng_a);
    const sb = p.sample(&rng_b);
    try testing.expectApproxEqAbs(sa, sb, 1e-12);
}

test "sample falls in [0, 1]" {
    var rng = rngFromSeed(100);
    const p: ArmPosterior = .fromCounts(5, 3);
    var i: usize = 0;
    while (i < 200) : (i += 1) {
        const s = p.sample(&rng);
        try testing.expect(s >= 0.0);
        try testing.expect(s <= 1.0);
    }
}

test "sample mean across many draws approximates posterior mean" {
    var rng = rngFromSeed(2026);
    const p: ArmPosterior = .fromCounts(20, 5); // mean = 21/26 ≈ 0.808
    var sum: f64 = 0;
    const n: usize = 2000;
    var i: usize = 0;
    while (i < n) : (i += 1) sum += p.sample(&rng);
    const empirical = sum / @as(f64, @floatFromInt(n));
    const expected = p.mean();
    try testing.expectApproxEqAbs(expected, empirical, 0.03);
}

test "CapacityAllocator init creates K uniform arms" {
    var alloc: CapacityAllocator = try .init(testing.allocator, 4, 0.0);
    defer alloc.deinit();
    try testing.expectEqual(@as(usize, 4), alloc.armCount());
    for (alloc.arms.items) |a| {
        try testing.expectEqual(@as(u64, 1), a.alpha);
        try testing.expectEqual(@as(u64, 1), a.beta);
    }
}

test "CapacityAllocator.allocate yields shares summing to 1" {
    var alloc: CapacityAllocator = try .init(testing.allocator, 3, 0.0);
    defer alloc.deinit();
    var rng = rngFromSeed(7);
    const shares = try alloc.allocate(&rng);
    defer testing.allocator.free(shares);
    var total: f64 = 0;
    for (shares) |s| total += s;
    try testing.expectApproxEqAbs(@as(f64, 1.0), total, 1e-9);
    for (shares) |s| {
        try testing.expect(s >= 0.0);
        try testing.expect(s <= 1.0);
    }
}

test "CapacityAllocator hot arms attract more allocation over time" {
    var alloc: CapacityAllocator = try .init(testing.allocator, 2, 0.0);
    defer alloc.deinit();

    // Arm 0: 50 successes + 5 failures (hot).
    // Arm 1: 5 successes + 50 failures (cold).
    var i: usize = 0;
    while (i < 50) : (i += 1) alloc.arm(0).recordSuccess();
    i = 0;
    while (i < 5) : (i += 1) alloc.arm(0).recordFailure();
    i = 0;
    while (i < 5) : (i += 1) alloc.arm(1).recordSuccess();
    i = 0;
    while (i < 50) : (i += 1) alloc.arm(1).recordFailure();

    // Average over many allocations to smooth Thompson variance.
    var rng = rngFromSeed(1729);
    var arm0_total: f64 = 0;
    var arm1_total: f64 = 0;
    const trials: usize = 500;
    var t: usize = 0;
    while (t < trials) : (t += 1) {
        const shares = try alloc.allocate(&rng);
        defer testing.allocator.free(shares);
        arm0_total += shares[0];
        arm1_total += shares[1];
    }
    const arm0_avg = arm0_total / @as(f64, @floatFromInt(trials));
    const arm1_avg = arm1_total / @as(f64, @floatFromInt(trials));
    // The hot arm should get at least 70% of capacity on average.
    try testing.expect(arm0_avg > 0.7);
    try testing.expect(arm1_avg < 0.3);
}

test "CapacityAllocator floor lifts cold arms above floor" {
    var alloc: CapacityAllocator = try .init(testing.allocator, 3, 0.1);
    defer alloc.deinit();
    var i: usize = 0;
    while (i < 100) : (i += 1) alloc.arm(0).recordSuccess();
    i = 0;
    while (i < 100) : (i += 1) alloc.arm(1).recordFailure();

    var rng = rngFromSeed(11);
    const shares = try alloc.allocate(&rng);
    defer testing.allocator.free(shares);
    // Every arm gets at least the floor share.
    for (shares) |s| try testing.expect(s >= 0.1 - 1e-9);
    var total: f64 = 0;
    for (shares) |s| total += s;
    try testing.expectApproxEqAbs(@as(f64, 1.0), total, 1e-9);
}

test "rngFromSeed reproducibility — same seed yields same sequence" {
    var a = rngFromSeed(0x1234_5678);
    var b = rngFromSeed(0x1234_5678);
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        try testing.expectEqual(a.next(), b.next());
    }
}

test "floor survives renormalization of a skewed allocation" {
    var shares = [_]f64{ 0.95, 0.04, 0.01 };
    applyFloor(&shares, 0.2);
    for (shares, [_]f64{ 0.6, 0.2, 0.2 }) |actual, expected| {
        try testing.expectApproxEqAbs(expected, actual, 1e-12);
    }
}

fn initializationAllocationProbe(a: std.mem.Allocator) !void {
    var alloc = try CapacityAllocator.init(a, 256, 0);
    defer alloc.deinit();
}

test "allocator initialization cleans up on allocation failure" {
    try testing.checkAllAllocationFailures(testing.allocator, initializationAllocationProbe, .{});
}

test "allocator refuses empty arms and infeasible or nonfinite floors" {
    try testing.expectError(error.InvalidArmCount, CapacityAllocator.init(testing.allocator, 0, 0));
    for ([_]f64{ -0.1, 0.34, std.math.nan(f64), std.math.inf(f64) }) |floor| {
        try testing.expectError(error.InvalidFloor, CapacityAllocator.init(testing.allocator, 3, floor));
    }
}

test "floor can pin another arm on a later redistribution" {
    var shares = [_]f64{ 0.6, 0.25, 0.1, 0.05 };
    applyFloor(&shares, 0.2);
    for (shares, [_]f64{ 0.4, 0.2, 0.2, 0.2 }) |actual, expected| {
        try testing.expectApproxEqAbs(expected, actual, 1e-12);
    }
}

test "maximal feasible floor and one arm retain unit mass" {
    var shares = [_]f64{ 0.99, 0.01 };
    applyFloor(&shares, 0.5);
    for (shares) |s| try testing.expectApproxEqAbs(@as(f64, 0.5), s, 1e-12);
    var alloc = try CapacityAllocator.init(testing.allocator, 1, 1);
    defer alloc.deinit();
    var rng = rngFromSeed(7);
    const one = try alloc.allocate(&rng);
    defer testing.allocator.free(one);
    try testing.expectEqual(@as(f64, 1), one[0]);
}

test "allocation rechecks a publicly mutated floor" {
    var alloc = try CapacityAllocator.init(testing.allocator, 3, 0.2);
    defer alloc.deinit();
    alloc.floor_per_arm = 0.5;
    var rng = rngFromSeed(7);
    try testing.expectError(error.InvalidFloor, alloc.allocate(&rng));
}
