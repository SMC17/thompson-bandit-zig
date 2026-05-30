//! samplers: v0.0.7 substrate from L118 Aletheia zmonte.
//!
//! L113 already extracted the Pcg64 RNG (shipped v0.0.6); L118
//! confirmed zmonte's distribution sampler surface and named
//! gaussian as the highest-fit primitive for bandit consumers
//! that need arm-posterior sampling beyond the existing Beta(a,
//! b) order-statistics path.
//!
//! Port three samplers (gaussian, exponential, poisson) plus
//! one MC primitive (integrateMC) verbatim from zmonte. Each
//! takes a duck-typed `rng: anytype` exposing nextF64() so
//! callers can compose with Pcg64, std.Random.DefaultPrng, or
//! any future named generator (Mt19937_64) without recompiling
//! this module.
//!
//! Scope of this revision:
//!   gaussian(rng, mean, sigma) returns one N(mean, sigma) draw
//!     via Marsaglia polar rejection; second draw discarded.
//!   exponential(rng, lambda) returns one Exp(lambda) draw via
//!     inverse-CDF -ln(1 - U) / lambda. Maps [0, 1) -> (0, inf).
//!   poisson(rng, mean) returns one Pois(mean) draw via Knuth
//!     product for lambda <= 30 and Gaussian approximation for
//!     lambda > 30.
//!   IntegrateMcResult { estimate, stderr } record.
//!   integrateMC(rng, lo, hi, n_samples, integrand) returns the
//!     crude MC estimate of integrand over [lo, hi] plus the
//!     Bessel-corrected standard error.
//!
//! Out of scope: sobol_1d (caller composes via aqe-replanner
//! cost-model or a separate dq-zig QMC primitive once a real
//! consumer demands); per-distribution batch APIs (caller loops
//! over the single-draw form).
//!
//! Citations: G. Marsaglia and T. Bray, "A convenient method
//! for generating normal variables," SIAM Review 6 (1964); D. E.
//! Knuth, "The Art of Computer Programming Vol. 2" (1997) for
//! the Poisson product method.

const std = @import("std");

pub const IntegrateMcResult = struct {
    estimate: f64,
    stderr: f64,
};

/// Marsaglia polar rejection. Two independent N(0, 1) draws per
/// loop iteration; second discarded. sigma must be positive;
/// negative or zero sigma returns NaN.
pub fn gaussian(rng: anytype, mean: f64, sigma: f64) f64 {
    if (sigma <= 0.0 or !std.math.isFinite(sigma)) return std.math.nan(f64);
    while (true) {
        const u = 2.0 * rng.nextF64() - 1.0;
        const v = 2.0 * rng.nextF64() - 1.0;
        const s = u * u + v * v;
        if (s <= 0.0 or s >= 1.0) continue;
        const factor = @sqrt(-2.0 * @log(s) / s);
        return mean + sigma * u * factor;
    }
}

/// Inverse-CDF: -ln(1 - U) / lambda. lambda must be positive;
/// otherwise returns NaN.
pub fn exponential(rng: anytype, lambda: f64) f64 {
    if (lambda <= 0.0 or !std.math.isFinite(lambda)) return std.math.nan(f64);
    const u = rng.nextF64();
    return -@log(1.0 - u) / lambda;
}

/// Knuth product for mean <= 30; Gaussian approximation for
/// mean > 30 (crossover where exp(-30) ~ 9.4e-14 is below f64
/// underflow margin). Non-positive or non-finite mean returns 0.
pub fn poisson(rng: anytype, mean: f64) u64 {
    if (mean <= 0.0 or !std.math.isFinite(mean)) return 0;
    if (mean <= 30.0) {
        const l = @exp(-mean);
        var k: u64 = 0;
        var p: f64 = 1.0;
        while (true) {
            k += 1;
            p *= rng.nextF64();
            if (p <= l) return k - 1;
        }
    }
    // Gaussian approximation: N(mean, sqrt(mean))
    const g = gaussian(rng, mean, @sqrt(mean));
    if (g < 0.0) return 0;
    return @intFromFloat(@round(g));
}

/// Crude Monte Carlo: estimate = (hi - lo) * mean(f(U)) for U
/// uniform on [lo, hi]. stderr is Bessel-corrected. Requires
/// n_samples >= 2; lower values yield estimate-only with stderr
/// 0. lo > hi yields a sign-flipped estimate (caller-policy).
pub fn integrateMC(
    rng: anytype,
    lo: f64,
    hi: f64,
    n_samples: usize,
    integrand: *const fn (f64) f64,
) IntegrateMcResult {
    if (n_samples == 0) return .{ .estimate = 0.0, .stderr = 0.0 };
    const width = hi - lo;
    // One-pass Welford for variance.
    var mean: f64 = 0.0;
    var m2: f64 = 0.0;
    var i: usize = 0;
    while (i < n_samples) : (i += 1) {
        const u = lo + width * rng.nextF64();
        const y = integrand(u);
        const delta = y - mean;
        mean += delta / @as(f64, @floatFromInt(i + 1));
        const delta2 = y - mean;
        m2 += delta * delta2;
    }
    const estimate = width * mean;
    if (n_samples < 2) return .{ .estimate = estimate, .stderr = 0.0 };
    const variance = m2 / @as(f64, @floatFromInt(n_samples - 1));
    const stderr = @abs(width) * @sqrt(variance / @as(f64, @floatFromInt(n_samples)));
    return .{ .estimate = estimate, .stderr = stderr };
}

// --- Tests -----------------------------------------------------------------

const testing = std.testing;
const Pcg64 = @import("pcg64.zig").Pcg64;

test "gaussian returns NaN for non-positive sigma" {
    var g: Pcg64 = .initDefault(1);
    try testing.expect(std.math.isNan(gaussian(&g, 0.0, 0.0)));
    try testing.expect(std.math.isNan(gaussian(&g, 0.0, -1.0)));
}

test "gaussian mean and stddev over 10000 draws are close to spec" {
    var g: Pcg64 = .initDefault(42);
    var sum: f64 = 0.0;
    var sumsq: f64 = 0.0;
    const n: usize = 10_000;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const x = gaussian(&g, 5.0, 2.0);
        sum += x;
        sumsq += x * x;
    }
    const mean = sum / @as(f64, @floatFromInt(n));
    const variance = sumsq / @as(f64, @floatFromInt(n)) - mean * mean;
    const sigma = @sqrt(variance);
    try testing.expect(@abs(mean - 5.0) < 0.1);
    try testing.expect(@abs(sigma - 2.0) < 0.1);
}

test "exponential returns NaN for non-positive lambda" {
    var g: Pcg64 = .initDefault(1);
    try testing.expect(std.math.isNan(exponential(&g, 0.0)));
    try testing.expect(std.math.isNan(exponential(&g, -1.0)));
}

test "exponential mean over 10000 draws is close to 1/lambda" {
    var g: Pcg64 = .initDefault(99);
    var sum: f64 = 0.0;
    const n: usize = 10_000;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        sum += exponential(&g, 2.0);
    }
    const mean = sum / @as(f64, @floatFromInt(n));
    try testing.expect(@abs(mean - 0.5) < 0.05);
}

test "poisson returns 0 for non-positive mean" {
    var g: Pcg64 = .initDefault(1);
    try testing.expectEqual(@as(u64, 0), poisson(&g, 0.0));
    try testing.expectEqual(@as(u64, 0), poisson(&g, -3.0));
}

test "poisson mean over 5000 draws (lambda 5) is close to spec" {
    var g: Pcg64 = .initDefault(7);
    var sum: u64 = 0;
    const n: usize = 5_000;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        sum += poisson(&g, 5.0);
    }
    const mean = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(n));
    try testing.expect(@abs(mean - 5.0) < 0.2);
}

test "poisson mean over 5000 draws (lambda 50, Gaussian regime)" {
    var g: Pcg64 = .initDefault(11);
    var sum: u64 = 0;
    const n: usize = 5_000;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        sum += poisson(&g, 50.0);
    }
    const mean = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(n));
    try testing.expect(@abs(mean - 50.0) < 1.5);
}

fn identity(x: f64) f64 {
    return x;
}

test "integrateMC of identity over [0, 1] is ~ 0.5" {
    var g: Pcg64 = .initDefault(13);
    const r = integrateMC(&g, 0.0, 1.0, 10_000, identity);
    try testing.expect(@abs(r.estimate - 0.5) < 0.02);
    try testing.expect(r.stderr > 0.0);
    try testing.expect(r.stderr < 0.01);
}

fn parabola(x: f64) f64 {
    return x * x;
}

test "integrateMC of x^2 over [0, 1] is ~ 1/3" {
    var g: Pcg64 = .initDefault(17);
    const r = integrateMC(&g, 0.0, 1.0, 10_000, parabola);
    try testing.expect(@abs(r.estimate - (1.0 / 3.0)) < 0.02);
}

test "integrateMC with n_samples 0 returns 0.0/0.0" {
    var g: Pcg64 = .initDefault(1);
    const r = integrateMC(&g, 0.0, 1.0, 0, identity);
    try testing.expectEqual(@as(f64, 0.0), r.estimate);
    try testing.expectEqual(@as(f64, 0.0), r.stderr);
}

test "samplers compose with seeded Pcg64 for reproducible streams" {
    var a: Pcg64 = .initDefault(2026);
    var b: Pcg64 = .initDefault(2026);
    try testing.expectEqual(gaussian(&a, 0.0, 1.0), gaussian(&b, 0.0, 1.0));
    try testing.expectEqual(exponential(&a, 1.0), exponential(&b, 1.0));
    try testing.expectEqual(poisson(&a, 3.0), poisson(&b, 3.0));
}
