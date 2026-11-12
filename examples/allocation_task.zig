const std = @import("std");
const bandit = @import("thompson_bandit");

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len > 3) return error.InvalidArguments;
    const seed = if (args.len > 1) try std.fmt.parseInt(u64, args[1], 10) else 42;
    const floor = if (args.len > 2) try std.fmt.parseFloat(f64, args[2]) else 0.2;
    const counts = [_][2]u64{ .{ 8, 2 }, .{ 1, 9 }, .{ 0, 0 } };
    var allocator = try bandit.CapacityAllocator.init(init.gpa, counts.len, floor);
    defer allocator.deinit();
    var alpha: [3]u64 = undefined;
    var beta: [3]u64 = undefined;
    var means: [3]f64 = undefined;
    var weights: [3]f64 = undefined;
    var rng = bandit.rngFromSeed(seed);
    var replay = rng;
    for (counts, 0..) |c, i| {
        allocator.arm(i).* = .fromCounts(c[0], c[1]);
        alpha[i] = allocator.arm(i).alpha;
        beta[i] = allocator.arm(i).beta;
        means[i] = allocator.arm(i).mean();
        weights[i] = allocator.arm(i).sample(&replay);
    }
    const shares = try allocator.allocate(&rng);
    defer init.gpa.free(shares);

    const probe = bandit.ArmPosterior.fromCounts(2, 1); // Beta(3, 2).
    var probe_rng = bandit.rngFromSeed(2026);
    const draws: usize = 5000;
    var sum: f64 = 0;
    var sum_squared: f64 = 0;
    var below_half: usize = 0;
    for (0..draws) |_| {
        const sample = probe.sample(&probe_rng);
        sum += sample;
        sum_squared += sample * sample;
        if (sample <= 0.5) below_half += 1;
    }
    const large = bandit.ArmPosterior.fromCounts(1024, 1);
    const record = try std.json.Stringify.valueAlloc(init.gpa, .{
        .schema = 1,
        .seed = seed,
        .floor = floor,
        .counts = counts,
        .alpha = alpha,
        .beta = beta,
        .posterior_means = means,
        .sampled_weights = weights,
        .shares = shares,
        .distribution_probe = .{
            .alpha = probe.alpha,
            .beta = probe.beta,
            .draws = draws,
            .mean = sum / @as(f64, @floatFromInt(draws)),
            .second_moment = sum_squared / @as(f64, @floatFromInt(draws)),
            .cdf_half = @as(f64, @floatFromInt(below_half)) / @as(f64, @floatFromInt(draws)),
        },
        .mean_fallback = .{ .alpha = large.alpha, .beta = large.beta, .sample = large.sample(&probe_rng) },
    }, .{ .whitespace = .indent_2 });
    defer init.gpa.free(record);
    try std.Io.File.stdout().writeStreamingAll(init.io, record);
    try std.Io.File.stdout().writeStreamingAll(init.io, "\n");
}
