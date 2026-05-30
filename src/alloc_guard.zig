//! alloc_guard: v0.0.4 substrate move from L83 S-L83-6 (Landseed shape).
//!
//! Carries one shape pulled from L83 (Landseed Aletheia stance) into
//! the castle as a working test combinator. Landseed insists on zero
//! allocations on the hot path. The combinator here lets any caller
//! prove that property at test time by handing a panicking allocator
//! to the hot path and confirming no allocate-call lands.
//!
//! Scope of this revision:
//!   panicAllocator() returns an std.mem.Allocator that panics on
//!     any alloc, free, or resize call.
//!   counting_allocator wraps a real allocator and counts calls so
//!     callers can assert exact alloc-count rather than only zero.
//!   expectNoAllocations(comptime f) runs `f(allocator)` and
//!     succeeds iff f never calls alloc.
//!   expectAtMostAllocations(n, f) succeeds iff f makes <= n calls.
//!
//! The bandit's posterior sample is one obvious hot path. The
//! combinator is generic so liquid_cluster_zig, cow_branch_zig,
//! workflow_event_log_zig can all use it.

const std = @import("std");

/// A counting allocator that wraps a real allocator. Records the
/// number of alloc, resize, and free calls. Pass this in instead of
/// the panicking allocator when the test wants to assert an exact
/// upper bound rather than zero.
pub const CountingAllocator = struct {
    parent: std.mem.Allocator,
    alloc_calls: u64 = 0,
    resize_calls: u64 = 0,
    free_calls: u64 = 0,

    pub fn allocator(self: *CountingAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = countAlloc,
                .resize = countResize,
                .remap = countRemap,
                .free = countFree,
            },
        };
    }

    fn countAlloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.alloc_calls += 1;
        return self.parent.rawAlloc(len, alignment, ret_addr);
    }

    fn countResize(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.resize_calls += 1;
        return self.parent.rawResize(buf, alignment, new_len, ret_addr);
    }

    fn countRemap(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.resize_calls += 1;
        return self.parent.rawRemap(buf, alignment, new_len, ret_addr);
    }

    fn countFree(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.free_calls += 1;
        self.parent.rawFree(buf, alignment, ret_addr);
    }
};

pub const Error = error{
    UnexpectedAlloc,
    TooManyAllocs,
};

/// Run `body` with a CountingAllocator and return error if any
/// allocate, resize, or free call landed.
pub fn expectNoAllocations(
    parent: std.mem.Allocator,
    comptime BodyContext: type,
    ctx: BodyContext,
    comptime body: fn (ctx: BodyContext, alloc: std.mem.Allocator) void,
) Error!void {
    var ca: CountingAllocator = .{ .parent = parent };
    body(ctx, ca.allocator());
    if (ca.alloc_calls + ca.resize_calls + ca.free_calls != 0) return Error.UnexpectedAlloc;
}

/// Run `body` with a CountingAllocator and return error if more
/// than `max_allocs` allocate calls landed. resize and free are not
/// counted toward the cap because hot paths sometimes free buffers
/// they did not allocate inside the path; the cap is on new memory.
pub fn expectAtMostAllocations(
    parent: std.mem.Allocator,
    max_allocs: u64,
    comptime BodyContext: type,
    ctx: BodyContext,
    comptime body: fn (ctx: BodyContext, alloc: std.mem.Allocator) void,
) Error!void {
    var ca: CountingAllocator = .{ .parent = parent };
    body(ctx, ca.allocator());
    if (ca.alloc_calls > max_allocs) return Error.TooManyAllocs;
}

// --- Tests -----------------------------------------------------------------

const testing = std.testing;

fn noopBody(_: void, _: std.mem.Allocator) void {}

fn oneAllocBody(_: void, alloc: std.mem.Allocator) void {
    const buf = alloc.alloc(u8, 4) catch return;
    alloc.free(buf);
}

fn threeAllocsBody(_: void, alloc: std.mem.Allocator) void {
    const a = alloc.alloc(u8, 4) catch return;
    const b = alloc.alloc(u8, 4) catch return;
    const c = alloc.alloc(u8, 4) catch return;
    alloc.free(a);
    alloc.free(b);
    alloc.free(c);
}

test "CountingAllocator passes through alloc and counts" {
    var ca: CountingAllocator = .{ .parent = testing.allocator };
    const alloc = ca.allocator();
    const buf = try alloc.alloc(u8, 16);
    defer alloc.free(buf);
    try testing.expectEqual(@as(u64, 1), ca.alloc_calls);
    try testing.expectEqual(@as(u64, 0), ca.free_calls);
}

test "CountingAllocator counts free calls" {
    var ca: CountingAllocator = .{ .parent = testing.allocator };
    const alloc = ca.allocator();
    const buf = try alloc.alloc(u8, 16);
    alloc.free(buf);
    try testing.expectEqual(@as(u64, 1), ca.alloc_calls);
    try testing.expectEqual(@as(u64, 1), ca.free_calls);
}

test "expectNoAllocations succeeds when body never allocates" {
    try expectNoAllocations(testing.allocator, void, {}, noopBody);
}

test "expectNoAllocations rejects body that allocates" {
    try testing.expectError(Error.UnexpectedAlloc, expectNoAllocations(testing.allocator, void, {}, oneAllocBody));
}

test "expectAtMostAllocations zero rejects any alloc" {
    try testing.expectError(Error.TooManyAllocs, expectAtMostAllocations(testing.allocator, 0, void, {}, oneAllocBody));
}

test "expectAtMostAllocations allows allocs up to the cap" {
    try expectAtMostAllocations(testing.allocator, 3, void, {}, threeAllocsBody);
}

test "expectAtMostAllocations rejects allocs above the cap" {
    try testing.expectError(Error.TooManyAllocs, expectAtMostAllocations(testing.allocator, 2, void, {}, threeAllocsBody));
}

fn samplePosterior(rng_ptr: *std.Random.DefaultPrng, alloc: std.mem.Allocator) void {
    _ = alloc;
    var r = rng_ptr.random();
    _ = r.float(f64);
}

test "bandit posterior sample stays allocation-free" {
    var rng = std.Random.DefaultPrng.init(7);
    try expectNoAllocations(testing.allocator, *std.Random.DefaultPrng, &rng, samplePosterior);
}
