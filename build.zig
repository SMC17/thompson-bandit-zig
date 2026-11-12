const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("thompson_bandit", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const root_tests = b.addTest(.{ .root_module = mod });
    const run_root_tests = b.addRunArtifact(root_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_root_tests.step);

    const task = b.addExecutable(.{
        .name = "bandit-task",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/allocation_task.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    task.root_module.addImport("thompson_bandit", mod);
    b.installArtifact(task);
    const run_task = b.addRunArtifact(task);
    if (b.args) |args| run_task.addArgs(args);
    b.step("task", "Emit a seeded synthetic allocation and posterior record").dependOn(&run_task.step);
}
