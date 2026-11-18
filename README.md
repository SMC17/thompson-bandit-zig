# thompson-bandit-zig

**[Status](STATUS.md) · [Roadmap](ROADMAP.md) · [Contributing](CONTRIBUTING.md) · [Issues](https://github.com/SMC17/thompson-bandit-zig/issues) · [Public roadmap](https://github.com/users/SMC17/projects/16)**

Beta posteriors and sampled capacity shares in Zig 0.16.0. This repository stays
private and retains AGPL-3.0-or-later terms. The complete license is in LICENSE;
NOTICE preserves the original declaration and copyright.

## Keep one seeded allocation record

```sh
zig build test --summary all
zig build -Doptimize=ReleaseSafe
./zig-out/bin/bandit-task 42 0.2 > allocation.json
python3 tools/check_allocation.py
```

The original synthetic fixture has three success/failure pairs: (8,2), (1,9)
and (0,0). With a Beta(1,1) prior their posteriors are Beta(9,3), Beta(2,10)
and Beta(1,1). The task emits counts, parameters, means, sampled weights and
shares. Every share must be at least the requested 0.2 and their sum must be
one. Repeat the same seed, toolchain and command to get identical JSON bytes.

The independent Python checker uses rational posterior moments and a bisection
allocation reference. It checks 15 seed/floor combinations, seven command-line
refusals and repeat identity. A fixed 5,000-draw probe checks Beta(3,2) mean,
second moment and CDF at one-half within a declared 0.03 tolerance. These are
bounded numerical sanity checks, not distribution certification.

The floor previously fell below its promise after normalization. The allocator
now reserves floored mass before redistributing the rest. Initialization also
cleans up on allocation failure. Empty arm sets and infeasible, negative or
nonfinite floors return errors; allocation rechecks a publicly changed floor.

## Scope

- `ArmPosterior`: count-based updates, mean and integer-parameter sampling.
- `CapacityAllocator`: normalized posterior samples with a uniform lower bound.
  This is a capacity-sharing policy; it does not select a single winning arm.
- `rngFromSeed`: repeatability with the pinned Zig generator/toolchain.
- The existing reward/window, PCG, WCET and other helper modules remain exported.
  The allocation task does not establish their suitability for a deployment.

The sampler draws and sorts `alpha + beta - 1` uniforms when that count is at
most 1024. Above that boundary it returns the posterior mean and has no sampling
variance. The task makes that fallback visible. Direct posterior fields must
remain positive and counts/additions must fit u64; arbitrary count overflow is
not handled by a checked update API. See [the task contract](docs/ALLOCATION_TASK.md).

No online learning, service-capacity guarantee, regret bound, performance result,
customer workload, persistent state or automatic connection to other portfolio
projects is demonstrated. Historical research-source notes are context rather
than a measured DynamoDB/Doordash/Datadog integration. Package visibility and
licensing do not change through the task.

## Unified C ABI, Python Wheel & WASM

Core primitives from this repository are also cross-pollinated into **[SMC17/zinging (`zig-substrate`)](https://github.com/SMC17/zinging)** with C11 ABI bindings (`199` symbols), standalone Python wheels, JSON-RPC daemon (`substrate serve`), and `wasm32` builds.
