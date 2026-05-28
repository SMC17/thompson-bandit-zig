# thompson-bandit-zig

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](LICENSE)
[![Zig](https://img.shields.io/badge/Zig-0.16-orange.svg)](https://ziglang.org/)

Beta posterior + Thompson sampling + adaptive capacity allocator in Zig 0.16. Three-source synthesis: AWS DynamoDB L650 adaptive capacity throttling + Doordash L46 Thompson-bandit search ranking + Datadog L37 Thompson-based cardinality control.

## Status

**v0.0.1 — 12/12 tests pass on Zig 0.16.** `ArmPosterior` tracks Beta(α, β) per arm; `CapacityAllocator` samples each arm's posterior and returns a fractional allocation summing to 1.0; deterministic Wyhash-seeded RNG so allocations are reproducible.

## What ships

- `ArmPosterior` with `uniform()` / `fromCounts(successes, failures)` / `recordSuccess` / `recordFailure` / `mean` / `sample(rng)`.
- Beta(α, β) sampler via order statistics (integer α + β; sufficient for the count-update regime v0.0.1 ships).
- `CapacityAllocator` with `init(K, floor)` / `armCount` / `arm(idx)` / `allocate(rng)` / `deinit`.
- Floor option: every arm gets at least a minimum share so cold arms still get explored (matches AWS DynamoDB's 25%/N baseline pattern).
- `rngFromSeed(u64)` — deterministic `DefaultPrng` for reproducible allocations + tests.

## Build

```sh
zig build test                  # 12 unit tests
```

## What ships does NOT do (yet)

- **No real-valued Beta(α, β).** v0.0.1 uses the integer-parameter order-statistic shortcut. v0.0.2 wires the gamma-distribution ratio for arbitrary real α + β.
- **No windowed posteriors.** v0.0.1 accumulates indefinitely; concept drift erodes accuracy over long horizons. v0.0.2 ships sliding-window posteriors (Datadog L37 pattern).
- **No multi-armed allocator with capacity floors per arm.** v0.0.1 ships a single uniform floor. v0.0.3 wires per-arm capacity floors + minimum-throughput contracts (AWS DynamoDB partition-quota pattern).
- **No persistence.** Memory-only at v0.0.1. v0.0.2 wires the on-disk layer (composes with shipped `tableformat-zig`).

## Composes with shipped substrate

- **`expt-substrate` Wyhash seed-derivation.** Same seed family → reproducible A/B → bandit transitions.
- **`tableformat-zig` (v0.0.2 storage)** — manifest of arm-state snapshots.
- **Spark AQE L846 (mined this turn)** — Thompson posterior drives adaptive replanning at shuffle boundaries. Same primitive serves orderbook + DynamoDB-class + Spark-class.

## Honest non-claims

- Pre-1.0 substrate.
- Order-statistic Beta sampler caps `α + β` at 1024; beyond that v0.0.1 falls back to the mean (no sample variance). v0.0.2 ships gamma-ratio sampling without that cap.
- No multi-threading guarantees; caller serialises access.

## Credit

Concepts adapted from AWS DynamoDB adaptive capacity (castle L650), Doordash search ranking (castle L46), Datadog cardinality control (castle L37). Frontier port by Sean Collins (`sean@sunlitmoon.online`).

## License

AGPL-3.0-or-later. See `LICENSE`.
