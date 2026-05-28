# Changelog

## 0.0.1 — 2026-05-28

ArmPosterior with uniform / fromCounts / recordSuccess / recordFailure
/ mean / sample. Beta(integer α, integer β) sampler via order
statistics. CapacityAllocator with init(K, floor) / arm(idx) /
allocate(rng). Deterministic Wyhash-seeded RNG via rngFromSeed.
12/12 tests pass on Zig 0.16. Hot-arm convergence test: an arm with
50 successes vs 5 failures attracts >70% of capacity on average
across 500 trials vs a 5-success 50-failure arm.
