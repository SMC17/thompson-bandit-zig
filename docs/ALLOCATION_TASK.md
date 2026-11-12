# Bounded allocation exercise

Inputs are original synthetic success/failure counts. Each arm starts with
one prior success and one prior failure. The task records the RNG seed, floor,
posterior parameters, sampled weights and resulting capacity fractions.

The checker computes each mean as the exact rational a/(a+b). For the separate
Beta(3,2) probe, the exact mean is 3/5, second moment 2/5, and CDF at 1/2 is
5/16 (the integer-parameter binomial tail). Observed 5,000-draw statistics must
stay within 0.03 of those references. This tolerance is a fixed sanity bound;
it does not make an accuracy guarantee for every posterior or random seed.

Allocation is checked through a separate numerical definition: find a scale
such that sum(max(floor, scale * sampled_weight)) = 1. The Python checker solves
that monotone equation by bisection. It does not copy the Zig implementation's
pin-and-redistribute loop. Unit regressions separately cover [0.95,0.04,0.01]
with a 0.2 floor, cascading clamps, a maximal feasible floor, one arm, invalid
parameters and every initialization allocation failure.

```sh
zig build test -Doptimize=ReleaseSafe --summary all
zig build -Doptimize=ReleaseSafe
python3 tools/check_allocation.py
./zig-out/bin/bandit-task 42 0.2
```

The CLI accepts optional decimal seed and floor arguments; it always uses the
same three synthetic arms. It refuses malformed/negative seeds, infeasible or
nonfinite floors and excess arguments before writing output. The checker runs
five seeds at three floors (including zero and the maximal feasible 1/3),
repeats each command and checks seven refusal cases.

The order-statistic sampler has an explicit mean-only fallback for
alpha+beta-1 > 1024. The large posterior probe reports Beta(1025,2), whose
sample is its mean 1025/1027. That is not a draw from the Beta distribution.
Count overflow and direct invalid posterior-field mutation remain caller
preconditions. No synthetic task establishes a capacity SLO, policy optimality,
production deployment, economic outcome or external user adoption.

Existing AGPL-3.0-or-later terms are preserved. LICENSE contains the full text
retrieved from GNU; NOTICE retains the previous license declaration and owner
copyright verbatim. The selfserve recipe creates private local evidence packs;
it does not publish source or a release.
