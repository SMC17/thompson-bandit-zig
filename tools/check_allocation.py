#!/usr/bin/env python3
"""Check original synthetic allocations using exact moments and a bisection oracle."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess


def check(record):
    assert record["schema"] == 1
    counts = record["counts"]
    assert counts == [[8, 2], [1, 9], [0, 0]]
    for i, (success, failure) in enumerate(counts):
        a, b = success + 1, failure + 1
        assert (record["alpha"][i], record["beta"][i]) == (a, b)
        assert abs(record["posterior_means"][i] - float(Fraction(a, a + b))) < 1e-12
    weights, shares, floor = record["sampled_weights"], record["shares"], record["floor"]
    assert len(weights) == len(shares) == 3
    assert all(math.isfinite(x) and 0 <= x <= 1 for x in weights + shares)
    assert abs(sum(shares) - 1) < 1e-12
    assert min(shares) >= floor - 1e-12
    # Independent numerical root: sum(max(floor, scale * weight)) == 1.
    # This does not reproduce the implementation's pin-and-redistribute loop.
    lo, hi = 0.0, 1.0
    while sum(max(floor, hi * w) for w in weights) < 1:
        hi *= 2
    for _ in range(100):
        scale = (lo + hi) / 2
        if sum(max(floor, scale * w) for w in weights) > 1:
            hi = scale
        else:
            lo = scale
    expected = [max(floor, (lo + hi) / 2 * w) for w in weights]
    assert max(abs(a - b) for a, b in zip(shares, expected)) < 1e-12

    probe = record["distribution_probe"]
    a, b, n = probe["alpha"], probe["beta"], probe["draws"]
    assert (a, b, n) == (3, 2, 5000)
    mean = Fraction(a, a + b)
    second = Fraction(a * (a + 1), (a + b) * (a + b + 1))
    # At 1/2 the integer-Beta CDF is an exact binomial tail.
    degree = a + b - 1
    cdf = Fraction(sum(math.comb(degree, k) for k in range(a, degree + 1)), 2 ** degree)
    assert mean == Fraction(3, 5) and second == Fraction(2, 5) and cdf == Fraction(5, 16)
    for field, reference in [("mean", mean), ("second_moment", second), ("cdf_half", cdf)]:
        assert abs(probe[field] - float(reference)) < 0.03, field
    fallback = record["mean_fallback"]
    assert (fallback["alpha"], fallback["beta"]) == (1025, 2)
    assert fallback["sample"] == float(Fraction(1025, 1027))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path, default=Path("zig-out/bin/bandit-task"))
    args = parser.parse_args()
    binary = args.bin.resolve()
    tested = 0
    for seed in [0, 1, 7, 42, 2026]:
        for floor in [0.0, 0.2, 1 / 3]:
            command = [str(binary), str(seed), repr(floor)]
            first = subprocess.run(command, capture_output=True, check=True, timeout=15).stdout
            assert first == subprocess.run(command, capture_output=True, check=True, timeout=15).stdout
            record = json.loads(first)
            assert record["seed"] == seed and record["floor"] == floor
            check(record)
            tested += 1
    invalid = [["bad"], ["-1"], ["42", "-0.1"], ["42", "0.34"],
               ["42", "nan"], ["42", "inf"], ["42", "0.2", "extra"]]
    for case in invalid:
        result = subprocess.run([str(binary), *case], capture_output=True, timeout=15)
        assert result.returncode != 0 and not result.stdout, case
    print(f"ALLOCATION_TASK_OK cases={tested} refusals={len(invalid)} posterior=exact floor=bisection repeat=identical distribution=bounded-sanity")


if __name__ == "__main__":
    main()
