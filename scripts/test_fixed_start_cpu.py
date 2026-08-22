#!/usr/bin/env python3
"""Cross-platform semantic tests for the fixed-start path (no Metal required)."""
from __future__ import annotations

import os
import sys
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "lib"))

from cmf_walk_absolute import projected_ratio, walk_projected_absolute  # noqa: E402
from cmf_walk_corrected import STATUS_ZERO_AXIS_DENOMINATOR, walk  # noqa: E402


def legacy_shift_from_absolute(start):
    return [int(v) - 1 for v in start[:6]] + [int(v) - 2 for v in start[6:]]


def assert_projection_matches_full(start, direction, zn, zd, pair, N):
    """Two projected rows must equal the same rows of the full corrected walk."""
    legacy = legacy_shift_from_absolute(start)
    full, rank, st_full = walk(legacy, direction, zn, zd, N, field="fraction")
    rows, rank2, st_proj, pos, _ = walk_projected_absolute(
        start, direction, zn, zd, N, pair[0], pair[1], field="fraction"
    )
    assert rank == rank2
    assert st_full == st_proj == 0, (st_full, st_proj, start, direction, N)
    assert rows[0] == full[pair[0]][:rank]
    assert rows[1] == full[pair[1]][:rank]
    assert pos == [a + N * b for a, b in zip(start, direction)]


def main():
    base = [2] * 6 + [3] * 5
    pair = (5, 4)

    # Forward-only numerator path: deliberately simple and guaranteed not to
    # invoke an inverse operator. It is ideal for testing N=0/N=1 semantics.
    direction = [1] + [0] * 10

    rows, rank, st, pos, snaps = walk_projected_absolute(
        base, direction, -1, 1, 0, pair[0], pair[1],
        field="fraction", snapshot_depths=(0,)
    )
    assert st == 0
    assert pos == base
    assert rows[0][5] == 1 and sum(rows[0]) == 1
    assert rows[1][4] == 1 and sum(rows[1]) == 1
    assert snaps[0] == rows

    for N in (1, 2, 3, 5):
        assert_projection_matches_full(base, direction, -1, 1, pair, N)

    # A separated start exercises absolute coordinates without root
    # collisions. Test both a numerator-forward and denominator-inverse path.
    separated = [2, 4, 6, 8, 10, 12, 15, 17, 19, 21, 23]
    x_forward = [0, 1] + [0] * 9
    y_inverse = [0] * 6 + [1, 0, 0, 0, 0]
    for N in (1, 2, 3):
        assert_projection_matches_full(separated, x_forward, 1, 2, pair, N)
        assert_projection_matches_full(separated, y_inverse, -1, 1, pair, N)

    # Mixed forward-only numerator path also tests |direction_i| > 1 and the
    # level-descending unit-step decomposition without introducing inverses.
    mixed_forward = [1, 0, 2, 0, 0, 1] + [0] * 5
    for N in (1, 2, 3):
        assert_projection_matches_full(separated, mixed_forward, -1, 1, pair, N)

    # Known pole: x0=2, direction -1. First step reaches x0=1;
    # second negative step has denominator x0-1=0.
    singular = [-1] + [0] * 10
    _, _, st, _, _ = walk_projected_absolute(
        base, singular, -1, 1, 2, pair[0], pair[1], field="fraction"
    )
    assert st == STATUS_ZERO_AXIS_DENOMINATOR

    # The ratio object in exact mode must remain a reduced Fraction.
    rows, rank, st, _, _ = walk_projected_absolute(
        base, direction, -1, 1, 3, pair[0], pair[1], field="fraction"
    )
    if st == 0 and rows[1][rank - 1] != 0:
        assert isinstance(projected_ratio(rows, rank), Fraction)

    print("FIXED-START CPU SEMANTICS VERIFIED")


if __name__ == "__main__":
    main()
