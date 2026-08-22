"""Absolute-start 6F5 CMF reference helpers.

This module deliberately does NOT apply the legacy trajectory-file
translation x_i=shift_i+1, y_j=shift_j+2. ``start`` is the actual CMF
coordinate vector v0.

A depth-N walk applies the Ramanujan Dreams trajectory operator at
macro positions v0, v0+t, ..., v0+(N-1)t. N=0 therefore means the
identity product at v0; it does not mean "pre-step by t".
"""
from __future__ import annotations

from fractions import Fraction

from cmf_walk_corrected import (
    STATUS_OK,
    WalkSingularity,
    apply_trajectory_step,
    rank_for_z,
)

NSHIFT = 11


def _backend(z_num: int, z_den: int, field: str, dps=None):
    if field == "fraction":
        return Fraction(z_num, z_den), Fraction(0), Fraction(1)
    if field == "mpf":
        import mpmath as mp
        if dps is not None:
            mp.mp.dps = dps
        return mp.mpf(z_num) / mp.mpf(z_den), mp.mpf(0), mp.mpf(1)
    raise ValueError(field)


def projected_identity_rows(rank: int, row_num: int, row_den: int, zero, one):
    if not (0 <= row_num < rank and 0 <= row_den < rank and row_num != row_den):
        raise ValueError(f"invalid projected rows ({row_num},{row_den}) for rank {rank}")
    rows = [[zero for _ in range(rank)] for _ in range(2)]
    rows[0][row_num] = one
    rows[1][row_den] = one
    return rows


def walk_projected_absolute(start, dirv, z_num, z_den, N,
                            row_num, row_den, field="fraction", dps=None,
                            snapshot_depths=()):
    """Walk only two selected rows from an absolute start point.

    Returns ``(rows, rank, status, final_pos, snapshots)``. Each snapshot
    maps depth -> a deep copy of the two projected rows.
    """
    if len(start) != NSHIFT or len(dirv) != NSHIFT:
        raise ValueError("start and direction must each contain 11 coordinates")
    if N < 0:
        raise ValueError("N must be non-negative")

    z, zero, one = _backend(z_num, z_den, field, dps=dps)
    rank = rank_for_z(Fraction(z_num, z_den))
    pos = [int(v) for v in start]
    direction = [int(v) for v in dirv]
    rows = projected_identity_rows(rank, row_num, row_den, zero, one)

    wanted = set(int(v) for v in snapshot_depths)
    if any(v < 0 or v > N for v in wanted):
        raise ValueError("snapshot depth outside 0..N")
    snapshots = {}
    if 0 in wanted:
        snapshots[0] = [r[:] for r in rows]

    status = STATUS_OK
    try:
        for depth in range(1, N + 1):
            apply_trajectory_step(rows, pos, direction, z, rank, zero, one)
            if field == "mpf":
                # Projective renormalization controls magnitude growth and
                # is common to both rows, so the target ratio is unchanged.
                mx = max(abs(v) for row in rows for v in row)
                if mx:
                    rows = [[v / mx for v in row] for row in rows]
            if depth in wanted:
                snapshots[depth] = [r[:] for r in rows]
    except WalkSingularity as exc:
        status = exc.status

    return rows, rank, status, pos, snapshots


def projected_ratio(rows, rank: int):
    """Last-active-column ratio for the two projected rows."""
    den = rows[1][rank - 1]
    if den == 0:
        raise ZeroDivisionError("projected denominator row has zero last-column entry")
    return rows[0][rank - 1] / den


def exact_delta_from_fraction(approx: Fraction, target_mpf, dps=120):
    """Ramanujan Dreams-style delta from an exact rational approximant.

    delta = -1 - log(|target-p/q|)/log(q), with q the reduced exact
    denominator. This is a real delta estimate, unlike the GPU height proxy.
    """
    import mpmath as mp
    mp.mp.dps = dps
    q = abs(int(approx.denominator))
    if q <= 1:
        return None
    a = mp.mpf(approx.numerator) / mp.mpf(approx.denominator)
    err = abs(target_mpf - a)
    if err == 0:
        return mp.inf
    return -1 - mp.log(err) / mp.log(q)
