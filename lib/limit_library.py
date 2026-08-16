"""
limit_library.py
================
~77,000-entry constant library for Float64 15-digit limit matching.

Built from:
  - Primitive constants (rationals, zeta, pi powers, logs, gamma, catalan, etc.)
  - Scalar multiples of each primitive
  - Pairwise combinations (sum, diff, product, ratio) of all base primitives
  - Deduplicated by value (12 significant digits)
"""
from __future__ import annotations

import math
import os
from fractions import Fraction

import mpmath as mp
import numpy as np

_CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_FILE = os.path.join(_CACHE_DIR, "limit_library_cache.npz")


def _build_primitives() -> dict[str, float]:
    """Build base primitive constants at 50 dps, return as float64 dict."""
    mp.mp.dps = 50
    P: dict[str, float] = {}

    # 1 is the unit
    P["1"] = 1.0

    # Rationals a/b for a in [-20,20], b in [1,20] (dedup by value)
    seen_rat = set()
    for a in range(-20, 21):
        if a == 0:
            continue
        for b in range(1, 21):
            f = Fraction(a, b)
            if f in seen_rat:
                continue
            seen_rat.add(f)
            P[f"{a}/{b}"] = float(f)

    # Zeta values zeta(2) through zeta(20)
    for s in range(2, 21):
        P[f"zeta({s})"] = float(mp.zeta(s))

    # Powers of pi
    for k in range(1, 7):
        P[f"pi^{k}"] = float(mp.pi ** k)

    # Logs ln(n) for n=2..20
    for n in range(2, 21):
        P[f"ln({n})"] = float(mp.log(n))

    # Other constants
    P["euler_gamma"] = float(mp.euler)
    P["catalan"] = float(mp.catalan)
    P["phi"] = float((1 + mp.sqrt(5)) / 2)
    P["glaisher"] = float(mp.glaisher)
    P["khinchin"] = float(mp.khinchin)
    P["e"] = float(mp.e)

    # Polylogarithms Li_s(1/2) for s=2..6
    half = mp.mpf(1) / 2
    for s in range(2, 7):
        P[f"Li_{s}(1/2)"] = float(mp.polylog(s, half))

    # Dirichlet beta values beta(s) for s=2..6
    for s in range(2, 7):
        P[f"beta({s})"] = float(mp.nsum(lambda n: (-1)**n / (2*n+1)**s, [0, mp.inf]))

    # Square roots
    for n in range(2, 13):
        P[f"sqrt({n})"] = float(mp.sqrt(n))

    # Gamma values
    g13 = mp.gamma(mp.mpf(1) / 3)
    g14 = mp.gamma(mp.mpf(1) / 4)
    P["gamma_1_3"] = float(g13)
    P["gamma_1_4"] = float(g14)
    P["gamma_1_3^3"] = float(g13 ** 3)
    P["gamma_1_4^2"] = float(g14 ** 2)

    # Common products/ratios
    P["zeta3/pi^3"] = float(mp.zeta(3) / mp.pi ** 3)
    P["zeta3/pi^2"] = float(mp.zeta(3) / mp.pi ** 2)
    P["zeta5/pi^5"] = float(mp.zeta(5) / mp.pi ** 5)
    P["zeta5/pi^4"] = float(mp.zeta(5) / mp.pi ** 4)
    P["catalan/pi"] = float(mp.catalan / mp.pi)
    P["pi*sqrt3"] = float(mp.pi * mp.sqrt(3))
    P["log2^2"] = float(mp.log(2) ** 2)
    P["log2^3"] = float(mp.log(2) ** 3)

    return P


def build_library() -> tuple[np.ndarray, list[str]]:
    """Build the full ~77K-entry library. Returns (sorted_values, names)."""
    if os.path.exists(_CACHE_FILE):
        data = np.load(_CACHE_FILE, allow_pickle=True)
        return data["values"], data["names"].tolist()

    primitives = _build_primitives()
    prim_items = list(primitives.items())
    prim_names = [n for n, _ in prim_items]
    prim_vals = np.array([v for _, v in prim_items], dtype=np.float64)

    # Collect all entries: (value, name)
    entries: dict[str, tuple[float, str]] = {}

    def _add(val, name):
        if not np.isfinite(val) or val == 0.0:
            return
        key = f"{val:.12g}"
        if key not in entries:
            entries[key] = (val, name)

    # 1. Primitives themselves
    for name, val in prim_items:
        _add(val, name)
        _add(-val, f"-{name}")

    # 2. Scalar multiples: each primitive * (n/d for n,d in 1..12)
    rationals = []
    seen_r = set()
    for n in range(1, 13):
        for d in range(1, 13):
            f = Fraction(n, d)
            if f not in seen_r:
                seen_r.add(f)
                rationals.append((float(f), f"{n}/{d}"))
    for pname, pval in prim_items:
        for rval, rname in rationals:
            _add(pval * rval, f"{rname}*{pname}")
            _add(-pval * rval, f"-{rname}*{pname}")

    # 3. Pairwise combinations of base primitives (sum, diff, product, ratio)
    n_prim = len(prim_items)
    for i in range(n_prim):
        ni, vi = prim_items[i]
        for j in range(i, n_prim):
            nj, vj = prim_items[j]
            _add(vi + vj, f"{ni}+{nj}")
            _add(vi - vj, f"{ni}-{nj}")
            _add(vj - vi, f"{nj}-{ni}")
            _add(vi * vj, f"{ni}*{nj}")
            if vj != 0:
                _add(vi / vj, f"{ni}/{nj}")
            if vi != 0 and i != j:
                _add(vj / vi, f"{nj}/{ni}")

    # Build sorted arrays
    all_vals = np.array([v for v, _ in entries.values()], dtype=np.float64)
    all_names = [n for _, n in entries.values()]

    sort_idx = np.argsort(all_vals)
    all_vals = all_vals[sort_idx]
    all_names = [all_names[i] for i in sort_idx]

    # Cache to disk
    np.savez(_CACHE_FILE, values=all_vals, names=np.array(all_names, dtype=object))

    print(f"[limit_library] Built {len(all_vals)} unique constants")
    return all_vals, all_names


def match_limit(L_float: float, lib_vals: np.ndarray, lib_names: list[str],
                tol: float = 1e-12) -> tuple[str | None, float, int]:
    """Match a float64 limit against the sorted library.

    Returns (library_name, rel_error, matched_digits) or (None, inf, 0).
    """
    if not np.isfinite(L_float) or L_float == 0.0:
        return None, float("inf"), 0

    idx = np.searchsorted(lib_vals, L_float)

    best_name = None
    best_err = float("inf")

    for i in [idx - 1, idx]:
        if 0 <= i < len(lib_vals):
            cand = lib_vals[i]
            if cand == 0:
                continue
            rel_err = abs(L_float - cand) / max(abs(L_float), abs(cand))
            if rel_err < best_err:
                best_err = rel_err
                best_name = lib_names[i]

    if best_name is None or best_err > tol:
        return None, float("inf"), 0

    matched_digits = int(-math.log10(best_err)) if best_err > 0 else 16
    return best_name, best_err, matched_digits


def is_rational_name(name: str) -> bool:
    """Check if a library name represents a pure rational number."""
    # Names like "3/4", "-2/1", "6/1" are rational
    # Names like "1*zeta(3)" or "pi+log(2)" are not
    import re
    return bool(re.match(r'^-?\d+/\d+$', name))


if __name__ == "__main__":
    vals, names = build_library()
    print(f"Library size: {len(vals)}")
    print(f"Range: [{vals[0]:.6g}, {vals[-1]:.6g}]")
    # Quick test
    test = float(mp.zeta(3))
    name, err, digits = match_limit(test, vals, names)
    print(f"zeta(3) -> {name} err={err:.2e} digits={digits}")
