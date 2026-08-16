"""
cmf_walk_corrected.py
=====================
CORRECTED Ramanujan Dreams 6F5 CMF trajectory walk - reference oracle.

Semantics (matching upstream ramanujantools Matrix.walk):

  Start point from the trajectory file's shift vector:
      x_i(0) = shift_i   + 1   (i = 0..5)
      y_j(0) = shift_{6+j} + 2 (j = 0..4)
  so shift = (1,...,1)  <=>  v0 = (2,2,2,2,2,2; 3,3,3,3,3).

  Theta differential operator:
      D(theta) = theta * prod_j (theta + y_j - 1)  -  z * prod_i (theta + x_i)
               = d_0 + d_1 theta + ... + d_r theta^r
  Rank r = 6 for z != 1, r = 5 for z = 1 (d_6 = 1 - z).
  Monic companion C(v): subdiagonal ones, last column c_i = -d_i / d_r.

  Axis operators (I + C/a), four cases:
      x_i +1 : A =  I + C(v)/x_i            then x_i += 1
      x_i -1 : A = (I + C(v-e_i)/(x_i-1))^-1 then x_i -= 1
      y_j -1 : A =  I + C(v)/(y_j-1)         then y_j -= 1
      y_j +1 : A = (I + C(v+e_j)/y_j)^-1     then y_j += 1

  A trajectory step T(v, t) decomposes t into unit axis steps:
      for level = max|t| .. 1:  for axis = 10 .. 0:
          if |t[axis]| >= level: apply one sign(t[axis]) step
  Full walk: W_N = T(v0,t) T(v0+t,t) ... T(v0+(N-1)t,t).

Two field backends:
  field="fraction": exact rational arithmetic (golden oracle)
  field="mpf":      mpmath at current mp.mp.dps (deep walks / PSLQ)
"""
from __future__ import annotations

from fractions import Fraction

MAX_DIM = 6
NSHIFT = 11
NX = 6   # numerator roots x_i
NY = 5   # denominator roots y_j

STATUS_OK = 0
STATUS_ZERO_AXIS_DENOMINATOR = 1
STATUS_THETA_LEAD_DEGENERATE = 2
STATUS_INVERSE_SINGULAR = 3
STATUS_NONFINITE = 4
STATUS_NEEDS_REGULARIZATION = 5


class WalkSingularity(Exception):
    def __init__(self, status: int, msg: str = ""):
        super().__init__(msg or f"status={status}")
        self.status = status


def rank_for_z(z) -> int:
    """rank(6F5, z) = 5 if z == 1 else 6."""
    return 5 if z == 1 else 6


def start_pos(shift):
    """Actual Dreams start point from a trajectory-file shift vector."""
    assert len(shift) == NSHIFT
    return ([int(shift[i]) + 1 for i in range(NX)] +
            [int(shift[NX + j]) + 2 for j in range(NY)])


def _poly_from_roots(vals, zero, one):
    """Coefficients of prod (theta + v) for v in vals, ascending order."""
    coeffs = [one] + [zero] * len(vals)
    deg = 0
    for v in vals:
        for k in range(deg + 1, 0, -1):
            coeffs[k] = coeffs[k - 1] + v * coeffs[k]
        coeffs[0] = v * coeffs[0]
        deg += 1
    return coeffs  # len = len(vals)+1


def theta_coeffs(pos, z, zero, one):
    """d_0..d_6 of D(theta) at position pos (11 field elements)."""
    x = [one * pos[i] for i in range(NX)]
    ym1 = [one * pos[NX + j] - one for j in range(NY)]

    px = _poly_from_roots(x, zero, one)       # degree 6
    py = _poly_from_roots(ym1, zero, one)     # degree 5

    d = [-z * px[k] for k in range(NX + 1)]   # -z * prod(theta+x)
    for k in range(NY + 1):                   # + theta * prod(theta+y-1)
        d[k + 1] = d[k + 1] + py[k]
    return d


def companion_column(pos, z, rank, zero, one):
    """Monic companion last column c_i = -d_i/d_r. Raises on degenerate lead."""
    d = theta_coeffs(pos, z, zero, one)
    lead = d[rank]
    if lead == 0:
        raise WalkSingularity(STATUS_THETA_LEAD_DEGENERATE)
    return [-d[i] / lead for i in range(rank)]


def _right_mul_forward(W, col, a, rank):
    """W <- W * (I + C/a), row-wise O(r^2). C: subdiag ones + last col."""
    for row in W:
        dot = sum(row[k] * col[k] for k in range(rank))
        new_last = row[rank - 1] + dot / a
        for j in range(rank - 1):
            row[j] = row[j] + row[j + 1] / a
        row[rank - 1] = new_last


def _right_mul_inverse(W, col, a, rank):
    """W <- W * (I + C/a)^{-1}, row-wise via alpha_j + beta_j * t."""
    for row in W:
        w = row[:rank]
        alpha = [None] * rank
        beta = [None] * rank
        alpha[rank - 1] = w[rank - 1] * 0
        beta[rank - 1] = w[rank - 1] * 0 + 1
        for j in range(rank - 2, -1, -1):
            alpha[j] = w[j] - alpha[j + 1] / a
            beta[j] = -beta[j + 1] / a
        Sa = sum(alpha[k] * col[k] for k in range(rank))
        Sb = sum(beta[k] * col[k] for k in range(rank))
        den = 1 + Sb / a
        if den == 0:
            raise WalkSingularity(STATUS_INVERSE_SINGULAR)
        t = (w[rank - 1] - Sa / a) / den
        row[rank - 1] = t
        for j in range(rank - 2, -1, -1):
            row[j] = alpha[j] + beta[j] * t


def apply_axis_step(W, pos, axis, sign, z, rank, zero, one):
    """One +-1 step along an axis. Updates W and pos in place."""
    if axis < NX:                       # numerator x_i
        if sign > 0:
            eval_pos = pos
            a_int = pos[axis]
            inverse = False
        else:
            eval_pos = pos[:]
            eval_pos[axis] -= 1
            a_int = pos[axis] - 1
            inverse = True
    else:                               # denominator y_j
        if sign < 0:
            eval_pos = pos
            a_int = pos[axis] - 1
            inverse = False
        else:
            eval_pos = pos[:]
            eval_pos[axis] += 1
            a_int = pos[axis]
            inverse = True

    if a_int == 0:
        raise WalkSingularity(STATUS_ZERO_AXIS_DENOMINATOR)

    col = companion_column(eval_pos, z, rank, zero, one)
    a = one * a_int
    if inverse:
        _right_mul_inverse(W, col, a, rank)
    else:
        _right_mul_forward(W, col, a, rank)
    pos[axis] += sign


def apply_trajectory_step(W, pos, dirv, z, rank, zero, one):
    """T(v, t): decompose t into unit axis steps (level desc, axis desc)."""
    max_abs = max(abs(int(d)) for d in dirv)
    for level in range(max_abs, 0, -1):
        for axis in range(NSHIFT - 1, -1, -1):
            if abs(int(dirv[axis])) < level:
                continue
            sign = 1 if int(dirv[axis]) > 0 else -1
            apply_axis_step(W, pos, axis, sign, z, rank, zero, one)


def walk(shift, dirv, z_num, z_den, N, field="fraction", dps=None):
    """Full corrected walk. Returns (W, rank, status).

    W is a rank x rank list-of-lists in the requested field.
    On singularity, returns partial W with the corresponding status.
    """
    if field == "fraction":
        z = Fraction(z_num, z_den)
        zero, one = Fraction(0), Fraction(1)
    elif field == "mpf":
        import mpmath as mp
        if dps is not None:
            mp.mp.dps = dps
        z = mp.mpf(z_num) / mp.mpf(z_den)
        zero, one = mp.mpf(0), mp.mpf(1)
    else:
        raise ValueError(field)

    rank = rank_for_z(Fraction(z_num, z_den))
    pos = start_pos(shift)
    W = [[one if i == j else zero for j in range(rank)] for i in range(rank)]

    status = STATUS_OK
    try:
        for _ in range(N):
            apply_trajectory_step(W, pos, dirv, z, rank, zero, one)
    except WalkSingularity as e:
        status = e.status
    return W, rank, status


def projective_normalize(W):
    """W / max|W_ij| as floats, for parity comparisons."""
    vals = [[float(v) for v in row] for row in W]
    mx = max(abs(v) for row in vals for v in row)
    if mx == 0:
        return vals
    return [[v / mx for v in row] for row in vals]


if __name__ == "__main__":
    # tiny self-test: shift=1^11, dir = +1 on x_0, z=1/2, N=3
    shift = [1] * NSHIFT
    dirv = [1] + [0] * 10
    W, r, st = walk(shift, dirv, 1, 2, 3)
    print(f"rank={r} status={st}")
    for row in projective_normalize(W):
        print(["%.6f" % v for v in row])
