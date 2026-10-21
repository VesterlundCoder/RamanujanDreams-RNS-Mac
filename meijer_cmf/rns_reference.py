"""Portable exact modular helpers; no ramanujantools or Metal dependency."""
import math
import operator
import numpy as np
import sympy as sp

# ------------------------------------------------------------ prime sizing
def gen_primes(k, below=2**31):
    """k largest primes below 2^31 (fit uint32; products fit uint64)."""
    if not isinstance(k,int) or k<1 or not (1<<30)<below<=1<<31:
        raise ValueError("Expected a positive count and 31-bit prime range")
    out = []
    cand = below - 1
    if cand%2==0: cand-=1
    while len(out) < k and cand>=(1<<30):
        if sp.isprime(cand):
            out.append(cand)
        cand -= 2
    if len(out)!=k: raise ValueError("Not enough primes in the 31-bit range")
    return out


def required_primes(coeffs, m, deg, depth, margin=8):
    """Conservative integer growth bound for selecting 31-bit moduli.

    Bound each entry of M(t) by (deg+1)*maxcoeff*max(t,1)^deg, and each
    product step by m times that value. Integer bit lengths round every
    factor upward; two extra bits reserve a strictly sufficient signed range.
    """
    if not all(isinstance(x,int) and x>=0 for x in (depth,deg,margin)) or m<1:
        raise ValueError("Invalid capacity-bound parameters")
    if len(coeffs)!=m or any(len(row)!=m for row in coeffs):
        raise ValueError("Expected a square coefficient matrix")
    if any(len(entry)>deg+1 for row in coeffs for entry in row):
        raise ValueError("Coefficient degree exceeds the declared bound")
    coefficients=[operator.index(c) for row in coeffs for entry in row for c in entry]
    maxcoeff=max((abs(c) for c in coefficients),default=0)
    # An integer upper bound on each factor's binary exponent. No floating logs
    # or heuristic agreement test is used to certify the modulus capacity.
    bits=sum((m*(deg+1)*maxcoeff*max(t,1)**deg).bit_length() for t in range(depth))
    return max(4,(bits+2+29)//30+margin)



# ------------------------------------------------------------ RNS reference walk
def encode_residues(coeffs, primes):
    """coeffs[i][j] (Horner ints) -> uint32 residue tensor [K][E][deg1]."""
    m = len(coeffs)
    deg1 = len(coeffs[0][0])
    K = len(primes)
    out = np.zeros((K, m * m, deg1), dtype=np.uint64)
    for k, p in enumerate(primes):
        for i in range(m):
            for j in range(m):
                for d, c in enumerate(coeffs[i][j]):
                    out[k, i * m + j, d] = c % p
    return out


def walk_rns(coeffR, primes, m, depth, checkpoints):
    """Vectorized-over-primes modular walk.  Bit-exact mirror of the Metal
    kernel: P <- P * M(t), t = 0..depth-1, snapshot at each checkpoint."""
    if depth<0 or m<1 or len(set(primes))!=len(primes) or not primes:
        raise ValueError("Invalid walk dimensions or duplicate primes")
    if any(not 2<=p<(1<<31) or not sp.isprime(p) for p in primes):
        raise ValueError("Expected prime moduli below 2^31")
    if any(c<0 or c>depth for c in checkpoints) or len(set(checkpoints))!=len(checkpoints):
        raise ValueError("Invalid or duplicate checkpoints")
    coeffR=np.asarray(coeffR)
    if not np.issubdtype(coeffR.dtype,np.integer):
        raise ValueError("Expected integer coefficient residues")
    if coeffR.ndim!=3 or coeffR.shape[:2]!=(len(primes),m*m) or coeffR.shape[2]<1:
        raise ValueError("Expected coefficients[K,m*m,degree+1]")
    if any(int(v)<0 or int(v)>=primes[k] for k in range(len(primes)) for v in coeffR[k].flat):
        raise ValueError("Coefficients must be canonical residues")
    coeffR=coeffR.astype(np.uint64,copy=False)
    K = len(primes)
    pv = np.asarray(primes, dtype=np.uint64)
    P = np.zeros((K, m, m), dtype=np.uint64)
    P[:, range(m), range(m)] = 1
    snaps = {}
    cps = sorted(checkpoints)
    if cps and cps[0]==0:
        snaps[0]=P.copy()
        cps=cps[1:]
    ci = 0
    deg1 = coeffR.shape[2]
    for t in range(depth):
        tmod = np.uint64(t) % pv                      # [K]
        val = coeffR[:, :, 0].copy()                  # [K][E]
        for j in range(1, deg1):
            val = (val * tmod[:, None] + coeffR[:, :, j]) % pv[:, None]
        M = val.reshape(K, m, m)
        newP = np.zeros_like(P)
        for z in range(m):
            newP = (newP + P[:, :, z][:, :, None] * M[:, z, :][:, None, :]) \
                % pv[:, None, None]
        P = newP
        while ci < len(cps) and cps[ci] == t + 1:
            snaps[cps[ci]] = P.copy()
            ci += 1
    return snaps


# ------------------------------------------------------------ CRT reconstruction
def garner_crt(residues, primes):
    """Signed-centered CRT lift of one integer from residues (Python ints)."""
    if not primes or len(residues)!=len(primes) or len(set(primes))!=len(primes):
        raise ValueError("CRT needs matching nonempty arrays and distinct primes")
    x = 0
    mod = 1
    for r, p in zip(residues, primes):
        if not 2<=p<(1<<31) or not sp.isprime(p) or not 0<=int(r)<p:
            raise ValueError("Invalid prime or noncanonical residue")
        t = ((int(r) - x) * pow(mod, -1, p)) % p
        x += t * mod
        mod *= p
    if 2*x >= mod:
        x -= mod
    return x


def reconstruct_state(snap, primes, m):
    """First column of the product matrix as exact integers (s0, s1, ...)."""
    return [garner_crt([snap[k, i, 0] for k in range(len(primes))], primes)
            for i in range(m)]


def crt_capacity_ok(snap, primes, m):
    """Heuristic agreement check only; this cannot certify unknown magnitudes.

    CRT aliases differing by a multiple of the full modulus can pass this test.
    Use the integer coefficient bound when choosing the prime count.
    """
    full = reconstruct_state(snap, primes, m)
    part = reconstruct_state(snap[:-2], primes[:-2], m)
    return full == part, full


