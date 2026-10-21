# Portable RNS reference and capacity bound

Meijer modular helpers are separated into meijer_cmf/rns_reference.py. They can
be tested without ramanujantools or Metal (NumPy and SymPy are required).

Prime selection now uses a conservative integer bound on coefficient and
matrix-product growth. It avoids floating logarithms when determining a safe
signed CRT capacity. Coefficient degrees and shapes are checked. Walks support
checkpoints at zero and reject unsupported moduli or noncanonical coefficients.
Signed CRT handles both odd and even products at the midpoint.

Dropping primes and obtaining the same reconstructed integer is labelled a
consistency heuristic, not a capacity proof. A regression demonstrates why:
an unseen integer M+1 has the same residues as 1 and can pass prefix agreement.

```bash
python -m pip install numpy sympy
python scripts/test_rns_reference.py
cmake -S . -B build-cpu -DBUILD_METAL=OFF
ctest --test-dir build-cpu --output-on-failure
```

Local validation: **6 exact-reference tests**, the existing fixed-start CPU
semantic check, and portable CMake/CTest in Debug and Release. No Metal device
was available here. The original Objective-C++ exact RNS host still has
unimplemented program/prime loaders; this patch does not turn that executable
into an end-to-end research release. macOS CI retains its host/Metal compilation
check; actual Metal execution and complete loader integration remain release gates.
