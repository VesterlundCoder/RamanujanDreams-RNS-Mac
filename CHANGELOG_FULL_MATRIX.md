# Full-matrix update

- Replaced numerator/denominator-only GPU output with full `[C][K][B][E]` matrix snapshots.
- Added 2-4 configurable checkpoints, with N=1000 required as the final checkpoint.
- CRT now reconstructs every one of the 100 matrix elements for every checkpoint.
- Added generic ordered/unordered matrix-ratio scan.
- Added absolute and relative checkpoint deltas for every candidate ratio.
- Added convergence threshold filtering before TSV output.
- PSLQ output now keeps numerator and denominator matrix coordinates.
- Preserved fail-fast 6F5 bytecode/prime integration guards to prevent placeholder science.
