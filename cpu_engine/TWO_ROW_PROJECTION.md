# From a full 6x6 product to projected 2x6 row propagation

## Why it is exact for a fixed row pair

Let

`P_n = A_0 A_1 ... A_{n-1}`

be the 6x6 CMF product. If only rows `r_num` and `r_den` are required, define the 2x6 selector

`S = [e_{r_num}^T; e_{r_den}^T]`

and projected state

`R_n = S P_n`.

Because every update is a right multiplication,

`R_{n+1} = S P_n A_n = R_n A_n`.

Therefore the two selected rows can be propagated directly as a 2x6 state. They are exactly the same two rows that would be obtained by constructing the complete 6x6 product and discarding the other four rows. The row projection itself introduces no mathematical approximation.

This property depends on right multiplication. If the algorithm were reformulated as left multiplication, columns rather than rows would be the independently propagated objects.

## Computational benefit

A generic full 6x6 state contains 36 entries. Two rows contain 12 entries, reducing row state by 66.7% and reducing the row-update part of a generic right multiply by a theoretical factor of three.

The end-to-end kernel cannot be expected to become 3x faster, because theta construction, companion-column formation, coordinate/axis logic, inverses and normalization remain shared overhead. The M5 benchmark indicates that these shared operations are important enough that generic 6x6 GEMM is no longer the dominant optimization target.

## What is preserved

For the chosen row pair, the 2x6 engine preserves:

- all six components of both selected rows internally at every step;
- the same forward/inverse CMF transformations;
- the same axis order and `N=0` semantics;
- the same common power-of-two normalization;
- the selected-row values produced by full 6x6 right-product propagation, subject to identical finite-precision primitives/order;
- ratios whose needed values are retained from the selected rows.

The current `FSDF6401` file is narrower than the internal state: it serializes only the last active-column value of the numerator and denominator row at two checkpoints.

## Information lost by the projection

### Non-selected rows

The other four rows are not computed and cannot be reconstructed from the 2x6 state. A different row pair requires a rerun or a wider/full projection.

### Post-hoc row-pair discovery

The numerator and denominator rows must be chosen before execution. A full-matrix census can search many row pairs after the product has been computed; a 2x6 fixed-pair run cannot.

### Full-matrix invariants

The projected state is insufficient for general quantities requiring the complete product, including determinant, trace, characteristic polynomial, arbitrary minors, complete singular-value/eigenvalue information, and matrix-wide norms or invariants involving the missing rows.

### Arbitrary element scans

Internally the selected rows contain six columns, but `FSDF6401` currently writes only the last active column at each checkpoint. Post-hoc ratios involving other columns are therefore unavailable from the serialized file unless the output format is widened.

### Discovery versus optimization

The 2x6 path is ideal after a fixed start and useful ratio pair have already been identified. It is not a replacement for a full 6x6 or sufficiently wide census during discovery when the interesting pair is not known in advance.

### Normalization interpretation

A common power-of-two scale across the two retained rows preserves their ratio. It does not provide the scale/history of the four rows that were never propagated. Do not infer full-matrix absolute scaling information from the 2x6 state.

### Exact reconstruction

The projected df64 engine is a screening/optimization backend. It does not replace full exact RNS/CRT or multiprecision reconstruction when those data are required downstream.

## Recommended operating model

1. **Discovery/census:** full 6x6 or a sufficiently wide projection so many candidate ratios remain available.
2. **Fixed-start optimization:** 2x6 selected-row propagation for high throughput after `(row_num,row_den)` is fixed.
3. **Exact validation:** RNS/CRT or high precision on finalists.

The 2x6 representation is therefore best understood as an exact specialization for a known observable, with deliberate information loss outside that observable.
