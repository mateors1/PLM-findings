# Rank diagnosis follow-up — 2026-09-25

This addition records the completed all-source-union bound and membership rank
separability diagnosis. It preserves the original snapshot and earlier update.
The 12 source copies are byte-identical; snapshot.json maps their original paths
and SHA256 values. SHA256SUMS.txt covers this addition, including this README
and the snapshot manifest.

Only 479/666 saved query-seed rankings are strictly separable. Even oracle
answer sizes produce fewer exact sets than the existing selector (569/666).
There are 16 gains and 106 losses; 81 of the selector's 97 failures have rank
overlap. No threshold was fitted and no policy was promoted. A separate proof
shows that adding triple/four-way unions alone cannot increase the current
exact count under the unchanged score and tie rule.

The complete per-row reports and model artifacts remain in ignored runs/ in
the source repository. This addition includes portable counts, method code,
audit receipts and identity bindings; it is not a standalone replication bundle.
The proof and its bitmask audit were written by the same agent with separate
implementations. Membership diagnosis and its auditor used separate workers.
Copied lesson links may refer to the source repository's earlier lesson paths.
No protected-test outputs or new neural measurements are included here.
