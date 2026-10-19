# Symmetric affine offsets: completed negative screen

Campaign `symmetric-affine8000-v3` completed 8000 real residual updates and passed
independent evidence audit. It tied the historical 8000-update bilinear control
at 216/222 exact answers and macro F1 0.9998843626799785. The predeclared strict
exactness-improvement gate failed. Evidence accepted; model not promoted.

Read [paper 31](../../../papers/31-symmetric-affine-offsets.md),
[primary summary](artifacts/runs/learning/symmetric-affine8000-v3/summary.json),
[independent audit](artifacts/runs/learning/symmetric-affine8000-v3/independent-audit.json),
[owner decision](artifacts/runs/learning/symmetric-affine8000-v3/decision.json),
[manifest](manifest.json), and [checksums](SHA256SUMS.txt).

The experiment trained fresh zero A/u/b on 1637 selected training queries, kept
93 original tensors byte-identical, and evaluated 222 development validation
queries. The final checkpoint identity is
`17d59e7aeae81a88914a523c76fd27e8fd33909a5694d28d3913a8240bc82220`.
This bundle excludes weight/optimizer payloads, raw large score reports, training
membership examples, graph databases and the whole corpus. Recorded hashes
identify excluded local dependencies; they do not establish durable backup.

Each `artifacts/` file is an exact byte copy from the source path recorded in the
manifest. Embedded original absolute paths remain historical identifiers and
are not rewritten into portable replay claims. The archive contains the executed
runner/helper/config/source snapshots, declarations, recipes, test evidence and
terminal completion receipts. No model computation is performed to build it.

V1 and V2 remain separate implementation failures before candidate quality was
measured. See [runtime failure](../2026-09-26-symmetric-affine8000-runtime-failure-v1/README.md)
and [lifecycle failure](../2026-09-26-symmetric-affine8000-lifecycle-failure-v2/README.md).
The V3 integration test used a simulated fitting trajectory; production execution
and the independent saved-evidence audit provide the separate actual-8000-update
record. The audit did not rerun optimization or neural prediction.

The result is adaptive single-seed validation evidence, with no protected-test,
oracle-parity, serving-performance, energy or generalization conclusion.
Standard serving does not execute the experimental affine scorer.

## Complete bounded exports

The [portable result](artifacts/docs/experiments/2026-09-26-symmetric-affine8000-v3.json)
binds the comparison, all 222 identical selected sets and their source hashes.
The [38-checkpoint inventory](artifacts/docs/experiments/2026-09-26-symmetric-affine8000-v3-model-version-inventory.json)
retains the earlier 37 entries and adds the trained candidate without promotion.
All 8000 training-history rows are retained as exact ordered slices:
[1–2000](artifacts/docs/experiments/2026-09-26-symmetric-affine8000-v3-training-part-01.json),
[2001–4000](artifacts/docs/experiments/2026-09-26-symmetric-affine8000-v3-training-part-02.json),
[4001–6000](artifacts/docs/experiments/2026-09-26-symmetric-affine8000-v3-training-part-03.json),
[6001–8000](artifacts/docs/experiments/2026-09-26-symmetric-affine8000-v3-training-part-04.json).
The chunks identify the original full training report; they are exports, not
replacement claims that optimization was rerun.
