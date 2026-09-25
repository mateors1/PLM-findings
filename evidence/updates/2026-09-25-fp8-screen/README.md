# FP8 inference screen — 2026-09-25

This addition copies the declared FP32/FP8 comparison, its portable result, the raw summary and the exact executed benchmark script from the PLM source checkout. `SHA256SUMS.txt` records their copied bytes. The [compiled FP8 plan](2026-09-25-fp8-compiled-plan.md) is a later declaration with no result in this snapshot.

The completed screen used the guided seed-1729 checkpoint on 222 validation queries. The eager Windows torchao FP8 path converted 56 decoder-block linear layers while leaving other weights and KV cache at BF16. The original checkpoint was unchanged. This was a direct native-call and fixed-batch comparison on one RTX 5070 Ti, with no protected final-test evaluation or energy measurement.

The experiment's performance gate failed: FP8 produced about 12 output tokens/s versus 148 for FP32 in serial native calls, and about 233 versus 2,969 at offline batch 32. Postprocessed macro F1 was 0.88184 versus 0.88571. Exact sets were 152 versus 151, illustrating why F1 and exactness must be reported separately. The quality percentages in the report use a different accepted PLM policy as a frontier; they are not percentages of the graph oracle.
