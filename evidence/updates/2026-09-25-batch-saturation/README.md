# Offline batch-saturation follow-up — 2026-09-25

This addition preserves the declared fixed-group CUDA sweep from the PLM source
checkout. The plan, portable receipt and four raw run summaries are byte-identical
copies. `snapshot.json` maps each local copy to its source path and SHA256;
`SHA256SUMS.txt` inventories this addition.

The highest measured median was **30,980 completed output tokens/s** and
**188.3 completed requests/s** at batch 640 on one RTX 5070 Ti. Batch 656 was
level at 30,973 TPS; batch 672 fell to 15,749 TPS near device-memory capacity.
Batch 1024 passed token parity but its 136.54-second warmup was untimed.
All completed outputs matched archived serial tokens. These numbers count
generated output tokens including EOS, not prompt tokens or finished-row slots.

This is an **offline FP32 fixed-batch decoder measurement**, not throughput of
the current serial HTTP scheduler. NVML device-wide busy time does not establish
GPU arithmetic saturation, and the precise cause of the near-capacity slowdown
was not kernel-profiled. No production arrival-rate, latency SLO, matched-quality
oracle or energy-per-request result is claimed. The raw reports are recorded
execution evidence copied from the source checkout, not a new replay here.
