## 2026-09-26 — PLM servings-per-watt mapping

- Derived servings/W at batches 320 (1.117), 336 (1.165), 640 (1.237), and 656 (1.248) from measured request rate and GPU board power; added a graph separating the newer targeted run from the prior full-saturation sweep.
- The plotted serving count is one completed full decode per capped 16-ID response. Decode did not stop at ID 16; batch-640/656 power comes from the earlier run with device isolation unestablished.
