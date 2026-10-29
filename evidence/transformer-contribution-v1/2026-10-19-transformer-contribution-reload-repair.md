# Optimizer reload verification repair, before attempt 2

Attempt1 seal: b64f9d302e438bd30b91a89af0584964c302e869cf4478a95960f2ad78faeb73.
It completed seed1729 relation-only parent training and32000 adaptation updates,
saved the final checkpoint, then failed before any new-arm development evaluation.
The reference replay succeeded. No result informed a recipe change.

Failure: `torch.equal` compared the original noncapturable Adam step scalar on
CPU against the reloaded scalar on CUDA (`map_location="cuda"`). CPU synthetics
had not exercised scalar relocation. This is a verification implementation bug,
not a failed quality finding. Model-state reload had already passed.

Compare optimizer tensor values after explicit CPU relocation; do not relax
equality or modify training/scoring. Add the same short lifecycle test on CUDA.
All11 ablation synthetics now pass, including CPU and CUDA reload.

Preserve attempt1 files, failure history, final/failure checkpoint, executed
runner and test snapshots under runs/learning/transformer-contribution-v1/.
Executed runner SHA256963fd9d48872355d2ecc3469a0deea1c95872c03a44b66faa7dec99e875c7e33.
Start attempt2 in runs/learning/transformer-contribution-attempt2-v1/ with a new
seal. Repeat the full fixed campaign from its declared initial states, rather
than resume or choose a result. No change to seeds, losses, budget, labels,
thresholds, numerical settings or protected-test closure. Attempt1 compute is
additional failed-execution cost and is not part of attempt2's per-fit budget.
