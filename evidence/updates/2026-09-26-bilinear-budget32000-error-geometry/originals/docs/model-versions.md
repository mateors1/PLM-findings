## Bilinear 32000 saved-error geometry: accepted descriptive diagnostic

Campaign `bilinear-budget32000-error-geometry-v1` creates no checkpoint or
inference policy. All 41 registered checkpoint identities remain unchanged.
The accepted 32000 scorer is still outside standard serving and unpromoted.

The diagnostic independently reconstructs all 666 saved validation rows:
660 exact, 4 separated-missing, 1 separated-extra and 1 overlap. Six failed outcomes
correspond to five distinct dual-TYPE queries, with FEZANDIPITI failing in two
seeds. No single threshold can recover all saved predictions for 1729 or 1731;
for 1731 the query rankings separate but their required intervals conflict.
These are label-informed descriptive bounds, not selected thresholds.

Summary `9d5ebfce23e58e5cef6fa9c9747db687ab921a9923c65ad6c81db23ae60c21d8`,
audit `78ab0f8fbbadadf59e20a6b11dac1f537a65e76f193901c5f36ad29c77674025`,
and owner decision `a9defd6e81f0902669a73307d8eaf7107e7da6a2f16ee4250d539d6916189567`
identify the accepted local evidence. See the complete report at
`docs/experiments/2026-09-26-bilinear-budget32000-error-geometry-report.md`.
An opt-in serving adapter is separately designed, not yet implemented. No
weights, training records, graph relations or protected examples were opened
by this diagnostic; no new predictions, training or model promotion occurred.
