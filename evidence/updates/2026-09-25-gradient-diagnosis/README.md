# 2026-09-25: gradient strength and interference

Accepted bounded diagnostic on existing weights: five states, three fixed
training batches per state, 96 distinct queries. No training, model promotion,
validation quality measurement or protected-test use. Final-checkpoint inventory
stays at 25. The separate acceptance binds the unchanged primary and audit.

The auditor reconstructs saved-vector norms, cosines, ratios, batch means,
symmetric BCE/margin and radial identities. Token CE is aggregated from saved
unreduced losses; prompt BCE is authenticated only. Full-model nonmutation and
checkpoint loading are runner attestations. No independent autograd proof,
neural replay, or AdamW trajectory reconstruction is claimed.

The weighted margin is strong at initialization and locally conflicts with
existing losses at the failed final weights. Its positive radial contribution
does not make the combined gradient shrink W in every batch. This describes
local derivatives; it does not prove the cause of the training failure.

Files in snapshot.json are byte-identical copies. Raw NPZ gradients and model
checkpoint payloads remain in ignored local source runs with bound hashes;
this is not a durable backup of those payloads. Source/config archives and all
15 per-batch reports are included. The copied lesson's source-code links may
refer to the separate implementation repository. SHA256SUMS.txt covers every
addition file except itself. Older dated evidence is preserved.
