## Bilinear 32000 campaign: three audited fits, improvement gate passed

The [fixed campaign](experiments/2026-09-26-bilinear-budget32000-plan.md)
adds fresh residual fits from the three original parents; it does not resume
the 8000-update children. The architecture, objective and inference remain
`plm-frozen-symmetric-bilinear-residual-v1`,
`plm-bilinear-residual-balanced-bce-v1`, and
`dense-positive-bilinear-residual-subject-excluded-ascending-fp32-b8-last6-v1`.
The scientific evaluator is `plm-bilinear-budget32000-screen-v1`.

Three final files have been produced and their identities verified by the audits:

| Parent seed | Checkpoint SHA256 | Evidence status |
| --- | --- | --- |
| 1729 | `8c2edf986f20df78e6d111f934ea5bde5b60a939df6ff92f4eeeb095c625bc27` | Completed and independently audited; 218/222 exact versus 216/222 |
| 1730 | `8f315c94ba21965e534654119a73a6abea9ef3797cc096f6997f7441c87e775e` | Completed and independently audited under evidence v2; 222/222 exact versus 220/222 |
| 1731 | `ceb17a1e1ac92e1dbc44656c05132b518032dec6bf1ac4b4019de5dc5dbabf74` | Completed and independently audited continuation; 220/222 exact versus 214/222 |

Paths are `runs/learning/bilinear-budget32000-v1/seed-<seed>/checkpoint-final.pt`.
The seed-1730 audit rejected equivalent timezone spellings in a predecessor
receipt before inspecting its tensors. The
[versioned evidence repair](experiments/2026-09-26-bilinear-budget32000-evidence-repair-v2.md)
preserves that failed audit and all original receipts. The full saved-checkpoint
re-audit passed, SHA256
`83d48edb5e78ee66eeab4c9e87970b882f8d0783c5829a07b8b9cd56e5ca728d`,
with observed terminal receipt
`d84131fe3fce048141347d4e415889664cb5c7b8bf92073e29634448c77e31ed`.
No training was repeated and the checkpoint hash is unchanged. The v2 readiness
SHA256 is `9df87cdafc02a3449b3b78d9bf48be93f0840da4ba64b35c21c09c286c96a69a`.
Seed 1731 completed once under that continuation, with audit SHA256
`59cf6a51207aea5ad86d4fc5b0d0ef9cf24ec3f9a494645620f6a68ae66a6430`.
The aggregate `8f4539583080275efe53409ce6881af3b50f2b89a092e0a21801ceafe6fd0708`
and independent audit `bacc8614e11afceee2080a3faf9850c1b70de92cf5cbd3ac912025da4edb0b3c`
support the separate decision `4a508abaf3fac7e2e678cd5604f73636241aabcb18c876076c134468257e44af`:
all three strict improvement gates pass, with 660/666 exact seed-query outcomes
versus 650/666, ten gains and no losses. All six remaining nonexact outcomes are
dual-TYPE. The same 222 validation queries are reused; no protected evaluation
or standard-serving promotion occurred. These three local artifacts supplement
the preceding 38-entry inventory. The
[additive 41-entry inventory](experiments/2026-09-26-bilinear-budget32000-model-version-inventory.json),
SHA256 `eaf8080f8ee8cc9cd7d2c55a4ddad233767943a8e2fe2e5c291e556d009c079c`,
carries those 38 prior identities forward unchanged and freshly hashes only
the three new checkpoint files. It does not reverify the old weights. The
[portable result](experiments/2026-09-26-bilinear-budget32000.json), SHA256
`b9a0682a0e63f497494fe8046b248dd7ab8d679ef44d0b8e1753affdd5b34b90`,
indexes 48 loss chunks containing every update and 102 bounded diagnostic chunks.
Hashes identify local bytes and do not establish durable or public availability.
