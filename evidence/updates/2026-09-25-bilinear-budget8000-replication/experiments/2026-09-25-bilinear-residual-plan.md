# Frozen-parent symmetric bilinear residual screen

Declared 2026-09-25 before implementation execution or new measurements. Pokemon
TYPE/COLOR SAME only. Run ID `bilinear-residual-refit-v1`; this is a single-seed
architecture/training screen, not a protected evaluation or serving integration.

## Question and parent

Test whether cross-coordinate interactions improve dense membership prediction
under a fixed training budget. The two W-only refits obtained122 and128 exact
validation sets versus112 for the original dense head and201 for the stronger
eight-branch selector. The diagonal-feasibility diagnostic was inconclusive;
it does not establish that more capacity is necessary. This intervention is a
separate hypothesis, not a consequence proved by that diagnostic.

Start from original seed1729 checkpoint
`e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1`,
never from either rejected refit child. Inherited config SHA256
`6e0314a34ddec1ae31eb1b591c14e289339b950a1894f7e66f29ecc3a1508148`;
parent steps2000. Authenticate the accepted mean-refit chain:
summary `0457ac78065c55b4786a436d436c4de6ce2cbad510056a1d9951aa744ab44a61`,
audit `6535d9e85920c1c0e316ff0f9f60f391629c1c74be90d095759d5954692cf3bc`,
decision `da0bbc0bd9a1284c6dae6f072e46bcddf30b0c13cdd0a81e34328768042e90a6`,
runner `79299541b9ccf3cebe1562194a3a9e85a1027bf69e0347701d92eedb8b7ee63c`.
Reuse its authenticated width-eight comparator and original source/config ZIPs:
`1d74e018836cc4e87a8eb46428c0d710492b8dd90ba3b11231115711bcde5376` and
`51f06619d9b2b37d47dc8a3f7f84de1d292aec36481c64d4c48d0d854e44b970`.
Extract a fresh isolated archived runtime; verify module origins and source hashes.
Do not silently replace archived model or checkpoint behavior with live code.

The frozen split SHA256 is
`b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d`:
1637 training,222 validation,191 protected queries. No protected predictions.
Training membership must reproduce the existing manifest SHA256
`0c49346f61ab689ae4c0567f88ec96ce62da1735fedc97799ef2e368542fe5ad`.

## Explicit architecture and arithmetic

Architecture ID `plm-frozen-symmetric-bilinear-residual-v1`. Keep all93 existing
state tensors byte-identical, including embeddings, decoder and original W.
After loading the original parent, attach one new registered FP32 parameter
`symmetric_bilinear_residual`, A with shape[2,256,256], initialized exactly zero.
Only A requires gradients. Index0 is TYPE token32, index1 COLOR token33; reject
unknown dimensions. No oracle attributes or group routing enter the head.

Use the exact archived detached normalization/scaling:
E=sqrt(256)*normalize(product_embeddings), E[1025,256], and corresponding U.
The new score is

    M_d = (A_d + A_d.transpose(-1,-2)) * 0.5
    z_new(s,d,i) = z_parent(s,d,i) + E_s^T M_d E_i / 16

Retain the original parent computation and operation order through the frozen
mean-refit `_head`: F.linear(E[subjects]*F.linear(U,W),E)/16. Do not fold the
parent diagonal into a new GEMM, which could change its rounding.
Compute the residual in this explicit order: construct both symmetric matrices;
stack F.linear(E,M_TYPE) and F.linear(E,M_COLOR) to shape[2,1025,256]; gather
the requested dimension/subject rows to[B,256]; apply F.linear(gathered,E)/16
to obtain[B,1025]; then add the unchanged parent logits. No output-level
symmetrization, threshold change, bias or normalization change is permitted.
Constructing transformed embeddings for all entities is allowed in the existing
transductive setup; compute membership scores only for train/validation queries.

Full A stores131072 scalar parameters. The two symmetric matrices have at most
65792 independently effective coefficients; frozen E can introduce further
dependencies. Antisymmetric components cancel from the real-arithmetic score.
The score is symmetric in real arithmetic, but swapped FP32 computations need
not be bitwise equal. State this distinction instead of silently changing the
implemented order. This family contains diagonal corrections and off-diagonal
interactions, but is still restricted by E and is not an arbitrary1025x1025matrix.

Implement as an explicitly versioned experiment-local head, without replacing
the archived decoder's forward or monkeypatching shared helpers. The inherited
forward does not evaluate the new residual; never present its output as the
child head. A reusable experiment-local factory must attach the parameter before
strict child checkpoint loading. Core config and ordinary serving remain unchanged.

## Fixed training recipe

Objective `plm-bilinear-residual-balanced-bce-v1`, evaluator
`plm-bilinear-residual-screen-v1`. Use archived `_prompt_set_loss` with positive
and nonself-negative softplus means weighted equally within each query, then
equal query averaging. Use only1637train labels; reject empty positive/negative
groups. No auxiliary loss, causal loss, margin term or validation selection.

Exactly500 full-batch updates, ordered by the authenticated train split, with
fresh archived AdamW: lr0.0003, betas(0.9,0.999), epsilon1e-8, weight_decay0,
non-fused; no scheduler, clipping, accumulation or minibatch sampling. FP32,
autocast off, highest float32 matmul precision, CUDA matmul TF32=false,
cuDNN TF32=true, seed1729, deterministic_algorithms=false, matching parent replay.
Recompute the differentiable residual each update; frozen base logits/features
may be cached. Training logits have shape[1637,1025]. Freeze source, recipe and
focused synthetic test receipt before any real model execution.

Record every pre-update loss, final post-update loss, finite gradients/parameters,
completed update count and descriptive elapsed/peak memory. On any nonfinite
value or invariant failure, retain immutable partial evidence and stop. No
post-result learning-rate adjustment, update extension or restart. Save only
the final trained checkpoint; an unsuccessful quality screen retains that child.

With E fixed, the loss is convex in A in exact arithmetic. That does not imply
500 AdamW updates find an optimum. This changes parameterization, trainable
dimension and optimization geometry relative to W-only refits. Equal optimizer
settings do not isolate a causal effect of capacity or prove equal optimization
strength. Compare the interventions faithfully under their declared budgets.

## Replay, prediction and checkpoint requirements

Before fitting, replay all222original validation logits in historical order,
B8 with finalB6. Require exact equality with authenticated width-eight values
and archived model forward, using the existing parent evaluator. Through the
actual new head path at zero A, require exact equality with all222parent logits
and strict-positive sets; record zero residual and unchanged full state.
Also require zero-A full-training loss to equal the authenticated mean-refit
initial loss0.003822767175734043 exactly. Do not relax checks on failure.

After500updates, write a full94-entry child state containing the original93
tensors and A, plus optimizer/RNG/config/identity through archived checkpoint
helpers. Config and metadata must explicitly bind architecture ID, dimension
mapping, residual shape, objective, evaluator, recipe and executed source hashes.
Parent steps2000 and new residual updates500 remain separate; child global_step500.
Reload into a fresh archived model with A attached before strict load. Validate
all metadata, all94tensor hashes and full optimizer state exactly; only A may
change from the zero-initialized state. Check A actually changed.

Run the reloaded child head once on222validation queries, same B8/finalB6,
and preserve every FP32 logit. Prediction uses nonself z>0 products sorted by
ID; no true cardinality, group routing, threshold search, truncation or fallback.
Retain empty/oversized sets. Serialization-compatible count is1..506products.
This evaluates dense set prediction, not autoregressive generation or EOS.
Standard serving must continue rejecting this new architecture/objective until
a separate reviewed integration; do not relabel it as causal-next-token-v1.

## Fixed gate, audit and publication

Require all execution/replay/frozen-state/reload invariants,222serialization-
compatible outputs, exact count>201, macroF1>=0.9799255176742276 and group exact
counts COLOR>=103, singleTYPE>=50, dualTYPE>=48. Reuse the same explicit gate as
the prior mean-refit screen. Also report the paired dense-parent comparison.
Historical mean/worst refit outcomes are context, not newly replayed controls.
No protected evaluation or seeds1730/1731 without a separate declared campaign.

Focused tests must cover dimension mapping, zero-init parity, cross-coordinate
terms, algebraic symmetry/antisymmetric cancellation, loss and subject masking,
only-A optimization, unchanged parent tensors, checkpoint/optimizer exact reload,
architecture identity rejection, split isolation, fixed gates, output bounds,
immutable artifacts and partial failures. Include a portable code fixture for
archived runtime APIs as needed; no hidden ignored run dependency in unit tests.

An independently authored auditor validates saved labels/split, selections,
logits/metrics, paired comparisons, loss/update trace, all93unchanged parent
tensors plus trained A, full checkpoint metadata/optimizer and declared gate.
It must not import the primary prediction/gate arithmetic or claim to independently
repeat CUDA training. Bind audited evidence separately from the owner's decision.

This creates a new full neural derivative if training completes. Add it to the
academic inventory even if the quality gate fails (currently29final checkpoints).
Preserve local checkpoint/sidecar hashes without claiming durable archive. Update
Lesson42, register, registry, research/devlogs and PLM-findings, then immediately
notify the existing Luna Max publication task with an immutable audited packet.
Coordinate GPU execution; LMStudio remains offline. A passing single-seed screen
permits a separately declared replication, not automatic serving promotion.
