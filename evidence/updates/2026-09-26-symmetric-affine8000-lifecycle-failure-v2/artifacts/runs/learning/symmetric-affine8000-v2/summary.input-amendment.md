# Symmetric affine pilot: selected-partition input amendment

Applies to `symmetric-affine8000-v1` before any candidate execution or real-data
preflight. Original plan remains immutable, SHA256
`d102c6d00ab536b89c0a0021097976fb997831184a4e4c2db82d172a4c61a19a`.
This amendment changes input plumbing only. Scorer, scale, original parent,
seed, training membership, query order, optimizer, budget, final endpoint and
fixed comparison gates are unchanged. No new candidate result informed it.

## Why the inherited loader cannot be used

Independent source review of the pinned runtime archive found that
`load_inference_runtime` calls corpus validation and `load_corpus_records`,
which decode all records before constructing train/validation/test partitions.
The old primary preflight also opens whole-corpus files to hash bytes, although
it does not decode them. Both violate this pilot's stricter unopened-test rule.
This is not evidence that prior runs trained or selected on test labels.
Do not rewrite those earlier runs or infer leakage from materialization alone.

## Required adapter

Introduce input identity `plm-authenticated-partition-membership-v1`. Construct
only the original ordered 1637 train and 222 validation queries from the
hash-authenticated accepted `bilinear-budget8000-v1` train-membership artifact
and parent/validation response evidence. Verify their accepted summary, audit,
owner decision, output hashes, query identities/order, five-token prompt grammar,
dimension IDs, product-column order, membership validity and train/validation
disjointness. Labels supplied to optimization come exclusively from training
membership. Validation labels are used exclusively for replay/evaluation.

Re-encode each query as the original five-token prompt, ascending unique target
membership and EOS, with -100 labels through ANSWER and supervised targets/EOS.
Record the adapter identity, exact evidence hashes, derived ordered partition
hashes and encoding rule. These are derived membership records, not byte-exact
copies of the original serialized corpus rows. Balanced BCE uses product
membership, so target ordering is irrelevant; prove both loss and score-gradient
invariance to target permutation with the archived objective on synthetic inputs.

Load the original parent model/checkpoint separately using the authenticated
archived model implementation. Preserve all existing checkpoint, model-config,
training-objective, global-step, vocabulary, corpus/split and original training
identity checks. Carry corpus/split identities from the authenticated accepted
evidence and checkpoint, without re-splitting or reconstructing protected rows.
Record this as inherited authenticated identity, not a fresh whole-corpus audit.
Exact original-parent replay, initial loss and original 93-state checks remain
mandatory. Runtime device, dtype and numerical settings are unchanged.

Production and independent audit may read/hash an explicit allowlist of required
source/config archives, vocabulary/config metadata, original/child checkpoints,
accepted train/validation evidence and their own provenance/test artifacts.
Never open corpus records, graph database or protected partitions, even for
hashing. Do not recursively traverse historical input manifests: retain their
historical identities as such, and separately record dependencies actually
revalidated now. No graph-backed answer lookup or new label generation.

## Admission and evidence

Both primary runner and independent auditor bind this amendment and adapter
identity before production. Their recipes, readiness contract and test receipts
must agree on the input contract. Reject attempts to invoke the old whole-corpus
preflight or runtime loader. Synthetic tests must make corpus/graph/protected
loaders fail if called, verify original query/membership order and first-five
prompt/mask construction, and reject malformed or swapped partitions. No actual
preflight or candidate execution until independent review confirms the adapter.

Preserve the pre-amendment source/test identities and their synthetic receipts
as development history. They were never used for candidate training. Later
freeze the revised implementation and tests under new hashes; no training retry
or evaluator-gate change is authorized by this amendment.
