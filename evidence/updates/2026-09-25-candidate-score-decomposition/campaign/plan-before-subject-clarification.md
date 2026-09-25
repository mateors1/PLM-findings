# Saved candidate pools and score vectors — diagnostic v1

Date: 2026-09-25. Post hoc diagnostic declared before its new measurements.
This does not reopen the failed weak-margin gate or authorize replication.

## Question and frozen inputs

The fixed coefficient-.001 screen improved selected exact answers191->194 but
reduced strict membership separation168->164. Separate changes in available
candidate sets from changes in their learned scores using only saved evidence.

Authenticate the completed campaign at
`runs/learning/weak-margin-screen-v1/summary.json`, SHA256
`3d535d1e298497609169dda8791aef3920d3d8993afab9189df359644b8c7a93`,
its independent audit SHA256
`5cf9d5b556f06f068bcfda4709d737f351260f21144bc70ec42c305feef9caab`
and separate rejection SHA256
`607b576806a618d9b658cc3213d4108d99f8eefaa61030e7b3588151018c7797`.
Bind both evaluation reports through the summary's report hashes. Authenticate
the existing auditor's source hash and verify its summary/decision chain. No
mutable latest-model discovery, original corpus loading or protected-test access.

Use all222 aligned validation queries, one training seed1729, two saved
checkpoint identities. Match subject/dimension/group, prompt IDs, expected sets,
product columns, vocabulary/corpus/split identity and inference numerical settings
across arms. There are222 distinct queries, not888 independent samples.

## Reconstruction before diagnosis

Validate all saved logits as finite exactly representable FP32 values, with
1025 columns mapped to IDs1024..2048. Validate four source paths, ten canonical
slots (four originals then six ordered pairs), sorted distinct entity IDs and
source eligibility. Rebuild original sets from source paths and each pair union;
preserve slot duplicates and order. Validate protocol termination and uniqueness
against saved prompts/IDs. Use the existing constrained-path eligibility
contract; a generated subject ID remains permitted by that protocol, whereas
teacher target sets exclude the subject. Do not silently remove entities.

Rebuild diagonal slot scores with Python `math.fsum` over canonical sorted
set IDs, select the first eligible slot in a score tie and require exact equality
with saved scores, selected slot/set and reported set metrics. No tolerance
tuning or alternative set-score definition. Expected targets enter metrics and
oracle diagnostics only, never learned selection.

## Availability and selection loss

For each arm/query record exact-answer availability among eligible original
slots and among all ten slots. Let A_q indicate that some eligible slot equals
the teacher set and E_q that the learned selected slot equals it. Record
R_q=A_q-E_q, the recoverable exact-selection miss. Every query falls in exactly
one state: selected exact, exact available but missed, exact unavailable.

Also record whether the union of all four eligible source sets contains every
true entity. Lack of this containment proves no union of those sets can be
exact. Containment does not prove any of the ten available slots is exact:
extra entities and required higher-order combinations can prevent that.
This is a diagnostic ceiling, not a deployable oracle selector or new policy.

Report counts and macro selected F1 by arm and COLOR/TYPE_single/TYPE_dual.
Include the3x3 transition table between the arms' three availability states,
and the19 changed-exact queries (11 gains,8 losses) with availability transitions.
Do not extrapolate these observed change counts to other seeds.

## Fixed2x2 saved-score substitution

Let P_c/P_t be the saved control/treatment candidate pools and z_c/z_t their
saved prompt-only membership vectors. For each query compute all four cells:

    selected(P_a,z_b) = first argmax over eligible S in P_a of fsum(z_b[S])

Both diagonal cells must exactly reconstruct their original choices. Off-diagonal
cells are offline rescoring of fixed pools with the other checkpoint's saved
vector. Do not rerun guidance, generation or the neural head; do not combine
vectors or introduce per-query coefficients. Save all10 scores, chosen slot/set,
exactness and set precision/recall/F1 for every cell/query. Report macro metrics
and group counts over222 queries for each cell.

For exact counts and macro F1, report the two ordered decompositions of the
diagonal change Q_tt-Q_cc:

    (Q_tc-Q_cc) + (Q_tt-Q_tc)  [pool first, then score]
    (Q_ct-Q_cc) + (Q_tt-Q_ct)  [score first, then pool]

Also report interaction Q_tt-Q_tc-Q_ct+Q_cc. This conditional, saved-output
factorial analysis is not causal module attribution: the head participates in
first-choice guidance, parameters are shared and regenerated pools could change
under a real head swap. No selection of the best cell for deployment follows.

## Evidence and acceptance

Use a stdlib-only runner with focused synthetic tests covering unavailable vs
missed exact answers, higher-union containment, duplicate/tied slots, masks,
query misalignment, FP32 validity, diagonal corruption and path reconstruction.
No Torch, GPU, training, model inference, serving or threshold search.
Freeze plan/script/input hashes, preserve all per-query cells and input identities,
refuse overwrites, and rehash inputs after execution. Primary acceptance stays
false until an independent saved-evidence reconstruction and separate decision.
The independent implementation must not import the primary scoring/reduction
helpers. It can use exact rational sums of saved binary floats as a cross-check.
Its audit authenticates old neural outputs; it does not regenerate or prove them.

The accepted application family,27-checkpoint inventory and protected-test
boundary stay unchanged. A diagnostic can explain where candidate availability
or selection limits a saved result without establishing a better model or the
cause of the training trajectory.
