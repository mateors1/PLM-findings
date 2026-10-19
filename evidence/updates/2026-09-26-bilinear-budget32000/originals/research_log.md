## 2026-09-26 — Audit repair verifies the second 32000-update result

The versioned evidence repair is sealed at readiness SHA256
9df87cdafc02a3449b3b78d9bf48be93f0840da4ba64b35c21c09c286c96a69a.
Root independently reran all 120 auditor tests (terminal 9e1bc0); the seal
rechecked all 22 original dependencies and 19 unchanged scientific functions.
The actual CPU re-audit of the saved seed-1730 checkpoint passed (aa9ccc), audit
83d48edb5e78ee66eeab4c9e87970b882f8d0783c5829a07b8b9cd56e5ca728d,
with observed terminal receipt d84131fe3fce048141347d4e415889664cb5c7b8bf92073e29634448c77e31ed.

The earlier provisional 222/222 is now independently verified: no false
positives or negatives, all 103 COLOR, 51 single-TYPE and 68 dual-TYPE sets
exact. Two dual-TYPE sets improve over the matched 8000-update control, with
no losses. No training was repeated and the original failure remains intact.
This is exact agreement for one seed on the fixed validation split, not a
protected-test result or a completed three-seed campaign. Seed 1729 still
misses four validation sets; the all-three rule is unchanged.

Seed 1731 passed the repaired predecessor admission without importing Torch.
After the serving owner acknowledged a fresh GPU reservation, root observed no
Python/LM Studio process and launched the declared continuation once. Desktop
GPU usage remained resident; this run makes no isolated-energy claim.

## 2026-09-26 — All three 32000-update fits pass the fixed quality gates

Seed 1731 completed once (session 19740, observed terminal e71471), with final
checkpoint ceb17a1e1ac92e1dbc44656c05132b518032dec6bf1ac4b4019de5dc5dbabf74.
Its CPU audit passed (eeac8c), followed by the aggregate (a94246) and separate
aggregate audit (1e1ed8). The owner decision accepts the evidence under the
original all-three rule: decision-v2.json SHA256
4a508abaf3fac7e2e678cd5604f73636241aabcb18c876076c134468257e44af.

The matched exact counts are 216 -> 218, 220 -> 222, and 214 -> 220.
There are ten gains and no losses, all gains on dual-TYPE queries. Aggregate
exactness is 660/666 seed-query outcomes versus 650/666; these reuse 222
validation queries, so they are not 666 independent examples. All six remaining
nonexact outcomes are dual-TYPE (four from seed 1729, two from seed 1731).
Aggregate F1 is 0.9999637956903659, with seven false negatives and one false
positive. Longer optimization helps this frozen-feature scorer across all
three declared parents, but does not eliminate seed sensitivity or establish
protected-test generalization. No model or serving promotion is made.

The GPU reservation was released after observed completion. Next, export bounded
loss/diagnostic evidence and consolidate all three checkpoint identities, then
publish the complete campaign including the preserved audit failure and repair.
Before another intervention, diagnose the remaining saved errors; do not tune
thresholds on validation or open the protected partition.

### Bounded 32000-campaign evidence export

The reviewed v2 exporter completed (349cfe) and root independently verified all
152 output hashes, reconstructed all 96000 training updates from48chunks and
all9822 diagnostic rows from102chunks. Largest file is470615bytes. Export
review SHA256 ce95040b1d223bf6f4f195c7af169afcd99d481bd34a6aa2ef3497739c55fd4f.
The additive41checkpoint inventory inherits38unchanged identities and freshly
hashes only the new3weightfiles; it does not imply previousweights were reread
or anyweights were uploaded. The publication handoff preserves failed1730audit
and evidence repair; canonical synthesis is coordinated separately with the
paper owner so the in-flight earlier revision is not overwritten.
