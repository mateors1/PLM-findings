# Versioned repair of replication evidence verification

Declared after seed 1730 training completed and its first independent audit
failed, before any repaired audit or seed 1731 training. The scientific plan
`2026-09-25-bilinear-budget8000-replication-plan.md` remains frozen at
`750d115b2fd0daf356cb3a633248e826c264f30533ddf8e8df2ee125d2ecdc6b`.
Campaign, model recipe, inference rule, evaluator quality criteria and budget
are unchanged. This declaration versions evidence-processing code and records
its failure; it does not authorize a neural retry or redefine success.

## Observed failure and retained evidence

Seed 1730 completed its one 8000-update fit, observed terminal 23750 exit 0.
Summary `a7feed80dbab71e4f66e271a4ba5500b49a19ee068578840b5a6c32b6e8808fe`
reports 220/222 exact and a passing provisional quality gate. It remains
unaccepted pending independent verification. No seed 1731 fit has run yet.

The frozen auditor
`495db4e3d65e3893fdc68579cad62792c3ab73fd614738b7e97263935af00daa`
failed on its first actual invocation, observed exit 1. At provenance line 1016,
it passed two Python sets to the inherited `exact()` helper. That helper compares
JSON encodings, and sets are not JSON serializable. The error was
`TypeError: Object of type set is not JSON serializable`. An equivalent call
exists in the aggregate dependency-inventory check at line 1144.

The failure occurred before checkpoint payload loading. It establishes a
verification-code defect, not a scientific discrepancy in the trained child.
No audit JSON was emitted. Preserve all these original files unchanged:

- `independent-audit.py`, source hash above.
- `seed-1730/audit-stdout.txt`, SHA
  `8c5773b049fc57518f2c389bbd616046d70743704c90a8eba7f9c343732cd8fb`.
- `seed-1730/audit-execution-receipt.json`, SHA
  `d25163d1315aafbc98d9a2cc2bd7f1c7e9c8bdd1355ba27a5d4e3fab2b2cbbb6`.
- Original auditor test receipt
  `e130b1dee6d8cedae18af6abdf3cdaabeb5ac78668c8e72802357bde6b8e1ad1`
  and its 131-test source/stdout. Passing tests did not cover this production
  dependency-check path with the actual receipt schema.
- Original aggregator source
  `78cae68bdf741ec35c6deff6a62b57673092be0cf61968e88bbf40274878789f`
  and test receipt
  `0dc5c87d41db7ae3e1a3d9cd591717d0941501df201e836abcad765ead2e3f36`.
  That aggregator has not executed; its version change below is an integration
  change, not a falsely claimed failed aggregation attempt.

## Narrow auditor repair

Create `independent-audit-v2.py` and a new
`bilinear-budget8000-replication-auditor-tests-v2` directory. Use explicit set
equality or sorted JSON-compatible dependency-name lists in the seed and
aggregate inventory checks. A shared production helper may make the tested
boundary explicit. Preserve the required exact dependency inventories and all
other scientific arithmetic, gates, metadata, trace, chronology and lineage
checks. Do not skip provenance or catch-and-ignore the serialization failure.

Regression tests must invoke that production inventory check with authenticated
actual-schema seed and aggregate test receipts (17 and 4 entries respectively),
plus missing, extra and wrong-type cases. These prerequisite receipts contain
no new model predictions. Retain the previous synthetic tests, check other uses
of `exact()` for the same incompatible type, and report all new test results.
No checkpoint inspection or neural operation is part of synthetic verification.

Bind this declaration, the old failed source/stdout/receipt and the new source,
tests and test receipt. Set audit version 2. Use these distinct output paths for
both fresh seeds:

- `independent-audit-v2.json`
- `audit-v2-stdout.txt`
- `audit-v2-execution-receipt.json`

The successful execution receipt must retain the agreed seed, summary, auditor,
audit, stdout and test-receipt hashes plus observed exit 0. It must not overwrite
the retained v1 failure. Audit version 2 first verifies the already saved seed
1730 artifacts; do not train that seed again. Only after this audit succeeds may
the already planned seed 1731 fit proceed, once, with the unchanged runner.

## Aggregate integration and owner acceptance

The frozen aggregator hardcodes the v1 auditor filename and evidence paths.
Create `scripts/aggregate_bilinear_budget8000_replication_v2.py` with new tests
and receipt, routing to v2 auditor/output/receipt paths and binding this repair
provenance. Preserve all pooling, gates, observed-completion checks and exact
timestamp semantics. Retain and rerun its existing 34 tests, adding coverage of
the actual v2 evidence routing and unchanged receipt-binding requirements.
Its new tested dependency inventory contains six entries: the four equivalent
execution/test roles plus the retained v1 aggregate source and test. Test both
the original four-entry receipt and the new six-entry production contract;
do not relax exact inventory equality to subset membership.

The new aggregate output is `aggregate-v2.json`, with distinct
`aggregate-v2-stdout.txt` and `aggregate-v2-execution-receipt.json`. Its independent
audit is `independent-audit-v2.json` in the campaign root, with its own observed
execution receipt/output. New sources and test receipts must be frozen and
schema-reviewed before their actual use. Avoid circular hash dependencies by
passing authenticated auditor identities to the aggregate invocation.

Primary runner
`5da5ad02365f72f5e8a7123183d0b5d2f56e50c3538e64e2ac984e2b491230dd`,
recipe `20fad007be0d1852e86086229edc7c9c760117039f0685288d4e8e22f695075f`,
and primary test receipt
`9b01b18b15b00a98653d057f952bc8882b2d002d81b17a1cc20438e5dc1583a0`
remain unchanged. Seed 1730's checkpoint and reports remain immutable. No
scientific quality gate or declared hypothesis changes due to this repair.

Owner acceptance must follow both successful v2 seed audits and the completed
independent v2 aggregate audit. Preserve and publish the original failure,
repair declaration, versioned code/tests and successful verification separately.
The model register and final paper must distinguish checkpoint, inference,
scientific evaluator, evidence-verifier and publication identities. No protected
evaluation, serving promotion or weight-backup claim follows from this repair.
