# Completed composition verification — 2026-09-25

The revised shape-aware v2 contract is independently audited and accepted.
Same-shape scores are exact; source paths, selected slots and sets remain equal
across shapes. Verification includes 666 HTTP responses (222 reused,444 fresh),
666 fresh serial references and24 repeats, and three full auxiliary suites
(one reused,two fresh). Selected exact sets remain569/666. The first campaign
remains failed under its original contract; neither evidence nor gates were
silently replaced. The inference option remains disabled by default.

Eight source files are byte-identical copies, mapped by snapshot.json.
SHA256SUMS.txt also covers this README and the snapshot manifest. Full raw
responses, model files and source archives remain in the source repository's
ignored runs/ tree. Copied methods require that layout; this is not a standalone
replication bundle. Earlier evidence copies keep their historical pending status.

The audit reconstructs saved neural vectors rather than independently generating
them and checks owned-server shutdown receipts rather than independent socket
probes. Acceptance covers the declared validation integration contract only:
no protected-test, oracle-parity, latency/SLO, throughput or energy claim follows.
