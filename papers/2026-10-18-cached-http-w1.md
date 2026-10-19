# Stronger SQLite baseline wins the fresh matched comparison

Date2026-10-18. Campaign `cached-http-w1-v1`. Accepted local evidence; this
note is not a publication receipt. The source repository retains the complete
plan, frozen readiness, saved request traces, independent audit/review and owner
decision. No checkpoint or production default changed.

SQLite with one synchronous endpoint worker passes the declared >=10% throughput
and no-worse-p95 gate in all nine seed/repetition pairs at2/4/8/16 clients.
C1 is mixed. Paired SQLite/model throughput ranges are1.323-2.026 atC2,
1.309-1.898 atC4,1.313-1.550 atC8 and1.308-1.691 atC16. The earlier model win
against32 SQLite workers does not survive this stronger matched baseline.

352 frozen tests passed without skips. The first real campaign and independent
audit exited zero; separate review reconstructed all81,252 rows and traces,
including59,940 timed,19,980 warmup and1,332 preflight requests. Six owned servers
shut down normally. Same222 reused development queries, three seeds, CPU FP32 B1,
matching HTTP serialization and admission32; learned endpoint32, SQLite1.

Learned exact counts remain218/222,222/222,220/222; SQLite222/222 each. Both arms
meet100-ms p95 throughC16. This proves neither maximum capacity nor an energy
advantage. Six seed/query errors remain; the191 protected queries stay closed.
No model promotion. Next research priority is quality, under a new declaration.

Source report: `docs/experiments/2026-10-18-cached-http-w1-report.md`.
Portable evidence: `docs/experiments/2026-10-18-cached-http-w1-portable-evidence.json`.
Owner decision SHA256:
`b61bf10379a1a9dada7f55a774073dcc400f439b93c974df7c15ed44e113225f`.
