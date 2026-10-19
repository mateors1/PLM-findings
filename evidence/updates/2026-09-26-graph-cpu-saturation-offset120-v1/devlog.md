## 2026-09-26 — Graph CPU saturation and +120 W offset

- Ran 13 loopback SQLite graph HTTP load levels with a 16-ID cap; all matched the frozen graph oracle, with zero failed responses.
- Concurrent throughput plateaued at 435.4–460.9 servings/s from concurrency 8 through 128. Measured CPU-package power was 104.1–110.9 W median; recorded a separate +120 W derived series.
- Added the report, portable summary, CSV and three graphs. The adjusted wattage is not a wall-power reading; the current run's power is higher than the prior clean sweep, with the cause unresolved.
- Updated the canonical paper to v0002, archived the exact v0001 bytes, and updated the paper-history evidence and PLM-findings path index.
