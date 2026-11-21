
---

## §C addendum — item 17, the guard the PI chose, and three rows the 2026-09-19 rulings moved

**DataFlyWheel, 2026-09-19, after `537b028` and the landing note above. Append-only.** Package:
`TanitAD Research Lab/Data Engineering/Research/2026-09-19-pi-queue-audit/` (revision section in
`RESULT.md`; `raw/evidence_rev2.json`).

| item | verdict | evidence |
|---|---|---|
| **17** — clip ids in banked records | **CLOSED-BY-RULING** | PI 2026-09-17: *"Its hygiene and reproducibility."* ⇒ option 2: the banked record is **left alone**, and `tools/clipid_scan.py` stops the count growing (:802-814). No history rewrite is proposed. |
| **3**, **5** — the strategic vocabulary; the `g_str` retrain | **CLOSED-BY-RULING — deferred**, not overtaken | PI 2026-09-19: *"the strategic layer will be only switched off temporarily until we proved that both tactical and operative layers are driving with high quality."* Both return when the strategic family resumes; the corpus gap (6 of 15 tokens) is unchanged. |
| **13** — refcv6 needs a pod | **CLOSED-BY-RULING** — reinforced | 2026-09-19: *"5. we will do it later"* — the pod is parked until the dev-box checklist is complete. |

### The guard, run on tip `07541b7`

| half | how read | at the ruling `9d19020` | at `07541b7` | grew? |
|---|---|---|---|---|
| UUID (default) | the CLI against `tools/clipid_baseline.json` | 120 in 27 files | **120 in 27 files** | **`NO-GROWTH`**, exit 0 |
| prefix, corpus list (4,719 ids) | the tool's own per-file counts, both trees | 530 | **530** | **0 files** |
| prefix, full index (306,152 ids) | the same | 870 | **870** | **0 files** |

⚠️ **The prefix half has no stored floor.** `clipid_baseline.json` records **0 prefixes for every
file** — it was written without `--clips` — so running `--clips` against it reads red on every
existing prefix. That is an unrecorded floor, **not growth**: the ruling-to-tip comparison above
shows none, and the corpus-list 530 reproduces the ruling day's own count. Recording it
(`--write-baseline --clips <list>`, naming the list) is the guard owner's task.

⇒ **Unchanged: only item 6 (optional) still wants the PI, and the v6F runbook only if v6F is to
be revived.**

<!-- PIQ-AUDIT-ADDENDUM-17-GUARD-2026-09-19 -->
