
<!-- S4-RETIER-EXECUTOR-SHIPPED-DRY-RUN-VERIFIED-2026-09-22 -->

### ⭐ 2026-09-22 — the §4 pass now EXISTS as a shipped, md5-verified, dry-run-by-default executor, and both of its refusals are proven to fire

MEASURED by me, read-only on Thor, nothing written to Thor or HF
(`TanitAD Research Lab/Architecture & Inference/Research/2026-09-22-s4-retier-executor/`).
Completes the preparation begun in `b513a26` (premise) and `cb4fec6` (mechanism → PI item 19).

`eval/s4_retier.py` shipped to Thor, **md5 `f1678bfe31580c53863add8c88453bf1` verified on both
ends**. ⛔ **Dry run is the DEFAULT**; `--apply` is explicit.

**THE PLAN, computed on live data (4,587/4,719 at the time):**

| | |
|---|---|
| flagged entries in the manifest | **103** |
| selected for re-tier | **80** — every clip carrying the *"near path unlabelled…"* flag |
| **refused** | **0** |
| compliant share under (a) | **exactly 1.0** — not an estimate |
| (c) withheld share | min **0.100077** · median **0.137173** · mean **0.172167** · max **0.460268** · **0** above 0.5 |

The other 23 flagged clips are *"path untestable (parked / stopped ego)"* and are correctly **not**
selected — they carry no `path_classes`, and the ruling's scope excludes them.

⭐ **THE PREMISE IS RE-ASSERTED PER CLIP, NOT INHERITED FROM `b513a26`.** Each selected clip must
have **zero** NON_DRIVABLE path cells and **zero** cells outside `{0} ∪ ROAD ∪ NON_DRIVABLE`; a clip
failing either is **REFUSED, never re-tiered**, because the flag's own definition would have been
violated. 0 of 80 were refused — and the refusal path exists rather than being assumed unnecessary.

**BOTH REFUSALS PROVEN TO FIRE, AND ONE CONTROL PROVING THEY ARE NOT MERELY ALWAYS-ON:**

| check | result |
|---|---|
| `--apply` before the DONE marker | **REFUSED, exit 5** — one pass, afterwards, or the corpus is split into two populations judged by two rules |
| `--option C` without the PI's word | **REFUSED, exit 4** — C deletes published files; the §4 ruling authorises a judgement, not a deletion |
| `--option C --i-have-the-pi-ruling C` | **ADMITTED, exit 0**, still dry-run, still writes nothing |

⛔ That third row is the discriminating control. A guard that refuses under every condition proves
nothing; this one refuses for the stated **reason** and admits when the condition is met.

⭐ **PROVENANCE IS DESIGNED IN, NOT ADDED AFTER.** A re-tiered clip KEEPS its `flag`,
`failed_checks` and `path_classes` and GAINS a `retier` block naming the rule, the ruling, its
compliant share and its per-clip withheld share. **A clip judged by rule (a) must stay
distinguishable from one that passed outright**, or the corpus silently loses which rule judged it —
and that per-clip withheld share is exactly what (c) requires published beside every number.

⚠️ **The HF commit half is deliberately NOT wired yet** (`--apply` returns 6 with that stated). The
operations differ per layout option, so they are written **once**, for the option the PI picks —
rather than writing three and executing one. Everything up to that point is done and verified.

**Tracking:** 77 clips at 95.19 % → **80** at 97.2 %, against the prereg's projected **78** at
completion. The (c) figures moved as expected with the extra clips (median 0.1372 → 0.137173).
