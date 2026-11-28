"""Report v29: the DAgger read (ROBUST, both seeds) as 7.19, in the abstract, in 8.1, 8.2 and 8.4 row 1r; the re-reads as row 1s.
python patch_v29.py <tip>"""
import sys

TP = sys.argv[1][:8]
R = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/report/"
p = R + "V8F_SCIENTIFIC_REPORT.md"
s = open(p, encoding="utf-8").read()

OLD_28 = ("It read **HARMS** at both seeds (§7.18), so the committed next is on-policy states in DAgger form at p = 0.1 (`PREREG_V8F_DAGGER`,"
          " §8.4).")
NEW_28 = ("It read **HARMS** at both seeds (§7.18), so the committed next was on-policy states in DAgger form at p = 0.1 (`PREREG_V8F_DAGGER`)."
          " That arm read **ROBUST** at both seeds (§7.19): T's lateral and longitudinal decisions on the model's own motion improve past"
          " every pre-registered bar, and the executed 6 s ADE falls by 0.54 / 0.40 m.")
OLD_ABS_TAIL = ("The cost is T0 longitudinal error, separated-worse at both seeds. ORC is not adopted; the committed next is"
                " the DAgger arm at p = 0.1.")
NEW_ABS_TAIL = (OLD_ABS_TAIL + "\n"
                "* **DAgger** (p = 0.1, two seeds): 10 % of training samples are replaced by the model's OWN rolled-out states, labelled from the"
                " human future, against the same 3,000-step fine-tune with p = 0. The committed outcome is **ROBUST** (§7.19), with all 176 controls"
                " passing. At T1-kin, T's lateral macro-F1 rises by +0.108 / +0.118 and its longitudinal macro-F1 by +0.055 / +0.081, separated and"
                " past bars that include the run-to-run replicate floor. The executed ADE6 falls by 0.54 / 0.40 m (2.28 against 2.82 m, and 2.46"
                " against 2.86 m), and S's checkpoints beat a kinematic arc at every horizon. T0 ADE6 is unchanged at seed 0 and +0.048 m at"
                " seed 1, inside the +F/2 guard. It is the first v8F training lever to clear a T1 bar on both heads. DAG is adopted as the base, and"
                " its dominance over P10 and its FLAT repeat are pre-registered and running (`PREREG_V8F_DAG_REREAD`).")

SEC = """### 7.19 DAgger: supervising T on the model's own states repairs its decisions on its own motion

* **Setup** (`PREREG_V8F_DAGGER` + Amendment 1, `2a78f059`, before any bank or run).
  * **The bank.** Frozen M2b is rolled out at T1-kin (m = 10, causal bucket, dead-reckoned nav, PLAIN) over the 4,512 TRAIN clips. Its own
    states at ticks 1–2 are kept with their own 16-row ego history and re-anchored poses, and labelled from the human future at the
    matched route point.
  * **COVERAGE:** 0.756 / 0.769 of the windows are eligible, against a bar of 0.5.
  * **DAG:** M2b fine-tuned 3,000 steps with p = 0.1 of eligible samples swapped for a bank state. The swap share reached 0.1013 / 0.1008.
  * **DAG-0:** the same argv with p = 0 (swap share exactly 0).
  * Both seeds were trained on box 2 through the launch gate, and read on the dev-box CPU.
* **The amendment.** The first read REFUSED at the bank stage, before any row: the reader read the generator's `limit_clips` 0 (no limit) as
  a limit. Amendment 2 (`7d29172b`) is post-data and control-only. It aligned that one clause with the generator, and the reader was
  re-proved (160 / 160 mutants) and run once.
* **Result [M]** (`raw/2026-10-27-dagger-run/read/DAG_READ.json`, sha256 `3393fdd5…`). Paired episode-cluster bootstrap, B = 2,000, 141
  episodes, 2,399 windows, DAG − DAG-0. A head clears at a seed iff it is separated-better with point ≥ max(2F, 2·RF). RF is the replicate
  floor: DAG-0 against ORC-0, a same-argv replicate trained by another build.

| row | tier | seed 0 | seed 1 | bar s0 / s1 | result |
|---|---|---|---|---|---|
| R-TLAT: lat macro-F1, ticks ≥ 1 | T1-kin | **+0.1082 [+0.0491, +0.1572]** (0.603 vs 0.495) | **+0.1184 [+0.0755, +0.1585]** (0.639 vs 0.520) | 0.0248 / 0.0248 | **clears at both** |
| R-TLON: lon macro-F1, ticks ≥ 1 | T1-kin | **+0.0554 [+0.0300, +0.0847]** (0.321 vs 0.266) | **+0.0806 [+0.0560, +0.1043]** (0.331 vs 0.251) | 0.0502 / 0.0278 (2·RF) | **clears at both** |
| R-ADE: ADE6, m | T1-kin | **−0.5435 [−0.7564, −0.3690]** (2.278 vs 2.821) | **−0.4046 [−0.5496, −0.2782]** (2.456 vs 2.861) | not separated-worse | held, separated-better |
| R-T0: ADE6, m | T0 | +0.0019 [−0.0230, +0.0266] | +0.0483 [+0.0178, +0.0807] | point ≤ +F/2 = 0.0604 | held |

* **Size against the noise.** R-TLAT is 4.4× / 4.8× 2F. R-TLON is 1.10× / 2.90× its replicate-floor bar, so the s0 margin is the thinnest in
  the table. R-ADE is 4.5 F / 3.4 F.
* **STRATEGIC, beside** (unpaired, within each run, not paused for v8F). S-POS against CTRV at 30 / 50 / 80 m:
  * DAG: −1.72 / −4.48 / −9.69 m (s0) and −1.87 / −4.90 / −10.33 m (s1), all separated.
  * DAG-0: separated only at 80 m at s0 (−3.89 m). At s1 it is separated-worse at 30 m (+0.81 m).
* **Reading.** DAgger does what §7.15 diagnosed was missing. T was trained only on the human's states, so on its own drifted states its
  decisions degraded; supervising it on those states repairs them. It does not buy logged-state accuracy: tick-0 F1 and T0 ADE are flat or
  slightly worse. The gain is specific to the model's own state distribution, which is the one it is deployed on.
  * T1 lateral F1 (0.60 / 0.64) is still below tick 0's (0.85). The gap narrows; it is not closed.
* **Outcome, as committed: ROBUST.** DAG is adopted as the base. The committed next:
  * T1 dominance over P10 is re-read on DAG with a registered bar;
  * FLAT's repeat is run, at its own tier and at T1-kin;
  * if the crawl rows still fail, DAgger iteration 2 follows.
  `PREREG_V8F_DAG_REREAD` (`bdf8d561`) registered these before any of their rows.

"""
ANCHOR_8 = "\n---\n\n## 8. Discussion"
OLD_81 = ("7. **The strategic level works on the model's own motion** (§7.15): S's checkpoints beat a kinematic arc at 30 / 50 / 80 m, separated"
          " at both seeds.")
NEW_81 = (OLD_81 + "\n8. **Own-state supervision repairs T on the model's own motion** (§7.19): DAgger at p = 0.1 lifts T's lateral and longitudinal"
          " macro-F1 at T1-kin past bars that include the run-to-run replicate floor, at both seeds, and cuts the executed 6 s ADE by 0.54 /"
          " 0.40 m against its own fine-tune control.")
OLD_82 = "O-RECOVER moved the lateral errors on the model's own motion but read HARMS on position (§7.18). The DAgger arm is the pre-registered next lever."
NEW_82 = ("O-RECOVER moved the lateral errors on the model's own motion but read HARMS on position (§7.18). DAgger (§7.19) lifts lat F1 at"
          " ticks ≥ 1 to 0.60 / 0.64 (DAG-0 0.49 / 0.52), separated and past its bar. The gap to tick 0's 0.85 narrows; it is not closed.")
OLD_1R = ("**pre-registered** (`PREREG_V8F_DAGGER` + Amendment 1, `2a78f059`, before any bank or run). Data library, trainer wiring and"
          " reader built and mutation-proved on the dev box; landing in progress. The box-2 run waits on the PI's go |")
NEW_1R = ("**read 2026-10-27: ROBUST** (§7.19; `DAG_READ.json`, 176 / 176 controls PASS; Amendment 2, post-data and control-only, before the"
          " one re-read). R-TLAT +0.108 / +0.118, R-TLON +0.055 / +0.081, both clearing at both seeds; T1-kin ADE6 −0.54 / −0.40 m. Adopted as"
          " the base |\n"
          "| 1s | **DAG re-reads** (zero training): (A) DAG − P10 at T1-kin on all four families; (B) DAG − FLAT at FLAT's tier and at T1-kin |"
          " A DOMINANT: ADE6 separated-better by ≥ 2F, no family row separated-worse, and a separated-better row in every family, at both"
          " seeds. B ESTABLISHED: ADE6 ≤ −2F, separated at both seeds; B at T1-kin is confounded by DAgger and is attributed only by"
          " FLAT-DAG | **pre-registered** (`PREREG_V8F_DAG_REREAD`, `bdf8d561`, before any of its rows); running on the dev-box CPU |")
E = [
    (OLD_28, NEW_28),
    (OLD_ABS_TAIL, NEW_ABS_TAIL),
    (ANCHOR_8, "\n\n" + SEC.rstrip("\n") + "\n" + ANCHOR_8),
    (OLD_81, NEW_81),
    (OLD_82, NEW_82),
    (OLD_1R, NEW_1R),
    ("**Status:** research report, v28 of 2026-10-26, results through repository tip `2a78f059`",
     f"**Status:** research report, v29 of 2026-10-27, results through repository tip `{TP}`"),
]
for a, b in E:
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)
assert all("closed loop" not in b.lower() and "drives" not in b.lower() for _, b in E[:-1])
open(p, "w", encoding="utf-8").write(s)
t = open(R + "page_template.html", encoding="utf-8").read()
assert "</script" not in s.lower()
open(R + "v8f_report.html", "w", encoding="utf-8").write(t.replace("%%MD%%", s))
print("report v29 built", len(s))
