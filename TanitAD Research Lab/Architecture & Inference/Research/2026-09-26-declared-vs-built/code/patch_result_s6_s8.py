"""RESULT.md sections 6-8: G3's eval mode, the test table, the gate-chain description, the
next-launch list, and the BACKLOG note."""
import sys

P = sys.argv[1]
s = open(P, encoding="utf-8").read()


def edit(old, new, tag):
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"[patch] {tag}: anchor found {n} times")
    s = s.replace(old, new)


edit('''  code under test). A real corpus without a sidecar REFUSES; a synthetic one is skipped and says so; uncovered clips
  above `LABEL_CLOCK_MAX_UNVERIFIED_FRAC = 0.01` REFUSE (E2).''',
     '''  code under test). A real corpus without a sidecar REFUSES; a synthetic one is skipped and says so. On the TRAIN
  split uncovered clips above `LABEL_CLOCK_MAX_UNVERIFIED_FRAC = 0.01` REFUSE (`--label-clock-max-unverified` is the
  recorded operator override); on the EVAL split (E2(b)) they are EXCLUDED from the TACTICAL family only —
  `V3Dataset.tactical_excluded_sids`, read in `__getitem__` as "no record" for `lat_v7` / `lon_v7` / the tac-goal
  targets — with n, the sids and the reason stamped, and the other three families keep every clip.''', "s6")

edit('''| `stack/tests/test_label_clock_guard.py` (6) | true clock PASS with its numbers | the historical `(t + w − 1)·0.1` → `3 of 3 label reads` refused; no sidecar refused; coverage cap |''',
     '''| `stack/tests/test_label_clock_guard.py` (9) | true clock PASS with its numbers; E2(b): 1 of 3 clips unverified → PASS with `tactical_excluded_clips == 1`, the unverified clip's tactical rows all IGNORE while a verified clip keeps its 40 rows (57–96), and its other targets are still produced | the historical `(t + w − 1)·0.1` → `3 of 3 label reads` refused; no sidecar refused; coverage cap; the TRAIN policy refuses the same coverage; clearing the excluded set brings the unverified clip's tactical rows back (the exclusion is the mechanism) |''', "s7 table")

edit('''RUNNING at the time of listing (`code/run_gate_suites.sh`: one pytest process per file, RAM-gated at ≥ 9 GB to
start and killed below 8 GB — the box is shared). Trees: `C:/Users/Admin/dvb_tip_0926b` and
`C:/Users/Admin/dvb_fix_0926e`, both `git archive 5de9363`, the second overlaid with exactly the 13 files of batch 1
(overlay verified by sha256). Results land in `raw/suite_{TIP,FIX}_{stack,taniteval}.jsonl` and are appended here.''',
     '''`code/run_gate_suites.sh`, one pytest process per file with a junit XML per file, on three trees built by
`git archive da8400b` (the current tip): `C:/Users/Admin/dvb_tip_0926c` (TIP), `C:/Users/Admin/dvb_fix_0926f`
(FIX = TIP + exactly the 13 files of batch 1, overlay verified by sha256) and `C:/Users/Admin/dvb_tipred_0926b`
(TIPRED = TIP + only the two new modules and the five new test files — which new tests go RED on the unfixed
code). Every run asserts `tanitad.__file__` and `taniteval` resolve inside its tree.
**RAM floor (Master Mind, 2026-09-26 ~22:30):** the box sat below the brief's 8 GB for > 1 h from other sessions'
jobs, so this CPU-only chain starts a file only after three consecutive samples 30 s apart read ≥ 7.5 GB available
and kills it below 6.5 GB (the NavSim scorer aborts only after 120 s < 3 GB, `run_navsim_refcv6.py:317`); the
8 GB rule stays for GPU jobs. The floor is written into every result file's header. Results:
`raw/gate/{FIXNEW,TIPRED,FIX,TIP}_{stack,taniteval}.jsonl`, per-file logs and junit under `raw/gate/*_logs/`.''', "s7.1")

edit('''5. The G3 coverage decision (E2) is taken and recorded.''',
     '''5. G3: `label_clock.train.g3.g3 == "PASS"` (train 22/4,369 unverified, under the 1 % cap) and
   `label_clock.eval.g3.tactical_excluded_clips == 3` with the reason stamped (E2(b)).
6. The E1 decision is carried in argv (and, if nav compliance is ON, `--nav-compliance-tau-rad` equals the value in
   `raw/navc_tau_train.json`, derived on the FULL train split).

## 8a. BACKLOG note (for the Master Mind to add to `Project Steering/BACKLOG.md`)

- 2026-09-26 **CLOCK-25** (owner: Data). The clip-clock sidecar refused 25 clips (15 stationary, 7 K1 non-uniform
  grid, 3 K3), so their tactical labels cannot be placed within 0.05 s: 22 train clips train on the pose-dt
  fallback (under G3's 1 % cap) and 3 eval-139 clips are EXCLUDED from TACTICAL scoring (E2(b)).
  - Durable fix: clock all 25 from the raw PhysicalAI per-frame camera timestamps (not on the dev box), rebuild
    the sidecar, and drop the exclusion. Evidence: `…/2026-09-26-declared-vs-built/raw/clock_refusals_diagnosis.json`.''', "s8")

open(P, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
