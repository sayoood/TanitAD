# SPEC addendum A6 — the DEPLOYABLE time-localised nav, soft rule, on a denser held-out capture

**Written 2026-10-04 by the Master Mind after A5 was scored (RESULT_A5.md).** POST-HOC: A5's reported arm T2 FAILED
its bar on criterion 4 only (seed-1 turn ΔADE −0.204 [−0.447, +0.028]); its secondary arm T3 cleared all four
criteria; A5's post-hoc finding is that 54 % of T2's turn gain sits on 7 windows where `nav_command` had SUPPRESSED a
turn (4 "curve, not a turn", 2 "contested turn (obstacle pass)", PI rule 2026-08-29) while `nav_30s` still lists it.
A real navigation system does not announce a road curve or an obstacle pass, so that part of the gain is information
deployment would not have. A6 fixes the arm that IS deployable, before any A6 number exists. A5's verdicts stand as
registered (T2 FAILED); nothing here re-judges A5.

## Nav definition (literal)
* **announced(entry, record)** = the `nav_command` builder's own rule applied to that entry as if it were the next
  manoeuvre (turn vs curve vs contested/obstacle-pass, the PI 2026-08-29 suppression included). Implement it by
  calling or mirroring the builder's function; do not re-derive thresholds.
  **Gating control G1 (must pass before any arm is scored):** on `entries[0]` of every record (train 4,572 + eval 147),
  announced(entries[0]) must reproduce the shipped `nav_command.token` side on ≥ 99 % of records, and on 100 % of the
  records whose `nav_command.reason` names a suppression. If G1 fails, A6 stops and reports.
* **nav_ann (H = 6.0 s):** A5's nav_tl rule (first unfinished entry, start ≤ 6.0 s ahead or under way) restricted to
  ANNOUNCED entries; a non-announced entry is skipped (the next announced one is considered).
* Clock, join and K1/K2 exactly as A5 (`code/nav_tl_a5.py`), re-run as controls.

## Windows (a denser capture of the SAME 139 held-out eval episodes)
* Every eval window whose GT 6-s heading change is ≥ 30° (A5/SPEC §5's GT-turn rule) at stride 1, plus a seeded
  (seed 0) random sample of the remaining windows of equal count; sampler seeds 0 and 1 (replicate). Capture with the
  package's `run_route.py` on Thor's GPU under `flock /home/nvidia/refcv7_post/thor_gpu.lock`.
* ⚠️ Same 38 turn episodes as A5: this narrows within-episode noise only. The claim it can support is "confirmed on a
  denser capture of the same held-out episodes", not "replicated on new episodes". The A5 windows are reported
  separately as a subset.

## Arms (scored on the new capture; nothing fitted on EVAL)
| id | rule | status |
|---|---|---|
| **T3a** | A5's T3 (nav-compliance term ×10, reading (a)) with nav_ann | **REPORTED ARM** |
| T3 | A5's T3 with A5's nav_tl (all entries, incl. non-announced) | foil |
| T2a | A5's T2 (hard nav-side filter) with nav_ann | secondary |
| T3a-c | CONTROL: T3a with a seeded derangement of clips (A5's T2c construction) | must NOT pass the bar |

## Bar
SPEC §5's four criteria, unchanged; paired episode-cluster bootstrap vs V0; plus the fraction of the B1t bound
captured, and the four metric families in the §2.1 format.

## Reading rule (fixed now)
* T3a passes, T3a-c fails ⇒ a deployable time-localised nav lifts route following at inference on refcv7 as trained
  (confirmed on denser windows of the same episodes); it ships as an opt-in inference rule and L2 enters refcv8.
* T3a fails while T3 passes ⇒ the inference gain needs non-announced turns (curves, obstacle passes) — information
  deployment lacks; nothing ships at inference; L2 still enters refcv8 as a training input with announced entries only.
* Both fail ⇒ A5's T3 result was within-episode noise; route following needs the training levers (L1 + selector).
* T3a-c passes ⇒ the instrument is broken; nothing from A6 is quotable.
