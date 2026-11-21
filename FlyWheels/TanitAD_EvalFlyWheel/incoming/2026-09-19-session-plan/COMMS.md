# COMMS — EvalFlyWheel session 2026-09-19

## PI decisions made in chat (2026-09-19, verbatim option labels)

| # | question | PI answer | consequence |
|---|---|---|---|
| 1 | Download NavSim `navhard_two_stage` (official EPDMS column) from `huggingface.co/datasets/OpenDriveLab/OpenScene` (`navsim-v2/`): scene_pickles 0.21 GB · curr_sensors 12.8 GB · hist_sensors 25.5 GB (38.54 GB, sizes MEASURED by HEAD) | **"Pickles + sensors (Recommended)"** | pickles fetched + sha256-verified 11:08 local; sensors fetching in the background (package `2026-09-19-navhard-download/`) |
| 2 | nuScenes: a human must register at nuscenes.org and accept the Terms (CC BY-NC-SA, research-only); OL planning is inadmissible as a criterion (H-EVAL-6) but is the most-quoted external number and the SAM3 paint reference the DataFlyWheel needs | **"Yes, I'll register"** | E5 turns its blocker note into a **PI ACTION LIST** (exact steps + minimal file set + sizes + landing path); agents still never register or accept terms |

⚠️ Decision 2 does **not** change H-EVAL-6 or D-BENCH-PORT: a nuScenes open-loop number stays a
**cited, non-claim-bearing** external row. What it adds is the ability to produce that row for our own
model, and one download that also serves H-SAM3-FUSION-1.

## Coordination

- Master Mind (`TanitAD project kickoff (fork 8)`) informed 11:0x local of the session, the five
  streams, and exclusive file ownership: `LEADERBOARD.md` (E4); `taniteval/adapters/navsim.py`,
  `CRITERIA_REGISTRY.json`, `tools/criteria_check.py`, `tools/tests/test_criteria_check.py` (E3).
- E1 and E3 told the navhard pickles are local (wait for `…/navhard_two_stage/EXTRACT_DONE.json`);
  E1 scope-extended to price and, if ≤ ~6 h CPU, run the navhard reference-agent EPDMS.
- E5 told the PI will register.

## Master Mind brief + re-scope (2026-09-19, cross-session)

- **Brief received:** STEP 0 ACK; task 1 old-session triage; task 2 S1 collision-gate harness
  (propose the fan source before building); task 3 E9 (`t1_eval.py` on refcv6 @416×1024); rules
  CPU-only, stage-never-commit, *"no subagent fan-out, because the PI's weekly usage is nearly spent"*.
- **Budget, MEASURED via the usage API:** weekly all-models at **87 %**, then **88 %** 45 min later.
  ⇒ E3/E4/E5 PAUSED (transcripts resumable); E1/E2 continue (the PI-authorised NavSim critical path).
- **Delivered before the re-scope:** task 1 → `../2026-09-19-old-eval-session-triage/RESULT.md`;
  task 3 → `../2026-09-19-e9-t1eval-refcv6/RESULT.md` (t1_eval UNWIRED for REF-C by construction;
  `refcv3_arm.py` runs at ≤ 19.2 s/window CPU all-in); task 2 → a fan-source proposal only
  (`TO_MASTER_MIND.md` §3), no code.
- **Re-scope (MM):** the PI's commission takes precedence; S1 and E9 move to the TrainingFlyWheel.
  E9 is handed over as DONE; the S1 proposal is handed over as input.
- ⚠️ **Channel defect:** four `SendMessage`s to the MM were held for approval and expired
  undelivered. `TO_MASTER_MIND.md` carries their content; the MM reports the PI has since been asked
  to approve messages from this session.

## Proposed register rows (for the Master Mind to land — verbatim)

| id | text | status |
|---|---|---|
| `D-NAVHARD-DL-1` | PI authorised the navhard_two_stage download in chat 2026-09-19 ("Pickles + sensors"). MEASURED: the three archives total **38,537,174,373 B** (HEAD `X-Linked-Size`: pickles 211,118,643 · curr 12,804,089,546 · hist 25,521,966,184) — ⚠️ the "31 GB" in NavSim `docs/splits.md` (quoted in the 08-27 scout) does not match the archives. Content verification is **sha256 == HF `X-Linked-ETag`**, measured equal on the pickles. 16 concurrent byte ranges sustain **12.4 MB/s** on the dev box vs 1.3–1.4 MB/s single-connection (08-29) — a ~9× lever for every future large pull | **DECIDED + MEASURED** |
| `D-NUSC-REG-1` | the PI will register at nuscenes.org and accept the Terms personally (chat, 2026-09-19); agents still never register or accept terms. nuScenes OL planning remains non-claim-bearing (H-EVAL-6, D-BENCH-PORT unchanged) | **DECIDED** |

## PI decisions, 2026-09-19 afternoon (chat, verbatim)

| id | decision | consequence |
|---|---|---|
| `D-NAVTEST-DL-1` | *"download NavSIM v1: YES, but only on the D drive"* | navtest camera shards (32, **127,882,665,618 B** MEASURED by HEAD) fetching to `D:/Archive/devbox-C/navsim/data/openscene-v1.1/openscene_sensor_test_camera/`, 6 ranges (gentle on D:, which training reads); v1.1 code pinned at `3e8291bfa89ff247231e0227778840cd0a036896` (2025-06-05), unpacked on D: |
| `D-NAVHARD-GPU-GAP-1` | *"regarding navhard inference, wait for a gap in the gpu"* | our-model navhard inference runs ONLY through a GPU-gap launcher (no training process + GPU mem < 1 GB); reference agents stay CPU (after-A8 waiter) |
| `D-NAVSIM-NO-SUBMIT-1` | *"dont submit now until I approve"* | the suite REFUSES submission commands without a named PI approval |
| `D-EVALSUITE-BUILD-1` | *"buil what is missing for a one-command suite, NavSim v1, nuScenes, and a leaderboard that regenerates itself … add high quality visualizations and reporting"* | `../2026-09-19-eval-suite-build/BUILD_PLAN.md` — six work packages W1–W6; agents launch after the weekly usage reset (94 % at 14:17 local; reset 00:00 Berlin) so the cap cannot halt the other sessions |
