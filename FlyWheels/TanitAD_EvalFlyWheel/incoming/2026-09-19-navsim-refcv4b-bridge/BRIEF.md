# BRIEF E2 — TanitAD → NavSim bridge: refcv4b scored by the OFFICIAL harness, with a paired vision-pure arm

**From** EvalFlyWheel orchestrator, 2026-09-19 (session commissioned by the PI in chat).
**Schema** `Project Steering/TANITAD_PROGRAMME.md` §3 (SPEC / PLAN / tests / code / raw / RESULT / COMMS).
**Package** = this folder.

## Operating rules (binding)

1. STAGE, NEVER PUSH. `git add` every deliverable into the working tree when you finish — EXACT FILE
   PATHS ONLY, never a directory (the shared index holds other agents' staged work). Do NOT
   `git commit`, do NOT `git push`, do NOT switch branches. Verify: a NEW file must appear in
   `git ls-files --stage -- <path>`; a MODIFIED tracked file needs its index blob to equal
   `git hash-object <path>` (both 40 chars, else INCONCLUSIVE). Re-verify at the END of your turn.
2. END WITH A DELIVERABLE MANIFEST — every artifact and WHERE it lives (`repo:<path>` /
   `devbox:<path>`). Mark anything that exists in only ONE place.
3. ESCALATE INTEGRATION in your report's headline. Never write "please merge" into a README.
4. QUOTE ONLY PRIMARY SOURCES. Model facts from `Project Steering/MODEL_REGISTRY.md` or raw eval JSON;
   published numbers from banked PDFs (`TanitAD Research Lab/Library/`, cite the library key).
5. FAIL LOUD, REPORT HONESTLY. UNVERIFIED beats a guess. Evidence class on EVERY number: MEASURED
   (ours + artifact path) · PUBLISHED (library key + table) · INHERITED · ESTIMATED · HYPOTHESIS.
- Research banking: cite a paper ⇒ bank it with `python tools/kb_add.py … --cited-by "<your RESULT>"`
  (into `TanitAD Research Lab/Library/`; "Research Hub" is DEAD). Verify with `--verify`.
- Keys.txt is git-ignored — never commit, print or copy tokens.

## Environment facts (MEASURED 2026-09-19 unless marked)

- Repo `D:\Projects\TanitAD` (exFAT, canonical). ⛔ Never read/write any `G:\Meine Ablage\…` path.
- TanitAD venv `C:\Users\Admin\venvs\tanitad\Scripts\python.exe`; set `PYTHONPATH=D:/Projects/TanitAD/stack`
  and ASSERT `tanitad.__file__` is under `D:/Projects/TanitAD` (an editable install can serve G:).
- NavSim venv `C:\Users\Admin\navsim\venv\Scripts\python.exe` (Python 3.9.25); devkit
  `C:\Users\Admin\navsim\devkit` @ **0a380a9**; env vars in `C:\Users\Admin\navsim\env.sh`.
- ⛔ The RTX 4060 is OCCUPIED by a live training run (`refc_v3_train.py … a8-occupancy-5k`, PIDs
  16996/21724). **CPU inference only.** Do not touch that process. Keep your RAM ≤ ~5 GB and system free
  RAM ≥ ~3 GB. Stream E1 runs the NavSim scorer on the same box — keep worker counts ≤ 4.
- cp1252 console: `PYTHONIOENCODING=utf-8`. ⛔ Never read `$?` through a pipe; assert on artifacts.

## Goal

Score a REAL TanitAD checkpoint on the official NavSim v2 harness (`warmup_two_stage`, the only split
on disk), producing the programme's first external-protocol number for its own model — honestly
labelled, with the ego-status contribution measured as a **paired arm**.

**Checkpoint** (priority 1): refcv4b — `D:\Projects\TanitAD-artifacts\refcv4b_final\ckpt_40284_FINAL.pt`
with `config.json`, `anchors.pt`, `anchors.units.json`, `README.md` alongside; registry §4.6
`refcv4b-b1-v72-40k`. (Priority 2, only after refcv4b is scored: refcv5-v2
`D:\Projects\TanitAD-artifacts\refcv5v2_final\ckpt.pt`, registry §4.8.)

**Facts read from refcv4b's `config.json` / `anchors.units.json` this session (MEASURED — re-verify):**
`--image-hw 256 640`; `horizons` `[5,10,15,20,30,40,50,60]` steps × `dt_s 0.1` ⇒ knots at
**0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0 s** (non-uniform, 6 s horizon, 117 v0-conditioned anchors × 8);
`ego.channels ["v0","a_long","yaw_rate","curvature","keep"]` at t0 with `ego_dropout 0.5` (so `keep=0`
— ego withheld — is IN-distribution); `nav_cmd_derivation` = *"v7.2 nav_command token (oracle,
provenance ego-future; allow_oracle_nav=True)"*, vocabulary follow/left/right.

## Design (use it unless source reading shows it cannot work — then say why)

- **Two processes, two venvs, one seam file.** (1) TanitAD venv: load frames + DECLARED inputs →
  refcv4b → trajectory per scene token → seam file (`raw/traj_<arm>.npz|json` keyed by token + an
  input manifest). (2) NavSim venv: a tiny agent that returns the precomputed trajectory per token
  (or the submission-pickle route + `run_pdm_score_from_submission.py` — check which fits two-stage
  scoring), scored by the official scorer. The seam + manifest IS the ego-status-enforcement
  mechanism the blocking gate `navsim.ego_enforcement` requires.
- **Frames**: the DataFlyWheel's NavSim warmup corpus — package
  `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-08-28-navsim-corpus-adaptation/`
  (RESULT.md, ADAPTER_CHANGES.md, EGO_SIDECAR.md, FRAME_DECISION.md, `code/navsim_loader.py`,
  `code/build_navsim_eval.py`, `raw/frames_provenance.parquet`). Bank at
  `C:\Users\Admin\tanitad-wt\_s2build\navsim\corpus` (417 MB; verify a sample of per-scene sha256
  against `frames_provenance.parquet` BEFORE use). Variants: `rig_clean` (176×624,
  PHYSICALAI_RIG_CLEAN, 100 % observed) and `wide` (256×640, PHYSICALAI_WIDE120, 89.29 % observed +
  mask). Determine FROM SOURCE (`stack/scripts/refc_v3_train.py`, `stack/tanitad/data/v2_dataset.py`,
  `stack/tanitad/data/calib.py`, `stack/tanitad/refs/refc_v3.py`) exactly which frame and how many
  history frames refcv4b consumes, and use the matching variant. ⛔ Never resize to make shapes fit —
  a frame-tag mismatch is REFUSED with both tags printed (ADAPTER_CHANGES §1). ⚠️ `build_navsim_eval.py`
  carries a stale `G:` sys.path line — never run it against G:.
- **Time**: NavSim gives 4 history frames at 2 Hz (Δt 0.5 s, measured by the DataFlyWheel). If refcv4b's
  native frame spacing differs, the input is out-of-distribution in time: use the closest admissible
  construction, STATE it, and if cheap add a declared sensitivity arm.
- **Ego at t0** (PI ruling 2026-09-02: *"It is allowed to use the velocity as initial measured state at
  its cycle time. It is not allowed to use the future dynamic information from the ground truth."*):
  NavSim `EgoStatus` supplies `ego_velocity`, `ego_acceleration` (ego frame), `driving_command`. Derive
  v0, a_long and curvature/yaw-rate from t0 quantities only, with the model's own caps (find
  `alat_v_floor` / `kappa_cap` in the refc code) — write the exact formulas.
- **ARMS (priority order), same checkpoint, same scenes:**
  - **A1 `ego+cmd`** — frames + declared t0 ego state + NavSim `driving_command` mapped onto follow/left/right
    (NavSim's command is a 4-dim one-hot — find its semantics in the devkit; declare the mapping,
    including what 'unknown' becomes).
  - **A2 `vision-pure`** — frames only; `keep=0`; nav withheld (`nav_valid=False` convention).
  - **A3 `ego, no cmd`** if cheap.
  A1 vs A2 on identical scenes is the paired ego-status contribution the Lab recommended
  (`TanitAD Research Lab/Opponent Analysis/Research/2026-08-28-navsim-v2-camera-lane/RESULT.md`, rec. 3) —
  no NavSim leader publishes that pair.
- **Output**: NavSim expects poses at 0.5 … 4.0 s (verify the agent's `TrajectorySampling`; poses
  `(x, y, heading)`, ego frame). refcv4b has knots at 0.5, 1.0, 1.5, 2.0, 3.0, 4.0 s exactly; **2.5 and
  3.5 s must be interpolated** — state the method. Headings from the trajectory, frame and lateral sign
  VERIFIED by a control: push the logged human future (from the OpenScene logs) through your SAME
  conversion path and require it to reproduce the human agent's poses to < 1e-3 m
  (`taniteval/adapters/navsim.py::verify_frame` / `assert_lateral_sign_convention` exist for this).
- **Scoring**: stream E1 builds the metric cache at
  `C:\Users\Admin\navsim\exp\metric_cache_warmup_two_stage` and writes `CACHE_DONE.json` there when done —
  poll for it while you build the bridge; ⛔ do NOT build a second cache. Score every arm with the
  official two-stage scorer; `score` never `pdm_score`; all 9 sub-metrics; per stage; counts asserted
  (reuse `C:\Users\Admin\navsim\smoke_assert.py` logic — success over an empty set is a FAILURE).
- **Enforcement evidence** (gate `navsim.ego_enforcement`): (i) the declared-input manifest per arm;
  (ii) a MUTATION: randomise every UNDECLARED EgoStatus field at the seam and show the trajectories are
  byte-identical; (iii) for A2, show invariance to the ego values (or report that it isn't invariant).
- **Artifact**: via `taniteval/adapters/navsim.py::build_artifact` (⛔ do not edit that file — stream E3
  owns it; wrap in your `code/` if needed) with protocol `EPDMS_v2_warmup_two_stage`, devkit SHA,
  estimator `{status: UNAVAILABLE, reason, n}` (7 logs < the RG-14 floor of 8 — warmup NEVER carries a
  CI), `protocol.sensor_set` / `protocol.setting` (e.g. "3-camera stitch → 256×640 cylindrical,
  perception-free, zero-shot from PhysicalAI"), declared ego inputs, `route_leak_check` = UNVERIFIED
  (stream E3 is settling it). Run `python tools/criteria_check.py` over it; record PASS/FAIL per gate.
- **Four families**: `scenes_to_win` + `four_families_block` exist in the adapter. Compute what the
  stage-1 scenes allow (GT future from the OpenScene logs), refuse the rest PER FAMILY with reason + n.
  If time is short: EPDMS first, then name the family work as the next step.

## Pre-registration (write SPEC.md BEFORE scoring)

Bar: **A1's EPDMS > the CV agent's EPDMS on the same scenes** (E1 produces CV; paired per-scene
deltas, no CI — n_logs 7). Both outcomes committed in advance. Honest prior: refcv4b ties hold-action
at T1 on PhysicalAI (registry §4.6) and this is zero-shot (PhysicalAI → nuPlan, different cameras,
2 Hz history). ⭐ RULE ZERO: if A1 fails the bar, that is a waypoint — decompose by sub-metric (which
of NC/DAC/DDC/TLC/EP/TTC/LK/HC/EC kills it, per stage), name the next lever, and if a cheap one exists
(e.g. a declared input-construction fix you found), run it and report both.

## Tier / loop stamps

NavSim = the agent's own trajectory through a 4 s LQR + bicycle rollout vs non-reactive logged agents,
perception fixed at t0 ⇒ **T1-family, loop OPEN** (PI 2026-09-02) for stage 1; stage 2's re-rendered
start is **UNRULED** — say so. ⛔ Never call it closed loop.

## Do NOT

download anything · use the GPU · touch the live training process · edit devkit files in place · edit
`taniteval/adapters/navsim.py`, `CRITERIA_REGISTRY.json`, `tools/criteria_check.py` (E3) or
`LEADERBOARD.md` (E4) · train anything.

## Deliverables (this folder)

`SPEC.md` · `PLAN.md` · `tests/` (the GT round-trip control; the ego-mutation check) · `code/` (bridge,
agent, scripts) · `raw/` (seam files, manifests, score CSVs, artifact JSONs, criteria_check output) ·
`RESULT.md` (findings first; evidence class + tier/loop on every number) · `COMMS.md`. Stage exact
paths (keep large binaries out of git — anything > ~20 MB stays on the dev box and is listed in the
manifest with its sha256). Manifest at the end of your report.
