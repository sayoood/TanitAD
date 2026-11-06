# Brief: T2 kit for v8F, steps E3 to E6 and the stage-1 box-2 package (CPU build; do NOT launch on box 2)

Read first and follow: `E:/Projects/TanitAD/Project Steering/AGENT_OPERATING_STANDARD.md` (preamble).

**Repo and files**
* Never commit or push. Never delete files: move them aside. LF line endings.
* Read the repo via `git -C C:/Users/Admin/tanitad-push show 24b97ba9:<path>`.

**Compute, network and box 2**
* CPU only (`CUDA_VISIBLE_DEVICES=-1`). No downloads, no network.
* ⛔ No ssh to box 2 and no job there: the Master Mind launches stage 1 after freezing the card.

**Python:** `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`; assert `tanitad.__file__` is not on G:.

**Context budget:** ≤ 150k tokens. Hand off at about 130k with `HANDOFF_E3E6.md`, listing what is left.

**Banned words in what you write:** void, superseded, retracted, formerly, "no longer", correction, withdrawn, "closed loop". Say
"T2 re-perception rollout"; existing file names may be quoted.

## Read first (excerpts)
* **The card (DRAFT, your spec):** `E:/Projects/TanitAD-artifacts/dagland/motion_c/t2_prereg/PREREG_T2_CRAWL.md`. Sections 3 to 5:
  conditions T2 and RR, arms, starts, executor, metric, gates T0 to T5.
* **What E1/E2 built and left open:** `motion_c/t2v8f/HANDOFF_E1E2.md`, `code/t2a_v8f_policy.py`, `code/t2a_v8f_encoder.py`, `tests/`.
  * Gate status: G1, G2, G3 and G4 pass; the Master Mind re-ran 6/6.
* **The kit:** `F:/TanitAD-scratch/t2v8f/kit` (exported) and the repo folders `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-17-t2a-closedloop-prep/`
  and `-box2/` (`closedloop_drive.py`, `t2a_run.py`, `t2a_score.py`, `stub_gsplat_renderer.py`, `make_kit.py`, `RUN_ON_BOX2.md`).
* **The T1 reader functions the scorer must reuse unchanged:**
  * `motion_c/lf6_read/code/lf6_read.py`;
  * the `lf_read.py` / `lfo_read.py` it loads by pinned path (`logged_speed`, the M1 row, the 19 rows);
  * a banked T1-kin dump for the pin: the RIC0_s0 arrays and ticks named in `lf6_read/raw/` and `lf5_read/raw/arrays/`.

## Build (into `motion_c/t2v8f/`, extending E1/E2)
1. **E3, the executor and the rollout loop**, as a hook beside the harness actuator (`closedloop_drive.py` stays bit for bit for the
   historical arms; prove it with their existing tests).
   * **PLAYBACK:** the car's pose follows the plan's knots exactly. Each start uses the logged 16-row history, then 3 replans × 10
     executed knots at 0.2 s (6 s).
   * The encoder ticks every 0.2 s. Call `set_history` on every raw tick (E1 note).
   * **Condition T2:** render at the executed pose and time.
   * **Condition RR:** render at the LOGGED pose whose along-track arc equals the car's own executed arc, at that logged time, i.e.
     `reindexed` inside the renderer. Reuse the reindexing function of the `reindexed` tool for the index.
   * Agents: `objects` (replayed at log time) or `empty`.
2. **E4, the kit on the tip tree.**
   * Register the v8F arms in `EXTRA_ARMS`: C0_s0, C0_s1, DAG-OACC_s0/_s1, P10, ZERO, and BLIND (scene tokens := the banked training
     mean; find or compute it from the train cache stats, CPU).
   * Nav from the harness must equal the T1 tool's nav construction for the same logged route. Test it on a banked window.
   * Starts come from `t2_scenes/raw/SCREEN_TABLE.json` (the screening agent writes it; if absent, build against its schema in
     `t2_scenes/SCREEN_RULE.md`): t_c − 4 s and t_c − 2 s for launch scenes, `default_rng(1)` for guards.
   * `set_t0` per start.
3. **E5, the scorer.**
   * The M1 crawl share and the 19 T1 rows come from the T1 reader's OWN functions, per rollout window.
   * The paired scene-cluster bootstrap (B 2000, seed 0) uses `taniteval/ci.py`.
   * Contrasts X1 and X2 of the card, with the outcome logic of card section 6 as a pure function of the contrasts.
   * Tier stamp T2, never pooled with T1-kin.
4. **E6, the gates, as tests, each with a deliberate-regression mutation that must go RED.**
   * **T2g playback identity:** the logged plan executed by playback reproduces the logged ego track to 1e-6 m.
   * **T3:** ZERO equals P10 exactly, and an attribution of P10 against itself reads exactly 0.
   * **T4 metric pin:** the scorer reproduces the T1 reader's M1 share and the 19 rows bit for bit on the banked RIC0_s0 dump.
   * **Live path:** `plan(frames, intr)` through the stub renderer runs end to end and gives the same plan as the cached-feature path
     when the stub returns a logged frame.
   * **RR index:** with the car at the logged speed, RR equals logged playback.
5. **The stage-1 box-2 package** (built, NOT launched).
   * **What it runs:** a `launch.ps1`-style job (see memory `box2-second-computer.md`, `devbox-drive-letter-and-detached-jobs.md`) for
     the 2 scenes on hand. It runs:
     * the GPU gates: T1's render gap (rendered frame at the logged pose against the real frame, pooled-token cosine);
     * T5 determinism;
     * the actor falsifier (`objects` draws the lead; a planted no-lead scene reads absent);
     * a plumbing run of every arm.
   * **Outputs and markers:** results are written to a pull folder, with done / fail markers.
   * **Safe mode:** one job, no other process.
   * **Rehearsal:** dry-run the whole package on CPU with the stub renderer.
6. Run the tests and mutations, then `PINS.txt`.

## Hand-back (≤ 300 words)
Gate results with numbers, the mutation count, what the stage-1 package runs and its exact launch steps, open items, blockers, and a
deliverable manifest.
