# HANDOFF M-GYM-1 (written 2026-11-08 09:27). State: environment BUILT and TESTED, 20-state measurement RUNNING detached.

**Do not wait alive.** Job: WMI-detached `F:\tanitad_scratch_mgym1\run_all.cmd` (pid 36844 at launch). Expected ~20-30 min.
Done marker: `F:\tanitad_scratch_mgym1\ALL_DONE.txt`. Progress: `F:\tanitad_scratch_mgym1\run.log` ("state N done ..."), then `det1.txt/det2.txt`, then `summarize.out`.
It runs: `run_pilot.py main --workers 6` (20 states x 24 first actions x {log, idm} x 17 cycles, + regression policies), twice `determinism_child.py 1` (fresh processes), `summarize.py`.

## Built (all in `pilot/scorer_gym/`, hashes in `../CODE_PIN.json`; code is FROZEN for this run)
* `gym_env.py` env: state (nuPlan DB+map) -> 24 proposals (15 PDM-Closed fan + 9 centreline speed-sweep) -> nuPlan LQR + bicycle at 10 Hz ->
  agents `log` (replay) / `idm` (nuPlan IDMAgents) -> per-0.5 s-cycle reward (multiplicative, hard terminations, all 8 q terms logged).
  Continuation = PDM-Closed re-planned each 0.5 s; first action 0.5 s + 8.0 s continuation = 17 cycles.
* `run_pilot.py`, `summarize.py` (verdict per `PREREG_M_GYM_1.md`, written BEFORE measuring), `determinism_child.py`, `tests/test_gym.py` (6 tests, all PASS; run `python tests/test_gym.py`, no pytest in the venv).
* Smoke (3 states, 4 arms, MEASURED): 2-5 s per arm per mode single worker => roughly 15-20 s/state effective at 12 workers, i.e. GO is plausible but NOT yet the verdict.

## Exact next steps for the fresh agent (Sonnet, ~40k tokens)
1. `until [ -f /f/tanitad_scratch_mgym1/ALL_DONE.txt ]; do sleep 5; done` (single background Bash). If `run.log` lacks `RUN_DONE`, inspect the tail; if a worker crashed, rerun only missing states (do_state is per-state) - do NOT change code or thresholds.
2. Read `F:/tanitad_scratch_mgym1/summarize.out` and `run/results.json` (cost verdict, activation, hard-event rates, regression arms). `diff det1.txt det2.txt` must be empty (bit-identical rewards across fresh processes).
3. Copy small files into `pilot/`: `results.json`, `cost_rows.json`, `run_meta.json`, `states_used.json` (= F:/tanitad_scratch_mgym1/states.json; 20 states, 10 logs, navtest-excluded), `run.log` tail, `summarize.out`. NOT the jsonl.gz (size) unless < 5 MB: gzip is small, copy it as `pilot/raw_states.jsonl.gz` if it is.
4. Write `M_GYM_1_RESULT.md`: verdict (GO/REDUCE/BLOCKED by mean eff12 s/state), cost table (buckets, per-mode, cold-map-load vs warm), per-term activation (flag any term fired < 1 % as `D-RL-COLL-SPARSE-1`), hard-event rates by mode, regression arms vs the PREREG criteria (report non-"known" arms honestly; the do-nothing post-hoc number is labelled POST_HOC), determinism. If REDUCE: state the largest scale fitting 3 days (states, K, horizon) and the cheapest knob (cost scales linearly in K and in horizon cycles; planning is the largest bucket).
4b. NOTE `summarize.py` was NOT tested on a full run (only syntax-level by design); if it errors, fix it (analysis code, not the measured env) and rerun it on `run/states.jsonl.gz`; keep the PREREG criteria as written.
5. Honest caveats to carry into the result (all MEASURED/known): (a) the 8 soft terms are pilot operationalisations of design 2.5 (corridor-based lead/conflict, no red-light stall except via lane-connector check, TTC hard rule has a closing-speed guard and ignores agents behind); (b) at-fault collision is a simplified classifier; (c) PDM replans each 0.5 s with a centreline rebuilt per episode, drivable map radius 150 m around the START state for the evaluator; (d) `q_comf` fired in ALL arms of the smoke state incl. PDM-Closed (2 channels) - check `comfort_channel_counts`, it may be a measured-jerk artefact of the tracker, not driving; (e) ESTIMATE of 12-worker scaling is linear from a 6-worker run on a box shared with GPU chains; (f) the 147 navtest logs ARE among the 1,349 local test DBs (147/147, MEASURED), the pilot excluded them (1,202 candidates); the design's "0 of 325 overlap" row is about a different subset.
6. Stage nothing (`git add` not requested for artifacts outside the repo). Write `MANIFEST.md` update. DELETE scratch at the end by literal path and log it: `F:/tanitad_scratch_mgym1` (run, smoke, pt, *.py, outputs; ~tens of MB) - first confirm the pilot/ copies exist.
7. Hand back <= 250 words + paths.

## UPDATE 2026-11-08 (M-GYM-1-READ agent): measurement READ, result written
* Verdict REDUCE (mean 28.08 s/state at 12 workers; GO needs <= 26). See `M_GYM_1_RESULT.md`. Do-nothing and oracle arms MISS as pre-registered (reported, not tuned).
* Copied into pilot/: results.json, cost_rows.json, run_meta.json, raw_states.jsonl.gz (705 KB), summarize.out, det1/det2.txt, run_log_tail.txt, analysis scripts in scorer_gym/analysis_*.py.
* DELETION NOT DONE: `rm -rf` of `F:\tanitad_scratch_mgym1` (875 KB) was refused by the safety check; not worked around. All outputs are banked in pilot/, so the folder is safe for the PI/session to delete by that literal path.
