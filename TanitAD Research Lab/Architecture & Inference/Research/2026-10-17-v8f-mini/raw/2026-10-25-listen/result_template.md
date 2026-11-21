# PREREG_V8F_LISTEN: result (executed 2026-10-25; zero training, CPU, M2b s0 and s1 banked checkpoints)

**Evidence class:** MEASURED (ours). Artifacts: `READ.json` (every number below), `runs/*.npz` (per-window arrays), `fig_listen_grid.png`. **Tier:** T1-class open loop, DIAGNOSTIC (donor inputs are interventions, not driving). **Estimator:** paired episode-cluster bootstrap, B = 2,000, seed 0 (`taniteval/ci.py`, through the eval's own `paired` and macro-F1 reducer). It answers "would another draw of EPISODES say this?"; it does not answer "would another TRAINING RUN say this?" (the two training seeds are shown side by side for that) nor "another inference run" (the planner is deterministic here). The PREREG is `3fdb1b61:Project Steering/PREREG_V8F_LISTEN.md`; nothing was staged or committed.

## 1. Both outcomes (the two committed tables, applied verbatim)

| table | outcome | condition met |
|---|---|---|
| Outcome (division of labour) | **NOT SHOWN** | P-UPPER fails at both seeds (T-lat's largest n is S checkpoints, not scene or nav); P-LOWER fails at both seeds (O's largest n is ego, not S checkpoints or T code); neither holds at any seed, so it is not SEEDS DISAGREE |
| Q-NAV (nav through T or around it) | **THROUGH T** | D = ADE6(nav at T only) - ADE6(nav at O only) is +0.461 m [+0.254, +0.687] at seed 0 and +0.510 m [+0.273, +0.793] at seed 1, separated at both |

Committed next step, outcome table, NOT SHOWN (quoted): "the report states that the levels' input dependence does not differ as the attention suggested, and the abstraction claim rests on the oracle and donor reads alone".
Committed next step, Q-NAV, THROUGH T (quoted): "the answer to the PI's question is yes: the trajectory follows nav mainly through T's decision".

**All five controls pass** (section 3), so the grid is reported. The reading rules R1 to R5 (how "largest n", "holds", the donor-seed floor, the headline precedence and Q-NAV are computed) were written into `listen_read.py` before any LISTEN number existed: the file's mtime is 10:28, the first grid run was written at 10:31 (`ls -l`, this directory).

### The largest cell per row per seed (R1: argmax of the unclipped n = delta(L,X) / delta(L,all))

{LARGEST}

P-UPPER and P-LOWER, per row and seed:

{HYP}

### What the grid says beyond the two committed tables (DESCRIPTIVE and POST HOC: not promoted, each would need its own pre-registration)

1. **The layers do differ in what they listen to; they differ along other lines than the two hypotheses drew.** Largest n per row, seeds 0 / 1: **S = nav** (0.52 / 0.48), then scene (0.40 / 0.43); **T-lat = S checkpoints** (0.76 / 0.77), then nav (0.34 / 0.41); **T-lon = ego** (0.67 / 0.68), then scene (0.50 / 0.49); **O = ego** (4.087 / 3.585), then T code (3.870 / 3.141). "NOT SHOWN" is a statement about P-UPPER and P-LOWER as written, not a statement that all four rows look alike.
2. **Attention is not dependence, shown in one cell.** T puts 97 to 100 % of its cross-attention on scene tokens (`D-V8F-ATTN-1`, INHERITED from the PREREG, not re-measured here), yet T-lat x scene is **-0.003 F1 [-0.059, +0.048] and +0.010 [-0.033, +0.051]**, not separated at either seed. The cell works as a probe: T-lon x scene is +0.185 / +0.187 F1, separated, and the deliberate-regression copy with a constant scene path reads T x scene as exactly 0.0 (control 3). T's lateral decision is carried by S's checkpoints, not by the image. Whether that is a real hierarchy or an echo of S's own nav read (S x nav n = 0.52 / 0.48) is **not tested here**.
3. **O barely reads the scene or its own nav token.** O x scene = +0.307 m [+0.214, +0.400] at seed 0 and +0.008 m [-0.015, +0.031] (not separated) at seed 1, against ALL = +1.394 / +1.315 m; O x nav = +0.015 m [+0.001, +0.030] and +0.011 m [-0.002, +0.024] (not separated at seed 1). The branch of the outcome table that sends O's scene to FLAT (PARTIAL, "if O's largest cell is the scene") was not reached, so FLAT is not triggered by this read; the descriptive fact is that O's scene cell is small.
4. **O's n exceeds 1, so n is not a share for that row.** ALL for O (+1.394 / +1.315 m) is **smaller** than ego alone (+5.697 / +4.716 m) or T code alone (+5.395 / +4.132 m). Swapping one conditioning token leaves the others from the window's own world (and the prior on its own `p10f`); swapping all gives a self-consistent donor world. HYPOTHESIS (untested): O reacts strongly to disagreement between its conditioning tokens, and the argmax-n rule (R1, fixed before data) therefore picks ego for O. On raw delta, ego and T code are both large with overlapping intervals (seed 0: +5.70 [4.49, 6.98] vs +5.40 [4.21, 6.63]; seed 1: +4.72 [3.72, 5.81] vs +4.13 [3.14, 5.14]). The verdict above is unchanged by this paragraph.
5. **"Listens" has no size threshold.** It is delta > 0, separated, and above the donor-seed floor. O x tick ages (+0.008 / +0.012 m, n 0.006 / 0.009) carries a LISTENS flag at both seeds; read the size from delta and n, not from the asterisk.
6. **Q-NAV beside.** Nav reaches the trajectory mostly through S and T: nav at S only +0.726 / +0.821 m (the route S -> T -> O, and S -> O), at T only +0.476 / +0.521 m, at O only +0.015 / +0.011 m, at all three +1.174 / +1.180 m. The committed table compares only T against O (THROUGH T); the S-only arm is larger than the T-only arm, so "through T's decision" understates that the route also passes through S's checkpoints.

### The cheapest next lever (RULE ZERO: a verdict must leave the next arm behind)

NOT SHOWN eliminates the two hypotheses as written; it does not leave nothing to do. Cheapest discriminating experiments, each CPU-only and about 1 to 2 minutes per run on the same harness (`listen.py` already supports single-cell swaps), **none run here because none is pre-registered**:
* **O coherence test:** swap pairs and triples of O's conditioning tokens (ego + S checkpoints + T code in the same donor window, the prior untouched) to see whether n > 1 is the inconsistency effect of point 4.
* **A second intervention family without cross-input conflict:** replace one input by the training-set mean token instead of a donor, to check that the rows' ordering is not an artefact of donor swapping.
* **T-lat echo test:** does T-lat's dependence on S checkpoints follow S's nav read (inject donor nav at S only and read T-lat), which separates a hierarchy from an echo.

What blocks them: only a pre-registration (the Master Mind's call) and the PI's go to spend the box; there is no compute or data blocker.

## 2. The full grid, per seed

Sign convention: S and O are donor minus clean (metres; larger is worse), T is clean minus donor (macro-F1 drop). n windows / episodes differ per row (S 2,319 / 137, T-lat 2,107 / 139, T-lon 2,240 / 136, O 2,399 / 141) because the label masks differ; the PREREG's "141 episodes" is the O row. The donor-seed floor is control 4 (seed 0, donor seeds 0 and 1), applied to both seeds (R3).

Clean level of each row's metric:

{CLEAN}

### Seed 0 (M2b_s0, donor seed 0)

{GRID_S0}

### Seed 1 (M2b_s1, donor seed 0)

{GRID_S1}

## 3. Controls

{CONTROLS}

Control 5 beside the cited banked range: the banked `--tac shuffle` deltas computed from `banked/` are +5.5365 m (seed 0) and +4.4198 m (seed 1), both at donor seed 1; ours at donor seed 0 are +5.395 m and +4.132 m, same sign and order of size. The seed-1 value is 0.29 m below the banked one; the donor-seed difference for that cell was measured only on seed 0 (0.141 m).

## 4. Q-NAV

{QNAV_T}

## 5. Deviations and disclosures

* **D1 (numerical, no number changes).** `v8f_mini_eval._f1_reducer` raises an IndexError on this numpy when the bootstrap hands it float64 arrays; `listen_read.f1_reducer` wraps the eval's own reducer with an exact integer cast (the encoded values are integers).
* **D2 (resume).** A previous agent stopped the run on memory pressure; this agent finished it. The runs `s0_d0 s1_d0 s0_d1 s0_id s1_id s0_navc s0_tsoff` used 8 torch threads; `s1_navc`, `s1_tsoff` and the analysis used 6 (`LISTEN_THREADS=6`, now an env var in `run_listen.py`), all with `LISTEN_LRU=8`. Every control compares arrays inside one run, and the clean pass is bit-identical to SFRESH's CPU re-run, so the thread count can touch only float noise across runs.
* **D3 (cosmetic, after the numbers).** The figure's cell font, the Q-NAV bar-label rotation and the caption text were adjusted after the first render; no cell content, ordering, colour rule or threshold changed.
* **D4 (reading rules).** R1 to R5 are the previous agent's, fixed before data (see section 1).
* **Memory and disk.** Peak of this agent's processes (`peakmem.log`): the two collect runs **2,643 and 2,646 MB peak private commit (1,895 / 1,896 MB peak working set)**; the analysis **1,119 MB (518 MB)**. The previous agent's unfixed run was 3.1 GB private / 2.2 GB working set (inherited, not re-measured). Guard readings before each step (`guard.log`): commit headroom 8.7 to 11.8 GB (threshold 5), C: free 0.87 to 1.6 GB (threshold 0.7). Other agents' processes were running throughout and were not touched.
* **Tests.** `test_listen.py`: 20 passed (`pytest_log_final.txt`); `run_mutations.py`: 9 source-level mutations all RED, unmutated GREEN (`mutation_log.txt`, previous agent). The tests create `pt_tmp/` in this directory; nothing was deleted.

## 6. Manifest (everything under `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/listen/`)

`LISTEN_RESULT.md` (this), `READ.json`, `fig_listen_grid.png`, `fig_listen_grid.py`, `listen.py`, `listen_read.py`, `run_listen.py`, `write_result.py`, `result_template.md`, `test_listen.py`, `run_mutations.py`, `runs/*.npz` (9), `repro_s0.json`, `mutation_log.txt`, `pytest_log.txt`, `pytest_log_final.txt`, `run_all.log`, `resume_runs.log`, `run_analyze.log`, `peakmem.log`, `guard.log`, `guard.ps1`, `resume_runs.sh`, `HANDOFF.md` (stale: superseded by this result), `banked/`, `code/` (tip export), `mock/` (figure test only), `pt_tmp/` (pytest scratch). Nothing was git-added or committed; landing this under `Research/` and updating `GOALS_AND_CLAIMS.md` (the abstraction claim, `D-V8F-ATTN-1`) is the Master Mind's integration step.
