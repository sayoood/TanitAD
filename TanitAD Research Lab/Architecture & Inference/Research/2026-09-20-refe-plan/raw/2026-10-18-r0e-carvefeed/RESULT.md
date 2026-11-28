# B8: the CARVE FEED of the SAFEOBJ dual (`--epo-con-feed carve`) -- R0e-pick launch blocker B8

Agent: B8 implementation subagent, 2026-10-18. Tip `821915a7fefea3fe6d68d2a493ab39ada4febd61` (mirror `C:/Users/Admin/tanitad-push`). Nothing staged in git, nothing committed or pushed; every file is under the OUT dir
(`C:/Users/Admin/b8carve/out/`), listed in `out/paths.json`. Evidence classes: MEASURED = our run, artifact path given; INHERITED = another package, not re-verified; ESTIMATED / HYPOTHESIS as marked.

## 0. Headline and ESCALATIONS (read first)

1. **B8 is built and proven reachable from its caller.** `refe/test_carve_feed.py` prints `ZZTEST_PASS 18` (2 unmutated suites + 16 RED mutations), MEASURED, `logs/test_carve_feed.log`. The default feed is bit-identical to the pre-B8 trainer
   through the real `main()` (every logged row and every saved tensor, max |diff| exactly 0.0).
2. ⛔ **INTEGRATION: the trainer edit and `gdvb_patch.diff` MUST LAND TOGETHER, AFTER the P6 gate fix.** The gate refuses a trainer flag with no G-DVB entry for EVERY argv, so on the tip gate the new trainer fails
   EIGHT tests by design: `test_launch_gate`, `test_rl_s1b` (`G_dvb_pass_built_0_25`), `test_rl_pickcon` (4 G checks), `test_rl_r0c`, `test_rl_safeobj`, `test_aux_s1_wiring`, `test_s1c`, `test_sft4r_gate` (MEASURED, section 5.1). With the patch applied to the P6-fixed files those are expected GREEN: see section 5 (gate lane).
3. **Six existing test files carry a minimal pin edit** (the sixth, `eval/test_safeobj_outcome.py`, plus `test_rl_safeobj`'s mutation-harness peel, were found by the classification pass: section 5.1) (B8 sits on top of PICKCON, as PICKCON sat on SAFEOBJ): `test_rl_s1b.py` (`pickcon_reverse_file` peels the B8 layer first), `test_rl_pickcon.py` / `test_rl_r0c.py` / `test_rl_safeobj.py` (their trainer suite peels the layer),
   `test_dump_scores.py` (four more default-off dests named). Base blobs in section 7. ⚠️ the `test_rl_r0c` and `test_rl_safeobj` peels were made AFTER their first lane run (r0c failed 2 P checks without it, MEASURED `logs/`): re-run GREEN in `gate_full` (`rl_r0c` 26, `rl_safeobj` 61; section 5.3).
4. ⚠️ **The carve LIST does not exist yet.** `refe/carve_logs.py` builds it with SO-CARVE's rule, but it needs the RL-train index's log names (pod `sets_train`; not on this box) and a PI choice of m. Until it is built and its sha256 frozen
   there is nothing to pass to `--epo-carve-logs`. This is blocker B8b (section 6).
5. ⚠️ **The 4th deliverable (K2 grad-norm logging) is NOT DONE.** Design in section 6; I did not touch the pinned `clip_grad_norm_` loop lines without a mutation test for them.
6. ⚠️ **Disk.** `C:` hit 0 bytes free during this work (two parallel test lanes wrote tens of GB of temp; killed, 33.9 GB free afterwards). `D:` has 3.85 GB free at 2026-10-18 (MEASURED, `Get-PSDrive`). Both drives are shared with the pods' pulls.

## 1. What was built (trainer `refe/scorer_finetune.py`, base blob `869b1862...` = tip, new blob `e357ede0...`)

| flag | default | meaning |
|---|---|---|
| `--epo-con-feed {train,carve}` | `train` | `train` = the dual reads the training micro-batches' violation = every arm before B8 |
| `--epo-carve-logs F` | `""` | the registered carve log list (whitespace-separated RL-train log names); only with `carve` |
| `--epo-carve-sha256 H` | `""` | sha256 of the list's BYTES, frozen before any data; the trainer recomputes it and REFUSES a mismatch |
| `--epo-carve-micro N` | `1` | carve micro-batches (`--batch` sets each) scored per optimiser update (draft prereg 2.3: one) |

**Semantics.** With `carve` the trainer (a) cuts the carve's WHOLE logs out of `tr_idx` (so the policy never trains on them), (b) builds a second `DataLoader` over them with its OWN `torch.Generator` (seed `--seed + 1`: it never
draws from the training loader's generator), (c) once per optimiser update, BEFORE `o.step()` (same weights the update's training micro-batches saw), runs the loop's own `fwd` and the loop's own `epo_con_term` (same rule, T, eps, lam)
under `torch.no_grad()` on `--epo-carve-micro` carve micro-batches and sums the parts, (d) feeds THAT accumulator to `dual.step`. The in-batch (training) accumulator is still summed and logged. The log row gains `carve_n`,
`carve_g_ttc/nc`, `carve_cA_*`, `carve_cB_*`, `carve_p1gap_*`, `carve_hinge_*` ONLY under `carve` (default rows are unchanged).

**Gradient path, stated.** The carve carries NO graph: `fwd` runs with grad disabled, the parts are Python floats, and no `.grad` is touched (U_1). It reaches the scorer ONLY through the VALUE of the duals `dual.lam`
(Python floats) that `train_losses` multiplies into the training micro-batches' constraint term. This is exactly how `con_acc` reaches it today.

**Hop trace** (MEASURED, patched trainer, from `logs/test_carve_feed.log`): argparse `:870` -> main cuts `tr_idx` `:1016`, builds `cdl` `:1019` -> `carve_stream` `:1144` -> `carve_feed_acc` no_grad `:1153`, `fwd` `:1159`,
`epo_con_term` `:1161` -> loop `carve_feed_acc()` before the optimiser loop `:1793` -> `dual.step(con_acc if carve_acc is None else carve_acc)` `:1807` -> `EpoDual.step` g to lambda `:331` -> `train_losses` passes `dual.lam`
`:1138`, `lB = lB + ct` `:1139` -> `((lA + lB) / a.accum).backward()` `:1779`; G7 re-reads the disjointness and probes the feed `:1568`.

**Refusals (rc 4, before any training; MEASURED through the real `main()`):** wrong sha256; a carve log that is an RL-val log; a carve log that is not an RL-train log (typo / dropped by a filter); an empty list; a carve with fewer than
`--batch` sets; carve levers named while the feed is `train`; carve without `--epo --epo-con g5`; carve with `--aux-drv`; malformed sha / micro < 1. Preflight G7 gains the carve fields and a real no-grad probe of the feed (`ok` includes them).
Navtest: carve is a subset of the RL-train index, and navtrain has 0 logs in navtest (INHERITED, `eval/HANDOFF_RL_S1.md`:97); `carve_logs.py` additionally checks a `--navtest-logs` list directly.

⚠️ **The carve removal is itself a lever** (draft prereg 2.3): it changes the training stream. The test therefore includes the CONTROL the arm needs for attribution: carve logs removed from training through `--heldout-logs`,
feed = train. Same training stream (asserted), arm A bit-identical to the carve-feed run (0.0), arm B and the lambda trajectory differ: the feed is isolated from the data removal. The real arm needs that control on the pod too.

Cost (ESTIMATED, not measured): one extra no-grad forward per carve micro-batch per update, ~+12 % of the 8 micro-batch forward work at `--epo-carve-micro 1`.

## 2. Tests (`refe/test_carve_feed.py`, `ZZTEST_PASS 18`; MEASURED, `logs/test_carve_feed.log`)

* **P** blob pin: peeling the B8 layer (`refe/carvefeed_layer.py`, 14 reversal chunks) gives the tip blob `869b1862` exactly; the parser gained exactly the four flags; exactly `build_parser, main, epo_con_refusals, epo_carve_refusals, main.carve_stream, main.carve_feed_acc` differ.
* **R** 9 static hops (file:line printed). **U** the real closure `carve_feed_acc` exec'd from `main()`'s source: no graph, no param grad, floats only, equals an independent `epo_con_term` call; ZERO CONTROL (B := A) reads EXACTLY 0.0; micro = 3 sums three; a stray row raises; the refusals.
* **E** through the real `main()` (CPU stubs for the GPU parts only), recorder on `epo_con_term` / `EpoDual.step`: default == explicit `train` == the pre-B8 trainer (rows, tensors, G7: max diff EXACTLY 0.0); in the carve run the dual's accumulator `n == micro` and `g ==` the no-grad
  call's g EXACTLY, every no-grad call scores carve-log sets only (25 calls), every training call scores non-carve sets only (50 calls); control-run attribution (above); a list built by `carve_logs.py` is accepted and its sha is the recorded one; the 5 refusals.
* **C** `carve_logs.py`: its draw IS SO-CARVE's `carve_draws` first draw (imported from `raw/2026-10-18-so-carve/so_carve_analysis.py`, independent of input order), deterministic, sha of its bytes, refuses a pool overlapping RL-val, a carve overlapping navtest, m outside [1, n-1], CLI.
* **16 RED mutations**, each editing source and required to turn named checks RED: M1 feed ignored; M2a carve loader reads TRAINING sets (stray-row guard in); M2b same with the guard removed; M3 carve not removed from training (G7 FAILs the preflight); M3b same with G7's term removed;
  M4 carve forward keeps its graph; M5 default flipped to carve; M6 RL-val overlap refusal removed; M7 sha check removed; M8 micro ignored; M9 refusals not wired; M10 unknown-log refusal removed; C-M1..C-M4 (seed, sort, val overlap, navtest).

## 3. SO-CARVE's carve definition, as reused (not invented)

PREREG_SOCARVE s.5 / `so_carve_analysis.py` (`carve_draws`, `SEED = 20260927`, `M_PRIMARY = 13`): pool logs sorted by name; `rng = np.random.default_rng(20260927)`; the carve is the first `rng.choice(n_logs, size=13, replace=False)`.
What changes is the pool: SO-CARVE drew from a 30-log harness pool of R0's states, so its carve was IN-SAMPLE (F1, R0 consumed every RL-train log). Here the pool is the RL-train index and the trainer cuts the carve out, which is what makes it held out.
`--heldout-logs` is NOT reused. **PI / Master Mind choice:** m = 13 of 166 logs is ~8 % of logs; set count per log is heavy-tailed (median 52, max 1,080: INHERITED, PREREG_SOCARVE s.0), so the realised carve may be 2-12 % of the sets. Report `--counts` in the manifest before freezing.

## 4. G-DVB patch (`gdvb_patch.diff`, against the P6-fixed files in `C:/Users/Admin/p6gate/out/`; MEASURED: `git apply --check -p1` clean, the result equals the gate tree byte for byte)

Apply with `git apply -p1` from the `refe/` parent after P6 lands. It touches `refe/gate_checks.py` (+43), `refe/launch_gate.py` (+8), `refe/test_launch_gate.py` (2 pins). Contents:
* `DVB_KINDS_SCORER`: `epo_con_feed`, `epo_carve_sha256`, `epo_carve_micro` = `built`; `epo_carve_logs` = `data`.
* `scorer_dvb_rows`: three rows for every `--epo` argv (feed, sha256, micro) against the data event's `epo_con_carve` block, which the trainer writes ONLY when the carve loader was built (absent = the training batches feed the dual); for a carve argv one more row (no training row in a carve log, no carve row in an RL-val log, non-empty).
  The default R0d argv observes 27 rows (was 24); a carve argv 28.
* **Regression arm `carve_feed_ignored`**: the trainer built the OPPOSITE feed to the argv's (a declared carve with the block removed; a declared train with a carve block injected), so it FAILS on ANY argv. Mutation GM3 (arm inert) turns its check RED.
* `judge_g7_carve(g7, ns)` re-judges G7's carve fields from raw values (sha equals the registered one, 0 training rows in carve logs, 0 carve rows in RL-val, carve >= one batch, probe micro-batches == the argv's, probe violation finite per constrained component; a train-fed argv must carry NO carve field), wired into `consume_scorer_preflight`.
* `launch_gate.py`: `RLS1_DATA` lists `--epo-carve-logs` (the PASS token fingerprints the list's bytes); the four flags join `AUXS1_EPO_FLAGS` (the aux and sft4r profiles forbid them).
* **Not done (R0e-pick's own blocker, not mine):** registering the carve arm (`eval/rls1e_arms.json`, profile `arm_must_name` for `--epo-con-feed`, `--epo-carve-*`, `arm_epo_registered`). A registered arm MUST name `--epo-con-feed carve`: the default `train` silently reproduces R0d.
* `gdvb/test_carve_feed_gate.py` (move to `refe/` with the patch): `ZZTEST_PASS 10` (1 unmutated + 9 RED mutations GM1-GM9), MEASURED in the gate tree, `logs/test_carve_feed_gate_gate_tree.log`.

## 5. Existing tests that name `scorer_finetune.py` / the gate files: BASE vs MINE vs GATE (updated 2026-10-18 by the classification agent)

**Three trees, all built from tip `d90d0335f0593c9098e219c780636402fd3d0a42` (the branch tip; the trainer and the 5 tests are byte-identical to `821915a`, so every base blob below is unchanged; `gate_checks/launch_gate/test_launch_gate` HAVE moved: P6 fix `bcc04b5`, blobs below):**
`full/base_full` = the tip's whole plan dir (`git archive`, 7,087 files, 443 MB; NOT the 652-file `refe/`+`eval/` subset the first lanes used: that subset had no `raw/`, which made `s1aux_chain`, `rls1_chain`, `r0e_diagnostics`, `safeobj_outcome` and `aux_s1_wiring` fail on a missing tracked file in BOTH lanes: harness incompleteness, not code);
`full/mine_full` = base + the B8 trainer + 6 edited tests + 3 new files, gate files as at the tip (expected: gate-interplay RED); `full/gate_full` = mine + `gdvb_patch.diff` applied (`git apply -p1` clean on the tip's gate files: the P6-fixed blobs ARE the tip's) + `test_carve_feed_gate.py` (= what lands).

### 5.1 Verdicts reached from the first-pass logs (MEASURED, `logs/partial_tree_821915a/{mine,base,gate}_lane/`; the lane then ran on the 652-file subset, box starved)

* **(c) THREE REAL REGRESSIONS in the pin layer (items 1-3), all found, all fixed in OUT, all GREEN in `gate_full` (section 5.3; `rl_safeobj` 61, `safeobj_outcome` 153, `rl_r0c` 26, each equal to BASE):**
  1. `refe/test_rl_safeobj.py`: the gate lane died in the MUTATION loop: `mutation M4 ... the snippet occurs 0 times in scorer_finetune.py: '                dual.step(con_acc)\n'` (`logs/partial_tree_821915a/gate_lane/test_rl_safeobj.py.log`). Cause: B8 rewrote that line to `dual.step(con_acc if carve_acc is None else carve_acc)` (trainer `:1807`), and the mutations edit the RAW trainer
     while only `suite_sf` peeled it. Fix (OUT blob `ff1b2fcd`): `main()` peels ONCE, `srcs[SF_PATH] = CL.peel(srcs[SF_PATH])`, so every mutation edits the pre-B8 text it was written against. In the mine lane this was hidden behind the gate-interplay G checks (mutations are never reached when the unmutated suite is RED).
  2. `eval/test_safeobj_outcome.py` (a SIXTH edited test; the brief said five): two source pins read the raw trainer and require the substring `"dual.step(con_acc)"` (`pin:trainer_log_cadence_columns_and_counter_semantics`, `pin:the_trainer_loop_lines_sim_log_copies`). A string count (MEASURED, static scan of every string literal of every test naming the trainer): 1 occurrence in the tip trainer, **0 in the B8 trainer**.
     The first lanes could not show it (the subset tree lacked `raw/2026-10-14-rls1b-attribution/dose.json`, so the test died earlier in BOTH lanes). Fix (OUT blob `8f1ec064`, CRLF file, CRLF kept): `sf_pre_b8()` = `CL.peel(SF_PATH.read_text())` for those two pins. The static scan found NO other literal removed by B8 in any test.
  3. `refe/test_rl_r0c.py` (found in the FULL-tree gate lane, caused by the G-DVB patch's LAYOUT): `AssertionError: mutation lg_aux_forbids_all_epo_flags_but_the_new_one: the snippet occurs 0 times in launch_gate.py: '"--epo-terms-train", "--epo-terms-heldout", "--epo-ref")'` (`logs/gate_lane/test_rl_r0c.py.log`, 56 s, i.e. the UNMUTATED suite incl. its G checks was GREEN).
     The patch appends the four carve flags after `"--epo-ref"` in `AUXS1_EPO_FLAGS`, so the tuple no longer ends with it. Fix (OUT blob `8fba321b`): the snippet is now `'"--epo-terms-heldout", "--epo-ref"'` -> `'"--epo-terms-heldout"'` (occurs exactly once in the unpatched AND the patched file: MEASURED by `str.count` on all three trees). RE-RUN DONE: `test_rl_r0c` `ZZTEST_PASS 26` in `gate_full` (`logs/gate_lane/test_rl_r0c.py.log`; again on the LF gate files, `logs/gate_lf_lane/`).
* **(a) expected gate-interplay rows, RED in MINE exactly as designed (a trainer flag with no G-DVB entry is refused by the gate for every argv):** `test_launch_gate` (static), `test_rl_s1b` (`G_dvb_pass_built_0_25`), `test_rl_pickcon` (4 G checks), `test_rl_r0c` (3 G checks), `test_rl_safeobj` (5 G checks), `test_aux_s1_wiring` (4 `gate/G_DVB_passes_...` checks),
  `test_s1c` (`trainer flags with NO declared-vs-built entry ... ['epo_carve_logs', 'epo_carve_micro', 'epo_carve_sha256', 'epo_con_feed']`), `test_sft4r_gate` (`set(DVB_KINDS_SCORER) == dests` fails by exactly those four names). ⚠️ The brief listed three such rows; there are EIGHT, and `r0c`, `aux_s1_wiring`, `s1c`, `sft4r_gate`, `rl_safeobj` were NOT in the first gate lane: they are in the full gate lane.
  GATE FULL-TREE results so far (MEASURED, `full/gate_logs`): `launch_gate` 677, `rl_safeobj` 61 (== base 61: regression 1 FIXED), `carve_feed_gate` 10, `rl_s1b` 18, `rl_pickcon` 20, `dump_scores` 16, `label_dir_refusal` 6, `rl_terms_label` 39, `aux_s1_wiring` 20, `aux_g12_bf16` 7, `s1c` 220, `sft4r_gate` 153, `s1aux_chain` 256, `r0e_diagnostics` 44 (== base 44), all rc 0. The rest is in 5.3. GATE first pass (subset tree, MEASURED): `launch_gate` `ZZLG_STATIC_PASS 677` (base 674 at `821915a`), `rl_s1b` 18, `rl_pickcon` 20, `dump_scores` 16, `carve_feed` 18, `carve_feed_gate` 10, `p6_levers` 199, `p6_chain` 351, all rc 0.
* **(b) environmental (same failure in BASE and MINE; `test_rls1_chain` additionally needs `Project Steering/MODEL_REGISTRY.md` ABOVE the package: `assert reg_file is not None, "MODEL_REGISTRY.md not found above the package"`, test line 1435; the trees now carry the tip's copy, blob `6cf109fb`, so it is re-run):** `test_aux_navtest_forward` (`AssertionError: the head module is not the D: copy`: it asserts the imported head module lives under `D:/Projects/TanitAD`; GREEN `ZZTEST_PASS 21` from a `D:/Projects/TanitAD_b8trees/` copy in BOTH gate and base, section 5.3.2); `test_rls1_queue` (BASE: `ps: unknown option -- o`, Git-bash `ps` has no `-o`: the queue's pgid check is pod-only; on the COMPLETE trees MINE fails at the same `ps -o` step as BASE; GATE's first pass died EARLIER at `returned before S1 was over` (a transient, not reproduced in 4 of 4 re-runs): section 5.3.1).

### 5.2 What was unknown at the first pass, and its answer (all MEASURED, section 5.3)

The 8 gate-interplay rows: all GREEN in `gate_full`. The fixes: all three turn their tests GREEN (`rl_safeobj` 61, `safeobj_outcome` 153, `rl_r0c` 26). `rls1_chain` 522, `s1aux_chain` 256, `aux_s1_wiring` 20, `r0e_diagnostics` 44, `safeobj_outcome` 153: GREEN on the COMPLETE tree in BASE, MINE and GATE (`aux_s1_wiring` is RED in MINE by design).

### 5.3 Full-tree table (BASE d90d033 / MINE / GATE; MEASURED, `logs/{base,mine,gate}_lane/`)

(`finish.py table`, then a 5th column added by the classification; the three `*_summary.json` are the unmodified lane records. Rows: 31. Runs: BASE 29, MINE 19, GATE 31; `test_carve_feed*` are new files (no BASE/MINE run); the 10 "gate-file consumer" rows were run in BASE and GATE only (the EXTRA lane).)

| test | BASE (tip d90d033) | MINE (B8, tip gate) | GATE (B8 + G-DVB patch) | class |
|---|---|---|---|---|
| `test_launch_gate.py_static_no-banked` | rc 0: ZZLG_STATIC_PASS 674 | **rc 1**: AssertionError: (['epo_carve_logs', 'epo_carve_micro', 'epo_carve_sha256', 'epo_con_feed'], []) | rc 0: ZZLG_STATIC_PASS 677 | (a) |
| `test_rl_safeobj.py` | rc 0: ZZTEST_PASS 61  (1 unmutated suite + 60 RED mutations) | **rc 1**: AssertionError: the unmutated sources FAIL their own checks: ['G_aux_s1_forbids_every_epo_flag_including_the_five_new_ones', 'G_dvb_PASS_R0d_built_g5' | rc 0: ZZTEST_PASS 61  (1 unmutated suite + 60 RED mutations) | (a) + (c) regression 1, FIXED |
| `test_carve_feed_gate.py` | not run | not run | rc 0: ZZTEST_PASS 10  (1 unmutated suite + 9 RED mutations) | NEW (lands with the unit) |
| `test_rl_s1b.py` | rc 0: ZZTEST_PASS 18  (1 unmutated suite + 17 RED mutations) | **rc 1**: AssertionError: the unmutated sources FAIL their own checks: ['G_dvb_pass_built_0_25'] | rc 0: ZZTEST_PASS 18  (1 unmutated suite + 17 RED mutations) | (a) |
| `test_rl_pickcon.py` | rc 0: ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | **rc 1**: AssertionError: the unmutated sources FAIL their own checks: ['G_the_aux_and_sft4r_profiles_forbid_the_flag', 'G_dvb_PASS_default_named_0.02_and_expli | rc 0: ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | (a) |
| `test_dump_scores.py` | rc 0: ZZTEST_PASS 16  (1 unmutated suite of 41 checks + 14 RED mutations + coverage) | rc 0: ZZTEST_PASS 16  (1 unmutated suite of 41 checks + 14 RED mutations + coverage) | rc 0: ZZTEST_PASS 16  (1 unmutated suite of 41 checks + 14 RED mutations + coverage) | GREEN in all three, no interplay |
| `test_label_dir_refusal.py` | rc 0: ZZTEST_PASS 6 | rc 0: ZZTEST_PASS 6 | rc 0: ZZTEST_PASS 6 | GREEN in all three, no interplay |
| `test_rl_terms_label.py` | rc 0: ZZTEST_PASS 39  (1 unmutated suite + 37 RED mutations + coverage) | rc 0: ZZTEST_PASS 39  (1 unmutated suite + 37 RED mutations + coverage) | rc 0: ZZTEST_PASS 39  (1 unmutated suite + 37 RED mutations + coverage) | GREEN in all three, no interplay |
| `test_aux_s1_wiring.py` | rc 0: ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | **rc 1**: AssertionError: the unmutated scorer_finetune.py FAILS its aux wiring checks: ['gate/G_DVB_passes_on_the_real_H1_data_event_and_observes_the_head', 'g | rc 0: ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | (a) |
| `test_aux_g12_bf16.py` | rc 0: ZZTEST_PASS 7  (1 unmutated suite of 30 checks + 6 RED mutations) | rc 0: ZZTEST_PASS 7  (1 unmutated suite of 30 checks + 6 RED mutations) | rc 0: ZZTEST_PASS 7  (1 unmutated suite of 30 checks + 6 RED mutations) | GREEN in all three, no interplay |
| `test_s1c.py` | rc 0: ZZTEST_PASS 220  (2 skipped, not counted) | **rc 1**: AssertionError: ["trainer flags with NO declared-vs-built entry (a new lever needs its check): ['epo_carve_logs', 'epo_carve_micro', 'epo_carve_sha256 | rc 0: ZZTEST_PASS 220  (2 skipped, not counted) | (a) |
| `test_sft4r_gate.py` | rc 0: ZZTEST_PASS 153 | **rc 1**: AssertionError: (['epo_carve_logs', 'epo_carve_micro', 'epo_carve_sha256', 'epo_con_feed'], []) | rc 0: ZZTEST_PASS 153 | (a) |
| `test_s1aux_chain.py` | rc 0: ZZTEST_PASS 256 | rc 0: ZZTEST_PASS 256 | rc 0: ZZTEST_PASS 256 | GREEN in all three, no interplay |
| `test_rls1_queue.py` | **rc 1**: AssertionError: {'rc': 1, 'out': "ps: unknown option -- o\nTry `ps --help' for more information.\n", 'seconds': 1.3005321025848389, 'timed_out': False | **rc 1**: AssertionError: {'rc': 1, 'out': "ps: unknown option -- o\nTry `ps --help' for more information.\n", 'seconds': 1.211488962173462, 'timed_out': False, | **rc 1**: AssertionError: returned before S1 was over | (b) environmental (pod-only `ps -o`), see 5.3.1 |
| `test_aux_navtest_forward.py` | **rc 1**: AssertionError: the head module is not the D: copy | **rc 1**: AssertionError: the head module is not the D: copy | **rc 1**: AssertionError: the head module is not the D: copy | (b) environmental (D:-path pin), see 5.3.2 |
| `test_r0e_diagnostics.py` | rc 0: ZZTEST_PASS 44  (1 unmutated suite of 28 checks + the CLI end to end, 43 RED mutations) | rc 0: ZZTEST_PASS 44  (1 unmutated suite of 28 checks + the CLI end to end, 43 RED mutations) | rc 0: ZZTEST_PASS 44  (1 unmutated suite of 28 checks + the CLI end to end, 43 RED mutations) | GREEN in all three, no interplay |
| `test_safeobj_outcome.py` | rc 0: ZZTEST_PASS 153  (1 unmutated suite of 189 checks + the CLI end to end, 152 RED mutations) | rc 0: ZZTEST_PASS 153  (1 unmutated suite of 189 checks + the CLI end to end, 152 RED mutations) | rc 0: ZZTEST_PASS 153  (1 unmutated suite of 189 checks + the CLI end to end, 152 RED mutations) | (c) regression 2, FIXED (GREEN in all three) |
| `test_rl_epo.py` | rc 0: ZZTEST_PASS 42  (1 unmutated suite + 41 RED mutations) | rc 0: ZZTEST_PASS 42  (1 unmutated suite + 41 RED mutations) | rc 0: ZZTEST_PASS 42  (1 unmutated suite + 41 RED mutations) | GREEN in all three, no interplay |
| `test_carve_feed.py` | not run | not run | rc 0: ZZTEST_PASS 18  (2 unmutated suites + 16 RED mutations) | NEW (lands with the unit) |
| `test_p6_levers.py` | rc 0: ZZP6_PASS 199 | not run | rc 0: ZZP6_PASS 199 | GREEN, no interplay (gate-file consumer: BASE+GATE only, MINE not run by design) |
| `test_p6_chain.py` | rc 0: ZZP6CHAIN_PASS 351 | not run | rc 0: ZZP6CHAIN_PASS 351 | GREEN, no interplay (gate-file consumer: BASE+GATE only, MINE not run by design) |
| `test_gate_suite_prerun.py` | rc 0: ZZTEST_PASS 144 | not run | rc 0: ZZTEST_PASS 144 | GREEN, no interplay (gate-file consumer: BASE+GATE only, MINE not run by design) |
| `test_suite_nesting.py` | rc 0: ZZTEST_PASS 95 | not run | rc 0: ZZTEST_PASS 95 | GREEN, no interplay (gate-file consumer: BASE+GATE only, MINE not run by design) |
| `test_p2rep_chain.py` | rc 0: ZZTEST_PASS 353 | not run | rc 0: ZZTEST_PASS 353 | GREEN, no interplay (gate-file consumer: BASE+GATE only, MINE not run by design) |
| `test_p2r_chain.py` | rc 0: ZZTEST_PASS 139 | not run | rc 0: ZZTEST_PASS 139 | GREEN, no interplay (gate-file consumer: BASE+GATE only, MINE not run by design) |
| `test_gate_gpu_probes.py` | rc 0: ZZTEST_PASS 16  (incl. the CUDA probe on this GPU) | not run | rc 0: ZZTEST_PASS 16  (incl. the CUDA probe on this GPU) | GREEN, no interplay (gate-file consumer: BASE+GATE only, MINE not run by design) |
| `test_p6_cpu_probe.py` | rc 0: ZZTEST_PASS 111 | not run | rc 0: ZZTEST_PASS 111 | GREEN, no interplay (gate-file consumer: BASE+GATE only, MINE not run by design) |
| `test_p6_overlap.py` | rc 0: ZZTEST_PASS 54 | not run | rc 0: ZZTEST_PASS 54 | GREEN, no interplay (gate-file consumer: BASE+GATE only, MINE not run by design) |
| `test_sft4r_chain.py` | rc 0: ZZTEST_PASS 31 | not run | rc 0: ZZTEST_PASS 31 | GREEN, no interplay (gate-file consumer: BASE+GATE only, MINE not run by design) |
| `test_rl_r0c.py` | rc 0: ZZTEST_PASS 26  (1 unmutated suite + 25 RED mutations) | **rc 1**: AssertionError: the unmutated sources FAIL their own checks: ['G_aux_s1_forbids_every_epo_flag_including_the_new_one', 'G_dvb_PASS_built_armA', 'G_dvb | rc 0: ZZTEST_PASS 26  (1 unmutated suite + 25 RED mutations) | (a) + (c) regression 3, FIXED |
| `test_rls1_chain.py` | rc 0: ZZTEST_PASS 522 | rc 0: ZZTEST_PASS 522 | rc 0: ZZTEST_PASS 522 | GREEN in all three, no interplay |

**Per-class counts (MEASURED, computed from the summaries):**

| class | rows | state in `gate_full` (= what lands) |
|---|---|---|
| (a) gate-interplay: RED in MINE by design, MUST be GREEN in GATE | **8** (`launch_gate` static, `rl_s1b`, `rl_pickcon`, `rl_r0c`, `rl_safeobj`, `aux_s1_wiring`, `s1c`, `sft4r_gate`) | **8 of 8 GREEN** (8 of 8 RED in MINE, all GREEN in BASE) |
| (b) environmental | **2** (`rls1_queue` pod-only `ps -o`; `aux_navtest_forward` D:-path pin) | RED in BASE, MINE and GATE alike on the lane's `C:` trees; `aux_navtest_forward` GREEN (`ZZTEST_PASS 21`) from a `D:/Projects/TanitAD_b8trees/` copy in BOTH gate and base (5.3.2); `rls1_queue` fails at the SAME step in all three (5.3.1) |
| (c) REAL regression still open | **0** | none |
| (c) REAL regression found by this pass and FIXED in OUT | **3** (`rl_safeobj` M4 pin; `safeobj_outcome` two source pins; `rl_r0c` mutation snippet) | all three GREEN: `rl_safeobj` 61 (= BASE 61), `safeobj_outcome` 153 (= BASE 153), `rl_r0c` 26 (= BASE 26). Two of them are also in (a). |
| NEW files | **2** (`carve_feed` 18, `carve_feed_gate` 10) | GREEN |
| GREEN everywhere, no interplay | **18** (`dump_scores`, `label_dir_refusal`, `rl_terms_label`, `aux_g12_bf16`, `s1aux_chain`, `r0e_diagnostics`, `rl_epo`, `rls1_chain`, plus the 10 gate-file consumers: `p6_levers`, `p6_chain`, `gate_suite_prerun`, `suite_nesting`, `p2rep_chain`, `p2r_chain`, `gate_gpu_probes`, `p6_cpu_probe`, `p6_overlap`, `sft4r_chain`) | GREEN |

Totals: 8 (a) + 2 (b) + 2 NEW + 18 GREEN-everywhere + 1 (`safeobj_outcome`, the (c)-only row, GREEN in all three) = 31. REDs: GATE 2 (both (b)), BASE 2 (the same two), MINE 10 (the 8 (a) rows + the same two). **Every (a) row is GREEN in `gate_full`; no (c) row remains.**

#### 5.3.1 `test_rls1_queue`: environmental, but NOT for the reason first written (MEASURED; logs in `logs/{base,mine,gate}_lane/` and `logs/rerun_queue/`)

The test runs `queue_scenarios` (`test_rls1_queue.py:938`) and then `kmon_scenarios` (`:939`). The handoff's rule: environmental only if BASE fails at the same step or earlier. The FIRST PASS did not satisfy it (GATE died EARLIER than BASE), so it was diagnosed instead of waved through:

| run | tree | where it stops | decisive line |
|---|---|---|---|
| lane, first pass | base_full | `queue_scenarios` PASSED; `kmon_scenarios` `healthy` | `logs/base_lane/test_rls1_queue.py.log:11` `ps: unknown option -- o` (test `:423`; the chain's `ps -o pgid=` is `refe/rls1_pod.sh:553`), 69.8 s |
| lane, first pass | mine_full | same as BASE | `logs/mine_lane/test_rls1_queue.py.log:11`, 68.1 s |
| lane, first pass | gate_full | **EARLIER**: `queue_scenarios` scenario `s1` | `logs/gate_lane/test_rls1_queue.py.log:11` `AssertionError: returned before S1 was over` (test `:184`), 2.7 s |
| re-run 1 | gate_full | the same early death | `logs/rerun_queue/gate_rerun1_FAIL_s1_2.9s.log:11`, 2.9 s |
| re-run | base_full | `kmon` `ps -o` | `logs/rerun_queue/base_rerun1_69.8s_ps-o.log:11` |
| re-runs 2, 3, 4 (lane_runner) and 1 direct `python refe/test_rls1_queue.py` | gate_full | **`kmon` `ps -o`, exactly as BASE** | `logs/rerun_queue/gate_rerun2_71.5s_ps-o.log:11`, `gate_rerun4_70.1s_ps-o.log:11`, `gate_direct_run_ps-o.log:11` (3 of 3 lane re-runs + the direct run: 4 of 4) |

* **Code identity (MEASURED, sha1 of every file in `refe/` and `eval/`, gate_full vs base_full):** `refe/test_rls1_queue.py`, `refe/rls1_pod.sh`, `refe/rls1_checks.py`, `refe/test_rls1_chain.py` are BYTE-IDENTICAL; the 14 files that differ are exactly the landing unit and none is imported by scenario `s1` (it needs `rls1_pod.sh` text, bash and perl only).
* **Isolated scenario (MEASURED):** `Queue("q_s1")` + `other("s1aux", "sleep 4")` + `start()` run on BOTH trees with the lane's TEMP layout: `rc 0`, 4.8 s elapsed, `ZZRLS1_QUEUED` then `ZZRLS1_QUEUE_LOCK` logged (so S1's lock was seen held), in both.
* **Count of `s1` executions:** gate_full 2 fail (lane, re-run 1) / 6 pass (re-runs 2-4, direct run, 2 isolated); base_full 0 fail / 4 pass (lane, re-run, 2 isolated); mine_full 0 / 1.
* **Mechanism (HYPOTHESIS, not proven: the failing sandbox is deleted by the test):** `s1` gives the `other` S1 process a hard-coded 1.0 s head start (`:179` `time.sleep(1.0)`) before `t_start`; if Windows Git-bash has not yet taken the lock, the queue finds S1 not holding it, returns `rc 0` in about 1 s, and `:184` fires. The box is shared (another session's SR-TTC S0 dump ran throughout; free RAM was 4-7 GB and CPU 45 % when sampled afterwards); the load at the two failing moments was NOT recorded. A latent timing flake in an UNEDITED test; the test is deliberately left alone (editing it would enlarge the landing unit).
* **VERDICT: (b) environmental.** The deterministic failure in every tree is the pod-only `ps -o pgid=` (`kmon_scenarios`, `:423`); the gate_full early death at `:184` is a transient that did not reproduce in 4 of 4 later runs. It is not B8's and it needs the pod (or a Linux `ps`) to go green.

#### 5.3.2 `test_aux_navtest_forward`: path-pinned, GREEN from a `D:` copy (MEASURED; `logs/navtest_forward/`)

The test asserts the imported head module lives under `D:/Projects/TanitAD` (`eval/test_aux_navtest_forward.py:396-397`). Each tree was copied to `D:/Projects/TanitAD_b8trees/{gate_full,base_full}` (robocopy, 7,148 / 7,137 files), the test run once from each (Git-bash first on PATH, `tanitad` venv), then BOTH COPIES WERE DELETED (`Test-Path D:\Projects\TanitAD_b8trees` = False):

* **gate_full copy: `ZZTEST_PASS 21  (11 checks + 2 real-store + 8 RED mutations)  29 s`** (`logs/navtest_forward/gate_full_D_copy.log`)
* **base_full copy: `ZZTEST_PASS 21  (11 checks + 2 real-store + 8 RED mutations)  5 s`** (`logs/navtest_forward/base_full_D_copy.log`)

Identical tokens: B8 changes nothing this test sees. The lane's `rc 1` (`the head module is not the D: copy`) is purely the `C:` location of the lane trees.

#### 5.3.3 LANDING BYTES: line endings (MEASURED; found by the tip-blob check, fixed in OUT)

The manifest blobs in section 7 are the LF-normalised hashes (what `git hash-object --path=<repo path>` gives through the mirror's `core.autocrlf=true`). The OUT files were NOT all in that form:

| file | tip blob EOL | OUT EOL before | raw OUT blob before | manifest blob | action |
|---|---|---|---|---|---|
| `refe/gate_checks.py` | LF | **CRLF** (1,780 lines) | `7527376f` | `86981ced` | **converted to LF in OUT and gate_full; raw blob now `86981ced`** |
| `refe/launch_gate.py` | LF | **CRLF** (1,153) | `15c3db0c` | `3a890085` | **converted; raw blob now `3a890085`** |
| `refe/test_launch_gate.py` | LF | **CRLF** (1,731) | `cb8e0df5` | `63909985` | **converted; raw blob now `63909985`** |
| `eval/test_safeobj_outcome.py` | **CRLF** (tip blob `f2c358c6` is raw CRLF) | CRLF (2,067) | `64108c2f` | `8f1ec064` (= the LF form) | **kept CRLF; the CORRECT landing blob is the raw one, `64108c2f`** |

Why it matters: landing the three gate files as raw bytes would have been a whole-file line-ending diff (1,100 to 1,800 lines each) on the three files the launch gate binds by sha256 (`gate_token.py:473-475`); and landing `test_safeobj_outcome.py` through the autocrlf filter would flip a CRLF file to LF (2,059 lines). **LANDER: stage the 3 gate files and every other OUT file as raw bytes; for `eval/test_safeobj_outcome.py` use `git hash-object -w --no-filters` and assert the staged blob is `64108c2f` (a `git add` / filtered hash gives `8f1ec064` = whole-file flip).** The check after staging: `git diff --cached --stat` for the three gate files must be of the size of `gdvb_patch.diff` (15 KB, a few hundred lines in total), not 1,100-1,800 lines each; for `test_safeobj_outcome.py` a handful of lines (2,059 -> 2,067).

Independent confirmation: `git apply -p1 gdvb_patch.diff` onto the tip's three LF blobs (scratch dir, `core.autocrlf=false`) reproduces `86981ced`, `3a890085`, `63909985` exactly.

Re-verification on the LF bytes (`full/gate_lf_summary.json`, logs `logs/gate_lf_lane/`, gate_full, MEASURED): **all 8 (a) rows and both NEW rows re-run GREEN with tokens identical to the CRLF lane rows above.** (The lane rows above were produced on the CRLF bytes; the conversion changes nothing Python parses.) One slip is on record: the first `aux_s1_wiring` attempt used a wrong path (`eval/`, rc 2, 0.2 s) and was re-run from `refe/`; only the final entry is in the table.

| test (gate_full, LF gate files) | rc | secs | token |
|---|---|---|---|
| `test_launch_gate.py_static_no-banked` | 0 | 67.8 | ZZLG_STATIC_PASS 677 |
| `test_carve_feed_gate.py` | 0 | 14.8 | ZZTEST_PASS 10  (1 unmutated suite + 9 RED mutations) |
| `test_rl_s1b.py` | 0 | 18.5 | ZZTEST_PASS 18  (1 unmutated suite + 17 RED mutations) |
| `test_rl_pickcon.py` | 0 | 23.5 | ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) |
| `test_sft4r_gate.py` | 0 | 11.6 | ZZTEST_PASS 153 |
| `test_rl_r0c.py` | 0 | 89.8 | ZZTEST_PASS 26  (1 unmutated suite + 25 RED mutations) |
| `test_carve_feed.py` | 0 | 69.0 | ZZTEST_PASS 18  (2 unmutated suites + 16 RED mutations) |
| `test_s1c.py` | 0 | 135.4 | ZZTEST_PASS 220  (2 skipped, not counted) |
| `test_aux_s1_wiring.py` | 0 | 154.4 | ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) |
| `test_rl_safeobj.py` | 0 | 148.4 | ZZTEST_PASS 61  (1 unmutated suite + 60 RED mutations) |

The gate-file consumers NOT re-run on LF bytes (UNVERIFIED on LF; their lane GREENs are on CRLF bytes, which differ only in `\r`): the 10 EXTRA rows, `rls1_chain`, `s1aux_chain`, `rl_epo`, `dump_scores`, `rl_terms_label`. The lander's own suite run covers them.

#### 5.3.4 Base re-check against the CURRENT remote tip (MEASURED, 2026-10-18)

`git ls-remote origin refs/heads/agent/arch-inf-20260803` = `3c799095d345a7d74e05823675ed92fe77be616b` (fetched; `d90d033` IS its ancestor, 7 commits ahead: `7cccc9b`, `be1d0c0`, `ad4140f`, `d2556c9`, `58bd819`, `2eb61a9`, `3c79909`). `git rev-parse 3c79909:<path>` equals the `d90d033` base blob for **all 10 EDIT files (0 differ)**; the 4 NEW paths are absent at the tip. The 7 commits touch under this plan dir only `raw/2026-10-18-sel-rank/**`. So the landing unit applies to `3c79909` without a rebase of any `refe/` or `eval/` file.


## 6. Open items

* **B8b -- freeze the carve list.** Needs the RL-train log names (pod). `python refe/carve_logs.py --train-logs <166 names> --rlval-logs rl_val_logs.txt --navtest-logs <136 names> --counts <log: n_sets json> --m 13 --out carve_logs.txt`; record its sha256 in the prereg BEFORE the run. PI decides m.
* **K2 grad-norm clause (deliverable 4): NOT implemented.** `rls1_checks.py` kills only on a non-finite loss (`:488` `_row_finite`, `:515` K2 in the monitor, `:558` in replay). Enforcement needs (i) the trainer to log the return value of
  `torch.nn.utils.clip_grad_norm_(ps, 1.0)` per update for BOTH optimisers (the lines in the update loop, `scorer_finetune.py:1795-1800` of the new file), default-safe = write a separate `gnorm.jsonl` only when `dual.active` (no change to `sft.jsonl` rows, so the
  identity tests keep comparing identical rows), and (ii) `rls1_checks.py` K2 to compare the arm's gnorm series with the control arm's median (> 20x = kill). Both need their own mutation tests; the loop line is pinned by several suites.
* **Pod test of the real data path.** Everything here ran on CPU stubs (`test_rl_epo.smoke`). The carve loader with `num_workers=1`, the real bank and `IndexedSubset` row indices are exercised only by the preflight G7 probe on the pod (it will FAIL loudly if the carve batch is not carve-only).
* **Gate files are staged as FULL patched files** (`refe/gate_checks.py`, `refe/launch_gate.py`, `refe/test_launch_gate.py`, `refe/test_carve_feed_gate.py`), built by applying `gdvb_patch.diff` to the tip's blobs; the diff stays as provenance. The raw copy `raw/.../gdvb/test_carve_feed_gate.py` was removed (it lives at `refe/`).

## 7. Manifest (repo-relative under `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/`; `out/paths.json` is the machine list). Base blobs are at tip `d90d033`, RE-VERIFIED IDENTICAL at the remote tip `3c799095` (all 10 EDIT, section 5.3.4). The `new blob` column is the blob of the file AS IT SITS IN OUT (raw bytes; the three gate files were converted to LF to make that true, section 5.3.3).

| file | status | base blob (tip d90d033) | new blob |
|---|---|---|---|
| `refe/scorer_finetune.py` | EDIT | `869b186293150dc6c07253391aeed9f84738225f` | `e357ede0f538caf7f7bf68278c7382127a8eb96f` |
| `refe/test_rl_s1b.py` | EDIT (pin) | `95a6a0aabd42045b6ad2c403d07ea24ab6fe9f64` | `f9b8d4e31ae0868e297fd72d1b63ea6e9e3c0da3` |
| `refe/test_rl_pickcon.py` | EDIT (pin) | `2f15b3ffa3bc23f38d8fb435413a918d3c0fda94` | `2690632f44fb75a7846e82a6f7368806eea2ebe9` |
| `refe/test_dump_scores.py` | EDIT (pin) | `a27c44420047bf818f230f0c8e143fa4b788ec0a` | `1be47490b60aa5e22d8f525077eb2ff3502dee67` |
| `refe/test_rl_r0c.py` | EDIT (peel + layout-independent `lg_aux_forbids...` snippet; fixes real regression 3) | `0df790c0e54d209498494a2b98d68f4f52f0e5dd` | `8fba321b56cb42c30a65077d9c14b016ef96465b` |
| `refe/test_rl_safeobj.py` | EDIT (peel once in `main()`; fixes real regression 1) | `9f74bb4193071d86b700070311f11543ebcf5a21` | `ff1b2fcdc3fff1d84a1c7b4f460cbd29411e1bf7` |
| `eval/test_safeobj_outcome.py` (CRLF) | EDIT (`sf_pre_b8()`; fixes real regression 2) | `f2c358c60c9fbb3825b5e17d4322e24351b7b1a3` | `64108c2fd4a691c4f8cd7a1b28afa50fb7f1efb7` (RAW CRLF bytes, as in OUT; corrected 2026-10-18: the earlier `8f1ec064` is the LF-normalised form, which would flip this CRLF file; section 5.3.3) |
| `refe/gate_checks.py` | EDIT (G-DVB rows, `judge_g7_carve`, arm `carve_feed_ignored`) | `a2f69e1c75a6dd885547b990bf907b7bc3cec6d1` | `86981ceddc63b9766a70853be62639f9c0fcfa9d` |
| `refe/launch_gate.py` | EDIT (`RLS1_DATA`, `AUXS1_EPO_FLAGS`) | `91c33774d2f49478965bd0d25bcfc4a34b009383` | `3a890085e7b780ff1feb1b23010ca9ec4871880f` |
| `refe/test_launch_gate.py` | EDIT (2 pins) | `e818fa3bfe0e4f1e2892c1e6a239d8cb877bcd8a` | `63909985b2137b2b87540c9d808384d4f6d18392` |
| `refe/carvefeed_layer.py` | NEW | -- | `97a0a302d540ae2f5a70f564d0bf03568311701b` |
| `refe/carve_logs.py` | NEW | -- | `16108ff686b3d5840a852c9792a0bc9585fb3ac0` |
| `refe/test_carve_feed.py` | NEW | -- | `0442cfd46fdea200b373a9a1336d227e2f36e83a` |
| `refe/test_carve_feed_gate.py` | NEW (lands with the gate files) | -- | `43cfb08bdd2dbccb54ba883623f41208e918e42a` |
| `raw/2026-10-18-r0e-carvefeed/gdvb_patch.diff` | provenance only (the patched files above are the landing unit) | -- | `e8c604b2e369796445114675f5ca00ce921ce9b2` |
| `raw/2026-10-18-r0e-carvefeed/{RESULT.md,HANDOFF.md,logs/**,provenance/*}` | NEW | -- | -- |

LANDING UNIT = the 10 EDIT rows + the 4 NEW `refe/` rows, all together (a subset breaks `test_launch_gate`, `test_s1c`, `test_sft4r_gate`, `test_aux_s1_wiring`, `test_rl_s1b`, `test_rl_pickcon`, `test_rl_r0c`, `test_rl_safeobj` by design).
Everything exists in ONE place only (this dev box, `C:/Users/Admin/b8carve/out/`) until the Master Mind lands it. STRATEGIC family: PAUSED (PI 2026-09-19), not a pass; nothing here is a model result.
