# PREREG_P5C Amendment A3.3 (DRAFT, not frozen): R-A at 1e-8 for TB-0 / TB-0r, and the one-row TB-5 Tc20s-label fallback. REFe agent, 2026-11-16

Evidence class: MEASURED (artifacts in this hold and on the pod). Status: DRAFT. Nothing is committed, landed or launched. Hold: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_a33/` (code under `a33w/`).

## The two rulings (PI, 2026-11-16, verbatim)
1. R-A: *"Bar 1e-8 (Recommended)"*, i.e. "Set R-A to 1e-8, about 4x the largest measured floor (2.55e-9), for TB-0 and TB-0r on all ranks. R-B stays the decision gate on the labels C1 trains on. The distributions are disclosed."
2. TB-5 row: *"Its Tc20s label (Recommended)"*, i.e. "Use the same row's Tc20s(RJ5) warm-start label, which passes the sanity check. It is recorded per row as a disclosed fallback; TB-5 then passes 150,603/150,603."

## Disclosure: both rulings are POST-DATA
Both were made after run 3 (`/workspace/data/p5c_label_a32`, 09:44Z to 11:49Z, `ZZP5CL_FAIL gates`; findings `raw/2026-11-16-p5c-window/run3/RUN3_FINDINGS.md`) measured:
- **R-A:** TB-0r's Tc20s part max **2.547964894628052e-09** (45 of 16,133 rows over 1e-9); the control pass max **1.4901395672950457e-09**. 1e-8 is 3.9x the larger. R-B (the labels C1 trains on) passed in full on all 18,533 compared rows, and R-B is unchanged. TB-1 stays 1e-9.
- **TB-5:** exactly 1 of 150,603 rows fails S2 / the yaw-branch test: rank 0, token `d06ff3cbd9025da7`, RJ5 traj sha256 `926ffc6698300e477c7d38e78dd13c0277ffdfb4d59bb91c233f475cd3ff5f91` (bank traj sha256 `08fb25c31164660889cc98e83031aef6472c6fb42a187f1da6b16220b24e6aea`). It is a **genuine P5-R defect**, not a gate artifact: the P5-R tail heading flips 3.578 rad on a near-standstill RJ5 trajectory. Its Tc20s label is sane: |dyaw vs bank traj| max **0.5498322625075645**.
- Consequence for C1: **92,895 rank-0 rows carry a P5-R label and 1 rank-0 row carries its Tc20s label** (rank-1 rows stay raw under the forced `rank0` policy). Every report quoting C1 states this.
- Neither is a pre-registered bar. No other tolerance, gate or rule moves.

## The changes (and only these)
1. `p5c_table_gates.py` (p5r mode):
   - `--tb0-tol` (default none): TB-0's R-A tolerance. None = TB-0 uses `--tc20s-tol`, i.e. today's behaviour bit for bit. TB-1 keeps `--tc20s-tol`.
   - `--tb5-fallback-allow TOKEN:RJ5_TRAJ_SHA256` (repeatable, default none) and the record field `p5r_fallback`. A marked row is honoured only if (a) it is on the allow-list, (b) its reason is `tb5_s2_tc20s`, (c) its label equals its own `tc20s` record exactly, (d) at most `MAX_LABEL_FALLBACK_ROWS = 1` such rows exist. Any violation is a TB-5 FAIL.
   - A fallback row is NOT a P5-R row for S1/S2/S5. The base TB-5 check (finite, |dyaw vs traj| < pi, sim_hz == 10) runs on its NEW label like on every row. Any other row failing S2 is still a TB-5 FAIL.
   - TB-2: a fallback row is judged by its Tc20s knot deviation (<= 1e-9, the Tc20s TB-2 bar) instead of S3, which describes the discarded label. Its `n` is unchanged (150,603). TB-4/6/7 read it as a p5r row with its new label (measured effect: none, below).
   - New outputs only when used: `detail.label_fallback` (TB-5), `detail.label_fallback_rows_judged_by_tc20s_knot_dev` (TB-2), `p5r.tb0_tol_R_A` and `p5r_label_fallback` (meta).
2. `p5c_rj5_path.py`:
   - new `fallback` step (after `rekey`, before the main gates): the ruled row failing S2 gets `p5r := tc20s`, `p5r_discarded := the P5-R label`, `p5r_fallback := tb5_s2_tc20s`. Allow-list only, cap 1 (a longer allow-list is REFUSED), a candidate not on the list is left untouched and counted (`n_unlisted_s2_candidates`). Writes `labels_rekeyed_fb.jsonl` + manifest (`n_fallback`, `fallback_rows`, source sha256s).
   - `provenance`: a row in the manifest's `fallback_rows` gets `fallback_reason = "tb5_s2_tc20s"` and `p5r_label_replaced_by = "tc20s"` in the C1 sidecar; the a3 meta block gets `label_fallback_rows` (n, reasons, keys). A manifest without fallback rows gives the old provenance byte for byte.
   - `tb0r --tc20s-tol` (default 1e-9 = old behaviour).
3. `p5c_pod_label.sh`:
   - constants `TB0_TOL=1e-8`, `TB1_TOL=1e-9`, `FB_TOKEN`, `FB_RJ5SHA` (never env vars).
   - control gates: `--tc20s-tol "$TB1_TOL" --tb0-tol "$TB0_TOL" --tc20s-ranks all` (**A3.3 supersedes A3.2's rank-0 scope**; the rank-1 numbers are in the gate detail as part of the normal TB-0 value, n 2,400).
   - step 7b: the `fallback` call; the main gates, classifier and provenance read `labels_rekeyed_fb.jsonl` (+ manifest). Main gates get `--tb5-fallback-allow`. TB-0r gets `--tc20s-tol "$TB0_TOL"`.
   - classifier: `ZZP5CL_GATE main LABEL-FALLBACK`; `n_fallback > 1` or any unlisted S2 candidate is a fault.
   - **re-use of a finished run's shards**: `P5CL_REUSE_SHARDS` (a sha256 manifest of the shard files, relative to `P5CL_WORK`), `P5CL_REUSE_SHARDS_SHA256` (its pin), `P5CL_REUSE_FEED_MAP_SHA256`. Nothing is launched for the two labelling passes; each shard file must equal the manifest's sha256; a rebuilt feed map must equal the pinned one; running in place on `*p5c_label_a32` is refused. Unset = the A3.2 behaviour.
   - `PIN_GATES`, `PIN_RJ5PATH` updated.

## Pins (`git hash-object --no-filters`; 0 CR bytes in every file)
| file (hold `a33w/`) | blob | sha256 | was (blob / sha256 prefix) |
|---|---|---|---|
| `a1/p5c_table_gates.py` | `bc79b1fcfb52d30130671440d01baac74fb2defe` | `718290ce656457c7ca1d7b78c09ab8706835ee49a8910ae0984dc9328117bab4` | `1711a72f` / `0c567514` |
| `a1/p5c_pod_label.sh` | `68ba57056e6b06c6c78eea601b8e6a2925da4776` | `3770cbe99b1dbe912a9197995a2360dee5f779badb0d06f6f5af2cca494eab41` | `8cfe7a72` / `2fb22f24` |
| `a3/p5c_rj5_path.py` | `45a1664df8767220a10a09866ae5c77fed540580` | `117434b30cf6cb5fc1bd39923f62ea6689a45ff19cdd902047d5e9e37b50d791` | `409f4481` / `951d3e1d` |
| `a1/test_p5c_table_gates.py` | `92301bc5f5a5ede82f1bca29fdfdb325480b399b` | `1ed251ad61b13d4969d11b8d8b6fc1cedfb739de0c2fce667836aaca608f77f3` | `1695fef2` / `9bda9e9f` |
| `a1/test_p5c_pod_label_a31.py` | `c695996903b906c4357dd660b10850e03b54c961` | `d3c2da29ea58f28154237fc8978d17e7cd75bea508f93e9e1ab2601a82ea74d6` | `6830e33c` / `46a67a4a` |
| `a1/test_p5c_a33_gates.py` (new) | `34a8e2bc29c545939c0e09687a97c753c3fea353` | `32e51810f5a4e3d8cb07a085d9f7b656d5b85f712e8c03b2c6439f65a072ac9e` | |
| `a3/test_p5c_a33_rj5.py` (new) | `e126fa7a21b3f9f94e34201df71ee46d92ecec7e` | `514341f7c3838bff52c89a712067cdd599614248e403fdb3e7cacc07d2e50e42` | |
| `a3/test_p5c_rj5_path.py` (unchanged) | `c3cc16a361d96d36762032ddc24fe45c7c60d215` | `43f28014a1d1dca7397812b922ada941e7c209103ce862170e9c1f35c9bed63f` | same |

`PIN_GATES` and `PIN_RJ5PATH` in the runner equal the new sha256 values (asserted by `test_a33_runner_text_pins`; `a33w/repin.sh` recomputes them). If any source file changes, rerun `repin.sh` and the suites.

## Tests (CPU, `CUDA_VISIBLE_DEVICES=-1`, TMP on F:; logs in `a33w/proof/logs/`; env in `a33w/env.sh`, driver `a33w/runall.sh`)
| suite | result | what changed |
|---|---|---|
| `test_p5c_table_gates.py` (A3.2 gate suite) | **199/199** (same count) | 3 pin edits below |
| `test_p5c_a33_gates.py` (new) | **15 passed** | 12 mutants + real module all-True + defaults byte-identical to the A3.2 module (gates.json, meta, merged labels, 3 CLI scope variants) + merge check |
| `test_p5c_pod_label_a31.py` (runner) | **81 passed** (was 49) | A3.2's 7 rank-scope tests replaced by A3.3 equivalents, +39 new (12 scenarios, 8 mutants, classifier cap, 7 re-use scenarios, 5 re-use mutants) |
| `test_p5c_rj5_path.py` (A3) | 23 passed, file unchanged | none |
| `test_p5c_a33_rj5.py` (new) | **11 passed** (9 mutants + real + defaults == A3 module byte for byte) | |

**Pin assertions I had to update (all inside copies in this hold; no frozen file edited in place):**
1. `test_p5c_table_gates.py`, A3.2 mutant "run_all_p5r default scope 0": its target text now includes `tb0_tol=None, fallback_allow=()` in the signature.
2. same file, A3.2 mutant "scope not passed through run_all_p5r to the gate": target and replacement carry the new `tb0_tol` argument.
3. same file, `RUNNER` reads `P5C_A1_RUNNER` (default unchanged): its R-D tests parse the A1 runner's 6b block, which is not the runner beside it in this layout.
4. `test_p5c_pod_label_a31.py`: the A3.2 section (`scenario_ra_rank1_only`, `scenario_ra_rank0_fails`, `scenario_scope_flag_only_on_control`, `test_a32_runner_text_pins`, 3 `A32_MUTANTS`) asserted `--tc20s-ranks 0` and tolerance 1e-9 on the control call. **A3.3 supersedes exactly that**, so these became: rank-1 1.5e-9 PASSES with all ranks judged (n 8, max 1.49e-9, tol 1e-8); rank-0 2e-8 FAILS; rank-1 2e-8 FAILS; text pins for `TB0_TOL` / `TB1_TOL` / `--tc20s-ranks all` / the fallback call; mutants rewritten. The STUB labeller gained an opt-in `STUB_S2FAIL` env (default off) to make a fixture row fail S2.

**Every mutant goes RED** (the module or runner text edited, the same checks must read at least one False):
| required mutant | where it is killed |
|---|---|
| tolerance applied to TB-1 | gates `tolerance applied to TB-1`; runner `a33_tolerance_applied_to_tb1` |
| tolerance ignored | gates `tolerance ignored`, `tolerance not passed from run_all_p5r`, `CLI drops --tb0-tol`; runner `a33_tb0_tolerance_ignored` |
| fallback applied to an unlisted row | gates `allow-list not enforced`, `CLI drops the allow-list`; rj5 `applied to an unlisted row`; runner `a33_gates_allow_list_dropped` |
| fallback applied to 2 rows | gates `cap lifted`; rj5 `cap lifted`; runner `a33_fallback_applied_to_more_rows`, classifier cap mutant |
| fallback leaves the P5-R label in place | gates `label == own Tc20s not required`; rj5 `leaves the P5-R label in place`, `marker not written` |
| fallback not recorded in provenance | rj5 `not recorded in the provenance sidecar`, `not counted in the provenance meta`; runner `a33_fallback_not_recorded_in_provenance` |
| TB-5 skips the new label's dyaw check | gates `TB-5 skips the new label's dyaw check` |
| tb0r tolerance ignored | rj5 `tb0r tolerance ignored` (function and CLI); runner `a33_tb0r_tolerance_not_passed` |
Extra: S1/S2/S5 exclusion dropped, marker reason ignored, TB-2 knot check dropped, rekeyed-file integrity unchecked, fallback step skipped; and five re-use mutants (sha not checked, manifest pin dropped, feed map not checked, in-place allowed, shard presence not checked).

## Real-data proof (pod, read-only against run 3; new dir `/workspace/data/p5c_a33_check/`)
New files shipped there (sha256 verified): gates `718290ce...`, rj5 path `117434b3...`. Nothing was written into `p5c_label_a32`, `p5c_ship` or `p5c_label` (the only other write: the copy `p5c_label_a33`, 525 files, sha256 of every file equal to the source, `cmp` of the two sorted manifests). Banked outputs: `a33w/proof/pod/`. Steps: `a33w/proof/podproof.sh`, compare: `a33w/proof/podcompare.py`, result text `proof/pod/podcompare_out.txt`.
| step | result |
|---|---|
| fallback on run 3's rekeyed labels | **n_fallback 1, n_unlisted_s2_candidates 0**, 150,603 records; only that row differs in the merged table |
| control gates, new (tb0 1e-8, all ranks) | **TB-0 PASS**, Tc20s part max **1.4901395672950457e-09**, n 2,400, tol 1e-08; TB-1 PASS; TB-2..7 byte-identical to run 3 |
| control gates, default (1e-9, ranks 0) | byte-identical to run 3 on every gate, rank policy and failing list |
| main gates, new (fallback + allow-list) | **TB-5 PASS 150,603/150,603**, `label_fallback` n 1, violations none, new label's max |dyaw| 0.5498; TB-3/4/6/7 byte-identical to run 3; TB-2 identical except the added detail key (n 150,603, max knot dev 2.8e-17); rank policy identical (`all`); failing = `TB-0`, `TB-1` (NOT_RUN on the main set, as in run 3, judged on the control pass) |
| main gates, default (no fallback, original labels) | byte-identical to run 3 on all 8 gates and the rank policy; TB-5 FAIL 150,602/150,603 |
| tb0r, new (`--tc20s-tol 1e-8`) | **PASS**, max 2.547964894628052e-09, n 16,133, R-B ok |
| tb0r, default | FAIL at 2.547964894628052e-09, identical to run 3's `tb0r.json` apart from bar text and the gates-module sha |
Meta: only new key `p5r_label_fallback`; with the old arguments the meta differs from run 3's only in the two code blob shas.

## The reuse run (the MM launches it after the freeze; I did not)
Ship to the pod first (the runner pins them): the frozen `p5c_pod_label.sh`, `p5c_table_gates.py`, `p5c_rj5_path.py` into `/workspace/data/p5c_ship/` (sha256 as in the pin table). The inputs below already exist: `/workspace/data/p5c_label_a33` (copy of run 3's work dir), `/workspace/data/p5c_a33_check/SHARDS_run3.sha256` (30 shard shas, itself `614562c2206722ec613b81e8099f506fd27d2109be8769f58c2503797f12e35e`), feed map `38a94869962da5c4de9d21b97399c3981bdcb21b486aa6b78f8f5263115d8fbc`.
```
cd /workspace/data && P5CL_LABEL=p5r-rj5 P5CL_NSHARD=15 P5CL_SHIP=/workspace/data/p5c_ship P5CL_WORK=/workspace/data/p5c_label_a33 \
 P5CL_JOIN=/workspace/data/p5c_ship/rj5_bank_join.jsonl.gz P5CL_JOIN_SHA256=4670f88ac409e71a1f7e1bf02328d93229fb3e153ad63f4127f08d53bbfed739 \
 P5CL_A31_SHA256=cda1567b52aaf53180eb591a87d54453903d55d0d2cc04125d932dd5d5258d8f P5CL_PLATFORMS="A=devbox B=devbox C=pod local=devbox" \
 P5CL_R0_ROWS="A=/workspace/data/p5c_r0rows/A B=/workspace/data/p5c_r0rows/B local=/workspace/data/p5c_r0rows/local" \
 P5CL_REUSE_SHARDS=/workspace/data/p5c_a33_check/SHARDS_run3.sha256 P5CL_REUSE_SHARDS_SHA256=614562c2206722ec613b81e8099f506fd27d2109be8769f58c2503797f12e35e \
 P5CL_REUSE_FEED_MAP_SHA256=38a94869962da5c4de9d21b97399c3981bdcb21b486aa6b78f8f5263115d8fbc \
 nohup nice -n 19 bash /workspace/data/p5c_ship/p5c_pod_label.sh > /workspace/data/p5c_label_a33.log 2>&1 &
```
`P5CL_PLATFORMS` is read from run 3's feed map (A, B, local = devbox; C = pod); the R0 dirs are the three under `p5c_r0rows` (the feed-map sha check refuses any mismatch). Expected log: `ZZP5CL_REUSE`, `ZZP5CL_REUSED control` / `rj5` (no `launched` lines), `ZZP5CL_FALLBACK ... n_fallback 1`, `ZZP5CL_GATE control TB-0 PASS`, `ZZP5CL_GATE main TB-5 PASS`, `LABEL-FALLBACK OK n=1`, `TB-0r PASS`, class `pass`, then the table. Est. runtime: minutes (rekey, 4 repair runs, gates, table build; no labelling).

## Not changed
A3, A3.1, A3.2 files in place; R-B; TB-1 (1e-9); the other gates' bars; the rank policy; C0; the feed, the labeller and the table builder.
