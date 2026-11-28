# OUTCOME_BUILD -- `eval/safeobj_outcome.py` (IMPL_PLAN step 8, second half)

Built 2026-10-17 by a TanitAD build agent (Master Mind brief). STAGED ONLY: nothing committed, pushed, landed; the pod and git were not touched.
Package: `C:/Users/Admin/sobj_pkg/raw/2026-10-17-safety-objective/outcome/`. Working copy the tool was built and run in: `C:/so/oc/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval/`
(a copy of the tree; `eval/safeobj_read.py` was imported, never edited; `refe/rls1_checks.py`, `build/apply_all.sh`, `BUILD.md`, `TEST_TABLE.md`, `build/`, `deliverables/`, `C:/so/t4`, `C:/so/t6`..`C:/so/t8` were not touched).

**Verdict (after addendum 1): BUILT and TESTED. The tool applies bars G, L, E and the table O1-O7 of `PREREG_SAFEOBJ.md` WITH `PREREG_SAFEOBJ.addendum1.md` (R1-R4) mechanically, fails closed, and REFUSES TO RUN (rc 4) unless the addendum is the registered file (sha256 `3c6a25c14d42...`).
No reading is left to the tool's discretion except the six the addendum itself does not settle (section 0.5), and the record names each of them.
It is NOT yet wired into the chain: the chain does not write `rlval_B_vs_ownA.json` (bar G's input) and nothing produces `navtest_E.json` (bar E's input, IMPL step 9). Both are named below with the exact command / contract.**

## 0. ADDENDUM 1 APPLIED (supersedes sections 3 and 4 below, which describe the pre-addendum tool and are kept only as the audit trail)

Tool string `safeobj_outcome/2`. Registered, pinned files: `PREREG_SAFEOBJ.md` sha256 `b404c2fb7b2f724775a1dda5a9de06c9778505a3374e6467060ae7c7a9e04bcb`; `PREREG_SAFEOBJ.addendum1.md` sha256 `3c6a25c14d4266230281b03d92d8201d21dbaa0ab27d9699f934af69a96761c4`;
`dose.json` sha256 `735a6d909bc0ef41b87421bed225d3b206b0b3bd886a5e0a9776088c2e8d9d5c`. Every `safeobj_outcome.json` and every arm record in it carries `addendum: "PREREG_SAFEOBJ.addendum1.md"` and that sha256.

### 0.1 R1 -- bar G
* **dose.json's exact vertices, read from the file** (key `full[alpha].{TTC,PDMS}.mean`, alphas 0.25 / 0.375 / 0.5, x100), after the file's sha256 equals the pinned one; any other bytes, a missing file or a missing key reads **G NOT_READ**. The printed 2-decimal vertices are a cross-check only (`G_POINTS_PRINTED`; the record says whether they round-match).
* Floor **+0.60 exact**, comparison **`>=`** as written, lower 95 % bound **> 0**. The record names the vertices with their source file and sha256 and the read it judged (`rlval_B_vs_ownA.json`, its path and sha256), the pool counts (3,000 sets / 75 logs; a different pool is a note, not a refusal: A-G3b) and `disclosures.A-G2 = NOT_COMPUTED_BY_THIS_TOOL`.
* **The vertex test (replaces the IMPL_PLAN selftest, which was wrong -- the addendum says so):** alpha 0.25 reads **G FAIL by 0.00125** (margin -0.0012495138888889; the point sits BELOW the +0.60 floor, so the floor binds); alpha 0.375 and alpha 0.5 sit **exactly on F** (margin 0.0) and read **G PASS** by `>=`. MEASURED: tests `G:vertex_alpha_*`.

### 0.2 R2 -- the fixed precedence **O6 > O7 > O4/O5 > O1/O2/O3**
The first stage that matches is the outcome; the rows it beat are recorded under `table.shadowed`. An unknown fact that could win at an earlier stage reads **NOT_READ** (Kleene three-valued logic); it never falls through to a later row.
* **O6** (lambda_TTC = lambda_NC = 0.0 at every logged update, through the last logged update) beats everything, including O1 and O7.
* **O7**: any kill whose RECORDED RULE is not K3, or "g3 alone" (S fails and no g5 breach at any check). **g3-alone is O7, not O4/O5** (O4/O5 need `breached` = a K3 kill, or S fails with at least one g5 breach).
* **O4/O5**: exactly one row -> that row; **both match -> `O4+O5`** (a determinate outcome, exit code 0); **neither -> `O4` flagged `SLACK`** (the flag is in `outcome_flags`, printed as `[SLACK]`).
* **K-MON is judged by its recorded reason, not its code:** a reason matching `g3/g5 bound breached at <tag> and <tag>` (the real monitor's K3 text, `rls1_checks.judge`) counts as K3 whatever the code; the chain's own `kill rules unenforceable: monitor died rc N` / `not READY within Ns` are not the K3 rule, so that kill is O7.

### 0.3 R3 -- the trainer's log
* **Signed per-term window means:** `con_g_ttc` and `con_g_nc` (the 25-update mean of the signed per-update g the dual step consumes); violation **held iff both window means are <= 0.001**. The positive part is NOT used (a window oscillating about 0 reads held).
* **Window** = the 100 logged updates immediately before (ending AT) the first breached check; for a K3 kill, the first of the two consecutive breaches. A window that overlaps the lambda_pg ramp is **flagged in the arm's notes** (`window_overlaps_ramp`), not moved.
* **Post-ramp share** (lambda > 0 on >= 50 % of updates 201..c) and **"lambda reached lambda_max"** (cap hits, or running max >= lambda_max, from update 0) are both counted **through the first breached check c**, not over the whole run. O6 is read through the last logged update (an unlogged tail under 25 updates is disclosed).
* The tool names the logged columns it reads (`log.columns_read`).

### 0.4 R4
* **NOT_APPLICABLE / PENDING:** bar L is PENDING until R1o is given (never PASS); bar E is PENDING until `--navtest-e` is given. A killed arm has no u1000, so its G reads NOT_READ.
* **L is decided on TTC only** (effect > 0 and > 2 x F_seed on both seeds, F_seed = the larger of |dTTC(R0 vs R1o)| and |dTTC(R0d vs R1d)|); the PDMS effect and its yardstick are reported beside it and never move the verdict (a PASS with a PDMS cost is tested).
* **E**: the pooled contrast is the mean of the two pairs' contrasts with a **JOINT** resample of the 136 logs (`pooled.resample == "joint"`; `n_tokens == 12146`; `n_logs == 136`; the pooled mean must equal the mean of the pairs, to 1e-9); **clause 4 is per pair** (TTC and NC upper bounds >= 0, four bounds);
  the seed floor is the **larger of the RL-val training-seed floor (PDMS: max(|R0d - R1d|, |R0 - R1o|) from bar L's inputs) and the navtest seed floor |R0d - R1d|**, then the I-rep. A `floors.seed_floor_abs` in the file is IGNORED (the tool derives the floors). Without R0 and R1o the floor cannot be built, clause 3 is unread and E reads NOT_READ unless REFUTED.

### 0.5 What the addendum does NOT settle (listed in every record under `prereg_ambiguities`; 19 settled ids under `settled_by_addendum`)
| id | the open reading | how the tool reads it |
|---|---|---|
| **A-E3b** | section 7's training-seed row names two quantities and does not say PDMS or TTC | PDMS (E's unit), the larger of the two; both terms are in the record |
| **A-G2b** | the own-A vs C0-A difference "beside G": the tool is not given C0-A's u1000 dumps | NOT computed; the owner reports it beside G |
| **A-O9b** | the addendum gives no text for the K-MON "recorded reason" that counts as K3 | the real monitor's K3 text (regex pinned in a test against `rls1_checks.py`); any other reason is O7 |
| **A-G3b** | "(3,000 sets / 75 logs)" is parenthetical | a different pool is a note on the arm, not NOT_READ |
| **A-E1b** | the JOINT pooled interval is the producer's (IMPL step 9) | only the declaration, the counts and the pooled mean are checked; the interval is taken from the file |
| **A-L1b** | "the same 2 x F_seed yardstick printed beside" the PDMS cost | recomputed on the PDMS contrasts; the TTC F_seed that decides L is printed beside it; neither moves the verdict |

### 0.6 Evidence (MEASURED, ours; Git bash `"C:/Program Files/Git/bin/bash.exe"`, `driverl-eval` venv, CPU only; the staged `.py` files are byte-identical to the ones that ran: sha256 in section 1)
* **`eval/test_safeobj_outcome.py`, full form: `ZZTEST_PASS 133  (1 unmutated suite of 172 checks + the CLI end to end, 132 RED mutations)`, `ZZEXIT rc=0 seconds=678`.** Log: `outcome/ZZTEST_safeobj_outcome.log` (132 `RED` lines). Every mutation is asserted to go RED **at the checks it exists for**.
  The pre-addendum run (INHERITED, superseded; `ZZTEST_PASS 99`, 142 checks, 98 mutations, 327 s) is kept as `outcome/ZZTEST_safeobj_outcome.pre_addendum1.log`.
* **`eval/test_safeobj_read.py`, unchanged, full form: `ZZTEST_PASS 34  (1 unmutated suite of 26 scenarios + boundary + end-to-end + pins, 26 RED mutations, 7 fixture-sanity checks)`, `ZZEXIT rc=0 seconds=11`.** Log: `outcome/ZZTEST_safeobj_read.log` (the pre-addendum copy, `*.pre_addendum1.log`, differs only in the seconds: 10 vs 11).
* The 172 checks: 45 `outcome:`, 13 `G:`, 25 `notread:`, 12 `L:`, 40 `E:`, 5 `joint:`, 7 `record:`, 1 `refuse:`, 2 `pending:`, 1 `tolerated:`, 1 `table:`, 17 `pin:`, 3 `real_main:`; then 15 `e2e:` (the CLI end to end: rc 0 / 2 / 3 / 4).
* **One scenario per precedence rule** (each from a real-producer arm): O6 beats O1 (`o1_inert`), O6 beats O7 (`o6_beats_o7`), O6 beats O3 (`o3_o6_tie`); **g3-alone is O7 not O4** (`o7_g3_o4`: O4 and O5 do not match at all) and O7 beats a g5 breach (`o7_beats_o4`); **the overlap is O4+O5** (`o5_cap_o4_tie`); **the gap is O4 flagged SLACK** (`gap`, `gap_k3`, `gap_share_below`);
  **a K-MON kill with the K3 reason counts as K3** (`kmon_k3`: the real K3 `KILLED.json` with its code rewritten) while a monitor-died K-MON is O7 (`kmon_monitor_died`); the window ends at the FIRST of two isolated breaches (`iso_two`) and at the first of a K3 pair (`k3_late_pair`).
* **The five RED mutations the brief names**, each asserted to fail where it exists to: *precedence reordered* (O7 before O6; O4/O5 before O6/O7; O1/O2/O3 first; O4/O5 before O7; and the equivalent O1-O3-before-O4/O5, which only the addendum-text pin can see); *positive part instead of the signed mean* (`outcome:osc_signed->O4`, `terms_max`); *window moved* one row earlier / later / to 50 updates / ended at the last check;
  *printed vertices used* (the three `G:vertex_alpha_*` checks); *addendum sha mismatch not refused* (`refuse:` and `e2e:rc4_*`). The others cover every other R1-R4 item (dose.json sha not checked, floor, `>=`, the share counted from update 0 or over the whole run, lambda_max over the whole run, the K3 reason regex, the K3 pair's first / last check, the pooled mean / joint resample / n_tokens / n_logs of E, the floors of E, L requiring a PDMS gain, the addendum stamp, ...).
* Fixtures: REAL = the per-check record, `kill_checks.jsonl`, `KILLED.json` (K2 K3 K4 K6), `kill_replay.json` (`rls1_checks.RlValChecker / monitor / replay`); `rlval_vs_A0.json` / `rlval_B_vs_ownA.json` (`rl_val_eval.run`, 10,000 resamples); `safeobj_S.json` (`safeobj_read`); the dual's trajectory in `sft.jsonl` (the real `scorer_finetune.EpoDual` inside the trainer's own row-assembly text, pinned) and ONE real `scorer_finetune.main()` run.
  SYNTHETIC (no producer exists, labelled in the test): every `navtest_E.json`; exact-boundary numbers (dose.json's alpha points, a PDMS point exactly on G(t), dyadic L ties, a window violation of exactly 0.001); a `KILLED.json` code `K-MON` (a shell-level event); the L controls' dyadic deltas.
* **Compatibility, INFERRED by reading a diff and NOT re-run:** the tool and its tests ran against `build/safeobj_read.py` (sha256 `0154772b0f6d...`, the working copy's). `deliverables/eval/safeobj_read.py` (sha256 `b1b17341d256...`, written 13:48 by another agent) differs by a refactor of the digest into `record_digest()` and added `differing_fields()` / `comparable()`; the canonicalisation is the same, so the S freshness check is expected to behave identically -- re-run both suites against the delivered reader before landing.

## 1. Manifest (all in this directory)

| file | what | sha256 (of the staged copy) |
|---|---|---|
| `safeobj_outcome.py` | the tool (stdlib only; imports `eval/safeobj_read.py`) -> `safeobj_outcome.json` | `e534e691d69f202b172a6057aa30a396e6b116d102b92a4502aa00f07e333faf` (983 lines) |
| `test_safeobj_outcome.py` | its self-test, real-producer fixtures, RED mutations | `61e32c8cca1500eca00d8d25b7c733c34467e9a7d837921c3940fdd29c6329f3` (1,727 lines) |
| `ZZTEST_safeobj_outcome.log` | the FULL-form run of the test with addendum 1 applied (MEASURED) | -- |
| `ZZTEST_safeobj_read.log` | `eval/test_safeobj_read.py` run unchanged beside it (MEASURED) | -- |
| `ZZTEST_safeobj_outcome.pre_addendum1.log`, `ZZTEST_safeobj_read.pre_addendum1.log` | the runs before addendum 1 (superseded; audit trail) | -- |
| `OUTCOME_BUILD.md` | this file | -- |

Copy both `.py` files into `eval/` of the package tree (they import `safeobj_read`, `test_safeobj_read`, `rl_val_eval`, `rls1_checks`, `scorer_finetune`, `test_rl_epo` by relative location: `eval/` and `refe/`). The test also reads `raw/2026-10-17-safety-objective/{PREREG_SAFEOBJ.md,PREREG_SAFEOBJ.addendum1.md}` and `raw/2026-10-14-rls1b-attribution/dose.json` relative to that tree: they must be there.

## 2. What each bar reads (every number in `safeobj_outcome.json` carries `{value, source, sha256}`) -- current reading, addendum 1 applied

| bar | input | rule applied | missing / unreadable |
|---|---|---|---|
| **S** | `safeobj_read.read_arm(arm_dir)`, re-run in the tool and required to EQUAL the arm's `safeobj_S.json` (self-hash and the fresh record) | PASS / FAIL / NOT_READ as the reader says | stale or hand-edited S file -> NOT_READ |
| **G** | `<arm>/rlval_B_vs_ownA.json` (paired B - own-A with interval), cross-checked against `<arm>/rlval_vs_A0.json` (TTC and PDMS means within 1e-9, same `sets_used`, `scores_sha256` tied to the real npz files); `dose.json` (sha256-pinned) | PASS iff PDMS lower 95 % bound > 0 AND PDMS point >= G(t), t = the TTC point delta; G(t) = max(+0.60, F(t)), F = linear interpolation of dose.json's exact alpha 0.25 / 0.375 / 0.5 points, flat beyond | NOT_READ |
| **L** | `<arm>/rlval_vs_A0.json` of R0 (banked), R0d, R1d, R1o | **TTC only**: effect = (constrained B-A) - (unconstrained B-A) on BOTH seeds, each > 0 and > 2 x F_seed; PDMS reported, never decides | **PENDING** iff R1o not given; NOT_READ for any other gap |
| **E** | `--navtest-e navtest_E.json` (contract in section 7; no producer exists) | per pair PDMS >= +0.25 and mean >= +0.5; pooled (JOINT) lower > 0; mean contrast > 2 x the larger floor; TTC and NC upper bounds >= 0 per pair; REFUTED iff pooled upper < +0.25, else NOT_PROVEN; ADOPT and REFUTED together -> NOT_READ | **PENDING** iff no file is given; NOT_READ for a bad file |
| **O1-O7** | S, G, the kill codes AND reasons (`KILLED.json`, `kill_replay.json`), and the trainer's `con_*` columns in `sft.jsonl` | the table with the precedence of section 0.2, in Kleene three-valued logic | unknown fact that could change the answer -> NOT_READ; PENDING only for an arm directory with nothing read yet |

How O4 / O5 / O6 are read from the log (`sft.jsonl`, the trainer's own writer; MEASURED against the real `main()` in the test): one `train` row per 25 updates; `con_g_*` = the 25-update window mean of the signed g; `con_lam_*` = lambda after the step;
`con_lam_pos_updates` = updates with lambda > 0 at the START; `con_cap_hits` = updates with raw >= lambda_max; `con_lam_max_*` = the running max. The `data` event holds the BUILT `epo.ramp` (200) and `epo_con` block, which must equal the registered values; the `done` event's state is cross-checked with the last row.
"end of run unknown" (a log with neither a `done` event nor a `KILLED.json`) is **NOT_READ**, not PENDING, when the chain's reads exist.

Exit code: **0** only if every constrained arm got a determinate outcome (O1-O7 or O4+O5); **2** otherwise (NOT_READ / PENDING); **3** usage; **4** the addendum is missing or not the registered file (`ZZSAFEOBJ_REFUSED`, no record written).
Output lines: `ZZSAFEOBJ_OUTCOME <arm> <O#|O4+O5|NOT_READ|PENDING> [flags] <why>`, `ZZSAFEOBJ_BAR <S|G|L|E> <verdict> <why>`, `ZZSAFEOBJ_JOINT <...>`, `ZZSAFEOBJ_FILE <path> record_sha256=... addendum_sha256=...`.

## 3. (SUPERSEDED by section 0 -- the pre-addendum evidence, kept as the audit trail)

The pre-addendum tool read G on the 2-decimal printed vertices, broke no tie (AMBIGUOUS / NO_ROW for the O4/O5 overlap and gap, O3/O6 tie, an unlisted K-MON kill) and ran 99 passes (142 checks, 98 mutations, 327 s; `ZZTEST_safeobj_outcome.pre_addendum1.log`).
Its IMPL_PLAN selftest ("an arm at dose.json alpha 0.25 sits on G's boundary") could not be satisfied and was replaced by the vertex test of section 0.1.

## 4. (SUPERSEDED by section 0 -- the prereg ambiguities found before the addendum)

A-G1, A-G2, A-G3, A-O1 ... A-O10, A-L1, A-L2, A-E1 ... A-E4 were listed in every record and none was resolved by the tool. The addendum settles 19 of the ids (`settled_by_addendum` in every record); the six it leaves open are in section 0.5.

## 5. Findings for the Master Mind (no action taken on any)

1. **The chain writes NO paired B - own-A interval.** `rlval_vs_A0.json` has A0 / CTRL / R each against the BASE model. Bar G needs the interval of B - own-A, so ONE extra `rl_val_eval.py --base A` call per constrained arm is required (section 6, step 2). Without it G reads NOT_READ and O1 / O2 / O3 cannot be decided.
2. **Nothing in the chain runs the S reader, the extra `rl_val_eval` read, or this tool.** MEASURED by grep over `refe/*.sh`, `refe/rls1_checks.py`, `eval/*.py` of the tree copy used here: `safeobj_read` / `safeobj_outcome` are named only in their own files and tests;
   `rls1_checks.py safeobj-gate` (line 577) only READS `<arm>/safeobj_S.json`. So **R1o's gate refuses (missing S file) unless step 1 below was run by hand after R0d and R1d.** The chain's second `rl_val_eval` read (`rls1_pod.sh` lines 589-601) is against the seed-matched C<n>
   (`--base C`), not against the arm's own A, so it is not bar G's input. Wiring all three in is an integration step for the owner of `rls1_pod.sh` / `build/apply_all.sh` (another agent is editing those; not touched here) -- this hand-back is the escalation.
3. **IMPL step 9 (navtest adapter) is the only thing between the tool and bar E.** E stays PENDING, never PASS, until `navtest_E.json` exists, and it must carry the JOINT pooled interval (section 7).
4. **R1d needs the PI's Stage-2 decision; R1o runs only if both S files read PASS** (`rls1_checks.py safeobj-gate`). L is PENDING until R1o. The joint rule ("R0d O1 and R1d not -> NOT REPLICATED") is reported in `ZZSAFEOBJ_JOINT`.
5. ~~The O table needs a precedence statement~~ **RESOLVED by addendum R2** (section 0.2). The K-MON reason text (A-O9b) and the unit of the training-seed floor (A-E3b) are the two open readings most likely to matter on a real arm: the first decides O7 vs O4 for a monitor-killed arm.

## 6. Exact pod commands for after R0d / R1d / R1o (run from `/workspace/refe-plan-rls1`; paths are the chain's own, `rls1_pod.sh`)

```bash
R=/workspace/data/rls1
PART=/workspace/refe-plan-rls1/raw/rl_pool/partition_runs; PNAME=A_no_trained_list_seed20261012_n16000
VAL_TSV=$PART/$PNAME/rl_val_scenes.tsv
PY="$DRIVERL_EVAL_PYTHON"; cd /workspace/refe-plan-rls1

# 1. bar S (writes <arm>/safeobj_S.json). `rls1_checks.py safeobj-gate` only READS that file and no chain script runs the reader (finding 2), so R1o's gate REFUSES until this has run by hand
$PY eval/safeobj_read.py --arm-dir $R/arms/R0d --arm-dir $R/arms/R1d

# 2. the paired B - own-A read WITH its interval, per constrained arm (bar G's input; NOT written by the chain today)
for ARM in R0d R1d; do
  $PY eval/rl_val_eval.py \
    --scores A=$R/arms/$ARM/scores/scores_update1000_A.npz B=$R/arms/$ARM/scores/scores_update1000_B.npz --base A \
    --pdm $R/pdm_val --terms $R/terms_val --sets $R/sets_val --rl-val-scenes $VAL_TSV \
    --require-full --rule navsim_v1_lane --out $R/arms/$ARM/rlval_B_vs_ownA.json
done

# 3. the outcome, WITH THE ADDENDUM (the tool exits 4 without the registered file).  R1o and the banked R0 only once they exist; add --navtest-e when IMPL step 9 has written it
$PY eval/safeobj_outcome.py --arm R0d=$R/arms/R0d --arm R1d=$R/arms/R1d \
    [--arm R1o=$R/arms/R1o --arm R0=<the banked R0 arm dir, with its rlval_vs_A0.json>] [--navtest-e <navtest_E.json>] \
    --dose raw/2026-10-14-rls1b-attribution/dose.json --prereg raw/2026-10-17-safety-objective/PREREG_SAFEOBJ.md \
    --addendum raw/2026-10-17-safety-objective/PREREG_SAFEOBJ.addendum1.md \
    --out $R/arms/safeobj_outcome.json
echo rc=$?        # 0 = every constrained arm has a determinate outcome (O1-O7 or O4+O5); 2 = NOT_READ / PENDING (read `problems`); 3 = usage; 4 = the addendum is not the registered file
```
The flag spellings in step 2 were checked against the `add_argument` list of `eval/rl_val_eval.py` and the chain's own `$RV` line (`rls1_pod.sh` line 586); the command itself was NOT run on a pod. Read the result with
`grep '^ZZSAFEOBJ' <log>` and open `safeobj_outcome.json`: `problems`, `outcome_flags` (`SLACK`), `prereg_ambiguities` and the arm `notes` (ramp overlap, unlogged tail) are the part a human must read.

## 7. File contracts the tool reads that have no producer yet

`navtest_E.json` (IMPL step 9 must write exactly this; all values x100; the tool refuses any other tier or estimator string, any other `n_tokens` / `n_logs`, a pooled mean that is not the mean of the pairs, and a pooled interval not declared `"joint"`):
```json
{"tier": "T1", "estimator": "paired log-cluster bootstrap, 10000 resamples, 95 %, seed 20260927", "n_tokens": 12146, "n_logs": 136,
 "pairs": {"R0d": {"PDMS": {"mean": 0.0, "ci95": [0.0, 0.0]}, "NC": {}, "DAC": {}, "EP": {}, "TTC": {}, "C": {}, "DDC": {}}, "R1d": {}},
 "pooled": {"PDMS": {"mean": 0.0, "ci95": [0.0, 0.0]}, "resample": "joint"}, "floors": {"irep_abs_delta": 0.0}}
```
(every sub-score object is `{"mean": m, "ci95": [lo, hi]}`; the empty `{}` above only shortens the example; exactly the pairs R0d and R1d; `ci95[0] <= ci95[1]`; `pooled.PDMS.mean` = the mean of the two pairs' PDMS means; **`pooled.resample: "joint"` declares ONE index set of the 136 logs per draw applied to BOTH pairs**; there is no `seed_floor_abs` -- the tool derives the floors from the RL-val reads and from |R0d - R1d|, and ignores one if present.)

## 8. Not done, by design
* Not committed / landed / pushed; not run on a pod; no GPU.
* No ambiguity resolved beyond the addendum, no threshold changed: the committed numbers are pinned in the test against the prereg TEXT (sha256 `b404c2fb7b2f...`), the ADDENDUM text (sha256 `3c6a25c14d42...`) and the trainer's constants.
* Bar E and L are not exercised on real data: E has no producer; L needs R1o. The own-A vs C0-A disclosure beside G (A-G2b) is not computed by the tool.
