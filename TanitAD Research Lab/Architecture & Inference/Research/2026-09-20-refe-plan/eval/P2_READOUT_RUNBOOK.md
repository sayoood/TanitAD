# P2 readout runbook: R1 then R2, on the dev box, 0 pods (2026-10-12)

> Rule: `eval/PREREG_P2_AMENDMENT_1.md` (sections 3-5) and `eval/PREREG_P2.md`. Nothing here sets a number after seeing a P2 score;
> no P2 score exists. Instrument for R2: `eval/stage3_full.py` (tests `eval/selftest_stage3_full.py`).
> ⛔ Never ssh to a pod to do any of this. P2's checkpoint comes off HF (`Sayood/tanitad-refe-p2-joint-pdm`, `run/model_final.pt` + `run/summary.json`).
> Tier of every number: T1 (NAVSIM v1.1 PDMS, self-action open loop). Interval: paired log-cluster bootstrap, 10,000, seed 20260927; it answers
> *another draw of logs* only (H-ESTIM-SEED-1: one seed per arm, training variance unmeasured).

## 0. State on 2026-10-12: R2's instrument precondition is MET (MEASURED, dev box)

| amendment section 3 requires | result | where |
|---|---|---|
| u400 paired with itself reads +0.00 [0.00, 0.00] on 12,146 tokens | **+0.0 [0.0, 0.0] exactly**; better / worse / tied 0 / 0 / 12,146; all six sub-scores exactly [0, 0]; 136 logs | `raw/p2_r2_instrument/selfcheck.json` `checks.self_pair_u400`, and through the whole pipeline `checks.pipeline_self_pair` |
| swapping the two arms flips the sign exactly | point estimate **bit-exact**; CI ends exact to **<= 1.8e-15** (a 1-ulp difference on one end, e.g. +3.047028951778753 vs +3.0470289517787528). ⚠️ NOT bitwise on every CI end: `np.percentile`'s interpolation is not bit-antisymmetric. The check gates on tolerance 1e-12 and records the bitwise flags. At the 2-dp precision of every report the flip is exact | `checks.swap_flips_sign`, `checks.pipeline_swap` |
| token file md5 equals `3a8b3653...` | `3a8b3653fd70891e50a230a37eb76ae8`, 12,146 tokens, 136 logs, equal to the export's token set | `checks.token_file_md5` |
| u400's per-token table md5 recorded | CSV md5 `b17c4fbac26d064f17ce790dca4d9d08` (sha256 `df005c47...1051`), seam md5 `b009e941620b6cbbbe63e1dd8a94a506`; the CSV's means reproduce the harness summary to 4 dp (88.2469 / 98.942 / 96.575 / 81.196 / 95.6282 / 99.9671 / 97.6618) | `checks.u400_table_recorded`, `checks.u400_csv_reproduces_harness_summary` |
| (added) known value | u400 minus deployed through the new code = **+3.7908187056340563 [3.0470289517787528, 4.542726430876108]**, 5,167 / 3,188 / 3,791, equal to `RESULT_SFT4.md`'s `result.json` **to the bit** | `checks.known_value_u400_vs_deployed` |
| (added) the decision table on real tables | u400-vs-deployed = GO; deployed-vs-u400 = NO-GO (P2 below u400; adverse NC, DAC, EP, TTC); u400-vs-u400 = NO-GO (NOT PROVEN); u400 with EP made worse = NO-GO (adverse EP) | `checks.decision_table_rows` |
| (added) the families6 fallback | the repo copy of `families6.py` (md5 `964c4ae0...`) reproduces u400's banked families block with **0 differences** in 61 s | `checks.families6_reproduces_banked_u400` |
| deliberate regressions (each MUST fail its check) | shuffled token->score mapping fails the self-pair (CI [-1.16, +1.25], 979 tied of 12,146); a reverse read with the wrong seed fails the swap (CI dev 0.016); a token file one short fails md5 and n; the deployed CSV is refused where the u400 pin is checked; the whole pipeline on an arm table missing one token reads `INVALID` (failed gate: `rows`) while the same plumbing with the whole table reads normally | `regressions.*`, all `failed_as_required: true` |

**R1's instrument is byte-identical:** `eval/stage2_sft.py` is untouched (md5 `831a8d76bb5b8be2016288b02cd29ee5`, recorded in `selfcheck.json`).
R2 is a *sibling* that reuses its estimator (`proxy_eval.boot`, `read_csv`, `fam_adverse`), not a flag on it.
Re-verify in ~2 min before the read: `$PY eval/selftest_stage3_full.py` (11/11) then `$PY eval/stage3_full.py --selfcheck` (`ZZSELFCHECK ... R2_PRECONDITION_MET=True`).

## 1. Setup (once)

```bash
cd "D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
export PYTHONIOENCODING=utf-8
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
DATA=E:/Projects/TanitAD/data/refe_navtest
CK=E:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt        # where you put P2's file
OUT=<a scratch dir for .out files>
```
1. Pull `run/model_final.pt` and `run/summary.json` from the HF repo (read-only). `sha256sum $CK` must equal the sha256 in `summary.json`; keep that value as `$SHA`.
   ⛔ A result exists only when the checksum matches (`PREREG_P2_AMENDMENT_1.md` section 2).
2. Pre-flight (each is a one-line check):
   - `diff -rq --exclude=__pycache__ --exclude=raw eval "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval"` prints nothing.
     ⚠️ `stage2_sft.py` runs the seam from, and writes its raw output under, the **E: mirror** of this package; D: and E: were identical on 2026-10-12 (also `refe/`, `code/`).
   - `md5sum C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/a085b71c-4fb3-409b-ae6b-65f605ed575e/scratchpad/ev6e/families6.py` = `e275637c93e1f18ec80a347719680e02`.
     ⚠️ `stage2_sft.py` hard-codes that scratch directory (`EV6E`). If it is gone, `families()` returns None and the stage-2 verdict silently becomes **NOT PROVEN** (missing families fail closed).
     Fix: recreate the directory from the repo copy `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/code/families6.py` with its two `D:/Projects/TanitAD/taniteval` paths changed to `E:/...`
     (that is the whole 2-line difference; the adapters are byte-identical on D: and E:). `stage3_full.py` itself falls back to the repo copy when it must compute the arm's families (`REFE_FAMILIES6` overrides),
     and that copy is MEASURED to reproduce u400's banked families exactly (section 0).
   - No `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock` or `devbox_gpu.chain_active`; free RAM > 4 GB (the harness guard waits otherwise); nothing else on the RTX 4060.
3. **Smoke the REAL P2 file for 3 minutes before spending 5 hours on it** (the entry-point check below used the deployed `model_final.pt`; P2's own file does not exist yet, so its loading is UNVERIFIED until this step):
   ```bash
   $PY eval/stage2_sft.py --ckpt $CK --name p2_smoke4 --rule navsim_v1 --tokens $DATA/p2_dryrun4_tokens.json --exploratory
   ```
   Expect `ZZSTAGE2 p2_smoke4 EXPLORATORY ...`, seam `rows 4 misses 0`, `"rule": "navsim_v1"`, harness `PASS 4/4`. Its verdict decides nothing.

## 2. R1 (cheap, first): 923 tokens, 43 logs, against the deployed system (85.61 PDMS)

```bash
$PY eval/stage2_sft.py --ckpt $CK --name p2_R1 --rule navsim_v1 > $OUT/p2_R1.out 2>&1      # run detached; ~25 min
```
- Time (MEASURED, u400's 923-token read, `RESULT_SFT4.md`): seam 1,236 s, harness 109.4 s, families a few minutes. Holds the dev-box GPU lock for the seam.
- Output: `ZZSTAGE2 p2_R1 <ADOPT|REFUTED|NOT PROVEN> <mean> [lo, hi]`; `result.json` in `E:/.../raw/2026-10-05-sft-stage2/p2_R1/` (the rule is `PREREG_P2.md`).
- Do **not** pass `--exploratory` (that labels the read as deciding nothing).
- **R2 runs unless R1 is REFUTED** (then P6 is NO-GO without R2, `PREREG_P2_AMENDMENT_1.md` section 3).

## 3. R2: all 12,146 tokens, 136 logs, against SFT-4 arm B u400 (88.25 PDMS) = the P6 GO / NO-GO read

One detached command; it reuses the seam on its own second call. ~5.3 h on the dev box (MEASURED for u400: seam 17,496.8 s, harness 1,335.7 s, families ~2 min):

```bash
ARGS="--ckpt $CK --name p2_full --rule navsim_v1 --tokens $DATA/navtest_full_tokens.json"
nohup bash -c "$PY eval/stage2_sft.py $ARGS; test -f $DATA/seams/stage2_p2_full.npz && $PY eval/stage2_sft.py $ARGS && $PY eval/stage3_full.py --name p2_full --ckpt $CK --expect-ckpt-sha256 $SHA" > $OUT/p2_full.out 2>&1 &
```
What to expect, in order (all from u400's full-navtest read, MEASURED):
1. **Attempt 1 ends `ZZSTAGE2 p2_full INCOMPLETE (seam failed)` after ~17.5 ks. That is EXPECTED, not a failure.** The seam wrote all 12,146 rows (0 misses) and then exited 1 at its frame-control gate,
   because 36 declared C1 tokens over 4 logs (a nuPlan-DB vs OpenScene pose disagreement, no planner involved) read max 6.18063877792441 m against a 0.001 m bar (`RESULT_SFT4.md` Deviation 1).
   The `test -f` guard continues only if the seam `.npz` exists, so a seam that really died does not silently restart for another 5 hours.
2. **Attempt 2 reuses the seam** and scores the whole navtest filter without `--tokens` (a 12,146-token override is 230,812 characters against Windows' 32,767 limit: `WinError 206`, Deviation 2). It prints
   `ZZSTAGE2 p2_full <verdict> ...`. ⚠️ **That verdict is P2 against the DEPLOYED system (R2b). It is not the P6 verdict.**
3. `stage3_full.py` prints `ZZSTAGE3 p2_full <outcome> <mean> [lo, hi] adverse=<...> gates_ok=<...>` and writes `raw/p2_readout/p2_full/r2_result.json`. **Only this line decides P6.**
   - **GO**: lo > 0 AND no PDMS sub-score (NC, DAC, EP, TTC, C, DDC) adverse (its paired interval entirely below 0).
   - **NO-GO**: (adverse sub-score, named) / (NOT PROVEN) / (P2 below u400), each with its next lever, as `PREREG_P2_AMENDMENT_1.md` section 4.
   - **INVALID**: an instrument gate failed. Never a GO; read `gates` in the JSON. Gates include the pinned token file and baselines, the harness's own `PASS 12146/12146`, seam `rows = 12146, misses = 0`,
     `rule == navsim_v1`, seam settings equal to u400's (route fix, A7), frame control bit-identical to u400's (the declared C1 set and nothing more), same-pose tokens scoring identically, the checkpoint's sha256, and R2b equal to stage 2's own primary to the bit.
4. Also in `r2_result.json`, reported and **not gating**: the 21 longitudinal / lateral components against u400 and against deployed (adverse ones named), the tactical rows (no interval, so reported not tested),
   R2b (P2 vs deployed 84.46), and the minus-C1 sensitivity (12,110 tokens). STRATEGIC is PAUSED and UNAVAILABLE in NAVSIM: reported as such, never a pass.
5. Watch without model turns: one background Monitor on `F=$OUT/p2_full.out`:
   `until grep -q '^ZZSTAGE3' $F || grep -q Traceback $F || [ "$(grep -c INCOMPLETE $F)" -ge 2 ]; do sleep 60; done`.
   The FIRST `INCOMPLETE` is the expected frame-control exit (step 1); a second one, a `Traceback`, or `ZZSTAGE3` ends the wait. Match the failures as well as the success.
   ⛔ Do not hold a live agent for the 5 hours: write a <= 2k-token `HANDOFF.md` (this file, the command, `$F`, `$SHA`) and end; the completion launches a fresh one.

## 4. After the read (the deliverable)

1. Copy the E: raw dirs into the package: `raw/2026-10-05-sft-stage2/p2_R1/` and `p2_full/` (their `result.json`, `score_status.json`, `*families.json`, logs), and bank the arm's per-token table and seam under `raw/p2_readout/p2_full/arm/`
   (`$DATA/score/refe_stage2_p2_full/refe_stage2_p2_full.csv`, `$DATA/seams/stage2_p2_full.npz` and `.report.json`) with their md5: those two files exist only on the E: disk until copied.
2. Write the result with the rule's wording, the interval's estimator and the amendment section 6 confounds verbatim (labels 146,185 vs 174,912; one seed each; u400 is update 400 of 2,335; six sub-scores gate, so a NO-GO can be a false negative).
   **GO is a direction decision, not a lever claim; the spend is the PI's call.**
3. Update the register (`Project Steering/GOALS_AND_CLAIMS.md`, the `D-REFE-P2` / `D-REFE-SFT-4` rows) in the same turn. Stage with `git add <exact path>`; never `git add <dir>`.

## 5. Entry-point dry run (MEASURED 2026-10-12): the deployed `model_final.pt` through the exact stage-2 and stage-3 commands

Command: `stage2_sft.py --ckpt E:/Projects/TanitAD/data/refe_runs_eval/model_final.pt --name p2_dryrun4 --rule navsim_v1 --tokens $DATA/p2_dryrun4_tokens.json --exploratory`, then
`stage3_full.py --name p2_dryrun4 --dry-run --tokens $DATA/p2_dryrun4_tokens.json --ckpt <same> --out-dir raw/p2_r2_instrument/dryrun_p2_dryrun4`.
4 tokens, 2 logs, none in the C1 set. Artifacts: `raw/p2_r2_instrument/dryrun_p2_dryrun4/`.

| shown | result |
|---|---|
| the seam accepts a `model_final.pt`-class file with rule `navsim_v1` | rc 0, 4 rows, 0 misses, `rule navsim_v1`, `repair_last_heading` true, `goal_fix pdm_route`, `per_sample_calib=True`, 17.4 s |
| it is the deployed pick, not u400's | executed poses **bit-equal to the deployed seam** on 4 of 4 tokens (max abs 0.0), against 2.7-9.3 m from u400's `navsim_v1_lane` poses |
| the harness scores the subset | `PASS 4/4`, C1 max abs delta 0.0; stage 2 total about 3 min |
| stage 3 reads stage 2's output | all gates true; R2b equals stage 2's own primary to the bit (the cross-instrument control); the outcome is stamped `DRYRUN (not a read)` and can never be a GO |
| the checkpoint's identity is recorded | sha256 `4246a320be71ad0e315f8f298b7dadab3dbbba2c24f4fc30706e0129f5057e66`, 1,288,484,600 B |

**Not shown (UNVERIFIED until P2's file exists):** that P2's own `model_final.pt` loads. It is the same trainer's output class (P2 warm-starts from `model_final.pt`, same 18.99 M trainable set, six-output scorer, so the strict `ckpt_io` loader applies), but that is INFERRED; step 1.3 is the check.
Nor was the 12,146-token path run: the full-set branch of `stage2_sft.py` and its C1 frame-control exit are MEASURED only on u400's read.

## 6. Traps met while building this (each cost a run or nearly did)

- A `--tokens` file for `stage2_sft.py` must be W3-format: `{"tokens": [...], "token_log": {...}}`. Without `token_log` the harness dies with `KeyError: 'token_log'` and stage 2 then retries six times with 120 s sleeps. (MEASURED 2026-10-12, my first dry run.)
- A swap or self-pair test on data whose per-token differences are all equal cannot fail: every resample mean is identical, so no seed moves the interval. The first draft of `t_swap_with_wrong_seed_fails` had no teeth for that reason.
- "Exactly" on a bootstrap interval is a claim about `np.percentile`'s interpolation as well as about the data; see section 0.
- The harness CSV's physical last line is an `average` row (token `average`, skipped by `proxy_eval.read_csv`). The first draft of the defective-arm regression dropped *that* line, removed no token, correctly left every gate green, and so failed to fail: the selfcheck said `regressions_failed_as_required=False` and exposed the regression, not the gate.
- `core.autocrlf=true` on this checkout: an LF-only text file under `raw/p2_r2_instrument/baseline/` (the minus-C1 token file) can turn CRLF on a fresh checkout, which changes its md5 and makes the pin refuse that in-repo copy.
  The `E:/Projects/TanitAD/data/refe_navtest/` originals are tried first and are the primary; the repo copies are the durability fallback (convert to LF before relying on one).
