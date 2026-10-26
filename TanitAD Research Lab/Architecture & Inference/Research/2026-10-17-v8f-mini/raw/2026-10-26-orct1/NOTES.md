# orct1: the dev-box CPU chain of PREREG_V8F_ORECOVER (BUILT AND SMOKED, NOT LAUNCHED), 2026-10-26

Nothing was launched against real O-RECOVER data, box 2 and `C:/Users/Admin/box2_ship/` were not touched, nothing is staged, committed or pushed. Every number below carries its class:
MEASURED (file path) or ESTIMATED. No smoke number is a result.

## 1. What it is

`orc_t1_chain.sh` (+ `orc_t1_chain.cmd`, CRLF, for a detached WMI start) is ONE chain, zero training, zero GPU. For each run in the PREREG's order
(ORC s0, ORC-0 s0, ORC s1, ORC-0 s1; `ORCT_RUNS` is honoured in SMOKE only):

1. **waits for the pull to be COMPLETE** (`orct1_pullcheck.py`, section 3);
2. runs **two T1-kin rollouts** on eval-141, plain (no post-map), CPU: `--limit causal` (decisive) and `--limit logged` (K4 only, Amendment 2), both
   `v8f_t1_eval.py run --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --dump-ticks --dump-heads --device cpu`;
3. two lanes (A = every causal rollout, B = every logged rollout) so there are **at most 2 rollout processes**; `CUDA_VISIBLE_DEVICES=-1`, `OMP/MKL_NUM_THREADS=8`;
4. after all four runs: `orct1_manifest.py` writes `runs.json`, then `orc_read.py` runs with every required argument.
   Markers `ZZORCT1-DONE <n>/8 rollouts, reader rc <rc> (...)` or `ZZORCT1-STOP <why>` in `logs/chain.out`.
   A REFUSED read (rc 2) is a finished chain with a verdict, so it ends in DONE with the rc, not in STOP.

Resume = re-run the same command. Finished rollouts are skipped only if their `done_<tag>.txt` names THIS checkpoint's sha256 and THIS tool digest; otherwise the old
output is MOVED to `<OUT>/_incomplete_*` (never deleted) and redone.

## 2. Parity with the T1-HEADS rollouts (what "the SAME" was checked against)

| item | T1-HEADS (`C:/Users/Admin/t1hd_chain.sh`, `t1hd_run/logs/`) | this chain |
|---|---|---|
| tool tree | `E:/Projects/TanitAD-artifacts/v8f-t1heads-run/tool_81aa3e36` (tool sha256 ffdd1469...) | the same tree (`ORCT_TOOL_TREE`); tool file sha256 ffdd1469cc0bca39... in `PINS.txt` |
| flags | `--replan-steps 10 --device cpu --limit L --dump-ticks --dump-heads`, nav / scene / schedule at their defaults | the same plus `--nav deadreckon --scene logged --schedule periodic` named explicitly (= the defaults; the reader's T1-CONFIG control checks them in the json header) |
| inputs | scratchpad `sfresh` cache (141 clips), `native_tracks.pt` md5 189f4e68..., grid md5 34ed8ac5..., join md5 532fcd64... | the same files, md5-asserted |
| threads, batch | OMP/MKL 8, default `--batch 32` | the same |
| native crash | causal s0 segfaulted once when two seeds ran in parallel; one retry on rc 139 | one retry on rc 139, crashed output never read |

**MEASURED equivalence (full size, 2,399 windows):** the chain's decisive command, run on the T1-HEADS causal-s0 checkpoint (`M2b_s0_hacc.pt`, sha256 71de3c61...),
gives a tick file whose **32 of 32 arrays are bit-identical** (NaNs in the unscored GT counted as equal) to `t1hd_run/ticks/ticks_HACC_s0_m10_causal_heads.npz`, and a
result json equal apart from tag / ckpt / ticks_file. `smoke/equiv_compare.out`, produced by `smoke/equiv.sh` (output `E:/Projects/TanitAD-artifacts/orct1/smoke_equiv`).
Whole-process wall of that one rollout: **181 s** (`smoke/equiv/equiv.out`). T1-HEADS' rollout lines read 244 / 335 / 360 / 298 s (`t1hd_run/logs/*.out`).

## 3. "Never start on a half-pulled file": the completion signal

Source: `chain_v8f_orc.sh` (`runone`). Box 2's chain pulls into `<ART>/M2b_s<S>_<orc|orc0>/` (`ART` = `E:/Projects/TanitAD-artifacts/v8f-orecover`):
`selfcheck.json summary.json argv.json train_log.jsonl train.log ckpt_last.pt`, `eval_M2b-<arm>_s<S>.{json,npz,log}`, and `launch_gate_token.json` (written there before training).
Only AFTER the last `scp` returns it runs `sha256sum` over EIGHT files into `SHA256SUMS.txt`: `ckpt_last.pt, eval_<tag>.json, eval_<tag>.npz, selfcheck.json, summary.json, argv.json, train_log.jsonl, launch_gate_token.json`.

A run is COMPLETE iff, by content: SHA256SUMS.txt exists, is older than 3 s, ends with a newline and lists exactly those eight names once; each file exists, is non-empty and
re-hashes to its listed sha256; the four small JSON records parse; `summary.json` says `done: true`. Presence of the directory, of the ckpt, or of an old sums file is never the signal.
It does not read box2_ship or ask box 2 anything. Optional watchers: `ORCT_B2_OUT` (a `ZZV8FMINI-STOP` that appears after this chain started stops it) and a per-pull timeout (`ORCT_MAX_WAIT_H`, 40 h).

MEASURED (`smoke/selftest_output.txt`, 39/39): the positive control reads COMPLETE; 16 mutants of the pull state (no sums, young sums, 7 or 9 lines, no final newline, line cut in its name
or its hash, half-pulled ckpt, one flipped bit, absent file, empty file, summary `done: false`, unparseable json, wrong arm / seed) all read INCOMPLETE. Mutation of the checker itself
(`smoke/mut_120808/`): deleting the sha compare, the age test, the name-set test, the `done` test, the final-newline test, the non-empty test, or the exists test each turns at least one test RED.
LIVE (`smoke/halfpull/test.out`): the real chain, SMOKE, was started on a run with no SHA256SUMS.txt and a 40 MB of 87 MB ckpt: it logged `waiting for the pull`, started no rollout; with the sums
put back over the still-half ckpt, `orct1_pullcheck.py` read INCOMPLETE (sha mismatch); after the ckpt was completed it logged `pull ... COMPLETE` and started both rollouts.

## 4. Guards on the tool (pattern of `hmc_chain.sh`)

`grep -cE 'add_argument\("--dump-heads"'`, `--dump-ticks`, and `^LIMIT_MODES = (... "causal"` on the tool, each with a same-breath control: positive (`--replan-steps`, `"logged"` must be >= 1) and negative
(`--zzz-no-such-flag-zzz`, `"zzz-no-such-limit"` must be 0); then the tool's own `--help` must list `--dump-heads` and a `{logged,...,causal,...}` choice. MEASURED (`smoke/guard_stub_results.txt`):
six stub trees (the real tool with ONE fact removed: no `--dump-heads`; no `--dump-ticks`; no causal; `--dump-heads` kept only in the docstring; a broken positive control for each grep) are each REFUSED with the
right reason (`ZZORCT1-STOP`), and the unmodified tree passes. Pins (`<OUT>/PINS.txt`): sha256 digest of every `.py` the tool can import, 7 key files, the 3 reader scripts and the PREREG; re-asserted before and after every
rollout and before and after the reader. MEASURED: a tampered tool-tree digest, and a tampered PREREG pin, each STOP the chain at start (`smoke/halfpull/logs/chain.out`, 12:21:06 and 12:21:12).

## 5. The manifest and the reader call

`orct1_manifest.py` writes the schema `orc_read.load_manifest` / `read()` consume (`test_orc_read.Fx` is the fixture): `{"seeds":[0,1], "arms":{"ORC":{"0":E,"1":E}, "ORC0":{...}}}`, `E = {ckpt_tail, train_dir, token,
t1:{tag,ticks,dump_dir,json}, t1_logged:{...}, t0:{json,npz}}` (full example: `smoke/dry_run_real_config.txt`). No `M2b` arm, on purpose: no plain-M2b T1-kin rollout with heads exists (the T1-HEADS runs are M2b + HACC, another
checkpoint and ckpt_tail) and Amendment 2 reads ORC - M2b at T0 only; the reader treats the arm as optional.

Reader command (printed by `--dry-run`): `python -I orc_read.py --manifest <OUT>/runs.json --out <OUT>/read/ORC_READ.json --tool-root <tool tree> --t1hd-read .. --t1k-read .. --eval-cache --tracks --eval-grid --eval-join
--m2b-sha 0=.. --m2b-sha 1=.. --prereg <PREREG> --key-built-p ego_perturb.ego_perturb_p` (B stays 2,000; `--key-perturb-n/eligible`, `--key-init-sha` keep their defaults, which match the trainer's keys; MEASURED on the real trainer's
synthetic-data summary: `ego_perturb_n 7`, `ego_perturb_eligible 23`, `levers.ego_perturb.ego_perturb_p 0.3`, `levers.init_ckpt.sha256`).
`--m2b-sha` comes from the runs' own `levers.init_ckpt.sha256` (all four records must agree per seed; `ORCT_M2B_SHA_JSON` adds the box-2 chain's measured values as a cross-check); the reader compares them with the PREREG's
601cb435... / 6104fb15... (M2B-CKPT). The M2b checkpoint file is not on this box. Consequence: INIT-CKPT compares the records with themselves; M2B-CKPT is the real test.
The reader tree defaults to `E:/Projects/TanitAD-artifacts/oc/tree` (the overlay: `orc_read.py` 35eaad3b9663 with Amendment 3; the git mirror's tip still holds an older `orc_read.py` 8bdd3e206224 and no Amendment 3 PREREG).

## 6. OPEN DEFECT 1 (MEASURED): K4 cannot pass against box 2's standard T0 eval

K4 (PREREG section 3, Amendment 2) = tick 0 of the `--limit logged` rollout equals the arm's own T0 eval to 1e-3. Tick 0 runs at ages (0, 0). Box 2's `evalorc` is `v8f_mini_eval.py run ... --project --families`
with the default **`--ages cycle`** (the queue line, `orcchain_build/scratch/chain_dry_a3.out`). The two differ by construction: M2b s0 S pos@30 **7.0231 m** (cycle, `raw/full/eval/eval_M2b_s0.json`) vs **0.9774 m** (fresh,
`sfresh-fresh` json = the T1-HEADS K4 comparator). The reader stages the manifest's `t0` json/npz as the K4 comparator, and the same `t0` feeds R-T0 and T0-CKPT, so the manifest cannot carry both.
MEASURED with the reader's own `RealBackend.k4` at full size (`smoke/k4_demo.out`, one checkpoint, T1-HEADS' banked logged rollout): vs a **fresh** T0 eval K4 **PASSES**, max |d pos| 4.4e-5 m, lat / lon F1 diff 0.0 / 0.0, 2,399 / 2,399 argmax equal;
vs a **cycle** T0 eval K4 **FAILS**, d pos 6.05 / 5.72 / 5.32 m, d lat F1 0.0195, d lon F1 0.0070. So with box 2's eval as written every one of the four runs reads REFUSED on K4.
Cheapest repair already built, default OFF (the PREREG as written stays the default): `ORCT_T0_FRESH=1` adds `v8f_mini_eval.py run --ages fresh` (the SFRESH argv) per run on CPU, inside the logged lane (so still <= 2 processes),
MEASURED 21 s per run; `ORCT_T0=fresh` additionally points the manifest's `t0` at it (the standard one is kept as `t0_standard`, the fresh one as `t0_fresh` either way). Then K4 can pass, but R-T0 is read on the fresh-age
own-code ADE6 for both arms. That is a PREREG decision (a K4 comparator / R-T0 protocol amendment, written before data), not mine. Alternative: add `--ages fresh` to box 2's `evalorc` (a second eval, box 2 is the parent's).
Tool: `orc_read.py` is not edited (its note: do not edit after reading an outcome).

## 7. Other open items

* **K5 / K6 fail on 24-window data** (reader smoke), pass on full-size T1-HEADS data per `orcread/out/smoke_real.out`. Not verifiable on ORC data before it exists. ESTIMATED: a size artefact.
* **Inputs live in the session scratchpad** (`.../scratchpad/sfresh`: cache, tracks, v9, p10), as for `hmc_chain.sh` and `t1hd_chain.sh`. If that directory is cleaned the chain STOPs loudly at its md5 / clip-count check
  (`ORCT_INPUTS` names a copy).
* The reader's `PREREG-TEXT` control passes on the oc-overlay PREREG (verbatim cells present). It would read the mirror's older PREREG differently.
* `launch.out` of the `.cmd` is overwritten by each launch; the chain's own log is `logs/chain.out`, appended across resumes (use the LAST marker).
* I created two `__pycache__` dirs in the shared T1-HEADS tool tree with an import probe (11:58); they are moved to `smoke/moved_pycache/`; the tree digest (`.py` only) never changed (c8d2990e3d04).
* 12:20: `E:/Projects/TanitAD-artifacts/v8f-orecover/M2b_s0_orc/` appeared (empty), i.e. the box-2 chain's gate stage had started; the dry-run reads it as INCOMPLETE (no SHA256SUMS.txt).

## 8. Evidence index (all under `C:/Users/Admin/orct1/smoke/`)

| what | file |
|---|---|
| `bash -n` | clean (final) |
| dry-run, real configuration (touches nothing: `logs/` empty, `v8f-orecover-t1` not created) | `dry_run_real_config.txt`, with fresh T0: `dry_run_real_config_fresh_T0.txt`, via the `.cmd` with inline env: `launcher_dry_run_with_env.out` |
| SMOKE chain, 24 windows, run 1 | `chain_smoke_run1.stdout`, `logs/chain.out`, `out/` (`runs.json`, ticks, dumps, `done_*`), reader refusal `logs/orc_read.out` |
| SMOKE with fresh-T0 stage, incl. two resumes (rollouts skipped by their done record) | `chain_smoke_run2_t0f.stdout` (the STOP that found the missing `--eval-kin`), `run3` (found a manifest prefix bug), `run4` (clean), `out_t0f/`, `logs_t0f/` |
| half-pulled live test, changed-checkpoint redo, pin tamper | `halfpull_test.stdout`, `halfpull/test.out`, `halfpull/resume_changed_ckpt.stdout` |
| self-test 39/39 | `selftest_output.txt` |
| guard stubs | `guard_stub_results.txt`, `stubs/` |
| equivalence with T1-HEADS | `equiv_compare.out` |
| K4 fresh vs cycle | `k4_demo.out` |

SMOKE wall: **13 s** per 24-window rollout with two running at once (`logs/chain.out` run 1: `wall=13s` x2), 15 s and 12 s in two later runs; fresh-T0 eval **21 s**; the reader's refusal on the 4-slot SMOKE manifest took **33 s**
(`logs/chain.out`, 12:10:00 to 12:10:33).

## 9. Where the reader refuses on SMOKE data, and why (`smoke/logs/orc_read.out`, rc 2, nothing written)

The SMOKE manifest replicates ONE stand-in run into all four slots: the banked M2b_s0 HACC checkpoint as `ckpt_last.pt`, the banked T0 eval of M2b_s0 as the run's T0 eval, the real trainer's summary / selfcheck / argv from a 6-step SYNTHETIC-data fine-tune,
and a placeholder token. The reader reached `RealBackend.analyze_t1` / `analyze_t1_light` for all eight rollouts (K1, K2, K3, K7 x1, T1-CONFIG, DUMP-WINDOWS, SAME-WINDOWS, R2-ADE6, PREREG-TEXT, INIT-CKPT, PERTURB-REACHED for ORC all PASS, i.e. they are absent from the
refusal list), then refused on: `M2B-CKPT` (stand-in sha 33fcb079... is not the banked 601cb435... / 6104fb15...), `ARGV` / `SUMMARY` (6 steps, not 3,000; seed 0 in the seed-1 slots; ORC-0 slots carry ORC's argv), `GATE-TOKEN` (placeholder), `T0-CKPT` (the banked eval names another
checkpoint path), `PERTURB-REACHED` on ORC-0 (n = 7, not 0), and, from the 24-window size, `GRID`, `GRID-COUNTS`, `K4`, `K5`, `K6`. No outcome row is computed under a failed control.

## 10. ESTIMATED wall time of the real chain

* Dev-box compute: 8 full rollouts, 181 s (this chain's command, one process, MEASURED) to 360 s (T1-HEADS) each: 24 to 48 min of process time, 12 to 24 min per lane. ESTIMATED 4 to 8 min each with two lanes.
* The chain's elapsed time is set by box 2 (4 gates first, then ~0.7 h per run by the box-2 chain's own estimate, plus eval and pull): ESTIMATED 3.5 to 4.5 h after box 2 starts.
* Tail after the LAST pull completes: last run's two rollouts (ESTIMATED 4 to 8 min, +21 s for the optional fresh eval) + manifest + reader (ESTIMATED 4 to 8 min; the reader's NOTES say about 1 min per run at B 2000) = ESTIMATED 10 to 16 min.

## 11. Amendment 4 (`PREREG_V8F_ORECOVER` A4-2026-10-26, landed 853f2fc3 before any run): the K4 decision is taken

The PREREG decision §HANDOFF asked for was written before data: K4's comparator is the arm's own T0 eval at `--ages fresh`, and R-T0,
T0-CKPT and the four families stay on box 2's standard (`--ages cycle`) eval. They come from SEPARATE manifest keys. The chain follows it:

* `orc_t1_chain.sh`: the fresh-age T0 eval always runs (`T0F=1`). `ORCT_T0_FRESH=0` is refused, and `ORCT_T0` no longer exists (setting it
  is refused): the old `ORCT_T0=fresh` switch repointed `t0` at the fresh eval, which would have moved R-T0 too, so A4 removes it.
* `orct1_manifest.py`: `t0` is always box 2's eval and `t0_fresh` the CPU one. `--t0-use` is gone, and `verify()` refuses a manifest
  without `t0_fresh`.
* `orct1_selftest.py`: **40/40** (`smoke/selftest_output_a4.txt`; it was 39/39 in `smoke/selftest_output.txt`).
  * MF0's fixture carries the fresh evals.
  * MF1's schema includes `t0_fresh`.
  * MF8: `t0` stays standard.
  * MF9: `--t0-use` is rejected.
  * MF12 (new): a manifest without `t0_fresh` is refused.
* The reader (`raw/2026-10-26-orcread/`, A4) reads K4 from `t0_fresh`. It adds `T0-AGES` (`t0` must be `cycle`, `t0_fresh` must be
  `fresh`) and `T0F-CKPT`, so neither eval can stand in for the other.
* **Reader tree.** The chain's default reader tree moved from `oc/tree` (the A3 reader) to `rt4/tree`, a fresh export of the tip that
  carries the A4 reader. A launch therefore reads the reader it pins, and an old tree cannot be used by accident.
* The SMOKE runs in §8 predate A4. They ran the old defaults, and no ORC data existed then.
