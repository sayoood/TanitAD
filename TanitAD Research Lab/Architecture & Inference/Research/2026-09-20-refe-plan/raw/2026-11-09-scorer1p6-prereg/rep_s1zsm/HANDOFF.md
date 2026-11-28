# HANDOFF: S1ZSM1 = seed-1 replicate of S1ZSM (2026-11-10). CPU only; nothing committed, nothing on a pod, no GPU/lock, no git write.

## State
Overlay on tip `cb413ab19a1f88521926bf2496068356f8673b6d`. `files/` = 7 files, ALL MOD (no new file), LF. `rep_s1zsm_vs_tip_cb413ab1.patch` (`git apply`) reproduces `files/` byte for byte (MEASURED, cmp on all 7). `CHANGED.txt` has the tip blobs: launch_gate cc973a306d, rep_pod.sh 6acfd8130f, rep_chain.sh 4c01cec270, rep_pause_p7.sh b15d110035, test_rep_gate 50c837c33b, test_scorer1p6_gate f0e3830082, test_s1zsm_gate 50c6b49896. Land only if they still match.
- `launch_gate.py` (additive, 11 lines): `REP_ARMS["S1ZSM1"]` (parent S1ZSM, family s1zsm, seed 1), `REP_PARENT_SEEDS["s1zsm"]`, `rep_argv` takes `s1zsm_argv` for that family, NEW profile `refe_rep_s1zsm_p6` (deep copy of refe_s1zsm_p6 + replicate root/table/suite S1ZSM_SUITE + test_rep_gate). New profile needed: the twin and speed-neutral rules (`sn_spec`, required `--twin-speed-neutral`) are profile-wide; on refe_rep_s1z_p6 the lever would be refused.
- `rep_pod.sh`: ARM S1ZSM1 -> runs `s1zsm_pod.sh` MODE=argv (adds `S1ZSM_WS=$WS`), swaps `--seed` and `--out` only; no lambda -> rc 3; a non-speed-neutral parent pair_weight.json -> parent rc 4 -> rc 2 here; mutants noband, nosnbalance, droploss, noradius, notopk (REFE_TWIN_MUTATE_*).
- `rep_chain.sh S1ZSM1`: parent card PREREG_S1ZSM_P6 sha256 ac75b289... in `/workspace/data/refe_s1zsm/`; lambda_z READ from `refe_s1zsm/calibrate_s1zsm/pair_weight.json`, and the inputs stage refuses it unless speed_neutral true / sn_dv 0.15 / sn_balance true / sn_seed 20261107 / lever twin; waits also for the `refe_s1zsm` lock; NEED list adds s1zsm files; 5 mutants each must fail preflight AND read G18 ok false; run check adds kept_F == kept_S > 0, out_of_band 0, pairs == 2 kept_F and data event speed_neutral true. Dirs `/workspace/data/refe_rep/s1zsm1/`, HF `rep/s1zsm1` in `Sayood/tanitad-refe-scorer1p6` (existence UNVERIFIED), terminal `ZZREP_CHAIN_DONE S1ZSM1`.
- `rep_pause_p7.sh`: accepts arm S1ZSM1.
- Existing tests edited (one line each): test_scorer1p6_gate:154 (profile set), test_s1zsm_gate:141 (new-name set), test_rep_gate (extended).
- Digests (MEASURED, test R3): refe_s1zsm_p6 86380fbb786daa75, refe_s1z_p6 5c3bff457aa3bf1e, refe_scorer1p6 e74d64617f514d35, refe_rep_p6 dd348f44c9638f1a, refe_rep_s1z_p6 2356afd4efeeb52a all unmoved; the 21 TIP_FP of test_s1zsm_gate G3 unmoved; NEW refe_rep_s1zsm_p6 27fe236a2fd68ca4. Profiles 22 -> 23.

## Decisions (check them)
1. lambda_z = the parent's measured speed-neutral value, read not re-measured (a re-measured weight is a second moved lever). So S1ZSM must have calibrated first. 2. Seed 1 (parent 0). The speed-neutral generator seed 20261107 is not a flag and is untouched; R1 proves only `--seed`/`--out` differ (exactly two tokens, by position). 3. Chain `others_hold` now has 2 added locks (refe_s1z, refe_s1zsm) over S1Z's list; s1zsm_chain already waits for refe_rep. 4. Same `dd` 6144 MB, same HF repo as the parents.

## Trace (seed + lever reach training), files in `files/`, refe/
`rep_chain.sh` arm_argv (`rep_pod.sh ARM=S1ZSM1 MODE=argv`) -> `rep_pod.sh` PARGS from `s1zsm_pod.sh` MODE=argv (`--twin-speed-neutral` in its TWIN string, `--seed 0`), sed swaps `--seed 1` / `--out .../refe_rep/s1zsm1` -> `launch_gate.py run --profile refe_rep_s1zsm_p6` (`rep_argv` pin via `s1zsm_argv`; `ns.seed != arm_seed` and the parent-seed refusal in `scorer1_refusals`) -> `rep_pod.sh MODE=run` -> `scorer_finetune.py $ARGS` -> `ap.add_argument("--seed")` -> `torch.manual_seed(a.seed)` and the data-order `Generator().manual_seed(a.seed)`; `--twin-speed-neutral` -> `TwinLoss(speed_neutral=...)` with its own generator seed 20261107 -> `lB = lB + twin_term` -> `.backward()` (s1zsm HANDOFF trace, line numbers there: 1587, 1674, 1916-1938, 2806; seed lines per the rep HANDOFF: 1453/1555/1764/1715 at tip 8ef2cdda, re-grep after landing).
Mutations (test_rep_gate, run on copies, each RED): delete the S1ZSM1 REP_ARMS entry (KeyError), S1ZSM1 seed 0 in launch_gate (ValueError "equals its parent"), SEED=0 in rep_pod.sh (R5 assert).

## Tests (MEASURED, CPU, CUDA hidden, one at a time, tanitad from the export; test_results.json)
rc 0 ZZTEST_PASS: test_rep_gate (R1-R5, 17 s), test_scorer1p6_gate, test_s1z_gate, test_s1zsm_gate, test_s1zsm_preflight, test_s1z_preflight, test_scorer1p6_preflight, test_scorer1_layer, test_s1z_layer, test_s1zsm_layer, test_pair_loss, test_pair_loss_p6, test_dump_scores (needs eval/ exported), test_s1zsm_loss, test_s1z_loss, test_suite_nesting, test_scorer1_gate, test_scorer1_mutation, test_rl_epo. test_launch_gate: NOT PASSED, stops at the missing `D:\Projects\TanitAD\data\refe_consistent\targets_rank1.jsonl` (known); after it UNVERIFIED.

## read_rep.py
Hard-codes names: `RECIPE_OF` (line 46) and `CARD_SIZE` (47) hold only C0 C0r C0r2 S1E S1E1 S1Z S1Z1; `--run` refuses other names (line 231-233). A S1ZSM vs S1ZSM1 read needs: a recipe `S1ZSM` = {S1ZSM, S1ZSM1} in RECIPE_OF/CARD_SIZE, the slot logic at lines 238/288 (S1Z runs go through `S.s1z_read` in the S1Z slot; S1ZSM runs need read_s1zsm's wrapper), and a card (PREREG_REP_P6 is frozen and has no S1ZSM recipe, not changed). Without that: read each arm with `read_s1zsm.py --s1zsm DIR` (SM1-SM3 outcome map) and take the seed pair by hand; read_s1zsm's `--rep-read` expects a read_rep.py rep_read.json (s_p, df, t).

## Pod commands (NOT run)
1. Land `files/`, regenerate SHIP_MANIFEST (`launch_gate.py ship-manifest --commit <40-hex>`), ship to `/workspace/refe-plan-rep` (never over a dir holding a token; refe-plan-s1zsm is such a dir). Run suite-prerun: `python refe/launch_gate.py suite-prerun --profile refe_rep_s1zsm_p6 --pkg /workspace/refe-plan-rep --out <json>` (on a tree copy first).
2. Dry run: `REP_STOP_AFTER=inputs REP_HF_TOKEN_FILE=<tok> bash /workspace/refe-plan-rep/refe/rep_chain.sh S1ZSM1`.
3. Run: `nohup setsid bash /workspace/refe-plan-rep/refe/rep_pause_p7.sh <p7_pod.sh pid> <train.py pid> S1ZSM1 > /workspace/data/refe_rep/pause.out 2>&1 < /dev/null &` (or rep_chain.sh directly when P7 is paused).
ESTIMATED L40 time: train 400 x 8.1 s = about 54 min (S1ZSM MEASURED), preflight + 5 mutants about 54 min (S1ZSM MEASURED), gate/egress about 10 min: about 2.0 L40-h. No calibrate stage.

## Not verified
Pod runs of any stage, the chain end to end (bash -n + text tests only), rep_pause_p7.sh (bash -n + text), the 69-test pod suite, HF prefix, parent's real pair_weight.json, test_launch_gate after full().

## Cleanup
Scratch `F:/tanitad_scratch/rep_s1zsm_work/` (tree export, mut copies, patchrepo, tmp) is deleted at the end; deletion log in the hand-back.
- deleted F:/tanitad_scratch/rep_s1zsm_work (135 MB, own scratch: export, mutation copies, patch repo)
