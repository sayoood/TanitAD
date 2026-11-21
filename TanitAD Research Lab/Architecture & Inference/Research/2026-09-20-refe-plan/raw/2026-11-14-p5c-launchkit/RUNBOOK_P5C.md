# RUNBOOK_P5C.md: from "P7's final checkpoint is banked" to "C1 and C0 training have started on the pod"

Tags: **[MM]** the Master Mind, **[PI]** the PI decides, **[POD]** runs on the pod, **[DEV]** dev box. Evidence class on every number. Nothing here has been run on a pod. Card = `scorer_gym_20261107/p5c/PREREG_P5C.md` + `PREREG_P5C_A1.md`.

## 0. Preconditions (check, do not assume)
1. **[MM]** P7's LAST arm has trained (the land-closure rule: a land touching `train.py` between an OVF record bind and its chain restart voids the record). The launchkit edits `train.py`, `gate_checks.py`, `launch_gate.py`. Land AFTER that.
2. **[MM]** Re-read the tip: `git ls-remote origin refs/heads/agent/arch-inf-20260803`. The landable overlay is valid only while the 7 MOD tip blobs read as in `REBASE.md` (table "tip blob"); otherwise re-run the 3-way merge for the drifted file.
3. **[POD]** P7 finished (nothing else on the pod: labelling needs every CPU, 25.1 CPU-h, and P7 trains on 13 label workers of the pod's ~15.3 CPUs [INHERITED, `p7_pod.sh` header]).

## 1. Seal the base **[POD] then [MM]**
`sha256sum /workspace/data/refe_runs/p7_reserve_head/model_final.pt` (path = `P5C_DEFAULT_PATHS["init_from"]`, a TEMPLATE; use P7's real `$RUN`). Record it: it becomes `P5C_BASE_SHA256` (step 8). **[MM]** reads `SPEC_P7.md` section 5's verdict for the base (card 2.1): ADOPT / NOT PROVEN -> B-P7; REFUTED / INVALID -> B-P6 (then register `p5c_profile(arm, "p6")` in `PROFILES`; it is built and tested but not registered). If P7's verdict is not yet banked, card 2.1 says launch C1 on B-P7 and stop at the next checkpoint if it turns REFUTED.

## 2. Land the kit **[MM]**
1. Land `p5c_launchkit/files/<repo path>` (9 files: `train.py`, `gate_checks.py`, `launch_gate.py`, `test_ego_yaw.py`, `test_launch_gate.py`, `test_p7_head_wiring.py`, NEW `traj_label_table.py`, `test_traj_label_table.py`, `test_p5c_profile.py`) with the coordinated lander. Land a COPY of pinned/run-tree files (the lander rewrites CRLF sources in place); every file is LF.
2. Regenerate `SHIP_MANIFEST.json` and the ship list: `python refe/p7_ship_list.py derive --pkg <FULL pkg> --out <list.json> --profile refe_p5c` (`test_p5c_profile.py` names `p7_pod.sh`: it ships).
3. Run the pin-sensitive tests on the landed tree (list in `TESTS.md`): a launch_gate pin or profile edit moves profile fingerprints pinned in other tests.

## 3. Ship the labelling kit **[MM] -> [POD]** (`a1/POD_SHIP_LIST.txt`, ~115 MB)
Ship everything in that list EXCEPT two files that the launchkit replaces:
- `traj_label_table.py` = `p5c_launchkit/files/.../refe/traj_label_table.py` (adds `--rank-policy`, `--meta-json`);
- `p5c_pod_label.sh` = `p5c_launchkit/runner/p5c_pod_label.sh` (two changed lines in `build_table`, one changed pin; `diffs/p5c_pod_label.sh.diff`). **[MM/PI]: the frozen A1 card pins the runner's blob `f9220d1f` and the ship list pins `traj_label_table.py` at `9592b62c...`; both move. They need a disclosed pre-data amendment (A2) before any pod label exists.**

## 4. Label the bank **[POD]**
```
P5CL_NSHARD=<N> bash /workspace/data/p5c_ship/p5c_pod_label.sh        # resumable with the SAME N
```
- **N = the pod's real core count, <= 15** on the P7 L40 pod (15.3 CPUs [INHERITED]); the runner's default 16 is for 16 shards and A1's "0.49 h wall" assumes 64 cores. Wall at N = 15: 25.1 CPU-h / (15 x 0.8) = **~2.1 h** [ESTIMATED; 25.1 CPU-h is the dev-box MEASURED 0.60 s/row x 150,603, `PREREG_P5C_A1.md` Point 6; eta 0.8 is an assumption]. Plus the repair sample (~7 min, [ESTIMATED]).
- Outputs: `ZZP5CL_LABEL tc20s|p5r|p5t`, `ZZP5CL_TABLE <label> rank_policy=<all|rank0> sha256=<hex> table=<path>`, `ZZP5CL_DONE`, files `$WORK/P5C_LABEL_TABLE_SHA256_{tc20s,p5r}.txt`, per label `gates/<label>/{gates.json,table_meta_fields.json}`.
- The runner now builds each table with `--rank-policy <the gate set's decision>` and `--meta-json gates/<label>/table_meta_fields.json` (sha-bound into the table), and asserts `rank_policy` in `verify`.
- **[PI/MM] Which label?** PI 2026-11-12: run on **P5-R labels regardless of the R4 trigger**. A1 Point 0 (frozen) says P5-R applies ONLY if R4 fires; the runner selects by the R4 order (`SEL`). If Tc20s passes TB-6 the runner prints `ZZP5CL_LABEL tc20s`, but it still builds the P5-R table when its gate set passes: use `P5C_LABEL_TABLE_SHA256_p5r.txt`. That choice contradicts the frozen A1 text, so it needs the A2 amendment (same one as step 3). `ZZP5CL_LABEL p5t` = STOP (exit 3; next lever P5-T).
- **[MM]** reads `gates.json`: ZZTB_ALL_PASS (or rank1only), TB-4 coverage 150,603/150,603.

## 5. Pull back and check **[MM], [DEV]**
Pull `table`, `table_verify_<label>.json`, `gates/<label>/` (small, ~100-200 MB for the table; check C: >= 12 GB first). `traj_label_table.py verify --table T --bank-dir <P7 bank>`: rc 0, `rows == covered == 150603`, `rank_policy` as in the gates. Rank policy `rank0` still has coverage 1.0 (rank-1 rows are identity rows; see `HANDOFF.md`, open item). Seal sha256 of the table.

## 6. Fill the three pins **[MM]** (in `launch_gate.py`, P5-C block)
`P5C_BASE_SHA256` (step 1), `P5C_LABEL_TABLE_SHA256` (step 4/5), `P5C_RANK_POLICY` (`all`|`rank0`). Until all three are filled `refe_p5c` refuses; the controls refuse until `P5C_BASE_SHA256` is. `test_p5c_profile.py` stays green with any value or none. Re-run the pin-sensitive tests, land, ship `refe/` again, re-derive `SHIP_MANIFEST.json`.

## 7. Gate inputs **[POD]**
The profile's checks are `G-HYG G-DVB G-LIVE G-CLOCK G-EVAL G-CKPT G-SUITE G-LIVE-LAB G-LIVE-MEM G-EGR` (no G-RESERVE). `launch_gate.py run` needs the pod-run evidence files exactly as `p7_pod.sh` (stage 8-11) produces them: `--gate-bank`, `--gate-lab-json`, `--gate-mem-metrics`, `--gate-egr-json`. **[MM] decides** whether P7's evidence files can be re-used (they are about P7's model with the head; the memory file is NOT valid for a head-less model) or must be re-measured: there is no `p5c_pod.sh` chain; the commands below are the manual form of `p7_pod.sh` section 11.

## 8. Mint the tokens and launch **[POD], [MM] go**
ARGS are `launch_gate.p5c_argv(arm, paths={...})` with the REAL paths of P7's `config.json` (diff the P7 flags against it first: `test_p5c_profile.py argv_delta` pins every non-path lever to `p7_pod.sh:346`). Per arm:
```
env -u HF_TOKEN OMP_NUM_THREADS=4 python launch_gate.py run --profile refe_p5c --pkg $PKG --args "$ARGS_C1" --out-dir $LG_C1 --gate-bank $GB --gate-bank-n 64 --gate-lab-json $LAB --gate-mem-metrics $MEM --gate-egr-json $EGR
env -u HF_TOKEN OMP_NUM_THREADS=4 python launch_gate.py run --profile refe_p5c_c0 --pkg $PKG --args "$ARGS_C0" --out-dir $LG_C0 ...   (no --traj-label-table; same other flags)
python launch_gate.py verify --profile <profile> --token $LG/PASS_*.json --args "$ARGS" --pkg $PKG --json <out>
python train.py $ARGS --preflight        # must print PREFLIGHT_OK
```
- `--out` must hold no `ckpt_last.pt` at MINT time (refused otherwise). Order: **C1 first, then C0** (card 2); the supervisor refuses to start the trainer without a matching token.
- Seed: C1 = C0 = 0. **C0r** (`refe_p5c_c0r`, `--seed 1`) is the card's CONDITIONAL replicate: mint it **only after the PI approves** (+28-29 L40-h; trigger = primary reads SEPARATED-SINGLE, card 6.3).
- Both arms' `model_final.pt` hashed and sealed in ONE manifest before either is read (card 2.1 / 11).

## 9. Cost **[ESTIMATED]** (card 14; rate INHERITED from `LS` l.1048: P7 L40 51.2 s/step)
C1 2,000 steps x 51.2 s = 28.4 h (card: 28-29); C0 28-29; **pair 57-58 L40-h** (the PI-approved ~54 used P6's 57.0 s x 0.863); C0r +28-29 (PI approval); label table 25.1 CPU-h of pod CPU (above); gate run per arm: [U] not measured here (P7's gate time is in its `launch_gate.log`). Both arms do not run concurrently (card: same pod, same GPU, C1 then C0).

## 10. Built by HANDOFF_2 (was "not built")
- **G-LIVE (f) warm-start identity** and **(h) lever isolation on the real model**: `refe/p5c_gate.py`, wired into `gate_checks.check_live` (a P5-C profile only) through `p5c_live_rows`; `launch_gate.run_checks` now builds the build capture for a P5-C profile even if only G-LIVE is asked. Tests: `refe/test_p5c_live.py`.
- **Post-training G-ADAM** (card 10.4): `refe/p5c_postgate.py` + `refe/test_p5c_postgate.py`.
- **The pod chain** `refe/p5c_pod.sh` + `refe/test_p5c_chain.py` (section 11).
- Still the MM's: fill the three pins, A2 amendment, the W-A7 / W-TRANSP / W-FAM read blockers, `P5C_LABEL` (tc20s | p5r) and `P5C_LABEL_WORKERS`.

## 11. The chain: `bash $PKG/refe/p5c_pod.sh [--with-c0r] [--dry-run] [--clear-fail <stage>]` **[POD]**
Replaces the manual sections 4, 5, 7 and 8 (they remain the reference for what each stage does). Needs in the env: `P5C_LABEL` (`tc20s` or `p5r`: the A2 decision, never defaulted), `P5C_LABEL_WORKERS` (pinned, as P7: 13 on the L40 pod), `P5C_HF_TOKEN_FILE`; optional `P5C_NSHARD` (default = `nproc`, REFUSED above it), `P5C_REUSE_P7_LAB=1`, `P5C_STOP_AFTER=<stage>`. Stages and markers (`$DATA/refe_p5c/stage/<name>.ok` = date + key):
`precheck` (always runs: token file, label, workers, NSHARD <= nproc, tree == SHIP_MANIFEST, no FAIL marker, no `ckpt_last.pt` in an unfinished arm's `--out`) -> `inputs` (base sha256 recorded, pod idle) -> `label` (`p5c_pod_label.sh`) -> `table` (own `traj_label_table.py verify`: rows == covered == 150,603) -> `pins` (REFUSES until the MM has filled `P5C_BASE_SHA256`, `P5C_LABEL_TABLE_SHA256`, `P5C_RANK_POLICY` in the landed source AND they equal the measured base sha, table sha and policy; never fills them) -> `lab`, `mem`, `egr` (gate evidence) -> per arm `gate_<arm>` (launch_gate run + verify), `preflight_<arm>`, `train_<arm>`, `postgate_<arm>` (G-ADAM; C0's also `--other` C1: the arms differ), `egress_<arm>` (HF, verified by the remote listing). C1, then C0; C0r only with `--with-c0r`.
- A failing stage writes `<stage>.FAIL` itself and the chain refuses until `--clear-fail <stage>` (logged). A killed trainer leaves `trainstart_<arm>.ok` and a `ckpt_last.pt`: archive the run dir (`mv ... .killed_N`), then `--clear-fail train_<arm>` releases it and the arm restarts at step 0.
- Markers are keyed: label and table by (label kit sha, NSHARD), everything after the pins by the shipped tree's `SHIP_MANIFEST.json` and the base / table sha. **Shipping a new tree after the labelling does not redo the labelling.**
- Dry run: prints every command, runs none, writes nothing on the volume.
- Per-arm `--n-params-from $P7RUN/config.json` is the exact trainable count the G-ADAM gate expects (a P5-C model is P7's model; the head is a sibling and is not in `params.trainable`).
