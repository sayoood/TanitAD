# RUNBOOK_MAPFT -- the map-only fine-tune, from NT A1's done-marker to the K-PLAN-ID verdict

Prepared 2026-10-26 (08:00 CET) by the launch-preparation agent. **Nothing in this kit has been run on Thor.** Sealed pre-registration
`PREREG_MAPFT.md` sha256 `f8774a7b5eed0d8ac427f49c6097f32e82cd989a51bb629bd9b1708689f5739c` (matches its `.sha256`). PI decision
D-R9-MAPHEAD-PRECISION (a): cap 6 Thor-hours, Thor after NT A1, 50-step timing smoke first, STOP if over the cap.
Kit directory (dev box): `C:/Users/Admin/refcv8_pf/refcv9_mapft/kit` = `$K` below. Thor root: `/home/nvidia/r8ns/mapft` = `$R`.
Evidence classes: MEASURED = run in this session (artifact named); INHERITED = from NT / refcv8 notes, not re-measured; ESTIMATED = arithmetic.

## 0. READ FIRST -- this is not one command

| why | consequence |
|---|---|
| MEASURED (reading `launch_gate.py` at `70211d6`): `PROFILES["refcv8-mapft"]` carries open item `MAPFT-BUDGET`; `finalize()` turns ANY open item into verdict INCOMPLETE and `verify_token` refuses a non-PASS token. | At `70211d6` **no argv can ever PASS this profile.** A `launch_gate.py` edit closing the item must be COMMITTED first (the item's own `closes_when`). That edit is the Master Mind's / PI's. `mapft_close_budget.py` makes it a reviewable, tested diff. |
| The gate's bind records (G-MAP-OVERFIT, G-BOX-OVERFIT) and the dev-box G-SUITE-PINNED evidence are bound to the **argv sha256**, which contains N. | They cannot be pre-run: they start only after `smoke` -> `budget` has produced the final argv. The Thor G-SUITE (commit + tree bound) is the only argv-free piece. |
| The 6 Thor-h formula `N = min(10000, floor(0.85*(21600 - smoke_wall_s - 1800)/s_step))` budgets the smoke and the run. | **Gating (bind ~2 h INHERITED, gate, 2 RED arms) is outside it.** Whether the PI's 6 Thor-h cap includes gating time is an open question (D3 below). |

ESTIMATED timeline from NT's done-marker T0 (~19:00 CET): ship 5 min -> smoke 10 min -> budget 1 min -> bind ~2 h (Thor suite 45 min and dev-box pinned 25 min run alongside)
-> gate ~45 min -> RED arms ~30 min -> **fire at about T0 + 3.5-4 h**; the run itself projects to 4.57 h of the 6 h (MEASURED arithmetic: 0.85*(21600-420-1800) s at any s_step).

Sealed budget at representative `s_step` (smoke_wall 420 s; MEASURED with `make_mapft_argv.py`): 3.0 s -> N 5491 | 4.0 -> 4118 | 6.0 -> 2745 | 8.0 -> 2059 |
9.7 (refcv8's full-model rate, INHERITED) -> 1698 | 12.0 -> 1372 | 16.0 -> 1029 | 16.33 -> 1008 | **N < 1000 -> STOP** (s_step >= 16.34 at that smoke wall).

## 1. Decisions the Master Mind owns -- BEFORE 19:00

* **D1. Closure edit (needs MM/PI go: launch_gate.py edits are flagged Security Weaken).**
  `python $K/mapft_close_budget.py --repo <checkout root> --mode open-only --note "<who/when>"` -- prints the diff; `--dry-run` first.
  `open-only` (RECOMMENDED): `open_items=()`, the budget flags stay un-pinned by the profile; the token still binds the FULL argv by content, so only the gated N can launch.
  It is N-independent, so **C0 can be committed today** and the tree/plan/Thor-suite do not wait for the smoke. `literals` additionally writes N/warmup/ema/eval-every into
  `required_values` -- strictest, but the commit then follows the smoke and everything waits (not recommended).
  After the edit run `stack/tests/test_map_only_ft.py` (40 pass on both modes, MEASURED on scratch copies) and commit with `mm_commit.py` -> **C0** (40 hex). `C12 = ${C0:0:12}`.
* **D2. Suite TIP.** Use the commit immediately before C0 (then the overlay is just the 2 edited files). `mapft_gsuite.sh plan` refuses deletions / mode changes and cross-checks TIP+overlay == `git archive C0`.
* **D3. Does the 6 Thor-h cap include gating?** If yes the cap is already gone before the run starts (gating ~3.5 h); if no, the run (4.6 h) plus gating is ~8 h of Thor.
* **D4. `pi_cost_approval_mapft.json` is a DRAFT, not a PI record** (`max_s_per_step` 16.0 = floor of 0.85*(21600-1800-900)/1000, the s_step at which N hits 1000). The gate's cost rule is 6.4 s/step x 1.25 = 8.0, else
  PI-DECISION. If the token reads PI-DECISION on cost, pass `--pi-cost-approval` ONLY after replacing the file's `pi_verbatim` with the PI's actual words.
* **D5. Post-run zero-gradient gate.** The supervisor runs it by default and a FAIL is FINAL (no done-marker). `smoke` rehearses it on the smoke checkpoint with the control `_map_hires\.(refine|near|near_refine)\..*`
  and `fire` refuses unless that rehearsal PASSed; `fire --zerograd-off` is the explicit, logged opt-out. (Rehearsal mechanics MEASURED on a synthetic map-only-shaped checkpoint + a dead-parameter mutant; the real checkpoint is untested.)

## 2. Before 19:00 -- dev box only, zero Thor contact (all MEASURED working in this session)

```
cd $K
bash mapft_gsuite.sh plan --tip <TIP40> --cand <C0> --stage F:/refcv9_mapft/gate          # ZZGS-PLAN-OKZZ   (24 s for a real pair)
bash mapft_ship.sh --dry-run tree <C0>                                                    # ZZMAPFTSHIP-TREE-OKZZ  (builds the 459 MB tar, verifies the box-audit md5 literals; touches nothing on Thor)
bash mapft_devbox_post.sh selfcheck                                                       # ZZMAPFT-DEVSELFCHECK-OKZZ  (CPU: identity + comparator controls on the banked traces)
bash mapft_devbox_post.sh rehearse                                                        # GPU ~3.5 min, queues behind the dev-box gpu_lock, needs >= 6 GB free commit: the BASE ckpt stands in for model_ema.pt;
                                                                                          # expect  ZZMKKPLANID_SMOKEZZ PASS  (proves the rig reproduces its own banked trace bit for bit -- K-PLAN-ID's premise)
```
`rehearse` was NOT run (the dev-box GPU lock was held by refcv9-ssfresh and free commit was 7 GB). It is the one unmeasured premise of K-PLAN-ID; run it while Thor trains.

## 3. At NT's done-marker (T0)

Check first: `ssh -n tanitad-thor 'grep -o "\"done\": *true" /home/nvidia/r8ns/nt/runs/nt-a1-s0/summary.json'` must print `"done": true`; `nvidia-smi` shows no compute app.
`H="bash $K/mapft_thor_run.sh"` runs a long Thor step detached and waits on its marker (read on Thor, bracket-escaped). Every step below ends at a **marker**; no step is "green" without it.

| # | where | command | expected marker | STOP / meaning of a miss |
|---|---|---|---|---|
| 1 | dev | `bash $K/mapft_ship.sh kit` | `ZZMAPFTSHIP-KIT-OKZZ` | md5 mismatch -> re-ship, never extract |
| 2 | dev | `bash $K/mapft_ship.sh tree <C0>` | `ZZMAPFTSHIP-TREE-OKZZ` | CRLF export (box-audit md5) / implausible tar / import probe resolves outside the new tree |
| 3 | Thor | `$H start pf mapft_prelaunch.sh preflight` ; `$H wait pf ZZMAPFTPREFLIGHT-OKZZ` | `ZZMAPFTPREFLIGHT-OKZZ` | any `PF-FAIL` (NT not done, GPU not empty, a data path missing, <60 GB disk) |
| 4 | Thor | `$H start smoke mapft_prelaunch.sh smoke` ; `$H wait smoke ZZMAPFTSMOKE-OKZZ --max-min 40` | `ZZMAPFTSMOKE-OKZZ` and a line `ZZMAPFT-N-<n>ZZ` | **N < 1000 -> `STOP (PREREG_MAPFT s1)`: do NOT launch; report s_step to the PI.** Smoke not reaching step 50 / FrozenStateGuard refusal -> read `$R/smoke/train.log` |
| 5 | Thor | `$H start budget mapft_prelaunch.sh budget` ; `$H wait budget ZZMAPFTBUDGET-OKZZ` | `ZZMAPFTBUDGET-OKZZ` (argv at `$R/argv/refcv8-mapft-s0.argv.json`) | `ZZMAPFTBUDGET-STOPZZ` (exit 11) = N < 1000: STOP. Audit mismatch -> do not use the argv |
| 6a | dev | `scp tanitad-thor:$R/argv/refcv8-mapft-s0.argv.json F:/refcv9_mapft/gate/` ; compare `sha256sum` with Thor's | equal | else re-copy |
| 6b | dev | `bash $K/mapft_gsuite.sh thor-start --cand <C0>` | `ZZGS-STARTED-OKZZ` | refuses unless NT's done-marker exists; Thor CPU only (`nice 19`, CUDA hidden), ~45 min |
| 6c | dev | `bash $K/mapft_gsuite.sh devbox-pinned --cand <C0> --argv-file F:/refcv9_mapft/gate/refcv8-mapft-s0.argv.json` | `ZZGS-PINNED-OKZZ` | a non-PASS evidence record is refused; ~25 min; needs >= 6 GB free commit |
| 7 | Thor | `$H start bind mapft_prelaunch.sh bind --commit <C0>` ; `$H wait bind ZZMAPFTBIND-OKZZ --max-min 240` | `ZZMAPFTBIND-OKZZ` | closure/record assertion failure (the map then the box binding run; ~34 + ~82 min INHERITED) |
| 8 | dev | `bash $K/mapft_gsuite.sh thor-wait --cand <C0>` then `thor-verdict --cand <C0>` | `ZZGS-SUITE-BOUND-OKZZ`; `F = $R/suite_<C12>/suite_bound.json` | verdict not a clean PASS / precheck FAIL (the launch tree and the suite's candidate tree differ) |
| 9 | Thor | `$H start gate mapft_prelaunch.sh gate --commit <C0> --devbox-evidence $R/devbox_evidence_<C12> --suite-verdict $R/suite_<C12>/suite_bound.json [--pi-cost-approval $R/kit/pi_cost_approval_mapft.json]` ; `$H wait gate ZZMAPFTGATE-PASSZZ --max-min 240` | `ZZMAPFTGATE-PASSZZ`, token `$R/gate_<C12>/PASS_<C12>.json` | any non-PASS verdict prints its reasons; the profile must be CLOSED in this tree (preflight `profile-closed`) |
| 10 | Thor | same args with `redarms` ; `$H wait red ZZMAPFTRED-OKZZ` | `ZZMAPFTRED-OKZZ` | an arm graded `-ESCAPED-OR-MISCREDITED`: the freeze guard is decorative -> do NOT launch |
| 11 | Thor | **MM:** `$H start fire mapft_prelaunch.sh fire --commit <C0> [--zerograd-off]` ; `$H wait fire ZZMAPFTFIRE-OKZZ` | `ZZMAPFTFIRE-OKZZ` | verify != MATCH, no CAUGHT grade, rehearsal not PASS, no supervisor process afterwards |

Steps 6b, 6c and 7 are concurrent (the NT chain ran its suite beside its bind the same way). `mapft_prelaunch.sh --dry-run <mode> --root <scratch> ...` prints what a step would do without a lock or GPU.

## 4. During the run

* Progress marker in `$R/runs/refcv8-mapft-s0/sup_refcv8-mapft-s0.log`: `ZZrefcv8-mapft-s0-<step>-<steps>-<errs>-<launch>ZZ`. Trainer `step_s` in this trainer's rows is `elapsed_s` CUMULATIVE (÷ step), not the train_v6 pre-divided form.
* To stop: kill the **supervisor** first, then the trainer, by explicit PID from `ps -eo pid,args | grep 'sup_refcv[7].sh'` (never `pgrep -f`/`pkill -f`). The supervisor owns the GPU lock fd 9 by design.
* The run writes `summary.json {"done": true}` and the supervisor adds its zero-gradient record. `ZEROGRAD_FAIL.json` present = the gate rejected the checkpoint (files remain complete).
* `NT_LAUNCH_PENDING` is left in place on purpose (NT's L7 chain yields to it).

## 5. After the run (PREREG_MAPFT s2) -- Thor side first

```
$H start post_status mapft_postrun.sh status        # expect ZZMAPFT-RUN-COMPLETEZZ (last step >= N-1, model_ema.pt + ckpt.pt, done-marker)
$H start kfrozen     mapft_postrun.sh kfrozen       # CPU only. wait ZZMAPFT-KFROZEN-PASSZZ    -> $R/post/KFROZEN.json
$H start manifest    mapft_postrun.sh manifest      # wait ZZMAPFT-MANIFEST-OKZZ (refuses if K-FROZEN is not PASS)
```
K-FROZEN = the landed `assert_map_only_finetune.py` on `ckpt.pt` (raw + EMA views) and on `model_ema.pt`, AND "at least one changed tensor in EACH of the 3 map modules" (the tool alone passes a run that trained nothing),
AND two controls: src-vs-src must read FAIL on the per-module criterion; `model_ema.pt` must equal `ckpt.pt`'s EMA view. Both dst files must carry step N exactly.
Then on the dev box (the checkpoint is sent over the LAN: ~63 MB/s INHERITED, 2 x 400 MB):

```
bash $K/mapft_devbox_post.sh fetch                  # ZZMAPFT-DEVFETCH-OKZZ: every file's sha256 + size equal Thor's MANIFEST.sha256 (else do not use anything)
bash $K/mapft_devbox_post.sh kplanid                # one GPU job under gpu_lock
```
`kplanid` = identity checks (seals, base ckpt md5, windows md5, **mapsel_roll.py byte-identical to the script that banked the base trace**) -> MAPFT **smoke roll** (308 windows, ~3.5 min) bit-compared with `trace_smoke.npz`
-> if it differs, a BASE smoke re-roll as the rig-determinism control -> else the FULL roll (4,754 windows, ~48 min, MEASURED 2,903 s for the base) -> `K-PLAN-ID` on plan8 / fan / sel (the sealed three; every other array is listed beside, never silently dropped)
-> the MAPPATH twin derived with the registered `mappath_transform.py` and compared with `trace_s0_mappath.npz` -> **K-RAST-MOVED** vs `rast_s0` with the GT plane bit-equal on every window as the alignment control.
Verdict lines (`ZZMK...ZZ` in `F:/refcv9_mapft/post/logs/chain.log`): `K-PLAN-ID = PASS | FAIL | RIG-NONDETERMINISTIC | INVALID`. **RIG-NONDETERMINISTIC means the BASE checkpoint does not reproduce its own banked trace:
bit-equality is unmeasurable here -- STOP and report; do not relax the sealed text.** Only if K-FROZEN, K-PLAN-ID and K-RAST-MOVED are all PASS are the PREREG s3 bars (M1-M5 on `MAPSOFT-G-XF(MAPFT)`) read -- that instrument is not part of this kit.

## 6. STOP conditions (named blockers, per RULE ZERO s3)

N < 1000 (PI report) · launch-gate refusal (reasons are printed; the profile must be closed) · smoke failure · a RED arm not CAUGHT · K-FROZEN FAIL · K-PLAN-ID FAIL (planner reads the map modules -> report to the PI before anything else) ·
RIG-NONDETERMINISTIC · any md5/sha256 mismatch on a shipped file.

## 7. Verified in this session (MEASURED unless marked; the self-test outputs are in `verification_log.out`, the dry check in `dryrun_check.out/.json`)

* `mapft_prelaunch.sh`: every mode dry-run on a closed scratch tree (`F:/refcv9_mapft_scratch/dryroot`), a negative control on the unclosed tree, the RED-arm grader rehearsed on synthetic tokens, `bash -n` clean.
* `make_mapft_argv.py` + `test_make_mapft_argv.py`: 29 tests pass (literal budget cases incl. the N < 1000 STOPs, mutants, smoke-reader literal, CLI round trips); the smoke argv (201 tokens, sha256 prefix `ad8d7b8c9feee71e`) now written LF-only.
* `mapft_dryrun_check.py`: 26/26 — the derived argv PARSES with the landed trainer's parser (clean autocrlf=false export of `70211d6`, `tanitad.__file__` asserted inside it) and is accepted by `map_only_ft`'s argv refusals; broken argvs (keeping `--grad-share-every`, `--b1-guard-*`, ...) are REFUSED, each by its own signature.
* `mapft_ship.sh kit|tree`: rehearsed against a local stand-in for Thor (md5 at both ends, aside-move on repeat, import-probe pattern); the real `70211d6` tree tar is 459,018,240 B, byte-faithful (box-audit md5 literals match).
* `mapft_gsuite.sh`: `plan` on a real pair (9 changed paths, 3,216 compared, 24 s); `thor-verdict` rehearsed locally with the REAL `suite-bind` and precheck on NT's real verdict: PASS, and a one-line mutation of the launch tree turns the precheck FAIL (tree digest mismatch), restore -> PASS.
* `mapft_kfrozen.py`: 6 cases (good + 5 mutants: nothing changed, two modules only, trunk weight, BN stat, BN counter) read as expected; end-to-end `run` on a two-view `ckpt.pt` fixture, plus a C2 mutant.
* `mapft_kplanid.py`: 14 literal self-tests; on the REAL banked traces the comparator reads trace_s0 == trace_s0 (PASS, both controls), reads trace_s0 vs its MAPPATH twin as FAIL on `fan` only, and the registered derivation reproduces the banked twin bit for bit; K-RAST-MOVED discriminates (same -> FAIL, one pixel -> PASS, GT plane differs -> INVALID, different file sets -> INVALID).
* Thor paths confirmed read-only (08:00): all 18 argv data paths + `ckpt_ema_48852.pt` (403,200,712 B); `thor_gate_runner2.py`, `gate_verdict.py`, `refcv8_audit/pyt`, `refcv8_run/gsuite_precheck.py`, the train venv python, `thor_gpu.lock`; `/home/nvidia/r8ns/mapft` does NOT exist yet (the ship creates it); 118 GB free; NT A1 was at step 8,101 and the GPU at 97 %.

## 8. Known gaps (not hidden)

* The gate, bind and RED arms have never run on a map-only argv; only their argv-level acceptance (dry check) and the grader logic are verified. A live refusal there is expected to be informative, not a kit bug.
* `s_step` and smoke wall are unmeasured until the smoke. The smoke logs every 10 steps (the launch uses 50): the pooled `s_step` from rows with step >= 20 is conservative.
* The dev-box MAPSEL roll's bit-reproducibility (`rehearse`) and the zero-gradient gate on a real map-only checkpoint (`smoke`) are the two unmeasured premises; both are cheap and both fail loudly before any 6-hour spend.
* The closure edit changes `launch_gate.py` and one test only; the full Thor suite (CAND vs TIP) is what proves nothing else regressed.
