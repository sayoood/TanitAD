# Brief: ROLLHIST phase B2 prep (pod tree + chain scripts; do NOT launch)

You are a TanitAD build agent working for the Master Mind (MM). Read `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip first:
`git -C C:/Users/Admin/tanitad-push show $(git -C C:/Users/Admin/tanitad-push ls-remote origin agent/arch-inf-20260803 | cut -f1):"Project Steering/AGENT_OPERATING_STANDARD.md"`.
Stage only: no commit, no push. No secrets; never write an IP, port or pod id into any file. No edits to protected files.

## What is decided
* PI go, first-hand, 2026-11-20: **both seeds, about 5.2 LoRA pod-h** (RH_s0, B0_s0, RH_s1, B0_s1).
* Code of record: commit `2bda9e1a382986cfdce3bb455822678b42a816e2`, branch `agent/v8f-rollhist-20261120`, pushed.
* The card: `F:/TanitAD-scratch/rollhist/PREREG_V8F_ROLLHIST_DRAFT.md`. Read sections 1.3, 2, 3.1 (R1-R5), 4, 5 and 9. Build records: `F:/TanitAD-scratch/rollhist/build/b1/HANDOFF_B1.md`. Its "B2, pod side" list, steps 1-5, is your task list, with the fix-round notes below it.
* **STILL OPEN with the PI:** the readout of the training history, C* (`--readout causal_g --acc-gain 1 --path-exec`, token `causal_g1x`, recommended) or ADD63 `causal` (`--readout causal_g --acc-gain 1`, no path-exec, token `causal_g1`).
  * Make the readout ONE parameter of the chain (`RH_READOUT=causal_g1x|causal_g1`), rendering both the bank flags and the `--dagger-bank-readout` token from it.
  * Prove that both settings render, and that `rollhist_pair_check.py` accepts each with its matching bank header. If the checker hard-codes `causal_g1x`, say so; do not edit committed code.

## The template, which ran successfully
STAGEB / dscale10k on the pod: `/root/dscale10k/` (`argv/`, `bank4.sh`, `PILOT_REPORT.json`, `argv_gen_new.py`) and its dev-box records under `F:/TanitAD-scratch/datascale/` (stageb, reps1/build). Adapt them; do not redesign them. Also look at `F:/TanitAD-scratch/n1dc/build/b2/` for the per-step `.done` markers, the flock-on-fd-9 hold and the sentinel style.

## Pod access and its traps
* Run `source C:/Users/Admin/lorapod/podenv.sh` for `pssh` and `pscp`, with `"$POD_TARGET:/path"`.
* ⛔ **Another session (TanitADRuled) trains on the pod GPU.** Light work only:
  * tar push and extract;
  * `sha256sum`;
  * `cat` and `ls`;
  * writing scripts.
* ⛔ No pytest, no model load, no bank pilot, no python beyond a sub-second check.
* ⛔ Never touch `/root/gpu_queue.lock`, `/root/N1DW_READY`, `/root/ruled`, `/root/strat1`, `/root/dscale10k` (read only) or any process.
* Work under `/root/rollhist/` only.
* Don't use `pgrep -f` or `pkill -f`.
* Pod python is `/root/venv/bin/python`. Pod git has no credentials, so build the tree with `git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive 2bda9e1a stack taniteval` on the dev box. Verify it blob by blob on the pod and report the count.

## Do
1. **Pod tree** `/root/rollhist/tree`, verified blob-by-blob against 2bda9e1a.
2. **Inputs:** locate on the pod (read only) or name the HF path of:
   * UE_s0.A `ckpt_last.pt` (sha256 `2fd4b747...`) and UE_s1.A (`f13df8ce...`);
   * the `U_s{0,1}.B.tmpl.json` argv templates;
   * the union data inputs the stage-B argv names.
   Check `/root` free space against the stage-B gate's 60 GB.
3. **Scripts** in `/root/rollhist/`, written and pushed, NOT started:
   * `rh_gsuite.sh`: G-SUITE on the new tree against the STAGEB tree's base. Reuse a base junit if one is on the pod.
   * `rh_pilot.sh`: HANDOFF_B1 step 3, checks (a)-(d), with the RH config from `RH_READOUT` and the pinned identity config. The checks run on the pilot outputs and write `PILOT_RH.json`.
   * `render_argv_rh.py`: HANDOFF_B1 step 4 for s0 and s1, then the pair check.
   * `rh_arm.sh`: selfcheck, mint, train, zero-grad gate, K-CLIP, as STAGEB and N1DC do.
   * `rh_chain.sh`, one process and one log, in this order:
     1. take `flock` on fd 9 of `/root/gpu_queue.lock` and hold it to the end;
     2. G-SUITE;
     3. the pilot, refusing on any failed check;
     4. bank s0 (all shards, merge), re-checked;
     5. RH_s0, then B0_s0;
     6. bank s1, then RH_s1, then B0_s1.
     Use per-step `.done` markers for resume. The last line is `ZZ ROLLHIST DONE ZZ` or `ZZ ROLLHIST FAIL ZZ <why>`. The marker protocol (the MM sets `/root/N1DW_READY` before the launch so the other session's next slice skips) is the MM's; just document it.
   * `rh_marker_watch.sh`: modelled on `/root/strat1/n1dc/n1dc_marker_watch2.sh`. Read it, do not edit it.
4. **A local waiter**, `C:/Users/Admin/actsens_launch/wait_rollhist.sh`.
5. **The exact launch command** for the MM.

## Deliverables in `F:/TanitAD-scratch/rollhist/build/b2/`
* the scripts;
* `REPORT_B2_PREP.json`: tree count and blob check, where each input was found, both readout renders, disk;
* `HANDOFF_B2.md`, at most 1.5k tokens;
* a sha256 manifest.

Hand back in at most 250 words. Never wait on anything for more than about 4 minutes.
