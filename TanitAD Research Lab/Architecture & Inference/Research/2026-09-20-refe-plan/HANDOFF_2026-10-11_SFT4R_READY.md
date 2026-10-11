# HANDOFF 2026-10-11 (b): SFT-4R tooling is built; the SFT-4 u400 full-navtest read is CONFIRMED

Successor to `HANDOFF_2026-10-11.md`. Read this and the files it names, nothing else.

## State

| item | state | where |
|---|---|---|
| Stop consequences (weekly-limit stop + pod stop) | DONE | `STOP_CONSEQUENCES_2026-10-11.md` |
| P4 deliverables | landed `1730bc29` | — |
| PREREG_SFT4R + TRAPS egress entry | landed `57bf6fcd` | `eval/PREREG_SFT4R.md` |
| SFT-4 u400 stage 2 on the 923 A5 tokens | **ADOPT** +3.04 [+1.21, +4.87], landed `b582f3cc` | `eval/RESULT_SFT4.md`, register row `D-REFE-SFT-4` |
| SFT-4R pod tooling + PREREG_SFT4R Amendment 1 | landed with this file | `refe/hf_egress.py`, `refe/pdm_label_complete.py`, `refe/sft4r_{relabel,chain,pod}.sh` |
| Unseen-subset read (reported, not gating) | script landed; controls PASS | `eval/unseen_subset.py` |
| Chain review: 3 defects fixed (G3 grep could never match, stale markers, unread lane rc) | landed with PREREG_SFT4R Amendment 2 | `refe/sft4r_chain.sh`, `refe/test_sft4r_chain.py` (20/20, 4/4 mutations RED) |
| **Full navtest of the u400 checkpoint** | **DONE: CONFIRMED** +3.79 [+3.05, +4.54], 84.46 → 88.25 PDMS, 136 logs, 0 adverse; unseen +3.85, minus-C1 +3.75 | `eval/RESULT_SFT4.md` "Full navtest", register row `D-REFE-SFT-4` |

## The full-navtest read (DONE 2026-10-11; kept for the record: steps 1–5 below were all carried out)

- **Command (do NOT restart it while it runs):**
  `eval/stage2_sft.py --ckpt E:/Projects/TanitAD/data/refe_runs_eval/sft4_peek/model_sft4_B_u400.pt --name
  sft4_B_u400_navtest_full --rule navsim_v1_lane --tokens E:/Projects/TanitAD/data/refe_navtest/navtest_full_tokens.json`
- **Rule, committed before the run** (`eval/RESULT_SFT4.md`, "Next arm" §1):
  - 12,146 tokens, 136 logs, against the deployed 84.46 PDMS.
  - The script's ADOPT, REFUTED and NOT PROVEN map one-to-one to **CONFIRMED**, **REFUTED** and **NOT CONFIRMED**.
  - CONFIRMED makes it a deployment **candidate**. Deploying is the PI's call. The lever is not claimed before a
    training-seed replicate.
- **Outputs:**
  - `ZZSTAGE2 sft4_B_u400_navtest_full <verdict> <mean> [lo, hi]` in the task output;
  - `result.json` in the E: package copy, under `raw/2026-10-05-sft-stage2/sft4_B_u400_navtest_full/`;
  - the CSV at `E:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_sft4_B_u400_navtest_full/`.
- **Then, in order:**
  1. Run `python eval/unseen_subset.py`. It defaults to the 11,023 tokens and 122 logs no earlier read of this
     checkpoint touched (md5 `ab59e479`) and prints `ZZUNSEEN …`. Report it as **REPORTED, NOT GATING**.
     - Its controls passed 2026-10-11: the identity reads exactly +0.00 [+0.00, +0.00]; the A5 reproduction gives
       +3.04 [+1.21, +4.87]; a wrong log count gives `GATE_FAIL`.
  2. Copy the E: raw dir to the D: package, without the `.npz` seam.
  3. Append a "Full navtest" section to `eval/RESULT_SFT4.md` with all four families. STRATEGIC is UNAVAILABLE in
     NAVSIM.
  4. Update the `D-REFE-SFT-4` row in `Project Steering/GOALS_AND_CLAIMS.md`, on a fresh tip copy.
  5. Land through the mirror and verify the remote ref.

## What the SFT-4R relaunch needs (all blocked on the PI)

1. **PI:** check in RunPod whether the old pod is **Stopped** or **Terminated**.
   - If Stopped, copy off `/workspace/data/refe_sft2/{train_pdm,heldout_pdm}`, `/workspace/data/refe_sft4/` and
     `/workspace/data/refe_p2/` first. The PDM labels cost ~11 h CPU.
2. **PI:** provision a new pod and put a **write-scoped HF token** on it. No agent copies a token anywhere.
3. **PI or a fresh agent:** confirm the lane-subset queue source. The candidate is the HF banks' `onpolicy_queue`. Set
   `LANE_QUEUE_SRC` to it, or `sft4r_relabel.sh` refuses with `ZZLANE_TRAIN_EXIT 9`.
4. **Deploy:**
   - copy `refe/` to `/workspace/refe-sft4r/run`;
   - copy the two `raw/2026-10-04-lane-discipline/heldout_*_lane_labels.jsonl.xz` files to
     `/workspace/refe-sft4r/raw/2026-10-04-lane-discipline/`;
   - restore the banks from `Sayood/tanitad-refe-navtrain-banks`;
   - run `sft4r_relabel.sh`, then `sft4r_chain.sh` after `ZZRELABEL_SFT4R_DONE`.
   - ⚠️ The pod scripts have had `bash -n`, a flag-to-argparse trace, and a test of the chain's gate helpers on
     SFT-4's banked smoke logs (`refe/test_sft4r_chain.py`). They have never had a dry run. Read `chain.log` for
     the first `REFUSED:` line.
5. P2 follows on the same labels, after an egress amendment to `eval/PREREG_P2.md`.

## Do not

- The dev-box GPU is free again (the full-navtest run finished 14:15:46Z).
- Do not land `MANIFEST*.txt` or any scratchpad `pod_probe.py`: they hold the pod IP and port, and the repo is public.
