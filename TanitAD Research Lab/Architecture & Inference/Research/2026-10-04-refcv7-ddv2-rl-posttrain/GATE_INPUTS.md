# GATE_INPUTS — what the launch gate needs for the WP-RL arms, and where each piece of evidence lives

**For the Master Mind.** The binding PI rule of 2026-09-26: no training starts without a launch-gate
PASS token. `stack/scripts/launch_gate.py` has refc profiles only. It has **no profile for an RL
post-training** (a different trainer, a different object). So this file maps the gate's checks
onto this run and names the artifact for each. Every artifact below is MEASURED on Thor, on the
copy of the launch tree that will run the arms. **You run the gate; WP-RL does not mint tokens.**

**What would be launched (one token for all four arms):**

* **Tree:** `/home/nvidia/refcv7_post/rl/tree`. It is a copy of `/home/nvidia/refcv7_run/fec3a0dccf`
  (the refcv7 launch tree) with exactly three files added: `stack/tanitad/rl/ddv2_refcv7.py` (md5 `701638df39c958d5d1b8580b4c0aef14`),
  `stack/scripts/ddv2_rl_refcv7.py` (md5 `b46d4014128a608b3abaa56c720dc2c1`) and `stack/tests/test_ddv2_refcv7.py`
  (md5 `b049a992bf5c90aacc3b5e232eb013d7`). The copies in `code/stack/` are byte-identical.
  No existing file is changed. One DATA file was added for the stage-3 battery's distance keeping:
  the banked B1 eval lead block, md5 `33a48e15a52eb9dbd69ecd6026fd5023`, at its repo path.
* **Cold start:** `/home/nvidia/refcv7_post/rl/ckpt_50400.pt`, md5 `b418d0fc4a92a6848c246a6a7c50207b`
  (= `D:/refcv7_eval_kit/ckpt/ckpt_50400.pt`). It is checked by the trainer at every start.
* **Config:** `/home/nvidia/refcv7_run/runs/refcv7-r101-s0/config.json`, md5
  `e6512a01b9c70f0e4a4dac581621984a`.
* **Data:** the train split through the launch tree's own loader (`refcv7_loader.build_eval_dataset`
  with the train remap). The eligible-window list is `/home/nvidia/refcv7_post/rl/out/
  windows_train.json`, digest `2e53c5d8ab5474bc…` (507,588 windows over 4,256 clips).
* **Argv:** one canonical JSON per arm, written by `code/thor/wprl_argv.py --steps 2000 --batch 32
  --micro 8` (SPEC A-1). Copies are in `raw/argv/` and on Thor in `/home/nvidia/refcv7_post/rl/gate/`.
  `wprl_arm.sh` refuses to start unless `/home/nvidia/refcv7_post/rl/gate/PASS_WPRL.json` exists
  **and** contains the sha256 of that arm's argv file. The token format is yours. The script only
  needs these sha256 strings to appear in it:

  | arm | file | sha256 |
  |---|---|---|
  | RL (seed 0) | `ARGV_rl_s0.json` | `9484eb09f16a60a67153ccf736ad1f656f81e7dc3b4dbd58164927eed3d67065` |
  | RLOFF (seed 0) | `ARGV_rloff_s0.json` | `56b8d44283e8ecbf73485ab9b4629323af61a7834e93bf057fc35f49b5fbea0a` |
  | RL-SHUF (seed 0) | `ARGV_rlshuf_s0.json` | `caa3dcfca2a6145ad44aba547364d7c5420374aeb0ac8866f42d2b964969cbf1` |
  | RL-s1 (seed 1) | `ARGV_rl_s1.json` | `7fff2c82dc03aed5ce24d7559fb59c2187529fa339a8953a91c66ffcdfbcb890` |

  `ARGV_AUDIT.json` (I-11): RLOFF differs from RL only in `--arm` and `--rl-weight`, RL-SHUF only in
  `--arm`, and RL-s1 only in `--seed` (plus `--out-dir`). **PASS.**
* **Launch, once the token exists:** `setsid nohup /home/nvidia/refcv7_post/rl/wprl_queue.sh`. It
  runs the four arms in order, checks each with `wprl_check.py`, exports, then runs the T0 heldout
  reads. The T1 battery rolls (`wprl_t1.sh <name> <ckpt> <seed>`) follow per export. They were
  validated by a CPU smoke on Thor (`battery_cpu/roll.json`, 2 windows).

| gate check (refc analogue) | what must hold for WP-RL | evidence (all on Thor; copies in `raw/`) | status |
|---|---|---|---|
| **G-SUITE** | the new tests pass on the tree that runs; existing RL suites unchanged | `test_ddv2_refcv7.py` **36/36** on Thor and on the dev box (14 mutation cases). `test_ddv2_rl / _il / _refc_chain / pdm_proxy` pass. 3 `pdm_proxy` savgol tests fail on Thor for want of scipy: pinned ENV failures (`launch_gate_thor_env_failures.json`). | PASS |
| **G-DVB** (declared = built) | the trainable set is the generator only; every arm's argv differs from RL in its ONE named variable (SPEC I-11) | `run.json` → `trainable` (13,804,924 params / prefixes `traj_proj, time_mlp, layers, adaln, cascade, control_head`). `grad_groups` per step lists every group that received gradient. `ARGV_AUDIT.json` from `wprl_argv.py`. | PASS |
| **G-EVAL** (the loader builds the identical model) | the RL binding IS the deployed sampler; the trainer's own forward feeds it | `diagnose_train.json` D1: **12/12 windows bitwise** (fan and u0), and the trainer's `compute_losses_v3` captures the **same** sampler inputs 12/12. D3: per-tick states equal the decoder's roll **bitwise** 12/12. | PASS |
| **identity at step 0** | `--lr 0 --rl-weight 0` → the export equals the cold start bitwise | `smoke/identity/identity.json`: **1131/1131 tensors IDENTICAL**. Must-fail arm: the 4-step RL export reads **DIFFERENT**, 134/1131 tensors (`smoke/rl_full/identity.json`). | **PASS** (must-fail fired) |
| **G-LIVE** (smoke at the real recipe) | every loss term finite; gradient reaches every trainable group; the RL coefficient is live on RL and exactly 0 on RLOFF; RL-SHUF permutations are derangements; DAC live | `smoke/{rl_full,rloff,rlshuf,timing32}/metrics.jsonl` read by `wprl_check.py`: **PASS on all 6 runs**. Each I-check fired on its own mutation (`--selftest`). RLOFF coefficient exactly 0.0 (2/2 steps). RL-SHUF derangements 2/2. DAC live (cand DAC 0.34–0.51). Gradient reaches every generator group. | **PASS** |
| **G-CKPT** (save → resume) | a run segmented 2 + 2 reads the same windows in the same order, and continues the same RNG streams, as an uninterrupted 4-step run | `smoke/rl_seg` vs `smoke/rl_full`: **same windows on every step**, reward and param-delta **identical**, logged loss within 5e-8. Cross-process resume: the CPU dry run. Lock yield and re-acquire: the CPU test on a private lock (`raw/dry_yield/`). | **PASS** |
| **reward known values** | the human control; constructed off-road / collision / faster / slower candidates; mutations RED | `census_train300.json` (300 TRAIN windows): human NC 0.997, DAC 0.983 (rule R2, A-0), TTC 0.983, C 0.950. Unit tests and 11 mutation tests in `test_ddv2_refcv7.py`. `diagnose_train.json` D5. | PASS (R1 FAILED its control and was replaced by R2 under A-0, pre-registered) |
| **cost** | the measured s/step and memory at the registered batch | `smoke/timing32/metrics.jsonl`: **30.85 s/step** at B = 32 / micro 8. D4: peak 5.53 GB at micro 8. ⇒ K = 2,000 ≈ 17.1 h per arm (≤ 24 h). | **PASS** (A-1) |

**Regression arms that must FAIL (and do, or are tests that do):**

1. An RL export vs the cold start → **DIFFERENT**: the identity check can see a change.
2. `wprl_check.py --selftest` makes each I-check fire on its own reintroduced defect:
   a non-finite loss; a non-zero RLOFF coefficient; zero positive advantage; human DAC 0.5; a dead
   DAC (cand_dac_mean ≡ 1).
3. `test_ddv2_refcv7.py`'s mutation tests must each go RED:
   * sidewalk counted as road, under both R1 and R2;
   * DAC from the centre point;
   * the lateral axis flipped;
   * agents dropped;
   * static and dynamic NC swapped;
   * the TTC projection removed;
   * a binding that drops agents, drops BEV, or rolls on the wrong speed;
   * a query-mixing decoder pass;
   * tick states rolled from the wrong speed.
4. The driver REFUSES a dataset without a 10 cm map store (DAC would be dead) and a join without
   track ids (NC/TTC would be blind). Both are refusals in `Ctx.__init__`.

**What the gate cannot check from these artifacts (named, not hidden):**

* Training-seed variance before RL-s1 runs.
* Whether CUDA non-determinism moves a resumed run numerically. Measured on the smoke as a loss
  difference, reported, not gated.
* The battery roll of an export. That is stage 3 (the export loads strictly through
  `refcv7_loader`, which the identity export already exercises).
