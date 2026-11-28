# PLAN: the navtest read of the four zero-training scorer points (BASE, C0A, R0B, M25)

Written 2026-10-17 by the executing agent, AFTER the frozen `PREREG_FREEPOINT.md` (sha256 `2a3c1928...`, verified unchanged) and BEFORE any navtest number.
Deviations from the prereg live ONLY in `ADDENDUM_1_2026-10-17.md`. Evidence classes: M = measured (artifact cited), E = estimated, U = unverified.
`<pod>` = the A40 pod (address deliberately not written here: this package lands in a PUBLIC repo).

## 1. Is there already a navtest per-proposal PDM table for the deployed model's 64 proposals? NO.

| candidate | what it is | covers the need? |
|---|---|---|
| `data/refe_navtest/proptable/navtest_final/proposals.npz` | 12,146 x 64 proposals (NAVSIM grid) + the SIX logits of `model_final.pt`, **pre route-fix**, **no PDM scores** | no. M (2026-10-17, CPU): the u400 executed pose is bit-equal (<= 1e-6 m) to a member of this fan on only 8,774 / 12,146 tokens; 2,051 tokens differ by > 1e-3 m (up to 41 m) because the adopted config (Amendment 9 `pdm_route` goal fix) changes the goal and therefore the fan. It is NOT the deployed config's fan. |
| C3 `raw/2026-10-11-c3-selection-headroom/table.npz` | all 64 proposals harness-scored, u400 adopted config | **1,058 tokens only** (usable as an extra end-to-end check, not as the read) |
| `seams/p2tab/main` (p2-zero-tail) | P2's own 64 columns | 509 tokens, P2's fan, not u400's |
| the RL-val dumps (`scores_*.npz`, dose.json) | scorer logits on 3,000 navtrain RL-val sets | wrong tier (selection, navtrain); no navtest logits exist for any scorer state |

Consequence: a navtest read of a scorer state needs the scorer's logits on navtest, and those need the scorer's visual context `sctx` (`REFe.score_trajectories(trajs, sctx)`, `refe/model.py:699`; `sctx = visual_ctx`, 7.9 MB per token fp32 => 96 GB for 12,146 tokens, so it cannot be cached).
**A trunk pass over all 12,146 tokens is therefore unavoidable, and it is shared by the four points** (generator, trunk and LoRA are frozen in every state: `scorer_finetune.py` trains only `score_q_mlp`, `score_dec`, `score_head`).

## 2. The cheapest correct route (one trunk pass, three scorers, picks-only harness)

```
A  dev box, CPU   fp_capture.py   the seam's own loop with a STUB model: records ego[7], goal[G], 4 frame refs per token + the per-log camera rig   (0 GPU)
B  dev box->pod   fp_ship.py      ships the 546 frame zips (9.7 GB) in <= 1.5 GB log-chunks, deletes each chunk after it is processed
C  pod, A40       fp_forward.py   u400 model, fp32, batch 1 (the planner's numerics): ONE forward per token  ->  traj[64,20,3], logits of BASE, C0A, R0B  [64,7]
D  dev box, CPU   fp_picks.py     M25 = c + 0.25 (r - c) in float64; picks by the trainer's sel_score (AT.picks); executed poses (A7 repair + to_navsim); one seam npz per point
                  fp_score.py     W3's harness (score_e.py, unchanged) on each point's seam; full navtest, no --tokens
E  dev box, CPU   fp_read.py      paired log-cluster bootstrap (proxy_eval.boot, 10,000, seed 20260927), the prereg rows mechanically
```
Why not the alternatives:
* **Four stage-3 reads** (`stage2_sft.py`, one full seam per checkpoint): M 4.9 h seam + 0.37 h harness EACH on the 4060 => about 21 h of GPU, and M25 has no checkpoint (it mixes two scorers' logits), so it cannot be read this way at all.
* **A 64-column PDM table** (`proposal_table.py` pattern): 64 x 12,146 = 777k harness scores = 64 x 0.37 h = about 24 CPU-h (E, from M 1,335.7 s per 12,146), and it still needs the trunk pass for the logits. A picks-only read scores at most 3 distinct poses per token (BASE is banked): at most 36.4k scores, about 1.1 h.
* **The dev-box GPU for the trunk pass**: M seam 1.44 s/token (4060, 17,496.8 s / 12,146) => about 4.9 h and it is in use by another job today (nvidia-smi 5.6 / 8.2 GB, 60 %). The PI named the A40.

## 3. Costs (every rate cited)

| step | resource | estimate | basis |
|---|---|---|---|
| A capture | dev box CPU, 6 shard processes, no GPU | 0.5-0.8 h wall | M `aux_navtest_forward` warm cost: scenario init + route 0.63 s/token, so 12,146 x 0.65 s = 2.2 h single process, / 6 shards. U: native access violations were seen in the seam on the nuPlan builder (`run_a9_chunked.py`); the capture runs per shard and is resumable |
| B ship | uplink dev box -> pod | **2.1-2.3 h**, runs concurrently with C | M today: 100 MB in 80 s = 1.25 MB/s; 4 parallel streams 160 MB in 124 s = 1.29 MB/s (the link is the cap, not the flow); 9.7 GB / 1.29 MB/s = 2.1 h |
| C forward | A40, 1 process, `OMP_NUM_THREADS=2`, `nice -n 10`, fp32 | 1.7-2.5 h GPU if fed; **the run is upload-bound at 0.67 s/token (2.3 h)** | M A40 fan dump 0.25 s/set (bf16 autocast, batch 4, pre-extracted pixels, `rls1/dump_val_s0of1.log`); the planner path is fp32 and batch 1, so E 2-3x slower => 0.5-0.75 s/token. The 4 JPEG decodes + resizes per token run on 1-2 threads: E 0.1 s |
| C pod disk | `/workspace` (97 % full, 13 GB free and falling about 1 GB/h under the bank build) | footprint <= 2 GB frames + 0.5 GB outputs; the worker REFUSES to take a chunk when free < 6 GB | M `df` 2026-10-17 |
| C outputs | pod -> dev box | 12,146 x (traj 15 KB + 3 x 7 KB logits) = about 0.45 GB | E |
| D picks + seams | dev box CPU | minutes | |
| D harness | dev box CPU (the harness is CPU only, RAM guard >= 4 GB free) | 3 points x 0.37 h = 1.1 h; BASE re-score 0.37 h; determinism re-run 0.1 h | M 1,335.7 s per 12,146-token run (`P2_READOUT_RUNBOOK` s.3), sequential |
| E read | dev box CPU | families6 about 2 min per arm | M |
| **Total** | | **about 3.5 h to the forward outputs + 1.6 h harness**; pod GPU about 2.3 h | |

**Pod constraints honoured** (the P6 twin-bank build runs on the pod's CPUs, ~11-CPU quota): the forward worker starts ONLY after `/workspace/ops/bank3c/full/started_epoch` exists (its own first line checks it and exits 3 otherwise); it uses `OMP_NUM_THREADS=2 nice -n 10`, one process; it never touches bank3c files; no `pgrep -f` / `pkill -f` (it records its own pid and is stopped by that pid). State at the time of writing (M, 2026-10-17 ~07:0x UTC): `parts_started_epoch` exists, `PARTS_DONE` and `started_epoch` do NOT => the re-roll has not launched, so no GPU job was started (see `HANDOFF.md`).

## 4. Exact inputs and pins

* Model: `model_sft4_B_u400.pt`, md5 `8e382eebb35b...` (dev box `E:/Projects/TanitAD/data/refe_runs_eval/sft4_peek/`; pod `/workspace/data/rls1/base/`). BASE = this checkpoint's own scorer (identical to `r0 step0` / `c0 step0` base in dose_rls1b.py:40).
* C0A = `rls1-c0/scorer_states/scorer_states_u1000.pt`, key `A`; R0B = `rls1-r0/scorer_states/scorer_states_u1000.pt`, key `B` (dev box `C:/Users/Admin/rls1_adam_dl/`; 34.4 MB each; format `{A:{score_q_mlp,score_dec,score_head}, B:{...}, update:1000}`, M). Strict `load_state_dict`; the worker records the sha256 of every loaded tensor set and refuses if `update != 1000`.
* Adopted config of the u400 read, reproduced: rule `navsim_v1_lane`, A7 last-pose heading repair ON, Amendment 9 goal fix `pdm_route`, goal mode, per-sample rig, token set `navtest_full_tokens.json` (12,146 tokens, 136 logs, md5 `3a8b3653...`).
* Selection of the picks: `attrib_rls1b.picks` (`sigmoid` in float64, `p0*p1*(5p2+5p3+2p4)/12*p6`, ties -> lowest index): the trainer's `sel_score` and the exact function `dose_rls1b.py` used. M25 = `c + 0.25*(r - c)` on all 7 outputs in float64 (the FULL variant).
* Estimator: `proxy_eval.boot` (log-cluster, 10,000, seed 20260927) through `stage3_full.read_pair`. It answers "another draw of LOGS" only (one training seed per state, deterministic selection).

## 5. Identity controls (committed in `ADDENDUM_1` before any point number exists)

IC1 capture == the A9 recorded inputs (3,410 tokens, bit-exact) | IC2 the replicated scorer path == the model's own scores (bit-exact, every token, in-run) | IC3 BASE picks == the banked u400 executed poses | IC4 BASE through the harness == the banked CSV (same-pose tokens exactly; mean within 0.10) | IC5 C0A twice bit-identical (logits and harness) | IC6 C3 table lookup == harness on the 1,058 C3 tokens | deliberate-regression arms D1-D3, each must turn its check RED.

## 6. Reading rule (mechanical, prereg verbatim)
DEPLOYABLE-SAFE(point) <=> against BASE: PDMS CI lower bound > 0 AND NC, DAC and TTC CI upper bounds >= 0. Rows: FP-R0B if R0B is safe; else FP-M25 if M25 is; else FP-NONE. Reported with every sub-score x100 and its paired CI, the four families (STRATEGIC = PAUSED, PI 2026-09-19 item 25), and the lever view R0B - C0A and M25 - C0A.
