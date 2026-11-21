# PLAN_IREP: the inference floor of bar E clause 3 (`floors.irep_abs_delta`), as a command sequence

Written 2026-10-17 by a TanitAD build agent (Master Mind brief). **Nothing here has run on a pod and no SAFEOBJ state exists** (no R0d). Every number below is either **M** measured (artifact cited) · **I** inherited from the free-point / navtestE packages, not re-run · **E** estimated · **U** unverified.
Companion: `BUILD_IREP.md` (what was built, what the tests prove), `../navtestE/PLAN_E.md` (the main read this feeds). `<pod>` = the A40 pod (address deliberately not written: this package lands in a public repo; the shipper reads it from `<work>/../pod_ssh.txt`).
`PY` = `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I -B -X utf8`. `P` = `C:/Users/Admin/sobj_pkg/raw/2026-10-17-safety-objective`. `IW` = `C:/Users/Admin/sobjIrep` (any short path: MAX_PATH). Run from Git bash (`C:/Program Files/Git/bin/bash.exe`; a bare `bash` is WSL).

## 0. What is registered, and what this plan produces

`PREREG_SAFEOBJ.md` §7, row *inference*: "the I-rep of `PREREG_RL_S1.md` §3 (**eval batch 8 vs 32, shuffled order, non-deterministic kernels**) on R0d-B u1000, dev box". Bar E clause 3: the mean contrast must exceed 2 × max(seed floor, I-rep |Δ|). The consumer, `navtest_E.py irep`, wants two harness tables of ONE state read twice and writes `{"irep_abs_delta", "unit": "PDMS x100"}`.

| | |
|---|---|
| question the number answers | *would another INFERENCE run (a different eval batch, a different token order, whatever kernels the library then picks) give another picked pose, and how much PDMS does that move?* (H-ESTIM-SEED-1, the third variance. Not the episode draw, not the training seed.) |
| the two reads | **read A**: eval batch **8**, token order = `RandomState(1).permutation`; **read B**: eval batch **32**, token order = `RandomState(2).permutation`. Same checkpoint (u400 trunk), same scorer state (R0d-B u1000), same machine, same 12,146 tokens, fp32, no autocast. The seeds are conventions of this plan (the prereg says "shuffled" and names no seed); they are recorded in every chunk |
| "non-deterministic kernels" | read as: **no determinism flag is set anywhere**, so the batch shape alone selects the library's kernels. **M**: `grep -c "deterministic\|float32_matmul_precision\|allow_tf32\|cudnn.benchmark"` over `fp_forward.py`, `model.py`, `ckpt_io.py` = 0 for all three, with the same-breath control `grep -c no_grad fp_forward.py` = 1. The variant adds none |
| producer | `irep/fp_forward_irep.py` = `navtestE/fp/fp_forward.py` + 82 added lines (`--eval-batch`, `--token-order-seed`); **byte-identical outputs at the defaults** (`BUILD_IREP.md` §2) |
| output | `irep.json` for `navtest_E.py build --irep-json` (PLAN_E step 9), plus `irep.check.json` (every gate, every fact) |

## 1. Where it runs: the dev box does NOT fit the registered read (VRAM)

**M** — `irep/logs/vram_estimate.log` (`vram_estimate.py`): the REAL `REFe` forward (ViT-L, 4 cameras of 512×960, 1,925 tokens per image, fp32, `no_grad`, `model.py` sha256[:16] `e6011cd83c846dfb` = the file the worker runs) executed on the meta device with a dispatch-mode byte tracker; the tracker reads its two analytic controls exactly (C1 peak = three tensors = 12,582,912 B; C2 a freed tensor leaves) before any result is printed.

| eval batch | images | weights | activation peak | model peak | **~nvidia-smi** (×1.15 + 0.4 GB context, **E**) | peak op |
|---|---|---|---|---|---|---|
| 1 (the planner's own) | 4 | 1.29 GB | 0.43 GB | 1.71 GB | **2.4 GB** | `gelu` (the MLP hidden) |
| **8** (read A) | 32 | 1.29 GB | 3.41 GB | 4.69 GB | **5.8 GB** | `gelu` |
| **32** (read B) | 128 | 1.29 GB | 13.62 GB | 14.91 GB | **17.5 GB** | `gelu` |

* The dev-box card is an RTX 4060 with **8,188 MiB**, shared with another tenant whose footprint varies. **M** `nvidia-smi` 2026-10-17: **4,229 MiB used / 3,729 MiB free** early in this session (from the transcript, not a saved artifact) and **1,582 MiB used / 6,376 MiB free** at 15:40 (`logs/nvidia_smi_devbox.log`). ⇒ **batch 32 never fits the card (17.5 GB vs 8.2 GB total, even idle); batch 8 (5.8 GB, ±15 % **E**) fits only in a window like the second one, with a margin of 0.6 GB that is inside the estimate's own error; batch 1-2 fits always.** Past this card's capacity the Windows driver spills silently to host RAM (the package's own finding: `POD_HANDOFF.md:120`, and REVIEW_5 §4.2, where the 4-camera training rows were "host-memory-fallback thrash, not compute"), so "it ran" would not mean "it fit".
* **The attention backend is a switch and both settings were measured.** With PyTorch's math SDPA (which is what the meta device runs) the peak is 18.6 GB at batch 8 and 70.6 GB at batch 32 (the [images, 16, 1925, 1925] score matrix); with the memory-efficient kernel (fp32, head_dim 64, no mask: eligible) the table above. The dev box was **not** touched to find out which one CUDA picks. **Derived from a measurement of the package's own:** `raw/2026-09-21-review5-cameras/cost2.txt` of the refe-plan package (RTX 4060, ViT-L, 1 camera, **training**, `torch.cuda.max_memory_allocated`: peaks 3.55 → 5.84 GB for batch 1 → 2, i.e. **2.29 GB marginal per image**) is impossible under math SDPA (24 layers × one saved [16, 1925, 1925] fp32 score matrix = 24 × 237 MB ≈ **5.7 GB per image**), so the efficient kernel is what that card ran. The batch-32 verdict does not depend on this (it fails under both).
* Distinct rigs per shuffled batch: the navtest set has **17 distinct rigs over 136 logs (M, `freept/work/calib.npz`)**, so a shuffled batch of 32 builds up to 17 frustums; the tracker with 32 distinct rigs gives the identical peak (the pos3d tensors are below the trunk's peak).
* **Two reads on two machines would not be one I-rep, and the guard refuses it** (`one_checkpoint_and_one_machine`: ckpt md5, device, GPU name, torch and CUDA versions must be single-valued across both reads). So batch 8 on the dev box plus batch 32 on the pod is not an option.

**Proposal: both reads on the A40** (48 GB: batch 32 needs ≈ 17.5 GB), in **the slot between R0d and R1d**, with R0d-B u1000 as the scorer state. This replaces the registered "dev box" with a place that can hold the registered batch sizes; the registered batch sizes are kept. *(The alternative that keeps the dev box, e.g. batch 2 vs 8, is a different definition of the I-rep and is not proposed: a smaller batch change moves the kernels less, so it would understate the floor.)* **Master Mind decision.**

## 2. Gates before anything starts

| # | gate | why |
|---|---|---|
| I-G0 | R0d finished; `scorer_states_last.pt` is at **update 1000** and downloaded to the dev box (`<dev>/rls1-r0d/scorer_states/scorer_states_u1000.pt`, key `B`) | the worker refuses any other update |
| I-G1 | the A40 is free for ≈ 5 h of mostly-idle GPU and ≈ 2 CPU threads | CLAUDE.md: never add load to a pod that is training. The slot is the Master Mind's |
| I-G2 | the worker's `started_epoch` gate (P6 re-roll): honoured, or the Master Mind passes `--no-gate` | inherited from `fp_forward.py serve`; same call as PLAN_E G2 |
| I-G3 | `python irep/make_variant.py --check` prints `ZZVARIANT_CHECK_OK` and `irep/logs/ZZTEST_irep.log` ends in `ZZTEST_PASS` | the variant on the pod is the tested one |
| I-G4 | `/workspace/ops/sobjIrepA` and `…/sobjIrepB` are owned by this read alone; the shipper and `navtest_E.py plan` refuse `/workspace/ops/freept` | the free-point run's directory |
| I-G5 | **the one path this build could not exercise: the REAL model at batch > 1.** The tests use a stand-in model (the brief: tiny synthetic CPU state); the real `REFe.forward` takes `calib [B, 4, 16]` and folds the camera axis into the batch, so batch > 1 is the code's design, but it was not run here. Run the free-point smoke recipe at batch 2 with the order swapped (CPU, no GPU, about 2-4 min and ~6 GB RAM; the free-point's own 2-token CPU forward took 120 s): see below | **U** until run. Expected (**E**): `token` array identical, `traj` and every `logits_*` within 1e-4 of `freept/work/smoke/out.npz` row for row (float noise, not O(1)), `ic2_maxabs` all 0.0 |

```
OMP_NUM_THREADS=4 REFE_BACKBONE_ROOT=E:/Projects/TanitAD/data/backbones $PY -c "import sys, runpy; sys.dont_write_bytecode=True; sys.path.insert(0,'C:/fp/PK/refe'); f='P/irep/fp_forward_irep.py'; sys.argv=[f,'run','--inputs','C:/Users/Admin/freept/work/smoke/inputs.json','--calib','C:/Users/Admin/freept/work/smoke/calib.npz','--ckpt','E:/Projects/TanitAD/data/refe_runs_eval/sft4_peek/model_sft4_B_u400.pt','--frames-dir','E:/Projects/TanitAD/data/refe_navtest/frames','--device','cpu','--eval-batch','2','--token-order-seed','2','--state','C0A=C:/Users/Admin/rls1_adam_dl/rls1-c0/scorer_states/scorer_states_u1000.pt:A','--state','R0B=C:/Users/Admin/rls1_adam_dl/rls1-r0/scorer_states/scorer_states_u1000.pt:B','--out','IW/smoke_irep.npz']; runpy.run_path(f, run_name='__main__')"
```
(`P` and `IW` written out in full when run. Seed 2 gives `RandomState(2).permutation(2) = [1, 0]`: the two tokens are visited swapped, so a misalignment shows as an O(1) difference.) Then compare `IW/smoke_irep.npz` to `freept/work/smoke/out.npz` with `np.abs(a - b).max()` per array.

## 3. The sequence (dev box `D`, pod `P`)

`R` = `/workspace/data/rls1`. `<dev>` = dev-box dir of the arm states. `DATA` = `E:/Projects/TanitAD/data/refe_navtest`. Repeat 3–7 for read A (batch 8, seed 1, `r=A`) then read B (batch 32, seed 2, `r=B`); **read B starts after read A is pulled** (the uplink is the binding constraint, not the GPU).

| step | where | command | done when |
|---|---|---|---|
| 1 | D | `for r in A B; do mkdir -p IW/$r; cp C:/Users/Admin/freept/work/{inputs_full.json,calib.npz,plan.json,ship_plan.json} IW/$r/; done` | the four files exist in both (IC1 of the free-point run: 3,410 tokens bit-equal to the A9 recorded inputs, **I**) |
| 2 | D | `$PY irep/irep_tool.py expected --state R0d_B=<dev>/rls1-r0d/scorer_states/scorer_states_u1000.pt:B --out IW/EXPECTED_IREP.json` | `ZZIREP_EXPECTED_OK`: the tensor-set sha256 of R0d-B u1000, **computed from the dev-box copy, not from the pod**. When the main read's `EXPECTED_STATES_E.json` exists, `make` can take it instead and its `R0d_B` entry must equal this one |
| 3 | D→P | `$PY navtestE/sobj_ship.py prep-pod --work IW/$r --remote /workspace/ops/sobjIrep$r` then `ssh <pod> "cat > /workspace/ops/sobjIrep$r/fp_forward_irep.py" < irep/fp_forward_irep.py` and `ssh <pod> "md5sum /workspace/ops/sobjIrep$r/fp_forward_irep.py"` | `ZZSHIP_PREP_OK` (md5 of inputs/rig/`fp_forward.py`, the u400 md5 `8e382eeb…`, the four model-code hashes) **and** the pod's md5 of `fp_forward_irep.py` equals `md5sum irep/fp_forward_irep.py`. *(Not run: `sobj_ship.py` ships the ORIGINAL `fp_forward.py`; the variant is the one extra file, **U**)* |
| 4 | P | `ssh <pod> 'cd /workspace/ops/sobjIrep<r> && OMP_NUM_THREADS=2 REFE_BACKBONE_ROOT=/workspace/data/backbones PYTHONPATH=/workspace/refe-plan-rls1/refe nice -n 10 setsid nohup /workspace/venv-teacher/bin/python fp_forward_irep.py serve --work-dir /workspace/ops/sobjIrep<r> --inputs inputs_full.json --calib calib.npz --ckpt /workspace/data/rls1/base/model_sft4_B_u400.pt --state R0d_B=/workspace/data/rls1/arms/R0d/scorer_states_last.pt:B --device cuda --eval-batch <8 or 32> --token-order-seed <1 or 2> [--no-gate] > serve.log 2>&1 < /dev/null & echo $! > worker.pid'` | (`<r>` = A or B, `<8 or 32>` and `<1 or 2>` per read; typed out, because inside single quotes the local shell would not expand a variable) `serve.log`: `worker up … states ['R0d_B']`. Stop only by pid: `kill $(cat worker.pid)` (never `pgrep -f` / `pkill -f`) |
| 5 | D (detached: WMI `Win32_Process.Create`, CreateFlags `0x08000200`) | `$PY navtestE/sobj_ship.py run --work IW/$r --remote /workspace/ops/sobjIrep$r > IW/$r/ship.log` | `ZZSHIP_ALL_PULLED`; `serve.log` has 9 `ZZFWD_CHUNK_DONE … ic2_max 0` lines |
| 6 | D | `$PY irep/irep_tool.py seams --chunks IW/$r/out --work IW/$r --state R0d_B` | `ZZIREP_SEAMS_OK R0d_B 12146 tokens`; `IW/$r/seam_R0d_B.npz`, `seam_BASE.npz`, `points_E.npz`, `picks_report_E.json`. This is `navtest_E.build_seams(names=("R0d_B",))`: the free-point picker (`attrib_rls1b.picks`, float64) + A7 pose repair, **reproduces the live picker's seam bit for bit on the real dry chunk** (`BUILD_IREP.md` §2). It refuses a chunk that carries no `irep` record |
| 7 | D | `$PY navtestE/fp/fp_score.py score --work IW/$r --points R0d_B --label-prefix refe_sobj_irep${r}_ --parallel 1` | `ZZSCORE_OK R0d_B`; the harness's OWN status `PASS` in `IW/$r/score_logs/refe_sobj_irep${r}_R0d_B.status.json`; the CSV `DATA/score/refe_sobj_irep${r}_R0d_B/refe_sobj_irep${r}_R0d_B.csv`. **This is the harness step that turns the forward outputs into the CSV** (step 6's seam → W3's `score_e.py`, unchanged). The two reads get different labels, so neither overwrites the other. Step 7 of read A can overlap read B's upload |
| 8 | D | `$PY irep/irep_tool.py make --state R0d_B --a-chunks IW/A/out --b-chunks IW/B/out --a-work IW/A --b-work IW/B --a-csv DATA/score/refe_sobj_irepA_R0d_B/refe_sobj_irepA_R0d_B.csv --b-csv DATA/score/refe_sobj_irepB_R0d_B/refe_sobj_irepB_R0d_B.csv --expected-states IW/EXPECTED_IREP.json --out IW/irep.json` | `ZZIREP_MAKE_OK`; `IW/irep.json` (the consumer's `{"irep_abs_delta", "unit"}` plus `irep_guard`) and `IW/irep.check.json`. **Exit 2 and no `irep.json`** if any gate fails: it names the gate |
| 9 | D | `$PY navtestE/navtest_E.py build --work W --irep-json IW/irep.json --out-dir W/result` (PLAN_E step 9) | `ZZNAVTEST_E_OK`. `build` reads only `irep_abs_delta` and `unit`; the guard block rides along unread |

`make` runs, before it calls the consumer: both reads are ONE scorer state (tensor-set sha256 equal across both reads and every chunk) and equal to the independently computed hash · update 1000 and the key is the name's suffix · one checkpoint md5, one device/GPU/torch/CUDA · the same tokens chunk by chunk and exactly 12,146 · eval batch exactly 8 and 32 (self-reported **and** consistent with the batch sizes the run says it executed), token-order seeds present and different, the two permutations different · each CSV was scored from the seam built from **its own** chunks (the seam's arm string and token set; the harness's counts.json names that seam file; the harness's status is PASS on that many tokens; the CSV carries exactly those tokens) · the two reads are two files. Then it runs `navtest_E.py irep` in its own process and **recomputes the value from the two CSVs itself**; the two must agree to 1e-9.

## 4. Cost (every rate cited)

| step | resource | estimate | basis |
|---|---|---|---|
| 2–3 | dev box, small uploads | < 10 min | **E** |
| 4–5, per read | A40 forward | **≈ 2.0 h at the free-point rate**: 12,146 × 0.60 s | **M** `freept/work/out/chunk_000.npz` and `chunk_001.npz` meta: **1,521 tokens in 918.3 s = 0.6037 s/token** and **1,440 in 866.3 s = 0.6016 s/token** on the A40 (fp32, batch 1, 4 JPEG decodes per token in the loop). A batched trunk can only remove launch overhead (the work per token is the same, and the decode stays serial in the loop), so the batched reads are expected at **≤ 2.0 h, probably 1.6–2.0 h (E, U)** |
| 5, per read | uplink dev → pod, **10.06 GB of frames** (the worker deletes each chunk after use, so nothing is reusable) | **≈ 2.6 h** | **M** `freept/work/ship.log`: the four chunks uploaded so far moved 4.96 GB in 4,642 s = **1.07 MB/s** (one upload failed and was retried in that span); 10.06 GB / 1.07 MB/s = 9,400 s. The forward is hidden behind the upload, so **a read costs ≈ 2.6 h wall** |
| 4–5, both reads | | **≈ 5.2 h wall** (2 × 2.6 h, sequential); A40 GPU busy ≈ 3.2–4.0 h of it | sum |
| pull, per read | pod → dev | **≈ 0.25 GB** | **M** 31.9 MB for 1,521 tokens with 3 logit sets (21 KB/token); this read carries 2 |
| 6 seams | dev CPU | minutes | **E**; the real 2-token dry chunk took 6-23 s including the imports (`logs/ZZTEST_irep.log`, `real:i_rep_seam_step…`) |
| 7 harness, per read | dev CPU, RAM guard ≥ 4 GB free | **0.371 h** at `--parallel 1` | **I** 1,335.7 s per 12,146 tokens (`P2_READOUT_RUNBOOK` §3 via free-point PLAN.md) |
| 8 make | dev CPU | seconds (1-4 s on the 40-token fixture, `logs/ZZTEST_irep.log`); ≈ 15 s expected on 12,146 tokens | **M** fixture / **E** |
| **total** | | **≈ 5.2 h forward + 0.37 h (the tail harness, read B) + 0.1 h ≈ 5.7 h wall**; read A's harness overlaps read B's upload. A40 training-hours: 0 | |
| if the dev box could host it (it cannot at batch 32) | RTX 4060, fp32 | ≈ 2.7 h per read | **I** the u400 seam ran 1.44 s/token (17,496.8 s / 12,146) of which the scene build is 0.63 s/token ⇒ the forward ≈ 0.8 s/token **E**; and **M** a CPU forward of the real model is 60 s/token (`freept/work/smoke1.log`), so CPU is not an option |

**Not built, and why:** running both reads off ONE upload (a worker that processes each chunk twice before deleting it) would save the second 2.6 h upload but make the A40 the binding resource (2 × 2.0 h ≈ 4.1 h vs 5.2 h): a 1.1 h saving for a changed `serve` loop that nothing here can test against a pod. Offered, not built.

## 5. Reading rule

`irep.json` goes to `navtest_E.py build --irep-json`, which writes it as `floors.irep_abs_delta`; the READER applies bar E clause 3: the mean contrast must exceed 2 × max(RL-val seed floor, navtest |R0d−R1d|, this value). It is **one draw** of a quantity that is itself noisy (two reads → one |Δ|); like the other floors it is *necessary, not sufficient*, and a third read is not registered. `irep.check.json` carries the diagnostics that make a small number readable: `facts.pick_changes_a_vs_b` (how many tokens changed their picked proposal between the reads) and `facts.logit_maxabs_a_vs_b`. **If the value is exactly 0.0** the file says so in `note`: either no pick changed (then read the two numbers above) or the two reads were bit-identical (batch-invariant kernels): in both cases clause 3 rests on the other floors, and **that is a finding, not an error**. A value that is large is not suspect for being large.
The I-rep is **navtest-level** (it moves PDMS ×100 through the trunk and the scorer). The prereg's RL-S1 §3 I-rep was defined on the RL-val lookup; if the Master Mind wants that one too it is a different producer (not built here), and its unit is not the one the E file carries.

## 6. Failure handling

* `ZZFWD_IC2_FAILED` / driver `rc 2`: the replicated scorer path differs from the model's own score. **Inherited gate, kept.** On the GPU this can mean the scorer kernels are not repeatable at that batch shape; that is information about the floor, so bring it to the Master Mind rather than loosening the gate.
* `REFUSED` from `make`: read `irep.check.json` → `failed`. Do not edit a CSV, a seam or a chunk to pass a gate. The two usual causes: a read restarted with other settings (the dev-box driver refuses mixed settings on resume; the pod worker does not, so check the `irep` record of every chunk), or the CSV of the wrong label.
* A killed worker: rerun step 4 (finished chunks are skipped by their `.done`). Dropped ssh: rerun step 5.
* Pod disk < 6 GB free: the shipper holds. Free nothing outside `/workspace/ops/sobjIrep*`.
* Cleanup: delete `/workspace/ops/sobjIrep{A,B}` only after the pulled chunks are verified on the dev box and banked.

## 7. Open points (each blocks or changes the read; none is decided here)

1. **Dev box vs A40** (§1): the registered batch 32 does not fit the dev box. Proposal: A40, both reads, the slot between R0d and R1d. *Master Mind.*
2. **Two uploads** (§4): 5.2 h of mostly-idle A40 occupancy, because the free-point worker deletes frames after use. The one-upload variant is not built.
3. **`started_epoch` gate** (I-G2): the Master Mind's call, as in PLAN_E.
4. **`fp_score.py` hard-codes `SP`** (another session's scratchpad where `score_e.py` lives; PLAN_E open point 7): if it was cleaned the harness step fails; copy `score_e.py` and edit `SP` in the COPY.
5. **The step-3 variant upload is manual and untested against a pod** (**U**).
6. **The seeds 1 and 2 are this plan's, not the prereg's.** They are recorded in every chunk and in `irep.json`.
7. **One draw.** See §5.
8. **"Shuffled" is shuffled within each frame chunk, not across the whole navtest.** The worker receives the frames in 9 chunks (341 to 1,619 tokens, about 15 logs each; `ship_plan.json`) and deletes each after use, so the permutation is drawn per chunk. A batch of 32 therefore mixes tokens of the ~15 logs of its chunk and never tokens of two chunks. The model scores every token independently (no cross-token term), and what a batch changes numerically is mostly its shape, so this is expected to matter little; it is stated because the prereg's words ("shuffled order") do not say which.
