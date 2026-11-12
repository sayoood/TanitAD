# BUILD4: VDAC A5 arm HB (the LoRA trunk copy that feeds only the drivable head) and HC's reachability fixes

UTC 2026-11-12. CPU only (`CUDA_VISIBLE_DEVICES=-1`), interpreter `C:/Users/Admin/venvs/tanitad` (Python 3.13.5, torch 2.11.0+cu128), source = git mirror `a3cc6a77` exported read-only to scratch. Nothing launched, no pod, no GPU, no network, nothing staged or committed.
Evidence class of every line: MEASURED (this build; test output quoted below) unless marked. PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.
**The draft's A5 arm is still DRAFT; nothing here freezes anything.** Hold = `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/vdac_a5_build4/` (full files + `_diffs/*.diff` against `a3cc6a77`).

## 0. Headline

**DONE through priority 6** for HB (H1-H4, H6, H7, H8, H9) and HC (C1, C2). **NOT built:** H5 (micro-batch accumulation of the copy's backward), H10 (the in-sample dump at the end of every arm), the `aux_drv/2` loader for Stage 0 / S2 / G-EVAL (the strict primitive `aux_hb.load_v2` exists and is tested; the tools that call it do not), and the HC reader (C3). Reasons in section 5.
**Needs the MM:** section 6 (LoRA lr confirmation, pinned-hash and fingerprint moves, VDAC_SUITE, SHIP_MANIFEST, memory probe on the pod, pause timeouts).

## 1. P6's own LoRA learning rate (MM decision 1)

**2e-4, the same as the global `--lr`: P6 had NO separate LoRA lr.** P6 was trained by `train.py` (the checkpoint `refe_runs/p6_paper_faithful/model_final.pt`; `vdac_stage0_forward.py` rebuilds HA from `model_sft_A.pt` meta and the P6 checkpoint path, `launch_gate.py:516`):
* `PK/refe/train.py:1462`  `params = [p for p in model.parameters() if p.requires_grad]` (the backbone's LoRA `A` / `B` are trainable parameters of the model: `model.py:216-232`), and `PK/refe/train.py:1469` `opt = torch.optim.AdamW(params, lr=a.lr, ...)`: ONE group, no per-group lr, no `lr_mult`.
* `PK/refe/p6_pod.sh:487` P6's recorded argv: `... --epochs 25 --epoch-unit scenes --batch 4 --accum 64 --lr 2e-4 --weight-decay 0.01 ...`.
* HB therefore uses **`--aux-drv-lora-lr 0.0002`** (`launch_gate.VDAC_HB_LORA_LR`, `aux_hb.P6_LORA_LR`); the draft's 5e-5 was a HYPOTHESIS and is replaced in the DRAFT diff of the prereg (flagged "MEASURED by BUILD4, MM to confirm"): `PK/raw/2026-11-09-scorer1p6-prereg/vdac/a5_draft/PREREG_VDAC_P6_A5_DRAFT.md` in the hold. Schedule: the trainer scales the scorer's warm-up + cosine `lr_at(upd)` by `--aux-drv-lora-lr / --lr` (`scorer_finetune.py:3027`), so the LoRA group's peak is exactly 2e-4 and P6's cosine shape is kept.
* Caveat (INHERITED, not verified): P6's `--lr` is read from the pod script's argv text, not from a run record of the final checkpoint (no `config.json` of the P6 run is in the repo mirror). If the pod run was launched with another `--lr`, the value moves.

## 2. Trace table: flag -> namespace -> model attribute -> forward -> loss -> backward (every hop `file:line`, tip + this build)

HB (`PK/refe/scorer_finetune.py` unless named):

| hop | what | file:line |
|---|---|---|
| flag | `--aux-drv-trunk-lora {off,on}` (default off), `--aux-drv-lora-lr` (default None) | `aux_hb.py:50-60` (`add_hb_args`), called `scorer_finetune.py:1700` |
| refusals | the lr without the flag, the flag without `--aux-drv` / without an lr / with `--aux-drv-feed on` / with a non-visual tap | `aux_hb.py:63-86`, called `scorer_finetune.py:1712-1713` |
| namespace -> build | `if a.aux_drv_trunk_lora == "on": hb = HB.HBTrunk(model).to(dev); optL = HB.build_lora_optimizer(hb, a.aux_drv_lora_lr)` | `scorer_finetune.py:1775-1777` |
| model attribute | `HBTrunk.backbone = copy.deepcopy(model.backbone)`, every base weight frozen, only `LoRALinear.A/B` trainable; a SIBLING (not a model attribute, not in `model.state_dict()`) | `aux_hb.py:115-134` |
| forward | `fwd()`: `actx = HB.select_actx(hb, model, img, cal, sctx, ...)` assigned BEFORE the TAP / G19 mutation lines | `scorer_finetune.py:1993` -> `aux_hb.py:166-171` -> `HBTrunk.ctx` `aux_hb.py:136-150` (the six lines of `model.py:743-759` on the copy: `reshape`, `backbone`, `+ _pos3d`, `scene_proj`) |
| head | `aux.predict(actx)`; the head's forward pre-hook records what it read | `scorer_finetune.py:1998` (`logit, sdf = aux.predict(actx)`), hook `:1774` |
| loss | `aux.loss(...)`, `aux.combine(0, L_aux)/div`, `tot.backward()` | `scorer_finetune.py:2130-2138` (`aux_backward`, unchanged) |
| backward reach | graph: L_aux -> head -> `scene_proj` (frozen, passes gradient) -> copy backbone LoRA `A/B`; stops there (P6's tensors are not in the graph; the scorer reads `sctx`, not the copy's ctx) | measured by G21 (section 3) |
| optimiser | own AdamW over the copy's LoRA tensors, own clip norm 1.0, lr = `lr_at(upd) * lora_lr/lr` | `scorer_finetune.py:3026-3027` -> `aux_hb.py:192-199` |
| save | `aux_drv.pt` format `aux_drv/2` = head + `hb_lora` + `opt` + `opt_lora` + update + meta; `scorer_states_last.pt` gains `HB` | `scorer_finetune.py:2372-2374` -> `aux_hb.py:202-206`; `:3073` |
| G12 tap | the tap row reads the COPY's ctx = `outs[2]` (scene_proj called 3 times); `scene_proj_calls == 3`, `tap_index == 2`, P6's `outs[1]` must NOT be what the head read | `scorer_finetune.py:2428` -> `aux_hb.py:174-189`; judge `gate_checks.py` `judge_g12` |
| G21 | in the preflight after G12 | `scorer_finetune.py:2711-2721` -> `aux_hb.g21_fields`, `two_update_probe`, `vdac_probes.judge_g21`; required iff `--aux-drv-trunk-lora on` (`gate_checks.scorer_required_gates`), re-judged in `launch_gate.consume_scorer_preflight` against the PROFILE's 3,145,728 |
| G-DVB | data event block `aux_hb` (BUILT facts) vs the argv: copy built, count == 3,145,728 (profile), all-LoRA, storage disjoint, not in `state_dict`, `p6_params_requires_grad == 0`, `lora_lr`, optimiser holds exactly the copy's tensors | `scorer_finetune.py:1959`; rows `gate_checks.py` `scorer_dvb_rows` (`aux_hb` block) |
| argv | `vdac_argv("HB-s0")` NAMES `--aux-drv-trunk-lora on --aux-drv-lora-lr 0.0002` (defaults `off` / None disable them) | `launch_gate.py` `vdac_argv`; `vdac_pod.sh` `HBX` |

HC:

| hop | what | file:line |
|---|---|---|
| flag | `--aux-drv-cell` (default 2.0), `--aux-drv-bw` (default 0.0), `--aux-drv-bw-band-m` (default 3.0) | `aux_hb.py:50-60` |
| namespace -> build | **the C1 fix**: `AuxDrivableConfig(..., cell=a.aux_drv_cell, bw=a.aux_drv_bw, bw_band_m=a.aux_drv_bw_band_m, ...)` (the tip built it WITHOUT `cell`) | `scorer_finetune.py:1770` |
| model attribute | `DrivableAuxHead(cell=...)`: 80 x 64 queries at 1.0 | `aux_drivable_head.py` (class unchanged; `AuxDrivableConfig` gains `bw`, `bw_band_m`) |
| loss | `AuxDrivable.loss` passes `bw`, `bw_band_m` to `aux_loss`; the Huber term x `(1 + bw)` on cells with `|block-mean SDF| < band`; `bw == 0.0` executes nothing extra | `aux_drivable_head.py:222-232`, `:330` |
| held-out read | `AuxEvalAcc(aux.cfg.cell)` reads at the head's own grid | `scorer_finetune.py:2357-2358` |
| G-DVB | geometry rows now bind the DECLARED cell (`cell`, `nx`, `ny`); with cell != 2.0 two more rows bind `n_params == 3,423,490` and the optimiser; `bw` rows read the multiplier the BUILT loss applies (`bw_probe`: inside / outside ratio through `aux.loss`) | `gate_checks.py` `scorer_dvb_rows`; `aux_drivable_head.py:360` (`bw_probe`) |

## 3. Tests (each file run, counts quoted from the run; CPU)

| file | result (CPU, `CUDA_VISIBLE_DEVICES=-1`) |
|---|---|
| **`refe/test_vdac_hb.py` (NEW)** | **`ZZTEST_PASS 133`**, 39 mutation arms printed, all RED. Module layer on the REAL tiny `REFe` + real head (T1-T12 of READINESS: step-0 identity vs model.py's own `visual_ctx`, live non-zero gradient on every copy LoRA tensor, exact-0.0 control, P6 untouched incl. state sha256 over 2 real updates, storage independence, no scorer path, G12 tap `outs[2]`, G19 on HB + the trunk-side mutant, `aux_drv/2` round trip + 2 resumed updates bit-identical + three strict-load refusals, post-gate Adam state of the LoRA tensors, cell 1.0 = 80 x 64 / 3,423,490, boundary weight 4 / 1, bw 0 bit-identical). Caller layer through the REAL `scorer_finetune.main()` (stub trunk with a real `LoRALinear`): preflight PASS with G21 judged, the copy moves, P6 and the scorer arms bit-identical to an HA run, `aux_drv/2` + post-gate on the real file, HC reaches the build, 5 runtime switches, 4 deleted wiring lines, `vdac_argv` HB / HC, the A5 layer peels to the a3cc6a77 blob |
| `refe/test_aux_s1_wiring.py` (2 lines changed: the tap-bypass mutation text) | `ZZTEST_PASS 20` (1 unmutated suite + 19 RED mutations); includes the default-OFF bit-identity of the real `main()` against the pre-edit trainer |
| `refe/test_vdac_gate.py` (7 lines: the arm set now allows the four A5 arms; the drift loop reads the A5-peeled trainer) | `ZZTEST_PASS` (V1-V11; V7 peel to the tip blob e8b86260 through the new outermost A5 layer) |
| `refe/test_vdac_stage0_forward.py` (unchanged) | `Ran 33 tests ... OK` |
| `refe/test_vdac_stage0.py` (unchanged) | `Ran 22 tests ... OK` |
| `refe/test_launch_gate.py` (1 line: `G21_` exempt like G19_/G20_ for arms that do not oblige it) | **NOT green here, environment**: it now runs past every assertion that failed before the fixes and stops at `test_launch_gate.py:2313` with `FileNotFoundError: D:\Projects\TanitAD\data\refe_consistent\targets_rank1.jsonl` (`diag_onpolicy_train.carve`); whether the same stop occurs on pristine `a3cc6a77` is UNVERIFIED (not run). The MM must run it where `D:\Projects\TanitAD\data` exists |
| `refe/test_aux_drivable_head.py` (unchanged) | 12 checks FAIL in this scratch tree (`StoreRefused: no manifest.json under ...label_store`, head-proto self-test file absent); **the pristine `a3cc6a77` export gives the same 12 FAIL (run)**: environment, not this build |
| other tests that peel trainer layers (DIV, S1ZSM, S1Z, SCORER1 gate tests) | NOT run. They peel through `vdac_layer.peel`, which now calls the A5 layer first, so the peel chain is intact (test_vdac_gate V7 asserts DIV's d665ed99 and S1ZSM's 6a16d35c still peel); their own "drifted trainer" loops that replace a layer's `new` text in the CURRENT trainer may need the same one-line `A5L.peel(TRAINER)` fix as test_vdac_gate. UNVERIFIED |


Mutation arms (every one RED; the full table is printed by `test_vdac_hb.py`): the select_actx wiring line DELETED; `HB.lora_step` line DELETED (control: the copy stays equal to P6's, so the "moved" check can read a stall); the build lines DELETED; cell / bw dropped from the build; the loss weight line dropped; five runtime switches
(`REFE_AUX_MUTATE_HB_P6CTX / _SHARE / _UNFREEZE_SP / _DETACH / _STALE_TAP`) each fail the REAL preflight and withhold `PREFLIGHT_PASS.json`; `REFE_AUX_MUTATE_EGO_TRUNK` turns G19 RED by gradient alone on the real tiny model (the trainer's G19 block runs only with `--aux-drv-exclude-logs`, whose fixture is test_vdac_gate's; the trainer line `actx = select_actx(...)` precedes the existing EGO mutation line, so both mutants land on the copy's output: UNVERIFIED end to end through the trainer's own G19, which needs the H48 list fixture).

## 4. Defect-class checklist (a)-(e)

(a) trace: section 2. (b) non-zero gradient on every copy LoRA tensor + exact-0.0 control (`lora_grad_w0.max_abs == 0.0`, `none == 0`) + P6 untouched (`.grad is None`, state sha256 equal over 2 real updates): `test_vdac_hb.py` module layer on the REAL tiny `REFe` and caller layer through the REAL `main()`. (c) mutation arms: above. (d) defaults: every new default disables its mechanism (asserted), `vdac_argv("HB-s0")` names both HB levers, `HC-s0` names the cell and the weight. (e) G-DVB entries for all five dests (`gate_checks.py` table), observers asserted by `built_without_observer`.

## 5. NOT built, and why

* **H5** micro-batch accumulation of the copy's backward: needs the pod's memory (ViT-L, 4 x 1,920 tokens x batch 8, copy backward beside P6's forward and three scorer copies is UNMEASURED); the preflight's `gpu_mem_gb` is the instrument. If batch 8 does not fit, the lever is `--accum` (the head's `aux_backward(lv, div)` already divides) or activation checkpointing in `HBTrunk.ctx`; not guessed here.
* **H10** the in-sample dump at the end of each arm (Stage 0's frames): not in the priority list given; `vdac_stage0_forward.py` is the HA producer.
* **The `aux_drv/2` loader for the tools** (Stage 0 forward, S2, G-EVAL): `aux_hb.load_v2` is the strict primitive (refuses a head-only file, a different LoRA key set, a missing optimiser state; round trip + 2 resumed updates bit-identical tested). `vdac_stage0_forward.load_head` still reads `aux_drv/1` only and refuses `/2`.
* **HC reader (C3)** and `s1_read_v2` HC grids: not built (READINESS section 3 C3).
* **The function fact** of the draft's section 2.1 item 4 (P6's forward reproduces `proposals_p6final.npz` bit-identically at training time): needs the banked file and the pod; G21 proves structure, gradient and state, and step-0 ctx identity, not this.
* **bf16 caveat (UNVERIFIED):** G21's `step0_ctx_equal` is bit-exact; on the pod the copy's forward runs with `requires_grad` and could pick a different SDPA kernel than P6's no-grad forward, reading a ~1e-3 difference that is a kernel choice, not a defect. The raw field `step0_ctx_max_abs_diff` is recorded; decide the bar on the pod rehearsal.

## 6. For the MM to decide / do

1. **Confirm 2e-4** (section 1) and freeze the prereg draft diff by sha256.
2. **Pinned files changed** (the launch gate pins them by sha256; all profile fingerprints move; run the ~69 launch_gate-reading tests): `refe/scorer_finetune.py`, `refe/aux_drivable_head.py`, `refe/gate_checks.py`, `refe/launch_gate.py`, `refe/vdac_probes.py`, `refe/vdac_postgate.py`, `refe/vdac_layer.py`, `refe/vdac_pod.sh`, `refe/vdac_chain.sh`, `refe/vdac_pause_p7.sh`, `refe/test_vdac_gate.py` (the arm-set assertion now allows the four A5 arms). **New files:** `refe/aux_hb.py`, `refe/vdac_a5_layer.py`, `refe/test_vdac_hb.py`. `model.py`, `train.py`, `ckpt_io.py` are untouched.
3. **SHIP_MANIFEST.json** and the chain's `NEED` list: `vdac_chain.sh` now names `aux_hb.py vdac_probes.py test_vdac_hb.py`; the manifest must list the three new files or the chain refuses.
4. **VDAC_SUITE** (`launch_gate.py`): add `test_vdac_hb.py` (`ZZTEST_PASS`) for the A5 arms if G-SUITE should run it; not done (it would change the HA arms' suite).
5. **AUX_HEAD_FLAGS / SFT4R_FORBIDDEN** (`launch_gate.py:71, 229`) were NOT extended with the five new flags: the HA / SFT-4R arms are pinned token for token, and the trainer refuses the new flags without `--aux-drv`.
6. **The pod pre-run** (`suite-prerun`): a test that passes here can fail on the pod; also run `test_vdac_hb.py` there once.
7. **Pause script stage names `A5HB` / `A5HC` and timeouts** (HB-s0 36,000 s, HB-SHUF 14,400 s, HC-s0 21,600 s) are set from the draft's cost UPPER bound (7.6 / 4.3 L40-h); replace from the preflight speed probe before freeze.
8. **Memory**: the first pod preflight of HB prints `gpu_mem_gb` and `s_per_microbatch`; send them to the PI before the cap is fixed (H5).
9. `test_aux_drivable_head.py` fails 12 checks in this scratch tree because the label store is absent here (`StoreRefused: no manifest.json under D:\...\label_store`, and the head-proto self-test file); the same 12 fail on the pristine `a3cc6a77` export (run, section 3).

## 7. Deliverable manifest

See the hand-back; every artifact is in the hold only (ONE place, plus the scratch which is deleted at the end). `_diffs/SHA256.txt` lists the sha256 of every file.

| artifact | where (all in the hold `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/vdac_a5_build4/` + `PK/refe/` unless noted; ONE place each; sha256 in `_diffs/SHA256.txt`) |
|---|---|
| NEW `aux_hb.py`, `vdac_a5_layer.py`, `test_vdac_hb.py` | `PK/refe/` |
| MODIFIED (full files) `scorer_finetune.py`, `aux_drivable_head.py`, `gate_checks.py`, `launch_gate.py`, `vdac_probes.py`, `vdac_postgate.py`, `vdac_layer.py`, `vdac_pod.sh`, `vdac_chain.sh`, `vdac_pause_p7.sh`, `test_vdac_gate.py`, `test_aux_s1_wiring.py`, `test_launch_gate.py` | `PK/refe/`; diffs vs `a3cc6a77` in `_diffs/<file>.diff` |
| DRAFT prereg copy (5e-5 -> 2e-4, flagged) | `PK/raw/2026-11-09-scorer1p6-prereg/vdac/a5_draft/PREREG_VDAC_P6_A5_DRAFT.md` |
| `BUILD4.md` | hold root |
`vdac_a5_layer.py` (generated from the trainer diff by the build; each reversal text occurs once) is why `vdac_layer.peel` still reproduces the pinned older trainer blobs.
