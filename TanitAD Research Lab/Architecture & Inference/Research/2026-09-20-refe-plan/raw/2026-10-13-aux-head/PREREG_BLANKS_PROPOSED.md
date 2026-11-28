# Auxiliary drivable-area head, S1: PROPOSED pre-registration blanks (not in force)

> **2026-10-13: APPROVED by the Master Mind and REGISTERED** through `eval/PREREG_AUX_DRIVABLE_S1.md` (§1–§4 and §6 unchanged; §5's open reads and §7's items are answered there). This file stays as the record of the proposal.

> ⛔ **PROPOSED. Not registered until the Master Mind approves this file and it is landed in git BEFORE the first S1 training step.** After landing, nothing below
> changes without a dated amendment written before the read. `eval/PREREG_AUX_DRIVABLE_DRAFT.md` is **not edited** by this file: this file fills, for host S1 only,
> the blanks that draft leaves open (its last table), and states what S1 can and cannot read.
> Written 2026-10-13. Wiring: `refe/scorer_finetune.py` (`--aux-drv`, default OFF). Gate: profile `refe_aux_s1` (`refe/launch_gate.py`, constants `AUXS1_*`; an amendment
> to a value here is a one-line change there, and the gate then refuses the old value).
> Evidence classes: **MEASURED** (ours, artifact named) · **INHERITED** (another doc, not re-verified) · **ANALYTIC** · **ESTIMATED** · **UNVERIFIED**.
> Rulings taken from `raw/2026-10-13-aux-head/WIRING_PLAN.md` "Master Mind rulings" (2026-10-13): separate head LR flag with a pre-registered value; u400 is a reference row
> only; no resume; no context cache; the shuffle + feed refusal stays.

## 1 · The two values this file proposes

| blank | proposed | rule that fixes it (written before any S1 read) | evidence |
|---|---|---|---|
| **head learning rate** `--aux-drv-lr` | **2e-4** (peak; it follows the scorer's schedule, warm-up 50 then cosine over 400 updates, scaled by 2e-4 / 1e-4 = 2, so `aux_lr` at update u is `lr_at(u) * 2`) | **Rule L.** The head is a freshly initialised 2,440,450-parameter module. Its rate is the rate at which the programme already trains freshly initialised parameters in a registered run: P6's AdamW 2e-4 (`eval/PREREG_P6.md:20`; `SPEC_P6_PAPER_FAITHFUL.md:116`), not a value searched on this rig. `scorer_finetune`'s own `--lr 1e-4` is a fine-tune rate for a pre-trained scorer and is not used for the head. **No S1 read selects it:** not PDMS, not the held-out selection metrics, not the head's IoU / SDF-MAE, not the `aux_bce` curve. If K2 or K6 fires the finding is "the head at this recipe does not fit", and a second rate is a dated amendment written before the next arm's first step, with both rates' results kept. | value INHERITED (P6 + WIRING_PLAN D-W1); optimiser `AdamW(wd 0.01)`, own clip norm 1.0, own optimiser: MEASURED in `scorer_finetune.py` and pinned by `test_aux_s1_wiring.py` (`h1/head_lr_follows_the_scorer_schedule_at_the_head_peak`) |
| **loss weight** `--w-aux-drv` | **0.1** | **Rule W.** 0.1 is `SPEC_AUX_DRIVABLE_HEAD.md` §2.3's initial value and equals `score_w`. The §2.3 gradient-norm balance (‖∇_LoRA L_aux‖ / ‖∇_LoRA (l_traj + l_score)‖, accept 0.05–0.5) is **undefined at S1**: the trunk is frozen, so no parameter receives both gradients. It is therefore **not measured here and 0.1 stands unmodified**; the band is read at S2, where it can be. Never tuned on PDMS. | value INHERITED (SPEC §2.3). **ANALYTIC:** the head has its own AdamW and its own clip, so a uniform rescale of its loss leaves the head's update unchanged to first order (Adam is scale-invariant up to ε = 1e-8; the clip rescales only when the head's gradient norm exceeds 1.0, UNVERIFIED whether it does on real data). So at S1 `w_aux` is close to inert for the head under test, and a sweep of it would be a sweep of nothing. |

Every other head lever is fixed by the profile and not proposed: `--aux-drv-tap visual` (the only tap this host builds; `scene` is refused), `--aux-drv-depth 2`,
`--aux-drv-feed off` (H1) / `on` (H2), no `--aux-drv-shuffle-seed` (§6).

## 2 · Arms, seeds, order

Init for every arm: **`model_sft4_B_u400.pt`** (md5 `8e382eeb…`, 1,288,489,838 B; `eval/RESULT_SFT4.md`), then **400 further updates** of the SFT-4 recipe (`--max-updates 400 --eval-every 400`).
⇒ **an arm's score minus u400's is NOT an attribution** (it mixes the head with 400 more scorer updates, and u400 stopped at update 400 of a 2,335-update schedule). u400 is a
**reference row only**; attribution is each head arm vs the **no-head replicate of the same seed**.

| arm | head | seeds | `--arm-name` | what it is for |
|---|---|---|---|---|
| **N** no-head replicate | none | 0, 1, 2 | `N-s0 N-s1 N-s2` | the comparator **and** the run-to-run floor (N vs N across seeds) |
| **H2** head + feed | `--aux-drv --aux-drv-feed on` | 0, 1, 2 | `H2-s0 H2-s1 H2-s2` | the lever under test; paired by seed against N |
| **H1** head, feed off | `--aux-drv` | 0, 1, 2 | `H1-s0 H1-s1 H1-s2` | perception proof without the feed, and the empirical form of "the head does not touch the scorer" (G12 (a) on CUDA) |

**Priority order** (a killed programme still yields value; one run at a time, sequential, on the shared dev-box GPU, each behind `wait_idle.py`):
1. `N-s0`, 2. `H2-s0` (first paired read), 3. `H1-s0` (CUDA identity of the scorer vs `N-s0`), 4. `N-s1`, 5. `H2-s1`, 6. `N-s2`, 7. `H2-s2`, 8. `H1-s1`, 9. `H1-s2`.
`H1-s1` / `H1-s2` are registered but last: if `H1-s0`'s scorer files are bit-identical to `N-s0`'s **and** its head fit equals `H2-s0`'s within the N-vs-N floor, the remaining two add
nothing to the S1 question and are dropped **by this stated rule**, not by convenience. (`H2`'s head is trained by the labels alone, its feed is detached, so the head fit and the perception proof are
read from every H2 run's `aux_eval` rows as well.)

**Cost per run: 3.9–5.1 h, ESTIMATED** (A40 11.4–14.4 s/update MEASURED/INHERITED × a 2.5× 4060 ratio ESTIMATED; `WIRING_PLAN.md` §7) with an **11.9 h ceiling** (the one MEASURED dev-box rate). Nine runs: **35–46 h
estimated, 107 h ceiling**; the six attribution runs alone are 23–31 h. **MEASURED 2026-10-13 on the 4060 after `wait_idle.py` exited 0 (`cuda_smoke_2026-10-13.json`, this directory): the head ALONE at the real context size** (`ctx` [B, 7680, 256] bf16, forward + aux loss + backward + AdamW step, 4 steps):
**B = 8: peak 0.527 GB over the resident tensors, 0.476 GB retained after the forward, 0.097 s/step; B = 4: peak 0.297 GB, 0.053 s/step; initial `aux_bce` 0.697 (ln 2 = 0.693).** The head is under K3's 1 GB at either micro-batch.
Per 64-set update that is ≈ 0.78–0.83 s, i.e. **2.1–2.9 % of the ESTIMATED 28–37 s/update, close to K3's 3 % bar** (the denominator is ESTIMATED, so this is not a pass).
**UNVERIFIED: that the 4060 (8 GB) fits ViT-L, 4 cameras, bf16, `--batch 8` with the head resident** (the smoke had no trunk). `--preflight` prints `gpu_mem_gb`;
G12 reports the head's retained activations on CUDA; a result over 1 GB (K3) or an OOM stops the programme and is reported, not worked around.

## 3 · Exact argv (the strings the gate token binds)

Variables (UNVERIFIED on the dev box until resolved at launch; the token fingerprints every data path, so a wrong path cannot silently pass):

```
$U400   model_sft4_B_u400.pt                       $D       the SFT-4 label root (train_teacher_lane, heldout_teacher_lane, heldout_lane, train_pdm, heldout_pdm)
$TR     refe_navtrain/train_grow (calib_table.json inside)   $PIX   navtrain_pixels    $ONP  refe_navtrain/onpolicy/sets
$HO     refe_heldout/sets                          $HOLOGS  refe_heldout/heldout_logs.txt         $OUT  the run's output root
LABELS  D:/Projects/TanitAD-artifacts/aux_drivable/label_store   (103,139 frames, store_sha256 4a17db67…f287a: pinned in the module, the gate and the trainer's data event)
```

`<S>` = the arm's seed (0, 1 or 2), `<ARM>` = the arm name. One line each; `--lr 1e-4 --b-mode expreward --lam 1.0 --tau-s 0.1 --lane-head --lane-weight 1.0 --workers 3` and the label flags are `refe/sft4_pod.sh`'s `COMMON` (INHERITED).
`--warmup 50 --batch 8 --accum 8` are the parser defaults and are **not named**; the gate checks the effective values (a changed default is refused).

```
# N-s<S>
python scorer_finetune.py --ckpt $U400 --targets $TR --images $PIX --calib $TR/calib_table.json --onpolicy $ONP --heldout $HO --heldout-logs $HOLOGS --workers 3 --lr 1e-4 --b-mode expreward --lam 1.0 --tau-s 0.1 --lane-head --lane-weight 1.0 --teacher-lane-train $D/train_teacher_lane --teacher-lane-heldout $D/heldout_teacher_lane --lane-labels-heldout $D/heldout_lane --pdm-labels-train $D/train_pdm --pdm-labels-heldout $D/heldout_pdm --require-pdm --max-updates 400 --eval-every 400 --dump-scores $OUT/<ARM>/dump --seed <S> --out $OUT/<ARM>
# H1-s<S>  = N-s<S> + 
        --aux-drv --w-aux-drv 0.1 --aux-drv-lr 2e-4 --aux-drv-labels D:/Projects/TanitAD-artifacts/aux_drivable/label_store
# H2-s<S>  = H1-s<S> + 
        --aux-drv-feed on
```

Properties the gate enforces on these strings (`launch_gate.py:auxs1_refusals`, tests `test_launch_gate.py` section A): the head flags `--aux-drv --w-aux-drv --aux-drv-lr --aux-drv-labels` are **named** on a head arm (a flag whose default
disables the head must be in the argv, or the arm did not test it) and **absent** from N; `--aux-drv-feed on` is named on H2 and absent on H1; the effective `w_aux 0.1`, `lr 2e-4`, `tap visual`, `depth 2` and the whole SFT-4
recipe equal the registered values on **every** arm; the declared arm name and the argv's class and seed agree; every EPO flag, `--preflight`, `--lane-labels-train`, `--b-compw`, `--tau-t` and `--aux-drv-shuffle-seed` are refused; flags are matched as exact tokens.
`--dump-scores` is OBSERVATIONAL (`test_dump_scores.py` pins `evaluate()` AST-identical to the pre-edit function); it is on **every arm, N included**, so the arms differ by the head alone. Its disk cost is UNVERIFIED for this rig.

**To mint one token per arm** (the profile refuses an unnamed arm; the token binds commit, argv sha256 and every data path's sha256, and the supervisor refuses to start the trainer without it):
`python refe/launch_gate.py run --profile refe_aux_s1 --arm-name H2-s0 --pkg <pkg> --args "$ARGS" --out-dir <dir> --gate-preflight-json <OUT>/preflight.jsonl` after `scorer_finetune.py … --preflight` has written its G1–G12 (G12 only on a head arm).

## 4 · Rules that travel with every S1 arm

1. **No resume.** `scorer_finetune.py` has none (`grep -ci resume` = 0). A killed run restarts from update 0 under a **fresh token**; its output directory is renamed `<arm>_killed_<k>` and kept, **never stitched**, and the restart is logged in the arm's manifest row. A run whose `start` event does not read `updates == 400` is not an S1 arm.
2. **No context cache.** `sctx` for 25,600 set-visits is ≈ 101 GB (ESTIMATED) against ~45 GB free on D: (`WIRING_PLAN.md` D-W4). Revisit only if a measured per-seed time exceeds the 11.9 h ceiling.
3. **Same data order within a seed.** N, H1 and H2 of one seed share `--seed` and therefore the loader's generator (`torch.Generator().manual_seed(a.seed)`); the head draws from its own RNG stream (tested), so adding it moves no scorer initial weight.
4. **A zero-weight / detached arm is an exact test, not a run** (G12: `w_aux = 0` reads every head gradient exactly 0.0; `autograd.grad(score, head)` is `None` for every parameter; feed on starts at the base scorer exactly). They are in the preflight of every head arm and have deliberate-regression arms (`REFE_AUX_MUTATE_TAP / _DROP_LOSS / _DETACH_HEAD`) that make G12 FAIL and withhold `PREFLIGHT_PASS.json`.
5. **Evidence tier.** Everything S1 reads is **T0-like** (a held-out selection / perception read on teacher-forced banked sets), never driving performance; no S1 number is quoted without that stamp (`EVAL_DOCTRINE.md`).

## 5 · What S1 can read today, and what it cannot (escalation)

| read | available at S1 now? | how |
|---|---|---|
| **head fit** on held-out logs: mask IoU, accuracy, SDF-MAE (m), per log and overall, at step 0 and update 400 | **yes** | the trainer's `aux_eval` rows (an independent recompute row for row is pinned by `test_aux_s1_wiring.py`) |
| **H2 effect on the scorer's selection** on the 3,137 held-out sets: picked / random / best PDMS, skill, zero-picks, DAC AUC, per rule | **yes, point estimates**; an interval needs the per-set logits | the trainer's `eval` rows; `--dump-scores` gives the per-set logits for a paired cluster bootstrap. **Which question the interval answers:** another draw of logs (not training: that is the N-vs-N floor; not inference: the scorer is deterministic, **asserted** only as a second pass over the same checkpoint) |
| the **seed floor** | **yes** | N-s0 / N-s1 / N-s2: the largest |Δ| between two N runs on each held-out metric |
| **P1 bands** ([0,20), [20,40), [40,72) m) and **P2/P3** (SDF MAE near the boundary; AUC of min predicted SDF along each proposal for DAC = 0 vs 1) | **NO: not produced by the trainer.** `aux_eval` is overall + per log, over the 40 × 32 extent | needs an **offline reader** over `aux_drv.pt` + the held-out sets (+ the C3 tokens for P3). **UNBUILT.** K2's 0.80 threshold is defined on P1's [0, 40) m band, so until the reader exists the trainer's overall IoU is reported as a **proxy** and K2 is not declared passed on it |
| **navtest PDMS (E1) and the DAC-zero share (E2) for H2** | **NO.** H2's feed lives in the scorer path of the *run*; the planner / harness do not apply it (`planner.py` strict-loads `model_final.pt`; the 3-scalar term needs the one `model.py` edit this feature ever needs, `WIRING_PLAN.md` §3 row 10) | **an integration decision for the Master Mind** (§7). Until then S1's H2 verdict is a held-out one only and may not be written as an ADOPT of the draft's rule |

## 6 · The shuffle control (not in this registration)

The Master Mind ruled the refusal of **shuffle + feed stays**. The draft's C-shuffle arm (a derangement of the label keys, seed **20261011**, 0 fixed points) is therefore **not an S1 arm here**, and the profile forbids
`--aux-drv-shuffle-seed`. Adding it needs a **dated amendment written before its first step** that (a) relaxes `AUX.aux_refusals`'s shuffle-with-feed refusal for exactly one named arm `H2-shuf`, (b) removes
`--aux-drv-shuffle-seed` from `AUXS1_FORBIDDEN` for that arm, and (c) fixes its seed. ⚠️ **Two seeds are in circulation:** the draft registers **20261011**; `WIRING_PLAN.md` §7's illustrative argv uses `7`. The pre-registered
one is **20261011**; `7` appears only in the unit tests. A shuffle arm *with the feed off* would test perception only (the scorer is bit-identical to N by construction) and is not proposed.

## 7 · Open items and who must act (Rule Zero: each names what would unblock it)

| item | blocks | unblock |
|---|---|---|
| the **offline reader** for P1 / P2 / P3 on `aux_drv.pt` | K2 and the perception gate on H2 | a 0-GPU implementation (head + held-out sets + navtest labels, `AUX.head_metrics` is the per-cell core); a work package for the Master Mind to issue |
| a **harness path that applies the feed** (planner / seam) | any navtest PDMS or DAC-zero read of H2 | a `model.py` + `planner.py` change (outside this wiring's scope, forbidden to this agent); or accept a held-out-only S1 verdict |
| **dev-box paths** for `$U400`, the on-policy banks, the held-out sets, the teacher-lane and PDM labels | every launch | a path audit (UNVERIFIED that 25,600 on-policy sets are on D:, `WIRING_PLAN.md` §8) |
| **GPU fit and speed** of the WHOLE run at `--batch 8` with the head (the head alone is measured, §2) | the cost table in §2 | the first `--preflight` (`gpu_mem_gb`); **CUDA bit-identity of the default-OFF path is MEASURED on a stub trunk only** (baseline twice equal, edited == pre-edit, a 1e-4 lr perturbation reads different; same json): not on the real ViT-L kernels, where **G12 (a)'s determinism claim stays UNVERIFIED** until a head-on preflight runs there |
| `rls1_pod.sh` line 108 ships a file list without `aux_drivable_head.py` / `field_loss.py` | any re-ship of the RL-S1 package | add them when RL-S1 is next re-shipped (the head flags are default-OFF, but `import aux_drivable_head` is unconditional in `scorer_finetune.py`) |

## 8 · Blanks of `PREREG_AUX_DRIVABLE_DRAFT.md`, for host S1

| draft blank | proposed value |
|---|---|
| label store: path, frame count, manifest sha256 | `D:/Projects/TanitAD-artifacts/aux_drivable/label_store/`; 103,139 frames; `store_sha256` `4a17db673b1548ba35724ab848a7dc551e4b0bc1e49c57ae93e6357454bf287a` (already in the draft; pinned in `aux_drivable_head.py`, `gate_checks.py` and the trainer's data event) |
| navtest label store | _ unchanged: needs the navtest build (draft)_ |
| head config and `w_aux` with its balance ratio | cell 2 m, 2 blocks, tap `visual`, feed off / on; **`w_aux` 0.1, balance not measurable at S1 (frozen trunk), read at S2** (§1) |
| comparator checkpoints and sha256 | `model_sft4_B_u400.pt` md5 `8e382eeb…` as the **init** and a reference row; the comparator of record is the same-seed N run |
| seeds for every arm; the derangement seed | **0, 1, 2** for N, H1, H2; derangement **20261011** (not an S1 arm, §6) |
| harness commit and the seam's flags | _ at landing_ |
| gate PASS token id | one per arm (`PASS_*.json`), minted by `launch_gate.py run --profile refe_aux_s1 --arm-name <ARM>` |
| S1 seed floor | read from N-s0 / N-s1 / N-s2; **if fewer than three N runs complete the verdict is capped at "separated, single seed, not yet a lever claim"** (draft rule 5) |
