# BRAKE LOCUS -- the re-roll that settles what the banked dumps cannot (design, NOT run)

Status: DESIGN ONLY.  Nothing here has been rolled.  CPU only, no GPU, no touching `devbox_gpu.lock`.  Written 2026-10-19 alongside `BRAKE_LOCUS.{json,md}`.

## 0. What it settles

| open question | needs | why banked data cannot |
|---|---|---|
| (ii)/(iii) of the brief: does the refined fan hold a good cruising candidate on false-brake windows; does selection pick a braking plan anyway (statistic E, `brake_locus_r8.fan_cells`) | the REFINED fan `[149,8,2]` per window | not banked (only terminal headings and the reach mask) |
| ALONG-RAND (coordinator add-on): along-track alignment r / g\* of a RANDOM and a MEDIAN-progress fan candidate vs the emitted plan, both inference seeds (`along_stats`, `along_r_diff_boot`, `along_verdict`) | the same fan | same |
| does the head CAUSE the pick? | an intervention | banked data are associations only |

## 1. Capture (a ~15-line addition to the roll stage of `code/force_lon_reroll_r8.py`, function `grab(o, forced)`, line ~352; the model output `o` already carries these keys, `refc.py` 3940-3985, 4009)

```python
fan = o["anchor_traj"][0].detach().float().cpu().numpy()                    # [149, 8, 2] refined fan, all model slots
r["fan_f16"] = fan.astype(np.float16).tolist()                              # S6 grid = slots [1,3,4,5,6,7]; S2 grid = slots [0,1,2,3]
r["sel_score"] = o["sel_score"][0].detach().float().cpu().numpy().tolist()  # core ranking score [149]
r["sel_score_v3"] = o["sel_score_v3"][0].detach().float().cpu().numpy().tolist()   # final blended ranking [149]
r["lon6_tag"] = o["r8_lon6"][0].cpu().numpy().tolist()                      # candidate tag in LON6 index space
r["kind"] = o["r8_kind"].cpu().numpy().tolist()                             # 0 base / 1 appended (allocated) / 2 prior-free
r["cons_raw"] = o["r8_cons_raw"][0].detach().float().cpu().numpy().tolist() # the head's CONSTRAINT output per candidate (NOT banked)
r["cons_valid"] = o["r8_cons_valid"][0].detach().float().cpu().numpy().tolist()
r["reach_keep"] = o["reach_keep"][0].float().cpu().numpy().tolist()
r["bank"] = o["anchor_bank"][0].detach().float().cpu().numpy().astype(np.float16).tolist()   # [117, 8, 2] unrefined base bank
r["prior_path"] = o["residual_prior_path"][0].detach().float().cpu().numpy().tolist()        # causal prior P for the along-track control
r["sel_idx_base"] = int(o["sel_idx_base"][0])
```
`json.dump` of `fan_f16` is ~4.8 KB / window as fp16 binary, ~25 KB as JSON text; 1,100 windows is tens of MB.  Save it as a sidecar `.npz` per chunk, not inside `roll_full.json`.

## 2. Arms

* **N** native, fan captured (the SAME call path that produced the banked `os`; per-window sampler seed identical).
* **F2** force CRUISE (`model.set_r8_force(lon6=1)`, existing in the roll) -- HEAD intervention incl. allocation.  This is the same arm as the force_lon design; reuse its outcome table, do not redefine it.
* **S0a** `r8_sel.gamma_prog := 0` (the head's lon CONSTRAINT term off), **S0b** all four `r8_sel` scalars `:= 0` (the whole head-coupled ranking term off); eval only, save and restore the parameters around the call (`torch.no_grad`).  **Why two arms, MEASURED 2026-10-19** (`BRAKE_LOCUS_ckpt_scalars.json`): the learned EMA-30k scalars are `beta_lat -0.0388, beta_lon -0.0068, gamma_prog +0.1010, gamma_head +0.3811`, i.e. the head's lon POSTERIOR hardly enters the ranking (beta_lon ~ 0); if the head reaches the pick through the ranking it is through `gamma_prog` (its progress constraint), which the argmax-based HEAD test cannot see.
* Optional **R7**: refcv7-50,400 fan on the same windows (`D:/refcv7_eval_kit/ckpt/ckpt_50400.pt`).  UNVERIFIED whether `force_lon_reroll_r8.py`'s loader (`RR.install_patches`, `refcv7_loader.build_model`) builds the refcv7 checkpoint unchanged; if not, drop R7 and use the CT windows of refcv8 itself as the E_ctrl reference.

## 3. Windows (one roll serves both questions)

* **Brake locus:** all seed-0 r8 false-brake windows (493 by the banked class; the roll re-derives the class from N) + a seeded random 250 GT-cruise non-false-brake control windows, S6 surface.
* **ALONG-RAND:** a stratified sample of S2 compliant windows, 5 per episode over the 138 episodes (about 700), unioned with the above (S6 windows are S2 windows).  The full 4,159 would cost ~11 h of N-only CPU; ~700 windows keep all 138 episode clusters (ESTIMATED CI half-width on r about 0.08 against the sibling's 0.05 at n 3,819; the sibling's r values are 0.40 emitted / 0.76 a_star-refined).
* **Interventions F2 / S0a / S0b:** the 250 false-brake windows of the force_lon plan plus 100 control windows (a subset; each extra forward is ~5 s).
* **Inference seeds:** the banked seed-0 sampler seeds for the false-brake analysis; ALONG-RAND needs BOTH inference seeds (the pre-registered rule is per seed): add the seed-1 N forward on the S2 sample (`window_seed(INFER_SEED + 1, ...)`, mapping of the banked `--infer-seed 1` to this offset UNVERIFIED), N arm only.

## 4. Cost and safety (from MEASURED numbers in `raw/force_lon/HANDOFF.md`)

* ~10 s per native forward incl. trunk, ~4.5-5 s per extra forced forward, 5 sequential child processes (one model load each), OMP 8.  Phase A (N, ~1,100 windows, seed 0 + the seed-1 N on ~700) about 4 h; phase B (F2, S0a, S0b on 350 windows) about 1.5 h.  Total about 5.5 h of CPU, ESTIMATED, resumable, detached.
* The dev-box COMMIT limit binds: free commit read 3.69-7.51 GB during this session; start a chunk only at >= 5 GB.  One model process at a time; do not run beside another roll (the force_lon roll's last banked line is 2026-10-18 19:37, 20 / 100 windows; whether it is alive or dead is UNVERIFIED here -- check before launching, never kill by name).
* HANDOFF per the token rule: the roll is long, so the launching agent writes a <= 2k-token `HANDOFF.md` and ends; a fresh agent runs the analysis.

## 5. Controls on the re-roll (read before any number)

* **R6 (re-stated: NOT a bit-equality control).**  The sampler is stochastic DDIM (`manifest.inference_seed`: `stochastic_at_inference` True) and the banked dumps were drawn on the GPU; a CPU re-roll with the same per-window seed uses a different RNG stream and fp32 numerics, so the N arm will NOT reproduce the banked `os` bit for bit (UNVERIFIED how closely; the force_lon roll has the same issue and its N2 arm measures the draw-to-draw floor).  The control is therefore distributional: the re-rolled N false-brake rate on the sampled GT-cruise windows must sit within the N2 floor of the banked 0.266, and the along-track r and g\* of the N arm must sit within that floor of the r and g\* of the BANKED `os` on the same windows.  The population for E is defined by the re-rolled N arm's OWN plan class; its overlap with the banked false-brake set is reported.  The full-set banked `os` g\* must still reproduce `GAINTRAJ.json` 0.4552 / 0.4616 (the sibling's own control, banked data only).
* a_star and `oracle_sel` from the re-rolled fan reproduce the banked `a_star` / `oracle_sel` (the GT-nearest rule is `refcv3_arm._oracle_anchor_index`, squared error over the 8 slots).
* `fan_cells` and `along_stats` known-value controls already pass (C6, C7 in `BRAKE_LOCUS.json`).
* Each intervention arm carries its own positive control (force_lon's `Fbrake` on control windows must CAUSE the defect) and the N2 sampler-noise floor.

## 6. Analysis and reading (all committed in `PREREG_BRAKE_LOCUS.md` + `AMENDMENT1`)

1. `brake_locus_r8.fan_cells(F, G, v0, fb, keep)`: **E** = share of false-brake windows with a good non-braking candidate (X = 2.91 m FDE at 6 s; sensitivity ADE 1.21 m; reach-surviving fan; sensitivities all 149 and base 117).  `E < 0.3` CANDIDATE, `E >= 0.5` SELECTION; HEAD test as in `BRAKE_LOCUS.json` (currently FAIL on clause 2); label per prereg section 5.
2. ALONG-RAND: `dR = r_os - r_RAND` per seed, paired episode-cluster bootstrap; SELECTOR-ADDS-VALUE iff the CI lower bound > 0 at both seeds, SELECTION-SUPPORTED iff not; MED beside it; the best-of-k curve (k in 1, 2, 4, 8, 16, 32, all) bounds how much of the a_star-refined r of 0.76 is the by-construction effect of picking by GT.  Stated in advance: os separated above RAND is the EXPECTED outcome and says only that the selector beats chance.
3. Intervention readout (to be sealed as its own prereg before the roll, in the force_lon style): closure of the false-brake rate by F2 vs S0a vs S0b, with the N2 floor.  Reading: F2 closes and S0a/S0b do not -> the head acts through allocation or other channels, not the ranking; S0a closes -> the progress-constraint term is the locus; neither closes -> the pick is made by the core ranking independent of the head's outputs (the association is then common-cause).

## 7. Blockers

None on compute or PI decision (CPU, ~5.5 h, no spend).  What is needed is the 15-line capture patch to `force_lon_reroll_r8.py` (a new file or an edit by its owner -- the force_lon agent is the natural owner), a launching agent, and the roll status of the existing force_lon job.
