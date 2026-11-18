# PREREG_V8F_DSCALE_MIX (DRAFT 2026-11-18, NOT FROZEN, NOTHING RUN): the mixture arm UM_s0, corpus-only tactical supervision on the union

Status: drafted by an agent for the MM. Not frozen. Nothing trained, launched, staged or landed. No pod command was run. This card amends `card/CARD_DATASCALE_TRAIN.md` (frozen 2026-11-16, called "the parent card" below). It adds ONE arm, UM_s0, at stage A.
Evidence classes: MEASURED (M, artifact named), PUBLISHED (P), INHERITED (I, another agent or doc, not re-verified here), ESTIMATED (E), HYPOTHESIS (H), UNVERIFIED (U).
Source reads: mirror `C:/Users/Admin/tanitad-push`, commit `cdbff971` (the DSCALE10K code of record: `read3/ckpts/UE_s0.A/BANK_RECORD.json` `commit`), exported with `git -c core.autocrlf=false show`. Line numbers are at `cdbff971`. `DS` = `F:/TanitAD-scratch/datascale`.

## 0. Escalations for the MM (each needs a ruling before freeze)

1. **The 10k clips are NOT "unlabelled for tactics".** Only two parts of T's supervision are absent on them: the 15 VLM-provenance goal bits and the lane-change constraint cells. T's lateral and longitudinal class losses, the 7 geometry goal bits and the lat / lon constraint cells are live on 10k rows (M, section 2). So the arm tests the **share of corpus rows in all of T's tactical supervision**, not only the VLM label share. The register row (section 1) is worded that way. The MM confirms the wording.
2. **No isolating rebalancing is reachable from the trainer's argv today** (section 3.1). The isolating form needs code in a shared library file, `stack/tanitad/models/v8f_mini.py` (`V8FMini.loss`), plus trainer wiring. The specification is in section 3.3 and nothing is applied.
   * Landing it moves the sha256 of `v8f_mini.py`. Behaviour is bit-for-bit unchanged when the new batch key is absent (section 3.4, identity 1), but **any study that pins this file's sha256 would see a change.** The DSCALE reads pin the read tree's copy (`read3/READ_PINS.json`: `388f453a...`). They read the `cdbff971` export, so they are unaffected.
   * Proposed: land the change as a child commit of `cdbff971`, used as UM_s0's code of record only. Read UM_s0 on the unchanged `cdbff971` read tree (the checkpoint's cfg is UE_s0's, section 3.3).
   * Before the land, check other sessions' bound records for a closure dependency on `stack/tanitad` (memory: a land that touches `stack/tanitad` between a record's bind and its chain restart invalidates that record).
3. **PI go for pod time.** About 4.4 pod-h GPU-holding (E, section 8), plus a G-SUITE run in the pod environment (CPU, 1 to 2 h wall, U). The parent card's go (Q1) priced blocks 0 to 3 only.
4. **Power is marginal, and the card says so up front.** The stage-A TAC-LAT floor is F_cell 0.0767. The replicate pair itself is separated: B_rep − B_s0 −0.0767 [−0.115, −0.041] (M, `read3/out/RESULT_DSCALE10K_B3.json` `pairs`). The block-3 cost UM is built to remove is −0.0855, which is 1.11 F_cell. So even a true dilution effect of that size clears the bar only narrowly. DILUTION-PARTIAL is a likely outcome under either hypothesis, and its committed next step is the replicate pair.
   * Option for the MM: ask the PI for UM_s0 and its replicate pair (UM_s1, UE_s1) together, about 12.6 pod-h (E).
   * Recommendation: UM_s0 first (priority order), with the replicates staged.
5. **Pod state is UNVERIFIED.** UM needs the union view `/root/data/cache_union_dscale/` and `/root/dscale10k/merged/*` (the merged release md5 `9881b170a889f743c5b0894f9ad13f1d`, lead, speed, tracks, `join_union.npz`). The parent card section 7 lists the union view for deletion after the reads. If it is gone, rebuilding it is cheap: symlinks, plus C-CACHE at 124 s (M, `gate/GATE_REPORT_BLOCK1.md` row C-CACHE).
6. **The floor is a 19,164-step, B-set floor applied at 61,568 steps**, as block 3 applied it (M, `RESULT_DSCALE10K_B3.json` `floors.definition`). No 61,568-step union floor exists. UM_rep (seed 2) would measure one (section 5.6).
7. **`launch_gate.py` has no v8f-mini profile** (M: at `cdbff971`, `PROFILES["v8f"]` at `:1043` is `_v8f_profile("pose")`, built at `:989`, the refcv8 profile; no line names `v8f_mini_train`). The G-DVB entry therefore lives in the trainer's `FLAG_CHECKS` and the study checker (section 7). The text of a `launch_gate.py` row is given there, unapplied.

## 1. Hypothesis and register row

**Question.** At 61,568 steps, does restoring the 4,512 corpus clips' share of T's tactical supervision to 100 %, with the union's batches, data order and every non-tactical loss term unchanged, remove the TAC-LAT cost that the 10k clips carry?

**What is MEASURED (stage A, T1-kin, eval-141: 141 episodes / 2,399 windows; paired episode-cluster bootstrap B 2,000 seed 0; ONE seed; F_cell from one replicate pair), from `read3/out/RESULT_DSCALE10K_B3.json`:**

| pair | TAC-LAT Δ [95 %] | TAC-LON Δ [95 %] | corpus-equivalent tactical steps (E) | corpus window visits, epochs (E) |
|---|---|---|---|---|
| U_s0 − B_s0 | −0.0871 [−0.1297, −0.0446] cost | −0.0013 [−0.0284, +0.0254] | 5,964 vs 19,164 | 1.25 vs 4.00 |
| UE_s0 − B_s0 | −0.0038 [−0.0425, +0.0362] none | +0.0592 [+0.0179, +0.1022] gain | 19,161 vs 19,164 | 4.00 vs 4.00 |
| BL_s0 − B_s0 | +0.0817 [+0.0467, +0.1175] gain | +0.0550 [+0.0229, +0.0875] gain | 61,568 vs 19,164 | 12.85 vs 4.00 |
| UE_s0 − BL_s0 | −0.0855 [−0.1216, −0.0492] cost | +0.0042 [−0.0342, +0.0433] | 19,161 vs 61,568 | 4.00 vs 12.85 |
| B_rep − B_s0 (the floor pair) | −0.0767 [−0.115, −0.041] | −0.0179 [−0.0451, +0.0075] | equal | equal |

* "Corpus-equivalent tactical steps" = steps × the corpus share of windows, 306,566 / 985,048 = 0.3112 (M counts, `gate/pod_out/probe/U_s0.A.json` `n_windows`). "Visits" = steps × 64 × that share / 306,566. Both are arithmetic, so E.
* **The pattern (H).** TAC-LAT rises with the corpus-equivalent count: 5,964 < 19,161 ≈ 19,164 < 61,568 gives −0.087 / −0.004 / 0 / +0.082. The 10k windows look worth about zero for eval-141 TAC-LAT. TAC-LON tracks total steps instead (UE ≈ BL), so for TAC-LON the 10k windows count like corpus windows.
* **Why UM is the discriminating arm.** In all four trained arms, the corpus **gradient share** and the corpus **visit count** move together, so the data cannot separate them. UM separates them:
  * UM gives the corpus rows 100 % of the T-block weight at 61,568 steps, which is BL's share;
  * it keeps UE's batches, so the corpus windows are still visited 4.0 times, which is UE's count.

**Prediction, written so it can be wrong (H).** UM − UE TAC-LAT lies in [+0.04, +0.10], with UM − BL TAC-LAT not separated worse. That reads DILUTION-CONFIRMED if Δ > 0.0767 and DILUTION-PARTIAL otherwise. Beside it, a TAC-LON price is likely (UM − UE TAC-LON ≤ 0), because the arm removes the 10k rows' longitudinal supervision. STRATEGIC stays at UE's level, because S's term is untouched. If TAC-LAT follows visits rather than share, UM ≈ UE and the label is NOT-DILUTION.

**Register row (proposed text for `GOALS_AND_CLAIMS.md`):**

| id | claim | status | next |
|---|---|---|---|
| **H-V8F-DSCALE-MIX-1** (2026-11-18, DRAFT) | The union's TAC-LAT cost (UE_s0 − BL_s0 −0.0855, stage A, T1-kin, one seed) is carried by the corpus rows' share of T's tactical supervision (T-block weight: goal + lat + lon + cons). Arm UM_s0 = UE_s0's argv, plus a per-source T-block row weight of 0 on the 10,000 new clips. Every non-tactical term, the batches and the data order equal UE_s0's. Claim: UM − UE TAC-LAT is a gain beyond F_cell, and UM − BL TAC-LAT is not separated worse. | **PRE-REGISTERED (DRAFT), NOT RUN.** Card `DS/card_mix/CARD_DSCALE_MIX.md`. Needs code (section 3.3), a gate PASS and the PI's go. | By outcome, section 5.4. A NOT-DILUTION moves the lever to corpus visits and content (X_s0, the sampler-weight arm). It never ends the stream. |

## 2. What tactical supervision the 10k clips carry (from source)

**The loss (T's block), `stack/tanitad/models/v8f_mini.py`:**
* `:1168-1170` goal: `w = lt["goal_w"] * lt["has"][:, None].float()`, then `T["goal"] = (bce * w).sum() / w.sum().clamp_min(1.0)`. This is a **per-element mean over labelled bits**. A bit with `goal_w` 0 carries no weight and is not counted in the denominator.
* `:1171-1172` with `:1270-1276` (`_partial_ce`): the lat / lon partial-label CE, a mean over rows with a non-empty allowed set and `has`.
* `:1173-1180` cons: Huber over finite cells. With `cons_absent` `mask` (UE's lever, M `summary.json` `levers.cons_absent`), a NaN cell is masked.
* `:1237` total: `cfg.w_tac * (T["goal"] + 0.5 * (T["lat"] + T["lon"]) + cfg.w_cons * T["cons"])`, which is one block next to S's `ckpt` and O's `ade` / `fde`.

**The join, `stack/tanitad/data/v8f_mini_data.py`:** `goal_y` / `goal_w` at `:229-230`, `lat_cls` / `lon_cls` / `lat_allowed` / `lon_allowed` at `:231-234` and the cons cells at `:235-236` are copied from the v9 release row wherever `inside` holds. `_lab_pack` at `:607-614` unpacks them per tick.

**So are the 10k clips' tactical goal bits masked, or present but empty?**
* **In T's loss: masked.**
  * The merged release carries `goal_w` = 0 on every VLM-class bit of every 10k row. C-MASK PASS: M, `gate/pod_out/gate/tokens/U_s0.A/checks.json` `C-MASK`.
  * On a 10k-only batch, `goal_vlm` reads value 0.0 and grad exactly 0.0. The corpus-batch control reads grad 115.59. (M, `gate/pod_out/probe/U_s0.A.json`.)
  * These are not imputed negatives.
* **In the GT code O is teacher-forced on, they are present as "unknown", not empty.**
  * `_code_gt` (`v8f_mini.py:894-896`): `g = y * w + 0.5 * (1 - w)`, so a masked bit reads 0.5. O reads it with probability p_gt (1.0 → 0.2) through `:1055-1060`.
  * This is the same value the corpus's own unlabelled VLM bits carry.
* **In the consistency penalties, not read at all.** `gt_constraints_t` (`:651-663`) reads `y * w` only at TURN_L, TURN_R and STOP_POINT, which are geometry bits.
* **The geometry bits, the lat / lon classes and the lat / lon cons cells are live on 10k rows.** On the 10k-only batch the gradients are `tac_lat` 1642.0, `tac_lon` 871.1, `goal_geom` 425.4, `cons_lat` 247.9 and `cons_lon` 261.3. `cons_lc` is exactly 0.0 on both sources. (M, `probe/U_s0.A.json`.)

**Labelled share of tactical-loss-bearing windows (M unless marked).** Source: `DS/card_mix/share_probe.py` on the corpus join `navtok/raw/join_train.npz` (md5 `9ebc7a48...` = the parent card's join), with GOAL_PROV from `v9_labels_train.manifest.json` (release md5 `83e30bd3...`). Output: `share_probe_out.json`.
* Controls: the probe reproduces B's `train_sha256` `37f4f9ee...` and `n_windows` 306,566, both equal to `BL_s0.A/summary.json`.
* The VLM bits are the 15 classes whose provenance contains `vlm`: YIELD, CORRIDOR_OFFSET, EVADE_IN_CORRIDOR, OVERTAKE_VEHICLE, MERGE, GAP_TARGET, REACT_ON_ONCOMING, TAKE_EXIT_L, TAKE_EXIT_R, the four TRAFFIC_LIGHT_REACT bits and LANE_CHANGE_L / R. The geometry bits are the other 7.

| term | B / BL (4,512 clips, 306,566 windows) | U / UE (14,512 clips, 985,048 windows) |
|---|---|---|
| windows with a VLM goal bit labelled | 93,458 = **30.49 %** | 93,458 = **9.49 %** (10k rows: 0, C-MASK) |
| VLM share of the goal term's denominator (labelled elements) | **44.28 %** | **≈ 19.8 % (E**: assumes the 10k rows carry geometry elements at the corpus rate, 5.53 per window; the 10k join is pod-only) |
| windows with a geometry goal bit / lat class / lon class / any cons cell | 98.47 % / 97.99 % / 98.75 % / 99.67 % | corpus rows as B; 10k rows live (G-LIVE above), per-row rate U |
| corpus rows' share of the T-block rows per batch | 100 % | **31.12 %** on average (E: the window share; the sampler draws clips uniformly in chunks of 48, `ChunkedSampler`, `v8f_mini_data.py:1064-1087`), about 19.9 of 64 rows |

## 3. Rebalancing: what is reachable, what is chosen, what it needs

### 3.1 Reachable from the argv at `cdbff971`: none in an isolating form

The parser has 114 flags, every one a literal `add_argument("--...")` call (M, grep count on `v8f_mini_train.py`). For each candidate:

| candidate | at `cdbff971` | verdict |
|---|---|---|
| sampler weight / per-source mixing ratio | `ChunkedSampler(windows, batch, seed, chunk, max_age_t, max_age_s)` (`v8f_mini_data.py:1069`): a uniform permutation of clips, `chunk` clips at a time (`:1078-1081`). It takes no weight argument. The trainer builds it at `v8f_mini_train.py:3320-3321`. No flag names a source, ratio or weight, and `FLAG_CHECKS` (`:70-143`) has no such entry. | **not reachable** |
| per-clip tactical loss weight | the T block (`v8f_mini.py:1167-1180`) uses masks only. The one row mask in the loss, `aux_ok` (`:1184`), is set only by the T2-bank swap (`--t2-bank-p` with a bank) and is applied to acc / pen / wm, not to the T block. | **not reachable** |
| `--w-tac` (`:311`) | scales the whole T block for both sources alike (`:1237`), so it moves the tactical-vs-other balance, not the source mix | not a rebalancing |
| `--data-frac`, `--max-clips` (`build_data` `:3098-3100`) | select clips, they do not mix sources | not a rebalancing |
| data-side mask: a merged release with every 10k tactical column masked (`goal_w` 0, `lat_allowed` = `lon_allowed` 0, cons NaN) through the existing `--v9-train` / `--v9-train-md5` / `--fit-join` | **reachable today**, and it zeroes the 10k rows in T's block exactly as λ = 0 would. But the same columns feed O's teacher-forced code (`_code_gt` `:894-902`: goal → 0.5, lat / lon → uniform, cons → 0) and the GT consistency penalties (`gt_constraints_t` `:651-663`: TURN / STOP bits, `v_hi`, `lon_d_reach`, `turn_t_*`, `lead_tg_min`) on 69 % of the windows. Two non-tactical paths would move with the lever. | **rejected**: not isolating |

### 3.2 The choice: a per-source T-block row weight, λ = 0 on the 10k rows

**Lever.** Every row of a batch carries a weight ω = 1 if its clip is one of the 4,512 B-set clips, and ω = λ otherwise. ω multiplies the three T-block masks (goal, lat / lon, cons) and their normalisers. **Dose λ = 0.0**, set by the hypothesis rather than tuned: the corpus share of the T-block weight equals B's 100 %. Then T's tactical terms are the means over the corpus rows of each batch, with the same expectation per step as BL's.

**Why this one (two sentences).** It is the only form that leaves the batches, the data order (same seed, same sampler), S's term, O's trajectory, acceleration and teacher-forced code, the consistency penalties and the world model identical to UE_s0's. So UM − UE moves the source of T's tactical supervision alone, and UM − BL moves the source of everything else alone. Batch-balanced sampling would change the data seen by every term, and the data-side mask would change O's input, so either would make the contrast non-attributable.

**What it does not do (named, not hidden):**
* It restores the corpus **share**, not the corpus **visits** (4.0 epochs, as UE).
* The T-block mean averages about 19.9 corpus rows per batch instead of 64. Its per-step gradient variance is therefore about 3.2x BL's (E).
* It also removes the 10k rows' lateral and longitudinal class supervision, which UE − B_s0 suggests helps TAC-LON (+0.0592, M).

### 3.3 Minimal implementation (NOT applied; the MM lands it)

**The one change, `stack/tanitad/models/v8f_mini.py`, `V8FMini.loss` (`:1167-1180`) and `_partial_ce` (`:1270-1276`).** A new optional batch key `tac_w` ([B] float32) multiplies T's three masks. An ABSENT key runs the original lines, which keeps the tip bit for bit.

```diff
         # T: masked BCE / partial-label CE / Huber at T's tick frame
+        rw = batch.get("tac_w")                  # PREREG_V8F_DSCALE_MIX: [B] row weight on the T block ONLY; ABSENT = the tip, bit for bit
         w = lt["goal_w"] * lt["has"][:, None].float()
+        if rw is not None:
+            w = w * rw[:, None]
         bce = F.binary_cross_entropy_with_logits(out["goal_logits"], lt["goal_y"], reduction="none")
         T["goal"] = (bce * w).sum() / w.sum().clamp_min(1.0)
-        T["lat"] = self._partial_ce(out["lat_logits"], lt["lat_al"], lt["has"])
-        T["lon"] = self._partial_ce(out["lon_logits"], lt["lon_al"], lt["has"])
+        T["lat"] = self._partial_ce(out["lat_logits"], lt["lat_al"], lt["has"], rw)
+        T["lon"] = self._partial_ce(out["lon_logits"], lt["lon_al"], lt["has"], rw)
         ...
         hub = F.huber_loss(out["cons_std"], torch.nan_to_num(tgt), reduction="none", delta=1.0)
-        T["cons"] = (hub * m).sum() / m.sum().clamp_min(1.0)
+        if rw is None:
+            T["cons"] = (hub * m).sum() / m.sum().clamp_min(1.0)
+        else:
+            mf = m.float() * rw[:, None]
+            T["cons"] = (hub * mf).sum() / mf.sum().clamp_min(1.0)
 ...
-    def _partial_ce(logits, allowed, has):
+    def _partial_ce(logits, allowed, has, rw=None):
         ...
         l = torch.where(valid, lse - lsa, torch.zeros_like(lse))
-        return l.sum() / valid.float().sum().clamp_min(1.0)
+        if rw is None:
+            return l.sum() / valid.float().sum().clamp_min(1.0)
+        return (l * rw).sum() / (valid.float() * rw).sum().clamp_min(1.0)
```

* Nothing else reads `tac_w`: not S's `ckpt`, `_code_gt`, `gt_constraints_t`, `traj_terms`, `accel_term`, the pens or the world model.
* There is **no new config field**. `V8FMiniConfig` is unchanged, so G-HYG is untouched, the checkpoint's cfg equals UE_s0's, and the `cdbff971` eval loader reads UM unchanged.
* `clamp_min(1.0)` on a weighted count: at λ = 0 a batch with no corpus row gives a T block of exactly 0 (E: under 1 such batch in 61,568 steps). The count is logged as `tac_src_empty_batches`.

**Wiring, `stack/scripts/v8f_mini_train.py`:**

| hop | where (lines at `cdbff971`) | what |
|---|---|---|
| argparse | after `--w-tac` (`:311`) | `--tac-src-w` (float, default **1.0** = off), `--tac-src-clips` (path, default "" = off): the clips whose rows keep weight 1, one id per line. `--tac-src-clips-sha256` (default ""). |
| G-DVB table | `FLAG_CHECKS` (`:70-143`); `declared_vs_built` refuses a flag without an entry (`:1657-1660`) | `"tac_src_w": "data:tac_src", "tac_src_clips": "data:tac_src", "tac_src_clips_sha256": "data:tac_src"` |
| build | `main`, after `make_dagger` (`:3304`) | `src = make_tac_src(a, J, clips, wins, info)`: the per-clip-index bool `full_ci`, λ, the list's file sha256, n listed clips in train (4,512), n others (10,000), n listed windows (306,566), n other windows (678,482), all written to `info["tac_src"]`. **Refused (SystemExit `ZZDVB-REFUSEDZZ`):** λ not finite or outside [0, 1]; λ ≠ 1 with no list; a list with λ = 1 (a lever declared that does nothing); file sha256 ≠ declared; a listed clip not in the train clips; a list holding none or all of the train clips. |
| G-DVB clause | `bad_tac_src(a, src, data_info)` inside `declared_vs_built` (beside `:1654-1656`) | declared λ, list and sha equal the BUILT stage's values |
| stage | the loop's label-stage site (`:3558`, before `ego.apply` / `dag.apply`), attached after the stages (beside `:3591-3592`) | `tw = tac_src_stage(b, src)`: `torch.where(src.full_ci[b["ci"]], 1.0, λ)`, from the collated `ci` (`v8f_mini_data.py:737-738`). Then `b["tac_w"] = tw`. Loop guard: with λ ≠ 1, a batch without `tac_w` raises `ZZTACSRC-NOT-APPLIEDZZ`. Counts: rows listed / other, the realised weight share on listed rows, empty batches. |
| loss → backward | `model.loss(b, ...)` `:3619` → `v8f_mini.py:1167-1180, :1237` → `T["total"].backward()` `:3624` | the weighted T block in the total |
| G-LIVE | `tac_src_probe(model, probe, cfg, src)` beside `acc_probe` (`:3380`), on the probe batch with `tac_w` attached as at `:3337-3340`. Its record goes to `selfcheck.json`. | section 3.4 |
| run record | `summary.json` (beside `:3676`) | `tac_src_w`, `tac_src_list_sha256`, `tac_src_rows_listed`, `tac_src_rows_other`, `tac_src_weight_share_listed` (= exactly 1.0 at λ = 0), `tac_src_empty_batches` |

New tests: `stack/tests/test_v8f_mini_tac_src.py`. The existing `FLAG_CHECKS` completeness test in `test_v8f_mini.py` covers the three new dests.

### 3.4 Reachability proof (G-LIVE `tac_src_probe`; committed values; each check has a mutation that must turn it RED)

The probe batch's rows are split by source. If a source is missing from the first batch, a 64-window batch is drawn from that source's windows with a fixed generator. Each check below names its committed value, then the mutation that must turn it RED.

1. **Upper identity.** On the mixed batch, `tac_w` all ones reproduces `tac_w` ABSENT bit for bit: every term, the total, and every parameter gradient, with difference exactly 0.0.
   * RED: replace `w * rw[:, None]` by `w * (rw[:, None] + 1e-3)`.
2. **Lower identity (λ = 0).**
   * On a 10k-only batch, the T block `w_tac * (goal + 0.5 * (lat + lon) + w_cons * cons)` reads exactly 0.0. Its backward alone gives exactly 0.0 on every parameter, including `h_goal`, `h_lat`, `h_lon`, `h_cons`, `trunk_t` and `read_t`.
   * **Control:** the same batch with `tac_w` absent gives > 0 on all four heads.
   * RED: delete the `rw` argument at the lat call. `h_lat` then reads > 0. The same holds for goal and cons.
3. **Partial, not a switch.** At λ = 0.5, `T["lat"]` equals the analytic weighted mean (Σ_listed l + 0.5 Σ_other l) / (n_listed + 0.5 n_other) to rtol 1e-6. It differs from both its λ = 0 and its λ = 1 value.
   * RED: `rw = (rw > 0.5).float()`.
4. **Non-tactical invariance.** On the mixed batch with the same generator, every non-T term (`ckpt`, `ade`, `fde`, `acc`, `pen_*`, `pens_*`, `wm`) is bit-equal at λ = 0 and λ = 1. The gradient of their sum is bit-equal on every parameter.
   * RED: multiply `rw` into S's `ok` mask (`:1163`). The difference then reads ≠ 0.
5. **Corpus rows untouched.** On a corpus-only batch, λ = 0 and ABSENT are bit-identical (every ω is 1).
   * RED: build the weight as `torch.where(full, λ, 1.0)`, which inverts the source.
6. **Loop reach.** A 200-step union smoke must log `tac_src_rows_other` > 0, `tac_src_weight_share_listed` = 1.0 and `tac_src_w` = 0.0.
   * RED: delete the attach line `b["tac_w"] = tw`. The loop guard must exit.
7. **Default off.** An argv without the three flags builds no stage, so `tac_w` is absent. UE_s0's dry build then gives the same `selfcheck.json` term values as at `cdbff971`.
   * RED: set the default of `--tac-src-w` to 0.0. The trainer must refuse: λ ≠ 1 with no list.

## 4. Arm UM_s0

**argv = UE_s0's argv, plus the rebalancing only.**
* UE_s0.A = U_s0.A (`gate/pod_out/argv/U_s0.A.json`) with `--steps 61568` and `--out /root/dscale10k/runs/UE_s0/A` (rule `gate/block3/block3_files/gen_argv_b3.py`).
* UE's `argv_sha` is `23ba3ecdf581a4e7`, `steps` 61,568, `S` 15,392, `n_clips` 14,512, `n_windows` 985,048 and `train_sha256` `9844151f...` (M, `read3/ckpts/UE_s0.A/summary.json`).

```
UE_s0.A: --arm M2b --seed 0 --grid pool4x10 --out /root/dscale10k/runs/UE_s0/A --cache-dir /root/data/cache_union_dscale
         --v9-train /root/dscale10k/merged/v9_labels_union.npz --v9-train-md5 9881b170a889f743c5b0894f9ad13f1d
         --lead-train /root/dscale10k/merged/lead_union.npz --speed-train /root/dscale10k/merged/speed_union.jsonl
         --join-train /root/dscale10k/merged/join_union.npz --p10 /root/strat1/inputs/p10/p10_prior.pt --kin-mode yaw
         --tracks /root/dscale10k/merged/native_tracks_union.pt --batch 64 --steps-mult 4 --device cuda --fit-join --steps 61568
UM_s0.A: the same tokens, except
         --out /root/dscale10k/runs/UM_s0/A
         + --tac-src-w 0.0 --tac-src-clips /root/dscale10k/mix/bset_clips.txt
           --tac-src-clips-sha256 37f4f9ee468e0872134074a82d63c2915458596fdbcf3d0fcf5b09f1fe0d8cc1
```

* `bset_clips.txt` holds the 4,512 B-set ids in sha12 order, newline-joined, with no trailing newline. Its byte sha256 is then B_s0's MEASURED `train_sha256` `37f4f9ee...` (the same join rule as `build_data` `:3110`, M via `share_probe_out.json`). It is written from B_s0's dry record clip list.
* **Argv refusal (rc 2).** UM's argv may differ from UE_s0.A's only in `--out` and the addition of the three `--tac-src-*` flags with exactly these values. Any other difference is refused. Deliberate-regression arms, each must read REFUSED:
  * G-RM1: `--steps 61569`;
  * G-RM2: `--seed 1`;
  * G-RM3: `--tac-src-w 0.5` (the dose is pre-registered);
  * G-RM4: a list with one clip removed (sha differs);
  * G-RM5: `--tac-src-w 0.0` without a list (the trainer's own refusal);
  * G-RM6: `--tac-src-w 1.0` with the list (declared, does nothing);
  * G-RM7: an added `--exclude 540:600` (parent C-EXCL).
* Stage A only, like UE_s0 and BL_s0. Seed 0. LoRA pod, one GPU job at a time, under `flock /root/gpu_queue.lock`.
* **`--fit-join` rewrites `join_union.npz`** from the merged release. This is deterministic and gives UE's bytes. The checker hashes the file before and after the run; if they differ, the arm is BLOCKED.

## 5. Contrasts, estimator, outcomes (committed before any number)

### 5.1 Pairs (stage A, the same 2,399 windows / 141 episodes)
* **Deciding: UM_s0 − UE_s0.** The rebalancing at matched data, steps, batches and non-tactical terms.
* **Deciding: UM_s0 − BL_s0.** Is the tactical cost against BL gone? Here only the non-tactical data source differs, plus the visit count.
* Beside: UM_s0 − B_s0 and UM_s0 − U_s0.
* Reproduced as controls (K-RE): UE − BL, U − B_s0 and BL − B_s0 from block 3.

### 5.2 Tiers, estimator, floor, seeds
* **Tiers.** LONGITUDINAL / LATERAL / TACTICAL use **T1-kin** (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene; deterministic planner). STRATEGIC uses **T0 at the window's first tick** (real / donor nav on the read tree; zero / realx / donorx on the STRAT1 tree), as block 3. T's latcons and T's own head accuracies are T0, beside: a diagnostic, never driving performance.
* **Estimator.** `taniteval.ci.paired_episode_cluster_bootstrap`, B 2,000, seed 0. It answers "another draw of EPISODES" only.
  * Training-run variance enters only through F_cell. Inference variance: not applicable, the planner is deterministic (one dump environment, one thread count).
* **F_cell** = |B_rep − B_s0| per cell at stage A, ONE replicate pair (B_rep seed 2), the parent card's rule (M, `RESULT_DSCALE10K_B3.json` `floors.stage_A`):

| headline cell | F_cell |
|---|---|
| T1 TAC-LAT | 0.0767 |
| T2 TAC-LON | 0.0179 |
| L1 op speed MAE | 0.0496 m/s |
| L2 op headway abs error | 0.0466 m |
| A1 op yaw-rate MAE | 0.3804 deg/s |
| A2 op curvature MAE | 0.0077 1/m |
| S1 pos_err@80 | 0.2070 m |
| S2 skill@80 | 0.0164 |

* **ONE seed.** No label of this card is quotable as a lever effect until the replicate in 5.6 exists (`H-ESTIM-SEED-1`).

### 5.3 Cell and family rule (the parent card's, unchanged)
* Δ = UM − X per cell.
* **Gain:** separated on the better side and |Δ| > F_cell. **Cost:** separated on the worse side and |Δ| > F_cell.
* **Family gain:** at least one headline gain and no headline cost. **Family cost:** at least one headline cost.
* All four families are reported for every pair, per band (operative knots 1-10, tactical knots 11-30), never pooled. ADE and FDE are beside and decide nothing alone.
* **STRATEGIC is reported in full** (PI 2026-11-16: the strategic pause has ended), with its members not in this card named as the parent card names them:
  * detected information n/a (no ctx head);
  * derived goals not in this arm;
  * strategic intent 6-30 s not computable.

### 5.4 The deciding label (TAC-LAT, T1-kin, stage A) and what each outcome makes the next lever

**Precedence:** BLOCKED (a gate FAIL, a refused argv, K-TACSRC failing, the post-training Adam-state count ≠ 252) > CONTROL-FAILED (K1, K-RE, K-EVALX; "not quotable") > NOT-YET (an item unread) > the label. The four labels cover every case: separated better, not separated, separated worse.

| label | rule (TAC-LAT is accuracy: better = higher) | what it says | next lever (Rule Zero) |
|---|---|---|---|
| **DILUTION-CONFIRMED** | UM − UE TAC-LAT is a **gain** (separated better AND Δ > 0.0767) **AND** UM − BL TAC-LAT is **not separated worse** (its interval's upper end ≥ 0) | the cost travels with the corpus share of T's tactical supervision, at one seed | 1. The replicate pair UM_s1 + UE_s1 (and BL_s1 for the UM − BL side) before the register row is quoted as a lever effect. 2. UM through stage B (the OBD-CTL fine-tune, 3,000 steps, its own DAgger bank): the deployable read. 3. Split the share: a lateral-only weight (λ on `lat` and the four lateral geometry bits) against a VLM-only renormalisation of the goal term. 4. The VLM-CoT pass on the 10k (PI) as the label-side fix. If STRAT-KEPT also holds, UM is the candidate base of the next data-scale rung. |
| **DILUTION-PARTIAL** | UM − UE TAC-LAT separated better, and not DILUTION-CONFIRMED (Δ ≤ 0.0767, or UM − BL still separated worse) | the share moves TAC-LAT, but inside the one-pair floor, or not all the way to BL | 1. The replicate pair first: about 1.1 F is the expected size even under dilution (section 0 item 4). 2. If UM − BL is still separated worse: the **visit** form. That needs the sampler-weight arm: code in `ChunkedSampler`, a per-source draw rate, its G-DVB entry, an amendment. |
| **NOT-DILUTION** | UM − UE TAC-LAT not separated | the T-loss share is eliminated as the carrier of the cost at one seed. Remaining: corpus visits, and content through the non-tactical paths (O's trajectory on 10k scenes, O's teacher-forced code with VLM bits at 0.5 on 90.5 % of windows, the shared adapter) | 1. X_s0 (built and staged, waiting on the PI's go): does 10k content alone cost TAC-LAT? 2. The sampler-weight arm (visits). 3. The per-country read of UM (Spain and Sweden cost TAC-LAT at both seeds in block 2, M, `diag/RESULT_DIAG.md`). |
| **REBALANCE-HURTS** | UM − UE TAC-LAT separated worse | the 10k rows' tactical labels help TAC-LAT. This is consistent with R_s0 − U_s0 TAC-LAT −0.0825 [−0.128, −0.034] (M, `read/out/RESULT_DSCALE10K_B1.json`): deranged 10k labels cost TAC-LAT. The share form is refuted at one seed. | 1. The VLM-CoT pass on the 10k: fill the masked bits instead of removing the live ones. 2. X_s0. 3. The replicate of UM. |

**Beside labels** (they never change the deciding label, and each is reported):
* **STRAT-KEPT:** UM − UE STRATEGIC is not a family cost AND UM − BL STRATEGIC is a family gain. UE − BL read S1 −0.7246 [−1.3748, −0.147] and S2 +0.1189 [+0.0513, +0.2016], both gains (M, B3 `pairs`).
* **TAC-LON-PRICE:** UM − UE TAC-LON is a cost. Next: the lateral-only weight (item 3 of DILUTION-CONFIRMED) moves ahead of the VLM-only split.
* **Other families:** a LONGITUDINAL or LATERAL family cost in UM − UE is a price of the rebalancing. It is named in the result, with the cell.

### 5.5 Reach precondition and controls of the read
* **K-TACSRC** (from UM's `summary.json`): `tac_src_w` 0.0, `tac_src_list_sha256` `37f4f9ee...`, `tac_src_weight_share_listed` exactly 1.0, `tac_src_rows_other` > 0 (E: about 61,568 x 64 x 0.689 = 2.71 M), `tac_src_empty_batches` ≤ 5 (E: under 1). Any miss is BLOCKED (reach).
* **Post-training gate:** every registered parameter has Adam state, 252 = B_s0's count (I, `gate/GATE_REPORT_BLOCK1.md` section 5). There are no new parameters.
* **K1:** UM vs itself reads 0.0 [0, 0].
* **K-RE (B3):** the block-3 rows for UE − BL, U − B_s0 and BL − B_s0 (the 216 T1 rows, the ADE6 means and the STRATEGIC rows) reproduce `RESULT_DSCALE10K_B3.json` exactly.
* **K-EVALX** at stage A for UM, as block 3.
* **C-INIT** (gate): UM's step-0 state-dict sha256 equals UE's and U_s0's. The gate measured `0fdf9534...` for B_s0 = U_s0 (I, GATE_REPORT row G-HYG / C-INIT). The new stage draws no random number.
* **C-ORDER** (gate): the sha256 of the first 1,000 epoch-0 sampler batches `(ci, t, a_t, a_s)` of UM's dry build equals UE's. RED: the same with `--seed 1`.

### 5.6 The replicate this needs (ONE seed now)
* For UM − UE: **UM_s1 and UE_s1** (seed 1), about 2 x 4.1 pod-h (E).
* For UM − BL: **BL_s1**, about 3.9 pod-h (E, from BL's MEASURED 12,827 s loop time).
* For a 61,568-step union floor: **UM_rep** (seed 2), about 4.1 pod-h (E).
* None of these is launched by this card. Each would come in a short amendment with the PI's go.

## 6. Read plan (dev-box CPU; reuse `DS/read3/`)

Copy `read3/` to `DS/read_mix/`. What changes:

| file | change |
|---|---|
| `ds_lib.py` | `ARMS_MIX = ("UM_s0",)`; `STEPS_ARM["UM_s0"] = 61568`; `ck_root`: UM reads `read_mix/ckpts`, UE / BL `read3/ckpts`, the block-1 arms `read/ckpts`; `DS_BLOCK = "mix"` items; `bank_problem` wants 61,568 for UM |
| `ds_pull.py` | the bank directory `read_mix/banker`, the HF path `UM_s0/A/ckpt_last.pt` (private repo of the other arms), `--min-free` 25 GB |
| `ds_run.py` | one dump `DS-UM_s0-A_m10_causal_heads`, UM's T0 mechanism evals, five stage-A route passes for UM, K-EVALX for UM; the read tree is the `cdbff971` export (pins `READ_PINS.json` unchanged) |
| `ds_read_mix.py` (from `ds_read_b3.py`) | `PAIRS` = UM − UE, UM − BL (deciding), UM − B_s0, UM − U_s0 (beside); `outcome_mix` (5.4, with precedence); the beside labels; K-TACSRC, K1, K-RE (B3), K-EVALX; the four families per pair |
| `diag/diag_read.py` | re-pointed at the stage-A dumps of UM, UE and BL: per-goal-class F1, VLM vs geometry bits (beside; power-limited, as `RESULT_DIAG.md` says) |
| tests | `test_ds_read_mix.py`: each label, both signs, Δ exactly at F_cell, the UM − BL upper end exactly at 0, every precedence; `mutate_check_mix.py` (pattern `mutate_check_b3.py`), including swapping the UE and BL roles and dropping the UM − BL condition |
| smoke | stand-ins with known answers. **UM := BL's dumps** must read DILUTION-CONFIRMED, with UM − UE = BL − UE = +0.0855 and UM − BL = 0.0 [0, 0]. **UM := UE's dumps** must read NOT-DILUTION, with UM − UE 0.0 [0, 0]. Both outcomes are exercised before the real read. |

* Cost (E, from block 3's MEASURED stage timings in `read3/BUILD_READ3.md` section 5): 1 dump about 6 min, the mechanism evals about 4 min, 5 route passes 3 to 8 min, the reader about 6 min. **About 30 to 45 min of dev-box CPU.** Under 1 GB of disk. The F: bar is 25 GB. No pod contact.

## 7. Launch-gate plan

**Gate route.** A block-mix copy of the study checker `dscale10k_check.py` mints `dscale10k-gate-token-v1`. It binds:
* the child commit of `cdbff971` carrying section 3.3;
* the argv sha256;
* the tree manifest;
* the trainer, model and selfcheck sha256;
* every data sha256 / md5 of UE_s0's token, plus `bset_clips.txt`.

The runner exits 2 without a matching token.

| check | in this card |
|---|---|
| **G-DVB** | The trainer's `FLAG_CHECKS` entries and `bad_tac_src` (3.3). The checker adds C-ARGV-MIX (the argv rule of section 4 and G-RM1 to G-RM7), C-TACSRC-LIST (the list parses to 4,512 ids, all in U's train clips, none of the 10,000; its sorted join sha256 equals `37f4f9ee...`), C-STEPS (61,568 = 4 x S_U), C-CHUNK (48), C-ORDER and C-INIT. **The row for `launch_gate.py`, if a v8f-mini profile is ever added (NOT applied):** `--tac-src-w` → `data:tac_src.w` (the BUILT stage's λ, float-equal); `--tac-src-clips` → `data:tac_src.list_sha256` (the file the trainer hashed) and `data:tac_src.n_listed` 4,512 / `n_other` 10,000; `--tac-src-clips-sha256` → equal to both. |
| **G-LIVE** | The trainer's `live_check` on the probe batch with `tac_w` attached: every declared term finite, every trainable group with a gradient, the frozen prior exactly 0.0. Then `tac_src_probe`, checks 1 to 7 of 3.4 with their RED arms. Then the parent card's reach table (`reach_probe.py`) on corpus-only / 10k-only / mixed batches at λ = 0: on the 10k-only batch every T term (`tac_lat`, `tac_lon`, `goal_geom`, `goal_vlm`, `cons_*`) reads exactly 0.0, every non-T term stays live. **Control:** the same table at λ = 1 reproduces `probe/U_s0.A.json` for the 10k-only batch (`tac_lat` grad 1642.0). |
| G-HYG | no config field; trainable parameters 7,214,495 and cfg equal to UE_s0's (M, summary `n_params_trainable`) |
| G-EVAL | UM's pilot checkpoint loads strictly through the `cdbff971` T1 eval's construction (`v8f_t1_eval.py:510-511`, I via GATE_REPORT), with cfg byte-equal to UE's; grid 2,399 / 141 |
| G-CKPT | the stage is stateless (a function of `ci` and λ), so the resume half (`V8F_TRAIN_CRASH_AT`) on a union smoke must be bit-equal to the uninterrupted run |
| G-CLOCK | the inputs are UE's, byte-equal (re-hashed). The block-1 G-CLOCK record applies by sha; a mismatch re-runs it |
| **G-SUITE** | NOT by identity this time, since the tree changes: `pytest stack/tests` on a clean export of the child commit, in the pod environment, **0 regressions** against `cdbff971` run the same way. The reference counts are from the STRAT1 baseline: 14,851 tests, 431 failed, 205 errors (I, GATE_REPORT row G-SUITE). The new `test_v8f_mini_tac_src.py` and the `FLAG_CHECKS` completeness test must pass. It runs CPU-only, never beside a training job on the pod. |
| post-training | Adam-state count 252; every non-frozen group moved; K-TACSRC |

**Pilot.** No separate pilot. The rate is MEASURED on the identical data and argv (UE_s0). The G-LIVE smoke (200 union steps) checks the s/step: a smoke rate above 1.5 x UE's 0.2242 s/step goes back to the MM before the launch.

## 8. Cost (pod = LoRA pod, RTX 4000 Ada; one GPU job at a time)

| item | pod-h | source |
|---|---|---|
| UM_s0 training loop, 61,568 steps | **3.83** | E from UE_s0's MEASURED loop time, **13,803.9 s** (`read3/ckpts/UE_s0.A/summary.json` `seconds`, 0.2242 s/step). `seconds` is the loop only, from `t0` at `v8f_mini_train.py:3545` to `:3676`. The banker records `read3/banker/UE_s0.A.json` carry no duration; this `summary.json` is the HF-banked file, pulled with its checkpoint (sha256 `2fd4b747...` verified at three copies, `read3/BUILD_READ3.md` section 1). Same data, argv and batches; the weighting adds a gather and three multiplies per step (E: under 1 %). |
| pre-loop (`--fit-join` on the union, selfcheck and G-LIVE probes, checkpoint writes) | ≤ 0.3 | E. Not in `summary.json`, so U. |
| gate smokes (G-LIVE 200 union steps, G-CKPT resume) | ≈ 0.3 | E |
| **total GPU-holding** | **≈ 4.4** | E |
| G-SUITE in the pod environment | 0 GPU, 1 to 2 h CPU wall | U: block 1 passed G-SUITE by identity, so this study measured no suite duration |
| banking (87 MB checkpoint to HF private) | minutes | I, block-3 banker |
| read | 0 (dev-box CPU, 30 to 45 min) | E, section 6 |

* Beside, for comparison: BL_s0 took 12,827.1 s (0.2083 s/step, M, `BL_s0.A/summary.json`).
* **Disk:** one run directory with an 87 MB checkpoint (M size of the UE / BL checkpoints) and `bset_clips.txt` (about 170 KB). The chain's 60 GB guard applies.
* **Deleted afterwards** (literal paths, logged with sizes, only once banked by sha256): the intermediate checkpoints under `/root/dscale10k/runs/UM_s0/A/` (keep `ckpt_last.pt`). The union view follows the parent card's section 7 rule, after every arm that reads it has finished.

## 9. Known risks
* **One seed, one-pair floor, marginal effect size** (section 0 item 4). The label waits for the replicate before it is quoted as a lever effect.
* **Gradient variance.** The T-block mean averages about 19.9 rows (E). A NOT-DILUTION could partly reflect a noisier tactical gradient, not the absence of dilution. The sampler-weight arm, which raises corpus rows per batch, is the test of that.
* **Share vs visits.** UM restores the share only. Its NOT-DILUTION does not exclude the visit form.
* **The TAC-LAT floor pair is itself separated** (−0.0767). The rig's training-run noise on this cell is about the size of the effect.
* **Country / platform shift** (10k stratified over 25 countries; eval-141 follows the corpus mix). The per-country read is reported beside, and no stratum decides.
* **Shared-file land** (section 0 item 2).

## 10. Deliverable manifest

| artifact | where | single copy? |
|---|---|---|
| this card | `F:/TanitAD-scratch/datascale/card_mix/CARD_DSCALE_MIX.md` | **yes** (F: scratch, not staged: the MM lands it with the study folder) |
| the share probe and its output (the MEASURED rows of section 2) | `F:/TanitAD-scratch/datascale/card_mix/share_probe.py` (sha256 `403c3e72...`), `share_probe_out.json` (sha256 `6e324848...`) | **yes** (re-creatable from the two hashed inputs in minutes) |
| code of section 3.3 | not written (specification only) | n/a |

**Deletions by this agent (own temporary data, literal paths):** listed in the hand-back. They are scratchpad exports of `cdbff971` files (re-creatable from git) and the scratchpad copies of the share probe (banked beside this card first).

## Log
* 2026-11-18: draft written (agent). Not frozen. Nothing trained, launched, staged or landed.
