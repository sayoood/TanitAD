# FACTS_ROLLADD: what `--dagger-mode add` touches between `dag.apply` and `loss.backward()`

Evidence class: MEASURED from source at the code of record **cf27dbb7** (`agent/v8f-rollhist-cdb-20261120`), read with `git show` from the mirror (blobs identical to the exported tree). No GPU. Nothing staged, committed or pushed. Line numbers are cf27dbb7 (the build adds lines; the changed files are listed in `CHANGED_FILES.txt`).

## 0. MM design check: no line contradicted; two additions flagged
* `DaggerSwap.apply` `v8f_dagger.py:733`; `swap_rows` `:570` returns the FULL tensors (`new = {k: batch[k].clone() for k in REBUILT}` `:585`) with the selected rows replaced, plus `lab_s` (`:610`), plus `gt`/`s_h` only under `speed_v1` (`:619`). So "the N rows swap_rows produced for rows" = `upd[k][rows]` for every key `upd` carries, and `batch[k][rows]` (the logged value of the same window) for every other per-row key. That is exactly what the swap leaves in those rows today (`out.update(upd)` `:763`).
* ADDITION 1 (flag for the card): **the WM auxiliary's SIGReg is a batch statistic with an n prefactor** (`hjepa.py:131-134`, `:185-187`: `stat = (...) * float(n_total)`; called `:747`). RH is arm M2b (WM b), so it is ON. Under `add` n = B + N, so that term is ~(B+N)/B larger when the latents are away from the Gaussian null floor, and its ECF is taken over rows that include near-duplicate windows (same `feats`, ego tensors differ). This is NOT a fixed-B normaliser (no STOP), but it is a second difference between RA and RH besides "logged rows kept"; the card should name it.
* ADDITION 2: the batch has **B + N rows only when N > 0**; at a step with no eligible selected row the batch is the logged batch (B rows). The row count varies per step (p50/p95 in `REAL_SHARD_CENSUS.json` when run).

## 1. Every batch key, and whether it is per-row (`collate` `v8f_mini_data.py:724`; item `:663-692`)
| key | per-row? | under `add` |
|---|---|---|
| feats [B,F,n_tok,d] f16 (`:728`), frame_ok, limit_ok, gt_ok | yes | appended = logged value of the window |
| p10f, act, nav, pose, limit, gt, base, kin3, s_h, lead_gap_h, pose_now, pose_t, pose_s, p10_t (`:731-735`) | yes | REBUILT ones (`p10f base kin3 p10_t pose pose_t pose_s nav`, `v8f_mini_data.py:760`) appended from `swap_rows`; the rest = logged |
| a_t, a_s, t, ci | yes (long) | logged |
| stok, stok_ok | yes, only with a StokStore (`:741`) | logged |
| lab, lab_t, lab_s (dicts: has, nav, goal_y, goal_w, lat_cls, lat_al, lon_cls, lon_al, cons, rc, rc_ok, lead_valid) | every entry per-row | concatenated entry by entry; `lab_s.rc` appended = the frame-changed `rc` from `swap_rows` |
| gt, s_h under `--dagger-label speed_v1` | yes | appended = the speed label (`swap_rows :611-619`); RH has `human`, so logged |
| frames_u8 [U,...], frame_map | frames_u8 is NOT per-row (U unique frames); popped before the stages (`v8f_mini_train.py:3604`) and only with `--encoder lora` | REFUSED (lora) |
| launch, level_y, level_ok, lead_y, lead_has, lead_near, aux_ok | attached AFTER the stage (`:3646-3653`), per-row of the B logged rows | REFUSED (misaligned) |
Rule implemented: a tensor whose dim 0 equals B is per-row and is concatenated; a dict is concatenated entry by entry; anything else stays. Keys of the output = keys of the input (the existing probe's `set(pert) == set(batch)`).

## 2. Consumers between `dag.apply` and `loss.backward()` that assume B rows (trainer `v8f_mini_train.py`, cf27dbb7)
| consumer | file:line | assumes B? | action under `add` |
|---|---|---|---|
| `launch_stage(b)` -> `b["launch"] = lmask` | `:3607`, `:3646` | computed on the B logged rows BEFORE the stage, attached after: [B] against B+N rows | **REFUSED** (`--launch-intent-w > 0`) |
| `level_stage` -> `level_y/ok` | `:3609`, `:3652` | same | **REFUSED** (`--level-head on`) |
| `lead_stage` -> `lead_*` | `:3611`, `:3654` | same | **REFUSED** (`--lead-head on`) |
| `EgoDaggerCompose` (`comp`) | class `:714`, loop `:3613` | its overlap bookkeeping is a B-length mask (`dag.last["sel"]`, `:750`) | **REFUSED** (`--ego-perturb-with-dagger`) |
| `ego.apply` (`--ego-perturb-p`) | `:3618` | runs BEFORE dag; at p 0 returns the batch itself; at p > 0 it would perturb rows that `add` then duplicates | **REFUSED** (`--ego-perturb-p > 0`); RH has 0 |
| `rcv.apply` (`--recover-p`) | `:3622` | exclusive with dagger already; its counts use `b["kin3"] != b_logged["kin3"]` row-wise | **REFUSED** |
| `t2b.apply` (`--t2-bank-p`) | `:3633` | exclusive with the stages above; masks launch/level/lead itself | **REFUSED** |
| counters `dag_int/dag_tot` | `:3536-3537`, `:3637-3643`, `:3695-3699`, `:3734` | `n_rows` = B of the INPUT; eligible/selected are over the input rows | kept on B; ADD `appended` and `rows_out` keys ONLY under `add` (the swap key set is unchanged) |
| `to_dev` | `:3217` | row-count free (dict/tensor `.to`) | none |
| `lora_step(model, enc, b, fr[0], fr[1], ...)` | `:3001`, `:3680` | `frame_map` [B,hmax] and `frames_u8` index the B logged rows | **REFUSED** (`--encoder lora`) |
| `model.wm_only_loss` / `wm_windows wmonly` | `:3673`; batch has no dagger keys | the wm-only batch carries none of the swapped tensors | **REFUSED** (`--wm-only`, `--wm-windows wmonly`) |
| `dagger_probe` (G-LIVE) | `:2721-2860` | compares `pert` to `batch` row by row | **add branch** (section below); the swap branch is untouched |
| `ego_dagger_compose_probe` | `:2880-` | only runs with `comp` | not reached (refused) |
| `live_check`, `model.loss` | `:1804`, `v8f_mini.py:1156` | shape-generic (see 3) | none |

## 3. How every loss term is normalised (`v8f_mini.py:1156-1235`): NO fixed-B constant
* `ckpt` `:1167`, `goal` `:1171`, `cons` `:1181`, `lat`/`lon` (`_partial_ce` `:1255`), `level`, `lead_*`: `sum(term * mask) / mask.sum().clamp_min(1)`: a mean over the VALID rows.
* `ade`/`fde` `traj_terms :459-463`: `(d.mean(1) * gok).sum() / gok.sum().clamp_min(1)`; `acc` `accel_term :497-504`: same form (the RH argv's `--w-acc 49.6969`): means over GT-valid rows.
* `pen_*`, `pens_*` `:1221-1222` (`aux is None`): `.mean()` over all rows.
* `wm` `:1226`: `HJEPAStack` losses: `pred` mean over batch (`hjepa.py:724`), IDM mean, **SIGReg x n** (see 0).
* Therefore under `add` every mean-type term weights each row by 1/(valid rows in B+N) instead of 1/(valid rows in B); the per-row share of a logged row falls by B/(B+N). Nothing is divided by a literal B. **STOP condition not triggered.**
* No BatchNorm / batch-variance layer in `v8f_mini.py` (grep `BatchNorm|batch_norm|.std(|.var(` = 0 hits).

## 4. Global-RNG draws inside `model.loss`, and how they scale with the row count
The loop seeds the GLOBAL RNG per step: `torch.manual_seed(a.seed * 1_000_003 + step)` (`:3664`), then `model.loss(b, p_gt=p_gt, gen=None)` (`:3669`):
* S ckpt teacher forcing `torch.rand(B, generator=gen)` `v8f_mini.py:1042` (B = `feats.shape[0]` = rows of the batch given);
* code mix: `code_mix == "kind"` `torch.rand(B, N_CODE_UNITS)` `:1019` inside `_code_mix_mask`; `whole` `torch.rand(B)` `:1059`;
* SIGReg directions `torch.randn(dim, 256)` `hjepa.py:169`, one draw per level per call, independent of n;
* `p_gt` `:3663` is a scalar from the schedule (not row-dependent).
So the number of draws per call grows with the row count: under `add` the draw vectors are B+N long, and every draw AFTER the first is shifted in the stream against the swap run (the first B values of a CPU `torch.rand(B+N)` are the `torch.rand(B)` values in the mt19937 stream, but the later calls consume a different offset). Consequence: RA and RH do not share the same teacher-forcing draws per row, even at the same seed. This is a property of any batch-size change, not a defect; a replicate must not expect bit-for-bit draws against RH.

## 5. Stages that cannot run beside `add`: REFUSED at gate time (G-DVB messages in `bad_dagger`)
`--launch-intent-w > 0`, `--level-head on`, `--lead-head on`, `--ego-perturb-with-dagger`, `--ego-perturb-p > 0`, `--recover-p > 0`, `--t2-bank-p > 0`, `--encoder lora`, `--wm-only`, `--wm-windows wmonly`, and `add` at `--dagger-p 0` (a lever that does nothing). RH's argv names none of them (`--ego-perturb-p 0`, `--w-acc 49.6969 --detach-upper --fit-join`, no launch/level/lead/compose/recover/t2-bank): `add` is accepted for it. `--dagger-label speed_v1` composes with `add` (appended rows carry the speed label); the RA arm uses `human`.

## 6. Reachability (file:line, post-build line numbers in `CHANGED_FILES.txt` / `FACTS` re-stated in `REACHABILITY_ROLLADD.md`)
argparse `--dagger-mode` -> `a.dagger_mode` -> `make_dagger` -> `DG.DaggerSwap(mode=...)` -> `built_levers()["dagger_mode"]` -> `bad_dagger` (G-DVB) -> `dag.apply(b, step)` in the loop (`:3621`) -> `append_rows` -> `b` with B+N rows -> `to_dev` -> `model.loss(b)` -> `T["total"].backward()` (`:3671`). The probe in `dagger_probe` (add branch) proves the gradient reaches the parameters.
