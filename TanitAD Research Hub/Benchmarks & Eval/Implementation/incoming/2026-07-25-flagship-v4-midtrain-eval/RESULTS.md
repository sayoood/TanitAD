# flagship-v4-fromscratch @ step 15,000 — mid-training eval on the clean val

**Eval ran:** 2026-07-25 **22:05–22:15 UTC** = 2026-07-26 **00:05–00:15 Europe/Berlin** on
`tanitad-eval`. *(Folder is dated 2026-07-25 for the UTC day the run belongs to; local wall-clock
had just rolled past midnight. Pods and logs are UTC.)* ·
**Agent:** benchmarks-eval · **Status of the run being measured:** *TRAINING, NOT FINISHED*

> **This is a MID-TRAINING checkpoint at step 15,000 of a 30,000-step schedule** — exactly half,
> from a **random-init trunk** (`from_scratch: true`, no v1 warm-start). It is **not** the final
> model and must never be quoted as "flagship v4". Per the brief this report states **where the
> arm sits on the descent** and does **not** adjudicate pass/fail.

---

## 1. Headline — the number

| | value | interval | estimator |
|---|---|---|---|
| **v4-fromscratch @ 15k — ADE@2s** | **0.5839 m** | **[0.4962, 0.6821]** | `episode_cluster_bootstrap` (B=2000, unit = val episode) |
| deployed v1 (`flagship-30k`, step 29999) | 0.4271 m | [0.3675, 0.4871] | `episode_cluster_bootstrap` (B=2000) |
| **paired Δ (v4@15k − v1)** | **+0.1568 m** | **[+0.0630, +0.2504]** | **`paired_episode_cluster_bootstrap`** |

**Evidence class: MEASURED** (ours; artifacts listed in §6).
Surface: **881 windows / 40 episodes**, clean val `physicalai-val-0c5f7dac3b11`, 4-waypoint
convention (steps 5/10/15/20 = 0.5/1/1.5/2 s) — the only convention other MODEL_REGISTRY rows use.

**Read:** at half its schedule the from-scratch arm is **CI-separated *behind* the completed v1 by
≈0.16 m** (`separated: true`, `p_delta_gt0 = 1.0`). It is **not** behind the trivial floors — it
beats CV and hold-v0 with separation (§3). This is the expected place on the descent for a
co-evolving WM+planner at 15k/30k from random init; **no restart signal is claimed here.**

> ⚠️ **Two intervals for the same point estimate.** The canonical harness block
> (`driving_*.json`) reports **0.5839 [0.4962, 0.6821]** (grouping by v4's own episode labels); the
> paired analysis, which must use ONE common grouping, reports the identical mean with
> **[0.4916, 0.6855]**. Same estimator, same partition — the difference is bootstrap Monte-Carlo
> draw order only. Quote the harness interval; both are in the artifacts.

### ⛔ The comparison that must NOT be made

The in-trainer val hovering **~0.48** is **NOT** comparable to v1's 0.4271. Measured here, the
trainer's statistic — the **dense 20-horizon** mean — is **`dense_headhorizons_ade_2s = 0.4596`**,
while the v1-comparable 4-waypoint mean on the *same forward pass* is **0.5839**. The dense mean is
diluted by small early-horizon errors and reads **~0.12 m lower**. Reading ~0.48 against 0.4271 and
concluding "nearly caught v1" would be a **RETRACTION_LOG class C1** error (*faster-moving source
than the harness* — trainer logs watch curves, only `eval_*.py` output is quotable).

---

## 2. Lateral vs longitudinal decomposition — the safety-relevant axis

Ego frame, **axis0 = along-track (longitudinal), axis1 = cross-track (lateral)**
(`taniteval/lateral.py`, `from_sparse_windows`, `mode="ego"`, surface `sparse_4wp`, dt 0.5 s).

| | **v4@15k** | v1 (`flagship-30k`) |
|---|---|---|
| longitudinal share of squared error | **0.6191** | **0.8733** |
| lateral share of squared error | **0.3809** | 0.1267 |
| long. share by knot (0.5→2.0 s) | 0.795 / 0.707 / 0.648 / 0.603 | 0.874 / 0.899 / 0.924 / 0.854 |
| peak \|XTE\| mean | **0.6336** [0.4760, 0.8188] | 0.2857 [0.2290, 0.3479] |
| peak \|XTE\| p90 | **1.4277** [1.1248, 2.0713] | 0.7119 [0.5046, 0.8000] |
| windows with peak \|XTE\| > 1.75 m | **6.4 %** | 0.8 % |

All intervals: `episode_cluster_bootstrap`, B=2000, 40 episodes.

**Read — this is the finding, not the ADE gap.** The v4@15k residual is **not a scaled-up v1
residual**. v1's error is overwhelmingly longitudinal (87.3 %); v4@15k's is **38.1 % lateral**, and
its lane-scale cross-track tail is **8× v1's** (6.4 % vs 0.8 % of windows beyond 1.75 m).
An undecomposed L2 would have hidden this entirely. At 15k the arm has **not yet converged its
lateral/geometric behaviour**, which is the axis where departures are safety-critical.
*(v1's 0.8733 reproduces the independently measured 0.873 — a cross-check on this instrument.)*

Frenet-frame companion (`taniteval.driving` tier-0, GT-tangent frame — a different, complementary
decomposition): along `0.8567` [0.7098, 1.0126] · cross `0.6387` [0.4778, 0.8267] ·
RMSE long `1.2354` / lat `1.0494` · `long_frac_of_2s_sqerr = 0.5809`.

---

## 3. Against the trivial floors (paired, same 881 windows)

Orientation **floor − model**; positive = model wins. Estimator
`paired_episode_cluster_bootstrap`.

| test | Δ | CI95 | separated | favours |
|---|---|---|---|---|
| ADE vs **CV** (0.8377) | **+0.2538** | [+0.1124, +0.4337] | ✅ yes | model |
| along-track vs CV | +0.2388 | [+0.0751, +0.4208] | ✅ yes | model |
| cross-track vs CV | +0.3701 | [+0.1153, +0.6923] | ✅ yes | model |
| speed MAE vs CV | −0.0263 | [−0.0803, +0.0277] | ❌ no | **tie** |
| speed MAE vs hold-v0 | −0.0123 | [−0.0595, +0.0309] | ❌ no | **tie** |

`where_the_win_lives = "both axes"` · `beats_cv_ade_separated = true` ·
`tracks_speed_better_than_cv = **false**`. hold-v0 ADE floor = 0.7876.

**Read:** the arm is already a genuine model at 15k — separated wins on *both* axes over CV, which
v1 itself does not achieve on the along-track axis (registry §1.2: v1's along-track vs CV is
**not** separated). But like every arm in the program it **does not track speed better than the
trivial baselines** — the longstanding longitudinal weakness is present here too.

---

## 4. Secondaries (as emitted by the harness)

| metric | value | interval / note |
|---|---|---|
| FDE@2s | 1.2317 | [1.0407, 1.4437] |
| miss@2m | 0.1691 | [0.1057, 0.2398] |
| **oracle-in-fan (4wp)** | **0.2797** | anchor fan can reach 0.28 m — comparable to v1.5-ab's 0.3073 |
| WM canary ADE@2s | 2.0739 | trunk-integrity rollout under TRUE actions, n=881 |
| seam_norm_ratio_max | 0.1244 | head telemetry |
| dense-20 ADE@2s | 0.4596 | **trainer-loop-comparable only** (see §1) |
| dense-20 oracle / sel-gap / miss | 0.2401 / 0.2195 / 0.1691 | |
| speed bias | +0.0844 m/s | [+0.0206, +0.1521] |
| heading MAE@2s | 7.1458° | [3.9541, 11.2846] — heavy-tailed, prefer median |
| curvature sign agreement | 0.8468 | [0.8122, 0.8801] |

**Cross-check passed:** the harness's own forward-pass ADE (0.58392) and `taniteval.driving`'s
recomputation from the persisted windows (0.5839) agree — `agree_within_1pct: true`. Two
independent code paths; disagreement would have indicated an ego-frame convention mismatch.

**The harness's pre-registered thresholds** (recorded for completeness — **NOT adjudicated here**,
per the brief; these were written for a *10k gate on a warm-started* arm and this is a *15k
from-scratch* arm): primary `ade_0_2s ≤ 0.60` → 0.5839 **pass**; `oracle_in_fan ≤ 0.30` → **pass**;
`seam_norm_ratio_max ≤ 1.0` → **pass**; `wm_canary_ade_2s ≤ 0.55` → **2.0739 fail**;
`miss_at_2m ≤ 0.10` → **0.1691 fail**. Four secondaries were **not computed** by design
(`speed_benefit_recovered_frac`, `deploy_tick_p99_ms`, `nonav_route_beats_majority` — no emitter /
not reachable on this ckpt; `encoder_touching_levers` is a static design fact, not a measurement).

> The **canary at 2.0739** is the most informative secondary: the *world model itself*, rolled out
> under true actions, is still far from v1's ~0.43. The planner head is currently carrying the arm
> (0.5839 planner vs 2.0739 canary). For a from-scratch co-evolving run at 15k this is consistent
> with a trunk that has not finished converging — it is a **descent position, not a verdict**.

---

## 5. Exact eval command (reproduction)

Run on **`tanitad-eval`** only. `--labels v3` is **not** an eval argument — `FlagshipV4Dataset`
mints the v3 factorised + strategic labels on the fly (the trainer flag has no eval counterpart).

```bash
export PYTHONPATH=/root/v4eval/stack:/root/taniteval:/root/v4eval/stack/scripts
cd /root/v4eval/stack/scripts
python3 -u /root/v4eval/stack/scripts/eval_flagship_v4.py \
  --ckpt          /workspace/models/flagship-v4-fromscratch-15k/ckpt_step15000.pt \
  --anchors-dense /workspace/models/flagship-v4-fromscratch-15k/flagship_v4_anchors_dense.pt \
  --head-config   /workspace/models/flagship-v4-fromscratch-15k/config.json \
  --val-cache     /root/valdata/physicalai-val-0c5f7dac3b11 \
  --key           flagship-v4-fromscratch-15k \
  --out           /root/v4eval/results/flagship-v4-fromscratch-15k.json \
  --results-dir   /root/v4eval/results \
  --episodes 40 --stride 8 --batch 16 --device cuda
# then, for §1 paired + §2 decomposition:
PYTHONPATH=/root/taniteval python3 /root/analyze_v4_15k.py
```

MODE **B** (planner path) was auto-selected because the ckpt carries a `head` key
(`['controller','goal_head','grounding','head','lam_mult','model','opt','phases','step']`).
Head config read from the **run's own `config.json`** (`n_anchors=256`, dense horizons 1..20,
`factorised=true`, `cond(states/imag/vtarget/route)=T/F/T/T`) — the `v4_config()` defaults would
not have STRICT-loaded. Trained dense anchors were loaded (not the seed-0 FPS fallback).
Wallclock: canary 93 s + planner path ~160 s.

### ⚠️ Two operational findings worth carrying forward

1. **`PYTHONPATH=/root/v4eval/stack` alone is NOT sufficient** — it must be
   `/root/v4eval/stack:/root/taniteval:/root/v4eval/stack/scripts`, or `taniteval.driving` (the
   episode-cluster-bootstrap primary) never runs and the decision-grade interval is silently lost.
   Also: `taniteval/driving.py:80-82` hard-inserts `/root/TanitAD/stack` at `sys.path[0]`, which
   **repoints `tanitad`** to a copy lacking `lake/vocab.py`. Importing `taniteval` *before*
   `flagship_v4_data` therefore dies with `ModuleNotFoundError: tanitad.lake.vocab`. The harness
   happens to import in the safe order; anything reusing these modules may not.
2. **The eval pod's `/root` overlay is 99 % full (3.0 GB free)** — too small for a 3.24 GB
   checkpoint. The first transfer produced a **truncated, md5-mismatched** file that was *present*
   and would have loaded as garbage. Checkpoints belong on **`/workspace`** (verified by a real
   4 GB `dd` write at 505 MB/s, not by `df`). **Presence is not integrity — md5 both ends.**

---

## 6. Deliverable manifest

| artifact | repo path (this folder) | also lives at | only one copy? |
|---|---|---|---|
| Raw merged eval result (bench + v4 diagnostics + driving block) | `flagship-v4-fromscratch-15k.json` | `tanitad-eval:/root/v4eval/results/` | no |
| Gate/diagnostics summary (primary + secondaries + cross-check) | `flagship-v4-fromscratch-15k_v4_diagnostics.json` | `tanitad-eval:/root/v4eval/results/` | no |
| Driving tier-0 block (all bootstrap intervals, floors, strata) | `driving_flagship-v4-fromscratch-15k.json` | `tanitad-eval:/root/v4eval/results/` | no |
| **Lateral decomposition + paired-vs-v1** (§1 Δ, §2 table) | `flagship-v4-fromscratch-15k_lateral_and_paired.json` | `tanitad-eval:/root/v4eval/results/` | no |
| **Per-window dump** `pred/gt/cv/eid/speed/head_deg` (881×4×2) | `windows_flagship-v4-fromscratch-15k.pt` | `tanitad-eval:/root/v4eval/results/` | no |
| Full eval stdout log | `eval_run_2026-07-25.log` | `tanitad-eval:/workspace/v4eval_15k.log` | no |
| **Analysis driver** (paired-vs-v1 + alignment proof + lateral block) | `analyze_v4_15k.py` | `tanitad-eval:/root/analyze_v4_15k.py` | no — **rescued into the repo** |
| Eval launch script (exact invocation) | `run_v4eval.sh` | `tanitad-eval:/root/run_v4eval.sh` | no |
| This report | `RESULTS.md` | — | **repo only** |

**Live on pods only (NOT in the repo, by intent — large or environment-specific):**

| artifact | location | note |
|---|---|---|
| The evaluated checkpoint (3.24 GB, md5 `8e2facba19b8f8a639ec457d7941c00e`) | `tanitad-eval:/workspace/models/flagship-v4-fromscratch-15k/ckpt_step15000.pt` | **kept**, as instructed. Source of truth remains `tanitad-pod2:/workspace/experiments/flagship-v4-fromscratch/ckpt_step15000.pt` (untouched) |
| Dense anchors + run config (copies) | same dir: `flagship_v4_anchors_dense.pt`, `config.json` | originals on pod2 |
| Analysis driver | `tanitad-eval:/root/analyze_v4_15k.py` | also copied into this folder — not stranded |
| `taniteval/lateral.py` (installed for this eval) | `tanitad-eval:/root/taniteval/taniteval/lateral.py` | copy of the in-repo file; repo is authoritative |

### 🔺 Escalation — needs a decision, not a README line

1. **`analyze_v4_15k.py` should be promoted into `taniteval/`.** It is the only code that computes
   the **paired v4-vs-v1 bootstrap** and the **alignment proof** that makes that pairing
   admissible. It is no longer stranded (a copy sits in this folder), but it is a one-off script in
   an `incoming/` dir — it belongs in `taniteval/` as a reusable *"compare two window dumps"*
   driver, because this exact comparison will be re-run at **20k and 30k**.
2. **`windows_*.pt` episode labels are not comparable across harnesses.** v4 writes the raw
   `episode_id` (e.g. `808464434`), the older `rollout.collect` wrote sequential `0..39`. A naive
   eid-equality check **wrongly rejects a valid paired test** (it rejected mine on the first pass).
   Alignment must be established on the data (`gt`/`cv`/`speed` identity + block structure +
   bijection), or the emitters should be reconciled. Worth a one-line fix in the v4 harness.
3. **Files in this folder are UNSTAGED** — written, not `git add`ed, per the orchestrator's
   instruction that concurrent staging corrupts the index while sibling agents run.
   **The orchestrator must stage them.**

---

## 7. Provenance of every quoted comparison figure

| figure | class | source |
|---|---|---|
| v1 ADE@2s = **0.4271** (full-set) | **MEASURED, re-derived here** | recomputed from `windows_flagship-30k.pt`; reproduced **0.4271 exactly**, pinning this analysis path to `MODEL_REGISTRY.md` §1.2 |
| v1 longitudinal energy share 0.873 | **MEASURED, re-derived here** | 0.8733 via `lateral.from_sparse_windows` |
| CV floor 0.8377 · hold-v0 0.7876 | **MEASURED** | `driving_*.json` `floor_values`, computed on these same 881 windows |
| v1.5-ab oracle-in-fan 0.3073 | **INHERITED** | harness note; not re-verified this session |
| v4 schedule = 30k, `from_scratch: true`, milestones 5/10/15/20/30k | **MEASURED** | the run's own `config.json` (transferred from pod2) |

**Estimator policy honoured throughout:** every interval above is an **episode-cluster bootstrap**
(paired form for every two-arm comparison), B=2000, resampling unit = the 40 val episodes.
**`overlapping_holdout_se` appears nowhere in this report** — `taniteval.driving` refuses to emit
it (`assert_no_deprecated_estimator`), and it is measured to bias both interval width and the point
estimate.

---

## 8. Safety record for this eval

- **pod2 (training flagship-v4, RAM 53.7/55 GB): READ-ONLY `scp` of already-existing files only.**
  No python, no writes, no `cp`, no HF push, no `nvidia-smi` loop. The rolling `ckpt.pt` was
  **never** read — only the immutable milestone `ckpt_step15000.pt`.
- Transfer used the sanctioned relay `pod2 → devbox → eval`. **md5 verified end-to-end**
  (`8e2facba19b8f8a639ec457d7941c00e`); the first attempt's truncated file was detected by md5 and
  discarded, not evaluated.
- All compute ran on **`tanitad-eval`** (GPU verified free, 0 MiB / 46 GB, before launch).
  pod1 / pod3 untouched.
- No process was killed; no `pkill -f` / `pgrep -f` was ever issued.
- `ckpt_step20000.pt` has since appeared on pod2 but was **deliberately not chased** — a clean 15k
  number now beats a restarted 1.5 h transfer, per the coordinator's direction.
