# INTAKE — production semantic labeling with Cosmos-Reason2-8B (2026-07-21)

**Research note:** [`../../../Research/2026-07-21-cosmos-reason2-production-semantic-labeling.md`](../../../Research/2026-07-21-cosmos-reason2-production-semantic-labeling.md)
**Harness:** `stack/scripts/vlm_semantic_labels.py` (prompt v2) · **Scorer:** `stack/scripts/vlm_semantic_score.py` (no pod, no GPU)
**Lake converter:** `stack/scripts/vlm_labels_to_lake.py` — per-window rows for the TanitEval scenario strata **and** the episode-level `vlm_pending_*` sidecars `stack/tanitad/lake/enrich.py` has been stubbing.
**Tests:** `stack/tests/test_vlm_semantic.py` — **26 tests**, the suite's first VLM coverage (full suite 637 passed / 2 skipped).

> ⏹ **CLOSED (for the val corpus) 2026-08-16 — the sidecars are NO LONGER a to-do; DO NOT commission
> a "fill the VLM sidecars" task without reading this first.** The converter landed and the fill is
> real: `stack/scripts/vlm_labels_to_lake.py` writes `_pending = False` into all four skeletons
> (`:329` scene_tags, `:351` lead_state, `:365` sign_reads, `:378` language), and
> `stack/tests/test_vlm_semantic.py:295` asserts `ls["_pending"] is False` — so it is guarded, not
> just written once. **24 filled sidecars** are banked at `…/2026-07-21-vlm-production-semantic/sidecars_val/`
> (`ep_*.goal.vlm.json`) alongside `scenario_strata_val.jsonl`.
> ✅ **STILL TRUE as stated, and deliberately so:** `stack/tanitad/lake/enrich.py:49-82` **still
> defines** `vlm_pending_scene_tags / _lead_state / _sign_reads / _language` and still wires them as
> the `EpisodeEnrichment` **default factories** (`:113-116`), and its module docstring (`:17`) still
> reads *"The VLM augmentation is the ONLY stubbed step"*. That is correct behaviour — the skeleton is
> the default for an **unlabelled** episode; the converter overwrites it for a labelled one. There is
> **no** merge-back path in `enrich.py` itself (probed: no `merge_vlm` / `apply_vlm` / `from_vlm`
> symbol anywhere under `stack/`), so the fill is applied by running the converter, not automatically.
> ⚠️ **Scope limit, MEASURED:** 24 sidecars, not 80 — the fill covers a subset of the pod3 val build,
> and `val_build_episode_map.json` already records that only **8 of 40** canonical val episodes are
> reachable from pod3. "Filled" ≠ "complete".
> Swept by the 2026-08-16 stale-blocker sweep.
**Pod:** `tanitad-pod3` (A40 46 GB), GPU lock held as `vlm-production` for the whole campaign.

## What is in this directory

> **The campaign runs for hours. To pull whatever exists on the pod at any moment, including mid-run:**
> ```
> bash stack/scripts/pod_ops/pull_vlm_records.sh
> ```
> Idempotent, safe to re-run, and it skips a record left half-written by a kill. A partial run is a valid
> smaller corpus: the val manifest is **t-major** (a stopped run has covered *all 80 episodes* at a coarser time
> stride, not the first N episodes) and the train manifest is **shuffled** (any prefix is a uniform subsample of
> the stratified draw; measured stratum drift ≤ 3.8 pp at n=300 of 600).

| file | what it is |
|---|---|
| `val_build_episode_map.json` | **Read this first.** The join key between the TanitEval canonical val build and the build physically present on pod3 — and the measurement that only **8 of 40** canonical val episodes are reachable from pod3. |
| `enums.json` | The enum table **exactly as shipped in the prompt**. The scorer reads this, never the vocabulary module, so scoring can never drift from what was actually asked. |
| `probe_windows.json` | Shared window manifest for the enum-order probe and all Phase-1 arms (200 windows / 40 held-out val episodes) with the kinematic v2.1 route label stamped inline. |
| `r2_as_written.jsonl`, `r2_right_first.jsonl` | The **enum-order probe**: identical windows, identical prompt except `right` listed before `left`. `prompt_version` differs (`…-a` vs `…-a-rswap`) so the two can never be pooled. |
| `probe_results.json` | Scored probe output (`vlm_compare_score.py`). |
| `p1_v1.jsonl`, `p1_v2.jsonl`, `p1_v2b.jsonl` | Prompt **before/after** on identical windows: v1 @ 2200 tokens, v2a @ 3500, v2b @ 3500. `p1_v2` was stopped at n=13 and `p1_v2b` at n=3, both by explicit PID, once their results were unambiguous — see the note's §0. |
| `ab_base.jsonl`, `ab_dense_early.jsonl`, `ab_wide_cheap.jsonl`, `ab_dense_hist.jsonl` | The **frame ablation** — one file per frame plan, identical windows and prompt (100 windows each). |
| `ab_base_randenum.jsonl` | The **enum-order sensitivity** arm on `road_geometry`: same windows and plan as `ab_base`, every enum permuted per window. |
| `val_full.jsonl` | Production labels, pod3 val build, all 80 episodes. **The manifest is t-major**, so a partial file covers every episode at a coarser time stride. |
| `train_strat.jsonl` | Production labels, rare-event-weighted sample of the canonical 2376-episode train build. Manifest **shuffled**, so any prefix is a uniform subsample. |

> ⛔ **STRANDED ARTIFACT FOUND 2026-08-16 — `train_strat.jsonl` DOES NOT EXIST in this package.**
> This table row documents a file that is not here. Three probes: (1) `find . -name "train_strat*.jsonl"`
> over the whole repo → **zero hits**; (2) `git ls-files | grep train_strat` → only
> **`train_strat_windows.json`** (the manifest), never the records; (3) `git check-ignore` → **not
> gitignored**, so it is genuinely absent rather than hidden. The manifest and the
> `train_candidate_census.json` (21,393 candidate windows) ARE both present, so the *inputs* survived
> and the train draw is re-runnable — but the **production train labels themselves were never banked
> in the repo** and, if they were ever produced, they existed only on `tanitad-pod3` (which the
> package's own §last row notes is now unreachable). This is the "an artifact on one disk is NOT done"
> class. ⇒ Either re-run the train draw from `train_strat_windows.json`, or strike this row.
> Swept by the 2026-08-16 stale-blocker sweep.
| `train_strat_windows.json` | The stratified manifest (600 windows, equal quota per stratum) + the candidate-stratum census summary. |
| `train_candidate_census.json` | **All 21,393 candidate windows** over the canonical 2,376-episode train corpus with their kinematic v2.1 route labels and stratum tags (~8 MB). This is what makes re-sampling the train draw a pod-free operation. |
| `prompt_A_v1.txt`, `prompt_A_v2b.txt` | The exact Pass-A prompt text of each version — the scorer's evidence-contamination detector n-grams against these, and they are the authoritative record of what was asked. |
| `*_results.json`, `res_cmp*.json`, `res_enumorder.json` | Scorer output for each comparison. |
| `audit_val.tsv` | Stratified spot-check sheet **for a human** — over-samples the eventful geometries and puts `SHIPPED_geometry` (Pass A, what we ship) next to `passB_geometry_CONTAMINATED`, the kinematic label and the model's own evidence sentence, with blank `HUMAN_VERDICT_*` columns. Scenario labels have no ground truth; this is the cheapest instrument that can produce some. |
| `scenario_strata_val.jsonl` | The **per-window rows TanitEval v2's scenario strata consume**, keyed `(episode, t)` with `taniteval_window_start = t − 8` precomputed. Geometry and event times come from **Pass A** (independent); Pass B's are carried alongside marked `_CONTAMINATED`. |
| `sidecars_val/` | The **episode-level `vlm_pending_*` sidecars** `stack/tanitad/lake/enrich.py` has been stubbing, filled shape-for-shape — with per-axis agreement shares, and `gap_m`/`closing_speed_ms`/`ttc_s` left `None` plus `_metric_fields_unavailable` explaining why. |
| `res_val_full.json` | Scorer output for the production run. |
| `legacy_pod3_passA.jsonl`, `legacy_pod3_passB.jsonl`, `vlm_crossval.json`, `route_audit_v21.json` | The **rescued** pre-existing corpus that had been living only on `tanitad-pod3:/workspace` (400 + 160 records, prompt `vlmroute-2026-07-20-a`). Different schema — no inline kinematic ground truth, no token counts — so it does not score with `vlm_semantic_score.py`. Kept because a pod is not storage. |

**Production settings, all chosen on measurements in the note:** frame plan **`dense_early`** (§3),
Pass-B budget **`--max-new-b 1200`** (§2 — *lower* than v1's 2200; 0 of 34 completed replies exceeded 1146
tokens), enum order **`as_written`** (§1d — randomizing costs 3 % of answers for no measured gain).

## Reproduce without a pod

```bash
python stack/scripts/vlm_semantic_score.py \
  --out "TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-21-vlm-production-semantic" \
  --arms val_full --json /tmp/val.json

python stack/scripts/vlm_compare_score.py \
  --out "TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-21-vlm-production-semantic" \
  --arms r2_as_written,r2_right_first --json /tmp/probe.json

# turn the records into what the lake and the metric suite consume
python stack/scripts/vlm_labels_to_lake.py --jsonl <dir>/val_full.jsonl \
  --windows-out /tmp/scenario_strata.jsonl --sidecars-out /tmp/sidecars
```

Both read the `.jsonl` files directly. `windows.json` must be named `windows.json`
for `vlm_compare_score.py`; `probe_windows.json` is a copy under its descriptive name.
