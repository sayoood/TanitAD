# READ_PLAN: STRAT1 reads, what runs unchanged and what must be added (Architecture agent, 2026-11-16)

Card: `PREREG_V8F_STRAT1_DRAFT.md` (sections 4-6). Build: `BUILD_SPEC.md`. Evidence classes as there. Nothing here has run.

## 0. Where and how

* **Where:** the dev box, CPU only (`CUDA_VISIBLE_DEVICES=-1`, 2 threads), zero training.
* **Tree:** one read tree, a clean export of the STRAT1 branch commit with `git -c core.autocrlf=false archive` (memory: CRLF exports break bash). `PYTHONPATH` is `;`-joined Windows paths (memory: a `:`-joined path imports the wrong `tanitad`), and the reader asserts `tanitad.__file__`.
* **One environment:** every arm, C0 included, is dumped in one environment and one thread count. Memory: dumps at other thread counts drift by about a millimetre; `v8f_t1_eval` writes `t1v8f_<tag>.json` and no done file.
* **Pulls:** checkpoints come from the pod by sha256 against the arm's `BANK_MANIFEST.json`. Temporary dumps live on F: two levels deep and are deleted after the read is banked (CLAUDE.md delete rule; each deletion logged).

## 1. Runs unchanged (instrument of record: the OBD / OTBAND read)

| reader | source | gives |
|---|---|---|
| T1-kin dump driver | OTBAND `run_dumps.py` (`raw/2026-11-09-v8f-otband/read/`, I, via the OBD README) over `stack/scripts/v8f_t1_eval.py` | per-episode dumps, m = 10, causal bucket, dead-reckoned nav, logged scene |
| band rows | `read_lib.py` (OTBAND), the 11 metric rows per band | LONGITUDINAL, LATERAL, ADE/FDE per band |
| TAC-LAT / TAC-LON accuracy | same | TACTICAL (plan-derived) |
| T's own decisions, beside | `heads_read.py`, called as `beside_obd.py heads` does | T-LAT-F1, T-LON-F1, goal macro-F1 |
| top-3, beside | `topk_read.py` | best-of-3 minus top-1 |
| estimator | `taniteval.ci.paired_episode_cluster_bootstrap`, B 2,000, seed 0 | paired CI over 141 episodes ("another draw of episodes") |
| RC-A checkpoint error | `checkpoint_metrics` (`v8f_mini_eval.py:111-120`) and the per-window `ck_pos_err` the `paired` contrasts use (`:552-581`) | STRATEGIC route-setting rows (card 4.2) |
| S_CKPT rule | `strategic_cells.py ckpt_cell`, `side_table` (`raw/2026-11-16-v8f-strategic-audit/`, commit `0c58187e`), **imported, not copied** | STRATEGIC route following (card 4.4) |

The OBD pattern carries over: a thin `run_dumps_strat1.py` wrapper adds the STRAT1 tags and refuses an arm with a missing checkpoint, the wrong commit or a sha mismatch. The OTBAND driver itself is not edited.

## 2. Must be added

### 2.1 Nav arguments for text-fed arms (all readers that run a model)
Both evaluators refuse a text checkpoint without `--nav-text-npz` / `--nav-text-sha256` (`BUILD_SPEC.md` section 7). The wrapper reads `cfg.nav_source` from each checkpoint and passes the pinned npz and sha for S1 and N1, and nothing for C0. Test: a C0 tag given an npz is refused, and an S1 tag without one is refused.

### 2.2 `strat1_read.py`: the card's rules (pattern `obd_read.py`)
* Inputs: the dumps of C0_s0, C0_rep, C0_s1, S1_s0, S1_s1, N1_s0, N1_s1.
* Pairs:
  * the floor F = |C0_rep − C0_s0| per cell, read at seed 0 and applied to both seeds;
  * the decision pair S1 − C0 per seed;
  * the attribution pairs N1 − C0 and S1 − N1 per seed;
  * beside: C0 against the box-2 OBD-CTL bank at the same seed (cross-venue).
* Rules: card section 6, verbatim, including SEEDS-DISAGREE, and the next arm printed with each outcome.
* Output: `RESULT_STRAT1.json` / `.md` with one block per family, each carrying its tier stamp, estimator and n:
  ```
  {"LONGITUDINAL": {...}, "LATERAL": {...}, "TACTICAL": {...},
   "STRATEGIC": {"route_setting": {...}, "route_following": {...}, "detected_info": {...},
                 "not_in_arm": {"derived_goals": {"status": "not in this arm", "reason": ...},
                                "intent_6_30s": {"status": "not computable", "reason": "no output beyond 80 m of arc", "would_have_n": ...},
                                "visibility": {"status": "not built", "reason": "label 25/40 below the constant 31/40"}}},
   "floors": {...}, "controls": {...}, "outcome": ..., "next_step": ...}
  ```
* **No pause string.** The OBD/OTBAND readers print STRATEGIC as PAUSED (`PREREG_V8F_OBD.md` read section, I). That ruling ended on 2026-11-16, so `strat1_read.py` carries no such literal. Guard-in-implementation test: `grep -c "PAUSED" strat1_read.py` returns 0, with a same-breath control that greps a literal the file must contain (`"route_following"`) and returns >= 1.
* Tests: planted arrays for every outcome row of the card (KEEPS, ADDS, COSTS-NAV, COSTS-HEADS, SEEDS-DISAGREE, ROUTE-COST), the K1 self-pair exactly 0.0, a sub-floor separated delta that is neither gain nor cost, and NaN masking.
* Mutation: flip the floor comparison and require at least one RED test.

### 2.3 STRATEGIC T0 passes (route following and detected info)
Per arm, three runs of `v8f_mini_eval.py run` on the 2,399-window grid with `--ages fresh` (S's tick = NOW, so the text and the label are read at the window's own step):
* real nav (neither flag);
* donor (`--nav-shuffle`, `donor_indices` seed 0, episode-disjoint);
* zero (`--nav-withheld`, new).

Each run writes `eval_<tag>_<variant>.npz` with `ckpt`, `ctx_logits` (S1), `win_ci`, `win_t`.
The donor and zero passes run for C0 too: its route-following cells, read against the text, are the reference for "what an oracle-nav S does".

### 2.4 `strat1_strategic.py`: route following (card 4.4)
* Ground truth per window, at the eval step `eval_off[ci] + t`:
  * side from the text class (`side_table` over the navtok vocab; `navtok_table.npz` `eval_cls_id`, sha256 `2836369e...`);
  * `d_text` = the countdown distance `navtext_nav.npz` `eval_nav_d` (the rule of BUILD_SPEC section 5), NaN where the class has no side;
  * text flag != 0 gives NaN in every cell.
* The raw text distance (`eval_dist_m`) is a sensitivity row, as are COMPLETE_M 10 and 30. None of them decide.
* Cells: `ckpt_cell(ckpt[..., 2], (30, 50, 80), side, theta, d_text, text_ok, rc_ok)`, per arm, variant and seed.
* Paired contrasts over episodes:
  * real − donor (deciding);
  * real − zero;
  * real − best-constant (always the majority side among scorable windows; deterministic);
  * S1 − C0 under real nav (reported).
* n per arc and per episode is printed beside every number. Bands with fewer than 30 windows or 10 episodes are reported as n/a with their n (STRATEGIC_PLAN section 3 rule, I).
* Controls K7 (HOLD 0.0, mirrored flip, announced-side x 40 deg = 1.0) and K8 (synthetic nav-copying arm: the donor moves the cell) as tests.
* **T1 row (beside, real nav):** `v8f_t1_eval.py run --dump-ticks --dump-heads --dump-nav`.
  * Per tick: S's checkpoint (`s_ckpt`, NOW frame) and the nav fed at that tick (`nav_in`). Side is the text class at the window's t0, held as the dead-reckoned token is; `d = nav_in[3] x 100` where `nav_in[4] = 1`.
  * S_CKPT is scored per tick and pooled per episode.
  * K9: the tick-0 `s_ckpt` equals the T0 real-nav `ckpt` of the same window (atol 1e-6), else the T1 row is not quoted.

### 2.5 `strat1_ctx_read.py`: detected information (card 4.5)
* **Inputs:** S1's `eval_<tag>_real.npz` `ctx_logits`, split by `CTX_N` in `CTX_VARS` order. Labels come from `strat_labels_table.npz` (sha256 `8055a321...`) at `eval_off[ci] + t`, masked as LABEL_SPEC section 3. Platform per clip comes from the sidecar `strat_platform_by_clip.npz` (requested, LABEL_SPEC section 5).
* **Per variable and seed:**
  * accuracy and balanced accuracy;
  * per-class recall and the confusion matrix;
  * the constant baseline, recomputed from the table and asserted equal to LABELS_REPORT's (country 0.092, weather 0.846, surface 0.785, illumination 0.595, period of day 0.241, I);
  * the paired contrast (model correct − constant correct) per window, episode-cluster bootstrap over the 141 clips.
* **Country:**
  * rig baseline = argmax over train clips of country given platform, applied to eval clips; the paired contrast model − rig;
  * accuracy within each platform stratum against that stratum's constant.
* **Illumination:** twilight steps (240 steps, 1 clip, I) are excluded and reported as n/a with n. The 3-class accuracy is the deciding one.
* **Time-base sensitivity:** weather and illumination accuracy against labels shifted by 0.2 s (beside).
* **Controls K2-K6:**
  * K4 (C-RAND) needs one extra eval pass on S1's checkpoint with `ctx_head` re-initialised (seed 0). The pass is a flag of this reader, which re-initialises in memory and never writes a checkpoint.
  * If K4 reads DETECTS on any variable, the reader prints INSTRUMENT-FAIL and no detected-info number is quoted.
* **Tests:** K2 exact, K3 1.0, K6 exact, twilight exclusion, -1 masking, a misaligned table refused.
* **Mutation:** compute accuracy over masked rows too, and require a RED test.

### 2.6 Label and nav provenance in the result
`RESULT_STRAT1.json` carries the sha256 of:
* `navtok_table.npz`, `navtext_nav.npz` (and its QA verdict G-CLOCK-NAV);
* `strat_labels_table.npz`, the platform sidecar;
* the eval grid json, every checkpoint, and the read tree.

Every number carries its arm tag and artifact path.

## 3. Order of work (after the arms are banked)

1. Pull and verify every checkpoint (sha256).
2. T1 dumps of all seven arms in one chain, real nav (about 10 min per arm, I, E). Add `--dump-heads --dump-nav` for S1 and N1, and for C0 (the K9 identity and the C0 reference).
3. T0 STRATEGIC passes: 3 variants x 7 arms, plus K4 on S1_s0 and S1_s1 (about 2 h, E).
4. `strat1_read.py`, then `strat1_strategic.py`, then `strat1_ctx_read.py`; each writes its block. The first assembles the result and prints the outcome lines and next steps.
5. The `TanitAD_BenchmarkCriteria` check on `RESULT_STRAT1.json`: every family present, a tier stamp on every block, the estimator named, STRATEGIC members either computed or n/a with reason and n.
6. Register rows in `GOALS_AND_CLAIMS.md` in the same turn (MM).

## 4. What this plan cannot answer (named)

* Driving with the route text at T2 (re-rendered camera): not provisioned.
* Whether the heads help planning: not asked. Their outputs do not reach the planner in this arm (goal-input rule). The derived-goals card asks it.
* The 6-30 s strategic band: no output exists. The 120 m checkpoint card asks it.
* Inference-seed variance: none; the planner is deterministic.
