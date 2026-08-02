# ADVERSARIAL VERIFICATION — stream `refc-optimisation`

**VERDICT: the report is ~70 % sound on code/line facts and headline eval numbers, but carries 3 claims that must be struck, 6 that must be materially qualified, and its entire cost/priority section is built on a compute premise that is false as of today.** 13 findings below, then the corrected version.

---

## PART A — REFUTATIONS

### ⛔ A1. RANK 4 is a fabricated defect. STRIKE IT.
The report claims an "artifact inconsistency, flagged not silently reconciled (R14 class)": stages summing to 28.5 ms vs `plan_step` p50 21.78 ms vs the registry's 90.7 %, "these cannot all be true", costing "< 0.5 GPU-hours" to settle.

The artifact already publishes both the aggregate **and** the reconciliation, in the same block the report read the stages from — `stack/experiments/pod-rescue-20260802/pod3/root/taniteval/results/eff_refc-base-30k.json`:
```
/fp32/stage_shares/isolated_stage_sum_ms            = 28.9578
/fp32/stage_shares/isolated_stage_sum_pct_of_plan_step = 132.8
/fp32/stage_shares/stage_sum_note = ">100% => the stages overlap CPU-launch with GPU-execute
   inside the full step (launch-bound); <100% => untimed glue. Treat shares as attribution,
   not an exact partition."
/fp32/stage_shares/encoder_pct                      = 90.7
```
`encoder_pct 90.7` = 19.7772/21.8034 — identical to `MODEL_REGISTRY.md:1504`. Registry and artifact **agree**; there is no conflict. Worse, the report's "28.5" is a hand-recomputation that mixes p50 into a mean-based aggregate and drops the `aux_heads` term — a number that appears nowhere in the artifact.

### ⛔ A2. "LATERAL (complete)" is FALSE — the artifact refuses three of the five numbers the report headlines.
`…/2026-08-02-ctrv-floor/raw/four_families_vs_floors.json`, `arms.refc-xl-30k.refused_metrics`:
```
heading_mae_deg      : "per-window form disagrees with four_families by 5.12e-02 (>0.001);
                        point estimate published, INTERVAL REFUSED"
yaw_rate_mae_degps   : "... by 8.68e-02 ... INTERVAL REFUSED"
curvature_mae_1pm    : "... by 1.08e-02 ... INTERVAL REFUSED"
```
`LATERAL.vs_floor_paired` contains **only** `cross_mae_m` and `cross_bias_m`. Under the binding rule ("each family carries its estimator and its CI"), LATERAL is **2 of 5 with intervals**, not complete. The report quoted 1.1484° / 1.8216 °/s / 0.012138 1/m with no mention of the refusal.

### ⛔ A3. "Thor is currently pulling four REF-C checkpoints" — wrong twice, and the urgency premise collapses.
`TanitAD Research Hub/Production & Optimization/THOR_ACCESS_BRIEF.md:70` lists the four pulling dirs as `refc-base`, `refc-xl`, `refc-base-e1b-clsft`, **`flagship-v4.2b`** — three REF-C, one flagship. The report itself says flagship-v4.2b is not REF-C in §1 and then contradicts that in RANK 1.
Second probe (line 69) finds a **fifth** REF-C derivative already on the box: `refc-base-e1f-junction`, inside the **public** `Sayood/tanitad-rollout-recovery` repo — the report missed it.
And the gate is not imminent: `THOR_ACCESS_BRIEF.md` §4 states **"⛔ No validation data yet. The val caches live on RunPod pods (currently stopped)"**, and `Project Steering/LOOP_STATE.md:1210` gives the near-term Thor plan as *"torch install → pull v1_modelonly + val19 windows → four-family panel dry run"*. "Thor pulling toward a gate that cannot score them" overstates a real but non-urgent gap.

### ⛔ A4. Executability: every GPU-costed step in §4 is gated on compute the programme does not have today.
`Project Steering/POD_SHUTDOWN_2026-08-02.md` (dated today): *"the PI reported **$3.61 of remaining credit** … Across four A40s that is **under two hours of runway**"*, all four pods being shut down. So RANK 1's re-collect, RANK 3's "~2 GPU-hours", RANK 4's "< 0.5 GPU-hours" are all **not currently executable**. The report never mentions this.

### 🟠 A5. The registry-absence finding is under-scoped by ~6×.
MEASURED by grep over `Project Steering/MODEL_REGISTRY.md` (only hits for `e1b|e1c|clsft|corridor` are md5/commit-hash false positives at lines 784 and 794): **e1b, e1c, e1d, e1e-A, e1e-B and e1f are ALL absent**; §4 stops at §4.4. The report reported one. One of the missing arms (`refc-base-e1f-junction`) is already published to a **public** HF repo and is on Thor — a strictly higher-priority registry gap than e1b's gated repo.

### 🟠 A6. The E1b primary's n is misattributed — 6, not 44.
Report: "corridor-departure@K185 Δ −0.4270 … paired episode-cluster bootstrap, **44 heldout ep**".
`e1b_eval_result.json` → `PRIMARY_junction_corridor_departure_K185/paired_delta_ft_minus_base`: **`n_windows = 6, n_episodes = 6`**. 44 is the val-dir count and is the n for the *guardrail* (967 win / 44 ep). The primary rests on **n = 6**.

### 🟠 A7. The speed_bias "SEPARATED win" reads the sign backwards, and its flagship comparison is UNPAIRED.
The file declares `estimator.orientation = "floor - model; positive = model wins"`. REF-C's `speed_bias_mps` vs CV is **−0.0754 [−0.1435, −0.0137]** — *negative*, i.e. "model loses" under the file's own convention. What is actually true is that |bias| is smaller (0.0209 vs −0.0545); a **signed** bias metric does not admit that orientation at all. Separately, "materially better than the flagship's +0.1911" is a **cross-arm point comparison with no paired delta** — the artifact contains no refc-vs-flagship pairing.

### 🟠 A8. "My precision conclusion is immune" — it is not.
The three precision blocks were captured at different GPU states: `gpu_state_before.util_pct` = **43 (fp32) / 87 (tf32) / 28 (amp16)**. And the "LOSS" ratios carry **no interval**: `decoder_full_steps2` p50 6.6484 (std 0.5983) vs 7.0666 (std 0.7501) — a 0.42 ms gap inside ~1 std. The *direction* (encoder gains, decoder/heads do not) is supportable; **0.94× / 0.87× / 0.76× as decision-grade point ratios is not**.

### 🟠 A9. The headline REF-C ADEs are missing their own route-provenance stamp (the C6 confound).
`taniteval/taniteval/refc_eval.py` return dict (lines 172–194) carries, verbatim:
> *"`route_input_exercised` False means the decoder saw ONE constant command for every window, so it was compared on its marginal — the 07-21 C6 confound. **Every REF-C number published before 2026-07-26 (base 0.4728, XL 0.4714) was collected that way.**"*

Registry §4.3's Eval row independently says `nav=follow`. The report quotes 0.4714/0.4728 clean, while §2 simultaneously leans on `resolve_nav`'s `"produced"` branch — **which is not the path that produced those numbers**.

### 🟠 A10. "4-family panel ❌ NONE" for REF-C-base / REF-C-small — absence found at one location.
`taniteval/results/driving_refc-base-30k.json` (in-repo) `headline` carries LONGITUDINAL surrogates (`speed_mae_mps`, `speed_bias_mps`, `long_abs_2s_m`, `long_signed_2s_m`, `progress_*`) and LATERAL surrogates (`lat_abs_2s_m`, `heading_mae_2s_deg`, `heading_exceed_5deg`, `pathgeom_crosstrack_m`, `curv_sign_agree`), all `episode_cluster_bootstrap` n=881/40 — plus a `refused` register whose distance-keeping reason is **different from the report's**: *"no lead-agent state exists (`lead_state` is a None stub)"* vs the report's *"ingest does not read `obstacle.offline`"*. Same for small in `…/pod-rescue-20260802/pod3/root/taniteval/results/refc-small-30k.json` (`driving.vs_floor_paired`, `longitudinal_regime`). What is genuinely absent for **every** REF-C arm is a `four_families.all_families()` panel and any TACTICAL/STRATEGIC family.

### 🟡 A11. Line-cite error. `nn.MultiheadAttention` cross-attention is `stack/tanitad/refs/refc.py:482` (`self.cross = nn.MultiheadAttention(d, n_heads, batch_first=True)`), not **488**; the second is at **630** (`self.attn`).

### 🟡 A12. O14 "INAPPLICABLE" is over-claimed. The runbook writes O14 as sparsity *"on the predictor"* — the target is absent, but the lever is not: REF-C-base is 287.274 of 292.461 GFLOPs convolution, and 2:4 structured sparsity applies to conv/GEMM. Correct verdict = **RE-AIM to the encoder**, not INAPPLICABLE. O4 / O8 *are* correctly inapplicable (`meta.sequential_steps = 3`, `decoder_passes = 3` — no recursive roll).

### 🟡 A13. Unclassed or mis-classed numbers.
- "the REF-C-base eval historically ran in ~1 minute, 05:18–05:19 UTC" — that is a **minute-resolution timestamp range** in registry §4.3, not a runtime. "< 0.1 GPU-hours" is **ESTIMATED**. (Program memory: *"don't assume slow-eval = slow-compute"*.)
- "~1 h implementation", "~2 GPU-hours", "< 0.5 GPU-hours" — all unclassed → **ESTIMATED**.
- "**0 of 881** 'accelerate' on the flagship" is traceable only to the `four_families.tactical()` **docstring** (code prose). Grep of `MODEL_REGISTRY.md` for `0 of 881` / `accelerate` returns **zero hits** → **INHERITED**, and it is the whole motivation for RANK 2.
- "registry key `refc-v12*`" — no such key in `MODEL_REGISTRY.md` and no `refc-v12` entry in `taniteval/taniteval/registry.py` (whose refc keys are `refc-xl`, `refc-xl-live`, `refc-xl-30k`, `refc-base-30k`, `refc-small-30k`, lines 162–264). It **is** a real TanitEval *results* key (`taniteval/results/driving_refc-v12.json`, `-identity`, `-k16reg`).
- The e1b row's 0.6693/0.4747 are on **967 windows / 44 episodes** of `/workspace/v4run/valcache/physicalai-val-heldout-79d4e3d2d4c6` — a **different val cache** from the canonical 881/40 `physicalai-val-0c5f7dac3b11` in the rows above it. The report does supply the matched control (correct), but the table placement invites the cross-set comparison.
- **No 0.4271-vs-19-episode violation found** — the report never quotes either number.

### ✅ CONFIRMED (survived; re-verified against primary artifacts)
All headline ADE/FDE/miss/CI/estimator for XL and base against **in-repo raw JSON** `taniteval/results/driving_refc-{xl,base}-30k.json`; small's 0.5260717/1.1115/0.171396 against `…/pod-rescue-20260802/pod3/root/taniteval/results/refc-small-30k.json#full_set`. Params 251,932,584 / 104,191,577 / 54,690,001 (registry §4 preset table). Δ(base−XL) +0.0013 [−0.0281, +0.0316] NOT sep; 2.42× params / 2.20× encoder (§4.3). Oracle gap ~92 % irreducible / 47 arms / ≤8.4 % / v1.0 0.0 % / GT-perfect speed-matcher **1.1236** / anchors = 0.048 M of **buffers** (registry 1258–1270, 1477). E1c 17-point frontier, primary 15/17, guardrails 0/17, intersection empty (`e1c_frontier_result.json#VERDICT`). E1b Δ −0.4270 [−0.6838,−0.1648]; ADE 0.6693 vs 0.4747, Δ +0.1947 [+0.1415,+0.2522]; 13,732,945 / 90,458,632; verdict BOUND. **Every code line-cite except A11**: `hierarchy.py:431-433` (hard skip), `:478`, `:486`; zero `refc` matches in `hierarchy.py`; `RefCModel` has no `tactical_policy`/`strategic_policy`; `four_families.py:120-126, :166, :206, :254, :313`; `loaders.py:129`; `refc_eval.py` `collect`@109, produced-block 94–104, return 172–194; `refc.py:691, 707, 733, 736, 823, 824, 591-593`; `run_four_families_vs_floors.py:37` (`ff.DT_S = 0.5`). All stage p50s. FLOP split 287.274/292.461 = 98.2 %; aux heads 0.0859 ms = 0.394 % of 21.7785. MHA-fastpath/opset-17 rel-err 0.726 (runbook 56–58, 63, 66–67, 90–95) and 0.86× predictor (108, 113). O1–O14 as listed (148–171). cross_mae vs CTRV Δ +0.0294 [+0.0149,+0.0433] SEP; CTRV curvature floor better (0.008967 vs 0.012138); TACTICAL/STRATEGIC UNAVAILABLE with the exact reason strings.

---

## PART B — CORRECTED REPORT

# REF-C OPTIMISATION — stream report (verified 2026-08-02)

## 1. REF-C variants

| variant | key | params | last MEASURED eval (estimator, n) | 4-family status |
|---|---|---|---|---|
| **REF-C-XL** §4.1 | `refc-xl-30k` | **251,932,584** ᴹ | ADE@2s **0.4714** [0.3896,0.5556]; FDE 1.0061; miss 0.1419; TMS-ol 0.2135 — episode-cluster bootstrap B=2000, **881 win / 40 ep** ᴹ `taniteval/results/driving_refc-xl-30k.json` ⚠️ **C6-confounded: `nav=follow`, route input never exercised** | 🟡 partial (see §1b) |
| **REF-C-base** §4.3 | `refc-base-30k` | **104,191,577** ᴹ | ADE@2s **0.4728** [0.3835,0.5699]; FDE 1.0031; miss 0.1419; TMS-ol 0.1957 — same ᴹ `…/driving_refc-base-30k.json` ⚠️ same C6 stamp | 🟡 partial |
| **REF-C-small** §4.2 | `refc-small-30k` | **54,690,001** ᴹ | ADE@2s **0.5260717**; FDE 1.1115; miss 0.171396 — `…/pod-rescue-20260802/pod3/root/taniteval/results/refc-small-30k.json#full_set` ᴹ. *(No `driving_refc-small-30k.json` in-repo; registry row 1691 has no `<!-- src -->` pointer.)* | 🟡 partial |
| **REF-C v1.0 / v1.2** re-rank | *no registry §; TanitEval results keys `refc-v12`, `-identity`, `-k16reg`* | rescorer only | v1.0 0.4714 (λ=0 best, **0.0 %** recovered); v1.2 **0.46251** vs 0.47144, Δ +0.00893 [−0.0062,+0.0250] NOT sig ᴹ (registry 1286-7) | ❌ none |
| **`refc-base-e1b-clsft`** ⚠️ unregistered | — | 13,732,945 trainable / 90,458,632 frozen ᴹ | primary corridor-dep@K185 Δ **−0.4270** [−0.6838,−0.1648] SEP — **paired episode-cluster bootstrap, n = 6 windows / 6 episodes**; guardrail open-loop ADE@2s **0.6693 vs base 0.4747**, Δ **+0.1947** [+0.1415,+0.2522] SEP-WORSE, **967 win / 44 ep on `physicalai-val-heldout-79d4e3d2d4c6` — a DIFFERENT val cache from the 881/40 rows above** ᴹ. Verdict **BOUND** | ❌ none |
| `refc-base-e1c/e1d/e1e-A/e1e-B/e1f-junction` ⚠️ all unregistered | — | — | E1c: 17-point frontier, primary **15/17**, guardrails **0/17**, intersection EMPTY, verdict BOUND ᴹ (`e1c_frontier_result.json#VERDICT`). E1f: primary 0/4, guardrails 0/4, BOUND ᴹ | ❌ none |
| REF-C closed-loop §4.4 | base+XL | — | base 6/12 pass 0.345; XL 5/12 0.246 — ⚠️ **RECONSTRUCTION-OOD CONFOUNDED** (RETRACTION_LOG C6) ᴹ | ❌ none |

ᴹ = MEASURED. **Registry defect (corrected scope): SIX REF-C fine-tune arms are absent from `MODEL_REGISTRY.md` — e1b, e1c, e1d, e1e-A, e1e-B, e1f.** §4 ends at §4.4. Highest-priority of these is **`refc-base-e1f-junction`, which is in a PUBLIC HF repo (`Sayood/tanitad-rollout-recovery`) and already on Thor** (`THOR_ACCESS_BRIEF.md:69`) — an unregistered, publicly downloadable arm. `Sayood/flagship-v4.2b` is not REF-C (registry §1.5.x).

### 1b. What the four-family surface actually is today
- **`four_families.all_families()` panel: exists for REF-C-XL only**, in `…/2026-08-02-ctrv-floor/raw/four_families_vs_floors.json` (881 win / 40 ep, paired episode-cluster bootstrap B=2000, ⚠️ **`cadence_s = 0.5`** — `run_four_families_vs_floors.py:37` sets `ff.DT_S = 0.5` vs the module default 0.1, so speeds/accels are 0.5 s steps and are **not** comparable to a dense-path run).
  - **LONGITUDINAL** (partial): speed_mae 0.4545 (vs CV Δ +0.0132, NOT sep); speed_bias **+0.0209**; along_mae 0.4168; along_final_bias 0.0511. `distance_keeping` **UNAVAILABLE, n=0**. ⚠️ The signed-bias-vs-floor delta (−0.0754 [−0.1435,−0.0137], flagged `separated`) **cannot be read as a win** — the file's orientation is *"floor − model; positive = model wins"* and this is negative; the true statement is only that |REF-C's bias| (0.0209) < |CV's| (0.0545). **No paired refc-vs-flagship delta exists in this artifact.**
  - **LATERAL** (2 of 5 with intervals): `cross_mae_m` **0.1310 m**, beats CTRV Δ **+0.0294 [+0.0149,+0.0433] SEP**; `cross_bias_m` −0.0000, not sep. ⛔ `heading_mae_deg` 1.1484°, `yaw_rate_mae_degps` 1.8216, `curvature_mae_1pm` 0.012138 are **point estimates with INTERVALS REFUSED** by the artifact's own `refused_metrics` (per-window form disagrees with the module by 5.12e-2 / 8.68e-2 / 1.08e-2 > tol 1e-3). Directionally, the **CTRV floor beats REF-C on curvature (0.008967 vs 0.012138)** — but that comparison has no admissible interval.
  - **TACTICAL / STRATEGIC**: UNAVAILABLE, n=0, reason = *"decisions not present in the scored pass (missing `[maneuver_pred, maneuver_gt]` / `[route_pred, route_gt]`)"*.
- **Partial family surface for base and small exists elsewhere** (this is where the original report over-claimed absence): `taniteval/results/driving_refc-base-30k.json#headline` carries longitudinal (`speed_mae_mps`, `speed_bias_mps`, `long_abs_2s_m`, `progress_*`) and lateral (`lat_abs_2s_m`, `heading_mae_2s_deg`, `heading_exceed_5deg`, `pathgeom_crosstrack_m`, `curv_sign_agree`) with episode-cluster CIs, plus a `refused` register. Its distance-keeping refusal reason differs from `four_families`': *"no lead-agent state exists (`lead_state` is a None stub)"* — **two independent instruments, two different root causes; both must be closed.**

⇒ **Correct statement: no REF-C arm has a complete four-family panel, and NO REF-C arm has ANY tactical or strategic family at all.**

## 2. Exact gap list for `four_families.all_families()` on REF-C — VERIFIED

Path: `taniteval/taniteval/refc_eval.py` (`collect`, **line 109**) → `taniteval/taniteval/loaders.py:129` (`elif arch == "refc"`) → keys at `registry.py:162–264`.

| family | status | blocker | fix |
|---|---|---|---|
| LATERAL | ✅ computes | none | close the per-window/module disagreement so the intervals stop being refused |
| LONGITUDINAL | 🟡 all but one | `distance_keeping` hardcoded UNAVAILABLE at `four_families.py:120-126` | corpus-wide ingest work item, **not** REF-C-specific |
| TACTICAL | ❌ | `_decision_family(win,…,"maneuver_pred","maneuver_gt")` (`four_families.py:166`); `collect`'s return (lines 172–194) lacks those keys | model already computes it: `refc.py:823` → `"maneuver_logits"` [B,5]; GT = `refb_labels.classify_maneuver` (the exact call `hierarchy.py:478` makes); classes `refb.MANEUVER_CLASSES` |
| STRATEGIC | ❌ | needs `route_pred`/`route_gt`; same omission | `refc.py:824` → `"route_logits"` [B,3]; `refc_eval.resolve_nav` (94–104) already argmaxes them for `nav_mode="produced"` and discards the class. GT = `refb_labels.route_target(nav_command(...))` (`hierarchy.py:486`); classes `refb.ROUTE_CLASSES` |

⛔ The `hier=` route is structurally CLOSED to REF-C — verified at two locations: (1) `hierarchy.py:431-433` returns `{"skipped": …}` unless the model has **both** `tactical_policy` and `strategic_policy`; (2) `RefCModel` (`refc.py:691+`) has neither — it has `maneuver_head` (733), `route_head` (736), `law_head`, `speed_cls`, `StrategicCtx` (707). `grep -ci refc hierarchy.py` = **0**. ⇒ REF-C **must** use the `_decision_family` fallback: a **window-dict plumbing gap, not a modelling gap**. Cost of both aux heads: **0.0859 ms p50 = 0.394 % of the 21.7785 ms tick**, already inside the forward pass.

⚠️ Two caveats the original report omitted: (a) `collect` does not retain `poses`/`last` per window, so the GT labels must be minted **inside** the batch loop where `ep.poses` and `last` are live; (b) `all_families` requires pred `[n,H,2]` and the current dump is the sparse 4-waypoint 0.5 s surface — the panel must carry the cadence stamp or it is not comparable to a dense run.

## 3. Thor runbook O1–O14 vs REF-C

**Per-stage precision, MEASURED on REF-C-base** (`eff_refc-base-30k.json`, A40, p50 ms):

| stage | fp32 | tf32 | amp16 | fp32→amp16 |
|---|---|---|---|---|
| `encode_window_8frames` | 19.7595 | 13.9396 | **8.2954** | **2.38×** |
| `decoder_full_steps2` (2 denoise) | 6.6484 | 6.2919 | 7.0666 | 0.94× |
| `decoder_classifier_steps0` | 2.3492 | 2.1835 | 2.4914 | 0.94× |
| `law_head` | 2.1276 | 2.4304 | 2.8032 | 0.76× |
| `aux_heads_maneuver+route` | 0.0859 | 0.1034 | 0.0986 | 0.87× |

⚠️ **These ratios carry no interval and the blocks are not state-matched** — `gpu_state_before.util_pct` = 43 / 87 / 28 for fp32/tf32/amp16, and the decoder gap (0.42 ms) sits inside ~1 std (0.60 / 0.75). **Admissible claim: DIRECTIONAL — precision helps REF-C's encoder and does not help its decoder or heads.** ⛔ Not admissible: "0.94× / 0.87× / 0.76×" as decision-grade numbers, and not "the SAME answer" as Thor (Thor's encoder gain is 6.76×, REF-C's 2.38× — same sign, different magnitude). The runbook's own predictor figure is **0.86×** (`THOR_DEPLOYMENT_RUNBOOK.md:108,113`).

**Mechanism (MEASURED, same artifact):** 292.461 GFLOPs total, **287.274 convolution (98.2 %)**; the whole non-conv side is `addmm 5.002 + SDPA 0.151 + mm 0.035 = 5.188 GFLOPs (1.8 %)` — note this covers law_head and aux heads too, not the decoder alone, and the counter **excludes elementwise + norm**. The decoder burns 6.65 ms for a small FLOP share ⇒ **launch-bound**, so the lever is graph capture / fusion, not precision.

✅ **No stage-sum anomaly exists.** The artifact publishes `isolated_stage_sum_ms 28.9578`, `= 132.8 % of plan_step`, with the note *"the stages overlap CPU-launch with GPU-execute inside the full step (launch-bound) … treat shares as attribution, not an exact partition"*, and `encoder_pct 90.7` matching `MODEL_REGISTRY.md:1504` exactly. **Nothing to reconcile; no re-measure warranted.**

| verdict | items |
|---|---|
| ✅ applies | **O1** four-family gate (blocking, doubly so for REF-C per §2) · **O2** TRT encoder · **O3** end-to-end tick · **O5/O6** INT8/NVFP4 *encoder-only* · **O7** nvpmodel · **O10** resolution · **O11** multi-camera · **O12** Orin · **O13** DLA (98.2 % conv) |
| ⛔ inapplicable | **O4** (K 20→10) and **O8** (one engine for the 20-step roll) — REF-C has no recursive roll (`meta.sequential_steps = 3`, `decoder_passes = 3`) |
| 🔁 re-aim | **O9** → REF-C already decodes 128–256 anchors in parallel; its O8 analogue is one CUDA graph over the 3 decoder passes. **O14** → the *predictor* target does not exist, but 2:4 sparsity applies to REF-C's 98.2 %-conv **encoder**; re-aim, do not close |

Binding for any REF-C export: `torch.backends.mha.set_fastpath_enabled(False)` — REF-C's cross-attention is `nn.MultiheadAttention` at **`refc.py:482`** (second at 630), exactly the op opset 17 exported silently wrong at rel-err **0.726** (runbook 56–58, 63, 66–67).

## 4. Ranked next steps — re-costed against actual compute

⚠️ **Compute premise (MEASURED, `Project Steering/POD_SHUTDOWN_2026-08-02.md`, today): $3.61 credit ≈ under two hours across four A40s; all pods shutting down. Thor has no torch and NO VALIDATION DATA (`THOR_ACCESS_BRIEF.md` §4).** Every GPU item below is therefore **BLOCKED-ON-COMPUTE**, and the 0-GPU items are the executable ones.

**⭐ RANK 1 — R1a (0 GPU, EXECUTABLE NOW): implement the 4 keys in `refc_eval.collect`.** `maneuver_pred = out["maneuver_logits"].argmax(-1)`, `route_pred = out["route_logits"].argmax(-1)`, GTs from `refb_labels.classify_maneuver` / `route_target(nav_command(...))` minted inside the batch loop. Land it staged; it is pure plumbing. **R1b (BLOCKED): the re-collect over 881 windows** — needs a GPU *and* the val cache, which is on stopped pods. Cost **ESTIMATED**, not measured (the "05:18–05:19 UTC" in registry §4.3 is a minute-resolution timestamp, not a runtime). **Falsifier:** `_decision_family` still returns UNAVAILABLE after the patch ⇒ heads are unpopulated at eval time and the gap is architectural.

**RANK 2 — register the six missing REF-C arms in `MODEL_REGISTRY.md` (0 GPU, EXECUTABLE NOW).** `e1f-junction` first (public HF + on Thor), then e1b (BOUND, ADE 0.6693 vs 0.4747), e1c (BOUND, 15/17 primary vs 0/17 guardrails), e1d, e1e-A, e1e-B. Each with its BOUND verdict and its n.

**RANK 3 — re-verify "0 of 881 accelerate" from a raw eval JSON before it decides anything (0 GPU).** It is currently **INHERITED** (traceable only to the `four_families.tactical()` docstring; absent from `MODEL_REGISTRY.md`). It is the entire motivation for the `never_predicted` read on REF-C's tactical head. If it holds, the REF-C read still reframes the hierarchy thesis: REF-C's independent 5-way `maneuver_head` also drives anchor selection (`refc.py:591-593`, `maneuver_to_anchor` — **conditional on the graft being built and on `refc.py:809` falling through to `man_logits`**), so a collapse there makes the defect corpus/label-level, not flagship-specific.

**RANK 4 — close the LATERAL interval refusal (0 GPU).** Three of five lateral metrics currently have no admissible CI. Fix the per-window reimplementation in `run_four_families_vs_floors.py` until it agrees with `four_families.lateral` to < 1e-3, then republish with intervals. Until then, no lateral verdict on REF-C is admissible beyond cross-track.

**RANK 5 — stamp the C6 route provenance on every published REF-C number (0 GPU).** `refc_eval.collect` already carries the warning in its own output; the registry rows and the leaderboard do not. Add it wherever 0.4714 / 0.4728 appear.

**RANK 6 (BLOCKED-ON-COMPUTE) — O2, encoder-only, on Thor.** REF-C is 98.2 % conv and ~90.7 % encoder-share of the tick; all arithmetic levers go on the encoder, the decoder is excluded by §3. Cost **ESTIMATED ~2 GPU-hours**. Falsifier: TRT engine ≤ bf16 autocast ⇒ keep autocast, close.

**⛔ STRUCK from the original report:** "re-measure `eff_refc-base-30k` to settle the 28.5 vs 21.78 ms conflict" — no conflict exists (§3).

**⛔ Do NOT fund** (MEASURED, registry 1258–1270, 1477): selection / re-ranking on REF-C — across **47 trained arms** the oracle gap is **~92 % irreducible**, a learned re-scorer recovers **≤ 8.4 %** on its own training data, hand-written cost re-rank recovers **0.0 %** (pure cost −171 %), and a target-speed selection term is **REFUTED** (GT-perfect speed-matcher scores **1.1236**, worse than baseline). Nor widen the encoder: 2.42× params / 2.20× encoder bought **+0.0013 m [−0.0281, +0.0316], NOT separated**. The fan lever is **anchor-vocabulary width** (0.048 M of *buffers*, not parameters).

---

**Key paths (absolute):**
`G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/taniteval/taniteval/refc_eval.py` (gap: lines 172–194; `collect`@109; produced-branch 94–104) · `.../taniteval/taniteval/four_families.py` (:120-126, :166, :206, :254, :313) · `.../taniteval/taniteval/hierarchy.py` (:431-433 hard skip; :478, :486 label calls) · `.../taniteval/taniteval/loaders.py:129` · `.../taniteval/taniteval/registry.py:162-264` · `.../stack/tanitad/refs/refc.py` (:482 MHA, :691, :707, :733, :736, :591-593, :823-824) · `.../Project Steering/MODEL_REGISTRY.md` §4.1-4.4 (1185–1562), leaderboard rows 1689–1691, oracle-gap block 1258–1270, anchors 1477, encoder-share 1504 · `.../Project Steering/POD_SHUTDOWN_2026-08-02.md` · `.../TanitAD Research Hub/Production & Optimization/THOR_DEPLOYMENT_RUNBOOK.md` (:56-58, :63, :90-95, :108, :113, :148-171) · `.../THOR_ACCESS_BRIEF.md` (:69-70, §4) · `.../TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-08-02-ctrv-floor/raw/four_families_vs_floors.json` + `run_four_families_vs_floors.py:37` · `.../taniteval/results/driving_refc-{xl,base}-30k.json` · `.../stack/experiments/pod-rescue-20260802/pod3/root/taniteval/results/{eff_refc-base-30k.json,refc-small-30k.json}` · `.../TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-25-e1b-failure-gated-clsft/e1b_eval_result.json` · `.../incoming/2026-07-26-e1c-heldout-gated-clsft/e1c_frontier_result.json` · `.../incoming/2026-07-28-e1f-junction-buffer/E1F_RESULT.md`

**UNVERIFIED (could not close from the repo):** whether the Thor HF pulls listed as "pulling" have completed; the intended deployment target of `refc-base-e1b-clsft` on Thor (no doc states one — the ACCESS_BRIEF only lists the directory).

**Read-only respected:** nothing modified, no training launched, no pod touched.