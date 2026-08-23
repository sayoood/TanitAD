# `planner_beats_cv` — the verdict is OPEN-LOOP, the banked dump is CLOSED-LOOP

**Date:** 2026-08-18 · **Branch:** `agent/arch-inf-20260803` · **HEAD at start:** `4d50bcd`
**Compute:** CPU only. No GPU used, no model loaded, no pod contacted, **Thor untouched**.
**Estimator:** `episode_cluster_bootstrap` / `paired_episode_cluster_bootstrap`
(`taniteval/ci.py`), `n_boot=2000`, `seed=0`. **Every point estimate is the `full_set` mean.**
⛔ `overlapping_holdout_se` appears **only** in the reproduction gate (§2) and decides nothing.

---

## 0. THE ANSWER

> ### The open-loop verdict `planner_beats_cv` remains **UNDECIDED**. It cannot be settled from banked data, and the reason is sharper than "we didn't have the file".
>
> ### But the question it was standing in for **IS now settled, at the primary tier, and the answer is NO — decisively.**
> **Closed-loop (T1), paired on the same 221 windows: the planner is WORSE than constant
> velocity by `+0.2585 m [+0.0869, +0.4309]`, CI-separated, `p(δ>0) = 0.9975`.**

| verdict | tier | status after this task |
|---|---|---|
| `planner_beats_cv` (the artifact's field) | **open loop**, n=881/40 ep | ⛔ **STILL UNDECIDED** — needs the ~400 s GPU re-drive (§3) |
| **closed-loop planner vs CV** *(new, never computed)* | **T1**, n=221/20 ep | ✅ **planner LOSES, CI-separated** (§4) |
| `G1_pass` / `…separated` | open loop | unchanged (flip needs −73.6 %); corrected arms **independently reproduced** here |
| `G4_pass` | closed loop | unchanged: `0.9799 [0.7456, 1.2312]` vs `1.7318` — passes, CI-separated |
| `beats_head` ×9 + `beats_head_all` | 8-ep sweep | **not estimator-reachable** — a **4.686× ratio** (§4b) |

⚠️ **The brief's proposed zero-GPU shortcut does not reach this verdict**, and that is the
single most important finding here — see §1. It was a reasonable hypothesis; it is false.

---

## 1. ⛔ THE DEFINITIONAL FINDING — `planner_beats_cv` IS NOT THE CLOSED-LOOP COMPARISON

Established from source, not from prose:

```
taniteval/taniteval/planner_p2.py:621
    "planner_beats_cv": bool(boot["plan"]["mean"] < boot["cv"]["mean"]),
```

That line lives inside **`analyze_openloop`** (`:555`). Its `boot` is built at `:561` from
`ade = {k: _ade2(col[f"{k}_wp"], col["gt_wp"]) …}` over `collect_openloop`'s (`:361`)
`plan_wp` / `cv_wp` — the **OPEN-LOOP** arms, **n = 881 windows / 40 episodes, stride 8**.
The artifact confirms it: the field sits in the `open_loop` block, beside `n_windows: 881`.

The banked dump the brief pointed at holds the **other** collection:

| file | contents | n |
|---|---|---|
| `raw_windows/p2win_flagship-30k.pt` | `closed_bike`, `open_grnd`, `cv`, `gt`, `speed`, `head_deg`, `eid` | **221 win / 20 ep, stride 16** |
| `raw_windows/clwin_flagship-30k.pt` | head-driven closed loop + `plan_direct` (the G1 head arm) | 881 / 40 |
| `taniteval/results/windows_flagship-30k.pt` | `pred` (operative, true actions), `cv`, `gt` | 881 / 40 |

⇒ **`p2win` is the closed-loop collection. It does not contain the open-loop CEM planner.**

### 1.1 Absence, established at more than one location

Per CLAUDE.md rule 2, "not banked" was **not** asserted from one probe:

1. `taniteval/results/*.pt` → `pred`/`cv`/`gt` only — `pred` is the operative rollout under
   **true actions**, not the CEM search.
2. `…/2026-07-26-closedloop-artifact-rerun/raw_windows/*.pt` → closed-loop (`p2win`) and the
   head-driven loop (`clwin`).
3. **An exhaustive walk of EVERY `.pt` in the repo** (`code/scan_pt.py`, all 4 900+ files
   opened and keyed). **No open-loop CEM planner arm exists at n=881 anywhere.**

This independently confirms `JACK_IN_GATES.md` §3.1, which reached the same conclusion from
three probes. The planner arm is the *one* input that was never dumped.

---

## 2. REPRODUCTION GATE — the banked windows ARE the objects the gate ran on

Recomputing the **banned** estimator from the banked windows, via
`planner_p2._jack_scalar` itself (not a reimplementation), over the harness's own
`gates.split_by_episode` structure (seeds 0–7, `val_frac 0.2`):

| arm | published | recomputed (banned) | |
|---|---|---|---|
| `cv_ade2s` | 0.7704 ± 0.1704 | **0.7704 ± 0.1704** | ✅ **bit-exact** |
| `open_grnd_ade2s` | 0.4244 ± 0.0573 | **0.4244 ± 0.0573** | ✅ **bit-exact** |
| `closed_bike_ade2s` | 1.0377 ± 0.2022 | 1.0375 ± 0.2023 | ✅ drift **0.0193 %** |

`cv` is **model-free**, so its bit-exact reproduction proves the window set and the split
structure are the same objects the 2026-07-19 gate ran on. The `closed_bike` residual is the
**unseeded CEM** — it matches the 0.019 % recorded in `planner_p2_G4.CORRECTED.json` to three
significant figures, and it is exactly the non-determinism `fa4b3d1` removed. It is **not** a
reproduction failure and is labelled as such in the JSON (`gate_pass: true`).

---

## 3. THE OPEN-LOOP VERDICT — why it is UNDECIDED, stated exactly

Every arm that **is** banked, re-decided here (and reproducing `JACK_IN_GATES.md` to 4 dp,
which is an independent check of that work):

| arm | BANNED | **CORRECTED full_set** [lo, hi] | point error |
|---|---|---|---|
| `constant_velocity` | 0.8248 | **0.8377** [0.6234, 1.0716] | −1.540 % |
| `operative_rollout_trueA` | 0.4522 | **0.4271** [0.3675, 0.4871] | **+5.877 %** |
| `tactical_head` | 3.1501 | **3.3839** [2.8336, 3.9722] | **−6.909 %** |
| **`planner` (CEM, open loop)** | **0.8929 ± 0.1143** | ⚠️ **NOT BANKED — unknown** | — |

**The flip arithmetic, MEASURED:**

* Corrected CV floor: **0.8377** — *higher* than the banned 0.8248, so the correction moves
  the comparison **toward** the planner.
* The planner's corrected mean must land **below 0.8377** to flip.
* From 0.8929 that is a required downward correction of **6.589 %**.
* Measured envelope **on this exact window set, this exact 8-split structure**:
  **−6.909 % … +5.877 %** (three arms).
* Programme-wide 27-arm envelope: **−6.67 % … +11.69 %**.

⇒ **6.589 % sits ABOVE the local upper edge (5.877 %) but well INSIDE the programme-wide
envelope (11.69 %).** That is the definition of undecided. It is materially unlike G1, whose
flip needs **−73.6 %** — about 11× anything ever measured.

### 3.1 Why no bound closes this without the re-drive

The banned statistic is a weighted mean `Σ w_i v_i` with
`w_i = (1/S) Σ_s 1[i ∈ V_s]/|V_s|`. On this window set it gives **7 of the 40 val episodes
weight exactly 0** (`JACK_IN_GATES.md` §5). The windows in those episodes are therefore
**entirely unconstrained** by the published `heldout` mean — the `full_set` mean is
**unbounded above** by it. The generic bound `|correction| ≤ Σ|d_i| · spread`
(`d_i = w_i − 1/N`) is far too loose to matter at a 6.6 % margin.

⇒ **There is no arithmetic path from the published number to the answer.** The planner's
per-window ADE must be produced. Any claim in either direction before then is inadmissible —
which is exactly what C91 says.

---

## 4. ⭐ NEW — THE CLOSED-LOOP PLANNER-vs-CV COMPARISON, PAIRED (T1)

**This comparison had never been made.** The published G4 compared the planner against the
**head**, never against **constant velocity** — so the closed-loop form of the question
`planner_beats_cv` asks was simply never asked.

**Tier: T1 (action-closed loop — the model conditioned on its OWN actions).** Per
`EVAL_DOCTRINE.md` this is the **PRIMARY tier for a capability claim**; the open-loop block
where `planner_beats_cv` lives is the diagnostic tier, not driving performance.

| arm | **corrected full_set** [lo, hi] | n |
|---|---|---|
| planner `closed_bike` | **0.9799** [0.7456, 1.2312] | 221 / 20 ep |
| `constant_velocity` | **0.7214** [0.4680, 1.0360] | 221 / 20 ep |
| `open_grnd` (operative, true actions) | **0.4063** [0.3293, 0.4907] | 221 / 20 ep |

**Paired episode-cluster bootstrap, same windows, same episodes:**

| comparison | delta | 95 % CI | separated | p(δ>0) |
|---|---|---|---|---|
| **planner − CV** | **+0.2585 m** | **[+0.0869, +0.4309]** | ✅ **yes** | **0.9975** |
| operative − CV | −0.3151 m | [−0.5679, −0.1189] | ✅ yes | 0.0000 |

> ⇒ **`closedloop_planner_beats_cv = FALSE`, and unlike the open-loop field this one is
> CI-SEPARATED and paired.** The CEM planner is **35.8 % worse** than constant velocity in
> closed loop. The operative WM under *true* actions beats CV comfortably — so the loss is in
> the **action search**, not in the world model.

⚠️ **This does NOT decide the open-loop field.** They are different window sets (221/20 vs
881/40), different strides (16 vs 8), and different regimes. It is reported as its own result,
at its own tier, not as a substitute. But it is the one a driving claim should cite.

### 4b. Every other verdict in the artifact — enumerated FROM the artifact

Per C91, the inventory was taken by walking the JSON, not by reading a headline:
**14 boolean instances across 6 distinct verdict names.**

| name | instances | status |
|---|---|---|
| `separated` (G1 delta) | 1 | unchanged; flip needs −73.6 % |
| `G1_pass` | 1 | unchanged |
| `planner_beats_cv` | 1 | ⛔ **UNDECIDED** |
| `G4_pass` | 1 | unchanged — `0.9799` (hi `1.2312`) < `1.7318`, CI-separated |
| `beats_head` | **9** | not estimator-reachable |
| `beats_head_all` | 1 | not estimator-reachable |

⚠️ **C91 itself says "FIVE".** The exact count is **6 distinct names / 14 instances** — C91
collapsed the 9 identical `beats_head` grid entries and their `beats_head_all` roll-up. The
substance of C91 is untouched (`planner_beats_cv` is the one that matters), but *the
correction to an imprecise inventory was itself slightly imprecise* — the same root-cause
class, one level down. **The count is now reproducible from `code/redrive_planner_vs_cv.py`
rather than asserted.**

**`beats_head` — settled by margin, so the estimator is irrelevant.** Planner ADE@2s ranges
**0.6467–0.6689** against a constant head **3.1342**: a **4.686× ratio**. Flipping needs the
planner to rise **+368.6 %** or the head to fall **−79.4 %** — roughly **30×** the widest
measured estimator error. ⚠️ **UNVERIFIED:** the `_sweep` that produced these rows is *not* in
this repo's git history (`git log -S beats_head -- taniteval/taniteval/planner_p2.py` returns
nothing — the P2 harness was stranded on a pod before it landed), so which estimator produced
them is unknown. It does not matter at a 4.7× margin, and that is why it is stated as a margin
rather than a re-decision.

---

## 5. THE FOUR METRIC FAMILIES — per family, never pooled

⛔ An ADE-only answer is incomplete. Both tiers below; `_` marks a family that genuinely
cannot be computed, **with its reason and its n**.

### 5.1 Closed loop (T1), n = 221 / 20 episodes

| arm | LONG `long_rmse@2s` | LONG `speed_err@2s` | LONG `speed_bias` | LONG `long_frac` | LAT `crosstrack@2s` | LAT `heading@2s` | LAT `curv`¹ | LAT `yawrate`¹ |
|---|---|---|---|---|---|---|---|---|
| **planner** | **1.9062 m** | **0.9431 m/s** | **+0.2737 m/s** | 0.4851 | **1.9637 m** | 7.676° | 0.01340 1/m | 4.561 °/s |
| constant velocity | 1.6705 m | 0.7607 m/s | −0.0995 m/s | 0.4879 | 1.7115 m | 6.734° | 0.01230 1/m | 3.607 °/s |
| operative (true actions) | 1.0410 m | 0.7602 m/s | +0.4587 m/s | 0.9166 | 0.3141 m | 7.080° | 0.00330 1/m | 1.311 °/s |

> ⭐ **THE PLANNER LOSES TO CV ON BOTH COMPUTABLE FAMILIES — INCLUDING THE ONE IT IS DESIGNED
> FOR.** The P2 cost is explicitly *"LONGITUDINAL + comfort + progress only"*
> (`planner_p2.py:44-51`), yet closed-loop it is **worse longitudinally** than constant
> velocity (1.9062 vs 1.6705 m; speed error 0.9431 vs 0.7607 m/s) **and** worse laterally
> (1.9637 vs 1.7115 m). Its speed bias is **+0.2737 m/s (over-fast)** where CV is −0.0995.
> The lateral loss was predicted by the module's own honest scope note; **the longitudinal
> loss was not**, and it is the more informative half.

### 5.2 Open loop, n = 881 / 40 episodes — the tier `planner_beats_cv` lives at

| arm | LONG `long_rmse@2s` | LONG `speed_err@2s` | LONG `long_frac` | LAT `crosstrack@2s` | LAT `curv`¹ | LAT `yawrate`¹ |
|---|---|---|---|---|---|---|
| constant velocity | 1.7014 m | 0.7874 m/s | 0.4134 | 2.0268 m | 0.01100 1/m | 3.683 °/s |
| operative (true actions) | 1.0420 m | 0.7903 m/s | 0.8933 | 0.3601 m | 0.00320 1/m | 1.431 °/s |
| tactical head | 7.1292 m | 2.7863 m/s | 0.9239 | 2.0455 m | 0.01410 1/m | 2.958 °/s |
| **planner (CEM)** | ⚠️ **ABSENT — not banked** | | | | | |

**This table is the re-drive's deliverable.** The missing row is precisely the four-family
profile that would close both `planner_beats_cv` and G1's fourth arm.

### 5.3 The families that cannot be computed — reason and n, per family

| family | computable | n | reason |
|---|---|---|---|
| **LONGITUDINAL** — target speed | ✅ | 221 / 881 | full |
| **LONGITUDINAL** — distance-keeping (headway/time-gap/TTC) | ❌ | 221 / 881 | Needs a **lead-agent track**. These dumps bank ego waypoints only, and the P2 cost itself **skips the gap term for the same reason** (`planner_p2.py:36-38`: *"no lead-agent labels in our front-cam+pose data"*). `obstacle.offline` (3D agent tracks, 97.44 % coverage) is a **pod-side join** and was not part of this eval. |
| **LATERAL** | ✅ | 221 / 881 | full — but see ¹ |
| **TACTICAL** | ❌ | 221 / 881 | Two independent reasons. (a) No manoeuvre logits are banked. (b) **The CEM planner emits no manoeuvre class at all** — it searches a continuous action sequence (`planner_p2.py:280`), so there is no discrete decision to score. The class exists only for the 5-way head, i.e. the arm P2 *replaces*. **Work item:** scoring a tactical family for a continuous planner needs a manoeuvre **labeller** applied to both planned and GT paths. Not implemented. |
| **STRATEGIC** | ❌ | 221 / 881 | **Genuinely N/A, not a gap in measurement:** the P2 cost carries **no route or goal term whatsoever** (`planner_p2.py:44-51` — *"no lateral / route / goal term; the strategic goal module is P3"*). There is no strategic output to score. |

¹ **Curvature/yaw are reported on the MOVING subset only (n=205 of 221; n=819 of 881;
criterion: every GT leg > 0.5 m).** ⚠️ **Curvature is undefined for a stopped ego** — it
divides heading change by arc length, and **11 windows have v0 < 0.5 m/s with GT legs down to
0.0000 m**. Unmasked, GT `|κ|` has **mean 34.83 1/m against a MEDIAN of 0.00081 1/m**, with a
max of **23 004 1/m** — the mean is a *division artifact*, not a lateral metric. Heading is
likewise carried forward over degenerate segments (`pathspeed.segment_tangents:68-70`), so the
same mask gates it. **This is masked, not clamped**, and the excluded n is published.
Curvature/yaw are on **0.5 s legs** (the banked resolution) — coarser than the harness's K=20
0.1 s path, and comparable **across arms** here because every arm is measured identically.

---

## 6. WHAT THE RE-DRIVE NEEDS — verified available, NOT executed

The re-drive is **feasible on the dev box** and every input was located and checked:

| input | status |
|---|---|
| **Checkpoint** | ✅ **LOCAL** — `_pod_backup/pod2-2026-08-03/ckpts/flagship4b-speedjerk-30k_ckpt.pt`, 3.31 GB, **`step` field = 29999**, matching the artifact's `ckpt_step`. (Per CLAUDE.md this — not `phase0-30k` — is deployed v1.) |
| **Harness** | ✅ `taniteval.planner_p2` **imports cleanly on this box** (verified by real import, not by file existence); `analyze_openloop` is already migrated to the decision-grade estimator. |
| **CEM seeding** | ✅ `CEM_SEED_DEFAULT = 0` present at `fa4b3d1`; `cem_seed` threads one generator per collection run. |
| **GPU** | ✅ dev-box RTX 4060, ~7.4 GB free. Thor untouched. |
| **Val episodes** | ⚠️ **NOT LOCAL.** Found on HF at `Sayood/tanitad-physicalai-w120-256x640cyl` → **`epcache-256px-phase0/physicalai-val-0c5f7dac3b11/ep_*.pt`** — the *exact* cache `planner_p2._load:854` reads (**not** the `-w120-256x640cyl` sibling, which is a different geometry and would break parity). 600 episodes present; the run needs the **first 40** = **4 697 689 792 B ≈ 4.70 GB**. |

### ⛔ NOT DOWNLOADED — this needs the PI's go-ahead

**A 4.70 GB download is an explicit-permission action and an agent brief is not user consent,
so I did not start it.** Everything else is staged and ready; this is the only gate.

**Parity is provable once it lands, and must be checked before the number is quoted:** re-drive
`collect_openloop`, then verify the emitted `cv_wp` and `gt_wp` reproduce
`taniteval/results/windows_flagship-30k.pt`'s `cv`/`gt` **bit-exactly**. `cv` is model-free, so
that check is a proof of window-set identity — the same argument §2 uses. If it does not
reproduce, the cache is the wrong deployment and the run must be discarded, not reported.

```
# after the 40 episodes are in <valdir>/physicalai-val-0c5f7dac3b11/
python -m taniteval.planner_p2 --arm flagship-30k --episodes 40 --device cuda
#   -> report cem_seed=0 explicitly with the result
#   -> settles planner_beats_cv AND G1's fourth arm in ONE job
```

**Cost:** ~400 s on an A40; expect appreciably longer on the 4060. This one job closes
**both** open items.

---

## 7. EVIDENCE CLASSES

| claim | class |
|---|---|
| `planner_beats_cv` is the open-loop comparison; `p2win` is closed-loop | **MEASURED** — `planner_p2.py:621` + `:555` + `:361`; dump keys read directly |
| No open-loop CEM planner arm is banked anywhere | **MEASURED** — exhaustive `.pt` walk, `code/scan_pt.py` |
| Reproduction gate (cv/open_grnd bit-exact; closed_bike 0.0193 %) | **MEASURED** — `raw/planner_beats_cv_banked_analysis.json` §2 |
| Corrected open-loop arms 0.8377 / 0.4271 / 3.3839 | **MEASURED** — this run; independently reproduces `JACK_IN_GATES.md` §3 |
| Flip needs 6.589 %; local envelope −6.909 %…+5.877 % | **MEASURED** — this run |
| Programme-wide envelope −6.67 %…+11.69 % | **INHERITED** — `CLAUDE.md` / `JACK_BLAST_RADIUS.md`, not re-verified here |
| Closed-loop planner − CV = +0.2585 [+0.0869, +0.4309], separated | **MEASURED** — this run, paired bootstrap, seed 0, n_boot 2000 |
| Four-family tables (both tiers) | **MEASURED** — this run |
| `beats_head` margin 4.686× | **MEASURED** (from the artifact); the estimator behind those rows is **UNVERIFIED** |
| Val cache location + 4.70 GB size | **MEASURED** — HF API listing with `files_metadata=True` |
| Re-drive wall time ~400 s | **INHERITED** — published artifact's `wall_s`, measured on an A40, **not** this box |

---

## 8. DELIVERABLE MANIFEST

| artifact | where it lives | notes |
|---|---|---|
| `REDRIVE.md` (this file) | `repo:TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-08-18-planner-beats-cv-redrive/REDRIVE.md` | staged |
| `code/redrive_planner_vs_cv.py` | same dir, `code/` | the whole analysis; re-runnable on CPU in ~40 s |
| `code/scan_pt.py` | same dir, `code/` | the exhaustive `.pt` walk behind the absence claim |
| `raw/planner_beats_cv_banked_analysis.json` | same dir, `raw/` | every number above, machine-readable |

Nothing lives in only one place — all four are in the repo and staged. No pod was touched; no
worktree was used.

---

## 9. ESCALATION — two items, both needing a decision, neither a "please merge" in a doc

1. ⛔ **The 4.70 GB val-cache download needs the PI's explicit approval.** It is the *only*
   thing between here and a decided `planner_beats_cv`. Source, path and exact byte count are
   in §6. One job then closes both `planner_beats_cv` and G1's fourth arm.

2. ⚠️ **`planner_beats_cv: false` is live in a published artifact and must not be quoted.**
   Until the re-drive it is inadmissible **in either direction** (C91). Anywhere the P2 result
   is cited, the citable closed-loop fact is now §4 — **planner loses to CV by +0.2585 m,
   CI-separated, T1** — which is a *stronger* and better-tiered statement than the open-loop
   field ever was.

3. ⚠️ **`C91` says "FIVE" booleans; the artifact has 6 distinct names / 14 instances** (§4b).
   Minor, but it is the same root-cause class C91 exists to name, so it is worth a one-line
   correction in `RETRACTION_LOG.md` rather than leaving a second imprecise count standing.
