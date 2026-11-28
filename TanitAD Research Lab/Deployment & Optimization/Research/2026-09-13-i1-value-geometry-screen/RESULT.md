<title>I-1 value-geometry screen — the frozen trunk orders time no better than pixels</title>

# I-1 value-geometry screen: a learned metric clears the bar, and a learned PIXEL metric ties it

**2026-09-13 · Research Lab (LAB-RUN-012) · Deployment & Optimization · serves INJECTED row I-1 (top of lane) via D10-1**
⛔ **Tier: none.** A representation diagnostic on frozen features — no rollout, no driving, no four-family eval, and not presented as one.

---

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **F1** | ⛔ **`E-DEP-VGEO-1` FAILED its pre-registered bar → verdict `PARTIAL`.** Raw latent distance on the frozen refcv5-v2 trunk ranks time-to-goal at **ρ = 0.4223 [0.3943, 0.4498]** (short band Δ ≤ 6 s, 139 clips) against a committed bar of **0.50**. | MEASURED `raw/vgeo_screen.json` |
| **F2** | ⛔⭐ **The trunk is WORSE than raw pixels at this.** The 1,440-d pixel floor reads **0.4950 [0.4651, 0.5243]**; paired latent − pixel **−0.0727 [−0.1067, −0.0404]** — separated, in all three bands (long band −0.0628, all −0.0792). A 90.5 M-param encoder's mean-pooled geometry carries *less* temporal ordering than a 64×160 thumbnail. | MEASURED, same file |
| **F3** | ⭐ **Rule Zero lever run in the same pass (`E-DEP-VGEO-2`, POST-HOC, pre-registered before computing): a learned diagonal metric on the 2×5-pooled tokens clears the bar** — **ρ = 0.5637 [0.5293, 0.5961]** on 42 held-out clips. | MEASURED `raw/vgeo_learned_metric.json` |
| **F4** | ⛔⛔ **…and a learned metric on PIXELS ties it: paired +0.0146 [−0.0269, +0.0534].** Same for PCA-64 (+0.042 [−0.010, +0.092]) and raw (−0.007). ⇒ **pre-registered outcome `HEAD-CLEARS-BAR-BUT-PIXELS-TIE`**: goal-distance is recoverable over frozen tokens, but **the trunk adds nothing a thumbnail does not** for this signal. | MEASURED |
| **F5** | ✅ **Every control read its known value** in both runs: C-const **0.0000 [0, 0]**; C-shuf **−0.0041 [−0.0118, 0.0031]** / **−0.0072 [−0.0209, 0.0074]**; C-mut (within-clip row permutation, the deliberate regression) **−0.0019 [−0.0078, 0.0043]** / **−0.0018 [−0.0120, 0.0088]**. The pipeline cannot manufacture ordering. | MEASURED |

**Verdict against D10-1's committed branches:** not *"ρ ≥ 0.5 ⇒ adequate"* (raw fails) and not *"at the shuffled control ⇒ binding"* (0.42 is far from 0). **The honest middle: I-1 is NOT blocked outright, but the frozen trunk is not what makes a distance-shaped value work.**

## 1 · What was run (full design in `SPEC.md` + `PREREG_ADDENDUM_L2_LEARNED_METRIC.md`, hashes pinned in `raw/prereg_pin.json` before computing)

* **Bank:** `C:/Users/Admin/tanitad-caches/bevhead-20260913/tokens/` — the A&I FlyWheel's frozen refcv5-v2 s32 tokens **[27,664, 704, 8, 20]**, ckpt step **40284**, plus the pixel floor on the same rows; **139 clips**, 10 Hz v2ep grid. ⚠️ **Deviation stated in SPEC §1:** D10-1 said *"v7 latents"*; no v7 frame bank exists on the dev box; the refcv5-v2 trunk is the current frozen trunk.
* **Rows:** every 4th row ⇒ **6,954** (a live MM BEV-head arm held the 4060 at 95 % and ~13.5 GB host RAM; this ran CPU-only at below-normal priority, 26 s + 83 s wall).
* **Statistic:** mean over clips of per-clip Spearman ρ(‖z_s − z_g‖, Δ), Δ ∈ [0.4 s, 16 s]; decision band Δ ≤ 6 s.
* **Estimator:** clip-cluster bootstrap, 2,000 resamples. ⚠️ **It answers "another draw of CLIPS" only** — one checkpoint, one deterministic extraction ⇒ blind to training variance (`H-ESTIM-SEED-1`).
* **n and d (probe rule):** VGEO-2 fit **97 clips / scored 42**, clip-disjoint by construction (asserted). Diagonal NNLS fit pairs: 704-d **56,818**; 1,440-d **27,777**; 7,040-d **5,681** ⚠️ **n < d for the winning 7,040-d arm** — it won while underpowered, which cuts *for* the signal being real, but its point estimate is the least stable in the table.

### 1.1 Short band (Δ ≤ 6 s), the decision band

| arm | VGEO-1 (139 clips, raw) | VGEO-2 raw (42 scored) | diag (learned) | PCA-64 (learned) |
|---|---|---|---|---|
| latent 704-d (mean pool) | 0.4223 [0.3943, 0.4498] | 0.4875 [0.4446, 0.5290] | 0.5032 [0.4592, 0.5431] | 0.5238 [0.4898, 0.5568] |
| latent 7,040-d (2×5 pool) | 0.4609 [0.4345, 0.4882] | 0.5193 [0.4785, 0.5596] | **0.5637 [0.5293, 0.5961]** | 0.5239 [0.4882, 0.5598] |
| **pixel 1,440-d** | **0.4950 [0.4651, 0.5243]** | 0.5261 [0.4729, 0.5747] | 0.5491 [0.4968, 0.5983] | 0.4819 [0.4319, 0.5313] |
| paired 7,040 − pixel | −0.0341 [−0.0657, −0.0023] | −0.0068 [−0.0615, 0.0489] | **+0.0146 [−0.0269, 0.0534]** | +0.0420 [−0.0099, 0.0917] |

⚠️ The 42-clip scored subset reads ~0.06 higher than the 139-clip set on the raw arm — a subset effect, which is why every VGEO-2 comparison is paired on the same 42 clips and never against VGEO-1's numbers.
⚠️ **Selection among six learned variants** inflates the "best arm" by construction; the verdict rests on the *paired* test, which ties for every variant, so the selection does not change it.

## 2 · Five-dimension analysis

| dim | |
|---|---|
| **RELEVANCE** ⭐⭐⭐ | Directly gates injected **I-1** and **L-3** (value head vs fan scoring), and informs the v7f **trunk-freeze** decision. |
| **CONSEQUENCE** | D10-1's *"no value-head arm this cycle"* branch does **not** fire. But any value head over this trunk must be **benchmarked against the same head over pixels** — the trunk has not earned the credit for a distance signal. Moves `LAB_BACKLOG` D10-1 / I-1 and escalation 1 of LAB-RUN-011 (trunk-freeze). |
| **COMBINATION** | **Fifth independent sign this month that the frozen trunk is the ceiling** (LDAD encoder-shaping, AITS, the IQL value *encoder* loss `2601.00844`, Qwen-Drive's head-only failure — and now this). It also lands on the same day the A&I FlyWheel's frozen-s32 BEV-occupancy head failed to separate from its pixel arm: `main − pixel` **+0.0291 [−0.0040, +0.0693]** test AP (INHERITED, `…/2026-09-13-bev-lidar-corpus-and-head/RESULT.md` §6). Two unrelated read-outs — geometry-to-goal and occupancy — one trunk, **the same tie with pixels**. |
| **CHANCES / RISKS** | **Upside:** a 7,040-weight diagonal metric fitted by NNLS on CPU in seconds clears the bar — a value head is cheap. **Risks:** (a) *time*-to-goal rewards visual smoothness, which pixels have for free — the target itself may be too easy to discriminate encoders; (b) a 10 Hz frame gap is not an action-conditioned cost; (c) one trunk, one seed. |
| **EXPERIMENT** | ⭐ **`E-DEP-VGEO-3` — swap the target so pixels cannot win by smoothness:** rank by **ego path-length-to-goal** (stationary spans stop counting) and add **cross-clip distractor goals at matched pixel distance**. **Committed:** learned-latent − learned-pixel paired CI lower > 0 ⇒ the trunk carries goal-relevant geometry and I-1 proceeds on it; tie again ⇒ **a distance-shaped value should not read the frozen trunk** and the next lever is an unfrozen or separately-trained value encoder (MM decision). Blocker: the token bank carries no ego poses — needs a join from the eval v2ep cache (0 GPU, ~1 h, **after** the live MM arm frees the disk). |

## 3 · What this changes (≤3)

1. ⭐⭐ **Mark D10-1 SERVED with verdict PARTIAL→TIE, not CLOSED.** I-1 is unblocked for a *metric head*, blocked for *trunk credit*.
2. ⛔ **Every future value/metric head on the frozen trunk ships with a same-head pixel arm** — this is T-2 (critic ships with a raw-input floor) applied to value geometry, and it decided today's verdict.
3. ⭐ **Escalate to the v7f freeze owner:** add this as the fifth measured exclusion; if a partial unfreeze is cheap, keep it reachable.

## 4 · Stopping condition (Rule Zero)

The bar was **cleared by the lever (F3) and the discriminating control then tied (F4)**. The next cheapest lever (`E-DEP-VGEO-3`) is **blocked on a pose join** that would contend for disk and RAM with a live Master-Mind arm — **named blocker: the MM BEV-head run on the dev box; unblocks when it exits.**

## 5 · Manifest

| artifact | location |
|---|---|
| SPEC / pre-registration | `repo:TanitAD Research Lab/Deployment & Optimization/Research/2026-09-13-i1-value-geometry-screen/SPEC.md` |
| Lever addendum | `repo:…/PREREG_ADDENDUM_L2_LEARNED_METRIC.md` |
| Code | `repo:…/code/vgeo_screen.py`, `repo:…/code/vgeo_learned_metric.py` |
| Raw | `repo:…/raw/vgeo_screen.json`, `…/raw/vgeo_screen.log`, `…/raw/vgeo_learned_metric.json`, `…/raw/vgeo_learned_metric.log`, `…/raw/prereg_pin.json`, `…/raw/search_log.md` |
| Input bank (not copied — 20 GB, MM-owned, live) | `devbox:C:/Users/Admin/tanitad-caches/bevhead-20260913/tokens/` ⚠️ **lives only on the dev box**; the MM package documents its builder |
