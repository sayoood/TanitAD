<title>I-1 path-length value geometry — VOID as registered, and what it still shows</title>

# I-1 `E-DEP-VGEO-3`: VOID as registered (twice). Descriptively, the frozen trunk separates PLACES, not DISTANCES

**2026-09-15 · Research Lab (LAB-RUN-013) · Deployment & Optimization · serves INJECTED row I-1 (top of lane) via DP13-1**
⛔ **Tier: none.** A representation diagnostic on frozen features — no rollout, no driving, no four-family eval.

---

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **F1** | ⛔ **Run 1 (as pre-registered) is VOID.** Two controls missed: the P2 raw-pixel distractor arm read **0.8327 [0.7925, 0.8715]** against a bar of [0.45, 0.55], and C-vshift read **−0.1201 [−0.2603, +0.0166]** against \|point\| ≤ 0.03. Diagnosis from the run's own numbers: the distractor match was unbounded (same-clip goals are closer than almost any cross-clip row, so "nearest match" is still farther), and a circular shift reverses monotone speed trends. | MEASURED `raw/vgeo3_pathlen.json` |
| **F2** | ⛔ **Run 2 (post-hoc control repair, pinned before running) is VOID too — and per its own rule there is no third repair.** The P2 fix worked: raw-pixel arm **0.5148 [0.4923, 0.5374]** ✅; C-mut **+0.0039 [−0.0404, +0.0437]** ✅; C-const exact ✅. ⛔ **C-vswap** (speeds from a different clip) **+0.0942 [−0.0447, +0.2275]**: CI contains 0, but \|point\| exceeds the 0.05 bar. The STEP = 2 sensitivity repeats it (**+0.0738 [−0.0609, +0.2129]**). | MEASURED `raw/vgeo3b_step4.json`, `raw/vgeo3b_step2.json` |
| **F3** | ⚠️ **Descriptive only — the primary contrast is a tie leaning to pixels.** P1 (within-Δ path ordering), `latent_7040|diag − pixel_1440|diag` = **−0.0690 [−0.2177, +0.0797]** (29 clips); STEP 2: **−0.0900 [−0.2339, +0.0601]**. The latent arm itself is **+0.1510 [+0.0218, +0.2837]**, only ~0.06 above the cross-clip speed null, so most of it is a population-level position effect, not path geometry. | MEASURED, descriptive |
| **F4** | ⭐ **A robust raw-arm pattern, independent of every failed control: at a fixed frame gap, the raw latent moves LESS when the ego travels FARTHER.** Raw `latent_7040` P1 **−0.2266 [−0.3687, −0.0696]**, raw pixels **+0.2191 [+0.0446, +0.3991]**; paired **−0.4457 [−0.5857, −0.2989]**. Identical in run 1 (no fit is involved). | MEASURED |
| **F5** | ⭐ **Signature `SCENE-IDENTITY-ONLY`, descriptively.** With the repaired matching (control passes), raw latent picks the true goal over a pixel-matched cross-clip distractor at **0.7617** vs pixels **0.5148**: paired **+0.2468 [+0.1726, +0.3146]**. But with learned metrics the gap closes: `7040|diag − pixel|diag` **+0.0607 [−0.0066, +0.1287]** (STEP 2: +0.0918 [+0.0142, +0.1709]). **The trunk knows which place it is in beyond a thumbnail. It does not order distances better than one.** | MEASURED, descriptive |

**Verdict against DP13-1's committed branches: no verdict fires (VOID).** Every descriptive read points the same way as 09-13's `HEAD-CLEARS-BAR-BUT-PIXELS-TIE`. **I-1 stays unblocked for a metric head and still blocked for trunk credit.**

## 1 · What was run

* **Bank:** refcv5-v2 s32 tokens, `C:/Users/Admin/tanitad-caches/bevhead-20260913/tokens/` (139 clips). **Speed join:** `ego_v_ms` from the A&I LiDAR-BEV GT artifacts (`…/bev_gt/<sha12>.bevgt.npz`, same raw v2ep frame index). **134 / 139 clips joined**; 5 excluded (no bev_gt file, ids in the JSON); 411 invalid-label frames interpolated. Fit 94 / scored 40 clips (seeded, disjoint, asserted). P1 is defined on **29** scored clips (the rest have < 5 % path-length variation at every gap).
* **Dev-box state at launch:** `tasklist` showed **no python.exe**, 4060 at 9 % / 1.2 GB. CPU only, below-normal priority, 75–147 s per run.
* **Pre-registration:** `SPEC.md` (sha256 `ef5e1cea…`) and `PREREG_ADDENDUM_CONTROL_REPAIR.md` (`68148999…`), both in `raw/prereg_pin.json`, each pinned before its run computed anything. ⚠️ The addendum is **post-hoc by declaration** and changed only the controls and the P2 match. It also loosened C-mut's tolerance from 0.03 to 0.05 after a 0.0015 miss; that is stated in the addendum.
* **Estimator:** clip-cluster bootstrap, 2,000 resamples. It covers another draw of clips only: one checkpoint, blind to training variance (`H-ESTIM-SEED-1`).

## 2 · Five-dimension analysis

| dim | |
|---|---|
| **RELEVANCE** ⭐⭐⭐ | Gates injected **I-1** / **L-3**, and adds evidence to the trunk-freeze decision (**AI13-2**). |
| **CONSEQUENCE** | Nothing new is licensed. DP13-1's cheap next step is spent. Pixels still tie the trunk on every *distance* read we have (VGEO-1, VGEO-2, and descriptively VGEO-3), while the trunk wins on *place identity* (F5). ⇒ a value head that needs goal *distance* should not be credited to this trunk. A value head that needs *which goal / which scene* might be. |
| **COMBINATION** | ⭐ **F4 matches our own measured history.** Latent change is anti-correlated with ego travel at a fixed gap, which is what a trunk would do if its latent tracked scene-content change (turns, intersections, pedestrians at low speed) rather than ego translation. This is the representation-side face of **P-2** ("the predictor does not use its actions"): if translation is weakly encoded, an action channel carrying translation has little to condition. It also sits beside the A&I BEV head's `main − pixel` tie (09-13) as another frozen-trunk read-out where pixels hold their own. |
| **CHANCES / RISKS** | **Upside:** F5 is the first read where the trunk beats pixels with a passing control. It is a candidate *retrieval / goal-identity* signal for the strategic level. **Risks:** (a) both runs VOID, so nothing here is a verdict; (b) a population-wide speed-by-position trend contaminates any cross-clip null, and the control that failed is exactly that one; (c) 29 defined clips; (d) one trunk, one checkpoint. |
| **EXPERIMENT** | **`E-DEP-VGEO-4` (proposed as DP15-1, 0 GPU):** replace the cross-clip speed null with a **within-clip, position-matched** null: permute path-length *residuals* after regressing path length on frame position per clip, so no population trend survives. **Committed in advance:** that null must read \|point\| ≤ 0.03 on the latent arm *before* the P1 contrast is read. Then primary `7040|diag − pixel|diag` CI lower > 0 ⇒ trunk carries path geometry; CI contains 0 ⇒ tie confirmed with a valid null, and the value line moves off the frozen trunk (MM decision). ⭐ Also pre-register **F5 as its own primary** (`raw latent − raw pixel` on matched distractors) so the place-identity result can become a verdict rather than a footnote. |

## 3 · What this changes (≤3)

1. ⛔ **Record `E-DEP-VGEO-3` as VOID (both runs) in the claims register.** Do not cite F3 as a tie *result*, only as a descriptive lean.
2. ⭐⭐ **Split I-1's representation question in two: goal DISTANCE (pixels tie, three reads) vs goal IDENTITY (trunk wins, one read with a passing control).** A strategic value that selects among goals may legitimately use the frozen trunk; a tactical cost-to-go should not claim it.
3. ⭐ **Standing null-design rule for any cross-episode control:** remove within-episode position trends before permuting across episodes. The failure mode here (a population trend surviving a swap) applies to every "shuffle across clips" control in the programme.

## 4 · Stopping condition (Rule Zero)

Both registered attempts failed a control. The addendum committed *"no second repair this pass"*, so the pass stops here. The next lever is `E-DEP-VGEO-4` above: 0 GPU, ~1 h, no blocker.

## 5 · Manifest

| artifact | location |
|---|---|
| SPEC / pre-registration | `repo:TanitAD Research Lab/Deployment & Optimization/Research/2026-09-15-i1-path-length-value-geometry/SPEC.md` |
| Control-repair addendum | `repo:…/PREREG_ADDENDUM_CONTROL_REPAIR.md` |
| Code | `repo:…/code/vgeo3_pathlen.py` (run 1), `repo:…/code/vgeo3b_control_repair.py` (run 2) |
| Raw | `repo:…/raw/vgeo3_pathlen.{json,log}`, `…/raw/vgeo3b_step4.{json,log}`, `…/raw/vgeo3b_step2.{json,log}`, `…/raw/prereg_pin.json`, `…/raw/search_log.md` |
| Inputs (not copied: 20 GB token bank + LiDAR GT, MM/A&I-owned) | `devbox:C:/Users/Admin/tanitad-caches/bevhead-20260913/{tokens,bev_gt}/` ⚠️ **live only on the dev box**; builders documented in the A&I 2026-09-13 package |
