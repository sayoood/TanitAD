# TanitAD — validated achievements, best results, and assets

**Requested by the PI 2026-07-29: "an inventory about our main validated achievements, our best
results and assets… Are these documented, their code clean, committed and pushed?"**

Only **validated, quotable** items appear here. Every number cites `MODEL_REGISTRY.md` or a raw
artifact, never prose. Estimator for every interval: **episode-cluster bootstrap**.

⛔ **THIS FILE IS INCOMPLETE ON ITS OWN — READ `ASSET_INVENTORY_PART2.md` WITH IT.** The PI flagged
2026-07-29 that it omits IDM, hierarchical planning, strategic routing, the side-camera attention
work (H2), AlpaSim/TanitResim, pseudo-simulation, the VLM curation pipeline and the datasets. The
omission was systematic: I inventoried what I had recently touched and treated that as what the
programme has. **Part 2 carries those seven workstreams and their measured numbers.**

---

## 1. ⭐ THE BEST RESULT — flagship v1 beats the constant-velocity floor, twice, separated

`flagship4b-speedjerk-30k` — **the deployed model**, 286.3 M params, complete at 30 k.

| surface | `ade_0_2s` | CI95 | windows | eps | CV floor | Δ vs CV |
|---|---|---|---|---|---|---|
| **40 eps (canonical)** | **0.4271** | [0.3675, 0.4871] | 881 | 40 | 0.8377 | **+0.4106 [+0.2050, +0.6240] ✅ separated** |
| ⭐ **600 eps** | **0.4108** | **[0.3956, 0.4273]** | **13,198** | **600** | 0.6917 | **+0.2809 [+0.2457, +0.3142] ✅ separated** |

⭐ **The 600-episode result is the strongest evidence the programme has**: 13,198 windows, CI
half-width **0.0159** (half the 40-ep width), and separated from the trivial floor by a wide margin.
It is a *confirmation on 15× more data*, not a correction.

**Calibrated context** (all in the registry, all measured): CV floor 0.8377 · CTRV oracle 0.523 ·
best-of-3 kinematic floor 0.5005 · learned ego-status **no-vision** ceiling 0.5735 · in-sample
re-scoring ceiling 0.4907. ⇒ **v1 sits below every kinematic floor and below the no-vision ceiling —
the vision pathway is doing real work.**

⚠️ **Read `0.4271` correctly**: it is `wm_fidelity_ade_2s` — the world model integrating *known*
actions. It is the right number for "does the world model work"; it is **not** a planning bar.

## 2. Completed, evaluated model arms

| arm | params | status | headline |
|---|---|---|---|
| **flagship v1** `flagship4b-speedjerk-30k` | 286.3 M | ✅ 30 k | **0.4271 / 0.4108** (§1) |
| **REF-B v2** `refb-refbpatch-v2-30k` | — | ✅ 29,999 | ADE@2s **0.6152** full-set — **the first CV-beater** |
| **REF-C-XL** `refc-diffusion-xl-30k` | 251.9 M | ✅ 29,999 | full-set `ade_0_2s` **0.4714** |
| **REF-C-base** `refc-diffusion-base-v21-30k` | **104.2 M** ✅ measured | ✅ 29,999 · evaluated | the base under the whole E1 closed-loop chain |
| **REF-C-small** `refc-diffusion-small-v21-30k` | 54.7 M | ✅ 29,999 · evaluated | closes the D-030 scale ladder |
| **REF-A dyn-in** `refa-dynin-4brain-30k` | — | ✅ 29,999 | closed H4 (the frozen-encoder question) |
| **flagship v1.6** `flagship-v16-ab-ft` | — | ✅ 5,999 | 0.4886 heldout — **tied** with v1 |
| **flagship-v4 from-scratch** | 286 M | ✅ 30 k (`rc=0`) | step 29,999; gate run 2026-07-28 |

⭐ **The D-030 scale ladder is a complete, three-point result** (small 54.7 M → base 104.2 M →
XL 251.9 M, all at 29,999 on the identical corpus). Its verdict: **the knee is ANCHOR-COUNT, not
encoder scale** — a genuine architecture finding, not a null.

## 3. Instruments — the reusable assets

1. **TanitEval** — the evaluation substrate. Episode-cluster bootstrap (`taniteval/ci.py`), paired
   form for two arms on shared windows, lateral/longitudinal Frenet decomposition, corridor metrics,
   closed-loop rollout. **This is what makes every number above quotable.**
2. **Parity system** — canonical corpus `physicalai-train-e438721ae894` (2,376 episodes, skip-hash
   `f09e44db`) with a **committed cryptographic manifest**. The trainer verifies clip sha256 at
   launch and refuses on mismatch. Proven in use: today's v5 launch printed
   `2400 clips, clip sha256 e61a04553df5… matches the committed manifest`.
3. **GATE_PROTOCOL + gate cards** — pre-registration machinery with cards registered *before* the
   checkpoint exists (the v4-30k card was registered at step 29,650). Includes VOID-by-construction
   handling so a healthy arm cannot die on a label bug.
4. **v2 compressed cache** — 80 GB train (2,400 clips) + 20 GB val (600), w120/256×640 cylindrical,
   with a `_geometry.json` carrying population-level rig observability.
5. **Wide-FOV geometry** — validated 2026-07-29: 176×624 = **HFOV 117.000°**, VFOV 32.131°, a pure
   centred pixel slice; masked-pixel fraction **0.0768 → 0.0063 (12×)**; pose/action semantics
   cross-checked numerically.
6. **Cross-DC pod relay** — 42 MB/s pod→pod (C56), md5-verified. Independent of HF quota.
7. **IDM / YouTube pipeline** — GeoCalib geometry + IDM end-to-end; 2,240 windows,
   `frac_in_plausible_0_45_mps = 1.0`. ⚠️ pseudo-labels, no ground truth — no accuracy claim.

## 4. Validated findings worth keeping

- ⭐ **The speed-input fix** — adding v0 as a third action channel: REF-A fwd_ade **3.73 → 0.83 m**,
  speed R² **0.61 → 0.965**. The single largest measured improvement in the programme.
- ⭐ **Closed-loop corridor recovery is real and large** — E1c cuts peak cross-track excursion
  **38.944 → 3.042 m (−92 %)**, `mean_xte` 14.306 → 1.391 m, separated on both. The base arm leaves
  the *road*; the fine-tuned arm stays in a 3 m band.
- ⭐ **`λ_replay` is a calibrated control**, not a knob that either works or doesn't: a monotone,
  non-crossing trade-off curve (λ=1 → −0.4407 @ +0.2158; λ=3 → −0.3911 @ +0.0958; λ=8 → −0.2891 @
  +0.0500). The programme can now *choose* an operating point.
- ⭐ **Widening the field buys something, and it is NOT angular resolution** — the wide 120°/640-token
  frame is separated-better than the deployed frame (+0.04246 AP, +0.02774 R²), and **98.1 % of that
  survives at matched px/deg**. That vindicated the field decision on its own terms and is why v5
  trains wide today.
- ⭐ **Route recalibration** — balanced route accuracy **0.4242 → 0.5493**, right-turn recall
  **0.041 → 0.289 (7×)**, from one threshold constant, no training.

## 5. ✅ Documented, clean, committed, pushed — verified 2026-07-29

| check | result |
|---|---|
| working tree | **clean** (only `.claude/settings.local.json`, an IDE file) |
| unpushed commits | **0** — everything is on `origin` |
| test suite | **1,586 passed, 12 skipped** (`stack/`, full run) |
| HF backups | **11 `tanitad-*` model repos** under `Sayood/`, incl. today's `tanitad-flagship-v4-fromscratch-30k` (md5-verified) |
| registry | `MODEL_REGISTRY.md` is the single quotable source and is current |
| artifacts | every result above has a dated `…/incoming/<date>-<topic>/` folder with raw JSON + the script that produced it |

✅ **RESOLVED 2026-07-29 — the PI said "merge".** `origin/main` fast-forwarded `2d903ba → d33a101`
(92 commits, **no force**), and `main` now tracks the programme.
⚠️ **The check that mattered:** my *local* `main` was **62 commits stale**, so the "clean
fast-forward" reported against it was against a stale reference — pushing that could have discarded
other people's work. **After `git fetch`, `origin/main` was confirmed an ancestor of the branch and
0 commits would be lost.** Verify against `origin/`, never a local branch that has sat unused.
