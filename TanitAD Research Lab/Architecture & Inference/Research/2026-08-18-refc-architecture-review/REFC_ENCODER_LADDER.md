# RL — THE READOUT LADDER ON REF-C's OWN ENCODER: **the vision-driving arm reads EGO SPEED superbly (0.84), its apparent lead-gap readout is ~⅔ AN EGO-SPEED ECHO, and its deployed pooled surface is nearly AGENT-BLIND**

**Date:** 2026-08-18 · **Branch:** `agent/arch-inf-20260803` · **Agent:** refc-architecture-review (JOB 2)
**Sibling:** `REFC_ARCHITECTURE_REVIEW.md` (JOB 1, same package) · **Precedents:** C104 / ER10
(`…/incoming/2026-08-18-pooling-ladder-ER10/`), C123 (readability ⇏ driving), C92 (ego-speed echo),
C97/C109 (degenerate passes), C103 (seeds), C115 (sensitivity), C119 (degenerate-input controls).
**Eval tier:** ⛔ **T0-DIAGNOSTIC.** A frozen-latent linear readout is a representation diagnostic.
It is **never** driving performance — and C123 measured that linear readability does **not** predict
driving. No number here is an ADE or a closed-loop result.
**Compute:** dev-box CPU/RTX 4060 only, 415 s wall. ⛔ Zero training · zero pod · **Thor untouched**
· **no checkpoint downloaded** (banked latents only) · no episode selected — **parity untouched**.
**Evidence classes:** `MEASURED (ours + artifact path)` · `INHERITED (not re-verified)`.

---

## ⭐ THE ANSWER, IN ONE PARAGRAPH

**`MEASURED`** (`raw/rl_main.json`; targets `raw/rl_targets.npz`, gates `raw/rl_gates.json`).
On the **canonical 881 val40 windows** — the exact windows REF-C's published driving numbers live
on, identity-gated five ways — the banked, **vision-only** pooled latents of `refc-base-30k` and
`refc-xl-30k` were probed with the ER10 protocol (RP to 2 048, 5 seeds, C92-repaired ridge,
episode-cluster bootstrap, even/odd episode split). **REF-C's from-scratch encoder reads the ego's
own speed from pixels at r²_ceiling 0.843 (base) / 0.789 (XL)** — sixteen times v6's 0.052
(INHERITED, its own windows) and above DINOv2-B's 0.717 (INHERITED, its own windows) — a genuine
visual-odometry capability, not an echo: `pooled` is computed from frames alone, and the ego-dropout
training regime (v0 withheld on 50 % of samples) gave the trunk an incentive to estimate speed
visually. **But the apparently strong lead-gap readout (0.377 base / 0.455 XL, the DINOv2 band) is
mostly the C92 ego-speed proxy in its fourth appearance: the v0-only scalar already reads lead_gap
at 0.392 on these windows, and with v0 partialled out the residual collapses to 0.056 (base) /
0.131 (XL)** — with an elevated permutation floor (up to 0.23 on one cell) warning that the small
lead-n amplifies episode-level confounds. The one clean v0-independent object-dynamics signal is
**lead_closing: v0-partial r² 0.158 (base) / 0.194 (XL), against a 0.091 v0-proxy and a 0.000 noise
floor** — small but present, and larger in XL. **Agent count is essentially absent from the deployed
pooled surface (raw ≤ 0.035, BELOW the 0.122 v0-proxy)** — the arm the programme's vision-grounded
driving rides on is nearly agent-blind at the surface its aux heads read. Base-vs-XL and
pooled-vs-temporal deltas are **not separated** at this n (0/5 seeds on nearly every pair).

⇒ Read with C123: the programme's one vision-driving arm does **not** drive on a rich readable
object representation — it drives while carrying superb visual ego-speed and only a weak,
mostly-XL residual of object state. The readout ladder now says on BOTH sides (v6: reads almost
nothing; REF-C: drives anyway with little more) that **linear scene-state readability is not where
the driving signal lives** — which is C101/C123's conclusion reached from a third, independent
direction.

---

## 0. WHAT WAS PROBED, AND WHY THESE SURFACES

| arm | shape | what it is |
|---|---|---|
| `refc_base_pooled` | [881, 704] | globally mean-pooled conv map, LAST window frame — **the exact surface `route_head` / `maneuver_head` / `goal_head` read at inference** (refc.py:2030-2101) |
| `refc_xl_pooled` | [881, 992] | same, REF-C-XL |
| `refc_base_pooled_seq` | [881, 5632] | all 8 window frames' pooled vectors — the surface `StrategicCtx` consumes (temporal, 0.8 s) |
| `refc_xl_pooled_seq` | [881, 7936] | same, XL |

Latents are the **banked 2026-08-04 dumps** (`…/incoming/2026-08-04-lambda-findability/raw/
latents_refc-{base,xl}-30k-ep.pt`, `instrument_fail: []`, step 29 999, raster 256×256, grid 8×8) —
**vision-only by construction**: the dump provenance separates them from the ego+nav `measurement`
echo path, and `encode_pooled` is `encoder(frames)[1]` (refc.py:1838-1840). No checkpoint, no new
inference. ⚠️ The 64-token **pre-pool** surface (what the anchor decoder cross-attends) is NOT
banked and would need a checkpoint forward — stated as the one surface this run cannot see; the
pooled 64:1 surface is the deployment-relevant one for the aux/goal heads.

## 1. WINDOW IDENTITY — the five gates (`raw/rl_gates.json`, all PASS)

1. **G1** poses-only canonical view == `manifest_EVALPOD_val40.json` by per-episode **sha256**, 40/40.
2. **G2** dump eid partition == rebuilt `window_last_indices` grid (22×39 + 23×1 = 881).
3. **G3** 2026-08-04 lead block row-aligns (eid equal; block `speeds` vs dump `v0` max |Δ| 1.8 mm/s — independently derived channels).
4. **G4** the dumps' own `controls_vs_bank` (fan/gt/sel/v0 bit-identity vs the canonical fan bank) all true; XL dump rows identical to base.
5. **G5** the lead **re-derived here** (fresh registration + `select_lead_causal` from the local label zips) vs the banked block: state agreement **99.55 %**, and on all 270 shared LEAD windows **|Δgap| median = max = 0.0 m** — bit-identical lead selection; registration residual median 1.3 mm.

## 2. TARGETS (`raw/rl_targets.npz`; definitions in `rl_gates.json.definitions`)

`ego_v0` (n=881, the dump's own input channel) · `lead_present` (n=821 labelled; NO_LABEL excluded,
never counted as clear road) · `lead_gap` (n=270, banked `gap0_m`) · `lead_closing` (n=270,
**instantaneous** −v_rel·heading at t0 from the two nearest distinct cuboid samples + egomotion
vx/vy — the ll1 quantity, computed from the same obstacle.offline the ladder's join read) ·
`n_agents_grid` (n=821; distinct tracks within ±0.15 s of t0 inside the sp1 grid 0<cx≤60, |cy|≤16,
any class — declared reimplementation of sp1:242) · `n_agents_any` (n=821, no grid filter).
Eval-side n: 440 / 418 / 124 / 124 / 418 / 418 (odd episodes).

## 3. THE MAIN LADDER — r²_ceiling, mean [min..max] over 5 RP seeds

Split: even eids probe-train (20 eps), odd eids eval (20 eps). Ridge `intercept_col=-1` (C92 gate
demonstrated at startup: penalised solve predicts 0.000000, repaired predicts the mean). `(pv …)` =
r² of the **v0-partial** correlation. Stamps: C97 `K1_DEGENERATE` never fired; `K1 k/5` = seeds
whose MAE separably beat the constant-median floor (episode-cluster bootstrap; small-n lead rungs
rarely separate on MAE even with real correlation — reported, not hidden).

| target | n_ev | base_pooled | xl_pooled | base_pooled_seq | xl_pooled_seq | **v0-proxy** |
|---|---|---|---|---|---|---|
| **ego_v0** | 440 | **0.843** [0.836..0.848] | 0.789 [0.769..0.818] | 0.840 | **0.869** [0.835..0.884] | (1.0 by construction) |
| lead_present | 418 | 0.033 (pv 0.035, AUC 0.57) K1 0/5 | 0.045 (pv 0.053, AUC 0.63) K1 0/5 | 0.047 (pv 0.053) K1 0/5 | 0.055 (pv 0.064, AUC 0.62) K1 0/5 | 0.013 |
| lead_gap | 124 | 0.377 [0.299..0.468] **(pv 0.056)** K1 2/5 | 0.455 [0.342..0.503] **(pv 0.131)** | 0.410 (pv 0.081) K1 2/5 | 0.488 [0.403..0.516] **(pv 0.166)** | **0.392** |
| lead_closing | 124 | 0.142 [0.125..0.155] **(pv 0.158)** K1 0/5 | 0.247 [0.175..0.273] **(pv 0.194)** K1 0/5 | 0.164 (pv 0.175) K1 0/5 | 0.268 [0.189..0.298] **(pv 0.213)** K1 0/5 | 0.091 |
| n_agents_grid | 418 | 0.035 (pv 0.012) K1 0/5 | 0.016 (pv 0.012) K1 0/5 | 0.028 (pv 0.012) K1 0/5 | 0.027 (pv 0.012) K1 0/5 | **0.122** |
| n_agents_any | 418 | 0.021 (pv 0.032) K1 0/5 | 0.007 (pv 0.007) K1 0/5 | 0.014 (pv 0.043) K1 0/5 | 0.010 (pv 0.019) K1 0/5 | **0.074** |

**Controls (seed 0; full table in `rl_main.json`).** PLANT (target written into the raw features at
1× feature-sd along a fixed direction, then the same RP+fit): **0.93–1.00 on every pooled-arm rung**
— the instrument has full power where the headline claims live; on the seq arms the same-amp plant
is diluted (0.07–0.95) so seq rows are the weaker-instrumented supplement, stated. NOISE
(matched-random features, the D1 floor): ≤ 0.010 everywhere. YPERM (episode-block permutation of
the target, one permutation): ≤ 0.03 on 21 of 24 cells — ⚠️ **but 0.230 on XL/lead_gap and 0.280 on
XL-seq/lead_closing**: with 124 lead windows over ~20 episodes, episode-level structure alone can
manufacture r² ≈ 0.2 on a permuted target. ⇒ the lead-rung raw values must not be quoted without
the v0-partial column beside them, and even the partial values carry small-n caveats.
**Sensitivity (C115):** the pooled representation is not target-invariant — single features
correlate with v0 up to |r| = 0.832 (mean 0.382 over 704), and the LEAD/NO_LEAD class means are
separated at 0.37× the within-class radius (computed on the banked features, this package).

## 4. PAIRED DELTAS — episode-cluster bootstrap on Δr²_ceiling, per seed

Nothing separates. base − XL (pooled): ego_v0 +0.054 [envelope −0.087, +0.178] 0/5 seeds;
lead_gap −0.036 [−0.359, +0.113] 1/5; lead_closing −0.119 [−0.314, +0.120] 0/5 (partial-v0
−0.038 [−0.178, +0.131] 0/5). pooled − pooled_seq: all 0-1/5. Full listing in `rl_main.json`
`paired_deltas`. ⇒ "XL reads objects better" is a **trend across every lead rung, not a separated
finding** at n=124.

## 5. THE INHERITED CONTEXT — and the C122 fence around it

C104/ER10 numbers, **different windows (130-clip lead-enriched train slice), different corpus
composition, different encoder input geometry** (256×640 cylindrical vs REF-C's 256×256 crop):
v6F-SW@11250 through its deployed pool: ego_v0 **0.052** · lead_gap **0.005** · lead_closing
**0.000** · n_agents_grid 0.029 / all 0.102. DINOv2-B on the same ER10 windows: 0.717 · 0.450 ·
0.017 · 0.328 / 0.336. (`er10_main.json` / `er10_dino.json`, seed-mean.)
⛔ **No ratio between those numbers and this run's is admissible** (C122: different windows,
different quantities' base rates — lead-enriched vs parity val40; the v0-proxy alone differs
0.392-here vs unknown-there). What IS admissible: the **within-run** contrasts of §3 (every arm,
proxy, floor and CI lives on identical windows), and the qualitative convergence: **neither the
v6 encoder (reads ~nothing) nor the REF-C encoder (drives, reads speed + weak residual) supports
"swap in a stronger encoder" as the driving lever — C123's verdict from a third direction.**
The same-window cross-encoder discriminator (DINOv2 on THESE 881 windows) needs the 40 canonical
val camera chunks (~2 GB class of download) or a post-training Thor pass — **not run; a PI/orchestrator
call, flagged.**

## 6. WHAT THIS CHANGES

1. **The sitclf/goal design space:** REF-C's deployed pooled surface carries visual ego-speed
   (0.84) and weak object residuals — any future goal/situation head reading `pooled` inherits a
   speed-dominated basis; heads needing agent state need the pre-pool tokens or the obstacle join.
2. **The encoder-swap question** (PREREG_PRETRAINED_ENCODER_ARM, E-RECON-2): this run adds the
   missing REF-C cell to C123's table — the driving arm's encoder is not readably richer than v6's
   once the v0 echo is removed, at the surface its heads consume.
3. **The C92 discipline is now 4-for-4**: every lead_gap headline in this programme so far has
   shrunk by ~⅔ or more under the v0-partial. It should be a default column in every future probe
   (it is, in this instrument).
4. ⚠️ **lead_closing on REF-C is NOT the degenerate rung it was on ER10's windows** (C123: K1-fails
   everywhere there): here it carries pv 0.16–0.21 with clean floors — the val40 windows +
   instantaneous definition + REF-C latents give the first non-trivial closing readout in the
   programme. Still K1-MAE-unseparated at n=124 — a bigger-n re-read is the cheap sharpening.

## 7. DELIVERABLE MANIFEST

| artifact | where |
|---|---|
| this report | `…/Research/2026-08-18-refc-architecture-review/REFC_ENCODER_LADDER.md` (repo, staged) |
| main result JSON | `raw/rl_main.json` (arms × targets × 5 seeds + controls + paired deltas) |
| targets + masks | `raw/rl_targets.npz` · gates `raw/rl_gates.json` · fit log `raw/log_rl_main.txt` |
| code | `code/rl1_targets.py` (targets+gates) · `code/rl2_fit.py` (imports er10/pc6/ll1/taniteval.ci — one-implementation rule) · `code/rl3_summarise.py` |
| suite state | `stack/tests/test_refc*.py`: **65 passed** (this session, dev box) |

*Everything in the repo working tree, staged, on `agent/arch-inf-20260803`. Nothing pod-side,
nothing on Thor, no checkpoint bytes moved.*

⚠️ **Reproduction dependency, stated (the C123 stranded-cache lesson):** rl1 read the canonical
poses-only val40 view from THIS session's scratchpad (`…/scratchpad/val40hf/
physicalai-val-0c5f7dac3b11/`, 508 KB). It is sha256-pinned by the in-repo manifest and
HF-restorable, and everything derived from it (`rl_targets.npz`) is banked in-repo — so the
analysis re-runs from the repo alone; only re-BUILDING targets from scratch needs the view (or any
fresh pull of the val40 poses).
