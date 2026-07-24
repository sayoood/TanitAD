# P1 — decision-grade the real-footage observation-OOD envelope (flagship v1)

**Date:** 2026-07-23 (Berlin) · **Host:** `tanitad-pod` (pod1, RTX A6000, `gpu_lock` = `lowood-harden`) ·
**Status:** MEASURED, banked. Harness `lowood_ci.py`; raw `lowood_flagship_ci.json`; stdout
`lowood_flagship_ci.log` (all staged this dir).

**Evidence class on every number:** `MEASURED (ours + artifact)` unless tagged otherwise.

---

## 0. What P1 asked for vs what was reachable — the sourcing reality (stated, not hidden)

The brief asked to (a) re-run the Δ=0 + deviation sweep on the **full 40-ep clean val**, and (b) add
**REF-C base** as a 2nd arm. **Both inputs are unreachable from pod1** — established by ≥3 path probes
and 2 independent doc sources, not one check (RETRACTION_LOG class **C2**):

| input | where it actually lives | probes run | verdict |
|---|---|---|---|
| 40-ep clean val `physicalai-val-0c5f7dac3b11` | **`tanitad-eval` only** (registry §0.3: "40 episodes → 881 windows … on `tanitad-eval`, NOT in this repo"); prior design §2.1 says the same. pod1 `/root/valdata/` holds **12** eps | `ls` val dir (12); `find /root /workspace` for other `*val*` dirs (none); `find … ep_00039.pt` (none); HF datasets cache (empty) | **BLOCKED** — 40-ep set is on the off-limits eval pod; not on pod1, not on HF |
| REF-C **base** ckpt (`refc-diffusion-base-v21-30k`, 104.2 M) | `tanitad-pod3:/workspace/experiments/…` + `tanitad-eval:/root/models/refc-base-30k/ckpt.pt` (registry §4.3); transferred pod3→eval by **direct scp, never HF** | `find` pod1 (only flagship `ckpt.pt`); HF `list_models(author=Sayood)` | **BLOCKED** — only on off-limits pod3/eval |
| REF-C **small** ckpt (HF fallback, `Sayood/tanitad-refc-small-evalonly`) | HF private repo (file `ckpt_evalonly.pt`) | `hf_hub_download` → **HTTP 403** | **BLOCKED** — *"Private repository storage limit reached for Sayood"* (account over quota) |

Getting any of these to pod1 requires touching **pod2/pod3/eval** (explicitly off-limits — they run
v4.2b / Branch B / Gate-1 proto) or altering Sayed's HF account (out of scope, and deleting repos is
destructive). So the literal P1 was **not executable on pod1 without violating a constraint.** What was
achievable — and is a genuine "decision-grade" upgrade the prototype lacked — is below.

---

## 1. What was delivered — episode-cluster bootstrap CIs on the flagship envelope

The prototype (`lowood_envelope.json`) reported **bare point estimates** ("+6.3 % at 2 m lateral", "≤1.16×")
with **no interval** — inadmissible under CLAUDE.md ("Never quote an interval without its estimator"). P1
re-runs the identical sweep (Δ=0 byte-for-byte the gate rollout; same warp geometry, imported verbatim
from `lowood_probe.py`) but **retains per-window ADE and episode ids** and applies the program's mandated
decision-grade estimator: the **episode-cluster bootstrap** (`taniteval/ci.py`, `n_boot=2000`), plus a
**paired** bootstrap of every condition vs the Δ=0 baseline on the *same* windows (the shared per-window
difficulty cancels — strictly more powerful than quadrature, and the only valid paired test here).

**flagship v1 `flagship-30k` (step 29999), 12 clean-val eps → 265 windows.**

### 1.1 Headline — reconstruction-OOD elimination is CI-robust

| source (open-loop ADE@2s, force-GT, same protocol) | ADE@2s | 95 % CI (episode-cluster) |
|---|---:|---|
| **Real-footage log-replay (Δ=0)** | **0.4045** | **[0.3128, 0.5149]** |
| NuRec / AlpaSim reconstruction (REF-C base, INHERITED) | 1.5157 | — |

NuRec's 1.5157 is **2.94× the real-footage baseline's own UPPER CI (0.5149)** and 3.75× its mean. The
elimination of the ~3.2× reconstruction-OOD is not a point coincidence — the real-footage baseline's
entire 95 % interval sits far below the NuRec level. *(The NuRec/real REF-C numbers are INHERITED base
scalars; a flagship-on-NuRec number was never measured, so this is a labelled cross-source ratio, not a
paired CI. The contrast is nonetheless decisive on magnitude.)*

### 1.2 The deviation envelope — now with paired CI-separation

Δ vs Δ=0 baseline is the **paired episode-cluster bootstrap**; **SEP** = CI excludes 0.

| Δ_lat (m) | ADE@2s [CI] | paired Δ [CI] | sep? | | Δψ (deg) | ADE@2s [CI] | paired Δ [CI] | sep? |
|---:|---|---|:--:|---|---:|---|---|:--:|
| 0.5 | 0.4014 [0.305,0.517] | −0.003 [−0.016,+0.009] | n.s. | | 2 | 0.4080 [0.316,0.522] | +0.004 [−0.014,+0.020] | n.s. |
| 1.0 | 0.4071 [0.306,0.530] | +0.003 [−0.018,+0.023] | n.s. | | 3 | 0.4211 [0.325,0.542] | +0.017 [+0.001,+0.034] | **SEP** |
| 1.5 | 0.4219 [0.319,0.545] | +0.017 [−0.006,+0.039] | n.s. | | 5 | 0.4303 [0.338,0.549] | +0.026 [+0.001,+0.050] | **SEP** |
| 2.0 | 0.4299 [0.337,0.537] | +0.025 [−0.011,+0.076] | n.s. | | 8 | 0.4438 [0.350,0.555] | +0.039 [+0.003,+0.075] | **SEP** |
| 3.0 | 0.4703 [0.369,0.578] | +0.066 [+0.010,+0.138] | **SEP** | | 12 | 0.4596 [0.368,0.567] | +0.055 [+0.021,+0.095] | **SEP** |

**pixshift** (calibration-free cross-check): **nothing separates through 32 px** (largest paired Δ +0.009
[−0.010,+0.028] at 32 px).

### 1.3 Reading it (decision-grade)

1. **Lateral offset up to 2.0 m carries NO CI-separated OOD** — the source is statistically flat out to a
   two-metre lateral excursion. Only at **3 m** does the rise resolve (+0.066, [+0.010,+0.138]).
2. **Yaw is the more sensitive axis** — it separates at **3°** and grows monotonically to +0.055 at 12°.
3. **Every separated rise is an order of magnitude below the NuRec gap.** The largest (3 m lat +0.066;
   12° yaw +0.055) is **~17–20× smaller** than the baseline→NuRec gap (+1.11 m). Across the *entire*
   ±3 m / ±12° envelope the real-frame ADE stays ≤ 0.47 — never within 3× of NuRec's 1.52.
4. **The CIs are wide because n = 12 episodes** (baseline [0.31, 0.52]). This is exactly the limit the
   40-ep set would tighten; the paired test still resolves the envelope's shape despite the wide
   single-arm bands. **The direction is decision-grade; the absolute envelope width would firm up ~1.8×
   at n = 40** (√(40/12)).

**Net:** the prototype's "≤1.16× out to ±3 m/±12°" is upheld and sharpened — *within ±2 m lateral and
≤2° yaw the observation-OOD is indistinguishable from on-path; beyond that it is detectable but stays
an order of magnitude under NuRec.* This is the confound-free core a longitudinal-first Gate-1 needs.

---

## 2. Honest limits carried into P2

- **This is still the OBSERVATION-OOD** (open-loop force-GT), apples-to-apples with how 0.47/1.52 were
  measured — *not* yet a closed-loop planner's on-policy divergence. That is P2.
- **n = 12, one arm.** REF-C as a 2nd decoder family remains unmeasured (blocked, §0). The lateral
  homography is ground-plane-only → the lateral envelope is an optimistic bound (yaw is exact).
- **To unblock the literal P1:** either (i) a sanctioned read-only pull of the 40-ep val + REF-C base/small
  ckpt **from the eval pod** when it is free, or (ii) freeing Sayed's HF private-storage quota so
  `tanitad-refc-small-evalonly` (and a future `…-base-evalonly` push) resolve. Neither was available to a
  pod1-only agent under the off-limits constraint.
