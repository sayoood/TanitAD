# PRELIMINARY — the zero-GPU ablation panel on `ckpt_30000.pt`

⛔⛔ **THIS IS NOT THE LANDING RESULT AND MUST NOT BE QUOTED AS ONE.** It is
`ckpt_30000.pt` (not the final 40,284), on **20 clips at `--window-stride 20` = 171 windows /
20 episodes**, run **CPU-only on the training pod while the run was still training**. The landing
read is **141 clips at stride 5 = 4,823 windows**, the grid refcv3 @40,284 was scored on. ⇒ levels
here are **not comparable** to any refcv3 number; only the SHAPES are informative, and each is
labelled with what would change it.

**Why it exists at all:** the PI has queued refcv5 for this A40. If the GPU is taken before the
landing panel finishes, this is what survives — and it cost no GPU.

**Artifacts:** `pod:/workspace/eval/ab20/{BASE,h19_off,sel_refined,ego_zero,frames_blind}.json`
and `d_*/` dumps; `raw/paired_ablate.py`; `raw/paired_h19.json`.

---

## 1. The arms at n = 171 windows / 20 episodes (episode-cluster bootstrap, `ade_dense_m`)

| arm | ADE (m) | CI95 |
|---|---|---|
| `os` — the model | **0.3055** | [0.2441, 0.3883] |
| `ha` — hold-action | **0.2860** | [0.2417, 0.3476] |
| **`ha0_ext` — the echo control** | **0.2769** | [0.2366, 0.3334] |
| `ha0` — constant velocity | 0.6542 | [0.4854, 0.8544] |

⛔ **Both trivial controls still beat the model at step 30,000 on this subset.** That is the same
defect shape refcv3 carries (`os − ha` = **+0.1423** [+0.1187, +0.1658] at 40,284, separated the
wrong way).

⭐ **But the margin is much smaller here: `os − ha` = +0.0195 m against refcv3's +0.1423 m — 7.3×
smaller.** ⚠️ **DIFFERENT SURFACE, DIFFERENT CHECKPOINT — this is a shape, not a comparison.**
Whether it survives on the 4,823-window grid at step 40,284 is **the single question the landing
eval answers**, and it is the one I will report first.

⚠️ `ha0` is **2.36× worse** than `ha0_ext` here, which is why an `ha0`-only bar certifies arms the
trivial control already beats. And `ha` (0.2860) and `ha0_ext` (0.2769) are **close but distinct**
on this surface — so the two halves of the acceptance bar are not degenerate at scale, even though
they read bit-identically on a 3-window probe.

---

## 2. `h19_off` — properly estimated, and the answer is INCONCLUSIVE, which is the honest third branch

Both runs share the **same window grid** — asserted, not assumed: `ws` and episode ids are compared
element-wise and the script REFUSES if they differ.

| | |
|---|---|
| BASE `os` | 0.3055 m |
| `h19_off` `os` | **0.3021 m** |
| paired delta | **0.0034 m, CI [−0.0040, +0.0111], `separated: false`**, `p_delta_gt0` 0.798 |
| estimator | `paired_episode_cluster_bootstrap` (`taniteval/ci.py`), n_boot 2000, seed 0 |
| windows whose selection flips | **10/171 = 5.85 %** |
| windows whose ADE changes at all | 10/171 = 5.85 % |
| ⛔ CONTROL, A paired against ITSELF | delta **0.0000000000**, lo 0.0000000000, hi 0.0000000000, `separated: false` — the estimator exercised end to end on a known null |
| CONTROL, base-vs-base selection flips | **0** |

**Verdict at this n: INCONCLUSIVE** — the pre-registered third branch, reported as such and not as
a null. The direction is very slightly toward *"removing the prior helps"* (0.3055 → 0.3021) and
the interval straddles zero.

⭐ **AND THE EFFECT SIZE IS THE FINDING, WHICHEVER WAY THE SIGN LANDS.** H19's anchor prior changes
**5.85 %** of picks and moves ADE by **0.0034 m** on an arm whose deficit to `ha` is 0.0195 m and
whose ADE is 0.3055 m. ⇒ **"repair the longitudinal kin3 head so H19's prior works better" is a
LOW-value refcv5 lever regardless of which way the landing read resolves it** — it is a
sub-1 %-of-ADE edge. Ranking levers by measured effect size, not by how interesting the mechanism
is, sends refcv5 elsewhere.

⚠️ It is NOT inert, and that was checked BEFORE this was written, because *a lever multiplied by a
zero coefficient is not a null about the lever*: `core.decoder.lat_to_anchor.weight` absmean
**2.802e-01**, `core.decoder.lon_to_anchor.weight` **7.047e-02**, neither all-zero, against
non-zero controls (`conf_head.weight` 1.730e-02, `goal_gate` 6.424e-02).

⭐ **An independent corroboration of the curve finding fell out of those weights:** the longitudinal
prior's matrix is **4.0× smaller in absolute mean** than the lateral one. The network itself
down-weighted the head whose cross-entropy cannot beat its own class marginal (eval CE 1.0090 nats
vs a prior-predictor floor of 0.8964).

---

## 3. What still has to come from the landing read

* the same panel at **4,823 windows / 141 episodes** on `ckpt_40284_FINAL.pt` — 28× the windows and
  7× the episodes, i.e. the power this read does not have;
* `sel_refined`, `ego_zero`, `frames_blind` at that n (running here on CPU, ~18 min/arm);
* the four families per arm, never pooled, each with its own paired interval;
* **turn recall per class beside ADE** — a ranking change that improves ADE by picking straighter
  paths is a regression wearing a win;
* the eval-set kin3 marginal, which converts §2's floor from a TRAIN EMA to a measured one.
