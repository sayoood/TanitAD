# REF-A v1 — the ledger of PROVEN improvements

**What REF-A is.** The **frozen-encoder variant of the 4B TanitAD world model**: a
frozen visual encoder feeding trainable downstream modules. It is the programme's
H4 control — the arm that answers *"does a frozen encoder suffice?"* — and under
standing decision **D-003** it is a **COMPARISON ARM, NOT a hedge to adopt**
(violating that is retraction class **C129**).

**Scope.** Only measurements. Open hypotheses live in
`Project Steering/GOALS_AND_CLAIMS.md`; retracted claims in `RETRACTION_LOG.md`.

**Tier.** T0-DIAGNOSTIC unless a row says otherwise. The ADE figures below are
`wm_fidelity` numbers, **not driving performance** (C131: flagship v1's open-loop
0.4271 → closed-loop 1.7318, a 4.05× divergence).

---

## 1. The starting point, and the reading that was wrong

`refa-4brain-speed-30k` — frozen **DINOv2-B/14** + trainable adapter + supervised
trajectory head — plateaued at **ADE@2s 2.1322 ± 0.1821 m** (registry §2.1;
paper §11 quotes 2.1675 m for the earlier variant). The programme's default reading
attributed this to *the frozen encoder*.

⛔ **That attribution is REFUTED**, by two independent pre-registered lines.

---

## 2. ⭐ PROVEN: the information is NOT lost in the frozen encoder (P2-PRESERVED)

A five-stage readout ladder measured lead-gap decodability at every point of REF-A's
pipeline. `MEASURED`, verdict **`P2-PRESERVED`**, pre-registered **before any readout
ran**:

| stage | lead-gap readout |
|---|---|
| raw frozen features | **0.5285** |
| trained adapter | 0.4751 |
| predictor latent ẑ(t+0.1 s) | 0.4762 |
| predictor latent ẑ(t+0.4 s) | 0.4863 |

**The adapter did NOT collapse.** Per-dimension standard deviation **0.8011** against
**0.220** at random initialisation — training *expanded* the representation's
variance by **3.6×**, with **zero dead dimensions**. The predictor step costs
**+0.0011 [−0.0044, +0.0065]** — nothing.

⇒ **The signal arrives intact at precisely the latent the evaluation decoded** — the
latent that produced the programme's worst driving number.

---

## 3. ⭐ PROVEN: readability does NOT discriminate driving

A separate pre-registered test: REF-A's readout advantage buys nothing where it
should matter most. On **lead-present windows** the contrast is
**−0.0146 [−0.5988, +0.5551]** — not separated. And **REF-C**, whose encoder is
trained from scratch and is nearly agent-blind, **drives best of all arms**.

**The operative conclusion is a PROHIBITION.** Let ρ denote a linear-probe readout
score and D a driving metric. The programme had been reasoning as though ρ were a
sufficient statistic for D. Five measured stages across two independent streams show
no such relation:

> ⛔ **No readout number is admissible as a reason to swap, freeze, or distil an
> encoder** until an experiment establishes the ρ→D link on our own arms.

⚠️ This prohibition binds the 2026-08-23 campaign too: E-DEC-8's distillation result
(`n_agents` −1.04 → +0.3274) is a **readout** number. It licenses a representation
claim and **not** a driving claim.

---

## 4. ⭐ PROVEN: the failure was the PAIRING, not "frozen"

Eleven primary sources were banked and read for one question: **under what conditions
do frozen encoders succeed?** They succeed in exactly two configurations, and REF-A
occupied neither.

**Configuration A — large frozen encoder, wide interface, supervised head.**
`PUBLISHED` (`2601.03460`, FROST-Drive), Waymo Open E2E, 5 cameras at 448²:

| arm | RFS ↑ | ADE@3s ↓ |
|---|---|---|
| frozen VLM 78 B | 8.24 | 0.95 m |
| frozen VLM 14 B | 8.17 | 1.04 m |
| the same 14 B, **fine-tuned** | 8.13 | **1.47 m** |
| ViT (ImageNet), fully fine-tuned | 7.79 | 1.20 m |
| ViT (ImageNet), **frozen** | **7.39** | **2.28 m** |

**Freezing beats fine-tuning for a strong encoder, and loses badly for a weak one.**

**Configuration B — partial adaptation, not full fine-tuning.** `PUBLISHED`
(`2509.11417`): full fine-tuning measurably *degrades* pretrained structure (OpenVLA
36.7 % → 12.1 % under paraphrased instructions). The winning form is a **dual
encoder — a frozen anchor concatenated with a trainable branch**: 35.03 → 55.55, and
78.46 with tokenizer and co-training. `PUBLISHED` (`2303.18240`, CortexBench) agrees
from the other side: no frozen representation dominates 17 tasks, but task-specific
*adaptation* beats the best known result on every benchmark.

⇒ **REF-A held configuration A's CONSUMER (a supervised head) over configuration B's
ENCODER CLASS (86 M, image-only SSL), through a narrow 256-token / 51.39° monocular
interface.** That combination appears nowhere in the successful literature.
**The failure was never "frozen"; it was this specific pairing.**

---

## 5. ⭐ PROVEN (derivation): why the OBJECTIVE matters more than the encoder

Let `z ∈ ℝ^{N×d}` be the frozen feature field (N patch tokens, width d), `a_t` the
action, `g_θ` the trainable module downstream of the frozen encoder.

**Supervised-head recipe (the old REF-A).** `g_θ` must learn

&nbsp;&nbsp;&nbsp;&nbsp;`g_θ : z_{t−W:t} × a_{t:t+K} ↦ y_{t:t+K} ∈ ℝ^{K×2}`

a **contraction** of roughly `d·N·W → 2K` — at v1's geometry **640·1024·4 ≈ 2.6×10⁶
inputs to 40 outputs**. Everything not needed for the trajectory is free to be
discarded, and gradient descent *will* discard it, because **nothing in the objective
pays for keeping it**. The objective supplies at most `2K` real numbers of
supervision per window.

⭐ **The empirical signature is in our own logs:** the only quantity REF-A's training
separably improved was **ego speed** — precisely the target its auxiliary losses
named. **Supervision moved what it pointed at, and nothing else.**

**Feature-prediction recipe (v1).** `g_θ` instead learns

&nbsp;&nbsp;&nbsp;&nbsp;`f_θ : z_t × a_t ↦ ẑ_{t+1}`,&nbsp;&nbsp;
`ℒ = 𝔼‖f_θ(z_t, a_t) − z_{t+1}‖²`

The target lives in **the same space as the input**. The supervision budget per
window becomes `N·d·K` — at v1's geometry **≈1.97×10⁷ per 6 s rollout, five orders of
magnitude richer than 2K** — and the map is close to the identity: writing
`f_θ(z,a) = z + δ_θ(z,a)`, the module learns only the **action-conditioned change**.

Two consequences follow directly:

1. **The demand on the adapter is weaker** — it must *preserve and propagate* a
   representation, not compress it into a different one. **Preservation is exactly
   what §2's ladder measured the REF-A adapter already doing**, so this objective
   asks it for something it demonstrably can do.
2. **Nothing licenses discarding.** Any dimension of `z` the objective drops incurs
   L2 cost at the next step. The collapse mode a supervised head permits is
   **penalised here by construction**.

⇒ **v1 changes the OBJECTIVE before it changes the encoder.**

---

## 6. ⚠️ Cross-reading against the 2026-08-23 campaign

The v7 campaign measured things that bear directly on REF-A v1. Stated because they
cut **both** ways.

**(a) Supporting v1's derivation.** E-DEC-7 established independently that a
**self-generated target** admits an "ego + noise" optimum: our two-term objective
produced exactly the signature §5 predicts for a contraction — *the only thing that
improved was ego*. REF-A v1's feature-prediction target is **also self-generated**
(`z_{t+1}` comes from the frozen encoder, not from the trainable module) — but with a
crucial difference: **the encoder is FROZEN, so `z_{t+1}` cannot be moved by the
optimiser.** The target is external *to the trainable module*. That is precisely the
property E-DEC-8 measured to be the one that works.

⭐ **This is the strongest architectural argument the campaign produced for REF-A v1,
and it was not available when v1 was designed.**

**(b) Bounding it.** ⛔ **We have never beaten frozen DINOv3** (C138, paired: speed
−0.1251, t −2.72), and a frozen encoder that never saw our data reads **+0.2754** on
`n_agents` where every trained arm of ours sat *below a constant predictor*. That
strengthens the frozen-anchor case — and it must not be read as licence to adopt
frozen as *the* encoder (**D-003**).

**(c) The dual-encoder form (config B) is now testable.** `--freeze-encoder` exists in
the v6/v7 trainer (default off), so *frozen anchor + trainable branch* can be run on
our own stack rather than cited. E-DEC-14 is that probe.

---

## 7. ⛔ What is NOT proven for REF-A v1

* **No driving claim.** The ρ→D prohibition in §3 is REF-A's own finding and it binds
  REF-A hardest: nothing here says v1 will drive better.
* **v1 is a DESIGN with a derivation, not a trained result.** §5 is an argument from
  supervision budget, not a measurement of v1.
* **REF-A I-JEPA's val number is unusable** — ~80 % of val leaked into its train set
  (registry §2.2).
* **`flagship4b-phase0-30k` is the no-speed ablation CONTROL (2.918 m)**, not the
  deployed v1 (that is `flagship4b-speedjerk-30k`, 0.452 m). The HF repo name invites
  this inversion.
