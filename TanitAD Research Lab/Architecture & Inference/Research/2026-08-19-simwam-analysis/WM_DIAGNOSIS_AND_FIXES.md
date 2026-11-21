# What to improve in the TRAINED WM and in the FROZEN WM — the diagnosis, and where it comes from

`MEASURED (ours)` + `PUBLISHED-PRIMARY/SECONDARY` (banked, `Library/`) ·
**T0-DIAGNOSTIC for our own numbers** · written 2026-08-20 after the PI
challenged a recommendation that contradicted `D-003` (retraction **C129**).

⚠️ **Read `FROZEN_ENCODER_LITERATURE.md` (2026-08-18) first** — it already ranks
the decisive axes and this document does **not** restate it. What is new here is
(a) a MEASURED property of the v6 latent, and (b) one architectural distinction
that review could not make because the measurement did not exist yet.

---

## 1. The measurement that is new

`E_TRUNK_2` / `E_TRUNK_3`: the v6 operative latent (`d_op` = 2048) decodes
**nothing** — not `lead_gap_m` (**−0.018**), not lane occupancy (**AUC .53**),
**not even its own speed (−0.005)** — at **every** measured step
(2000 / 16000 / 18000 / 20000). Frozen **DINOv3 through v6's own 40× pool**
reaches `lead_gap_m` **+0.379** and occupancy **AUC .83**, and reads **ego speed
at +0.696**.

⭐ **And SIGReg is ON and WORKING**: `w_o6 = 0.1`, `sigreg_slices = 512`;
participation ratio **rises** 3.29 → 6.94, top-8 share **falls** 0.836 → 0.806.
Dimensional collapse is being prevented exactly as designed.

⇒ **The finding is not collapse and not missing machinery.** It is that
**an anti-collapse regulariser guarantees the dimensions are USED, never that
they are INFORMATIVE — an isotropy constraint can be satisfied by noise.**

## 2. ⭐ The distinction the literature review could not yet make

v6 **has** a feature-prediction objective (O1), so by the review's taxonomy it
sits in Configuration **B** — yet its latent is uninformative. The difference is
**what the target is made of**:

| system | prediction target | can the target DRIFT? |
|---|---|---|
| DINO-WM, V-JEPA 2-AC | a **frozen** encoder's features | ⛔ no — fixed by construction |
| DeepSight | **DINOv3** future BEV features | ⛔ no — external |
| LAW | feature prediction as an **auxiliary** to a supervised task | anchored by the main task |
| **v6 (S-W)** | **its own** future latent, `z_true_steps`, *"detached by the caller"* | ✅ **yes — the trainable encoder determines it** |

A **detached** self-target is still self-referential in the way that matters: the
encoder receives no gradient *through* the target, but it **decides what the
target will be** on the next iteration. That is the BYOL/SimSiam setting, which
is why those methods need an EMA teacher or predictor asymmetry. v6 deliberately
declines both — `refa.py`: *"LeJEPA doctrine: **no stop-grad/EMA crutch**, A1"* —
and substitutes SIGReg.

⇒ ⭐ **MEASURED OUTCOME OF THAT CHOICE: the LeJEPA recipe prevented dimensional
collapse and did not, on this corpus and at this scale, produce
environment-informative features.** That is a specific, falsifiable statement
about a design decision — not a verdict on JEPA-family methods in general.

⚠️ **`sigreg_free_dims = 0`** — `position_relaxed`'s docstring says this
*"reproduces plain SIGReg on the full latent"*, the configuration the §B.3
relaxation exists to replace (*"the diagnosed step-21k regression mechanism"*).
**Open item for the PI**, not a diagnosis: whether v6's latent has reserved
ego-motion channels for the exemption to protect is unverified.

## 3. ⛔ v6 has NO grounding signal at all — verified twice

`"S-W": ("encoder", "readout", "predictor_op", "aux")`, and `aux` is
**`masked_cells.` + `sigreg.` only** — the source calls them *"label-free trunk"*
objectives. `grep` for `aux_ego` / `aux_speed` / `aux_accel` / `egomotion` /
`proprio` in `train_v6_staged.py` returns **nothing**. Every trunk-shaping term
(O1, O2, O3, O5, O6) is self-supervised.

**REF-A is the contrast, and it is measured:** it carries an
`InverseDynamicsHead` — *"Forces the controllable state into the compact latent —
**the cheapest and most effective grounding signal we have**"* — and its runs used
`--aux-egomotion --aux-accel`, reaching **`aux_speed_r2 0.9825`**, `aux_yaw_r2
0.7575`.

⇒ **REF-A's latent encoded ego-motion because it was supervised to. v6's does not
(−0.005) because nothing asks it to.** The head is already implemented in
`stack/tanitad/models/inverse_dynamics.py`.

⚠️ **Do not confuse the two speed mechanisms.** v6 *receives* `v0` as an action
channel (the validated speed-input fix). That supplies speed as an **input**; it
does not require the latent to **encode** it. Only a supervised head or an IDM
does.

## 4. ⛔ A confound in our own frozen-vs-trained comparison

REF-A vs flagship/v6 differs on **two axes at once**: *frozen-vs-trained* **and**
*grounded-vs-ungrounded*. Any "frozen doesn't work" or "trained doesn't work"
read off that pair is **non-attributable** — the same two-variables-at-once error
as E-ACTSTREAM-2 and C6.

⚠️ This also weakens my own **C129** reasoning. I retracted *"decodability implies
driving"* by pointing at REF-A driving worse — but that comparison is confounded,
**and a banked primary says the opposite**: `2602.04880` — *"Probing accuracy for
environment STATE **correlates** with downstream policy performance."* ⇒ The
correct position is **neither** *"decodability implies driving"* **nor**
*"decodability is irrelevant"*: it is **evidence, of a strength the literature
supports and our own confounded pair cannot settle.**

## 5. What to improve — the TRAINED WM

1. ⭐ **Anchor the target (highest leverage).** Predict an **external, non-drifting**
   future feature — DeepSight's recipe is MSE against **DINOv3 future features**.
   This keeps our encoder **trainable** (so `D-003`'s from-scratch main track is
   untouched) while making the target impossible to game.
   ⇒ **"Frozen TARGET, trained ENCODER"** is a combination neither `D-003` nor the
   literature forbids, and **nobody in this programme has run it.**
2. **Add grounding.** Wire `InverseDynamicsHead` (and/or aux ego-motion) into
   S-W. Cheap, already implemented, and directly targets the −0.005.
3. **Then re-run `E_TRUNK_2`.** It is 0-GPU on banked features and would say
   within hours whether either change moved decodability.

## 6. What to improve — the FROZEN WM (REF-A v1)

Against `FROZEN_ENCODER_LITERATURE.md` §5's ranked axes:

| axis | status |
|---|---|
| 3 — encoder strength (frozen+weak is FROST-Drive's **worst** arm) | ✅ addressed if v1 is DINOv3-L, not DINOv2-B/86M |
| 4 — interface width (*"cheapest to fix"*) | ✅ addressed at 640 tokens / 120° vs 256 / 51.39° |
| **1 — what the objective asks of the features** ⭐ *prime suspect* | ⚠️ must be **feature prediction**, not an adapter re-encoding into a control manifold |
| **2 — where behaviour comes from** ⭐ *prime suspect* | ⚠️ must be **CEM/MPC at test time** or **head + world-model auxiliary** — never feed-forward regression alone |

⛔ **The blocker the review already flagged:** adopting DINO-WM's recipe imports
our **measured-broken CEM planner**. Axis 2 cannot be satisfied by intention —
the action search must be repaired or replaced, and **the planner is part of the
arm under test**.

## 7. ⚠️ What this document does NOT claim

* **No T1 claim.** Every one of our numbers is T0. Whether either fix improves
  *driving* is unmeasured, and C129 exists because I crossed that line once today.
* **Not a verdict on JEPA/LeJEPA.** It is a measured outcome for **this corpus,
  this scale, this configuration** — 2,376 episodes, 130-clip probe, linear probe.
* **`refa.py` in-tree is still the OLD geometry** (`[T, 256, 768]` = DINOv2-B/14
  @224). The PI states REF-A **v1** was redesigned for DINOv3; **I have not
  verified which design is implemented versus documented**, and §6 assumes the
  redesign only where marked ✅.
* The three fixes in §5/§6 are **hypotheses with mechanisms**, and each needs
  pre-registration before it decides a GPU-day.
