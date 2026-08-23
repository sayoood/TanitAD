# E-PROBE-POWER — the decoder is NOT the bottleneck

`MEASURED (ours; dev-box RTX 4060, 2026-08-21)` · **T0-DIAGNOSTIC** · same 5,617
keys, same **episode-disjoint** folds, same **episode-cluster bootstrap** as
E-TRUNK-2 — **only the decoder changes**.

⛔ **The PI's question:** *"Could our decoder be the problem? or how we decode?"*
Every null so far came from ONE probe — a **linear** ridge on a **single frame's**
latent. That is a real limitation and it had never been tested.

⭐ **It is also LeWM's own caveat**: their §5.1 reports linear **and** MLP probes,
and Block Angle goes **linear r 0.902 → MLP r 0.990**.

---

## 1. Result

| arm | `lead_gap_m` linear | **MLP** | window | `left_occ` linear | **MLP** |
|---|---|---|---|---|---|
| ⭐ **supervised CONTROL** | +0.9934 | **+0.9771** | — | .9941 | **.9875** |
| ⭐ **`dino_pooled`** | **+0.3792** | **+0.3056** | +0.3481 | .8312 | — |
| `v6_cells` | −0.0176 | −0.1477 | −0.0168 | .5391 | **.6019** |
| `cls192` | −0.0068 | −0.3411 | −0.0068 | .5124 | **.6111** |
| `v6shape` | −0.0087 | −1.1479 | −0.0102 | .5343 | .5428 |
| **`random` UNTRAINED** | −0.0156 | −0.6838 | −0.0160 | .5082 | **.5192** |

*(supervised also: `right_occ` MLP .9878, `ego_speed` MLP +0.9368)*

## 2. ⭐⭐ The answer: NO — three independent reasons

1. **The MLP finds information wherever it exists.** **+0.9771** on the
   supervised control, **+0.3056** on DINOv3. It is not blind.
2. ⭐ **It is WORSE than linear on the arm that HAS the content** — DINOv3
   +0.3056 vs +0.3792. A strictly more expressive decoder extracts **less**,
   because it overfits at n = 5,617. ⇒ **expressiveness was never the limit.**
3. **Temporal context changes nothing.** The window probe
   ([z_{t-2}, z_{t-1}, z_t]) gives −0.0068 — identical to single-frame. The
   model is a sequence model; the missing content is not hiding in the sequence.

⇒ ⛔ **When the MLP reads −0.3411 on `cls192` and −0.6838 on `random`, that is
overfitting on nothing — not a decoder failing to see something.**

## 3. ⭐ One real nuance, and the control that sizes it

Occupancy **did** lift under the MLP: `cls192` .5124 → **.6111**, `v6_cells`
.5391 → **.6019**. ⚠️ **But `random` lifted too** (.5082 → .5192), so the honest
margins over the untrained floor are **+0.092** and **+0.083**.

⇒ There **is** a little **non-linearly encoded occupancy** the linear probe
missed. It is real, it is small, and it is far below `dino_pooled`'s **.83**.
**It refines the picture; it does not change it.**

⚠️ Without the `random` arm this would have been reported as *"the MLP recovers
occupancy at .61"* — the same winner's-curse shape SEL-1 exists to refuse, in a
probe costume.

## 4. ⚠️ A defect this run exposed

⛔ **`supervised-CONTROL` was SILENTLY SKIPPED** in the batch pass: the latents
are `z_supervised_s0.npy`, the arm is named `supervised-CONTROL`, and the loader
printed `[skip] … latents absent` and continued. **The one arm that calibrates
the entire comparison was missing, and the run still "succeeded".** Fixed with a
name fallback. *(Same family as the `git add` exit-code rule: a tool reporting
success is not evidence its output is right.)*

## 5. What this licenses

✅ **Licensed:** *the decodability nulls are a property of the REPRESENTATIONS,
not of the probe. A non-linear decoder and a temporal decoder were tested against
a positive control (0.98) and an untrained floor, and neither changes the
verdict.*

⛔ **NOT licensed:** any claim that the representations contain nothing —
**linear + 2-layer MLP + 3-frame window is still a finite probe family**. Kernel
methods, deeper heads, or a spatial (patch-level) probe are untested.

⛔ Still no **T1/driving** claim, and still nothing about **LeWorldModel** — the
E-V6SHAPE gate remains open (Push-T never run).
