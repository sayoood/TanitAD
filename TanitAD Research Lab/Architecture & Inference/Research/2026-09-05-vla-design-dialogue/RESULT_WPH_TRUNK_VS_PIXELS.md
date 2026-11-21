# RESULT — WP-H: the resolution hypothesis is NOT supported, and RAW PIXELS BEAT THE TRUNK

**2026-09-06 · TanitAD_TrainingFlyWheel · Tier T0 · ⚠️ NON-PARITY pilot, n=600, 15 episodes,
episode-disjoint · Evidence class: MEASURED (ours) · ⛔ REF-C's encoder, NOT champ30k.**

---

## 1. The numbers

| target | trunk AUC | **raw-pixel floor** | base rate |
|---|---|---|---|
| vru ahead < 20 m (NEAR) | 0.6070 | **0.8142** | 0.058 |
| vru ahead < 60 m (ALL) | 0.8026 | **0.8329** | 0.142 |

| control | value | required |
|---|---|---|
| CONSTANT-only | **0.5000** | ✅ exactly the no-information value |
| SHUFFLED-target | 0.5876 | ⚠️ should be ~0.5 — **0.088 of residual optimism** |

**NEAR − ALL = −0.1957.**

## 2. ⛔⛔ THE HEADLINE: the learned trunk LOSES to 4×8 average-pooled pixels

`CLAUDE.md`'s own probe rule: *"a RAW-INPUT floor (pixels) — a learned representation that does not
beat raw input has added nothing."*

**The floor is a 4×8 average-pool of the input frame. It beats REF-C's 704-d encoder features on
BOTH targets** — decisively on the near band (0.8142 vs 0.6070).

⇒ **For VRU presence, REF-C's encoder has not merely failed to add information — it has DESTROYED
information that survives naive average pooling.** That is a statement about the trunk, not about
the objective's difficulty, and it is the most consequential thing measured in this series.

⚠️ **The floor is not a weak baseline here — it is a strong one, and that is the point.** Average
pooling preserves coarse spatial colour/intensity structure; a person against tarmac is a
low-frequency signal that survives it. The trunk, trained for trajectory prediction, has evidently
learned to discard it.

## 3. The resolution hypothesis: NOT SUPPORTED, with a power caveat I will not paper over

NEAR − ALL is **−0.1957** — the near band decodes *worse*, the opposite of the prediction. On its
face the objective explanation survives and WP-G's resolution story does not transfer to decoding.

⚠️ **But the near arm is badly underpowered.** The canonical lateral gate (`|cy| ≤ 4.0`, which I had
to adopt for comparability) dropped the base rate to **0.058** — roughly **14 positives in 600
windows, ~6 in the scored split**. An AUC on six positives is not a measurement.

⇒ **Honest verdict: the resolution hypothesis is not supported by this run, and this run is not
strong enough to refute it.** ⛔ It must not be quoted as a refutation. The decisive version needs
either many more episodes or a base-rate-matched near/far comparison.

## 4. ⚠️ Three limits on everything above

1. **SHUFFLED reads 0.5876, not 0.500.** ~0.09 of optimism is present in the fitting setup and is
   shared by both arms. Differences of that magnitude are not readable; **the pixel-vs-trunk gap of
   0.207 on the near band is larger than it, the −0.196 near/far gap is not cleanly so.**
2. ⛔ **This is REF-C's encoder, not champ30k.** P4-3c is a champ30k finding. This makes the
   confound *plausible for REF-C* and says nothing settled about champ30k, whose checkpoint is on
   Thor. Same class as the earlier `refc.py`-vs-refcv5 scope error.
3. n=600 over 15 episodes, 6 of them scoring. Pilot corpus, NON-PARITY.

## 5. ⭐ What this changes — and it is larger than the reasoning question

If REF-C's trunk is worse than pooled pixels at seeing VRUs, then **every architecture discussed in
this dialogue — the latent reasoner, the relational graph, the language layer — is being designed on
top of a representation that does not carry the agents they must reason about.**

⇒ **The next question is no longer "which reasoner", it is "does the trunk carry agent
information at all".** That is answerable cheaply and it gates everything else:

* **WP-I** — replicate the pixel-vs-trunk comparison at power: more episodes, several targets
  (`vru_ahead`, `left_occupied`, `nearest_any_m`, and `lead_gap_m` as the positive control that
  MUST favour the trunk), base-rate matched. ⛔ If the trunk loses on agent targets while winning on
  `lead_gap_m`, the finding is real and specific.
* **WP-J** — if it replicates: is it the objective, the resolution, or the pooling? Compare
  mean-pooled trunk features against the **spatial** feature map, since mean-pooling over 8×20 cells
  may itself be destroying the localisation an agent target needs.

⚠️ **WP-J's hypothesis is the one I would bet on**, and it is embarrassing for the probe rather than
for the trunk: I pooled the trunk to a 704-d vector and compared it against pixels pooled to a 4×8
GRID. **The pixel floor kept spatial structure the trunk arm was denied.** That is not a fair
comparison, and it is the sixth specification defect in this series — caught while writing it up
rather than by a control.

---

## ⛔⛔ RETRACTION (WP-J, same day) — "RAW PIXELS BEAT THE TRUNK" IS WITHDRAWN

**It was my pooling, not the trunk.** §2 above compared a trunk pooled to a SINGLE 704-d vector
against pixels pooled to a 4×8 **GRID** — the floor kept spatial localisation the trunk arm was
denied. Re-run with **both sides on the same 2×4 grid**:

| target | trunk AUC | pixel floor | before (unfair) |
|---|---|---|---|
| vru ahead < 20 m | **0.6683** | 0.4758 | trunk 0.6070 vs pixels 0.8142 |
| vru ahead < 60 m | **0.7245** | 0.6951 | trunk 0.8026 vs pixels 0.8329 |
| lead < 25 m (POSITIVE CONTROL) | 0.7019 | 0.7097 | — |

| control | value | |
|---|---|---|
| CONSTANT-only | **0.5000** | ✅ |
| SHUFFLED-target | **0.4584** | ✅ much cleaner than the 0.5876 of the unfair run |

⇒ ⭐ **The trunk BEATS pixels on both VRU targets** — decisively on the near band (+0.19).
**REF-C's encoder is NOT blind to agents**, and the alarming §2 finding was an artifact of my own
comparison.

⚠️ **Why this matters beyond the correction:** §5 argued that every architecture in this dialogue
was being designed on a representation that does not carry the agents it must reason about.
**That argument is withdrawn.** The trunk carries VRU information.

⛔ **AND THE POSITIVE CONTROL IS NOT CLEAN, so the absolute numbers stay soft.** `lead < 25 m` is
the target the trunk should own, and it reads 0.7019 against a 0.7097 pixel floor — a tie, not a
win. With d = 5632 on 360 fit rows and λ driven to 1e3–1e4, the probe is heavily shrunk and
over-parameterised. ⇒ **Ordering within a run is readable; absolute AUCs are not**, and a stronger
version needs far more episodes or a lower-dimensional trunk read.

⚠️ **The resolution verdict is unchanged and still underpowered:** NEAR − ALL = −0.0562 at a 0.058
base rate (~6 positives in the scored split). Not supported, not refuted.

⭐ **The lesson, and it is the sixth in this series:** a floor is only a floor if it is given the
SAME representational affordances as the arm. Provenance parity (learned vs raw) is not enough —
**spatial structure, dimensionality and pooling must match, or the comparison measures the
harness.** Caught here by writing the result up, not by a control; the control that would have
caught it is the positive control, and it is now in the probe.

---

# ⛔ AMENDMENT 2 (2026-09-06) — the `NEAR − ALL` contrast is not readable as specified

**Full record: `RESULT_WPI_FOV_FRAME_DEFECT.md`. Registered as TRAIN-C13 / `D-TLANG-FOV`.**

The target this document's central discriminator rests on was built in the **wrong frame**.

| | half-angle | HFOV | source |
|---|---|---|---|
| the join's `occ` flag | 60.00° | 120° | `build_obstacle_join.py --hfov-deg` default = the **sensor** |
| the encoder's frame | **25.70°** | **51.4°** | `physicalai.as_frame(None, 256, F_REF)` → `CanonicalFrame(256×256, f_ref=266.0, projection='pinhole')` |

MEASURED over the 2,983 join lines: **48.5 %** of forward `automobile` and **46.7 %** of
forward `person` cuboids are outside the encoder's field of view; **30.4 %** of `vru<20m`
positive windows and **8.5 %** of `vru<60m` positive windows are positives for an agent
the encoder never saw.

⛔ **This runs OPPOSITE to the premise of the discriminator.** WP-H assumed the NEAR band
is easier because a nearby VRU spans more pixels. Geometrically, at a fixed lateral gate
`|cy| ≤ 4 m` the azimuth *grows as the agent gets closer* — so the NEAR band is
simultaneously higher-resolution **and** 3.6× noisier in its label. The headline

```
  NEAR − ALL  =  −0.0562
```

is a sum of a resolution effect and a frame-mismatch effect of unknown relative size and
**cannot be read as evidence against the resolution hypothesis**.

**Status of the resolution hypothesis: NOT SUPPORTED, NOT REFUTED** — now for two
independent reasons (power, and the frame).

⚠️ **What this amendment does NOT touch.** WP-J's correction stands in full: with matched
2×4 pooling the trunk beat the pixel floor on both VRU targets (**0.6683** vs 0.4758 near;
**0.7245** vs 0.6951 all), so *"raw pixels beat the trunk"* remains **RETRACTED** and §2/§5's
alarming argument remains **withdrawn**. Both arms saw the same frame and the same labels,
so the frame mismatch is **common-mode** for the trunk-vs-floor comparison and cancels.

**Repaired in:** `code/wpi2_semantic_floor.py`, which reports every VRU target in BOTH
forms — RAW (the historical definition) and IN-FOV (gated on azimuth rather than on the
rectangular `|cy|` predicate) — and prints the per-arm FOV-gate effect, so the size of the
correction is a measured quantity in the panel rather than a claim in prose.
