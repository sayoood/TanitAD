# The `vru_ahead` target was asking about agents the encoder never saw

**WP-I precondition · TanitAD_TrainingFlyWheel · 2026-09-06**
Tier **T0**, NON-PARITY pilot corpus. Evidence class **MEASURED (ours)**.
Raw artifact: `raw/fov_frame_audit.json` · code: `code/wpi2_semantic_floor.py` (§docstring)
Affects: `RESULT_WPH_TRUNK_VS_PIXELS.md` (WP-H and its WP-J retraction), and any
P4-3c-derived statement that REF-C's trunk is "at chance on `vru_ahead`".

---

## 1. The finding, in one line

> ⛔ **The agent-join label is computed in the SENSOR's 120° frustum; the encoder is
> fed a 51.4° square crop. 48.5 % of forward `automobile` and 46.7 % of forward
> `person` cuboids lie outside the encoder's field of view — so a large share of every
> `vru_ahead` positive is a label for an agent that is not in the input.**

This is not a modelling result. It is a **specification defect in the target**, and it
sits underneath every arm that has been compared on that target.

---

## 2. The two frames, each read from source

| | half-angle | HFOV | where it comes from |
|---|---|---|---|
| **the JOIN's `occ` flag** | 60.00° | 120° | `stack/scripts/build_obstacle_join.py` — `--hfov-deg` default is the **sensor** field, `camera_front_wide_120fov` |
| **the ENCODER's frame** | **25.70°** | **51.4°** | `tanitad.data.physicalai.as_frame(None, 256, F_REF)` → `CanonicalFrame(height=256, width=256, f_ref=266.0, projection='pinhole')`; `atan(128/266) = 0.4485 rad` |

⭐ **The join's own docstring warns about exactly this**, verbatim: *"PASS THE
ENCODER'S FRAME INSTEAD when the consumer was fed a centred sub-frame — v5f's 176x624
crop is 117.0 deg, and the 1.5 deg annulus it removes is flagged `visible` here while
the encoder never saw it."* The warning was written for a **1.5°** annulus. Here the
annulus is **68.6° wide**.

⚠️ **The projection is stated, not assumed.** `CanonicalFrame` declares
`projection='pinhole'`, so the pinhole half-angle `atan((W/2)/f)` is the correct
formula for this artifact. (Do not import the cylindrical corpus's numbers here: that
corpus is 256×640 with `f_ref` 305.577 and a column **linear in azimuth**, giving 120°.
Two different corpora, two different formulas — quoting either outside its own
projection is the FOV trap in `CLAUDE.md`.)

---

## 3. The measurement

Over all 2,983 join lines / 15 clips of `pilot_val_agents.jsonl`, azimuth
`|atan2(cy, cx)|` against the **encoder's** half-angle, forward agents only (`cx > 0`):

| class | n (forward) | in encoder FOV | fraction |
|---|---:|---:|---:|
| automobile | 31,224 | 16,086 | **0.515** |
| person | 8,688 | 4,632 | **0.533** |
| bus | 732 | 501 | 0.684 |
| rider | 577 | 381 | 0.660 |
| heavy_truck | 310 | 208 | 0.671 |
| trailer | 122 | 55 | 0.451 |
| protruding_object | 76 | 59 | 0.776 |
| stroller | 58 | 47 | 0.810 |

At the **window** level, for the canonical target (`cls ∈ {person, rider, stroller,
animal}`, `|cy| ≤ 4.0 m`):

| target | positive windows | ≥1 positive **in** the encoder FOV | unanswerable share |
|---|---:|---:|---:|
| `vru < 20 m` (NEAR) | 158 | 110 | **30.4 %** |
| `vru < 60 m` (ALL) | 433 | 396 | **8.5 %** |

---

## 4. Why this inverts the WP-H reading

WP-H's discriminator was: *decode the same target restricted to NEAR agents versus ALL
ranges; if NEAR separates and ALL does not, the input is the bottleneck (resolution),
not the objective.* The premise was that the NEAR band is **easier** because a nearby
VRU spans more pixels.

⛔ **The NEAR band carries 3.6× more unanswerable label than the ALL band** (30.4 % vs
8.5 %), and the mechanism is geometric and unavoidable: at a fixed lateral gate of
`|cy| ≤ 4 m`, azimuth grows as the agent gets **closer**. A VRU at `cx = 5 m, cy = 4 m`
sits at 38.7° — outside a 25.70° half-angle. At `cx = 20 m` the same lateral offset is
11.3° and comfortably inside.

⇒ the NEAR band is simultaneously **higher resolution** and **higher label noise**, and
those two effects run in opposite directions. WP-J's headline

```
  NEAR − ALL  =  −0.0562
```

therefore **cannot** be read as evidence against the resolution hypothesis. It is a
sum of a resolution effect and a frame-mismatch effect of unknown relative size.

**Status of the resolution hypothesis: NOT SUPPORTED, NOT REFUTED — and now for a
second, independent reason.** (The first was power: at base rate 0.058 only ~6
positives were scored.)

⚠️ **What is NOT retracted.** WP-J's primary correction stands: with **matched 2×4
pooling**, the trunk beat the pixel floor on both VRU targets (0.6683 vs 0.4758 near;
0.7245 vs 0.6951 all), so *"raw pixels beat the trunk"* remains **retracted**. The
present defect changes the reading of the **NEAR-vs-ALL contrast**, not the reading of
the **trunk-vs-floor contrast** — both arms saw the same frame and the same labels, so
the mismatch is common-mode for that comparison.

---

## 5. The repair, applied

`wpi2_semantic_floor.py` reports **both** forms of every VRU target:

```
  vru<20m RAW (ungated)   the historical definition — kept so the size of the
  vru<60m RAW (ungated)   artifact is visible rather than asserted
  vru<20m IN-FOV          gated by |atan2(cy,cx)| <= 25.70 deg
  vru<60m IN-FOV
```

and prints the per-arm **FOV-gate effect** (in-FOV AUC minus raw AUC) so the correction
is a measured quantity in the panel rather than a claim in prose.

---

## 6. Root-cause class

⭐ Same family as the `df` / Thor `free` / cgroup `usage_in_bytes` / `step_s` /
cylindrical-FOV / anchor-units traps already in `CLAUDE.md`: **a true quantity quoted
outside its scope**, with the scope being the **FRAME**.

It is the *ninth* member of that family to bite this programme and the **seventh
specification defect in this campaign alone** — and every one of the seven has the same
shape: **two things compared that were not given the same affordances.**

| # | defect | the mismatched affordance |
|---|---|---|
| 1 | WP-B window-level features | features identical across candidates ⇒ only the marginal was learnable |
| 2 | WP-B control spec | shuffled arm judged against MAJORITY when a noise fit lands near UNIFORM |
| 3 | WP-F run 1 | arms denied the rank information the baseline used |
| 4 | WP-F validity/effect | the effect test placed inside the validity gate |
| 5 | WP-H AUC ties | constant predictor read 0.6765 instead of 0.5000 |
| 6 | WP-H/J pooling | trunk pooled to one vector, pixels to a 4×8 grid |
| **7** | **this one** | **the label's frame ≠ the arm's frame** |

⇒ **The generalised guard, now stated once so it can be applied without re-deriving
it:** before comparing two things, enumerate every affordance — spatial structure,
dimensionality, temporal extent, **field of view**, the information the baseline is
given — and assert parity on each. Provenance parity is not affordance parity.

---

## 7. Consequences to carry forward

1. Any statement of the form *"the trunk is at chance on `vru_ahead`"* sourced from
   P4-3c must be re-read against the **IN-FOV** target before it is quoted again. It is
   not retracted here — it is **not yet admissible** until re-measured.
2. `build_obstacle_join.py` should be run with `--hfov-deg 51.4` for any join whose
   consumer is the square 256 frame; the default 120° is correct only for a
   full-frustum consumer. *(Escalated as an integration item rather than left in a doc
   — the join is shared machinery.)*
3. The lateral gate `|cy| ≤ 4.0 m` is a **rectangular** predicate applied to a
   **conical** field of view. An azimuth gate is the geometrically correct form and is
   what the IN-FOV targets use.
