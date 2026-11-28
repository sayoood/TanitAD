# WP-I — a strong pretrained backbone carries agent semantics that REF-C's trunk does not

**TanitAD_TrainingFlyWheel · 2026-09-06 · Tier T0, NON-PARITY pilot corpus.**
Evidence class **MEASURED (ours)**.
Code `code/wpi2_semantic_floor.py` · raw `raw/wpi2_semantic_floor.json`,
`raw/wpi2_power_audit.json`, `raw/wpi2_null_adjusted.json`
Precondition: `RESULT_WPI_FOV_FRAME_DEFECT.md` (the target's frame).
Answers `H-TLANG-SEM-1`; bears on `DIALOGUE_07` §7(a).

---

## 0. The question and the one-line answer

The PI asked whether we can *"inject semantics about the physical world into the latent…
maybe by using in some stages of the training a strong pretrained vision backbone."*

> ⭐ **MEASURED: DINOv3 ViT-L/16 is the ONLY arm of four that beats its own
> within-episode chance level on agent presence (`vru<60m IN-FOV`, p 0.000) and on scene
> crowding (p 0.000). Neither REF-C's trained trunk nor a RANDOM-WEIGHT copy of its own
> architecture clears that bar on either VRU target.** ⇒ the distillation route of
> `DIALOGUE_07` §7(a) is **OPEN**, and it is the strongest lever this panel found.

---

## 1. Design — four arms, identical everything

`n = 1,800` windows · 15 episodes · 120/episode · leave-one-**episode**-out · ridge in
the dual form so no arm is penalised for its width · λ chosen inside each fold's training
data only · 200 permutations per null.

| arm | what it is | d |
|---|---|---:|
| `pixels` | raw 9-channel frame, pooled 2×4 | 72 |
| `rand` | **the same ResNet architecture, random weights**, pooled 2×4 | 5,632 |
| `trunk` | REF-C's trained encoder, pooled 2×4 | 5,632 |
| `dinov3` | DINOv3 ViT-L/16 frozen, all 3 sub-frames, pooled 2×4 | 24,576 |

⭐ `rand` is the arm that makes this readable. It separates *"a deep conv stack pooled to
2×4"* from *"features REF-C's training produced"* — without it, any trunk number is
uninterpretable.

⛔ Every arm sees the **same windows** and the **same 2×4 spatial grid** (the WP-J lesson),
and DINOv3 is given **all three sub-frames** so the trunk keeps no temporal advantage
(the same lesson in the time axis).

---

## 2. ⛔ THE READING RULE — the raw AUC column is NOT comparable across arms

The four arms do **not share a chance level**. MEASURED within-episode permutation null
means on `vru<60m IN-FOV`:

```
   pixels 0.4758      rand 0.5256      trunk 0.5786      dinov3 0.4833
```

A null at **0.5786** means that on this target, with this estimator, an arm with the
trunk's geometry scores 0.579 **on labels carrying no information at all**. The four
ridges have widths 72 / 5,632 / 5,632 / 24,576 on ~1,700 rows, and a wider basis absorbs
more of the episode structure that survives a within-episode permutation.

⇒ **The admissible statistics are the permutation `p` and the null-adjusted delta
`AUC − that arm's own null mean`.** The raw column is reported for transparency and must
not be read down.

⚠️ **And the paired episode-cluster CIs on RAW differences are confounded** — a
`dinov3 − trunk` interval on raw AUC mixes the representational difference with the
difference in estimator bias. They are banked for continuity, not used to decide. With 15
clusters they are wide enough that every one overlaps zero, which is itself worth saying
plainly: **at 15 episodes this design cannot separate arms by interval, only by
permutation.**

---

## 3. The panel

`p_w` = within-episode permutation p (the conservative null; episode prevalence held
fixed). `adj` = AUC − that arm's own within-null mean.

| target | base | n⁺ | pixels adj (p_w) | rand adj (p_w) | trunk adj (p_w) | **dinov3 adj (p_w)** |
|---|---:|---:|---|---|---|---|
| `vru<20m IN-FOV` | 0.038 | 69 | −0.0395 (0.805) | −0.1352 (0.985) | +0.0059 (0.475) | +0.0366 (0.300) |
| `vru<60m IN-FOV` | 0.127 | 228 | +0.0566 (0.035) | −0.0705 (0.955) | −0.0868 (0.915) | **+0.1939 (0.000)** |
| `crowd ≥3 in FOV` | 0.568 | 1023 | +0.0009 (0.495) | −0.0460 (0.845) | +0.0356 (0.130) | **+0.0816 (0.000)** |
| `lead < 25 m` | 0.198 | 357 | **+0.0779 (0.000)** | −0.1774 (1.000) | −0.1336 (0.995) | +0.0034 (0.495) |

**Null-adjusted arm differences** (the deciding read):

| target | dinov3 − trunk | trunk − rand | trunk − pixels |
|---|---:|---:|---:|
| `vru<20m IN-FOV` | +0.0306 | +0.1411 | +0.0455 |
| `vru<60m IN-FOV` | **+0.2807** | −0.0164 | −0.1434 |
| `crowd ≥3 in FOV` | +0.0460 | +0.0816 | +0.0347 |
| `lead < 25 m` | +0.1370 | +0.0439 | −0.2115 |

Raw AUCs, for the record only:

| target | pixels | rand | trunk | dinov3 |
|---|---:|---:|---:|---:|
| `vru<20m RAW (ungated)` | 0.5844 | 0.4070 | 0.6630 | 0.4785 |
| `vru<60m RAW (ungated)` | 0.5326 | 0.4232 | 0.4878 | 0.7292 |
| `vru<20m IN-FOV` | 0.6330 | 0.5270 | 0.5554 | 0.6176 |
| `vru<60m IN-FOV` | 0.5324 | 0.4552 | 0.4917 | **0.6772** |
| `crowd ≥3 in FOV` | 0.6170 | 0.6215 | 0.6935 | **0.7402** |
| `lead < 25 m` | 0.7155 | 0.5069 | 0.5835 | **0.7539** |

### What the panel says

1. ⭐ **DINOv3 clears the within-episode null on `vru<60m IN-FOV` (p 0.000) and on
   `crowd ≥3` (p 0.000). No other arm does.** On the VRU target its null-adjusted margin
   is **+0.1939** against the trunk's **−0.0868** — a **+0.2807** null-adjusted gap. On
   `crowd` the gap is +0.0460 and on the near VRU band +0.0306.
2. ⛔ **NEITHER the trunk NOR the random-weight architecture floor clears its own
   within-episode null on EITHER VRU target** — trunk p_w 0.475 / 0.915, rand 0.985 /
   0.955. On agent presence, 30 k steps of REF-C training did not buy a representation
   that separates pedestrian-bearing windows from empty ones within a clip.
   ⚠️ **Stated exactly, because the weaker and the stronger claim differ.** The trunk is
   *not* uniformly indistinguishable from `rand`: null-adjusted it is **+0.1411** better on
   the near band and **−0.0164** on the far band. The defensible statement is that
   **neither reaches significance**, not that they are equal — a null result about two
   arms is not a claim that the arms are the same, and `H-ESTIM-SEED-1` forbids reading it
   that way.
3. **`lead < 25 m` is the one target the pixel floor wins outright** (p_w 0.000, the only
   arm with within-episode signal there). A large near vehicle changes coarse photometry;
   it does not need semantics. ⚠️ That is also a warning about the *original* WP-J positive
   control — `lead<25m` was never going to discriminate representations.
4. **Crowding is mostly an episode-level property**: pixels, rand and trunk all reach
   p_glob 0.000 and p_within ≥ 0.13. Only DINOv3 tracks it *within* a clip.

---

## 4. Controls — what passed, what fired, and why the flag is wrong

| control | reads | verdict |
|---|---|---|
| CONSTANT-only | **0.5000** exactly | ✅ the no-information value |
| PC-PIXEL (brightness > median) | pixels **0.9970** > rand 0.9737 > trunk 0.7709 > dinov3 0.7373 | ✅ won by the pixel arm, as required |
| ⚠️ PC-PIXEL under null-adjustment | rand **+0.4330** > pixels +0.3073 > dinov3 +0.2786 > trunk +0.2095 | the ORDER FLIPS. The control is specified on raw AUC and passes there; the flip is a **saturation artifact** (pixels sit at 0.9970 with a 0.6897 null, so there is no headroom left to be credited). ⇒ null-adjustment is the right lens for a *discriminative* target and the wrong one for a *saturated* one — say which lens before reading either. |
| PC-TRUNK (REF-C's own selected-anchor lateral sign) | trunk **0.9400** > dinov3 0.7866 > rand 0.5771 > pixels 0.3787 | ✅ won by the trunk arm, by a wide margin |
| feature health | all four arms: `nonfinite 0`, `const-cols 0` | ✅ (this is the check that caught the fp16 disaster) |
| within-null power | 6–10 of 15 episodes non-constant per target; 69–1023 within-episode positives; **all six targets USABLE** | ✅ `p_within` is admissible, not an artifact of a constant target |

⭐ **The two positive controls are the reason this panel is readable at all.** Each is won
by exactly one named arm — the pixel floor demonstrably works, and the trunk extraction
demonstrably works. WP-J's `lead<25m` control was passed by *both* arms and therefore
validated nothing; that failure is what this design fixes.

### ⛔ The `controls_ok` gate printed FAIL. The gate is wrong, not the data.

The gate required **every arm's null mean to sit within 0.05 of 0.5000**, and fired on the
trunk's 0.5786. ⚠️ **That condition re-imposes exactly the assumption a permutation test
exists to avoid.** The whole point of generating the null through the same pipeline is
that the pipeline's bias is absorbed; demanding the null be centred at the theoretical
value throws that away.

This is the **third** time in this campaign that a control's *gate* — not the control, and
not the data — has been the defect (WP-B judged a shuffled arm against MAJORITY when a
noise fit lands near UNIFORM; WP-I v1 used a point threshold on a statistic with ±0.15
uncertainty; now this). The pattern is worth naming: **a control needs a decision rule
derived from the control's own distribution, not from a remembered constant.**

---

## 5. The FOV gate — how much the frame defect was worth

`in-FOV target minus raw target`, per arm (from `RESULT_WPI_FOV_FRAME_DEFECT.md`, the
correction applied here):

| target | pixels | rand | trunk | dinov3 |
|---|---:|---:|---:|---:|
| `vru<20m` | +0.0486 | +0.1200 | −0.1076 | +0.1391 |
| `vru<60m` | −0.0002 | +0.0320 | +0.0039 | −0.0520 |

⇒ the gate matters most exactly where the defect was largest — the **NEAR** band, where
30.4 % of positives were unanswerable. Three of four arms improve there once the
unanswerable positives are removed; the trunk does not, which is consistent with the trunk
having no VRU signal to recover.

---

## 6. What this changes

| | |
|---|---|
| ⭐ **`DIALOGUE_07` §7(a) — distillation** | **OPEN and promoted.** It was gated on "does the teacher carry anything the trunk does not". It does, on the two targets that matter for the PI's crowded-scene example. |
| ⛔ **The trunk as the reasoner's only ground** | **Refuted as sufficient.** A reasoner reading `cond₀` alone would be reasoning over a representation that cannot distinguish a pedestrian-bearing window from an empty one better than random weights. |
| ⭐ **refcv5's agent tokens** | **Their value goes up.** The audit established refcv5's grounding is agent tokens (`[B,100,384]`, default OFF) and that no BEV or map token exists. This panel says the trunk will not supply agent semantics on its own — so those tokens, or a distilled teacher, are not optional extras but the load-bearing path. |
| ⚠️ **WP-H / P4-3c "the trunk is agent-blind"** | **Now supported by an independent measurement** — but for a reason WP-H could not have established, since its own target was in the wrong frame and its floor was unmatched. Support, not vindication. |

---

## 7. Scope, and what would overturn this

- **T0. NON-PARITY pilot corpus. 15 episodes.** Not a driving-performance number, and it
  carries no metric family. A capability claim needs T1 and the four families.
- **One seed, one checkpoint** (`refc-base-30k`). ⛔ Per `H-ESTIM-SEED-1` a separated
  interval on a one-seed arm is necessary and not sufficient — and here **no interval
  separates at all**; the evidence is the permutation p.
- **The trunk is read as a POOLED 2×4 feature map.** A negative from a linear probe on a
  pooled representation is not a negative about the representation: the trunk may encode
  agents in a form this pooling destroys or this linear map cannot read. ⚠️ **The
  implication runs one way only.** The honest statement is *"the trunk's 2×4-pooled
  features are not linearly separable on VRU presence, and a random-weight copy is not
  worse"* — not *"REF-C cannot see pedestrians."*
- **What would overturn it:** a nonlinear probe, or a finer pooling grid, on which the
  trunk beats `rand` on the VRU targets. That is the cheapest next experiment and it is
  the one to run before any distillation head is built.

---

## 8. Next levers, in the order the evidence ranks them

1. **Finer pooling + a nonlinear probe on the trunk** — settles the scope caveat in §7 for
   ~1 GPU-hour, and it must run **before** distillation is built, because if the trunk
   does carry agents at 4×8 with an MLP, the distillation head is solving a pooling
   problem, not a semantics problem.
2. **Distillation head**: predict DINOv3's pooled features from `cond₀` as an auxiliary
   loss on a REF-C arm, with the mandatory deliberate-regression arm (a head predicting
   *shuffled* teacher features must not help).
3. **Turn refcv5's agent tokens ON** (`AgentSeamConfig.enable` + `DecoderConfig.cross_agent`,
   both default `False`) and re-run this panel reading the agent-token bus rather than
   `cond₀`. ⛔ Blocked: the seams are not in `HEAD` (BACKLOG escalation, 2026-09-06).
4. **More episodes.** 15 clusters cannot separate anything by interval. The agent-join
   extension is the unblock.

---

# ⛔⛔ AMENDMENT 1 (2026-09-06, same night) — the JOIN THIS PANEL SCORED AGAINST IS CONTAMINATED

**Every number above is computed against `pilot_val_agents.jsonl`, and MEASURED by the
DataFlyWheel: 4 of its 15 val episodes (26.7 %) carry agent boxes from the WRONG CLIP.**

`physicalai.py:740` derives `episode_id` from the **first four characters** of the clip
uuid, and the join builder's ambiguity guard asks *"is this prefix unique among the cached
zips?"* rather than *"which clip is this?"*. So `ep_00013`, `ep_00065`, `ep_00076` and
`ep_00087` were joined to **real cuboids from a real but different recording** — and
nothing downstream shows a symptom. (The train half is worse: **13 of 54**. `ep_00107` and
`ep_00108` were assigned the *same* clip; 54 `clip_id`s map to 53 distinct real clips.)

⇒ **the VRU, crowding and lead targets on those four episodes describe agents that were
never in front of this car.**

### What that does and does not touch

| | |
|---|---|
| **Most likely to survive** | the **arm ORDERING**. The contamination is **common-mode** — all four arms are scored against the same wrong labels — and label noise attenuates every arm in expectation rather than reordering them. |
| **Not trustworthy** | every **magnitude**: AUCs, null-adjusted deltas, the +0.2807 gap. Label noise on 27 % of clusters attenuates toward chance by an unknown amount. |
| ⛔ **Structurally damaged** | anything **per-episode**. The within-episode permutation null and the leave-one-episode-out folds both take the episode as the unit, and 4 of 15 units are wrong. `p_within` is the panel's deciding statistic, so this is the worst-placed of the three. |
| **Unaffected** | `PC-PIXEL` and `PC-TRUNK`. Neither target comes from the join — one is frame brightness, the other REF-C's own selected-anchor sign. **Both positive controls still stand**, which is the only reason the panel's machinery can be trusted at all. |
| **Unaffected** | the **encoder-vs-join frame mismatch** (`RESULT_WPI_FOV_FRAME_DEFECT.md`). That is a geometric fact about two `CanonicalFrame` declarations, independent of which clip the boxes came from. ⚠️ Its *percentages* are computed over the contaminated join and are re-measured in v3. |

### ⛔ A third target defect, found in the same audit

**292 frames lie outside their clip's obstacle label span and carry `agents: []`.** This
panel read every empty list as "no VRU", i.e. as a **NEGATIVE**. It is **NO_LABEL, not
road-clear** — the same absence-is-not-evidence error as a search reporting "no matches"
for a file it could not open. v3 masks on the per-episode `label_span_s`.

### The re-run, launched in the same turn

`code/wpi3_semantic_floor.py` — identical arms, identical 2×4 pooling, identical
encoder-frame azimuth gate, identical permutation nulls and positive controls, against the
**repaired** join:

* **identity by CONTENT**: every prefix candidate is fitted to the episode's own pose track
  (`register_poses_to_time`); a candidate that is not the clip cannot register. Median
  residual **0.00105 m** vs **4.403 m** for a rejected sibling — ~1,600× separation.
  100/100 val episodes resolved to exactly one candidate.
* **the overlap is byte-clean**: on the 11 episodes whose identity was already correct, old
  and new agree on **2,189/2,189 frames (100.0 %)** — so the repair perturbs nothing that
  was right.
* ⭐ **15 → 34 episodes.** Episodes are the clustering unit, so this is the power axis that
  actually matters; windows per episode drop 120 → 60 to keep the run inside the night.
* NO_LABEL frames masked.

⇒ **Until v3 lands, treat §3's table as an ORDERING claim only, and §0's headline as
provisional.** The design consequence in §6 — that the trunk cannot be the reasoner's only
ground — rests on the ordering and is the part most likely to hold; the distillation
decision should nevertheless wait for v3.

⚠️ **Escalation beyond this panel:** both live pilot joins are contaminated, not merely
low-power. **Every banked probe or pilot result scored against `pilot_val_agents.jsonl` or
`pilot_train_agents.jsonl` needs re-reading** — WP-F, WP-G and WP-H/WP-J among them. That
is a programme decision, not this stream's to make alone; raised to the Master Mind.

---

# ⭐ §9 — WP-I v3: THE RESULT ON THE VERIFIED JOIN (this section supersedes §0–§6)

**2026-09-06 · Tier T0, NON-PARITY pilot · MEASURED (ours).**
Join `pilot_val_agents_ext.jsonl` — clip identity resolved by **content registration**.
`n = 1,954` windows · **33 episodes** (vs 15) · 60/episode · NO_LABEL frames masked ·
leave-one-episode-out · 200 permutations per null.
Code `code/wpi3_semantic_floor.py`, `code/wpi3_power_audit.py` ·
raw `raw/wpi3_{semantic_floor,power_audit,null_adjusted}.json`, `raw/wpi3_full.log`.

`adj` = AUC − that arm's **own** within-episode null mean (the only cross-arm comparable
column). `p_w` = within-episode permutation p. Within-null power: **all four targets
USABLE** (11–23 of 33 episodes non-constant; 92–1,306 within-episode positives).

| target | base | n⁺ | pixels adj (p_w) | rand adj (p_w) | trunk adj (p_w) | **dinov3 adj (p_w)** |
|---|---:|---:|---|---|---|---|
| `vru<20m IN-FOV` | 0.047 | 92 | +0.0169 (0.315) | +0.0596 (0.145) | +0.0943 (0.110) | **+0.2401 (0.000)** |
| `vru<60m IN-FOV` | 0.149 | 291 | −0.0011 (0.475) | −0.0152 (0.630) | +0.0162 (0.355) | **+0.0797 (0.020)** |
| `crowd ≥3 in FOV` | 0.668 | 1306 | +0.1084 (0.000) | +0.0797 (0.000) | **+0.1110 (0.000)** | +0.0880 (0.000) |
| `lead < 25 m` | 0.287 | 560 | +0.0258 (0.020) | +0.1109 (0.000) | +0.1171 (0.000) | **+0.2415 (0.000)** |

**Controls — all three pass.**

| control | reads | |
|---|---|---|
| CONSTANT-only | **0.5000** exactly | ✅ |
| PC-PIXEL (brightness) | pixels **0.9989** > rand 0.9800 > dinov3 0.9111 > trunk 0.7694; trunk `p_w` 0.850 | ✅ won by pixels alone |
| PC-TRUNK (REF-C's own selected-anchor sign) | trunk **0.9232**, adj **+0.3428**, `p_w` 0.000; pixels 0.4687, `p_w` 0.900 | ✅ won by the trunk alone |

## What REPLICATED

⭐ **The headline survives the label repair and gets STRONGER.** On both agent-presence
targets, **DINOv3 is the only arm of four that clears its own within-episode null** —
`vru<20m` p 0.000 (v2: 0.300) and `vru<60m` p 0.020 (v2: 0.000). Null-adjusted it leads
the trunk by **+0.1458** near and **+0.0635** far.
⇒ **`H-TLANG-SEM-1` is SUPPORTED on the verified join.** The distillation route of
`DIALOGUE_07` §7(a) is open, and this is the panel's strongest lever.

## ⛔ What did NOT replicate — two v2 claims corrected

1. ⛔ **"DINOv3 is the only arm that tracks CROWDING within a clip" is WITHDRAWN.**
   On the verified join **all four arms clear** `crowd` at p 0.000, and null-adjusted
   DINOv3 is **not** best: trunk **+0.1110** ≈ pixels +0.1084 > dinov3 +0.0880
   (`dinov3 − trunk = −0.0230`). The v2 result was an artifact of the contaminated join
   and 15 episodes. Crowding is legible to *any* of these representations.
2. ⚠️ **"The trunk has no VRU signal / is indistinguishable from random weights" is
   TOO STRONG.** With correct labels the trunk is **marginal, not absent** on the near
   band (adj +0.0943, p 0.110 — the closest any non-teacher arm comes), and it clearly
   clears `crowd` (+0.1110) and `lead` (+0.1171). The defensible statement is narrower:
   **only DINOv3 reaches significance on agent presence**, not that the trunk is empty.
3. ⚠️ **`lead < 25 m` flipped owner.** v2 had the pixel floor winning it (0.7155, the only
   within-episode signal). v3 gives **dinov3 0.9490, adj +0.2415**, with pixels weakest of
   the four (+0.0258). A near lead vehicle *is* partly photometric, but with correct labels
   it is far more semantic than v2 suggested.

⇒ these three are exactly the categories Amendment 1 predicted: the **ordering on the VRU
targets survived**, the **magnitudes moved**, and the **per-episode statistics changed most**.

## The FOV gate, re-measured on correct labels

`in-FOV minus raw`, per arm:

| target | pixels | rand | trunk | dinov3 |
|---|---:|---:|---:|---:|
| `vru<20m` | −0.0184 | −0.0041 | **+0.0688** | **+0.1197** |
| `vru<60m` | +0.0153 | +0.0124 | +0.0301 | −0.0126 |

⭐ The azimuth gate helps **exactly the two arms that carry semantics**, and **only in the
NEAR band** — where 30.4 % of positives were unanswerable. The floors gain nothing from it,
which is what it should look like if the gate removes unanswerable labels rather than
adding signal.

## ⛔ The gate that cried wolf, fixed in code

`controls_ok` printed **CONTROLS FAIL** on both v2 and v3 — sound panels each time. It
required every arm's null mean within 0.05 of 0.5000, **re-imposing the very assumption a
permutation test exists to avoid**. Third occurrence in this campaign of a control's
*decision rule* being the defect.

**Fixed:** the **global** null must sit near 0.5 (it permutes across everything, so it has
no structure left to fit), while the **within-episode** null is allowed to sit anywhere —
that offset *is* the estimator bias the p-value already measures against, and it is now
printed rather than gated (v3: pixels 0.6053 · rand 0.4319 · trunk 0.4781 · dinov3 0.5414).

## Next lever, unchanged in rank and now better justified

**A finer pooling grid and a nonlinear probe on the trunk**, before any distillation head is
built. The trunk at **p 0.110** on the near band is close enough that a 4×8 grid or an MLP
could move it across — and if it does, a distillation head would be solving a *pooling*
problem rather than a *semantics* one. That distinction decides the whole of stage C.
