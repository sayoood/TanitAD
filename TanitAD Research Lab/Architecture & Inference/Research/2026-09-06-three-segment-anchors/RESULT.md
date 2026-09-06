# RESULT — D-TRISEG: the three-segment pulse family, run against its own bar

**Author:** Arch+Inference, three-segment anchor agent · **Date:** 2026-09-06
**Pre-registration:** `PREREG_THREE_SEGMENT_ANCHORS.md`, committed **`07290a0`
BEFORE any three-segment number existed in this turn.**
**Predecessor:** `.../2026-09-06-two-segment-anchors/` (`5e7001a`, `8c9d01c`, `f23c3ef`).

⭐⭐ **HEADLINE — AND THE SCOPE IS PART OF IT, NOT A FOOTNOTE.**
**All nine bars PASS.** **TWO** three-segment candidates beat the shipped **SIX**
two-segment ones on lane-change supply (**+0.2133 m** vs **+0.1645 m**) while
paying **30 % less curvature** on their own repicks and returning the heading the
two-segment family gave away. ⛔ **AND refcv4b HAS 117 LOGITS AND STRUCTURALLY
CANNOT SELECT THESE CANDIDATES.** What changes is the **SUPERVISION** — the
geometric `a_star` now points at a three-segment shape on **44.18 %** of
lane-change windows — **not the behaviour**. ⛔ **The dominant selection defect is
LONGITUDINAL and this is not a fix for it.**

---

## 0. THE INHERITED BAR, VERBATIM — and yes, I had to write it out

> **PRE-REGISTERED BAR for that arm, written now:** a 3-segment family PASSES if
> it beats the shipped 2-segment family on lane-change supply **and** costs less
> curvature on its own repicks, with the turn control still ≤ 0.01 m and R1/R2-style
> degenerate arms still reading exact zeros.
> — `.../2026-09-06-two-segment-anchors/RESULT.md` §10.2

⭐ **The bar was WRITTEN, it is REAL, and it is BINDING. It was also AMBIGUOUS in
four places, so I wrote the operational form FIRST and said so** (PREREG §0.1).
Every ambiguity was resolved to the **harder** reading:

| # | ambiguity | resolution taken |
|---|---|---|
| **A1** | the schedule `(t₁, t₂)` is not fixed | **RULE S** (§2), a stated rule — not the exploratory ranking |
| **A2** | *"the shipped 2-segment family"* = the **six** or the equal-cost **two**? | ⛔ **the SIX.** The test family is **two**, so this asks 2 to beat 6 |
| **A3** | *"curvature on its own repicks"* compares two **different** window sets | decided **as written**, with the **common-repick** read reported beside it |
| **A4** | *"R1/R2-style degenerate arms"* names none for this family | **four** committed, each with the reason its expectation is **structural** |

⛔ The inherited sentence is a **capability** bar and says nothing about parity,
the integrator limit, the 2 s grid, kinematics, or the artifact declaring its
extra columns. Those are inherited non-negotiables and were added as **B1, B2,
B5, B6, B7, B8**. **The arm does not ship if any of them fails, whatever B0 says.**

---

## 1. RULE S — and the check that it is not the exploratory table in disguise

**S1** `t₁` INHERITED from the shipped split grid `{2.0, 3.0, 4.0}` s (fixed at
commit `56dc078`, a probe predating every three-segment number), with its
`t ≥ 2.0 s` constraint. **S2** `t₂ = 2·t₁` DERIVED — the net-yaw-zero condition:
under `a_lon = 0` the yaw rate is constant, so heading returns iff the `−` segment
matches the `+` segment. **S3** drop any schedule whose third segment is empty
(`t₂ ≥ horizon`) — it **is** a two-segment candidate already in the shipped bank.
**S4** magnitudes inherited (`±0.75`, `a_lon = 0`).

⇒ **exactly one schedule survives: `t₁ = 2.0 s, t₂ = 4.0 s`. 2 candidates,
`2/117 = +1.71 %` — a THIRD of the two-segment family's `+5.13 %`.**

⚠️ **The predecessor's caveat is SCOPED, not ignored.**
`anchor_twoseg.net_yaw_zero_split_s` carries a measured warning that net-yaw-zero
is *not* where the two-segment gain lives. That is a fact about the **two**-segment
family, where net-yaw-zero **forces** the counter-steer to consume the whole
remaining horizon — the very defect §9 found. The third segment **decouples**
*"return the heading"* from *"spend the horizon"*.

### ⭐ THE POST-HOC CHECK, on my own re-derived numbers

**RULE S's row `2.0 → 4.0` is the ARGMAX OF NOTHING** (`raw/p2_triseg_decomp.py`,
computed, not asserted):

| criterion | winner | is it RULE S? |
|---|---|---|
| lane-change gain | `1.5 → 3.0` (0.2266) | ⛔ no |
| all-window gain | `2.0 → 3.0` (0.0592) | ⛔ no |
| lowest Δcurvature | `2.5 → 4.0` (+0.000519) | ⛔ no |
| lowest Δcross-track | `2.0 → 3.5` (−0.7217) | ⛔ no |
| lowest Δheading | `2.0 → 3.0` (−0.862) | ⛔ no |

⛔ **A rule that lands on a row best on no column cannot be that ranking wearing a
justification.** A post-hoc selector would have taken `2.0 → 3.0` — the row the
brief that sent me here named as the one *not* to take.
⭐ And `1.5 → 3.0`, the largest lane-change gain in the table, **repicks 720
windows on the 2 s grid** — it buys its gain by giving up the structural zero.
**The inherited `t ≥ 2.0` constraint is load-bearing, not decoration.**

---

## 2. The corpus

**141 B1-v7.2 EVAL clips, refcv4b T1 stride-1 dump: 24,114 dumped windows →
18,615** with a full 6.0 s recorded future. **1,297 lane-change (6.97 %)**,
2,602 turn, 14,716 other. `v0` 0.00–36.27 m/s.
Source `controls` sha256 `b072f4c0…89664`, read from `ckpt_40284_FINAL.pt`.

⭐ **The predecessor's numbers reproduce EXACTLY** on this corpus (`base117` LC
**1.5315**, `ext123` LC **1.3670**, straight-line floor **0.002726**, `A1` Δcurv
**+0.003361**), so the two results are on one measuring stick.

⛔ **THE VARIANCE, NAMED — AND THEREFORE NO INTERVAL.** Model-free, deterministic,
**every** scoreable window scored: no training draw (`H-ESTIM-SEED-1` does not
apply — nothing is trained), no inference sampling (nothing is planned), no
episode resampling. **There is no population being estimated, so NO CI IS
QUOTED.** Manufacturing one here would be an interval that answers no question.
⛔ **This is a SUPPLY CEILING and may never be quoted beside an achievement.**

---

## 3. THE BARS — PASS or FAIL as written

| bar | verdict | the number |
|---|---|---|
| **B0** inherited capability bar (4 clauses) | ⭐ **PASS** | see §4 |
| **B1** default-OFF sha256 parity | ⭐ **PASS** | §5 |
| **B2** limit bit-identical + mutation control | ⭐ **PASS** | §5 |
| **B3** supply ≥ 0.10 m, turn ≤ 0.01 m | ⭐ **PASS** | +0.2133 / +0.0031 |
| **B4** four deliberate-regression arms | ⭐ **PASS** | §6 |
| **B5** 2 s structural zero | ⭐ **PASS** | exactly 0.0, 0/18,615 |
| **B6** kinematics + vacuity gate | ⭐ **PASS** | §8 |
| **B7** supervision ≥ 20 % | ⭐ **PASS** | 44.18 % |
| **B8** schedule declared, NO override | ⭐ **PASS** | §9 |

**Supply, all arms (6 s grid, best-in-fan vs the recorded ego path):**

| arm | N | ADE all | ADE LC | LC gain | ALL gain | TURN gain | LC windows improved |
|---|---|---|---|---|---|---|---|
| `A0` incumbent | 117 | 1.2943 | 1.5315 | — | — | — | — |
| `A1` **SHIPPED** two-segment | 123 | 1.2425 | 1.3670 | **+0.1645** | +0.0519 | +0.0064 | 506 |
| `A1p` equal-cost two-seg pair | 119 | 1.2518 | 1.3676 | +0.1640 | +0.0425 | +0.0026 | 506 |
| ⭐ **`A2` RULE S three-segment** | **119** | **1.2396** | **1.3183** | ⭐ **+0.2133** | **+0.0547** | +0.0031 | **573** |
| `A3` composed (both families) | 125 | 1.2327 | 1.3178 | +0.2138 | **+0.0616** | +0.0065 | 574 |
| *floor* straight family | 13 | 2.3702 | 1.5750 | — | — | — | — |
| *floor* constant velocity | 1 | 3.8387 | 2.5151 | — | — | — | — |
| *floor* zero path (no information) | 0 | 32.6312 | 36.0329 | — | — | — | — |

⭐ **Per candidate the three-segment family is 3.9× more efficient**: 0.1067 m of
lane-change ceiling per candidate against the two-segment family's 0.0274 m.

---

## 4. B0 — the inherited bar, clause by clause

* **B0a — beats the shipped family on lane-change supply.** ⭐ **PASS.**
  `A2` **+0.21327135** vs the **SHIPPED SIX** **+0.16452706**. **Two candidates
  beat six.** *(Equal-cost context, NOT the verdict: the 2-candidate two-segment
  pair reads +0.16395383.)*
* **B0b — costs less curvature on its own repicks.** ⭐ **PASS.**
  `A2` **+0.002344** (n = 573) vs `A1` **+0.003361** (n = 506) — **30.3 % less.**
  ⚠️ **The robustness read the pre-registration promised, on the COMMON repick set
  (n = 505), where the two arms are the same object:** base **0.002577** → `A1`
  **0.005895** (**+0.003318**) vs `A2` **0.004728** (**+0.002151**). ⭐ **The
  common-set read AGREES with the own-repicks read**, so the verdict is not an
  artefact of the two arms repicking different windows.
* **B0c — turn control holds.** ⭐ **PASS.** `|+0.0031| ≤ 0.01`.
* **B0d — degenerate arms read exact zeros.** ⭐ **PASS** (§6).

⇒ **B0 PASS on all four clauses.** (Three-of-four would have been a FAIL.)

---

## 5. B1 + B2 — the OFF path is BIT-IDENTICAL, proved by comparison

**B1 — parity.** `build_twoseg_anchors.py` with **neither** flag writes the
source's tensors untouched:
`anchors 51f930dc6f3564ff8f21c9070ca97b805d4e82908e42d0ef1f99be9f5a3a66df`,
`controls b072f4c052331beb79bef117c7b233f702802b517429cc9157a841294e089664`,
`controls [117, 2]`, `control_schedule="constant"`. ⭐ **Two independent sources,
one hash:** the constants recorded in `restamp_refcv4b_anchors.py` and the live
`ckpt_40284_FINAL.pt`, re-derived here.
⚠️ **The guard demonstrated itself:** my first parity run passed a hash whose
**middle I had reconstructed from the abbreviated `51f930dc…a66df`** and
`--assert-anchors-sha` **REFUSED it** before writing anything. An abbreviation is
not a hash, and the tool is what caught it rather than my care.

**B2 — the limit, by comparison, with a MUTATION CONTROL.** ⭐ **PASS.**

| check | over | result |
|---|---|---|
| **L1** 117×2 vs 117×4 (`t₁ = t₂ = horizon`) | **34,847,280 floats** | **BIT-IDENTICAL** |
| **L2** 123×3 vs 123×4 (`t₂ = horizon`) | **36,634,320 floats** | **BIT-IDENTICAL** |
| **L3 MUTATION** same controls, `(t₁, t₂) = (2, 4)` | — | **max \|diff\| 112.5986 m** |

⛔ **L3 is why L1/L2 mean anything.** An equality that cannot fail proves nothing;
the mutation shows the comparison is capable of reporting a difference.
The nesting is **exact**: `t₂ ≥ horizon` **is** the two-segment candidate and
`t₁ ≥ horizon` **is** the incumbent, so all three families live in one integrator.

---

## 6. B4 — the deliberate-regression arms, and why each zero is STRUCTURAL

⭐ Both apparatus facts were established **before** the expectations were
committed (`raw/p0_bank_facts.py`, reading only the incumbent checkpoint): the
bank holds `(a_lon, a_lat) = (0, 0)` **at index 67**, and its `a_lat` grid
`{0, ±0.75, ±1.5, ±2.25, ±3.0}` is symmetric with **0 of 117** rows missing their
`(a_lon, −a_lat)` mirror.

| arm | schedule | why structural | LC gain | ALL gain | improved |
|---|---|---|---|---|---|
| **R1** | `t₁ = t₂ = 6.0` | `+1` every tick ⇒ existing constant arcs | **+0.0000000000** | **+0.0000000000** | **0 / 18,615** |
| **R2** | `t₁ = 0, t₂ = 6.0` | `−1` every tick ⇒ mirror arcs, grid is mirror-complete | **+0.0000000000** | **+0.0000000000** | **0 / 18,615** |
| **R4** | `t₁ = t₂ = 0` | `0` every tick ⇒ **candidate 67**, the straight line | **+0.0000000000** | **+0.0000000000** | **0 / 18,615** |
| **R3** *(grading)* | `t₁ = 2.0, t₂ = 6.0` | third segment empty ⇒ **IS** the two-segment pair | **0.1639538252** | — | — |

⭐ **R3 GRADES, not merely fires.** Its controls are `torch.equal` to
`as_four_column(A1p)` and its **per-window ADE array is EXACTLY equal** to the
two-segment pair's — 18,615 windows, `np.array_equal`, no tolerance. That is
strictly stronger than the predecessor's `±0.005` grading arm, and it is the
positive proof that the two families are nested rather than merely similar.

---

## 7. FOUR METRIC FAMILIES — never pooled, and in BOTH DIRECTIONS

**Lane-change windows (n = 1,297):**

| arm | ADE | along | cross | speed | **curv** | **heading** |
|---|---|---|---|---|---|---|
| `A0` incumbent | 1.5315 | 0.8285 | 1.1617 | 0.4322 | 0.002784 | 2.7262 |
| `A1` two-segment | 1.3670 | 0.8315 | 0.9533 | 0.4338 | 0.004095 | 3.0818 |
| ⭐ **`A2` three-segment** | **1.3183** | 0.8328 | **0.8773** | 0.4341 | **0.003819** | **2.7935** |
| `A3` composed | 1.3178 | 0.8325 | 0.8777 | 0.4340 | 0.003870 | 2.8103 |
| **floor** straight family | 1.5750 | 0.8294 | 1.2196 | 0.4313 | **0.002726** | 2.7727 |
| **floor** constant velocity | 2.5151 | 1.9152 | 1.2273 | 0.8408 | 0.002726 | 2.7727 |

**All windows (n = 18,615):**

| arm | ADE | along | cross | speed | curv | heading |
|---|---|---|---|---|---|---|
| `A0` | 1.2943 | 0.8922 | 0.7040 | 0.4739 | 0.005055 | **3.13861** |
| `A1` | 1.2425 | 0.8933 | 0.6365 | 0.4746 | 0.005317 | 3.16380 |
| **`A2`** | **1.2396** | 0.8932 | **0.6342** | 0.4744 | 0.005236 | **3.13880** |
| `A3` | 1.2327 | 0.8935 | 0.6232 | 0.4746 | 0.005267 | 3.12829 |
| floor straight | 2.3702 | 1.2755 | 1.7073 | 0.5772 | 0.007186 | 5.90804 |

**Turn control (n = 2,602):** `A0` 1.9476 / 0.018196 → `A2` 1.9445 / 0.018304.

### ⛔ THE TRADE, STATED PLAINLY — what it buys AND what it pays

* **BUYS, on lane-change windows:** ADE **−13.9 %** (1.5315 → 1.3183) and
  cross-track **−24.5 %** (1.1617 → 0.8773). Against the two-segment family it
  buys a further **−0.0487 m** ADE and **−0.0760 m** cross-track.
* **PAYS:** curvature MAE **0.002784 → 0.003819**, **×1.372**. ⛔ It is a real
  cost and it is not hidden. *(The two-segment family pays ×1.471.)*
* ⭐ **AND IT RETURNS THE HEADING THE TWO-SEGMENT FAMILY GAVE AWAY** — the
  net-yaw-zero prediction, confirmed: on lane-change windows heading rises
  **+0.0673 deg** against the two-segment family's **+0.3556 deg** (**18.9 %** of
  the cost), and **on all windows it is 3.13861 → 3.13880, +0.00019 deg —
  unchanged to four decimals**, while the two-segment family pays **+0.02519**.
* **LONGITUDINAL is untouched**, as it must be — `a_lon` is unchanged on every new
  candidate: along **+0.0043**, speed **+0.0019** on lane-change windows.

### ⚠️ THE VACUITY GATE ON CURVATURE — read the incumbent against its floor

The straight-line floor on lane-change windows is **0.002726**, and `A0` sits at
**0.002784**. ⛔ **The incumbent's "good" curvature score is bought by barely
steering at all — +2.13 % over a plan that never steers.** `A2` sits at **+40.1 %**
over the floor because it actually turns; `A1` at **+50.2 %**.
⚠️ These are **6 s-grid** curvature numbers and are **NOT** comparable to the 2 s
re-rolled arm floors `ha0` 0.006841 / `os` 0.008024 — different grid, different
object. Do not cross-quote them.
⚠️ **NO REFINEMENT EFFECT IS ATTRIBUTED HERE.** Nothing in this work touches the
refinement path.

---

## 8. Did the third segment fix the TAIL it was built for? — the mechanism, MEASURED

⛔ **CONTROL AT A KNOWN VALUE, asserted:** on lane-change windows whose pick did
**not** change, the two paths are the same object and every family reads
**EXACTLY** equal — `np.array_equal` on the paths, `True` for both arms
(A1: n = 791, A2: n = 724).

**Per-segment curvature error on each arm's own lane-change repicks:**

| segment | `A1` base | `A1` | Δ`A1` | `A2` base | `A2` | Δ`A2` |
|---|---|---|---|---|---|---|
| 0.5–1.0 s | 0.002809 | 0.005018 | +0.002209 | 0.002640 | 0.005125 | +0.002485 |
| 1.0–1.5 s | 0.003190 | 0.004604 | +0.001414 | 0.002985 | 0.004736 | +0.001751 |
| 1.5–2.0 s | 0.003148 | 0.004768 | +0.001621 | 0.002973 | 0.004855 | +0.001882 |
| 2.0–3.0 s *(the flip)* | 0.002614 | 0.003948 | +0.001334 | 0.002546 | 0.003859 | +0.001313 |
| **3.0–4.0 s** | 0.002164 | 0.008605 | +0.006441 | 0.002208 | 0.008739 | **+0.006531** |
| **4.0–5.0 s** | 0.001925 | 0.007595 | +0.005670 | 0.001934 | 0.004485 | ⭐ **+0.002551** |
| **5.0–6.0 s** | 0.002266 | 0.007101 | +0.004835 | 0.002161 | 0.002056 | ⭐ **−0.000105** |
| **TOTAL** | | | **+0.023524** | | | ⭐ **+0.016408** |
| **3–6 s TAIL** | | | +0.016946 (**72.04 %**) | | | +0.008977 (**54.71 %**) |

⭐⭐ **THE PREDICTION IS CONFIRMED, AND EXACTLY WHERE THE SCHEDULE SAYS IT SHOULD
BE.** `t₂ = 4.0 s` means the candidate is **still counter-steering through 3–4 s**
and **straight after 4 s** — and that is precisely the pattern: the 3–4 s band is
**unchanged** (+0.006441 → +0.006531) while 4–5 s falls **55 %**
(+0.005670 → +0.002551) and 5–6 s **crosses zero to −0.000105**, i.e. in the final
second the three-segment pick tracks curvature **better than the incumbent**.
Total degradation falls **30.3 %**.
⚠️ **The remaining defect is now the 3–4 s band**, which is *inside* the
counter-steer segment — so the next lever is a **shorter middle segment**, not a
fourth one. ⛔ That is a **new schedule** and needs **its own pre-registration**;
choosing it from this table would be the post-hoc selection this whole document
exists to avoid. **Named, not taken.**

**Strata (lane-change):**

| stratum | n | ADE b → e | curv b → e | cross b → e | head b → e |
|---|---|---|---|---|---|
| `A1` REPICKED | 506 | 1.6248 → 1.2031 | 0.002588 → 0.005949 | 1.3929 → 0.8587 | 2.716 → 3.628 |
| `A1` unchanged | 791 | *exactly equal* | *exactly equal* | *exactly equal* | *exactly equal* |
| ⭐ `A2` REPICKED | **573** | 1.5717 → **1.0889** | 0.002493 → 0.004837 | 1.3465 → **0.7028** | 2.659 → **2.812** |
| `A2` unchanged | 724 | *exactly equal* | *exactly equal* | *exactly equal* | *exactly equal* |

⭐ On the windows it changes, `A2` improves ADE **30.7 %** and cross-track
**47.8 %**, and pays only **+0.153 deg** of heading against `A1`'s **+0.912 deg**.

---

## 9. B6 · B7 · B8

**B6 — kinematics with the vacuity gate.** ⭐ **PASS.** New candidates peak
**0.0765 g** against the incumbent bank's **0.5097 g**; **0 of 38**
(candidate, speed) pairs over `μ = 0.7`; **`kappa_cap` is never reached** — where
the **incumbent bank does reach it**, so the incumbent holds candidates that do
not deliver the `a_lat` their column declares and these do not.
⛔ **THE MANOEUVRE RATE, BESIDE IT:** **573 / 1,297** lane-change and
**1,906 / 18,615** overall windows **select a new candidate**. **The safety zero is
not bought by declining the manoeuvre** *(MEASURED precedent: a `turn_left` recall
of exactly 0.0000).*

**B7 — SUPERVISION rate.** ⭐ **PASS** at the **INHERITED** 20 % threshold.

| arm | lane-change | turn | all |
|---|---|---|---|
| ⭐ `A2` RULE S | **573 / 1,297 = 44.18 %** | 1.04 % | 10.24 % |
| `A1` shipped two-segment | 506 / 1,297 = 39.01 % | 2.19 % | 11.35 % |
| `A3` composed | 574 / 1,297 = 44.26 % | 2.27 % | 11.98 % |

Per candidate: `a_lat = −0.75` takes 282 lane-change / 1,069 all;
`a_lat = +0.75` takes 291 / 837. ⭐ **A gradient, not a takeover** — and `A2`
captures **more** lane-change supervision on **fewer** all-window picks than the
two-segment family, which is the shape a targeted vocabulary fix should have.
⛔ **B7 MEASURES SUPERVISION, NOT EXECUTION** (§11).

**B8 — the artifact declares its schedule, and there is NO override.** ⭐ **PASS.**
`control_schedule="three_segment_alat_pulse"`, `controls_columns =
["a_lon_ms2", "a_lat_ms2", "t1_s", "t2_s"]`, and the file declares its three
populations so no reader re-derives them (`n_constant` 117, `n_two_segment` 0,
`n_three_segment` 2; composed: 117 / 6 / 2). A `[N, 4]` file that declares nothing
raises **`AnchorScheduleMissing`** naming the columns it cannot identify; a
declared schedule disagreeing with the column count raises
**`AnchorScheduleConflict`**; a row with `t₂ < t₁` is refused at **build** time and
in the **integrator**, because silently it renders as a two-segment candidate
wearing a three-segment row.
⛔ **NO CLI OVERRIDE EXISTS AND A TEST ASSERTS ITS ABSENCE** — by signature
inspection *and* behaviourally (the units override, the only override there is,
does not suppress the schedule refusal). **No legacy 3- or 4-column file exists
anywhere**, so a permitted guess could not rescue a real artifact — it could only
**invent** one. *(That is the whole difference from `control_units`, where the
live `anchors.pt` is a genuine legacy file that must keep loading.)*

---

## 10. CONTEXT — the exploratory sweep, re-derived. ⛔ It may not move the default.

Each `(t₁, t₂)` scored as its own **+2-candidate** family. **RULE S fixed the
schedule before any of this existed.**

| family | LC gain | ALL gain | TURN gain | LC picks | Δcurv | Δcross | Δhead | 2 s repicks |
|---|---|---|---|---|---|---|---|---|
| 2-seg t = 2.0 *(equal-cost ref)* | 0.1640 | 0.0425 | 0.0026 | 506 | +0.003359 | −0.5320 | +0.913 | 0 |
| 3-seg 2.0 → 3.0 | 0.1867 | **0.0592** | 0.0048 | 433 | +0.000864 | −0.7088 | **−0.862** | 0 |
| 3-seg 2.0 → 3.5 | 0.2161 | 0.0588 | 0.0037 | 510 | +0.001639 | **−0.7217** | −0.418 | 0 |
| ⭐ **3-seg 2.0 → 4.0 — RULE S** | **0.2133** | 0.0547 | 0.0031 | **573** | +0.002344 | −0.6438 | +0.152 | **0** |
| 3-seg 2.0 → 4.5 | 0.1937 | 0.0494 | 0.0028 | 567 | +0.002727 | −0.5768 | +0.539 | 0 |
| 3-seg 1.5 → 3.0 | **0.2266** | 0.0523 | 0.0018 | 767 | +0.002163 | −0.5093 | +0.248 | ⛔ **720** |
| 3-seg 1.5 → 3.5 | 0.1382 | 0.0334 | 0.0014 | 668 | +0.002495 | −0.3402 | +0.789 | ⛔ **720** |
| 3-seg 2.5 → 4.0 | 0.0957 | 0.0460 | 0.0051 | 317 | **+0.000519** | −0.4754 | −0.690 | 0 |
| 3-seg 2.5 → 5.0 | 0.1234 | 0.0465 | 0.0043 | 362 | +0.001237 | −0.5726 | −0.281 | 0 |
| 3-seg 3.0 → 5.0 | 0.0265 | 0.0307 | 0.0051 | 143 | +0.000661 | −0.2950 | −0.727 | 0 |

⛔ **All NINE three-segment rows beat the two-segment reference on curvature
(every Δcurv < +0.003359), and FIVE of nine beat it on lane-change gain — so the
FAMILY, not the particular schedule, is what the evidence supports.** RULE S's row is the argmax of none of
them (§1), and the schedule was fixed before any of these numbers existed.

---

## 11. SCOPE — what this can and cannot touch

⛔⛔ **THE MODEL STRUCTURALLY CANNOT SELECT THESE CANDIDATES.** refcv4b has **117
logits**; the extension makes the bank **119** (or 125 composed). **What changes
is the SUPERVISION, not the behaviour**: the geometric `a_star` now points at a
three-segment lane-change shape on **44.18 %** of lane-change windows. A trained
execution rate needs a retrain and is **blocked** (§12.1). ⛔ **Nothing here is a
behavioural result and none of it may be read as one.**

⛔ **THE DOMINANT SELECTION DEFECT IS LONGITUDINAL, AND THIS IS NOT A FIX FOR IT.**
*(INHERITED, commit `a3b232b`, not re-verified here:* **4,350 / 24,114** *windows
carry the wrong longitudinal manoeuvre while the fan held a correct candidate; an*
`a_lon` *oracle recovers* **74.2 %** *of the regret against* `a_lat`'s
**20.9 %**.*)*

**MEASURED here, the supply-side bound:** the extension moves the all-window 6 s
supply ceiling by **0.0547 of 1.2943 m = 4.23 %**, and the lane-change ceiling by
**0.2133 of 1.5315 m = 13.93 %**, on the **6.97 %** of windows (1,297 / 18,615)
that execute a lane-change shape.

⇒ ⛔ **A lateral vocabulary fix reaches at most the ~20.9 % lateral share of the
regret, and within it only the lane-change sub-population.** It is a real
capability the vocabulary did not have — **not** a fix for the selection defect,
and it is not presented as one.

⛔ **AND IT CANNOT APPEAR IN `ade_0_2s` AT ALL** (B5): every RULE S candidate flips
at 2.0 s and slot 3 **is** 2.0 s, so the 2 s grid moves by **exactly
0.0000000000** with **0 / 18,615** windows repicking, and every one of the four
families changes by exactly zero. **That is the scope limit, reported as a
limitation.**

⛔ **No label generation** (PI-forbidden), **no Alpamayo re-ask**. The nav command
is an **INPUT simulating the vehicle's nav system**, never a training signal.

---

## 12. BLOCKED, each with what would unblock it

1. ⛔ **Executed lane-change rate of a trained model.** refcv4b cannot rank a
   118th candidate. **Unblocked by** a retrain — a tiny-rig v7 arm (~17 min on
   Thor) or an A40 slot. ⛔ **The A40 is FREE and RESERVED for the composed refcv5
   arm and was NOT taken.**
2. ⛔ **The tactical head's three dead lane-change classes.** A **different head**;
   its repair **is** a label change. **Blocked on a PI decision**, not compute.
3. ⛔ **The consumer.** `refc.py::AnchoredDiffusionDecoder.roll_bank` rolls
   `anchor_controls` as a constant and `anchor_control_seq` expands to
   `[B, N, S, 2]`; a `[N, 4]` bank is **refused** by those shape checks — loudly,
   which is correct. **`refc.py` is a sibling's file this turn**, so
   `anchor_twoseg.roll_bank` is delivered as the drop-in reference, pinned
   `torch.equal` against the live decoder on the 2-column path, and the
   decoder-side read of columns 2–3 is a **named hand-off with an exact diff**
   (§13), not an omission.
4. ⚠️ **The 3–4 s band** is now the largest remaining curvature defect and sits
   *inside* the counter-steer segment ⇒ a **shorter middle segment**. **Named, not
   taken** — it needs its own pre-registration.

---

## 13. HAND-OFF — the exact diff a sibling must apply to `refc.py`

⛔ **Not applied here: `refc.py` is a sibling's file this turn.** The decoder
currently refuses a wide bank on its shape checks. The change is to route the
bank through the reference integrator, which is **bit-identical on the 2-column
path** (pinned by `test_two_column_path_matches_the_live_decoder_roll_bank`):

```
  in AnchoredDiffusionDecoder.roll_bank, replace the constant expansion
      seq = ctrl[:, :, None, :].expand(B, N, S, 2)
  with
      from tanitad.refs.anchor_twoseg import roll_bank as _rb
      return _rb(self.anchor_controls, v0, control_units=self.control_units,
                 steps=self.steps, slots=self.anchor_slots, dt=self.dt,
                 alat_v_floor=self.alat_v_floor, kappa_cap=self.kappa_cap)
  and widen the `anchor_controls` shape assertion from [N, 2] to [N, 2|3|4].
```

⭐ **Escalated as an exact diff, not as a "please merge" line in a README** — an
orthogonality instrument once sat unmerged for **10 days** because the request
lived in a document nobody re-read.

---

## 14. Deliverable manifest

| artifact | where it lives |
|---|---|
| `stack/tanitad/refs/anchor_twoseg.py` | repo (EDITED, additive) — RULE S, the pulse family, `lateral_sign`, the 4-column integrator |
| `stack/tanitad/refs/anchor_meta.py` | repo (EDITED, additive) — `THREE_SEGMENT_SCHEDULE`, 4-column declaration + population counts, `t₂ < t₁` refusal |
| `stack/scripts/build_twoseg_anchors.py` | repo (EDITED, additive) — `--three-segment`, `--t1-grid`, and deliberately no `--t2` |
| `stack/tests/test_anchor_twoseg.py` | repo (EDITED) — **41 tests**, 19 inherited + **22 new**, all green |
| `PREREG_THREE_SEGMENT_ANCHORS.md` | this directory (committed `07290a0`, before any number) |
| `RESULT.md` | this directory |
| `raw/p0_bank_facts.py`, `raw/p1_triseg.py`, `raw/p2_triseg_decomp.py`, `raw/_env.py` | this directory |
| `raw/out_p1_triseg.json`, `raw/out_p2_triseg_decomp.json`, `raw/*.log` | this directory |
| `anchors_117_parity.pt`, `anchors_119_triseg.pt`, `anchors_125_composed.pt` | `C:/Users/Admin/tanitad-triseg-20260906/` (**local disk only** — build outputs, reproducible from the builder + the checkpoint in one command) |

**Built bank hashes:** `119` anchors `47994684…eab27` / controls `7c188df8…f070a`;
`125` anchors `b25a0b1b…9d104` / controls `6341a74f…d29b60`.

**Suite:** `tests/test_anchor_twoseg.py` **41 passed**; with the anchor regression
set (`test_anchor_meta`, `test_anchor_flyability`, `test_withheld_bank`,
`test_anchor_tactical`, `test_anchor_prefilter`, `test_v6_anchor_loss`) the copied
`stack/tests` reads **141 passed, 17 skipped** (was 118/17 before this work).

---

## 15. THE ONE LINE

⭐ **Yes — on its own pre-registered terms the three-segment schedule beats the
shipped two-segment pair, and beats the shipped SIX with only TWO candidates:
+0.2133 m of lane-change supply against +0.1645 m, at 30 % less curvature on its
own repicks, with the heading it gives back almost exactly whole and the 2 s grid
provably untouched. ⛔ And it can possibly touch only the ~20.9 % LATERAL share of
the selection regret, within that only the 6.97 % lane-change sub-population, and
only as SUPERVISION — the 117-logit model cannot select these candidates at all
until something retrains.**
