# `os_navpred` / `os_navflip` — refcv4b's OWN predicted route, and what the nav edge actually carries

**Evidence class: MEASURED (ours).** **Tier: T1** for every arm (`tier_ruling: UNRULED`).
**n = 4,823 windows / 141 held-out episodes** — the grid refcv3 @40,284 and refcv4b's landing read
were scored on, reproduced exactly. **Estimator: paired episode-cluster bootstrap**
(`taniteval/ci.py`), `n_boot 2000`, `seed 0`. ⛔ `overlapping_holdout_se` appears nowhere.
**Pre-registration: `SPEC.md`, written and banked before the roll** (hypothesis `H-NAVPRED-1`).

| | |
|---|---|
| ckpt | `ckpt_40284_FINAL.pt`, md5 **`99b573e8277d94a5e3bfbf630cb4d751`** — byte-identical to the landing read's, verified at both ends |
| compute | **Thor** (`tanitad-thor-wifi`): roll A (7 arms, 4 rows/window) **1,352 s**; roll B (8 arms, 5 rows/window) **1,585 s**. ⛔ The A40 was never touched — refcv5's training queue is intact. |
| roller | `taniteval/tools/refcv3_arm.py` md5 `3f4280abe4852280bc8f1b3b9f678a77` (`--with-navpred`, `--with-navflip`) |
| analyser | the same file + the strategic corrections, md5 `ea03bfa53c5a887808ce6fad402a5933`, via `--analyze-only` (zero GPU) |
| artifacts | `raw/refcv4b_navflip.json` (md5 `24e0834b53a1eac35f724dfa4b2dc3a0`), `raw/flip_analysis.json`, `raw/refcv4b_navpred_CORRECTED.json`, `raw/paired_navpred.json`, `raw/E13_PRESENCE_VS_CONTENT_MECHANISM.json`, `raw/navpred_treated_subsets.json`, `raw/PREDICTION_ZERO_GPU.json`, `raw/ECHO_INDEX_CORRECTION.json`, `raw/navpred_dump.tgz`, `raw/navflip_dump.tgz` |

---

## 0. ⭐⭐⭐ THE ONE-LINE ANSWER

**The model's own predicted route recovers 0.0914 m of the 0.0961 m the oracle nav was worth — 95.1 %
— and the deployment arm now TIES the echo control instead of losing to it by +0.1054 separated.
⛔ But the recovery is NOT route information: a nav token that is DELIBERATELY WRONG on every
commanded window (`os_navflip`) is statistically indistinguishable from the true one
(+0.0022 m [−0.0006, +0.0052], not separated). What `os_navzero` removes is the E13 CONDITIONING
PATHWAY — 97.7 % presence, 2.3 % content.**

⇒ ⭐ **The landing read's "0.1054 m deployment gap" is right as a measurement and wrong as an
interpretation.** It was read as *the size of the oracle dependency*. **MEASURED: the true cost of
replacing the oracle route with the model's own prediction is +0.0047 m [+0.0012, +0.0083] — 22×
smaller — and the cost of feeding a route that is actively WRONG is not separable from zero.**

---

## 1. THE HEADLINE TABLE — eight arms, ONE process, ONE surface

| arm | tier | ADE (m) | CI95 | what it is |
|---|---|---|---|---|
| `ha0_ext` | T1 | **0.2874** | [0.2646, 0.3137] | ⭐ **THE ECHO CONTROL** — constant `a0` and `κ0` at t0 |
| `os` | T1 | **0.2965** | [0.2682, 0.3272] | the deployed planner on the **ORACLE** v7.2 nav token |
| ⭐ **`os_navflip`** | **T1** | **0.2987** | [0.2701, 0.3289] | ⛔ **the token WRONG (left↔right) on all 1,743 commanded windows** |
| `ha` | T1 | 0.2996 | [0.2749, 0.3280] | hold-action control |
| `os_navshuf` | T1 | 0.3006 | [0.2722, 0.3301] | nav **pairing** broken, marginal preserved |
| ⭐ **`os_navpred`** | **T1** | **0.3012** | [0.2721, 0.3314] | ⭐ **the planner on its OWN predicted route** |
| `os_navzero` | T1 | 0.3926 | [0.3652, 0.4212] | nav **withheld** (`nav_cmd=None`) |
| `ha0` | T1 | 0.6723 | [0.6017, 0.7437] | constant velocity — the straight-line floor |

**Paired margins** (n = 4,823 / 141 unless stated):

| margin | delta (m) | CI95 | separated | windows differing |
|---|---|---|---|---|
| **`os_navpred` − `os_navzero`** | **−0.0914** | [−0.1057, −0.0774] | **YES** | 4,823 |
| **`os_navpred` − `ha0_ext`** | **+0.0138** | [−0.0011, +0.0303] | **no — a TIE** | 4,823 |
| `os_navpred` − `ha` | +0.0016 | [−0.0135, +0.0181] | no — a TIE | 4,823 |
| `os_navpred` − `os` | **+0.0047** | [+0.0012, +0.0083] | **YES** | 1,908 |
| ⛔ `os_navpred` − `os_navshuf` | **+0.0006** | [−0.0033, +0.0042] | **no** | 2,373 |
| ⭐⭐ **`os_navflip` − `os`** | **+0.0022** | **[−0.0006, +0.0052]** | ⛔ **no** | 1,743 |
| **`os_navflip` − `ha0_ext`** | **+0.0113** | [−0.0033, +0.0274] | **no — a TIE** | 4,823 |
| `os_navflip` − `os_navzero` | −0.0939 | [−0.1081, −0.0802] | YES | 4,823 |
| `os_navflip` − `os_navpred` | −0.0025 | [−0.0065, +0.0016] | no | — |
| `os_navflip` − `os_navshuf` | −0.0019 | [−0.0051, +0.0014] | no | — |
| *`os` − `os_navshuf`* | *−0.0041* | *[−0.0076, −0.0004]* | *YES* | *2,406* |
| *`os` − `os_navzero`* | *−0.0961* | *[−0.1103, −0.0827]* | *YES* | *4,823* |
| *`os` − `ha0_ext`* | *+0.0091* | *[−0.0055, +0.0254]* | *no* | *4,823* |
| *`os` − `ha`* | *−0.0031* | *[−0.0179, +0.0133]* | *no* | *4,823* |
| *`os_navzero` − `ha0_ext`* | *+0.1052* | *[+0.0876, +0.1246]* | *YES (worse)* | *4,823* |

**Recovery fraction** `R = (ADE(os_navzero) − ADE(os_navpred)) / (ADE(os_navzero) − ADE(os))`
= 0.0914 / 0.0961 = **0.951**.
**Presence/content split** `= (ADE(os_navflip) − ADE(os)) / (ADE(os_navzero) − ADE(os))`
= 0.0022 / 0.0961 = **content 2.3 %, presence 97.7 %**.

⭐ **On the 1,743 windows the flip actually touches** (selection on the INTERVENTION, never on the
outcome — the treated subpopulation, stated so a reader can check):

| margin, on the 1,743 COMMANDED windows | delta | CI95 | separated |
|---|---|---|---|
| `os_navflip` − `os` | +0.0062 | [−0.0024, +0.0142] | **no** |
| `os_navpred` − `os` | +0.0028 | [−0.0024, +0.0082] | **no** |
| `os_navshuf` − `os` | +0.0031 | [−0.0055, +0.0114] | **no** |
| **`os_navzero` − `os`** | **+0.0648** | [+0.0463, +0.0857] | **YES** |
| **`os_navflip` − `os_navzero`** | **−0.0586** | [−0.0791, −0.0410] | **YES** |

⇒ **Even where the true command is a TURN — the windows where route content has the most leverage —
a wrong token, a random token and a predicted token all cost nothing separable, while removing the
token costs 0.065 m separated.**

---

## 2. ⛔⛔ THE VERDICT AGAINST THE PRE-REGISTERED CRITERIA

`SPEC.md` §3 committed four rows before the roll. Reported exactly as written:

| pre-registered row | criterion | MEASURED | verdict |
|---|---|---|---|
| ⭐ SUCCESS | `os_navpred − os_navzero` separated NEGATIVE **AND** `os_navpred − ha0_ext` NOT separated worse | −0.0914 [−0.1057, −0.0774] **separated** · +0.0138 [−0.0011, +0.0303] **not separated** | ✅ **BOTH MET** |
| ⛔ FAILURE | `os_navpred − os_navzero` not separated or positive | separated and negative | not triggered |
| ⛔ **ADDITIONAL, gating** | `os_navpred − os_navshuf` **must be separated NEGATIVE** for any claim that the **prediction** rather than the **marginal** is doing the work | **+0.0006 [−0.0033, +0.0042], NOT separated** | ⛔ **FAILS** |

⇒ **`H-NAVPRED-1` is SUPPORTED on its primary endpoint and its CAUSAL claim is REFUSED by its own
gating control** — reported verbatim per the SPEC as *"indistinguishable from a
distribution-matched random token"*.

⭐⭐ **THAT CONTROL IS THE RESULT, AND `os_navflip` — RUN IN THE SAME SESSION BECAUSE THE CONTROL
FAILED — TURNED IT FROM A CAVEAT INTO A FINDING.** Without them this turn would have shipped
*"the route head closes the deployment gap"*: true in ADE, and the wrong causal story.

⚠️ **`os_navflip` was NOT pre-registered as a hypothesis test** — it was named in the SPEC only as
an intervention that exists in the harness and was not rolled. It is reported here as a
**post-hoc, pre-committed-direction diagnostic**: the SPEC §2 text stated in advance what a null
flip would mean. It carries no pass/fail bar of its own and any claim built on it needs its own
pre-registration.

---

## 3. ⭐⭐ THE MECHANISM, read off the model's OWN internals rather than from ADE

`raw/E13_PRESENCE_VS_CONTENT_MECHANISM.json`, zero GPU, from the banked dump:

| what changed | mean \|Δ `g_str`\| vs `nav_true` | max | selected anchor changed |
|---|---|---|---|
| the token's VALUE — **shuffled** | **0.0103** | 0.0548 | 586 / 4,823 (12.2 %) |
| the token's VALUE — **predicted** | **0.0080** | 0.0548 | 534 / 4,823 (11.1 %) |
| **the token REMOVED** (`nav_cmd=None`) | **0.1902** | 0.5433 | **778 / 4,823 (16.1 %)** |

⇒ **Removing the token moves the strategic goal 18.6× further than changing its value does.**
On the 1,743 commanded windows the selection changes 17.1 % / 13.4 % under a value change vs
**24.1 %** under removal.
**Control that must read known values:** `nav_injected` = **1.0** under any token and **0.0** under
`nav_cmd=None` — the E13 edge is live and the null is real (`refc_v3.py:437-441`).

⇒ ⛔⛔ **E13's nav path behaves as a PRESENCE-GATED BIAS on the strategic goal, with only weak
dependence on WHICH command it carries.** That is the mechanism behind
`os_navshuf ≈ os_navpred ≈ os_navflip ≈ os` while `os_navzero` is separated worse, and it makes the
refcv5 requirement concrete: **the conditioning edge must be made VALUE-SENSITIVE (or replaced by a
geometric goal point), not merely present.**

---

## 4. ⭐⭐ AND THE NAV CONTENT *DOES* REACH SOMETHING — ADE JUST CANNOT SEE IT

⛔ **An ADE-only report of this experiment would have concluded "the nav content is inert". The
LATERAL and TACTICAL families say otherwise.** This is the four-families rule earning its keep.

| arm | **CURVATURE MAE (1/m)** | heading MAE (deg) | LAT κ | `turn_right` recall (n 396) |
|---|---|---|---|---|
| `os` (true token) | **0.008150** | 1.2964 | **0.8277** | **0.889** |
| `os_navpred` | 0.008168 | 1.3176 | 0.8283 | 0.884 |
| `os_navshuf` | 0.008685 | 1.3165 | 0.8217 | 0.866 |
| ⛔ **`os_navflip` (WRONG token)** | **0.009270** | **1.3773** | **0.8120** | **0.828** |
| `os_navzero` | 0.009207 | 1.3527 | 0.8266 | 0.866 |

⇒ **The flip degrades curvature by 13.7 %, heading by 6.2 %, LAT κ by 1.9 % and `turn_right`
recall by 6.1 points — while its ADE does not move.** ⇒ **the nav content reaches the PATH SHAPE
and the LATERAL DECISION; it does not reach the 2 s positional error**, which is dominated by
longitudinal accuracy where a route command has little leverage.
⚠️ **These are per-arm values with NO paired margins** (`_intervals_complete: false`, work item
NP-2). The pattern is consistent across four independent rows and is reported as an **observation,
not a separated difference**. ⛔ Do not quote any of these as separated until NP-2 lands.

---

## 5. THE PRE-REGISTERED CONTROLS — all seven, PASS *and* FAIL, before any family was read

| # | control | MEASURED | verdict |
|---|---|---|---|
| **C1** | model-free `ha`/`ha0` must reproduce the pod's banked values | `ha` **0.2996** (abs diff **0.000000**) · `ha0` **0.6723** (abs diff **0.000000**), in **both** rolls | ✅ **PASS** — the two surfaces are one surface |
| **C2** | grid | **4,823 / 141**, `window_stride 5`, grid `2s`; `nav_shuffle changed 2,406/4,823`; `nav_flip changed 1,743/4,823` | ✅ **PASS** |
| **C3** | `os` reproduction within 0.001 m | 0.2965 vs 0.2975, **abs diff 0.001031** | ⛔ **MISSES by 3 % of its own tolerance** — §6 |
| **C4** | route head must not move when fed its own prediction | `route_pred(nav_predicted) == route_pred(nav_zero)` on **4,823/4,823**; STRATEGIC identical under all four conditionings (acc 0.7791, κ 0.4864) | ✅ **PASS — not circular** |
| **C5** | same token ⇒ same path | max abs path diff **exactly 0.0 m** on the 2,915 same-token windows; **same-breath control** on the 1,908 different-token windows reads **3.0065 m** | ✅ **PASS** |
| **C6** | arm not degenerate: 1,917 windows must differ | **1,908** differ; predicted-nav distribution **[3,467 / 609 / 747]** vs pre-registered [3,465 / 614 / 744] | ⛔ **MISSES by 9 windows (0.19 %)** — §6 |
| **C7** | the route→nav map is IMPORTED, not re-derived | asserted equal to `refb_labels._ROUTE_TO_NAV` at `run_dump` entry; the run refuses otherwise | ✅ **PASS** |

⭐ **The analysis instrument was itself validated against known values before touching new data:**
`raw/paired_navpred.py` on the **banked landing dump** reproduces `os` 0.2975, `ha` 0.2996,
`ha0` 0.6723, `ha0_ext` 0.2874, `os_navshuf` 0.3013, `os_navzero` 0.3928 and the margins
−0.0021 / +0.0101 / −0.0953 / **+0.1054** exactly
(`raw/paired_navpred_SELFTEST_on_banked_landing_dump.json`).

---

## 6. ⚠️ WHY C3 AND C6 MISS — a measured mechanism, not an excuse

**This is a CROSS-HARDWARE reproduction and the SPEC's tolerances were written for a same-hardware
re-roll.**

| | landing read | this run |
|---|---|---|
| GPU | **NVIDIA A40** | **NVIDIA Thor** |
| arch | x86_64 | **aarch64** |
| torch | **2.8.0+cu128** | **2.13.0+cu130** |

Different kernels ⇒ different float32 rounding ⇒ a different tie-break in **two ARGMAXes** (the
anchor selection over 117 anchors, and the route head). Both are discrete, so a 1e-7 perturbation
gives a **decimetre-scale** change on a near-tied window. The selection profile shows it:
`n_distinct` **50 → 51**, modal share **0.4879 → 0.4889**.

⭐ **The decisive fact is C1: `ha` and `ha0` reproduce to abs diff 0.000000** — they consume no
model output, so the windows, the grid and the ground truth are provably identical; **only the
model's float path moved.** Magnitudes: `os` moves **0.0010 m (0.35 %)**, **89× smaller** than the
0.0914 m effect; the route argmax moves on **9 of 4,823 windows (0.19 %)**.

⛔ **Reported as MISSES, not waived.** What they bound is the **cross-run** comparison to the banked
landing numbers. **Every margin in §1–§4 is WITHIN-RUN**, so none is exposed. **Work item NP-3.**

---

## 7. THE REMAINING FOUR-FAMILY ROWS (full tables in `raw/refcv4b_navflip.json`)

### LONGITUDINAL

| arm | speed MAE (m/s) | bias | tgt-speed acc @0.5 | along-track MAE (m) | accel MAE |
|---|---|---|---|---|---|
| `os` | 0.2900 | +0.0331 | 0.8329 | 0.2544 | 0.4346 |
| `os_navflip` | **0.2897** | +0.0233 | **0.8340** | 0.2552 | 0.4366 |
| `os_navpred` | 0.2956 | +0.0268 | 0.8262 | 0.2590 | 0.4412 |
| `os_navshuf` | 0.2952 | +0.0236 | 0.8300 | 0.2581 | 0.4472 |
| `os_navzero` | 0.3634 | −0.1309 | 0.7985 | 0.3564 | 0.4682 |
| `ha` / `ha0_ext` | 0.2540 | −0.0632 | 0.8662 | 0.2348 / 0.2341 | 0.3166 |
| `ha0` | 0.4880 | −0.0410 | 0.7047 | 0.4705 | 0.4786 |

⇒ ⛔ **the nav token does not touch the longitudinal family at all** (`os_navflip` is if anything
marginally the best speed MAE), and **every planner arm still loses to hold-action** (0.2540).
**R1 — encode CLOSING RATE — is untouched by any routing lever and still ranks first.**

**Distance-keeping** (`os_navpred`, status OK, n = 1,224 / 67 eps): min headway **28.44 m**
[24.43, 32.73]; min time-gap **4.11 s** [3.29, 5.09] (n = 1,150); min TTC **24.89 s**
[23.42, 26.21]. ⚠️ **754 of 1,224 censored at TTC_CAP 30 s, n_closing = 470** — the TTC mean is
over censored data and must never be quoted alone. (`os_navflip`: 28.36 / 4.04 / 24.88, n_closing
468 — indistinguishable.)

### LATERAL — the shape defect the landing read found is UNCHANGED

`os_navpred` curvature **0.008168 > `ha0`'s 0.006802** (the straight-line floor) and **2.2× the
echo control's 0.003712**, while cross-track **0.0985 m** is second-best of every arm (only `os`
0.0978 is lower). ⇒ **the car is in the right place on an over-active path.** No nav source
touches this: it is cost geometry (landing read R4), not routing.

### TACTICAL

| arm | LAT acc / κ | turnL / turnR | LON acc / κ | brake_stop / accelerate | goal FDE (m) | goal-bearing MAE |
|---|---|---|---|---|---|---|
| `os` | 0.9579 / 0.8277 | 0.813 / 0.889 | 0.8260 / 0.5186 | 0.539 / 0.506 | 0.6345 | 1.5484° |
| `os_navpred` | 0.9579 / **0.8283** | **0.829** / 0.884 | 0.8252 / **0.5269** | **0.563** / **0.528** | 0.6453 | **1.5327°** |
| `os_navflip` | 0.9550 / 0.8120 | 0.817 / **0.828** | 0.8271 / 0.5206 | 0.536 / 0.507 | 0.6381 | 1.7888° |
| `os_navshuf` | 0.9571 / 0.8217 | 0.797 / 0.866 | 0.8283 / 0.5195 | 0.528 / 0.498 | 0.6437 | 1.6157° |
| `os_navzero` | 0.9581 / 0.8266 | 0.817 / 0.866 | 0.8184 / 0.5060 | 0.491 / 0.563 | 0.7386 | 1.5988° |
| `ha` / `ha0_ext` | 0.9382 / 0.7374 · 0.9419 / 0.7548 | 0.729 / 0.750 · 0.753 / 0.770 | 0.8443 / **0.6071** | **0.697 / 0.649** | 0.6588 / 0.6323 | 1.6673° / 1.5529° |
| `ha0` | 0.8659 / **0.0000** | 0.000 / 0.000 | 0.7576 / **0.0000** | 0.000 / 0.000 | 1.4029 | 2.8832° |

⛔ Longitudinally every planner arm still trails hold-action (κ 0.6071).
⚠️ v7.2 factored **kin3** labels, not the `|dyaw| > 0.15` gate the human fails 3/9 (`M74`/`M75`) —
quotable only against this label set. ⛔ `anchor_acc` / `oracle_sel` are **not quoted**:
`D-REFCV4B-ASTAR-GEOMETRY` is unrepaired and `--with-oracle-sel` was deliberately OFF.

### STRATEGIC

Route acc **0.7791**, κ **0.4864**, n = 3,622 / 128; per-class recall `route_left` 0.355 (n 470) ·
`route_straight` 0.966 (n 2,442) · `route_right` 0.415 (n 710); majority rate 0.6742 — ⭐⭐ **IDENTICAL
UNDER ALL FIVE CONDITIONINGS: `nav_true`, `nav_shuffled`, `nav_zero`, `nav_predicted` AND
`nav_flipped`** (a deliberately INVERTED command moves the route head by exactly nothing — the
strongest available form of the non-echo result), `paired_true_minus_shuffled` =
0.0000, CI [0, 0]. Corrected echo index **0.6422** (legacy raw-index form 0.1623 — §8).
Anti-echo, corrected: on the **1,736** changed windows LABEL **0.7454** [0.6823, 0.8081] vs
SHUFFLED NAV **0.3335** [0.2651, 0.3951]; on the **n = 1,311** exclusive subset **LABEL 0.7155**
[0.6395, 0.7896] vs **NAV 0.1701** [0.1133, 0.2292] — non-overlapping.

---

## 8. ⛔⛔ AN INSTRUMENT DEFECT FOUND AND FIXED IN PASSING — two published numbers were TYPE ERRORS

`refcv3_arm.py` compared `route_pred` (**3**-wide `ROUTE_CLASSES`) **directly** against
`nav_cmd` / `nav_cmd_shuf` (**4**-wide `NAV_COMMANDS`), on a shipped comment asserting they are
*"both 3-wide and index-aligned in refb"*. **Refuted at source:** `refb.py:64`
`("follow","left","right","straight")`, `refb.py:68`
`("route_left","route_straight","route_right")`, `refb_labels.py:483-484`
`_ROUTE_TO_NAV = {0:1, 1:0, 2:2}` — **not the identity**.

| statistic | published (landing read) | **corrected** |
|---|---|---|
| `nav_echo_index` | 0.1621 | **0.6405** (n 3,622); re-rolled **0.6422** |
| `route_follows_SHUFFLED_NAV_under_shuffle` | 0.2264 | **0.3370** (n 1,736); re-rolled **0.3335** |

⭐ **ROOT-CAUSE CLASS: a control re-implemented beside the harness it controls, drifted — into a
REGRESSION against a CORRECT SIBLING.** `refav1_arm.py:2308` has always derived the map and
compared in ROUTE space ⇒ **every refav1 strategic number is unaffected.**
⚠️ **A third defect, measured false:** the block asserted the two rates are *"different classes by
construction, so mutually exclusive"* — `route_label` is **per-window**, `nav_cmd` a **per-clip**
25 s token, and on **357/1,736** changed windows they coincide (rates sum to 1.0806).
⭐⭐ **`H-REFCV4B-ROUTE-ECHO` STANDS AND IS SHARPER** — it never rested on the index arithmetic but
on the head's output being identical under every conditioning.

**Fixed, staged, MUTATION-PROVEN:** both maps imported from `refb_labels`, both legacy forms kept
under `*_RAW_INDEX_LEGACY`, `changed_exclusive_subset` emitted, the refuted comment deleted, and
`stack/tests/test_refcv3_route_nav_alignment.py` added with a deliberate-regression arm (a route
head that is a *perfect* echo: the corrected index must read exactly 1.0, the legacy form must fail
to see it). **5/5 pass on the fixed file; 4/5 FAIL on the pre-fix file (md5
`f60d1f974038d3785c74706b03046fde`, repo HEAD).** End-to-end proof: the same dump now emits
`nav_echo_index` **0.6422** beside `nav_echo_index_RAW_INDEX_LEGACY` **0.1623**.
⭐ **NP-5 audit CLOSED:** `four_families.py` compares route↔route, `refav1_arm.py` maps before
comparing, `nav_compliance.py` does no label-index comparison — **these two sites were the only
ones, and both are fixed.**
Logged in `RETRACTION_LOG.md`; corrections applied with provenance to `MODEL_REGISTRY.md`,
`GOALS_AND_CLAIMS.md` and the landing read's own `LANDING_RESULT.md`.

---

## 9. ⭐ WHAT THIS MAKES THE NEXT LEVER

1. ⭐⭐ **MAKE THE E13 NAV EDGE VALUE-SENSITIVE, or replace it with a GEOMETRIC GOAL POINT — a
   refcv5 ARCHITECTURE item, not a tuning knob.** MEASURED here: the best possible 3-way
   categorical token is worth **0.0047 m**, and a deliberately wrong one costs **+0.0022 m
   [−0.0006, +0.0052], not separated**. The literature lever the PI cited is the **goal point
   (+4.7 PDMS)** over the categorical command (**+0.2**); this run is the categorical **ceiling**
   on our own surface and it is ~0. ⇒ `D-REFCV5-LEVERS` item 2 is **DOWNGRADED as an eval-time
   lever and PROMOTED as a design requirement**.
2. ⛔ **`H-NAVC-1` / `H-NAVC-3` (plan-compliance follows the nav command) are now strongly
   predicted to FAIL** — the emitted plan is nearly indifferent to which command it is given.
   The `nav_compliance` sidecar is populated in both records here and can be read with **zero
   GPU**; that read is the cheapest next measurement in the programme.
3. **R1 — encode CLOSING RATE — is untouched and still ranks first.** No routing change reaches
   it: `os_navflip` has the *best* speed MAE of any planner arm and every planner arm still trails
   hold-action.
4. **R4 — curvature shaping.** `os_navpred` 0.008168 still above the `ha0` floor 0.006802 with
   near-best cross-track. A curvature-smoothed re-score of **either banked dump is zero GPU**.

## 10. WORK ITEMS

| id | item | cost |
|---|---|---|
| **NP-1** | ⭐ **DONE this turn** — `os_navflip` rolled and analysed (§1, §3, §4) | — |
| **NP-2** | paired margins for the LATERAL/TACTICAL family metrics (`_intervals_complete: false`) — §4's pattern is an observation until this lands | analysis only |
| **NP-3** | a **same-hardware** re-roll of `os` on the A40, or re-baseline the reference arms on Thor, to close C3/C6 exactly | ~15 min GPU |
| **NP-4** | ⛔ `refcv3_arm.py` `a_star` geometry (`D-REFCV4B-ASTAR-GEOMETRY`) — **still open, deliberately not touched here**: fixing it needs a validating `--with-oracle-sel` re-roll and it blocks every selector claim | ~2 lines + a T0 re-roll |
| **NP-5** | ⭐ **DONE this turn** — label-space audit, two sites, both fixed, no others found | — |
| **NP-6** | read `nav_compliance` out of both banked records (§9.2) — the behavioural test of the same question | **zero GPU** |
