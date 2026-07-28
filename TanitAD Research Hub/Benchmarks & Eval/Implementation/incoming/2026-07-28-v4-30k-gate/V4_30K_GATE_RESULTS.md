# flagship-v4 FROM-SCRATCH @ 30k — the formal gate, run 2026-07-28

**Arm:** `flagship-v4-fromscratch`, **step 29,999 / 30,000**, ckpt md5 `8771c1d9d3da696dcde2a745d628f6a8`.
**Card:** `Project Steering/Gates/flagship-v4-30k.card.json` — `gate_step: 30000`, **registered
2026-07-26 at step 29,650, i.e. BEFORE this checkpoint existed.** No threshold was chosen after
seeing a number.
**Host:** pod2 (A40). **Estimator: episode-cluster bootstrap** (`taniteval.driving/tier0` v2.0.0),
**n = 881 windows / 40 episodes**, horizon **2 s** (4 waypoints 0.5 s apart).
`overlapping_holdout_se` used nowhere. Evidence class **MEASURED (ours)** unless stated.

## ⛔ 0. VERDICT: **INCOMPLETE — the gate cannot render a formal decision**, and of what IS
measured the arm fails two live secondaries decisively.

**Three of eight secondaries cannot be produced at all** (§3), and the **co-primary was not run**.
Per GATE_PROTOCOL that is a NO-VERDICT, not a PASS and not a KILL. What follows is reported so the
PI decides with the numbers in front of him, not instead of the protocol.

---

## 1. Primary — `ade_0_2s`, **both goal modes as a pair** (card requirement)

⚠️ **Card role: DIAGNOSTIC (DEMOTED per GATE_PROTOCOL 0.2).** Threshold `<= 0.6`.

| goal mode | `ade_0_2s` | 95 % CI (episode-cluster bootstrap) | vs threshold 0.6 |
|---|---|---|---|
| **oracle** (upper bound) | **0.6423** | **[0.5348, 0.7586]** | above, **CI straddles** — not separated |
| **produced** (deployable) | **0.8563** | **[0.7282, 1.0035]** | above, **CI entirely above** |

**The goal producer costs +0.2140 m** (0.6423 → 0.8563). ⚠️ These are two independent CIs, **not a
paired delta** — the paired episode-cluster bootstrap between the two modes was not run, so that
+0.2140 carries **no interval** and must not be quoted as separated.

### 1.1 ⭐ Lateral / longitudinal decomposition (card-required; an undecomposed L2 hides this)

| goal mode | **along-track** | **cross-track** | ratio |
|---|---|---|---|
| oracle | **1.0389** | 0.5242 | **1.98×** |
| produced | **1.4649** | 0.5515 | **2.66×** |

⇒ **The error is LONGITUDINAL-dominated, and the goal producer damages ONLY the longitudinal axis**
(along +0.4260, cross +0.0273). This is consistent with the program's standing
longitudinal-blindness finding and is the sharpest thing in this gate.

⚠️ **THESE ARE NOT THE SAME AGGREGATION AS THE `ade_0_2s` COLUMN ABOVE — settled in code
2026-07-28, because "along 1.0389 > ADE 0.6423" reads as impossible until you know why.**
`driving.py:201 frenet()`: the split is taken **on the GT TANGENT (Frenet) frame** — *"along + =
pred is AHEAD of GT along the path; cross + = pred is LEFT; orthonormal basis ⇒
along² + cross² == ‖pred−gt‖² exactly"* — and the reported fields are **`long_abs_2s_m` /
`lat_abs_2s_m`, i.e. AT the 2 s horizon**, whereas `ade_0_2s` is **averaged over 0–2 s**.
**Consistency check passes:** √(1.0389² + 0.5242²) = **1.1637**, which sits below `fde@2s`
**1.3230** in exactly the direction Jensen requires (mean-of-norms ≥ norm-of-means). ⇒ compare
along/cross against **fde@2s**, never against `ade_0_2s`.

### 1.2 ⭐ It does not beat constant velocity on speed

| goal mode | speed MAE | hold-v0 (CV) | `tracks speed > CV` |
|---|---|---|---|
| oracle | 0.5405 | **0.4818** | **False** |
| produced | 0.7507 | **0.4818** | **False** |

The harness's own summary for both arms: **`win lives: lateral only`.**

⚠️ **`0.4271` IS NOT THE BAR HERE.** `rollout.py:170` sets `actions_source="expert_future"` — it is
`wm_fidelity_ade_2s`, the world model **handed the expert's true future actions**. Registry §1.2:
*"0.4271 IS NOT A PLANNING BAR AND MUST NOT BE USED AS ONE."* The same-surface planning reference is
**0.4907** (881 windows / 40 episodes).

### 1.3 ⭐⭐ PAIRED episode-cluster bootstrap — added 2026-07-28, and it CHANGES THE READING

§1 left the +0.2140 without an interval. It has one now. **Paired** (`taniteval.ci`, B=2000,
seed 0), same arm, **same 881 windows / 40 episodes**, refused unless the eid sequence *and* the
ground truth are identical — both verified. Quadrature over §1's two intervals would have been
**invalid**, not merely weaker, because the arms are not independent.

| metric | produced | oracle | **Δ (produced − oracle)** | CI95 | |
|---|---|---|---|---|---|
| `ade_0_2s` | 0.8563 | 0.6423 | **+0.2140** | [+0.1602, +0.2759] | **SEPARATED** |
| de@0.5 s | 0.1533 | 0.1155 | +0.0377 | [+0.0272, +0.0492] | SEPARATED |
| de@1 s | 0.4929 | 0.3635 | +0.1294 | [+0.0952, +0.1701] | SEPARATED |
| de@1.5 s | 1.0411 | 0.7672 | +0.2739 | [+0.2039, +0.3555] | SEPARATED |
| **fde@2 s** | 1.7378 | 1.3230 | **+0.4148** | [+0.3099, +0.5344] | SEPARATED |

**The goal producer's cost COMPOUNDS: ~11× from 0.5 s to 2 s** (+0.0377 → +0.4148).

#### ⭐⭐ Against the constant-velocity floor — the single most decision-relevant number here

| arm | ADE | CV floor | **Δ vs CV** | CI95 | |
|---|---|---|---|---|---|
| **produced** (deployable) | 0.8563 | 0.8377 | **+0.0186** | **[−0.1711, +0.1940]** | **overlaps 0** |
| oracle (upper bound) | 0.6423 | 0.8377 | **−0.1954** | [−0.3713, −0.0418] | SEPARATED, better |

⇒ **THE ARM'S ENTIRE MEASURED ADVANTAGE OVER CONSTANT VELOCITY DEPENDS ON BEING HANDED AN ORACLE
GOAL.** With the goal it can actually produce, it is **statistically indistinguishable from doing
nothing** — nominally worse, CI straddling zero. This is consistent with, and much sharper than,
the harness's own `tracks speed > CV: False` and `win lives: lateral only`.
⚠️ Stated precisely: *indistinguishable*, **not** "proven no better" — the CI is wide
([−0.171, +0.194]) on 40 episodes, so a real effect either way is not excluded.

⚠️ **The lateral/longitudinal split is NOT quoted in this paired block, deliberately.** A
self-check was built in: reproduce `taniteval.driving`'s own along/cross before quoting mine. ADE
matched to **d = 0.0000** on both arms, but my ego-frame |Δx|/|Δy| gave along **0.5159** where
driving reports **1.0389** (and cross 0.2389 vs 0.5242) — a different definition, ~2×, not a clean
factor. **The guard fired and the decomposition was withheld.** §1.1's decomposition stands because
it is `driving.py`'s own output; only my re-derivation was refused.

✅ **RECONCILED the same day (see §1.1's warning).** My version was wrong in **two** ways at once,
which is why the ratio looked arbitrary rather than like a clean factor: wrong **frame** (I used the
ego frame; `driving.py` uses the **GT-tangent Frenet** frame, which rotates with the path — the only
frame in which "along-track" and "cross-track" mean what the words say) **and** wrong **aggregation**
(I averaged over the window; `driving.py` reports **at 2 s**). ⇒ **`driving.py`'s numbers are the
correct ones and mine were not a competing estimate, just a different quantity.** A corrected paired
decomposition would need `frenet()` applied per window — worth doing, not done here.

## 2. Co-primary — **NOT RUN**

`corridor_departure_rate` @ K=185 (18.5 s), 1.75 m corridor. The card marks it
**REPORT_ONLY_THIS_GATE** (no kill threshold has ever been agreed, and inventing one now would be
forking-path abuse). It requires the closed-loop corridor harness, which was not exercised here.
**Absent, and its absence changes no threshold.**

## 3. Secondaries — 3 PASS · 2 FAIL · **2 UNPRODUCIBLE** · 1 VOID

| # | metric | threshold | measured | verdict |
|---|---|---|---|---|
| 1 | `wm_canary_ade_2s` | ≤ 0.55 | **1.1409** (n=881) | 🔴 **FAIL — 2.07× the bar** |
| 2 | `speed_benefit_recovered_frac` | ≥ 0.70 | **null** | ⚪ **NO EMITTER EXISTS** anywhere in the codebase |
| 3 | `oracle_in_fan` | ≤ 0.30 | **0.2330** (produced 0.2505) | ✅ PASS |
| 4 | `miss_at_2m` | ≤ 0.10 | **0.2123** (produced 0.3190) | 🔴 **FAIL — 2.1×** |
| 5 | `seam_norm_ratio_max` | ≤ 1.0 | **0.1208** (n=56 samples) | ✅ PASS, with margin |
| 6 | `encoder_touching_levers` | ≤ 2 | **2** | ✅ PASS (PUBLISHED static design fact, not re-derived) |
| 7 | `deploy_tick_p99_ms` | ≤ 50 | **null** | ⚪ **NO v4-AWARE PANEL** (`efficiency.py` has zero v4 awareness) |
| 8 | `nonav_route_beats_majority` | ≥ 1 | — | ⚫ **VOID BY CONSTRUCTION → INSTRUMENT-FAIL, NEVER MODEL-FAIL** (GATE_PROTOCOL 0.7) |

**#8 printed explicitly per the card's required_reporting.** Its reason: the route TARGET is a lookup
of the route INPUT (`route_target = _NAV_TO_ROUTE[nav_cmd]`), measured identical on **100.00 %** of
CE-eligible windows under v1, v2 AND v2.1 — **no labeler swap can fix it.** A healthy arm must never
die on this.

⭐ **The two FAILs are the informative ones and they agree with each other.** `wm_canary` 1.1409 says
the **world model itself** integrates known actions far worse than v1 (0.4215 on the identical
surface, measured minutes earlier — §5). `miss_at_2m` 0.2123 says one window in five ends >2 m from
truth. Together with §1.1 they describe one failure, not three: **a longitudinally weak world model**.

## 4. ⚠️ Caveats that bound every number above

1. **Frame UNVERIFIED.** The checkpoint carries no `geometry` block, so the harness states plainly:
   *"the eval frame above is an assumption, not a match."* Evaluated at the deployed
   256×256 / f_ref 266 / pinhole / **HFOV 51.394°**.
2. **Parity is COUNT-ONLY** for this val split (600/600 episodes present; **no uid digest is
   committed**, so membership is not cryptographically checked). Fix once with
   `scripts/make_parity_manifest.py --record --split val`.
3. **No paired bootstrap** between goal modes (§1), and none against v1.
4. `--require-parity` was **not** passed on the reported runs.

## 5. ✅ MODE A — the harness check that makes §1 admissible (GATE_PROTOCOL O-03)

Run **before** any v4 number was quoted, on the identical surface:

| field | value |
|---|---|
| arm | v1 `flagship4b-speedjerk-30k`, md5 `b5f07d9e3dd2ca643949bc86832e6585` |
| `n_windows` | **881** |
| `canary_ade_2s_MEASURED` | **0.42148** |
| Δ vs full-set 0.4271 | **−0.0056** (tolerance 0.05) |
| Δ vs heldout 0.4522 | −0.0307 |
| verdict | **HARNESS VALIDATED** |

The harness states the methodology point itself: the matched comparison is the **full-set** figure,
not the 0.4522 split-mean — *"a different statistical construction"*.

⚠️ **Do not misread the canary run on the v4 arm (§3 #1), which prints "HARNESS NOT VALIDATED".**
That message mechanically compares **any** `--canary-only` run to **v1's** reference; on a v4 arm it
is a category confusion in the script's messaging, not a harness failure. The harness was validated
above, on v1, minutes earlier.

## 6. What it took to run this at all — 3 defects, all fixed or disclosed

1. **The arm could not be LOADED.** Five missing head keys. **Four were self-inflicted:** the ckpt
   sat at `v4fs_ckpt.pt` with its config at `v4fs_config.json`, so the loader's
   `<ckpt-dir>/config.json` auto-detect missed it and built the head from **current** defaults, which
   enable imagination — the run's own config says `"cond_imagination": false`. `--head-config` fixed
   all four. *(Syncing pod2 forward 91 commits is right for LAUNCHING and is exactly what shifts the
   architecture out from under an OLD checkpoint being EVALUATED.)*
2. **The fifth key was real and benign:** `vision_rank_proj.basis_loaded`, a scalar bool buffer added
   after this arm trained and **never read in `forward()`** (verified in source). Fixed with a
   one-entry allowlist; **anything else missing or unexpected still raises**, pinned by 5 tests
   including a red half that fails if the allowlist widens or the load degrades to bare
   `strict=False`. `stack`: 73 passed / 1 skipped on the eval+preflight+vision_rank suites.
3. **The B4 trap fired exactly as documented.** `eval_flagship_v4.py` imports `taniteval`
   **non-fatally**; without it on `PYTHONPATH` the run **exits 0 and silently drops the bootstrap
   primary to null**. The first oracle run "succeeded" that way. ⚠️ **And the JSON's
   `cross_check…driving_py_from_persisted_windows` field stays `null` even when driving DID run** —
   the real result is in the `driving` block and the `[driving]` log line. **Never conclude the
   primary is missing from that cross-check field alone; I did, and cried wolf.**

## 7. Provenance

`pod2:/workspace/v4gate30k/` — `v4fs-30k-oracle.json` · `v4fs-30k-produced.json` ·
`modeA_v1.json` · `canary_v4fs.json` · `windows_v4fs-30k-{oracle,produced}.pt` ·
`modeB_{oracle,produced}.log`. Runner: `code/gate30k_modeB.sh`.
Checkpoints reached pod2 via the **old US-TX-1 pod as a cross-DC relay** (37 MB/s per hop, 336 s
total, **md5 identical end-to-end**) because pod→pod direct SSH does **not** work same-datacenter.

## 8. What the PI owes a decision on

1. **This is a NO-VERDICT, not a failure to decide.** Do we (a) accept an incomplete gate on the two
   measured FAILs, (b) build the two missing emitters first, or (c) run the co-primary and re-adjudicate?
2. **`speed_benefit_recovered_frac` and `deploy_tick_p99_ms` have no emitter and nobody owns them** —
   this is the third gate they have blocked.
3. **The checkpoint is still SINGLE-DISK.** The HF backup is written and ready but was
   **classifier-blocked**; it needs your permission (task #41).
