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
