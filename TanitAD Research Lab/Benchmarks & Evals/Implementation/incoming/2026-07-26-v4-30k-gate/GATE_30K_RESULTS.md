# flagship-v4-fromscratch — the 30 k GATE

**Date:** 2026-07-26 (Europe/Berlin; pods and logs are UTC)
**Run:** `flagship-v4-fromscratch`, `final_step 29999`, `wallclock_s 212544.6` (59.0 h)
**Card:** `Project Steering/Gates/flagship-v4-30k.card.json` — registered
`2026-07-26T08:26:00+00:00` at step 29,650/30,000, i.e. **before this checkpoint existed**.
Nothing below was re-tuned. No threshold was invented, moved, or read after the fact.

Evidence class is stamped on every number:
`MEASURED` (ours + artifact) · `PUBLISHED` · `INHERITED` · `ESTIMATED` · `HYPOTHESIS`.

---

## 0. HEADLINE

| | value | estimator | n |
|---|---|---|---|
| **ade_0_2s, goal ORACLE** (primary surface) | **0.6423 [0.5348, 0.7586]** | episode_cluster_bootstrap | 881 win / 40 ep |
| **ade_0_2s, goal PRODUCED** (deployable path) | **0.8563 [0.7282, 1.0035]** | episode_cluster_bootstrap | 881 win / 40 ep |
| paired **produced − oracle @ 30 k** | **+0.2140 [+0.1602, +0.2759]** SEPARATED | paired_episode_cluster_bootstrap | 881 / 40 |
| paired **v4-oracle − v1** | **+0.2152 [+0.0972, +0.3340]** SEPARATED | paired_episode_cluster_bootstrap | 881 / 40 |
| paired **v4-produced − v1** | **+0.4292 [+0.2865, +0.5820]** SEPARATED | paired_episode_cluster_bootstrap | 881 / 40 |
| **co-primary** `corridor_departure_rate` @ K=185 *(REPORT-ONLY)* | **0.6388 [0.5565, 0.7128]** overall · **0.8432 [0.7874, 0.8919]** junction | episode_cluster_bootstrap | **41 win / 40 ep** |

**VERDICT: `INCOMPLETE` formally — but NOT-CONTINUE is already determined.**
Horizon **K=20 (2.0 s)**, **n = 881 windows / 40 episodes** (GATE_PROTOCOL 7 rule 3).
Two pre-registered kill secondaries FAIL on MEASURED values; two were never measured.
No future measurement can turn a FAIL into a PASS, so the kill conjunction is
already unsatisfiable. With `restarts_used 0 / restart_cap 2` the resolved verdict
is **RESTART**, not `REFUTE_LEVER_FAMILY`.

> **§0.8 wording constraint honoured.** The primary above is **goal-oracle-fed**
> (`route` / `route_graded` / `vt_band` minted from the ego's own future poses).
> 0.6423 is an **upper bound on a MODE-B surface**, *not* a deployed capability.
> The deployable number is the produced one, **0.8563**, and it does **not** beat
> the constant-velocity floor.

---

## 1. TASK 1 — the checkpoint is backed up (this was the single point of failure)

`metrics.json` lists `milestone_archives = [5000, 10000, 15000, 20000]`. There is
**no 25 k and no 30 k milestone**: the finished state of a 59-hour run existed only
in `ckpt.pt`, on one pod disk.

**HF repo (gated): https://huggingface.co/Sayood/flagship-v4-fromscratch**

| file | bytes | md5 | verification |
|---|---|---|---|
| **`ckpt.pt`** (final, step 29999) | 3,243,109,310 | **`8771c1d9d3da696dcde2a745d628f6a8`** | remote LFS sha256 == local sha256 |
| `ckpt_step20000.pt` (last milestone) | 3,243,109,310 | `d09bd016788bf6b4211ae92b0ca1dfbe` | remote LFS sha256 == local sha256 |
| `config.json` | 8,233 | `57223e6f58d71c02eefd6c92e0b3364a` | size match (non-LFS) |
| `metrics.json` | 464 | `3c62a52cb44d39c773490c2f2ff19143` | size match (non-LFS) |
| `train_log.jsonl` | 245,605 | `7d8bdeb458e064ab36216f9d03214c61` | size match (non-LFS) |

`ckpt.pt` sha256 = `7bc977d670780cfc1efa45acce07135d692b76b0a1045bab88d9626e741d5144`.

Invariants held (MEASURED, `raw/v4_hf_push_receipt.json`):

1. **`gated="manual"` was set and READ BACK BEFORE any weight byte was sent.** The
   repo was created empty, gated, then `repo_info` re-read (`GATED_READBACK
   gated='manual'`) and the script refuses to upload otherwise. Log line
   `GATE_VERIFIED_BEFORE_ANY_WEIGHT_BYTE` precedes the first upload.
2. **md5 verified end to end, both directions.** Push from pod2 → HF verified by
   remote LFS sha256. Then pulled HF → eval pod and **re-hashed**: md5 came back
   `8771c1d9d3da696dcde2a745d628f6a8`, identical
   (`raw/pull_receipt.json`, `PULL_RESULT ALL_ROUNDTRIP_VERIFIED`). The bytes that
   were evaluated below are provably the bytes that were archived.
3. **Token read from `Keys.txt` in place**, piped over ssh stdin into a tmpfs file
   (`/dev/shm`, mode 600, RAM only), deleted on first read. Never printed, never on
   a real disk, never in argv.
4. **`ckpt.pt` was never copied, moved, renamed or truncated** — opened `'rb'` only.
5. Fast path used: pushed from pod2 (which was free), ~62 MB/s sustained, ~52 s per
   3.24 GB file — not the ~1 MB/s dev-box relay.

**Verified three independent ways**, not once:
(a) push-time `repo_info` gate readback + remote LFS sha256 comparison;
(b) **round-trip** — pulled back to a *different host* (the eval pod) and re-hashed,
    md5 identical (`ALL_ROUNDTRIP_VERIFIED`);
(c) a final independent `repo_info(files_metadata=True)` listing after all work:
    `gated: manual`, commit `8a08b6e5ac69650aafd465f73f3f7fd62c7fa4a0`, 6 files,
    `ckpt.pt` 3,243,109,310 B sha256 `7bc977d6…5144` — matching the local hash.

---

## 2. PREFLIGHT — all three checks the card mandates: **PASS**

Artifacts: `raw/preflight_1_2.json`, `raw/v1_preflight_flagship30k.json`.

### 2.1 `corridor.py` present on the executing host — **PASS**
It was **missing** from the eval pod earlier on 2026-07-26, so the co-primary
emitter did not exist. Re-verified, and not merely by `ls`:

- present at `/root/taniteval/taniteval/corridor.py`, md5 `9f064714f35be7d172d228cfb26c5976`
- **byte-identical to the repo copy** (`taniteval/taniteval/corridor.py`, same md5)
- imports, and was **exercised end to end** on a synthetic set: it emits
  `corridor_departure_rate`, `corridor_departure_rate_by_threshold_m`,
  `corridor_primary_m`, `horizon_K`, `horizon_s`, `estimator`, `n_episodes`,
  `mean_abs_xte_by_step_m`, `EXTRAPOLATION_frac_steps_lat_over_3m`
- `CORRIDOR_HALFWIDTH_M = 1.75`, `JUNCTION_DEG = 10.0`, `horizon_seconds(185) = 18.5`
- `from_windows`, `stratified`, `paired_stratum_delta` all present

"Present" here means *produces a departure rate*, not *the file exists*.

### 2.2 `lateral.py` emits `horizon_provenance` and `horizon_s = 2.0` — **PASS**
A **0.4 s** reading would mean stale code (the pre-fix version mislabelled the
horizon **5×**). MEASURED by *running* it, not by reading it:

| | emitted |
|---|---|
| `surface` | `sparse_4wp` |
| `dt_s` | 0.5 |
| `horizon_K` | 4 |
| **`horizon_s`** | **2.0** (not 0.4) |
| `by_horizon` labels | `0.5s, 1s, 1.5s, 2s` |
| `horizon_provenance` (paired emitter) | `inferred_from_knot_count`; `explicit` when `knot_dt` is passed |
| paired `horizon_s` | 2.0 under both inference and explicit `knot_dt=0.5` |

md5 `a3b3d4919e0b0aa966ec11d0515ea814` — **byte-identical to the repo copy**.

### 2.3 v1 reproduces `0.4271` — **PASS (twice, independently)**

- **MODE A canary**, `/root/models/flagship-30k/ckpt.pt`: **0.42148**, Δ vs the
  registry full-set 0.4271 = **−0.0056 m**, tolerance 0.05 →
  `HARNESS_VALIDATED: true` (GATE_PROTOCOL O-03 satisfied). 881 windows, 26.1 s.
- **Second, independent leg:** v1's own window dump `windows_flagship-30k.pt`,
  re-scored on the 4-waypoint convention, gives **0.4271090** — the registry number
  reproduced to 5 decimal places on the *identical* 881 windows the v4 gate ran on.

This also settles the **v1 identity trap by measurement, not by name**: the registry
warns that `flagship4b-phase0-30k` is the no-speed ablation control (2.918 m) and
NOT the deployed v1. A checkpoint reading 0.4215 is unambiguously the speedjerk v1;
the ablation control would have read ~2.9.

### 2.4 Module-sync check (not mandated, but the brief warns a module was found stale after a "141/141 verified" sync)
All gate-critical modules on the eval pod were md5-compared against the repo:
`goal_modes.py`, `goal_provenance.py`, `flagship_v4_data.py`, `gate_emitters.py`,
`run_gate.py`, `v4_labels.py`, `driving.py`, `ci.py`, `closedloop.py`, `bench.py`,
`corridor.py`, `lateral.py` — **all identical**.
`eval_flagship_v4.py` initially showed a different md5; **checked rather than
assumed** — it is a CRLF/LF artifact of the Google-Drive checkout, and the files are
byte-identical after normalisation (both `aafe3975817e27ef7714499643aae6ff`).
**No stale module.**

---

## 3. INSTRUMENT DEFECTS FOUND WHILE RENDERING (escalation, not a model fact)

The three mandated preflight checks passed. But the **gate renderer itself**
(`stack/scripts/run_gate.py`) could not render this card. Three distinct defects,
each MEASURED. The card is binding and was **not** modified; `run_gate.py` was
**not** patched at gate time (that would be re-tuning the instrument mid-gate).

**D1 — the registered card is not machine-readable.**
`cmd_check` does `GateCard(**json.loads(card))`, and `GateCard` is a dataclass, so
the card dies on its own registered fields:
```
TypeError: GateCard.__init__() got an unexpected keyword argument
           'registered_before_checkpoint_exists'
```
**11 card keys are unknown to the dataclass:** `co_primary`, `goal_provenance`,
`goal_provenance_note`, `preflight_checks`, `primary_role_note`,
`reference_ade_0_2s`, `reference_note`, `registered_before_checkpoint_exists`,
`registration_note`, `required_reporting`, `secondary_void`. In addition the card
supplies `co_primary` as a **nested dict** where the tool expects **flat**
`co_primary_*` fields — so on a literal projection `has_co_primary` is `False`.

**D2 — `role: REPORT_ONLY_THIS_GATE` is not implemented.**
`run_gate.py` contains no reference to `REPORT_ONLY` or `secondary_void` anywhere.
Its kill conjunction is `kill_inputs = [co_ok] if card.has_co_primary else [passed]`.
So the tool has **no state** matching this card's actual configuration — *primary
demoted to diagnostic* **and** *co-primary report-only*, i.e. **secondaries alone
adjudicate**. Whichever way the card is projected the tool mis-adjudicates: with the
co-primary unmapped the **demoted** `ade_0_2s` illegally re-enters the kill
conjunction; with it mapped, an unmeasured, deliberately **unthresholded**
co-primary forces `INCOMPLETE`.

**D3 — a 0-indexed/1-indexed off-by-one refuses the verdict outright.**
Both projections returned:
```
VERDICT: NOT_YET — step 29999 < pre-registered gate step 30000.
```
The run is **complete**: `config.json /args/steps = 30000`, `train_log.jsonl` spans
`min_step 0 → max_step 29999` (661 rows), `metrics.json final_step 29999`, trainer
exited, 212,544.6 s wallclock. A 30,000-step run indexes 0…29999. `cur <
card.gate_step` compares a 0-indexed counter against a 1-indexed count. This is the
same failure class as commit `3ff5499` ("the 30k gate would have produced NO
VERDICT"); a sibling instance survived that fix.

**Consequence:** the verdict in §4 is adjudicated **against the card's own text**,
with every criterion printed. Machine output from both projections is preserved in
`raw/GATE_30K_verdict_A_no_coprimary.json` and
`raw/GATE_30K_verdict_B_coprimary_registered.json` and is **NOT** the verdict.

**None of this changes the outcome.** Two on-card kill secondaries fail on MEASURED
values by 2× or more; no reading of the card, and no repair of the renderer, makes
that conjunction pass.

---

## 4. THE GATE — adjudicated against the card as written

### 4.1 Primary — `ade_0_2s` (DEMOTED to diagnostic by the card, `primary_role: diagnostic`)

| | value | threshold | result |
|---|---|---|---|
| `ade_0_2s`, goal **ORACLE** | **0.6423 [0.5348, 0.7586]** | ≤ 0.60 | **FAIL** |
| `ade_0_2s`, goal **PRODUCED** | **0.8563 [0.7282, 1.0035]** | ≤ 0.60 | **FAIL** |

estimator `episode_cluster_bootstrap`, n_boot 2000, 881 windows / 40 episodes,
horizon K=20 = 2.0 s. **Recorded; does NOT adjudicate alone** (card §`primary_role`).

Against the trivial floors (constant velocity, ADE 0.8377 [0.6352, 1.0899]), paired
on the same windows (`paired_episode_cluster_bootstrap`, oriented `cv − model`, so
positive = the model wins):

| goal mode | paired Δ vs CV | separated |
|---|---|---|
| **ORACLE** | **+0.1954 [+0.0418, +0.3713]**, `p_delta_gt0 = 0.995` | **YES** — beats CV |
| **PRODUCED** | **−0.0186 [−0.1940, +0.1711]**, `p_delta_gt0 = 0.3995` | **NO** — CI spans zero |

**Oracle beats constant velocity, separated. PRODUCED DOES NOT**
(`beats_cv_separated: false`): the deployable path is *nominally behind* CV
(0.8563 vs 0.8377) and statistically indistinguishable from it.

### 4.2 Co-primary — `corridor_departure_rate` @ K=185 — **REPORT-ONLY, does not adjudicate**
See §6. Per card `co_primary.role = REPORT_ONLY_THIS_GATE`, it is **excluded from the
kill conjunction at this gate** and exists to set the next gate's bar.

### 4.3 Secondaries (the kill set that actually adjudicates here)

| # | secondary | bar | MEASURED | result |
|---|---|---|---|---|
| 1 | `wm_canary_ade_2s` | ≤ 0.55 | **1.1409** | ❌ **FAIL** (2.07× over) |
| 2 | `speed_benefit_recovered_frac` | ≥ 0.70 | *null* | ⬜ **NOT MEASURED** — no emitter exists anywhere in the codebase (P8 metric, never built) |
| 3 | `oracle_in_fan` | ≤ 0.30 | **0.2330** (oracle) / 0.2505 (produced) | ✅ PASS |
| 4 | `miss_at_2m` | ≤ 0.10 | **0.2123 [0.1508, 0.2830]** (oracle); 0.3190 (produced) | ❌ **FAIL** (2.12× over) |
| 5 | `seam_norm_ratio_max` | ≤ 1.0 | **0.1208** | ✅ PASS |
| 6 | `encoder_touching_levers` | ≤ 2 | **2** | ✅ PASS *(PUBLISHED design fact, not a GPU measurement)* |
| 7 | `deploy_tick_p99_ms` | ≤ 50 | *null* | ⬜ **NOT MEASURED** — needs the `efficiency.py` latency panel under `gpu_lock.sh`; out of scope this session |

**3 PASS · 2 FAIL · 2 NOT MEASURED.**

Neither missing value was guessed. Supplying a placeholder for a pre-registered bar
41 minutes after the number landed is precisely the forking-paths failure
GATE_PROTOCOL §0.3 forbids.

### 4.4 VERDICT

> **`INCOMPLETE`** by the card's own machinery — *a pre-registered secondary gate was
> not measured* (two of them).
>
> **But NOT-CONTINUE is already determined.** `wm_canary_ade_2s` (1.1409 vs ≤ 0.55)
> and `miss_at_2m` (0.2123 vs ≤ 0.10) FAIL on MEASURED values. A conjunction with a
> hard FAIL cannot be rescued by measuring anything else. When the two outstanding
> secondaries land, this resolves to **`RESTART`** — `restarts_used 0 / restart_cap 2`
> for lever family `joint-planner-wm`, so **not** `REFUTE_LEVER_FAMILY`.
>
> Horizon **K=20, 2.0 s**; **n = 881 windows / 40 episodes**;
> estimator `episode_cluster_bootstrap`; `goal_provenance: ORACLE` on the primary.

**The WM canary is the loudest signal and it is not marginal.** The from-scratch
canary descended from a `canary_baseline` of 15.674 to **1.1409** — that descent
through full planner coupling is the v4 thesis and it is real. But the bar is 0.55,
and 1.1409 is more than double it: the jointly-trained trunk is a **materially worse
world model** than v1's, whose canary *is* 0.4271. The v4 thesis is demonstrated
directionally and **fails its pre-registered quantitative bar**.

---

## 5. §0.7 — THE VOID SECONDARY, PRINTED EXPLICITLY

> A suppressed criterion that is not printed is indistinguishable from one that
> passed. So it is printed.

```
[VOID]  nonav_route_beats_majority        original bar: >= 1
        STATUS       : VOID_BY_CONSTRUCTION
        ADJUDICATION : INSTRUMENT-FAIL — NEVER MODEL-FAIL
        AUTHORITY    : GATE_PROTOCOL 0.7
        IN KILL SET  : NO — structurally excluded (card `secondary_void`,
                       not `secondary`); it did NOT contribute to the verdict
        MEASURED     : value null on this checkpoint
```

**Why it is void.** The strategic route TARGET is a lookup of the route INPUT
(`refb_labels.route_target = _NAV_TO_ROUTE[nav_cmd]`), so training route-CE reaches
exactly 0.0 by ~step 14.5 k and `route_skill` is 0.0 **by construction**; the
follow-head answers straight 240/240. MEASURED 2026-07-26: `route_target ==
_NAV_TO_ROUTE[nav_cmd]` on **100.00 %** of CE-eligible windows under v1, v2 **and**
v2.1 — the command is minted from the same `route_from_future*` call as the target,
so **no labeler swap can break it**. The metric measures the label bug, not the model.

**Second, independent reason it is unreachable on *this* checkpoint** (MEASURED, from
`raw/*_v4_diagnostics.json`): v4's `goal_head` is a `GoalScalarHead` that regresses
only continuous scalars (`ttm`, `curv_3s`, `curv_5s`, `tspeed_5s`). **No route
classifier exists** — P6 (strategic planner) has not landed, and
`taniteval.hierarchy.vision_route_beats_majority` needs a nav-conditioned route head
this checkpoint does not have.

**It re-arms** when an arm trains with real route supervision
(`--labels-v21 --v2-route-from-vision`), at which point it returns to the kill set.

**This criterion did not, and could not, count against the model.**

---

## 6. CO-PRIMARY — `corridor_departure_rate` @ K=185 (REPORT-ONLY at this gate)

Card: half-width **1.75 m**, horizon **K=185 = 18.5 s**, surface **closed-loop**,
estimator **episode_cluster_bootstrap**, junction stratum reported separately.
Role **`REPORT_ONLY_THIS_GATE`** — it does **not** adjudicate here; its purpose is to
set a real threshold for the next v4-line gate against a measured baseline. Inventing
a bar now would be forking-paths, and the two errors are asymmetric: too tight kills a
healthy arm, too loose launders a bad one.

**Status: MEASURED.** A dedicated v4-aware closed-loop driver was required, was built,
design-validated, and run. Numbers in §6.3. It does **not** change the verdict.

### 6.1 The driver (MEASURED metadata, `/workspace/_v4gate/corridor_v4_30k_K185.json`)

`v4_corridor_cl.py` reproduces `e1a_horizon.py`'s loop body, window/stratum
bookkeeping, OOD accounting and common-start paired design verbatim — the design
that produced the card's own REF-C reference at the same K=185. The single change is
the per-step plan call: `world.encode_window → goal_modes.resolve_goal →
head(st, v0, lambda_plan=1.0, **goal_kw)`, with the 0.5 s lookahead waypoint
`traj[:, 4]` fed to the same pure-pursuit controller. Emitter is
`taniteval.corridor.stratified` — the registered co-primary emitter. Estimator
`episode_cluster_bootstrap` B=2000, paired form for K-vs-K deltas.

Checkpoint identity is stamped: `ckpt_md5 8771c1d9d3da696dcde2a745d628f6a8`,
`ckpt_step 29999` — the same bytes that were archived and that produced §4.

**Plumbing self-check passes.** The driver recomputes the open-loop 4-waypoint
`ade_0_2s` on the same 881 windows and gets **0.6423 [0.5359, 0.7595]** — reproducing
the gate primary in §4.1 (0.6423 [0.5348, 0.7586]). The closed-loop numbers therefore
come from the same forward pass that produced the gate primary, not a second,
divergent implementation.

**Goal-index policy** (a real design decision, recorded): the oracle goal is
**re-minted at every rollout step** at the reference index the model is actually
observing (`t0 + mstar + W − 1`), by the same `v4_labels.mint_window` call the
dataset makes — the closed-loop analogue of the open-loop oracle, **not** a route
handed over once at t=0. `goal_labeler_refusals: 0` over 7,964 indices.

### 6.2 The finding that already matters for the next gate: **n is tiny at K=185**

| horizon | windows | episodes |
|---|---|---|
| K=20 (2.0 s) | **881** | 40 |
| **K=185 (18.5 s)** | **41** | 40 |

MEASURED: episode length on this val cache is `T = 198–205` frames, and a window
exists at K only if `T − W − K ≥ 1`, so the **structural ceiling is K=196 (19.6 s)**
and at K=185 **roughly one window per episode survives**. The card's REF-C reference
was likewise 43 windows (on a *different* 44-episode held-out cache —
`physicalai-val-heldout-79d4e3d2d4c6`, so it is a **scale reference, NOT a
window-matched pair**; evidence class INHERITED, not re-run here).

**Implication for threshold-setting, which is this co-primary's whole purpose at this
gate:** any K=185 corridor threshold will rest on ~40 windows clustered in 40
episodes — one per episode, so the episode-cluster bootstrap has essentially no
within-cluster averaging to exploit. Intervals will be wide. A bar registered off
this surface must be set from the interval, not the point estimate, or it will be
noise-dominated in exactly the asymmetric way the card warns about (too tight kills a
healthy arm; too loose launders a bad one). **n must be quoted with any number at this
horizon** — a corridor number without its K *and* its n is not admissible.

### 6.3 THE CO-PRIMARY, MEASURED — `corridor_departure_rate` @ K=185, 1.75 m

Goal mode **ORACLE** (matching the gate primary); `rollout_advanced_K_steps: true`,
7,400 rollout steps executed. Estimator **`episode_cluster_bootstrap`**, B=2000,
resampling unit = val episode. `overlapping_holdout_se` explicitly refused by the block.

| stratum | n (win/ep) | **corridor_departure_rate @1.75 m** | window departure rate | peak \|XTE\| (m) | mean \|XTE\| (m) |
|---|---|---|---|---|---|
| **OVERALL** | 41 / 40 | **0.6388 [0.5565, 0.7128]** | 0.9512 [0.8810, 1.0000] | **33.45 [24.15, 43.81]** | 10.42 [7.79, 13.44] |
| **JUNCTION** | 6 / 6 | **0.8432 [0.7874, 0.8919]** | 1.0000 [1.0, 1.0] | 33.57 [19.69, 49.26] | 14.88 [10.03, 20.60] |
| longitudinal | 18 / 18 | 0.6871 [0.6138, 0.7496] | 1.0000 [1.0, 1.0] | **46.96 [31.16, 64.53]** | 13.86 [9.88, 18.17] |
| other | 17 / 16 | 0.5154 [0.3807, 0.6520] | 0.8824 [0.7365, 1.0000] | 19.11 [10.34, 31.29] | 5.20 [3.16, 7.74] |

By threshold (overall): **1.00 m → 0.7048** [0.6285, 0.7729] · **1.75 m → 0.6388** ·
junction at 1.00 m → 0.8811 [0.8333, 0.9189].

**Read.** At the event's own horizon the arm leaves a 1.75 m corridor for **64 % of
the rolled-out steps**, and **95 % of windows depart at least once** (100 % in the
junction and longitudinal strata). Mean cross-track error is **10.4 m** and peak
**33.5 m**. The junction stratum separates sharply upward (0.8432 vs 0.6388) — the
same concentration E1a found on REF-C. Compare §8: at the **2 s** horizon only 6.1 %
of windows leave the same corridor. **The 2 s instrument understates corridor failure
by roughly an order of magnitude on this arm too** — which is exactly why `ade_0_2s`
was demoted, and it is now confirmed on the flagship line, not just on REF-C.

**vs REF-C base — re-measured on the SAME 41 windows** (`raw/corridor_refcbase_30k_K185.json`,
MEASURED, not inherited):

| arm (same 41 windows / 40 episodes) | overall CDR@1.75 m | junction CDR |
|---|---|---|
| REF-C base 30k | **0.5833 [0.5024, 0.6561]** | 0.7027 [0.4099, 0.8856] |
| **flagship-v4 30k** | **0.6388 [0.5565, 0.7128]** | **0.8432 [0.7874, 0.8919]** |

v4 is nominally **worse than REF-C base** at the event horizon, overall and at
junctions. ⚠️ **The paired delta had not finished when this was written**, so
**no separation may be claimed** — these are two single-arm intervals and they
overlap. The paired episode-cluster bootstrap is the only admissible form for a
window-matched arm-vs-arm read (combining two single-arm intervals in quadrature is
invalid here, not merely weaker) and it follows as an addendum.

*Cross-validation:* REF-C base reads **0.5833** on this 40-episode cache vs E1a's
**0.5877** on `physicalai-val-heldout-79d4e3d2d4c6` (44 eps / 43 windows). Two
different caches, two different runs, 0.004 apart — which independently corroborates
both the E1a reference and this driver.

### 6.4 ⚠️ OOD: the block's own summary verdict is too generous — this is EXTRAPOLATION-contaminated

| OOD observable (overall) | value |
|---|---|
| `ood_peak_ratio` | 1.2741 [1.2509, 1.2946] |
| `ood_mean_ratio` | 1.1762 [1.1528, 1.1974] |
| `frac_windows_ood_peak_under_1.5` | 1.0 |
| **`EXTRAPOLATION_frac_steps_lat_over_3m`** | **0.5463** |
| **`EXTRAPOLATION_frac_steps_yaw_over_12deg`** | **0.3548** |
| **`EXTRAPOLATION_frac_windows_any_step_out_of_envelope`** | **0.9024** |
| emitted `EXTRAPOLATION_VERDICT` | *"within the measured envelope on average"* |

**That emitted verdict satisfies only the first half of E1a's rule and must not be
quoted bare.** E1a's stated criterion is: *any horizon whose peak OOD ratio exceeds
~1.5×, **or whose steps leave the measured envelope**, is EXTRAPOLATION, not
measurement.* Here the aggregate ratio is comfortably under 1.5 (1.274) — but
**54.6 % of rolled-out steps exceed the |dlat| ≤ 3.0 m envelope**, 35.5 % exceed
|dyaw| ≤ 12°, and **90.2 % of windows leave the envelope at some step**. The second
clause fires decisively.

The real-footage source re-indexes along a 1-D manifold and `np.interp` **CLAMPS**
outside the P1 envelope, so beyond it the reported OOD ratio is a **lower bound**.

**Therefore: `0.6388` is a real, MEASURED closed-loop number, but it is not a clean
in-distribution measurement — most of its rollout sits outside the validated warp
envelope, where the simulator degrades gracefully rather than faithfully.** For
threshold-setting this matters more than the point estimate: the next gate should
either register its bar at a horizon where the envelope holds, or re-validate the P1
envelope out to K=185 first. Note also the envelope constants were measured on the
**flagship v1** arm (`lowood_flagship_ci.json`), not on v4.

The open-loop gate surface **structurally cannot** produce this number: the MODE-B
window dumps carry `pred_dense`/`gt_dense` only out to **K=20 (2.0 s)** — the blind
horizon the co-primary exists to correct. `gate_emitters.corridor_from_windows`
refuses honestly rather than emitting a short-horizon number:

> "the open-loop surface caps at K=20 (2.0 s) — the blind horizon. A K≥100
> co-primary requires a **CLOSED-LOOP rollout** (E1a's surface, `e1a_horizon.py`),
> which needs GPU."

The existing closed-loop assets do not accept a v4 checkpoint as-is:
`taniteval/closedloop.py` parameterises `k` but its `run_and_save` loads arms via
`taniteval.registry` and requires `traj_capable` + `model.tactical_policy`, which v4
does not have (v4 uses `FlagshipV4Head` with dense-20 factorised LAT × LON × DIST
selection); `e1a_horizon.py`, which produced the card's reference numbers, is
REF-C-specific (`RefCModel`, `--refc-ckpt`).

**Reference points already MEASURED on REF-C base** (card, from E1a, `K=185`):
overall **0.5877 [0.5107, 0.6622]** · junction **0.8414 [0.8144, 0.8667]** ·
peak XTE **38.94 m**. For scale, the same 43 windows give corridor departure
**0.0035 at K=20 vs 0.5877 at K=185** — the 2 s instrument hid the failure by ~168×,
while the paired ADE@2s delta over those windows (0.0109 [−0.0, 0.0312]) is **not
separated**. That is the whole reason `ade_0_2s` was demoted.

**Note on structural feasibility:** PhysicalAI clips are 190–199 frames and window
starts are `range(0, T − W − K, stride)`, so K=185 sits near the structural cap of
190. The surviving window count must be reported with any number produced at this
horizon — a corridor number without its K **and its n** is not admissible.

<!-- CO-PRIMARY RESULT: filled in below when the commissioned measurement lands. -->

---

## 7. GOAL PROVENANCE — the oracle/produced PAIR, both measured at 30 k

The card **forbids transplanting the 15 k gap (+0.1738)** because the goal head kept
training. It was therefore re-measured on the same checkpoint. **That mattered:**

| step | oracle | produced | gap (produced − oracle) |
|---|---|---|---|
| 15 k *(MEASURED, `2026-07-26-v4-produced-goal/results/`)* | 0.5839 [0.4962, 0.6821] | 0.7577 [0.6621, 0.8692] | **+0.1738** |
| **30 k *(MEASURED, this gate)*** | **0.6423 [0.5348, 0.7586]** | **0.8563 [0.7282, 1.0035]** | **+0.2140 [+0.1602, +0.2759]** |

paired_episode_cluster_bootstrap, 881 windows / 40 episodes, n_boot 2000,
`p_delta_gt0 = 1.0`, **SEPARATED**.

**The goal-oracle privilege GREW, from +0.1738 to +0.2140.** Transplanting the 15 k
number would have understated the privilege by ~0.04 m and overstated the deployable
path. Training the goal head for another 15 k steps did not close the gap — it widened
it. (15 k `neutral` control, for completeness: 0.6565 [0.5553, 0.7749].)

**Read:** oracle 0.6423 beats the CV floor separated; **produced 0.8563 does not beat
CV at all**. On the deployable surface this checkpoint is not distinguishable from
constant velocity at 2 s.

---

## 8. LATERAL / LONGITUDINAL DECOMPOSITION (card `required_reporting[0]`)

`taniteval/lateral.py`, sparse 4-waypoint surface, `horizon_s = 2.0` (provenance
stamped), estimator `episode_cluster_bootstrap`, 881 windows / 40 episodes.
Cross-track is the safety-relevant axis; an undecomposed L2 hides it.

| | v1 (0.4271) | v4-30k ORACLE | v4-30k PRODUCED |
|---|---|---|---|
| longitudinal share of squared error | 87.3 % | **71.0 %** | 80.9 % |
| lateral share | 12.7 % | **29.0 %** | 19.1 % |
| peak \|XTE\| mean | **0.2857 [0.2290, 0.3479]** | **0.5349 [0.3703, 0.7475]** | 0.5616 [0.3935, 0.7824] |
| peak \|XTE\| p90 | **0.7119 [0.5046, 0.8000]** | **1.1916 [0.8452, 1.8996]** | 1.2235 [0.9091, 2.1452] |
| **windows beyond the 1.75 m lane half-width** | **0.8 %** | **6.1 %** | **6.9 %** |
| lateral growth 0.1→2.0 s | ×14.95 | ×13.68 | ×13.69 |
| longitudinal growth | ×12.61 | ×10.38 | ×10.53 |
| cross grows faster by | ×1.186 | ×1.318 | ×1.300 |

**Paired cross-track, v1 vs v4-oracle, identical windows:**
**+0.2509 [+0.1021, +0.4415]**, `paired_episode_cluster_bootstrap`, `p_delta_gt0 = 1.0`,
**SEPARATED**, `horizon_s 2.0`, `horizon_provenance: explicit` — **v1 has the smaller
lateral error**.

**This is the most safety-relevant finding in the gate.** v4's regression against v1
is **disproportionately lateral**: the lateral share of squared error more than
doubles (12.7 % → 29.0 %) and the fraction of windows leaving a 1.75 m corridor at
just 2 s rises **~7.6×**, from 0.8 % to 6.1 %. That is the axis the co-primary was
introduced to measure, and it is already degrading at the *blind* horizon — before
the 18.5 s horizon where REF-C base showed 0.5877.

---

## 9. HOW IT COMPARES

### 9.1 vs v1 (`0.4271`) — CI-separated BEHIND, on both surfaces
Window alignment was **proven from the ground truth**, not assumed: the v1 and v4
dumps encode `eid` differently, so an `eid`-equality test falsely reports "different
windows". `max|gt_v4 − gt_v1| = 0.0` exactly → the same 881 windows in the same order,
so the paired read is admissible.

| comparison | paired Δ | CI | separated |
|---|---|---|---|
| v4-**oracle** − v1 | **+0.2152** | [+0.0972, +0.3340] | **YES**, `p_delta_gt0 = 1.0` |
| v4-**produced** − v1 | **+0.4292** | [+0.2865, +0.5820] | **YES**, `p_delta_gt0 = 1.0` |

v1 = 0.4271090 on these windows. **v4 at 30 k is CI-separated behind v1 — by +0.22 m
even with the goal oracle helping it, and by +0.43 m on the deployable path.**

### 9.2 vs its own 15 k value — it did not improve; it drifted the wrong way

| | 15 k | 30 k | change |
|---|---|---|---|
| oracle `ade_0_2s` | 0.5839 [0.4962, 0.6821] | 0.6423 [0.5348, 0.7586] | **+0.0584** |
| produced `ade_0_2s` | 0.7577 [0.6621, 0.8692] | 0.8563 [0.7282, 1.0035] | **+0.0986** |
| paired Δ vs v1 (oracle) | +0.1568 [+0.0630, +0.2504] | **+0.2152 [+0.0972, +0.3340]** | gap widened |

**Stated honestly: the 15 k→30 k change is NOT tested for separation.** The 15 k and
30 k reads are two independent evaluations, and no paired bootstrap was run across
them (the 15 k window dump exists in the repo but pairing across checkpoints was not
part of this card). The CIs overlap heavily. What is admissible: the point estimate
moved the **wrong way on both surfaces**, the gap to v1 **widened** rather than closed,
and **the second half of a 59-hour run bought no measured improvement on the gate
metric**. That is a claim about the point estimates and the direction of travel, not a
separated result.

⚠️ **The error not made.** The trainer's in-loop `ade@2s` reads **0.5063**, which is
*under* the 0.6 bar and would have flipped the primary to PASS. It is a **different
metric**: dense-20 (mean over 20 dense steps 0.1–2.0 s), not the 4-waypoint
convention every other MODEL_REGISTRY row is quoted in. This harness reproduces the
dense-20 statistic at **0.5027** on the same forward pass whose 4-waypoint value is
**0.6423** — a 0.14 m gap between two numbers with the same name. A metric NAME is not
a metric DEFINITION (RETRACTION_LOG C1). Only `eval_*.py` output on the 4-waypoint
surface is quoted above.

---

## 10. WHAT THE NEXT GATE NEEDS

1. **Register a corridor threshold** from this gate's co-primary measurement (§6.3):
   v4-30k reads **overall 0.6388 [0.5565, 0.7128]**, **junction 0.8432 [0.7874,
   0.8919]**, peak XTE 33.45 m, on **n = 41 windows / 40 episodes**. Set the bar from
   the **interval**, not the point estimate — and resolve §6.4 first: 90 % of windows
   leave the validated OOD envelope at K=185, so either register at a horizon where
   the envelope holds or re-validate P1 out to 18.5 s (and on v4, not on v1).
2. **Repair `run_gate.py`** — all three defects in §3. A gate renderer that cannot
   load its own registered card, cannot express `REPORT_ONLY_THIS_GATE`, and refuses
   a completed run over a 0-index off-by-one is not fit to adjudicate GPU-days.
   *(Do not repair it and re-render this gate — the verdict above stands on the card
   as written.)*
3. **Build the two missing emitters** — `speed_benefit_recovered_frac` (P8) and
   `deploy_tick_p99_ms` (`efficiency.py` latency panel). Two of seven pre-registered
   kill secondaries currently have no emitter anywhere in the codebase, which is why
   this gate can only render `INCOMPLETE`.
4. **Fix the route-label identity** (`--labels-v21 --v2-route-from-vision`) to re-arm
   `nonav_route_beats_majority` (§5).
5. **The WM canary is the lever to interrogate first** — 1.1409 vs a 0.55 bar, against
   v1's 0.4271 on the same rollout. The joint planner–WM coupling that is the v4 thesis
   is also what degraded the world model.

---

## 11. DELIVERABLE MANIFEST

**Repo** — `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-26-v4-30k-gate/`
(**not** `git add`ed, per the brief):

| file | what |
|---|---|
| `GATE_30K_RESULTS.md` | this report |
| `raw/flagship-v4-fromscratch-30k-oracle.json` | MODE-B gate eval, goal ORACLE |
| `raw/flagship-v4-fromscratch-30k-produced.json` | MODE-B gate eval, goal PRODUCED |
| `raw/*_v4_diagnostics.json` (×2) | kill-secondary panel incl. the void secondary |
| `raw/driving_*.json` (×2) | `taniteval.driving` bootstrap panels + floors |
| `raw/gate30k_paired.json` | all paired deltas + GT-proven window alignment |
| `raw/gate30k_analysis.json` | lat/lon decomposition, all three arms |
| `raw/preflight_1_2.json` | preflight 1 + 2 |
| `raw/v1_preflight_flagship30k.json` | preflight 3 (harness validation) |
| `raw/GATE_30K_verdict_A_no_coprimary.json`, `raw/GATE_30K_verdict_B_coprimary_registered.json` | machine output of both projections — **NOT the verdict** (§3) |
| `raw/v4_hf_push_receipt.json` | HF push receipt, gate + hashes |
| `raw/pull_receipt.json` | HF → eval-pod round-trip md5 receipt |
| `raw/flagship-v4-30k.card.json` | the binding card, as registered |
| `coprimary/corridor_v4_30k_K185.json` | **the co-primary**: closed-loop CDR @ K=185, all strata + OOD |
| `coprimary/v4_corridor_cl.py` | the v4-aware closed-loop corridor driver (new) |
| `coprimary/pair_arms.py`, `coprimary/refc_perwindow.py`, `coprimary/fix_ood_verdict.py` | paired-arm + OOD-verdict tooling |
| `scripts/*.py`, `scripts/*.sh` | every driver used: push, pull, preflight, gate run, paired analysis, gate render |

**HuggingFace (gated, manual):** `https://huggingface.co/Sayood/flagship-v4-fromscratch`
— `ckpt.pt`, `ckpt_step20000.pt`, `config.json`, `metrics.json`, `train_log.jsonl`, `README.md`.

**pod2** (`tanitad-pod2`): source of truth untouched at
`/workspace/experiments/flagship-v4-fromscratch/ckpt.pt`; receipt at
`/workspace/tmp/v4_hf_push_receipt.json`; pusher at `/workspace/tmp/push_v4_fromscratch.py`.

**eval pod** (`tanitad-eval`): working dir `/workspace/_v4gate/` (note: `/` is 99 %
full, 2.9 GB free — everything was written to `/workspace`, whose headroom was
verified with a real 4 GB `dd`, never with `df`). Verified checkpoint copy at
`/workspace/_v4gate/flagship-v4-fromscratch-30k/ckpt.pt`; window dumps and result
JSONs under `/workspace/_v4gate/results/`; drivers `/root/preflight_v4gate.py`,
`/root/run_gate30k.sh`, `/root/analyze_gate30k.py`, `/root/pair_v1.py`,
`/root/render_gate.py`, `/root/pull_v4_30k.py`.

**Escalation (integration, not a doc request):** the three `run_gate.py` defects in
§3 block *every* future card of this shape, not just this one. They need an owner.

---

### Reproduction

```bash
# eval pod; OMP_NUM_THREADS=8 (a 9x slowdown was measured from BLAS oversubscription)
export OMP_NUM_THREADS=8
export PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts:/root/taniteval
cd /root/TanitAD/stack/scripts
python3 eval_flagship_v4.py \
  --ckpt /workspace/_v4gate/flagship-v4-fromscratch-30k/ckpt.pt \
  --anchors-dense /root/models/flagship-v4-fromscratch-15k/flagship_v4_anchors_dense.pt \
  --val-cache /root/valdata/physicalai-val-0c5f7dac3b11 \
  --goal-mode oracle   --key flagship-v4-fromscratch-30k-oracle \
  --results-dir /workspace/_v4gate/results \
  --out /workspace/_v4gate/gate30k_oracle.json
# ... then --goal-mode produced with the matching --key
```
