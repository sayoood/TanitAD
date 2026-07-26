# The horizon, measured against the envelope — and a guardrail that is arithmetically incapable of firing

**Date:** 2026-07-26 (Europe/Berlin; pods log UTC) · **Host:** `tanitad-pod2` (A40) · **Stream:** Benchmarks & Eval
**Closes:** `POD2_EVAL_HOST.md` escalations **§9.8** (T1, primary) · **§9.6** (T2) · **§9.9** (T3)
**Constraints honoured:** pod1 (TRAINING v2corpus) never contacted · the 600-episode cache read-only ·
`OMP_NUM_THREADS=8` · **one job at a time on pod2** · staged, never committed, never pushed.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED` (another
agent/doc, **not** re-verified) · `ESTIMATED` · `HYPOTHESIS`.

---

## 1. PRE-REGISTRATION — written and staged BEFORE the K-sweep produced a number

*This section was written while the sweep was still executing its first horizon. Both outcomes are
committed here in advance, per the standing rule that a discriminating experiment names its two
answers before it has one.*

### 1.1 The question

`POD2_EVAL_HOST.md` §4.5 recommends registering the next gate's closed-loop co-primary at
**K = 60 (6.0 s) primary · K = 70 (7.0 s) hard maximum · K = 185 report-only-pooled**. That
recommendation is `MEASURED` **on stratum yield** — the junction stratum crosses the 200-cluster
two-arm bar between K = 70 (204 clusters) and K = 75 (196). Its **envelope clause is not**:

> *"A 6 s rollout accumulates far less lateral drift than an 18.5 s one, so K = 60 is likely to sit
> inside the envelope — **but that has NOT been measured, and it must be**."* (§4.5, labelled
> `HYPOTHESIS` by its own author.)

`GATE_30K_RESULTS.md` §10.1's binding instruction is *"either register at a horizon where the envelope
holds or re-validate P1 out to 18.5 s"*. **Nobody has checked where the envelope holds.** This run does.

### 1.2 The two outcomes, committed in advance

| | outcome | what we will write |
|---|---|---|
| **A** | the envelope **holds** for K ≤ 70 | the K = 60 / K = 70 recommendation is confirmed **on both grounds** — stratum yield AND envelope — and is registerable as written. |
| **B** | the envelope **breaks below K = 70** | the horizon recommendation is **TOO OPTIMISTIC**. The primary K drops to where the envelope actually holds, stated plainly. **No rescue** — not by re-defining the envelope, not by moving to a "PARTIAL" reading, not by arguing the affected windows are few. |

**Pre-committed decision rule** (E1a's rule as implemented in `taniteval/ood.py`, the FULL disjunction —
`e1a_horizon.py:28-30`): a horizon is **EXTRAPOLATION, not measurement**, if the peak OOD ratio exceeds
~1.5× **OR** the rollout's steps leave the MEASURED P1 envelope (`|dlat| ≤ 3.0 m`, `|dyaw| ≤ 12°`).
`ood.verdict()` returns `MEASUREMENT` only when **zero** steps are outside; `PARTIAL EXTRAPOLATION`
when a minority are; `EXTRAPOLATION` when a majority are. `assert_envelope_verdict_consistent()` raises
rather than warns.

**The threshold we will read the answer at, fixed now:** the largest K whose overall stratum still
returns `MEASUREMENT` is the envelope-honest horizon. If **no** K in the sweep returns `MEASUREMENT`,
we say so — that is a third possibility and it is not outcome A.

### 1.3 What is NOT blind, disclosed

Two of the seven sweep points were already public before this run and I had read them:
`K = 20 → 12.3 %` of windows out of envelope and `K = 185 → 90.2 %`, from the committed gate log
(`…/2026-07-26-v4-30k-gate/coprimary/v4cl_oracle.log`, `PUBLISHED`, same checkpoint). So the sweep's two
endpoints are not a surprise; **K ∈ {60, 70, 90, 120, 150} are genuinely unmeasured**, and the
K = 20 reading is already enough to make outcome A's plain form ("the envelope holds") *unlikely* —
which is precisely why the rule above is written in terms of `MEASUREMENT` / `PARTIAL` / `EXTRAPOLATION`
rather than a yes/no, and why B is the outcome I expect to be writing. Saying so in advance is cheaper
than being seen to discover it.

### 1.4 The design, fixed before the run

* **Arm:** `flagship-v4-fromscratch` @ step **29999**, ckpt md5 **`8771c1d9d3da696dcde2a745d628f6a8`** —
  **byte-identical to the checkpoint that produced the committed 30 k gate co-primary.** Chosen because
  the envelope question is about the arm that will be gated, and because it makes K = 185 and K = 20
  *reproduction checks* against committed numbers rather than new numbers.
* **Surface:** closed loop, real-footage-in-the-loop (`taniteval.clhorizon.corridor_rollout`, K free).
  Never pooled with the open-loop dense surface.
* **Deployment:** the canonical **40 episodes** (`ep_00000…ep_00039`), stride 8 — the deployment the
  co-primary is registered on. ⚠️ Stated as a limit, not hidden: the junction stratum has 22 clusters
  at K = 20 and 6 at K = 185 on this deployment. The 600-episode build would give 232 / 58 but the full
  sweep there is ~15× the compute (≈ 37 GPU-h) and was not affordable in one session.
* **Goal provenance:** `oracle` — matching the registered gate co-primary exactly. **Stamped ORACLE per
  `GATE_PROTOCOL.md` §0.8; no number here is a deployed-capability claim.**
* **Estimator:** `episode_cluster_bootstrap` (`taniteval/ci.py`), B = 2000, unit = **val episode**;
  **paired** form for every K-vs-K delta on the common-start subset. `overlapping_holdout_se` appears
  nowhere.
* **Priority order** (so a killed run still yields value): **185 → 20 → 60 → 70 → 90 → 120 → 150**, each
  banked to disk the moment it finishes.

---

## 2. Preflight — MANDATORY, and **two of the four checks FAILED and were fixed**

The brief's four preflights plus three of my own. Artifacts:
`artifacts/preflight_pod2_closeout.json`, `artifacts/preflight2_pod2.json`.

| # | check | result | class |
|:--:|---|---|---|
| **P0** | `sys.path` audit — every entry, every loaded module, every other tree on the box | ✅ **PASS.** 2 verified roots (`/root/taniteval`, `/root/TanitAD/stack`); **0 mismatch / 0 missing** against the repo manifest at *import time*; **0 of 46** loaded `taniteval.*`/`tanitad.*` modules resolved outside them. **8 other trees exist on the box** (incl. the stale `/workspace/TanitAD/stack` @ `0f93b98`); **none on `sys.path`.** ⚠️ I created two of those trees myself by renaming the old roots aside, and **deleted them before the sweep** rather than leave a fresh shadowing hazard. | `MEASURED` |
| **P1** | `corridor.py` present **on the executing host** | ✅ **PASS.** Imported *and exercised*: `cross_track_from_paths` → `corridor_block` → `stratified` on a synthetic 12-window/4-episode set. `CORRIDOR_HALFWIDTH_M 1.75` · `CORRIDOR_GRID_M (1.0, 1.75, 2.5)` · `JUNCTION_DEG 10.0` · `horizon_seconds(185) = 18.5`. | `MEASURED` |
| **P2** | `lateral.py` emits `horizon_s = 2.0` on the sparse 4-knot surface | ✅ **PASS.** `paired_cross_track(..., step=4)` → **`horizon_s = 2.0`**, `horizon_provenance = "inferred_from_knot_count"`, `n_knots = 4`; explicit `knot_dt=0.5` → 2.0, provenance `"explicit"`. **The stale `0.4 s` signature (the 5× mislabel) is ABSENT.** | `MEASURED` |
| **P3** | v1 reproduces `ade_0_2s = 0.4271` on the 40-episode deployment | ✅ **PASS, exactly.** `python3 -m taniteval.runner run --model flagship-30k --episodes 40`, 103.2 s, `ckpt_step 29999` → **`0.4271 [0.3675, 0.4871]`**, 881 windows / 40 clusters, `episode_cluster_bootstrap` B = 2000. CI bounds identical to the registry. | `MEASURED` |
| ⛔ **P4** | `clhorizon.py` + `ood.py` present on the executing host | 🔴 **FAILED — both were MISSING from pod2.** They landed in the repo *after* the standup's sync (its §2.6 drift). **Fixed:** whole-tree re-sync, bundle md5 `48b0534521da54aadac439bd7ba4b9e8` verified on both ends, then re-verified: `taniteval` **221/221 md5-identical to the repo, 0 mismatch**; `stack` **344/344, 0 mismatch** (18 repo-only files = the deliberate training-artifact exclusion). | `MEASURED` |
| ⭐ **P5** | the new guard **can actually fire** | ✅ **PASS.** Fed `ood.assert_envelope_verdict_consistent` the exact node the 30 k gate emitted (`MEASUREMENT` verdict beside `54.63 %` of steps outside) → **raises `EnvelopeVerdictError`**. `readjudicate` on the legacy string moves the class `MEASUREMENT → EXTRAPOLATION`. *(A guard whose own ability to fire is untested is the defect it was written to remove, in a new costume.)* | `MEASURED` |
| ⛔ **P6** | the harness **imports at all** on the executing host | 🔴 **FAILED — and this one is a live defect in a committed artifact.** See §2.1. **Fixed and re-verified.** | `MEASURED` |
| ⛔ **P7** | `clhorizon.run_v4` is runnable | 🔴 **FAILED — reproduced, not inferred.** See §2.2. **Worked around in the driver and reported; NOT patched by me** (see §2.2 for why). | `MEASURED` |

### 2.1 🔴 D1 — `eval_flagship_v4.py` uses **Python ≥ 3.12 syntax**, and pod2 runs **3.11.10**

```
File "/root/TanitAD/stack/scripts/eval_flagship_v4.py", line 478
    f"{' [FALLBACK=' + str(goal_rec.get('fallback')) + ']'
    ^
SyntaxError: unterminated string literal (detected at line 478)
```

A **multi-line expression inside an f-string replacement field** is **PEP 701**, i.e. Python 3.12+ only.
`python3 -m compileall` over the whole tree under **3.11.10** finds exactly two such files:

| file:line | construct | PEP 701 feature | consequence |
|---|---|---|---|
| `stack/scripts/eval_flagship_v4.py:478` | multi-line expression in an f-string | ≥ 3.12 | ⛔ **`import eval_flagship_v4` raises at import time**, so **every v4 eval path is un-runnable on pod2** — including `v4_corridor_cl.py` and `clhorizon.run_v4`, which both import `load_v4_from_ck` from it. **The registered closed-loop co-primary could not have been re-rendered on the designated n ≥ 200 eval host.** |
| `stack/scripts/vlm_kin_crossval.py:117` | backslash in an f-string expression | ≥ 3.12 | `import vlm_kin_crossval` raises on 3.11 |

Introduced **today**, in commit `87131fd` — whose own headline is *"the eval pod was 62 % stale and
MISSING `corridor.py`"*. **Root-cause class: C2** (absence found at ONE location is not absence) in its
interpreter-version form: *"it imports on the host I tested"* is a one-probe absence claim.
`taniteval/` itself compiles clean under 3.11.

**FIXED, behaviour-preserving, verified:** both suffixes are now built outside the f-string; the
produced strings were asserted equal for both branches, and `vlm_kin_crossval`'s header is written
`r"kin \\ vlm"` so its **two** output backslashes are reproduced byte-for-byte (a single `\` was probably
intended — this is a portability fix, not a cosmetic one). `compileall` under 3.11 is now clean and
`import eval_flagship_v4` succeeds on pod2. Staged, not committed.

### 2.2 🔴 D2 — `clhorizon.run_v4` cannot run: `RawEp` has no `.frames`

`taniteval/clhorizon.py` exists, in its own docstring, so *"the co-primary is not stranded behind a
driver in `incoming/`"*. Its entry point `run_v4` (line 509) does:

```python
eps = _data.load_frames(_data.list_val_episodes(val_dir, episodes))
pw  = corridor_rollout(planner, eps, goals, device, K, ...)   # frames_of = _default_frames
```

`_data.load_frames` wraps every episode in `RawEp`, which exposes the frames as **`.feats`**
(`taniteval/data.py:220`), while `clhorizon._default_frames` reads **`ep.frames`**. **REPRODUCED on
pod2:** `AttributeError: 'RawEp' object has no attribute 'frames'`. The committed gate driver used
`load_episode(...)` directly and is unaffected — which is why nobody noticed. The one-line fix is
`load_raw` instead of `load_frames`; **this run uses `_data.load_raw`**, which is also the
surface-matching choice, since it is what the committed driver used.

⚠️ **I did NOT patch `clhorizon.py`.** `test_clhorizon.py::test_port_is_tensor_identical_to_the_driver`
pins that module bit-identical to the driver that produced the committed co-primary, and it landed
hours ago from a sibling stream. Editing it during their run is exactly the "pushing a mid-edit tree"
hazard the standup refused. **Escalated in §8 as a one-line owner fix.**

---

## 3. ⭐⭐ The finding that arrived before the sweep did: the OOD ratio criterion has an **arithmetic ceiling**, and it sits **below every threshold that consumes it**

The brief asked whether the new envelope verdict *agrees* with what the two live `<= 1.30` constants
would have said. The answer is stronger than agree/disagree, and it needs **no model and no rollout**.

### 3.1 The supremum, computed from the envelope JSON alone

`OODMap.ratio_arr` is
`1 + clip((interp(|dlat|) − base)/base, 0, ∞) + clip((interp(|dψ|) − base)/base, 0, ∞)`,
and `np.interp` is piecewise linear through the P1 sweep points and **CLAMPS** outside them. So its
supremum over **all possible inputs** — including `|dlat| = 10⁹ m` — is a constant:

| quantity | value | source |
|---|---|---|
| P1 baseline ADE@2s | **0.4045** | `lowood_flagship_ci.json` |
| max lat ADE (at `|dlat| = 3.0 m`) | 0.4703 → excess **+0.16267** | same |
| max yaw ADE (at `|dψ| = 12°`) | 0.4596 → excess **+0.136218** | same |
| ⭐ **`sup(ratio_arr)`** | ⭐ **1.298888** | `artifacts/ood_blast_radius.json` |

> ### ⛔ **1.298888 < 1.30.** The guardrail `ood <= 1.30` is **TRUE for every possible input**. It is not
> "saturating", "conservative" or "a lower bound" — it is a **tautology**, by a margin of **0.001112**.
> And `RATIO_EXTRAPOLATION_X = 1.5` — clause 1 of E1a's own disjunction, inside the *fixed* module — is
> **unreachable by 0.201112**. On this envelope map **E1a's disjunction has only one live clause.**

MEASURED, on pod2's own interpreter as well: `ratio_arr(3 m) = ratio_arr(30 m) = ratio_arr(300 m) =
1.16267`, constant; both axes saturated → 1.298888 (`artifacts/preflight2_pod2.json` → `saturation_demo`).

### 3.2 The blast radius — every committed OOD node, re-adjudicated

`scripts/ood_blast_radius.py` re-adjudicates every OOD node in `TanitAD Research Hub/**/*.json` with
`taniteval.ood.readjudicate` (the packaged rule — no second implementation).

| | |
|---|---:|
| OOD nodes found and re-adjudicated | **181** |
| nodes carrying a ratio | **139** |
| ⭐ nodes where the old `<= 1.30` test says **IN BAND (pass)** | ⭐ **139 / 139** |
| nodes where it says **fail** | ⛔ **0** |
| ⭐ nodes whose verdict **CLASS FLIPS** under the real rule | ⭐ **14** |
| nodes sitting at **exactly** the supremum (ratio 1.2989) | **10** |

**All 14 flips are `MEASUREMENT → EXTRAPOLATION`** (12) **or `→ PARTIAL EXTRAPOLATION`** (2). None goes
the other way. Affected files, all in the two most decision-proximate artifacts in the program:

* `…/2026-07-26-v4-30k-gate/{raw,coprimary}/corridor_v4_30k_K185.json` — the **registered gate
  co-primary**, all four strata at K = 185, plus overall and junction at K = 20 in the paired block.
* `…/2026-07-26-v4-30k-gate/{raw,coprimary}/corridor_refcbase_30k_K185.json` — the REF-C reference arm.

⭐ **The single cleanest demonstration, and it is in the gate's own artifact:** the v4 **junction**
stratum at K = 185 records `ood_peak_ratio = 1.2989` — **the supremum, to four decimals** — beside
`frac_windows_any_step_out_of_envelope = 1.0`. **Every window is outside the envelope and the estimator
is pinned at its own ceiling**, and the string it emitted was *"within the measured envelope on average"*.

### 3.3 The two live constants, and what they actually decided

| constant | evaluations found | passes | failures | could it EVER have failed? |
|---|---:|---:|---:|:--:|
| `e1b_eval.py:403` `c_ood_in_band` | **1** (`e1b_eval_result.json`) | 1 | 0 | ⛔ **No** |
| `e1c_common.py:34` `OOD_BAND = 1.30` → `Gc_ood_in_band` | **19** (17 frontier checkpoints + 2 smoke) | 19 | 0 | ⛔ **No** |

⛔ **And in E1b it is the ONLY guardrail that "held".** `GUARDRAIL_SUMMARY` reads
`a_openloop_ade2s_ok: false · b_anchor_acc_ok: false · b_anchor_traj_l1_ok: false ·`
**`c_ood_in_band: true (c_ood_ft = 1.1339)`** → `all_ok: false` → verdict **BOUND**. The verdict is
unchanged (three real guardrails failed), **but the artifact publishes a passing OOD guardrail that
carried exactly zero bits.** In E1c, `Gc` is a *"must hold"* gate evaluated at **17 frontier
checkpoints**, and it held 17/17 for the same reason. Max ratio ever recorded there: **1.2919** — under
the ceiling, as it must be.

> ### ⭐ ADJUDICATION, per `GATE_PROTOCOL.md` §0.7
> `c_ood_in_band` / `Gc_ood_in_band` are **INSTRUMENT-FAIL (VOID)**: a metric whose value is fixed by a
> harness defect carries no information about the model and may not contribute to a kill conjunction —
> **nor to a pass one.** They must be printed as VOID, not silently dropped, because a suppressed
> criterion that is not printed is indistinguishable from one that passed. **The replacement is already
> written and needs no new science:** `taniteval.ood.verdict`'s **clause 2**, the out-of-envelope
> fractions, which are model-dependent, unbounded, and — as §4 shows — fire hard.


