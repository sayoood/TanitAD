# The horizon, measured against the envelope — and a guardrail that is arithmetically incapable of firing

**Date:** 2026-07-26 (Europe/Berlin; pods log UTC) · **Host:** `tanitad-pod2` (A40) · **Stream:** Benchmarks & Eval
**Closes:** `POD2_EVAL_HOST.md` escalations **§9.8** (T1, primary) · **§9.6** (T2) · **§9.9** (T3)
**Constraints honoured:** pod1 (TRAINING v2corpus) never contacted · the 600-episode cache read-only ·
`OMP_NUM_THREADS=8` · **one job at a time on pod2** · staged, never committed, never pushed.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED` (another
agent/doc, **not** re-verified) · `ESTIMATED` · `HYPOTHESIS`.

---

## 0. Headline

| # | Result | Class |
|:--:|---|---|
| **1** | ⭐⭐ **The OOD ratio criterion has an EXACT arithmetic ceiling — `sup(ratio_arr) = 1.298888` — computable from the envelope JSON with no model, no rollout and no GPU. `1.298888 < 1.30`, so the two live guardrails (`e1b_eval.py:403`, `e1c_common.py:34`) are TAUTOLOGIES by a margin of 0.001112**, and `RATIO_EXTRAPOLATION_X = 1.5` is unreachable by 0.201112. **Clause 1 of E1a's disjunction is dead; only clause 2 can ever fire.** | `MEASURED` |
| **2** | ⛔ **Corroborated repo-wide: 181 OOD nodes re-adjudicated, 139 carry a ratio, the old `≤ 1.30` test passes on 139/139 and fails on 0.** `Gc` was evaluated 20 times across E1b/E1c — 20 passes. **In E1b it is the ONLY guardrail that "held"** (the other three failed). **14 verdict classes FLIP**, all `MEASUREMENT → EXTRAPOLATION`, in the gate's own co-primary artifact and REF-C's. | `MEASURED` |
| **3** | ⭐⭐ **OUTCOME B, and worse than pre-registered. The envelope does not break "below K = 70" — it breaks at 0.5 s.** Last pure-MEASUREMENT horizon: **k = 4 (0.4 s)**, on 881 windows / 40 clusters. `GATE_PROTOCOL` §0.3 refuses K ≤ 20, so **NO admissible gate horizon is a measurement rather than an extrapolation — including the K = 20 the program already uses.** | `MEASURED` |
| **4** | ⭐⭐ **The envelope failure and HP-2's stratum of interest are THE SAME WINDOWS.** At K = 20: `other` **0.0000** outside, `longitudinal` 0.0027, **`junction` 0.5879** — and it leaves via the **heading** axis (0.5824 vs 0.1813 lateral), with the *median* junction window peaking at **14.95°** against a 12° validated edge. The co-primary is a clean measurement exactly where it is uninformative and an extrapolation exactly where it carries the signal. | `MEASURED` |
| **5** | ⭐ **Therefore the envelope is NOT a horizon problem — it is a RENDERER-VALIDATION problem.** `K = 60` stays the register (stratum yield is untouched and still binds at K ≤ 70), **but on ONE ground, not two**, and the co-primary must be stamped `EXTRAPOLATION` with its fractions printed. **`POD2_EVAL_HOST.md` headline #10 stamped `MEASURED` on a compound whose envelope half its own §4.5 labels `HYPOTHESIS`** — retraction appended, class **C4**. | `MEASURED` |
| **6** | ⭐ **Nine independent reproductions of committed numbers before any new number was quoted** — v1 = 0.4271 [0.3675, 0.4871], and **all 8 strata × 2 horizons** of the registered gate co-primary, bit-identical, on a freshly re-synced host and through the **packaged** `taniteval.clhorizon` rather than the `incoming/` driver. | `MEASURED` |
| **7** | 🔴 **Two preflights FAILED and are fixed. `stack/scripts/eval_flagship_v4.py:478` uses PEP 701 (Python ≥ 3.12) syntax and pod2 runs 3.11.10** — so **every v4 eval path, including the registered co-primary, was un-runnable on the designated n ≥ 200 eval host.** Introduced today, in the commit whose headline is *"the eval pod was 62 % stale"*. Class **C2** (interpreter-version form). Fixed, behaviour-preserving, 70 tests pass. | `MEASURED` |
| **8** | 🔴 **`taniteval/clhorizon.py::run_v4` — the entry point written to un-strand the co-primary — raises on its first step** (`RawEp` has no `.frames`). REPRODUCED. **Not patched** (sibling stream owns it); escalated as a one-line fix. Class **C2**. | `MEASURED` |
| **9** | ✅ **T2 done: `MODEL_REGISTRY.md` §1.2a — v1 @ 600 = `0.4108 [0.3956, 0.4273]`, 13,198 windows / 600 clusters, as a NEW row** with both n's, the estimator, the CV floor of its own deployment (**0.6917 vs 0.8377** — the 600 is EASIER) and a binding no-substitution rule. §5 leaderboard stamped as the 40-episode deployment. `registry_lint` PASS. | `MEASURED` |
| **11** | ⭐⭐ **T3 done, and the `HYPOTHESIS` is now MEASURED: the S3 bars move with `OMP_NUM_THREADS`.** The fit is **BITWISE deterministic** across separate processes at a fixed thread count and seed (**5/5, every digit**); changing only the thread count on one host moves the bar by up to **0.0137** — the full magnitude of the cross-host discrepancy. ⛔ **But the SEED moves it MORE (spread 0.0202 / 0.0270), so pinning threads is necessary and not sufficient.** ⇒ **The bars are quotable to ONE decimal place: `S3 lat 0.66 · lon 0.54 · S3-W lat 0.26 · lon 0.30`**, honest form `mean ± sd over 10 seeds` = **0.6557 ± 0.0068 · 0.5350 ± 0.0097 · 0.2578 ± 0.0038 · 0.3021 ± 0.0075`. **A 4-dp bar may not sit in a kill conjunction; an arm within `bar ± spread` is a TIE.** All four published pod2 bars reproduced exactly at the reference condition. | `MEASURED` |
| **10** | ⚠️ **The one branch `GATE_30K_RESULTS` §10.1 leaves open is "re-validate P1", and this run prices it:** covering K = 60 needs the homography validated to **13.55 m / 39.25°** overall and **20.82 m / 60.03°** on the junction stratum — **4.5×–6.9× / 3.3×–5.0×** the current edge, from a `f_eff = 266 px` camera at 1.5 m. ⭐ **And it is already owed at K = 20**, whose p90 yaw is **1.27× outside** while its lateral p90 is comfortably inside — so the extension must be on the **YAW** arm first. It is also the **ONLY** action that can raise the ratio criterion's ceiling. | `MEASURED` |

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

---

## 4. T1 — the K-sweep, MEASURED

`flagship-v4-fromscratch` @ 29999 (`ckpt_md5 8771c1d9d3da696dcde2a745d628f6a8`), closed loop,
40 episodes, stride 8, goal-mode **ORACLE**, `episode_cluster_bootstrap` B = 2000.
Artifacts: `artifacts/ksweep_results.json`, `artifacts/perwindow_K*.pt`, log `artifacts/ksweep.log`.

> ⚠️ **SCOPE NARROWED, stated plainly rather than quietly.** The brief asked for
> **K ∈ {20, 60, 70, 90, 120, 150, 185}**. I ran **{185, 20, 60, 70}** — the priority order the brief
> itself set — and **stopped the sweep by explicit PID after K = 70**, because pod2 runs one job at a
> time and T3 is a deliverable. **K = 90 / 120 / 150 are NOT measured here.** What is lost is bounded
> and I am not pretending otherwise: those three K sit **above the HP-2 ceiling of K = 70**, so no gate
> can register there; the envelope verdict is already `EXTRAPOLATION` at K = 60 and monotone in K; and
> the K = 185 **truncation curve** covers 9.0 / 12.0 / 15.0 s indicatively (`ksweep_results.json` →
> `results.185.truncation_curve`, and §4.4 measures how far a truncation may be trusted). The rollouts
> would cost ≈ 82 GPU-min and can be resumed with the same command and `--horizons 90,120,150`.

### 4.1 Two harness checks first — both reproduce the committed gate artifact EXACTLY

⭐ **All 8 strata × 2 horizons reproduce bit-identically** against
`…/2026-07-26-v4-30k-gate/coprimary/corridor_v4_30k_K185.json` — same checkpoint md5, different host,
freshly re-synced tree, and via the **packaged** `taniteval.clhorizon`/`taniteval.corridor` rather than
the `incoming/` driver that produced the original:

| K | stratum | n win / n ep | this run, CDR@1.75 | committed gate artifact |
|---:|---|---:|---|---|
| 185 | overall | 41 / 40 | **0.6388 [0.5565, 0.7128]** | 0.6388 [0.5565, 0.7128] ✅ |
| 185 | **junction** | **6 / 6** | **0.8432 [0.7874, 0.8919]** | 0.8432 [0.7874, 0.8919] ✅ |
| 185 | longitudinal | 18 / 18 | **0.6871 [0.6138, 0.7496]** | 0.6871 [0.6138, 0.7496] ✅ |
| 185 | other | 17 / 16 | **0.5154 [0.3807, 0.6520]** | 0.5154 [0.3807, 0.652] ✅ |
| 20 | overall | 881 / 40 | **0.0203 [0.0078, 0.0364]** | 0.0203 [0.0078, 0.0364] ✅ |
| 20 | **junction** | **182 / 22** | **0.0909 [0.0421, 0.1413]** | 0.0909 [0.0421, 0.1413] ✅ |
| 20 | longitudinal | 374 / 24 | **0.0035 [0.0010, 0.0068]** | 0.0035 [0.001, 0.0068] ✅ |
| 20 | other | 325 / 24 | **0.0000 [0.0, 0.0]** | 0.0 [0.0, 0.0] ✅ |

Also identical: peak XTE (33.452 m / 0.630 m), OOD peak ratio (1.2741 / 1.0504), and
`frac_steps_lat_over_3m` / `frac_windows_out` = **0.54634 / 0.9024** — i.e. the *54.63 % / 90.24 %*
quoted in `GATE_PROTOCOL.md` §0.1. Together with **P3** (v1 = 0.4271 [0.3675, 0.4871]) that is
**nine independent reproductions** of committed numbers before any new number was quoted.

⚠️ **And it confirms the standup's escalation #3 in passing:** the gate report's headline quotes the
junction co-primary `0.8432 [0.7874, 0.8919]` beside the *overall* n (`41 win / 40 ep`). It rests on
**6 windows / 6 episodes.** The fix is one column: `41/40 overall · 6/6 junction`.

### 4.2 ⭐ THE SWEEP TABLE — yield × envelope × departure

`yield` columns are `MEASURED` here on the **40-episode** deployment; the **600-episode** cluster yields
are `PUBLISHED` (`POD2_EVAL_HOST.md` §3.1) and are what HP-2's 200-cluster bar is read against.
Envelope columns are the `taniteval.ood` disjunction: **clause 1 (ratio > 1.5) is VOID at every row**
(§3), so **every verdict below is carried by clause 2 alone.**

**Overall stratum** — the co-primary as a gate would read it:

| K | s | win / **clusters** (40 eps) | **clusters @600** *(PUBLISHED)* | `CDR@1.75` [cluster-bootstrap CI95] | peak XTE m | **steps out of envelope** | **windows out of envelope** | OOD ratio *(ceiling 1.298888)* | c1 | c2 | **verdict** |
|---:|---:|---:|---:|---|---:|---:|---:|---:|:--:|:--:|---|
| **20** | 2.0 | 881 / **40** | 600 | **0.0203** [0.0078, 0.0364] | 0.630 | 0.0531 | **0.1226** | 1.0504 | ⛔ VOID | 🔥 | PARTIAL |
| **60** | 6.0 | 681 / **40** | 600 | **0.2618** [0.2025, 0.3243] | 5.351 | 0.2291 | **0.5066** | 1.1693 | ⛔ VOID | 🔥 | ⛔ **EXTRAP** |
| **70** | 7.0 | 638 / **40** | 600 | **0.3195** [0.2549, 0.3839] | 6.961 | 0.2759 | **0.5987** | 1.1898 | ⛔ VOID | 🔥 | ⛔ **EXTRAP** |
| **185** | 18.5 | 41 / **40** | 596 | **0.6388** [0.5565, 0.7128] | 33.452 | 0.5900 | **0.9024** | 1.2741 | ⛔ VOID | 🔥 | ⛔ **EXTRAP** |

**Junction stratum** — reported separately, always (`GATE_PROTOCOL` §0.4). ⚠️ n is the binding number here, not the value:

| K | win / **clusters** (40 eps) | **clusters @600** *(PUBLISHED, HP-2 bar = 200)* | `CDR@1.75` [CI95] | **windows out of envelope** | **verdict** |
|---:|---:|---:|---|---:|---|
| **20** | 182 / **22** | **232** ✅ | **0.0909** [0.0421, 0.1413] | **0.5879** | ⛔ **EXTRAP** |
| **60** | 136 / **19** | **207** ✅ | **0.5809** [0.5289, 0.6268] | **0.9485** | ⛔ **EXTRAP** |
| **70** | 125 / **19** | **204** ✅ | **0.6330** [0.5838, 0.6743] | **0.9600** | ⛔ **EXTRAP** |
| **185** | 6 / **6** | ⛔ **58** | **0.8432** [0.7874, 0.8919] | **1.0000** | ⛔ **EXTRAP** |

**longitudinal / other** — the non-junction strata, for contrast:

| K | longitudinal: CDR · winOUT · verdict | other: CDR · winOUT · verdict |
|---:|---|---|
| **20** | 0.0035 · 0.0027 · PARTIAL *(n 374/24)* | 0.0000 · 0.0000 · ✅ MEASURE *(n 325/24)* |
| **60** | 0.2437 · 0.4897 · PARTIAL *(n 290/23)* | 0.1122 · 0.2902 · PARTIAL *(n 255/22)* |
| **70** | 0.3121 · 0.5956 · ⛔ **EXTRAP** *(n 272/23)* | 0.1653 · 0.4149 · PARTIAL *(n 241/22)* |
| **185** | 0.6871 · 0.9444 · ⛔ **EXTRAP** *(n 18/18)* | 0.5154 · 0.8235 · ⛔ **EXTRAP** *(n 17/16)* |

*Rendered from `artifacts/ksweep_results.json` by `scripts/render_sweep_table.py` — no number in this table was transcribed by hand. `sup(ood_peak_ratio) = 1.298888`, so column `c1` is VOID at every row.*

### 4.3 How far outside the envelope the loop actually goes

The fraction says *whether*; this says *how far*. Peak deviation per window, **each from its own real
rollout** (not truncated), against the P1 validated edges **3.0 m / 12°**:

| K | s | n win | p50 peak &#124;dlat&#124; | **p90** | max | p50 peak &#124;dψ&#124; | **p90** | max | p90 as a multiple of the edge |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20 | 2.0 | 881 | 0.28 m | 1.74 m | 5.19 m | 1.73° | **15.19°** | 43.57° | lat 0.58× · ⛔ **yaw 1.27×** |
| **60** | **6.0** | 681 | **2.79 m** | ⛔ **13.55 m** | 50.92 m | 6.63° | ⛔ **39.25°** | 80.95° | ⛔ **lat 4.52× · yaw 3.27×** |
| **70** | **7.0** | 638 | ⛔ **3.94 m — the MEDIAN window is outside** | ⛔ **17.16 m** | 68.72 m | 8.08° | ⛔ **43.32°** | 88.67° | ⛔ **lat 5.72× · yaw 3.61×** |
| 185 | 18.5 | 41 | 22.42 m | ⛔ **72.24 m** | **138.91 m** | 27.24° | 77.67° | 135.19° | ⛔ **lat 24.08× · yaw 6.47×** |

**Junction stratum alone**, p90 peak deviation (this is the population a re-validated P1 would have to
cover for HP-2 to be measurable):

| K | n junction win | p90 &#124;dlat&#124; | × edge | p90 &#124;dψ&#124; | × edge |
|---:|---:|---:|---|---:|---|
| 20 | 182 | 3.51 m | ⛔ 1.17× | 28.14° | ⛔ **2.35×** |
| **60** | 136 | **20.82 m** | ⛔ **6.94×** | **60.03°** | ⛔ **5.00×** |
| 70 | 125 | 25.79 m | ⛔ 8.60× | 61.52° | ⛔ 5.13× |
| 185 | 6 | 58.80 m | ⛔ 19.60× | 71.99° | ⛔ 6.00× |

> ⚠️ **This is the number that decides whether "re-validate P1" is cheap. It is not.** Covering K = 60
> to the p90 means validating the ground-plane homography to **13.6 m lateral / 39° yaw** overall, and
> to **20.8 m / 60°** on the junction stratum — from a `f_eff = 266 px` front-wide camera at
> `h = 1.5 m`. `clhorizon.sampling_homography`'s own docstring says the warp *"degrades gracefully
> rather than faithfully"* beyond the measured range. ⭐ **Note also the axis switch:** at K = 20 the
> lateral p90 is comfortably INSIDE (0.58×) and only the **yaw** axis is out (1.27×); by K = 60 lateral
> is the worse of the two (4.52× vs 3.27×). *A validation that extended only the lateral arm of the P1
> sweep would not fix the 2 s instrument.*

### 4.4 ⚠️ Method note, measured rather than assumed: **truncating a long rollout is NOT a short rollout**

Each K is its own experiment. `corridor_rollout`'s nearest-reference search runs over the **K + 1**
future poses, so the same window at the same `t0` is fed a different frame at different K and the
trajectories diverge. MEASURED on the 41 windows the K = 185 and K = 20 runs share:

| | max &#124;Δ&#124; over the first 20 steps | mean &#124;Δ&#124; |
|---|---:|---:|
| lateral deviation | **0.149 m** | 0.00085 m |
| heading deviation | **9.24°** | 0.056° |

Consequence, on those same 41 windows: **`CDR@1.75` is identical** (0.01463 vs 0.01463) — the corridor
metric is unaffected — but the **envelope fraction moves 0.0732 → 0.0488**, because a 9° yaw excursion
straddles the 12° edge. **So: the truncation curves in `ksweep_results.json` are quotable for the
lateral/CDR family and INDICATIVE ONLY for the yaw-driven envelope fraction.** Every envelope number in
§4.2 is from a real rollout at that K, never a truncation.

### 4.2b Cross-reference: a sibling stream measured the half-width itself, and it is ~26 % too permissive

Landed while this ran (`…/2026-07-26-vectormap-corridor/VECTORMAP_CORRIDOR.md`, `INHERITED` — I did not
re-verify it): `CORRIDOR_HALFWIDTH_M = 1.75` is *vindicated as half a lane width* (measured
**1.802 m [1.686, 1.939]**) but is **~26 % too permissive as a DEPARTURE threshold** — the median
*realised clearance* is **1.391 m**, and **85.7 % of steps have less room than 1.75 m**.

**Consequence for every number above: the CDR column is an UNDER-estimate.** The full grid is emitted,
so the sensitivity is already measured rather than argued:

| K | overall @1.0 m | @1.75 m *(headline)* | @2.5 m | junction @1.0 m | @1.75 m | @2.5 m |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.0530 | **0.0203** | 0.0087 | 0.2071 | **0.0909** | 0.0407 |
| 60 | 0.3628 | **0.2618** | 0.2046 | 0.6674 | **0.5809** | 0.5028 |
| 70 | 0.4214 | **0.3195** | 0.2565 | 0.7125 | **0.6330** | 0.5629 |
| 185 | 0.7048 | **0.6388** | 0.5872 | 0.8811 | **0.8432** | 0.8135 |

**No conclusion in this report is a knife-edge on the half-width** — the horizon effect and the
envelope verdict hold at all three thresholds, and a tighter (more realistic) corridor makes the
departure *worse*, not better. ⚠️ **The envelope finding is entirely independent of this**: the P1
envelope (3.0 m / 12°) is a property of the *renderer*, not of the corridor definition.

### 4.4b The common-start PAIRED contrast — the horizon effect on IDENTICAL windows

41 windows / 40 episodes shared by all four rollouts (the K = 185 start set is a subset of every
smaller K's), `paired_episode_cluster_bootstrap` B = 2000, oriented `CDR(K) − CDR(K=20)`.
Artifact: `artifacts/common_start_paired.json`. Computed offline from the per-window dumps — no GPU.

| K | overall Δ | **junction** Δ | longitudinal Δ | other Δ |
|---:|---|---|---|---|
| 60 | **+0.2309** [+0.1603, +0.3025] ✅ sep | **+0.4750** [+0.3361, +0.5861] ✅ sep | +0.2556 [+0.1519, +0.3602] ✅ sep | +0.1186 [+0.0386, +0.2177] ✅ sep |
| 70 | **+0.2920** [+0.2151, +0.3689] ✅ sep | **+0.5357** [+0.4107, +0.6393] ✅ sep | +0.3238 [+0.2127, +0.4310] ✅ sep | +0.1723 [+0.0786, +0.2839] ✅ sep |
| 185 | **+0.6241** [+0.5455, +0.6952] ✅ sep | **+0.7432** [+0.6259, +0.8270] ✅ sep | +0.6871 [+0.6138, +0.7496] ✅ sep | +0.5154 [+0.3807, +0.6520] ✅ sep |

⭐ **The K = 185 row is a tenth reproduction**: `0.0146 → 0.6388, Δ +0.6241 separated` is exactly the
figure `RETRACTION_LOG`'s C13 entry records as the v4 replication of the horizon finding.
✅ **The horizon finding is untouched by everything in §3 and §5** — the effect is monotone in K and
**CI-separated at every K and in every stratum**, on windows that are identical by construction.

### 4.5 ⭐ Where the envelope actually breaks — 0.5 s, and the whole failure is the junction stratum

On the **881-window / 40-cluster** K = 20 rollout — the largest window set in this run, and the one at
the program's *standing* horizon:

| elapsed | 0.4 s | **0.5 s** | 1.0 s | 1.5 s | **2.0 s** |
|---|---:|---:|---:|---:|---:|
| windows outside the P1 envelope | **0.0000** | ⛔ **0.0034** | 0.0443 | 0.0965 | **0.1226** |

> ### ⛔ **The last horizon at which the closed loop is a pure MEASUREMENT is k = 4 — 0.4 s.**
> The first window leaves the renderer's validated envelope at **0.5 s**, which is the controller's own
> lookahead (`LOOKAHEAD_STEP = 5`). **`GATE_PROTOCOL` §0.3 refuses any K ≤ 20, so there is no admissible
> gate horizon at which this surface is a measurement rather than an extrapolation** — not K = 60, not
> K = 70, and **not the K = 20 the program has been using all along.**

And it is one stratum, on one axis. At K = 20, by stratum (`junction = |Δheading| ≥ 10° / 2 s`):

| stratum | n win / n ep | out of envelope | via &#124;dlat&#124; > 3 m | via &#124;dψ&#124; > 12° | p50 peak &#124;dψ&#124; | verdict |
|---|---:|---:|---:|---:|---:|---|
| **junction** | 182 / **22** | ⛔ **0.5879** | 0.1813 | ⛔ **0.5824** | ⛔ **14.95°** | **EXTRAPOLATION** |
| longitudinal | 374 / 24 | 0.0027 | 0.0000 | 0.0014 | 1.21° | PARTIAL |
| other | 325 / 24 | ✅ **0.0000** | 0.0000 | 0.0000 | — | ✅ **MEASUREMENT** |
| overall | 881 / 40 | 0.1226 | 0.0375 | 0.1215 | 1.73° | PARTIAL |

> ### ⭐⭐ **The envelope failure and HP-2's stratum of interest are THE SAME WINDOWS.**
> Straight cruising is a clean measurement — `other` is **0.0000** outside at 2 s. Everything outside the
> envelope is a **turn**, and it leaves via the **heading** axis: the *median* junction window's peak
> heading deviation is **14.95°** against a 12° validated edge. So the corridor co-primary is a genuine
> measurement exactly where it is uninformative (straight driving, CDR = 0.0000) and an extrapolation
> exactly where it carries the signal (junction, CDR = 0.0909 → 0.8432 at K = 185).
>
> ⚠️ **One honest confound, named because it partly softens this.** `dψ = wrap(ego_yaw − yaw_ref[m*])`
> is measured against the **nearest** logged pose. In a turn, heading changes fast along the path, so an
> *along-track* lead/lag converts into apparent heading deviation. Part of the junction's 14.95° is
> therefore longitudinal error wearing a heading costume. It does **not** rescue the conclusion — the
> lateral clause fires on **18.13 %** of junction windows on its own, and the renderer is warped by the
> *reported* `(dlat, dψ)` whatever their provenance, so the fidelity claim is void either way — but the
> number should not be read as pure heading control error.

---

## 5. ⭐⭐ THE VERDICT: **OUTCOME B**, and it is worse than the pre-registration anticipated

**Pre-registered outcome B was: "the envelope breaks *below* K = 70 → the horizon recommendation is too
optimistic and the primary K must drop to where the envelope actually holds. Say so plainly; do not
rescue the recommendation."**

**It breaks below K = 70. It breaks below K = 60. It breaks below K = 20. It breaks at 0.5 s.**

So the second half of outcome B — *"drop the primary K to where the envelope holds"* — **cannot be
executed**, and saying otherwise would be a rescue of a different kind. The horizon that satisfies the
envelope is **0.4 s**, which `GATE_PROTOCOL` §0.3 refuses (rightly: it is *shorter* than the blind
instrument being replaced) and which would measure nothing. The correct conclusion is therefore not a
smaller K:

> ### ⭐ **The envelope is not a HORIZON problem. It is a RENDERER-VALIDATION problem, and it indicts the standing 2 s instrument as hard as the proposed 6 s one.**
> `POD2_EVAL_HOST.md` §4.5's reasoning — *"a 6 s rollout accumulates far less lateral drift than an
> 18.5 s one, so K = 60 is likely to sit inside the envelope"* — is **directionally right and
> conclusionally wrong**: K = 60 does accumulate far less drift than K = 185 (p90 peak XTE 13.55 m vs
> 72.24 m), and it is still **4.5× outside** a 3.0 m validated edge. The envelope was never a function
> of K alone; it is a function of **where the closed loop goes**, and at a junction it goes outside
> within one second at any K.

### 5.1 The registerable recommendation

**Register `K = 60` (6.0 s) as the primary closed-loop co-primary horizon, `K = 70` (7.0 s) as the
documented hard maximum, and `K = 185` as report-only-pooled — and stamp the co-primary
`envelope_verdict: EXTRAPOLATION` with its measured fractions PRINTED in the verdict.**

The K itself is **unchanged from the standup's recommendation**, but the *grounds* are now one, not two,
and the co-primary carries a disclosure it did not have before:

| ground | status | evidence |
|---|---|---|
| **stratum yield** — K ≤ 70 is the ceiling for any stratified verdict | ✅ **CONFIRMED, unchanged.** Junction clusters on the 600 build: 232 (K=20) → **207 (K=60)** → **204 (K=70)** → 196 (K=75) → 58 (K=185); 600 is the corpus maximum, so above K = 70 HP-2 is unmeasurable **permanently** | `PUBLISHED`, `POD2_EVAL_HOST.md` §3.1 |
| **envelope** — "register where the envelope holds" | 🔴 **REFUTED, and not satisfiable at any admissible K.** 0.4 s is the last measurement horizon | `MEASURED`, this run, §4.5 |
| **within-cluster averaging** — K = 60 keeps ~17 windows/episode, K = 185 has 1.00 | ✅ unchanged | `PUBLISHED` + `MEASURED` here |

**Why K = 60 is still the right register even though it is extrapolation.** The three things that make
a horizon useful are unaffected by the envelope finding: it is **3× longer than the blind instrument**,
it keeps **every E1a stratum above the 200-cluster bar on the 600 build**, and the failure it detects is
enormous and CI-separated (`CDR 0.0203 → 0.6388` from K = 20 to K = 185, paired Δ separated). **A failure
observed under a partly-extrapolated renderer is still a failure** — what is withdrawn is the claim that
it is an *in-distribution* failure. That distinction changes what may be *said*, not whether the arm
departed its corridor.

⚠️ **What may NOT be said, and this is the load-bearing part:** no corridor number on this surface may be
described as "in-distribution", "low-OOD", or "a measurement of the model's dynamics". The admissible
wording is *"closed-loop corridor departure under a re-rendered real-footage surface whose fidelity is
validated only to |dlat| ≤ 3.0 m / |dψ| ≤ 12°; **50.66 %** of windows at K = 60 exceed that."*

### 5.2 The experiment that would actually fix it — and its price, MEASURED

`GATE_30K_RESULTS.md` §10.1 offered two branches: *"register at a horizon where the envelope holds"*
**or** *"re-validate P1 out to 18.5 s"*. **Branch one is now closed.** Branch two is the only one left,
and this run prices it:

| to make this horizon a MEASUREMENT | P1 must cover (p90 of peak deviation) | multiple of the current 3.0 m / 12° edge |
|---|---|---|
| K = 20 (2.0 s), overall | 1.74 m / **15.19°** | lat 0.58× (already covered) · ⛔ **yaw 1.27×** |
| K = 20 (2.0 s), **junction only** | 3.51 m / **28.14°** | ⛔ 1.17× / **2.35×** |
| **K = 60 (6.0 s), overall** | **13.55 m / 39.25°** | ⛔ **4.52× / 3.27×** |
| **K = 60 (6.0 s), junction** | **20.82 m / 60.03°** | ⛔ **6.94× / 5.00×** |
| K = 70 (7.0 s), overall | 17.16 m / 43.32° | ⛔ 5.72× / 3.61× |
| K = 185 (18.5 s), overall | 72.24 m / 77.67° | ⛔ 24.08× / 6.47× |

⚠️ **A ground-plane homography from a `f_eff = 266 px` front-wide camera at `h = 1.5 m` will not be
faithful at 13 m lateral displacement.** So the honest reading is that **re-validating P1 will not
extend the envelope to K = 60; it will establish where the renderer stops.** ⭐ **But the cheap version
is worth doing first and it is CPU-hours, not a GPU-week** — and it is *already decision-relevant at
K = 20*, because the **2 s instrument's own p90 yaw is 1.27× outside**: extend the P1 sweep from
`{0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0} m` × `{0, 1, 2, 3, 5, 8, 12}°` to, say, **6 m × 30°** and see
whether the ADE-vs-offset curve stays smooth or breaks. That single experiment does **three** things at
once: it un-voids the ratio criterion (its ceiling is set *entirely* by how far the sweep went, §3.1),
it tells us how much of the junction stratum is recoverable **at the horizon we already use**, and it
puts a number on the only branch `GATE_30K_RESULTS` §10.1 has left. ⚠️ **And it must extend the YAW
arm, not just the lateral one** — at K = 20 lateral is already covered and yaw is not.

⭐ **And there is a second, independent reason to run it: the ratio criterion cannot be repaired any
other way.** Its ceiling (§3) is `1 + (max lat ADE − base)/base + (max yaw ADE − base)/base`, i.e. it is
set **entirely by how far the P1 sweep went.** Extending the sweep is the *only* action that raises the
ceiling. Until then, clause 1 is void and clause 2 is the whole rule.

---

## 6. T2 (§9.6) — v1 @ 600 registered as a NEW row. ✅ DONE

`Project Steering/MODEL_REGISTRY.md` gains **§1.2a "v1's TWO val DEPLOYMENTS — two rows, never one
number"**, plus a deployment stamp on the §5 leaderboard table. `tools/registry_lint.py` →
**PASS (0 errors, 2 pre-existing warnings, neither on the edited lines)**.

| deployment | `ade_0_2s` full-set | episode-cluster bootstrap CI95 (B = 2000) | n windows | **n clusters** | CV floor, same deployment | paired Δ vs CV |
|---|---|---|---:|---:|---|---|
| 40 eps — CANONICAL | **0.4271** | [0.3675, 0.4871] hw 0.0299 | 881 | **40** | 0.8377 | +0.4106 [+0.2050, +0.6240] ✅ sep |
| **600 eps — NEW** | **0.4108** | [0.3956, 0.4273] hw 0.0159 | **13 198** | **600** | ⚠️ **0.6917** | +0.2809 [+0.2457, +0.3142] ✅ sep |

Every element the brief required is in the row: **n as BOTH counts**, the deployment name, the
estimator, and the non-substitutability note — written as a **rule block**, not a caveat, with three
binding clauses (never mix deployments · always quote both n · name the estimator) and the *reason*
stated as a measurement rather than a style preference: **the 600 is an EASIER corpus** (CV floor
0.8377 → 0.6917), so v1's margin over the floor **falls** 0.4106 → 0.2809 while its absolute ADE
improves. **A 40-vs-600 delta is a corpus-composition result, not a model result.**

⚠️ **The one thing I added beyond the brief, because it is the sharper consequence:** the row records
that **`along_track_vs_cv` flips from "tie" to "model wins, separated" between the two deployments with
the point estimate moving 0.7 %** — so *any verdict in the registry that rests on a 40-episode "not
separated" is **unpowered**, not refuted.* That is a standing caution about the registry's own contents
and it belongs in the registry.

⛔ **The 600-episode number is deliberately NOT in the §5 leaderboard table.** That table is now stamped
as the 40-episode / 881-window deployment on its own line, immediately above the header, because a
ranked table is exactly where a substitution would happen.

---

## 7. T3 (§9.9) — pinning the S3 skill bars

### 7.1 What the bar is, and why "it should be deterministic" is not an answer

`S3 bar = max(QWK(B1), QWK(B2), QWK(B3))` (`run_s3_characterisation.operative_blind_floor`) ·
`S3-W bar = QWK(B1_sensor_only)` · `skill = QWK(model) − bar`, quoted to **4 dp**.
Each arm's QWK comes from `s3_blind_baseline._fit_mlp`: a 2-layer MLP (hidden 64), **full-batch** Adam,
400 epochs, class-weighted CE, `torch.manual_seed(seed)`, **no dropout, no shuffling, no augmentation.**
Nothing in that path is *deliberately* stochastic once the seed is fixed — which is exactly why the
observed ±0.01 across hosts is a finding rather than noise, and why "just fix the seed" does not close it.

MEASURED across hosts on md5-identical code and an identical corpus (`PUBLISHED`,
`POD2_EVAL_HOST.md` §3.3.1): lat **0.6534 → 0.6493**, lon **0.5323 → 0.5420**, S3-W lat
**0.2566 → 0.2591**, S3-W lon **0.2881 → 0.2881**.

### 7.2 The design — three separable causes, one host, one variable at a time

The label mining is provably deterministic (12/12 strata reproduce digit-for-digit across two hosts on
two data surfaces), so it is done **once** and cached; every condition below re-runs only the **fit**.
`band_metrics` **rounds QWK to 4 dp** — the precision under test — so the unrounded
`quadratic_weighted_kappa` is taken instead.

| family | what varies | what it separates |
|---|---|---|
| **E-A threads** | `OMP/MKL/OPENBLAS_NUM_THREADS ∈ {1, 2, 4, 8, 16}`, seed 0 | CPU GEMM **reduction order** depends on thread count; 400 Adam steps amplify a 1e-7 difference into a different argmax on borderline rows. If the bar moves here, **pinning threads pins the bar.** |
| **E-B repeatability** | 5 repeats, threads 8, seed 0, **separate processes** | is the fit deterministic *given* the thread count? |
| **E-C seed** | seeds 0…9, threads 8 | the bar's own **estimator variance** — the honest precision it may be quoted at even in a perfectly reproducible pipeline. *A bar pinned by freezing seed 0 is pinned by fiat, not measured.* |

**Pre-committed read-out:** a value is quotable to `d` decimals iff the observed spread is
`< 0.5 × 10⁻ᵈ`. That number is **computed from the runs, not chosen**. If the spread across seeds
exceeds 0.005, the bar is not quotable to 2 dp either, and the honest form is `mean ± spread` — or, if
even that is unstable, **the bar cannot sit in a kill conjunction and must be said so.**

### 7.3 ⭐ RESULT — the mechanism is threading, the fit is bitwise deterministic given it, and the bar is still only quotable to **ONE decimal place**

20 conditions, one host, `artifacts/s3_bar_pinning.json` (+ per-condition `cond_*.json` on pod2).
Mining reproduced the published counts exactly first — `n_train 136,484 / 99,036`,
`n_test 34,337 / 24,987`, `n_test_episodes 558 / 520`, `rows = 102,532` — so the features under test are
the published features.

⭐ **And the reference condition reproduces all four published pod2 bars to every quoted digit:**
`OMP=8, seed 0` → **S3 lat 0.649310 → 0.6493 · S3 lon 0.542019 → 0.5420 · S3-W lat 0.259102 → 0.2591 ·
S3-W lon 0.288112 → 0.2881**. (`POD2_EVAL_HOST.md` §3.3.1: 0.6493 / 0.5420 / 0.2591 / 0.2881.)

| family | what varied | S3 lat | S3 lon | S3-W lat | S3-W lon |
|---|---|---|---|---|---|
| **E-B** repeatability | 5 separate processes, threads 8, seed 0 | ✅ **0.649310 ×5 — BITWISE identical** | ✅ 0.542019 ×5 | ✅ 0.259102 ×5 | ✅ 0.288112 ×5 |
| **E-A** threads | `OMP/MKL/OPENBLAS ∈ {1,2,4,8,16}`, seed 0 | ⛔ spread **0.008690** (0.6471 → 0.6558) | ⛔ spread **0.012528** (0.5335 → 0.5460) | 0.004871 | ⛔ 0.013671 |
| **E-C** seed | seeds 0…9, threads 8 | ⛔ spread **0.020169** (0.6469 → 0.6671) | ⛔ spread **0.027015** (0.5184 → 0.5454) | 0.011630 | ⛔ 0.025699 |
| — | **mean ± sd over 10 seeds** | **0.6557 ± 0.0068** | **0.5350 ± 0.0097** | **0.2578 ± 0.0038** | **0.3021 ± 0.0075** |
| — | **decimal places actually quotable** | **1** | **1** | **1** *(2 with threads pinned)* | **1** |

> ### ⭐ THE ANSWER, in three parts
> **1. The source is the BLAS/threading path, and it is now MEASURED, not hypothesised.** E-B shows the
> fit is **bitwise deterministic across separate processes** at a fixed thread count and seed — 5/5, to
> every digit. E-A shows the *same* seed on the *same* host with only `OMP_NUM_THREADS` changed moves
> the bar by up to **0.0137**. So the ±0.01 across pods was never randomness; it was an **unrecorded
> configuration variable**. Thread count changes CPU GEMM reduction order, and 400 full-batch Adam steps
> amplify a 1e-7 difference into a different argmax on borderline rows.
>
> **2. Pinning threads is necessary and NOT sufficient.** The **seed** moves the bar *more* than the
> thread count does — spread **0.0202 / 0.0270** vs **0.0087 / 0.0125**. That is the bar's own estimator
> variance and no amount of environment pinning removes it. **Freezing seed 0 would pin the bar by fiat,
> not by measurement**, and would hard-code the value of one arbitrary draw into a kill conjunction.
>
> **3. Therefore the bar is quotable to ONE decimal place, not four.**
> `S3 lat = 0.66` · `S3 lon = 0.54` · `S3-W lat = 0.26` · `S3-W lon = 0.30`. Anything finer is noise
> from a variable the artifact does not record.

### 7.4 What this changes for anything the bar adjudicates

`skill = QWK(model) − bar`, so **the bar's spread propagates directly into every skill number.**

* ⛔ **A 4-decimal bar may not sit in a kill conjunction.** The published `0.6534` and `0.6493` are two
  draws from a distribution with sd **0.0068** and range **0.0202**; an arm at 0.6510 clears one and
  fails the other, and both are *correct* runs of the same code.
* ✅ **The honest published form is `mean ± sd over n ≥ 5 seeds at a stated thread count`, with the
  thread count stamped in the artifact** — implemented here, and the `cond_*.json` files carry
  `omp_num_threads_env`, `torch_num_threads`, `torch_version` and a `pred_sha1` per arm so a future
  re-fit can be checked bit-for-bit rather than eyeballed.
* ✅ **The decision rule that follows: an arm whose QWK falls within `bar ± spread` is a TIE, not a
  pass and not a fail.** With the measured spreads that is a **±0.02** dead band on the lateral axis and
  **±0.027** on the longitudinal one.
* ✅ **Every S3 firewall VERDICT survives** — the standup already showed the R1/R2/R3 conclusions are
  ~10× the noise, and nothing here touches them. The `operative_blind_floor` arm was
  **`B3_FULL_CONDITIONING` in all 20 conditions**, so the max-over-arms selection contributes no
  variance of its own. **What is withdrawn is the precision, not the conclusions.**

⚠️ **UNVERIFIED, and stated as such:** I did not reproduce pod3's exact 0.6534, because pod3 has a
different CPU and BLAS build and I did not run there. What is MEASURED is that a **single** environment
variable on **one** host reproduces the full magnitude of the cross-host discrepancy — which is
sufficient to identify the mechanism and to fix the reporting, and is not sufficient to attribute
pod3's specific value to a specific thread count.

---

## 8. Escalations — raised here, not left in a README

1. ⭐⭐ **The horizon recommendation must be re-issued with its envelope clause corrected before any gate
   card is written.** `POD2_EVAL_HOST.md` headline #10 stamps `MEASURED` on a compound that includes the
   envelope clause its own §4.5 labels `HYPOTHESIS`; that clause is now measured and is **false**
   (§5, §7). The *yield* half is untouched and still binds at K ≤ 70. **Owner: whoever writes the next
   `GATE_*.md` / the `POD2_EVAL_HOST.md` author.** Retraction row appended.
2. ⛔ **Two live constants adjudicate on a criterion that is arithmetically incapable of failing** —
   `e1b_eval.py:403` and `e1c_common.py:34`, both `<= 1.30` against an estimator whose supremum is
   **1.298888**. They must be re-adjudicated **INSTRUMENT-FAIL (VOID)** per `GATE_PROTOCOL` §0.7 and
   replaced by `ood.verdict`'s clause 2, which already exists. **This is not a re-run: `readjudicate`
   works from fields the artifacts already carry.** **Owner: E1b/E1c's author.**
3. ⛔ **`RATIO_EXTRAPOLATION_X = 1.5` in `taniteval/ood.py` is itself unreachable** (by 0.201112) on this
   envelope map. The module is correct to report `informative: false`, but a constant that can never
   fire should say so at its definition, and `criterion_1` should be stamped VOID rather than merely
   `fires: false` — otherwise the next reader will treat "clause 1 did not fire" as evidence.
   **Owner: `taniteval/ood.py`'s author.**
4. ⛔ **`taniteval/clhorizon.py::run_v4` is not runnable** (`RawEp` has no `.frames`, §2.2). One-line fix:
   `_data.load_raw` instead of `_data.load_frames`. **I did not patch it** — it is pinned by a sibling
   stream's test and landed hours ago. **Owner: `clhorizon.py`'s author.**
5. ⚠️ **The repo has no interpreter-portability floor and it has now cost a designated eval host.**
   `python3 -m compileall` under the oldest supported interpreter (**3.11**, which is what pod2 runs) is
   a 30-second check that catches the whole PEP 701 class. **Owner: `tools/ci_gate.py`.**
6. ⚠️ **The `p1_envelope` JSON was not on pod2** and had to be copied from the repo
   (`…/2026-07-23-lowood-lanekeeping-refc/lowood_flagship_ci.json` → `pod2:/root/lanekeep/`). Any future
   eval that needs an OOD ratio on pod2 needs it there. **Owner: whoever maintains the pod2 harness
   bundle** — it belongs in the sync, not in an agent's memory.
7. ⚠️ **`GATE_PROTOCOL.md` §0.1's retraction block is imprecise in a binding document.** It says the
   ratio is *"structurally incapable of exceeding its own **1.5** threshold"* — true, but it leaves open
   whether the **1.30** constants the two live scripts use could fire. **They cannot: `sup = 1.298888`.**
   Proposed one-line amendment, offered rather than applied because that file was amended hours ago by a
   sibling stream and editing a mid-flight binding doc is the hazard this program keeps paying for:
   > *"The ceiling is exact and computable from `lowood_flagship_ci.json` alone: `sup(ratio_arr) =
   > 1.298888`. Every threshold above it — including the `<= 1.30` used at `e1b_eval.py:403` and
   > `e1c_common.py:34`, and `RATIO_EXTRAPOLATION_X = 1.5` — is decided before the model runs."*

   **Owner: `GATE_PROTOCOL.md`'s maintainer.**
8. ⛔ **`PRE_REGISTRATION_S3.md` §5.3's skill bars must be re-stated to 1 dp with their spread, before
   they adjudicate anything.** MEASURED (§7): the fit is bitwise deterministic given
   `(OMP_NUM_THREADS, seed)`, thread count alone moves the bar up to **0.0137**, and the **seed moves it
   more (up to 0.0270)**. Replace `0.6534` / `0.5323` with **`0.66 ± 0.007`** / **`0.54 ± 0.010`**
   (mean ± sd over 10 seeds at a stated thread count), and adopt the **tie rule**: an arm inside
   `bar ± spread` is neither a pass nor a fail. ⚠️ The fix is *not* "freeze seed 0" — that pins one
   arbitrary draw by fiat. **Owner: S3's author / `PRE_REGISTRATION_S3.md` §5.3.**
9. ⚠️ **`s3_blind_baseline._fit_mlp` should record its own environment.** It is a *measurement* path and
   its output is a function of `OMP_NUM_THREADS`, `torch.__version__` and the seed — none of which the
   result JSON carried. `scripts/s3_bar_pinning.py`'s `fit` phase stamps all three plus a `pred_sha1`
   per arm; the same three lines belong in `run_s3_characterisation.run_firewall`. **Owner: S3's
   author.** *(Sibling of `tanitad.instruments.numerics.strict_numerics`, which exists for exactly this
   reason on the GPU path and has no CPU counterpart.)*

---

## 9. Deliverable manifest

All repo paths are relative to the working tree on the dev box and are **`git add`-ed, NOT committed and
NOT pushed**. Anything that exists in only ONE place is marked ⚠️.

| artifact | where it lives | what it is |
|---|---|---|
| `HORIZON_ENVELOPE.md` | `repo:TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-26-horizon-envelope-closeout/` | this report |
| `scripts/k_sweep_envelope.py` | repo (same dir) · `pod2:/root/taniteval/_ksweep.py` | the K-sweep driver (uses `taniteval.clhorizon` + `taniteval.ood`, the packaged modules — §9.8's integration) |
| `scripts/ood_blast_radius.py` | repo (same dir) | the supremum proof + repo-wide re-adjudication; runs on the dev box, no GPU |
| `scripts/s3_bar_pinning.py` | repo (same dir) · `pod2:/root/s3pin/_s3pin.py` | T3: mine-once / fit-many, `mine` · `fit` · `sweep` phases |
| `artifacts/ksweep_results.json` | repo (same dir) · `pod2:/root/ksweep/ksweep_results.json` | every K: stratified corridor block + full OOD disjunction + truncation curve |
| `artifacts/ksweep.log` | repo · `pod2:/root/ksweep.log` | the run log, one line per K |
| `artifacts/perwindow_K*.pt` | repo · `pod2:/root/ksweep/perwindow_K*.pt` | per-window `lat`/`yaw`/`ade2s`/`hd2s`/`hdK`/`speed`/`eid`/`t0`/`epi`/`de_fixed` — **the arithmetic-only path: any half-width, stratification or OOD rule can be recomputed from these with no GPU** |
| `artifacts/ood_blast_radius.json` | repo (same dir) | supremum 1.298888 · 3 live constants adjudicated · 181 nodes re-adjudicated · 14 class flips |
| `artifacts/preflight_pod2_closeout.json` | repo · `pod2:/root/preflight_pod2_closeout.json` | P0 `sys.path` audit / P1 corridor / P2 lateral / P3 val chokepoint |
| `artifacts/preflight2_pod2.json` | repo · `pod2:/root/preflight2_pod2.json` | P4 clhorizon+ood present & exercised · P5 saturation demo · P6 guard-can-fire · P7 repo↔pod md5 |
| `artifacts/RESULT_v1_40ep_ksweep_preflight.json` | repo · `pod2:/root/taniteval/results/` | P3 — v1 = 0.4271 [0.3675, 0.4871] on this host, this session |
| `scripts/render_sweep_table.py` | repo (same dir) | renders §4.2 from the JSON so no sweep number is hand-transcribed |
| `scripts/common_start_paired.py` | repo (same dir) | the paired K contrast on identical windows, offline from the dumps (no GPU) |
| `artifacts/common_start_paired.json` / `.txt` | repo (same dir) | 41 shared windows, paired Δ per stratum per K |
| `artifacts/s3_bar_pinning.json` | repo (same dir) · `pod2:/root/s3pin/s3_bar_pinning.json` | T3: 20 conditions + the pinning verdict |
| `artifacts/s3pin_mine.log`, `artifacts/s3pin_sweep.log` | repo · `pod2:/root/` | T3 run logs (mining reproduces the published S3 counts) |
| ⚠️ `pod2:/root/s3pin/cond_*.json` (20 files) | **pod2 only** | per-condition detail incl. `pred_sha1` per arm; the summary JSON in the repo carries every number they contain |
| ⚠️ `pod2:/root/s3pin/s3_features.npz` (14 MB) | **pod2 only** | the cached mined features — **deliberately not staged**: it is a 39-minute derived intermediate, reproducible by `s3_bar_pinning.py mine`, and the repo should not carry mined feature matrices |
| **registry row** | `repo:Project Steering/MODEL_REGISTRY.md` §1.2a + §5 deployment stamp | T2 |
| **4 retraction rows** | `repo:Project Steering/RETRACTION_LOG.md` | C13 (analytic) · C4 (headline outruns body) · C2 ×2 (PEP 701, `run_v4`) |
| **2 code fixes** | `repo:stack/scripts/eval_flagship_v4.py`, `repo:stack/scripts/vlm_kin_crossval.py` | the PEP 701 portability fix (§2.1) |
| ⚠️ **pod2 harness** | `pod2:/root/taniteval` (221 files) · `pod2:/root/TanitAD/stack` (344 files) | re-synced from the repo this session, **0 md5 mismatch**; the repo is the other copy |
| ⚠️ `pod2:/root/lanekeep/lowood_flagship_ci.json` | pod2 only *(the repo has the source copy)* | the P1 envelope; **not part of the standard sync — escalation #6** |

**Nothing that took real effort exists in only one place.** The two pod-only rows are a derived
intermediate and a per-condition expansion of a summary that *is* staged; both are marked and both are
reproducible from staged code.

---

## 10. What was deliberately NOT done, and the honest limits

* ⛔ **pod1 was never contacted.** It is training v2corpus.
* ⛔ **Exactly one job ran on pod2 at a time.** The K-sweep and the T3 refits were serialised, and T3
  specifically must not share a box with a GPU job — **thread contention is one of the candidate
  mechanisms T3 is testing**, so running it under contention would confound the measurement it exists
  to make.
* ⛔ **The 600-episode cache was read-only.** Only `list_val_episodes` + `torch.load(..., mmap=True)`
  touched it.
* ⛔ **No interval anywhere in this report comes from `overlapping_holdout_se`.** Where a result JSON
  carries one it is named `legacy_overlapping_holdout_se` and is ignored.
* ⛔ **`taniteval/clhorizon.py` was NOT patched** despite a reproduced defect (§2.2) — a sibling stream
  owns it and it landed hours ago. Escalated instead.
* ⛔ **`GATE_PROTOCOL.md` was NOT edited** — proposed amendment text is in escalation #7.
* ⚠️ **The sweep ran on the 40-episode deployment, not the 600.** That is the deployment the co-primary
  is registered on and it makes K = 185 and K = 20 *reproductions* of committed numbers, but it means
  the **junction** stratum carries 22 clusters at K = 20 and 6 at K = 185. On the 600 build those become
  232 and 58 (`PUBLISHED`, `POD2_EVAL_HOST.md` §3.1). The full sweep at 600 is ≈ 15× the compute
  (≈ 37 GPU-h) and was not affordable in one session. **The envelope FRACTIONS are proportions over
  tens of thousands of steps and are precise at this n; the stratified CDR INTERVALS are not, and are
  quoted with their n every time.**
* ⚠️ **Goal provenance is ORACLE on every closed-loop number here** (matching the registered
  co-primary). Per `GATE_PROTOCOL` §0.8 **none of these is a deployed-capability claim.** The
  envelope conclusion is, if anything, *conservative* under this caveat: an arm without the oracle would
  drift further, not less.
* ⚠️ **The P1 envelope constants are from the flagship v1 arm**, not v4 (`ood.py` stamps this on every
  node). They describe the **renderer's validated range**, which is arm-independent — but the *ratio*
  they map to is a v1 ADE ratio, which is one more reason the ratio clause is the weak half.


