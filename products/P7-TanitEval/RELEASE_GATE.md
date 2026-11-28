# THE TANITAD RELEASE GATE — the criteria every model release must pass

`Specified 2026-08-23 by TanitAD_EvalFlyWheel at the PI's direction: "finalize the
tests and criteria which must run at each Model release." Implementation:
tools/release_gate.py · tests: tools/tests/test_release_gate.py (35, incl. a
deliberate-regression arm per criterion) · criteria source:
products/P7-TanitEval/CRITERIA_REGISTRY.json v2.2.0.`

---

## 0. What the gate is, and what it refuses to be

One command, one verdict:

```bash
python tools/release_gate.py --model <key> --artifacts <dir> --baseline <key> --json out.json
```

- ⛔ **The verdict is a CONJUNCTION over named criteria — never a pooled score.**
  A single number hides exactly the trade-off the four-families rule exists to expose.
- ⛔ **Four verdict states, not two**: `PASS` · `FAIL` · **`CANNOT-RULE`** · `N/A-with-reason`.
  `CANNOT-RULE` is load-bearing. This programme shipped a gate that *could never rule*
  (the O6 rank gate: spectrum n=24 against a ceiling of 1024) and read its silence as a
  pass for weeks. A criterion that cannot decide at the configured settings must say so
  **loudly** rather than returning a perpetual INCONCLUSIVE.
- ⛔ **The gate prints the scope it swept.** A gate that silently swept one directory
  reports that directory's verdict, not the model's (C133).

## 1. The criteria

| id | blocking | what it checks | the failure that earned it |
|---|---|---|---|
| **RG-01** | ✔ | the sweep found artifacts for this model | a gate with no input passes vacuously — see C134 |
| **RG-02** | ✔ | four families PRESENT or REFUSED-with-reason-and-n | three ADE-only reports went out after the rule was binding |
| **RG-03** | ✔ | every artifact carries a recognised tier stamp | 93 of 135 in-scope artifacts carry none |
| **RG-04** | ✔ | **a T1 artifact is present** | ⛔ T0 is a WM diagnostic. The same checkpoint reads 0.3659 (T0) and 9.3697 (T1) |
| **RG-05** | ✔ | paired episode-cluster bootstrap | quadrature-combined intervals are not valid for two arms on one window set |
| **RG-06** | ✔ | `overlapping_holdout_se` refused | it biases the POINT ESTIMATE bidirectionally, up to a sign flip |
| **RG-07** | ✔ | vision-only at inference, DECLARED | labels may use ego; inference may not (PI, 2026-08-03) |
| **RG-08** | ✔ | goal ⟂ situation-classifier output, DECLARED | a shared path makes planner gains non-attributable (PI, 2026-08-03) |
| **RG-09** | ✔ | route-head echo test when a route score is near 1.0 | flagship v1's route head scored 1.0000 as an exact bijection of its own input (369/369, 81/81) |
| **RG-10.{LON,LAT,TAC,STR,CMP}** | ✔ | **the release beats its trivial control, per family** | the control beat the model 22× and nothing was watching |
| **RG-11.{…}** | advisory | no family regresses vs the previous release | advisory until two clean releases exist to compare |
| **RG-12** | ✔ | parity: canonical corpus + skip-hash stamped | without it no cross-arm delta is interpretable |
| **RG-13** | ✔ | every compared arm sits on one identical window set | comparing arms on different grids is not a comparison |
| **RG-14** | ✔ | the estimator CAN RULE: episode clusters ≥ floor | the O6 precedent, generalised |

**Blocking policy is the PI's call** (open as of 2026-08-23). Recommendation on record:
blocking on the list above, **advisory on regression** — a regression gate with no
trustworthy baseline manufactures false failures.

## 2. First dry run — the honest verdict

Run against our best banked T1 artifacts (`stageA`, baseline `v5f30k`, n=6844 windows /
40 episodes, episode-cluster bootstrap):

> ## ⛔ RELEASE-BLOCKED
> **8 blocking FAILs · 0 CANNOT-RULE · 3 N/A-with-reason · 8 PASS**

⭐ **The gate was not weakened to make our own arms pass.** That was the point of running
it against them first.

### 2.1 The control beats the release on every family

`RG-10` compares the release against its own hold-action control on the same windows:

| family | metric | release | control | ratio |
|---|---|---|---|---|
| COMPANION | `ade_dense_m` | 9.3697 | **0.4246** | **22.1×** |
| LONGITUDINAL | `speed_mae_mps` | 9.7291 | **0.5671** | **17.2×** |
| LATERAL | `cross_mae_m` | 0.7446 | **0.1689** | **4.4×** |
| TACTICAL | `lat_decision_kappa` | 0.3795 | **0.6449** | control higher |

⚠️ **Read the tactical row carefully.** κ is chance-corrected, so this is not a
class-imbalance artifact: **the do-nothing control agrees with the executed manoeuvre
better than the trained model does.** Previously we knew only that ADE was worse; the
per-family gate shows the deficit is not confined to displacement.

### 2.2 A regression against the previous release

`RG-11` (advisory): `heading_mae_deg` 2.7171 → **3.8776** · `yaw_rate_mae_degps` 4.5151 →
**4.9188** · `lat_decision_kappa` 0.5083 → **0.3795**. stage-a-repaired is *worse* than
v5f30k on lateral and tactical. Either fix it or state the trade explicitly — the gate
accepts a stated trade, never a silent one.

### 2.3 The rest

- **RG-02** — 4 required criteria silently absent (neither emitted nor refused).
- **RG-07 / RG-08** — **UNDECLARED**, not violated. The artifacts simply do not record what
  the model consumed at inference. ⚠️ This is the right verdict: an undeclared leak guard
  cannot be scored, and the gate refuses to assume compliance. Fixing it is a one-line
  addition to the emitter, not a model change.
- **RG-09 / RG-10.STR / RG-11.STR** — `N/A` **with a reason**: no strategic metric exists on
  this corpus (PhysicalAI-AV carries no map, lane graph, junction label or route signal),
  so there is nothing to echo-test or compare. An honest N/A, not a pass.

## 3. What the gate cannot check yet

| gap | why | what would close it |
|---|---|---|
| tactical **anchor/fan selection** | arms commit to ONE path per window, so there is no candidate fan to score | emit `<arm>_fan_err` / `<arm>_sel_idx`; `taniteval.selgap` already scores it |
| STRATEGIC, at all | corpus-blocked, not instrument-blocked | AlpaSim or an external corpus |
| NavSim / nuScenes rows | adapter built and synthetic-tested; no dataset on disk | a PI decision on the download |
| cross-tier comparison | ⛔ deliberately refused — comparisons across tiers are invalid | nothing; this is correct behaviour |

## 4. How to run it at a release

1. Produce the eval artifacts (four families, tier stamped, parity recorded).
2. `python tools/release_gate.py --model <key> --artifacts <dir> --baseline <previous>`
3. Non-zero exit = blocking failure. Read the per-criterion table; **never the exit code alone**.
4. Bank the JSON beside the artifacts, and update `MODEL_REGISTRY.md` + the leaderboard.

⚠️ **Run it from a checkout whose path you have verified**, and check the printed scope
line. C134 measured a sibling guard skipping 373 of 373 files because a path filter matched
the worktree root — the suite passed on no input at all.
