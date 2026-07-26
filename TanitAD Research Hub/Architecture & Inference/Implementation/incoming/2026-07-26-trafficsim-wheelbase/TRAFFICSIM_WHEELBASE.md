# `trafficsim` reactivity (full-runtime) + the wheelbase option-B execution

**Date:** 2026-07-26 (Europe/Berlin) · **Host:** `tanitad-eval` only. pod1 / pod2 / pod3 never contacted.
**Author:** trafficsim+wheelbase agent
**Status:** PRE-REGISTERED before any run. Sections filled in order; banked incrementally.

**Evidence classes** (CLAUDE.md operating standard 1): `MEASURED` (ours + artifact path) ·
`PUBLISHED` (cited) · `INHERITED` (another doc, NOT re-verified) · `ESTIMATED` · `HYPOTHESIS`.
**Tiers:** `PROVISIONAL` / `CONFIRMED` / `DECISION-GRADE`.

**Estimator, named once and used throughout:** paired episode-cluster bootstrap
(`taniteval/ci.py`, B = 2000). Resampling unit stated per test. **`overlapping_holdout_se` is never used.**

---

## 0. PRE-REGISTRATION — written before any measurement

### 0.1 Task 1 — what I am actually testing, and why it is NOT what the brief says

The brief's premise is: *"`RUN_RECIPE.md:26` says trafficsim is disabled by default, so the reactive-agent
model may simply never have been enabled — in which case the tactical gate's `agents don't react` failure
is a configuration artefact and the negative result is void."*

⚠️ **I am recording, before running, that this premise appears to be circular, and that the check for it
is the first thing I will do.** The `[−0.21, +0.14] m against a 4.5 m noise floor` figure the brief quotes
as the gate's failure is — on its face — the figure published in
`…/incoming/2026-07-26-4brain-gates/GATE_RESULTS.md` §2.4, and that document's §2.1 states that the same
session *fetched the CATK weights, built the PyG extensions and ran the service*. If so, the number the
brief wants to void **was produced with trafficsim enabled**, and "it was never enabled" cannot explain it.

**Pre-registered outcome A (premise void):** trafficsim was already enabled when the gate-2 number was
produced. Then the brief's proposed mechanism is dead, and the honest report is *"the premise is
falsified, the gate stands on this ground"* — **not** a manufactured reason to re-run.

**Pre-registered outcome B (premise live):** trafficsim was genuinely off, or non-functional, during
gate 2. Then the gate-2 verdict is void and must be withdrawn loudly.

**Either way, one open question remains and it is named by the gate document itself** (§2.4, the
"one thing that could overturn this" box): gate 2 drove the trafficsim service **directly over its own
gRPC contract**, with a session hand-built after `runtime/services/traffic_service.py` — *faithful, but
not byte-identical to a full runtime integration*. Its own stated decisive follow-up is **one full
closed-loop rollout with `trafficsim=catk`**. **That is the experiment I will run**, because it is the
only remaining way the "agents don't react" verdict could still be a harness artefact.

### 0.2 The full-runtime reactivity test — design fixed in advance

Same contrast as gate 2, so the comparison is like-for-like, but through the **complete runtime**
(wizard-generated configs → renderer + physics + controller + trafficsim + driver → `alpasim_runtime.simulate`):

| arm | ego behaviour | how |
|---|---|---|
| **GO** | ego drives forward | constant-forward driver (the M2 `simple_driver.py` policy) |
| **STOP** | ego halts and stays put | same driver, zero-velocity trajectory |
| **GO2** | identical in construction to GO | ⭐ the **stochastic-floor control** |

Non-ego agent positions are read from the runtime's own `rollout.asl` log — the runtime's record, not a
hand-built session. Statistic, fixed in advance and identical in form to gate 2:

    Δ = between_arm_mean_pairwise_distance − within_arm_mean_pairwise_distance

with `between` = GO-vs-STOP pairs and `within` = GO-vs-GO2 pairs, **like-for-like and equal under the
null**. Reaction ⇒ Δ **positive and CI-separated**. Unit of resampling = **agent**.

**Pre-registered decision rule, both outcomes committed now:**

| result | verdict I will publish |
|---|---|
| Δ **positive and separated**, concentrated in the near-ego stratum | **The tactical gate's "agents don't react" verdict is VOID.** I withdraw it, loudly, in the headline — the direct-gRPC harness was the artefact. |
| Δ **not separated**, near-ego null | **The verdict SURVIVES, and is now stronger**, having been reproduced through a second, independent integration path. |
| Δ separated but **only far-field**, near-ego null | **Weakened, not void** — reported as a partial, with the multiplicity count stated. |

**MDE, stated before running** (this is the rule that killed a control today at MDE 2.8× the leak it
existed to catch). The effect this test exists to catch is *an ego-induced displacement of nearby
agents large enough to make `Y_outcome` a function of the policy's choice.* Gate 2's best-powered scene
bounded that at ±0.21 m against a 4.5 m floor. **I will report this test's own achieved CI half-width
and state explicitly whether it is tight enough to catch an effect of the size T1–T4 need** — and if the
full-runtime test is *less* powered than gate 2's, I will say so and will **not** claim it overturns anything.

**Proof the test CAN fail (both directions, required):**
1. **Fidelity direction** — the arms must actually differ at the ego: I will measure the GO-vs-STOP ego
   separation and require it to be large (gate 2 measured 19.94 m mean / 60.99 m max). If the ego does not
   differ, the test is void and I will say so rather than reporting a null.
2. **Deliberately-failing-input direction** — a **replay control**: compare returned agent positions
   against the agents' own logged tracks. If they match, `trafficsim` is replay and the whole construct
   collapses regardless of the arm contrast.

### 0.3 Licence constraint, acknowledged before touching anything

AlpaSim's NuRec/gsplat renderer is under **NGC-DL-CONTAINER-LICENSE, which forbids derivatives.**
I will **run and configure** it only. `trafficsim=catk` is an existing wizard config option
(`src/wizard/configs/trafficsim/catk.yaml`) — selecting it is configuration, not a derivative.
**If any step requires editing renderer code I stop and report.** Recorded in §4.

### 0.4 Task 2 — wheelbase option B, pre-registered scope

Option B is **fix-forward-only**. Before executing I commit to the parity rule: **if B would change
episode selection, I stop and escalate instead of proceeding.** The concrete B is whatever
`…/incoming/2026-07-26-wheelbase-impact/WHEELBASE_IMPACT.md` §5 specifies — read first, followed as
written, not as summarised to me. Its parity statement is reproduced and checked in §5.

---

*(Sections 1–6 are filled in as the work runs. Nothing below this line was written before its measurement.)*

---

## 1. TASK 1 STEP 1 — the premise, verified. **It is FALSE, and it was falsified by our own artifact.**

### 1.1 Is `trafficsim` present, importable, functional? — YES, on four independent probes

All `MEASURED` 2026-07-26 on `tanitad-eval`, tier **CONFIRMED**:

| # | probe | result |
|---|---|---|
| 1 | package present | `…/alpasim/src/trafficsim/alpasim_trafficsim/` with `catk/`, `grpc/`, `config/`, an `.egg-info` **and populated `__pycache__`** (i.e. it has been imported before) |
| 2 | importable | `import alpasim_trafficsim, alpasim_trafficsim.catk.model_adapter` → **OK** in the alpasim venv |
| 3 | weights real | `data/trafficsim-models/catk_v120/latest.ckpt` = **69,960,427 B**, sha256 `7c5a89bc6e876c025a82572b72f87ca97dd75fe5f57245dbf2b63fc3b3c4455e` — **byte-identical to the hash `GATE_RESULTS.md` §2.1 published**. Token vocabularies present. |
| 4 | runnable | PyG extensions import (`torch_cluster` 1.6.3); `catk_trafficsim_server` entrypoint exists in the venv |

**And a fifth, which is the one that matters:** `src/wizard/configs/trafficsim/catk.yaml` exists as a **stock config group**, selected with `trafficsim=catk`. Enabling it is a **configuration** act, not a code change.

### 1.2 ⛔ The brief's premise is circular — stated plainly

The brief asks me to test whether the tactical gate's `[−0.21, +0.14] m vs a 4.5 m noise floor` failure
"may be an artefact of configuration, because trafficsim was never enabled".

**That number was produced BY the run that enabled it.** `GATE_RESULTS.md` §2.1 (dated 2026-07-26,
same day) records that the gate-2 session **fetched the CATK weights via the LFS batch API**
(sha256-verified — the same hash I re-measured above), **built the PyG extensions from source**, and
**ran the CATK service**; §2.4 then reports agents deviating from their logged tracks by **8.37–78.17 m**
with **≥95.5 % of poses not the logged pose** — i.e. `IS_REPLAY: false`, CATK genuinely simulating.

The program harvest read `RUN_RECIPE.md:26` — *"trafficsim (disabled by default)"*, written **2026-07-22**
— and inferred "never enabled". That line is still **literally true and still the default**, and it is
**stale as evidence** about a run performed four days later.

⇒ **Pre-registered outcome A. The brief's proposed mechanism for voiding the tactical gate is dead.**
The gate did not fail because trafficsim was off. I am not going to manufacture a reason to re-run it.

### 1.3 ⭐ But the premise being wrong surfaced a DIFFERENT true finding, and it is material

`src/wizard/configs/trafficsim/disabled.yaml` sets `runtime.endpoints.trafficsim.skip: true`, and
`runtime/services/traffic_service.py:simulate_traffic` shows what `skip` means (`MEASURED`, source-read):

> `if self.skip: … return traffic positions from recorded trajectories`

**`skip` is literal REPLAY.** So every closed-loop number the program has published — the REF-C n=12
suite, the flagship-vs-REF-C n=12 suite, and the native-1080 re-run (`RUN_RECIPE.md` §12, §15, §16) —
was produced against **replayed, non-reactive traffic**, because they all used the default.
`MEASURED`, tier **CONFIRMED**. This does not invalidate those results (both arms saw identical
replayed traffic, so the *paired* comparisons stand) but it does bound what they can mean: **no
published TanitAD closed-loop number has ever involved a reactive agent.** That belongs in the record
and was not previously written down anywhere.
