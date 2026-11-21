# refcv6 — BUILD RESULT: pre-registration, launch scripts, pre-launch gate

**2026-09-10 · Architecture & Inference FlyWheel · branch `agent/arch-inf-20260803` · HEAD at start
`b42ab460`**

⛔ **NO TRAINING RUN LAUNCHED. 0 GPU spent.** There is no A40 — the pod was stopped and is gone.
Thor and the dev-box 4060 were used for CPU/import-level checks only; no GPU or RAM load was added
to any box.

⭐ **The headline: refcv6 is a CONFIGURATION, not new code — and that is now MEASURED, not asserted.**
All four levers are declared **and consumed** in `refc_v3_train.py`, and DiffusionDrive's exact
Hungarian matching plus coupling (1) already exist. The work in this package is therefore *the
discipline around the arms*, not the arms themselves: an argv single-source-of-truth, a
one-variable checker driven by the trainer's own parser, a pre-launch gate that refuses, a
supervisor that cannot lose its lock, and **the bar as executable code that cannot silently drop its
non-regression clauses**.

---

## 1. Are all four levers present? — YES, by two probes each

`code/lever_census.py` · `raw/lever_census.json`. Trainer md5 **`4e9c9834f20d6c50b35c163aae4ab4a8`**
(5,640 lines), byte-identical to the repo's `stack/scripts/refc_v3_train.py`.

⭐ **The census asks TWO questions, and the second is the one that matters:** is the flag
*declared*, and is its parsed value *consumed*? That second question exists because of
`tac_goal_tok_head` — **11,286 parameters, `grad_abs_sum` exactly 0 for all 40,284 steps** of
refcv5-v2: parsed, stamped into `config.json`, and reaching nothing.

| arm | flag | declared | mentions | consumption sites | verdict |
|---|---|---|---|---|---|
| **D** | `--w-u0` | **:5095** | 7L / 7occ | 4 — `:4046` → `model._w_u0` → `:2545` read → **`:2564`/`:2584` added to `loss`** | ✅ `PRESENT_AND_CONSUMED` |
| **A** | `--agents` | **:5151** | 31L / 32occ | 5 — `:565`, `:568` → `core.agents` | ✅ `PRESENT_AND_CONSUMED` |
| **B** | `--wp-index` | **:5340** | 25L / 26occ | 1 — `:639` `getattr(args,"wp_index")`, which gates the `WaypointIndexConfig` build at **`:653`** | ✅ `PRESENT_AND_CONSUMED` |
| **C** | `--w-tac-goal` | **:4922** | 11L / 11occ | 5 — `:4043` → `model._w_tac_goal` → `:2751` → loss | ✅ `PRESENT_AND_CONSUMED` |

**DiffusionDrive's machinery, all present:** `agent_slots.py` (md5 `5c6c163b…`) — `AgentSlotDecoder`
**:256**, `hungarian` **:384**, `match_slots` **:481**, `slot_set_loss` **:523**, `targets_from_join`
**:690**. `refc_wp_index.py` (md5 `bfa75ec5…`) — `WaypointIndexConfig` **:128**,
`waypoint_agent_geometry` **:192**, `WaypointIndexBias` **:340**, `build_relation` **:423**.

⚠️ **Two honest notes on the numbers.**
1. **`grep -c` counts LINES; `str.count` counts OCCURRENCES**, and they differ: `--agents` is
   **31 lines / 32 occurrences**, `--wp-index` **25 / 26**. The brief's 31 and 25 are the *line*
   counts and reproduce exactly. Neither number is wrong; quoting one as the other would be. The
   artifact emits both so a reader never has to guess which question was asked.
2. **B's single consumption site is correct, not a defect.** `args.wp_index` is read exactly once
   (`:639`), bound to a local, and that read gates the whole WP-B construction at `:653` — verified
   by reading source, because `--wp-index` is precisely the flag that previously carried the
   *parsed, stamped and inert* defect and it deserved the extra look.

⛔ **Every count is paired with a same-breath control that must read non-zero** (`def main` = 1,
`add_argument` = 110). On this mount a search tool reports "no matches" for files it could not
*open*, so a zero is a claim about the READ. The census returns `INCONCLUSIVE` — never "absent" —
when a control fails.

---

## 2. Each arm's exact argv, and the namespace diff

**BASE = refcv5-v2's own 65-token argv**, read back from the banked
`C:\Users\Admin\refcv5v2_final\config.json` and **programmatically verified**: verdict `MATCH`,
65/65, `first_difference: null`. ⛔ Not hand-transcribed — `arms.verify_base_against_banked_config()`
re-reads the JSON on every gate run, and reports `INCONCLUSIVE` (never agreement) if the file cannot
be read.

### 2.1 ⛔ One deliberate deviation, and its consequence

**BASE additionally carries `--agent-join` in EVERY arm, control included.** WP-D's precedent
verbatim: *"the join must be in both arms or the A/B also changes the dataset."*

⇒ ⛔ **`V0` IS NOT A REPLICATION OF refcv5-v2**, and this is stated rather than hidden. Every refcv6
delta pairs against `V0`, never against refcv5-v2's banked 40,284-step numbers — doing otherwise
would differ in **the BASE** *and* **the step budget** as well as in the lever. refcv5-v2's values
are the **provenance** of the non-regression clauses, not their comparison operand.

### 2.2 The arms

| arm | tokens changed from BASE | pairs against |
|---|---|---|
| **V0** | *(none)* | — |
| **D** | `--w-u0 0` | V0 |
| **A** | `--agents head` + `--w-agent <W>` | V0 |
| **B** | `--agents head --w-agent <W>` + `--wp-index on` | **A** |
| **C** | `--w-tac-goal <PI>` | V0 |
| **V0b/Db/Ab/Bb/Cb** | `--seed 1` | their own treatment |

### 2.3 Does the parsed-namespace diff show exactly one key differing? — YES

`code/check_one_variable.py`, through the trainer's **own** `build_parser()` ·
`raw/one_variable_probe.json` · verdict **PASS**:

```
A vs V0  (3 keys): agents=LEVER, w_agent=CONSTITUTIVE, out=BOOKKEEPING
B vs A   (2 keys): wp_index=LEVER, out=BOOKKEEPING
C vs V0  (2 keys): w_tac_goal=LEVER, out=BOOKKEEPING
D vs V0  (2 keys): w_u0=LEVER, out=BOOKKEEPING
V0b vs V0 · Db vs D · Ab vs A · Bb vs B · Cb vs C  (2 keys): seed, out = BOOKKEEPING
```

⇒ **every lever pair moves exactly ONE key classified `LEVER`.**

⛔ **Arm A is 3 keys, and I am not calling that one.** `w_agent` is **CONSTITUTIVE**: the trainer
*refuses* `--agents head` at `w_agent <= 0` (**`:583`** — *"a REFUTATION manufactured by a missing
loss"*), and `AGENT_WEIGHT_DEFAULT = 0.0`. **There exists no argv in which `agents` moves alone.**
This is enforced, not documented: the allow-list is exactly `{"A": ("w_agent",)}` with the guard's
file:line, and **any** third key is classified `VIOLATION`. `--agent-join` does not appear because
it is in BASE for every arm.

⛔ **`anchors` is `PARITY_PINNED`, not bookkeeping** — a per-arm anchor file would be a hidden
second lever, so a differing `anchors` key is a `VIOLATION_PARITY`, never tolerated as a path.

### 2.4 ⭐ The checker is PROVEN able to go RED — 6/6

*A check that shares the defect it checks for is green forever.* `code/mutation_proof.py` ·
`raw/mutation_proof.json`: **baseline GREEN, 6/6 mutants KILLED.**

| mutant | killed by |
|---|---|
| M1 arm D also carries `--wp-index on` (the `--v2` conflation failure) | `wp_index (VIOLATION)` |
| M2 arm D silently moves `--lr 1e-4 → 3e-4` | `lr (VIOLATION)` |
| M3 replicate carries the same seed as its treatment | *"it would measure nothing"* |
| M4 arm D points at its own `anchors_D.pt` | `anchors (VIOLATION_PARITY)` |
| **M5** arm D sets `--w-u0 0.9` instead of `0` | ⭐ *"arm is 0.9, pre-registered target is 0.0"* |
| M6 arm B paired against V0 instead of A | `agents` + `w_agent (VIOLATION)` |

⭐ **M5 is the discriminating one.** It differs in **exactly one key**, so a key-*counting* check
passes it happily — it is the control wearing D's name. Only the **value** assertion sees it, and
`EXPECTED_MOVE` states each lever's from/to as **literals**, never as an expression over the
trainer's own defaults (an expectation computed from the code under test measures *determinism, not
correctness*).

---

## 3. Are the lateral and strategic clauses undroppable? — YES, and it is PROVEN

⭐ **This was the half most likely to be lost, and prose could not have protected it.** A clause
written only in a paragraph is dropped by being *not mentioned* in the results table — which is
exactly how ADE-only reports kept going out after four families were made binding.

⇒ **The bar is executable code** (`code/verdict_refcv6.py`) which **REFUSES to emit `SUCCESS` when a
required clause's data is ABSENT**. `MISSING_DATA` is a **distinct verdict from `FAIL`**, and
neither is `SUCCESS`:
* absence **blocks** the verdict rather than passing it, and
* absence is **not** scored as a failure either — which would invite *deleting* a family to "fix" it.

`code/verdict_dropproof.py` · `raw/verdict_dropproof.json`. A deliberately **generous** reference
panel (clears ADE against both baselines with room to spare) reads `SUCCESS`, so each mutant fails
for exactly one reason. **MEASURED: baseline `SUCCESS`, 13/13 BLOCKED.**

| mutant | verdict |
|---|---|
| omit `heading_err` / `yaw_rate_err` / `cross_track` / `curvature_mae_masked` (4) | `MISSING_DATA` |
| omit the whole **LATERAL** family (ADE-only reporting) | `MISSING_DATA` |
| omit the whole **STRATEGIC** family | `MISSING_DATA` |
| `route_acc` present but **`n = 0`** — refcv4b's exact state | `MISSING_DATA` |
| masked curvature **without** its straight-line floor | `MISSING_DATA` |
| omit the `ha` baseline (report only `ha0_ext`) | `MISSING_DATA` |
| omit the **replicate floor** | `MISSING_DATA` |
| ADE **separated but margin 0.01 < 0.10** | ⭐ **`FAIL`** — *"clearing the CI while missing the margin is a FAIL as written"* |
| estimator = `overlapping_holdout_se` | `REFUSED` |
| tier stamped **T0** | `REFUSED` |

The clauses carry refcv5-v2's reference values as literals so a regression is legible:
heading **1.2121**, yaw-rate **1.0534**, cross-track **0.0994**, masked curvature **0.003485**
(= 0.512× the straight-line floor); route accuracy **0.7708** [0.7146, 0.8254], κ 0.4614,
**n 3,622**, chance 0.3333, from **771 parameters**.

---

## 4. What does the pre-launch gate refuse on?

`code/prelaunch_gate.py`. ⛔ **The verdict is the JSON artifact, never the exit code** — *"the
admissible evidence that this gate did not run is the MISSING JSON."* `launch_refcv6.sh` reads the
file and refuses on anything but `PASS`. ⛔ **`INCONCLUSIVE` IS NOT A PASS.**

| check | refuses on | today |
|---|---|---|
| **C1** BASE provenance | `arms.BASE_V5V2` ≠ the banked `config.json['argv']` | ✅ **PASS** (65/65 `MATCH`) |
| **C2** one variable | any pair moving more than its declared lever | ✅ **PASS** (9 pairs, 0 refused) |
| **C3** import closure | `MISSING_REMOTE > 0` **or** `DRIFT > 0` (`launch_closure_audit.py --verify-import`) | ⏳ needs a box |
| **C4** label ∩ join coverage | **< 0.90** | ✅ **PROVEN BOTH WAYS** |
| **C5** 20-step smoke | `metrics.json` absent, non-finite, or loss **CONSTANT** | ⏳ needs a box |
| **C6** anchor parity | shared `anchors.pt` unreadable/empty | ⏳ needs the file |

### 4.1 ⭐ C4 — the manufactured-negative guard, MEASURED both ways

⛔ **The failure it exists for:** a join sharing only **182** clips with v7.2 is
**182 / 4,572 = 3.98 %**. The arm trains on ~4 % of its intended supervision and reads as *"the
lever does not help"* — **a manufactured negative, worse than a crash because it looks like a
result.**

| join | clips | coverage | verdict |
|---|---|---|---|
| the real B1 TRAIN shape | 4,427 / 4,572 | **0.968285** | ✅ **PASS** — and it reproduces `D-B1TRAIN-JOIN-1`'s published 0.9683 |
| ⛔ the 182-clip disaster | 182 / 4,572 | **0.039808** | ⛔ **FAIL** |

⛔ **A zero is never reported as coverage.** If either side yields an empty id set the check returns
`INCONCLUSIVE` — *an empty intersection from a file that could not be READ is indistinguishable from
a genuine absence.*

### 4.2 C3 / C5 — the two traps

⚠️ The gate sets **`MSYS_NO_PATHCONV=1`**; without it MSYS rewrites `--remote-root` and the audit
reports **120/120 `MISSING_REMOTE`** — *"a clean, plausible, catastrophic-looking finding that is
pure artifact."* The gate additionally treats `MISSING_REMOTE == n_rows` as `INCONCLUSIVE`, never as
a finding. ⛔ md5 agreement proves **transfer, not function** (C99), hence `--verify-import`.

C5 asserts on **CONTENT**: finite losses that **actually moved** (a constant loss is a disconnected
graph). A stale `metrics.json` is deleted first, so a previous run's artifact cannot pass. ⛔ Never
`ls` — *a decode that raises into a pre-allocated memmap leaves a full-size file of zeros and the job
can still exit 0.*

---

## 5. The supervisor — audited, not assumed

All three scripts pass `bash -n`. **MEASURED: an audit of every executable child-spawning line finds
6 of 6 carrying `200>&-`** — the `nohup` trainer, both `python3 -` heredocs, and **both `sleep`s**.
⭐ That last clause is the one that matters: `200>&-` on the trainer alone was shipped once and
failed the same day, because the holder turned out to be the supervisor's own `sleep`. The
lock-holder diagnostic (`/proc/*/fd` + cmdline) is printed **inside the failure message**, since the
answer has twice been a process nobody suspected.

`assert_supervisor.sh` reads **`/proc/<pid>/cmdline`** and asserts the live pid is a `sup_refcv6`
**for this arm** — never `grep -c`, which self-matches its own echoed command line; and a pid file
alone would pass for a reused pid. The progress marker is `ZZ<arm>-<step>-<steps>-<errs>-<launch>ZZ`
with the error pattern **assembled at runtime**, so the command line cannot contain the literal it
searches for. The done-marker is written from `metrics.json` **content** and checked at startup, so
a finished run is never resurrected.

---

## 6. ⛔ What still needs the PI, and what is runnable the moment a GPU exists

**Runnable immediately, needing only a GPU:** **`V0`, `V0b`, `D`, `Db`** — arm **D** is one flag
(`--w-u0 0`) and free, and the `V0`/`V0b` pair establishes **the replicate floor every other arm's
bar is stated against**, so this cut is a prerequisite for the rest rather than merely the cheapest.
`launch_refcv6.sh --arm D` runs the gate and starts a supervised run with no further decisions.

**Blocked on the PI:**

1. ⛔ **`--w-tac-goal`'s value — PI queue item 10 / `D-TACGOAL-2`.** Blocks arm **C** only. A third
   tactical term at 0.05 either **raises `MANEUVER_WEIGHT` to 0.15** (lat/lon untouched, so the arm
   stays paired with the banked one) **or forces a `/3.0` re-split** (which changes lat/lon pressure
   and is a **separate arm, not a tweak**). Arm C is built parameterised; `arms.py` raises
   `PIDecisionRequired` rather than inventing a value.
2. ⚠️ **`--w-agent` — `1.0` has provenance but is NOT pre-registered; the PI should confirm it.**
   Blocks arms **A** and **B** only until then. ⭐ I searched rather than guessing, and the answer is
   specific: `refc_v3_train.py:2593-2594` records *"MEASURED on the tiny rig 2026-09-05: with
   `--agents head --w-agent 1.0`…"*, and `1.0` is the codebase's uniform value across refusal
   messages, `effective_weights.py:191`, `refc.py:3439`, `refcv5_preflight.py` and four tests.
   ⛔ **But this codebase annotates genuinely pre-registered weights inline and this one is not
   annotated:** `GOAL_POINT_WEIGHT_DEFAULT` reads *"1.0 is the **PRE-REGISTERED** launch value
   (`…/2026-09-06-goal-point/PREREG.md` §9)"*, while `AGENT_WEIGHT_DEFAULT` says only *"WP-6: the
   GT-supervised detection set loss"*. The absence is meaningful **because the convention exists**.
   ⚠️ And the run that source line describes **failed for an unrelated reason** (missing labels), so
   `1.0` is *the weight that arm carried*, not one shown to balance against the planner loss at
   108 M. ⇒ recorded as **`MEASURED (provenance) / NOT PRE-REGISTERED`**; `arms.py` still refuses to
   supply it silently, so the run record shows who chose it.
3. ⛔ **Compute.** A fresh pod for the 40 k arms. Nothing else here is hardware-blocked.

**Owed, and named rather than quietly carried:** refcv5-v2's per-family reference values (§5B of the
prereg) are **INHERITED** from the architecture review, not re-read from raw eval JSON by this
package. They decide no GPU-day — they decide which metrics the clauses name — but ⛔ **they must be
re-read from raw JSON before any refcv6 panel is scored**, because a transcription error there would
silently move a bar.

---

## 7. Operating conditions encountered

⚠️ **The G: mount was severely degraded throughout**, exactly as briefed. Observed and worked around:
`fatal: not a git repository` mid-command; `grep: Invalid request code` on files that exist; and the
documented split where **metadata resolves while content reads fail** — WP-B's `code/check_trainer.py`
was listed by `ls` but could not be copied in **15 consecutive attempts**, so this package ships its
own checker (which needed the constitutive-key classification anyway).

⛔ **Every zero was treated as a claim about the mount until a same-breath control succeeded.** The
first lever probe returned 0 for all four levers *and* for the `def main` control — correctly read as
`INCONCLUSIVE`, not as absence. The working method was: copy to local disk with retries, verify the
copy is non-empty, and probe the local copy; all analysis ran against a clone whose trainer is
**byte-identical** to the repo's (md5 `4e9c9834f20d6c50b35c163aae4ab4a8`).

---

## 8. Deliverable manifest

⚠️ **Everything below exists in the repo working tree and is STAGED. Nothing lives in only one
place**, other than the local scratch copies used to author it, which are redundant.

| artifact | repo path | state |
|---|---|---|
| pre-registration | `Project Steering/PREREG_REFCV6.md` | staged |
| register entries `E-REFCV6-A/B/C/D` | `Project Steering/GOALS_AND_CLAIMS.md` (appended) | staged |
| this write-up | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-10-refcv6-build/RESULT.md` | staged |
| argv single source of truth | `…/2026-09-10-refcv6-build/code/arms.py` | staged |
| one-variable checker | `…/code/check_one_variable.py` | staged |
| its mutation proof (6/6) | `…/code/mutation_proof.py` | staged |
| lever census | `…/code/lever_census.py` | staged |
| the bar as code | `…/code/verdict_refcv6.py` | staged |
| its drop-proof (13/13) | `…/code/verdict_dropproof.py` | staged |
| pre-launch gate | `…/code/prelaunch_gate.py` | staged |
| supervisor | `…/code/sup_refcv6.sh` | staged |
| supervisor assertion | `…/code/assert_supervisor.sh` | staged |
| launcher | `…/code/launch_refcv6.sh` | staged |
| measured evidence | `…/raw/lever_census.json`, `one_variable_probe.json`, `one_variable_no_pi.json`, `mutation_proof.json`, `verdict_dropproof.json`, `coverage_proof.json`, `gate_devbox_dryrun.json` | staged |

⛔ **ESCALATION — three items need a decision, not a merge:** (1) `--w-agent`'s value, (2)
`--w-tac-goal`'s weight (PI item 10), (3) a GPU. Arms **V0/V0b/D/Db** need none of them.
