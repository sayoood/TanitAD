# The v6 stage chain — written, and validated by EXECUTION

**2026-08-16 · branch `agent/arch-inf-20260803` · base `5725d95`.**
⛔ **CPU only. Nothing was trained, no GPU was touched, Thor was not contacted.** The live v6F S-W
run (step ~6,400 of 30,000, 27.18 s/step) is unaffected: nothing here changes a `state_dict` key, a
shape, an optimiser group or a schedule.

**Suite:** `3083 passed / 0 failed / 17 skipped / 2 xfailed` (437 s), MEASURED — baseline at brief
time was 2996; **+49 are this file's** (`tests/test_v6_chain.py`) and the rest arrived from other
agents' concurrently staged files.

⚠️ **Two later full-suite runs each showed 1–2 failures, and neither is real or mine.**
`tests/test_p8.py::test_pos_weight_default_is_auto` and
`tests/test_e_ag1_anchor_floor.py::test_no_situation_classifier_path` failed against
`scripts/train_p8_occupancy.py` and `scripts/e_ag1_anchor_floor.py`, whose mtimes (03:25:10 /
03:16:04) fall **inside those runs** — two other live agents rewriting their own files mid-suite.
**Both pass in isolation immediately afterwards (`52 passed, 1 skipped`), neither file is in
anything staged here**, and my own scope is green: the nine v6 test modules together are
**`215 passed`**. *(Stated rather than reported as a clean number: a suite result taken while
siblings are writing is a torn read, and calling it a regression would be a false alarm of exactly
the class this programme logs.)*

---

## 0. What was asked, and the one thing that changed under it

The brief asked for the chain S-W → S-T → S-S → S-J with the S-T **`goal`/`mlp` arm pair** as a
first-class requirement. **Mid-task, E-WC2 fired REFUSED**, and the arm pair stopped being the
default. The chain below implements the re-scoped shape:

> ⛔ **S-T's default is `--selector none`, `--w-select 0`.** A selector arm is reachable only via
> an explicit `--st-arms`, and is **refused at launch** — not merely left out of the plan — until
> the S-W latent surface reaches a threshold **pre-registered in code before that measurement is
> taken**.

Everything else in the brief stands and is implemented: per-stage `--out`, `--init-from` +
`--prev-gate` on every stage above S-W, refusal on any non-PASS gate, S-S's required
`sel_gap_revalidated` / `TACTICAL_revalidated`, Thor's constants, and validation by execution.

---

## 1. ⭐ THE HEADLINE: building the chain found FIVE more defects, and every one of them was
##    found by EXECUTING something

The prior was explicit in the brief — *six ladder defects in two days, all found by running a
transition, none by reading code*. It held again. **Five new defects, and the executed dry ladder
caught the first of them on its very first run.**

| # | defect | how it was found | class |
|---|---|---|---|
| **D1** | ⛔ `--i-know-this-is-the-control-arm` **does not clear the refusal that names it** | executing `preflight` with the flag the message tells you to pass | *the escape hatch a refusal documents does not work* |
| **D2** | ⛔ the `--selector X --w-select 0` refusal is **stage-blind**, so **every possible S-S launch command was wrong** | building the S-S launch line | *right in spirit, blind in practice* — the `strict=True` init blocker exactly |
| **D3** | ⛔ `--gate-probes <missing file>` is read **only after the training loop** — ~3.1 GPU-days paid, then death **before `stage_gate.json` AND before `summary.json`** | executing a dry-run with the flag | *analysis-time failure after the compute is paid for* (the `t1_eval.py` family) |
| **D4** | ⛔ `--n-candidates` differing between two stages breaks `--init-from` with a **shape mismatch that `load_stage_init`'s adjudication never sees** | the dry ladder's **first** execution | *a guard that does not cover the case it looks like it covers* |
| **D5** | ⛔ a `~` in any path makes the emitted launch line **silently create a directory literally named `~`** | reading the generated Thor commands | *a quoting rule that is correct for one argument class and wrong for another* |

Plus one **pre-existing structural gap** the chain could not be built around:

| **D0** | ⛔ **`--dry-run` ignored `--prev-gate` and `--init-from` entirely.** |
|---|---|

`dry_run()` exists to *"verify the pod launch BEFORE the corpus is mounted"* — and it skipped the
two flags the staged protocol is **made of**. A dry-run of S-T printed `dry-run OK` while the real
launch of the same command died on `Missing key(s): cand_score.*`. That is **C13 in the instrument
that is supposed to be the guard**: a pre-launch verifier structurally incapable of catching the
class of failure it exists to catch. It is fixed, and requirement 6 of the brief (*"dry-run every
transition with real `--init-from` and real gate files"*) was not satisfiable until it was.

### 1.1 D2 in full — the S-S launch pincer, MEASURED

Raw: `raw/ss_launch_pincer.json`. Against a real S-T checkpoint carrying four `cand_score.*` keys:

| the command you would write | what happened |
|---|---|
| `--stage S-S --selector goal` *(correct: carry the geometry, planner frozen)* | ⛔ **REFUSED** by preflight — *"the scorer would be built … and never receive a gradient"* |
| `--stage S-S --selector goal --i-know-this-is-the-control-arm` *(exactly what that message says to do)* | ⛔ **STILL REFUSED** — D1 |
| `--stage S-S --selector goal --w-select 1.0` | passes… but `for_stage("S-S")` returns `w_select 0.0`, so the launch advertises a loss that is **not in force** |
| `--stage S-S --selector none` | ⛔ **FATAL** at `load_stage_init`: `unexpected: ['cand_score.cand_bias', 'cand_score.goal_point.bias', 'cand_score.goal_point.weight', 'cand_score.log_tau']` |

**Every available S-S command was wrong**, and the pre-registered inert-scorer control arm at S-T
(`V6F_PLANNER_DESIGN` §4.1) was unlaunchable for the same reason.

**Root cause of D1, exactly:** `main()` registers the flag with `dest="control_arm_ack"`, and
`preflight` checked `getattr(a, "i_know_this_is_the_control_arm", False)` — an attribute argparse
never creates. It could only ever return `False`.

**Root cause of D2:** *"the scorer never receives a gradient"* is a defect only where the planner
group **trains**. `STAGE_GROUPS["S-S"] == ("layer_str",)`, so in S-S the planner is frozen **by
design** — and S-S must nevertheless carry `--selector` forward, because every stage saves the whole
`V6Stack` and a selector-less S-S stack turns the checkpoint's scorer into unexpected keys.

**Fixes** (both in `preflight`, both additive): read the correct dest (accepting either spelling),
scope the refusal to stages where `"planner" in stage_trainable_groups(stage)`, and add the
*opposite* refusal at S-S — `--w-select` there is a run row that lies about what moved.

### 1.2 D4 — the shape mismatch that never reaches the adjudicator

The dry ladder's first execution died at S-T:

```
RuntimeError: Error(s) in loading state_dict for V6Stack:
  size mismatch for cand_queries.weight: copying a param with shape torch.Size([8, 256])
  from checkpoint, the shape in current model is torch.Size([3, 256]).
```

My chain had emitted `--n-candidates` for the stages that carry a scorer and let S-W take the
trainer's default. ⚠️ **The important half is not my bug, it is where the failure surfaces:**
`load_stage_init` loads non-strict and adjudicates missing/unexpected KEYS against
`STAGE_MAY_INTRODUCE` — but `load_state_dict(strict=False)` still **raises** on a shape mismatch, so
a geometry change of this kind bypasses the whole allowance mechanism. The message is decent, so
this is not a silent failure; it is a launch-time crash on an edge nobody had run.
⇒ the fan size is a **ladder-wide constant**, emitted on every step including S-W, and pinned.

---

## 2. What was built

| artifact | path | state |
|---|---|---|
| **the chain driver** | `stack/scripts/v6_chain.py` (NEW, ~880 lines) | staged |
| **49 tests, incl. an end-to-end executed ladder** | `stack/tests/test_v6_chain.py` (NEW) | staged |
| trainer fixes D0–D3 + the Thor memory log | `stack/scripts/train_v6_staged.py` | staged |
| this writeup + 7 raw artifacts | `…/incoming/2026-08-16-v6-stage-chain/` | staged |

### 2.1 Why Python and not `*_chain.sh`

`stack/scripts/` holds five `*_chain.sh` files and **every one orchestrates evaluation tools**:
fixed steps, no branching, no state. The v6 ladder **branches on a gate verdict**, carries an
optional arm fork, and has already produced eleven defects on its edges. Its logic is
**adjudication**, and adjudication has to be pinned by tests. A bash chain would move the branch
points somewhere no test can reach — which is how the earlier defects survived. The driver emits
the exact bash lines an operator pastes, so nothing is lost.

### 2.2 ⛔ The chain never runs the four real stages in one process

Each stage is a multi-day GPU job. A long-lived orchestrator on a pod dies with its ssh session and
takes the ladder's state with it. **The state lives in the filesystem** — gate files and
done-markers — and `status` / `next` recompute it from scratch. `run` exists for the CPU dry ladder.
On Thor you use `next` + `commands` and launch one stage.

### 2.3 The subcommands

```
v6_chain.py plan        # the resolved ladder + wall-clock, as JSON
v6_chain.py admission   # ⭐ SEL-1's fired pre-registration + the S-W thresholds
v6_chain.py commands    # the copy-pasteable pod lines
v6_chain.py status      # per-step: gate verdict, done-marker, ckpt provenance
v6_chain.py next        # what may launch NOW, or the exact refusal (exit 3)
v6_chain.py manifests   # ONE supervise_run.sh manifest PER STAGE
v6_chain.py verify      # the /proc probe that cannot match itself
v6_chain.py run --dry   # the CPU ladder, ~21 s
```

⚠️ **torch is imported lazily.** `uv pip install <anything>` has twice replaced a pod's torch with a
wheel its driver cannot run; on that pod `plan`, `commands`, `status` and `manifests` still work.

---

## 3. ⛔ SEL-1 IS REFUSED — and the chain enforces it, it does not merely omit it

E-WC2 fired against `V6F_PLANNER_DESIGN` §5.2's pre-registration, **both outcomes committed in
advance**. INHERITED from the coordinator's measurement, banked in the chain as `SEL1_ADMISSION`:

| quantity | value | class |
|---|---|---|
| σ(2 s) per-axis | **4.7104 m** [3.8087, 5.6860] | MEASURED (REF-C-XL `refc-xl-30k` step 29999, 881 windows / 40 eps) |
| σ(6 s) per-axis | **18.3519 m** [15.8621, 20.9608], n 681 | MEASURED |
| **σ/ADE** | **9.9915** [7.4492, 13.5119] vs incumbent ADE 0.4714 | MEASURED |
| σ/oracle | **28.7307** vs oracle 0.1639 | MEASURED |
| REF-C-base replication | σ/ADE **9.6337** | MEASURED |
| pre-registered lines | funded ≤ 1.7 · **refused ≥ 3.0** | PRE-REGISTERED |

**It is not close: the interval's LOWER bound (7.4492) is 2.48× the refusal threshold.**
INCONCLUSIVE was never entered ⇒ **the capacity control is not needed to decide this.**

Estimator: point estimates `full_set`, intervals episode-cluster bootstrap over the 40 val episodes,
2000 draws. `overlapping_holdout_se` used nowhere. **Tier `T0-DIAGNOSTIC` — no T1 claim may cite it.**

### 3.1 ⭐ The reason matters more than the verdict, and the chain says so in its refusal

A **0-parameter constant-yaw-rate** goal reaches **σ(2 s) = 1.1888 m — 3.96× BETTER than the ridge
on frozen REF-C latents.** So the reading is **NOT** *"a 6 s goal point is unpredictable"*. It is
**"these latents are the wrong surface"**. SEL-1's **estimand survives** — a candidate-independent
reference still has no degenerate minimiser — but its **input does not**.

⚠️ **And the other half must be said with it:** even that kinematic floor lands at **σ/ADE 2.52**,
which is *still not FUNDED*. **Neither surface is good enough today.**

### 3.2 ⛔ No 6 s threshold is emitted, and that is a rule firing, not an omission

σ(6 s)/σ(2 s) = **3.7481**, past §5.3's 3× line ⇒ **REDERIVE**. The ratio form does not transfer, so
scaling the 2 s threshold across a 3× horizon change would be exactly the ≤2× extrapolation rule's
prohibition. `threshold_6s_m` is `null`, with its reason, and a test pins that it stays `null`.

### 3.3 ⚠️ Scope, stated so it cannot be overstated

This is the **REF-C** surface, **not** the frozen S-W latents §5.2 nominally names — those have
**never been dumped**. The **ratios** transfer (REF-C is the arm the 1.7/3.0 thresholds were derived
on, and the denominators are §3.1's own published fan references). The **absolute metres are
REF-C's**.

### 3.4 ⭐ THE PRE-REGISTRATION FOR THE S-W SURFACE — committed in code BEFORE the dump is taken

The only remaining input is one **~10–25 GPU-min** latent dump at the **S-W → S-T boundary** (the
instrument, the endpoint backfill and the val40 poses are all in place). Its thresholds are
`SW_LATENT_ADMISSION` in `v6_chain.py`, written today, **before** that measurement exists:

| σ(2 s) on frozen S-W latents | verdict | what it means |
|---|---|---|
| **≤ 0.80 m** | **FUNDED** | 5.89× better than REF-C's ridge and **1.49× better than the 0-param kinematic floor — from vision alone**. Selector arms reopen. |
| 0.80 < σ ≤ 1.41 m | **INCONCLUSIVE** | REFUSED stands. The first re-run is the capacity control, not a weight sweep. |
| **> 1.41 m** | **REFUSED stands** | |

⚠️ 2 s only — see §3.2. REF-C's own 4.7104 m adjudicates as `REFUSED` (test-pinned).

### 3.5 How the chain enforces it

`assert_selector_admissible` runs **first** in `assert_may_launch`, before anything expensive. A
step with `selector != "none"` is refused unless `<S-W out>/ewc2_sw_latents.json` exists and
adjudicates **FUNDED**. *"We left it out of the default plan"* is a preference; this is a fired
pre-registration, so it refuses the launch. **The arms are gated, not deleted** — `MLPCandidateScorer`
and `GoalDistanceScorer` stay implemented, dry-run-verified and reachable on a better surface, and
the opt-in ladder is executed end-to-end in `raw/dry_ladder_arms.json`.

⚠️ **The committed branch is the pre-registered fallback: `ANCHOR_GOAL` supervision** (PH0 +
`obstacle.offline` agent slots) — **not another cost.** The refusal message says so.

---

## 4. THE LADDER

```
S-W ──stage_gate.json──► S-T ──stage_gate.json──► S-S ──stage_gate.json──► S-J
    └──── ckpt.pt ──────►    └──── ckpt.pt ──────►    └──── ckpt.pt ──────►
 v6F-SW-30k           v6F-ST-10k              v6F-SS-8k              v6F-SJ-3k
   (LIVE)              10,000 st                8,000 st               3,000 st
                       lr 1e-4                  lr 1e-4                lr 3e-5
                    --max-horizon 60                              --max-horizon 60
                    selector: none            selector: none        selector: none
```

⛔ **`--out` is unique per step**, and `assert_plan` refuses a plan where two steps share one — the
trap is **unbuildable**, not merely caught. The trainer's stage-labelled `--resume auto` refusal is
the last line of defence; a chain that leans on it has already built the trap.

⚠️ The refusal quotes the reason the accidental barrier is worthless: it holds **only** because the
per-stage trainable-tensor counts happen to differ (**S-W 240 · S-T 80 · S-S 54 · S-J 374**,
MEASURED). One `STAGE_GROUPS` edit and it passes silently.

### 4.1 Wall-clock at Thor's MEASURED rate

**27.18 s/step** — MEASURED 2026-08-16, marginal over steps 6300→6400 of the live S-W run; three
statistics with different startup exposure agree to 0.5 % (50-step 27.21 · 100-step 27.18 ·
cumulative-with-startup 27.32). A40 = 20.46 s/step (1.33×).

| step | steps | hours | days | class |
|---|---|---|---|---|
| S-W (remaining 23,600) | 23,600 | 178.2 | **7.42** | MEASURED rate, ESTIMATED remainder |
| **S-T** | 10,000 | 75.50 | **3.15** | ESTIMATED |
| **S-S** | 8,000 | 60.40 | **2.52** | ESTIMATED |
| **S-J** *(optional)* | 3,000 | 22.65 | **0.94** | ESTIMATED |
| **total after S-W** | 21,000 | 158.6 | **6.61** | ESTIMATED |

⚠️ **Evidence class matters here.** The 27.18 s/step is S-W's, and **S-T/S-S/S-J have never run.**
S-T trains only `layer_tac` + `planner` on a frozen trunk and should be **cheaper per step**; S-T's
`--max-horizon 60` cuts windows/episode by ~42.6 %, which changes the epoch, not the step. ⇒ these
are **upper-ish estimates at S-W's rate**, and the rule is unchanged: **read `step_s` at step 500
and re-cost.** Nothing here may be quoted as a measured S-T cost.

### 4.2 Thor's constraints, applied — and one folklore rule that does NOT bind

| constraint | what the chain does |
|---|---|
| ⛔ 20 SMs **saturate at batch 8**; throughput flat at 12.3–14.1 windows/s across a 6× batch range | `--batch 8` is the default (`THOR_BATCH`); `--a40` switches to 16 + 20.46 s/step |
| ⛔ only `torch.cuda.max_memory_allocated()` is admissible — `mem_get_info` / `free` / `tegrastats` / `VmRSS` all misreport on unified memory, **in both directions** | the trainer now logs `cuda_max_mem_gb` **per log row**, with a note naming the four probes not to cross-check it against |
| ⚠️ *"each dataloader worker costs ~8.6 GB host RAM ⇒ few workers"* | ⛔ **DOES NOT BIND THIS TRAINER.** `train_v6_staged.train()` collates **synchronously in the main process** (`default_collate([ds_train[i] for i in idx])`) and never constructs a `DataLoader`. **There are zero workers to tune.** The host-RAM knob here is `--v2-lru`, defaulted to **6** — and on Thor host RAM *is* device memory, so the cache competes with the model rather than sitting beside it. *(Repeating a rule that does not apply is how a constraint list stops being read.)* |
| torch spawns ~113 threads/process | `OMP_NUM_THREADS=6` in every emitted line (the trainer also sets it defensively) |

---

## 5. THE EXACT PER-STAGE LAUNCH COMMANDS

Generated by `v6_chain.py commands`; full text in `raw/pod_commands.txt`. Substitute Thor's real
`$HOME` — ⛔ **do not paste a `~`**: every path is `shlex.quote`d (correctly — `--gate-off-reason
'PI directive: …'` is why), and quoting **suppresses tilde expansion**, so `mkdir -p
'~/experiments/…'` creates a directory literally named `~` (D5). The chain refuses a `~` rather than
expanding it, because `~` on the box that *generates* the line is not `~` on the pod that *runs* it.

```bash
# ---- S-T · 10,000 steps · 3.15 d at 27.18 s/step ----
mkdir -p /root/experiments/v6F-ST-10k && cd /root/TanitAD/stack && \
PYTHONPATH=/root/TanitAD/stack OMP_NUM_THREADS=6 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 -u \
  scripts/train_v6_staged.py --stage S-T --out /root/experiments/v6F-ST-10k \
  --steps 10000 --batch 8 --lr 0.0001 --n-candidates 8 \
  --init-from /root/experiments/v6F-SW-30k/ckpt.pt \
  --prev-gate /root/experiments/v6F-SW-30k/stage_gate.json \
  --max-horizon 60 --plan-wta-eps 0.05 --w-t1 1.0 \
  --v2-cache /root/data/physicalai-train-e438721ae894-w120-256x640cyl \
  --v2-val-cache /root/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
  --v2-lru 6 --frame-h 256 --frame-w 640 --frame-hfov 120 \
  --projection cylindrical --v2-subframe 176x624 --require-parity \
  > /root/experiments/v6F-ST-10k/train.out 2>&1 < /dev/null &

# ---- S-S · 8,000 steps · 2.52 d ----
#   …--stage S-S --out …/v6F-SS-8k --steps 8000 --lr 0.0001 --w-s1 1.0
#     --init-from …/v6F-ST-10k/ckpt.pt --prev-gate …/v6F-ST-10k/stage_gate.json
#   (no --max-horizon; stride_str keeps 94 windows/episode)

# ---- S-J · 3,000 steps · 0.94 d · OPTIONAL, only if S-T/S-S plateau ----
#   …--stage S-J --out …/v6F-SJ-3k --steps 3000 --lr 3e-05 --max-horizon 60
#     --w-t1 1.0 --w-s1 1.0 --plan-wta-eps 0.05
#     --init-from …/v6F-SS-8k/ckpt.pt --prev-gate …/v6F-SS-8k/stage_gate.json
```

⚠️ **Generating these from Git Bash on the dev box mangles absolute POSIX paths** — MSYS rewrote
`/root/experiments` to `C:/Program Files/Git/root/experiments`. Use `MSYS_NO_PATHCONV=1`, or generate
on the pod. *(Same family as everything else here: a tool silently rewriting what you typed.)*

### 5.1 The recommended sequence

1. `v6_chain.py next` — it prints what may launch, or the exact refusal (exit 3).
2. At the **S-W → S-T boundary**, take the **E-WC2-SW latent dump** (~10–25 GPU-min) and write
   `ewc2_sw_latents.json` next to the S-W run. That is the one pause worth taking, and its
   thresholds are already committed (§3.4).
3. `v6_chain.py commands --step S-T`, paste, launch.
4. `v6_chain.py verify --step S-T` — confirm the RUNNING process carries the flags you meant.
5. When it finishes: run the frozen battery, re-gate, then `next` again.

---

## 6. THE SUPERVISOR SEAM

⚠️ **`supervise_run.sh` SOURCES ITS MANIFEST ONCE, at supervisor startup**, and replays the
`TRAIN_CMD` it captured. Two consequences, both mechanised:

1. ⛔ **`v6_chain.py manifests` writes ONE MANIFEST PER STAGE, and every `TRAIN_CMD` is the
   TRAINER.** A supervised *chain* would replay stage 1 after a mid-ladder crash. A test parses the
   `TRAIN_CMD` line (not the file — the comment block deliberately *names* the chain to warn about
   it) and asserts `v6_chain` never appears in it.
2. **Editing a manifest under a live supervisor changes nothing.** The manifest says so in its own
   header: *edit → kill the SUPERVISOR first → kill the trainer → start a fresh supervisor*, and
   **verify by grepping the RUNNING PROCESS**.

⛔ **And the verification probe is not `pgrep -f` and not `ps | grep`.** Both put the searched token
into the searching process's own command line — measured three times in this programme, most
recently as a monitor that reported `Traceback CUDA out of memory` for a run that was healthy and
three minutes in. `v6_chain.py verify` emits a heredoc that reads `/proc/*/cmdline` (its own cmdline
is `python3 -`, containing neither token), builds the searched string by adjacent-literal
concatenation so **`train_v6_staged.py` never appears literally in the probe**, and emits an
**opaque `ZZ…ZZ` marker** disjoint from anything it searches for.

Each manifest also sets `TRAIN_MATCH='train_v6_staged\.py.*<run_id>'` — run-scoped, so a supervisor
waits for *its own* stage's trainer and can never adopt a sibling's.

---

## 7. VALIDATION BY EXECUTION

### 7.1 ⭐ The whole ladder, executed on CPU — `raw/dry_ladder_default.json` / `.log`

`v6_chain.py run --dry` — four real subprocesses, tiny geometry (8.71 M params, same wiring, same
seams), **real `--init-from` loads and real gate files**. **4 steps, 0 failures, 20.68 s.**

| edge | `introduced_keys` | `missing` / `unexpected` | trunk md5 after load | X3 |
|---|---|---|---|---|
| S-W → S-T | `[]` | `[]` / `[]` | `18a8d80dc91f` | pass |
| S-T → S-S | `[]` | `[]` / `[]` | `980a195aaa27` | pass |
| S-S → S-J | `[]` | `[]` / `[]` | `980a195aaa27` | pass |

⭐ **S-S → S-J preserves the trunk md5 exactly** (`980a195aaa27` both sides) — S-S trains
`layer_str` only, and that is now *demonstrated over the transition* rather than asserted.

**No fabricated PASS anywhere.** Each dry stage writes a **real** gate through `run_stage_gate`, and
it comes out **INCONCLUSIVE** because no frozen-battery probe ran — which is the honest verdict. The
ladder advances **only** through `--allow-inconclusive-gate` with a recorded reason, stamped into
every run's config. That is the real gate machinery being executed, not simulated.

⛔ **A dry gate can never license a real launch:** it carries `"_dry_run": true`, and
`assert_stage_precondition(dry_run=False)` refuses it (§7.2, case 4).

### 7.2 The opt-in arm ladder, executed — `raw/dry_ladder_arms.json`

With a **synthetic hypothetical FUNDED** surface file (`sigma_2s_m 0.75`, stamped
`_evidence_class: HYPOTHESIS`, *not a measurement*), the arm ladder runs
**S-W → S-T:goal → S-T:mlp → S-S → S-J: 5 steps, 0 failures, 26.11 s**, so the gated path is proved
alive rather than assumed:

| arm | `introduced_keys` | trunk md5 after load |
|---|---|---|
| S-T:**goal** | `cand_score.{cand_bias, goal_point.bias, goal_point.weight, log_tau}` | `18a8d80dc91f` |
| S-T:**mlp** | `cand_score.{cand_bias, fc1.bias, fc1.weight, fc2.bias, fc2.weight}` | `18a8d80dc91f` |

⭐ **Both arms stand on a BIT-IDENTICAL trunk** — same md5 — so if the arms ever run, the comparison
is on the same world model by construction, not by hope.

### 7.3 Every refusal, EXECUTED — `raw/chain_refusals.json`

*A guard nobody has seen fire is a guard nobody knows about.*

| # | attempted | verdict |
|---|---|---|
| 1 | advance on an INCONCLUSIVE gate, no override | ⛔ REFUSED |
| 2 | `--allow-inconclusive-gate` with **no recorded reason** | ⛔ REFUSED |
| 3 | advance on a **FAIL** gate with **every override flag set** | ⛔ REFUSED — *"no override for a FAIL"* |
| 4 | a **dry-run gate** licensing a real launch | ⛔ REFUSED |
| 5 | two steps sharing one `--out` | ⛔ REFUSED (at plan time) |
| 6 | launch into a dir holding another stage's `ckpt.pt` | ⛔ REFUSED |
| 7 | a selector arm while **SEL-1 is REFUSED** | ⛔ REFUSED |
| 8 | a selector arm on an **INCONCLUSIVE** S-W surface (σ 1.20) | ⛔ REFUSED |
| 9 | the pre-registered threshold table at its boundaries | ✅ 0.80 FUNDED · 0.8001 INCONCLUSIVE · 1.41 INCONCLUSIVE · 1.4101 REFUSED |
| 10 | a selector arm on a **FUNDED** surface (σ 0.75) | ✅ ALLOWED |
| 11 | S-S while the **capacity control has no gate** | ⛔ REFUSED (C6) |
| 12 | the same, **with `--unpaired-arm-reason`** | ✅ ALLOWED + recorded as *not attributable* |
| 13 | S-S with **no `--st-winner`** declared | ⛔ REFUSED |
| 14 | S-S built with a selector its ancestor did not have | ⛔ REFUSED |
| 17 | S-J on an S-S gate that **never revalidated** `sel_gap` / TACTICAL | ⛔ REFUSED |
| 18 | `--gate-probes <missing>` | ✅ now refused at **preflight** (was: after the run) |
| 15 | the `/proc` verify probe self-matching | ✅ cannot — no `pgrep -f`, no `ps|grep`, searched token never literal, opaque marker |
| 16 | a manifest supervising the chain | ✅ cannot — `TRAIN_CMD` is the trainer in all stages |

### 7.4 S-S's revalidation, mechanised

`STAGE_GATE_SPEC["S-S"]["required"]` carries `sel_gap_revalidated` and `TACTICAL_revalidated`
because S-S trains `layer_str` → `e_g_str` → `goal_head_tac(cond=e_g_str)` → `e_g_tac`, **the frozen
selector's only input**. The chain adds `assert_revalidations_present`, which refuses anything
consuming S-S's checkpoint on a gate that skipped them, **naming the seam and the committed
response** (the S-S′ planner-refit micro-stage) rather than saying "INCONCLUSIVE" and sending the
operator hunting.

⚠️ **The waiver is exactly the one the design commits to and no wider:** omitting the re-measurement
is waivable with a recorded reason; a measured **regression is a FAIL, and a FAIL has no override
anywhere in this ladder.** Both halves are test-pinned.

---

## 8. WHAT I DID NOT DO / OPEN

1. ⛔ **Nothing ran on a GPU and nothing ran on a real corpus.** Every number above is either a
   synthetic CPU smoke or arithmetic on someone else's measured rate. **The chain is validated as a
   chain; the stages are not validated as training.**
2. ⚠️ **The S-T/S-S/S-J wall-clock is at S-W's rate.** S-T should be cheaper per step. Re-cost at
   step 500.
3. ⚠️ **`--gate-probes` is not emitted by the chain**, deliberately: D3 showed that pointing it at a
   file that does not exist yet kills the run *after* the compute. The gate probes are supplied on a
   **re-gate**, and the chain refuses to advance until they are there.
4. ⚠️ **`V6_GO_PACKAGE.md` §2.1–2.5 now disagrees with this chain** — it predates all of today's
   learnings: `--batch 16` (A40), the old `v6-SW-30k` naming, no admission gate, no arm handling. It
   should be marked superseded by whoever owns it; I did not edit it.
5. ⚠️ **`assert_geometry_carry` reads the ancestor's `config.json`**, which does not exist until that
   stage has started. Before then it reports `ok: null` with the reason; `load_stage_init` remains
   the backstop.

---

## Deliverable manifest

| artifact | repo path | state |
|---|---|---|
| the chain driver | `stack/scripts/v6_chain.py` | **staged** (new) |
| 49 tests incl. the executed ladder | `stack/tests/test_v6_chain.py` | **staged** (new) |
| trainer fixes D0–D3 + `cuda_max_mem_gb` | `stack/scripts/train_v6_staged.py` | **staged** |
| this writeup | `…/incoming/2026-08-16-v6-stage-chain/V6_STAGE_CHAIN.md` | **staged** |
| executed default ladder (JSON + log) | `…/raw/dry_ladder_default.json`, `…/raw/dry_ladder_default.log` | **staged** |
| executed opt-in arm ladder | `…/raw/dry_ladder_arms.json` | **staged** |
| the refusal battery | `…/raw/chain_refusals.json` | **staged** |
| the S-S launch pincer (D1/D2) | `…/raw/ss_launch_pincer.json` | **staged** |
| the resolved plan + wall-clock | `…/raw/plan_thor.json` | **staged** |
| the pod-side launch lines | `…/raw/pod_commands.txt` | **staged** |

**Nothing was committed and nothing was pushed.**

## Escalations — requests, not notes

1. ⭐ **`--i-know-this-is-the-control-arm` was inert (D1), so the pre-registered inert-scorer control
   arm at S-T could never have been launched.** It is fixed here, but **`V6F_PLANNER_DESIGN` §4.1's
   "Pre-registered controls" list documents a flag that did not work** — worth a line wherever that
   list is maintained.
2. ⭐ **Three of today's five defects are the same root-cause class** — *a guard that is right in
   spirit and blind in practice* (D1, D2, D4), which is the class `RETRACTION_LOG` already carries
   for the `strict=True` init blocker and the optimiser param-group barrier. **D3 is the
   `t1_eval.py` class** (analysis-time failure after the compute is paid). I did not edit
   `RETRACTION_LOG.md` — another agent owns it this session — but four entries are warranted.
3. ⚠️ **`V6_GO_PACKAGE.md` §2 is now stale in five ways** (see §8.4) and is the document an operator
   would reach for at 3 a.m. It should point at `v6_chain.py commands`.
4. ⚠️ **The E-WC2-SW latent dump is the one GPU pause worth taking**, at the S-W → S-T boundary,
   ~10–25 GPU-min. Its thresholds are already committed in code (§3.4). Taking it *after* S-T starts
   costs a restart; taking it *before* costs 25 minutes.

**Evidence class:** every defect, transcript and refusal above is **MEASURED (ours)** and reproduced
by `stack/tests/test_v6_chain.py`. The SEL-1 numbers are **INHERITED** from the coordinator's E-WC2
run (REF-C surface, T0-DIAGNOSTIC) and are banked verbatim in `SEL1_ADMISSION`. The wall-clock is
**ESTIMATED** from a MEASURED S-W rate.
