# TanitAD — working agreements

## ⭐⭐⭐ THE REPO HAS MOVED — `D:\Projects\TanitAD` IS CANONICAL (PI, 2026-09-10)

**Sayed, verbatim:** *"I want you to move the whole projetc to the following folder
D:\Projects\TanitAD sicne we have a big problems with G mount. So move all trhe files we need,
checkout the repo and continue the work, do this for all the TanitAD agents and the research lab
inclduing all necessary files"*

⛔ **THIS SUPERSEDES EVERY `G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD` PATH IN THIS FILE
AND IN EVERY BRIEF.** Work in `D:\Projects\TanitAD`. ⛔ Do not start new work on G:.

### Why — the diagnosis, not the frustration

The section above records it in full: **G:'s `.git` mutable control files will not hydrate**.
`HEAD`, `config`, `index`, `packed-refs` and `refs/heads/*` all return **`Invalid request code`**
while directory listings, pack files and worktree files read perfectly. Git must read `HEAD` and
`config` to recognise a repository at all, so every git command there reports *"fatal: not a git
repository"* — a message about one unreadable file, dressed as a claim about the whole repo.
⛔ **Retrying does not fix it.** That is why the PI moved the project rather than waiting.

### What was moved, and how it was verified

| what | where | verification |
|---|---|---|
| the git object store | `D:\Projects\TanitAD\.git` | `fsck --connectivity-only` **rc 0, zero output**; HEAD matches; the only 8 absent objects were **unreachable** blobs |
| the working tree | `D:\Projects\TanitAD` | checked out from HEAD, **not** copied from G: — so no stale or half-edited file rode along |
| model artifacts | `D:\Projects\TanitAD-artifacts\` | refcv5-v2, refcv4b, refcv3, the compare harness, the A40 rescue bundle |

⛔ **The working tree was rebuilt from git, never copied from G:.** A copy would have carried
G:'s stale `CLAUDE.md` and reverted the day's commits — the same revert hazard the git-hygiene
section documents, arriving through a filesystem instead of an index.

### ⚠️ D: is exFAT, and three of its properties matter

1. ⛔ **No ownership recording** ⇒ git refuses with *"detected dubious ownership"* until
   `git config --global --add safe.directory D:/Projects/TanitAD` is set. ⚠️ **A `git status`
   that returns zero lines under that refusal is the REFUSAL, not a clean tree** — the same
   "a zero is a claim about the probe" trap, in a new costume.
2. ⚠️ **No symlinks and no hardlinks.** `git clone --local` cannot hardlink here, and any
   mode-`120000` entry materialises as a plain file.
3. ⚠️ **Large allocation unit**: 34,955 loose objects occupy **~38 GB allocated for 5 GB of
   actual data**. ⭐ `git gc` packs them and reclaims most of it — worth running.

⚠️ **It is an EXTERNAL drive.** Unplugging it reproduces the class of failure we just left, so
`C:\Users\Admin\tanitad-push\.git` stays as the working mirror and **GitHub remains the durable
copy**. Push often.

### ⛔ What was deliberately NOT rewritten

**233 tracked files contain a hardcoded G: path. 220 of them are historical banked artifacts** —
run logs, raw JSON, one-off probe scripts under `Implementation/incoming/`. ⛔ **Rewriting those
would falsify the record of where things actually ran.** A banked artifact says where it was
produced; that is a fact, not a setting. Only **live operational** files are updated.

⚠️ **And the code barely hardcodes the path at all** — it comes from the working directory. So a
session or agent launched with the right cwd needs no edits; one launched against G: will fail
loudly rather than silently, which is the correct behaviour.

## ⭐⭐⭐ RULE ZERO — A REFUTATION IS A WAYPOINT, NOT A DELIVERABLE (Sayed, 2026-09-05, HIGHEST PRIORITY)

**Sayed, verbatim:** *"I dont need refutaions, I need xcellent results … so if the results are not
enough continue automatically, detecting refutations is just not enough for this frontier
programme."*

And earlier the same day: *"our goals in TanitAD programme is not to refute hypotheses, its about
achieving excellent results and really driving autonmously with a reference implementation. Why im
saying this, we have too many refutes, and I have the feeling, we are satisfied sometimes if the
hypotheses is refuted rathern than solving the problem."*

⛔ **THIS RULE OUTRANKS EVERY OTHER PROCESS RULE IN THIS FILE.** The evidence rules below tell you
how to be *right*; this one tells you what you are *for*.

### The rule

1. ⛔ **"X is refuted" is NEVER a finished turn.** The complete form is: **"X is refuted — Y is the
   next lever — here is Y's RESULT"**, or, if Y truly cannot run, *"here is exactly what blocks Y."*
   A turn that ends at the refutation has delivered a diagnosis and called it a product.
2. ⭐ **WHEN AN ARM FAILS ITS BAR, CONTINUE AUTOMATICALLY.** Do not stop and report. Diagnose, pick
   the next lever, and **execute it in the same run**. An agent that stops to ask permission to keep
   working has spent its budget on the part that was already finished.
3. **You are done when one of two things is true**, and you must say which:
   * the result **clears its committed bar**, or
   * **every remaining lever is blocked** — on compute, on a corpus that does not exist, or on a PI
     decision — and you have **named each one and what would unblock it**.
4. ⛔ **A failed arm must leave behind the NEXT arm, not just its own verdict.** Before you write the
   verdict, answer: *what is the cheapest experiment that could still make this work, and did I run
   it?* If it was cheap and you did not run it, the turn is not finished.
5. **Prefer the lever with the largest measured effect, not the most interesting one.** Attribution
   exists to rank levers. If the arithmetic says one term dominates, go there — even when a more
   elegant hypothesis is available.

### ⛔ What this rule does NOT license — read this before it is misused

⚠️ **"I need excellent results" means BUILD until it works. It does NOT mean report good news.**
This rule makes the *stopping condition* stricter, never the *evidence bar* lower. Specifically:

* ⛔ **A committed criterion is still reported as written.** An arm that misses its pre-registered
  bar **FAILED**, and you say so — then you keep going. *(MEASURED 2026-09-05: an RL arm's formal
  exit was FAILURE on a separated `ade_m` +0.0362, and the correct behaviour was to report that
  plainly **and** carry the zero-training gate it produced into validation.)*
* ⛔ **Never move a goalpost after seeing the data**, never drop a control because it is
  inconvenient, and never promote a post-hoc finding without its own pre-registration.
* ⛔ **A refutation that ELIMINATES a candidate is real progress and must still be logged** — four
  eliminations in one day (head, goal, vocabulary, reward) are what localised refav1's defect to the
  cost geometry. The rule is not *"stop refuting"*; it is **"never stop AT a refutation."**
* ⚠️ **If continuing would need compute, spend, or an architecture change the PI has not
  authorised, STOP AND ASK** — that is a named blocker under (3), not a violation of this rule.

## ⛔⛔ BINDING — TOKEN BUDGET: SHORT, SCOPED, CHEAP AGENTS (PI, 2026-10-11)

**Why (MEASURED, `Project Steering/TOKEN_BURN_ANALYSIS_2026-10-11.md`, 35,293 calls):** the weekly limit was
exhausted in ~1.5 days (Oct 4–5) and the week before (Sep 26–27) — **two programme weeks lost**. Output is
**4 %** of the bill. **61 %** is re-reading huge contexts (calls above 300k context = 92 % of cost), and
**29 %** is subagent prompt caches (5-minute TTL) re-written after an agent waited 5–60 min on a Thor job.
96 % ran on Opus.

1. ⛔ **No agent waits on a long job while alive.** A job that runs longer than ~4 min ⇒ write a ≤ 2k-token
   `HANDOFF.md` (state, artifact paths, exact next step) and END. The job's completion launches a FRESH agent
   from the handoff. ⛔ Never `SendMessage`-resume a big agent after a gap longer than 5 min: that re-buys its
   whole context at 1.25×.
2. ⛔ **One agent = one scoped deliverable; hand off at ~200k context.** No agent owns a whole work package's
   lifecycle.
3. ⛔ **`model` on every Agent call: Sonnet by default.** Haiku for pure mechanics (staging, landing,
   polling). Opus only for root-cause diagnosis, pre-registration design and architecture.
4. **The Master Mind runs as ONE session, compacted at ~300k** (or restarted daily from `LOOP_STATE.md`).
   Agent hand-backs are ≤ 300 words plus file paths; the full report stays in the package.
5. **Polling is done by scripts, never by model turns** (Monitor / background tasks that wake on the event).
6. **Daily budget check:** `python tools/token_meter/token_meter.py --days 2` every morning. A day above 1/6 of
   the weekly budget ⇒ park waiting streams and move everything except diagnosis to Sonnet.
7. "≥ 5 parallel streams" (below) counts **streams of work**, not live agents. A stream whose job is running
   is **parked as a handoff file and costs zero tokens**.


## ⭐⭐ PROGRAMME REDESIGN (PI, 2026-08-22) — READ THESE TWO FILES FIRST

**`Project Steering/TANITAD_PROGRAMME.md` is the CONSTITUTION** — the 8 products,
the org (Research Lab · Master Mind · 4 FlyWheels), the uniform work-package
schema, the quality system, the TanitAD skills. It must be followed.
**`Project Steering/GOALS_AND_CLAIMS.md` is the LIVE REGISTER** of goals, claims
and hypotheses — ⛔ any session that asserts, supports, or refutes a claim
updates it IN THE SAME TURN; a fresh context reads it before acting.

Three binding consequences (also in auto-memory, so they survive resets):
1. **TanitAD is a frontier RESEARCH programme until its claims are proven: all
   open-source models/datasets licensed for research use are AVAILABLE. Never
   refuse or hedge on licensing.** Augmented/changed datasets are published
   PRIVATE on the PI's paid HF account; HF Pro GPU/inference/SSH-dev-box is
   first-class compute alongside Thor and the dev box.
2. The **Research Hub agent rotation is superseded**: ONE Research Lab agent,
   triggered daily ~08:00 by the Master Mind (not per-slot crons — those died on
   permission prompts). It is decoupled from production: literature research +
   small dev-box GPU experiments (incl. reverse-engineering recent papers),
   sequentially across Data Eng / Arch+Inference / Deploy+Opt / Opponent+Benchmarks.
3. The main agent is the **TanitAD Master Mind**: orchestrates the FlyWheel
   teammate agents, owns model design and the GitHub repo, maintains the
   scientific paper AUTONOMOUSLY, reviews/overrules teammate results, and sends
   multiple unprompted reports per day.

Sub-300M hierarchical 4-brain latent world model for autonomous driving. PI: Sayed.

## Source of truth (this rule exists because prose lied to us)

**`Project Steering/MODEL_REGISTRY.md` is the ONLY quotable source for model facts** —
architecture, params, training args, parity key, results, status. Any number in any report
cites the registry or the **raw eval JSON**, never a summary, changelog, or weekly report.

Three errors propagated for days because they were copied from prose:
- `flagship4b-phase0-30k` is the **no-speed ablation control** (2.918 m), NOT the deployed v1
  (that is **`flagship4b-speedjerk-30k`**, 0.452 m). The HF repo name invites this inversion.
- "REF-B v2 died at 22,600" — it did not; `metrics.json` says step **29999**. The *log* went stale.
- REF-A **I-JEPA's val number is unusable**: ~80% of val leaked into its train set.

If a doc and the registry disagree, the registry wins and the doc gets fixed.

**Never quote a learning-curve exponent bare.** Any exponent carries its **fit window, R² and n**,
or it is not admissible — and it may never decide a restart. Below **R² 0.80** there is no quotable
exponent at all; use the matched-step ratio. Never extrapolate more than **2×** beyond the fitted
range, and never compare exponents fit over different windows. *(Measured 2026-07-20: the same
`g_op_fwd_ade_m` log gives −0.387/−0.505/−0.564/−0.621/−0.738 depending on the window, all at
R² 0.09–0.58; v1's reference "−0.84" is the 1500–7500 window at R² 0.541, and on matched windows v1
and v3enc are statistically indistinguishable.)* Restart/continue decisions follow
`Project Steering/GATE_PROTOCOL.md` via `stack/scripts/run_gate.py`.

**Never quote an interval without its estimator.** The block historically labelled *"8-split
episode-disjoint jackknife"* is neither a jackknife nor a valid SE — it is
`overlapping_holdout_se`. The decision-grade interval is the **episode-cluster bootstrap** over the
40 val episodes (`taniteval/ci.py`); for two arms on the same windows use the **paired** version,
never a combination in quadrature.

⚠️ **It is not only an interval problem — `overlapping_holdout_se` also BIASES THE POINT ESTIMATE**,
because its central value is a **mean-of-split-means (`heldout`)**, not the **`full_set`** mean.
*(Measured 2026-07-25 over 27 dumps — **= 25 distinct arms, two double-banked pairs (C126,
`taniteval/results/dump_exclusions.json`)** — with raw per-window data: headline `ade_0_2s` shifts
**−6.67 % to +11.69 %**, **bidirectional — 11 inflated / 14 deflated over distinct arms (11/16 over
dumps), none flat**; on hierarchy seams up to
**×3.3**, and on paired deltas up to **×−4.15 including a SIGN FLIP**. It manufactured the program's
one "load-bearing" hierarchy seam: `ctx→tactical` +0.0439 → true **+0.0148**.)* Interval narrowing
is **1.107–3.100×, median 1.499×** (27 dumps = 25 distinct arms; the older "1.28–2.06×" came from only 10 and was
under-sampled). **Before trusting ANY pre-2026-07-25 number, check whether it is the `heldout`
split-mean or the `full_set` mean — `MODEL_REGISTRY.md` publishes both and they differ.** Blast
radius + per-arm corrections: `…/incoming/2026-07-25-jack-blast-radius/JACK_BLAST_RADIUS.md`.

⛔⛔ **A SEPARATED INTERVAL FROM A ONE-SEED ARM IS NECESSARY, NOT SUFFICIENT — THE PAIRED
BOOTSTRAP IS STRUCTURALLY BLIND TO TRAINING VARIANCE.** MEASURED 2026-09-05 on the v7-tiny rig
by an arm no SPEC asked for: **`A0b_replicate`** — A0's flags, A0's **seed**, run again, **zero
levers moved** (verified by argv audit) — produced **"separated" differences from A0 on 4 of 42
family cells**, a **9.5 % false-positive rate for `separated`** on that rig, about twice the nominal 5 %
(**CORRECTED 2026-09-26**, `RETR-2026-09-26-YAWMASK`: this rule was first written as *6 of 42 =
14.3 %*, but 2 of those 6 were yaw-rate cells from a shared metric that scored standstill jitter,
and they vanish once the cell masks tangent-less steps —
`…/2026-09-26-yaw-rate-mask/raw/claims/replicate_fp_rate_fixed.json`. First derived 2026-09-06 from
`panel_report.json` by recursing `A0b_replicate.paired_vs_A0`; restricted to the seven family rows
the panel's `VERDICT.md` tabulates at 2 s it is 2 of 14, formerly 3 of 14.
⚠️ The earlier "3 of 18 / ~17 %" form does NOT reproduce at any scoping, and `raw/NOISE_FLOOR.md`
cannot be its source: that artifact CRASHED mid-write on a cp1252 `UnicodeEncodeError` and carries
no numbers at all — a TRUNCATED artifact that reads like a complete one). The mechanism is
not a bug in `taniteval/ci.py`: the episode-cluster bootstrap resamples **EPISODES with the
models held fixed**, so it estimates *"would another draw of episodes say this?"* and never
*"would another training run say this?"*. Two runs differing in **nothing** cleared it.
⇒ **For any claim that a LEVER moved a metric, a separated CI is a necessary condition and not
a sufficient one.** The sufficient form needs a **replicate arm** (same flags, same or different
seed) so the lever's effect is read against the rig's own run-to-run noise floor. On the tiny rig
the cheapest next experiment is therefore **SEEDS, NOT WINDOWS** — more episodes narrow an
interval that was already answering the wrong question.
⚠️ **Scope it honestly, and do not blanket-void the register.** MEASURED: **8 rows** rest on a
tiny-rig separated result. They are not equally exposed — a **structural zero** is not a noisy
difference. `H-ECHO-4`'s deliberate-regression arm reads scene degradation **exactly +0.0000 with
CI [0, 0]** because a constant-image arm *cannot* degrade under a scene intervention; that is an
identity, not an estimate, and no seed changes it. The exposed claims are the ones asserting a
**difference between two trained arms**. Each needs re-reading against a replicate before it is
quoted as a lever effect; none is retracted by this rule alone.
⭐ Same family as the two estimator rules above — *never quote an interval without its estimator*,
and *`overlapping_holdout_se` biases the point estimate* — with the object swapped: here the
estimator is correct and its **question** is narrower than the claim being hung on it. Registered
as `H-ESTIM-SEED-1`; instrument `…/2026-09-05-withheld-bank-panel/raw/NOISE_FLOOR.md`.

⛔⛔ **AND ON A STOCHASTIC PLANNER THERE IS A THIRD VARIANCE THE SAME INTERVAL IS
BLIND TO — INFERENCE.** MEASURED 2026-09-05 on refav1: iCEM **samples**, so **the same
checkpoint evaluated twice does not give the same answer** — no retraining involved — and
this rig's **seed floor is ≈0.30 m ADE**, while *"the programme has been comparing
single-seed arms"* (`D-REFAV1-SEED-GOAL-MISMATCH`). ⇒ **Three different questions ride on
one interval:** *would another draw of EPISODES say this?* (what the bootstrap actually
answers), *would another TRAINING RUN say this?* (`H-ESTIM-SEED-1`), and *would another
INFERENCE RUN say this?* (this one). **Name which one your separated CI has answered.**
⇒ For any arm whose planner samples, a replicate must vary the **inference** seed, and an
effect smaller than that floor is not an effect. *(The refav1 A/B's +0.8927 m ADE clears it
at 11.2× and is reported that way; a 0.2 m "win" on that rig would not have been one.)*

## Briefing a subagent — the contract

Every subagent brief MUST carry the preamble in
`Project Steering/AGENT_OPERATING_STANDARD.md`. Its three binding rules:

1. **Stage, never push.** Agents `git add` their deliverables into the working tree and
   **never commit and never push**. They must NOT leave work only on a pod or only in a
   worktree. *(The old "commit nothing" default stranded REF-B v2's architecture, the entire
   TanitEval harness, the pod ops bundle, and 486 lines of TanitResim — each on a single disk.)*
2. **End with a deliverable manifest** — every artifact and **where it lives**
   (repo path / pod:path / worktree). Stranding must be visible in the report, not discovered
   in an audit months later.
3. **Escalate integration, don't write "please merge" into a doc.** An orthogonality instrument
   sat unmerged for **10 days** because the request lived in a README nobody re-read.

## Traps — the catalogue lives in `Project Steering/TRAPS.md` (read the entry before acting)

Moved out of this file on 2026-10-11 (token budget, P6): every session and agent re-read 13k tokens of it on every call. Also available as the `tanitad-traps` skill. Index (full text, measurements and fixes in TRAPS.md):

- A POLLING MONITOR WHOSE FILTER CONTAINS THE PATTERN IT SEARCHES FOR WILL MATCH ITS OWN
- `pgrep -f <trainer>` / `pkill -f <trainer>` self-matches your own ssh command
- `PYTHONPATH=/workspace/TanitAD/stack` is REQUIRED
- Never judge pod disk with `df`.
- `step_s` in trainer logs is ACCUMULATED over `--log-every`
- Moving multi-GB files between pods: POD→POD DIRECT SSH WORKS — the long-standing "pods cannot
- A RunPod volume resize stops the pod and reassigns its SSH port
- `torch` spawns ~113 threads PER PROCESS, and concurrent eval arms then make NO PROGRESS —
- `git add <exact path>` can SILENTLY NO-OP on a file in a newly created directory
- PODS HAVE NO GIT CREDENTIALS — `git fetch` on a pod HANGS (not fails), and the checkout's
- A pod's `stack/` checkout drifts silently and a launch from it resurrects fixed bugs.
- On the Jetson Thor, THREE standard memory probes all lie — and they lie in BOTH directions.
- Thor inverts both A40 batching instincts.
- `memory.usage_in_bytes` (cgroup v1) COUNTS RECLAIMABLE PAGE CACHE — it is NOT memory
- `supervise_run.sh` SOURCES ITS MANIFEST ONCE, at supervisor startup — not per relaunch.
- A supervisor whose run never wrote its DONE-MARKER will RESURRECT the finished run the
- Restarting a supervisor immediately after killing the old one RACES ITS `flock`
- `uv pip install <anything>` CAN SILENTLY REPLACE TORCH WITH A WHEEL THE DRIVER CANNOT RUN.
- `CUDNN_STATUS_NOT_INITIALIZED` on a healthy pod usually means CUDA NEVER INITIALISED —
- AN ANALYSIS-TIME IMPORT THAT FAILS AFTER THE ROLLOUT DESTROYS THE RUN'S OUTPUT WHILE THE
- THE GOTTY PTY DROPS OUTPUT IN A SYSTEMATIC PATTERN, AND A PLAIN `base64` PULL GOES SILENTLY
- A DENSE WINDOWED TENSOR DUPLICATES EVERY FRAME `W` TIMES AND SILENTLY SWAPS — AND THE
- AN ARTIFACT'S COST IS THE COST OF THE FILE ITS
- A DERIVED CONSTANT SILENTLY CHANGES THE EXPERIMENT WHEN ITS INPUT CHANGES.
- A DECODE THAT RAISES INTO A PRE-ALLOCATED MEMMAP LEAVES A FULL-SIZE FILE OF ZEROS —
- THE PINHOLE FOV FORMULA IS WRONG ON A CYLINDRICAL PROJECTION.
- A PROBE THAT TUNES ON THE DATA IT SCORES WILL MANUFACTURE A RESULT — AND THE ONLY
- A NEGATIVE FROM A LINEAR PROBE IS NOT A NEGATIVE ABOUT LEARNABILITY.
- `git ls-tree -r` SILENTLY TRUNCATES ON THE G: MOUNT — IT EXITS 0, AND IT TRUNCATES
- A SEARCH TOOL WILL REPORT "NO MATCHES" FOR FILES IT COULD NOT OPEN — AND
- SEARCH THE COMMA-FORMATTED FORM OF EVERY NUMBER.
- A number carries its ARM and its ARTIFACT PATH, or it is not quotable — and
- A CORRECT FORMULA APPLIED UNDER THE WRONG UNITS READS EXACTLY LIKE AN ANSWER — AND AN
- Verify before alarming.
- **Git hygiene** — commits sweep the whole index; phantom deletions / stale index blobs; `git commit -- <pathspec>` segfaults; use the coordinated lander (`land.sh`) or `scoped_commit.py`.


## Operating standard — raised by Sayed 2026-07-21

The program's pace goes up, and so does the bar. Five rules, each with the failure that earned it.

**1. State the evidence class or don't state the claim.** Every number carries
`MEASURED (ours + artifact path)` · `PUBLISHED (cited)` · `INHERITED (another agent/doc, NOT
re-verified)` · `ESTIMATED` · `HYPOTHESIS`. **A claim that decides a GPU-day must be MEASURED or
PUBLISHED — never INHERITED.** *(2026-07-21 alone: five retractions, every one from quoting a
faster-moving source than the harness. "v1.6 is best-in-program" was a **trainer log**, ~10 % optimistic
vs `eval_*.py`. Trainer val watches a curve; only eval output is quotable.)*

**2. Absence found at ONE location is not absence.** Before writing "X does not exist", probe a second
path, a second name, and the tool that owns the fact. *(Cost this session: the Vulkan ICD is in
`/etc/vulkan/icd.d/`, not `/usr/share/` → "our pods cannot render" stood for **12 days** and blocked
AlpaSim + CARLA. `ps -C python3` returns EMPTY for a healthy job because pods run
`/workspace/venv/bin/python` → a near-miss "the VLM job is dead". `obstacle.offline` — 3D agent tracks
on **97.44 %** of the corpus — was declared non-existent for days, and **IS NOW READ**, so the
programme reads **6** of 36 features.)*
⚠️ **This count went stale FOUR TIMES in this very sentence: "2 of 36" → 4 (2026-07-26) → 5 → 6
(both 2026-08-16).** ⛔ **It is now PINNED — do not hand-edit it. `stack/tests/test_physicalai_feature_readset.py`
asserts the counts AND the exact feature names against source, and its failure message names the
documents to update.** The stale count had reached **14 documents (17 sites) plus a code docstring**
(`stack/tanitad/data/bev_raster.py:12`) — a stale absence-claim living inside the rule that warns
about stale absence-claims, four times over.
⇒ **ROOT CAUSE (both the rot AND why 5 was wrong): "our ingest" was never DEFINED.** There are three
legitimate read-sets, so **always state the layer, never the bare phrase "our ingest"**:

| layer | count | features |
|---|---|---|
| `physicalai_r0.py` (r0 clip selection) | **2** | `egomotion`, `camera_front_wide_120fov` |
| `physicalai.py` (the **episode build**) | **5** | + `camera_intrinsics`, `sensor_extrinsics`, `vehicle_dimensions` |
| **program-wide** (incl. the pod-side join) | **6** | + `obstacle.offline` |

`obstacle.offline` is deliberately OUTSIDE the episode build — the join is a pod-side step
(`stack/scripts/build_obstacle_join.py`), which is why `grep obstacle stack/tanitad/data/physicalai.py`
still returns zero. ⚠️ **The fourth rot and the 2026-08-03 stale-blocker case have the SAME trigger:**
`obstacle.offline` landing, and neither the prose count nor the "UNAVAILABLE" blocker lines being
revisited. Sweep: `…/incoming/2026-08-16-stale-blocker-sweep/STALE_BLOCKER_SWEEP.md`.
**And the answer to what is in the other 30 is now settled, at five independent probes** *(it read
"the other 32" until 2026-08-16 — the complement rots in lockstep with the count, so the pinning test
covers the denominator too)***:** there is
**no map, lane graph, junction annotation, roundabout label, traffic-light feature or route/goal signal**
in PhysicalAI-AV — the card says verbatim *"we do not include open maps data"*, and `obstacle.offline`'s
enum over **87,481 cuboids is 10 classes, all dynamic agents**. Stop re-asking; the strategic-brain
topology must come from AlpaSim or an external corpus. *(Also settled: `egomotion` carries **no
lat/lon/GNSS** — coordinates are clip-local metres, so **OSM map-matching on our traces is impossible**.)*

⛔⛔ **AND THAT PARAGRAPH IS A LAYER-1 FACT THAT HAS ALREADY BEEN QUOTED TO SETTLE A
LAYER-3 QUESTION.** It is true of the **published PhysicalAI-AV feature set** and SILENT about our
own **AUGMENTED v7.x label release**. MEASURED 2026-09-06 on
`…/incoming/2026-09-04-v72-label-release/raw/s2_labels_v7.2_train.jsonl.gz` (schema
`s2-geom-v7`, vocab `v7`, **4,572 records, one per clip**): **779/4,572 records carry a GT
traffic-light tactical goal WITH THE COLOUR ATTACHED** — `g_tac.goals.TRAFFIC_LIGHT_REACT_RED`
**376/4,572**, `_GREEN` **363/4,572**, `_YELLOW` **22/4,572**, colourless `TRAFFIC_LIGHT_REACT`
**18/4,572** — each carrying `state`, `object_kind: "traffic_light"` and `provenance:
"vlm-cot"`, extracted from the Alpamayo CoT by `stack/tanitad/data/alpamayo_semantics.py` and
frozen into the vocabulary by `stack/tanitad/models/vocab_v7.py`. ⇒ **"PhysicalAI-AV publishes no
traffic-light feature" is TRUE; "the programme has no traffic-light label" is FALSE.** The PI
corrected exactly this conflation on 2026-09-06.
⚠ **So state the LAYER here too** — published corpus / episode build / **augmented v7.x label
release** — exactly as the feature-count table above forces for the read-set. This is the same
rot the pinning test was built for, one table down. *(What IS true, measured the same day and far
more damaging: that label **never reaches training** — `D-TLIGHT-1`, and
`stack/tests/test_tactical_label_reach.py` now pins it.)*

**3. Finish before you start. An artifact on one disk or in one agent's context is NOT done.**
Definition of done = **in the repo, staged, with its provenance**. *(LAL-v2 anticipation: implemented,
tested, **unmerged 12 days**. An orthogonality instrument: **10 days**. TanitEval, REF-B v2's
architecture, the pod ops bundle — each stranded on a single disk.)*

**4. Retractions are the learning mechanism — log the ROOT-CAUSE CLASS, not just the correction.**
`Project Steering/RETRACTION_LOG.md` is append-only and **must be read before asserting in a known
class**. A retraction with no class taught nobody anything.

**5. Aim above the published state of the art, and settle conflicts with experiments, not deference.**
When ambition meets inconvenient evidence, the answer is the **cheapest discriminating experiment**,
pre-registered with **both outcomes committed in advance** — not a scoped-down goal. *(The "strategic
choice is a ~2 % lever" refusal was **confounded**: REF-C evaluates with `nav_cmd=None`, so a decoder
that never had a working route input learned the marginal. I nearly designed the hierarchy away on it.)*

**Orchestration.** Parallel streams are the default, but: every brief carries a **priority order** so a
killed agent still yields value; agents **bank incrementally** rather than holding a final synthesis;
and **fan-out is capped** — uncontrolled sub-spawning exhausted the weekly API budget on 2026-07-21 and
cost three agents' work.

## Invariants

- **`Keys.txt` is git-ignored — NEVER commit it.** Read tokens in place
  (`grep -oE 'hf_[A-Za-z0-9]+'`); never copy, print, or write them to args.
- **Agents never commit to `main`** and never edit `Project Steering/Mission Plan.md`.
- **Parity is sacred:** the canonical train corpus is `physicalai-train-e438721ae894`
  (2376 episodes) with skip-hash `f09e44db`. Anything that re-selects episodes breaks
  cross-arm comparability and must be refused.
- **Never add GPU/RAM load to a pod that is training**, and never eval on a training pod.
- Full suite lives at `stack/` — `pytest -q` must stay green before any commit.

## ⛔ NEVER IDLE — a report is not work (Sayed, 2026-07-29, flagged THREE times)

**The failure pattern:** check the fleet → find the top item blocked (pod down, checkpoint not
ready, PI decision pending) → write a well-organised report about being blocked → re-arm the loop.
**The report feels like output. It is not.** Twice in one session a turn ended with *"next step is
implementing X"* and no implementation followed.

**The root cause:** treating the priority list as a **queue** — top item blocked ⇒ wait. It is a
**PRIORITY ORDER, NOT A DEPENDENCY CHAIN.** Item 1 being blocked says nothing about items 2–12.

**The rule:**
1. ⛔ **Never end a turn having only reported.** If the top item is blocked, **drop to the next
   UNBLOCKED item and execute it in the same turn.**
2. **`Project Steering/BACKLOG.md` is the live pull-list.** When the headline work is gated, pull
   from it. **Gated ≠ idle.**
3. **"Blocked on the PI" blocks ONE item, not the programme.** There are always 0-GPU items:
   implementation, banking stranded artifacts, pre-registrations, instrument fixes, doc corrections,
   re-verification of refuted-but-load-bearing claims.
4. A progress report is the **last 10 % of a turn**, never the whole turn.
5. ⚠️ **Long heartbeats are for GENUINELY gated states only.** Do not use a long heartbeat to make
   idling look deliberate.

*(Same class as "A report is not a launch": when an agent files, refill the pod in the SAME turn
before writing the summary.)*

### ⛔ THE RE-ARM IS NOT AN ACTION (Sayed, 2026-08-02, FOURTH flag)

**The specific failure:** I ended a turn with a status report followed by `ScheduleWakeup`, and
called that "continuing the loop". **Scheduling the next turn is not work in this turn.** It is the
same idling the rule above forbids, wearing a timer.

**Hard rules, mechanically checkable before ending any turn:**

1. ⛔ **Never call `ScheduleWakeup` in a turn whose only outputs were a report, a probe, or a
   commit of documentation.** The wakeup is admissible ONLY after something was *executed* —
   code changed, a job launched, an artifact produced, a measurement taken.
2. ⛔ **"I'll do X next iteration" is banned.** If X is unblocked, X happens in THIS turn. If X is
   blocked, name the blocker and **execute the next unblocked item**, then report both.
3. **A blocked headline item does not license a short turn.** `BACKLOG.md` exists precisely so a
   gated turn still ships something. Gated ≠ idle.
4. **Before ending a turn, ask: "what did I CHANGE?"** If the honest answer is "nothing — I looked
   and I described", the turn is not finished.
5. ⚠️ A **PI decision** blocks that ONE item. Provision, spend, and publish are the PI's; every
   implementation, measurement, instrument fix and banked artifact around them is mine, and none of
   them wait.


## ⛔ BINDING — A GOAL INPUT IS ADMISSIBLE, BUT MUST NOT CARRY THE SITUATION CLASSIFIER (Sayed, 2026-08-03)

**Sayed:** *"yes a goal input is admissible, at the same time, we need to be careful not to include
the result of the situation classification in the goal input."*

| | |
|---|---|
| ✅ **Admissible at inference** | a goal / route signal — including a **predicted geometric goal point**, which the literature shows is the lever that actually works (categorical command +0.2 PDMS; goal point **+4.7**). |
| ⛔ **NOT admissible inside it** | the **output of the situation classifier**, in any form — class posterior, argmax, embedding, or any feature derived from them. |

**Why the second half matters more than the first.** If the goal input carries the situation
classifier's output, then:
1. **Attribution dies.** A planner improvement can no longer be assigned to the goal or to the
   classifier — they are one path. That is the `--v2` conflation failure again (ten levers on two
   axes, result non-attributable), and the C6 confound again.
2. **The classifier stops being independently evaluable.** Its errors enter the planner and come back
   as planner metrics; we would be scoring a loop.
3. **It is the nav-echo defect in a new costume.** MEASURED today: flagship v1's route head is an
   exact bijection of the nav we feed it (369/369 and 81/81) and scored **1.0000** — an echo of its
   own input read as skill. A goal built from the classifier would let the *planner* echo the
   *classifier* the same way.

**The design rule:** the goal path and the situation path stay **information-disjoint at inference**.
State, for any arm, what the goal is computed from — and if a shared trunk feeds both, say so and
justify why that is not a back door.

⚠️ **The admissibility check to run:** for any goal signal, ask *"could this have been computed from
the situation classifier's output?"* If yes, it is inadmissible until shown otherwise. Same family as
the leak test in [[the vision-only rule]]: **ask whether an input at inference contains something the
thing being measured also produces.**

⚠️ **A supplied route is optimistic by construction on PhysicalAI** — our only route supplier there is
the ego's own future path. Prefer a **predicted** goal, which sidesteps both this and the
vision-only rule.

## ⛔ BINDING — LABELS MAY USE EGO; INFERENCE IS VISION-ONLY (Sayed, 2026-08-03)

**Sayed, verbatim:** *"for ground truth data of scenario classification you can use both ego and
other label, for inference only vision."*

| stage | what may be used |
|---|---|
| **Ground truth / label derivation** | ego state, other agents, maps, future poses — anything. Labels are built offline; privileged signals are FINE here. |
| **Inference** | ⛔ **VISION ONLY.** No ego state, no privileged channel. |

⇒ For the scenario classifier the deployable arm is **`head_img` (image-only)** — NOT `head_img_ego`
and NOT `head_ego`. This supersedes both earlier candidates: the ego-only swap (already ruled out,
*"no ego heads"*) **and** score-level fusion of image+ego, because that is still ego-at-inference.

⭐ **WHY THIS PROBABLY DISSOLVES THE SITCLF ANOMALY — verify, do not assume.** The situation labels
are derived from **ego dynamics** (`stack/tanitad/data/situations.py`). If a label is a function of
ego state, a classifier *given* ego state at inference is partly reading **the label's own source** —
that is a **LEAK, not a capability**. It would make the banked ranking (`head_ego` 0.0697 >
`head_img_ego` 0.0525 > `head_img` 0.0376) a measure of leak magnitude rather than evidence about
vision, and it makes `situations.py:19`'s *"VISION ADDS NOTHING OVER EGO STATE"* **unfalsifiable as
stated**. Establish the label provenance from source (two probes, quote file:line) before any further
sitclf claim.

⚠️ **"Vision scores worse" is never a reason to reopen ego at inference.** If the vision-only arm is
weak, the finding is *how much*, *why*, and *what would fix it*.

⚠️ **Generalise the test, not just the case:** for ANY head, ask whether its inputs at inference
include something the label was derived from. That is the same family as the C6 confound (a decoder
compared on its marginal) and the REF-A I-JEPA leak (~80 % of val inside train).

## ⛔ BINDING — AT LEAST FIVE PARALLEL STREAMS, ALWAYS (Sayed, 2026-08-03)

> ⚠️ **Amended 2026-10-11 by the TOKEN BUDGET rule above:** streams are counted as WORK, not live agents; a stream waiting on a job is parked as a handoff file.

**Sayed:** *"dont idle, always have at least 5 streams in parallel, use the loop, goal and workflow
concepts of claude."*

**The rule:** at every point in the programme there are **≥5 work streams in flight simultaneously**.
One agent running while the orchestrator waits is IDLING WITH EXTRA STEPS. A turn that ends with
fewer than five live streams and unexhausted backlog is not finished.

**The three mechanisms, and when each is right:**

| concept | use it for |
|---|---|
| **Loop** (`/loop` + a CRON DRUMBEAT) | the heartbeat — re-enters, re-checks the fleet, and picks the next unblocked item. Must be a COMPLETE handoff: a fresh session reads only the cron prompt. Rewrite it every run. |
| **Goal / priority order** | the PI's numbered list. It is a **PRIORITY ORDER, NOT A DEPENDENCY CHAIN** — item 1 blocked says nothing about 2–12. Drop to the next unblocked item IN THE SAME TURN. |
| **Workflow** (`Workflow` tool) | fan-out across independent streams, each with its own adversarial verify. Prefer `pipeline()` so a stream verifies as soon as it finishes; use a barrier only when a stage genuinely needs ALL prior results. |

**How to keep five alive without thrash:**
1. **Never let a stream finish without launching its successor.** When an agent reports, commit its
   work AND start the next item in the same turn.
2. **Streams must be INDEPENDENT.** Five agents editing the same file is one stream with a merge
   conflict. Partition by directory//concern, and say which files each owns.
3. **Mix horizons:** long compute (training, engine builds), medium (experiments), short (docs,
   instrument fixes, banking stranded artifacts). A GPU-gated turn still ships the short ones.
4. **A blocked stream is replaced, not waited on.** `Project Steering/BACKLOG.md` is the pull-list.
5. ⚠️ **Fan-out is still capped** — uncontrolled sub-spawning exhausted the weekly API budget on
   2026-07-21 and cost three agents' work. Five to eight concurrent is the band; scale depth, not
   breadth, beyond that.

*(Root cause: the orchestrator repeatedly ran ONE agent and then reported. Sequential delegation is
still sequential. The programme's throughput is set by how many independent questions are being
answered at once, not by how carefully one is.)*

## ⛔ BINDING — EVERY NUMBER CARRIES ITS EVAL TIER (EVAL_DOCTRINE.md, 2026-08-06)

**T0** (teacher-forced / true-future-conditioned) is a **WM diagnostic — NEVER "driving
performance"**. **T1** (action-closed loop: the model conditioned on its OWN actions) is the
**PRIMARY tier for any capability claim**. **T2** (re-perception sim) is not provisioned.
*Why this exists (MEASURED, §1.12): open-loop lateral skill was an ACTION ECHO — S-curve
reproduction 97.9 % open-loop, 0.0 % hold-action, ~5 % closed-loop.* A registry row or report
quoting a number without its tier stamp is incomplete; comparisons across tiers are invalid.
Instrument: `taniteval/tools/t1_eval.py` (E1.2).

## ⛔ BINDING — RESEARCH BANKING: BANK THE PRIMARY, OR THE DELIVERABLE IS INCOMPLETE (Sayed, 2026-08-18)

**A research deliverable that cites a paper it did not bank is incomplete.** Every literature pass
downloads its primary sources into `TanitAD Research Lab/Library/`, indexes them, and extends the
knowledge base incrementally.

```bash
python tools/kb_add.py <arxiv-id-or-url> --tag <topic> --note "<the finding, 1 line>"     --cited-by "<repo path of your report>"
python tools/kb_add.py --verify     # sha256 every entry against disk
```

* ⭐ **Why it binds.** The 2026-08-18 frozen-encoder review had to mark **five numeric claims
  `PUBLISHED-SECONDARY`** — read from aggregator summaries and therefore **inadmissible for the
  registry or the paper** — purely because the primaries were not to hand. Banking the PDF is what
  converts SECONDARY into PRIMARY. A URL is a claim about the internet; a banked sha256 is a claim
  about a file we hold.
* **The seven `KNOWLEDGE_BASE.md` files stay the FINDINGS layer** (`[YYYY-MM-DD] [source] finding —
  impact — link`). The Library is the EVIDENCE layer beneath them. A `[PUBLISHED]` entry should cite
  the **library key**, not only a URL.
* ⛔ **Never hand-edit `LIBRARY.md`** — it is generated from `library.json` by `--reindex`.
* ⛔ **Verify by CONTENT, never by presence.** `--verify` re-hashes; a file that exists with changed
  bytes is worse than a missing one.
* ⚠️ **The tool banked garbage on its first run and reported success** — arXiv's Atom feed opens with
  a feed-level `<title>` echoing the query, so an unscoped regex filed all 11 papers under
  *"arXiv Query: search_query=..."*. Fixed by scoping to `<entry>`, pinned by
  `tools/tests/test_library.py`. The lesson is the general one: **a tool reporting success is not
  evidence that its output is right.**

## ⛔ BINDING — EVERY EVAL REPORTS FOUR METRIC FAMILIES, NOT ADE (Sayed, 2026-08-02)

**Sayed, verbatim, after asking repeatedly:** *"Despite I told you many times, don't consider only
ADE at the different horizons, you are still doing this and ignoring the other metrics I requested.
So we will make it now very formal: I want you to ADD (not replace) ... Any future eval must include
these metrics and this is binding."*

**ADE stays. These are ADDED to it. An eval that reports ADE alone is INCOMPLETE and must not be
presented as a result.**

| family | must report | why it is not optional |
|---|---|---|
| **LONGITUDINAL** | target-speed accuracy **and distance-keeping** (headway / time-gap / TTC to the lead agent) | 88.7 % of our oracle gap is longitudinal. ADE hides it — an arm can win ADE while setting the wrong speed. |
| **LATERAL** | heading error, **curvature error, yaw-rate error**, cross-track | "lateral is fine" has been asserted from cross-track alone; curvature and yaw are where a smooth-but-wrong path shows up. |
| **TACTICAL** | manoeuvre-decision quality **and tactical goal-setting** (selected vs executed manoeuvre, confusion over the classes, goal/anchor selection) | the 5-way softmax that MIXES lat+lon is our single largest known defect; a scalar ADE cannot see a decision error. |
| **STRATEGIC** | strategic decision + goal/route setting quality | the hierarchy is the programme's thesis. If we never measure the strategic level we cannot claim it works. |

**Rules that travel with them:**

1. **Per-family, never pooled into one score.** A single composite hides exactly the trade-off we
   are trying to see.
2. Each family carries its **estimator** (paired episode-cluster bootstrap) and its **CI**, on the
   same windows as the ADE it accompanies.
3. **A missing metric is a work item, not an excuse.** If a family has no instrument, implement it —
   do not report the eval as complete with the family absent.
4. ⛔ **Never present a horizon sweep of ADE as "the result".** It is one row of four families.
5. Where a family genuinely cannot be computed (no lead agent in frame, no route label), say so
   **per family with the reason and the n**, rather than silently dropping it.

*(Root cause this rule exists: ADE is the cheapest number to produce and the easiest to compare, so
it crowded out the metrics that actually decide whether the car drives well. Three separate reports
went out with ADE-only tables after this was requested.)*

## ⛔ BINDING — "G: IS DOWN" IS THE WRONG DIAGNOSIS: `.git`'S CONTROL FILES DEHYDRATE WHILE EVERYTHING ELSE READS FINE (2026-09-10)

**MEASURED 2026-09-10 with interleaved same-breath controls.** The failure on this mount is
**not uniform**, and reading it as a whole-mount outage has cost real time — including mine,
this session, retrying `git rev-parse` five times against a repository that was perfectly
healthy.

| probe | result |
|---|---|
| `git rev-parse HEAD` (also `git -C`, also explicit `--git-dir`) | ⛔ `fatal: not a git repository` |
| `cat .git/HEAD`, `.git/config`, `.git/index`, `.git/packed-refs`, `.git/refs/heads/<branch>` | ⛔ **`Invalid request code`** |
| `ls .git/refs/heads/` , `ls .git/objects/pack/` | ✅ full listings |
| `head -c 32 .git/objects/pack/*.idx` | ✅ **READ_OK** |
| `wc -c < CLAUDE.md` (a worktree file) | ✅ **88,276**, matching `ls` exactly |

⇒ **The small MUTABLE control files under `.git/` are the ones that fail; pack files and
worktree files hydrate normally.** That is the entire explanation for the confusing signature
everyone keeps re-deriving: **git must read `HEAD` and `config` to recognise a directory as a
repository at all**, so a `.git` whose control files will not hydrate reports *"not a git
repository"* — a message about **one unreadable file**, dressed as a claim about the whole repo.

⭐ **THE THREE CONSEQUENCES, and they change what you should do:**

1. ⛔ **Retrying does not help and is not evidence.** This is Google Drive cloud-only
   dehydration, not a transient flap — the same mechanism already recorded for
   `.claude/worktrees/` (27 identical copies, 0 readable, metadata perfect). ⚠️ It IS
   intermittent per file: `.git/HEAD` read successfully in one probe and errored two minutes
   later. So a single success proves nothing either.
2. ⭐ **This is WHY `commit_cacheinfo.py` works when `git add` does not**, and the reason is
   now mechanical rather than folklore: it stages **from blobs already in the object store**
   and never reads the worktree — and **pack reads survive this failure mode**. Its immunity is
   not luck; it is the one route that touches only the half of `.git` that still hydrates.
3. ⭐ **THE WORKING ROUTE IS THE LOCAL MIRROR** at `C:/Users/Admin/tanitad-push/.git`. Its
   `read-tree` takes **0–1 s** where G:'s takes **13 minutes** under contention, and every
   plumbing operation succeeds. Author locally → `hash-object -w` → scratch index from the tip
   → `update-index --cacheinfo` → `write-tree` → **revert guard** → `commit-tree` → CAS
   `update-ref` → push. Three commits landed this way on 2026-09-10 while G: git was refusing
   every command.

⚠️ **AND THE MIRROR CAN BE BEHIND THE REMOTE — CHECK, DO NOT ASSUME.** MEASURED the same day:
the mirror's `refs/remotes/origin/<branch>` read `ad68b3d` and its own tip read `ad68b3d`, so it
looked fully synced; **GitHub actually held `b42ab460`**, one commit ahead, and `b42ab460` was
**not even a valid object** in the mirror. A commit built on the stale tip was correctly
**rejected as non-fast-forward**. ⇒ **`git ls-remote` before building**, and rebuild on the true
tip; ⛔ a stale remote-tracking ref is a cached claim, never a measurement.

⭐ **AND VERIFY THE PUSH BY THE REMOTE REF, NOT BY THE PUSH.** `git ls-remote origin <branch>`
must equal your local tip. A push has previously logged **"Everything up-to-date"** as its last
line **after an HTTP 500** while origin had not moved. **Assert on the artifact, not the status.**

⚠️ **One filter trap caught in the same investigation, and it is the `pgrep -f` family again:**
`ps -W | grep -ci git` read **9**, which looks exactly like heavy git contention. Every one was
`C:\Program Files\Git\bin\bash.exe` — **the install PATH matched, not a running `git.exe`.**
⇒ match the executable, not a substring of its directory, or you will diagnose contention that
does not exist and start killing processes to fix it.

## ⛔⛔ BINDING — "BUILT, TESTED, AND UNREACHABLE FROM ITS CALLER" IS A CLASS, NOT A SERIES OF ACCIDENTS. FOUR INSTANCES IN ONE DAY (2026-09-10/11)

Every one of these passed its own unit tests. Every one was importable, rollable and correct. **Not
one of them ran.** And three were found only *after* an arm had been trained on them.

| mechanism | why it never ran | what it cost |
|---|---|---|
| `tac_goal_tok_head` | `--w-tac-goal` declared `default=0.0` (`refc_v3_train.py:4922`) and never passed | **11,286 params, `grad_abs_sum` exactly 0.00000 for all 40,284 steps** of refcv5-v2 |
| the ≥GT truncation | `rl_pilot_refc21.py` exposed **no `--gt-bar` flag** — 13 flags, none reaching it | a 2,000-step "V2" arm that was a **V1 baseline**; the mask had been gradient-proven the same day |
| DD-v2's two-scalar sampler | `--noise-mode` did not exist on that caller (`grep -c` **0**, control **4**) | the released sampler was unreachable **from this caller** — though not from the programme |
| the posttrain spec guard | the guard existed **only inside its own test**; `grep` for its message = **2 hits, both in the test file, 0 in `posttrain.py`** | a **RED test on the tip**, while `GOALS_AND_CLAIMS.md` recorded the row **CLOSED** |

⭐ **AND A FIFTH, FROM THE OTHER DIRECTION — 18,472 PARAMETERS, NOT 11,286.** A probe run while
verifying an upload found **three** untrained heads in refcv5-v2, not one: `offset_head` (6,160),
`tac_goal_tok_head` (11,286), and `scorer.goal_point` (1,026, **exactly zero** — zero-init plus zero
gradient, so it **emits the constant origin**). ⇒ a single explanation covering one third of the
evidence is how the count stayed wrong.

### ⛔ THE RULE

⭐ **"Rollable and trained are different claims." A test that the mechanism WORKS is not evidence
that the mechanism RUNS.** Before any arm is launched on a new mechanism, prove **reachability from
the actual caller**, not existence in the library:

1. ⛔ **Trace the flag to the loss, or to the buffer that reaches it** — argparse → namespace →
   model attribute → forward → the term in `loss` → `.backward()`. Name the `file:line` at each hop.
2. ⭐ **Assert a NON-ZERO gradient on the new parameters, and a control that must read EXACTLY
   0.0.** The zero control is what proves the probe works at all; without it, a zero reading is
   indistinguishable from a broken probe.
3. ⛔ **MUTATE, do not inspect.** Delete the single line that emits the value and require the check
   to go RED. *(MEASURED: an AST census read **0 suspects on BOTH the fixed and the broken
   trainer**.)* ⭐ The strongest form is an **analytic identity**: a bar below every achievable
   reward must reproduce the unbarred gradient **bit-for-bit**; a bar above it must give **exactly
   `0.0`**; and the middle must be **partial** — *a mask that only ever admits all or nothing is a
   switch, not a truncation.*
4. ⚠️ **Check the DEFAULT, not just the flag.** `default=0.0` and `default="off"` are how a wired,
   tested mechanism silently does nothing. **A flag whose default disables it must be named in the
   arm's argv, or the arm did not test it.**
5. ⛔ **A grep for a guard's message that finds it ONLY in the test file means the guard does not
   exist.** Pair every such probe with a same-breath control that must read non-zero in the
   *implementation*.

⚠️ **AND THE REGISTER CAN BE WRONG IN THE SAME DIRECTION.** One of these rows read **CLOSED** while
its test was failing on the tip. ⇒ a register entry is a claim like any other; ⛔ **"the register
says it is done" is INHERITED evidence and may not decide a GPU-day.**

⭐ **Durable instrument:** `…/2026-09-10-refcv5v2-zerograd-heads/probe_zerograd3.py` reads untrained
parameters straight out of a checkpoint's optimizer state — **Adam allocates state lazily, so a
registered parameter with no entry never received a gradient**, and the count is exact (357
registered against 351 with state). ⛔ **It belongs in `stack/scripts/` as a POST-TRAINING GATE.**
This defect class has now surfaced five times, and every single time **after** an arm was trained
and, twice, after it was published.

<!-- STRATEGIC-PAUSE-2026-09-19 -->
⏸ **TEMPORARY PAUSE OF THE STRATEGIC FAMILY (Sayed, 2026-09-19, `PI_DECISION_QUEUE.md` ITEM 25).**
*"the strategic layer will be only switched off temporarily until we proved that both tactical and
operative layers are driving with high quality. So the 4 layer rule is just temporarily paused and
concerns only the strategic part."* ⇒ While the strategic layer is switched off, report the
STRATEGIC family as **PAUSED, with this ruling** — ⛔ never as a pass, ⛔ never silently omitted,
and ⛔ never extended to longitudinal, lateral or tactical, which keep full force. Do not "fix"
the pause; lifting it is the PI's call (`verdict_refcv6.py --unpause-strategic` restores S1).
⛔ **Nav is an INPUT at training AND inference** — stated by the PI many times; it is not an oracle
that disappears at deployment.

## ⛔ BINDING — NO TRAINING LAUNCH WITHOUT A LAUNCH-GATE PASS (PI, 2026-09-26)

**PI, verbatim:** *"Assure that this is not happening again because we lost a lot of time and energy. If we are starting a training session, we are sure about the correctness of the config."*

refcv6 trained with three config-vs-build defects, each found by hand, weeks late, and each costing GPU-days:
- the per-stage loss was absent for 34,500 steps;
- a declared image equalisation was dropped for all 38,250 steps;
- 2 of 3 declared selection mechanisms were never built.

The PI stopped it. From now on:

1. **No training run starts without a PASS token** from `stack/scripts/launch_gate.py`, bound to the exact commit, argv sha256 and data-manifest sha256s.
   - The supervisor REFUSES to start the trainer without a matching token.
   - A token is never hand-written or hand-edited.
2. **The gate's checks** (`Project Steering/SPEC_REFCV7.md` §2):
   - **G-HYG:** config dataclasses refuse undeclared attributes.
   - **G-DVB:** every argv lever equals the built model's value.
   - **G-LIVE:** a real-data smoke in which every declared loss term appears with a finite value and every trainable group gets a gradient.
   - **G-CLOCK:** label time.
   - **G-EVAL:** the eval loader builds the identical model.
   - **G-CKPT:** checkpoint round-trip.
   - **G-SUITE:** clean-tree tests with 0 regressions.
   - Each check has a deliberate-regression arm that must make the gate FAIL.
3. **A new trainer flag without a G-DVB entry is refused by the gate itself.** Adding a lever means adding its check.
4. **A config that DECLARES a lever the model does not BUILD is a launch blocker, never a footnote.**
