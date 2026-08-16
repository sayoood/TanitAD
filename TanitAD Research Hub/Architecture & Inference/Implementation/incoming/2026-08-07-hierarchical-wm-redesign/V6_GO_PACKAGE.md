# V6 GO PACKAGE — the v6 ladder's OPERATOR RUNBOOK
### §2 is the part you paste at 3 a.m. It is generated, not written. Decisions: `PI_DECISIONS_2026-08-12.md`.

> ## ⛔ STATUS — 2026-08-16. §2 WAS REWRITTEN; §1's "BLOCKED" VERDICT IS SUPERSEDED.
>
> **S-W IS LIVE.** It is training on **Thor**, not on a provisioned A40 pod — step ~6,400 of
> 30,000 at a **MEASURED 27.18 s/step**, trainer **pid 25477**, ops loop **pid 29587**, run dir
> **`~/experiments/v6F-SW-30k`**. *(INHERITED from the coordinator's fleet read, 2026-08-16; not
> re-probed here — Thor holds the only GPU and this turn was CPU-only by instruction.)*
> ⇒ **Do not follow §2.1 to "start S-W". It started.** The next launch is **S-T**, and it launches
> from `v6_chain.py`.
>
> **⛔ THE LAUNCH LINES IN THIS DOCUMENT ARE NO LONGER HAND-WRITTEN.** They are the output of
> **`stack/scripts/v6_chain.py commands`**, and `stack/tests/test_runbook_commands.py` FAILS THE
> SUITE if this file and that generator disagree. If you are reading a line here, it parsed
> against the trainer's real `build_parser()` the last time `pytest` ran.
>
> **SEL-1 is REFUSED** (E-WC2, 2026-08-16). S-T's default is `--selector none`, `--w-select 0`,
> and a selector arm is **refused at launch**, not merely left out of the plan. §2.6.

**Tier stamp.** Nothing produced by this trainer is quotable as driving performance. Capability
claims are **T1** (`taniteval/tools/t1_eval.py`) with the **four metric families** and the
**paired episode-cluster bootstrap** (`taniteval/ci.py`). `EVAL_DOCTRINE.md`.

**Sources.** Model facts cite `Project Steering/MODEL_REGISTRY.md §x` or a raw gate JSON.
Rows marked *(mine 2026-08-12)* were measured by this doc's original author; rows marked
*(mine 2026-08-16)* were re-measured on the dev box during the §2 rewrite.

---

## 1. GO / NO-GO — every precursor for S-W

> ⚠️ **THIS TABLE IS A 2026-08-12 SNAPSHOT AND ROWS 9–16 ARE SUPERSEDED.** S-W is LIVE on Thor
> (see the STATUS block at the top), so the provisioning blockers were resolved by a different
> route than the one this table anticipated. The rows are corrected in place below rather than
> deleted, because *how* a blocker cleared is the useful record. **§2 is the current runbook.**

| # | precursor | state | evidence |
|---|---|---|---|
| 1 | v6 model, staged trainer, tests exist in the repo | ✅ **DONE** | `stack/tanitad/models/v6.py` · `stack/scripts/train_v6_staged.py` · `stack/tests/test_v6_staged.py`, all in HEAD `2b8d09e` |
| 2 | the test suite is green | ✅ **DONE** | **MEASURED (mine 2026-08-12): `pytest tests/test_v6_staged.py -q -W error::UserWarning` → 80 passed, 0 warnings** |
| 3 | it imports, builds and takes real optimiser steps | ✅ **DONE** | **MEASURED (mine 2026-08-12): `--stage S-W --dry-run --device cpu` → two `[v6 dry N]` rows with live O1/O2/O3/O5/O6 terms, then `dry-run OK`** |
| 4 | sub-300M invariant enforced **before** GPU time | ✅ **DONE** | **MEASURED (mine): `[v6] params 87.89 M / budget 300 M · arm shared-encoder+adapters`** — `build_stack_from_args` refuses over budget pre-launch |
| 5 | X3 gradient isolation is a real autograd probe and passes | ✅ **DONE** | **MEASURED (mine): `[v6] X3 isolation pass=True violations={'planner_to_encoder': 0, 'tactical_to_below': 0, 'strategic_to_below': 0}`** |
| 6 | the 6 s horizon is trainable — v4's `max_horizon` 20 is NOT inherited | ✅ **DONE** | **MEASURED (mine): derived `max_horizon` = 20 (S-W as specced, S-S) / 60 (S-T, S-J)** from `train_v6_staged.py:1071–1084`; the trainer prints its choice beside v4's |
| 7 | §4b seam-free by construction — ONE 60-step (a,κ) rollout, bands are slices | ✅ **DONE** | `V6Config` refuses a band gap/overlap at construction; **MEASURED (mine): `plan_steps` 60, `dt` 0.1, bands (0,2) and (2,6)** |
| 8 | resume + done-marker guards (the supervisor-resurrection defence) | ✅ **DONE** | `V6_TRAINER_DESIGN §3.4b`: refuses a DONE dir without `--force-rerun`; refuses `--resume off` over an existing `ckpt.pt`; writes `summary.json {"done": true}` in the same turn it finishes |
| 9 | **the corpus caches exist on the target box** | ✅ **RESOLVED** (superseded) | the target became **Thor**, not a pod. §2.0(c) is still the check, at `/root/data/…` |
| 10 | ~~a free A40 pod for 7–12 days~~ | ✅ **OVERTAKEN — S-W runs on THOR** | *(INHERITED 2026-08-16)* trainer pid 25477, `~/experiments/v6F-SW-30k`. The A40 provisioning question never had to be answered |
| 11 | ~~cost authorisation~~ | ✅ **OVERTAKEN** | ESTIMATED 175–290 A40-h became **MEASURED 27.18 s/step on Thor**; §2.1 |
| 12 | code shipped, md5-verified | ⚠️ **THE FILE LIST IN THIS ROW WAS WRONG WHEN WRITTEN** | ⛔ **FIVE artifacts, not three.** `train_stage_a.py` and `stage_a_probes.py` are *also* top-level imports (`train_v6_staged.py:114`, `:117`) — and **they already were at `2b8d09e`, the commit this row cites** (MEASURED mine 2026-08-16, `git show 2b8d09e:…`). This was never a staleness; it was one dependency measured and the *closure* assumed. §2.0(a) has the corrected list |
| 13 | the direct SSH mapping answers | ✅ **N/A on Thor** | the RunPod direct-mapping trap (**MEASURED 2026-08-11: both pod4 and pod5 refuse on their own `$RUNPOD_PUBLIC_IP:$RUNPOD_TCP_PORT_22`**) does not apply to Thor. It still applies to any pod route |
| 14 | disk headroom | 🟡 **still the operator's check** | ⛔ **never judge pod disk with `df`** (reports the whole cluster, hides the per-pod MooseFS quota — a full quota killed the flagship mid-checkpoint). Use a real `dd` write test — §2.0(d) |
| 15 | S-W's gate probes (P1/P3/P6) are runnable | 🟡 **PARTIAL** | the battery is a **separate frozen instrument by design** — the trainer does not run it in-loop. Owners are named in `STAGE_GATE_SPEC`: P1 `scripts/probe_latent_state.py`, P3/P6 `scripts/stage_a_probes.py`. ⛔ Folded in on a **RE-GATE**, never on the launch — §2.9 |
| 16 | a `runs.d/<run>.env` manifest + supervisor per stage | ✅ **MECHANISED** | `v6_chain.py manifests` emits **one manifest per stage**, `TRAIN_CMD` always the trainer — §2.8 |
| 17 | **W5/E-H1 — v5.8f's 6 s baseline** | ⛔ **NOT-DONE** | ⚠️ **This is a YARDSTICK requirement, not a launch blocker.** See §1.1 |
| 18 | **S2 — `g_str` supervision from PH2** | ⛔ **NOT-STARTED** | ⚠️ blocks **S-S's gate**, not S-W and not S-T. See §1.1 |
| 19 | whole-suite `pytest -q` green (the commit invariant) | ✅ **RESOLVED** | the 23 failures of 2026-08-12 are gone. **MEASURED at HEAD `eca7106`: 3154 passed / 0 failed / 17 skipped / 2 xfailed.** F5 can close |

### 1.1 The two escalated precursors, answered precisely

**W5/E-H1 (#17) — does it block the launch? NO. It blocks the comparison.**

`HIERARCHY_VOCABULARY §4b` promotes it to a REQUIRED precursor: *"it baselines v5.8f at 6 s
before v6 trains against it."* But **MEASURED (mine, source read 2026-08-12): S-W consumes no
v5.8f MEASUREMENT.** S-W's required gate is **P1 / P3 / P6**, none of which reference v5.8f, and
no path in `train_v6_staged.py` reads a v5.8f **checkpoint, gate JSON or eval result**.

> ⚠️ **Precision, because the grep is misleading.** v6 *does* import v5.8f **code** —
> `train_v58f_unicycle_head`'s `A_MAX`/`KAPPA_MAX` (v6.py:1096, train_v6_staged.py:116),
> `build_train_episodes` (:1001), `make_sampler` (:1159), `UnicycleEmission`. That is
> **deliberate reuse of gated parts**, not a dependency on a v5.8f *number*. It has one
> operational consequence, in precursor #12: **`train_v58f_unicycle_head.py` is a hard
> import-time dependency and must be shipped and md5-verified alongside the two v6 files** —
> §2.0 covers it.

⇒ **S-W can start; the 6 s v6-vs-v5.8f comparison is not quotable until W5 lands.** Run them in
parallel on different pods (D3). If the PI prefers serial, that is a schedule choice, not a
technical one.

**S2 / PH2 (#18) — does it block the launch? NO. It blocks S-S's GATE.**

| stage | can it start? | why |
|---|---|---|
| **S-W** | ✅ **YES** | S2 appears in no S-W loss term and in no S-W gate probe |
| **S-T** | ✅ yes, once S-W's gate exists | its gate is TACTICAL family + `sel_gap`; S2 is not involved |
| **S-S** | ⛔ **NO — structurally** | `STAGE_GATE_SPEC["S-S"]["required"] = ("STRATEGIC_family",)` and S2 is its only supervision source. **The stage will terminate at `pass: null`, and `pass: null` is not a pass** — S-J then refuses to launch |

⇒ **exactly as the brief suspected: S-W can start, S-T can follow its gate, and S-S cannot
complete a valid gate before PH2.** The fix is the pre-registered amendment in **D6** (promote
`S1_ade_8_30s` to required for the PH2-pending case, STRATEGIC reported `n/a` **with its reason
and its n**) — a ~1 h code change in `stack/`, filed as **F1**. It must land BEFORE S-S runs, not
after seeing S1's number.

---

## 2. The exact command lines — ⭐ GENERATED BY `v6_chain.py`, PINNED BY A TEST

⛔ **DO NOT HAND-WRITE A V6 LAUNCH LINE, AND DO NOT EDIT THE ONES BELOW.** Since 2026-08-16 the
ladder has a launcher — **`stack/scripts/v6_chain.py`** — and it is the only place a v6 launch line
is constructed, so *"the command someone improvised at 3 a.m."* has nowhere to come from. Every
block in §2.2–§2.5 is the verbatim output of `v6_chain.py commands`, and
**`stack/tests/test_runbook_commands.py` fails the suite the moment this file and that generator
disagree** — which is the only reason you may trust a runbook you did not re-derive.

**Regenerate rather than edit:**

```bash
cd /root/TanitAD/stack && python3 scripts/v6_chain.py commands \
  --root /root/experiments --workdir /root/TanitAD/stack \
  --train-cache /root/data/physicalai-train-e438721ae894-w120-256x640cyl \
  --val-cache   /root/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl
```

⚠️ **Generating these from Git Bash on the dev box mangles absolute POSIX paths** — MSYS rewrites
`/root/experiments` to `C:/Program Files/Git/root/experiments`. Use `MSYS_NO_PATHCONV=1`, or
generate on the pod. ⛔ **And never paste a `~`:** every path is `shlex.quote`d (correctly), and
quoting **suppresses tilde expansion**, so `mkdir -p '~/experiments/…'` creates a directory
*literally named* `~` and the next stage's `--init-from` then points at nothing. `assert_no_tilde`
refuses a `~` rather than expanding it, because `~` on the box that *generates* the line is not `~`
on the box that *runs* it.

**The two env vars that are not optional, and are already in every line below:**
⛔ **`PYTHONPATH=<workdir>`** or the trainer dies with `ModuleNotFound: tanitad` — `cd` alone is not
enough. ⛔ **`OMP_NUM_THREADS=6`** — torch spawns ~113 threads per process; 7 concurrent arms sat at
GPU `sm` **0–6 % for 50 minutes** without it (MEASURED 2026-07-27). Set in the launch line so it is
visible in `ps`, not only defensively inside the trainer.

### 2.0 STEP ZERO — on the pod, before any launch (~1 min, 0 GPU)

```bash
# a. the files arrived by md5-verified FILE-SHIP (pods have NO git credentials)
#    ⛔ FIVE artifacts, not three. MEASURED (mine 2026-08-16) by importing the trainer and
#    listing the modules it resolved out of scripts/: train_stage_a.py and stage_a_probes.py
#    are ALSO top-level imports (train_v6_staged.py:114 and :117). Ship three and the pod
#    dies at import with `ModuleNotFound: train_stage_a` — and a STALE copy of any of them
#    does not fail loudly, it changes the emission envelope.
md5sum /root/TanitAD/stack/tanitad/models/v6.py \
       /root/TanitAD/stack/scripts/train_v6_staged.py \
       /root/TanitAD/stack/scripts/train_v58f_unicycle_head.py \
       /root/TanitAD/stack/scripts/train_stage_a.py \
       /root/TanitAD/stack/scripts/stage_a_probes.py
#    …and the launcher itself, if you intend to use `next` / `status` / `manifests` on the pod:
md5sum /root/TanitAD/stack/scripts/v6_chain.py

# b. grep-verify the fix you think is there IS there
grep -c "assert_isolation" /root/TanitAD/stack/tanitad/models/v6.py

# c. the caches exist
ls -d /root/data/physicalai-train-e438721ae894-w120-256x640cyl \
      /root/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl

# d. disk headroom — ⛔ NOT `df` (it reports the whole cluster, not the per-pod quota)
dd if=/dev/zero of=/root/_ddtest bs=1M count=4096 oflag=direct && rm -f /root/_ddtest

# e. it imports and steps — 0 GPU, 0 corpus, ~20 s
cd /root/TanitAD/stack && \
PYTHONPATH=/root/TanitAD/stack OMP_NUM_THREADS=6 \
python3 scripts/train_v6_staged.py --stage S-W --dry-run --device cpu \
    --out /root/experiments/v6-dryrun --dry-batch 1 --dry-steps 2 \
    --dry-k 12 --o5-k 12

# f. ⭐ ask the LADDER what may launch, instead of deciding yourself. Exit 3 = refused,
#    and the refusal text names the exact missing artifact.
python3 scripts/v6_chain.py next --root /root/experiments --workdir /root/TanitAD/stack
```

**Expect from (e) — MEASURED (mine 2026-08-16), by executing it on the dev box today:**

```
[v6] params 87.89 M / budget 300 M · arm shared-encoder+adapters · per-group {...}
[v6] X3 isolation pass=True violations={'planner_to_encoder': 0, 'tactical_to_below': 0, 'strategic_to_below': 0}
[v6 dry 1] {...}
[v6 dry 2] {...}
[v6] stage gate INCONCLUSIVE -> .../stage_gate.json
[v6] dry-run OK -> .../dry_run.json + config.json + stage_gate.json (INCONCLUSIVE, _dry_run)
```

⚠️ **`INCONCLUSIVE` here is the correct result, not a failure.** A dry-run runs no frozen-battery
probe, so the honest verdict is INCONCLUSIVE; the gate is additionally stamped `"_dry_run": true`
and `assert_stage_precondition(dry_run=False)` refuses it, so **a dry gate can never license a real
launch.**

⭐ **The dry-run now honours `--prev-gate` and `--init-from`** (fixed 2026-08-16, defect D0). Before
that fix a dry-run of S-T printed `dry-run OK` while the real launch of the *same command* died on
`Missing key(s): cand_score.*` — a pre-launch verifier structurally incapable of catching the class
of failure it exists to catch. ⇒ **dry-running a transition is now worth something.**

⛔ **If any of a–f disagrees, do not launch.** A launch from a stale checkout resurrects fixed bugs
(MEASURED 2026-07-27: pod2 sat at `0f93b98` while the v5 gate fix was at HEAD).

### 2.1 S-W — the WORLD stage · ⛔ **ALREADY RUNNING. DO NOT LAUNCH IT.**

**Host:** **Thor**, not an A40 pod. **Trainer pid 25477 · ops loop pid 29587 ·
`~/experiments/v6F-SW-30k`.** Step ~6,400 of 30,000. *(INHERITED, 2026-08-16.)*

**`step_s`: MEASURED 27.18** — marginal over steps 6300→6400 of this run; three statistics with
different startup exposure agree to 0.5 % (50-step 27.21 · 100-step 27.18 · cumulative-with-startup
27.32). ⇒ **remaining ≈ 23,600 steps ≈ 178 h ≈ 7.4 days.** An A40 would be 20.46 s/step (1.33×).

⚠️ **This supersedes §4's "ESTIMATED 21–35 s/step (A40)" for S-W.** That band was an A40 estimate
for a stage that has since run on different silicon; 27.18 is not "in band", it is a different
measurement of a different machine. **S-T/S-S/S-J have never run, so their rows in §4 remain
ESTIMATED — at S-W's rate, which makes them upper-ish.** The rule is unchanged: **read `step_s` at
step 500 of each new stage and re-cost.**

⛔ **The launch line for S-W is included in §2.2's block below for completeness only.** Running it
would create a **second** S-W beside the live one. The chain refuses to write into a directory
holding another stage's checkpoint (`assert_out_dir_free`), but it will not save you from launching
a duplicate of the *same* stage into a *different* directory — that judgement is yours.

### 2.2 The four stage launch lines — verbatim `v6_chain.py commands`

⚠️ Substitute Thor's real `$HOME` if it is not `/root` — read it with `echo $HOME` **on the pod**.

```bash
# ---- S-W · S-W · 30000 steps · 9.44 d at 27.18 s/step ----
# ⛔ ALREADY RUNNING (pid 25477). Shown so the ladder is complete; do not paste it.
mkdir -p /root/experiments/v6F-SW-30k && cd /root/TanitAD/stack && PYTHONPATH=/root/TanitAD/stack OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 -u scripts/train_v6_staged.py --stage S-W --out /root/experiments/v6F-SW-30k --steps 30000 --batch 8 --lr 0.0001 --n-candidates 8 --v2-cache /root/data/physicalai-train-e438721ae894-w120-256x640cyl --v2-val-cache /root/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl --v2-lru 6 --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical --v2-subframe 176x624 --require-parity > /root/experiments/v6F-SW-30k/train.out 2>&1 < /dev/null &

# ---- S-T · S-T · 10000 steps · 3.15 d at 27.18 s/step ----
# ⭐ THE NEXT LAUNCH. --selector none / --w-select 0: SEL-1 is REFUSED (§2.6).
mkdir -p /root/experiments/v6F-ST-10k && cd /root/TanitAD/stack && PYTHONPATH=/root/TanitAD/stack OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 -u scripts/train_v6_staged.py --stage S-T --out /root/experiments/v6F-ST-10k --steps 10000 --batch 8 --lr 0.0001 --n-candidates 8 --init-from /root/experiments/v6F-SW-30k/ckpt.pt --prev-gate /root/experiments/v6F-SW-30k/stage_gate.json --max-horizon 60 --plan-wta-eps 0.05 --w-t1 1.0 --v2-cache /root/data/physicalai-train-e438721ae894-w120-256x640cyl --v2-val-cache /root/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl --v2-lru 6 --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical --v2-subframe 176x624 --require-parity > /root/experiments/v6F-ST-10k/train.out 2>&1 < /dev/null &

# ---- S-S · S-S · 8000 steps · 2.52 d at 27.18 s/step ----
# ⛔ Its gate REQUIRES sel_gap_revalidated + TACTICAL_revalidated — see §2.7.
mkdir -p /root/experiments/v6F-SS-8k && cd /root/TanitAD/stack && PYTHONPATH=/root/TanitAD/stack OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 -u scripts/train_v6_staged.py --stage S-S --out /root/experiments/v6F-SS-8k --steps 8000 --batch 8 --lr 0.0001 --n-candidates 8 --init-from /root/experiments/v6F-ST-10k/ckpt.pt --prev-gate /root/experiments/v6F-ST-10k/stage_gate.json --w-s1 1.0 --v2-cache /root/data/physicalai-train-e438721ae894-w120-256x640cyl --v2-val-cache /root/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl --v2-lru 6 --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical --v2-subframe 176x624 --require-parity > /root/experiments/v6F-SS-8k/train.out 2>&1 < /dev/null &

# ---- S-J · S-J · 3000 steps · 0.94 d at 27.18 s/step ----
# OPTIONAL. Run only if S-T/S-S plateau.
mkdir -p /root/experiments/v6F-SJ-3k && cd /root/TanitAD/stack && PYTHONPATH=/root/TanitAD/stack OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 -u scripts/train_v6_staged.py --stage S-J --out /root/experiments/v6F-SJ-3k --steps 3000 --batch 8 --lr 3e-05 --n-candidates 8 --init-from /root/experiments/v6F-SS-8k/ckpt.pt --prev-gate /root/experiments/v6F-SS-8k/stage_gate.json --max-horizon 60 --w-t1 1.0 --w-s1 1.0 --plan-wta-eps 0.05 --v2-cache /root/data/physicalai-train-e438721ae894-w120-256x640cyl --v2-val-cache /root/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl --v2-lru 6 --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical --v2-subframe 176x624 --require-parity > /root/experiments/v6F-SJ-3k/train.out 2>&1 < /dev/null &
```

**What each stage trains, and what is frozen:**

| stage | trains | frozen | losses in force |
|---|---|---|---|
| **S-W** | `encoder`, `readout`, `predictor_op` (+`step_readout_op`), `aux` | `layer_tac`, `layer_str`, `planner` | O1 (ctrl 1.0 / fact 1.0 / scene 0.3) · O2 1.0 · O3 1.0 · O5 1.0 · O6 0.1. **λ_plan ≡ 0** — the trainer **refuses to start** S-W with a non-zero `--lambda-plan`, and refuses `--selector` here at all |
| **S-T** | `layer_tac` + `planner` | everything below (Drive-JEPA's shape: the planner is a *post-trained consumer*) | λ_plan 1.0; `plan_ade_0_2s` and `plan_ade_2_6s` reported **separately**, because a pooled 0–6 s number cannot see the seam |
| **S-S** | `layer_str` only | everything else | `w_s1` 1.0; **λ_plan 0**, and `for_stage('S-S')` zeroes `w_select` regardless |
| **S-J** | everything, **isolation still ON** | — | all terms live again |

**Three flags in those lines that were NOT in the pre-2026-08-16 runbook, and why each is there:**

| flag / token | why | evidence |
|---|---|---|
| **`--batch 8`** (was 16) | ⛔ Thor's 20 SMs **saturate at batch 8** — throughput is FLAT at 12.3–14.1 windows/s across a 6× batch range, so a bigger batch buys **nothing** and only costs memory. This **inverts** the A40 instinct. | `v6_chain.py:122` `THOR_BATCH`; `--a40` switches back to 16 |
| **`--v2-lru 6`** (was absent ⇒ **64**) | ⚠️ **The "each dataloader worker costs ~8.6 GB host RAM" rule DOES NOT BIND THIS TRAINER** — `train_v6_staged.train()` collates **synchronously in the main process** (`default_collate([ds_train[i] for i in idx])`) and never constructs a `DataLoader`. **There are zero workers to tune.** The real host-RAM knob is `--v2-lru`, whose trainer default is **64**; and on Thor host RAM **is** device memory, so the frame cache competes with the model rather than sitting beside it. *(Repeating a rule that does not apply is how a constraint list stops being read.)* | trainer default MEASURED at `train_v6_staged.py:2210`; `v6_chain.py:124-130` |
| **`--n-candidates 8` on EVERY step, S-W included** | ⛔ The fan size is a **ladder-wide constant**. Let one stage take the default and another name it, and `--init-from` dies with `size mismatch for cand_queries.weight: [8, 256] vs [3, 256]` — and that failure **bypasses `load_stage_init`'s adjudication entirely**, because `load_state_dict(strict=False)` tolerates missing/unexpected *keys* but still **raises** on a shape mismatch, so `STAGE_MAY_INTRODUCE` never sees it. | MEASURED 2026-08-16, defect D4, on the dry ladder's first execution |
| **`setsid` + `-u` + `< /dev/null`** | `setsid` detaches the job from the ssh session's process group; `-u` unbuffers so `train.out` is readable live; `< /dev/null` is the **nested-stdin** defence (a nested `ssh` inside a piped script eats the rest of the script's stdin, and the tail silently never runs) | `v6_chain.py:launch_line` |

### 2.3 ⛔ Memory on Thor — the only admissible probe

**`torch.cuda.max_memory_allocated()`, in-process. Nothing else.** MEASURED 2026-08-03:
`torch.cuda.mem_get_info()` reported 3.4 GB free while **60 GB was allocated AND written**;
`free`/`tegrastats` showed 106 GB "used" on a **completely idle box**; `/proc/self/status` `VmRSS`
read 0.62 GB against 24 GB allocated. On unified memory these are not imprecise, they are
**unrelated to the question, in both directions.**

⇒ The trainer now logs **`cuda_max_mem_gb` per log row** (`train_v6_staged.py:1675`), with a note
naming the four probes not to cross-check it against. **Read that field. Do not open `tegrastats`.**

### 2.4 The `--init-from` chain

```
S-W ──stage_gate.json──► S-T ──stage_gate.json──► S-S ──stage_gate.json──► S-J
    └──── ckpt.pt ──────►    └──── ckpt.pt ──────►    └──── ckpt.pt ──────►
 v6F-SW-30k             v6F-ST-10k              v6F-SS-8k              v6F-SJ-3k
   (LIVE)                10,000 st               8,000 st               3,000 st
```

Every downstream stage passes **both** `--prev-gate` (the X5 certificate) and `--init-from` (the
weights). Missing either is a **refusal**, not a warning — in the chain and again in the trainer.
⛔ **`--out` is unique per step**, and `assert_plan` refuses a plan where two steps share one, so
the trap is **unbuildable** rather than merely caught.

⚠️ The reason that matters: the only *accidental* barrier against a stage resuming another stage's
checkpoint is that the per-stage trainable-tensor counts happen to differ (**S-W 240 · S-T 80 ·
S-S 54 · S-J 374**, MEASURED). **One `STAGE_GROUPS` edit and it passes silently.**

### 2.5 The pre-registered control arms

| pre-registered control arm | flag | ⚠️ never a default |
|---|---|---|
| E-ENC (b) per-layer encoders | `--per-layer-encoders` | D5 |
| uplink = V-JEPA teacher | `--uplink ema --ema-decay 0.996` | |
| O4 off (attributability control) | `--o4-alpha 0` | reproduces uniform sampling exactly |
| O5 endpoint (reproduces the defect O5 fixes) | `--o5-mode endpoint` | |
| ⛔ isolation off (the co-trained path) | `--no-isolate-planner` / `--no-isolate-uplink` **+ `--i-know-this-is-the-control-arm`** | preflight refuses without the acknowledgement |
| inert-scorer control at S-T | `--selector goal --w-select 0 --i-know-this-is-the-control-arm` | ⛔ **also gated on §2.6** |

⚠️ **`--i-know-this-is-the-control-arm` DID NOT WORK until 2026-08-16 (defect D1).** `main()`
registers it with `dest="control_arm_ack"`, and `preflight` checked
`getattr(a, "i_know_this_is_the_control_arm", False)` — an attribute argparse **never creates**, so
it could only ever return `False`. **The escape hatch that a refusal message named did not clear
that refusal**, and the pre-registered inert-scorer control arm was therefore unlaunchable. Fixed at
`train_v6_staged.py:2280-2281` (both spellings accepted). ⇒ **If `V6F_PLANNER_DESIGN` §4.1's
"Pre-registered controls" list is still read as evidence those arms were reachable, it is wrong for
every run before today.**

### 2.6 ⛔ SEL-1 IS REFUSED — a selector arm is REFUSED AT LAUNCH, not merely off by default

**E-WC2 fired 2026-08-16** against `V6F_PLANNER_DESIGN` §5.2's pre-registration, **both outcomes
committed in advance**:

| quantity | value | class |
|---|---|---|
| σ(2 s) per-axis, REF-C-XL | **4.7104 m** [3.8087, 5.6860], 881 windows / 40 eps | MEASURED |
| **σ/ADE** | **9.9915** [7.4492, 13.5119] vs incumbent ADE 0.4714 | MEASURED |
| pre-registered lines | funded ≤ 1.7 · **refused ≥ 3.0** | PRE-REGISTERED |
| REF-C-base replication | σ/ADE **9.6337** | MEASURED |

**It is not close: the interval's LOWER bound (7.4492) is 2.48× the refusal threshold**, and
INCONCLUSIVE was never entered ⇒ the capacity control is not needed to decide it. Estimator: point
estimates `full_set`, intervals episode-cluster bootstrap over the 40 val episodes, 2000 draws;
`overlapping_holdout_se` used nowhere. **Tier `T0-DIAGNOSTIC` — no T1 claim may cite it.**

⭐ **The reason matters more than the verdict.** A **0-parameter constant-yaw-rate** goal reaches
σ(2 s) = **1.1888 m — 3.96× better than the ridge on frozen REF-C latents.** So the reading is **not**
*"a 6 s goal is unpredictable"*; it is **"these latents are the wrong surface"**. SEL-1's *estimand*
survives; its *input* does not. ⚠️ And the other half must be said with it: even that kinematic floor
lands at σ/ADE **2.52**, still **not FUNDED**. **Neither surface is good enough today.**
⚠️ **Scope:** this is the **REF-C** surface, not the frozen S-W latents §5.2 nominally names —
those have **never been dumped**. The **ratios** transfer; the **absolute metres are REF-C's**.

**What this means at the keyboard:**

* S-T's default is **`--selector none`, `--w-select 0`** — and that is what the §2.2 line above does
  (the trainer's defaults are `selector="none"`, `w_select=0.0`).
* Adding `--selector goal|mlp` to any stage is **REFUSED by `assert_selector_admissible`**, which
  runs **first** in `assert_may_launch`, before anything expensive. *"We left it out of the default
  plan"* is a preference; this is a fired pre-registration, so it refuses the launch.
* **The arms are gated, not deleted.** `GoalDistanceScorer` and `MLPCandidateScorer` stay
  implemented, dry-run-verified and reachable on a better surface.
* ⚠️ **If arms are ever reopened, `goal` and `mlp` are an ARM PAIR.** A `goal` win judged without
  the +33,801-param information-matched capacity control is unattributable between MECHANISM and
  CAPACITY — the **C6 confound** verbatim — and it **cannot be fixed later**, because after S-S the
  S-T GPU-days are spent and the two arms can no longer be compared on the same trunk.
  `--unpaired-arm-reason '<why>'` makes it a *recorded* decision instead of a silent one.

#### 2.6.1 ⭐ The ONE thing that could reopen it — pre-registered in code BEFORE the measurement

The only missing input is one **~10–25 GPU-min** latent dump at the **S-W → S-T boundary**. Its
thresholds are `SW_LATENT_ADMISSION` in `v6_chain.py`, **committed before that dump is taken**:

| σ(2 s) on frozen S-W latents | verdict |
|---|---|
| **≤ 0.80 m** | **FUNDED** — 5.89× better than REF-C's ridge and 1.49× better than the 0-param kinematic floor, from vision alone. Selector arms reopen. |
| 0.80 < σ ≤ 1.41 m | **INCONCLUSIVE** — REFUSED stands. The first re-run is the capacity control, not a weight sweep. |
| **> 1.41 m** | **REFUSED stands** |

⚠️ **2 s only. No 6 s threshold is emitted, and that is a rule firing rather than an omission:**
σ(6 s)/σ(2 s) = **3.7481**, past §5.3's 3× line ⇒ **REDERIVE**. Scaling the 2 s threshold across a
3× horizon change is exactly the ≤2× extrapolation prohibition.

To adjudicate a dump: write `<S-W out>/ewc2_sw_latents.json` with `sigma_2s_m`, then
`python3 scripts/v6_chain.py admission`. ⛔ **Absence of the probe is NOT an admission.**

**⚠️ The committed branch if it stays refused is `ANCHOR_GOAL` supervision** (PH0 +
`obstacle.offline` agent slots) — **not another cost function.** A goal-distance selector whose goal
is unsupervised is being asked to invent its own reference.

### 2.7 ⛔ S-S's gate now REQUIRES two REVALIDATIONS — plan for three probes, not one

`STAGE_GATE_SPEC["S-S"]["required"]` is **`("STRATEGIC_family", "sel_gap_revalidated",
"TACTICAL_revalidated")`** (MEASURED at `train_v6_staged.py:365-372`). Any doc — including §1 row 18
and §3's table below — that still says S-S's only required probe is `STRATEGIC_family` **predates
this and is wrong.**

**Why they are `required` and not `reported`:** S-S trains `layer_str`, and its output flows
`goal_head_str → e_g_str → goal_head_tac(cond=e_g_str) → e_g_tac`, and **`e_g_tac` is the frozen
selector's ONLY input.** The selector and `goal_head_tac` are frozen in S-S — but their **input
distribution moves**. S-T certified `sel_gap` and the TACTICAL family against the *S-T-era* `e_g_tac`,
and **that certificate does not survive S-S.** It is registry §1.14's consumer-invalidation one
level up: *you cannot repair a trunk and keep its planner.* Being `required` is what makes an S-S
gate that omits them read **INCONCLUSIVE and never PASS** — a *reported-only* probe is exactly what
a silent carry-forward looks like.

⚠️ **The waiver is the one the design commits to and no wider.** Omitting the re-measurement is
waivable with `--allow-inconclusive-gate` **and** a recorded `--gate-off-reason`. A measured
**REGRESSION is a FAIL, and a FAIL has no override anywhere in this ladder** — it is the trigger for
the S-S′ planner-refit micro-stage, not a carry-forward. `assert_revalidations_present` refuses
anything consuming S-S's checkpoint on a gate that skipped them, and **names the seam** rather than
saying "INCONCLUSIVE" and sending you hunting.

### 2.8 The supervisor seam, and how to verify what is REALLY running

```bash
python3 scripts/v6_chain.py manifests --dest ops/runs.d --root /root/experiments \
  --workdir /root/TanitAD/stack
python3 scripts/v6_chain.py verify --step S-T --root /root/experiments   # prints a /proc probe
```

⛔ **ONE MANIFEST PER STAGE, and every `TRAIN_CMD` is the TRAINER — never `v6_chain.py`.**
`supervise_run.sh` **sources its manifest ONCE, at supervisor startup**, and replays the command it
captured, so a supervised *chain* would replay stage 1 after a mid-ladder crash. Each manifest also
sets `TRAIN_MATCH='train_v6_staged\.py.*<run_id>'` — run-scoped, so a supervisor waits for **its
own** stage's trainer and can never adopt a sibling's.

**To change a supervised run:** edit the manifest → kill the **SUPERVISOR** first → kill the trainer
→ start a fresh supervisor. Killing the trainer first just makes the supervisor restore the stale
command. ⚠️ And do **not** restart immediately: the new supervisor **races the old one's `flock`**,
prints *"another supervisor holds …lock"* and dies, leaving **nothing running**. Poll `ps` until
both are gone.

⛔ **Verify by grepping the RUNNING PROCESS, never by reading the manifest back — and the probe is
not `pgrep -f` and not `ps | grep`.** Both put the searched token into the searching process's own
command line, so the probe **matches itself**; measured three times in this programme, most recently
as a monitor that reported `Traceback CUDA out of memory` for a run that was **healthy and three
minutes in**. `v6_chain.py verify` emits a heredoc that reads `/proc/*/cmdline` (its own cmdline is
just `python3 -`, containing neither token), builds the searched string by adjacent-literal
concatenation so `train_v6_staged.py` **never appears literally in the probe**, and emits an opaque
`ZZ…ZZ` marker disjoint from anything it searches for.

⚠️ **Kill by explicit PID.** `pkill -f train_v6_staged` self-matches your own ssh command line: it
returns empty output, looks like nothing happened, and kills your session.

### 2.9 ⛔ `--gate-probes` — supply it on a RE-GATE, never on the launch

The chain deliberately does **not** emit `--gate-probes`. MEASURED 2026-08-16 (defect D3): pointing
it at a file that does not exist yet was only read **after the training loop**, so the run spent its
entire budget and then died **before writing `stage_gate.json` AND before writing `summary.json`** —
i.e. ~3.1 GPU-days paid, no gate produced, and a supervisor left with **no done-marker to stop on**,
which is the resurrection trap. Same class as `t1_eval.py` rolling both arms and then dying on an
import in `analyze()`.

It is now refused at **preflight**, in milliseconds. But the operational rule stands: **run the
frozen battery first, write `probes.json`, then re-gate.** §3.3 has the JSON shape.

---

## 3. Gates — what is measured, what passes, and what happens on FAIL

`--gate` runs automatically at the end of every stage and writes `<out>/stage_gate.json`.
Launching stage N+1 runs `assert_stage_precondition`, which reads stage N's gate.

| stage | REQUIRED probes | passes if | reported-only |
|---|---|---|---|
| **S-W** | P1, P3, P6 | P1 retention **≥ 0.85×** R²(z) at k=10 **per target** · P3 sign **≥ 0.95** on **both** channels · P3 gain median **∈ [0.5, 2.0]** *without post-training* · P6 action-subspace dims **≤ 32** | P2, P5, P8, O6 spectrum |
| **S-T** | TACTICAL family, `sel_gap` | `sel_gap` **≤ 0.5×** the fan oracle **at T1 tier** · TACTICAL confusion improves on the E4.1-derived strata | P7 (ρ ≥ 0.3, CI excluding 0, **per stratum**), LATERAL family, X2 seam |
| **S-S** | ⭐ **STRATEGIC family + `sel_gap_revalidated` + `TACTICAL_revalidated`** — THREE, not one | STRATEGIC **computable at all** (measured vs `n/a` today) · `sel_gap_revalidated` still **≤ 0.5×** the fan oracle *after* S-S moved `e_g_tac`, paired bootstrap vs the S-T reading · TACTICAL does not regress vs its S-T reading | S1 ADE(8–30 s) vs CV/corridor at T1, X2 seam |
| **S-J** | X3 isolation, no-harm | zero live forbidden edges · battery **FLAT** across the joint phase | P1/P3/P6, four families |

### 3.1 Three verdicts — and why the third exists

| verdict | meaning | effect |
|---|---|---|
| `pass: true` | every required probe ran and passed | may launch the next stage |
| `pass: false` | a required probe **FAILED** | ⛔ **REFUSED, no override.** X5: *a failed stage never propagates upward.* A FAIL is a finding **about the layer below**; propagating it is how a defect gets attributed to the wrong layer three stages later |
| `pass: null` | a required probe **did not run** or reported null | ⛔ **INCONCLUSIVE IS NOT A PASS.** Refused unless `--allow-inconclusive-gate` **AND** a non-empty `--gate-off-reason`, stamped into `config.json` and printed as a banner |

An unavailable probe is recorded with **what** could not be reached and **where it lives**
(`STAGE_GATE_SPEC[...]["owners"]`) — never silently dropped, never counted as a pass.
**X3 is the one gate this module measures on its own, always.**

### 3.2 Pre-registered FAIL branches — both outcomes committed in advance

| stage | on FAIL | ⛔ what is NOT allowed |
|---|---|---|
| **S-W / P1 retention** | the WM did not retain the driving targets. Branch to the **objective**, not the schedule: the LF-levers (`JEPA_PHYSICS_SURVEY §3`) — O3 masking rate, O2 τ, then LF4 (`--o5-k 60`) — one lever at a time | re-running with more steps and re-reading the gate. `CLAUDE.md`: an exponent below R² 0.80 has no quotable value and **may never decide a restart** |
| **S-W / P3 sign or gain** | the action interface is muffled — this is the **exact** defect stage-A repaired post-hoc (0.27 → 0.97) and O1 exists to prevent from step 0. If O1 does not prevent it, **O1's response form is the finding**, and the fix is the `L_ctrl` weighting, not the trunk | declaring it fixed from a **trainer log**. Trainer val watches a curve; only eval output is quotable |
| **S-W / P6 dims > 32** | the action subspace did not factorise. Report it and continue to S-T **only** if P1+P3 pass — P6 is a structure diagnostic, and a FAIL here is a paper result | quietly widening the threshold after seeing the number |
| **S-T / `sel_gap`** | ⚠️ **the most likely FAIL on the board.** See §5 risk R2 — selection has failed three times already. Branch: the noise-robust selection rule (top-m, not argmin) + `--w-prog`, pre-registered off the banked windows **before** S-T is re-run | reading a T0 number as the verdict. The gate says **at T1 tier** |
| **S-S / STRATEGIC** | if PH2 has not landed this is `pass: null`, not `pass: false` — the D6 amendment is the answer, not an override | `--allow-inconclusive-gate` without the amendment, as a habit |
| **S-J / no-harm** | the joint phase moved the battery ⇒ **stop S-J and keep S-S's checkpoint.** That is the H-COTRAIN rule applied preventively | keeping S-J because its planner number improved while the battery moved |

### 3.3 Folding in the externally-run battery

The battery is a **separate, frozen instrument by design** — an instrument that moves with the
trainer stops being a yardstick. Run it, then:

```bash
cat > /workspace/experiments/v6-SW-30k/probes.json <<'JSON'
{"P1": {"pass": true, "status": "run", "artifact": "p12_gate.json"},
 "P3": {"pass": true, "status": "run", "artifact": "w3_gate.json"},
 "P6": {"pass": true, "status": "run", "artifact": "w3_gate.json"}}
JSON
# then re-run the gate hook with --gate-probes, or pass it on the next launch
```

---

## 4. The cost ladder — ⚠️ ESTIMATED, and the re-cost is the first action on the pod

### 4.1 The basis (MEASURED — the only two rows that anchor this)

| run | wallclock | steps | s/step | source |
|---|---|---|---|---|
| `flagship4b-speedjerk-30k` (v1) | `wallclock_s` **191 206.2** | 30 000 | **6.374** | `MODEL_REGISTRY.md §1.2` |
| `flagship-v4-fromscratch` | **59.04 h** = `wallclock_s` **212 544.6** | 30 000 | **7.085** | `MODEL_REGISTRY.md §1.5.5` |

> ⚠️ **Conflict reported.** `V6_TRAINER_DESIGN §5` attributes the 59.04 h basis to **"v4.2"**.
> The registry says it belongs to **`flagship-v4-fromscratch` (§1.5.5)**; `flagship-v4.2-30k`
> (§1.5.3) is a different arm, **killed at ~step 5 k**. **Registry wins** — the estimate is
> unaffected (same two MEASURED runs), the name is not. Filed as **F4**.

### 4.2 The ladder — ⭐ RE-BASED ON THOR'S MEASURED RATE (2026-08-16)

⛔ **The A40 band below is superseded for S-W.** S-W runs on **Thor at a MEASURED 27.18 s/step**,
batch 8 (marginal over steps 6300→6400; 50-step 27.21 · 100-step 27.18 · cumulative-with-startup
27.32, agreeing to 0.5 %). An A40 would be **20.46** s/step (1.33× faster).

| stage | steps | s/step | wall-clock | class |
|---|---|---|---|---|
| **S-W** *(remaining ≈23,600)* | 23 600 | **27.18** | **178.2 h ≈ 7.42 d** | ⭐ **MEASURED rate**, ESTIMATED remainder |
| **S-T** | 10 000 | 27.18 | 75.5 h ≈ **3.15 d** | ESTIMATED |
| **S-S** | 8 000 | 27.18 | 60.4 h ≈ **2.52 d** | ESTIMATED |
| **S-J** *(optional)* | 3 000 | 27.18 | 22.7 h ≈ **0.94 d** | ESTIMATED |
| **total after S-W** | 21 000 | | **158.6 h ≈ 6.61 d** | ESTIMATED |

⚠️ **Evidence class matters here, and the arithmetic hides it.** 27.18 is **S-W's** rate, and
**S-T/S-S/S-J have never run.** S-T trains only `layer_tac` + `planner` on a frozen trunk and should
be **cheaper per step**; its `--max-horizon 60` cuts windows/episode by ~42.6 %, which changes the
*epoch*, not the *step*. ⇒ these are **upper-ish estimates at S-W's rate**, and **nothing here may
be quoted as a measured S-T cost.**

**Why an estimate must not be trusted as-is:** this programme's own history says estimates here run
**~11 % low** — `MODEL_REGISTRY §1.5.5` records *"~53 h ESTIMATED"* against **MEASURED 59.04 h**,
understating the spend by **≈2.5 GPU-days**. *(The old A40 ladder — S-W 175–290 A40-h, total
≈220–370 — is kept only in §4.1's basis; it costed a machine the run never used.)*

### 4.3 ⛔ RE-COST FROM THE RUN'S OWN LOG, AT STEP 500 OF **EVERY** STAGE

The trainer logs `step_s` **already divided**, with `step_s_note` naming the divisor — precisely so
nobody re-derives the false *"training is 430 s/step"* alarm from an accumulated counter (the older
trainers accumulate `step_s` over `--log-every`, ÷50).

⚠️ **The old D1 branch table (21–35 → proceed · >50 → STOP) was built on a band from two MEASURED
A40 runs, and the ladder now runs on Thor.** Do not read 27.18 as "in band" — it is a different
measurement of different silicon. **The rule that survives is the procedure, not the numbers:** read
`step_s` at step 500 of each new stage, compare it to **that stage's own** expectation, and treat
**>2× the reference** as a defect to diagnose rather than a cost to accept. `--o5-k 10` is the
first lever.

**Cost levers, in the order they cost the least science**

| lever | effect | what it costs |
|---|---|---|
| `--o5-k 10` (from 20) | halves the O5 roll **and** the future-frame encode | O5's compounding shaping measured over 1 s instead of 2 s |
| ~~`--batch 8`~~ | ⛔ **NOT A LEVER ON THOR — it is already the default.** Thor's 20 SMs saturate at batch 8; throughput is FLAT at 12.3–14.1 windows/s across a 6× batch range, so there is no throughput to buy back by lowering it further, and raising it to the A40's 16 buys nothing and costs memory | — |
| `--v2-lru <n>` | the real host-RAM knob **on this trainer** — ⚠️ *not* dataloader workers, of which it spawns **zero** | a smaller frame cache ⇒ more re-decodes. Default in the chain: **6** |
| drop the O1 `random` arm | 5 rolls instead of 6 | −17 % of O1; the named channels are the gated ones |
| `--steps 20000` | linear | shorter ladder — state it in the run row |
| ⛔ **not a lever** | reducing `--plan-steps` below 60 | that is §4b, and it is **binding** |

---

## 5. Risk register — what could make v6 fail, and the early warning

Each row names the **MEASURED** failure that earned it. The early-warning column is what to watch
in the first hours, not after 30 k steps.

| # | risk | why it could bite v6 | ⭐ EARLY WARNING | defence |
|---|---|---|---|---|
| **R1** | **representation collapse** — the latent narrows and the WM stops carrying the world | a from-scratch trunk with five simultaneous losses is exactly where collapse lives | the **O6 spectrum series**, logged every `--spectrum-every 200`: participation ratio and top-k share. **Falling PR in the first 2 k steps is the alarm** | SIGReg (LeJEPA, λ 0.1, 512 slices) is the defence and it is **MEASURED to work**: retention **1.532** vs the ≥0.8× gate — effective rank *expanded* 53 % (PR 4.53→6.94 of 2048) across the full λ_plan ramp (`h_cotrain_curve.json`). ⚠️ that was measured on the **v5f** run, so S-W re-measures it as a **series**, never assumes it |
| **R2** | **selection fails again — the winner's curse** | argmin over a large noisy-cost fan selects for cost **under-estimation**. MEASURED: W7-FULL `sel_gap` **3.2075** vs the ≤0.4505 m gate (**FAIL**), and the roll-cost's argmin sits at error-rank **132.32 of 256** — the median, i.e. **no better than chance at the extreme** (`w7_full_gate.json`, `w7_selection_rules.json`, EXPLORATORY) | S-T's `plan_ade_0_2s` vs `plan_ade_2_6s` **reported separately** from step 1; and the shortlist **ceiling** vs the **pick** — a falling ceiling with a flat pick is the winner's curse, live | the top-m ceilings **do** fall (0.356 at m=32), so a better rule inside a top-m set exists — **just not argmin**. Pre-register the noise-robust rule and the `--w-prog` anti-degeneracy term (**weight 0.0 in every W7 run to date**) off the banked windows, 0-GPU, **before** S-T. Filed as **F3** |
| **R3** | **consumer invalidation** — S-T's planner is trained on S-W's trunk, and any later trunk change invalidates it | MEASURED: repairing the trunk moved the frozen selector from **0.7933 → 4.4159** — a **5.6×** degradation of a consumer that was never retrained (`w7_full_gate.json`). *You cannot repair a trunk and keep its planner* | any change to S-W's checkpoint after S-T starts — including an "innocent" resume from a different `ckpt.pt` | `--init-from` is `strict=True` and the run config records the **md5 of the loaded trunk**, so a row always names exactly which S-W it stands on. ⇒ **freeze and md5 the S-W checkpoint the moment its gate passes**, and never re-run S-W into the same directory |
| **R4** | **the action-echo trap** — open-loop "skill" that is action recitation | MEASURED (`EVAL_DOCTRINE §1.12`): S-curve reproduction **97.9 % open-loop, 0.0 % hold-action, ~5 % closed-loop**. It is the reason T1 is the primary tier | any capability claim quoted from a **T0** number, or from a trainer log | **T1 is the primary tier and the gates say so** (`sel_gap ≤ 0.5× oracle at T1 tier`). T0 is a WM diagnostic, never driving performance. Instrument: `taniteval/tools/t1_eval.py` |
| **R5** | **stage N+1 silently trains on a random trunk** | `--init-from` omitted; the gate passed, the log looks healthy, and the ladder is four unrelated models | the run config's recorded trunk md5 being absent | preflight **refuses** S-T/S-S/S-J without `--init-from`; `strict=True` load |
| **R6** | **the 6 s horizon is quietly abandoned as "a corpus limitation"** | inheriting v4's `plan.max_horizon` (**MEASURED 20**) makes §4b untrainable — S-T could never start, because a 6 s planner has no target in a 2 s window | the `[v6] windowing:` line printing 20 on an S-T launch | v6 **derives its own** and prints it beside v4's; below-need / below-`maneuver_h` / 0-window are explicit refusals. *A silently shortened horizon is not the same experiment* |
| **R7** | **a supervised run is RESURRECTED days after it finished**, or a relaunch restarts at step 0 over a live checkpoint | MEASURED 2026-08-09/11: the v5f run finished but never wrote its done-marker; its supervisor relaunched for **two days**, then a fixed crash-cause let a relaunch SUCCEED, resume from a stale `ckpt.pt`, and overwrite `config.json`/`metrics.json`/`ckpt.pt` **next to a live eval** | any `ps` showing a supervisor for a run whose `summary.json` says done | the trainer writes `summary.json {"done": true}` **in the same turn it finishes**; `resume_guard` refuses a DONE dir (only `--force-rerun`) and refuses `--resume off` over an existing `ckpt.pt`. ⭐ the done-marker is also the correct **remote off-switch** |
| **R8** | **the manifest edit that changes nothing** | `supervise_run.sh` **sources its manifest ONCE, at supervisor startup**, and replays the captured `TRAIN_CMD`. The relaunch looks successful while running the OLD config | flags in `ps` disagreeing with `runs.d/<run>.env` | to change a supervised run: edit the manifest → kill the **SUPERVISOR** first → kill the trainer → start a fresh supervisor. **Verify by grepping the flags out of the RUNNING process**, never by reading the manifest. ⚠️ and do not restart immediately — the new supervisor **races the old one's `flock`**, prints *"another supervisor holds …lock"* and dies, leaving **nothing running** |
| **R9** | **a probe that reports the wrong scope, read as an answer** | `df` reports the 965 TB cluster and hides the pod quota (a full quota killed the flagship mid-checkpoint); cgroup `memory.usage_in_bytes` counted **37.2 GB of reclaimable page cache with nothing running** and cost ~40 min of training to an invented OOM diagnosis | any alarm raised from a single system counter | disk ⇒ a real `dd` write test. Memory ⇒ `memory.stat`'s **`rss`** and **`memory.failcnt`** — `failcnt 0` settles it. **Verify before alarming**, always two samples |
| **R10** | **the estimate is the plan** | ESTIMATED 175–290 A40-h is an extrapolation, and this programme's estimates ran ~11 % low once already | `step_s` at step 500 outside 21–35 | the D1 re-cost rule, with all four branches pre-registered **before** the number is seen |
| **R11** | **`pkill -f train_v6_staged` kills your own ssh session** | the pattern self-matches the ssh command line — it returns empty output and looks like nothing happened | — | kill by **explicit PID** |
| **R12** | **concurrent arms at GPU `sm` 0–6 %, looking exactly like a hang** | torch spawns ~113 threads **per process**; MEASURED 2026-07-27: 7 arms, **50 minutes**, zero progress → `OMP_NUM_THREADS=6` and the same arm finished in **232 s** | `nvidia-smi` `sm` in single digits with jobs "running" | `OMP_NUM_THREADS=6` in the launch line so it is visible in `ps`. **Do not diagnose it as a deadlock** |

### 5.1 The one risk with no built-in defence — ⭐ UPDATED 2026-08-16: IT NOW HAS ONE

**R2 (selection) was the open one.** It is now **adjudicated rather than mitigated**: E-WC2 fired
**REFUSED** (§2.6), so the default S-T runs `--selector none` and no scorer is trained at all —
and `assert_selector_admissible` **refuses at launch**, so the risk cannot re-enter by someone
adding a flag at 3 a.m.

⚠️ **That is not the same as "selection is solved".** What changed is that we stopped paying GPU-days
to find out. The finding is that **the frozen REF-C latents are the wrong surface** — a 0-parameter
constant-yaw-rate goal is 3.96× better than a ridge on them, and *still* not funded. **S-T's gate
still carries `sel_gap`**, and the committed branch is **`ANCHOR_GOAL` supervision**, not another
cost function. The one thing that reopens a selector arm is the pre-registered **E-WC2-SW dump**
(§2.6.1), ~10–25 GPU-min at the S-W → S-T boundary, with its thresholds already committed in code.

---

## 6. Deliverable manifest

| artifact | where it lives | state |
|---|---|---|
| `…/incoming/2026-08-07-hierarchical-wm-redesign/V6_GO_PACKAGE.md` | this file — repo working tree + index | **staged**, verified with `git ls-files --cached` |
| `…/incoming/2026-08-07-hierarchical-wm-redesign/PI_DECISIONS_2026-08-12.md` | repo working tree + index | **staged**, verified with `git ls-files --cached` |
| `Project Steering/BACKLOG.md` §F | repo working tree + index | **staged**, verified with `git ls-files --cached` |
| the v6 trainer itself | `stack/tanitad/models/v6.py` · `stack/scripts/train_v6_staged.py` · `stack/tests/test_v6_staged.py` | already in HEAD `2b8d09e` — **not** stranded |

**This agent committed nothing and pushed nothing** — the `AGENT_OPERATING_STANDARD` contract.
**Nothing produced here lives only on a pod or only in a worktree.**

### 6.1 Escalations — stated here, not buried

1. ⛔ **S-W is GO on code and blocked ONLY on D1 (cost) + D2 (pod).** Both are provision/spend.
2. ⚠️ **S-S cannot pass a valid gate before PH2** — `STAGE_GATE_SPEC["S-S"]["required"]` is
   `("STRATEGIC_family",)` and S2 is unwired. Needs the **D6 amendment in `stack/`**, which this
   agent does **not** own. **Filed as F1 with an owner-file, not written into a doc as
   "please merge".**
3. ⚠️ **Registry-vs-doc conflict:** `V6_TRAINER_DESIGN §5` names the 59.04 h basis "v4.2"; the
   registry (§1.5.5) says `flagship-v4-fromscratch`. **Registry wins.** F4 fixes the doc — this
   agent does not own `V6_TRAINER_DESIGN.md`.
4. ⚠️ **23 standing `pytest` failures**, none of them v6, against the *"`pytest -q` must stay
   green before any commit"* invariant. F5.
