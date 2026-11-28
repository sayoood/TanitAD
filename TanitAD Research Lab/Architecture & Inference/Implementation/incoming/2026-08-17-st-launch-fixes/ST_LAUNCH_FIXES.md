# S-T LAUNCH — Thor SYNCED, E1/E2/E5 FIXED AND EXECUTED, E4 PINNED

**Date:** 2026-08-17 · **Branch:** `agent/arch-inf-20260803` (HEAD `1ebd261`) · **Author:** arch-inf subagent (S-T launch-fixes stream)
**Source of truth for the defects:** `…/incoming/2026-08-17-st-launch-readiness/ST_LAUNCH_READINESS.md`
**Deadline this serves:** `v6F-SW-30k` reaches 30 000 in ~5.3 days. **Nothing was launched. Thor's GPU was never touched** — every Thor call was a read, an `scp`, a `tar -x`, or a CPU-only import with `CUDA_VISIBLE_DEVICES=""`.

---

> ## ⛔ ESCALATION — FOUR THINGS THE ORCHESTRATOR MUST CARRY, NOT A README
>
> **1. THOR IS SYNCED AND THE LIVE RUN NEVER FLINCHED.** 14 files shipped (8 drifted + 6 absent), md5-verified on both sides, and every ⛔ row of the readiness probe flipped: `STAGE_MAY_INTRODUCE`, `RESUME_CONTRACT`, `LADDER_UNTRAINED_GROUPS` present; `--selector`, `--tac-goal-cond`, `--dump-seam-plan` in the CLI; `v6_chain` imports. **PID 25477: step 12 300 before, step 12 450 after two ships, `step_s` 26.4759, GPU 98 %, `Rsl`, still no supervisor.** ⭐ And `v6_chain.py commands --step S-T` now runs **ON THOR**, reads the live run's own `config.json`, and emits the full geometry with **no `--geometry-from`** — §3.5.
>
> **2. ⚠️ RETRACTION AGAINST THE READINESS STUDY — 7 OF ITS 10 "DRIFT" ROWS WERE A CRLF/LF ARTIFACT, NOT DRIFT.** Its md5 table compared the dev box's **CRLF** working-tree bytes against Thor's **LF** bytes. Normalising **both** sides, `tactical.py`, `train_flagship_v4.py`, `eval_flagship_v4.py`, `v2_dataset.py`, `parity.py`, `calib.py` and `geometry.py` are **byte-identical** to Thor's copies. The two rows that mattered (`train_v6_staged.py`, `v6.py`) were real, and so were 6 outright absences. §1. **Root-cause class: a comparison run in the wrong normalisation — the `df`/Thor-`free`/cgroup family, where a probe answers a different question than the one asked.**
>
> **3. E1's FIX EXPOSED A SECOND, WORSE DEFECT IN THE SAME GUARD, AND IT IS FIXED TOO.** `assert_geometry_carry` read only `<prev.out>/config.json`; on the dev box — *where launch commands are actually generated* — that path does not exist, so the guard returned `ok: None` and **a geometry-free argv passed it**. MEASURED: the first version of my own fix shipped that hole. §3.3.
>
> **4. ⛔ E4 IS **NOT** FIXED — IT IS PINNED, AND THE PI OWES A DECISION BEFORE S-T's GATE IS READ.** The S-T gate requires `sel_gap`, which is emitted only when `w_select > 0`, which requires `--selector != none`, which SEL-1 refuses. The verdict is **INCONCLUSIVE by construction**. `stack/tests/test_v6_st_launch_fixes.py::test_E4_PIN_*` (3 tests) holds it still with both legitimate fixes written down, so it cannot be silently resolved in a direction nobody chose. §6.
>
> **5. ⚠️ I TOUCHED TWO FILES OUTSIDE MY DECLARED OWNERSHIP, AND BOTH WERE FORCED BY A RED SUITE — SAY SO WHEN YOU COMMIT.** `V6_GO_PACKAGE.md` §2 and three tests in `test_runbook_commands.py`/`test_v6_gstr_port.py`. **The runbook was banking a launch line the trainer's own preflight now REFUSES** — the E2 sub-frame, in the document an operator opens at 3 a.m. — and `test_every_runbook_launch_line_passes_the_trainers_own_preflight` caught it the moment the guard existed. §7.

---

## 1. Thor — what was actually drifted, and what was a measurement artifact

⛔ **`git fetch` was never run on Thor** (it hangs — no credentials — and a `checkout -B` after a failed fetch resets the tree to an ancient HEAD, destroying shipped files). Everything below is md5s and a real import.

### 1.1 The ship set was DERIVED, not taken from the 13-file list

The launch's import closure was computed by **importing it**: `train_v6_staged`, `v6_chain`, every lazy import read out of the trainer's own source (`tanitad.models.metric_dynamics`, `tanitad.models.flagship_v15`, `s2_labels`, `eval_flagship_v4`, `train_flagship4b`, `train_flagship_v4`, `train_v58f_unicycle_head`, `taniteval.seam*`), plus the S-W gate battery (`probe_latent_state`, `stage_a_probes`) — then every loaded module whose `__file__` lives in the repo.

**76 repo files**, against the readiness study's 13. Producer: `code/thor_closure.py` · raw: `raw/thor_closure.json`.

### 1.2 ⚠️ The correction: 7 of 10 "drift" rows were CRLF vs LF

The dev-box working tree is **mixed** — 41 of the 76 files are CRLF, 35 LF. The readiness study md5'd the raw working-tree bytes on one side and Thor's LF bytes on the other.

| file | repo RAW (CRLF) | repo LF-NORMALISED | Thor | verdict |
|---|---|---|---|---|
| `stack/tanitad/models/tactical.py` | `c45375b8…` | **`6e8c80c4…`** | **`6e8c80c4…`** | ⭐ **IDENTICAL** |
| `stack/tanitad/data/parity.py` | `7b429918…` | **`eb8c2dad…`** | **`eb8c2dad…`** | ⭐ **IDENTICAL** |
| `stack/scripts/eval_flagship_v4.py` | `21c1e0b5…` | **`bd021afa…`** | **`bd021afa…`** | ⭐ **IDENTICAL** |
| `stack/tanitad/geometry.py` | `b7798dfe…` | **`18969b47…`** | **`18969b47…`** | ⭐ **IDENTICAL** |
| `stack/tanitad/models/v6.py` | `1e91b074…` | `f36b476d…` | `24160085…` | ⛔ **REAL DRIFT** |
| `stack/scripts/train_v6_staged.py` | `a37ddba0…` | `a37ddba0…` | `97daef49…` | ⛔ **REAL DRIFT** |

The same held for `train_flagship_v4.py`, `v2_dataset.py` and `calib.py`. `bev_raster.py` is a third case: **both sides are byte-identical CRLF**, so it was neither drift nor shipped.

⚠️ **Thor's side was normalised too** (`tr -d '\r' | md5sum`), because a one-sided normalisation is the same error in the other direction.

**True state of the closure: 61 identical · 8 drifted · 6 absent · 1 identical-but-CRLF.**

### 1.3 The 14 files shipped, md5-verified on both sides

⛔ **8 DRIFTED** — `train_v6_staged.py` `a37ddba0…` · `v6.py` `f36b476d…` · `sigreg.py` `494d8448…` · `v6_probe_trunk.py` `c8bea043…` · `train_flagship4b.py` `441e7c85…` · `train_p8_occupancy.py` `86aed2be…` · `refb_labels.py` `bf190dd3…` · `tools/eval_four_families.py` `4017cfe4…`
⛔ **6 ABSENT** — `v6_chain.py` `ee638491…` · `s2_labels.py` `d389f454…` · `models/agent_slots.py` `39ac4d81…` · `taniteval/seam.py` `74d99d4b…` · `taniteval/seam_dump.py` `0ae78592…` · `tools/seam_probe.py` `1acec152…`

Method: LF-normalised locally → `tar.gz` (md5 `ea4e5f5d…` verified on both sides) → `scp` → **backup of the 8 pre-existing files to `tanitad-thor-wifi:/home/nvidia/_thor_backup_2026-08-17-stsync/pre_ship.tgz`** → extract → per-file `md5sum`. **All 14 match the bytes sent.**

### 1.4 Why it was safe, verified rather than assumed

* `ps -eo pid,cmd | grep supervise` → **0**, before and after. Nothing can relaunch PID 25477 with new code.
* A running Python process has already imported its modules, so replacing `.py` on disk cannot reach it — but the trainer does have **lazy** imports. Each was checked: `flagship_v15` arrives at startup through `eval_flagship_v4`/`train_flagship_v4` (both imported in the setup path) and `_lift3` is called every step, so it is long since cached; `s2_labels` is gated on `--s2-labels` and `taniteval.seam_dump` on `--dump-seam-plan`, and the live run carries neither.
* `sigreg.py` was the one drifted file in the live loss path. Its diff (55 lines) **adds an opt-in `generator=` argument and states in code that `generator=None` must stay the literal incumbent call so a resumed v6F run reproduces its loss**. A relaunch is loss-continuous.

### 1.5 The import probe after the ship — every ⛔ row flipped

`code/thor_import_probe.sh` re-run verbatim (`/home/nvidia/venvs/tanitad-train/bin/python`, `CUDA_VISIBLE_DEVICES=""`):

| symbol / flag | before | after |
|---|---|---|
| `STAGE_MAY_INTRODUCE` · `RESUME_CONTRACT` | ⛔ absent | ✅ **present** |
| `--selector` · `--tac-goal-cond` · `--dump-seam-plan` in the CLI | ⛔ absent | ✅ **present** |
| `v6.LADDER_UNTRAINED_GROUPS` | ⛔ absent | ✅ **present** |
| `import v6_chain` | ⛔ `ModuleNotFoundError` | ✅ `/home/nvidia/TanitAD/stack/scripts/v6_chain.py` |
| `import taniteval.seam_dump` under `PYTHONPATH=<stack>` | ⛔ absent module | ⛔ **still `ModuleNotFoundError`** — the FILE now exists; the PATH is E5 |
| `import taniteval.seam_dump` under `<stack>:<taniteval>` | ⛔ absent module | ✅ `/home/nvidia/TanitAD/taniteval/taniteval/seam_dump.py`, symbols resolve |

The last two rows together are E5 in one line: **shipping the file was necessary and not sufficient.**

### 1.6 Live-run health, before and after

| | before ship | after ship |
|---|---|---|
| PID 25477 | `Ssl`, elapsed 1-20:49 | ✅ `Ssl`, elapsed 1-21:24 |
| step | 12 300 | ✅ **12 450** (two ships later) |
| `step_s` | 26.4797 | 26.4759 |
| loss | 2.0354 | 2.0775 |
| GPU | — | 98 % |
| supervisor | none | none |

---

## 2. What is fixed, and the one-line proof of each

| | defect | fix | the executed proof |
|---|---|---|---|
| **E1** | the emitted S-T line carried **no model geometry**, so the stack built at trainer defaults | `trainer_argv` **carries** the ancestor's geometry out of its own `config.json`; `assert_geometry_carry` diffs the **full derived set** | `load_stage_init` against the **REAL** S-W snapshot: `missing []`, `unexpected []`, `introduced [cond_tac_dyn.bias, cond_tac_dyn.weight]` |
| **E2** | `--v2-subframe 176x624` moved the DATA, not the MODEL; died at the first forward | flag removed from the chain; **two** guards in the trainer, one args-only at startup | preflight returns the refusal; the raw encoder still raises `ValueError: encoder input is (176, 624) but the config declares (256, 640)` |
| **E5** | the seam dump was never turned on, and its import failed under the launch's PYTHONPATH | chain emits `--dump-seam-plan` (S-T/S-S/S-J) and **both roots** on `PYTHONPATH`; trainer **refuses at startup** | a subprocess with `PYTHONPATH=<stack>` exits **2** in seconds with the named fix |
| **E4** | the S-T gate is INCONCLUSIVE **by construction** | ⛔ **not fixed — PINNED.** Both legitimate fixes recorded; the choice is the PI's | `stage_gate_dict("S-T", …)` → `INCONCLUSIVE` with every other required probe passing |

---

## 3. E1 — the geometry, carried and checked

### 3.1 ⭐ The comparison set is DERIVED from the trainer's source, never listed

A hand-list fixes today's two omissions and installs tomorrow's — the `LADDER_UNTRAINED_GROUPS` lesson (`718855f`). So `v6_chain.geometry_dests()` parses `build_stack_from_args` with `ast` and collects **every argparse dest that function reads**, in all three shapes present in the real code: `a.enc_dim`, `getattr(a, "mpc_topk", 2)`, and `resolve_gc(a, "enc_grad_checkpoint")`.

**MEASURED: 76 dests.** A new `V6Config` lever wired to a new flag joins the check the moment it is wired — `agent_slots`, `proposals` and `mpc_refine` are already in the set with no edit to `v6_chain`. It **refuses** if it ever derives zero, because an empty check that returns OK is worse than no check.

The flag spec (`opt`, `action`, `nargs`, `default`) is read the same way, out of `build_parser`'s `add_argument` calls, so `plan`/`commands`/`status` keep working on a pod whose torch a `uv pip install` has broken. `test_the_arg_spec_read_from_SOURCE_matches_the_REAL_parser` pins that read against the real parser, dest by dest.

⚠️ **The AST read alone was NOT enough, and the suite is what proved it.** `--a-max`, `--kappa-max` and `--plan-steps` default to **imported constants** (`A_MAX`, `KAPPA_MAX`), which `ast.literal_eval` cannot evaluate. The first version therefore compared an ancestor's real `4.0` against a successor's `None` and **falsely refused the dry ladder**. Fix: the real parser's defaults win when it is importable, and a dest whose default is genuinely unknowable is **skipped** rather than compared against `None` — a guard that cannot see a field must say so, never invent a mismatch.

### 3.2 The carry is `geometry_dests()` and nothing else

⛔ **A blanket "carry every recorded arg" is tempting and wrong.** The ancestor's record also holds RUN CONTROL — `--force-rerun`, `--resume`, `--gate-off-reason`, its own `--out` — and carrying `--force-rerun` forward would let a successor **overwrite a DONE run**. Pinned by `test_E1_the_carry_never_forwards_RUN_CONTROL`.

⛔ **Geometry is emitted even when it equals today's default.** An "omit the defaults" line means whatever the defaults happen to be *on the day it runs*. MEASURED while building this: the omit-defaults version silently dropped `--frame-h/--frame-w/--patch/--window/--horizons/--d-goal-embed/--adapter-hidden/--dt/--uplink/--ema-decay/--sigreg-slices/--n-registers` — all correct that day, all silently wrong after one default moves.

**The skip set is derived from the argv, not listed:** every flag `trainer_argv` already decided (the per-stage ones, the data ones, the identity block) is excluded *because it is present*, so a new per-stage flag needs no edit, and the carry is spliced **after** the identity block so `step.extra` still wins on any collision.

### 3.3 ⛔ The guard's second defect — and it was in my own first fix

`assert_geometry_carry` read only `<prev.out>/config.json`. On the dev box that path does not exist, so it returned `{"ok": None}` — **"could not verify" reported as not-a-failure** — and the geometry-free argv passed. MEASURED: `e1_guard_on_old_argv.refused == false` on the first run of `code/verify_st_launch_fixes.py`.

The guard now resolves its record through the **same** `geometry_source` the emitter used: the real predecessor `config.json` when readable (the pod case, where the check is against the true record), else `--geometry-from`, else it says it could not verify. Pinned by `test_E1_the_guard_does_not_evaporate_when_the_ancestor_is_off_box`.

It also runs **last** in `assert_may_launch`, after the certificate and the weights: "the stage below did not pass a gate" is the diagnosis an operator needs first.

### 3.4 What the guard says now, on the old line

Fed the pre-fix argv, it refuses on **17 of 76** fields, every value read from the ancestor's own record:

```
[chain] ⛔ S-T's MODEL GEOMETRY does not match its ancestor S-W, on 17 of the 76 fields:
    --enc-dim      ancestor  768   this launch 384
    --enc-depth    ancestor   12   this launch   8
    --pred-dim     ancestor 1024   this launch 768
    --d-tac        ancestor  768   this launch 512
    --vit5-encoder ancestor True   this launch False
    …
```

### 3.5 The line it now emits, and the load it now performs

MEASURED end-to-end by `code/verify_st_launch_fixes.py` — the real `v6_chain.trainer_argv`, the real `build_parser`, the real `build_stack_from_args`, the real `load_stage_init`, against the **REAL** fp16 S-W snapshot (`a4e2c0e1eb0ca455448472853ccc46d7`, 673 312 891 B):

| | pre-fix argv | post-fix argv |
|---|---|---|
| params | **87 926 473** | ⭐ **336 575 049** |
| state-dict keys | 407 | ⭐ **575** |
| `encoder.pos` | `[1, 640, 384]` | ⭐ **`[1, 640, 768]`** |
| `preflight(a)` | — | **`[]`** |
| `load_stage_init` vs the real ckpt | `RuntimeError … size mismatch` | ⭐ **`missing []` · `unexpected []` · `introduced [cond_tac_dyn.bias, cond_tac_dyn.weight]` · `init_step 10000` · `prev_stage "S-W"`** |

That reproduces the readiness study's numbers exactly (336.54 M / 575 keys) from a command the chain now generates on its own.

⭐ **AND IT WAS RE-RUN ON THOR, AGAINST THE LIVE RUN'S OWN `config.json`, WITH NO `--geometry-from` AT ALL.** After the fixed `v6_chain.py` was shipped (md5 `afaa7d94f7a7854dfbfdafad1742fb77`, both sides), `v6_chain.py commands --step S-T` executed in `/home/nvidia/TanitAD/stack` reads `/home/nvidia/experiments/v6F-SW-30k/config.json` directly and emits **`--enc-dim 768 --enc-depth 12 --pred-dim 1024 --d-tac 768 --d-str 512 --frame-h 256 --frame-w 640 --vit5-encoder --pred-modern --grad-checkpoint …`**, with **`--v2-subframe` count 0**, **`--dump-seam-plan /home/nvidia/experiments/v6F-ST-10k/seam`** present, and `PYTHONPATH=/home/nvidia/TanitAD/stack:/home/nvidia/TanitAD/taniteval`. That is the deadline artifact: the operator can now generate the S-T launch **on the box**, from the run's own record, with nothing typed by hand.

⚠️ **One deliberate difference from the banked `st_launch_line.txt`:** it carries `--o5-k 60 --o1-k 10 --eps-per-batch 4 --wd 0.05 --clip 1.0 --log-every 50 --device cuda --seed 0`; the chain emits only what is non-default, plus an explicit `--save-every 250` (the trainer's default is **1000**, and the seam dump fires at that boundary). `o5_k` is inert at S-T — `for_stage("S-T")` zeroes `o5`, so `needs_ztrue` is False and `need_k` is 0.

---

## 4. E2 — the sub-frame, refused in milliseconds instead of after the corpus

`--v2-subframe` is gone from `trainer_argv` (the live S-W run has `v2_subframe: null`, and the ladder carries that forward). The check moved to **two** places, both before the first forward:

1. **`train_v6_staged.subframe_desync(a)` — args only**, called from `preflight`, so it fires at startup in milliseconds. A **no-op** sub-frame equal to the declared frame is consistent and is NOT refused (MEASURED across `None`/`"none"`/`"256x640"`/`"176x624"`).
2. **After `resolve_eval_frames`, on the real objects** — the frame the DATA will be delivered at against `stack.cfg.encoder.image_hw()` — still before the corpus mounts. Two locks because in this trainer they are genuinely different objects.

MEASURED on the built production encoder: the forward at the declared frame returns `[1, 640, 768]` in 0.5 s; the raw encoder at `(176, 624)` still raises `ValueError: encoder input is (176, 624) but the config declares (256, 640)`. The guard is what now stands between that error and 1.8 h of mounted corpus.

⚠️ **The refusal names both legal ways out** — drop the flag, or declare `--frame-h 176 --frame-w 624` — and says plainly that the second is a **different model** that cannot `--init-from` a 256x640 checkpoint.

---

## 5. E5 — the seam dump: on, importable, and loud

`SEAM_INSTRUMENT.md` §8 is **corrected in place** (the blocker was closed in `6a7c006`; the doc still described it as open).

1. **The chain turns it on** — `--dump-seam-plan <out>/seam` for **S-T, S-S, S-J**, and the launch line `mkdir -p`s the directory. ⚠️ **Not S-W**: its emission head is at zero-init, every control is exactly (0, 0), and `seam_dump_from_plan` correctly refuses a DEGENERATE plan. A degenerate dump cannot answer the seam question and must not be produced to make the row non-empty.
2. **`PYTHONPATH` carries both roots** in `launch_line` **and** `manifest_text`, derived from `workdir` rather than configured — the two directories are siblings by construction and a second knob is a second thing to forget.
3. ⛔ **The trainer REFUSES AT STARTUP.** MEASURED: a subprocess with `PYTHONPATH=<stack>` and `--dump-seam-plan` now exits **2** in seconds:

```
[v6] ⛔ --dump-seam-plan …/seam but `import taniteval.seam_dump` FAILS:
     ModuleNotFoundError: No module named 'taniteval'
     `taniteval` is a SIBLING of stack/, not a member of it … Without this
     refusal the run would bank NOTHING while printing '[v6 seam] dump FAILED
     … — training continues' at every save boundary, the first one ~1.8 h in.
     ⇒ PYTHONPATH=<repo>/stack:<repo>/taniteval …
```

The **in-loop** catch stays non-fatal — a diagnostic must never kill a 3-day run — which is exactly why the failure had to be moved to parse time. ⚠️ `--dry-run` returns before the save boundary, so this wiring had **never been executed by anything**; the startup probe is what makes that visible before 3.15 GPU-days are spent.

---

## 6. ⛔ E4 — PINNED, NOT FIXED. THE PI OWES A DECISION.

**`STAGE_GATE_SPEC["S-T"]["required"]` contains `sel_gap`.** `sel_gap` is emitted only inside `if w.w_select:`; `w_select > 0` requires `--selector != none` (the trainer refuses otherwise); and the chain's default S-T step is `selector="none", w_select=0.0` because **SEL-1 fired REFUSED** on 2026-08-16 against a pre-registered threshold. **The gate spec and the default plan contradict each other, and the contradiction resolves as "INCONCLUSIVE, always".**

MEASURED: `stage_gate_dict("S-T", {every other required probe passing})` → `verdict INCONCLUSIVE`, with `sel_gap` in the missing/inconclusive list. It propagates — `STAGE_GATE_SPEC["S-S"]["required"]` contains `sel_gap_revalidated`, same dependency.

⚠️ Same class as the three vacuous gates found that week (K3 pinned at 0.5; the pre-S2 goal-provenance audit; `_grad_census`'s zero-parameter group): **a criterion whose verdict is decided by construction rather than by the model.** Here it cannot *pass* instead of cannot *fail*.

**Three tests hold it still** (`test_E4_PIN_the_S_T_gate_cannot_read_PASS_on_the_arm_the_chain_plans`, `test_E4_PIN_it_propagates_to_S_S`, `test_E4_PIN_the_readiness_finding_is_banked_with_both_options`). **They do not choose.** The two legitimate options, both the PI's:

* **(a)** move `sel_gap` from `required` to `reported` **when `selector == "none"`**, and record that the S-T certificate then rests on `TACTICAL_family` alone; or
* **(b)** keep it required and make the refusal explicit and early — `assert_may_launch` refuses an S-T step whose `selector == "none"` while `sel_gap` is required, naming the contradiction, instead of letting it surface as an unexplained INCONCLUSIVE **3.15 GPU-days** later.

The operational cost of not deciding is that S-S gets launched under a blanket `--allow-inconclusive-gate`, which erodes the one mechanism that stops a bad stage propagating.

---

## 7. ⚠️ The runbook was carrying the broken line — and files I touched outside my brief

**`V6_GO_PACKAGE.md` §2 is *the document an operator opens at 3 a.m. and does not re-derive*, and it banked the E2 sub-frame on all four stages.** The moment the guard existed, `test_every_runbook_launch_line_passes_the_trainers_own_preflight` failed with the trainer's own refusal. That test is the mechanism working exactly as designed — the runbook is a **rendering** of `v6_chain.py`, and it went red the instant the generator moved.

**Regenerated** §2.2 from the fixed chain (`--geometry-from <the banked S-W config>`), and updated three prose blocks the change makes wrong: the `PYTHONPATH` line (now **both roots**), the regeneration recipe (now carries `--geometry-from`, and says why it is not optional off-pod), and the flag table (three new rows: `--v2-subframe` **removed**, `--dump-seam-plan` **added**, `--save-every 250` **now explicit**).

⚠️ **Two test-file changes were also forced, and one is a finding in its own right:**

* `test_runbook_commands.py` — `THOR_CFG` gains `geometry_from`; and the preflight test now reads the **`PYTHONPATH` out of the runbook's own command** instead of evaluating the argv under *this process's* environment. Evaluating a correct line under the wrong environment is a **scope error of exactly the kind that file exists to prevent**, and it would have failed a line that is right. It additionally asserts that any line carrying `--dump-seam-plan` **declares a `taniteval` root**, so E5 cannot come back through the runbook.
* `test_runbook_commands.py` — `--uplink` **left** the `prose_only` leak-canary set: since E1 the chain emits geometry explicitly (defaults included), so `--uplink stopgrad` is now genuinely in every launch line. The four remaining canaries are control-arm flags the chain never emits, which is what still makes the check valid.
* `test_v6_gstr_port.py` — three tests asked for an argv with no ancestor record on disk; they now seed it, derived the same way as everywhere else.

⛔ **Declared, not buried:** `V6_GO_PACKAGE.md`, `test_runbook_commands.py` and `test_v6_gstr_port.py` are **not** in this stream's stated ownership. They were changed because the fixes made them red or wrong, and leaving a red suite would have blocked every other stream. **Please review those three in the commit.**

---

## 8. Still open — NOT closed by this stream

* ⛔ **The S-W gate at 30k will read INCONCLUSIVE** unless the **P1/P3/P6 battery** is run against `ckpt.pt` and folded in with `--gate-probes`. That is a **GPU job that must be scheduled inside the remaining ~5.3 days**. Thor's stack is now current, so it *can* run there — the emitters (`scripts/probe_latent_state.py`, `scripts/stage_a_probes.py`) were already present and are byte-identical to HEAD.
* **`--spectrum-accum` is not set on the live run** (only `--spectrum-every 200`), so `O6_rank_retention` and `X4_rank_retention` stay INCONCLUSIVE. Reported-only, so it does not block S-T.
* **E4** — above.
* ⚠️ **`--v2-lru`:** the chain's Thor default is **6**; the live run's measured-stable value is **64**. The emitted command must be generated with `--v2-lru 64` (as the banked one was) until that default is revisited.

---

## 9. Suite

| suite | result | baseline |
|---|---|---|
| `stack` (`PYTHONUTF8=1 OMP_NUM_THREADS=6 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q`) | **3803 passed · 0 failed · 7 skipped · 2 xfailed** (422 s) | 3782 / 0 / 7 / 2 — **+21 is exactly this stream's new file** |
| `taniteval` | **1092 passed · 0 failed** (139 s) | 1092 / 0 |
| ladder subset (`test_v6_chain`, `test_v6_staged`, `test_v6_ladder_edges`, `test_v6_stage_init_introduction`, `test_v6_stage_revalidation`) | **187 passed** | 187 |
| ⭐ new: `test_v6_st_launch_fixes.py` | **21 passed** | — |

⚠️ **The `pytest -q` baseline runs inside `stack/`, where `taniteval` is NOT importable — which is E5's exact geometry, reproduced by the suite itself.** The new test module adds the sibling to `sys.path` explicitly and says why; the test that a launch *without* it is refused runs in a **subprocess** with the narrow path, so that line cannot mask the defect it verifies.

⚠️ **Three fixture changes in `test_v6_chain.py` were required, and one of them is a finding.** `touch_ancestor` wrote a `stage_gate.json` and **no `config.json`** — a state that cannot exist, since the trainer writes `config.json` at startup long before any gate. That omission is part of why E1 survived: the successor's geometry was never compared against anything, because there was nothing to compare it against. The fixture now writes the ancestor's record, with every field derived (the geometry through the chain's own parser, the two introducible fields off the `ChainStep`).

---

## 10. Evidence classes

| claim | class |
|---|---|
| every md5, the closure size, param counts, key counts, `encoder.pos` shapes, `missing/unexpected/introduced_keys`, the `ValueError`, exit codes, gate verdicts, step/`step_s`/GPU | **MEASURED (ours)** — producers in `code/`, outputs in `raw/` |
| the CRLF/LF retraction | **MEASURED (ours)** — both sides normalised, `raw/thor_drift_normalised.json` |
| "no supervisor is running", "the live run has no auto-restart" | **MEASURED (ours)** — `ps`, before and after |
| SEL-1's σ/ADE 9.9915 [7.4492, 13.5119] vs the 3.0 refusal line | **INHERITED** — quoted from `v6_chain.SEL1_ADMISSION`, not re-measured |
| "the fp32 `ckpt.pt` at 30k carries the same key set as the fp16 snapshot at 10k" | **UNVERIFIED** — same writer, same stack; the 30k artifact does not exist yet |
| "HEAD is behaviourally identical to Thor's pre-ship vintage for the live run" | **UNVERIFIED** — key-and-shape identity, argv-parseability and the `sigreg` opt-in contract were checked; behavioural identity was not |

---

## 11. Deliverable manifest

| artifact | where it lives | staged |
|---|---|---|
| `ST_LAUNCH_FIXES.md` (this file) | `repo:TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-17-st-launch-fixes/` | yes |
| `code/thor_closure.py` — derives the launch's import closure by importing it | same, `code/` | yes |
| `code/thor_sync.py` — LF-normalises, tars, md5s the ship set | same, `code/` | yes |
| `code/verify_st_launch_fixes.py` — **executes** E1/E2/E5 end-to-end incl. the real checkpoint load | same, `code/` | yes |
| `raw/thor_closure.json` · `raw/thor_drift_normalised.json` · `raw/thor_ship_verified.json` | same, `raw/` | yes |
| `raw/verify_st_launch_fixes.json` — every number in §3–§5 | same, `raw/` | yes |
| `raw/st_launch_line_fixed.txt` — the command the fixed chain emits | same, `raw/` | yes |
| `raw/v6F-SW-30k.config.json` — the live run's own record, md5 `2cb83239cc19d13b9bc7a49a27459b82` **identical on Thor** | same, `raw/` | yes |
| `stack/scripts/v6_chain.py` — E1, E2, E5 | repo | yes |
| `stack/scripts/train_v6_staged.py` — E2 + E5 preflights | repo | yes |
| `stack/tests/test_v6_chain.py` — fixture correction | repo | yes |
| `stack/tests/test_v6_st_launch_fixes.py` — **21 tests, incl. the 3 E4 PINS** | repo | yes |
| `…/2026-08-16-seam-instrument/SEAM_INSTRUMENT.md` — §8 correction | repo | yes |
| the 14 shipped files | `tanitad-thor-wifi:/home/nvidia/TanitAD/` — **all from the repo, none authored there** | n/a |
| pre-ship backup | `tanitad-thor-wifi:/home/nvidia/_thor_backup_2026-08-17-stsync/pre_ship.tgz` | n/a |
| the pulled fp16 snapshot (673 MB) | **scratchpad only**; source of truth stays `tanitad-thor-wifi:~/ckpt_snaps/` | no (deliberate) |

**Nothing is stranded.** Every file on Thor came FROM the repo; nothing was authored there.
