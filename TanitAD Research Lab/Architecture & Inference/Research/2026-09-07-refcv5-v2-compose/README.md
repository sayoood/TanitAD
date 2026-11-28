# refcv5-v2 — the composed arm, PREPARED AND NOT LAUNCHED

**Author:** TanitAD Architecture & Inference · **Date:** 2026-09-07
**Branch:** `agent/arch-inf-20260803` · **Repo HEAD at composition:** `52ca682`
⚠️ **The repo advanced under this document** (`5088b7f` → `6ae8acf` → `d1c929c` → …);
every finding below was **re-checked against the moving tip**, and the two that went
stale are corrected in place with the correction named, never overwritten.
**Status:** ⛔ **NOT LAUNCHED.** A sibling owns the P1 gate
(`REFCV5_MISSING_PIECES_PLAN.md` §4, restated §8.2). This document makes the
launch instant the moment that gate clears; it does not clear it.

---

## 0. ONE LINE

⭐ **READY:** both launch commands are written, **verified by execution** (38/38
flags, both guard families mutation-proven, a full smoke run stamping
`sampler_ranks_the_fan: True`); the A40 is **idle at 0 MiB / 0 %**; all agent-join
data is staged and md5-verified on the pod; and the comparison protocol is
pre-registered with its trivial-control bar.

⛔ **MISSING — and it is NOT P1:** the pod runs a trainer **845 lines and nine
flags behind HEAD** and cannot accept a **single lever** this composition depends
on, so the true blocker is an unperformed **ship step**. Beyond that: **P4 cannot
enter** (its module is wired to nothing), **D-TRISEG cannot enter** (the decoder
refuses the shape), and **9 of the PI's 12 video observations will not move.**

⛔⛔ **AND THE ONE THE PI WILL CARE ABOUT MOST:** as HEAD stands,
refcv5-v2 would train **NEITHER** the tactical nor the strategic vocabulary
(§4.1) — the tactical head is gated behind a flag that is still in a sibling's
worktree, and the strategic head has no call site at all. The arm would **load**
the v7.2 vocabulary and train against **no head**. The tactical half is **one
commit** away; the strategic half is **not written**.

---

## 1. ⛔⛔ THE FINDING THAT CHANGES THE PLAN: the box is not running the repo

`REFCV5_MISSING_PIECES_PLAN.md` §8 P14 records `sel_refined` / `sel_score_emitted`
at **0 occurrences** in `refc_v3_train.py`, with a control of **`sel_` = 8**.

That census was taken against the **POD's copy**, not the repo.

| surface | lines | `add_argument` | `sel_refined` | `sel_score_emitted` | CONTROL `sel_` |
|---|---:|---:|---:|---:|---:|
| **pod** `/workspace/TanitAD/stack/scripts/refc_v3_train.py` | 3,634 | 79 | **0** | **0** | **8** |
| repo **HEAD** `52ca682` | 4,479 | 88 | 6 | 16 | **33** |
| repo **INDEX** ⚠️ | 4,028 | — | 0 | 0 | — |
| repo **worktree** (sibling mid-edit) | 4,576 | 89 | 6 | 16 | 33 |

pod md5 `acad9ae6e336e52d1b61a314f93a39fa`, mtime **2026-09-05 23:38Z**.

⇒ **The control value 8 is the pod's. HEAD's is 33.** P14's plumbing **HAS
landed** in the repo — commit `c8601d0`, with a paired guard, a refusal on each
half alone, and a `sampler_ranks_the_fan` stamp in `config.json`. It has simply
**never been shipped to the box**. This is exactly the repo→box gap
`AGENT_OPERATING_STANDARD.md` names as C99/C102/C105: *"presence proves transfer,
md5 proves bytes, a successful import proves loading — none of them proves
currency."*

**Nine flags exist in HEAD and are absent on the pod — every one a lever this
composition needs:**

```
--sel-refined  --sel-score-emitted  --sel-score-emitted-t   (P14)
--no-strategic                                              (P8)
--goal-point-inject --goal-point-geo-prior
--goal-point-t --goal-point-w                               (P2)
--nav-args                                                  (P11)
```

Flags on the pod but not in HEAD: **none** — the pod is cleanly **POD-BEHIND**,
not diverged, so the fix is a plain ship. ⇒ **`ship_and_gate.sh` must run before
either launch command.**

⚠️ **And a trap while doing it:** the repo **INDEX** (4,028 lines) is *smaller
and older than HEAD*. `git checkout -- stack/scripts/refc_v3_train.py` would
**regress the file by 451 lines**. Use `git checkout HEAD -- <path>`.

---

## 2. THE EXACT LAUNCH COMMANDS

Full runnable scripts: `LAUNCH_refcv5_v2.sh` (both arms), gated by
`ship_and_gate.sh`. The flag bodies, with every flag justified:

### 2.1 Common core — **identical to refcv5-v1**, so the delta *is* the levers

| flag | why, and the plan row |
|---|---|
| `--arm hier --size base` | refcv5-v1 identity; changing it would confound. |
| `--v2-cache /root/data/train` (4,573 eps) · `--eval-cache /root/data/eval` (142) | the B1 corpus already on the box. |
| `--v7-labels …v7.2_train.jsonl.gz` `--eval-labels …v7.2_eval.jsonl.gz` | the released tactical vocabulary; coverage is printed and <50 % is refused. |
| `--eval-every 500 --eval-batches 8` `--image-hw 256 640` | refcv5-v1. |
| `--steps 40284 --batch 20 --workers 6 --prefetch-factor 1 --v2-lru 24` | refcv5-v1; 40,284 is refcv4b's landing step, so the comparison is step-matched. |
| `--lr 1e-4 --warmup 2000 --seed 0 --log-every 50 --save-every 500 --u8-batches` | refcv5-v1. |
| `--nav-from-v7` | **INPUT 1/3** (nav command). ⚠️ see §3 Finding N — this is an *oracle* nav on this corpus. |
| `--ego-state-inject --ego-dropout 0.5` | **INPUT 2/3** (measured `v0` at t0). `--ego-valid-channel` is set as a **precondition, not an option** (`refc_v3_train.py:234`). |
| `--anchors $OUT/anchors.pt --n-anchors 117 --anchor-v0-conditioned --anchor-control-units alat` | refcv4b's **own** 117-anchor v0-conditioned bank, restamped with declared units + sha256 (`/workspace/anchors_117_alat_declared.pt`). Holding the vocabulary constant across baseline and arm is what makes the comparison clean. |
| `--sel-accel-max 2.0` | refcv5-v1. |
| `--sampler ddim --w-u0 0.5` | `ddim` requires a v0-conditioned vocabulary **and** `w_u0 > 0`. |
| ⭐ `--sel-refined --sel-score-emitted` | **P14, the one validated new lever.** |
| `--goal-str` | refcv5-v1's geometric 3-unit bearing head. ⚠️ **not** the 15-token vocabulary. |
| *(absent)* `--no-strategic` | the conditioning layer stays **ON**; the ablation is a separate arm (plan §0.1). |
| *(absent)* `--require-parity` | B1 is an unregistered corpus key — the flag would **refuse**. Expect the loud NON-PARITY line; that is correct. |

### 2.2 P1-**OUT** adds

```
--agents off
```

### 2.3 P1-**IN** adds

```
--agents head \
--w-agent 1.0 \
--agent-join /workspace/TanitAD/data/joins/b1_train_plus_eval_agents.jsonl.xz \   # ⛔ COMBINED, see 2.4
--agent-queries 100
```

* `head`, **not** `oracle` — `oracle` feeds GT boxes at inference and is
  **INADMISSIBLE as a capability claim** (`refc_v3_train.py:4250`).
* `--w-agent > 0` and `--agent-join` are both **hard preconditions** of
  `--agents head` (`:505`, `:513`); each refusal exists because an arm without
  them reads as *"the agent head does not help"* — a refutation manufactured by
  a missing loss. ⚠️ `1.0` is the codebase **convention, not a validated value**.
* `--agent-queries 100` is the M17 ruling (train max 94 over 12,122,129 boxes).
* `--agent-w-project` / `--agent-w-ground` stay **0.0**: the preflight measured
  `w-ground` computing a **tautology** (2.6e-08, gradient 8.7e-11).

⭐ **P1's data blocker is CLEARED.** `b1train_agents.jsonl.xz`, **317,028,572 B**,
md5 **`1c985e6d6ad34e605c4ebd30cb353558`** — staged to
`/workspace/TanitAD/data/joins/`, **md5 byte-identical on the pod**, decompresses
to valid JSONL (**849,263 rows**, matching the plan's count exactly). It had
existed in exactly **one** place
(`C:\Users\Admin\tanitad-caches\b1-train-join-20260906\`): **not** in the repo
(only its `.meta.json` is committed) and **not** on the pod. Transfer 343 s @
899 KB/s.

### 2.4 ⛔⛔ SECOND P1-IN BLOCKER, FOUND BY READING THE EVAL PATH

**The train join alone makes `--agents head` REFUSE AT STARTUP.**

`refc_v3_train.py:3560` applies the **same** `--agent-join` file to the **eval**
dataset, restricted to the eval episode ids. And the B1 TRAIN join and the
141-episode EVAL grid have **ZERO clip overlap** — MEASURED: 141 eval clips,
4,572 train clips, **intersection = 0**.

⇒ `enable_agent_join` computes `n_stable = 0`, `n_legacy = 0`, and raises at
`refc_v3_train.py:1349`:

> `[v3] REFUSING: the agent join covers ZERO of 141 episodes … That is the wrong
> join for this corpus — a run would stamp the flag and train no detector.`

⇒ **P1-IN with `--eval-every 500` would die before the first step.** This is the
refusal working correctly; it is not a bug.

⭐ **THE FIX, PREPARED.** The **B1 EVAL join already exists** as
`CONTROL_b1eval_recon.jsonl.xz` (10,012,564 B, md5
`3ddb42ecbd3926066795a94587af2aed`, **139/141 clips**, 26,394 rows, built with
`--pose-source reconstruct`; the 2 misses are `no_obstacle`). The trainer takes
**one** `--agent-join`, so the two must be **one file**. `xz` is a multi-stream
format and `JoinFileReader` opens with `lzma.open(p, "rt")`
(`train_p8_occupancy.py:333`), which reads across concatenated streams:

```
cat b1train_agents.jsonl.xz CONTROL_b1eval_recon.jsonl.xz \
  > b1_train_plus_eval_agents.jsonl.xz     # 327,041,136 B = 317,028,572 + 10,012,564
```

⭐ **VERIFIED THROUGH THE TRAINER'S OWN CODE PATH**, not asserted:
`lzma.open(combined, "rt")` reads **875,657** rows = **849,263 + 26,394**,
exactly the sum. Both streams are visible to the reader.

⇒ **`--agent-join` for the P1-IN arm must point at the COMBINED file**, not the
train join. `LAUNCH_refcv5_v2.sh` uses the combined path and gates on the row
count before launching.

---

## 3. ⛔ THE VOCABULARY × ROLE TABLE, with the enforcement site

**The rule.** INPUTS = the **nav command**, measured **`v0` at t0**, **max speed**.
Everything else is a TRAINING SIGNAL. **Inference is vision-only**; labels may
use ego, inference may not.

Paths relative to repo root. Line numbers are worktree (HEAD ±1 where noted).

| # | field | tokens / definition site | **ROLE** | enforcement site | the enforcing line |
|---|---|---|---|---|---|
| 1 | **nav command** | 3 · `stack/tanitad/models/vocab_v7.py:331-333` | **INPUT** *declared*; ⛔ **ORACLE in fact** | decl `vocab_v7.py:552`; gate `stack/tanitad/data/v7_labels.py:292`; forward `stack/tanitad/refs/refc_v3.py:1277` | `**{t: "input" for t in NAV_COMMAND_TOKENS},` / `if not manifest.allow_oracle_nav: raise OracleNavRefused(` |
| 2 | **nav args** `distance_m`,`time_s` | 2 · `vocab_v7.py:334` | **INPUT, both slots** ⛔ split **NOT enforced** | `refc_v3.py:1367-1380`; requires both at `refc_v3_train.py:1200` | `_ok = ("distance_m" in _a) and ("time_s" in _a)` |
| 3 | **`v0` at t0** | `refc_v3.py:232` | ⭐ **INPUT** (admissible) | forward `refc_v3.py:1278`; E11′ `:58-69`; precondition `refc_v3_train.py:234` | `v0 = poses_win[:, -1, 3]   # MEASURED speed` |
| 4a | **tactical vocabulary** | **22** · `vocab_v7.py:95-113` (pinned `test_tac_goal_wiring.py:131`) | ⚠️ **TRAINING SIGNAL** *(since 2026-09-06; was audit-only)* | routing `v7_labels.py:235`; vision-only guard `test_tac_goal_wiring.py:444-461` | `tac_goals=frozenset(g_tac.get("goals") or {}),` |
| 4b | `audit` / `goal_flags` | `v7_labels.py:246` | ⭐ **AUDIT-ONLY** | `v7_labels.py:135` | **`#: audit-only, NEVER a training input (spec §6)`** |
| 5 | **strategic vocabulary** | 8 + 7 = **15** · `vocab_v7.py:62-71`, `:75-83` | **TRAINING SIGNAL** — ⛔ **but see §4: not wired** | head `stack/tanitad/refs/refc_strategic.py:151-185`; loss `:225-265` | `The heads EXTRACT from vision + measured v0. Alpamayo GT SUPERVISES.` |
| 5b | `REDUCE_TO_FOLLOW_ROUTE` | **REMOVED** from the 7 | n/a | `stack/tests/test_vocab_v7_frozen.py:94` | `assert "REDUCE_TO_FOLLOW_ROUTE" not in V.STRATEGIC_ACTION_TOKENS_V7` |
| 6 | **lateral actions** | **8** · `vocab_v7.py:290-293` | **TRAINING SIGNAL** (5 of 8 reachable) | `stack/scripts/s2_geom_emit_v7.py:446,:449,:452` | `:446 lat = "NUDGE_L" if biggest[2] > 0 else "NUDGE_R"` |
| 7a | **`SPEED_BAND`** | `vocab_v7.py:100` | **TRAINING SIGNAL** — ⛔ **degenerate** (present on 4,572/4,572 ⇒ weight 1.0, target 1.0, zero negatives) | emit `s2_geom_emit_v7.py:309-315`; window `:199-211`; band `:51` | `"v_hi_ms": round(float(v.max()), 2),` |
| 7b | `_MEASURED_GEOMETRY_TOKENS` | `v7_labels.py:274` | negative-policy gate | `v7_labels.py:643` | `w.append(1.0 if tok in _MEASURED_GEOMETRY_TOKENS else IGNORE_W)` |
| 7c | **max speed (INPUT)** | 8 steps · `stack/tanitad/refs/max_speed_input.py:142` | ⛔ **UNENFORCED-FINDING — module has ZERO consumers** | declared `max_speed_input.py:10-11`; **no consumer, no CLI flag** | `the PI has ruled that class is an INPUT, not a training signal` |
| 8 | **traffic light** | 4 of the 22 · `vocab_v7.py:108-111` | **TRAINING SIGNAL**, negatives ignored; ⛔ **no quality filter in code**; ⚠️ see the retraction below | `v7_labels.py:591-601`; grounding `stack/tanitad/data/alpamayo_fusion.py:273-277` | `only **998 were asked the traffic-light grounding question at all**` |
| 9 | **agents** | `refc_v3_train.py:4250`; `AGENT_WEIGHT_DEFAULT = 0.0` `:116` | `head` = **TRAINING SIGNAL** (vision-only at inference) · `oracle` = **INADMISSIBLE ceiling** | `stack/tanitad/refs/refc.py:3117-3123` | `# ⛔ THE VISION-ONLY RULE IS ENFORCED BY WHERE THE TENSOR IS READ, not by a comment.` |
| 10 | **geometric goal point** | `stack/tanitad/refs/goal_point.py:99`; weight 0.0 `refc_v3_train.py:129` | ⭐ **DIAGNOSTIC-ONLY** | `goal_point.py:126`; tests `test_rl_forward_keys_cover_signature.py:78-82,:153-167` | `DIAGNOSTIC_ONLY_FORWARD_KWARGS = ("gp_point", "gp_valid")` |
| 11 | **`road_class`** | `stack/scripts/refb_labels.py:1806-1811` | **TRAINING SIGNAL only; forbidden at inference** | `refb_labels.py:1816-1821` | `fine as LABELS …, never as inference inputs (vision-only rule …)` |
| 12 | **`a_star`** | `refc_v3_train.py:1734` | **TRAINING SIGNAL** (target, never an input) | loss `:1736`; metric `:2181` | `loss_cls = F.cross_entropy(out["anchor_logits"], a_star)` |

### Three findings inside this table

⛔ **Finding N — the nav command is an ORACLE on this corpus, not the production
input the rule describes.** All 4,719 v7.2 nav records carry
`provenance: "ego-future"`; `vocab_v7.py:314-327` states only
`provenance="nav-system"` is admissible and `"ego-future"` is *"USABLE FOR
TRAINING ONLY"*. The refusal machinery is real and correct
(`OracleNavRefused`, keyed on provenance because the `oracle` boolean is missing
on 529/4,719 = 11.2 %) — **but `refc_v3_train.py:3461` and `:3544` both call
`load_v7_labels(..., allow_oracle_nav=True)`.** ⇒ every REF-C arm using
`--nav-from-v7`, refcv4b and refcv5-v2 alike, is an **oracle-nav arm**. It is
stamped and visible, which is the correct handling; it is **not** the production
nav command, and no capability claim may describe it as one.

⚠️ **Finding M — `max_speed_input.py` is NOT an input to the model.**
⛔ **RE-MEASURED after the repo advanced mid-session (commit `6ae8acf`,
"max-speed input — pinned quantizer, OFF proof, output-scored constraint"), and
my first reading is now PARTLY STALE — corrected rather than overwritten.**

* **Was true at 22:0xZ:** zero consumers, and the test the module's own docstring
  cites at `:87` did not exist.
* **True now:** the module has **two** consumers —
  `stack/tanitad/eval/speed_limit_scoring.py` and `stack/tests/test_max_speed_input.py`
  (the missing test **landed**).
* **Still true, and it is the load-bearing half:** there is **no
  `--max-speed-input` trainer flag** — HEAD's trainer is unchanged at
  **4,479 lines / 88 `add_argument`**, and `max.speed` matches **0 times** in it.

⇒ **P15 is now an EVAL-SIDE SCORER, not a model INPUT.** The PI named max speed
as one of the three INPUTS; as it stands the model still cannot read it. The
launch leaves the slot and takes no flag, as briefed — but the slot is *not yet
fillable*.

⚠️ **Finding T — the brief's P3 premise is one day stale.** `g_tac.goals` no
longer routes *only* into `audit["goal_flags"]`. Since 2026-09-06 it is promoted
to its own field `tac_goals` (`v7_labels.py:235`) and is **actively supervised**.
The verbatim *"audit-only, NEVER a training input (spec §6)"* at `v7_labels.py:135`
annotates the **`audit` dict**, which remains audit-only — it does **not**
annotate `tac_goals`. `v7_labels.py:124` states the transition in so many words:
*"until 2026-09-06, read into `audit` only and therefore trained by nothing."*
The anti-echo direction is still properly policed: `test_tac_goal_wiring.py:459-461`
asserts the forward signature exposes no `tac_goal*` target channel at all.

---

## 4. ⛔ WHAT DOES NOT ENTER, AND WHY

| piece | verdict | evidence |
|---|---|---|
| **P4** 15-token strategic vocabulary | ⛔ **CANNOT ENTER as a trained head — WIRED TO NOTHING** | `refc_strategic.py` is imported by exactly **one** file, `stack/tests/test_p4_p13_p14_wiring.py:36` — a test. `str_goal_tok_head` = **1 hit**, in its own docstring (`:17`). **Zero** references from `refc_v3.py` or `refc_v3_train.py`; **no CLI flag**. Two independent probes, live control `tac_goal_tok_head` = **107** occurrences. The class is `StrategicTokenHead` (`:151`) and is complete — heads, loss, majority control, off-vocabulary refusal — but nothing calls it. Same failure mode as P6's 4,530,444-param selector. |
| **P13** DD `t`-draw | ⛔ does not enter (mechanism PASS, benefit not established) | ⚠️ **record correction:** refcv5 trained at **t ∈ {10, 0}**, **not t = 8** — a live spy on the real `_sample` showed the realised timesteps are the *inference ladder*, identical under `t_max = 50` and `t_max = 1`. |
| **D-TRISEG** three-segment anchors | ⛔ **CANNOT ENTER — the consumer refuses the shape** | `build_twoseg_anchors.py:55-58`: *"`refc.py::AnchoredDiffusionDecoder.roll_bank` rolls `anchor_controls` as a constant and its shape checks REFUSE a `[N, 3]` or `[N, 4]` bank (loudly, which is correct). `tanitad.refs.anchor_twoseg.roll_bank` is the drop-in reference the decoder needs; **wiring it is a named hand-off**."* Corroborated verbatim at `anchor_twoseg.py:94-101`. The measurement (nine bars PASS, +0.2133 m) stands; the **wiring does not exist**, and no 3- or 4-column bank exists on the pod. |
| **`--tac-goal-tok-head`** | ⛔ **not in HEAD** — and ⛔⛔ **its absence silences the TACTICAL vocabulary too** | 20 hits in the **worktree**, **0** in HEAD, **0** in the index. ⇒ not composed against, per the brief's rule. **But see §4.1 — this is not a nice-to-have.** |

### 4.1 ⛔⛔ WITHOUT `--tac-goal-tok-head`, refcv5-v2 TRAINS **NEITHER** VOCABULARY

`refc_v3.py:948` is the gate:

```python
if _vv != "kin3" and bool(getattr(cfg, "tac_goal_tok_head", False)):
    self.tac_goal_tok_head = TacGoalTokenHead(...)
```

Without the flag, `self.tac_goal_tok_head = None`, so `tac_goal_logits` is never
written (`:1236-1237`) and the **22 tactical tokens are supervised by nothing**.
The **15 strategic tokens** have no head in the model at all (§4, P4).

⇒ **As HEAD stands today, the composed arm satisfies the PI's *"use the whole
tactical and strategic vocabulary"* on NEITHER half.** The `--v7-labels`
vocabulary would be loaded and joined — and then trained against by no head.

**What it takes to honour the instruction:**

| half | what is needed | distance |
|---|---|---|
| **tactical (22)** | the sibling lands `--tac-goal-tok-head`; the arm passes it (it **requires `--v7-labels`**, which this arm already passes) | **one commit** — the code is written and guarded in the worktree |
| **strategic (15)** | a call site for `refc_strategic.StrategicTokenHead` in `refc_v3.py`, a loss hook in `refc_v3_train.py`, and a `--str-goal-tok-head` flag | **not written** — the module is complete and referenced by nothing but a test |

⚠️ And even fully wired, the honest ceiling holds: **88.13 %** of the usable
horizon carries no v7 tactical GT, and only **6 of 15** strategic tokens are
populated (**11.43 %** of the horizon supervisable). ⛔ `SPEED_BAND` is
supervised **degenerately** — present on 4,572/4,572 and inside
`_MEASURED_GEOMETRY_TOKENS`, so every cell gets weight 1.0 and target 1.0: a
constant-1 BCE target with **zero negatives**. It cannot teach; it can only
saturate a logit while contributing loss.
| **P2** geometric goal point | ⛔ **mutually exclusive with the nav-command INPUT** | `--goal-point-inject` sets `nav_inject → False` **by pre-registration** (`refc_v3_train.py:4513`), and `--nav-args` refuses when `nav_inject` is off (`:381-384`). The PI names the nav command as an INPUT ⇒ goal-point stays OFF. |
| **P11** `--nav-args` | ⛔ **feeds an INADMISSIBLE channel** | `NAV_ARG_DIMS = 3` = `(distance_norm, time_norm, args_valid)` (`refc_v3.py:167`); batch built at `refc_v3_train.py:1543`; passed to forward at `:1715`. Plan §8 P11 rules **`time_s` NOT admissible** — it is the ego's own future speed profile inverted. **There is no distance-only mode**: `refc_v3_train.py:1200` requires *both* slots, and `refc_v3.py:837` projects both through one `Linear(NAV_ARG_DIMS, d_nav)`. ⇒ enabling it would put a future-ego longitudinal oracle into a model whose gap is **88.7 % longitudinal** — it would manufacture exactly the win we are trying to measure. |
| **P15** max speed | slot left, no flag | see Finding M. A sibling owns the channel. |

---

## 5. THE PRE-REGISTERED COMPARISON PROTOCOL vs refcv4b

⛔ **Written before the run, as required. Both outcomes committed.**

### 5.1 Baseline

`C:\Users\Admin\refcv4b_final\ckpt_40284_FINAL.pt` — md5
**`99b573e8277d94a5e3bfbf630cb4d751`** ✅ *verified this session*,
`n_anchors 117`, `--anchor-v0-conditioned`.
⚠️ **Trap:** the same directory holds `ckpt_40284.pt` (428,616,885 B, md5
`b99c66a13ce42059d4201be14f965fab`) — **same step, different bytes**. Name the
`_FINAL` file and its md5 explicitly, never "the 40,284 checkpoint".

### 5.2 Grid

The **141-episode / 4,823-window B1 v7.2 EVAL grid** the landing used. Both arms
on the **same windows** — pairing is what makes the interval valid and strictly
more powerful than combining two intervals in quadrature.

### 5.3 Estimator

⛔ **`taniteval/taniteval/ci.py:275` `paired_episode_cluster_bootstrap`.**
⛔ **NEVER `overlapping_holdout_se`** (`ci.py:121`) — it biases the point estimate
bidirectionally, up to a sign flip. **T1** stamp on every roll.

### 5.4 ⛔ Four metric families, never pooled, never ADE alone

LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC.
**LATERAL is read on curvature MAE with the straight-line floor beside it** —
an arm can win ADE while tracking the road worse than a plan that never steers.

### 5.5 ⛔⛔ THE PRE-COMMITTED TRIVIAL-CONTROL BAR

`stack/tanitad/eval/echo_gate.py` — `REQUIRED_REFERENCES = ("ha", "ha0_ext")`
(`:85`); the panel **refuses** to score without both.

On the landing grid: **`ha` 0.2996** · **`ha0` 0.6723** · **`ha0_ext` 0.2874**;
refcv4b **`os` 0.2975**.

|   | value | verdict |
|---|---:|---|
| `os − ha` | **−0.0021** | not separated — **TIE** |
| `os − ha0_ext` | **+0.0101** | not separated — **TIE, wrong direction** |

⇒ **refcv4b only TIES the do-nothing baselines.**

> ### ⭐ SUCCESS CRITERION, COMMITTED NOW
> **refcv5-v2 must BEAT `ha0_ext`, SEPARATED, on the paired episode-cluster
> bootstrap. If it does not, it has not learned to drive either — and that is
> the reported result, regardless of any ADE improvement over refcv4b.**
>
> ⛔ Beating `ha` while **tying** `ha0_ext` means **the gain is echo**, and the
> gate reads it that way (`echo_gate.py:159-163`). `ha` is the *weaker* form
> (finite-difference accel, steer read at `t0-1`); `ha0_ext` is the sharpened
> form built on the corpus's own `ax`. `ha` **is** `ha0_ext` in construction —
> constant longitudinal acceleration and constant curvature from the measured
> t0 state.
> ⛔ `echo_gate` requires **both** CI separation **and** a pre-registered
> relative margin (`margins={"ha0_ext": …}`); either alone is not the criterion.
> ⛔ `assert_not_echoing` must be run with **`gate2b` supplied** — omitting it
> yields a verdict explicitly marked `STRUCTURAL_ONLY`, which is **not** an
> anti-echo pass. MEASURED 2026-09-03: the deliberate-regression arm **passed**
> `gate2`'s scene direction, because a live encoder moves the output for any
> model, including one that ignores the scene.

### 5.6 Contaminated numbers — do not carry them forward

* ⛔ **All pre-`a_star`-fix `anchor_acc` / `sel_agrees_oracle` are CONTAMINATED**
  — `a_star` is consumed on **every roll**, not only `--with-oracle-sel` rolls.
  Corrected full-grid `anchor_acc` = **0.5275**, *not* 0.0993.
* Chance: refcv3 **1/128 = 0.007812**; refcv4b **1/117 = 0.008547**. ⛔ A stale
  `1/128` string sits in several sidecars — **do not copy it**.
* ⛔ `--sel-refined`'s −0.0259 m result is **n = 171 windows / 20 episodes on
  `ckpt_30000`**, a *different surface* from the landing grid (its model-free
  arms read `ha` 0.2860 / `ha0` 0.6542 / `ha0_ext` 0.2769). Admissible as a
  **direction**, never comparable in magnitude to a landing number.
* Quotable: refcv4b's **A40 landing roll** `os` 0.2975 / `os_navzero` 0.3928
  (`C3_os_reproduction` PASSES at abs_diff 2.6e-05). The **Thor re-roll**
  (0.2965 / 0.3926) misses by 0.001031 from cross-hardware argmax tie-breaks.
  **A number carries its ROLL.**

### 5.7 ⚠️ Replication

A separated CI from **one seed is necessary, not sufficient** — measured
replicate false-positive rate **6/42 = 14.3 %** on a *zero-lever* replicate.
⇒ a refcv5-v2 PASS is **entry to the register, not a published result**.

### 5.8 ⛔ Vacuity gate

Report the **manoeuvre rate beside every constraint-satisfaction number**. One
zero-violation result was MEASURED to have been bought by a `turn_left` recall
of exactly **0.0000**.

---

## 6. ⭐ THE LAUNCH COMMAND IS VERIFIED BY EXECUTION, NOT BY READING

⛔ *"An AST census once read 0 suspects on BOTH the fixed and the broken
trainer."* Every claim below was produced by **running** HEAD's trainer on the
dev box (CPU, `--smoke`, zero GPU, zero pod contact).

### 6.1 Flag existence — 38/38, with three negative controls

All 38 flags of the composed command are present in HEAD's `--help`
(536 lines). **Controls that must read absent, and did:**
`--max-speed-input` **absent** (P15 unwired) · `--tac-goal-tok-head` **absent**
from HEAD (worktree-only) · no P4 strategic-token flag.

### 6.2 ⭐ The P14 paired guard — MUTATION-PROVEN IN BOTH DIRECTIONS

| arm | flags | result |
|---|---|---|
| **A** | `--sel-refined` alone | ⛔ **REFUSED** — *"ranks the fan by a score one pass STALE … 0.0259 m separated WORSE, 29.82 % of picks flipped"* |
| **B** | `--sel-score-emitted` alone | ⛔ **REFUSED** — *"is INERT … computed and then DISCARDED … while `sampler_ranks_the_fan` still stamps False"* |
| **C** *(control)* | **both** + `--sampler ddim` | ⭐ **ACCEPTED**, and auto-corrects `sel_score_emitted_t -1 → 0`, stamped `auto-zero-on-ddim` |

⇒ the guard **discriminates**; it is not a blanket refusal.

### 6.3 ⭐ The P1 agent guards — MUTATION-PROVEN

| arm | flags | result |
|---|---|---|
| **D** | `--agents head` (w_agent 0) | ⛔ **REFUSED** — *"a REFUTATION manufactured by a missing loss"* |
| **E** | `--agents head --w-agent 1.0`, no join | ⛔ **REFUSED by the effective-weight audit** — *"stamped and the loss term is skipped"* |
| **F** *(control)* | `--agents off` | ⭐ **ACCEPTED**, `--w-agent` reported `OFF_BY_DEFAULT` |

### 6.4 ⭐⭐ End-to-end: the composed P1-OUT command RUNS, and stamps the fact

The full flag body trained 2 smoke steps and wrote `config.json`:

```
sel_refined                = True
sel_score_emitted          = True
sel_score_emitted_t        = 0
sel_score_emitted_t_source = 'auto-zero-on-ddim'
sampler_ranks_the_fan      = True      <-- refcv5-v1 shipped this FALSE
```

Anchors resolved to `units=alat (source: file+cli), schedule=constant`,
`straight_ahead_control_present: True`, sha256 `51f930dc6f3564ff…`.
The effective-weight table read **1/5 terms build a graph** (`--w-u0 0.5` TRAINS;
the four agent/goal-point terms `OFF_BY_DEFAULT` with their missing preconditions
named) — which is correct for the P1-OUT arm.

⚠️ The smoke ran `tac_vocab kin3` because no `--v7-labels` file exists on the dev
box; the real launch passes it and must print `v7.0`. **Check that line.**

---

## 7. THE PI'S VIDEO FEEDBACK — mapped, with an honest "will not move" column

⛔⛔ **PROVENANCE FINDING: the 12 observations are NOT IN THE REPO.** They exist
only in the Claude session transcript
(`C:\Users\Admin\.claude\projects\G--Meine-…-TanitAD\57b6753a-…jsonl:131387`,
`ts 2026-09-06T15:22:05Z`, plus two forked copies). A scan of **10,076**
working-tree files and **11,420** tracked files found **no repo file carrying the
list verbatim** — the repo refers to them only *by number*. ⇒ **The PI's own
words are stranded outside git**, which is precisely the failure the
`AGENT_OPERATING_STANDARD.md` exists to prevent. **Recommend banking them.**

⚠️ **And the numbering is only half-anchored:** just **6 of 12** carry an
explicit numbered back-reference in the repo — (1), (2), (6), (8), (10), (12).
The rest are topic-matched. Marked below.

| # | the PI's observation (abridged; typos are his) | what addresses it | ⭐ **will refcv5-v2 move it?** |
|---|---|---|---|
| **1** ✅ | *"missing … the contraints of startegic and tacticals goals and actions … estimate the right contsriants like position and time"* | `C-ENV-1`, `stack/tanitad/eval/constraints.py` | ⛔ **NO.** Eval-side scoring only. P4's strategic head is **wired to nothing** (§4), and the tactical vocabulary is supervisable on **11.43 %** of the horizon. |
| **2** ✅ | *"The selector is not choosing the right trajectory … better ones are included"* | ⭐ **P14** | ⭐ **PARTIALLY — the one genuine win.** Ceiling **+0.2813 m [+0.2127, +0.3543]**, a >2×-better path in the fan on 41.09 % of windows. ⚠️ **98.9 % ALONG-TRACK; curvature NOT separated.** The road-mark half of his remark is **NOT MEASURABLE**, and the measured defect is **longitudinal** (`a_lon` oracle recovers 74.2 %). **This is not a steering fix and must not be sold as one.** |
| **3** ⚠️ | *"you wrote the mode is not reading the nav token at inbference … Is it true?"* | direct question to us | ⭐ **ANSWERED — and the overlay was wrong.** The nav token **IS** consumed, at train **and** eval (`refc_v3.py:1277`; `--nav-from-v7`). ⛔ **But the honest answer has a second half:** its provenance is `ego-future` on **4,719/4,719** records, so what the model reads is an **ORACLE**, not the production nav command (§3, Finding N). |
| **4** ⚠️ | *"the model is not … following the nav commands as stated in the roadabout example"* | P10 `g_str` sign | ⛔ **NO.** The repo's own verdict is **"the PI is right"** and *"not a cherry-pick; it is the typical case"*. The repaired steering signal turns left and **the plan does not follow it** — converting it needs a ~40k retrain, a **PI spend decision**. |
| **5** ⚠️ | *"Why does the gt trajectory in the roaubout expample is not smooth … the gradiant is not continious"* | — | ⛔ **NO — and there is NO refcv4b-scoped answer anywhere.** The only prior art is **refcv3**-scoped `D-REFCV3-SMOOTH4` (part rendering artefact ×7.44, part real defect). ⇒ **an open GAP.** |
| **6** ✅ | *"The trajectory selection is jumoping between consecutive frames"* | `D-SELQ-STAB-9`; deadband **REFUTED** | ⛔ **NO.** Nothing in this composition targets temporal stability of the pick. |
| **7** ⚠️ | *"the spped of refcv4b is overshooting and very wrong, this must be fixed"* | ⛔ **no repo stream claims (7)** | ⛔ **NO.** ⚠️ **Correcting the brief:** the repo binds **P15 to instruction (10), not (7)**. The nearest instrument is `frac_over` / `frac_under_when_allowed` in `constraints.py` — **eval-side scoring, not a training lever.** |
| **8** ✅ | *"not learning to avoid collisions or keeping enough dspace to the infrastructuire"* | **P1** (agents) + P7 (unported) | ⚠️ **P1-IN ONLY, and only half.** Agent conditioning is the environment link and computes `v_rel_x`. ⛔ **Infrastructure clearance is NOT MEASURABLE AT ALL** (`D-CLEARANCE-IS-AGENT-NOT-INFRA-1`): the 11th obstacle class is `train_or_tram_car`, and `protruding_object` is overhead (bottom face +1.68 m) while the join **drops `z`**. P7's distance-keeping cost is still unported. **P1-OUT moves this by nothing.** |
| **9** ⚠️ | *"the output trahjectory are not smooth, I thought we are forcing this by conctrsuction"* | P5 (already landed) | ⛔ **NO NEW MOVEMENT.** P5 landed **before** this arm: curvature **−26 %** on a held-out split, ADE not separated. ⚠️ *"halves the ADE"* was an overstatement propagated into four artifacts — it is **−30.7 %**. ⛔ And REF-C remains **~84× worse on curvature than a plan that never steers** (0.02737 vs 2.30973). **Smoothness is not addressed by refcv5-v2.** |
| **10** ✅ | *"max speed … the model muist learn to adapt it speed … and reach the max speed when the situation allows"* | **P15** | ⛔ **NO — the channel is DEAD CODE.** `max_speed_input.py` has **zero consumers**, no CLI flag, and the test its own docstring cites **does not exist** (§3, Finding M). The slot is left and the flag is absent, as briefed. |
| **11** ⚠️ | *"The model is not reactiong to traffic light by braking"* | `D-TLIGHT-1` | ⛔ **NO.** The label exists and **reaches nothing**. ⛔ **RETRACTED FIGURES — do not carry "175 grounded / 604 disputed" forward.** `D-TLGROUND-DISPUTED` (commit `47a2302`, `RETRACTION_LOG.md`) re-measured on n = 4,719 clips / 805 emissions: **182 GROUNDED (22.6 %) · 601 NOT_CHECKED (74.7 %) · 22 NO_QUESTION (2.7 %) · 0 CONTRADICTED.** The counts barely moved (175→182, 604→601) **and the verdict inverted**: "disputed" claimed the grounding looked and contradicted, and **nothing contradicted anything** — only one grounding question is asked per clip and on those 601 it asked about something else. Slot left; dependency flagged. |
| **12** ✅ | *"compltelty ignroing or not activating lane changes, checvk this if there is a systematic error"* | root cause **FOUND** | ⛔ **NO.** The emitter reaches **5 of 8** lateral actions; `LANE_CHANGE_L`/`LANE_CHANGE_R`/`ABORT_LC` are structurally unreachable (`s2_geom_emit_v7.py:446/:449/:452`, verified). Text-side supply ceiling **105 clips ≈ 2.2 %** — a **CORPUS** fact ⇒ repairing the emitter makes the class **expressible, not supplied**. |

### ⭐ The honest total

**Of 12 observations, refcv5-v2 as composed moves ONE (#2, and only its
longitudinal half), answers ONE (#3, with a correction the PI will want), and
half-moves ONE (#8, P1-IN only). NINE will not move.**

⇒ **This arm is a selection-ranking and environment-conditioning arm. It is not
a smoothness arm, not a lane-change arm, not a traffic-light arm, and not a
speed-limit arm.** Reporting it as *"addresses the video feedback"* would be
false.

---

## 8. DELIVERABLE MANIFEST

| artifact | where it lives | only one place? |
|---|---|---|
| this document | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-07-refcv5-v2-compose/README.md` | no |
| `ship_and_gate.sh` | same dir | no |
| `LAUNCH_refcv5_v2.sh` | same dir | no |
| `raw/pod_vs_head_flags.json` | same dir | no |
| **`b1train_agents.jsonl.xz`** 317,028,572 B md5 `1c985e6d…3558`, 849,263 rows | `devbox:C:\Users\Admin\tanitad-caches\b1-train-join-20260906\` **+** `tanitad-a40:/workspace/TanitAD/data/joins/` | ⚠️ **NOT IN GIT** — only its `.meta.json` is committed. Two disks now, neither of them the repo. |
| **`CONTROL_b1eval_recon.jsonl.xz`** 10,012,564 B md5 `3ddb42ec…2aed`, 26,394 rows, 139/141 clips | `devbox:C:\Users\Admin\tanitad-caches\b1-train-join-20260906\` | ⛔ **ONE PLACE ONLY, and NOT IN GIT.** |
| **`b1_train_plus_eval_agents.jsonl.xz`** 327,041,136 B, 875,657 rows — *the file `--agents head` must use* | `devbox:scratchpad` **+** `tanitad-a40:/workspace/TanitAD/data/joins/` | ⚠️ **NOT IN GIT** (derived: `cat train eval`, reproducible in one command). |

### ⛔ ESCALATION — integration needed, and it will otherwise sit unread

1. **Ship `stack/` to `tanitad-a40`** once the sibling's trainer edits land.
   Until then **no composed arm can carry a single one of its levers.**
   `ship_and_gate.sh` does it and verifies by content with a live control.
2. **Bank the PI's 12 video observations into the repo.** They exist only in a
   session transcript (§7). The programme has been answering them by number for
   a day against a source that is not under version control.
3. **Bank the three agent-join artifacts** (or record them in
   `ASSET_INVENTORY.md` with their md5s and build commands). Two of the three
   live on exactly one disk each.
4. **P4 needs wiring**, not building. `refc_strategic.py` is complete and
   unreferenced; a `--str-goal-tok-head` flag plus a call site would let the
   15-token vocabulary actually train. Until then the PI's *"use the whole
   tactical and strategic vocab"* is met on the tactical half only.
5. **D-TRISEG needs its named hand-off closed** — substitute
   `anchor_twoseg.roll_bank` into `AnchoredDiffusionDecoder.roll_bank`. The
   measurement passed nine bars; the decoder still refuses the shape.

### Verification status

| check | result |
|---|---|
| pod trainer flag census | ⭐ **MEASURED** (md5 + flag diff + control) |
| 38/38 launch flags in HEAD `--help` | ⭐ **PASS**, 3 negative controls correct |
| P14 paired guard, both directions + control | ⭐ **PASS** (mutation, not inspection) |
| P1 agent guards, both + control | ⭐ **PASS** (mutation) |
| composed P1-OUT end-to-end smoke | ⭐ **PASS**, `sampler_ranks_the_fan: True` |
| combined join readable by `lzma.open` | ⭐ **PASS on the dev box AND ON THE POD**, 875,657 rows both sides |
| both joins md5 on pod | ⭐ **PASS**, byte-identical (`1c985e6d…3558` / `0c31a3a6…ef46`) |
| A40 GPU untouched throughout | ⭐ **0 MiB, 0 %** at start, middle and end |
| `pod_currency_audit.py --host tanitad-a40` | ⭐ **COMPLETED, AND IT FAILED — INDEPENDENT CORROBORATION.** Verdict: *"FAIL: 26 file(s) in ['HISTORY-UNKNOWN', 'POD-BEHIND', 'POD-DIVERGED'] — the box is NOT running the repo."* ⚠️ **But 26 is an UPPER BOUND on drift, not a count of it:** at least 6 of the 26 are `HISTORY-UNKNOWN` from `git log` returning **rc 3221225478 = 0xC0000006 (STATUS_IN_PAGE_ERROR)** — the G: mount paging out mid-probe, not a stale file. ⇒ the audit **confirms the direction** and the direct md5 + flag census of `refc_v3_train.py` (§1) remains the **sharp** evidence. ⛔ Re-run it as part of `ship_and_gate.sh`, and **do not pipe it through `tail`** — doing so here discarded the head of the report, where the POD-BEHIND rows live. |
| `refcv5_preflight.py` on the pod | ⛔ **NOT RUN** — it must run *after* the ship step, against the shipped code. |
