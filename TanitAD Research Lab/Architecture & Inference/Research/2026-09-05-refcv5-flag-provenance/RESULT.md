# The flags now REFUSE, the record now states what it trained, and 100 queries cost **+8.2 %**

**Stream:** Architecture & Inference · **2026-09-05** · branch `agent/arch-inf-20260803`
**Implements:** `Project Steering/Decisions/2026-09-05-mm-decisions.md` **§M17** and **§M18**
**Evidence class:** MEASURED (ours), artifact path per claim; `raw/` holds every JSON quoted here.

---

## The headline, in four lines

| | |
|---|---|
| **P1** | `_rig_camera` is **WIRED**, not removed. `--agent-rig-camera {off,nominal,extrinsics}` builds a real `RigCamera`; a non-zero `--agent-w-project`/`--agent-w-ground` with **no** camera now **REFUSES at config-pin time**, before a model and before the GPU. **No third state**: a camera under `--agents off` refuses too. |
| **P2** | `w_agent`/`w_u0` were **already** in `config.json` (inside `seams`) — escalation #3 is **partly refuted, MEASURED on its own author's banked artifact**. The real gap was structural, and is now closed: the stamp is **derived from argparse**, and `train()` **refuses to start** if any `--agent-*`/`--w-*` knob does not reach the record. |
| **P3** | `--agent-queries` default **32 → 100**, help text corrected. Cost MEASURED through the shipped path at the corpus geometry: step time **+8.2 %** (above a 3.8 % replicate noise floor), parameters **+17,408 = +0.077 %**, peak memory **no detectable difference**. ⇒ **the cost does not bind.** |
| **P4** | `tanitad/data/join_meta.py` (NEW): builders **declare** the digest's artifact scope, consumers **REFUSE** an undeclared one, and `backfill` **MEASURES** the scope of a legacy digest. Demonstrated end to end on the **real 136,689,648-byte train join**. |

`pytest -q` green on the touched surface; **38 new tests**, every gate shown to FAIL its defect.

---

## P1 — the flag that parsed, stamped, and did nothing

### The defect, restated at source

`refc_v3_train.py:1720` read `model._rig_camera = None` and the attribute was **never assigned**;
`refc_agents.agent_losses` guards both monocular terms on `cam is not None`; and
`AgentSeamConfig.as_dict()` writes `w_project` / `w_ground` into `config.json` regardless. So a run
could pass `--agent-w-project 0.2`, have it recorded, and **compute nothing**. That is worse than a
missing flag: it makes every later comparison between arms unfalsifiable, because the record lies
about the arms.

### The answer to the second question: **WIRED, not removed**

The two terms are not decoration. `monocular_projection_loss` and `ground_range_prior` exist,
are unit-tested with known-value controls, and answer the one thing a monocular head structurally
cannot see — range — in the space where its evidence lives. Removing the flags would delete real
capability to tidy a defect. ⇒ **the camera is built.**

```
--agent-rig-camera {off,nominal,extrinsics}     # NEW, default off
--agent-rig-extrinsics <sensor_extrinsics.json> # NEW, required by `extrinsics`
--agent-cam-height <m>                          # NEW, `nominal` mount height
```

`_build_rig_camera(cfg, args)` returns `(cam, stamp)` and is called at **pin time** (inside
`_pin_refcv5_seams`, hence inside `_pin_trainer_cfg`), at model setup, and at stamp time — so the
earliest of them is a startup refusal. `model._rig_camera` is assigned from it.

**Every state is either computed or refused. There is no third state:**

| configuration | behaviour |
|---|---|
| `w_project > 0` or `w_ground > 0`, `--agent-rig-camera off` | ⛔ **REFUSES** at pin time, naming `config.json` and the M18 incident |
| `--agent-rig-camera nominal\|extrinsics`, `--agents off` | ⛔ **REFUSES** — the camera is consumed only by `agent_losses`, so this would be the same defect mirrored |
| **any** agent weight > 0 with `--agents off` (`--w-agent`, `--agent-w-project`, `--agent-w-ground`) | ⛔ **REFUSES** — see the fourth direction below |
| `--agent-rig-camera extrinsics` with no `--agent-rig-extrinsics` | ⛔ **REFUSES** |
| an extrinsics JSON with no quaternion | ⛔ **REFUSES** |
| a geometry with no **declared** `CanonicalFrame` (e.g. the 64×64 tiny rig) | ⛔ **REFUSES** rather than inventing an `f_ref` |
| both weights 0, camera off | unchanged — bit-identical to every arm banked so far |

### ⛔ Why the geometry refusal is not pedantry

A `CanonicalFrame` is **not its pixel count**: it carries `f_ref` **and the projection**, and this
corpus is **cylindrical** (the column is linear in azimuth). Inventing an `f_ref`, or reaching for
the pinhole formula, reads **92.6°** for a **120°** camera and looks entirely plausible — the trap
`CLAUDE.md` already catalogues. Declared geometries are `(256, 640)`, `(176, 624)`, `(128, 576)`,
all at `f_ref = 305.5774907364391`, `projection = "cylindrical"`; anything else is refused with a
message naming the fix (declare the frame in `tanitad.data.calib`).

⚠️ **Consequence, stated rather than hidden: the monocular terms cannot run on the 64×64 tiny
rig.** That is the correct outcome — a 64 px grey crop has no honest `f_ref` — and it is why the
"the terms actually compute" test drives the loss directly at 256×640 instead of through the tiny
rig.

### ⛔⛔ A FOURTH DEAD-FLAG DIRECTION, FOUND WHILE FIXING THE FIRST — and one of them was SILENT in a way the M18 audit did not name

M18 named `--agent-w-project` / `--agent-w-ground`. **The same defect exists one
flag over, on `--w-agent` itself.** MEASURED at source: with `--agents off` no
`AgentSeamConfig` is built, so the model emits no `agent_slots`; and
`compute_losses_v3`'s existing guard fires only when `agent_box` is **ABSENT**
from the batch. So

```
--agents off --w-agent 1.0 --agent-join joins/train2400_agents.jsonl.xz
```

passes that guard — **the join does put `agent_box` in the batch** — then falls
through the `"agent_slots" in out` condition and computes **nothing**, while
`w_agent: 1.0` is stamped into `config.json`. That is a run reading *"the agent
head does not help"* manufactured by a seam that was never built: the exact
shape of the refusal `_pin_refcv5_seams` already carries for `--agents head`
with `--w-agent 0`, in the opposite direction.

⇒ `--agents off` now refuses **every** non-zero agent weight and the camera with
it, so there is no third state in this direction either. ⚠️ The converse control
is asserted too: `--agents off` with **default** weights — the arm every banked
run used — is untouched.

### ⚠️ The mount pose carries its provenance (the `anchor_meta` pattern)

`nominal` is stamped `mount_pose: "NOMINAL-no-pitch"` and warns, loudly and once, that
**`ground_range_prior` back-projects through the road plane, so any real mount pitch biases its
range DIRECTLY**, while `monocular_projection_loss` is a *difference* of two projections and
cancels a common mount error to first order. `extrinsics` stamps the quaternion, `z`, and the
derived `optical_axis_pitch_rad`. A reader of `config.json` can therefore tell a corpus-calibrated
projection arm from a pitch-free one — which is exactly what the anchor-units incident taught.

⚠️ **Known limitation, escalated not buried:** `agent_losses(cam=…)` takes **one** camera per call,
so a run declares **one** mount pose for the whole batch. The corpus has **two rigs** (cy ≈ 543 /
cy ≈ 755) and per-clip extrinsics. Per-clip cameras need a signature change in
`refc_agents.agent_losses`, which this stream does not own. **Named as an integration item below.**

### The controls (`stack/tests/test_refc_v3_agent_provenance.py`)

* the **deliberate regression**: the exact config that used to run silently (`--agent-w-project 0.2`
  with no camera) **must raise** — parametrised over both weights;
* the **converse control**: zero weights with no camera must **not** raise, so the refusal fires on
  the lying config only and every banked arm is untouched;
* the **positive assertion**: with the camera the trainer builds, `loss_project` and `loss_ground`
  are present, finite, computed over **n > 0**, and **add to `total`**; without it they are absent;
* the **known-value control** on top of it: a *perfect* prediction makes the image-plane term read
  **exactly 0.0 over n > 0**, so the non-zero above is an error signal and not an instrument offset
  (⚠️ the first draft of this test used a random untrained head and read `loss_project = 0.0` over
  **n = 0** — a legitimate no-information value that would have "passed" as evidence the term ran);
* a **structural regression pin**, asserted on the parsed AST: `model._rig_camera` may never again
  be assigned a constant `None`;
* the `extrinsics` pitch test carries **`pitch = 0` as its own control** — an unpitched extrinsic
  must reproduce `RigCamera.nominal` bit-for-bit, so the 0.05 rad reading is the pitch.

---

## P2 — ⚠️ the escalation is partly REFUTED, and the durable fix is bigger than the one it asked for

**MEASURED, on the banked artifact of the very run that raised the escalation**
(`…/2026-09-05-agent-join-into-batch/raw/smoke_agt_head_600.config.json`):

```
config.json["seams"]["w_agent"] = 1.0      config.json["seams"]["w_u0"] = 0.5
```

Both were present, at `3fd2291`, **before** the escalation was written. `w_agent` and `w_u0` are
**not** absent from `config.json`; they are absent from its **top level**, and they were read for
there. ⇒ M18's third bullet ("absent from `config.json` entirely") is **corrected**, and the
one-line fix it proposed was already in the tree.

⭐ **But the escalation was pointing at something real, and the real thing is structural.** The
stamp was a **hand-written dict**, so the property "every knob reaches the record" held only for as
long as someone remembered to add each new line. That is the defect deferred, not removed — it is
correct the day it is written and wrong the day a knob is added. So:

1. **`agent_knob_dests(parser)`** reads every `--agent-*` / `--w-*` option's `dest` **off argparse**
   (17 today: `agents`, `agent_cam_height`, `agent_join`, `agent_join_allow_legacy_ids`,
   `agent_join_no_rates`, `agent_join_verify`, `agent_miss_rate`, `agent_pad`,
   `agent_presence_hard`, `agent_queries`, `agent_rig_camera`, `agent_rig_extrinsics`,
   `agent_sigma_range`, `agent_w_ground`, `agent_w_project`, `w_agent`, `w_u0`).
2. **`agent_knob_stamp(args)`** writes them all into `config.json["seams"]["agent_knobs"]`. A knob
   added to `build_parser` tomorrow is stamped tomorrow, with **no list here to rot**.
3. **`assert_knobs_stamped(args, stamp)`** runs in `train()` **immediately before `config.json` is
   written**: a run whose record cannot state its own weights **refuses to start**.

**The test is derived and asserts BY VALUE, not by key name.** For every knob argparse declares, it
sets a distinctive non-default value (a knob with a domain, like a mount height, takes the first
admissible candidate — a *rule*, not a per-knob table), pins the config through the shipped
`_pin_trainer_cfg`, and asserts that value is reachable among the stamp's leaves. It therefore
survives any renaming of the stamp's keys, and it **cannot be satisfied by a rotted list**. Its
same-breath control asserts that at least 12 knobs were actually probed, because a loop that
checked nothing passes vacuously.

Deliberate regressions, both green: dropping one knob from the stamp must raise; a stamp with no
`agent_knobs` block must raise. And an AST assertion pins the link this file's other tests depend
on — `config.json["seams"]` **is** the checked object, and the check happens **before** the write.

### ⭐ Confirmed on a REAL `config.json`, not only in the unit tests

A 2-step `--smoke --agents oracle` run was driven to `summary.json` and its own
`config.json` read back (`raw/e2e_run_config.json`):

```
seams.agent_rig_camera = {"source": "off", "reason": "not requested",
                          "w_project": 0.0, "w_ground": 0.0}
seams.agent_knobs      = all 17 knobs, incl. w_agent 0.0, w_u0 0.0,
                         agent_queries 100, agent_rig_camera "off"
seams.agents.queries   = 100          agent_join_digest = null (no join)
```

⚠️ Stated because a unit test asserting on `_seam_stamp` is an assertion about
a function; this is the assertion about the **file a reader will actually
open**.

---

## P3 — M17 implemented, and the cost MEASURED rather than assumed

`AGENT_QUERIES_DEFAULT = 100`, used by both the argparse default and the `getattr` fallback in
`_pin_refcv5_seams`, so the trainer has one source. The help text no longer asserts the val40
claim; it now carries the train numbers (max 94, 41,362 boxes on 3,250 frames, nearest sacrificed
target **13.1 m**, `match_slots` keeps the **nearest** N) — pinned by a test that asserts the string
`"32 drops ZERO targets"` is **gone**.

### The measurement

Through the **shipped path** — `build_parser` → `_pin_trainer_cfg` → `RefCV3Model` →
`compute_losses_v3` — at the **corpus geometry** `256×640` (grid 8×20 = **160 memory tokens**),
`--size tiny --arm hier --agents head --w-agent 1.0`, agent seam `d_model 256 · depth 3 · heads 8`,
`n_pad = 94` targets/window (the train corpus max, so Hungarian matching is priced at its worst
case). `OMP_NUM_THREADS=6`, torch 2.11.0+cu128.

⚠️ **SCOPE, stated first because a number quoted outside it is this programme's most expensive
mistake.** These are **dev-box CPU** timings. The RTX 4060 was **busy with another stream
throughout** (5.0–5.5 GB of 8.2 GB, 100 % util), so **no GPU load was added**. What transfers is the
**ratio** and the parameter/activation accounting; the **absolute step time does not**, and neither
does it stand in for a pod A40/A6000. `tanitad-refcv3` and Thor were **not touched**.

**`raw/query_cost_b4_replicate_cpu.json`** — batch 4, 10 steps, **each arm run twice**, because a
single timing pair cannot tell a lever from the rig's own run-to-run noise:

| arm | median step (s) | replicate |
|---|---|---|
| N = 32 | **3.4096** | 3.2829 |
| N = 100 | **3.5878** | 3.6567 |

* replicate spread: **3.8 %** (N = 32), **1.9 %** (N = 100)
* the two arms' medians **do not overlap** — max(N=32) **3.4096** < min(N=100) **3.5878**
* pooled: **3.346 → 3.622 s = +8.2 %**

**`raw/query_cost_b2_cpu.json`** — batch 2, 6 steps: 1.7522 → 1.8477 s = **+5.45 %** (single run,
quoted as corroboration only).

**Parameters — exact, and it is an identity, not an estimate:**

```
22,740,396 (N=32)  ->  22,757,804 (N=100)   delta +17,408 = 68 queries x 256 d_model
                                            = +0.077 % of the arm
```

**Peak memory: NO DETECTABLE DIFFERENCE.** Peak process RSS delta over the step was
`{492.7, 533.9} MB` at N = 32 and `{516.1, 487.3} MB` at N = 100 — the ranges **fully overlap**, so
the measurement reads a null. That null is explained, not merely reported: the activation tensors
that scale with N, priced analytically (fp32, batch 4, depth 3, heads 8, `d = 256`, `n_tok = 160`),
are

| tensor | N = 32 | N = 100 |
|---|---|---|
| self-attention `B·H·N²` | 0.125 MiB/layer | 1.221 MiB/layer |
| cross-attention `B·H·N·n_tok` | 0.625 MiB/layer | 1.953 MiB/layer |
| token states `B·N·d` | 0.125 MiB/layer | 0.391 MiB/layer |
| **× 3 layers** | **2.63 MiB** | **10.69 MiB** |

⇒ **Δ ≈ 8.1 MiB against a ~500 MB step footprint (1.7 %)** — below the noise floor of the probe,
which is why the RSS reading is null rather than the probe being blind.

### ⭐ The verdict M17 asked for: **the cost does NOT bind**

+8.2 % step time, +0.077 % parameters, ~8 MiB of activations. Nothing here argues for a different
near-field architecture, and nothing argues for a silently smaller N. **N = 100 ships.**

⚠️ **Two honest limits on that verdict.** (1) It is a **CPU** measurement; on a GPU the step is
dominated by the convolutional trunk, so the relative decoder cost should be **no larger** — but
that is a **PREDICTION, not a measurement**, and it is marked as one. (2) It is the **tiny** rung
at batch 4; the registered arm's size and batch differ. **The pod-side re-measure is escalated**,
and the instrument (`measure_query_cost.py`) ships with this package so it is one command there.

---

## P4 — M18 implemented: a hash carries its artifact scope, or the consumer refuses

### The incident, re-MEASURED end to end (`raw/join_digest_scope_incident.json`)

| sidecar | `summary.md5` | `summary.out` names | `xz_of_md5` | declares a scope? |
|---|---|---|---|---|
| `train2400_agents.meta.json` (repo bank) | `24cbdca8…cf76` | `train2400_agents.jsonl.xz` | — | **no** |
| `train2400_agents.jsonl.xz.meta.json` (box) | `24cbdca8…cf76` | `train2400_agents.jsonl.xz` | — | **no** |
| `val40_agents.jsonl.meta.json` | `6d3552ff…40f5` | `val40_agents.jsonl` | — | **no** |
| `val40_agents.jsonl.xz.meta.json` | `6d3552ff…40f5` | ⚠️ **`val40_agents.jsonl`** | **= the same value** | **no** |

Re-hashed here, not copied, on the 136,689,648-byte file:

```
md5(compressed .xz)      = 24cbdca8c3b23aafc2fb17e6bf99cf76   <- what train's sidecar records
md5(decompressed .jsonl) = b7c6b5a7da2d942e8be9dc9fc316e9a9
```

⇒ **a val40-style checker on the train join hashes the decompressed stream, gets
`b7c6b5a7…`, compares it against `24cbdca8…`, and REFUSES a perfectly good file** — reproduced
exactly, in `raw/`.

⛔ **And the obvious repair is MEASURED wrong.** "Read the extension off `summary.out`" fails on
val40's `.xz`-side sidecar, whose `summary.out` names the **`.jsonl`** and whose `xz_of_md5` holds
the **same** digest as `summary.md5`. The heuristic is wrong in the other direction, and a wrong
inference does not fail loudly — it either rejects a good file or **accepts the wrong one and
reports success**. So the module offers no inference at all (pinned by a test that asserts no
`infer`/`guess` symbol exists).

### The contract — `stack/tanitad/data/join_meta.py` (NEW)

Deliberately the **same** shape as `tanitad/refs/anchor_meta.py`, not a second pattern:

* **builders write** — `declare(digest, scope=…, filename=…, algo=…)` / `attach(meta, …)` put a
  top-level `digest_scope` block carrying `algo`, `digest`, `scope ∈ {compressed, decompressed}`,
  the **exact basename**, `declared_by` and a `schema` tag. The legacy `summary.md5` is left
  untouched, so nothing that reads these sidecars today changes behaviour;
* **consumers require** — `read_digest_scope` raises `JoinDigestScopeMissing` on an undeclared
  sidecar, and `verify` additionally refuses when the declared **filename** is not the artifact
  offered ("the digest is about a DIFFERENT file") or when same-scope bytes disagree;
* **migration** — `backfill(meta, path)` **hashes both artifacts** and declares the one the recorded
  digest matched, reporting what it ruled out. It refuses when **neither** matches (an integrity
  failure must not be papered over with an invented scope) **and** when **both** match — which
  happens for an uncompressed `.jsonl`, where a declaration derived from it would later be quoted
  about an `.xz` it does not cover;
* one command: `python -m tanitad.data.join_meta <join> [--write]` (keeps a `.bak`).

### The consumer, wired: `--agent-join-verify {auto,off}` (default `auto`)

| situation | behaviour | stamped in `config.json["agent_join_digest"]` |
|---|---|---|
| sidecar declares a scope | verify under **that** scope | `{verified: true, algo, scope, filename, digest, sidecar}` |
| sidecar exists, declares **nothing** | ⛔ **REFUSE**, naming the one-command migration | — (run does not start) |
| no sidecar anywhere (3 names probed) | **warn**, run continues | `{verified: false, reason: "no sidecar found…"}` |
| `--agent-join-verify off` | skip | `{verified: false, reason: "operator passed --agent-join-verify off"}` |
| sidecar present but unreadable | ⛔ **REFUSE** — an unreadable sidecar is not an absent one | — |

⚠️ The "no sidecar" case **warns rather than refuses** on purpose: an absent sidecar is a *missing*
check, not a *lying* one, and the record says which of the two it was so no reader has to assume.
⚠️ `sidecar_path` probes **three** names (`<file>.meta.json`, the `.xz`-stripped form,
`<stem>.meta.json`) because absence found at one location is not absence — MEASURED: the train
sidecar is `<file>.xz.meta.json`, not the `<stem>.meta.json` the package doc names.

### Demonstrated on the real artifact (`raw/backfill_train2400.json`)

```
$ python -m tanitad.data.join_meta .../train2400_agents.jsonl.xz --write
  recorded    24cbdca8c3b23aafc2fb17e6bf99cf76
  candidates  compressed   24cbdca8c3b23aafc2fb17e6bf99cf76   <- matched
              decompressed b7c6b5a7da2d942e8be9dc9fc316e9a9
  -> digest_scope {scope: compressed, filename: train2400_agents.jsonl.xz,
                   declared_by: backfill-MEASURED}

$ (trainer's own consumer, on the same 137 MB file)
  [v3] agent join verified: md5(compressed of train2400_agents.jsonl.xz)
       = 24cbdca8c3b23aafc2fb17e6bf99cf76
```

⭐ The decompressed digest `b7c6b5a7da2d942e8be9dc9fc316e9a9` reproduces the DataFlyWheel's
independently — a cross-check that the scope determination is right and not an artefact of this
implementation.

⛔ **The repo-banked sidecars were NOT edited.** They are another stream's artifacts; the backfill
ran on the box copy. A test pins the incident against the banked files —
`test_P4_the_two_REAL_sidecars_declare_nothing_today` — and it is written so that when the
DataFlyWheel backfills them it **starts failing**, which is the signal the migration landed.

---

## What changed in the repo

| file | change |
|---|---|
| `stack/scripts/refc_v3_train.py` | `AGENT_QUERIES_DEFAULT = 100`; `_build_rig_camera` + `_read_rig_extrinsics` + `_agent_cam_frames` and the four refusals; `--agent-rig-camera` / `--agent-rig-extrinsics` / `--agent-cam-height` / `--agent-join-verify`; `model._rig_camera` **assigned**; `agent_knob_dests` / `agent_knob_stamp` / `assert_knobs_stamped`; `_verify_agent_join`; `config.json` gains `seams.agent_rig_camera`, `seams.agent_knobs`, `agent_join_digest`; `--agent-queries` help corrected. |
| `stack/tanitad/data/join_meta.py` | **NEW** — the digest-scope contract, its refusals, the measuring backfill and a `__main__` migration CLI. |
| `stack/tests/test_refc_v3_agent_provenance.py` | **NEW** — 38 tests over P1–P4, each with the control that must read a known value and the deliberate regression that must fail. |

**Backward compatibility:** every default is the previous behaviour except `--agent-queries`
(32 → 100, the M17 ruling). A run with both monocular weights at 0 — i.e. every arm banked so far —
is unchanged, and `--agent-join-verify auto` only bites on a sidecar that exists and declares
nothing.

---

## The suite, and the attribution done by CONTROL rather than by argument

**Touched surface — the binding check.** Every test file that imports any module
this change touches (`refc_v3_train`, `refc_agents`, `rig_projection`,
`anchor_meta`), run against the final code: **239 passed, 1 skipped**.

**Whole `stack/` suite:** `6,213 passed · 27 failed · 7 errors · 115 skipped ·
2 xfailed` in 22:30.

⛔ **27 of 27 failures and 7 of 7 errors are NOT attributable to this change —
MEASURED, not argued.** The pre-change `refc_v3_train.py` (blob at `1ea30c8`,
asserted to lack `AGENT_QUERIES_DEFAULT` and `_build_rig_camera`) was shadowed
onto `PYTHONPATH` ahead of the current one — the shadow verified active by
importing it and printing its `__file__` — and the failing files re-run against
it:

| set | with the change | with the PRE-change trainer |
|---|---|---|
| the 9 files failing in run 1 | 13 failed, 7 errors | **13 failed, 7 errors — identical names** |
| the 5 remaining files | 14 failed | **14 failed — identical names** |
| **total** | **27 + 7** | **27 + 7, same set** |

⚠️ **And most of them are not code failures at all** — they are the mirror-resync
trap in `CLAUDE.md`, measured: the off-Drive run surface was missing **93 files**
under `taniteval/` and `tools/`, plus `Project Steering/`'s record files
(`test_decision_check` fails with *"record file(s) absent: DECISIONS_2026-07-20.md,
MODEL_REGISTRY.md, RETRACTION_LOG.md"* — all three present in the repo) and
`E4_SELECTOR_RESOLUTION.md` (present in the repo, absent from the mirror).
Syncing `taniteval/` turned **4** of them green — with this change in place.
The rest are other streams' known items: `test_mktree_commit` (6 — the tool
hardcodes the G: repo path, so its fixture repo is not the repo it acts on),
`test_refav1_kin_contract::test_A6` (named in mm-decisions **M14** as failing
independently), the v6 trainer's own runbook/horizon refusals, and
`test_text_encoding_is_explicit`, whose 9 offenders are all other files.

⚠️ **The first full run reported 38 failures and I could not attribute them from
it** — the background command piped through `tail -20`, so the short summary was
truncated and *"the failures I can see"* was a claim about the pipe, not about the
suite. Re-run with `-rf` and the whole output captured. Same family as *"0 hits is
a claim about the SEARCH"*.

---

## ⚠️ ESCALATIONS — for the Master Mind, not for a reader to find later

1. ⛔ **`AgentSeamConfig.queries` still defaults to 32 in `tanitad/refs/refc_agents.py`, and its
   docstring still carries the val40 claim.** This stream does not own that file (the DataFlyWheel
   was editing it in the same hours), so the trainer was made self-consistent and the library
   default was **not** touched. Any consumer that constructs `AgentSeamConfig()` directly still
   gets 32. **One-line change, needs an owner.**
2. ⛔ **`agent_losses(cam=…)` takes ONE camera per call**, so a run declares one mount pose for the
   whole batch, while the corpus has **two rigs** and per-clip extrinsics. `--agent-rig-camera
   extrinsics` is therefore honest but coarse. Per-clip cameras need a signature change in
   `refc_agents.py`. **Named, not silently accepted.**
3. ⚠️ **The M17 cost number wants a pod-side re-measure** at the registered arm's size and batch.
   `measure_query_cost.py` ships here; it is one command. The CPU verdict (+8.2 %, cost does not
   bind) is not expected to invert, but that expectation is a prediction.
4. ⚠️ **The banked sidecars still declare no scope** and the refcv5 density script's C5 check is
   still the val40-shaped one. The DataFlyWheel owns both; `join_meta.backfill` is the migration
   and `join_meta.attach` is the one-line builder change.
5. ⚠️ **M18 bullet 3 is corrected** (P2): `w_agent`/`w_u0` were in `config.json["seams"]` already.
   Worth a line in the decisions file so the next reader does not re-fix a fixed thing.
6. ⛔ **M18's list of silent flags was INCOMPLETE**: `--w-agent` under `--agents off` is the same
   defect and is the worst of the three, because a join makes the existing loss-time guard miss it.
   Fixed here; flagged so the decisions file records that the audit had a gap, not just a finding.

---

## Manifest

| artifact | lives at |
|---|---|
| trainer changes | `repo:stack/scripts/refc_v3_train.py` |
| digest-scope contract | `repo:stack/tanitad/data/join_meta.py` |
| tests (37 passed + 1 skipped) | `repo:stack/tests/test_refc_v3_agent_provenance.py` |
| this report | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refcv5-flag-provenance/RESULT.md` |
| cost instrument | `repo:…/2026-09-05-refcv5-flag-provenance/measure_query_cost.py` |
| incident probe | `repo:…/2026-09-05-refcv5-flag-provenance/probe_digest_scope_incident.py` |
| cost raw (b2 / b4 replicate) | `repo:…/raw/query_cost_b2_cpu.json`, `raw/query_cost_b4_replicate_cpu.json` |
| digest-scope incident raw | `repo:…/raw/join_digest_scope_incident.json` |
| backfill evidence | `repo:…/raw/backfill_train2400.json` |
| a real run's `config.json` | `repo:…/raw/e2e_run_config.json` |
| backfilled sidecar (box only) | `devbox:C:/Users/Admin/tanitad-data/joins/joins/train2400_agents.jsonl.xz.meta.json` (`.bak` kept) — **exists in ONE place**, and is a local convenience, not a deliverable |

Nothing else lives in only one place.

---

# ADDENDUM (2026-09-05, later the same day) — "every gate shown to FAIL its defect" is now a COMMAND, not a claim

**Scope:** this section adds no fix. Both items of the incoming brief were **already landed** and are
re-verified here by content; what was missing was *evidence that the guards work*, and that is what
this addendum builds.

## 0. The brief was stale, and the first duty was to say so

The brief asked for one of two fixes to `--agent-w-project` / `--agent-w-ground`, plus `w_agent` /
`w_u0` in `_seam_stamp`. **All three already exist at `HEAD` (`49c3aa5`)**, landed by `93bcdd7` and
`606d938` — the work described in the sections above. Verified by content, not by changelog:

| brief item | state at `HEAD` | evidence |
|---|---|---|
| the two monocular flags are silent no-ops | **FIXED, and by BOTH routes** — the camera is built *and* an unbuildable one refuses | `refc_v3_train.py:442` `_build_rig_camera`; `:2108` `model._rig_camera, _cam_stamp = _build_rig_camera(cfg, args)` |
| `model._rig_camera = None`, never assigned | **gone**; a constant-`None` assignment is refused by a parsed (not grepped) test | `test_refc_v3_agent_provenance.py::test_P1_the_trainer_never_assigns_a_CONSTANT_None_rig_camera` |
| `w_agent` / `w_u0` absent from `config.json` | **stamped at the top level of `_seam_stamp`**, plus the derived `agent_knobs` closure | `refc_v3_train.py:1629-1630`, `:1640` |

⇒ **Nothing in the brief needed writing.** Reporting that and stopping would have been a report, not
work — so the question became the one the brief itself poses: *is a guard that cannot fail evidence?*

## 1. The claim that needed testing was OUR OWN

`test_refc_v3_agent_provenance.py` opens with the right doctrine — *"every gate here is shown to FAIL
the defect: a control that has never read the wrong value certifies nothing"* — and the report above
says **"38 new tests, every gate shown to FAIL its defect."**

**That showing was done once, by hand, and lived only in a transcript.** A static assertion cannot
demonstrate it, and the next refactor of `_build_rig_camera` can turn all 38 into tautologies while
the suite still reads green. The claim was true and **unreproducible** — the same shape as a headline
number with no artifact behind it.

## 2. ⛔ THE OBVIOUS INSTRUMENT IS BLIND — MEASURED, AND IT WOULD HAVE PASSED THE BUG

Before writing the audit, the natural tool was built first: an AST census over `build_parser` asking
*"is every CLI dest read somewhere other than the parser and the stamps?"* — the general form of
"a flag that parses and does nothing".

| trainer under test | census verdict |
|---|---|
| `HEAD` (fixed) | **0 suspects** of 71 dests |
| `HEAD` + the ORIGINAL M18 defect re-installed (`model._rig_camera = None`) | **0 suspects** of 71 dests |

⇒ **The census reads identically on the fixed and the broken trainer.** It cannot see the defect,
because `--agent-w-project` **was** read: it reached `cfg.core.agents.w_project` and then died against
a `cam is not None` guard that was always False. **Reachability of the FLAG is not reachability of the
TERM.** Same family as `df` hiding the MooseFS quota and Thor's `free`: a probe answering the adjacent
question, in a voice indistinguishable from the right one. The census is **refuted and not shipped**;
its refutation is recorded in the audit's docstring so it is not rebuilt in six months.

## 3. What was built instead — `stack/scripts/guard_mutation_audit.py`

Nine MEASURED defects, each **reintroduced into the working tree**, each naming the guard that must
catch it. Three verdicts: `CAUGHT` (the named test failed — the guard is load-bearing), `ESCAPED`
(the suite stayed green — **the guard is decorative**), `MISCREDITED` (red, but via some other test —
the guard credited is not the guard holding). Non-zero exit on anything but `CAUGHT`.

```
python stack/scripts/guard_mutation_audit.py            # the audit (~2 min)
python stack/scripts/guard_mutation_audit.py --list     # the registry
python stack/scripts/guard_mutation_audit.py --check-anchors   # fast, no pytest
```

### The result: **9 / 9 CAUGHT by the named guard**

| mutation | reintroduces | caught by |
|---|---|---|
| `rig_camera_none` | `model._rig_camera = None` — the literal 2026-09-05 defect | `…provenance.py::test_P1_the_trainer_never_assigns_a_CONSTANT_None_rig_camera` |
| `off_camera_refusal_removed` | the pin-time refusal deleted, so an argv exists that stamps a weight it never trains | `…::test_P1_weight_without_a_camera_REFUSES` (both params) |
| `monocular_terms_skipped` | `agent_losses` drops the projection term | `…::test_P1_the_camera_the_TRAINER_builds_makes_the_terms_COMPUTE` |
| `ground_term_skipped` | …and the ground-range prior, **registered separately** so one guard cannot certify both addends for free | same |
| `unsupervised_detector_allowed` | `--agents head --w-agent 0` — the fourth dead flag | `…wiring.py::test_agent_head_without_its_LOSS_REFUSES` |
| `weights_absent_from_config` | `w_agent`/`w_u0` dropped from `_seam_stamp` | `…wiring.py::test_seam_stamp_carries_every_refcv5_lever` |
| `knob_stamp_emptied` | the derived `agent_knobs` closure defeated | `…::test_P2_every_knob_is_recoverable_from_the_stamp_BY_VALUE` |
| `registry_anchor_rotted` | **the audit's own anchors rotted** against a refactor | `test_guard_mutation_audit.py::test_every_anchor_is_present_…` |
| `registry_names_a_dead_guard` | **the audit names a renamed/deleted test** | `test_guard_mutation_audit.py::test_every_named_guard_actually_exists` |

The last two are the recursive pair: they mutate the audit file itself, which the pytest subprocess
re-imports from disk while the parent keeps its own in-memory registry. **The audit rots itself,
watches the rot check catch it, and puts itself back** — because the rot check is the single thing
standing between this tool and silent vacuity, and it too had never been seen to fail.

## 4. ⛔⛔ TWO DEFECTS IN THE INSTRUMENT ITSELF, BOTH IN THE BRANCH THAT MATTERS MOST

Neither would have been found by reading the code, and both sat in `ESCAPED` — the verdict that fires
least often and says the most.

**(a) `9/9 CAUGHT` was, for one iteration, produced by a tool that COULD NOT SAY ANYTHING ELSE.**
The rot check ran inside every mutation run, and the applied mutation's own anchor is — correctly —
absent while it is applied. So that test went red on *every* mutation, `named` was never the only
failure, and **`ESCAPED` became unreachable by construction**: a decorative guard would have read
`MISCREDITED`. Fixed by passing the applied mutation's **key** (not a boolean) into the subprocess and
excluding exactly that one anchor; every other anchor is still checked, which is what catches
`registry_anchor_rotted`. **This is the audited defect class occurring inside the audit** — an
instrument whose only reachable answer is the reassuring one.

**(b) the `ESCAPED` message crashed the tool.** This console is cp1252; `print("⛔ the suite stayed
GREEN …")` raised `UnicodeEncodeError` on `U+26D4`. The single most important sentence this tool can
emit — *your guard is decorative* — **died instead of being read**, on the branch that had never once
been exercised. Every literal reaching stdout is ASCII now, `_harden_streams()` is the backstop for
prose that is not ours, and both are pinned by tests.

**(c) and a third, on the DECODE side, found by the repo's own standard.** The audit reads pytest's
output with `subprocess.run(..., text=True)` and no `encoding=`, which decodes with the locale codec.
The text being decoded is pytest's failure summary — and the assertion messages of *the very guards
under audit* contain `U+26D4`. So the **first genuinely-caught defect** would have surfaced as
`UnicodeDecodeError` inside the audit instead of as a verdict. Caught by
`tests/test_text_encoding_is_explicit.py::test_subprocess_text_mode_always_names_its_encoding`, which
is **already red at baseline on nine pre-existing offenders** — this file is not among them; it was
fixed before landing.

⇒ **Three encoding defects in one instrument, all on the failure path**: encode on the way out (b),
decode on the way in (c), and the verdict-collapse (a) that hid both by making the failure path
unreachable. The repo's text-encoding sweep and its `test_the_guard_can_actually_fail` were doing
exactly their job.

⭐ **The crash paid for a safety proof.** It aborted mid-mutation, and the `finally` restore returned
both files to byte-identical (`md5` match against the repo) and removed the snapshot directory. The
in-place mutation design was **shown** safe under a real crash, not argued to be.

## 5. All three verdicts are reachable — validated, not assumed

`9/9 CAUGHT` is worthless from an instrument that can only say `CAUGHT`. Two synthetic mutations
(not committed) were run against the shipped audit:

| synthetic mutation | expected | got |
|---|---|---|
| an **inert** edit (a comment reworded), declared as guarded by a real test | `ESCAPED` | **`ESCAPED`**, with the "the guard is decorative" line |
| the **real** `rig_camera_none` defect, credited to a real test that does not fail | `MISCREDITED` | **`MISCREDITED`**, naming the guard that did not hold |
| audit exit code | non-zero | **1** |

## 6. Manifest (addendum)

| artifact | lives at |
|---|---|
| the audit (9 mutations, ~2 min) | `repo:stack/scripts/guard_mutation_audit.py` |
| the guard-for-the-guards (11 tests, 0.15 s) | `repo:stack/tests/test_guard_mutation_audit.py` |
| audit output, 9/9 CAUGHT | `repo:…/raw/guard_mutation_audit_2026-09-05.txt` |
| verdict-reachability validation (ESCAPED + MISCREDITED) | `repo:…/raw/verdict_reachability_2026-09-05.txt` |
| the refuted census, both trainers | `repo:…/raw/census_blind_spot_2026-09-05.txt` |

Nothing in this addendum exists in only one place.

## 7. Open, named rather than silently accepted

1. ⚠️ **The audit covers four test files and nine defects — not the whole suite.** A guard outside
   `TEST_FILES` is still an unmeasured guard. Extending the registry is cheap; the anchor discipline
   (exactly one occurrence) is what keeps it honest.
2. ⚠️ **It mutates the working tree in place.** That is documented, snapshot-backed, sha256-verified
   and crash-recovering, but it is not safe to run two copies at once, and it refuses nothing today
   if a mutated path carries uncommitted edits — the snapshot would faithfully restore *those*, which
   is correct, but the run would be measuring an untracked tree. **Named, not fixed.**
3. ⚠️ **`ESCAPED` and `MISCREDITED` are validated by a throwaway script, not by the committed suite.**
   Committing them would mean shipping a deliberately decorative guard, which is worse; the
   validation script's content is reproduced in
   `raw/verdict_reachability_2026-09-05.txt` so the check can be repeated.

## 8. ⛔ ESCALATION — this wants a line in the standing cadence, and I did not take it

`Project Steering/AGENT_OPERATING_STANDARD.md` already runs three drift checks on a cadence, on the
stated principle that *"presence proves transfer, md5 proves bytes, a successful import proves
loading — none of them proves currency."* **This audit is the fourth rung of that same ladder: none
of the three proves a guard still guards.**

It belongs beside them under *Standing cadence*, run when the refcv5 surface is touched:

```
python stack/scripts/guard_mutation_audit.py   # non-zero if any guard is decorative
```

**I did not make that edit.** The file carries another stream's uncommitted changes right now
(` M` in `git status`), and editing a shared steering doc under someone else's in-flight edit is how
work gets silently reverted. **Raised here rather than written into a README** — per rule 3 of the
standard, which exists because "please merge this" in a README went unread for ten days.

---

# ADDENDUM 2 — the rot check earned its keep **within the hour**, and a tenth defect

Written after `e8de537` landed. Everything below is MEASURED on that commit.

## 9. ⭐⭐ THE INSTRUMENT'S FIRST REAL REFACTOR CAME BEFORE ITS FIRST DAY

The audit committed in `a1a894a` claims nine guards are load-bearing. About an hour later
`e8de537` landed the per-clip camera and rewrote `agent_losses`' guard from
`if cam is not None and cfg.w_project > 0.0:` to a per-row
`if _n_cam and cfg.w_project > 0.0:`.

**Two of the nine anchors stopped matching, and the very next full-suite run said so** —
by name, in the same run that was checking it:

```
FAILED tests/test_guard_mutation_audit.py::test_every_anchor_is_present_in_the_shipped_source_exactly_once
E   monocular_terms_skipped: anchor occurs 0x in tanitad/refs/refc_agents.py (need exactly 1)
E   ground_term_skipped:     anchor occurs 0x in tanitad/refs/refc_agents.py (need exactly 1)
```

⇒ **Without that check the audit would have applied nothing, caught nothing, and printed a clean
`9/9` over a registry that no longer touched the guard it claimed to test** — a green audit
certifying an absent guard, on day one. The anchors are **re-pointed, not deleted**: the guard
still exists, so its evidence must too. That rule is now written into the audit's docstring beside
the incident.

⚠️ **This is also the honest reading of the full-suite delta below.** My run shows one more failure
than baseline, and that failure is *the instrument working correctly*, not a regression.

## 10. A TENTH DEFECT — the leg of the trip nothing was watching

Found by asking which end of the wire each existing gate reads. **Every P1 gate reads argv**: the
refusals read `args`, and `config.json`'s `agent_rig_camera.w_project` reads `args`. The
"terms COMPUTE" test hand-builds an `AgentSeamConfig(w_project=1.0)`. **But the loss reads
`cfg.core.agents`** — and nothing asserted that argv reaches it.

So dropping the weight in `_pin_refcv5_seams` alone re-opens M18 in its **exact original shape**:

| what still passes with the weight dropped | why |
|---|---|
| the camera builds | `_build_rig_camera` reads argv |
| `--agent-rig-camera off` still refuses correctly | reads argv |
| `config.json` still reads `agent_rig_camera.w_project = 0.2` | reads argv |
| **the addend is gone** | the loss reads `cfg.core.agents.w_project` |

MEASURED 2026-09-05: **the plumbing was correct, the coverage was not** — argv → config was
verified working, and `grep` over `stack/tests/**` found no assertion on it. Now pinned by
`test_P1_the_two_weights_SURVIVE_the_trip_from_argv_to_the_LOSS_CONFIG`, which gives the two
weights **different** values (0.2 / 0.1, so a copy-paste reading `agent_w_project` into both is
caught), requires the two independent stamps to **agree**, and carries a zero-weight control.
Registered as mutation `weight_lost_between_argv_and_config`.

**Result: 10 / 10 defects CAUGHT by the named guard**, anchors 10/10 present exactly once.

## 11. Convergent evidence — the sibling stream found the same family one level deeper

`e8de537` reports that `--agent-w-ground` computes a **tautology**: `ground_range_prior` projects a
foot at rig z = 0 and back-projects onto `ROAD_PLANE_Z_M`, which *is* 0.0, so the two are exact
inverses and the loss is identically zero (2.61e-08, parameter gradient 8.73e-11). The flag is set,
the camera is built, **the term runs** — and adds nothing.

⇒ Three distinct depths of one defect, found independently on one day:

| depth | the flag… | caught by |
|---|---|---|
| M18 | never reaches the term (`_rig_camera = None`) | a refusal + these mutations |
| **§10 here** | reaches the *refusal* and the *stamp* but not the *loss config* | the new pin |
| `e8de537` | reaches the loss, and the loss is **identically zero** | a **gradient** probe |

**No flag guard and no camera guard can see the third**, and no mutation of a guard can either —
that one needs a gradient. Recorded here because the three together say the real invariant is
*"the weight changes the gradient"*, and only the third measures it.

## 12. Full-suite regression — the numbers, and their admissibility

| run | failed | passed | skipped | errors | `Errno 22` | wall |
|---|---|---|---|---|---|---|
| baseline (`raw/fullsuite_postchange.txt`, before this work) | 27 | 6,213 | 115 | 7 | **0** | 22:30 |
| with this work | **28** | 6,223 | 115 | 7 | **0** | 32:14 |

`comm` over the two sorted FAILED sets: **exactly one entry differs**, and it is
`test_every_anchor_is_present_in_the_shipped_source_exactly_once` — §9. **Zero in baseline, zero in
mine.** Nothing regressed; `+10 passed` is this addendum's 11 tests minus that one. Both runs carry
**zero `Errno 22`**, which is what makes either number quotable at all
(`gdrive-mount-hard-failure`: a `G:` failure count is admissible only with zero `Errno 22`).

⚠️ **The 28-failure run predates the fix in §9 and the pin in §10.** The four touched test files are
green at 79 passed / 1 skipped on `e8de537`, and the audit is 10/10 — but **the whole suite has not
been re-run since**, and I am not claiming a 27 I did not measure.

## 13. Manifest (addendum 2)

| artifact | lives at |
|---|---|
| audit, re-pointed + 10th mutation | `repo:stack/scripts/guard_mutation_audit.py` |
| the new pin | `repo:stack/tests/test_refc_v3_agent_provenance.py` |
| audit output, 10/10 on `e8de537` | `repo:…/raw/guard_mutation_audit_2026-09-05.txt` |
| full-suite log, this work | `repo:…/raw/fullsuite_guardaudit_2026-09-05.txt` |

Nothing here exists in only one place.
