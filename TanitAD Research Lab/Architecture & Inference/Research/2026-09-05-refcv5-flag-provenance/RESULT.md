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
| backfilled sidecar (box only) | `devbox:C:/Users/Admin/tanitad-data/joins/joins/train2400_agents.jsonl.xz.meta.json` (`.bak` kept) — **exists in ONE place**, and is a local convenience, not a deliverable |

Nothing else lives in only one place.
