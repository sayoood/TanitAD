# refcv6 §4/§5 — the tactical behaviour decoder and the 4-value set-speed now REACH TRAINING

**Date** 2026-09-17 · **Base** `837c308` (branch `agent/arch-inf-20260803`) ·
**Worktree** `C:/Users/Admin/tanitad-wt-tactical` · **Device** CPU only (the GPU
was in use: 1,451 MiB / 8,188 MiB, 29 % util at start).

⛔ **THIS IS A WIRING RESULT. NO CAPABILITY CLAIM IS MADE AND NO METRIC FAMILY IS
REPORTED.** No GPU arm was launched from this package. Every number below is
about whether a gradient exists and whether a guard fires — never about whether
the car drives better. The four metric families are not reported because nothing
here measures driving.

---

## 1. The headline

| PI ask | before (tip `837c308`) | after |
|---|---|---|
| §4 tactical layer learns valid behaviours from the scene | built, forward-run, **`grad_abs_sum` EXACTLY 0** — the trainer contained **ZERO** occurrences of `tacv6` | **all 8 probed heads REACH**, `grad_abs_sum` 0.294 – 699.2 on in-band steps |
| §5 four-value max-speed as an INPUT | **no sidecar payload on disk**, no flag, no data channel | sidecar built for both splits, `--max-speed-input-v6` reaches the condition, **23,772/23,772 eval windows fed** |

⚠️ **AND ONE HALF OF THE §4 ASK IS BLOCKED AND WAS NOT DELIVERED.** The PI asked
for behaviours learned *"from the scene embeddings, for the agent **and the
map**"*. **Only the agent half is reachable.** §5 of this document names the seam
exactly; the run record stamps `sources: ["agent"]` and
`bev_tokens_reach_decoder: false` so no later reader can credit an arm with a map
it never had.

---

## 2. What I verified before building (the brief's claims, re-checked)

MEASURED, all at `837c308`:

| claim | verdict |
|---|---|
| `tacv6` has zero readers in the trainer | **CONFIRMED** — `grep -c tacv6 scripts/refc_v3_train.py` = **0** |
| no CLI flag exists | **CONFIRMED** — `--tac-goal-tok-head` / `--tac-goal-negatives` are the *old* v7 goal head |
| `graft_behaviour_sel` defaults False with no flag | **CONFIRMED** (`refc.py:818`, `:1649`) |
| 2,262,020 params at `grad_abs_sum` 0 | **CONFIRMED, and the config identified**: that figure is the **`d_bev = 256`** build. Default `d_bev = 128` → 2,229,252; `d_bev = 1024` → 2,458,628. The arm actually buildable today is **agent-only, 2,228,996** (`d_agent = 384` from `cfg.core.decoder.d`). |
| the §5 builder "was never run and no sidecar exists" | **PARTLY WRONG.** The builder *was* run on 2026-09-16 — `speed_max_window_v6_{train,eval}.meta.json` are banked at `…/2026-09-16-refcv6-tactical/`. What is missing is the **payload**: only the `.meta.json` was banked, never the `.jsonl`. My rebuild reproduces that meta's census **to the digit** (1741/1593/1014/224, entropy 1.7555, over-ceiling 86, source md5 `fa89ea55…`), which is what makes the rebuild trustworthy rather than merely plausible. |

⛔ **AND A LABEL-FACT CORRECTION THAT MATTERS FOR EVERY TACTICAL NUMBER.** The
brief states *"the head mask is 21/22 trainable, 9 of 22 on the `goal_pos_weight`
cap"*. MEASURED on `s2_labels_v8.0_train.jsonl.gz` (md5 `fa89ea55…`, n = 4,572),
those are the **`cot-absence-negative`** numbers. **The trainer's DEFAULT is
`measured`**, and it gives different ones:

| `--tac-goal-negatives` | trainable | ON the pos_weight cap (50) | under the n = 200 floor |
|---|---|---|---|
| `measured` **(the default)** | **17 / 22** | **3** | 10 |
| `cot-absence-negative` (the PI's ruling) | **21 / 22** | **9** | 10 |

⇒ **A tactical number must name its negatives policy**, not just its split. The
floor count is the exception and is policy-independent (it counts positives).
Raw: `raw/label_facts.json`. The trainer now computes all four figures **from the
loaded split** and stamps them into `config.json` under
`refcv6_tactical.label_limits`, and prints them at launch — so an arm cannot be
quoted against the wrong policy from its own artifact.

---

## 3. What I wired

### §4 — the behaviour decoder

* **Flags** (`refc_v3_train.py`): `--tac-decoder-v6`, `--w-tac-v6`,
  `--tac-decoder-d-bev`, `--tac-decoder-valid-threshold`,
  `--graft-behaviour-sel`.
* **Pin** — new `_pin_refcv6_tactical(cfg, args)`, called from
  `_pin_trainer_cfg`, builds `TacticalDecoderConfig(d_agent=cfg.core.decoder.d,
  d_bev=…, sources=…)` and carries six refusals (§4 below).
* **Loss** — a new block in `compute_losses_v3` reads `out["tacv6_*"]` and calls
  `refcv6_tactical.tactical_behaviour_losses` with the **same** `goal_pos_weight`
  and `goal_class_mask` the `--w-tac-goal` channel fits from the train split.
  ⛔ At `w <= 0` the term **never enters the graph** — an absence, not a
  multiplication by zero — which is what buys the bit-identity in §6.
* **Labels** — `ds.tac_goal_targets` now switches on for `--w-tac-v6 > 0` as well
  as `--w-tac-goal > 0`, on **both** the train and the eval dataset. The eval half
  is not symmetry: `compute_losses_v3` raises `SystemExit`, which
  `except Exception` cannot catch, so without it a refcv6 arm dies at the FIRST
  in-training eval with the training compute already paid for (arm `C_w0p05` of
  the 2026-09-11 sweep died exactly this way).
* **Telemetry** — `tacv6_goal_bce`, `tacv6_goal_conf_bce`, `tacv6_lat_ce`,
  `tacv6_lon_ce`, `tacv6_total_weighted`, and **`n_supervised` for all three
  terms** plus `tacv6_n_scene_mean`.
* **Stamps** — `seams.tac_decoder_v6` carries `requested` / `cfg` / `built` /
  **`w`** / `param_breakdown` / `provenance` / `loss_weights` / `sources` /
  `bev_blocked_by`. The `w` field is the fact the `tac_goal_tok_head`
  post-mortem lacked: that head was `requested`, `cfg` and `built` all true for
  40,284 steps and still learned nothing, and nothing in the record said whether
  a loss could reach it.
* **`REFC_WEIGHT_GATES`** gains a `w_tac_v6` row. This is not packaging: the
  audit **enumerates declared weights** and is therefore structurally blind to a
  head that has none — the exact blindness that hid `tac_goal_tok_head`. The
  exhaustiveness test over the parser passes (`found == set(REFC_WEIGHT_GATES)`,
  10 flags).

### §5 — the four-value set-speed

* **Builder rerun** for both splits, and **extended**: every row now carries
  `sid = stable_episode_id(clip_id)`, and `--omit-clip-id` drops the plaintext
  column. Two independent reasons, both binding — `sid` is the corpus's only
  admissible join key (`LazyV2Episode` carries the stable id, not the clip_id
  string), **and** a raw clip_id is UUID-shaped, which
  `tests/test_refcv6_no_session_paths.py` forbids in a repo artifact. A
  collision on `sid` refuses rather than silently dropping a clip.
* **Reader** — `refcv6_max_speed.read_speed_max_sidecar_v6`, in the module that
  owns the ladder. It ships the **raw `v_hi_ms`** and applies the ladder exactly
  once, on the model side; it **re-derives each row's bin independently** and
  refuses a disagreement; it refuses an md5 mismatch against the label blob.
* **Dataset** — `V3Dataset.enable_max_speed_v6`, reusing the `v_max_ms` batch key
  (safe only because the two ceiling channels are mutually exclusive, refused in
  two places).
* **Stamp** — `speed_max_derivation_v6` in `config.json`, enforced by
  `assert_speed_max_stamp_v6` **in both directions** (a channel with no stamp,
  and a control carrying one).

**Sidecars produced** (md5 of the repo-bankable, `sid`-keyed, clip_id-free form):

| file | rows | valid | md5 |
|---|---|---|---|
| `raw/speed_max_window_v6_train.jsonl` | 4,572 | 4,572 | `65d1bdad68712566408caee24bf8b8d2` |
| `raw/speed_max_window_v6_eval.jsonl` | 147 | 147 | `551a20b9281359836c927c7aa5e02f27` |

Census reproduces the 2026-09-16 banked meta exactly: **(0,30] 1,741 (38.08 %) ·
(30,50] 1,593 (34.84 %) · (50,100] 1,014 (22.18 %) · (100,120] 224 (4.90 %)**, of
which **86 (1.88 %) are CLAMPED from above 120 km/h** — on those the fed ceiling
is one the ego demonstrably exceeded. Entropy **1.7555 bits**.

---

## 4. The refusals — proven by MUTATION, with green controls

`code/refusals.py` → `raw/refusals.json`: **25 cases, 25 OK, 0 wrong, exit 0.**
Fifteen REDs, each paired with the nearest GREEN that must pass.

⭐ **The green controls earned their place on the first run.** My base argv was
missing `--agent-join`, so a **pre-existing** guard refused it — and **all four
green controls read REFUSED while all fifteen REDs also read REFUSED**. A
RED-only table would have read as "every guard fires" with a *brick* underneath.

| refusal | why it exists |
|---|---|
| `--tac-decoder-v6` with `--w-tac-v6 0` | **THE defect**: 2.26 M params built, stamped, zero gradient |
| `--w-tac-v6 > 0` without the decoder | no `tacv6_goal_logits` for the loss to read |
| decoder on `--arm flat` | the agent slots and the selection seam are hierarchy-only |
| decoder without `--v7-labels` | kin3 has no 22-token goal set |
| decoder **and** `--tac-goal-tok-head` | two heads on one target = non-attributable |
| `--tac-decoder-d-bev > 0` | **the blocked map seam** (§5) |
| `--graft-behaviour-sel` without the decoder | silently inert flag |
| `--max-speed-input-v6` without a sidecar | the 4 slots would be a constant all-zero pad |
| E16 **and** refcv6 ceilings together | two ladders over two source fields |
| `--max-speed-input-v6` without the decoder | the one-hot's only consumer is absent |
| sidecar: no `sid` / bin ≠ ladder / all-invalid / empty / md5 mismatch | five reader guards |
| `assert_speed_max_stamp_v6`, both directions | missing stamp, and the mirror |

**Model-vs-argv refusal.** A second check reads `model.tac_decoder_v6` off the
**built object** and refuses a disagreement with the weight. This is *not*
redundant with the pin: the 2026-09-17 `--image-hw` post-mortem is exactly the
case where a refusal that diffs argv **passed** while the defect lived between
argv and the model.

**Permanent regression guard**: `stack/tests/test_refcv6_tactical_training.py`,
**26 tests, all passing**, including the green control, the gate-closes-on-each-
missing-precondition mutation, the built-object assertion below, and
literal-valued default assertions.

**Built-object verification** (not cfg, not argv): `--graft-behaviour-sel`
reaches `AnchoredDiffusionDecoder.graft_behaviour_sel` (`refc.py:1792`) — four
constructor hops from the Namespace. **MEASURED both arms**: `True` with the
flag, `False` without, with the decoder built either way (2,228,996 params). A
single-arm assertion would pass on a decoder that hardcodes `True`.

### Deliberate-regression arms — `raw/mutation_proof.json`, **5 / 5 proven**

Each arm edits the **real source**, runs the **real pytest**, and restores the
file with an **md5 assertion on the restore**. An arm counts only when the
**unmutated** test PASSES *and* the **mutated** test FAILS.

| defect reintroduced | unmutated | mutated | restored |
|---|---|---|---|
| the zero-weight refusal deleted | PASS | **FAIL** | ✓ |
| the BEV-token refusal deleted | PASS | **FAIL** | ✓ |
| `w_tac_v6` removed from the exhaustive audit | PASS | **FAIL** | ✓ |
| the `sid`-before-`valid` ordering reverted | PASS | **FAIL** | ✓ |
| `assert_within_budget` neutered | PASS | **FAIL** | ✓ |

⚠️ **The first run of this harness produced an artifact that parsed as nothing
while every number in it was correct** — `subprocess.run(text=True)` decoded
pytest's `⛔`/`⭐` output as **cp1252**, raising `UnicodeDecodeError` inside the
reader **thread**; the exception printed but did not propagate, so the traceback
landed *ahead of the JSON*. Fixed with `encoding="utf-8", errors="replace"`, and
recorded because it is the cp1252 family CLAUDE.md names — *a truncated artifact
that reads like a complete one*, here inverted into a complete artifact that
reads like a broken one.

---

## 5. ⛔ WHAT I COULD NOT DO — the blocked seam, named precisely

**The map half of the PI's §4 ask is not reachable and I did not fake it.**

The chain, read from source:

1. `refcv6_tactical.TacticalBehaviourDecoder.forward` **does** accept
   `bev_tokens` / `bev_pad` and declares `SCENE_SOURCES = ("agent", "bev")`.
2. `refc_v3.RefCV3Model._scene_hook`'s inner `scene_hook(agent_tokens,
   agent_pad, bev_tokens, bev_pad)` **does** forward them (`refc_v3.py:1624`).
3. `refc.py`'s core forward **does** take `bev_tokens` (`refc.py:3837`) and pass
   it to the hook (`refc.py:4154`).
4. ⛔ **`refc_v3.RefCV3Model.forward` never passes `bev_tokens=` to
   `self.core(...)`.** Its call site passes `**_core_kw`, which carries
   `scene_hook` and nothing else. `grep -n bev_tokens refc_v3.py` returns
   **four** hits, all inside `_scene_hook`, none at the `self.core(...)` call.
5. ⛔ And there is nothing to pass even if it did: the BEV encoder is
   `model._perception`, which lives on the **trainer's** wrapper and runs
   **after** the core forward, on `out["fmap_s16"]` (trainer line ~3490). At the
   instant the hook fires (`refc.py:4154`) no BEV token exists.

⇒ Wiring `d_bev > 0` today would declare a key/value source that never arrives,
and the arm would read as **"the map adds nothing to behaviours"** while never
having had a map. So `--tac-decoder-d-bev > 0` **REFUSES**, `d_bev = 0` is the
buildable arm, and the run record stamps the exclusion.

**What would unblock it** — a decision I did not take because it is an
architecture change, not wiring: the BEV encoder must move from the trainer
wrapper into the model so it runs on `fmap_s16` *inside* `refc.py`'s forward,
before line 4154. That also couples the tactical layer's gradient to the
perception branch, which needs a PI call on whether the behaviour decoder may
backprop into the shared trunk through the BEV path — the detachment argument in
`refcv6_tactical.planner_feeds` is about the *planning* loss and does not settle
this.

**Also not done, and deliberately:** no GPU arm, no capability claim, no metric
families, and **no rebalancing of `MANEUVER_WEIGHT`** — the tactical weights
re-split the existing 0.1 budget exactly (0.04 validity + 0.01 confidence + 0.025
lat + 0.025 lon), and `assert_within_budget` refuses an inflation. Changing the
total is PI queue item 10.

---

## 6. Bit-identity at the 0.0 default

`code/bitid.sh` + `code/bitid_check.py` → `raw/bit_identity.json`. Tip worktree
`C:/Users/Admin/tanitad-wt-tactical-tip` @ `837c308` vs the patched worktree,
same seed, flags absent.

| check | result |
|---|---|
| **`ckpt.pt`** | **BITWISE-IDENTICAL** — 193 tensors, **131,977 elements**, **0 tensors bitwise different**, no key added or removed |
| **mutation control** | **DETECTS-EVERY-ONE-BIT-FLIP — 193 / 193.** Flipping the low bit of byte 0 of *every* tensor in turn is caught every time |
| **`metrics.jsonl`** | **IDENTICAL-MODULO-WALLCLOCK** — the only differing field is `elapsed_s`, which also differs tip-vs-tip2 |
| **new metrics keys** | **none** (`tactical_keys_present: []`), with `has_loss_key: true` as the same-breath control |
| **`config.json`** | **342 → 376 leaves, 34 added, 0 removed, exactly ONE pre-existing leaf changed** |

⚠️ **I report the config verdict as `CHANGES-A-PRE-EXISTING-VALUE`, not as
green.** The one leaf is **`effective_weights.n_terms` 9 → 10** — the exhaustive
weight audit counting itself, which *must* increment when a weight flag is added.
Nothing that affects training changed. I state it this way rather than
special-casing it away, because a checker that exempts its own known diff is the
check-shares-the-defect failure.

⭐ **The mutation control is what makes the ckpt line evidence.** The perception
agent's file records that a `+ 1e-7` perturbation is **below the ULP** on a
float32 near 1.0, so `x + 1e-7 == x` exactly and the control reported
`detects: false` on a sound comparison. The bit-flip is representable by
construction on every dtype. I reused that file rather than re-deriving it.

---

## 7. Gradient reach — per head

`code/grad_reach.py` → `raw/grad_reach.json`, from a live CPU run on the real
artifacts (139-clip `v2ep-eval139-256x1024cyl`, resnet34 trunk, 3-D agent join,
per-clip extrinsics, v8.0 eval labels), `--w-tac-v6 1.0`, `--graft-behaviour-sel`,
`--max-speed-input-v6`. **13 steps, 8 in band (61.5 %).**

| module | params | `n_grad_none` | `grad_abs_sum` in band (min → max) | verdict |
|---|---|---|---|---|
| `tac_decoder_v6` (whole) | 2,228,996 | **0** | 90.41 → 699.2 | **REACHES** |
| `.validity_head` | 257 | 0 | 0.294 → 6.362 | **REACHES** |
| `.conf_head` | 257 | 0 | 0.540 → 0.711 | **REACHES** |
| `.lat_head` | 257 | 0 | 0.191 → 0.262 | **REACHES** |
| `.lon_head` | 257 | 0 | 0.162 → 0.262 | **REACHES** |
| `.queries` | 9,728 | 0 | 5.382 → 15.96 | **REACHES** |
| `.layers` | 2,118,144 | 0 | 77.75 → 621.4 | **REACHES** |
| `.agent_in` | 98,560 | 0 | 5.491 → 47.47 | **REACHES** |

Losses on a representative in-band step: `goal_bce 2.321 · goal_conf_bce 0.683 ·
lat_ce 2.067 · lon_ce 2.086 · total_weighted 0.2035`, with
`n_supervised_goal_cells 5 · lat 1 · lon 1` and `n_scene_mean 16.0`.

⛔⛔ **THE OUT-OF-BAND ZERO, AND WHY IT IS NOT A FAILURE — read this before
quoting any zero from this channel.** My first live run (3 steps) read
`grad_abs_sum` **0.0 on every head**. That is **correct behaviour**, not a dead
head: `v7_labels.tactical_goal_targets` returns **all-`IGNORE_W` outside the
record's ±2 s band** (`v7_labels.py:1019`), so an out-of-band step legitimately
produces loss 0.0 and gradient 0.0. `n_supervised_goal_cells = 0` is the only
thing that distinguishes it from a dead head — which is exactly why the
telemetry emits `n` for every term, and why `grad_reach.py` reports the two
populations separately and says **INCONCLUSIVE**, never "still zero", when no
in-band step was sampled.

⭐ **`n_grad_none` is the column that separates the two failure modes outright.**
`p.grad is None` means *never wired* (the tip state); a zeros gradient means
*wired and unsupervised*. It reads **0 on every module on every step**.

---

## 8. Suite

| suite | result |
|---|---|
| new module `test_refcv6_tactical_training.py` | **26 passed** |
| directly-affected modules (7) | **240 passed** |
| the 7 + the new module, re-run after every edit | **266 passed** |
| session-path guard, with my 17 files registered in `OWNED` | **4 passed** |
| `--preflight` with the tactical flags | **exit 0, ✅ PASS**, ledger shows `tac_decoder_v6: 2228996`, `tac_behaviour_gate_v6: 2816` (`raw/preflight.txt`) |
| full suite to completion | ⚠️ **NOT RUN TO COMPLETION** — see below |

⚠️ **I DID NOT RUN THE FULL SUITE TO GREEN, AND I AM NOT CLAIMING I DID.** On
this box it is hours long on CPU (it reached ~31 % in ~40 min with three jobs
competing, and ~6 % in several minutes running alone). What I have instead is
the identity comparison below, which is *stronger* than a count for the question
that matters — *did my change alter any test outcome?* — and weaker for the
question *is the suite green?*, which it is not and was not at `837c308` either.

⭐ **THE STRONGEST SUITE EVIDENCE IS AN IDENTITY, NOT A COUNT.** I ran the full
suite **concurrently on the patched worktree and on the untouched tip**
(`tanitad-wt-tactical-tip` @ `837c308`) and compared their pytest progress
streams line by line:

```
patched_lines 37 · tip_lines 37 · compared 37 · identical_lines 37
first_divergence  null
md5(prefix)       e02f63d462bf57e50ade5bb06ee6e1c2   — IDENTICAL on both
```

≈ **2,664 test outcomes, byte-identical, zero divergence** — including the
*failures*, which therefore all pre-date this work. (Both runs were stopped at
the same point to free CPU for the mutation arms; the identity holds over
everything that ran.)

⚠️ **Two collection ERRORS are PRE-EXISTING at `837c308` and are not mine** —
`tests/test_metric_decode_refusal.py` (`ImportError: cannot import name
'UntrainedMetricReadout' from 'tanitad.models.v6'`) and
`tests/test_refa_v1_dk_hook.py` (`ImportError: cannot import name
'DistanceKeepingSpec' from 'tanitad.refs.refa_v1'`). They abort collection
entirely, so both suites run with `--ignore` on those two, and the tip run is the
control that shows they pre-date this work.

⚠️ **The tracked diff is 4 files, 913 insertions, 23 deletions — and I read every
deletion.** All 23 are lines I rewrote in place (the builder's row construction,
the `v_max_ms` batch gate, the `w_tac_goal` condition, the `__all__` tuple).
None is a removal of someone else's work.

---

## 9. Defects found and fixed *in my own work* (reported as prominently as the wins)

1. ⛔ **My sidecar reader named a symptom as its cause.** On a `sid`-less
   sidecar it reported *"NOT ONE of the sidecar's 147 rows carries a valid
   ceiling"* — **false**: all 147 carried one and were skipped for having no
   `sid`; `n_valid` was 0 only as a consequence. A true-sounding message naming
   the wrong cause sends the reader to rebuild the labels instead of re-keying
   the sidecar. **Fixed** by ordering the cause before the symptom, and pinned by
   `test_a_sid_less_sidecar_names_the_CAUSE_not_the_symptom` — whose
   discriminating half asserts the symptom text is **absent**, since asserting
   only that `sid` is mentioned would pass on the old message too.
2. ⛔ **My first refusal harness mutated argv by index** and produced malformed
   command lines, so every case "refused" for the wrong reason and the table
   would have read as a perfect score. **Fixed** with an explicit `drop()` that
   **asserts it removed something**.
3. ⛔ **My first green controls all failed** because the base argv lacked
   `--agent-join` (§4). Caught only because green controls exist.
4. ⚠️ **I referenced `v7l.GOAL_POS_WEIGHT_CAP`, which does not exist** — the cap
   is a *default parameter* of `goal_pos_weight`, not a module constant. Fixed by
   reading it from the function signature rather than retyping `50.0`, so it
   cannot go stale.
5. ⛔ **My mutation harness produced an artifact that parsed as nothing while
   every number in it was correct** (cp1252 in a subprocess reader thread, §4).
6. ⛔ **Three banked JSONs carried this session's scratchpad path**, whose
   directory name is a UUID — the exact class that has blocked a landing before.
   Caught by `tests/test_refcv6_no_session_paths.py` after I registered my files
   in its `OWNED` list. **Fixed at the producer, not only in the artifact**:
   `code/redact.py` strips the scratch root, the worktree root and any
   UUID-shaped segment **by structure**, never by the one literal id — matching
   the id I happen to have would leave the next agent broken the same way and
   would put the forbidden string into the guard itself.
   ⚠️ Two follow-on misses in the same family: baked `C:/Users/...` defaults in
   `${TAC_PY:-…}` (now `${TAC_PY:?}`), and a `tempfile.mkdtemp()` path inside a
   recorded error message — whose **`repr` form doubles every backslash**, so my
   first scrub silently missed it while reporting success.

---

## 10. Deliverable manifest

⛔ Everything below is **staged, not committed**, in
`worktree: C:/Users/Admin/tanitad-wt-tactical` (branch `agent/arch-inf-20260803`,
base `837c308`). Nothing lives only on a pod or only in a scratch dir.

| artifact | where |
|---|---|
| trainer wiring (flags, pin, loss, stamps, dataset) | `repo: stack/scripts/refc_v3_train.py` |
| sidecar reader + `read_speed_max_sidecar_v6` | `repo: stack/tanitad/refs/refcv6_max_speed.py` |
| builder: `sid` join key + `--omit-clip-id` | `repo: stack/scripts/build_refcv6_speed_max_window.py` |
| regression tests (25) | `repo: stack/tests/test_refcv6_tactical_training.py` |
| the two sidecars (`sid`-keyed, clip_id-free) + metas | `repo: …/2026-09-17-refcv6-tactical-training/raw/` |
| bit-identity, refusals, grad reach, label facts, live run | `repo: …/raw/*.json`, `raw/live_run/` |
| run + proof scripts (all paths parameterised) | `repo: …/code/` |

**Integration escalation** — three things need a decision that is not mine:

1. ⛔ **The BEV-token seam (§5).** The map half of the PI's §4 ask is blocked on
   an architecture change (move the BEV encoder into the model forward) *and* a
   PI call on whether the behaviour decoder may backprop into the shared trunk.
2. ⚠️ **The negatives policy for any refcv6 tactical arm.** `measured` (default)
   trains 17/22 tokens; `cot-absence-negative` trains 21/22 but puts 9 on the
   pos_weight cap. The arm's numbers are not comparable across the two.
3. ⚠️ **Two pre-existing collection errors** at `837c308` (§8) — unrelated to
   this work, but they abort a bare `pytest -q` and should be fixed by whoever
   owns those modules.
