# T1 on the v7 arms: scoped, and it is a loader swap

**Written** 2026-08-26 · **Author** Master Mind
**Why now:** I have written *"T1 has never been run on any v7 arm, so no capability
claim is available"* three times today without once checking **why**. That is the
shape of a stale blocker, so this is the check.

⭐ **The answer: it is not blocked by architecture. It is a loader adapter.**

---

## 1. What T1 actually requires of a model

`taniteval/tools/t1_eval.py` — read from source, the rollout is two calls:

```python
z_hat = model.predictor(win_s, win_a)[1]        # t1_eval.py:830
d     = step_readout(win_s[:, -1], z_hat)       # t1_eval.py:831   -> [B, pose_dim]
```

That is the whole model interface. Everything else in the file is corpus plumbing,
lead-block alignment and reporting.

## 2. ⭐ The v7 stack already implements it — the same call, verbatim

| | flagship path (T1 today) | v6/v7 stack |
|---|---|---|
| predictor | `model.predictor(win_s, win_a)[1]` | **identical** — this is the call every probe in this campaign made |
| step readout | `step_readout(win_s[:, -1], z_hat)` | `self.step_readout_op(win_s[:, -1], z_hat)` — **`v6.py:5124`, byte-for-byte** |
| readout class | `UnicycleStepReadout` | `StepDisplacementReadout(cfg.d_op)` — `v6.py:4577`; `forward(z_t, z_next) -> [B, pose_dim]` |
| shared helper | `rollout_decode(predictor, states, actions, step_readout, k)` | **the same function**, `metric_dynamics.py:220` |

⇒ **The rollout math, the predictor signature and the step-readout signature all
match.** T1 was never architecturally incompatible with v7.

## 2b. ⭐⭐ EXECUTED 19:15 — the model side needs NO ADAPTER AT ALL

**`w.stack.step_readout_op` exists.** The probe wrapper returns the trunk, and
`load_trunk_auto` builds it as `trunk, V6Grounding(trunk.stack), step` — so the full
v6 stack was always reachable **one attribute deeper** than §3 assumed. Run on
`rdw8p30k`, the exact two calls `t1_eval.py:830-831` makes:

```
z_hat = predictor(win_s, win_a)[1]      ->  (2, 2048)
d     = step_readout(z_t, z_hat)        ->  (2, 3)      ✓ T1 expects [B, 3]
step_readout_op.pose_dim = 3
```

⇒ ⭐ **Both of §4's "NOT yet verified" model-side risks are now verified TRUE, and
neither required writing any code.** `pose_dim` is 3; the stack rebuilds from a v7
checkpoint outside the trainer (the probe loader has been doing it all campaign).

⚠️ **§3 below overstated the work and is superseded on the model side.** It is kept
because the reasoning that led there — reading `load_arm`, seeing only `predictor`
and `window`, and concluding an adapter was needed — is exactly the
**absence-found-at-one-location** error `CLAUDE.md` warns about: I probed the
wrapper, not the object it wraps. **The remaining cost is entirely §4's corpus
plumbing.**

---

## 3. ⛔ What is actually missing — one thing *(SUPERSEDED by §2b: the model side needs nothing)*

**`v7tiny_g2.load_arm` returns a PROBE TRUNK, not the full stack.** It calls
`tanitad.eval.v6_probe_trunk.load_trunk_auto`, which yields an object exposing
`predictor` and `window` — deliberately, because every probe this campaign ran
needed exactly those two. **Verified by execution:** `hasattr(w, 'step_readout_op')`
→ **False**; `predictor` → True.

⇒ **The adapter is: build the FULL v6 stack from a v7 checkpoint and hand T1 its
`predictor_op` + `step_readout_op`.** The trainer already does this on every launch,
so the construction path exists and is exercised daily.

## 4. Scope, honestly split

**Verified by reading source or executing:**
- T1's model interface is the two calls above (source).
- v6 makes the identical step-readout call (source, `v6.py:5124`).
- Both share `rollout_decode` (source).
- The probe-trunk loader does **not** expose `step_readout_op` (executed).
- v7's `predictor(z, a)[1]` returns the expected latent (executed, all day).

✅ **VERIFIED 19:15 by execution (§2b), both without writing code:**
- `pose_dim` **is 3**, matching T1's `[B, 3]`;
- a v7 checkpoint **does** rebuild the full stack outside the trainer.

⚠️ **STILL NOT verified — and this is now where ALL the cost lives:**
- the corpus / `--v2-val-cache` / lead-block plumbing for the v7 geometry
  (256×640 cylindrical) — T1's defaults are w120 flagship;
- whether `--grounding-readout` or the separate `--head` path is the right mode.

⇒ **Estimate: a bounded integration task, not an architecture change.** The
irreducible risk is the corpus plumbing, not the model.

## 5. Why it matters more than any experiment currently queued

**Every number this programme has produced is T0** — a world-model diagnostic. The
eval doctrine is explicit that **T1 is the primary tier for any capability claim**,
and that T0 is *never* driving performance. So:

⛔ **TanitAD currently has no capability claim of any kind**, and cannot acquire one
until this adapter exists — regardless of how the crossed cell reads.

⚠️ **This is a bigger hole than the action-conditioning result.** ⛔ *(That thread
was described here as "closed" — WITHDRAWN 2026-08-26: closing it was not my call,
and the evidence supports only "not found under the conditions tested".)* ⇒ **Recommended as
the next work item after the crossed cell reads**, ahead of any further T0 probe.


---

## 6. ⭐ EXECUTED 19:20 — the blockers are NAMED, and neither is corpus plumbing

Rather than estimate, I ran `t1_eval.py` against a v7 checkpoint and recorded each
failure in turn. **Three refusals, all clean, and the tool refused correctly every
time** (it never ran on a mismatched model):

| # | refusal | fix |
|---|---|---|
| 1 | `rollout mode needs --dump-dir` | pass one |
| 2 | `rollout mode needs --head or --grounding-readout` | `--grounding-readout` — the mode that uses the model's OWN step readout, which §2b verified exists |
| 3 | ⛔ **`ckpt has no 'model'+'grounding' keys — needs a flagship trunk checkpoint (v1/v4/v5f shape). Keys: ['config','opt','stack','step']`** | **the real one** |

### The two actual blockers

1. ⭐ **CHECKPOINT SHAPE (small).** `t1_eval` expects the flagship `'model'` +
   `'grounding'` layout; a v7 ckpt ships **`'stack'`**. But `load_trunk_auto`
   *already* loads exactly that shape — it is what every probe in this campaign
   used. ⇒ **Route a `'stack'`-keyed checkpoint through `load_trunk_auto` and hand
   T1 `stack.predictor_op` + `stack.step_readout_op`.** ~15 lines in the load path;
   §2b proved the resulting objects satisfy T1's interface.

2. ⛔ **GEOMETRY (the one I under-weighted).** The run printed
   `[geometry] t1_eval: DEPLOYED (unchanged) - 256x256px, f_ref 266.00, **pinhole**`
   while every v7 arm is **256×640, f_ref 305.58, cylindrical**. ⇒ **The geometry
   must follow the CHECKPOINT, not t1_eval's default.** ⚠️ This is the
   `CLAUDE.md` optics trap in eval form: *a correct formula quoted outside its
   projection*. Left unfixed it would produce T1 numbers that look valid and are
   computed in the wrong frame — **worse than a crash.**

### What this changes

⇒ **"Corpus plumbing, unknown cost" was wrong twice over.** The corpus was never the
blocker; the checkpoint adapter is small; and the geometry is the item that actually
needs care. ⚠️ **Estimating it twice produced two wrong answers; running it produced
the list in ten minutes.**

⛔ **I have NOT edited `t1_eval.py`.** It is the programme's primary eval instrument
and the doctrine's T1 authority; a change to its load path and geometry handling
should be a deliberate, reviewed edit rather than something appended to a probe
session. **The scoping is done and the work is ready to start.**


---

## 7. ⭐ FINAL SCOPE, 19:25 — one blocker, and it is ~15 lines

⚠️ **Third correction to this document in one afternoon, each time by executing
instead of estimating.** The trend is one-directional: every estimate was
pessimistic, and every execution shrank the job.

### Blocker 2 (geometry) — ⭐ FREE. The flags already exist.

`t1_eval --help` lists them; my earlier grep missed them because a shared helper
adds them, not a literal `add_argument("--frame…` in this file:

```
--frame-h 256  --frame-w 640  --projection cylindrical  --frame-hfov 120
```

⇒ **No code change.** The default was the deployed pinhole frame; v7's cylindrical
frame is expressible on the command line. ⚠️ **It still must be passed explicitly —
the default would silently compute T1 in the wrong projection**, which is the
failure mode worth more care than the crash.

### Blocker 1 (checkpoint) — the only real work, and it CANNOT be shimmed

`load_ext_trunk` routes through `eval_flagship_v4.load_v1_from_ck`, which **builds a
flagship architecture and loads STRICT**. A v7 `'stack'` checkpoint is **not a
different layout of the same model — it is a different model.** ⇒ **No checkpoint
conversion bridges this. The loader must branch.**

The branch is small because the destination already exists:

```python
# in load_ext_trunk, before the flagship path
if isinstance(ck, dict) and "stack" in ck:
    from tanitad.eval.v6_probe_trunk import load_trunk_auto
    trunk, _grounding, step = load_trunk_auto(ck, device, ckpt_path=a.ckpt)
    return trunk, trunk.stack, step     # predictor_op + step_readout_op live here
```

§2b verified by execution that the resulting objects satisfy T1's interface exactly
(`predictor(win_s, win_a)[1] -> [B, 2048]`, `step_readout(z_t, z_hat) -> [B, 3]`,
`pose_dim == 3`).

### ⇒ The whole job

| item | cost |
|---|---|
| geometry | **0** — pass four existing flags |
| loader branch | **~15 lines** in `t1_eval.load_ext_trunk` |
| verify T1's `grounding` consumer accepts `trunk.stack` | **unknown, small** — the one thing still unexecuted |

⛔ **Still not edited.** `t1_eval.py` is the doctrine's T1 authority; the branch is
trivial but it belongs in a reviewed edit, and the third bullet above should be
executed first. **The scoping is finished and the estimate has stopped moving.**
