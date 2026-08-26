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

## 3. ⛔ What is actually missing — one thing

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

⚠️ **NOT yet verified — and these are where the cost actually lives:**
- that `pose_dim` is 3, matching T1's `[B, 3]` expectation;
- that a v7 checkpoint rebuilds the full stack outside the trainer;
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

⚠️ **This is a bigger hole than the action-conditioning result.** That thread is
closed and well-powered; this one is open and never attempted. ⇒ **Recommended as
the next work item after the crossed cell reads**, ahead of any further T0 probe.
