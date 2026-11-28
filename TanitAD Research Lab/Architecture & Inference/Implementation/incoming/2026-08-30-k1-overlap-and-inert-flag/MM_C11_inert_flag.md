# MM-C11 — 2026-08-30 — ⛔ A FLAG THAT EXISTS, VALIDATES ITSELF, AND DOES NOTHING — and the fix for our "new" finding was built a week ago and never wired

**Class:** `existence-is-not-function` — a **new** member of the MM-C5 family. MM-C5
taught me to verify a flag EXISTS before writing a chain around it. This one exists,
passes its own consistency guard, and is **inert on the trainer we actually use.**

## What I told the PI, and what is true

I reported that option (a) — the single-frame encoder — *"needs no implementation:
`--newest-frame-only` already exists (`train_v6_staged.py:6400`) and `:4097` REFUSES
it unless `--in-channels 3`, so the inconsistent combination cannot be run by
accident."* ⛔ **Both halves of that are true and the conclusion is wrong.**

The arm crashed:

```
ValueError: REF-B windowing: episode 0 has 9 frame channels; encoder expects 3
  train_flagship4b.py:117 -> refb_train.py:177
```

**Every occurrence of `newest_frame_only` in `train_v6_staged.py`:**

| line | what it does |
|---|---|
| `:1438` | a docstring mention |
| `:4097-4098` | the consistency guard (refuses unless `--in-channels 3`) |
| `:6400` | the `add_argument` registration |

⇒ **It is registered, documented and guarded — and never passed to anything.** The
encoder gets built for 3 channels, the data path keeps serving 9, and the failure
surfaces from a *different module* as a channel mismatch that reads like a config
error rather than an unimplemented feature.

⚠️ **The guard made it worse, not better.** `:4097` validates the flag against
`--in-channels`, which *feels* like the feature being checked. A flag that validates
itself signals "wired". **A consistency guard is not an implementation, and it can
impersonate one.**

## ⭐⭐ THE LARGER FINDING — the fix was built on 2026-08-23 and never reached us

The feature IS fully implemented, in the data layer:

* `v2_dataset.py:112,128-129` — `newest_only`, *"emit ONLY the newest frame of each
  stack"* (H-RANK-8, **2026-08-23**)
* `v2_dataset.py:288,295,355` — carried and used by `LazyV2Episode`
* `v2_dataset.py:487,523` — `build_v2_providers(..., newest_frame_only=...)`
* `train_v58f_unicycle_head.py:343-348` — **passes it correctly**

**`train_v6_staged.py` — the v6/v7 trainer, the one every current arm uses — does
not.** Its train path goes `train_flagship4b.FeatureWindowDataset` →
`refb_train` windowing, which never sees the flag.

⭐ **And read the reason the fix was written, verbatim from `v2_dataset.py:133-134`:**

> *"WHY: with a 3-frame stack consecutive latents share 2/3 of their input, and
> lag-1 dz autocorrelation measured NEGATIVE (−0.075)"*

**That is the same 67 % overlap the data-efficiency review reported as a new finding
on 2026-08-30.** It was known on **2026-08-23**, a fix was implemented and tested,
it was wired into one trainer — and it never reached the production trainer. ⇒ **This
is the stranded-artifact failure the operating standard exists to prevent**
(*"an artifact on one disk or in one agent's context is NOT done"*), in its most
expensive form: not stranded on a disk, but stranded **one function call short of the
consumer**, with a self-validating flag advertising that it had arrived.

## Consequences

1. **MM-E8's single-frame arms cannot run** until the pass-through lands. The
   3-frame arms (A0, A0b) completed and are valid.
2. ⛔ **Correction owed to the PI**: (a) is **not** free. It is a small fix — one
   argument through the v6 train-provider path — but it is a code change, not a flag.
3. ⚠️ **The overlap finding's provenance changes.** It is not new; it is **re-derived
   independently a week later**, which *strengthens* it (two methods, same number)
   while removing any novelty claim.
4. **Owner:** the TrainingFlyWheel, who holds `train_v6_staged.py` this session for
   the nav wiring. ⛔ I did not edit it — a concurrent edit to the file the whole
   programme is waiting on, during a mount outage, is exactly how a half-written
   `v6.py` lands.

## ⭐ The rule

**MM-C5 said: verify the flag EXISTS. That is now insufficient — verify the flag
DOES SOMETHING.** The cheap check is a grep for the flag's *use*, not its
*declaration*: a symbol that appears only in `add_argument`, a docstring and a guard
is **inert by inspection**. And the sharper habit: when a feature is implemented in a
library, check *which callers pass it* — `newest_frame_only` appearing in
`v2_dataset.py` proved the capability existed and said nothing about whether our
trainer reached it.
