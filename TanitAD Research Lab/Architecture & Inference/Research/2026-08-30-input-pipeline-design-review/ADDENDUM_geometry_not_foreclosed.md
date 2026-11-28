# ADDENDUM — ⭐ THE B1 GEOMETRY DECISION IS **NOT** FORECLOSED BY THE RUNNING BUILD

**Written** 2026-08-30 by the Master Mind, **correcting an urgency I relayed to the PI.**

## What the review said, and what I did with it

§4.0 flags the rig shortcut (`v2_subframe: null` on every v6/v7 arm, 73.25 % of the
train cache rig B at 8.89 % masked) and frames the B1 build as *"the last cheap
moment to decide what four consumer lines inherit"* — cheap at build time, a full
retrain afterwards. **I relayed that to the PI as time-critical while the build was
already 44 % complete.**

## ⛔ It is not time-critical, and the reason is a decision already made

**The B1 build is `--codec png`, and a PNG cache is SLICEABLE to another geometry
BIT-EXACTLY, without a rebuild.** Verified in source:

* `stack/scripts/slice_v2_cache.py:19-21` — *"THE LOSSLESS PRECONDITION. The slice is
  bit-exact only for `codec="png"`… REFUSES a lossy source unless `--allow-lossy`."*
* `stack/tanitad/data/v2_dataset.py:320-330` — the sub-frame guard raises only when
  `codec != "png"`; for a PNG cache it sub-frames and returns. And it does so **at
  load time**, keyed on `self.frame != stored_frame_of(d)` — so an arm can train at
  176×624 **directly from the 256×640 cache**, with no separate slice step at all.

⇒ The 176×624 option (and any other centred geometry) remains available **after** the
build, two ways: on-the-fly at load, or a one-time offline slice. Nothing is lost by
letting the build finish.

## ⭐ Why this is worth recording rather than just correcting

**This is DE-C152 paying off.** That entry established the codec is *"load-bearing,
not a speed knob"* — PNG was chosen precisely so a cache could be re-geometried
without a rebuild, at the cost of a slower build (encode is 70 % of build time). The
review's *"last cheap moment"* framing is correct for a **lossy** cache and wrong for
ours, and it is wrong **because a past decision deliberately bought this option.**

⚠️ **The class of my error:** I relayed a scope-correct warning into a scope where its
precondition does not hold — the same family as the `df` / `step_s` / pinhole-FOV
traps, and the same one I logged twice today (MM-C5, MM-C8). The review's own
recommendation — bank per-clip validity masks as a sidecar, which forecloses nothing —
was the right call and is unaffected.

## What remains true from §4.0

The **finding** stands and is important: the rig shortcut is **live**, not a risk —
`v2_subframe: null` confirmed on the 336.5 M config E from the run's own
`config.json`, `/proc` argv of the live trainer, and each checkpoint's adopted eval
geometry. Rig B is the majority in both corpora and carries 8.89 % masked pixels that
every arm has been training on. **That is a real defect to decide about** — it is
simply not one the running build makes more expensive.
