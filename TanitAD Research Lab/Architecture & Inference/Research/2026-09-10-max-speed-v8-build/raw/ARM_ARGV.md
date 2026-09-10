# The E16 arm — argv for a pre-registered single-lever panel

⛔ **NOT LAUNCHED. There is no pod.** This is the handoff, ready for the moment a GPU
exists and the PI says go. Pre-registration:
`Project Steering/PREREG_E16_MAX_SPEED_INPUT_REFCV6.md`.

---

## ⭐ THE ONE THING THAT IS EASY TO GET WRONG, AND IT IS NOT THE FLAG

`--max-speed-input` needs the **v8** blob. So if the ON arm reads v8 and the OFF arm
reads v7.2, the two namespaces differ in **THREE** parsed keys —
`max_speed_input`, `v7_labels`, `eval_labels` — and **the panel is not single-lever.**
That is the `--v2` conflation again: *ten levers on two axes, result
non-attributable.*

⇒ ⛔ **BOTH ARMS READ THE v8 BLOB.** The OFF arm simply does not pass the flag.

⭐ **That is only sound because v8 is a PURE SUPERSET of v7.2, and it is ASSERTED, not
assumed** (`code/verify_v8_is_superset.py`, `raw/verify_v8_is_superset.json`):

| | train | eval |
|---|---|---|
| records | 4,572 | 147 |
| clip sequence identical (parity) | ✅ | ✅ |
| records adding **exactly** `speed_max_input` | **4,572 / 4,572** | **147 / 147** |
| `_provenance` a superset (all old sub-keys byte-equal) | 4,572 | 147 |
| `schema_version` + `vocab` **unchanged** | 9,144 / 9,144 | 294 / 294 |
| ⭐ **untouched-key comparisons EQUAL** | **91,440** | **2,940** |
| keys removed / unexpectedly changed | **0** | **0** |
| **CONTROL** — shared keys per record (vacuity guard) | 22 of 22 | 22 of 22 |

⚠️ **The test is a DESCENDANT check, not a difference check.** *"v8 differs from
v7.2"* is not the question — of course it does. The question is whether it differs
**only** where this build intended. A revert also differs.

⚠️ **CONSEQUENCE FOR THE PANEL, stated plainly:** the E16 OFF twin is **not**
bit-identical to the banked refcv5-v2 run (that one points at v7.2). It must be
**run**, not borrowed. Blob content is proven equivalent for everything the OFF path
reads, but the `v7_labels` manifest md5 in `config.json` will differ, and an arm is
identified by its own artifacts.

---

## THE TWO LINES

**BASE** = refcv5-v2's argv **verbatim** (`config.json` md5
`a03a3ebc46691fa70b5fc55c4efb2ef4`, three byte-identical copies under
`D:\Projects\TanitAD-artifacts\`), with the two label paths swapped to v8 **on both
arms**, per `Project Steering/REFCV6_ARM_FACTS.md:99-101` (*"a refcv6 baseline arm
equal to refcv5-v2's argv exactly"*).

```bash
# ---- OFF twin -------------------------------------------------------------
PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=6 \
python -u stack/scripts/refc_v3_train.py \
  --arm hier --size base \
  --v2-cache /root/data/train \
  --v7-labels /workspace/TanitAD/data/s2_labels_v8.0_train.jsonl.gz \
  --eval-cache /root/data/eval \
  --eval-labels /workspace/TanitAD/data/s2_labels_v8.0_eval.jsonl.gz \
  --eval-every 500 --eval-batches 8 \
  --image-hw 256 640 \
  --steps 40284 --batch 20 --workers 6 \
  --prefetch-factor 1 --v2-lru 24 \
  --lr 1e-4 --warmup 2000 --seed 0 \
  --log-every 50 --save-every 500 \
  --u8-batches \
  --out /workspace/experiments/refcv6-e16-OFF \
  --nav-from-v7 \
  --ego-state-inject --ego-dropout 0.5 \
  --anchors /workspace/experiments/refcv6-anchors/anchors.pt \
  --n-anchors 117 --anchor-v0-conditioned --anchor-control-units alat \
  --sel-accel-max 2.0 \
  --sampler ddim \
  --w-u0 0.5 \
  --sel-refined --sel-score-emitted \
  --goal-str --tac-goal-tok-head \
  --agents off

# ---- ON arm: IDENTICAL, plus one token ------------------------------------
#   … --agents off \
#   --max-speed-input
#   (and --out /workspace/experiments/refcv6-e16-ON)
```

⚠️ **`--max-speed-mode` is NOT passed.** It defaults to `quantized`, is inert without
the flag (the parser refuses it alone), and passing it would add a second moved key
for no gain. It is **recorded** in `config.json`, not varied.

⚠️ **`--anchors` must be the SAME FILE for both arms.** refcv5-v2's anchors lived
inside its own `--out` directory; if each arm builds its own, the anchor vocabulary
is a second lever. Build once, point both arms at it.

---

## THE ASSERTION — run it before launching, not after

```bash
python "TanitAD Research Lab/Architecture & Inference/Research/2026-09-10-max-speed-v8-build/code/assert_single_lever.py" \
  --argv-file ".../raw/arm_argv.json" --expect max_speed_input --ignore out
```

**MEASURED (ours)**, `raw/assert_single_lever.json`:

```
n_shared_keys                            109
n_equal                                  107
changed                                  {"max_speed_input": {"base": false, "arm": true}}
EXEMPTED_run_identity_keys               {"out": {...OFF, ...ON}}
STALE_exemptions_that_did_not_differ     []
CHECK_not_vacuous                        true
CHECK_only_the_expected_key_changed      true
CHECK_the_lever_actually_moved           true
CHECK_no_stale_exemptions                true
VERDICT_single_lever                     true

PASS — exactly one parsed key moved: max_speed_input False -> True (107 of 109 identical)
```

⭐ **Four checks, and three of them exist because the obvious one is not enough:**

* `CHECK_only_the_expected_key_changed` — the obvious one.
* ⛔ `CHECK_the_lever_actually_moved` — *"no unexpected differences"* is satisfied by
  **no differences at all**. Two identical argv lines would pass the obvious check and
  the panel would be measuring nothing. The lever must be shown to have MOVED.
* ⛔ `CHECK_not_vacuous` — a diff over two nearly-empty namespaces reports "one
  difference" and means nothing. ≥ 20 shared keys required; this panel has 109.
* ⛔ `CHECK_no_stale_exemptions` — an `--ignore` entry that did **not** differ hides
  nothing today and would silently absolve a real lever tomorrow.

⛔ **`out` is exempted BY NAME and the exemption is printed in the report.** Two arms
cannot write to one directory, so `out` must differ; the default ignore-list is
**empty** so every exemption is deliberate and visible. A silent allowance is how a
second lever hides.

---

## ⛔ BEFORE ANY RESULT IS QUOTED FROM THIS PANEL

1. **The stamp travels with the number.** `config.json` carries
   `speed_max_derivation` naming the source field, the window `[t0+2 s, +6 s]` and the
   word `oracle`. A result table without it is **inadmissible**, not merely incomplete.
   The trainer refuses to start without it (proven by mutation).
2. **A separated CI is necessary and NOT sufficient.** The tiny-rig false-positive rate
   for `separated` is **14.3 %**. The **REPLICATE** arm is required, plus **WITHHOLD**
   and **SHUFFLE** — see the pre-registration §3.
3. **All four metric families, per family, never pooled.** ADE alone is one row of four.
4. **The residual defect is pre-registered, not discovered later:** 38.1 % of the corpus
   sits at ≤ 30 km/h because snapping UP from a stopped ego reports the lowest limit, so
   the channel can teach *"slow ego ⇒ low limit"*. Any longitudinal win must be shown not
   to be that.
