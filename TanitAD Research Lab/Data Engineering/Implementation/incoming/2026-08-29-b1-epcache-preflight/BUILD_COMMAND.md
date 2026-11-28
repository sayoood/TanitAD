# B1 epcache — the exact build command, and what must be true before it runs

**Date** 2026-08-30 · **Owner** DataFlyWheel · For the Master Mind to arm behind
the τ-ramp. Flags verified against `v2_compressed.py build --help`; nothing here
is composed from memory.

---

## ⛔ STOP — a leak the command line would have caught, and the plan would not

**MEASURED 2026-08-30: 6 of the corpus's 4,719 clips are in the DEPLOYED VAL40**
(matched by `sha256(clip_id)` against
`stack/tanitad/data/deployed_val40_clip_digests.json`, corpus key
`physicalai-val-0c5f7dac3b11`). Training any of the four consumers on them
measures the model **partly on its own training data**, and val40 is the split
that produces THE published open-loop statistic.

* This is a **known, already-sanctioned** case: the builder's own `--help` names
  *"the Alpamayo 4,472 build (6 deployed-val episodes)"* — the same six.
* ⇒ the build **must** carry `--corpus-role train --exclude-parity-overlap`.
  With `--corpus-role train` and no repair flag the builder **refuses** (correct,
  but it burns the window). ⛔ **Never reach for `--sanctioned-audit` here** —
  that stamps `decision_grade False` and is for audits, not for getting a
  training build past a real leak.
* Effect: **4,719 − 6 = 4,713** clips enter the cache; the exclusion is recorded
  in the build's `_geometry.json`.
* Recorded in `MANIFEST.json` → `exclusions.parity_deployed_val40` (digests only,
  🔒 the ids are gated-confidential). New MANIFEST sha `5feda062a72a32ad`.

⚠️ **The validation gate's "exclusions: 0" was correct and INCOMPLETE.** It is a
DATA-QUALITY verdict (no corrupt clip), not a parity verdict. Both now sit in the
ledger, labelled — otherwise a reader concludes the corpus needs no exclusions.

---

## Where it should build: THOR, not the dev box — on transfer arithmetic

| plan | bytes moved |
|---|---|
| **build on Thor** (pull the 61.6 GB camera from the private dataset, build locally) | **61.6 GB** |
| build on the dev box, ship the cache | 161 GB up **+** 161 GB down = **322 GB** |

The dev box has every input already and its preflight passes, so it *can* build —
but shipping the result costs 5× the transfer. ⭐ **This is also exactly why the
PI wanted the camera INSIDE the dataset: it is what makes Thor able to build.**

---

## ⛔ CORRECTION to the suggested Thor paths — the camera bank placement

The suggested `<root>/camera/camera_front_wide_120fov/` is **exactly the layout
that fails**, and it fails silently. `_physicalai_root_of` (`physicalai.py:166`)
walks a clip path's parents for a directory literally named **`r0`**; without one
it returns `None`, and then intrinsics AND extrinsics both fall back with nothing
raised — `extrinsics_for_clip` → `None`, callers "treat the mount as level".

⇒ **`/home/nvidia/data/physicalai-b1/r0/camera_front_wide_120fov/`** (note `r0/`).
Everything else in the suggestion is good; output and root adopted as proposed.

```
ROOT   /home/nvidia/data/physicalai-b1
OUT    /home/nvidia/data/physicalai-b1-w120-256x640cyl
STACK  /home/nvidia/TanitAD/stack
PY     /home/nvidia/venvs/tanitad-train/bin/python
```

## The command

```bash
# 0. PREREQUISITES on Thor — run the two staged scripts (now portable):
python build_intr_csv.py --root "$ROOT" \
    --index "$BUNDLE/index/clip_to_chunk.parquet" \
    --clips "$BUNDLE/labels/s2_labels_v7.jsonl" \
    --cy-check "$BUNDLE/index/front_wide_cy.parquet"

python prepare_b1_root.py --root "$ROOT" \
    --camera-src "$BUNDLE/camera" \
    --index "$BUNDLE/index/clip_to_chunk.parquet" \
    --clips "$BUNDLE/labels/s2_labels_v7.jsonl" \
    --calib-src "$ROOT/calibration" \
    --intrinsics-csv "$ROOT/calibration/physicalai_front_wide_intrinsics.csv"
# prepare_b1_root ATTACHES the camera bank (symlink on POSIX) — 61.6 GB is never
# copied — and asserts the r0_selection covers EVERY clip rather than silently
# writing a short one.

# 1. GATE — must exit 0. Nothing starts if it does not.
python preflight_epcache_build.py --root "$ROOT" --sample 120

# 2. BUILD (PNG per the PI: "stick to the losless pngs")
PYTHONPATH="$STACK" python "$STACK/scripts/v2_compressed.py" build \
  --sel  "$ROOT/r0/r0_selection.parquet" \
  --root "$ROOT" \
  --out  "$OUT" \
  --codec png \
  --height 256 --width 640 --f-ref 305.5774907364391 \
  --projection-mode cylindrical \
  --corpus-role train \
  --exclude-parity-overlap
```

**Why each geometry flag is what it is** — `--height/--width/--f-ref` reproduce
`tanitad.data.calib.PHYSICALAI_WIDE120_256x640` exactly (HFOV 120.00°). `--f-ref`
is given explicitly rather than `--hfov 120` so the value is *pinned* rather than
re-solved. ⚠️ `--width 640` is not cosmetic: the builder's own help warns that
**a square frame clamps at the sensor and silently zooms**.

⚠️ **Do NOT add `--require-fully-observed`** on this frame. It aborts unless every
sampled clip of BOTH rigs fully observes the requested frame, and the 120° frame
is deliberately not fully observed on rig A (that is what the mask is for). It is
the flag for the *rig-clean* 176×624 frame, which the PNG cache can be sliced to
later without a rebuild — the reason the codec decision matters.

## ⚠️ Wall-clock re-derived for THOR'S 14 CORES — and why I will not quote it as measured

My 3.4–4.6 h was measured on a **24-core** box. Thor has **14**. Naively rescaling
is the `df`/scope error again, so here is the model **and its uncertainty**, and
then the cheap way to replace it with a fact.

**The structure that decides it: PNG encode is 70 % of the work and is
effectively SINGLE-THREADED per clip** (a Python loop over `tvio.encode_png`,
14.79 s of the 20.98 s). Decode is the part that wants 4 threads. So the build
scales with **worker COUNT**, not with threads per worker:

| config on 14 cores | note | projected |
|---|---|---|
| 6 workers × 4 decode threads | **24 threads on 14 cores — oversubscribed**, the trap the trainer's own comment warns about | worse than the table below |
| **3 workers × 4 threads** (keeps W×T ≤ cores) | starves the single-threaded encode | **~9.2 h** |
| **6–7 workers × 2 threads** ⭐ | encode-parallel, decode slightly slower | **~5–6 h** |

⇒ **suggest `PAI_DECODE_THREADS=2` with 6 workers**, NOT 6×4. The intuition
"6 workers held the dev box, so use 6 workers" imports the wrong half of the
config — the thread count is what has to shrink.

⛔ **This is a MODEL, not a measurement — do not put it in the morning report as
measured.** The builder ships `measure` for exactly this:

```bash
python "$STACK/scripts/v2_compressed.py" measure --root "$ROOT" --n 12 \
    --codec png --height 256 --width 640 --f-ref 305.5774907364391 \
    --projection-mode cylindrical
```

12 clips gives Thor's own per-clip cost in a few minutes. **Run it before the
build and quote THAT number**; my projection is a prior, and priors have been
wrong twice tonight already.

**Artifact** ~161 GB against 657 GB free — comfortable. CPU-bound, GPU-free.

---

## ⚠️ Honest gap in what I am handing over

`prepare_b1_root.py` and `build_intr_csv.py` have **dev-box paths hardcoded** and
are not Thor-runnable as written — they need their roots parameterised, and I
have not been able to test them on Thor. `preflight_epcache_build.py` already
takes `--root` and is portable. **Prerequisite (0) above is therefore a manual
step on Thor, not a script I have verified there** — I would rather say so than
hand over a script that fails at 07:45.
