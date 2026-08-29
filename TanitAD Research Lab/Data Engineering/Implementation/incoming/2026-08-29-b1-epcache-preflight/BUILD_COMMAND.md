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

## The command

```bash
# 0. PREREQUISITES on Thor (each must be true; none is implied by the others)
#    a. dataset pulled:  camera/*.mp4 (4,719), labels/, index/, egomotion/
#    b. calibration present: calibration/camera_intrinsics + sensor_extrinsics
#       chunk parquets, AND calibration/physicalai_front_wide_intrinsics.csv
#    c. layout: the camera bank MUST sit under an `r0/` parent, and
#       <root>/r0/r0_selection.parquet must list all 4,719 clips with chunk

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

**Expected**: ~3.4–4.6 h at 6–8 workers, CPU-bound, GPU-free (measured:
decode+remap 6.19 s + PNG encode 14.79 s = 20.98 s/clip). Artifact ~161 GB.

---

## ⚠️ Honest gap in what I am handing over

`prepare_b1_root.py` and `build_intr_csv.py` have **dev-box paths hardcoded** and
are not Thor-runnable as written — they need their roots parameterised, and I
have not been able to test them on Thor. `preflight_epcache_build.py` already
takes `--root` and is portable. **Prerequisite (0) above is therefore a manual
step on Thor, not a script I have verified there** — I would rather say so than
hand over a script that fails at 07:45.
