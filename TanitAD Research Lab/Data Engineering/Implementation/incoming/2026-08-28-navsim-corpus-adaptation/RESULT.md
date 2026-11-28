# NavSim warmup EVAL corpus — BUILT and validated (commission complete)

**Date** 2026-08-29 · **Owner** DataFlyWheel · **Commission** PI 2026-08-28 via
Master Mind: *"adapt the data to what we need … generate an eval data set in
production grade"* · **Status** BUILT · content-validated · adapter spec below

| | |
|---|---|
| corpus id | `89f26a994509c379` (sha256-16 over sorted scene tokens) |
| parity | ⛔ **NON-PARITY** — NavSim is a separate corpus. `physicalai-train-e438721ae894` / skip-hash `f09e44db` untouched. |
| scenes | **204 / 204 built, 0 failed** (7 log groups, 5 distinct rigs) |
| frames | 816 (4 history frames/scene at 2 Hz; `num_future_frames = 0`) |
| ego rows | 816 (schema-distinct sidecar) |
| build cost | **1.1 min**, dev box, no GPU |

## What was built

Each NavSim scene becomes frames in **the TanitAD training geometry**, produced
by the `CanonicalFrame` object from `tanitad.data.calib` — never a re-declared
literal — stitched from **cam_l0 + cam_f0 + cam_r0** with each camera's own
measured intrinsics, extrinsics and 5-param distortion.

Two variants ship from ONE bank of pixels (no duplication; the crop is the
exact `[40:216, 8:632]` slice `calib.py` documents):

| variant | frame | observed | use |
|---|---|---|---|
| `rig_clean` **(default)** | `PHYSICALAI_RIG_CLEAN_176x624` | **100.0000 %** — measured over all 204 scenes × 5 rigs | every capability claim |
| `wide` | `PHYSICALAI_WIDE120_256x640` | **89.29 %** mean / 88.14 % worst-case | only with the shipped mask, and only with the observed fraction stamped on the number |

## ⚠️ The finding that shaped the build, and a correction to my own brief

`FRAME_DECISION.md` argued for the 3-camera stitch from a **horizontal**
coverage analysis and concluded the 120° frame would be covered. Horizontally
that holds (camera share measured: l0 24.1 % · f0 41.0 % · r0 24.2 %). **The
vertical axis was not checked, and it does not hold:**

```
256x640 frame needs VFOV 45.3 deg      NavSim cameras have 38.5 deg
                                       -> 6.8 deg deficit
```

No stitch fixes it — all NavSim cameras share that VFOV. The hole is a
**scalloped band along the top and bottom edges**, zero at each camera's
boresight column and worst (14.5 % of a column) at the wedge boundaries, because
a cylindrical column at azimuth φ off a pinhole boresight needs `1/cos φ` more
vertical extent. Centre column: fully observed, rows 0–255.

⭐ **The resolution needed no new geometry: the program's existing
`PHYSICALAI_RIG_CLEAN_176x624` — defined to be unobserved-free on PhysicalAI's
two rigs — is 100.0000 % observed on NavSim's five.** The frame built to solve
this problem once solves it again on a corpus it was never designed for. The
largest clean centred sub-rectangle is in fact 198×640 (120° × 35.9°), but
176×624 is an established program frame with a registry pedigree, so it is the
default for comparability.

## Q: are the source jpgs already rectified? — settled by EXPERIMENT

Applying a distortion model to already-undistorted images would corrupt every
frame while looking plausible. Decided by measuring seam discontinuity
(mean |dI/dx| across stitch seams) both ways, **with a control**: the same
metric at non-seam columns, which must read the image's own local roughness and
therefore must NOT move between arms.

| model | seam | non-seam (control) | ratio |
|---|---|---|---|
| **WITH distortion (as built)** | **5.95** | 6.14 | **0.97** — seams indistinguishable from ordinary image texture |
| WITHOUT (assume pre-rectified) | 6.85 | 6.10 | 1.12 — a measurable step at every seam |

The control reads 6.14 / 6.10 — unmoved, as it must be — so the metric is
measuring the seam and not a global change. ⇒ **the jpgs carry distortion; the
model as built is correct.**

## Ego sidecar (deliverable 3, built)

`navsim_ego_sidecar.parquet`, 816 rows × 13 cols, schema in `EGO_SIDECAR.md`.
Measured: 7 logs, 204 scenes, `driving_command` distribution
**{straight 685, left 86, right 45}** — note the skew; any per-command
breakdown reports its n.

⛔ **The frame stream and the sidecar are separate files with different schemas
by design.** Ego kinematics are evaluator/label-side only (vision-only at
inference, PI 2026-08-03); `driving_command` is an **oracle goal input only** —
admissible if an arm declares it, never a hidden channel, and never derived from
a situation classifier's output.

## Grouping key (deliverable 1, done earlier)

`log_name`, Outcome A — 204 scenes / **7 groups**, 0 leakage.
⚠️ n_groups = 7 is small: an episode-cluster bootstrap over 7 clusters is wide.
**Report NavSim intervals over log clusters and say n=7**, or the interval will
look tighter than the evidence supports.

## Deliverable manifest

| artifact | where |
|---|---|
| `build_navsim_eval.py`, `validate_navsim.py`, `navsim_loader.py` | repo `code/` (this package) |
| `BUILD.json`, `clean_subframe.json`, `frames_provenance.parquet` (per-scene sha256 + observed frac) | repo `raw/` |
| frame bank (417 MB, 204×`.npy` + per-rig src maps), `navsim_ego_sidecar.parquet`, `valid_mask_intersection.npy` | dev box `C:/Users/Admin/tanitad-wt/_s2build/navsim/corpus` — not in git (size). ⭐ **BIT-REPRODUCIBLE, see below — so it is not a stranded artifact.** |

### ⭐ The 417 MB bank does not need shipping: it is bit-reproducible (VERIFIED)

"An artifact on one disk is NOT done" (operating standard 3) normally forces a
ship. Here the stronger guarantee was available and was **measured**, not assumed:
the corpus was rebuilt from the committed `build_navsim_eval.py` and compared
scene by scene against the first build's banked hashes.

```
scenes 204/204, same token set
per-scene frame-bank sha256 IDENTICAL   204/204
observed_frac identical                 True
mean_px identical                       True
rebuild cost                            1.2 min, no GPU
```

⇒ the repo holds everything needed to regenerate the bank exactly, and
`raw/frames_provenance.parquet` carries the per-scene sha256 so **any rebuild can
be VERIFIED rather than trusted**. That is a better guarantee than a shipped
blob: bytes on a disk can rot silently, whereas a hash-checked deterministic
rebuild cannot diverge without saying so. *(The build was written to be
deterministic by construction — no RNG, no timestamps, no thread-order
dependence in the output.)*
| `FRAME_DECISION.md`, `EGO_SIDECAR.md`, `ADAPTER_CHANGES.md` | this package |

## Remaining

`ADAPTER_CHANGES.md` lists the `taniteval` changes needed to score checkpoints
on this corpus. The corpus itself is complete and validated; nothing about it
blocks on the adapter.
