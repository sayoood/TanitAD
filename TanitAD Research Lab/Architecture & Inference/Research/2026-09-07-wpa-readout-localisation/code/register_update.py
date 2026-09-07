# -*- coding: utf-8 -*-
"""Add WP-A's claim to GOALS_AND_CLAIMS.md, and mark E-DEC-29's open NEXT answered.

⛔ Never writes to G: in place: authors a full copy on LOCAL disk, then ONE copy in.
⛔ Compare-and-swap on the source file's blob hash, so a sibling's edit between our
   read and our write FAILS rather than being clobbered.
⚠️ The idempotence guard is a SUBSTRING of the inserted text, contains NO newline,
   and is verified BY COUNT (0 before, exactly 2 after).
"""
import hashlib
import os
import shutil
import subprocess
import sys

REG = (r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
       r"\Project Steering\GOALS_AND_CLAIMS.md")
TMP = r"C:\Users\Admin\wpa-readout\_register_new.md"
GUARD = "E-READOUT-CEILING-1"

ROW = (
 "| E-READOUT-CEILING-1 | **How much BEV agent localisation does the 16x40 token grid carry that a "
 "POOLED readout does not? (WP-A, before refcv5c is designed against it)** | \u2b50 **MEASURED \u2014 "
 "THE READOUT IS A CEILING, NOT TODAY'S BOTTLENECK.** Under a **perfect front-end** pooling costs a "
 "lot; on the **real trunk** there is nothing to lose at ANY resolution. | "
 "`\u2026/Architecture & Inference/Research/2026-09-07-wpa-readout-localisation/RESULT.md`, "
 "`raw/oracle_s0*.json`, `raw/indexed_k8.json`, `raw/bearing_k8.json`, `raw/geometry.json`. "
 "Dev-box RTX 4060 only, **0 A40**. B1 EVAL 139 clips / **13,223 rows**, episode-disjoint "
 "84 fit / 25 inner-val / 30 test; target `bev_raster` [120,64]; \u26d4 occupancy **1.62 % on test "
 "\u21d2 an ALL-ZERO predictor scores 98.38 % accuracy**, so no headline number here is an accuracy. "
 "**(a) ORACLE ladder** (\u26d4 the feature map is BUILT FROM THE TARGET \u2014 a readout-ceiling "
 "measurement, NOT a perception result; every arm `avg_pool(k) \u2192 unpool to 16x40 \u2192 the SAME "
 "82.6 k-param head`, so capacity is matched and only the pool differs \u2014 the exact confound "
 "**E-DEC-25** flagged in its own row ladder): AP **16x40 0.4713 \u00b7 8x20 0.3374 \u00b7 4x40 0.3083 "
 "\u00b7 4x8 0.2077 \u00b7 16x4 0.2028 \u00b7 4x4 0.1583 \u00b7 4x2 0.1251 \u00b7 1x1 0.0909**, controls "
 "**pos_only 0.0317** and **all-zero 0.01634 = the test base rate 0.016198** \u2713 (IoU 0.000, F1 0.000). "
 "\u21d2 **4x4 costs 2.98x the AP of 16x40 and 1.95x the lateral error (2.669 m vs 1.372 m at "
 "15\u201330 m); the deployed 4x8 costs 2.27x.** \u2b50 **Axis attribution: pooling COLUMNS only to 4 "
 "costs \u221257.0 %, pooling ROWS only to 4 costs \u221234.6 % \u21d2 azimuth is 1.65x the more expensive "
 "axis.** \u2b50 **Replicate arms (same split, TRAINING seed 0/1/2 \u2014 `H-ESTIM-SEED-1`): noise floor "
 "\u2264 0.0122 AP against a smallest adjacent-rung gap of 0.130 (10.7x) and a headline gap of 0.313 "
 "(25.7x).** Paired episode-cluster bootstrap over the 30 test clips in `raw/bootstrap.json` \u2014 "
 "\u26a0 that interval answers *would another DRAW OF EPISODES say this?*, the replicates answer "
 "*would another TRAINING RUN say this?*. **(b) THE REAL TRUNK** (`k8clip05p30k`, step 30,000, "
 "0.97 M encoder): the same azimuth-indexed head **FITS its training split** (AP **0.2049** vs the "
 "marginal control's 0.0359) and **NONE of it transfers** (test AP **0.0295** vs pos_only **0.0325**); "
 "a linear dual-ridge panel reads **NEGATIVE R\u00b2 on every lateral target on every arm, raw pixels "
 "included**, with only the NON-spatial `n_occ` positive and the **pixel floor winning it (+0.099)**. "
 "\u2b50 **This ANSWERS `E-DEC-29`'s open NEXT** \u2014 the fine per-column structure is **not** present "
 "in the TOKENS either, so on this trunk the deficit is **upstream of the readout**, not the readout. "
 "\u21d2 **DESIGN CONSEQUENCE: WP-B (waypoint-indexed cross-attention) would index into a map with no "
 "agents in it; WP-D (a BEV auxiliary loss) is the real prerequisite, not WP-A's readout exposure** "
 "\u2014 and `E-DEC-8`'s DINOv3 distillation is the highest-measured-effect lever for it. "
 "\u26a0\u26a0 **TWO PREMISE CORRECTIONS, both MEASURED from artifacts:** (1) the live v7-tiny readout is "
 "**4x8**, not 4x4 \u2014 `k8-pull/ckpt.pt` carries `SpatialGridReadout grid/grid_w (4,8)`, "
 "`AvgPool2d((4,5))`, `proj.weight (64,128)`, argv `--readout-grid 4 --readout-grid-w 8`; the 4x4 / "
 "30\u00b0 figure is the **v1/v4/v5f flagship** line (`dynamics_encoder.py:146-148` passes no `grid_w`; "
 "`flagship_v15.py:97-98` pins 4*4*128). Both land on `d_op` 2048 \u2014 the geometry firewall \u2014 "
 "which is why they are confused. (2) \u26d4 **refcv5 has NO `SpatialGridReadout` at all**, so "
 "`REFCV5C_DESIGN.md` \u00a73's *\"pooled to 4 readout columns \u21d2 indexing into almost nothing\"* is a "
 "v6/v7 fact quoted into a REF-C design: refcv5-v2's own argv is `--image-hw 256 640` "
 "(`LAUNCH_RECEIPT.md` \u00a71), `refc_v3_train.py:216-221` sets `image_width=640`, REF-C is **stride "
 "32** (`refc.py:294-296`) \u21d2 the map is **8x20 = 20 azimuth columns = 6.0\u00b0/col**, and "
 "`refc.py:2112` shows the anchor decoder **already cross-attends** its 160 flattened tokens. \u21d2 "
 "refcv5c's missing piece is the **INDEXING**, not the resolution, and 8x20 retains **71.6 %** of the "
 "16x40 ceiling. \u26a0 And *\"indexing into a 4-column map is indexing into almost nothing\"* is **too "
 "strong even for 4x4**: it scores **5.0x its own marginal control and 9.7x all-zero**; the supported "
 "form is *loses ~2/3 of the available localisation*. \u26a0 **Geometry reproduced independently** "
 "(`raw/geometry.json`): cylindrical FOV **120.000\u00b0** (a pinhole formula gives 92.641\u00b0 and is "
 "wrong here \u2014 scoped to OUR 256x640 `f_ref` 305.577 corpus), `fov_mask` in-field **7,090/7,680 = "
 "92.32 %**, and `fov_census` at 4 columns **593 \u00b7 2,952 \u00b7 2,952 \u00b7 593** \u2014 the "
 "DataFlyWheel's census reproduces exactly. \u26a0 **NEGATIVES ARE TWO-STATE, NOT THREE:** out-of-field "
 "cells are EXCLUDED (`fov_mask`, 590 cells; 6,996/7,680 = 91.09 % scored) but a scored negative still "
 "MERGES *seen-and-empty* with *agent-occluded* \u2014 the third state remains missing programme-wide. "
 "\u26a0 **BLOCKER for turning (b) into a ceiling rather than a bound:** more episodes exist (the B1 "
 "TRAIN join covers 4,427 clips) but their **256x640 pixels are not on the dev box**. |"
)

NOTE_OLD = ("the deficit is the READOUT and the objective, not the encoder. |")
NOTE_NEW = ("the deficit is the READOUT and the objective, not the encoder. "
            "\u2b50 **ANSWERED 2026-09-07 \u2014 `E-READOUT-CEILING-1`: it is NOT present in the "
            "tokens either.** The same probe on the 16x40 TOKEN grid of `k8clip05p30k` reads "
            "**negative R\u00b2 on every lateral target** (linear), and an azimuth-indexed nonlinear "
            "head **fits its training split (AP 0.2049 vs the marginal control's 0.0359) and "
            "transfers nothing** (test 0.0295 vs pos_only 0.0325). \u21d2 on this trunk the deficit is "
            "**UPSTREAM of the readout**. \u26a0 Different arm from this row's (`k8clip05p30k` vs "
            "`splitp30k`/`rdw8p30k`), so this is corroboration, not identity. |")

ANCHOR = "| E-DEC-2 | Training WITH a wider-azimuth readout recovers"


def blob(path):
    out = subprocess.run(["git", "hash-object", path], capture_output=True,
                         text=True, cwd=os.path.dirname(REG))
    h = out.stdout.strip()
    return h if len(h) == 40 else None


before_hash = blob(REG)
if before_hash is None:
    sys.exit("INCONCLUSIVE: could not hash the register (mount flap?) - retry")
src = open(REG, encoding="utf-8").read()
print(f"[read] {len(src)} chars, blob {before_hash}")

n_guard = src.count(GUARD)
if n_guard:
    sys.exit(f"ALREADY APPLIED: guard {GUARD!r} appears {n_guard}x - refusing")
assert src.count(ANCHOR) == 1, f"anchor count {src.count(ANCHOR)} != 1"
assert src.count(NOTE_OLD) == 1, f"E-DEC-29 note anchor count {src.count(NOTE_OLD)} != 1"

new = src.replace(NOTE_OLD, NOTE_NEW).replace(ANCHOR, ROW + "\n\n" + ANCHOR)
assert new.count(GUARD) == 2, f"guard count after edit = {new.count(GUARD)}, want 2"
assert len(new) > len(src) + 3000, (len(src), len(new))
# ⭐ descendant check: everything that was in the source must survive
assert src.count("| E-DEC-29 |") == new.count("| E-DEC-29 |") == 1
assert new.count(ANCHOR) == 1
open(TMP, "w", encoding="utf-8", newline="").write(new)
print(f"[authored] {TMP} {len(new)} chars (+{len(new)-len(src)})")

# compare-and-swap: refuse if a sibling moved the file since we read it
now_hash = blob(REG)
if now_hash is None:
    sys.exit("INCONCLUSIVE: could not re-hash before write - retry")
if now_hash != before_hash:
    sys.exit(f"CONFLICT: register moved {before_hash} -> {now_hash}; re-run to rebase")
shutil.copyfile(TMP, REG)

back = open(REG, encoding="utf-8").read()
print(f"[verify] guard count in register = {back.count(GUARD)} (want 2); "
      f"len {len(back)} (want {len(new)}); "
      f"E-DEC-29 rows {back.count('| E-DEC-29 |')} (want 1)")
assert back.count(GUARD) == 2 and len(back) == len(new)
print("OK")
