"""D3 scoping probe 1: does a bigger train manifest silently move the RA-1 holdout?  (read-only, CPU, seconds)

The v8F-mini trainer holds the RA-1 60-clip selection holdout out by POSITION:
    excluded = sorted(clips_all, key=sha12)[540:600]            (v8f_mini_train.py build_data, --exclude 540:600)
and `ra1_pooled.holdout_clips` defines the same set as "the LAST 60 of the first 600 in sha12 order". Both are a function of
the MANIFEST. This probe measures what happens to that set when new clips are merged into the train manifest.

Steps (each prints a number):
  1. rebuild the banked holdout from the 4,572-clip train list and check its sha256 against the pinned f6a1dee0... (the
     instrument works: known value).
  2. merge N candidate new clips (random draw from the PhysicalAI-AV clips OUTSIDE B1, fixed seed) and recompute the positional
     [540:600]: how many of the ORIGINAL 60 are still excluded, and how many of them would now be TRAINED ON.
  3. the control: an explicit-list exclusion (by clip id) keeps all 60 out whatever the manifest size.
Run:  python -I probe_holdout_shift.py > probe_holdout_shift.out
"""
import hashlib
import os
import random

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
def lp(path):
    """Windows MAX_PATH: the export lives under a very deep scratch path, so open() needs the long-path prefix (normalised first)."""
    bs = chr(92)
    pre = bs + bs + "?" + bs
    p = os.path.normpath(os.path.abspath(path))
    return pre + p if os.name == "nt" and not p.startswith(pre) else p


REFAV1 = lp(os.path.join(HERE, "..", "..", "..", "2026-10-17-refav1-hj-mini", "raw", "ra1_full", "clips_all.txt"))
CLIP_INDEX = "C:/Users/Admin/tanitad-data/physicalai/clip_index.parquet"      # 306,152 rows (MEASURED), local
PINNED_HOLDOUT_SHA256 = "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"      # ra1_ridge_probe.py:52
PINNED_TRAIN4512_SHA256 = "37f4f9ee468e"            # PREREG_RA1_FULL / RESULT_RA1_FULL provenance.train_sha256 (prefix)


def sha12(c):
    return hashlib.sha256(c.encode("utf-8")).hexdigest()[:12]


def digest(lst):
    return hashlib.sha256(chr(10).join(lst).encode("utf-8")).hexdigest()


b1_train = [l.strip() for l in open(REFAV1, encoding="utf-8") if l.strip()]
assert len(b1_train) == 4572 and len(set(b1_train)) == 4572
order0 = sorted(b1_train, key=sha12)
hold0 = order0[540:600]
print(f"[1] B1 train list {len(b1_train)} clips; positional [540:600] has {len(hold0)} clips")
print(f"    sha256(holdout)      = {digest(hold0)}")
print(f"    pinned (RA-1 full)   = {PINNED_HOLDOUT_SHA256}")
print(f"    reproduces the pin   = {digest(hold0) == PINNED_HOLDOUT_SHA256}")
train0 = [c for c in order0 if c not in set(hold0)]
print(f"    train 4,512 digest   = {digest(train0)}  (pinned prefix {PINNED_TRAIN4512_SHA256}: {digest(train0).startswith(PINNED_TRAIN4512_SHA256)})")

idx = pd.read_parquet(CLIP_INDEX)
cams = set(os.path.basename(f)[:-4] for f in os.listdir("C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov") if f.endswith(".mp4"))
pool = sorted(set(idx.index) - cams)
print(f"[2] pool outside B1 = {len(pool)} clips (clip_index {len(idx)} - B1 mp4s {len(cams)})")
rng = random.Random(20261025)
H0 = set(hold0)
for n_new in (4512, 40000):
    new = rng.sample(pool, n_new)
    merged = b1_train + new
    order1 = sorted(merged, key=sha12)
    hold1 = order1[540:600]
    train1 = set(order1) - set(hold1)
    kept_out = len(H0 & set(hold1))
    leaked = len(H0 & train1)
    print(f"    +{n_new:>6} new clips -> manifest {len(merged)}; positional [540:600] still excludes {kept_out}/60 of the original holdout;"
          f" {leaked}/60 original holdout clips would be TRAINED ON; the other {60 - kept_out} excluded clips are new/other")
    explicit = set(order1) - H0
    print(f"        control, explicit-id exclusion: original holdout in train = {len(H0 & explicit)}/60 (must be 0)")
    print(f"        positional digest {digest(hold1)[:12]} vs pinned {PINNED_HOLDOUT_SHA256[:12]}: equal = {digest(hold1) == PINNED_HOLDOUT_SHA256}")
