"""Bridge pai_epcache (ep_*.pt) -> the ph0 pilot layout (mp4 + ego npz + clips.json).

WHY: the Alpamayo-2 augmentation ran on the PhysicalAI OFFICIAL VAL split
("TanitAD OOD-val", a2_batch.sh), cached here as 290 ep_*.pt with real frames.
Our PH0 VLM/SAM3 pipeline has never seen it -- it processed the 600-clip
w120 val cache, of which only 56 are in the Alpamayo record set.

v2_to_pilot.py cannot read this: it takes a v2 --corpus (provider format), and
the epcache is per-episode tensors. This is the missing adapter.

⛔ THE CHANNEL INDEX IS NOT ARBITRARY, AND MY FIRST VERSION GOT IT WRONG.
frames_u8 is (T, 9, 256, 256). The 9 channels are NOT three cameras -- there is
exactly ONE camera in this corpus, `_FRONT_WIDE_CAM = "camera_front_wide_120fov"`
(physicalai.py:232), i.e. the 120-degree wide front camera. The 9 channels are
the **D-015 temporal stack**: 3 frames at 100 ms, channel-stacked, and
`comma2k19.stack_frames` states the order verbatim --

    "Oldest frame first, CURRENT FRAME IN THE LAST 3 CHANNELS."

So the current 120-deg frame is channels **6:9**. An earlier version of this
script took 0:3 and therefore rendered the **t-200 ms** frame: the VLM would
label a scene 200 ms stale relative to the ego state it is given, and SAM3's
frame-exact VLM cross-check (the one whose confound we already fixed once)
would be misaligned by two ticks. Measured cost: one wasted VLM launch.

⚠️ GEOMETRY, stated rather than silently reshaped: this cache is 256x256
PINHOLE, not the 256x640 cylindrical of the w120 caches. Same physical 120-deg
camera, different projection and crop, so sign/agent recall is NOT comparable
between the two batches and the two must not be pooled.
"""
import glob, json, os, sys
import numpy as np
import torch

# --------------------------------------------------------------------------- #
# ⛔ THIS DOOR CANNOT CARRY THE PARITY INGEST GATE, AND THAT IS A FINDING       #
# --------------------------------------------------------------------------- #
# `tests/test_build_parity_guard.py` DERIVES the set of corpus writers and
# requires each to call `parity.guard_corpus_build`. This one is exempt, with a
# reason that is checkable rather than asserted:
#
#   the ids here are POSITIONAL — `ep_00000`, taken from the epcache filename —
#   and the parity oracle keys on the PhysicalAI **clip id**. `parity.py:101`
#   states the split verbatim: the v2 cache is `<clip_id>.v2ep.pt`, "a different
#   uid space from the epcache's positional `ep_%05d.pt`". A gate here would be
#   asking the oracle a question in the wrong vocabulary: every lookup would MISS
#   and the guard would print a reassuring "0 contaminated" forever.
#
# ⚠️ A guard that cannot fail is not a guard (C107), and one that reports a
# clean answer because it asked the wrong question is worse than none — it is
# the `df`-reports-the-cluster trap. So this script REFUSES the pretence and
# says what is unknown, at run time, in its own output.
#
# ⇒ THE REAL FIX IS UPSTREAM AND IS ALREADY IN PLACE: the epcache this reads is
#   written by `epcache.build_episodes_cached` (or `rebuild_pai_rolling`), and
#   BOTH are gated on the clip ids while they still have them. The exposure that
#   remains is a hand-assembled epcache of unknown provenance.
_PARITY_UID_GAP = (
    "epcache_to_pilot: ids are positional ep_%05d, NOT PhysicalAI clip ids; "
    "the parity ingest gate (parity.py §10c) CANNOT be evaluated here. "
    "Provenance is inherited from whoever built the epcache — which IS gated "
    "(epcache.build_episodes_cached / rebuild_pai_rolling). If this epcache was "
    "assembled by hand, its parity status is UNKNOWN.")
print(f"[parity] ⚠ {_PARITY_UID_GAP}", flush=True)

SRC = sys.argv[1]
OUT = sys.argv[2]
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 0
FPS = 10

os.makedirs(f"{OUT}/videos", exist_ok=True)
os.makedirs(f"{OUT}/ego", exist_ok=True)
import imageio.v2 as iio

eps = sorted(glob.glob(os.path.join(SRC, "ep_*.pt")))
if LIMIT:
    eps = eps[:LIMIT]
ids, n_ok, n_fail = [], 0, 0
for p in eps:
    cid = os.path.splitext(os.path.basename(p))[0]          # ep_00000
    try:
        d = torch.load(p, map_location="cpu", weights_only=False)
        fr = d["frames_u8"]                                  # (T, 9, H, W)
        if fr.shape[1] % 3:
            raise ValueError(f"{cid}: {fr.shape[1]} channels is not a multiple "
                             "of 3 — the D-015 stack assumption is wrong here")
        cur = fr.shape[1] - 3            # LAST 3 = the current 120-deg frame
        rgb = fr[:, cur:cur + 3].permute(0, 2, 3, 1).contiguous().numpy()
        iio.mimwrite(f"{OUT}/videos/{cid}.mp4", rgb, fps=FPS,
                     codec="libx264", quality=7)
        np.savez(f"{OUT}/ego/{cid}.npz",
                 poses=d["poses"].numpy().astype("float32"),
                 actions=d["actions"].numpy().astype("float32"))
        ids.append(cid)
        n_ok += 1
    except Exception as e:                                   # noqa: BLE001
        n_fail += 1
        print(f"FAIL {cid} {type(e).__name__}: {e}", flush=True)
    if n_ok % 25 == 0 and n_ok:
        print(f"BRIDGED {n_ok}/{len(eps)}", flush=True)
json.dump(ids, open(f"{OUT}/clips.json", "w"), indent=1)
print(f"EPC_BRIDGE_DONE ok={n_ok} fail={n_fail} clips={len(ids)}")
