"""Decode EVERY frame of the one v2ep through the corpus's own reader
(`tanitad.data.v2_dataset._decode_stacked`, codec from the payload) and,
separately, through the builder's exact call (`torchvision.io.decode_png` on
every 2nd frame, no mode arg). Reports per-frame health and that the two paths
agree byte-for-byte on the frames the encoder consumes."""
import json
import sys
import time
from pathlib import Path

import torch
import torchvision.io as tvio

import tanitad
from tanitad.data import v2_dataset as v2

print("tanitad ->", tanitad.__file__)
print("v2_dataset ->", v2.__file__)

REB = Path("C:/Users/Admin/refav1_probe/rebuild")
CLIP = "16d325e9-dbfe-439c-a0ed-4ff8850ad2e2"
ep = REB / f"{CLIP}.v2ep.pt"
d = torch.load(ep, map_location="cpu", weights_only=False)
buf, lens = d["jpeg_buf"], d["jpeg_len"]
n_stack, codec = int(d["n_stack"]), str(d.get("codec", "jpeg"))
offs = v2._jpeg_offsets(lens)
T_raw = int(lens.shape[0])
k = n_stack - 1
rep = {"clip": CLIP, "codec": codec, "n_stack": n_stack, "T_raw_frames": T_raw,
       "T_poses": int(d["poses"].shape[0])}

# --- corpus reader path: ALL stacked rows [0, T_raw-k) => raw frames [0, T_raw)
t0 = time.time()
stacked = v2._decode_stacked(buf, offs, n_stack, 0, T_raw - k, codec, None)
rep["reader_stacked_shape"] = list(stacked.shape)
rep["reader_stacked_dtype"] = str(stacked.dtype)
rep["reader_decode_s"] = round(time.time() - t0, 2)
# newest-only path gives the raw frames one per row (rows [0,T_raw-k) -> raw [k, T_raw))
newest = v2._decode_stacked(buf, offs, n_stack, 0, T_raw - k, codec, None,
                            newest_only=True)
rep["reader_newest_shape"] = list(newest.shape)

# --- every raw frame individually through the reader's decoder call
dec = tvio.decode_png if codec == "png" else tvio.decode_jpeg
per = []
bad = []
for i in range(T_raw):
    try:
        f = dec(buf[int(offs[i]):int(offs[i + 1])], mode=tvio.ImageReadMode.RGB)
        ff = f.float()
        per.append({"i": i, "shape": list(f.shape), "mean": round(float(ff.mean()), 3),
                    "std": round(float(ff.std()), 3),
                    "zero": bool((f == 0).all())})
    except Exception as e:  # noqa: BLE001
        bad.append({"i": i, "err": f"{type(e).__name__}: {e}"})
rep["frames_decoded_ok"] = len(per)
rep["frames_failed"] = bad
shapes = sorted({tuple(p["shape"]) for p in per})
rep["distinct_shapes"] = [list(s) for s in shapes]
rep["all_zero_frames"] = [p["i"] for p in per if p["zero"]]
means = [p["mean"] for p in per]
stds = [p["std"] for p in per]
rep["frame_mean_min_max"] = [min(means), max(means)]
rep["frame_std_min_max"] = [min(stds), max(stds)]

# --- the BUILDER's exact decode (no mode arg) on every 2nd frame, vs reader
t0 = time.time()
bframes = torch.stack([tvio.decode_png(buf[offs[j]:offs[j + 1]].clone())
                       for j in range(0, T_raw, 2)])
rep["builder_frames_shape"] = list(bframes.shape)
rep["builder_decode_s"] = round(time.time() - t0, 2)
rframes = torch.stack([dec(buf[int(offs[j]):int(offs[j + 1])],
                           mode=tvio.ImageReadMode.RGB) for j in range(0, T_raw, 2)])
rep["builder_vs_reader_identical"] = bool(torch.equal(bframes, rframes))
rep["builder_vs_reader_max_abs_diff"] = int((bframes.int() - rframes.int()).abs().max())

# --- the manifest scan the provider list uses (metadata-only read)
poses, actions, eid, ns, S, cid, H, W = v2._scan_meta(str(ep))
rep["scan_meta"] = {"T_out": int(poses.shape[0]), "episode_id": eid, "n_stack": ns,
                    "image_size": S, "clip_id": cid, "H": H, "W": W}
rep["verdict"] = ("SOURCE DECODES CLEANLY" if not bad and not rep["all_zero_frames"]
                  and len(shapes) == 1 else "SOURCE DEFECT")
print(json.dumps({k: v for k, v in rep.items()}, indent=1))
(REB / "decode_check.json").write_text(json.dumps(rep, indent=1))
print("banked ->", REB / "decode_check.json")
