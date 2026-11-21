"""MEASURE PNG size and encode time of real 416x1024 NavSim stitch frames (navtest bank shard, read-only).
torchvision.io.encode_png is the codec v2_compressed.build_compressed(codec="png") uses (v2_compressed.py:143)."""
import json, sys, time
import numpy as np, torch, torchvision.io as tvio
torch.set_num_threads(1)
src, out = sys.argv[1], sys.argv[2]
a = np.load(src, mmap_mode="r")
shape = list(a.shape)
flat = a.reshape(-1, *a.shape[-3:]) if a.ndim > 4 else a
n = flat.shape[0]
idx = np.linspace(0, n - 1, 40).astype(int)
sizes, secs, roundtrip = [], [], 0
for i in idx:
    fr = np.ascontiguousarray(flat[i])                      # [416, 1024, 3] u8
    if fr.mean() == 0:
        continue
    t = torch.from_numpy(fr).permute(2, 0, 1).contiguous()
    t0 = time.perf_counter(); enc = tvio.encode_png(t); secs.append(time.perf_counter() - t0)
    sizes.append(int(enc.numel()))
    dec = tvio.decode_png(enc)
    roundtrip += int(torch.equal(dec, t))
res = {"source": src, "array_shape": shape, "n_frames_in_file": int(n), "n_sampled": len(sizes),
       "png_bytes_mean": float(np.mean(sizes)), "png_bytes_min": int(min(sizes)), "png_bytes_max": int(max(sizes)),
       "raw_bytes_per_frame": 416 * 1024 * 3, "ratio_raw_over_png": float(416 * 1024 * 3 / np.mean(sizes)),
       "encode_s_mean_1thread": float(np.mean(secs)), "lossless_roundtrip_bitexact": f"{roundtrip}/{len(sizes)}",
       "projected_bank_GB_99824_frames": float(99824 * np.mean(sizes) / 1e9),
       "projected_encode_h_99824_frames_6threads": float(99824 * np.mean(secs) / 6 / 3600)}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
