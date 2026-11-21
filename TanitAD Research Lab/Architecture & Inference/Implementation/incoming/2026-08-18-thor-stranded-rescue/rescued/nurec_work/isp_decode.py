"""Decode the PPISP tensors + find the view indexing (3594 = 6 cams x 599 frames?)."""
import msgpack, json, numpy as np

PATH = "/home/nvidia/nurec_work/x/volume.msgpack"
obj = msgpack.unpackb(open(PATH, "rb").read(), raw=False, strict_map_key=False)
nre = obj["nre_data"]; sd = nre["state_dict"]; cfg = nre["config"]

print("config top keys:", list(cfg.keys()))
print("\n--- config['post_processing'] ---")
print(json.dumps(cfg.get("post_processing"), indent=1)[:2000])

P = ".post_processings.0.ppisp."
def arr(name, dtype=np.float16):
    k = P + name
    shp = sd[k + ".shape"]
    b = sd[k]
    n = int(np.prod(shp)) if len(shp) else 1
    bpe = len(b) / max(n, 1)
    a = np.frombuffer(b, dtype=dtype)
    print(f"{name}: shape={list(shp)} bytes={len(b)} bytes/elem={bpe:.2f} -> {dtype.__name__}")
    return a.astype(np.float64).reshape([int(x) for x in shp]) if len(shp) else a.astype(np.float64)

print("\n=== PPISP tensors ===")
exp_ = arr("exposure_params")
vig  = arr("vignetting_params")
col  = arr("color_params")
crf  = arr("crf_params")
chrom= arr("_default_source_chroms")
src_i = np.frombuffer(sd[P+"_smoothness_src_indices"], dtype=np.int64)
dst_i = np.frombuffer(sd[P+"_smoothness_dst_indices"], dtype=np.int64)
print(f"_smoothness_src_indices: {src_i.shape} head={src_i[:8]} tail={src_i[-4:]}")
print(f"_smoothness_dst_indices: {dst_i.shape} head={dst_i[:8]} tail={dst_i[-4:]}")
print(f"_default_source_chroms:\n{chrom}")

print(f"\nexposure: min={exp_.min():.4f} max={exp_.max():.4f} mean={exp_.mean():.4f}")
print(f"  first 12: {np.round(exp_[:12],4)}")
print(f"  per-block-of-599 means: {[round(float(exp_[i*599:(i+1)*599].mean()),4) for i in range(6)]}")
print(f"  per-block-of-599 stds : {[round(float(exp_[i*599:(i+1)*599].std()),5) for i in range(6)]}")
print(f"  block0 first 6: {np.round(exp_[:6],5)}   block0 last 3: {np.round(exp_[596:599],5)}")
# alternative interleave: view = frame*6 + cam
print(f"  stride-6 (cam0 if interleaved) first 6: {np.round(exp_[0::6][:6],5)} std={exp_[0::6].std():.5f}")

print(f"\nvignetting [6,3,5]:\n{np.round(vig,5)}")
print(f"\ncrf [6,3,7]:\n{np.round(crf,5)}")
print(f"\ncolor [3594,8]: min={col.min():.4f} max={col.max():.4f}")
print(f"  view0: {np.round(col[0],5)}")
print(f"  view1: {np.round(col[1],5)}")
print(f"  view599: {np.round(col[599],5)}")
print(f"  per-block-of-599 mean over 8: ")
for i in range(6):
    print(f"    blk{i}: {np.round(col[i*599:(i+1)*599].mean(0),5)}  std={np.round(col[i*599:(i+1)*599].std(0),5)}")

# camera embeds
k = ".calib.camera_view_geometry.embeds.weight"
emb = np.frombuffer(sd[k], dtype=np.float16).astype(np.float64).reshape(sd[k+".shape"])
print(f"\ncamera_view_geometry.embeds.weight {emb.shape}: min={emb.min():.4f} max={emb.max():.4f}")
print(f"  view0: {np.round(emb[0],5)}")
print(f"  |mean| per col: {np.round(np.abs(emb).mean(0),5)}")

# sensor models -> which of the 6 is front_wide?
for i in range(6):
    b = f".calib.camera_view_geometry.sensor_models.{i}."
    res = np.frombuffer(sd[b+"resolution"], dtype=np.int32)
    pp  = np.frombuffer(sd[b+"principal_point"], dtype=np.float16)
    fw  = np.frombuffer(sd[b+"fw_poly"], dtype=np.float16)
    print(f"sensor{i}: resolution={res} principal_point={pp} fw_poly={np.round(fw.astype(float),4)}")
