import glob, os, zipfile
import torch

val = "/root/valdata/physicalai-val-0c5f7dac3b11"
eps = sorted(glob.glob(val + "/ep_*.pt"))
print("n_val_eps:", len(eps))
d = torch.load(eps[0], map_location="cpu", weights_only=False)
print("ep_00000 keys:", list(d.keys()) if isinstance(d, dict) else str(type(d)))
if isinstance(d, dict):
    for k, v in d.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k}: {repr(v)[:90]}")

print("=== targeted raw-frame path checks (non-recursive) ===")
for p in ("/root/valdata", "/workspace/data/physicalai", "/root/data/physicalai",
          "/workspace/pai_epcache", "/root/pai", "/workspace/physicalai"):
    print(f"  {p}: exists={os.path.isdir(p)}", os.listdir(p)[:6] if os.path.isdir(p) else "")

print("=== USDZ camera-calib peek ===")
usdz = ("/workspace/scene_dl/sample_set/26.04_release/"
        "01d503d4-449b-46fc-8d78-9085e70d3554/"
        "01d503d4-449b-46fc-8d78-9085e70d3554.usdz")
z = zipfile.ZipFile(usdz)
names = z.namelist()
print("  entries:", len(names))
for x in names[:30]:
    print("   ", x)
cal = [x for x in names if any(t in x.lower() for t in
       ("calib", "intrinsic", "ftheta", "f_theta", "camera", "rig", "meta", "yaml", "json"))]
print("  calib-ish entries:", cal[:25])
print("RECON2_DONE")
