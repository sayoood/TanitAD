"""STEP 1e — the probe only (sources already dumped): first module emitting
bfloat16, and the autocast state through the trunk."""
import os
import sys

os.chdir("/content")
for p in ("/content/repo/colab", "/content/repo/stack",
          "/content/repo/stack/scripts"):
    if p not in sys.path:
        sys.path.insert(0, p)
import json                                                      # noqa: E402
import torch                                                     # noqa: E402
import ph0_sam3                                                  # noqa: E402
import ph0_pilot                                                 # noqa: E402
import s2_lab_lib as L2                                          # noqa: E402

fx = json.load(open("/content/repo/colab/fixtures/"
                    "sam3_backfill_expected.json"))
CID = fx["clips"][0]
mp4 = L2.hf_download(L2.DS_LABELS,
                     f"bridged_w120train_2400/videos/{CID}.mp4")
frames, _t, _n = ph0_pilot.sample_clip_frames(mp4, t0_s=8.0)
from PIL import Image                                            # noqa: E402
IMG = Image.fromarray(frames[len(frames) // 2])

PROC, _meta = ph0_sam3.build_processor(None)
model = PROC.model
log = []


def mk(name):
    def h(mod, args, out):
        it = args[0] if args and torch.is_tensor(args[0]) else None
        ot = out if torch.is_tensor(out) else (
            out[0] if isinstance(out, (list, tuple)) and out
            and torch.is_tensor(out[0]) else None)
        log.append((name, type(mod).__name__,
                    str(getattr(it, "dtype", None)).replace("torch.", ""),
                    str(getattr(ot, "dtype", None)).replace("torch.", ""),
                    torch.is_autocast_enabled("cuda")))
    return h


hs = [m.register_forward_hook(mk(n)) for n, m in model.named_modules()]
try:
    PROC.set_image(IMG)
    print("[probe] set_image OK")
except Exception as e:
    print("[probe] set_image raised:", type(e).__name__, str(e)[:110])
for h in hs:
    h.remove()

print(f"[probe] {len(log)} module records; first 30 (ac = autocast on):")
for r in log[:30]:
    print(f"   ac={int(r[4])} {r[1]:<20} in={r[2]:<10} out={r[3]:<10} {r[0]}")
fb = next((i for i, r in enumerate(log) if r[3] == "bfloat16"), None)
print("\n[probe] FIRST module emitting bfloat16: idx", fb)
if fb is not None:
    for r in log[max(0, fb - 3):fb + 2]:
        print(f"   ac={int(r[4])} {r[1]:<20} in={r[2]:<10} out={r[3]:<10} "
              f"{r[0]}")
print("[probe] any module with autocast ON:",
      any(r[4] for r in log))
print("DIAG5_DONE")
