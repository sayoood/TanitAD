"""SAM3 on Thor: liveness + road-paint prompts on one demo image, saved as an overlay for inspection."""
import os, sys, time
sys.path.insert(0, "/home/nvidia/sam3vendor")
import numpy as np
import torch
from PIL import Image, ImageDraw
import ph0_sam3                                                     # the programme's proven SAM3 wrapper bits

CKPT = "/home/nvidia/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
BPE = "/home/nvidia/sam3vendor/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
PROMPTS = ["road", "lane marking", "road marking", "crosswalk", "zebra crossing", "stop line", "arrow marking on road"]


def build(conf=0.25):
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    fix = ph0_sam3.install_dtype_agreement()                         # C77: before any forward
    model = build_sam3_image_model(bpe_path=BPE, checkpoint_path=CKPT, load_from_HF=False, device="cuda")
    proc = Sam3Processor(model, confidence_threshold=conf)
    assert abs(float(proc.confidence_threshold) - conf) < 1e-9, "threshold not applied"
    return proc, fix


if __name__ == "__main__":
    img_path = sys.argv[1]
    t0 = time.time()
    proc, fix = build()
    print("built in %.1fs  dtype_fix=%s  maxmem %.2f GB" % (time.time() - t0, fix.get("applied"), torch.cuda.max_memory_allocated() / 1e9), flush=True)
    img = Image.open(img_path).convert("RGB")
    t1 = time.time()
    state = proc.set_image(img)
    over = np.asarray(img).astype(np.float32).copy()
    colors = {"road": (80, 80, 255), "lane marking": (255, 220, 0), "road marking": (255, 120, 0), "crosswalk": (0, 230, 200),
              "zebra crossing": (0, 160, 255), "stop line": (255, 0, 0), "arrow marking on road": (255, 0, 255)}
    for pr in PROMPTS:
        out = proc.set_text_prompt(state=state, prompt=pr)
        sc = out["scores"].float().cpu().numpy().reshape(-1) if out.get("scores") is not None else np.zeros(0)
        ms = out["masks"].cpu().numpy() if out.get("masks") is not None else np.zeros((0, 1, 1, 1))
        ms = ms.reshape(len(sc), ms.shape[-2], ms.shape[-1]) if len(sc) else np.zeros((0,) + ms.shape[-2:])
        px = int(ms.any(axis=0).sum()) if len(sc) else 0
        print(f"  {pr:24s} n={len(sc):3d}  scores={[round(float(s), 2) for s in sorted(sc, reverse=True)[:6]]}  mask_px={px}", flush=True)
        if len(sc) and pr != "road":
            m = ms.any(axis=0)
            over[m] = 0.45 * over[m] + 0.55 * np.array(colors[pr])
    print("prompts in %.1fs  maxmem %.2f GB" % (time.time() - t1, torch.cuda.max_memory_allocated() / 1e9))
    Image.fromarray(over.clip(0, 255).astype(np.uint8)).resize((960, 540)).save(sys.argv[2])
    print("wrote", sys.argv[2])
