#!/usr/bin/env python3
"""Build refav1's stage-1 cache: DINOv3-L fp8 features, encoded on the dev-box
4060 and shipped to Thor over the LAN — Thor's GPU is NEVER touched.

WHY THIS SHAPE (all measured 2026-09-01):
  * fp16 at B1+eval scale is 655.1 GiB and does not fit Thor's 426 free;
    fp8 e4m3 is 327.5 GiB and passed the decodability gate
    (fp8_l2_gate.py: worst-episode probe-MSE +3.4 %, median <= +0.8 %).
  * the 4060 encodes ~2.9 s/episode in bf16 (fp16 OVERFLOWS ViT-L — caught by
    a content assertion, never use it); the LAN moves ~36 MB/s, so pull (35 MB)
    and push (~66 MB) overlap under the encode when batched.
  * RESUMABLE: episodes already present on Thor (size-verified) are skipped, so
    a crash costs one batch, not the night.

Targets on Thor:
  /home/nvidia/data/dinov3-b1-fp8-w120-256x640cyl/       (train, 4,713 eps)
  /home/nvidia/data/dinov3-val600-fp8-w120-256x640cyl/   (eval,    600 eps)
Each <episode>.pt = torch.float8_e4m3fn [T=ceil(T_ep/2), 640, 1024] on the
0.2 s grid; index.json carries the geometry `verify_cache` demands.
"""
import io
import json
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torchvision

import truststore

truststore.inject_into_ssl()

THOR = "tanitad-thor-wifi"
SRC = {"train": "/home/nvidia/data/physicalai-train-e438721ae894-w120-256x640cyl",
       "b1": "/home/nvidia/data/physicalai-b1-w120-256x640cyl",
       "val": "/home/nvidia/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl"}
DST = {"b1": "/home/nvidia/data/dinov3-b1-fp8-w120-256x640cyl",
       "val": "/home/nvidia/data/dinov3-val600-fp8-w120-256x640cyl"}
WORK = Path("C:/Users/Admin/refav1_probe/ship")
BATCH = 16
#: PNG decode releases the GIL; 4 workers cut ~1.5 s/ep serial decode to ~0.5.
_DECODERS = ThreadPoolExecutor(max_workers=4)
MID = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def ssh(cmd: str) -> str:
    return subprocess.run(["ssh", "-o", "ConnectTimeout=25", "-o",
                           "BatchMode=yes", THOR, cmd],
                          capture_output=True, text=True, check=True).stdout


def scp(args: list[str]) -> None:
    subprocess.run(["scp", "-o", "ConnectTimeout=25", *args],
                   check=True, capture_output=True)


def main(split: str) -> int:
    src, dst = SRC[split], DST[split]
    WORK.mkdir(parents=True, exist_ok=True)
    ssh(f"mkdir -p {dst}")

    all_eps = sorted(l.strip().replace(".v2ep.pt", "") for l in
                     ssh(f"ls {src}").splitlines() if l.endswith(".v2ep.pt"))
    done = {l.split()[-1].replace(".pt", ""): int(l.split()[0]) for l in
            ssh(f"cd {dst} && ls -l 2>/dev/null | awk '{{print $5, $NF}}'"
                ).splitlines() if l.endswith(".pt")}
    todo = [e for e in all_eps if e not in done or done[e] < 10_000_000]
    print(f"[{split}] {len(all_eps)} episodes, {len(all_eps)-len(todo)} "
          f"already shipped, {len(todo)} to build", flush=True)
    if not todo:
        return 0

    tok = re.search(r"hf_[A-Za-z0-9]+", io.open(
        "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/Keys.txt",
        encoding="utf-8", errors="ignore").read()).group(0)
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(MID, token=tok)
    model = AutoModel.from_pretrained(MID, token=tok,
                                      dtype=torch.bfloat16).eval().cuda()
    mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1).cuda().bfloat16()
    std = torch.tensor(proc.image_std).view(1, 3, 1, 1).cuda().bfloat16()
    n_special = 1 + getattr(model.config, "num_register_tokens", 0)

    # fp8 round-trip preflight: the LOADER must be able to read what we write.
    t = torch.randn(2, 3).to(torch.float8_e4m3fn)
    torch.save(t, WORK / "_pre.pt")
    back = torch.load(WORK / "_pre.pt", weights_only=True).float()
    assert torch.isfinite(back).all()
    (WORK / "_pre.pt").unlink()

    # MEASURED before this pipeline existed: the sequential loop ran at
    # 243 eps/h (14.8 s/ep, ETA 19.3 h) -- transfers + serial decode were ~60 %
    # of the wall clock while the GPU idled. Pull-ahead + push-behind threads
    # + the decode pool make the encode the pacing stage.
    def _pull(batch):
        scp([*(f"{THOR}:{src}/{e}.v2ep.pt" for e in batch), str(WORK)])

    def _push_verify(outs):
        scp([*(str(WORK / o) for o in outs), f"{THOR}:{dst}/"])
        far = ssh(f"cd {dst} && ls -l {' '.join(outs)} | awk '{{print $5, $NF}}'")
        far_sz = {l.split()[1]: int(l.split()[0]) for l in far.splitlines()}
        for o in outs:
            local = (WORK / o).stat().st_size
            assert far_sz.get(o) == local,                 f"SIZE MISMATCH {o}: {far_sz.get(o)} != {local}"
            (WORK / o).unlink()

    t0, built = time.time(), 0
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    puller = threading.Thread(target=_pull, args=(batches[0],))
    puller.start()
    pusher = None
    for bi, batch in enumerate(batches):
        puller.join()
        if bi + 1 < len(batches):
            puller = threading.Thread(target=_pull, args=(batches[bi + 1],))
            puller.start()
        outs = []
        for e in batch:
            o = torch.load(WORK / f"{e}.v2ep.pt", map_location="cpu",
                           weights_only=False)
            assert str(o["codec"]) == "png"
            buf, lens = o["jpeg_buf"], o["jpeg_len"]
            offs = torch.cat([torch.zeros(1, dtype=lens.dtype),
                              lens.cumsum(0)])
            frames = torch.stack(list(_DECODERS.map(
                lambda j: torchvision.io.decode_png(
                    buf[offs[j]:offs[j + 1]].clone()),
                range(0, len(lens), 2))))
            feats = []
            with torch.inference_mode():
                for j in range(0, frames.shape[0], 8):
                    x = frames[j:j + 8].cuda().bfloat16() / 255.0
                    h = model(pixel_values=(x - mean) / std).last_hidden_state
                    feats.append(h[:, n_special:, :].cpu())
            f = torch.cat(feats).float()
            assert f.shape[1:] == (640, 1024) and torch.isfinite(f).all()
            assert float(f.abs().mean()) > 1e-3          # content, not zeros
            torch.save(f.to(torch.float8_e4m3fn), WORK / f"{e}.pt")
            outs.append(f"{e}.pt")
            (WORK / f"{e}.v2ep.pt").unlink()
        # push + size-verify BEHIND the next batch's encode (torn transfers
        # change the size; the loader's finite/content checks cover the rest)
        if pusher is not None:
            pusher.join()
        pusher = threading.Thread(target=_push_verify, args=(outs,))
        pusher.start()
        built += len(batch)
        rate = built / (time.time() - t0)
        print(f"[{split}] {built}/{len(todo)}  {rate*3600:.0f} eps/h  "
              f"eta {((len(todo)-built)/max(rate,1e-9))/3600:.1f} h", flush=True)

    if pusher is not None:
        pusher.join()
    geo = {"episodes": all_eps, "grid": "0.2s (every 2nd frame)",
           "dtype": "float8_e4m3fn",
           "geometry": {"n_tokens": 640, "d_enc": 1024, "hfov_deg": 120.0},
           "model": MID, "special_dropped": n_special,
           "gate": "fp8_l2_gate.json PASS 2026-09-01"}
    (WORK / "index.json").write_text(json.dumps(geo, indent=1))
    scp([str(WORK / "index.json"), f"{THOR}:{dst}/"])
    print(f"[{split}] DONE {built} built in {(time.time()-t0)/3600:.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "b1"))
