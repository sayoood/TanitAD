"""Frozen-WM planner experiment — STAGE 1: encode-cache.
Encode every frame of a train subset + the full pod val set with v1's FROZEN
encoder (encoder+readout -> compact state 2048) ONCE, so planner training is
cheap (windows are slices of the cached per-frame states; encode_window is
per-frame independent -> caching per frame is exact).

Cache layout: /root/frozenwm/cache/{train,val}/ep_XXXXX.pt =
  {states [T,2048] fp16, actions [T,2] f32, poses [T,4] f32, eid str}
"""
import sys, time, json, argparse
from pathlib import Path
import torch
sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")
from taniteval.loaders import load
from tanitad.data.mixing import load_episode

DEV = "cuda"
TRAIN_DIR = "/workspace/data/physicalai_phase0/_epcache/physicalai-train-e438721ae894"
VAL_DIR = "/root/valdata/physicalai-val-0c5f7dac3b11"
OUT = Path("/root/frozenwm/cache")


def encode_ep(model, ep, bs=64):
    T = ep.frames.shape[0]
    outs = []
    for i in range(0, T, bs):
        fw = ep.frames[i:i+bs].to(DEV).float().div_(255.0)   # [b,9,256,256]
        with torch.no_grad():
            s = model.encode(fw)                              # [b,2048]
        outs.append(s.half().cpu())
    return torch.cat(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    entry = dict(key="flagship-30k", arch="flagship-worldmodel",
                 ckpt="/root/models/flagship-30k/ckpt.pt", speed_input=True)
    h = load(entry, device=DEV)
    model = h["model"]; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    print("loaded flagship-30k, state_dim", h["state_dim"], flush=True)

    train_files = sorted(Path(TRAIN_DIR).glob("ep_*.pt"))
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(train_files), generator=g)[:args.n_train].tolist()
    train_files = [train_files[i] for i in sorted(perm)]
    val_files = sorted(Path(VAL_DIR).glob("ep_*.pt"))
    print(f"train {len(train_files)} eps, val {len(val_files)} eps", flush=True)

    for split, files in (("val", val_files), ("train", train_files)):
        d = OUT / split; d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for j, f in enumerate(files):
            outp = d / f.name
            if outp.exists():
                continue
            ep = load_episode(str(f), mmap=True)
            T = min(ep.frames.shape[0], ep.actions.shape[0], ep.poses.shape[0])
            states = encode_ep(model, ep)[:T]
            torch.save({"states": states, "actions": ep.actions[:T].float(),
                        "poses": ep.poses[:T].float(), "eid": f.stem}, outp)
            if (j+1) % 25 == 0 or j == len(files)-1:
                el = time.time()-t0
                print(f"  {split} {j+1}/{len(files)}  {el:.1f}s  "
                      f"({(j+1)/el:.1f} ep/s)", flush=True)
    print("ENCODE_DONE", flush=True)


if __name__ == "__main__":
    main()
