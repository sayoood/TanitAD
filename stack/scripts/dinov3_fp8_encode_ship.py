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

⛔ THE PUSH IS ATOMIC ON THE FAR SIDE (2026-09-02): files travel as
`<episode>.pt.part`, are size-verified, and only then `mv`-ed to their final
name. MEASURED 2026-09-01 (ship_b1.log, episode ~432): a LAN blip killed the
push thread mid-file and left `16d325e9-….pt` on Thor at 61,624,320 of
66,193,268 B — 94.03 frames, no zip central directory. The resume rule below
("size >= 10 MB == done") then counted the torn file as BUILT, so it was never
rebuilt; the loader found it a day later and the clip had to be excluded from
the training split. A partial file must never carry the final name — the same
rule `build_compressed` applies to the v2ep itself ("must not leave a corrupt
.pt"). `--only <clip>` rebuilds named episodes through this SAME encode path
(ignores the resume scan, keeps the local copy, never rewrites index.json).
"""
import argparse
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
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def _run(cmd: list[str], tries: int = 3):
    """Transient-tolerant runner. MEASURED 2026-09-02: one scp exit-255 blip
    inside the puller THREAD was silently swallowed, the main loop then
    torch.load-ed a half-transferred file and died on miniz 432 episodes in —
    the C111 family (a failure after the compute is paid). Retries + explicit
    propagation are the fix, not hope."""
    last = None
    for k in range(tries):
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  check=True)
        except subprocess.CalledProcessError as e:
            last = e
            time.sleep(5 * (k + 1))
    raise last


def ssh(cmd: str) -> str:
    return _run(["ssh", "-o", "ConnectTimeout=25", "-o", "BatchMode=yes",
                 THOR, cmd]).stdout


def scp(args: list[str]) -> None:
    _run(["scp", "-o", "ConnectTimeout=25", *args])


def load_encoder(device: str, dtype: torch.dtype = torch.bfloat16):
    """The ONE encoder construction, shared by the batch pipeline and --only:
    DINOv3 ViT-L/16 (HF), its processor's ImageNet mean/std, and the count of
    special tokens (CLS + registers) to drop from the front of the sequence."""
    tok = re.search(r"hf_[A-Za-z0-9]+", io.open(
        "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/Keys.txt",
        encoding="utf-8", errors="ignore").read()).group(0)
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(MID, token=tok)
    model = AutoModel.from_pretrained(MID, token=tok,
                                      dtype=dtype).eval().to(device)
    mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1).to(device, dtype)
    std = torch.tensor(proc.image_std).view(1, 3, 1, 1).to(device, dtype)
    n_special = 1 + getattr(model.config, "num_register_tokens", 0)
    return model, mean, std, n_special


def encode_episode(o: dict, model, mean, std, n_special: int,
                   device: str) -> torch.Tensor:
    """One v2ep dict -> float32 [ceil(T_ep/2), 640, 1024] patch field on the
    0.2 s grid (every 2nd RAW frame; no resize — the 256x640 cylindrical frame
    IS the 16x40 patch grid). CLS + register tokens dropped. Factored out so
    --only runs byte-for-byte the code the batch pipeline runs."""
    assert str(o["codec"]) == "png"
    buf, lens = o["jpeg_buf"], o["jpeg_len"]
    offs = torch.cat([torch.zeros(1, dtype=lens.dtype), lens.cumsum(0)])
    frames = torch.stack(list(_DECODERS.map(
        lambda j: torchvision.io.decode_png(buf[offs[j]:offs[j + 1]].clone()),
        range(0, len(lens), 2))))
    feats = []
    with torch.inference_mode():
        for j in range(0, frames.shape[0], 8):
            x = frames[j:j + 8].to(device).to(mean.dtype) / 255.0
            h = model(pixel_values=(x - mean) / std).last_hidden_state
            feats.append(h[:, n_special:, :].cpu())
    f = torch.cat(feats).float()
    assert f.shape[1:] == (640, 1024) and torch.isfinite(f).all()
    assert float(f.abs().mean()) > 1e-3          # content, not zeros
    return f


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("split", nargs="?", default="b1", choices=sorted(DST))
    ap.add_argument("--only", nargs="+", metavar="CLIP",
                    help="rebuild ONLY these episode ids through the same "
                         "encode path: ignores the resume scan, keeps the "
                         "local fp8 copy, never rewrites index.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=sorted(DTYPES),
                    help="encoder dtype (the shipped cache is bf16)")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch CPU threads (0 = torch default)")
    ap.add_argument("--work", type=Path, default=WORK)
    ap.add_argument("--keep-fp16", action="store_true",
                    help="also write <clip>.fp16.pt beside the fp8 file "
                         "(the fp16 arm fp8_l2_gate.py scores against)")
    ap.add_argument("--no-ship", action="store_true",
                    help="encode only; push nothing to Thor")
    a = ap.parse_args(argv)
    split, work = a.split, a.work
    src, dst = SRC[split], DST[split]
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    work.mkdir(parents=True, exist_ok=True)
    ssh(f"mkdir -p {dst}")

    all_eps = sorted(l.strip().replace(".v2ep.pt", "") for l in
                     ssh(f"ls {src}").splitlines() if l.endswith(".v2ep.pt"))
    if a.only:
        missing = sorted(set(a.only) - set(all_eps))
        if missing:
            raise SystemExit(f"[{split}] --only: not in {src}: {missing}")
        todo = sorted(a.only)
        print(f"[{split}] --only: {len(todo)} episode(s) to rebuild on "
              f"{a.device}/{a.dtype} (resume scan bypassed)", flush=True)
    else:
        done = {l.split()[-1].replace(".pt", ""): int(l.split()[0]) for l in
                ssh(f"cd {dst} && ls -l 2>/dev/null | awk '{{print $5, $NF}}'"
                    ).splitlines() if l.endswith(".pt")}
        todo = [e for e in all_eps if e not in done or done[e] < 10_000_000]
        print(f"[{split}] {len(all_eps)} episodes, {len(all_eps)-len(todo)} "
              f"already shipped, {len(todo)} to build", flush=True)
    if not todo:
        return 0
    keep_local = bool(a.only)

    model, mean, std, n_special = load_encoder(a.device, DTYPES[a.dtype])

    # fp8 round-trip preflight: the LOADER must be able to read what we write.
    t = torch.randn(2, 3).to(torch.float8_e4m3fn)
    torch.save(t, work / "_pre.pt")
    back = torch.load(work / "_pre.pt", weights_only=True).float()
    assert torch.isfinite(back).all()
    (work / "_pre.pt").unlink()

    # MEASURED before this pipeline existed: the sequential loop ran at
    # 243 eps/h (14.8 s/ep, ETA 19.3 h) -- transfers + serial decode were ~60 %
    # of the wall clock while the GPU idled. Pull-ahead + push-behind threads
    # + the decode pool make the encode the pacing stage.
    def _pull(batch, box):
        try:
            scp([*(f"{THOR}:{src}/{e}.v2ep.pt" for e in batch), str(work)])
        except Exception as e:                      # propagated, never swallowed
            box["exc"] = e

    def _push_verify(outs, box):
      try:
        # ⛔ ATOMIC on the far side (module docstring): travel as `.part`,
        # size-verify, THEN rename — a torn transfer never gets the final
        # name, so the resume scan (final names only) rebuilds it.
        parts = [f"{o}.part" for o in outs]
        for o, p in zip(outs, parts):
            (work / o).replace(work / p)
        scp([*(str(work / p) for p in parts), f"{THOR}:{dst}/"])
        far = ssh(f"cd {dst} && ls -l {' '.join(parts)} | awk '{{print $5, $NF}}'")
        far_sz = {l.split()[1]: int(l.split()[0]) for l in far.splitlines()}
        for p in parts:
            local = (work / p).stat().st_size
            assert far_sz.get(p) == local, \
                f"SIZE MISMATCH {p}: {far_sz.get(p)} != {local}"
        ssh(f"cd {dst} && " + " && ".join(
            f"mv -f {p} {o}" for o, p in zip(outs, parts)))
        for o, p in zip(outs, parts):
            if keep_local:
                (work / p).replace(work / o)
            else:
                (work / p).unlink()
      except Exception as e:
        box["exc"] = e

    t0, built = time.time(), 0
    skipped: list[str] = []
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    pull_box: dict = {}
    puller = threading.Thread(target=_pull, args=(batches[0], pull_box))
    puller.start()
    pusher, push_box = None, {}
    for bi, batch in enumerate(batches):
        puller.join()
        if pull_box.get("exc") is not None:
            print(f"[{split}] pull retry for batch {bi} "
                  f"({type(pull_box['exc']).__name__})", flush=True)
            pull_box.clear()
            _pull(batch, pull_box)                 # one synchronous retry
            if pull_box.get("exc") is not None:
                print(f"[{split}] SKIP batch {bi}: {pull_box['exc']}",
                      flush=True)
                skipped.extend(batch)
                pull_box.clear()
                if bi + 1 < len(batches):
                    puller = threading.Thread(
                        target=_pull, args=(batches[bi + 1], pull_box))
                    puller.start()
                continue
        if bi + 1 < len(batches):
            pull_box.clear()
            puller = threading.Thread(target=_pull,
                                      args=(batches[bi + 1], pull_box))
            puller.start()
        outs = []
        for e in batch:
            try:
                o = torch.load(work / f"{e}.v2ep.pt", map_location="cpu",
                               weights_only=False)
            except (RuntimeError, FileNotFoundError):
                # a torn transfer: re-pull THIS episode once, then skip loudly
                try:
                    scp([f"{THOR}:{src}/{e}.v2ep.pt", str(work)])
                    o = torch.load(work / f"{e}.v2ep.pt", map_location="cpu",
                                   weights_only=False)
                except Exception as e2:
                    print(f"[{split}] SKIP {e}: {type(e2).__name__}",
                          flush=True)
                    skipped.append(e)
                    continue
            te = time.time()
            f = encode_episode(o, model, mean, std, n_special, a.device)
            if a.keep_fp16:
                torch.save(f.half(), work / f"{e}.fp16.pt")
            torch.save(f.to(torch.float8_e4m3fn), work / f"{e}.pt")
            outs.append(f"{e}.pt")
            if a.only:
                print(f"[{split}] {e} T={f.shape[0]} mean|x|="
                      f"{float(f.abs().mean()):.4f} {time.time()-te:.1f}s",
                      flush=True)
            else:
                (work / f"{e}.v2ep.pt").unlink()
        # push + size-verify BEHIND the next batch's encode (torn transfers
        # change the size; the loader's finite/content checks cover the rest)
        if pusher is not None:
            pusher.join()
            if push_box.get("exc") is not None:
                print(f"[{split}] PUSH FAILED (kept local for resume): "
                      f"{push_box['exc']}", flush=True)
                push_box.clear()
        if outs and not a.no_ship:
            push_box = {}
            pusher = threading.Thread(target=_push_verify,
                                      args=(outs, push_box))
            pusher.start()
        built += len(batch)
        rate = built / (time.time() - t0)
        print(f"[{split}] {built}/{len(todo)}  {rate*3600:.0f} eps/h  "
              f"eta {((len(todo)-built)/max(rate,1e-9))/3600:.1f} h", flush=True)

    if pusher is not None:
        pusher.join()
        if push_box.get("exc") is not None:
            print(f"[{split}] FINAL PUSH FAILED: {push_box['exc']}", flush=True)
            return 1
    if skipped:
        print(f"[{split}] {len(skipped)} episodes SKIPPED (rebuild on next "
              f"resume): {skipped[:8]}{'...' if len(skipped) > 8 else ''}",
              flush=True)
    if a.only or a.no_ship:
        print(f"[{split}] DONE {built} built in {(time.time()-t0)/60:.1f} min "
              f"(index.json untouched)")
        return 0
    geo = {"episodes": all_eps, "grid": "0.2s (every 2nd frame)",
           "dtype": "float8_e4m3fn",
           "geometry": {"n_tokens": 640, "d_enc": 1024, "hfov_deg": 120.0},
           "model": MID, "special_dropped": n_special,
           "gate": "fp8_l2_gate.json PASS 2026-09-01"}
    (work / "index.json").write_text(json.dumps(geo, indent=1))
    scp([str(work / "index.json"), f"{THOR}:{dst}/"])
    print(f"[{split}] DONE {built} built in {(time.time()-t0)/3600:.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
