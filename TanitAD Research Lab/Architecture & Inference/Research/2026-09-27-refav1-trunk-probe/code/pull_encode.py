"""SPEC A2 (refav1 trunk probe): pull N refav1-TRAIN episodes from Thor (READ-ONLY) and encode them exactly as the
refav1 fp8 cache was built -- `dinov3_fp8_encode_ship.encode_episode` imported, bf16 on the 4060, saved as
float8_e4m3fn -- into a LOCAL cache. Nothing is written on Thor. DINOv3-L is loaded from the local HF cache
(local_files_only), so no token file is read.

Train set = Thor's B1 source listing minus the 141 eval clips, checked equal to the v7.2 TRAIN label release's clip set;
the first N by sha12 order are taken. Opaque ZZ tokens; every output file is content-asserted.
"""
import argparse, gzip, hashlib, importlib.util, json, re, subprocess, sys, time
from pathlib import Path

import torch

THOR = "tanitad-thor-wifi"
SRC = "/home/nvidia/data/physicalai-b1-w120-256x640cyl"
MID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def sha12(u):
    return hashlib.sha256(u.encode()).hexdigest()[:12]


def ssh(cmd):
    r = subprocess.run(["ssh", "-n", "-o", "ConnectTimeout=25", "-o", "BatchMode=yes", THOR, cmd],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"ssh failed rc={r.returncode}: {r.stderr[-300:]}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--work", default="C:/Users/Admin/refav1_trainprobe")
    ap.add_argument("--eval-eps", default="C:/Users/Admin/tanitad-data/refav1-eval141/eps")
    ap.add_argument("--train-labels", required=True)
    ap.add_argument("--encoder-module", default="C:/Users/Admin/tipsnap/c36b6ddd/stack/scripts/dinov3_fp8_encode_ship.py")
    a = ap.parse_args()
    work = Path(a.work); (work / "eps").mkdir(parents=True, exist_ok=True); (work / "fp8").mkdir(exist_ok=True)
    listing = sorted(l.strip()[:-len(".v2ep.pt")] for l in ssh(f"ls {SRC}").splitlines() if l.endswith(".v2ep.pt"))
    ev = sorted(p.name[:-len(".v2ep.pt")] for p in Path(a.eval_eps).glob("*.v2ep.pt"))
    train = sorted(set(listing) - set(ev))
    lab = set(UUID.findall(gzip.decompress(Path(a.train_labels).read_bytes()).decode("utf-8", "replace")))
    print(f"ZZLIST thor={len(listing)} eval={len(ev)} eval_in_thor={len(set(ev) & set(listing))} train={len(train)} "
          f"labels_train_ids={len(lab)} train_eq_labels={set(train) == lab} train_minus_labels={len(set(train) - lab)} "
          f"labels_minus_train={len(lab - set(train))}ZZ", flush=True)
    if len(set(ev) & set(listing)) != len(ev):
        raise SystemExit("eval clips missing from Thor's listing -- the split is not what this probe assumes")
    if set(ev) & lab:
        raise SystemExit("an EVAL clip appears in the TRAIN labels -- refusing")
    pick = sorted(train, key=sha12)[: a.n]
    (work / "train_pick_sha12.json").write_text(json.dumps(
        {"n": len(pick), "rule": "first N of (B1 listing - eval141) by sha256(clip)[:12]",
         "sha12": [sha12(c) for c in pick]}, indent=1), encoding="utf-8")
    # encoder: the EXACT encode path, imported; model from the local HF cache only
    spec = importlib.util.spec_from_file_location("enc", a.encoder_module)
    enc = importlib.util.module_from_spec(spec); spec.loader.exec_module(enc)
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(MID, local_files_only=True)
    model = AutoModel.from_pretrained(MID, local_files_only=True, dtype=torch.bfloat16).eval().to("cuda")
    mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1).to("cuda", torch.bfloat16)
    std = torch.tensor(proc.image_std).view(1, 3, 1, 1).to("cuda", torch.bfloat16)
    n_special = 1 + getattr(model.config, "num_register_tokens", 0)
    t0 = time.time(); done = skipped = 0
    for i, c in enumerate(pick):
        fp8 = work / "fp8" / f"{c}.pt"; ep = work / "eps" / f"{c}.v2ep.pt"
        if fp8.exists() and ep.exists() and fp8.stat().st_size > 10_000_000:
            done += 1; continue
        r = subprocess.run(["scp", "-q", "-o", "ConnectTimeout=25", "-o", "BatchMode=yes",
                            f"{THOR}:{SRC}/{c}.v2ep.pt", str(ep)], capture_output=True, text=True, timeout=600)
        try:
            o = torch.load(ep, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"ZZSKIP {sha12(c)} {type(e).__name__} scp_rc={r.returncode}ZZ", flush=True); skipped += 1
            continue
        f = enc.encode_episode(o, model, mean, std, n_special, "cuda")
        q = f.to(torch.float8_e4m3fn)
        assert float(q.float().abs().mean()) > 1e-3               # content, not zeros
        tmp = fp8.with_suffix(".pt.part"); torch.save(q, tmp); tmp.replace(fp8)
        done += 1
        if i < 3 or i % 25 == 0:
            el = time.time() - t0
            print(f"[pull_encode] {i+1}/{len(pick)} {sha12(c)} T={f.shape[0]} {el:.0f}s", flush=True)
    print(f"ZZENC-DONE done={done} skipped={skipped} {time.time()-t0:.0f}sZZ", flush=True)


if __name__ == "__main__":
    main()
