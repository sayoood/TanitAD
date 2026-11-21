"""FULLRES encode (PREREG_V8F_PROOF_ADD9.md). Frozen DINOv3 ViT-L/16 on the 139 eval episodes at two resolutions,
channel-reduced (PCA 1024 -> 128, int8). GPU job: takes the shared dev-box GPU lock, releases it at the end.
No labels are read here. Resumable per episode."""
import glob, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

SRC = "E:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl"
CACHE = "E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval/clips"
OUT = "F:/TanitAD-scratch/v8proof/full"
HI_DIR = "C:/Users/Admin/v8proof_tmp/full"
MID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
LOCK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"
WAIT_IDLE = "E:/Projects/TanitAD-artifacts/_rlfan_work/wait_idle.py"
R = 128
MAX_USED_MIB = 3500
POOL = ThreadPoolExecutor(6)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.strip().splitlines()
    return int(out[0]) if out and out[0].strip().isdigit() else 10 ** 9


def take_lock(tag):
    content = tag + " pid=" + str(os.getpid()) + " t=" + time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.time()
    while True:
        subprocess.run([sys.executable, WAIT_IDLE], check=True)
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, content.encode("utf-8"))
            os.close(fd)
        except FileExistsError:
            if time.time() - t0 > 6 * 3600:
                raise SystemExit("ZZ lock wait timeout")
            log("lock held by another job, waiting")
            time.sleep(60)
            continue
        if used_mib() >= MAX_USED_MIB:
            release(content)
            time.sleep(60)
            continue
        return content


def release(content):
    try:
        with open(LOCK, encoding="utf-8") as f:
            same = f.read() == content
        if same:
            os.remove(LOCK)
    except FileNotFoundError:
        pass


def load_ep(path):
    o = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    assert str(o["codec"]) == "png" and int(o["image_h"]) == 416 and int(o["image_w"]) == 1024, path
    lens = o["jpeg_len"]
    offs = torch.cat([torch.zeros(1, dtype=lens.dtype), lens.cumsum(0)])
    return o, offs


def decode(o, offs, raw_idx):
    buf = o["jpeg_buf"]
    fr = list(POOL.map(lambda j: torchvision.io.decode_png(buf[int(offs[j]):int(offs[j + 1])].clone()), raw_idx))
    x = torch.stack(fr)
    assert x.shape[1:] == (3, 416, 1024) and x.dtype == torch.uint8, x.shape
    return x


def main():
    os.makedirs(OUT + "/feat", exist_ok=True)
    os.makedirs(HI_DIR, exist_ok=True)
    files = sorted(glob.glob(SRC + "/*.v2ep.pt"))
    assert len(files) == 139, len(files)
    clips, Ts = [], []
    for f in files:
        o, _ = load_ep(f)
        clips.append(str(o["clip_id"]))
        Ts.append(len(range(0, len(o["jpeg_len"]), 2)))
        assert os.path.basename(f).startswith(clips[-1]), f
    offs_rows = np.concatenate([[0], np.cumsum(Ts)]).astype(np.int64)
    N = int(offs_rows[-1])
    log("episodes", len(files), "grid frames", N)
    np.savez(OUT + "/feat/index.npz", clips=np.array(clips), T=np.array(Ts), row0=offs_rows[:-1], N=N)

    lock = take_lock("v8proof-fullres-encode")
    log("lock taken, used MiB", used_mib())
    try:
        from transformers import AutoImageProcessor, AutoModel
        dev = "cuda"
        proc = AutoImageProcessor.from_pretrained(MID)
        model = AutoModel.from_pretrained(MID, dtype=torch.bfloat16).eval().to(dev)
        mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1).to(dev, torch.bfloat16)
        std = torch.tensor(proc.image_std).view(1, 3, 1, 1).to(dev, torch.bfloat16)
        n_special = 1 + getattr(model.config, "num_register_tokens", 0)
        log("model loaded, special tokens", n_special)

        def enc(x_u8, lo, bs):
            out = []
            with torch.inference_mode():
                for j in range(0, x_u8.shape[0], bs):
                    x = x_u8[j:j + bs].to(dev)
                    if lo:
                        x = F.interpolate(x.float(), size=(260, 640), mode="bilinear", antialias=True, align_corners=False)
                        x = x[:, :, 2:258, :].clamp(0, 255)
                    x = x.to(torch.bfloat16) / 255.0
                    h = model(pixel_values=(x - mean) / std).last_hidden_state[:, n_special:, :]
                    out.append(h.float())
            h = torch.cat(out)
            assert h.shape[1:] == ((640, 1024) if lo else (1664, 1024)) and torch.isfinite(h).all(), h.shape
            return h

        # ---- pass 0: PCA basis (no labels) and control f2 (recipe against the cached 4x10 pool)
        pca_path = OUT + "/feat/pca.npz"
        if not os.path.exists(pca_path):
            s1 = torch.zeros(1024, dtype=torch.float64, device=dev)
            s2 = torch.zeros(1024, 1024, dtype=torch.float64, device=dev)
            cnt = 0
            f2cos, f2rel = [], []
            t0 = time.time()
            for i, f in enumerate(files):
                o, offs = load_ep(f)
                gs = [g for g in (10, 50, 90) if g < Ts[i]]
                x = decode(o, offs, [2 * g for g in gs])
                hl, hh = enc(x, True, 8), enc(x, False, 4)
                for h in (hl, hh):
                    t = h.reshape(-1, 1024).double()
                    s1 += t.sum(0)
                    s2 += t.T @ t
                    cnt += t.shape[0]
                c = torch.load(CACHE + "/" + clips[i] + ".pt", map_location="cpu", weights_only=False, mmap=True)["pool4x10"]
                for k, g in enumerate(gs):
                    if g not in (10, 90) or g >= c.shape[0]:
                        continue
                    mine = hl[k].reshape(4, 4, 10, 4, 1024).mean(dim=(1, 3)).reshape(40, 1024)
                    ref = c[g].to(torch.float32).to(dev)
                    f2cos.append(float(F.cosine_similarity(mine, ref, dim=1).mean()))
                    f2rel.append(float((mine - ref).norm() / ref.norm()))
            mu = s1 / cnt
            cov = s2 / cnt - torch.outer(mu, mu)
            ev, V = torch.linalg.eigh(cov)
            ev, V = ev.flip(0), V.flip(1)
            sigma = ev[:R].clamp_min(1e-12).sqrt()
            evr = float(ev[:R].sum() / ev.sum())
            np.savez(pca_path, mu=mu.cpu().numpy(), V=V[:, :R].cpu().numpy(), sigma=sigma.cpu().numpy(),
                     scale=(6.0 * sigma / 127.0).cpu().numpy(), evr=evr, n_tokens=cnt)
            f2 = {"n": len(f2cos), "cos_mean": float(np.mean(f2cos)), "cos_min": float(np.min(f2cos)),
                  "cos_p10": float(np.percentile(f2cos, 10)), "rel_l2_mean": float(np.mean(f2rel)), "bar": 0.85}
            f2["recipe_close"] = bool(f2["cos_mean"] >= 0.85)
            with open(OUT + "/feat/F2_RECIPE.json", "w", encoding="utf-8") as fh:
                json.dump(f2, fh, indent=1)
            log("pass 0 done", round(time.time() - t0), "s tokens", cnt, "explained variance of r", R, "=", round(evr, 4), "f2", f2)
        P = np.load(pca_path)
        mu = torch.tensor(P["mu"], dtype=torch.float32, device=dev)
        V = torch.tensor(P["V"], dtype=torch.float32, device=dev)
        scale = torch.tensor(P["scale"], dtype=torch.float32, device=dev)

        def reduce(h):
            z = ((h - mu) @ V) / scale
            return z.round().clamp(-127, 127).to(torch.int8).cpu().numpy()

        # ---- pass 1: all grid frames
        lo_path, hi_path = OUT + "/feat/LO_r128.i8", HI_DIR + "/HI_r128.i8"
        prog_path = OUT + "/feat/progress.json"
        done = []
        if os.path.exists(prog_path):
            with open(prog_path, encoding="utf-8") as fh:
                done = json.load(fh)
        mode = "r+" if (done and os.path.exists(lo_path) and os.path.exists(hi_path)) else "w+"
        if mode == "w+":
            done = []
        LO = np.memmap(lo_path, dtype=np.int8, mode=mode, shape=(N, 640, R))
        HI = np.memmap(hi_path, dtype=np.int8, mode=mode, shape=(N, 1664, R))
        t0 = time.time()
        for i, f in enumerate(files):
            if clips[i] in done:
                continue
            o, offs = load_ep(f)
            x = decode(o, offs, list(range(0, len(o["jpeg_len"]), 2)))
            assert x.shape[0] == Ts[i]
            r0 = int(offs_rows[i])
            LO[r0:r0 + Ts[i]] = reduce(enc(x, True, 16))
            for j in range(0, Ts[i], 24):
                HI[r0 + j:r0 + min(j + 24, Ts[i])] = reduce(enc(x[j:j + 24], False, 6))
            done.append(clips[i])
            if (i + 1) % 5 == 0 or i + 1 == len(files):
                LO.flush()
                HI.flush()
                with open(prog_path, "w", encoding="utf-8") as fh:
                    json.dump(done, fh)
                log("episode", i + 1, "of", len(files), "elapsed", round(time.time() - t0), "s used MiB", used_mib())
        LO.flush()
        HI.flush()
        nz_lo = float((np.asarray(LO[::97]) != 0).mean())
        nz_hi = float((np.asarray(HI[::97]) != 0).mean())
        assert nz_lo > 0.5 and nz_hi > 0.5, (nz_lo, nz_hi)
        with open(OUT + "/feat/ENCODE_DONE.json", "w", encoding="utf-8") as fh:
            json.dump({"N": N, "R": R, "lo": lo_path, "hi": hi_path, "nonzero_share_lo": nz_lo, "nonzero_share_hi": nz_hi,
                       "seconds": round(time.time() - t0), "evr": float(P["evr"])}, fh, indent=1)
        log("ZZ ENCODE-DONE", N, "frames")
    finally:
        release(lock)
        log("lock released")


if __name__ == "__main__":
    main()
