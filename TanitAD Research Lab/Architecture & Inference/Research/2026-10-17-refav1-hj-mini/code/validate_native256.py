"""S4b priority 1: prove the banked builder (`stack/scripts/pod_build_b1_epcache.py --contract-test`) reproduces the
native 256x640 v2ep format on THIS box (CPU), by rebuilding EVAL clips whose banked native v2ep sits in
`C:/Users/Admin/tanitad-data/refav1-eval141/eps` and comparing field by field, bit for bit.

The two eval clips are staged from the local PhysicalAI copies (the same bytes the train staging verified against HF).
Also builds `--train-clips` (default 2) from the real train stage and compares their stats/format against the eval ones.
Usage: python validate_native256.py [--n-eval 2] [--train-clips 2]
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, hashlib, json, shutil, subprocess, sys, time, zipfile
from pathlib import Path

import numpy as np, torch
assert not torch.cuda.is_available(), "CPU only"
import torchvision.io as tvio

STACK = "E:/Projects/TanitAD/stack"
PY = sys.executable
BUILDER = f"{STACK}/scripts/pod_build_b1_epcache.py"
LOCAL = Path("C:/Users/Admin/tanitad-data/physicalai")
EVAL_EPS = Path("C:/Users/Admin/tanitad-data/refav1-eval141/eps")
STAGE = Path("C:/Users/Admin/tanitad-data/refav1-train600-256_stage")
VAL = STAGE / "_valroot"


def stage_eval(clips):
    cam = VAL / "r0" / "camera_front_wide"
    (VAL / "labels" / "egomotion").mkdir(parents=True, exist_ok=True)
    (VAL / "calibration").mkdir(parents=True, exist_ok=True)
    cam.mkdir(parents=True, exist_ok=True)
    zp = VAL / "labels" / "egomotion" / "egomotion_all.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for c in clips:
            z.write(LOCAL / "labels" / "egomotion_alpamayo" / f"{c}.parquet", f"{c}.egomotion.parquet")
    for c in clips:
        src = LOCAL / "camera" / "camera_front_wide_120fov" / f"{c}.mp4"
        if not (cam / f"{c}.mp4").exists():
            try:
                os.link(src, cam / f"{c}.mp4")
            except OSError:
                shutil.copy2(src, cam / f"{c}.mp4")
        shutil.copy2(LOCAL / "camera" / "camera_front_wide_120fov" / f"{c}.timestamps.parquet", cam)
    shutil.copy2(LOCAL / "calibration" / "physicalai_front_wide_intrinsics.csv", VAL / "calibration")


def build(root, out, clip):
    env = dict(os.environ, TANITAD_STACK=STACK, V2_TORCH_THREADS="1", PAI_DECODE_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1")
    t0 = time.time()
    r = subprocess.run([PY, BUILDER, "--out", str(out), "--root", str(root), "--contract-test", clip],
                       capture_output=True, text=True, env=env, encoding="utf-8")
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:], time.time() - t0


def cmp(a, b):
    """Field-by-field comparison of two v2ep dicts. Returns a dict of results."""
    out = {"keys_equal": sorted(a) == sorted(b), "keys_only_a": sorted(set(a) - set(b)), "keys_only_b": sorted(set(b) - set(a))}
    fields = {}
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        if isinstance(x, torch.Tensor):
            same_meta = x.dtype == y.dtype and tuple(x.shape) == tuple(y.shape)
            eq = bool(same_meta and torch.equal(x, y))
            fields[k] = {"dtype": str(x.dtype), "shape": list(x.shape), "dtype_shape_equal": same_meta, "bit_equal": eq}
            if same_meta and not eq and x.is_floating_point():
                fields[k]["max_abs_diff"] = float((x.double() - y.double()).abs().max())
        else:
            fields[k] = {"equal": x == y, "a": str(x)[:120], "b": str(y)[:120]}
    out["fields"] = fields
    return out


def png_pixels(d, n_idx):
    lens = d["jpeg_len"]
    offs = torch.cat([torch.zeros(1, dtype=lens.dtype), lens.cumsum(0)])
    return [tvio.decode_png(d["jpeg_buf"][offs[j]:offs[j + 1]].clone()) for j in n_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=2)
    ap.add_argument("--train-clips", type=int, default=2)
    ap.add_argument("--report", default=str(STAGE / "validation_native256.json"))
    a = ap.parse_args()
    ev = sorted(p.name[:-len(".v2ep.pt")] for p in EVAL_EPS.glob("*.v2ep.pt"))
    ev_pick = ev[:a.n_eval]
    stage_eval(ev_pick)
    rep = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
           "eval_clips": ev_pick, "eval": {}, "train": {}}
    for c in ev_pick:
        outd = VAL / "out"
        outd.mkdir(exist_ok=True)
        msg, el = build(VAL, outd, c)
        mine = torch.load(outd / f"{c}.v2ep.pt", map_location="cpu", weights_only=False)
        banked = torch.load(EVAL_EPS / f"{c}.v2ep.pt", map_location="cpu", weights_only=False)
        r = cmp(mine, banked)
        r["builder_msg"], r["build_s"] = msg, round(el, 1)
        r["file_bytes"] = {"mine": (outd / f"{c}.v2ep.pt").stat().st_size, "banked": (EVAL_EPS / f"{c}.v2ep.pt").stat().st_size}
        r["file_sha256_equal"] = (hashlib.sha256((outd / f"{c}.v2ep.pt").read_bytes()).hexdigest()
                                  == hashlib.sha256((EVAL_EPS / f"{c}.v2ep.pt").read_bytes()).hexdigest())
        if not r["fields"]["jpeg_buf"].get("bit_equal"):   # decode and compare pixels of all (even) frames
            n = int(mine["jpeg_len"].shape[0])
            idx = list(range(0, n, max(1, n // 12)))
            pm, pb = png_pixels(mine, idx), png_pixels(banked, idx)
            dif = [(x.int() - y.int()).abs() for x, y in zip(pm, pb)]
            r["pixel_diff_sampled"] = {"n_frames": len(idx), "max": int(max(d.max() for d in dif)),
                                       "mean": float(np.mean([d.float().mean() for d in dif])),
                                       "frac_nonzero": float(np.mean([(d > 0).float().mean() for d in dif]))}
        rep["eval"][c] = r
        print(f"[val] eval {c[:8]}: {msg} {el:.0f}s  keys_equal={r['keys_equal']} jpeg_buf bit_equal="
              f"{r['fields']['jpeg_buf'].get('bit_equal')} poses={r['fields']['poses'].get('bit_equal')} "
              f"actions={r['fields']['actions'].get('bit_equal')} file_sha256_equal={r['file_sha256_equal']}", flush=True)
    # train clips from the real stage: format/stats vs an eval native file
    tr = (STAGE / "clips600.txt").read_text().split()[:a.train_clips]
    ref = torch.load(EVAL_EPS / f"{ev_pick[0]}.v2ep.pt", map_location="cpu", weights_only=False)
    outd = STAGE / "_valtrain"
    outd.mkdir(exist_ok=True)
    for c in tr:
        msg, el = build(STAGE, outd, c)
        d = torch.load(outd / f"{c}.v2ep.pt", map_location="cpu", weights_only=False)
        r = cmp(d, ref)
        r = {k: r[k] for k in ("keys_equal", "keys_only_a", "keys_only_b")}
        r.update(builder_msg=msg, build_s=round(el, 1), file_bytes=(outd / f"{c}.v2ep.pt").stat().st_size,
                 n=int(d["jpeg_len"].shape[0]), image_hw=[d["image_h"], d["image_w"]], codec=d["codec"],
                 projection_mode=d["projection_mode"], n_stack=d["n_stack"], frame=d["frame"],
                 poses_dtype=str(d["poses"].dtype), actions_dtype=str(d["actions"].dtype),
                 poses_finite=bool(torch.isfinite(d["poses"]).all()), actions_finite=bool(torch.isfinite(d["actions"]).all()),
                 v_range=[float(d["poses"][:, 3].min()), float(d["poses"][:, 3].max())],
                 steer_range=[float(d["actions"][:, 0].min()), float(d["actions"][:, 0].max())],
                 frame0_max=int(png_pixels(d, [0])[0].max()))
        rep["train"][c] = r
        print(f"[val] train {c[:8]}: {msg} {el:.0f}s n={r['n']} bytes={r['file_bytes']} keys_equal_to_eval={r['keys_equal']}", flush=True)
    rep["eval_ref_frame"] = ref["frame"]
    Path(a.report).write_text(json.dumps(rep, indent=1, default=str))
    print("ZZVALIDATE-DONEZZ", flush=True)


if __name__ == "__main__":
    main()
