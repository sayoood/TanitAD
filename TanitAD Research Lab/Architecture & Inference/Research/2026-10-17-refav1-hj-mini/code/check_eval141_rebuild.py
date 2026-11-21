"""S4c completion check for the matched native 256x640 v2ep rebuild of the 141 EVAL clips. CPU only.

Exit 0 and token ZZEVALCHECK-OKZZ iff ALL of:
  C1 the set of <clip>.v2ep.pt in --dir == the clip ids of the banked native eval v2ep dir (141), exactly
  C2 every file passes the banked content verify (pod_pull_b1_epcache.verify) and has the SAME key set and
     geometry meta as the banked native eval file; n frames equal (jpeg_len/jpeg_buf are NOT gated: PNG bytes differ)
  C3 poses and actions are BIT-IDENTICAL to the banked file (torch.equal), for all 141
  C4 no _failures.json, and build.log ends `[build] DONE ok=141 fail=0`
Also MEASURES (reported, not gating): pixel difference vs the banked eval frames (PNG decode of N sampled frames per clip),
file sizes, and how many jpeg_buf are bit-equal. Writes <dir>/_manifest_sha256.jsonl and <stage>/check_report.json.
Run:  python check_eval141_rebuild.py [--dir D] [--workers 4] [--frames 9]
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, hashlib, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, "E:/Projects/TanitAD/stack"); sys.path.insert(0, "E:/Projects/TanitAD/stack/scripts")
BANKED = Path("C:/Users/Admin/tanitad-data/refav1-eval141/eps")
META = ("image_h", "image_w", "codec", "projection_mode", "n_stack", "frame", "episode_id", "clip_id")


def pixels(d, idx):
    import torch
    import torchvision.io as tvio
    lens = d["jpeg_len"]
    offs = torch.cat([torch.zeros(1, dtype=lens.dtype), lens.cumsum(0)])
    return [tvio.decode_png(d["jpeg_buf"][offs[j]:offs[j + 1]].clone()) for j in idx]


def one(args):
    path, banked_path, n_frames = args
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_available()
    import pod_pull_b1_epcache as PP
    ok, why = PP.verify(path)
    clip = os.path.basename(path)[:-len(".v2ep.pt")]
    r = {"clip": clip, "verify_ok": ok, "why": why, "bytes": os.path.getsize(path), "bytes_banked": os.path.getsize(banked_path)}
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    r["sha256"] = h.hexdigest()
    if not ok:
        return r
    a = torch.load(path, map_location="cpu", weights_only=False)
    b = torch.load(banked_path, map_location="cpu", weights_only=False)
    r["keys_equal"] = sorted(a) == sorted(b)
    r["n"], r["n_banked"] = int(a["jpeg_len"].shape[0]), int(b["jpeg_len"].shape[0])
    r["n_equal"] = r["n"] == r["n_banked"]
    r["meta_equal"] = all(a.get(k) == b.get(k) for k in META)
    r["meta_diff"] = [k for k in META if a.get(k) != b.get(k)]
    for k in ("poses", "actions"):
        r[f"{k}_bit_equal"] = bool(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and torch.equal(a[k], b[k]))
    r["jpeg_len_equal"] = bool(torch.equal(a["jpeg_len"], b["jpeg_len"]))
    r["jpeg_buf_bit_equal"] = bool(a["jpeg_buf"].shape == b["jpeg_buf"].shape and torch.equal(a["jpeg_buf"], b["jpeg_buf"]))
    if r["n_equal"]:
        n = r["n"]
        idx = sorted({int(round(i * (n - 1) / max(n_frames - 1, 1))) for i in range(n_frames)})
        pa, pb = pixels(a, idx), pixels(b, idx)
        d = [(x.int() - y.int()).abs() for x, y in zip(pa, pb)]
        r["px_n_frames"] = len(idx)
        r["px_mae"] = float(sum(float(x.float().mean()) for x in d) / len(d))
        r["px_max"] = int(max(int(x.max()) for x in d))
        r["px_frac_nonzero"] = float(sum(float((x > 0).float().mean()) for x in d) / len(d))
        r["px_sum_abs"] = float(sum(float(x.double().sum()) for x in d))
        r["px_n_values"] = int(sum(x.numel() for x in d))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="C:/Users/Admin/tanitad-data/refav1-eval141-256rb")
    ap.add_argument("--stage", default="C:/Users/Admin/tanitad-data/refav1-eval141-256rb_stage")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--frames", type=int, default=9)
    a = ap.parse_args()
    d = Path(a.dir)
    banked = sorted(p.name[:-len(".v2ep.pt")] for p in BANKED.glob("*.v2ep.pt"))
    have = sorted(p.name[:-len(".v2ep.pt")] for p in d.glob("*.v2ep.pt"))
    rep = {"n_have": len(have), "n_banked": len(banked)}
    rep["C1_set_equals_banked_eval141"] = have == banked and len(banked) == 141
    rep["C1_missing"] = sorted(set(banked) - set(have))[:10]
    rep["C1_extra"] = sorted(set(have) - set(banked))[:10]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(one, [(str(d / f"{c}.v2ep.pt"), str(BANKED / f"{c}.v2ep.pt"), a.frames) for c in have], chunksize=2))
    rep["check_s"] = round(time.time() - t0, 1)
    # jpeg_len (per-frame PNG byte lengths) is NOT gated: the swscale residue changes the pixels, hence the compressed sizes
    # (measured on clip 126b3a97: jpeg_len differs, n equal). It is counted and reported below.
    c2 = [r for r in res if not (r["verify_ok"] and r.get("keys_equal") and r.get("n_equal") and r.get("meta_equal"))]
    c3 = [r for r in res if not (r.get("poses_bit_equal") and r.get("actions_bit_equal"))]
    rep["C2_n_bad"], rep["C2_bad"] = len(c2), [{k: v for k, v in r.items() if k != "sha256"} for r in c2[:5]]
    rep["C3_n_not_bit_equal"], rep["C3_bad"] = len(c3), [r["clip"] for r in c3[:5]]
    rep["n_jpeg_buf_bit_equal"] = sum(bool(r.get("jpeg_buf_bit_equal")) for r in res)
    rep["n_jpeg_len_equal"] = sum(bool(r.get("jpeg_len_equal")) for r in res)
    px = [r for r in res if "px_mae" in r]
    if px:
        tot_abs = sum(r["px_sum_abs"] for r in px); tot_v = sum(r["px_n_values"] for r in px)
        maes = sorted(r["px_mae"] for r in px)
        rep["pixel_diff_vs_banked"] = {
            "n_clips": len(px), "frames_per_clip": a.frames, "n_frames": sum(r["px_n_frames"] for r in px),
            "pooled_mae_of_255": tot_abs / tot_v, "per_clip_mae_min": maes[0], "per_clip_mae_median": maes[len(maes) // 2],
            "per_clip_mae_max": maes[-1], "max_abs_level": max(r["px_max"] for r in px),
            "mean_frac_pixels_differing": sum(r["px_frac_nonzero"] for r in px) / len(px)}
    rep["total_bytes"] = sum(r["bytes"] for r in res)
    rep["total_bytes_banked"] = sum(r["bytes_banked"] for r in res)
    rep["total_gb"] = round(rep["total_bytes"] / 1e9, 3)
    rep["total_gb_banked"] = round(rep["total_bytes_banked"] / 1e9, 3)
    rep["n_frames_set"] = sorted({r.get("n") for r in res})
    with open(d / "_manifest_sha256.jsonl", "w", encoding="utf-8") as fh:
        for r in res:
            fh.write(json.dumps({"clip": r["clip"], "bytes": r["bytes"], "sha256": r["sha256"]}) + "\n")
    log = Path(a.stage) / "build.log"
    tail = log.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[-3:] if log.exists() else []
    rep["C4_no_failures_json"] = not (d / "_failures.json").exists()
    rep["C4_done_line"] = next((ln for ln in tail if ln.startswith("[build] DONE")), None)
    rep["C4_done_ok141"] = bool(rep["C4_done_line"] and "ok=141 fail=0" in rep["C4_done_line"])
    rep["ALL_OK"] = bool(rep["C1_set_equals_banked_eval141"] and rep["C2_n_bad"] == 0 and rep["C3_n_not_bit_equal"] == 0
                         and rep["C4_no_failures_json"] and rep["C4_done_ok141"])
    (Path(a.stage) / "check_report.json").write_text(json.dumps({**rep, "per_clip": res}, indent=1, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("C2_bad",)}, indent=1, default=str))
    print("ZZEVALCHECK-OKZZ" if rep["ALL_OK"] else "ZZEVALCHECK-FAILZZ", flush=True)
    return 0 if rep["ALL_OK"] else 1


if __name__ == "__main__":
    sys.exit(main())
