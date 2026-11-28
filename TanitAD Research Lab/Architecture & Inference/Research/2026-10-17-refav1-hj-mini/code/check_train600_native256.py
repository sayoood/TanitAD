"""S4b completion check for the native 256x640 v2ep rebuild of the 600 stage-A TRAIN clips. CPU only.

Exit 0 and token ZZCHECK-OKZZ iff ALL of:
  C1 the set of <clip>.v2ep.pt in --dir == the picker's 600 (pull_encode_pooled.pick_clips, train, n=600) EXACTLY
  C2 that set is disjoint from the eval-141 set (local native eval eps dir)
  C3 every file passes the banked content verify (pod_pull_b1_epcache.verify: keys, n, 256x640, png, cylindrical,
     n_stack 3, jpeg_buf length, frame 0 decodes non-zero) AND has the SAME key set / dtypes / geometry meta as an eval
     native file; n frames == poses == actions == jpeg_len
  C4 no _failures.json, build.log ends with `[build] DONE ok=600 fail=0`
Writes <dir>/_manifest_sha256.jsonl (clip, bytes, sha256) so the copy on the encode machine can be verified, and
<stage>/check_report.json. Run:  python check_train600_native256.py [--dir D] [--workers 4]
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, hashlib, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, "E:/Projects/TanitAD/stack"); sys.path.insert(0, "E:/Projects/TanitAD/stack/scripts")

EXPECT_KEYS = None


def one(args):
    path, ref_keys = args
    import torch
    torch.set_num_threads(1)
    import pod_pull_b1_epcache as PP
    ok, why = PP.verify(path)
    r = {"clip": os.path.basename(path)[:-len(".v2ep.pt")], "ok": ok, "why": why, "bytes": os.path.getsize(path)}
    if ok:
        d = torch.load(path, map_location="cpu", weights_only=False)
        n = int(d["jpeg_len"].shape[0])
        r["n"] = n
        r["keys_equal_eval"] = sorted(d) == ref_keys
        r["shape_ok"] = (d["poses"].shape == (n, 4) and d["actions"].shape == (n, 2) and d["poses"].dtype == torch.float32
                         and d["actions"].dtype == torch.float32 and d["jpeg_len"].dtype == torch.int64
                         and d["jpeg_buf"].dtype == torch.uint8)
        r["finite"] = bool(torch.isfinite(d["poses"]).all() and torch.isfinite(d["actions"]).all())
        r["frame_ok"] = d["frame"] == {"height": 256, "width": 640, "f_ref": 305.5774907364391, "projection": "cylindrical"}
        r["clip_id_ok"] = d["clip_id"] == r["clip"]
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    r["sha256"] = h.hexdigest()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="C:/Users/Admin/tanitad-data/refav1-train600-256")
    ap.add_argument("--stage", default="C:/Users/Admin/tanitad-data/refav1-train600-256_stage")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import torch
    import pull_encode_pooled as P
    from types import SimpleNamespace
    ns = SimpleNamespace(eval_eps=P.D_EVAL_EPS, eval_cache=P.D_EVAL_CACHE, eval_labels=P.D_EVAL_LABELS,
                         train_labels=P.D_TRAIN_LABELS, listing=P.D_INDEX, thor=P.THOR)
    sets = P.clip_sets(ns)
    pick = P.pick_clips(sets, "train", 600)
    d = Path(a.dir)
    have = sorted(p.name[:-len(".v2ep.pt")] for p in d.glob("*.v2ep.pt"))
    rep = {"n_have": len(have), "n_pick": len(pick)}
    rep["C1_set_equals_pick"] = sorted(have) == sorted(pick)
    rep["C1_missing"] = sorted(set(pick) - set(have))[:10]
    rep["C1_extra"] = sorted(set(have) - set(pick))[:10]
    rep["C2_overlap_eval141"] = len(set(have) & set(sets["eval"]))
    ev0 = sorted(Path(P.D_EVAL_EPS).glob("*.v2ep.pt"))[0]
    ref = torch.load(ev0, map_location="cpu", weights_only=False)
    ref_keys = sorted(ref)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(one, [(str(d / f"{c}.v2ep.pt"), ref_keys) for c in have], chunksize=4))
    rep["check_s"] = round(time.time() - t0, 1)
    bad = [r for r in res if not (r["ok"] and r.get("keys_equal_eval") and r.get("shape_ok") and r.get("finite")
                                  and r.get("frame_ok") and r.get("clip_id_ok"))]
    rep["C3_n_bad"] = len(bad)
    rep["C3_bad"] = bad[:5]
    rep["n_frames_set"] = sorted({r.get("n") for r in res})
    rep["total_bytes"] = sum(r["bytes"] for r in res)
    rep["total_gb"] = round(rep["total_bytes"] / 1e9, 2)
    with open(d / "_manifest_sha256.jsonl", "w", encoding="utf-8") as fh:
        for r in res:
            fh.write(json.dumps({"clip": r["clip"], "bytes": r["bytes"], "sha256": r["sha256"]}) + "\n")
    log = Path(a.stage) / "build.log"
    tail = log.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[-3:] if log.exists() else []
    rep["C4_no_failures_json"] = not (d / "_failures.json").exists()
    rep["C4_done_line"] = next((l for l in tail if l.startswith("[build] DONE")), None)
    rep["C4_done_ok600"] = bool(rep["C4_done_line"] and "ok=600 fail=0" in rep["C4_done_line"])
    allok = (rep["C1_set_equals_pick"] and rep["C2_overlap_eval141"] == 0 and rep["C3_n_bad"] == 0
             and rep["C4_no_failures_json"] and rep["C4_done_ok600"])
    rep["ALL_OK"] = bool(allok)
    (Path(a.stage) / "check_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k != "C3_bad"}, indent=1, default=str))
    print("ZZCHECK-OKZZ" if allok else "ZZCHECK-FAILZZ", flush=True)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
