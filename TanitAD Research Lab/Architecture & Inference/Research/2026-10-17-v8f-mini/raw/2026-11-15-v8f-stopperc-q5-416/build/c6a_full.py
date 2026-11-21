"""c6a_full.py -- Amendment 7 control C6a as amended by 7a, over every eval clip (MM, 2026-11-15).  CPU, dev box.

For one clip: the pod-built 416 eval v2ep (pulled to a temp file by c6a_full.sh) against E:'s copy in v2ep-eval139-416x1024cyl.
  * decoded uint8 frames: max |diff|, number of differing values, number of values (7a: max |diff| <= 1 count; the fraction is reported);
  * every other record in the file (n_frames, poses, actions, the frame record, ...) bitwise equal (as C6c).
Appends one JSON line per clip to --out.  `summarise` folds the lines into the control verdict.
Usage: python c6a_full.py one --clip C --pod-file P --e-dir D --out X.jsonl
       python c6a_full.py summarise --out X.jsonl --json RESULT.json
"""
import argparse
import json
import os
import sys


def load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def frames(o):
    import numpy as np
    import torch
    import torchvision
    lens, buf = o["jpeg_len"], o["jpeg_buf"]
    offs = torch.cat([torch.zeros(1, dtype=lens.dtype), lens.cumsum(0)])
    for j in range(len(lens)):
        if int(lens[j]) == 0:
            yield j, None
            continue
        yield j, torchvision.io.decode_png(buf[offs[j]:offs[j + 1]].clone()).numpy().astype(np.int16)


def same(a, b):
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def cmd_one(a):
    import numpy as np
    p, e = load(a.pod_file), load(os.path.join(a.e_dir, a.clip + ".v2ep.pt"))
    rec = {"clip": a.clip, "keys_pod_only": sorted(set(p) - set(e)), "keys_e_only": sorted(set(e) - set(p))}
    other = sorted((set(p) & set(e)) - {"jpeg_len", "jpeg_buf"})
    rec["records_not_bitwise"] = [k for k in other if not same(p[k], e[k])]
    rec["n_frames"] = [int(len(p["jpeg_len"])), int(len(e["jpeg_len"]))]
    mx, nd, nv, nfr, empty_mismatch, shape_mismatch = 0, 0, 0, 0, 0, 0
    fe = dict(frames(e))
    for j, fp in frames(p):
        fq = fe.get(j)
        if (fp is None) != (fq is None):
            empty_mismatch += 1
            continue
        if fp is None:
            continue
        if fp.shape != fq.shape:
            shape_mismatch += 1
            continue
        d = np.abs(fp - fq)
        mx, nd, nv, nfr = max(mx, int(d.max())), nd + int((d > 0).sum()), nv + int(d.size), nfr + 1
    rec.update({"max_abs_diff": mx, "n_values_differ": nd, "n_values": nv, "n_frames_compared": nfr, "empty_frame_mismatch": empty_mismatch, "shape_mismatch": shape_mismatch})
    with open(a.out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec[k] for k in ("clip", "max_abs_diff", "n_values_differ", "n_values", "records_not_bitwise")}), flush=True)
    return 0


def cmd_summarise(a):
    rows = {}
    for line in open(a.out, encoding="utf-8"):
        r = json.loads(line)
        rows[r["clip"]] = r                                             # a re-run of one clip keeps its last line
    R = list(rows.values())
    n_val = sum(r["n_values"] for r in R)
    res = {"control": "C6a (Amendment 7a)", "n_clips": len(R),
           "max_abs_diff": max(r["max_abs_diff"] for r in R), "n_values": n_val, "n_values_differ": sum(r["n_values_differ"] for r in R),
           "fraction_values_differ": (sum(r["n_values_differ"] for r in R) / n_val) if n_val else None,
           "n_frames_compared": sum(r["n_frames_compared"] for r in R),
           "clips_records_not_bitwise": {r["clip"]: r["records_not_bitwise"] for r in R if r["records_not_bitwise"]},
           "clips_key_set_differs": [r["clip"] for r in R if r["keys_pod_only"] or r["keys_e_only"]],
           "clips_n_frames_differ": [r["clip"] for r in R if r["n_frames"][0] != r["n_frames"][1]],
           "frame_mismatches": sum(r["empty_frame_mismatch"] + r["shape_mismatch"] for r in R)}
    res["PASS"] = bool(res["max_abs_diff"] <= 1 and not res["clips_records_not_bitwise"] and not res["clips_key_set_differs"]
                       and not res["clips_n_frames_differ"] and res["frame_mismatches"] == 0 and len(R) >= a.min_clips)
    res["bar"] = f"max |diff| <= 1 count; every non-frame record bitwise; n_clips >= {a.min_clips}"
    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
    return 0 if res["PASS"] else 1


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    o = sp.add_parser("one")
    o.add_argument("--clip", required=True); o.add_argument("--pod-file", required=True); o.add_argument("--e-dir", required=True); o.add_argument("--out", required=True)
    s = sp.add_parser("summarise")
    s.add_argument("--out", required=True); s.add_argument("--json", required=True); s.add_argument("--min-clips", type=int, default=139)
    a = ap.parse_args()
    return cmd_one(a) if a.cmd == "one" else cmd_summarise(a)


if __name__ == "__main__":
    sys.exit(main())
