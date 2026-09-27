"""Stream the kit's train+eval agent joins ONCE and write the EVAL-ONLY line subsets (byte-identical lines).

Why: a full parse of the 327 MB / 409 MB .jsonl.xz files took 77 s / 103 s on a quiet box (boxes render,
render_record.json) and did not finish in 7 min under tonight's contention. The subset is selected by
the line's own "clip_id" value against the eval cache's clip set; every kept line is written unchanged.

Memory: one line at a time (~50 MB RSS). The source md5 is computed in the SAME pass.
Known-value control, from the boxes render's full-file load of the same 3 clips: 2-D 600 records,
3-D 600 lines and 78,336 agents. The script prints them for the 3 fixed clips.
Output (work dir, NEVER banked -- the lines carry raw clip ids): eval_agents.jsonl, eval_agents_3d.jsonl,
extract_record.json (sha12-only).
"""
import hashlib
import json
import lzma
import sys
import time
from pathlib import Path

KIT = Path("D:/refcv6_eval_kit/data")
SRC = {"2d": KIT / "joins/b1_train_plus_eval_agents.jsonl.xz",
       "3d": KIT / "join3d/b1_train_plus_eval_agents_3d.jsonl.xz"}
OUT = Path("C:/Users/Admin/qland/work/gtval")
DST = {"2d": OUT / "eval_agents.jsonl", "3d": OUT / "eval_agents_3d.jsonl"}
FIXED = ("0191487845ef", "9f8bedcfb9de", "34765c024267")
KNOWN = {"2d_records_3_fixed": 600, "3d_lines_3_fixed": 600, "3d_agents_3_fixed": 78336}


def sha12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


class HashingReader:
    def __init__(self, f):
        self.f, self.h, self.n = f, hashlib.md5(), 0

    def read(self, n=-1):
        b = self.f.read(n)
        self.h.update(b)
        self.n += len(b)
        return b

    def readable(self):
        return True


def clip_of_line(line: bytes):
    k = line.find(b'"clip_id"', 0, 400)
    if k < 0:
        k = line.find(b'"clip_id"')
        if k < 0:
            return None
    q1 = line.find(b'"', line.find(b":", k) + 1)
    q2 = line.find(b'"', q1 + 1)
    return line[q1 + 1:q2].decode("ascii")


def main(which):
    eval_ids = {p.name.split(".")[0] for p in (KIT / "refcv6-b1-416x1024-eval139").glob("*.v2ep.pt")}
    fixed_ids = {c for c in eval_ids if sha12(c) in FIXED}
    if len(fixed_ids) != 3:
        raise SystemExit(f"fixed clips found {len(fixed_ids)} of 3 in the eval cache")
    rec = {"n_eval_clips": len(eval_ids)}
    for tag in which:
        t0 = time.time()
        n_in = n_out = n_fixed = n_fixed_agents = 0
        per_clip = {}
        with open(SRC[tag], "rb") as raw:
            hr = HashingReader(raw)
            with lzma.open(hr, "rb") as src, open(DST[tag], "wb") as dst:
                for line in src:
                    n_in += 1
                    cid = clip_of_line(line)
                    if cid is None or cid not in eval_ids:
                        continue
                    dst.write(line)
                    n_out += 1
                    per_clip[cid] = per_clip.get(cid, 0) + 1
                    if cid in fixed_ids:
                        n_fixed += 1
                        if tag == "3d":
                            n_fixed_agents += len(json.loads(line)["agents"])
            while hr.read(1 << 22):          # hash any tail the decompressor did not pull
                pass
        rec[tag] = {"source": str(SRC[tag]), "source_bytes": hr.n, "source_md5": hr.h.hexdigest(),
                    "subset": str(DST[tag]), "subset_bytes": DST[tag].stat().st_size,
                    "n_lines_in": n_in, "n_lines_kept": n_out, "n_clips_kept": len(per_clip),
                    "n_lines_3_fixed": n_fixed, "wall_s": round(time.time() - t0, 1)}
        if tag == "3d":
            rec[tag]["n_agents_3_fixed"] = n_fixed_agents
        print(f"[extract] {tag}: {n_in:,} lines in, {n_out:,} kept over {len(per_clip)} clips; 3 fixed clips "
              f"{n_fixed} lines" + (f", {n_fixed_agents:,} agents" if tag == "3d" else "")
              + f"; md5 {rec[tag]['source_md5']}; {rec[tag]['wall_s']} s", flush=True)
    ctl = {}
    if "2d" in rec:
        ctl["2d_records_3_fixed"] = (rec["2d"]["n_lines_3_fixed"], KNOWN["2d_records_3_fixed"])
    if "3d" in rec:
        ctl["3d_lines_3_fixed"] = (rec["3d"]["n_lines_3_fixed"], KNOWN["3d_lines_3_fixed"])
        ctl["3d_agents_3_fixed"] = (rec["3d"]["n_agents_3_fixed"], KNOWN["3d_agents_3_fixed"])
    rec["known_value_control"] = {k: {"measured": v[0], "expected": v[1], "pass": v[0] == v[1]}
                                  for k, v in ctl.items()}
    ok = all(v["pass"] for v in rec["known_value_control"].values())
    rec["pass"] = ok
    tagname = "_".join(which)
    (OUT / f"extract_record_{tagname}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"[extract] control {json.dumps(rec['known_value_control'])} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["2d", "3d"]))
