"""Exactness of the pipeline-level measures (sam3map_front_stream.py: one model build per process, refine's ego masks on the loaded
model in fp32, CPU stages in the background) against the approval chain's arm with the same per-frame flags, where every stage ran
as its own process. Bit-level on every intermediate: raw frames, refine output (incl. ego_CAM_FW and stats), consensus output;
cell-level on the composed map. Also the refine summary JSON (ego-mask shares) must be equal.
Usage: stream_equal.py <chain arm tag> <stream tag>  ->  ZZSTREAMEQ-<pass>ZZ"""
import json, sys
from pathlib import Path
import numpy as np

SM = Path("/home/nvidia/sam3map")
arm, st = sys.argv[1], sys.argv[2]
ok_all = True
for c8 in ("73495082f98b", "4fbd97b6a4b7"):
    row = {}
    for stage, (da, db) in {"raw": (f"{c8}_{arm}raw", f"{c8}_{st}raw"), "refine": (f"{c8}_{arm}", f"{c8}_{st}"), "consensus": (f"{c8}_{arm}c", f"{c8}_{st}c")}.items():
        fa = sorted((SM / da).glob("[0-9][0-9][0-9].npz")); same = 0; bad = []
        for f in fa:
            g = SM / db / f.name
            if not g.exists():
                bad.append(f.name + ":missing"); continue
            a = np.load(f, allow_pickle=True); b = np.load(g, allow_pickle=True)
            keys_a = set(a.files); keys_b = set(b.files)
            eq = keys_a == keys_b
            for k in sorted(keys_a & keys_b):
                if k == "stats":
                    sa, sb = json.loads(str(a[k])), json.loads(str(b[k]))
                    for s_ in (sa, sb):                                     # provenance string names the tag; everything else must match
                        s_.pop("_front_only_from", None)
                    eq = eq and sa == sb
                elif k == "tok":
                    eq = eq and str(a[k]) == str(b[k])
                else:
                    eq = eq and a[k].shape == b[k].shape and a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k])
            same += eq
            if not eq:
                bad.append(f.name)
        row[stage] = f"{same}/{len(fa)}" + (f" bad {bad[:3]}" if bad else "")
        ok_all = ok_all and same == len(fa) and len(fa) == 96
    wa = np.load(SM / f"render5_{c8}_{arm}r" / "worldmap.npz", allow_pickle=True)["cls"]
    wb = np.load(SM / f"render5_{c8}_{st}r" / "worldmap.npz", allow_pickle=True)["cls"]
    row["map_identical"] = bool(wa.shape == wb.shape and np.array_equal(wa, wb))
    ja = json.loads((SM / f"refine_{arm}_{c8}.json").read_text()); jb = json.loads((SM / f"refine_{st}_{c8}.json").read_text())
    row["refine_summary_equal"] = ja == jb
    ok_all = ok_all and row["map_identical"] and row["refine_summary_equal"]
    print(c8, row)
print(f"ZZSTREAMEQ-{'PASS' if ok_all else 'FAIL'}ZZ")
