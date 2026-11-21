"""D-GSTR-1 P3 -- the number the nav-args claim depends on: what FRACTION of
WINDOWS would receive a REAL distance?

⛔ PER-RECORD IS NOT PER-WINDOW. A record is a CLIP; windows are unevenly
distributed over clips, so 30 % of records carrying args does NOT imply 30 % of
windows do. This measures the WINDOW fraction on both corpora, the same way
`V3Dataset.enable_nav_args` computes it.
Validity is read from the KEY'S PRESENCE, never from the value: 0.0 is the
default `_args_for_window` inserts for a MISSING slot.
ASCII-only.
"""
import json, sys, gzip
sys.path.insert(0, "/home/nvidia/navpred/stack")
from tanitad.data import v7_labels as v7l

out = {"tool": "D-GSTR-1 P3 nav-args window census",
       "evidence_class": "MEASURED (ours)",
       "rule": "args_valid comes from the PRESENCE of distance_m AND time_s "
               "in the record's args dict, never from the value"}
for split, path in (("train", "/home/nvidia/data/v72/labels/s2_labels_v7.2_train.jsonl.gz"),
                    ("eval", "/home/nvidia/data/v72/labels/s2_labels_v7.2_eval.jsonl.gz")):
    try:
        labels, man = v7l.load_v7_labels(path, allow_oracle_nav=True)
    except Exception as e:
        out[split] = {"ERROR": repr(e)[:200]}
        continue
    tot = ok = 0
    by_tok = {}
    ds, ts = [], []
    for lab in labels:
        nav = v7l.oracle_nav(lab, man) or {}
        tok = nav.get("token")
        a = nav.get("args") or {}
        good = ("distance_m" in a) and ("time_s" in a)
        tot += 1
        ok += int(good)
        r = by_tok.setdefault(tok, [0, 0])
        r[0] += 1; r[1] += int(good)
        if good:
            ds.append(float(a["distance_m"])); ts.append(float(a["time_s"]))
    out[split] = {"path": path, "md5": man.md5, "n_records": tot,
                  "n_records_with_real_args": ok,
                  "record_real_frac": round(ok / max(tot, 1), 4),
                  "by_token": {str(k): {"n": v[0], "with_args": v[1]}
                               for k, v in sorted(by_tok.items(), key=lambda kv: str(kv[0]))},
                  "distance_m_mean": round(sum(ds)/max(len(ds),1), 3),
                  "time_s_mean": round(sum(ts)/max(len(ts),1), 3),
                  "units": ["distance_m:metres", "time_s:seconds"]}
print(json.dumps(out, indent=1))
json.dump(out, open("/home/nvidia/navroute/NAVARGS_CENSUS.json", "w"), indent=1)
