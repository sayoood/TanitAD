"""Overlap of NuRec 26.04 scene clip ids (map.xodr carriers) with our label-release and B1 clip sets."""
import csv, gzip, hashlib, json, re, sys
nurec_csv, out = sys.argv[1], sys.argv[2]
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
nurec = [r["clip_id"].strip().lower() for r in csv.DictReader(open(nurec_csv, encoding="utf-8"))]
N = set(nurec)
LR = "TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-09-04-v72-label-release/raw/"
def label_ids(p):
    ids, n = set(), 0
    for line in gzip.open(p, "rt", encoding="utf-8"):
        n += 1
        rec = json.loads(line)
        cid = rec.get("clip_id") or rec.get("clip") or rec.get("uuid")
        if cid is None:
            m = UUID.search(line); cid = m.group(0) if m else None
        if cid: ids.add(str(cid).lower())
    return ids, n
res = {"nurec_csv_sha256": hashlib.sha256(open(nurec_csv, "rb").read()).hexdigest(), "nurec_clips": len(N),
       "nurec_rows": len(nurec), "nurec_uuid_shaped": sum(bool(UUID.fullmatch(x)) for x in N), "sets": {}}
for name, p in [("v7.2_train", LR + "s2_labels_v7.2_train.jsonl.gz"), ("v7.2_eval", LR + "s2_labels_v7.2_eval.jsonl.gz")]:
    ids, n = label_ids(p)
    res["sets"][name] = {"records": n, "ids": len(ids), "uuid_shaped": sum(bool(UUID.fullmatch(x)) for x in ids),
                         "overlap_with_nurec": len(ids & N), "overlap_ids": sorted(ids & N)}
A = "TanitAD Research Lab/Architecture & Inference/Implementation/incoming/2026-08-17-thor-concurrency-pilot/alpamayo_clip_ids.txt"
ids = set(x.strip().lower() for x in open(A, encoding="utf-8") if x.strip())
res["sets"]["alpamayo_clip_ids_4729"] = {"ids": len(ids), "uuid_shaped": sum(bool(UUID.fullmatch(x)) for x in ids),
                                        "overlap_with_nurec": len(ids & N), "overlap_ids": sorted(ids & N)}
res["expected_overlap_if_independent_train"] = round(1607 / 306152 * res["sets"]["v7.2_train"]["ids"], 1)
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: (v if k != "sets" else {s: {kk: vv for kk, vv in d.items() if kk != "overlap_ids"} for s, d in v.items()}) for k, v in res.items()}, indent=1))
