"""The run's TRAINING-row 10 cm map signal, pooled per step bucket (read-only on metrics.jsonl): per class, pooled
IoU = sum(inter)/sum(union) over all training rows in the bucket, under the declared rule (pc) and the raw argmax,
0-20 m band and all bands. Buckets: 250 steps below 1,000, then 1,000. Evidence: MEASURED in-run (metrics.jsonl
TRAIN rows, every 50 steps, batch 16, train-mode forward on the training batch)."""
import json
import sys

B = ("0_20", "20_40", "40_60", "60_80", "80_100")
CK = ("nocls", "drivable", "lane", "crosswalk", "arrow", "edge", "hatched", "sidewalk")


def main(path, out):
    agg = {}
    n_rows = {}
    for line in open(path):
        r = json.loads(line)
        if "map_hires_n_windows" not in r or "eval_map_hires_n_windows" in r or r.get("step") is None:
            continue
        s = int(r["step"])
        bucket = (s // 1000) * 1000 if s >= 1000 else (s // 250) * 250
        a = agg.setdefault(bucket, {})
        n_rows[bucket] = n_rows.get(bucket, 0) + 1
        for c in CK:
            for rule, i, u in (("pc", "inter", "union"), ("raw", "interraw", "unionraw")):
                for b in ("0_20", "all"):
                    bb = B if b == "all" else (b,)
                    I = sum(r.get(f"map_hires_{i}_{c}_{x}", 0) or 0 for x in bb)
                    U = sum(r.get(f"map_hires_{u}_{c}_{x}", 0) or 0 for x in bb)
                    k = f"{c}_{rule}_{b}"
                    a.setdefault(k, [0.0, 0.0])
                    a[k][0] += I
                    a[k][1] += U
    rows = []
    for s in sorted(agg):
        rows.append({"bucket_start": s, "n_train_rows": n_rows[s],
                     **{k: (v[0] / v[1] if v[1] else None) for k, v in agg[s].items()}})
    json.dump({"source": path, "evidence": "MEASURED in-run, TRAIN rows (metrics.jsonl)", "rows": rows},
              open(out, "w"), indent=0)
    print(f"[train-curve] {len(rows)} buckets -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
