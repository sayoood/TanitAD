#!/usr/bin/env python3
"""Does the OpenScene pkl's own camera pairing (frame['cams'][CAM]['data_path']) equal the nuPlan-DB rule P2/P6 use
(planner.FrameResolver: nearest DB `image` row to the token's lidar timestamp, <= 120 ms, per camera)?
MEASURED on ALL 12,146 navtest tokens x 4 cameras, using the rows the REFe navtest frame builder wrote
(D:/Projects/TanitAD/data/refe_navtest/frames/tokens_frames.json, DB rule) against the local test pkl. 0 download; CPU.
Why it matters: if equal, the exact jpg for a pool-B token can be named from the pkl alone (no DB), so the 53 DB-less logs
can still be frame-selected; the seam itself still reads the DB (see RESULT.md section 4)."""
import sys, os, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pklio
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")
rows = json.load(open("D:/Projects/TanitAD/data/refe_navtest/frames/tokens_frames.json"))
by_log = collections.defaultdict(dict)
for r in rows:
    by_log[r["log"]][r["token"]] = r
CAMS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")
tot = eq = 0; per_cam_eq = collections.Counter(); per_cam_n = collections.Counter(); ts_ok = 0; ts_n = 0; missing_tok = 0; diff_examples = []
for log, toks in sorted(by_log.items()):
    pk = pklio.load_pkl_bytes(open("D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test/%s.pkl" % log, "rb").read())
    idx = {f["token"]: f for f in pk}
    for t, r in toks.items():
        f = idx.get(t)
        if f is None:
            missing_tok += 1; continue
        ts_n += 1; ts_ok += int(int(f["timestamp"]) == int(r["timestamp_us"]))
        for c, rel in zip(CAMS, r["cams"]):
            tot += 1; per_cam_n[c] += 1
            same = os.path.basename(f["cams"][c]["data_path"]) == os.path.basename(rel)
            eq += int(same); per_cam_eq[c] += int(same)
            if not same and len(diff_examples) < 5:
                diff_examples.append([log, t, c, f["cams"][c]["data_path"], rel])
out = {"tokens": len(rows), "logs": len(by_log), "pairs_compared": tot, "pairs_equal": eq, "per_camera_equal": dict(per_cam_eq), "per_camera_n": dict(per_cam_n),
       "token_missing_in_pkl": missing_tok, "token_timestamp_equal_pkl_vs_db": [ts_ok, ts_n], "examples_unequal": diff_examples}
json.dump(out, open(os.path.join(RAW, "pairing_pkl_vs_db.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
