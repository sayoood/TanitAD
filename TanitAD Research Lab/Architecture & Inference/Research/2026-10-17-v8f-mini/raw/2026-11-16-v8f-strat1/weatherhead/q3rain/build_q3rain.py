"""Second blind rain draw (PREREG_Q3RAIN.md). Reuses build_q3.py (draw(), jpeg(), TPL, H) by import; only the parameters differ.
Writes next to this script: Q3RAIN_SHEET.html (frames embedded, LOCAL ONLY), key_q3rain.json (no images), draw_report_q3rain.json."""
import json, os, sys
import numpy as np
SHEET = "F:/TanitAD-scratch/strat1/whead/q3sheet"
sys.path.insert(0, SHEET)
import build_q3 as b

HERE = os.path.dirname(os.path.abspath(__file__))
N, SEED, PERM = 20, "q3rain-weather-v1", 20261118
# exclusion = PI's 60 pilot clips + the 40 Q3 clips, written in the pilot-answers format draw() reads
pilot = json.load(open(SHEET + "/ref/pilot_audit_answers_PI_60.json"))["frames"]
q3 = json.load(open(SHEET + "/key.json"))
ex = {x["clip_id"] for x in pilot} | {k["clip"] for k in q3}
os.makedirs(HERE + "/tmp", exist_ok=True)
json.dump({"frames": [{"clip_id": c} for c in sorted(ex)]}, open(HERE + "/tmp/excl.json", "w"))
b.EXCL, b.SEED, b.ORDER = HERE + "/tmp/excl.json", SEED, [("rain", N)]
items, counts, excl, nclips = b.draw()
assert len(excl) == len(ex)
items = [items[i] for i in np.random.default_rng(PERM).permutation(len(items))]
shown, key = [], []
for it in items:
    fid = b.H(f"{SEED}-id:{it['clip']}:{it['k']}")[:8]
    now = b.jpeg(it["clip"], 6 * it["k"])
    prev = b.jpeg(it["clip"], 6 * (it["k"] - 5)) if it["k"] >= 5 else None
    shown.append({"id": fid, "now": now, "prev": prev})
    key.append({"id": fid, "clip": it["clip"], "step": it["step"], "frame_idx": it["k"], "video_frame": 6 * it["k"], "label": it["label"], "prev_shown": prev is not None})
html = b.TPL.replace("__ITEMS__", json.dumps(shown, separators=(",", ":"))).replace("(40 frames)", f"({N} frames)").replace("q3draft", "q3raindraft")
assert "https://" not in html and "http://" not in html
open(HERE + "/Q3RAIN_SHEET.html", "w", encoding="utf-8").write(html)
kc = {k["clip"] for k in key}
assert len({k["id"] for k in key}) == N and len(kc) == N and not (kc & ex) and not (kc & {k["clip"] for k in q3})
assert all(k["label"] == "rain" for k in key)
rep = {"seed": SEED, "perm_seed": PERM, "table_sha256": b.TABLE_SHA, "prereg_sha256": b.sha(HERE + "/PREREG_Q3RAIN.md"), "excluded_clips": len(ex),
       "eligible_rain_clips_after_exclusion": nclips["rain"], "drawn": N, "overlap_with_excluded": 0, "frames_with_prev": sum(k["prev_shown"] for k in key),
       "sheet_bytes": os.path.getsize(HERE + "/Q3RAIN_SHEET.html"), "build_script_sha256": b.sha(__file__), "reused_build_q3_sha256": b.sha(b.__file__)}
json.dump(key, open(HERE + "/key_q3rain.json", "w"), indent=1)
json.dump(rep, open(HERE + "/draw_report_q3rain.json", "w"), indent=1)
print(json.dumps(rep))
