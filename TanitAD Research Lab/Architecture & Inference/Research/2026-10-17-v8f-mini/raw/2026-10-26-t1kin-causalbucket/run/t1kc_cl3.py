"""C-L3 of PREREG_V8F_T1KIN_CAUSALBUCKET: at tick 0 (logged history), the tool's causal bucket on 20 evenly spaced eval windows equals a HAND
computation from the cache's logged speed (an independent max + containing-window loop, not the tool's function).
python t1kc_cl3.py <tree> <eval cache> <grid> <join> <tracks> <out json>"""
import json
import sys

import numpy as np

TREE, CACHE, GRID, JOIN, TRACKS, OUT = sys.argv[1:7]
for p in (f"{TREE}/stack", f"{TREE}/taniteval", f"{TREE}/stack/scripts"):
    sys.path.insert(0, p)
import v8f_t1_eval as T1  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402

STEPS_KMH = (30, 50, 100, 120)
tracks = D.load_native_tracks(TRACKS)
J = D.load_join(JOIN)
store = D.ClipStore(CACHE, "eval", lru=4, grid="pool4x10", kin_mode="yaw", tracks=tracks)
wins = D.eval_grid_windows(GRID, J)
pick = np.linspace(0, len(wins) - 1, 20).round().astype(int)
rows, ok = [], True
for i in pick:
    ci, t0, _, _ = wins[i]
    clip = str(J["clip"][ci])
    base = store.get(clip)
    v = np.asarray(base["v"], np.float64)[max(0, t0 - 20):t0 + 1]
    v = v[np.isfinite(v)]
    kmh = 3.6 * max(float(v.max()), 0.0)
    hand = next((k for k, s in enumerate(STEPS_KMH) if kmh <= s + 3.6e-9), 3)
    o, n = int(J["off"][ci]), int(J["n"][ci])
    st = T1.WindowState(base, tracks[clip], J["nav"][o:o + n], ci, t0, "deadreckon")
    oh, okb = T1.causal_bucket(st.d["v"], st.j, T1.CAUSAL_STEPS)
    tool = int(np.argmax(oh))
    logged = int(np.argmax(J["limit"][ci])) if bool(J["limit_ok"][ci]) else None
    rows.append({"window": int(i), "clip": clip, "t0": int(t0), "max_kmh": round(kmh, 4), "hand": hand, "tool": tool, "tool_ok": bool(okb),
                 "logged": logged})
    ok &= okb and hand == tool
res = {"control": "C-L3", "n": len(rows), "PASS": bool(ok), "agree_with_logged": sum(r["tool"] == r["logged"] for r in rows), "rows": rows}
json.dump(res, open(OUT, "w"), indent=1)
print("C-L3", "PASS" if ok else "FAIL", f"{len(rows)} windows; causal == logged on {res['agree_with_logged']}")
