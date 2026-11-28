"""STOP_POINT goal vs independent stop (v < 0.5 m/s at any 10 Hz sample in [t0+2, t0+6])."""
import json, sys, numpy as np, d2_lib as L
side = L.load_clock_sidecar("D:/refcv6_eval_kit/data/refcv6_clip_clock_sidecar.jsonl")
trs = {t.sid: t for t in L.tracks_from_manifest("D:/Projects/TanitAD-artifacts/refcv8_audit/D2/train_v2manifest.pt", side)}
recs = L.read_jsonl_gz("D:/refcv6_eval_kit/data/a6/s2_labels_v8_train.jsonl.gz")
from collections import Counter
c = Counter()
for r in recs:
    tr = trs.get(L.stable_episode_id(r["clip_id"]))
    if tr is None:
        continue
    g = r["g_tac"]["goals"]
    sp = "STOP_POINT" in g
    turn = any(k.startswith(("TURN_", "YIELD_FOR_TURN_")) for k in g)
    g26 = L.lon_features(tr, 8.0, 2.0, 6.0)
    stop = g26["vmin"] < 0.5
    c[(sp, stop, turn)] += 1
out = {f"STOP_POINT={a}|indep_stop_in_band={b}|turn_goal={t}": n for (a, b, t), n in sorted(c.items())}
sp_pos = sum(n for (a, b, t), n in c.items() if a)
sp_pos_stop = sum(n for (a, b, t), n in c.items() if a and b)
nsp_stop_noturn = sum(n for (a, b, t), n in c.items() if (not a) and b and (not t))
out.update({"STOP_POINT_positive": sp_pos, "STOP_POINT_positive_with_indep_stop": sp_pos_stop,
            "no_STOP_POINT_but_indep_stop_and_no_turn_goal": nsp_stop_noturn})
json.dump(out, open("stop_goal_check_train.json", "w"), indent=1)
print(json.dumps(out, indent=1))
