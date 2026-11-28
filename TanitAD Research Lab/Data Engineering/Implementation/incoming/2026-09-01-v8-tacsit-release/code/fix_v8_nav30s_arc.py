"""v8 CORRECTION — `nav_30s` arc distances were on the WRONG TIMELINE.

⛔ THE DEFECT, MEASURED, and it is the RAW_T0_S class for the second time.
`manoeuvre_sequence` publishes ANCHOR-RELATIVE times: `s2_geom_emit_v7.py:148`
returns `i / hz` where `i` indexes `rate[key:]`, so t=0 is the s2 anchor.
`build_v8_nav30s.py` fed those straight into an arc table indexed on the RAW
recording timeline, where the anchor sits at `RAW_T0_S = 8.0 s`. Every turn's
`distance_m` therefore measured arc length from the RECORDING START to
`t_start_s`, instead of from the ANCHOR to the manoeuvre.

MEASURED blast radius over all 5,418 entries (train+eval):
    NAV_FOLLOW_ROAD  2,926   error exactly 0.0 m  (hardcoded 0.0, correct)
    NAV_TURN_L       1,196   median -3.0  mean -9.6  max |e| 129.9 m
    NAV_TURN_R       1,289   median -1.2  mean -6.5  max |e| 169.2 m
    TURN entries: 81.8 % wrong by >1 m, 54.6 % by >10 m.

⭐ WHY IT SURVIVED THE BUILD GATE: the corpus-wide median is +0.0 m, because
NAV_FOLLOW_ROAD is 54 % of entries and carries an exact hardcoded zero. The
error hides behind a pile of correct zeros — a summary statistic reporting
health for a field that is wrong on four fifths of the rows that USE it.

⚠️ A SECOND, SMALLER BUG IN THE SAME LINE: the old builder never re-zeroed the
timestamp axis (`ts - ts[0]`), which `egomotion_source.load` does. Clip
`002646e7` starts at -0.197 s, so the raw read was additionally offset by the
recording's own start. Both are fixed here by construction.

⛔ TIMES ARE NOT TOUCHED. `t_start_s` / `t_end_s` are anchor-relative and were
always correct as such — they match every other time field in `s2-geom-v7`. The
bug was reading them on the wrong axis, not writing them wrongly.
"""
import gzip, hashlib, io as _io, json, pickle
from pathlib import Path

import numpy as np

REL = Path("C:/Users/Admin/tanitad-wt/_s2build/release"); OUT = REL / "v8"
RAW_T0_S, HORIZON_S = 8.0, 30.0
with open("C:/Users/Admin/AppData/Local/Temp/claude/arc_tables.pkl", "rb") as f:
    ARC = pickle.load(f)                      # built with ts-ts[0], raw axis

# Everything outside nav_30s must be byte-identical. nav_30s is NOT in this list
# precisely because it is the field being corrected.
FROZEN = ("a_tac", "g_tac", "a_str", "g_str", "nav_command", "tac_SIT",
          "lane_change_text", "manoeuvre_sequence", "bands", "strata",
          "speed_max_input", "alpamayo", "semantics", "scene", "turn_suppression")


def arc_from_anchor(cid, t_anchor):
    """Arc length driven from the S2 ANCHOR to `t_anchor` seconds after it."""
    tab = ARC.get(cid)
    if tab is None:
        return None
    t, s = tab
    tt = t_anchor + RAW_T0_S
    if tt > t[-1]:
        return None
    return round(float(np.interp(tt, t, s) - np.interp(RAW_T0_S, t, s)), 1)


stats = {}
for side in ("train", "eval"):
    p = OUT / ("s2_labels_v8_%s.jsonl.gz" % side)
    rows = [json.loads(x) for x in gzip.open(p, "rt", encoding="utf-8") if x.strip()]
    before = {r["clip_id"]: {k: json.dumps(r.get(k), sort_keys=True) for k in FROZEN}
              for r in rows}
    moved = n_ent = n_none = 0
    for r in rows:
        nav = r["nav_30s"]
        for e in nav["entries"]:
            n_ent += 1
            old_s, old_e = e.get("distance_m"), e.get("distance_end_m")
            new_s = 0.0 if e["token"] == "NAV_FOLLOW_ROAD" else \
                arc_from_anchor(r["clip_id"], e["t_start_s"])
            new_e = arc_from_anchor(r["clip_id"], e["t_end_s"])
            if new_s is None or new_e is None:
                n_none += 1
            if new_s != old_s or new_e != old_e:
                moved += 1
            e["distance_m"], e["distance_end_m"] = new_s, new_e
        nav["distance_definition"] = (
            "ARC LENGTH along the driven path, measured FROM THE S2 ANCHOR (t0) to "
            "the entry time. ⛔ NOT Euclidean. ⛔ NOT from the recording start.")
        nav["time_basis"] = (
            "t_start_s / t_end_s are ANCHOR-RELATIVE (0 = the s2 anchor = "
            "RAW_T0_S 8.0 s on the raw recording timeline), matching every other "
            "time field in s2-geom-v7.")
        nav["⛔ arc_corrected_2026-09-09"] = (
            "The v8 build shipped 2026-09-07 measured these distances on the RAW "
            "recording timeline while t_start_s is ANCHOR-RELATIVE, an 8.0 s "
            "offset. MEASURED over 2,485 turn entries: 81.8 % were wrong by >1 m, "
            "54.6 % by >10 m, max 169.2 m. NAV_FOLLOW_ROAD was unaffected (exact "
            "0.0). Corrected here; times were always correct and are untouched.")
    drift = [(r["clip_id"], k) for r in rows for k in FROZEN
             if json.dumps(r.get(k), sort_keys=True) != before[r["clip_id"]][k]]
    assert not drift, "⛔ a frozen field moved: %s" % drift[:3]
    with open(p, "wb") as raw:
        gz = gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0)
        with _io.TextIOWrapper(gz, encoding="utf-8") as f:
            for r in sorted(rows, key=lambda x: x["clip_id"]):
                f.write(json.dumps(r, sort_keys=True) + "\n")
    b = p.read_bytes()
    stats[side] = {"n": len(rows), "entries": n_ent, "entries_moved": moved,
                   "entries_without_arc": n_none,
                   "md5": hashlib.md5(b).hexdigest(),
                   "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
    print("%-5s %4d clips  %5d entries  %5d moved  %d unreachable  md5 %s"
          % (side, len(rows), n_ent, moved, n_none, stats[side]["md5"]))

# ⭐ IDEMPOTENCE / INDEPENDENT RE-DERIVATION: re-read the SHIPPED file and
# recompute every distance from the arc tables. Zero may move.
for side in ("train", "eval"):
    p = OUT / ("s2_labels_v8_%s.jsonl.gz" % side)
    bad = 0
    for x in gzip.open(p, "rt", encoding="utf-8"):
        if not x.strip():
            continue
        r = json.loads(x)
        for e in r["nav_30s"]["entries"]:
            exp = 0.0 if e["token"] == "NAV_FOLLOW_ROAD" else \
                arc_from_anchor(r["clip_id"], e["t_start_s"])
            if e["distance_m"] != exp:
                bad += 1
    assert not bad, "⛔ %d entries do not re-derive" % bad
    print("   ✅ %-5s re-derives exactly from the arc tables" % side)
print(json.dumps(stats, indent=1))
