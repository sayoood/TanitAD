#!/usr/bin/env python3
"""DATASCALE label probe: can the v7 geometry/semantics emitter produce a record for a NEW clip without Alpamayo records? Reports what it does, never gates.
No address, port or pod id in this file."""
import json
import sys
import traceback

sys.path.insert(0, "/root/tad/stack")
sys.path.insert(0, "/root/tad/stack/scripts")
B = "/root/datascale"
out = {"probe": "s2_geom_emit_v7.emit_one on NEW clips (no Alpamayo record exists for them)", "results": []}
clips = [ln.strip() for ln in open(f"{B}/selection/pilot_clips.txt", encoding="utf-8") if ln.strip()][:3]
try:
    import s2_geom_emit_v7 as E
    from tanitad.data import alpamayo_records as AR
    out["import"] = "ok"
    try:
        out["alpamayo_records_available_n"] = len(AR.available())
    except Exception as e:                                           # noqa: BLE001
        out["alpamayo_records_available_n"] = f"{type(e).__name__}: {str(e)[:150]}"
    for c in clips:
        try:
            r = E.emit_one(c)
            out["results"].append({"clip": c[:8], "status": "RECORD", "keys": sorted(r)[:12] if isinstance(r, dict) else str(type(r))})
        except Exception as e:                                       # noqa: BLE001
            out["results"].append({"clip": c[:8], "status": "ERROR", "err": f"{type(e).__name__}: {str(e)[:200]}", "tb": traceback.format_exc()[-300:]})
except Exception as e:                                               # noqa: BLE001
    out["import"] = f"{type(e).__name__}: {str(e)[:200]}"
json.dump(out, open(f"{B}/labelprobe.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(out, indent=1))
