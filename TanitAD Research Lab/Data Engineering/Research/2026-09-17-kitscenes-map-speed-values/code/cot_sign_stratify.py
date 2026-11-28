"""E-DE-SIGN-2 -- stratify the CoT speed-limit readings by sign SYSTEM and by day/night (0 GPU).

Pre-committed in ../RESULT.md section 2 BEFORE this ran:
  >= 20 of the readings Vienna-convention AND >= 5 at night  -> usable as E-DE-SIGN-1's positive control
  skewed US or day-only                                      -> the control tests the wrong sign system,
                                                                E-DE-SIGN-1 has NO validation reference

The label records carry their own strata (strata.country, strata.daynight_clock), so no parquet join
is needed; the parquet is still read as an INDEPENDENT cross-check of the country field.
Same-breath controls: total record counts must read 4,572 (train) and 147 (eval), or the read is void.
"""
import gzip, json, sys
from collections import Counter
import pandas as pd

LR = "D:/Projects/TanitAD/TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-09-04-v72-label-release/raw/"
META = "C:/Users/Admin/tanitad-data/physicalai/metadata/data_collection.parquet"
OUT = sys.argv[1] if len(sys.argv) > 1 else "raw/cot_sign_stratify.json"
EXPECT = {"v7.2_train": 4572, "v7.2_eval": 147}
MUTCD = {"United States"}   # rectangular "SPEED LIMIT NN" in mph; every other country here is a Vienna-Convention signatory


def system(c):
    return "US_MUTCD_mph" if c in MUTCD else "Vienna_circle_kmh"


meta = pd.read_parquet(META)
meta.index = meta.index.str.lower()
res = {"labels": LR, "meta": META, "expected_counts": EXPECT, "sets": {}}

for name, p in [("v7.2_train", LR + "s2_labels_v7.2_train.jsonl.gz"), ("v7.2_eval", LR + "s2_labels_v7.2_eval.jsonl.gz")]:
    n = 0
    flagged = []          # clips whose CoT tokeniser set speed_limit
    tl_flagged = 0        # traffic_light, carried as a same-breath POSITIVE control (must be non-zero on train)
    for line in gzip.open(p, "rt", encoding="utf-8"):
        n += 1
        r = json.loads(line)
        ct = r.get("cot_tokens") or {}
        if ct.get("traffic_light"):
            tl_flagged += 1
        if ct.get("speed_limit"):
            st = r.get("strata") or {}
            flagged.append({"clip_id": r.get("clip_id"), "country": st.get("country"),
                            "daynight": st.get("daynight_clock"), "road_class": st.get("road_class"),
                            "speed_limit_value": ct.get("speed_limit") if not isinstance(ct.get("speed_limit"), bool) else None,
                            "evidence": (ct.get("evidence") or "")[:160]})
    by_sys = Counter(system(f["country"]) for f in flagged)
    by_dn = Counter(f["daynight"] for f in flagged)
    # independent cross-check of the country field against the dataset's own parquet
    agree = dis = missing = 0
    for f in flagged:
        cid = (f["clip_id"] or "").lower()
        if cid not in meta.index:
            missing += 1
        elif meta.loc[cid, "country"] == f["country"]:
            agree += 1
        else:
            dis += 1
    res["sets"][name] = {
        "records": n, "records_match_expected": n == EXPECT[name],
        "speed_limit_flagged": len(flagged),
        "traffic_light_flagged_control": tl_flagged,
        "by_sign_system": dict(by_sys), "by_daynight": dict(by_dn),
        "by_country": dict(Counter(f["country"] for f in flagged)),
        "by_road_class": dict(Counter(f["road_class"] for f in flagged)),
        "country_crosscheck_vs_parquet": {"agree": agree, "disagree": dis, "not_in_parquet": missing},
        "clips": flagged,
    }

t = res["sets"]["v7.2_train"]
vienna = t["by_sign_system"].get("Vienna_circle_kmh", 0)
night = t["by_daynight"].get("night", 0)
res["verdict"] = {
    "vienna_readings": vienna, "night_readings": night,
    "bar": "vienna >= 20 AND night >= 5",
    "fires": "USABLE-CONTROL" if (vienna >= 20 and night >= 5) else "WRONG-SYSTEM-OR-DAY-ONLY",
    "controls_ok": all(v["records_match_expected"] for v in res["sets"].values())
                   and res["sets"]["v7.2_train"]["traffic_light_flagged_control"] > 0,
}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "clips"} for k, v in res["sets"].items()}, indent=1))
print("VERDICT", json.dumps(res["verdict"], indent=1))
