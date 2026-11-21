"""E-DE-SIGN-3 -- re-derive VALUED speed-limit readings from the CoT text (0 GPU, no ego). See ../SPEC.md."""
import gzip, json, re, sys
from collections import Counter

LR = "D:/Projects/TanitAD/TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-09-04-v72-label-release/raw/"
OUT = sys.argv[1] if len(sys.argv) > 1 else "raw/cot_sign_values.json"
EXPECT = {"v7.2_train": 4572, "v7.2_eval": 147}
KMH_LEGAL = {5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130}
MPH_LEGAL = set(range(15, 90, 5))
MUTCD = {"United States"}

PATS = [
    (re.compile(r"\b(\d{1,3})\s*(?:km/h|kph|kmh)\b", re.I), "kmh"),
    (re.compile(r"\b(\d{1,3})\s*mph\b", re.I), "mph"),
    (re.compile(r"\b(\d{1,3})\s*(?:-\s*)?(?:km/h\s*)?speed[ -]limit", re.I), "bare"),
    (re.compile(r"speed[ -]limit(?:\s+sign)?(?:\s+(?:of|to|indicates|indicating|shows|showing|reads|reading|is|at))?\s+(\d{1,3})\b(?!\s*(?:seconds?|s\b|m\b|meters?))", re.I), "bare"),
]


def parse(text):
    """All (value, unit) readings bound to a speed-limit context; durations/ordinals excluded by pattern."""
    out = []
    for p, unit in PATS:
        for m in p.finditer(text or ""):
            out.append((int(m.group(1)), unit))
    return out


def naive(text):
    m = re.search(r"\d+", text or "")
    return int(m.group(0)) if m else None


LIT = [("a 50 km/h sign is posted", 50), ("the overhead speed limit sign indicates 100", 100),
       ("because of the 30 speed limit sign", 30), ("1. Type: speed limit sign", None), ("duration: 0-4 seconds", None)]
lit_res = []
for s, want in LIT:
    got = parse(s); v = got[0][0] if got else None
    lit_res.append({"text": s, "want": want, "got": v, "ok": v == want})
lit_ok = all(r["ok"] for r in lit_res)
naive_fails = sum(naive(s) != want for s, want in LIT)
assert lit_ok, lit_res
assert naive_fails >= 1, "literal tests cannot discriminate a naive parser"

res = {"spec": "SPEC.md E-DE-SIGN-3", "literal_tests": lit_res, "naive_parser_failures": naive_fails, "sets": {}}
for name in EXPECT:
    n = flags = 0; valued = []
    for line in gzip.open(LR + "s2_labels_" + name + ".jsonl.gz", "rt", encoding="utf-8"):
        n += 1; r = json.loads(line)
        ct = r.get("cot_tokens") or {}; flag = bool(ct.get("speed_limit")); flags += flag
        text = (r.get("semantics") or {}).get("cot") or ""
        vals = parse(text)
        if vals:
            st = r.get("strata") or {}
            c = st.get("country")
            system = "US_MUTCD_mph" if c in MUTCD else "Vienna_circle_kmh"
            uniq = sorted({v for v, _ in vals})
            legal = [(v in (MPH_LEGAL if system == "US_MUTCD_mph" else KMH_LEGAL)) for v in uniq]
            valued.append({"clip_id": r.get("clip_id"), "country": c, "system": system, "flagged": flag,
                           "daynight": st.get("daynight_clock"), "values": uniq, "units": sorted({u for _, u in vals}),
                           "all_legal": all(legal), "n_distinct_values": len(uniq)})
    res["sets"][name] = {
        "records": n, "records_ok": n == EXPECT[name], "speed_limit_flags": flags,
        "valued_clips": len(valued), "valued_and_flagged": sum(v["flagged"] for v in valued),
        "valued_not_flagged": sum(not v["flagged"] for v in valued),
        "by_system": dict(Counter(v["system"] for v in valued)),
        "by_daynight": dict(Counter(v["daynight"] for v in valued)),
        "clips_all_legal": sum(v["all_legal"] for v in valued),
        "clips_multi_value": sum(v["n_distinct_values"] > 1 for v in valued),
        "value_hist": dict(Counter(x for v in valued for x in v["values"])),
        "clips": valued}
t = res["sets"]["v7.2_train"]
vienna = t["by_system"].get("Vienna_circle_kmh", 0)
legal_frac = t["clips_all_legal"] / max(t["valued_clips"], 1)
controls = t["records_ok"] and res["sets"]["v7.2_eval"]["records_ok"] and t["speed_limit_flags"] == 69
if not controls:
    fires = "VOID"
elif legal_frac < 0.95:
    fires = "UNLAWFUL"
elif vienna >= 20:
    fires = "SPECIFIED"
elif vienna < 10:
    fires = "THIN"
else:
    fires = "MIDDLE"
res["verdict"] = {"vienna_valued_train": vienna, "legal_frac_train": round(legal_frac, 4), "controls_ok": controls,
                  "fires": fires, "quoted_31_reproduces": t["valued_clips"] == 31}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "clips"} for k, v in res["sets"].items()}, indent=1))
print(json.dumps(res["verdict"]))
