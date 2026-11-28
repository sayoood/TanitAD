"""Control C8 verdict. usage: c8_check.py <intact ia_read.json> <zero-act ia_read.json> <out.json>
Known values: with the level-1 action embedding zeroed, every L1 IA-3 gain (lon and lat, every horizon) reads EXACTLY 0.0; the intact control reads non-zero. Part A (forward response) is in
F:/TanitAD-scratch/actsens/c1/c8_zeroact.json (read here too)."""
import json, sys
I, Z = [json.load(open(p)) for p in sys.argv[1:3]]
out = sys.argv[3]
res = {"tier": "T0", "evidence": "MEASURED", "n_zeroed_params": Z["n_zeroed_params"], "zero_act": Z["zero_act"]}
l1 = lambda j: {k: v for k, v in j["beside"]["ia3"].items() if k.startswith("1|")}
zi, zz = l1(I), l1(Z)
res["intact_L1"] = {k: {"gain_lon": v["gain_lon"]["point"], "gain_lat": v["gain_lat"]["point"]} for k, v in zi.items()}
res["zero_L1"] = {k: {"gain_lon": v["gain_lon"]["point"], "gain_lat": v["gain_lat"]["point"]} for k, v in zz.items()}
res["zero_all_exactly_0.0"] = bool(zz) and all(v[g] == 0.0 for v in res["zero_L1"].values() for g in ("gain_lon", "gain_lat"))
res["intact_nonzero"] = bool(zi) and all(abs(v[g]) > 0.0 for v in res["intact_L1"].values() for g in ("gain_lon", "gain_lat"))
try:
    pa = json.load(open("F:/TanitAD-scratch/actsens/c1/c8_zeroact.json"))
    res["part_A"] = pa["verdict"]
except Exception as e:
    res["part_A"] = f"unreadable: {e}"
res["C8_PASS"] = bool(res["zero_all_exactly_0.0"] and res["intact_nonzero"] and isinstance(res["part_A"], dict) and all(res["part_A"].values()))
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
