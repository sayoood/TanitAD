"""Boundary checks of the two outcome functions against PREREG section 5 (synthetic rows; not data)."""
import sys, json
sys.path.insert(0, 'C:/Users/Admin/t1hdread')
W = 'E:/Projects/TanitAD-artifacts/v8f-t1heads-run/tool_81aa3e36'
import t1hd_read as R
R.setup_imports(W)
def srow(delta, lo, hi):
    sep = lo > 0 or hi < 0
    return {"delta": delta, "lo": lo, "hi": hi, "separated": sep, "worse": sep and delta > 0, "better": sep and delta < 0}
def S(vals):  # vals: {seed: [(d,lo,hi)x3]}
    return {s: {f"S-POS@{a}": srow(*v[i]) for i, a in enumerate(R.ARCS)} for s, v in vals.items()}
better, ns, worse = (-1, -2, -0.5), (-0.1, -1, 0.5), (0.5, 0.2, 0.9)
out = {}
out["all better both seeds"] = R.strategic_outcome(S({"s0": [better] * 3, "s1": [better] * 3}))["OUTCOME"]
out["one not separated"] = R.strategic_outcome(S({"s0": [better, better, ns], "s1": [better] * 3}))["OUTCOME"]
out["one worse (s1, arc 80)"] = R.strategic_outcome(S({"s0": [better] * 3, "s1": [better, better, worse]}))["OUTCOME"]
out["worse beats better"] = R.strategic_outcome(S({"s0": [worse, better, better], "s1": [better] * 3}))["OUTCOME"]
def T(vals):  # {seed: {"lat": (d,lo,hi), "lon": (...)}}
    return {s: {"T-LAT-F1": dict(zip(("delta", "lo", "hi"), v["lat"])), "T-LON-F1": dict(zip(("delta", "lo", "hi"), v["lon"]))} for s, v in vals.items()}
tf = (2 * 0.0124, 2 * 0.0073)
ok = {"lat": (-0.01, -0.02, 0.0), "lon": (-0.005, -0.012, 0.002)}               # lo >= -2F for both
out["HOLDS"] = R.tactical_outcome(T({"s0": ok, "s1": ok}))["OUTCOME"]
edge = {"lat": (-0.01, -0.0249, 0.0), "lon": (-0.005, -0.012, 0.002)}            # lat lo just below -2F
out["lo just under -2F -> INCONCLUSIVE"] = R.tactical_outcome(T({"s0": edge, "s1": ok}))["OUTCOME"]
deg = {"lat": (-0.05, -0.09, -0.01), "lon": (-0.005, -0.012, 0.002)}              # hi<0 and delta < -2F
out["DEGRADES at one seed, one head"] = R.tactical_outcome(T({"s0": ok, "s1": deg}))["OUTCOME"]
small = {"lat": (-0.02, -0.03, -0.01), "lon": (-0.005, -0.012, 0.002)}            # hi<0 but delta > -2F
out["hi<0 but point > -2F -> INCONCLUSIVE"] = R.tactical_outcome(T({"s0": small, "s1": ok}))["OUTCOME"]
print(json.dumps(out, indent=1))
exp = {"all better both seeds": "S ADDS ROUTE", "one not separated": "NO WORSE", "one worse (s1, arc 80)": "WORSE", "worse beats better": "WORSE",
       "HOLDS": "HOLDS", "lo just under -2F -> INCONCLUSIVE": "INCONCLUSIVE", "DEGRADES at one seed, one head": "DEGRADES", "hi<0 but point > -2F -> INCONCLUSIVE": "INCONCLUSIVE"}
assert out == exp, (out, exp)
print("ALL OUTCOME BOUNDARIES AS COMMITTED")
