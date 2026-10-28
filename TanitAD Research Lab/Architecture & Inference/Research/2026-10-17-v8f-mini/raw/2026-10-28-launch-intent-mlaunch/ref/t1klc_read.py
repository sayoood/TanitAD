"""The read of PREREG_V8F_T1_LATCOND (bdbd69a0) with Amendment 1 (72dd3df0), applied verbatim. The canonical rows are t1k_read.py's (AST);
L1-L3 are defined in the PREREG (four_families._seq_geometry, pair speed = ds_mid / dt). Controls N-1..N-3 and the float32 check are read
before the outcome. python t1klc_read.py <t1k export root> <t1k_read.py> <dir: decisive> <dir: logged> <dir: strict> <dir: hacc-lat> <out dir>"""
import ast
import json
import math
import os
import sys

import numpy as np

ROOT, READER, DC, DL, DS, DH, OUT = sys.argv[1:8]
sys.path.insert(0, os.path.join(ROOT, "taniteval"))
import torch  # noqa: E402
from taniteval import four_families as FF  # noqa: E402
from taniteval.ci import paired_episode_cluster_bootstrap as PB  # noqa: E402

DT, F, TWO_F, VMIN = 0.2, 0.1207, 0.2415, 5.2
ns = {"np": np, "torch": torch, "FF": FF, "PB": PB, "math": math, "DT": DT}
for node in ast.parse(open(READER, encoding="utf-8").read()).body:
    if (isinstance(node, ast.FunctionDef) and node.name in ("comps", "paired")) or \
       (isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "ROWS" for t in node.targets)):
        exec(compile(ast.Module([node], []), READER, "exec"), ns)
comps, paired, ROWS = ns["comps"], ns["paired"], ns["ROWS"]
DOM_ROWS = ("ade2_m", "speed_mae_mps", "along_mae_m", "accel_mae_mps2", "heading_mae_deg", "cross_mae_m", "L1_lat_accel_mae_mps2",
            "L2_curvature_mae_fast_1pm", "L3_yaw_rate_mae_fast_degps")
os.makedirs(OUT, exist_ok=True)


def lat_rows(X, G, vmin=VMIN):
    Xt, Gt = torch.as_tensor(np.asarray(X, np.float64)), torch.as_tensor(np.asarray(G, np.float64))
    P, Q = FF._seq_geometry(Xt, DT), FF._seq_geometry(Gt, DT)
    bp = P["pair_valid"] & Q["pair_valid"]
    vP = 0.5 * (P["speed"][:, 1:] + P["speed"][:, :-1])
    vG = 0.5 * (Q["speed"][:, 1:] + Q["speed"][:, :-1])
    fast = bp & (vG >= vmin)

    def mm(x, m):
        s, c = (x * m).sum(1), m.sum(1)
        return torch.where(c > 0, s / c.clamp_min(1), torch.full_like(s, float("nan")))
    out = {"L1_lat_accel_mae_mps2": mm((vP * P["yaw_rate"] - vG * Q["yaw_rate"]).abs(), bp),
           "L2_curvature_mae_fast_1pm": mm((P["curvature"] - Q["curvature"]).abs(), fast),
           "L3_yaw_rate_mae_fast_degps": torch.rad2deg(mm((P["yaw_rate"] - Q["yaw_rate"]).abs(), fast)),
           "_alat_pairs": (vP * P["yaw_rate"]), "_mask": bp}
    return {k: (v.numpy() if k == "_mask" else v.numpy().astype(np.float64)) for k, v in out.items()}


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def rows_of(X, G):
    C = comps(X, G)
    C.update({k: v for k, v in lat_rows(X, G).items() if not k.startswith("_")})
    return C


refuse, rep = [], {"prereg": "Project Steering/PREREG_V8F_T1_LATCOND.md (bdbd69a0) + Amendment 1 (72dd3df0)",
                   "tier": "T1 (T1-kin: self-action OPEN loop; never 'closed loop')", "vmin_mps": VMIN,
                   "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0; arm - P10 per seed",
                   "controls": {}, "decisive": {}, "beside": {}}
Dd = {s: load(os.path.join(DC, f"dumps_HACC_s{s}_m10_causal.npz")) for s in (0, 1)}
# N-3: the known-value circle (Amendment 1)
v, r, H = 10.0, 50.0, 30
t = DT * np.arange(1, H + 1)
th = v * t / r
circ = np.stack([r * np.sin(th), r * (1 - np.cos(th))], -1)[None]
line = np.stack([v * t, np.zeros(H)], -1)[None]
lr = lat_rows(circ, line)
a = np.abs(lr["_alat_pairs"][0][lr["_mask"][0]])
dth = v * DT / r
ident = 2 * r * math.sin(dth / 2) * dth / DT ** 2
n3 = {"max_abs_vs_chord_identity": float(np.abs(a - ident).max()), "max_abs_vs_v2_over_r": float(np.abs(a - v * v / r).max()),
      "L1_row": float(lr["L1_lat_accel_mae_mps2"][0]), "n_pairs": int(a.size)}
n3["PASS"] = n3["max_abs_vs_chord_identity"] <= 1e-9 and n3["max_abs_vs_v2_over_r"] <= 1e-3 and a.size == H - 1
rep["controls"]["N-3"] = n3
# N-1: L1's conditioning on the noise probe (GT + 5 mm, seed 0)
G0 = Dd[0]["g"].astype(np.float64)
rng = np.random.default_rng(0)
Ln = lat_rows(G0 + rng.normal(0.0, 0.005, G0.shape), G0)["L1_lat_accel_mae_mps2"]
step = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(G0), 1, 2)), G0], 1), axis=1), axis=-1)
vbar = step.mean(1) / DT
lo_b, hi_b = float(np.nanmean(Ln[vbar < 2])), float(np.nanmean(Ln[vbar >= 11.8]))
rep["controls"]["N-1"] = {"L1_noise_floor_5mm_below_2mps": lo_b, "above_11p8mps": hi_b, "ratio": lo_b / hi_b, "PASS": lo_b / hi_b <= 3.0}
# N-2: the masks
c0 = comps(Dd[0]["cl"], Dd[0]["g"])
z0 = lat_rows(Dd[0]["cl"], Dd[0]["g"], vmin=0.0)
d2 = np.nanmax(np.abs(z0["L2_curvature_mae_fast_1pm"] - c0["curvature_mae_1pm"]))
d3 = np.nanmax(np.abs(z0["L3_yaw_rate_mae_fast_degps"] - c0["yaw_rate_mae_degps"]))
same_nan = bool((np.isnan(z0["L2_curvature_mae_fast_1pm"]) == np.isnan(c0["curvature_mae_1pm"])).all())
Qs = FF._seq_geometry(torch.as_tensor(Dd[0]["g"].astype(np.float64)), DT)
vGm = (0.5 * (Qs["speed"][:, 1:] + Qs["speed"][:, :-1])).numpy()
slow = (vGm < VMIN).all(1)
z5 = lat_rows(Dd[0]["cl"], Dd[0]["g"])
n2 = {"thr0_vs_canonical_curv_max_abs": float(d2), "thr0_vs_canonical_yaw_max_abs": float(d3), "nan_pattern_equal": same_nan,
      "n_windows_slow_everywhere": int(slow.sum()), "slow_windows_all_nan": bool(np.isnan(z5["L2_curvature_mae_fast_1pm"][slow]).all()
                                                                               and np.isnan(z5["L3_yaw_rate_mae_fast_degps"][slow]).all())}
n2["PASS"] = d2 <= 1e-9 and d3 <= 1e-9 and same_nan and n2["n_windows_slow_everywhere"] > 0 and n2["slow_windows_all_nan"]
rep["controls"]["N-2"] = n2
# the float32 dumps
f32 = {}
for s, want in ((0, -0.7680), (1, -0.7336)):
    C, Cf = comps(Dd[s]["cl"], Dd[s]["g"]), comps(Dd[s]["fl"], Dd[s]["g"])
    dlt = paired(C["ade6_m"], Cf["ade6_m"], Dd[s]["eid"])["delta"]
    f32[f"s{s}"] = {"ade6_delta": dlt, "banked": want, "PASS": abs(dlt - want) <= 1e-3}
rep["controls"]["float32"] = dict(f32, PASS=all(x["PASS"] for x in f32.values()))
for k, c in rep["controls"].items():
    if not c["PASS"]:
        refuse.append(k)


def read_cfg(Dx):
    out = {}
    for s in (0, 1):
        d = Dx[s]
        Cx, Cf = rows_of(d["cl"], d["g"]), rows_of(d["fl"], d["g"])
        allrows = list(ROWS) + ["L1_lat_accel_mae_mps2", "L2_curvature_mae_fast_1pm", "L3_yaw_rate_mae_fast_degps"]
        vs = {rw: paired(Cx[rw], Cf[rw], d["eid"]) for rw in allrows}
        worse = [rw for rw in DOM_ROWS if vs[rw]["separated"] and vs[rw]["delta"] > 0]
        a6 = vs["ade6_m"]
        out[f"s{s}"] = {"rows_arm_minus_p10": vs, "dom_rows_separated_worse": worse,
                        "C_DOM": a6["delta"] <= -TWO_F and a6["hi"] < 0 and not worse}
    return out


rep["decisive"] = read_cfg(Dd)
for name, d, pat in (("hacc_logged", DL, "dumps_HACC_s{s}_m10.npz"), ("hacc_strict", DS, "dumps_HACC_s{s}_m10_strict.npz"),
                     ("hacclat_B", DH, "dumps_B_s{s}_m10.npz")):
    rep["beside"][name] = read_cfg({s: load(os.path.join(d, pat.format(s=s))) for s in (0, 1)})
dec = rep["decisive"]
LATSET = ("heading_mae_deg", "cross_mae_m", "L1_lat_accel_mae_mps2", "L2_curvature_mae_fast_1pm", "L3_yaw_rate_mae_fast_degps")
if refuse:
    outc = "REFUSED"
elif all(dec[f"s{s}"]["C_DOM"] for s in (0, 1)):
    outc = "CONDITIONED DOMINANCE"
elif any(rw in LATSET for s in (0, 1) for rw in dec[f"s{s}"]["dom_rows_separated_worse"]):
    outc = "CONDITIONED LATERAL DEFICIT"
else:
    outc = "OTHER"
rep["OUTCOME"] = {"outcome": outc, "refused_on": refuse}
json.dump(rep, open(os.path.join(OUT, "T1KLC_READ.json"), "w"), indent=1)
print("controls", {k: v["PASS"] for k, v in rep["controls"].items()})
print("OUTCOME", outc, "refused:", refuse)
for s in (0, 1):
    x = dec[f"s{s}"]
    print(f"s{s} C_DOM {x['C_DOM']} worse {x['dom_rows_separated_worse']} ADE6 {x['rows_arm_minus_p10']['ade6_m']['delta']:+.4f}")
    for rw in ("L1_lat_accel_mae_mps2", "L2_curvature_mae_fast_1pm", "L3_yaw_rate_mae_fast_degps", "heading_mae_deg", "cross_mae_m"):
        q = x["rows_arm_minus_p10"][rw]
        print(f"   {rw:28s} {q['delta']:+.4f} [{q['lo']:+.4f}, {q['hi']:+.4f}] sep {q['separated']} n {q['n_windows']}")
