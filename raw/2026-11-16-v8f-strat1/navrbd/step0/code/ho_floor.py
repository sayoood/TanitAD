"""NAVRBD step 0: held-out floor F_HO. CPU, zero training. TAC-LAT by the NAVRB2 read's own path:
navrb2_read.load_arm -> OR.load_dump / S.load_dump_n (dump 'cl' = plan track, 'g' = GT track) -> obd_geom.build().labels (= read_otband.labels:
factor_from_kinematics) ; cell TAC-LAT = (labels(cl) == labels(g)) as float (obd_read.to_cells, obd_read.py:347-351).
Estimator: obd_read.cell_delta -> taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0."""
import sys, os, json, hashlib, glob
import numpy as np
SR = "F:/TanitAD-scratch/strat1/read"
for p in (SR, SR + "/theory/obd/read", SR + "/theory/otband/read"):
    sys.path.insert(0, p)
import strat1_lib as L
import strat1_read as S
import obd_read as OR
import obd_geom as G

W = "F:/TanitAD-scratch/strat1/navrbd/step0"
CI = L.load_ci()
geom = G.build()
NB = 2000


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def lat_labels(dump):
    return geom.labels(dump["g"])[0], geom.labels(dump["cl"])[0]


def load(d, n):
    return S.load_dump_n(d, n)


ho = {a: load(f"{W}/t1/dump_NAVRBD-HO-{a}_m10_causal_heads", 497) for a in ("N1R_s0", "N1R_s1")}
a0, a1 = ho["N1R_s0"], ho["N1R_s1"]
for k in OR.SAME:
    assert np.array_equal(a0[k], a1[k]), k
eid = a0["eid"]
N = len(eid)
grid = json.load(open(f"{W}/in/grid/eval_grid_x_feasible.json"))
gsha = sha(f"{W}/in/grid/eval_grid_x_feasible.json")
assert gsha.startswith("ca289d0f")
assert N == grid["n_windows"] == 5736 and len(np.unique(eid)) == 497 == len(grid["order"])
join_clip = np.load("F:/TanitAD-scratch/strat1/navrbd/step0/in/join_X_clip.npy", allow_pickle=False) if os.path.exists(f"{W}/in/join_X_clip.npy") else None
if join_clip is not None:
    got = {}
    for ci, t in zip(a0["clip_index"], a0["ws"]):
        got.setdefault(str(join_clip[ci]), []).append(int(t))
    assert set(got) == set(grid["order"]) and all(sorted(got[c]) == sorted(grid["windows"][c]) for c in grid["order"]), "dump windows != feasible grid"
    grid_match = "windows equal the feasible grid exactly (clip, t)"
else:
    grid_match = "count only (5,736 / 497)"

gl, p0 = lat_labels(a0)
gl1, p1 = lat_labels(a1)
assert np.array_equal(gl, gl1)
acc0 = (p0 == gl).astype(float)
acc1 = (p1 == gl).astype(float)
CL = ("lane_keep", "turn_left", "turn_right")


def descr(p):
    out = {"pred_share": {c: float((p == i).mean()) for i, c in enumerate(CL)}}
    lk, tl = gl == 0, gl != 0
    out["lane_keep_recall"] = float((p[lk] == 0).mean()) if lk.any() else None
    out["turn_recall_any_side"] = float((p[tl] != 0).mean()) if tl.any() else None
    both = tl & (p != 0)
    out["side_given_turn"] = float((p[both] == gl[both]).mean()) if both.any() else None
    out["n_side_given_turn"] = int(both.sum())
    out["false_turn_share_on_gt_lane_keep"] = float((p[lk] != 0).mean())
    out["confusion_rows_gt_cols_pred"] = [[int(((gl == i) & (p == j)).sum()) for j in range(3)] for i in range(3)]
    return out


gt_share = {c: float((gl == i).mean()) for i, c in enumerate(CL)}
d1 = {"N1R_s0": float(acc0.mean()), "N1R_s1": float(acc1.mean())}
# F_HO and paired bootstrap interval
fho = abs(d1["N1R_s1"] - d1["N1R_s0"])
pair = OR.cell_delta(CI, acc1, acc0, eid, True, None, NB)
# controls
k1 = OR.cell_delta(CI, acc0, acc0, eid, True, 0.0, NB)
K1 = {"delta": k1["delta"], "lo": k1["lo"], "hi": k1["hi"], "pass": k1["delta"] == 0.0 and k1["lo"] == 0.0 and k1["hi"] == 0.0}
maj = int(np.bincount(gl, minlength=3).argmax())
const_acc = ((np.full(N, maj) == gl).astype(float)).mean()
maj_share = float((gl == maj).mean())
KC = {"majority_class": CL[maj], "majority_share": maj_share, "constant_accuracy": float(const_acc), "pass": bool(const_acc == maj_share)}
# K-RE: N1R_s0 (NAVRB2) minus N1_s0 (STRAT1), all 141-episode windows, TAC-LAT, vs the banked read
rb = json.load(open("F:/TanitAD-scratch/strat1/navrb2/read/out/RESULT_NAVRB2.json", encoding="utf-8"))
try:
    banked_cell = rb["pairs"]["N1R-N1"]["all_window_24_cells_under_F2"]["0"]["TAC-LAT"]
except KeyError:
    banked_cell = None
kre = {"banked_cell": banked_cell}
if banked_cell is not None:
    rr = load("F:/TanitAD-scratch/strat1/navrb2/read/t1/dump_NAVRB2-N1R_s0_m10_causal_heads", 141)
    bb = load("F:/TanitAD-scratch/strat1/read/t1/dump_STRAT1-N1_s0_m10_causal_heads", 141)
    g_r, pr = lat_labels(rr)
    g_b, pb = lat_labels(bb)
    ar, ab = (pr == g_r).astype(float), (pb == g_b).astype(float)
    mine = OR.cell_delta(CI, ar, ab, rr["eid"], True, None, NB)
    kre.update({"mine": {k: mine[k] for k in ("delta", "lo", "hi", "n_windows", "n_episodes")}, "acc_N1R_s0": float(ar.mean()), "acc_N1_s0": float(ab.mean()),
                "pass": bool(abs(mine["delta"] - banked_cell["delta"]) < 1e-12 and abs(mine["lo"] - banked_cell["lo"]) < 1e-9 and abs(mine["hi"] - banked_cell["hi"]) < 1e-9)})
out = {"tool": "code/ho_floor.py (NAVRBD step 0)", "evidence_class": "MEASURED (ours; T1-kin dumps, zero training, CPU)",
       "tier": "T1-kin (self-action open loop, m = 10, causal bucket; pinned t1 tool), route supplied along the driven path (optimistic by construction)",
       "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B {NB}, seed 0 ({CI.SOURCE}); answers 'another draw of EPISODES' only (not another training run, not another inference run)",
       "tac_lat_path": "obd_read.to_cells (obd_read.py:347-351) over obd_geom.build().labels = read_otband.labels (read_otband.py:112-118, factor_from_kinematics); load via strat1_read.load_dump_n (strat1_read.py:124); cell test via obd_read.cell_delta (obd_read.py:114)",
       "grid": {"windows": N, "episodes": int(len(np.unique(eid))), "sha256": gsha, "match": grid_match},
       "D1_HO": d1, "F_HO": fho, "F_HO_paired_bootstrap_s1_minus_s0": {k: pair[k] for k in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes")},
       "gt_class_share": gt_share, "descriptive": {"N1R_s0": descr(p0), "N1R_s1": descr(p1)},
       "controls": {"K1_self_pair": K1, "K-CONST": KC, "K-RE": kre},
       "dump_provenance": {a: {"done_file": open(f"{W}/t1/DONE_{a}").read().strip(), "t1_json_sha256": sha(f"{W}/t1/t1v8f_NAVRBD-HO-{a}_m10_causal_heads.json")} for a in ("N1R_s0", "N1R_s1")}}
p = f"{W}/out/HO_FLOOR.json"
json.dump(out, open(p, "w"), indent=1, default=float)
open(p + ".sha256", "w").write(sha(p) + "  HO_FLOOR.json\n")
print(json.dumps(out, indent=1, default=float))
