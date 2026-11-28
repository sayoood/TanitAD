"""Dry run of l2b_score.py on a SYNTHETIC locus trace with PLANTED answers (stage disabled by a runtime patch: the stage itself is pinned by K1/K_J elsewhere).
Planted: FB windows get a cruise candidate with p=0.7 (of which a GT-copy 'good' one with p=0.5); MB windows get a decel candidate with p=0.6; CT/DR windows always have all classes.
The scorer must read the planted shares exactly and the label logic must follow.  A MUTATION (class codes swapped) must make it disagree."""
import sys, os, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
import numpy as np
import l2_common as C, l2_score as S2, l2_oracle_s2_rows as OR
import l2b_score as B

seed = 0
OUT = "C:/Users/Admin/refcv8_pf/refcv9_l2/dev30k/l2b_dry"
os.makedirs(OUT, exist_ok=True)
P0 = OR.load_panel(0)
base = S2.load_l1_base(B.L1_BANK, 0)
E = C.load_eval_pf(0)
n = len(P0["ws"]); N = 149; SEL = 10
plan8 = np.stack([base["plan8"][base["idx"][(int(c), int(w))]] for c, w in zip(P0["clip"], P0["ws"])]).astype(np.float32)
v0 = P0["v0"]
T = C.ALL_T
rng = np.random.default_rng(7)
def path(v, y=0.0): return np.stack([v * T, np.full_like(T, y)], -1).astype(np.float32)
# sets as the scorer will compute them (stage disabled -> raw S6 slots)
S6s = plan8[:, C.S6_SLOTS]
gt_all = None
# expected truth bookkeeping on the S6 rows
key6 = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(E["ex_clip"][E["rows"]], E["ex_ws"][E["rows"]]))}
g6 = E["g"]; v06 = E["v0"]
gt6 = C.plan_class(g6, v06, E["dt"])
row_of = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(P0["clip"], P0["ws"]))}
idx6 = np.array([row_of[k] for k in key6])                     # S6 row -> trace row
em = C.plan_class(plan8[idx6][:, C.S6_SLOTS], v06, E["dt"])
comp = E["compliant"]
FB = comp & (gt6 == "cruise") & (em == "decel"); MB = comp & (gt6 == "decel") & (em != "decel")
plant_cruise = rng.random(len(comp)) < 0.7; plant_good = rng.random(len(comp)) < 0.5; plant_decel = rng.random(len(comp)) < 0.6
fan = np.zeros((n, N, 8, 2), np.float32)
block = rng.random(len(comp)) < 0.2                                   # FB windows whose planted cruise candidates are MASKED OUT (ineligible): EXIST must read False
reach = np.ones((n, N), np.int8)
inv = {int(t): int(i) for i, t in enumerate(idx6)}               # trace row -> S6 row
for r in range(n):
    i6 = inv.get(r)
    vv = max(float(v0[r]), 2.0)
    pc, pd, good, blk = True, True, False, False
    if i6 is not None and FB[i6]:
        pc, good, blk = bool(plant_cruise[i6]), bool(plant_cruise[i6] and plant_good[i6]), bool(block[i6])
    if i6 is not None and MB[i6]:
        pd = bool(plant_decel[i6])
    cands, tags = [], []
    for k in range(60): cands.append(path(0.0, rng.normal(0, 0.5)) if pd else path(vv + 4.0, rng.normal(0, 0.5))); tags.append("fill")
    for k in range(60): cands.append(path(vv + 4.0, rng.normal(0, 0.5))); tags.append("fill")
    if pc:
        for k in range(3): cands.append(path(vv, 40.0)); tags.append("cru")
    if good and i6 is not None:                                    # GT copy on the 8 knots (linear interp of the 6 GT instants)
        gx = np.interp(T, np.r_[0.0, C.S6_T], np.r_[0.0, g6[i6][:, 0]]); gy = np.interp(T, np.r_[0.0, C.S6_T], np.r_[0.0, g6[i6][:, 1]])
        cands.append(np.stack([gx, gy], -1).astype(np.float32)); tags.append("cru")
    while len(cands) < N - 1: cands.append(path(vv + 4.0, rng.normal(0, .5))); tags.append("fill")
    cands, tags = cands[:N - 1], tags[:N - 1]
    f = np.stack(cands[:SEL] + [plan8[r]] + cands[SEL:], 0)[:N]
    tg = tags[:SEL] + ["emit"] + tags[SEL:]
    fan[r] = f
    for j, t in enumerate(tg[:N]):
        if t == "fill" and rng.random() < 0.1: reach[r, j] = 0
        if t == "cru" and blk: reach[r, j] = 0
sv = rng.normal(0, 1, (n, N)).astype(np.float32); sv[:, SEL] = sv.max(1) + 1.0
reach[:, SEL] = 1
ones = np.ones((n, N), np.int8)
meta = {"final": True, "n_windows": n, "present_counts": {k: n for k in ("fan", "sel_score", "sel_score_v3", "prior_path", "kind", "lon6", "lat3", "reach_keep", "ceil_keep", "emit_keep")},
        "counters": {"fan_n_seen": [N], "batch_gt1": 0, "calls": n, "navnull_calls": 0}}
tr = dict(e_i=P0["clip"], t=P0["ws"] - 7, ws=P0["ws"], plan8=plan8, sel=np.full(n, SEL), sel_base=np.full(n, SEL + 1), v0_in=v0, fan=fan, sel_score=sv, sel_score_v3=sv, reach_keep=reach,
          ceil_keep=ones, emit_keep=ones, kind=np.zeros((n, N), np.int8), lon6=np.zeros((n, N), np.int8), lat3=np.zeros((n, N), np.int8), prior_path=np.asarray(P0["prior8"], np.float32),
          meta_json=np.array(json.dumps(meta)))
np.savez_compressed(f"{OUT}/locus_synth_s0.npz", **tr)
# ---- runtime patches (test only): stage = identity on the S6 slots
C.apply_stage = lambda p8, pr: np.asarray(p8)[:, C.S6_SLOTS]
B.stage_cands = lambda fan_, prior_, table, chunk=200: fan_[:, :, C.S6_SLOTS]
B.C.apply_stage = C.apply_stage
# seal: use a throw-away copy of the prereg seal check by pointing at the real files (the dry run needs PREREG_L2B.md + .sha256 to exist and match)
sys.argv = ["l2b_score.py", "--seed", "0", "--trace", f"{OUT}/locus_synth_s0.npz", "--out", f"{OUT}/score"]
rc = B.main()
r = json.load(open(f"{OUT}/score/L2B_RESULT.json"))
R = r["readings"]
exp = {"FB_exist": float((plant_cruise & ~block)[FB].mean()), "FB_good": float((plant_cruise & plant_good & ~block)[FB].mean()), "MB_exist_decel": float(plant_decel[MB].mean()),
       "n_FB": int(FB.sum()), "n_MB": int(MB.sum())}
got = {"FB_exist": R["FB"]["EXIST"]["mean"], "FB_good": R["FB"]["GOOD"]["mean"], "MB_exist_decel": R["MB"]["EXIST"]["mean"], "n_FB": R["FB"]["n"], "n_MB": R["MB"]["n"],
       "CT_exist": R["CT"]["EXIST"]["mean"], "DR_exist": R["DR"]["EXIST"]["mean"], "label": r["label"], "R1": r["R1_label"], "R2": r["R2_label"], "guard": r["guard_existence_saturated"]}
print("EXPECTED", exp); print("GOT", got); print("controls", {k: (v.get("pass") if isinstance(v, dict) else v) for k, v in r["controls"].items()})
ok = abs(exp["FB_exist"] - got["FB_exist"]) < 1e-3 and abs(exp["FB_good"] - got["FB_good"]) < 1e-3 and abs(exp["MB_exist_decel"] - got["MB_exist_decel"]) < 1e-3 and exp["n_FB"] == got["n_FB"] and exp["n_MB"] == got["n_MB"]
print("DRYRUN planted shares read exactly:", ok, "| CT identity exist == 1:", got["CT_exist"] == 1.0)

# ---- MUTATION: swap the decel/cruise class codes inside the scorer; the planted shares must NO LONGER be read
orig = B.cls_of
B.cls_of = lambda P, v0_, dt_: np.array([1, 0, 2])[orig(P, v0_, dt_)]
sys.argv = ["l2b_score.py", "--seed", "0", "--trace", f"{OUT}/locus_synth_s0.npz", "--out", f"{OUT}/score_mut"]
B.main()
rm = json.load(open(f"{OUT}/score_mut/L2B_RESULT.json"))["readings"]
print("MUTATED FB_exist", rm["FB"]["EXIST"]["mean"], "vs planted", exp["FB_exist"], "-> mutation detected:", abs(rm["FB"]["EXIST"]["mean"] - exp["FB_exist"]) > 0.05)
B.cls_of = orig
