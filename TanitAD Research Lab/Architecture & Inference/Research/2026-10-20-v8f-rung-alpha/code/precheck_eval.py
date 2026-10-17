"""PRE-CHECK step 2: score P10-pose on the S1 eval139 grid against ha0_ext_pose, P10 (steer, S1's) and refcv7-50,400 (banked).

usage: precheck_eval.py S2|S6 0|1 [--analysis] [--artifact <p10_pose_prior.pt>]
Writes C:/Users/Admin/v8fa1/pre/eval_<H>_s<seed>.json (+ analysis_<H>_s<seed>.json with --analysis).

The arms: os (banked refcv7-50,400 T1 dump), ha0_ext (steer, banked), P10 (S1's steer fit), ha0_ext_pose, kd_x_pose,
P10pose (this pre-check's S1-recipe fit; POSE ONLY), and with --artifact the SHIPPED 30-knot P10-pose read through the
decoder path's own tick conversion (knots_to_ticks) at the grid instants.
Estimator: paired episode-cluster bootstrap B 2000 seed 0 (taniteval/ci.py). Tier T1 (self-action open loop).
"""
import glob
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                                   # noqa: E402
import p10pose_np as L                                                               # noqa: E402
from taniteval import floors_kin as fk                                               # noqa: E402
from taniteval.ci import paired_episode_cluster_bootstrap as pb                      # noqa: E402
import torch                                                                         # noqa: E402

torch.set_num_threads(int(os.environ.get("S1_THREADS", "4")))
H, SEED = sys.argv[1], int(sys.argv[2])
DO_AN = "--analysis" in sys.argv
ART = sys.argv[sys.argv.index("--artifact") + 1] if "--artifact" in sys.argv else None
OUT = "C:/Users/Admin/v8fa1/pre"
CFG = {"S2": dict(sel=L.H_S2, n_fine=20, dt=0.5, tag=("dump_s0", "dump_s1"), man=("panel_s0", "panel_s1")),
       "S6": dict(sel=L.H_S6, n_fine=60, dt=1.0, tag=("s6_s0", "s6_s1"), man=("s6_s0", "s6_s1"))}[H]
sel, n_fine, dt = CFG["sel"], CFG["n_fine"], CFG["dt"]
N_BOOT = 2000
t_all = time.time()

eman = L.load_manifest(L.EVAL_MAN)
W = L.eval_windows(eman, CFG["tag"][SEED], sel)
n, ep, g = len(W["kin"]), W["ep"], W["d_g"].astype(np.float64)
assert W["check"]["v0"] < 1e-4 and W["check"]["g"] < 1e-4, W["check"]
fl = L.pose_floors(W["kin"], n_fine, sel)

# ---- S1's steer floors, rebuilt EXACTLY as S1 did (controls: they must reproduce the banked S1 numbers) -----------------
steer_kin = np.concatenate([fk.t0_kinematics(eman["poses"][int(c)][:, 3], eman["actions"][int(c)][:, 0],
                                              W["t0"][W["clip"] == c], stride=1, dt=0.1) for c in dict.fromkeys(W["clip"].tolist())])
idx = [h - 1 for h in sel]
h0s = fk.ha0_paths(steer_kin[:, 0], 0.1, n_fine)
hes = fk.ha0_ext_paths(steer_kin, 0.1, n_fine, action_units="steer")
kdx_s = fk.kd_x(h0s, hes)[:, idx]
ck = torch.load(f"{L.S1_SCRATCH}/p10_{H}.pt", weights_only=False)
net = fk._make_mlp(8, len(sel) * 2)
net.load_state_dict(ck["state"])
P10_steer = fk.predict_p10((net, (ck["xm"], ck["xs"], ck["ym"]), ck["best"]), fk.kin_features(steer_kin), kdx_s.astype(np.float32))
ck2 = torch.load(f"{OUT}/p10pose_{H}.pt", weights_only=False)
net2 = fk._make_mlp(8, len(sel) * 2)
net2.load_state_dict(ck2["state"])
P10_pose = fk.predict_p10((net2, (ck2["xm"], ck2["xs"], ck2["ym"]), ck2["best"]), fk.kin_features(W["kin"]),
                          fl["kd_x_pose"].astype(np.float32))

arms = {"os": W["d_os"].astype(np.float64), "ha0_ext": hes[:, idx], "P10": P10_steer.astype(np.float64),
        "ha0_ext_pose": fl["ha0_ext_pose"], "kd_x_pose": fl["kd_x_pose"], "P10pose": P10_pose.astype(np.float64)}
ctrl = {"v0_pose_vs_dump_max_abs": W["check"]["v0"], "g_vs_dump_max_abs": W["check"]["g"],
        "ha0_ext_steer_vs_dump_arm_max_abs": float(np.abs(hes[:, idx] - W["d_ha0_ext"]).max()),
        "os_is_dump_arm": True}

if ART:
    from tanitad.models.kinematic_prior import knots_to_ticks
    from tanitad.models.p10_prior import load_frozen_p10
    fp = load_frozen_p10(ART, variant="pose")
    pw = []
    for c in dict.fromkeys(W["clip"].tolist()):
        P = eman["poses"][int(c)]
        pw.append(L.pose_kin(P[:, 3], P[:, 2], W["t0"][W["clip"] == c]))
    kin3 = torch.tensor(np.concatenate(pw), dtype=torch.float64)
    with torch.no_grad():
        p30 = fp(kin3).to(torch.float64)
        ticks = knots_to_ticks(p30).numpy()
    arms["P10pose_artifact"] = ticks[:, idx]
    ctrl["artifact"] = {"path": ART, "file_sha256": fp.file_sha, "content_sha256": fp.content_sha()}

ade_w = {k: fk.ade(v, g) for k, v in arms.items()}
fde_w = {k: fk.fde(v, g) for k, v in arms.items()}
PAIRS = [("os", "P10pose"), ("os", "ha0_ext_pose"), ("P10pose", "ha0_ext_pose"), ("P10pose", "P10"), ("P10pose", "kd_x_pose"),
         ("os", "P10"), ("os", "ha0_ext"), ("ha0_ext_pose", "ha0_ext"), ("P10pose", "P10pose")]
if ART:
    PAIRS += [("os", "P10pose_artifact"), ("P10pose_artifact", "ha0_ext_pose"), ("P10pose_artifact", "P10pose")]
paired = {f"{a}-{b}": {"ade": pb(ade_w[a], ade_w[b], ep, n_boot=N_BOOT, seed=0),
                       "fde": pb(fde_w[a], fde_w[b], ep, n_boot=N_BOOT, seed=0)} for a, b in PAIRS}
means = {k: {"ade": float(ade_w[k].mean()), "fde": float(fde_w[k].mean())} for k in arms}
out = {"horizon": H, "inference_seed": SEED, "tier": "T1 (self-action OPEN loop)", "n_windows": int(n),
       "n_episodes": int(len(np.unique(ep))), "estimator": "paired episode-cluster bootstrap, B 2000, seed 0 (taniteval/ci.py)",
       "p10pose_fit": ck2["meta"], "controls": ctrl, "means": means, "paired_x_minus_y": paired}
json.dump(out, open(f"{OUT}/eval_{H}_s{SEED}{'_art' if ART else ''}.json", "w"), indent=1)
d = paired["os-P10pose"]["ade"]
print(f"[{H} s{SEED}] direct done {time.time()-t_all:.0f}s  os-P10pose ADE {d['delta']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]", flush=True)

if DO_AN:
    dd = f"{OUT}/dumps/{H}_s{SEED}"
    shutil.rmtree(dd, ignore_errors=True)
    os.makedirs(dd, exist_ok=True)
    files = sorted(glob.glob(f"{L.BATT}/{CFG['tag'][SEED]}/ep*.npz"))
    off = 0
    dump_arms = {k: v.astype(np.float32) for k, v in arms.items() if k not in ("os", "ha0_ext")}
    for f in files:
        z = np.load(f)
        m = len(z["ws"])
        sl = slice(off, off + m)
        off += m
        assert np.array_equal(W["t0"][sl], z["ws"])
        np.savez(os.path.join(dd, os.path.basename(f)), g=z["g"], v0=z["v0"], ws=z["ws"], eid=z["eid"],
                 clip_index=z["clip_index"], os=z["os"], ha=z["ha"], ha0=z["ha0"], ha0_ext=z["ha0_ext"],
                 **{k: v[sl] for k, v in dump_arms.items()})
    man = json.load(open(f"{L.BATT}/{CFG['man'][SEED]}/manifest.json", encoding="utf-8"))
    man["s1_derived"] = {"note": "DERIVED dump: os/ha/ha0/ha0_ext are the banked refcv7-50,400 T1 arrays; the other arms were ADDED "
                                 "by the v8F alpha-1 pre-check (pose-only P10).", "source_dump": f"{L.BATT}/{CFG['tag'][SEED]}"}
    json.dump(man, open(os.path.join(dd, "manifest.json"), "w"), indent=1)
    sys.path.insert(0, f"{L.ROOT}/taniteval/tools")
    import refcv3_arm as rc3                                                          # noqa: E402
    EXTRA = [("P10pose", "os", "paired_os_minus_P10pose"), ("ha0_ext_pose", "os", "paired_os_minus_ha0extpose"),
             ("ha0_ext_pose", "P10pose", "paired_P10pose_minus_ha0extpose"), ("P10", "P10pose", "paired_P10pose_minus_P10"),
             ("P10", "os", "paired_os_minus_P10"), ("P10pose", "P10pose", "paired_P10pose_minus_P10pose")]
    if ART:
        EXTRA += [("P10pose_artifact", "os", "paired_os_minus_P10poseart"),
                  ("ha0_ext_pose", "P10pose_artifact", "paired_P10poseart_minus_ha0extpose")]
    _orig = rc3.t1.analyze

    def _wrap(files_, **kw):
        kw["paired"].extend(EXTRA)
        return _orig(files_, **kw)
    rc3.t1.analyze = _wrap
    tiers = {k: "T1" for k in dump_arms}
    t = time.time()
    rec = rc3.analyze_refcv3(dd, n_boot=N_BOOT, seed=0, tiers=tiers, lead_block=rc3.ra.LEAD_BLOCK_DEFAULT)
    rec["precheck_note"] = "analyze_refcv3 over the derived dump; extra pairs: " + json.dumps([e[2] for e in EXTRA])
    json.dump(rec, open(f"{OUT}/analysis_{H}_s{SEED}{'_art' if ART else ''}.json", "w"), default=str)
    print(f"[{H} s{SEED}] four-family analysis {time.time()-t:.0f}s", flush=True)
print(f"ZZEVAL-{H}-s{SEED}-DONEZZ")
