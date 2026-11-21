"""Gate C0C-ID (PREREG_T2_DOMINANCE section 4, G1; AGENT M, 2026-11-06). CPU only. Banked data and the banked T1-kin dumps only: no stage-2/3/4 rollout is opened.

    python -I c0c_gate.py --out ../C0C_ID_RESULT.json [--n 8] [--n-kbox 4]

What it proves, for BOTH training seeds, and WHICH parts are bit for bit and why:
  ID-A1 (EXACT, every row) the banked T1-kin composite dump `rr_scratch/out/C0C_s{s}/ticks/ticks_C0C_s{s}.npz` (2399 windows x 3 ticks) and the reindexed dump `ri/out/RIC0C_s0/...`:
        the dump's OWN float32 `plan_raw` and `P` through `t2v8f_c0c.composite_plan` (the arm's arithmetic: the RI tool's smooth_plan then feasible_plan, a batch of one) give the dump's `plan`
        and `plan_s` BIT FOR BIT on every row. No forward is involved, so no executor noise.
  ID-B  (EXACT, live) the harness' own C0C arm (built by `t2v8f_run.build_arm` after `t2v8f_c0c.install`, fed the banked tokens, `.inner.plan(...)`) against the RI tool's OWN
        `model_planner(model, "cpu", smooth="sg72", feasible="kbox")` run through `ri.rollout` on the same window: plan, S-only plan and raw plan bit-equal.
  ID-A2 (TOLERANCE) the arm end to end against the banked dump, within TOL_M. A bit-exact end-to-end match with a BATCHED dump is not a property of any correct arm: the float32 CPU forward itself moves
        with the batch size (MEASURED here: `noise_attribution`). The tolerance is therefore declared and the identity is carried by A1 and ID-B, which are exact.
  RED   smooth off, feasible off (only counted on windows where the kbox projection is active: the draw adds such windows), and the prior swapped for zeros: each must break the exact identity
        in A1 (on the dump rows) AND in ID-B (against the live tool).
  OFF   with smooth AND feasible off the C0C arm returns the harness' own C0 arm's plan exactly (array_equal), with C0's `extra` unchanged apart from the keys the composite adds.
The result file is the schema `read_t2_dominance.py --g1` takes: {"gate": "G1_C0C_ID", "pass", "red_arms", "off_off_reproduces_C0", ...} plus the evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import types

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("T2A_STUB", "1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

ART = "E:/Projects/TanitAD-artifacts"
CKPT = {s: f"{ART}/v8f-recover/M2b_s{s}_recft_C0/ckpt_last.pt" for s in (0, 1)}
CACHE = ART + "/sfresh_inputs/cache"
TRACKS = ART + "/sfresh_inputs/inputs/native_tracks.pt"
DATA = ART + "/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
DUMP = {0: "C:/Users/Admin/rr_scratch/out/C0C_s0/ticks/ticks_C0C_s0.npz", 1: "C:/Users/Admin/rr_scratch/out/C0C_s1/ticks/ticks_C0C_s1.npz"}
DUMP_RI = {0: "F:/TanitAD-scratch/motion_c/ri/out/RIC0C_s0/ticks/ticks_RIC0C_s0.npz"}
ADDED_KEYS = ("arm", "plan_raw", "plan_s", "composite")          # the keys CompositePolicy.from_forward adds / rewrites
TOL_FWD_M = 1e-4      # ID-A2 only, on the raw plan and the prior. MEASURED (noise_attribution, 16 windows, seed 0): the float32 CPU forward moves by up to ~1.1e-5 m between a batch of 1 and a batch of 16, and by the same amount
                      # against the dump (a smoke over 8 more windows read up to 1.7e-5 m). 1e-4 m is >= 5x that noise.
TOL_M = 1e-3          # ID-A2 only, on the composite plan: the kbox curvature clamp is a nonlinear projection that AMPLIFIES the forward noise (MEASURED up to 1.5e-4 m from 1.7e-5 m on a clamped window). 1e-3 m is ~7x that,
                      # and ~4 orders below every regression's deviation (the A1  are metres). The tolerances were set after the noise was seen, and say so: they do NOT prove the identity, A1 and ID-B (exact) do.
VARIANTS = {"smooth_off": {"smooth": "off"}, "feasible_off": {"feasible": "off"}, "prior_zeros": {"prior_hook": "zeros"}}
_W = {}


def sha256_file(p, n=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ the world (banked inputs, shared by the two seeds)
def world():
    if _W:
        return _W
    import torch
    import t2v8f_c0c as C
    import t2v8f_run as R
    R.setup()                                                    # tip tree first on sys.path, kit renderer code after it (T2A_STUB=1: stub renderer)
    C.install(R)                                                 # BEFORE the arms are built: this is the harness' own wiring
    from tanitad.data import v8f_mini_data as D
    from tanitad.models import v8f_mini as M
    ri = C.load_ri_tool()                                        # by path, sha256 checked (0417c838...)
    ck = torch.load(CKPT[0], map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    tracks = D.load_native_tracks(TRACKS)
    J = D.load_join(DATA + "/join_eval141.npz")
    store = D.ClipStore(CACHE, "eval", lru=8, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    wins = D.eval_grid_windows(DATA + "/eval_grid_windows.json", J)
    proxy = ri._OneClipStore()
    ds = D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    models = {}
    for s in (0, 1):
        c = torch.load(CKPT[s], map_location="cpu", weights_only=False)
        m = M.V8FMini(M.V8FMiniConfig(**c["cfg"]), M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
        m.load_state_dict(c["model"])
        m.eval()
        models[s] = m
    R.resolve_ckpt = lambda arm, root: CKPT[int(arm[-1])]        # dev box: the two C0 checkpoints (box 2 resolves C0_s{s}.pt under the package; C0C_s{s} uses the same file)
    _W.update(R=R, C=C, ri=ri, D=D, M=M, tracks=tracks, J=J, store=store, wins=wins, proxy=proxy, ds=ds, models=models, torch=torch)
    return _W


def banked_tokens(clip):
    return world()["torch"].load(f"{CACHE}/eval/clips/{clip}.pt", map_location="cpu", weights_only=True)["pool4x10"]


def harness_policy(seed, win, arm):
    """The harness' own policy for `arm` in {C0, C0C}, built by `t2v8f_run.build_arm`, seeded from the banked tokens the way `tests/test_t2v8f_gates.py::make_policy` does.
    -> (inner policy: V8FMiniT2Policy for C0, CompositePolicy for C0C, v0)."""
    W = world()
    R = W["R"]
    args = types.SimpleNamespace(arm=f"{arm}_s{seed}", t2_encoder="stub", t2_hub=None, t2_device="cpu", t2_schedule="periodic", t2_ckpt_root=None, t2_blind_mean=None)
    entry = R.build_arm(args)
    pol = entry.inner
    ci, t0 = win[0], win[1]
    clip = str(W["J"]["clip"][ci])
    o, n = int(W["J"]["off"][ci]), int(W["J"]["n"][ci])
    tr = W["tracks"][clip][:2 * t0 + 1]
    nav = W["J"]["nav"][o:o + n]
    pol.seed_track(tr, nav_rows=nav)
    pol.set_nav(row=nav[t0])
    tok = banked_tokens(clip)
    for s in range(t0 + 1):
        pol.push_tokens(s, tok[s])
    return pol, float(tr[-1, 3])


def tool_plan(seed, win, smooth="sg72", feasible="kbox"):
    """The RI tool's OWN composite planner through `ri.rollout` (one tick, m = 30, CPU, batch of one). -> (plan float32 [30,2], plan_raw, plan_s)."""
    W = world()
    ri = W["ri"]
    st = ri.make_states(W["store"], W["tracks"], W["J"], [win], "deadreckon")
    planner = ri.model_planner(W["models"][seed], "cpu", smooth=smooth, feasible=feasible)
    plan0, _ = ri.rollout(planner, st, W["ds"], W["proxy"], m=30, schedule="periodic", scene="logged", limit="causal", device="cpu")
    return plan0[0].astype(np.float32), None if planner.plan_raw is None else planner.plan_raw[0], None if planner.plan_s is None else planner.plan_s[0]


# ------------------------------------------------------------------------------------------------ banked references
def dump_rows(seed, ri_dump=False):
    z = np.load((DUMP_RI if ri_dump else DUMP)[seed], allow_pickle=False)
    row = {int(w): i for i, (w, t) in enumerate(zip(z["win"], z["tick"])) if int(t) == 0}
    return z, row


def f32(a):
    return np.asarray(a, np.float64).astype(np.float32)


def exact32(a64):
    a32 = np.asarray(a64, np.float64).astype(np.float32)
    return a32, bool(np.array_equal(a32.astype(np.float64), np.asarray(a64, np.float64)))


def arithmetic_identity(seed, ri_dump=False, stride=1):
    """ID-A1, EXACT. The dump's OWN float32 `plan_raw` and `P` go through `composite_plan` (the arm's arithmetic, a batch of one); the result must equal the dump's `plan` and `plan_s` bit for bit
    on EVERY row. No forward is involved. The three deliberate regressions run on the same rows; each must differ from the dump's plan on at least one row."""
    C = world()["C"]
    T = world()["ri"]
    z = np.load((DUMP_RI if ri_dump else DUMP)[seed], allow_pickle=False)
    idx = range(0, len(z["plan"]), stride)
    out = {"dump": (DUMP_RI if ri_dump else DUMP)[seed], "rows": len(idx), "inputs_exact_float32": True, "eq_plan": 0, "eq_plan_s": 0, "max_abs_plan": 0.0,
           "red": {k: {"n_differ": 0, "max_abs_dev": 0.0} for k in VARIANTS}}
    for i in idx:
        raw, ok1 = exact32(z["plan_raw"][i])
        pri, ok2 = exact32(z["P"][i])
        want, ok3 = exact32(z["plan"][i])
        want_s, ok4 = exact32(z["plan_s"][i])
        out["inputs_exact_float32"] &= ok1 and ok2 and ok3 and ok4
        comp, s_ = C.composite_plan(raw, pri, tool=T)
        out["eq_plan"] += int(np.array_equal(comp, want))
        out["eq_plan_s"] += int(np.array_equal(s_, want_s))
        out["max_abs_plan"] = max(out["max_abs_plan"], float(np.abs(comp.astype(np.float64) - want).max()))
        for name, kw in VARIANTS.items():
            pr = np.zeros_like(pri) if kw.get("prior_hook") == "zeros" else pri
            c2, _ = C.composite_plan(raw, pr, kw.get("smooth", C.SMOOTH), kw.get("feasible", C.FEASIBLE), tool=T)
            out["red"][name]["n_differ"] += int(not np.array_equal(c2, want))
            out["red"][name]["max_abs_dev"] = max(out["red"][name]["max_abs_dev"], float(np.abs(c2.astype(np.float64) - want).max()))
    out["pass"] = bool(out["inputs_exact_float32"] and out["eq_plan"] == out["rows"] and out["eq_plan_s"] == out["rows"])
    for v in out["red"].values():
        v["verdict"] = "RED" if v["n_differ"] > 0 else "GREEN"
    return out


def pick_windows(seed, n, n_kbox):
    """n evenly spaced windows (as the gates' `pick`) + up to n_kbox windows where the banked composite differs from its own S-only plan (the kbox projection is ACTIVE there: only such
    a window can make the 'feasible off' regression bite)."""
    W = world()
    z, row = dump_rows(seed)
    wins = W["wins"]
    base = [int(i) for i in np.linspace(40, len(wins) - 40, n).round().astype(int)]
    active = [w for w, i in row.items() if not np.array_equal(z["plan"][i].astype(np.float32), z["plan_s"][i].astype(np.float32))]
    rng = np.random.default_rng(7)
    extra = sorted(int(w) for w in rng.choice(active, size=min(n_kbox, len(active)), replace=False)) if active else []
    out = base + [w for w in extra if w not in base]
    return out, {"n_even": len(base), "n_kbox_active_added": len(extra), "n_kbox_active_in_dump": len(active), "n_rows_tick0": len(row)}


def live_identity(seed, widx):
    """ID-B (exact, live), ID-A2 (toleranced, vs the dump), the live RED arms and off/off on one window."""
    W = world()
    win = W["wins"][widx]
    z, row = dump_rows(seed)
    i = row[widx]
    assert (int(z["ci"][i]), int(z["t0"][i])) == (int(win[0]), int(win[1])), f"window {widx}: the dump's (ci, t0) differ from the grid's"
    tp, praw, ps = tool_plan(seed, win)
    pol, v0 = harness_policy(seed, win, "C0C")
    plan, extra = pol.plan(None, None, v0, None)
    got = f32(plan)
    r = {"win": widx, "ci": int(win[0]), "t0": int(win[1]), "eq_tool": bool(np.array_equal(got, tp)), "eq_tool_raw": bool(np.array_equal(f32(extra["plan_raw"]), f32(praw))),
         "eq_tool_s": bool(np.array_equal(f32(extra["plan_s"]), f32(ps))), "kbox_active": bool(not np.array_equal(f32(ps), tp))}
    r["a2_max_abs_m"] = {"plan": float(np.abs(got.astype(np.float64) - z["plan"][i]).max()), "raw": float(np.abs(f32(extra["plan_raw"]).astype(np.float64) - z["plan_raw"][i]).max()),
                         "prior": float(np.abs(f32(extra["p10_prior"]).astype(np.float64) - z["P"][i]).max())}
    r["a2_within_tol"] = bool(r["a2_max_abs_m"]["plan"] <= TOL_M and r["a2_max_abs_m"]["raw"] <= TOL_FWD_M and r["a2_max_abs_m"]["prior"] <= TOL_FWD_M)
    if seed in DUMP_RI:
        zr, rowr = dump_rows(seed, ri_dump=True)
        r["a2_ri_dump_max_abs_m"] = float(np.abs(got.astype(np.float64) - zr["plan"][rowr[widx]]).max())
        r["a2_ri_dump_within_tol"] = bool(r["a2_ri_dump_max_abs_m"] <= TOL_M)
    red = {}
    for name, kw in VARIANTS.items():
        pol.smooth, pol.feasible = kw.get("smooth", "sg72"), kw.get("feasible", "kbox")
        pol._prior_hook = (lambda p: np.zeros_like(p)) if kw.get("prior_hook") == "zeros" else None
        p2, _ = pol.plan(None, None, v0, None)
        red[name] = bool(np.array_equal(f32(p2), tp))                                  # True = the regression did NOT break the identity
    pol.smooth, pol.feasible, pol._prior_hook = "off", "off", None
    p3, e3 = pol.plan(None, None, v0, None)                                            # off/off on the SAME forward ...
    p0, e0 = harness_policy(seed, win, "C0")[0].plan(None, None, v0, None)             # ... against the harness' own C0 arm
    r["red_still_equal"] = red
    r["off_off"] = {"plan_equal_C0": bool(np.array_equal(p3, p0)), "extra_equal_apart_from_added": bool({k: v for k, v in e0.items() if k != "arm"} == {k: v for k, v in e3.items() if k not in ADDED_KEYS}),
                    "added_keys": sorted(set(e3) - set(e0))}
    return r


def noise_probe(seed=0, k=16):
    """MEASURED attribution of the end-to-end difference to the dump: the SAME tool, CPU, a batch of k windows against a batch of 1 (raw plan, composite plan), max abs in metres."""
    W = world()
    ri = W["ri"]
    wins_idx = [int(i) for i in np.linspace(40, len(W["wins"]) - 40, k).round().astype(int)]
    wins = [W["wins"][i] for i in wins_idx]

    def run(ws):
        st = ri.make_states(W["store"], W["tracks"], W["J"], ws, "deadreckon")
        pl = ri.model_planner(W["models"][seed], "cpu", smooth="sg72", feasible="kbox")
        p0, _ = ri.rollout(pl, st, W["ds"], W["proxy"], m=30, schedule="periodic", scene="logged", limit="causal", device="cpu")
        return p0, pl.plan_raw
    pk, rk = run(wins)
    d_raw, d_comp = 0.0, 0.0
    for j, w in enumerate(wins):
        p1, r1 = run([w])
        d_raw = max(d_raw, float(np.abs(r1[0].astype(np.float64) - rk[j]).max()))
        d_comp = max(d_comp, float(np.abs(p1[0].astype(np.float64) - pk[j]).max()))
    return {"seed": seed, "k": k, "max_abs_m_raw_batch1_vs_batchk": d_raw, "max_abs_m_composite_batch1_vs_batchk": d_comp}


def manifest_ckpt_pins():
    """The dev-box checkpoints this gate ran on against the stage-1 package manifest's `ckpt/C0_s{s}.pt` (what box 2 runs): sha256 must be equal."""
    import tarfile
    with tarfile.open(f"{HERE}/../stage1/t2v8f_stage1.tar") as t:
        m = json.load(t.extractfile("pkg/MANIFEST.json"))
    want = {e["path"]: e["sha256"] for e in m["files"]}
    out = {}
    for s in (0, 1):
        got = sha256_file(CKPT[s])
        out[f"s{s}"] = {"dev_ckpt": CKPT[s], "sha256": got, "manifest_sha256": want[f"ckpt/C0_s{s}.pt"], "equal": got == want[f"ckpt/C0_s{s}.pt"]}
    return out


def run_gate(n=8, n_kbox=4, stride=1, seeds=(0, 1)):
    res = {"gate": "G1_C0C_ID", "pass": False, "red_arms": {}, "off_off_reproduces_C0": False, "tolerance_m_A2": {"composite": TOL_M, "raw_and_prior": TOL_FWD_M}, "seeds": {}, "why": []}
    for s in seeds:
        wins_idx, draw = pick_windows(s, n, n_kbox)
        a1 = arithmetic_identity(s, stride=stride)
        a1ri = arithmetic_identity(s, ri_dump=True, stride=stride) if s in DUMP_RI else None
        live = [live_identity(s, w) for w in wins_idx]
        res["seeds"][f"s{s}"] = {"draw": draw, "windows": wins_idx, "A1_arithmetic_vs_dump": a1, "A1_arithmetic_vs_ri_dump": a1ri, "live": live}
        if not a1["pass"]:
            res["why"].append(f"s{s} A1 arithmetic identity vs the dump failed: {a1['eq_plan']}/{a1['rows']}")
        if a1ri is not None and not a1ri["pass"]:
            res["why"].append(f"s{s} A1 arithmetic identity vs the reindexed dump failed: {a1ri['eq_plan']}/{a1ri['rows']}")
        for r in live:
            for k in ("eq_tool", "eq_tool_raw", "eq_tool_s", "a2_within_tol"):
                if not r[k]:
                    res["why"].append(f"s{s} win {r['win']}: {k} False")
            if not (r["off_off"]["plan_equal_C0"] and r["off_off"]["extra_equal_apart_from_added"]):
                res["why"].append(f"s{s} win {r['win']}: off/off != C0")
        if not any(r["kbox_active"] for r in live):
            res["why"].append(f"s{s}: no drawn window has the kbox projection active, so 'feasible off' could not bite")
    for a in VARIANTS:
        a1_ok = all(res["seeds"][f"s{s}"]["A1_arithmetic_vs_dump"]["red"][a]["verdict"] == "RED" for s in seeds)
        live_all = all(not r["red_still_equal"][a] for s in seeds for r in res["seeds"][f"s{s}"]["live"] if (a != "feasible_off" or r["kbox_active"]))
        res["red_arms"][a] = "RED" if (a1_ok and live_all) else "GREEN"
        if res["red_arms"][a] != "RED":
            res["why"].append(f"RED arm {a} did not break the identity (A1 {a1_ok}, live {live_all})")
    res["off_off_reproduces_C0"] = all(r["off_off"]["plan_equal_C0"] and r["off_off"]["extra_equal_apart_from_added"] for s in seeds for r in res["seeds"][f"s{s}"]["live"])
    res["noise_attribution"] = noise_probe(seeds[0])
    res["ckpt_pins"] = manifest_ckpt_pins()
    if not all(v["equal"] for v in res["ckpt_pins"].values()):
        res["why"].append("the dev-box checkpoints are not the stage-1 package's C0_s{s}.pt")
    res["pass"] = not res["why"]
    C = world()["C"]
    res["pins"] = {"ri_tool_sha256": C.RI_TOOL_SHA, "ri_tool_loaded_sha256": C.sha256_file(world()["ri"].__file__), "c0c_py_sha256": C.sha256_file(C.__file__), "c0c_gate_py_sha256": sha256_file(os.path.abspath(__file__)),
                   "dumps": {f"s{s}": {"path": DUMP[s], "sha256": sha256_file(DUMP[s])} for s in seeds}, "ri_dump_s0": {"path": DUMP_RI[0], "sha256": sha256_file(DUMP_RI[0])}}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--n-kbox", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1, help="A1 row stride (1 = every one of the 7,197 rows per dump)")
    a = ap.parse_args(argv)
    res = run_gate(a.n, a.n_kbox, a.stride)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"pass": res["pass"], "red_arms": res["red_arms"], "off_off_reproduces_C0": res["off_off_reproduces_C0"], "why": res["why"][:6]}))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
