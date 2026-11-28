#!/usr/bin/env python3
"""ka_lf6.py -- the known answers of PREREG_LEVELFLOOR_A6 (sha256 fbf5de02...) section 2, as 64-window smokes (s0, reindexed, C0_s0 + the LLR_s0 heads, CPU 12 threads), each compared SMOKE AGAINST SMOKE and bit for bit (a smoke is never compared with a banked
full rollout: batch composition changes the model's float32 output, A5 caveat), each with ONE deliberate mutation that must turn it RED.
  KA-1  DELTA = 0 equals the floor off:  --floor-lift-cap 0 (kappa 1.0, heads bare and composite; logged bare) reproduces the floor-off smoke on every common tick field and the analysis arms; F_B never binds (lf_bind all False, lf_dterm all False).
        Mutation M1 (a falsy zero: DELTA 0 read as "no cap") must break it.
  KA-2  DELTA = 1e9 equals the LF5 tool (88b275a1) at kappa 1.0: heads bare and composite, logged bare, on every common tick field (the lf_* fields included) and the analysis arms. Mutation M2 (the clip s' - s <= DELTA * t stuck at 1 m/s) must break it.
  KA-3  DELTA = 0.5 is partial: its bind rate (share of ticks with lf_bind) lies STRICTLY between DELTA 0's and DELTA 1e9's (heads bare, heads composite, logged bare), and s' - s <= DELTA * t + 1e-6 holds on every knot of every tick (lf_sp - lf_s from the tool), and an
        INDEPENDENT recomputation of s' from the composite smoke's dumped plan_s (the plan F read), v0 and lf_vf by fine-grid integration of the card's formula agrees. Mutation M3 (the DELTA term deleted) must break the strict-between leg.
Writes raw/KA_LF6.json.   usage: python -I ka_lf6.py [--only K1,K2,K3]"""
import hashlib
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
ART = "E:/Projects/TanitAD-artifacts"
MC = f"{ART}/dagland/motion_c"
RDB = f"{ART}/v8f-recover"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
SW = "F:/TanitAD-scratch/motion_c/lf6_build/smoke"
MUT_TREE = "F:/TanitAD-scratch/motion_c/lf6_build/mut/tree"
LF5_TOOL = "F:/TanitAD-scratch/motion_c/lf5_read/tree/stack/scripts/v8f_t1_eval_ri.py"
LF6_TOOL = "F:/TanitAD-scratch/motion_c/lf6_read/tree/stack/scripts/v8f_t1_eval_ri.py"
LF5_SHA = "88b275a1ca67894be01907f72b71302b72f6e0b4fa0af067f4e2c3d5286aba77"
DATA = f"{ART}/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
LF6_ONLY = ("lf_dterm", "lf_dterm_knots", "lf_s", "lf_sp")
DT = 0.2


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run_smoke(tag, tool, source=None, cap=None, composite=False, n_win=64):
    """source None = the floor OFF; else the floor on at kappa 1.0 with that v_hat source and (LF6 tool only) --floor-lift-cap ``cap``."""
    od = f"{SW}/{tag}"
    if os.path.isdir(od):
        os.replace(od, f"{od}_old_{time.strftime('%H%M%S')}")
    os.makedirs(od)
    fl = [] if source is None else ["--level-floor", "on", "--floor-kappa", "1.0", "--floor-ckpt", f"{RDB}/M2b_s0_recft_LLR/ckpt_last.pt", "--floor-vhat-source", source]
    if cap is not None:
        fl += ["--floor-lift-cap", str(cap)]
    cmd = [PY, "-I", tool, "run", "--ckpt", f"{RDB}/M2b_s0_recft_C0/ckpt_last.pt", "--tag", tag, "--tracks", f"{ART}/sfresh_inputs/inputs/native_tracks.pt", "--eval-cache", f"{ART}/sfresh_inputs/cache",
           "--eval-grid", f"{DATA}/eval_grid_windows.json", "--eval-join", f"{DATA}/join_eval141.npz", "--out-dir", od, "--replan-steps", "10", "--nav", "deadreckon", "--schedule", "periodic", "--limit", "causal",
           "--dump-ticks", f"{od}/ticks", "--dump-heads", "--device", "cpu", "--smooth", "sg72", "--scene", "reindexed", "--max-windows", str(n_win), "--n-boot", "20"] + fl + (["--feasible", "kbox"] if composite else [])
    env = dict(os.environ, OMP_NUM_THREADS="12", MKL_NUM_THREADS="12", OPENBLAS_NUM_THREADS="12", CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    os.makedirs(f"{PK}/logs", exist_ok=True)
    open(f"{PK}/logs/KA_LF6_{tag}.out", "w", encoding="utf-8", newline="\n").write(" ".join(cmd) + "\n" + r.stdout[-2000:] + r.stderr[-2000:])
    if r.returncode:
        raise SystemExit(f"smoke {tag} failed rc={r.returncode}: {r.stderr[-800:]}")
    print(f"  smoke {tag} done in {time.time() - t0:.0f}s", flush=True)
    return od


def ticks(od, tag):
    return dict(np.load(f"{od}/ticks/ticks_{tag}.npz", allow_pickle=True))


def arms(od, tag):
    return json.load(open(f"{od}/t1v8f_{tag}.json", encoding="utf-8"))["arms"]


def row_fields(t, skip=()):
    n = len(t["win"])
    return [k for k, v in t.items() if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n and v.dtype.kind in "fiub" and k not in skip]


def compare_bit(ta, tb, aa, ab, min_fields, skip=()):
    """Every common numeric row field and the analysis arms: bit for bit. Returns (ok, n_fields, mismatches)."""
    common = sorted(set(row_fields(ta, skip)) & set(row_fields(tb, skip)))
    bad = [k for k in common if not np.array_equal(ta[k], tb[k], equal_nan=True)]
    if json.dumps(aa, sort_keys=True) != json.dumps(ab, sort_keys=True):
        bad.append("analysis.arms")
    return (not bad and len(common) >= min_fields), len(common), bad


def bind_rate(t):
    return float(np.mean(np.asarray(t["lf_bind"], bool)))


def mutate(name, *pairs):
    src = open(LF6_TOOL, encoding="utf-8", newline="").read()
    for old, new in pairs:
        assert src.count(old) == 1, (name, old[:60], src.count(old))
        src = src.replace(old, new)
    os.makedirs(f"{MUT_TREE}/stack/scripts", exist_ok=True)
    p = f"{MUT_TREE}/stack/scripts/v8f_t1_eval_ri.py"
    open(p, "w", encoding="utf-8", newline="\n").write(src)
    return p


M1 = ("self.lift_cap = None if lift_cap is None else float(lift_cap)", "self.lift_cap = (float(lift_cap) or None) if lift_cap is not None else None")                 # a falsy zero
M2 = ("s_f = np.minimum(np.where(use_b[:, None], s_b, s_ramp), s_k + delta * op.T_KNOTS[None, :])", "s_f = np.minimum(np.where(use_b[:, None], s_b, s_ramp), s_k + min(delta, 1.0) * op.T_KNOTS[None, :])")   # a bound stuck at 1 m/s
M3 = ("cap_o = v_o + float(delta)", "cap_o = v_o + 1e9")                                                                                                                              # the DELTA term deleted


def independent_sp(t):
    """s'_k recomputed from a COMPOSITE smoke's own dump: plan_s (the plan F read), v0, lf_vf (float32) -> fine-grid integral of min(v_f, v0' + tau, v_O + DELTA), s' = max(s, s_f). Rows without a floor (lf_vf NaN) keep s."""
    sub = 400
    plan_s = np.asarray(t["plan_s"], np.float64)
    full = np.concatenate([np.zeros_like(plan_s[:, :1]), plan_s], axis=1)
    seg = np.linalg.norm(np.diff(full, axis=1), axis=2)
    S, vO = np.cumsum(seg, axis=1), seg / DT
    v0 = np.maximum(np.nan_to_num(np.asarray(t["v0"], np.float64), nan=0.0), 0.0)
    vf = np.asarray(t["lf_vf"], np.float64)
    h = DT / sub
    tau = (np.arange(30 * sub) + 0.5) * h
    k = np.arange(30 * sub) // sub
    out = S.copy()
    for i in np.nonzero(np.isfinite(vf))[0]:
        f = np.minimum(np.minimum(vf[i], v0[i] + tau), vO[i][k] + float(t["floor_lift_cap"]))
        out[i] = np.maximum(S[i], np.cumsum(f * h)[sub - 1::sub])
    return out, S


REAL_TREE = "F:/TanitAD-scratch/motion_c/lf6_read/tree"


def load_tool(path, name, tree):
    import importlib.util
    for p_ in (f"{tree}/stack", f"{tree}/taniteval"):
        if p_ not in sys.path:
            sys.path.insert(0, p_)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def load_op():
    import importlib.util
    spec = importlib.util.spec_from_file_location("level_floor_ka", f"{REAL_TREE}/stack/scripts/level_floor.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def same_input_rates(t, T1, OP, Tb):
    """The bind rate of the three operators on the SAME ticks (the inputs F read in this smoke's own rollout: plan after S rebuilt from the dumped plan_raw and P by the tool's own smooth_plan [checked bit for bit against the dumped plan_s where the smoke
    has one], v0, and the dumped v_hat, presence, gap): DELTA 0 and DELTA 0.5 by ``Tb.apply_floor_batch_bounded`` (Tb = the tool under test, or the mutant), DELTA = 1e9 by level_floor.apply_floor_batch (A5's F, an independent implementation)."""
    raw, P = np.asarray(t["plan_raw"], np.float64).astype(np.float32), np.asarray(t["P"], np.float64).astype(np.float32)
    plan_s = T1.smooth_plan(raw, P, "sg72")
    recon_ok = bool(np.array_equal(plan_s.astype(np.float64), np.asarray(t["plan_s"], np.float64))) if "plan_s" in t else None
    v0, vh, pi, gp = (np.asarray(t[k], np.float64) for k in ("v0", "lf_vhat", "lf_pi", "lf_gap"))
    gap_hat = gp / OP.GAP_SCALE_M
    b0 = Tb.apply_floor_batch_bounded(OP, plan_s, v0, vh, pi, gap_hat, 1.0, 0.0)[1]["bind"]
    bh = Tb.apply_floor_batch_bounded(OP, plan_s, v0, vh, pi, gap_hat, 1.0, 0.5)[1]["bind"]
    bi = OP.apply_floor_batch(plan_s, v0, vh, pi, gap_hat, 1.0)[1]["bind"]
    return {"n_ticks": int(len(bh)), "rate_delta0": float(b0.mean()), "rate_delta0.5": float(bh.mean()), "rate_delta1e9": float(bi.mean()), "n_bound_delta0.5_not_bound_by_delta1e9": int((bh & ~bi).sum()), "n_bound_delta1e9_not_bound_by_delta0.5": int((bi & ~bh).sum()),
            "delta0.5_recomputed_equals_dumped_bind_share": float(np.mean(bh == np.asarray(t["lf_bind"], bool))) if "lf_dterm" in t else None, "f_input_reconstruction_ok": True if recon_ok is None else recon_ok}



def main():
    only = set((sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else "K1,K2,K3").split(","))
    assert sha(LF5_TOOL) == LF5_SHA, "the LF5 reference tool differs from 88b275a1"
    os.makedirs(SW, exist_ok=True)
    out = {"card": "PREREG_LEVELFLOOR_A6 fbf5de02...", "tool_sha256": sha(LF6_TOOL), "lf5_tool_sha256": sha(LF5_TOOL), "n_windows": 64, "cases": {}}
    R = {}                                                                          # smoke name -> (dir, ticks, arms)

    def smoke(tag, tool, source=None, cap=None, composite=False):
        od = run_smoke(tag, tool, source, cap, composite)
        R[tag] = (od, ticks(od, tag), arms(od, tag))
        return R[tag]

    if "K1" in only:
        print("KA-1  delta 0 == floor off", flush=True)
        smoke("OFF_bare", LF6_TOOL)
        smoke("OFF_comp", LF6_TOOL, composite=True)
        cases = {}
        for tag, ref, src, comp in (("D0_heads_bare", "OFF_bare", "heads", False), ("D0_heads_comp", "OFF_comp", "heads", True), ("D0_logged_bare", "OFF_bare", "logged", False)):
            smoke(tag, LF6_TOOL, src, 0, comp)
            ok, nf, bad = compare_bit(R[ref][1], R[tag][1], R[ref][2], R[tag][2], min_fields=26)
            nobind = (not np.asarray(R[tag][1]["lf_bind"], bool).any()) and (not np.asarray(R[tag][1]["lf_dterm"], bool).any())
            cases[tag] = {"PASS": bool(ok and nobind), "vs": ref, "n_common_fields": nf, "mismatches": bad, "F_never_binds": bool(nobind), "n_ticks": int(len(R[tag][1]["win"]))}
        mp = mutate("M1", M1)
        Mo = run_smoke("M1_D0_heads_bare", mp, "heads", 0, False)
        tm = ticks(Mo, "M1_D0_heads_bare")
        okm, _, badm = compare_bit(R["OFF_bare"][1], tm, R["OFF_bare"][2], arms(Mo, "M1_D0_heads_bare"), min_fields=26)
        out["cases"]["K1"] = {"PASS": bool(all(c["PASS"] for c in cases.values())), "smokes": cases, "mutation_M1_falsy_zero": {"RED": not okm, "bind_rate_under_mutation": bind_rate(tm), "mismatched_fields": badm[:12]}}
        print("  K1", out["cases"]["K1"]["PASS"], "mutation RED:", not okm, flush=True)
    if "K2" in only:
        print("KA-2  delta 1e9 == LF5 tool", flush=True)
        cases = {}
        for tag5, tag6, src, comp in (("E5_heads_bare", "I6_heads_bare", "heads", False), ("E5_heads_comp", "I6_heads_comp", "heads", True), ("E5_logged_bare", "I6_logged_bare", "logged", False)):
            smoke(tag5, LF5_TOOL, src, None, comp)
            smoke(tag6, LF6_TOOL, src, 1e9, comp)
            ok, nf, bad = compare_bit(R[tag5][1], R[tag6][1], R[tag5][2], R[tag6][2], min_fields=30, skip=LF6_ONLY)
            br = bind_rate(R[tag6][1])
            cases[tag6] = {"PASS": bool(ok and br > 0.0), "vs": tag5, "n_common_fields": nf, "mismatches": bad, "bind_rate": br, "dterm_ticks": int(np.asarray(R[tag6][1]["lf_dterm"], bool).sum())}
        mp = mutate("M2", M2)
        Mo = run_smoke("M2_I6_heads_bare", mp, "heads", 1e9, False)
        tm = ticks(Mo, "M2_I6_heads_bare")
        okm, _, badm = compare_bit(R["E5_heads_bare"][1], tm, R["E5_heads_bare"][2], arms(Mo, "M2_I6_heads_bare"), min_fields=30, skip=LF6_ONLY)
        out["cases"]["K2"] = {"PASS": bool(all(c["PASS"] for c in cases.values())), "smokes": cases, "mutation_M2_bound_stuck_at_1": {"RED": not okm, "mismatched_fields": badm[:12]}}
        print("  K2", out["cases"]["K2"]["PASS"], "mutation RED:", not okm, flush=True)
    if "K3" in only:
        print("KA-3  delta 0.5 is partial", flush=True)
        for need in ("D0_heads_bare", "D0_heads_comp", "D0_logged_bare"):
            if need not in R:
                smoke(need, LF6_TOOL, need.split("_")[1], 0, need.endswith("comp"))
        for need, src, comp in (("I6_heads_bare", "heads", False), ("I6_heads_comp", "heads", True), ("I6_logged_bare", "logged", False)):
            if need not in R:
                smoke(need, LF6_TOOL, src, 1e9, comp)
        T1, OP = load_tool(LF6_TOOL, "lf6_ka_tool", REAL_TREE), load_op()
        cases = {}
        for tag, d0, inf, src, comp in (("H5_heads_bare", "D0_heads_bare", "I6_heads_bare", "heads", False), ("H5_heads_comp", "D0_heads_comp", "I6_heads_comp", "heads", True), ("H5_logged_bare", "D0_logged_bare", "I6_logged_bare", "logged", False)):
            smoke(tag, LF6_TOOL, src, 0.5, comp)
            t = R[tag][1]
            r0, rh, ri = bind_rate(R[d0][1]), bind_rate(t), bind_rate(R[inf][1])
            s_, sp = np.asarray(t["lf_s"], np.float64), np.asarray(t["lf_sp"], np.float64)
            tk = DT * np.arange(1, 31)[None, :]
            slack = float((sp - s_ - 0.5 * tk).max())
            ge = bool((sp >= s_).all())
            bound = np.asarray(t["lf_bind"], bool)
            same = same_input_rates(t, T1, OP, T1)
            c = {"rollout_level_bind_rates": {"delta0": r0, "delta0.5": rh, "delta1e9": ri, "literal_strictly_between": bool(r0 < rh < ri),
                                            "note": "NOT gating: each rollout drives its own states (a smaller lift leaves the car slower, so F meets more crawl ticks), so rollout-level bind rates are not ordered by DELTA; the bind rate at fixed inputs is (below)"},
                 "same_input_bind_rates": same, "strictly_between": bool(same["rate_delta0"] < same["rate_delta0.5"] < same["rate_delta1e9"]),
                 "max_over_knots_of_sp_minus_s_minus_delta_t": slack, "bound_holds_1e-6": bool(slack <= 1e-6), "sp_ge_s": ge,
                 "n_bound_ticks": int(bound.sum()), "n_ticks": int(len(bound)), "dterm_share_of_bound": None if not bound.any() else float(np.asarray(t["lf_dterm"], bool)[bound].mean()),
                 "mean_lift_mps_where_bound": None if not bound.any() else float(np.asarray(t["lf_lift"], np.float64)[bound].mean()), "unbound_ticks_have_sp_equal_s": bool(np.array_equal(sp[~bound], s_[~bound]))}
            if comp:
                sp_ref, s_ref = independent_sp(t)
                c["independent_recomputation_from_plan_s"] = {"max_abs_diff_sp_m": float(np.abs(sp_ref - sp).max()), "max_abs_diff_s_m": float(np.abs(s_ref - s_).max()), "tolerance_m": 1e-3, "PASS": bool(np.abs(sp_ref - sp).max() <= 1e-3 and np.abs(s_ref - s_).max() <= 1e-9)}
            c["PASS"] = bool(c["strictly_between"] and same["delta0.5_recomputed_equals_dumped_bind_share"] >= 0.99 and same["f_input_reconstruction_ok"] and c["bound_holds_1e-6"] and ge and c["unbound_ticks_have_sp_equal_s"]
                             and c.get("independent_recomputation_from_plan_s", {"PASS": True})["PASS"] and c["n_bound_ticks"] > 0)
            cases[tag] = c
        mp = mutate("M3", M3)
        Mo = run_smoke("M3_H5_heads_bare", mp, "heads", 0.5, False)
        tm = ticks(Mo, "M3_H5_heads_bare")
        Tm = load_tool(mp, "lf6_ka_tool_m3", MUT_TREE)
        same_m = same_input_rates(tm, T1, OP, Tm)          # the inputs of the mutant's own rollout; the DELTA = 0.5 rate from the MUTATED operator
        between_m = bool(same_m["rate_delta0"] < same_m["rate_delta0.5"] < same_m["rate_delta1e9"])
        out["cases"]["K3"] = {"PASS": bool(all(c["PASS"] for c in cases.values())), "smokes": cases,
                              "mutation_M3_delta_term_deleted": {"RED": not between_m, "same_input_bind_rates_under_mutation": same_m, "strictly_between_under_mutation": between_m, "rollout_level_bind_rate_under_mutation": bind_rate(tm)}}
        print("  K3", out["cases"]["K3"]["PASS"], "mutation RED:", not between_m, flush=True)
    out["PASS"] = bool(all(c["PASS"] for c in out["cases"].values()) and all(v.get("RED") for c in out["cases"].values() for k, v in c.items() if k.startswith("mutation_")))
    name = "KA_LF6.json" if only == {"K1", "K2", "K3"} else "KA_LF6_" + "_".join(sorted(only)) + ".json"
    os.makedirs(f"{PK}/raw", exist_ok=True)
    open(f"{PK}/raw/{name}", "w", encoding="utf-8", newline="\n").write(json.dumps(out, indent=1))
    print("KA_LF6 PASS" if out["PASS"] else "KA_LF6 NOT PASS", flush=True)
    sys.exit(0 if out["PASS"] else 1)


if __name__ == "__main__":
    main()
