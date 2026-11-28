"""SLOWWM read package controls (PREREG_V8F_SLOWWM; the read is unusable if one fails). Each control has a deliberate-regression arm that must go RED. CPU, 1-2 threads, tier T0. Writes CONTROLS_READ_SLOWWM.json.
  a  Adapter identity. R35H s0 through THIS read path (``lp_slowwm.prepare_arm_read``, the route every arm takes) against (a1) ``lp_lib.prepare_arm`` (every ctx array bit-equal), (a2) ``lp_lib.unit3`` on the same
     8 windows (bit-equal), (a3) the BANKED ladder unit ``p1_R35H_s0_c1_flat1_true.npz``, first 64 windows = batch 0 of the banked run (bit-equal; reported with its max abs difference).
     Red arms: a perturbed ctx array is flagged unequal; a different lambda is flagged unequal; an adapter model sent down the basis branch RAISES.
  b  Basis route. A planted basis arm (the R35H checkpoint with config wm_basis = slow, a planted basis file, planted token means) has z1 = proj tail of (nm - mean) @ W by the hook (1e-4); the cached
     ``nm_of`` route on planted features equals an independent float64 numpy computation. Red arm: the OTHER basis' u fails the same equality.
  c  Mismatch. Wrong sha, wrong kind, no readback, a ubasis built with another basis, an argv that declares a basis the BUILT model does not carry, and a name/kind mismatch are each REFUSED before any scoring.
     Red arm: the matching dir is accepted.
  d  Outcome logic. Planted margin tables give each of the four outcomes (and INCOMPLETE, and UNUSABLE when a control failed). Red arms: three mutants of ``decide`` (seeds, floor, sign) each change a planted outcome.
Usage: python controls_read_slowwm.py [a b c d]   (default all)   -> CONTROLS_READ_SLOWWM.json (keys merged; all_pass = a, b, c and d all pass)."""
import inspect
import json
import os
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import report_slowwm as R  # noqa: E402
import lp_slowwm as S  # noqa: E402

LR, LP, SH = S.LR, S.LP, S.SH
SB = SH.SB
OUT = os.path.join(HERE, "CONTROLS_READ_SLOWWM.json")
WORK = os.environ.get("SLOWWM_READ_WORK", "F:/TanitAD-scratch/slowwm_read_build")
R35H_DIR = f"{LR.PULLS}/ladder_R35H_s0"
LADDER_RUNS = R.LADDER_RUNS
D_IN = 1024


def _eqn(a, b):
    return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))


def _wait_ram(min_gb=8.0):
    if LR.free_gb() < min_gb:
        raise MemoryError(f"free RAM {LR.free_gb():.1f} GB < {min_gb} GB: controls wait, nothing is killed")


# ----------------------------------------------------------------------------------------------------------------------------- planted arms
def planted_basis(kind, seed):
    """A planted basis file (kind, mean, proj): {path, sha, mean, proj}. Values are random: they identify a file, they do not claim to be a slow or PCA basis."""
    g = np.random.default_rng(seed)
    mean = (0.1 * g.standard_normal(D_IN)).astype(np.float32)
    proj = (g.standard_normal((D_IN, 256)) / 32.0).astype(np.float32)
    os.makedirs(WORK, exist_ok=True)
    path = os.path.join(WORK, f"planted_basis_{kind}_{seed}.npz")
    sha = SB.save_basis(path, kind, mean, proj, np.linspace(1.0, 0.0, 256), {"planted": True, "seed": seed})
    return dict(path=path, sha=sha, mean=mean, proj=proj, kind=kind)


def planted_nm(fr, seed=5):
    """Planted token means for every clip of the frames dict, [T, 1024] each."""
    g = np.random.default_rng(seed)
    return [g.standard_normal((int(T), D_IN)).astype(np.float32) for T in fr["T"]]


def planted_ubasis_file(name, fr, nms, pb, kind=None, sha=None):
    """The hook's own file format (``build_eval_basis_frames``) with u = apply_basis(nm) of the planted basis; ``kind`` / ``sha`` may be set to lie."""
    path = os.path.join(WORK, f"planted_ubasis_{name}.npz")
    u = np.concatenate([SB.apply_basis(nm, pb["mean"], pb["proj"]) for nm in nms])
    np.savez(path, clips=np.array(fr["clips"]), T=np.array(fr["T"]), ubasis=u, kind=np.array(kind or pb["kind"]), basis_sha=np.array(sha or pb["sha"]))
    return path


def planted_run(tag, cfg_kind, argv_kind, rb_kind, rb_sha, cache_problems=None, with_readback=True):
    """A run folder: R35H s0's checkpoint with config wm_basis = ``cfg_kind`` (no state-dict key changes), its argv + ``--wm-basis argv_kind``, a selfcheck.json carrying the given readback."""
    import torch
    d = os.path.join(WORK, f"run_{tag}")
    os.makedirs(d, exist_ok=True)
    ck = torch.load(os.path.join(R35H_DIR, "ckpt_last.pt"), map_location="cpu", weights_only=False)
    ck["cfg"] = dict(ck["cfg"], wm_basis=cfg_kind)
    torch.save(ck, os.path.join(d, "ckpt_last.pt"))
    txt = open(os.path.join(R35H_DIR, "argv.txt"), encoding="utf-8").read().strip()
    open(os.path.join(d, "argv.txt"), "w", encoding="utf-8").write(txt + (f" --wm-basis {argv_kind}" if argv_kind else "") + "\n")
    sc = {"levers": {"wm_basis_readback": {"kind": rb_kind, "basis_sha256": rb_sha, "dim": 256, "cache_problems": cache_problems or []}}} if with_readback else {"levers": {}}
    json.dump(sc, open(os.path.join(d, "selfcheck.json"), "w", encoding="utf-8"))
    return d


def refused(fn):
    """(True, exception text) when ``fn`` raised one of the refusal types."""
    try:
        fn()
    except (SH.BasisMismatch, LR.LadderMismatch, ValueError) as e:
        return True, f"{type(e).__name__}: {str(e)[:200]}"
    return False, "accepted"


# ----------------------------------------------------------------------------------------------------------------------------- control a
def control_a():
    import torch
    _wait_ram()
    torch.set_num_threads(1)                      # the ladder run's own setting
    t0 = time.time()
    mine = S.prepare_arm_read("R35H_s0", R35H_DIR)
    ref = LP.prepare_arm("R35H_s0")
    fields = ("z_now", "z_fut", "Zw", "z_hist", "a_hist")
    a1 = {k: bool(torch.equal(getattr(mine, k), getattr(ref, k))) for k in fields}
    a1.update({k: _eqn(getattr(mine, k), getattr(ref, k)) for k in ("A", "v0", "hist_ok", "tune_idx", "score_idx", "donor", "std_raw")})
    sel = LP.sel_arrays(mine, "c1")
    idx = sel["all"][:64]
    r64 = LP.unit3(mine, "flat1", "true", "c1", idx, 0.0, seed_mode="prior", collect=False)
    bank = np.load(f"{LADDER_RUNS}/p1_R35H_s0_c1_flat1_true.npz")
    same_idx = _eqn(bank["idx"][:64], idx)
    p_mine, p_bank = r64["plan_raw"].astype(np.float32), bank["plan_raw"][:64]
    a3 = dict(windows=int(len(idx)), same_window_idx=same_idx, bit_equal=_eqn(p_mine, p_bank), max_abs_diff=float(np.abs(p_mine.astype(np.float64) - p_bank.astype(np.float64)).max()))
    i8 = idx[:8]
    m8, r8 = LP.unit3(mine, "flat1", "true", "c1", i8, 0.0, seed_mode="prior", collect=False), LP.unit3(ref, "flat1", "true", "c1", i8, 0.0, seed_mode="prior", collect=False)
    a2 = dict(windows=8, plan_raw_bit_equal=_eqn(m8["plan_raw"], r8["plan_raw"]), cost1_bit_equal=_eqn(m8["cost1"], r8["cost1"]))
    # red arms
    z = mine.Zw.clone()
    z[0, 0, 0] += 1e-6
    red_perturb_flagged = not bool(torch.equal(z, ref.Zw))
    m8_lam = LP.unit3(mine, "flat1", "true", "c1", i8, 0.3, seed_mode="hold", collect=False)
    red_lambda_flagged = not _eqn(m8_lam["plan_raw"], r8["plan_raw"])
    raised, why = refused(lambda: SH.clip_z1_basis(mine.wm, LR.load_eval_frames(f"{LP.FRAMES}/frames_s0.npz")))
    ok = all(a1.values()) and a2["plan_raw_bit_equal"] and a2["cost1_bit_equal"] and a3["same_window_idx"] and a3["bit_equal"] and red_perturb_flagged and red_lambda_flagged and raised
    return dict(pass_=ok, a1_ctx_arrays_equal_lp_prepare_arm=a1, a2_unit3_8_windows_vs_lp_lib=a2, a3_banked_first_batch=a3, banked_file=f"{LADDER_RUNS}/p1_R35H_s0_c1_flat1_true.npz", threads=1,
                red_arms=dict(perturbed_ctx_flagged=red_perturb_flagged, other_lambda_flagged=red_lambda_flagged, adapter_model_down_basis_branch_raises=raised, raise_text=why), wall_s=round(time.time() - t0, 1))


# ----------------------------------------------------------------------------------------------------------------------------- controls b and c (planted basis arm)
def _expected_z1(model, nms, pb):
    """Independent of the hook: float64 numpy (nm - mean) @ W, then the model's own proj tail."""
    import torch
    out = []
    for nm in nms:
        u = ((nm.astype(np.float64) - pb["mean"].astype(np.float64)) @ pb["proj"].astype(np.float64)).astype(np.float32)
        with torch.no_grad():
            out.append(model.wm._z1_post(torch.as_tensor(u)[None])[0].float())
    return out


def control_b_c():
    import torch
    _wait_ram()
    torch.set_num_threads(1)
    t0 = time.time()
    os.makedirs(WORK, exist_ok=True)
    fr = LR.load_eval_frames(f"{LP.FRAMES}/frames_s0.npz")
    B1, B2 = planted_basis("slow", 1), planted_basis("slow", 2)
    nms = planted_nm(fr)
    ub1, ub2 = planted_ubasis_file("1", fr, nms, B1), planted_ubasis_file("2", fr, nms, B2)
    run1 = planted_run("ok", "slow", "slow", "slow", B1["sha"])
    # ---- b: the positive read and its z1 identity
    ctx = S.prepare_arm_read("SLOW_s0", run1, B1["path"], ub1)
    model, *_ = LR.load_arm(run1)
    exp1 = _expected_z1(model, nms, B1)
    exp2 = _expected_z1(model, nms, B2)
    got = SH.clip_z1_basis(model.wm, SH.attach_basis_frames(dict(fr), SH.SB.load_basis(B1["path"]), ub1))
    d_ok = max(float((got[c] - e).abs().max()) for c, e in zip(fr["clips"], exp1))
    d_swap = max(float((got[c] - e).abs().max()) for c, e in zip(fr["clips"], exp2))
    g0 = int(ctx.t[0]) - 15
    win0 = float((ctx.Zw[0] - exp1[fr["clips"].index(str(ctx.eid[0]))][g0:g0 + 16]).abs().max())
    # ``ubasis_of_feats`` on planted features: the cache route (nm_of) equals a float64 recomputation
    g = np.random.default_rng(9)
    feats = g.standard_normal((6, 40, D_IN)).astype(np.float32)
    nm = SB.nm_of(feats)
    u_hook = SH.ubasis_of_feats(SB.load_basis(B1["path"]), feats)
    u_ref = ((nm.astype(np.float64) - B1["mean"].astype(np.float64)) @ B1["proj"].astype(np.float64)).astype(np.float32)
    d_feats = float(np.abs(u_hook - u_ref).max())
    tag_ok = ctx.basis_kind == "slow" and ctx.basis_sha == B1["sha"]
    unit = LP.unit3(ctx, "flat1", "true", "c1", LP.sel_arrays(ctx, "c1")["all"][:8], 0.0, seed_mode="prior", collect=False)
    plan_finite = bool(np.isfinite(unit["plan_raw"]).all()) and unit["plan_raw"].shape[0] == 8
    b = dict(pass_=bool(d_ok <= 1e-4 and win0 <= 1e-4 and d_feats <= 1e-4 and tag_ok and plan_finite and d_swap > 1e-3), z1_vs_independent_float64_max_abs=d_ok, window0_ctx_vs_independent_max_abs=win0,
             nm_of_route_vs_float64_max_abs=d_feats, tolerance=1e-4, basis_tag_on_ctx=tag_ok, planted_basis_plan_8_windows_finite=plan_finite,
             red_arm_other_basis_z1_max_abs=d_swap, red_arm_other_basis_flagged=bool(d_swap > 1e-4), wall_s=round(time.time() - t0, 1))
    # ---- c: each refusal, and the accepted control
    runs = dict(
        wrong_sha=lambda: S.prepare_arm_read("SLOW_s0", run1, B2["path"], ub2),
        wrong_kind_readback=lambda: S.prepare_arm_read("SLOW_s0", planted_run("kind", "slow", "slow", "pca", B1["sha"]), B1["path"], ub1),
        no_readback=lambda: S.prepare_arm_read("SLOW_s0", planted_run("norb", "slow", "slow", "slow", B1["sha"], with_readback=False), B1["path"], ub1),
        cache_problems=lambda: S.prepare_arm_read("SLOW_s0", planted_run("cp", "slow", "slow", "slow", B1["sha"], cache_problems=["x"]), B1["path"], ub1),
        ubasis_of_other_basis=lambda: S.prepare_arm_read("SLOW_s0", run1, B1["path"], ub2),
        ubasis_kind_mismatch=lambda: SH.attach_basis_frames(dict(fr), SB.load_basis(B1["path"]), planted_ubasis_file("kind", fr, nms, B2, kind="pca")),
        argv_declares_basis_model_has_none=lambda: S.prepare_arm_read("SLOW_s0", planted_run("nob", "adapter", "slow", "slow", B1["sha"]), B1["path"], ub1),
        model_has_basis_argv_says_adapter=lambda: S.prepare_arm_read("R35H_s0", planted_run("rev", "slow", None, "slow", B1["sha"]), None, None),
        name_kind_mismatch=lambda: S.prepare_arm_read("PCA-B_s0", run1, B1["path"], ub1),
    )
    res = {k: dict(zip(("refused", "text"), refused(fn))) for k, fn in runs.items()}
    accepted_control = ctx is not None and tag_ok
    c = dict(pass_=bool(all(v["refused"] for v in res.values()) and accepted_control), refusals=res, matching_dir_accepted=accepted_control, red_arm="the matching planted dir is accepted (b); every row above must read refused: True",
             wall_s=round(time.time() - t0, 1))
    return b, c


# ----------------------------------------------------------------------------------------------------------------------------- control d
def _tab(m, pl, fl):
    """m / pl given per group as (s0, s1) per family dict; floor per family."""
    return ({g: {s: dict(v[s]) for s in (0, 1)} for g, v in m.items()}, {g: {s: dict(v[s]) for s in (0, 1)} for g, v in pl.items()}, dict(fl))


def planted_tables():
    both = lambda lon, lat=0.0: ({"LONGITUDINAL": lon, "LATERAL": lat}, {"LONGITUDINAL": lon, "LATERAL": lat})
    yes = lambda: ({"LONGITUDINAL": True, "LATERAL": False}, {"LONGITUDINAL": True, "LATERAL": False})
    no = lambda: ({"LONGITUDINAL": False, "LATERAL": False}, {"LONGITUDINAL": False, "LATERAL": False})
    fl = {"LONGITUDINAL": 0.05, "LATERAL": 0.05}
    T = {}
    T["SLOW-PLANS"] = _tab(dict(zip(("R35H", "SLOW", "PCA-B"), (both(0.0), both(0.5), both(0.1)))), {"R35H": no(), "SLOW": yes(), "PCA-B": no()}, fl)
    T["BASIS-ONLY"] = _tab(dict(zip(("R35H", "SLOW", "PCA-B"), (both(0.0), both(0.30), both(0.34)))), {"R35H": no(), "SLOW": yes(), "PCA-B": yes()}, fl)
    T["NO-EFFECT"] = _tab(dict(zip(("R35H", "SLOW", "PCA-B"), (both(0.10), both(0.12), both(0.08)))), {"R35H": no(), "SLOW": no(), "PCA-B": no()}, fl)
    T["SLOW-COSTS"] = _tab(dict(zip(("R35H", "SLOW", "PCA-B"), (both(0.10), both(0.0), both(0.40)))), {"R35H": no(), "SLOW": no(), "PCA-B": yes()}, fl)
    one_seed = _tab(dict(zip(("R35H", "SLOW", "PCA-B"), (both(0.0), both(0.5), both(0.0)))), {"R35H": no(), "SLOW": yes(), "PCA-B": no()}, fl)
    for f in ("LONGITUDINAL", "LATERAL"):
        one_seed[0]["SLOW"][1][f] = 0.0                    # SLOW beats R35H and PCA-B at seed 0 only
    T["one-seed-only"] = one_seed
    notplan = _tab(dict(zip(("R35H", "SLOW", "PCA-B"), (both(0.0), both(0.5), both(0.1)))), {"R35H": no(), "SLOW": no(), "PCA-B": no()}, fl)
    T["not-plannable"] = notplan
    return T


# not-plannable: SLOW beats R35H and PCA-B beyond the floor but is not PLANNABLE (SLOW-PLANS needs it); PCA-B is not within the floor of SLOW (no BASIS-ONLY); SLOW beats R35H (no NO-EFFECT): no outcome holds
EXPECT = {"SLOW-PLANS": ["SLOW-PLANS"], "BASIS-ONLY": ["BASIS-ONLY"], "NO-EFFECT": ["NO-EFFECT"], "SLOW-COSTS": ["SLOW-COSTS"], "one-seed-only": ["NO-EFFECT"], "not-plannable": []}


def mutants():
    src = inspect.getsource(R.decide)
    out = {}
    for tag, (a, b) in {"seeds_0_only": ("for s in (0, 1)", "for s in (0,)"), "floor_ignored": ("fl = floor[f]", "fl = 0.0"), "sign_flipped": ("m[\"SLOW\"][s][f] - m[\"R35H\"][s][f] > fl", "m[\"R35H\"][s][f] - m[\"SLOW\"][s][f] > fl")}.items():
        assert a in src, f"mutation anchor missing: {a}"
        ns = {"FAMS": R.FAMS, "GROUPS": R.GROUPS}
        exec(src.replace(a, b), ns)
        out[tag] = ns["decide"]
    return out


def control_d():
    T = planted_tables()
    got = {k: R.decide(*v) for k, v in T.items()}
    rows = {k: dict(expected=EXPECT[k], got=got[k]["outcomes"], ok=got[k]["outcomes"] == EXPECT[k]) for k in T}
    inc = _tab({g: {0: {"LONGITUDINAL": None, "LATERAL": 0.0}, 1: {"LONGITUDINAL": 0.0, "LATERAL": 0.0}} for g in R.GROUPS}, {g: {0: {}, 1: {}} for g in R.GROUPS}, {"LONGITUDINAL": 0.05, "LATERAL": 0.05})
    incomplete_ok = R.decide(*inc)["status"] == "INCOMPLETE" and R.decide(*inc)["outcomes"] == []
    agree = R.combine_cells({"a": got["SLOW-PLANS"], "b": got["SLOW-PLANS"]})
    disagree = R.combine_cells({"a": got["SLOW-PLANS"], "b": got["NO-EFFECT"]})
    unusable = R.final_outcome({"usable": False, "status": "FAIL"}, agree)
    comb_ok = agree["status"] == "CELLS-AGREE" and agree["agree"] == ["SLOW-PLANS"] and disagree["status"] == "CELLS-DISAGREE" and unusable.startswith("UNUSABLE") and R.final_outcome({"usable": True}, agree) == ["SLOW-PLANS"]
    red = {}
    for tag, fn in mutants().items():
        diff = [k for k, v in T.items() if fn(*v)["outcomes"] != got[k]["outcomes"]]
        red[tag] = dict(changes_planted_outcomes=diff, red=bool(diff))
    ok = all(r["ok"] for r in rows.values()) and incomplete_ok and comb_ok and all(r["red"] for r in red.values())
    return dict(pass_=ok, planted=rows, incomplete_when_a_margin_is_missing=incomplete_ok, combine_and_unusable=dict(ok=comb_ok, unusable_text=unusable, disagree=disagree["status"]), red_arms=red,
                note="SLOW-COSTS planted with PCA-B beating R35H so that it stands alone; with PCA-B not beating R35H both NO-EFFECT and SLOW-COSTS hold (the card gives SLOW-COSTS NO-EFFECT's next step)")


def main(argv):
    want = [a for a in argv if a in "abcd"] or ["a", "b", "c", "d"]
    res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    res.setdefault("tier", "T0")
    # An "error" belongs to the run that raised it. Without this pop, an error merged from an EARLIER refused run kept all_pass False through
    # four fresh runs that each read a, b, c and d pass True (MEASURED 2026-11-13 01:18-02:21). A control that raises in THIS run sets it again below.
    stale = res.pop("error", None)
    if stale is not None:
        res["error_previous_run"] = stale
    try:
        if "a" in want:
            res["a_adapter_identity"] = control_a()
        if "b" in want or "c" in want:
            b, c = control_b_c()
            res["b_basis_route"], res["c_mismatch"] = b, c
        if "d" in want:
            res["d_outcome_logic"] = control_d()
    except BaseException:
        traceback.print_exc()
        res["error"] = traceback.format_exc()[-1500:]
    for v in res.values():
        if isinstance(v, dict) and "pass_" in v:
            v["pass"] = v.pop("pass_")
    keys = ("a_adapter_identity", "b_basis_route", "c_mismatch", "d_outcome_logic")
    res["all_pass"] = all(res.get(k, {}).get("pass") is True for k in keys) and "error" not in res
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print({k: res[k].get("pass") for k in keys if k in res}, "all_pass", res["all_pass"])
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
