"""LON-PLAN TRAIN-selection runner (AMEND_HWMPLAN_3 section 3). BUILT AND UNIT-TESTED ONLY: it has not been run on a real window (the MM runs it after the build review).
It reads TRAIN windows of R35's own TRAIN latents only. No eval window is read for selection (the eval clip ids are read once, for the disjointness assertion, and no eval window loader is called).
Pieces:
  build_selection_windows   the frame table, the valid NOW positions, 4 per clip at unique(round(linspace(0, n_valid-1, 4)))
  criterion                 C_k (0-2 s acc_mae, hwm_eval.band_metrics' expression), D_k / D_yaw (hold_distance in TRAIN prior std over knots 1-10), DIR-ACC
  choose / run_selection    base grid, sign check, extension up / down, admissibility D >= 0.10, tie rule (round C to 4 decimals, smaller lambda), LAMBDA-COLLAPSE
  the record                LAMBDA_SELECTION_R35_s{k}.json: grid as run, C, D, D_yaw, rule trace, lambda*, every argv, sha256 of every input and of every code-of-record file
usage (MM): python lonplan_select.py --chassis s0 --root <NEW ROOT DIR> [--workers 1]   (defaults: the R35 paths of AMEND_HWMPLAN_3; OMP_NUM_THREADS=2)
Tier T0 on windows the world model trained on: a selection record, never a result."""
import argparse, json, math, os, sys, time, types
import lonplan_common as C
import numpy as np
import torch
import run_hwm as R
import hwm_plan as H
import hwm_eval as E
import hwm_data as D
import plantest2_lib as Q
import lonplan_run as W

GRID = (0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
SIGN_LAM = -1.0
EXT_UP = (300.0, 1000.0, 3000.0)
EXT_DOWN = (0.1, 0.03, 0.01)
D_MIN = 0.10                                   # the card's HOLD-COLLAPSE threshold, in TRAIN prior std
PER_CLIP = 4
N_KNOTS = 10                                   # knots 1-10 = the 0-2 s band (n_op)
NOW_FRAMES = 31                                # frames g .. g+30 (T3 2 puts the goal at frame 30)
DEFAULTS = dict(
    s0=dict(tag="R35_s0", ckpt="C:/Users/Admin/t2v8f_pulls/ladder_R35_s0/ckpt_last.pt", latents="F:/TanitAD-scratch/hwm_plan/latents_train_R35_s0.pt", latents_sha="76a2d4f5482bd98670bab40ebbf4b3c85fde8ebf222596947fa16f0c7e82754c"),
    s1=dict(tag="R35_s1", ckpt="C:/Users/Admin/t2v8f_pulls/ladder_R35_s1/ckpt_last.pt", latents="F:/TanitAD-scratch/hwm_plan/latents_train_R35_s1.pt", latents_sha="692eb0021a4b39295a52589ab2b7f2de0a45615345bca7a7acea62101d32708a"))
EVAL_LATENTS = os.path.join(C.CODE_ROOT, "wmact", "latents.pt")


# ----------------------------------------------------------------------------------------------- selection windows (section 3.1)
def frame_table(L, tag):
    """(eid, frame) -> (z, a, ok) from the overlapping 16-frame windows. A window j holds frames t[j] .. t[j]+15. Returns the dict and the max |dz|, |da| over repeated entries (the spec's 0.0)."""
    Z, A, OK, T, EID = L["Z1"][tag].float(), L["A_raw"].float(), L["ok"], np.asarray(L["t"]).astype(int), np.array(L["eid"])
    tab, dz, da = {}, 0.0, 0.0
    for j in range(Z.shape[0]):
        for i in range(Z.shape[1]):
            k = (str(EID[j]), int(T[j]) + i)
            if k in tab:
                z0, a0, o0 = tab[k]
                dz = max(dz, float((z0 - Z[j, i]).abs().max())); da = max(da, float((a0 - A[j, i]).abs().max()))
                assert bool(o0) == bool(OK[j, i]), ("ok flag differs across windows", k)
            else:
                tab[k] = (Z[j, i], A[j, i], bool(OK[j, i]))
    return tab, dz, da


def valid_positions(tab, eids, n_frames=NOW_FRAMES):
    """Per clip, the NOW frames g with frames g .. g+n_frames-1 all present and ok."""
    out = {}
    for e in eids:
        fr = sorted(f for (ee, f) in tab if ee == e)
        have = {f for f in fr if tab[(e, f)][2]}
        out[e] = [g for g in fr if all((g + i) in have for i in range(n_frames))]
    return out


def pick_positions(valid, per_clip=PER_CLIP):
    """unique(round(linspace(0, n_valid - 1, per_clip))) of the valid positions of a clip."""
    n = len(valid)
    if n == 0:
        return []
    idx = np.unique(np.round(np.linspace(0, n - 1, per_clip)).astype(int))
    return [valid[i] for i in idx]


def build_selection_windows(latents_pt, tag, per_clip=PER_CLIP, n_frames=NOW_FRAMES):
    """The TRAIN selection set as a windows.npz-schema dict (z (n, 31, D), A (n, 31, 2), eid, now, v0 = 0, donor, meta). Reads the TRAIN latents file only."""
    L = torch.load(latents_pt, map_location="cpu", weights_only=False)
    tab, dz, da = frame_table(L, tag)
    eids = sorted({e for (e, _) in tab})
    valid = valid_positions(tab, eids, n_frames)
    rows = [(e, g) for e in eids for g in pick_positions(valid[e], per_clip)]
    z = np.stack([np.stack([tab[(e, g + i)][0].numpy() for i in range(n_frames)]) for e, g in rows]).astype(np.float32)
    A = np.stack([np.stack([tab[(e, g + i)][1].numpy() for i in range(n_frames)]) for e, g in rows]).astype(np.float64)
    eid = np.array([e for e, _ in rows], dtype="U36"); now = np.array([g for _, g in rows], dtype=np.int64)
    meta = dict(tag=tag, F=n_frames, no_goal_frame=False, n=len(rows), n_clips=len(eids), per_clip=per_clip, n_valid_positions=int(sum(len(v) for v in valid.values())), min_valid_per_clip=int(min(len(v) for v in valid.values())),
                n_unique_frames=len(tab), max_abs_dz_repeated=dz, max_abs_da_repeated=da, kind="LONPLAN_TRAIN_SELECTION", source=str(latents_pt))
    return dict(z=z, A=A, eid=eid, now=now, v0=np.zeros(len(rows)), donor=np.arange(len(rows)), meta=meta, table=tab, eids=eids)


def save_windows(w, path):
    np.savez(path, z=w["z"], A=w["A"], eid=w["eid"], now=w["now"], v0=w["v0"], donor=w["donor"], meta=json.dumps(w["meta"]))
    return C.sha256_file(path)


def windows_equal_source(w, L, tag):
    """C-TRAIN-ISO: each selected window equals its source frames frame for frame: max |dz|, |dA| against the first source window that holds each frame (0.0)."""
    Z, A, T, EID = L["Z1"][tag].float().numpy(), L["A_raw"].float().numpy(), np.asarray(L["t"]).astype(int), np.array(L["eid"])
    src = {}
    for j in range(Z.shape[0]):
        for i in range(Z.shape[1]):
            src.setdefault((str(EID[j]), int(T[j]) + i), (j, i))
    mz = ma = 0.0
    for r in range(len(w["eid"])):
        for i in range(w["z"].shape[1]):
            j, k = src[(str(w["eid"][r]), int(w["now"][r]) + i)]
            mz = max(mz, float(np.abs(w["z"][r, i] - Z[j, k]).max())); ma = max(ma, float(np.abs(w["A"][r, i] - A[j, k].astype(np.float64)).max()))
    return mz, ma


def assert_disjoint(train_eids, eval_eids):
    """C-TRAIN-ISO: 0 TRAIN clips among the eval clips."""
    both = sorted(set(map(str, train_eids)) & set(map(str, eval_eids)))
    assert not both, f"{len(both)} TRAIN clips are eval clips, e.g. {both[:3]}"
    return len(both)


# ----------------------------------------------------------------------------------------------- the criterion (section 3.3)
def criterion_C(plan_raw, A):
    """C: the mean over windows of the 0-2 s acc_mae, written as hwm_eval.band_metrics writes it (hwm_eval.py:94): mean over knots 1-10 of |plan acc - logged acc|. test_lonplan asserts equality with band_metrics on 30-knot data."""
    return float(np.mean(np.abs(np.asarray(plan_raw)[:, :N_KNOTS, 0] - np.asarray(A)[:, :N_KNOTS, 0]).mean(1)))


def distance_D(plan_raw, A, std_raw):
    """(D_acc, D_yaw): mean |plan - HOLD| over windows and knots 1-10, in TRAIN prior std (plantest2_lib.hold_distance, restricted to knots 1-10)."""
    hold = H.hold_plan(A[:, 0], plan_raw.shape[1])
    d = Q.hold_distance(plan_raw[:, :N_KNOTS], hold[:, :N_KNOTS], std_raw)
    return float(d[0]), float(d[1])


def dir_acc(plan_raw, A, eid, n_boot=2000, seed=0):
    """DIR-ACC (no bar): Pearson r of x = mean over knots 1-10 of (plan acc - a0) and y = the same for the logged acc, over windows; CI from the episode-cluster bootstrap. n/a when x or y is constant (HOLD)."""
    a0 = A[:, 0, 0]
    x = plan_raw[:, :N_KNOTS, 0].mean(1) - a0
    y = A[:, :N_KNOTS, 0].mean(1) - a0

    def r_of(ix):
        xs, ys = x[ix], y[ix]
        if xs.std() < 1e-12 or ys.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(xs, ys)[0, 1])

    r = r_of(np.arange(len(x)))
    if not np.isfinite(r):
        return dict(r=float("nan"), lo=float("nan"), hi=float("nan"), n_windows=int(len(x)), reason="n/a: x or y is constant (a plan equal to HOLD has no departure to correlate)")
    eids = np.asarray(eid); ue = sorted(set(eids)); by = {e: np.nonzero(eids == e)[0] for e in ue}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, len(ue), len(ue))
        ix = np.concatenate([by[ue[p]] for p in pick])
        v = r_of(ix)
        if np.isfinite(v):
            bs.append(v)
    lo, hi = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else (float("nan"), float("nan"))
    return dict(r=r, lo=lo, hi=hi, n_windows=int(len(x)), n_episodes=int(len(ue)), estimator="episode_cluster_bootstrap", B=int(n_boot), seed=int(seed))


# ----------------------------------------------------------------------------------------------- the procedure (section 3.2 and 3.3), pure logic
def c4(x):
    return round(float(x), 4)


def admissible(lam, rec):
    return lam > 0 and rec["D"] >= D_MIN


def choose(runs):
    """lambda* = the admissible lambda > 0 with the lowest C (rounded to 4 decimals); a tie goes to the smaller lambda. None if none is admissible."""
    cand = [l for l, r in runs.items() if admissible(l, r)]
    return min(cand, key=lambda l: (c4(runs[l]["C"]), l)) if cand else None


def select(evaluate):
    """The whole procedure. evaluate(lam) -> dict(C, D, D_yaw, ...). Fixed here; never changed after a value is seen. Returns dict(runs, lam_star, label, trace, sign_check)."""
    runs, trace = {}, []

    def run(lam, why):
        runs[lam] = evaluate(lam)
        trace.append(dict(step=len(trace) + 1, lam=lam, why=why, C=c4(runs[lam]["C"]), D=float(runs[lam]["D"]), admissible=bool(admissible(lam, runs[lam]))))

    for lam in GRID:
        run(lam, "base grid")
    run(SIGN_LAM, "sign check (never a candidate)")
    cand_runs = lambda: {l: r for l, r in runs.items() if l > 0}
    star = choose(cand_runs())
    if star is None:
        for lam in EXT_DOWN:
            run(lam, "extension down: no lambda > 0 admissible")
            star = choose(cand_runs())
            if star is not None:
                trace.append(dict(note="first admissible value found, extension down stops", lam=lam)); break
    else:
        def running_min():
            return min(c4(r["C"]) for l, r in cand_runs().items() if admissible(l, r))
        if star == max(cand_runs()):                                           # the chosen lambda is the largest value run so far, and (being admissible) not collapsed
            for lam in EXT_UP:
                if choose(cand_runs()) != max(cand_runs()):
                    break
                before = running_min()
                run(lam, "extension up: the chosen lambda is the largest run")
                if c4(runs[lam]["C"]) > before or runs[lam]["D"] < D_MIN:
                    trace.append(dict(note="extension up stops: C above the running minimum or D < 0.10", lam=lam)); break
            star = choose(cand_runs())
        elif star == min(cand_runs()):                                         # the chosen lambda is the smallest positive value run (0.3 on the base grid)
            for lam in EXT_DOWN:
                if choose(cand_runs()) != min(cand_runs()):
                    break
                before = running_min()
                run(lam, "extension down: the chosen lambda is the smallest run")
                if c4(runs[lam]["C"]) > before:
                    trace.append(dict(note="extension down stops: C above the running minimum", lam=lam)); break
            star = choose(cand_runs())
    label = "LAMBDA-COLLAPSE" if star is None else "SELECTED"
    sc = None
    if 0.0 in runs and SIGN_LAM in runs:
        sc = dict(C_neg1_minus_C0=float(runs[SIGN_LAM]["C"] - runs[0.0]["C"]), ok=bool(c4(runs[SIGN_LAM]["C"]) > c4(runs[0.0]["C"])), rule="C(-1) > C(0) on TRAIN before any SCORE plan at lambda != 0 (C-REG)")
    return dict(runs=runs, lam_star=star, label=label, trace=trace, sign_check=sc)


# ----------------------------------------------------------------------------------------------- real evaluation of one lambda
def lam_dir(root, lam):
    return os.path.join(root, "lam_" + ("m" if lam < 0 else "") + f"{abs(lam):g}")


def plan_argv(cfg, out, lam, workers):
    return ["--tag", cfg["tag"], "--ckpt", cfg["ckpt"], "--train-latents", cfg["latents"], "--train-tag", cfg["tag"], "--seed-mode", "hold", f"--lam={lam!r}",
            "--arms", "hwm_oracle", "--seeds", "0", "--workers", str(workers), "--out", out]


def make_evaluator(cfg, root, windows_path, windows_sha, workers=1, n_boot=2000, enforce_threads=True):
    """evaluate(lam): plan the selection windows through the run wrapper (run_hwm.cmd_plan, code of record) and read C, D, D_yaw, DIR-ACC."""
    npz = np.load(windows_path, allow_pickle=False)
    A = npz["A"]; eid = npz["eid"]
    a0 = W.parse(plan_argv(cfg, os.path.join(root, "_prep"), 1.0, workers))
    w_ns = types.SimpleNamespace(z=torch.from_numpy(npz["z"]), A=A, eid=eid, now=npz["now"], v0=npz["v0"], donor=npz["donor"], n=len(eid))
    m, wm, st, geom = R.prepare(a0, w_ns)
    std_raw = Q.std_raw_of(st, float(wm.ch1_scale))
    pl = R.make_planner(a0, wm, st, geom, w_ns)
    refs = E.run_refs(pl, w_ns)
    ref_C = {k: criterion_C(v, A) for k, v in refs.items()}
    cache = {}

    def evaluate(lam):
        out = lam_dir(root, lam)
        argv = plan_argv(cfg, out, lam, workers)
        sol = W.run_lambda(argv, lam, "hold", windows_path, windows_sha, enforce_threads=enforce_threads)
        with np.load(os.path.join(out, "plans_hwm_oracle_s0.npz"), allow_pickle=False) as z:
            plan = z["plan_raw"].astype(np.float64)
        d_acc, d_yaw = distance_D(plan, A, std_raw)
        rec = dict(C=criterion_C(plan, A), D=d_acc, D_yaw=d_yaw, dir_acc=dir_acc(plan, A, eid, n_boot), out=out, argv=argv, solver_json=os.path.join(out, W.SOLVER_JSON), plan_sha256=sol["plans"]["plans_hwm_oracle_s0.npz"]["sha256"])
        cache[lam] = rec
        return rec
    evaluate.refs = dict(C_hold=ref_C["hold"], C_prior=ref_C["prior"], std_raw=[float(x) for x in std_raw])
    return evaluate


def selection_record(cfg, res, windows_meta, windows_sha, refs, eval_overlap):
    runs = res["runs"]
    return dict(kind="LAMBDA_SELECTION", amendment="AMEND_HWMPLAN_3 (frozen, sha256 7bbdc9fe..., commit 4dc4876c)", tier="T0 diagnostic on TRAIN windows: a selection record, never a result", evidence="MEASURED when run; this file is written by the runner",
                chassis=cfg["tag"], lam_star=res["lam_star"], label=res["label"], grid_as_run=sorted(runs), sign_check=res["sign_check"], trace=res["trace"],
                runs={repr(l): dict(C=r["C"], C_rounded=c4(r["C"]), D=r["D"], D_yaw=r["D_yaw"], admissible=bool(admissible(l, r)), dir_acc=r["dir_acc"], argv=r["argv"], plan_sha256=r["plan_sha256"]) for l, r in sorted(runs.items())},
                beside_no_bar=dict(C_at_0=runs[0.0]["C"] if 0.0 in runs else None, C_HOLD=refs["C_hold"], C_PRIOR=refs["C_prior"], TRAIN_CEILING_flag=(None if res["lam_star"] is None else bool(runs[res["lam_star"]]["C"] < refs["C_hold"]))),
                prior_std_raw=refs["std_raw"], windows=dict(sha256=windows_sha, meta=windows_meta), train_clips_in_eval=eval_overlap,
                inputs_sha256=dict(ckpt=C.sha256_file(cfg["ckpt"]), train_latents=C.sha256_file(cfg["latents"])), code_of_record_sha256=C.code_of_record_sha256(), rule=dict(grid=list(GRID), sign=SIGN_LAM, ext_up=list(EXT_UP), ext_down=list(EXT_DOWN), D_min=D_MIN, tie="C rounded to 4 decimals, then the smaller lambda"),
                written=time.strftime("%Y-%m-%dT%H:%M:%S"))


def run_selection(chassis, root, workers=1, n_boot=2000, cfg=None, eval_latents=EVAL_LATENTS, enforce_threads=True, n_frames=NOW_FRAMES):
    cfg = cfg or DEFAULTS[chassis]
    if enforce_threads and os.environ.get("OMP_NUM_THREADS") != str(C.THREADS):
        raise W.RefusedRun(f"OMP_NUM_THREADS must be {C.THREADS}")
    if os.path.exists(root) and os.listdir(root):
        raise W.RefusedRun(f"selection root {root} is not empty")
    os.makedirs(root, exist_ok=True)
    if cfg.get("latents_sha") and C.sha256_file(cfg["latents"]) != cfg["latents_sha"]:
        raise W.RefusedRun("TRAIN latents sha256 differs from the phase-1 record")
    w = build_selection_windows(cfg["latents"], cfg["tag"], n_frames=n_frames)
    ev = torch.load(eval_latents, map_location="cpu", weights_only=False)["eid"]                       # the eval clip ids only: for the disjointness assertion, no eval window is read
    overlap = assert_disjoint(w["eids"], ev)
    wp = os.path.join(root, "selection_windows.npz")
    wsha = save_windows(w, wp)
    evaluate = make_evaluator(cfg, root, wp, wsha, workers, n_boot, enforce_threads)
    res = select(evaluate)
    rec = selection_record(cfg, res, w["meta"], wsha, evaluate.refs, overlap)
    k = chassis[-1]
    C.dump_json(rec, os.path.join(root, f"LAMBDA_SELECTION_R35_s{k}.json"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chassis", required=True, choices=sorted(DEFAULTS)); ap.add_argument("--root", required=True); ap.add_argument("--workers", type=int, default=1); ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    rec = run_selection(a.chassis, a.root, a.workers, a.n_boot)
    print("LAMBDA_STAR", rec["chassis"], rec["lam_star"], rec["label"])


def check_creg(contrast_row, d_r, d_a0):
    """C-REG (the regression arm, read on SCORE): R - A0 on 0-2 s acc_mae must be separated-worse (interval wholly above 0) and D(R) > D(A0). Else the lambda path does not act in the declared direction: INVALID.
    contrast_row: a hwm_eval.contrast row with 'lo' and 'label'."""
    return bool(contrast_row["lo"] > 0 and contrast_row["label"] == "worse" and d_r > d_a0)


if __name__ == "__main__":
    main()
