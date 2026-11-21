"""LON-PLAN run wrapper (AMEND_HWMPLAN_3 section 8.2, C-L4 and C-READBACK). One folder per lambda; the code of record's `run_hwm.py plan` does the planning, in this process, through its own cmd_plan.
What the wrapper adds (and the code of record lacks, section 8.1):
  * `--seed-mode` and `--lam` must be NAMED in the argv (the default `--lam 0.0` disables REG), and the parsed solver must equal the declared values (C-L4);
  * the out folder must not exist or must be empty (run_hwm skips an existing `plans_<arm>_s<seed>.npz`, whose name carries no lambda);
  * windows.npz is copied into the new folder from a source whose sha256 is declared;
  * SOLVER.json: argv, str(SolverCfg), lambda, checkpoint sha256, window-list sha256, TRAIN-latents sha256, code sha256s, thread counts;
  * every plans_*.npz is stamped with the lambda it was planned with (`lam_planned`, `seed_mode_planned`, `argv_sha256`) and read back;
  * readback(): batch 0 of a banked plan file re-planned from its SOLVER.json argv, bit-equal (C-READBACK).
usage (MM, per lambda; OMP_NUM_THREADS=2):
  python lonplan_run.py plan --windows-src <windows.npz> --windows-sha256 <hex> --declare-lam <lam> --declare-seed-mode hold -- \\
      --tag R35_s0 --ckpt <ckpt> --train-latents <pt> --train-tag R35_s0 --seed-mode hold --lam <lam> --arms hwm_oracle --seeds 0 --workers 1 --out <NEW DIR>
  python lonplan_run.py readback --out <DIR> [--arm hwm_oracle --seed 0]
Tier T0 (oracle goal): a world-model diagnostic, never driving performance. Nothing here selects a lambda or scores a window."""
import argparse, dataclasses, json, os, shutil, sys, time
import lonplan_common as C
import numpy as np
import torch
import run_hwm as R
import hwm_eval as E
import plantest_lib as P

ALLOWED_ARMS = ("hwm_oracle", "flat_oracle")
SOLVER_JSON = "SOLVER.json"


class RefusedRun(Exception):
    """The wrapper refuses to plan (C-L4): a mismatch, a non-empty folder, a wrong checksum."""


def parse(argv):
    """Parse run_hwm's own argv (add_common is the code of record's parser, reused unchanged) plus the three stage options of its main."""
    ap = argparse.ArgumentParser(prog="run_hwm plan")
    R.add_common(ap)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--arms", nargs="+", default=list(E.ARMS)); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--n-boot", type=int, default=2000)
    return ap.parse_args(list(argv))


def _named(argv, flag):
    return any(t == flag or t.startswith(flag + "=") for t in argv)


def check_declared(argv, lam, seed_mode):
    """C-L4, argv -> built. Returns the parsed namespace; raises RefusedRun on any mismatch."""
    argv = list(argv)
    for flag in ("--seed-mode", "--lam"):
        if not _named(argv, flag):
            raise RefusedRun(f"{flag} is not named in the argv: its default ({'hold' if flag == '--seed-mode' else '0.0'}) would be a silent choice")
    a = parse(argv)
    s = R.solver_from(a)
    if float(s.lam) != float(lam):
        raise RefusedRun(f"declared lam {lam} != built solver lam {s.lam} (argv lam {a.lam})")
    if s.seed_mode != seed_mode:
        raise RefusedRun(f"declared seed_mode {seed_mode!r} != built solver seed_mode {s.seed_mode!r}")
    if float(lam) != 0.0 and seed_mode != "hold":
        raise RefusedRun("REG needs the HOLD seed (hwm_plan.py:421)")
    bad = [x for x in a.arms if x not in ALLOWED_ARMS]
    if bad:
        raise RefusedRun(f"arms {bad}: LON-PLAN arms are the oracle arms {ALLOWED_ARMS}")
    lv = dict(stitch=a.stitch, vocab=a.vocab, constr_weight=a.constr_weight, plan_ctx=list(a.plan_ctx), decode_lam=a.decode_lam, goal_module=a.goal_module, goal_head=a.goal_head, no_goal_frame=a.no_goal_frame)
    ok = dict(stitch="decode", vocab="off", constr_weight=0.0, plan_ctx=[1], decode_lam=0.0, goal_module=None, goal_head=None, no_goal_frame=False)
    if lv != ok:
        raise RefusedRun(f"levers off their phase-1 defaults: {lv} (LON-PLAN levers stay at {ok})")
    if not a.out:
        raise RefusedRun("no --out")
    return a


def check_fresh_out(out):
    """The folder must not exist or be empty. Any plans_* (or anything else) inside refuses: run_hwm plan would skip an existing plans_<arm>_s<seed>.npz whatever its lambda."""
    if os.path.exists(out):
        if not os.path.isdir(out):
            raise RefusedRun(f"{out} is a file")
        left = sorted(os.listdir(out))
        if left:
            pl = [f for f in left if f.startswith("plans_")]
            raise RefusedRun(f"out folder {out} is not empty ({len(left)} entries, {len(pl)} plan files: {pl[:3]})")


def stamp_plans(out, lam, seed_mode, argv_sha):
    """Add lam_planned / seed_mode_planned / argv_sha256 to every plans_*.npz in out (atomic replace), and return {file: sha256}."""
    rec = {}
    for f in sorted(os.listdir(out)):
        if not (f.startswith("plans_") and f.endswith(".npz")):
            continue
        p = os.path.join(out, f)
        with np.load(p, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        d["lam_planned"] = np.array(float(lam), np.float64)
        d["seed_mode_planned"] = np.array(str(seed_mode))
        d["argv_sha256"] = np.array(str(argv_sha))
        tmp = p + ".tmp.npz"
        np.savez(tmp, **d)
        os.replace(tmp, p)
        rec[f] = C.sha256_file(p)
    return rec


def read_planned_lam(plan_file):
    """The lambda a plan file records, or None if it records none (a file the wrapper did not write)."""
    with np.load(plan_file, allow_pickle=False) as z:
        return float(z["lam_planned"]) if "lam_planned" in z.files else None


def verify_plan_lambdas(out):
    """Read back: every plans_* in out records the lambda of out/SOLVER.json. Returns {file: lam}; raises on a missing or different value."""
    sol = json.load(open(os.path.join(out, SOLVER_JSON), encoding="utf-8"))
    got = {}
    for f in sorted(os.listdir(out)):
        if f.startswith("plans_") and f.endswith(".npz"):
            lam = read_planned_lam(os.path.join(out, f))
            if lam is None or lam != float(sol["lam"]):
                raise RefusedRun(f"{f} records lam {lam}, SOLVER.json {sol['lam']}")
            got[f] = lam
    if not got:
        raise RefusedRun(f"{out} holds no plan file")
    return got


def run_lambda(argv, lam, seed_mode, windows_src, windows_sha256, ckpt_sha256=None, enforce_threads=True):
    """Plan one lambda into a fresh folder. argv is run_hwm's own plan argv (with --out). Returns the SOLVER.json dict."""
    a = check_declared(argv, lam, seed_mode)
    if enforce_threads and os.environ.get("OMP_NUM_THREADS") != str(C.THREADS):
        raise RefusedRun(f"OMP_NUM_THREADS must be {C.THREADS} for every LON-PLAN run (readback is bit-equal only at one thread count); it is {os.environ.get('OMP_NUM_THREADS')!r}")
    check_fresh_out(a.out)
    if C.sha256_file(windows_src) != windows_sha256:
        raise RefusedRun(f"windows source sha256 {C.sha256_file(windows_src)[:12]} != declared {windows_sha256[:12]}")
    ck_sha = C.sha256_file(a.ckpt)
    if ckpt_sha256 is not None and ck_sha != ckpt_sha256:
        raise RefusedRun(f"checkpoint sha256 {ck_sha[:12]} != declared {ckpt_sha256[:12]}")
    os.makedirs(a.out, exist_ok=True)
    shutil.copyfile(windows_src, os.path.join(a.out, "windows.npz"))
    argv_sha = C.sha256_text(json.dumps(list(argv)))
    sol = dict(kind="LONPLAN_SOLVER", argv=list(argv), argv_sha256=argv_sha, lam=float(lam), seed_mode=seed_mode, solver=str(R.solver_from(a)), arms=list(a.arms), seeds=list(a.seeds), workers=int(a.workers),
               ckpt=a.ckpt, ckpt_sha256=ck_sha, windows_sha256=windows_sha256, train_latents=a.train_latents, train_latents_sha256=(C.sha256_file(a.train_latents) if os.path.exists(a.train_latents) else None),
               code_sha256=C.code_of_record_sha256(), omp_num_threads=os.environ.get("OMP_NUM_THREADS"), torch_threads=int(torch.get_num_threads()), batch=int(P.BATCH),
               started=time.strftime("%Y-%m-%dT%H:%M:%S"), tier="T0", evidence="MEASURED at run time")
    C.dump_json(sol, os.path.join(a.out, SOLVER_JSON))
    R.cmd_plan(a)
    sol["plans"] = {k: dict(sha256=v, lam=float(lam)) for k, v in stamp_plans(a.out, lam, seed_mode, argv_sha).items()}
    sol["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    C.dump_json(sol, os.path.join(a.out, SOLVER_JSON))
    verify_plan_lambdas(a.out)
    return sol


def readback(out, arm="hwm_oracle", seed=0, lam_override=None):
    """C-READBACK: re-plan batch 0 of out/plans_<arm>_s<seed>.npz from out/SOLVER.json's argv and compare plan_raw and cost1 bit for bit.
    lam_override replaces the argv's lambda (the deliberate-regression arm: a file read back at the wrong lambda must fail). Returns dict(equal, ...)."""
    sol = json.load(open(os.path.join(out, SOLVER_JSON), encoding="utf-8"))
    if int(sol["torch_threads"]) != int(torch.get_num_threads()):
        raise RefusedRun(f"readback at {torch.get_num_threads()} torch threads, the run used {sol['torch_threads']}: not comparable")
    a = parse(sol["argv"])
    a.out = out
    if lam_override is not None:
        a.lam = float(lam_override)
    R._init({k: v for k, v in vars(a).items() if k != "fn"})
    _, r = R._work((arm, int(seed), 0, 1))
    with np.load(os.path.join(out, f"plans_{arm}_s{seed}.npz"), allow_pickle=False) as z:
        n = len(r["plan_raw"])
        eq_plan = bool(np.array_equal(z["plan_raw"][:n], r["plan_raw"]))
        eq_cost = bool(np.array_equal(z["cost1"][:n], r["cost1"]))
    return dict(equal=eq_plan and eq_cost, plan_equal=eq_plan, cost1_equal=eq_cost, n=int(n), lam_used=float(a.lam), arm=arm, seed=int(seed))


def readback_all(out):
    """Every plan file in out, batch 0 each."""
    res = {}
    for f in sorted(os.listdir(out)):
        if f.startswith("plans_") and f.endswith(".npz"):
            arm, s = f[len("plans_"):-len(".npz")].rsplit("_s", 1)
            res[f] = readback(out, arm, int(s))
    return res


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--windows-src", required=True); p.add_argument("--windows-sha256", required=True)
    p.add_argument("--declare-lam", type=float, required=True); p.add_argument("--declare-seed-mode", required=True, choices=["hold", "prior"])
    p.add_argument("--ckpt-sha256"); p.add_argument("argv", nargs=argparse.REMAINDER)
    q = sp.add_parser("readback")
    q.add_argument("--out", required=True); q.add_argument("--arm", default=None); q.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "plan":
        argv = [t for t in a.argv if t != "--"]
        sol = run_lambda(argv, a.declare_lam, a.declare_seed_mode, a.windows_src, a.windows_sha256, a.ckpt_sha256)
        print("PLANNED", sol["lam"], sorted(sol["plans"]))
    else:
        res = readback_all(a.out) if a.arm is None else {a.arm: readback(a.out, a.arm, a.seed)}
        print(json.dumps(res, indent=1))
        sys.exit(0 if all(v["equal"] for v in res.values()) else 3)


if __name__ == "__main__":
    main()
