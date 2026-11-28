"""AMEND_HWMPLAN_4 run wrapper (section 4, section 8.1 / C-L4-IMG, C-READBACK). One new folder per arm and chassis, in LON-PLAN's layout (windows.npz, SOLVER.json, plans_hwm_oracle_s<seed>.npz, DONE).
It reuses LON-PLAN's wrapper (lonplan_run: argv parser of the code of record, C-L4 declared == built, fresh-folder rule, thread rule, window / checkpoint sha256) and adds:
  * `--l2-image {off,prim,prim23}` and `--l1-sub {own,donor}` MUST be named in the argv (the default `off` disables the lever), and the parsed values must equal the declared ones;
  * every plan file is stamped l2_image_planned, l1_sub_planned (and lam / seed mode / argv sha256 as LON-PLAN does) and read back; SOLVER.json carries the sha256 of l2img.py and of this wrapper;
  * lam is 0 and the seed mode is hold for every arm; one worker (the generator streams and the thread count must match LON-PLAN's runs);
  * readback(): batch 0 of a banked plan file re-planned from its SOLVER.json argv, bit-equal on plan_raw and cost1; `l2_override` is the deliberate-regression arm (P read back at off must fail).
usage (OMP_NUM_THREADS=2):
  python amend4_run.py plan --windows-src <windows.npz> --windows-sha256 <hex> --declare-l2-image <off|prim|prim23> [--declare-l1-sub own|donor] [--ckpt-sha256 <hex>] -- \\
      --tag R35_s0 --ckpt <ckpt> --train-latents <pt> --train-tag R35_s0 --seed-mode hold --lam=0 --l2-image prim [--l1-sub donor] --arms hwm_oracle --seeds 0 1 --workers 1 --out <NEW DIR>
  python amend4_run.py readback --out <DIR> [--seed 0]
Tier T0 (oracle goal), a world-model diagnostic. Nothing here selects or tunes anything."""
import argparse, json, os, shutil, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LONPLAN_BUILD = os.environ.get("AMEND4_LONPLAN_BUILD", "F:/TanitAD-scratch/actrsp/lonplan/build")
sys.path.insert(0, LONPLAN_BUILD)
import lonplan_common as C                  # sets CUDA_VISIBLE_DEVICES, puts the code of record on sys.path
import lonplan_run as W
import numpy as np
import torch
import hwm_eval as E
import plantest_lib as P
import run_hwm as R
import l2img as L

SOLVER_JSON = "SOLVER.json"
RefusedRun = W.RefusedRun


def split_l2(argv):
    """(namespace(l2_image, l1_sub), the rest of the argv for run_hwm's own parser)."""
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--l2-image", choices=list(L.MODES)); ap.add_argument("--l1-sub", choices=list(L.L1SUB), default="own")
    return ap.parse_known_args(list(argv))


def check_declared_l2(argv, declare_l2, declare_l1sub="own"):
    """C-L4-IMG, argv -> built. Returns (namespace of run_hwm's parser, l2_image, l1_sub); raises RefusedRun."""
    argv = list(argv)
    if not W._named(argv, "--l2-image"):
        raise RefusedRun("--l2-image is not named in the argv: its default (off) would disable the lever silently")
    ns, rest = split_l2(argv)
    if ns.l2_image != declare_l2:
        raise RefusedRun(f"declared l2_image {declare_l2!r} != argv l2_image {ns.l2_image!r}")
    if ns.l1_sub != declare_l1sub:
        raise RefusedRun(f"declared l1_sub {declare_l1sub!r} != argv l1_sub {ns.l1_sub!r}")
    if ns.l1_sub == "donor" and ns.l2_image != "prim":
        raise RefusedRun("--l1-sub donor is the regression arm of the prim lever: it needs --l2-image prim")
    a = W.check_declared(rest, 0.0, "hold")
    if int(a.workers) != 1:
        raise RefusedRun("one worker only (the generator streams and the thread count must match LON-PLAN's runs)")
    return a, ns.l2_image, ns.l1_sub


def module_sha256():
    return dict(l2img_py=C.sha256_file(os.path.join(HERE, "l2img.py")), amend4_run_py=C.sha256_file(os.path.abspath(__file__)),
                lonplan_run_py=C.sha256_file(os.path.join(LONPLAN_BUILD, "lonplan_run.py")), lonplan_common_py=C.sha256_file(os.path.join(LONPLAN_BUILD, "lonplan_common.py")))


def _unit(ctx, arm, seed, b, l2, l1sub, mutate=None):
    ws, geom, a, st = ctx["ws"], ctx["geom"], ctx["a"], ctx["st"]
    idx = np.arange(b * P.BATCH, min(ws.n, (b + 1) * P.BATCH))
    t = time.time()
    r = L.plan_unit(ctx["planner"], l2, R.goal_fn_for(arm, ws, geom, a, st), ws.z[:, 0], ws.A[:, 0], idx, seed, b, eids=ws.eid, l1_sub=l1sub, mutate=mutate)
    r["wall_s"] = np.array(time.time() - t)
    return r


def stamp(out, lam, seed_mode, argv_sha, l2, l1sub):
    rec = {}
    for f in sorted(os.listdir(out)):
        if not (f.startswith("plans_") and f.endswith(".npz")):
            continue
        p = os.path.join(out, f)
        with np.load(p, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        d.update(lam_planned=np.array(float(lam), np.float64), seed_mode_planned=np.array(str(seed_mode)), argv_sha256=np.array(str(argv_sha)),
                 l2_image_planned=np.array(str(l2)), l1_sub_planned=np.array(str(l1sub)))
        tmp = p + ".tmp.npz"
        np.savez(tmp, **d)
        os.replace(tmp, p)
        rec[f] = C.sha256_file(p)
    return rec


def read_planned(plan_file):
    with np.load(plan_file, allow_pickle=False) as z:
        return (str(z["l2_image_planned"]) if "l2_image_planned" in z.files else None, str(z["l1_sub_planned"]) if "l1_sub_planned" in z.files else None)


def verify_stamps(out):
    sol = json.load(open(os.path.join(out, SOLVER_JSON), encoding="utf-8"))
    got = {}
    for f in sorted(os.listdir(out)):
        if f.startswith("plans_") and f.endswith(".npz"):
            l2, l1 = read_planned(os.path.join(out, f))
            if l2 != sol["l2_image"] or l1 != sol["l1_sub"]:
                raise RefusedRun(f"{f} records l2_image {l2!r} / l1_sub {l1!r}, SOLVER.json {sol['l2_image']!r} / {sol['l1_sub']!r}")
            got[f] = l2
    if not got:
        raise RefusedRun(f"{out} holds no plan file")
    return got


def run_arm(argv, declare_l2, windows_src, windows_sha256, declare_l1sub="own", ckpt_sha256=None, enforce_threads=True, prepare=None, mutate=None):
    """Plan one arm into a fresh folder. `prepare` (tests only) replaces run_hwm.prepare, as LON-PLAN's tests do for the toy stack. `mutate` is refused here unless prepare is set (tests)."""
    a, l2, l1sub = check_declared_l2(argv, declare_l2, declare_l1sub)
    if mutate is not None and prepare is None:
        raise RefusedRun("mutation hooks are for the controls only")
    if enforce_threads and os.environ.get("OMP_NUM_THREADS") != str(C.THREADS):
        raise RefusedRun(f"OMP_NUM_THREADS must be {C.THREADS} for every AMEND4 run; it is {os.environ.get('OMP_NUM_THREADS')!r}")
    W.check_fresh_out(a.out)
    if C.sha256_file(windows_src) != windows_sha256:
        raise RefusedRun(f"windows source sha256 {C.sha256_file(windows_src)[:12]} != declared {windows_sha256[:12]}")
    ck_sha = C.sha256_file(a.ckpt) if os.path.exists(a.ckpt) else None
    if ckpt_sha256 is not None and ck_sha != ckpt_sha256:
        raise RefusedRun(f"checkpoint sha256 {str(ck_sha)[:12]} != declared {ckpt_sha256[:12]}")
    os.makedirs(a.out, exist_ok=True)
    shutil.copyfile(windows_src, os.path.join(a.out, "windows.npz"))
    argv_sha = C.sha256_text(json.dumps(list(argv)))
    sol = dict(kind="AMEND4_SOLVER", argv=list(argv), argv_sha256=argv_sha, lam=0.0, seed_mode="hold", l2_image=l2, l1_sub=l1sub, solver=str(R.solver_from(a)), arms=list(a.arms), seeds=list(a.seeds), workers=1,
               ckpt=a.ckpt, ckpt_sha256=ck_sha, windows_sha256=windows_sha256, train_latents=a.train_latents,
               train_latents_sha256=(C.sha256_file(a.train_latents) if os.path.exists(a.train_latents) else None), code_sha256=C.code_of_record_sha256(), module_sha256=module_sha256(),
               omp_num_threads=os.environ.get("OMP_NUM_THREADS"), torch_threads=int(torch.get_num_threads()), batch=int(P.BATCH), started=time.strftime("%Y-%m-%dT%H:%M:%S"), tier="T0", evidence="MEASURED at run time")
    C.dump_json(sol, os.path.join(a.out, SOLVER_JSON))
    ad = {k: v for k, v in vars(a).items() if k != "fn"}
    if prepare is not None:
        R.prepare = prepare
    R._init(ad)
    ctx = R._CTX
    nb = (ctx["ws"].n + P.BATCH - 1) // P.BATCH
    for arm in a.arms:
        for seed in a.seeds:
            parts = []
            for b in range(nb):
                parts.append((b, _unit(ctx, arm, seed, b, l2, l1sub, mutate)))
                print("unit", (arm, seed, b), round(float(parts[-1][1]["wall_s"]), 1), "s", flush=True)
            np.savez(os.path.join(a.out, f"plans_{arm}_s{seed}.npz"), **L.merge_units(parts))
    if l2 != "off":                                                          # the MM's requirement for AMEND_HWMPLAN_5: level 2's chosen primitives are in every P / P23 / RS plan file
        for f in sorted(os.listdir(a.out)):
            if f.startswith("plans_") and f.endswith(".npz"):
                with np.load(os.path.join(a.out, f), allow_pickle=False) as z:
                    if "l2_prim_raw" not in z.files or z["l2_prim_raw"].shape != (len(z["plan_raw"]), ctx["geom"].h2 * ctx["geom"].c2, 2):
                        raise RefusedRun(f"{f} lacks l2_prim_raw (n, h2*c2, 2)")
    sol["plans"] = {k: dict(sha256=v, l2_image=l2, l1_sub=l1sub) for k, v in stamp(a.out, 0.0, "hold", argv_sha, l2, l1sub).items()}
    sol["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    C.dump_json(sol, os.path.join(a.out, SOLVER_JSON))
    verify_stamps(a.out)
    return sol


def readback(out, arm="hwm_oracle", seed=0, l2_override=None, prepare=None):
    """C-READBACK: batch 0 of out/plans_<arm>_s<seed>.npz re-planned from out/SOLVER.json's argv; plan_raw and cost1 bit-equal. l2_override is the deliberate-regression arm."""
    sol = json.load(open(os.path.join(out, SOLVER_JSON), encoding="utf-8"))
    if int(sol["torch_threads"]) != int(torch.get_num_threads()):
        raise RefusedRun(f"readback at {torch.get_num_threads()} torch threads, the run used {sol['torch_threads']}: not comparable")
    ns, rest = split_l2(sol["argv"])
    a = W.parse(rest)
    a.out = out
    l2 = ns.l2_image if l2_override is None else l2_override
    l1 = ns.l1_sub if not (l2_override == "off") else "own"
    if prepare is not None:
        R.prepare = prepare
    R._init({k: v for k, v in vars(a).items() if k != "fn"})
    r = _unit(R._CTX, arm, int(seed), 0, l2, l1)
    with np.load(os.path.join(out, f"plans_{arm}_s{seed}.npz"), allow_pickle=False) as z:
        n = len(r["plan_raw"])
        eq_plan = bool(np.array_equal(z["plan_raw"][:n], r["plan_raw"]))
        eq_cost = bool(np.array_equal(z["cost1"][:n], r["cost1"]))
    return dict(equal=eq_plan and eq_cost, plan_equal=eq_plan, cost1_equal=eq_cost, n=int(n), l2_image_used=l2, arm=arm, seed=int(seed))


def readback_all(out):
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
    p.add_argument("--declare-l2-image", required=True, choices=list(L.MODES)); p.add_argument("--declare-l1-sub", default="own", choices=list(L.L1SUB))
    p.add_argument("--ckpt-sha256"); p.add_argument("argv", nargs=argparse.REMAINDER)
    q = sp.add_parser("readback")
    q.add_argument("--out", required=True); q.add_argument("--arm", default=None); q.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "plan":
        argv = [t for t in a.argv if t != "--"]
        sol = run_arm(argv, a.declare_l2_image, a.windows_src, a.windows_sha256, a.declare_l1_sub, a.ckpt_sha256)
        open(os.path.join(sol["argv"][sol["argv"].index("--out") + 1], "DONE"), "w").write("ok\n")
        print("PLANNED", sol["l2_image"], sol["l1_sub"], sorted(sol["plans"]))
    else:
        res = readback_all(a.out) if a.arm is None else {a.arm: readback(a.out, a.arm, a.seed)}
        print(json.dumps(res, indent=1))
        sys.exit(0 if all(v["equal"] for v in res.values()) else 3)


if __name__ == "__main__":
    main()
