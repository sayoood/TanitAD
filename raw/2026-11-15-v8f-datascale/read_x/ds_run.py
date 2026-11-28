"""ds_run.py -- the DSCALE10K block-1 dumps and T0 passes (CPU, 2 threads, zero training), the programme's own evaluators UNEDITED from the code-of-record export (cdbff971, sha256-pinned in READ_PINS.json).
  python ds_run.py dumps [ITEM ...]   T1-kin dump per checkpoint (card 4: self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene) -> t1/dump_DS-<arm>-<stage>_m10_causal_heads
  python ds_run.py mech  [ITEM ...]   the four T0 evals of the DSCALE instrument per checkpoint (E0 --project --families, E1 --tac absmask, E2 --tac oracle, E3 oracle + all-latcons; queue :evaldscale) -> out/t0mech/
  python ds_run.py route [ITEM ...]   T0 route passes (--ages fresh; real / donor --nav-shuffle / zero --nav-withheld) for the stage-B checkpoints (STRATEGIC family) -> out/t0route/
ITEM = <arm>.<stage>, default: every block-1 item that applies.  Every item is REFUSED (nothing launched) unless ds_lib.bank_problem() is None (sha256 against the banker's record, commit, step count)
and the tree pins hold; an F: with under 25 GB free or a dev-box commit headroom under 5.5 GB (waited for up to 5 h) stops the run.  ONE environment and ONE thread count (2) for every dump.
The recipe of record takes the v9 ego-future token as nav (no text-nav file): no nav arguments are passed (the STRAT1 C0 pattern).  A pass whose output exists is skipped.  --smoke-windows N: a code check, never a result."""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_lib as L  # noqa: E402
import ds_t0 as T0  # noqa: E402

SMOKE = int(os.environ.get("DS_SMOKE_WINDOWS", "0"))
VARIANT_FLAGS = {"real": [], "donor": ["--nav-shuffle"], "zero": ["--nav-withheld"], "realx": [], "donorx": ["--nav-shuffle"]}   # realx / donorx = the K-EVALX identity passes: `real` / `donor` again on the zero tree (only the 008f48b7 evaluator writes win_ci / win_t, which the reader checks)
# which tree evaluator runs which variant = L.VARIANT_TREE (real, donor: the read tree cdbff971 / 50c70ae8; zero, realx: the STRAT1 tree e81bde01 / 008f48b7, the only one with --nav-withheld); both trees sha256-pinned in READ_PINS.json
MECH_SUFFIX = ("", "-tacabs", "-tacorc", "-tacorc-all-latcons")
ROUTE_STAGES = ("A", "B") if L.BLOCK == 4 else (("A",) if L.BLOCK == 3 else ("B",))      # block X: X_s0 is routed at BOTH stages (STRATEGIC read at stage A and stage B)
ROUTE_STAGE = ROUTE_STAGES[0]         # block 1 / 2: the route passes (STRATEGIC) are made for the stage-B checkpoints; block 3 has stage-A arms only, so its route passes are stage A (the five arms of L.ROUTE_ITEMS_B3)


def ck_path(arm, stage):
    return f"{L.ck_root(arm)}/{L.item(arm, stage)}/ckpt_last.pt"


def env(tree=None):
    tree = tree or L.TREE
    return dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
                PYTHONPATH=";".join(os.path.normpath(p) for p in (tree + "/stack", tree + "/taniteval")))


def t1_argv(arm, stage, out=None):
    od = out or L.T1
    tag = L.t1_tag(arm, stage)
    a = [L.PY, os.path.normpath(L.TREE + "/stack/scripts/v8f_t1_eval.py"), "run", "--ckpt", os.path.normpath(ck_path(arm, stage)), "--tag", tag,
         "--tracks", L.SF + "/inputs/native_tracks.pt", "--eval-cache", L.SF + "/cache", "--eval-grid", L.pinned("eval_grid_windows.json"), "--eval-join", L.pinned("join_eval141.npz"),
         "--out-dir", od, "--replan-steps", "10", "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic", "--limit", "causal", "--dump-ticks", od + "/ticks",
         "--dump-heads", "--device", "cpu"]
    return a + (["--max-windows", str(SMOKE)] if SMOKE else []), od, tag


def _t0_common(ckpt, tag, out, tree=None):
    return [L.PY, os.path.normpath((tree or L.TREE) + "/stack/scripts/v8f_mini_eval.py"), "run", "--ckpt", os.path.normpath(ckpt), "--tag", tag,
            "--eval-grid", L.pinned("eval_grid_windows.json"), "--eval-kin", L.RAWT + "/eval141_kin.npz", "--eval-join", L.pinned("join_eval141.npz"),
            "--v9-eval", L.SF + "/inputs/v9/v9_labels_eval139.npz", "--lead-eval", L.SF + "/inputs/v9/lead_eval139.npz", "--speed-eval", L.SF + "/inputs/v9/refcv6_speed_max_v8_eval.jsonl",
            "--kin-mode", "yaw", "--p10", L.SF + "/inputs/p10/p10_prior.pt", "--eval-cache", L.SF + "/cache", "--tracks", L.SF + "/inputs/native_tracks.pt",
            "--out-dir", out, "--device", "cpu"]


def mech_argvs(arm, stage, out):
    """the four evals of the queue's :evaldscale, one argv each; tag NAME_s0 + suffix (ds_t0.name_of)"""
    nm = T0.name_of(arm, stage) + "_s0"
    ck = ck_path(arm, stage)
    extra = {"": ["--project", "--families"], "-tacabs": ["--tac", "absmask"], "-tacorc": ["--tac", "oracle"],
             "-tacorc-all-latcons": ["--tac", "oracle", "--tac-groups", "all-latcons", "--tac-alpha", "1"]}
    return [(nm + s, _t0_common(ck, nm + s, out) + extra[s]) for s in MECH_SUFFIX]


def route_argv(arm, stage, variant, out):
    tag = f"DS-{arm}-{stage}_{variant}"
    return tag, _t0_common(ck_path(arm, stage), tag, out, L.tree_of(variant)) + ["--ages", "fresh"] + VARIANT_FLAGS[variant]


def wait_commit():
    for _ in range(300):
        if L.commit_free_gb() >= L.MIN_COMMIT_GB:
            return
        time.sleep(60)
    raise SystemExit(f"REFUSED: commit headroom stayed under {L.MIN_COMMIT_GB} GB for 5 h")


def import_check(tree=None):
    r = subprocess.run([L.PY, "-c", "import tanitad, taniteval; print(tanitad.__file__); print(taniteval.__file__)"], env=env(tree), capture_output=True, encoding="utf-8")
    root = os.path.normpath(tree or L.TREE).lower()
    paths = [x.strip() for x in r.stdout.splitlines()]
    if r.returncode or len(paths) != 2 or not all(os.path.normpath(x).lower().startswith(root) for x in paths):
        return f"tanitad / taniteval do not import from the read tree: {paths} {r.stderr[-200:]}"
    return None


_DRY = "\n".join(["import argparse, os, runpy, sys",
                  "def _pa(self, args=None, namespace=None):",
                  "    ns, unk = self.parse_known_args(args, namespace)",
                  "    if unk:",
                  "        print('UNRECOGNIZED', unk); sys.exit(2)",
                  "    print('ACCEPTED'); sys.exit(0)",
                  "argparse.ArgumentParser.parse_args = _pa",
                  "sys.argv = sys.argv[1:]",
                  "runpy.run_path(sys.argv[0], run_name='__main__')", ""])


def argv_accepted(argv, tree):
    """-> (True, '') iff the evaluator in `tree` ACCEPTS `argv` under its own argparse (a dry parse in a subprocess: parse_args is replaced by parse_known_args, nothing is loaded or run).  argv = [python, script, ...]"""
    r = subprocess.run([argv[0], "-c", _DRY, *argv[1:]], env=env(tree), capture_output=True, encoding="utf-8", timeout=300)
    out = (r.stdout or "").strip().splitlines()
    return (r.returncode == 0 and bool(out) and out[-1] == "ACCEPTED"), (out[-1] if out else (r.stderr or "")[-200:])


def route_argv_problems(arm="U_s0", stage=None):
    stage = stage or ROUTE_STAGE
    """every route variant's argv is accepted by the evaluator of the tree it is routed to (-> list of reasons, empty = all accepted)"""
    bad = []
    for v in VARIANT_FLAGS:
        ok, why = argv_accepted(route_argv(arm, stage, v, "o")[1], L.tree_of(v))
        if not ok:
            bad.append(f"{v}: {why}")
    return bad


def preflight(items, route=False):
    why = L.disk_problem() or L.tree_problem() or import_check()
    if not why and route:        # the zero tree is pinned and import-checked on its own: it runs the `zero` and `realx` passes
        why = L.tree_problem(L.TREE_ZERO, "tree_zero_sha256") or import_check(L.TREE_ZERO)
        if not why and route_argv_problems():
            why = "a route variant's argv is not accepted by its evaluator: " + "; ".join(route_argv_problems())
    if why:
        raise SystemExit("REFUSED: " + why)
    bad = {f"{a}.{s}": L.bank_problem(a, s) for a, s in items if L.bank_problem(a, s)}
    return bad


def _run(argv, log, tree=None):
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w", encoding="utf-8") as lf:
        return subprocess.run(argv, env=env(tree), stdout=lf, stderr=subprocess.STDOUT, encoding="utf-8").returncode


def cmd_dumps(items):
    rc = 0
    for arm, stage in items:
        a, od, tag = t1_argv(arm, stage)
        done = f"{od}/done_{tag}.txt"
        if os.path.exists(done):
            print("skip (done)", tag, flush=True)
            continue
        wait_commit()
        os.makedirs(od + "/ticks", exist_ok=True)
        open(f"{od}/argv_{tag}.txt", "w", encoding="utf-8").write("\n".join(a) + "\n")
        t0 = time.time()
        r = _run(a, f"{od}/log_{tag}.txt")
        dd = f"{od}/dump_{tag}"
        n = len([f for f in os.listdir(dd) if f.startswith("ep") and f.endswith(".npz")]) if os.path.isdir(dd) else 0
        ok = r == 0 and os.path.exists(f"{od}/t1v8f_{tag}.json") and (n == L.EVAL_N_EPISODES or SMOKE) and os.path.exists(f"{od}/ticks/ticks_{tag}.npz")
        if ok and L.sha256_file(ck_path(arm, stage)) != json.load(open(f"{L.ck_root(arm)}/{L.item(arm, stage)}/BANK_RECORD.json"))["sha256"]:
            ok = False
        if ok and not SMOKE:
            json.dump({"tag": tag, "arm": arm, "stage": stage, "commit": L.COMMIT, "n_dumps": n, "threads": 2, "wall_s": round(time.time() - t0, 1), "written_by": "ds_run.py"}, open(done, "w"), indent=1)
        print(time.strftime("%H:%M:%S"), tag, "rc", r, "ok" if ok else "FAILED", round(time.time() - t0), "s", flush=True)
        rc |= (not ok)
    return rc


def _t0_loop(jobs, out):
    rc = 0
    os.makedirs(out, exist_ok=True)
    for job in jobs:
        tag, a = job[0], job[1]
        tree = job[2] if len(job) > 2 else None
        npz = f"{out}/eval_{tag}.npz"
        if os.path.exists(npz) and os.path.exists(f"{out}/eval_{tag}.json"):
            print("skip (exists)", tag, flush=True)
            continue
        wait_commit()
        open(f"{out}/argv_{tag}.txt", "w", encoding="utf-8").write("\n".join(a) + "\n")
        t0 = time.time()
        r = _run(a, f"{out}/log_{tag}.txt", tree)
        ok = r == 0 and os.path.exists(npz)
        print(time.strftime("%H:%M:%S"), tag, "rc", r, "ok" if ok else "FAILED", round(time.time() - t0), "s", flush=True)
        rc |= (not ok)
    return rc


def cmd_mech(items):
    out = L.OUT + "/t0mech"
    return _t0_loop([j for arm, stage in items if stage == "A" for j in mech_argvs(arm, stage, out)], out)


def kevalx_compare(real_npz, real_json, x_npz, x_json):
    """K-EVALX: the `real` pass of the 50c70ae8 evaluator == the `real` pass of the 008f48b7 evaluator on EVERY array the old one wrote (np.array_equal, atol 0, NaN == NaN; dtype and shape equal) and every
    number of its result JSON (the new evaluator's extra keys nav_variant / nav_source are reported, not compared).  -> (ok, diffs, extra_keys)"""
    import numpy as np
    diffs = []
    with np.load(real_npz, allow_pickle=False) as a, np.load(x_npz, allow_pickle=False) as b:
        for k in a.files:
            if k not in b.files:
                diffs.append(f"{k}: missing in the 008f48b7 output")
                continue
            x, y = a[k], b[k]
            if x.shape != y.shape or x.dtype != y.dtype:
                diffs.append(f"{k}: shape/dtype {x.shape}/{x.dtype} != {y.shape}/{y.dtype}")
            elif x.dtype.kind in "fc":
                if not np.array_equal(x, y, equal_nan=True):
                    diffs.append(f"{k}: arrays differ (atol 0)")
            elif not np.array_equal(x, y):
                diffs.append(f"{k}: arrays differ")
        extra = sorted(set(b.files) - set(a.files))
    ja, jb = json.load(open(real_json, encoding="utf-8")), json.load(open(x_json, encoding="utf-8"))

    def walk(u, v, path):
        if isinstance(u, dict):
            if not isinstance(v, dict):
                diffs.append(f"json {path}: type")
                return
            for k in u:
                if k in ("tag", "argv", "ckpt", "out_dir"):
                    continue
                if k not in v:
                    diffs.append(f"json {path}/{k}: missing")
                else:
                    walk(u[k], v[k], f"{path}/{k}")
        elif isinstance(u, list):
            if not isinstance(v, list) or len(u) != len(v):
                diffs.append(f"json {path}: list length")
            else:
                for i, (p, q) in enumerate(zip(u, v)):
                    walk(p, q, f"{path}[{i}]")
        elif u != v and not (u != u and v != v):
            diffs.append(f"json {path}: {u!r} != {v!r}")
    walk(ja, jb, "")
    return not diffs, diffs, extra


def cmd_kevalx(items, out=None):
    out = out or L.OUT + "/t0route"
    rec, rc = {}, 0
    for arm, stage in items:
        if stage not in ROUTE_STAGES:
            continue
        for v, vx in (("real", "realx"), ("donor", "donorx")):
            ok, diffs, extra = kevalx_compare(f"{out}/eval_DS-{arm}-{stage}_{v}.npz", f"{out}/eval_DS-{arm}-{stage}_{v}.json", f"{out}/eval_DS-{arm}-{stage}_{vx}.npz", f"{out}/eval_DS-{arm}-{stage}_{vx}.json")
            rec[f"{arm}.{stage} {v}"] = {"identical": ok, "diffs": diffs[:20], "extra_keys_in_008f48b7": extra}
            rc |= (not ok)
    passed = (not rc) and bool(rec)
    json.dump({"gate": "K-EVALX", "pass": passed, "items": rec, "evaluators": {"real, donor": "50c70ae8 (cdbff971 tree)", "realx, donorx": "008f48b7 (e81bde01 tree)"}}, open(out + "/KEVALX.json", "w"), indent=1)
    print("K-EVALX", "PASS" if passed else "FAIL (the zero passes are not admissible)", json.dumps({k: v["identical"] for k, v in rec.items()}), flush=True)
    return 0 if passed else 1


def route_jobs(items, out):
    return [route_argv(arm, stage, v, out) + (L.tree_of(v),) for arm, stage in items if stage in ROUTE_STAGES for v in VARIANT_FLAGS]


def cmd_route(items):
    out = L.OUT + "/t0route"
    rc = _t0_loop(route_jobs(items, out), out)
    return rc or cmd_kevalx(items, out)


def parse_items(rest, route=False):
    if not rest:
        return list(L.ROUTE_ITEMS_X if (route and L.BLOCK == 4) else (L.ROUTE_ITEMS_B3 if (route and L.BLOCK == 3) else L.DEFAULT_ITEMS))
    out = []
    for x in rest:
        arm, stage = x.split(".")
        assert arm in L.ARMS and stage in L.STAGES_OF[arm], f"unknown item {x}"
        out.append((arm, stage))
    return out


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if not a or a[0] not in ("dumps", "mech", "route"):
        raise SystemExit(__doc__)
    items = parse_items(a[1:], route=a[0] == "route")
    bad = preflight(items, route=a[0] == "route")
    if bad:
        for k, v in bad.items():
            print("REFUSED", k, v, flush=True)
        sys.exit(1)
    sys.exit({"dumps": cmd_dumps, "mech": cmd_mech, "route": cmd_route}[a[0]](items))


if __name__ == "__main__":
    main()
