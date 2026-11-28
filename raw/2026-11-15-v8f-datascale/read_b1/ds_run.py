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
VARIANT_FLAGS = {"real": [], "donor": ["--nav-shuffle"], "zero": ["--nav-withheld"]}
MECH_SUFFIX = ("", "-tacabs", "-tacorc", "-tacorc-all-latcons")


def ck_path(arm, stage):
    return f"{L.CK}/{L.item(arm, stage)}/ckpt_last.pt"


def env():
    return dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
                PYTHONPATH=";".join(os.path.normpath(p) for p in (L.TREE + "/stack", L.TREE + "/taniteval")))


def t1_argv(arm, stage, out=None):
    od = out or L.T1
    tag = L.t1_tag(arm, stage)
    a = [L.PY, os.path.normpath(L.TREE + "/stack/scripts/v8f_t1_eval.py"), "run", "--ckpt", os.path.normpath(ck_path(arm, stage)), "--tag", tag,
         "--tracks", L.SF + "/inputs/native_tracks.pt", "--eval-cache", L.SF + "/cache", "--eval-grid", L.pinned("eval_grid_windows.json"), "--eval-join", L.pinned("join_eval141.npz"),
         "--out-dir", od, "--replan-steps", "10", "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic", "--limit", "causal", "--dump-ticks", od + "/ticks",
         "--dump-heads", "--device", "cpu"]
    return a + (["--max-windows", str(SMOKE)] if SMOKE else []), od, tag


def _t0_common(ckpt, tag, out):
    return [L.PY, os.path.normpath(L.TREE + "/stack/scripts/v8f_mini_eval.py"), "run", "--ckpt", os.path.normpath(ckpt), "--tag", tag,
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
    return tag, _t0_common(ck_path(arm, stage), tag, out) + ["--ages", "fresh"] + VARIANT_FLAGS[variant]


def wait_commit():
    for _ in range(300):
        if L.commit_free_gb() >= L.MIN_COMMIT_GB:
            return
        time.sleep(60)
    raise SystemExit(f"REFUSED: commit headroom stayed under {L.MIN_COMMIT_GB} GB for 5 h")


def import_check():
    r = subprocess.run([L.PY, "-c", "import tanitad, taniteval; print(tanitad.__file__); print(taniteval.__file__)"], env=env(), capture_output=True, encoding="utf-8")
    root = os.path.normpath(L.TREE).lower()
    paths = [x.strip() for x in r.stdout.splitlines()]
    if r.returncode or len(paths) != 2 or not all(os.path.normpath(x).lower().startswith(root) for x in paths):
        return f"tanitad / taniteval do not import from the read tree: {paths} {r.stderr[-200:]}"
    return None


def preflight(items):
    why = L.disk_problem() or L.tree_problem() or import_check()
    if why:
        raise SystemExit("REFUSED: " + why)
    bad = {f"{a}.{s}": L.bank_problem(a, s) for a, s in items if L.bank_problem(a, s)}
    return bad


def _run(argv, log):
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w", encoding="utf-8") as lf:
        return subprocess.run(argv, env=env(), stdout=lf, stderr=subprocess.STDOUT, encoding="utf-8").returncode


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
        if ok and L.sha256_file(ck_path(arm, stage)) != json.load(open(f"{L.CK}/{L.item(arm, stage)}/BANK_RECORD.json"))["sha256"]:
            ok = False
        if ok and not SMOKE:
            json.dump({"tag": tag, "arm": arm, "stage": stage, "commit": L.COMMIT, "n_dumps": n, "threads": 2, "wall_s": round(time.time() - t0, 1), "written_by": "ds_run.py"}, open(done, "w"), indent=1)
        print(time.strftime("%H:%M:%S"), tag, "rc", r, "ok" if ok else "FAILED", round(time.time() - t0), "s", flush=True)
        rc |= (not ok)
    return rc


def _t0_loop(jobs, out):
    rc = 0
    os.makedirs(out, exist_ok=True)
    for tag, a in jobs:
        npz = f"{out}/eval_{tag}.npz"
        if os.path.exists(npz) and os.path.exists(f"{out}/eval_{tag}.json"):
            print("skip (exists)", tag, flush=True)
            continue
        wait_commit()
        open(f"{out}/argv_{tag}.txt", "w", encoding="utf-8").write("\n".join(a) + "\n")
        t0 = time.time()
        r = _run(a, f"{out}/log_{tag}.txt")
        ok = r == 0 and os.path.exists(npz)
        print(time.strftime("%H:%M:%S"), tag, "rc", r, "ok" if ok else "FAILED", round(time.time() - t0), "s", flush=True)
        rc |= (not ok)
    return rc


def cmd_mech(items):
    out = L.OUT + "/t0mech"
    return _t0_loop([j for arm, stage in items if stage == "A" for j in mech_argvs(arm, stage, out)], out)


def cmd_route(items):
    out = L.OUT + "/t0route"
    return _t0_loop([route_argv(arm, stage, v, out) for arm, stage in items if stage == "B" for v in VARIANT_FLAGS], out)


def parse_items(rest):
    if not rest:
        return list(L.READ_ITEMS)
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
    items = parse_items(a[1:])
    bad = preflight(items)
    if bad:
        for k, v in bad.items():
            print("REFUSED", k, v, flush=True)
        sys.exit(1)
    sys.exit({"dumps": cmd_dumps, "mech": cmd_mech, "route": cmd_route}[a[0]](items))


if __name__ == "__main__":
    main()
