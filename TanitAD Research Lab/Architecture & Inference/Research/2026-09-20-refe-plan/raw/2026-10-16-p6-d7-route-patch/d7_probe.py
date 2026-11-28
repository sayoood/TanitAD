"""D7 diagnosis (Master Mind, 2026-10-16): why are P6's rank-1 lane twins identical to the logged route?

H1: the spawned pool workers never get the route patch (route_rank.apply runs in the parent only).
H2: PDM-Closed's centreline/route lanes (D9 parts[3], parts[4]) never call `_choose_route_edge`,
    so the patch cannot change them whatever process it is installed in.

For ONE rank-1 twin token, in THIS process (patch applied via route_rank.apply(1)) we count
`_choose_route_edge` calls during metric_cache_parts(sc), and compare the route digest with an
unpatched rank-0 build of the same token in a fresh subprocess. A spawned worker reports whether
the patch is installed there.
Prints ZZD7DIAG lines only. Read-only: writes nothing to the cache.
"""
import json, os, sys, subprocess, multiprocessing as mp

REFE = "/workspace/refe-plan-p6/refe"
sys.path.insert(0, REFE)
os.chdir(REFE)
DATA = "/workspace/data"
CBANK = f"{DATA}/refe_navtrain/train_grow_cmd"
DBS = f"{DATA}/navtrain_dbs"
MAPS = os.environ.get("NUPLAN_MAPS_ROOT", "")


def first_twin(k=1):
    import p6_prep
    for r in p6_prep._bank_rows(CBANK):
        aug = r.get("aug")
        if isinstance(aug, dict) and aug.get("kind") == "lane_rank" and int(aug.get("value")) == k:
            return r["log_name"], r["token"], aug
    return None


def worker_state(_):
    try:
        import route_lane_rank_patch as P
        inst = getattr(P, "_INSTALLED", False)
    except Exception as e:  # noqa
        inst = f"import error {type(e).__name__}"
    return {"pid": os.getpid(), "env": os.environ.get("DRIVERL_EVAL_ROUTE_LANE_RANK"), "installed": inst,
            "sitecustomize": getattr(sys.modules.get("sitecustomize"), "__file__", None)}


def parts_digest(log, tok):
    import build_metric_cache as B, navsim_pdm_targets as PT, p6_prep
    scs = B._scenarios(DBS, MAPS, log, {tok})
    sc = scs.get(tok)
    if sc is None:
        return None
    parts = PT.metric_cache_parts(sc)
    return p6_prep.route_digest(parts), p6_prep.canon(parts[4])[:300]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "parent"
    if mode == "rank0":
        log, tok = sys.argv[2], sys.argv[3]
        d = parts_digest(log, tok)
        print("ZZD7DIAG rank0_digest " + json.dumps(d), flush=True)
        return
    t = first_twin(1)
    print("ZZD7DIAG twin " + json.dumps(t), flush=True)
    log, tok, aug = t
    import route_rank
    print("ZZD7DIAG apply " + route_rank.apply(1), flush=True)
    from nuplan.planning.script import driverl_runtime_map_features as M
    calls = {"n": 0}
    inner = M._choose_route_edge

    def counting(*a, **kw):
        calls["n"] += 1
        return inner(*a, **kw)
    M._choose_route_edge = counting
    d1 = parts_digest(log, tok)
    print("ZZD7DIAG rank1_inprocess_digest " + json.dumps(d1) + f" choose_route_edge_calls={calls['n']}", flush=True)
    with mp.get_context("spawn").Pool(1) as pool:
        print("ZZD7DIAG spawned_worker " + json.dumps(pool.map(worker_state, [0])[0]), flush=True)
    env = {k: v for k, v in os.environ.items() if k != "DRIVERL_EVAL_ROUTE_LANE_RANK"}
    out = subprocess.run([sys.executable, __file__, "rank0", log, tok], env=env, capture_output=True, text=True, timeout=900)
    print(out.stdout.strip()[-600:], flush=True)
    if out.returncode:
        print("ZZD7DIAG rank0_err " + out.stderr.strip()[-400:], flush=True)


if __name__ == "__main__":
    main()
