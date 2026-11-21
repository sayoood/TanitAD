"""PRIVGEOM-SAFE extraction (LABEL-FREE): per-proposal privileged geometry for S0's fit sets, from NAVSIM's OWN PDM objects.

READS NO LABEL: the S0 stores are opened with `fastneg_lib.read_members`, which REFUSES any member whose name contains 'label'.  What it computes per set is the landed labeller's call
(`navsim_pdm_targets.score_plans` on `executed_plans8` of the set's 64 proposals; the landed `onpolicy_relabel_pdm.label_log` sequence) -- the recomputed NAVSIM verdicts `rl6` are stored beside
the scalars and are compared with the stored labels ONLY by the gate, after the registration is sealed.  The ego footprint / simulated states are READ FROM THE SCORER (`PDMScorer._states`,
`_ego_coords`) after that very call, so the geometry is evaluated on exactly the states the verdict was computed on.

    python pg_extract.py cost --out <dir> [--n-sets 20]        # Step 1: seeded 20-set cost measurement
    python pg_extract.py work --subset <subset.json> --out <scalars dir>    # claim logs (O_EXCL locks) and extract; resumable; RAM-guarded
Environment: C:/Users/Admin/venvs/driverl-eval, CUDA_VISIBLE_DEVICES=-1, PYTHONPATH not needed (set below).
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys
import time
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "D:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
TREE = "C:/Users/Admin/p2heldout/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
NAVSIM = "D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
for _p in (os.path.normpath(os.path.join(HERE, "..", "ref")), HERE, TREE + "/refe", TREE + "/code", NAVSIM):
    sys.path.insert(0, _p)

import numpy as np                                                                       # noqa: E402
import psutil                                                                            # noqa: E402

import fastneg_lib as FL                                                                 # noqa: E402
import pg_geom as G                                                                      # noqa: E402

SETS_DIR = "C:/Users/Admin/srttc_s0/work/sets"
SEED = 20261019
MIN_START_GB, MIN_RUN_GB = 4.0, 3.0
SCHEMA = "privgeom-extract-v1"
DRIV_LAYERS = ("ROADBLOCK", "INTERSECTION", "DRIVABLE_AREA", "CARPARK_AREA")          # navsim pdm_scorer._calculate_ego_area `drivable_area_idcs`


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def avail_gb() -> float:
    return psutil.virtual_memory().available / 2 ** 30


def wait_ram(min_gb: float, tag: str = "") -> None:
    while avail_gb() < min_gb:
        log(f"[ram] available {avail_gb():.2f} GB < {min_gb} GB {tag}; pausing 30 s")
        time.sleep(30)


def peak_gb() -> float:
    mi = psutil.Process().memory_info()
    return float(getattr(mi, "peak_wset", mi.rss)) / 2 ** 30


def eligible_logs() -> dict:
    """{log: db path} for the S0 logs whose nuPlan DB is held locally (the landed `index_dbs`)."""
    import navtrain_scenarios as NS
    dbs = NS.index_dbs()
    out = {}
    for p in sorted(glob.glob(f"{SETS_DIR}/*.npz")):
        n = os.path.basename(p)[:-4]
        if n in dbs:
            out[n] = dbs[n]
    return out


def store_keys(name: str) -> dict:
    return FL.read_members(f"{SETS_DIR}/{name}.npz", ["token", "step", "rank", "ckpt_step"])


def store_traj(name: str):
    return FL.read_members(f"{SETS_DIR}/{name}.npz", ["traj3"])["traj3"]


class Engine:
    """One process' NAVSIM objects (landed `navsim_pdm_targets`, landed `navtrain_scenarios`)."""

    def __init__(self):
        import navsim_pdm_targets as PT
        import navtrain_scenarios as NS
        self.PT, self.NS = PT, NS
        PT._setup()
        self.sc = PT._S["scorer"]

    def scenarios(self, db: str, tokens: list):
        return {s.scenario_name: s for s in self.NS.build_scenarios_for_log(db, sorted(tokens), history_rows=40, future_rows=100)}

    def token_geometry(self, scenario) -> dict:
        t0 = time.time()
        parts = self.PT.metric_cache_parts(scenario)
        t_parts = time.time() - t0
        t0 = time.time()
        dam, obs = parts[5], parts[2]
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer as SL
        idx = dam.get_indices_of_map_type([getattr(SL, n) for n in DRIV_LAYERS])
        drv = G.Drivable([dam._geometries[i] for i in idx])
        slots = G.build_slots([(m.tokens, m._geometries) for m in obs._occupancy_maps], dt_slot=0.2, exclude=obs.red_light_token)
        return {"parts": parts, "drv": drv, "slots": slots, "t_parts": t_parts, "t_prep": time.time() - t0, "n_agents": int(sum(s.n for s in slots[:1])), "n_poly": drv.n_polys, "n_seg": drv.n_seg}

    def score_set(self, tg: dict, traj3: np.ndarray) -> dict:
        t0 = time.time()
        p8 = self.PT.executed_plans8(np.asarray(traj3, np.float64), repair=True)
        lab = self.PT.score_plans(tg["parts"], p8)
        t_score = time.time() - t0
        sc = self.sc
        states = np.asarray(sc._states[1:], np.float64)                                   # [64, 41, 11]; index 0 is PDM-Closed
        coords = np.asarray(sc._ego_coords[1:], np.float64)                               # [64, 41, 5, 2]
        t0 = time.time()
        g = G.all_scalars(coords, states, tg["drv"], tg["slots"])
        t_geom = time.time() - t0
        rl6 = np.stack([np.asarray(lab[k], np.float64) for k in ("nc", "dac", "ep", "ttc", "c", "ddc")], axis=-1)
        return {"g": g.astype(np.float32), "rl6": rl6.astype(np.float32), "t_score": t_score, "t_geom": t_geom}


def process_log(eng: Engine, name: str, db: str, only_rows: np.ndarray | None, out_dir: str | None) -> dict:
    """Extract every set of a log (or only `only_rows`).  Returns the arrays + a timing record."""
    k = store_keys(name)
    traj = store_traj(name)
    tok = k["token"].astype("U")
    rows = np.arange(len(tok)) if only_rows is None else np.asarray(only_rows)
    toks = sorted(set(tok[rows].tolist()))
    t0 = time.time()
    scs = eng.scenarios(db, toks)
    t_build = time.time() - t0
    n = len(rows)
    G_out = np.full((n, 64, G.K_DAC + G.K_NC), np.nan, np.float32)
    RL6 = np.full((n, 64, 6), np.nan, np.float32)
    ok = np.zeros(n, bool)
    why = [""] * n
    timing = {"log": name, "n_sets": int(n), "n_tokens": len(toks), "t_build_s": t_build, "tokens": [], "sets": []}
    by_tok: dict = {}
    for i, r in enumerate(rows):
        by_tok.setdefault(tok[r], []).append(i)
    for t, idxs in by_tok.items():
        wait_ram(MIN_RUN_GB, f"({name} {t})")
        sc = scs.get(t)
        if sc is None:
            for i in idxs:
                why[i] = "scenario guard (no 2 s history / 5 s future)"
            continue
        try:
            tg = eng.token_geometry(sc)
        except Exception as e:                                                          # noqa: BLE001
            for i in idxs:
                why[i] = f"token_geometry {type(e).__name__}: {str(e)[:100]}"
            continue
        timing["tokens"].append({"token": t, "n_sets": len(idxs), "t_parts_s": tg["t_parts"], "t_prep_s": tg["t_prep"], "n_poly": tg["n_poly"], "n_seg": tg["n_seg"]})
        for i in idxs:
            try:
                o = eng.score_set(tg, traj[rows[i]])
                if not np.isfinite(o["g"]).all():
                    raise FloatingPointError("non-finite scalar")
                G_out[i], RL6[i], ok[i] = o["g"], o["rl6"], True
                timing["sets"].append({"t_score_s": o["t_score"], "t_geom_s": o["t_geom"]})
            except Exception as e:                                                      # noqa: BLE001
                why[i] = f"score_set {type(e).__name__}: {str(e)[:100]}"
                if sum(1 for w in why if w) <= 3:
                    log(traceback.format_exc()[-500:])
    res = {"token": tok[rows], "step": k["step"][rows].astype(np.int64), "rank": k["rank"][rows].astype(np.int64), "ckpt_step": k["ckpt_step"][rows].astype(np.int64),
           "store_row": rows.astype(np.int64), "g": G_out, "rl6": RL6, "ok": ok}
    timing["n_ok"] = int(ok.sum())
    timing["fail"] = [{"i": int(i), "why": w} for i, w in enumerate(why) if w][:50]
    timing["n_fail"] = int(sum(1 for w in why if w))
    timing["peak_gb"] = peak_gb()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        tmp = f"{out_dir}/{name}.tmp.npz"
        np.savez(tmp, **res)
        os.replace(tmp, f"{out_dir}/{name}.npz")
        with open(f"{out_dir}/{name}.timing.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(timing, f)
    return {"res": res, "timing": timing}


# ---- Step 1: the seeded 20-set cost measurement --------------------------------------------------------------------------------------------------------------------------------------
def stage_cost(a) -> int:
    wait_ram(MIN_START_GB, "(start gate)")
    os.makedirs(a.out, exist_ok=True)
    elig = eligible_logs()
    pool = []
    for name in elig:
        k = store_keys(name)
        pool += [(name, int(j)) for j in range(len(k["token"]))]
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(len(pool), size=a.n_sets, replace=False).tolist())
    chosen = [pool[i] for i in pick]
    by_log: dict = {}
    for name, j in chosen:
        by_log.setdefault(name, []).append(j)
    log(f"cost: pool {len(pool)} sets in {len(elig)} logs; {len(chosen)} sets in {len(by_log)} logs; avail {avail_gb():.2f} GB")
    t_import = time.time()
    eng = Engine()
    t_setup = time.time() - t_import
    recs, t_all0 = [], time.time()
    for name, js in by_log.items():
        o = process_log(eng, name, elig[name], np.array(js), None)
        recs.append(o["timing"])
        log(f"  {name}: sets {o['timing']['n_sets']} tokens {o['timing']['n_tokens']} ok {o['timing']['n_ok']} build {o['timing']['t_build_s']:.1f}s peak {o['timing']['peak_gb']:.2f} GB")
    wall = time.time() - t_all0
    sets = [s for r in recs for s in r["sets"]]
    toks = [t for r in recs for t in r["tokens"]]
    n_prop = 64 * len(sets)
    rep = {"schema": SCHEMA, "created": now(), "seed": SEED, "n_pool_sets": len(pool), "n_pool_logs": len(elig), "n_sets_requested": a.n_sets, "n_sets_ok": len(sets), "n_logs": len(by_log),
           "n_tokens_with_geometry": len(toks), "chosen": [[n, j] for n, j in chosen], "t_engine_setup_s": t_setup, "wall_s": wall,
           "t_build_per_log_s": [r["t_build_s"] for r in recs], "t_parts_per_token_s": [t["t_parts_s"] for t in toks], "t_prep_per_token_s": [t["t_prep_s"] for t in toks],
           "t_score_per_set_s": [s["t_score_s"] for s in sets], "t_geom_per_set_s": [s["t_geom_s"] for s in sets], "peak_gb": max(r["peak_gb"] for r in recs),
           "avail_gb_end": avail_gb(), "fail": [f for r in recs for f in r["fail"]]}
    rep["s_per_proposal_score"] = float(np.mean(rep["t_score_per_set_s"])) / 64 if sets else float("nan")
    rep["s_per_proposal_geom"] = float(np.mean(rep["t_geom_per_set_s"])) / 64 if sets else float("nan")
    rep["s_per_proposal_total_amortised"] = wall / n_prop if n_prop else float("nan")
    with open(f"{a.out}/cost.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1)
    log(f"cost: {len(sets)} sets ok, {rep['s_per_proposal_score']*1e3:.2f} ms/prop score + {rep['s_per_proposal_geom']*1e3:.2f} ms/prop geom, amortised {rep['s_per_proposal_total_amortised']*1e3:.2f} ms/prop, peak {rep['peak_gb']:.2f} GB")
    return 0


# ---- Step 3: the detached worker -----------------------------------------------------------------------------------------------------------------------------------------------------
def stage_work(a) -> int:
    wait_ram(MIN_START_GB, "(start gate)")
    sub = json.load(open(a.subset, encoding="utf-8"))
    logs = sub["logs_in_order"]
    elig = eligible_logs()
    os.makedirs(a.out, exist_ok=True)
    pid = os.getpid()
    eng = Engine()
    done = 0
    for name in logs:
        if os.path.exists(f"{a.out}/{name}.npz"):
            continue
        lock = f"{a.out}/{name}.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{pid} {now()}".encode())
            os.close(fd)
        except FileExistsError:
            continue
        t0 = time.time()
        try:
            o = process_log(eng, name, elig[name], None, a.out)
            done += 1
            log(f"DONE {name}: sets {o['timing']['n_sets']} ok {o['timing']['n_ok']} fail {o['timing']['n_fail']} in {time.time()-t0:.0f}s, peak {o['timing']['peak_gb']:.2f} GB")
        except Exception:                                                              # noqa: BLE001
            log(f"FAILED {name}: {traceback.format_exc()[-600:]}")
            with open(f"{a.out}/{name}.failed", "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
        finally:
            try:
                os.remove(lock)
            except OSError:
                pass
    log(f"WORKER_DONE pid {pid} logs_done {done}")
    with open(f"{a.out}/WORKER_DONE_{pid}.txt", "w", encoding="utf-8") as f:
        f.write(f"{now()} logs_done {done}\n")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("cost", "work"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-sets", type=int, default=20)
    ap.add_argument("--subset", default=None)
    a = ap.parse_args(argv)
    return {"cost": stage_cost, "work": stage_work}[a.stage](a)


if __name__ == "__main__":
    raise SystemExit(main())
