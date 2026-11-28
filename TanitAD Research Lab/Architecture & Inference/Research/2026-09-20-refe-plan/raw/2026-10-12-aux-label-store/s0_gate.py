"""S0 gate for the auxiliary drivable-area label store (SPEC_AUX_DRIVABLE_HEAD.md sections 3.4 and 7). CPU only.

  --mode pre   BEFORE the full build: >= 64 real bank samples (stratified over the four maps, one per log) + DB-origin samples.
  --mode post  AFTER the build: re-hash every shard against the manifest, assert the keys, then re-run the check on random STORED keys.

Criteria (bars fixed BEFORE the first run; the SPEC's own where it has one):
  AB      path A (NuPlanScenario, or the devkit ego-state query for a frame the scenario builder refuses) vs path B (bank/DB origin
          + the log's map): 0 differing cells, mask AND signed distance        [SPEC: 0 of 20,480 x 20]
  ORIGIN  rear-axle pose of path A vs path B: <= 1e-6 m and <= 1e-9 rad          [SPEC: 0.0 m, 0.0 rad on 20]
  CORNER  footprint-corner verdicts of the label (bilinear SDF > 0) vs navsim_dac's exact point-in-polygon, 13 lateral shifts of
          each rank-0 target: >= 98.0 %                                         [SPEC measured 99.12 % bilinear / 99.03 % nearest]
  VERDICT whole-proposal verdict (any step with < 4 corners inside) agreement: >= 95 %   [SPEC G-LIVE alignment; measured 98.1 %]
  TARGET  >= 90 % of the UNSHIFTED rank-0 targets lie inside the label            [SPEC G-LIVE alignment; teacher leaves 4.97 %]
  LEAK    navtest tokens in the key universe = 0 and navtest logs in the key universe = 0   [SPEC G-LIVE no-leakage]
  ARMS    every deliberate regression must FAIL this same verdict function (y flipped, x/y swapped, padding removed, clip changed,
          wrong map for the log, label offset by one frame = 10 DB rows = 0.5 s)

Run (driverl-eval venv; export PYTHONIOENCODING=utf-8):
  python s0_gate.py --mode pre  --out s0_gate_pre.json
  python s0_gate.py --mode post --store D:/Projects/TanitAD-artifacts/aux_drivable/label_store --out s0_gate_post.json
Last line: ZZGATE_PASS <mode> | ZZGATE_FAIL <mode> <failed criteria>.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import random
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
REFE = PKG / "refe"
ROOT = "D:/Projects/TanitAD"
os.environ.setdefault("REFE_NUPLAN_DB_ROOT", f"{ROOT}/data/nuplan/nuplan-v1.1/splits")
os.environ.setdefault("NUPLAN_MAPS_ROOT", f"{ROOT}/data/nuplan-maps/nuplan-maps-v1.0")
sys.path.insert(0, str(REFE))
import aux_label_store as ALS  # noqa: E402
import navsim_dac as ND  # noqa: E402
import navtrain_scenarios as NS  # noqa: E402

BARS = dict(ab_cells_differing_max=0, origin_dev_m_max=1e-6, origin_dev_rad_max=1e-9, corner_agree_min=0.98,
            verdict_agree_min=0.95, target_inside_min=0.90, navtest_overlap_max=0)
SHIFTS = (-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
DEFAULT_BANK = f"{ROOT}-artifacts/_rlfan_work/bank_full/targets_rank0.jsonl"
NAVTEST_JSON = (f"{ROOT}/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-estimator-and-route-leak/raw/cluster_maps/navtest.json")


# ------------------------------------------------------------------ the verdict function (arms are judged by this too) ----
def gate_verdict(m: dict) -> dict:
    c = {
        "AB": m["ab_cells_differing"] <= BARS["ab_cells_differing_max"],
        "ORIGIN": m["origin_max_m"] <= BARS["origin_dev_m_max"] and m["origin_max_rad"] <= BARS["origin_dev_rad_max"],
        "CORNER": m["corner_agree_bilinear"] >= BARS["corner_agree_min"],
        "VERDICT": m["verdict_agree_bilinear"] >= BARS["verdict_agree_min"],
        "TARGET": m["target_inside_frac"] >= BARS["target_inside_min"],
        "LEAK": m.get("navtest_overlap", 0) <= BARS["navtest_overlap_max"],
    }
    return {"criteria": c, "pass": all(c.values()), "failed": [k for k, v in c.items() if not v]}


# ------------------------------------------------------------------ raster verdicts, as in the prototype ------------------
def raster_inside(sdf_i8, pts):
    """Per-point verdict from the stored label: nearest cell (sdf > 0) and bilinear SDF > 0. pts [..., 2] ego frame."""
    X0, Y0, RES, NX, NY = ALS.X0, ALS.Y0, ALS.RES, ALS.NX, ALS.NY
    mask = sdf_i8 > 0
    fx = (pts[..., 0] - X0) / RES - 0.5
    fy = (pts[..., 1] - Y0) / RES - 0.5
    in_ext = (fx >= 0) & (fx <= NX - 1) & (fy >= 0) & (fy <= NY - 1)
    ix = np.clip(np.rint(fx).astype(int), 0, NX - 1)
    iy = np.clip(np.rint(fy).astype(int), 0, NY - 1)
    nearest = mask[ix, iy]
    x0 = np.clip(np.floor(fx).astype(int), 0, NX - 2)
    y0 = np.clip(np.floor(fy).astype(int), 0, NY - 2)
    wx, wy = np.clip(fx - x0, 0, 1), np.clip(fy - y0, 0, 1)
    s = sdf_i8.astype(np.float64) * ALS.SDF_UNIT_M
    bil = ((1 - wx) * (1 - wy) * s[x0, y0] + wx * (1 - wy) * s[x0 + 1, y0] + (1 - wx) * wy * s[x0, y0 + 1] + wx * wy * s[x0 + 1, y0 + 1])
    return nearest, bil > 0, in_ext


def fidelity(sdf_i8, polys, pose, traj20):
    """13 shifted copies of the rank-0 target: label verdicts vs NAVSIM's exact verdict (navsim_dac.violations) on the same corners."""
    import shapely
    out = dict(pts=0, out_ext=0, near_dis=0, bil_dis=0, n=0, near_vdis=0, bil_vdis=0, exact_viol=0, bil_viol=0, tgt_exact_viol=None,
               tgt_bil_viol=None)
    for shift in SHIFTS:
        q = np.asarray(traj20, float).copy()
        q[:, 1] += shift
        p8 = ND.to_navsim(q)
        rel = ND.at_10hz(p8)
        ce = ND.corners_global(rel, 0.0, 0.0, 0.0)                      # ego frame [41, 4, 2]
        cg = ND.corners_global(rel, *pose)                              # map frame, the scorer's raw-mode maths
        exact_v = bool(ND.violations(polys, cg[None])[0])
        flat = cg.reshape(-1, 2)
        ins = np.zeros(len(flat), bool)
        for pg in polys:
            ins |= shapely.contains_xy(pg, flat[:, 0], flat[:, 1])
        ins = ins.reshape(cg.shape[:-1])
        near, bil, in_ext = raster_inside(sdf_i8, ce)
        out["pts"] += ins.size
        out["out_ext"] += int((~in_ext).sum())
        out["near_dis"] += int((near != ins).sum())
        out["bil_dis"] += int((bil != ins).sum())
        v_near = not bool((near.sum(-1) == 4).all())
        v_bil = not bool((bil.sum(-1) == 4).all())
        out["n"] += 1
        out["exact_viol"] += int(exact_v)
        out["bil_viol"] += int(v_bil)
        out["near_vdis"] += int(v_near != exact_v)
        out["bil_vdis"] += int(v_bil != exact_v)
        if shift == 0.0:
            out["tgt_exact_viol"], out["tgt_bil_viol"] = exact_v, v_bil
    return out


def aggregate(fids, ab, origin, extra=None):
    tot = collections.Counter()
    for f in fids:
        for k in ("pts", "out_ext", "near_dis", "bil_dis", "n", "near_vdis", "bil_vdis", "exact_viol", "bil_viol"):
            tot[k] += f[k]
    tg = [f for f in fids if f["tgt_bil_viol"] is not None]
    m = dict(samples=len(fids), proposals=tot["n"], corners=tot["pts"], corners_outside_extent=tot["out_ext"],
             corner_agree_nearest=1 - tot["near_dis"] / max(tot["pts"], 1), corner_agree_bilinear=1 - tot["bil_dis"] / max(tot["pts"], 1),
             verdict_agree_nearest=1 - tot["near_vdis"] / max(tot["n"], 1), verdict_agree_bilinear=1 - tot["bil_vdis"] / max(tot["n"], 1),
             exact_violation_rate=tot["exact_viol"] / max(tot["n"], 1), label_violation_rate=tot["bil_viol"] / max(tot["n"], 1),
             target_inside_frac=(sum(not f["tgt_bil_viol"] for f in tg) / len(tg)) if tg else float("nan"),
             target_inside_frac_exact=(sum(not f["tgt_exact_viol"] for f in tg) / len(tg)) if tg else float("nan"),
             ab_cells_differing=int(sum(ab)), ab_samples=len(ab), origin_max_m=max(origin, default=0.0)[0] if origin else 0.0,
             origin_max_rad=max((o[1] for o in origin), default=0.0))
    if origin:
        m["origin_max_m"] = max(o[0] for o in origin)
    m.update(extra or {})
    return m


# ------------------------------------------------------------------ inputs --------------------------------------------------
def fetch_rows(bank_path, keys):
    """Full bank rows (with the 20-pose traj) for a set of (log, token) keys -- one streaming pass over the ~350 MB file."""
    want, out = set(keys), {}
    with open(bank_path, encoding="utf-8") as f:
        for line in f:
            if '"token"' not in line:
                continue
            r = json.loads(line)
            k = (r["log_name"], r["token"])
            if k in want:
                out[k] = r
                if len(out) == len(want):
                    break
    return out


def db_pose_at_offset(db_path, token, k_rows):
    """Rear-axle pose k_rows later in lidar_pc time order (the 'label offset by one frame' regression: 10 rows = 0.5 s)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT lower(hex(lp.token)), ep.x, ep.y, ep.qw, ep.qx, ep.qy, ep.qz FROM lidar_pc lp "
                           "JOIN ego_pose ep ON lp.ego_pose_token = ep.token ORDER BY lp.timestamp").fetchall()
    finally:
        con.close()
    pos = {r[0]: i for i, r in enumerate(rows)}
    r = rows[min(pos[token] + k_rows, len(rows) - 1)]
    return (r[1], r[2], ALS.yaw_from_quaternion(*r[3:]))


def path_a(dbs, log, token, step=0):
    """Path A. NuPlanScenario (what onpolicy_label.py does); for a frame the scenario builder refuses (margin guard), the devkit's own
    ego-state query + the map the DB names. -> (polys, rear-axle pose, kind)."""
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    scs = list(NS.build_scenarios_for_log(dbs[log], [token]))
    if scs:
        sc = scs[0]
        ego = sc.get_ego_state_at_iteration(step)
        return ND.drivable_polygons(sc.map_api, ego), (ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading), "scenario"
    from nuplan.database.nuplan_db.nuplan_scenario_queries import get_ego_state_for_lidarpc_token_from_db
    ego = get_ego_state_for_lidarpc_token_from_db(dbs[log], token)
    api = get_maps_api(os.environ["NUPLAN_MAPS_ROOT"], NS.MAP_VERSION, ALS.db_map_name(dbs[log]))
    return ND.drivable_polygons(api, ego), (ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading), "devkit_ego_state"


def pick_samples(bank_idx, dbs, logmap, n, n_db, seed, yaml_toks, nt_toks):
    """n bank samples (one per log, stratified over the maps) + n_db samples whose origin comes from the DB (no bank row)."""
    rng = random.Random(seed)
    by_map = collections.defaultdict(list)
    for lg in sorted(bank_idx):
        if lg in dbs and logmap.get(lg, {}).get("map"):
            by_map[logmap[lg]["map"]].append(lg)
    maps = sorted(by_map)
    picks, per = [], [n // len(maps) + (1 if i < n % len(maps) else 0) for i in range(len(maps))]
    for m, k in zip(maps, per):
        logs = sorted(by_map[m])
        rng.shuffle(logs)
        for lg in logs[:k]:
            picks.append((lg, rng.choice(sorted(bank_idx[lg])), "bank"))
    extras = []
    if n_db:
        cand = []
        for lg in sorted(dbs):
            if lg in logmap and logmap[lg].get("map"):
                con = sqlite3.connect(f"file:{dbs[lg]}?mode=ro", uri=True)
                try:
                    toks = [r[0] for r in con.execute("SELECT lower(hex(token)) FROM lidar_pc")]
                finally:
                    con.close()
                free = [t for t in toks if t in yaml_toks and t not in nt_toks and t not in bank_idx.get(lg, {})]
                if free:
                    cand.append((lg, free))
        rng.shuffle(cand)
        extras = [(lg, rng.choice(free), "db") for lg, free in cand[:n_db]]
    return picks + extras


# ------------------------------------------------------------------ pre mode -----------------------------------------------
def run_pre(a) -> int:
    t0 = time.time()
    logmap = json.load(open(a.logmap, encoding="utf-8"))["logs"]
    dbs = NS.index_dbs()
    _yl, yaml_toks = ALS.load_yaml_tokens(a.yaml)
    nt_toks, nt_logs = ALS.load_navtest(a.navtest)
    bank = ALS.load_bank([a.bank], with_traj=False)
    bank_idx = {lg: {t: v[:3] for t, v in d.items()} for lg, d in bank.items()}
    # ---- LEAK over the whole key universe the builder will see: bank rows + DB-held yaml tokens
    uni_tokens, uni_logs = set(), set(bank)
    for lg, d in bank.items():
        uni_tokens |= set(d)
    db_extra = 0
    for lg, p in dbs.items():
        if lg in logmap:
            uni_logs.add(lg)
            for t in ALS._db_origins(p):
                if t in yaml_toks and t not in uni_tokens:
                    uni_tokens.add(t)
                    db_extra += 1
    leak = dict(universe_tokens=len(uni_tokens), universe_logs=len(uni_logs), navtest_tokens=len(nt_toks), navtest_logs=len(nt_logs),
                token_overlap=len(uni_tokens & nt_toks), log_overlap=len(uni_logs & nt_logs), yaml_vs_navtest_token_overlap=len(yaml_toks & nt_toks),
                bank_tokens=sum(len(d) for d in bank.values()), db_only_tokens=db_extra,
                universe_in_yaml=len(uni_tokens & yaml_toks), universe_not_in_yaml=len(uni_tokens - yaml_toks))
    print(f"[pre] universe {leak['universe_tokens']} tokens / {leak['universe_logs']} logs; navtest overlap tokens {leak['token_overlap']} "
          f"logs {leak['log_overlap']}", flush=True)
    picks = pick_samples(bank_idx, dbs, logmap, a.n, a.n_db, a.seed, yaml_toks, nt_toks)
    rows = fetch_rows(a.bank, [(lg, t) for lg, t, src in picks if src == "bank"])
    backend = ALS.NuplanBackend(os.environ["NUPLAN_MAPS_ROOT"])
    maps_cycle = {m: ALS.MAP_NAMES[(i + 1) % len(ALS.MAP_NAMES)] for i, m in enumerate(ALS.MAP_NAMES)}
    wrong_logmap = {lg: {"map": maps_cycle[v["map"]]} for lg, v in logmap.items() if v.get("map")}
    samples = []
    for k, (lg, tok, src) in enumerate(picks):
        ts = time.time()
        pa, pose_a, kind = path_a(dbs, lg, tok)
        if src == "bank":
            r = rows[(lg, tok)]
            pose_b = tuple(r["origin_world"])
            traj = np.asarray(r["traj"], float)
        else:
            pose_b = ALS._db_origins(dbs[lg])[tok]
            traj = None
        mask_a, sdf_a = ALS.build_label(pa, pose_a)
        t_b = time.time()
        mask_b, sdf_b = ALS.resolve_label(backend, logmap, lg, pose_b)
        t_b = time.time() - t_b
        ab = int((mask_a != mask_b).sum() + (sdf_a != sdf_b).sum())
        dm = math.hypot(pose_a[0] - pose_b[0], pose_a[1] - pose_b[1])
        dr = abs((pose_a[2] - pose_b[2] + math.pi) % (2 * math.pi) - math.pi)
        assert (mask_b == (sdf_b > 0)).all(), "mask != sign(sdf)"
        fid = fidelity(sdf_b, pa, pose_b, traj) if traj is not None else None
        samples.append(dict(i=k, log=lg, token=tok, map=logmap[lg]["map"], src=src, path_a=kind, ab_cells_differing=ab, origin_dev_m=dm,
                            origin_dev_rad=dr, drivable_frac=float(mask_b.mean()), n_polys=len(pa), t_label_b_s=t_b, fid=fid,
                            pose_b=list(pose_b), traj=None if traj is None else traj.tolist(), polys=pa, sdf_b=sdf_b, sdf_a=sdf_a))
        print(f"[{k + 1}/{len(picks)}] {lg[:26]} {logmap[lg]['map'][:10]:<10} {src:<4} A={kind:<16} A!=B cells {ab} dpose {dm:.2e} m "
              f"drv {mask_b.mean():.2f} B {t_b * 1e3:.0f} ms ({time.time() - ts:.1f}s)", flush=True)

    def metrics_for(sel, variant=None):
        """variant: callable(sample) -> sdf_i8 (the label under test). Default = the real path-B label."""
        fids, abs_, orig = [], [], []
        for s in sel:
            sdf = s["sdf_b"] if variant is None else variant(s)
            abs_.append(int(((sdf > 0) != (s["sdf_a"] > 0)).sum() + (sdf != s["sdf_a"]).sum()))
            orig.append((s["origin_dev_m"], s["origin_dev_rad"]))
            if s["traj"] is not None:
                fids.append(fidelity(sdf, s["polys"], tuple(s["pose_b"]), np.asarray(s["traj"])))
        return aggregate(fids, abs_, orig, extra=dict(navtest_overlap=leak["token_overlap"] + leak["log_overlap"]))

    main = metrics_for(samples)
    main_v = gate_verdict(main)
    # ---- the deliberate-regression arms: same samples, same verdict function; every one must FAIL
    orig_tg, orig_clip, orig_pad = ALS.to_global, ALS.SDF_CLIP_M, ALS.PAD_M

    def rebuild(s, pose=None, lm=None):
        return ALS.resolve_label(backend, lm or logmap, s["log"], pose or tuple(s["pose_b"]))[1]

    arms = {}
    try:
        ALS.to_global = lambda x, y, pose: orig_tg(x, -y, pose)
        arms["y_axis_flipped"] = metrics_for(samples, lambda s: rebuild(s))
        ALS.to_global = lambda x, y, pose: orig_tg(y, x, pose)
        arms["xy_swapped"] = metrics_for(samples, lambda s: rebuild(s))
        ALS.to_global = orig_tg
        ALS.PAD_M = 0.0
        arms["padding_removed"] = metrics_for(samples, lambda s: rebuild(s))
        ALS.PAD_M = orig_pad
        ALS.SDF_CLIP_M = 6.0
        arms["clip_changed_to_6m"] = metrics_for(samples, lambda s: rebuild(s))
        ALS.SDF_CLIP_M = orig_clip
        arms["wrong_map_for_log"] = metrics_for(samples, lambda s: rebuild(s, lm=wrong_logmap))
        arms["label_offset_one_frame_0p5s"] = metrics_for(
            samples, lambda s: rebuild(s, pose=db_pose_at_offset(dbs[s["log"]], s["token"], 10)))
    finally:
        ALS.to_global, ALS.SDF_CLIP_M, ALS.PAD_M = orig_tg, orig_clip, orig_pad
    arm_out = {}
    for name, m in arms.items():
        v = gate_verdict(m)
        arm_out[name] = dict(gate_fails=not v["pass"], failed_criteria=v["failed"], ab_cells_differing=m["ab_cells_differing"],
                             corner_agree_bilinear=m["corner_agree_bilinear"], verdict_agree_bilinear=m["verdict_agree_bilinear"],
                             target_inside_frac=m["target_inside_frac"])
    arms_ok = all(v["gate_fails"] for v in arm_out.values())
    ok = main_v["pass"] and arms_ok
    res = dict(mode="pre", bars=BARS, seed=a.seed, n_bank_samples=sum(s["src"] == "bank" for s in samples),
               n_db_origin_samples=sum(s["src"] == "db" for s in samples), samples_by_map=dict(collections.Counter(s["map"] for s in samples)),
               path_a_kinds=dict(collections.Counter(s["path_a"] for s in samples)), metrics=main, verdict=main_v, leak=leak, arms=arm_out,
               all_arms_fail_the_gate=arms_ok, gate_pass=ok, seconds=time.time() - t0,
               label_b_ms=dict(median=float(np.median([s["t_label_b_s"] for s in samples]) * 1e3), mean=float(np.mean([s["t_label_b_s"] for s in samples]) * 1e3)),
               samples=[{k: v for k, v in s.items() if k not in ("polys", "sdf_a", "sdf_b", "traj")} for s in samples])
    Path(a.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("metrics", "verdict", "leak", "arms", "all_arms_fail_the_gate", "gate_pass")}, indent=1, default=float))
    print("ZZGATE_PASS pre" if ok else f"ZZGATE_FAIL pre main={main_v['failed']} arms_not_failing={[k for k, v in arm_out.items() if not v['gate_fails']]}")
    return 0 if ok else 1


# ------------------------------------------------------------------ post mode ----------------------------------------------
def run_post(a) -> int:
    t0 = time.time()
    store = Path(a.store)
    man = json.load(open(store / "manifest.json", encoding="utf-8"))
    logmap = json.load(open(a.logmap, encoding="utf-8"))["logs"]
    dbs = NS.index_dbs()
    # ---- every shard re-hashed against the manifest, and the keys counted from the shard headers
    bad_files, keys = [], {}
    for rel, ent in man["files"].items():
        p = store / rel
        if not p.exists() or p.stat().st_size != ent["bytes"] or ALS.sha256_file(p) != ent["sha256"]:
            bad_files.append(rel)
            continue
        head, _ = ALS.read_shard(p, with_blobs=False)
        keys[head["log"]] = {e["t"]: e for e in head["entries"]}
    on_disk = {os.path.relpath(p, store).replace("\\", "/") for p in glob.glob(str(store / "shards" / "*.aux1"))}
    extra_files = sorted(on_disk - set(man["files"]))
    n_keys = sum(len(v) for v in keys.values())
    _yl, yaml_toks = ALS.load_yaml_tokens(a.yaml)
    nt_toks, nt_logs = ALS.load_navtest(a.navtest)
    stored = {t for d in keys.values() for t in d}
    leak = dict(stored_keys=n_keys, token_overlap=len(stored & nt_toks), log_overlap=len(set(keys) & nt_logs),
                stored_in_yaml=len(stored & yaml_toks), stored_not_in_yaml=len(stored - yaml_toks), yaml_tokens=len(yaml_toks),
                coverage_frac=len(stored & yaml_toks) / len(yaml_toks), missing_vs_yaml=len(yaml_toks - stored))
    bank_keys = {(lg, t) for lg, d in ALS.load_bank([a.bank], with_traj=False).items() for t in d}
    stored_keys = {(lg, t) for lg, d in keys.items() for t in d}
    leak.update(train_bank_rows=len(bank_keys), train_bank_coverage=len(bank_keys & stored_keys) / max(len(bank_keys), 1))
    coverage_ok = leak["train_bank_coverage"] >= 0.999                       # SPEC section 7: coverage of the training scenes >= 99.9 %
    manifest_ok = (not bad_files and not extra_files and man["counts"]["stored_keys"] == n_keys and man["counts"]["logs"] == len(man["files"]))
    print(f"[post] {len(man['files'])} shards, {n_keys} keys, bad files {len(bad_files)}, extra files {len(extra_files)}, "
          f"navtest overlap tokens {leak['token_overlap']} logs {leak['log_overlap']}, coverage {leak['coverage_frac']:.4f}", flush=True)
    # ---- QC over ALL stored keys: rank-0 target points inside the label, by where the map name came from
    by_src = {"db": [], "inferred": []}
    for lg, d in keys.items():
        src = logmap.get(lg, {}).get("src", "db")
        for e in d.values():
            if e.get("q") and e["q"][1] > 0:
                by_src.setdefault(src, []).append(e["q"][0] / e["q"][1])
    qc = {k: dict(n=len(v), mean=float(np.mean(v)) if v else None, share_ge_0p9=float(np.mean(np.asarray(v) >= 0.9)) if v else None,
                  share_zero=float(np.mean(np.asarray(v) == 0)) if v else None) for k, v in by_src.items()}
    # ---- random stored keys, two strata: logs whose DB we hold (path A possible) and logs whose map was inferred
    rng = random.Random(a.seed)
    db_keys = [(lg, t) for lg, d in sorted(keys.items()) if lg in dbs for t in sorted(d)]
    inf_keys = [(lg, t) for lg, d in sorted(keys.items()) if lg not in dbs for t in sorted(d)]
    picks = [("db", k) for k in rng.sample(db_keys, min(a.n, len(db_keys)))] + [("inferred", k) for k in rng.sample(inf_keys, min(a.n, len(inf_keys)))]
    rows = fetch_rows(a.bank, [k for _s, k in picks])
    backend = ALS.NuplanBackend(os.environ["NUPLAN_MAPS_ROOT"])
    out_rows, strata = [], {"db": dict(fids=[], ab=[], orig=[]), "inferred": dict(fids=[], ab=[], orig=[])}
    for k, (stratum, (lg, tok)) in enumerate(picks):
        stored_sdf = ALS.read_label(store, lg, tok)
        ent = keys[lg][tok]
        pose_s = tuple(ent["p"])
        if stratum == "db":
            pa, pose_a, kind = path_a(dbs, lg, tok)
            _m, sdf_a = ALS.build_label(pa, pose_a)
            ab = int((stored_sdf != sdf_a).sum())
            dm = math.hypot(pose_a[0] - pose_s[0], pose_a[1] - pose_s[1])
            dr = abs((pose_a[2] - pose_s[2] + math.pi) % (2 * math.pi) - math.pi)
            strata["db"]["orig"].append((dm, dr))
        else:                                                        # no DB: the exact polygons come from the inferred map, path B recomputed
            kind = "recompute_B"
            cx, cy = ALS.ego_centre(pose_s)
            pa = backend.polys(logmap[lg]["map"], cx, cy)
            ab = int((stored_sdf != ALS.build_label(pa, pose_s)[1]).sum())
            strata["inferred"]["orig"].append((0.0, 0.0))
        strata[stratum]["ab"].append(ab)
        r = rows.get((lg, tok))
        if r is not None:
            strata[stratum]["fids"].append(fidelity(stored_sdf, pa, pose_s, np.asarray(r["traj"], float)))
        out_rows.append(dict(stratum=stratum, log=lg, token=tok, src=ent["s"], path=kind, stored_vs_recomputed_cells_differing=ab, q=ent.get("q")))
    res_strata, ok = {}, manifest_ok and coverage_ok and leak["token_overlap"] == 0 and leak["log_overlap"] == 0
    for sname, d in strata.items():
        m = aggregate(d["fids"], d["ab"], d["orig"], extra=dict(navtest_overlap=leak["token_overlap"] + leak["log_overlap"]))
        v = gate_verdict(m)
        res_strata[sname] = dict(metrics=m, verdict=v)
        ok = ok and v["pass"] and m["samples"] > 0
    res = dict(mode="post", bars=BARS, seed=a.seed, store=str(store), manifest_ok=manifest_ok, coverage_ok=coverage_ok, bad_files=bad_files, extra_files=extra_files,
               store_sha256_in_manifest=man["store_sha256"], counts=man["counts"], leak_and_coverage=leak, qc_rank0_targets_inside_label_by_map_source=qc,
               strata=res_strata, n_checked=len(out_rows), checked=out_rows, gate_pass=bool(ok), seconds=time.time() - t0)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("manifest_ok", "coverage_ok", "leak_and_coverage", "qc_rank0_targets_inside_label_by_map_source", "strata", "gate_pass")}, indent=1, default=float))
    print("ZZGATE_PASS post" if ok else "ZZGATE_FAIL post")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pre", "post"), required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--n-db", type=int, default=16, help="pre: extra samples whose origin comes from the DB (no bank row)")
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--bank", default=DEFAULT_BANK)
    ap.add_argument("--yaml", default=str(PKG / "splits" / "navtrain.yaml"))
    ap.add_argument("--navtest", default=NAVTEST_JSON)
    ap.add_argument("--logmap", default=f"{ROOT}-artifacts/aux_drivable/log_map.json")
    ap.add_argument("--store", default=f"{ROOT}-artifacts/aux_drivable/label_store")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.exit(run_pre(a) if a.mode == "pre" else run_post(a))


if __name__ == "__main__":
    main()
