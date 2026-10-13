"""S0 gate for the NAVTEST label store (twin of raw/2026-10-12-aux-label-store/s0_gate.py --mode post; the same bars, the same verdict function). CPU only.

WHAT IS CHECKED (bars fixed in s0_gate.BARS, i.e. BEFORE this run, and not changed here)
  FILES     every shard re-hashed against the manifest, no extra shard, counts consistent with the headers
  COVERAGE  stored keys == navtest.json tokens exactly (12,146 / 136 logs); the manifest's store_sha256 recomputed from its own file map
  DISJOINT  key / token / log overlap with the TRAIN store = 0 (the train store is re-hashed and must be the pinned store_sha256)
  AB        path A (NuPlanScenario, or the devkit ego-state query for a frame the scenario builder refuses) vs the STORED label: 0 differing cells,
            mask AND signed distance, ONE random token in EVERY navtest log (136 samples >= the 64 the SPEC asks)
  ORIGIN    rear-axle pose of path A vs the stored pose: <= 1e-6 m, <= 1e-9 rad
  CORNER / VERDICT / TARGET   label vs navsim_dac's exact point-in-polygon, 13 lateral shifts of a 20-pose 5 Hz path, exactly as s0_gate.fidelity.
            The path is the LOGGED FUTURE EGO PATH from the DB (navtest has no teacher rank-0 bank; the train gate used the rank-0 target). Samples whose
            log has < 4.0 s of tail after the token are counted and skipped for these three criteria.
  ARMS      the six deliberate regressions of s0_gate (y flipped, x/y swapped, padding removed, clip 6 m, wrong map, one frame = 0.5 s offset): each must
            make this SAME verdict function FAIL.
Also reported (not criteria): the share of ALL stored keys whose rear-axle origin lies inside the label (an ego on the road), by map.

Run (driverl-eval venv; PYTHONIOENCODING=utf-8):  python s0_gate_navtest.py --out s0_gate_navtest.json
Last line: ZZGATE_PASS navtest | ZZGATE_FAIL navtest <why>
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
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
sys.path.insert(0, str(PKG / "raw" / "2026-10-12-aux-label-store"))
import s0_gate as G  # noqa: E402  (also puts refe/ on sys.path and sets the nuPlan env defaults)

ALS, ND, NS = G.ALS, G.ND, G.NS
ROOT = G.ROOT
ART = f"{ROOT}-artifacts/aux_drivable"
TRAIN_PINNED_SHA = "4a17db673b1548ba35724ab848a7dc551e4b0bc1e49c57ae93e6357454bf287a"


def gt_future_rel(db_path: str, token: str, pose, n_poses: int = 20, dt_s: float = 0.2):
    """The logged future ego path as 20 rear-axle poses [x, y, yaw] at 0.2 .. 4.0 s, in the rear-axle frame at t0 (the bank-traj convention). None when the
    log has less than 4.0 s after the token. Time-nearest ego pose of the DB (lidar_pc is 20 Hz)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT lower(hex(lp.token)), lp.timestamp, ep.x, ep.y, ep.qw, ep.qx, ep.qy, ep.qz FROM lidar_pc lp "
                           "JOIN ego_pose ep ON lp.ego_pose_token = ep.token ORDER BY lp.timestamp").fetchall()
    finally:
        con.close()
    ts = np.asarray([r[1] for r in rows], np.float64)
    pos = {r[0]: i for i, r in enumerate(rows)}
    i0 = pos[token]
    t_want = ts[i0] + 1e6 * dt_s * np.arange(1, n_poses + 1)
    if t_want[-1] > ts[-1] + 25_000:                                   # < half a lidar period of slack
        return None
    j = np.clip(np.searchsorted(ts, t_want), 1, len(ts) - 1)
    j = np.where(np.abs(ts[j - 1] - t_want) <= np.abs(ts[j] - t_want), j - 1, j)
    if np.abs(ts[j] - t_want).max() > 60_000:                          # a gap in the log
        return None
    x0, y0, h0 = pose
    c, s = math.cos(h0), math.sin(h0)
    out = np.zeros((n_poses, 3))
    for k, jj in enumerate(j):
        r = rows[jj]
        h = ALS.yaw_from_quaternion(*r[4:8])
        dx, dy = r[2] - x0, r[3] - y0
        out[k] = (dx * c + dy * s, -dx * s + dy * c, (h - h0 + math.pi) % (2 * math.pi) - math.pi)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=f"{ART}/label_store_navtest")
    ap.add_argument("--train-store", default=f"{ART}/label_store")
    ap.add_argument("--navtest", default=G.NAVTEST_JSON)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    store, tstore = Path(a.store), Path(a.train_store)
    man = json.load(open(store / "manifest.json", encoding="utf-8"))
    logmap = json.load(open(store / "inputs" / "log_map_navtest.json", encoding="utf-8"))["logs"]
    dbs = NS.index_dbs()
    # ---------------- FILES / COVERAGE
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
    sha_again = hashlib.sha256("\n".join(f"{k}:{v['sha256']}" for k, v in sorted(man["files"].items())).encode()).hexdigest()
    nt_toks, nt_logs = ALS.load_navtest(a.navtest)
    stored_tok = {t for d in keys.values() for t in d}
    files_ok = (not bad_files and not extra_files and man["counts"]["stored_keys"] == n_keys and man["counts"]["logs"] == len(man["files"])
                and sha_again == man["store_sha256"])
    coverage = dict(navtest_tokens=len(nt_toks), navtest_logs=len(nt_logs), stored_keys=n_keys, stored_logs=len(keys), tokens_missing=len(nt_toks - stored_tok),
                    tokens_not_in_navtest=len(stored_tok - nt_toks), logs_missing=len(nt_logs - set(keys)), logs_not_in_navtest=len(set(keys) - nt_logs))
    coverage_ok = (coverage["tokens_missing"] == 0 and coverage["tokens_not_in_navtest"] == 0 and coverage["logs_missing"] == 0 and coverage["logs_not_in_navtest"] == 0)
    # ---------------- DISJOINT from the (re-hashed, pinned) train store
    tman = json.load(open(tstore / "manifest.json", encoding="utf-8"))
    train_bad = [rel for rel, ent in tman["files"].items() if ALS.sha256_file(tstore / rel) != ent["sha256"]]
    tkeys = set()
    for rel in tman["files"]:
        head, _ = ALS.read_shard(tstore / rel, with_blobs=False)
        tkeys |= {(head["log"], e["t"]) for e in head["entries"]}
    nkeys = {(lg, t) for lg, d in keys.items() for t in d}
    disjoint = dict(train_store_sha256=tman["store_sha256"], train_store_is_pinned=tman["store_sha256"] == TRAIN_PINNED_SHA, train_shards_rehashed=len(tman["files"]),
                    train_shards_bad=len(train_bad), train_keys=len(tkeys), navtest_keys=len(nkeys), key_overlap=len(tkeys & nkeys),
                    token_overlap=len({t for _l, t in tkeys} & stored_tok), log_overlap=len({l for l, _t in tkeys} & set(keys)))
    disjoint_ok = (disjoint["train_store_is_pinned"] and disjoint["train_shards_bad"] == 0 and disjoint["key_overlap"] == 0
                   and disjoint["token_overlap"] == 0 and disjoint["log_overlap"] == 0)
    print(f"[navtest gate] {len(man['files'])} shards, {n_keys} keys; files_ok {files_ok} coverage_ok {coverage_ok} disjoint_ok {disjoint_ok} "
          f"(key overlap {disjoint['key_overlap']}, token {disjoint['token_overlap']}, log {disjoint['log_overlap']})", flush=True)
    # ---------------- QC over ALL stored keys: the rear-axle origin is inside the label (an ego on the road)
    qc = collections.defaultdict(lambda: [0, 0])
    ix0 = int(math.floor((0.0 - ALS.X0) / ALS.RES))
    iy0 = int(math.floor((0.0 - ALS.Y0) / ALS.RES))
    for lg, d in sorted(keys.items()):
        _h, blobs = ALS.read_shard(store / "shards" / f"{lg}.aux1")
        for t in d:
            sdf = ALS.decode_sdf(blobs[t])
            m_ = qc[man["files"][f"shards/{lg}.aux1"]["map"]]
            m_[0] += int(sdf[ix0, iy0] > 0)
            m_[1] += 1
    qc_out = {m_: dict(n=v[1], origin_inside_share=v[0] / v[1]) for m_, v in sorted(qc.items())}
    qc_all = sum(v[0] for v in qc.values()) / max(sum(v[1] for v in qc.values()), 1)
    print(f"[navtest gate] origin inside the label on {qc_all:.4f} of all keys; {json.dumps(qc_out)}", flush=True)
    # ---------------- AB / ORIGIN / fidelity: one random token per log
    rng = random.Random(a.seed)
    picks = [(lg, rng.choice(sorted(keys[lg]))) for lg in sorted(keys)]
    backend = ALS.NuplanBackend(os.environ["NUPLAN_MAPS_ROOT"])
    maps_cycle = {m_: ALS.MAP_NAMES[(i + 1) % len(ALS.MAP_NAMES)] for i, m_ in enumerate(ALS.MAP_NAMES)}
    wrong_logmap = {lg: {"map": maps_cycle[v["map"]]} for lg, v in logmap.items() if v.get("map")}
    samples, kinds, no_tail = [], collections.Counter(), 0
    for k, (lg, tok) in enumerate(picks):
        pa, pose_a, kind = G.path_a(dbs, lg, tok)
        kinds[kind] += 1
        stored = ALS.read_label(store, lg, tok)
        pose_s = tuple(keys[lg][tok]["p"])
        mask_a, sdf_a = ALS.build_label(pa, pose_a)
        dm = math.hypot(pose_a[0] - pose_s[0], pose_a[1] - pose_s[1])
        dr = abs((pose_a[2] - pose_s[2] + math.pi) % (2 * math.pi) - math.pi)
        traj = gt_future_rel(dbs[lg], tok, pose_s)
        no_tail += traj is None
        samples.append(dict(log=lg, token=tok, map=logmap[lg]["map"], path_a=kind, polys=pa, pose_s=list(pose_s), sdf_s=stored, sdf_a=sdf_a, mask_a=mask_a,
                            origin_dev_m=dm, origin_dev_rad=dr, traj=traj, drivable_frac=float((stored > 0).mean())))
        if (k + 1) % 20 == 0 or k + 1 == len(picks):
            print(f"[{k + 1}/{len(picks)}] path A kinds {dict(kinds)} no-tail {no_tail}  {time.time() - t0:.0f}s", flush=True)

    def metrics_for(variant=None):
        fids, abs_, orig = [], [], []
        for s in samples:
            sdf = s["sdf_s"] if variant is None else variant(s)
            abs_.append(int(((sdf > 0) != s["mask_a"]).sum() + (sdf != s["sdf_a"]).sum()))
            orig.append((s["origin_dev_m"], s["origin_dev_rad"]))
            if s["traj"] is not None:
                fids.append(G.fidelity(sdf, s["polys"], tuple(s["pose_s"]), s["traj"]))
        return G.aggregate(fids, abs_, orig, extra=dict(navtest_overlap=disjoint["key_overlap"] + disjoint["log_overlap"]))

    main_m = metrics_for()
    main_v = G.gate_verdict(main_m)
    orig_tg, orig_clip, orig_pad = ALS.to_global, ALS.SDF_CLIP_M, ALS.PAD_M

    def rebuild(s, pose=None, lm=None):
        return ALS.resolve_label(backend, lm or logmap, s["log"], pose or tuple(s["pose_s"]))[1]

    arms = {}
    try:
        ALS.to_global = lambda x, y, pose: orig_tg(x, -y, pose)
        arms["y_axis_flipped"] = metrics_for(lambda s: rebuild(s))
        ALS.to_global = lambda x, y, pose: orig_tg(y, x, pose)
        arms["xy_swapped"] = metrics_for(lambda s: rebuild(s))
        ALS.to_global = orig_tg
        ALS.PAD_M = 0.0
        arms["padding_removed"] = metrics_for(lambda s: rebuild(s))
        ALS.PAD_M = orig_pad
        ALS.SDF_CLIP_M = 6.0
        arms["clip_changed_to_6m"] = metrics_for(lambda s: rebuild(s))
        ALS.SDF_CLIP_M = orig_clip
        arms["wrong_map_for_log"] = metrics_for(lambda s: rebuild(s, lm=wrong_logmap))
        arms["label_offset_one_frame_0p5s"] = metrics_for(lambda s: rebuild(s, pose=G.db_pose_at_offset(dbs[s["log"]], s["token"], 10)))
    finally:
        ALS.to_global, ALS.SDF_CLIP_M, ALS.PAD_M = orig_tg, orig_clip, orig_pad
    arm_out = {}
    for name, m_ in arms.items():
        v = G.gate_verdict(m_)
        arm_out[name] = dict(gate_fails=not v["pass"], failed_criteria=v["failed"], ab_cells_differing=m_["ab_cells_differing"],
                             corner_agree_bilinear=m_["corner_agree_bilinear"], verdict_agree_bilinear=m_["verdict_agree_bilinear"],
                             target_inside_frac=m_["target_inside_frac"])
    arms_ok = all(v["gate_fails"] for v in arm_out.values())
    ok = bool(files_ok and coverage_ok and disjoint_ok and main_v["pass"] and arms_ok and main_m["samples"] >= 64)
    res = dict(mode="navtest", bars=G.BARS, seed=a.seed, store=str(store), store_sha256_in_manifest=man["store_sha256"], counts=man["counts"],
               files_ok=files_ok, bad_files=bad_files, extra_files=extra_files, sha_recomputed_equals_manifest=sha_again == man["store_sha256"],
               coverage=coverage, coverage_ok=coverage_ok, disjoint_vs_train=disjoint, disjoint_ok=disjoint_ok,
               qc_origin_inside_label=dict(by_map=qc_out, all=qc_all), n_logs_sampled=len(picks), path_a_kinds=dict(kinds), samples_without_4s_tail=no_tail,
               samples_by_map=dict(collections.Counter(s["map"] for s in samples)), metrics=main_m, verdict=main_v, arms=arm_out,
               all_arms_fail_the_gate=arms_ok, gate_pass=ok, seconds=time.time() - t0)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("files_ok", "coverage_ok", "disjoint_ok", "path_a_kinds", "samples_without_4s_tail", "metrics", "verdict", "arms",
                                           "all_arms_fail_the_gate", "gate_pass")}, indent=1, default=float))
    print("ZZGATE_PASS navtest" if ok else f"ZZGATE_FAIL navtest files={files_ok} coverage={coverage_ok} disjoint={disjoint_ok} main={main_v['failed']} "
          f"arms_not_failing={[k for k, v in arm_out.items() if not v['gate_fails']]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
