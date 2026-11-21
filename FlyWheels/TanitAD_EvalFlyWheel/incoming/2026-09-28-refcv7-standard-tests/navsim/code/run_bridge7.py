#!/usr/bin/env python3
"""Run refcv7 over NavSim scenes, one SEAM FILE per arm (TANITAD VENV). Adapted from the refcv6
suite's ``run_bridge6.py`` (same seam format, same resume/stand-in/refusal rules), for refcv7's
build (``refcv7_bridge.load_refcv7``), lift (0.25 m, the model's own bank) and derived arms.

    python code/run_bridge7.py --split warmup_two_stage --arms R7_A1,R7_A1_s1 \
        --inputs <export.json> --speed <speed_limits.json> --bank2 <416 stage-2 bank> \
        --road-plane <road_plane.json> --ckpt <ckpt.pt> --config <config.json> \
        --device cpu --out raw/.../bridge_warmup --derived

Per arm: ``rows_<arm>.jsonl`` (streamed, RESUMABLE), ``seam_<arm>.npz`` (E2's seam format, scored
by E2's seam agent unchanged) and ``seam_<arm>.manifest.json`` (the DECLARED-INPUT manifest).
``--derived`` also writes, from R7_A1's rows and with no extra forward:
  * ``R7_CEILDECL_d`` -- ⛔ DIAGNOSTIC (SPEC amendment A1): the speed ceiling applied to the
    EMITTED pick as SPEC_REFCV7 A2 declares it (the built model's E9 re-selection ignores it);
    rows whose replication check failed are REFUSED (the seam is then PARTIAL, never scored);
  * ``PRIOR_ha0p``   -- the prior path P alone, MODEL-FREE, for EVERY token of the split (it needs
    no pixel, so it has no stand-in), plus control KPR (model-emitted P vs model-free P).

⛔ CUDA only with ``--device cuda`` AND ``--gpu-lock-token`` naming the CURRENT holder of the
dev-box lock (``gpu_lock.py``), with no OTHER python compute app on the card -- re-checked every
200 rows; a lost lock stops the arm (resumable). ⛔ One device per arm (a mixture is refused).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import gpu_lock  # noqa: E402
import refcv7_bridge as R7  # noqa: E402

KPR_TOL_M = 1e-4


def times_rel_t0(rec: dict) -> list:
    ts = [int(x) for x in rec["timestamps_us"]]
    return [(t - ts[-1]) / 1e6 for t in ts]


def stage1_rigs(tokens: list, toks: dict, logs_root: str) -> dict:
    """``{token: rig_record}`` from each token's LOG pickle, one log in memory at a time."""
    by_log: dict = {}
    for t in tokens:
        by_log.setdefault(toks[t]["log_name"], []).append(t)
    out = {}
    for ln, tl in sorted(by_log.items()):
        fr = R7.F4.E2BF.load(os.path.join(logs_root, f"{ln}.pkl"))
        idx = {f["token"]: f for f in fr}
        for t in tl:
            out[t] = R7.F4.rig_record(idx[toks[t]["frame_tokens"][-1]]["cams"])
        del fr, idx
    return out


def read_rows(path: str) -> dict:
    recs = {}
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue                      # a torn last line from a kill: recomputed
            recs[r["token"]] = r
    return recs


def write_seam(out_dir: str, arm: str, order: list, recs: dict, toks: dict, man: dict,
               poses_key: str = "poses", knots_key: str = "knots") -> dict:
    """E2's seam format from the rows; a token without a row is MISSING (never a silent CV)."""
    tok_l, fps, srcs, poses_all, knots_all = [], [], [], [], []
    missing = [t for t in order if t not in recs]
    for t in order:
        if t not in recs:
            continue
        r = recs[t]
        tok_l.append(t)
        fps.append(toks[t]["fingerprint"])
        if r["source"] in ("refcv7", "prior_model_free"):
            srcs.append("precomputed")
            poses_all.append(np.asarray(r[poses_key], np.float32))
            knots_all.append(np.asarray(r[knots_key], np.float32))
        else:
            srcs.append("cv_standin")
            poses_all.append(np.full((8, 3), np.nan, np.float32))
            knots_all.append(np.full((8, 2), np.nan, np.float32))
    if not tok_l:
        return {"arm": arm, "n_rows": 0, "partial": True, "n_missing": len(missing)}
    np.savez(os.path.join(out_dir, f"seam_{arm}.npz"), token=np.asarray(tok_l),
             fingerprint=np.asarray(fps), source=np.asarray(srcs),
             poses=np.stack(poses_all), knots=np.stack(knots_all),
             sampling=np.asarray([8, 0.5]), arm=np.asarray(arm))
    n_model = sum(1 for s in srcs if s == "precomputed")
    man = dict(man)
    man.update({"arm": arm, "n_expected": len(order), "n_rows": len(tok_l),
                "missing": [f"#{order.index(t)}" for t in missing[:50]],
                "n_missing": len(missing), "partial": bool(missing),
                "n_model_rows": n_model, "n_cv_standin_rows": len(tok_l) - n_model})
    R7.json_dump(man, os.path.join(out_dir, f"seam_{arm}.manifest.json"))
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--speed", required=True)
    ap.add_argument("--speed-oracle", default="")
    ap.add_argument("--bank2", default="")
    ap.add_argument("--bank1", default="")
    ap.add_argument("--bank1-kind", choices=("v2", "navtest"), default="v2")
    ap.add_argument("--tokens-file", default="")
    ap.add_argument("--logs-root", default="C:/Users/Admin/navsim-crun/data/openscene/navsim_logs/test")
    ap.add_argument("--road-plane", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt-md5", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--exact-dedup", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--derived", action="store_true")
    ap.add_argument("--gpu-lock-token", default="")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    import torch
    torch.set_num_threads(int(a.threads))
    if a.device.startswith("cuda"):
        if not (a.gpu_lock_token and gpu_lock.held_by(a.gpu_lock_token)):
            sys.exit("⛔ --device cuda without holding the dev-box GPU lock (gpu_lock.py)")
        busy = gpu_lock.other_python_compute(own_pids=(os.getpid(),))
        if busy:
            sys.exit(f"⛔ other python compute app(s) on the card: {busy}")
    elif torch.cuda.is_available():
        sys.exit("⛔ --device cpu but CUDA is visible -- run with CUDA_VISIBLE_DEVICES=-1")
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    for arm in arms:
        if arm not in R7.ARMS7:
            sys.exit(f"unknown arm {arm!r}; known {sorted(R7.ARMS7)}")
    if a.derived and "R7_A1" not in arms:
        sys.exit("⛔ --derived reads R7_A1's rows: include R7_A1 in --arms")
    os.makedirs(a.out, exist_ok=True)
    doc = (json.load(gzip.open(a.inputs, "rt", encoding="utf-8")) if a.inputs.endswith(".gz")
           else json.load(open(a.inputs, encoding="utf-8")))
    toks = doc["tokens"]
    speed = json.load(open(a.speed, encoding="utf-8"))
    if speed["split"] != a.split:
        sys.exit(f"⛔ speed file split {speed['split']} != {a.split}")
    oracle = None
    if any(R7.ARMS7[x]["vmax"] == "oracle" for x in arms):
        if not a.speed_oracle:
            sys.exit("⛔ an ORACLE max-speed arm needs --speed-oracle")
        oracle = json.load(open(a.speed_oracle, encoding="utf-8"))
        if oracle["split"] != a.split:
            sys.exit(f"⛔ oracle file split {oracle['split']} != {a.split}")
    rp = json.load(open(a.road_plane, encoding="utf-8"))
    road_z = float(rp["summary"]["median"])
    if a.split == "navtest":
        for r in toks.values():
            r["stage"] = 1
    s1 = sorted(t for t, r in toks.items() if r["stage"] == 1)
    s2 = sorted(t for t, r in toks.items() if r["stage"] == 2)
    if a.tokens_file:
        keep = set(json.load(open(a.tokens_file, encoding="utf-8"))["tokens"])
        s1 = [t for t in s1 if t in keep]
        s2 = [t for t in s2 if t in keep]
        if not (s1 or s2):
            sys.exit("⛔ --tokens-file selects no token")
    if a.limit:
        s2 = s2[:a.limit]
        s1 = s1[:max(1, a.limit // 10)] if s2 else s1[:a.limit]
    order = s1 + s2
    bank2 = R7.Bank416(a.bank2) if a.bank2 else None
    bank1 = ((R7.BankNavtest416(a.bank1) if a.bank1_kind == "navtest" else R7.Bank416(a.bank1))
             if a.bank1 else None)
    ckpt_md5 = R7.R6.B2.md5_file(a.ckpt)
    if a.ckpt_md5 and ckpt_md5 != a.ckpt_md5:
        sys.exit(f"⛔ checkpoint md5 {ckpt_md5} != expected {a.ckpt_md5}")
    t_load = time.time()
    model, cfg, targs, rec, meta = R7.load_refcv7(a.ckpt, a.config, a.device, a.precision)
    if a.exact_dedup:
        R7.exact_dedup(model.core.encoder)
    steps = int(rec["decoder_steps"])
    bank = model._lift_bank_hires
    load_s = round(time.time() - t_load, 1)
    blind_grey = int(round((bank2 or bank1).mean_px)) if (bank2 or bank1) else 128
    need_rig = [t for t in s1 if bank1 is None or t not in bank1.prov or not hasattr(bank1, "rigs")]
    s1_rigs = stage1_rigs(need_rig, toks, a.logs_root) if need_rig else {}
    model_rec = {"ckpt": a.ckpt, "ckpt_md5": ckpt_md5, "step": rec["state_dict"]["step"],
                 "label": a.label, "registry": "MODEL_REGISTRY.md -- refcv7-r101-s0 (TRAINING)",
                 "decoder_steps": steps, "load_s": load_s, **meta, "torch": torch.__version__,
                 "threads": int(a.threads), "exact_dedup": bool(a.exact_dedup),
                 "lift": {"road_z_m": road_z, "road_plane_source": os.path.abspath(a.road_plane),
                          "bank": "model._lift_bank_hires (0.25 m, stride 8; parameters read off "
                                  "the built bank)"}}
    print(f"[bridge7] step={model_rec['step']} steps={steps} precision={meta['precision']} "
          f"device={a.device} load {load_s}s md5={ckpt_md5} label={a.label}", flush=True)
    lift_cache: dict = {}

    def lift_for(rig: dict):
        k = rig["rig_key"]
        if k not in lift_cache:
            lift_cache[k] = R7.hires_lift_for_rig(rig, road_z, bank)
        return lift_cache[k]

    base_man = {"split": a.split, "label": a.label, "model": model_rec,
                "bank2": a.bank2 or None, "bank1": a.bank1 or None,
                "inputs": os.path.abspath(a.inputs), "speed": os.path.abspath(a.speed),
                "speed_oracle": os.path.abspath(a.speed_oracle) if a.speed_oracle else None,
                "conversion": R7.knots_to_navsim.__doc__,
                "nav_map": {str(k): v for k, v in R7.R6.NAVSIM_CMD_TO_NAV_NAME.items()},
                "gpu_lock_token": a.gpu_lock_token or None}
    for arm in arms:
        spec = R7.ARMS7[arm]
        rows_path = os.path.join(a.out, f"rows_{arm}.jsonl")
        done = read_rows(rows_path)
        t_arm = time.time()
        n_new = 0
        with open(rows_path, "a", encoding="utf-8") as fh:
            for stage, tlist in ((1, s1), (2, s2)):
                for tok in tlist:
                    if tok in done:
                        continue
                    if a.device.startswith("cuda") and n_new % 200 == 199:
                        if not gpu_lock.held_by(a.gpu_lock_token):
                            sys.exit(f"⛔ {arm}: the GPU lock is no longer ours -- stopping "
                                     f"(resumable at {len(done)} rows)")
                    r = toks[tok]
                    row = {"token": tok, "stage": stage, "arm": arm}
                    times = times_rel_t0(r)
                    frames_mode = spec["frames"]
                    bk = bank2 if stage == 2 else bank1
                    key = r["scene_token"] if stage == 2 else tok
                    if a.split == "navtest" and (bk is None or key not in bk.prov):
                        sys.exit(f"⛔ a navtest token (#{order.index(tok)}) is not in the 416 bank "
                                 f"-- the bank is incomplete; refusing")
                    if frames_mode != "BLIND" and (bk is None or key not in bk.prov):
                        row.update({"source": "cv_standin",
                                    "why": ("no ORIGINAL camera frames for this stage-1 scene on "
                                            "this box -- the devkit ConstantVelocityAgent is the "
                                            "DECLARED stand-in")})
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
                        continue
                    decl = R7.declare7(r["ego_statuses"], arm)
                    nav = R7.nav_input(decl, arm)
                    vsrc = oracle if spec["vmax"] == "oracle" else speed
                    vmax = R7.max_speed_input(vsrc["tokens"].get(tok), arm)
                    hist = R7.ego_history_poses(decl, times)
                    src = R7.slot_sources7(times, frames_mode)
                    if bk is not None and key in bk.prov:
                        fr, rk, sha = bk.load(key)
                        rig = bk.rigs[rk] if hasattr(bk, "rigs") else s1_rigs[tok]
                        if rig["rig_key"] != rk:
                            sys.exit(f"⛔ #{order.index(tok)}: rig {rig['rig_key']} != bank rig {rk}")
                    else:
                        rig = s1_rigs[tok]
                        fr, sha = np.zeros((1, 416, 1024, 3), np.uint8), None
                    rows_u8 = R7.pack_rows(fr, src, blind_grey if frames_mode == "BLIND" else None)
                    g, v, lsumm = lift_for(rig)
                    seed = R7.scene_seed(spec["seed"], tok)
                    t1 = time.time()
                    res = R7.run_model7(model, rows_u8.numpy(), decl, nav, vmax, hist, g, v, steps,
                                        seed, a.device, filter_on=spec["filter"])
                    pmf = R7.prior_model_free(hist, res["v0"])
                    kpr = (None if res["prior_emitted_path"] is None else
                           float(np.abs(res["prior_emitted_path"] - pmf["path"]).max()))
                    poses = R7.knots_to_navsim(res["traj"])
                    row.update({"source": "refcv7", "scene_token": r["scene_token"],
                                "frame_sha16": sha, "frames": frames_mode, "slot_sources": src,
                                "blind_grey": blind_grey if frames_mode == "BLIND" else None,
                                "navsim_frame_times_s": [round(x, 4) for x in times],
                                "declared_values": decl, "v0": res["v0"], "nav": nav,
                                "vmax": vmax, "ceiling_expected_ms": R7.ceiling_ms(vmax),
                                "ego_history_poses": hist.tolist(), "rig_key": rig["rig_key"],
                                "lift": lsumm, "seed": seed, "knots": res["traj"].tolist(),
                                "poses": poses.tolist(), "diag": res["diag"],
                                "prior_model_free": {"a0": pmf["a0"], "kappa0": pmf["kappa0"],
                                                     "path": pmf["path"].tolist()},
                                "prior_emitted_ctrl": res["prior_emitted_ctrl"],
                                "KPR_max_abs_m": kpr,
                                "s": round(time.time() - t1, 3), "device": a.device,
                                "precision": meta["precision"]})
                    if "traj_ceiling_declared" in res:
                        row["knots_ceiling_declared"] = res["traj_ceiling_declared"].tolist()
                        row["poses_ceiling_declared"] = R7.knots_to_navsim(
                            res["traj_ceiling_declared"]).tolist()
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    n_new += 1
                    done[tok] = row
                    if n_new % 25 == 0:
                        el = time.time() - t_arm
                        print(f"  [{arm}] {len(done)}/{len(order)} ({n_new} new) "
                              f"{el / n_new:.2f} s/scene", flush=True)
        recs = read_rows(rows_path)
        devs = {recs[t].get("device") for t in order if t in recs and recs[t]["source"] == "refcv7"}
        precs = {recs[t].get("precision") for t in order if t in recs
                 and recs[t]["source"] == "refcv7"}
        if len(devs) > 1 or len(precs) > 1:
            sys.exit(f"⛔ {arm}: mixed devices {devs} / precisions {precs} in one arm -- refused")
        withheld = [f"{f}[{k}]" for f in R7.FIELDS for k in range(4)
                    if f"{f}[{k}]" not in R7.declared(arm)]
        kprs = [recs[t]["KPR_max_abs_m"] for t in order if t in recs
                and recs[t]["source"] == "refcv7" and recs[t].get("KPR_max_abs_m") is not None]
        man = dict(base_man, spec=spec, declared_inputs=list(R7.declared(arm)),
                   withheld_egostatus_fields=withheld, device=sorted(devs),
                   precision=sorted(precs), rows_jsonl=os.path.abspath(rows_path),
                   seconds_this_launch=round(time.time() - t_arm, 1),
                   KPR={"n": len(kprs), "max_abs_m": max(kprs) if kprs else None,
                        "tol_m": KPR_TOL_M,
                        "verdict": ("PASS" if kprs and max(kprs) <= KPR_TOL_M else
                                    ("NO_ROWS" if not kprs else "FAIL"))})
        m = write_seam(a.out, arm, order, recs, toks, man)
        print(f"[bridge7] {arm}: {m.get('n_model_rows')} refcv7 rows, "
              f"{m.get('n_cv_standin_rows')} stand-ins, missing {m.get('n_missing')}, "
              f"KPR {man['KPR']['verdict']} max {man['KPR']['max_abs_m']}", flush=True)
    if a.derived:
        recs = read_rows(os.path.join(a.out, "rows_R7_A1.jsonl"))
        # R7_CEILDECL_d (DIAGNOSTIC, SPEC A1): the ceiling applied to the EMITTED pick
        dr = {}
        n_bad = 0
        for t in order:
            r = recs.get(t)
            if r is None:
                continue
            if r["source"] != "refcv7":
                dr[t] = r
            elif r["diag"].get("derived_ok") and "poses_ceiling_declared" in r:
                dr[t] = {"token": t, "source": "refcv7", "poses": r["poses_ceiling_declared"],
                         "knots": r["knots_ceiling_declared"]}
            else:
                n_bad += 1
        own = [t for t in order if t in recs and recs[t]["source"] == "refcv7"]
        a1_devs = sorted({recs[t].get("device") for t in order if t in recs
                          and recs[t]["source"] == "refcv7"})
        a1_precs = sorted({recs[t].get("precision") for t in order if t in recs
                           and recs[t]["source"] == "refcv7"})
        man = dict(base_man, spec=R7.DERIVED_ARMS["R7_CEILDECL_d"], device=a1_devs,
                   precision=a1_precs,
                   derived_from="rows_R7_A1.jsonl", n_replication_failures=n_bad,
                   n_decl_changed_pick=sum(1 for t in own if recs[t]["diag"].get("decl_changed_pick")),
                   n_emitted_over_ceiling=sum(1 for t in own
                                              if recs[t]["diag"].get("emitted_over_ceiling")),
                   n_ceiling_binds=sum(1 for t in own
                                       if (recs[t]["diag"].get("n_candidates_over_ceiling") or 0) > 0))
        m = write_seam(a.out, "R7_CEILDECL_d", order, dr, toks, man)
        print(f"[bridge7] R7_CEILDECL_d: {m.get('n_model_rows')} rows, replication failures "
              f"{n_bad}, ceiling binds {man['n_ceiling_binds']}, emitted plan over the ceiling "
              f"{man['n_emitted_over_ceiling']}, declared ceiling changes the pick "
              f"{man['n_decl_changed_pick']}", flush=True)
        # PRIOR_ha0p: model-free, EVERY token (no pixel, so no stand-in)
        pr = {}
        for t in order:
            r = toks[t]
            decl = R7.declare7(r["ego_statuses"], "R7_A1")
            hist = R7.ego_history_poses(decl, times_rel_t0(r))
            pmf = R7.prior_model_free(hist, R7.v0_of(decl))
            pr[t] = {"token": t, "source": "prior_model_free", "knots": pmf["path"].tolist(),
                     "poses": R7.knots_to_navsim(pmf["path"]).tolist(),
                     "a0": pmf["a0"], "kappa0": pmf["kappa0"]}
        with open(os.path.join(a.out, "rows_PRIOR_ha0p.jsonl"), "w", encoding="utf-8") as fh:
            for t in order:
                fh.write(json.dumps(pr[t]) + "\n")
        man = dict(base_man, spec=R7.DERIVED_ARMS["PRIOR_ha0p"], device=["model-free"],
                   precision=["fp32 (kinematic_prior, CPU)"],
                   note=("model-free: kinematic_prior.prior_controls('ha0_ext_pose') + prior_path "
                         "on the SAME declared window R7_A1 reads; knots -> NavSim poses through "
                         "the same knots_to_navsim spline"))
        m = write_seam(a.out, "PRIOR_ha0p", order, pr, toks, man)
        print(f"[bridge7] PRIOR_ha0p: {m.get('n_model_rows')} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
