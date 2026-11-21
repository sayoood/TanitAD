#!/usr/bin/env python3
"""Run refcv8 over NavSim scenes, one SEAM FILE per arm (TANITAD VENV).  Port of ``run_bridge7.py`` (same seam format, same
resume / stand-in / refusal rules; its ``read_rows`` / ``stage1_rigs`` / ``times_rel_t0`` are imported unchanged).

    python code/run_bridge8.py --split navtest --arms R8_LEGAL,R8_LEGAL_s1 --inputs <navtest_inputs.json.gz> \
        --bank1 <416 navtest frame_bank> --bank1-kind navtest --logs-root <navsim_logs/test> --road-plane <road_plane.json> \
        --ckpt <ckpt_ema_NNNNN.pt> --config <config.json> --device cpu --out <dir> --derived \
        [--retime --retime-table <gains.json> --retime-stage-file <retime_stage.py>] [--tokens-file <subset.json>] [--limit N]

Per arm: ``rows_<arm>.jsonl`` (streamed, RESUMABLE), ``seam_<arm>.npz`` (E2's seam format, scored by the seam agent unchanged) and
``seam_<arm>.manifest.json`` (the DECLARED-INPUT manifest: arm spec, declared fields, token / r8_nav / speed feed, premises).
``--derived`` also writes, with no extra forward: ``PRIOR_ha0p`` (the prior alone, model-free, EVERY token) and control KPR.
``--retime`` (needs R8_LEGAL in ``--arms`` and an explicit ``--retime-table``) writes ``R8_RT`` from R8_LEGAL's rows.

CUDA only with ``--device cuda`` AND ``--gpu-lock-token`` naming the CURRENT holder of the dev-box lock, with no OTHER python
compute app on the card -- re-checked every 200 rows.  One device per arm (a mixture is refused).
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

import refcv8_bridge as B  # noqa: E402
import run_bridge7 as RB7  # noqa: E402  (generic row/rig helpers, unchanged)
import gpu_lock  # noqa: E402

KPR_TOL_M = 1e-4
read_rows = RB7.read_rows
times_rel_t0 = RB7.times_rel_t0
stage1_rigs = RB7.stage1_rigs
_MODEL_SOURCES = ("refcv8", "prior_model_free")


def write_seam(out_dir: str, arm: str, order: list, recs: dict, toks: dict, man: dict) -> dict:
    """E2's seam format from the rows; a token without a row is MISSING (never a silent CV)."""
    tok_l, fps, srcs, poses_all, knots_all = [], [], [], [], []
    missing = [t for t in order if t not in recs]
    for t in order:
        if t not in recs:
            continue
        r = recs[t]
        tok_l.append(t)
        fps.append(toks[t]["fingerprint"])
        if r["source"] in _MODEL_SOURCES:
            srcs.append("precomputed")
            poses_all.append(np.asarray(r["poses"], np.float32))
            knots_all.append(np.asarray(r["knots"], np.float32))
        else:
            srcs.append("cv_standin")
            poses_all.append(np.full((8, 3), np.nan, np.float32))
            knots_all.append(np.full((8, 2), np.nan, np.float32))
    if not tok_l:
        return {"arm": arm, "n_rows": 0, "partial": True, "n_missing": len(missing)}
    np.savez(os.path.join(out_dir, f"seam_{arm}.npz"), token=np.asarray(tok_l), fingerprint=np.asarray(fps),
             source=np.asarray(srcs), poses=np.stack(poses_all), knots=np.stack(knots_all),
             sampling=np.asarray([8, 0.5]), arm=np.asarray(arm))
    n_model = sum(1 for s in srcs if s == "precomputed")
    man = dict(man)
    man.update({"arm": arm, "n_expected": len(order), "n_rows": len(tok_l),
                "missing": [f"#{order.index(t)}" for t in missing[:50]], "n_missing": len(missing),
                "partial": bool(missing), "n_model_rows": n_model, "n_cv_standin_rows": len(tok_l) - n_model})
    B.json_dump(man, os.path.join(out_dir, f"seam_{arm}.manifest.json"))
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--speed", default="", help="map speed limits json(.gz); needed ONLY by R8_PRIV")
    ap.add_argument("--bank2", default="")
    ap.add_argument("--bank1", default="")
    ap.add_argument("--bank1-kind", choices=("v2", "navtest"), default="v2")
    ap.add_argument("--tokens-file", default="")
    ap.add_argument("--logs-root", default="D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test")
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
    ap.add_argument("--retime", action="store_true", help="derive R8_RT from R8_LEGAL's rows (needs --retime-table)")
    ap.add_argument("--retime-table", default="")
    ap.add_argument("--retime-table-sha256", default="")
    ap.add_argument("--retime-stage-file", default="")
    ap.add_argument("--gpu-lock-token", default="")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if B.MUTATION:
        sys.exit("[bridge8] R8_MUTATION is set -- a mutation-test environment never runs a checkpoint")
    import torch
    torch.set_num_threads(int(a.threads))
    if a.device.startswith("cuda"):
        if not (a.gpu_lock_token and gpu_lock.held_by(a.gpu_lock_token)):
            sys.exit("[bridge8] --device cuda without holding the dev-box GPU lock (gpu_lock.py)")
        busy = gpu_lock.other_python_compute(own_pids=(os.getpid(),))
        if busy:
            sys.exit(f"[bridge8] other python compute app(s) on the card: {busy}")
    elif torch.cuda.is_available():
        sys.exit("[bridge8] --device cpu but CUDA is visible -- run with CUDA_VISIBLE_DEVICES=-1")
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    for arm in arms:
        if arm not in B.ARMS8:
            sys.exit(f"unknown arm {arm!r}; known {sorted(B.ARMS8)}")
    if a.retime and "R8_LEGAL" not in arms:
        sys.exit("[bridge8] --retime derives R8_RT from R8_LEGAL's rows: include R8_LEGAL in --arms")
    if a.retime and not a.retime_table:
        sys.exit("[bridge8] --retime needs an explicit --retime-table (there is no default table)")
    need_speed = any(B.ARMS8[x]["speed"] == "map" for x in arms)
    if need_speed and not a.speed:
        sys.exit("[bridge8] an R8_PRIV arm needs --speed")
    os.makedirs(a.out, exist_ok=True)
    opener = gzip.open if a.inputs.endswith(".gz") else open
    doc = json.load(opener(a.inputs, "rt", encoding="utf-8"))
    toks = doc["tokens"]
    speed = None
    if a.speed:
        sp_open = gzip.open if a.speed.endswith(".gz") else open
        speed = json.load(sp_open(a.speed, "rt", encoding="utf-8"))
        if speed["split"] != a.split:
            sys.exit(f"[bridge8] speed file split {speed['split']} != {a.split}")
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
            sys.exit("[bridge8] --tokens-file selects no token")
    if a.limit:
        s2 = s2[:a.limit]
        s1 = s1[:max(1, a.limit // 10)] if s2 else s1[:a.limit]
    order = s1 + s2
    bank2 = B.Bank416(a.bank2) if a.bank2 else None
    bank1 = ((B.BankNavtest416(a.bank1) if a.bank1_kind == "navtest" else B.Bank416(a.bank1)) if a.bank1 else None)
    ckpt_md5 = B.R6.B2.md5_file(a.ckpt)
    if a.ckpt_md5 and ckpt_md5 != a.ckpt_md5:
        sys.exit(f"[bridge8] checkpoint md5 {ckpt_md5} != expected {a.ckpt_md5}")
    RS = table = rt_rec = None
    if a.retime:
        RS, table, rt_rec = B.load_retime(a.retime_stage_file, a.retime_table, a.retime_table_sha256 or None)
    t_load = time.time()
    model, cfg, targs, rec, meta = B.load_refcv8(a.ckpt, a.config, a.device, a.precision)
    if a.exact_dedup:
        B.exact_dedup(model.core.encoder)
    steps = int(rec["decoder_steps"])
    bank = model._lift_bank_hires
    load_s = round(time.time() - t_load, 1)
    blind_grey = int(round((bank2 or bank1).mean_px)) if (bank2 or bank1) else 128
    need_rig = [t for t in s1 if bank1 is None or t not in bank1.prov or not hasattr(bank1, "rigs")]
    s1_rigs = stage1_rigs(need_rig, toks, a.logs_root) if need_rig else {}
    model_rec = {"ckpt": a.ckpt, "ckpt_md5": ckpt_md5, "step": rec["state_dict"]["step"], "label": a.label,
                 "registry": "MODEL_REGISTRY.md -- refcv8-r101-s0", "decoder_steps": steps, "load_s": load_s, **meta,
                 "torch": torch.__version__, "threads": int(a.threads), "exact_dedup": bool(a.exact_dedup),
                 "lift": {"road_z_m": road_z, "road_plane_source": os.path.abspath(a.road_plane),
                          "bank": "model._lift_bank_hires (0.25 m, stride 8; parameters read off the built bank)"}}
    print(f"[bridge8] step={model_rec['step']} steps={steps} precision={meta['precision']} device={a.device} "
          f"load {load_s}s md5={ckpt_md5} label={a.label}", flush=True)
    lift_cache: dict = {}

    def lift_for(rig: dict):
        k = rig["rig_key"]
        if k not in lift_cache:
            lift_cache[k] = B.hires_lift_for_rig(rig, road_z, bank)
        return lift_cache[k]

    nav_row = B.r8_nav_row()
    base_man = {"split": a.split, "label": a.label, "model": model_rec, "bank2": a.bank2 or None, "bank1": a.bank1 or None,
                "inputs": os.path.abspath(a.inputs), "speed": os.path.abspath(a.speed) if a.speed else None,
                "conversion": B.knots_to_navsim.__doc__,
                "nav_map": {"navsim_onehot_key": B.NAVSIM_ONEHOT_KEY, "navsim_to_token": B.NAVSIM_LEGAL_NAV_CMD,
                            "token_names": B.NAV_NAMES8},
                "r8_nav_row": nav_row[0].tolist(), "r8_rc": None, "forward_kwargs": list(B.FORWARD_KWARGS),
                "gpu_lock_token": a.gpu_lock_token or None, "retime": rt_rec}
    for arm in arms:
        spec = B.ARMS8[arm]
        rows_path = os.path.join(a.out, f"rows_{arm}.jsonl")
        done = read_rows(rows_path)
        t_arm = time.time()
        n_new = 0
        with open(rows_path, "a", encoding="utf-8") as fh:
            for stage, tlist in ((1, s1), (2, s2)):
                for tok in tlist:
                    if tok in done:
                        continue
                    if a.device.startswith("cuda") and n_new % 200 == 199 and not gpu_lock.held_by(a.gpu_lock_token):
                        sys.exit(f"[bridge8] {arm}: the GPU lock is no longer ours -- stopping (resumable at {len(done)} rows)")
                    r = toks[tok]
                    row = {"token": tok, "stage": stage, "arm": arm}
                    times = times_rel_t0(r)
                    frames_mode = spec["frames"]
                    bk = bank2 if stage == 2 else bank1
                    key = r["scene_token"] if stage == 2 else tok
                    if a.split == "navtest" and (bk is None or key not in bk.prov):
                        sys.exit(f"[bridge8] a navtest token (#{order.index(tok)}) is not in the 416 bank -- refusing")
                    if frames_mode != "BLIND" and (bk is None or key not in bk.prov):
                        row.update({"source": "cv_standin", "why": "no ORIGINAL camera frames for this stage-1 scene on this "
                                    "box -- the devkit ConstantVelocityAgent is the DECLARED stand-in"})
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
                        continue
                    decl = B.declare8(r["ego_statuses"], arm)
                    nav = B.nav_input8(decl, arm)
                    vmax = B.speed_input8(speed["tokens"].get(tok) if speed else None, arm)
                    hist = B.ego_history_poses(decl, times)
                    src = B.slot_sources8(times, frames_mode)
                    if bk is not None and key in bk.prov:
                        fr, rk, sha = bk.load(key)
                        rig = bk.rigs[rk] if hasattr(bk, "rigs") else s1_rigs[tok]
                        if rig["rig_key"] != rk:
                            sys.exit(f"[bridge8] #{order.index(tok)}: rig {rig['rig_key']} != bank rig {rk}")
                    else:
                        rig = s1_rigs[tok]
                        fr, sha = np.zeros((1, 416, 1024, 3), np.uint8), None
                    rows_u8 = B.pack_rows(fr, src, blind_grey if frames_mode == "BLIND" else None)
                    g, v, lsumm = lift_for(rig)
                    seed = B.scene_seed(spec["seed"], tok)
                    t1 = time.time()
                    res = B.run_model8(model, rows_u8.numpy(), decl, nav, vmax, hist, g, v, steps, seed, a.device,
                                       r8_nav=nav_row)
                    pmf = B.prior_model_free(hist, res["v0"])
                    kpr = (None if res["prior_emitted_path"] is None else
                           float(np.abs(res["prior_emitted_path"] - pmf["path"]).max()))
                    poses = B.knots_to_navsim(res["traj"])
                    row.update({"source": "refcv8", "scene_token": r["scene_token"], "frame_sha16": sha, "frames": frames_mode,
                                "slot_sources": src, "blind_grey": blind_grey if frames_mode == "BLIND" else None,
                                "navsim_frame_times_s": [round(x, 4) for x in times], "declared_values": decl,
                                "v0": res["v0"], "nav": nav, "r8_nav": nav_row[0].tolist(), "vmax": vmax,
                                "ceiling_expected_ms": B.ceiling_ms(vmax), "ego_history_poses": hist.tolist(),
                                "rig_key": rig["rig_key"], "lift": lsumm, "seed": seed, "knots": res["traj"].tolist(),
                                "poses": poses.tolist(), "diag": res["diag"],
                                "prior_emitted_path": (None if res["prior_emitted_path"] is None
                                                       else res["prior_emitted_path"].tolist()),
                                "prior_model_free": {"a0": pmf["a0"], "kappa0": pmf["kappa0"], "path": pmf["path"].tolist()},
                                "prior_emitted_ctrl": res["prior_emitted_ctrl"], "KPR_max_abs_m": kpr,
                                "s": round(time.time() - t1, 3), "device": a.device, "precision": meta["precision"]})
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    n_new += 1
                    done[tok] = row
                    if n_new % 25 == 0:
                        el = time.time() - t_arm
                        print(f"  [{arm}] {len(done)}/{len(order)} ({n_new} new) {el / n_new:.2f} s/scene", flush=True)
        recs = read_rows(rows_path)
        mod = [t for t in order if t in recs and recs[t]["source"] == "refcv8"]
        devs = {recs[t].get("device") for t in mod}
        precs = {recs[t].get("precision") for t in mod}
        if len(devs) > 1 or len(precs) > 1:
            sys.exit(f"[bridge8] {arm}: mixed devices {devs} / precisions {precs} in one arm -- refused")
        withheld = [f"{f}[{k}]" for f in B.FIELDS for k in range(4) if f"{f}[{k}]" not in B.declared(arm)]
        kprs = [recs[t]["KPR_max_abs_m"] for t in mod if recs[t].get("KPR_max_abs_m") is not None]
        man = dict(base_man, spec=spec, declared_inputs=list(B.declared(arm)), withheld_egostatus_fields=withheld,
                   device=sorted(devs), precision=sorted(precs), rows_jsonl=os.path.abspath(rows_path),
                   seconds_this_launch=round(time.time() - t_arm, 1),
                   s_per_scene_forward_mean=(round(float(np.mean([recs[t]["s"] for t in mod])), 3) if mod else None),
                   KPR={"n": len(kprs), "max_abs_m": max(kprs) if kprs else None, "tol_m": KPR_TOL_M,
                        "verdict": ("PASS" if kprs and max(kprs) <= KPR_TOL_M else ("NO_ROWS" if not kprs else "FAIL"))})
        m = write_seam(a.out, arm, order, recs, toks, man)
        print(f"[bridge8] {arm}: {m.get('n_model_rows')} refcv8 rows, {m.get('n_cv_standin_rows')} stand-ins, missing "
              f"{m.get('n_missing')}, KPR {man['KPR']['verdict']} max {man['KPR']['max_abs_m']}", flush=True)
    if a.retime:
        recs = read_rows(os.path.join(a.out, "rows_R8_LEGAL.jsonl"))
        rt = {}
        for t in order:
            r = recs.get(t)
            if r is None:
                continue
            if r["source"] != "refcv8":
                rt[t] = r
                continue
            knots = B.retime_plan8(RS, table, np.asarray(r["knots"], np.float32),
                                   np.asarray(r["prior_emitted_path"], np.float32))
            rt[t] = {"token": t, "source": "refcv8", "knots": knots.tolist(), "poses": B.knots_to_navsim(knots).tolist()}
        man = dict(base_man, spec=B.DERIVED_ARMS["R8_RT"], derived_from="rows_R8_LEGAL.jsonl", device=["derived"],
                   precision=["derived"], note="R8_LEGAL's emitted plan and emitted prior re-timed; no extra forward")
        m = write_seam(a.out, "R8_RT", order, rt, toks, man)
        print(f"[bridge8] R8_RT: {m.get('n_model_rows')} rows", flush=True)
    if a.derived:
        pr = {}
        for t in order:
            r = toks[t]
            decl = B.declare8(r["ego_statuses"], "R8_LEGAL")
            hist = B.ego_history_poses(decl, times_rel_t0(r))
            pmf = B.prior_model_free(hist, B.v0_of(decl))
            pr[t] = {"token": t, "source": "prior_model_free", "knots": pmf["path"].tolist(),
                     "poses": B.knots_to_navsim(pmf["path"]).tolist(), "a0": pmf["a0"], "kappa0": pmf["kappa0"]}
        with open(os.path.join(a.out, "rows_PRIOR_ha0p.jsonl"), "w", encoding="utf-8") as fh:
            for t in order:
                fh.write(json.dumps(pr[t]) + "\n")
        man = dict(base_man, spec=B.DERIVED_ARMS["PRIOR_ha0p"], device=["model-free"],
                   precision=["fp32 (kinematic_prior, CPU)"],
                   note=("model-free: kinematic_prior.prior_controls('ha0_ext_pose') + prior_path on the SAME declared window "
                         "R8_LEGAL reads; knots -> NavSim poses through the same knots_to_navsim spline"))
        m = write_seam(a.out, "PRIOR_ha0p", order, pr, toks, man)
        print(f"[bridge8] PRIOR_ha0p: {m.get('n_model_rows')} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
