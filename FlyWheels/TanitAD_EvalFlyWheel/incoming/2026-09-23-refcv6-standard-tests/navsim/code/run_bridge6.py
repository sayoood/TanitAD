#!/usr/bin/env python3
"""Run refcv6 over NavSim two-stage scenes, one SEAM FILE per arm (TANITAD VENV).

    python code/run_bridge6.py --split warmup_two_stage --arms R6_A1,R6_BLIND \
        --inputs <export.json> --speed <speed_limits.json> --bank2 <416 stage-2 bank> \
        --ckpt D:/refcv6_eval_kit/ckpt/ckpt_step1000.pt --config D:/refcv6_eval_kit/ckpt/config.json \
        --device cpu --out raw/bridge_warmup_step1000

Per arm it writes ``seam_<arm>.npz`` (E2's seam format: token / fingerprint / source / poses
[N,8,3] / knots [N,8,2] / sampling / arm — scored by E2's seam agent, unchanged) and
``seam_<arm>.manifest.json`` (the DECLARED-INPUT MANIFEST: per token exactly what was fed).
Rows stream to ``rows_<arm>.jsonl`` as they are computed, so a killed run RESUMES (a token already
in the jsonl is not recomputed) and a partial run is still scorable on what it holds.

Stage-1 tokens: a real refcv6 row where the scene's ORIGINAL frames exist (``--bank1``), or where
the arm reads no pixels (``R6_BLIND``); otherwise the devkit's ConstantVelocityAgent is the
DECLARED stand-in (E2's rule) and the arm's official two-stage EPDMS is labelled HYBRID.

⛔ GPU: only with ``--device cuda``, and only after the brief's gate passes (memory.used < 4300 MiB,
no other python among ``nvidia-smi --query-compute-apps``, free host RAM >= 8 GB), re-checked
every 60 s up to ``--gpu-wait-s``. ⛔ One device per arm: a mixture is refused, never implied.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import refcv6_bridge as R6  # noqa: E402
import rig6  # noqa: E402

GPU_MEM_LIMIT_MIB = 4300
RAM_FLOOR_GB = 8.0


def free_ram_gb() -> float:
    import psutil
    return psutil.virtual_memory().available / 2**30


def gpu_gate() -> dict:
    """The brief's gate, read fresh. Returns {ok, reasons, ...}."""
    rep = {"ok": False, "reasons": []}
    try:
        mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        rep["mem_used_mib"] = int(mem[0])
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                               "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=30).stdout.strip().splitlines()
        py = [a for a in apps if "python" in a.lower() and str(os.getpid()) not in a.split(",")[0]]
        rep["other_python_compute_apps"] = py
    except Exception as e:                                               # noqa: BLE001
        rep["reasons"].append(f"nvidia-smi failed: {e!r}")
        return rep
    rep["free_ram_gb"] = round(free_ram_gb(), 2)
    if rep["mem_used_mib"] >= GPU_MEM_LIMIT_MIB:
        rep["reasons"].append(f"GPU memory.used {rep['mem_used_mib']} >= {GPU_MEM_LIMIT_MIB} MiB")
    if py:
        rep["reasons"].append(f"{len(py)} other python process(es) in compute-apps")
    if rep["free_ram_gb"] < RAM_FLOOR_GB:
        rep["reasons"].append(f"free host RAM {rep['free_ram_gb']} < {RAM_FLOOR_GB} GB")
    rep["ok"] = not rep["reasons"]
    return rep


def times_rel_t0(rec: dict) -> list:
    ts = [int(x) for x in rec["timestamps_us"]]
    return [(t - ts[-1]) / 1e6 for t in ts]


def stage1_rigs(tokens: list, toks: dict, logs_root: str) -> dict:
    """``{token: rig_record}`` for stage-1 / single-stage tokens, from each token's LOG pickle
    (its t0 frame's ``cams``) — ONE log in memory at a time (136 navtest logs would not fit)."""
    by_log: dict = {}
    for t in tokens:
        by_log.setdefault(toks[t]["log_name"], []).append(t)
    out = {}
    for ln, tl in sorted(by_log.items()):
        fr = R6.F4.E2BF.load(os.path.join(logs_root, f"{ln}.pkl"))
        idx = {f["token"]: f for f in fr}
        for t in tl:
            out[t] = R6.F4.rig_record(idx[toks[t]["frame_tokens"][-1]]["cams"])
        del fr, idx
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--speed", required=True)
    ap.add_argument("--speed-oracle", default="",
                    help="code/vmax_oracle.py output — REQUIRED by an arm whose vmax is 'oracle' "
                         "(SPEC §12, PRIVILEGED diagnostic)")
    ap.add_argument("--bank2", default="", help="stage-2 bank (two-stage splits)")
    ap.add_argument("--bank1", default="")
    ap.add_argument("--bank1-kind", choices=("v2", "navtest"), default="v2",
                    help="navtest = W3's unique-frame layout (build_navtest416.py)")
    ap.add_argument("--tokens-file", default="", help="restrict to a JSON {'tokens': [...]} subset")
    ap.add_argument("--logs-root", default="C:/Users/Admin/navsim-crun/data/openscene/navsim_logs/test")
    ap.add_argument("--road-plane", required=True, help="rig6.py output JSON (median road z)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt-md5", default="", help="expected md5; REFUSED on mismatch")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--exact-dedup", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-wait-s", type=int, default=0)
    ap.add_argument("--aux-dir", default=R6.AUX_DEFAULT)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    import torch
    torch.set_num_threads(int(a.threads))
    import tanitad
    tf = os.path.abspath(tanitad.__file__).replace("\\", "/").lower()
    if not tf.startswith(R6.REPO.replace("\\", "/").lower()):
        sys.exit(f"⛔ tanitad imported from {tanitad.__file__}, not under {R6.REPO}")
    gate_log = []
    if a.device.startswith("cuda"):
        t_w = time.time()
        while True:
            g = gpu_gate()
            gate_log.append({"t": round(time.time(), 1), **g})
            if g["ok"]:
                break
            if time.time() - t_w >= a.gpu_wait_s:
                print(f"⛔ GPU gate closed: {g['reasons']}", flush=True)
                os.makedirs(a.out, exist_ok=True)
                R6.json_dump({"refused": True, "gate": gate_log},
                             os.path.join(a.out, "GPU_GATE_REFUSED.json"))
                return 3
            print(f"[gate] waiting: {g['reasons']}", flush=True)
            time.sleep(60)
    elif torch.cuda.is_available():
        sys.exit("⛔ --device cpu but CUDA is visible — run with CUDA_VISIBLE_DEVICES=-1 so a CPU "
                 "job cannot touch the shared GPU")
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    for arm in arms:
        if arm not in R6.ARMS6:
            sys.exit(f"unknown arm {arm!r}; known {sorted(R6.ARMS6)}")
    os.makedirs(a.out, exist_ok=True)
    if a.inputs.endswith(".gz"):
        import gzip
        doc = json.load(gzip.open(a.inputs, "rt", encoding="utf-8"))
    else:
        doc = json.load(open(a.inputs, encoding="utf-8"))
    toks = doc["tokens"]
    speed = json.load(open(a.speed, encoding="utf-8"))
    if speed["split"] != a.split:
        sys.exit(f"⛔ speed file split {speed['split']} != {a.split}")
    oracle = None
    if any(R6.ARMS6[x]["vmax"] == "oracle" for x in a.arms.split(",") if x):
        if not a.speed_oracle:
            sys.exit("⛔ an ORACLE max-speed arm needs --speed-oracle (code/vmax_oracle.py output)")
        oracle = json.load(open(a.speed_oracle, encoding="utf-8"))
        if oracle["split"] != a.split:
            sys.exit(f"⛔ oracle file split {oracle['split']} != {a.split}")
    rp = json.load(open(a.road_plane, encoding="utf-8"))
    road_z = float(rp["summary"]["median"])
    if a.split == "navtest":                       # single-stage: every token an original frame
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
    bank2 = R6.Bank416(a.bank2) if a.bank2 else None
    if a.bank1:
        bank1 = (R6.BankNavtest416(a.bank1) if a.bank1_kind == "navtest" else R6.Bank416(a.bank1))
    else:
        bank1 = None
    ckpt_md5 = R6.B2.md5_file(a.ckpt)
    if a.ckpt_md5 and ckpt_md5 != a.ckpt_md5:
        sys.exit(f"⛔ checkpoint md5 {ckpt_md5} != expected {a.ckpt_md5}")
    t_load = time.time()
    model, cfg, targs, prov, arm_mod, meta = R6.load_refcv6(a.ckpt, a.config, a.device,
                                                            a.precision, a.aux_dir)
    if a.exact_dedup:
        R6.exact_dedup(model.core.encoder)
    steps = int(prov["decoder_steps"])
    pcfg = model._perception.cfg
    load_s = round(time.time() - t_load, 1)
    blind_grey = int(round((bank2 or bank1).mean_px))
    # rigs of stage-1 tokens whose bank carries no rig record (navtest) or that have no bank (BLIND)
    need_rig = [t for t in s1 if bank1 is None or t not in bank1.prov
                or not hasattr(bank1, "rigs")]
    s1_rigs = stage1_rigs(need_rig, toks, a.logs_root) if need_rig else {}
    model_rec = {"ckpt": a.ckpt, "ckpt_md5": ckpt_md5, "step": prov["step"],
                 "registry": "MODEL_REGISTRY.md — refcv6-r101-s0 (TRAINING; pipeline validation)",
                 "rebuilt_from": prov["rebuilt_from"], "state_dict_load": prov["state_dict_load"],
                 "window": prov["window"], "decoder_steps": steps, "n_anchors": prov["n_anchors"],
                 "horizons_steps": prov["horizons"], "load_s": load_s, **meta,
                 "torch": torch.__version__, "threads": int(a.threads),
                 "exact_dedup": bool(a.exact_dedup),
                 "lift": {"stride": int(pcfg.stride), "heights_m": list(pcfg.heights_m),
                          "equalize_bottom_rows_lift": R6.EQUALIZE_BOTTOM_ROWS,
                          "road_z_m": road_z, "road_plane_source": os.path.abspath(a.road_plane)}}
    print(f"[bridge6] step={prov['step']} steps={steps} precision={meta['precision']} "
          f"device={a.device} load {load_s}s md5={ckpt_md5}", flush=True)
    lift_cache: dict = {}

    def lift_for(rig: dict):
        k = rig["rig_key"]
        if k not in lift_cache:
            g, v, summ = rig6.lift_geometry(rig, road_z, stride=int(pcfg.stride),
                                            heights_m=tuple(pcfg.heights_m),
                                            equalize_bottom_rows=R6.EQUALIZE_BOTTOM_ROWS)
            lift_cache[k] = (g, v, summ)
        return lift_cache[k]

    for arm in arms:
        spec = R6.ARMS6[arm]
        rows_path = os.path.join(a.out, f"rows_{arm}.jsonl")
        done = {}
        if os.path.exists(rows_path):
            for ln in open(rows_path, encoding="utf-8"):
                try:
                    r = json.loads(ln)
                    done[r["token"]] = r
                except json.JSONDecodeError:
                    pass                  # a torn last line from a kill: recomputed below
        t_arm = time.time()
        n_new = 0
        with open(rows_path, "a", encoding="utf-8") as fh:
            for stage, tlist in ((1, s1), (2, s2)):
                for tok in tlist:
                    if tok in done:
                        continue
                    r = toks[tok]
                    rec = {"token": tok, "stage": stage, "arm": arm}
                    times = times_rel_t0(r)
                    frames_mode = spec["frames"]
                    bank = bank2 if stage == 2 else bank1
                    key = r["scene_token"] if stage == 2 else tok
                    if a.split == "navtest" and (bank is None or key not in bank.prov):
                        # single-stage: no stand-in exists — a token without frames is REFUSED
                        # (a CV row here would be scored as refcv6's)
                        sys.exit(f"⛔ navtest token {tok} is not in the 416 bank — the bank is "
                                 f"incomplete; refusing (build it, or pass --tokens-file)")
                    if frames_mode != "BLIND" and (bank is None or key not in bank.prov):
                        rec.update({"source": "cv_standin",
                                    "why": ("no ORIGINAL camera frames for this stage-1 scene "
                                            "on this box — the devkit ConstantVelocityAgent is "
                                            "the DECLARED stand-in")})
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                        continue
                    decl = R6.declare6(r["ego_statuses"], arm)
                    nav = R6.nav_input(decl, arm)
                    vsrc = oracle if spec["vmax"] == "oracle" else speed
                    vmax = R6.max_speed_input(vsrc["tokens"].get(tok), arm)
                    hist = R6.ego_history_poses(decl, times)
                    src = R6.slot_sources6(times, frames_mode)
                    if bank is not None and key in bank.prov:
                        fr, rk, sha = bank.load(key)
                        rig = bank.rigs[rk] if hasattr(bank, "rigs") else s1_rigs[tok]
                        if rig["rig_key"] != rk:
                            sys.exit(f"⛔ {tok}: rig {rig['rig_key']} != bank rig {rk}")
                    else:                           # BLIND on a stage-1 scene with no bank
                        rig = s1_rigs[tok]
                        fr, sha = np.zeros((1, 416, 1024, 3), np.uint8), None
                    rows_u8 = R6.pack_rows(fr, src, blind_grey if frames_mode == "BLIND" else None)
                    g, v, lsumm = lift_for(rig)
                    seed = R6.scene_seed(spec["seed"], tok)
                    t1 = time.time()
                    res = R6.run_model6(model, arm_mod, rows_u8.numpy(), decl, arm, nav, vmax,
                                        hist, g, v, steps, seed, a.device)
                    poses = R6.knots_to_navsim(res["traj"])
                    rec.update({"source": "refcv6", "scene_token": r["scene_token"],
                                "frame_sha16": sha, "frames": frames_mode,
                                "slot_sources": src, "blind_grey": (blind_grey if frames_mode
                                                                    == "BLIND" else None),
                                "navsim_frame_times_s": [round(x, 4) for x in times],
                                "declared_values": decl, "v0": res["v0"], "nav": nav,
                                "vmax": vmax, "ego_history_poses": hist.tolist(),
                                "rig_key": rig["rig_key"], "lift": lsumm, "seed": seed,
                                "knots": res["traj"].tolist(), "poses": poses.tolist(),
                                "diag": res["diag"], "s": round(time.time() - t1, 3),
                                "device": a.device, "precision": meta["precision"]})
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    n_new += 1
                    done[tok] = rec
                    if n_new % 25 == 0:
                        el = time.time() - t_arm
                        print(f"  [{arm}] {len(done)}/{len(s1)+len(s2)} ({n_new} new) "
                              f"{el/n_new:.2f} s/scene", flush=True)
        # ---- assemble the seam from the jsonl (resume-safe) ----------------------------- #
        recs = {}
        for ln in open(rows_path, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            recs[r["token"]] = r
        order = s1 + s2
        missing = [t for t in order if t not in recs]
        devs = {recs[t].get("device") for t in order if t in recs and recs[t]["source"] == "refcv6"}
        precs = {recs[t].get("precision") for t in order if t in recs
                 and recs[t]["source"] == "refcv6"}
        if len(devs) > 1 or len(precs) > 1:
            sys.exit(f"⛔ {arm}: mixed devices {devs} / precisions {precs} in one arm — refused")
        tok_l, fps, srcs, poses_all, knots_all = [], [], [], [], []
        for t in order:
            if t not in recs:
                continue
            r = recs[t]
            tok_l.append(t)
            fps.append(toks[t]["fingerprint"])
            if r["source"] == "refcv6":
                srcs.append("precomputed")
                poses_all.append(np.asarray(r["poses"], np.float32))
                knots_all.append(np.asarray(r["knots"], np.float32))
            else:
                srcs.append("cv_standin")
                poses_all.append(np.full((8, 3), np.nan, np.float32))
                knots_all.append(np.full((8, 2), np.nan, np.float32))
        np.savez(os.path.join(a.out, f"seam_{arm}.npz"), token=np.asarray(tok_l),
                 fingerprint=np.asarray(fps), source=np.asarray(srcs),
                 poses=np.stack(poses_all), knots=np.stack(knots_all),
                 sampling=np.asarray([8, 0.5]), arm=np.asarray(arm))
        n_model = sum(1 for s in srcs if s == "precomputed")
        withheld = [f"{f}[{k}]" for f in R6.FIELDS for k in range(4)
                    if f"{f}[{k}]" not in spec["declared"]]
        man = {"arm": arm, "spec": spec, "split": a.split, "declared_inputs": list(spec["declared"]),
               "withheld_egostatus_fields": withheld,
               "external_inputs": {"max_speed": {
                   "map": "nuPlan map posted limit of the ego lane at t0",
                   "off": "withheld",
                   "oracle": ("⛔ PRIVILEGED: the human's own realised max |v| over [t0+2, t0+6] s "
                              "(SPEC §12) — a diagnostic, never a result")}[spec["vmax"]],
                                   "lift": "NavSim rig calibration (rig6.py)",
                                   "frames": "416x1024 cylindrical 3-cam stitch (frames416.py)"},
               "n_expected": len(order), "n_rows": len(tok_l), "missing": missing[:50],
               "n_missing": len(missing), "partial": bool(missing),
               "n_refcv6_rows": n_model, "n_cv_standin_rows": len(tok_l) - n_model,
               "device": sorted(devs), "precision": sorted(precs), "model": model_rec,
               "gpu_gate": gate_log[-1] if gate_log else None,
               "bank2": a.bank2, "bank1": a.bank1 or None, "inputs": os.path.abspath(a.inputs),
               "speed": os.path.abspath(a.speed),
               "speed_oracle": os.path.abspath(a.speed_oracle) if a.speed_oracle else None,
               "conversion": R6.knots_to_navsim.__doc__,
               "nav_map": {str(k): v for k, v in R6.NAVSIM_CMD_TO_NAV_NAME.items()},
               "rows_jsonl": os.path.abspath(rows_path),
               "seconds_this_launch": round(time.time() - t_arm, 1)}
        R6.json_dump(man, os.path.join(a.out, f"seam_{arm}.manifest.json"))
        print(f"[bridge6] {arm}: {n_model} refcv6 rows, {len(tok_l)-n_model} stand-ins, "
              f"missing {len(missing)}, {man['seconds_this_launch']} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
