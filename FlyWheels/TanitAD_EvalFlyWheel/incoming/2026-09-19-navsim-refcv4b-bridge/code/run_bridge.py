#!/usr/bin/env python3
"""Run refcv4b over the NavSim warmup scenes, one SEAM FILE per arm (TANITAD VENV, CPU).

    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=D:/Projects/TanitAD/stack \
      python code/run_bridge.py --arms A1_ego_cmd,A2_vision_pure,... --out raw

Writes, per arm:
  raw/seam_<arm>.npz            fingerprint/token/source/poses[N,8,3]/knots[N,8,2]
  raw/seam_<arm>.manifest.json  the DECLARED-INPUT MANIFEST (gate navsim.ego_enforcement):
                                per token exactly what was fed (ego block, nav, frame
                                sha, slot sources), plus the arm's declared/withheld lists
⛔ GPU: refused. The RTX 4060 belongs to a live training run; this process asserts
``torch.cuda.is_available() is False`` before loading anything.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import tanitad_navsim_bridge as B  # noqa: E402

DEF_CKPT = "D:/Projects/TanitAD-artifacts/refcv4b_final/ckpt_40284_FINAL.pt"
DEF_BANK = "C:/Users/Admin/tanitad-wt/_s2build/navsim/corpus"
CKPT_MD5 = "99b573e8277d94a5e3bfbf630cb4d751"     # README.md §6 + registry §4.6


def _rss_gb() -> float | None:
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 2**30, 3)
    except Exception:            # noqa: BLE001
        return None


def times_rel_t0(rec: dict) -> list[float]:
    ts = [int(x) for x in rec["timestamps_us"]]
    return [(t - ts[-1]) / 1e6 for t in ts]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--inputs", default=None, help="raw/navsim_agent_inputs.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=DEF_CKPT)
    ap.add_argument("--bank", default=DEF_BANK)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N stage-2 tokens")
    ap.add_argument("--skip-md5", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for output names (mutation runs)")
    ap.add_argument("--mutate", default="", help="JSON file of per-token EgoStatus overrides "
                                                 "(tests only)")
    a = ap.parse_args(argv)
    import torch
    if torch.cuda.is_available():
        sys.exit("⛔ CUDA is visible — run with CUDA_VISIBLE_DEVICES=-1 (the GPU is "
                 "occupied by a live training run; CPU only).")
    torch.set_num_threads(int(a.threads))
    import tanitad
    if not os.path.abspath(tanitad.__file__).replace("\\", "/").lower().startswith(
            B.REPO.replace("\\", "/").lower()):
        sys.exit(f"⛔ tanitad imported from {tanitad.__file__}, not under {B.REPO}")

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    for arm in arms:
        if arm not in B.ARMS:
            sys.exit(f"unknown arm {arm!r}; known {sorted(B.ARMS)}")
    os.makedirs(a.out, exist_ok=True)
    inputs_path = a.inputs or os.path.join(a.out, "navsim_agent_inputs.json")
    doc = json.load(open(inputs_path, encoding="utf-8"))
    toks = doc["tokens"]
    if a.mutate:
        mut = json.load(open(a.mutate, encoding="utf-8"))
        for tok, statuses in mut["tokens"].items():
            toks[tok]["ego_statuses"] = statuses
    s1 = sorted(t for t, r in toks.items() if r["stage"] == 1)
    s2 = sorted(t for t, r in toks.items() if r["stage"] == 2)
    if a.limit:
        s2 = s2[:a.limit]
    if not s1 or not s2:
        sys.exit("⛔ empty token set")

    bank_build = json.load(open(os.path.join(a.bank, "BUILD.json"), encoding="utf-8"))
    tag = B.frame_tag_check(bank_build)
    bank = B.FrameBank.open(a.bank)
    t_load = time.time()
    ckpt_md5 = None if a.skip_md5 else B.md5_file(a.ckpt)
    if ckpt_md5 is not None and ckpt_md5 != CKPT_MD5:
        sys.exit(f"⛔ checkpoint md5 {ckpt_md5} != registry {CKPT_MD5}")
    model, cfg, targs, prov, arm_mod = B.load_refcv4b(a.ckpt)
    tr = arm_mod.trainer()
    steps = int(prov["decoder_steps"])
    W = int(prov["window"])
    units = json.load(open(os.path.join(os.path.dirname(a.ckpt), "anchors.units.json"),
                           encoding="utf-8"))
    if (units["alat_v_floor_ms"], units["kappa_cap_inv_m"]) != (B.ALAT_V_FLOOR_MS,
                                                                 B.KAPPA_CAP_INV_M):
        sys.exit(f"⛔ caps drifted: {units['alat_v_floor_ms']}/{units['kappa_cap_inv_m']}")
    load_s = time.time() - t_load
    print(f"[bridge] model step={prov['step']} window={W} steps={steps} "
          f"anchors={prov['n_anchors']} md5={ckpt_md5} load {load_s:.1f}s rss={_rss_gb()}GB",
          flush=True)
    blind_grey = int(round(float(np.mean([v for v in
                                          __import__('pandas').read_parquet(
                                              os.path.join(a.bank, 'frames_provenance.parquet')
                                          ).mean_px]))))
    frame_cache: dict = {}
    summary = {"arms": {}, "model": {
        "ckpt": a.ckpt, "ckpt_md5": ckpt_md5, "step": prov["step"],
        "registry": "MODEL_REGISTRY.md §4.6 refcv4b-b1-v72-40k",
        "rebuilt_from": prov["rebuilt_from"], "window_rows": W, "decoder_steps": steps,
        "n_anchors": prov["n_anchors"], "horizons_steps": prov["horizons"],
        "state_dict_load": prov["state_dict_load"], "device": "cpu",
        "torch": torch.__version__, "threads": int(a.threads), "load_s": round(load_s, 1)},
        "frame_tag": tag, "blind_grey_u8": blind_grey}

    for arm in arms:
        spec = B.ARMS[arm]
        t_arm = time.time()
        recs, fps, srcs, poses_all, knots_all, tok_list = [], [], [], [], [], []
        for tok in s1:
            r = toks[tok]
            if spec["frames"] == "BLIND":
                decl = B.declare(r["ego_statuses"], arm)
                rows = B.pack_frames(np.zeros((4, 256, 640, 3), np.uint8),
                                     B.slot_sources(times_rel_t0(r), "ST", W), blind_grey)
                res = B.run_model(model, tr, rows, decl, steps)
                poses = B.knots_to_navsim(res["traj"])
                recs.append({"token": tok, "stage": 1, "source": "refcv4b",
                             "declared_values": decl, "ego_block": res["ego"],
                             "nav": res["nav"], "frames": "BLIND(no pixels read)",
                             "knots": res["traj"].tolist(), "diag": res["diag"]})
                srcs.append("refcv4b")
                poses_all.append(poses.astype(np.float32))
                knots_all.append(res["traj"].astype(np.float32))
            else:
                recs.append({"token": tok, "stage": 1, "source": "cv_standin",
                             "why": ("no stage-1 camera frames on this box (0/192 original "
                                     "jpgs) — the devkit ConstantVelocityAgent is the "
                                     "DECLARED stand-in so the official script can run; "
                                     "stage-1 rows of this arm are NOT refcv4b")})
                srcs.append("cv_standin")
                poses_all.append(np.full((8, 3), np.nan, np.float32))
                knots_all.append(np.full((8, 2), np.nan, np.float32))
            fps.append(r["fingerprint"])
            tok_list.append(tok)
        for i, tok in enumerate(s2):
            r = toks[tok]
            decl = B.declare(r["ego_statuses"], arm)
            st = r["scene_token"]
            if st not in frame_cache:
                frame_cache[st] = B.FrameBank.load_wide(bank, st)
            fr, mask, sha = frame_cache[st]
            times = times_rel_t0(r)
            if spec["frames"] == "BLIND":
                src_idx = B.slot_sources(times, "ST", W)
                rows = B.pack_frames(fr, src_idx, blind_grey)
            else:
                src_idx = B.slot_sources(times, spec["frames"], W)
                rows = B.pack_frames(fr, src_idx)
            res = B.run_model(model, tr, rows, decl, steps)
            poses = B.knots_to_navsim(res["traj"])
            recs.append({"token": tok, "stage": 2, "source": "refcv4b",
                         "scene_token": st, "frame_sha16": sha,
                         "frames": spec["frames"], "slot_sources": src_idx,
                         "navsim_frame_times_s": [round(x, 4) for x in times],
                         "declared_values": decl, "ego_block": res["ego"],
                         "nav": res["nav"], "knots": res["traj"].tolist(),
                         "poses": poses.tolist(), "diag": res["diag"]})
            fps.append(r["fingerprint"])
            srcs.append("refcv4b")
            poses_all.append(poses.astype(np.float32))
            knots_all.append(res["traj"].astype(np.float32))
            tok_list.append(tok)
            if (i + 1) % 25 == 0:
                el = time.time() - t_arm
                print(f"  [{arm}] {i+1}/{len(s2)} stage-2 scenes, {el/(i+1):.2f} s/scene, "
                      f"rss={_rss_gb()}GB", flush=True)
        name = f"seam_{arm}{a.tag}"
        np.savez(os.path.join(a.out, name + ".npz"),
                 token=np.asarray(tok_list), fingerprint=np.asarray(fps),
                 source=np.asarray(srcs), poses=np.stack(poses_all),
                 knots=np.stack(knots_all), sampling=np.asarray([8, 0.5]),
                 arm=np.asarray(arm))
        withheld = [f"{f}[t{k}]" for f in B.FIELDS for k in ("-3", "-2", "-1", "0")
                    if f"{f}[t0]" not in spec["declared"] or k != "0"]
        man = {"arm": arm, "spec": spec, "declared_inputs": list(spec["declared"]),
               "withheld_egostatus_fields": withheld,
               "frames_construction": spec["frames"],
               "frames_variant": "wide (PHYSICALAI_WIDE120_256x640), sha-verified per scene",
               "conversion": B.knots_to_navsim.__doc__,
               "nav_map": {str(k): v for k, v in B.NAVSIM_CMD_TO_NAV_NAME.items()},
               "n_stage1": len(s1), "n_stage2": len(s2),
               "n_refcv4b_rows": int(sum(1 for s in srcs if s == "refcv4b")),
               "n_cv_standin_rows": int(sum(1 for s in srcs if s == "cv_standin")),
               "seconds": round(time.time() - t_arm, 1), "rss_gb": _rss_gb(),
               "tag": a.tag, "mutated_from": a.mutate or None,
               "model": summary["model"], "frame_tag": tag, "blind_grey_u8": blind_grey,
               "rows": recs}
        B.json_dump(man, os.path.join(a.out, name + ".manifest.json"))
        summary["arms"][arm] = {k: man[k] for k in ("n_stage1", "n_stage2", "n_refcv4b_rows",
                                                    "n_cv_standin_rows", "seconds", "rss_gb")}
        print(f"[bridge] {arm}: {man['n_refcv4b_rows']} refcv4b rows, "
              f"{man['n_cv_standin_rows']} cv stand-ins, {man['seconds']} s", flush=True)
    B.json_dump(summary, os.path.join(a.out, f"bridge_run_summary{a.tag}.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
