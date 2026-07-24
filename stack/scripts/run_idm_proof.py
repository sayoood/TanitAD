"""Pod runner for the IDM cross-domain proof (IDM_VIDEO_PRETRAIN_DESIGN §5).

Frozen flagship-v1 encoder -> cache latents once -> fit the small non-causal IDM
head (scripts/idm_head.py) for the two PRE-REGISTERED contrasts:
  (#2) rig-A -> rig-B within PhysicalAI  (cheap intrinsics-shift pre-probe)
  (#3) PhysicalAI -> comma2k19            (the real rig gap, the go/no-go)

Stages (resumable; latents are the only heavy step, cached to disk):
  rig     build ep-index -> clip_id -> cy -> rig table from the order TSV + calib
  encode  frozen encoder over the selected episodes -> per-episode latent .pt
  fit     build windows, train the head per contrast, write results.json

Run on a NON-training pod (pod3, A40), under gpu_lock.sh acquire idm-proof.
Loads ONLY the encoder+readout weights from the ckpt (the encoder is purely
visual: no action/speed channel), so the v1 speed-input action_dim is irrelevant.

Usage:
  PYTHONPATH=/workspace/TanitAD/stack python3 scripts/run_idm_proof.py --stage all \
    --ckpt /workspace/tmp/idm/ckpt.pt \
    --pai-cache /workspace/pai_epcache/physicalai-train-e438721ae894 \
    --pai-val-cache /workspace/pai_epcache/physicalai-val-f1b378f295ae \
    --comma-cache /workspace/data/comma2k19-val-61c46fca8f7f \
    --order /workspace/tmp/train_clip_order.tsv \
    --calib-root /workspace/pai_build \
    --work /workspace/tmp/idm --out /workspace/tmp/idm/results.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))          # scripts/
import idm_head as ih  # noqa: E402

RIG_SPLIT_CY = 650.0            # filtering.RIG_CLUSTERS physicalai_av split
FRONT_WIDE = "camera_front_wide_120fov"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# rig table: ep-index -> clip_id -> cy -> rig                                  #
# --------------------------------------------------------------------------- #
def _chunk_cy_map(calib_root: str, chunk: int, _cache: dict = {}) -> dict:
    if chunk in _cache:
        return _cache[chunk]
    import pandas as pd
    p = (Path(calib_root) / "calibration" / "camera_intrinsics" /
         f"camera_intrinsics.chunk_{chunk:04d}.parquet")
    if not p.exists():
        _cache[chunk] = {}
        return {}
    df = pd.read_parquet(p).reset_index()
    name_col = ("camera_name" if "camera_name" in df.columns else
                "sensor_name" if "sensor_name" in df.columns else None)
    if name_col is None:
        df = df.rename(columns={"level_0": "clip_id", "level_1": "name"})
        name_col = "name"
    fw = df[df[name_col].astype(str) == FRONT_WIDE]
    m = {str(r.clip_id): float(r.cy) for r in fw.itertuples(index=False)}
    _cache[chunk] = m
    return m


def build_rig_table(order_tsv: str, calib_root: str, out_json: str) -> dict:
    rows = []
    with open(order_tsv) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                rows.append((int(p[0]), p[1], int(p[2])))
    table = {}
    n_a = n_b = n_unk = 0
    for idx, cid, chunk in rows:
        cy = _chunk_cy_map(calib_root, chunk).get(cid)
        if cy is None:
            rig = "unknown"
            n_unk += 1
        else:
            rig = "a" if cy < RIG_SPLIT_CY else "b"
            n_a += rig == "a"
            n_b += rig == "b"
        table[str(idx)] = {"clip_id": cid, "chunk": chunk, "cy": cy, "rig": rig}
    Path(out_json).write_text(json.dumps(table))
    log(f"rig table: {len(table)} clips -> rig_a {n_a} rig_b {n_b} unknown {n_unk}")
    return table


# --------------------------------------------------------------------------- #
# frozen encoder                                                              #
# --------------------------------------------------------------------------- #
def load_encoder(ckpt_path: str, device: str):
    """Build the flagship-v1 encoder+readout and load ONLY those weights from the
    ckpt (strict on the two submodules). Returns (enc, readout, meta)."""
    from tanitad.config import flagship4b_config
    from tanitad.models.encoder import ViTEncoder
    from tanitad.models.readout import SpatialGridReadout
    cfg = flagship4b_config()
    enc = ViTEncoder(cfg.encoder)
    readout = SpatialGridReadout(enc.n_tokens, cfg.encoder.d_model,
                                 grid=cfg.readout.grid,
                                 d_readout=cfg.readout.d_readout)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items()
              if k.startswith("encoder.")}
    ro_sd = {k[len("readout."):]: v for k, v in sd.items()
             if k.startswith("readout.")}
    miss_e, unexp_e = enc.load_state_dict(enc_sd, strict=True)
    miss_r, unexp_r = readout.load_state_dict(ro_sd, strict=True)
    state_dim = readout.out_dim
    step = int(ck.get("step", -1)) if isinstance(ck, dict) else -1
    enc.to(device).eval()
    readout.to(device).eval()
    for p in list(enc.parameters()) + list(readout.parameters()):
        p.requires_grad_(False)
    log(f"encoder loaded: {sum(p.numel() for p in enc.parameters())/1e6:.1f}M "
        f"enc + {sum(p.numel() for p in readout.parameters())/1e6:.1f}M readout, "
        f"state_dim {state_dim}, ckpt step {step}")
    return enc, readout, {"state_dim": state_dim, "ckpt_step": step,
                          "enc_keys": len(enc_sd), "ro_keys": len(ro_sd)}


@torch.no_grad()
def encode_frames(enc, readout, frames_u8: torch.Tensor, device: str,
                  batch: int = 32) -> torch.Tensor:
    """frames_u8 [T,9,256,256] uint8 -> z [T, state_dim] fp16 (mirrors
    WorldModel.encode: readout(encoder(frames/255)))."""
    zs = []
    T = frames_u8.shape[0]
    for i in range(0, T, batch):
        fb = frames_u8[i:i + batch].to(device).float().div_(255.0)
        z = readout(enc(fb))
        zs.append(z.half().cpu())
    return torch.cat(zs)


def _load_ep(path: str) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(d, dict):                                    # ToyEpisode obj
        d = {"frames_u8": d.frames, "poses": d.poses, "actions": d.actions}
    fr = d.get("frames_u8", d.get("frames"))
    return {"frames_u8": fr, "poses": d["poses"], "actions": d["actions"]}


def encode_set(enc, readout, ep_paths: list[tuple[str, str]], latent_dir: str,
               device: str) -> list[str]:
    """Encode each (tag, ep_path) to <latent_dir>/<tag>.pt = {z, poses, actions}.
    Resumable: skips tags already on disk. Returns the list of latent files."""
    Path(latent_dir).mkdir(parents=True, exist_ok=True)
    out = []
    for j, (tag, p) in enumerate(ep_paths):
        lf = Path(latent_dir) / f"{tag}.pt"
        out.append(str(lf))
        if lf.exists():
            continue
        ep = _load_ep(p)
        z = encode_frames(enc, readout, ep["frames_u8"], device)
        torch.save({"z": z, "poses": ep["poses"].float(),
                    "actions": ep["actions"].float()}, lf)
        if j % 25 == 0:
            log(f"encode {j}/{len(ep_paths)} -> {lf.name} z{tuple(z.shape)}")
    return out


# --------------------------------------------------------------------------- #
# windows from cached latents                                                 #
# --------------------------------------------------------------------------- #
def windows_from_latents(latent_files: list[str], k: int = 4, stride: int = 2):
    Z, S, T = [], [], []
    for lf in latent_files:
        d = torch.load(lf, map_location="cpu", weights_only=False)
        zw, sc, tj = ih.build_windows(d["z"].float(), d["poses"].float(),
                                      d["actions"].float(), k=k, stride=stride)
        if zw.shape[0]:
            Z.append(zw)
            S.append(sc)
            T.append(tj)
    if not Z:
        raise RuntimeError("no windows built (episodes too short?)")
    return torch.cat(Z), torch.cat(S), torch.cat(T)


# --------------------------------------------------------------------------- #
# orchestration                                                               #
# --------------------------------------------------------------------------- #
def select_episodes(rig_table: dict, pai_cache: str, cap_a: int, cap_b: int):
    """Ordered (tag, path) lists of rig-A and rig-B PhysicalAI TRAIN episodes that
    exist in the cache (skip indices with no ep file)."""
    a, b = [], []
    for idx in sorted(int(i) for i in rig_table):
        rig = rig_table[str(idx)]["rig"]
        p = Path(pai_cache) / f"ep_{idx:05d}.pt"
        if not p.exists():
            continue
        if rig == "a" and len(a) < cap_a:
            a.append((f"pai_a_{idx:05d}", str(p)))
        elif rig == "b" and len(b) < cap_b:
            b.append((f"pai_b_{idx:05d}", str(p)))
        if len(a) >= cap_a and len(b) >= cap_b:
            break
    return a, b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["rig", "encode", "fit", "all"],
                    default="all")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pai-cache", required=True)
    ap.add_argument("--pai-val-cache", required=True)
    ap.add_argument("--comma-cache", required=True)
    ap.add_argument("--order", required=True)
    ap.add_argument("--calib-root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-a", type=int, default=300)
    ap.add_argument("--cap-b", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--git-hash", default="unknown")
    args = ap.parse_args()

    Path(args.work).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rig_json = str(Path(args.work) / "rig_table.json")
    latent_dir = str(Path(args.work) / "latents")

    # ---- rig table ----
    if args.stage in ("rig", "encode", "all") or not Path(rig_json).exists():
        rig_table = build_rig_table(args.order, args.calib_root, rig_json)
    else:
        rig_table = json.loads(Path(rig_json).read_text())
    if args.stage == "rig":
        return

    # ---- selection ----
    a_eps, b_eps = select_episodes(rig_table, args.pai_cache, args.cap_a, args.cap_b)
    val_eps = [(f"pai_val_{i:05d}", str(p))
               for i, p in enumerate(sorted(Path(args.pai_val_cache).glob("ep_*.pt")))]
    comma_eps = [(f"comma_{i:05d}", str(p))
                 for i, p in enumerate(sorted(Path(args.comma_cache).glob("ep_*.pt")))]
    log(f"selected: rig_a {len(a_eps)} rig_b {len(b_eps)} pai_val {len(val_eps)} "
        f"comma {len(comma_eps)}")

    # ---- encode ----
    enc = readout = enc_meta = None
    if args.stage in ("encode", "all"):
        enc, readout, enc_meta = load_encoder(args.ckpt, device)
        encode_set(enc, readout, a_eps + b_eps + val_eps + comma_eps,
                   latent_dir, device)
        log("encode stage done")
    if args.stage == "encode":
        return

    # ---- fit ----
    def lf(tag: str) -> str:
        return str(Path(latent_dir) / f"{tag}.pt")

    def files(eps):
        return [lf(t) for (t, _p) in eps]

    if enc_meta is None:                       # infer state_dim from a latent
        sample = torch.load(files(a_eps or comma_eps)[0], weights_only=False)
        enc_meta = {"state_dim": int(sample["z"].shape[1]), "ckpt_step": -1}
    state_dim = enc_meta["state_dim"]

    # split rig-A into train / in-rig held-out (85/15 by clip)
    n_a = len(a_eps)
    a_cut = max(1, int(round(n_a * 0.85)))
    a_tr, a_ho = a_eps[:a_cut], a_eps[a_cut:]
    log(f"rig-A: train {len(a_tr)} clips / in-rig heldout {len(a_ho)} clips; "
        f"rig-B eval {len(b_eps)} clips")

    def W(eps):
        return windows_from_latents(files(eps), k=args.k, stride=args.stride)

    results = {
        "meta": {
            "experiment": "idm_cross_domain_proof",
            "design": "IDM_VIDEO_PRETRAIN_DESIGN §5",
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": device, "git_hash": args.git_hash,
            "ckpt": args.ckpt, "ckpt_md5": md5_of(args.ckpt),
            "ckpt_step": enc_meta.get("ckpt_step"),
            "state_dim": state_dim, "k": args.k, "window": 2 * args.k + 1,
            "stride": args.stride, "epochs": args.epochs, "seed": args.seed,
            "horizons": list(ih.DEFAULT_HORIZONS), "rig_split_cy": RIG_SPLIT_CY,
            "n_clips": {"rig_a_train": len(a_tr), "rig_a_heldout": len(a_ho),
                        "rig_b": len(b_eps), "pai_val": len(val_eps),
                        "comma_val": len(comma_eps)},
            "pass_rule": "cross-domain speed R2>0.9 AND yaw R2>0.9 AND "
                         "traj ADE@2s < 1.5x in-domain heldout ADE@2s",
        },
        "experiments": {},
    }

    # ---- Experiment #2: rig-A -> rig-B ----
    log("=== Experiment #2: rig-A -> rig-B ===")
    ex2 = ih.train_head(
        W(a_tr),
        {"in_rig_heldout_rigA": W(a_ho), "cross_rig_rigB": W(b_eps)},
        state_dim=state_dim, epochs=args.epochs, seed=args.seed, device=device,
        log=log)
    results["experiments"]["rigA_to_rigB"] = ex2

    # ---- Experiment #3: PhysicalAI -> comma2k19 ----
    log("=== Experiment #3: PhysicalAI -> comma2k19 ===")
    pai_train = a_tr + b_eps                           # all rigs (rig-A train + rig-B)
    ex3 = ih.train_head(
        W(pai_train),
        {"in_corpus_heldout_paival": W(val_eps),
         "cross_domain_comma": W(comma_eps)},
        state_dim=state_dim, epochs=args.epochs, seed=args.seed, device=device,
        log=log)
    results["experiments"]["physicalai_to_comma2k19"] = ex3

    # ---- verdicts ----
    def verdict(cross: dict, in_dom_ade: float) -> dict:
        r2 = cross["r2"]
        ok_speed = r2["speed"] > 0.9
        ok_yaw = r2["yaw_rate"] > 0.9
        ok_ade = cross["ade_2s"] < 1.5 * in_dom_ade
        return {"speed_r2": r2["speed"], "yaw_r2": r2["yaw_rate"],
                "steer_r2_secondary": r2["steer"], "cross_ade_2s": cross["ade_2s"],
                "in_domain_ade_2s": in_dom_ade,
                "ade_ratio": cross["ade_2s"] / max(in_dom_ade, 1e-9),
                "speed_ok": ok_speed, "yaw_ok": ok_yaw, "ade_ok": ok_ade,
                "PASS": bool(ok_speed and ok_yaw and ok_ade)}

    results["verdicts"] = {
        "rigA_to_rigB": verdict(ex2["val"]["cross_rig_rigB"],
                                ex2["val"]["in_rig_heldout_rigA"]["ade_2s"]),
        "physicalai_to_comma2k19": verdict(
            ex3["val"]["cross_domain_comma"],
            ex3["val"]["in_corpus_heldout_paival"]["ade_2s"]),
    }
    results["go_no_go"] = {
        "primary": "physicalai_to_comma2k19",
        "PASS": results["verdicts"]["physicalai_to_comma2k19"]["PASS"],
        "note": "program go/no-go follows the real rig gap (#3); #2 corroborates",
    }

    Path(args.out).write_text(json.dumps(results, indent=2))
    log(f"WROTE {args.out}")
    log("VERDICT " + json.dumps(results["verdicts"]))
    log("IDM_PROOF_DONE")


if __name__ == "__main__":
    main()
