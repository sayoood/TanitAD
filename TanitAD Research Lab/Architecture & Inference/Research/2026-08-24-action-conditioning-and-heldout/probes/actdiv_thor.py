"""MM-E10 — ACTION-DIVERGENCE probe, Thor-side and self-contained.

⭐ THE QUESTION (H-ARCH-ACTINS): is the v7 T1 floor ACTION-INSENSITIVITY — the model
barely uses its action input — or a degenerate rollout in which no action could move
any metric? The banked T1 numbers cannot separate those: both predict the ~1 %
closed-loop-vs-hold-action gap we measured. This holds the SCENE fixed and varies the
ACTION.

⭐ WHY IT RUNS ON THOR AND NOT THE DEV BOX (MM-C12): the dev-box probe helper
hardcodes `sys.path.insert(0, "C:/Users/Admin/tanitad-mirror/stack")` as its LAST
insert, so `tanitad` resolves from a tree whose `v6.py` matches neither the repo nor
G:. `load_trunk_auto` REBUILDS the model from the checkpoint's config, so the code
version is load-bearing. ⇒ **These checkpoints were TRAINED on Thor with Thor's
stack, so Thor's stack is the authoritative code for rebuilding them** — better than
any mirror and better than G:. The probe prints `tanitad.__file__` into its own
output for exactly this reason.

CONTROLS — each must read a KNOWN value or the panel is void:
  C0 IDENTITY  : same actions twice -> spread must be EXACTLY 0. Non-zero means the
                 forward is non-deterministic and every other number is noise.
  C1 SCENE REF : the denominator. If it is ~0 too, the model is degenerate in BOTH
                 directions and the ratio is 0/0 — VOID, not collapse.
⚠️ Action variants are drawn by ROLL, never random permutation: a permutation leaves
~1/B items holding their OWN actions, biasing the numerator DOWN toward the collapse
conclusion. Same defect class caught in the nav control today.
"""
import glob
import io
import json
import os
import sys

import numpy as np
import torch

ST = "/home/nvidia/TanitAD/stack"
sys.path.insert(0, ST)

VAL = os.environ.get("ADV_CORPUS",
                     "/home/nvidia/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl")
ARMS = os.environ.get("ADV_ARMS", "emao14_30k,o14fut30k,postrain30k").split(",")
OUT = os.environ.get("ADV_OUT", "/home/nvidia/staging/actdiv.json")
N_CLIPS = int(os.environ.get("ADV_NCLIPS", "24"))
N_VAR = int(os.environ.get("ADV_NVAR", "8"))
F_MAX = 60
N_STACK = 3
SPEED_SCALE = 30.0


def frames_of(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    raw = d["jpeg_buf"].numpy().tobytes()
    lens = d["jpeg_len"].numpy().tolist()
    off = [0]
    for L in lens:
        off.append(off[-1] + int(L))
    return d, raw, off, len(lens)


def encode_clip(world, path, dev, max_frames):
    from PIL import Image
    d, raw, off, n = frames_of(path)
    n = min(n, max_frames)
    imgs = []
    for i in range(n):
        im = Image.open(io.BytesIO(raw[off[i]:off[i + 1]])).convert("RGB")
        imgs.append(torch.from_numpy(np.asarray(im).copy())
                    .permute(2, 0, 1).float() / 255.0)
    if not imgs or float(imgs[0].abs().mean()) == 0.0:
        raise SystemExit(f"[FATAL] {path} decoded to all-zero frames")
    Z, B = [], 16
    with torch.no_grad():
        for s in range(0, n, B):
            chunk = []
            for i in range(s, min(s + B, n)):
                idx = [max(i - j, 0) for j in range(N_STACK - 1, -1, -1)]
                chunk.append(torch.cat([imgs[k] for k in idx], 0))
            x = torch.stack(chunk)[:, None].to(dev)
            Z.append(world.encode_window(x)[:, 0].float().cpu())
    return torch.cat(Z), d["actions"].float()[:n], d["poses"].float()[:n, 3]


def main():
    import tanitad
    from tanitad.eval.v6_probe_trunk import load_trunk_auto
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clips = sorted(glob.glob(os.path.join(VAL, "*.v2ep.pt")))[:N_CLIPS]
    res = {"_evidence_class": "MEASURED (ours; Thor)", "eval_tier": "T0-DIAGNOSTIC",
           "hypothesis": "H-ARCH-ACTINS",
           "tanitad_imported_from": tanitad.__file__,     # MM-C12: state the tree
           "corpus": VAL, "n_clips": len(clips), "n_action_variants": N_VAR,
           "variant_draw": "roll (never permutation)", "arms": {}}
    for arm in ARMS:
        p = f"/home/nvidia/v7tiny/{arm}/ckpt.pt"
        if not os.path.exists(p):
            print(f"  {arm}: NO CKPT", flush=True); continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        world, _g, _s = load_trunk_auto(ck, dev, ckpt_path=p)
        W = int(world.window)
        horizons = sorted(int(h) for h in world.stack.cfg.predictor.horizons)
        Z, A, V = [], [], []
        for c in clips:
            z, act, spd = encode_clip(world, c, dev, F_MAX)
            n = min(len(z) - W, len(act) - W, len(spd) - W)
            if n < 4:
                continue
            for i in range(0, n, max(1, n // 5)):
                Z.append(z[i:i + W]); A.append(act[i:i + W]); V.append(spd[i])
        if len(Z) < 16:
            print(f"  {arm}: too few windows ({len(Z)})", flush=True); continue
        zs = torch.stack(Z).to(dev)
        aa = torch.stack(A).to(dev)
        vv = torch.stack(V).to(dev).reshape(-1)
        print(f"[{arm}] {zs.shape[0]} windows W={W} horizons={horizons}", flush=True)

        def fwd(a2):
            a3 = torch.cat([a2, (vv / SPEED_SCALE)[:, None, None]
                            .expand(-1, W, -1)], -1)
            with torch.no_grad():
                return world.predictor(zs, a3)

        base = fwd(aa)
        arm_out = {}
        for h in horizons:
            b_h = base[h].float()
            ident = float((fwd(aa)[h].float() - b_h).abs().max())
            preds = [b_h]
            for j in range(1, N_VAR):
                preds.append(fwd(torch.roll(aa, shifts=j, dims=0))[h].float())
            P = torch.stack(preds, 0)
            act_sp = float(P.std(dim=0).mean())
            scene_sp = float(b_h.std(dim=0).mean())
            ratio = act_sp / scene_sp if scene_sp > 1e-9 else None
            arm_out[f"h{h}"] = {
                "action_spread": round(act_sp, 6),
                "scene_spread": round(scene_sp, 6),
                "ratio_action_over_scene": round(ratio, 6) if ratio else None,
                "C0_identity_max_abs_diff": ident, "C0_passes": ident == 0.0,
            }
            print(f"   h={h} action {act_sp:.5f} scene {scene_sp:.5f} "
                  f"RATIO {ratio if ratio is None else round(ratio,4)} "
                  f"C0 {'PASS' if ident == 0.0 else f'FAIL {ident:.2e}'}", flush=True)
        res["arms"][arm] = arm_out
        del world, zs, aa
        torch.cuda.empty_cache()
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print("WROTE", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
