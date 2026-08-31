"""MM-E10 — ACTION-DIVERGENCE probe, dev-box variant of ``actdiv_thor.py``.

⭐ THE QUESTION (H-ARCH-ACTINS): is the v7 T1 floor ACTION-INSENSITIVITY — the model
barely uses its action input — or a degenerate rollout in which no action could move
any metric? This holds the SCENE fixed and varies the ACTION. It is the instrument
behind MM-E19's PRIMARY criterion: the incumbent ``postrain30k`` h1
``ratio_action_over_scene`` = **0.005947** (MEASURED ours; Thor,
``actdiv_result.json``), and HORIZON-WORKS requires **≥10×** that (≥ 0.0595).

⚠️ WHY A LOCAL VARIANT EXISTS AT ALL. ``actdiv_thor.py`` runs on Thor because
Thor's stack is the tree these checkpoints were TRAINED with (MM-C12). Tonight
Thor is TRAINING the k=60 arm and must not take eval load, so the read runs on
the dev box against the pulled checkpoint — through a stack VERIFIED against
Thor's (see ``_stackresolve.py``: the load-bearing modules are identical or a
nav-gated superset, and the STRICT state-dict load is the runtime check).
⚠️ Cross-device note for the record: the banked incumbent number is Thor-GPU;
this variant re-measures the incumbent ON THE SAME DEVICE as the k=60 arm in the
same invocation, so the paired comparison never crosses devices — the banked
Thor number is a reference, not the control.

THE COMPUTE SECTION IS ``actdiv_thor.py``'s VERBATIM (windows sampling, roll
variants, C0/C1 controls, spread ratio). Only path/stack resolution and the
MM-C12 preflight/refusal differ. Two implementations of one computation is the
``within_clip_r`` trap — so if the compute here ever needs to change, change
``actdiv_thor.py`` and re-port.

CONTROLS — each must read a KNOWN value or the panel is void:
  C0 IDENTITY  : same actions twice -> spread must be EXACTLY 0. Non-zero means the
                 forward is non-deterministic and every other number is noise.
  C1 SCENE REF : the denominator. If it is ~0 too, the model is degenerate in BOTH
                 directions and the ratio is 0/0 — VOID, not collapse.
⚠️ Action variants are drawn by ROLL, never random permutation: a permutation leaves
~1/B items holding their OWN actions, biasing the numerator DOWN toward the collapse
conclusion.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import pathlib
import sys

import numpy as np
import torch

SP = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SP))

from _stackresolve import preflight_stack, resolve_stack  # noqa: E402

F_MAX = 60
N_STACK = 3
SPEED_SCALE = 30.0


def resolve_config(argv=None, env=None) -> argparse.Namespace:
    env = os.environ if env is None else env
    ap = argparse.ArgumentParser(description="MM-E10 action-divergence (local)")
    ap.add_argument("--corpus", default=env.get("ADV_CORPUS", ""),
                    help="dir of *.v2ep.pt clips (default: <assets>/sp2/cache/"
                         "physicalai-val-w120-256x640cyl — the local prefix of "
                         "Thor's physicalai-val-0c5f7dac3b11-w120-256x640cyl)")
    ap.add_argument("--arms",
                    default=env.get("ADV_ARMS", "k60clip05p30k,postrain30k"))
    ap.add_argument("--out", default=env.get("ADV_OUT", ""),
                    help="default: <script dir>/actdiv_local.json")
    ap.add_argument("--nclips", type=int, default=int(env.get("ADV_NCLIPS", "24")),
                    help="24 = the banked Thor instrument's n")
    ap.add_argument("--nvar", type=int, default=int(env.get("ADV_NVAR", "8")))
    ap.add_argument("--assets", default=env.get("ADV_ASSETS", ""),
                    help="dir holding v7tiny_<arm>/ckpt.pt (default: script dir)")
    ap.add_argument("--stack", default=env.get("ADV_STACK", ""))
    ap.add_argument("--device", default=env.get("ADV_DEVICE", ""),
                    choices=("", "cuda", "cpu"),
                    help="default: cuda if available else cpu (the Thor "
                         "instrument's rule)")
    ap.add_argument("--preflight-only", action="store_true")
    a = ap.parse_args(argv)
    a.assets = pathlib.Path(a.assets) if a.assets else SP
    a.corpus = (pathlib.Path(a.corpus) if a.corpus
                else a.assets / "sp2/cache/physicalai-val-w120-256x640cyl")
    a.out = pathlib.Path(a.out) if a.out else SP / "actdiv_local.json"
    return a


# ---- compute section: actdiv_thor.py VERBATIM --------------------------------

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


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = resolve_config(argv)
    stack = resolve_stack(cfg.stack or None)
    tanitad_file = preflight_stack(stack)
    if cfg.preflight_only:
        print(f"  [preflight] OK — stack {stack}")
        return 0
    from tanitad.eval.v6_probe_trunk import load_trunk_auto
    dev = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    N_VAR = int(cfg.nvar)
    clips = sorted(glob.glob(os.path.join(str(cfg.corpus), "*.v2ep.pt")))[:cfg.nclips]
    if len(clips) < cfg.nclips:
        print(f"[REFUSED] corpus {cfg.corpus} holds {len(clips)} clips, "
              f"--nclips {cfg.nclips} requested — n would differ from the "
              f"banked instrument (same corpus, same n, same instrument).",
              file=sys.stderr)
        return 2
    res = {"_evidence_class": f"MEASURED (ours; dev-box {dev})",
           "eval_tier": "T0-DIAGNOSTIC",
           "hypothesis": "H-ARCH-ACTINS",
           "tanitad_imported_from": tanitad_file,     # MM-C12: state the tree
           "banked_thor_reference": {
               "postrain30k_h1_ratio": 0.005947,
               "source": "actdiv_result.json (MEASURED ours; Thor; 24 clips of "
                         "physicalai-val-0c5f7dac3b11-w120-256x640cyl)"},
           "corpus": str(cfg.corpus), "n_clips": len(clips),
           "n_action_variants": N_VAR,
           "variant_draw": "roll (never permutation)", "arms": {}}
    for arm in str(cfg.arms).split(","):
        p = str(cfg.assets / f"v7tiny_{arm}" / "ckpt.pt")
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
        arm_out = {"step": int(_s), "n_windows": int(zs.shape[0])}
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
    with open(cfg.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("WROTE", str(cfg.out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
