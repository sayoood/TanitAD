"""MM-E17 — WHERE DOES THE ACTION SIGNAL DIE? A stage-by-stage attenuation trace.

⭐ THE QUESTION, pre-committed by MM-E11's O1-INERT outcome: *"the defect is NOT the
missing objective — it is architectural (how actions enter the predictor), and the next
lever is the conditioning path itself, not another loss weight."* This localises it.

Actions enter as `act_emb(actions)` -> FiLM `cond` -> per-block modulation -> `h_last`
-> head. MM-E10 measured the END of that chain: the action moves the prediction 0.4-0.6 %
as much as the scene. This measures EVERY LINK, so the answer is a stage rather than a
verdict.

⛔ TWO HYPOTHESES ALREADY ELIMINATED, CHEAPLY, BEFORE WRITING THIS:
  * "the FiLM never left zero-init" — REFUTED. Measured on o1ctrl30k's 30k ckpt:
    |W| 2.97 / 3.06 / 5.44 across the three blocks. It trained.
  * "O1 was simply off" — REFUTED by MM-E11 itself (ratio FELL 0.40x with O1 on).

THE MEASUREMENT. Hold the SCENE fixed, vary the ACTION (by ROLL, never permutation —
a permutation leaves ~1/B items holding their own actions and biases the numerator DOWN,
toward the conclusion). At each stage compute the spread across action variants, and
divide by the spread the SCENE produces at that same stage. That ratio is comparable
across stages because both numerator and denominator are measured in that stage's own
units — a raw norm is not.

  stage_ratio(s) = spread_over_actions(s) / spread_over_scenes(s)

A ratio that is healthy early and collapses at one stage NAMES that stage. A ratio that
is low everywhere says the action embedding itself carries little, which is a data
question, not an architecture one.

CONTROLS — each must read a KNOWN value or the panel is VOID:
  C0 IDENTITY : same actions twice -> spread EXACTLY 0 at every stage. Non-zero means
                the forward is non-deterministic and every number here is noise.
  C1 SCENE    : the denominators must be LARGE. If a stage's scene spread is ~0 too,
                that stage is degenerate in BOTH directions and its ratio is 0/0 — VOID
                at that stage, reported as None rather than as a collapse.
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

VAL = os.environ.get("CP_CORPUS",
                     "/home/nvidia/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl")
ARMS = os.environ.get("CP_ARMS", "postrain30k,o1ctrl30k").split(",")
OUT = os.environ.get("CP_OUT", "/home/nvidia/staging/condpath.json")
N_CLIPS = int(os.environ.get("CP_NCLIPS", "24"))
N_VAR = int(os.environ.get("CP_NVAR", "8"))
F_MAX, N_STACK, SPEED_SCALE = 60, 3, 30.0


def encode_clip(world, path, dev, max_frames):
    from PIL import Image
    d = torch.load(path, map_location="cpu", weights_only=False)
    raw = d["jpeg_buf"].numpy().tobytes()
    lens = d["jpeg_len"].numpy().tolist()
    off = [0]
    for L in lens:
        off.append(off[-1] + int(L))
    n = min(len(lens), max_frames)
    imgs = []
    for i in range(n):
        im = Image.open(io.BytesIO(raw[off[i]:off[i + 1]])).convert("RGB")
        imgs.append(torch.from_numpy(np.asarray(im).copy())
                    .permute(2, 0, 1).float() / 255.0)
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
           "hypothesis": "MM-E17 (where does the action signal attenuate?)",
           "tanitad_imported_from": tanitad.__file__,
           "variant_draw": "roll (never permutation)",
           "n_clips": len(clips), "n_variants": N_VAR, "arms": {}}

    for arm in ARMS:
        p = f"/home/nvidia/v7tiny/{arm}/ckpt.pt"
        if not os.path.exists(p):
            print(f"  {arm}: NO CKPT", flush=True)
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        world, _g, _s = load_trunk_auto(ck, dev, ckpt_path=p)
        pred = world.predictor
        W = int(world.window)

        Z, A, V = [], [], []
        for c in clips:
            z, act, spd = encode_clip(world, c, dev, F_MAX)
            n = min(len(z) - W, len(act) - W, len(spd) - W)
            if n < 4:
                continue
            for i in range(0, n, max(1, n // 5)):
                Z.append(z[i:i + W]); A.append(act[i:i + W]); V.append(spd[i])
        if len(Z) < 16:
            print(f"  {arm}: too few windows", flush=True)
            continue
        zs, aa = torch.stack(Z).to(dev), torch.stack(A).to(dev)
        vv = torch.stack(V).to(dev).reshape(-1)

        taps: dict[str, torch.Tensor] = {}
        hooks = []

        def mk(name):
            def fn(_m, _i, o):
                taps[name] = (o[0] if isinstance(o, tuple) else o).detach().float()
            return fn

        hooks.append(pred.act_emb.register_forward_hook(mk("1_act_emb")))
        for bi, blk in enumerate(pred.blocks):
            if hasattr(blk, "film"):
                hooks.append(blk.film.register_forward_hook(mk(f"2_film{bi}")))
            hooks.append(blk.register_forward_hook(mk(f"3_block{bi}")))
        hooks.append(pred.norm.register_forward_hook(mk("4_h_last_norm")))

        def fwd(a2):
            a3 = torch.cat([a2, (vv / SPEED_SCALE)[:, None, None]
                            .expand(-1, W, -1)], -1)
            taps.clear()
            with torch.no_grad():
                out = pred(zs, a3)
            return {k: v.clone() for k, v in taps.items()}, out

        base_taps, base_out = fwd(aa)
        # C1 denominators: spread ACROSS the batch (different scenes), per stage.
        scene = {k: float(v.reshape(v.shape[0], -1).std(0).mean())
                 for k, v in base_taps.items()}
        scene["5_pred_h1"] = float(base_out[1].float()
                                   .reshape(base_out[1].shape[0], -1).std(0).mean())

        # C0: identity — same actions twice must give EXACTLY zero spread.
        id_taps, id_out = fwd(aa)
        c0 = max(float((id_taps[k] - base_taps[k]).abs().max()) for k in base_taps)
        c0 = max(c0, float((id_out[1] - base_out[1]).abs().max()))

        stacks: dict[str, list[torch.Tensor]] = {k: [v] for k, v in base_taps.items()}
        stacks["5_pred_h1"] = [base_out[1].float()]
        for j in range(1, N_VAR):
            t, o = fwd(torch.roll(aa, shifts=j, dims=0))
            for k, v in t.items():
                stacks[k].append(v)
            stacks["5_pred_h1"].append(o[1].float())

        rows = {}
        for k in sorted(stacks):
            P = torch.stack(stacks[k], 0)
            act_sp = float(P.std(dim=0).mean())
            sc = scene.get(k, 0.0)
            rows[k] = {"action_spread": round(act_sp, 8),
                       "scene_spread": round(sc, 8),
                       "ratio": round(act_sp / sc, 8) if sc > 1e-9 else None}
        res["arms"][arm] = {"C0_max_abs_diff": c0, "C0_passes": c0 == 0.0,
                            "n_windows": int(zs.shape[0]), "stages": rows}
        print(f"\n[{arm}] n={zs.shape[0]} C0={'PASS' if c0 == 0.0 else f'FAIL {c0:.2e}'}",
              flush=True)
        for k in sorted(rows):
            r = rows[k]
            rr = "VOID(0/0)" if r["ratio"] is None else f"{r['ratio']:.6f}"
            print(f"   {k:<18} action {r['action_spread']:.6f}  "
                  f"scene {r['scene_spread']:.6f}  RATIO {rr}", flush=True)
        for h in hooks:
            h.remove()
        del world, zs, aa
        torch.cuda.empty_cache()

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print("\nWROTE", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
