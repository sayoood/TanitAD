"""E-PRED-1 — does v6's PREDICTOR model dynamics at all? A label-free test.

⛔⛔ THE GAP THIS CLOSES, and it was mine. Every probe in this programme has read
the ENCODER — `z_t`, or a concatenation of several `z`'s. But
`V6Stack.encode_window` folds the window into the BATCH dimension
(`v6.py:4840`), so the encoder is strictly PER-FRAME and `z_t` contains no
history at all. v6's history lives in `predictor_op`, which consumes W = 6
latents plus actions — and **the predictor has never been probed.**

That matters twice over:
  * it is the actual WORLD-MODEL claim. An encoder is perception; a predictor is
    the model of dynamics.
  * `E_V6MOVE` measured `predictor_op` as the ONLY module whose LR-normalised
    weight movement is RISING (x1.29) while the encoder decays (x0.55). The one
    module still actively learning is the one nobody has measured.

⭐ THE TEST NEEDS NO LABELS, WHICH IS WHY IT IS TRUSTWORTHY HERE. Nine nulls rest
on external targets that v6's latent may simply not encode. This asks only
whether the predictor does ITS OWN JOB out-of-sample:

      given z_{t-W+1..t} and the true actions, does zhat_{t+k} reach z_{t+k}?

scored against the ONE baseline that makes the number meaningful:

  ⛔ `hold`   zhat := z_t — predict NO CHANGE.
              A predictor that cannot beat this has learned no dynamics. A
              driving latent moves only ~1.12 % of its magnitude at k = 1
              (MEASURED, E-TRUNK-3), so holding is a STRONG baseline and raw MSE
              alone would flatter the model badly.

PRIMARY METRIC — **explained movement**:

      EM = 1 - ||zhat_{t+k} - z_{t+k}||^2 / ||z_t - z_{t+k}||^2

  EM > 0   the predictor explains some of the latent's actual movement
  EM = 0   exactly as good as holding — no dynamics learned
  EM < 0   WORSE than holding

⚠️ EM is scale-free, which is what makes it comparable across checkpoints and
horizons. But a tiny denominator would make it loud and meaningless, so
`move_frac` (how far the latent actually travels, relative to its magnitude) is
reported beside every row: an EM whose denominator is negligible is UNDEFINED,
not a result.

⛔⛔ STATUS 2026-08-22: THIS PROBE IS NOT YET TRUSTWORTHY. DO NOT QUOTE IT.

First run returned explained_movement = -65,720 (mse_pred 247 vs mse_hold
0.0038). That is not a finding about v6, it is a mis-wiring in THIS file, and
the trainer's own log refutes the pessimistic reading directly:

    step   1950  o5_step1 0.32321   o3_loss 0.04475
    step   7950  o5_step1 0.37874   o3_loss 0.01256
    step  13950  o5_step1 0.24792   o3_loss 0.00856
    step  19950  o5_step1 0.17673   o3_loss 0.00431
    step  25950  o5_step1 0.13664   o3_loss 0.00250

`o5_step1` IS the predictor's own 1-step error and it falls monotonically after
the early rise. A predictor that had learned no dynamics would not do that.

KNOWN DEFECTS IN THIS FILE, both found before any number was published:
  1. ⛔ HORIZON KEYS. `PredictorConfig.horizons` is (1, 2, 4) — NOT (1, 2, 3).
     The first cut asked for k=3 and SILENTLY fell back to `pred[max(pred)]`
     (key 4). Silent substitution of a different horizon is exactly the class of
     bug this programme keeps retracting; there must be no fallback.
  2. ⛔ WRONG INTERFACE. The trainer never calls `predictor_op(...)` directly.
     It calls `rollout_transitions` (metric_dynamics.py:247), whose `_step` uses
     ONLY head 1 and rolls AUTOREGRESSIVELY, feeding z_hat back into the window.
     Heads 2 and 4 are not the rollout and must not be read as it.
  3. ⚠️ UNRESOLVED SCALE. Head 1's output has norm^2 ~212 against a banked-latent
     norm^2 of ~0.70. `out[k] = z_t + delta` with `delta` produced from a
     LayerNorm'd `h_last`, so delta is O(1) while z_t is O(0.018). Either the
     banked `cells` are not the scale the predictor trains on, or the roll must
     be entered differently. UNTIL THIS IS EXPLAINED, NO NUMBER FROM THIS FILE
     IS ADMISSIBLE.

NEXT STEP TO FIX IT: round-trip check — encode a 256x640 frame LIVE through the
loaded trunk and compare to the banked `cells` for that same (clip_id,
frame_idx). If they differ, the cache is the problem; if they match, the roll
entry is.

TIER: T0-DIAGNOSTIC. Dev-box only; Thor untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, str(SP))
sys.path.insert(0, str(Path.cwd() / "stack"))

CKPT = SP / "ckpt"
#: ⚠️ MUST match PredictorConfig.horizons — (1, 2, 4) for config E.
#: Never fall back to another key when one is missing (defect 1).
KS = (1, 2, 4)


def load_world(ckpt: Path, config_json: Path | None, device):
    from sp_common import merge_run_args, read_fp16_snapshot
    from tanitad.eval.v6_probe_trunk import load_trunk_auto
    sd, run_args, art_step, prov = read_fp16_snapshot(
        str(ckpt), str(config_json) if config_json else None)
    merged = vars(merge_run_args(run_args))
    ck = {"stack": sd, "config": {"args": merged}, "step": art_step}
    world, _g, base_step = load_trunk_auto(ck, device, ckpt_path=str(ckpt))
    if any(p.requires_grad for p in world.parameters()):
        raise SystemExit("[FATAL] trunk is not frozen")
    return world, base_step


def explained_movement(zhat, z_fut, z_now) -> dict:
    """EM, plus everything needed to judge whether EM is meaningful at all."""
    err = ((zhat - z_fut) ** 2).sum(-1)
    mov = ((z_now - z_fut) ** 2).sum(-1)
    mag = (z_now ** 2).sum(-1)
    ok = mov > 0
    em = 1.0 - err[ok].sum() / mov[ok].sum()
    return {"explained_movement": round(float(em), 4),
            "mse_pred": round(float(err.mean()), 6),
            "mse_hold": round(float(mov.mean()), 6),
            "move_frac": round(float(np.sqrt(mov.mean()
                                             / max(mag.mean(), 1e-12))), 5),
            "n": int(len(zhat))}


def main() -> int:
    ap = argparse.ArgumentParser(description="E-PRED-1")
    ap.add_argument("--ckpt", default=str(CKPT / "v6F_sw_step020000.fp16.pt"))
    ap.add_argument("--config-json", default=None)
    ap.add_argument("--cache", default=str(SP / "sp2/lewm_frames"))
    ap.add_argument("--max-windows", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out", default=str(SP / "e_pred_probe.json"))
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world, step = load_world(Path(a.ckpt),
                             Path(a.config_json) if a.config_json else None,
                             dev)
    stack = world.stack
    W = int(getattr(world, "window", 6))
    print(f"  trunk FROZEN @ step {step} · d_op {world.state_dim} · window {W}",
          flush=True)

    # ⭐ NO ENCODER FORWARD. `cache_tok20000_s4` already holds v6's PER-FRAME
    # operative latents (`cells`, 16 x 128 = d_op 2048) for the 5,617 probe rows
    # at the geometry the trunk was trained on. The LeWM frame cache is
    # 128x320 and `encoder.py:387` correctly REFUSES it — the joint APE is
    # sized for 256x640 — so re-encoding there would have been wrong, not merely
    # slower. Reading banked latents is both cheaper and the only correct path.
    import e_detect_prep as P
    obj = torch.load(SP / "sp2/cache_tok20000_s4/latents.pt",
                     map_location="cpu", weights_only=False)
    rows = [r for r in obj["rows"] if r.get("cells") is not None]
    keys = [(r["clip_id"], int(r["frame_idx"])) for r in rows]
    Z = torch.stack([r["cells"].reshape(-1).float() for r in rows])   # [N,d_op]
    del obj
    print(f"  banked latents {tuple(Z.shape)} over "
          f"{len({k[0] for k in keys})} clips", flush=True)

    # ⚠️ THE PREDICTOR TAKES action_dim=3, NOT 2. v6 trains on the "speed-append"
    # format: `_lift3(actions2, v0)` in train_v6_staged.py:2490 concatenates
    # v0 / SPEED_SCALE as a THIRD channel, broadcast across the window. Feeding
    # the raw 2-channel (steer, accel) is caught by `validate_operative_inputs`
    # — correctly — and reproducing the lift here is what makes this probe read
    # the predictor as it was actually trained.
    from tanitad.models.flagship_v15 import SPEED_SCALE
    EPS = SP / "sp2/cache/slotprobe-lead130-w120-256x640cyl"
    act_by, spd_by = {}, {}
    for cid in sorted({k[0] for k in keys}):
        d = torch.load(EPS / f"{cid}.v2ep.pt", map_location="cpu",
                       weights_only=False)
        act_by[cid] = d["actions"].float()          # [n, 2] (steer, accel)
        spd_by[cid] = d["poses"].float()[:, 3]      # [n]    speed m/s
    A2 = torch.stack([act_by[c][min(f, len(act_by[c]) - 1)] for c, f in keys])
    V0 = torch.stack([spd_by[c][min(f, len(spd_by[c]) - 1)] for c, f in keys])
    print(f"  actions {tuple(A2.shape)} + v0 (mean {float(V0.mean()):.2f} m/s) "
          f"-> action_dim 3 via _lift3, SPEED_SCALE={SPEED_SCALE}", flush=True)

    # windows of W+K CONSECUTIVE banked rows inside one clip. ⚠️ The banked rows
    # are already stride-4 in FRAME index, so consecutive rows are one
    # predictor tick apart — `--stride` does not apply here and is ignored.
    pos = {}
    for i, (c, f) in enumerate(keys):
        pos.setdefault(c, []).append((f, i))
    starts = []
    need = W + max(KS)
    for c, lst in pos.items():
        lst.sort()
        idx = [i for _, i in lst]
        fr = [f for f, _ in lst]
        for j in range(len(idx) - need):
            if fr[j + need - 1] - fr[j] == (need - 1) * (fr[1] - fr[0]):
                starts.append(idx[j:j + need])
    starts = np.array(starts)
    rng = np.random.default_rng(0)
    if len(starts) > a.max_windows:
        starts = starts[np.sort(rng.choice(len(starts), a.max_windows,
                                           replace=False))]
    print(f"  {len(starts):,} contiguous windows of {need}, horizons {KS}",
          flush=True)

    out = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC",
           "question": "does v6's PREDICTOR beat the hold baseline — i.e. has "
                       "it learned any dynamics?",
           "metric": "explained_movement = 1 - ||zhat-z_fut||^2 / "
                     "||z_now-z_fut||^2 ; >0 beats hold, 0 == hold, <0 worse",
           "baseline": "hold: zhat := z_t (predict NO CHANGE)",
           "ckpt": Path(a.ckpt).name, "step": int(step), "window": W,
           "stride_frames": a.stride, "n_windows": int(len(starts)),
           "horizons": {}}

    t0 = time.time()
    acc = {k: {"zhat": [], "zfut": [], "znow": []} for k in KS}
    B = 32
    with torch.no_grad():
        for i in range(0, len(starts), B):
            win = starts[i:i + B]                       # [b, W+K] row indices
            z_all = Z[torch.from_numpy(win)].to(dev)    # [b, W+K, d_op]
            a2 = A2[torch.from_numpy(win[:, :W])].to(dev)          # [b, W, 2]
            # v0 is the WINDOW's speed, broadcast — exactly `_lift3`
            v = (V0[torch.from_numpy(win[:, 0])].to(dev)
                 / SPEED_SCALE)[:, None, None].expand(-1, W, -1)
            acts = torch.cat([a2, v], dim=-1)                       # [b, W, 3]
            pred = stack.predictor_op(z_all[:, :W], acts)
            for k in KS:
                if isinstance(pred, dict):
                    if k not in pred:
                        raise SystemExit(
                            f"[FATAL] horizon {k} absent; predictor has "
                            f"{sorted(pred)}. Refusing to substitute another "
                            f"head — that is defect 1 in this file's header.")
                    zk = pred[k]
                else:
                    zk = pred
                acc[k]["zhat"].append(zk.float().cpu().numpy())
                acc[k]["zfut"].append(
                    z_all[:, W + k - 1].float().cpu().numpy())
                acc[k]["znow"].append(z_all[:, W - 1].float().cpu().numpy())
            if i % (B * 16) == 0:
                print(f"    {i:,}/{len(starts):,} ({time.time() - t0:.0f}s)",
                      flush=True)

    for k in KS:
        r = explained_movement(np.concatenate(acc[k]["zhat"]),
                               np.concatenate(acc[k]["zfut"]),
                               np.concatenate(acc[k]["znow"]))
        out["horizons"][f"k{k}"] = r
        v = ("BEATS hold" if r["explained_movement"] > 0.02 else
             "WORSE than hold" if r["explained_movement"] < -0.02
             else "== hold (NO dynamics learned)")
        print(f"  k={k}  EM {r['explained_movement']:+.4f}  "
              f"mse_pred {r['mse_pred']:.5f} vs hold {r['mse_hold']:.5f}  "
              f"move_frac {r['move_frac']:.4f}  -> {v}", flush=True)
    out["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
