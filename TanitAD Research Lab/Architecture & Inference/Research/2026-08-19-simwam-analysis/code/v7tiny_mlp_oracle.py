"""NONLINEAR oracle — closes the one-directional gap the ridge probe leaves.

⛔ THE INFERENCE ERROR THIS FIXES. `v7tiny_oracle.py` fits a LINEAR ridge and
found the per-tick latent movement explains ~+0.02 (v6F) / ~+0.002 (v7-tiny) of
its own energy. I wrote that up as "noise-dominated / unreachable by ANY
predictor". That is NOT what a linear probe can establish:

    linear oracle BEATS hold   =>  the target IS learnable          (valid)
    linear oracle FAILS        =>  not learnable BY A LINEAR MAP    (all it says)

In 2048 dims, with scene motion that is nonlinear in the features (optical flow,
occlusion, parallax), a linear map is a WEAK lower bound. The honest test of
"is there structure here at all" needs a nonlinear function class.

WHAT THIS RUNS. A small MLP (2 hidden layers, GELU) fit on HELD-OUT clips 0..k/2
and scored on the DISJOINT other half, predicting dz from [z, a]. Same EM, same
episode-cluster bootstrap, same materiality floor as the ridge, so the two
numbers sit in one table.

⭐ THE CONTROLS DECIDE WHETHER THE RUN IS READABLE, and E-DENSE-1 died for want
of them:
  ego     the MLP on ego kinematics -- MUST score high, or the harness is broken
  pixel   the same MLP on raw pooled frames -- the perception floor
  shuffle dz shuffled ACROSS TIME within a clip. Destroys the temporal relation
          while preserving every marginal. ⭐ MUST score ~0. If a "signal"
          survives shuffling, it is leakage or an estimator artifact, not
          dynamics.

⚠️ Early-stopped on a validation split carved from the FIT clips only, never on
the scored clips.

TIER: T0-DIAGNOSTIC.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, str(SP))
sys.path.insert(0, str(Path(r"G:\Meine Ablage\SayBouBase\raw\Projects"
                            r"\TanitAD\stack")))
HELD = SP / "sp2/cache/v7tiny-heldout24-w120-256x640cyl"
MATERIAL = 0.01


def pooled_frames(path, n):
    d = torch.load(path, map_location="cpu", weights_only=False)
    raw = d["jpeg_buf"].numpy().tobytes()
    off = np.concatenate([[0], np.cumsum(d["jpeg_len"].tolist())]).astype(
        np.int64)
    rows = []
    for i in range(min(n, len(off) - 1)):
        im = Image.open(io.BytesIO(raw[off[i]:off[i + 1]])).convert("L")
        rows.append(np.asarray(im.resize((40, 16), Image.BOX),
                               dtype=np.float32).ravel() / 255.0)
    px = np.stack(rows)
    if float(np.abs(px).mean()) == 0.0:
        raise SystemExit(f"[FATAL] {path.name} decoded to all-zero frames")
    return px


def fit_mlp(Xtr, Ytr, Xva, Yva, dev, hidden=512, epochs=200, seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.GELU(),
                        nn.Linear(hidden, hidden), nn.GELU(),
                        nn.Linear(hidden, Ytr.shape[1])).to(dev)
    # ⭐ last layer down-scaled: the SAME identity-start discipline the
    # predictor fix applies, so the probe does not spend its budget shrinking an
    # oversized initial output and mistake that for "cannot learn".
    with torch.no_grad():
        net[-1].weight.mul_(1e-3)
        net[-1].bias.mul_(1e-3)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.as_tensor(Xtr, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(Ytr, dtype=torch.float32, device=dev)
    xv = torch.as_tensor(Xva, dtype=torch.float32, device=dev)
    yv = torch.as_tensor(Yva, dtype=torch.float32, device=dev)
    best, best_sd, bad = 9e9, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xt), device=dev)
        for s in range(0, len(xt), 256):
            j = perm[s:s + 256]
            opt.zero_grad()
            nn.functional.mse_loss(net(xt[j]), yt[j]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(nn.functional.mse_loss(net(xv), yv))
        if v < best - 1e-9:
            best, bad = v, 0
            best_sd = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 20:
                break
    if best_sd:
        net.load_state_dict(best_sd)
    net.eval()
    return net


def em_ci(errs, movs, seed=0, n=4000):
    errs, movs = np.array(errs), np.array(movs)
    e = 1.0 - errs.sum() / movs.sum()
    rng = np.random.default_rng(seed)
    bs = np.empty(n)
    for b in range(n):
        j = rng.integers(0, len(errs), len(errs))
        bs[b] = 1.0 - errs[j].sum() / movs[j].sum()
    return float(e), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def run(states, actions, dev, tag, seed=0):
    k = len(states)
    half = k // 2
    fit_ix, te_ix = list(range(half)), list(range(half, k))
    def rows(ix):
        X, Y = [], []
        for i in ix:
            s, a = states[i], actions[i]
            n = min(len(s), len(a)) - 1
            X.append(np.concatenate([s[:n], a[:n]], 1))
            Y.append(s[1:n + 1] - s[:n])
        return np.concatenate(X), np.concatenate(Y)
    # validation split comes from the FIT clips only
    nv = max(1, len(fit_ix) // 4)
    Xtr, Ytr = rows(fit_ix[:-nv])
    Xva, Yva = rows(fit_ix[-nv:])
    xm, xs = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    net = fit_mlp((Xtr - xm) / xs, Ytr, (Xva - xm) / xs, Yva, dev, seed=seed)
    errs, movs = [], []
    with torch.no_grad():
        for i in te_ix:
            s, a = states[i], actions[i]
            n = min(len(s), len(a)) - 1
            Xi = (np.concatenate([s[:n], a[:n]], 1) - xm) / xs
            Yi = s[1:n + 1] - s[:n]
            p = net(torch.as_tensor(Xi, dtype=torch.float32,
                                    device=dev)).cpu().numpy()
            errs.append(float(((Yi - p) ** 2).sum()))
            movs.append(float((Yi ** 2).sum()))
    return em_ci(errs, movs, seed=seed)


def main() -> int:
    ap = argparse.ArgumentParser(description="nonlinear oracle")
    ap.add_argument("--arm", default="fixed")
    ap.add_argument("--v6f", action="store_true")
    ap.add_argument("--v6f-ckpt", default=str(SP / "ckpt/v6F_sw_step020000.fp16.pt"))
    ap.add_argument("--v6f-config", default=str(SP / "sp2/v6F_config.json"))
    ap.add_argument("--clips", type=int, default=24)
    ap.add_argument("--frames-per-clip", type=int, default=120)
    ap.add_argument("--out", default=str(SP / "v7tiny_mlp_oracle.json"))
    a = ap.parse_args()

    import v7tiny_g2 as G
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.v6f:
        import e_pred_probe as E
        world, step = E.load_world(Path(a.v6f_ckpt), Path(a.v6f_config), dev)
        name = f"v6F@{step}"
    else:
        world, step = G.load_arm(a.arm, dev)
        name = f"v7tiny-{a.arm}@{step}"

    paths = sorted(HELD.glob("*.v2ep.pt"))[:a.clips]
    Z, AC, PO, PX = [], [], [], []
    for n, p in enumerate(paths, 1):
        z, act, _spd = G.encode_clip(world, p, dev, a.frames_per_clip)
        d = torch.load(p, map_location="cpu", weights_only=False)
        m = min(len(z), len(d["poses"]), len(act))
        Z.append(z.numpy()[:m].astype(np.float32))
        AC.append(act.numpy()[:m].astype(np.float32))
        PO.append(d["poses"].numpy()[:m].astype(np.float32))
        PX.append(pooled_frames(p, m)[:m])
        print(f"    [{n}/{len(paths)}] {p.name[:10]} {m} frames", flush=True)

    rng = np.random.default_rng(0)
    Zsh = []
    for z in Z:                       # shuffle dz across time, keep marginals
        dz = z[1:] - z[:-1]
        perm = rng.permutation(len(dz))
        s = np.concatenate([z[:1], z[:1] + np.cumsum(dz[perm], 0)])
        Zsh.append(s.astype(np.float32))

    panels = {"latent": (Z, AC), "pixel (floor)": (PX, AC),
              "ego (must be high)": (PO, AC),
              "shuffled dz (must be ~0)": (Zsh, AC)}
    res = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC", "model": name, "parity": False,
           "n_clips": len(Z), "probe": "MLP 2x512 GELU, early-stopped on a "
                                       "split of the FIT clips only",
           "materiality_floor": MATERIAL, "panels": {}}
    print(f"\n  NONLINEAR ORACLE ({name}) — episode-disjoint, {len(Z)} clips")
    print(f"  {'representation':<26}{'dim':>6}{'EM':>10}  {'CI95':<24}verdict")
    print("  " + "-" * 82)
    for label, (S, A) in panels.items():
        e, lo, hi = run(S, A, dev, label)
        v = ("PREDICTABLE" if lo > 0.10 else
             "weak" if lo > MATERIAL else "== hold")
        res["panels"][label] = {"dim": int(S[0].shape[1]),
                                "em": round(e, 6),
                                "ci95": [round(lo, 6), round(hi, 6)],
                                "verdict": v}
        print(f"  {label:<26}{S[0].shape[1]:>6}{e:>+10.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}]   {v}")

    ego_ok = res["panels"]["ego (must be high)"]["ci95"][0] > 0.10
    sh_ok = res["panels"]["shuffled dz (must be ~0)"]["ci95"][0] < MATERIAL
    lat = res["panels"]["latent"]
    res["controls_pass"] = bool(ego_ok and sh_ok)
    res["verdict"] = (
        (f"CONTROLS PASS. Nonlinear EM on the latent = {lat['em']:+.4f} "
         f"(CI {lat['ci95']}). "
         + ("⇒ there IS nonlinear structure in the per-tick movement that the "
            "trained predictor is not capturing — the target is learnable and "
            "the predictor is underperforming it."
            if lat["ci95"][0] > MATERIAL else
            "⇒ even a nonlinear probe cannot beat HOLD, which is much stronger "
            "evidence than the ridge alone that the per-tick movement of THIS "
            "representation carries little learnable structure."))
        if ego_ok and sh_ok else
        f"⛔ CONTROLS FAILED (ego_ok={ego_ok}, shuffle_ok={sh_ok}) — the panel "
        f"is UNREADABLE and no latent number from it is admissible.")
    print(f"\n  VERDICT: {res['verdict']}")
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
