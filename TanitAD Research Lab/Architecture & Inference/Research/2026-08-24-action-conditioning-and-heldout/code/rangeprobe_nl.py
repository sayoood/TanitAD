"""E-DEC-36 — IS RANGE NONLINEARLY DECODABLE? (the honest version of E-DEC-35)

⛔ WHY E-DEC-35 IS NOT ENOUGH, IN ITS OWN WORDS. It is a RIDGE probe. The standing
rule: *a linear oracle FAILING proves only "not learnable by a linear map"*. And
monocular metric range is a **strongly nonlinear** function of apparent size and
image position — roughly `range ∝ 1/apparent_height` — so a linear null is close
to the weakest possible evidence about it. E-DEC-35 established that no column
carries range or closing in a LINEARLY-accessible form. This asks the question
that actually matters.

⭐⭐ THE CONTROL THAT MAKES A NONLINEAR PROBE READABLE, AND WITHOUT WHICH IT IS
WORSE THAN THE LINEAR ONE. A nonlinear probe with enough capacity can fit
anything, including clip identity — and clip identity is highly predictive of a
clip's mean range. So the panel carries, on IDENTICAL rows, folds and capacity:

    TRUE targets       the quantity of interest
    TIME-SHUFFLED      the SAME targets permuted WITHIN each clip. This destroys
                       the frame-to-frame correspondence while preserving each
                       clip's marginal distribution exactly. ⛔ Any score the
                       shuffled version achieves is what the probe can get from
                       clip-level statistics ALONE — i.e. LEAKAGE, not dynamics.
                       **The readable quantity is TRUE minus SHUFFLED**, never
                       the raw R².
    constant           reads exactly 0.0000 by construction
    pixels (floor)     raw input; a representation not beating it added nothing

⚠️ EVERY hyper-parameter (PCA basis, standardisation, early stopping) is fit on
the FIT split only. The scored clip is scored, never tuned on. `n` and `d` are
printed. (C-probe: four separate failures in one afternoon came from tuning on
the scored data or from an unreadable normalisation.)

T0-DIAGNOSTIC. Held-out, lead-matched. MEASURED (ours; dev-box RTX 4060).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")
CACHE = SP / "rangeprobe_cache.npz"
OUT = Path(os.environ.get("SPD_OUT", str(SP / "rangeprobe_nl.json")))
D_EFF, EPOCHS, SEED = 128, 300, 0


def mlp_fold(Xtr, ytr, Xte, yte, d_eff=D_EFF, seed=SEED):
    """One LOEO fold: PCA + standardise + a small MLP, ALL fit on Xtr only."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu = Xtr.mean(0, keepdims=True)
    Xc = Xtr - mu
    # PCA basis from the FIT split only
    k = min(d_eff, Xc.shape[1], max(Xc.shape[0] - 1, 1))
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:k].T
    A = (Xc @ V)
    B = ((Xte - mu) @ V)
    s = A.std(0, keepdims=True) + 1e-8
    A, B = A / s, B / s
    ym, ys = ytr.mean(), ytr.std() + 1e-8

    g = torch.Generator(device="cpu").manual_seed(seed)
    At = torch.tensor(A, dtype=torch.float32, device=dev)
    Bt = torch.tensor(B, dtype=torch.float32, device=dev)
    yt = torch.tensor((ytr - ym) / ys, dtype=torch.float32, device=dev)[:, None]
    # an inner validation slice OF THE FIT SPLIT for early stopping — never the
    # scored clip. Deterministic split, so the comparison across columns is fair.
    n = At.shape[0]
    idx = torch.randperm(n, generator=g).to(dev)
    nv = max(int(0.15 * n), 8)
    vi, ti = idx[:nv], idx[nv:]
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(k, 256), torch.nn.GELU(),
        torch.nn.Linear(256, 64), torch.nn.GELU(),
        torch.nn.Linear(64, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(EPOCHS):
        net.train(); opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(At[ti]), yt[ti])
        loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(torch.nn.functional.mse_loss(net(At[vi]), yt[vi]))
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {kk: t.detach().clone() for kk, t in net.state_dict().items()}
        else:
            bad += 1
            if bad > 40:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(Bt).cpu().numpy().ravel() * ys + ym
    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - ym) ** 2).sum())     # ⛔ the FIT-split mean, not yte's
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def main() -> int:
    if not CACHE.is_file():
        print(f"[FATAL] {CACHE} missing — run rangeprobe.py with SPD_DUMP=1 first")
        return 2
    z = np.load(CACHE, allow_pickle=True)
    COLS = {k: list(v) for k, v in z["cols"].item().items()}
    TG = {k: list(v) for k, v in z["targets"].item().items()}
    rng = np.random.default_rng(0)

    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC", "split": "HELD-OUT, LEAD-MATCHED",
           "function_class": "MLP 128->256->64->1, early-stopped on an inner "
                             "slice of the FIT split only",
           "readable_quantity": "TRUE minus TIME-SHUFFLED. A raw R2 is not "
                                "readable: a nonlinear probe can reach clip-level "
                                "statistics, and clip identity predicts a clip's "
                                "mean range.", "targets": {}}
    hdr = (f"\n  {'target':<16}{'column':<28}{'TRUE':>9}{'SHUFFLED':>10}"
           f"{'TRUE-SHUF':>11}{'t':>8}{'n':>7}{'d':>6}")
    print("\n  E-DEC-36 · IS RANGE NONLINEARLY DECODABLE?"
          "\n  the readable quantity is TRUE - TIME-SHUFFLED, never the raw R2\n")
    print(hdr); print("  " + "-" * (len(hdr) - 3), flush=True)
    for tn, Y in TG.items():
        rep["targets"][tn] = {"columns": {}}
        for cname, X in COLS.items():
            tr, sh = [], []
            for i in range(len(X)):
                Xtr = np.concatenate([X[j] for j in range(len(X)) if j != i])
                ytr = np.concatenate([Y[j] for j in range(len(Y)) if j != i]).ravel()
                tr.append(mlp_fold(Xtr, ytr, X[i], Y[i].ravel()))
                # ⛔ shuffle WITHIN each clip: preserves every clip's marginal,
                # destroys only the frame correspondence.
                Ys = [y.ravel()[rng.permutation(len(y))] for y in Y]
                ytr_s = np.concatenate([Ys[j] for j in range(len(Y)) if j != i])
                sh.append(mlp_fold(Xtr, ytr_s, X[i], Ys[i]))
            tr, sh = np.array(tr), np.array(sh)
            d = tr - sh
            t = float(d.mean()) / max(float(d.std(ddof=1) / np.sqrt(len(d))), 1e-12)
            nrow = sum(len(y) for y in Y)
            rep["targets"][tn]["columns"][cname] = {
                "true": round(float(tr.mean()), 4),
                "shuffled": round(float(sh.mean()), 4),
                "true_minus_shuffled": round(float(d.mean()), 4),
                "t": round(t, 2), "n_rows": nrow, "d_raw": int(X[0].shape[1]),
                "carries_signal": bool(t > 2.0 and d.mean() > 0.02)}
            print(f"  {tn:<16}{cname:<28}{tr.mean():>+9.4f}{sh.mean():>+10.4f}"
                  f"{d.mean():>+11.4f}{t:>8.2f}{nrow:>7}{X[0].shape[1]:>6}",
                  flush=True)
        print()
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
