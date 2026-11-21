"""COLLAPSE MONITOR (PRH_JEPA_FOR_V8F.md work item W3): effective rank of the world-model latents of a checkpoint.  Tier T0 latent diagnostic, never a driving number.  Evidence class MEASURED.

  python collapse_monitor.py --run-dir C:/Users/Admin/t2v8f_pulls/ladder_R35H_s0 --frames F:/TanitAD-scratch/ladder_read/cache/frames_s0.npz --out out.json
  (basis arm)  ... --basis-npz <basis.npz> --ubasis-npz <ubasis_eval.npz> --selfcheck-json <run>/selfcheck.json

Per level (level 1 = z1, 0.2 s; levels 2 and 3 = the stack's encoded levels, strides read from the built model) the pooled rows of all 141 eval clips form Z [N, D] and the monitor reports
  PR = (tr C)^2 / tr(C^2)  (participation ratio of the covariance C, rows = samples, columns = feature dims, centred per feature, float64),
  top-20 eigenvalue fractions, the number of dims holding 90 % / 99 % of the variance, and the count of dead dims (per-dimension std < 1e-4).
z1 comes from the existing readers (lp_lib.clip_z1, or slowwm_hook.clip_z1_basis for a --wm-basis arm); levels 2 and 3 from lr_lib.encode_clip (basis: slowwm_hook.encode_clip_basis).
Nothing is re-implemented on the model side; the only new maths is `spectrum_stats`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THEORY = os.path.normpath(os.path.join(HERE, "..", ".."))
DEAD_STD = 1e-4
TOP_K = 20
RULE = {"collapsed": 0.02, "partial": 0.10}      # PR/dim < 0.02 collapsed, < 0.10 partial, else healthy (stated before the first R35H number)


def _cov(Z):
    """Covariance of Z [N, D]: rows are samples, columns are feature dims; centred per feature; float64. Returns [D, D]."""
    Z = np.asarray(Z, dtype=np.float64)
    Zc = Z - Z.mean(0, keepdims=True)
    return Zc.T @ Zc / max(len(Z) - 1, 1)


def spectrum_stats(Z):
    """Effective-rank statistics of Z [N, D] (see module doc). Every number is a plain float/int/list."""
    C = _cov(Z)
    D = C.shape[0]
    ev = np.clip(np.linalg.eigvalsh(C)[::-1], 0.0, None)
    tr = float(ev.sum())
    pr = float(tr ** 2 / float((ev ** 2).sum())) if tr > 0 else 0.0
    frac = ev / tr if tr > 0 else np.zeros_like(ev)
    cum = np.cumsum(frac)
    n90 = int(np.searchsorted(cum, 0.90 - 1e-12) + 1) if tr > 0 else 0
    n99 = int(np.searchsorted(cum, 0.99 - 1e-12) + 1) if tr > 0 else 0
    std = np.sqrt(np.clip(np.diag(C), 0.0, None))
    return {"n": int(len(Z)), "dim": int(D), "pr": pr, "pr_over_dim": pr / D, "trace": tr,
            "top20_frac": [float(x) for x in frac[:TOP_K]], "dims_for_90pct": n90, "dims_for_99pct": n99,
            "dead_dims": int((std < DEAD_STD).sum()), "std_min": float(std.min()), "std_median": float(np.median(std))}


def reading(pr_over_dim, rule=RULE):
    return "collapsed" if pr_over_dim < rule["collapsed"] else "partial" if pr_over_dim < rule["partial"] else "healthy"


def selfcheck(seed=0, N=3000, D=64):
    """Planted controls. Returns {check: bool} and 'ok'. A monitor that reads the wrong axis fails 'offset_iso' (a large per-feature offset survives a wrong-axis centring) and the dim check."""
    rng = np.random.default_rng(seed)
    out = {}
    iso = rng.standard_normal((N, D))
    s = spectrum_stats(iso)
    out["iso_pr_within_5pct_of_dim"] = abs(s["pr"] - D) / D < 0.05 and s["dim"] == D
    off = iso + 50.0 * rng.standard_normal(D)[None]             # per-feature offset: centring over the right axis removes it
    out["offset_iso_pr_within_5pct_of_dim"] = abs(spectrum_stats(off)["pr"] - D) / D < 0.05
    r1 = np.outer(rng.standard_normal(N), rng.standard_normal(D))
    out["rank1_pr_near_1"] = abs(spectrum_stats(r1)["pr"] - 1.0) < 0.05
    for r in (4, 16):
        # equal-variance rank-r: orthonormal r directions, unit-variance coefficients
        Q, _ = np.linalg.qr(rng.standard_normal((D, r)))
        zr = rng.standard_normal((N, r)) @ Q.T
        out[f"rank{r}_pr_within_5pct"] = abs(spectrum_stats(zr)["pr"] - r) / r < 0.05
    dead = iso.copy()
    dead[:, :10] = 0.0
    out["dead_dims_counted"] = spectrum_stats(dead)["dead_dims"] == 10
    out["ok"] = all(out.values())
    return out


# ----------------------------------------------------------------------------------------------- model side (lazy imports)
def _setup_paths():
    for d in ("ladder_read", "ladder_plantest", "slowwm"):
        p = os.path.join(THEORY, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect_latents(model, fr, clip_idx=None, basis=False):
    """{1: [N1, D], 2: [N2, D], 3: [N3, D]} pooled over clips, plus the per-clip z1 dict (for the bit-equality control).
    z1: lp_lib.clip_z1 (basis arm: slowwm_hook.clip_z1_basis) on the frames of the chosen clips. Levels 2, 3: lr_lib.encode_clip (basis: encode_clip_basis) z list entries 1 and 2."""
    _setup_paths()
    import torch
    import lr_lib as LR
    import lp_lib as LP
    import slowwm_hook as SH
    idx = list(range(len(fr["clips"]))) if clip_idx is None else list(clip_idx)
    sub = dict(fr)
    for k in ("clips", "T", "zmean", "act", "ok", "zcell", "ubasis"):
        if k in fr:
            sub[k] = [fr[k][i] for i in idx]
    z1 = SH.clip_z1_basis(model.wm, sub) if basis else LP.clip_z1(model.wm, sub)
    rows = {1: [], 2: [], 3: []}
    for j, c in enumerate(sub["clips"]):
        rows[1].append(z1[c].numpy())
        if basis:
            zl, _ = SH.encode_clip_basis(model, sub, j, sub["act"][j])
        else:
            zl, _ = LR.encode_clip(model, LR.z_in(model, sub, j), sub["act"][j])
        with torch.no_grad():
            for l in (2, 3):
                rows[l].append(zl[l - 1][0].float().numpy())
    return {l: np.concatenate(v, 0) for l, v in rows.items()}, z1


def monitor(run_dir, frames, basis_npz=None, ubasis_npz=None, selfcheck_json=None, clip_idx=None, step_s=None):
    _setup_paths()
    import lr_lib as LR
    import slowwm_hook as SH
    LR.memory_guard()
    model, cfg, argv, ck = LR.load_arm(run_dir)
    is_b = SH.is_basis(model)
    fr = LR.load_eval_frames(frames)
    if not is_b and LR.adapter_sha(model) != fr["adapter_sha"]:      # the frames hold the frozen-adapter token means of THIS arm's adapter (as lp_lib.prepare_arm requires)
        raise LR.LadderMismatch(f"adapter hash {LR.adapter_sha(model)[:12]} != frames file {fr['adapter_sha'][:12]}")
    if is_b:
        if not (basis_npz and ubasis_npz and selfcheck_json):
            raise SystemExit("a --wm-basis arm needs --basis-npz, --ubasis-npz and --selfcheck-json")
        B = SH.require_arm_basis(selfcheck_json, basis_npz)
        SH.attach_basis_frames(fr, B, ubasis_npz)
    elif basis_npz:
        raise SystemExit("--basis-npz given but the model is an adapter arm")
    Z, _ = collect_latents(model, fr, clip_idx, basis=is_b)
    strides = [1] + [int(x) for x in argv.get("wm-strides", [])]
    levels = {}
    for l in (1, 2, 3):
        s = spectrum_stats(Z[l])
        s["step_seconds"] = round(0.2 * int(np.prod(strides[:l])), 3) if len(strides) >= l else None
        s["reading"] = reading(s["pr_over_dim"])
        levels[str(l)] = s
    return {"tool": "collapse_monitor.py", "evidence_class": "MEASURED", "tier": "T0 latent diagnostic (never a driving number)",
            "run_dir": run_dir.replace("\\", "/"), "checkpoint_sha256": sha256_file(os.path.join(run_dir, "ckpt_last.pt")),
            "frames_file": frames.replace("\\", "/"), "frames_sha256": sha256_file(frames), "basis_arm": bool(is_b),
            "n_clips": len(fr["clips"]) if clip_idx is None else len(list(clip_idx)), "rule": {"collapsed_below": RULE["collapsed"], "partial_below": RULE["partial"], "statistic": "PR/dim"},
            "wm_strides_argv": argv.get("wm-strides"), "levels": levels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--basis-npz")
    ap.add_argument("--ubasis-npz")
    ap.add_argument("--selfcheck-json")
    ap.add_argument("--max-clips", type=int)
    a = ap.parse_args()
    sc = selfcheck()
    if not sc["ok"]:
        raise SystemExit(f"planted self-check failed: {sc}")
    res = monitor(a.run_dir, a.frames, a.basis_npz, a.ubasis_npz, a.selfcheck_json, clip_idx=range(a.max_clips) if a.max_clips else None)
    res["planted_selfcheck"] = sc
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for l, s in res["levels"].items():
        print(f"level {l} ({s['step_seconds']} s): n={s['n']} dim={s['dim']} PR={s['pr']:.2f} PR/dim={s['pr_over_dim']:.4f} 90%={s['dims_for_90pct']} 99%={s['dims_for_99pct']} dead={s['dead_dims']} -> {s['reading']}")


if __name__ == "__main__":
    main()
