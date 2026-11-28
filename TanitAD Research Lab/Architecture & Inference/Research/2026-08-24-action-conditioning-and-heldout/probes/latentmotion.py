"""E-DEC-59 — DOES EGO MOTION PREDICT THE **LATENT'S OWN CHANGE**?

⭐ THE TARGET E-DEC-58 POINTED AT. The geometric panel failed because its targets
were LABEL-DERIVED: "mean bearing of detected agents" churns as detections appear
and disappear, and over 0.4 s that noise buried a rotation that is a physical
certainty — the CLOSED FORM could not clear its own null either, which is what
diagnosed the target rather than the hypothesis. ⇒ **Use a target that cannot
churn: the latent itself.** Ego motion moves the image by construction, so it must
move the latent.

⚠️ THIS IS NOT NEW GROUND — AND SAYING SO IS THE POINT. `deltaz.py` (E-DEC-40)
already asked almost this question and got **action −0.0109 (t −0.57)** against
drift **+0.1952 (t 8.38)**. What is new is THREE FIXES APPLIED TOGETHER, each of
which was a measured defect in that panel:

  1. ⭐ the **ω parameterisation** (PI directive): `[yaw_rate, a_long, v]` instead
     of `atan(L·κ)`, which is speed-blind and carries a legacy 2.9 m wheelbase.
     ω = v·κ is what actually moves the image.
  2. ⭐ **K-fold fit / per-clip score at ALL usable clips** instead of 20 —
     leave-one-out at n=129 is O(n²) and unaffordable; this is 13× cheaper for the
     same statistic (`panel_kfold.py`).
  3. ⭐ a **MATCHED NULL through the identical code path** (`SPD_NULL=1`). The
     20-clip panels had none, and a null measured under a different estimator does
     not transfer — E-DEC-58's K-fold null reached 4.15 where the leave-one-out one
     reached 3.49.

TARGETS — Δz = z_{t+k} − z_t projected on its top PCA directions, the same
construction E-DEC-40 used, so the numbers are comparable to the banked ones.

COLUMNS
    z_t                    the DRIFT baseline, and the POSITIVE CONTROL — E-DEC-40
                           measured it at t 8.38, so a panel that cannot reproduce
                           that is broken and must not be read.
    ego_state [ω, a, v]    ⭐ the PI's channels, as MEASURED STATE
    z_t + ego_state        the joint; the readable quantity is its MARGINAL over z_t
    constant               reads EXACTLY 0.0000

⭐ THE READ: `(z_t + ego) − z_t` is what ego motion adds to the latent's own drift.
If that clears the matched null, ego-motion conditioning has a real target at last.
If it does not — with the right channels, the right target and adequate power — then
the transition genuinely does not respond to ego motion, and that is a much stronger
negative than anything this campaign has produced so far.

--------------------------------------------------------------------------------
MM-E19 REVISION (2026-09-01, Benchmarks & Evals FlyWheel) — the two recorded probe
defects (class MM-C12; prereg `PREREG_MM_E19_K60_HORIZON.md` §"TWO PROBE DEFECTS")
are fixed, BEHAVIOUR-PRESERVING for the banked K=4 default:

  1. ``K`` WAS HARDCODED (``F, K, DT, K_FOLDS = 100, 4, 0.1, 10``). It is now
     ``--k`` / ``SPD_K`` with DEFAULT 4, so every previous invocation reproduces.
     MM-E19 runs the read at BOTH k=4 AND k=60 (per arm), because extending the
     probe to k=60 changes the horizon of the PROBE as well as of the ARM — both
     arms at both k values, or probe-horizon confounds arm-horizon (prereg §3b).
  2. it hardcoded ``sys.path.insert(0, r"C:\\Users\\Admin\\tanitad-mirror\\stack")``
     — the MM-C12 trap verbatim. ⚠️ That tree EXISTS on the dev box, which makes it
     WORSE, not better: its ``models/v6.py`` matches neither the repo nor G:, and
     ``load_trunk_auto`` REBUILDS the model from the checkpoint's config, so the
     code version is load-bearing. A silent import from it produces numbers that
     LOOK like results. The stack is now ``--stack`` / ``SPD_STACK`` with known-good
     auto-candidates (tanitad-mirror is deliberately NOT one), the import is
     PREFLIGHTED, the probe REFUSES (exit 2) when ``tanitad`` cannot be imported
     from the requested tree or resolves to a different one, and
     ``tanitad.__file__`` is stamped into the output — the ``actdiv_thor.py`` idiom.

  Also parameterised, defaults preserving the banked behaviour:
    ``--assets`` / ``SPD_ASSETS``  the dir holding ``v7tiny_<arm>/ckpt.pt``,
                                   ``sp2/cache/...`` and the helper modules
                                   (``v7tiny_g2.py``, ``panel_kfold.py``,
                                   ``rangeprobe_rff.py``). Default: this script's
                                   own directory — the original implicit layout.
                                   ``v7tiny_g2``'s own hardcoded scratchpad SP is
                                   OVERRIDDEN to this after import (that module
                                   reads its ``SP`` global at call time).
    ``--device`` / ``SPD_DEVICE``  default ``cuda`` (unchanged); ``cpu`` allowed.
  ``--preflight-only`` resolves + verifies the stack and exits — the 2-second
  failure instead of the after-the-rollout one (the `t1_eval.py` import trap).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch

SP = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SP))          # for _stackresolve + helper fallbacks

from _stackresolve import preflight_stack, resolve_stack  # noqa: E402

F, DT, K_FOLDS = 100, 0.1, 10        # K is a parameter since the MM-E19 revision


def resolve_config(argv=None, env=None) -> argparse.Namespace:
    """CLI > env (SPD_*) > banked default. Pure — testable without torch/cuda."""
    env = os.environ if env is None else env
    ap = argparse.ArgumentParser(description="E-DEC-59 latent-motion probe")
    ap.add_argument("--k", type=int, default=int(env.get("SPD_K", "4")),
                    help="probe horizon in ticks (dz = z_{t+k} - z_t). "
                         "DEFAULT 4 — the banked E-DEC-59 instrument. "
                         "MM-E19 also reads k=60 (6.0 s).")
    ap.add_argument("--band", default=env.get("SPD_BAND", "0:8"),
                    help="PCA directions of dz to score, lo:hi. ⛔ MM-E19 reads "
                         "run on BOTH 0:8 AND 8:16 or they are not admissible.")
    ap.add_argument("--corpus", default=env.get("SPD_CORPUS", ""),
                    help="dir of *.v2ep.pt clips (default: "
                         "<assets>/sp2/cache/physicalai-val130-heldout)")
    ap.add_argument("--arms", default=env.get("SPD_ARMS", "rdw8p30k"),
                    help="comma-separated arm names; ckpt at "
                         "<assets>/v7tiny_<arm>/ckpt.pt")
    ap.add_argument("--out", default=env.get("SPD_OUT", ""),
                    help="output JSON (default: <script dir>/latentmotion.json)")
    ap.add_argument("--nclips", type=int, default=int(env.get("SPD_NCLIPS", "80")))
    ap.add_argument("--assets", default=env.get("SPD_ASSETS", ""),
                    help="base dir for v7tiny_<arm>/, sp2/ and the helper "
                         "modules (default: this script's directory)")
    ap.add_argument("--stack", default=env.get("SPD_STACK", ""),
                    help="tanitad stack dir. Omit to auto-select a known-good "
                         "candidate. ⛔ REFUSES rather than silently importing "
                         "a wrong tree (MM-C12).")
    ap.add_argument("--device", default=env.get("SPD_DEVICE", "cuda"),
                    choices=("cuda", "cpu"))
    ap.add_argument("--null", action="store_true",
                    default=env.get("SPD_NULL") == "1",
                    help="matched null: inputs replaced by Gaussian noise "
                         "through the IDENTICAL code path")
    ap.add_argument("--null-seed", type=int,
                    default=int(env.get("SPD_NULL_SEED", "0")))
    ap.add_argument("--preflight-only", action="store_true",
                    help="resolve + verify the stack import, print the tree, exit")
    a = ap.parse_args(argv)
    a.assets = pathlib.Path(a.assets) if a.assets else SP
    a.corpus = (pathlib.Path(a.corpus) if a.corpus
                else a.assets / "sp2/cache/physicalai-val130-heldout")
    a.out = pathlib.Path(a.out) if a.out else SP / "latentmotion.json"
    b = str(a.band).split(":")
    a.band_lo, a.band_hi = int(b[0]), int(b[1])
    return a


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def clip_rows(zt, a, v, yaw, K, DT=DT):
    """Per-clip row construction — the banked loop body verbatim, K a parameter.

    Returns ``(z_t rows, ego_state rows, dz rows)`` or ``None`` when the clip is
    too short (< 30 usable rows), exactly as the banked instrument skipped it.
    """
    m = min(len(zt) - K, len(a) - K, len(v) - K, len(yaw) - K - 1)
    if m < 30:
        return None
    i = np.arange(m)
    omega = wrap(yaw[i + 1] - yaw[i]) / DT        # MEASURED yaw rate
    return zt[i], np.column_stack([omega, a[i, 1], v[i]]), zt[i + K] - zt[i]


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = resolve_config(argv)
    K = int(cfg.k)
    BAND_LO, BAND_HI = cfg.band_lo, cfg.band_hi
    N_DIR = BAND_HI - BAND_LO

    # ---- stack preflight FIRST: import tanitad from the verified tree BEFORE
    # any helper import can poison sys.path (v7tiny_g2 inserts its own trees at
    # import time; a module already in sys.modules is immune to that).
    stack = resolve_stack(cfg.stack or None)
    tanitad_file = preflight_stack(stack)
    if cfg.preflight_only:
        print(f"  [preflight] OK — stack {stack}")
        return 0

    # assets go AHEAD of the script dir so the run-copies of the helpers win,
    # exactly as they did in the banked layout (script and assets were one dir).
    sys.path.insert(0, str(cfg.assets))
    sys.path.insert(0, str(cfg.assets / "sp2"))
    import panel_kfold as PK
    from rangeprobe_rff import rff_fold, within_clip_r
    import v7tiny_g2 as G                # LAST — its import edits sys.path
    G.SP = cfg.assets                    # load_arm reads <SP>/v7tiny_<arm>/ckpt.pt
                                         # at CALL time from this module global

    dev = torch.device(cfg.device)
    LEAD = cfg.corpus
    clips = sorted(LEAD.glob("*.v2ep.pt"))[:cfg.nclips]
    ARMS = str(cfg.arms).split(",")
    arms = [a for a in ARMS if (cfg.assets / f"v7tiny_{a}" / "ckpt.pt").is_file()]
    if not clips:
        print(f"[REFUSED] no *.v2ep.pt clips under {LEAD}", file=sys.stderr)
        return 2
    if not arms:
        print(f"[REFUSED] none of {ARMS} has <assets>/v7tiny_<arm>/ckpt.pt "
              f"under {cfg.assets}", file=sys.stderr)
        return 2
    null = bool(cfg.null)
    print("\n  E-DEC-59 — DOES EGO MOTION PREDICT THE LATENT'S OWN CHANGE?")
    print("  channels [yaw_rate, a_long, v] · K-fold fit / per-clip score")
    print(f"  k={K} ({K * DT:.1f} s) · band [{BAND_LO}:{BAND_HI}) · "
          f"device {cfg.device}")
    if null:
        print("  NULL MODE - inputs are Gaussian noise; every t is a null draw")
    print(flush=True)
    rep = {"_evidence_class": ("MEASURED (ours; dev-box RTX 4060)"
                               if cfg.device == "cuda"
                               else "MEASURED (ours; dev-box CPU)"),
           "_tanitad_imported_from": tanitad_file,   # MM-C12: state the tree
           "_helper_files": {"panel_kfold": PK.__file__,
                             "rangeprobe_rff": sys.modules["rangeprobe_rff"].__file__,
                             "v7tiny_g2": G.__file__},
           "_assets": str(cfg.assets), "_corpus": str(LEAD),
           "eval_tier": "T0-DIAGNOSTIC", "split": "HELD-OUT", "k": K,
           "frames_per_clip": F, "device": cfg.device,
           "null_mode": null,
           # ⭐ the band travels WITH the numbers, so a banked JSON can never be
           # read as the top-8 panel when it is not.
           "pca_band": [BAND_LO, BAND_HI], "arms": {}}

    for arm in arms:
        w, st = G.load_arm(arm, dev)
        ZT, EGO, DZ = [], [], []
        with torch.no_grad():
            for c in clips:
                d = torch.load(c, map_location="cpu", weights_only=False)
                yaw = np.asarray(d["poses"], dtype=np.float64)[:, 2]
                z, act, spd = G.encode_clip(w, c, dev, F)
                zt = z.float().numpy().astype(np.float64)
                a = act.float().numpy().astype(np.float64)
                v = spd.float().numpy().astype(np.float64).ravel()
                rows = clip_rows(zt, a, v, yaw, K)
                if rows is None:
                    continue
                ZT.append(rows[0])
                EGO.append(rows[1])
                DZ.append(rows[2])
        del w
        torch.cuda.empty_cache()
        if len(DZ) < 10:
            print(f"  {arm}: too few clips"); continue

        if null:
            g = np.random.default_rng(int(cfg.null_seed))
            ZT = [g.standard_normal(x.shape) for x in ZT]
            EGO = [g.standard_normal(x.shape) for x in EGO]

        ALL = np.concatenate(DZ)
        mu = ALL.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(ALL - mu, full_matrices=False)
        COL = {"z_t (DRIFT / POSITIVE CONTROL)": ZT,
               "ego_state [w, a, v]": EGO,
               "z_t + ego_state": [np.concatenate([a1, b1], 1) for a1, b1 in zip(ZT, EGO)],
               "constant (control)": [np.ones((len(x), 1)) for x in ZT]}
        nrow = sum(len(x) for x in ZT)
        # ⚠️ LABEL THE SUBSPACE, NOT THE COUNT. This printed "top-8 PCs" while
        # SPD_BAND=8:16 was scoring directions 8-15 — the computation was right and
        # the LABEL was wrong, which is the shape of every stale-claim defect in
        # this programme: an artifact that says something other than what it is.
        print(f"  === {arm} (step {st}) — {len(ZT)} clips, {nrow} rows, "
              f"PCA directions [{BAND_LO}:{BAND_HI}) of Δz ===")
        print(f"  {'column':<32}{'r':>9}{'shuf':>9}{'t-shuf':>9}{'t':>7}")
        print("  " + "-" * 68)
        cells = {}
        for cn, X in COL.items():
            tr, sh = [], []
            for j in range(BAND_LO, BAND_HI):
                Y = [(dz - mu) @ Vt[j][:, None] for dz in DZ]
                rngj = np.random.default_rng(100 + j)
                Ysh = [y.ravel()[rngj.permutation(len(y))][:, None] for y in Y]
                tr.append(PK.kfold_clip_scores(X, Y, rff_fold, within_clip_r, K_FOLDS))
                sh.append(PK.kfold_clip_scores(X, Ysh, rff_fold, within_clip_r, K_FOLDS))
            tr, sh = np.concatenate(tr), np.concatenate(sh)
            cells[cn] = (tr, sh)
            dd = tr - sh
            t = float(dd.mean()) / max(
                float(dd.std(ddof=1) / np.sqrt(len(dd))), 1e-12)
            print(f"  {cn:<32}{tr.mean():>+9.4f}{sh.mean():>+9.4f}"
                  f"{dd.mean():>+9.4f}{t:>7.2f}", flush=True)

        def tt(x):
            return float(x.mean()) / max(float(x.std(ddof=1) / np.sqrt(len(x))), 1e-12)
        marg = cells["z_t + ego_state"][0] - cells["z_t (DRIFT / POSITIVE CONTROL)"][0]
        ctrl = tt(cells["z_t (DRIFT / POSITIVE CONTROL)"][0]
                  - cells["z_t (DRIFT / POSITIVE CONTROL)"][1])
        rep["arms"][arm] = {
            "step": int(st), "n_clips": len(ZT), "n_rows": nrow,
            "columns": {cn: {"r": round(float(v2[0].mean()), 4),
                             "t": round(tt(v2[0] - v2[1]), 2)}
                        for cn, v2 in cells.items()},
            "drift_control_t": round(ctrl, 2),
            "ego_marginal_over_drift": {"delta": round(float(marg.mean()), 4),
                                        "t": round(tt(marg), 2)}}
        print(f"\n  drift control t {ctrl:+.2f}  (E-DEC-40 banked it at 8.38)")
        print(f"  EGO-STATE's marginal over the drift: {marg.mean():+.4f} "
              f"(t {tt(marg):+.2f})\n", flush=True)

    cfg.out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"-> {cfg.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
