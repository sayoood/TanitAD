"""refcv5 landing analysis -- written BEFORE the checkpoint exists, so the landing
is a RE-RUN and not a design exercise.

Pre-registration: SPEC.md in this package's parent directory (H-REFCV5-DDIM-1).

WHAT THIS DOES, AND WHAT IT DELIBERATELY DOES NOT
-------------------------------------------------
It runs at ZERO GPU against dumps already banked by
``taniteval/tools/refcv3_arm.py --dump-dir``.

  --preflight            a 2-second import probe.  GATE 0 step 4.
  --floor R0 R1 [R2..]   refcv5's OWN inference-seed floor F(m), plus G-STOCH in
                         BOTH directions.
  --pair A B             paired episode-cluster bootstrap between two dumps.
  --families A [--vs B]  the four families per arm and, with --vs, the PAIRED
                         MARGINS that close W-1 (refcv4b shipped LATERAL and
                         TACTICAL with per-arm CIs and no paired margins,
                         `_intervals_complete: false`).

  It does NOT re-roll anything.  ALWAYS try ``refcv3_arm.py --analyze-only
  <dump_dir>`` before re-running a rollout: the rollout is the only expensive
  part and an analysis-time failure is recoverable at zero GPU.  That trap has
  already destroyed a paid-for 2-arm / 40-episode rollout in this programme.

THE ESTIMATOR
-------------
``taniteval.ci.paired_episode_cluster_bootstrap`` only.  ``overlapping_holdout_se``
is never imported and never used: it is not a jackknife, it biases the POINT
estimate (mean-of-split-means, not full_set), and on 27 dumps it shifted headline
ade_0_2s by -6.67% to +11.69% -- bidirectionally, including a sign flip on a
paired delta.

THE THREE VARIANCES -- say which one every interval answers
-----------------------------------------------------------
  V1 episode draw   -> what the paired episode-cluster bootstrap estimates.
  V2 training run   -> NOT priced here.  refcv5 is a single-seed arm.
                       H-ESTIM-SEED-1: a zero-lever replicate produced
                       "separated" differences on 3 of 18 family metrics, a
                       ~17% false-positive rate for `separated`.
  V3 inference run  -> what --floor prices.  refcv5 SAMPLES at eval
                       (refc.py:1815, `eps = torch.randn_like(x0_n)` with no
                       self.training guard, under a comment saying so by
                       design), so a single-roll separated CI answers a
                       question nobody asked.

  A margin is QUOTABLE only if its paired CI excludes zero AND |delta| exceeds
  F(m).  refav1's ~0.30 m ADE floor is NOT imported as refcv5's bar -- that was
  a different rig (an iCEM planner sampling 128 rollouts).  It is the REASON the
  replicate is mandatory, never the bar.

ASCII-ONLY OUTPUT: non-ASCII in print() is fatal under cp1252.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# --------------------------------------------------------------------------- #
# GATE 0 step 4 -- THE PREFLIGHT IMPORT PROBE.
#
# MEASURED: `taniteval` was absent from the refcv5 pod entirely, so the landing
# eval would have died on `import taniteval.ci` AFTER ~45 h of paid compute.
# The durable fix is a 2-second probe that fails BEFORE the expensive part, and
# names the missing module instead of dying inside analyze().
# --------------------------------------------------------------------------- #
_DEFAULT_ROOTS = ("/workspace/TanitAD/stack", "/workspace/TanitAD/taniteval")


def _add_roots(extra=()):
    roots = list(extra) + [p for p in os.environ.get(
        "REFCV5_ANALYSIS_PATH", "").split(os.pathsep) if p]
    roots += list(_DEFAULT_ROOTS)
    for r in roots:
        if r and os.path.isdir(r) and r not in sys.path:
            sys.path.insert(0, r)


def preflight(extra_roots=(), verbose=True):
    """Return (ok, report).  Imports every module the landing needs, by NAME."""
    _add_roots(extra_roots)
    report = []
    ok = True
    for mod, why in (
            ("numpy", "per-window arithmetic"),
            ("taniteval.ci", "the ONLY admissible estimator"),
    ):
        try:
            __import__(mod)
            report.append(("OK", mod, why, ""))
        except Exception as exc:                       # noqa: BLE001
            ok = False
            report.append(("MISSING", mod, why, "%s: %s"
                           % (type(exc).__name__, exc)))
    # a POSITIVE assertion, not merely an import: the two functions must exist
    if ok:
        try:
            from taniteval.ci import (episode_cluster_bootstrap,   # noqa: F401
                                      paired_episode_cluster_bootstrap)
            import taniteval.ci as _ci
            forbidden = "overlapping_holdout_se"
            report.append(("OK", "taniteval.ci.paired_episode_cluster_bootstrap",
                           "the paired estimator", ""))
            # a same-breath NEGATIVE control: the forbidden estimator exists in
            # that module, and this script must never call it.  If the symbol
            # has vanished the module is not the one this SPEC was written
            # against, and that is worth saying out loud.
            report.append(("OK" if hasattr(_ci, forbidden) else "WARN",
                           "taniteval.ci." + forbidden,
                           "present-but-never-called (control)",
                           "" if hasattr(_ci, forbidden)
                           else "symbol absent: not the module SPEC.md names"))
        except Exception as exc:                       # noqa: BLE001
            ok = False
            report.append(("MISSING", "taniteval.ci symbols",
                           "the paired estimator",
                           "%s: %s" % (type(exc).__name__, exc)))
    if verbose:
        for state, mod, why, err in report:
            print("  [%-7s] %-58s %s%s"
                  % (state, mod, why, ("  <- " + err) if err else ""))
        print("PREFLIGHT OK" if ok else "PREFLIGHT FAILED")
    return ok, report


# --------------------------------------------------------------------------- #
# dump loading
# --------------------------------------------------------------------------- #
ARMS = ("os", "ha", "ha0", "ha0_ext", "os_navshuf", "os_navzero", "os_navpred")


def _np():
    import numpy
    return numpy


def load_paths(dump_dir, arms=ARMS):
    """-> (paths{arm:[N,K,2]}, gt[N,K,2], eid[N], ws[N], present[list])."""
    np = _np()
    files = sorted(glob.glob(os.path.join(dump_dir, "ep*.npz")))
    if not files:
        sys.exit("[refcv5] no ep*.npz under %s" % dump_dir)
    acc, gt, eid, ws = {}, [], [], []
    present = None
    for f in files:
        z = np.load(f, allow_pickle=True)
        if "g" not in z.files:
            sys.exit("[refcv5] %s carries no GT key 'g' (have %s)"
                     % (f, sorted(z.files)))
        here = [a for a in arms if a in z.files]
        if present is None:
            present = here
        elif here != present:
            sys.exit("[refcv5] arm set differs across episodes: %s vs %s"
                     % (present, here))
        g = np.asarray(z["g"], dtype=np.float64)
        gt.append(g)
        for a in here:
            acc.setdefault(a, []).append(np.asarray(z[a], dtype=np.float64))
        eid.append(np.full(g.shape[0], int(np.asarray(z["eid"]).ravel()[0])))
        ws.append(np.asarray(z["ws"]).ravel())
    return ({a: np.concatenate(v) for a, v in acc.items()},
            np.concatenate(gt), np.concatenate(eid), np.concatenate(ws),
            present)


def load_decisions(dump_dir):
    """-> dict of concatenated per-window decision arrays, or {} if absent."""
    np = _np()
    files = sorted(glob.glob(os.path.join(dump_dir, "decisions", "ep*.npz")))
    if not files:
        return {}
    out = {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        for k in z.files:
            v = np.asarray(z[k])
            if k in ("ep_poses",):          # per-EPISODE, not per-window
                continue
            out.setdefault(k, []).append(v)
    res = {}
    for k, vs in out.items():
        try:
            res[k] = np.concatenate([v.reshape(v.shape[0], -1) if v.ndim > 1
                                     else v for v in vs])
        except ValueError:
            continue
    return res


def read_manifest(dump_dir):
    p = os.path.join(dump_dir, "manifest.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def grid_dt_k(dump_dir, default_dt=0.5):
    """dt_s and k from the dump's OWN manifest -- never hardcoded.

    A derived constant silently changes the experiment when its input changes:
    GRIDS = {"2s": (0.5, 4), "6s": (1.0, 6)} in refcv3_arm.py, and a panel that
    assumed one while reading the other would report a different experiment as a
    reproduction.
    """
    g = (read_manifest(dump_dir).get("grid") or {})
    return float(g.get("dt_s", default_dt)), g.get("k")


# --------------------------------------------------------------------------- #
# THE PAIRING ASSERTIONS -- before any number, and they REFUSE rather than align
# --------------------------------------------------------------------------- #
def assert_same_grid(name_a, ea, wa, name_b, eb, wb):
    np = _np()
    if ea.shape != eb.shape:
        sys.exit("[refcv5] REFUSED: %s has %d windows, %s has %d -- not the "
                 "same grid, and aligning them would be a fiction"
                 % (name_a, ea.size, name_b, eb.size))
    if not np.array_equal(ea, eb):
        sys.exit("[refcv5] REFUSED: episode ids differ between %s and %s"
                 % (name_a, name_b))
    if not np.array_equal(wa, wb):
        sys.exit("[refcv5] REFUSED: window origins (ws) differ between %s and %s"
                 % (name_a, name_b))
    print("PAIRING OK: n_windows=%d  n_episodes=%d  grid identical (%s vs %s)"
          % (ea.size, len(set(ea.tolist())), name_a, name_b))


# --------------------------------------------------------------------------- #
# metric components -- every one a PURE function of the banked [N,K,2] paths
#
# These are DERIVED from the dump.  The harness's own --analyze-only JSON stays
# authoritative for LEVELS; this script exists for the PAIRED MARGINS the
# harness does not emit.  `ade_m` is computed both ways on purpose: it is the
# CONTROL that must read a known value.  If the derived ADE does not reconcile
# with the harness's, none of the other derived rows is trustworthy either.
# --------------------------------------------------------------------------- #
def _headings(path, np):
    """path [N,K,2] in the ego frame at t0; the window origin is (0,0)."""
    z = np.zeros((path.shape[0], 1, 2), dtype=np.float64)
    p = np.concatenate([z, path], axis=1)                    # [N, K+1, 2]
    d = np.diff(p, axis=1)                                   # [N, K, 2]
    return np.arctan2(d[..., 1], d[..., 0]), d               # [N,K], [N,K,2]


def _curvature(path, np):
    """Menger curvature at each interior vertex of (0,0) + path."""
    z = np.zeros((path.shape[0], 1, 2), dtype=np.float64)
    p = np.concatenate([z, path], axis=1)                    # [N, K+1, 2]
    a, b, c = p[:, :-2], p[:, 1:-1], p[:, 2:]                # [N, K-1, 2]
    ab, bc, ca = b - a, c - b, a - c
    cross = np.abs(ab[..., 0] * bc[..., 1] - ab[..., 1] * bc[..., 0])
    denom = (np.linalg.norm(ab, axis=-1) * np.linalg.norm(bc, axis=-1)
             * np.linalg.norm(ca, axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(denom > 1e-9, 2.0 * cross / np.maximum(denom, 1e-12), 0.0)
    return k                                                 # [N, K-1]


def components(path, gt, dt, np):
    """-> {metric: per_window[N]}  (means over the K slots)."""
    err = path - gt                                          # [N,K,2]
    dist = np.linalg.norm(err, axis=-1)                      # [N,K]
    h_p, d_p = _headings(path, np)
    h_g, d_g = _headings(gt, np)
    # along / cross decomposition in the GT tangent frame
    tang = d_g / np.maximum(np.linalg.norm(d_g, axis=-1, keepdims=True), 1e-9)
    normal = np.stack([-tang[..., 1], tang[..., 0]], axis=-1)
    along = np.abs((err * tang).sum(-1))
    cross = np.abs((err * normal).sum(-1))
    # speed from the path's own spacing -- the same dt the dump declares
    sp_p = np.linalg.norm(d_p, axis=-1) / dt
    sp_g = np.linalg.norm(d_g, axis=-1) / dt
    dh = np.arctan2(np.sin(h_p - h_g), np.cos(h_p - h_g))    # wrapped
    yaw_p = np.diff(np.unwrap(h_p, axis=1), axis=1) / dt
    yaw_g = np.diff(np.unwrap(h_g, axis=1), axis=1) / dt
    k_p, k_g = _curvature(path, np), _curvature(gt, np)
    deg = 180.0 / np.pi
    return {
        "ade_m":                     dist.mean(1),
        "fde_m":                     dist[:, -1],
        "LON_speed_mae_mps":         np.abs(sp_p - sp_g).mean(1),
        "LON_speed_bias_mps":        (sp_p - sp_g).mean(1),
        "LON_along_mae_m":           along.mean(1),
        "LAT_cross_mae_m":           cross.mean(1),
        "LAT_heading_mae_deg":       (np.abs(dh) * deg).mean(1),
        "LAT_yaw_rate_mae_degps":    (np.abs(yaw_p - yaw_g) * deg).mean(1),
        "LAT_curvature_mae_1pm":     np.abs(k_p - k_g).mean(1),
    }


DP = {"LAT_curvature_mae_1pm": 6, "LAT_yaw_rate_mae_degps": 4}


def decision_components(dec, np):
    """TACTICAL / STRATEGIC per-window 0/1 components from decisions/*.npz.

    Returns {metric: (per_window[N], mask[N] or None, note)}.  A family that
    cannot be built is REFUSED with its reason and its n -- never silently
    dropped.
    """
    out, refused = {}, {}
    pairs = (("TAC_lat_correct", "lat_pred_nav_zero", "lat_label"),
             ("TAC_lon_correct", "lon_pred_nav_zero", "lon_label"),
             ("STR_route_correct", "route_pred_nav_zero", "route_label"))
    for name, pk, lk in pairs:
        if pk not in dec or lk not in dec:
            refused[name] = ("absent from decisions/: need %s and %s (have %s)"
                             % (pk, lk, ",".join(sorted(dec)[:12])))
            continue
        p = np.asarray(dec[pk]).ravel().astype(np.int64)
        l = np.asarray(dec[lk]).ravel().astype(np.int64)
        if p.shape != l.shape:
            refused[name] = "shape mismatch %s vs %s" % (p.shape, l.shape)
            continue
        valid = l >= 0
        out[name] = ((p == l).astype(np.float64), valid,
                     "prediction key %s vs label %s" % (pk, lk))
        # class-restricted recalls, with their n printed by the caller
        for cls, cname in ((0, "left"), (2, "right")):
            if name == "TAC_lat_correct":
                m = valid & (l == cls)
                if m.sum() > 0:
                    out["TAC_turn_%s_recall" % cname] = (
                        (p == l).astype(np.float64), m,
                        "recall of lat class %d among %d labelled windows"
                        % (cls, int(m.sum())))
    return out, refused


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _fmt(v, dp):
    return ("%." + str(dp) + "f") % v


def verdict(delta, separated, floor):
    if floor is None:
        return "ns" if not separated else "SEPARATED (no floor yet -- NOT quotable)"
    if not separated:
        return "ns"
    return "QUOTABLE" if abs(delta) > floor else "WITHIN-NOISE"


def cmd_pair(args):
    ok, _ = preflight(args.stack_root, verbose=False)
    if not ok:
        sys.exit("[refcv5] preflight failed; run --preflight for the report")
    np = _np()
    from taniteval.ci import paired_episode_cluster_bootstrap
    pa, ga, ea, wa, arms_a = load_paths(args.pair[0])
    pb, gb, eb, wb, arms_b = load_paths(args.pair[1])
    assert_same_grid(args.pair[0], ea, wa, args.pair[1], eb, wb)
    if not np.allclose(ga, gb, atol=1e-6):
        sys.exit("[refcv5] REFUSED: the two dumps disagree on GROUND TRUTH -- "
                 "same grid indices, different targets")
    arm = args.arm
    for tag, p in ((args.pair[0], pa), (args.pair[1], pb)):
        if arm not in p:
            sys.exit("[refcv5] arm %r not in %s (have %s)"
                     % (arm, tag, sorted(p)))
    dt, k = grid_dt_k(args.pair[0])
    ca = components(pa[arm], ga, dt, np)
    cb = components(pb[arm], gb, dt, np)
    floors = _load_floor(args.floor_json)
    print("dt_s=%s  k=%s  arm=%s  n_windows=%d  n_episodes=%d"
          % (dt, k, arm, ea.size, len(set(ea.tolist()))))
    rows = {}
    for m in ca:
        dp = DP.get(m, 4)
        r = paired_episode_cluster_bootstrap(cb[m], ca[m], ea,
                                             n_boot=args.n_boot, seed=args.seed)
        f = floors.get(m)
        r["floor"] = f
        r["verdict"] = verdict(r["delta"], r["separated"], f)
        r["A_mean"], r["B_mean"] = float(ca[m].mean()), float(cb[m].mean())
        rows[m] = r
        print("  %-26s A=%s  B=%s  B-A=%s [%s, %s] sep=%-5s floor=%-10s %s"
              % (m, _fmt(r["A_mean"], dp), _fmt(r["B_mean"], dp),
                 _fmt(r["delta"], dp), _fmt(r["lo"], dp), _fmt(r["hi"], dp),
                 r["separated"], ("n/a" if f is None else _fmt(f, dp)),
                 r["verdict"]))
    # THE ESTIMATOR SELF-CONTROL: A against itself must read exactly zero.
    ctrl = paired_episode_cluster_bootstrap(ca["ade_m"], ca["ade_m"], ea,
                                            n_boot=200, seed=args.seed)
    good = (ctrl["delta"] == 0.0 and ctrl["lo"] == 0.0 and ctrl["hi"] == 0.0
            and not ctrl["separated"])
    print("CONTROL paired(A,A): delta=%.10f lo=%.10f hi=%.10f separated=%s -> %s"
          % (ctrl["delta"], ctrl["lo"], ctrl["hi"], ctrl["separated"],
             "OK" if good else "FAILED -- NO NUMBER ABOVE IS ADMISSIBLE"))
    n_diff = int((ca["ade_m"] != cb["ade_m"]).sum())
    print("windows whose ade_m differs at all: %d/%d = %.2f%%"
          % (n_diff, ca["ade_m"].size, 100.0 * n_diff / ca["ade_m"].size))
    _write(args.out, {"A": args.pair[0], "B": args.pair[1], "arm": arm,
                      "dt_s": dt, "k": k,
                      "n_windows": int(ea.size),
                      "n_episodes": len(set(ea.tolist())),
                      "paired_B_minus_A": rows,
                      "control_paired_A_vs_A": ctrl,
                      "control_ok": bool(good),
                      "n_windows_ade_differs": n_diff,
                      "arms_present": arms_a,
                      "estimator": "paired_episode_cluster_bootstrap "
                                   "(taniteval/ci.py) n_boot=%d seed=%d"
                                   % (args.n_boot, args.seed),
                      "variance_answered": "V1 episode draw; V3 priced only if "
                                           "floor_json was supplied; V2 "
                                           "(training run) NOT priced"})


def cmd_floor(args):
    """G-STOCH + F(m): refcv5's OWN inference-seed floor, measured here."""
    ok, _ = preflight(args.stack_root, verbose=False)
    if not ok:
        sys.exit("[refcv5] preflight failed; run --preflight for the report")
    np = _np()
    from taniteval.ci import paired_episode_cluster_bootstrap
    dirs = args.floor
    if len(dirs) < 2:
        sys.exit("[refcv5] --floor needs at least two rolls of the SAME ckpt")
    loaded = [load_paths(d) for d in dirs]
    base_e, base_w = loaded[0][2], loaded[0][3]
    for d, (_, _, e, w, _) in zip(dirs[1:], loaded[1:]):
        assert_same_grid(dirs[0], base_e, base_w, d, e, w)
    # ---- ARGV AUDIT: the replicate must move NOTHING but its output paths ----
    argvs, argv_ok = [], True
    for d in dirs:
        m = read_manifest(d)
        argvs.append(((m.get("model") or {}).get("argv")
                      or m.get("argv") or []))
    if argvs[0]:
        ignore = ("--dump-dir", "--out")
        def strip(a):
            out, skip = [], False
            for tok in a:
                if skip:
                    skip = False
                    continue
                if tok in ignore:
                    skip = True
                    continue
                out.append(tok)
            return out
        ref = strip(argvs[0])
        for d, a in zip(dirs[1:], argvs[1:]):
            if strip(a) != ref:
                argv_ok = False
                print("  ARGV AUDIT FAILED: %s differs from %s beyond "
                      "--dump-dir/--out" % (d, dirs[0]))
        print("ARGV AUDIT: %s" % ("OK -- the replicate moved zero levers"
                                  if argv_ok else "FAILED -> THE FLOOR IS VOID"))
    else:
        print("ARGV AUDIT: INCONCLUSIVE -- no argv in manifest.json; "
              "record it manually before quoting the floor")
    dt, k = grid_dt_k(dirs[0])
    comp = [components(p[args.arm], g, dt, np) for (p, g, _, _, _) in loaded]
    # ---- G-STOCH, both directions ------------------------------------------
    n = comp[0]["ade_m"].size
    diffs = [int((comp[0]["ade_m"] != c["ade_m"]).sum()) for c in comp[1:]]
    stoch_ok = all(x > 0 for x in diffs)
    print("G-STOCH  os differs across rolls on %s of %d windows -> %s"
          % ("/".join(str(x) for x in diffs), n,
             "OK (the sampler is live at eval)" if stoch_ok else
             "VOID -- the rolls are bit-identical, so this arm is NOT the "
             "stochastic arm the SPEC describes"))
    frozen_ok, frozen = True, {}
    for arm in ("ha", "ha0", "ha0_ext"):
        if all(arm in p for (p, _, _, _, _) in loaded):
            cc = [components(p[arm], g, dt, np)["ade_m"]
                  for (p, g, _, _, _) in loaded]
            same = all(bool(np.array_equal(cc[0], c)) for c in cc[1:])
            frozen[arm] = same
            frozen_ok = frozen_ok and same
            print("G-STOCH  model-free %-8s bit-identical across rolls: %s"
                  % (arm, "YES" if same else "NO -> VOID"))
    # ---- F(m) = max pairwise |delta| ----------------------------------------
    floors, detail = {}, {}
    for m in comp[0]:
        dp = DP.get(m, 4)
        best, rows = 0.0, []
        for i in range(len(comp)):
            for j in range(i + 1, len(comp)):
                r = paired_episode_cluster_bootstrap(
                    comp[j][m], comp[i][m], base_e,
                    n_boot=args.n_boot, seed=args.seed)
                rows.append({"a": dirs[i], "b": dirs[j], **r})
                best = max(best, abs(float(r["delta"])))
        floors[m] = best
        detail[m] = rows
        n_sep = sum(1 for r in rows if r["separated"])
        print("  F(%-26s) = %s   [%d of %d zero-lever pairs read SEPARATED]"
              % (m, _fmt(best, dp), n_sep, len(rows)))
    print("NOTE: a zero-lever pair that reads SEPARATED is exactly "
          "H-ESTIM-SEED-1's failure mode, one variance down. It is expected, "
          "not a bug, and it is WHY the floor exists.")
    _write(args.out, {
        "rolls": dirs, "arm": args.arm, "dt_s": dt, "k": k,
        "n_windows": int(n), "n_episodes": len(set(base_e.tolist())),
        "argv_audit_ok": bool(argv_ok),
        "g_stoch_os_windows_differing": diffs,
        "g_stoch_ok": bool(stoch_ok),
        "g_stoch_model_free_bit_identical": frozen,
        "g_stoch_model_free_ok": bool(frozen_ok),
        "floor": floors, "pairwise": detail,
        "variance_answered": "V3 INFERENCE draw ONLY -- not V1, not V2",
        "not_imported": "refav1's ~0.30 m ADE floor is a DIFFERENT rig "
                        "(iCEM, 128 sampled rollouts) and is not this bar",
        "estimator": "paired_episode_cluster_bootstrap (taniteval/ci.py) "
                     "n_boot=%d seed=%d" % (args.n_boot, args.seed)})
    if not (stoch_ok and frozen_ok and argv_ok):
        sys.exit("[refcv5] G-STOCH / ARGV FAILED -> the floor is VOID and no "
                 "margin may be quoted against it")


def cmd_families(args):
    ok, _ = preflight(args.stack_root, verbose=False)
    if not ok:
        sys.exit("[refcv5] preflight failed; run --preflight for the report")
    np = _np()
    from taniteval.ci import (episode_cluster_bootstrap,
                              paired_episode_cluster_bootstrap)
    A = args.families
    pa, ga, ea, wa, arms_a = load_paths(A)
    dt, k = grid_dt_k(A)
    floors = _load_floor(args.floor_json)
    print("dump=%s  dt_s=%s  k=%s  n_windows=%d  n_episodes=%d  arms=%s"
          % (A, dt, k, ea.size, len(set(ea.tolist())), ",".join(arms_a)))

    # ---- the CONTROL that must read a known value --------------------------
    rec = {}
    if args.harness_json and os.path.exists(args.harness_json):
        with open(args.harness_json, encoding="utf-8") as fh:
            hj = json.load(fh)
        mine = float(components(pa["os"], ga, dt, np)["ade_m"].mean())
        theirs = _find_ade(hj, "os")
        if theirs is None:
            print("ADE RECONCILIATION: INCONCLUSIVE -- no os/ade_m found in %s"
                  % args.harness_json)
            rec = {"state": "INCONCLUSIVE"}
        else:
            d = abs(mine - theirs)
            state = "RECONCILED" if d <= args.ade_tol else "MISMATCH"
            print("ADE RECONCILIATION: derived=%.4f harness=%.4f |d|=%.5f "
                  "tol=%.5f -> %s%s" % (mine, theirs, d, args.ade_tol, state,
                                        "" if state == "RECONCILED" else
                                        "  <- the DERIVED family rows below "
                                        "are NOT trustworthy"))
            rec = {"state": state, "derived": mine, "harness": theirs,
                   "abs_diff": d, "tol": args.ade_tol}
    else:
        print("ADE RECONCILIATION: NOT RUN (pass --harness-json). The derived "
              "rows below carry no control and are DIRECTIONAL ONLY.")
        rec = {"state": "NOT_RUN"}

    per_arm, comp_by_arm = {}, {}
    for arm in arms_a:
        c = components(pa[arm], ga, dt, np)
        comp_by_arm[arm] = c
        per_arm[arm] = {m: episode_cluster_bootstrap(
            v, ea, n_boot=args.n_boot, seed=args.seed, dp=DP.get(m, 4))
            for m, v in c.items()}
    print("\nPER-ARM LEVELS (episode-cluster bootstrap; V1 only)")
    hdr = ["ade_m", "LON_speed_mae_mps", "LAT_cross_mae_m",
           "LAT_curvature_mae_1pm", "LAT_heading_mae_deg"]
    print("  %-12s %s" % ("arm", " ".join("%-22s" % h for h in hdr)))
    for arm in arms_a:
        print("  %-12s %s" % (arm, " ".join(
            "%-22s" % _fmt(per_arm[arm][m]["mean"], DP.get(m, 4))
            for m in hdr)))
    # ---- L1: the committed LATERAL bar, with the straight-line floor beside it
    if "ha0" in per_arm and "os" in per_arm:
        cur_os = per_arm["os"]["LAT_curvature_mae_1pm"]["mean"]
        cur_h0 = per_arm["ha0"]["LAT_curvature_mae_1pm"]["mean"]
        xt_os = per_arm["os"]["LAT_cross_mae_m"]["mean"]
        xt_ref = (per_arm.get("ha0_ext", {}).get("LAT_cross_mae_m", {})
                  .get("mean"))
        print("\nL1 (committed): os curvature %s vs ha0 straight-line floor %s "
              "-> %s" % (_fmt(cur_os, 6), _fmt(cur_h0, 6),
                         "BELOW the floor" if cur_os < cur_h0
                         else "AT/ABOVE the floor -- L1 REFUTED on this half"))
        if xt_ref is not None:
            print("L1 guard: os cross-track %s vs ha0_ext %s -> %s"
                  % (_fmt(xt_os, 4), _fmt(xt_ref, 4),
                     "held" if xt_os <= xt_ref else
                     "DEGRADED -- curvature bought by flattening the path"))

    # ---- TACTICAL / STRATEGIC ----------------------------------------------
    dec = load_decisions(A)
    dcomp, drefused = decision_components(dec, np) if dec else ({}, {
        "TACTICAL/STRATEGIC": "no decisions/ep*.npz under %s" % A})
    dec_levels = {}
    if dcomp:
        print("\nTACTICAL / STRATEGIC (per-window 0/1 from decisions/)")
        for m, (v, mask, note) in sorted(dcomp.items()):
            mm = np.ones_like(v, dtype=bool) if mask is None else mask
            if mm.sum() == 0:
                drefused[m] = "n = 0 after masking"
                continue
            r = episode_cluster_bootstrap(v[mm], ea[mm], n_boot=args.n_boot,
                                          seed=args.seed)
            dec_levels[m] = {**r, "n": int(mm.sum()), "note": note}
            print("  %-26s %s [%s, %s]  n=%d   %s"
                  % (m, _fmt(r["mean"], 4), _fmt(r["lo"], 4),
                     _fmt(r["hi"], 4), int(mm.sum()), note))
    for m, why in sorted(drefused.items()):
        print("  REFUSED %-24s %s" % (m, why))

    # ---- the PAIRED MARGINS that close W-1 ---------------------------------
    paired = {}
    if args.vs:
        pb, gb, eb, wb, arms_b = load_paths(args.vs)
        assert_same_grid(A, ea, wa, args.vs, eb, wb)
        arm = args.arm
        cb = components(pb[arm], gb, dt, np)
        ca = comp_by_arm[arm]
        print("\nPAIRED MARGINS  (%s) - (%s)  on arm %s   [closes W-1]"
              % (A, args.vs, arm))
        for m in ca:
            dp = DP.get(m, 4)
            r = paired_episode_cluster_bootstrap(ca[m], cb[m], ea,
                                                 n_boot=args.n_boot,
                                                 seed=args.seed)
            f = floors.get(m)
            r["floor"] = f
            r["verdict"] = verdict(r["delta"], r["separated"], f)
            paired[m] = r
            print("  %-26s delta=%s [%s, %s] sep=%-5s floor=%-10s %s"
                  % (m, _fmt(r["delta"], dp), _fmt(r["lo"], dp),
                     _fmt(r["hi"], dp), r["separated"],
                     ("n/a" if f is None else _fmt(f, dp)), r["verdict"]))
        decb = load_decisions(args.vs)
        dcb, _ = decision_components(decb, np) if decb else ({}, {})
        for m, (v, mask, note) in sorted(dcomp.items()):
            if m not in dcb:
                continue
            vb, mb, _ = dcb[m]
            mm = np.ones_like(v, dtype=bool) if mask is None else mask
            mmb = np.ones_like(vb, dtype=bool) if mb is None else mb
            use = mm & mmb
            if use.sum() == 0 or v.shape != vb.shape:
                continue
            r = paired_episode_cluster_bootstrap(v[use], vb[use], ea[use],
                                                 n_boot=args.n_boot,
                                                 seed=args.seed)
            f = floors.get(m)
            r["floor"], r["n"] = f, int(use.sum())
            r["verdict"] = verdict(r["delta"], r["separated"], f)
            paired[m] = r
            print("  %-26s delta=%s [%s, %s] sep=%-5s n=%d  %s"
                  % (m, _fmt(r["delta"], 4), _fmt(r["lo"], 4),
                     _fmt(r["hi"], 4), r["separated"], int(use.sum()),
                     r["verdict"]))

    _write(args.out, {
        "dump": A, "vs": args.vs, "arm": args.arm, "dt_s": dt, "k": k,
        "n_windows": int(ea.size), "n_episodes": len(set(ea.tolist())),
        "arms_present": arms_a,
        "ade_reconciliation_control": rec,
        "per_arm_levels": per_arm,
        "decision_levels": dec_levels,
        "decision_refused": drefused,
        "paired_margins": paired,
        "_intervals_complete": bool(paired) if args.vs else False,
        "estimator": "episode_cluster_bootstrap / "
                     "paired_episode_cluster_bootstrap (taniteval/ci.py) "
                     "n_boot=%d seed=%d" % (args.n_boot, args.seed),
        "forbidden_estimator_used": False,
        "variance_answered": "V1 episode draw; V3 only where a floor was "
                             "supplied; V2 (training run) NOT priced -- "
                             "H-ESTIM-SEED-1 applies in full",
        "scope": "B1 corpus, NOT the parity corpus "
                 "physicalai-train-e438721ae894 -- valid as an ARM delta, "
                 "INADMISSIBLE as a LEVEL against any parity arm",
        "excluded_metrics": {
            "oracle_sel": "a_star bound against decoder.anchors on a "
                          "v0-conditioned vocabulary (refcv3_arm.py:1492 vs "
                          "refc_v3_train.py:538-545)",
            "anchor_acc": "same mechanism",
            "sel_agrees_oracle": "same mechanism",
            "dyaw_turn_gate": "|dyaw|>0.15 demands R 19 m at v0 1.40 m/s; the "
                              "human fails it 3 of 9"},
        "anchor_units_provenance": "OPERATOR-ASSERTED: the live 117-anchor bank "
                                   "declares no control_units; the run carries "
                                   "--anchor-control-units alat, recorded as "
                                   "control_units_source=cli-override-legacy-file"})


def _find_ade(obj, arm):
    """Best-effort pull of arm ade_m from a refcv3_arm.py record."""
    if not isinstance(obj, dict):
        return None
    for key in ("arms", "families", "metrics", "results"):
        sub = obj.get(key)
        if isinstance(sub, dict) and arm in sub and isinstance(sub[arm], dict):
            for mk in ("ade_m", "ade", "ade_0_2s"):
                v = sub[arm].get(mk)
                if isinstance(v, dict):
                    v = v.get("mean")
                if isinstance(v, (int, float)):
                    return float(v)
    for v in obj.values():
        if isinstance(v, dict):
            got = _find_ade(v, arm)
            if got is not None:
                return got
    return None


def _load_floor(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return (json.load(fh).get("floor") or {})


def _write(path, payload):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=float)
    # verify by CONTENT, never by the write's exit status
    n = os.path.getsize(path)
    print("wrote %s (%d bytes)%s" % (path, n,
                                     "" if n > 2 else "  <- SUSPECT: empty"))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="refcv5 landing analysis (zero GPU, banked dumps only)")
    ap.add_argument("--preflight", action="store_true",
                    help="import probe only; GATE 0 step 4")
    ap.add_argument("--floor", nargs="*", metavar="DUMP",
                    help="two or more rolls of the SAME ckpt -> F(m) + G-STOCH")
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"),
                    help="paired episode-cluster bootstrap between two dumps")
    ap.add_argument("--families", metavar="DUMP",
                    help="four families for every arm in this dump")
    ap.add_argument("--vs", default=None, metavar="DUMP",
                    help="with --families: the paired margins (closes W-1)")
    ap.add_argument("--arm", default="os")
    ap.add_argument("--floor-json", default=None,
                    help="the --floor output; turns 'separated' into QUOTABLE")
    ap.add_argument("--harness-json", default=None,
                    help="refcv3_arm.py --out record, for the ADE control")
    ap.add_argument("--ade-tol", type=float, default=0.02)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stack-root", nargs="*", default=(),
                    help="extra sys.path roots (also REFCV5_ANALYSIS_PATH)")
    a = ap.parse_args(argv)

    if a.preflight:
        ok, _ = preflight(a.stack_root)
        return 0 if ok else 2
    if a.floor:
        cmd_floor(a)
        return 0
    if a.pair:
        cmd_pair(a)
        return 0
    if a.families:
        cmd_families(a)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
