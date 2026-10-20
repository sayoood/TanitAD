"""PREREG_V8F_MINI Amendment 4: RESULT_FULL.md from box 2's chain output -- the verdict, the per-run eval files, and PAIRED family deltas.

  python v8f_mini_full_report.py --verdict verdict.json --eval-dir <eval> --out RESULT_FULL.md [--json report.json] [--notes notes.md]

It decides NOTHING. K-1 and WM-1 are COPIED from verdict.json (`v8f_mini_full_verdict.py`, which copied K-1 from `compare.json`). It adds:
  * the four families per run, copied from each eval_<tag>.json (P10-yaw-4512's from floors_report.json), point [lo, hi];
  * DESCRIPTIVE paired LONGITUDINAL / LATERAL deltas through `ra1_hjepa_eval.paired_family_block` (paired episode-cluster bootstrap over
    the 141 eval episodes): M2b - M1 per seed (WM-1's pairing), and every run - P10-yaw-4512 (K-1's pairing). No bar was registered on
    them, so they carry no PASS / FAIL.
Ground truth is REBUILT from the repo's eval inputs (`v8f_mini_data.eval_context`, kin_mode yaw), never taken from the run's own files.
Every guard REFUSES (rc 2), because a report over the wrong windows reads exactly like a right one:
  * each run's eid equals the rebuilt grid's, and its saved ade6 equals ||pred - gt|| per window to 1e-5 (else the GT is not its grid);
  * the P10 predictions reproduce floors_perwindow.npz's P10_refit__ade6 per window to 1e-5 when the file is present;
  * the verdict's numbers are re-derived from the npz files: WM-1 per seed = mean(M2b) - mean(M1), K-1 per run = mean(arm) - mean(P10),
    each to 1e-6 (a verdict computed over other files is refused);
  * the paired M1 - M1 control must read exactly +0 [0, 0] on every component.
Estimator, every paired interval: it answers ANOTHER DRAW OF EPISODES only. Training-seed variance enters only through F (2 seeds).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PKG = HERE.parents[1]
REPO = HERE.parents[5]
RAW = PKG / "raw"
P10_PRED = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-refav1-hj-mini" / "raw" / "p10_yaw4512" / \
    "p10_eval_pred6.npy"
ARMS, SEEDS, CONTROLS = ("M1", "M2b"), (0, 1), ("M1-navshuf",)
TOL_ADE, TOL_VERDICT = 1e-5, 1e-6
B = 2000


def refuse(msg):
    print(f"REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _paths():
    for p in (REPO / "stack", REPO / "taniteval"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _eval_module():
    _paths()
    spec = importlib.util.spec_from_file_location("ra1_hjepa_eval", REPO / "stack" / "scripts" / "ra1_hjepa_eval.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ground_truth(a):
    """-> (gt [N,30,2], eid [N], dt, provenance).  ``--gt-npz`` (keys gt, eid, dt) replaces the rebuild; tests use it, a real report does not."""
    if a.gt_npz:
        z = np.load(a.gt_npz)
        return z["gt"].astype(np.float64), z["eid"], float(z["dt"]), {"source": "gt-npz (TEST INPUT)", "path": a.gt_npz}
    _paths()
    from tanitad.data import v8f_mini_data as D
    grid = json.load(open(a.eval_grid, encoding="utf-8"))
    X = D.eval_context(grid, a.eval_kin, D.load_join(a.eval_join), kin_mode="yaw")
    return X["gt"].astype(np.float64), X["eid"], float(D.DT), {
        "source": "v8f_mini_data.eval_context(kin_mode='yaw') over the repo's eval inputs",
        "files": {os.path.basename(p): sha256(p) for p in (a.eval_grid, a.eval_kin, a.eval_join)}}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_runs(eval_dir, gt, eid):
    runs = {}
    for arm in ARMS + CONTROLS:
        for s in SEEDS:
            tag = f"{arm}_s{s}"
            pz, pj = os.path.join(eval_dir, f"eval_{tag}.npz"), os.path.join(eval_dir, f"eval_{tag}.json")
            if not os.path.exists(pz):
                if arm in ARMS:
                    refuse(f"eval_{tag}.npz is missing from {eval_dir}")
                continue
            z = np.load(pz)
            if not np.array_equal(z["eid"], eid):
                refuse(f"eval_{tag}.npz is not on the rebuilt eval grid (eid differs)")
            ade = np.linalg.norm(z["pred"].astype(np.float64) - gt, axis=-1).mean(1)
            dev = float(np.abs(ade - z["ade6"]).max())
            if dev > TOL_ADE:
                refuse(f"eval_{tag}: ||pred - gt|| differs from its saved ade6 by {dev:.3g} m (> {TOL_ADE}): the ground truth is not its grid")
            raw = json.load(open(pj, encoding="utf-8"))["raw"] if os.path.exists(pj) else None
            runs[tag] = dict(arm=arm, seed=s, pred=z["pred"].astype(np.float64), ade6=np.asarray(z["ade6"], np.float64), raw=raw,
                             max_ade_dev=dev)
    return runs


def load_p10(a, gt, eval_dir):
    p = np.load(a.p10_pred).astype(np.float64)
    if p.shape != gt.shape:
        refuse(f"P10 predictions {p.shape} are not on the eval grid {gt.shape}")
    ade = np.linalg.norm(p - gt, axis=-1).mean(1)
    fp = os.path.join(eval_dir, "floors_perwindow.npz")
    chk = "floors_perwindow.npz absent: checked against the verdict's K-1 instead"
    if os.path.exists(fp):
        dev = float(np.abs(ade - np.load(fp)["P10_refit__ade6"]).max())
        if dev > TOL_ADE:
            refuse(f"the P10 predictions do not reproduce floors_perwindow.npz's P10_refit__ade6 (max dev {dev:.3g} m)")
        chk = f"reproduces floors_perwindow.npz P10_refit__ade6 per window (max dev {dev:.2g} m)"
    return dict(pred=p, ade6=ade, sha256=sha256(a.p10_pred), check=chk)


def check_verdict(v, runs, p10):
    """Re-derive the verdict's point estimates from the files this report reads; any mismatch refuses."""
    out = []
    for c in v["WM1"]["per_seed"]:
        s = c["seed"]
        d = runs[f"M2b_s{s}"]["ade6"].mean() - runs[f"M1_s{s}"]["ade6"].mean()
        if abs(d - c["mean"]) > TOL_VERDICT:
            refuse(f"verdict WM-1 seed {s} reads {c['mean']:+.6f}, the eval files give {d:+.6f}")
        out.append(f"WM-1 s{s} {d:+.6f}")
    for arm in ARMS:
        for s, c in zip(SEEDS, v["K1"]["per_arm"][arm]["K1_ade6_minus_P10"]):
            d = runs[f"{arm}_s{s}"]["ade6"].mean() - p10["ade6"].mean()
            if abs(d - c["mean"]) > TOL_VERDICT:
                refuse(f"verdict K-1 {arm} seed {s} reads {c['mean']:+.6f}, the eval files and P10 predictions give {d:+.6f}")
            out.append(f"K-1 {arm}_s{s} {d:+.6f}")
    return out


def paired(E, runs, p10, gt, eid, dt, n_boot):
    import torch
    g = torch.from_numpy(gt).float()
    fpw = lambda pred: E.family_per_window(torch.from_numpy(pred).float(), g, dt)      # noqa: E731
    pw = {t: fpw(r["pred"]) for t, r in runs.items()}
    pw["P10"] = fpw(p10["pred"])
    blocks = {}
    for s in SEEDS:
        blocks[f"M2b_s{s} - M1_s{s}"] = E.paired_family_block(pw[f"M2b_s{s}"], pw[f"M1_s{s}"], eid, n_boot, 0)
    for t in runs:
        blocks[f"{t} - P10"] = E.paired_family_block(pw[t], pw["P10"], eid, n_boot, 0)
    ctl = E.paired_family_block(pw["M1_s0"], pw["M1_s0"], eid, min(n_boot, 200), 0)
    cells = [c for f in ("longitudinal", "lateral") for c in ctl[f].values() if c.get("status") != "UNAVAILABLE"]
    ok = bool(cells) and all(c["delta_exact"] == 0.0 and c["lo"] == 0.0 and c["hi"] == 0.0 and not c["separated"] for c in cells)
    if not ok:
        refuse("the paired M1 - M1 control did not read exactly +0 [0, 0]: the pairing is broken")
    return blocks, dict(m1_minus_m1_ok=ok, n_components=len(cells))


def g(d, path):
    """``path`` is dotted, or a tuple when a key itself holds a dot (``within_1.0_mps``)."""
    for k in (path if isinstance(path, tuple) else path.split(".")):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def fmt(x, dp=3):
    if x is None:
        return "--"
    if isinstance(x, dict):
        m = x.get("mean", x.get("delta"))
        if m is None:
            return str(x.get("status", "--"))
        if x.get("lo") is None:
            return f"{m:.{dp}f}"
        return f"{m:.{dp}f} [{x['lo']:.{dp}f}, {x['hi']:.{dp}f}]"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return f"{x:.{dp}f}"
    return str(x)


FAM_ROWS = (  # (family, label, path into eval_<tag>.json raw, decimals)
    ("LONGITUDINAL", "speed MAE (m/s)", "families.longitudinal.ci.components.speed_mae_mps", 3),
    ("LONGITUDINAL", "speed bias (m/s)", "families.longitudinal.ci.components.speed_bias_mps", 3),
    ("LONGITUDINAL", "target speed within 1 m/s (step share)", ("families", "longitudinal", "target_speed_acc", "within_1.0_mps"), 3),
    ("LONGITUDINAL", "along-track MAE (m)", "families.longitudinal.ci.components.along_mae_m", 3),
    ("LONGITUDINAL", "along-track final bias (m)", "families.longitudinal.ci.components.along_final_bias_m", 3),
    ("LONGITUDINAL", "accel MAE (m/s^2)", "families.longitudinal.accel_mae_mps2", 3),
    ("LONGITUDINAL", "ego progress ratio", "families.longitudinal.ego_progress.progress_ratio_mean", 3),
    ("LONGITUDINAL", "distance keeping", "families.longitudinal.distance_keeping.status", 0),
    ("LATERAL", "heading MAE (deg)", "families.lateral.ci.components.heading_mae_deg", 3),
    ("LATERAL", "yaw-rate MAE (deg/s)", "families.lateral.ci.components.yaw_rate_mae_degps", 3),
    ("LATERAL", "curvature MAE (1/m)", "families.lateral.ci.components.curvature_mae_1pm", 5),
    ("LATERAL", "cross-track MAE (m)", "families.lateral.ci.components.cross_mae_m", 3),
    ("LATERAL", "cross-track final MAE (m)", "families.lateral.ci.components.cross_final_mae_m", 3),
    ("TACTICAL", "lateral decision accuracy (from path)", "families.tactical_from_trajectory.lateral_decision.ci.accuracy", 3),
    ("TACTICAL", "lateral decision kappa", "families.tactical_from_trajectory.lateral_decision.kappa", 3),
    ("TACTICAL", "longitudinal decision accuracy (from path)", "families.tactical_from_trajectory.longitudinal_decision.accuracy", 3),
    ("TACTICAL", "longitudinal decision kappa", "families.tactical_from_trajectory.longitudinal_decision.kappa", 3),
    ("TACTICAL", "5-way collapsed accuracy", "families.tactical_from_trajectory.maneuver_5way_collapsed.accuracy", 3),
    ("TACTICAL", "goal point error (m)", "families.tactical_from_trajectory.goal_setting.goal_point_error_m", 3),
    ("TACTICAL", "tactical head lat macro-F1 (majority in notes)", "tactical.lat.macro_f1", 3),
    ("TACTICAL", "tactical head lon macro-F1 (majority in notes)", "tactical.lon.macro_f1", 3),
    ("STRATEGIC", "checkpoint position error @50 m (m)", "strategic.pos_err_m@50", 3),
    ("STRATEGIC", "checkpoint position error @80 m (m)", "strategic.pos_err_m@80", 3),
    ("STRATEGIC", "checkpoint heading error @50 m (deg)", "strategic.head_err_deg@50", 3),
    ("CONSISTENCY", "consistency (a) violation rate, any", "consistency_a.any.ci", 3),
)


def nav_used(v):
    """compare.json's paired dicts carry ``mean`` (v8f_mini_eval.paired); the taniteval form carries ``delta``.  Both are accepted."""
    d = g(v, "strategic_tactical_from_compare.nav_used_M1navshuf_minus_M1_pos_err") or {}
    return {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in x.items() if kk in ("mean", "delta", "lo", "hi", "separated")}
            for k, x in d.items()}


def render(v, runs, p10_raw, p10, gtprov, blocks, ctl, checks, notes):
    tags = [t for t in (f"{a}_s{s}" for a in ARMS + CONTROLS for s in SEEDS) if t in runs]
    L = ["# v8F-mini full corpus: RESULT (PREREG_V8F_MINI Amendment 4)", "",
         "_Generated by `code/v8f_mini_full_report.py` from box 2's chain output. Every bar is COPIED from `verdict.json`; the paired family "
         "rows are DESCRIPTIVE (no bar was registered on them)._", "",
         f"* **Tier:** verdict `{v.get('tier')}`; the eval files stamp `{g(runs['M1_s0']['raw'], 'tier') if runs['M1_s0']['raw'] else '--'}`.",
         f"* **Estimator:** {v.get('estimator')}.",
         f"* **Provenance:** `{json.dumps(v.get('provenance'))}`.",
         f"* **Ground truth:** {gtprov['source']}; `{json.dumps(gtprov.get('files', gtprov.get('path')))}`.",
         f"* **P10-yaw-4512 predictions:** sha256 `{p10['sha256']}`; {p10['check']}.",
         f"* **Verdict re-derived from the eval files (to {TOL_VERDICT}):** " + "; ".join(checks) + ".",
         f"* **Summary (verdict):** {v.get('summary')}", ""]
    k1 = v["K1"]
    L += ["## H-MINI-K-1: ADE@6s arm - P10-yaw-4512, paired, both seeds separated and |delta| > 2F", "",
          "| arm | seed 0 | seed 1 | F (m) | 2F (m) | PASS |", "|---|---|---|---|---|---|"]
    for arm in ARMS:
        r = k1["per_arm"][arm]
        c = r["K1_ade6_minus_P10"]
        sep = lambda x: f"{x['mean']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]{' sep' if x['separated'] else ''}"   # noqa: E731
        L.append(f"| {arm} | {sep(c[0])} | {sep(c[1])} | {r['seed_floor_F_m']:.4f} | {2 * r['seed_floor_F_m']:.4f} | "
                 f"**{'PASS' if r['K1_pass'] else 'FAIL'}** |")
    L += ["", f"K-1 overall: **{'PASS' if k1['pass'] else 'FAIL'}** ({k1['multiplicity']}).", ""]
    w = v["WM1"]
    L += ["## H-MINI-WM-1: M2b - M1, ADE@6s, paired per seed", "", "| seed | M2b - M1 (m) | separated |", "|---|---|---|"]
    for c in w["per_seed"]:
        L.append(f"| {c['seed']} | {c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] | {c['separated']} |")
    L += ["", f"Seed floors F: `{json.dumps({k: round(x, 4) for k, x in w['seed_floor_F_m'].items()})}`; 2F = "
              f"{2 * w['seed_floor_F_m']['max']:.4f} m. WM-1: **{'PASS' if w['pass'] else 'FAIL'}** "
              f"({w['n_windows']} windows, {w['n_episodes']} episodes).", ""]
    cols = ["P10"] + tags
    src = {"P10": p10_raw, **{t: runs[t]["raw"] for t in tags}}
    L += ["## The four families, per run (point [lo, hi]; single-arm intervals)", "",
          "STRATEGIC is un-paused for v8F-mini (PI ruling of 2026-10-17), so it is read. P10 has no checkpoint or tactical head (`--`).", "",
          "| family | metric | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for fam, lab, path, dp in FAM_ROWS:
        L.append(f"| {fam} | {lab} | " + " | ".join(fmt(g(src[c], path), dp) if src[c] else "--" for c in cols) + " |")
    dk = g(runs["M1_s0"]["raw"], "families.longitudinal.distance_keeping") or {}
    L += ["", f"* Distance keeping is `{dk.get('status')}`: {dk.get('reason', '--')} (n = {dk.get('n')}). A work item, not a pass.",
          f"* Tactical-head majority-class macro-F1: lat {fmt(g(runs['M1_s0']['raw'], 'tactical.lat_majority.macro_f1'))}, "
          f"lon {fmt(g(runs['M1_s0']['raw'], 'tactical.lon_majority.macro_f1'))}.",
          f"* P10 consistency (a) rate: {fmt(v.get('P10_consistency_a_rate'))}.",
          f"* Nav used (M1-navshuf - M1 checkpoint error, seed 0): "
          f"`{json.dumps(nav_used(v))}`.",
          ""]
    L += ["## Paired LONGITUDINAL / LATERAL deltas (DESCRIPTIVE)", "",
          "Each cell is `delta [lo, hi]` of a mean ABSOLUTE error, paired over the 141 eval episodes (`ra1_hjepa_eval.paired_family_block`), "
          "so a NEGATIVE delta means the first run is closer to the human path. `*` marks a separated interval. "
          f"Control M1_s0 - M1_s0 reads exactly +0 [0, 0] on all {ctl['n_components']} components: {ctl['m1_minus_m1_ok']}.", ""]
    comps = [(f, n) for f in ("longitudinal", "lateral") for n in blocks[next(iter(blocks))][f]]
    L += ["| pair | " + " | ".join(n for _f, n in comps) + " |", "|---|" + "---|" * len(comps)]
    for name, b in blocks.items():
        cells = []
        for f, n in comps:
            c = b[f][n]
            if c.get("status") == "UNAVAILABLE":
                cells.append("UNAVAILABLE")
            else:
                dp = 5 if "curvature" in n else 4
                cells.append(f"{c['delta']:+.{dp}f} [{c['lo']:+.{dp}f}, {c['hi']:+.{dp}f}]{'*' if c['separated'] else ''}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "TACTICAL and STRATEGIC are not paired here: the eval files save no per-window manoeuvre classes. They are read per run above. "
              "A work item, not a pass.", "", "## Next (verdict)", ""] + [f"* {x}" for x in v.get("next", [])]
    if notes:
        L += ["", "## Author's reading", "", notes.strip()]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdict", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json")
    ap.add_argument("--notes")
    ap.add_argument("--p10-pred", default=str(P10_PRED))
    ap.add_argument("--eval-grid", default=str(RAW / "eval_grid_windows.json"))
    ap.add_argument("--eval-kin", default=str(RAW / "eval141_kin.npz"))
    ap.add_argument("--eval-join", default=str(RAW / "join_eval141.npz"))
    ap.add_argument("--gt-npz", help="TEST INPUT: npz with gt, eid, dt in place of the rebuild")
    ap.add_argument("--n-boot", type=int, default=B)
    a = ap.parse_args(argv)
    if not os.path.exists(a.verdict):
        refuse(f"{a.verdict} is missing")
    v = json.load(open(a.verdict, encoding="utf-8"))
    gt, eid, dt, gtprov = ground_truth(a)
    runs = load_runs(a.eval_dir, gt, eid)
    p10 = load_p10(a, gt, a.eval_dir)
    checks = check_verdict(v, runs, p10)
    E = _eval_module()
    blocks, ctl = paired(E, runs, p10, gt, eid, dt, a.n_boot)
    fr = os.path.join(a.eval_dir, "floors_report.json")
    p10_raw = g(json.load(open(fr, encoding="utf-8")), "floors.P10_refit") if os.path.exists(fr) else None
    notes = open(a.notes, encoding="utf-8").read() if a.notes else ""
    Path(a.out).write_text(render(v, runs, p10_raw, p10, gtprov, blocks, ctl, checks, notes), encoding="utf-8")
    if a.json:
        json.dump(dict(verdict_checks=checks, ground_truth=gtprov, p10_sha256=p10["sha256"], p10_check=p10["check"], paired=blocks,
                       control=ctl, max_ade_dev={t: r["max_ade_dev"] for t, r in runs.items()}),
                  open(a.json, "w", encoding="utf-8"), indent=1, default=float)
    print(f"wrote {a.out}: {v.get('summary')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
