"""r9_e2_dose_verdict.py -- the PRE-REGISTERED DOSE-HYPOTHESIS reader of the refcv9 E2 K2DC arm (K2DC vs K2 = K2M1N20, with K0 as the floor).

Encodes PREREG_K2DC_2026-11-12.md "Read -> Dose hypothesis" EXACTLY (frozen; PI-approved "Approve K2DC"); nothing here may move after any K2DC step:

  DIAGNOSTIC, one training run per arm (H-ESTIM-SEED-1).  Per inference seed s in {0, 1}, DEV windows, paired episode-cluster bootstrap (taniteval.ci), delta = A - B:
  * SUPPORTED    at BOTH seeds: (i) K2DC - K0 LAT_cross_mae is NOT separated worse  AND  (ii) K2DC - K2 LAT_cross_mae <= -2.5 mm (half of K2 - K0 at seed 1)
                 AND  (iii) the frame-to-frame |d e| (lat_jitter.py `dlag`) K2DC - K2 is NEGATIVE.
  * REFUTED      at BOTH seeds: the K2DC - K2 LATERAL DISTANCE rows (LAT_cross_mae_m and LAT_cross_abs_final_m, the two lateral rows in metres) are within +-1 mm.
  * INCONCLUSIVE anything between; the next step is a replicate arm (same flags, seed + 1), NOT a reinterpretation.
  * DOSE-VOID    the run is valid for the dose question only if the trainer's logged trunk share (metrics.jsonl `dc_share_read`) lies in [0.05, 0.15] at >= 2 probes
                 at steps >= 3,000.  Outside => reported DOSE-VOID (the controller did not hold), never silently re-run.  The three outcomes above are still computed and
                 printed under `outcome_if_valid`; `outcome` reads DOSE-VOID.
  All four metric families are reported per seed (K2DC - K0 and K2DC - K2), per family and never pooled; STRATEGIC = PAUSED (PI 2026-09-19, ITEM 25).  The map primary / box AP /
  S6c rate K2DC - K2 are reported with NO committed direction (the PREREG says so), from the two reads' EVAL / S6c files when supplied.

The E2 GO rule itself (REFCV9_DESIGN 4.4) is NOT re-implemented here: `r9_e2_verdict.py` reads K2DC vs K0 exactly as it read K2.

Ambiguities in the PREREG text, resolved here and STATED in the output (`resolutions`): (1) "at both seeds" governs the whole SUPPORTED sentence; (2) "lateral rows" in the REFUTED
clause = the two lateral rows that are lengths (a heading in degrees or a curvature in 1/m has no millimetre band); (3) "separated worse" = CI excludes zero with delta > 0 (lower is
better); (4) the jitter clause is the POINT delta < 0 (the PREREG sets no CI requirement on it; its interval is reported).

CPU only.  Usage:
  python r9_e2_dose_verdict.py --k2dc-pf <E2_READ_K2DC/E2_K2_pf> --k2-pf <E2_READ_K2M1N20/E2_K2_pf> --k0-pf <E2_READ_K2M1N20/E2_K0_pf> --dev-split DEV_HOLD_SPLIT.json \
      --metrics <runs/r9e2-K2DC/metrics.jsonl> --out-dir <dir> [--k2dc-read <E2_READ_K2DC/read> --k2-read <E2_READ_K2M1N20/read>]
  (each --*-pf dir holds dump_s0 / dump_s1; PYTHONPATH = <tree>/stack:<tree>/stack/scripts:<tree>/taniteval, `;` on Windows.)
EXIT: 0 SUPPORTED | 1 REFUTED | 2 INCONCLUSIVE | 3 DOSE-VOID | 4 REFUSED.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SCHEMA = "r9e2-dose-verdict-v1"
PREREG = "TanitAD-artifacts/dagland/e2/PREREG_K2DC_2026-11-12.md"
SEEDS = ("0", "1")
CROSS = "LAT_cross_mae_m"
LAT_DIST_ROWS = ("LAT_cross_mae_m", "LAT_cross_abs_final_m")
BAR_M = 0.0025              # SUPPORTED: K2DC - K2 LAT_cross_mae <= -2.5 mm (half of K2 - K0 at seed 1, which was +5.0 mm)
BAND_M = 0.0010             # REFUTED: K2DC - K2 lateral rows within +-1 mm
EPS = 1e-9                  # float slack on the two inclusive bounds (the CI rounds its point to 4+ digits)
SHARE_BAND = (0.05, 0.15)   # in-training validity: logged trunk share, design 2.2's dose band
SHARE_MIN_STEP = 3000
SHARE_MIN_PROBES = 2
EXIT = {"SUPPORTED": 0, "REFUTED": 1, "INCONCLUSIVE": 2, "DOSE-VOID": 3}
RESOLUTIONS = ("'at both seeds' governs the whole SUPPORTED sentence",
               "REFUTED 'lateral rows' = LAT_cross_mae_m and LAT_cross_abs_final_m (the lateral rows that are lengths)",
               "'separated worse' = the paired CI excludes zero AND delta > 0 (lower is better)",
               "jitter clause = the POINT delta of the frame-to-frame |d e| (K2DC - K2) < 0; its interval is reported, no CI is required by the PREREG")


def _fail(msg: str):
    raise SystemExit(f"[r9-e2-dose-verdict] REFUSED: {msg}")


# ---------------------------------------------------------------------------------------------------------------- #
# the decision: pure functions of the per-seed numbers                                                              #
# ---------------------------------------------------------------------------------------------------------------- #
def _row(d: dict, key: str, where: str) -> dict:
    try:
        r = d[key]
        float(r["delta"])
    except (KeyError, TypeError, ValueError):
        _fail(f"{where}: no usable row {key!r} (got {sorted(d)[:6]})")
    return r


def seed_clauses(ps: dict, seed: str) -> dict:
    """ps = {"dc_vs_k0": {row_id: {delta, lo, hi, separated}}, "dc_vs_k2": {...}, "jitter_dlag_dc_vs_k2": {delta, lo, hi, separated}}."""
    w = _row(ps["dc_vs_k0"], CROSS, f"seed {seed} K2DC-K0")
    d = _row(ps["dc_vs_k2"], CROSS, f"seed {seed} K2DC-K2")
    j = ps["jitter_dlag_dc_vs_k2"]
    try:
        jd = float(j["delta"])
    except (KeyError, TypeError, ValueError):
        _fail(f"seed {seed}: no usable jitter row (got {j!r})")
    not_worse = not (bool(w["separated"]) and float(w["delta"]) > 0.0)
    bar = float(d["delta"]) <= -BAR_M + EPS
    jit = jd < 0.0
    band = {r: abs(float(_row(ps["dc_vs_k2"], r, f"seed {seed} K2DC-K2")["delta"])) <= BAND_M + EPS for r in LAT_DIST_ROWS}
    return {"i_dc_k0_cross_not_separated_worse": not_worse, "ii_dc_k2_cross_le_minus_2p5mm": bar, "iii_jitter_delta_negative": jit,
            "supported": bool(not_worse and bar and jit), "refuted_band": band, "refuted": bool(all(band.values()))}


def decide(per_seed: dict) -> dict:
    if set(per_seed) != set(SEEDS):
        _fail(f"inference seeds 0 AND 1 are both required, got {sorted(per_seed)}")
    cl = {s: seed_clauses(per_seed[s], s) for s in SEEDS}
    sup = all(cl[s]["supported"] for s in SEEDS)
    ref = all(cl[s]["refuted"] for s in SEEDS)
    if sup and ref:                                              # cannot happen (-2.5 mm vs +-1 mm); a contradiction in the constants is refused, not resolved
        _fail("SUPPORTED and REFUTED both hold: the bar and the band overlap")
    out = "SUPPORTED" if sup else "REFUTED" if ref else "INCONCLUSIVE"
    why = {"SUPPORTED": "at both seeds K2DC-K0 LAT_cross_mae is not separated worse, K2DC-K2 LAT_cross_mae <= -2.5 mm and the frame-to-frame |d e| K2DC-K2 is negative",
           "REFUTED": "at both seeds the K2DC-K2 lateral distance rows are within +-1 mm: the dose is ELIMINATED; next arm = K2 minus --r9-hist-supervise-geom, then K2 minus map-banded "
                      "(each its own pre-registration)",
           "INCONCLUSIVE": "between the bars: next step = a replicate arm (same flags, seed + 1), NOT a reinterpretation"}[out]
    return {"outcome_if_valid": out, "reason": why, "per_seed": cl}


# ---------------------------------------------------------------------------------------------------------------- #
# the in-training validity check                                                                                    #
# ---------------------------------------------------------------------------------------------------------------- #
def void_check(metrics_path) -> dict:
    """Reads the trainer's metrics.jsonl: rows carrying `dc_share_read` (map_dose_ctl.MapDoseController.fire, merged into the metrics row at the firing step)."""
    if metrics_path is None:
        return {"status": "NOT-CHECKED", "reason": "no --metrics given: the in-training validity condition was not read; the outcome is UNCHECKED-VOID", "valid": None}
    p = Path(metrics_path)
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        _fail(f"--metrics {p}: unreadable ({e})")
    probes = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if isinstance(r, dict) and "dc_share_read" in r and "step" in r:
            probes.append((int(r["step"]), float(r["dc_share_read"])))
    late = [(s, v) for s, v in probes if s >= SHARE_MIN_STEP]
    inside = [(s, v) for s, v in late if SHARE_BAND[0] <= v <= SHARE_BAND[1]]
    out = {"metrics": str(p), "n_probes_total": len(probes), "n_probes_step_ge_3000": len(late), "n_inside_band": len(inside), "band": list(SHARE_BAND),
           "rule": f">= {SHARE_MIN_PROBES} probes at step >= {SHARE_MIN_STEP} inside [{SHARE_BAND[0]}, {SHARE_BAND[1]}]",
           "probes_step_ge_3000": [{"step": s, "share": round(v, 5)} for s, v in late]}
    if not probes:
        out.update(status="VOID", valid=False, reason="metrics.jsonl holds no dc_share_read row: the controller never fired (or the log format changed): refusing to call the dose held")
    else:
        ok = len(inside) >= SHARE_MIN_PROBES
        out.update(status="HELD" if ok else "VOID", valid=ok, reason="" if ok else "the controller did not hold the share inside the band: DOSE-VOID")
    return out


# ---------------------------------------------------------------------------------------------------------------- #
# the numbers: families (r9_e2_families.build) and the jitter (lat_jitter.py's dlag)                                #
# ---------------------------------------------------------------------------------------------------------------- #
def _families():
    import r9_e2_families as F
    return F


def jitter_dlag(a_dir, b_dir, dev: set, n_boot: int = 2000, seed: int = 0) -> dict:
    """lat_jitter.py's `dlag`: signed final-waypoint lateral error e_w = P[w, -1, 1] - G[w, -1, 1]; per episode (windows ordered by ws) |e_{w+1} - e_w|; paired episode-cluster
    bootstrap, delta = A - B.  The arms must hold the SAME windows and the SAME GT (REFUSED otherwise)."""
    import numpy as np
    from taniteval import ci
    F = _families()

    def per_episode(d):
        dd, _ = F.restrict_dev(F.load_dump(d), dev)
        by = defaultdict(list)
        for i, (c, w) in enumerate(dd["keys"]):
            by[c].append((w, i))
        out = {}
        for c, lst in by.items():
            lst.sort()
            idx = [i for _, i in lst]
            out[c] = ([w for w, _ in lst], dd["P"][idx, -1, 1] - dd["G"][idx, -1, 1], dd["G"][idx])
        return out
    A, B = per_episode(a_dir), per_episode(b_dir)
    if set(A) != set(B):
        _fail(f"jitter: episode sets differ ({len(set(A) ^ set(B))} episodes only in one arm)")
    a, b, eid = [], [], []
    for c in sorted(A):
        if A[c][0] != B[c][0]:
            _fail(f"jitter: episode {c} holds different windows in the two arms")
        if float(np.abs(A[c][2] - B[c][2]).max()) > 1e-4:
            _fail(f"jitter: episode {c} GT differs between the arms (different label clocks)")
        da, db = np.abs(np.diff(A[c][1])), np.abs(np.diff(B[c][1]))
        a += list(da)
        b += list(db)
        eid += [c] * len(da)
    if not a:
        _fail("jitter: no consecutive-window pair in any episode")
    a, b = np.asarray(a), np.asarray(b)
    r = ci.paired_episode_cluster_bootstrap(a, b, eid, n_boot=n_boot, seed=seed)
    return {"row": "frame-to-frame |d e| of the final-waypoint lateral error (lat_jitter.py dlag), m", "n": int(r["n_windows"]), "n_episodes": int(r["n_episodes"]),
            "a_mean": float(a.mean()), "b_mean": float(b.mean()), "delta": float(r["delta"]), "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"]),
            "better": "lower"}


def _slim(r: dict) -> dict:
    return {k: r[k] for k in ("delta", "lo", "hi", "separated", "better", "n", "n_episodes", "a_mean", "b_mean") if k in r}


def seed_numbers(k2dc_pf, k2_pf, k0_pf, dev_split, seed: str, n_boot: int = 2000) -> dict:
    F = _families()
    dev = F.load_dev(dev_split)
    sub = lambda root: Path(root) / f"dump_s{seed}"                                  # noqa: E731
    out = {"dumps": {"k2dc": str(sub(k2dc_pf)), "k2": str(sub(k2_pf)), "k0": str(sub(k0_pf))}}
    docs = {}
    for tag, b, lab in (("dc_vs_k0", k0_pf, "K0"), ("dc_vs_k2", k2_pf, "K2")):
        doc = F.build(sub(k2dc_pf), sub(b), mode="families", dev=dev, n_boot=n_boot, labels=("K2DC", lab))
        docs[tag] = doc
        out[tag] = {rid: _slim(r) for fam in F.FAMILIES for rid, r in doc["rows"][fam].items()}
        out[tag + "_ade_m"] = _slim(doc["ade_m"])
        out[tag + "_by_family"] = {fam: {rid: _slim(r) for rid, r in doc["rows"][fam].items()} for fam in F.FAMILIES}
        out[tag + "_by_family"]["STRATEGIC"] = dict(doc["rows"]["STRATEGIC"])
        out[tag + "_window_set_sha256"] = doc["window_set_sha256"]
        out[tag + "_n_windows_paired"] = doc["n_windows_paired"]
    if out["dc_vs_k0_window_set_sha256"] != out["dc_vs_k2_window_set_sha256"]:
        _fail(f"seed {seed}: the K2DC-K0 and K2DC-K2 pairings use different window sets")
    out["jitter_dlag_dc_vs_k2"] = jitter_dlag(sub(k2dc_pf), sub(k2_pf), dev, n_boot)
    out["jitter_dlag_dc_vs_k0"] = jitter_dlag(sub(k2dc_pf), sub(k0_pf), dev, n_boot)
    return out


def _dig(o, *ks):
    for k in ks:
        if not isinstance(o, dict) or k not in o:
            return None
        o = o[k]
    return o


def beside(k2dc_read, k2_read) -> dict:
    """Map primary / box AP (T0 evals) and the S6c sidewalk rate, K2DC - K2, NO committed direction (PREREG).  Same digs as r9_e2_verdict.build."""
    if not k2dc_read or not k2_read:
        return {"status": "NOT-SUPPLIED", "reason": "--k2dc-read / --k2-read not given"}
    a, b = Path(k2dc_read), Path(k2_read)
    res = {"status": "REPORTED", "direction": "none committed (PREREG: 'map primary K2DC - K2 reported with no committed direction')", "tier": "T0 label level (map, box); T1 (S6c)"}

    def ev(p):
        try:
            return json.loads((p / "EVAL_K2.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    ea, eb = ev(a), ev(b)
    for name, path in (("map_primary_nf_swedge_prec_lv_5_20", ("nearfield", "ci", "nf_swedge_prec_lv_5_20", "point")), ("box_ap2m_all", ("boxes", "ap2m_all", "point"))):
        va, vb = _dig(ea, *path), _dig(eb, *path)
        res[name] = {"k2dc": va, "k2": vb, "delta": (None if va is None or vb is None else float(va) - float(vb))}
    for s in SEEDS:
        def rate(p):
            try:
                j = json.loads((p / "s6c" / f"S6C_K2_s{s}.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            return j.get("rate") if "rate" in j else _dig(j, "map_compliance", "S6c", "arms", "BASE", "off_drivable", "rate")
        ra, rb = rate(a), rate(b)
        res[f"s6c_sidewalk_rate_s{s}"] = {"k2dc": ra, "k2": rb, "delta": (None if ra is None or rb is None else float(ra) - float(rb)), "better": "lower"}
    return res


# ---------------------------------------------------------------------------------------------------------------- #
def build(numbers: dict, void: dict, extra: dict | None = None) -> dict:
    d = decide({s: {k: numbers[s][k] for k in ("dc_vs_k0", "dc_vs_k2", "jitter_dlag_dc_vs_k2")} for s in SEEDS})
    if void.get("valid") is False:
        outcome = "DOSE-VOID"
    elif void.get("valid") is None:
        outcome = d["outcome_if_valid"] + " (UNCHECKED-VOID: the in-training share condition was not read)"
    else:
        outcome = d["outcome_if_valid"]
    doc = {"schema": SCHEMA, "class": "DIAGNOSTIC (one training run per arm, H-ESTIM-SEED-1)", "prereg": PREREG, "outcome": outcome, **d, "dose_validity": void,
           "thresholds": {"bar_m": -BAR_M, "band_m": BAND_M, "share_band": list(SHARE_BAND), "share_min_step": SHARE_MIN_STEP, "share_min_probes": SHARE_MIN_PROBES},
           "resolutions": list(RESOLUTIONS),
           "tier": "T1 (self-action open loop on logged eval139 frames; the `os` arm)", "windows": "DEV (DEV_HOLD_SPLIT.json); HOLD is never read",
           "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0, over EPISODES (clips); delta = A - B; it answers 'another draw of EPISODES', not 'another training run' "
                        "(H-ESTIM-SEED-1) and not 'another inference draw' (each seed is one call)",
           "families_note": "four families per seed, per family, never pooled: LONGITUDINAL / LATERAL / TACTICAL rows in `seeds.<s>.dc_vs_k2_by_family` and `dc_vs_k0_by_family`; ade_m beside; "
                            "STRATEGIC = PAUSED (PI 2026-09-19, ITEM 25), never a pass, never omitted",
           "STRATEGIC": {"status": "PAUSED", "reason": "PAUSED (PI 2026-09-19, PI_DECISION_QUEUE ITEM 25): the strategic layer is switched off until tactical and operative drive with high quality"},
           "seeds": numbers}
    if extra:
        doc["beside_no_direction"] = extra
    return doc


def md(doc: dict) -> str:
    L = [f"# K2DC dose hypothesis: **{doc['outcome']}**", "", f"{doc['reason']}", "",
         f"*{doc['class']}.* Pre-registration: `{doc['prereg']}`. Tier: {doc['tier']}. Windows: {doc['windows']}.", "", f"Estimator: {doc['estimator']}.", "",
         f"In-training validity (`dose_validity`): **{doc['dose_validity'].get('status')}** ({doc['dose_validity'].get('rule', '')}; "
         f"{doc['dose_validity'].get('n_inside_band', '-')} of {doc['dose_validity'].get('n_probes_step_ge_3000', '-')} late probes inside the band).", "",
         "## Pre-registered rows (delta = A - B, metres; lower is better)", "",
         "| seed | K2DC-K0 LAT_cross_mae | separated worse? | K2DC-K2 LAT_cross_mae | <= -2.5 mm | K2DC-K2 LAT_cross_abs_final | K2DC-K2 jitter abs(d e) | jitter < 0 | supported | refuted |",
         "|" + "---|" * 11]
    for s in SEEDS:
        n, c = doc["seeds"][s], doc["per_seed"][s]
        a, b, f, j = n["dc_vs_k0"][CROSS], n["dc_vs_k2"][CROSS], n["dc_vs_k2"]["LAT_cross_abs_final_m"], n["jitter_dlag_dc_vs_k2"]
        L.append(f"| {s} | {a['delta']:+.5f} [{a['lo']:+.5f}, {a['hi']:+.5f}] | {'no' if c['i_dc_k0_cross_not_separated_worse'] else 'YES'} | {b['delta']:+.5f} [{b['lo']:+.5f}, {b['hi']:+.5f}] "
                 f"| {'yes' if c['ii_dc_k2_cross_le_minus_2p5mm'] else 'no'} | {f['delta']:+.5f} [{f['lo']:+.5f}, {f['hi']:+.5f}] | {j['delta']:+.5f} [{j['lo']:+.5f}, {j['hi']:+.5f}] "
                 f"| {'yes' if c['iii_jitter_delta_negative'] else 'no'} | {'yes' if c['supported'] else 'no'} | {'yes' if c['refuted'] else 'no'} |")
    L += ["", "## All four families (K2DC - K2 | K2DC - K0), per seed, never pooled", ""]
    for s in SEEDS:
        L += [f"### seed {s}", "", "| family | row | K2DC - K2 | sep | K2DC - K0 | sep |", "|---|---|---|---|---|---|"]
        for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL"):
            for rid, r in doc["seeds"][s]["dc_vs_k2_by_family"][fam].items():
                r0 = doc["seeds"][s]["dc_vs_k0_by_family"][fam].get(rid, {})
                L.append(f"| {fam} | {rid} | {r['delta']:+.5f} | {'*' if r['separated'] else ''} | {r0.get('delta', float('nan')):+.5f} | {'*' if r0.get('separated') else ''} |")
        L += ["| STRATEGIC | - | PAUSED | | PAUSED | |", ""]
    b = doc.get("beside_no_direction")
    if b:
        L += ["## Beside (no committed direction)", "", "```", json.dumps(b, indent=1, default=float), "```", ""]
    L += ["## Resolutions of PREREG wording", ""] + [f"* {r}" for r in doc["resolutions"]]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--k2dc-pf", required=True, help="E2_READ_K2DC/E2_K2_pf (holds dump_s0, dump_s1)")
    ap.add_argument("--k2-pf", required=True, help="E2_READ_K2M1N20/E2_K2_pf")
    ap.add_argument("--k0-pf", required=True, help="E2_READ_K2M1N20/E2_K0_pf (the K0 the K2DC read reuses)")
    ap.add_argument("--dev-split", required=True)
    ap.add_argument("--metrics", help="runs/r9e2-K2DC/metrics.jsonl (the in-training share check); without it the outcome reads UNCHECKED-VOID")
    ap.add_argument("--k2dc-read", help="E2_READ_K2DC/read (EVAL_K2.json, s6c/) for the no-direction map / box / S6c rows")
    ap.add_argument("--k2-read", help="E2_READ_K2M1N20/read")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    try:
        numbers = {s: seed_numbers(a.k2dc_pf, a.k2_pf, a.k0_pf, a.dev_split, s, a.boot) for s in SEEDS}
        doc = build(numbers, void_check(a.metrics), beside(a.k2dc_read, a.k2_read))
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 4
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "DOSE_VERDICT.json").write_text(json.dumps(doc, indent=1, default=float) + "\n", encoding="utf-8")
    (out / "DOSE_VERDICT.md").write_text(md(doc), encoding="utf-8")
    print(f"[r9-e2-dose-verdict] {doc['outcome']} -> {out / 'DOSE_VERDICT.json'}")
    base = doc["outcome"].split(" ")[0]
    return EXIT.get(base, 4)


if __name__ == "__main__":
    raise SystemExit(main())
