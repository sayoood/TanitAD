#!/usr/bin/env python3
"""r9_e1_floor.py -- the refcv9 E1 REPLICATE-FLOOR read (REFCV9_DESIGN 4.3): C0 vs C1 (the same flags, training seed 0 vs 1) on
the SAME DEV windows, per metric: C0, C1, delta = C1 - C0, the 95 % paired-bootstrap CI, separated?, the FLOOR |C1 - C0| and
the KEEP THRESHOLD 2 x floor. Tier T0 (label level), perception only.

    python r9_e1_floor.py --eval-c0 EVAL_C0.json --eval-c1 EVAL_C1.json --paired EVAL_C1_vs_C0.json \\
                          [--rast-c0 <C0 rast dir> --rast-c1 <C1 rast dir> [--rast-names F] [--rast-stride 1] [--psw-key psw]] \\
                          --out-dir <dir>      # -> FLOOR.json + FLOOR.md

NO SECOND ESTIMATOR. Everything statistical is ``r9_e1_eval.py``'s and ``taniteval.ci``'s:
  * the BOX metrics' paired interval is the ``paired_vs_B`` block of ``r9_e1_eval.py --packs <C1 dump> --packs-b <C0 dump>``
    (A - B = C1 - C0), i.e. ``ci.paired_stat_cluster_bootstrap`` over the aligned windows, B 2000 seed 0;
  * the MAP metrics' interval is the same ``ci.paired_stat_cluster_bootstrap`` over per-EPISODE ``nearfield`` count blocks
    (``r9_e1_eval.nearfield_per_file`` / ``nearfield_stats`` -- the very objects its single-arm ``nearfield.ci`` bootstraps);
  * the POINT values are the single-arm evals' UNROUNDED ``point`` fields (the paired record rounds ``a``/``b``/``delta`` to
    4 dp, which would quantise a floor of a few 1e-5), cross-checked against the paired record to its rounding tolerance;
  * the keep rule is ``taniteval.ci.e1_floor_read`` (``lever_read`` below is a thin call into it).

SAME-WINDOWS IS ENFORCED, NOT ASSUMED. The two arms must hold the identical window set (paired alignment ``n_only_a`` =
``n_only_b`` = 0, equal ``n_windows`` / ``n_clips`` / window file / split / P=R gate / B / seed across the three evals; raster
name sets equal) or the script REFUSES (rc 2, ``FLOOR.json`` status REFUSED, no ``FLOOR.md`` table): a floor on different
windows is a difference of window sets, not of training seeds.

WHAT THE FLOOR IS AND IS NOT (H-ESTIM-SEED-1). The CI answers *another draw of EPISODES*; it is structurally blind to training
variance. The floor |C1 - C0| is ONE seed pair (n = 1) -- a single draw of the training noise, not an estimate of its spread:
a lever must clear 2 x this number AND a separated interval, and a floor of exactly 0 is reported ``floor_degenerate`` (two
runs agreeing to the last digit is not a floor).

FOUR FAMILIES (CLAUDE.md, binding). LONGITUDINAL / LATERAL / TACTICAL are stated "not computable" with the reason and n = 0
(the E1 rig builds no planner); STRATEGIC is stated PAUSED with the PI ruling. Never omitted, never a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9_e1_eval as ev                                       # noqa: E402  (the evaluator whose estimator this reuses)

SCHEMA = "r9e1-floor-v1"
EVAL_SCHEMA = ev.SCHEMA
TIER = "T0 (label level)"
DEFAULT_BOOT = 2000
DEFAULT_SEED = 0
#: the paired record rounds a / b / delta to >= 4 dp (ci._render_bounds), so a cross-check against it carries this tolerance
ROUND_TOL = 1.0e-4
KEEP_FACTOR = 2.0
ESTIMATOR = "paired_episode_cluster_bootstrap_stat"

FAMILIES = (
    ("LONGITUDINAL", "not computable: the E1 rig builds no planner (perception-only, T0 label level), n = 0"),
    ("LATERAL", "not computable: the E1 rig builds no planner (perception-only, T0 label level), n = 0"),
    ("TACTICAL", "not computable: the E1 rig builds no planner (perception-only, T0 label level), n = 0"),
    ("STRATEGIC", "PAUSED (PI 2026-09-19, PI_DECISION_QUEUE ITEM 25): the strategic layer is switched off until tactical and "
                  "operative drive well; reported as paused, not as a pass"),
)
KEEP_RULE = ("Keep a lever iff its primary is separated in the good direction AND |delta| > 2 x |C1 - C0| on that metric AND no "
             "other perception primary is separated worse beyond 2 x its floor (REFCV9_DESIGN 4.3; taniteval.ci.e1_floor_read).")


class FloorRefused(Exception):
    """The inputs do not describe the same windows / the same read: no floor is quoted."""


# --------------------------------------------------------------------------------------------------- small helpers
def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _num(x):
    """JSON-safe number: a non-finite value is None (the floor JSON carries no NaN token)."""
    return float(x) if _finite(x) else None


def higher_is_better(metric: str):
    """True / False / None (a count, not a quality). The direction is metadata for the keep rule; the floor itself is |C1-C0|."""
    if metric == "n_pos":
        return None
    if metric.startswith(("dx_", "boxes_per_object", "frac_objects_ge2")):
        return False
    return True                         # ap*, tp_class_acc, prec_gate_*, rec_gate_*, nf_prec*


def metric_group(metric: str) -> str:
    if metric.startswith("ap2m_all_NMS"):
        return "AP@2 m with post-hoc NMS (declared column)"
    if metric.startswith("ap2m"):
        return "AP@2 m, all and by range (head output, no NMS)"
    if metric.startswith("ap1m"):
        return "AP@1 m, all and by range (head output, no NMS)"
    if metric.startswith(("dx_", "tp_class_acc")):
        return "localisation / class of the 2 m greedy TPs"
    if metric.startswith(("prec_gate", "rec_gate", "boxes_per_object", "frac_objects_ge2")):
        return "P=R census at the FIXED gate (same gate for both arms)"
    if metric.startswith("nf_prec05"):
        return "corridor sidewalk precision, within 0.5 m of a GT sidewalk cell"
    if metric.startswith("nf_prec"):
        return "corridor sidewalk precision, exact GT sidewalk cell"
    return "other"


def _load_eval(path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("schema") != EVAL_SCHEMA:
        raise FloorRefused("%s: schema %r != %r (not an r9_e1_eval output)" % (path, d.get("schema"), EVAL_SCHEMA))
    return d


def _check(checks: list, name: str, ok: bool, detail: str) -> None:
    checks.append({"check": name, "ok": bool(ok), "detail": detail})


# --------------------------------------------------------------------------------------------------- comparability
def comparability(e0: dict, e1: dict, ep: dict, boot: int, seed: int) -> list:
    """Every check that makes C0, C1 and the paired record ONE read of the SAME windows. A failed check refuses the floor."""
    cs = []
    pv = ep.get("paired_vs_B")
    _check(cs, "paired record present", isinstance(pv, dict) and isinstance(pv.get("boxes"), dict),
           "the paired file needs r9_e1_eval --packs <C1 dump> --packs-b <C0 dump>")
    if not (isinstance(pv, dict) and isinstance(pv.get("boxes"), dict)):
        return cs
    al = pv.get("alignment", {})
    _check(cs, "same windows: paired alignment drops nothing", al.get("n_only_a") == 0 and al.get("n_only_b") == 0,
           "alignment %s" % json.dumps(al))
    for key in ("n_windows", "n_clips"):
        _check(cs, "equal %s (C0 / C1 / paired)" % key, e0.get(key) == e1.get(key) == ep.get(key),
               "%s / %s / %s" % (e0.get(key), e1.get(key), ep.get(key)))
    _check(cs, "paired common windows == n_windows", al.get("n_common") == e0.get("n_windows"),
           "n_common %s vs n_windows %s" % (al.get("n_common"), e0.get("n_windows")))
    _check(cs, "equal window file / split", (e0.get("windows_file"), e0.get("split")) == (e1.get("windows_file"), e1.get("split"))
           == (ep.get("windows_file"), ep.get("split")),
           "%s" % [(x.get("windows_file"), x.get("split")) for x in (e0, e1, ep)])
    _check(cs, "equal P=R gate (the census metrics are read at ONE operating point)",
           e0.get("gate") == e1.get("gate") == ep.get("gate"), "%s" % [x.get("gate", {}).get("value") for x in (e0, e1, ep)])
    _check(cs, "paired file's first arm is the C1 dump", ep.get("packs") == e1.get("packs"),
           "paired packs %s vs C1 packs %s" % (ep.get("packs"), e1.get("packs")))
    _check(cs, "C0 and C1 are different dump files", e0.get("packs") != e1.get("packs"), "%s" % e0.get("packs"))
    nb = {int(r.get("n_boot", -1)) for r in pv["boxes"].values() if isinstance(r, dict)}
    nbs = {int(r.get("n_boot", -1)) for r in e0["boxes"].values() if isinstance(r, dict)} | \
          {int(r.get("n_boot", -1)) for r in e1["boxes"].values() if isinstance(r, dict)}
    _check(cs, "B == %d everywhere" % boot, nb == {boot} and nbs == {boot}, "paired %s, single-arm %s" % (sorted(nb), sorted(nbs)))
    _check(cs, "seed == %d in the recorded estimators" % seed,
           all(("seed=%d" % seed) in str(x.get("estimator", "")) for x in (e0, e1)), str([x.get("estimator") for x in (e0, e1)]))
    ks = [set(x["boxes"]) for x in (e0, e1)] + [set(pv["boxes"])]
    _check(cs, "the same metric set in all three", ks[0] == ks[1] == ks[2], "%d / %d / %d keys" % tuple(len(k) for k in ks))
    return cs


# --------------------------------------------------------------------------------------------------- rows
def make_row(metric: str, c0, c1, paired: dict, *, n_windows: int, n_episodes: int, unit: str, family: str,
             seed: int = DEFAULT_SEED) -> dict:
    """ONE metric's floor row. ``c0`` / ``c1`` are the UNROUNDED single-arm points; ``paired`` is the ci paired record."""
    est = "%s B=%s seed=%s" % (ESTIMATOR, paired.get("n_boot", "?"), seed)
    row = {"metric": metric, "family": family, "group": metric_group(metric), "higher_is_better": higher_is_better(metric),
           "n_windows": int(n_windows), "n_episodes": int(n_episodes), "n_unit": unit, "estimator": est}
    if paired.get("undefined") or not (_finite(c0) and _finite(c1)):
        row.update(undefined=True, c0=_num(c0), c1=_num(c1), separated=False,
                   reason="undefined on these windows (no positive in the band / no flag): reported, never read as 0")
        return row
    delta = float(c1) - float(c0)
    floor = abs(delta)
    row.update(undefined=False, c0=float(c0), c1=float(c1), delta=delta, ci_lo=float(paired["lo"]), ci_hi=float(paired["hi"]),
               separated=bool(paired["separated"]), floor=floor, keep_threshold=KEEP_FACTOR * floor,
               floor_degenerate=bool(floor == 0.0), ci_degenerate=bool(paired.get("degenerate", False)))
    return row


def _crosscheck(row: dict, paired: dict, problems: list) -> None:
    if row.get("undefined"):
        return
    for k, v in (("a", row["c1"]), ("b", row["c0"]), ("delta", row["delta"])):
        if not _finite(paired.get(k)) or abs(float(paired[k]) - v) > ROUND_TOL:
            problems.append("%s: paired %s %r != single-arm %.9g (tol %g): the three evals are not one read (gate / windows / "
                            "arm order)" % (row["metric"], k, paired.get(k), v, ROUND_TOL))


def box_rows(e0: dict, e1: dict, ep: dict, problems: list, seed: int = DEFAULT_SEED) -> dict:
    rows = {}
    pv = ep["paired_vs_B"]["boxes"]
    for m in sorted(pv):
        rec = pv[m]
        c0, c1 = e0["boxes"][m].get("point"), e1["boxes"][m].get("point")
        # NOT rec["n_windows"]: in the paired record that field is len(items) of the bootstrap, i.e. the number of per-CLIP units
        # (r9_e1_eval pools windows by clip before calling ci.paired_stat_cluster_bootstrap); the window count is the eval's own.
        r = make_row(m, c0, c1, rec, n_windows=e0["n_windows"], n_episodes=rec.get("n_episodes", ep.get("n_clips")),
                     unit="windows", family="BOXES", seed=seed)
        _crosscheck(r, rec, problems)
        rows[m] = r
    return rows


def nearfield_rows(rast_c0, rast_c1, *, names_file=None, stride=1, psw_key="psw", boot=DEFAULT_BOOT, seed=DEFAULT_SEED,
                   e0=None, e1=None, problems=None) -> tuple[dict, dict]:
    """The paired near-field corridor precision: the SAME ``nearfield_stats`` over per-EPISODE count blocks that r9_e1_eval's
    single-arm ``nearfield.ci`` bootstraps, through ``ci.paired_stat_cluster_bootstrap``. -> (rows, meta)."""
    ci = ev._import_ci()
    problems = problems if problems is not None else []

    def listing(d):
        return {p.name: str(p) for p in Path(d).glob("*.npz")}

    f0, f1 = listing(rast_c0), listing(rast_c1)
    if names_file:
        keep = {ln.strip() for ln in Path(names_file).read_text(encoding="utf-8").splitlines() if ln.strip()}
        f0 = {k: v for k, v in f0.items() if Path(k).stem in keep}
        f1 = {k: v for k, v in f1.items() if Path(k).stem in keep}
    common = sorted(set(f0) & set(f1))
    meta = {"n_files_c0": len(f0), "n_files_c1": len(f1), "n_common": len(common), "only_c0": len(set(f0) - set(f1)),
            "only_c1": len(set(f1) - set(f0)), "stride": int(stride), "psw_key": psw_key}
    if meta["only_c0"] or meta["only_c1"] or not common:
        problems.append("raster name sets differ (only C0 %d, only C1 %d, common %d): not the same windows"
                        % (meta["only_c0"], meta["only_c1"], len(common)))
        return {}, meta
    n0, acc0, _ = ev.nearfield_per_file([f0[n] for n in common], stride, psw_key)
    n1, acc1, _ = ev.nearfield_per_file([f1[n] for n in common], stride, psw_key)
    if n0 != n1 or not n0:
        problems.append("rasters with a gt plane differ between the arms (%d vs %d): the GT plane is the label and must be the same"
                        % (len(n0), len(n1)))
        return {}, meta
    eps = [ev.nearfield_episode_of(n) for n in n0]
    ue = sorted(set(eps))
    ua = [sum(acc0[i] for i in range(len(n0)) if eps[i] == e) for e in ue]
    ub = [sum(acc1[i] for i in range(len(n0)) if eps[i] == e) for e in ue]
    pair = ci.paired_stat_cluster_bootstrap(ub, ua, ue, ev.nearfield_stats, n_boot=boot, seed=seed)      # C1 - C0
    p0, p1 = ev.nearfield_stats(ua), ev.nearfield_stats(ub)
    meta.update(n_files=len(n0), n_episodes=len(ue))
    rows = {}
    for m in sorted(pair):
        r = make_row(m, p0[m], p1[m], pair[m], n_windows=len(n0), n_episodes=len(ue), unit="raster files", family="MAP", seed=seed)
        _crosscheck(r, pair[m], problems)
        rows[m] = r
    for tag, e in (("C0", e0), ("C1", e1)):                    # the single-arm nearfield block, when the eval had rasters
        nf = (e or {}).get("nearfield")
        if nf is None:
            continue
        if nf.get("n_files") != len(n0) or nf.get("n_episodes") != len(ue):
            problems.append("%s eval's nearfield saw %s files / %s episodes, the floor read %d / %d (a different --rast-names / "
                            "--rast-stride?)" % (tag, nf.get("n_files"), nf.get("n_episodes"), len(n0), len(ue)))
            continue
        for m, r in rows.items():
            pt = nf["ci"][m].get("point")
            want = r["c0"] if tag == "C0" else r["c1"]
            if _finite(pt) and _finite(want) and abs(pt - want) > 1e-9:
                problems.append("%s: %s eval nearfield point %.12g != floor read %.12g" % (m, tag, pt, want))
    return rows, meta


# --------------------------------------------------------------------------------------------------- the read
def build_floor(eval_c0: str, eval_c1: str, paired: str, *, rast_c0=None, rast_c1=None, rast_names=None, rast_stride=1,
                psw_key="psw", boot=DEFAULT_BOOT, seed=DEFAULT_SEED) -> dict:
    e0, e1, ep = _load_eval(eval_c0), _load_eval(eval_c1), _load_eval(paired)
    out = {"schema": SCHEMA, "tier": TIER, "status": "OK", "keep_rule": KEEP_RULE, "keep_factor": KEEP_FACTOR,
           "estimator": {"name": ESTIMATOR, "B": boot, "seed": seed,
                         "answers": "another draw of EPISODES, NOT another training run: the floor |C1-C0| is ONE seed pair (n = 1)"},
           "inputs": {k: {"path": str(p), "sha256": _sha256(p)} for k, p in (("eval_c0", eval_c0), ("eval_c1", eval_c1), ("paired", paired))},
           "families": [{"family": f, "status": s} for f, s in FAMILIES]}
    checks = comparability(e0, e1, ep, boot, seed)
    out["comparability"] = checks
    problems = ["comparability: %s -- %s" % (c["check"], c["detail"]) for c in checks if not c["ok"]]
    out["n_windows"], out["n_clips"], out["gate"] = e0.get("n_windows"), e0.get("n_clips"), e0.get("gate")
    rows = {}
    if not problems:
        rows.update(box_rows(e0, e1, ep, problems, seed))
    meta = None
    if (rast_c0 is None) != (rast_c1 is None):
        problems.append("--rast-c0 and --rast-c1 go together")
    elif rast_c0 is not None and not problems:
        mrows, meta = nearfield_rows(rast_c0, rast_c1, names_file=rast_names, stride=rast_stride, psw_key=psw_key, boot=boot,
                                     seed=seed, e0=e0, e1=e1, problems=problems)
        rows.update(mrows)
    out["map_inputs"] = meta
    out["rows"] = rows
    if problems:
        out["status"] = "REFUSED"
        out["problems"] = problems
        out["rows"] = {}
    out["summary"] = _summary(rows) if rows else None
    return out


def _summary(rows: dict) -> dict:
    live = [r for r in rows.values() if not r.get("undefined") and r["higher_is_better"] is not None]
    return {"n_metrics": len(rows), "n_defined": len(live), "n_undefined": sum(1 for r in rows.values() if r.get("undefined")),
            "n_separated_c1_vs_c0": sum(1 for r in live if r["separated"]),
            "n_floor_degenerate": sum(1 for r in live if r["floor_degenerate"])}


def lever_read(floor: dict, metric: str, delta: float, separated: bool) -> dict:
    """Judge ONE lever metric against this floor: a thin call into ``taniteval.ci.e1_floor_read`` (the keep rule is its, not ours)."""
    r = floor["rows"][metric]
    if r.get("undefined") or r["higher_is_better"] is None:
        raise ValueError("%s has no floor (undefined / a count)" % metric)
    return ev._import_ci().e1_floor_read(delta, separated, r["c0"], r["c1"], higher_is_better=bool(r["higher_is_better"]))


# --------------------------------------------------------------------------------------------------- rendering
def _f(x, nd=5) -> str:
    return "n/a" if x is None else "%.*f" % (nd, x)


def _table(rows: list, tier_line: str) -> list:
    out = [tier_line, "",
           "| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        n = "%d %s / %d episodes" % (r["n_windows"], r["n_unit"], r["n_episodes"])
        if r.get("undefined"):
            out.append("| `%s` | %s | %s | undefined | | | | | %s |" % (r["metric"], _f(r["c0"]), _f(r["c1"]), n))
            continue
        out.append("| `%s` | %s | %s | %+.6f | [%.6g, %.6g]%s | %s | %s%s | %s | %s |" % (
            r["metric"], _f(r["c0"]), _f(r["c1"]), r["delta"], r["ci_lo"], r["ci_hi"],
            " (degenerate)" if r["ci_degenerate"] else "", "yes" if r["separated"] else "no", _f(r["floor"], 6),
            " (DEGENERATE: not a floor)" if r["floor_degenerate"] else "", _f(r["keep_threshold"], 6), n))
    return out


def render_md(f: dict) -> str:
    L = ["# E1 replicate floor -- C0 vs C1 (same flags, training seed 0 vs 1), DEV", ""]
    if f["status"] != "OK":
        L += ["**STATUS: REFUSED -- no floor is quoted.**", ""] + ["* " + p for p in f.get("problems", [])]
        return "\n".join(L) + "\n"
    g = f.get("gate") or {}
    est = f["estimator"]
    L += ["Tier: **%s** (perception only). Estimator: **%s**, B = %d, seed %d -- it answers *another draw of EPISODES*, not another "
          "training run." % (TIER, est["name"], est["B"], est["seed"]),
          "Windows: %s DEV windows / %s episodes (clips), identical for both arms (paired alignment drops none). Fixed P=R gate "
          "for the census rows: %s (%s)." % (f["n_windows"], f["n_clips"], g.get("value"), g.get("source")), "",
          "## The keep rule (REFCV9_DESIGN 4.3)", "", KEEP_RULE, "",
          "`floor` = abs(C1 - C0) is ONE seed pair (n = 1): a single draw of the training noise, not its spread. A floor of exactly "
          "0 is **degenerate** (two runs agreeing to the last digit), and the keep decision then rests on separation alone "
          "(`taniteval.ci.e1_floor_read`; `r9_e1_floor.lever_read` calls it).", "",
          "## The four metric families (CLAUDE.md, binding)", "", "| family | status |", "|---|---|"]
    L += ["| %s | %s |" % (x["family"], x["status"]) for x in f["families"]]
    L += ["", "Perception is the family this rig reads: boxes and the near-field map raster, below."]
    for fam, title in (("BOXES", "Perception -- boxes (T0)"), ("MAP", "Perception -- map raster, corridor |y| < 3 m (T0)")):
        rows = [r for r in f["rows"].values() if r["family"] == fam]
        if not rows:
            if fam == "MAP":
                L += ["", "## " + title, "", "Not read: no raster directories were given (`--rast-c0` / `--rast-c1`)."]
            continue
        L += ["", "## " + title]
        for grp in dict.fromkeys(r["group"] for r in rows):
            sub = [r for r in rows if r["group"] == grp]
            n0 = sub[0]
            L += ["", "### " + grp, ""] + _table(sub, "Tier %s | estimator %s | n = %d %s / %d episodes" % (
                TIER, n0["estimator"], n0["n_windows"], n0["n_unit"], n0["n_episodes"]))
    s = f.get("summary") or {}
    L += ["", "## Reading notes", "",
          "* %s of %s defined metrics have C1 vs C0 separated by the episode bootstrap. A separated C1 - C0 is NOT a failure of the "
          "rig: it is the proof (H-ESTIM-SEED-1) that a separated interval alone cannot certify a lever -- which is why the keep rule "
          "adds the 2 x floor clause." % (s.get("n_separated_c1_vs_c0"), s.get("n_defined")),
          "* %s defined metrics have floor exactly 0 (degenerate)." % s.get("n_floor_degenerate"),
          "* Map rows are what `r9_e1_eval.py` emits: sidewalk-flag (`psw`) precision on GT-seen cells of the corridor; the lift-valid "
          "subset and a sidewalk+edge union are NOT computed here (not emitted by the evaluator).",
          "* The census rows (`prec_gate_*`, `rec_gate_*`, `boxes_per_object_*`, `frac_objects_ge2_*`) are read at the fixed gate in "
          "the header for both arms, not at each arm's own P=R point.",
          "* Inputs: " + "; ".join("%s sha256 %s" % (k, v["sha256"][:12]) for k, v in f["inputs"].items()) + "."]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-c0", required=True, help="r9_e1_eval.py single-arm output for C0")
    ap.add_argument("--eval-c1", required=True, help="r9_e1_eval.py single-arm output for C1")
    ap.add_argument("--paired", required=True, help="r9_e1_eval.py --packs <C1 dump> --packs-b <C0 dump> output (delta = C1 - C0)")
    ap.add_argument("--rast-c0")
    ap.add_argument("--rast-c1")
    ap.add_argument("--rast-names")
    ap.add_argument("--rast-stride", type=int, default=1)
    ap.add_argument("--psw-key", default="psw")
    ap.add_argument("--boot", type=int, default=DEFAULT_BOOT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    try:
        floor = build_floor(a.eval_c0, a.eval_c1, a.paired, rast_c0=a.rast_c0, rast_c1=a.rast_c1, rast_names=a.rast_names,
                            rast_stride=a.rast_stride, psw_key=a.psw_key, boot=a.boot, seed=a.seed)
    except FloorRefused as e:
        floor = {"schema": SCHEMA, "tier": TIER, "status": "REFUSED", "problems": [str(e)], "rows": {}}
    (od / "FLOOR.json").write_text(json.dumps(floor, indent=1, default=float, allow_nan=False) + "\n", encoding="utf-8")
    (od / "FLOOR.md").write_text(render_md(floor), encoding="utf-8")
    if floor["status"] != "OK":
        print("[r9-e1-floor] REFUSED: %s" % "; ".join(floor["problems"]), file=sys.stderr)
        return 2
    s = floor["summary"]
    print("wrote %s: %d metrics (%d defined, %d separated C1 vs C0, %d floor-degenerate); status OK"
          % (od / "FLOOR.json", s["n_metrics"], s["n_defined"], s["n_separated_c1_vs_c0"], s["n_floor_degenerate"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
