#!/usr/bin/env python3
"""sweep_ood_verdicts.py — which COMMITTED artifacts carry a WRONG OOD verdict?

DEFECT 2, the sweep half. ``OODMap.ratio_arr`` maps a deviation to an ADE ratio
with ``np.interp``, which **CLAMPS** at ``|dlat| = 3.0 m`` / ``|dyaw| = 12 deg``.
Beyond the envelope the ratio **SATURATES**, so:

* every long-horizon OOD ratio in this program is a **LOWER BOUND**, and
* the ``ratio > ~1.5x`` criterion **structurally cannot fire** out of envelope.

E1a's rule was always the DISJUNCTION (``e1a_horizon.py:28-30``): ratio > ~1.5x
**OR steps leave the measured envelope**. Only the ratio half was implemented.

WHAT THIS DOES. Walks every git-tracked JSON under the repo, finds every node
that carries an OOD reading (``ood_peak_ratio`` and/or an ``EXTRAPOLATION_*``
fraction), re-adjudicates it under the FULL rule via
:func:`taniteval.ood.readjudicate` — pure arithmetic on fields the emitters
already wrote, **no tensors, no GPU, no re-run** — and reports every node whose
standing verdict is wrong or whose ratio was quoted without its saturation.

REUSE, not a rewrite. Where per-window tensors survive, the RE-EMISSION is
``coprimary/fix_ood_verdict.py`` (2026-07-26-v4-30k-gate), which already did
exactly this for the v4 30 k case; ``--reemit`` shells out to it rather than
duplicating it. This file only ADJUDICATES and REPORTS.

    python3 sweep_ood_verdicts.py --repo <repo> [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# The canonical rule lives in taniteval; this script is a reader, not a copy.
def _add_taniteval(repo: Path):
    for p in (repo / "taniteval", repo / "stack", repo / "stack" / "scripts"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


OOD_KEYS = ("ood_peak_ratio", "ood_mean_ratio",
            "EXTRAPOLATION_frac_steps_lat_over_3m",
            "EXTRAPOLATION_frac_steps_yaw_over_12deg",
            "EXTRAPOLATION_frac_windows_any_step_out_of_envelope",
            "EXTRAPOLATION_VERDICT")
# Below this horizon the loop has barely deviated and the envelope genuinely
# holds; the wrong verdicts all live at long horizon. Reported either way — the
# split is a READING aid, never a filter that hides a hit.
LONG_HORIZON_K = 100


def _iter_ood_nodes(obj, path=""):
    """Every dict in the document that carries an OOD reading, with its path."""
    if isinstance(obj, dict):
        if any(k in obj for k in OOD_KEYS):
            yield path or "$", obj
        for k, v in obj.items():
            yield from _iter_ood_nodes(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _iter_ood_nodes(v, f"{path}[{i}]")


def _horizon_of(node, path):
    if isinstance(node.get("horizon_K"), (int, float)):
        return int(node["horizon_K"])
    for part in path.replace("[", ".").replace("]", "").split("."):
        if part.isdigit():
            return int(part)
    return None


def _fractions(node):
    fw = node.get("EXTRAPOLATION_frac_windows_any_step_out_of_envelope")
    fs = [node.get(k) for k in ("EXTRAPOLATION_frac_steps_lat_over_3m",
                                "EXTRAPOLATION_frac_steps_yaw_over_12deg",
                                "EXTRAPOLATION_frac_steps_any")]
    fs = [float(v) for v in fs if isinstance(v, (int, float))]
    return (max(fs) if fs else None), (float(fw) if isinstance(fw, (int, float))
                                       else None)


def sweep(repo: Path, tracked_only=True) -> dict:
    from taniteval import ood as _ood                                # noqa: E402

    if tracked_only:
        files = subprocess.run(["git", "ls-files", "-z", "*.json"], cwd=str(repo),
                               capture_output=True, text=True).stdout.split("\0")
        files = [repo / f for f in files if f.strip()]
    else:
        files = list(repo.rglob("*.json"))

    rows, scanned = [], 0
    for f in files:
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        hits = list(_iter_ood_nodes(doc))
        if not hits:
            continue
        scanned += 1
        for path, node in hits:
            fs, fw = _fractions(node)
            K = _horizon_of(node, path)
            emitted = node.get("EXTRAPOLATION_VERDICT")
            if fs is None and fw is None:
                rows.append({
                    "file": str(f.relative_to(repo)).replace("\\", "/"),
                    "node": path, "horizon_K": K,
                    "emitted_verdict": emitted,
                    "peak_ratio": (node.get("ood_peak_ratio") or {}).get("mean")
                    if isinstance(node.get("ood_peak_ratio"), dict) else None,
                    "frac_steps_out": None, "frac_windows_out": None,
                    "status": "UNKNOWN_ENVELOPE",
                    "why": ("carries an OOD ratio but NO EXTRAPOLATION_* "
                            "fraction, so the envelope clause cannot be "
                            "evaluated — the ratio is uncorroborated and may "
                            "not be quoted as in-distribution")})
                continue
            fixed = _ood.readjudicate(node, _path=f"{f.name}:{path}")
            # Compare CLASSES, never strings: a re-wording is not a retraction
            # and a retraction is not a re-wording.
            wrong = bool(fixed["_class_changed"])
            lower_bound_unstamped = bool(
                fixed["ratio_is_lower_bound"] and not node.get("ratio_is_lower_bound"))
            if wrong:
                status = "WRONG_VERDICT"
            elif emitted is None and fixed["ratio_is_lower_bound"]:
                status = "RATIO_QUOTED_AS_MEASUREMENT"
            elif lower_bound_unstamped:
                status = "SATURATION_UNDECLARED"
            else:
                status = "OK"
            rows.append({
                "file": str(f.relative_to(repo)).replace("\\", "/"),
                "node": path, "horizon_K": K,
                "long_horizon": bool(K is not None and K >= LONG_HORIZON_K),
                "emitted_verdict": emitted,
                "emitted_class": fixed["_class_before"],
                "correct_verdict": fixed["EXTRAPOLATION_VERDICT"],
                "correct_class": fixed["_class_after"],
                "peak_ratio": fixed["criterion_1_ratio_over_1p5"]["peak_ratio_mean"],
                "ratio_criterion_informative":
                    fixed["criterion_1_ratio_over_1p5"]["informative"],
                "frac_steps_out": fs, "frac_windows_out": fw,
                "ratio_is_lower_bound": fixed["ratio_is_lower_bound"],
                "status": status})
    bad = [r for r in rows if r["status"] != "OK"]
    return {
        "_what": "committed artifacts re-adjudicated under E1a's FULL rule",
        "_rule": _ood.RULE,
        "_saturation": _ood.SATURATION_NOTE,
        "_method": ("pure arithmetic on the EXTRAPOLATION_* fractions the "
                    "emitters already wrote — no tensors, no GPU, no re-run. "
                    "Re-emission (where per-window tensors survive) is "
                    "coprimary/fix_ood_verdict.py, reused not rewritten."),
        "n_files_with_ood": scanned,
        "n_nodes": len(rows),
        "n_wrong": sum(1 for r in rows if r["status"] == "WRONG_VERDICT"),
        "n_saturation_undeclared": sum(
            1 for r in rows if r["status"] == "SATURATION_UNDECLARED"),
        "n_unknown_envelope": sum(
            1 for r in rows if r["status"] == "UNKNOWN_ENVELOPE"),
        "affected_files": sorted({r["file"] for r in bad}),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[5]))
    ap.add_argument("--json", default=None)
    ap.add_argument("--all-files", action="store_true",
                    help="include untracked JSON too (default: git-tracked only)")
    a = ap.parse_args()
    repo = Path(a.repo)
    _add_taniteval(repo)
    res = sweep(repo, tracked_only=not a.all_files)
    bad = [r for r in res["rows"] if r["status"] != "OK"]
    print(f"[ood-sweep] {res['n_nodes']} OOD nodes in {res['n_files_with_ood']} "
          f"committed artifacts | WRONG {res['n_wrong']} · saturation "
          f"undeclared {res['n_saturation_undeclared']} · envelope unknown "
          f"{res['n_unknown_envelope']}")
    for r in sorted(bad, key=lambda r: (r["file"], str(r["node"]))):
        print(f"  [{r['status']}] {r['file']} :: {r['node']} K={r['horizon_K']} "
              f"ratio={r.get('peak_ratio')} stepsOut={r.get('frac_steps_out')} "
              f"winOut={r.get('frac_windows_out')}")
        if r.get("emitted_verdict"):
            print(f"      emitted : {r['emitted_verdict']}")
        if r.get("correct_verdict"):
            print(f"      correct : {r['correct_verdict']}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"-> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
