"""Zero-GPU preflight for the refcv4b hierarchy panel.

⛔ **WHY THIS EXISTS.** MEASURED 2026-08-11: `t1_eval.py` rolled **both arms, all 40
episodes, 6,844 windows each (~11 min/arm)** and then died in `analyze()` on a missing
import. `T1_EXIT=NO_ARMS_PRODUCED` reads like a total failure; it was a 100 %-complete
run with a broken last step. MEASURED 2026-09-05: the STRATEGIC nav-compliance family
was REFUSED on every refcv3 arm for its whole life because a manifest key held a dict
where the reader wanted a path -- and nothing failed, because the caller converted the
`TypeError` into a polite UNAVAILABLE block.

Both are the same shape: **the expensive part succeeds and the cheap part at the end
destroys or silently hollows out the result.** This runs every cheap part FIRST.

⚠️ **What it cannot check.** It does not roll the model, so it cannot tell you an arm
will produce a *good* number -- only that each arm and each readout can RUN, that the
decisions taken today are actually in force, and that the panel will not discover a
missing input after the GPU time is spent.

⛔ Every check is a POSITIVE assertion, and a check that cannot read its subject reports
**INCONCLUSIVE and is counted as a failure, never as a pass** (`CLAUDE.md`: on this mount
an empty result is indistinguishable from a failed query).

Usage::

    python taniteval/tools/panel_preflight.py --ckpt <ckpt.pt> --labels <v72.jsonl.gz> \
        [--dump-dir <an existing dump, to check the manifest shape>] [--min-free-gb 40]
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

#: The eight arms of `PREREG_REFCV4B_HIERARCHY_EVAL.md` section 3 that needed new
#: routes. The other four (`os_navzero`, `os_navshuf`, `navflip`, and FULL needing
#: none) had CLI flags already, so they are deliberately not in the enum.
NEW_ENUM = ("gstr_zero", "gstr_shuffle", "e7_off", "e9_off", "h19_off",
            "ego_zero", "sel_refined", "frames_blind")

results = []


def record(name, ok, detail):
    state = "PASS" if ok is True else ("INCONCLUSIVE" if ok is None else "FAIL")
    results.append((state, name, detail))
    print("  [%-12s] %s\n                 %s" % (state, name, detail), flush=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_arm_source():
    p = os.path.join(REPO, "taniteval", "tools", "refcv3_arm.py")
    try:
        with open(p, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


# ------------------------------------------------------------------ the checks --

def check_navcomp_labels(dump_dir, labels):
    """The defect that cost the STRATEGIC family, checked on the REAL manifest."""
    try:
        nc = load("_nc_pf", os.path.join(REPO, "taniteval", "taniteval",
                                         "nav_compliance.py"))
    except Exception as ex:                                     # noqa: BLE001
        record("nav_compliance imports", False, "%s: %s" % (type(ex).__name__, ex))
        return
    if not hasattr(nc, "resolve_labels_path"):
        record("nav_compliance labels shape", False,
               "resolve_labels_path is gone -- the dict/path fix was reverted")
        return
    if not dump_dir:
        record("nav_compliance labels shape", None,
               "no --dump-dir given; the manifest shape is UNCHECKED")
        return
    man = os.path.join(dump_dir, "manifest.json")
    try:
        with open(man, encoding="utf-8") as fh:
            corpus = (json.load(fh).get("corpus") or {})
    except Exception as ex:                                     # noqa: BLE001
        record("nav_compliance labels shape", None,
               "could not read %s: %s" % (man, type(ex).__name__))
        return
    got = nc.resolve_labels_path(labels, corpus)
    if not isinstance(got, str) or not got:
        record("nav_compliance labels shape", False,
               "resolved to %s %r -- os.path.exists would raise"
               % (type(got).__name__, got))
        return
    record("nav_compliance labels shape", True,
           "corpus.labels is %s; resolved to a str"
           % type(corpus.get("labels")).__name__)
    record("labels blob present", os.path.exists(got), got)


def check_ablation_bijection():
    """Every section-3 arm has a route, and no route exists that section 3 omits."""
    src = read_arm_source()
    if src is None:
        record("ablation bijection with the PREREG", None, "refcv3_arm.py unreadable")
        return
    keys = None
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "ABLATIONS" for t in n.targets):
            keys = tuple(k.value for k in n.value.keys if isinstance(k, ast.Constant))
    if keys is None:
        record("ablation bijection with the PREREG", None, "ABLATIONS not found")
        return
    missing = [a for a in NEW_ENUM if a not in keys]
    extra = [k for k in keys if k not in NEW_ENUM]
    record("ablation bijection with the PREREG", not missing and not extra,
           "%d enum arms; missing=%s; unregistered=%s"
           % (len(keys), missing or "none", extra or "none"))


def check_ha0ext_is_the_integrator():
    """M11 / D-MM-ADJ-1: the two derivations differ by 1.862923 m at 15 s, MORE
    than the whole margin, so this decides the verdict rather than its precision."""
    src = read_arm_source()
    if src is None:
        record("ha0_ext is the INTEGRATOR (M11)", None, "refcv3_arm.py unreadable")
        return
    #: A SUBSTRING SEARCH IS THE WRONG INSTRUMENT HERE, and this check's own
    #: first run proved it: both `echo_gate.ha0_ext` mentions in refcv3_arm.py
    #: are COMMENTS explaining why the integrator is used, and the check reported
    #: a violation. A preflight that cries wolf on prose gets ignored, which is
    #: worse than not having it. So: assert on the AST, where a comment does not
    #: exist at all.
    try:
        tree = ast.parse(src)
    except SyntaxError as ex:
        record("ha0_ext is the INTEGRATOR (M11)", None, "unparseable: %s" % ex)
        return
    calls_integrator = any(
        isinstance(n, ast.Call) and (
            (isinstance(n.func, ast.Attribute) and n.func.attr == "hold_ext_controls")
            or (isinstance(n.func, ast.Name) and n.func.id == "hold_ext_controls"))
        for n in ast.walk(tree))
    uses_closed_form = any(
        (isinstance(n, ast.Attribute) and n.attr == "ha0_ext")
        or (isinstance(n, ast.ImportFrom) and (n.module or "").endswith("echo_gate")
            and any(al.name == "ha0_ext" for al in n.names))
        for n in ast.walk(tree))
    record("ha0_ext is the INTEGRATOR (M11)", calls_integrator and not uses_closed_form,
           "AST: calls hold_ext_controls=%s; touches echo_gate.ha0_ext=%s "
           "(comments do not count -- they are not in the tree)"
           % (calls_integrator, uses_closed_form))


def check_defect_classifier():
    """A TypeError from our own module must be recorded as a DEFECT, not a refusal."""
    src = read_arm_source()
    if src is None:
        record("defect-vs-refusal classifier", None, "refcv3_arm.py unreadable")
        return
    ok = "DEFECT_EXCEPTIONS" in src and 'ref.setdefault("_defects", [])' in src
    record("defect-vs-refusal classifier", ok,
           "a swallowed TypeError is how the STRATEGIC family went missing; the "
           "record must be able to say 'broken' rather than 'unavailable'")


def check_h19_erratum(ckpt):
    """ERRATUM-1: on a FACTORED build `maneuver_to_anchor` is ALREADY None, so the
    prereg's registered mechanism removes nothing. The corrected arm targets the
    heads the build actually carries -- which requires the build to be factored."""
    if not ckpt or not os.path.exists(ckpt):
        record("H19-OFF targets a real head (ERRATUM-1)", None,
               "checkpoint not readable: %s" % ckpt)
        return
    try:
        import torch
        obj = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = obj.get("config", {}) if isinstance(obj, dict) else {}
    except Exception as ex:                                     # noqa: BLE001
        record("H19-OFF targets a real head (ERRATUM-1)", None,
               "%s: %s" % (type(ex).__name__, str(ex)[:120]))
        return
    fac = cfg.get("factored_maneuver")
    if fac is None and isinstance(cfg.get("core"), dict):
        fac = cfg["core"].get("factored_maneuver")
    record("H19-OFF targets a real head (ERRATUM-1)", fac is True,
           "factored_maneuver=%r -- the live prior is lat_to_anchor + lon_to_anchor"
           % (fac,))


def check_estimator():
    """The paired episode-cluster bootstrap must be importable AND present."""
    try:
        ci = load("_ci_pf", os.path.join(REPO, "taniteval", "taniteval", "ci.py"))
    except Exception as ex:                                     # noqa: BLE001
        record("paired episode-cluster bootstrap", False,
               "%s: %s" % (type(ex).__name__, ex))
        return
    have = [n for n in dir(ci) if "boot" in n.lower() and not n.startswith("_")]
    record("paired episode-cluster bootstrap", bool(have),
           "callable: %s" % (have or "NONE FOUND"))


def check_disk(min_free_gb):
    try:
        free = shutil.disk_usage(REPO).free / 1e9
    except OSError as ex:
        record("free disk for the dumps", None, str(ex))
        return
    record("free disk for the dumps", free >= min_free_gb,
           "%.1f GB free, need >= %.0f GB for 12 arms" % (free, min_free_gb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--min-free-gb", type=float, default=40.0)
    a = ap.parse_args()

    print("refcv4b hierarchy-panel preflight -- ZERO GPU\n")
    check_navcomp_labels(a.dump_dir, a.labels)
    check_ablation_bijection()
    check_ha0ext_is_the_integrator()
    check_defect_classifier()
    check_h19_erratum(a.ckpt)
    check_estimator()
    check_disk(a.min_free_gb)

    n_fail = sum(1 for s, _, _ in results if s == "FAIL")
    n_inc = sum(1 for s, _, _ in results if s == "INCONCLUSIVE")
    print("\n%d checks: %d PASS, %d FAIL, %d INCONCLUSIVE"
          % (len(results), len(results) - n_fail - n_inc, n_fail, n_inc))
    if n_fail or n_inc:
        #: INCONCLUSIVE counts as a failure. A check that could not read its
        #: subject has told you nothing, and treating it as a pass is exactly how
        #: "0 hits" from an unreadable file became an absence claim.
        print("DO NOT LAUNCH THE PANEL -- an INCONCLUSIVE check is not a pass.")
        return 1
    print("every cheap part runs; the expensive part is safe to start.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
