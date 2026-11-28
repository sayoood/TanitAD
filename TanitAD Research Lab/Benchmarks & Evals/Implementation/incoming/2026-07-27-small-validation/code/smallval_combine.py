#!/usr/bin/env python3
"""Score the small-validation arms from their per-window ``.npz`` — CPU only, no GPU.

⭐ THE PANEL GATE, verbatim in rule from
``…/2026-07-27-pseudosim-arm-panel/scripts/panel_combine.py``: a weighted component
enters the composite **only if it is admissible for EVERY arm in the panel**.
⚠️ This is not a detail. Under the shipped PER-ARM gate the arms carry *different
weight sets*, so a paired delta mixes a metric change with a model change — MEASURED
on the published panel, ``refc_base − v4_oracle`` reads **−0.1269** per-arm against
**−0.0217** panel-gated, a **5.2×** inflation with a verdict flip.

⭐ BOTH PROGRESS TERMS are emitted. ``twosided_v2`` is the registered PRIMARY (it is
the shipped default and what the trainer's own mid-run gate stops on); ``clamp_v1``
is emitted beside it **only** so the numbers are comparable to the published 20-arm
panel, whose every value is a ``clamp_v1`` value. ⛔ They are DIFFERENT METRICS and
are never compared to each other.

Estimator: :func:`taniteval.ci.paired_episode_cluster_bootstrap`, B = 2000, unit =
episode. ⛔ ``overlapping_holdout_se`` is never called.

usage:
  python3 smallval_combine.py --in-dir <dir of pw_*.npz> --out <result.json> \\
      [--blind-suffix _blind] [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ⛔ THE SHADOWING GUARD — see smallval_pseudosim.pin_stack.__doc__. ~15 taniteval
# submodules hardcode sys.path.insert(0, "/root/TanitAD/stack"), which on pod2 is a
# 12 MB PRE-v5 tree. TANITEVAL_STACK_OVERRIDE (shipped in taniteval/__init__.py)
# pins `tanitad` from the right tree FIRST, so the sys.modules cache wins.
_STACK = os.environ.get("SMALLVAL_STACK", "/workspace/TanitAD/stack")
os.environ.setdefault("TANITEVAL_STACK_OVERRIDE", _STACK)
for _p in (_STACK, str(Path(_STACK).parent / "taniteval")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
from taniteval import ci as _ci                                   # noqa: E402
from taniteval import pseudosim as PS                             # noqa: E402

if os.path.isdir(_STACK):
    import tanitad                                                # noqa: E402
    if not str(Path(tanitad.__file__).resolve()).startswith(str(Path(_STACK).resolve())):
        raise SystemExit(f"[sv] ⛔ STACK SHADOWING: tanitad resolved to "
                         f"{tanitad.__file__}, not under {_STACK}.")

TERMS = ("twosided_v2", "clamp_v1")
PRIMARY_TERM = "twosided_v2"


def load_pw(path):
    z = np.load(path, allow_pickle=False)
    return {k: torch.as_tensor(z[k]) for k in
            ("traj", "ref_path", "ref_yaw", "v0", "pt_dlat", "pt_dyaw",
             "pt_dlon", "anchor", "ep_i")} | {"eid": [str(x) for x in z["eid"]]}


def key_of(pw):
    """Row identity — two arms may only be paired if their rows ARE the same rows."""
    return np.stack([pw["ep_i"].numpy(), pw["anchor"].numpy(),
                     np.round(pw["pt_dlat"].numpy(), 6) * 1e3,
                     np.round(pw["pt_dyaw"].numpy(), 6) * 1e3,
                     pw["pt_dlon"].numpy()], axis=1).astype(np.int64)


def paired(a, b, eid, n_boot):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2 or len(set(np.asarray(eid)[m])) < 2:
        return {"REFUSED": "fewer than 2 finite rows or fewer than 2 episodes"}
    out = _ci.paired_episode_cluster_bootstrap(a[m], b[m],
                                               list(np.asarray(eid)[m]),
                                               n_boot=n_boot)
    out["n_rows_paired"] = int(m.sum())
    out["n_episodes_paired"] = int(len(set(np.asarray(eid)[m])))
    out["coverage_frac"] = round(float(m.mean()), 6)   # the selected_frac discipline
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    in_dir = Path(a.in_dir)
    arms = {f.name[3:-4]: load_pw(f) for f in sorted(in_dir.glob("pw_*.npz"))}
    metas = {n: json.loads(p.read_text())
             for n in arms
             if (p := in_dir / f"meta_{n}.json").exists()}
    assert arms, f"no pw_*.npz in {in_dir}"
    print(f"[sv] {len(arms)} arms: {sorted(arms)}", flush=True)

    # ---- 1. ROW IDENTITY. A paired bootstrap on non-identical rows is void. ---
    ref = sorted(arms)[0]
    kref, row_id, usable = key_of(arms[ref]), {}, []
    for n, pw in arms.items():
        k = key_of(pw)
        same = (k.shape == kref.shape) and bool((k == kref).all())
        row_id[n] = {"rows": int(k.shape[0]), "identical_to_reference": same}
        (usable if same else []).append(n)
    refused = [n for n in arms if n not in usable]
    print(f"[sv] row identity vs {ref}: usable={sorted(usable)} refused={refused}",
          flush=True)

    res = {
        "reference_arm": ref, "row_identity": row_id,
        "arms_refused_on_row_identity": refused,
        "estimator": (f"paired episode-cluster bootstrap "
                      f"(taniteval.ci, B={a.n_boot}, unit = episode). "
                      f"overlapping_holdout_se is NEVER used."),
        "gate": ("PANEL GATE: a weighted component enters the composite only if "
                 "discriminative_range admits it for EVERY arm. Stated before "
                 "application. The per-arm gate would give the arms different "
                 "weight sets and a paired delta would mix a metric change with "
                 "a model change."),
        "primary_term": PRIMARY_TERM,
        "term_warning": ("twosided_v2 and clamp_v1 are DIFFERENT METRICS. Every "
                         "PSS value published before 2026-07-28 is clamp_v1. "
                         "Never compare a value under one term to a value under "
                         "the other."),
        "collision_and_ttc": {"emitted": False,
                              "reason": PS.COLLISION_UNAVAILABLE_REASON},
        "arm_meta": metas, "terms": {},
    }
    out_p = Path(a.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    def bank():
        out_p.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")

    for term in TERMS:
        by_arm = {n: PS.score_windows(pw, progress_term=term)
                  for n, pw in arms.items()}
        per_arm_ranges = {n: PS.discriminative_range(by_arm[n], by_arm=by_arm)
                          for n in arms}
        panel_ok, why = {}, {}
        for comp in PS.COMPONENT_WEIGHTS:
            bad = [n for n in arms
                   if not per_arm_ranges[n].get(comp, {}).get("admissible")]
            panel_ok[comp] = not bad
            why[comp] = ("admissible for every arm" if not bad else
                         "INADMISSIBLE for " + ", ".join(sorted(bad)) +
                         " -> dropped from EVERY arm")
        node = {"PANEL_GATE": {"admitted": [k for k, v in panel_ok.items() if v],
                               "dropped": {k: why[k] for k, v in panel_ok.items()
                                           if not v},
                               "per_arm_admissibility": {
                                   n: {c: per_arm_ranges[n].get(c, {}).get("admissible")
                                       for c in PS.COMPONENT_WEIGHTS}
                                   for n in sorted(arms)}},
                "arms": {}, "paired": {}, "decomposition": {}}
        print(f"[sv][{term}] PANEL GATE admitted={node['PANEL_GATE']['admitted']} "
              f"dropped={sorted(node['PANEL_GATE']['dropped'])}", flush=True)

        vals = {}
        for n in arms:
            pr = dict(per_arm_ranges[n])
            for c, ok in panel_ok.items():
                if not ok and c in pr:
                    pr[c] = dict(pr[c], admissible=False,
                                 reason="dropped by the PANEL GATE: " + why[c])
            try:
                comp = PS.composite(by_arm[n], pr, progress_term=term)
                v = np.asarray(comp.pop("value"), float)
                vals[n] = v
                fin = np.isfinite(v)
                comp["ci"] = PS._boot(v, arms[n]["eid"], a.n_boot, 0)
                comp["n_rows_total"] = int(v.size)
                comp["n_rows_finite"] = int(fin.sum())
                comp["coverage_frac"] = round(float(fin.mean()), 6)
                comp["n_episodes"] = int(len(set(arms[n]["eid"])))
                comp["component_defined_frac"] = {
                    c: round(float(np.isfinite(np.asarray(by_arm[n][c], float)).mean()), 6)
                    for c in ("ego_progress", "recovery", "comfort")}
                node["arms"][n] = comp
                ci = comp["ci"] or {}
                print(f"[sv][{term}] {n:24s} PSS={ci.get('mean')} "
                      f"[{ci.get('lo')},{ci.get('hi')}] "
                      f"cov={comp['coverage_frac']}", flush=True)
            except PS.VacuousMetric as exc:
                node["arms"][n] = {"REFUSED_TO_EMIT": str(exc)}
                vals[n] = None
                print(f"[sv][{term}] {n:24s} REFUSED_TO_EMIT", flush=True)

        # ---- pairwise, only over row-identical arms ------------------------- #
        for x, y in itertools.combinations(sorted(usable), 2):
            if vals.get(x) is None or vals.get(y) is None:
                continue
            node["paired"][f"{x}__minus__{y}"] = paired(vals[x], vals[y],
                                                        arms[x]["eid"], a.n_boot)
        # ---- the pre-registered DECOMPOSITION, per component ---------------- #
        # ego_progress = the LONGITUDINAL (along-track) axis;
        # recovery     = the LATERAL (cross-track) axis.
        for comp_name in ("ego_progress", "recovery", "comfort"):
            blk = {}
            for x, y in itertools.combinations(sorted(usable), 2):
                blk[f"{x}__minus__{y}"] = paired(by_arm[x][comp_name],
                                                 by_arm[y][comp_name],
                                                 arms[x]["eid"], a.n_boot)
            node["decomposition"][comp_name] = blk
        # raw metres, for a reader who wants the physical axis not the score
        for raw, axis in (("along_track_end_m", "LONGITUDINAL"),
                          ("cross_track_end_m", "LATERAL")):
            blk = {"_axis": axis}
            for x, y in itertools.combinations(sorted(usable), 2):
                blk[f"{x}__minus__{y}"] = paired(by_arm[x][raw], by_arm[y][raw],
                                                 arms[x]["eid"], a.n_boot)
            node["decomposition"][raw] = blk

        res["terms"][term] = node
        bank()

    bank()
    print(f"[sv] wrote {out_p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
