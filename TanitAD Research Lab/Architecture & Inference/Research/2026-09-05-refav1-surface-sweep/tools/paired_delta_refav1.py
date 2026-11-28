#!/usr/bin/env python3
"""H-REFAV1-SURFACE-1 phase 1 — the DECISION-GRADE paired delta across N refav1 dumps.

The banked `D-REFAV1-CCOS-EVAL` verdict rests on PER-ARM intervals plus a
`paired_decision_grade` block that carries only **two** metrics (`ade_m`,
`speed_mae_mps`). The binding four-families rule needs the paired delta on EVERY
family separately, so this tool produces it.

What it does that the predecessor's `paired_dumps_refav1.py` does not:

* takes **N** dumps at once (so every pair is drawn from ONE set of components,
  never from two runs that could have diverged);
* pairs against **`ha` as well as** `ha0` / `ha0_ext` — the brief's floor set;
* carries the **LONGITUDINAL distance-keeping** metrics (headway / time-gap /
  min-TTC) per window from the B1 lead block, so LONGITUDINAL is not reported as
  speed-only (binding rule, clause 3: a missing metric is a WORK ITEM);
* emits **STRATEGIC as UNAVAILABLE with its reason and n**, never omitted
  (binding rule, clause 5);
* asserts the shared floors (`ha`, `ha0`, `ha0_ext`, `ol`) are **bit-identical**
  across the dumps — they must be, since they do not depend on the cost metric,
  and a difference would mean the dumps are not comparable;
* prints the **known-value control** (an arm against ITSELF) which must read
  exactly 0.0000 with a zero-width interval on every metric.

⛔ Nothing is pooled into a composite. Estimator: `taniteval.ci.
paired_episode_cluster_bootstrap` over the episode clusters, n_boot 2000.

Per-window components come from `refav1_arm._components` (four_families' own
geometry + the canonical trajectory labeller) — never re-derived here.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np

#: STRATEGIC has no instrument on this surface, and saying so is mandatory.
STRATEGIC_UNAVAILABLE = {
    "status": "UNAVAILABLE",
    "n": 0,
    "reason": (
        "no route/goal channel exists in this eval surface. The refav1 open-loop grid is "
        "evaluated with `--no-navshuf` on the v7.2 EVAL labels and the planner's own strategic "
        "input is the tactical head's imagined goal token, which is MODEL OUTPUT, not a route "
        "label: scoring the plan against it would score the model against itself. "
        "PhysicalAI-AV ships no map, lane graph, junction annotation or route signal "
        "(CLAUDE.md, the 6-of-36 read-set table), so no external strategic reference exists "
        "either. This is a WORK ITEM (the strategic reference must come from AlpaSim's map.xodr "
        "or an external corpus), NOT a pass and NOT an omission."
    ),
}


def _load_arm(tools_dir: str):
    spec = importlib.util.spec_from_file_location(
        "refav1_arm_paired_delta", os.path.join(tools_dir, "refav1_arm.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_dump(d: str):
    files = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    if not files:
        raise SystemExit(f"REFUSED: no ep*.npz in {d}")
    E = []
    for f in files:
        with np.load(f) as z:
            e = {k: z[k] for k in z.files}
        e["_eid"] = os.path.basename(f)[:-4]
        E.append(e)
    mp = os.path.join(d, "manifest.json")
    man = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else {}
    return files, E, man


def cat(E, k):
    return np.concatenate([e[k] for e in E])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="append", required=True, metavar="NAME=PATH",
                    help="repeatable; the FIRST one supplies the shared floors")
    ap.add_argument("--pair", action="append", default=[], metavar="B-A",
                    help="repeatable, 'B_minus_A' arm names; floors are added automatically")
    ap.add_argument("--arm", default="cl")
    ap.add_argument("--lead-block", default=None)
    ap.add_argument("--stack", required=True)
    ap.add_argument("--taniteval", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    a = ap.parse_args(argv)

    sys.path.insert(0, a.stack)
    sys.path.insert(0, a.taniteval)
    from taniteval import ci as CI
    ra = _load_arm(os.path.join(a.taniteval, "tools"))

    names, dumps = [], {}
    for spec in a.dump:
        nm, _, path = spec.partition("=")
        names.append(nm)
        dumps[nm] = load_dump(path)

    # ---- the windows must be THE SAME, asserted, never assumed ---------------
    ref = names[0]
    fr, Er, mr = dumps[ref]
    ws, v0, G = cat(Er, "ws"), cat(Er, "v0"), cat(Er, "g")
    cix = cat(Er, "clip_index")
    for nm in names[1:]:
        _, E, _ = dumps[nm]
        for k, v in (("ws", ws), ("v0", v0), ("g", G), ("clip_index", cix)):
            o = cat(E, k)
            if o.shape != v.shape or not np.array_equal(o, v):
                raise SystemExit(f"REFUSED: {nm} is not on {ref}'s windows ({k} differs)")
    eid = np.concatenate([[e["_eid"]] * len(e["ws"]) for e in Er])
    dt = float(mr.get("grid", {}).get("dt_s", 0.2))
    k_dump = int(mr.get("grid", {}).get("horizon_k", G.shape[1]))

    # ---- the floors are shared and must be bit-identical across the dumps ----
    floor_ctl = {}
    for fl in ("ha", "ha0", "ha0_ext", "ol"):
        if not all(fl in e for e in Er):
            continue
        base = cat(Er, fl)
        worst = 0.0
        for nm in names[1:]:
            _, E, _ = dumps[nm]
            if all(fl in e for e in E):
                worst = max(worst, float(np.abs(cat(E, fl) - base).max()))
        floor_ctl[fl] = {"max_abs_diff_across_dumps": worst, "bit_identical": worst == 0.0}

    arms = {f"{nm}.{a.arm}": cat(dumps[nm][1], a.arm) for nm in names}
    for fl in ("ha", "ha0", "ha0_ext", "ol"):
        if all(fl in e for e in Er):
            arms[fl] = cat(Er, fl)

    comps = {k: ra._components(v, G, dt) for k, v in arms.items()}
    fam_of = dict(ra._FAMILY_OF)

    # ---- LONGITUDINAL distance-keeping, per window, from the B1 lead block ---
    lead_info = {"status": "UNAVAILABLE", "reason": "no --lead-block passed", "n": 0}
    if a.lead_block:
        import torch
        from taniteval import four_families as ff
        lead, lead_info = ra.attach_lead_block(fr, mr, a.lead_block, k=k_dump, dt=dt)
        if lead is not None:
            for k, v in arms.items():
                dk = ff._distance_keeping(torch.as_tensor(v).float(), dt, lead)
                pw = dk.get("_per_window") or {}
                for mk, arr in (("LON_dk_headway_min_m", "headway_min_m"),
                                ("LON_dk_time_gap_min_s", "time_gap_min_s"),
                                ("LON_dk_min_ttc_s", "min_ttc_s")):
                    if arr in pw:
                        comps[k][mk] = np.asarray(pw[arr], dtype=np.float64)
                        fam_of[mk] = "longitudinal"

    # ---- the pairs ----------------------------------------------------------
    pairs = []
    for p in a.pair:
        b, _, x = p.partition("-")
        pairs.append((f"{x}.{a.arm}" if x in names else x, f"{b}.{a.arm}" if b in names else b))
    for nm in names:  # every arm against every shared floor
        for fl in ("ha", "ha0", "ha0_ext"):
            if fl in arms:
                pairs.append((fl, f"{nm}.{a.arm}"))
    pairs.append((f"{ref}.{a.arm}", f"{ref}.{a.arm}"))  # KNOWN-VALUE CONTROL
    seen, ordered = set(), []
    for x, y in pairs:
        if (x, y) not in seen:
            seen.add((x, y))
            ordered.append((x, y))

    out = {
        "tool": "paired_delta_refav1.py",
        "question": ("does the D-REFAV1-CCOS-ARMS refutation survive the DECISION-GRADE "
                     "paired episode-cluster bootstrap, on every family separately?"),
        "dumps": {nm: {"path": a.dump[i].partition("=")[2],
                       "cost": dumps[nm][2].get("cost"),
                       "arm": dumps[nm][2].get("arm")} for i, nm in enumerate(names)},
        "arm": a.arm,
        "n_windows": int(len(eid)),
        "n_episode_clusters": int(len(set(eid.tolist()))),
        "same_windows_asserted": "ws, v0, g, clip_index bit-exact across every dump",
        "floors_identical_across_dumps": floor_ctl,
        "tier": {"cl": "T1 (self-action open loop)", "ha": "T1", "ha0": "T1",
                 "ha0_ext": "T1", "ol": "T0 (world-model diagnostic, NOT driving)"},
        "estimator": "taniteval.ci.paired_episode_cluster_bootstrap",
        "n_boot": a.n_boot, "seed": a.seed,
        "lead_block": lead_info,
        "families_reported": ["ADE", "longitudinal", "lateral", "tactical", "strategic"],
        "STRATEGIC": STRATEGIC_UNAVAILABLE,
        "pooling": "NONE — every family is reported separately; no composite exists",
        "means": {}, "paired": {},
    }
    for k, c in comps.items():
        out["means"][k] = {mk: float(np.nanmean(v)) for mk, v in c.items()}

    for x, y in ordered:
        d = {}
        for mk, fam in fam_of.items():
            xv, yv = comps[x][mk], comps[y][mk]
            keep = np.isfinite(xv) & np.isfinite(yv)
            if keep.sum() == 0:
                d[mk] = {"family": fam, "status": "NOT-APPLICABLE", "n_windows": 0,
                         "reason": "no window has a finite value for both arms"}
                continue
            e = [ee for ee, kp in zip(eid, keep) if kp]
            r = CI.paired_episode_cluster_bootstrap(yv[keep], xv[keep], e,
                                                    n_boot=a.n_boot, seed=a.seed)
            r["family"] = fam
            r["n_dropped_nonfinite"] = int((~keep).sum())
            d[mk] = r
        out["paired"][f"{y} - {x}"] = d

    ctl = out["paired"][f"{ref}.{a.arm} - {ref}.{a.arm}"]
    bad = [mk for mk, r in ctl.items()
           if r.get("status") != "NOT-APPLICABLE"
           and not (r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0)]
    out["known_value_control"] = {
        "pair": f"{ref}.{a.arm} - {ref}.{a.arm}",
        "expected": "exactly 0.0000 [0, 0] on every metric",
        "PASS": not bad, "metrics_failing": bad,
    }

    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=float)

    order = ["ADE", "longitudinal", "lateral", "tactical"]
    mk_by_fam = {f: [mk for mk, fm in fam_of.items() if fm == f] for f in order}
    L = [f"### paired episode-cluster bootstrap, n = {out['n_windows']} windows / "
         f"{out['n_episode_clusters']} clusters, n_boot {a.n_boot}",
         f"tier: `cl` T1 · floors T1 · `ol` T0 · known-value control "
         f"{'PASS' if out['known_value_control']['PASS'] else 'FAIL'}", ""]
    for fam in order:
        L += [f"#### {fam}", "| pair | " + " | ".join(mk_by_fam[fam]) + " |",
              "|---|" + "---|" * len(mk_by_fam[fam])]
        for pr, d in out["paired"].items():
            cells = []
            for mk in mk_by_fam[fam]:
                r = d.get(mk)
                if not r or r.get("status") == "NOT-APPLICABLE":
                    cells.append("—")
                    continue
                m = "**" if r["separated"] else ""
                cells.append(f"{m}{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]{m}"
                             + (f" (n={r['n_windows']})" if r["n_windows"] != out["n_windows"] else ""))
            L.append(f"| {pr} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["#### strategic", f"**UNAVAILABLE (n = 0).** {STRATEGIC_UNAVAILABLE['reason']}", ""]
    md = "\n".join(L)
    print(md)
    if a.md:
        open(a.md, "w", encoding="utf-8").write(md + "\n")


if __name__ == "__main__":
    main()
