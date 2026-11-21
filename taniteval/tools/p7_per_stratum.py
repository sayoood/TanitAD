#!/usr/bin/env python3
"""CLI: PER-STRATUM P7 calibration on a banked fan dump + an `obstacle.offline`
lead block. This is the instrument the F-9/T3 gate row needs.

    python taniteval/tools/p7_per_stratum.py \
        --fan   taniteval/results/fan_refc-xl-30k.pt \
        --lead  "<pkg>/raw/val40_lead_block.npz" \
        --arm   refc-xl-30k \
        --out   p7_per_stratum.json

⛔ THE ALIGNMENT GUARD RUNS FIRST, BEFORE ANY ARITHMETIC. The lead block and the
fan dump are joined POSITIONALLY (both are `rollout.collect`'s window order), so a
mismatched pair would produce a complete, plausible, wrong report. The tool exits
non-zero on any of: different n, different ``eid`` sequence, or a lead block
missing a required key. *(Same family as the analysis-time-import trap: fail in
seconds, not after the expensive part.)*

⚠️ ``--lead`` accepts BOTH containers. Two builders emit lead blocks -- `.pt`
(`taniteval/tools/build_lead_block.py`) and `.npz`
(`…/2026-08-04-distance-keeping-arms/code/build_val40_lead_block.py`) -- and the
only block banked in this repo is the `.npz`. A loader that assumed `.pt` is what
kept the LONGITUDINAL family UNAVAILABLE for twelve days with the data present.

Spread measures (both computed, both reported -- they are different probes of the
same fan, and P7's own artifact reports both):
  * ``selector_entropy``        -- entropy of softmax(logits) over candidates
  * ``endpoint_dispersion_m``   -- prob-weighted std of the candidate endpoints, m
Realised error: ADE of the SELECTED candidate against the ground-truth waypoints.
Both follow ``tools_p7_calibration.py`` exactly.

Tier: **T0**. A banked fan dump comes from `rollout.collect`, which is fed the
expert's true future actions. Nothing here is a driving claim.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PKG_PARENT = _HERE.parent                       # <repo>/taniteval
_REPO = _PKG_PARENT.parent
for _p in (str(_PKG_PARENT), str(_REPO / "stack"), str(_REPO / "stack" / "scripts")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from taniteval.p7_strata import (  # noqa: E402
    DEFAULT_GAP_EDGES_M, MIN_N_EPISODES, MIN_N_WINDOWS, STRATIFIER_KIND_LABEL,
    arm_controls, lead_state_strata, p7_per_stratum, p7_strata_report)

REQUIRED_FAN_KEYS = ("fan", "logits", "sel", "gt", "eid")
REQUIRED_LEAD_KEYS = ("state", "gap0_m", "eid")

STRATIFIER_DECL = {
    "name": "obstacle.offline lead state + lead proximity band",
    "kind": STRATIFIER_KIND_LABEL,
    "derived_from": ("obstacle.offline 3D agent cuboids (the dataset's own "
                     "annotation of other traffic), via taniteval.lead_source's "
                     "causal in-corridor lead rule"),
    "why_admissible": (
        "It is an EXTERNAL annotation of other agents. (a) It is not computed "
        "from ego dynamics, so it does not cut on the source the situation "
        "labels are derived from (stack/tanitad/data/situations.py) -- the "
        "disjointness the 2026-08-03 ruling protects. (b) It is not computed "
        "from any model output, so the arm being graded does not choose the "
        "strata that grade it; O4's own docstring admits the obstacle join as "
        "'frozen-probe/eval-strata material'. (c) The stratifying quantities are "
        "lead PRESENCE and lead GAP -- facts about other traffic. Ego speed "
        "travels in the same block and is deliberately NOT used to stratify; it "
        "is used only as the trivial-proxy control."),
}


# --------------------------------------------------------------------------- #
def load_lead_block(path: str | Path) -> dict:
    """Load a lead block from `.npz` OR `.pt`, always returning a plain dict."""
    path = Path(path)
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=True)
        blk = {k: z[k] for k in z.files}
    else:
        import torch
        blk = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(blk, dict):
            raise SystemExit(f"[p7-strata] {path} did not contain a dict")
    missing = [k for k in REQUIRED_LEAD_KEYS if k not in blk]
    if missing:
        raise SystemExit(
            f"[p7-strata] lead block {path} is missing {missing}; it carries "
            f"{sorted(blk)}. Build it with "
            f"`…/2026-08-04-distance-keeping-arms/code/build_val40_lead_block.py` "
            f"or `taniteval/tools/build_lead_block.py`.")
    return blk


def load_fan(path: str | Path) -> dict:
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    missing = [k for k in REQUIRED_FAN_KEYS if k not in d]
    if missing:
        raise SystemExit(f"[p7-strata] fan dump {path} is missing {missing}; it "
                         f"carries {sorted(d)}")
    return d


def _canon_eid(e) -> np.ndarray:
    """`eid` arrives as ints (fan dumps) or 'ep_00000' strings (lead blocks)."""
    out = []
    for x in np.asarray(e).ravel().tolist():
        s = str(x)
        out.append(f"ep_{int(s):05d}" if s.lstrip("-").isdigit()
                   else (s if s.startswith("ep_") else s))
    return np.asarray(out)


def assert_aligned(fan: dict, lead: dict) -> dict:
    """⛔ Fail loud BEFORE any arithmetic. A positional join of two artifacts that
    do not describe the same windows yields a complete and wrong report."""
    n_f, n_l = int(np.asarray(fan["sel"]).size), int(np.asarray(lead["state"]).size)
    if n_f != n_l:
        raise SystemExit(f"[p7-strata] REFUSING: fan dump has {n_f} windows, lead "
                         f"block has {n_l}. These are not the same eval.")
    ef, el = _canon_eid(fan["eid"]), _canon_eid(lead["eid"])
    if not np.array_equal(ef, el):
        n_bad = int((ef != el).sum())
        raise SystemExit(f"[p7-strata] REFUSING: eid sequences differ in {n_bad} "
                         f"of {n_f} positions -- the positional join is invalid.")
    return {"n_windows": n_f, "n_episodes": int(np.unique(ef).size),
            "join": "positional (rollout.collect window order), eid-verified"}


def spreads_and_error(fan: dict) -> dict:
    """Selector entropy + prob-weighted endpoint dispersion + realised ADE of the
    SELECTED candidate -- the arithmetic of ``tools_p7_calibration.py``."""
    f = np.asarray(fan["fan"], dtype=np.float64)            # [N, C, H, 2]
    sc = np.asarray(fan["logits"], dtype=np.float64)        # [N, C]
    gt = np.asarray(fan["gt"], dtype=np.float64)            # [N, H, 2]
    sel = np.asarray(fan["sel"], dtype=np.int64).ravel()    # [N]
    n = f.shape[0]
    p = np.exp(sc - sc.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(1)
    ep = f[:, :, -1, :]
    mu = (p[..., None] * ep).sum(1, keepdims=True)
    disp = np.sqrt((p * ((ep - mu) ** 2).sum(-1)).sum(1))
    err = np.linalg.norm(f[np.arange(n), sel] - gt, axis=-1).mean(-1)
    return {"selector_entropy": ent, "endpoint_dispersion_m": disp, "err_m": err}


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fan", required=True, action="append",
                    help="banked fan dump (.pt); repeatable, one per arm")
    ap.add_argument("--arm", action="append", default=None,
                    help="arm name per --fan (default: file stem)")
    ap.add_argument("--lead", required=True,
                    help="lead block (.npz or .pt) on the SAME windows")
    ap.add_argument("--gap-edges", type=float, nargs=2,
                    default=list(DEFAULT_GAP_EDGES_M), metavar=("LO", "HI"))
    ap.add_argument("--min-n", type=int, default=MIN_N_WINDOWS)
    ap.add_argument("--min-episodes", type=int, default=MIN_N_EPISODES)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    lead = load_lead_block(a.lead)
    arms_out: dict[str, dict] = {}
    prov = {"lead_block": str(a.lead), "gap_edges_m": list(a.gap_edges),
            "tier": "T0",
            "tier_note": ("banked fan dumps come from rollout.collect, which is "
                          "fed the expert's true future actions -- a WM / "
                          "instrument diagnostic, never a driving claim"),
            "estimator": "episode_cluster_bootstrap (taniteval.ci._draws)",
            "n_boot": int(a.n_boot), "seed": int(a.seed),
            "min_n_windows": int(a.min_n), "min_n_episodes": int(a.min_episodes),
            "evidence_class": "MEASURED (ours; artifact = this JSON)"}

    for i, fp in enumerate(a.fan):
        name = (a.arm[i] if a.arm and i < len(a.arm) else Path(fp).stem)
        fan = load_fan(fp)
        align = assert_aligned(fan, lead)
        lab, spec = lead_state_strata(lead["state"], lead["gap0_m"],
                                      edges=tuple(a.gap_edges))
        sig = spreads_and_error(fan)
        eid = _canon_eid(fan["eid"])
        speed = np.asarray(fan.get("speed", lead.get("speeds")),
                           dtype=np.float64).ravel()
        block = {"fan_dump": str(fp), "alignment": align,
                 "n_candidates": int(np.asarray(fan["fan"]).shape[1]),
                 "err_m_mean_dispersion_not_a_ci": {
                     "mean": round(float(sig["err_m"].mean()), 4),
                     "sd": round(float(sig["err_m"].std()), 4),
                     "bracket_kind": "dispersion_not_a_ci"},
                 "reads": {}, "controls": {}}
        for meas in ("selector_entropy", "endpoint_dispersion_m"):
            block["reads"][meas] = p7_per_stratum(
                sig[meas], sig["err_m"], eid, lab, stratifier=STRATIFIER_DECL,
                stratum_spec=spec, min_n=a.min_n, min_eps=a.min_episodes,
                n_boot=a.n_boot, seed=a.seed)
            block["controls"][meas] = arm_controls(
                sig[meas], sig["err_m"], eid, lab, stratifier=STRATIFIER_DECL,
                stratum_spec=spec, trivial_proxy=speed,
                trivial_proxy_name="ego speed at t0 (poses[:,3])",
                trivial_proxy_note=(
                    "a scalar the arm gets for free and does not have to model. "
                    "⛔ It is a CONTROL, never a stratifier -- stratifying on ego "
                    "state would cut on the situation labels' own source."),
                min_n=a.min_n, min_eps=a.min_episodes, n_boot=a.n_boot,
                seed=a.seed)
        arms_out[name] = block

        print(f"\n=== {name}  ({align['n_windows']} windows / "
              f"{align['n_episodes']} episodes, T0) ===", flush=True)
        for meas, rd in block["reads"].items():
            print(f"  [{meas}]  verdict {rd['verdict']}  "
                  f"(pooled rho {rd['pooled']['spearman_rho']} "
                  f"{rd['pooled']['rho_ci_cluster']} <- NOT the gate)")
            for s in rd["stratum_spec"]["stratum_order"]:
                r = rd["strata"].get(s)
                if r is None:
                    continue
                tag = {True: "interaction", False: "free-flow",
                       None: "unlabelled"}[r["interaction_rich"]]
                if r["status"] != "OK":
                    print(f"    {s:<14} n={r['n']:<4} ep={r['n_episodes']:<3} "
                          f"{tag:<12} REFUSED (min-n)")
                else:
                    print(f"    {s:<14} n={r['n']:<4} ep={r['n_episodes']:<3} "
                          f"{tag:<12} rho {r['spearman_rho']:+.4f} "
                          f"CI {r['rho_ci_cluster']} "
                          f"{'PASS' if r['gate_pass'] else 'fail'}")

    rep = p7_strata_report(arms_out, provenance=prov)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str),
                           encoding="utf-8")
    print(f"\n[p7-strata] wrote {a.out}", flush=True)
    print("P7_STRATA_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
