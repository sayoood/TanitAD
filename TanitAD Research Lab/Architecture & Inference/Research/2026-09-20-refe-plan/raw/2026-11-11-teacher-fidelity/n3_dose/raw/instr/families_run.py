#!/usr/bin/env python3
"""W-FAM: the four metric families on the EXECUTED trajectory (primary) and on the submitted reference (diagnostic).

Reads the rows `transp_score.py` wrote (the `fam` payload) and runs the programme's own instrument, REUSED, not re-implemented:
    taniteval/adapters/navsim.py :: scenes_to_win(...)            (the families6.py:84-86 pattern)
    taniteval/adapters/navsim.py :: four_families_block(win, tier="T1")  -> taniteval.four_families.all_families
with ONE addition that closes the adapter's own documented work item (`navsim.py` "lead_block ... UNAVAILABLE"): `win["lead"]`
built from the nuPlan log's tracked objects by `transp_score.lead_track`, so the distance-keeping half of LONGITUDINAL
(headway / time-gap / min-TTC) is computed where a lead exists, with n, and refused with its reason where none does.

Frames, each against the LOGGED HUMAN FUTURE (8 poses, 0.5 s, ego frame), cluster unit = log_name:
    executed_tracked      the LQR + bicycle states of the plan                      PRIMARY (MM ruling 2026-11-12; G2 guards read this)
    executed_transparent  the plan's cubic-spline 41 states sampled at 0.5 s       diagnostic (inflated by construction on a pre-distorted plan)
    reference             the plan as submitted (to_navsim 8 poses)                 diagnostic
STRATEGIC is PAUSED (PI ruling 2026-09-19, PI_DECISION_QUEUE item 25): printed as PAUSED, never as a pass, never omitted.

  python families_run.py --rows <rows.jsonl|glob ...> --arm <name> --label <name> --out <families.json> [--n-boot 2000] [--tanitad-repo <dir with taniteval/>]
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO = os.path.abspath(os.path.join(HERE, *([".."] * 6)))   # repo root: PK/eval/instruments is 6 levels below it
FRAMES = (("executed_tracked", "exec_tracked8", "PRIMARY"), ("executed_transparent", "exec_transparent8", "diagnostic"),
          ("reference", "reference8", "diagnostic"))
STRATEGIC_PAUSED = {"status": "PAUSED", "reason": "PI ruling 2026-09-19 (PI_DECISION_QUEUE item 25): the strategic layer is switched "
                    "off temporarily. NOT a pass; also unavailable in NAVSIM.", "tier": "T1"}


def load_adapter(repo: str):
    te = os.path.join(repo, "taniteval")
    for pth in (te, os.path.join(repo, "stack")):        # stack/ carries tanitad.refs.refc_tactical (the TACTICAL labeller)
        if os.path.isdir(pth) and pth not in sys.path:
            sys.path.append(pth)
    spec = importlib.util.spec_from_file_location("navsim_adapter_fam", os.path.join(te, "adapters", "navsim.py"))
    ad = importlib.util.module_from_spec(spec)
    sys.modules["navsim_adapter_fam"] = ad
    spec.loader.exec_module(ad)
    return ad


def read_rows(patterns) -> list:
    rows, seen = [], set()
    for pat in patterns:
        for fp in sorted(glob.glob(pat)):
            for ln in open(fp, encoding="utf-8"):
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if "error" in r or "shared" not in r or r["token"] in seen:
                    continue
                seen.add(r["token"])
                rows.append(r)
    return sorted(rows, key=lambda r: r["token"])


def lead_block(rows, dt_s: float = 0.5) -> dict:
    n = len(rows)
    K = len(rows[0]["shared"]["human8"])
    leads = np.full((n, K, 2), np.nan)
    lens = np.full(n, np.nan)
    state = []
    for i, r in enumerate(rows):
        ld = r["shared"]["lead"]
        state.append(ld["state"])
        if ld["state"] == "LEAD":
            leads[i] = np.asarray([[np.nan if v is None else v for v in pt] for pt in ld["leads"]], np.float64)
            lens[i] = ld["len"]
    return {"leads": leads, "lead_lens": lens, "speeds": np.asarray([r["shared"]["v0"] for r in rows], np.float64),
            "eid": [r["log"] for r in rows], "state": state, "dt_s": dt_s, "path_steps": None}


def frame_poses(rows, arm: str, key: str) -> np.ndarray:
    return np.asarray([r["arms"][arm]["fam"][key] for r in rows], np.float64)


def run(rows, arm: str, label: str, repo: str, n_boot: int, per_window: bool = False) -> dict:
    ad = load_adapter(repo)
    toks = [r["token"] for r in rows]
    logs = [r["log"] for r in rows]
    v0 = [r["shared"]["v0"] for r in rows]
    human = np.asarray([r["shared"]["human8"] for r in rows], np.float64)
    out = {"_label": label, "arm": arm, "n": len(rows), "n_logs": len(set(logs)), "tier": "T1-family (loop OPEN: the plan is a single query)",
           "estimator": "log-cluster bootstrap (taniteval four_families, eid = log_name), B=%d" % n_boot,
           "instrument": "taniteval/adapters/navsim.py scenes_to_win + four_families_block (REUSED) + transp_score.lead_track (NEW)",
           "frames": {}, "strategic": STRATEGIC_PAUSED}
    if per_window:    # fam_ci pairs by IDENTITY: array order of every per-window array below (default-off: default JSON is unchanged)
        out["tokens"], out["logs"] = list(toks), list(logs)
        out["per_window_schema"] = "taniteval.per_window/1: blocks carry `_per_window`; arrays are in `tokens` order (read_rows sorts by token)"
    lead = lead_block(rows)
    out["lead_states"] = {s: lead["state"].count(s) for s in sorted(set(lead["state"]))}
    for name, key, role in FRAMES:
        try:
            w = ad.scenes_to_win(frame_poses(rows, arm, key), human, frame="ego", origin_included=False, dt_s=0.5,
                                 scene_tokens=toks, ego_speed_mps=v0, log_names=logs)
            w["lead"] = dict(lead)
            blk = ad.four_families_block(w, tier="T1", n_boot=n_boot, per_window=per_window)
            blk["strategic"] = STRATEGIC_PAUSED
            out["frames"][name] = {"role": role, "block": blk}
        except Exception as e:                                                   # noqa: BLE001
            out["frames"][name] = {"role": role, "block": {"status": "RAISED", "error": f"{type(e).__name__}: {e}"}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--arm", required=True, help="arm name in the rows' `arms` dict (e.g. C1, T, human)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--per-window", action="store_true", help="DEFAULT OFF. attach per-window arrays for EVERY longitudinal/lateral/tactical metric under each frame block `_per_window`, plus top-level `tokens`/`logs` (array order); needed by fam_ci.py for a paired CI")
    ap.add_argument("--tanitad-repo", default=DEFAULT_REPO)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rows = [r for r in read_rows(a.rows) if a.arm in r["arms"]]
    if not rows:
        raise SystemExit("no rows with a fam payload")
    out = run(rows, a.arm, a.label, a.tanitad_repo, a.n_boot, a.per_window)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps({"n": out["n"], "n_logs": out["n_logs"], "lead_states": out["lead_states"],
                      "frames": {k: {f: (v["block"].get(f) or {}).get("status", "OK")
                                     for f in ("longitudinal", "lateral", "tactical")} if isinstance(v["block"], dict) else "?"
                                 for k, v in out["frames"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
