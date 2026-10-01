#!/usr/bin/env python3
"""SPEC_NAVTEST AMENDMENT 4 -- the pre-registered test of the on-policy scorer (PI decision B, 2026-09-26).

Reads, never recomputes: proptable/<name>/readout.json (E-6, Amendments 3 + 3a), points/<name>.json (the learning-curve
point with its paired floors) and the trainer's own metrics.jsonl (bank lines, epoch lines, on-policy lines).
Writes eval/RESULT_A4_<name>.md and proptable/<name>/amendment4.json; prints ZZA4 <name> <outcome>.

    python eval/amendment4_readout.py --name sub200_ep013 --metrics <local copy of the run's metrics.jsonl>

The rule, as written in SPEC_NAVTEST.md (AMENDMENT 4, 2026-09-26 ~12:15 Berlin, before the switch):
  PRIMARY    E-6 at the FIRST snapshot whose whole epoch trained with on-policy sets in its bank (the snapshot after
             the first epoch that STARTS after the switch). SUCCESS iff the selection skill's 95 % lower bound
             exceeds 0.25; FAILURE iff its upper bound is below 0.25; otherwise UNDETERMINED and the next snapshot
             decides.
  SECONDARY  (reported, not gating) the within-scene AUC of the drivable-area output (must rise above 0.60 to call the
             NAVSIM label learned); the pick's PDMS against the same tokens' STOP and the pre-switch snapshots; the
             training-side on-policy skill the trainer logs.
Operationalisation fixed HERE, 2026-09-26 ~13:55 Berlin, before any on-policy snapshot existed (the snapshot after
epoch 13 is due ~19:45): a snapshot N is written at the boundary INTO epoch N, so it trained on epochs < N; it is
eligible iff EVERY bank line of epoch N-1 reads scorer_mode "onpolicy" with > 0 sets; its rank among eligible snapshots
is reported (1 = the primary test; later ranks decide only after an UNDETERMINED). "Above 0.60" for the DAC AUC is the
point estimate, its 95 % interval printed beside it. The training-side skill pools the trainer's per-step on-policy
windows of epoch N-1 weighted by their set counts (no interval: the trainer logs window means, not per-set values;
within that epoch every set is met for the first time, so it is a held-out read).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "D:/Projects/TanitAD/data/refe_navtest"
BAR = 0.25            # the selection-skill bar (AMENDMENT 4)
DAC_BAR = 0.60        # the drivable-area within-scene AUC the NAVSIM label must exceed (secondary)
PRE_SWITCH = ("sub200_ep011", "sub200_ep012")   # the last fixed-candidate snapshots on the same tokens


def outcome(lo: float, hi: float, bar: float = BAR) -> str:
    if lo > bar:
        return "SUCCESS"
    if hi < bar:
        return "FAILURE"
    return "UNDETERMINED"


def read_metrics(path: str):
    """The trainer's own lines: every bank line per epoch (a restart re-logs one), epoch starts, on-policy windows."""
    banks, starts, op = {}, {}, {}
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln.replace("NaN", "null"))
        except ValueError:
            continue
        ev = r.get("event")
        if ev == "bank":
            banks.setdefault(r["epoch"], []).append(r)
        elif ev == "epoch":
            starts[r["epoch"]] = r["step"]          # a replayed boundary keeps its (identical) last step
        elif ev == "onpolicy" and (r.get("sets") or 0) > 0:
            op[r["step"]] = r                        # a replayed step keeps its last record
    return banks, starts, op


def onpolicy_epoch(lines) -> bool:
    return bool(lines) and all(b.get("scorer_mode") == "onpolicy" and (b.get("onpolicy_sets") or 0) > 0 for b in lines)


def eligibility(snapshot_epoch: int, banks: dict):
    """(eligible, rank among eligible snapshots, why) for the snapshot written at the boundary INTO `snapshot_epoch`."""
    e = snapshot_epoch - 1
    if not onpolicy_epoch(banks.get(e, [])):
        modes = [(b.get("scorer_mode", "fixed"), b.get("onpolicy_sets")) for b in banks.get(e, [])]
        return False, None, f"epoch {e} (the one this snapshot trained last) was not on-policy from its start: {modes}"
    first = min(k for k in banks if onpolicy_epoch(banks[k]))
    return True, e - first + 1, f"epoch {e} read {banks[e][0]['onpolicy_sets']:,} on-policy sets at its start"


def snapshot_epoch(name: str, banks: dict):
    """`..._epNNN` was written at the boundary INTO epoch NNN; `..._final` is the model after the last step, placed one
    past the last epoch the trainer read a bank for (so it has trained that epoch, the last one, partly)."""
    if name.endswith("_final"):
        return (max(banks) + 1) if banks else None
    tail = name.rsplit("ep", 1)[-1]
    return int(tail) if tail.isdigit() else None


def decided_before(banks: dict, rank: int, data: str):
    """The first EARLIER on-policy snapshot whose read decided the test (SUCCESS / FAILURE), else None. Only an
    UNDETERMINED primary hands the decision to the next snapshot; after a decision later reads are reported, not gating."""
    first = min(k for k in banks if onpolicy_epoch(banks[k]))
    for r in range(1, rank):
        name = f"sub200_ep{first + r:03d}"
        p = os.path.join(data, "proptable", name, "amendment4.json")
        if not os.path.exists(p):
            return {"name": name, "outcome": "MISSING"}      # a gap in the chain is not an UNDETERMINED
        o = json.load(open(p, encoding="utf-8"))["primary"]["outcome"]
        if o in ("SUCCESS", "FAILURE"):
            return {"name": name, "outcome": o}
    return None


def training_skill(op: dict, lo_step: int, hi_step: int):
    rows = [op[s] for s in sorted(op) if lo_step <= s < hi_step]
    n = sum(r["sets"] for r in rows)
    if not n:
        return None
    p, rd, b = (sum(r[k] * r["sets"] for r in rows) / n for k in ("pick", "random", "best"))
    return {"steps": len(rows), "sets": n, "pick": round(p, 4), "random": round(rd, 4), "best": round(b, 4),
            "skill": round((p - rd) / (b - rd), 4) if b - rd > 1e-9 else None}


def pair(point: dict, key: str):
    p = ((point.get("floors") or {}).get("pairs") or {}).get(key)
    if not p:
        return None
    iv = p.get("interval") or {}
    f = lambda v: None if v is None else round(100.0 * v, 2)
    return {"delta": p["delta_x100"], "lo": f(iv.get("lo")), "hi": f(iv.get("hi")), "separated": iv.get("separated")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="e.g. sub200_ep013")
    ap.add_argument("--metrics", required=True, help="a local copy of the live run's metrics.jsonl")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--doc-dir", default=HERE, help="where RESULT_A4_<name>.md goes (the self-test passes a temp dir)")
    a = ap.parse_args()
    banks, starts, op = read_metrics(a.metrics)
    ep = snapshot_epoch(a.name, banks)
    if ep is None:
        print(f"ZZA4 {a.name} NOT_ELIGIBLE the name is neither ..._epNNN nor ..._final")
        return 2
    ok, rank, why = eligibility(ep, banks)
    if not ok:
        print(f"ZZA4 {a.name} NOT_ELIGIBLE {why}")
        return 2
    R = json.load(open(os.path.join(a.data, "proptable", a.name, "readout.json"), encoding="utf-8"))
    P = json.load(open(os.path.join(a.data, "points", a.name + ".json"), encoding="utf-8"))
    s = R["a_pdms"]["selection_skill"]
    res = outcome(s["ci95"][0], s["ci95"][1])
    dac = R["c_components"]["DAC"]["within_token_auc"]
    # the final model's last epoch has no closing boundary: its window runs to the last on-policy step logged
    lo_step, hi_step = starts.get(ep - 1), starts.get(ep, (max(op) + 1) if op else None)
    tr = training_skill(op, lo_step, hi_step) if lo_step is not None and hi_step is not None else None
    base = {}
    for b in PRE_SWITCH:
        rp = os.path.join(a.data, "proptable", b, "readout.json")
        if os.path.exists(rp):
            x = json.load(open(rp, encoding="utf-8"))["a_pdms"]
            base[b] = {"oracle": x["oracle"]["mean"], "pick": x["actual"]["mean"], "random": x["random"]["mean"],
                       "skill": x["selection_skill"]["value"], "skill_ci95": x["selection_skill"]["ci95"]}
    prior = decided_before(banks, rank, a.data) if rank > 1 else None
    decides = prior is None                     # rank 1, or every earlier read was UNDETERMINED
    out = {
        "name": a.name, "rule": "SPEC_NAVTEST AMENDMENT 4", "bar": BAR, "rank_among_onpolicy_snapshots": rank,
        "eligibility": why, "primary": {"skill": s["value"], "ci95": s["ci95"], "outcome": res,
                                        "decides": decides, "decided_earlier_by": prior},
        "e6_verdict_3a": R["verdict"]["outcome"],
        "pdms": {"oracle": R["a_pdms"]["oracle"]["mean"], "pick": R["a_pdms"]["actual"]["mean"],
                 "pick_ci95": R["a_pdms"]["actual"]["ci95"], "random": R["a_pdms"]["random"]["mean"]},
        "secondary": {
            "dac_within_scene_auc": {"mean": dac["mean"], "ci95": dac["ci95"], "n": dac.get("n"),
                                     "above_0p60": dac["mean"] > DAC_BAR},
            "pick_vs_stop": pair(P, "REFe__minus__STOP"),
            "pick_vs_pre_switch": {b: pair(P, "REFe__minus__REFe_" + b) for b in PRE_SWITCH},
            "training_side_skill": tr, "training_epoch": ep - 1, "training_steps": [lo_step, hi_step],
        },
        "pre_switch_e6": base,
    }
    json.dump(out, open(os.path.join(a.data, "proptable", a.name, "amendment4.json"), "w", encoding="utf-8"), indent=1)

    f = lambda v, d=3: "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{d}f}"
    iv = lambda v, d=3: f"{f(v['mean'] if 'mean' in v else v['skill'], d)} [{f(v['ci95'][0], d)}, {f(v['ci95'][1], d)}]"
    pr = lambda p: "–" if not p else (f"{p['delta']:+.2f} [{f(p['lo'], 2)}, {f(p['hi'], 2)}]"
                                     + (" · separated" if p["separated"] else " · not separated"))
    sec = out["secondary"]
    L = [f"# RESULT -- SPEC_NAVTEST AMENDMENT 4 -- `{a.name}` (the on-policy scorer, PI decision B)", "",
         "*Every number below is read from `proptable/" + a.name + "/readout.json`, `points/" + a.name + ".json` and "
         "the trainer's `metrics.jsonl`; none is typed. Written by `eval/amendment4_readout.py`.*", "",
         f"**Eligibility:** {why}; rank {rank} among on-policy snapshots"
         + (" -- this is the PRIMARY test." if rank == 1
            else " -- read because every earlier on-policy snapshot was UNDETERMINED; this read decides." if decides
            else f" -- the test was already decided ({prior['outcome']} at `{prior['name']}`); this read is REPORTED, "
                 f"NOT GATING."),
         "",
         (f"## Primary: **{res}**" if decides else f"## Reported, not gating: {res}"), "",
         f"Selection skill (pick - random) / (oracle - random), paired log-cluster bootstrap over 93 logs: "
         f"**{f(s['value'])} [{f(s['ci95'][0])}, {f(s['ci95'][1])}]** against the bar {BAR}: SUCCESS iff the lower "
         f"bound exceeds it, FAILURE iff the upper bound is below it. E-6's own verdict (Amendment 3a): "
         f"{R['verdict']['outcome']}.", "",
         "| snapshot | scorer learned from | best of 64 | pick | random | skill [95 %] |", "|---|---|---|---|---|---|"]
    for b, x in base.items():
        L.append(f"| {b} | fixed candidates | {x['oracle']:.1f} | {x['pick']:.1f} | {x['random']:.1f} | "
                 f"{f(x['skill'])} [{f(x['skill_ci95'][0])}, {f(x['skill_ci95'][1])}] |")
    L.append(f"| **{a.name}** | its own proposals | {out['pdms']['oracle']:.1f} | **{out['pdms']['pick']:.1f}** | "
             f"{out['pdms']['random']:.1f} | **{f(s['value'])} [{f(s['ci95'][0])}, {f(s['ci95'][1])}]** |")
    L += ["", "## Secondary (reported, not gating)", "",
          f"- **Drivable-area output, within-scene AUC:** {iv(sec['dac_within_scene_auc'])} (n {sec['dac_within_scene_auc']['n']}) "
          f"-- {'ABOVE' if sec['dac_within_scene_auc']['above_0p60'] else 'NOT above'} {DAC_BAR} (point estimate).",
          f"- **Pick vs standing still (same tokens), PDMS points:** {pr(sec['pick_vs_stop'])}."]
    for b, p in sec["pick_vs_pre_switch"].items():
        L.append(f"- **Pick vs {b}:** {pr(p)}.")
    if tr:
        L.append(f"- **Training-side on-policy skill, epoch {ep - 1} (steps {lo_step}-{hi_step - 1}):** {f(tr['skill'])} "
                 f"over {tr['sets']:,} sets in {tr['steps']} steps (pick {f(tr['pick'])}, random {f(tr['random'])}, "
                 f"best {f(tr['best'])}; no interval -- the trainer logs window means; every set met for the first time).")
    else:
        L.append("- **Training-side on-policy skill:** not available in the metrics given.")
    L += ["", "A FAILURE goes to the PI with this table; nothing in the recipe changes again without the PI "
          "(SPEC_NAVTEST, AMENDMENT 4)."]
    open(os.path.join(a.doc_dir, f"RESULT_A4_{a.name}.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print(f"ZZA4 {a.name} {res} skill={f(s['value'])} [{f(s['ci95'][0])}, {f(s['ci95'][1])}] rank={rank} "
          f"gating={'yes' if decides else 'no (decided by ' + prior['name'] + ': ' + prior['outcome'] + ')'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
