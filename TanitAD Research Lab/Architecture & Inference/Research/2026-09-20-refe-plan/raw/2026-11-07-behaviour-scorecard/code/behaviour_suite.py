#!/usr/bin/env python
"""behaviour_suite.py -- suite aggregation and the paired adoption estimator of BEHAVIOUR_SCORECARD.md section 4.

  aggregate : a directory of per-scenario *.json (behaviour_score.py output) -> per-rule pass counts, per-category rows,
              the four metric families (TACTICAL decision table N/A, STRATEGIC PAUSED), Gate H rule-definition defects,
              EPDMS passthrough. Never pools across rules; never pools the synthetic block with the nuPlan-seeded block.
  compare   : candidate vs baseline (+ baseline replicate = the sim floor) -> section 4 conditions 1-3, the EPDMS guard and
              the verdict. Estimator as written: paired bootstrap, cluster = scenario/log, 10,000 draws, seed 20260927,
              95 % percentile.
"""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

RULES = ("a1", "a2", "b1", "b2", "c1", "c2", "c3", "c4", "d1", "d2", "e1", "e2", "f1", "f2")
FAMILIES = {"LONGITUDINAL": ("a1", "a2", "b1", "b2", "d1", "d2", "c4"), "LATERAL": ("e1", "e2", "d1"), "TACTICAL": ("c1", "c2", "c3", "c4")}
BOOT_DRAWS, BOOT_SEED, ALPHA = 10000, 20260927, 0.05
EPDMS_GUARD = -0.03
PDMS_BAR = -0.5
GATE_H_DEFECT_SHARE = 0.30
STRATEGIC = "PAUSED (PI ruling 2026-09-19, PI_DECISION_QUEUE.md item 25): never a pass, never omitted"
TACTICAL_DECISION = "N/A: executed-decision agreement (go/yield/stop/follow/change-lane confusion table) is not implemented, see AMBIGUITIES.md A-14"
BLOCKS = ("nuplan_seeded", "synthetic_until_map_lifting")
# section 5: which open-loop proxy should have warned for which rule (for the "failed adoption leaves the next arm" block)
PROXY_FOR = {"a1": ["P-1", "P-2", "P-3", "P-4", "P-5(EP)"], "a2": ["P-1", "P-2", "P-3", "P-4"], "c1": ["P-3"], "d2": ["P-3"],
             "f1": ["P-5(NC)"], "f2": ["P-5(TTC)"], "e1": ["P-5(DAC)"], "d1": ["P-5(C)"]}
VARIANCE_STATEMENT = ("The interval answers 'would another draw of SCENARIOS (logs) say this?' with models held fixed. The sim floor (baseline run "
                      "twice, seeds 20261107/20261108) bounds sim variance. Training variance is NOT answered unless a training-seed replicate "
                      "of the candidate recipe is supplied (else the result is ONE-SEED and applies to that checkpoint only). Inference "
                      "variance is zero by construction (deterministic planner).")


def load_dir(d):
    out = []
    for f in sorted(glob.glob(str(Path(d) / "*.json"))):
        j = json.loads(Path(f).read_text(encoding="utf-8"))
        if "rules" in j and "scenario_id" in j:
            out.append(j)
    return sorted(out, key=lambda s: s["scenario_id"])


def block_of(sc):
    return BLOCKS[1] if sc.get("synthetic") else BLOCKS[0]


def _count(statuses):
    c = {k: 0 for k in ("PASS", "FAIL", "INFORMATIONAL", "NA")}
    for s in statuses:
        c[s] += 1
    sc = c["PASS"] + c["FAIL"]
    return dict(passes=c["PASS"], fails=c["FAIL"], informational=c["INFORMATIONAL"], na=c["NA"], n=sc,
                score=(c["PASS"] / sc if sc else None))


def gate_h_defects(scs):
    """a rule the human fails on > 30 % of its applicable scenarios is a rule-definition defect, not applied"""
    out = {}
    for r in RULES:
        hs = [s["rules"][r]["human_status"] for s in scs if s["rules"][r]["human_status"] in ("PASS", "FAIL")]
        share = (hs.count("FAIL") / len(hs)) if hs else None
        out[r] = dict(n=len(hs), human_fail_share=share, defect=bool(share is not None and share > GATE_H_DEFECT_SHARE))
    return out


def status_of(sc, r, defects):
    st = sc["rules"][r]["status"]
    return "INFORMATIONAL" if (defects[r]["defect"] and st in ("PASS", "FAIL")) else st


def aggregate(scs):
    res = {"scorecard_sha256": scs[0].get("scorecard_sha256") if scs else None, "n_scenarios": len(scs), "blocks": {}}
    for b in BLOCKS:
        bs = [s for s in scs if block_of(s) == b]
        if not bs:
            res["blocks"][b] = None
            continue
        dfx = gate_h_defects(bs)
        per_rule = {r: _count([status_of(s, r, dfx) for s in bs]) for r in RULES}
        cats = {}
        for c in sorted({s["category"] for s in bs}):
            cs = [s for s in bs if s["category"] == c]
            cats[c] = dict(n_scenarios=len(cs), rules={r: _count([status_of(s, r, dfx) for s in cs]) for r in RULES})
        slots = {}   # AMENDMENT_3 CH-4: DESCRIPTIVE per-slot rows (J-UL, J-R, J-S, TL-G, TL-S); never pooled, never an adoption count
        for sl in sorted({s["slot"] for s in bs if s.get("slot")}):
            ss = [s for s in bs if s.get("slot") == sl]
            slots[sl] = dict(n_scenarios=len(ss), rules={r: _count([status_of(s, r, dfx) for s in ss]) for r in RULES})
        fam = {f: {r: per_rule[r] for r in rs} for f, rs in FAMILIES.items()}
        fam["TACTICAL"] = dict(fam["TACTICAL"], executed_decision_agreement=TACTICAL_DECISION)
        fam["STRATEGIC"] = STRATEGIC
        ep = [s["epdms"]["value"] for s in bs if s.get("epdms", {}).get("value") is not None]
        idn = {k: sorted({str(s["identity"].get(k)) for s in bs}) for k in ("checkpoint_sha256", "dtype", "seed", "suite_tokens_sha256")}
        res["blocks"][b] = dict(
            label=("nuPlan-seeded slots" if b == BLOCKS[0] else "SYNTHETIC UNTIL MAP LIFTING EXISTS (not pooled with nuPlan-seeded slots)"),
            n_scenarios=len(bs), per_rule=per_rule, per_category=cats, families=fam, gate_h=dfx,
            **({"per_slot": slots} if slots else {}),
            tier_stamp=sorted({str(s.get("tier_stamp")) for s in bs}), n_agent_cls_defaulted=sum(1 for s in bs if s.get("agent_cls_defaulted")),
            epdms=dict(n=len(ep), mean=(float(np.mean(ep)) if ep else None), note="guard only; never a goal"),
            identity=dict(idn, consistent=all(len(v) == 1 for v in idn.values()), complete=all(s["identity"].get("complete") for s in bs)),
            human_pass_note="a rule the human log fails in a scenario is INFORMATIONAL there and in no pass count")
    return res


def md_table(agg):
    L = []
    for b, blk in agg["blocks"].items():
        if not blk:
            continue
        L += [f"### {blk['label']} (n={blk['n_scenarios']})", "", "| rule | passes | fails | informational | n/a | score |", "|---|---|---|---|---|---|"]
        for r in RULES:
            c = blk["per_rule"][r]
            L.append(f"| {r}{' (DEFECT, not applied)' if blk['gate_h'][r]['defect'] else ''} | {c['passes']} | {c['fails']} | {c['informational']} | {c['na']} | "
                     + (f"{c['score']:.3f} (n={c['n']})" if c["score"] is not None else "n/a") + " |")
        L += ["", "STRATEGIC: " + STRATEGIC, ""]
    return "\n".join(L)


# ----------------------------------------------------------------------------------------------------------
# paired scenario-cluster bootstrap (section 4 item 1)
# ----------------------------------------------------------------------------------------------------------
def paired_bootstrap(d, clusters, weights=None, draws=BOOT_DRAWS, seed=BOOT_SEED, alpha=ALPHA):
    """mean of paired differences d with a cluster bootstrap: resample clusters (logs) with replacement; the statistic of a
    draw is sum(w*d)/sum(w) over the resampled clusters. Percentile CI. Re-seeded per call so results do not depend on order."""
    d = np.asarray(d, float)
    w = np.ones_like(d) if weights is None else np.asarray(weights, float)
    cl = np.asarray(clusters)
    u, inv = np.unique(cl, return_inverse=True)
    K = len(u)
    sums = np.bincount(inv, weights=w * d, minlength=K)
    cnts = np.bincount(inv, weights=w, minlength=K)
    mean = float(sums.sum() / cnts.sum()) if cnts.sum() > 0 else float("nan")
    if K < 2:
        return dict(mean=mean, lo=mean, hi=mean, n=int(len(d)), n_clusters=int(K), draws=0, note="fewer than 2 clusters: degenerate interval")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, K, size=(draws, K))
    stat = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1e-12)
    lo, hi = np.percentile(stat, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(mean=mean, lo=float(lo), hi=float(hi), n=int(len(d)), n_clusters=int(K), draws=draws)


def _cells(scs, dfx):
    """rule -> {scenario_id: (indicator 1/0, log_id)} for PASS/FAIL (human-pass) cells"""
    out = {r: {} for r in RULES}
    for s in scs:
        for r in RULES:
            st = status_of(s, r, dfx)
            if st in ("PASS", "FAIL"):
                out[r][s["scenario_id"]] = (1.0 if st == "PASS" else 0.0, s.get("log_id") or s["scenario_id"])
    return out


def _rate(cells, ids):
    v = [cells[i][0] for i in ids if i in cells]
    return float(np.mean(v)) if v else None


def compare(cand, base, rep=None, tier=1, pdms=None, training_seed_replicate=False):
    cmap, bmap = {s["scenario_id"]: s for s in cand}, {s["scenario_id"]: s for s in base}
    rmap = {s["scenario_id"]: s for s in rep} if rep else {}
    out = dict(tier=tier, scorecard_sha256=(cand[0].get("scorecard_sha256") if cand else None), variance_answered=VARIANCE_STATEMENT,
               scenarios_only_in_one_arm=sorted(set(cmap) ^ set(bmap)), floor_available=bool(rep), blocks={})
    worse_any, fixed_any, notes = [], [], []
    for b in BLOCKS:
        ids = sorted(i for i in cmap if i in bmap and block_of(cmap[i]) == b)
        if not ids:
            continue
        cs, bs = [cmap[i] for i in ids], [bmap[i] for i in ids]
        rs = [rmap[i] for i in ids if i in rmap]
        # the human is the same in all arms; defects are decided on the baseline arm's human_status
        dfx = gate_h_defects(bs)
        cc, bc, rc = _cells(cs, dfx), _cells(bs, dfx), _cells(rs, dfx)
        blk = dict(n_paired=len(ids), rules={}, families={})
        for r in RULES:
            both = sorted(set(cc[r]) & set(bc[r]))
            if not both:
                blk["rules"][r] = dict(status="NOT PAIRED (no scenario where both arms are PASS/FAIL)")
                continue
            d = [cc[r][i][0] - bc[r][i][0] for i in both]
            bt = paired_bootstrap(d, [cc[r][i][1] for i in both])
            fl = None
            if rep:
                rr = sorted(set(bc[r]) & set(rc[r]))
                if rr:
                    fl = abs(_rate(bc[r], rr) - _rate(rc[r], rr))
            worse = bt["hi"] < 0 and fl is not None and (-bt["mean"]) > fl     # bad direction = pass-rate down
            better = bt["lo"] > 0 and fl is not None and bt["mean"] > fl
            n_fixed = sum(1 for i in both if bc[r][i][0] == 0 and cc[r][i][0] == 1)
            n_broke = sum(1 for i in both if bc[r][i][0] == 1 and cc[r][i][0] == 0)
            blk["rules"][r] = dict(n=len(both), cand_rate=_rate(cc[r], both), base_rate=_rate(bc[r], both), effect=bt["mean"], ci=[bt["lo"], bt["hi"]],
                                   n_clusters=bt["n_clusters"], floor=fl, separated_worse=bool(bt["hi"] < 0), worse_separated_above_floor=bool(worse),
                                   worse_ci_bound_above_floor=bool(fl is not None and (-bt["hi"]) > fl and bt["hi"] < 0),
                                   improved_separated_above_floor=bool(better), n_baseline_fail_cand_pass=n_fixed, n_baseline_pass_cand_fail=n_broke,
                                   fixed=bool(better and n_fixed >= 1))
            if worse:
                worse_any.append(f"{b}:{r}")
            if better and n_fixed >= 1:
                fixed_any.append(f"{b}:{r}")
        for f, rs_ in FAMILIES.items():                       # per family: mean pass over the family's applicable rules, per scenario
            def fvec(cells, i, rules=rs_):
                v = [cells[r][i][0] for r in rules if i in cells[r]]
                return float(np.mean(v)) if v else None
            ids_f = [i for i in ids if fvec(cc, i) is not None and fvec(bc, i) is not None]
            if not ids_f:
                blk["families"][f] = dict(status="NOT PAIRED")
                continue
            d = [fvec(cc, i) - fvec(bc, i) for i in ids_f]
            bt = paired_bootstrap(d, [cmap[i].get("log_id") or i for i in ids_f])
            fl = None
            if rep:
                rr = [i for i in ids_f if fvec(rc, i) is not None]
                if rr:
                    fl = abs(float(np.mean([fvec(bc, i) for i in rr])) - float(np.mean([fvec(rc, i) for i in rr])))
            worse = bt["hi"] < 0 and fl is not None and (-bt["mean"]) > fl
            blk["families"][f] = dict(n=len(ids_f), effect=bt["mean"], ci=[bt["lo"], bt["hi"]], floor=fl, worse_separated_above_floor=bool(worse))
            if worse:
                worse_any.append(f"{b}:family:{f}")
        out["blocks"][b] = blk
    # EPDMS guard (paired over scenarios)
    ids_e = [i for i in sorted(cmap) if i in bmap and cmap[i].get("epdms", {}).get("value") is not None and bmap[i].get("epdms", {}).get("value") is not None]
    if ids_e:
        bt = paired_bootstrap([cmap[i]["epdms"]["value"] - bmap[i]["epdms"]["value"] for i in ids_e], [cmap[i].get("log_id") or i for i in ids_e])
        out["epdms_guard"] = dict(n=len(ids_e), effect=bt["mean"], ci=[bt["lo"], bt["hi"]], bar=EPDMS_GUARD, veto=bool(bt["lo"] <= EPDMS_GUARD),
                                  note="guard only: can veto, can never adopt")
    else:
        out["epdms_guard"] = dict(status="NOT SUPPLIED", veto=False)
    # condition 3: navtest PDMS non-inferiority
    if pdms:
        bt = paired_bootstrap(pdms["diff_x100"], pdms["log_id"], weights=pdms.get("weight"))
        out["cond3_pdms"] = dict(n=bt["n"], effect_x100=bt["mean"], ci_x100=[bt["lo"], bt["hi"]], bar=PDMS_BAR, holds=bool(bt["lo"] > PDMS_BAR))
    else:
        out["cond3_pdms"] = dict(status="NOT SUPPLIED", holds=None)
    c1 = (len(worse_any) == 0) if rep else None
    c2 = len(fixed_any) > 0 if rep else None
    c3 = out["cond3_pdms"]["holds"]
    out["cond1_no_rule_worse"] = dict(holds=c1, worse=worse_any, note=None if rep else "baseline replicate missing: the sim floor cannot be read, so item 1 is NOT EVALUABLE")
    out["cond2_a_failing_rule_fixed"] = dict(holds=c2, fixed=fixed_any)
    veto = bool(out["epdms_guard"].get("veto"))
    label = "ONE-SEED (applies to this checkpoint only, never to the recipe)" if not training_seed_replicate else "training-seed replicate supplied"
    if not rep:
        verdict = "NOT EVALUABLE (baseline replicate / sim floor missing)"
    elif c1 is False:
        verdict = "REJECT (a rule is worse with a separated interval above the floor)"
    elif veto:
        verdict = "NOT ADOPTED (EPDMS guard veto)"
    elif tier == 1:
        verdict = "CANDIDATE nominated (tier 1 can reject or nominate, never adopt)" if (c2 and c3) else "NO DECISION at tier 1 (no rule worse; nomination needs conditions 2 and 3)"
    else:
        verdict = "ADOPTED" if (c1 and c2 and c3) else "NOT ADOPTED (conditions not all met)"
    out["verdict"] = verdict
    out["scope_label"] = label
    if verdict.startswith(("REJECT", "NOT ADOPTED", "NO DECISION")):
        out["next_arm"] = dict(failing_rules=sorted({w.split(":")[-1] for w in worse_any if "family" not in w}),
                               proxies_that_should_have_warned={w.split(":")[-1]: PROXY_FOR.get(w.split(":")[-1], []) for w in worse_any if "family" not in w},
                               cheapest_lever="NOT COMPUTED BY CODE: Rule Zero requires the reader to name it (section 4 last bullet)")
    return out


def load_pdms(cand_path, base_path):
    def rd(p):
        z = np.load(p, allow_pickle=False)
        return {str(t): (float(v), str(l), float(w)) for t, v, l, w in zip(z["token"], z["pdms"], z["log_id"], z["weight"] if "weight" in z.files else np.ones(len(z["token"])))}
    c, b = rd(cand_path), rd(base_path)
    ks = sorted(set(c) & set(b))
    return dict(diff_x100=[100.0 * (c[k][0] - b[k][0]) for k in ks], log_id=[c[k][1] for k in ks], weight=[c[k][2] for k in ks])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("aggregate")
    a.add_argument("--dir", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--md", default=None)
    c = sub.add_parser("compare")
    c.add_argument("--cand", required=True)
    c.add_argument("--base", required=True)
    c.add_argument("--base-replicate", default=None)
    c.add_argument("--tier", type=int, choices=(1, 2), required=True)
    c.add_argument("--pdms-cand", default=None)
    c.add_argument("--pdms-base", default=None)
    c.add_argument("--training-seed-replicate", action="store_true")
    c.add_argument("--out", required=True)
    x = ap.parse_args(argv)
    if x.cmd == "aggregate":
        agg = aggregate(load_dir(x.dir))
        Path(x.out).write_text(json.dumps(agg, sort_keys=True, indent=1), encoding="utf-8")
        if x.md:
            Path(x.md).write_text(md_table(agg), encoding="utf-8")
    else:
        pd = load_pdms(x.pdms_cand, x.pdms_base) if (x.pdms_cand and x.pdms_base) else None
        res = compare(load_dir(x.cand), load_dir(x.base), load_dir(x.base_replicate) if x.base_replicate else None, x.tier, pd, x.training_seed_replicate)
        Path(x.out).write_text(json.dumps(res, sort_keys=True, indent=1, default=lambda o: None if isinstance(o, float) and not math.isfinite(o) else str(o)), encoding="utf-8")
        print(res["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
