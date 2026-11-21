#!/usr/bin/env python
"""read_ip1.py -- card PREREG_V8F_PROOF_ADD44 (INTEG-PRE). numpy only. Written BEFORE any run of this read existed. Names are those of section 4 of the card as written; nothing is added.

  python read_ip1.py --plan  --small DIR --pull DIR --arms FILE --out plan.json     -> prints the seeds (bases) that can be read, e.g. 0,1,2,3 ; the rest is reported as missing
  python read_ip1.py --read  --plan-file plan.json --hw RESULT_HEADWAY.json --q3 RESULT_Q3.json --small DIR --pull DIR --out-json READ_IP1.json --out-md READ_IP1.md

Runs (pull_t1/<RUN>_s<k>): C = RLAm, K = RLFT; IR0 = no rule, IRL = rule fed by labels, IRQ = rule with g_used = min(g_head, q_head); banked IPox_<arm>_s<k> = the run without a planner (rows 91 and 95).
Stage 1: the controls c1 to c6 (all must pass, else no name and no outcome difference is printed). Stage 2: D1 to D5, share reached, contact rates, the family reads, the names.
Tier: T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and the reader has no strategic instrument)."""
import argparse
import hashlib
import io
import json
import os
import re
import sys

import numpy as np

TIER = "T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance"
STRATEGIC = "STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and the reader has no strategic instrument)"
C_ARM, K_ARM = "RLAm", "RLFT"
H5 = "h5_check_coverage_in_0.15_0.25"
FIT_TOOL_SHA = "57e1242713e0b42eab05cf142b8d152fbb87db94354290ddd3f275f06292036b"
CLIP_SETS = {"fit": [0, 3048], "select": [3048, 3810], "check": [3810, 4572]}
MIN_BASES = 3
N_WIN, N_LEAD, N_EP = 2399, 642, 56
C6_TOL = 1e-9


def jl(p):
    return json.load(io.open(p, encoding="utf-8")) if p and os.path.exists(p) else None


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run_name(tag, arm):
    return "%s_%s" % (tag, arm)


def pair_key(a, b):
    return "%s_minus_%s" % (a, b)


def pairs_of_card():
    """The pairs the readers must be given (a, b) = the difference a minus b of the card."""
    c, k = C_ARM, K_ARM
    return [("IRQ_" + c, "IR0_" + k), ("IRQ_" + c, "IR0_" + c), ("IRQ_" + c, "IRL_" + c), ("IRQ_" + c, "IRQ_" + k), ("IR0_" + c, "IR0_" + k),
            ("IRL_" + c, "IR0_" + c), ("IRL_" + k, "IR0_" + k), ("IR0_" + c, "IPox_" + c), ("IR0_" + k, "IPox_" + k)]


def arms_bases(path):
    out = []
    for ln in io.open(path, encoding="utf-8"):
        t = ln.split()
        if t and not t[0].startswith("#") and len(t) >= 5:
            out.append(t[2])
    return out


def base_index(base):
    return int(base.rsplit("_s", 1)[1])


def plan(small, pull, arms, out):
    """A base is read only if both of its checkpoints have a head inside the coverage band and the three rollouts of both are pulled."""
    bases = arms_bases(arms)
    idx = sorted({base_index(b) for b in bases})
    missing, ok = {}, []
    for k in idx:
        why = []
        for arm in (C_ARM, K_ARM):
            b = "%s_s%d" % (arm, k)
            rec = jl(os.path.join(small, "heads_%s.qhead.json" % b))
            if os.path.exists(os.path.join(small, "run_MISSING_%s.txt" % b)):
                why.append("%s: MISSING marker (check coverage outside [0.15, 0.25])" % b)
            elif rec is None:
                why.append("%s: no fit record" % b)
            elif rec.get(H5) is not True:
                why.append("%s: check coverage outside [0.15, 0.25]" % b)
            for t in ("IR0", "IRL", "IRQ"):
                if not os.path.exists(os.path.join(pull, "%s_%s" % (t, b), "PULLED.txt")):
                    why.append("%s_%s not pulled" % (t, b))
        if why:
            missing["s%d" % k] = why
        else:
            ok.append(k)
    res = {"bases_read": ok, "missing": missing, "n_bases_read": len(ok), "bases_requested": idx}
    with io.open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    return res


def cell(c):
    return {"per_base": [round(float(x), 6) for x in c["d_s"]], "pooled": round(float(c["D"]), 6), "interval": [round(float(c["A_lo"]), 6), round(float(c["A_hi"]), 6)],
            "excludes_0": bool(c["A_excludes_0"]), "n_same_sign_as_pooled": int(sum(1 for x in c["d_s"] if np.sign(x) == np.sign(c["D"]) and x != 0))}


class Reader:
    def __init__(self, hw, q3, bases):
        self.hw, self.q3, self.bases = hw, q3, bases

    def U(self, a, b):
        k = pair_key(a, b)
        return self.hw["pairs"][k]["cells"]["U"] if self.hw and k in self.hw.get("pairs", {}) else None

    def contact(self, a, b):
        k = pair_key(a, b)
        try:
            return {n: cell(v) for n, v in self.hw["pairs"][k]["cells"].items() if n.startswith("C") and isinstance(v, dict) and "d_s" in v}
        except Exception as ex:                                   # noqa: BLE001
            return "not available: %s" % ex

    def fam(self, a, b):
        k = pair_key(a, b)
        return self.q3["pairs"][k] if self.q3 and k in self.q3.get("pairs", {}) else None


def controls(R, bases, small, pull, hwj, q3j):
    C = {}
    c, k = C_ARM, K_ARM
    # c1 heads
    c1 = {}
    for arm in (c, k):
        for kk in bases:
            b = "%s_s%d" % (arm, kk)
            rec = jl(os.path.join(small, "heads_%s.qhead.json" % b))
            hp = os.path.join(small, "heads_%s.qhead.npz" % b)
            pl = os.path.join(pull, "IRQ_%s" % b, "PULLED.txt")
            rl = os.path.join(pull, "IRQ_%s" % b, "ticks", "rule_log_IRQ_%s.npz" % b)
            e = {"PASS": False}
            if rec is None or not os.path.exists(hp) or not os.path.exists(pl):
                e["why"] = "record, head file or PULLED.txt missing"
                c1[b] = e
                continue
            hs = sha_file(hp)
            ck = re.search(r"ckpt_sha256 ([0-9a-f]{64})", io.open(pl, encoding="utf-8").read())
            run_sha = ""
            if os.path.exists(rl):
                z = np.load(rl, allow_pickle=False)
                run_sha = str(z["qhead_sha256"].item()) if "qhead_sha256" in z.files else ""
            sc = (rec.get("coverage_check") or {}).get("share_q_above_label")
            tool = (rec.get("tool_sha256") or {}).get("qhead_fit_n1.py")
            e.update({"head_sha_equals_record": hs in (rec.get("head_file_sha256") or {}).values(),
                      "record_checkpoint_is_the_run_checkpoint": bool(ck and (rec.get("states_record") or {}).get("ckpt_sha256") == ck.group(1)),
                      "head_in_run_equals_head_file": run_sha == hs,
                      "fit_form_is_ADD42_N2": bool(tool == FIT_TOOL_SHA and rec.get("clip_sets") == CLIP_SETS),
                      "coverage_in_band": bool(rec.get(H5) is True and sc is not None and 0.15 <= sc <= 0.25), "check_coverage": sc})
            e["PASS"] = all(v is True for kk2, v in e.items() if kk2 not in ("PASS", "check_coverage"))
            c1[b] = e
    C["c1_heads"] = {"per_checkpoint": c1, "PASS": bool(c1 and all(v["PASS"] for v in c1.values()))}
    # c2 known positive
    c2 = {}
    for arm in (c, k):
        u = R.U("IRL_" + arm, "IR0_" + arm)
        c2[arm] = dict(cell(u), PASS=bool(u["D"] < 0 and u["A_excludes_0"] and all(d < 0 for d in u["d_s"]))) if u else {"PASS": False, "why": "pair missing"}
    C["c2_label_rule_cuts_U"] = {"per_arm": c2, "PASS": bool(all(v["PASS"] for v in c2.values()))}
    # c3 identity of the tool on this arm type
    lg = ""
    p = os.path.join(small, "run_integpre_queue.log")
    if os.path.exists(p):
        lg = io.open(p, encoding="utf-8", errors="replace").read()
    m = re.findall(r"c3 the identity form .*?: .*?files=(\d+) differing=(\d+)", lg)
    C["c3_identity_on_C_base0"] = {"compare_lines": m, "PASS": bool(m and all(int(d) == 0 and int(n) > 0 for n, d in m) and "control c3 passed" in lg)}
    # c4 counts
    cnt = (hwj or {}).get("counts", {})
    miss = (hwj or {}).get("missing", None)
    pk = [pair_key(x, y) for x, y in pairs_of_card()]
    pres = bool(all(q in hwj.get("pairs", {}) for q in pk) and all(q in q3j.get("pairs", {}) for q in pk))
    C["c4_counts"] = {"n_windows_total": cnt.get("n_windows_total"), "n_lead_windows": cnt.get("n_lead_windows"), "n_episodes_with_lead": cnt.get("n_episodes_with_lead"), "missing_dumps": miss, "all_pairs_present": pres,
                      "PASS": bool(cnt.get("n_windows_total") == N_WIN and cnt.get("n_lead_windows") == N_LEAD and cnt.get("n_episodes_with_lead") == N_EP and miss == [] and pres)}
    # c5 reader controls
    C["c5_reader_controls"] = {"headway_h1_to_h6": bool(hwj and hwj["controls"].get("ALL_PASS")), "family_q1_to_q5": bool(q3j and q3j["controls"].get("ALL_PASS")),
                               "PASS": bool(hwj and q3j and hwj["controls"].get("ALL_PASS") and q3j["controls"].get("ALL_PASS"))}
    # c6 same model as banked
    c6 = {}
    for arm in (c, k):
        u = R.U("IR0_" + arm, "IPox_" + arm)
        c6[arm] = {"per_base_difference": [float(x) for x in u["d_s"]], "PASS": bool(all(abs(x) <= C6_TOL for x in u["d_s"]))} if u else {"PASS": False, "why": "pair missing (banked IPo dump not read)"}
    C["c6_R0_equals_banked_IPo"] = {"per_arm": c6, "tolerance": C6_TOL, "PASS": bool(all(v["PASS"] for v in c6.values()))}
    return C


def names(R, bases):
    D = {}
    N = {}
    d1, d2, d3, d4, d5 = R.U("IRQ_" + C_ARM, "IR0_" + K_ARM), R.U("IRQ_" + C_ARM, "IR0_" + C_ARM), R.U("IRQ_" + C_ARM, "IRL_" + C_ARM), R.U("IRQ_" + C_ARM, "IRQ_" + K_ARM), R.U("IR0_" + C_ARM, "IR0_" + K_ARM)
    ul = R.U("IRL_" + C_ARM, "IR0_" + C_ARM)
    for nm, x in (("D1_Cq_minus_Kr0", d1), ("D2_Cq_minus_Cr0", d2), ("D3_Cq_minus_Clabel", d3), ("D4_Cq_minus_Kq", d4), ("D5_Cr0_minus_Kr0", d5)):
        D[nm] = cell(x) if x else None
    share = [float(a) / float(b) for a, b in zip(d2["d_s"], ul["d_s"])]
    D["share_reached_per_base"] = {"values": [round(s, 4) for s in share], "bases": ["s%d" % kk for kk in bases], "n_at_least_0.5": int(sum(1 for s in share if s >= 0.5))}
    D["contact_Cq_minus_Clabel"] = R.contact("IRQ_" + C_ARM, "IRL_" + C_ARM)
    nneg = int(sum(1 for d in d1["d_s"] if d < 0))
    N["cut"] = "INTEG-PRE-CUTS-UNSAFE" if (nneg >= MIN_BASES and d1["A_excludes_0"] and d1["A_hi"] < 0) else "INTEG-PRE-NO-CUT"
    N["reaches_label_rule"] = "INTEG-PRE-REACHES-LABEL-RULE" if (D["share_reached_per_base"]["n_at_least_0.5"] >= MIN_BASES and d2["A_hi"] < 0) else "not named"
    N["levers"] = "INTEG-PRE-LEVERS-INTERFERE" if d4["A_lo"] > 0 else "INTEG-PRE-LEVERS-COMPATIBLE"
    f1, f2 = R.fam("IRQ_" + C_ARM, "IR0_" + K_ARM), R.fam("IRQ_" + C_ARM, "IR0_" + C_ARM)
    q = f1["Q3"]
    cost = q.get("cost_families") or []
    N["price"] = "INTEG-PRE-NO-COST" if not cost else " ".join("INTEG-PRE-COSTS-" + str(f) for f in cost)
    N["price_final_flag"] = bool(q.get("final"))
    fam = {"C_RQ_against_K_R0": {"Q3": q, "ade6_m": cell(f1["beside"]["ade6_m|all"]) if f1.get("beside", {}).get("ade6_m|all") else None},
           "C_RQ_against_C_R0": {"Q3": f2["Q3"] if f2 else None, "ade6_m": cell(f2["beside"]["ade6_m|all"]) if f2 and f2.get("beside", {}).get("ade6_m|all") else None}}
    return N, D, fam


def render_md(res):
    L = ["# READ_IP1 (card PREREG_V8F_PROOF_ADD44, INTEG-PRE)", "", "Tier: " + TIER, "", STRATEGIC, "", "Bases read: %s. Missing: %s" % (res["bases_read"], json.dumps(res["missing"])), "", "## Controls"]
    for k, v in res["controls"].items():
        L.append("* %s: %s" % (k, "PASS" if v.get("PASS") else "FAIL"))
    L.append("")
    if res.get("names"):
        L += ["## Names", "```", json.dumps(res["names"], indent=1), "```", "## Differences (U, per base | pooled | two-level interval | same-sign count)"]
        for k, v in res["differences"].items():
            L.append("* %s: %s" % (k, json.dumps(v)[:400]))
        L += ["## Families", "```", json.dumps(res["families"], indent=1)[:3000], "```"]
    else:
        L.append("## Names: NOT GIVEN (%s); the outcome differences are not printed" % res.get("why_no_names"))
    return "\n".join(L) + "\n"


def read(plan_p, hw_p, q3_p, small, pull, out_json, out_md):
    pl = jl(plan_p)
    bases = pl["bases_read"]
    hwj, q3j = jl(hw_p), jl(q3_p)
    res = {"card": "PREREG_V8F_PROOF_ADD44", "tier": TIER, "strategic": STRATEGIC, "bases_read": bases, "missing": pl["missing"], "controls": {}, "names": None}
    if len(bases) < MIN_BASES:
        res["controls_all_pass"] = None
        res["why_no_names"] = "fewer than %d bases read (%d)" % (MIN_BASES, len(bases))
    elif not (hwj and q3j):
        res["controls_all_pass"] = False
        res["why_no_names"] = "a reader result is absent"
    else:
        R = Reader(hwj, q3j, bases)
        res["controls"] = controls(R, bases, small, pull, hwj, q3j)
        allp = all(v.get("PASS") for v in res["controls"].values())
        res["controls_all_pass"] = bool(allp)
        if allp:
            res["names"], res["differences"], res["families"] = names(R, bases)
        else:
            res["why_no_names"] = "a control failed: " + ", ".join(k for k, v in res["controls"].items() if not v.get("PASS"))
    print("== READ_IP1. " + TIER)
    for k, v in res["controls"].items():
        print("  ", k, "PASS" if v.get("PASS") else "FAIL")
    print("   bases read %s, missing %s" % (bases, json.dumps(pl["missing"])))
    print("   names:", json.dumps(res["names"]) if res["names"] else "NOT GIVEN (%s)" % res.get("why_no_names"))
    print("   " + STRATEGIC)
    with io.open(out_json, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    with io.open(out_md, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_md(res))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--read", action="store_true")
    ap.add_argument("--plan-file", default="")
    ap.add_argument("--hw", default="")
    ap.add_argument("--q3", default="")
    ap.add_argument("--small", default="")
    ap.add_argument("--pull", default="")
    ap.add_argument("--arms", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--pairs", action="store_true", help="print the pairs the readers must be given")
    a = ap.parse_args(argv)
    if a.pairs:
        print(",".join("%s:%s" % p for p in pairs_of_card()))
        return 0
    if a.plan:
        r = plan(a.small, a.pull, a.arms, a.out)
        print(",".join(str(x) for x in r["bases_read"]))
        return 0
    if a.read:
        read(a.plan_file, a.hw, a.q3, a.small, a.pull, a.out_json, a.out_md)
        return 0
    ap.error("one of --plan, --read, --pairs")


if __name__ == "__main__":
    sys.exit(main())
