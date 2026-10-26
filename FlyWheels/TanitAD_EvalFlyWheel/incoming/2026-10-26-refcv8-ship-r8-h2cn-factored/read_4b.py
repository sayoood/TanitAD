"""read_4b.py -- AMENDMENT 4b: (a) the H2CN_DEPLOY8 REPRODUCTION (fresh out_4b/ vs the landed out/), (b) a SUMMARY of the 4b read.  Pure json / npz: no battery import,
no GPU, no write outside out_4b/.  NEW file (EvalFlyWheel 2026-10-26).

    python read_4b.py            -> out_4b/REPRO_H2CN_DEPLOY8_vs_landed.json, out_4b/SUMMARY_4B.json, and a text table on stdout

REPRODUCTION is judged on ARTIFACT CONTENT (exact equality of every leaf), never on an exit code.  The only exclusions are the ones a fresh run cannot reproduce by
construction and the 4b-only additions, each NAMED in the record:
  * wall-clock fields: clause 1's `registration_gate_passed_epoch` and `first_eval_output_epoch` (this run's own timestamps; the registration-before-output ORDER is
    still asserted by K_REG in both);
  * file-path / run-tag strings that name the output directory (`out/arms/h2cn` vs `out_4b/arms/h2cn_cap8`) -- compared with the directory token normalised;
  * the H2CN_CAP8 arm and its `cap8` per-window block, which the landed run does not have.
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

W = Path("C:/Users/Admin/refcv8_pf/ship_r8_sub")
OLD, NEW = W / "out", W / "out_4b"
TREE = "step48852ema_pf"
LANDED_ARMS = ["H2CN_DEPLOY8", "H2CN_NEAR", "H2CN_PERSURFACE", "RETIME_DEPLOY8", "RETIME_DEPLOY_PERSURFACE", "CONST_G1"]
WALL_CLOCK = ("registration_gate_passed_epoch", "first_eval_output_epoch")


def jl(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def norm(x):
    """normalise the output-directory token in strings, so a path that merely names out/ vs out_4b/ is not a content difference."""
    if isinstance(x, str):
        return x.replace("/out_4b/", "/OUT/").replace("/out/", "/OUT/").replace("arms/h2cn_cap8", "arms/ARMS").replace("arms/h2cn", "arms/ARMS").replace("\\", "/")
    return x


def exact_equal(a, b, path="", skip=()):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if k in skip:
                continue
            out += [f"{path}/{k}: one side only"] if (k not in a or k not in b) else exact_equal(a[k], b[k], f"{path}/{k}", skip)
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += exact_equal(x, y, f"{path}[{i}]", skip)
        return out
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return []
    return [] if norm(a) == norm(b) else [f"{path}: {a!r} != {b!r}"]


def leaves(x):
    if isinstance(x, dict):
        return sum(leaves(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(leaves(v) for v in x)
    return 1


def repro() -> dict:
    fo, fn = jl(OLD / "forward" / f"FORWARD_H2CN_{TREE}.json"), jl(NEW / "forward" / f"FORWARD_H2CN_CAP8_{TREE}.json")
    rec = {"landed_forward": str(OLD / "forward" / f"FORWARD_H2CN_{TREE}.json"), "fresh_forward": str(NEW / "forward" / f"FORWARD_H2CN_CAP8_{TREE}.json"), "blocks": {}}
    # (1) forward: every seed / surface / set / arm block + bars + the landed controls
    sn = copy.deepcopy(fn["seeds"])
    for s in sn.values():
        for su in s["surfaces"].values():
            for st in su["sets"].values():
                for a in list(st["arms"]):
                    if a not in LANDED_ARMS:
                        st["arms"].pop(a)
    m = exact_equal(fo["seeds"], sn)
    rec["blocks"]["forward_seeds_all_landed_arms"] = {"n_leaves": leaves(fo["seeds"]), "n_mismatches": len(m), "first": m[:8]}
    bn = {k: v for k, v in fn["bars"].items() if k.split("|")[1] in LANDED_ARMS}
    m = exact_equal(fo["bars"], bn)
    rec["blocks"]["forward_bars_all_landed_arms"] = {"n_leaves": leaves(fo["bars"]), "n_mismatches": len(m), "first": m[:8]}
    ctl_keys = [k for k in fo["controls"] if k not in ("K_REG_detail", "required")]
    m = exact_equal({k: fo["controls"][k] for k in ctl_keys}, {k: fn["controls"][k] for k in ctl_keys}, skip=WALL_CLOCK)
    rec["blocks"]["forward_controls_landed_keys"] = {"n_leaves": leaves({k: fo["controls"][k] for k in ctl_keys}), "n_mismatches": len(m), "first": m[:8]}
    # `required` (the list ALL_PASS is the conjunction of) legitimately gains the three K-CAP names: first 11 must equal the landed list, the extra must be EXACTLY those three, in order
    ro, rn = fo["controls"]["required"], fn["controls"]["required"]
    req_ok = bool(rn[:len(ro)] == ro and rn[len(ro):] == ["K_CAP_ID", "K_CAP_COMPLY", "K_CAP_MONO"])
    rec["blocks"]["forward_controls_required_list"] = {"n_leaves": len(ro), "n_mismatches": 0 if req_ok else 1, "landed": ro, "fresh_extra": rn[len(ro):]}
    pw = {k: {kk: vv for kk, vv in v.items() if kk != "cap8"} for k, v in fn["per_window_extra"].items()}
    m = exact_equal(fo["per_window_extra"], pw)
    rec["blocks"]["per_window_extra_minus_cap8"] = {"n_leaves": leaves(fo["per_window_extra"]), "n_mismatches": len(m), "first": m[:8]}
    cn = {k: {"arms": {a: v for a, v in c["arms"].items() if a != "H2CN_CAP8"}, **{kk: vv for kk, vv in c.items() if kk != "arms"}} for k, c in fn["compliance_row"]["cells"].items()}
    m = exact_equal(fo["compliance_row"]["cells"], cn)
    rec["blocks"]["compliance_row_cells_landed_arms"] = {"n_leaves": leaves(fo["compliance_row"]["cells"]), "n_mismatches": len(m), "first": m[:8]}
    # (2) the per-arm position packages: arrays bit-equal
    for arm in ("H2CN_DEPLOY8", "H2CN_NEAR", "RETIME_DEPLOY8"):
        n_arr = n_bad = 0
        bad = []
        for sd in (0, 1):
            zo = np.load(OLD / "arms" / "h2cn" / arm / f"DEPLOY8_POS_{TREE}_s{sd}.npz")
            zn = np.load(NEW / "arms" / "h2cn_cap8" / arm / f"DEPLOY8_POS_{TREE}_s{sd}.npz")
            if sorted(zo.files) != sorted(zn.files):
                n_bad += 1
                bad.append(f"s{sd}: key sets differ")
                continue
            for k in zo.files:
                n_arr += 1
                if k == "meta_json":
                    mo, mn = json.loads(str(zo[k])), json.loads(str(zn[k]))
                    mm = exact_equal(mo, mn)
                    if mm:
                        n_bad += 1
                        bad.append(f"s{sd}/meta_json: {mm[:3]}")
                elif zo[k].dtype != zn[k].dtype or zo[k].shape != zn[k].shape or zo[k].tobytes() != zn[k].tobytes():
                    n_bad += 1
                    bad.append(f"s{sd}/{k}")
        rec["blocks"][f"npz_package_{arm}"] = {"n_arrays": n_arr, "n_mismatches": n_bad, "first": bad[:6]}
        # (3) the TAC rows
        to, tn = jl(OLD / "arms" / "h2cn" / arm / f"TAC_{TREE}.json"), jl(NEW / "arms" / "h2cn_cap8" / arm / f"TAC_{TREE}.json")
        m = exact_equal(to["clause5"], tn["clause5"])
        co = {k: v for k, v in to["controls"].items() if not k.endswith("_detail") and k != "note"}
        cnn = {k: v for k, v in tn["controls"].items() if not k.endswith("_detail") and k != "note"}
        m2 = exact_equal(co, cnn)
        rec["blocks"][f"TAC_{arm}"] = {"clause5_n_leaves": leaves(to["clause5"]), "clause5_n_mismatches": len(m), "controls_n_mismatches": len(m2), "first": (m + m2)[:6]}
    # (4) the verdict: H2CN_DEPLOY8 fresh vs landed -- every clause (status, reason, evidence), per_clause, overall, beside, tier, estimator
    vo, vn = jl(OLD / "ship_r8" / "SHIP_R8_H2CN_DEPLOY8.json"), jl(NEW / "ship_r8" / "SHIP_R8_H2CN_DEPLOY8.json")
    vrows = {"overall": exact_equal(vo["overall"], vn["overall"]), "per_clause": exact_equal(vo["per_clause"], vn["per_clause"]),
             "tier": exact_equal(vo["tier"], vn["tier"]), "estimator": exact_equal(vo["estimator"], vn["estimator"]), "next_step": exact_equal(vo["next_step"], vn["next_step"])}
    for co_, cn_ in zip(vo["clauses"], vn["clauses"]):
        vrows[f"clause_{co_['id']}"] = exact_equal({"status": co_["status"], "reason": co_["reason"], "evidence": co_["evidence"]},
                                                   {"status": cn_["status"], "reason": cn_["reason"], "evidence": cn_["evidence"]}, skip=WALL_CLOCK)
    bo, bn_ = copy.deepcopy(vo["beside"]), copy.deepcopy(vn["beside"])
    bn_["compliance_row"]["cells"] = cn
    bo["compliance_row"] = {k: v for k, v in bo["compliance_row"].items() if k != "cells"} | {"cells": bo["compliance_row"]["cells"]}
    bn_["compliance_row"] = {k: v for k, v in bn_["compliance_row"].items() if k in bo["compliance_row"]}
    vrows["beside"] = exact_equal(bo, bn_)
    rec["blocks"]["verdict_H2CN_DEPLOY8"] = {k: {"n_mismatches": len(v), "first": v[:6]} for k, v in vrows.items()}
    rec["verdict_clauses_compared"] = [str(c["id"]) for c in vo["clauses"]]
    rec["exclusions"] = {"wall_clock_fields_skipped": list(WALL_CLOCK), "path_directory_token_normalised": "out/arms/h2cn == out_4b/arms/h2cn_cap8",
                         "cap8_only_additions_dropped": ["arm H2CN_CAP8", "per_window_extra[*].cap8", "compliance_row arm H2CN_CAP8", "controls K_CAP_* (and their names at the END of controls.required, checked explicitly)"],
                         "verdict_inputs_not_compared": "inputs / inputs_sha256 / generated / script_sha256 name this run's own files and time"}
    tot = 0
    for k, v in rec["blocks"].items():
        if k == "verdict_H2CN_DEPLOY8":
            tot += sum(x["n_mismatches"] for x in v.values())
        else:
            tot += sum(x for kk, x in v.items() if kk.endswith("n_mismatches"))
    rec["total_mismatches"] = int(tot)
    rec["REPRODUCED_EXACTLY"] = bool(tot == 0)
    return rec


def cell(c):
    return f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" + (" SEP" if c.get("separated") else "")


def summary() -> dict:
    fn = jl(NEW / "forward" / f"FORWARD_H2CN_CAP8_{TREE}.json")
    vc, vd = jl(NEW / "ship_r8" / "SHIP_R8_H2CN_CAP8.json"), jl(NEW / "ship_r8" / "SHIP_R8_H2CN_DEPLOY8.json")
    out = {"controls": {k: v for k, v in fn["controls"].items() if isinstance(v, bool)},
           "overall": {"H2CN_CAP8": vc["overall"], "H2CN_DEPLOY8_reread": vd["overall"]},
           "per_clause": {"H2CN_CAP8": vc["per_clause"], "H2CN_DEPLOY8_reread": vd["per_clause"]}, "cells": {}, "clamp_touched": {}, "compliance": {}}
    for nm, v in (("H2CN_CAP8", vc), ("H2CN_DEPLOY8", vd)):
        out["cells"][nm] = {"E1_S2": {s: cell(c) for s, c in v["clauses"][2]["evidence"]["cells"].items()}, "E3_S6": {s: cell(c) for s, c in v["clauses"][3]["evidence"]["cells"].items()}}
    for k, e in fn["per_window_extra"].items():
        out["clamp_touched"][k] = e["cap8"]
    for k, c in fn["compliance_row"]["cells"].items():
        for arm in ("os_model_plan", "H2CN_CAP8", "H2CN_DEPLOY8", "RETIME_DEPLOY8"):
            for g in ("panel_grid", "full_plan_0_6s"):
                for s in ("all_windows", "compliant_set"):
                    e = c["arms"][arm][g][s]
                    out["compliance"][f"{k}|{arm}|{g}|{s}"] = {"n_windows": e["n_windows"], "n_exceed_strict": e["n_exceed_strict"], "rate_mean": e["exceed_rate_strict"]["mean"],
                                                               "lo": e["exceed_rate_strict"]["lo"], "hi": e["exceed_rate_strict"]["hi"]}
    return out


def main() -> int:
    r = repro()
    (NEW / "REPRO_H2CN_DEPLOY8_vs_landed.json").write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
    chk = json.loads((NEW / "REPRO_H2CN_DEPLOY8_vs_landed.json").read_text(encoding="utf-8"))          # assert on the ARTIFACT
    print(f"REPRO total_mismatches={chk['total_mismatches']} REPRODUCED_EXACTLY={chk['REPRODUCED_EXACTLY']}")
    for k, v in chk["blocks"].items():
        print("  ", k, {kk: vv for kk, vv in v.items() if kk != "first"} if k != "verdict_H2CN_DEPLOY8" else {kk: vv["n_mismatches"] for kk, vv in v.items()})
    s = summary()
    (NEW / "SUMMARY_4B.json").write_text(json.dumps(s, indent=1, default=str), encoding="utf-8")
    print("OVERALL", s["overall"], "\nper_clause", s["per_clause"])
    for nm, c in s["cells"].items():
        print(nm, c)
    print("controls", {k: v for k, v in s["controls"].items() if k.startswith("K_") or k in ("ALL_PASS", "rt_forward")})
    for k, e in s["clamp_touched"].items():
        print("clamp", k, {kk: e[kk] for kk in ("n_windows", "n_changed_by_clamp", "share_changed_by_clamp", "n_compliant", "n_changed_by_clamp_compliant",
                                                "share_changed_by_clamp_compliant", "n_nonfinite_lim", "n_lim_le_0", "n_cast_positions_differ_from_H2CN_DEPLOY8")})
    for g in ("full_plan_0_6s", "panel_grid"):
        for sc in ("compliant_set", "all_windows"):
            for k in ("S2|s0", "S2|s1", "S6|s0", "S6|s1"):
                print("compliance", g, sc, k, {a: f"{s['compliance'][f'{k}|{a}|{g}|{sc}']['rate_mean']:.4f} ({s['compliance'][f'{k}|{a}|{g}|{sc}']['n_exceed_strict']}/"
                                                f"{s['compliance'][f'{k}|{a}|{g}|{sc}']['n_windows']})" for a in ("os_model_plan", "RETIME_DEPLOY8", "H2CN_DEPLOY8", "H2CN_CAP8")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
