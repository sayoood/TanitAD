"""bars_4c.py -- SHIP-R8 AMENDMENT 4c section 3: bars L1 / L2 (+ L3 read from the SHIP-R8 verdict) and the mutation arm, on the SAME battery draw.
NEW file (EvalFlyWheel 2026-10-26).  CPU only, pure npz / json in, out_4c/BARS_4C.json + BARS_4C.md out.

MD / FD are computed EXACTLY as refcv9_l2c/code/h2cn_fdmd_diag.py does: E = l2_common.load_eval_pf(seed); comp = E['compliant'];
fd, md = l2_common.lon_indicators(S6_pos, E, comp) (false decel on GT-cruise, missed decel on GT-decel, NaN outside the population); cells =
pair_read_r8.paired_cell(new, ref, eid) (paired episode-cluster bootstrap B 2000, bootstrap seed 0, over EPISODES, models fixed).
  L1  H2CNL_DEPLOY8 - H2CN_DEPLOY8 MD separated BELOW 0 (CI hi < 0) at BOTH inference seeds
  L2  H2CNL_DEPLOY8 - H2CN_DEPLOY8 FD NOT separated above 0 (NOT CI lo > 0) at BOTH inference seeds
  L3  SHIP_R8_H2CNL_DEPLOY8.json OVERALL == PASS
Controls (a failed control VOIDS the bars):
  K-ALIGN    every arm package's S6 ws / clip == load_eval_pf's rows (the diag's own assert)
  K-H8-POS   out_4c H2CN_DEPLOY8 S6 positions bit-equal the LANDED out/arms/h2cn H2CN_DEPLOY8 package (same forward, same stage)
  K-DIAG     the H2CN_DEPLOY8 - RETIME_DEPLOY8 FD / MD cells reproduce H2CN_FDMD_DIAG.txt (every printed number, 4 dp, + n)
  K-ZERO     H2CN_DEPLOY8 - H2CN_DEPLOY8 reads delta 0, CI [0, 0] (an estimator that cannot read a known zero is not admissible)
  K-MUT-L1   the mutation arm H2CNL_MUT8 (logits deranged across FIT episodes before the fit, true eval logits) must NOT pass L1
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

W = Path("C:/Users/Admin/refcv8_pf/ship_r8_sub")
OUT = W / "out_4c"
TREE = "step48852ema_pf"
ARMS = OUT / "arms" / "h2cn_h2cnl"
LANDED_H8 = W / "out" / "arms" / "h2cn" / "H2CN_DEPLOY8"
DIAG_TXT = Path("C:/Users/Admin/refcv8_pf/refcv9_l2c/result/H2CN_FDMD_DIAG.txt")
DIAG_PY = Path("C:/Users/Admin/refcv8_pf/refcv9_l2c/code/h2cn_fdmd_diag.py")
L2_DIR = Path("C:/Users/Admin/refcv8_pf/refcv9_l2/code")
FWD_4C = OUT / "forward" / f"FORWARD_H2CN_H2CNL_{TREE}.json"
FWD_4B = W / "out_4b" / "forward" / f"FORWARD_H2CN_CAP8_{TREE}.json"
VERDICT = OUT / "ship_r8" / "SHIP_R8_H2CNL_DEPLOY8.json"
VERDICT_H8 = OUT / "ship_r8" / "SHIP_R8_H2CN_DEPLOY8.json"
AMD = W / "SHIP_R8_AMENDMENT_4C_H2CNL.md"
ARM_LIST = ("H2CNL_DEPLOY8", "H2CNL_MUT8", "H2CN_DEPLOY8", "RETIME_DEPLOY8")


def sha256_file(p) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def cell(PR, a, b, eid) -> dict:
    c = PR.paired_cell(a, b, eid)
    keep = ("delta", "lo", "hi", "separated", "n_windows", "n_episodes", "mean_new", "mean_ref", "n_dropped_nonfinite", "status")
    return {k: (float(c[k]) if isinstance(c.get(k), (float, np.floating)) else c.get(k)) for k in keep if k in c}


def fmt(c) -> str:
    return f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" + (" SEP" if (c["hi"] < 0 or c["lo"] > 0) else "") + f" ({c['mean_new']:.4f} vs {c['mean_ref']:.4f}, n={c['n_windows']})"


def parse_diag() -> dict:
    out = {}
    pat = re.compile(r"^s(\d) (\S+)\s+(FD|MD): ([0-9.]+) vs RETIME ([0-9.]+)\s+delta ([+-][0-9.]+) \[([+-][0-9.]+), ([+-][0-9.]+)\] n=(\d+)")
    for line in DIAG_TXT.read_text(encoding="utf-8").splitlines():
        m = pat.match(line)
        if m:
            out[(int(m.group(1)), m.group(2), m.group(3))] = {"mean_new": m.group(4), "mean_ref": m.group(5), "delta": m.group(6), "lo": m.group(7), "hi": m.group(8), "n": int(m.group(9))}
    return out


def main() -> int:
    t0 = time.time()
    sys.path.insert(0, str(L2_DIR))
    import l2_common as L2
    PR = L2.stack()["PR"]
    amd_sha = sha256_file(AMD)
    seal = Path(str(AMD).replace(".md", ".sha256")).read_text(encoding="utf-8").split()[0]
    if len(amd_sha) != 64 or amd_sha != seal:
        raise SystemExit(f"[bars_4c] amendment 4c does not match its seal ({amd_sha} vs {seal})")
    diag = parse_diag()
    res = {"tool": "bars_4c.py", "amendment_4c_sha256": amd_sha, "tier": "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)",
           "estimator": "pair_read_r8.paired_cell -> taniteval.ci.paired_episode_cluster_bootstrap, B 2000, bootstrap seed 0, over EPISODES with the models FIXED: answers "
                        "'another draw of episodes' only -- not another training run (one training seed, one stage fit), not another inference run (seeds 0 and 1 reported separately)",
           "population": "S6 surface, compliant windows (E['compliant']); FD on GT-cruise windows, MD on GT-decel windows (l2_common.lon_indicators, plan class from the LAST chord, "
                         "+-1.5 m/s vs v0)", "seeds": {}, "controls": {}}
    k_align, k_h8, k_zero, k_diag = True, True, True, {}
    for sd in (0, 1):
        E = L2.load_eval_pf(sd)
        comp = np.asarray(E["compliant"], bool)
        eid = list(E["eid"])
        ind = {}
        for arm in ARM_LIST:
            z = np.load(ARMS / arm / f"DEPLOY8_POS_{TREE}_s{sd}.npz")
            al = bool(np.array_equal(z["S6_ws"], E["ws"]) and np.array_equal(z["S6_clip"], E["clip"]))
            k_align &= al
            if not al:
                raise SystemExit(f"[bars_4c] row alignment failed for {arm} seed {sd}")
            fd, md, _gt, _pc = L2.lon_indicators(z["S6_pos"], E, comp)
            ind[arm] = {"FD": fd, "MD": md, "pos": z["S6_pos"]}
        zl = np.load(LANDED_H8 / f"DEPLOY8_POS_{TREE}_s{sd}.npz")
        h8eq = bool(zl["S6_pos"].dtype == ind["H2CN_DEPLOY8"]["pos"].dtype and zl["S6_pos"].tobytes() == ind["H2CN_DEPLOY8"]["pos"].tobytes())
        k_h8 &= h8eq
        cells = {}
        for k in ("MD", "FD"):
            cells[f"H2CNL_DEPLOY8-H2CN_DEPLOY8|{k}"] = cell(PR, ind["H2CNL_DEPLOY8"][k], ind["H2CN_DEPLOY8"][k], eid)
            cells[f"H2CNL_MUT8-H2CN_DEPLOY8|{k}"] = cell(PR, ind["H2CNL_MUT8"][k], ind["H2CN_DEPLOY8"][k], eid)
            cells[f"H2CNL_DEPLOY8-H2CNL_MUT8|{k}"] = cell(PR, ind["H2CNL_DEPLOY8"][k], ind["H2CNL_MUT8"][k], eid)
            for a in ("H2CNL_DEPLOY8", "H2CN_DEPLOY8", "H2CNL_MUT8"):
                cells[f"{a}-RETIME_DEPLOY8|{k}"] = cell(PR, ind[a][k], ind["RETIME_DEPLOY8"][k], eid)
            z0 = cell(PR, ind["H2CN_DEPLOY8"][k], ind["H2CN_DEPLOY8"][k], eid)
            cells[f"ZERO_H2CN_DEPLOY8-H2CN_DEPLOY8|{k}"] = z0
            k_zero &= bool(z0["delta"] == 0.0 and z0["lo"] == 0.0 and z0["hi"] == 0.0)
            # K-DIAG: the landed descriptive diag, every printed number
            ref = diag.get((sd, "H2CN_DEPLOY8", k))
            c = cells[f"H2CN_DEPLOY8-RETIME_DEPLOY8|{k}"]
            mine = {"mean_new": f"{c['mean_new']:.4f}", "mean_ref": f"{c['mean_ref']:.4f}", "delta": f"{c['delta']:+.4f}", "lo": f"{c['lo']:+.4f}", "hi": f"{c['hi']:+.4f}", "n": int(c["n_windows"])}
            k_diag[f"s{sd}|{k}"] = {"diag": ref, "mine": mine, "equal": bool(ref is not None and ref == mine)}
        l1 = cells["H2CNL_DEPLOY8-H2CN_DEPLOY8|MD"]
        l2 = cells["H2CNL_DEPLOY8-H2CN_DEPLOY8|FD"]
        m1 = cells["H2CNL_MUT8-H2CN_DEPLOY8|MD"]
        res["seeds"][str(sd)] = {"cells": cells, "n_compliant_s6": int(comp.sum()), "n_s6": int(len(comp)), "n_episodes": int(len(set(eid))),
                                 "L1_md_separated_below": bool(l1["hi"] < 0), "L2_fd_not_separated_above": bool(not (l2["lo"] > 0)),
                                 "MUT_md_separated_below": bool(m1["hi"] < 0), "H2CN_DEPLOY8_S6_pos_bit_equal_landed": h8eq,
                                 "n_windows_S6_positions_differ_H2CNL_vs_H2CN": int((ind["H2CNL_DEPLOY8"]["pos"] != ind["H2CN_DEPLOY8"]["pos"]).any((1, 2)).sum())}
    S = res["seeds"]
    L1 = bool(S["0"]["L1_md_separated_below"] and S["1"]["L1_md_separated_below"])
    L2b = bool(S["0"]["L2_fd_not_separated_above"] and S["1"]["L2_fd_not_separated_above"])
    MUT_L1 = bool(S["0"]["MUT_md_separated_below"] and S["1"]["MUT_md_separated_below"])
    v = json.loads(VERDICT.read_text(encoding="utf-8")) if VERDICT.exists() else None
    L3 = (v or {}).get("overall")
    ctl = {"K-ALIGN": bool(k_align), "K-H8-POS_bit_equal_landed": bool(k_h8), "K-DIAG_reproduces_H2CN_FDMD_DIAG": bool(k_diag and all(x["equal"] for x in k_diag.values())),
           "K-ZERO_known_zero": bool(k_zero), "K-MUT-L1_mutation_does_not_pass_L1": bool(not MUT_L1)}
    ctl["ALL_PASS"] = bool(all(ctl.values()))
    res["controls"] = ctl
    res["controls_detail"] = {"K-DIAG": k_diag, "diag_file": str(DIAG_TXT), "diag_file_sha256": sha256_file(DIAG_TXT), "diag_code_sha256": sha256_file(DIAG_PY)}
    res["bars"] = {"L1": {"pass": L1, "rule": "H2CNL_DEPLOY8 - H2CN_DEPLOY8 MD: CI hi < 0 at seed 0 AND seed 1",
                          "cells": {s: S[s]["cells"]["H2CNL_DEPLOY8-H2CN_DEPLOY8|MD"] for s in S}},
                   "L2": {"pass": L2b, "rule": "H2CNL_DEPLOY8 - H2CN_DEPLOY8 FD: NOT CI lo > 0 at seed 0 AND seed 1",
                          "cells": {s: S[s]["cells"]["H2CNL_DEPLOY8-H2CN_DEPLOY8|FD"] for s in S}},
                   "L3": {"pass": L3 == "PASS", "overall": L3, "per_clause": (v or {}).get("per_clause"), "verdict_file": str(VERDICT),
                          "verdict_sha256": sha256_file(VERDICT) if VERDICT.exists() else None},
                   "mutation_H2CNL_MUT8": {"passes_L1": MUT_L1, "must_not_pass_L1": True,
                                           "cells_MD": {s: S[s]["cells"]["H2CNL_MUT8-H2CN_DEPLOY8|MD"] for s in S},
                                           "cells_FD": {s: S[s]["cells"]["H2CNL_MUT8-H2CN_DEPLOY8|FD"] for s in S}}}
    instr_bad = [k for k in ("K-ALIGN", "K-H8-POS_bit_equal_landed", "K-DIAG_reproduces_H2CN_FDMD_DIAG", "K-ZERO_known_zero") if not ctl[k]]
    mech = f"read mechanically beside (NOT admissible as bars): L1 {'PASS' if L1 else 'FAIL'}, L2 {'PASS' if L2b else 'FAIL'}, L3 {L3}"
    if instr_bad:
        word = "VOID (instrument control failed: " + ", ".join(instr_bad) + ")"
        nxt = "fix the instrument and re-read (0 GPU); no bar is admissible while a control fails."
    elif not ctl["K-MUT-L1_mutation_does_not_pass_L1"]:
        # the instrument is sound; the ATTRIBUTION control failed: deranged-logit fits also move MD below H2CN_DEPLOY8, so L1 cannot be credited to the decision
        word = "VOID (K-MUT-L1: the mutation arm H2CNL_MUT8 also passes L1, so the MD gain is not attributable to the lon decision; Amendment 4c s2 voids the bars); " + mech
        nxt = ("H2CN_DEPLOY8 stands: L1 is void (not attributable)" + ("" if L2b else " and L2 fails on its own (FD separated ABOVE H2CN_DEPLOY8 at both seeds: the amendment's FAIL-L2 branch)") +
               (f"; L3 {L3}" if L3 != "PASS" else "") + ". The amendment's committed lever for an L1 that does not hold is a learned per-window gate between H2cn progress and "
               "the self-sharpened progress, fitted on the fresh split -- BLOCKED on one SELFSHARP GPU roll on the fresh clips and on its pre-registration (Master Mind).")
    elif L1 and L2b and L3 == "PASS":
        word = "PASS (L1, L2, L3)"
        nxt = ("the PI chooses the shipped stage among H2CN / H2CN-L (and their capped forms if Amendment 4b passes); recommendation follows the four-family rows. "
               "refcv9's design input: the progress head consumes the lon decision.")
    elif not L1:
        word = "FAIL L1" + ("" if L2b else " + L2") + ("" if L3 == "PASS" else f" + L3 ({L3})")
        nxt = ("the decision is not usable by this head class: next = a learned per-window gate between H2cn progress and the self-sharpened progress, fitted on the "
               "fresh split (needs a SELFSHARP roll on the fresh clips: one GPU roll), pre-registered first." +
               ("" if (L2b and L3 == "PASS") else " L2 / L3 also failed: H2CN_DEPLOY8 stands."))
    else:
        bad = (["L2"] if not L2b else []) + ([f"L3 (SHIP-R8 {L3}: clauses " + ", ".join(k for k, x in ((v or {}).get("per_clause") or {}).items() if x == "FAIL") + ")"] if L3 != "PASS" else [])
        word = "FAIL " + " + ".join(bad)
        nxt = "H2CN_DEPLOY8 stands; the failing bar/clause is named above."
    res["verdict"], res["next_step"] = word, nxt
    res["disclosed_deviations"] = (v or {}).get("disclosed_deviations")
    # beside: the compliance row (A-R8-CLAMP-2 s2.3) of H2CNL_DEPLOY8 vs H2CN_DEPLOY8 / H2CN_CAP8 (Amendment 4b forward) / RETIME_DEPLOY8 / os, both grids, both sets
    f4c = json.loads(FWD_4C.read_text(encoding="utf-8"))
    f4b = json.loads(FWD_4B.read_text(encoding="utf-8")) if FWD_4B.exists() else None
    comp_rows = {}
    for key, crow in f4c["compliance_row"]["cells"].items():
        for g in ("full_plan_0_6s", "panel_grid"):
            for sname in ("compliant_set", "all_windows"):
                row = {}
                for arm in ("os_model_plan", "RETIME_DEPLOY8", "H2CN_DEPLOY8", "H2CNL_DEPLOY8", "H2CNL_MUT8"):
                    e = crow["arms"][arm][g][sname]
                    row[arm] = {"rate": e["exceed_rate_strict"]["mean"], "lo": e["exceed_rate_strict"]["lo"], "hi": e["exceed_rate_strict"]["hi"],
                                "n_exceed": e["n_exceed_strict"], "n": e["n_windows"]}
                if f4b is not None:
                    e = f4b["compliance_row"]["cells"][key]["arms"]["H2CN_CAP8"][g][sname]
                    row["H2CN_CAP8 (4b forward)"] = {"rate": e["exceed_rate_strict"]["mean"], "lo": e["exceed_rate_strict"]["lo"], "hi": e["exceed_rate_strict"]["hi"],
                                                     "n_exceed": e["n_exceed_strict"], "n": e["n_windows"]}
                comp_rows[f"{key}|{g}|{sname}"] = row
    res["beside_compliance_row"] = comp_rows
    res["beside_E1_E3"] = {}
    for nm, p in (("H2CNL_DEPLOY8", VERDICT), ("H2CN_DEPLOY8", VERDICT_H8)):
        if p.exists():
            vv = json.loads(p.read_text(encoding="utf-8"))
            res["beside_E1_E3"][nm] = {"E1_S2": vv["clauses"][2]["evidence"].get("cells"), "E3_S6": vv["clauses"][3]["evidence"].get("cells")}
    res["inputs_sha256"] = {"forward_4c": sha256_file(FWD_4C), "forward_4b": sha256_file(FWD_4B) if FWD_4B.exists() else None,
                            **{f"pos_{a}_s{s}": sha256_file(ARMS / a / f"DEPLOY8_POS_{TREE}_s{s}.npz") for a in ARM_LIST for s in (0, 1)},
                            "l2_common.py": sha256_file(L2_DIR / "l2_common.py")}
    res["script_sha256"] = sha256_file(__file__)
    res["generated"], res["wall_s"] = time.strftime("%FT%T%z"), round(time.time() - t0, 1)
    jp = OUT / "BARS_4C.json"
    jp.write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    chk = json.loads(jp.read_text(encoding="utf-8"))                       # assert on the ARTIFACT
    assert chk["verdict"] == word and set(chk["bars"]) == {"L1", "L2", "L3", "mutation_H2CNL_MUT8"}
    Lm = [f"VERDICT: {word}", "", "# SHIP-R8 AMENDMENT 4c -- bars L1 / L2 / L3 (H2CNL_DEPLOY8 vs H2CN_DEPLOY8, same battery draw)", "",
          f"tier: {res['tier']}", f"estimator: {res['estimator']}", f"population: {res['population']}", "",
          "| cell (S6 compliant) | s0 | s1 |", "|---|---|---|"]
    for k in ("H2CNL_DEPLOY8-H2CN_DEPLOY8|MD", "H2CNL_DEPLOY8-H2CN_DEPLOY8|FD", "H2CNL_MUT8-H2CN_DEPLOY8|MD", "H2CNL_MUT8-H2CN_DEPLOY8|FD",
              "H2CNL_DEPLOY8-RETIME_DEPLOY8|MD", "H2CNL_DEPLOY8-RETIME_DEPLOY8|FD", "H2CN_DEPLOY8-RETIME_DEPLOY8|MD", "H2CN_DEPLOY8-RETIME_DEPLOY8|FD"):
        Lm.append(f"| {k} | {fmt(S['0']['cells'][k])} | {fmt(S['1']['cells'][k])} |")
    Lm += ["", f"L1 {L1} | L2 {L2b} | L3 {L3} | mutation passes L1: {MUT_L1}", f"controls: {json.dumps(ctl)}", "", f"NEXT (committed in the amendment): {nxt}", "", "ZZBARS4CENDZZ"]
    (OUT / "BARS_4C.md").write_text("\n".join(Lm), encoding="utf-8")
    print("\n".join(Lm), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
