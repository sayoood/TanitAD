"""ssguard_s2_rows.py -- the S2 (0.5-2 s) four-family rows for SSGUARD GUARD-ALL and SELFSHARP vs BASE-os, from banked arrays (0 GPU, CPU).

PREREG_SSGUARD / _S1 state the S2 rows PENDING with no bar; this fills that family DESCRIPTIVELY (no bar is read here).  Machinery = l2_oracle_s2_rows.py
(load_panel / join / cells, the shipped stage on the full 8-slot plan sliced to S2 slots [0,1,2,3]); the guard = ssguard_score.guard on the 8-slot plans.

TIER: T1 self-action OPEN LOOP on logged eval139 frames.  ESTIMATOR: paired episode-cluster bootstrap, B 2000, seed 0, over EPISODES only.

    python ssguard_s2_rows.py --out <dir> --seed 0 [--bank-ss F:/refcv9_selfsharp]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ssguard_score as SG  # noqa: E402
import l2_oracle_s2_rows as O  # noqa: E402  (on sys.path via ssgate_score -> l2 code dir)

C, S = SG.C, SG.S


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bank-ss", default="F:/refcv9_selfsharp")
    a = ap.parse_args()
    t0 = time.time()
    sd = a.seed
    PR = C.stack()["PR"]
    rows = [r for r in S.metric_rows(PR) if r[1] in O.KEYS]
    P = O.load_panel(sd)
    base = O.join(S.load_l1_base(O.L1_BANK, sd), P)
    ss = O.join(S.load_trace(Path(a.bank_ss) / f"trace_selfsharp_s{sd}.npz"), P)
    gd = SG.guard(base, ss).astype(base.dtype)
    table = C.deploy_table()

    def stage4(p8):
        return C.stack()["RS"].retime_plan(np.asarray(p8), np.asarray(P["prior8"]), C.ALL_T, table, cast=True)[:, O.S2_SLOTS]
    mask = P["compliant"]
    Em = dict(P, compliant=mask)
    ctl = {"alignment": P["checks"], "K1_S2_stage_bit_equal_banked": bool(np.array_equal(stage4(P["plan8"]), P["stage_banked"])),
           "K2_S2_compliant_n": {"n": int(mask.sum()), "literal": 4159, "pass": int(mask.sum()) == 4159},
           "K-ID_guard_base_base_bit_equal": bool(np.array_equal(SG.guard(base, base).astype(base.dtype), base))}
    A = {nm: O.cells(Em, stage4(p), mask) for nm, p in (("BASE-os", base), ("SELFSHARP", ss), ("GUARD-ALL", gd))}
    echo = O.cells(Em, P["echo"], mask)
    res = {"tool": "ssguard_s2_rows.py", "seed": sd, "script_sha256": SG.G.sha(__file__), "tier": "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)",
           "estimator": "paired episode-cluster bootstrap (pair_read_r8.paired_cell), B 2000, seed 0, over EPISODES only; one training seed; one inference seed",
           "reading": "DESCRIPTIVE -- the preregs state S2 PENDING with no bar", "surface": {"name": "S2_compliant, stage on", "n_windows": int(mask.sum()),
                                                                                             "n_episodes": int(len(set(P["eid"][mask].tolist())))},
           "controls": ctl, "families_not_computed": {"distance-keeping": "no lead track; n = 0", "STRATEGIC": "--no-strategic; n = 0",
                                                      "false/missed-decel": "6 s horizon only (S6 rows)"},
           "GUARD-ALL_vs_BASE-os": S.compare(PR, Em, A["GUARD-ALL"], A["BASE-os"], rows),
           "SELFSHARP_vs_BASE-os": S.compare(PR, Em, A["SELFSHARP"], A["BASE-os"], rows),
           "E1_style_ADE_minus_echo_ha0_ext": {nm: {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in PR.paired_cell(A[nm]["ade_m"], echo["ade_m"], list(Em["eid"])).items()
                                                    if k in ("delta", "lo", "hi", "separated")} for nm in A}}
    res["GUARD-ALL_rows_separated_worse"] = sorted(k for k, c in res["GUARD-ALL_vs_BASE-os"].items() if c.get("worse") is True)
    res["wall_s"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    C.jdump(res, out / f"SSGUARD_S2_ROWS_s{sd}.json")
    fmt = lambda c: f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" if c and "delta" in c else "-"  # noqa: E731
    L = [f"# SSGUARD S2 rows, inference seed {sd} (DESCRIPTIVE; S2_compliant {res['surface']['n_windows']} w / {res['surface']['n_episodes']} e, stage on)", "",
         f"tier: {res['tier']}", f"estimator: {res['estimator']}", f"controls: {json.dumps({k: (v if not isinstance(v, dict) else v.get('pass', v)) for k, v in ctl.items() if k != 'alignment'})}", "",
         "| S2 row | GUARD-ALL - BASE-os | worse | SELFSHARP - BASE-os | worse |", "|---|---|---|---|---|"]
    for fam, k, lab, unit, lower in rows:
        g, s_ = res["GUARD-ALL_vs_BASE-os"].get(k), res["SELFSHARP_vs_BASE-os"].get(k)
        L.append(f"| {k} | {fmt(g)} | {g.get('worse') if g else '-'} | {fmt(s_)} | {s_.get('worse') if s_ else '-'} |")
    L += ["", "E1-style ADE(0-2 s) minus echo ha0_ext: " + "; ".join(f"{n} {fmt(v)}" for n, v in res["E1_style_ADE_minus_echo_ha0_ext"].items()),
          f"GUARD-ALL rows separated worse: {res['GUARD-ALL_rows_separated_worse'] or 'none'}", "", "ZZSSGS2ENDZZ"]
    (out / f"SSGUARD_S2_ROWS_s{sd}.md").write_text("\n".join(L), encoding="utf-8")
    print("worse:", res["GUARD-ALL_rows_separated_worse"], {k: (v if not isinstance(v, dict) else v.get("pass", v)) for k, v in ctl.items() if k != "alignment"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
