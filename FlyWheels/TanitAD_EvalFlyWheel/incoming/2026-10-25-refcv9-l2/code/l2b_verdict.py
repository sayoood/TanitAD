"""l2b_verdict.py -- combine the per-seed L2-B readings into ONE verdict per PREREG_L2B section 4 (seed 0 primary, seed 1 the replicate, never pooled).
    python l2b_verdict.py --root <result dir containing l2b_s0 [l2b_s1]> --out <dir>"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

NEXT = {"SELECTION": "TRAIN-fitted selector / score correction over the fan (the L1-type lever); the FD it can remove is the FB share with a cruise candidate (see floors)",
        "CANDIDATE": "fan generation / allocation / decoder speed-profile lever (no cruise-class candidate exists to select)",
        "MIXED": "both: rank the two levers by their window shares and by the FD each can remove (floors below); run the selector arm first if SELECTION share >= CANDIDATE share"}


def load(d):
    p = Path(d) / "L2B_RESULT.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--s0-dir", default=None)
    ap.add_argument("--s1-dir", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d0 = a.s0_dir or str(Path(a.root) / "l2b_s0")
    d1 = a.s1_dir or str(Path(a.root) / "l2b_s1")
    r0, r1 = load(d0), load(d1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if r0 is None:
        txt = "# L2-B VERDICT: PENDING (the seed-0 roll was not scored)\n"
        (out / "VERDICT.md").write_text(txt, encoding="utf-8")
        return 0
    lab0 = r0.get("label_stamped") or r0["label"]
    lab1 = (r1.get("label_stamped") or r1["label"]) if r1 else "PENDING (seed 1 not rolled / not scored)"
    inconclusive = str(lab0).startswith("INCONCLUSIVE")
    final = lab0
    if not inconclusive and r1 is not None and not str(lab1).startswith("INCONCLUSIVE") and lab1 != lab0:
        final = f"{lab0} (seed-unstable: seed 1 reads {lab1})"
    R = r0["readings"]
    L = [f"# L2-B VERDICT: {final}", "",
         f"PREREG_L2B sha256 {r0['prereg_sha256']}", f"tier: {r0['tier']}", f"estimator: {r0['estimator']}", "",
         f"* seed 0 (primary): **{lab0}** -- basis {r0['label_basis']}; R1 {r0['R1_label']}, R2 {r0['R2_label']}, existence saturated {r0['guard_existence_saturated']}",
         f"* seed 1 (replicate): **{lab1}**", "",
         "| set (seed 0) | n | EXIST target-class cand [CI] | GOOD (ADE<=tau) [CI] | pre-blend pick is target class |", "|---|---|---|---|---|"]
    for nm in ("FB", "CT", "DR", "MB"):
        x = R[nm]
        L.append(f"| {nm}{' (target decel)' if nm == 'MB' else ''} | {x['n']} | {x['EXIST']['mean']:.3f} [{x['EXIST']['lo']:.3f}, {x['EXIST']['hi']:.3f}] | "
                 f"{x['GOOD']['mean']:.3f} [{x['GOOD']['lo']:.3f}, {x['GOOD']['hi']:.3f}] | {x['BASE_PICK_is_target']['mean']:.3f} |")
    f = r0["FD"]
    L += ["", f"FD now {f['now']:.4f} (n GT-cruise {f['n_GT_cruise_compliant']}); floor if every SELECTION window were fixed: class-existence "
          f"{f['floor_if_every_SELECTION_window_fixed(class existence)']:.4f}, good-existence {f['floor_if_every_SELECTION_window_fixed(GOOD existence)']:.4f}",
          f"descriptive headroom: {json.dumps(r0['headroom_descriptive'], default=str)}", f"pre-blend (refcv6) pick class shares on FB (decel, cruise, accel): {r0['base_pick']}",
          f"controls hard pass (seed 0): {r0['controls_hard_pass']}; " + json.dumps({k: (v.get('pass') if isinstance(v, dict) else v) for k, v in r0["controls"].items()}), ""]
    key = lab0.split()[0] if not inconclusive else None
    L.append("**Next lever:** " + (NEXT.get(key, "") if key else "controls failed: fix the failing control before reading a label"))
    L.append("Not computed: " + "; ".join(f"{k}" for k in ("distance-keeping (no lead track; n=0)", "STRATEGIC (--no-strategic; n=0)")))
    (out / "VERDICT.md").write_text("\n".join(L), encoding="utf-8")
    print(f"ZZL2BVERDICTZZ {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
