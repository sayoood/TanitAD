"""read_gaprule_strat.py -- card PREREG_V8F_PROOF_ADD95 (GAPRULE10KB), an addition written after the names of the card were opened: the STRATEGIC family of the twelve pairs
(each planted arm against RR and against RLc), read as the reader of card ADD85 reads it (the engine's strategic_grid with its own controls, then the pair read at the tick t0 + 20 on
the arcs of 30, 50 and 80 m and the family reading). The reader of the kit carries the LONGITUDINAL, LATERAL and TACTICAL readings of the engine's pair read and not this one.
It gives no name of the card and changes none; it refuses unless the controls file of the card says that every control passed. CPU only. Refuses an existing output file.
Usage: python read_gaprule_strat.py --pull PULL --code CODE --grid-dir DIR --w W --tree TREE --controls CONTROLS.json --out NEW.json"""
import argparse
import io
import json
import os
import sys

import read_gaprule10kb2 as RG

R, EC = RG.R, RG.EC


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    for n in ("pull", "code", "grid-dir", "w", "tree", "controls", "out"):
        ap.add_argument("--" + n, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    a.n_boot_control = 300
    if os.path.exists(a.out):
        print("REFUSED: %s exists" % a.out)
        return 2
    cj = json.load(io.open(a.controls, encoding="utf-8"))
    if cj.get("card") not in (None, RG.CARD) or cj.get("controls_all_pass") is not True or any(v.get("PASS") is not True for v in cj.get("controls", {}).values()):
        print("REFUSED: the controls file of the card is not all-pass")
        return 3
    log = lambda s: print(s, flush=True)                                                       # noqa: E731
    E = R.engine(a)
    H = E[0]
    g30 = R.jl(os.path.join(a.grid_dir, "grid_p30.json"))
    J = H.load_join()
    keys = R.grid_keys(g30, [str(c) for c in J["clip"]])
    runs = dict(RG.RUN)
    runs.update(RG.BANKED)
    for r in runs.values():
        R.ALL_RUNS.setdefault(r, ("RLc", "K", 30, "deadreckon", "30", "plan", 5, 0.2))
    R.ALL_RUNS.setdefault(EC.SETTINGS["AGE"]["runs"]["RR"], ("RR", "K", 30, "deadreckon", "30", "plan", 5, 0.2))
    p_of = {"R0": 0}
    p_of.update({r: 30 for r in runs.values()})
    SR, cells_s, eid_s, S = R.strategic_grid(a, keys, sorted(p_of), p_of, log, {})
    ctl = {"PASS": bool(SR.get("stage_A_pass") and SR.get("stage_B_pass")), "detail": dict({k: v for k, v in SR["controls"].items() if k != "s3"}, s3=SR["controls"]["s3"].get("pass"))}
    out = {"card": RG.CARD, "tier": RG.TIER, "what": "addition after the names of the card: STRATEGIC family of the twelve pairs, no name", "controls_file_sha256": RG.sha_file(a.controls),
           "strategic_controls": ctl, "pairs": {}}
    print("s_strategic %s" % ("PASS" if ctl["PASS"] else "FAIL"))
    if ctl["PASS"]:
        for x in RG.ARMS:
            for ref in ("RR", "RLc"):
                A, B = RG.RUN[x], RG.BANKED[ref]
                pr = S.pair_read(cells_s, A, B, eid_s, ticks=(2,))
                pos = pr["tick2"]["pos"]
                out["pairs"][x + "-VS-" + ref] = {"A": A, "B": B, "pos_30_50_80m_tick_t0_plus_20": [{"arc_m": arc, "mean": b["mean"], "lo": b["lo"], "hi": b["hi"], "verdict": b["verdict"],
                                                                                                    "n_per_seed": b.get("n_per_seed")} for arc, b in zip(S.ARCS, pos)],
                                                  "reading": R.family_of([c["verdict"] for c in pos])}
    else:
        out["withheld"] = "the strategic controls of the engine did not pass: no strategic reading is written"
    io.open(a.out, "w", encoding="utf-8", newline="\n").write(json.dumps(out, indent=1, default=lambda o: float(o) if hasattr(o, "item") else str(o)))
    for k in sorted(out["pairs"]):
        print("STRATEGIC %s %s" % (k, out["pairs"][k]["reading"]))
    print("ZZ READ-GAPRULE-STRAT rc 0 strategic_controls %s" % ctl["PASS"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
