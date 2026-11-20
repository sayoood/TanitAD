"""open_unguarded2.py -- card PREREG_V8F_PROOF_ADD89 (KINCLOSE2) after its read with controls k1 and k5 failed: opens ONLY what neither failed control guards.

k1 guards every arm that uses the ego path (K, E, B and their pairs and names); k5 guards the planted arms (PW, PS and the GAP-ERROR names). Not guarded by either, and resting on the passed
controls k0 and k3: the slopes of the reader's gap S_W, the slopes of the label gap BS_W, the cell L25, the pair S25-L25, the error of the reader's gap against the label gap, and the two
WINDOW names (rule of the card on BS_10 and BS_25). Everything else of the result file stays under its withheld key and is neither printed nor copied.
Refuses unless the control file says exactly: k0, k2, k3, k4 passed, k1 and k5 failed, and the result file is bound to it."""
import hashlib, io, json, os, sys

OPEN_ARMS = ["BS5", "BS10", "BS25", "BS50", "S5", "S10", "S25", "S50", "L25"]
OPEN_PAIRS = ["S25-L25"]
REQ_R = 0.9


def sha(p):
    return hashlib.sha256(io.open(p, "rb").read()).hexdigest()


def main():
    cpath, rpath, out = sys.argv[1], sys.argv[2], sys.argv[3]
    C, R = json.load(io.open(cpath, encoding="utf-8")), json.load(io.open(rpath, encoding="utf-8"))
    st = {k: bool(v["pass"]) for k, v in C["controls"].items()}
    if st != {"k0": True, "k1": False, "k2": True, "k3": True, "k4": True, "k5": False}:
        sys.exit("REFUSED: the control file is not the one this filter was written for: %s" % st)
    if R.get("controls_sha256") != sha(cpath) or R.get("names") != [] or "withheld" not in R or "arms" in R:
        sys.exit("REFUSED: the result file is not bound to the control file or is not in the withheld form")
    if os.path.exists(out):
        sys.exit("REFUSED: %s exists" % out)
    W = R["withheld"]
    arms = {n: W["arms"][n] for n in OPEN_ARMS}
    pairs = {n: W["paired"][n] for n in OPEN_PAIRS}
    names = []
    for w in (10, 25):
        b = arms["BS%d" % w]
        if b and b["r_ci"][0] >= REQ_R:
            names.append("KINCLOSE2-WINDOW-%d-REACHES-THE-REQUIREMENT" % w)
        elif b and b["r_ci"][1] < REQ_R:
            names.append("KINCLOSE2-WINDOW-%d-BELOW-THE-REQUIREMENT" % w)
        else:
            names.append("KINCLOSE2-WINDOW-%d-NOT-DECIDED" % w)
    res = {"card": R["card"], "tier": "T0", "evidence": "MEASURED", "controls": st, "opened": {"arms": OPEN_ARMS, "paired": OPEN_PAIRS, "other": ["gap_error"]},
           "not_opened": sorted(set(W["arms"]) - set(OPEN_ARMS)) + sorted("pair " + p for p in set(W["paired"]) - set(OPEN_PAIRS)),
           "names": names, "arms": arms, "paired": pairs, "gap_error": W["gap_error"], "bound": {"controls": sha(cpath)[:16], "result": sha(rpath)[:16]},
           "families": {"LONGITUDINAL": "not measured by this read (T0)", "LATERAL": "not measured by this read", "TACTICAL": "not measured by this read", "STRATEGIC": "not measured by this read"}}
    with io.open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    for n in names:
        print("NAME", n)
    for n in OPEN_ARMS:
        a = arms[n]
        print("%-5s r %+.3f [%+.3f, %+.3f] per seed %s mae %.3f err_sd %.3f lag1 %s n %d" % (n, a["r_pooled"], a["r_ci"][0], a["r_ci"][1], ", ".join("%+.3f" % x for x in a["r_per_seed"]), a["mae"],
              a["err_sd"], ", ".join("%+.3f" % x for x in a["err_lag1"]), a["n_frames"]))
    for n, p in pairs.items():
        print("pair %s diff %+.3f [%+.3f, %+.3f] n %d" % (n, p["diff_abs_err"], p["ci"][0], p["ci"][1], p["n_frames"]))
    for i, g in enumerate(W["gap_error"]):
        print("gap error of the reader against the label gap, seed index %d: n %d mean %+.2f m sd %.2f m mean abs %.2f m lag1 %+.3f" % (i, g["n"], g["mean_m"], g["sd_m"], g["mean_abs_m"], g["lag1"]))
    print("not opened: %d arms and %d pairs" % (len(set(W["arms"]) - set(OPEN_ARMS)), len(set(W["paired"]) - set(OPEN_PAIRS))))


if __name__ == "__main__":
    main()
