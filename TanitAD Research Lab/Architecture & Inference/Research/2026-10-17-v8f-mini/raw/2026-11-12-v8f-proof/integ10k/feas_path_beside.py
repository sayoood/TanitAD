"""feas_path_beside.py -- card PREREG_V8F_PROOF_ADD56, section 4 ("Beside, fixed now"): AFTER the read of card ADD56, the pair results of the path-form runs of card ADD53
(banked on the pod read of that card, not opened until then) are printed next to those of the chord-form runs. Names nothing. json only, no numpy.
Refuses to run unless the read of card ADD56 exists (READ_FEASC10K.json): the path-form results are not opened before it.
Usage: python feas_path_beside.py <read_feas/pod> <read_feasc/pod> > FEAS_PATH_BESIDE.txt
Tier of every number: T1-kin (kinematic rollout on the model's own actions); never driving performance."""
import io
import json
import os
import sys

PAIRS = [("I10R0%s5", "I10R0M5", "SP no rule: limit at 1.0 s minus no limit at 1.0 s"), ("I10RLc%s5", "I10RLcM5", "SP label-fed"), ("I10RQ50%s5", "I10RQ50M5", "SP reader-fed"),
         ("I10R0%s5", "I10R0", "GC no rule: limit at 1.0 s minus banked 2.0 s"), ("I10RLc%s5", "I10RLc", "GC label-fed"), ("I10RQ50%s5", "I10RQ50", "GC reader-fed"),
         ("I10R0%s10", "I10R0", "PR: limit at 2.0 s minus banked 2.0 s"), ("I10RLc%s5", "I10R0%s5", "RU label-fed rule minus no rule, both limited, 1.0 s"),
         ("I10RQ50%s5", "I10R0%s5", "RU reader-fed"), ("I10R0%s5L", "I10R0%s5", "dose: limit 1.5 minus limit 3.0"), ("I10R0%s5L", "I10R0M5", "dose: limit 1.5 minus no limit")]
SPEED = "LONGITUDINAL:speed_mae_mps"


def jl(d, n):
    return json.load(io.open(os.path.join(d, n), encoding="utf-8"))


def main():
    path_dir, chord_dir = sys.argv[1], sys.argv[2]
    if not os.path.exists(os.path.join(chord_dir, "READ_FEASC10K.json")):
        raise SystemExit("REFUSED: the read of card ADD56 is not there; the path-form results stay unopened")
    src = {"A": (jl(path_dir, "RESULT_HEADWAY.json"), jl(path_dir, "RESULT_Q3.json")), "E": (jl(chord_dir, "RESULT_HEADWAY.json"), jl(chord_dir, "RESULT_Q3.json"))}
    print("Pair results of the limit in path form (runs A, card ADD53; that card named nothing) beside the limit in chord form (runs E, card ADD56). Names nothing.")
    print("TIER: T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance")
    for a, b, what in PAIRS:
        print("== %s" % what)
        for letter, form in (("A", "path "), ("E", "chord")):
            hw, q3 = src[letter]
            key = "%s_minus_%s" % (a.replace("%s", letter), b.replace("%s", letter))
            cells = hw["pairs"][key]["cells"]
            sp = q3["pairs"][key]["families"][SPEED]
            txt = "  ".join("%s %+.4f [%+.4f, %+.4f]" % (c, cells[c]["D"], cells[c]["A_lo"], cells[c]["A_hi"]) for c in ("U", "C", "F"))
            print("   %s %-28s %s | speed error %+.4f [%+.4f, %+.4f] m/s" % (form, key, txt, sp["D"], sp["A_lo"], sp["A_hi"]))
    print("ZZ FEAS-PATH-BESIDE DONE")


if __name__ == "__main__":
    main()
