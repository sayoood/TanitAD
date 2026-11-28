"""ctx_verdict.py -- the outcome of CTXROLL (PREREG_V8F_PROOF_ADD17 + note N1) from the files of run_ctxroll.sh.  CPU, reads json only.
  python ctx_verdict.py --dir spatial/q2/ctx --arms G_s0,G_s1 [--beside W_s0] --out spatial/q2/ctx/VERDICT_CTXROLL.md
Per seed: CTX-RECOVERS if (i) minus (ii) of R3h is above 0 with the interval excluding 0; else CTX-HELPS if R3h minus R1 (paired) is; else CTX-FLAT.
The name needs both seeds; a mixed pair reads the weaker name (order RECOVERS > HELPS > FLAT)."""
import argparse
import json
import os
import re
import sys

ORDER = ["CTX-FLAT", "CTX-HELPS", "CTX-RECOVERS"]
NUM = r"([+-][0-9.]+) \[([+-][0-9.]+), ([+-][0-9.]+)\]"


def grab(path, key):
    for line in open(path, encoding="utf-8"):
        if key in line:
            m = re.search(NUM, line)
            if m:
                return [float(x) for x in m.groups()]
    raise SystemExit("REFUSED: %s not found in %s" % (key, path))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--beside", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d = a.dir
    arms = a.arms.split(",")
    beside = [x for x in a.beside.split(",") if x]
    rows, names, ctl = {}, {}, True
    for arm in arms + beside:
        r = json.load(open(os.path.join(d, "RESULT_CTXROLL_%s.json" % arm), encoding="utf-8"))
        ok = bool(r["controls_pass"])
        v = {}
        for var in ("R1", "R3h", "R3g"):
            vp = os.path.join(d, "VERDICT_RB_%s_%s.md" % (arm, var))
            txt = open(vp, encoding="utf-8").read()
            ok &= ("controls_pass: True" in txt)
            v[var] = {"roll": grab(vp, "(i) rolled-out state"), "copy": grab(vp, "(ii) present state"), "true": grab(vp, "(iii) true future state"),
                      "roll_minus_copy": grab(vp, "(i) minus (ii)")}
        pr = {k: grab(os.path.join(d, "PAIR_RB_%s_%s.md" % (arm, k)), "(i) rolled-out state") for k in ("R3h_minus_R1", "R3g_minus_R1", "R3h_minus_R3g")}
        ch = json.load(open(os.path.join(d, "CHANGE_CTX_%s.json" % arm), encoding="utf-8"))
        ok &= bool(ch["controls_pass"])
        ctl &= ok
        if v["R3h"]["roll_minus_copy"][1] > 0:
            name = "CTX-RECOVERS"
        elif pr["R3h_minus_R1"][1] > 0:
            name = "CTX-HELPS"
        else:
            name = "CTX-FLAT"
        names[arm] = name
        rows[arm] = {"controls": ok, "one": r["one_step_mse"], "lat": r["latent_mse_at_2s"], "v": v, "pairs": pr, "change": ch, "x": r["controls"], "ctx": r["context"]}
    final = ORDER[min(ORDER.index(names[x]) for x in arms)] if ctl else "NOT-NAMED (a control failed)"
    f3 = lambda t: "%+.3f [%+.3f, %+.3f]" % tuple(t)                                   # noqa: E731
    L = ["# VERDICT CTXROLL (PREREG_V8F_PROOF_ADD17, note ADD17_N1)", "",
         "MEASURED, tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).",
         "No training: the banked checkpoints, start frame 4 of each window, target frame 14 (+2.0 s). R1 = rollout from one state with context 1 (the banked rule); R3h = seeded with the true states of frames 2, 3, 4; R3g = from one state, context grown to 3 by its own predictions.", "",
         "## Controls (first)", ""]
    for arm in arms + beside:
        x = rows[arm]["x"]
        L.append("* %s: x1 banked rollout reproduced, max abs diff %.4f (tol %.3f); x2 %d windows matched, %d unmatched, frame 10 against banked true state %.4f; x3 seeded loop at context 1 against the stack %.1e (tol 1e-05); x4 %d overlap pairs, state %.4f, rollout %.4f; lead read t1 to t6 and change form k1 to k3: %s" % (
            arm, x["x1_reproduce_banked_rollout"]["max_abs_diff"], x["x1_reproduce_banked_rollout"]["tol"], x["x2_mapping"]["windows"], x["x2_mapping"]["unmatched"],
            x["x2_mapping"]["frame10_vs_banked_true_2s_max_abs_diff"], x["x3_seeded_loop_context_1_equals_R1"]["max_abs_diff"], x["x4_overlap"]["pairs"],
            x["x4_overlap"]["state_max_abs_diff"], x["x4_overlap"]["rollout_max_abs_diff"], "PASS" if rows[arm]["controls"] else "FAIL"))
    L += ["* ALL_PASS: %s" % ctl, "", "## Lead gap R2 at +2.0 s (r-b read, unchanged tool; episode-cluster bootstrap B 2,000)", "",
          "| arm | rollout | R2 of the rolled-out state | present state (copy) | true future state | rollout minus copy | minus R1 (paired) |", "|---|---|---|---|---|---|---|"]
    for arm in arms + beside:
        for var in ("R1", "R3h", "R3g"):
            v = rows[arm]["v"][var]
            pm = "-" if var == "R1" else f3(rows[arm]["pairs"][var + "_minus_R1"])
            L.append("| %s | %s | %s | %+.3f | %+.3f | %s | %s |" % (arm, var, f3(v["roll"]), v["copy"][0], v["true"][0], f3(v["roll_minus_copy"]), pm))
    L += ["", "R3h minus R3g (paired): " + "; ".join("%s %s" % (arm, f3(rows[arm]["pairs"]["R3h_minus_R3g"])) for arm in arms + beside) + ".", "",
          "## Outcome (card section 4)", ""]
    for arm in arms:
        L.append("* %s: %s" % (arm, names[arm]))
    L += ["* **Name (both seeds, a mixed pair reads the weaker name): %s**" % final]
    for arm in beside:
        L.append("* beside, %s: %s (one seed, names nothing)" % (arm, names[arm]))
    L += ["", "## Beside the name (names nothing)", "",
          "| arm | one-step error over the copy error, context 1 | context 3 | latent MSE at +2.0 s: copy | R1 | R3h | R3g | change skill R1 | change skill R3h | true future state |", "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in arms + beside:
        o, lt, ch = rows[arm]["one"], rows[arm]["lat"], rows[arm]["change"]["arms"]
        c = rows[arm]["ctx"]
        L.append("| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %s | %s | %s |" % (arm, o["ratio_context_1_over_copy"], o["ratio_context_%d_over_copy" % c], lt["copy"], lt["R1"], lt["R3h"], lt["R3g"],
                 f3([ch[arm + "_R1"]["rollout"]["skill"]] + ch[arm + "_R1"]["rollout"]["ci95"]), f3([ch[arm + "_R3h"]["rollout"]["skill"]] + ch[arm + "_R3h"]["rollout"]["ci95"]),
                 f3([ch[arm + "_R1"]["ceiling"]["skill"]] + ch[arm + "_R1"]["ceiling"]["ci95"])))
    L += ["", "A one-step ratio above 1 means the predictor is worse than the copy of the state one step (0.2 s) ahead.",
          "Limits: one checkpoint per arm and seed; the probe is fitted on true states and applied to the rolled-out ones; the start frame is 4, so the windows differ from the frame-0 read; 139 episodes.", ""]
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    json.dump({"names": names, "final": final, "controls_pass": ctl}, open(a.out[:-3] + ".json", "w", encoding="utf-8"), indent=1)
    print("\n".join(L))
    return 0 if ctl else 2


if __name__ == "__main__":
    sys.exit(main())
