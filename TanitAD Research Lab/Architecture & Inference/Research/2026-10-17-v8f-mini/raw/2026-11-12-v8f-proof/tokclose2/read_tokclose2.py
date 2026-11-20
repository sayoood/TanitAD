"""read_tokclose2.py -- reader of card PREREG_V8F_PROOF_ADD86 (TOKCLOSE2), tier T0. Written before any fit of the card has ended.

Two stages, run one after the other by the lead:
  controls  reads the pass flags of the scorer's controls t0 to t4 and judges c0 and c1 of the card. Prints and writes control lines only: no r, no error, no loss value.
  names     refuses unless the control file says that every control passed and is bound to the same scorer files by sha256; then writes the names of section 3 of the card
            and the table that the card reports with no name.

The scorer is the unchanged tool of card ADD81 (tokclose_fit.py score, with --allow-hp-override); this file adds nothing to its numbers.
"""
import argparse, hashlib, io, json, os, sys

CARD = "PREREG_V8F_PROOF_ADD86"
T_TAGS = ["T10_s0", "T10_s1", "T25_s0", "T25_s1"]
COPIED_TAGS = ["E10_s0", "E10_s1", "E25_s0", "E25_s1", "T10shuf_s0"]
SUFFIXES = [".pt", "_log.json", "_table.npz"]
HP_T_LONG = {"batch": 8, "lr": 1e-3, "clip_norm": 1.0, "max_evals": 24, "patience": 4}
REQUIRED_R = 0.9


def sha16(path):
    return hashlib.sha256(io.open(path, "rb").read()).hexdigest()[:16]


def load(path):
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def read_sha_list(path):
    """lines '<hash> [*]<name>' -> {name: hash}"""
    out = {}
    for line in io.open(path, encoding="utf-8").read().replace("\r", "").split("\n"):
        if not line.strip():
            continue
        h, name = line.split(None, 1)
        out[name.strip().lstrip("*")] = h
    return out


def control_c0(pod):
    want = sorted("tokclose_%s%s" % (t, s) for t in COPIED_TAGS for s in SUFFIXES)
    a = read_sha_list(os.path.join(pod, "c0_copies.sha256"))
    b = read_sha_list(os.path.join(pod, "c0_sources.sha256"))
    names_ok = sorted(a) == want and sorted(b) == want
    equal = names_ok and all(a[k] == b[k] and len(a[k]) == 64 for k in want)
    return {"n_copies": len(a), "n_sources": len(b), "names_ok": names_ok, "all_equal": bool(equal), "pass": bool(names_ok and equal and len(want) == 15)}


def hp_equal(hp, want):
    return isinstance(hp, dict) and set(hp) == set(want) and all(float(hp[k]) == float(want[k]) for k in want)


def control_c1(pod):
    rec, ok = {}, True
    for t in T_TAGS:
        L = load(os.path.join(pod, "fit_logs", "tokclose_%s_log.json" % t))
        good = bool(L.get("hp_override") is True and hp_equal(L.get("hp"), HP_T_LONG) and L.get("shuffled") is False)
        rec[t] = {"hp_override": L.get("hp_override"), "hp": L.get("hp"), "ok": good}
        ok = ok and good
    for t in COPIED_TAGS:
        L = load(os.path.join(pod, "fit_logs", "tokclose_%s_log.json" % t))
        good = bool(L.get("hp_override") is False)
        rec[t] = {"hp_override": L.get("hp_override"), "ok": good}
        ok = ok and good
    return {"fits": rec, "pass": bool(ok)}


def stage_controls(a):
    pod = a.pod_dir
    cfile = os.path.join(pod, "TOKCLOSE_CONTROLS.json")
    C = load(cfile)
    lines = {}
    for k in ("t0", "t1", "t2", "t3", "t4"):
        lines[k] = {"pass": bool(C["controls"][k]["pass"])}
    t2 = C["controls"]["t2"]["fits"]
    lines["t2"]["n_fits"] = len(t2)
    lines["t2"]["loss_judged_on"] = sorted(k for k, v in t2.items() if v.get("loss_judged"))
    lines["t2"]["planted_loss_judged"] = bool(t2.get("T10shuf_s0", {}).get("loss_judged"))
    lines["t2"]["note_N1_form"] = bool(lines["t2"]["pass"] and not lines["t2"]["planted_loss_judged"] and len(lines["t2"]["loss_judged_on"]) == 8)
    lines["t2"]["pass"] = bool(lines["t2"]["note_N1_form"])
    lines["scorer_controls_all_pass"] = {"pass": bool(C.get("controls_all_pass"))}
    lines["c0"] = control_c0(pod)
    lines["c1"] = control_c1(pod)
    n_pass = sum(1 for v in lines.values() if v["pass"])
    out = {"card": CARD, "stage": "controls", "controls": lines, "n_pass": n_pass, "n_total": len(lines), "all_pass": n_pass == len(lines),
           "bound": {"TOKCLOSE_CONTROLS.json": sha16(cfile), "TOKCLOSE_RESULT.json": sha16(os.path.join(pod, "TOKCLOSE_RESULT.json"))}}
    if os.path.exists(a.out):
        sys.exit("REFUSED: %s exists" % a.out)
    with io.open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    for k in sorted(lines):
        print("CONTROL %-26s %s" % (k, "PASS" if lines[k]["pass"] else "FAIL"))
    print("ZZ TOKCLOSE2-CONTROLS %d/%d all_pass %s" % (n_pass, len(lines), out["all_pass"]))
    return 0 if out["all_pass"] else 1


def stage_names(a):
    pod = a.pod_dir
    K = load(a.controls)
    if not K.get("all_pass"):
        sys.exit("REFUSED: the control file does not say that every control passed; no name is written")
    for name, h in K["bound"].items():
        if sha16(os.path.join(pod, name)) != h:
            sys.exit("REFUSED: %s is not the file the controls were read from" % name)
    R = load(os.path.join(pod, "TOKCLOSE_RESULT.json"))
    C = load(os.path.join(pod, "TOKCLOSE_CONTROLS.json"))
    if not R.get("controls_all_pass"):
        sys.exit("REFUSED: the scorer result does not carry controls_all_pass")
    R0 = load(a.add81_result)
    C0 = load(a.add81_controls)
    names, arms = [], {}
    for x in ("T10", "T25"):
        A, A0 = R["arms"][x], R0["arms"][x]
        mapped = [n.replace("TOKCLOSE-%s-" % x, "TOKCLOSE2-%sL-" % x) for n in R["names"][x]]
        names += mapped
        arms[x + "L"] = {"H": A["H"], "cell": A["cell"], "r_pooled": A["r_pooled"], "r_ci": A["r_ci"], "r_per_seed": A["r_per_seed"], "mae": A["mae"],
                         "mae_constant_zero": A["mae_constant_zero"], "mae_constant_mean": A["mae_constant_mean"], "cell_r_pooled": A["cell_r_pooled"], "cell_mae": A["cell_mae"],
                         "paired_diff_abs_err_x_minus_cell": A["paired_diff_abs_err_x_minus_cell"], "paired_diff_ci": A["paired_diff_ci"], "names": mapped,
                         "card_ADD81": {"r_pooled": A0["r_pooled"], "r_ci": A0["r_ci"], "r_per_seed": A0["r_per_seed"], "mae": A0["mae"],
                                        "paired_diff_abs_err_x_minus_cell": A0["paired_diff_abs_err_x_minus_cell"], "paired_diff_ci": A0["paired_diff_ci"]},
                         "cell_equal_to_card_ADD81": bool(abs(A["cell_r_pooled"] - A0["cell_r_pooled"]) < 1e-12 and abs(A["cell_mae"] - A0["cell_mae"]) < 1e-12)}
    t25_better = "TOKCLOSE2-T25L-BETTER-THAN-OUTPUTS" in names
    names.append("TOKCLOSE2-FULL-FIT-READS-MORE" if t25_better else "TOKCLOSE2-FULL-FIT-NO-GAIN")
    hi = [arms[x]["r_ci"][1] for x in ("T10L", "T25L")]
    lo25 = arms["T25L"]["r_ci"][0]
    if all(h < REQUIRED_R for h in hi):
        req = "TOKCLOSE2-POOLED-TOKENS-BELOW-THE-REQUIREMENT"
    elif lo25 >= REQUIRED_R:
        req = "TOKCLOSE2-POOLED-TOKENS-REACH-THE-REQUIREMENT"
    else:
        req = "TOKCLOSE2-REQUIREMENT-NOT-DECIDED"
    names.append(req)
    fits = {}
    for t in T_TAGS:
        L = load(os.path.join(pod, "fit_logs", "tokclose_%s_log.json" % t))
        n_evals = len(L.get("curve") or [])
        fits[t] = {"loss_before": L["loss_before"], "loss_at_stop": L["loss_at_stop"], "best_step": L["best_step"], "steps": L["steps"], "steps_per_eval": L["steps_per_eval"],
                   "stopped_by": L["stopped_by"], "n_evals": n_evals, "ended_at_the_cap": bool(L["stopped_by"] != "patience"), "seconds": L.get("seconds"),
                   "card_ADD81": {k: C0["controls"]["t2"]["fits"][t][k] for k in ("loss_before", "loss_at_stop")}}
    e_same = all(abs(R["arms"][x]["r_pooled"] - R0["arms"][x]["r_pooled"]) < 1e-12 and abs(R["arms"][x]["mae"] - R0["arms"][x]["mae"]) < 1e-12 for x in ("E10", "E25"))
    out = {"card": CARD, "tier": "T0", "evidence": "MEASURED", "names": names, "arms": arms, "fits": fits, "required_r": REQUIRED_R,
           "added_check_no_name": {"E_arms_equal_card_ADD81": bool(e_same), "t1_planted_r": C["controls"]["t1"]["r"]},
           "families": {"LONGITUDINAL": "not measured by this read (T0 diagnostic of what the token features carry)", "LATERAL": "not measured by this read",
                        "TACTICAL": "not measured by this read", "STRATEGIC": "not measured by this read"},
           "bound": {"controls_file": sha16(a.controls), "TOKCLOSE_RESULT.json": sha16(os.path.join(pod, "TOKCLOSE_RESULT.json"))}}
    if os.path.exists(a.out):
        sys.exit("REFUSED: %s exists" % a.out)
    with io.open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    for n in names:
        print("NAME", n)
    for x in ("T10L", "T25L"):
        A = arms[x]
        print("%s r %+.3f [%+.3f, %+.3f] per seed %s mae %.3f cell r %+.3f mae %.3f diff %+.3f [%+.3f, %+.3f] | card ADD81: r %+.3f [%+.3f, %+.3f] mae %.3f diff %+.3f" % (
            x, A["r_pooled"], A["r_ci"][0], A["r_ci"][1], ", ".join("%+.3f" % v for v in A["r_per_seed"]), A["mae"], A["cell_r_pooled"], A["cell_mae"],
            A["paired_diff_abs_err_x_minus_cell"], A["paired_diff_ci"][0], A["paired_diff_ci"][1], A["card_ADD81"]["r_pooled"], A["card_ADD81"]["r_ci"][0], A["card_ADD81"]["r_ci"][1],
            A["card_ADD81"]["mae"], A["card_ADD81"]["paired_diff_abs_err_x_minus_cell"]))
    for t in T_TAGS:
        F = fits[t]
        print("FIT %s loss %.4f -> %.4f (card ADD81 %.4f -> %.4f) best step %s of %s, %d evaluations, stopped by %s" % (
            t, F["loss_before"], F["loss_at_stop"], F["card_ADD81"]["loss_before"], F["card_ADD81"]["loss_at_stop"], F["best_step"], F["steps"], F["n_evals"], F["stopped_by"]))
    print("E arms equal to card ADD81:", e_same)
    print("ZZ TOKCLOSE2-NAMES DONE")
    return 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("controls"); c.add_argument("--pod-dir", required=True); c.add_argument("--out", required=True)
    n = sub.add_parser("names"); n.add_argument("--pod-dir", required=True); n.add_argument("--controls", required=True); n.add_argument("--add81-result", required=True)
    n.add_argument("--add81-controls", required=True); n.add_argument("--out", required=True)
    a = p.parse_args()
    sys.exit(stage_controls(a) if a.cmd == "controls" else stage_names(a))


if __name__ == "__main__":
    main()
