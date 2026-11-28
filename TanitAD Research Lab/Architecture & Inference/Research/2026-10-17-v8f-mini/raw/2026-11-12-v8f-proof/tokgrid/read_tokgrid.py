"""read_tokgrid.py -- reader of card PREREG_V8F_PROOF_ADD87 (TOKGRID), tier T0. Written before any fit of the card has ended.

Two stages, run one after the other by the lead:
  controls  judges g0 (the arm values of the score of the generated tool on the fits of card ADD86 equal, as numbers, those of the result file of card ADD86), the scorer's flags t0 to t4
            (t2 in the form of note N1 of card ADD81), c0 (the twelve files of the four E fits), c1 (the logs of the five T fits and of the four E fits), and the flag of the controls of card ADD86.
            Prints and writes control lines only: no r, no error, no loss value.
  names     refuses unless every control of this card passed and the files are bound by sha256; then writes the names of section 3 of the card and the table the card reports with no name.
            When a control of card ADD86 did not pass, the pair (finer grid against the card-ADD86 fit) has no name (section 4 of the card); the other names are written.

The scorer is the tool of card ADD81 as generated for this card (tokclose_fit_g8.py score, with --allow-hp-override); this file adds nothing to its numbers.
"""
import argparse, hashlib, io, json, math, os, sys

CARD = "PREREG_V8F_PROOF_ADD87"
T_TAGS = ["T10_s0", "T10_s1", "T25_s0", "T25_s1"]
PLANTED = "T10shuf_s0"
E_TAGS = ["E10_s0", "E10_s1", "E25_s0", "E25_s1"]
SUFFIXES = [".pt", "_log.json", "_table.npz"]
HP_T = {"batch": 8, "lr": 1e-3, "clip_norm": 1.0, "max_evals": 24, "patience": 4}
FINE_GRID, FINE_N_TOK = "pool8x20", 160
REQUIRED_R = 0.9
G0_RESULT = "g0_TOKCLOSE_RESULT.json"


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


def count_numbers(x):
    if isinstance(x, bool):
        return 0
    if isinstance(x, (int, float)):
        return 1
    if isinstance(x, dict):
        return sum(count_numbers(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(count_numbers(v) for v in x)
    return 0


def same_values(a, b):
    """exact equality of two json trees: same keys, same lengths, numbers equal as numbers (no tolerance), other leaves equal."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same_values(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same_values(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def control_g0(pod, add86_result):
    g0 = load(os.path.join(pod, G0_RESULT))
    r86 = load(add86_result)
    A, B = g0.get("arms"), r86.get("arms")
    ok = bool(isinstance(A, dict) and isinstance(B, dict) and A and same_values(A, B))
    return {"n_numeric_values": count_numbers(A) if isinstance(A, dict) else 0, "pass": ok}


def control_c0(pod):
    want = sorted("tokclose_%s%s" % (t, s) for t in E_TAGS for s in SUFFIXES)
    a = read_sha_list(os.path.join(pod, "c0_copies.sha256"))
    b = read_sha_list(os.path.join(pod, "c0_sources.sha256"))
    names_ok = sorted(a) == want and sorted(b) == want
    equal = names_ok and all(a[k] == b[k] and len(a[k]) == 64 for k in want)
    return {"n_copies": len(a), "n_sources": len(b), "names_ok": names_ok, "all_equal": bool(equal), "pass": bool(names_ok and equal and len(want) == 12)}


def hp_equal(hp, want):
    return isinstance(hp, dict) and set(hp) == set(want) and all(float(hp[k]) == float(want[k]) for k in want)


def control_c1(pod):
    rec, ok = {}, True
    for t in T_TAGS + [PLANTED]:
        L = load(os.path.join(pod, "fit_logs", "tokclose_%s_log.json" % t))
        dims = L.get("dims") or {}
        good = bool(L.get("grid") == FINE_GRID and dims.get("n_tok") == FINE_N_TOK and L.get("hp_override") is True and hp_equal(L.get("hp"), HP_T)
                    and L.get("shuffled") is (t == PLANTED))
        rec[t] = {"grid": L.get("grid"), "n_tok": dims.get("n_tok"), "hp_override": L.get("hp_override"), "hp": L.get("hp"), "ok": good}
        ok = ok and good
    for t in E_TAGS:
        L = load(os.path.join(pod, "fit_logs", "tokclose_%s_log.json" % t))
        good = bool(L.get("hp_override") is False)
        rec[t] = {"hp_override": L.get("hp_override"), "ok": good}
        ok = ok and good
    return {"fits": rec, "pass": bool(ok)}


def stage_controls(a):
    pod = a.pod_dir
    cfile = os.path.join(pod, "TOKCLOSE_CONTROLS.json")
    C = load(cfile)
    lines = {"g0": control_g0(pod, a.add86_result)}
    for k in ("t0", "t1", "t2", "t3", "t4"):
        lines[k] = {"pass": bool(C["controls"][k]["pass"])}
    t2 = C["controls"]["t2"]["fits"]
    lines["t2"]["n_fits"] = len(t2)
    lines["t2"]["loss_judged_on"] = sorted(k for k, v in t2.items() if v.get("loss_judged"))
    lines["t2"]["planted_loss_judged"] = bool(t2.get(PLANTED, {}).get("loss_judged"))
    lines["t2"]["note_N1_form"] = bool(lines["t2"]["pass"] and not lines["t2"]["planted_loss_judged"] and len(lines["t2"]["loss_judged_on"]) == 8)
    lines["t2"]["pass"] = bool(lines["t2"]["note_N1_form"])
    lines["scorer_controls_all_pass"] = {"pass": bool(C.get("controls_all_pass"))}
    lines["c0"] = control_c0(pod)
    lines["c1"] = control_c1(pod)
    lines["add86_controls_all_pass"] = {"pass": bool(load(a.add86_controls).get("all_pass") is True)}
    card_lines = {k: v for k, v in lines.items() if k != "add86_controls_all_pass"}
    n_pass = sum(1 for v in lines.values() if v["pass"])
    all_pass = all(v["pass"] for v in card_lines.values())
    out = {"card": CARD, "stage": "controls", "controls": lines, "n_pass": n_pass, "n_total": len(lines), "all_pass": bool(all_pass),
           "add86_controls_all_pass": lines["add86_controls_all_pass"]["pass"],
           "bound": {"TOKCLOSE_CONTROLS.json": sha16(cfile), "TOKCLOSE_RESULT.json": sha16(os.path.join(pod, "TOKCLOSE_RESULT.json")), G0_RESULT: sha16(os.path.join(pod, G0_RESULT)),
                     "PAIR_TOKGRID.json": sha16(os.path.join(pod, "PAIR_TOKGRID.json"))},
           "bound_outside": {"add86_result": sha16(a.add86_result), "add86_controls": sha16(a.add86_controls)}}
    if os.path.exists(a.out):
        sys.exit("REFUSED: %s exists" % a.out)
    with io.open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    for k in sorted(lines):
        print("CONTROL %-26s %s" % (k, "PASS" if lines[k]["pass"] else "FAIL"))
    print("ZZ TOKGRID-CONTROLS %d/%d all_pass %s add86_controls_all_pass %s" % (n_pass, len(lines), out["all_pass"], out["add86_controls_all_pass"]))
    return 0 if all_pass else 1


def finer_grid_name(p):
    lo, hi = float(p["ci"][0]), float(p["ci"][1])
    if not (math.isfinite(lo) and math.isfinite(hi)):
        sys.exit("REFUSED: the interval of the pair is not finite")
    if hi < 0:
        return "TOKGRID-FINER-GRID-READS-MORE"
    if lo > 0:
        return "TOKGRID-FINER-GRID-READS-LESS"
    return "TOKGRID-FINER-GRID-NOT-SEPARATED"


def requirement_name(hi_both, lo25):
    if all(h < REQUIRED_R for h in hi_both):
        return "TOKGRID-BELOW-THE-REQUIREMENT"
    if lo25 >= REQUIRED_R:
        return "TOKGRID-REACHES-THE-REQUIREMENT"
    return "TOKGRID-REQUIREMENT-NOT-DECIDED"


def stage_names(a):
    pod = a.pod_dir
    K = load(a.controls)
    if not K.get("all_pass"):
        sys.exit("REFUSED: the control file does not say that every control passed; no name is written")
    for name, h in K["bound"].items():
        if sha16(os.path.join(pod, name)) != h:
            sys.exit("REFUSED: %s is not the file the controls were read from" % name)
    for key, path in (("add86_result", a.add86_result), ("add86_controls", a.add86_controls)):
        if sha16(path) != K["bound_outside"][key]:
            sys.exit("REFUSED: %s is not the file the controls were read from" % path)
    R = load(os.path.join(pod, "TOKCLOSE_RESULT.json"))
    if not R.get("controls_all_pass"):
        sys.exit("REFUSED: the scorer result does not carry controls_all_pass")
    R86 = load(a.add86_result)
    P = load(os.path.join(pod, "PAIR_TOKGRID.json"))
    names, arms = [], {}
    for x in ("T10", "T25"):
        A = R["arms"][x]
        mapped = [n.replace("TOKCLOSE-%s-" % x, "TOKGRID-%sG-" % x) for n in R["names"][x]]
        names += mapped
        arms[x + "G"] = {"H": A["H"], "cell": A["cell"], "r_pooled": A["r_pooled"], "r_ci": A["r_ci"], "r_per_seed": A["r_per_seed"], "mae": A["mae"],
                         "mae_constant_zero": A["mae_constant_zero"], "mae_constant_mean": A["mae_constant_mean"], "cell_r_pooled": A["cell_r_pooled"], "cell_mae": A["cell_mae"],
                         "paired_diff_abs_err_x_minus_cell": A["paired_diff_abs_err_x_minus_cell"], "paired_diff_ci": A["paired_diff_ci"], "names": mapped}
        if x in R86.get("arms", {}):
            B = R86["arms"][x]
            arms[x + "G"]["card_ADD86"] = {"r_pooled": B["r_pooled"], "r_ci": B["r_ci"], "r_per_seed": B["r_per_seed"], "mae": B["mae"]}
    add86_ok = bool(K.get("add86_controls_all_pass"))
    pair_name = finer_grid_name(P["H25"]) if add86_ok else None
    if pair_name:
        names.append(pair_name)
    req = requirement_name([arms[x]["r_ci"][1] for x in ("T10G", "T25G")], arms["T25G"]["r_ci"][0])
    names.append(req)
    fits = {}
    for t in T_TAGS:
        L = load(os.path.join(pod, "fit_logs", "tokclose_%s_log.json" % t))
        fits[t] = {"loss_before": L["loss_before"], "loss_at_stop": L["loss_at_stop"], "best_step": L["best_step"], "steps": L["steps"], "steps_per_eval": L.get("steps_per_eval"),
                   "stopped_by": L["stopped_by"], "n_evals": len(L.get("curve") or []), "seconds": L.get("seconds")}
    out = {"card": CARD, "tier": "T0", "evidence": "MEASURED", "names": names, "arms": arms, "fits": fits, "required_r": REQUIRED_R,
           "pair_fine_minus_coarse": {"named": pair_name is not None, "name": pair_name, "H25": P["H25"], "H10_no_name": P["H10"]},
           "T25G_error_per_seed": P.get("T25_fine_error_per_seed"),
           "added_check_no_name": {"t1_planted_r": load(os.path.join(pod, "TOKCLOSE_CONTROLS.json"))["controls"]["t1"]["r"]},
           "families": {"LONGITUDINAL": "not measured by this read (T0 diagnostic of what the token features carry)", "LATERAL": "not measured by this read",
                        "TACTICAL": "not measured by this read", "STRATEGIC": "not measured by this read"},
           "bound": {"controls_file": sha16(a.controls), "TOKCLOSE_RESULT.json": sha16(os.path.join(pod, "TOKCLOSE_RESULT.json")), "PAIR_TOKGRID.json": sha16(os.path.join(pod, "PAIR_TOKGRID.json"))}}
    if os.path.exists(a.out):
        sys.exit("REFUSED: %s exists" % a.out)
    with io.open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    for n in names:
        print("NAME", n)
    if pair_name is None:
        print("NOTE the controls of card ADD86 did not all pass: the pair has no name")
    for x in ("T10G", "T25G"):
        A = arms[x]
        print("%s r %+.3f [%+.3f, %+.3f] per seed %s mae %.3f cell r %+.3f mae %.3f diff %+.3f [%+.3f, %+.3f]" % (
            x, A["r_pooled"], A["r_ci"][0], A["r_ci"][1], ", ".join("%+.3f" % v for v in A["r_per_seed"]), A["mae"], A["cell_r_pooled"], A["cell_mae"],
            A["paired_diff_abs_err_x_minus_cell"], A["paired_diff_ci"][0], A["paired_diff_ci"][1]))
    for h in ("H25", "H10"):
        q = P[h]
        print("PAIR %s fine minus coarse %+.4f [%+.4f, %+.4f]" % (h, q["paired_diff_abs_err_fine_minus_coarse"], q["ci"][0], q["ci"][1]))
    for t in T_TAGS:
        F = fits[t]
        print("FIT %s loss %.4f -> %.4f best step %s of %s, %d evaluations, stopped by %s" % (t, F["loss_before"], F["loss_at_stop"], F["best_step"], F["steps"], F["n_evals"], F["stopped_by"]))
    for k, v in sorted((P.get("T25_fine_error_per_seed") or {}).items()):
        print("T25G seed %s error sd %.4f lag-1 autocorrelation %+.3f (%d pairs)" % (k, v["sd_error"], v["lag1_autocorr"], v["n_pairs"]))
    print("ZZ TOKGRID-NAMES DONE")
    return 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("controls")
    c.add_argument("--pod-dir", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--add86-result", required=True)
    c.add_argument("--add86-controls", required=True)
    n = sub.add_parser("names")
    n.add_argument("--pod-dir", required=True)
    n.add_argument("--controls", required=True)
    n.add_argument("--add86-result", required=True)
    n.add_argument("--add86-controls", required=True)
    n.add_argument("--out", required=True)
    a = p.parse_args()
    sys.exit(stage_controls(a) if a.cmd == "controls" else stage_names(a))


if __name__ == "__main__":
    main()
