#!/usr/bin/env python3
"""run_mutations_lf.py -- the mutation proof of the LF read package: each entry plants ONE exact-once change in a copy of a code file (reader / pre-check / read-out) and runs the tests that must catch it; the run must be RED
(a failing test, not an infrastructure error) while the same tests on the UNMUTATED file are GREEN (the clean control).  Groups run separately (they share no file): `reader`, `precheck`, `chain`; `--combine` merges the three
group logs into logs/mutations_lf.log, whose last line is `MUTATIONS ALL RED` only if every control is GREEN and every mutation RED.
usage: python -I run_mutations_lf.py reader|precheck|chain   |   python -I run_mutations_lf.py --combine"""
import os
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
SCR = "F:/TanitAD-scratch/motion_c/lf_read/pkgtest/mut"
GROUPS = {
    "reader": ("code/lf_read.py", "LF_READ_PATH", "tests/test_lf_read.py", [
        ("R1", "the reach bar 0.01 becomes 0.0", "DT, VMIN, BAND_LO, CRAWL_RATIO, REACH_BAR = 0.2, 5.2, 3.0, 0.5, 0.01", "DT, VMIN, BAND_LO, CRAWL_RATIO, REACH_BAR = 0.2, 5.2, 3.0, 0.5, 0.0", "reach_control_one_percent"),
        ("R2", "an unreached dose still reads its rules", '    if reach is not None and not reach["PASS"]:\n        res.update(', '    if False:\n        res.update(', "unreached_reads_no_rule"),
        ("R3", "LEAK-BORNE does not override the label", '    if leak["LEAK_BORNE"]:\n        return {"label": "LEAK-BORNE"', '    if False:\n        return {"label": "LEAK-BORNE"', "leak_borne_overrides"),
        ("R4", "the ADE6 / speed / along-track branch is skipped", "    if set(rows) & set(LONG_ROWS):\n        mid", "    if False:\n        mid", "outcome_labels_every_branch"),
        ("R5", "the headline omits a dominant dose", '    dom = [d for d in DOSES if d in lab and lab[d] == "D-HOLDS"]', "    dom = []", "headline_names_the_dose"),
        ("R6", "the bare arm copies --smooth off", 'BASE_SMOOTH = "sg72"', 'BASE_SMOOTH = "off"', "flags_copy_the_banked_base"),
        ("R7", "the M1 bind rate counts all ticks", '"bind_rate_in_m1_ticks": _rate(bind, m1)', '"bind_rate_in_m1_ticks": _rate(bind, allt)', "instruments_known_values"),
        ("R8", "M1's pair threshold 5.2 becomes 3.0", "DT, VMIN, BAND_LO, CRAWL_RATIO, REACH_BAR = 0.2, 5.2, 3.0, 0.5, 0.01", "DT, VMIN, BAND_LO, CRAWL_RATIO, REACH_BAR = 0.2, 3.0, 3.0, 0.5, 0.01", "m1_tick_definition or instruments_known"),
        ("R9", "reach needs only one seed", '"PASS": bool(len(per) == len(SEEDS) and all(p["reached"] for p in per.values()))}', '"PASS": bool(len(per) == len(SEEDS) and any(p["reached"] for p in per.values()))}', "reach_control_one_percent"),
        ("R10", "the contrast pairs RILF against C0", '_rows(ar, eid, cls, "RILF", "RIC0", n_boot, M1)', '_rows(ar, eid, cls, "RILF", "C0", n_boot, M1)', "ka_null_runs_green"),
        ("R11", "P is read from the logged scene", "    P, G = LR.rule_P(cells_ri), LR.rule_G(xy_ri, rows_ri)", "    P, G = LR.rule_P(cells_lg), LR.rule_G(xy_ri, rows_ri)", "ka_llr_runs_green"),
        ("R12", "the tool sha control is skipped", '"tool_sha256_is_patched_tool": bool(tool_pin and d["tool_sha256"] == tool_pin)', '"tool_sha256_is_patched_tool": bool(True)', "production_controls_pass"),
        ("R13", "the cap tick control is skipped", '"cap_only_where_presence_at_least_half": bool((pi[cap] >= 0.5 - 1e-6).all())', '"cap_only_where_presence_at_least_half": bool(True)', "tick_controls_catch"),
        ("R14", "the kappa bound tick control is skipped", '"v_f_between_0_and_kappa_v_hat": bool(((vf[fin] >= -1e-6) & (vf[fin] <= kappa * vh[fin] + 1e-3)).all())', '"v_f_between_0_and_kappa_v_hat": bool(True)', "tick_controls_catch"),
        ("R15", "ka-llr feeds the logged arrays as the reindexed role", '(("RILF", "RILLR"), ("RILFC", "RILLRC"), ("LF", "LLR"), ("LFC", "LLRC"))', '(("RILF", "LLR"), ("RILFC", "RILLRC"), ("LF", "LLR"), ("LFC", "LLRC"))', "ka_llr_runs_green"),
        ("R16", "the two-dose disclosure is dropped", 'DISCLOSURE = "Two doses were read', 'DISCLOSURE = "One dose was read', "headline_names_the_dose"),
    ]),
    "precheck": ("code/lf_precheck.py", "LF_PRECHECK_PATH", "tests/test_lf_chain_gates.py", [
        ("P1", "the kappa control is skipped", '"floor_kappa_is_the_dose": bool(kap_ok)', '"floor_kappa_is_the_dose": bool(True)', "rollout_check_refuses"),
        ("P2", "the LLR sha control is skipped", '"floor_ckpt_sha256_is_the_LLR_sha": fi["floor_ckpt_sha256"] == want_llr', '"floor_ckpt_sha256_is_the_LLR_sha": True', "rollout_check_refuses"),
        ("P3", "the banked checkpoint sha is not compared", "            if want is None or got != want:", "            if False:", "sha_mismatch_fails"),
        ("P4", "an absent checkpoint is not recorded", '                res["absent"].append(ck)', '                None', "absent_checkpoint"),
        ("P5", "the nested floor layout is ignored", "    if isinstance(lf, dict):\n        kap = lf.get", "    if False:\n        kap = lf.get", "floor_info_layouts"),
    ]),
    "chain": ("lf_read_chain.sh", "LF_CHAIN_PATH", "tests/test_lf_chain_gates.py", [
        ("C1", "the 075 rows carry kappa 0.5", '"RILF075_s0 0 reindexed off 075 0.75"', '"RILF075_s0 0 reindexed off 075 0.5"', "dry_run_lists"),
        ("C2", "the tree tool is not compared with the pin", 'got=$(sha256sum "$TOOL" | cut -c1-64); [ "$want" = "$got" ] || fail "tree tool sha', 'got=$(sha256sum "$TOOL" | cut -c1-64); true || fail "tree tool sha', "differs_from_the_pinned"),
        ("C3", "an absent build pins file is tolerated", '[ -f "$BUILD_PINS" ] || fail "lf_build/PINS.txt is absent: the patched tool\'s pin cannot be checked" 2', "true", "refuses_without_the_patched"),
        ("C4", "a missing patched_tool key is tolerated", 'want=$(bpin patched_tool); [ -n "$want" ] || fail "lf_build/PINS.txt carries no patched_tool key" 2', "want=$(bpin patched_tool)", "refuses_without_the_patched"),
    ]),
}


def pytest(test, env_key, path, sel):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1")
    if path:
        env[env_key] = path
    r = subprocess.run([PY, "-I", "-m", "pytest", f"{PK}/{test}", "-q", "-x", "-k", sel, "-p", "no:cacheprovider", "--basetemp", f"{SCR}/bt_{abs(hash((env_key, sel))) % 10**6}"], capture_output=True, text=True, env=env, cwd=PK)
    return r.returncode, r.stdout[-900:]


def run_group(g):
    src, env_key, test, muts = GROUPS[g]
    text = open(f"{PK}/{src}", encoding="utf-8", newline="").read()
    out = []
    for mid, what, old, new, sel in muts:
        n = text.count(old)
        if n != 1:
            out.append(f"{mid} INVALID: the old text matches {n} times ({what})")
            continue
        d = f"{SCR}/{g}_{mid}"
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        p = f"{d}/{os.path.basename(src)}"
        open(p, "w", encoding="utf-8", newline="").write(text.replace(old, new))
        rc_c, _ = pytest(test, env_key, None, sel)
        rc, tail = pytest(test, env_key, p, sel)
        infra = rc != 0 and not re.search(r"\d+ failed", tail)
        verdict = "RED" if (rc_c == 0 and rc != 0 and not infra) else ("INVALID (clean control not green)" if rc_c != 0 else ("INVALID (infrastructure: no failed test)" if infra else "STILL-GREEN"))
        out.append(f"{mid} {verdict}: {what} | tests -k '{sel}' | clean rc {rc_c}, mutated rc {rc} | {tail.strip().splitlines()[-1] if tail.strip() else ''}")
        print(out[-1], flush=True)
    open(f"{PK}/logs/mutations_lf_{g}.log", "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
    return out


def combine():
    lines = []
    for g in GROUPS:
        lines += [l for l in open(f"{PK}/logs/mutations_lf_{g}.log", encoding="utf-8").read().splitlines() if l.strip()]
    n = sum(len(v[3]) for v in GROUPS.values())
    ok = len(lines) == n and all(" RED:" in l for l in lines)
    open(f"{PK}/logs/mutations_lf.log", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + f"\n{'MUTATIONS ALL RED' if ok else 'MUTATIONS NOT ALL RED'}: {sum(' RED:' in l for l in lines)} of {n}\n")
    print(lines[-1] if lines else "", "|", "ALL RED" if ok else "NOT ALL RED")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--combine" in sys.argv:
        sys.exit(combine())
    run_group(sys.argv[1])
