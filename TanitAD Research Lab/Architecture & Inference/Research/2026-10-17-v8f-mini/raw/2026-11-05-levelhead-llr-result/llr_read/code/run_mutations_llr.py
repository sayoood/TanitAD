#!/usr/bin/env python3
"""run_mutations_llr.py -- the mutation proof of the LLR read package: each entry plants ONE exact-once change in a copy of a code file (reader / pre-check / read-out) and runs the tests that must catch it; the run must be RED
(a failing test, not an infrastructure error) while the same tests on the UNMUTATED file are GREEN (the clean control).  Groups run separately (they share no file): `reader`, `precheck`, `readout`; `--combine` merges the three
group logs into logs/mutations_llr.log, whose last line is `MUTATIONS ALL RED` only if every control is GREEN and every mutation RED.
usage: python -I run_mutations_llr.py reader|precheck|readout   |   python -I run_mutations_llr.py --combine"""
import os
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
SCR = "F:/TanitAD-scratch/motion_c/llr_read/mut"
GROUPS = {
    "reader": ("code/llr_read.py", "LLR_READ_PATH", "tests/test_llr_read.py", [
        ("R1", "an attribution-dependent branch in decide()", '    else:\n        branch = "P-FAILS-FAMILY-EXHAUSTED"\n    if not P and not G:', '    elif n_attr_separated_better >= 1:\n        branch = "P-HOLDS-G-OR-D-FAILS"\n    else:\n        branch = "P-FAILS-FAMILY-EXHAUSTED"\n    if not P and not G:', "attribution_is_reported or decision_table"),
        ("R2", "the share bar 0.5 -> 0.4", "SHARE_BAR, LEAK_FRACTION = 0.5, 0.5", "SHARE_BAR, LEAK_FRACTION = 0.4, 0.5", "P_is_per_seed"),
        ("R3", "LEAK-BORNE no longer overrides the label", '    if leak:\n        return {"label": "LEAK-BORNE"', '    if False:\n        return {"label": "LEAK-BORNE"', "leak_borne_overrides"),
        ("R4", "P holds when ANY seed clears it", '"PASS": bool(len(per) == len(SEEDS) and all(p["holds"] for p in per.values())),', '"PASS": bool(len(per) == len(SEEDS) and any(p["holds"] for p in per.values())),', "P_is_per_seed"),
        ("R5", "the banked RICLR base is read from the wrong directory", '"CLR": CLR_ARR, "RICLR": CLR_ARR,', '"CLR": CLR_ARR, "RICLR": RI_ARR,', "reused_arrays_are_the_pinned"),
        ("R6", "the frozen wiring control reads add", 'fl.get("WIRING") == "token"', 'fl.get("WIRING") == "add"', "frozen_files_controls"),
        ("R7", "KA-NEW2 control ignores the three deciding cases", ' and all(ka["cases"][c]["PASS"] for c in "ABC")', '', "ka_new2_control"),
        ("R8", "an attribution-base rollout role comes back", '"RILLRC": ("reindexed", "kbox")}', '"RILLRC": ("reindexed", "kbox"), "RIRAPRP01": ("reindexed", "off")}', "roles_runs"),
        ("R9", "the frozen ruling file is not checked against FROZEN_SHA.txt", 'ctl["frozen_" + nm] = {"PASS": bool(want_f is not None and got_f == want_f)', 'ctl["frozen_" + nm] = {"PASS": bool(True)', "frozen_files_controls"),
        ("R10", "production keeps going without the pre-check record", 'bool(pc["PASS"] is True and same and base_ok)', 'bool(True)', "production_without_llr_rollouts"),
    ]),
    "precheck": ("code/llr_precheck.py", "LLR_PRECHECK_PATH", "tests/test_llr_chain_gates.py", [
        ("P1", "the three-lead-token audit always passes", '"added_exactly_the_three_lead_tokens": bool(set(added) == set(LEAD_NEW) and added.get("--lead-head") == "on")', '"added_exactly_the_three_lead_tokens": bool(True)', "argv_audit_is_exact"),
        ("P2", "marker sha not compared", "                elif ev != got:", "                elif False:", "marker_sha_that_differs"),
        ("P3", "cfg aux_cond not checked", "            if cfg.get(\"aux_cond\") != wiring:", "            if False:", "wiring_the_checkpoint"),
        ("P4", "token wiring tolerates e_level / missing e_aux keys", '            if wiring == "token" and (not ea or el or ee):', "            if False:", "without_the_lead_head"),
        ("P5", "recover_n not required", "        if not (isinstance(rn, (int, float)) and not isinstance(rn, bool) and rn > 0):", "        if False:", "recovery_evidence"),
        ("P6", "the base-checkpoint record is not compared", '            if clr_pc is not None and clr_pc["seeds"][k].get("ckpt_sha256") != gc:', "            if False:", "banked_arrays_came"),
        ("P7", "the gate token's commit is not compared", '                if tip is not None and b.get("commit") != tip:', "                if False:", "gate_token_bound"),
        ("P8", "an absent checkpoint exits 8 instead of 5", 'return 0 if res["PASS"] else (5 if absent else 8)', 'return 0 if res["PASS"] else 8', "absent_checkpoint or missing_marker"),
        ("P9", "a CLR base with the lead head is tolerated", '                if bk.get("level_head") != "on" or bk.get("lead_head", "off") == "on":', "                if False:", "clr_checkpoint_that_has_the_lead_head"),
        ("P10", "the frozen file's W_LEAD is not validated", '    if not float(d.get("W_LEAD", "nan")) > 0:', "    if False:", "missing_or_bad_frozen_file"),
    ]),
    "readout": ("code/llr_readout.py", "LLR_READOUT_PATH", "tests/test_llr_readout.py", [
        ("O1", "the presence set is not applied", '            x[:, 1] = st["pres"]', "            pass", "end_to_end"),
        ("O2", "the level shift lands on the wrong column", 'x[:, 0] = x[:, 0] + st["lvl"]', 'x[:, 1] = x[:, 1] + st["lvl"]', "end_to_end"),
        ("O3", "the reach control is not enforced", 'if not c["reach_control_big_e_aux_output_perturbation"]["mean_abs_plan_speed_change_ms"] > 0.0:', "if False:", "control_failures_name"),
        ("O4", "the read-out's own copy of A1's threshold changes", "SENS_HI, SENS_LO = CR.SENS_HI, CR.SENS_LO", "SENS_HI, SENS_LO = 0.6, 0.1", "labels_use"),
        ("O5", "the merge passes whatever the controls say", '    return 0 if not fails else 1\n\n\ndef main', '    return 0\n\n\ndef main', "merge_writes"),
        ("O6", "the gap set is not applied", '            x[:, 2] = st["gap"]', "            pass", "end_to_end"),
        ("O7", "the zero-shift control is not enforced", 'if c["zero_shift_change_max_abs"] != 0.0:', "if False:", "control_failures_name"),
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
    open(f"{PK}/logs/mutations_llr_{g}.log", "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
    return out


def combine():
    lines = []
    for g in GROUPS:
        lines += [l for l in open(f"{PK}/logs/mutations_llr_{g}.log", encoding="utf-8").read().splitlines() if l.strip()]
    n = sum(len(v[3]) for v in GROUPS.values())
    ok = len(lines) == n and all(" RED:" in l for l in lines)
    open(f"{PK}/logs/mutations_llr.log", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + f"\n{'MUTATIONS ALL RED' if ok else 'MUTATIONS NOT ALL RED'}: {sum(' RED:' in l for l in lines)} of {n}\n")
    print(lines[-1] if lines else "", "|", "ALL RED" if ok else "NOT ALL RED")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--combine" in sys.argv:
        sys.exit(combine())
    run_group(sys.argv[1])
