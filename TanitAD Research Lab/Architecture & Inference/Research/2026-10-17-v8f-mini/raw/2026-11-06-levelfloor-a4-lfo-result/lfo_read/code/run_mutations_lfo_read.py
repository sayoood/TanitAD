#!/usr/bin/env python3
"""run_mutations_lfo_read.py -- the mutation proof of the LFO read package: each entry plants ONE exact-once change in a copy of a code file (reader / pre-check / chain script) and runs the tests that must catch it; the run must be RED
(a failing test, not an infrastructure error) while the same tests on the UNMUTATED file are GREEN (the clean control). Groups run separately (they share no file): `reader`, `precheck`, `chain`; `--combine` merges the three group logs
into logs/mutations_lfo.log, whose last line is `MUTATIONS ALL RED` only if every control is GREEN and every mutation RED.
usage: python -I run_mutations_lfo_read.py reader|precheck|chain   |   python -I run_mutations_lfo_read.py --combine"""
import os
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
SCR = "F:/TanitAD-scratch/motion_c/lfo_read/pt/mut"
GROUPS = {
    "reader": ("code/lfo_read.py", "LFO_READ_PATH", "tests/test_lfo_read.py", [
        ("R1", "the reach control does not come first", '    if reach is not None and not reach["PASS"]:\n        return {"label": "F-UNREACHED"', '    if False:\n        return {"label": "F-UNREACHED"', "outcome_labels_every_branch"),
        ("R2", "P_O failing does not give CEILING-SHORT", '    if not P["PASS"]:\n        return {"label": "CEILING-SHORT"', '    if False:\n        return {"label": "CEILING-SHORT"', "outcome_labels_every_branch"),
        ("R3", "G_O failing does not give CEILING-HARMS", '    if not G["PASS"]:\n        rows = failing_rows(G)', '    if False:\n        rows = failing_rows(G)', "outcome_labels_every_branch"),
        ("R4", "CEILING-CLEARS does not signal the estimator levers", '"label": "CEILING-CLEARS", "estimator_levers_signalled": True', '"label": "CEILING-CLEARS", "estimator_levers_signalled": False', "outcome_labels_every_branch"),
        ("R5", "the deciding contrast pairs RILFO against C0", '_rows(ar, eid, cls, "RILFO", "RIC0", n_boot, M1)', '_rows(ar, eid, cls, "RILFO", "C0", n_boot, M1)', "ka_lf_runs_green or ka_null_runs_green"),
        ("R6", "the attribution is taken against RIC0 instead of RILF075", '_rows(ar, eid, cls, "RILFO", "RILF", n_boot)[0]', '_rows(ar, eid, cls, "RILFO", "RIC0", n_boot)[0]', "ka_lf_runs_green"),
        ("R7", "the ceiling of D is marked counted", '"counted": False, "note": "the CEILING of D', '"counted": True, "note": "the CEILING of D', "ka_lf_runs_green"),
        ("R8", "the logged-speed tick control is skipped", '"v_hat_is_the_logged_speed_of_the_dumped_gt": bool(both.any() and np.abs(vh[both] - v_log[both]).max() <= 1e-5 and np.array_equal(np.isnan(vh), ~ok)),', '"v_hat_is_the_logged_speed_of_the_dumped_gt": bool(True),', "tick_controls_catch"),
        ("R9", "the no-floor-where-no-route tick control is skipped", '"no_floor_where_no_complete_route": bool(np.isnan(vf[nr]).all() and not bind[nr].any()),', '"no_floor_where_no_complete_route": bool(True),', "tick_controls_catch"),
        ("R10", "the flags drop the logged source", ' --floor-ckpt {floor_ckpt(s)} --floor-vhat-source logged")', ' --floor-ckpt {floor_ckpt(s)}")', "rollouts_and_flags"),
        ("R11", "the failing rows ignore ADE6", '        if not p["ade6_within_guard"]:\n            rows.add("ADE6")', '        if False:\n            rows.add("ADE6")', "failing_rows"),
        ("R12", "the no-route count is inverted", '"n_no_route_ticks": int((~ok).sum())', '"n_no_route_ticks": int((ok).sum())', "oracle_block"),
        ("R13", "the tool sha control is skipped", '"tool_sha256_is_patched_tool": bool(tool_pin and d["tool_sha256"] == tool_pin)', '"tool_sha256_is_patched_tool": bool(True)', "production_controls"),
        ("R14", "ka-lf feeds the base arrays instead of the LF075 arrays", '(z["RILF"], z["RILFC"]) if mode == "ka-lf" else (z["RIC0"], z["RIC0C"])', '(z["RIC0"], z["RIC0C"]) if mode == "ka-lf" else (z["RIC0"], z["RIC0C"])', "ka_lf_runs_green"),
        ("R15", "the CEILING statement is dropped from the headline", 'CEILING_NOTE = ("CEILING: the level input', 'CEILING_NOTE = ("NOTE: the level input', "headline_carries"),
        ("R16", "the shortfall check uses (1 + kappa)", '(1.0 - kappa) * v_log[unc]', '(1.0 + kappa) * v_log[unc]', "oracle_block"),
        ("R17", "the dumped-input control accepts a logged source flag-less done marker", 'd["flags"] == expected_flags(role, s)', 'bool(True)', "production_controls"),
    ]),
    "precheck": ("code/lfo_precheck.py", "LFO_PRECHECK_PATH", "tests/test_lfo_chain_gates.py", [
        ("P1", "the kappa control is skipped", '"floor_kappa_is_0_75": bool(kap_ok)', '"floor_kappa_is_0_75": bool(True)', "rollout_check_refuses"),
        ("P2", "the CEILING label control is skipped", '"header_label_is_CEILING": hdr.get("label") == "CEILING"', '"header_label_is_CEILING": True', "rollout_check_refuses"),
        ("P3", "the logged source control is skipped", '"header_floor_vhat_source_is_logged": hdr.get("floor_vhat_source") == "logged"', '"header_floor_vhat_source_is_logged": True', "rollout_check_refuses"),
        ("P4", "the done marker's flags are not checked for the logged source", '"done_flags_name_the_logged_source": "--floor-vhat-source logged" in str(done.get("flags", ""))', '"done_flags_name_the_logged_source": True', "rollout_check_refuses"),
        ("P5", "an LF tag is accepted as an LFO tag", 'TAG_RE = re.compile(r"^RILFO075(C)?_s([01])$")', 'TAG_RE = re.compile(r"^RILFO?075(C)?_s([01])$")', "rollout_check_refuses"),
        ("P6", "the LLR sha control is skipped", '"floor_ckpt_sha256_is_the_LLR_sha": fi["floor_ckpt_sha256"] == want_llr', '"floor_ckpt_sha256_is_the_LLR_sha": True', "rollout_check_refuses"),
    ]),
    "chain": ("lfo_chain.sh", "LFO_CHAIN_PATH", "tests/test_lfo_chain_gates.py", [
        ("C1", "the s1 bare row carries seed 0", '"RILFO075_s1 1 reindexed off"', '"RILFO075_s1 0 reindexed off"', "dry_run_lists"),
        ("C2", "the tree tool is not compared with the pin", 'got=$(sha256sum "$TOOL" | cut -c1-64); [ "$want" = "$got" ] || fail "tree tool sha', 'got=$(sha256sum "$TOOL" | cut -c1-64); true || fail "tree tool sha', "differs_from_the_pinned"),
        ("C3", "an absent build pins file is tolerated", '[ -f "$BUILD_PINS" ] || fail "lfo_build/PINS.txt is absent: the LFO tool\'s pin cannot be checked" 2', "true", "refuses_without_the_patched"),
        ("C4", "a missing patched_tool key is tolerated", 'want=$(bpin patched_tool); [ -n "$want" ] || fail "lfo_build/PINS.txt carries no patched_tool key" 2', "want=$(bpin patched_tool)", "refuses_without_the_patched"),
        ("C5", "the operator is not compared with its pin", 'gotop=$(sha256sum "$OPR" | cut -c1-64); [ "$wantop" = "$gotop" ] || fail "tree level_floor.py sha', 'gotop=$(sha256sum "$OPR" | cut -c1-64); true || fail "tree level_floor.py sha', "operator_that_differs"),
        ("C6", "the EXIT trap leaves no fail marker", 'if [ ! -f "$PK/raw/LFO.done" ] && [ ! -f "$PK/raw/LFO.fail" ]; then echo', 'if false; then echo', "chain_that_dies"),
        ("C7", "the rollout argv does not name the logged source", '--floor-vhat-source logged "$@" < /dev/null', '"$@" < /dev/null', "names_the_logged_source"),
        ("C8", "a pinned file may differ", '  [ -n "$want" ] && [ "$want" = "$got" ] || fail "$f sha $got != PINS.txt ($want)" 2', '  true || fail "$f sha $got != PINS.txt ($want)" 2', "pinned_file_that_differs"),
    ]),
}


def pytest(test, env_key, path, sel):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1", TMP="F:/TanitAD-scratch/motion_c/lfo_read/pt", TEMP="F:/TanitAD-scratch/motion_c/lfo_read/pt")
    if path:
        env[env_key] = path
    r = subprocess.run([PY, "-I", "-m", "pytest", f"{PK}/{test}", "-q", "-x", "-k", sel, "-p", "no:cacheprovider", "--basetemp", f"{SCR}/bt_{abs(hash((env_key, sel, path))) % 10**8}"], capture_output=True, text=True, env=env, cwd=PK)
    return r.returncode, r.stdout[-900:]


def run_group(g):
    src, env_key, test, muts = GROUPS[g]
    text = open(f"{PK}/{src}", encoding="utf-8", newline="").read()
    out = []
    os.makedirs(SCR, exist_ok=True)
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
    open(f"{PK}/logs/mutations_lfo_{g}.log", "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
    return out


def combine():
    lines = []
    for g in GROUPS:
        lines += [l for l in open(f"{PK}/logs/mutations_lfo_{g}.log", encoding="utf-8").read().splitlines() if l.strip()]
    n = sum(len(v[3]) for v in GROUPS.values())
    ok = len(lines) == n and all(" RED:" in l for l in lines)
    open(f"{PK}/logs/mutations_lfo.log", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + f"\n{'MUTATIONS ALL RED' if ok else 'MUTATIONS NOT ALL RED'}: {sum(' RED:' in l for l in lines)} of {n}\n")
    print(lines[-1] if lines else "", "|", "ALL RED" if ok else "NOT ALL RED")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--combine" in sys.argv:
        sys.exit(combine())
    run_group(sys.argv[1])
