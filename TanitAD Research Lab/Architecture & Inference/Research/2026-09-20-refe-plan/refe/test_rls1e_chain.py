#!/usr/bin/env python3
"""R0e-pick B5 (raw/2026-10-18-r0e-pick/PREREG_R0E_PICK.md section 10; launch blockers B2 + B5): the POD CHAIN's wiring of the e arms, RUN under bash (the chain's own text, extracted verbatim) and judged by the REAL launch gate.

`check(text)` returns the violations ([] = all hold) of refe/rls1_pod.sh against: eval/rls1e_arms.json and eval/rls1d_arms.json (the argv of R0e-id / R0e-pick / R1e-pick / R1o / R0d, token for token and by sha256), launch_gate.PROFILES
(the profile the chain's own arm_profile picks, per arm and per RLS1_R1O_LINEAGE; the chain's argv CLEAN under it and REFUSED under refe_rls1d), the R1o S gate (RLS1_R1O_LINEAGE=e gates on R0e-pick and R1e-pick, the default on R0d and R1d,
before any directory, preflight or GPU work; a refusal names the arms), the precheck (a misspelled lineage is refused; the files the e-lineage gate needs are required), and the reference each e arm is read against (its OWN arm A).
Then EVERY mutation of the chain text below must make `check` RED: "built, tested, unreachable from its caller" is the class this test exists for.   Prints ZZTEST_PASS <n>."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(HERE))
import gate_token as GT  # noqa: E402
import launch_gate as L  # noqa: E402

POD = HERE / "rls1_pod.sh"
E_ARMS = ("R0e-id", "R0e-pick", "R1e-pick")
BASH = shutil.which("bash")
N = 0


def posix(p):
    return str(p).replace("\\", "/")


def fn_body(text, name):
    m = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", text, re.S | re.M)
    return m.group(0) if m else ""


def bash(script, env=None, timeout=120):
    base = {k: val for k, val in os.environ.items() if k != "RLS1_R1O_LINEAGE"}                 # the lineage knob only when the caller gives it
    r = subprocess.run([BASH, "-c", script], capture_output=True, text=True, env=dict(base, **(env or {})), timeout=timeout)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def write_sh(path, text):
    Path(path).write_text(text, encoding="utf-8", newline="\n")
    os.chmod(path, 0o755)


def arm_args(text, tmp, arm):
    hdr = "\n".join(ln for ln in text.split("\n") if re.match(r"^[A-Z0-9_]+=", ln) and not ln.startswith("export") and "$(" not in ln and "FINAL_SHA256" not in ln and not ln.startswith("NSH"))
    sc = Path(tmp) / "aa.sh"
    write_sh(sc, hdr + "\n" + fn_body(text, "arm_args") + '\narm_args "$1"\n')
    rc, out, err = bash(f'bash "{posix(sc)}" {arm}')
    assert rc == 0, (arm, err)
    return out.split()


def arm_profile(text, tmp, arm, lineage=None):
    sc = Path(tmp) / "ap.sh"
    write_sh(sc, fn_body(text, "arm_profile") + '\narm_profile "$1"\n')
    rc, out, _ = bash(f'bash "{posix(sc)}" {arm}', {"RLS1_R1O_LINEAGE": lineage} if lineage is not None else None)
    return out if rc == 0 else None


def gate_block(text):
    """the arm loop's R1o S-gate block, verbatim: from `OBS=""` to the line before `mkdir -p "$ARMDIR" "$LG"`"""
    m = re.search(r'^  OBS=""\n.*?(?=^  mkdir -p "\$ARMDIR" "\$LG"\n)', text, re.S | re.M)
    return m.group(0) if m else ""


def run_gate_block(text, tmp, arm, lineage, verdict):
    """run the S-gate block for `arm` with a shim in place of rls1_checks.py: it records its argv and answers `verdict` (PASS | REFUSED).  -> (rc, stdout, the recorded gate calls, the OBS value)"""
    d = Path(tmp) / f"gb_{arm}_{lineage}_{verdict}"
    d.mkdir(parents=True, exist_ok=True)
    shim = d / "fakepy.sh"
    write_sh(shim, '#!/bin/bash\necho "$@" >> "' + posix(d / "calls.txt") + '"\n'
             'case "$*" in *safeobj-gate*) if [ "' + verdict + '" = PASS ]; then echo "ZZRLS1_SAFEOBJ_S_PASS ok"; else echo "ZZRLS1_SAFEOBJ_S_REFUSED the shim says no"; fi;; *id-gate*) echo "ZZRLS1_ID_PASS shim";; esac\n')
    sc = d / "gb.sh"
    write_sh(sc, f'set -u\nR="{posix(d)}"; S="$R/stage"; ARM="{arm}"; PY="{posix(shim)}"\nrefuse() {{ echo "REFUSED: $*"; exit 1; }}\n{gate_block(text)}\necho "OBS=[${{OBS:-}}]"\n')
    rc, out, err = bash(f'bash "{posix(sc)}"', {"RLS1_R1O_LINEAGE": lineage} if lineage is not None else None)
    calls = (d / "calls.txt").read_text().splitlines() if (d / "calls.txt").exists() else []
    return rc, out, calls


def check(text):
    global N
    v = []
    tmp = tempfile.mkdtemp(prefix="rls1e_chain_")
    arms_e = json.loads((PKG / "eval" / "rls1e_arms.json").read_text(encoding="utf-8"))["arms"]
    arms_d = json.loads((PKG / "eval" / "rls1d_arms.json").read_text(encoding="utf-8"))["arms"]

    def c(name, cond):
        global N
        N += 1
        if not cond:
            v.append(name)
    rc = subprocess.run([BASH, "-n", posix(POD)], capture_output=True) if text == POD.read_text(encoding="utf-8") else None
    if rc is not None:
        c("CHAIN_DOES_NOT_PARSE", rc.returncode == 0)
    m = re.search(r'^REGISTERED_ARMS="([^"]*)"', text, re.M)
    reg = m.group(1).split() if m else []
    c("E_ARMS_NOT_REGISTERED", all(a in reg for a in E_ARMS) and reg[:11] == ["C0", "C1", "R0", "R1", "R2", "R0b", "R1b", "R0c", "R0d", "R1d", "R1o"])
    # --- the argv, RUN under bash: the chain builds EXACTLY the registered argv of each arm, and the real gate accepts it under the profile the chain's own arm_profile picks
    got = {}
    for arm in E_ARMS + ("R1o", "R0d"):
        try:
            got[arm] = arm_args(text, tmp, arm)
        except AssertionError:
            got[arm] = None
    for arm in E_ARMS:
        ar = arms_e[arm]
        c(f"ARGV_{arm}_NOT_THE_ARMS_FILE", got[arm] == ar["argv"] and GT.argv_sha256(got[arm] or []) == ar["argv_sha256"] and got[arm] == L.rls1e_expected_argv(arm))
        sw_ = [t.replace(f"/arms/{arm}", "/arms/R0d") for t in (got[arm] or [])[:-2]]                  # PREREG 2.1: minus the lever, own paths swapped back, seed read as R0d's, the argv IS R0d's
        if "--seed" in sw_:
            sw_[sw_.index("--seed") + 1] = "0"
        c(f"ARGV_{arm}_NOT_R0D_PLUS_THE_LEVER", got["R0d"] is not None and sw_ == got["R0d"] and (got[arm] or [])[-2:] == ["--epo-con-temp", {"R0e-id": "0.1", "R0e-pick": "0.01", "R1e-pick": "0.01"}[arm]])
        prof = arm_profile(text, tmp, arm)
        c(f"PROFILE_{arm}_NOT_REFE_RLS1E", prof == "refe_rls1e")
        if got[arm] is not None and prof in L.PROFILES:
            c(f"ARGV_{arm}_REFUSED_BY_ITS_PROFILE", L.profile_refusals({**L.PROFILES[prof], "declared_arm": arm}, got[arm]) == [])
        if got[arm] is not None and arm != "R0e-id":                  # R0e-id's T 0.1 IS the inherited default: refe_rls1d reads it as R0d (the profile bound into its token is what tells them apart)
            c(f"ARGV_{arm}_ACCEPTED_BY_REFE_RLS1D", L.profile_refusals({**L.PROFILES["refe_rls1d"], "declared_arm": "R0d" if arm != "R1e-pick" else "R1d"}, got[arm]) != [])   # no older profile takes the lever
    c("ARGV_R1O_NOT_SAFEOBJS", got["R1o"] == arms_d["R1o"]["argv"] == arms_e["R1o"]["argv"] and "--epo-con-temp" not in (got["R1o"] or []))
    c("ARGV_R0D_NOT_SAFEOBJS", got["R0d"] == arms_d["R0d"]["argv"])
    for lin, want in ((None, "refe_rls1d"), ("d", "refe_rls1d"), ("e", "refe_rls1e")):
        c(f"PROFILE_R1O_LINEAGE_{lin}", arm_profile(text, tmp, "R1o", lin) == want)
        if got["R1o"] is not None:
            c(f"ARGV_R1O_REFUSED_UNDER_{want}", L.profile_refusals({**L.PROFILES[want], "declared_arm": "R1o"}, got["R1o"]) == [])
    c("PROFILE_R0D_MOVED", arm_profile(text, tmp, "R0d", "e") == "refe_rls1d")                  # the lineage knob moves R1o ALONE
    # --- the R1o S gate, RUN: the e lineage gates on R0e-pick and R1e-pick, the default on R0d and R1d; a refusal names its arms; no other arm is gated
    for lin, dirs, names in ((None, ("R0d", "R1d"), "R0d and R1d"), ("d", ("R0d", "R1d"), "R0d and R1d"), ("e", ("R0e-pick", "R1e-pick"), "R0e-pick and R1e-pick")):
        rc, out, calls = run_gate_block(text, tmp, "R1o", lin, "PASS")
        gate = [x for x in calls if "safeobj-gate" in x]
        c(f"R1O_GATE_PASS_{lin}", rc == 0 and "OBS=[--observe-only]" in out and len(gate) == 1 and re.search(rf"--dirs \S+/arms/{dirs[0]} \S+/arms/{dirs[1]}$", gate[0]) is not None)
        rc, out, calls = run_gate_block(text, tmp, "R1o", lin, "REFUSED")
        c(f"R1O_GATE_REFUSED_{lin}", rc == 1 and out.startswith("REFUSED: R1o") and names in out and len([x for x in calls if "safeobj-gate" in x]) == 1)
    for arm in E_ARMS + ("R0d",):
        rc, out, calls = run_gate_block(text, tmp, arm, "e", "REFUSED")
        # B9 (R0e-id: K3 observe-only) and B10 (R0e-pick: the ID gate, answered ID-PASS by the shim; its truth table is test_r0e_b9b10.py's) moved this pin: no arm but R1o has an S gate, OBS is set for R0e-id alone
        c(f"NO_S_GATE_FOR_{arm}", rc == 0 and not [x for x in calls if "safeobj-gate" in x] and (("OBS=[--observe-only]" if arm == "R0e-id" else "OBS=[]") in out) and len([x for x in calls if "id-gate" in x]) == (1 if arm == "R0e-pick" else 0) and (arm != "R0e-pick" or [x for x in calls if "id-gate" in x][0].endswith("--stage-dir " + posix(Path(tmp) / f"gb_{arm}_e_REFUSED") + "/stage")))
    gi, mk, pf = text.find('safeobj-gate --dirs "$R/arms/R0e-pick" "$R/arms/R1e-pick"'), text.find('mkdir -p "$ARMDIR" "$LG"'), text.find("scorer_finetune.py $ARGS --preflight")
    c("R1O_E_GATE_AFTER_THE_DIRECTORY_WORK", 0 < gi < mk and gi < pf)
    # --- the precheck: a misspelled lineage is refused (it must not default to d); the files the e-lineage gate needs on disk are required, and exist
    pc = re.search(r'^case "\$\{RLS1_R1O_LINEAGE:-d\}" in .*$', text, re.M)
    c("NO_LINEAGE_PRECHECK", pc is not None)
    if pc:
        sc = Path(tmp) / "lin.sh"
        write_sh(sc, 'set -u\nrefuse() { echo "REFUSED: $*"; exit 1; }\n' + pc.group(0) + '\necho ACCEPTED\n')
        for val, okv in (("d", True), ("e", True), ("E", False), ("x", False), ("de", False), ("0", False)):
            rc, out, _ = bash(f'bash "{posix(sc)}"', {"RLS1_R1O_LINEAGE": val})
            c(f"LINEAGE_PRECHECK_{val}", (rc == 0 and out == "ACCEPTED") if okv else (rc == 1 and "REFUSED: RLS1_R1O_LINEAGE" in out and "ACCEPTED" not in out))
        rc, out, _ = bash(f'bash "{posix(sc)}"')
        c("LINEAGE_PRECHECK_UNSET", rc == 0 and out == "ACCEPTED")
    lst = re.search(r'for f in (.*?); do\n  \[ -f', text, re.S)
    listed = lst.group(1).replace("\\\n", " ").split() if lst else []
    for f in ("refe/test_rl_pickcon.py", "eval/rls1d_arms.json", "eval/rls1e_arms.json"):
        c(f"PRECHECK_LACKS_{f}", f in listed)
    for e in L.PROFILES["refe_rls1e"]["suite"]:
        c(f"SUITE_FILE_NOT_REQUIRED_{e['script']}", f"refe/{e['script']}" in listed and (PKG / "refe" / e["script"]).is_file())
    # --- the reference: each e arm is read against its OWN arm A (K3 / K4 / the replay), the seed-matched C<n> is the second control
    own = 'case "$ARM" in R0e-id|R0e-pick|R1e-pick) REFDIR="$ARMDIR/scores";; esac'
    gen = text.find('if [ -z "$RD" ] ||')
    c("E_REFDIR_NOT_OWN_ARM_A", text.count(own) == 1 and text.find(own) > gen > 0)
    c("E_SECOND_CONTROL_NOT_NAMED", 0 < text.find('case "$ARM" in R0e-id|R0e-pick) RD=C0;; R1e-pick) RD=C1;; esac') < gen)
    return v


def mutations(text):
    sw = lambda a, b: (text.replace(a, b) if text.count(a) == 1 else None)       # noqa: E731
    reg = 'REGISTERED_ARMS="C0 C1 R0 R1 R2 R0b R1b R0c R0d R1d R1o R0e-id R0e-pick R1e-pick"'
    return [
        ("an e arm is not registered", sw(reg, 'REGISTERED_ARMS="C0 C1 R0 R1 R2 R0b R1b R0c R0d R1d R1o R0e-id R0e-pick"')),
        ("R0e-pick drops its lever", sw('--epo-con-lam0 0 --epo-con-temp 0.01" ;;\n    R1e-pick', '--epo-con-lam0 0" ;;\n    R1e-pick')),
        ("R0e-pick names another T", sw('--epo-con-lam0 0 --epo-con-temp 0.01" ;;\n    R1e-pick', '--epo-con-lam0 0 --epo-con-temp 0.02" ;;\n    R1e-pick')),
        ("R0e-id forgets to name the inherited default", sw(' --epo-con-temp 0.1" ;;', '" ;;')),
        ("R1e-pick runs seed 0", sw('R1e-pick) seed=1;', 'R1e-pick) seed=0;')),
        ("R1o carries the lever", sw('--epo-terms-heldout $TERMS_VA --epo-con off" ;;', '--epo-terms-heldout $TERMS_VA --epo-con off --epo-con-temp 0.1" ;;')),
        ("the e arms are judged under refe_rls1d", sw("R0e-id|R0e-pick|R1e-pick) echo refe_rls1e ;;", "R0e-id|R0e-pick|R1e-pick) echo refe_rls1d ;;")),
        ("the lineage knob is ignored by arm_profile", sw('[ "$1" = R1o ] && [ "${RLS1_R1O_LINEAGE:-d}" = e ] && { echo refe_rls1e; return 0; }', ':')),
        ("the e lineage gates on R0d and R1d", sw('safeobj-gate --dirs "$R/arms/R0e-pick" "$R/arms/R1e-pick"', 'safeobj-gate --dirs "$R/arms/R0d" "$R/arms/R1d"')),
        ("the e lineage gates on one arm only", sw('safeobj-gate --dirs "$R/arms/R0e-pick" "$R/arms/R1e-pick"', 'safeobj-gate --dirs "$R/arms/R0e-pick"')),
        ("the e lineage gate cannot refuse", sw('|| refuse "R1o (R0e-pick lineage) runs only if', '|| echo "R1o (R0e-pick lineage) runs only if')),
        ("a misspelled lineage is accepted", sw('case "${RLS1_R1O_LINEAGE:-d}" in d|e) ;; *) refuse', 'case "${RLS1_R1O_LINEAGE:-d}" in d|e|*) ;; *) refuse')),
        ("the precheck forgets test_rl_pickcon.py", sw("refe/test_rl_pickcon.py eval/rls1d_arms.json eval/rls1e_arms.json", "eval/rls1d_arms.json eval/rls1e_arms.json")),
        ("the precheck forgets the e arms file", sw("refe/test_rl_pickcon.py eval/rls1d_arms.json eval/rls1e_arms.json", "refe/test_rl_pickcon.py eval/rls1d_arms.json")),
        ("the e arms are read against C<n>'s dumps", sw('case "$ARM" in R0e-id|R0e-pick|R1e-pick) REFDIR="$ARMDIR/scores";; esac', ":")),
        ("the second control is left to ${ARM:1:1}", sw('case "$ARM" in R0e-id|R0e-pick) RD=C0;; R1e-pick) RD=C1;; esac', ":")),
    ]


def main():
    text = POD.read_text(encoding="utf-8")
    assert "\r" not in text
    v = check(text)
    assert v == [], v
    for name, mut in mutations(text):
        assert mut is not None and mut != text, f"mutation {name!r} does not apply to the chain text"
        got = check(mut)
        assert got, f"mutation {name!r} left every check green"
        print(f"  RED  {name}  ->  {len(got)} check(s): {got[0]}", flush=True)
    print(f"ZZTEST_PASS {N}  (1 unmutated chain + {len(mutations(text))} RED mutations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
