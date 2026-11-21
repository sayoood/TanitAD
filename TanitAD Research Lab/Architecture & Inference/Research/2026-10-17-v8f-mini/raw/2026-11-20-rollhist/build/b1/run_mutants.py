"""Mutation runner for ROLLHIST B1. For each mutant: apply ONE textual substitution to a file of build/tree, run the named test file(s) once, record which tests went RED, restore the file from
build/b1/save (byte for byte, sha256 checked). One pytest at a time. Writes b1/mutation_table_rollhist.json incrementally."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = "F:/TanitAD-scratch/rollhist/build"
TREE = ROOT + "/tree/stack"
OUT = ROOT + "/b1/mutation_table_rollhist.json"
SAVE = ROOT + "/b1/save"
os.makedirs(SAVE, exist_ok=True)

BANK = "scripts/v8f_dagger_bank.py"
MUT = [
    ("M1_no_readout_install", BANK, "    frames = RO.readout_frames(orig, gen[\"readout\"], gen[\"acc_gain\"], gen[\"path_exec\"])\n", "    frames = orig\n",
     "tests/test_v8f_rollhist.py", "R4: the line that installs the readout deleted"),
    ("M2_no_heading_bound", BANK, "    tool.hermite_frames = HB.bounded_frames(frames, HB.parse_kappa(gen[\"kappa_max\"]))\n", "    tool.hermite_frames = frames\n",
     "tests/test_v8f_rollhist.py", "R4: the line that installs the bound deleted"),
    ("M3_no_restore", BANK, "    finally:\n        tool.hermite_frames = orig\n", "    finally:\n        pass\n",
     "tests/test_v8f_rollhist.py", "N2: the executor is not restored after the run"),
    ("M4_default_installs_the_bound", BANK, "    tool.hermite_frames = HB.bounded_frames(frames, HB.parse_kappa(gen[\"kappa_max\"]))\n", "    tool.hermite_frames = HB.bounded_frames(frames, 0.2)\n",
     "tests/test_v8f_rollhist.py", "N2: the defaults no longer reproduce today's generator (the bound is always on)"),
    ("M5_nondefault_header_without_generator", BANK, "\"rollout\": G.ROLLOUT, **({} if gen_rec is None else {\"generator\": gen_rec}), \"constants\"", "\"rollout\": G.ROLLOUT, \"constants\"",
     "tests/test_v8f_rollhist.py", "N2/N3: a NON-default bank carries no generator record in its header"),
    ("M6_inert_param_accepted", BANK, "    if readout == \"pinned\" and (gain is not None or pex):\n        raise SystemExit(", "    if False:\n        raise SystemExit(",
     "tests/test_v8f_rollhist.py", "N2: a declared-and-inert --acc-gain / --path-exec is accepted"),
    ("M7_built_levers_ignore_header", "tanitad/data/v8f_dagger.py", "\"dagger_bank_readout\": readout_token(gen.get(\"readout\", \"pinned\"), gen.get(\"acc_gain\"), bool(gen.get(\"path_exec\", False))), \"dagger_bank_kappa\": str(gen.get(\"kappa_max\", \"none\")),",
     "\"dagger_bank_readout\": \"pinned\", \"dagger_bank_kappa\": \"none\",", "tests/test_v8f_rollhist.py", "N3: built_levers does not read the loaded bank header"),
    ("M8_no_declared_vs_header_clause", "scripts/v8f_mini_train.py", "        if lv.get(k) != declared[k]:\n            bad.append(f\"lever {k}: declared {declared[k]!r} but the loaded bank's header says",
     "        if False:\n            bad.append(f\"lever {k}: declared {declared[k]!r} but the loaded bank's header says", "tests/test_v8f_rollhist.py", "N4/R5: declared != loaded header is not refused"),
    ("M9_no_p0_clause", "scripts/v8f_mini_train.py", "        if a.dagger_p <= 0 and declared[k] != dfl[k]:\n", "        if False:\n", "tests/test_v8f_rollhist.py", "N4/R5: a declaration at --dagger-p 0 is not refused"),
    ("M10_no_flag_check_entry", "scripts/v8f_mini_train.py", "    \"dagger_bank_readout\": \"dagger:dagger_bank_readout\", \"dagger_bank_kappa\": \"dagger:dagger_bank_kappa\",\n", "",
     "tests/test_v8f_rollhist.py", "N4: the FLAG_CHECKS entries removed (G-DVB table)"),
    ("M11_readout_arithmetic", "scripts/v8f_t1_readout.py", "v[2 * (i - 1)] = s[i] - 0.5 * g * (s[i] - s[i - 1])", "v[2 * (i - 1)] = s[i] - 0.25 * g * (s[i] - s[i - 1])",
     "tests/test_v8f_t1_readout.py", "N1: the readout's arithmetic changed (study golden)"),
    ("M12_gain1_not_routed_through_causal", "scripts/v8f_t1_readout.py", "    if not force_general and g == 1.0:\n", "    if False:\n",
     "tests/test_v8f_t1_readout.py", "N1: gain 1.0 no longer routed through the formula of causal (identity control)"),
    ("M13_pair_check_drops_declaration", "scripts/rollhist_pair_check.py", "    for k, want in DECL.items():\n        if k not in a:", "    for k, want in {}.items():\n        if k not in a:",
     "tests/test_rollhist_pair_check.py", "chain checker: the declaration flags are no longer required in RH's argv"),
    ("M14_compare_readout_base_name_only", "scripts/v8f_mini_train.py", "        if lv.get(k) != declared[k]:\n            bad.append(f\"lever {k}: declared {declared[k]!r} but the loaded bank's header says",
     "        if (str(lv.get(k)).rstrip(\"x0123456789.\") if k == \"dagger_bank_readout\" else lv.get(k)) != (str(declared[k]).rstrip(\"x0123456789.\") if k == \"dagger_bank_readout\" else declared[k]):\n            bad.append(f\"lever {k}: declared {declared[k]!r} but the loaded bank's header says",
     "tests/test_v8f_rollhist.py", "N4 fix 2: only the readout BASE NAME is compared (gain and path-exec bound only through the sha)"),
]


import shutil as _sh
import time as _t
DELLOG = "F:/TanitAD-scratch/rollhist/DELETIONS.log"


def free_gb():
    return _sh.disk_usage("F:/").free / 2**30


def wait_disk():
    for _ in range(20):
        if free_gb() >= 4.0:
            return True
        _t.sleep(60)
    raise SystemExit("F: under 4 GB free for 20 min: not running")


def cleanup_tmp(run):
    p = ROOT + "/tmp/" + run
    if os.path.isdir(p):
        sz = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)
        _sh.rmtree(p, ignore_errors=True)
        open(DELLOG, "a", encoding="utf-8").write(f"{_t.strftime('%Y-%m-%dT%H:%MZ', _t.gmtime())} rollhist-fix-round deleted {p} ({sz} bytes) after its pytest run\n")



def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run_pytest(testfile, name):
    wait_disk()
    r = subprocess.run(["C:/Program Files/Git/usr/bin/bash.exe", ROOT + "/runpy.sh", "tree", "mut_" + name, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=no", testfile], capture_output=True, text=True, encoding="utf-8", errors="replace")
    txt = r.stdout + r.stderr
    cleanup_tmp("mut_" + name)
    red = sorted(set(re.findall(r"^FAILED (\S+)", txt, re.M)))
    m = re.findall(r"(\d+) (passed|failed|error)", txt)
    return r.returncode, red, txt[-600:]


def main():
    table = []
    only = set(sys.argv[1:])
    for name, f, old, new, tf, what in MUT:
        if only and name not in only:
            continue
        p = f"{TREE}/{f}"
        bak = f"{SAVE}/{name}.orig"
        shutil.copyfile(p, bak)
        s0 = sha(p)
        src = open(p, encoding="utf-8", newline="").read()
        n = src.count(old)
        rec = {"mutant": name, "file": "stack/" + f, "what": what, "pattern_found": n, "test_file": "stack/" + tf}
        if n != 1:
            rec["result"] = "NOT-APPLIED (pattern count %d)" % n
            table.append(rec)
            continue
        try:
            open(p, "w", encoding="utf-8", newline="").write(src.replace(old, new))
            rc, red, tail = run_pytest(tf, name)
        finally:
            shutil.copyfile(bak, p)
        rec.update({"rc": rc, "went_red": bool(red), "red_tests": [x.split("::")[-1] for x in red], "restored_sha_ok": sha(p) == s0, "tail": tail[-200:]})
        table.append(rec)
        json.dump(table, open(OUT, "w", encoding="utf-8"), indent=1)
        print(name, "RED" if red else "NOT RED", len(red), "restored", rec["restored_sha_ok"], flush=True)
    json.dump(table, open(OUT, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
