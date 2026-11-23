"""make_mutants_s3.py -- every mutant of the SEED3 kit must turn its test RED. For each (file, old, new, test): copy the kit folder to a scratch copy, apply the ONE edit (the old text must occur exactly once), run the test from the copy;
the mutant is RED (good) when the test exits non-zero. The unmutated copy must pass (control). usage: python make_mutants_s3.py <scratch dir> <bash exe> <python exe> [only-substring]"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCR, BASH, PYX = (os.path.abspath(a) for a in sys.argv[1:4])
ONLY = sys.argv[4] if len(sys.argv) > 4 else ""
PY = PYX.replace("\\", "/")
T = {"argv": ["test_s3_argv.py"], "gate": ["test_s3_gate_record.py"], "bank": ["test_s3_bank_check.py"],
     "chain": ["-I", "test_chain_dry_s3.py", "@K", "@S", BASH.replace("\\", "/"), PY], "marker": ["@BASH", "test_s3_marker_watch.sh", "@S"]}
M = [
    ("argv", "s3_argv.py", 'want = {"--seed": ["2"]', 'want = {"--seed": ["1"]'),
    ("argv", "s3_argv.py", "if diff - allowed - app:", "if False:"),
    ("argv", "s3_argv.py", "if want and not fsha(p).startswith(want):", "if False:"),
    ("argv", "s3_argv.py", '        new = setf(new, "--init-ckpt", UE2)\n', ""),
    ("argv", "s3_argv.py", '            new = setf(new, "--dagger-bank-sha256", sha)\n', ""),
    ("argv", "s3_argv.py", "new = new + [BASE_FLAG, BASE_VAL]", 'new = new + [BASE_FLAG, "own"]'),
    ("argv", "s3_argv.py", "if any(i not in iv for i in pos):", "if False:"),
    ("argv", "s3_argv.py", "if dups(new):", "if False:"),
    ("argv", "s3_argv.py", 'if arm != "KI_s2" and BASE_FLAG in a:', "if False:"),
    ("argv", "s3_argv.py", "if any(k in a for k in LEVER_FLAGS):", "if False:"),
    ("argv", "s3_argv.py", 'if arm != "B0_s2" and a.get("--dagger-p") != ["0.1"]:', "if False:"),
    ("argv", "s3_argv.py", "if len(new) != len(src) + n_add:", "if False:"),
    ("argv", "s3_argv.py", 'new = setf(new, "--dagger-seed", seed)', 'new = setf(new, "--dagger-seed", "0")'),
    ("gate", "rh_gate_record_s3.py", '        bad += bank_s3_problems(F["bank_s3"])\n', ""),
    ("gate", "rh_gate_record_s3.py", 'if gs.get("n_regressions") or gs.get("new_test_failures") or gs.get("missing_in_new"):\n        bad.append("G-SUITE reads', 'if False:\n        bad.append("G-SUITE reads'),
    ("gate", "rh_gate_record_s3.py", 'if gs.get("new") != COMMIT and gs.get("new_commit") != COMMIT:\n        bad.append(f"G-SUITE new commit is {gs.get(\'new_commit\', gs.get(\'new\'))!r}, not {COMMIT[:8]}")\n    if gs.get("n_regressions")',
     'if False:\n        bad.append(f"G-SUITE new commit is {gs.get(\'new_commit\', gs.get(\'new\'))!r}, not {COMMIT[:8]}")\n    if gs.get("n_regressions")'),
    ("gate", "rh_gate_record_s3.py", 'if "ZZ ROLLHIST-PAIR PASS" not in open(F["pair_log"]).read():', "if False:"),
    ("gate", "rh_gate_record_s3.py", "    if tag not in TAGS:\n        print(f\"PROBLEM rh_gate_record_s3.py serves", "    if False:\n        print(f\"PROBLEM rh_gate_record_s3.py serves"),
    ("gate", "rh_gate_record_s3.py", 'or a.tag not in TAGS or rec.get("tag") != a.tag:', ":"),
    ("gate", "rh_gate_record_s3.py", 'if pl.get("pass") is not True or pl.get("readout") != ro:', "if False:"),
    ("gate", "rh_gate_record_s3.py", 'if (sc.get("dagger_probe") or {}).get("pass") is not True:', "if False:"),
    ("gate", "rh_gate_record_s3.py", 'if o.get("generator_ckpt_sha256") != UE2_SHA:', "if False:"),
    ("gate", "rh_gate_record_s3.py", "if not isinstance(cov, (int, float)) or isinstance(cov, bool) or cov < 0.5:", "if False:"),
    ("gate", "rh_gate_record_s3.py", 'if o.get("partial") is not False:', "if False:"),
    ("gate", "ki_gate_record_s3.py", '    bad += bank_s3_problems(F["bank_s3"])\n', ""),
    ("gate", "ki_gate_record_s3.py", "if set(bs) != {BASE}:", "if False:"),
    ("gate", "ki_gate_record_s3.py", "if len(base_rows) != 1 or base_rows[0] != 0 or isinstance(base_rows[0], bool):", "if False:"),
    ("gate", "ki_gate_record_s3.py", "if not ns or max(ns) <= 0:", "if False:"),
    ("gate", "ki_gate_record_s3.py", "if not z or max(z) != 0.0:", "if False:"),
    ("gate", "ki_gate_record_s3.py", "if ll != {\"human\"}:", "if False:"),
    ("gate", "ki_gate_record_s3.py", "if lb != {BASE}:", "if False:"),
    ("gate", "ki_gate_record_s3.py", "    if m.get(\"--dagger-base\") != BASE:", "    if False:"),
    ("gate", "ki_gate_record_s3.py", "if f in m:\n            bad.append(f\"KI argv names", "if False:\n            bad.append(f\"KI argv names"),
    ("gate", "ki_gate_record_s3.py", "if tag not in TAGS or not re.fullmatch", "if not re.fullmatch"),
    ("gate", "ki_gate_record_s3.py", "or a.tag not in TAGS or rec.get(\"tag\") != a.tag:", ":"),
    ("gate", "ki_gate_record_s3.py", 'if gs.get("n_regressions") or gs.get("new_test_failures") or gs.get("missing_in_new"):\n        bad.append("G-SUITE reads', 'if False:\n        bad.append("G-SUITE reads'),
    ("bank", "s3_bank_check.py", "if com.get(\"ckpt_sha256\" ) != ue_sha:".replace('" )', '")'), "if False:"),
    ("bank", "s3_bank_check.py", 'cov["fraction"] < 0.5 or', "cov[\"fraction\"] < 0.0 or"),
    ("bank", "s3_bank_check.py", "if rep[\"partial\"] is not False:", "if False:"),
    ("bank", "s3_bank_check.py", 'if not expect_windows or rep["n_windows"] != expect_windows:', "if False:"),
    ("bank", "s3_bank_check.py", 'if com.get("generator") != WANT_GEN:', "if False:"),
    ("bank", "s3_bank_check.py", 'if c.get("pass") is not True or c.get("readout") != "causal_g1x":', "if False:"),
    ("chain", "rh_s3_chain.sh", '(cd $HERE && sha256sum -c SHA256_S3.txt > /dev/null 2>&1) ||', "true ||"),
    ("chain", "rh_s3_chain.sh", "vstep shared chk_shared\n  vstep trees chk_trees", "vstep trees chk_trees"),
    ("chain", "rh_s3_chain.sh", "chk_trees() { chk_one_tree $RH_TREE_DIR $RH_BLOBS rh && chk_one_tree $KI_TREE_DIR $KI_BLOBS ki; }", "chk_trees() { chk_one_tree $RH_TREE_DIR $RH_BLOBS rh; }"),
    ("chain", "rh_s3_chain.sh", "chk_gsuite() { gs_valid $RH_GSUITE $RH_COMMIT && gs_valid $KI_GSUITE $KI_COMMIT; }", "chk_gsuite() { gs_valid $RH_GSUITE $RH_COMMIT; }"),
    ("chain", "rh_s3_chain.sh", "chk_gsuite() { gs_valid $RH_GSUITE $RH_COMMIT && gs_valid $KI_GSUITE $KI_COMMIT; }", "chk_gsuite() { gs_valid $KI_GSUITE $KI_COMMIT; }"),
    ("chain", "rh_s3_chain.sh", "[ \"$(sha256sum $f | cut -d' ' -f1)\" = $sha ] ||", "true ||"),
    ("chain", "rh_s3_chain.sh", "  $PY $HERE/s3_argv.py show --arm B0_s2 || return 1\n", ""),
    ("chain", "rh_s3_chain.sh", "$PY $HERE/rh_gate_record_s3.py precheck-lite --tag B0_s2 --readout $RH_READOUT || return 1", "true"),
    ("chain", "rh_s3_chain.sh", "$PY $HERE/ki_gate_record_s3.py precheck-lite --tag KI_s2 --readout $RH_READOUT || return 1", "true"),
    ("chain", "rh_s3_chain.sh", "  $PY $HERE/s3_argv.py template --check $HERE/U_s2.B.tmpl.json || return 1", "  true"),
    ("chain", "rh_s3_chain.sh", "[ -e $R/runs/$t/ARM_DONE ] && [ -s $R/runs/$t/argv.json ] ||", "true ||"),
    ("chain", "rh_s3_chain.sh", "[ \"$(cat $R/RH_READOUT.chosen 2>/dev/null)\" = causal_g1x ] ||", "true ||"),
    ("chain", "rh_s3_chain.sh", "(cd $DS && sha256sum -c $fd > /dev/null 2>&1) ||", "true ||"),
    ("marker", "rh_s3_marker_watch.sh", '[ "$(cat "$M" 2>/dev/null)" = "$(cat "$MINE")" ]', "true"),
    ("marker", "rh_s3_marker_watch.sh", "( set -C; printf", "( printf"),
    ("marker", "rh_s3_marker_watch.sh", "NEW=$(tail -n +$((B + 1)) $CL 2>/dev/null)", "NEW=$(tail -n +1 $CL 2>/dev/null)"),
    ("marker", "rh_s3_marker_watch.sh", 'grep -q "GPU queue lock held to the end of the chain"', 'grep -q "NEVER-MATCHES-LOCK-LINE"'),
]


def run(kind, d):
    a = [x.replace("@K", d).replace("@S", os.path.join(SCR, "w_" + kind)) for x in T[kind]]
    if kind == "marker":
        a = [BASH.replace("\\", "/")] + a[1:]
        cmd = a
    elif kind == "chain":
        cmd = [PY] + a
    else:
        cmd = [PY] + a
    os.makedirs(os.path.join(SCR, "w_" + kind), exist_ok=True)
    r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=1800, env=dict(os.environ, S3_TMP=SCR))
    return r.returncode, (r.stdout + r.stderr)[-300:]


def main():
    res = []
    kinds = sorted({m[0] for m in M})
    for kind in kinds:
        d = os.path.join(SCR, "c_" + kind)
        if ONLY and ONLY not in kind:
            continue
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(HERE, d, ignore=shutil.ignore_patterns("src", "__pycache__", "*.md"))
        rc, out = run(kind, d)
        print(f"CONTROL {kind}: rc={rc} ({'ok' if rc == 0 else 'NOT GREEN: ' + out[-200:]})", flush=True)
        shutil.rmtree(d, ignore_errors=True)
        for k, f, old, new in M:
            if k != kind:
                continue
            shutil.copytree(HERE, d, ignore=shutil.ignore_patterns("src", "__pycache__", "*.md"))
            p = os.path.join(d, f)
            s = open(p, encoding="utf-8", newline="").read()
            if "\r\n" in s:
                old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
            if s.count(old) != 1:
                print(f"MUTANT {f}: {old[:60]!r} occurs {s.count(old)} times: NOT APPLIED", flush=True)
                res.append(False)
                shutil.rmtree(d, ignore_errors=True)
                continue
            open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new))
            rc, out = run(kind, d)
            red = rc != 0
            res.append(red)
            print(f"MUTANT {'RED' if red else 'GREEN (SURVIVED)'} {f}: {old[:70]!r}", flush=True)
            shutil.rmtree(d, ignore_errors=True)
    print("RESULT", f"{sum(res)}/{len(res)} mutants RED", "ALL RED" if all(res) else "SURVIVORS")
    return 0 if all(res) else 1


if __name__ == "__main__":
    sys.exit(main())
