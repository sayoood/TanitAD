"""test_negconst.py -- tests of the tool kit of card ADD82 (NEGCONST10K): the three tools with a selftest (make_negconst_files.py, check_negconst.py, read_negconst10k.py), direct cases on the pure functions of the
reader, the queue generator and the queue it writes (bash -n and DRY=1 output; no pod path is touched), the generators of the pod check, the pod read and the chain, and the wiring of the reader (static text tests: the
mechanisms of the reader are reached from its run). NC_TOOL_DIR (default: this folder) names the folder of the tools under test; the mutation runner points it at a mutated copy.
  PYCAP_MB=3000 bash ops/pycap.sh -m pytest -q -p no:cacheprovider negconst/test_negconst.py --basetemp negconst/tmp_pt/x
"""
import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.environ.get("NC_TOOL_DIR") or HERE
ROOT = os.path.dirname(HERE)
CC = os.path.join(ROOT, "constclose")
ENV = dict(os.environ, CLOSEST_DIR=os.environ.get("CLOSEST_DIR") or os.path.join(ROOT, "closeread"), CC_ESTREAD_DIR=os.environ.get("CC_ESTREAD_DIR") or os.path.join(ROOT, "estclose_read"),
           ECR_INTEG_DIR=os.environ.get("ECR_INTEG_DIR") or os.path.join(ROOT, "integ10k"), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1")


def _bash():
    for p in (os.environ.get("NC_BASH", ""), "C:/Program Files/Git/usr/bin/bash.exe", "C:/Program Files/Git/bin/bash.exe", "/usr/bin/bash", "/bin/bash"):
        if p and os.path.isfile(p):
            return p
    return "bash"


BASH = _bash()
ORDER = [("P30Hb20", "M10"), ("P30Hb20", "M20"), ("P30Hb20", "M40"), ("Hb20", "M10"), ("Hb20", "M20"), ("Hb20", "M40")]


def selftest(tool):
    r = subprocess.run([sys.executable, os.path.join(TOOLS, tool), "--selftest"], env=ENV, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1500, cwd=TOOLS)
    return r.returncode, r.stdout + r.stderr


@pytest.mark.parametrize("tool,last", [("make_negconst_files.py", "ZZ NEGCONSTFILES-SELFTEST rc 0"), ("check_negconst.py", "ZZ SELFTEST-CHECK-NEGCONST ok: True"), ("read_negconst10k.py", "ZZ SELFTEST-NEGCONST10K ok: True")])
def test_selftest_green(tool, last):
    rc, out = selftest(tool)
    assert rc == 0, out[-1500:]
    assert out.strip().splitlines()[-1].startswith(last), out[-400:]
    assert "False" not in [l.split()[-1] for l in out.splitlines() if l.startswith("   ")], out[-1500:]
    assert not [l for l in out.splitlines() if l.startswith("FAIL ")], out[-1500:]


def test_reader_pure_functions():
    sys.path.insert(0, TOOLS)
    for k, v in ENV.items():
        os.environ.setdefault(k, v)
    import read_negconst10k as RN
    assert RN.VALUES == ("M10", "M20", "M40") and RN.CVAL == {"M10": -1.0, "M20": -2.0, "M40": -4.0} and RN.OLDVAL == {"C10": 1.0, "C20": 2.0, "C40": 4.0}
    pr = RN.pairs_of(True)
    assert len(pr) == 15 and pr["CONST-VS-EST-L25-M40"] == ("RRe_M40", "RRe_L25") and pr["CONST-VS-RR-M10"] == ("RRe_M10", "RR")
    assert RN.cname("CONST-VS-RR-M10", "FRESH", "U", "LOWER") == "NEGCONST10K-CONST-VS-RR-M10-FRESH-U-LOWER"
    ns = {"C": "NOT-SEPARATED", "U": "NOT-SEPARATED", "F": "NOT-SEPARATED", "fam": {f: "NOT-SEPARATED" for f in RN.FAMS}}
    words = {st: {p: ns for p in pr} for st in RN.SETTING_ORDER}
    names, why = RN.all_names(words, {"AGE": True, "FRESH": True}, {"AGE": True, "FRESH": True}, True)
    assert names["VERDICT_M20"] == "NEGCONST10K-M20-NOT-SUPPORTED" and names["VERDICT_N10"] == "NEGCONST10K-N10-NOT-BEYOND-A-CAUTIOUS-CONSTANT"
    assert RN.NEWC == ("RRe_M10", "RRe_M20", "RRe_M40") and RN.OLDC == ("RRe_C10", "RRe_C20", "RRe_C40")
    assert [c for _, c in RN.DOSE] == [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
    st = RN.settings(True)
    assert st["AGE"]["runs"]["RRe_M20"] == "RQ50eM20KF5P30Hb20" and st["FRESH"]["runs"]["RRe_C40"] == "RQ50eC40KF5Hb20" and st["AGE"]["runs"]["RRe_N10"] == "RQ50eN10KF5P30Hb20"
    lev = {(f, s): v for f, v in (("RRe_M10", 0.10), ("RRe_C10", 0.12), ("RRe_M40", 0.05), ("RRe_C40", 0.20)) for s in RN.SEEDS}
    assert RN.ctl_k4(lev)["PASS"] and not RN.ctl_k4({**lev, ("RRe_M40", 0): 0.5, ("RRe_M40", 1): 0.5})["PASS"]


# ---------------------------------------------------------------------------------------------------- wiring of the reader (static text: the mechanisms are reached from run())
def _src(name):
    return open(os.path.join(TOOLS, name), encoding="utf-8").read()


def test_reader_wiring():
    t = _src("read_negconst10k.py")
    rs = t[t.index("def run_setting("):t.index("def run(a):")]
    r = t[t.index("def run(a):"):t.index("def find_closest(")]
    m = t[t.index("def main("):]
    assert 'ctl["k4_" + st] = ctl_k4(lev)' in rs and 'if st == "AGE":' in rs
    assert 'ctl["k3_constclose_" + st] = ctl_k3(' in rs and 'ctl["k3_estclose_" + st] = ctl_k3(' in rs and "CONST_CARD)" in rs and "PARENT_CARD)" in rs
    assert '"dose": (dose_table(G["levels"], runs) if q else hw_text)' in rs
    assert "forms_of(e_on)" in rs and "FORMS = forms_of(e_on)" in rs
    assert 'controls["k3_estclose_json_admitted"]' in r and 'controls["k3_constclose_json_admitted"]' in r
    assert "SET = settings(True)" in r and "bank, ej, cj, True, cache, log, CL" in r and "read_c_json(getattr(a, \"constclose_json\", None), R.jl)" in r
    assert "render_dose(res)" in r and "all_names(words," in r and "gate, True," in r
    assert 'ap.add_argument("--estclose-json", required=True)' in m and 'ap.add_argument("--constclose-json", required=True)' in m
    # the arms of the card in the run table and the form table, and the old runs in the run list the grid reader reads
    assert "for c in VALUES + OLDV}" in t and "return NEWC + OLDC + (NEWE if e_on else ()) + BANKED" in t and 'for c in VALUES + OLDV})' in t


def test_banned_words_absent_in_every_new_file():
    bad = ["vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lo" + "nger", "clo" + "sed loop"]
    for fn in sorted(os.listdir(TOOLS)):
        if fn.endswith((".py", ".sh", ".md", ".json")) and not fn.startswith("tmp_") and fn != "run_mutations_nc.py" and fn != "test_negconst.py" and not fn.startswith("MUTATIONS"):
            txt = open(os.path.join(TOOLS, fn), encoding="utf-8").read().lower()
            hit = [b for b in bad if b in txt]
            assert not hit, (fn, hit)


# ---------------------------------------------------------------------------------------------------- the generators and the scripts they write
def _gen(tmp_path, script, env_extra, args=()):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **env_extra)
    r = subprocess.run([sys.executable, os.path.join(TOOLS, script), *args], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=TOOLS)
    return r


@pytest.fixture(scope="module")
def queue(tmp_path_factory):
    d = tmp_path_factory.mktemp("q")
    out = d / "q.sh"
    r = _gen(d, "make_negconst_queue.py", {"NC_SRC": os.path.join(CC, "pod_constclose10k_queue.sh"), "NC_OUT": str(out)}, ("ADD82",))
    assert r.returncode == 0, r.stdout + r.stderr
    return out, out.read_text(encoding="utf-8")


def dry(path):
    env = dict(os.environ, DRY="1", I10_W="/x/y")
    r = subprocess.run([BASH, path.as_posix()], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-500:]
    return r.stdout


def test_queue_syntax_and_refusals(queue, tmp_path):
    path, text = queue
    assert subprocess.run([BASH, "-n", path.as_posix()]).returncode == 0
    r = _gen(tmp_path, "make_negconst_queue.py", {"NC_SRC": os.path.join(CC, "pod_constclose10k_queue.sh"), "NC_OUT": str(path)}, ("ADD82",))
    assert r.returncode != 0 and "REFUSED" in (r.stdout + r.stderr)
    bad = tmp_path / "other.sh"
    bad.write_text("echo not the queue of card ADD80\n")
    r = _gen(tmp_path, "make_negconst_queue.py", {"NC_SRC": str(bad), "NC_OUT": str(tmp_path / "o.sh")}, ("ADD82",))
    assert r.returncode != 0 and not (tmp_path / "o.sh").exists()


def test_queue_text_markers_names_and_checks(queue):
    path, text = queue
    for m in ("START_NEGCONST10K.txt", "NEGCONST10K_DONE.txt", "NEGCONST10K_STOP.txt", "queue_negconst.log", "small_nc"):
        assert m in text, m
    for m in ("CONSTCLOSE10K", "queue_constclose", "START_CONSTCLOSE", "constclose_test", "ESTCLOSE10K", "small_cc"):
        assert m not in text, m
    assert "a747abeb51ef041f" in text and "^ZZ CHECK-NEGCONST PASS" in text and "CHECK_NEG_DONE_$CKTAG.txt" in text and "CHECK_NEGCONST_$CKTAG.txt.hashes" in text
    assert 'CKTAG=${NC_CHECK_TAG:-nk1}' in text
    assert 'grep -q "$got" "$W/negconst_test/CHECK_NEGCONST_$CKTAG.txt.hashes"' in text and 'sha256sum "$ESTC/const_M20_H1.npz"' in text
    assert 'NO-CONST-FILE' in text and "for c in M10 M20 M40; do" in text
    assert 'ESTC=${I10_ESTC:-$V/closeread/est_negconst}' in text
    assert 'echo "$(ts) all steps done" > "$W/NEGCONST10K_DONE.txt"' in text and 'echo "$(ts) $*" > "$W/NEGCONST10K_STOP.txt"' in text and '> "$W/START_NEGCONST10K.txt"' in text
    assert 'LOG=$W/queue_negconst.log' in text and 'SM=$W/small_nc' in text
    # banked runs: the earlier ones and those of cards ADD79 and ADD80
    for nm in ("I10RQ50cKF5P30Hb20", "I10RQ50cKF5Hb20", "I10R0Hb20", "I10RQ50eN10KF5P30Hb20", "I10RQ50eL25KF5P30Hb20", "I10RQ50eN10KF5Hb20", "I10RQ50eL25KF5Hb20",
               "I10RQ50eC10KF5P30Hb20", "I10RQ50eC20KF5P30Hb20", "I10RQ50eC40KF5P30Hb20", "I10RQ50eC10KF5Hb20", "I10RQ50eC20KF5Hb20", "I10RQ50eC40KF5Hb20"):
        assert (" %s " % nm) in text or (" %s;" % nm) in text, nm
    assert text.count("\r") == 0


def test_twelve_rollouts_in_order_with_the_right_files(queue):
    path, text = queue
    out = dry(path)
    cmds = [l for l in out.splitlines() if l.startswith("CMD:")]
    assert len(cmds) == 12
    want = []
    for setting, c in ORDER:
        for k in (0, 1):
            want.append(("I10RQ50e%sKF5%s_s%d" % (c, setting, k), c, setting, k))
    for cmd, (arm, c, setting, k) in zip(cmds, want):
        assert " --form RRe " in cmd
        assert "--closing-est /workspace/v8proof/closeread/est_negconst/const_%s_H1.npz" % c in cmd
        assert "--rule-every 5" in cmd and "--readout causal_g" in cmd and "--acc-gain 0" in cmd and "--path-exec" in cmd and "--kappa-max 0.2" in cmd
        assert "--reader /x/y/lr16/TK16_s%d/ckpt_last.pt" % k in cmd and "--qmap /x/y/qmap_sweep/QMAP_LR16_s%d_q50.json" % k in cmd and "/x/y/ckpt/UE_s%d/A/ckpt_last.pt" % k in cmd
        assert "--out-dir /x/y/t1/%s " % arm in cmd and ("--tag %s_m10_causal_heads" % arm) in cmd
        assert "v8f_t1_eval_reader_rule_rzgfhqe.py" in cmd
        if setting == "P30Hb20":
            assert "--preroll-steps 30" in cmd and "grid_p30.json" in cmd
        else:
            assert "--preroll-steps 0" in cmd and "eval_grid_windows.json" in cmd
    assert out.strip().splitlines()[-1] == "DRY DONE"


def test_log_markers_in_dry(queue):
    out = dry(queue[0])
    assert "NEGCONST10K" in out and "12 rollouts" in out and "-1.0, -2.0, -4.0" in out
    assert not re.search(r"ESTCLOSE|MIDLOOK|CONSTCLOSE", out)


def test_pod_check_script(tmp_path):
    out = tmp_path / "c.sh"
    r = _gen(tmp_path, "make_pod_check_negconst.py", {"NC_SRC": os.path.join(CC, "pod_check_constclose.sh"), "NC_OUT": str(out)})
    assert r.returncode == 0, r.stdout + r.stderr
    t = out.read_text(encoding="utf-8")
    assert subprocess.run([BASH, "-n", out.as_posix()]).returncode == 0
    assert "TAG=${TAG:-nk1}" in t and "negconst_test" in t and "CHECK_NEG_DONE_$TAG.txt" in t and "CHECK_NEG_STOP_$TAG.txt" in t and "CHECK_NEGCONST_$TAG.txt" in t
    assert "check_negconst.py" in t and "const_M20_H1.npz" in t and "est_negconst" in t and "ZZ POD-CHECK-NEGCONST" in t and 'for c in M10 M20 M40' in t
    assert "CD=${CONSTDIR:-$V/closeread/est_negconst}" in t and '"$B/check_negconst.py"' in t and "T=$W/negconst_test" in t and 'tail -n 1 "$OUT" > "$T/CHECK_NEG_DONE_$TAG.txt"' in t
    for bad in ("CONSTCLOSE", "constclose", "const_C", "CHECK_CONST_"):
        assert bad not in t, bad
    r2 = _gen(tmp_path, "make_pod_check_negconst.py", {"NC_SRC": os.path.join(CC, "pod_check_constclose.sh"), "NC_OUT": str(out)})
    assert r2.returncode != 0 and "REFUSED" in (r2.stdout + r2.stderr)


def test_pod_read_and_chain_scripts(tmp_path):
    p, c = tmp_path / "p.sh", tmp_path / "c.sh"
    r = _gen(tmp_path, "make_podread_negconst.py", {"NC_CCDIR": CC, "NC_OUT_P": str(p), "NC_OUT_C": str(c)})
    assert r.returncode == 0, r.stdout + r.stderr
    tp, tc = p.read_text(encoding="utf-8"), c.read_text(encoding="utf-8")
    assert subprocess.run([BASH, "-n", p.as_posix()]).returncode == 0 and subprocess.run([BASH, "-n", c.as_posix()]).returncode == 0
    # the pod read: both result files are required and admitted, the runs of both cards are required, the new reader is the one run
    assert "--constclose-json) CONSTJ=$2" in tp and "for v in PULL CODE GRIDDIR FG MIDJ CONSTDIR W TREE OUT WSHA ESTJ CONSTJ; do" in tp
    assert 'stop "ADD79-READ-NOT-ADMITTED"' in tp and 'stop "ADD80-READ-NOT-ADMITTED"' in tp and "PREREG_V8F_PROOF_ADD79" in tp and "PREREG_V8F_PROOF_ADD80" in tp
    assert "$ERUNS $OLDRUNS; do" in tp and 'OLDRUNS="RQ50eC10KF5P30Hb20' in tp and 'ERUNS="RQ50eN10KF5P30Hb20 RQ50eL25KF5P30Hb20 RQ50eN10KF5Hb20 RQ50eL25KF5Hb20"' in tp and "RQ50eM10KF5P30Hb20" in tp and "RQ50eM40KF5Hb20" in tp
    assert 'EARGS="--estclose-json $ESTJ --constclose-json $CONSTJ"' in tp and "$EARGS" in tp
    assert "read_negconst10k.py --selftest" in tp and "read_negconst10k.py --pull" in tp and "READ_NEGCONST10K.json" in tp and "NEGCONST10K_DONE.txt" in tp and "const_M*_H1.npz" in tp
    assert "ZZ PODREAD-NEGCONST10K" in tp and "^ZZ READ-NEGCONST|" in tp
    for bad in ("CONSTCLOSE", "constclose10k", "const_C"):
        assert bad not in tp, bad
    # the chain
    assert "NEGCONST10K_DONE.txt" in tc and "GO_READ_NEGCONST10K.txt" in tc and "START_READ_NEGCONST10K.txt" in tc and "NEGCONST10K_STOP.txt" in tc
    assert "OUT=$P/out/read_negconst10k" in tc and "podread_negconst10k.sh" in tc and "I10RQ50eM*_s[01]" in tc
    assert 'EJ=$P/out/read_estclose10k_n2/READ_ESTCLOSE10K.json' in tc and 'CJ=$P/out/read_constclose10k/READ_CONSTCLOSE10K.json' in tc
    assert '--estclose-json "$EJ" --constclose-json "$CJ"' in tc and "--const-dir /workspace/v8proof/closeread/est_negconst" in tc and "--wrapper-sha a747abeb51ef041f" in tc
    assert "CONSTCLOSE10K" not in tc.replace("READ_CONSTCLOSE10K.json", "")
    r2 = _gen(tmp_path, "make_podread_negconst.py", {"NC_CCDIR": CC, "NC_OUT_P": str(p), "NC_OUT_C": str(c)})
    assert r2.returncode != 0 and "REFUSED" in (r2.stdout + r2.stderr)
