"""test_tabclose.py -- tests of the queue, the pod read scripts and the reader of card ADD84 (TABCLOSE10K). Synthetic only; nothing runs on the pod.
The queue and the pod scripts can be replaced by a mutated copy through the environment (TB_QUEUE, TB_PODREAD, TB_CHAIN) for run_mutations_tb.py.
Run: PYCAP_MB=3000 bash ops/pycap.sh -m pytest tabclose/test_tabclose.py -q -p no:cacheprovider --basetemp tabclose/tmp_pt/<new name>"""
import importlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
V = HERE.parent
QUEUE = Path(os.environ.get("TB_QUEUE", str(HERE / "pod_tabclose10k_queue.sh")))
PODREAD = Path(os.environ.get("TB_PODREAD", str(HERE / "podread_tabclose10k.sh")))
CHAIN = Path(os.environ.get("TB_CHAIN", str(HERE / "chain_read_tabclose10k.sh")))
WSHA = "02a58e655e5c1979"
ARMS_A = ["TK", "STK"]
ARMS_B = ["L100", "W70", "W50", "W90", "W00", "S70"]
ORDER = [("TK", "P30Hb20"), ("STK", "P30Hb20"), ("L100", "P30Hb20"), ("W70", "P30Hb20"), ("W50", "P30Hb20"), ("W90", "P30Hb20"), ("W00", "P30Hb20"), ("S70", "P30Hb20"), ("TK", "Hb20")]


def _bash():
    for c in ("C:/Program Files/Git/bin/bash.exe", "C:/Program Files/Git/usr/bin/bash.exe"):
        if os.path.isfile(c):
            return c
    return shutil.which("bash")


def _run(args, env=None, stdin=None):
    e = dict(os.environ, **(env or {}))
    return subprocess.run(args, env=e, input=stdin, capture_output=True, text=True, encoding="utf-8", timeout=300)


def dry(part=None):
    env = {"DRY": "1"}
    if part:
        env["TB_PART"] = part
    r = _run([_bash(), str(QUEUE).replace("\\", "/")], env)
    return r.returncode, r.stdout, r.stderr


def cmds(out):
    return [x for x in out.splitlines() if x.startswith("CMD:")]


def tag_of(c):
    return re.search(r"--tag (\S+)_s(\d)_m10_causal_heads", c).groups()


def text(p):
    return Path(p).read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------------ queue
def test_queue_syntax():
    assert _run([_bash(), "-n", str(QUEUE).replace("\\", "/")]).returncode == 0
    assert "\r" not in text(QUEUE)


def test_queue_dry_all_eighteen_in_card_order():
    rc, out, err = dry()
    assert rc == 0 and out.rstrip().endswith("DRY DONE"), err
    cs = cmds(out)
    assert len(cs) == 18
    got = [tag_of(c) for c in cs]
    want = [("I10RQ50e%sKF5%s" % (a, s), str(k)) for a, s in ORDER for k in (0, 1)]
    assert got == want


def test_queue_parts():
    rca, outa, _ = dry("A")
    rcb, outb, _ = dry("B")
    ta, tb = [tag_of(c)[0] for c in cmds(outa)], [tag_of(c)[0] for c in cmds(outb)]
    assert rca == 0 and rcb == 0 and len(ta) == 6 and len(tb) == 12
    assert sorted(set(ta)) == sorted("I10RQ50e%sKF5%s" % (a, s) for a, s in (("TK", "P30Hb20"), ("STK", "P30Hb20"), ("TK", "Hb20")))
    assert sorted(set(tb)) == sorted("I10RQ50e%sKF5P30Hb20" % a for a in ARMS_B)
    assert not set(ta) & set(tb)
    assert dry("ALL")[0] == 0 and len(cmds(dry("ALL")[1])) == 18 and len(cmds(dry("")[1])) == 18
    rc, out, _ = dry("Z")
    assert rc == 2 and "REFUSED" in out and not cmds(out)


def test_queue_command_lines():
    for c in cmds(dry()[1]):
        base, k = tag_of(c)
        age = "P30Hb20" in base
        arm = re.match(r"I10RQ50e(.+?)KF5", base).group(1)
        assert "rzgfhqet.py" in c and "--form RRe" in c
        assert "--closing-table /workspace/v8proof/closeread/est_tab/tab_%s_s%s.npz" % (arm, k) in c
        assert "--closing-est" not in c
        assert "--reader /workspace/v8proof/integ10k/lr16/TK16_s%s/ckpt_last.pt" % k in c
        assert "--qmap /workspace/v8proof/integ10k/qmap_sweep/QMAP_LR16_s%s_q50.json" % k in c
        assert "--ckpt /workspace/v8proof/integ10k/ckpt/UE_s%s/A/ckpt_last.pt" % k in c
        for f in ("--rule-every 5", "--kappa-max 0.2", "--acc-gain 0", "--path-exec", "--readout causal_g", "--replan-steps 10", "--scene logged", "--schedule periodic", "--limit causal", "--dump-heads", "--rule-log"):
            assert f in c, f
        assert ("--preroll-steps 30" in c) == age and ("--preroll-steps 0" in c) == (not age)
        assert ("grid_p30.json" in c) == age and ("eval_grid_windows.json" in c) == (not age)
        assert "--nav deadreckon" in c


def test_queue_markers_and_folders():
    t = text(QUEUE)
    for s in ('START_TABCLOSE10K$PS.txt', 'TABCLOSE10K_DONE$PS.txt', 'TABCLOSE10K_STOP.txt', "queue_tabclose.log", "small_tb", "est_tab", "TB_PART", "I10_TABLES"):
        assert s in t, s
    assert t.count("START_TABCLOSE10K$PS.txt") == 2 and "PS=\"\"; [ \"$PART\" = ALL ] || PS=\"_$PART\"" in t
    assert "SHUF" not in t.replace("CHECK_SHUF_DONE_", "").replace("CHECK-SHUFCLOSE", "").replace("CHECK_SHUFCLOSE_", "").replace("shufclose_test", "")
    assert not re.search(r"\b(rm|mv|kill|pkill|pgrep|killall|ps)\b ", re.sub(r"#.*", "", t))
    assert "MAXGPU=3" in t and "MINGB=${I10_MIN_FREE_GB:-28}" in t and "sleep 45" in t and "SLOT-PROBE-UNREADABLE" in t


def test_queue_admission_lines():
    t = text(QUEUE)
    assert 'TAG=${SC_TAG:-sk1}' in t and 'CHECK_SHUF_DONE_$TAG.txt' in t and '^ZZ CHECK-SHUFCLOSE PASS' in t
    assert t.count(WSHA) >= 3 and 'CHECK_SHUFCLOSE_$TAG.txt.hashes' in t and 'TAB_TABLES_RECORD.json' in t
    for nm in ("I10RQ50KF5P30Hb20", "I10RLcKF5P30Hb20", "I10RQ50KF5Hb20", "I10RLcKF5Hb20", "I10R0KP30Hb20", "I10R0KHb20", "I10R0Hb20", "I10RQ50eN10KF5P30Hb20", "I10RQ50eL25KF5P30Hb20", "I10RQ50eN10KF5Hb20", "I10RQ50eL25KF5Hb20"):
        assert nm in t, nm
    for stop in ("NO-PASS-OF-THE-WRAPPER-CHECK", "WRAPPER-SHA", "CHECK-RECORD-HAS-NOT-THIS-WRAPPER", "NO-TAB-TABLES-RECORD", "TABLE-ADMISSION", "NO-BANKED-RUN"):
        assert stop in t, stop


def _admit_py():
    t = text(QUEUE)
    m = re.search(r"admit\(\) \{[^\n]*\n[^\n]*<<'PYEOF'\n(.*?)\nPYEOF\n", t, re.S)
    assert m
    return m.group(1)


def _admit(tmp, rec, files, part, arms):
    d = Path(tmp)
    for fn, data in files.items():
        (d / fn).write_bytes(data)
    (d / "TAB_TABLES_RECORD.json").write_text(json.dumps(rec), encoding="utf-8")
    r = _run([sys.executable, "-", str(d / "TAB_TABLES_RECORD.json"), str(d), part, ",".join(arms)], stdin=_admit_py())
    return r.returncode, r.stdout.strip()


def _mk(arms):
    import hashlib
    files = {"tab_%s_s%d.npz" % (a, k): ("table %s %d" % (a, k)).encode() for a in arms for k in (0, 1)}
    return files, {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}


def test_admission_python(tmp_path_factory):
    def go(name, **kw):
        tmp = tmp_path_factory.mktemp(name)
        part = kw.get("part", "ALL")
        arms = ARMS_A + ARMS_B if part == "ALL" else (ARMS_A if part == "A" else ARMS_B)
        files, shas = _mk(arms)
        rec = {"pass": True, "tables": dict(shas), **{k: {"PASS": True} for k in ("a0", "a1", "a4_u1", "a4_u4")}}
        rec.update(kw.get("rec", {}))
        for k in kw.get("drop", []):
            rec.pop(k, None)
        for k, v in kw.get("files", {}).items():
            if v is None:
                files.pop(k)
            else:
                files[k] = v
        return _admit(tmp, rec, files, part, arms)
    assert go("ok")[0] == 0 and go("okA", part="A")[0] == 0 and go("okB", part="B")[0] == 0
    assert go("passfalse", rec={"pass": False})[0] == 1
    assert go("nopass", drop=["pass"])[0] == 1
    # a part switch needs the controls that guard the part, not the top-level pass
    assert go("A_passfalse", part="A", rec={"pass": False})[0] == 0 and go("B_passfalse", part="B", rec={"pass": False})[0] == 0
    assert go("A_nopass", part="A", drop=["pass"])[0] == 0
    assert go("a0", rec={"a0": {"PASS": False}})[0] == 1 and go("a1", rec={"a1": {"PASS": False}})[0] == 1
    assert go("a4u1", rec={"a4_u1": {"PASS": False}})[0] == 1 and go("a4u4", rec={"a4_u4": {"PASS": False}})[0] == 1
    # the controls of the other part do not stop a part (the record's top-level pass is still required)
    assert go("a0_in_A", part="A", rec={"a0": {"PASS": False}})[0] == 0 and go("a4_in_B", part="B", rec={"a4_u1": {"PASS": False}, "a4_u4": {"PASS": False}})[0] == 0
    r_ = go("hash", files={"tab_W70_s1.npz": b"changed"}); assert r_[0] == 1 and "TABLE-HASH-DIFFERS tab_W70_s1.npz" in r_[1]
    assert go("hashTK", files={"tab_TK_s0.npz": b"changed"})[0] == 1
    r_ = go("missing", files={"tab_S70_s0.npz": None}); assert r_[0] == 1 and r_[1] == "NO-TABLE tab_S70_s0.npz"
    r_ = go("notinrec", rec={"tables": {}}); assert r_[0] == 1 and "TABLE-NOT-IN-RECORD" in r_[1] and go("notables", drop=["tables"])[0] == 1


# ------------------------------------------------------------------------------------------------ pod read scripts
def test_podread_and_chain():
    for p in (PODREAD, CHAIN):
        assert _run([_bash(), "-n", str(p).replace("\\", "/")]).returncode == 0
        assert "\r" not in text(p) and "shuf" not in text(p).lower()
        assert not re.search(r"\b(rm|mv|kill|pkill|pgrep|killall)\b ", re.sub(r"#.*", "", text(p)))
    t = text(PODREAD)
    for s in ('read_tabclose10k.py', 'READ_TABCLOSE10K.json', 'TABCLOSE10K_DONE$PS.txt', 'TABCLOSE10K_STOP.txt', '--tab-dir) TABDIR=$2', '--part) PART=$2', 'TAB_TABLES_RECORD.json', 'ZZ PODREAD-TABCLOSE10K DONE', 'ZZ PODREAD-TABCLOSE10K STOP',
              '--tab-dir "$TABDIR" --part "$PART"', "ALREADY-READ", "SELFTEST-ON-POD", "PREREG_V8F_PROOF_ADD79", "grid_p30.json"):
        assert s in t, s
    for r in ("RQ50eTKKF5P30Hb20", "RQ50eSTKKF5P30Hb20", "RQ50eTKKF5Hb20", "RQ50eL100KF5P30Hb20", "RQ50eW70KF5P30Hb20", "RQ50eW50KF5P30Hb20", "RQ50eW90KF5P30Hb20", "RQ50eW00KF5P30Hb20", "RQ50eS70KF5P30Hb20",
              "RQ50eN10KF5P30Hb20", "RQ50eL25KF5Hb20", "RQ50cKF5P30Hb20", "RLcKF5Hb20", "R0KP30Hb20", "R0Hb20"):
        assert r in t, r
    assert 'ARMS_N="TK STK"' in t and "L100 W70 W50 W90 W00 S70" in t
    c = text(CHAIN)
    for s in ('TABCLOSE10K_DONE$PS.txt', 'GO_READ_TABCLOSE10K.txt', 'START_READ_TABCLOSE10K$PS.txt', 'TABCLOSE10K_STOP.txt', 'read_tabclose10k$PS', 'est_tab', 'I10RQ50e{TK,STK,L100,W00,W50,W70,W90,S70}*_s[01]', 'podread_tabclose10k.sh',
              '--tab-dir "$SD" --part "$PART"', "ZZ CHAIN-READ-TABCLOSE10K END", WSHA, "TB_PART"):
        assert s in c, s
    assert c.rstrip().splitlines()[-1] == 'echo "ZZ CHAIN-READ-TABCLOSE10K END rc $rc"'
    assert "[ -f \"$W/TABCLOSE10K_DONE$PS.txt\" ] && [ -f \"$W/GO_READ_TABCLOSE10K.txt\" ] && break" in c


def test_chain_links_every_new_run_and_nothing_else(tmp_path):
    import fnmatch
    names = ["I10RQ50e%sKF5%s_s%d" % (a, s, k) for a, s in ORDER for k in (0, 1)]
    other = ["I10RQ50eN10KF5P30Hb20_s0", "I10RQ50eL25KF5Hb20_s1", "I10RQ50KF5P30Hb20_s0", "I10RLcKF5Hb20_s0", "I10RQ50eSL25KF5Hb20_s0"]
    m = re.search(r'"\$W"/t1/(I10RQ50e\{[A-Z0-9,]+\}\*_s\[01\])', text(CHAIN))
    assert m
    alts = re.search(r"\{([^}]+)\}", m.group(1)).group(1).split(",")
    pats = ["I10RQ50e%s*_s[01]" % a for a in alts]
    for n in names:
        assert any(fnmatch.fnmatchcase(n, p) for p in pats), n
    for n in other:
        assert not any(fnmatch.fnmatchcase(n, p) for p in pats), n


# ------------------------------------------------------------------------------------------------ reader
@pytest.fixture(scope="module")
def RT():
    os.environ.setdefault("TB_ESTREAD_DIR", str(V / "estclose_read"))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    sys.path.insert(0, str(HERE))
    return importlib.import_module("read_tabclose10k")


def test_reader_selftest_passes(RT, capsys):
    assert RT.selftest() == 0
    assert "ok: True" in capsys.readouterr().out


def test_reader_runs_equal_queue_runs(RT):
    qnames = sorted({tag_of(c)[0] for c in cmds(dry()[1])})
    cfg = RT.settings("ALL")
    rnames = sorted("I10" + r for st in cfg for f, r in cfg[st]["runs"].items() if f in RT.NEW_FORMS)
    assert qnames == rnames and len(qnames) == 9
    assert RT.arms_of("A") == ("TK", "STK") and set(RT.arms_of("B")) == set(ARMS_B) and RT.arms_of("ALL") == RT.arms_of("A") + RT.arms_of("B")


def test_reader_tables_in_pod_scripts_match(RT):
    t = text(PODREAD)
    for a in RT.NEW_ARMS:
        assert a in t


def test_kit_words():
    bad = ["vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lo" + "nger", "clo" + "sed loop", "driving " + "performance"]
    for p in (QUEUE, PODREAD, CHAIN, HERE / "read_tabclose10k.py", HERE / "TABCLOSE_DEFS.md"):
        if Path(p).exists():
            low = text(p).lower()
            assert not [w for w in bad if w in low], (p, [w for w in bad if w in low])


def test_generators_refuse_to_overwrite():
    for g, env in (("make_tab_queue.py", {}), ("make_read_tabclose.py", {}), ("make_pod_read_tabclose.py", {})):
        r = _run([sys.executable, str(HERE / g)], dict(env, PYTHONDONTWRITEBYTECODE="1", TB_OVERWRITE_OWN="0"))
        assert r.returncode != 0 and "REFUSED" in (r.stdout + r.stderr), g
