"""test_smoothreq.py -- tests of the queue, the pod read scripts, the reader and the table tool of card ADD85 (SMOOTHREQ10K). Synthetic only; nothing runs on the pod.
The queue and the pod scripts can be replaced by a mutated copy through the environment (SR_QUEUE, SR_PODREAD, SR_CHAIN) for run_mutations_sr.py; SR_READER and SR_TOOL do the same for the reader and the table tool.
Run: PYCAP_MB=3000 bash ops/pycap.sh -m pytest smoothreq/test_smoothreq.py -q -p no:cacheprovider --basetemp smoothreq/tmp_pt/<new name>"""
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
QUEUE = Path(os.environ.get("SR_QUEUE", str(HERE / "pod_smoothreq10k_queue.sh")))
PODREAD = Path(os.environ.get("SR_PODREAD", str(HERE / "podread_smoothreq10k.sh")))
CHAIN = Path(os.environ.get("SR_CHAIN", str(HERE / "chain_read_smoothreq10k.sh")))
READER = Path(os.environ.get("SR_READER", str(HERE / "read_smoothreq10k.py")))
TOOL = Path(os.environ.get("SR_TOOL", str(HERE / "make_smooth_tables.py")))
WSHA = "02a58e655e5c1979"
ORDER = ["STKM", "TKD", "Z00", "S90", "S50", "S7B", "W7B"]
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"


def _bash():
    for c in ("C:/Program Files/Git/bin/bash.exe", "C:/Program Files/Git/usr/bin/bash.exe"):
        if os.path.isfile(c):
            return c
    return shutil.which("bash")


def _run(args, env=None, cwd=None):
    e = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **(env or {}))
    return subprocess.run(args, env=e, capture_output=True, text=True, encoding="utf-8", timeout=900, cwd=cwd)


def dry(seed=None):
    env = {"DRY": "1"}
    if seed is not None:
        env["SR_SEED"] = seed
    r = _run([_bash(), str(QUEUE).replace("\\", "/")], env)
    return r.returncode, r.stdout, r.stderr


def cmds(out):
    return [x for x in out.splitlines() if x.startswith("CMD:")]


def text(p):
    return Path(p).read_text(encoding="utf-8")


def tag_of(c):
    return re.search(r"--tag (\S+)_s(\d)_m10_causal_heads", c).groups()


# ------------------------------------------------------------------------------------------------ queue
def test_queue_syntax():
    assert _run([_bash(), "-n", str(QUEUE).replace("\\", "/")]).returncode == 0
    assert b"\r" not in QUEUE.read_bytes()


def test_queue_dry_all():
    rc, out, _ = dry()
    cs = cmds(out)
    assert rc == 0 and len(cs) == 14 and "DRY DONE" in out
    tags = [tag_of(c) for c in cs]
    assert tags == [("I10RQ50e%sKF5P30Hb20" % a, str(k)) for a in ORDER for k in (0, 1)]
    for c in cs:
        a = tag_of(c)[0][len("I10RQ50e"):-len("KF5P30Hb20")]
        k = tag_of(c)[1]
        assert "--closing-table /workspace/v8proof/closeread/est_smooth/tab_%s_s%s.npz" % (a, k) in c
        assert "--form RRe" in c and "--preroll-steps 30" in c and "--rule-every 5" in c and "--readout causal_g" in c and "--acc-gain 0" in c and "--path-exec" in c and "--kappa-max 0.2" in c
        assert "--eval-grid /workspace/v8proof/integ10k/longroll/grid_p30.json" in c and "v8f_t1_eval_reader_rule_rzgfhqet.py" in c and "--qmap /workspace/v8proof/integ10k/qmap_sweep/QMAP_LR16_s%s_q50.json" % k in c
        assert "--reader /workspace/v8proof/integ10k/lr16/TK16_s%s/ckpt_last.pt" % k in c


@pytest.mark.parametrize("seed", ["0", "1"])
def test_queue_dry_one_seed(seed):
    rc, out, _ = dry(seed)
    cs = cmds(out)
    assert rc == 0 and len(cs) == 7
    assert [tag_of(c) for c in cs] == [("I10RQ50e%sKF5P30Hb20" % a, seed) for a in ORDER]
    assert "queue start (DRY=1, seeds %s)" % seed in out


def test_queue_seed_switch_refused():
    for bad in ("2", "01", "all", ""):
        if bad == "":
            continue
        rc, out, err = dry(bad)
        assert rc == 2 and "REFUSED" in out + err and not cmds(out)


def test_queue_markers_and_log_text():
    t = text(QUEUE)
    for s in ('START_SMOOTHREQ10K$PS.txt', 'SMOOTHREQ10K_STOP$PS.txt', 'SMOOTHREQ10K_DONE$PS.txt', 'LOG=$W/queue_smoothreq$PS.log', 'SM=$W/small_sr', 'PS="_s$SEEDSW"', 'ARMS=("$SEEDSW")',
              'TABLES=${I10_TABLES:-$V/closeread/est_smooth}', 'SMOOTH_TABLES_RECORD.json', 'case "$SEEDSW" in 0|1|ALL)', 'SR_SEED:-ALL', '[ "$got" = "%s" ]' % WSHA, 'CHECK_SHUF_DONE_$TAG.txt', 'ZZ CHECK-SHUFCLOSE PASS',
              'I10RQ50eTKKF5Hb20', 'I10RQ50cKF5P30Hb20', 'I10RQ50eL100KF5P30Hb20', 'I10RQ50eS70KF5P30Hb20', 'I10RQ50eW90KF5P30Hb20', 'I10RQ50eN10KF5P30Hb20', 'I10RQ50eL25KF5P30Hb20', 'I10R0KP30Hb20'):
        assert s in t, s
    code = chr(10).join(x for x in t.split(chr(10)) if not x.startswith("#"))
    assert "TABCLOSE" not in code and "tabclose" not in code and "PART" not in code
    assert t.count("RGRID=$G30 PREROLL=30") == 7 and "Hb20 RRe" not in t.replace("P30Hb20 RRe", "")


ADMIT_RE = re.compile(r"<<'PYEOF'\n(.*?)\nPYEOF", re.S)


def _admit_code():
    m = [x for x in ADMIT_RE.findall(text(QUEUE)) if "ADMIT-OK" in x]
    assert len(m) == 1
    return m[0]


def _admit(tmp, mod=None, drop=None, alter=None):
    import hashlib
    d = Path(tmp)
    d.mkdir(parents=True, exist_ok=True)
    tabs = {}
    for a in ORDER:
        for k in (0, 1):
            fn = "tab_%s_s%d.npz" % (a, k)
            (d / fn).write_bytes(("data %s %d" % (a, k)).encode())
            tabs[fn] = hashlib.sha256((d / fn).read_bytes()).hexdigest()
    rec = {"pass": True, "tables": tabs, "b0": {"PASS": True}, "b1": {"PASS": True}, "b2": {"PASS": True}}
    rec.update(mod or {})
    if drop:
        (d / drop).unlink()
    if alter:
        (d / alter).write_bytes(b"other")
    (d / "rec.json").write_text(json.dumps(rec), encoding="utf-8")
    code = d / "admit.py"
    code.write_text(_admit_code(), encoding="utf-8")
    r = _run([PY, str(code), str(d / "rec.json"), str(d), ",".join(ORDER)])
    return r.returncode, r.stdout.strip()


def test_admission_passes(tmp_path):
    rc, out = _admit(tmp_path / "a")
    assert rc == 0 and out == "ADMIT-OK tables 14"


@pytest.mark.parametrize("name,mod,why", [("top", {"pass": False}, "RECORD-PASS-NOT-TRUE"), ("b0", {"b0": {"PASS": False}}, "CONTROL-NOT-PASS b0"), ("b1", {"b1": {"PASS": False}}, "CONTROL-NOT-PASS b1"),
                                           ("b2", {"b2": {"PASS": False}}, "CONTROL-NOT-PASS b2"), ("b2miss", {"b2": None}, "CONTROL-NOT-PASS b2"), ("notabs", {"tables": None}, "RECORD-HAS-NO-TABLES")])
def test_admission_refuses_record(tmp_path, name, mod, why):
    rc, out = _admit(tmp_path / name, mod)
    assert rc == 1 and out == why


def test_admission_refuses_tables(tmp_path):
    assert _admit(tmp_path / "m", drop="tab_Z00_s1.npz") == (1, "NO-TABLE tab_Z00_s1.npz")
    assert _admit(tmp_path / "h", alter="tab_S7B_s0.npz") == (1, "TABLE-HASH-DIFFERS tab_S7B_s0.npz")
    assert _admit(tmp_path / "n", mod={"tables": {"tab_S50_s0.npz": "x"}})[1].startswith("TABLE-NOT-IN-RECORD")


# ------------------------------------------------------------------------------------------------ pod scripts
def test_pod_scripts_syntax():
    for p in (PODREAD, CHAIN):
        assert _run([_bash(), "-n", str(p).replace("\\", "/")]).returncode == 0
        assert b"\r" not in p.read_bytes()


def test_chain_text():
    t = text(CHAIN)
    for s in ('SMOOTHREQ10K_DONE.txt', 'SMOOTHREQ10K_DONE_s0.txt', 'SMOOTHREQ10K_DONE_s1.txt', 'GO_READ_SMOOTHREQ10K.txt', '"$W"/SMOOTHREQ10K_STOP*.txt', 'START_READ_SMOOTHREQ10K.txt',
              'I10RQ50e{STKM,TKD,Z00,S90,S50,S7B,W7B,TK,STK,L100,W00,W50,W70,W90,S70}KF5P30Hb20_s[01]', 'ZZ CHAIN-READ-SMOOTHREQ10K END', 'OUT=$P/out/read_smoothreq10k', '--tabclose-json "$TBJ"',
              'est_smooth', 'WSHA=%s' % WSHA, 'podread_smoothreq10k.sh', '&& break'):
        assert s in t, s
    assert "TABCLOSE10K_DONE" not in t and "PART" not in t and "rm " not in t and "mv " not in t and "pkill" not in t and "ps " not in t


def test_podread_text():
    t = text(PODREAD)
    for s in ('--tabclose-json) TBJ=$2', 'SMOOTHREQ10K_DONE_s0.txt', 'SMOOTHREQ10K_STOP*.txt', 'read_smoothreq10k.py', 'SMOOTH_TABLES_RECORD.json', 'PREREG_V8F_PROOF_ADD84', 'PREREG_V8F_PROOF_ADD79', 'READ_SMOOTHREQ10K.json',
              'ZZ PODREAD-SMOOTHREQ10K DONE', 'STKM TKD Z00 S90 S50 S7B W7B', 'RQ50e${x}KF5P30Hb20', '--tabclose-json "$TBJ" --tab-dir "$TABDIR"'):
        assert s in t, s
    assert "TAB_TABLES" not in t and "PART" not in t and " rm " not in t and "pkill" not in t


BANNED = ["vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lo" + "nger", "clo" + "sed loop"]


@pytest.mark.parametrize("p", [QUEUE, PODREAD, CHAIN, READER, TOOL, HERE / "SMOOTHREQ_DEFS.md"])
def test_banned_words(p):
    s = text(p).lower()
    assert not [w for w in BANNED if w in s]


# ------------------------------------------------------------------------------------------------ the python tools
def test_reader_selftest():
    r = _run([PY, str(READER), "--selftest"], {"SR_ESTREAD_DIR": str(V / "estclose_read")})
    assert r.returncode == 0 and "ZZ SELFTEST-SMOOTHREQ10K ok: True" in r.stdout.splitlines()[-1], r.stdout[-800:] + r.stderr[-500:]


def test_tool_selftest():
    r = _run([PY, str(TOOL), "--selftest"], {"SMOOTH_TOOL_DIR": str(V / "tabclose")})
    assert r.returncode == 0 and r.stdout.splitlines()[-1].startswith("ZZ SELFTEST PASS"), "\n".join(x for x in r.stdout.splitlines() if x.startswith("FAIL"))[:800] + r.stderr[-500:]
