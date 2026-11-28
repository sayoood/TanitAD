"""Tests of pod_integpre_queue.sh (card PREREG_V8F_PROOF_ADD44): the DRY=1 output is parsed; each checkpoint's extract, lead read, fit and rollouts must use the files of that checkpoint, the rule forms must be the ones of
the card, the slot wait precedes every job, no process-name test, no rm, markers, volume check."""
import os
import re
import shutil
import subprocess

import pytest

import ip_testlib as T

BASH = shutil.which("bash")
SCRIPT = os.path.join(T.SRC, "pod_integpre_queue.sh")
ARMS = os.path.join(T.SRC, "integpre_arms.txt")
pytestmark = pytest.mark.skipif(BASH is None, reason="no bash on this box")


def sh(p):
    return str(p).replace("\\", "/")


def run(env_extra, tmp_path, timeout=60):
    env = dict(os.environ)
    env.update(env_extra)
    return subprocess.run([BASH, sh(SCRIPT)], env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=sh(tmp_path))


def dry(tmp_path):
    r = run({"DRY": "1", "IP_D": sh(tmp_path / "d"), "IP_ARMS": sh(ARMS)}, tmp_path)
    assert r.returncode == 0, r.stderr + r.stdout
    assert not os.path.exists(tmp_path / "d"), "DRY must create nothing"
    return r.stdout


def cmds(out):
    return [l for l in out.splitlines() if l.startswith("DRY: ") and "slot wait" not in l and "volume check" not in l and not l.startswith("DRY: at the end")]


def test_syntax_lf_and_forbidden_words():
    r = subprocess.run([BASH, "-n", sh(SCRIPT)], capture_output=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    raw = open(SCRIPT, "rb").read()
    assert b"\r" not in raw
    code = "\n".join(l for l in raw.decode("utf-8").splitlines() if not l.lstrip().startswith("#"))
    for bad in ("pgrep", "pkill", "killall", "ps -", "ps aux", "pidof"):
        assert bad not in code, bad
    assert not re.search(r"(^|[;&|\s])rm(\s|$)", code) and not re.search(r"(^|[;&|\s])mv(\s|$)", code), "rm or mv in the script"
    assert "--query-compute-apps=pid" in code and "memory.max" in code and "memory.stat" in code and "set -o noclobber" in code and "stat -f" in code
    assert "INTEGPRE_STARTED.txt" in code and "INTEGPRE_DONE.txt" in code and "INTEGPRE_STOP.txt" in code


def test_arms_file_has_the_eight_checkpoints_of_the_card():
    rows = [l.split() for l in open(ARMS, encoding="utf-8") if l.strip() and not l.startswith("#")]
    assert len(rows) == 8
    got = {(r[2]) for r in rows}
    assert got == {"%s_s%d" % (a, k) for a in ("RLAm", "RLFT") for k in range(4)}
    q = {r[2]: r[1] for r in rows}
    assert q["RLAm_s0"] == q["RLFT_s0"] == "actcons_c0" and q["RLAm_s1"] == "actcons_c1" and q["RLAm_s2"] == "actcons_e2" and q["RLFT_s3"] == "actcons_e3"
    assert all(r[0] == "IRQ_" + r[2] and int(r[3]) == int(r[2][-1]) for r in rows)


def test_each_checkpoint_uses_its_own_files(tmp_path):
    out = cmds(dry(tmp_path))
    rows = [l.split() for l in open(ARMS, encoding="utf-8") if l.strip() and not l.startswith("#")]
    for NEW, Q, BASE, SEED, P in rows:
        ck = "/runs/%s/%s/ckpt_last.pt" % (Q, BASE)
        ext = [l for l in out if "extract_states.py" in l and "--ckpt " in l and ck in l]
        lead = [l for l in out if "pod_lead_apply.py" in l and ck in l]
        assert len(ext) == 1 and len(lead) == 1, BASE
        assert re.search(r"--out \S*/states/%s_train\.npz" % BASE, ext[0]) and "--lo 0 --hi 4572" in ext[0]
        assert re.search(r"--states \S*/states/%s_train\.npz" % BASE, lead[0]) and re.search(r"--out \S*/states/%s_train_lead\.npz" % BASE, lead[0])
        fit = [l for l in out if "qhead_fit_n1.py" in l and "--arm %s " % BASE in l]
        assert len(fit) == 1, BASE
        f = fit[0]
        assert "--seed %s " % SEED in f and re.search(r"--states \S*/states/%s_train\.npz " % BASE, f) and "--lead " in f and "%s_train_lead.npz" % BASE in f
        assert "--fit-clips 3048 --select-clips 762 --check-clips 762" in f and "--device cpu" in f and "qhead_dose/pack/train_pack_all.npz" in f
        # the other arms' state files never appear in this checkpoint's lines
        others = [r[2] for r in rows if r[2] != BASE]
        assert not any(("/states/%s_train" % o) in f or ("/states/%s_train" % o) in ext[0] or ("/states/%s_train" % o) in lead[0] for o in others), BASE
        # the three rollouts
        for tag in ("IR0", "IRL", "IRQ"):
            line = [l for l in out if re.search(r" %s_%s %s %s " % (tag, BASE, Q, BASE), l) or re.search(r"bash \S+ %s_%s %s %s " % (tag, BASE, Q, BASE), l)]
            assert len(line) == 1, (tag, BASE)
        rq = [l for l in out if re.search(r"pod_integpre_run\.sh IRQ_%s " % BASE, l)][0]
        assert 'DR_MIX="head head head %s --qhead %s/heads/%s.qhead.npz"' % (P, "/".join(rq.split("DISTRULE_DIR=")[1].split()[0:1]), BASE) in rq
        assert "DR_RULEPY=v8f_t1_eval_rule_qhead.py" in rq and "--q-equals-head" not in rq


def test_rule_forms_are_the_cards(tmp_path):
    out = cmds(dry(tmp_path))
    for l in out:
        m = re.search(r"pod_ruleattr_run\.sh (IR\w+) (\S+) (\S+) (\w+) >", l)
        if m:
            tag = m.group(1).split("_")[0]
            assert {"IR0": "off", "IRL": "label", "IRH": "head"}[tag] == m.group(4), l
    assert sum(1 for l in out if re.search(r"pod_ruleattr_run\.sh IR0_\S+ \S+ \S+ off ", l)) == 8
    assert sum(1 for l in out if re.search(r"pod_ruleattr_run\.sh IRL_\S+ \S+ \S+ label ", l)) == 8
    assert sum(1 for l in out if "pod_integpre_run.sh IRQ_" in l) == 8
    assert sum(1 for l in out if "DR_FLOOR=cap-only" in l) == 8 * 3 + 2


def test_control_c3_is_on_candidate_base0_only(tmp_path):
    out = cmds(dry(tmp_path))
    h = [l for l in out if "pod_ruleattr_run.sh IRH_" in l]
    qi = [l for l in out if "pod_integpre_run.sh IRQI_" in l]
    assert len(h) == 1 and "IRH_RLAm_s0 actcons_c0 RLAm_s0 head" in h[0]
    assert len(qi) == 1 and "IRQI_RLAm_s0 actcons_c0 RLAm_s0 mix" in qi[0] and 'DR_MIX="head head head 0.5 --q-equals-head' in qi[0]
    c = [l for l in out if "cmp_dumps.py" in l]
    assert len(c) == 1 and "IRQI_RLAm_s0" in c[0] and "IRH_RLAm_s0" in c[0] and "differing=0" in c[0]


def test_order_and_slot_waits(tmp_path):
    out = dry(tmp_path)
    lines = out.splitlines()
    # a slot wait directly before every job: extract, lead read, fit, rollout
    jobs = [i for i, l in enumerate(lines) if l.startswith("DRY: ") and ("extract_states.py" in l or "pod_lead_apply.py" in l or "qhead_fit_n1.py" in l or "pod_ruleattr_run.sh" in l or "pod_integpre_run.sh" in l)]
    assert len(jobs) == 8 * 3 + 8 * 3 + 2
    for i in jobs:
        assert "slot wait before" in lines[i - 1], lines[i][:100]
    assert out.count("slot wait before") == len(jobs)
    # one checkpoint after the other: states < lead < fit < R0 < RL < RQ of base 0 C come before base 0 K
    pos = lambda s: out.find(s)  # noqa: E731
    assert pos("extract_states.py --tree") < pos("pod_lead_apply.py") < pos("--arm RLAm_s0 ") < pos("IR0_RLAm_s0 ") < pos("IRL_RLAm_s0 ") < pos("IRQ_RLAm_s0 ") < pos("IRH_RLAm_s0 ") < pos("IRQI_RLAm_s0 ") < pos("--arm RLFT_s0 ")
    assert pos("--arm RLAm_s3 ") < pos("--arm RLFT_s3 ")


def test_expected_volume_is_printed_and_checked(tmp_path):
    out = dry(tmp_path)
    m = re.search(r"expected volume: 8 checkpoints x \(state dump (\d+) \+ lead read (\d+)\) \+ (\d+) rollouts x (\d+) = (\d+) bytes", out)
    assert m, out[:400]
    s, l, n, rb, tot = map(int, m.groups())
    assert n == 26 and tot == 8 * (s + l) + n * rb and 4.5e9 < tot < 7e9
    assert "stat -f" in out and "refuse if free < expected + 10 GB" in out
    code = open(SCRIPT, encoding="utf-8").read()
    assert "REFUSED" in code and 'VOLFREE" -ge $(( EXPECT + VOL_MARGIN_GB * 1073741824 ))' in code


def test_second_start_and_missing_inputs_are_refused(tmp_path):
    d = tmp_path / "d"
    (d / "run").mkdir(parents=True)
    (d / "run" / "INTEGPRE_STARTED.txt").write_text("x")
    # a fake volume with plenty of space is not needed: the marker check comes after the volume check, so give the volume check room by pointing IP_W at the tmp folder
    r = run({"IP_W": sh(tmp_path), "IP_D": sh(d), "IP_ARMS": sh(ARMS), "IP_VOL_MARGIN_GB": "0"}, tmp_path)
    assert r.returncode == 2 and "REFUSED" in r.stdout and "INTEGPRE_STARTED.txt exists" in r.stdout


def test_volume_refusal(tmp_path):
    r = run({"IP_W": sh(tmp_path), "IP_D": sh(tmp_path / "d"), "IP_ARMS": sh(ARMS), "IP_VOL_MARGIN_GB": "99999999"}, tmp_path)
    assert r.returncode == 2 and "less than the expected" in r.stdout
    assert not os.path.exists(tmp_path / "d" / "run" / "INTEGPRE_STARTED.txt")
