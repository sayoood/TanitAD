# test_rolltrace.py -- synthetic data only (no real dump, rollout or result file is opened).
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import export_traces as X  # noqa: E402

BANNED = tuple(w.replace("_", "") for w in ("vo_id", "super_seded", "re_tracted", "for_merly", "cor_rection", "with_drawn", "no_ longer", "closed_ loop"))   # written in parts: the hold scan reads this file too
FILES = ("export_traces.py", "podrun_export_traces.sh", "make_fig_rolltrace.py", "push_rolltrace.sh", "pull_rolltrace.sh", "ROLLTRACE_DEFS.md", "test_rolltrace.py")


@pytest.fixture(scope="module")
def eng():
    return X.get_engine("")


def _run(eng, tmp_path, fault):
    R, H = eng
    root = str(tmp_path)
    exp, nl = X.make_synth(root, fault)
    a = types.SimpleNamespace(pull=os.path.join(root, "pull"), grid_dir=os.path.join(root, "grid"), join=os.path.join(root, "join.npz"))
    return X.run(R, H, a, expect=exp, n_lead=nl, n_win=40), exp


def test_clean_run_passes_and_selects_by_sha(eng, tmp_path):
    (rc, line, rec), exp = _run(eng, tmp_path, False)
    assert rc == 0 and line == "ZZ EXPORT-TRACES t0 PASS windows 6"
    assert rec["n_windows_the_rule_chose_from"] == 6
    hx = [w["key_sha256_16"] for w in rec["windows"]]
    assert hx == sorted(hx)
    # the order is the sha256 hex of "<seed>|<clip id>/<window start>"
    full = [hashlib.sha256(("%d|%s/%d" % (w["seed"], {"aaaaaaaa": "aaaaaaaa-0000"}[w["clip8"]], w["ws"])).encode()).hexdigest()[:16] for w in rec["windows"]]
    assert full == hx
    t = rec["table_2x2_contact"]
    assert t["pooled"] == {"rr_yes_vd5c_yes": 8, "rr_yes_vd5c_no": 6, "rr_no_vd5c_yes": 0, "rr_no_vd5c_no": 26}
    assert t["0"]["rr_yes_vd5c_no"] == 3 and t["1"]["rr_yes_vd5c_no"] == 3
    for w in rec["windows"]:
        assert w["forms"]["RR"]["C"] == 1.0 and w["forms"]["VD5c"]["C"] == 0.0 and w["forms"]["RLc"]["C"] == 0.0
        assert len(w["t_s"]) == 30 and abs(w["t_s"][-1] - 6.0) < 1e-9
        # synthetic design: gap = 15 + 2k - 4k for a rolled ego at 20 m/s, 15 + 2k - 2k for 10 m/s
        assert abs(w["forms"]["RR"]["gap_m"][7] - (15 + 16 - 32)) < 1e-3 and abs(w["forms"]["VD5c"]["gap_m"][7] - 15.0) < 1e-3
        assert abs(w["forms"]["RR"]["speed_mps"][3] - 20.0) < 1e-3


def test_planted_fault_fails_t0_and_writes_nothing(eng, tmp_path):
    (rc, line, rec), _ = _run(eng, tmp_path, True)
    assert rc == 3 and rec is None and line.startswith("ZZ EXPORT-TRACES t0 FAIL rate-RR")


def test_wrong_lead_count_fails(eng, tmp_path):
    R, H = eng
    root = str(tmp_path)
    exp, nl = X.make_synth(root, False)
    a = types.SimpleNamespace(pull=os.path.join(root, "pull"), grid_dir=os.path.join(root, "grid"), join=os.path.join(root, "join.npz"))
    rc, line, rec = X.run(R, H, a, expect=exp, n_lead=nl + 1, n_win=40)
    assert rc == 3 and "n-lead" in line and rec is None


def test_json_is_small_and_has_no_path_or_address(eng, tmp_path):
    (rc, line, rec), _ = _run(eng, tmp_path, False)
    s = json.dumps(rec)
    assert len(s) < 200_000
    assert "/workspace" not in s and ":\\" not in s and not re.search(r"\d+\.\d+\.\d+\.\d+", s.replace("0.", ""))
    assert all(len(w["clip8"]) == 8 for w in rec["windows"])


def test_figure_renders_from_synthetic_json(eng, tmp_path):
    (rc, line, rec), _ = _run(eng, tmp_path, False)
    p = os.path.join(str(tmp_path), "ROLLTRACE.json")
    json.dump(rec, open(p, "w"))
    out = os.path.join(HERE, "tmp_pt", "fig_" + os.path.basename(str(tmp_path)))
    import make_fig_rolltrace as F
    F.main(["--in", p, "--out-dir", out])
    for ext in ("png", "pdf"):
        assert os.path.getsize(os.path.join(out, "fig_rolltrace." + ext)) > 5000


def test_scripts_syntax_and_line_ends():
    bash = shutil.which("bash")
    for f in FILES:
        b = open(os.path.join(HERE, f), "rb").read()
        if f.endswith(".sh"):
            assert b"\r" not in b, f
            assert subprocess.run([bash, "-n", os.path.join(HERE, f)], capture_output=True, encoding="utf-8").returncode == 0, f
        t = b.decode("utf-8").lower()
        if f != "test_rolltrace.py":
            for w in BANNED:
                assert w not in t, (f, w)
            for m in re.finditer(r"driving performance", t):
                assert t[max(0, m.start() - 4):m.start()] == "not ", f


def test_dry_mode_makes_nothing():
    r = subprocess.run([shutil.which("bash"), os.path.join(HERE, "podrun_export_traces.sh")], env=dict(os.environ, DRY="1"), capture_output=True, encoding="utf-8")
    assert r.returncode == 0 and r.stdout.strip().endswith("ZZ PODRUN-ROLLTRACE DRY") and "refused if it exists" in r.stdout


def test_path_lengths():
    for f in FILES:
        assert len(os.path.join(HERE, f).replace("\\", "/")) <= 200
