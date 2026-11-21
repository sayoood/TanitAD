"""Dry-run of chain_trainfit_pf.sh with STUB tools (no GPU, no lock file, no roll): the control flow, the gates, the one-retry rule, resumability.

Every expectation is a LITERAL marker / file the chain must (or must not) produce.  Mutation arms: a seed that fails twice, a FAIL parity, a wrong
checkpoint md5 -- each must stop the chain BEFORE the freeze (no frozen file, no FROZEN marker).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1]
CHAIN = CODE / "chain_trainfit_pf.sh"
SH = shutil.which("sh") or "C:/Program Files/Git/usr/bin/sh.exe"
PY = os.environ.get("PF_TEST_PY", "C:/Users/Admin/venvs/tanitad/Scripts/python.exe")
V9 = "E:/refcv8_eval_kit/data/refcv8_v9labels/release_train4572/v9_labels_train.npz"

STUB_LOCK = r'''
import json, sys, os
a = sys.argv[1:]; cmd = a[0]; d = dict(zip(a[1::2], a[2::2]))
log = os.environ["STUB_LOG"]
open(log, "a").write(f"gpu_lock {cmd} {d.get('--job')}\n")
if cmd == "acquire" and "--out" in d:
    json.dump({"ok": os.environ.get("STUB_LOCK_FAIL") != "1"}, open(d["--out"], "w"))
'''
STUB_MARK = r'''
import sys, os
open(os.environ["STUB_LOG"], "a").write("chain_marker " + " ".join(sys.argv[1:2]) + "\n")
'''
STUB_ROLL = r'''
import json, sys, os
a = sys.argv[1:]; d = dict(zip(a[0::2], a[1::2]))
s = d["--seed"]
open(os.environ["STUB_LOG"], "a").write(f"roll seed {s}\n")
if s in os.environ.get("STUB_ROLL_FAIL_SEEDS", "").split(","):
    sys.exit(1)
os.makedirs(d["--dump-dir"], exist_ok=True)
json.dump({"grid": {"n_windows": 4855}}, open(os.path.join(d["--dump-dir"], "manifest.json"), "w"))
json.dump({"smoke": False, "n_windows": 4855, "n_episodes": 139, "device": "cuda", "wall_s": 1.0,
           "pf": {"ok": True, "census": {"n_args_known": 7, "n_token_differs_from_clip": 3}}}, open(d["--out-json"], "w"))
'''
STUB_PAR = r'''
import json, sys, os
a = sys.argv[1:]; d = {a[i]: a[i + 1] for i in range(len(a) - 1) if a[i].startswith("--")}
st = "FAIL" if os.environ.get("STUB_PARITY_FAIL") == "1" else "PASS"
json.dump({"status": st, "label": d.get("--label"), "counts": {"n_windows": 4855}, "checks": {"T": st == "PASS"}}, open(d["--out"], "w"))
'''
STUB_FIT = r'''
import json, sys, os, hashlib
cmd = sys.argv[1]
tfl, out = os.environ["STUB_TFL"], os.environ["STUB_OUT"]
open(os.environ["STUB_LOG"], "a").write(f"fit {cmd}\n")
if cmd == "panels":
    for s in (0, 1):
        for p in (f"panel_s{s}", f"s6_s{s}"):
            os.makedirs(os.path.join(tfl, p), exist_ok=True); json.dump({}, open(os.path.join(tfl, p, "manifest.json"), "w"))
elif cmd == "freeze":
    os.makedirs(out, exist_ok=True)
    fz = os.path.join(out, "FROZEN_GAINS_RETIME_train_pf.json")
    json.dump({"tree": "stub", "seeds_pooled": [0, 1], "prior": "ha0_ext_pose", "n_windows_by_surface": {}, "n_compliant_by_surface": {}}, open(fz, "w"))
    sh = hashlib.sha256(open(fz, "rb").read()).hexdigest()
    open(os.path.join(out, "FROZEN_GAINS_RETIME_train_pf.sha256"), "w").write(sh + "\n")
    g = {"S2": {"instants_s": [0.5], "RETIME": [0.9]}, "S6": {"instants_s": [1.0], "RETIME": [0.6]}}
    json.dump({"sha256": sh, "fit_code": {"sha256": "x", "prereg_sha256": "y", "sha256_at_clip_fed_freeze": "z", "diff_vs_clip_fed_freeze": "role"},
               "fit_guard": {}, "feed_parity": [], "rolls": {}, "gains_pf": g, "beside_clip_fed_table": {"gains": g}},
              open(os.path.join(out, "FROZEN_GAINS_RETIME_train_pf.provenance.json"), "w"))
'''


def _w(p: Path, txt: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt, encoding="utf-8")
    return p


@pytest.fixture()
def rig(tmp_path):
    bat = tmp_path / "bat"
    _w(bat / "gpu_lock.py", STUB_LOCK)
    _w(bat / "chain_marker.py", STUB_MARK)
    stubs = tmp_path / "stubs"
    _w(stubs / "roll.py", STUB_ROLL)
    _w(stubs / "par.py", STUB_PAR)
    _w(stubs / "fit.py", STUB_FIT)
    hold = tmp_path / "hold"
    (hold / "code").mkdir(parents=True)
    shutil.copy(CODE / "trainfit_pf_report.py", hold / "code" / "trainfit_pf_report.py")
    clip = tmp_path / "clipref"
    for s in (0, 1):
        _w(clip / f"roll_s{s}.json", json.dumps({"n_windows": 4855}))
    ck = _w(tmp_path / "ckpt" / "ckpt_ema_30000.pt", "weights")
    md5 = hashlib.md5(b"weights").hexdigest()
    _w(tmp_path / "ckpt" / "ckpt_30000.md5", f"0123  ckpt_30000.pt\n{md5}  ckpt_ema_30000.pt\n")
    root = tmp_path / "root"
    tree = root / "trainfit" / "step30000ema_train139_pf"

    def m(p):
        return str(p).replace("\\", "/")
    env = dict(os.environ)
    env.update({"TFPF_HOLDW": m(hold), "TFPF_HOLDL": m(hold), "TFPF_BAT": m(bat), "TFPF_R8CODE": m(tmp_path), "TFPF_PY": PY,
                "TFPF_ROOTW": m(root), "TFPF_ROOTL": m(root), "TFPF_CLIPREF_L": m(clip), "TFPF_CLIPREF_W": m(clip),
                "TFPF_CKPT": m(ck), "TFPF_CKPT_L": m(ck), "TFPF_CKMD5F": m(tmp_path / "ckpt" / "ckpt_30000.md5"), "TFPF_V9": V9,
                "TFPF_ROLLPY": m(stubs / "roll.py"), "TFPF_PARPY": m(stubs / "par.py"), "TFPF_FITPY": m(stubs / "fit.py"),
                "TFPF_LOG": m(tmp_path / "chain.log"), "TFPF_FREEGB_CMD": "echo 50", "TFPF_GATE_GAP_S": "0",
                "STUB_LOG": m(tmp_path / "stub.log"), "STUB_TFL": m(tree), "STUB_OUT": m(hold / "raw" / "retime_trainfit_pf")})
    return {"env": env, "tmp": tmp_path, "hold": hold, "tree": tree}


def run(rig, **extra):
    env = dict(rig["env"])
    env.update(extra)
    subprocess.run([SH, str(CHAIN).replace("\\", "/")], env=env, timeout=600, stdin=subprocess.DEVNULL, capture_output=True)
    log = (rig["tmp"] / "chain.log").read_text(encoding="utf-8") if (rig["tmp"] / "chain.log").exists() else ""
    stub = (rig["tmp"] / "stub.log").read_text(encoding="utf-8") if (rig["tmp"] / "stub.log").exists() else ""
    return log, stub


def frozen(rig) -> Path:
    return rig["hold"] / "raw" / "retime_trainfit_pf" / "FROZEN_GAINS_RETIME_train_pf.json"


def test_happy_path_then_resume(rig):
    log, stub = run(rig)
    for m in ("ZZTFPFCKPTOKZZ", "ZZTFPFV9OKZZ", "ZZTFPFLOCKEDZZ", "ZZTFPFROLL_s0_OKZZ", "ZZTFPFROLL_s1_OKZZ", "ZZTFPFRELEASEDZZ",
              "ZZTFPFPARITY_s0_PASSZZ", "ZZTFPFPARITY_s1_PASSZZ", "ZZTFPFPANELSOKZZ", "ZZTFPFFROZENZZ", "ZZTFPFREPORTOKZZ", "ZZTFPFENDZZ"):
        assert m in log, m
    assert stub.count("gpu_lock acquire refcv8-trainfit-pf-48852") == 1          # ONE hold for both seeds
    assert stub.count("gpu_lock release refcv8-trainfit-pf-48852") == 1
    assert stub.index("roll seed 1") < stub.index("gpu_lock release")             # both rolls inside the hold
    assert frozen(rig).exists()
    sh = hashlib.sha256(frozen(rig).read_bytes()).hexdigest()
    assert sh in (rig["hold"] / "TRAINFIT_PF_FROZEN.md").read_text(encoding="utf-8")
    # resume: nothing is rolled or re-frozen, the GPU is not taken
    (rig["tmp"] / "chain.log").unlink()
    (rig["tmp"] / "stub.log").unlink()
    log2, stub2 = run(rig)
    assert log2.count("ZZTFPFROLLHAVEZZ") == 2 and "ZZTFPFPARITYHAVEZZ" in log2 and "ZZTFPFPANELSHAVEZZ" in log2 and "ZZTFPFFROZENHAVEZZ" in log2
    assert "ZZTFPFLOCKEDZZ" not in log2 and "gpu_lock acquire" not in stub2 and "roll seed" not in stub2 and "fit freeze" not in stub2


def test_mutation_seed_fails_twice_halts_before_freeze(rig):
    log, stub = run(rig, STUB_ROLL_FAIL_SEEDS="1")
    assert stub.count("roll seed 1") == 2                                           # exactly one retry
    assert "ZZTFPFROLLRETRYZZ" in log and "ZZTFPFROLLGIVEUPZZ" in log
    assert "ZZTFPFFROZENZZ" not in log and "ZZTFPFENDZZ" not in log and not frozen(rig).exists()
    assert "gpu_lock release refcv8-trainfit-pf-48852" in stub                      # the EXIT trap releases the lock


def test_mutation_parity_fail_halts_before_freeze(rig):
    log, stub = run(rig, STUB_PARITY_FAIL="1")
    assert "ZZTFPFPARITYFAILZZ" in log
    assert "fit freeze" not in stub and "fit panels" not in stub and not frozen(rig).exists()


def test_mutation_wrong_ckpt_md5_never_takes_the_gpu(rig):
    (rig["tmp"] / "ckpt" / "ckpt_ema_30000.pt").write_text("other weights")
    log, stub = run(rig)
    assert "ZZTFPFCKPTBADZZ" in log
    assert "gpu_lock acquire" not in stub and "roll seed" not in stub and not frozen(rig).exists()


def test_mutation_lock_not_acquired_no_roll(rig):
    log, stub = run(rig, STUB_LOCK_FAIL="1")
    assert "ZZTFPFNOLOCKZZ" in log and "roll seed" not in stub and not frozen(rig).exists()
