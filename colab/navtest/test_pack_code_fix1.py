"""NT-FIX1: ``pack_code.py`` must ship ``run_v1_fix1.py`` beside ``run_v1.py`` in the bundle's ``w3code/``.

``linux_run_v1.py`` now loads W3's harness through ``run_v1_fix1.load()`` and exits if the file is missing, so a
bundle without it is dead on arrival. The packer is run against a FAKE tree (monkeypatched module paths; no D:
files, no frames, no network) and its tarball + ``code_manifest.json`` are read back. Pointed at the pre-NT-FIX1
packer (``PACK_CODE=<path>``) the tests go RED -- that is the mutation arm. The last test closes the chain with
the REAL W3 pair: pack -> extract ``w3code/`` -> ``linux_run_v1.load_w3`` returns the byte-verified, patched module.

    python -m pytest -q colab/navtest/test_pack_code_fix1.py
    PACK_CODE=<pre-NT-FIX1 pack_code.py> python -m pytest -q colab/navtest/test_pack_code_fix1.py   # RED
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tarfile
import zipfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CALLER = os.environ.get("PACK_CODE", os.path.join(HERE, "pack_code.py"))
LAUNCHER = os.environ.get("LINUX_RUN_V1", os.path.join(HERE, "linux_run_v1.py"))
REAL_W3 = os.path.join(REPO, "FlyWheels", "TanitAD_EvalFlyWheel", "incoming", "2026-09-19-navsim-v1-navtest", "code")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _touch(path, text="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return path


def _fake_tree(tmp_path, with_fix1=True):
    """Every path pack_code.main reads, under tmp. Returns (module-global patches, evidence dir, W3 dir)."""
    t = str(tmp_path / "tree")
    colab, here = f"{t}/colab", f"{t}/colab/navtest"
    for n in ("vm_p0.py", "linux_run_v1.py", "navsim_v1_linux.py", "e6_linux.py", "seam_shards.py",
              "zip_member_fetch.py"):
        _touch(f"{here}/{n}", f"# fake {n}\n")
    _touch(f"{colab}/signed_pull.py")
    w3 = f"{t}/w3code_src"
    _touch(f"{w3}/run_v1.py", "# FAKE run_v1\n")
    _touch(f"{w3}/navsim_v1_win.py")
    _touch(f"{w3}/w3_agents_v1.py")
    if with_fix1:
        _touch(f"{w3}/run_v1_fix1.py", "# FAKE run_v1_fix1\n")
    _touch(f"{t}/e1/navsim_win.py")
    pkg = f"{t}/refe-plan"
    for m in ("planner", "model", "ckpt_io", "load_dinov3", "navtrain_scenarios", "calib_table"):
        _touch(f"{pkg}/refe/{m}.py")
    _touch(f"{pkg}/code/augment_routes.py")
    _touch(f"{pkg}/eval/refe_navtest_seam.py")
    plan = f"{colab}/raw/2026-09-27-navtest-plan"
    _touch(f"{plan}/navsim_crun_venv_freeze.txt")
    _touch(f"{plan}/driverl_eval_venv_freeze.txt")
    _touch(f"{plan}/jpeg_devbox_hashes.json", json.dumps({"token": "tok0", "log": "log0", "hashes": ["a.jpg"]}))
    frames = f"{t}/frames"
    _touch(f"{frames}/tokens_frames.json", json.dumps([{"token": "tok0", "cams": ["x/CAM_F0/a.jpg"]}]))
    with zipfile.ZipFile(f"{frames}/log0_CAM_F0.zip", "w") as z:
        z.writestr("a.jpg", b"jpeg-bytes")
    ev = f"{t}/evidence"
    _touch(f"{ev}/db_manifest_sub200.json", "{}")
    _touch(f"{ev}/maps_manifest.json", "{}")
    patches = dict(HERE=here, COLAB=colab, PKG=pkg, W3C=w3, E1=f"{t}/e1/navsim_win.py", PLAN=plan, FRAMES=frames)
    return patches, ev, w3


def _pack(monkeypatch, tmp_path, patches, ev):
    pc = _load(CALLER, "pack_code_under_test")
    for k, v in patches.items():
        monkeypatch.setattr(pc, k, v)
    out = str(tmp_path / "bundle.tgz")
    monkeypatch.setattr(sys, "argv", ["pack_code.py", "--out", out, "--evidence", ev])
    return pc, out


def test_bundle_carries_run_v1_fix1_beside_run_v1(monkeypatch, tmp_path, capsys):
    patches, ev, w3 = _fake_tree(tmp_path)
    pc, out = _pack(monkeypatch, tmp_path, patches, ev)
    assert pc.main() == 0
    with tarfile.open(out) as tf:
        names = tf.getnames()
        assert "w3code/run_v1.py" in names
        assert "w3code/run_v1_fix1.py" in names, "the bundle ships run_v1.py WITHOUT run_v1_fix1.py"
        assert tf.extractfile("w3code/run_v1_fix1.py").read() == open(f"{w3}/run_v1_fix1.py", "rb").read()
    man = json.load(open(f"{ev}/code_manifest.json", encoding="utf-8"))["files"]
    want = hashlib.sha256(open(f"{w3}/run_v1_fix1.py", "rb").read()).hexdigest()
    assert man["w3code/run_v1_fix1.py"]["sha256"] == want
    assert "ZZCODE_OK" in capsys.readouterr().out


def test_missing_run_v1_fix1_stops_the_packer(monkeypatch, tmp_path):
    """A bundle must never be built without it (linux_run_v1.py would exit on the VM, after the upload)."""
    patches, ev, _ = _fake_tree(tmp_path, with_fix1=False)
    pc, out = _pack(monkeypatch, tmp_path, patches, ev)
    with pytest.raises(FileNotFoundError):
        pc.main()


def test_chain_pack_extract_load_w3_reaches_the_patched_module(monkeypatch, tmp_path):
    """pack (REAL W3 pair) -> extract w3code/ -> linux_run_v1.load_w3: the loader finds fix1 in the packed dir."""
    if not (os.path.isfile(os.path.join(REAL_W3, "run_v1.py"))
            and os.path.isfile(os.path.join(REAL_W3, "run_v1_fix1.py"))):
        pytest.skip(f"real W3 pair not in this checkout: {REAL_W3}")
    patches, ev, _ = _fake_tree(tmp_path)
    patches["W3C"] = REAL_W3
    pc, out = _pack(monkeypatch, tmp_path, patches, ev)
    assert pc.main() == 0
    dest = tmp_path / "vm"
    with tarfile.open(out) as tf:
        for m in tf.getmembers():
            if m.name.startswith("w3code/"):      # what the VM unpacks: plain files only in this tar
                (dest / m.name).parent.mkdir(parents=True, exist_ok=True)
                (dest / m.name).write_bytes(tf.extractfile(m).read())
    launcher = _load(LAUNCHER, "linux_run_v1_for_pack_chain")
    a = argparse.Namespace(w3_code=str(dest / "w3code"), nv_py="NVPY", v11_tree="V11", exp=str(tmp_path / "exp"),
                           openscene="OSC", maps="MAPS", e1_wrapper="E1.py", nuplan_tree="NUP",
                           windowspath_shim=False, out=str(tmp_path / "out"))
    w3 = launcher.load_w3(a)
    assert w3.FIX1["original"]["matches_pinned"] is True
    assert w3.cmd_score.__qualname__.startswith("_wrap_cmd_score")
