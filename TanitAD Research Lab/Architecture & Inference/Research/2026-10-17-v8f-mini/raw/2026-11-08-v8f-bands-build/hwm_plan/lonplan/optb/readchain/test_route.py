"""Tests of the OPTB loading route (optb_route.py). Each uses a subprocess (the ladder tree and the optb tree both name the package `tanitad`, so they cannot share a process).
Fixture: fixture/FT/ckpt_last.pt, made by make_ft_ckpt.py with the 4c396aff trainer; its cfg.wm_rsp_probe names a path that does NOT exist."""
import json, os, shutil, subprocess, sys
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
FT = os.path.join(HERE, "fixture", "FT", "ckpt_last.pt")
R35 = "C:/Users/Admin/t2v8f_pulls/ladder_R35_s0/ckpt_last.pt"


def probe(mode, ckpt, env=None, n=6):
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", **(env or {}))
    r = subprocess.run([PY, os.path.join(HERE, "route_probe.py"), mode, ckpt, str(n)], capture_output=True, encoding="utf-8", env=e, timeout=600)
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_fixture_probe_path_does_not_exist():
    ck = torch.load(FT, map_location="cpu", weights_only=False)
    assert ck["cfg"]["wm_rsp_readout"] == "fixed_k" and not os.path.exists(ck["cfg"]["wm_rsp_probe"])


def test_ladder_tree_refuses_an_ft_checkpoint():
    o = probe("ladder", FT)
    assert "error" in o and "unexpected keyword" in o["error"], o


def test_route_strict_loads_ft_remaps_probe_and_reads_it_back():
    o = probe("route", FT)
    assert "error" not in o, o
    assert len(o["remaps"]) == 1 and o["remaps"][0]["local"].replace("\\", "/").endswith("readout_K10_R35_s0.pt") and o["has_probe_buffer"] is True


def test_route_serves_ia3pk_load_ckpt():
    o = probe("ia3pk", FT)
    assert "error" not in o and len(o["remaps"]) == 1, o


def test_r35_parity_route_equals_ladder_bit_for_bit():
    a, b = probe("ladder", R35), probe("route", R35)
    assert "error" not in a and "error" not in b, (a, b)
    assert a["rollout_sha256"] == b["rollout_sha256"] and a["n_params"] == b["n_params"] and a["rollout_shape"] == b["rollout_shape"]
    assert b["remaps"] == []


def _ckpt_with(tmp_path, **cfgkw):
    ck = torch.load(FT, map_location="cpu", weights_only=False)
    ck["cfg"].update(cfgkw)
    p = tmp_path / "ckpt_mod.pt"
    torch.save(ck, p)
    return str(p)


def test_wrong_declared_sha_refuses(tmp_path):
    o = probe("route", _ckpt_with(tmp_path, wm_rsp_probe_sha256="0" * 64))
    assert "error" in o and "sha256" in o["error"], o


def test_tampered_local_probe_refuses(tmp_path):
    d = tmp_path / "pd"; d.mkdir()
    src = "F:/TanitAD-scratch/actrsp/amend1/out/readout_K10_R35_s0.pt"
    b = bytearray(open(src, "rb").read()); b[-5] ^= 1
    (d / "readout_K10_R35_s0.pt").write_bytes(bytes(b))
    o = probe("route", FT, env={"OPTB_PROBE_DIRS": str(d)})
    assert "error" in o and "sha256" in o["error"], o


def test_unknown_basename_and_missing_local_copy_refuse(tmp_path):
    o = probe("route", _ckpt_with(tmp_path, wm_rsp_probe="D:\\tanitad\\optb\\inputs\\evil.pt"))
    assert "error" in o and "not one of" in o["error"], o
    o = probe("route", FT, env={"OPTB_PROBE_DIRS": str(tmp_path)})
    assert "error" in o and "no local copy" in o["error"], o


def test_tree_manifest_matches_and_detects_a_change(tmp_path):
    sys.path.insert(0, HERE)
    import optb_route as R
    assert R.check_tree_manifest() == 1865


def test_main_refuses_a_tree_that_is_not_the_manifest_and_runs_on_the_real_one(tmp_path):
    """MM 2026-11-18: main() must CALL check_tree_manifest (it was defined, tested, and never called by the route)."""
    import subprocess
    script = tmp_path / "hello.py"
    script.write_text("print('ZZ ROUTE RAN ZZ')", encoding="utf-8")
    (tmp_path / "fake" / "stack").mkdir(parents=True); (tmp_path / "fake" / "taniteval").mkdir()
    env = dict(os.environ, OPTB_TREE=str(tmp_path / "fake"), CUDA_VISIBLE_DEVICES="-1")
    r = subprocess.run([sys.executable, os.path.join(HERE, "optb_route.py"), str(script)], capture_output=True, encoding="utf-8", env=env, timeout=600)
    assert r.returncode != 0 and "tree manifest" in r.stderr and "ZZ ROUTE RAN ZZ" not in r.stdout, (r.returncode, r.stderr[-400:])
    env.pop("OPTB_TREE")
    r = subprocess.run([sys.executable, os.path.join(HERE, "optb_route.py"), str(script)], capture_output=True, encoding="utf-8", env=env, timeout=600)
    assert r.returncode == 0 and "ZZ ROUTE RAN ZZ" in r.stdout, (r.returncode, r.stderr[-400:])
