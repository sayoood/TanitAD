"""test_r8v_feed_parity.py -- the VIDEO path feeds and shows the nav the binding per-frame battery fed (R8-EVAL-NAV-FEED-1 / R57).  CPU, no model.

Fixed window set = the pf CPU-smoke battery dump (run_battery_r8_pf.py -> roll_seed_r8_pf.py, EMA-40k, 2 clips x 3 windows, MEASURED 2026-10-20 07:34):
    C:/Users/Admin/refcv8_pf/smoke_batt/smoke40kema_pf/dump_s0     (its own feed_parity_pf.py read PASS: hold_r8pf raw/smoke/FEED_PARITY_cpusmoke_battery_s0.json)
LITERALS (read off that artifact by hand, then frozen here -- never computed by the code under test):
    ep000 ws 7, 12, 17 : FED token 0,0,0 (FOLLOW_ROAD), CLIP token 2,2,2 (TURN_R)      <- 3 DISCRIMINATING windows (a clip-fed path reads TURN_R)
    ep001 ws 11, 16, 21: FED token 0,0,0,           CLIP token 0,0,0
    ep000 ws 7 r8_nav  : [0.92943823, 1.1067319, -0.7211111, -1.0, 3.0445225, 1.0]      (args known)
    patched refcv3_arm sha256 64e6ffbb... == that battery's roll_s0.json pf.patched_refcv3_arm.patched_sha256

MUTATION ARM:  R8V_MUTATE_FEED=clip  python -m pytest ...   ->  test_a_*, test_b_*, test_c_* MUST FAIL (the R57 defect: no overlay, clip token shown as fed).
test_d_* carries its own deliberate-regression artifact (a heads dump fed the clip token) and is RED-by-construction on it.
A missing input is a FAILURE, never a skip.

    set PYTHONPATH=C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval      (test_c loads the launch tree's refcv3_arm through the registered overlay)
    python -m pytest -q -p no:cacheprovider test_r8v_feed_parity.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r8v_feed as F  # noqa: E402
import r8v_data as D  # noqa: E402

SMOKE_ROOT = Path("C:/Users/Admin/refcv8_pf/smoke_batt")
SMOKE_TREE = "smoke40kema_pf"
SMOKE_DUMP = SMOKE_ROOT / SMOKE_TREE / "dump_s0"
V9 = Path(F.V9_EVAL)
FED_LIT = {(0, 7): 0, (0, 12): 0, (0, 17): 0, (1, 11): 0, (1, 16): 0, (1, 21): 0}
CLIP_LIT = {(0, 7): 2, (0, 12): 2, (0, 17): 2, (1, 11): 0, (1, 16): 0, (1, 21): 0}
R8_EP0_WS7 = [0.92943823, 1.1067319, -0.7211111, -1.0, 3.0445225, 1.0]
PATCHED_SHA_LIT = "64e6ffbb1d0434d13badf82f39daf46608ad8404e2ec9e8797c375be5e5df543"
OVERLAY_SHA_LIT = "98293dd7411270160cb47d070bb1538babe1ff3a0f8ab85da61dcc84129ba72f"
B208 = "b208c9c1e4236075de107ef49eaef39c1dddddba257f6e594bf6b84f8a24b1ee"


def test_inputs_present():
    for p in (SMOKE_DUMP / "manifest.json", SMOKE_DUMP / "decisions/ep000.npz", SMOKE_ROOT / SMOKE_TREE / "battery_summary.json", V9, F.PF_CODE / "feed_parity_pf.py"):
        assert p.exists(), f"missing input {p} (a FAILURE, not a skip)"


# ===================================================================================== (a) the renderer reads the FED token from the artifact
def test_a_dump_reader_returns_the_fed_token_not_the_clip_token():
    df = F.read_dump_feed(SMOKE_DUMP)
    assert df["mode"] == "perframe_full"
    got = {(x["episode_index"], x["ws"]): x["nav_cmd_fed"] for x in df["rows"]}
    clip = {(x["episode_index"], x["ws"]): x["nav_cmd_clip"] for x in df["rows"]}
    assert got == FED_LIT, f"fed tokens {got} != literal {FED_LIT}"
    assert clip == CLIP_LIT
    r = next(x for x in df["rows"] if (x["episode_index"], x["ws"]) == (0, 7))
    assert r["r8_nav_known"] is True and np.allclose(r["r8_nav"], R8_EP0_WS7, atol=1e-6)


def test_a_renderer_bank_shows_fed_and_clip_nav_on_the_pf_tree():
    B = D.EvalBank(SMOKE_TREE, arch_root=SMOKE_ROOT, load_agents=False)
    assert B.feed_mode == "perframe_full"
    e = B.episode(0)
    assert list(e["ws"]) == [7, 12, 17]
    assert list(e["nav_fed"]) == [0, 0, 0] and list(e["nav_cmd"]) == [0, 0, 0], "the renderer must show the FED (per-frame) token"
    assert list(e["nav_clip"]) == [2, 2, 2]
    fed_txt, clip_txt = F.label(int(e["nav_fed"][0]), int(e["nav_clip"][0]), bool(e["r8_nav_known"][0]), e["nav_feed"])
    assert fed_txt.startswith("FOLLOW_ROAD") and "per-frame v9" in fed_txt
    assert clip_txt.startswith("TURN_R") and "differs" in clip_txt


# ===================================================================================== (b) == the pf battery sidecar AND == the independent v9 lookup
def test_b_video_feed_equals_pf_battery_and_independent_v9():
    B = D.EvalBank(SMOKE_TREE, arch_root=SMOKE_ROOT, load_agents=False)
    rows = []
    for ci in B.indices():
        e = B.episode(ci)
        for j, w in enumerate(e["ws"]):
            rows.append({"sha12": B.sha12(ci), "episode_index": ci, "ws": int(w), "nav_cmd_fed": int(e["nav_fed"][j]), "nav_cmd_clip": int(e["nav_clip"][j]),
                         "r8_nav_known": bool(e["r8_nav_known"][j]), "r8_nav": e["r8_nav_fed"][j]})
    r = F.check_feed_parity(rows, SMOKE_DUMP, V9, F.V9_EVAL_MD5, "renderer path vs smoke pf battery")
    assert r["status"] == "PASS", json.dumps(r, default=str)[:1500]
    assert r["counts"]["n_windows"] == 6 and r["counts"]["n_discriminating_windows"] == 3


# ===================================================================================== (c) the heads dump's WIRING == the pf battery's wiring
def test_c_install_feed_is_the_pf_battery_wiring():
    import tanitad
    assert "Lfb47a58376" in str(Path(tanitad.__file__).resolve()), f"tanitad imported from {tanitad.__file__}, not the launch tree"
    RR = types.SimpleNamespace(_RA3=None)
    rec = F.install_feed(RR, "perframe_full")
    assert rec["nav_feed"] == "perframe_full" and rec["mutated"] is False
    assert rec["overlay"]["navfeed_overlay_sha256"] == OVERLAY_SHA_LIT
    assert rec["patched_refcv3_arm"]["patched_sha256"] == PATCHED_SHA_LIT
    roll = json.load(open(SMOKE_ROOT / SMOKE_TREE / "roll_s0.json", encoding="utf-8"))
    assert roll["pf"]["patched_refcv3_arm"]["patched_sha256"] == PATCHED_SHA_LIT, "the pf battery ran another patched refcv3_arm"
    R = RR._RA3
    assert R is not None and R.__navfeed__["patches"] == ["P1-argparse", "P2-nav_true", "P3-item-assert", "P4-model-call", "P5-sidecar", "P6-manifest"]
    got = []
    R._r8v_orig_main = lambda av: got.append(list(av))
    R.main(["--ckpt", "x", "--no-navshuf"])
    assert got == [["--ckpt", "x", "--no-navshuf", "--nav-feed", "perframe_full"]]
    with pytest.raises(SystemExit):
        R.main(["--nav-feed", "clip"])


def test_c_clip_feed_installs_nothing():
    RR = types.SimpleNamespace(_RA3=None)
    rec = F.install_feed(RR, "clip")
    assert rec["nav_feed"] == "clip" and rec["argv_tail"] == [] and RR._RA3 is None
    with pytest.raises(SystemExit):
        F.install_feed(RR, "perframe_token")


# ===================================================================================== (d) the post-roll guard on a heads artifact (+ its deliberate regression)
def _write_heads(d: Path, mutant: bool):
    z0 = np.load(SMOKE_DUMP / "decisions/ep000.npz")
    z1 = np.load(SMOKE_DUMP / "decisions/ep001.npz")
    for ci, z in ((0, z0), (1, z1)):
        fed = np.asarray(z["nav_cmd_clip"] if mutant else z["nav_cmd"], np.int64)       # mutant: the R57 artifact -- the clip token was fed
        r8 = np.zeros((len(z["ws"]), 6), np.float32) if mutant else np.asarray(z["r8_nav_fed"], np.float32).reshape(-1, 6)
        np.savez_compressed(d / f"heads_ep{ci:03d}.npz", ws=np.asarray(z["ws"], np.int64), nav_feed=np.array("clip" if mutant else "perframe_full"),
                            nav_cmd_fed=fed, nav_cmd_clip=np.asarray(z["nav_cmd_clip"], np.int64), r8_nav=r8,
                            r8_nav_known=np.zeros(len(z["ws"]), bool) if mutant else np.asarray(z["r8_nav_known"], bool))


def _by():
    return {int(k): v for k, v in json.load(open(SMOKE_ROOT / SMOKE_TREE / "battery_summary.json", encoding="utf-8"))["stages"]["roll_s0"]["extras"]["clip_sha12_by_index"].items()}


def test_d_heads_guard_passes_a_pf_heads_artifact(tmp_path):
    _write_heads(tmp_path, mutant=False)
    r = F.check_feed_parity(F.rows_from_heads(tmp_path, _by()), SMOKE_DUMP, V9, F.V9_EVAL_MD5, "heads (pf)")
    assert r["status"] == "PASS", json.dumps(r, default=str)[:1500]


def test_d_heads_guard_goes_red_on_a_clip_fed_heads_artifact(tmp_path):
    _write_heads(tmp_path, mutant=True)
    r = F.check_feed_parity(F.rows_from_heads(tmp_path, _by()), SMOKE_DUMP, V9, F.V9_EVAL_MD5, "heads (clip-fed mutant)")
    assert r["status"] == "FAIL"
    assert r["counts"]["n_fail"] == {"PF": 6, "V9": 6}, r["counts"]       # all 6: token on ep000, r8_nav / known on all six


def test_d_legacy_heads_without_a_recorded_feed_is_refused(tmp_path):
    np.savez_compressed(tmp_path / "heads_ep000.npz", ws=np.array([7, 12, 17]), v0=np.zeros(3))
    with pytest.raises(SystemExit):
        F.rows_from_heads(tmp_path, _by())


def test_d_empty_or_non_discriminating_set_is_never_pass(tmp_path):
    assert F.check_feed_parity([], SMOKE_DUMP, V9, F.V9_EVAL_MD5)["status"] == "INCONCLUSIVE"
    _write_heads(tmp_path, mutant=False)
    rows = [x for x in F.rows_from_heads(tmp_path, _by()) if x["episode_index"] == 1]       # ep001: fed == clip everywhere
    assert F.check_feed_parity(rows, SMOKE_DUMP, V9, F.V9_EVAL_MD5)["status"] == "INCONCLUSIVE"


# ===================================================================================== (e) the RT layer: registered per-frame table or OFF, never b208c9c1
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _chain(tmp: Path, tree: str, source_sha: str, reg_sha: str, frozen_sha: str | None = None):
    dep = tmp / "pkg/raw/retime_deploy"
    dep.mkdir(parents=True)
    rs = dep / D.PF_RS_TABLE_NAME
    rs.write_bytes((json.dumps({"source_sha256": source_sha}) + "\n").encode())
    np.savez(dep / f"DEPLOY8_POS_{tree}_s0.npz", meta_json=np.array(json.dumps({"tree": tree, "table_sha256": _sha(rs.read_bytes())})))
    (dep / f"FORWARD_DEPLOY8_{tree}.json").write_text(json.dumps({"frozen_gains_sha256": frozen_sha or source_sha}))
    (tmp / "REG.sha256").write_text(reg_sha + "\n")
    return tmp / "REG.sha256", tmp / "pkg"


def test_e_rt_on_only_for_the_registered_chain(tmp_path):
    x = _sha(b"a per-frame frozen table")
    reg, pkg = _chain(tmp_path, "step48852ema_pf", x, x)
    r = D.resolve_deploy_pf("step48852ema_pf", D.AUTO, reg, pkg)
    assert r["on"] is True and r["label"] == f"pf table {x[:8]}", r


def test_e_rt_off_when_unregistered_or_clip_fed(tmp_path):
    assert D.resolve_deploy_pf("step48852ema_pf", D.AUTO, tmp_path / "absent.sha256", tmp_path)["on"] is False
    x = _sha(b"a per-frame frozen table")
    reg, pkg = _chain(tmp_path / "a", "step48852ema_pf", B208, x)                     # stage table built from the CLIP-FED table
    assert D.resolve_deploy_pf("step48852ema_pf", D.AUTO, reg, pkg)["on"] is False
    reg, pkg = _chain(tmp_path / "b", "step48852ema_pf", x, x, frozen_sha=B208)       # forward record says clip-fed
    assert D.resolve_deploy_pf("step48852ema_pf", D.AUTO, reg, pkg)["on"] is False
    reg, pkg = _chain(tmp_path / "c", "step48852ema_pf", B208, B208)                  # the clip-fed table "registered"
    r = D.resolve_deploy_pf("step48852ema_pf", D.AUTO, reg, pkg)
    assert r["on"] is False and "CLIP-FED" in r["note"]
    reg_d, pkg_d = _chain(tmp_path / "d", "zz", x, x)                                   # a valid registration, but the 30k CLIP-FED npz passed explicitly
    r = D.resolve_deploy_pf("step48852ema_pf", str(D.DEPLOY8_30K), reg_d, pkg_d)
    assert r["on"] is False, "the clip-fed 30k npz passed explicitly on a pf tree must not turn the RT layer on"


def test_e_pf_tree_never_defaults_to_the_clip_fed_30k_file(tmp_path):
    B = D.EvalBank(SMOKE_TREE, arch_root=SMOKE_ROOT, load_agents=False, pf_registered=tmp_path / "absent.sha256", pf_pkg=tmp_path)
    assert B.dep is None and B.deploy_label == "RT OFF" and B.rt_note.startswith("RT layer OFF")
    e = B.episode(0)
    assert "ship8" not in e and "prior8" not in e


# ===================================================================================== (f) a tree's name must agree with its recorded feed
def _fake_tree(root: Path, name: str, mode: str | None):
    d = root / name / "dump_s0"
    d.mkdir(parents=True)
    man = {"episodes": [], "grid": {"n_windows": 0}}
    if mode:
        man["nav_feed"] = {"mode": mode, "args_fed": mode == "perframe_full"}
    (d / "manifest.json").write_text(json.dumps(man))
    return root / name


def test_f_tree_name_and_manifest_feed_must_agree(tmp_path):
    assert F.tree_feed(D.ARCH / "step40000ema_unc")["mode"] == "clip"
    with pytest.raises(SystemExit):
        F.tree_feed(_fake_tree(tmp_path, "x_unc", "perframe_full"))
    with pytest.raises(SystemExit):
        F.tree_feed(_fake_tree(tmp_path, "y_pf", None))
    with pytest.raises(SystemExit):
        F.tree_feed(_fake_tree(tmp_path, "z_pf", "perframe_full"))                     # no battery_summary pf block


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
