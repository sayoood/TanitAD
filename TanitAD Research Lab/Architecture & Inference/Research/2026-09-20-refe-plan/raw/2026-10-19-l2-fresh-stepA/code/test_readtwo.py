"""Tests of the READ 2 preparation tooling.  CPU only (CUDA_VISIBLE_DEVICES=-1).  Run:  python -m pytest test_readtwo.py -q
The G2b bridge needs the GPU for the real run; here its wrapper logic (identity + RED arm + the reachability of the perturbation) runs end to end through the real CLI against a FAKE planner and a FAKE bundle
script, with RED arms of its own: a gate that cannot see the perturbation, a perturbation that never reaches the forward, and an identity mismatch must each FAIL the bridge."""
import json
import os
import subprocess
import sys
import textwrap

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import bridge_g2b as BG  # noqa: E402
import readtwo_geometry as RG  # noqa: E402

PLANNER_PY = '''
import os, torch
class REFePlanner:
    def __init__(self, *a, **k): self._scenario = None
    def _image_for(self, ego): return torch.zeros(1, 4, 3, 8, 8)
    def infer(self, ego, img):
        s = float(img.sum()) if os.environ.get("FAKE_PLANNER_MODE", "sensitive") == "sensitive" else 0.0
        return torch.full((64, 20, 3), 1.0 + s), torch.full((64, 6), 2.0 + s), 0
'''

BUNDLE_PY = '''
import argparse, json, os, sys, types
import numpy as np
sys.path.insert(0, os.path.join(os.environ["L1_TREE"], "refe"))
import planner as PLM
def main():
    ap = argparse.ArgumentParser()
    for f in ("--ckpt", "--plan", "--sample", "--table", "--proposals20", "--out-dir", "--device"):
        ap.add_argument(f)
    a = ap.parse_args()
    toks = sorted(json.load(open(a.sample))["tokens"])
    t = np.load(a.table, allow_pickle=True); ix = {str(k): i for i, k in enumerate(t["token"])}
    p = np.load(a.proposals20, allow_pickle=True); pix = {str(k): i for i, k in enumerate(p["token"])}
    pl, bad = PLM.REFePlanner(), []
    for tok in toks:
        pl._scenario = types.SimpleNamespace(_initial_lidar_token=tok)
        img = PLM.torch.zeros(1, 4, 3, 8, 8) if os.environ.get("FAKE_SKIP_IMAGE_FOR") == "1" else pl._image_for(None)
        traj, score, k = pl.infer(None, img)
        r = traj.float().numpy().astype(np.float32); rep = r.copy(); rep[:, -1, 2] = rep[:, -2, 2]
        lg = score.float().numpy().astype(np.float32)
        if not (np.array_equal(rep, p["proposals20"][pix[tok]]) and np.array_equal(lg, t["logits"][ix[tok]]) and int(k) == int(t["pick"][ix[tok]])):
            bad.append((tok, "differs"))
    if bad:
        print(f"ZZBUNDLE_FAIL missing 0 bad {len(bad)} {bad[:3]}", flush=True); return 1
    print(f"ZZBUNDLE_OK {len(toks)} 0 s", flush=True); return 0
if __name__ == "__main__":
    sys.exit(main())
'''


def make_world(tmp_path, table_logit=2.0):
    tree = tmp_path / "tree"
    (tree / "refe").mkdir(parents=True)
    (tree / "eval").mkdir()
    (tree / "refe" / "planner.py").write_text(PLANNER_PY)
    (tmp_path / "fake_bundle.py").write_text(BUNDLE_PY)
    toks = ["aa01", "bb02", "cc03"]
    np.savez(tmp_path / "table.npz", token=np.array(toks), logits=np.full((3, 64, 6), table_logit, np.float32), pick=np.zeros(3, np.int64))
    np.savez(tmp_path / "p20.npz", token=np.array(toks), proposals20=np.ones((3, 64, 20, 3), np.float32))
    (tmp_path / "sample.json").write_text(json.dumps({"tokens": toks}))
    (tmp_path / "plan.json").write_text("{}")
    return tree


def run_bridge(tmp_path, tree, **env):
    e = dict(os.environ, L1_TREE=str(tree), CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", **env)
    cmd = [sys.executable, "-I", os.path.join(HERE, "bridge_g2b.py"), "--ckpt", "x", "--plan", str(tmp_path / "plan.json"), "--sample", str(tmp_path / "sample.json"), "--table", str(tmp_path / "table.npz"),
           "--proposals20", str(tmp_path / "p20.npz"), "--out-dir", str(tmp_path / "out"), "--bundle-script", str(tmp_path / "fake_bundle.py"), "--device", "cpu"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=300)
    return p.returncode, p.stdout + p.stderr, (json.load(open(tmp_path / "out" / "bridge_g2b.json")) if (tmp_path / "out" / "bridge_g2b.json").exists() else None)


def test_bridge_PASSES_when_the_forward_is_bit_identical_and_the_gate_sees_a_one_pixel_change(tmp_path):
    tree = make_world(tmp_path)
    rc, out, res = run_bridge(tmp_path, tree)
    assert rc == 0 and "ZZBRIDGE_PASS" in out and "ZZBRIDGE_IDENTITY_OK 3" in out and "ZZBRIDGE_RED_OK 3" in out, out[-600:]
    assert res["pass"] and res["red"]["perturbation_applied"] == 3 and res["red"]["all_tokens_differ"] and res["identity"]["all_tokens_bit_equal"]


def test_bridge_RED_a_gate_blind_to_the_perturbation_FAILS(tmp_path):
    """a planner whose output does not depend on the image: the RED arm reads 'equal' -> the bridge must FAIL (the identity would certify nothing)"""
    tree = make_world(tmp_path)
    rc, out, res = run_bridge(tmp_path, tree, FAKE_PLANNER_MODE="blind")
    assert rc == 1 and "ZZBRIDGE_FAIL" in out
    assert not res["red"]["pass"] and any("cannot see" in w for w in res["red"]["why_not_pass"]) and any("did not fire" in w for w in res["red"]["why_not_pass"])


def test_bridge_RED_a_perturbation_that_never_reaches_the_forward_FAILS(tmp_path):
    tree = make_world(tmp_path)
    rc, out, res = run_bridge(tmp_path, tree, FAKE_SKIP_IMAGE_FOR="1")
    assert rc == 1 and res["red"]["perturbation_applied"] == 0
    assert any("never reached the forward" in w for w in res["red"]["why_not_pass"])


def test_bridge_RED_an_identity_mismatch_FAILS(tmp_path):
    """the banked rows differ from what the forward returns: the IDENTITY arm fails (and the RED arm cannot rescue it)"""
    tree = make_world(tmp_path, table_logit=2.5)
    rc, out, res = run_bridge(tmp_path, tree)
    assert rc == 1 and not res["identity"]["pass"] and res["identity"]["rc"] == 1 and not res["pass"]


def test_judge_is_pure_and_requires_every_token():
    bank = {t: (np.full((64, 6), 2.0, np.float32), 0, np.ones((64, 20, 3), np.float32)) for t in ("a", "b", "c")}
    good = lambda v: {t: (np.full((64, 20, 3), v, np.float32), np.full((64, 6), 2.0 + (v - 1.0), np.float32), 0) for t in ("a", "b", "c")}
    ident = {"rc": 0, "stdout": "ZZBUNDLE_OK 3", "fired": 0, "calls": good(1.0)}
    red = {"rc": 1, "stdout": "ZZBUNDLE_FAIL missing 0 bad 3", "fired": 3, "calls": good(1.001)}
    assert BG.judge(ident, red, ["a", "b", "c"], bank)["pass"]
    red_one_equal = dict(red, calls={**good(1.001), "c": good(1.0)["c"]})                  # one perturbed forward still equals the bank: the bridge cannot pass
    assert not BG.judge(ident, red_one_equal, ["a", "b", "c"], bank)["pass"]
    ident_missing = dict(ident, calls={k: v for k, v in good(1.0).items() if k != "c"})    # a token with no recorded forward is not an identity
    assert not BG.judge(ident_missing, red, ["a", "b", "c"], bank)["pass"]


def test_sample_of_three_is_in_B_and_in_three_logs():
    s = json.load(open(f"{PKG}/raw/bridge3_sample.json"))
    b = json.load(open(f"{PKG}/raw/fresh_sample_token_disjoint.json"))
    assert len(s["tokens"]) == 3 and len(set(s["logs"])) == 3 and set(s["tokens"]) <= set(b["tokens"])


def test_certified_script_pin_matches_disk():
    assert BG.sha_lf(BG.L1_SCRIPT) == BG.L1_SCRIPT_SHA256_LF


# ------------------------------------------------------------------------------------------------------------ the D carve and its exposure (MEASURED artifacts)
def test_D_carve_is_the_103_minus_the_4_and_log_disjoint_from_StepA():
    r = json.load(open(f"{PKG}/raw/fresh_sample_D_scene_unseen.json"))
    probe = json.load(open(f"{PKG}/raw/logless251_probe.json"))
    assert len(probe["found"]) == 103 and r["status"] == "SHORT" and r["n"] == 99 and len(r["excluded_in_the_24_heldout_logs"]) == 4
    assert set(r["tokens"]) == set(probe["found"]) - set(r["excluded_in_the_24_heldout_logs"])
    assert r["stepA_overlap"] == {"tokens": 0, "logs": 0} and r["sampler_identity_at_n_equals_universe"] is True and r["seed"] not in (20260927, 20261019)
    import hashlib
    assert hashlib.sha256("\n".join(r["tokens"]).encode()).hexdigest() == r["tokens_sha256"]
    plan = json.load(open("C:/Users/Admin/p2heldout/plan/heldout_tokens.json"))
    assert not (set(r["tokens"]) & set(plan["tokens"]))                                       # none of D's tokens is a held-out plan token


def test_D_exposure_sources_are_named_by_file_and_line_and_the_lines_say_what_they_claim():
    e = json.load(open(f"{PKG}/raw/readtwo_exposure.json"))
    src = open(f"{HERE}/fresh_carve.py", encoding="utf-8").read().splitlines()
    for k, v in e["sources"].items():
        if v.startswith("code/fresh_carve.py:"):
            ln = int(v.split(":")[1])
            assert any(w in src[ln - 1] for w in ("SERVED", "TOKTAB", "SCAN", "train_tok_A", "train_log_A", "scan")), (k, v, src[ln - 1])
    assert e["logs_with_any_mpdm_token"] == sum(1 for x in e["per_log"].values() if x["mpdm_tokens"] > 0)


def test_fetch_list_is_the_carve_times_four_cameras_and_unique():
    f = json.load(open(f"{PKG}/raw/readtwo_fetch_list.json"))
    r = json.load(open(f"{PKG}/raw/fresh_sample_D_scene_unseen.json"))
    assert f["n_tokens"] == 99 and f["n_jpgs"] == 396 == f["n_unique_paths"] and f["max_dt_ms"] <= 50.0
    assert {i["token"] for i in f["items"]} == set(r["tokens"])
    lines = open(f"{PKG}/raw/readtwo_fetch_list.txt").read().split()
    assert len(lines) == 396 and all(p.count("/") == 2 and p.endswith(".jpg") for p in lines)


def test_geometry_model_reproduces_StepA_interval_by_construction_and_shrinks_with_more_tokens():
    num = np.array([5.0, 6.0, 4.0, 7.0] * 100)
    den = np.array([8.0, 8.0, 7.0, 9.0] * 100)
    g = RG.geometry(num, den, hw_logcluster=0.05, n_tokens_A=400, n_logs_A=24, m_D=99, logs_D=99)
    assert g["HW_D_estimated"] > RG.geometry(num, den, 0.05, 400, 24, m_D=400, logs_D=400)["HW_D_estimated"]          # same cluster size (1), more tokens -> narrower
    one = RG.geometry(num, den, hw_logcluster=0.05, n_tokens_A=400, n_logs_A=24, m_D=400, logs_D=24)           # D = Step A's own shape -> the measured half-width back
    assert one["HW_D_estimated"] == pytest.approx(0.05, rel=1e-9)


def test_certified_script_pin_is_enforced_for_the_L1_path_in_either_slash_style(tmp_path):
    """a differing L1 script (simulated by a different expected sha) is REFUSED even when the path is spelled with forward slashes (as bridge_g2b.cmd passes it)"""
    import importlib
    old = BG.L1_SCRIPT_SHA256_LF
    try:
        BG.L1_SCRIPT_SHA256_LF = "0" * 64
        with pytest.raises(SystemExit) as e:
            BG.main(["--ckpt", "x", "--plan", "x", "--sample", f"{PKG}/raw/bridge3_sample.json", "--table", "x", "--out-dir", str(tmp_path), "--bundle-script", BG.L1_SCRIPT.replace("\\", "/")])
        assert "REFUSED" in str(e.value)
    finally:
        BG.L1_SCRIPT_SHA256_LF = old
