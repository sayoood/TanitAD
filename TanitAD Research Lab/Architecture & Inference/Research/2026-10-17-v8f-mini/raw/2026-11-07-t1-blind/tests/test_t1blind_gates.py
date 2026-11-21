"""Gates of PREREG_T1_BLIND.md section 3, tested on (a) synthetic artifacts and (b) real 64-window smoke rollouts of the wrapper (CPU, ~45 s each).
Deliberate-regression arms: the token-unchanged mutant must FAIL G-MEAN (both as a mode and as a SOURCE mutation of the wrapper that deletes the replacement line); the ego-perturbing mutant must FAIL G-EGO;
one perturbed bit in a rollout dump must FAIL G-ID. Run: C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_t1blind_gates.py   (CUDA hidden; F: scratch)"""
import os
import sys
import json
import shutil
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np
import pytest

PK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
sys.path.insert(0, PK)
import t1blind_gates as G  # noqa: E402

ART = "E:/Projects/TanitAD-artifacts"
DATA = f"{ART}/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
SCR = "F:/TanitAD-scratch/motion_c/t1_blind/pytest"
os.makedirs(SCR, exist_ok=True)
NWIN = 64


def _receipt(**kw):
    r = {"frames_received": 3072, "frames_equal_mean_bitwise": 3072, "frames_not_equal_mean": 0, "items_built": 384, "items_replaced": 384, "items_other_keys_changed": 0, "items_keyset_changed": 0, "mode": "blind"}
    r.update(kw)
    return r


# ------------------------------------------------------------------------------------------------ synthetic
def test_g_mean_synthetic():
    assert G.gate_mean(_receipt())["PASS"]
    assert not G.gate_mean(_receipt(frames_equal_mean_bitwise=3071, frames_not_equal_mean=1))["PASS"]        # one frame off
    assert not G.gate_mean(_receipt(frames_received=999, frames_equal_mean_bitwise=999))["PASS"]            # below 1,000 frames
    assert not G.gate_mean(_receipt(frames_equal_mean_bitwise=0, frames_not_equal_mean=3072, items_replaced=0))["PASS"]   # the unchanged-token mutant
    assert G.gate_mean(_receipt(frames_received=1000, frames_equal_mean_bitwise=1000))["PASS"]              # exactly 1,000 passes


def _dig(n=8, flip=None, tick1_flip=False):
    tick = np.array([0] * n + [1] * n, np.int16)
    win = np.array(list(range(n)) * 2, np.int32)
    d = np.array([f"{i:064x}" for i in range(2 * n)], "S64")
    if flip is not None:
        d[flip] = b"f" * 64
    if tick1_flip:
        d[n:] = np.array([f"{i + 99:064x}" for i in range(n)], "S64")
    return {"tick": tick, "win": win, "digest": d}


def test_g_ego_synthetic():
    off = _dig()
    assert G.gate_ego(_receipt(), off, _dig())["PASS"]
    assert G.gate_ego(_receipt(), off, _dig(tick1_flip=True))["PASS"]                  # ticks >= 1 are the executed motion: not compared
    assert not G.gate_ego(_receipt(), off, _dig(flip=3))["PASS"]                       # one tick-0 digest differs
    assert not G.gate_ego(_receipt(items_other_keys_changed=1), off, _dig())["PASS"]   # the wrapper touched a non-feats key


def test_g_id_raw_synthetic():
    base = SCR + "/synth_id"
    shutil.rmtree(base, ignore_errors=True)
    for root, tag in ((base + "/run", "ID_s0"), (base + "/bank", "RIC0_s0")):
        os.makedirs(f"{root}/{tag}/ticks")
        os.makedirs(f"{root}/{tag}/dump_{tag}")
        r = np.random.default_rng(0)
        np.savez(f"{root}/{tag}/ticks/ticks_{tag}.npz", plan=r.normal(size=(5, 30, 2)), v0=r.normal(size=5), ckpt=np.asarray("x" + tag), tag=np.asarray(tag), clip_names=np.asarray(["a", "b"]))
        for i in range(3):
            np.savez(f"{root}/{tag}/dump_{tag}/ep{i:04d}.npz", cl=r.normal(size=(4, 30, 2)), g=r.normal(size=(4, 30, 2)))
    assert G.gate_id_raw(base + "/run", "ID_s0", "RIC0_s0", bank_root=base + "/bank")["PASS"]
    z = dict(np.load(f"{base}/run/ID_s0/dump_ID_s0/ep0001.npz"))
    z["cl"] = z["cl"].copy()
    z["cl"].view(np.uint64)[0, 0, 0] ^= 1                                                            # ONE bit of one executed path
    np.savez(f"{base}/run/ID_s0/dump_ID_s0/ep0001.npz", **z)
    r = G.gate_id_raw(base + "/run", "ID_s0", "RIC0_s0", bank_root=base + "/bank")
    assert not r["PASS"] and r["episode_files_unequal"] == ["ep0001.npz"]
    t = dict(np.load(f"{base}/run/ID_s0/ticks/ticks_ID_s0.npz"))
    t["v0"] = t["v0"].copy()
    t["v0"][2] += 1e-12
    np.savez(f"{base}/run/ID_s0/ticks/ticks_ID_s0.npz", **t)
    assert "v0" in G.gate_id_raw(base + "/run", "ID_s0", "RIC0_s0", bank_root=base + "/bank")["ticks_keys_unequal"]


def test_wrapper_refuses_a_wrong_mean():
    import torch
    bad = SCR + "/zeros_mean.pt"
    torch.save(torch.zeros(40, 1024, dtype=torch.float16), bad)
    r = subprocess.run([sys.executable, "-I", f"{PK}/t1blind_run.py", "--blind-mode", "blind", "--receipt-dir", SCR + "/rc_bad", "--mean", bad], capture_output=True, text=True)
    assert r.returncode != 0 and "REFUSED" in (r.stdout + r.stderr), r.stderr[-500:]


# ------------------------------------------------------------------------------------------------ real smoke rollouts
def _smoke(mode, tag, script=None, extra=()):
    out = f"{SCR}/out/{tag}"
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="12", MKL_NUM_THREADS="12", OPENBLAS_NUM_THREADS="12")
    cmd = [sys.executable, "-I", script or f"{PK}/t1blind_run.py", "--blind-mode", mode, "--receipt-dir", f"{SCR}/receipts", *extra,
           "--ckpt", f"{ART}/v8f-recover/M2b_s0_recft_C0/ckpt_last.pt", "--tag", tag, "--tracks", f"{ART}/sfresh_inputs/inputs/native_tracks.pt", "--eval-cache", f"{ART}/sfresh_inputs/cache",
           "--eval-grid", f"{DATA}/eval_grid_windows.json", "--eval-join", f"{DATA}/join_eval141.npz", "--out-dir", out, "--replan-steps", "10", "--nav", "deadreckon", "--schedule", "periodic",
           "--limit", "causal", "--dump-ticks", f"{out}/ticks", "--dump-heads", "--device", "cpu", "--smooth", "sg72", "--scene", "reindexed", "--max-windows", str(NWIN), "--no-analyze"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, (r.stdout[-800:], r.stderr[-800:])
    return json.load(open(f"{SCR}/receipts/receipt_{tag}.json")), np.load(f"{SCR}/receipts/ego_digests_{tag}.npz")


@pytest.fixture(scope="module")
def smoke():
    shutil.rmtree(SCR + "/receipts", ignore_errors=True)
    return {m: _smoke(m, f"T_{m}") for m in ("off", "blind", "mutant", "mutant-ego")}


def test_real_blind_passes_g_mean_and_g_ego(smoke):
    rb, db = smoke["blind"]
    gm = G.gate_mean(rb)
    assert gm["PASS"] and gm["frames_received"] >= 1000 and gm["frames_equal_mean_bitwise"] == gm["frames_received"], gm
    ge = G.gate_ego(rb, smoke["off"][1], db)
    assert ge["PASS"] and ge["tick0_windows_compared"] == NWIN and ge["tick0_windows_equal"] == NWIN, ge
    assert rb["tool_sha256"] == "0417c838355a51f6f7084c17b5acada96c5b6cbd9eef54cc2602d377bf3c61e7" and rb["mean_tensor_sha256"].startswith("cd6a7ab5")


def test_real_off_and_token_mutant_fail_g_mean(smoke):
    for m in ("off", "mutant"):
        gm = G.gate_mean(smoke[m][0])
        assert not gm["PASS"] and gm["frames_equal_mean_bitwise"] == 0 and gm["frames_received"] >= 1000, (m, gm)


def test_real_ego_mutant_fails_g_ego(smoke):
    rm, dm = smoke["mutant-ego"]
    ge = G.gate_ego(rm, smoke["off"][1], dm)
    assert not ge["PASS"] and ge["tick0_windows_equal"] == 0 and ge["items_other_keys_changed"] > 0, ge
    ge2 = G.gate_ego(dict(rm, items_other_keys_changed=0), smoke["off"][1], dm)              # the digest layer alone also catches it
    assert not ge2["PASS"] and ge2["tick0_windows_equal"] == 0, ge2


def test_source_mutant_deleting_the_replacement_line_fails_g_mean():
    """MUTATE, do not inspect: delete the single line that writes the mean into the item and run the blind mode; G-MEAN must go RED."""
    src = open(f"{PK}/t1blind_run.py", encoding="utf-8").read().split("\n")
    idx = [i for i, ln in enumerate(src) if "<<REPLACE>>" in ln]
    assert len(idx) == 1, idx
    ind = src[idx[0]][:len(src[idx[0]]) - len(src[idx[0]].lstrip())]
    src[idx[0]] = ind + "pass"
    mdir = SCR + "/mut"
    os.makedirs(mdir, exist_ok=True)
    mpath = mdir + "/t1blind_run_mut.py"
    open(mpath, "w", encoding="utf-8", newline="\n").write("\n".join(src))
    rc, _ = _smoke("blind", "T_srcmut", script=mpath, extra=("--mean", f"{PK}/blind_mean_pinned.pt"))
    gm = G.gate_mean(rc)
    assert not gm["PASS"] and gm["frames_equal_mean_bitwise"] == 0, gm
