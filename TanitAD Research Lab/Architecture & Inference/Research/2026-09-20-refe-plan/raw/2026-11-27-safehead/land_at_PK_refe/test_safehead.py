#!/usr/bin/env python3
"""L2 SAFE-HEAD: tests of the cache contract, the head-only trainer and its launch-gate profiles (PREREG_SAFEHEAD.md 6-7).  SYNTHETIC data only, CPU only.

  python test_safehead.py           everything: + every deliberate-regression arm of the gate, + mint / verify / the trainer's own token refusal end to end
  python test_safehead.py --fast    the contract, the trainer, CTRL-REACH (+ its mutation), the held-out rule, CTRL-RED, refusals, a clean gate pass (G-SUITE's form)

Prints ZZSAFEHEAD_PASS <n> (or ZZSAFEHEAD_FAIL).  Temp dirs are tracked and removed at exit; a leftover is named on stderr as ZZTMP_LEFT.
"""
from __future__ import annotations

import atexit
import copy
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gate_checks as GC  # noqa: E402
import gate_token as GT  # noqa: E402
import launch_gate as L  # noqa: E402
import safehead_data as D  # noqa: E402
import safehead_gate as SHG  # noqa: E402
import safehead_train as T  # noqa: E402

FAST = "--fast" in sys.argv
N = 0
_TMP: list = []
# whole-profile fingerprints sha256(repr(sorted(items)))[:16] of EVERY profile that existed before SAFE-HEAD (tip 1df04b58): the SAFE-HEAD patch must move none
PRE_FP = {'refe_aux_s1': 'f95f48727b62ccc6', 'refe_aux_s1c': 'dcdf2dbd2a674f1c', 'refe_div_p6': 'e51f9374a8f29816', 'refe_dose_s1z_p6': '9085f9e5cf59c016',
          'refe_field': '5ddc49053a99fa78', 'refe_field_ctrl': '0b6281b1bfa996ed', 'refe_field_p7': '6950f70944a0fbda', 'refe_field_p7_ctrl': '3dc54a1a4de1a7d7',
          'refe_field_p7_ctrl_s1': '84134e3ebb2d3f93', 'refe_field_p7_s1': 'b502f0275a4862d8', 'refe_hnscreen': 'd30126bbb5a32957', 'refe_p2': '627341286aea3506',
          'refe_p2rep': '44ca107f53aceee8', 'refe_p5c': 'ed995961b67113bd', 'refe_p5c_c0': '65b0a2e8fc15b36e', 'refe_p5c_c0r': 'e89c54c4fa046afd'}
PRE_FP_FILE = HERE / "test_safehead_prefp.json"          # the full 40-profile pin (written once from the unpatched tip; see main)


def mkdtemp() -> Path:
    d = Path(tempfile.mkdtemp(prefix="shtest_"))
    _TMP.append(d)
    return d


@atexit.register
def _cleanup():
    for d in _TMP:
        for _ in range(3):
            shutil.rmtree(d, ignore_errors=True)
            if not d.exists():
                break
            time.sleep(0.5)
        if d.exists():
            print(f"ZZTMP_LEFT {d}", file=sys.stderr)


def ok(name):
    global N
    N += 1
    print(f"ok {N:02d} {name}", flush=True)


def expect_raises(fn, exc, needle=""):
    try:
        fn()
    except exc as e:
        assert needle in str(e), f"raised {type(e).__name__} without {needle!r}: {e}"
        return
    raise AssertionError(f"{fn} did not raise {exc.__name__}")


# ============================================================================================ the synthetic world
def make_world(root: Path, *, n_tr_logs=4, n_ho_logs=2, per_log=24, seed=0, s_dtype=np.float32, layout="per_log", navtest_overlap=False) -> dict:
    """A synthetic cache in the contract's form.  Safety labels depend on a hidden direction of s that C1's head only partly reads (so head training can help),
    PDMS is the NAVSIM v1.1 formula of the labels, and the cached logits / pick are C1's own head + planner rule."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "cache"
    cache.mkdir()
    U = rng.normal(0, 1, (6, D.D_S)).astype(np.float32) / np.sqrt(D.D_S)
    W0 = (0.25 * U * 3.0 + rng.normal(0, 0.05, (6, D.D_S))).astype(np.float32)
    b0 = np.array([2.0, 2.0, 0.0, 1.5, 1.0, 2.0], np.float32)
    logs_tr = [f"2021.01.01.00.00.{i:02d}_veh-1_tr{i}" for i in range(n_tr_logs)]
    logs_ho = [f"2021.02.02.00.00.{i:02d}_veh-2_ho{i}" for i in range(n_ho_logs)]
    rows, recs = [], []
    k = 0
    for split, logs in (("train", logs_tr), ("heldout", logs_ho)):
        for lg in logs:
            for _ in range(per_log):
                s32 = rng.normal(0, 1, (D.N_PROP, D.D_S)).astype(np.float32)
                s = s32.astype(s_dtype)
                sf = s.astype(np.float32)
                z = sf @ U.T * 3.0
                lab = {}
                for r, nm in ((0, "tr_nc"), (1, "tr_dac"), (3, "tr_ttc"), (5, "tr_ddc")):
                    pr = 1 / (1 + np.exp(-(z[:, r] + 1.2)))
                    v = (rng.random(D.N_PROP) < pr).astype(np.float32)
                    if nm in ("tr_nc", "tr_ddc"):
                        v[(v == 0) & (rng.random(D.N_PROP) < 0.2)] = 0.5
                    lab[nm] = v
                lab["tr_ep"] = np.clip(1 / (1 + np.exp(-z[:, 2])) + rng.normal(0, 0.1, D.N_PROP), 0, 1).astype(np.float32)
                lab["tr_c"] = (rng.random(D.N_PROP) < 0.9).astype(np.float32)
                lab["tr_pdms"] = (lab["tr_nc"] * lab["tr_dac"] * (5 * lab["tr_ep"] + 5 * lab["tr_ttc"] + 2 * lab["tr_c"]) / 12.0).astype(np.float32)
                lg_ = (s32 @ W0.T + b0).astype(np.float32)
                pick = int(D.select(lg_))
                tok = f"{k:016x}"
                rec = {"s": s, "logits": lg_, "pick": np.int64(pick), "props": rng.normal(0, 1, (D.N_PROP, 20, 3)).astype(np.float32), **lab}
                recs.append(rec)
                rows.append((tok, lg, split))
                recs[-1]["_log"], recs[-1]["_split"], recs[-1]["_tok"] = lg, split, tok
                if layout == "per_token":
                    np.savez(cache / f"{tok}.npz", **{k_: v_ for k_, v_ in rec.items() if not k_.startswith("_")})
                k += 1
    with open(root / "cache" / "index.csv", "w", encoding="utf-8", newline="") as fh:
        if layout == "per_token":
            fh.write("token,log,split,file\n")
            for tok, lg, sp in rows:
                fh.write(f"{tok},{lg},{sp},{tok}.npz\n")
        elif layout == "per_log":                           # DATA_CONTRACT.md v1: one npz per LOG, a `token` array, 0-d log / split, sha256 of the file in the index
            fh.write("token,log,split,file,sha256\n")
            for lg in dict.fromkeys(r["_log"] for r in recs):
                R_ = [r for r in recs if r["_log"] == lg]
                f_ = cache / f"{lg}.npz"
                np.savez(f_, token=np.array([r["_tok"] for r in R_]), log=np.array(lg), split=np.array(R_[0]["_split"]),
                         **{key: np.stack([r[key] for r in R_]) for key in R_[0] if not key.startswith("_")})
                for r in R_:
                    fh.write(f"{r['_tok']},{lg},{r['_split']},{lg}.npz,{D.sha256_file(f_)}\n")
        else:
            np.savez(cache / "stacked.npz", **{key: np.stack([r[key] for r in recs]) for key in recs[0] if not key.startswith("_")})
            fh.write("token,log,split,file,row\n")
            for i, (tok, lg, sp) in enumerate(rows):
                fh.write(f"{tok},{lg},{sp},stacked.npz,{i}\n")
    ckpt = root / "c1.pt"
    torch.save({"model": {"score_head.weight": torch.from_numpy(W0), "score_head.bias": torch.from_numpy(b0), "traj_head.weight": torch.randn(3, 3)},
                "meta": {"synthetic": True}}, ckpt)
    (root / "heldout_logs.txt").write_text("\n".join(logs_ho + ["2021.03.03.00.00.00_veh-3_unused"]) + "\n", encoding="utf-8")
    nt = [f"2021.09.09.00.00.{i:02d}_veh-9_nt{i}" for i in range(3)] + ([logs_tr[0]] if navtest_overlap else [])
    (root / "navtest_logs.txt").write_text("\n".join(nt) + "\n", encoding="utf-8")
    sealed = root / "run" / "index_sealed.csv"
    D.seal_index(root / "cache" / "index.csv", sealed)
    return {"root": root, "index": root / "cache" / "index.csv", "sealed": sealed, "ckpt": ckpt, "W0": W0, "b0": b0,
            "heldout_logs": root / "heldout_logs.txt", "navtest_logs": root / "navtest_logs.txt"}


def argv_for(w: dict, out: Path, *, seed=0, perm=None, steps=None, lr=None, extra=()) -> list:
    a = ["--cache-index", str(w["sealed"]), "--init-from", str(w["ckpt"]), "--heldout-logs", str(w["heldout_logs"]), "--navtest-logs", str(w["navtest_logs"]),
         "--out", str(out), "--seed", str(seed), "--steps", str(steps or SHG.REGISTERED["--steps"]), "--lr", str(lr or SHG.REGISTERED["--lr"]),
         "--batch", SHG.REGISTERED["--batch"], "--gamma", SHG.REGISTERED["--gamma"], "--rows", SHG.REGISTERED["--rows"], "--save-steps", SHG.REGISTERED["--save-steps"]]
    if perm is not None:
        a += ["--permute-labels-seed", str(perm)]
    return a + list(extra)


def trainer(argv, env_extra=None) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1")
    env.pop(GC.ARM_ENV, None)
    env.pop(T.TOKEN_ENV, None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, str(HERE / "safehead_train.py"), *argv], cwd=str(HERE), env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def test_profile(w: dict, arm="s0") -> dict:
    """The REAL profile with ONLY the two pins a synthetic world cannot meet swapped: C1's sha256 -> the synthetic checkpoint's, the suite -> one trivial entry."""
    p = copy.deepcopy(L.PROFILES[SHG.PROFILE_NAME[arm]])
    p["init_from_sha256"] = D.sha256_file(w["ckpt"])
    p["suite"] = [{"raw": [sys.executable, "-c", "print('ZZTEST_PASS')"], "token": "ZZTEST_PASS"}]
    return p


def gate(prof, argv, only=None, arm=None) -> dict:
    out = mkdtemp()
    st = L.run_checks(prof, argv, HERE.parent, out, only=set(only) if only else None, arm=arm, commit="ab" * 20)
    return st


# ============================================================================================ tests
def t_mapping():
    src = (HERE / "train.py").read_text(encoding="utf-8")
    assert 'COMPONENTS = ("collision", "drivable_area", "progress", "ttc", "comfort", "ddc")' in src, "train.py ScorerBank.COMPONENTS moved"
    pl = (HERE / "planner.py").read_text(encoding="utf-8")
    assert "0 collision (NC) · 1 drivable area (DAC) · 2 progress (EP)" in pl and "3 time to collision (TTC) · 4 comfort (C) · 5 driving direction (DDC)" in pl
    assert "return p[..., 0] * p[..., 1] * (p[..., 2] * w1[0] + p[..., 3] * w1[1] + p[..., 4] * w1[2]) / w1.sum()" in pl, "planner navsim_v1 aggregate moved"
    assert [D.ROW_NAMES[r] for r in D.SAFETY_ROWS] == ["nc", "dac", "ttc", "ddc"] and [D.ROW_NAMES[r] for r in D.FROZEN_ROWS] == ["ep", "c"]
    assert SHG.C1_SHA256 == L.SCORERFT_BASE_SHA256, "the C1 pin differs from the SCORER-FT base pin"
    for nm in SHG.PROFILE_NAME.values():
        assert L.PROFILES[nm]["init_from_sha256"] == SHG.C1_SHA256
    ok("row mapping NC,DAC,EP,TTC,C,DDC from train.py:269 + planner.py:547-549; aggregate text pinned; C1 sha == SCORERFT_BASE_SHA256")


def t_agg32():
    rng = np.random.default_rng(1)
    Lg = rng.normal(0, 2, (500, 64, 6)).astype(np.float32)
    p = 1 / (1 + np.exp(-Lg.astype(np.float64)))
    ref = p[..., 0] * p[..., 1] * (5 * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / 12
    assert np.abs(D.agg32(Lg) - ref).max() < 1e-6
    L2 = Lg.copy()
    L2[..., 5] = rng.normal(0, 5, L2[..., 5].shape)
    assert np.array_equal(D.agg32(Lg), D.agg32(L2)), "DDC moved the aggregate"
    assert D.agg32(Lg).dtype == np.float32
    ok("agg32 = navsim_v1 (float64 reference within 1e-6), float32, DDC does not enter")


def t_delta():
    rng = np.random.default_rng(2)
    s = rng.normal(0, 1, (7, 64, 256)).astype(np.float32)
    W0, b0 = rng.normal(0, .05, (6, 256)).astype(np.float32), rng.normal(0, 1, 6).astype(np.float32)
    base = D.direct_logits(s, W0, b0)
    assert np.array_equal(D.delta_logits(base, s, W0, b0, W0, b0), base), "CTRL-ID: the untrained head is not bit-exact"
    W, b = W0.copy(), b0.copy()
    W[[0, 1, 3, 5]] += rng.normal(0, .01, (4, 256)).astype(np.float32)
    b[[0, 1, 3, 5]] += 0.3
    Ld = D.delta_logits(base, s, W, b, W0, b0)
    assert np.array_equal(Ld[..., [2, 4]], base[..., [2, 4]])
    assert np.abs(Ld - D.direct_logits(s, W, b)).max() < 1e-4
    ok("delta form: untrained head bit-exact, EP/C untouched, = direct form within 1e-4")


def t_contract(tmp: Path):
    for layout, dt in (("per_log", np.float32), ("per_token", np.float16), ("stacked", np.float32)):
        w = make_world(tmp / f"c_{layout}", layout=layout, s_dtype=dt, per_log=6)
        c = D.load_cache(w["sealed"], splits=D.SPLITS_NAVTRAIN)
        assert c["s"].shape == (36, 64, 256) and c["s"].dtype == np.float32 and set(c["split"]) == {"train", "heldout"}
        assert c["diag"]["pdms_identity_max_abs"] < 1e-6
    expect_raises(lambda: D.load_cache(w["index"], splits=D.SPLITS_NAVTRAIN), D.ContractError, "NOT SEALED")
    expect_raises(lambda: D.seal_index(w["index"], w["index"].parent / "sub" / "x.csv"), D.ContractError, "inside the cache directory")
    w = make_world(tmp / "c_bad", per_log=4, layout="per_token")
    f0 = sorted((w["root"] / "cache").glob("*.npz"))[0]
    z = dict(np.load(f0))
    for mut, needle in ((lambda r: r.update(tr_dac=np.full(64, 0.5, np.float32)), "outside"), (lambda r: r.update(tr_c=np.full(64, 0.5, np.float32)), "outside"),
                        (lambda r: r.update(pick=np.int64((int(r["pick"]) + 1) % 64)), "CTRL-S"), (lambda r: r.update(s=np.full((64, 256), np.nan, np.float32)), "non-finite"),
                        (lambda r: r.update(logits=r["logits"][:, :5]), "contract")):
        r = dict(z)
        mut(r)
        np.savez(f0, **r)
        expect_raises(lambda: D.load_cache(w["sealed"], splits=D.SPLITS_NAVTRAIN, verify_sha=False), D.ContractError, needle)
    np.savez(f0, **z)
    D.load_cache(w["sealed"], splits=D.SPLITS_NAVTRAIN)                      # restored bytes: the seal holds again
    np.savez(f0, **{**z, "props": z["props"] + 1.0})                         # a valid file whose CONTENT moved after sealing
    expect_raises(lambda: D.load_cache(w["sealed"], splits=D.SPLITS_NAVTRAIN), D.ContractError, "changed after sealing")
    ok("cache contract: the per-log (data agent's), per-token and stacked layouts; unsealed refused, seal outside the cache, 5 breaches refused, sha change after sealing refused")


def t_leak(tmp: Path):
    w = make_world(tmp / "leak", per_log=4, navtest_overlap=True)
    c = D.load_cache(w["sealed"], splits=D.SPLITS_NAVTRAIN)
    pr = D.leak_problems(c, D.read_log_list(w["heldout_logs"]), D.read_log_list(w["navtest_logs"]))
    assert any("NAVTEST" in p for p in pr), pr
    assert any("EMPTY" in p for p in D.leak_problems(c, D.read_log_list(w["heldout_logs"]), []))
    r = trainer(argv_for(w, tmp / "leak_out", extra=["--smoke-steps", "2"]))
    assert r.returncode != 0 and "CTRL-LEAK" in (r.stdout + r.stderr), r.stdout[-300:] + r.stderr[-300:]
    ok("CTRL-LEAK: a navtest log in the training cache is refused by leak_problems and by the trainer; an empty navtest list is refused")


def t_feat(tmp: Path, w: dict):
    bad = w["root"] / "c1_other.pt"
    torch.save({"model": {"score_head.weight": torch.from_numpy(w["W0"] * 1.5), "score_head.bias": torch.from_numpy(w["b0"])}}, bad)
    a = argv_for(w, tmp / "feat_out", extra=["--smoke-steps", "2"])
    a[a.index("--init-from") + 1] = str(bad)
    r = trainer(a)
    assert r.returncode != 0 and "CTRL-FEAT" in (r.stdout + r.stderr), r.stderr[-300:]
    w16 = make_world(tmp / "feat16", per_log=4, s_dtype=np.float16)
    r = trainer(argv_for(w16, tmp / "feat16_out", extra=["--smoke-steps", "2"]))
    assert r.returncode != 0 and "CTRL-FEAT" in (r.stdout + r.stderr), r.stderr[-300:]
    ok("CTRL-FEAT (max, p99 <= 1e-4): a checkpoint whose head is not the cache's head, and an s stored at float16, are refused")


def t_token_refusal(tmp: Path, w: dict):
    r = trainer(argv_for(w, tmp / "notok"))
    assert r.returncode != 0 and "no launch-gate token" in (r.stdout + r.stderr), r.stderr[-300:]
    r = trainer(argv_for(w, tmp / "notok2"), {GC.ARM_ENV: "sh_dvb_lr_drift"})
    assert r.returncode != 0 and "regression arm is active" in (r.stdout + r.stderr)
    ok("the trainer refuses a real run without a launch-gate token, and under a regression arm")


def t_reach(tmp: Path, w: dict):
    out = tmp / "reach"
    r = trainer(_swap(argv_for(w, out, extra=["--smoke-steps", "5"]), "--save-steps", "0,5"))
    assert r.returncode == 0, r.stderr[-500:]
    M = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines()]
    assert len(M) == 5 and all(m["grad_abs"][f"row{g}"] > 0 for m in M for g in (0, 1, 3, 5))
    assert all(v == 0.0 for m in M for v in m["frozen_grad_abs"].values()) and set(M[0]["frozen_grad_abs"]) == {"row2", "row4"}
    h0, h5 = D.load_head(out / "head_step00000.npz"), D.load_head(out / "head_step00005.npz")
    assert np.array_equal(h0["W"], w["W0"]) and np.array_equal(h0["b"], w["b0"])
    assert np.array_equal(h5["W"][[2, 4]], w["W0"][[2, 4]]) and np.array_equal(h5["b"][[2, 4]], w["b0"][[2, 4]])
    assert all(np.abs(h5["W"][g] - w["W0"][g]).sum() > 0 for g in (0, 1, 3, 5))
    prof = test_profile(w)
    st = gate(prof, argv_for(w, tmp / "reach_launch"), only=["G-LIVE"], arm="sh_reach_row_dropped")
    assert st["G-LIVE"] == "FAIL", st
    st = gate(prof, argv_for(w, tmp / "reach_launch2"), only=["G-LIVE"], arm="sh_reach_ep_in_loss")
    assert st["G-LIVE"] == "FAIL", st
    ok("CTRL-REACH: non-zero grad on rows 0,1,3,5; EXACTLY 0.0 on EP/C; EP/C bit-identical; removing a row from the optimiser and EP in the loss turn G-LIVE RED")


def _swap(argv, flag, val):
    a = list(argv)
    a[a.index(flag) + 1] = val
    return a


def t_rule_and_red(tmp: Path):
    w = make_world(tmp / "rule", n_tr_logs=6, n_ho_logs=3, per_log=24, seed=5)
    o_sh, o_red = tmp / "rule_sh", tmp / "rule_red"
    r1 = trainer(_swap(argv_for(w, o_sh, extra=["--smoke-steps", "2000"]), "--save-steps", "0,250,500,1000,2000"))
    r2 = trainer(_swap(argv_for(w, o_red, perm=SHG.RED_PERM_SEED, extra=["--smoke-steps", "2000"]), "--save-steps", "0,250,500,1000,2000"))
    assert r1.returncode == 0 and r2.returncode == 0, (r1.stderr[-400:], r2.stderr[-400:])
    s, rd = json.loads((o_sh / "heldout_select.json").read_text()), json.loads((o_red / "heldout_select.json").read_text())
    p0, p_red = s["per_step"]["0"], rd["per_step"]["2000"]
    pc = s["per_step"][str(s["chosen_step"])]
    assert s["chosen_step"] > 0 and pc["safety_miss_rate"] < p0["safety_miss_rate"], s
    assert all(s["per_step"][k]["safety_miss_rate"] >= pc["safety_miss_rate"] for k in s["per_step"]), "the rule did not take the minimum"
    for k in ("auc_nc", "auc_dac", "auc_ttc"):
        assert p_red[k] < p0[k], (k, p0[k], p_red[k])
    assert p_red["safety_miss_rate"] > p0["safety_miss_rate"], (p0, p_red)
    assert "descriptive_cache_pdms" in p0 and "pdms" not in p0 and "rule" in s and "safety" in s["rule"]
    ties = {"0": 1.0, "250": 1.0}
    assert min(int(k) for k, v in ties.items() if v == max(ties.values())) == 0
    ok(f"held-out rule (safety labels only) chooses step {s['chosen_step']} (miss rate {p0['safety_miss_rate']:.3f} -> {pc['safety_miss_rate']:.3f}); CTRL-RED (permuted "
       f"labels): AUC nc/dac/ttc fall and the miss rate rises ({p_red['safety_miss_rate']:.3f})")


def t_refusals(tmp: Path, w: dict):
    for arm, nm in SHG.PROFILE_NAME.items():
        prof = L.PROFILES[nm]
        a = argv_for(w, tmp / f"r_{arm}", seed=SHG.ARMS[arm]["seed"], perm=SHG.ARMS[arm]["perm"])
        assert L.profile_refusals(prof, a) == [], (nm, L.profile_refusals(prof, a))
        assert SHG.profile_name_for_argv(a) == nm
    p0 = L.PROFILES["refe_safehead_s0"]
    base = argv_for(w, tmp / "r_s0")
    bad = {"extra flag": base + ["--smoke-steps", "3"], "rows": _swap(base, "--rows", "0,1,2,3"), "lr": _swap(base, "--lr", "0.001"),
           "seed": _swap(base, "--seed", "1"), "perm on s0": base + ["--permute-labels-seed", "7"], "save": _swap(base, "--save-steps", "0,2000"),
           "twice": base + ["--seed", "0"], "build": base + ["--build-only"]}
    for k, a in bad.items():
        assert L.profile_refusals(p0, a), f"{k} was not refused"
    red = argv_for(w, tmp / "r_red")
    assert L.profile_refusals(L.PROFILES["refe_safehead_red"], red), "red without --permute-labels-seed was not refused"
    full = tmp / "r_full"
    full.mkdir()
    (full / "x").write_text("x")
    assert any("not empty" in r for r in L.profile_refusals(p0, argv_for(w, full)))
    orig = T.build_parser

    def bp():
        ap = orig()
        ap.add_argument("--new-lever", type=float, default=0.0)
        return ap
    T.build_parser = bp
    try:
        assert any("NO G-DVB entry" in r for r in L.profile_refusals(p0, base)), "a new flag without a G-DVB entry was not refused"
    finally:
        T.build_parser = orig
    ok("refusals: the three template argvs are clean; 9 moved / extra / missing levers, a non-empty --out and a new flag without a G-DVB entry are refused")


def t_profiles_unmoved():
    pin = json.loads(PRE_FP_FILE.read_text(encoding="utf-8")) if PRE_FP_FILE.is_file() else PRE_FP
    assert all(pin.get(k) == v for k, v in PRE_FP.items()), "the inline pin disagrees with the 40-profile file"
    moved = [n for n, fp in pin.items() if hashlib.sha256(repr(sorted(L.PROFILES[n].items())).encode()).hexdigest()[:16] != fp]
    assert not moved, f"pre-existing profiles moved: {moved}"
    assert set(SHG.PROFILE_NAME.values()) <= set(L.PROFILES) - set(pin), sorted(set(L.PROFILES) - set(pin))     # a LATER profile of another card must not turn this RED
    ok(f"no pre-existing profile moved ({len(pin)} pinned); the three SAFE-HEAD arms are registered")


def t_gate_clean(tmp: Path, w: dict):
    prof = test_profile(w)
    st = gate(prof, argv_for(w, tmp / "g_clean"), only=[c for c in SHG.CHECKS if c != "G-SUITE"] + ["G-CLOCK"])
    assert all(st[c] == "PASS" for c in SHG.CHECKS if c != "G-SUITE") and st["G-CLOCK"] == "NA", st
    real = L.PROFILES["refe_safehead_s0"]
    st = gate(real, argv_for(w, tmp / "g_real"), only=["G-DVB"])
    assert st["G-DVB"] == "FAIL", "the REAL profile accepted a checkpoint that is not C1"
    ok("gate: G-HYG G-DVB G-LIVE G-EVAL G-CKPT PASS on the synthetic cache, G-CLOCK NA; the real profile's G-DVB refuses a non-C1 checkpoint")


ARM_CHECK = {"sh_hyg_cfg_guard_removed": "G-HYG", "sh_hyg_undeclared_read": "G-HYG", "sh_dvb_lr_drift": "G-DVB", "sh_dvb_new_flag": "G-DVB",
             "sh_live_term_dropped": "G-LIVE", "sh_reach_row_dropped": "G-LIVE", "sh_reach_ep_in_loss": "G-LIVE", "sh_eval_rowshift": "G-EVAL",
             "sh_ckpt_bias_dropped": "G-CKPT", "sh_ckpt_nondet": "G-CKPT", "suite_failing_test": "G-SUITE"}


def t_gate_arms(tmp: Path, w: dict):
    prof = test_profile(w)
    for arm, chk in ARM_CHECK.items():
        st = gate(prof, argv_for(w, tmp / f"arm_{arm}"), only=[chk], arm=arm)
        assert st[chk] == "FAIL", (arm, st)
    ok(f"every deliberate-regression arm turns its check RED ({len(ARM_CHECK)} arms over G-HYG G-DVB G-LIVE G-EVAL G-CKPT G-SUITE)")


def t_e2e(tmp: Path, w: dict):
    prof = test_profile(w)
    keyf = tmp / "keys" / "gate.key"
    key = GT.load_key(keyf, create=True)
    out = tmp / "e2e_gate"
    run_out = tmp / "e2e_run"
    argv = argv_for(w, run_out)
    L.run_checks(prof, argv, HERE.parent, out, commit="ab" * 20, key=key)
    verdict, path, tok = L.finalize(prof, argv, HERE.parent, out, key, commit="ab" * 20)
    assert verdict == "PASS", tok["reasons"]
    okv, why, _ = GT.verify_token(path, argv, HERE.parent, key=key, profile="refe_safehead_s0", data_flags=SHG.DATA_FLAGS)
    assert okv, why
    okv, why, _ = GT.verify_token(path, _swap(argv, "--lr", "0.001"), HERE.parent, key=key, profile="refe_safehead_s0", data_flags=SHG.DATA_FLAGS)
    assert not okv
    env = {"TANITAD_GATE_KEY_FILE": str(keyf), T.TOKEN_ENV: str(path)}
    r = trainer(_swap(argv, "--lr", "0.001"), env)
    assert r.returncode != 0 and "does not verify" in (r.stdout + r.stderr), r.stderr[-300:]
    r = trainer(argv, env)
    assert r.returncode == 0 and "ZZSH_TRAIN_DONE" in r.stdout, r.stderr[-600:]
    rj = json.loads((run_out / "run.json").read_text())
    assert rj["token"]["profile"] == "refe_safehead_s0" and (run_out / "head_step02000.npz").is_file()
    assert D.load_head(run_out / "head_step02000.npz")["meta"]["smoke"] is False
    ok("end to end: the gate mints PASS, verify MATCHes the argv and REFUSES a moved lever, the trainer refuses a mismatched token and trains on a matching one")


def main() -> int:
    tmp = mkdtemp()
    try:
        t_mapping()
        t_agg32()
        t_delta()
        t_profiles_unmoved()
        t_contract(tmp)
        w = make_world(tmp / "w")
        t_leak(tmp)
        t_feat(tmp, w)
        t_token_refusal(tmp, w)
        t_reach(tmp, w)
        t_rule_and_red(tmp)
        t_refusals(tmp, w)
        t_gate_clean(tmp, w)
        if not FAST:
            t_gate_arms(tmp, w)
            t_e2e(tmp, w)
    except Exception as e:                                                      # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"ZZSAFEHEAD_FAIL {type(e).__name__}: {str(e)[:300]}")
        return 1
    print(f"ZZSAFEHEAD_PASS {N}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
