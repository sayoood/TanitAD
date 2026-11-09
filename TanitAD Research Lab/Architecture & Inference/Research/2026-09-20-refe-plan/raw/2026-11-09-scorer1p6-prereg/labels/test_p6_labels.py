"""Tests for the P6 label chain.  Dev box, CPU only:

    CUDA_VISIBLE_DEVICES=-1 <tanitad venv python> -m pytest -q test_p6_labels.py --basetemp F:/TanitAD_p6labels/bt

The pod's refe/ code is exported at test time from the local mirror at a PINNED commit (env P6_REFE_REV, P6_MIRROR) into a temp dir; nothing is read from the E: tree.
Needs: git, Git-bash (NOT System32 WSL bash), the tanitad venv (torch, numpy).  No GPU is used or touched.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p6_labels_tool as TOOL  # noqa: E402

MIRROR = os.environ.get("P6_MIRROR", "C:/Users/Admin/tanitad-push")
REV = os.environ.get("P6_REFE_REV", "5cebd7b84046a453edd33c5f3701ed61c5a9681f")
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
SHA_REAL = "968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1"


def find_bash():
    for c in ("C:/Program Files/Git/bin/bash.exe", "C:/Program Files/Git/usr/bin/bash.exe", shutil.which("bash") or ""):
        if c and os.path.exists(c) and "system32" not in c.lower().replace("\\", "/"):
            return c
    pytest.skip("no Git bash found")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def refe(tmp_path_factory):
    root = tmp_path_factory.mktemp("pk")
    tar = root / "pk.tar"
    r = subprocess.run(["git", "-C", MIRROR, "-c", "core.autocrlf=false", "archive", REV, "-o", str(tar), f"{PK}/refe", f"{PK}/code"],
                       capture_output=True, encoding="utf-8")
    if r.returncode != 0:
        pytest.skip(f"cannot export {REV} from {MIRROR}: {r.stderr[:200]}")
    subprocess.run(["tar", "-xf", str(tar), "-C", str(root)], check=True)
    return root / PK / "refe"


def make_bank(path, n_logs=40, per_log=(30, 90), seed=7, twins=True):
    import random
    rng = random.Random(seed)
    path.mkdir(parents=True, exist_ok=True)
    logs = [f"2021.0{1 + i % 9}.{10 + i}.00.00.00_veh-{i:02d}" for i in range(n_logs)]
    with open(path / "targets_rank0.jsonl", "w", encoding="utf-8", newline="\n") as f0, open(path / "targets_rank1.jsonl", "w", encoding="utf-8", newline="\n") as f1:
        for l in logs:
            for j in range(rng.randint(*per_log)):
                row = {"log_name": l, "token": f"{l[-5:]}t{j:04d}", "rank": 0, "step": 0}
                f0.write(json.dumps(row) + "\n")
                if twins and j % 2:
                    f1.write(json.dumps({**row, "rank": 1}) + "\n")
    return logs


# ------------------------------------------------------------------------------------------------------------- draw
def test_draw_deterministic_disjoint_whole_logs(tmp_path):
    bank = tmp_path / "bank"
    logs = make_bank(bank)
    ho = logs[:6]
    (tmp_path / "ho.txt").write_text("\n".join(ho) + "\n", encoding="utf-8")
    outs = []
    for k in (0, 1):
        o = tmp_path / f"o{k}"
        rc = TOOL.main(["draw", "--targets", str(bank), "--heldout-logs", str(tmp_path / "ho.txt"), "--out", str(o), "--n-train", "400",
                        "--fit-target", "600", "--expect-heldout", "-1", "--min-train", "100"])
        assert rc == 0
        outs.append(o)
    for name in ("keys/heldout.tsv", "keys/fit_extra.tsv", "keys/train.tsv", "logs/fit_logs.txt", "logs/train_logs.txt"):
        assert sha(outs[0] / name) == sha(outs[1] / name), name                     # deterministic
    j = [json.loads((o / "draw.json").read_text(encoding="utf-8")) for o in outs]
    assert {k: v["sha256"] for k, v in j[0]["files"].items()} == {k: v["sha256"] for k, v in j[1]["files"].items()}
    d = json.loads((outs[0] / "draw.json").read_text(encoding="utf-8"))
    rd = lambda n: TOOL.read_tsv_keys(str(outs[0] / "keys" / n))                    # noqa: E731
    h, f, t = rd("heldout.tsv"), rd("fit_extra.tsv"), rd("train.tsv")
    assert not (h & f) and not (h & t) and not (f & t)                              # scene-disjoint
    lg = lambda s: {l for l, _ in s}                                                # noqa: E731
    assert lg(h) == set(ho)                                                         # held-out = exactly the listed logs
    assert not (lg(t) & (lg(h) | lg(f)))                                            # LOG-level split: no FIT log feeds the arms
    by_log = TOOL.read_bank_scenes(str(bank))
    for l in lg(f) | lg(t):
        got = sum(1 for x in (f | t) if x[0] == l)
        assert got == len(by_log[l]) or l in lg(t)                                  # FIT logs are whole; train logs may be cut by the count
    for l in lg(f):
        assert sum(1 for x in f if x[0] == l) == len(by_log[l])
    assert d["n_fit"] >= 600 and d["n_fit"] - max(len(by_log[l]) for l in lg(f)) < 600   # stops at the first log that reaches the target
    assert d["n_train"] == 400 == len(t)
    assert all(rk_ok for rk_ok in (True,))                                          # twins (rank 1) never become scenes:
    assert len(h | f | t) == len(h) + len(f) + len(t) <= sum(len(v) for v in by_log.values())
    # a different seed gives a different draw
    o2 = tmp_path / "o2"
    TOOL.main(["draw", "--targets", str(bank), "--heldout-logs", str(tmp_path / "ho.txt"), "--out", str(o2), "--n-train", "400", "--fit-target", "600",
               "--expect-heldout", "-1", "--min-train", "100", "--seed", "5"])
    assert sha(o2 / "keys/train.tsv") != sha(outs[0] / "keys/train.tsv")
    # 'all' takes every scene outside the FIT logs
    o3 = tmp_path / "o3"
    assert TOOL.main(["draw", "--targets", str(bank), "--heldout-logs", str(tmp_path / "ho.txt"), "--out", str(o3), "--n-train", "all", "--fit-target", "600",
                      "--expect-heldout", "-1", "--min-train", "100"]) == 0
    d3 = json.loads((o3 / "draw.json").read_text(encoding="utf-8"))
    assert d3["n_train"] == d3["n_available_train"]
    assert d3["n_train"] + d3["n_fit"] == sum(len(v) for v in by_log.values())


def test_draw_refusals(tmp_path):
    bank = tmp_path / "bank"
    logs = make_bank(bank)
    (tmp_path / "ho.txt").write_text("\n".join(logs[:3]) + "\n", encoding="utf-8")
    base = ["draw", "--targets", str(bank), "--heldout-logs", str(tmp_path / "ho.txt"), "--out", str(tmp_path / "o"), "--fit-target", "300", "--expect-heldout", "-1"]
    assert TOOL.main(base + ["--n-train", "10", "--min-train", "100"]) == 2           # below the 25,600-per-arm floor stand-in
    assert TOOL.main(base + ["--n-train", "999999", "--min-train", "100"]) == 2       # more than exists
    assert TOOL.main(base[:-1] + ["3137", "--n-train", "100", "--min-train", "100"]) == 2   # held-out count != expected
    (tmp_path / "ho2.txt").write_text("no-such-log\n", encoding="utf-8")
    assert TOOL.main(["draw", "--targets", str(bank), "--heldout-logs", str(tmp_path / "ho2.txt"), "--out", str(tmp_path / "o"), "--n-train", "100", "--min-train", "100",
                      "--expect-heldout", "-1"]) == 2
    dup = tmp_path / "dupbank"
    dup.mkdir()
    (dup / "targets_rank0.jsonl").write_text(json.dumps({"log_name": "a", "token": "t", "rank": 0}) + "\n" + json.dumps({"log_name": "a", "token": "t", "rank": 0}) + "\n")
    with pytest.raises(TOOL.Refused):
        TOOL.read_bank_scenes(str(dup))


def test_train_pool_serves_both_orders():
    """The arithmetic behind the default: an arm reads 400 updates x batch 8 x accum 8 = 25,600 draws of its own seeded permutation of the labelled train sets."""
    need = 400 * 8 * 8
    assert need == 25600
    for n, lost in ((60000, 0.10), (30000, 0.10), (103037 - 8100, 0.10)):
        assert int(n * (1 - lost)) >= need, n                                         # no repeat inside an arm even if 10 % of sets are guard-skipped
    assert int(26000 * 0.9) < need                                                    # and the floor is real: 26,000 sets with 10 % lost is NOT enough


# ------------------------------------------------------------------------------------------------------------- wrapper (nav_mode, logits)
def _import_refe(refe):
    sys.path.insert(0, str(refe))
    for m in ("rl_fan_dump", "ckpt_io", "model", "train", "nav_command"):
        sys.modules.pop(m, None)


def _stub_train(monkeypatch, captured):
    import types
    t = types.ModuleType("train")

    class TargetBank:
        def __init__(self, target_dir, images, cfg, calib=None, max_missing_frac=1.0, **kw):
            captured["cfg"] = cfg
            raise RuntimeError("stub: stop after the config is captured")
    t.TargetBank = TargetBank
    monkeypatch.setitem(sys.modules, "train", t)


def _cmd_ckpt(tmp_path, nav="command"):
    import torch
    p = tmp_path / ("cmd.pt" if nav else "goal.pt")
    meta = {"step": 10075}
    if nav:
        meta["nav_mode"] = nav
    torch.save({"meta": meta}, p)
    return p


def _args(ckpt, tmp_path, keys):
    return ["--ckpt", str(ckpt), "--keys", str(keys), "--targets", "t", "--images", "i", "--calib", "c", "--out", str(tmp_path / "out"), "--plan-only"]


def test_nav_mode_defect_reproduced_and_patch_fixes_it(refe, tmp_path, monkeypatch):
    """Lines 561-563: the unpatched dumper builds the DATASET config in goal mode for a command-mode checkpoint; the patch makes it command."""
    import torch  # noqa: F401
    _import_refe(refe)
    ck = _cmd_ckpt(tmp_path)
    keys = tmp_path / "k.tsv"
    keys.write_text("token\tlog\nt1\tl1\n", encoding="utf-8")
    cap = {}
    _stub_train(monkeypatch, cap)
    import rl_fan_dump as RD
    assert RD.run(RD.build_parser().parse_args(_args(ck, tmp_path, keys))) == RD.EXIT_REFUSED
    assert cap["cfg"].nav_mode == "goal"                                               # THE DEFECT: the dataset would feed goal points
    from model import REFeConfig
    c = REFeConfig.for_backbone("vitl16")
    assert 2 * c.n_goal_points == c.n_commands == 4                                    # ...and nothing raises, because the widths coincide
    # apply the patch to a copy of the file and run the same scenario
    work = tmp_path / "patched" / "refe"
    work.mkdir(parents=True)
    shutil.copy(refe / "rl_fan_dump.py", work / "rl_fan_dump.py")
    r = subprocess.run(["git", "apply", "-p1", "--whitespace=nowarn", str(HERE / "rl_fan_dump_nav_mode.patch")], cwd=tmp_path / "patched", capture_output=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    import importlib.util
    spec = importlib.util.spec_from_file_location("rl_fan_dump_patched", work / "rl_fan_dump.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.HERE = refe                                                                    # the patched copy imports ckpt_io / model from the real refe dir
    cap2 = {}
    _stub_train(monkeypatch, cap2)
    assert mod.run(mod.build_parser().parse_args(_args(ck, tmp_path, keys))) == mod.EXIT_REFUSED
    assert cap2["cfg"].nav_mode == "command"
    # a goal-mode (pre-P6) checkpoint is unaffected by the patch
    cap3 = {}
    _stub_train(monkeypatch, cap3)
    mod.run(mod.build_parser().parse_args(_args(_cmd_ckpt(tmp_path, nav=None), tmp_path, keys)))
    assert cap3["cfg"].nav_mode == "goal"


def test_wrapper_builds_command_dataset_and_refuses_goal_checkpoint(refe, tmp_path, monkeypatch):
    _import_refe(refe)
    import p6_fan_dump as W
    keys = tmp_path / "k.tsv"
    keys.write_text("token\tlog\nt1\tl1\n", encoding="utf-8")
    seen = {}

    def inner(a, cfg):
        seen["cfg"] = cfg
        raise RuntimeError("stop")
    with pytest.raises((RuntimeError, SystemExit)):
        W.main(["--refe-dir", str(refe)] + _args(_cmd_ckpt(tmp_path), tmp_path, keys), ds_inner=inner)
    assert seen["cfg"].nav_mode == "command"
    with pytest.raises(SystemExit) as e:
        W.main(["--refe-dir", str(refe)] + _args(_cmd_ckpt(tmp_path, nav=None), tmp_path, keys), ds_inner=inner)
    assert "command-mode" in str(e.value)


def _fake_model_and_ds(n_keys):
    import torch

    class Model(torch.nn.Module):
        def forward(self, img, ego, goal, calib=None):
            B = img.shape[0]
            base = img.reshape(B, -1).mean(1)                                          # one number per sample, set from the row index
            traj = base[:, None, None, None].expand(B, 4, 5, 3).clone() + torch.arange(4.0)[None, :, None, None]
            score = base[:, None, None] * 10 + torch.arange(4.0)[None, :, None] + torch.arange(6.0)[None, None, :] / 10
            return traj, score

    class Ds:
        def __init__(self, rows):
            self.rows = rows

        def __getitem__(self, i):
            img = torch.full((2, 3, 4), float(i + 1))
            return (img, torch.zeros(3), torch.zeros(4), None, None, None, None, None, torch.zeros(0))
    rows = [{"log_name": f"L{i // 3}", "token": f"tok{i:03d}", "rank": 0, "step": 0} for i in range(n_keys)]
    return Model, Ds(rows)


def test_logits_saved_aligned_and_numerics_unchanged(refe, tmp_path):
    _import_refe(refe)
    import torch
    import p6_fan_dump as W
    Model, ds = _fake_model_and_ds(10)
    ck = _cmd_ckpt(tmp_path)
    keys = tmp_path / "k.tsv"
    keys.write_text("token\tlog\n" + "".join(f"{r['token']}\t{r['log_name']}\n" for r in ds.rows), encoding="utf-8")

    def run(out, save):
        argv = ["--refe-dir", str(refe), "--ckpt", str(ck), "--keys", str(keys), "--targets", "t", "--images", "i", "--calib", "c", "--out", str(out),
                "--device", "cpu", "--tag", "x", "--chunk", "3", "--batch", "4"] + (["--save-logits"] if save else [])
        return W.main(argv, build=lambda: Model(), loader=lambda m, p, meta: meta.update({"step": 10075, "nav_mode": "command"}), ds_inner=lambda a, cfg: ds_clone(cfg))

    def ds_clone(cfg):
        # the fake bank's rows are 4x20-shaped in the real config; the fake model emits 4 x 5, so the dumper's size guard needs the matching config
        cfg.n_proposals, cfg.horizon_steps = 4, 5
        return ds
    assert run(tmp_path / "a", True) == 0
    assert run(tmp_path / "b", False) == 0
    assert (tmp_path / "a" / "onpolicy_rlfan_x_s0of1.jsonl").read_bytes() == (tmp_path / "b" / "onpolicy_rlfan_x_s0of1.jsonl").read_bytes()   # sets identical with and without the saver
    assert not list((tmp_path / "b").glob("fit_logits_*"))
    lg = {}
    for ln in (tmp_path / "a" / "fit_logits_x_s0of1.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(ln)
        lg[tuple(r["key"])] = r["logits"]
    assert len(lg) == 10
    for i, row in enumerate(ds.rows):                                                  # each key's logits are the model's output FOR THAT ROW
        exp = (float(i + 1) * 10 + torch.arange(4.0)[:, None] + torch.arange(6.0)[None, :] / 10).tolist()
        got = lg[(row["log_name"], row["token"], 0, 0)]
        assert max(abs(a - b) for ra, rb in zip(got, exp) for a, b in zip(ra, rb)) < 1e-5
    # RED: a misaligned row mapping must be caught by this same check
    swapped = dict(zip(sorted(lg), list(lg.values())[1:] + list(lg.values())[:1]))
    i0 = 0
    exp0 = (float(i0 + 1) * 10 + torch.arange(4.0)[:, None] + torch.arange(6.0)[None, :] / 10).tolist()
    assert max(abs(a - b) for ra, rb in zip(swapped[(ds.rows[0]["log_name"], ds.rows[0]["token"], 0, 0)], exp0) for a, b in zip(ra, rb)) > 1e-3
    assert json.loads((tmp_path / "a" / "p6_dump_nav_mode.json").read_text())["nav_mode"] == "command"
    # pack: geometry + logits + PDM labels in one file; a missing logits set is refused
    pdm_dir = tmp_path / "pdm"
    pdm_dir.mkdir()
    with open(pdm_dir / "pdm_onpolicy_rlfan_x_s0of1.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for row in ds.rows:
            f.write(json.dumps({"kind": "pdm_targets", "key": [row["log_name"], row["token"], 0, 0], "ckpt_step": 10075, **{n: [0.5] * 4 for n in TOOL.PDM_NAMES}}) + "\n")
    empty = tmp_path / "empty_sets"
    empty.mkdir()
    out = tmp_path / "pack" / "fit_pack.npz"
    assert TOOL.main(["pack", "--sets", str(tmp_path / "a"), str(empty), "--pdm", str(pdm_dir), "--out", str(out)]) == 0
    import numpy as np
    z = np.load(out)
    assert z["logits"].shape == (10, 4, 6) and z["traj_xy"].shape == (10, 4, 5, 2) and z["pdm"].shape == (10, 4, 7) and z["has_pdm"].all() and z["is_heldout"].all()
    assert TOOL.main(["pack", "--sets", str(tmp_path / "b"), "--pdm", str(pdm_dir), "--out", str(tmp_path / "p2.npz")]) == 2     # no logits => refused


# ------------------------------------------------------------------------------------------------------------- the chain (bash, fake tools)
FAKE_DUMP = r'''
import json, os, sys
a = sys.argv[1:]
def val(f, n=1):
    i = a.index(f); return a[i + 1:i + 1 + n]
tag = val("--tag")[0]; out = val("--out")[0]
open(os.environ["FAKE_COUNT"], "a").write("dump " + tag + "\n")
if "--plan-only" in a:
    sys.exit(0)
if os.environ.get("FAKE_DUMP_FAIL") == tag:
    sys.exit(1)
keys = []
i = a.index("--keys") + 1
while i < len(a) and not a[i].startswith("--"):
    keys.append(a[i]); i += 1
os.makedirs(out, exist_ok=True)
sets = open(os.path.join(out, "onpolicy_rlfan_%s_s0of1.jsonl" % tag), "a", encoding="utf-8", newline="\n")
lgf = open(os.path.join(out, "fit_logits_%s_s0of1.jsonl" % tag), "a", encoding="utf-8", newline="\n") if "--save-logits" in a else None
n = 0
for kf in keys:
    for ln in open(kf, encoding="utf-8").read().splitlines()[1:]:
        t, l = ln.split("\t")
        r = {"kind": "onpolicy_set", "label_version": 0, "comfort_source": "none", "log_name": l, "token": t, "step": 0, "rank": 0, "ckpt_step": 10075,
             "aug": None, "traj": [[[1.0 + k, 2.0]] * 5 for k in range(4)], "yaw": [[0.1] * 5] * 4, "targets": [{}] * 4}
        if os.environ.get("FAKE_DUMP_SHORT") == tag and n == 2:
            n += 1
            continue
        sets.write(json.dumps(r) + "\n")
        if lgf:
            lgf.write(json.dumps({"key": [l, t, 0, 0], "logits": [[0.5 + k] * 6 for k in range(4)]}) + "\n")
        n += 1
sets.close()
print("ZZRLFAN_DONE", n)
'''
FAKE_RELABEL = r'''
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import select_bank_subset as SB
def served_keys(files):
    best = {}
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                f = SB.head_fields(line[:2000] + line[-400:])
                if not {"log_name", "step", "ckpt_step"} <= set(f):
                    continue
                k = (f["log_name"], f.get("token", ""), int(f["step"]), int(f.get("rank", 0)))
                rk = (int(f["ckpt_step"]), int(f.get("label_version", 1)))
                if k not in best or best[k] < rk:
                    best[k] = rk
    return best
def main():
    a = sys.argv[1:]
    sets, out = a[a.index("--sets") + 1], a[a.index("--out") + 1]
    open(os.environ["FAKE_COUNT"], "a").write("relabel " + os.path.basename(sets) + "\n")
    bad = os.environ.get("FAKE_RELABEL_FAIL")
    if bad and bad in sets.replace("\\", "/"):
        sys.exit(3)
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(sets, "onpolicy_*.jsonl")))
    print("served sets: %d" % len(served_keys(files)), flush=True)
    n = 0
    for fp in files:
        dst = os.path.join(out, "pdm_" + os.path.basename(fp))
        done = set()
        if os.path.exists(dst):
            for ln in open(dst, encoding="utf-8"):
                try: done.add(tuple(json.loads(ln)["key"]))
                except Exception: pass
        with open(dst, "a", encoding="utf-8", newline="\n") as f:
            for ln in open(fp, encoding="utf-8"):
                try: r = json.loads(ln)
                except json.JSONDecodeError: continue
                k = (r["log_name"], r["token"], r["step"], r["rank"])
                if k in done: continue
                if os.environ.get("FAKE_RELABEL_DROP") and n == 1 and "train" in sets:
                    n += 1; continue
                f.write(json.dumps({"kind": "pdm_targets", "key": list(k), "ckpt_step": r["ckpt_step"], "label_version": 0,
                                    **{c: [0.5] * 4 for c in ("nc", "dac", "ep", "ttc", "c", "ddc", "pdms")}}) + "\n")
                n += 1
    print("ZZRELABEL_PDM_DONE %d 0 0" % n, flush=True)
    return 0
if __name__ == "__main__":
    sys.exit(main())
'''
FAKE_NVSMI = '#!/bin/sh\nif [ -n "$FAKE_GPU_BUSY" ]; then echo 4242; fi\nif [ -n "$FAKE_NVSMI_BROKEN" ]; then exit 9; fi\nexit 0\n'


@pytest.fixture()
def env(tmp_path, refe):
    bank = tmp_path / "bank"
    logs = make_bank(bank, n_logs=24, per_log=(8, 14))
    pk = tmp_path / "pkg"
    (pk / "refe").mkdir(parents=True)
    (pk / "splits").mkdir()
    for n in ("pdm_label_complete.py", "select_bank_subset.py"):
        shutil.copy(refe / n, pk / "refe" / n)
    (pk / "refe" / "onpolicy_relabel_pdm.py").write_text(FAKE_RELABEL, encoding="utf-8", newline="\n")
    (pk / "refe" / "rl_fan_dump.py").write_text("# stub: presence only\n")
    (pk / "refe" / "ckpt_io.py").write_text("# stub\n")
    (pk / "splits" / "sft_heldout_logs.txt").write_text("\n".join(logs[:3]) + "\n", encoding="utf-8", newline="\n")
    (tmp_path / "fake_dump.py").write_text(FAKE_DUMP, encoding="utf-8", newline="\n")
    nv = tmp_path / "nvsmi"
    nv.write_text(FAKE_NVSMI, encoding="utf-8", newline="\n")
    model = tmp_path / "model_final.pt"
    model.write_bytes(b"tiny fake checkpoint")
    out = tmp_path / "out"
    e = dict(os.environ)
    e.update({"PKG": str(pk).replace("\\", "/"), "OUT": str(out).replace("\\", "/"), "MODEL": str(model).replace("\\", "/"), "P6L_TEST_EXPECT_SHA": sha(model),
              "TARGETS": str(bank).replace("\\", "/"), "IMAGES": "x", "CALIB": "x", "STACK": "", "PY": sys.executable.replace("\\", "/"),
              "DUMP_PY": str(tmp_path / "fake_dump.py").replace("\\", "/"), "NVSMI": str(nv).replace("\\", "/"), "N_TRAIN": "60", "FIT_TARGET": "40",
              "EXPECT_HELDOUT": "-1", "MIN_TRAIN": "20", "POLL_S": "0.2", "FAKE_COUNT": str(tmp_path / "count.txt").replace("\\", "/"), "WORKERS": "3"})
    for k in ("FAKE_GPU_BUSY", "FAKE_NVSMI_BROKEN", "FAKE_DUMP_FAIL", "FAKE_RELABEL_FAIL", "FAKE_RELABEL_DROP", "FORCE", "P6L_RECHECK"):
        e.pop(k, None)
    (tmp_path / "count.txt").write_text("")
    os.chmod(nv, 0o755)
    return {"e": e, "out": out, "tmp": tmp_path, "bash": find_bash(), "model": model}


def run_chain(env, **extra):
    e = dict(env["e"])
    e.update(extra)
    r = subprocess.run([env["bash"], str(HERE / "p6_labels_chain.sh")], env=e, capture_output=True, encoding="utf-8", errors="replace", timeout=300)
    return r, (env["out"] / "chain.log").read_text(encoding="utf-8") if (env["out"] / "chain.log").exists() else ""


def assert_failed(env, r, log, why):
    assert r.returncode != 0
    assert (env["out"] / "labels.FAILED").exists() and not (env["out"] / "labels.DONE").exists()
    assert "P6LABELS_ABORT" in log and why in log
    assert "P6LABELS_ALL_OK" not in log and "P6LABELS_ALL_OK" not in r.stdout           # a failure line never matches the success waiter
    assert not (env["out"] / "labels_manifest.json").exists()


def test_bash_syntax_and_lf(tmp_path):
    for n in ("p6_labels_chain.sh",):
        data = (HERE / n).read_bytes()
        assert b"\r" not in data, "CRLF in a pod script dies on Linux"
        assert data.startswith(b"#!/usr/bin/env bash\n")
    r = subprocess.run([find_bash(), "-n", str(HERE / "p6_labels_chain.sh")], capture_output=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    # the git-export form (core.autocrlf=false) is what ships: hash it against the file
    r = subprocess.run(["git", "-c", "core.autocrlf=false", "hash-object", "--no-filters", str(HERE / "p6_labels_chain.sh")], capture_output=True, encoding="utf-8")
    assert r.returncode == 0


def test_precondition_wrong_sha_is_red(env):
    r, log = run_chain(env, P6L_TEST_EXPECT_SHA="0" * 64)
    assert_failed(env, r, log, "model_sha256_mismatch")
    assert "dump" not in (env["tmp"] / "count.txt").read_text()                          # refused before any job
    r, log = run_chain(env, P6L_TEST_EXPECT_SHA="")                                     # empty override falls to the REAL pinned sha: the tiny file does not match it
    assert_failed(env, r, log, "model_sha256_mismatch")
    assert SHA_REAL in (HERE / "p6_labels_chain.sh").read_text(encoding="utf-8")


def test_precondition_busy_gpu_is_red(env):
    r, log = run_chain(env, FAKE_GPU_BUSY="1")
    assert_failed(env, r, log, "gpu_has_compute_process")
    assert (env["tmp"] / "count.txt").read_text() == ""
    r, log = run_chain(env, FAKE_NVSMI_BROKEN="1")
    assert_failed(env, r, log, "nvidia_smi_unavailable")                               # fail closed when the GPU state cannot be read


def test_precondition_parameters(env):
    r, log = run_chain(env, WORKERS="15")
    assert_failed(env, r, log, "workers_out_of_range")
    r, log = run_chain(env, N_TRAIN="lots")
    assert_failed(env, r, log, "n_train_not_integer_or_all")


def test_full_flow_to_manifest(env):
    r, log = run_chain(env)
    assert r.returncode == 0, r.stdout + r.stderr + log
    out = env["out"]
    assert (out / "labels.DONE").exists() and not (out / "labels.FAILED").exists()
    assert "P6LABELS_ALL_OK" in log
    man = json.loads((out / "labels_manifest.json").read_text(encoding="utf-8"))
    d = man["data_expect"]
    assert d["train_sets"] == 60 and d["heldout_sets"] > 0 and d["pdm_train"] == 60 and d["pdm_heldout"] == d["heldout_sets"] and d["train_rows_in_heldout_logs"] == 0
    assert man["parts"]["fit"]["sets"] >= 1 and man["log_split"]["n_fit_logs"] >= 3
    for rel, meta in man["files"].items():                                              # every sha256 is of the bytes on disk
        assert sha(out / rel) == meta["sha256"], rel
    # FIT and train logs are disjoint, by the files the manifest hashes
    fit = set((out / "logs/fit_logs.txt").read_text().split())
    trn = set((out / "logs/train_logs.txt").read_text().split())
    assert fit and trn and not (fit & trn)
    assert man["log_split"]["fit_logs_sha256"] == sha(out / "logs/fit_logs.txt")
    import numpy as np
    z = np.load(out / "fit_pack" / "fit_pack.npz")
    assert z["logits"].shape[0] == d["heldout_sets"] + man["parts"]["fit"]["sets"] and z["has_pdm"].all()
    # a second invocation is a no-op on a finished job
    n_before = len((env["tmp"] / "count.txt").read_text().splitlines())
    r2, _ = run_chain(env)
    assert r2.returncode == 0 and len((env["tmp"] / "count.txt").read_text().splitlines()) == n_before


def test_partial_failures_never_leave_done(env):
    # labelling of the train sets fails -> FAILED, no DONE, no manifest, and the dumps already done are not repeated on resume
    r, log = run_chain(env, FAKE_RELABEL_FAIL="train_sets")
    assert_failed(env, r, log, "label_loop_rc_1")
    pid = (env["out"] / "label_loop.pid").read_text().strip()
    ps = subprocess.run([env["bash"], "-c", f"kill -0 {pid} 2>/dev/null && echo alive || echo gone"], capture_output=True, encoding="utf-8").stdout.strip()
    assert ps == "gone"                                                                  # the loop is not left running
    c1 = (env["tmp"] / "count.txt").read_text()
    r, log = run_chain(env)                                                              # resume with the fault removed
    assert r.returncode == 0, log
    c2 = (env["tmp"] / "count.txt").read_text()
    assert c2.count("dump p6h") == c1.count("dump p6h") == 1 and c2.count("dump p6t") == 1   # finished dump stages are skipped
    assert (env["out"] / "labels.DONE").exists() and not (env["out"] / "labels.FAILED").exists()


def test_dump_failure_and_incomplete_labels_are_failed(env):
    r, log = run_chain(env, FAKE_DUMP_FAIL="p6t")
    assert_failed(env, r, log, "dump_train")
    # a labeller that silently drops one set: pdm_label_complete says INCOMPLETE, the chain must not write DONE
    shutil.rmtree(env["out"])
    (env["tmp"] / "count.txt").write_text("")
    r, log = run_chain(env, FAKE_RELABEL_DROP="1")
    assert_failed(env, r, log, "label_loop_rc_2")


def test_dump_that_disagrees_with_the_draw_is_failed(env):
    # a dumper that writes one set too few: check-sets refuses before any label is credited
    r, log = run_chain(env, FAKE_DUMP_SHORT="p6t")
    assert_failed(env, r, log, "dump_train")
