#!/usr/bin/env python3
"""Validity tests for the dev-box proxy trainer (refe/proxy_train.py) and the shared cache reader. Every check has a
deliberate-regression arm that must go RED.

  T1 score_loss on the COVERED samples == train.py's masked on-policy formula over the FULL batch, written here as a
     literal copy of train.py's lines (value and every scorer gradient).
     MUTATION: the denominator from the covered count instead of the full batch -> must differ
  T2 a micro-batch with no covered sample: every scorer tensor gets a zero-VALUED gradient (AdamW then decays it, as
     live's `score.sum() * 0.0` does). MUTATION: no zero term -> the gradients are None
  T3 make_batch + Cache: covered samples' visual rows come from `vindex` (exact bits); uncovered samples read none;
     a LABELLED sample without a cached visual_ctx is REFUSED. MUTATION: a reader using `index` -> wrong / missing row
  T4 prep_sets: reads the unpacked part names (pod_extract_partN_part_XXXX.jsonl.gz), keeps ckpt_step <= max, REFUSES
     a dir with no parts. MUTATION: the old `part_*.jsonl.gz` glob finds no file in the real sets dir
  T5 gate G3 end-to-end: proxy_train.py on the REAL snapshot 015 (CPU, a tiny fp32 fake cache, real set lines):
     lr 0 -> 0 of N decoder/scorer tensors change. MUTATION: lr 1e-3 -> tensors change

    python refe/selftest_proxy_train.py   -> ZZSELFTEST_PROXYTRAIN_OK / _FAIL
    writes raw/2026-09-27-m6-proxy/selftest_proxy_train.json (checks + the tested bytes' sha256)
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PKG = HERE.parent
SNAP = "D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch015.pt"
REAL_SETS = Path("D:/Projects/TanitAD/data/refe_proxy/sets")
OUT = PKG / "raw" / "2026-09-27-m6-proxy" / "selftest_proxy_train.json"
if "--out" in sys.argv:                          # AMENDMENT 1: epoch 2's record lives apart from epoch 1's (G4 per epoch)
    OUT = Path(sys.argv[sys.argv.index("--out") + 1])
checks: dict = {}


def check(name, ok, detail=""):
    checks[name] = {"ok": bool(ok), "detail": detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def sha256(p) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def fake_cache(d: Path, n: int, visual_rows: list, dtype: str, seed: int = 0):
    """one shard, n frames; frame i has a visual row iff i in visual_rows (vindex = its position there)"""
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    npdt = np.float32 if dtype == "fp32" else np.int16
    if dtype == "fp32":
        scene = (rng.standard_normal((n, 64, 256)) * 0.5).astype(np.float32)
        vis = (rng.standard_normal((len(visual_rows), 7680, 256)) * 0.5).astype(np.float32)
    else:
        import torch
        scene = torch.randn(n, 64, 256).mul(0.5).bfloat16().view(torch.int16).numpy()
        vis = torch.randn(len(visual_rows), 7680, 256).mul(0.5).bfloat16().view(torch.int16).numpy()
    scene.astype(npdt).tofile(d / "shard_0000.scene.bin")
    vis.astype(npdt).tofile(d / "shard_0000.visual.bin")
    keys = [("fake_log_%d" % (i % 2), "tok%04d" % i, 0) for i in range(n)]
    with open(d / "frames.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i, k in enumerate(keys):
            f.write(json.dumps({"key": list(k), "shard": 0, "index": i,
                                "vindex": visual_rows.index(i) if i in visual_rows else -1}) + "\n")
    with open(d / "rows.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i, k in enumerate(keys):
            traj = [[2.0 * (t + 1), 0.05 * t, 0.01 * t] for t in range(20)]
            f.write(json.dumps({"key": list(k), "rank": 0, "ego": [8.0, 0.0, 0.1, 0.0, 0.0, 0.0, 8.0],
                                "goal": [40.0, 1.0, 0.0, 0.0], "traj": traj}) + "\n")
    json.dump({"version": 1, "split": "train", "dtype": dtype, "frames": n}, open(d / "manifest.json", "w"))
    return keys, scene, vis


def real_set_line():
    for p in sorted(REAL_SETS.glob("*.jsonl.gz")):
        for line in gzip.open(p, "rt", encoding="utf-8"):
            return json.loads(line)
    raise SystemExit(f"no real set line under {REAL_SETS}")


def write_sets(d: Path, keys_ckpt: list, name="pod_extract_part9_part_0000.jsonl.gz"):
    d.mkdir(parents=True, exist_ok=True)
    base = real_set_line()
    with gzip.open(d / name, "wt", encoding="utf-8") as f:
        for (lg, tok, st), ck in keys_ckpt:
            r = dict(base, log_name=lg, token=tok, step=st, rank=0, ckpt_step=ck)
            f.write(json.dumps(r) + "\n")


def main() -> int:
    import torch
    import torch.nn.functional as F
    import proxy_train as PT
    tmp = Path(tempfile.mkdtemp(prefix="selftest_proxytrain_"))
    torch.manual_seed(0)
    model, cfg, snap_state = PT.build(SNAP, "cpu")
    sco = [p for n, p in model.named_parameters() if n.startswith(PT.SCO) and p.requires_grad]
    names = [n for n, p in model.named_parameters() if n.startswith(PT.SCO) and p.requires_grad]

    # ---- T1: covered-only == the live masked formula
    B, M, T = 4, cfg.n_proposals, cfg.horizon_steps
    cov = [0, 2]
    vi = torch.randn(B, 7680, 256) * 0.5
    cx = torch.zeros(B, M, T, 3)
    cg = torch.zeros(B, M, 6)
    cm = torch.zeros(B, M)
    for j in cov:
        cx[j] = torch.randn(M, T, 3)
        cg[j] = torch.rand(M, 6).round()
        cm[j] = 1.0
    cov_norm, score_w = 7.3, 0.1

    def grads():
        return [None if p.grad is None else p.grad.clone() for p in sco]
    model.zero_grad(set_to_none=True)
    sx = model.score_trajectories(cx, vi)                      # live: the full batch, zero sets under mask 0
    per = F.binary_cross_entropy_with_logits(sx, cg, reduction="none").mean(-1)
    denom = max(cov_norm * float(B), 1.0)
    l_live = (per * cm).sum() / denom * score_w
    l_live.backward()
    g_live = grads()
    model.zero_grad(set_to_none=True)
    l_px = PT.score_loss(model.score_trajectories(cx[cov], vi[cov]), cg[cov], B, cov_norm, score_w, sco)
    l_px.backward()
    g_px = grads()
    rel = max(float((a - b).abs().max() / (a.abs().max() + 1e-12)) for a, b in zip(g_live, g_px))
    check("T1 covered-only loss == live masked formula (value)", abs(float(l_live) - float(l_px)) <= 1e-6 *
          abs(float(l_live)), f"live {float(l_live):.8f} proxy {float(l_px):.8f}")
    check("T1 ... every scorer gradient", rel <= 1e-4 and all(g is not None for g in g_px),
          f"max rel diff {rel:.2e} over {len(sco)} tensors")
    model.zero_grad(set_to_none=True)
    l_mut = PT.score_loss(model.score_trajectories(cx[cov], vi[cov]), cg[cov], len(cov), cov_norm, score_w, sco)
    check("T1 MUTATION (denominator from the covered count) goes RED",
          abs(float(l_mut) - float(l_live)) > 1e-3 * abs(float(l_live)), f"mutated {float(l_mut):.8f}")

    # ---- T2: no covered sample -> zero-valued grads on every scorer tensor
    model.zero_grad(set_to_none=True)
    l0 = PT.score_loss(None, None, B, cov_norm, score_w, sco)
    l0.backward()
    g0 = grads()
    check("T2 uncovered micro-batch: zero-VALUED gradient on every scorer tensor",
          all(g is not None and float(g.abs().max()) == 0.0 for g in g0) and float(l0) == 0.0,
          f"{sum(g is not None for g in g0)}/{len(g0)} tensors carry a gradient, all exactly 0")
    model.zero_grad(set_to_none=True)
    traj = PT.decode(model, torch.randn(1, 64, 256), torch.zeros(1, 7), torch.zeros(1, 4))
    traj.sum().backward()                                        # mutation: no zero term
    check("T2 MUTATION (no zero term) goes RED: scorer gradients are None",
          all(p.grad is None for p in sco), f"{sum(p.grad is None for p in sco)}/{len(sco)} None")

    # ---- T3: make_batch + Cache vindex
    cd = tmp / "cache_bf16"
    keys, scene, vis = fake_cache(cd, 4, [0, 2], "bf16")
    cache = PT.Cache(cd)
    rows = [{"log_name": k[0], "token": k[1], "step": k[2], "rank": 0, "ego": [8.0] * 7, "goal": [40.0, 1.0, 0, 0],
             "traj": [[0.0, 0.0, 0.0]] * 20} for k in keys]

    class Op:
        def __init__(self, labelled):
            self.lab = labelled

        def get(self, lg, tok, st, rk):
            if (lg, tok, st) in self.lab:
                return np.full((M, T, 3), 0.5, np.float32), np.ones((M, 6), np.float32), 4000
            return None
    b = PT.make_batch([0, 1, 2, 3], rows, cache, Op({keys[0], keys[2]}), cfg.n_goal_points)
    vb = b[4].view(torch.int16).numpy() if b[4] is not None else None
    ok3 = vb is not None and vb.shape[0] == 2 and np.array_equal(vb[0], vis[0]) and np.array_equal(vb[1], vis[1]) \
        and b[5].shape[0] == 2 and np.array_equal(b[0].view(torch.int16).numpy(), scene)
    check("T3 covered samples' visual rows read at vindex, exact bits; scene bits exact", ok3,
          f"visual rows {None if vb is None else vb.shape[0]}")
    b_none = PT.make_batch([1, 3], rows, cache, Op({keys[0], keys[2]}), cfg.n_goal_points)
    check("T3 an all-uncovered batch reads no visual_ctx", b_none[4] is None and b_none[5] is None)
    try:
        PT.make_batch([1], rows, cache, Op({keys[1]}), cfg.n_goal_points)
        refused = False
    except RuntimeError:
        refused = True
    check("T3 a LABELLED sample without a cached visual_ctx is REFUSED", refused)

    class Mut(PT.Cache):
        def visual(self, key):
            s, i, _v = self.pos[key]
            return np.array(self.mm[s][1][i])
    try:
        bm = PT.make_batch([0, 1, 2, 3], rows, Mut(cd), Op({keys[0], keys[2]}), cfg.n_goal_points)
        mut_ok = np.array_equal(bm[4].view(torch.int16).numpy()[1], vis[1])
    except (IndexError, ValueError):
        mut_ok = False
    check("T3 MUTATION (reader uses index, not vindex) goes RED", not mut_ok)

    # ---- T4: prep_sets
    sd = tmp / "sets"
    write_sets(sd, [(keys[0], 4000), (keys[2], 5000)])
    wd = PT.prep_sets(sd, 4933, tmp / "work")
    kept = [json.loads(l) for l in open(wd / "onpolicy_proxy.jsonl", encoding="utf-8")]
    check("T4 prep_sets reads pod_extract_partN_part_XXXX names and keeps ckpt_step <= 4933",
          len(kept) == 1 and kept[0]["ckpt_step"] == 4000, f"kept {len(kept)}")
    (tmp / "empty").mkdir()
    try:
        PT.prep_sets(tmp / "empty", 4933, tmp / "work2")
        ref4 = False
    except SystemExit:
        ref4 = True
    check("T4 a sets dir with no parts is REFUSED", ref4)
    old = len(list(REAL_SETS.glob("part_*.jsonl.gz")))
    new = len(list(REAL_SETS.glob("*.jsonl.gz")))
    check("T4 MUTATION (the old part_*.jsonl.gz glob) goes RED on the real sets dir", old == 0 and new > 0,
          f"old glob {old} files, fixed glob {new}")

    # ---- T5: G3 end-to-end, real snapshot, CPU
    c32 = tmp / "cache_fp32"
    k32, _, _ = fake_cache(c32, 4, [0, 1, 2, 3], "fp32", seed=1)
    s32 = tmp / "sets32"
    write_sets(s32, [(k32[0], 4000), (k32[3], 4000)])

    def run(name, lr):
        o = tmp / name
        cmd = [sys.executable, str(HERE / "proxy_train.py"), "--cache", str(c32), "--sets", str(s32), "--snapshot",
               SNAP, "--out", str(o), "--device", "cpu", "--amp", "none", "--zero-lr-steps", "0", "--steps", "2",
               "--batch", "2", "--accum", "1", "--lr-override", str(lr), "--log-every", "1"]
        p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="4"))
        sm = json.load(open(o / "summary.json")) if (o / "summary.json").exists() else {}
        return p.returncode, sm, p.stdout[-1500:] + p.stderr[-1500:]
    rc0, s0, log0 = run("g3_lr0", 0.0)
    check("T5 G3: lr 0 -> 0 decoder/scorer tensors change (real snapshot 015, CPU)",
          rc0 == 0 and s0.get("tensors_changed") == 0 and s0.get("tensors", 0) > 0 and s0.get("covered_samples", 0) > 0,
          f"rc {rc0} changed {s0.get('tensors_changed')}/{s0.get('tensors')} covered {s0.get('covered_samples')}"
          + ("" if rc0 == 0 else " LOG " + log0))
    rc1, s1, log1 = run("g3_mut", 1e-3)
    check("T5 MUTATION (lr 1e-3) goes RED: tensors change", rc1 == 0 and s1.get("tensors_changed", 0) > 0,
          f"rc {rc1} changed {s1.get('tensors_changed')}/{s1.get('tensors')}" + ("" if rc1 == 0 else " LOG " + log1))

    # ---- T6 (AMENDMENT 1): --init-from starts EXACTLY from an arm's tensors (lr 0 keeps them bit-identical)
    import torch
    snap_state = PT.build(SNAP, "cpu")[2]
    g = torch.Generator().manual_seed(7)
    fake = {k: v + 0.01 * torch.randn(v.shape, generator=g, dtype=v.dtype) if v.is_floating_point() else v.clone()
            for k, v in snap_state.items()}
    torch.save({"state": fake, "complete": True, "steps_done": 453}, tmp / "fake_arm.pt")

    def run_init(name, init):
        o = tmp / name
        cmd = [sys.executable, str(HERE / "proxy_train.py"), "--cache", str(c32), "--sets", str(s32), "--snapshot",
               SNAP, "--out", str(o), "--device", "cpu", "--amp", "none", "--zero-lr-steps", "0", "--steps", "2",
               "--batch", "2", "--accum", "1", "--lr-override", "0", "--log-every", "1"] + (
            ["--init-from", str(tmp / "fake_arm.pt")] if init else [])
        p_ = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="4"))
        st = torch.load(o / "final.pt", weights_only=False)["state"] if (o / "final.pt").exists() else {}
        return p_.returncode, st
    rc6, st6 = run_init("init_on", True)
    same6 = bool(st6) and all(torch.equal(st6[k], fake[k]) for k in fake)
    check("T6 --init-from: lr 0 ends bit-identical to the INIT tensors (not the snapshot)",
          rc6 == 0 and same6 and not all(torch.equal(st6[k], snap_state[k]) for k in snap_state), f"rc {rc6}")
    rc6m, st6m = run_init("init_off", False)
    check("T6 MUTATION (the flag ignored) goes RED: the run ends on the snapshot, not the init",
          rc6m == 0 and bool(st6m) and not all(torch.equal(st6m[k], fake[k]) for k in fake))

    # ---- T8 (M6b): --yaw-loss plain_tangent --tan-w reaches the loss, end to end through the real trainer
    def run_tan(name, mode, tw):
        o = tmp / name
        cmd = [sys.executable, str(HERE / "proxy_train.py"), "--cache", str(c32), "--sets", str(s32), "--snapshot",
               SNAP, "--out", str(o), "--device", "cpu", "--amp", "none", "--zero-lr-steps", "0", "--steps", "2",
               "--batch", "2", "--accum", "1", "--lr-override", "1e-3", "--log-every", "1", "--yaw-loss", mode,
               "--tan-w", str(tw)]
        p_ = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="4"))
        st = torch.load(o / "final.pt", weights_only=False)["state"] if (o / "final.pt").exists() else {}
        cf = json.load(open(o / "config.json")) if (o / "config.json").exists() else {}
        return p_.returncode, st, cf
    rcp, stp, _ = run_tan("tan_plain", "plain", 0.1)
    rc0, st0, cf0 = run_tan("tan_w0", "plain_tangent", 0.0)
    rc5, st5, cf5 = run_tan("tan_w5", "plain_tangent", 0.5)
    same0 = bool(stp) and bool(st0) and all(torch.equal(stp[k], st0[k]) for k in stp)
    check("T8 plain_tangent with --tan-w 0 is BIT-IDENTICAL to plain (the term is the only difference)",
          rcp == 0 and rc0 == 0 and same0 and cf0.get("args", {}).get("tan_w") == 0.0)
    diff5 = bool(st5) and not all(torch.equal(stp[k], st5[k]) for k in stp)
    check("T8 MUTATION arm: --tan-w 0.5 changes the weights (the flag is not ignored)", rc5 == 0 and diff5,
          f"rc {rc5}")

    # ---- T7 (AMENDMENT 1): --skip-samples CONTINUES the seed's stream (across an epoch boundary)
    import itertools
    import train as T
    groups = [[i] if i % 3 else [i, 100 + i] for i in range(10)]
    s_a, s_b = T.SceneEpochSampler(groups, 0, True), T.SceneEpochSampler(groups, 0, True)
    N, K = 23, 17                                       # 10 scenes per epoch: N crosses two epoch boundaries
    full = list(itertools.islice(PT.sample_stream(s_a, 0), N + K))
    cont = list(itertools.islice(PT.sample_stream(s_b, N), K))
    check("T7 --skip-samples N yields exactly samples N+1..N+K of the unskipped stream", cont == full[N:],
          f"{cont[:6]} vs {full[N:N + 6]}")
    s_c = T.SceneEpochSampler(groups, 0, True)
    naive = list(itertools.islice(PT.sample_stream(s_c, 0), K))       # mutation: 'continue' by restarting at epoch 0
    check("T7 MUTATION (restarting the stream instead of continuing it) goes RED", naive != full[N:])

    n_ok = sum(v["ok"] for v in checks.values())
    res = {"checks": checks, "n_checks": len(checks), "failed": [k for k, v in checks.items() if not v["ok"]],
           "tested_sha256": {f: sha256(HERE / f) for f in ("proxy_train.py", "proxy_cache.py", "measures.py",
                                                           "selftest_proxy_train.py")},
           "snapshot": SNAP}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8", newline="\n"), indent=1)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  {n_ok}/{len(checks)} checks pass -> {OUT}")
    print("ZZSELFTEST_PROXYTRAIN_OK" if n_ok == len(checks) else "ZZSELFTEST_PROXYTRAIN_FAIL")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
