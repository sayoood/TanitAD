#!/usr/bin/env python3
"""smoke_ri.py -- the real-checkpoint smoke of the REINDEXED tool (8 threads, CPU, 32 windows): C0 s0 (M2b_s0_recft_C0), the reader's flags (--replan-steps 10 --nav deadreckon --schedule periodic --limit causal
--smooth sg72 --dump-ticks --dump-heads), A = --scene logged, B = --scene reindexed.  Checks:
  (a) the checkpoint's sha256 equals its SHA256SUMS.txt line;  (b) B's scene_reindex receipt (items, rows by class, lag distribution, lateral offset, past_rows_bit_identical true, future frames 0), A carries none;
  (c) the largest plan difference A vs B (ticks 'plan' and 'plan_raw'), at tick 0 (must be exactly 0: only past rows) and at ticks >= 1, and the largest executed-path difference (dumps 'cl');
  (d) the windows and the P10 floor of A and B are equal (the floor reads no frames);
  (e) KRI4 on the real rig, 6 windows: --scene logged / frozen / hold_now of the SOURCE (FRAMEFREEZE) tool and the RI tool give byte-identical dumps and ticks and the same header.
python -I code/smoke_ri.py  ->  logs/smoke_ri.out"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

S = "F:/TanitAD-scratch/motion_c/ri"
ART = "E:/Projects/TanitAD-artifacts"
DATA = f"{ART}/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
CKD = f"{ART}/v8f-recover/M2b_s0_recft_C0"
TOOL_RI = f"{S}/tree_ri/stack/scripts/v8f_t1_eval_ri.py"
TOOL_SRC = f"{S}/tree_ri/stack/scripts/v8f_t1_eval_framefreeze.py"
TH = 8
env = dict(os.environ, OMP_NUM_THREADS=str(TH), MKL_NUM_THREADS=str(TH), OPENBLAS_NUM_THREADS=str(TH), PYTHONDONTWRITEBYTECODE="1")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(tool, label, n_win, *flags):
    out = f"{S}/smoke/{label}"
    os.makedirs(out, exist_ok=True)
    cmd = [PY, "-I", tool, "run", "--ckpt", f"{CKD}/ckpt_last.pt", "--tag", f"SMOKERI_{label}",
           "--tracks", f"{ART}/sfresh_inputs/inputs/native_tracks.pt", "--eval-cache", f"{ART}/sfresh_inputs/cache", "--eval-grid", f"{DATA}/eval_grid_windows.json", "--eval-join", f"{DATA}/join_eval141.npz",
           "--out-dir", out, "--replan-steps", "10", "--nav", "deadreckon", "--schedule", "periodic", "--limit", "causal", "--smooth", "sg72", "--dump-ticks", f"{out}/ticks", "--dump-heads",
           "--device", "cpu", "--max-windows", str(n_win), "--n-boot", "20", *flags]
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode:
        print(label, "RUN FAILED rc", r.returncode, r.stderr[-1200:])
        sys.exit(1)
    z = {k: v for k, v in np.load(f"{out}/ticks/ticks_SMOKERI_{label}.npz", allow_pickle=True).items() if k not in ("tag", "ckpt")}
    dd = {}
    for f in sorted(glob.glob(f"{out}/dump_SMOKERI_{label}/ep*.npz")):
        e = np.load(f)
        for k in e.files:
            dd.setdefault(k, []).append(e[k])
    hdr = json.load(open(f"{out}/t1v8f_SMOKERI_{label}.json", encoding="utf-8"))["t1v8f"]
    print(f"SMOKERI run {label} [{os.path.basename(tool)}]: {time.time() - t0:.0f} s, {len(z['tick'])} tick rows", flush=True)
    return z, {k: np.concatenate(v) for k, v in dd.items()}, hdr, out


def same(za, da, ha, zb, db, hb):
    ok = set(za) == set(zb) and all(za[k].tobytes() == zb[k].tobytes() and za[k].dtype == zb[k].dtype for k in za)
    ok &= set(da) == set(db) and all(da[k].tobytes() == db[k].tobytes() for k in da)
    ok &= {k: v for k, v in ha.items() if k not in ("ticks_file", "tag")} == {k: v for k, v in hb.items() if k not in ("ticks_file", "tag")}
    return bool(ok)


want = [l.split()[0] for l in open(f"{CKD}/SHA256SUMS.txt") if l.strip().endswith("ckpt_last.pt")][0]
got = sha(f"{CKD}/ckpt_last.pt")
assert got == want, (got, want)
print(f"SMOKERI (a) checkpoint {CKD}/ckpt_last.pt sha256 {got} == SHA256SUMS.txt line")
print(f"SMOKERI tool sha256 {sha(TOOL_RI)}; threads {TH}", flush=True)
za, da, ha, _ = run(TOOL_RI, "A_logged", 32, "--scene", "logged")
zb, db, hb, _ = run(TOOL_RI, "B_reindexed", 32, "--scene", "reindexed")
assert "scene_reindex" not in ha and "scene" not in za
sr = hb["scene_reindex"]
assert sr["past_rows_bit_identical"] is True and sr["future_frame_rows"] == 0 and sr["rows_reindexed"] > 0, sr
print("SMOKERI (b) receipt B (--scene reindexed, 32 windows, 10-step replans):")
print(json.dumps({k: v for k, v in sr.items() if k not in ("rule", "limits", "role_note")}, indent=1))
assert str(zb["scene"]) == "reindexed" and any("REINDEXED" in c for c in hb["open_channels"]) and hb["strict"]["camera"] is False
# (c) plan differences
t0, t1 = zb["tick"] == 0, zb["tick"] >= 1
dev = {nm: np.abs(za[nm].astype(np.float64) - zb[nm].astype(np.float64)).reshape(len(za[nm]), -1).max(1) for nm in ("plan", "plan_raw")}
for nm in ("plan", "plan_raw"):
    per_tick = {int(t): float(dev[nm][zb["tick"] == t].max()) for t in sorted(set(zb["tick"].tolist()))}
    print(f"SMOKERI (c) {nm}: max abs difference A vs B at tick 0 {dev[nm][t0].max():.3e} m (must be 0.0: tick 0 has only past rows); ticks >= 1 {dev[nm][t1].max():.3f} m; "
          f"median over ticks >= 1 {np.median(dev[nm][t1]):.3f} m; share of ticks >= 1 rows with a difference over 1e-3 m {float((dev[nm][t1] > 1e-3).mean()):.3f}; per tick {per_tick}")
assert dev["plan"][t0].max() == 0.0 and dev["plan_raw"][t0].max() == 0.0 and dev["plan"][t1].max() > 0.0
big = int(np.argmax(dev["plan"]))
print(f"SMOKERI (c) largest plan difference: {dev['plan'].max():.3f} m at window {int(zb['win'][big])} tick {int(zb['tick'][big])} (clip index {int(zb['ci'][big])}, t0 {int(zb['t0'][big])})")
move = float(np.abs(db["cl"] - da["cl"]).max())
print(f"SMOKERI (c) executed path: max abs cl difference A vs B {move:.3f} m; ol (tick-0 plan) difference {float(np.abs(db['ol'] - da['ol']).max()):.3e} m")
# (d)
assert all(np.array_equal(db[q], da[q]) for q in ("clip_index", "ws", "eid", "g", "v0")) and np.array_equal(db["fl"], da["fl"])
print("SMOKERI (d) windows, ground truth, v0 and the P10 floor of A and B are equal")
# (e) KRI4 real rig, 6 windows
for sc in ("logged", "frozen", "hold_now"):
    x = run(TOOL_SRC, f"src_{sc}", 6, "--scene", sc)
    y = run(TOOL_RI, f"ri_{sc}", 6, "--scene", sc)
    assert same(x[0], x[1], x[2], y[0], y[1], y[2]), f"KRI4 real rig: --scene {sc} differs between the source tool and the RI tool"
    print(f"SMOKERI (e) KRI4 real rig, 6 windows: --scene {sc}: dumps, ticks and header byte-identical, source tool vs RI tool")
print("SMOKERI OK")
