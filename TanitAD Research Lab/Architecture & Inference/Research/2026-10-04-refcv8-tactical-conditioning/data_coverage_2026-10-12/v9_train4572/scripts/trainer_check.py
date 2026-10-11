"""TRAINER-SIDE READER CHECK for the new v9 train release.

Loads the release through the trainer's own entry (`tanitad.train.refcv8_train.load_v9_join`: md5 check + the
`v9_integrity_census` contract refusal), then joins EVERY dataset window NOW of all 4,572 clips through the join's
own `row()` (= `v9_labels.row_for_now`, |release now_s - trainer now_s| <= 1e-6 s) using the TRAINER's clock code
(`clip_clock.read_clip_clock_sidecar` + `clip_clock.pose_dt` fallback, `refc_v3_train.py::_clock_for/_now_s` restated),
then pulls `item()` (inputs + targets + speed + constraints) for a sample of windows of 5 ADD + 5 old clips.

usage: trainer_check.py <npz> <expect_md5> <train_manifest.pt> <clock_sidecar.jsonl> <out.json>
"""
import sys, json, time
import numpy as np, torch

npz, md5, man_path, side_path, outj = sys.argv[1:6]
from tanitad.train import refcv8_train as R8
from tanitad.data import clip_clock as CC

rep = {"npz": npz, "expect_md5": md5}
t0 = time.time()
try:
    join = R8.load_v9_join(npz, expect_md5=md5)
    rep["load_v9_join"] = "OK"
except SystemExit as e:
    rep["load_v9_join"] = "REFUSED: " + str(e)[:600]
    json.dump(rep, open(outj, "w"), indent=1)
    print(rep["load_v9_join"])
    raise
cen = join.manifest["census"]
rep["census_violations"] = cen["violations"]
rep["census_n_rows"] = cen["n_rows"]
rep["census_nonzero_violations"] = {k: v for k, v in cen["violations"].items() if v}
print("load_v9_join OK; census n_rows", cen["n_rows"], "violations non-zero:", rep["census_nonzero_violations"], flush=True)

mn = torch.load(man_path, map_location="cpu", weights_only=False)
cids = [str(c) for c in mn["clip_id"]]
from tanitad.data.v2_dataset import stable_episode_id as _sei
sids = [int(s) for s in mn["episode_uid"]]          # ep.episode_id in the trainer (v2_dataset.py:429)
assert sids == [_sei(c) for c in cids], "manifest episode_uid != v2_dataset.stable_episode_id(clip_id)"
poses = mn["poses"]
T_out = [int(x) for x in mn["T_out"]]
side = CC.read_clip_clock_sidecar(side_path)
W, MAXH, RAWOFF = 8, 20, 2          # window, max_horizon, n_stack-1 (refc_v3_train.py:_raw_offset: 9 ch // 3 - 1)
n_win = n_ok = 0
n_side = n_pose = n_nom = 0
errs = []
per_clip_windows = {}
for i, sid in enumerate(sids):
    if sid in side.table:
        g0, dt = side.table[sid]
        n_side += 1
    else:
        dt_ = CC.pose_dt(poses[i])
        if dt_ is None:
            g0, dt = 0.0, float(CC.NOMINAL_DT_S)
            n_nom += 1
        else:
            g0, dt = 0.0, float(dt_)
            n_pose += 1
    tmax = T_out[i] - (W + MAXH)
    per_clip_windows[sid] = tmax
    for t in range(tmax):
        n_win += 1
        k = t + W - 1 + RAWOFF
        now = float(g0) + k * float(dt)
        try:
            join.row(sid, k, now)
            n_ok += 1
        except Exception as e:                       # noqa: BLE001
            if len(errs) < 10:
                errs.append(f"sid {sid} t {t}: {type(e).__name__}: {str(e)[:200]}")
rep.update({"n_windows": n_win, "n_joined_ok": n_ok, "n_join_errors": n_win - n_ok, "first_errors": errs,
            "clock_sources": {"sidecar": n_side, "pose_dt": n_pose, "nominal": n_nom},
            "wall_s_all_windows": round(time.time() - t0, 1)})
print(f"all-window join: {n_ok}/{n_win} windows OK, errors {n_win - n_ok}; clock sidecar/pose_dt/nominal {n_side}/{n_pose}/{n_nom}", flush=True)

# sample item() pull: 5 ADD + 5 old clips (+ the parked ADD clip)
add_ids = json.load(open("add_ids.json"))["add"]
add_set = set(add_ids)
rng = np.random.RandomState(0)
add_idx = [i for i, c in enumerate(cids) if c in add_set]
old_idx = [i for i, c in enumerate(cids) if c not in add_set]
parked = [i for i, c in enumerate(cids) if c == "12c90362-a57f-4c3d-bf0d-ebe47a687186"]
sample = list(rng.choice(add_idx, 5, replace=False)) + parked + list(rng.choice(old_idx, 5, replace=False))
item_res = []
n_items = 0
for i in sample:
    sid = sids[i]
    g0, dt = side.table.get(sid, (0.0, None))
    if dt is None:
        d_ = CC.pose_dt(poses[i])
        dt = float(d_) if d_ is not None else float(CC.NOMINAL_DT_S)
    tmax = per_clip_windows[sid]
    ts = sorted(set(np.linspace(0, tmax - 1, 12).astype(int).tolist()))
    ok = 0
    for t in ts:
        k = t + W - 1 + RAWOFF
        now = float(g0) + k * float(dt)
        for sm in (None, "n2", "n3"):
            it = join.item(sid, k, now, speed_mode=sm, cons=True)
            n_items += 1
        ok += 1
    item_res.append({"clip_id": cids[i], "group": "ADD" if cids[i] in add_set else "old", "sid": sid,
                     "windows_pulled": ok, "keys": sorted(it.keys())[:3] + ["..."]})
rep["item_sample"] = item_res
rep["n_item_pulls"] = n_items
rep["verdict"] = "PASS" if (rep["n_join_errors"] == 0 and not rep["census_nonzero_violations"]) else "FAIL"
json.dump(rep, open(outj, "w"), indent=1)
print("item() pulls OK:", n_items, "over", len(sample), "clips")
print("TRAINER-SIDE CHECK:", rep["verdict"])
