"""census_rolladd.py -- ROLLADD real-bank census (CPU, one python process, no GPU, nothing written to the pod). Inputs pulled read-only from the pod into tmp/real/ (sha256-verified against the pod's stream; deleted afterwards):
  join_union.npz members clip, n; the four bank/s0 shards' members ci, r, reason (whole); shard 0's first 20,000 records of p10f, pose, arc, kin3, j and its header.
A. ELIGIBILITY on the real bank (all four shards) and the train window list rebuilt from the join: compared with BANK_CHECK.json's coverage (784,954 of 985,048 = 0.79687, MEASURED on the pod).
B. APPENDED ROWS PER STEP: the exact selection arithmetic of DaggerSwap.apply (private generator seeded by (dagger-seed 0, step), u[0] < p & eligible) over 3,000 steps of B = 64 windows drawn uniformly from the window list
   (the trainer's ChunkedSampler is a chunked shuffle; uniform is the approximation). N = selected = appended under add; the step's row count is B + N.
C. CPU WALL PER STEP of DaggerSwap.apply, swap vs add, on a batch with REAL shapes (B 64, hmax 16, feats f16 [B, 16, 40, 1024]: n_tok / d are ESTIMATED from the pool4x10 grid and DINOv3-L) and a bank built from shard 0's first
   20,000 REAL records (gather + swap_rows are the real code); N = 5 (the p50 of B) and N = 9 (the p95)."""
import json
import sys
import time

sys.path.insert(0, "F:/TanitAD-scratch/rolladd/tree/stack")
sys.path.insert(0, "F:/TanitAD-scratch/rolladd/tree/taniteval")
import numpy as np
import torch

import tanitad
from tanitad.data import v8f_dagger as G

assert tanitad.__file__.replace("\\", "/").startswith("F:/TanitAD-scratch/rolladd/tree/"), tanitad.__file__
torch.set_num_threads(1)
R = "F:/TanitAD-scratch/rolladd/tmp/real/"
OUT = "F:/TanitAD-scratch/rolladd/REAL_BANK_CENSUS.json"
t_all = time.time()
clip = np.load(R + "join_clip.npy", allow_pickle=False)
n_of = np.load(R + "join_n.npy", allow_pickle=False).astype(np.int64)
ci = np.concatenate([np.load(R + f"s0_k{k}_ci.npy") for k in range(4)]).astype(np.int64)
rr = np.concatenate([np.load(R + f"s0_k{k}_r.npy") for k in range(4)]).astype(np.int64)
rs = np.concatenate([np.load(R + f"s0_k{k}_reason.npy") for k in range(4)]).astype(np.int64)
ok = rs == G.OK
key_ok = np.unique(ci[ok] * (1 << 24) + rr[ok])
res = {"n_records_all_shards": int(len(ci)), "reason_counts": {G.REASONS[i]: int((rs == i).sum()) for i in range(5)}, "n_clips_join": int(len(clip))}


def windows(first, clips=None):
    out = []
    for c in (range(len(n_of)) if clips is None else clips):
        T = int(n_of[c])
        for t in range(first, T - 30):
            out.append((c, t))
    return np.array(out, np.int64)


want = (784954, 985048)
found = None
for first in (3, 9):
    for label, clips in (("all join clips", None), ("clips with a bank record", np.unique(ci))):
        w = windows(first, clips)
        el = np.isin(w[:, 0] * (1 << 24) + w[:, 1], key_ok)
        res.setdefault("window_list_trials", []).append({"first_t": first, "clips": label, "n_windows": int(len(w)), "n_eligible": int(el.sum()), "matches_BANK_CHECK": (int(el.sum()), int(len(w))) == want})
        if (int(el.sum()), int(len(w))) == want and found is None:
            found = (w, el, first, label)
if found is None:                                                    # no spelling reproduces the pod's coverage: report it, and use the closest as an APPROXIMATION
    best = max(res["window_list_trials"], key=lambda d: -abs(d["n_windows"] - want[1]))
    first, label = best["first_t"], best["clips"]
    w = windows(first, None if label == "all join clips" else np.unique(ci))
    el = np.isin(w[:, 0] * (1 << 24) + w[:, 1], key_ok)
    res["window_list"] = {"reproduces_BANK_CHECK_coverage": False, "used": best}
else:
    w, el, first, label = found
    res["window_list"] = {"reproduces_BANK_CHECK_coverage": True, "first_t": first, "clips": label}
res["coverage_fraction"] = float(el.mean())
# ---- B
B, STEPS, P, SEED = 64, 3000, 0.1, 0
rng = np.random.default_rng(0)
Ns = np.empty(STEPS, np.int64)
for s in range(STEPS):
    idx = rng.integers(0, len(w), B)
    g = torch.Generator().manual_seed((SEED * 1_000_003 + s) % (1 << 62))
    u = torch.rand(2, B, generator=g, dtype=torch.float64)
    Ns[s] = int(((u[0] < P).numpy() & el[idx]).sum())
pc = lambda a, q: float(np.percentile(a, q))
res["appended_per_step"] = {"B": B, "p": P, "steps": STEPS, "N_mean": float(Ns.mean()), "N_p50": pc(Ns, 50), "N_p95": pc(Ns, 95), "N_max": int(Ns.max()), "N_min": int(Ns.min()), "steps_with_N0": int((Ns == 0).sum()),
                         "rows_out_p50": pc(Ns + B, 50), "rows_out_p95": pc(Ns + B, 95), "rows_out_mean": float((Ns + B).mean()), "extra_rows_share_mean": float(Ns.mean() / B),
                         "cross_check": "RL train logs: 15,411 of 192,000 rows (8.03 %) swapped at s0 = 5.14 per step of 64; this census mean is the line above"}
# ---- C
H = np.load(R + "s0_k0_header_json_head.npy", allow_pickle=False)
header = json.loads(bytes(H.astype(np.uint8)).decode("utf-8")) if H.dtype != object else json.loads(str(H.item()))
NREC = 20000


def head(name, shape, dt):
    a = np.fromfile(R + f"s0_k0_{name}_head.npy", dtype=np.uint8)[128:]
    return a[: NREC * int(np.prod(shape)) * np.dtype(dt).itemsize].view(dt).reshape(NREC, *shape)


F_ = 16
arrays = {"ci": ci[:NREC].astype(np.int32), "t0": np.zeros(NREC, np.int32), "tick": np.ones(NREC, np.int8), "j": head("j", (), np.int32).astype(np.int32), "p10f": head("p10f", (F_, 8), np.float32),
          "pose": head("pose", (F_, 3), np.float64), "arc": head("arc", (F_,), np.float64), "kin3": head("kin3", (3,), np.float32), "r": rr[:NREC].astype(np.int32), "e_lat": np.zeros(NREC, np.float32),
          "dpsi": np.zeros(NREC, np.float32), "reason": rs[:NREC].astype(np.int8)}
bank = G.DaggerBank(arrays, header, clip_names=clip)
okk = np.flatnonzero(arrays["reason"] == G.OK)
res["timing_bank"] = {"records": NREC, "ok": int(len(okk)), "note": "the first 20,000 REAL records of shard 0 (gather + swap_rows are the real code); the logged batch tensors are fabricated with the real shapes"}
dev = torch.device("cpu")
g0 = torch.Generator().manual_seed(5)


def make_batch(rows_ok):
    ci_b = torch.from_numpy(np.concatenate([arrays["ci"][rows_ok], np.zeros(B - len(rows_ok), np.int64) + 10 ** 6]).astype(np.int64))
    t_b = torch.from_numpy(np.concatenate([arrays["r"][rows_ok], np.zeros(B - len(rows_ok), np.int64) + 10 ** 6]).astype(np.int64))
    z = lambda *s, dt=torch.float32: torch.randn(*s, generator=g0).to(dt)
    b = {"feats": z(B, F_, 40, 1024, dt=torch.float16), "frame_ok": torch.ones(B, F_, dtype=torch.bool), "limit_ok": torch.ones(B, dtype=torch.bool), "gt_ok": torch.ones(B, dtype=torch.bool),
         "p10f": z(B, F_, 8), "act": z(B, F_, 2), "nav": torch.rand(B, F_, 5, generator=g0), "pose": z(B, F_, 3), "limit": z(B, 8), "gt": z(B, 30, 2), "base": z(B, 3), "kin3": z(B, 3), "s_h": z(B, 30),
         "lead_gap_h": z(B, 30), "pose_now": z(B, 3), "pose_t": z(B, 3), "pose_s": z(B, 3), "p10_t": z(B, 8), "a_t": torch.ones(B, dtype=torch.long), "a_s": 3 * torch.ones(B, dtype=torch.long), "t": t_b, "ci": ci_b}
    lab = lambda: {"has": torch.ones(B, dtype=torch.bool), "nav": z(B, 5), "goal_y": z(B, 4), "goal_w": z(B, 4), "lat_cls": torch.zeros(B, dtype=torch.long), "lat_al": torch.ones(B, 5, dtype=torch.bool),
                   "lon_cls": torch.zeros(B, dtype=torch.long), "lon_al": torch.ones(B, 5, dtype=torch.bool), "cons": z(B, 6), "rc": z(B, 3, 3), "rc_ok": torch.ones(B, 3, dtype=torch.bool), "lead_valid": torch.ones(B, dtype=torch.bool)}
    b["lab"], b["lab_t"], b["lab_s"] = lab(), lab(), lab()
    return b


timing = {}
for Nsel in (5, 9):
    rows_ok = okk[:Nsel]
    b = make_batch(rows_ok)
    b["pose_now"] = torch.from_numpy(arrays["pose"][np.concatenate([rows_ok, np.zeros(B - Nsel, np.int64)]), -1].astype(np.float32))
    row = {}
    for mode in ("swap", "add"):
        sw = G.DaggerSwap(bank, p=1.0, seed=0, kin_mode="yaw", mode=mode)
        try:
            for _ in range(3):
                sw.apply(b, 0)
        except Exception as e:                                               # a fabricated-batch problem is reported, never papered over
            row[mode] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            continue
        ts = []
        for k in range(30):
            t1 = time.perf_counter()
            out = sw.apply(b, k)
            ts.append(time.perf_counter() - t1)
        row[mode] = {"ms_median": round(1000 * float(np.median(ts)), 2), "ms_p95": round(1000 * float(np.percentile(ts, 95)), 2), "n_selected": sw.last["n_selected"], "n_appended": sw.last["n_appended"], "rows_out": int(out["p10f"].shape[0])}
    if "ms_median" in row.get("swap", {}) and "ms_median" in row.get("add", {}):
        row["add_minus_swap_ms_median"] = round(row["add"]["ms_median"] - row["swap"]["ms_median"], 2)
    timing[f"N={Nsel}"] = row
res["cpu_wall_per_step_apply"] = timing
res["total_wall_s"] = round(time.time() - t_all, 1)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
