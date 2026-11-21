"""info_extras: banks the small inputs town_result.py needs so the result can be re-derived from town/raw/ alone, and computes two INFORMATIONAL blocks. Decides nothing.
  1. copies the four run receipts (receipt_*.json) and the six done markers from the rollout scratch into raw/receipts/ (sha256 recorded), because the controls G-REG read them;
  2. raw/INFO_DRIFT_vs_banked.json : what differs between the chain's C0 runs (4 threads) and the banked RIC0 runs (12 threads, same ckpt, same tool sha), column by column;
  3. raw/INFO_DV_SIGNS.json        : counts of the slope rows by sign of the deficit (descriptive counts only, nothing is refitted).
CPU only, read-only on the scratch. usage: python -I info_extras.py"""
import os
import sys
import glob
import json
import shutil
import hashlib

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
RAW = f"{HERE}/raw"
OUT = "F:/TanitAD-scratch/motion_c/town/out"
BANK = "F:/TanitAD-scratch/motion_c/ri/out"
TAGS = ("C0_s0", "C0_s1", "REG_s0", "REG_s1")
DV_MIN = 0.5


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def copy_receipts():
    os.makedirs(f"{RAW}/receipts", exist_ok=True)
    rec = {}
    for t in TAGS:
        for src, dst in ((f"{OUT}/receipts/receipt_{t}.json", f"{RAW}/receipts/receipt_{t}.json"), (f"{OUT}/done_{t}.txt", f"{RAW}/receipts/done_{t}.txt")):
            shutil.copyfile(src, dst)
            assert sha(src) == sha(dst)
            rec[os.path.basename(dst)] = sha(dst)
    for t in ("BARE_s0", "BARE_s1"):
        shutil.copyfile(f"{OUT}/done_{t}.txt", f"{RAW}/receipts/done_{t}.txt")
        rec[f"done_{t}.txt"] = sha(f"{RAW}/receipts/done_{t}.txt")
    for t in ("RIC0_s0", "RIC0_s1"):
        shutil.copyfile(f"{BANK}/done_{t}.txt", f"{RAW}/receipts/banked_done_{t}.txt")
        rec[f"banked_done_{t}.txt"] = sha(f"{RAW}/receipts/banked_done_{t}.txt")
    return rec


def drift(s):
    a = np.load(f"{OUT}/C0_s{s}/ticks/ticks_C0_s{s}.npz", allow_pickle=True)
    b = np.load(f"{BANK}/RIC0_s{s}/ticks/ticks_RIC0_s{s}.npz", allow_pickle=True)
    cols = {}
    for k in a.files:
        if k in ("ckpt", "tag"):
            continue
        x, y = a[k], b[k]
        if x.dtype.kind in "fc":
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            cols[k] = {"equal": bool(np.array_equal(x, y, equal_nan=True)), "max_abs": float(np.nanmax(d)), "mean_abs": float(np.nanmean(d))}
        else:
            cols[k] = {"equal": bool(np.array_equal(x, y))}
    la, lb = a["t_lon_logits"].argmax(-1), b["t_lon_logits"].argmax(-1)
    ta, tb = a["t_lat_logits"].argmax(-1), b["t_lat_logits"].argmax(-1)
    mx = []
    for f in sorted(glob.glob(f"{OUT}/C0_s{s}/dump_C0_s{s}/ep*.npz")):
        z = np.load(f)
        w = np.load(f.replace(f"{OUT}/C0_s{s}/dump_C0_s{s}", f"{BANK}/RIC0_s{s}/dump_RIC0_s{s}"))
        mx.append(max(float(np.abs(z[k].astype(np.float64) - w[k].astype(np.float64)).max()) for k in ("cl", "fl", "ol")))
    mx = np.array(mx)
    return {"columns": cols, "columns_float_unequal": sorted(k for k, v in cols.items() if "max_abs" in v and not v["equal"]), "columns_integer_or_bool_unequal": sorted(k for k, v in cols.items() if "max_abs" not in v and not v["equal"]),
            "t_lon_argmax_agreement": float((la == lb).mean()), "t_lat_argmax_agreement": float((ta == tb).mean()), "rows": int(la.size),
            "episode_track_max_abs_delta": {"n_episodes": int(len(mx)), "n_episodes_nonzero": int((mx > 0).sum()), "median": float(np.median(mx)), "p90": float(np.percentile(mx, 90)), "max": float(mx.max())}}


def dv_signs(tag):
    z = np.load(f"{OUT}/{tag}/ticks/twin_{tag}.npz", allow_pickle=True)
    dv = (z["v_log"] - z["v_own"]).astype(np.float64)
    tick = z["tick"]
    sel = np.abs(dv) >= DV_MIN
    return {"n_rows": int(dv.size), "n_slope_rows": int(sel.sum()), "n_own_slower_ge_0.5": int((dv >= DV_MIN).sum()), "n_own_faster_ge_0.5": int((dv <= -DV_MIN).sum()),
            "share_own_slower_among_slope_rows": float((dv[sel] > 0).mean()), "by_tick_n_slope_rows": [int(sel[tick == k].sum()) for k in range(int(tick.max()) + 1)],
            "dv_quantiles_5_25_50_75_95": [float(x) for x in np.percentile(dv, [5, 25, 50, 75, 95])], "twin_npz_sha256": sha(f"{OUT}/{tag}/ticks/twin_{tag}.npz")}


def main():
    rec = copy_receipts()
    json.dump({"note": "copies of the rollout receipts and done markers (source F:/TanitAD-scratch/motion_c/{town,ri}/out); sha256 of each copy", "sha256": rec}, open(f"{RAW}/receipts/SHA256.json", "w", encoding="utf-8", newline="\n"), indent=1)
    d = {"informational": True, "decides": "nothing", "what": "C0 (4 threads, this chain) against banked RIC0 (12 threads, same ckpt sha and tool sha), tick files and episode dumps",
         "s0": drift(0), "s1": drift(1)}
    json.dump(d, open(f"{RAW}/INFO_DRIFT_vs_banked.json", "w", encoding="utf-8", newline="\n"), indent=1)
    v = {"informational": True, "decides": "nothing", "dv_min_m_s": DV_MIN, "definition": "dv = v_logged - v_own at the replan time; positive = the car is slower than the logged state. Counts only, nothing refitted.",
         "tags": {t: dv_signs(t) for t in TAGS}}
    json.dump(v, open(f"{RAW}/INFO_DV_SIGNS.json", "w", encoding="utf-8", newline="\n"), indent=1)
    print("wrote receipts/, INFO_DRIFT_vs_banked.json, INFO_DV_SIGNS.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
