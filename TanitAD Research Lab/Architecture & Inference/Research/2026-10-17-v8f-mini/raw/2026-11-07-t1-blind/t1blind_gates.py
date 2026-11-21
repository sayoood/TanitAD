"""t1blind_gates: the three gates of PREREG_T1_BLIND.md section 3, each a pure function of existing artifacts (receipts, rollout dumps, arrays), so a gate can be run on a mutant's artifacts and must FAIL there.
  G-ID    the same wrapper path with the replacement switched off (mode `off`, tag ID_s{s}) reproduces the banked RIC0_s{s} BIT FOR BIT: every tick-dump array, every per-episode dump array, every extracted array (a_i, b_i, g, v0, ...),
          and the paired delta ID - RIC0 on the 19 rows reads EXACTLY 0 (lo = hi = 0).
  G-MEAN  the tokens the policy RECEIVES (counted at the forward_infer entry, int16-bitwise) equal the pinned mean on every received frame, on >= 1,000 frames.
  G-EGO   the ego inputs the policy receives equal RIC0's: (a) the wrapper re-assigned no key other than `feats` in any item (identity count 0), (b) the per-row digest of every received non-feats input at TICK 0 of every window equals
          the `off` run's, which G-ID makes RIC0's. Ticks 1 and 2 are NOT compared: the ego history there is the executed motion, which legitimately differs when the plan differs (that is the effect under test).
usage: t1blind_gates.py id --seed S [--out-root R] [--work-dir W] [--raw RAW]      -> RAW/GATE_ID_s{S}.json  (rc 5 = FAIL)
       t1blind_gates.py mean --receipt FILE --out FILE
       t1blind_gates.py ego --off-dir D --blind-dir D --tag-off ID_s0 --tag-blind BLIND1_s0 --out FILE
       t1blind_gates.py all [--raw RAW] [--out-root R]                              -> RAW/GATES.json  (rc 5 = any gate FAILS)"""
import os
import sys
import glob
import json
import argparse
import importlib.util

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
ART = "E:/Projects/TanitAD-artifacts"
LF_PATH = f"{ART}/dagland/motion_c/lf_read/code/lf_read.py"
BANK_OUT = "F:/TanitAD-scratch/motion_c/ri/out"                     # the banked RIC0_s{0,1} rollouts (tool 0417c838...)
SCRATCH = "F:/TanitAD-scratch/motion_c/t1_blind"
OUT_ROOT = f"{SCRATCH}/out"
WORK_DIR = f"{SCRATCH}/work"
RAW = f"{HERE}/raw"
MIN_FRAMES = 1000
SEEDS = (0, 1)
TICK_META_SKIP = ("ckpt", "tag")                                    # path / name strings of the tick file; every array and the clip names are compared


def sha256_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _eq(a, b):
    """Exact equality of two arrays: same shape, same dtype, same bytes for numeric arrays (NaN equals NaN at equal bit patterns is not required: value equality with NaN == NaN)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def compare_npz(pa, pb, skip=()):
    """-> (all_equal, {key: bool}, only_in_a, only_in_b) over the keys of two .npz files."""
    za, zb = np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True)
    ka, kb = [k for k in za.files if k not in skip], [k for k in zb.files if k not in skip]
    per = {k: _eq(za[k], zb[k]) for k in ka if k in kb}
    oa, ob = sorted(set(ka) - set(kb)), sorted(set(kb) - set(ka))
    return bool(per and all(per.values()) and not oa and not ob), per, oa, ob


# ------------------------------------------------------------------------------------------------ G-MEAN
def gate_mean(receipt, min_frames=MIN_FRAMES):
    """``receipt`` = a receipt dict. PASS iff the policy received >= ``min_frames`` frames and EVERY received frame equals the pinned mean bitwise (and the item builder replaced every item it built)."""
    n, ne = int(receipt["frames_received"]), int(receipt["frames_equal_mean_bitwise"])
    ok = bool(n >= min_frames and ne == n and int(receipt["frames_not_equal_mean"]) == 0 and int(receipt["items_replaced"]) == int(receipt["items_built"]) > 0)
    return {"gate": "G-MEAN", "PASS": ok, "frames_received": n, "frames_equal_mean_bitwise": ne, "frames_not_equal_mean": int(receipt["frames_not_equal_mean"]), "min_frames": min_frames,
            "items_built": int(receipt["items_built"]), "items_replaced": int(receipt["items_replaced"]), "mode": receipt.get("mode"), "mean_tensor_sha256": receipt.get("mean_tensor_sha256")}


# ------------------------------------------------------------------------------------------------ G-EGO
def gate_ego(receipt_blind, dig_off, dig_blind):
    """``dig_*`` = the digest arrays of the two runs ({tick, win, digest}). PASS iff the blind wrapper changed no non-feats key of any item AND the tick-0 per-row digests are identical, window for window, to the off run's."""
    t0o, t0b = np.asarray(dig_off["tick"]) == 0, np.asarray(dig_blind["tick"]) == 0
    wo, wb = np.asarray(dig_off["win"])[t0o], np.asarray(dig_blind["win"])[t0b]
    do, db = np.asarray(dig_off["digest"])[t0o], np.asarray(dig_blind["digest"])[t0b]
    same_windows = bool(wo.shape == wb.shape and np.array_equal(wo, wb))
    n_eq = int((do == db).sum()) if (same_windows and do.shape == db.shape) else 0
    n = int(wo.size)
    ok = bool(same_windows and n > 0 and n_eq == n and int(receipt_blind["items_other_keys_changed"]) == 0 and int(receipt_blind["items_keyset_changed"]) == 0)
    return {"gate": "G-EGO", "PASS": ok, "tick0_windows_compared": n, "tick0_windows_equal": n_eq, "same_window_index": same_windows, "items_other_keys_changed": int(receipt_blind["items_other_keys_changed"]),
            "items_keyset_changed": int(receipt_blind["items_keyset_changed"]), "later_ticks": "not compared: the ego history at ticks 1 and 2 is the executed motion, which differs when the plan differs (the effect under test)"}


# ------------------------------------------------------------------------------------------------ G-ID (raw rollout files)
def gate_id_raw(out_root, tag, bank_tag, bank_root=BANK_OUT):
    """Bitwise: the tick dump and all per-episode dumps of ``tag`` against the banked ``bank_tag``."""
    pa, pb = f"{out_root}/{tag}/ticks/ticks_{tag}.npz", f"{bank_root}/{bank_tag}/ticks/ticks_{bank_tag}.npz"
    ok_t, per_t, oa, ob = compare_npz(pa, pb, skip=TICK_META_SKIP)
    fa = sorted(glob.glob(f"{out_root}/{tag}/dump_{tag}/ep*.npz"))
    fb = sorted(glob.glob(f"{bank_root}/{bank_tag}/dump_{bank_tag}/ep*.npz"))
    same_names = [os.path.basename(x) for x in fa] == [os.path.basename(x) for x in fb]
    bad = []
    if same_names:
        for x, y in zip(fa, fb):
            ok_e, per_e, _, _ = compare_npz(x, y)
            if not ok_e:
                bad.append(os.path.basename(x))
    ok = bool(ok_t and same_names and len(fa) > 0 and not bad)
    return {"ticks_equal": ok_t, "ticks_keys_unequal": [k for k, v in per_t.items() if not v], "ticks_only_in_run": oa, "ticks_only_in_bank": ob, "episode_files": len(fa), "episode_files_bank": len(fb),
            "episode_files_same_names": same_names, "episode_files_unequal": bad, "PASS": ok}


def _load_lf():
    spec = importlib.util.spec_from_file_location("lf_read_for_t1blind", LF_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.init()
    return m


def gate_id(seed, out_root=OUT_ROOT, work_dir=WORK_DIR, raw=RAW, lf=None, tag=None, bank_root=BANK_OUT):
    """The full G-ID of one seed: raw rollout files bitwise, then the extracted arrays bitwise against the sha256-pinned RIC0_s{seed}, then the paired delta ID - RIC0 on the 19 rows (must be exactly 0)."""
    tag, bank_tag = tag or f"ID_s{seed}", f"RIC0_s{seed}"
    res = {"gate": "G-ID", "seed": seed, "tag": tag, "against": bank_tag}
    res["raw"] = gate_id_raw(out_root, tag, bank_tag, bank_root)
    lf = lf or _load_lf()
    rr, GD, M1 = lf.rr, lf.GD, lf.M1
    rr.PK = HERE
    rr.WORK = f"{work_dir}/rig"
    arr_dir = f"{raw}/arrays"
    z_ric0, ctl = lf.reused_arrays(seed)
    pin = ctl[bank_tag]
    res["ric0_array_sha256_pinned"] = {"path": pin["path"], "sha256": pin["sha256"], "pinned": pin["pinned"], "PASS": pin["PASS"]}
    dst = rr.extract(tag, seed, out_root, arr_dir)
    res["id_array"] = {"path": dst, "sha256": sha256_file(dst)}
    ok_a, per_a, oa, ob = compare_npz(dst, pin["path"])
    res["arrays_equal"] = ok_a
    res["arrays_keys_unequal"] = [k for k, v in per_a.items() if not v]
    res["arrays_only_in_run"], res["arrays_only_in_bank"] = oa, ob
    res["arrays_sha256_equal"] = res["id_array"]["sha256"] == pin["sha256"]
    zi = rr.read_arrays(dst)
    z0 = z_ric0["RIC0"]
    eid = z0["eid"]
    cls, _ = GD.classify(z0["v0"], z0["g"])
    rows, _ = rr.rows_of(rr.Aof(zi), rr.Aof(z0), eid, cls, 200, M1)                  # B 200: an exact zero does not depend on B
    res["delta_ID_minus_RIC0"] = {lab: {k: float(r[k]) for k in ("delta", "lo", "hi")} for lab, r in rows.items()}
    res["delta_exactly_zero_all_19_rows"] = bool(all(r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0 for r in rows.values()) and len(rows) == len(GD.LABELS) == 19)
    res["PASS"] = bool(res["raw"]["PASS"] and pin["PASS"] and ok_a and res["delta_exactly_zero_all_19_rows"])
    return res


def _dump(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8", newline="\n"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("id"); p.add_argument("--seed", type=int, required=True)
    for q in (p,):
        q.add_argument("--out-root", default=OUT_ROOT); q.add_argument("--work-dir", default=WORK_DIR); q.add_argument("--raw", default=RAW)
    p = sub.add_parser("mean"); p.add_argument("--receipt", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("ego")
    p.add_argument("--off-dir", required=True); p.add_argument("--blind-dir", required=True); p.add_argument("--tag-off", required=True); p.add_argument("--tag-blind", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("all"); p.add_argument("--raw", default=RAW); p.add_argument("--out-root", default=OUT_ROOT)
    a = ap.parse_args(argv)
    if a.cmd == "id":
        r = gate_id(a.seed, a.out_root, a.work_dir, a.raw)
        _dump(r, f"{a.raw}/GATE_ID_s{a.seed}.json"); print("G-ID", a.seed, "PASS" if r["PASS"] else "FAIL"); return 0 if r["PASS"] else 5
    if a.cmd == "mean":
        r = gate_mean(json.load(open(a.receipt))); _dump(r, a.out); print("G-MEAN", "PASS" if r["PASS"] else "FAIL", r["frames_equal_mean_bitwise"], "/", r["frames_received"]); return 0 if r["PASS"] else 5
    if a.cmd == "ego":
        rb = json.load(open(f"{a.blind_dir}/receipt_{a.tag_blind}.json"))
        r = gate_ego(rb, np.load(f"{a.off_dir}/ego_digests_{a.tag_off}.npz"), np.load(f"{a.blind_dir}/ego_digests_{a.tag_blind}.npz"))
        _dump(r, a.out); print("G-EGO", "PASS" if r["PASS"] else "FAIL", r["tick0_windows_equal"], "/", r["tick0_windows_compared"]); return 0 if r["PASS"] else 5
    if a.cmd == "all":
        allr = {"card": "PREREG_T1_BLIND.md section 3", "seeds": {}}
        ok = True
        for s in SEEDS:
            ref = f"{a.out_root}/receipts"
            g_id = json.load(open(f"{a.raw}/GATE_ID_s{s}.json"))
            rb = json.load(open(f"{ref}/receipt_BLIND1_s{s}.json"))
            g_mean = gate_mean(rb)
            g_ego = gate_ego(rb, np.load(f"{ref}/ego_digests_ID_s{s}.npz"), np.load(f"{ref}/ego_digests_BLIND1_s{s}.npz"))
            allr["seeds"][f"s{s}"] = {"G-ID": g_id["PASS"], "G-MEAN": g_mean, "G-EGO": g_ego}
            ok = ok and g_id["PASS"] and g_mean["PASS"] and g_ego["PASS"]
        allr["PASS"] = ok
        _dump(allr, f"{a.raw}/GATES.json"); print("GATES", "PASS" if ok else "FAIL"); return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
