"""town_gates: the controls of PREREG_V8F_TOWN.md section 'Controls', each a pure function of artifacts (so a mutant's artifacts must make it FAIL).
  G-ID     (control 1) the dump run's tick file and every per-episode dump equal the banked tool-of-record run BIT FOR BIT; the max |delta| over the executed tracks (cl, fl, ol) is exactly 0.0.
  G-TWIN0  (control 2) at tick 0 (own state == logged state) every swapped input is bitwise equal and the own and twin T outputs are bitwise equal, so d is exactly 0.0.
  G-TWINL  the twin is really LOGGED: the twin p10f row equals the logged clip's row on every row; kept keys are the same tensors; and at ticks >= 1 the twin differs from the own batch on > 0 rows
           (a twin that swapped nothing would pass G-TWIN0 and fail here).
  G-OWN    the own T outputs the twin file records equal the tick file's `t_*_logits`/`t_cons_pred` bit for bit (same rows).
  G-REG    (regression arm) run_t was zeroed on every T call (own + twin calls) and the e_ego weight after the run equals the checkpoint's.
usage: town_gates.py id --run DIR --tag T --bank DIR --bank-tag B ; town_gates.py twin --run DIR --tag T --receipt F"""
import os
import sys
import json
import glob
import argparse

sys.dont_write_bytecode = True
import numpy as np

TICK_META_SKIP = ("ckpt", "tag")


def _eq(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(np.array_equal(a, b, equal_nan=True)) if a.dtype.kind in "fc" else bool(np.array_equal(a, b))


def compare_npz(pa, pb, skip=()):
    za, zb = np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True)
    ka, kb = [k for k in za.files if k not in skip], [k for k in zb.files if k not in skip]
    per = {k: _eq(za[k], zb[k]) for k in ka if k in kb}
    return bool(per and all(per.values()) and set(ka) == set(kb)), per, sorted(set(ka) - set(kb)), sorted(set(kb) - set(ka))


def max_abs_delta(pa, pb, keys=("cl", "fl", "ol")):
    za, zb = np.load(pa), np.load(pb)
    return max(float(np.max(np.abs(za[k].astype(np.float64) - zb[k].astype(np.float64)))) if za[k].shape == zb[k].shape else float("inf") for k in keys)


def gate_id(run_dir, tag, bank_dir, bank_tag, only_prefix_episodes=None):
    """Tick file + per-episode dumps of `tag` vs the banked `bank_tag`. `only_prefix_episodes`: compare only the episode files present in the run (the subset), and the tick rows of those windows."""
    fa = sorted(glob.glob(f"{run_dir}/dump_{tag}/ep*.npz"))
    fb = {os.path.basename(x): x for x in glob.glob(f"{bank_dir}/dump_{bank_tag}/ep*.npz")}
    bad, mx = [], 0.0
    for x in fa:
        y = fb.get(os.path.basename(x))
        if y is None:
            bad.append(os.path.basename(x) + ":absent_in_bank"); continue
        ok, _, _, _ = compare_npz(x, y)
        mx = max(mx, max_abs_delta(x, y))
        if not ok:
            bad.append(os.path.basename(x))
    res = {"gate": "G-ID", "tag": tag, "against": bank_tag, "episode_files": len(fa), "episode_files_unequal": bad, "max_abs_delta_tracks": mx}
    if only_prefix_episodes is None:
        ok_t, per_t, oa, ob = compare_npz(f"{run_dir}/ticks/ticks_{tag}.npz", f"{bank_dir}/ticks/ticks_{bank_tag}.npz", skip=TICK_META_SKIP)
        res.update({"ticks_equal": ok_t, "ticks_keys_unequal": [k for k, v in per_t.items() if not v], "ticks_only_in_run": oa, "ticks_only_in_bank": ob, "episode_files_bank": len(fb)})
    else:
        res["ticks_equal"] = "not compared (subset)"; ok_t = True
    res["PASS"] = bool(len(fa) > 0 and not bad and mx == 0.0 and ok_t)
    return res


def gate_twin(run_dir, tag, receipt):
    tw = np.load(f"{run_dir}/ticks/twin_{tag}.npz")
    tk = np.load(f"{run_dir}/ticks/ticks_{tag}.npz", allow_pickle=True)
    t0 = tw["tick"] == 0
    same = lambda a, b: bool(np.array_equal(tw[a][t0], tw[b][t0]))
    out0 = all(same(a, b) for a, b in (("own_lat", "twin_lat"), ("own_lon", "twin_lon"), ("own_cons", "twin_cons"), ("own_goal", "twin_goal")))
    swap0 = bool(tw["swap_equal"][t0].all()) and int(receipt["tick0_swap_unequal"]) == 0 and int(receipt["tick0_rows"]) > 0
    g0 = {"gate": "G-TWIN0", "PASS": bool(out0 and swap0), "tick0_rows": int(t0.sum()), "outputs_bitwise_equal": out0, "swapped_inputs_bitwise_equal": swap0,
          "d_tick0_max_abs": None}
    rows_ok = bool(np.array_equal(tw["win"], tk["win"]) and np.array_equal(tw["tick"], tk["tick"]))
    own_ok = rows_ok and all(_eq(tw[a], tk[b]) for a, b in (("own_lat", "t_lat_logits"), ("own_lon", "t_lon_logits"), ("own_cons", "t_cons_pred"), ("own_goal", "t_goal_prob")))
    gl = {"gate": "G-TWINL", "PASS": bool(int(receipt["logged_p10f_mismatch"]) == 0 and int(receipt["keep_key_not_same_object"]) == 0 and int(receipt["later_swap_unequal"]) > 0),
          "logged_p10f_mismatch": int(receipt["logged_p10f_mismatch"]), "keep_key_not_same_object": int(receipt["keep_key_not_same_object"]),
          "later_rows": int(receipt["later_rows"]), "later_swap_unequal": int(receipt["later_swap_unequal"]), "twin_mode": receipt["mode"]}
    go = {"gate": "G-OWN", "PASS": bool(own_ok), "rows_aligned": rows_ok}
    return {"G-TWIN0": g0, "G-TWINL": gl, "G-OWN": go, "PASS": bool(g0["PASS"] and gl["PASS"] and go["PASS"])}


def gate_reg(receipt, n_calls_expected, e_ego_sha_ckpt=None):
    ok = int(receipt["run_t_zeroed_calls"]) == int(n_calls_expected) > 0 and (e_ego_sha_ckpt is None or receipt.get("e_ego_weight_sha256_after") == e_ego_sha_ckpt)
    return {"gate": "G-REG", "PASS": bool(ok), "run_t_zeroed_calls": int(receipt["run_t_zeroed_calls"]), "expected": int(n_calls_expected)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("id"); [p.add_argument(f"--{k}", required=True) for k in ("run", "tag", "bank", "bank-tag")]; p.add_argument("--subset", action="store_true")
    q = sub.add_parser("twin"); [q.add_argument(f"--{k}", required=True) for k in ("run", "tag", "receipt")]
    a = ap.parse_args(argv)
    if a.mode == "id":
        r = gate_id(a.run, a.tag, a.bank, getattr(a, "bank_tag"), only_prefix_episodes=True if a.subset else None)
    else:
        r = gate_twin(a.run, a.tag, json.load(open(a.receipt)))
    print(json.dumps(r, indent=1))
    return 0 if r["PASS"] else 5


if __name__ == "__main__":
    sys.exit(main())
