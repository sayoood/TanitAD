"""tswap_gates: the four controls of PREREG_V8F_TSWAP.md ('Controls'), each a pure function of artifacts (so a mutant's artifacts make it FAIL).
  G-ID     (control 1) the wrapper's OWN arm == the bare pinned tool at 4 threads, BIT FOR BIT, on every record: the tick file (all keys) and every per-episode dump (all keys), max |delta| over the executed tracks 0.0.
  G-TICK0  (control 2) at the first replan of every window the own state equals the logged state: the swapped inputs are bitwise equal, the code O read equals T's own code, and the TSWAP tick-file rows of tick 0
           (plan, tau, P, T heads, S heads, ...) equal OWN's bit for bit.
  G-MUT    (control 3) the mutant (twin := own) reproduces OWN bit for bit on every record (tick file + dumps), with 0 swapped inputs and 0 swapped codes at every tick.
  G-LIVE   (control 4) the swap is live: on later replans the code O reads (the four T outputs) differs from T's own on > 0 rows (counts reported), the swapped inputs differ on > 0 rows, the twin p10f row equals the logged
           clip's row on every row, the kept keys are the same tensors, the code has 60 dims.
usage: tswap_gates.py --own DIR --own-tag T --bare DIR --bare-tag T --tswap DIR --tswap-tag T --mutant DIR --mutant-tag T --rec-own F --rec-tswap F --rec-mutant F --out F
(DIR = a rollout out-dir holding dump_<tag>/ep*.npz and ticks/ticks_<tag>.npz [+ ticks/tswap_<tag>.npz]; rec-* = the wrapper receipts)"""
import os
import sys
import json
import glob
import argparse

sys.dont_write_bytecode = True
import numpy as np

META = ("ckpt", "tag")


def _eq(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(np.array_equal(a, b, equal_nan=True)) if a.dtype.kind in "fc" else bool(np.array_equal(a, b))


def compare_npz(pa, pb, skip=META):
    za, zb = np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True)
    ka, kb = [k for k in za.files if k not in skip], [k for k in zb.files if k not in skip]
    per = {k: _eq(za[k], zb[k]) for k in ka if k in kb}
    return bool(per and all(per.values()) and set(ka) == set(kb)), sorted(k for k, v in per.items() if not v), sorted(set(ka) ^ set(kb))


def max_abs_delta(pa, pb, keys=("cl", "fl", "ol")):
    za, zb = np.load(pa), np.load(pb)
    return max(float(np.max(np.abs(za[k].astype(np.float64) - zb[k].astype(np.float64)))) if za[k].shape == zb[k].shape else float("inf") for k in keys)


def records_equal(dir_a, tag_a, dir_b, tag_b):
    """Tick file + every per-episode dump of run A vs run B, bit for bit."""
    fa = sorted(glob.glob(f"{dir_a}/dump_{tag_a}/ep*.npz"))
    fb = {os.path.basename(x): x for x in glob.glob(f"{dir_b}/dump_{tag_b}/ep*.npz")}
    bad, mx = [], 0.0
    for x in fa:
        y = fb.get(os.path.basename(x))
        if y is None:
            bad.append(os.path.basename(x) + ":absent"); continue
        ok, _, _ = compare_npz(x, y)
        mx = max(mx, max_abs_delta(x, y))
        if not ok:
            bad.append(os.path.basename(x))
    ok_t, keys_bad, keys_only = compare_npz(f"{dir_a}/ticks/ticks_{tag_a}.npz", f"{dir_b}/ticks/ticks_{tag_b}.npz")
    r = {"episode_files": len(fa), "episode_files_in_other": len(fb), "episode_files_unequal": bad, "max_abs_delta_tracks": mx, "ticks_equal": ok_t, "ticks_keys_unequal": keys_bad, "ticks_keys_only_in_one": keys_only}
    r["PASS"] = bool(len(fa) > 0 and len(fa) == len(fb) and not bad and mx == 0.0 and ok_t)
    return r


def gate_id(own_dir, own_tag, bare_dir, bare_tag):
    return {"gate": "G-ID", "own": own_tag, "bare": bare_tag, **records_equal(own_dir, own_tag, bare_dir, bare_tag)}


def gate_tick0(ts_dir, ts_tag, own_dir, own_tag, rec_ts):
    tz = np.load(f"{ts_dir}/ticks/ticks_{ts_tag}.npz", allow_pickle=True)
    oz = np.load(f"{own_dir}/ticks/ticks_{own_tag}.npz", allow_pickle=True)
    sd = np.load(f"{ts_dir}/ticks/tswap_{ts_tag}.npz")
    n = len(tz["tick"])
    m_t, m_o = tz["tick"] == 0, oz["tick"] == 0
    keys = [k for k in tz.files if k not in META and k in oz.files and tz[k].ndim >= 1 and len(tz[k]) == n and len(oz[k]) == len(oz["tick"])]
    per = {k: _eq(tz[k][m_t], oz[k][m_o]) for k in keys}
    t0 = sd["tick"] == 0
    code_eq = bool(np.array_equal(sd["code_read"][t0], sd["code_own"][t0])) and not bool(sd["code_unequal"][t0].any())
    swap_eq = bool(sd["swap_equal"][t0].all()) and int(rec_ts["tick0_swap_unequal"]) == 0 and int(rec_ts["tick0_code_unequal"]) == 0 and int(rec_ts["tick0_rows"]) > 0
    rows_ok = bool(np.array_equal(sd["win"], tz["win"]) and np.array_equal(sd["tick"], tz["tick"]))
    ok = bool(per and all(per.values()) and code_eq and swap_eq and rows_ok and int(t0.sum()) > 0)
    return {"gate": "G-TICK0", "PASS": ok, "tick0_rows": int(t0.sum()), "tick_file_keys_compared": len(keys), "tick_file_keys_unequal_at_tick0": sorted(k for k, v in per.items() if not v),
            "code_read_equals_own_at_tick0": code_eq, "swapped_inputs_bitwise_equal_at_tick0": swap_eq, "side_rows_aligned_with_tick_file": rows_ok}


def gate_mut(mu_dir, mu_tag, own_dir, own_tag, rec_mu):
    r = records_equal(mu_dir, mu_tag, own_dir, own_tag)
    sd = np.load(f"{mu_dir}/ticks/tswap_{mu_tag}.npz")
    none_swapped = bool(sd["swap_equal"].all()) and not bool(sd["code_unequal"].any()) and int(rec_mu["later_swap_unequal"]) == 0 and int(rec_mu["later_code_unequal"]) == 0 and int(rec_mu["tick0_code_unequal"]) == 0
    return {"gate": "G-MUT", "mutant": mu_tag, "own": own_tag, **r, "nothing_swapped_at_any_tick": none_swapped, "PASS": bool(r["PASS"] and none_swapped)}


def gate_live(ts_dir, ts_tag, rec_ts, own_dir, own_tag):
    sd = np.load(f"{ts_dir}/ticks/tswap_{ts_tag}.npz")
    tz = np.load(f"{ts_dir}/ticks/ticks_{ts_tag}.npz", allow_pickle=True)
    oz = np.load(f"{own_dir}/ticks/ticks_{own_tag}.npz", allow_pickle=True)
    later = sd["tick"] >= 1
    per_tick = {int(k): {"rows": int((sd["tick"] == k).sum()), "code_differs": int(sd["code_unequal"][sd["tick"] == k].sum()), "plan_differs_from_OWN": int((~np.all(tz["plan"][tz["tick"] == k] == oz["plan"][oz["tick"] == k], axis=(1, 2))).sum())}
                for k in range(int(sd["tick"].max()) + 1)}
    ok = bool(int(rec_ts["later_code_unequal"]) > 0 and int(rec_ts["later_swap_unequal"]) > 0 and int(rec_ts["logged_p10f_mismatch"]) == 0 and int(rec_ts["keep_key_not_same_object"]) == 0
              and int(rec_ts["code_dim_bad"]) == 0 and int(sd["code_unequal"][later].sum()) == int(rec_ts["later_code_unequal"]) and int(rec_ts["later_rows"]) == int(later.sum()))
    return {"gate": "G-LIVE", "PASS": ok, "later_rows": int(rec_ts["later_rows"]), "later_rows_code_O_reads_differs_from_T_own": int(rec_ts["later_code_unequal"]), "later_rows_swapped_inputs_differ": int(rec_ts["later_swap_unequal"]),
            "logged_p10f_mismatch": int(rec_ts["logged_p10f_mismatch"]), "keep_key_not_same_object": int(rec_ts["keep_key_not_same_object"]), "code_dim_bad": int(rec_ts["code_dim_bad"]), "per_tick": per_tick}


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ("own", "own-tag", "bare", "bare-tag", "tswap", "tswap-tag", "mutant", "mutant-tag", "rec-tswap", "rec-mutant", "out"):
        ap.add_argument(f"--{k}", default="")
    ap.add_argument("--skip-mutant", action="store_true")
    a = ap.parse_args(argv)
    L = lambda p: json.load(open(p))
    res = {"G-ID": gate_id(a.own, a.own_tag, a.bare, a.bare_tag), "G-TICK0": gate_tick0(a.tswap, a.tswap_tag, a.own, a.own_tag, L(a.rec_tswap)),
           "G-LIVE": gate_live(a.tswap, a.tswap_tag, L(a.rec_tswap), a.own, a.own_tag)}
    if not a.skip_mutant:
        res["G-MUT"] = gate_mut(a.mutant, a.mutant_tag, a.own, a.own_tag, L(a.rec_mutant))
    res["PASS"] = bool(all(v["PASS"] for k, v in res.items() if k.startswith("G-")))
    txt = json.dumps(res, indent=1)
    if a.out:
        open(a.out, "w", encoding="utf-8", newline="\n").write(txt + "\n")
    print(txt)
    return 0 if res["PASS"] else 5


if __name__ == "__main__":
    sys.exit(main())
