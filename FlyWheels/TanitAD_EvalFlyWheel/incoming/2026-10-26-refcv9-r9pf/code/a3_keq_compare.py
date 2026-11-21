"""a3_keq_compare.py -- PREREG_REFCV9_INMODEL_PA AMENDMENT1 K-OFF / K-EQ from the banked subset rolls (a3_keq_roll.py) -> result/R9PF_KEQ.json.
CPU only, no model.  NEW file (EvalFlyWheel 2026-10-26).

  K-OFF (amended, change 2): ONE tree (tip + R9 patch + the pinned RNG fix), flags-ON vs flags-OFF, per inference seed 0 and 1:
     * the same windows in the same order (the per-call index sequence is identical);
     * ON traj_preretime == OFF traj, BIT for bit, on every call (digest) and every recorded nav_true row (array);
     * every output OFF produces other than `traj` has an identical digest in ON on every call ("every non-R9 output bit-equal");
     * the keys ON adds over OFF are R9 keys only (traj_preretime, r9_*); anything else fails;
     * >= 500 eval windows.
  K-EQ (amended, change 3): same-input form, per inference seed 0 and 1 (ON roll): the offline composition (r9pf_common.compose, the registered
     head file) on the forward's OWN selected path (traj_preretime), the forward's own z_tac (lat_head_tac input) and own x9 (r9_x), and the
     scoring prior8 (EVAL_INPUTS) -> per window max |d| <= 2 x ulp_float32(max |coordinate| of that window), AND 100 % decision agreement,
     on >= 500 eval windows.  The input agreements (forward z vs the eval bank, forward x vs the offline x9, forward s_prior vs the offline
     arc of prior8) are reported beside.

    python -X utf8 a3_keq_compare.py [--head R9PF_fs0] [--min-windows 500]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9pf_common as RC  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--head", default="R9PF_fs0")
ap.add_argument("--min-windows", type=int, default=500)
A = ap.parse_args()
TREE = "step48852ema_pf"
R9_KEYS_OK = ("traj_preretime", "r9_")


def load_roll(seed: int, flag: str) -> dict:
    d = RC.SCRATCH / f"keq_s{seed}_{flag}"
    rec = json.loads((d / "ROLL_RECORD.json").read_text(encoding="utf-8"))
    for f, k in (("keq_records.npz", "records_sha256"), ("OUTPUT_DIGESTS.json", "digests_sha256")):
        got = RC.sha256_file(d / f)
        if len(got) != 64 or got != rec[k]:
            raise SystemExit(f"[keqc] {d / f} does not re-hash to its roll record ({got} vs {rec[k]}) -- refusing")
    return {"dir": d, "rec": rec, "z": np.load(d / "keq_records.npz", allow_pickle=False),
            "dig": json.loads((d / "OUTPUT_DIGESTS.json").read_text(encoding="utf-8"))}


def k_off(seed: int, on: dict, off: dict) -> dict:
    Don, Doff = on["dig"], off["dig"]
    out = {"n_calls_on": len(Don), "n_calls_off": len(Doff), "n_windows": int(on["rec"]["n_records"]),
           "subset_equal": on["rec"]["subset"] == off["rec"]["subset"], "same_tree": on["rec"]["launch_tree"] == off["rec"]["launch_tree"],
           "same_refc_v3_sha": on["rec"]["refc_v3_sha256"] == off["rec"]["refc_v3_sha256"],
           "flags": [on["rec"]["flags"], off["rec"]["flags"]]}
    same_seq = len(Don) == len(Doff) and all(a["idx"] == b["idx"] and a["navnull"] == b["navnull"] for a, b in zip(Don, Doff))
    out["same_call_sequence"] = bool(same_seq)
    added, pre_bad, nonr9_bad, addr = set(), 0, {}, set()
    # PLAN ALIASES: outputs that ARE the emitted plan tensor (refc_v3.py:2742 `out["wp_seq"] = traj`, set AFTER the R9 re-time at :2723).
    # Established from data, not assumed: digest(k) == digest("traj") on EVERY call in BOTH runs.  Such a key carries R9's plan by
    # construction, so it is checked like `traj`: ON traj_preretime == OFF k.
    aliases = sorted(k for k in (set(Doff[0]["d"]) & set(Don[0]["d"])) - {"traj"}
                     if all(c["d"].get(k) == c["d"]["traj"] for c in Don) and all(c["d"].get(k) == c["d"]["traj"] for c in Doff)) if same_seq else []
    alias_bad = 0
    if same_seq:
        for a, b in zip(Don, Doff):
            added |= set(a["d"]) - set(b["d"])
            for k, v in b["d"].items():
                if v.startswith("OBJ_ADDR|"):
                    addr.add(k)
                    continue
                if k == "traj":
                    if a["d"].get("traj_preretime", "<absent>") != v:
                        pre_bad += 1
                elif k in aliases:
                    if a["d"].get("traj_preretime", "<absent>") != v:
                        alias_bad += 1
                elif a["d"].get(k, "<absent>") != v:
                    nonr9_bad[k] = nonr9_bad.get(k, 0) + 1
    unexpected = sorted(k for k in added if not k.startswith(R9_KEYS_OK))
    za, zb = on["z"], off["z"]
    arr_pre_eq = bool(np.array_equal(za["pre"], zb["traj"]))
    arr_z_eq = bool(np.array_equal(za["z"], zb["z"]))
    r9_moves = int((np.abs(za["traj"].astype(np.float64) - zb["traj"].astype(np.float64)).reshape(len(za["traj"]), -1).max(1) > 0).sum()) \
        if za["traj"].shape == zb["traj"].shape else None
    out.update({"plan_aliases_checked_as_traj": aliases, "plan_alias_source": "refc_v3.py:2742 out['wp_seq'] = traj (after the R9 re-time, :2723)",
                "calls_alias_ne_off": alias_bad, "keys_added_by_R9": sorted(added), "unexpected_added_keys": unexpected, "calls_traj_preretime_ne_off_traj": pre_bad,
                "non_R9_keys_differing_calls": nonr9_bad, "address_bearing_keys_excluded": sorted(addr),
                "records_pre_on_eq_traj_off_bit": arr_pre_eq, "records_z_on_eq_z_off_bit": arr_z_eq,
                "beside_windows_where_R9_moved_the_plan": r9_moves})
    out["PASS"] = bool(out["subset_equal"] and out["same_tree"] and out["same_refc_v3_sha"] and same_seq and pre_bad == 0 and alias_bad == 0 and not nonr9_bad
                       and not unexpected and arr_pre_eq and arr_z_eq and out["n_windows"] >= A.min_windows)
    return out


def k_eq(seed: int, on: dict) -> dict:
    h, blob, sh = RC.load_head(RC.HEADS / f"{A.head}.pt", registered_sha=reg_sha)
    if blob["decision"] not in ("argmax", "regime"):
        raise SystemExit(f"[keqc] K-EQ head {A.head} decides by {blob['decision']} -- needs the trunk lon logits; not wired")
    z = on["z"]
    zb = np.load(RC.EVAL_BANK / f"ztac_eval_s{seed}.npz", allow_pickle=False)
    W = int(json.loads(str(zb["meta_json"]))["window_W"])
    bk = {(int(c), int(t)): j for j, (c, t) in enumerate(zip(zb["e_i"], zb["t"]))}
    jb = np.array([bk.get((int(c), int(t)), -1) for c, t in zip(z["e_i"], z["t"])])
    I = np.load(RC.SCRATCH / "arms" / f"EVAL_INPUTS_{TREE}_s{seed}.npz", allow_pickle=False)
    ik = {(int(c), int(w)): j for j, (c, w) in enumerate(zip(I["S2_clip"], I["S2_ws"]))}
    ji = np.array([ik.get((int(c), int(t) + W - 1), -1) for c, t in zip(z["e_i"], z["t"])])
    m = ji >= 0
    n = int(m.sum())
    off = RC.compose(h, blob, z["z"][m], z["x"][m], I["S2_prior8"][ji[m]], z["pre"][m])
    fwd = z["traj"][m].astype(np.float64)
    offt = off["traj"].astype(np.float64)
    d = np.abs(fwd - offt).reshape(n, -1).max(1)
    coord = np.maximum(np.abs(fwd).reshape(n, -1).max(1), np.abs(offt).reshape(n, -1).max(1))
    tol = 2.0 * np.spacing(coord.astype(np.float32)).astype(np.float64)
    dec_agree = float((z["dec"][m] == off["decision"]).mean()) if n else 0.0
    sp_off = RC.r9p().arc_at_slots(__import__("torch").from_numpy(I["S2_prior8"][ji[m]].astype(np.float32))).numpy().astype(np.float64)
    res = {"head": A.head, "head_sha256": sh, "n_windows_rolled": int(len(m)), "n_windows_joined_to_eval_inputs": n,
           "max_abs_m": float(d.max()) if n else None, "max_ratio_to_2ulp": float((d / tol).max()) if n else None,
           "n_windows_over_tol": int((d > tol).sum()), "max_abs_coordinate_m": float(coord.max()) if n else None,
           "decision_agreement": dec_agree,
           "beside_input_agreement": {
               "z_forward_vs_eval_bank_max_abs": (float(np.abs(z["z"][jb >= 0].astype(np.float64) - zb["z_tac"][jb[jb >= 0]].astype(np.float64)).max())
                                                  if (jb >= 0).any() else None),
               "z_joined_to_bank": int((jb >= 0).sum()),
               "x_forward_vs_offline_x9_max_abs": float(np.abs(z["x"][m].astype(np.float64) - I["S2_x9"][ji[m]].astype(np.float64)).max()) if n else None,
               "s_prior_forward_vs_offline_arc_max_abs_m": float(np.abs(z["s_p"][m].astype(np.float64) - sp_off).max()) if n else None,
               "s_target_forward_vs_offline_max_abs_m": float(np.abs(z["s_t"][m].astype(np.float64) - off["s_target"].astype(np.float64)).max()) if n else None}}
    res["PASS"] = bool(n >= A.min_windows and res["n_windows_over_tol"] == 0 and dec_agree == 1.0)
    # ---- BESIDE (not the registered reading): where the residual comes from
    import torch
    R = RC.r9p()
    cfg = R.R9Config(**blob["r9_config"])
    with torch.no_grad():
        o2 = R.head_forward(h, cfg, torch.from_numpy(np.ascontiguousarray(z["z"][m], np.float32)), torch.from_numpy(np.ascontiguousarray(z["x"][m], np.float32)),
                            torch.from_numpy(np.ascontiguousarray(z["s_p"][m], np.float32)), None)
        t2 = R.retime(torch.from_numpy(np.ascontiguousarray(z["pre"][m], np.float32)), o2["r9_s_target"]).to(torch.float32).numpy().astype(np.float64)
    d2 = np.abs(fwd - t2).reshape(n, -1).max(1)
    pr = np.abs(I["S2_prior8"][ji[m]].astype(np.float64)).reshape(n, -1).max(1)
    tol_in = 2.0 * np.spacing(np.maximum(coord, pr).astype(np.float32)).astype(np.float64)
    res["BESIDE_not_registered"] = {
        "A_forward_own_s_prior": {"rule": "offline head_forward + retime with the forward's OWN s_prior (r9_s_prior) instead of arc_at_slots(scoring prior8)",
                                  "max_abs_m": float(d2.max()), "n_windows_over_2ulp_output_coord": int((d2 > tol).sum()),
                                  "decision_agreement": float((z["dec"][m] == o2["r9_decision"].numpy()).mean())},
        "B_coordinate_scope_incl_prior": {"rule": "registered offline composition; tolerance 2 ulp of max |coordinate| over the window's plan AND its prior8",
                                          "n_windows_over": int((d > tol_in).sum()), "max_ratio": float((d / tol_in).max())}}
    return res


RC.check_prereg()
amend = RC.check_amendment1()
fixpin = RC.check_fix_pin()
reg_sha = ""
for ln in (RC.WORK / "REGISTERED_R9PF_PA.sha256").read_text(encoding="utf-8-sig").splitlines():
    p = ln.strip().split(None, 1)
    if len(p) == 2 and Path(p[1].lstrip("*").strip()).as_posix() == (RC.HEADS / f"{A.head}.pt").as_posix():
        reg_sha = RC.hex64(p[0])
if len(reg_sha) != 64:
    raise SystemExit("[keqc] head not pinned in REGISTERED_R9PF_PA.sha256 -- refusing")
t0 = time.time()
KOFF, KEQ, rolls = {}, {}, {}
for seed in (0, 1):
    on, off = load_roll(seed, "on"), load_roll(seed, "off")
    for flag, r in (("on", on), ("off", off)):
        if Path(r["rec"]["launch_tree"]).resolve() != RC.FIX_TREE.resolve():
            raise SystemExit(f"[keqc] roll s{seed}_{flag} ran on {r['rec']['launch_tree']}, not the pinned fix tree -- refusing")
        if r["rec"]["fix_pin"]["pins"].get(Path(r["rec"]["refc_v3_file"]).resolve().as_posix()) != r["rec"]["refc_v3_sha256"]:
            raise SystemExit(f"[keqc] roll s{seed}_{flag} imported an unpinned refc_v3.py -- refusing")
        rolls[f"s{seed}_{flag}"] = {"roll_record": str(r["dir"] / "ROLL_RECORD.json"), "sha256": RC.sha256_file(r["dir"] / "ROLL_RECORD.json"),
                                    "n_records": r["rec"]["n_records"], "counters": r["rec"]["counters"], "pose_clock": r["rec"]["pose_clock"],
                                    "wall_s": r["rec"]["wall_s"]}
    KOFF[f"s{seed}"] = k_off(seed, on, off)
    KEQ[f"s{seed}"] = k_eq(seed, on)
rec = {"tool": "a3_keq_compare.py", "script_sha256": RC.sha256_file(__file__), "amendment1": amend, "fix_pin": fixpin, "rolls": rolls,
       "tier": "in-forward subset roll on logged eval139 frames (os-only, per-frame nav feed) -- a CONTROL, not a bar",
       "K-OFF": {"rule": "AMENDMENT1 change 2: one tree (tip + patch + pinned fix), flags-ON vs flags-OFF: traj_preretime and every non-R9 output "
                         "bit-equal, >= 500 eval windows, both inference seeds", "per_seed": KOFF, "PASS": all(v["PASS"] for v in KOFF.values())},
       "K-EQ": {"rule": "AMENDMENT1 change 3: in-forward R9 plan vs the offline composition on the forward's OWN selected path (same inputs); per window "
                        "max |d| <= 2 x ulp_float32(max |coordinate|) AND 100 % decision agreement, >= 500 eval windows; both seeds required "
                        "(the conservative reading)", "per_seed": KEQ, "PASS": all(v["PASS"] for v in KEQ.values())},
       "wall_s": round(time.time() - t0, 1), "generated": time.strftime("%FT%T%z")}
RC.jdump(rec, RC.RESULT / "R9PF_KEQ.json")
print(f"[keqc] K-OFF {rec['K-OFF']['PASS']} K-EQ {rec['K-EQ']['PASS']}  ZZKEQC_{int(rec['K-OFF']['PASS'])}{int(rec['K-EQ']['PASS'])}ZZ", flush=True)
