"""a3_keq_amend2_compare.py -- PREREG_REFCV9_INMODEL_PA AMENDMENT2 (865c9569..., sealed 16:20:56, post-failure, disclosed): K-EQ made
DEVICE-MATCHED -> result/R9PF_KEQ_AMEND2.json.  CPU only, no model.  NEW file (EvalFlyWheel run 2, 2026-10-26).  Never touches result/R9PF_KEQ.json.

  K-EQ (AMENDMENT2 change 1): the in-forward ON roll re-run with the model on CPU (a3_keq_roll_amend2.py = a3_keq_roll.py with ONLY the output dir
     renamed + a CPU/eval139/ON-only guard; same pinned fix tree, same flags, same window rule) at inference seeds 0 AND 1, compared with the CPU
     offline composition (r9pf_common.compose, the registered head) on the forward's OWN selected path (traj_preretime), own z_tac, own r9_x and the
     scoring prior8 (EVAL_INPUTS) -- EXACTLY a3_keq_compare.k_eq (copied verbatim below, parameterised by the roll) -- per window
     max |d| <= 2 x ulp_float32(max |coordinate|) AND 100 % decision agreement, >= 500 windows; both seeds required.
  BESIDE (change 2): the CUDA-vs-CPU residual -- (a) the banked CUDA ON roll through the same k_eq (must reproduce R9PF_KEQ.json: a known-value
     control on this copy), (b) CUDA in-forward plan vs CPU in-forward plan directly (same windows; path / decision agreement).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9pf_common as RC  # noqa: E402

HEAD = "R9PF_fs0"
MIN_WINDOWS = 500
TREE = "step48852ema_pf"
AMEND2 = RC.WORK / "PREREG_REFCV9_INMODEL_PA_AMENDMENT2.md"
AMEND2_SHA = "865c95692383f78a0c6355003c6d1b0ec63413a5816827babacb28cc5674cc3d"
VOID_KEQ = RC.RESULT / "R9PF_KEQ.json"
VOID_KEQ_SHA = "d1de8a28b421379659b24bfa38787e30b3f8ce0452268db367d90faf7b281da7"
TEST = "--plumbing-test" in sys.argv     # PLUMBING ONLY: the banked CUDA roll stands in for the CPU roll; output to scratch, never result/


def check_amendment2() -> dict:
    got = RC.sha256_file(AMEND2) if AMEND2.is_file() else ""
    seal = RC.hex64(AMEND2.with_suffix(".sha256").read_text(encoding="utf-8-sig")) if AMEND2.with_suffix(".sha256").is_file() else ""
    if len(got) != 64 or len(seal) != 64 or not (got == seal == AMEND2_SHA):
        raise SystemExit(f"[keqa2] AMENDMENT2 seal mismatch / INCONCLUSIVE ({got} / {seal} / {AMEND2_SHA}) -- refusing")
    return {"file": str(AMEND2), "sha256": got, "seal_matches": True}


def load_roll(d: Path) -> dict:
    rec = json.loads((d / "ROLL_RECORD.json").read_text(encoding="utf-8"))
    for f, k in (("keq_records.npz", "records_sha256"), ("OUTPUT_DIGESTS.json", "digests_sha256")):
        got = RC.sha256_file(d / f)
        if len(got) != 64 or got != rec[k]:
            raise SystemExit(f"[keqa2] {d / f} does not re-hash to its roll record ({got} vs {rec[k]}) -- refusing")
    if Path(rec["launch_tree"]).resolve() != RC.FIX_TREE.resolve():
        raise SystemExit(f"[keqa2] roll {d.name} ran on {rec['launch_tree']}, not the pinned fix tree -- refusing")
    if rec["fix_pin"]["pins"].get(Path(rec["refc_v3_file"]).resolve().as_posix()) != rec["refc_v3_sha256"]:
        raise SystemExit(f"[keqa2] roll {d.name} imported an unpinned refc_v3.py -- refusing")
    return {"dir": d, "rec": rec, "z": np.load(d / "keq_records.npz", allow_pickle=False)}


# ---- verbatim from a3_keq_compare.k_eq (7db3749f...), `A.head` / `A.min_windows` -> HEAD / MIN_WINDOWS; ONE field added (n_windows_bit_equal)
def k_eq(seed: int, on: dict) -> dict:
    h, blob, sh = RC.load_head(RC.HEADS / f"{HEAD}.pt", registered_sha=reg_sha)
    if blob["decision"] not in ("argmax", "regime"):
        raise SystemExit(f"[keqa2] K-EQ head {HEAD} decides by {blob['decision']} -- needs the trunk lon logits; not wired")
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
    res = {"head": HEAD, "head_sha256": sh, "n_windows_rolled": int(len(m)), "n_windows_joined_to_eval_inputs": n,
           "max_abs_m": float(d.max()) if n else None, "max_ratio_to_2ulp": float((d / tol).max()) if n else None,
           "n_windows_over_tol": int((d > tol).sum()), "max_abs_coordinate_m": float(coord.max()) if n else None,
           "decision_agreement": dec_agree,
           "n_windows_bit_equal": int((d == 0).sum()),
           "beside_input_agreement": {
               "z_forward_vs_eval_bank_max_abs": (float(np.abs(z["z"][jb >= 0].astype(np.float64) - zb["z_tac"][jb[jb >= 0]].astype(np.float64)).max())
                                                  if (jb >= 0).any() else None),
               "z_joined_to_bank": int((jb >= 0).sum()),
               "x_forward_vs_offline_x9_max_abs": float(np.abs(z["x"][m].astype(np.float64) - I["S2_x9"][ji[m]].astype(np.float64)).max()) if n else None,
               "s_prior_forward_vs_offline_arc_max_abs_m": float(np.abs(z["s_p"][m].astype(np.float64) - sp_off).max()) if n else None,
               "s_target_forward_vs_offline_max_abs_m": float(np.abs(z["s_t"][m].astype(np.float64) - off["s_target"].astype(np.float64)).max()) if n else None}}
    res["PASS"] = bool(n >= MIN_WINDOWS and res["n_windows_over_tol"] == 0 and dec_agree == 1.0)
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


def beside_batch(seed: int, on: dict) -> dict:
    """BESIDE, NOT REGISTERED (added after a pre-roll diagnostic showed CPU batch-515 vs batch-1 composition of the SAME inputs differs by > 2 ulp on
    13-14 / 515 windows): the in-forward call is batch 1 (counters batch_gt1 = 0), the registered offline composition is batched.  Reads:
    C1 offline composition one window at a time (registered inputs: scoring prior8) vs the in-forward plan;
    C2 head_forward + retime one window at a time on the forward's OWN inputs (z, r9_x, r9_s_prior, traj_preretime) vs the in-forward plan;
    C3 the offline composition's own batch floor: batched vs one-at-a-time, same inputs."""
    import torch
    h, blob, _sh = RC.load_head(RC.HEADS / f"{HEAD}.pt", registered_sha=reg_sha)
    R = RC.r9p()
    cfg = R.R9Config(**blob["r9_config"])
    z = on["z"]
    zb = np.load(RC.EVAL_BANK / f"ztac_eval_s{seed}.npz", allow_pickle=False)
    W = int(json.loads(str(zb["meta_json"]))["window_W"])
    I = np.load(RC.SCRATCH / "arms" / f"EVAL_INPUTS_{TREE}_s{seed}.npz", allow_pickle=False)
    ik = {(int(c), int(w)): j for j, (c, w) in enumerate(zip(I["S2_clip"], I["S2_ws"]))}
    ji = np.array([ik.get((int(c), int(t) + W - 1), -1) for c, t in zip(z["e_i"], z["t"])])
    m = np.flatnonzero(ji >= 0)
    P = I["S2_prior8"][ji[m]]
    fwd = z["traj"][m].astype(np.float64)
    batched = RC.compose(h, blob, z["z"][m], z["x"][m], P, z["pre"][m])["traj"].astype(np.float64)
    c1 = np.concatenate([RC.compose(h, blob, z["z"][i:i + 1], z["x"][i:i + 1], P[k:k + 1], z["pre"][i:i + 1])["traj"]
                         for k, i in enumerate(m)]).astype(np.float64)
    c2 = []
    with torch.no_grad():
        for i in m:
            f = lambda a: torch.from_numpy(np.ascontiguousarray(z[a][i:i + 1], np.float32))  # noqa: E731
            o = R.head_forward(h, cfg, f("z"), f("x"), f("s_p"), None)
            c2.append(R.retime(f("pre"), o["r9_s_target"]).to(torch.float32).numpy())
    c2 = np.concatenate(c2).astype(np.float64)
    n = len(m)

    def rd(a, b):
        d = np.abs(a - b).reshape(n, -1).max(1)
        coord = np.maximum(np.abs(a).reshape(n, -1).max(1), np.abs(b).reshape(n, -1).max(1))
        tol = 2.0 * np.spacing(coord.astype(np.float32)).astype(np.float64)
        return {"max_abs_m": float(d.max()), "n_over_2ulp": int((d > tol).sum()), "n_bit_equal": int((d == 0).sum()), "n": n}
    return {"rule": "BESIDE, NOT REGISTERED -- batch-size arithmetic vs the registered (batched) offline composition", "C1_offline_batch1_vs_inforward": rd(fwd, c1),
            "C2_own_inputs_batch1_vs_inforward": rd(fwd, c2), "C3_offline_batched_vs_batch1_same_inputs": rd(batched, c1)}


def direct(cpu: dict, cuda: dict) -> dict:
    a, b = cpu["z"], cuda["z"]
    same_keys = bool(np.array_equal(a["e_i"], b["e_i"]) and np.array_equal(a["t"], b["t"]))
    if not same_keys:
        return {"same_window_sequence": False}
    n = len(a["e_i"])
    pre_d = np.abs(a["pre"].astype(np.float64) - b["pre"].astype(np.float64)).reshape(n, -1).max(1)
    tr_d = np.abs(a["traj"].astype(np.float64) - b["traj"].astype(np.float64)).reshape(n, -1).max(1)
    coord = np.maximum(np.abs(a["traj"].astype(np.float64)).reshape(n, -1).max(1), np.abs(b["traj"].astype(np.float64)).reshape(n, -1).max(1))
    tol = 2.0 * np.spacing(coord.astype(np.float32)).astype(np.float64)
    z_d = float(np.abs(a["z"].astype(np.float64) - b["z"].astype(np.float64)).max())
    x_d = float(np.abs(a["x"].astype(np.float64) - b["x"].astype(np.float64)).max())
    return {"same_window_sequence": True, "n_windows": n,
            "selected_path_traj_preretime": {"n_bit_equal": int((pre_d == 0).sum()), "max_abs_m": float(pre_d.max()),
                                             "n_over_1e-3_m": int((pre_d > 1e-3).sum())},
            "in_forward_plan": {"max_abs_m": float(tr_d.max()), "n_over_2ulp": int((tr_d > tol).sum()), "n_bit_equal": int((tr_d == 0).sum())},
            "decision_agreement": float((a["dec"] == b["dec"]).mean()), "z_tac_max_abs": z_d, "r9_x_max_abs": x_d,
            "note": "CUDA and CPU forwards each select their OWN path; where the selected path differs the plan difference is a selection "
                    "difference, not head arithmetic -- the like-for-like CUDA-vs-CPU head residual is (a)"}


RC.check_prereg()
amend1 = RC.check_amendment1()
amend2 = check_amendment2()
fixpin = RC.check_fix_pin()
if RC.sha256_file(VOID_KEQ) != VOID_KEQ_SHA:
    raise SystemExit("[keqa2] result/R9PF_KEQ.json changed since the VOID verdict -- refusing")
reg_sha = ""
for ln in (RC.WORK / "REGISTERED_R9PF_PA.sha256").read_text(encoding="utf-8-sig").splitlines():
    p = ln.strip().split(None, 1)
    if len(p) == 2 and Path(p[1].lstrip("*").strip()).as_posix() == (RC.HEADS / f"{HEAD}.pt").as_posix():
        reg_sha = RC.hex64(p[0])
if len(reg_sha) != 64:
    raise SystemExit("[keqa2] head not pinned in REGISTERED_R9PF_PA.sha256 -- refusing")
void = json.loads(VOID_KEQ.read_text(encoding="utf-8"))
t0 = time.time()
KEQ, BES, rolls, same = {}, {}, {}, {}
NOT_RUN = {}
for seed in (0, 1):
    cpu_dir = RC.SCRATCH / (f"keq_s{seed}_on" if TEST else f"keq_amend2_cpu_s{seed}_on")
    if not (cpu_dir / "ROLL_RECORD.json").is_file():
        # K-EQ is a CONJUNCTION over both seeds: a FAILED seed decides it; an absent roll with nothing failed is INCOMPLETE (no record written)
        if any(not v["PASS"] for v in KEQ.values()) or any(not v["ALL"] for v in same.values()):
            NOT_RUN[f"s{seed}"] = {"status": "NOT RUN -- an earlier inference seed already FAILED the conjunction (both seeds required); "
                                             "no roll record exists for this seed", "roll_dir": str(cpu_dir)}
            continue
        raise SystemExit(f"[keqa2] INCOMPLETE: {cpu_dir}/ROLL_RECORD.json absent and no earlier seed failed -- no record written")
    cpu = load_roll(cpu_dir)
    cuda = load_roll(RC.SCRATCH / f"keq_s{seed}_on")
    rc_, ru_ = cpu["rec"], cuda["rec"]
    hc = {k: v for k, v in rc_["head"].items() if k != "device"}
    hu = {k: v for k, v in ru_["head"].items() if k != "device"}
    same[f"s{seed}"] = {"device_cpu": rc_["device"] == "cpu" and str(rc_["head"].get("device")) == "cpu",
                        "device_banked": ru_["device"], "flags_on_both": rc_["flags"] == ru_["flags"] == "ON (head built, decision applied)",
                        "same_tree": rc_["launch_tree"] == ru_["launch_tree"], "same_refc_v3_sha": rc_["refc_v3_sha256"] == ru_["refc_v3_sha256"],
                        "same_head_attached_except_device": hc == hu, "same_subset": rc_["subset"] == ru_["subset"],
                        "same_seed": rc_["seed"] == ru_["seed"] == seed, "same_pose_clock": rc_["pose_clock"] == ru_["pose_clock"],
                        "same_window_sequence": bool(np.array_equal(cpu["z"]["e_i"], cuda["z"]["e_i"]) and np.array_equal(cpu["z"]["t"], cuda["z"]["t"])),
                        "counters_cpu": rc_["counters"]}
    same[f"s{seed}"]["ALL"] = all(v for k, v in same[f"s{seed}"].items() if isinstance(v, bool))
    for tag, r in (("cpu_on", cpu), ("cuda_on_banked", cuda)):
        rolls[f"s{seed}_{tag}"] = {"roll_record": str(r["dir"] / "ROLL_RECORD.json"), "sha256": RC.sha256_file(r["dir"] / "ROLL_RECORD.json"),
                                   "device": r["rec"]["device"], "n_records": r["rec"]["n_records"], "wall_s": r["rec"]["wall_s"],
                                   "script_sha256": r["rec"]["script_sha256"]}
    KEQ[f"s{seed}"] = k_eq(seed, cpu)
    KEQ[f"s{seed}"]["BESIDE_batch_size_not_registered"] = beside_batch(seed, cpu)
    cu = k_eq(seed, cuda)
    vk = void["K-EQ"]["per_seed"][f"s{seed}"]
    repro = all(cu[k] == vk[k] for k in ("max_abs_m", "n_windows_over_tol", "decision_agreement", "max_ratio_to_2ulp", "n_windows_joined_to_eval_inputs"))
    BES[f"s{seed}"] = {"a_cuda_inforward_vs_cpu_offline": {k: cu[k] for k in ("max_abs_m", "max_ratio_to_2ulp", "n_windows_over_tol", "n_windows_bit_equal",
                                                                               "decision_agreement", "n_windows_joined_to_eval_inputs")},
                       "a_reproduces_banked_R9PF_KEQ_json": bool(repro),
                       "b_cuda_inforward_vs_cpu_inforward": direct(cpu, cuda),
                       "c_cuda_inforward_batch_size_not_registered": beside_batch(seed, cuda)}
keq_pass = (not NOT_RUN) and all(v["PASS"] for v in KEQ.values()) and all(v["ALL"] for v in same.values())
ctl_ok = all(v["a_reproduces_banked_R9PF_KEQ_json"] for v in BES.values())
rec = {"tool": "a3_keq_amend2_compare.py", "script_sha256": RC.sha256_file(__file__), "amendment1": amend1, "amendment2": amend2, "fix_pin": fixpin,
       "label": "K-EQ under AMENDMENT2 (post-failure, device-matched)",
       "tier": "in-forward subset roll on logged eval139 frames (os-only, per-frame nav feed) -- a CONTROL, not a bar",
       "rolls": rolls, "same_tree_flags_windows": same,
       "K-EQ_AMEND2": {"rule": "AMENDMENT2 change 1: in-forward ON roll with the model on CPU vs the CPU offline composition on the forward's own selected "
                               "path; per window max |d| <= 2 x ulp_float32(max |coordinate|) AND 100 % decision agreement, >= 500 windows, inference "
                               "seeds 0 AND 1; same tree / flags / windows asserted against the banked CUDA roll", "per_seed": KEQ, "seeds_not_run": NOT_RUN,
                       "PASS": bool(keq_pass)},
       "BESIDE_cuda_vs_cpu_residual": {"rule": "AMENDMENT2 change 2 -- the device a CUDA deployment would run on", "per_seed": BES,
                                       "control_copy_reproduces_banked_KEQ": bool(ctl_ok)},
       "void_record_untouched": {"file": str(VOID_KEQ), "sha256": RC.sha256_file(VOID_KEQ)},
       "wall_s": round(time.time() - t0, 1), "generated": time.strftime("%FT%T%z")}
RC.jdump(rec, (RC.SCRATCH / "KEQ_AMEND2_PLUMBING_TEST.json") if TEST else (RC.RESULT / "R9PF_KEQ_AMEND2.json"))
print(f"[keqa2] K-EQ(AMEND2) {keq_pass}  copy-control {ctl_ok}  ZZKEQA2_{int(keq_pass)}{int(ctl_ok)}ZZ", flush=True)
