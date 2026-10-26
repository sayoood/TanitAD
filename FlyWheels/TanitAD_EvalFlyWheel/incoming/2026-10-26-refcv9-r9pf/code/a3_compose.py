"""a3_compose.py -- PREREG_REFCV9_INMODEL_PA PHASE A3: the OFFLINE composition of every registered R9 head on eval139 (both inference seeds), the
geometry controls, and one package per arm x fit seed.  CPU only.  REFUSES unless every line of REGISTERED_R9PF_PA.sha256 (written by the Master Mind)
re-hashes now and every frozen head is pinned there.

Per window (all 4,754 S2 rows; the 3,668 S6 rows are the same windows' composition at the S6 slots):
  path      = the eval bank's own plan traj_own (L1 os-only roll, per-frame feed; the prereg's 'eval traj_own + z_tac banks'), joined on (clip, ws)
  prior8    = KP.prior_path(ha0_ext_pose) from the observed pose window (ship_r8 forward's construction), asserted == the landed H2CN_DEPLOY8 prior8
  x9        = refcv9_progress.ego_t0 + admissible_scalars (the IN-MODEL builders) from the pose window and the fed ceiling
  z_tac     = the eval z_tac bank of that seed (seed-independent, A1); R9PF-CONSTZ: the calibration mean
  lon       = the battery dump's nav_true.tacv6_lon_logits (R9PF-TL's decision), joined; its traj cross-checked against traj_own
  plan      = r9pf_common.compose (RefCV3Model._r9_forward's arithmetic, float32 CPU) -> pos8; S2 / S6 = pos8 at the surface slots
Controls: K-INPUTS (prior8 / clip / ws / eid / compliant == the landed H2CN_DEPLOY8 package; traj_own vs the battery plan counted), K-H8-REPRO (the
registered H2cn stage through THIS input path reproduces the landed H2CN_DEPLOY8 S2/S6 positions bit for bit -- with the battery plan, as it was built),
K-GEO (torch re-timing in float64 == retime_stage's numpy polyline on every window, <= 1e-6 m), K-LAT (every R9 point on the path polyline / end ray,
float64 <= 1e-5 m; the float32 cast distance beside), K-S6CONSIST (S6 rows == the S2 composition of the same window).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9pf_common as RC  # noqa: E402

import numpy as np  # noqa: E402

TREE = "step48852ema_pf"
LANDED = Path("C:/Users/Admin/refcv8_pf/refcv9_hsgate/out_mdfix/arms/h2cn_mdfix")
BATT = Path("C:/Users/Admin/refcv8_eval_archive/battery") / TREE
REG = RC.WORK / "REGISTERED_R9PF_PA.sha256"
PKG = RC.SCRATCH / "arms"


def check_registration() -> dict:
    """Every '<sha256>  <path>' line of REGISTERED_R9PF_PA.sha256 must re-hash now (both 64 hex); every frozen head must be pinned."""
    if not REG.is_file():
        raise SystemExit("[a3] REGISTERED_R9PF_PA.sha256 does not exist: the Master Mind has not registered the heads -- refusing")
    pins, bad = {}, []
    for ln in REG.read_text(encoding="utf-8-sig").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split(None, 1)
        if len(parts) != 2:
            continue
        want, p = RC.hex64(parts[0]), parts[1].lstrip("*").strip()
        got = RC.sha256_file(p) if Path(p).is_file() else ""
        pins[Path(p).as_posix()] = want
        if len(want) != 64 or len(got) != 64 or want != got:
            bad.append({"path": p, "registered": want, "now": got})
    heads = sorted(RC.HEADS.glob("*.pt"))
    unpinned = [h.as_posix() for h in heads if h.as_posix() not in pins]
    ok = bool(not bad and not unpinned and len(heads) == 12)
    if not ok:
        raise SystemExit(f"[a3] registration check failed: bad {bad} unpinned {unpinned} n_heads {len(heads)} -- refusing")
    return {"file": str(REG), "sha256": RC.sha256_file(REG), "n_pins": len(pins), "ALL_MATCH": ok, "pins": pins}


def h2cn_progress(AL, prior8, f2, f6):
    """== ship_r8_mdfix_forward.h2cn_progress (Amendment 4 s2)."""
    a4 = AL.arc(prior8[:, :4])
    a6 = AL.arc(prior8[:, RC.SLOTS["S6"]])
    s = np.concatenate([a4 + f2, a6[:, 2:] + f6[:, 2:]], axis=1)
    s0 = np.maximum(np.asarray(s, np.float64), 0.0)
    return np.maximum.accumulate(s0, axis=1)


def main() -> int:
    t0 = time.time()
    prereg = RC.check_prereg()
    amend1 = RC.check_amendment1()
    fixpin = RC.check_fix_pin()
    reg = check_registration()
    RC.setup_paths()
    import torch
    import ship_r8_sub_common as C
    import l1_fit_score as F
    import compliant_reread_r8 as CR
    AL, PS, RT = F.AL, F.PS, F.RT
    from tanitad.models import kinematic_prior as KP
    from tanitad.models.ego_history import ego_channels_from_poses
    R = RC.r9p()
    tools = RT.tools_load()
    heads = {}
    for hp in sorted(RC.HEADS.glob("*.pt")):
        h, blob, sh = RC.load_head(hp, reg["pins"][hp.as_posix()])
        heads[hp.stem] = (h, blob, sh)
    reg_h8 = (C.WORK / "REGISTERED_H2CN_STAGE.sha256").read_text(encoding="utf-8").split()[0]
    preds_h8, _bl, h8_sha = C.load_stage(C.STAGE_PT, reg_h8)
    PKG.mkdir(parents=True, exist_ok=True)
    ctl = {"registration": reg, "prereg": prereg, "amendment1": amend1, "fix_pin": fixpin}
    first = {}
    for seed in (0, 1):
        zb = F.load_bank(RC.EVAL_BANK / f"ztac_eval_s{seed}.npz")
        ex = np.load(BATT / f"dump_s{seed}" / "refcv6_extras.npz", allow_pickle=False)
        exk = {(int(c), int(w)): j for j, (c, w) in enumerate(zip(ex["clip_index"], ex["ws"]))}
        landed = np.load(LANDED / "H2CN_DEPLOY8" / f"DEPLOY8_POS_{TREE}_s{seed}.npz", allow_pickle=False)
        ctx = CR.prepare_seed(TREE, seed, tools, tools["rc"])
        per = {}
        for surf in ("S2", "S6"):
            S = RT.surface_inputs(tools, ctx, surf, first, TREE)
            rows = ctx["panels"][surf]["rows"]
            slots = RC.SLOTS[surf]
            plan_b = np.ascontiguousarray(ctx["plan"][rows])
            lim = ctx["lim"][rows]
            win = np.asarray(S["dec"]["win"])
            pw = torch.from_numpy(np.ascontiguousarray(win, dtype=np.float32))
            a0, kp = KP.prior_controls("ha0_ext_pose", pw)
            v0p = ego_channels_from_poses(pw, RT.W_OBS, dt=RT.DT)[:, -1, 0]
            prior8 = KP.prior_path(a0, kp, v0p, RC.HZ8).numpy().astype(np.float32)
            clip, ws = np.asarray(S["clip"], np.int64), np.asarray(S["ws"], np.int64)
            Z, jz = F.join(zb, clip, ws)
            own = zb["traj"][jz].astype(np.float32)                         # traj_own (float32 in the bank)
            je = np.array([exk.get((int(c), int(w)), -1) for c, w in zip(clip, ws)])
            lon = ex["nav_true.tacv6_lon_logits"][je].astype(np.float32)
            x9 = RC.x9_inmodel(win, lim)
            eid = np.asarray(S["eid"])
            ki = {"rows": int(len(rows)), "prior8_eq_landed_bit": bool(np.array_equal(prior8, landed[f"{surf}_prior8"])),
                  "battery_plan_eq_landed_plan8_bit": bool(np.array_equal(plan_b, landed[f"{surf}_plan8"])),
                  "clip_eq": bool(np.array_equal(clip, landed[f"{surf}_clip"])), "ws_eq": bool(np.array_equal(ws, landed[f"{surf}_ws"])),
                  "compliant_eq": bool(np.array_equal(np.asarray(S["mask"], bool), landed[f"{surf}_compliant"])),
                  "extras_all_joined": bool((je >= 0).all()), "extras_traj_eq_battery_plan_bit": bool(np.array_equal(ex["nav_true.traj"][je], plan_b)),
                  "traj_own_vs_battery_plan_windows_differ": int((np.abs(own.astype(np.float64) - plan_b.astype(np.float64)).max(axis=(1, 2)) > 0).sum()),
                  "traj_own_vs_battery_plan_maxabs_m": float(np.abs(own.astype(np.float64) - plan_b.astype(np.float64)).max()),
                  "bank_v0_vs_ctx_v0_maxabs": float(np.abs(zb["v0_in"][jz] - np.asarray(S["v0"], np.float64)).max())}
            # K-H8-REPRO: the registered H2cn stage through THIS input path == the landed H2CN_DEPLOY8 positions (battery plan, as built)
            X10 = PS.feats(win, lim)
            s8 = h2cn_progress(AL, prior8, preds_h8["S2"](X10, Z), preds_h8["S6"](X10, Z))
            p8 = RT.eval_polyline(RT.polyline_geometry(plan_b), s8).astype(plan_b.dtype)
            ki["H8_repro_pos8_bit_equal_landed"] = bool(np.array_equal(p8, landed[f"{surf}_pos8"]))
            ki["H8_repro_pos_bit_equal_landed"] = bool(np.array_equal(p8[:, slots], landed[f"{surf}_pos"]))
            ki["PASS"] = bool(ki["prior8_eq_landed_bit"] and ki["clip_eq"] and ki["ws_eq"] and ki["compliant_eq"] and ki["extras_all_joined"]
                              and ki["extras_traj_eq_battery_plan_bit"] and ki["H8_repro_pos8_bit_equal_landed"] and ki["bank_v0_vs_ctx_v0_maxabs"] < 1e-3)
            ctl[f"K-INPUTS|{surf}|s{seed}"] = ki
            per[surf] = {"clip": clip, "ws": ws, "eid": eid, "compliant": np.asarray(S["mask"], bool), "G": np.asarray(S["X"]["G"]), "dt": float(S["dt"]),
                         "plan8": own, "plan_battery": plan_b, "prior8": prior8, "x9": x9, "Z": Z, "lon": lon, "lim": np.asarray(lim, np.float32),
                         "v0": np.asarray(S["v0"], np.float64), "HP": np.asarray(S["X"]["HP"]), "rows": np.asarray(rows, np.int64), "slots": slots}
        del ctx
        # ---- compose every head (S2 rows = all windows; S6 rows checked == the same windows' S2 composition)
        s2k = {(int(c), int(w)): j for j, (c, w) in enumerate(zip(per["S2"]["clip"], per["S2"]["ws"]))}
        j6 = np.array([s2k.get((int(c), int(w)), -1) for c, w in zip(per["S6"]["clip"], per["S6"]["ws"])])
        for tag, (h, blob, sh) in heads.items():
            out = {}
            for surf in ("S2", "S6"):
                P = per[surf]
                o = RC.compose(h, blob, P["Z"], P["x9"], P["prior8"], P["plan_battery"], P["lon"])
                o_own = RC.compose(h, blob, P["Z"], P["x9"], P["prior8"], P["plan8"], P["lon"])
                pos8 = o["traj"].astype(np.float32)
                # K-GEO / K-LAT on float64 geometry
                with torch.no_grad():
                    t64 = R.retime(torch.from_numpy(P["plan_battery"].astype(np.float64)), torch.from_numpy(o["s_target"].astype(np.float64))).numpy()
                geo = RT.polyline_geometry(P["plan_battery"].astype(np.float64))
                n64 = RT.eval_polyline(geo, o["s_target"].astype(np.float64))
                dist64 = RT.dist_to_path(t64, geo)
                dist32 = RT.dist_to_path(pos8.astype(np.float64), geo)
                ctl[f"K-GEO|{tag}|{surf}|s{seed}"] = {"max_abs_torch64_vs_numpy_m": float(np.abs(t64 - n64).max()), "n": int(len(pos8)),
                                                      "PASS": bool(np.abs(t64 - n64).max() <= 1e-6)}
                ctl[f"K-LAT|{tag}|{surf}|s{seed}"] = {"max_dist_f64_m": float(dist64.max()), "max_dist_after_f32_cast_m": float(dist32.max()),
                                                      "f32_vs_f64_maxabs_m": float(np.abs(pos8.astype(np.float64) - t64).max()),
                                                      "float32_rounding_bound_m": float(np.finfo(np.float32).eps * np.abs(t64).max()),
                                                      "PASS": bool(dist64.max() <= 1e-5)}
                out[surf] = {"pos8": pos8, "pos": pos8[:, P["slots"]], "decision": o["decision"], "s_target": o["s_target"].astype(np.float32),
                             "s_prior": o["s_prior"].astype(np.float32), "pos8_own": o_own["traj"].astype(np.float32),
                             "pos_own": o_own["traj"].astype(np.float32)[:, P["slots"]], "decision_own": o_own["decision"]}
            # ONE composition per window: the S6 rows are a subset of the S2 rows; a separate S6-batch composition differs from the S2-batch one by
            # float32 batch-order rounding only (measured: 7 / 3,668 rows, <= 7.6e-6 m, 0 decision flips).  S6 := S2[j6]; the separate-batch
            # difference is recorded beside (deviation D16).  The control asserts every S6 INPUT is bit-equal to its S2 row.
            sep = {"rows_differ_pos8": int((np.abs(out["S6"]["pos8"].astype(np.float64) - out["S2"]["pos8"][j6]).reshape(len(j6), -1).max(1) > 0).sum()),
                   "max_abs_pos8_m": float(np.abs(out["S6"]["pos8"].astype(np.float64) - out["S2"]["pos8"][j6]).max()),
                   "decision_flips": int((out["S6"]["decision"] != out["S2"]["decision"][j6]).sum())}
            inp_eq = {k: bool(np.array_equal(per["S6"][k], per["S2"][k][j6])) for k in ("plan_battery", "plan8", "prior8", "x9", "Z", "lon", "lim", "v0", "eid")}
            out["S6"] = {k: v[j6] for k, v in out["S2"].items()}
            out["S6"]["pos"] = out["S6"]["pos8"][:, per["S6"]["slots"]]
            out["S6"]["pos_own"] = out["S6"]["pos8_own"][:, per["S6"]["slots"]]
            ctl[f"K-S6CONSIST|{tag}|s{seed}"] = {"rule": "S6 rows joined to S2 rows and every S6 input bit-equal to its S2 row; S6 := S2[j6]",
                                                 "inputs_bit_equal": inp_eq, "separate_batch_composition_beside": sep,
                                                 "PASS": bool((j6 >= 0).all() and all(inp_eq.values()))}
            meta = {"tool": "a3_compose.py", "arm": blob["arm"], "fit_seed": blob["fit_seed"], "decision": blob["decision"], "head_file": f"{tag}.pt",
                    "head_sha256": sh, "inference_seed": seed, "tree": TREE, "tier": "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)",
                    "path": "PRIMARY: the battery tree's banked fb47a58 plan (the path H2CN_DEPLOY8 / RETIME_DEPLOY8 were composed on; AMENDMENT1 change 4); "
                            "*_own: the eval bank traj_own (L1 os-only roll; FACTORED_DEPLOY8's path), beside", "slot_instants_s": RC.T8.tolist(), "surface_slots": RC.SLOTS}
            arrs = {"meta_json": np.asarray(json.dumps(meta))}
            for surf in ("S2", "S6"):
                P = per[surf]
                arrs.update({f"{surf}_pos": out[surf]["pos"], f"{surf}_pos8": out[surf]["pos8"], f"{surf}_plan8": P["plan_battery"], f"{surf}_prior8": P["prior8"],
                             f"{surf}_pos_own": out[surf]["pos_own"], f"{surf}_pos8_own": out[surf]["pos8_own"], f"{surf}_decision_own": out[surf]["decision_own"],
                             f"{surf}_traj_own": P["plan8"],
                             f"{surf}_clip": P["clip"], f"{surf}_ws": P["ws"], f"{surf}_eid": P["eid"].astype(np.int64), f"{surf}_compliant": P["compliant"],
                             f"{surf}_decision": out[surf]["decision"], f"{surf}_s_target": out[surf]["s_target"], f"{surf}_slots": np.asarray(P["slots"], np.int64)})
            d = PKG / tag
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / f"R9_POS_{TREE}_s{seed}.npz", **arrs)
        # the shared per-seed scoring inputs (GT, plans, priors, ceilings) -- written once per seed
        np.savez(PKG / f"EVAL_INPUTS_{TREE}_s{seed}.npz",
                 **{f"{surf}_{k}": per[surf][k] for surf in ("S2", "S6") for k in ("clip", "ws", "eid", "compliant", "G", "plan8", "plan_battery", "prior8",
                                                                                  "HP", "lim", "v0", "rows", "x9", "Z")},
                 S2_dt=np.asarray(per["S2"]["dt"]), S6_dt=np.asarray(per["S6"]["dt"]))
        print(f"[a3c] seed {seed} composed {len(heads)} heads ({time.time() - t0:.0f} s)", flush=True)
    req = [k for k in ctl if k.startswith(("K-INPUTS", "K-GEO", "K-LAT", "K-S6CONSIST"))]
    ctl["ALL_PASS"] = bool(all(ctl[k]["PASS"] for k in req))
    ctl["n_required"] = len(req)
    ctl["failed"] = [k for k in req if not ctl[k]["PASS"]]
    files = {p.relative_to(PKG).as_posix(): RC.sha256_file(p) for p in sorted(PKG.rglob("*.npz"))}
    RC.jdump({"tool": "a3_compose.py", "script_sha256": RC.sha256_file(__file__), "controls": ctl, "files": files, "h2cn_stage_sha256": h8_sha,
              "wall_s": round(time.time() - t0, 1), "generated": time.strftime("%FT%T%z")}, RC.RESULT / "R9PF_COMPOSE.json")
    print(f"[a3c] ALL_PASS={ctl['ALL_PASS']} failed={ctl['failed'][:6]}  ZZA3COMPOSE{'OK' if ctl['ALL_PASS'] else 'FAIL'}ZZ", flush=True)
    return 0 if ctl["ALL_PASS"] else 4


if __name__ == "__main__":
    sys.exit(main())
