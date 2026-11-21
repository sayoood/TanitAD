"""preflight_l7.py -- ZERO-TRAINING readiness check of the L7 PILOT on the REAL Thor capture.

    python preflight_l7.py --train-dumps <train139_s0/dump> <train139_s1/dump> --eval-dumps <eval139_s0/dump> <eval139_s1/dump> \
        --targets-dir <dir with <bank>.npz built by l7_targets_pai.py> --gt6-dir <dir with <bank>/ep*.npz from build_gt6.py> \
        --out-json preflight_l7.json [--batches 6] [--fanceil-dir C:/Users/Admin/refcv8_pf/fanceil]

NO OPTIMIZER IS EVER CONSTRUCTED here and no checkpoint is written: the only backward passes are for the gradient-coverage assertion, and the
model's state_dict hash is asserted unchanged at the end.  What it asserts (every line is a gate; the JSON carries the numbers):
  * inventory   -- every key / shape / dtype the trainer, the target builder and the loss READ is present in the real files
  * provenance  -- train banks read as TRAIN, eval banks as EVAL (train_l7._bank_provenance); train and eval clip sets (sha12) are disjoint (K8)
  * targets     -- built targets pair with their banks, are finite where used, every builder control passed; the eval banks' own-pick /
                   oracle-in-fan / random S2 ADE reproduce the banked FANCEIL numbers (INHERITED references, read from the json)
  * masks       -- the e9 effective keep (train_l7.effective_keep) equals the capture's OWN dumped ``e9_keep`` on every row (an independent derivation)
  * controls    -- K1 const scorer = lowest emittable index; K2 oracle scorer = the masked argmin; K5 sign-flipped scorer worse than random
  * model       -- L7-E and L7-F build with the PREREG sec. 1 settings; forward + pai_loss on real batches are finite; every parameter group
                   receives a finite non-zero gradient (union over the batches); untrained evaluate() runs on the validation slice AND an eval bank
  * refusals    -- train_l7.py REFUSES an eval bank and an unknown-provenance bank (subprocess; it exits before any model exists)
  * cost        -- per-batch / per-epoch CPU time, peak working set
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import refc_scorer as RS  # noqa: E402
import train_l7 as TL  # noqa: E402
import l7_targets_pai as TP  # noqa: E402

R: dict = {"gates": {}}


def gate(name: str, ok: bool, detail=None, blocking: bool = True):
    """blocking=False is a FINDING: reported (and listed under ``findings_failed``) but it does not decide ``all_gates_pass``.  It is for
    properties of the LABELS, not of the pipeline."""
    R["gates"][name] = {"pass": bool(ok), "blocking": bool(blocking), "detail": detail}
    print(("PASS " if ok else ("FAIL " if blocking else "FINDING-FAIL ")) + name + ("" if detail is None else "  " + json.dumps(detail, default=str)[:300]), flush=True)
    return ok


def npz_header(path: str) -> dict:
    out = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            with zf.open(info.filename) as fh:
                ver = np.lib.format.read_magic(fh)
                shape, _, dt = (np.lib.format.read_array_header_1_0(fh) if ver == (1, 0) else np.lib.format.read_array_header_2_0(fh))
            out[info.filename[:-4]] = {"dtype": str(dt), "shape": list(shape)}
    return out


def state_hash(m) -> str:
    h = hashlib.sha256()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def group_of(name: str) -> str:
    p = name.split(".")
    return ".".join(p[:2]) if p[0] in ("trunk", "heads") else p[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dumps", nargs="+", required=True)
    ap.add_argument("--eval-dumps", nargs="+", required=True)
    ap.add_argument("--targets-dir", required=True)
    ap.add_argument("--gt6-dir", required=True)
    ap.add_argument("--fanceil-dir", default=None)
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    t_all = time.time()
    torch.manual_seed(0)
    names = {d: os.path.basename(os.path.dirname(d.rstrip("/\\"))) for d in a.train_dumps + a.eval_dumps}

    # ------------------------------------------------------------------ inventory
    need_fan = set(TL.BANK_KEYS) | set(TL.MASK_KEYS) | {"inference_seed", "v0", "e9_keep"}
    need_ep = {"g", "v0", "os", "ws", "eid", "clip_index"}
    need_lab = {"r7_agent_fut", "r7_agent_fut_valid", "r7_agent_fut_label", "map_drivable_u8", "map_seen", "map_label", "clip_index", "ws"}
    inv, missing = {}, []
    for d in a.train_dumps + a.eval_dumps:
        fh = npz_header(os.path.join(d, "fan_extras.npz"))
        lh = npz_header(os.path.join(d, "l7cap_labels.npz"))
        eh = npz_header(sorted(glob.glob(os.path.join(d, "ep*.npz")))[0])
        inv[names[d]] = {"fan_extras": fh, "l7cap_labels": lh, "ep000": eh, "n_ep_files": len(glob.glob(os.path.join(d, "ep*.npz"))),
                         "has_manifest": os.path.exists(os.path.join(d, "manifest.json")),
                         "has_decisions": os.path.isdir(os.path.join(d, "decisions"))}
        missing += [f"{names[d]}:fan:{k}" for k in need_fan - set(fh)]
        missing += [f"{names[d]}:ep:{k}" for k in need_ep - set(eh)]
        missing += [f"{names[d]}:labels:{k}" for k in need_lab - set(lh)]
    R["inventory"] = inv
    gate("inventory_all_read_keys_present", not missing, {"missing": missing})
    f0 = inv[names[a.train_dumps[0]]]["fan_extras"]
    gate("shapes_N149_S8", f0["anchor_traj"]["shape"][1:] == [149, 8, 2] and f0["sel_score"]["shape"][1] == 149,
         {"anchor_traj": f0["anchor_traj"], "sel_score": f0["sel_score"]})

    # ------------------------------------------------------------------ provenance + K8
    prov = {}
    for d in a.train_dumps + a.eval_dumps:
        prov[names[d]] = TL._bank_provenance(os.path.join(d, "fan_extras.npz"))
    R["provenance"] = {k: {"kind": v["kind"], "evidence": v["evidence"]} for k, v in prov.items()}
    gate("provenance_train_banks_read_as_train", all(prov[names[d]]["kind"] == "train" for d in a.train_dumps),
         {names[d]: prov[names[d]]["kind"] for d in a.train_dumps})
    gate("provenance_eval_banks_read_as_eval", all(prov[names[d]]["kind"] == "eval" for d in a.eval_dumps),
         {names[d]: prov[names[d]]["kind"] for d in a.eval_dumps})

    def clips(d):
        return [e["clip_id"] for e in json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))["episodes"]]
    tr_clips, ev_clips = [clips(d) for d in a.train_dumps], [clips(d) for d in a.eval_dumps]
    inter = set(tr_clips[0]) & set(ev_clips[0])
    gate("K8_train_eval_clip_sets_disjoint_sha12", len(inter) == 0, {"train": len(set(tr_clips[0])), "eval": len(set(ev_clips[0])), "intersection": len(inter)})
    gate("train_seeds_share_clip_order", all(c == tr_clips[0] for c in tr_clips), {"n_banks": len(tr_clips)})
    fz = [np.load(os.path.join(d, "fan_extras.npz")) for d in a.train_dumps]
    gate("train_seeds_same_windows", all(np.array_equal(z["clip_index"], fz[0]["clip_index"]) and np.array_equal(z["ws"], fz[0]["ws"]) for z in fz))
    gate("train_seeds_differ_inference_seed", len({int(z["inference_seed"][0]) for z in fz}) == len(fz), [int(z["inference_seed"][0]) for z in fz])
    del fz

    # ------------------------------------------------------------------ targets
    tpaths = {n: os.path.join(a.targets_dir, n + ".npz") for n in names.values()}
    tj = {}
    for n, p in tpaths.items():
        tj[n] = json.load(open(p[:-4] + ".json", encoding="utf-8"))
    bad = {n: [k for k, v in j["controls"].items() if v.get("pass") is False] for n, j in tj.items()}
    gate("targets_builder_controls_all_pass", all(not v for v in bad.values()), bad)
    cov = {}
    for n, p in tpaths.items():
        t = np.load(p, allow_pickle=True)
        W = len(t["cost"])
        cov[n] = {"windows": int(W), "grid8": int((t["grid"] == 8).sum()), "grid4": int((t["grid"] == 4).sum()),
                  "mask_window_frac": {k: round(float(t[k + "_mask"].any(1).mean()), 4) for k in ("comf", "ep", "nc", "ttc", "dac")},
                  "mask_cell_frac": {k: round(float(t[k + "_mask"].mean()), 4) for k in ("comf", "ep", "nc", "ttc", "dac")},
                  "cost_nonfinite": int((~np.isfinite(t["cost"])).sum()), "ade_s2_nonfinite": int((~np.isfinite(t["ade_s2"])).sum()),
                  "along_nonfinite": int((~np.isfinite(t["along"])).sum()),
                  "binary_rate_where_masked": {k: round(float(t[k][t[k + "_mask"]].mean()), 4) for k in ("comf", "nc", "ttc", "dac") if t[k + "_mask"].any()},
                  "targets_s_per_window": tj[n]["timing"]["targets_s_per_window"]}
    R["targets"] = cov
    gate("targets_finite", all(v["cost_nonfinite"] == 0 and v["ade_s2_nonfinite"] == 0 and v["along_nonfinite"] == 0 for v in cov.values()))
    gate("targets_every_head_has_labels",
         all(all(v["mask_window_frac"][k] > 0.5 for k in ("comf", "ep", "nc", "ttc", "dac")) for v in cov.values()),
         {n: v["mask_window_frac"] for n, v in cov.items()})

    # eval banks reproduce the banked FANCEIL numbers (INHERITED references, read here from the json)
    ref = {}
    if a.fanceil_dir:
        for n, d in zip(names.values(), a.train_dumps + a.eval_dumps):
            pass
    for d in a.eval_dumps:
        n = names[d]
        t = np.load(tpaths[n], allow_pickle=True)
        z = np.load(os.path.join(d, "fan_extras.npz"))
        seed = int(z["inference_seed"][0])
        ar = np.arange(len(t["ade_s2"]))
        got = {"own_pick_ade_s2": float(t["ade_s2"][ar, t["sel_idx"]].mean()), "oracle_in_fan_ade_s2": float(t["ade_s2"].min(1).mean()),
               "random_expectation_ade_s2": float(t["ade_s2"].mean())}
        row = {"seed": seed, "builder": {k: round(v, 5) for k, v in got.items()}}
        if a.fanceil_dir:
            j = json.load(open(os.path.join(a.fanceil_dir, f"FANCEIL_R8_s{seed}.json"), encoding="utf-8"))
            exp = {"own_pick_ade_s2": j["controls"]["C2_selected_ade_vs_banked_own_pick"]["fan_selected_ade_2s"],
                   "oracle_in_fan_ade_s2": j["panel_2s"]["oracle_in_fan"]["mean"], "random_expectation_ade_s2": j["panel_2s"]["random_expectation"]["mean"]}
            row["fanceil_INHERITED"] = exp
            row["abs_diff"] = {k: round(abs(got[k] - exp[k]), 5) for k in got}
            row["pass"] = bool(row["abs_diff"]["own_pick_ade_s2"] <= 2e-3 and row["abs_diff"]["oracle_in_fan_ade_s2"] <= 5e-4
                               and row["abs_diff"]["random_expectation_ade_s2"] <= 5e-4)
        ref[n] = row
    R["eval_reference_reproduction"] = ref
    if a.fanceil_dir:
        gate("eval_banks_reproduce_fanceil_own_oracle_random", all(r["pass"] for r in ref.values()), ref)

    # ------------------------------------------------------------------ the HUMAN path as a candidate: frame / unit sanity of the NC / DAC / COMF targets
    from tanitad.refs import refcv7_oracle as O
    d0 = a.train_dumps[0]
    gt_tab = TP.load_gt(d0, os.path.join(a.gt6_dir, names[d0]))
    labs = TP.load_labels(os.path.join(d0, "l7cap_labels.npz"))
    full = sorted(k for k, v in gt_tab.items() if np.isfinite(v["g8"]).all() and k in labs["amap"])
    pick = [full[i] for i in np.random.default_rng(0).choice(len(full), min(320, len(full)), replace=False)]
    st = torch.tensor(TP.SLOT_T, dtype=torch.float32)
    acc = {"comf": [], "nc": [], "ttc": [], "dac": [], "dac_shifted_6m_left": [], "dac_first_slot": [], "dac_turning": [], "dac_turning_mirrored_y": []}
    for c0 in range(0, len(pick), 8):
        ks = pick[c0:c0 + 8]
        ii = np.asarray([labs["amap"][k] for k in ks])
        g8 = torch.from_numpy(np.stack([gt_tab[k]["g8"] for k in ks]).astype(np.float32))
        v0 = torch.tensor([gt_tab[k]["v0"] for k in ks], dtype=torch.float32)
        val = labs["r7_agent_fut_valid"][ii]
        cols = np.flatnonzero(val.any(axis=(0, 1)))
        cols = cols if cols.size else np.zeros(1, np.int64)
        kw = dict(agents=torch.from_numpy(np.ascontiguousarray(labs["r7_agent_fut"][ii][:, :, cols])).permute(0, 2, 1, 3),
                  agents_valid=torch.from_numpy(np.ascontiguousarray(val[:, :, cols])).permute(0, 2, 1),
                  drivable=torch.from_numpy(labs["map_drivable_u8"][ii].astype(np.float32) / 255.0),
                  seen=torch.from_numpy(labs["map_seen"][ii].astype(bool)))
        o = O.oracle_subscores(g8[:, None], st, v0, g8, **kw)
        sh = O.oracle_subscores((g8 + torch.tensor([0.0, 6.0]))[:, None], st, v0, g8, **kw)
        lab_ok = torch.from_numpy(labs["r7_agent_fut_label"][ii].all(axis=1))
        mok = torch.from_numpy(labs["map_label"][ii].astype(bool))
        acc["comf"] += o["comf"][:, 0].tolist()
        acc["nc"] += o["nc"][:, 0][lab_ok].tolist()
        acc["ttc"] += o["ttc"][:, 0][lab_ok].tolist()
        both = o["dac_mask"][:, 0] & sh["dac_mask"][:, 0] & mok
        acc["dac"] += o["dac"][:, 0][both].tolist()
        acc["dac_shifted_6m_left"] += sh["dac"][:, 0][both].tolist()
        flip = torch.tensor([1.0, -1.0])
        mi = O.oracle_subscores((g8 * flip)[:, None], st, v0, g8, **kw)
        turn = g8[:, -1, 1].abs() > 3.0                                   # the human ends >= 3 m off its start heading line: a turning window
        tb = turn & o["dac_mask"][:, 0] & mi["dac_mask"][:, 0] & mok
        acc["dac_turning"] += o["dac"][:, 0][tb].tolist()
        acc["dac_turning_mirrored_y"] += mi["dac"][:, 0][tb].tolist()
        kin1 = O.path_kinematics(g8[:, None, :1], st[:1], v0[:, None])
        d1, m1 = O.drivable_compliance(g8[:, None, :1], kin1["heading"], kw["drivable"], kw["seen"], O.OracleConfig())
        acc["dac_first_slot"] += d1[:, 0][m1[:, 0] & mok].tolist()
    hp = {k: (round(float(np.mean(v)), 4) if v else None) for k, v in acc.items()}
    hp["n_windows"] = len(pick)
    hp["n_nc_labelled"] = len(acc["nc"])
    hp["n_dac"] = len(acc["dac"])
    R["human_path_as_candidate"] = hp
    gate("human_path_comfort_rate_ge_0.97", hp["comf"] is not None and hp["comf"] >= 0.97,
         {"comf": hp["comf"], "basis": "refc_v3_train.py comment: complete GT fails comfort 0/161; bar 0.97 is mine (UNVERIFIED bar)"})
    gate("human_path_nc_rate_ge_0.90", hp["nc"] is not None and hp["nc"] >= 0.90, {"nc": hp["nc"], "ttc": hp["ttc"], "n": hp["n_nc_labelled"],
         "bar": "mine, UNVERIFIED: a frame or unit error in the agent futures would make the human collide"})
    gate("human_path_dac_raster_orientation", hp["dac"] is not None and hp["dac"] - hp["dac_shifted_6m_left"] >= 0.15
         and hp["dac_turning"] is not None and hp["dac_turning"] - hp["dac_turning_mirrored_y"] >= 0.20,
         {"dac": hp["dac"], "same_path_6m_left": hp["dac_shifted_6m_left"], "turning_windows_gt": hp["dac_turning"],
          "turning_windows_y_mirrored": hp["dac_turning_mirrored_y"], "n_turning": len(acc["dac_turning"]),
          "bar": "mine, UNVERIFIED: a correct column order scores the real turn above its y-mirror and the path above a 6 m shift; a flipped raster inverts both"})
    gate("human_path_dac_rate_ge_0.80", hp["dac"] is not None and hp["dac"] >= 0.80,
         {"dac_8_slots": hp["dac"], "dac_first_slot_0.5s": hp["dac_first_slot"], "n": hp["n_dac"],
          "reading": "FINDING about the LABEL, not the pipeline: the human path is 'off drivable' on a large share of windows already at 0.5 s, so the "
                     "SAM3 drivable class under-covers real road (markings / crossings / unmarked); the dac head is trained against a noisy label.  "
                     "Bar 0.80 is mine and was set before the number was seen; it is NOT moved."}, blocking=False)

    # ------------------------------------------------------------------ banks + masks
    tdumps = [os.path.join(d, "fan_extras.npz") for d in a.train_dumps]
    ttg = [tpaths[names[d]] for d in a.train_dumps]
    t0 = time.time()
    fanz, tg = TL.load_bank(tdumps, ttg)
    R["load_bank_s"] = round(time.time() - t0, 2)
    W = len(fanz["ws"])
    R["train_bank"] = {"windows": int(W), "banks": len(a.train_dumps), "inference_seeds": fanz["inference_seeds"],
                       "keep_mean_candidates": round(float(fanz["keep"].sum(1).mean()), 3),
                       "reach_mean_candidates": round(float((fanz["reach_keep"] > 0.5).sum(1).mean()), 3)}
    e9d = np.concatenate([np.load(os.path.join(d, "fan_extras.npz"))["e9_keep"] for d in a.train_dumps])
    gate("e9_effective_keep_equals_capture_e9_keep_train", bool(np.array_equal(fanz["keep"], e9d)),
         {"rows": int(W), "mismatching_cells": int((fanz["keep"] != e9d).sum())})
    gate("reach_only_mask_would_differ_from_e9", bool((fanz["keep"] != (fanz["reach_keep"] > 0.5)).any()),
         {"cells_removed_by_ceiling_or_emit": int(((fanz["reach_keep"] > 0.5) & ~fanz["keep"]).sum())})
    eid = np.asarray(tg["eid"]).astype(str)
    tr, va = TL.episode_split(eid, 0.1, 20261021)
    gate("episode_split_disjoint", not (set(eid[tr]) & set(eid[va])), {"train_windows": len(tr), "val_windows": len(va),
         "train_episodes": len(set(eid[tr])), "val_episodes": len(set(eid[va]))})
    gate("val_has_both_seeds_of_each_episode", len(set(fanz["bank_of_row"][va])) == len(a.train_dumps))

    # ------------------------------------------------------------------ controls K1 / K2 / K5 on the real masks
    ii = va[:512]
    reach, ceil, emit = TL.select_masks(fanz, ii)
    k1 = RS.select(torch.zeros(len(ii), 149), reach, ceil, emit).numpy()
    k1_expect = np.argmax(fanz["keep"][ii], axis=1)
    gate("K1_const_scorer_picks_lowest_emittable_index", bool(np.array_equal(k1, k1_expect)), {"windows": len(ii)})
    cost = torch.from_numpy(np.asarray(tg["cost"][ii], np.float32))
    keep = torch.from_numpy(fanz["keep"][ii])
    p2 = RS.select(-cost, reach, ceil, emit)
    orac = cost.masked_fill(~keep, float("inf")).min(1).values
    got2 = cost[torch.arange(len(ii)), p2]
    gate("K2_oracle_scorer_equals_masked_argmin", bool(torch.allclose(got2, orac, atol=1e-6)), {"max_abs": float((got2 - orac).abs().max())})
    p5 = RS.select(cost, reach, ceil, emit)
    rnd = float(cost.masked_fill(~keep, 0).sum(1).div(keep.sum(1)).mean())
    worst = float(cost[torch.arange(len(ii)), p5].mean())
    gate("K5_sign_flipped_scorer_worse_than_random", worst > rnd, {"flipped": round(worst, 4), "random": round(rnd, 4)})

    # ------------------------------------------------------------------ model: forward + loss + grad coverage, ZERO steps
    rng = np.random.default_rng(0)
    R["models"] = {}
    for mode in ("E", "F"):
        torch.manual_seed(0)
        model = RS.L7Scorer(RS.L7Config(mode=mode))
        h0 = state_hash(model)
        seen = {n: 0.0 for n, _ in model.named_parameters()}
        head_cov = {h: 0 for h in model.cfg.heads}
        tt, tfb, finite, losses = [], [], True, []
        for b in range(a.batches + 1):
            jj = rng.choice(tr, 32, replace=False)
            t0 = time.time()
            inp, t, kp = TL.to_batch(fanz, tg, jj, mode)
            t1 = time.time()
            out = model(**inp)
            loss, lg = RS.pai_loss(out, t, kp, 0.5)
            model.zero_grad(set_to_none=True)
            loss.backward()
            t2 = time.time()
            if b == 0:
                continue                                   # warm-up batch: not timed, not counted
            tt.append(t2 - t1)
            tfb.append(t1 - t0)
            finite &= bool(torch.isfinite(loss)) and all(bool(torch.isfinite(v).all()) for v in out.values())
            finite &= all(np.isfinite(v) for k, v in lg.items() if k in ("imit", "reg", "total"))
            losses.append({k: round(float(v), 4) for k, v in lg.items()})
            for n, p in model.named_parameters():
                if p.grad is not None and bool(torch.isfinite(p.grad).all()):
                    seen[n] = max(seen[n], float(p.grad.abs().max()))
            for h in head_cov:
                m = t.get(h + "_mask")
                head_cov[h] += int((m & kp).sum()) if m is not None else int(kp.sum())
        groups = {}
        for n, v in seen.items():
            groups.setdefault(group_of(n), []).append(v)
        dead = sorted(n for n, v in seen.items() if not v > 0.0)
        zero_groups = sorted(g for g, vs in groups.items() if not all(v > 0.0 for v in vs))
        h1 = state_hash(model)
        # untrained evaluate on the validation slice and on an eval bank (finite, runs end to end)
        t0 = time.time()
        ev = TL.evaluate(model, fanz, tg, va[:256], mode)
        t_eval256 = time.time() - t0
        R["models"][mode] = {"params": RS.n_params(model), "param_tensors": len(seen), "groups": {g: len(vs) for g, vs in groups.items()},
                             "dead_param_tensors": dead, "groups_without_full_grad": zero_groups,
                             "head_supervised_cells_over_batches": head_cov, "batch_losses": losses,
                             "t_forward_backward_s_per_batch32_median": round(float(np.median(tt)), 4),
                             "t_to_batch_s_per_batch32_median": round(float(np.median(tfb)), 4),
                             "t_evaluate_256_windows_s": round(t_eval256, 3), "untrained_val256": ev,
                             "state_unchanged": bool(h0 == h1)}
        gate(f"L7-{mode}_loss_finite_on_{a.batches}_real_batches", finite, losses[0])
        gate(f"L7-{mode}_every_parameter_gets_a_nonzero_finite_grad", not dead, {"dead": dead, "groups": zero_groups})
        gate(f"L7-{mode}_zero_optimizer_steps_state_unchanged", h0 == h1, {"hash_prefix": h0[:6]})
        gate(f"L7-{mode}_untrained_evaluate_finite", all(np.isfinite(v) for v in (ev["loss"], ev["pick_cost_m"], ev["oracle_cost_m"])), ev)
        del model

    # per-epoch estimate (CPU, this box, 4 threads): train batches + the two evaluate() calls the trainer makes per epoch
    for mode in ("E", "F"):
        m = R["models"][mode]
        nb = int(np.ceil(len(tr) / 32))
        per_batch = m["t_forward_backward_s_per_batch32_median"] + m["t_to_batch_s_per_batch32_median"]
        ev_cost = (len(va) + 512) / 256 * m["t_evaluate_256_windows_s"]
        m["epoch_estimate_s"] = {"train_batches": nb, "train_s": round(nb * per_batch * 1.15, 1), "evaluate_s": round(ev_cost, 1),
                                 "total_s": round(nb * per_batch * 1.15 + ev_cost, 1),
                                 "basis": "median of measured batches x 1.15 (optimizer.step + python overhead, NOT measured: no step is taken)"}

    # ------------------------------------------------------------------ eval banks go through the same plumbing (untrained; no metric is read)
    efz, etg = TL.load_bank([os.path.join(d, "fan_extras.npz") for d in a.eval_dumps], [tpaths[names[d]] for d in a.eval_dumps])
    torch.manual_seed(0)
    mF = RS.L7Scorer(RS.L7Config(mode="F"))
    sel = rng.choice(len(efz["ws"]), 64, replace=False)
    inp, t, kp = TL.to_batch(efz, etg, sel, "F")
    with torch.no_grad():
        out = mF(**inp)
        loss, _ = RS.pai_loss(out, t, kp, 0.5)
    e9e = np.concatenate([np.load(os.path.join(d, "fan_extras.npz"))["e9_keep"] for d in a.eval_dumps])
    gate("eval_banks_forward_loss_finite_and_e9_keep_equal", bool(torch.isfinite(loss)) and bool(np.array_equal(efz["keep"], e9e)),
         {"windows": int(len(efz["ws"])), "loss": float(loss)})
    del mF

    # ------------------------------------------------------------------ refusals (subprocess; exits before a model exists)
    tmp = os.path.join(os.path.dirname(os.path.abspath(a.out_json)), "_refusal_tmp")
    os.makedirs(os.path.join(tmp, "unk", "dump"), exist_ok=True)
    open(os.path.join(tmp, "unk", "dump", "fan_extras.npz"), "wb").close()
    cases = {"eval_bank": (os.path.join(a.eval_dumps[0], "fan_extras.npz"), tpaths[names[a.eval_dumps[0]]]),
             "unknown_provenance": (os.path.join(tmp, "unk", "dump", "fan_extras.npz"), tpaths[names[a.train_dumps[0]]])}
    for cname, (fnpz, tnpz) in cases.items():
        pr = subprocess.run([sys.executable, "-I", "-X", "utf8", os.path.join(HERE, "train_l7.py"), "--fan-npz", fnpz, "--targets-npz", tnpz,
                             "--out-dir", os.path.join(tmp, "out_" + cname)], capture_output=True, text=True, timeout=300,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        txt = (pr.stderr or "") + (pr.stdout or "")
        gate(f"train_l7_refuses_{cname}", pr.returncode != 0 and "REFUSING" in txt and not os.path.exists(os.path.join(tmp, "out_" + cname, "best.pt")),
             {"rc": pr.returncode, "msg": txt.strip().splitlines()[-1][:200] if txt.strip() else ""})

    try:
        import psutil
        mi = psutil.Process().memory_info()
        R["peak_working_set_gb"] = round(mi.peak_wset / 2 ** 30, 2)
        R["peak_commit_gb"] = round(mi.peak_pagefile / 2 ** 30, 2)
    except Exception:  # noqa: BLE001
        R["peak_working_set_gb"] = None
    R["wall_s"] = round(time.time() - t_all, 1)
    R["all_gates_pass"] = bool(all(g["pass"] for g in R["gates"].values() if g["blocking"]))
    R["n_gates"] = len(R["gates"])
    R["failed"] = [k for k, g in R["gates"].items() if not g["pass"] and g["blocking"]]
    R["findings_failed"] = [k for k, g in R["gates"].items() if not g["pass"] and not g["blocking"]]
    with open(a.out_json, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    print("ZZPREFLIGHTZZ " + json.dumps({"all_gates_pass": R["all_gates_pass"], "n_gates": R["n_gates"], "failed": R["failed"], "findings_failed": R["findings_failed"],
                                         "peak_working_set_gb": R["peak_working_set_gb"], "wall_s": R["wall_s"]}))
    return 0 if R["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
