#!/usr/bin/env python3
"""ZERO-TRAINING GATE for the dev-box decoder-side proxy: WHERE does the 012 -> 013 fan-speed shift live?

QUESTION. Between snapshots 012 and 013 (the first on-policy epoch) every one of the 64 proposal slots got longer
(MEASURED from the stored E-6 tables, sub200, 4-s chord length: +2.35 m mean, per slot +1.22 .. +3.23 m, 64/64
slots). A DECODER-SIDE proxy -- the encoder side (DINOv3 trunk + LoRA + registers + pos3d + register compression +
scene_proj) frozen and cached at one snapshot, the decoders fine-tuned on the cache -- can only reproduce that shift
if the shift lives in the DECODER-side parameters. This probe measures the split with no training:

    2 x 2 factorial   {encoder 012, encoder 013} x {decoder 012, decoder 013}
    encoder side      the snapshot's full model up to the trajectory decoder's context (`scene_ctx`, captured by a
                      pre-hook on `dec[0]` during the PLANNER'S OWN `infer` call -- its input construction unchanged)
    decoder side      ego/goal encoder, queries, trajectory decoder, trajectory head -- loaded from the other snapshot
                      and run on the captured context (the lines of `REFe.forward` after `scene_proj`)

CONTROLS (gates -- no split is reported unless every one passes):
  C1  decoder S on encoder S's captured context reproduces the planner's full forward of snapshot S (bar 1e-5 m;
      the SAME kernels on the same tensors, expected exactly 0)
  C2  the full forward of snapshot S reproduces the stored E-6 table's proposals (to_navsim, bar 1e-4 m = E-6 G1)
  C3  the decoder tensors used for S are byte-identical to snapshot S's file (sha256 per tensor), and (C3b, added
      before the decision run, stricter only) the two snapshots' decoders differ -- else the swap is inert
RUN DESIGN: both snapshots loaded at once and ONE pass over the tokens (each scenario built once; both forwards on the
same inputs; the combos decoded per token), so a --deadline stop leaves complete pairs and the GPU is released before
the CPU analysis. It refuses to start with less than --min-free-gb free GPU memory (another job on the card).
READOUT: mean 4-s chord length per combo; the total shift and its encoder / decoder / interaction terms, each with a
paired log-cluster bootstrap CI over the tokens; the same for the shortest slot per token.
DECISION (pre-registered): the decoder-side proxy is admissible for the shift only if the decoder effect at the
proxy's own encoder (encoder 012) carries >= 50 % of the total shift with its CI excluding 0.

    python eval/fan_speed_swap_probe.py [--snaps 012,013] [--limit-tokens N] [--device cuda|cpu] [--deadline HH:MM]
Writes eval/raw/e6_sub200_ep015/fan_speed_swap.json (or --out). ZZSWAP_OK / ZZSWAP_FAIL.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_checkpoint as EC        # noqa: E402  the seam's interpreter + environment
import refe_navtest_seam as SEAM    # noqa: E402  to_navsim, the export, the test DBs; puts refe/ on sys.path

CHILD = "REFE_SWAP_CHILD"
SNAPDIR = "D:/Projects/TanitAD/data/refe_runs_eval"
TABLES = f"{EC.DATA}/proptable"
TOK = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/"
       "A1_sub200_tokens.json")
OUT = os.path.join(HERE, "raw", "e6_sub200_ep015", "fan_speed_swap.json")
DEC_PREFIXES = ("ego_enc.", "queries", "dec.", "traj_head.")


def chord_len(p):
    """[..., K, 2 or 3] poses (origin excluded) -> [...] cumulative chord length from the origin."""
    p = np.asarray(p, dtype=np.float64)[..., :2]
    P = np.concatenate([np.zeros(p.shape[:-2] + (1, 2)), p], -2)
    return np.linalg.norm(np.diff(P, axis=-2), axis=-1).sum(-1)


def past(deadline: str) -> bool:
    hh, mm = (int(x) for x in deadline.split(":"))
    lt = time.localtime()
    return (lt.tm_hour, lt.tm_min) >= (hh, mm)


def boot_ci(diff_tok, logs, n=10000, seed=0):
    """paired LOG-cluster bootstrap of a per-token difference -> (mean, lo, hi)"""
    rng = np.random.default_rng(seed)
    by = {}
    for i, lg in enumerate(logs):
        by.setdefault(lg, []).append(i)
    groups = [np.asarray(v) for v in by.values()]
    means = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[j] for j in pick])
        means.append(float(diff_tok[idx].mean()))
    return float(diff_tok.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def near(deadline: str, margin_s: float = 30.0) -> bool:
    """True from `margin_s` seconds BEFORE local HH:MM: no token is started that could run past it."""
    hh, mm = (int(x) for x in deadline.split(":"))
    lt = time.localtime()
    return lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec >= hh * 3600 + mm * 60 - margin_s


def sha16(t) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def gpu_stage(a) -> int:
    """ONE pass over the tokens with BOTH snapshots loaded: each scenario is built once, both full forwards run on
    the SAME inputs, and the four encoder x decoder combos are decoded right away -- so a deadline stop leaves
    complete pairs, and nothing touches the GPU after the loop."""
    import torch
    import ckpt_io
    import navtrain_scenarios as NS
    from model import REFe, REFeConfig
    from planner import REFePlanner
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization

    sA, sB = [s.strip() for s in a.snaps.split(",")]
    exp = json.load(gzip.open(a.export, "rt", encoding="utf-8"))["tokens"]
    sub = json.load(open(a.tokens, encoding="utf-8"))
    want = set(sub["tokens"] if isinstance(sub, dict) else sub)
    toks = [t for t in exp if t in want]
    by_log: dict = {}
    for t in toks:
        by_log.setdefault(exp[t]["log_name"], []).append(t)
    logs = sorted(by_log.items())
    if a.limit_tokens:
        keep, n = [], 0
        for lg, lt in logs:
            if n >= a.limit_tokens:
                break
            keep.append((lg, lt[:a.limit_tokens - n]))
            n += len(keep[-1][1])
        logs = keep
    if near(a.deadline):
        print(f"  REFUSING: already within 30 s of the deadline {a.deadline} (local {time.strftime('%H:%M:%S')})")
        print("ZZSWAP_FAIL deadline")
        return 3
    gpu = {}
    if a.device == "cuda":
        if not torch.cuda.is_available():
            print("ZZSWAP_FAIL no_cuda")
            return 3
        free, total = torch.cuda.mem_get_info()
        gpu = {"name": torch.cuda.get_device_name(0), "free_gb_at_start": round(free / 2**30, 2),
               "total_gb": round(total / 2**30, 2)}
        print(f"  GPU {gpu}", flush=True)
        if free < a.min_free_gb * 2**30:
            print(f"  REFUSING: {free / 2**30:.2f} GB free < {a.min_free_gb} GB -- another job holds the GPU")
            print("ZZSWAP_FAIL gpu_busy")
            return 3
        torch.cuda.reset_peak_memory_stats()
    t_load = time.time()
    ckA, ckB = f"{SNAPDIR}/snap_epoch{sA}.pt", f"{SNAPDIR}/snap_epoch{sB}.pt"
    planner = REFePlanner(checkpoint=ckA, images_root=a.frames, db_dir=a.db_dir, backbone="vitl16",
                          device=a.device, select="best", rule=None)
    mA, dev = planner.model, planner.device
    if a.device == "cuda" and dev != "cuda":
        print("ZZSWAP_FAIL planner_not_on_cuda")
        return 3
    # snapshot B built EXACTLY as REFePlanner builds its model (planner.py __init__), fed the planner's inputs
    mB = REFe(REFeConfig.for_backbone("vitl16")).to(dev).eval()
    metaB: dict = {}
    ckpt_io.load_for_inference(mB, ckB, map_location=dev, backbone="vitl16", meta_out=metaB)
    if bool(metaB.get("per_sample_calib", False)) != bool(planner.per_sample_calib):
        print("ZZSWAP_FAIL per_sample_calib differs between the snapshots -- one input path cannot serve both")
        return 3
    models = {sA: mA, sB: mB}
    c3, dec_sha = {}, {}
    for s, ck in ((sA, ckA), (sB, ckB)):
        part = torch.load(ck, map_location="cpu", weights_only=False)["model_partial"]
        wantd = {k: v for k, v in part.items() if k.startswith(DEC_PREFIXES)}
        live = {k: v.detach().cpu() for k, v in models[s].state_dict().items() if k.startswith(DEC_PREFIXES)}
        c3[s] = bool(set(live) == set(wantd) and all(torch.equal(live[k], wantd[k]) for k in wantd))
        dec_sha[s] = {k: sha16(v) for k, v in wantd.items()}
        del part
    c3b = any(dec_sha[sA][k] != dec_sha[sB].get(k) for k in dec_sha[sA])       # the swap must not be inert
    print(f"  loaded {sA} + {sB} in {time.time() - t_load:.0f} s; C3 {c3}; decoders differ {c3b}", flush=True)
    cap = {"on": False}

    def hook_for(s):
        def pre(_m, args):
            if cap["on"]:
                cap[s] = args[1].detach()
        return pre
    hs = [models[s].dec[0].register_forward_pre_hook(hook_for(s)) for s in (sA, sB)]

    def decode(m, ego_vec, goal, scene_ctx):
        """REFe.forward's lines after scene_proj, on a captured context (C1 pins them to the real forward)."""
        g = goal.unsqueeze(-1) * m.goal_freqs
        goal_feat = torch.cat([goal, g.sin().flatten(1), g.cos().flatten(1)], -1)
        ego_tok = m.ego_enc(torch.cat([ego_vec, goal_feat], -1)).unsqueeze(1)
        q = m.queries.expand(1, -1, -1) + ego_tok
        for blk in m.dec:
            q = blk(q, scene_ctx)
        return m.traj_head(q).view(1, m.cfg.n_proposals, m.cfg.horizon_steps, m.cfg.traj_dim)[0]
    combos = {(e, d): [] for e in (sA, sB) for d in (sA, sB)}
    full = {sA: [], sB: []}
    tok_ok, log_ok, sec = [], [], []
    c1 = {sA: 0.0, sB: 0.0}
    misses, stopped = [], None
    t0 = time.time()
    for lg, lt in logs:
        if stopped:
            break
        db = os.path.join(a.db_dir, f"{lg}.db")
        for sc in NS.build_scenarios_for_log(db, lt, history_rows=1, future_rows=80):
            if near(a.deadline):
                stopped = f"deadline {a.deadline} (-30 s) reached at {time.strftime('%H:%M:%S')} local"
                break
            ts = time.time()
            tok = sc._initial_lidar_token
            planner._scenario = sc
            planner.initialize(PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(),
                                                     mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
            ego = sc.get_ego_state_at_iteration(0)
            img = planner._image_for(ego)
            if img is None:
                misses.append(tok)
                continue
            cap["on"] = True
            trajA, _sa, _k = planner.infer(ego, img)          # snapshot A: the planner's OWN call, unchanged
            ego_vec = planner._ego_vec(ego)
            goal = planner._goal_for(ego).to(dev)
            calib = planner._calib_for(planner._log_hint) if planner.per_sample_calib else None
            with torch.no_grad():                            # snapshot B: infer's body, on the SAME inputs
                trajB, _sb = mB(img.to(dev), ego_vec, goal, calib=calib)
            cap["on"] = False
            ctx = {sA: cap.pop(sA), sB: cap.pop(sB)}
            fw = {sA: trajA, sB: trajB[0]}
            with torch.no_grad():
                for e in (sA, sB):
                    for d in (sA, sB):
                        out = decode(models[d], ego_vec, goal, ctx[e])
                        combos[(e, d)].append(out.float().cpu().numpy())
                        if e == d:
                            c1[d] = max(c1[d], float((out - fw[d]).abs().max()))
            for s in (sA, sB):
                full[s].append(fw[s].float().cpu().numpy())
            tok_ok.append(tok)
            log_ok.append(lg)
            sec.append(time.time() - ts)
            if len(tok_ok) % 20 == 0:
                print(f"    {len(tok_ok)} tokens  {np.median(sec):.2f} s/token  local {time.strftime('%H:%M:%S')}",
                      flush=True)
    for h in hs:
        h.remove()
    peak = round(torch.cuda.max_memory_allocated() / 2**30, 3) if dev == "cuda" else None
    del planner, mA, mB, models
    if dev == "cuda":
        torch.cuda.empty_cache()                             # the GPU is released before the CPU analysis
    print(f"  {len(tok_ok)} tokens in {time.time() - t0:.0f} s ({np.median(sec) if sec else float('nan'):.2f} "
          f"s/token median); misses {len(misses)}; stopped: {stopped}; GPU released {time.strftime('%H:%M:%S')}",
          flush=True)
    if not tok_ok:
        print("ZZSWAP_FAIL no_tokens")
        return 1
    c2 = {}
    for s in (sA, sB):
        tb = np.load(f"{TABLES}/sub200_ep{s}/table.npz")
        ti = {t: i for i, t in enumerate(tb["token"])}
        worst = 0.0
        for t, tr in zip(tok_ok, full[s]):
            nav = np.stack([SEAM.to_navsim(tr[j].astype(np.float64)) for j in range(64)]).astype(np.float32)
            worst = max(worst, float(np.abs(nav - tb["proposals"][ti[t]]).max()))
        c2[s] = worst
    np.savez_compressed(a.out.replace(".json", "_dump.npz"), tokens=np.asarray(tok_ok), logs=np.asarray(log_ok),
                        **{f"traj_e{e}_d{d}": np.stack(v) for (e, d), v in combos.items()})
    res = {"_label": "EXPLORATORY -- zero-training gate for the decoder-side proxy (no PDMS; plan lengths only)",
           "snaps": [sA, sB], "n_tokens": len(tok_ok), "n_asked": sum(len(v) for _, v in logs),
           "misses": misses, "device": str(dev), "gpu": gpu, "cuda_max_memory_allocated_gb": peak,
           "stopped": stopped, "deadline": a.deadline, "sec_per_token_median": float(np.median(sec)),
           "controls": {"C1_decode_equals_full_forward_m": c1, "C2_full_forward_vs_table_m": c2,
                        "C3_decoder_tensors": c3, "C3b_decoders_differ": c3b, "dec_sha256_16": dec_sha},
           "at_local": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    return analyze(a)


def analyze(a) -> int:
    res = json.load(open(a.out, encoding="utf-8"))
    z = np.load(a.out.replace(".json", "_dump.npz"))
    snaps = res["snaps"]
    e0, e1 = snaps[0], snaps[-1]
    logs = list(z["logs"])
    L = {(e, d): chord_len(z[f"traj_e{e}_d{d}"]) for e in (e0, e1) for d in (e0, e1)}        # [N, 64]
    m = {k: v.mean(1) for k, v in L.items()}                                                     # per token
    sh = {k: v.min(1) for k, v in L.items()}
    ctl = res["controls"]
    # C2's bar is E-6 G1's 1e-4 m ON THE DEVICE THE TABLES WERE MADE ON (the 4060). A CPU smoke differs by float32
    # kernel order through 24 ViT-L layers (MEASURED 1.4-1.7 mm on 2 tokens) -- a wrong input path would be metres.
    bar2 = 1e-4 if str(res.get("device", "")).startswith("cuda") else 5e-3
    gates = {"C1": all(v is not None and v <= 1e-5 for v in ctl["C1_decode_equals_full_forward_m"].values()),
             "C2": all(v <= bar2 for v in ctl["C2_full_forward_vs_table_m"].values()),
             "C3": all(ctl["C3_decoder_tensors"].values()) and bool(ctl.get("C3b_decoders_differ", True))}
    out = {"gates": gates, "C2_bar_m": bar2, "n_tokens": len(logs),
           "decision_grade": bool(len(logs) >= 150 and str(res.get("device", "")).startswith("cuda"))}
    for name, stat in (("mean_slot_len", m), ("shortest_slot_len", sh)):
        total = stat[(e1, e1)] - stat[(e0, e0)]
        dec_at_e0 = stat[(e0, e1)] - stat[(e0, e0)]
        enc_at_d0 = stat[(e1, e0)] - stat[(e0, e0)]
        dec_at_e1 = stat[(e1, e1)] - stat[(e1, e0)]
        inter = total - dec_at_e0 - enc_at_d0
        blk = {"per_combo_mean_m": {f"enc{e}_dec{d}": round(float(stat[(e, d)].mean()), 3) for (e, d) in stat}}
        for nm, dt in (("total", total), ("decoder_at_enc_" + e0, dec_at_e0), ("encoder_at_dec_" + e0, enc_at_d0),
                       ("decoder_at_enc_" + e1, dec_at_e1), ("interaction", inter)):
            mu, lo, hi = boot_ci(dt, logs)
            blk[nm] = {"mean_m": round(mu, 3), "ci95": [round(lo, 3), round(hi, 3)]}
        tt = blk["total"]["mean_m"]
        blk["decoder_share_at_enc_" + e0] = (round(blk["decoder_at_enc_" + e0]["mean_m"] / tt, 3)
                                            if abs(tt) > 1e-9 else None)
        out[name] = blk
    dm = out["mean_slot_len"]
    share = dm["decoder_share_at_enc_" + e0]
    lo = dm["decoder_at_enc_" + e0]["ci95"][0]
    out["decision"] = ("NOT DECISION-GRADE (a smoke: fewer than 150 tokens or not on the GPU) -- mechanics only"
                       if all(gates.values()) and not out["decision_grade"] else
                       "ADMISSIBLE: the decoder side carries the shift" if all(gates.values()) and share is not None
                       and share >= 0.5 and lo > 0 else
                       "NOT ADMISSIBLE for this shift: a decoder-side proxy cannot reproduce it"
                       if all(gates.values()) else "GATES FAILED -- no split reported")
    res["readout"] = out
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))
    print("ZZSWAP_OK" if all(gates.values()) else "ZZSWAP_FAIL gates " + json.dumps(gates))
    return 0 if all(gates.values()) else 1


def tables_only(a) -> int:
    """The OBSERVED shift, from the stored E-6 tables alone (no model, no GPU): per snapshot the mean 4-s chord
    length over the 64 slots and the shortest slot per token; per consecutive pair the change with a paired
    log-cluster bootstrap CI. The table stores to_navsim poses (8 x 0.5 s), so lengths are over those 8 poses."""
    exp = json.load(gzip.open(a.export, "rt", encoding="utf-8"))["tokens"]
    snaps = [s.strip() for s in a.table_snaps.split(",")]
    L, toks = {}, None
    for s in snaps:
        tb = np.load(f"{TABLES}/sub200_ep{s}/table.npz")
        if toks is None:
            toks = list(tb["token"])
        if list(tb["token"]) != toks:
            print(f"  REFUSING: sub200_ep{s} token order differs")
            return 1
        L[s] = chord_len(tb["proposals"])                                                     # [N, 64]
    logs = [exp[t]["log_name"] for t in toks]
    out = {"_label": "OBSERVED from the stored E-6 tables (sub200, to_navsim poses, 4-s chord length)",
           "n_tokens": len(toks), "per_snapshot": {}, "pairs": {}}
    for s in snaps:
        out["per_snapshot"][s] = {"mean_slot_len_m": round(float(L[s].mean()), 3),
                                  "median_shortest_slot_m": round(float(np.median(L[s].min(1))), 3),
                                  "mean_shortest_slot_m": round(float(L[s].min(1).mean()), 3)}
    for x, y in zip(snaps[:-1], snaps[1:]):
        d = L[y] - L[x]
        mu, lo, hi = boot_ci(d.mean(1), logs)
        smu, slo, shi = boot_ci(L[y].min(1) - L[x].min(1), logs)
        out["pairs"][f"{x}->{y}"] = {"mean_slot_len_change_m": [round(mu, 3), round(lo, 3), round(hi, 3)],
                                     "shortest_slot_change_m": [round(smu, 3), round(slo, 3), round(shi, 3)],
                                     "slots_longer": int((d.mean(0) > 0).sum()),
                                     "per_slot_change_range_m": [round(float(d.mean(0).min()), 3),
                                                                 round(float(d.mean(0).max()), 3)]}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snaps", default="012,013")
    ap.add_argument("--tables-only", action="store_true", help="the OBSERVED shift from the stored tables; no GPU")
    ap.add_argument("--table-snaps", default="011,012,013,014,015")
    ap.add_argument("--tokens", default=TOK)
    ap.add_argument("--frames", default=f"{EC.DATA}/frames")
    ap.add_argument("--db-dir", default=SEAM.TEST_DB_DIR)
    ap.add_argument("--export", default=SEAM.W3_EXPORT)
    ap.add_argument("--device", default="cuda", help="cuda, or cpu for a smoke run")
    ap.add_argument("--limit-tokens", type=int, default=0)
    ap.add_argument("--min-free-gb", type=float, default=4.5,
                    help="refuse to start with less free GPU memory (two fp32 ViT-L + activations)")
    ap.add_argument("--deadline", default="15:45", help="local HH:MM after which the loop stops")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    if a.tables_only:
        return tables_only(a)
    if a.analyze_only:
        return analyze(a)
    if os.environ.get(CHILD) != "1":
        env = EC.env_driverl()
        env[CHILD] = "1"
        return subprocess.call([EC.DRIVERL_PY, os.path.abspath(__file__), *sys.argv[1:]], cwd=HERE, env=env)
    return gpu_stage(a)


if __name__ == "__main__":
    sys.exit(main())
