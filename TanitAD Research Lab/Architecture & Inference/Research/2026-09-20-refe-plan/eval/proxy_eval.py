#!/usr/bin/env python3
"""M6 PROXY EVALUATION, exactly as registered in eval/PREREG_M6_YAWLOSS.md (blob de651f51).

  run       decode ONE arm (or the untouched snapshot: --arm none) on the eval cache (fp32, batch 1, the planner's
            numerics): proposals + scorer logits per token, the planner's own pick rule (navsim_v1 aggregate), the
            winner vs the human future, and the NAVSIM seams of the executed pick with the Amendment-7 repair ON and
            OFF (`planner.repair_last_heading`, the function itself)
  score     seams through eval/score_navtest_refe.py UNCHANGED (CPU; free-RAM guard), the four gating seams first
  families  the eval flywheel's families6.py on the gating seams (strategic UNAVAILABLE in NAVSIM, by design)
  analyze   gates G1-G5, the registered statistics and the verdict -> RESULT_M6_YAWLOSS.md + result_m6.json

    python eval/proxy_eval.py run --eval-caches <d1,d2> --arm <arm dir|none> --name W0 --out <root>
    python eval/proxy_eval.py score --out <root> [--workers 2] [--only refe_m6_W0_on,...]
    python eval/proxy_eval.py families --out <root>
    python eval/proxy_eval.py analyze --out <root> --train-cache <dir> --eval-caches <d1,d2> --arms-root <dir>

DEFINITIONS FIXED HERE, BEFORE ANY ARM RAN (the prereg leaves them implicit):
  * position error (the position guard) = the winner's mean EUCLIDEAN distance to the human over the 8 NAVSIM poses
    (ADE); the L1 form the winner is selected by is reported beside it;
  * "a family's CI separates adversely" = on a gating pair (P_s repair OFF vs W_s repair ON) a family metric whose
    P_s interval lies entirely on the worse side of W_s's interval (families6's own per-arm intervals);
  * the best of 64 needs every proposal scored by the harness (64x the scoring cost): reported, not gating, and run
    only after the verdict if the window allows.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PKG / "refe"))
import eval_checkpoint as EC        # noqa: E402
import refe_navtest_seam as SEAM    # noqa: E402

PREREG = HERE / "PREREG_M6_YAWLOSS.md"
PREREG_BLOB = "de651f51dc3873b11f49f7d48b00140a82614aea"
SNAP = "D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch015.pt"
TOK_W3 = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/"
          "A1_sub200_tokens.json")
TOK_A5 = str(HERE / "raw" / "a5_confirm" / "a5_confirm_tokens.json")
A7_NATIVE = "D:/Projects/TanitAD/data/refe_navtest/proptable/a7confirm_ep015/native_executed.npz"
ARMS = ("W0", "W1", "P0", "P1")


def arms_for(x: str = ""):
    """the four arms of one run: epoch 1 (x = "") or AMENDMENT 1's epoch 2 (x = "e2")"""
    return tuple(f"{n}{x}" for n in ("W0", "W1", "P0", "P1"))


def gating_for(x: str = ""):
    return (f"refe_m6_W0{x}_on", f"refe_m6_W1{x}_on", f"refe_m6_P0{x}_off", f"refe_m6_P1{x}_off")
# the harness requires labels starting with "refe" (its devkit scratch never collides with W3/refcv6)
GATING = ("refe_m6_W0_on", "refe_m6_W1_on", "refe_m6_P0_off", "refe_m6_P1_off")
SUBSCORES = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound",
             "comfort", "driving_direction_compliance")


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def git_blob(p) -> str:
    b = open(p, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


AMENDMENT_MARK = "\n---\n\n## AMENDMENT 1"


def registered_part_blob(p) -> str:
    """the git blob of the prereg text ABOVE the first appended amendment -- it must still be the registered blob"""
    s = open(p, encoding="utf-8").read()
    i = s.find(AMENDMENT_MARK)
    part = (s[:i].rstrip("\n") + "\n") if i >= 0 else s
    b = part.encode("utf-8")
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def tokens_1123():
    a, b = json.load(open(TOK_W3, encoding="utf-8")), json.load(open(TOK_A5, encoding="utf-8"))
    tok = {"rule": "M6 proxy eval (PREREG_M6_YAWLOSS): W3's 200 (A1_sub200_tokens.json) + Amendment 5's 923 "
                   "(a5_confirm_tokens.json)", "tokens": list(a["tokens"]) + list(b["tokens"]),
           "token_log": {**a["token_log"], **b["token_log"]}}
    return tok, set(a["tokens"]), set(b["tokens"])


# ================================================================================================ run
def run(a) -> int:
    import torch
    import measures as MS
    import planner as PL
    import proxy_train as PT
    dev = "cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"
    model, cfg, snap_state = PT.build(SNAP, dev)
    arm_meta = None
    if a.arm != "none":
        ck = torch.load(Path(a.arm) / "final.pt", map_location="cpu", weights_only=False)
        if not ck.get("complete"):
            print("ZZPROXYEVAL_FAIL the arm is not complete")
            return 1
        if set(ck["state"]) != set(snap_state):
            print("ZZPROXYEVAL_FAIL the arm's tensors are not the decoder side")
            return 1
        model.load_state_dict({k: v.to(dev) for k, v in ck["state"].items()}, strict=False)
        arm_meta = {k: ck.get(k) for k in ("steps_done", "yaw_loss", "slow_slots")}
        if ck.get("slow_slots"):
            MS.install_slow_slots(model, MS.SlowSlotConfig.from_meta(ck["slow_slots"]))
    model.eval()
    stub = type("RuleStub", (), {"rule": "navsim_v1", "V1_W": PL.REFePlanner.V1_W, "PDM_W": PL.REFePlanner.PDM_W})()
    exp = json.load(gzip.open(SEAM.W3_EXPORT, "rt", encoding="utf-8"))["tokens"]
    tokj, _w3, _a5 = tokens_1123()
    want = set(tokj["tokens"])
    rec = {k: [] for k in ("token", "log", "traj", "logits", "pick", "fwd_pick", "winner", "human", "fwd_traj",
                           "fwd_logits")}
    for cd in [Path(x) for x in a.eval_caches.split(",") if x]:
        mf = json.load(open(cd / "manifest.json", encoding="utf-8"))
        if mf.get("dtype") != "fp32":
            print(f"ZZPROXYEVAL_FAIL {cd} is not an fp32 eval cache")
            return 1
        frames = [json.loads(l) for l in open(cd / "frames.jsonl", encoding="utf-8")]
        rows = {}
        for l in open(cd / "rows.jsonl", encoding="utf-8"):
            r = json.loads(l)
            rows[tuple(r["key"])] = r
        n_per = {}
        for fr in frames:
            n_per[fr["shard"]] = max(n_per.get(fr["shard"], 0), fr["index"] + 1)
        mm = {s: (np.memmap(cd / f"shard_{s:04d}.scene.bin", dtype=np.float32, mode="r", shape=(n, 64, 256)),
                  np.memmap(cd / f"shard_{s:04d}.visual.bin", dtype=np.float32, mode="r", shape=(n, 7680, 256)))
              for s, n in n_per.items()}
        fwd = {s: np.load(cd / f"shard_{s:04d}.fwd.npz") for s in n_per}
        for fr in frames:
            key = tuple(fr["key"])
            if key[1] not in want:
                continue
            r = rows[key]
            sc = torch.from_numpy(np.array(mm[fr["shard"]][0][fr["index"]]))[None].to(dev)
            vi = torch.from_numpy(np.array(mm[fr["shard"]][1][fr["index"]]))[None].to(dev)
            ego = torch.tensor([r["ego"]], dtype=torch.float32, device=dev)
            goal = torch.tensor([r["goal"]], dtype=torch.float32, device=dev)
            with torch.no_grad():
                traj = PT.decode(model, sc, ego, goal)
                logits = PT.score(model, traj, vi)
                k = int(PL.REFePlanner.aggregate(stub, logits)[0].argmax())      # the planner's `_pick`, rule v1
                fl = torch.from_numpy(np.asarray(fwd[fr["shard"]]["logits"][fr["index"]]))[None]
                kf = int(PL.REFePlanner.aggregate(stub, fl)[0].argmax())
            t_ = traj[0].float().cpu().numpy()
            human = np.asarray(r["human_future_poses"], dtype=np.float64)                         # [8, 3]
            nav = np.stack([SEAM.to_navsim(t_[j].astype(np.float64)) for j in range(t_.shape[0])])  # [64, 8, 3]
            w = int(np.abs(nav[..., :2] - human[None, :, :2]).sum(-1).mean(-1).argmin())
            rec["token"].append(key[1])
            rec["log"].append(key[0])
            rec["traj"].append(t_)
            rec["logits"].append(logits[0].float().cpu().numpy())
            rec["pick"].append(k)
            rec["fwd_pick"].append(kf)
            rec["winner"].append(w)
            rec["human"].append(human.astype(np.float32))
            rec["fwd_traj"].append(fwd[fr["shard"]]["traj"][fr["index"]])
            rec["fwd_logits"].append(fwd[fr["shard"]]["logits"][fr["index"]])
    n = len(rec["token"])
    if len(set(rec["token"])) != n or (n != len(want) and not a.allow_partial):
        print(f"ZZPROXYEVAL_FAIL {n} rows ({len(set(rec['token']))} distinct) for {len(want)} eval tokens")
        return 1
    T = np.stack(rec["traj"])
    out = Path(a.out)
    (out / "dumps").mkdir(parents=True, exist_ok=True)
    (out / "seams").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "dumps" / f"{a.name}.npz", token=np.array(rec["token"]), log=np.array(rec["log"]),
                        traj=T, logits=np.stack(rec["logits"]), pick=np.array(rec["pick"]),
                        winner=np.array(rec["winner"]), human=np.stack(rec["human"]))
    d_fwd = float(np.abs(T - np.stack(rec["fwd_traj"])).max())
    d_log = float(np.abs(np.stack(rec["logits"]) - np.stack(rec["fwd_logits"])).max())
    for rep in ("on", "off"):
        poses = []
        for i in range(n):
            x = T[i, rec["pick"][i]]
            x = PL.repair_last_heading(x) if rep == "on" else x
            poses.append(SEAM.to_navsim(x).astype(np.float32))
        np.savez(out / "seams" / f"refe_m6_{a.name}_{rep}.npz", token=np.array(rec["token"]),
                 fingerprint=np.array([exp[t]["fingerprint"] for t in rec["token"]]), poses=np.stack(poses),
                 sampling=np.array([SEAM.NAVSIM_N, SEAM.NAVSIM_DT]), arm=np.array(f"refe_m6_{a.name}_{rep}"))
    json.dump(tokj, open(out / "tokens_1123.json", "w", encoding="utf-8"))
    summ = {"name": a.name, "arm": a.arm, "arm_meta": arm_meta, "n": n, "device": dev,
            "max_abs_vs_planner_forward_traj": d_fwd, "max_abs_vs_planner_forward_logits": d_log,
            "picks_equal_planner_forward_pick": f"{sum(int(x == y) for x, y in zip(rec['pick'], rec['fwd_pick']))}/{n}",
            "at_local": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(summ, open(out / "dumps" / f"{a.name}.json", "w"), indent=1)
    print(json.dumps(summ))
    print("ZZPROXYEVAL_RUN_OK")
    return 0


# ================================================================================================ score / families
def free_ram_gb() -> float:
    try:
        o = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                           capture_output=True, text=True).stdout.strip()
        return int(o) / 1024 / 1024
    except Exception:
        return 99.0


def status_of(txt):
    for line in reversed(txt.splitlines()):
        if line.startswith("{") and '"status"' in line:
            return json.loads(line)
    return None


SCORE_WORK = Path("D:/Projects/TanitAD/data/refe_proxy/m6_score")   # NO spaces: the harness's Hydra override grammar
#   cannot parse a seam path containing spaces or '&' (MEASURED 22:09: LexerNoViableAltException on
#   "TanitAD Research Lab/Architecture & Inference"); the CSVs are copied back into <out>/score afterwards


def score(a) -> int:
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    out = Path(a.out)
    work = SCORE_WORK / (out.name if out.name != "2026-09-27-m6-proxy" else "epoch1")
    (work / "seams").mkdir(parents=True, exist_ok=True)
    shutil.copy2(out / "tokens_1123.json", work / "tokens_1123.json")
    gating = gating_for(a.arm_suffix)
    labels = [x for x in a.only.split(",") if x] if a.only else \
        list(gating) + sorted(p.stem for p in (out / "seams").glob("refe_m6_*.npz") if p.stem not in gating)
    (out / "score_logs").mkdir(exist_ok=True)

    def one(lb):
        sp0 = out / "seams" / f"{lb}.npz"
        log = out / "score_logs" / f"{lb}.log"
        csvp = out / "score" / lb / f"{lb}.csv"
        if not sp0.exists():
            return lb, {"status": "NO_SEAM"}
        sp = work / "seams" / f"{lb}.npz"
        if not sp.exists() or sha256(sp) != sha256(sp0):
            shutil.copy2(sp0, sp)
        for _attempt in range(2):
            if csvp.exists() and log.exists():
                st = status_of(log.read_text(encoding="utf-8", errors="replace"))
                if st and st.get("status") == "PASS":
                    return lb, st
            while free_ram_gb() < a.min_free_ram_gb:                      # the RAM guard
                time.sleep(30)
            EC.run([EC.DRIVERL_PY, "score_navtest_refe.py", "--label", lb, "--seam", str(sp), "--tokens",
                    str(work / "tokens_1123.json"), "--out", str(work / "score")], str(HERE),
                   dict(os.environ, PYTHONIOENCODING="utf-8"), str(log))
            wcsv = work / "score" / lb / f"{lb}.csv"
            if wcsv.exists():
                csvp.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(wcsv, csvp)
        st = status_of(log.read_text(encoding="utf-8", errors="replace")) if log.exists() else None
        return lb, st
    ok = True
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for lb, st in ex.map(one, labels):
            good = bool(st and st.get("status") == "PASS")
            ok &= good
            print(f"  {lb}: {st.get('status') if st else None} rows {(st or {}).get('csv_valid_rows')} "
                  f"PDMS {((st or {}).get('summary_x100_4dp') or {}).get('PDMS')}  local {time.strftime('%H:%M:%S')}",
                  flush=True)
    print(f"ZZPROXYEVAL_SCORE_{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


def families(a) -> int:
    out = Path(a.out)
    (out / "families").mkdir(exist_ok=True)
    ok = True
    for lb in ([x for x in a.only.split(",") if x] if a.only else gating_for(a.arm_suffix)):
        fj = out / "families" / f"{lb}.json"
        if not fj.exists():
            EC.run([EC.TANITAD_PY, "families6.py", "--seam", str(out / "seams" / f"{lb}.npz"), "--inputs", EC.EXPORT,
                    "--stage", "1", "--label", lb, "--out", str(fj), "--n-boot", "2000"],
                   EC.EV6, dict(os.environ, PYTHONIOENCODING="utf-8"), str(out / "families" / f"{lb}.log"))
        ok &= fj.exists()
        print(f"  families {lb}: {'ok' if fj.exists() else 'MISSING'}  local {time.strftime('%H:%M:%S')}", flush=True)
    print(f"ZZPROXYEVAL_FAMILIES_{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


# ================================================================================================ analyze
def read_csv(p):
    rows = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r.get("token") in (None, "average"):
            continue
        rows[r["token"]] = r
    return rows


def boot(per_token: dict, token_log: dict, n=10000, seed=20260927):
    """paired log-cluster bootstrap of a per-token statistic's mean -> (mean, lo, hi). The generator is re-seeded
    identically on every call, so the same token set gets the SAME log resamples for every arm and seed."""
    toks = sorted(per_token)
    by: dict = {}
    for i, t in enumerate(toks):
        by.setdefault(token_log[t], []).append(i)
    groups = [np.asarray(v) for _k, v in sorted(by.items())]
    vals = np.asarray([per_token[t] for t in toks], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        means[b] = vals[np.concatenate([groups[j] for j in pick])].mean()
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def own_tangent19(T):
    """[..., 20, 3] -> (theta19, mask19): the backward-difference tangent of each slot's OWN path at native step 19,
    masked where the step is < 0.2 m or |theta| > 3.0 -- the numpy twin of measures.tangent_targets at t = 19"""
    d = T[..., 19, :2] - T[..., 18, :2]
    th = np.arctan2(d[..., 1], d[..., 0])
    return th, (np.linalg.norm(d, axis=-1) >= 0.2) & (np.abs(th) <= 3.0)


def heading_metrics(dump, keep=None) -> dict:
    z = np.load(dump)
    tok = np.array([str(t) for t in z["token"]])
    sel = np.ones(len(tok), bool) if keep is None else np.array([t in keep for t in tok])
    T, H, W = z["traj"][sel].astype(np.float64), z["human"][sel].astype(np.float64), z["winner"][sel]
    ar = np.arange(len(W))
    err = np.abs(wrap(T[ar, W, 19, 2] - H[:, 7, 2]))
    nav_w = np.stack([SEAM.to_navsim(T[i, W[i]]) for i in range(len(W))])                    # [n, 8, 3]
    ade = np.linalg.norm(nav_w[..., :2] - H[..., :2], axis=-1).mean(-1)
    l1 = np.abs(nav_w[..., :2] - H[..., :2]).sum(-1).mean(-1)
    th19, m19 = own_tangent19(T)
    allslot = np.abs(wrap(T[..., 19, 2] - th19))[m19]
    return {"n": int(len(W)),
            "median_allslot_h19_vs_own_tangent_rad": float(np.median(allslot)) if allslot.size else None,
            "allslot_h19_masked_in": int(m19.sum()), "allslot_h19_total": int(m19.size),
            "median_winner_h19_err_rad": float(np.median(err)),
            "pct_raw_h19_beyond_pi": 100.0 * float(np.mean(np.abs(T[..., 19, 2]) > np.pi)),
            "winner_pos_err_ade_m": float(ade.mean()), "winner_pos_err_l1_m": float(l1.mean()),
            "winner_heading_err_median_rad_by_navsim_pose_0.5_to_4.0s":
                [round(float(np.median(np.abs(wrap(nav_w[:, j, 2] - H[:, j, 2])))), 4) for j in range(8)],
            "pct_raw_heading_beyond_pi_by_native_step_0_to_19":
                [round(100.0 * float(np.mean(np.abs(T[..., j, 2]) > np.pi)), 3) for j in range(20)]}


FAM_SKIP = ("gt_progress_mean_m", "t0_axis_gt_self_ratio")          # properties of the human future, not of the arm


def fam_direction(metric: str) -> str:
    # 2026-09-28: "acc" was a SUBSTRING test meant for "accuracy"; it also matched `accel_mae_mps2`, so a LOWER
    # (better) acceleration error was read as worse -- it flipped SPEC Amendment 8 to NOT PROVEN. Match the word.
    m = metric.lower()
    if "within_" in m or "accuracy" in m or "kappa" in m:
        return "higher"
    if "bias" in m:
        return "magnitude"
    if "progress_ratio" in m:
        return "closer_to_1"
    return "lower"                                                    # mae / rmse / error / rate


def fam_adverse(fw: dict, fp: dict, arm_key: str = "refcv6") -> dict:
    """families6 blocks of W_s (repair ON) and P_s (repair OFF): every LONGITUDINAL / LATERAL component whose P_s
    interval (families6's own episode-cluster bootstrap) lies entirely on the WORSE side of W_s's interval. Unpaired
    interval non-overlap -- a conservative reading. Tactical carries no interval (accuracy / kappa reported beside)."""
    adverse, compared = [], 0
    for fam in ("longitudinal", "lateral"):
        cw = ((fw.get("families", {}).get(arm_key, {}).get(fam) or {}).get("ci") or {}).get("components", {})
        cp = ((fp.get("families", {}).get(arm_key, {}).get(fam) or {}).get("ci") or {}).get("components", {})
        for m in sorted(set(cw) & set(cp)):
            if m.rsplit(".", 1)[-1] in FAM_SKIP or not all(   # components arrive dotted (`ego_progress.gt_...`)
                    isinstance(x.get(k), (int, float)) for x in (cw[m], cp[m]) for k in ("lo", "hi")):
                continue
            compared += 1
            (wl, wh), (pl, ph) = (cw[m]["lo"], cw[m]["hi"]), (cp[m]["lo"], cp[m]["hi"])
            d = fam_direction(m)
            if d == "higher":
                bad = ph < wl
            elif d == "lower":
                bad = pl > wh
            elif d == "magnitude":                                    # both intervals on one side of 0, P farther
                bad = (pl > wh >= 0 and wl >= 0) or (ph < wl <= 0 and wh <= 0)
            else:                                                     # closer to 1
                bad = (pl > wh >= 1.0 and wl >= 1.0) or (ph < wl <= 1.0 and wh <= 1.0)
            if bad:
                adverse.append({"family": fam, "metric": m, "direction": d, "W_ci": [wl, wh], "P_ci": [pl, ph]})
    tac = {}
    for k in ("lateral_decision", "longitudinal_decision", "maneuver_5way_collapsed"):
        tw = ((fw.get("families", {}).get(arm_key, {}).get("tactical") or {}).get(k) or {})
        tp = ((fp.get("families", {}).get(arm_key, {}).get("tactical") or {}).get(k) or {})
        tac[k] = {"W": {q: tw.get(q) for q in ("accuracy", "kappa")}, "P": {q: tp.get(q) for q in ("accuracy", "kappa")}}
    return {"adverse": adverse, "components_compared": compared, "tactical_no_interval": tac,
            "strategic": "UNAVAILABLE in NAVSIM (no strategic decision scored; the arm has no strategic layer)"}


def decide(gates_ok: dict, primary_pass: dict, manip_ok: dict, have_pdms: bool, pdms_pass: bool, pdms_refute: bool,
           pos_ok: dict):
    """PREREG_M6_YAWLOSS section 6, verbatim in logic. ADOPT: every gate, PRIMARY for BOTH plain arms, the manipulation
    check for BOTH wrapped arms, PDMS (CI lower >= -1.0 and both seed floors <= 1.0) and the position guard for BOTH
    seeds. REFUTED: the gates pass AND (PRIMARY fails for BOTH plain arms, OR the PDMS CI upper bound < -1.0, OR the
    position guard fails for BOTH seeds). NOT PROVEN: everything else, with the reason named."""
    reasons = []
    if not all(gates_ok.values()):
        return "NOT PROVEN", ["gate(s) failed: " + ", ".join(g for g, ok in gates_ok.items() if not ok)]
    if (not any(primary_pass.values())) or pdms_refute or (not any(pos_ok.values())):
        if not any(primary_pass.values()):
            reasons.append("PRIMARY failed for BOTH plain arms")
        if pdms_refute:
            reasons.append("PDMS CI upper bound < -1.0")
        if not any(pos_ok.values()):
            reasons.append("position guard failed for BOTH seeds")
        return "REFUTED", reasons
    if all(primary_pass.values()) and all(manip_ok.values()) and have_pdms and pdms_pass and all(pos_ok.values()):
        return "ADOPT", reasons
    if not all(primary_pass.values()):
        reasons.append("PRIMARY split between the plain seeds")
    if not all(manip_ok.values()):
        reasons.append("a wrapped control un-trapped (manipulation check failed)")
    if not have_pdms:
        reasons.append("gating PDMS seams missing")
    elif not pdms_pass:
        reasons.append("PDMS: CI lower bound < -1.0, or a seed floor > 1.0, or invalid rows")
    if not all(pos_ok.values()):
        reasons.append("position guard split between the seeds")
    return "NOT PROVEN", reasons


def decide_m6b(gates_ok: dict, primary_pass: dict, manip_ok: dict, have_pdms: bool, mean_lo: float, mean_hi: float,
               per_seed_lo: dict, pos_ok: dict):
    """PREREG_M6B_TANGENT section 6, verbatim in logic (3 seeds). ADOPT: every gate, PRIMARY (a)(b)(c) for ALL T arms,
    manipulation for ALL Wt arms, the seed-mean D's CI lower >= -1.0 AND >= 2 of 3 per-seed CI lowers >= -1.0, and the
    position guard for ALL seeds. REFUTED: gates pass AND (PRIMARY fails for ALL T arms OR the seed-mean CI upper < -1.0
    OR the position guard fails for ALL seeds). NOT PROVEN: everything else, reason named."""
    if not all(gates_ok.values()):
        return "NOT PROVEN", ["gate(s) failed: " + ", ".join(g for g, ok in gates_ok.items() if not ok)]
    reasons = []
    if (not any(primary_pass.values())) or (have_pdms and mean_hi < -1.0) or (not any(pos_ok.values())):
        if not any(primary_pass.values()):
            reasons.append("PRIMARY failed for ALL T arms")
        if have_pdms and mean_hi < -1.0:
            reasons.append("seed-mean PDMS CI upper bound < -1.0")
        if not any(pos_ok.values()):
            reasons.append("position guard failed for ALL seeds")
        return "REFUTED", reasons
    pdms_ok = have_pdms and mean_lo >= -1.0 and sum(v >= -1.0 for v in per_seed_lo.values()) >= 2
    if all(primary_pass.values()) and all(manip_ok.values()) and pdms_ok and all(pos_ok.values()):
        return "ADOPT", reasons
    if not all(primary_pass.values()):
        reasons.append("PRIMARY split across the T seeds")
    if not all(manip_ok.values()):
        reasons.append("a wrapped control un-trapped")
    if not have_pdms:
        reasons.append("gating PDMS seams missing")
    elif not pdms_ok:
        reasons.append("PDMS: seed-mean CI lower < -1.0, or fewer than 2 of 3 per-seed CI lowers >= -1.0")
    if not all(pos_ok.values()):
        reasons.append("position guard split across the seeds")
    return "NOT PROVEN", reasons


def analyze_m6b(a) -> int:
    """PREREG_M6B_TANGENT: arms Wt0..2 (wrapped) and T0..2 (plain_tangent) under --arms-root; dumps/seams/scores in --out.
    G1-G4 as M6; G5: each seed pair differs only in --yaw-loss and --tan-w. Writes result_m6b.json + RESULT_M6B_TANGENT.md"""
    import torch
    out = Path(a.out)
    seeds = ("0", "1", "2")
    WT, TT = [f"Wt{s}" for s in seeds], [f"T{s}" for s in seeds]
    ARMS6 = WT + TT
    tokj, w3, a5 = tokens_1123()
    tl = tokj["token_log"]
    res = {"prereg": str(HERE / "PREREG_M6B_TANGENT.md"), "prereg_blob_now": git_blob(HERE / "PREREG_M6B_TANGENT.md")}
    gates = {}
    tm = json.load(open(Path(a.train_cache) / "manifest.json", encoding="utf-8"))
    v = subprocess.run([sys.executable, str(PKG / "refe" / "proxy_cache.py"), "verify", "--out", a.train_cache],
                       capture_output=True, text=True)
    c1s = [p_.get("C1_decode_vs_forward_max_abs") for p_ in tm.get("passes", [])]
    gates["G1"] = {"ok": bool(c1s) and all(x == 0.0 for x in c1s) and v.returncode == 0 and tm["dtype"] == "bf16"
                   and tm["snapshot_sha256"] == sha256(SNAP) and tm["frames"] == 10000, "verify_rc": v.returncode}
    ok2 = True
    for cd in a.eval_caches.split(","):
        em = json.load(open(Path(cd) / "manifest.json", encoding="utf-8"))
        ok2 &= (all(p_["C1_decode_vs_forward_max_abs"] == 0.0 for p_ in em["passes"]) and
                all(p_["C2_forward_vs_table_m"] <= 1e-4 for p_ in em["passes"]) and
                em["frozen_in_proxy_sha256"] == tm["frozen_in_proxy_sha256"])
    base = json.load(open(out / "dumps" / "base.json"))
    gates["G2"] = {"ok": ok2 and base["picks_equal_planner_forward_pick"] == f"{base['n']}/{base['n']}"}
    g3s = json.load(open(Path(a.arms_root) / "G3_zero_lr" / "summary.json"))
    gates["G3"] = {"ok": base["max_abs_vs_planner_forward_traj"] == 0.0 and g3s.get("tensors_changed") == 0}
    st = json.load(open(out / "selftest_measures.json", encoding="utf-8"))      # M6b's OWN record (the tested bytes)
    spt = json.load(open(out / "selftest_proxy_train.json", encoding="utf-8"))
    cfgs = {n_: json.load(open(Path(a.arms_root) / n_ / "config.json")) for n_ in ARMS6}
    t11 = [k for k in st.get("results", {}) if str(k).startswith("T11")] if isinstance(st.get("results"), dict) else []
    gates["G4"] = {"ok": not st["failed"] and not spt["failed"] and
                   all(c["code_sha256"]["measures.py"] == st["tested_sha256"]["measures.py"] for c in cfgs.values()) and
                   all(c["code_sha256"]["proxy_train.py"] == spt["tested_sha256"]["proxy_train.py"] for c in cfgs.values()),
                   "t11_records": len(t11)}

    def argv_core(c):
        ar = list(c["argv"])
        drop = set()
        for i, x in enumerate(ar):
            if x in ("--out", "--yaw-loss", "--tan-w", "--sets-work", "--deadline"):
                drop |= {i, i + 1}
        return [x for i, x in enumerate(ar) if i not in drop]
    fin = {n_: torch.load(Path(a.arms_root) / n_ / "final.pt", map_location="cpu", weights_only=False) for n_ in ARMS6}
    pairs = {s: argv_core(cfgs[f"Wt{s}"]) == argv_core(cfgs[f"T{s}"]) for s in seeds}
    gates["G5"] = {"ok": all(pairs.values()) and all(fin[n_]["complete"] and fin[n_]["steps_done"] == 453 for n_ in ARMS6)
                   and [fin[n_]["yaw_loss"] for n_ in ARMS6] == ["wrapped"] * 3 + ["plain_tangent"] * 3,
                   "argv_differs_only_in_loss": pairs}
    res["gates"] = gates
    hm = {n_: heading_metrics(out / "dumps" / f"{n_}.npz") for n_ in ARMS6}
    prim = {n_: {"a": hm[n_]["median_winner_h19_err_rad"], "b": hm[n_]["pct_raw_h19_beyond_pi"],
                 "c": hm[n_]["median_allslot_h19_vs_own_tangent_rad"]} for n_ in TT}
    primary_pass = {n_: v_["a"] <= 0.10 and v_["b"] == 0.0 and v_["c"] is not None and v_["c"] <= 0.10
                    for n_, v_ in prim.items()}
    manip_ok = {n_: hm[n_]["median_winner_h19_err_rad"] >= 0.50 for n_ in WT}
    pos = {s: hm[f"T{s}"]["winner_pos_err_ade_m"] - hm[f"Wt{s}"]["winner_pos_err_ade_m"] for s in seeds}
    pos_ok = {s: v_ <= 0.10 for s, v_ in pos.items()}
    need = [f"refe_m6_Wt{s}_on" for s in seeds] + [f"refe_m6_T{s}_off" for s in seeds]
    pd = {k: read_csv(out / "score" / k / f"{k}.csv") for k in need if (out / "score" / k / f"{k}.csv").exists()}
    toks = sorted(set(tokj["tokens"]))
    have = all(k in pd for k in need)
    pdms = {"missing": [k for k in need if k not in pd]}
    mean_lo = mean_hi = float("nan")
    per_lo = {}
    if have:
        sc_ = {k: {t: float(pd[k][t]["score"]) for t in toks} for k in need}
        Ds = {s: {t: (sc_[f"refe_m6_T{s}_off"][t] - sc_[f"refe_m6_Wt{s}_on"][t]) * 100.0 for t in toks} for s in seeds}
        Dm = {t: float(np.mean([Ds[s][t] for s in seeds])) for t in toks}
        mu, mean_lo, mean_hi = boot(Dm, tl)
        per = {s: dict(zip(("mean", "lo", "hi"), boot(Ds[s], tl))) for s in seeds}
        per_lo = {s: v_["lo"] for s, v_ in per.items()}
        m_ = lambda k: 100.0 * float(np.mean(list(sc_[k].values())))                    # noqa: E731
        pdms = {"seed_mean_D": mu, "ci95": [mean_lo, mean_hi], "per_seed": per,
                "between_seed_sd_of_D": float(np.std([per[s]["mean"] for s in seeds], ddof=1)),
                "between_seed_sd_Wt_pdms": float(np.std([m_(f"refe_m6_Wt{s}_on") for s in seeds], ddof=1)),
                "between_seed_sd_T_pdms": float(np.std([m_(f"refe_m6_T{s}_off") for s in seeds], ddof=1)),
                "means": {k: m_(k) for k in need}, "n_tokens": len(toks),
                "estimator": "paired log-cluster bootstrap, 10,000 resamples, percentile 95 %, seed 20260927"}
    verdict, reasons = decide_m6b({g: v_["ok"] for g, v_ in gates.items()}, primary_pass, manip_ok, have,
                                  mean_lo, mean_hi, per_lo, pos_ok)
    res.update({"metrics": hm, "primary": {"per_T_arm": prim, "pass": primary_pass}, "manipulation_ok": manip_ok,
                "position_guard": {s: {"T_minus_Wt_ade_m": pos[s], "ok": pos_ok[s]} for s in seeds}, "pdms": pdms,
                "verdict": verdict, "verdict_reason": reasons, "at_local": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "code_sha256_recorded_by_arms": {n_: cfgs[n_]["code_sha256"] for n_ in ARMS6}})
    json.dump(res, open(out / "result_m6b.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    L = [f"# RESULT: M6b tangent-consistency term -- **{verdict}**" + (f" ({'; '.join(reasons)})" if reasons else ""),
         "", f"Pre-registration `eval/PREREG_M6B_TANGENT.md`, blob at analysis `{res['prereg_blob_now']}`. Every number "
         "below is read from `result_m6b.json`.", "", "| gate | result |", "|---|---|"]
    L += [f"| {g} | {'PASS' if v_['ok'] else 'FAIL'} |" for g, v_ in gates.items()]
    L += ["", "| T arm | (a) winner median | (b) beyond-pi % | (c) all-slot vs own tangent | pass |", "|---|---|---|---|---|"]
    L += [f"| {n_} | {v_['a']:.4f} | {v_['b']:.4f} | {v_['c']} | {'PASS' if primary_pass[n_] else 'FAIL'} |"
          for n_, v_ in prim.items()]
    if have:
        L += ["", f"Seed-mean D = **{pdms['seed_mean_D']:+.3f} [{mean_lo:+.3f}, {mean_hi:+.3f}]**; per seed: " +
              "; ".join(f"{s}: {v_['mean']:+.3f} [{v_['lo']:+.3f}, {v_['hi']:+.3f}]" for s, v_ in pdms["per_seed"].items()) +
              f". Between-seed SD of D {pdms['between_seed_sd_of_D']:.3f} (reported).", ""]
    (out / "RESULT_M6B_TANGENT.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"verdict": verdict, "reasons": reasons, "primary": prim, "pdms": {k: v_ for k, v_ in pdms.items()
                                                                                     if k != "means"}}, default=float))
    print(f"ZZM6B_VERDICT {verdict}")
    return 0


def analyze(a) -> int:
    import torch
    X = a.arm_suffix                                   # "" = epoch 1 (the registered run); "e2" = AMENDMENT 1
    ARMS = arms_for(X)
    GATING = gating_for(X)
    out = Path(a.out)
    tokj, w3, a5 = tokens_1123()
    tl = tokj["token_log"]
    res = {"prereg": str(PREREG), "prereg_blob_registered": PREREG_BLOB, "prereg_blob_now": git_blob(PREREG),
           "prereg_registered_part_blob_now": registered_part_blob(PREREG),
           "prereg_amendments_appended": open(PREREG, encoding="utf-8").read().count("## AMENDMENT ")}
    gates = {}
    # ---- G1 train cache
    tm = json.load(open(Path(a.train_cache) / "manifest.json", encoding="utf-8"))
    v = subprocess.run([sys.executable, str(PKG / "refe" / "proxy_cache.py"), "verify", "--out", a.train_cache],
                       capture_output=True, text=True)
    c1s = [p.get("C1_decode_vs_forward_max_abs") for p in tm.get("passes", [])]
    snap_ok = tm["snapshot_sha256"] == sha256(SNAP)
    gates["G1"] = {"ok": bool(c1s) and all(x == 0.0 for x in c1s) and v.returncode == 0 and tm["dtype"] == "bf16"
                   and snap_ok and tm["frames"] == 10000,
                   "C1_per_pass": c1s, "verify_rc": v.returncode, "dtype": tm["dtype"], "frames": tm["frames"],
                   "visual_frames": tm.get("visual_frames"), "frozen_in_proxy_sha256": tm["frozen_in_proxy_sha256"],
                   "snapshot_sha256_matches": snap_ok}
    # ---- G2 eval caches
    ok2, info2 = True, {}
    for cd in a.eval_caches.split(","):
        em = json.load(open(Path(cd) / "manifest.json", encoding="utf-8"))
        ps = em.get("passes", [])
        c1 = [p["C1_decode_vs_forward_max_abs"] for p in ps]
        c2 = [p["C2_forward_vs_table_m"] for p in ps]
        good = (bool(ps) and all(x == 0.0 for x in c1) and all(x <= 1e-4 for x in c2) and
                em["frozen_in_proxy_sha256"] == tm["frozen_in_proxy_sha256"] and str(em["device"]).startswith("cuda"))
        ok2 &= good
        info2[cd] = {"C1": c1, "C2_m": c2, "same_frozen_side_as_train_cache":
                     em["frozen_in_proxy_sha256"] == tm["frozen_in_proxy_sha256"], "device": em["device"], "ok": good}
    zb = np.load(out / "dumps" / "base.npz")
    a7 = np.load(A7_NATIVE)
    ai = {str(t): i for i, t in enumerate(a7["token"])}
    bi = {str(t): i for i, t in enumerate(zb["token"])}
    worst, pick_eq, n_a5 = 0.0, 0, 0
    for t in sorted(a5):
        i, j = bi[t], ai[t]
        worst = max(worst, float(np.abs(zb["traj"][i, zb["pick"][i]] - a7["native"][j]).max()))
        pick_eq += int(int(zb["pick"][i]) == int(a7["pick"][j]))
        n_a5 += 1
    ok2 &= worst <= 1e-4 and pick_eq == n_a5
    gates["G2"] = {"ok": ok2, "caches": info2, "a5_executed_native_vs_A7_record_max_m": worst,
                   "a5_picks_equal": f"{pick_eq}/{n_a5}"}
    # ---- G3 zero-LR identity
    base_run = json.load(open(out / "dumps" / "base.json"))
    g3s = json.load(open(Path(a.arms_root) / "G3_zero_lr" / "summary.json"))
    g3e = json.load(open(out / "dumps" / "G3.json"))
    gates["G3"] = {"ok": base_run["max_abs_vs_planner_forward_traj"] == 0.0 and
                   base_run["max_abs_vs_planner_forward_logits"] == 0.0 and g3s.get("tensors_changed") == 0 and
                   g3s.get("complete") is True and g3e.get("max_abs_vs_planner_forward_traj") == 0.0 and
                   g3e.get("max_abs_vs_planner_forward_logits") == 0.0,
                   "untouched_decoders_vs_planner_forward": [base_run["max_abs_vs_planner_forward_traj"],
                                                             base_run["max_abs_vs_planner_forward_logits"]],
                   "zero_lr_run": g3s, "zero_lr_decoders_vs_planner_forward":
                       [g3e.get("max_abs_vs_planner_forward_traj"), g3e.get("max_abs_vs_planner_forward_logits")]}
    # ---- G4 code
    st = json.load(open(PKG / "raw" / "2026-09-27-training-measures" / "selftest_measures.json", encoding="utf-8"))
    spt = json.load(open(out / "selftest_proxy_train.json", encoding="utf-8"))
    cfgs = {n_: json.load(open(Path(a.arms_root) / n_ / "config.json")) for n_ in ARMS}
    ms_ok = all(c["code_sha256"]["measures.py"] == st["tested_sha256"]["measures.py"] for c in cfgs.values())
    pt_ok = all(c["code_sha256"]["proxy_train.py"] == spt["tested_sha256"]["proxy_train.py"] for c in cfgs.values())
    gates["G4"] = {"ok": not st["failed"] and st["n_checks"] >= 75 and ms_ok and not spt["failed"] and pt_ok,
                   "selftest_measures": f"{st['n_checks'] - len(st['failed'])}/{st['n_checks']}",
                   "selftest_proxy_train": f"{spt['n_checks'] - len(spt['failed'])}/{spt['n_checks']}",
                   "arms_ran_tested_measures_py": ms_ok, "arms_ran_tested_proxy_train_py": pt_ok}

    # ---- G5 arms
    def argv_core(c):
        ar = list(c["argv"])
        drop = set()
        for i, x in enumerate(ar):
            if x in ("--out", "--yaw-loss", "--sets-work", "--deadline", "--init-from"):
                drop |= {i, i + 1}
        return [x for i, x in enumerate(ar) if i not in drop]
    fin = {n_: torch.load(Path(a.arms_root) / n_ / "final.pt", map_location="cpu", weights_only=False) for n_ in ARMS}
    nan = {}
    for n_ in ARMS:
        bad = False
        for l in open(Path(a.arms_root) / n_ / "metrics.jsonl"):
            r = json.loads(l)
            bad |= not (math.isfinite(r.get("l_traj", 0.0)) and math.isfinite(r.get("l_score", 0.0)))
        bad |= not all(bool(torch.isfinite(t).all()) for t in fin[n_]["state"].values())
        nan[n_] = bad
    pairs_ok = {s: argv_core(cfgs[f"W{s}{X}"]) == argv_core(cfgs[f"P{s}{X}"]) for s in ("0", "1")}
    seeds = [cfgs[n_]["args"]["seed"] for n_ in ARMS]
    g5 = (all(pairs_ok.values()) and seeds == [0, 1, 0, 1] and not any(nan.values())
          and all(fin[n_]["complete"] and fin[n_]["steps_done"] == 453 for n_ in ARMS)
          and [fin[n_]["yaw_loss"] for n_ in ARMS] == ["wrapped", "wrapped", "plain", "plain"])
    gates["G5"] = {"ok": g5, "argv_differs_only_in_yaw_loss": pairs_ok, "seeds": seeds,
                   "steps_done": {n_: fin[n_]["steps_done"] for n_ in ARMS}, "nan_or_inf": nan,
                   "yaw_loss": {n_: fin[n_]["yaw_loss"] for n_ in ARMS}}
    res["gates"] = gates
    gates_ok = all(g["ok"] for g in gates.values())

    # ---- heading + position
    hm = {n_: heading_metrics(out / "dumps" / f"{n_}.npz") for n_ in ARMS}
    hm_sub = {sub: {n_: heading_metrics(out / "dumps" / f"{n_}.npz", keep) for n_ in ARMS}
              for sub, keep in (("W3_200", w3), ("A5_923", a5))}
    for n_ in ARMS:
        if hm[n_]["n"] != len(tokj["tokens"]):
            raise SystemExit(f"{n_}: {hm[n_]['n']} tokens, not {len(tokj['tokens'])}")
    prim = {n_: {"a_median_winner_h19_err_rad": hm[n_]["median_winner_h19_err_rad"],
                 "a_ok_le_0.10": hm[n_]["median_winner_h19_err_rad"] <= 0.10,
                 "b_pct_raw_h19_beyond_pi": hm[n_]["pct_raw_h19_beyond_pi"],
                 "b_ok_exactly_0": hm[n_]["pct_raw_h19_beyond_pi"] == 0.0} for n_ in (f"P0{X}", f"P1{X}")}
    primary_pass = {n_: v["a_ok_le_0.10"] and v["b_ok_exactly_0"] for n_, v in prim.items()}
    manip = {n_: {"median_winner_h19_err_rad": hm[n_]["median_winner_h19_err_rad"],
                  "ok_ge_0.50": hm[n_]["median_winner_h19_err_rad"] >= 0.50} for n_ in (f"W0{X}", f"W1{X}")}
    manip_ok = {n_: v["ok_ge_0.50"] for n_, v in manip.items()}
    posg = {}
    for s in ("0", "1"):
        d_ade = hm[f"P{s}{X}"]["winner_pos_err_ade_m"] - hm[f"W{s}{X}"]["winner_pos_err_ade_m"]
        posg[s] = {"P_minus_W_ade_m": d_ade,
                   "P_minus_W_l1_m": hm[f"P{s}{X}"]["winner_pos_err_l1_m"] - hm[f"W{s}{X}"]["winner_pos_err_l1_m"],
                   "ok_le_0.10": d_ade <= 0.10}
    pos_ok = {s: posg[s]["ok_le_0.10"] for s in posg}

    # ---- PDMS
    pd = {}
    for n_ in ARMS:
        for rep in ("on", "off"):
            lb = f"refe_m6_{n_}_{rep}"
            p = out / "score" / lb / f"{lb}.csv"
            if p.exists():
                pd[lb] = read_csv(p)
    toks = sorted(set(tokj["tokens"]))
    have = all(k in pd for k in GATING)
    pdms, pdms_pass, pdms_refute = {"missing": [k for k in GATING if k not in pd]}, False, False
    if have:
        valid = all(all(t in pd[k] and str(pd[k][t].get("valid", "True")) == "True" for t in toks) for k in GATING)
        sc_ = {k: {t: float(pd[k][t]["score"]) for t in toks if t in pd[k]} for k in pd}
        D = {t: float(np.mean([sc_[f"refe_m6_P{s}{X}_off"][t] - sc_[f"refe_m6_W{s}{X}_on"][t] for s in ("0", "1")])) * 100.0
             for t in toks}
        mu, lo, hi = boot(D, tl)

        def mean(k, keep=None):
            ts = [t for t in toks if (keep is None or t in keep) and t in sc_[k]]
            return 100.0 * float(np.mean([sc_[k][t] for t in ts]))
        FW, FP = abs(mean(f"refe_m6_W0{X}_on") - mean(f"refe_m6_W1{X}_on")), abs(mean(f"refe_m6_P0{X}_off") - mean(f"refe_m6_P1{X}_off"))
        per_seed = {s: dict(zip(("mean", "lo", "hi"), boot(
            {t: (sc_[f"refe_m6_P{s}{X}_off"][t] - sc_[f"refe_m6_W{s}{X}_on"][t]) * 100.0 for t in toks}, tl))) for s in ("0", "1")}
        subsets = {}
        for sub, keep in (("W3_200", w3), ("A5_923", a5)):
            subsets[sub] = dict(zip(("mean", "lo", "hi"), boot({t: D[t] for t in toks if t in keep}, tl)))
            subsets[sub]["means"] = {k: mean(k, keep) for k in pd}
        subs = {k: {c: 100.0 * float(np.mean([float(pd[k][t][c]) for t in toks if t in pd[k]]))
                    for c in SUBSCORES if c in next(iter(pd[k].values()))} for k in pd}
        pdms = {"D_mean": mu, "ci95": [lo, hi], "per_seed": per_seed, "seed_floor_W": FW, "seed_floor_P": FP,
                "all_rows_valid": valid, "means": {k: mean(k) for k in pd}, "n_tokens": len(toks),
                "n_logs": len(set(tl[t] for t in toks)), "subsets": subsets, "subscores_x100": subs,
                "estimator": "paired log-cluster bootstrap, 10,000 resamples, percentile 95 %, seed 20260927"}
        pdms_pass = lo >= -1.0 and FW <= 1.0 and FP <= 1.0 and valid
        pdms_refute = hi < -1.0

    # ---- families: reported in full; adverse separations named
    fam = {lb: (json.load(open(out / "families" / f"{lb}.json", encoding="utf-8"))
                if (out / "families" / f"{lb}.json").exists() else None) for lb in GATING}
    adverse = {s: (fam_adverse(fam[f"refe_m6_W{s}{X}_on"], fam[f"refe_m6_P{s}{X}_off"])
                   if fam[f"refe_m6_W{s}{X}_on"] and fam[f"refe_m6_P{s}{X}_off"] else {"missing": True}) for s in ("0", "1")}

    res.update({"metrics": hm, "metrics_by_subset": hm_sub, "primary": {"per_plain_arm": prim, "pass": primary_pass},
                "manipulation_check": manip, "position_guard": posg, "pdms": pdms,
                "families_adverse_separations": adverse, "families": fam})
    # ---- the registered decision (PREREG section 6)
    verdict, reasons = decide({g: v["ok"] for g, v in gates.items()}, primary_pass, manip_ok, have, pdms_pass,
                              pdms_refute, pos_ok)
    adverse_named = {s: [f"{x['family']}.{x['metric']}" for x in v.get("adverse", [])] for s, v in adverse.items()
                     if v.get("adverse")}
    if adverse_named:
        reasons.append(f"families separating adversely (named, not gating): {adverse_named}")
    res["verdict"] = verdict
    res["verdict_reason"] = reasons
    res["code_sha256"] = {f: sha256(PKG / f) for f in ("refe/proxy_train.py", "eval/proxy_eval.py", "refe/measures.py",
                                                         "refe/proxy_cache.py", "eval/proxy_eval_cache.py")}
    res["code_sha256_recorded_by_arms"] = {n_: cfgs[n_]["code_sha256"] for n_ in ARMS}   # what each arm actually ran
    res["at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if X:
        res["amendment1"] = amendment1_outcome(res, Path(a.epoch1_out), X, verdict)
        res["notes_root"] = str(Path(a.epoch1_out))
    res["result_json"] = f"result_m6{'_' + X if X else ''}.json"
    res["result_md"] = f"RESULT_M6_YAWLOSS{'_EPOCH2' if X == 'e2' else ('_' + X.upper() if X else '')}.md"
    json.dump(res, open(out / res["result_json"], "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    write_result_md(out, res)
    print(json.dumps({"verdict": verdict, "reasons": reasons, "gates": {g: v["ok"] for g, v in gates.items()},
                      "primary": prim, "manipulation": manip, "position_guard": posg,
                      "pdms": {k: v for k, v in pdms.items() if k not in ("means", "subsets", "subscores_x100")}},
                     indent=1, default=float))
    print(f"ZZM6_VERDICT {verdict}")
    return 0


CALIB_USED = "D:/Projects/TanitAD/data/refe_proxy/calib_table_pod_train_grow.json"


def amendment1_outcome(res: dict, epoch1_out: Path, X: str, verdict: str) -> dict:
    """PREREG_M6_YAWLOSS AMENDMENT 1, A1.4, verbatim in logic. PASS = section 6's ADOPT on the epoch-2 arms;
    NOT PROVEN = a gate fails, or the heading PRIMARY passes for both plain arms but another clause does not;
    otherwise FAIL BUT STILL SHRINKING when BOTH plain arms improve BOTH heading metrics by >= 25 % relative to
    epoch 1, else PLATEAU. The slope is reported per plain arm; a prediction only up to epoch 4 (2x the fit range)."""
    e1 = {}
    for n in ("W0", "W1", "P0", "P1"):
        h = heading_metrics(epoch1_out / "dumps" / f"{n}.npz")
        e1[n] = {k: h[k] for k in ("median_winner_h19_err_rad", "pct_raw_h19_beyond_pi", "winner_pos_err_ade_m")}
    e2 = {n: {k: res["metrics"][f"{n}{X}"][k] for k in ("median_winner_h19_err_rad", "pct_raw_h19_beyond_pi",
                                                         "winner_pos_err_ade_m")} for n in ("W0", "W1", "P0", "P1")}
    shrink, slope = {}, {}
    for n in ("P0", "P1"):
        rel = {}
        for k in ("median_winner_h19_err_rad", "pct_raw_h19_beyond_pi"):
            a1, a2 = e1[n][k], e2[n][k]
            r = (a2 / a1) if a1 > 0 else float("nan")
            rel[k] = {"epoch1": a1, "epoch2": a2, "ratio_e2_over_e1": r, "reduction_pct": 100 * (1 - r) if a1 > 0
                      else None, "predicted_epoch3": a2 * r if a1 > 0 else None,
                      "predicted_epoch4": a2 * r * r if a1 > 0 else None}
        slope[n] = rel
        shrink[n] = all(v["reduction_pct"] is not None and v["reduction_pct"] >= 25.0 for v in rel.values())
    gates_ok = all(g["ok"] for g in res["gates"].values())
    heading_pass = all(res["primary"]["pass"].values())
    if verdict == "ADOPT":
        outcome = "PASS -- passes after 2 proxy epochs"
    elif not gates_ok or heading_pass:
        outcome = "NOT PROVEN"
    elif all(shrink.values()):
        outcome = "FAIL BUT STILL SHRINKING"
    else:
        outcome = "PLATEAU"
    return {"outcome": outcome, "section6_verdict_on_epoch2_arms": verdict, "epoch1": e1, "epoch2": e2,
            "per_plain_arm_trend": slope, "shrinking_both_metrics_ge_25pct": shrink,
            "prediction_rule": "log-linear per epoch, reported only up to epoch 4 = 2x the fitted two-epoch range"}


def predata_deviations(out: Path, r: dict) -> list:
    """the implementation choices made BEFORE any arm ran that the prereg does not fix, each with its evidence. The
    text is fixed here before the data; every number is read from an artifact when the result is written."""
    out = Path(r.get("notes_root") or out)            # AMENDMENT 1 runs cite epoch 1's evidence files
    prereg_batch = next((l.strip() for l in open(PREREG, encoding="utf-8") if "effective batch" in l), "(not found)")
    mb = json.load(open(out / "microbatch_invariance.json")) if (out / "microbatch_invariance.json").exists() else None
    spt = json.load(open(out / "selftest_proxy_train.json")) if (out / "selftest_proxy_train.json").exists() else {}
    t1 = {k: v for k, v in spt.get("checks", {}).items() if k.startswith("T1")}
    g1 = r["gates"].get("G1", {})
    L = ["## Pre-data implementation deviations (declared; none changes the registered design)", "",
         "Each was chosen on 2026-09-27 BEFORE any arm ran; the arms' own start times are in `chain_times.json`.", "",
         f"1. **Micro-batch 64 x accum 4** (the proxy trainer's default is 32 x 8). The pre-registration says only: "
         f"\"{prereg_batch}\" Reason: the proxy is kernel-launch-latency bound while another session's job "
         f"time-slices the GPU. The two splits see the identical sample stream (the sampler does not depend on the "
         f"split) and the loss is split-invariant. PROOF (`refe/check_microbatch_invariance.py` -> "
         f"`microbatch_invariance.json`): one real 256-sample batch, the real trainer, CPU fp32 -- "
         + (f"max per-tensor relative gradient difference 32x8 vs 64x4 = {mb['clean_32x8_vs_64x4']['max_rel_per_tensor']:.2e} "
            f"(global relative L2 {mb['clean_32x8_vs_64x4']['global_rel_l2']:.2e}, bar {mb['bar_max_rel_per_tensor']:.0e}) "
            f"with covered samples per micro-batch {mb['covered_per_micro_batch']['clean_32x8']} vs "
            f"{mb['covered_per_micro_batch']['clean_64x4']} and cov_norm a GLOBAL constant "
            f"({mb['cov_norm_global_constant']:.4f}); the MUTATION (the scorer loss normalised per micro-batch) "
            f"disagrees by {mb['mutation_per_microbatch_norm_32x8_vs_64x4']['max_rel_scorer_tensors']:.2f} "
            f"(goes RED: {mb['mutation_goes_red']})." if mb else "PROOF FILE MISSING."),
         f"2. **The scorer loss is computed on the covered samples only** -- the live masked formula (train.py: "
         f"`(per x cand_m).sum() / (cov_norm x batch) x score_w`); an uncovered sample's all-zero set under mask 0 and "
         f"the proposals' own scores never enter the on-policy loss. Equivalence: `refe/selftest_proxy_train.py` T1 "
         f"({'; '.join(k + ': ' + str(v.get('detail')) for k, v in t1.items())}).",
         f"3. **The train cache stores the scorer's visual context only for frames carrying an on-policy set at any "
         f"rank** ({g1.get('visual_frames')} of {g1.get('frames')} frames); the proxy reads it only for a labelled "
         f"sample and refuses a labelled sample without it (selftest T3).",
         f"4. **Camera calibration: the live trainer's own table** (the pod's `train_grow/calib_table.json`, sha256 "
         f"`{sha256(CALIB_USED)[:16]}...` = phase A's DONE digest); the locally built table did not cover every "
         f"manifest log.",
         "5. **Definitions fixed in `eval/proxy_eval.py` before the data**: position error = the winner's ADE over the "
         "8 NAVSIM poses (L1 reported beside); a family 'separates adversely' when P_s's interval lies entirely on the "
         "worse side of W_s's (families6's own intervals); the best of 64 is not computed (64x the scoring cost).",
         "6. **Harness labels prefixed `refe_m6_`** -- `score_navtest_refe.py` refuses labels that do not start with "
         "'refe'.", ""]
    fc = json.load(open(out / "fast_copy.json")) if (out / "fast_copy.json").exists() else None
    stopped = sorted(x.name for x in (out / "arms").glob("*_stopped_*")) if (out / "arms").exists() else []
    L += ["## Implementation notes made during the run (before any arm COMPLETED)", ""]
    if fc and "path" in fc:
        rec = fc.get("files_manifest_source_copy_sha256", [])
        n3 = sum(1 for x in rec if x[1] and x[2] != "already-present" and x[1] == x[2] == x[3])
        npres = sum(1 for x in rec if x[2] == "already-present")
        nf = fc.get("first_copy_version_had_no_floor_check") or {}
        L.append(f"7. **The arms read a BIT-IDENTICAL copy of the train cache on the internal NVMe** (`{fc['path']}`; "
                 f"source `{fc['source']}`). Reason: D: (a USB SSD) reads at ~40 MB/s (MEASURED: both 14.2 GB hash "
                 f"passes ~352 s; the first W0 attempt read D: at 37.6 MB/s, disk 102 % busy) and the arms' visual "
                 f"working set exceeds RAM. Its history, every hash in `fast_copy.json`:")
        L.append(f"   - FIRST COPY: {len(rec)} files verified before any arm read it -- {n3} copied with source bytes == "
                 f"the manifest's sha256 == the copy read back, {npres} re-hashed against the manifest, frames/rows/"
                 f"manifest by source-vs-copy hash; then `proxy_cache.py verify`. C: free "
                 + (f"{nf.get('c_free_gib_before_any_copy')} GiB before any copy; the first "
                    f"{nf.get('files_copied_by_first_version')} files were copied by a chain version WITHOUT the floor "
                    f"check, replaced by the floor-guarded one (floor 20 GiB), which re-hashed them and measured "
                    if nf else "") + f"{fc['c_free_gib_before']} -> {fc['c_free_gib_after']} GiB.")
        ab = fc.get("abandoned")
        if ab:
            L.append(f"   - ABANDONED {ab['at']}: {ab['reason']}. Action: {ab['action']}; C: free "
                     f"{ab['c_free_gib_before_delete']} -> {ab['c_free_gib_after_delete']} GiB; {ab['arms_after']}.")
        for k in sorted(x for x in fc if x.startswith("copy_")):
            c = fc[k]
            r2 = c.get("files_manifest_source_copy_sha256", [])
            m3 = sum(1 for x in r2 if x[1] and x[2] != "already-present" and x[1] == x[2] == x[3])
            what = (f"verified {c.get('verified_at')}, for {c.get('for')}; authorised: {c['authorised']}"
                    if c.get("authorised") else          # a copy that records its own authorisation is rendered from it
                    f"{k[5:7]}:{k[7:9]}:{k[9:11]}, the PI's 21:18 decision 'accelerate', relayed; the coordinator "
                    f"lowered the floor for this copy to 10 GiB and required an emergency guard: C: < 8 GiB deletes "
                    f"the copy at once")
            L.append(f"   - RE-COPY ({what}): {len(r2)} files, {m3} copied three-way verified, "
                     f"{c['bytes_copied'] / 1e9:.2f} GB in {c['seconds']:.0f} s, C: free {c['c_free_gib_before']} -> "
                     f"{c['c_free_gib_after']} GiB.")
        for k in sorted(x for x in fc if x.startswith("third_copy_started_in_error_")):
            L.append(f"   - A COPY STARTED IN ERROR ({k[-6:-4]}:{k[-4:-2]}:{k[-2:]}): {fc[k].get('what')}.")
        for k in sorted(x for x in fc if x.startswith("aborted_")):
            L.append(f"   - ABORTED/EMERGENCY at {k[8:10]}:{k[10:12]}:{k[12:14]}: {fc[k].get('why')}; C: free "
                     f"{fc[k].get('c_free_gib_before_delete')} -> {fc[k].get('c_free_gib_after_delete')} GiB.")
        if fc.get("deleted_at"):
            L.append(f"   - DELETED after its last reader at {fc['deleted_at']}: C: free {fc['c_free_gib_before_delete']} "
                     f"-> {fc['c_free_gib_after_delete']} GiB; still exists: {fc.get('still_exists')}.")
    elif fc:
        L.append(f"7. The NVMe copy was ABORTED ({fc.get('aborted')}); the arms read D:.")
    if stopped:
        L.append(f"8. **Arm attempts stopped and set aside, never reused**: {', '.join(stopped)} -- each stopped when "
                 f"the arms' cache location changed (D: -> NVMe copy, or a chain replacement); every reported arm ran "
                 f"from step 0 in one uninterrupted process (its `config.json` records the cache it read).")
    L += ["", "## Process note -- a stale disk figure (reported per the coordinator's request)", "",
          "A 20:26 note quoted C: free = 71.4 GB. That figure was MEASURED at 17:15 (PowerShell Get-PSDrive and df "
          "agreed) and quoted three hours later without re-measuring; at 20:29 C: had 36.3 GiB free. Not a wrong "
          "drive -- a stale reading presented as current (the df / cgroup 'a probe outside its scope' family, with "
          "time as the scope). The ~35 GB drop predates the NVMe copy (its directory did not exist yet). A plausible, "
          "UNPROVEN contributor: pagefile.sys (25.44 GB, last resized 17:36:35) grew right after this run's 17:21-17:32 "
          "GPU smoke drove VRAM to 7.9 GB under the CUDA sysmem-fallback policy; its size before is not recorded.", ""]
    return L


def write_result_md(out: Path, r: dict) -> None:
    """RESULT_M6_YAWLOSS.md -- every number read from result_m6.json (nothing typed by hand)"""
    g = r["gates"]
    same = r["prereg_blob_now"] == r["prereg_blob_registered"]
    same_part = r.get("prereg_registered_part_blob_now") == r["prereg_blob_registered"]
    L = ["# RESULT: M6 `--yaw-loss plain` -- decoder-side proxy from snapshot 015", "",
         f"**Verdict: {r['verdict']}**" + (f" -- {'; '.join(r['verdict_reason'])}" if r["verdict_reason"] else ""), "",
         f"Pre-registration `eval/PREREG_M6_YAWLOSS.md`: registered blob `{r['prereg_blob_registered']}`; blob at "
         f"analysis `{r['prereg_blob_now']}` ("
         + ("UNCHANGED" if same else
            (f"the registered text is UNCHANGED -- its part above the appended amendment(s) still hashes to "
             f"`{r['prereg_registered_part_blob_now']}`; {r.get('prereg_amendments_appended')} amendment(s) appended "
             f"below it" if same_part else "CHANGED -- the registered text itself differs")) + ").",
         "Generated by `eval/proxy_eval.py analyze` from `result_m6.json`; no number here is typed by hand.", "",
         "## Gates (all must pass before any statistic is read)", "", "| gate | result | detail |", "|---|---|---|"]
    for k, v in g.items():
        det = {kk: vv for kk, vv in v.items() if kk not in ("ok", "caches")}
        L.append(f"| {k} | {'PASS' if v['ok'] else 'FAIL'} | `{json.dumps(det, default=float)[:600]}` |")
    L += ["", "## PRIMARY (each plain arm) and the manipulation check (each wrapped arm)", "",
          "| arm | median winner step-19 heading error (rad) | raw step-19 \\|h\\| > pi (%) | criterion | result |",
          "|---|---|---|---|---|"]
    for n_, v in r["primary"]["per_plain_arm"].items():
        L.append(f"| {n_} | {v['a_median_winner_h19_err_rad']:.4f} | {v['b_pct_raw_h19_beyond_pi']:.4f} | "
                 f"(a) <= 0.10 and (b) == 0.0 | {'PASS' if r['primary']['pass'][n_] else 'FAIL'} |")
    for n_, v in r["manipulation_check"].items():
        L.append(f"| {n_} (control) | {v['median_winner_h19_err_rad']:.4f} | "
                 f"{r['metrics'][n_]['pct_raw_h19_beyond_pi']:.4f} | stays >= 0.50 | "
                 f"{'PASS' if v['ok_ge_0.50'] else 'FAIL'} |")
    L += ["", "## Position guard (the winner over the 8 NAVSIM poses)", "",
          "| seed | P - W, ADE (m) | P - W, L1 (m) | ADE <= +0.10 |", "|---|---|---|---|"]
    for s, v in r["position_guard"].items():
        L.append(f"| {s} | {v['P_minus_W_ade_m']:+.4f} | {v['P_minus_W_l1_m']:+.4f} | "
                 f"{'PASS' if v['ok_le_0.10'] else 'FAIL'} |")
    p = r["pdms"]
    L += ["", "## PDMS non-inferiority (margin 1.0)", ""]
    if "D_mean" in p:
        L += [f"D = mean over the two seeds of PDMS(P_s, repair OFF) - PDMS(W_s, repair ON): **{p['D_mean']:+.3f} "
              f"[{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]** on {p['n_tokens']} tokens / {p['n_logs']} logs "
              f"({p['estimator']}).", "",
              f"Seed floors: F_W = {p['seed_floor_W']:.3f}, F_P = {p['seed_floor_P']:.3f} (each must be <= 1.0). "
              f"All rows valid: {p['all_rows_valid']}.", "",
              "Per seed: " + "; ".join(f"seed {s} {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]"
                                        for s, v in p["per_seed"].items()) + ".",
              "By subset: " + "; ".join(f"{s} {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]"
                                         for s, v in p["subsets"].items()) + ".", "",
              "| seam | PDMS x100 | " + " | ".join(SUBSCORES) + " |", "|---|---|" + "---|" * len(SUBSCORES)]
        for k in sorted(p["means"]):
            ss = p["subscores_x100"].get(k, {})
            L.append(f"| {k} | {p['means'][k]:.3f} | " + " | ".join(f"{ss.get(c, float('nan')):.2f}"
                                                                    for c in SUBSCORES) + " |")
    else:
        L.append(f"Gating seams missing: {p.get('missing')}")
    L += ["", "## Four families (families6 on the gating seams; strategic UNAVAILABLE in NAVSIM by design)", "",
          f"Adverse separations (a P_s interval entirely on the worse side of W_s's): "
          f"`{json.dumps(r['families_adverse_separations'], default=float)[:2000]}`", "",
          "Full blocks: `result_m6.json` -> `families`.", "", "## Heading profile (reported, not gating)", ""]
    for n_, v in r["metrics"].items():
        L.append(f"- {n_}: winner heading error, median by NAVSIM pose 0.5..4.0 s: "
                 f"{v['winner_heading_err_median_rad_by_navsim_pose_0.5_to_4.0s']}; raw |h| > pi (%) by native step "
                 f"0..19: {v['pct_raw_heading_beyond_pi_by_native_step_0_to_19']}")
    L += ["", "Subsets (W3's 200 / Amendment 5's 923): `result_m6.json` -> `metrics_by_subset`, `pdms.subsets`.", "",
          *predata_deviations(out, r),
          "## Limits (stated in the pre-registration before the data)", "",
          "The encoder side is frozen (54 % of the live-trainable parameters); the clip sees decoder-side gradients "
          "only; Adam moments are re-estimated; heavy repetition (10,000 frames about 10 times in 403 steps); the "
          "scorer trains only on sets labelled from the trapped live model (biases the PDMS clause AGAINST "
          "adoption); absolute proxy PDMS is not comparable with live snapshots -- only the paired contrasts are "
          "read. The best of 64 is not computed here (it needs every proposal scored by the harness).", "",
          f"Code on disk at analysis time (NOT what the arms ran): `{json.dumps(r['code_sha256'])}`", "",
          "What each arm ran (its own `config.json` code_sha256; G4 checks it against the selftest record): " +
          "; ".join(f"{n_}: proxy_train.py {v_['proxy_train.py'][:8]}, measures.py {v_['measures.py'][:8]}"
                    for n_, v_ in sorted(r.get("code_sha256_recorded_by_arms", {}).items())) + ".", "",
          f"Analysed {r['at_local']} (Europe/Berlin)."]
    if r.get("amendment1"):
        A = r["amendment1"]
        head = [f"# AMENDMENT 1 OUTCOME (epoch 2): **{A['outcome']}**", "",
                f"Section 6 applied to the epoch-2 arms gives {A['section6_verdict_on_epoch2_arms']}; A1.4 maps it as "
                f"above. Epoch-1 -> epoch-2 trend (plain arms; >= 25 % reduction on BOTH metrics for BOTH arms = "
                f"still shrinking):", "", "| arm | metric | epoch 1 | epoch 2 | reduction | predicted epoch 3 | "
                "predicted epoch 4 |", "|---|---|---|---|---|---|---|"]
        for n, rel in A["per_plain_arm_trend"].items():
            for k, v in rel.items():
                head.append(f"| {n} | {k} | {v['epoch1']:.4f} | {v['epoch2']:.4f} | "
                            f"{v['reduction_pct']:.1f} % | {v['predicted_epoch3']:.4f} | {v['predicted_epoch4']:.4f} |"
                            if v["reduction_pct"] is not None else f"| {n} | {k} | {v['epoch1']} | {v['epoch2']} | n/a "
                            f"| n/a | n/a |")
        head += ["", f"Predictions: {A['prediction_rule']}.", "", "---", ""]
        L = head + L
    (out / r.get("result_md", "RESULT_M6_YAWLOSS.md")).write_text("\n".join(L) + "\n", encoding="utf-8",
                                                                   newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "score", "families", "analyze", "analyze-m6b"))
    ap.add_argument("--eval-caches", default="")
    ap.add_argument("--arm", default="none")
    ap.add_argument("--name", default="base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-partial", action="store_true", help="SMOKE TESTS ONLY: fewer tokens than the 1,123")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--min-free-ram-gb", type=float, default=6.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--train-cache", default=None)
    ap.add_argument("--arms-root", default=None)
    ap.add_argument("--arm-suffix", default="", help="AMENDMENT 1: 'e2' analyses W0e2/W1e2/P0e2/P1e2")
    ap.add_argument("--epoch1-out", default=None, help="AMENDMENT 1: epoch 1's out dir (trend + evidence notes)")
    a = ap.parse_args()
    if a.cmd == "run" and os.environ.get("REFE_PROXYEVAL_CHILD") != "1":
        env = EC.env_driverl()
        env["REFE_PROXYEVAL_CHILD"] = "1"
        env["PYTHONPATH"] = env.get("PYTHONPATH", "") + os.pathsep + str(PKG / "refe")
        return subprocess.call([EC.DRIVERL_PY, os.path.abspath(__file__), *sys.argv[1:]], cwd=str(HERE), env=env)
    return {"run": run, "score": score, "families": families, "analyze": analyze,
            "analyze-m6b": analyze_m6b}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
