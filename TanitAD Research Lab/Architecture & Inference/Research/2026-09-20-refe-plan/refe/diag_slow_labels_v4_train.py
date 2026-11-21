#!/usr/bin/env python3
"""Measure 5 (label_version 4) REACHES THE RUNNING TRAINER'S LOSS -- the reachability gate. Dev box.

Every arm goes through the REAL trainer code: train.OnPolicyBank / train.TargetBank in-process, train.py as a
subprocess (ViT-S, the trainer's own --synthetic images, --weights none), with sets assembled by the REAL v4 code
(`onpolicy_label_v4.plan_slow / copy_traj / serve_set / _line`). Only the LABEL VALUES are synthetic -- a known rule of
each slot's OWN trajectory -- because the question here is plumbing and learnability, not the teacher.

  R-LOAD    v3 and v4 lines for the same 6 samples and ckpt_step: the bank keeps the v4 set for every key (label
            version 4 supersedes 3), 0 incomplete, NAVSIM drivable area on every set, and serves exactly the assembled
            set -- copies in their slots, their components in `cg`, mask all ones (TargetBank.__getitem__).
  M-APPEND  the same copies APPENDED (a 64+17 set, label_version 5): refused as incomplete (train.py:447-450); the
            key keeps its v4 set, so appended copies NEVER reach the loss -- why the design replaces K slots.
  M-EXTRA   an extra 64-set of copies under the same key (label_version 5): it REPLACES the own-proposal set -- in the
            running trainer an extra set is a replacement, not an addition (train.py:461-470).
  M-NODAC   one copy's target without `navsim_dac.violation`: the set silently loses NAVSIM's drivable area
            (has_nd False, train.py:457-460) -- so the labeller must write it on every copy.
  L-V4 / L-MUT / L-V3  the real trainer on sets whose collision label is `mean speed > 10 m/s` of the slot's OWN
            trajectory; originals 8-15 m/s (slow plans ABSENT from the originals, as on the pod after the fan sped up).
            L-MUT = the copy slots carry their SOURCE's labels (the 'label the original' wiring bug); L-V3 = no copies.
            Then the scorer's P(no collision) on NEW slow plans (2-7 m/s) and its slow-vs-fast AUC.
            GATES (fixed before the first run): L-V4 P_slow - L-MUT P_slow >= 0.20 (the copy slots' labels are what
            the loss consumes; a trainer that ignored them would read ~0) AND L-V4 AUC(slow pass vs fast fail) >= 0.80.
Prints ZZV4TRAIN_OK or ZZV4TRAIN_FAIL <arms>.
  python diag_slow_labels_v4_train.py --bank D:/Projects/TanitAD/data/refe_consistent [--work DIR] [--gpu] [--steps 120]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diag_onpolicy_train as DOT  # noqa: E402  carve() and run_train(): the on-policy gate's own harness
import onpolicy_label as OL  # noqa: E402
import onpolicy_label_v4 as L4  # noqa: E402

M, T_H = 64, 20
SPEC = L4.SlowSpec(factors=(0.75, 0.5), frac=1.0, sources="random")   # the banked run's composition (17 copies);
# no logits in synthetic sets -> the label-free random draw


def fan(rng, vlo, vhi, n=M) -> np.ndarray:
    """n smooth 4 s paths at constant speed U(vlo, vhi) m/s, lateral end offset U(-2, 2) m, yaw from the tangent."""
    t = np.arange(1, T_H + 1) * 0.2
    v = rng.uniform(vlo, vhi, n)[:, None]
    a = rng.uniform(-2.0, 2.0, n)[:, None]
    x = v * t[None]
    y = a * (t[None] / t[-1]) ** 2
    yaw = np.arctan2(np.gradient(y, axis=1), np.maximum(np.gradient(x, axis=1), 1e-3))
    return np.stack([x, y, yaw], -1)


def mean_speed(tr) -> float:
    xy = np.concatenate([np.zeros((1, 2)), np.asarray(tr)[:, :2]], 0)
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() / (0.2 * T_H))


def target(tr) -> dict:
    """The known rule: collision iff mean speed > 10 m/s; EP = distance / 60 m; everything else clean."""
    v = mean_speed(tr)
    d = {kk: 0.0 for kk in OL.KEEP}
    d.update({"collision.NuPlanCollision.info": 1.0 if v > 10.0 else 0.0, "progress.ep": min(1.0, v * 4.0 / 60.0),
              "progress.advance_m": v * 4.0, "ttc.NuPlanTTC.ttc_reward": 1.0, "comfort.Comfort.reward": 1.0})
    return OL.apply_navsim(d, 0.0, 1.0)


def key_row(k: dict) -> dict:
    return {"log_name": k["log_name"], "token": k.get("token", ""), "step": int(k["step"]),
            "rank": int(k.get("rank", 0)), "ckpt_step": 1, "aug": None}


def build(keys, seed: int) -> dict:
    """Per sample: originals, plan, copies, and the three label variants, from the REAL v4 functions."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in keys:
        r = key_row(k)
        P = fan(rng, 8.0, 15.0)
        key = (r["log_name"], r["token"], r["step"], r["rank"])
        plan = L4.plan_slow(key, M, None, SPEC)
        C = np.stack([L4.copy_traj(P, s, f) for s, f in zip(plan["src"], plan["factor"])])
        tg = [target(P[j]) for j in range(M)]
        ctg = [target(C[i]) for i in range(len(C))]
        ctg_mut = [dict(tg[s]) if s >= 0 else target(C[i]) for i, s in enumerate(plan["src"])]
        out[key] = {"r": r, "P": P, "plan": plan, "C": C, "tg": tg, "ctg": ctg, "ctg_mut": ctg_mut}
    return out


def line_of(r, served, stg, lv, slow=None) -> str:
    S = types.SimpleNamespace(stride=2)
    return L4._line(r, S, 10, served, stg, {kk: 0.0 for kk in OL.KEEP}, lv, slow, "diag", time.time())


def write(d: Path, name: str, lines: list) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text("".join(lines), encoding="utf-8")


def slow_block(plan) -> dict:
    return {"carries": True, "slots": plan["slots"], "src": plan["src"], "factor": plan["factor"],
            "src_rank": plan["src_rank"], "sources": plan["sources"], "spec": SPEC.sha()}


def comps(t) -> list:
    import train as TR
    c = TR.ScorerBank.components(t)
    c[1] = 1.0 - min(max(float(t["navsim_dac.violation"]), 0.0), 1.0)
    return c


def served_arrays(served) -> np.ndarray:
    """What the bank holds after the line's rounding (5 dp x, y; 6 dp yaw), float32."""
    xy = np.round(served[:, :, :2], 5)
    yw = np.round(served[:, :, 2], 6)
    return np.concatenate([xy, yw[..., None]], -1).astype(np.float32)


def loader_arms(work: Path, bank: Path, S: dict) -> dict:
    import train as TR
    from model import REFeConfig
    res = {}
    # R-LOAD
    d = work / "r_load"
    v3 = [line_of(s["r"], s["P"], s["tg"], 3) for s in S.values()]
    v4 = []
    for s in S.values():
        served, stg = L4.serve_set(s["P"], s["tg"], s["plan"], s["C"], s["ctg"])
        s["served"], s["stg"] = served, stg
        v4.append(line_of(s["r"], served, stg, 4, slow_block(s["plan"])))
    write(d, "onpolicy_a_v3.jsonl", v3)
    write(d, "onpolicy_b_v4.jsonl", v4)
    b = TR.OnPolicyBank(str(d), M, T_H)
    ok = len(b.by) == len(S) and b.n_incomplete == 0 and b.n_superseded == len(S) and b.n_navsim_dac == len(S)
    ok_serve = True
    for key, s in S.items():
        e = b.by.get(key)
        ok_serve &= e is not None and e[3] == (1, 4) and e[4]
        ok_serve &= bool(np.array_equal(e[1], served_arrays(s["served"])))
        ok_serve &= bool(np.array_equal(e[2], np.asarray([comps(t) for t in s["stg"]], np.float32)))
        for i, slot in enumerate(s["plan"]["slots"]):
            ok_serve &= bool(np.array_equal(e[2][slot], np.asarray(comps(s["ctg"][i]), np.float32)))
    cfg = REFeConfig.for_backbone("vits16")
    cfg.n_cameras = 1
    cfg.cameras = tuple(cfg.cameras[:1])
    ds = TR.TargetBank(str(bank), None, cfg, synthetic=True, onpolicy=b)
    ok_item = True
    for i, row in enumerate(ds.rows):
        key = (row["log_name"], row.get("token", ""), int(row.get("step", 0)), int(row.get("rank", 0)))
        it = ds[i]
        ct, cg, cm, ci = it[4], it[5], it[6], it[7]
        e = b.by[key]
        ok_item &= bool(torch.equal(ct, torch.from_numpy(e[1])) and torch.equal(cg, torch.from_numpy(e[2]))
                        and float(cm.sum()) == M and int(ci[0]) == 1)
    res["R-LOAD"] = bool(ok and ok_serve and ok_item)
    print(f"  R-LOAD    sets {len(b.by)}, incomplete {b.n_incomplete}, superseded {b.n_superseded}, NAVSIM DAC "
          f"{b.n_navsim_dac}; v4 set served with its copies + their components: {ok_serve}; "
          f"TargetBank.__getitem__ serves it (mask all ones): {ok_item}", flush=True)
    # M-APPEND
    d = work / "m_append"
    app = []
    for s in S.values():
        P2 = np.concatenate([s["P"], s["C"]], 0)
        app.append(line_of(s["r"], P2, s["tg"] + s["ctg"], 5))
    write(d, "onpolicy_b_v4.jsonl", v4)
    write(d, "onpolicy_c_append.jsonl", app)
    b = TR.OnPolicyBank(str(d), M, T_H)
    kept_v4 = all(b.by[k][3] == (1, 4) for k in S)
    res["M-APPEND"] = b.n_incomplete == len(S) and kept_v4
    print(f"  M-APPEND  64+{len(next(iter(S.values()))['C'])} sets: incomplete {b.n_incomplete}/{len(S)}, every key "
          f"still serves its v4 set: {kept_v4} -> appended copies never reach the loss", flush=True)
    # M-EXTRA
    d = work / "m_extra"
    ext = []
    for s in S.values():
        Cx = np.stack([L4.copy_traj(s["P"], j, 0.75) for j in range(M)])
        ext.append(line_of(s["r"], Cx, [target(c) for c in Cx], 5))
    write(d, "onpolicy_a_v3.jsonl", v3)
    write(d, "onpolicy_d_extra.jsonl", ext)
    b = TR.OnPolicyBank(str(d), M, T_H)
    replaced = all(b.by[k][3] == (1, 5) for k in S)
    no_orig = all(not any(np.array_equal(b.by[k][1][j], served_arrays(s["P"])[j]) for j in range(M))
                  for k, s in S.items())
    res["M-EXTRA"] = replaced and no_orig
    print(f"  M-EXTRA   an extra copies-only set per key: it REPLACED the own-proposal set on every key: {replaced}; "
          f"served sets hold no original: {no_orig}", flush=True)
    # M-NODAC
    d = work / "m_nodac"
    nd_lines = []
    for n, s in enumerate(S.values()):
        stg = [dict(t) for t in s["stg"]]
        if n == 0:
            stg[s["plan"]["slots"][0]].pop("navsim_dac.violation")
        nd_lines.append(line_of(s["r"], s["served"], stg, 4, slow_block(s["plan"])))
    write(d, "onpolicy_b_v4.jsonl", nd_lines)
    b = TR.OnPolicyBank(str(d), M, T_H)
    res["M-NODAC"] = b.n_navsim_dac == len(S) - 1
    print(f"  M-NODAC   one copy without navsim_dac.violation: NAVSIM DAC sets {b.n_navsim_dac}/{len(S)} "
          f"(detected: {res['M-NODAC']})", flush=True)
    return res


def eval_slow(final: Path, bank: Path, seed: int, device: str) -> dict:
    """The trained scorer on NEW plans: 64 slow (2-7 m/s, truth: no collision) + 64 fast (11-15, truth: collision)."""
    import train as TR
    from model import REFe, REFeConfig
    cfg = REFeConfig.for_backbone("vits16")
    cfg.n_cameras = 1
    cfg.cameras = tuple(cfg.cameras[:1])
    m = REFe(cfg)
    m.load_state_dict(torch.load(final, map_location="cpu", weights_only=False)["model"])
    m = m.to(device).eval()
    ds = TR.TargetBank(str(bank), None, cfg, synthetic=True)
    rng = np.random.default_rng(seed)
    ps, pf = [], []
    for i in range(len(ds)):
        img, ego, goal = ds[i][0], ds[i][1], ds[i][2]
        for vlo, vhi, acc in ((2.0, 7.0, ps), (11.0, 15.0, pf)):
            tr = torch.from_numpy(fan(rng, vlo, vhi).astype(np.float32))[None].to(device)
            with torch.no_grad():
                _, _, sx = m(img[None].to(device), ego[None].to(device), goal[None].to(device), score_extra=tr)
            acc += torch.sigmoid(sx[0, :, 0]).cpu().tolist()                     # P(no collision)
    s = np.asarray(ps + pf)
    y = np.asarray([True] * len(ps) + [False] * len(pf))
    auc = float(((s[y][:, None] > s[~y][None, :]).sum() + 0.5 * (s[y][:, None] == s[~y][None, :]).sum())
                / (y.sum() * (~y).sum()))
    return {"p_slow": float(np.mean(ps)), "p_fast": float(np.mean(pf)), "auc": auc}


def learning_arms(a, work: Path, bank: Path, S: dict) -> dict:
    res, ev = {}, {}
    dev = ([] if a.gpu else ["--cpu"]) + ["--n-cameras", "1"]
    base = ["--backbone", "vits16", "--synthetic", "--targets", str(bank), "--scorer-targets", str(bank),
            "--batch", "1", "--accum", "2", "--log-every", "1", "--weights", "none"]
    for name in ("L-V4", "L-MUT", "L-V3"):
        lines = []
        for s in S.values():
            if name == "L-V3":
                lines.append(line_of(s["r"], s["P"], s["tg"], 3))
                continue
            served, stg = L4.serve_set(s["P"], s["tg"], s["plan"], s["C"], s["ctg"] if name == "L-V4" else s["ctg_mut"])
            lines.append(line_of(s["r"], served, stg, 4, slow_block(s["plan"])))
        opd, od = work / f"op_{name}", work / f"run_{name}"
        write(opd, "onpolicy_synth.jsonl", lines)
        t0 = time.time()
        rc, out = DOT.run_train(HERE / "train.py", base + dev + ["--steps", str(a.steps), "--score-w", "1.0",
                                                                "--lr", "1e-3", "--scorer-mode", "onpolicy",
                                                                "--onpolicy-targets", str(opd), "--out", str(od)], HERE)
        ok_bank = (f"ON-POLICY scorer bank: {len(S)} complete sets" in out
                   and f"NAVSIM drivable area on {len(S)} sets" in out)
        n_op = sum(1 for l in out.splitlines() if "on-policy: sets" in l)
        if rc != 0 or not ok_bank or not n_op:
            print(f"  {name}: rc {rc} bank-line {ok_bank} on-policy lines {n_op}\n{out[-2000:]}", flush=True)
            res[name] = False
            continue
        ev[name] = eval_slow(od / "model_final.pt", bank, seed=99, device="cuda" if a.gpu else "cpu")
        print(f"  {name:6s} {a.steps} steps in {time.time() - t0:.0f} s, {n_op} on-policy lines; NEW plans: "
              f"P(no collision) slow {ev[name]['p_slow']:.3f} fast {ev[name]['p_fast']:.3f}, AUC {ev[name]['auc']:.3f}",
              flush=True)
    if "L-V4" in ev and "L-MUT" in ev:
        gap = ev["L-V4"]["p_slow"] - ev["L-MUT"]["p_slow"]
        res["L-V4_vs_L-MUT"] = gap >= 0.20
        res["L-V4_auc"] = ev["L-V4"]["auc"] >= 0.80
        print(f"  GATES     L-V4 - L-MUT on slow plans {gap:+.3f} (bar >= +0.20); L-V4 AUC {ev['L-V4']['auc']:.3f} "
              f"(bar >= 0.80); L-V3 (no copies, reported) P_slow {ev.get('L-V3', {}).get('p_slow', float('nan')):.3f}",
              flush=True)
    res["_eval"] = ev
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--loader-only", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    work = Path(a.work or tempfile.mkdtemp(prefix="refe_v4train_"))
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    bank = work / "bank"
    keys = DOT.carve(Path(a.bank), bank)
    print(f"  bank: {len(keys)} tuples carved from {a.bank}; spec {SPEC.as_dict()}", flush=True)
    S = build(keys, seed=11)
    res = loader_arms(work, bank, S)
    if not a.loader_only:
        res.update(learning_arms(a, work, bank, S))
    ev = res.pop("_eval", None)
    fails = [k for k, v in res.items() if not v]
    if a.json:
        json.dump({"results": res, "eval": ev, "steps": a.steps, "device": "cuda" if a.gpu else "cpu",
                   "spec": SPEC.as_dict(), "at_local": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  open(a.json, "w", encoding="utf-8"), indent=1)
    print(f"  results: {res}")
    print("ZZV4TRAIN_OK" if not fails else f"ZZV4TRAIN_FAIL {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
