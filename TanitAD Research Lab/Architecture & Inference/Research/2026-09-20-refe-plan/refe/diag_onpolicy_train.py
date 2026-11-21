#!/usr/bin/env python3
"""The on-policy trainer change (PI decision 2026-09-26, B) -- the gate before the live switch.

Every arm runs the REAL trainer as a subprocess on a 6-tuple bank carved from a real one, synthetic
images (the trainer's own --synthetic), ViT-S.
  F-EXACT     fixed mode: the NEW train.py and the OLD one (the staged blob the live run uses) train
              6 steps on CPU -> final weights must be IDENTICAL, tensor for tensor
  O-LEARN     on-policy mode on synthetic label sets whose drivable-area label is a GEOMETRIC rule
              (|y at 4 s| > 3 m -> violation, via `navsim_dac.violation`, label_version 2; the
              teacher's `dac.violation` is 0 everywhere, as measured on real proposals)
  O-SHUF      identical, but each set's labels are SHUFFLED across its slots (the pairing broken)
              -> after training, both models score 64 NEW trajectories per scene: O-LEARN's
              drivable-area AUC must be high and O-SHUF's near chance -- the proof that each label
              reaches its own trajectory and that the NAVSIM label is the one learned
  O-REFUSE    a fixed-mode checkpoint resumed in on-policy mode WITHOUT --declare-change -> exit 4
  O-DECLARE   the same WITH --declare-change scorer_mode -> resumes, prints the declared change,
              logs a `declared_change` event, trains on
  O-PREFLIGHT --preflight against a run dir -> PREFLIGHT_OK and the directory byte-identical after
Prints ZZOPTRAIN_OK or ZZOPTRAIN_FAIL <arms>; exit 0 / 1.

  python diag_onpolicy_train.py --bank D:/Projects/TanitAD/data/refe_consistent [--work DIR] [--gpu]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
M_PROP, T_H = 64, 20


def run_train(train_py: Path, args: list, cwd: Path) -> tuple:
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable, str(train_py)] + args, cwd=str(cwd), env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                       errors="replace")
    return p.returncode, p.stdout


def dir_digest(d: Path) -> str:
    h = hashlib.sha256()
    for q in sorted(d.rglob("*")):
        if q.is_file():
            h.update(q.name.encode())
            h.update(q.read_bytes())
    return h.hexdigest()


def carve(src: Path, dst: Path) -> list:
    """6 tuples in 4 scenes (2 with a rank-1 twin), as diag_train_resume.py builds them."""
    dst.mkdir(parents=True, exist_ok=True)
    scene = lambda r: (r["log_name"], r.get("token", ""), int(r["step"]))           # noqa: E731
    r1 = []
    with open(src / "targets_rank1.jsonl", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                r1.append(l)
            if len(r1) == 2:
                break
    want = {scene(json.loads(l)) for l in r1}
    twins0, single0 = {}, []
    with open(src / "targets_rank0.jsonl", encoding="utf-8") as f:
        for l in f:
            if not l.strip():
                continue
            k = scene(json.loads(l))
            if k in want:
                twins0.setdefault(k, l)
            elif len(single0) < 2:
                single0.append(l)
            if len(twins0) == len(want) and len(single0) == 2:
                break
    r0 = [twins0[k] for k in sorted(twins0)] + single0
    (dst / "targets_rank0.jsonl").write_text("".join(r0), encoding="utf-8")
    (dst / "targets_rank1.jsonl").write_text("".join(r1), encoding="utf-8")
    keys = [json.loads(l) for l in r0 + r1]
    toks = {k["token"] for k in keys}
    want_k = {scene(k) + (int(k.get("rank", 0)),) for k in keys}
    for fname in ("scorer_targets.jsonl", "scorer_targets_rank1.jsonl"):
        s = src / fname
        if s.exists():
            with open(s, encoding="utf-8") as f, open(dst / fname, "w", encoding="utf-8") as g:
                for l in f:
                    if any(t in l for t in toks):
                        r = json.loads(l)
                        if scene(r) + (int(r.get("rank", 0)),) in want_k:
                            g.write(l)
    return keys


def synth_trajs(rng) -> np.ndarray:
    """64 smooth 4 s paths: speed 0-15 m/s, lateral end offset -6..+6 m, yaw from the tangent."""
    t = np.arange(1, T_H + 1) * 0.2
    v = rng.uniform(0.0, 15.0, M_PROP)[:, None]
    a = rng.uniform(-6.0, 6.0, M_PROP)[:, None]
    x = v * t[None]
    y = a * (t[None] / t[-1]) ** 2
    dx = np.gradient(x, axis=1)
    dy = np.gradient(y, axis=1)
    yaw = np.arctan2(dy, np.maximum(dx, 1e-3))
    return np.stack([x, y, yaw], -1).astype(np.float32)                      # [64, 20, 3]


def rule_dac_violation(tr: np.ndarray) -> np.ndarray:
    return (np.abs(tr[:, -1, 1]) > 3.0).astype(np.float32)


def write_sets(keys: list, out: Path, seed: int, shuffle: bool, rand: bool = False) -> None:
    """rand=True: the INFORMATION-FREE control -- every label drawn independently of every trajectory
    (within-set shuffling is NOT information-free: it keeps each set's violation COUNT, which the set-level
    scoring decoder can turn back into the rule -- MEASURED 0.715 vs the untrained 0.423)."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    lines = []
    for k in keys:
        tr = synth_trajs(rng)
        viol = (rng.random(M_PROP) < 0.5).astype(np.float32) if rand else rule_dac_violation(tr)
        ep = rng.random(M_PROP) if rand else np.minimum(1.0, tr[:, -1, 0] / 40.0)
        tg = [{"collision.NuPlanCollision.info": 0.0, "dac.violation": 0.0, "off_road.OffRoad.info": 0.0,
               "progress.ep": float(ep[j]), "ttc.NuPlanTTC.ttc_reward": 1.0,
               "comfort.Comfort.reward": 1.0, "ddc.violation": 0.0,
               "navsim_dac.violation": float(viol[j])} for j in range(M_PROP)]
        if shuffle:
            perm = rng.permutation(M_PROP)
            tg = [tg[p] for p in perm]
        lines.append(json.dumps({
            "kind": "onpolicy_set", "log_name": k["log_name"], "token": k.get("token", ""),
            "step": int(k["step"]), "rank": int(k.get("rank", 0)), "ckpt_step": 1, "label_version": 2,
            "traj": tr[:, :, :2].tolist(), "yaw": tr[:, :, 2].tolist(), "targets": tg}) + "\n")
    (out / "onpolicy_synth.jsonl").write_text("".join(lines), encoding="utf-8")


def eval_dac_auc(final: Path, bank: Path, keys: list, seed: int, device: str) -> float:
    """Score 64 NEW trajectories per training scene with the trained model; AUC of predicted DAC-pass."""
    sys.path.insert(0, str(HERE))
    import train as T
    from model import REFe, REFeConfig
    cfg = REFeConfig.for_backbone("vits16")
    cfg.n_cameras = 1                                   # the learning arms train --n-cameras 1
    cfg.cameras = tuple(cfg.cameras[:1])
    if final is None:                                   # the UNTRAINED scorer: the trainer's own init
        torch.manual_seed(0)
    m = REFe(cfg)
    if final is not None:
        sd = torch.load(final, map_location="cpu", weights_only=False)
        m.load_state_dict(sd["model"])
    m = m.to(device).eval()
    ds = T.TargetBank(str(bank), None, cfg, synthetic=True)
    rng = np.random.default_rng(seed)
    scores, labels = [], []
    for i in range(len(ds)):
        img, ego, goal = ds[i][0], ds[i][1], ds[i][2]
        tr = synth_trajs(rng)
        with torch.no_grad():
            _, _, sx = m(img[None].to(device), ego[None].to(device), goal[None].to(device),
                         score_extra=torch.from_numpy(tr)[None].to(device))
        scores += torch.sigmoid(sx[0, :, 1]).cpu().tolist()                  # P(drivable-area pass)
        labels += (1.0 - rule_dac_violation(tr)).tolist()
    s, y = np.asarray(scores), np.asarray(labels) > 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def exact_and_learning(a, work, bank, keys, base) -> dict:
    res = {}
    old = work / "old_refe"
    shutil.copytree(HERE, old, ignore=shutil.ignore_patterns("__pycache__"))
    rel = str((HERE / "train.py").relative_to(Path("D:/Projects/TanitAD"))).replace("\\", "/")
    blob = subprocess.run(["git", "-C", "D:/Projects/TanitAD", "show", f":{rel}"], capture_output=True).stdout
    assert blob and b"def run(a)" in blob, "could not read the staged train.py"
    (old / "train.py").write_bytes(blob)
    outs = {}
    for name, tp in (("new", HERE / "train.py"), ("old", old / "train.py")):
        od = work / f"fx_{name}"
        rc, out = run_train(tp, base + ["--cpu", "--steps", "6", "--out", str(od)], tp.parent)
        outs[name] = (rc, od / "model_final.pt")
        if rc != 0:
            print(out[-1500:])
    if all(v[0] == 0 for v in outs.values()):
        A = torch.load(outs["new"][1], map_location="cpu", weights_only=False)["model"]
        B = torch.load(outs["old"][1], map_location="cpu", weights_only=False)["model"]
        same = A.keys() == B.keys() and all(torch.equal(A[k], B[k]) for k in A)
        res["F-EXACT"] = same
        print(f"  F-EXACT     new vs staged trainer, 6 steps, {len(A)} tensors: {'IDENTICAL' if same else 'DIFFER'}", flush=True)
    else:
        res["F-EXACT"] = False
        print(f"  F-EXACT     trainer failed: rc {outs['new'][0]} / {outs['old'][0]}")

    # O-LEARN / O-SHUF
    dev = ([] if a.gpu else ["--cpu"]) + ["--n-cameras", "1"]
    aucs = {}
    for name, shuf in (("O-LEARN", False), ("O-SHUF", True)):
        opd = work / f"op_{name}"
        write_sets(keys, opd, seed=7, shuffle=shuf)
        od = work / f"run_{name}"
        rc, out = run_train(HERE / "train.py", base + dev + ["--steps", str(a.learn_steps), "--score-w", "1.0",
                                                             "--lr", "1e-3", "--scorer-mode", "onpolicy",
                                                             "--onpolicy-targets", str(opd), "--out", str(od)], HERE)
        ok_bank = "ON-POLICY scorer bank: 6 complete sets" in out and "NAVSIM drivable area on 6 sets" in out
        lines = [l for l in out.splitlines() if "on-policy: sets" in l]
        if rc != 0 or not ok_bank or not lines:
            print(f"  {name}: rc {rc} bank-line {ok_bank} on-policy lines {len(lines)}")
            print(out[-2500:])
            res[name] = False
            continue
        auc = eval_dac_auc(od / "model_final.pt", bank, keys, seed=99, device="cuda" if a.gpu else "cpu")
        aucs[name] = auc
        print(f"  {name:10s}  {len(lines)} on-policy log lines; first '{lines[0].strip()[:90]}'", flush=True)
        print(f"  {name:10s}  last  '{lines[-1].strip()[:90]}'", flush=True)
        print(f"  {name:10s}  drivable-area AUC on 64 NEW trajectories x {len(keys)} scenes: {auc:.3f}", flush=True)
    if "O-LEARN" in aucs and "O-SHUF" in aucs:
        res["O-LEARN"] = aucs["O-LEARN"] >= 0.80

    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--gpu", action="store_true", help="O-LEARN / O-SHUF on CUDA (F-EXACT stays on CPU)")
    ap.add_argument("--learn-steps", type=int, default=120)
    ap.add_argument("--resume-only", action="store_true", help="only the resume arms (the switch gate)")
    ap.add_argument("--rand-only", action="store_true",
                    help="only the information-free control: O-BASE (untrained) and O-RAND (i.i.d. labels)")
    a = ap.parse_args()
    work = Path(a.work or tempfile.mkdtemp(prefix="refe_optrain_"))
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    bank = work / "bank"
    keys = carve(Path(a.bank), bank)
    print(f"  bank: {len(keys)} tuples carved from {a.bank}", flush=True)
    fails, res = [], {}
    base = ["--backbone", "vits16", "--synthetic", "--targets", str(bank), "--scorer-targets", str(bank),
            "--batch", "1", "--accum", "2", "--log-every", "1", "--lr", "2e-4", "--weights", "none"]

    if a.rand_only:
        dev = ([] if a.gpu else ["--cpu"]) + ["--n-cameras", "1"]
        base_auc = eval_dac_auc(None, bank, keys, seed=99, device="cuda" if a.gpu else "cpu")
        opd, od = work / "op_O-RAND", work / "run_O-RAND"
        write_sets(keys, opd, seed=7, shuffle=False, rand=True)
        rc, out = run_train(HERE / "train.py", base + dev + ["--steps", str(a.learn_steps), "--score-w", "1.0",
                                                             "--lr", "1e-3", "--scorer-mode", "onpolicy",
                                                             "--onpolicy-targets", str(opd), "--out", str(od)], HERE)
        if rc != 0:
            print(out[-2000:])
            print("ZZOPTRAIN_FAIL O-RAND trainer")
            return 1
        rand_auc = eval_dac_auc(od / "model_final.pt", bank, keys, seed=99, device="cuda" if a.gpu else "cpu")
        lines = [l for l in out.splitlines() if "on-policy: sets" in l]
        ok = abs(rand_auc - base_auc) <= 0.12
        print(f"  O-BASE      untrained scorer, drivable-area AUC on the same new trajectories: {base_auc:.3f}")
        print(f"  O-RAND      i.i.d. labels, {len(lines)} on-policy lines, last '{lines[-1].strip()[:80] if lines else ''}'")
        print(f"  O-RAND      drivable-area AUC {rand_auc:.3f}  (|O-RAND - O-BASE| = {abs(rand_auc - base_auc):.3f}, bar 0.12; "
              f"run 1: O-LEARN 0.998, O-SHUF 0.715)")
        print("ZZOPTRAIN_OK" if ok else "ZZOPTRAIN_FAIL O-RAND")
        return 0 if ok else 1
    if not a.resume_only:
        res.update(exact_and_learning(a, work, bank, keys, base))
    else:
        write_sets(keys, work / 'op_O-LEARN', seed=7, shuffle=False)

    # -- resume arms --
    # O-REFUSE / O-DECLARE / O-PREFLIGHT on a fixed-mode checkpoint -- as tested first (no --grow) AND
    # as production runs (G-: --grow, where the resumed epoch keeps its checkpointed snapshot and the
    # on-policy sets enter at the NEXT epoch boundary)
    opd = work / "op_O-LEARN"
    op = ["--scorer-mode", "onpolicy", "--onpolicy-targets", str(opd)]
    for pfx, extra in (("O-", []), ("G-", ["--grow"])):
        rd = work / f"resume_run_{pfx[0]}"
        rc, out = run_train(HERE / "train.py", base + extra + ["--cpu", "--steps", "6", "--out", str(rd),
                                                              "--halt-after-steps", "2", "--ckpt-every-steps", "2"], HERE)
        if rc != 7:
            print(f"  {pfx}halt run rc {rc} (expected 7)")
            print(out[-1200:])
            res[pfx + "REFUSE"] = res[pfx + "DECLARE"] = res[pfx + "PREFLIGHT"] = False
            continue
        rc1, out1 = run_train(HERE / "train.py", base + extra + ["--cpu", "--steps", "6", "--out", str(rd), "--resume"] + op, HERE)
        res[pfx + "REFUSE"] = rc1 == 4 and "--scorer-mode: 'fixed' -> 'onpolicy'" in out1
        print(f"  {pfx}REFUSE    rc {rc1}: {'refused, naming the change' if res[pfx + 'REFUSE'] else 'NOT refused as required'}")
        before = dir_digest(rd)
        rc2, out2 = run_train(HERE / "train.py", base + extra + ["--cpu", "--steps", "6", "--out", str(rd), "--resume",
                                                                "--preflight", "--declare-change", "scorer_mode"] + op, HERE)
        res[pfx + "PREFLIGHT"] = rc2 == 0 and "PREFLIGHT_OK" in out2 and dir_digest(rd) == before
        print(f"  {pfx}PREFLIGHT rc {rc2}: {'OK and the run dir is byte-identical' if res[pfx + 'PREFLIGHT'] else 'FAILED'}")
        if not res[pfx + "PREFLIGHT"]:
            print(out2[-1500:])
        rc3, out3 = run_train(HERE / "train.py", base + extra + ["--cpu", "--steps", "6", "--out", str(rd), "--resume",
                                                                "--declare-change", "scorer_mode"] + op, HERE)
        ev = [json.loads(l) for l in (rd / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        dec = [e for e in ev if e.get("event") == "declared_change"]
        n_op = sum(1 for l in out3.splitlines() if "on-policy: sets" in l)
        grow_ok = (not extra) or ("ON-POLICY sets 6" in out3)
        res[pfx + "DECLARE"] = (rc3 == 0 and "DECLARED RECIPE CHANGE" in out3 and len(dec) == 1 and n_op > 0
                                and "TRAIN_DONE" in out3 and grow_ok)
        print(f"  {pfx}DECLARE   rc {rc3}: declared events {len(dec)}, on-policy lines {n_op}, "
              f"{'TRAIN_DONE' if 'TRAIN_DONE' in out3 else 'no TRAIN_DONE'}"
              + (f", epoch-boundary line with ON-POLICY sets: {grow_ok}" if extra else ""))
        if not res[pfx + "DECLARE"]:
            print(out3[-2500:])
    fails = [k for k, v in res.items() if not v]
    print(f"  results: {res}")
    print("ZZOPTRAIN_OK" if not fails else f"ZZOPTRAIN_FAIL {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
