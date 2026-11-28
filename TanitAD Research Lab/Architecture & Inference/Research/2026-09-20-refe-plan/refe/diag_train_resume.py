"""Resume == uninterrupted, bit for bit -- with the arms that prove the test can FAIL.

⛔ WHY (R20, 2026-09-23). train.py saved nothing until today; the checkpoint/resume it now has is
what stands between a ~60-day single-A40 run and total loss on the first pod restart. "It resumed
and the loss kept falling" is not evidence: a resume that replays the epoch from its start, or
that restarts Adam's moments from zero, ALSO keeps the loss falling. The only admissible check is
that halt + resume reproduces the uninterrupted run EXACTLY, and that the two realistic ways to
get resume wrong each make it fail.

Arms (all run the REAL trainer as a subprocess, on CPU, deterministic synthetic images):
  U   uninterrupted, 6 optimiser steps                         -> model_final.pt
  H   halt right after the checkpoint at step 3, then --resume -> must equal U EXACTLY
  M1  H's checkpoint with the data position zeroed             -> must DIFFER from U
  M2  H's checkpoint with a fresh optimiser state              -> must DIFFER from U
  R1  resume with a changed --lr                               -> must be REFUSED (exit 4)
  R2  --epochs without --out                                   -> must be REFUSED (exit 4)
  R3  relaunch on a finished run dir                           -> TRAIN_ALREADY_DONE, model untouched
  R4  resume with --epoch-unit rows (the run was scenes)       -> must be REFUSED (exit 4)
  P   model_final.pt through planner.py's own load (strict, torch.load default weights_only)
  S   an epoch snapshot through ckpt_io.load_for_inference; a corrupted trunk fingerprint REFUSED
  SC  the SCENE sampler (R24): one row per scene per epoch; twins take turns 13/12 over 25
      epochs; rank-1 share mixed within every epoch -- with two mutations that must go RED:
      a parity sampler (whole epochs single-goal) and an epoch over ROWS (twins visited twice)

The bank is 6 real rows in 4 SCENES -- 2 scenes carrying a rank-0 AND a rank-1 (goal-augmented)
target, 2 rank-0-only -- so under scene epochs an epoch is 4 samples = 2 optimiser steps, and the
halt at step 3 lands MID-EPOCH (epoch 1, sample 2 of 4) inside the part of the sampler that picks
between twins -- the case a naive resume gets wrong.

Usage: python diag_train_resume.py --bank <dir with targets_rank0/1.jsonl + scorer_targets*.jsonl>
Prints RESUME_EXACT_AND_GUARDS_LIVE on success; anything else is a failure.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def sh(args, cwd=HERE):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, str(HERE / "train.py")] + args, cwd=cwd, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, r.stdout + r.stderr


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def same_model(p, q):
    a = torch.load(p, map_location="cpu")["model"]
    b = torch.load(q, map_location="cpu")["model"]
    if a.keys() != b.keys():
        return False, "key sets differ"
    diff = [k for k in a if not torch.equal(a[k], b[k])]
    return (not diff), (f"{len(diff)} tensor(s) differ, e.g. {diff[:2]}" if diff else "identical")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    work = Path(a.work or tempfile.mkdtemp(prefix="refe_resume_"))
    work.mkdir(parents=True, exist_ok=True)
    bank = work / "bank4"
    bank.mkdir(exist_ok=True)
    # a TWIN-SCENE bank (R24): 2 rank-1 rows + their rank-0 twins + 2 rank-0-only scenes
    def scene(r):
        return (r["log_name"], r.get("token", ""), int(r["step"]))
    r1 = []
    with open(Path(a.bank) / "targets_rank1.jsonl", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                r1.append(l)
            if len(r1) == 2:
                break
    want = {scene(json.loads(l)) for l in r1}
    twins0, single0 = {}, []
    with open(Path(a.bank) / "targets_rank0.jsonl", encoding="utf-8") as f:
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
    if len(r1) < 2 or len(twins0) != len(want) or len(single0) < 2:
        print(f"  cannot build a twin-scene bank from {a.bank}: rank-1 {len(r1)}/2, twins "
              f"{len(twins0)}/{len(want)}, rank-0-only {len(single0)}/2")
        print("RESUME_TEST_FAILED")
        return 1
    r0 = [twins0[k] for k in sorted(twins0)] + single0
    (bank / "targets_rank0.jsonl").write_text("".join(r0), encoding="utf-8")
    (bank / "targets_rank1.jsonl").write_text("".join(r1), encoding="utf-8")
    # carve ONLY these tuples' scorer rows: on the pod the files are GBs, and a 6-tuple test must
    # not load them (nor hash them into the run identity). A token prefilter keeps the scan cheap.
    keys = set()
    for l in r0 + r1:
        r = json.loads(l)
        keys.add(scene(r) + (int(r.get("rank", 0)),))
    toks = {k[1] for k in keys}
    n_sc = 0
    for fname in ("scorer_targets.jsonl", "scorer_targets_rank1.jsonl"):
        src = Path(a.bank) / fname
        if not src.exists():
            continue
        with open(src, encoding="utf-8") as f, open(bank / fname, "w", encoding="utf-8") as g:
            for l in f:
                if not any(t in l for t in toks):
                    continue
                r = json.loads(l)
                if scene(r) + (int(r.get("rank", 0)),) in keys:
                    g.write(l)
                    n_sc += 1
    print(f"  bank: {len(r0) + len(r1)} tuples in 4 scenes (2 with a rank-1 twin), {n_sc} scorer "
          f"rows carved from {a.bank}", flush=True)
    base = ["--backbone", "vits16", "--cpu", "--synthetic", "--targets", str(bank),
            "--scorer-targets", str(bank), "--batch", "1", "--accum", "2", "--steps", "6",
            "--log-every", "1", "--lr", "2e-4"]
    res, ok = {}, True

    def check(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        res[name] = ("PASS" if cond else "FAIL") + " -- " + detail
        print(f"  [{name}] {res[name]}", flush=True)

    # U: uninterrupted
    rc, out = sh(base + ["--out", str(work / "U")])
    check("U", rc == 0 and (work / "U/model_final.pt").exists()
          and (work / "U/snap_epoch001.pt").exists(), f"exit {rc}, final + epoch snapshots")
    # the trainer's OWN grouping, read against LITERALS (not against an expression over its code)
    check("U.scenes", "scenes: 4 over 6 tuples  (rows per scene: 1 x 2, 2 x 2)" in out
          and "epoch = one pass over 4 scenes" in out,
          "trainer groups 6 rows into 4 scenes (2 singles, 2 twins) and epochs over the scenes")
    # H: halt at the step-3 checkpoint, then resume
    rc1, out1 = sh(base + ["--out", str(work / "H"), "--ckpt-every-steps", "3",
                           "--halt-after-steps", "3"])
    st = torch.load(work / "H/ckpt_last.pt", map_location="cpu", weights_only=False)
    mid = st["state"]
    check("H.halt", rc1 == 7 and "TRAIN_HALTED" in out1 and mid["step"] == 3
          and mid["epoch"] == 1 and mid["pos"] == 2,
          f"exit {rc1}, halted at step {mid['step']} epoch {mid['epoch']} pos {mid['pos']} "
          f"(want 3 / 1 / 2: MID-epoch)")
    for m in ("M1", "M2", "R1", "R4"):
        shutil.copytree(work / "H", work / m)
    rc2, out2 = sh(base + ["--out", str(work / "H"), "--resume"])
    same, why = same_model(work / "U/model_final.pt", work / "H/model_final.pt")
    check("H.resume==U", rc2 == 0 and "RESUMED" in out2 and same, f"exit {rc2}, {why}")

    # M1: data position lost -> the resumed run replays the epoch from its start
    s1 = torch.load(work / "M1/ckpt_last.pt", map_location="cpu", weights_only=False)
    s1["state"]["pos"] = 0
    torch.save(s1, work / "M1/ckpt_last.pt")
    rc, _ = sh(base + ["--out", str(work / "M1"), "--resume"])
    same, why = same_model(work / "U/model_final.pt", work / "M1/model_final.pt")
    check("M1.pos_lost_DIFFERS", rc == 0 and not same, f"exit {rc}, {why}")

    # M2: optimiser state lost -> Adam's moments restart from zero
    s2 = torch.load(work / "M2/ckpt_last.pt", map_location="cpu", weights_only=False)
    s2["opt"]["state"] = {}
    torch.save(s2, work / "M2/ckpt_last.pt")
    rc, _ = sh(base + ["--out", str(work / "M2"), "--resume"])
    same, why = same_model(work / "U/model_final.pt", work / "M2/model_final.pt")
    check("M2.opt_lost_DIFFERS", rc == 0 and not same, f"exit {rc}, {why}")

    # R1: a changed defining argument must be refused, not reconciled
    b2 = list(base)
    b2[b2.index("--lr") + 1] = "3e-4"
    rc, out = sh(b2 + ["--out", str(work / "R1"), "--resume"])
    check("R1.changed_lr_REFUSED", rc == 4 and "REFUSING TO RESUME" in out
          and not (work / "R1/model_final.pt").exists(), f"exit {rc}")
    # R4: the epoch UNIT defines the run as much as the lr does
    rc, out = sh(base + ["--out", str(work / "R4"), "--resume", "--epoch-unit", "rows"])
    check("R4.changed_epoch_unit_REFUSED", rc == 4 and "REFUSING TO RESUME" in out
          and "epoch-unit" in out, f"exit {rc}")
    # R2: a real schedule without a run directory
    rc, out = sh(["--backbone", "vits16", "--cpu", "--synthetic", "--targets", str(bank),
                  "--scorer-targets", str(bank), "--epochs", "1"])
    check("R2.no_out_REFUSED", rc == 4 and "REFUSING TO TRAIN WITHOUT --out" in out, f"exit {rc}")
    # R3: a finished run must not be re-run by a supervisor
    h0 = sha(work / "U/model_final.pt")
    rc, out = sh(base + ["--out", str(work / "U"), "--resume"])
    check("R3.done_is_final", rc == 0 and "TRAIN_ALREADY_DONE" in out
          and sha(work / "U/model_final.pt") == h0, f"exit {rc}, model_final sha unchanged")

    # P: exactly planner.py's load -- strict, torch.load with the DEFAULT weights_only
    from model import REFe, REFeConfig
    import ckpt_io
    cfg = REFeConfig.for_backbone("vits16")
    try:
        m = REFe(cfg)
        sd = torch.load(work / "U/model_final.pt", map_location="cpu")
        m.load_state_dict(sd["model"] if "model" in sd else sd)
        fmt_full = ckpt_io.load_for_inference(REFe(cfg), str(work / "U/model_final.pt"))
        check("P.planner_strict_load", fmt_full == ckpt_io.FORMAT_FULL,
              f"{len(sd['model'])} tensors, strict; planner's loader reads it as {fmt_full}")
    except Exception as e:  # noqa: BLE001
        check("P.planner_strict_load", False, repr(e)[:200])

    # S: an epoch snapshot is evaluable, and a different trunk is refused
    try:
        m = REFe(cfg)
        fmt = ckpt_io.load_for_inference(m, str(work / "U/snap_epoch002.pt"), backbone="vits16")
        snap = torch.load(work / "U/snap_epoch002.pt", map_location="cpu", weights_only=False)
        sd_m = m.state_dict()
        eq = all(torch.equal(sd_m[k], v) for k, v in snap["model_partial"].items())
        check("S.snapshot_loads", fmt == ckpt_io.FORMAT_PARTIAL and eq,
              f"{fmt}, {len(snap['model_partial'])} partial tensors applied exactly")
        bad = copy.deepcopy(snap)
        bad["meta"]["frozen_sha256"] = "0" * 64
        torch.save(bad, work / "snap_bad.pt")
        try:
            ckpt_io.load_for_inference(REFe(cfg), str(work / "snap_bad.pt"), backbone="vits16")
            check("S.wrong_trunk_REFUSED", False, "a corrupted fingerprint was ACCEPTED")
        except ValueError as e:
            check("S.wrong_trunk_REFUSED", "fingerprint" in str(e), str(e)[:120])
    except Exception as e:  # noqa: BLE001
        check("S.snapshot_loads", False, repr(e)[:200])

    # SC: the scene sampler itself, against LITERAL expectations, with two mutations
    import train as T
    groups = [[0, 1], [2, 3], [4], [5]]          # two twin scenes, two single scenes
    owner = {i: q for q, g in enumerate(groups) for i in g}
    use = {i: 0 for i in range(6)}
    once = True
    smp = T.SceneEpochSampler(groups, seed=0, shuffle=True)
    for e in range(25):
        smp.set_epoch(e)
        o = smp.order()
        once &= sorted(owner[i] for i in o) == [0, 1, 2, 3]
        for i in o:
            use[i] += 1
    check("SC1.one_row_per_scene", once, "25 epochs, every scene exactly once in each")
    check("SC2.twins_take_turns", sorted((use[0], use[1])) == [12, 13]
          and sorted((use[2], use[3])) == [12, 13] and use[4] == use[5] == 25,
          f"uses over 25 epochs {use}")
    big = [[2 * q, 2 * q + 1] for q in range(400)]

    def shares(sm):
        out = []
        for e in range(25):
            sm.set_epoch(e)
            o = sm.order()
            out.append(sum(i % 2 for i in o) / len(o))
        return out
    f = shares(T.SceneEpochSampler(big, seed=0))
    check("SC3.mixed_within_epoch", all(0.35 <= x <= 0.65 for x in f),
          f"rank-1 share per epoch {min(f):.2f}-{max(f):.2f} over 400 twin scenes")
    par = T.SceneEpochSampler(big, seed=0)
    par.offset = [0] * len(big)                  # MUTATION: plain even/odd alternation
    fp = shares(par)
    check("SC-M1.parity_DETECTED", not all(0.35 <= x <= 0.65 for x in fp),
          f"parity sampler's per-epoch rank-1 share {min(fp):.2f}-{max(fp):.2f} (must be 0 or 1)")
    rs = T.EpochSampler(6, seed=0)               # MUTATION: an epoch over ROWS
    rs.set_epoch(0)
    hits = sorted(owner[i] for i in rs.order())
    check("SC-M2.row_epoch_DETECTED", hits != [0, 1, 2, 3],
          f"a row epoch visits scenes {hits} (twin scenes twice)")

    (work / "resume_report.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"  report: {work / 'resume_report.json'}")
    if not a.keep and ok:
        shutil.rmtree(work, ignore_errors=True)
    print("RESUME_EXACT_AND_GUARDS_LIVE" if ok else "RESUME_TEST_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
