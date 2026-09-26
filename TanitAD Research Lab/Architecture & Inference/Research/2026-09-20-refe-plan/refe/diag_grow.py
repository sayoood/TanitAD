"""--grow: training on a bank that is still being written -- exact, and the guards that prove it.

⛔ WHY (2026-09-24). The PI asked for training and data prep to run IN PARALLEL: the trainer starts
once rank 0 is complete while the augmentation search and the scorer keep appending rows. A bank
that changes under a running job is exactly how a resume stops being a resume, so the growing mode
is admissible only if (a) an epoch reads a FIXED bank -- the byte snapshot taken at its start, kept
in the checkpoint -- and (b) rows appended mid-epoch appear at the NEXT epoch, not before and not never.

Arms (the REAL trainer as a subprocess, CPU, ViT-S, synthetic images; the bank is built here):
  U    --grow, uninterrupted, 3 epochs over a STATIC 8-scene bank            -> snap_epoch001.pt
  H+R  same, halted mid-epoch 0; the harness then APPENDS 2 twins + 2 new scenes (and scorer rows)
       and a partial line; --resume                                         -> epoch 0 == U exactly
  G2   H+R's bank events: epoch 0 read 8 tuples / 8 scenes, epoch 1 read 12 / 10 (growth seen once)
  M    H+R again but the resume IGNORES the checkpointed snapshot (deliberate regression)
                                                                            -> epoch 0 must DIFFER
  N    WITHOUT --grow, halted, bank appended, --resume                      -> REFUSED (exit 4)
  I    --grow resume with a changed --grow-scenes                           -> REFUSED (exit 4)
  T    read_upto: a torn tail is not a row; a file SHORTER than its snapshot is refused
Prints ZZGROW_EXACT_AND_GUARDS_LIVEZZ on success.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CAMS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"]


def row(i, rank=0, seed=0):
    rnd = random.Random(i * 1000 + rank * 7 + seed)
    return {"image": [f"logA/{c}/{i:04d}.jpg" for c in CAMS], "cameras": CAMS,
            "ego": [rnd.uniform(-1, 1) for _ in range(7)],
            "goal": [rnd.uniform(-30, 30) for _ in range(4)],
            "traj": [[0.5 * (k + 1) + rnd.uniform(-0.1, 0.1), rnd.uniform(-0.3, 0.3),
                      rnd.uniform(-0.05, 0.05)] for k in range(20)],
            "rank": rank, "step": 0, "token": f"tok{i:04d}", "log_name": "logA",
            "source": "diag_grow"}


def scorer_rows(i, rank=0):
    out = []
    for name in ("teacher", "lat-2", "stopped"):
        rnd = random.Random(hash((i, rank, name)) & 0xFFFF)
        out.append({"log_name": "logA", "token": f"tok{i:04d}", "step": 0, "rank": rank,
                    "candidate": name, "traj": [[0.5 * (k + 1), rnd.uniform(-1, 1)] for k in range(20)],
                    "targets": {"collision.NuPlanCollision.info": float(name == "stopped"),
                                "dac.violation": 0.0, "progress.ep": rnd.uniform(0, 1),
                                "ttc.NuPlanTTC.ttc_reward": 1.0, "comfort.Comfort.reward": 1.0,
                                "ddc.violation": 0.0}})
    return out


def write(path, rows, mode="w"):
    with open(path, mode, encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def make_bank(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    write(d / "targets_rank0.jsonl", [row(i) for i in range(8)])
    write(d / "scorer_targets.jsonl", [x for i in range(0, 8, 2) for x in scorer_rows(i)])


def grow_bank(d: Path, torn: bool = True):
    write(d / "targets_rank0.jsonl", [row(i) for i in (8, 9)], "a")
    write(d / "targets_rank1.jsonl", [row(i, rank=1) for i in (0, 1)], "a")
    write(d / "scorer_targets.jsonl", [x for i in (1, 3, 8) for x in scorer_rows(i)], "a")
    if torn:
        with open(d / "targets_rank0.jsonl", "a", encoding="utf-8") as f:
            f.write('{"image": ["torn')              # a writer killed mid-row: NOT a row


def sh(args, env_extra=None):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", **(env_extra or {}))
    r = subprocess.run([sys.executable, str(HERE / "train.py")] + args, cwd=HERE, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, r.stdout + r.stderr


def partial(p):
    return torch.load(p, map_location="cpu", weights_only=False)["model_partial"]


def same(p, q):
    a, b = partial(p), partial(q)
    if a.keys() != b.keys():
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def bank_events(run):
    ev = []
    for l in open(run / "metrics.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r.get("event") == "bank":
            ev.append(r)
    return ev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None)
    a = ap.parse_args()
    work = Path(a.work or tempfile.mkdtemp(prefix="refe_grow_"))
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    base = ["--backbone", "vits16", "--cpu", "--synthetic", "--batch", "1", "--accum", "2",
            "--epochs", "3", "--grow-scenes", "8", "--log-every", "1", "--lr", "2e-4",
            "--ckpt-every-steps", "1"]
    res, ok = {}, True

    def check(name, cond, detail):
        nonlocal ok
        res[name] = {"pass": bool(cond), "detail": detail}
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}", flush=True)

    # U: static bank, uninterrupted
    bu = work / "bank_U"; make_bank(bu)
    rc, out = sh(base + ["--grow", "--targets", str(bu), "--scorer-targets", str(bu),
                         "--out", str(work / "U")])
    check("U_runs", rc == 0 and "TRAIN_DONE" in out, f"exit {rc}")
    if rc != 0:
        print(out[-3000:])
        print("ZZGROW_" + "FAILEDZZ")
        return 1

    # H: halt mid-epoch 0 (step 2 of 4 = sample 4 of 8), grow the bank, resume
    def halted_then_resumed(tag, env_extra=None, grow=True):
        bd = work / f"bank_{tag}"; make_bank(bd)
        g = ["--grow"] if grow else []
        rc1, o1 = sh(base + g + ["--targets", str(bd), "--scorer-targets", str(bd),
                                  "--out", str(work / tag), "--halt-after-steps", "2"])
        # ⛔ the torn tail only for --grow: without it the NON-grow reader crashes on the fragment
        # (exit 1) before its identity check can refuse (exit 4) -- MEASURED 2026-09-24, the first
        # run of this arm tested JSON parsing instead of the identity guard it is named for
        grow_bank(bd, torn=grow)
        rc2, o2 = sh(base + g + ["--targets", str(bd), "--scorer-targets", str(bd),
                                  "--out", str(work / tag), "--resume"], env_extra)
        return rc1, o1, rc2, o2

    rc1, o1, rc2, o2 = halted_then_resumed("HR")
    check("HR_halted_then_resumed", rc1 == 7 and rc2 == 0 and "TRAIN_DONE" in o2,
          f"halt exit {rc1} (want 7), resume exit {rc2}")
    if rc2 == 0:
        check("G1_epoch0_equals_U", same(work / "U" / "snap_epoch001.pt", work / "HR" / "snap_epoch001.pt"),
              "epoch-0 weights after a mid-epoch bank append == the uninterrupted static-bank run")
        ev = bank_events(work / "HR")
        e0 = [e for e in ev if e["epoch"] == 0]
        e1 = [e for e in ev if e["epoch"] == 1]
        check("G2_growth_seen_at_next_epoch",
              bool(e0) and all(e["n_tuples"] == 8 and e["n_scenes"] == 8 for e in e0)
              and bool(e1) and e1[-1]["n_tuples"] == 12 and e1[-1]["n_scenes"] == 10
              and e1[-1]["per_rank"].get("1") == 2,
              f"epoch 0 {[(e['n_tuples'], e['n_scenes']) for e in e0]}  epoch 1 "
              f"{[(e['n_tuples'], e['n_scenes'], e['per_rank']) for e in e1]}")
        check("G2b_scorer_growth_seen",
              bool(e1) and e1[-1]["scorer_frames"] > (e0[-1]["scorer_frames"] if e0 else 10 ** 9),
              f"scorer frames epoch 0 {[e['scorer_frames'] for e in e0]} -> epoch 1 "
              f"{[e['scorer_frames'] for e in e1]}")
    else:
        print(o2[-3000:])

    # M: the resume ignores the checkpointed snapshot -> epoch 0 must NOT equal U
    rc1, o1, rc2, o2 = halted_then_resumed("M", {"REFE_GROW_MUTATE_FRESH_SNAPSHOT": "1"})
    live = rc2 == 0 and "DELIBERATE REGRESSION" in o2
    check("M_mutation_goes_red",
          live and not same(work / "U" / "snap_epoch001.pt", work / "M" / "snap_epoch001.pt"),
          f"resume exit {rc2}, mutation {'ACTIVE' if live else 'NOT active'}; epoch-0 weights must differ")

    # N: without --grow a grown bank is a different bank -> refused
    rc1, o1, rc2, o2 = halted_then_resumed("N", grow=False)
    check("N_nongrow_refuses_grown_bank", rc1 == 7 and rc2 == 4 and "REFUSING TO RESUME" in o2,
          f"halt exit {rc1}, resume exit {rc2} (want 4)")

    # I: --grow-scenes is part of the identity
    bi = work / "bank_I"; make_bank(bi)
    rc1, _ = sh(base + ["--grow", "--targets", str(bi), "--scorer-targets", str(bi),
                        "--out", str(work / "I"), "--halt-after-steps", "2"])
    bad = [x if x != "8" else "16" for x in base]              # --grow-scenes 8 -> 16
    rc2, o2 = sh(bad + ["--grow", "--targets", str(bi), "--scorer-targets", str(bi),
                        "--out", str(work / "I"), "--resume"])
    check("I_grow_scenes_in_identity", rc1 == 7 and rc2 == 4 and "grow-scenes" in o2,
          f"halt exit {rc1}, resume exit {rc2} (want 4)")

    # T: read_upto on a torn tail and on a shrunken file
    import train as TR
    tp = work / "t.jsonl"
    tp.write_text('{"a": 1}\n{"a": 2}\n{"a": 3, "b"', encoding="utf-8")
    n_all = os.path.getsize(tp)
    got = [json.loads(l)["a"] for l in TR.read_upto(tp, n_all)]
    got_cut = [json.loads(l)["a"] for l in TR.read_upto(tp, len('{"a": 1}\n{"a"'))]
    try:
        TR.read_upto(tp, n_all + 5)
        shrink = False
    except SystemExit:
        shrink = True
    check("T_read_upto", got == [1, 2] and got_cut == [1] and shrink,
          f"full {got} (want [1, 2]), mid-line cut {got_cut} (want [1]), shorter-file refusal {shrink}")

    json.dump(res, open(work / "diag_grow.json", "w"), indent=1)
    print(f"  evidence: {work / 'diag_grow.json'}")
    print("ZZGROW_" + ("EXACT_AND_GUARDS_LIVEZZ" if ok else "FAILEDZZ"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
