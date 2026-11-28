"""Control p2b of card PREREG_V8F_PROOF_ADD27 (note N1, section 2): the tokens encode the GAP ALONE (one fixed direction per token, no closing-speed direction);
the closing speed has to come from the change of the gap over the five frames of the head. Same world, probe, loss, optimiser and held-out read as the built p2 test
(test_tok_probe.py), nothing tuned. CPU only, synthetic. Evidence class: MEASURED (synthetic). Tier T0.

Run: CUDA_VISIBLE_DEVICES=-1 python p2b_run.py [--steps 500,2000,5000] [--arms TK16_s0,TK64_s0] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import test_tok_probe as TT  # noqa: E402  (imports tok_probe, binds the tree, defines the planted world and the held-out read)
import torch  # noqa: E402

tp, TR = TT.tp, TT.TR


def planted_gap_only(seed, n_clips, T_=24, d=1024):
    """The world of TT.planted with the closing-speed term of the tokens removed: tokens = noise + 6 * (gap - 30)/10 * U[k] for a present lead; the closing speed c is NOT in the
    tokens (the V direction is not added). The gap moves with c over the clip (gap = g0 + c * 0.2 * frame), so c is physically in the change of the gap across frames."""
    rng = np.random.default_rng(seed)
    uv = np.random.default_rng(999)                       # same U as the built p2 (same draw order); V is drawn to keep the stream identical and never used
    U = uv.normal(size=(40, d)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    uv.normal(size=(40, d))
    clips = []
    for _ in range(n_clips):
        present = rng.random() < 0.85
        g0, c = rng.uniform(15, 45), rng.uniform(-3, 3)
        gap = g0 + c * 0.2 * np.arange(T_)
        tok = rng.normal(size=(T_, 40, d)).astype(np.float32)
        if present:
            tok += 6.0 * ((gap - 30.0) / 10.0)[:, None, None] * U[None]
        clips.append({"tok": tok.astype(np.float16), "pres": np.full(T_, float(present), np.float32), "gap": np.where(present, gap, np.nan).astype(np.float32),
                      "clos": np.full(T_, c if present else np.nan, np.float32)})
    return clips


def train_steps(clips, steps, arm, out):
    lr_at = TR.LRSchedule(3e-3, 50, steps)
    pr = tp.run_training(TT.batches_of(clips, 1), [arm], steps, lr_at, lr=3e-3, wd=1e-2, clip_grad=1.0, w_loss=1.0, device="cpu", out_dir=str(out))
    return pr[arm].eval()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="500,2000,5000")
    ap.add_argument("--arms", default="TK16_s0,TK64_s0")
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--n-train", type=int, default=16, help="train clips (16 = the built p2 world; other values are a labelled diagnostic, not p2b)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--tmp", default=str(HERE / "tmp" / "p2b"))
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    train, held = planted_gap_only(1, a.n_train),planted_gap_only(2, 8)
    res = {"card": "PREREG_V8F_PROOF_ADD27_N1 section 2 (p2b)", "tier": "T0 (a diagnostic on logged frames, never driving performance)", "evidence_class": "MEASURED (synthetic)",
           "world": "tokens = noise + 6*(gap-30)/10 * U[k]; no closing-speed direction; gap = g0 + c*0.2*frame (the gap moves with the closing speed); %d train / 8 held-out clips of 24 frames" % a.n_train,
           "bar": 0.9, "runs": []}
    for arm in a.arms.split(","):
        for st in [int(x) for x in a.steps.split(",")]:
            t0 = time.time()
            pr = train_steps(train, st, arm, Path(a.tmp) / ("%s_%d" % (arm, st)))
            rg, rc = TT.heldout_r2(pr, held)
            row = {"arm": arm, "steps": st, "heldout_r2_gap": rg, "heldout_r2_closing": rc, "both_above_0_9": bool(rg > 0.9 and rc > 0.9), "seconds": round(time.time() - t0, 1)}
            print(json.dumps(row), flush=True)
            res["runs"].append(row)
    by_arm = {}
    for r in res["runs"]:
        by_arm.setdefault(r["arm"], []).append(r["both_above_0_9"])
    res["p2b_pass_per_arm"] = {k: bool(any(v)) for k, v in by_arm.items()}
    res["p2b_pass"] = bool(all(res["p2b_pass_per_arm"].values())) if by_arm else False
    # physical readability of the closing speed in this world (an oracle read, not the probe): the gap signal summed over the 40 tokens of a frame, projected on U, and the
    # least-squares slope of that estimate over the five frames, against the label c
    rng = np.random.default_rng(5)
    uv = np.random.default_rng(999)
    U = uv.normal(size=(40, 1024)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    est_c, lab_c, est_g, lab_g = [], [], [], []
    for c in held:
        if not np.isfinite(c["gap"][0]):
            continue
        x = c["tok"].astype(np.float32)                                          # [T, 40, d]
        proj = np.einsum("tkd,kd->tk", x, U).mean(1) / 6.0 * 10.0 + 30.0         # per-frame gap estimate from the signal direction (known U), in metres
        for s in range(0, len(proj) - 5 + 1, 2):
            f = np.arange(5)
            slope = np.polyfit(f, proj[s:s + 5], 1)[0] / 0.2
            est_c.append(slope); lab_c.append(c["clos"][s + 4]); est_g.append(proj[s + 4]); lab_g.append(c["gap"][s + 4])
    res["oracle_readability"] = {"note": "per-frame gap estimate = mean over the 40 tokens of <token, U_k> (U known), closing = slope of that estimate over 5 frames; an upper-reference read of what the plant allows, not the probe",
                                 "r2_gap": TT.r2(np.array(lab_g), np.array(est_g)), "r2_closing": TT.r2(np.array(lab_c), np.array(est_c)), "n": len(lab_c)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / ("P2B_RESULT%s.json" % a.tag)).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    lines = ["# P2B_RESULT: control p2b of card ADD27 (note N1 section 2)", "",
             "Tier T0 (a diagnostic on logged frames, never driving performance). Evidence class: MEASURED (synthetic). STRATEGIC: PAUSED (PI item 25).", "",
             "World: " + res["world"] + ".", "", "| arm | steps | held-out R2 gap | held-out R2 closing | both above 0.9 |", "|---|---|---|---|---|"]
    for r in res["runs"]:
        lines.append("| %s | %d | %+.3f | %+.3f | %s |" % (r["arm"], r["steps"], r["heldout_r2_gap"], r["heldout_r2_closing"], "yes" if r["both_above_0_9"] else "no"))
    lines += ["", "p2b passes for an arm if both R2 are above 0.9 at one of the step counts. Per arm: %s. p2b overall: %s." % (json.dumps(res["p2b_pass_per_arm"]), "PASS" if res["p2b_pass"] else "FAIL"), "",
              "Oracle readability of this plant (not the probe): R2 gap %+.3f, R2 closing %+.3f on n=%d held-out windows." % (res["oracle_readability"]["r2_gap"], res["oracle_readability"]["r2_closing"], res["oracle_readability"]["n"]), ""]
    (Path(a.out) / ("P2B_RESULT%s.md" % a.tag)).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
