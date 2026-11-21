#!/usr/bin/env python3
"""r9e2_box_paired.py -- writes BOX_PAIRED.json: the PAIRED box AP@2 m record (K2 - K0) that clause (b) of ``stack/scripts/r9_e2_verdict.py`` reads.  NEW file (E2 clause (b) amendment).

PI ruling 2026-11-17 ("Not separated worse"): clause (b) is no longer ``K2 point > K0 point``; it PASSES unless the paired episode-cluster bootstrap interval of K2 - K0 on
``boxes.ap2m_all`` is ENTIRELY below 0 (higher is better).  The point delta is reported as information.  Existing banked verdicts are not re-scored.

    python r9e2_box_paired.py --rd <read dir> --split DEV_HOLD_SPLIT.json --eval-script <stack/scripts/r9_e1_eval.py> --out <read dir>/BOX_PAIRED.json
    python r9e2_box_paired.py --rd <read dir> --raw <r9_e1_eval paired output json>  --out ...        # extract only (no re-run)

It runs ``r9_e1_eval.py --packs <rd>/dump_K2/perc_det.pkl --packs-b <rd>/dump_K0/perc_det.pkl --windows <split> --split dev --gate <v> --boot 2000 --seed 0`` (NO new estimator:
``taniteval.ci.paired_stat_cluster_bootstrap``, the SAME call the single-arm ``EVAL_*.json`` read is built from) and lifts ``paired_vs_B.boxes.ap2m_all`` into a flat record.
GATE: the roll's ``--gates-json`` file is Thor-only; both arms' ``EVAL_*.json`` record the P=R gate they were read at (``gate.value`` + the gates file's sha256), so the paired read
reuses that VALUE (``--gate``) after checking K2's and K0's are identical -- the AP@2 m column does not depend on it, but the same windows / gate keep the read identical to the roll's.
Perception is deterministic at inference, so ONE record serves BOTH inference seeds (recorded as ``serves_seeds``); r9e2_inputs.py puts the same path on seeds 0 and 1.
The interval answers *another draw of EPISODES* only (H-ESTIM-SEED-1).  CPU only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

SCHEMA = "r9e2-box-paired-v1"
METRIC = "boxes.ap2m_all"
FILE_NAME = "BOX_PAIRED.json"


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _fin(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def read_gate(eval_k2: dict, eval_k0: dict) -> tuple:
    """-> (gate value, record).  REFUSES when an arm has no recorded gate or the two arms were read at different gates."""
    g2, g0 = (eval_k2 or {}).get("gate"), (eval_k0 or {}).get("gate")
    if not (isinstance(g2, dict) and isinstance(g0, dict) and _fin(g2.get("value")) and _fin(g0.get("value"))):
        raise SystemExit("[r9e2_box_paired] REFUSED: EVAL_K2.json / EVAL_K0.json record no finite P=R gate")
    if float(g2["value"]) != float(g0["value"]) or g2.get("sha256") != g0.get("sha256"):
        raise SystemExit(f"[r9e2_box_paired] REFUSED: K2 and K0 were read at different gates ({g2!r} vs {g0!r})")
    return float(g2["value"]), {k: g2.get(k) for k in ("source", "sha256", "value")}


def extract(raw: dict) -> dict:
    """-> the flat paired record from an ``r9_e1_eval.py --packs-b`` output.  REFUSES when the paired box block is absent."""
    row = ((raw or {}).get("paired_vs_B") or {}).get("boxes", {})
    row = row.get("ap2m_all") if isinstance(row, dict) else None
    if not isinstance(row, dict) or "delta" not in row:
        raise SystemExit("[r9e2_box_paired] REFUSED: no paired_vs_B.boxes.ap2m_all in the eval output (was --packs-b given?)")
    rec = {"schema": SCHEMA, "metric": METRIC, "better": "higher", "arm": "K2 - K0 (a = K2, b = K0)"}
    for k in ("delta", "lo", "hi", "separated", "a", "b", "n_windows", "n_episodes", "n_boot", "p_delta_gt0", "estimator", "undefined", "degenerate", "n_nonfinite_draws"):
        if k in row:
            rec[k] = row[k]
    rec["alignment"] = raw["paired_vs_B"].get("alignment")
    rec["serves_seeds"] = ["0", "1"]
    rec["note"] = ("perception is deterministic at inference: this ONE paired record serves BOTH inference seeds; the interval answers another draw of EPISODES, "
                   "not another TRAINING run (H-ESTIM-SEED-1)")
    return rec


def run_paired(rd, split, eval_script, gate, boot=2000, seed=0, py=None) -> dict:
    rd, eval_script = Path(rd), Path(eval_script)
    ka, kb = rd / "dump_K2" / "perc_det.pkl", rd / "dump_K0" / "perc_det.pkl"
    for p in (ka, kb, Path(split), eval_script):
        if not p.is_file():
            raise SystemExit(f"[r9e2_box_paired] REFUSED: {p} is missing")
    raw_path = rd / "BOX_PAIRED_raw.json"
    stack = eval_script.resolve().parents[1]
    paths = [str(stack)] + [str(c) for c in (stack.parent / "taniteval",) if c.is_dir()]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONIOENCODING="utf-8",
               PYTHONPATH=os.pathsep.join(paths + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])))
    cmd = [py or sys.executable, "-X", "utf8", str(eval_script), "--packs", str(ka), "--packs-b", str(kb), "--windows", str(split), "--split", "dev",
           "--gate", repr(float(gate)), "--boot", str(int(boot)), "--seed", str(int(seed)), "--out", str(raw_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env, stdin=subprocess.DEVNULL, timeout=10800)  # 3 h: the same paired eval took 39 min on Thor (E2-TAIL paired_diag, 2026-11-17)
    if r.returncode != 0 or not raw_path.is_file():
        raise SystemExit(f"[r9e2_box_paired] FAILED rc={r.returncode}: {r.stderr[-800:]}")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["_sha"] = {"packs_k2": _sha(ka), "packs_k0": _sha(kb), "windows": _sha(split)}
    return raw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rd", required=True, help="read dir: EVAL_K2.json, EVAL_K0.json, dump_K2/perc_det.pkl, dump_K0/perc_det.pkl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="", help="DEV_HOLD_SPLIT.json (the roll's R9E2_SPLIT)")
    ap.add_argument("--eval-script", default="", help="stack/scripts/r9_e1_eval.py")
    ap.add_argument("--raw", default="", help="an existing r9_e1_eval.py --packs-b output: extract only")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--py", default="")
    a = ap.parse_args(argv)
    rd = Path(a.rd)
    try:
        ev2 = json.loads((rd / "EVAL_K2.json").read_text(encoding="utf-8"))
        ev0 = json.loads((rd / "EVAL_K0.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SystemExit(f"[r9e2_box_paired] REFUSED: cannot read EVAL_K2.json / EVAL_K0.json in {rd} ({e})") from None
    gate, grec = read_gate(ev2, ev0)
    if a.raw:
        raw = json.loads(Path(a.raw).read_text(encoding="utf-8"))
    else:
        if not (a.split and a.eval_script):
            raise SystemExit("[r9e2_box_paired] REFUSED: --split and --eval-script are required unless --raw is given")
        raw = run_paired(rd, a.split, a.eval_script, gate, a.boot, a.seed, a.py or None)
    rec = extract(raw)
    rec["gate"] = grec
    rec["split"] = raw.get("split")
    rec["sha256"] = raw.get("_sha")
    rec["point_check"] = {"eval_k2_point": (ev2.get("boxes") or {}).get("ap2m_all", {}).get("point"), "eval_k0_point": (ev0.get("boxes") or {}).get("ap2m_all", {}).get("point")}
    Path(a.out).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(f"[r9e2_box_paired] wrote {a.out}: delta={rec.get('delta')} ci=[{rec.get('lo')}, {rec.get('hi')}] separated={rec.get('separated')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
