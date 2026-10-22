#!/usr/bin/env python3
"""TACTICAL "selection quality" rows of eval/PREREG_P6.md: random / actual / best-of-64 on the SAME tokens, the random control mandatory (P6-READ-GAPS G2, 2026-10-22).

  actual      PDMS of the proposal the model PICKED (the harness's per-token score of the arm: the primary endpoint's own number)
  random      the exact expectation of a skill-free selector = the MEAN over the 64 proposals' PDMS, per token (C3's "random", eval/RESULT_C3_SELECTION_HEADROOM.md)
  best-of-64  the per-token MAX over the 64 proposals' PDMS (a ceiling no deployable scorer reaches; PREREG_P6's "best-of-64")
  All x100, T1 (NAVSIM v1.1 PDMS, non-reactive pseudo-simulation). Level CIs and the paired differences (actual - random, best - actual, best - random) use
  the registered estimator: paired log-cluster bootstrap over the logs, 10,000 resamples, seed 20260927 (proxy_eval.boot). It answers "another draw of LOGS".

PER-PROPOSAL PDMS needs the harness: one single-proposal seam per index k, each scored by the UNCHANGED banked harness (64 x the scoring cost of one read, CPU only;
~777k token-columns on the full navtest = about 4 to 6.4 h on 6 workers, MEASURED for P2: raw/2026-10-16-p2-full-table/PLAN.md). The proposals come FREE from the
read's own seam run: `stage2_sft.py --dump-proposals` (no second GPU pass).

    python eval/selection_quality.py seams  --dump <seams/stage2_N.proposals.npz> --name N        # 64 column seams
    python eval/selection_quality.py score  --dump ... --name N --workers 4 [--tokens <W3 json>]  # 64 harness columns (resumable; whole-filter path when the dump is the full set)
    python eval/selection_quality.py table  --dump ... --name N --arm-csv <the arm's harness csv>  # assemble + gates -> table_N.npz
    python eval/selection_quality.py read   --table table_N.npz --arm-csv ... --out selection_quality_N.json
    python eval/selection_quality.py level  --arm-csv <the arm's harness csv> [--out level_N.json]   # the arm's OWN level CI (the PREREG goal-bar test): csv only, no 64 columns, seconds
Every step is resumable and prints ZZSELQ_* markers. Gates (any failure = INVALID, never a number): all 64 columns PASS with every row valid; the table at the pick equals the arm's
own harness csv bit for bit (the proposals ARE the ones that were scored); no NaN; random < actual (a random selector beating the scorer is an instrument failure, as in C3).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import proxy_eval as PE  # noqa: E402  (the registered estimator: boot, read_csv)

M = 64
DATA = Path(os.environ.get("REFE_NAVTEST_DIR", "E:/Projects/TanitAD/data/refe_navtest"))
PY_D = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
HARNESS = HERE / "harness"
EXPORT = "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
N_BOOT, SEED = 10000, 20260927
SUB = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance")
TIER = "T1 = NAVSIM v1.1 PDMS, an ego pseudo-simulation of an open-loop plan against logged agents (self-action open loop, not closed loop)"
ESTIMATOR = ("paired log-cluster bootstrap over the logs (cluster unit log_name), 10,000 resamples, 95 %, seed 20260927 (proxy_eval.boot). It answers 'another draw of LOGS' "
             "and nothing else: training variance (one seed per arm, H-ESTIM-SEED-1) is unmeasured; inference is deterministic.")


def label_k(name: str, k: int) -> str:
    return f"refe_selq_{name}_p{k:02d}"                      # the harness refuses labels not starting with "refe"


def seam_k(name: str, k: int, root: Path = DATA) -> Path:
    return root / "seams" / "selq" / name / f"{label_k(name, k)}.npz"


def csv_k(name: str, k: int, root: Path = DATA) -> Path:
    return root / "score" / label_k(name, k) / f"{label_k(name, k)}.csv"


# ------------------------------------------------------------------ seams
def write_column_seams(dump_npz, name: str, root: Path = DATA) -> int:
    """one seam per proposal index k in the SAME layout the seam writes (token, fingerprint, poses [N,8,3], sampling, arm); poses are the dump's proposals[:, k]"""
    with np.load(dump_npz, allow_pickle=False) as D:
        P, token, fp, sampling = D["proposals"], D["token"], D["fingerprint"], D["sampling"]
    assert P.ndim == 4 and P.shape[1] == M, P.shape
    seam_k(name, 0, root).parent.mkdir(parents=True, exist_ok=True)
    for k in range(M):
        p = seam_k(name, k, root)
        if not p.exists():
            np.savez(p, token=token, fingerprint=fp, poses=P[:, k].astype(np.float32), sampling=sampling, arm=np.array(f"REFe_selq_{name}_p{k:02d}"))
    return int(P.shape[0])


# ------------------------------------------------------------------ the 64 harness columns
def status_of(txt: str):
    for line in reversed(txt.splitlines()):
        if line.startswith("{") and '"status"' in line:
            return json.loads(line)
    return None


def default_runner(label: str, seam: Path, tokens: str | None, out: Path, log: Path) -> str:
    """the banked harness, exactly as stage2_sft.py calls it (cwd = eval/harness, RUN_V1_FIX1_ALLOW_DRIFT for the repointed run_v1.py)"""
    cmd = [PY_D, "score_e.py", "--label", label, "--seam", str(seam), *(["--tokens", tokens] if tokens else []), "--out", str(out)]
    with open(log, "w", encoding="utf-8") as f:
        subprocess.call(cmd, cwd=str(HARNESS), env=dict(os.environ, PYTHONIOENCODING="utf-8", RUN_V1_FIX1_ALLOW_DRIFT="1"), stdout=f, stderr=subprocess.STDOUT)
    return open(log, encoding="utf-8", errors="replace").read()


def score_columns(name: str, n_rows: int, workers: int = 4, tokens: str | None = None, only=None, runner=default_runner, root: Path = DATA, logdir: Path | None = None) -> dict:
    """score every column through the harness; resumable (a column whose log says PASS with n_rows valid rows is skipped). Returns {"ok", "failed", "n_pass"}"""
    logdir = logdir or (root / "score" / f"_selq_{name}_logs")
    logdir.mkdir(parents=True, exist_ok=True)
    ks = list(range(M)) if only is None else list(only)

    def one(k):
        lp = logdir / f"p{k:02d}.log"
        if csv_k(name, k, root).exists() and lp.exists():
            st = status_of(lp.read_text(encoding="utf-8", errors="replace"))
            if st and st.get("status") == "PASS" and st.get("csv_valid_rows") == n_rows:
                return k, st
        txt = runner(label_k(name, k), seam_k(name, k, root), tokens, root / "score", lp)
        return k, status_of(txt)

    status = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for k, st in ex.map(one, ks):
            status[k] = st
    bad = [k for k in ks if not (status.get(k) and status[k].get("status") == "PASS" and status[k].get("csv_valid_rows") == n_rows)]
    return {"ok": not bad, "failed": bad, "n_pass": len(ks) - len(bad), "n_columns": len(ks)}


# ------------------------------------------------------------------ the table
def read_cols(p):
    rows = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["token"] == "average":
            continue
        rows[r["token"]] = (r["valid"] == "True", float(r["score"]), tuple(float(r[c]) for c in SUB))
    return rows


def assemble_table(dump_npz, name: str, arm_csv, out_npz, token_log: dict, root: Path = DATA) -> dict:
    """pdms [N,64] (0..1) from the 64 column csvs + the gates; writes out_npz. Gate G2: pdms at the pick == the arm's own harness csv, bit for bit."""
    with np.load(dump_npz, allow_pickle=False) as D:
        tok = [str(t) for t in D["token"]]
        pick = D["pick"].astype(np.int64)
        logits = D["logits"] if "logits" in D.files else np.zeros(0)
    N = len(tok)
    pdms = np.full((N, M), np.nan)
    valid = np.zeros((N, M), dtype=bool)
    sub = np.full((N, M, 6), np.nan)
    for k in range(M):
        rows = read_cols(csv_k(name, k, root))
        for i, t in enumerate(tok):
            v, s, c = rows[t]
            valid[i, k], pdms[i, k], sub[i, k] = v, s, c
    arm = {t: float(r["score"]) for t, r in PE.read_csv(arm_csv).items()}
    at_pick = pdms[np.arange(N), pick]
    mism = [t for i, t in enumerate(tok) if t in arm and at_pick[i] != arm[t]]
    gates = {"G3_all_cells_valid_finite": bool(valid.all() and np.isfinite(pdms).all()),
             "G2_pdms_at_pick_equals_arm_csv": {"ok": bool(not mism and all(t in arm for t in tok)), "n_mismatch": len(mism), "first": mism[:3], "n_tokens": N}}
    gates["ok"] = bool(gates["G3_all_cells_valid_finite"] and gates["G2_pdms_at_pick_equals_arm_csv"]["ok"])
    np.savez(out_npz, token=np.array(tok), log=np.array([token_log[t] for t in tok]), pdms=pdms, sub=sub, valid=valid, pick=pick, logits=logits)
    return gates


# ------------------------------------------------------------------ the read
def read_selection_quality(pdms, pick, tokens, logs, n: int = N_BOOT, seed: int = SEED) -> dict:
    """pdms [N,64] in 0..1, pick [N]. Returns the PREREG TACTICAL selection-quality block (x100) with paired log-cluster CIs"""
    pdms = np.asarray(pdms, dtype=np.float64)
    N = pdms.shape[0]
    assert pdms.ndim == 2 and pdms.shape[1] == M and len(tokens) == N == len(pick) == len(logs)
    assert np.isfinite(pdms).all(), "NaN in the per-proposal table: not a read"
    tl = {t: l for t, l in zip(tokens, logs)}
    idx = np.arange(N)
    per = {"actual": pdms[idx, pick] * 100.0, "random": pdms.mean(1) * 100.0, "best_of_64": pdms.max(1) * 100.0}

    def stat(vec):
        mu, lo, hi = PE.boot({t: float(v) for t, v in zip(tokens, vec)}, tl, n=n, seed=seed)
        return {"mean": mu, "ci95": [lo, hi]}
    levels = {k: stat(v) for k, v in per.items()}
    diffs = {"actual_minus_random": stat(per["actual"] - per["random"]), "best_minus_actual": stat(per["best_of_64"] - per["actual"]),
             "best_minus_random": stat(per["best_of_64"] - per["random"])}
    sk = diffs["actual_minus_random"]
    return {"n_tokens": N, "n_logs": len(set(logs)), "n_proposals": M, "levels_x100": levels, "paired_differences_x100": diffs,
            "random_control": {"definition": "mean over the 64 proposals' PDMS per token (the exact expectation of a skill-free selector)",
                               "actual_beats_random": bool(sk["ci95"][0] > 0), "random_ge_actual_point": bool(levels["random"]["mean"] >= levels["actual"]["mean"])},
            "n_pick_zero": int((per["actual"] == 0).sum()), "n_best_zero": int((per["best_of_64"] == 0).sum()),
            "n_best_strictly_above_pick": int((per["best_of_64"] > per["actual"]).sum()),
            "estimator": ESTIMATOR, "tier": TIER, "evidence_class": "MEASURED when built from a P6 dump and its 64 harness columns"}


def level_from_csv(arm_csv, token_log: dict, n: int = N_BOOT, seed: int = SEED) -> dict:
    """the arm's level (mean PDMS x100) with the registered paired log-cluster bootstrap CI, from its harness csv alone. This is the number PREREG_P6's goal-bar test (94.55) needs;
    it does NOT depend on the 64-column selection-quality table (4 to 6.4 h). Every csv token must have a log; an unknown token is an error, never a silent drop."""
    rows = PE.read_csv(arm_csv)
    unknown = [t for t in rows if t not in token_log]
    if not rows or unknown:
        raise ValueError(f"{arm_csv}: {len(rows)} tokens, {len(unknown)} without a log (e.g. {unknown[:3]}): not a level")
    per = {t: float(r["score"]) * 100.0 for t, r in rows.items()}
    mu, lo, hi = PE.boot(per, {t: token_log[t] for t in per}, n=n, seed=seed)
    return {"arm_csv": str(arm_csv), "n_tokens": len(per), "n_logs": len({token_log[t] for t in per}), "level_x100": {"mean": mu, "ci95": [lo, hi]},
            "estimator": ESTIMATOR, "tier": TIER, "evidence_class": "MEASURED on the arm's own harness csv"}


def read_table(table_npz, out=None, arm_csv=None) -> dict:
    with np.load(table_npz, allow_pickle=False) as z:
        r = read_selection_quality(z["pdms"], z["pick"], [str(t) for t in z["token"]], [str(l) for l in z["log"]])
    r["table"] = str(table_npz)
    if r["random_control"]["random_ge_actual_point"]:
        r["INVALID"] = "random >= actual in the point estimate: an instrument failure (C3's rule), not a result"
    if out:
        Path(out).write_text(json.dumps(r, indent=1), encoding="utf-8")
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("seams", "score", "table"):
        p = sp.add_parser(c)
        p.add_argument("--dump", required=True)
        p.add_argument("--name", required=True)
    sp.choices["score"].add_argument("--workers", type=int, default=4)
    sp.choices["score"].add_argument("--tokens", default=None, help="W3 token json (omit on the whole navtest: the harness's whole-filter path)")
    sp.choices["score"].add_argument("--only", type=int, nargs="*", default=None)
    sp.choices["table"].add_argument("--arm-csv", required=True)
    sp.choices["table"].add_argument("--out", default=None)
    lv_ = sp.add_parser("level")
    lv_.add_argument("--arm-csv", required=True)
    lv_.add_argument("--out", default=None)
    r = sp.add_parser("read")
    r.add_argument("--table", required=True)
    r.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "seams":
        print("ZZSELQ_SEAMS", a.name, write_column_seams(a.dump, a.name))
        return 0
    if a.cmd == "score":
        n = write_column_seams(a.dump, a.name)
        t0 = time.time()
        res = score_columns(a.name, n, a.workers, a.tokens, a.only)
        print("ZZSELQ_SCORE", a.name, "OK" if res["ok"] else f"FAIL {res['failed'][:10]}", f"{res['n_pass']}/{res['n_columns']}", f"{time.time() - t0:.0f}s")
        return 0 if res["ok"] else 1
    if a.cmd == "table":
        with np.load(a.dump, allow_pickle=False) as D:
            dump_tokens = [str(x) for x in D["token"]]
        import gzip
        exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
        out = a.out or str(DATA / "seams" / "selq" / a.name / f"table_{a.name}.npz")
        g = assemble_table(a.dump, a.name, a.arm_csv, out, {t: exp[t]["log_name"] for t in dump_tokens})
        print("ZZSELQ_TABLE", a.name, "OK" if g["ok"] else "INVALID", json.dumps(g), out)
        return 0 if g["ok"] else 1
    if a.cmd == "level":
        import gzip
        exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
        res = level_from_csv(a.arm_csv, {t: v["log_name"] for t, v in exp.items()})
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        lv = res["level_x100"]
        print("ZZSELQ_LEVEL", f"{lv['mean']:.4f} [{lv['ci95'][0]:.4f}, {lv['ci95'][1]:.4f}] n={res['n_tokens']} logs={res['n_logs']}")
        return 0
    res = read_table(a.table, a.out)
    lv = res["levels_x100"]
    print("ZZSELQ_READ", f"actual {lv['actual']['mean']:.4f} random {lv['random']['mean']:.4f} best64 {lv['best_of_64']['mean']:.4f}", "INVALID" if "INVALID" in res else "ok")
    return 1 if "INVALID" in res else 0


if __name__ == "__main__":
    sys.exit(main())
