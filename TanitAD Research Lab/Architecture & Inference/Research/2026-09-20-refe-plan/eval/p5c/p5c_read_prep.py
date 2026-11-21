#!/usr/bin/env python3
"""P5-C read instrument B5 + B6: seam/dump -> plans npz, ONE transp_score pass over {C1, C0, base, human}, and the model-arm
identity control (tracked PDMS column == the pinned harness csv, bar 1e-6, with a RED arm).

PREREG_P5C section 5 (population, scorers) and section 10.5 W-TRANSP (i). Implements the card as written; it decides nothing.

Subcommands
  convert   --arm NAME=SEAM.npz[,DUMP.npz] ... --outdir D [--subset N --subset-seed S]
              -> D/plans_<NAME>.npz (tokens, plans8 = the seam's poses, plans20 = the dump's `pick20` when a dump with pick20 is given)
              -> D/convert_report.json  (which FORM each arm used: "pick20+plans8" or "plans8", and the pick20 control)
            Form rule: pick20 whenever the dump carries it (the registered form: the transparent arm is built from the 20-pose plan at
            dt 0.2); otherwise plans8 only (transparent arm built from the 8 seam poses at dt 0.5, ~1.5 cm coarser, NOT the registered form).
            The tracked arm is ALWAYS the seam's poses (== to_navsim(pick20) bit-for-bit, checked), so it is the harness's own input.
  score     --plans-dir D --out-dir O [--procs 2] [--limit n] [--transp-script path] [--python navsim-crun python]
              one transp_score pass over every plans_*.npz in D (all arms + the human arm), <= 2 processes, metric-cache route always on
  identity  --rows O/rows_*.jsonl --arm NAME --csv HARNESS.csv --plans D/plans_NAME.npz --out result.json [--bar 1e-6]
  red       --plans-dir D --arm NAME --csv H.csv --out-dir O --n 60
              mutation arm: transp_score with the cache-routing line deleted must FAIL the identity check (exit 0 iff it did)
Evidence class of anything this prints: MEASURED on the run that printed it. There is no navtest number about P5-C.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BAR = 1e-6
COL = {"nc": "no_at_fault_collisions", "dac": "drivable_area_compliance", "ddc": "driving_direction_compliance", "ep": "ego_progress",
       "ttc": "time_to_collision_within_bound", "c": "comfort", "pdms": "score"}
ROUTE_LINE = "parts = load_cache_parts(a.metric_cache_root, lg, tok) if a.metric_cache_root else PT.metric_cache_parts(sc)"
# pinned read-only location (no first-existing-root search; E:/Archive/devbox-C is never used)
NAVSIM_ROOT = os.environ.get("TANITAD_NAVSIM_ROOT", "F:/Archive/devbox-C/navsim")


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def refe_dir() -> str:
    for d in (os.environ.get("P5C_REFE_DIR", ""), os.path.join(HERE, "..", "..", "refe")):
        if d and os.path.exists(os.path.join(d, "navsim_dac.py")):
            return os.path.abspath(d)
    raise SystemExit("REFUSING: navsim_dac.py not found (set P5C_REFE_DIR to PK/refe)")


def to_navsim(p20: np.ndarray) -> np.ndarray:
    """navsim_dac.to_navsim (numpy only), the conversion transp_score and the seam both use."""
    d = refe_dir()
    if d not in sys.path:
        sys.path.insert(0, d)
    import navsim_dac as ND
    return ND.to_navsim(p20)


USED_POSE_IDX = [1, 2, 4, 6, 7, 9, 11, 12, 14, 16, 17, 19]     # the pick20 poses to_navsim reads (t = 0.5 s lies between idx 1 and 2, ...); the other 8 are NOT covered by the control


def pick20_equals_seam(pick20: np.ndarray, seam_poses: np.ndarray) -> bool:
    """P7-READ-ROUTE B3 control: float32(to_navsim(pick20[i])) == seam row i, bit for bit."""
    pick20, seam_poses = np.asarray(pick20), np.asarray(seam_poses)
    if pick20.ndim != 3 or pick20.shape[1:] != (20, 3) or len(pick20) != len(seam_poses):
        return False
    return all(np.array_equal(to_navsim(pick20[i]).astype(np.float32), seam_poses[i]) for i in range(len(pick20)))


def subset_tokens(tokens, n: int, seed: int):
    """stable, seeded subset: the n tokens with the smallest sha1(seed:token), returned sorted. n <= 0 -> all (sorted)."""
    toks = sorted(tokens)
    if n <= 0 or n >= len(toks):
        return toks
    key = sorted(toks, key=lambda t: hashlib.sha1(f"{seed}:{t}".encode()).hexdigest())[:n]
    return sorted(key)


def build_arm(seam_path: str, dump_path, keep):
    """-> (plans dict {tokens, plans8, [plans20]}, report). Refuses on any token / shape / control problem (fail loud)."""
    z = np.load(seam_path, allow_pickle=False)
    toks = [str(t) for t in z["token"]]
    poses = np.asarray(z["poses"])
    if poses.ndim != 3 or poses.shape[1:] != (8, 3) or len(toks) != len(poses):
        raise SystemExit(f"REFUSING {seam_path}: seam poses shape {poses.shape} vs {len(toks)} tokens")
    if len(set(toks)) != len(toks):
        raise SystemExit(f"REFUSING {seam_path}: duplicate tokens")
    if not np.isfinite(poses).all():
        raise SystemExit(f"REFUSING {seam_path}: non-finite seam poses")
    idx = {t: i for i, t in enumerate(toks)}
    sel = keep if keep is not None else sorted(toks)
    miss = [t for t in sel if t not in idx]
    if miss:
        raise SystemExit(f"REFUSING {seam_path}: {len(miss)} requested tokens absent (first {miss[0]})")
    rows = np.array([idx[t] for t in sel])
    out = {"tokens": np.array(sel), "plans8": poses[rows].astype(np.float64)}
    rep = {"seam": os.path.abspath(seam_path), "seam_sha256": sha256_file(seam_path), "n": len(sel), "form": "plans8",
           "form_note": "no pick20: transparent arm built from the 8 seam poses at dt 0.5 (NOT the registered 20-pose form)"}
    if dump_path:
        d = np.load(dump_path, allow_pickle=False)
        if "pick20" not in d.files:
            rep["dump_without_pick20"] = os.path.abspath(dump_path)
        else:
            di = {str(t): i for i, t in enumerate(d["token"])}
            if any(t not in di for t in sel):
                raise SystemExit(f"REFUSING {dump_path}: the dump lacks tokens the seam has")
            p20 = np.asarray(d["pick20"])[np.array([di[t] for t in sel])]
            ok = pick20_equals_seam(p20, poses[rows])
            rep["pick20_control_bitexact"] = bool(ok)
            rep["pick20_unverified_poses"] = [k for k in range(20) if k not in USED_POSE_IDX]
            rep["pick20_control_note"] = "bit-exact on the 12 poses to_navsim reads; the 8 listed poses feed only the transparent arm and are not covered by this control"
            if not ok:
                raise SystemExit(f"REFUSING {dump_path}: to_navsim(pick20) != the seam's poses bit-for-bit (wrong dump for this seam)")
            out["plans20"] = p20.astype(np.float64)
            rep.update(form="pick20+plans8", form_note="registered form: tracked = seam poses, transparent = cubic spline through pick20 at dt 0.2",
                       dump=os.path.abspath(dump_path), dump_sha256=sha256_file(dump_path))
    return out, rep


def cmd_convert(a) -> int:
    specs = []
    for s in a.arm:
        nm, rest = s.split("=", 1)
        seam, _, dump = rest.partition(",")
        specs.append((nm, seam, dump or None))
    common = None
    for _, seam, _d in specs:
        ts = set(str(t) for t in np.load(seam, allow_pickle=False)["token"])
        common = ts if common is None else common & ts
    keep = subset_tokens(common, a.subset, a.subset_seed)
    os.makedirs(a.outdir, exist_ok=True)
    reps = {}
    for nm, seam, dump in specs:
        plans, rep = build_arm(seam, dump, keep)
        np.savez(os.path.join(a.outdir, f"plans_{nm}.npz"), **plans)
        reps[nm] = rep
    forms = {r["form"] for r in reps.values()}
    summary = {"arms": reps, "n_tokens": len(keep), "n_common_before_subset": len(common),
               "subset": {"n": a.subset, "seed": a.subset_seed, "rule": "smallest sha1(seed:token), sorted"} if a.subset > 0 else None,
               "forms_used": sorted(forms), "mixed_forms": len(forms) > 1,
               "tokens_sha256": hashlib.sha256("\n".join(keep).encode()).hexdigest()}
    if len(forms) > 1:
        print("WARNING mixed forms across arms: a pick20 arm vs a plans8 arm is NOT like-for-like (transparent dt differs)", file=sys.stderr)
    with open(os.path.join(a.outdir, "convert_report.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1)
    print("ZZP5CR_CONVERT_OK", summary["n_tokens"], "tokens; forms:", ",".join(sorted(forms)))
    return 0


# --------------------------------------------------------------------------------------------- score
def build_score_cmds(plans_dir: str, out_dir: str, procs: int = 2, limit: int = 0, transp_script=None, python=None,
                     human: bool = True, no_fam: bool = False, cache: bool = True) -> list:
    """argv lists, one per shard (<= 2 processes). Pure (tested). `cache=False` exists ONLY for the RED arm; the real path always passes
    --metric-cache-root, because a transp_score pass without it is the not-harness-identical DB route."""
    if procs > 2:
        raise SystemExit("REFUSING: at most 2 scoring processes on the shared dev box")
    ts = transp_script or os.path.join(HERE, "..", "instruments", "transp_score.py")
    toklog = f"{NAVSIM_ROOT}/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
    plans = sorted(glob.glob(os.path.join(plans_dir, "plans_*.npz")))
    if not plans:
        raise SystemExit(f"REFUSING: no plans_*.npz in {plans_dir}")
    cmds = []
    for i in range(procs):
        c = [python or sys.executable, ts, "--refe", refe_dir(), "--split", "navtest", "--token-log", toklog]
        if cache:
            c += ["--metric-cache-root", f"{NAVSIM_ROOT}/exp/w3_navtest_v1/metric_cache_navtest"]
        c += ["--out", os.path.join(out_dir, f"rows_s{i}of{procs}.jsonl"), "--shard", f"{i}/{procs}", "--plans"]
        c += [f"{os.path.basename(p)[6:-4]}={p}" for p in plans]
        if human:
            c.append("--human-arm")
        if no_fam:
            c.append("--no-fam")
        if limit:
            c += ["--limit", str(limit)]
        cmds.append(c)
    return cmds


def score_env() -> dict:
    nav = f"{NAVSIM_ROOT}/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
    return dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                PYTHONPATH=os.pathsep.join([nav, refe_dir()]), TANITAD_NAVSIM_ROOT=NAVSIM_ROOT)


def cmd_score(a) -> int:
    os.makedirs(a.out_dir, exist_ok=True)
    cmds = build_score_cmds(a.plans_dir, a.out_dir, a.procs, a.limit, a.transp_script, a.python, not a.no_human, a.no_fam)
    ps = []
    for i, c in enumerate(cmds):
        lg = open(os.path.join(a.out_dir, f"transp_s{i}.log"), "w", encoding="utf-8")
        ps.append(subprocess.Popen(c, env=score_env(), stdin=subprocess.DEVNULL, stdout=lg, stderr=subprocess.STDOUT))
    rcs = [p.wait() for p in ps]
    print("ZZP5CR_SCORE_DONE rcs", rcs)
    return 0 if all(r == 0 for r in rcs) else 1


def make_red_transp(src: str, dst: str) -> None:
    """MUTATION arm: transp_score.py with the cache-routing line deleted (always the DB rebuild). Asserts exactly one occurrence."""
    txt = open(src, encoding="utf-8", newline="").read()
    if txt.count(ROUTE_LINE) != 1:
        raise SystemExit(f"red arm: routing line found {txt.count(ROUTE_LINE)} times in {src}, expected 1")
    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.write(txt.replace(ROUTE_LINE, "parts = PT.metric_cache_parts(sc)"))


# --------------------------------------------------------------------------------------------- identity (B6)
def read_rows(paths) -> tuple:
    """-> ({token: row}, [error rows]). A token with an error row and no good row stays missing; a later good row wins."""
    rows, errs = {}, {}
    for fp in paths:
        for ln in open(fp, encoding="utf-8"):
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if "error" in r:
                errs[r.get("token")] = {"token": r.get("token"), "error": str(r["error"])[:200]}
            else:
                rows[r["token"]] = r
                errs.pop(r["token"], None)
    return rows, list(errs.values())


def read_harness_csv(path: str) -> dict:
    return {r["token"]: r for r in csv.DictReader(open(path, encoding="utf-8", newline="")) if r["token"] != "average"}


def check_identity(rows: dict, errs: list, arm: str, banked: dict, expected_tokens, bar: float = BAR) -> dict:
    """tracked sub-scores of `arm` per token vs the harness csv, all 7 metrics. PASS iff coverage complete, 0 error rows, 0 over the bar."""
    exp = set(expected_tokens)
    mx = {m: 0.0 for m in COL}
    over, no_csv = [], []
    for t in sorted(exp & set(rows)):
        if t not in banked:
            no_csv.append(t)
            continue
        tr = rows[t]["arms"][arm]["tracked"]
        d = {m: abs(float(tr[m]) - float(banked[t][COL[m]])) for m in COL}
        for m in COL:
            mx[m] = max(mx[m], d[m])
        if max(d.values()) > bar:
            over.append({"token": t, **{m: round(v, 9) for m, v in d.items() if v > bar}})
    missing = sorted(exp - set(rows))
    ok = (not over) and (not missing) and (not errs) and (not no_csv) and len(exp) > 0
    return {"arm": arm, "bar": bar, "n_expected": len(exp), "n_scored": len(exp & set(rows)), "n_missing": len(missing), "n_error_rows": len(errs),
            "n_not_in_csv": len(no_csv), "n_over_bar": len(over), "max_abs_diff": mx, "over_first10": over[:10], "missing_first5": missing[:5],
            "pass": bool(ok), "verdict": "PASS" if ok else "FAIL"}


def cmd_identity(a) -> int:
    paths = [p for g in a.rows for p in sorted(glob.glob(g))]
    rows, errs = read_rows(paths)
    expected = [str(t) for t in np.load(a.plans, allow_pickle=False)["tokens"]]
    res = check_identity(rows, errs, a.arm, read_harness_csv(a.csv), expected, a.bar)
    res.update(control="B6_identity", csv=os.path.abspath(a.csv), csv_sha256=sha256_file(a.csv), evidence="MEASURED (this run); tier T1 executed (LQR)")
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print("ZZP5CR_IDENTITY", a.arm, res["verdict"], "n", res["n_scored"], "/", res["n_expected"], "over", res["n_over_bar"],
          "max", max(res["max_abs_diff"].values()))
    return 0 if res["pass"] else 1


def cmd_red(a) -> int:
    """RED arm: a small slice of the arm through the MUTATED transp_score (DB route, no cache); identity MUST read FAIL.
    Exit 0 iff it FAILED (the control can tell the routes apart); exit 1 if the mutated route 'passed' (the control is blind)."""
    os.makedirs(a.out_dir, exist_ok=True)
    mut = os.path.join(a.out_dir, "transp_score_RED.py")
    make_red_transp(os.path.join(HERE, "..", "instruments", "transp_score.py"), mut)
    sub = os.path.join(a.out_dir, "plans_red")
    os.makedirs(sub, exist_ok=True)
    z = np.load(os.path.join(a.plans_dir, f"plans_{a.arm}.npz"), allow_pickle=False)
    keep = subset_tokens([str(t) for t in z["tokens"]], a.n, 11)
    idx = {str(t): i for i, t in enumerate(z["tokens"])}
    sel = [idx[t] for t in keep]
    np.savez(os.path.join(sub, f"plans_{a.arm}.npz"), **{k: (np.array(keep) if k == "tokens" else z[k][sel]) for k in z.files})
    c = build_score_cmds(sub, a.out_dir, 1, 0, mut, a.python, human=False, no_fam=True, cache=False)[0]
    with open(os.path.join(a.out_dir, "red.log"), "w", encoding="utf-8") as lg:
        rc = subprocess.call(c, env=score_env(), stdin=subprocess.DEVNULL, stdout=lg, stderr=subprocess.STDOUT)
    rows, errs = read_rows(glob.glob(os.path.join(a.out_dir, "rows_s0of1.jsonl")))
    res = check_identity(rows, errs, a.arm, read_harness_csv(a.csv), keep, BAR)
    res.update(role="RED arm (cache routing line deleted): identity is EXPECTED to FAIL", score_rc=rc, red_detected=res["verdict"] == "FAIL")
    with open(os.path.join(a.out_dir, "red_result.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print("ZZP5CR_RED", "DETECTED" if res["red_detected"] else "BLIND", "n_over", res["n_over_bar"], "of", res["n_scored"])
    return 0 if res["red_detected"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("convert")
    c.add_argument("--arm", nargs="+", required=True)
    c.add_argument("--outdir", required=True)
    c.add_argument("--subset", type=int, default=0)
    c.add_argument("--subset-seed", type=int, default=20261113)
    s = sp.add_parser("score")
    s.add_argument("--plans-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--procs", type=int, default=2)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--transp-script", default=None)
    s.add_argument("--python", default=None, help="the navsim-crun venv python")
    s.add_argument("--no-human", action="store_true")
    s.add_argument("--no-fam", action="store_true")
    i = sp.add_parser("identity")
    i.add_argument("--rows", nargs="+", required=True)
    i.add_argument("--arm", required=True)
    i.add_argument("--csv", required=True)
    i.add_argument("--plans", required=True)
    i.add_argument("--out", required=True)
    i.add_argument("--bar", type=float, default=BAR)
    r = sp.add_parser("red")
    r.add_argument("--plans-dir", required=True)
    r.add_argument("--arm", required=True)
    r.add_argument("--csv", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--n", type=int, default=60)
    r.add_argument("--python", default=None)
    a = ap.parse_args()
    return {"convert": cmd_convert, "score": cmd_score, "identity": cmd_identity, "red": cmd_red}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
