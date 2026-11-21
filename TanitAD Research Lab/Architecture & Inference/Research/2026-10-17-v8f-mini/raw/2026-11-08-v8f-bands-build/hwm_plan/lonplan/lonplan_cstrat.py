"""C-STRAT (MM freeze line, AMEND_HWMPLAN_3): contrast_l1full_strat.py against phase2_l1full/contrast_l1full.py on the same inputs.
  (1) every non-STRATEGIC cell (rows, replicate rows, tactical kappa, counts) is bit-equal (json text equal, NaN included);
  (2) the STRATEGIC cells are computed (state not PAUSED, a strategic_block dict) and every full-6 s cell carries n, or an n/a with its reason when n = 0;
  (3) regression arm: the same script run with the code-of-record folder as --eval-src reads PAUSED and compare() flags it.
usage: python lonplan_cstrat.py --windows DIR --a A.npz --b B.npz [--rep-a ..] [--rep-b ..] --workdir NEWDIR [--n-boot N]   (runs the three scripts, prints the verdict, exit 0 only if (1) and (2) hold and (3) is flagged)"""
import argparse, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.environ.get("LONPLAN_CODE_ROOT", "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory")
RECORD_SCRIPT = os.environ.get("LONPLAN_RECORD_CONTRAST", "F:/TanitAD-scratch/hwm_plan/phase2_l1full/contrast_l1full.py")
RECORD_SCRIPT_GIT_BLOB = "3f08e24dd2cf9e845693b851ed1522d6fe884596"        # git blob id of phase2_l1full/contrast_l1full.py at 4dc4876c (sha256 b7661635...)
STRAT_SCRIPT = os.path.join(HERE, "contrast_l1full_strat.py")


def run(script, windows, a, b, out, rep_a=None, rep_b=None, eval_src=None, n_boot=None, py=sys.executable):
    cmd = [py, script, "--windows", windows, "--a", a, "--b", b, "--out", out]
    for k, v in (("--rep-a", rep_a), ("--rep-b", rep_b), ("--eval-src", eval_src), ("--n-boot", n_boot)):
        if v is not None:
            cmd += [k, str(v)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, r.stdout[-3000:], r.stderr[-3000:]


def _j(x):
    return json.dumps(x, sort_keys=True)


def compare(rec, strat):
    """rec, strat: parsed JSON of the two scripts (rec may be the regression arm's output when called for the PAUSED check)."""
    ns = [r for r in strat["rows"] if r["family"] != "STRATEGIC"]
    nr = [r for r in strat["replicate"] if r["family"] != "STRATEGIC"]
    out = dict(rows_equal=_j(ns) == _j(rec["rows"]), n_rows=len(ns), replicate_equal=_j(nr) == _j(rec["replicate"]), n_replicate=len(nr), tactical_equal=_j(strat["tactical_kappa"]) == _j(rec["tactical_kappa"]),
               counts_equal=(strat["n_windows"], strat["n_episodes"]) == (rec["n_windows"], rec["n_episodes"]), n_windows=strat["n_windows"], n_episodes=strat["n_episodes"])
    st = strat["strategic"]
    out["paused"] = isinstance(st, str) and "PAUSED" in st or (isinstance(st, dict) and "PAUSED" in json.dumps(st.get("state", "")))
    out["strategic_is_block"] = isinstance(st, dict)
    cells = [r for r in strat["rows"] if r["family"] == "STRATEGIC" and r["band"] == "full6s" and r["a"] == "A" and r["b"] == "B"]
    out["strategic_cells"] = {r["metric"]: dict(n_windows=r["n_windows"], n_episodes=r["n_episodes"], label=r["label"], reason=r.get("reason")) for r in cells}
    out["strategic_cells_have_n_or_reason"] = bool(cells) and all((r["n_windows"] > 0) or bool(r.get("reason")) for r in cells)
    out["non_strategic_equal"] = bool(out["rows_equal"] and out["replicate_equal"] and out["tactical_equal"] and out["counts_equal"])
    out["strategic_computed"] = bool(out["strategic_is_block"] and not out["paused"] and out["strategic_cells_have_n_or_reason"])
    out["pass"] = bool(out["non_strategic_equal"] and out["strategic_computed"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--rep-a"); ap.add_argument("--rep-b")
    ap.add_argument("--workdir", required=True); ap.add_argument("--n-boot", type=int)
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=False)
    f = lambda n: os.path.join(a.workdir, n)
    kw = dict(rep_a=a.rep_a, rep_b=a.rep_b, n_boot=a.n_boot)
    rc = [run(RECORD_SCRIPT, a.windows, a.a, a.b, f("record.json"), **kw), run(STRAT_SCRIPT, a.windows, a.a, a.b, f("strat.json"), **kw),
          run(STRAT_SCRIPT, a.windows, a.a, a.b, f("strat_on_record_import.json"), eval_src=os.path.join(CODE_ROOT, "hwm_plan"), **kw)]
    assert all(r[0] == 0 for r in rc), rc
    rec, strat, reg = (json.load(open(f(n), encoding="utf-8")) for n in ("record.json", "strat.json", "strat_on_record_import.json"))
    good, bad = compare(rec, strat), compare(rec, reg)
    print(json.dumps(dict(green=good, regression_arm_flagged=bool(bad["paused"] and not bad["pass"]), regression=bad), indent=1))
    sys.exit(0 if good["pass"] and bad["paused"] and not bad["pass"] else 1)


if __name__ == "__main__":
    main()
