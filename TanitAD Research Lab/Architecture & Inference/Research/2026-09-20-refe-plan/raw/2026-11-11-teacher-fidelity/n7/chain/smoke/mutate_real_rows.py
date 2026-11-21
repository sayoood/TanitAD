"""Deliberate-regression arms on the REAL navtrain smoke rows: for every selector mutation the stored real candidates of the clean derive (smoke/derive_clean_s*of2.jsonl) are re-selected with the
MUTATED selector (what a derive that ran a different selector would have recorded) and the real merge is run on them. Every mutation that changes a choice must turn C-SELECT RED (and the
control the card names). Needs N4_RUN, the navsim venv, and the smoke inputs of mk_smoke_inputs7.py.
usage: mutate_real_rows.py <smoke dir> <out json>"""
import contextlib
import glob
import io
import json
import os
import sys
import tempfile
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE)))
import n7_nt_build as BD  # noqa: E402

S = sys.argv[1]
OUT = sys.argv[2]
rows = {}
for fp in sorted(glob.glob(os.path.join(S, "derive_clean_s*of2.jsonl"))):
    for ln in open(fp, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            rows[r["token"]] = r
res = {}


def run(name, rws):
    d = tempfile.mkdtemp(dir=os.environ.get("N7_TMP"))
    rp = os.path.join(d, "rows.jsonl")
    with open(rp, "w", encoding="utf-8") as fo:
        for r in rws:
            fo.write(json.dumps(r) + "\n")
    kw = dict(rows=[rp], n5a_plans=S + "/in/n5a_plans.npz", n5a_seam=S + "/in/n5a_seam.npz", n5a_report="", expect_residual_json=S + "/in/resid.json",
              replay_expect_json=S + "/in/replay_expect.json", harness_csv_n5a=S + "/in/n5a_harness.csv", reach_json=S + "/reach_real.json", out_dir=os.path.join(d, "out"), expect_residual=20,
              n6_report="", n6t_plans="")
    with contextlib.redirect_stdout(io.StringIO()):
        rc = BD.cmd_merge(SimpleNamespace(**kw))
    rep = json.load(open(os.path.join(d, "out", "n7_build_report.json")))
    red = sorted(k for k, v in rep["controls"].items() if not v["pass"])
    res[name] = {"merge_rc": rc, "verdict": rep["verdict"], "red_controls": red, "applied": {k: len(v) for k, v in rep["repaired"].items()}}
    print(name, rep["verdict"], red, res[name]["applied"], flush=True)


def reselect(mut, arm_only=None):
    out = []
    for t in sorted(rows):
        r = json.loads(json.dumps(rows[t]))
        if r.get("residual"):
            row = {"ref": r["ref"], "levers": r["levers"]}
            for arm in BD.ARMS:
                fn = BD.make_selector(arm, mut if arm_only in (None, arm) else "none")
                c = fn(row)
                r["chosen"][arm] = None if c is None else {"tier": c["tier"], "lever": c["lever"], "k": c["k"], "param": c["param"], "plan": c["plan"], "m": c["m"], "m_recheck": dict(c["m"])}
        out.append(r)
    return out


run("clean", [rows[t] for t in sorted(rows)])
for mut in BD.SEL_MUTATIONS:
    run(mut, reselect(mut))
run("dac_inverted_N7X_only", reselect("dac_inverted", "N7X"))
off = []
for t in sorted(rows):                                       # the generator was not run (levers_off)
    r = json.loads(json.dumps(rows[t]))
    if r.get("residual"):
        r["levers"] = {}
        r["det_equal"] = False
        r["chosen"] = {a: None for a in BD.ARMS}
    off.append(r)
run("levers_off", off)
mut_files = sorted(glob.glob(os.path.join(S, "derive_mut_drop_T1_s*of2.jsonl")))      # a REAL derive CLI run with --mutate drop_T1 on 4 tokens: its rows replace the clean ones
if mut_files:
    mrows = {}
    for fp in mut_files:
        for ln in open(fp, encoding="utf-8"):
            if ln.strip():
                r = json.loads(ln)
                mrows[r["token"]] = r
    run("real_cli_drop_T1", [mrows.get(t, rows[t]) for t in sorted(rows)])
    res["real_cli_drop_T1"]["tokens_from_the_mutated_derive"] = sorted(mrows)
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
