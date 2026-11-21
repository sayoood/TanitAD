#!/usr/bin/env python3
"""behaviour_label.py (CPU, no NAVSIM): the record is exactly what scorer_finetune.load_beh_terms reads, the quarantine runs BEFORE any label is produced, a navtest row stops the run, and the file names / glob / sharding
agree with their consumers.  label_log() is run with FAKE navsim modules (the physics is behaviour_terms' own tests); what is NOT covered: the real NAVSIM scenario build and score_plans (needs the NAVSIM stack).
Run: CUDA_VISIBLE_DEVICES=-1 python test_behaviour_label.py -> ZZTEST_PASS n."""
import ast, json, os, sys, tempfile, types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scorer_finetune as S
import behaviour_label as L
import behaviour_gate as BG
import behaviour_terms as BT

fails = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)

# constants agree with the consumer
check("glob/prefix/version agree with scorer_finetune", L.BEH_PREFIX + "*.jsonl" == S.BEH_GLOB and L.TERMS_VERSION == S.BEH_TERMS_VERSION)
check("side_file_name", L.side_file_name("/x/y/onpolicy_a.jsonl") == "behaviour_onpolicy_a.jsonl")
import rl_terms_label as RT
check("shard_of == rl_terms_label.shard_of (a log lands in the same shard for both labellers)", all(L.shard_of(f"log{i}", 7) == RT.shard_of(f"log{i}", 7) for i in range(50)))

# the record round-trips through the trainer's loader
M = 6
rng = np.random.default_rng(0)
q = rng.uniform(0.2, 1.0, (M, 5)).astype(np.float32)
act = rng.uniform(size=(M, 5)) > 0.4
rec = L.make_record(["logA", "tok1", 3, 0], 400, 1, q, act)
by = {("logA", "tok1", 3, 0): (400,)}
with tempfile.TemporaryDirectory(dir=os.environ.get("A2_TEST_TMP")) as td:
    with open(os.path.join(td, "behaviour_onpolicy_a.jsonl"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    terms, st = S.load_beh_terms(td, by, M)
check("record -> load_beh_terms: matched 1, q/act equal (float32, bool)", st["matched"] == 1 and np.allclose(terms[("logA", "tok1", 3, 0)][0], q, atol=1e-6) and (terms[("logA", "tok1", 3, 0)][1] == act).all())
for name, bad in (("q > 1", lambda: L.make_record(["l", "t", 0, 0], 1, 1, q * 2.0, act)), ("NaN", lambda: L.make_record(["l", "t", 0, 0], 1, 1, np.where(act, np.nan, q), act)),
                  ("wrong width", lambda: L.make_record(["l", "t", 0, 0], 1, 1, q[:, :4], act[:, :4]))):
    try:
        bad(); check(f"make_record refuses {name}", False)
    except ValueError:
        check(f"make_record refuses {name}", True)

# label_log with fake NAVSIM modules: the gate first, then the records
logs, toks = BG.load()
nav_log = sorted(logs)[0]
ok_log, ok_tok = "zz_not_navtest_log", "0123456789abcdef"
assert ok_log not in logs and ok_tok not in toks
calls = {"score": 0, "scn": 0}
def terms_dict():
    t = {k: rng.uniform(0.3, 1.0, M) for k in BT.ORDER}
    t.update({"act_" + k[2:]: rng.uniform(size=M) > 0.5 for k in BT.ORDER})
    return t
PT = types.ModuleType("navsim_pdm_targets")
PT.metric_cache_parts = lambda sc: {"sc": sc}
PT.executed_plans8 = lambda props, repair=True: props
def score_plans(parts, plans8, behaviour=False):
    calls["score"] += 1
    return {"behaviour": terms_dict()} if behaviour else {}
PT.score_plans = score_plans
NS = types.ModuleType("navtrain_scenarios")
NS.index_dbs = lambda: {ok_log: object(), nav_log: object()}
def build(db, tokens, history_rows=40, future_rows=100):
    calls["scn"] += 1
    return [types.SimpleNamespace(scenario_name=t) for t in tokens]
NS.build_scenarios_for_log = build
sys.modules["navsim_pdm_targets"], sys.modules["navtrain_scenarios"] = PT, NS
def row(log, tok, step=0):
    return {"log_name": log, "token": tok, "step": step, "rank": 0, "ckpt_step": 400, "label_version": 1, "traj": np.zeros((M, 8, 2)).tolist(), "yaw": np.zeros((M, 8)).tolist()}
L._S.clear()
out, failed, skipped, err = L.label_log((ok_log, [row(ok_log, ok_tok, 0), row(ok_log, ok_tok, 1)]))
recs = [json.loads(x) for x in out]
check("label_log (fake NAVSIM): 2 records, 0 failed, each validates through the trainer's _beh_record", len(recs) == 2 and failed == 0 and all(S._beh_record(r, M) is not None for r in recs) and err is None)
n0 = dict(calls)
try:
    L.label_log((nav_log, [row(nav_log, ok_tok)]))
    raised = False
except BG.NavtestLeak:
    raised = True
check("NAVTEST row: label_log raises NavtestLeak BEFORE a scenario is built or scored", raised and calls == n0)
try:
    L.label_log((ok_log, [row(ok_log, ok_tok), row(ok_log, "")]))
    raised = False
except BG.NavtestLeak:
    raised = True
check("an unkeyed row (empty token) is refused too", raised)
os.environ[BG.MUTATE_ENV] = BG.MUTATION
try:
    outm, *_ = L.label_log((nav_log, [row(nav_log, ok_tok)]))
    leaked = len(outm) == 1
finally:
    del os.environ[BG.MUTATE_ENV]
check("REGRESSION ARM (gate switched off by REFE_LAB_MUTATE=no_navtest_gate): the navtest row IS labelled, i.e. the refusal above is the gate's doing", leaked)

# order in the source: the gate precedes the scenario build and the score call; main() gates on read and loads the lists up front
src = open(os.path.join(HERE, "behaviour_label.py"), encoding="utf-8").read()
tree = ast.parse(src)
fn = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
ll = ast.get_source_segment(src, fn["label_log"])
mn = ast.get_source_segment(src, fn["main"])
check("source order: label_log asserts the batch before _init/scenario build/make_record; main() calls BG.load() first and BG.assert_batch on every row it reads",
      ll.index("BG.assert_batch") < ll.index("_init()") < ll.index("build_scenarios_for_log") < ll.index("make_record(")
      and mn.index("BG.load()") < mn.index("glob.glob") and mn.index("BG.assert_batch") < mn.index("by.setdefault"))
leaks = [f for f in ("model.py", "planner.py") if os.path.isfile(os.path.join(os.path.dirname(HERE), "refe", f)) and "behaviour_label" in open(os.path.join(os.path.dirname(HERE), "refe", f), encoding="utf-8").read()]
check("the labeller is not imported by model.py / planner.py", not leaks)
print(("ZZTEST_PASS " + str(len(fails))) if not fails else ("ZZTEST_FAIL " + str(fails)))
sys.exit(1 if fails else 0)
