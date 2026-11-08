"""usage: next_block_v5.py DIR [DIR ...] -> prints JSON{n_eligible, n_reached, verdict_so_far, next_block}. DIR = a run dir with states.jsonl.gz (+ run_meta.json).
Uses reward1.b3_verdict / next_block (the functions the tests mutate) so the chain's extension decision is the tested rule."""
import gzip, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gym_env as g
import reward1 as r1

rows, reached = [], 0
for d in sys.argv[1:]:
    m = json.load(open(os.path.join(d, "run_meta.json"), encoding="utf-8"))
    reached = max(reached, m["range"][1])
    for l in gzip.open(os.path.join(d, "states.jsonl.gz"), "rt", encoding="utf-8"):
        r = json.loads(l)
        rets = {n: v["ret"] for n, v in r["ladder"].items()}
        hards = {n: any(x["hard"] for x in v["logs"]) for n, v in r["ladder"].items()}
        inf, kn = g.ladder_known(rets, hards)
        rows.append(dict(idx=r["idx"], informative=inf, known=kn, limit=r["limit"]))
v = r1.b3_verdict(rows, reached >= r1.CAP)
print("JSON" + json.dumps(dict(n_eligible=v["n_eligible"], n_reached=reached, verdict_so_far=v["verdict"], next_block=r1.next_block(v["n_eligible"], reached))))
