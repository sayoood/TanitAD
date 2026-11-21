"""Exact re-simulation of the six-rung ladder (log mode) with the curve_speed fix ON, through the real run_v5 caller
(run_v5._do_state -> GymState -> RewardEvaluator). CPU only, one process.
usage: resim_ladder.py MIN_SEG IDX [IDX ...]      (MIN_SEG = 'none' reproduces v5; used as the bit-for-bit control)
Writes raw/resim/resim_<idx>.json (or resim_none_<idx>.json). floor 0.20 as in the REWARD-1 run (pilot_v5/raw/ladder_20_160/run_meta.json)."""
import json, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "scorer_gym"))
import run_v5
import reward1 as r1

if __name__ == "__main__":
    arc = None if sys.argv[1] == "none" else float(sys.argv[1])
    idxs = [int(x) for x in sys.argv[2:]]
    sts = r1.load_states_v5()                     # refuses a file whose sha256 differs from the pinned one
    os.makedirs(os.path.join(HERE, "raw", "resim"), exist_ok=True)
    for i in idxs:
        r = run_v5.do_state((i, sts[i], "ladder", 0.20, arc))
        if "error" in r:
            print(i, "ERROR", r["error"], flush=True); continue
        lad = {}
        for rk, d in r["ladder"].items():
            lg = d["logs"]
            lad[rk] = dict(ret=d["ret"], hard=any(c["hard"] for c in lg), n_cycles=len(lg),
                           n_qfree_active=sum(bool(c["active"]["q_free"]) for c in lg), min_qfree=min(c["q"]["q_free"] for c in lg),
                           rewards=[c["reward"] for c in lg])
        name = f"resim_{i}.json" if arc is not None else f"resim_none_{i}.json"
        json.dump(dict(idx=i, arc=arc, floor=0.20, limit=r["limit"], ladder=lad), open(os.path.join(HERE, "raw", "resim", name), "w"), default=float)
        print(i, "done", flush=True)
