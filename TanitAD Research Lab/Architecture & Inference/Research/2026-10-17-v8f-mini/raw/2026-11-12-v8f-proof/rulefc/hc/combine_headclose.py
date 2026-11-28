"""combine_headclose.py -- card PREREG_V8F_PROOF_ADD152 (HEADCLOSE10K): the combined judgement of card section 5 from the grid141 read (pair V1) and the age read (pair V1P30).
Writes the name, the P30 name and the committed consequence; says whether the conditional ceiling arm runs. Tier T1-kin/hb0.2; never driving performance.
  python combine_headclose.py --v1 READ_V1.json --p30 READ_V1P30.json --free-gb <D: free on box 2> --out HEADCLOSE10K_RESULT.json [--ceiling READ_CEILING.json]"""
import argparse
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hc_common as HC  # noqa: E402


def blocks(v1, p30):
    if not (v1 and v1.get("controls_all_pass") is True and p30 and p30.get("controls_all_pass") is True):
        raise SystemExit("REFUSED: a read whose controls did not all pass has no name")
    return v1["pairs"]["V1"], p30["pair"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True)
    ap.add_argument("--p30", required=True)
    ap.add_argument("--free-gb", type=float, required=True)
    ap.add_argument("--ceiling", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    v1, p30 = HC.jl(a.v1), HC.jl(a.p30)
    b1, bp = blocks(v1, p30)
    res = {"card": "PREREG_V8F_PROOF_ADD152", "tier": "T1-kin/hb0.2 (kinematic rollout on the model's own actions; logged camera frames; the lead does not react); never driving performance",
           "estimator": "paired episode-cluster bootstrap pooled over the three model seeds; it answers another draw of episodes, not another training or inference run"}
    res.update(HC.combine(b1, bp, a.free_gb))
    if a.ceiling:
        c = HC.jl(a.ceiling)
        if not (c and c.get("controls_all_pass") is True):
            raise SystemExit("REFUSED: the ceiling read's controls did not all pass")
        res["ceiling"] = HC.ceiling_verdict(c["pairs"]["C1"], c["pairs"]["V1C1"])
        res["ceiling"]["names"] = c.get("names")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
