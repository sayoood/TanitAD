"""Post-process a dynamic dest-read trace (dynamic_reads.py): a dest is CONSUMED on the exercised paths when
a caller other than argparse's own parse loop or `effective_weights.explicit_dests` (a whole-namespace
re-parse) read it. Adds `consumers`, `never_consumed_on_exercised_paths`, `consumed_only_by_preflight`,
`_noise_excluded`. Run standalone to post-process raw/dynamic_reads_merge.json in place."""
import json
import sys
from pathlib import Path

NOISE = ("argparse.py:", "effective_weights.py:explicit_dests")


def add_consumers(res: dict) -> dict:
    cons = {d: [c for c in cs if not c.startswith(NOISE)] for d, cs in res["callers"].items()}
    res["consumers"] = cons
    res["never_consumed_on_exercised_paths"] = sorted(d for d, cs in cons.items() if not cs)
    res["consumed_only_by_preflight"] = sorted(d for d, cs in cons.items()
                                               if cs == ["train_v6_staged.py:preflight"])
    res["_noise_excluded"] = list(NOISE)
    return res


if __name__ == "__main__":
    p = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:/Users/Admin/v7f_gate/raw/dynamic_reads_merge.json")
    res = add_consumers(json.loads(p.read_text(encoding="utf-8")))
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("never consumed:", len(res["never_consumed_on_exercised_paths"]), res["never_consumed_on_exercised_paths"])
    print("consumed only by preflight:", res["consumed_only_by_preflight"])
