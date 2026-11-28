"""scratch (not in the package): build the synthetic-world bank with the repo's own generator, run RecoverDevTable.from_dagger_bank on it, write the table's quantiles and band coverage to JSON."""
import json
import sys

sys.path[:0] = ["F:/TanitAD-scratch/recover_ft/tree_patched/stack/tests"]
from test_v8f_recover_wiring import *  # noqa: F401,F403
from test_v8f_recover_wiring import pack, parts, REC  # noqa: F401


def test_report(pack):
    pt = parts(pack, REC(pack))
    rcv, info = pt[5], pt[4]
    out = {"n": len(rcv.table), "counts": rcv.table.counts, "quantiles": rcv.table.quantiles(), "band": info["recover"]["table_band_3_8"], "summary": info["recover"]["table"],
           "reason_counts": info["recover"]["reason_counts"]}
    open("F:/TanitAD-scratch/recover_ft/logs/synthetic_table_report.json", "w").write(json.dumps(out, indent=1, default=float))
