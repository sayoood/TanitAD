"""MM 2026-11-22: produce a REAL weight_rule_rank.py result JSON (tiny CPU bank, synthetic labels = NOT A DECISION) so the chain's JSON parse can be
checked against the tool's actual output, not a stub. usage: python wr_json_probe.py <refe dir> <out json>"""
import io, contextlib, json, shutil, sys
from pathlib import Path
refe, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(refe))
import test_rank_loss as TR          # noqa: E402  (the same tiny bank + rich synthetic labels the smoke check uses)
import weight_rule_rank as WR        # noqa: E402
import online_labeller as OL         # noqa: E402
P6 = TR.P6
td = P6._mkdtemp()
bank = P6.make_bank(td, 6)
args = " ".join(P6.argv_p6(bank, str(td / "wr_out"), extra=("--rank-heads", "ep,gate")))
real = OL.synthetic_labels
OL.synthetic_labels = TR._rich_labels()
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        rc = WR.main(["--args", args, "--refe", str(refe), "--out", str(out)])
finally:
    OL.synthetic_labels = real
shutil.rmtree(td, ignore_errors=True)
d = json.loads(out.read_text(encoding="utf-8"))
print("rc", rc, "top-level keys", sorted(k for k in d if k in ("status", "rank_w", "rule")), "rule", d["rule"])
print(buf.getvalue()[-600:])
