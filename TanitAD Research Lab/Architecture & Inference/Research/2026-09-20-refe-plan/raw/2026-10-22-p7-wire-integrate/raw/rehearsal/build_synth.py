"""P7-WIRE-INTEGRATE: build the INPUTS of the pin procedure on disk (synthetic, small): an unfiltered 3-C-shaped bank directory, a finalised FULL aux label store, a full calibration table, and the LF reserve
file.  `rehearse.sh` then runs the procedure's literal CLI commands on them.  usage: python build_synth.py <refe dir> <out dir>"""
import json
import sys
from pathlib import Path

REFE, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(REFE))
import aux_label_store as AL  # noqa: E402
import session_reserve as SR  # noqa: E402
import test_session_reserve as TS  # noqa: E402

doc = json.loads((REFE / "fixtures" / "p7_reserve_sessions.json").read_text(encoding="utf-8"))
logs = [lg["log"] for x in doc["sessions"] for lg in x["logs"]]
R_A, R_B = logs[0], logs[-1]
T1, T2, T3 = "2021.05.12.19.36.12_veh-35_00001_00099", "2021.06.09.14.50.36_veh-26_00007_00070", "2021.11.03.10.20.30_veh-17_00100_00200"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "reserve_sessions.json").write_bytes((REFE / "fixtures" / "p7_reserve_sessions.json").read_bytes().replace(b"\r\n", b"\n"))
root = TS.make_inputs(OUT / "raw_inputs", (R_A, R_B, T1, T2, T3))
with open(root / "bank" / "targets_rank1.jsonl", "ab") as f:
    for r in (TS.row(T1, "a1", 1, TS.GH), TS.row(T2, "b2", 1, TS.LR1), TS.row(R_B, "b2", 1, TS.LR1)):
        f.write(json.dumps(r).encode() + b"\n")
AL.finalize(root / "store")
(OUT / "train_grow_cmd_3c").mkdir(exist_ok=True)
for p in (root / "bank").glob("targets_rank*.jsonl"):
    (OUT / "train_grow_cmd_3c" / p.name).write_bytes(p.read_bytes())
(OUT / "calib_table.json").write_bytes((root / "calib.json").read_bytes())
import shutil  # noqa: E402
shutil.copytree(root / "store", OUT / "label_store", dirs_exist_ok=True)
print("built", OUT, "reserved logs", R_A[:30], R_B[:30])
