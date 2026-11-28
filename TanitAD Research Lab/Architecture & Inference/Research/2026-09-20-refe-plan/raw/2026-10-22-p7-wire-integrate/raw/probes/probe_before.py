"""MISMATCH PROBE (BEFORE the fix): the tip's gate_checks.check_reserve against the REAL session_reserve on a clean synthetic tree."""
import sys, json, tempfile, shutil
from pathlib import Path
REFE = Path(sys.argv[1]); sys.path.insert(0, str(REFE))
import gate_checks as C, session_reserve as SR, test_session_reserve as TS, launch_gate as L
reserve_doc = json.loads((REFE / "fixtures" / "p7_reserve_sessions.json").read_text(encoding="utf-8"))
real_sess = [x["session"] for x in reserve_doc["sessions"]]
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    root = TS.make_inputs(tmp / "tree", (TS.T1A, TS.T2A, TS.T3A))
    rj = tmp / "reserve_sessions.json"; rj.write_bytes((REFE / "fixtures" / "p7_reserve_sessions.json").read_bytes())
    argv = ["--targets", str(root / "bank"), "--metric-cache", str(root / "mc"), "--aux-drv-labels", str(root / "store"), "--images", str(root / "pix"), "--calib", str(root / "calib.json")]
    print("kw:", {k: (v if not isinstance(v, list) else f"{len(v)} files") for k, v in C.reserve_audit_inputs(argv).items()})
    ev = C.check_reserve(argv, L.PROFILES["refe_p7"], rj)
    print("status", ev["status"]); print("reasons", ev["reasons"])
    # the same real audit called with the load_reserve() frozenset, on the same tree
    r = SR.audit_inputs(SR.load_reserve(rj), **C.reserve_audit_inputs(argv))
    print("real audit with load_reserve(): ok =", r["ok"], r["counts"]["examined"])
