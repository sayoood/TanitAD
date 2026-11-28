"""P7-WIRE-INTEGRATE-2 / Gap 2 probe: is the G-CLOCK content check REACHABLE from the real BUILD flow (not only from a fake capture)?
Same scenario as test_launch_gate.py F3 (the tiny rehearsal fixture, the REAL trainer built by `gate_checks.instrumented_run`, `launch_gate.run_checks(only={"G-CLOCK"})`), with `targets_sha256_by_bank` registered.
  1. registered digest == the bank's            -> G-CLOCK PASS, evidence carries `bank_content`
  2. two rows of the bank SWAPPED (same row count, same size, trainer still loads it)  -> G-CLOCK FAIL, for the CONTENT
  3. the count-only profile (no key) on the swapped bank                               -> PASS (the defect the content pin closes)
usage (torch venv, CPU): python probe_real_build_clock.py <refe dir>"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REFE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REFE))
import gate_checks as C  # noqa: E402
import gate_fixture as F  # noqa: E402
import gate_token as GT  # noqa: E402
import launch_gate as L  # noqa: E402

root = Path(tempfile.mkdtemp(prefix="p7int2_clock_"))
try:
    fx = F.tiny_p2_fixture(str(root))
    argv = F.rehearsal_argv(fx, str(root / "launch_out"))
    prof = L.PROFILES["refe_rehearsal"]
    cl_argv = C._set(C._drop(list(argv), "--steps", 1), "--epochs", "1")
    bank = Path(GT.flag_values(cl_argv, "--targets")[0])
    dig = C.bank_targets_sha256(bank)
    print("bank", bank.name, "rank files", [p.name for p in sorted(bank.glob("targets_rank*.jsonl"))], "digest", dig[:16])
    sched = {"epochs": 1, "steps": 2, "epoch_len": 4}
    pinned = {**prof, "expect_schedule": {**sched, "targets_sha256_by_bank": {bank.name: dig}}}
    countonly = {**prof, "expect_schedule": dict(sched)}
    out = root / "o1"
    st = L.run_checks(pinned, cl_argv, L.PKG if hasattr(L, "PKG") else REFE.parent, out, only={"G-CLOCK"})
    ev = GT.read_json(out / "evidence_G-CLOCK.json")
    print("1. registered bank          ->", st, "| bank_content:", {k: (str(v)[:16] if v else v) for k, v in ev["details"].get("bank_content", {}).items()}, "| reasons:", ev["reasons"])
    # swap the first two rows of the first rank file: same count, same size, still a valid bank
    rk = sorted(bank.glob("targets_rank*.jsonl"))[0]
    lines = rk.read_bytes().splitlines(keepends=True)
    assert len(lines) >= 2 and lines[0] != lines[1], len(lines)
    rk.write_bytes(b"".join([lines[1], lines[0]] + lines[2:]))
    print("   swapped two rows of", rk.name, "- digest now", C.bank_targets_sha256(bank)[:16])
    out2 = root / "o2"
    st2 = L.run_checks(pinned, cl_argv, REFE.parent, out2, only={"G-CLOCK"})
    ev2 = GT.read_json(out2 / "evidence_G-CLOCK.json")
    print("2. same-count tampered bank ->", st2, "| reasons:", [r[:150] for r in ev2["reasons"]], "| trainer's built count:", ev2["details"].get("bank"))
    out3 = root / "o3"
    st3 = L.run_checks(countonly, cl_argv, REFE.parent, out3, only={"G-CLOCK"})
    print("3. count-only profile, same tampered bank ->", st3)
    print("RESULT:", "REACHABLE" if (st == {"G-CLOCK": "PASS"} and st2 == {"G-CLOCK": "FAIL"} and st3 == {"G-CLOCK": "PASS"}) else "NOT AS EXPECTED")
finally:
    shutil.rmtree(root, ignore_errors=True)
