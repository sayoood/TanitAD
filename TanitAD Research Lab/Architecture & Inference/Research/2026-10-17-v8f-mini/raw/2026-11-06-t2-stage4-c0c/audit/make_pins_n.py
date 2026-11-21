import hashlib
import os

T = "E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f"
P = "E:/Projects/TanitAD-artifacts/dagland/motion_c"
files = ["STAGE4_LAUNCH.md", "HANDOFF_N.md", "HANDOFF_O.md", "BRIEF_T2V8F_STAGE4_N.md",
         "code/t2v8f_stage4.py", "code/t2v8f_stage4_cell.py", "code/t2v8f_c0c.py", "code/c0c_gate.py", "code/make_stage4.py", "code/run_mutations_stage4.py", "stage4_src/jobs/stage4.cmd",
         "code/read_t2_dominance.py", "code/read_t2_dominance.py.sha256.txt", "C0C_ID_RESULT.json",
         "tests/test_t2v8f_stage4.py", "audit/make_pins_n.py", "audit/finish_n.sh",
         "stage4/t2v8f_stage4_overlay.tar", "stage4/t2v8f_stage4_overlay.tar.sha256", "stage4/BUILD_RESULT4.json",
         "logs/make_stage4.log", "logs/pytest_stage4_positive.log", "logs/pytest_stage4_arms_O1.log", "logs/mutations_stage4.log", "logs/pytest_full_M.log", "logs/N_FINISHED.txt"]
frozen = {"code/t2v8f_stage2.py": "33b65b19a563b8fe275de5e46e0f569db6487bd7fdf40b7eab34e021094f4898", "code/t2v8f_stage3.py": "0688bb4732d8ef54f19e318de08008a54ce9276b66c815331e797b04019f733c",
          "code/read_t2_dominance.py": "6345d4c704ea46806271761ad3b53e376228f942e605da140fff3057cb9d2f5c"}
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()      # noqa: E731
out = ["# PINS_N (AGENT N + O, stage 4: the composite arm C0C over the 62 scenes): sha256 of every file written or relied on, 2026-11-06. LF endings; nothing committed, staged or deleted.",
       "# FROZEN (must read OK): " + "  ".join(f"{k.split('/')[-1]}={'OK' if sha(f'{T}/{k}') == v else 'CHANGED'}" for k, v in frozen.items()),
       f"# the RI tool (dev file, 76,704 B, must stay 0417c838...): {sha(P + '/ri/code/v8f_t1_eval_ri.py')}",
       f"# the frozen cards (t2_prereg): " + "  ".join(f"{n}={sha(f'{P}/t2_prereg/{n}')[:12]}" for n in sorted(os.listdir(f"{P}/t2_prereg")))]
try:
    import json
    br = json.load(open(f"{T}/stage4/BUILD_RESULT4.json", encoding="utf-8"))
    out.append(f"# BUILD_RESULT4: tar {br['tar_sha256']} ({br['tar_bytes']} B), MANIFEST4 {br['manifest4_sha256']}, driver {br['driver_sha256'][:12]}, cell {br['cell_sha256'][:12]}, c0c {br['c0c_sha256'][:12]}, ri {br['ri_tool_sha256'][:12]}")
except Exception as e:                                                  # noqa: BLE001
    out.append(f"# BUILD_RESULT4 unreadable: {e!r}")
for f in files:
    p = f"{T}/{f}"
    if os.path.exists(p):
        b = open(p, "rb").read()
        crlf = b"\r" in b and not f.endswith((".tar", ".log"))
        out.append(f"{hashlib.sha256(b).hexdigest()}  {len(b):>10}  {f}{'  <-- CR FOUND' if crlf else ''}")
    else:
        out.append(f"{'MISSING':64}  {'':>10}  {f}")
with open(f"{T}/PINS_N.txt", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(out) + "\n")
print("\n".join(out))
