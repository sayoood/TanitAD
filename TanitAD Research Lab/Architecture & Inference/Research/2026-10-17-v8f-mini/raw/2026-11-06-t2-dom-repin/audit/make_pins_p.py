import hashlib
import json
import os

T = "E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f"
P = "E:/Projects/TanitAD-artifacts/dagland/motion_c"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()      # noqa: E731
files = ["BRIEF_T2V8F_DOMREPIN_P.md", "HANDOFF_P.md",
         "code/read_t2_dominance.py", "code/read_t2_dominance.py.sha256.txt", "code/run_mutations_dom_p.py",
         "tests/test_t2v8f_dominance_rscn.py", "tests/test_t2v8f_stage4.py", "audit/finish_p.sh", "audit/make_pins_p.py",
         "logs/mutations_dom_p.log", "logs/pytest_full_P.log",
         "code/scene_exclusions.py", "code/scene_exclusions.py.sha256.txt", "code/read_stage2.py", "code/read_stage2.py.sha256.txt", "tests/test_scene_exclusions.py",
         "tests/test_t2v8f_dominance.py", "tests/test_t2v8f_dominance_g2.py", "tests/test_t2v8f_gates.py", "tests/test_t2v8f_e5.py", "logs/pytest_full_M.log"]
frozen = {"code/t2v8f_stage2.py": "33b65b19a563b8fe275de5e46e0f569db6487bd7fdf40b7eab34e021094f4898", "code/t2v8f_stage3.py": "0688bb4732d8ef54f19e318de08008a54ce9276b66c815331e797b04019f733c",
          "code/scene_exclusions.py": "83a5f64716a02182a2a880bf8cec23c6e71589c98c841c6c4ebd1e609eceb373", "code/read_stage2.py": "f2136b39442f6df18819465baa37113385770f6ca940dd019acdf2d9bb945810"}
new_pin = open(f"{T}/code/read_t2_dominance.py.sha256.txt", encoding="utf-8").readline().strip()
out = ["# PINS_P (AGENT P, the dominance read re-pinned with rule R-SCN): sha256 of every file written, edited or relied on, 2026-11-06. LF endings; nothing committed, staged or deleted.",
       "# FROZEN (must read OK): " + "  ".join(f"{k.split('/')[-1]}={'OK' if sha(f'{T}/{k}') == v else 'CHANGED'}" for k, v in frozen.items()),
       f"# read_t2_dominance.py now {sha(T + '/code/read_t2_dominance.py')}; pin file line 1 {new_pin}; equal: {sha(T + '/code/read_t2_dominance.py') == new_pin}",
       "# the previous pins of the read: df4e863649d6d84d (AGENT L), 6345d4c704ea4684 (AGENT M); a copy of the 6345d4c7 file is on F:/TanitAD-scratch/t2v8f_P/read_t2_dominance.pin6345d4c7.py",
       "# the frozen cards (t2_prereg, untouched): " + "  ".join(f"{n}={sha(f'{P}/t2_prereg/{n}')[:12]}" for n in sorted(os.listdir(f"{P}/t2_prereg")))]
try:
    cs = open(f"{P}/t2_prereg/PREREG_T2_DOMINANCE.md", "rb").read()
    out.append(f"# card frozen prefix (first 8497 bytes) sha256 {hashlib.sha256(cs[:8497]).hexdigest()[:12]} (must be 38aa9df3...)")
except Exception as e:                                                  # noqa: BLE001
    out.append(f"# card unreadable: {e!r}")
for f in files:
    p = f"{T}/{f}"
    if os.path.exists(p):
        b = open(p, "rb").read()
        crlf = b"\r" in b and not f.endswith((".tar", ".log"))
        out.append(f"{hashlib.sha256(b).hexdigest()}  {len(b):>10}  {f}{'  <-- CR FOUND' if crlf else ''}")
    else:
        out.append(f"{'MISSING':64}  {'':>10}  {f}")
with open(f"{T}/PINS_P.txt", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(out) + "\n")
print("\n".join(out))
