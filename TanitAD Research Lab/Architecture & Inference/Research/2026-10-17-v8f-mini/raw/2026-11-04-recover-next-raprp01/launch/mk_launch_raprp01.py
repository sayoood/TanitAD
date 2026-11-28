"""Derive launch_raprp01.sh (addendum 2 row E: RAPRP01_0 RAPRP01_1, RAPR's frozen argv with --recover-p 0.1) from launch_rapr2.sh. Every replacement must match exactly once."""
from pathlib import Path

H = Path(__file__).resolve().parent
s = (H / "launch_rapr2.sh").read_text(encoding="utf-8")
BS = "\\"
rep = [
    ("# RECOVER-NEXT addendum row D (RAPR as frozen, 2 runs): wait for box 2's lock",
     "# RECOVER-NEXT addendum 2 row E (RAPR-P01 = RAPR's frozen argv with --recover-p 0.1, 2 runs): wait for box 2's lock"),
    ("the two-run chain (rnext_chain_rapr.cmd). RUN ONLY", "the two-run chain (rnext_chain_raprp01.cmd). RUN ONLY"),
    ("BOX2_BUSY.rapr2_launch_failed_", "BOX2_BUSY.raprp01_launch_failed_"),
    ("RECOVER-NEXT RAPR two-run chain (D:" + BS + BS + "tanitad" + BS + BS + "rnext" + BS + BS + "rnext_chain_rapr.cmd), owner",
     "RECOVER-NEXT RAPR-P01 two-run chain (D:" + BS + BS + "tanitad" + BS + BS + "rnext" + BS + BS + "rnext_chain_raprp01.cmd), owner"),
    ("PI go ${RAPR2_PI_GO:?set RAPR2_PI_GO to the PI go time}", "PI go ${RAPRP01_PI_GO:?set RAPRP01_PI_GO to the PI go time}"),
    ("CHAIN_RNEXT.rtl_rp01_$st.done" + BS + '"" || die', "CHAIN_RNEXT.rapr_$st.done" + BS + '"" || die'),
    ('echo "[$(ts)] old CHAIN_RNEXT.done moved aside to CHAIN_RNEXT.rtl_rp01_$st.done"', 'echo "[$(ts)] old CHAIN_RNEXT.done moved aside to CHAIN_RNEXT.rapr_$st.done"'),
    ("rnext_chain_rtl_rp01.cmd rnext_chain_rapr.cmd rnext_run.cmd", "rnext_chain_rtl_rp01.cmd rnext_chain_rapr.cmd rnext_chain_raprp01.cmd rnext_run.cmd"),
    ("for t in RAPR_0 RAPR_1 RTL_0 RTL_1 RP01_0 RP01_1; do", "for t in RAPR_0 RAPR_1 RTL_0 RTL_1 RP01_0 RP01_1 RAPRP01_0 RAPRP01_1; do"),
    ("L 'D:" + BS + "tanitad" + BS + "rnext" + BS + "rnext_chain_rapr.cmd' 'D:" + BS + "tanitad" + BS + "logs" + BS + "rnext_chain_rapr.log'",
     "L 'D:" + BS + "tanitad" + BS + "rnext" + BS + "rnext_chain_raprp01.cmd' 'D:" + BS + "tanitad" + BS + "logs" + BS + "rnext_chain_raprp01.log'"),
    ("Get-Content D:" + BS + BS + "tanitad" + BS + BS + "logs" + BS + BS + "rnext_chain_rapr.log -Tail 12",
     "Get-Content D:" + BS + BS + "tanitad" + BS + BS + "logs" + BS + BS + "rnext_chain_raprp01.log -Tail 12"),
]
for a, b in rep:
    assert s.count(a) == 1, (a[:80], s.count(a))
    s = s.replace(a, b)
assert s.count("ZZRAPR2-") == 6, s.count("ZZRAPR2-")
s = s.replace("ZZRAPR2-", "ZZRAPRP01-")
assert "RAPR2" not in s and "rapr2" not in s, [l for l in s.splitlines() if "RAPR2" in l or "rapr2" in l]
(H / "launch_raprp01.sh").write_text(s, encoding="utf-8", newline="\n")
print("wrote launch_raprp01.sh")
