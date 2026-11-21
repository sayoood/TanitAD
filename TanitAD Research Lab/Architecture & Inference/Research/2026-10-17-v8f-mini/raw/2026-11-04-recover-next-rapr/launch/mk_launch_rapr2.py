"""Derive launch_rapr2.sh (addendum row D: RAPR_0 RAPR_1 as frozen) from launch_rnext4.sh. Every replacement must match."""
from pathlib import Path

H = Path(__file__).resolve().parent
s = (H / "launch_rnext4.sh").read_text(encoding="utf-8")
BS = "\\"
rep = [
    ("# RECOVER-NEXT, PI scope 2026-11-03 (\"RTL + RP01\"): wait for box 2's lock, take it atomically, ship, verify bytes, start telemetry -> guard -> the\n"
     "# four-run chain (rnext_chain_rtl_rp01.cmd).",
     "# RECOVER-NEXT addendum row D (RAPR as frozen, 2 runs): wait for box 2's lock, take it atomically, MOVE ASIDE the previous chain's\n"
     "# CHAIN_RNEXT.done (never delete), ship, verify bytes, start telemetry -> guard -> the two-run chain (rnext_chain_rapr.cmd). RUN ONLY AFTER THE PI'S BOX-2 GO."),
    ("Prints ZZRNEXT4-LAUNCHED on success.", "Prints ZZRAPR2-LAUNCHED on success."),
    ("BOX2_BUSY.rnext4_launch_failed_", "BOX2_BUSY.rapr2_launch_failed_"),
    ("RECOVER-NEXT RTL + RP01 four-run chain", "RECOVER-NEXT RAPR two-run chain"),
    ("rnext_chain_rtl_rp01.cmd), owner", "rnext_chain_rapr.cmd), owner"),
    ("PI go 2026-11-03 20:1x", "PI go ${RAPR2_PI_GO:?set RAPR2_PI_GO to the PI go time}"),
    ("ETA ~8 h.", "ETA ~4 h."),
    ('[ "$(echo "$pre" | sed -n 3p)" = "False" ] || die "an old CHAIN_RNEXT.done exists (move it aside first)"',
     'if [ "$(echo "$pre" | sed -n 3p)" = "True" ]; then\n'
     '  st=$(date +%Y%m%d_%H%M%S)\n'
     '  $SSH "powershell -NoProfile -Command ' + BS + '"Move-Item D:/tanitad/runs/v8f_mini_full/rnext/CHAIN_RNEXT.done '
     'D:/tanitad/runs/v8f_mini_full/rnext/CHAIN_RNEXT.rtl_rp01_$st.done' + BS + '"" || die "moving the old CHAIN_RNEXT.done aside failed"\n'
     '  echo "[$(ts)] old CHAIN_RNEXT.done moved aside to CHAIN_RNEXT.rtl_rp01_$st.done"\n'
     'fi'),
    ("for f in rnext_check.py rnext_chain.cmd rnext_chain_rtl_rp01.cmd rnext_run.cmd",
     "for f in rnext_check.py rnext_chain.cmd rnext_chain_rtl_rp01.cmd rnext_chain_rapr.cmd rnext_run.cmd"),
    ("L 'D:" + BS + "tanitad" + BS + "rnext" + BS + "rnext_chain_rtl_rp01.cmd'", "L 'D:" + BS + "tanitad" + BS + "rnext" + BS + "rnext_chain_rapr.cmd'"),
]
for a, b in rep:
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)
for a, b in [("'D:" + BS + "tanitad" + BS + "logs" + BS + "rnext_chain.log' || die \"chain launch\"", "'D:" + BS + "tanitad" + BS + "logs" + BS + "rnext_chain_rapr.log' || die \"chain launch\""),
             ("Get-Content D:" + BS + BS + "tanitad" + BS + BS + "logs" + BS + BS + "rnext_chain.log -Tail 12", "Get-Content D:" + BS + BS + "tanitad" + BS + BS + "logs" + BS + BS + "rnext_chain_rapr.log -Tail 12")]:
    assert s.count(a) == 1, (a, s.count(a))
    s = s.replace(a, b)
s = s.replace("ZZRNEXT4-", "ZZRAPR2-")
assert "RNEXT4" not in s.replace("RNEXT4_WAIT_MAX", "")
(H / "launch_rapr2.sh").write_text(s, encoding="utf-8", newline="\n")
print("wrote launch_rapr2.sh")
