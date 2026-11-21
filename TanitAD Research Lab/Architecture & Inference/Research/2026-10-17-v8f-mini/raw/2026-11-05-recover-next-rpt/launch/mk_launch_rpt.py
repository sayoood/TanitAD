"""Derive launch_rpt.sh (RPT = R's frozen argv with --recover-dev-shape pertick: RPT_0 RPT_1) from launch_raprp01.sh. Every replacement must match exactly the stated number of times.
New against the RAPR-P01 launcher: a refusal without RPT_PI_GO, and the STACK-SHIP block (three files over box 2's repo D:/tanitad/repo, pinned sha256 read from box2_files/rnext_check.py SHIPPED)."""
import importlib.util
import re
from pathlib import Path

H = Path(__file__).resolve().parent
s = (H / "launch_raprp01.sh").read_text(encoding="utf-8")
BS = "\\"
NL = chr(10)


def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


pins = load("mk_rpt_pins_for_launch", H / "mk_rpt_pins.py")
shipped = pins.current()                                   # the SHIPPED block of rnext_check.py (the checker binds exactly these bytes)
assert shipped == pins.computed(), "SHIPPED differs from the source files: run mk_rpt_pins.py first"
rep = [
    ("# RECOVER-NEXT addendum 2 row E (RAPR-P01 = RAPR's frozen argv with --recover-p 0.1, 2 runs): wait for box 2's lock",
     "# RECOVER-NEXT addendum 4 (RPT = R's frozen argv with --recover-dev-shape pertick, 2 runs): wait for box 2's lock", 1),
    ("ship, verify bytes, start telemetry -> guard -> the two-run chain (rnext_chain_raprp01.cmd). RUN ONLY",
     "ship, verify bytes, SHIP THE THREE STACK FILES (trainer, v8f_recover, v8f_mini_data: sha256 pinned, box 2's old files moved aside, re-hashed on box 2), start telemetry -> guard -> the two-run chain (rnext_chain_rpt.cmd). RUN ONLY", 1),
    ("unchanged except the chain file.", "unchanged except the chain file and the stack ship.", 1),
    ("BOX2_BUSY.raprp01_launch_failed_", "BOX2_BUSY.rpt_launch_failed_", 1),
    ("set -u" + NL + "SSH=", "set -u" + NL + ': "${RPT_PI_GO:?set RPT_PI_GO to the PI go time (HH:MM): this launcher refuses to run without the go of the PI}"' + NL + "SSH=", 1),
    ("RECOVER-NEXT RAPR-P01 two-run chain (D:" + BS + BS + "tanitad" + BS + BS + "rnext" + BS + BS + "rnext_chain_raprp01.cmd), owner",
     "RECOVER-NEXT RPT two-run chain (D:" + BS + BS + "tanitad" + BS + BS + "rnext" + BS + BS + "rnext_chain_rpt.cmd), owner", 1),
    ("PI go ${RAPRP01_PI_GO:?set RAPRP01_PI_GO to the PI go time}", "PI go ${RPT_PI_GO:?set RPT_PI_GO to the PI go time}", 1),
    ("CHAIN_RNEXT.rapr_$st.done", "CHAIN_RNEXT.raprp01_$st.done", 2),
    ("rnext_chain_rapr.cmd rnext_chain_raprp01.cmd rnext_run.cmd", "rnext_chain_rapr.cmd rnext_chain_raprp01.cmd rnext_chain_rpt.cmd rnext_run.cmd", 1),
    ("for t in RAPR_0 RAPR_1 RTL_0 RTL_1 RP01_0 RP01_1 RAPRP01_0 RAPRP01_1; do", "for t in RAPR_0 RAPR_1 RTL_0 RTL_1 RP01_0 RP01_1 RAPRP01_0 RAPRP01_1 RPT_0 RPT_1; do", 1),
    ("L 'D:" + BS + "tanitad" + BS + "rnext" + BS + "rnext_chain_raprp01.cmd' 'D:" + BS + "tanitad" + BS + "logs" + BS + "rnext_chain_raprp01.log'",
     "L 'D:" + BS + "tanitad" + BS + "rnext" + BS + "rnext_chain_rpt.cmd' 'D:" + BS + "tanitad" + BS + "logs" + BS + "rnext_chain_rpt.log'", 1),
    ("Get-Content D:" + BS + BS + "tanitad" + BS + BS + "logs" + BS + BS + "rnext_chain_raprp01.log -Tail 12",
     "Get-Content D:" + BS + BS + "tanitad" + BS + BS + "logs" + BS + BS + "rnext_chain_rpt.log -Tail 12", 1),
]
for a, b, n in rep:
    assert s.count(a) == n, (a[:80], s.count(a), n)
    s = s.replace(a, b)
assert s.count("ZZRAPRP01-") == 6, s.count("ZZRAPRP01-")
s = s.replace("ZZRAPRP01-", "ZZRPT-")
left = [l for l in s.splitlines() if "RAPRP01" in l.upper() or "RAPR-P01" in l]          # the RAPR-P01 names that remain are the ones that must: its chain file / argv tags still shipped (the checker audits every shipped argv), and its marker's moved-aside name
assert "RAPR-P01" not in s and len(left) == 4 and sum("CHAIN_RNEXT.raprp01_" in l for l in left) == 2 and sum("rnext_chain_raprp01.cmd" in l or "RAPRP01_0 RAPRP01_1" in l for l in left) == 2, left

STACK = "/e/Projects/TanitAD-artifacts/dagland/pertick/stack_patch"
rows = []
for rel, h in shipped.items():
    src = pins.SOURCES[rel]
    src_sh = (STACK + "/" + rel) if str(src).replace("\\", "/").startswith(str(pins.PERTICK).replace("\\", "/")) else "$SRC/../rpt_stack/" + rel
    rows.append(f'  "{rel}|{src_sh}|{h}"')
block = r'''# ---- STACK-SHIP-BEGIN (RPT: the new trainer lever --recover-dev-shape, its library and the data module's ClipStore.get_kin: three files over box 2's repo, before the telemetry)
# Local sha256 of each source against the PINNED value first (nothing on box 2 is touched on a mismatch). Then per file: box 2's current file is MOVED ASIDE to <name>.pre_rpt_<ts> (never deleted), the new one is scp'd, and re-hashed ON box 2;
# any failure puts every file shipped so far back (the new one aside as <name>.rpt_failed_<ts>, the old one back in place) and dies, which releases our lock.
REPO2=D:/tanitad/repo; STAMP=$(date +%Y%m%d_%H%M%S); STK_DONE=()
STK=(
__ROWS__
)
restore_stack() { local e rel; for e in "${STK_DONE[@]:-}"; do [ -n "$e" ] || continue; rel=${e%%|*}; $SSH "powershell -NoProfile -Command \"if (Test-Path $REPO2/$rel) { Move-Item $REPO2/$rel $REPO2/$rel.rpt_failed_$STAMP }; if (Test-Path $REPO2/$rel.pre_rpt_$STAMP) { Move-Item $REPO2/$rel.pre_rpt_$STAMP $REPO2/$rel }\"" || echo "[$(ts)] WARNING restore of $rel failed: see $REPO2/$rel.pre_rpt_$STAMP"; done; }
for e in "${STK[@]}"; do
  IFS='|' read -r rel src want <<< "$e"
  [ -f "$src" ] || die "stack ship: source $src missing"
  got=$(sha256sum "$src" | cut -c1-64); [ "$got" = "$want" ] || die "stack ship: $src sha256 $got != pinned $want (re-pin with mk_rpt_pins.py, then regenerate this launcher)"
done
for e in "${STK[@]}"; do
  IFS='|' read -r rel src want <<< "$e"
  $SSH "powershell -NoProfile -Command \"if (Test-Path $REPO2/$rel) { Move-Item $REPO2/$rel $REPO2/$rel.pre_rpt_$STAMP }\"" || { restore_stack; die "moving box 2's $rel aside failed"; }
  STK_DONE+=("$rel|$src|$want")
  $SCP "$src" "tanitad-box2:$REPO2/$rel" || { restore_stack; die "scp $rel"; }
  have=$($SSH "powershell -NoProfile -Command \"(Get-FileHash $REPO2/$rel -Algorithm SHA256).Hash.ToLower()\"" | tr -d '\r') || { restore_stack; die "re-hash of $rel on box 2 failed"; }
  [ "$have" = "$want" ] || { restore_stack; die "stack ship: $rel on box 2 hashes '$have', expected $want"; }
  echo "[$(ts)] shipped $rel (${want:0:12}), old file at $rel.pre_rpt_$STAMP"
done
# ---- STACK-SHIP-END
'''.replace("__ROWS__", "\n".join(rows))
anchor = "# 4. telemetry first (must be FRESH), then the guard, then the chain\n"
assert s.count(anchor) == 1
s = s.replace(anchor, block + "\n" + anchor)
(H / "launch_rpt.sh").write_text(s, encoding="utf-8", newline="\n")
print("wrote launch_rpt.sh")
