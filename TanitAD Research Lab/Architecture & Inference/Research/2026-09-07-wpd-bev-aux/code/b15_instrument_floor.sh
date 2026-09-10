#!/usr/bin/env bash
# WP-D step 15 -- MEASURE THE INSTRUMENT FLOOR, the control A3's floor needs.
#
# `tok_D0dup` reads the IDENTICAL feature file as `tok_D0`: same bytes, same head
# architecture, same seed, same split, same invocation. The TRUE value of
# |AP(D0dup) - AP(D0)| is KNOWN TO BE ZERO. Whatever it reads is what the PROBE
# reports as a difference when there is none -- and every gap in this panel,
# including the +0.00173 lever gap A3 is tested against, inherits it.
#
# ⛔ The authoritative 5-arm score matrices are backed up in _pt_authoritative/
# BEFORE this runs, because b2 writes pt_* into the bank and would otherwise
# overwrite the panel this control exists to interpret.
set -u
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONIOENCODING=utf-8
CODE=/c/Users/Admin/wpd-probe/code
BANK='C:\Users\Admin\wpd-probe\bank_rep'
BK=/c/Users/Admin/wpd-probe/bank_rep/_pt_authoritative
RAW=/c/Users/Admin/wpd-probe/raw
cd "$CODE" || exit 9

[ -f "$BK/pt_cart_tok_D0.npy" ] || { echo "NO_BACKUP -- refusing"; exit 8; }

echo "=== instrument-floor probe: tok_D0 + tok_D0dup, ONE invocation ==="
"$PY" b2_probe_wpd.py --bank "$BANK" --geom cart --az-sign prog \
      --arms tok_D0,tok_D0dup --n-boot 2 \
      --out 'C:\Users\Admin\wpd-probe\raw\panel_instrument_cart.json' \
      > "$RAW/probe_instrument_cart.log" 2>&1
echo "PROBE_EXIT=$?"
grep -E "^\[CONTROL\]|^\[tok_" "$RAW/probe_instrument_cart.log"

echo "=== exact 2000-draw paired CI on the two identical-input arms ==="
"$PY" b5_fast_boot.py --bank "$BANK" --geom cart --n-boot 2000 \
      --arms tok_D0,tok_D0dup \
      --out 'C:\Users\Admin\wpd-probe\raw\boot_instrument_cart.json' \
      > "$RAW/boot_instrument_cart.log" 2>&1
echo "BOOT_EXIT=$?"
grep -E "^\[draws\]|^\[pair\]|   fast" "$RAW/boot_instrument_cart.log"

echo "=== RESTORE the authoritative score matrices ==="
cp "$BK"/pt_cart_tok_D0.npy  /c/Users/Admin/wpd-probe/bank_rep/pt_cart_tok_D0.npy
for f in D0b D0c D1 D2; do
  cp "$BK/pt_cart_tok_$f.npy" "/c/Users/Admin/wpd-probe/bank_rep/pt_cart_tok_$f.npy"
done
"$PY" - <<'PY'
# assert the restore by CONTENT, never by the cp exit code
import hashlib, os
B = r"C:\Users\Admin\wpd-probe\bank_rep"
ok = True
for a in ("D0", "D0b", "D0c", "D1", "D2"):
    h = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
    x = h(os.path.join(B, "_pt_authoritative", f"pt_cart_tok_{a}.npy"))
    y = h(os.path.join(B, f"pt_cart_tok_{a}.npy"))
    good = len(x) == 32 and len(y) == 32 and x == y
    ok &= good
    print(f"[restore] pt_cart_tok_{a}.npy {'VERIFIED' if good else 'MISMATCH'} {x[:12]}")
print("RESTORE_ALL_VERIFIED" if ok else "RESTORE_FAILED")
PY
echo "B15_DONE"
