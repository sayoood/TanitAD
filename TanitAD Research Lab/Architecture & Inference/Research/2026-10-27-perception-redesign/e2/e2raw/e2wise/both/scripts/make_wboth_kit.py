"""make_wboth_kit.py -- derive the WISE-BOTH kit (PREREG_E2WISE_2026-11-17_ADDENDUM2.md, landed 8153ee7d) from the WISE-ENC kit, LF out,
every substitution count-asserted. Runs ONLY if WISE-ENC fails (d). Also writes e2wboth_weights.sh (Thor): weight_interp for both arms on
S_plan (the 20 prefixes read from WISE-PLAN's provenance) + core.encoder, refusing unless n_interpolated 848, n_copied_in_S 110, outside-S bit-equal.
usage: python -I make_wboth_kit.py <k2dc scratch dir>"""
import sys
from pathlib import Path

D = Path(sys.argv[1])
PAIRS = [("E2WISE_ENC", "E2WISE_BOTH"), ("wise_enc", "wise_both"), ("WISE-ENC", "WISE-BOTH"), ("WENC", "WBOTH"), ("wenc", "wboth")]
NOTE = ("# DERIVED by k2dc/make_wboth_kit.py from the WISE-ENC script of the same role: WISE-BOTH = S_plan (20 prefixes) + core.encoder, alpha 0.5,\n"
        "# pre-registered in e2/e2raw/PREREG_E2WISE_2026-11-17_ADDENDUM2.md (landed 8153ee7d, before any WISE-ENC result). Runs only if WISE-ENC fails (d).\n")
for src in ("e2wenc_stageA.sh", "e2wenc_stageB_launch.sh", "e2wenc_s6c_pull_score.sh", "wait_e2wencA.sh", "wait_e2wencB_then_s6c.sh"):
    s = (D / src).read_bytes().decode("utf-8")
    assert "\r" not in s and "BOTH" not in s and "wboth" not in s, src
    if src == "e2wenc_stageA.sh":
        assert s.count("core.encoder interpolated") == 1
        s = s.replace("core.encoder interpolated", "S_plan + core.encoder interpolated")
    for old, new in PAIRS:
        s = s.replace(old, new)
    assert "wenc" not in s and "WENC" not in s and "_ENC" not in s and "WISE-ENC" not in s, (src, [ln for ln in s.split("\n") if "ENC" in ln.upper() and "ENCOD" not in ln.upper()][:3])
    assert "PYTHONIOENCODING" in s or "PYTHONIOENCODING" not in (D / src).read_text(encoding="utf-8")
    lines = s.split("\n")
    assert lines[0].startswith("#!"), src
    out = "\n".join([lines[0], NOTE.rstrip("\n")] + lines[1:])
    dst = src.replace("wenc", "wboth")
    (D / dst).write_bytes(out.encode("utf-8"))
    print("WROTE", dst, len(out))

W = r'''#!/bin/bash
# e2wboth_weights.sh -- MM 4a3d05, 2026-11-17 (Thor; run from /home/nvidia/r8ns/r9e2): make model_wise_both_a050.pt for K2DC and K0.
# PREREG_E2WISE_2026-11-17_ADDENDUM2.md (8153ee7d): S_both = S_plan (the 20 prefixes recorded in WISE-PLAN's provenance) + core.encoder, alpha 0.5.
# CONTROL (addendum 2): n_interpolated == 848 (324 + 524), n_copied_in_S == WISE-PLAN + WISE-ENC of the SAME arm (K2DC 6+104, K0 0+104), every tensor outside S_both bit-equal; else REFUSE.
# CPU only (fp32 lerp). Ends ZZWBOTHW-DONEZZ or ZZWBOTHW-FAIL-<why>ZZ.
set -u
R=/home/nvidia/r8ns/r9e2; PY=/home/nvidia/venvs/tanitad-train/bin/python; INIT=/home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt
TOOL=$R/inputs/weight_interp.py_a4faef50
cd $R || { echo "ZZWBOTHW-FAIL-CDZZ"; exit 2; }
[ -s $TOOL ] || { echo "ZZWBOTHW-FAIL-TOOLZZ"; exit 2; }
for A in K2DC K0; do
  P=runs/r9e2-$A/model_wise_plan_a050.pt.provenance.json; O=runs/r9e2-$A/model_wise_both_a050.pt
  [ -s $P ] || { echo "ZZWBOTHW-FAIL-NOPLANPROVZZ $P"; exit 3; }
  [ -e $O ] && { echo "ZZWBOTHW-FAIL-EXISTSZZ $O"; exit 3; }
  PFX=$($PY -c 'import json,sys; p=json.load(open(sys.argv[1]))["prefixes"]; assert len(p)==20 and "core.encoder" not in p, p; print(",".join(p+["core.encoder"]))' $P) || { echo "ZZWBOTHW-FAIL-PREFIXZZ"; exit 4; }
  $PY $TOOL --run runs/r9e2-$A/model_ema_tail5k6k.pt --init $INIT --alpha 0.5 --prefixes "$PFX" --out $O > logs/e2wboth_weights_$A.log 2>&1 || { echo "ZZWBOTHW-FAIL-INTERP-$A""ZZ"; tail -3 logs/e2wboth_weights_$A.log; exit 5; }
  $PY -c 'import json,sys; d=json.load(open(sys.argv[1])); p=json.load(open(sys.argv[2])); e=json.load(open(sys.argv[3])); ok=(d["n_interpolated"]==p["n_interpolated"]+e["n_interpolated"]==848 and d["n_copied_in_S"]==p["n_copied_in_S"]+e["n_copied_in_S"] and d["perception_untouched_all_bit_equal"] is True and len(d["prefixes"])==21); print(("CONTROL-OK" if ok else "CONTROL-BAD"), d["n_interpolated"], d["n_copied_in_S"], d["perception_untouched_all_bit_equal"], d["md5_out"]); sys.exit(0 if ok else 1)' $O.provenance.json $P runs/r9e2-$A/model_wise_enc_a050.pt.provenance.json || { echo "ZZWBOTHW-FAIL-CONTROL-$A""ZZ"; exit 6; }
done
echo "ZZWBOTHW-DONEZZ $(date -u +%H:%M:%SZ)"
'''
(D / "e2wboth_weights.sh").write_bytes(W.encode("utf-8"))
print("WROTE e2wboth_weights.sh", len(W))
