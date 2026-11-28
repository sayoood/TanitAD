#!/bin/bash
# e2prst_weights.sh -- MM 4a3d05, 2026-11-17 (Thor; CPU). model_wise_plan_a000.pt for K2DC and K0: S_plan (the 20 prefixes recorded in WISE-PLAN's
# provenance) at alpha 0. PREREG_E2WISE_2026-11-17_ADDENDUM3.md (93688696). CONTROL: counts equal to the SAME arm's WISE-PLAN provenance
# (K2DC 324 interpolated / 6 copied; K0 324 / 0, it has no decoder box_gate), outside-S bit-equal, every interpolated tensor torch.equal to
# the init's. Idempotent: an existing output is re-verified, not re-made. Ends ZZPRSTW-DONEZZ or ZZPRSTW-FAIL-<why>ZZ.
set -u
R=/home/nvidia/r8ns/r9e2; PY=/home/nvidia/venvs/tanitad-train/bin/python; INIT=/home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt
TOOL=$R/inputs/weight_interp.py_a4faef50
cd $R || { echo "ZZPRSTW-FAIL-CDZZ"; exit 2; }
for A in K2DC K0; do
  P=runs/r9e2-$A/model_wise_plan_a050.pt.provenance.json; O=runs/r9e2-$A/model_wise_plan_a000.pt
  [ -s $P ] || { echo "ZZPRSTW-FAIL-NOPLANPROVZZ $P"; exit 3; }
  PFX=$($PY -c 'import json,sys; p=json.load(open(sys.argv[1]))["prefixes"]; assert len(p)==20, p; print(",".join(p))' $P) || { echo "ZZPRSTW-FAIL-PREFIXZZ"; exit 4; }
  if [ -e $O ]; then echo "  $A: $O exists (made by an earlier run of this script): re-verifying, not re-making"
  else $PY $TOOL --run runs/r9e2-$A/model_ema_tail5k6k.pt --init $INIT --alpha 0 --prefixes "$PFX" --out $O > logs/e2prst_weights_$A.log 2>&1 || { echo "ZZPRSTW-FAIL-INTERP-$A""ZZ"; tail -3 logs/e2prst_weights_$A.log; exit 5; }; fi
  $PY -c '
import json, sys, torch
d = json.load(open(sys.argv[1] + ".provenance.json"))
ref = json.load(open(sys.argv[3]))   # WISE-PLAN provenance of the SAME arm: counts must match it (K2DC 324/6, K0 324/0 -- K0 has no box_gate)
ok = (d["n_interpolated"] == ref["n_interpolated"] == 324 and d["n_copied_in_S"] == ref["n_copied_in_S"] and d["n_untouched"] == ref["n_untouched"]
      and d["prefixes"] == ref["prefixes"] and d["perception_untouched_all_bit_equal"] is True and d["alpha"] == 0)
def sd(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    best = None
    for v in ([ck] + [x for x in ck.values() if isinstance(x, dict)]):
        n = sum(1 for t in v.values() if torch.is_tensor(t))
        if best is None or n > best[0]:
            best = (n, v)
    return best[1]
o, i = sd(sys.argv[1]), sd(sys.argv[2])
eq = sum(1 for k in d["interpolated"] if torch.equal(o[k], i[k]))
ok = ok and eq == len(d["interpolated"]) == 324
print("CONTROL-" + ("OK" if ok else "BAD"), d["n_interpolated"], d["n_copied_in_S"], d["perception_untouched_all_bit_equal"], "init-equal", eq, d["md5_out"])
sys.exit(0 if ok else 1)' $O $INIT $P || { echo "ZZPRSTW-FAIL-CONTROL-$A""ZZ"; exit 6; }
done
echo "ZZPRSTW-DONEZZ $(date -u +%H:%M:%SZ)"
