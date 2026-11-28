#!/bin/bash
# Note N5 of card ADD45: known-value read on the HALF run of card ADD60, then the read of the QUARTER run of card ADD64.
# last line: ZZ CHAIN-N5 DONE | STOP <where>
cd /f/TanitAD-scratch/v8proof/dscale_lead || exit 2
FRAC_DIR=half1 FRAC_N=2286 POD_H=/workspace/v8proof/dscale_lead/half1 LOCAL_L=/f/TanitAD-scratch/v8proof/dscale_lead/halfscale/half1 bash read_n5_frac.sh > read_n5_half1.out 2>&1
tail -n 1 read_n5_half1.out | cut -c1-200
FRAC_DIR=quarter1 FRAC_N=1143 bash read_n5_frac.sh > read_n5_quarter1.out 2>&1
tail -n 1 read_n5_quarter1.out | cut -c1-200
echo "ZZ CHAIN-N5 DONE"
