#!/usr/bin/env python
# run_on_probe.py : the gap-filter read of card PREREG_V8F_PROOF_ADD28 on the token-probe arms of card PREREG_V8F_PROOF_ADD27
# (the next step the card commits). gapfilter_read.py is imported unedited; the only change is the reference file of its
# alignment control x1, which for the probe arms is the finished token-probe read (same layout: arms[arm].M1.r2, arms[arm].M3.delta_n1.skill).
# Usage: python run_on_probe.py run --arms TK16_s0,... --in-dir <tokprobe/in> --states-dir <tokstates> --out-dir <dir> [--variant-jump-m 3.0]
import sys
import gapfilter_read as G

G.N1_RESULT = G.ROOT + "/tokprobe/out2/RESULT_TOKPROBE.json"

if __name__ == "__main__":
    sys.exit(G.main())
