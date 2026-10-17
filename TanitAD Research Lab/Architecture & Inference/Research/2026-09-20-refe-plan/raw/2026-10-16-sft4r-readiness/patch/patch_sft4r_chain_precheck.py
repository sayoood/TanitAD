#!/usr/bin/env python3
"""One-shot patcher (SFT-4R readiness, 2026-10-16): the chain's precheck must demand the new G-SUITE entry.  usage: patch_sft4r_chain_precheck.py <refe dir>
refe_sft4r's G-SUITE now runs refe/test_suite_nesting.py (patch_suite_nesting.py).  A package shipped without it would only fail LATE (G-SUITE: rc 2 on the missing file, after the GPU wait, the
preflight and both deliberate regressions); the precheck refuses it at stage 0 with the file's name.  Applies on top of the v2 chain (the post-image in ../post/refe/sft4r_chain.sh)."""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "sft4r_chain.sh"
s = open(p, encoding="utf-8", newline="").read()
assert "\r\n" not in s
old = "refe/sft4r_pod.sh refe/sft4r_chain.sh refe/test_sft4r_chain.py refe/test_sft4r_gate.py refe/test_launch_gate.py refe/train.py"
assert s.count(old) == 1
s = s.replace(old, "refe/sft4r_pod.sh refe/sft4r_chain.sh refe/test_sft4r_chain.py refe/test_sft4r_gate.py refe/test_suite_nesting.py refe/test_launch_gate.py refe/train.py")
open(p, "w", encoding="utf-8", newline="").write(s)
print("patched", p.name)
