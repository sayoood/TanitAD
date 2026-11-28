"""R2 fix #4 (found by the v7f gate profile's G-DVB, 2026-09-27): `--nav-cond` passed preflight but was never mapped
into V6Config by build_stack_from_args, so every v7-line launch built a model WITHOUT the nav conditioner. EOL kept."""
import sys
from pathlib import Path

CRLF = chr(13) + chr(10)
p = Path("C:/Users/Admin/v7f_merge/stack/scripts/train_v6_staged.py")
eol = CRLF if CRLF.encode() in p.read_bytes() else chr(10)
s = open(p, encoding="utf-8").read()
anchor = '        strategic_off=bool(getattr(a, "strategic_off", False)),\n'
new = (anchor +
       '        # ⛔⛔ R2 (PI 2026-09-27): THE NAV CONDITIONER. MEASURED 2026-09-27 by the v7f launch-gate\n'
       '        # profile (G-DVB): `--nav-cond` was REQUIRED by preflight for every v7-line run and then\n'
       '        # NEVER mapped here, so every such launch built a stack with NO nav conditioner -- a\n'
       '        # declared lever the built model did not have (the D-REFCV6-CONFIG-BUILD class). The\n'
       '        # nav fixes pinned by tests/test_v7f_r2_nav_fixes.py were tested on configs built\n'
       '        # directly and so never reached a real launch. Default False = byte-identical.\n'
       '        nav_cond=bool(getattr(a, "nav_cond", False)),\n')
n = s.count(anchor)
if n != 1:
    sys.exit(f"anchor matched {n} times -- refusing")
if 'nav_cond=bool(getattr(a, "nav_cond", False))' in s:
    sys.exit("nav_cond mapping already present -- refusing a duplicate")
open(p, "w", encoding="utf-8", newline=eol).write(s.replace(anchor, new))
print("nav_cond mapped in build_stack_from_args")
