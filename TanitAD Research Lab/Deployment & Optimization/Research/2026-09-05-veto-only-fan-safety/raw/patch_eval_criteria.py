#!/usr/bin/env python3
"""Add the criteria-completeness check to the veto T1 eval chain.

The four-families rule is BINDING, and "a missing metric is a work item, not an excuse".
`tools/criteria_check.py` machine-checks the eval artifact against
`products/P7-TanitEval/CRITERIA_REGISTRY.json`, so a family that is absent is NAMED rather
than quietly missing from a table nobody diffed.
"""
import io
import os

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "..", "veto_run", "run_eval_veto.sh")
P = os.path.normpath(r"C:\Users\Admin\veto_run\run_eval_veto.sh")
BS = chr(92)

s = io.open(P, encoding="utf-8").read().replace(chr(13) + chr(10), chr(10))
anchor = '    echo "ZZFANSAFE-$tag-rc$?-ZZ"' + chr(10)
if "criteria_check.py" in s:
    raise SystemExit("[patch] criteria check already present")
if s.count(anchor) != 1:
    raise SystemExit("[patch] anchor found %d times" % s.count(anchor))

block = (anchor
         + "    # The FOUR-FAMILIES rule is BINDING and machine-checked here, not asserted in" + chr(10)
         + "    # prose: a missing family is a WORK ITEM, and the checker NAMES it rather than" + chr(10)
         + "    # letting it be absent from a table nobody diffed." + chr(10)
         + '    "$PY" -u "$REPO/tools/criteria_check.py" "$OUT/eval/refcv3-40284-$tag.json" ' + BS + chr(10)
         + '       --registry "$REPO/products/P7-TanitEval/CRITERIA_REGISTRY.json" ' + BS + chr(10)
         + '       --json "$OUT/eval/criteria_${tag}.json" >> "$OUT/eval_$tag.log" 2>&1' + chr(10)
         + '    echo "ZZCRITERIA-$tag-rc$?-ZZ"' + chr(10))
io.open(P, "w", encoding="utf-8", newline=chr(10)).write(s.replace(anchor, block))
print("[patch] criteria check added to run_eval_veto.sh")
