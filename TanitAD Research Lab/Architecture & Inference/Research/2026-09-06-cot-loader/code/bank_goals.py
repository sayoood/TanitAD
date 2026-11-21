"""PR-2f control: bank `goals_from_cot()` over all 4,729 clips.

Run BEFORE the capture patch and AFTER it; the two dumps must be byte-identical.
A change here would mean the capture altered the emitted vocabulary, which the
pre-registration forbids. ASCII-only stdout.

usage: bank_goals.py <out.json>
"""
import hashlib
import json
import sys

from tanitad.data import alpamayo_fusion as AF
from tanitad.data import alpamayo_records as AR
from tanitad.data import cot_tokens_v7 as COT

out = {}
clips = sorted(AR.available())
for cid in clips:
    c = AR.get(cid)
    rich = AF.cot_text(cid)
    out[cid] = {
        "from_cot": COT.goals_from_cot(c.cot),
        "from_rich": COT.goals_from_cot(rich),
    }
blob = json.dumps(out, sort_keys=True, ensure_ascii=True)
open(sys.argv[1], "w", encoding="ascii").write(blob)
print("clips=%d  md5=%s  bytes=%d"
      % (len(clips), hashlib.md5(blob.encode("ascii")).hexdigest(), len(blob)))
