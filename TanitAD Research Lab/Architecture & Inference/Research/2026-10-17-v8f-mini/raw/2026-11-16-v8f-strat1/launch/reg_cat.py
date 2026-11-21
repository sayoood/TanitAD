"""Pod-side: count the G-SUITE regressions by cause (nav_source G-DVB / T1 tool pin / frozen-grad probe / other)."""
import collections
import glob
import json
import xml.etree.ElementTree as ET

reg = set(json.load(open("/root/strat1/gsuite_result.json"))["regressions"])
cat = collections.Counter()
other = []
seen = set()
for f in sorted(glob.glob("/root/strat1/junit/new_c2/*.xml")):
    for tc in ET.parse(f).getroot().iter("testcase"):
        tid = (tc.get("classname") or "") + "::" + (tc.get("name") or "")
        if tid not in reg:
            continue
        seen.add(tid)
        t = " ".join(((ch.get("message") or "") + (ch.text or "")) for ch in tc if ch.tag in ("failure", "error"))
        if "nav_source: declared v9 but the data was built from None" in t:
            k = "navsrc"
        elif "the T1 tool" in t:
            k = "t1pin"
        elif "does not require grad" in t:
            k = "grad"
        else:
            k = "other"
        cat[k] += 1
        if k == "other":
            other.append(tid + " :: " + t[:400].replace("\n", " "))
print(len(reg), len(seen), dict(cat))
print(collections.Counter(x.split("::")[0] for x in other))
print("unseen", sorted(reg - seen)[:12])
