import glob, sys, xml.etree.ElementTree as ET
def load(d):
    out = {}
    for f in glob.glob(d + "/*.xml"):
        for tc in ET.parse(f).getroot().iter("testcase"):
            k = tc.get("classname", "") + "::" + tc.get("name", "")
            st = "pass"
            for ch in tc:
                if ch.tag in ("failure", "error"):
                    st = "fail"
                elif ch.tag == "skipped" and st == "pass":
                    st = "skip"
            out[k] = st
    return out
b, n = load(sys.argv[1]), load(sys.argv[2])
bad = [k for k, s in n.items() if s == "fail" and b.get(k) != "fail"]
print("new-tree failures not failing in base:", len(bad))
for k in sorted(bad):
    print(f"  base={b.get(k, 'ABSENT'):6s} {k}")
gone = [k for k in b if k not in n]
print("in base, absent in new:", len(gone), gone[:10])
