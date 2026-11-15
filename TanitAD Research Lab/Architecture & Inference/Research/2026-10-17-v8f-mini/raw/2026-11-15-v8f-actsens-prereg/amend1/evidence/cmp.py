import hashlib, json, sys
D = "C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/c3diag"
R = "C:/Users/Admin/t2v8f_pulls/ladder_R35_s0"
rb = open(R + "/selfcheck.json", "rb").read()
print("R35_s0 selfcheck sha", hashlib.sha256(rb).hexdigest()[:16])
r = json.loads(rb)
a = json.load(open(D + "/sc_IDM_s0.json", encoding="utf-8"))


def flat(o, pre=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flat(v, pre + "." + str(k) if pre else str(k)))
    elif isinstance(o, list) and len(o) <= 8 and all(not isinstance(x, (dict, list)) for x in o):
        out[pre] = o
    elif isinstance(o, list):
        out[pre] = "list[%d] sha %s" % (len(o), hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()[:10])
    else:
        out[pre] = o
    return out


fr, fa = flat(r), flat(a)
keys = sorted(set(fr) | set(fa))
n = 0
for k in keys:
    if k.startswith("live.terms") or k.startswith("live.wm_groups_grad"):
        continue
    if fr.get(k) != fa.get(k):
        n += 1
        print("%-70s R35=%r  IDM=%r" % (k[:70], str(fr.get(k, "<absent>"))[:70], str(fa.get(k, "<absent>"))[:70]))
print("differing non-term keys:", n)
print("argv R35:", open(R + "/argv.txt", encoding="utf-8").read().strip()[:2000])
print("argv IDM:", open(D + "/../c3diag/sc_IDM_s0.json", encoding="utf-8").read()[:0])
