import gzip, json, sys, collections
path = sys.argv[1]
n = 0
top_keys = collections.Counter()
smi_present = 0
smi_nonempty = 0
vmax_present = 0
# same-breath positive controls that MUST read non-zero
ctrl_g_tac = 0
ctrl_clip = 0
allkeypaths = collections.Counter()

def walk(o, prefix, depth=0):
    if depth > 3: return
    if isinstance(o, dict):
        for k, v in o.items():
            p = prefix + "." + k if prefix else k
            allkeypaths[p] += 1
            walk(v, p, depth+1)

with gzip.open(path, "rt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        n += 1
        for k in r.keys(): top_keys[k] += 1
        if "speed_max_input" in r:
            smi_present += 1
            v = r["speed_max_input"]
            if v: smi_nonempty += 1
            if isinstance(v, dict) and v.get("v_max_ms") is not None: vmax_present += 1
        if r.get("g_tac"): ctrl_g_tac += 1
        if r.get("clip_id") or r.get("clip") or r.get("id"): ctrl_clip += 1
        walk(r, "")

print("RECORDS n =", n)
print("CONTROL g_tac present (must be >0):", ctrl_g_tac)
print("CONTROL clip-id present (must be >0):", ctrl_clip)
print("speed_max_input present:", smi_present, "/", n)
print("speed_max_input non-empty:", smi_nonempty, "/", n)
print("v_max_ms non-null:", vmax_present, "/", n)
print()
print("=== TOP-LEVEL KEYS (key: count/%d) ===" % n)
for k, c in top_keys.most_common():
    print("  %-34s %6d" % (k, c))
print()
print("=== ALL KEY PATHS depth<=3 containing speed/limit/max/v_ ===")
for k, c in sorted(allkeypaths.items()):
    kl = k.lower()
    if any(t in kl for t in ("speed", "limit", "vmax", "v_max", "max_v", "kmh", "ms_")):
        print("  %-56s %6d" % (k, c))
