import json, re
p = "/home/nvidia/experiments/v6F-SW-30k/train_log.jsonl"
rows = []
for ln in open(p, errors="ignore"):
    if ln.startswith("{") and "step_s" in ln:
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        m = re.search(r"over the (\d+) steps", d.get("step_s_note", ""))
        if m:
            rows.append((d["step"], int(m.group(1)), d))
segs, cur = [], []
for r in rows:
    if cur and r[1] <= cur[-1][1]:
        segs.append(cur); cur = []
    cur.append(r)
segs.append(cur)

def show(tag, seg):
    print(tag)
    for s, n, d in seg:
        print("  step %5d  loss %.4f  o1_fact_ade %.4f  gnorm %s  o5 %s"
              % (s, d["loss"], d["o1_factual_ade"], d["gnorm"], d.get("o5_loss")))

show("A40 tail (trained 0..6300):", segs[0][-3:])
show("THOR (resumed from the 6250 ckpt):", segs[-1])
