#!/usr/bin/env python3
"""v6F S-W health + per-objective trend + the COLLAPSE question.

⚠️ DISCIPLINE THIS SCRIPT ENFORCES ON ITSELF:
  * a loss term going DOWN is not evidence its measure WORKS — that needs an
    ablation. This reports trends and says so; it does not attribute.
  * `o6_sigreg` is the sigreg LOSS VALUE. It is not collapse evidence. The
    collapse evidence is the SPECTRUM / effective rank, which the trainer emits
    separately (`--spectrum-every 200`). If no spectrum record exists, say so
    rather than reading the loss as a proxy.
  * every number is quoted with its n and its window.
"""
import json
import os

P = os.path.expanduser("~/experiments/v6F-SW-30k/train_log.jsonl")
rows, spec = [], []
for ln in open(P, errors="ignore"):
    if not ln.startswith("{"):
        continue
    try:
        d = json.loads(ln)
    except ValueError:
        continue
    if "spectrum" in d and isinstance(d.get("spectrum"), dict):
        spec.append(d)
    if "step" in d and "loss" in d and "step_s" in d:
        rows.append(d)

# current process only (step_s divisor resets across processes)
import re
NOTE = re.compile(r"over the (\d+) steps")
seg, cur = [], []
for r in rows:
    n = NOTE.search(r.get("step_s_note", ""))
    r["_n"] = int(n.group(1)) if n else 0
    if cur and r["_n"] <= cur[-1]["_n"]:
        seg.append(cur); cur = []
    cur.append(r)
seg.append(cur)
live = seg[-1]

print(f"=== v6F S-W · {len(rows)} logged rows, live process has {len(live)} ===")
print(f"step {live[0]['step']} -> {live[-1]['step']}\n")

KEYS = ["loss", "o1_factual_ade", "o1_ctrl", "o1_fact", "o1_scene",
        "o2_loss", "o2_unweighted", "o3_loss", "o3_visible_err", "o3_mask_rate",
        "o5_loss", "o5_step1", "o5_stepK", "o5_growth", "o6_sigreg",
        "gnorm", "cuda_max_mem_gb"]

def band(rs, k):
    v = [r[k] for r in rs if isinstance(r.get(k), (int, float))]
    return v

print(f"{'term':<18} {'first':>10} {'last':>10} {'delta%':>9}  n")
print("-" * 60)
for k in KEYS:
    v = band(live, k)
    if len(v) < 2:
        # fall back to the whole run for terms the live segment lacks
        v = band(rows, k)
        if len(v) < 2:
            print(f"{k:<18} {'(absent)':>10}")
            continue
    a, b = v[0], v[-1]
    d = (b - a) / abs(a) * 100 if a else float("nan")
    print(f"{k:<18} {a:>10.4f} {b:>10.4f} {d:>+8.1f}%  {len(v)}")

# whole-run trend on the headline, quartiles so a single row cannot drive it
allv = [(r["step"], r["loss"], r.get("o1_factual_ade")) for r in rows
        if isinstance(r.get("loss"), (int, float))]
if len(allv) >= 8:
    q = len(allv) // 4
    print("\n=== whole-run quartile means (all processes, step-ordered) ===")
    for i in range(4):
        ch = allv[i * q:(i + 1) * q] if i < 3 else allv[3 * q:]
        L = sum(x[1] for x in ch) / len(ch)
        A = [x[2] for x in ch if isinstance(x[2], (int, float))]
        print(f"  Q{i+1} steps {ch[0][0]:>5}-{ch[-1][0]:<5} loss {L:7.4f}"
              + (f"  o1_ade {sum(A)/len(A):7.4f}" if A else "") + f"  n={len(ch)}")

print(f"\n=== SPECTRUM / COLLAPSE evidence: {len(spec)} record(s) ===")
if not spec:
    print("  ⛔ NO spectrum record in train_log.jsonl.")
    print("  ⇒ o6_sigreg is the LOSS VALUE only. It is NOT evidence against")
    print("    collapse — a regulariser's loss can fall because the term is")
    print("    satisfied OR because the representation degenerated. The S-W")
    print("    gate's own criterion is O6_rank_retention >= 0.8x EFFECTIVE RANK")
    print("    ACROSS PHASES, which needs the spectrum, not this scalar.")
else:
    for d in spec[-6:]:
        s = d["spectrum"]
        print(f"  step {d.get('step')}: " +
              " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in list(s.items())[:8]))
