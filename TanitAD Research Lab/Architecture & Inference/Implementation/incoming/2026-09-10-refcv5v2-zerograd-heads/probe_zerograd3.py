"""Gap-tolerant shape alignment of optimizer param ids -> parameter names.

The candidate name list over-counts by 3 tensors (buffers my filter missed). Align greedily by
SHAPE, skipping candidates that do not fit, and require EVERY present id to match exactly before
naming any never-trained param.
"""
import torch

CK = r"C:\Users\Admin\refcv5v2_final\ckpt.pt"
ck = torch.load(CK, map_location="cpu", weights_only=False)
sd, opt = ck["model"], ck["opt"]
st = opt["state"]
ids = []
for g in opt["param_groups"]:
    ids.extend(g["params"])
missing = set(i for i in ids if i not in st)

BUF = ("running_mean", "running_var", "num_batches_tracked", "pos_emb_cache", ".anchors")
cand = [n for n, t in sd.items()
        if t.is_floating_point() and not any(b in n for b in BUF)]
id_shape = {i: tuple(st[i]["exp_avg"].shape) for i in st if "exp_avg" in st[i]}

mapping, ptr, skipped = {}, 0, []
for pid in ids:
    if pid in id_shape:
        want = id_shape[pid]
        while ptr < len(cand) and tuple(sd[cand[ptr]].shape) != want:
            skipped.append(cand[ptr]); ptr += 1
        if ptr >= len(cand):
            print("ALIGNMENT FAILED at id", pid); raise SystemExit(1)
        mapping[pid] = cand[ptr]; ptr += 1
    else:
        mapping[pid] = cand[ptr] if ptr < len(cand) else None
        ptr += 1

print("alignment consumed %d of %d candidates; skipped %d as buffers:" % (ptr, len(cand), len(skipped)))
for s in skipped:
    print("    SKIPPED %-52s %s numel=%d" % (s, tuple(sd[s].shape), sd[s].numel()))

bad = [p for p in ids if p in id_shape and tuple(sd[mapping[p]].shape) != id_shape[p]]
print("\nVALIDATION: %d of %d present ids matched their aligned parameter's shape exactly"
      % (len(id_shape) - len(bad), len(id_shape)))
if bad:
    print("  ⛔ MISALIGNED -> naming is INCONCLUSIVE"); raise SystemExit(1)

tot_mapped = sum(sd[mapping[p]].numel() for p in ids if mapping[p])
print("sum numel over the 357 mapped params = %d   (config total 108257502)" % tot_mapped)

print("\n=== PARAMS THAT NEVER RECEIVED A GRADIENT IN 40,284 STEPS ===")
for pid in sorted(missing):
    nm = mapping[pid]
    print("   id %-4s %-52s %-14s numel=%d" % (pid, nm, tuple(sd[nm].shape), sd[nm].numel()))
print("   TOTAL never-trained numel =", sum(sd[mapping[p]].numel() for p in missing))

print("\n=== SAME-BREATH CONTROL: params that MUST have trained ===")
for want in ("str_goal_head", "scorer", "ego_inject", "nav_inject"):
    for pid in ids:
        if mapping[pid] and want in mapping[pid]:
            e = st.get(pid)
            print("   %-52s %s" % (mapping[pid],
                  ("exp_avg_abs_sum=%.6g  TRAINED" % float(e["exp_avg"].abs().sum())) if e
                  else "NO STATE  <-- CONTROL FAILED"))
            break
