"""Make probe_te.py resumable: each arm's out-of-fold predictions (2 s and 6 s) and selections are saved to
OUT/arm_<name>.npz the moment the arm completes; a re-run LOADS completed arms instead of refitting them.
Determinism is preserved: the shuffle permutations are drawn in the same arm order whether or not the arm is skipped
(the draw happens BEFORE the cache check), so a resumed run equals an uninterrupted one."""
import sys

p = "C:/Users/Admin/refav1_probe/probe_te.py"
s = open(p, encoding="utf-8").read()

old_head = '''        perm = rng.permutation(len(gT2)) if a in ("P5_shuf", "P8_shuf_sp") else None
        if perm is not None and not spatial:
            imgT = imgT[perm]
        out = {}
'''
new_head = '''        perm = rng.permutation(len(gT2)) if a in ("P5_shuf", "P8_shuf_sp") else None
        cache = f"{OUT}/arm_{a}.npz"
        if os.path.exists(cache):                  # RESUME: the draw above already happened, in arm order
            z = np.load(cache, allow_pickle=False)
            pred2[a], pred6[a] = z["pred2"], z["pred6"]
            meta["sel"][a] = json.loads(str(z["sel"]))
            print(f"[probe_te] {a} RESUMED from {cache} ADE2={P.ade(pred2[a], gE2).mean():.4f}", flush=True)
            continue
        if perm is not None and not spatial:
            imgT = imgT[perm]
        out = {}
'''
if s.count(old_head) != 1:
    sys.exit(f"head anchor matched {s.count(old_head)} -- refusing")
s = s.replace(old_head, new_head)

old_tail = '''        pred2[a], pred6[a] = out["2s"][0], out["6s"][0]
        meta["sel"][a] = {"2s": out["2s"][1], "6s": out["6s"][1]}
'''
new_tail = '''        pred2[a], pred6[a] = out["2s"][0], out["6s"][0]
        meta["sel"][a] = {"2s": out["2s"][1], "6s": out["6s"][1]}
        tmp = f"{OUT}/arm_{a}.part.npz"
        np.savez(tmp, pred2=pred2[a], pred6=pred6[a], sel=json.dumps(meta["sel"][a]))
        os.replace(tmp, f"{OUT}/arm_{a}.npz")     # atomic: a crash mid-save never leaves a readable half file
'''
if s.count(old_tail) != 1:
    sys.exit(f"tail anchor matched {s.count(old_tail)} -- refusing")
s = s.replace(old_tail, new_tail)

# P0 (the floor) is not fitted and needs no cache; make sure OUT exists before the loop
if "os.makedirs(OUT, exist_ok=True)" not in s:
    sys.exit("OUT creation missing -- refusing")
open(p, "w", encoding="utf-8").write(s)
print("probe_te.py is resumable per arm")
