"""dose_from_telemetry.py <arm> <w_map_hires_trained> <out.json> -- MM 4a3d05, 2026-11-12, DIAGNOSTIC, read-only, Thor-side.
Reads the trainer's OWN grad-share telemetry (gs_<group>_{norm,proj}_{map_hires,tac_v6,rest,total}: the launch gate's G-LIVE estimator, logged at the
--grad-share-at steps, one batch each) from runs/r9e2-<arm>/metrics.jsonl and, per step and group, recovers a = map grad at the trained weight,
b = everything else (rest + tac_v6), <a,b> = (|a+b|^2 - |a|^2 - |b|^2) / 2.  With a1 = a / w (per unit weight) the share at weight k is exactly
PREREG_K2_REDOSE's s(k) = (k^2|a1|^2 + k<a1,b>) / (k^2|a1|^2 + 2k<a1,b> + |b|^2).  Prints the recorded share, the reconstruction check, and the
weight k that puts the share at 0.05 / 0.10 / 0.15 (design 2.2 dose band 0.05..0.15) at that step.
|b| uses norm_rest and norm_tac_v6 as orthogonal-free pieces: |b|^2 = |total - a|^2 is computed from the identity instead, so no assumption."""
import json
import math
import sys

arm, W, OUT = sys.argv[1], float(sys.argv[2]), sys.argv[3]
rows = [json.loads(l) for l in open(f"/home/nvidia/r8ns/r9e2/runs/r9e2-{arm}/metrics.jsonl", encoding="utf-8") if l.strip()]
res = {"class": "DIAGNOSTIC", "source": f"runs/r9e2-{arm}/metrics.jsonl gs_* (trainer grad-share telemetry = the launch gate's G-LIVE estimator, one batch per step)",
       "w_trained": W, "band": [0.05, 0.15], "steps": {}}


def share(k, A2, AB, B2):
    return (k * k * A2 + k * AB) / (k * k * A2 + 2 * k * AB + B2)


def solve(t, A2, AB, B2):
    lo, hi = 1e-4, 1e4
    if share(hi, A2, AB, B2) < t:
        return None
    for _ in range(200):
        m = math.sqrt(lo * hi)
        if share(m, A2, AB, B2) < t:
            lo = m
        else:
            hi = m
    return round(hi, 3)


for r in rows:
    s = r.get("step")
    groups = sorted({k[3:].split("_norm_")[0] for k in r if k.startswith("gs_") and "_norm_total" in k})
    for g in groups:
        nm, nt = r.get(f"gs_{g}_norm_map_hires"), r.get(f"gs_{g}_norm_total")
        pm = r.get(f"gs_{g}_proj_map_hires")
        if nm is None or nt is None or pm is None or nt == 0:
            continue
        # proj_map = <a, a+b> / |a+b|^2  ->  <a,b> = pm*|a+b|^2 - |a|^2 ; |b|^2 = |a+b|^2 - |a|^2 - 2<a,b>
        T2, A2 = nt * nt, nm * nm
        AB = pm * T2 - A2
        B2 = T2 - A2 - 2 * AB
        a1A2, a1AB = A2 / W / W, AB / W
        rec = share(W, a1A2, a1AB, B2)
        d = {"share_recorded": round(pm, 5), "share_reconstructed_at_w": round(rec, 5), "norm_map": round(nm, 4), "norm_b": round(math.sqrt(max(B2, 0)), 4),
             "cos_ab": round(AB / math.sqrt(A2 * B2), 5) if A2 > 0 and B2 > 0 else None,
             "share_at_w1": round(share(1.0, a1A2, a1AB, B2), 5),
             "k_for_0.05": solve(0.05, a1A2, a1AB, B2), "k_for_0.10": solve(0.10, a1A2, a1AB, B2), "k_for_0.15": solve(0.15, a1A2, a1AB, B2)}
        res["steps"].setdefault(str(s), {})[g] = d
        print(f"step {s} {g:8s} share {pm:.4f} (rec {rec:.4f}) |a| {nm:.2f} |b| {d['norm_b']:.2f} cos {d['cos_ab']} s(w=1) {d['share_at_w1']:.5f} "
              f"k(0.05) {d['k_for_0.05']} k(0.10) {d['k_for_0.10']} k(0.15) {d['k_for_0.15']}")
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
print("wrote", OUT)
