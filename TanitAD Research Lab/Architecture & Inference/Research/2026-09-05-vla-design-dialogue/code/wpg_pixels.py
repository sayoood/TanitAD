"""WP-G precondition 2 — is a `rider` even RESOLVED in our frames?

⛔ WHY THIS RUNS BEFORE ANY TEACHER IS LOADED. WP-G asks whether a strong backbone
can separate motorcycle from bicycle inside the `rider` class. But no backbone can
classify what the sensor did not resolve. If a rider occupies ~10 px in our
256x640 cylindrical frames, the distillation path is closed at the DATA, not at
the model, and loading DINOv3 would answer a question the pixels already settled.

⚠️ AND THE SCORING QUESTION IS SEPARATE AND HARDER: we cannot measure a teacher's
motorcycle-vs-bicycle accuracy, because those labels are precisely what the corpus
lacks (measured: no `motorcycle`, no `bicycle`; both collapse into `rider`). So
WP-G proper must be an UNSUPERVISED separability test with a known-separable
control pair. This script settles whether that is worth attempting.

Projection: CYLINDRICAL, f_ref 305.5774907364391, W=640, H=256.
⛔ Column is LINEAR IN AZIMUTH; the pinhole formula is wrong here (CLAUDE.md FOV
trap). For a small object at range d the angular width is w/d, so
    px_width  = f_ref * (w / d)      <- cylindrical: f * angle
    px_height = f_ref * (h / d)      <- rows are still a tangent projection, but
                                        for small angles the two agree to <2%
⚠️ Reported as an ESTIMATE with the approximation named, not as a measurement of
the actual rendered crop.

Evidence class: MEASURED (agent geometry) + ESTIMATED (pixel footprint).
"""
import collections
import io
import json
import statistics as st

F_REF = 305.5774907364391
RIDER_H_M = 1.7          # a rider is taller than the cuboid's `l`/`w` suggest

p = r"C:/Users/Admin/tanitad-data/rl-pilot/pilot_val_agents.jsonl"
by_cls = collections.defaultdict(list)
for line in io.open(p, encoding="utf-8"):
    d = json.loads(line)
    for a in d["agents"]:
        cx, cy = float(a["cx"]), float(a["cy"])
        rng = (cx * cx + cy * cy) ** 0.5
        if rng < 0.5:
            continue
        by_cls[a.get("cls", "?")].append((rng, float(a.get("w", 0.8)),
                                          float(a.get("l", 1.9))))

print(f"{'class':<16}{'n':>7}{'range p50':>11}{'p10':>8}"
      f"{'px wide p50':>13}{'px tall p50':>13}{'>=32px tall':>12}")
out = {}
for c, v in sorted(by_cls.items(), key=lambda x: -len(x[1])):
    if len(v) < 50:
        continue
    rngs = [x[0] for x in v]
    w = st.median([x[1] for x in v])
    h = RIDER_H_M if c in ("rider", "person") else st.median([x[2] for x in v])
    r50, r10 = st.median(rngs), sorted(rngs)[len(rngs) // 10]
    pw, ph = F_REF * w / r50, F_REF * h / r50
    big = sum(1 for r in rngs if F_REF * h / r >= 32) / len(rngs)
    out[c] = {"n": len(v), "range_p50": r50, "px_w_p50": pw, "px_h_p50": ph,
              "frac_ge_32px_tall": big}
    print(f"{c:<16}{len(v):>7,}{r50:>11.1f}{r10:>8.1f}{pw:>13.1f}{ph:>13.1f}"
          f"{100*big:>11.1f}%")

r = out.get("rider")
if r:
    print()
    print(f"  RIDER at the median range {r['range_p50']:.1f} m: "
          f"~{r['px_w_p50']:.0f} x {r['px_h_p50']:.0f} px")
    print(f"  fraction of riders >= 32 px tall: {100*r['frac_ge_32px_tall']:.1f}%")
    print()
    # ⭐ the reference every vision backbone is trained at
    print("  reference: DINOv2/v3 and CLIP patch backbones use 14 or 16 px PATCHES.")
    print(f"  => a median rider spans ~{r['px_h_p50']/14:.1f} patches vertically "
          f"and ~{r['px_w_p50']/14:.1f} horizontally.")
    verdict = ("RESOLVED — a teacher has something to look at; WP-G proper is worth running"
               if r["px_h_p50"] >= 32 and r["frac_ge_32px_tall"] > 0.4 else
               "MARGINAL — only the near subset is resolved; scope WP-G to it and say so"
               if r["frac_ge_32px_tall"] > 0.15 else
               "NOT RESOLVED — the distillation path is closed at the DATA; "
               "no backbone can classify this footprint")
    print(f"\n  => {verdict}")
    out["_verdict"] = verdict
out["_note"] = ("cylindrical projection, f_ref 305.577; px = f * size / range. "
                "ESTIMATE of the footprint, not a measurement of the rendered crop.")
json.dump(out, open(r"C:/Users/Admin/tanitad-data/rl-pilot/wpg_pixels.json", "w"),
          indent=1)
print("-> wpg_pixels.json")
