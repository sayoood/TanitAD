"""WP-I null-adjustment — the panel's raw AUCs are NOT comparable across arms.

⛔⛔ THE DEFECT THIS EXISTS TO REPAIR, AND IT IS THE TENTH THIS CAMPAIGN.
`wpi2_semantic_floor.py` prints a table of raw AUCs, four arms wide, and invites
the reader to compare down the columns. **That comparison is invalid**, because
the four arms do not share a chance level. MEASURED on `vru<60m IN-FOV`, the
WITHIN-EPISODE permutation null is centred at:

    pixels 0.4758 · rand 0.5256 · trunk 0.5786 · dinov3 0.4833

A null at 0.5786 means that on this target, with this estimator, **an arm with
the trunk's geometry scores 0.579 on labels that carry no information at all.**
So the trunk's observed 0.4917 is not "slightly below the pixel arm" -- it is
**0.087 BELOW ITS OWN CHANCE LEVEL**, while dinov3's 0.6772 is **0.194 ABOVE**
its own. Reading the raw column would have ranked those two arms 0.49 vs 0.68, a
gap of 0.19; the honest gap is 0.28 and it points the same way but for a
different reason than the raw numbers suggest.

⭐ WHY THE NULLS DIFFER AT ALL, stated so it is not mistaken for a bug: each arm
is a different-width ridge (d = 72 / 5,632 / 5,632 / 24,576) solved on the same
~1,700 rows, and a wider basis fits more of the episode structure that survives
the within-episode permutation. That is exactly the bias a permutation test is
built to absorb -- **which is why the p-values were always the admissible
statistic and the raw AUCs never were.**

⚠️ AND IT MEANS THE PAIRED BOOTSTRAP CIs ON RAW DIFFERENCES ARE CONFOUNDED. A
`dinov3 - trunk` interval on raw AUC mixes the true representational difference
with the difference in estimator bias. They are reported here for continuity but
are NOT the deciding statistic; the permutation p and the null-adjusted delta
are.

⛔ AND THE `controls_ok` GATE IN THE PANEL IS ITSELF MIS-SPECIFIED. It required
every arm's null mean to sit within 0.05 of 0.5000 -- which re-imposes precisely
the assumption a permutation test exists to avoid. The gate fired `CONTROLS
FAIL` on the trunk's 0.5786. **The gate is wrong; the data is fine.** The
controls that actually matter all passed: the constant arm read 0.5000 exactly,
PC-PIXEL was won by pixels (0.9970) and PC-TRUNK by the trunk (0.9400).

Evidence class: MEASURED (ours). Tier T0, NON-PARITY pilot.
"""
import io
import json
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else r"C:/Users/Admin/tanitad-data/rl-pilot/wpi2_semantic_floor.json"
POW = sys.argv[2] if len(sys.argv) > 2 else r"C:/Users/Admin/tanitad-data/rl-pilot/wpi2_power_audit.json"
OUT = sys.argv[3] if len(sys.argv) > 3 else r"C:/Users/Admin/tanitad-data/rl-pilot/wpi2_null_adjusted.json"
d = json.load(io.open(SRC, encoding="utf-8"))
try:
    powr = json.load(io.open(POW, encoding="utf-8"))
except OSError:
    powr = {}

ARMS = ["pixels", "rand", "trunk", "dinov3"]
print("WP-I  n=%d  episodes=%d  grid=%s  perms=%d  encoder half-angle %.2f deg"
      % (d["n"], d["episodes"], d["grid"], d["n_perm"],
         d["encoder_half_angle_deg"]))
print("d: " + "  ".join("%s=%d" % (a, d["d"][a]) for a in ARMS))
print("\n%-28s%7s%6s%5s%s"
      % ("target", "base", "npos", "pwr", "".join("%24s" % a for a in ARMS)))
print("%-28s%7s%6s%5s%s" % ("", "", "", "",
                            "".join("%24s" % "AUC  null   adj    p_w"
                                    for _ in ARMS)))
out = {}
for t, r in d["results"].items():
    if not r["arms"][ARMS[0]]["null"]:
        continue                      # RAW targets carry no permutation null
    pw = powr.get(t, {}).get("within_null_power", "?")
    cells, row = "", {}
    for a in ARMS:
        auc = r["arms"][a]["auc"]
        nl = r["arms"][a]["null"]["within"]
        adj = auc - nl["mean"]
        row[a] = {"auc": auc, "null_within": nl["mean"], "adj": adj,
                  "p_within": nl["p"], "p_global": r["arms"][a]["null"]["global"]["p"]}
        cells += "%7.4f%7.4f%+7.4f%6.3f" % (auc, nl["mean"], adj, nl["p"])
    out[t] = {"base_rate": r["base_rate"], "n_pos": r["n_pos"],
              "within_null_power": pw, "arms": row}
    print("%-28s%7.3f%6d%5s%s" % (t[:27], r["base_rate"], r["n_pos"],
                                  pw[:4], cells))

print("\n  adj = AUC - that arm's OWN within-episode null mean. This is the only")
print("  cross-arm comparable column. p_w is the permutation p (within-episode).")

print("\nNULL-ADJUSTED ARM DIFFERENCES (the deciding read):")
print("%-30s%12s%12s%12s" % ("target", "dinov3-trunk", "trunk-rand", "trunk-pixels"))
for t, r in out.items():
    if t.startswith("PC-"):
        continue
    g = r["arms"]
    print("%-30s%+12.4f%+12.4f%+12.4f"
          % (t[:29], g["dinov3"]["adj"] - g["trunk"]["adj"],
             g["trunk"]["adj"] - g["rand"]["adj"],
             g["trunk"]["adj"] - g["pixels"]["adj"]))

print("\nPOSITIVE CONTROLS (each must be won by ONE named arm):")
for t, want in (("PC-PIXEL brightness", "pixels"), ("PC-TRUNK sel lateral>0", "trunk")):
    if t not in out:
        continue
    g = {a: out[t]["arms"][a]["auc"] for a in ARMS}
    win = max(g, key=g.get)
    print("  %-26s %s  (%s)  %s" % (t, "  ".join("%s %.4f" % (a, g[a]) for a in ARMS),
                                    "won by " + win, "PASS" if win == want else "FAIL"))
print("  CONSTANT-only control  %.4f  <- must be 0.5000 exactly (%s)"
      % (d["constant_control"], "PASS" if abs(d["constant_control"] - 0.5) < 1e-9
         else "FAIL"))

print("\nFOV-GATE EFFECT (in-FOV target minus raw target), per arm:")
for t, v in d["fov_gate_effect"].items():
    print("  %-24s%s" % (t.split()[0], " ".join("%s %+0.4f" % (a, v[a]) for a in ARMS)))

json.dump({"_tier": d["_tier"], "_source": SRC, "n": d["n"],
           "episodes": d["episodes"], "n_perm": d["n_perm"],
           "_note": "adj = AUC minus that arm's own within-episode permutation "
                    "null mean; the raw AUC column is NOT comparable across arms "
                    "because the four ridges have different null centres "
                    "(d = 72 / 5632 / 5632 / 24576 on ~1700 rows).",
           "targets": out},
          io.open(OUT, "w", encoding="utf-8"), indent=1)
print("\n-> wpi2_null_adjusted.json")
