"""E-DE-SIGN-4 (LR15-5): freeze the CoT sign-value oracle, then CHARACTERISE the gate it will run.

Two jobs, in this order, and the second exists because of the first:

(1) FREEZE. Write sign_value_oracle_v1.json from the LANDED E-DE-SIGN-3 artifact -- 42 Vienna
    clips (+ the 1 US clip, carried but excluded from the gate) -- and pin its sha256. The
    source json's own sha256 is recorded, so the freeze is traceable to a measurement and not
    to this script's reading of it.

(2) CHARACTERISE. LR15-5 committed the admission gate as:
        exact-value agreement >= 0.80  AND  Wilson 95% lower bound >= 0.65,  on the 42.
    A gate that is about to refuse or admit a reader before a PI decision is an INSTRUMENT, and
    an uncharacterised instrument is how a bar gets quoted as if it meant its nominal number.
    We compute, exactly (no simulation): which k pass, whether both clauses bind, the binomial
    operating characteristic P(pass | true accuracy p), and the two null floors a reader beats
    without reading anything (majority class, marginal-matching guesser).

0 GPU, no network, text/arithmetic only. argv: OUT_DIR.
"""
import hashlib
import json
import math
import os
import sys
from fractions import Fraction

SRC = ("D:/Projects/TanitAD/TanitAD Research Lab/Data Engineering/Research/"
       "2026-09-18-cot-sign-values/raw/cot_sign_values.json")
OUT = sys.argv[1] if len(sys.argv) > 1 else "raw"
Z = 1.959963984540054  # two-sided 95%

# ---------------------------------------------------------------- (1) freeze
src_bytes = open(SRC, "rb").read()
src_sha = hashlib.sha256(src_bytes).hexdigest()
d = json.loads(src_bytes)
train = d["sets"]["v7.2_train"]
clips = train["clips"]

# controls that must read known values, or the freeze is of the wrong thing
assert train["records"] == 4572, train["records"]
assert train["valued_clips"] == 43 == len(clips)
assert d["verdict"]["fires"] == "SPECIFIED"
assert train["by_system"] == {"Vienna_circle_kmh": 42, "US_MUTCD_mph": 1}
assert train["clips_multi_value"] == 0 and train["clips_all_legal"] == 43

vienna = [c for c in clips if c["system"] == "Vienna_circle_kmh"]
us = [c for c in clips if c["system"] != "Vienna_circle_kmh"]
assert len(vienna) == 42 and len(us) == 1
assert all(len(c["values"]) >= 1 and c["n_distinct_values"] == 1 for c in clips)
ids = [c["clip_id"] for c in clips]
assert len(set(ids)) == 43, "clip ids must be unique"


def row(c):
    return {"clip_id": c["clip_id"], "value": int(c["values"][0]),
            "unit": "km/h" if c["system"] == "Vienna_circle_kmh" else "mph",
            "country": c.get("country"), "daynight": c["daynight"],
            "flagged": bool(c["flagged"])}


gated = sorted((row(c) for c in vienna), key=lambda r: r["clip_id"])
carried = sorted((row(c) for c in us), key=lambda r: r["clip_id"])

oracle = {
    "name": "sign_value_oracle_v1",
    "what": "posted speed-limit VALUES read from the v7.2 train CoT text; the fixed validation "
            "set for E-DE-SIGN-1 (a candidate sign reader). Ego-free by construction.",
    "provenance": {
        "derived_from": "TanitAD Research Lab/Data Engineering/Research/2026-09-18-cot-sign-values/raw/cot_sign_values.json",
        "source_sha256": src_sha,
        "measuring_package": "TanitAD Research Lab/Data Engineering/Research/2026-09-18-cot-sign-values/RESULT.md",
        "frozen_by": "TanitAD Research Lab/Data Engineering/Research/2026-09-20-sign-oracle-freeze/ (E-DE-SIGN-4 / LR15-5)",
        "frozen_on": "2026-09-20",
        "corpus": "v7.2 train, 4,572 clips",
        "evidence_class": "MEASURED for the parse; the CoT itself is a VLM reading "
                          "(provenance vlm-cot), so agreement with this oracle is "
                          "reader-vs-VLM, NOT reader-vs-ground-truth",
    },
    "limits": [
        "RECALL IS NOT BOUNDED: 43 is a lower bound on the valued supply (a phrasing none of the "
        "four patterns covers is missed). Absence of a clip from this file is NOT evidence that "
        "the clip has no sign.",
        "n = 42 gated clips. Any accuracy read off this set carries a +-15 pp-class 95% interval.",
        "Class balance is skewed: 30 km/h is the plurality. See gate_characterisation.null_floors.",
        "Eval is 0/147 (E-DE-SIGN-3 F5): this set validates on TRAIN only, and a reader validated "
        "here has NO validation reference on eval.",
        "The boolean speed_limit flag is a leaky index in both directions (2 of 43 valued clips "
        "are NOT flagged); do not re-derive this set from the flag.",
    ],
    "gate": {
        "committed_by": "LAB_BACKLOG LR15-5 (proposed 2026-09-18 as E-DE-SIGN-4 / DE18-1)",
        "statement": "a candidate reader is admissible only if exact-value agreement >= 0.80 "
                     "AND the Wilson 95% lower bound >= 0.65, on the 42 Vienna clips",
        "scored_on": "gated_clips (42). carried_clips are reported separately and are NOT in the gate.",
        "comparison": "exact integer value match; unit must match.",
    },
    "gated_clips": gated,
    "carried_clips_not_in_gate": carried,
    "counts": {
        "gated": len(gated), "carried": len(carried),
        "day": sum(r["daynight"] == "day" for r in gated),
        "night": sum(r["daynight"] == "night" for r in gated),
        "value_hist": {str(v): sum(r["value"] == v for r in gated)
                       for v in sorted({r["value"] for r in gated})},
        "countries": {c: sum(r["country"] == c for r in gated)
                      for c in sorted({r["country"] for r in gated if r["country"]})},
    },
}

# ------------------------------------------------- (2) characterise the gate
N = len(gated)


def wilson_lo(k, n, z=Z):
    if n == 0:
        return float("nan")
    p = k / n
    z2n = z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / (1 + z2n)


def passes(k, n=N):
    return (k / n >= 0.80) and (wilson_lo(k, n) >= 0.65)


per_k = [{"k": k, "point": round(k / N, 4), "wilson_lo95": round(wilson_lo(k, N), 4),
          "clause_point": k / N >= 0.80, "clause_wilson": wilson_lo(k, N) >= 0.65,
          "passes": passes(k)} for k in range(N + 1)]
k_min = min((r["k"] for r in per_k if r["passes"]), default=None)
# does the Wilson clause ever bind? i.e. is there a k passing clause 1 but failing clause 2?
wilson_binds = [r["k"] for r in per_k if r["clause_point"] and not r["clause_wilson"]]
# at which n WOULD it bind? smallest k with k/n>=0.8 is ceil(0.8n); check its Wilson LB
n_binds = [n for n in range(5, 401)
           if wilson_lo(math.ceil(0.80 * n), n) < 0.65]


def binom_tail_ge(k0, n, p):
    """exact P(K >= k0), K~Bin(n,p), in Fractions where p is rational-friendly."""
    if k0 <= 0:
        return 1.0
    if k0 > n:
        return 0.0
    tot = 0.0
    for k in range(k0, n + 1):
        tot += math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    return tot


oc = [{"true_accuracy": p, "P_pass": round(binom_tail_ge(k_min, N, p), 4)}
      for p in (0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99)]
# smallest true accuracy whose pass-probability reaches 0.80 (bisection on a monotone function)
lo, hi = 0.5, 1.0
for _ in range(60):
    mid = (lo + hi) / 2
    if binom_tail_ge(k_min, N, mid) < 0.80:
        lo = mid
    else:
        hi = mid
p_for_80pct_power = (lo + hi) / 2

hist = oracle["counts"]["value_hist"]
tot = sum(hist.values())
majority = max(hist.values()) / tot
marginal = sum((v / tot) ** 2 for v in hist.values())
# a reader that answers the plurality value everywhere: k = majority count
null_floors = {
    "majority_class": {"rule": "answer the plurality value on every clip",
                       "value": max(hist, key=hist.get),
                       "expected_k": max(hist.values()),
                       "accuracy": round(majority, 4),
                       "passes_gate": passes(max(hist.values()))},
    "marginal_guesser": {"rule": "sample from the oracle's own value marginal",
                         "expected_accuracy": round(marginal, 4),
                         "expected_k": round(marginal * N, 2),
                         "passes_gate": passes(round(marginal * N))},
    "n_distinct_values": len(hist),
}

night = [r for r in gated if r["daynight"] == "night"]
day = [r for r in gated if r["daynight"] == "day"]
# smallest day-night accuracy gap detectable at 95% with these subgroup sizes, using the
# normal approximation to the difference of two proportions at the gate's own operating point
se_worst = math.sqrt(0.80 * 0.20 / len(day) + 0.80 * 0.20 / len(night))
oracle["gate_characterisation"] = {
    "n_gated": N,
    "k_min_to_pass": k_min,
    "gate_reduces_to": f"k >= {k_min} of {N} correct",
    "wilson_clause_binds_at_n42": bool(wilson_binds),
    "wilson_clause_note": ("the Wilson clause is VACUOUS at n=42: every k that clears the 0.80 "
                           "point estimate also clears the 0.65 lower bound, so the gate is a "
                           "single test" if not wilson_binds else "the Wilson clause binds"),
    "wilson_clause_would_bind_for_n_in": (f"{min(n_binds)}..{max(n_binds)} "
                                          f"({len(n_binds)} values in 5..400)") if n_binds else "no n in 5..400",
    "operating_characteristic": oc,
    "P_pass_at_nominal_bar_0.80": round(binom_tail_ge(k_min, N, 0.80), 4),
    "true_accuracy_needed_for_80pct_pass": round(p_for_80pct_power, 4),
    "null_floors": null_floors,
    "subgroups": {"day": len(day), "night": len(night),
                  "min_detectable_day_night_gap_95pct": round(Z * se_worst, 4),
                  "note": "normal-approximation SE of a difference of proportions at p=0.80 in "
                          "both arms; a night-vs-day gap smaller than this is invisible here"},
    "per_k_table": per_k,
}

os.makedirs(OUT, exist_ok=True)
body = json.dumps(oracle, indent=1, ensure_ascii=False)
path = os.path.join(OUT, "sign_value_oracle_v1.json")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    f.write(body)
sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
with open(os.path.join(OUT, "sign_value_oracle_v1.sha256"), "w", encoding="utf-8", newline="\n") as f:
    f.write(sha + "  sign_value_oracle_v1.json\n")

print("source sha256      ", src_sha)
print("oracle sha256      ", sha)
print("gated / carried    ", len(gated), "/", len(carried))
print("day / night        ", oracle["counts"]["day"], "/", oracle["counts"]["night"])
print("value hist         ", hist)
print("countries          ", oracle["counts"]["countries"])
print()
print("GATE reduces to    ", oracle["gate_characterisation"]["gate_reduces_to"])
print("Wilson clause binds?", oracle["gate_characterisation"]["wilson_clause_binds_at_n42"],
      "|", oracle["gate_characterisation"]["wilson_clause_would_bind_for_n_in"])
print("P(pass | p=0.80)   ", oracle["gate_characterisation"]["P_pass_at_nominal_bar_0.80"])
print("p for 80% power    ", oracle["gate_characterisation"]["true_accuracy_needed_for_80pct_pass"])
print("OC                 ", [(r["true_accuracy"], r["P_pass"]) for r in oc])
print("null floors        ", json.dumps(null_floors))
print("subgroups          ", json.dumps(oracle["gate_characterisation"]["subgroups"]))
print("k table (k>=30)    ", [(r["k"], r["point"], r["wilson_lo95"], r["passes"])
                              for r in per_k if r["k"] >= 30])
