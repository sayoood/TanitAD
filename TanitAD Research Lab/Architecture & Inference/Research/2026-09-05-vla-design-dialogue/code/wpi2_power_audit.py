"""WP-I power audit — is the WITHIN-EPISODE null answering a question, or nothing?

⛔ THE TRAP THIS EXISTS TO CATCH, BEFORE A SINGLE `p_within` IS READ.
The within-episode permutation null holds each episode's label prevalence fixed
and permutes only inside it. That is the conservative and correct null -- but it
has power ONLY where the target actually VARIES within an episode. If a clip
either has a pedestrian ahead for most of its length or has none at all, then
inside almost every episode the label is constant, the permutation is close to
the identity, and `p_within` is large **for every representation, however good**.

⇒ A large `p_within` is then a statement about the TARGET's within-episode
variance, not about the arm. Reading it as "no signal" would be the campaign's
recurring error in its tenth costume: an instrument's blind spot read as a
finding about the thing measured.

⭐ THE DISCRIMINATOR, and it is cheap: count, per target, how many episodes are
NON-CONSTANT and how many within-episode positives exist in them. A target whose
effective within-episode sample is tiny CANNOT be judged by `p_within`, and this
script says so per target rather than leaving the reader to assume.

Reproduces `wpi2_semantic_floor.py`'s window selection EXACTLY -- same seed, same
`MAX_PER_EP`, same `src.window(...) is None` and `raw[key].get(t) is None` skips
-- so the counts are about the rows that were actually scored, not about the
corpus. ⛔ No model is loaded and no GPU is touched: `PC-TRUNK` is the one target
that needs a forward pass and is therefore reported as UNAVAILABLE here rather
than silently omitted.

Tier: T0. Evidence class: MEASURED (ours).
"""
import collections
import importlib.util
import io
import json
import math
import sys

import numpy as np

sys.path.insert(0, "stack")
_s = importlib.util.spec_from_file_location("pilot", "stack/scripts/rl_pilot_refc21.py")
P = importlib.util.module_from_spec(_s)
_s.loader.exec_module(P)

O = r"C:/Users/Admin/tanitad-data/rl-pilot"
NEAR_M, FAR_M = 20.0, 60.0
HALF = math.atan(128.0 / 266.0)
MAX_PER_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 120
VRU = {"person", "rider", "stroller", "animal"}

raw = collections.defaultdict(dict)
for line in io.open(O + "/pilot_val_agents.jsonl", encoding="utf-8"):
    d = json.loads(line)
    raw[d["clip_id"]][d["frame_idx"]] = d["agents"]

src = P.WindowSource(
    r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836",
    O + "/pilot_val_agents.jsonl", seed=0, lru=6)

TGT = collections.defaultdict(list)
EPS = []
rng = np.random.default_rng(1234)          # MUST match wpi2_semantic_floor.py
for stem in src.stems:
    key = next((k for k in raw if k in stem or stem in k), None)
    if key is None:
        continue
    T = int(src._ep(stem)["poses"].shape[0])
    lo, hi = P.WINDOW - 1, T - P.HORIZONS[-1] - 1
    idx = sorted(rng.choice(np.arange(lo, hi),
                            size=min(MAX_PER_EP, hi - lo), replace=False))
    for t in idx:
        t = int(t)
        ags = raw[key].get(t)
        if ags is None:
            continue
        if src.window(stem, t) is None:
            continue

        def seen(a):
            return a["cx"] > 0 and abs(math.atan2(a["cy"], a["cx"])) <= HALF

        n_seen = sum(1 for a in ags if seen(a))
        lead = min([a["cx"] for a in ags
                    if a["cx"] > 0 and abs(a["cy"]) <= 2.0] or [FAR_M])
        TGT["vru<20m RAW (ungated)"].append(
            float(any(a.get("cls") in VRU and 0 < a["cx"] < NEAR_M
                      and abs(a["cy"]) <= 4.0 for a in ags)))
        TGT["vru<60m RAW (ungated)"].append(
            float(any(a.get("cls") in VRU and 0 < a["cx"] < FAR_M
                      and abs(a["cy"]) <= 4.0 for a in ags)))
        TGT["vru<20m IN-FOV"].append(
            float(any(a.get("cls") in VRU and 0 < a["cx"] < NEAR_M
                      and abs(a["cy"]) <= 4.0 and seen(a) for a in ags)))
        TGT["vru<60m IN-FOV"].append(
            float(any(a.get("cls") in VRU and 0 < a["cx"] < FAR_M
                      and abs(a["cy"]) <= 4.0 and seen(a) for a in ags)))
        TGT["crowd: >=3 agents in FOV"].append(float(n_seen >= 3))
        TGT["lead < 25 m"].append(float(lead < 25.0))
        EPS.append(stem)

E = np.array(EPS)
UE = sorted(set(EPS))
N = len(E)
print("windows n=%d - episodes %d  (must match the panel's n)" % (N, len(UE)))
print("\n%-28s%7s%9s%11s%13s%10s"
      % ("target", "base", "n_pos", "eps var/N", "within_pos", "verdict"))

out = {}
for name, v in TGT.items():
    y = np.asarray(v)
    n_var = 0
    within_pos = 0            # positives living inside a NON-CONSTANT episode
    for ep in UE:
        ye = y[E == ep]
        if 0 < ye.sum() < len(ye):
            n_var += 1
            within_pos += int(ye.sum())
    # A within-episode permutation can only reshuffle rows inside episodes that
    # are NOT constant. `within_pos` is therefore the effective positive count
    # the within-null actually has to work with.
    verdict = ("USABLE" if n_var >= 6 and within_pos >= 40 else
               "WEAK" if n_var >= 3 and within_pos >= 15 else
               "NO POWER")
    out[name] = {"base_rate": float(y.mean()), "n_pos": int(y.sum()),
                 "episodes_nonconstant": n_var, "episodes_total": len(UE),
                 "within_episode_positives": within_pos,
                 "within_null_power": verdict}
    print("%-28s%7.3f%9d%7d/%-3d%13d%10s"
          % (name, y.mean(), int(y.sum()), n_var, len(UE), within_pos, verdict))

print("\n  READING RULE, and it binds every p_within in the panel:")
print("    NO POWER  -> p_within says NOTHING about the arm; do not read it.")
print("    WEAK      -> p_within is a weak bound only; quote it with these counts.")
print("    USABLE    -> p_within is admissible as evidence about the arm.")
print("\n  PC-TRUNK sel lateral>0: UNAVAILABLE here (needs a model forward pass);")
print("  reported as absent rather than silently omitted.")
out["_note"] = ("within-null power audit; PC-TRUNK omitted (needs a forward pass). "
                "Window selection reproduces wpi2_semantic_floor.py exactly "
                "(seed 1234, same skips).")
out["_tier"] = "T0; NON-PARITY pilot; MEASURED (ours)"
json.dump(out, io.open(O + "/wpi2_power_audit.json", "w", encoding="utf-8"), indent=1)
print("\n-> wpi2_power_audit.json")
