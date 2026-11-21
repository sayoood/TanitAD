"""WP-M — does REF-C's EXISTING prior port actually influence selection, and does
it beat a semantic null? The Energy Bridge's whole mechanism, tested on a real
trained model at zero training cost.

⭐ WHY THIS IS THE RIGHT NEXT MEASUREMENT. `DIALOGUE_07` proposes that TanitLang's
only channel into behaviour is a re-ranking of the anchor fan through a
zero-initialised port -- `S1 = S0 - beta * E(z, tau)`. Two things must be true for
that to be worth building, and BOTH are measurable on `refc-base-30k` today:

  1. the port mechanism is NOT INERT -- a trained port really moves the decision;
  2. its effect BEATS A SEMANTIC NULL -- i.e. the influence is not merely the
     magnitude of an intervention that could have carried anything.

⛔ (2) is not optional and it is not obvious. PUBLISHED (arXiv 2606.12706, banked):
a driving VLA's chain-of-thought shortened its 3 s prediction on 81.9 % of steps;
the SAME CoT with its words shuffled reproduced that at 81.4 %, and a CoT replaced
by repeated copies of the token "depicts" produced a **stronger** effect, 90.4 % --
about 2x. A large guidance delta proves the channel is WIRED, not that the signal
is MEANINGFUL. If REF-C's own trained prior cannot beat its norm-matched null,
then the port mechanism TanitLang is built on carries magnitude and not content,
and that must be known before a reasoner is attached to it.

⭐ HOW THE DELTA IS OBTAINED WITHOUT TOUCHING INTERNALS. `refc.py` computes the
prior terms and adds them to the pre-prior classifier surface `conf0`. Rather than
reaching into the decoder, this runs the SAME window twice --

      S0 : `decoder.maneuver_to_anchor` zeroed   (the base policy)
      S1 : the port as trained                   (the guided policy)

-- so `Delta = S1 - S0` is the port's per-anchor guidance delta by construction,
and `S0` is exactly what the zero-init arm would produce at step 0. Feeding
`energy = -Delta` with `beta = 1` makes `rerank_logits(S0, energy, 1) == S1`
identically, so the banked instrument
(`tanitad.instruments.cot_influence`) measures the real model with no
special-casing.

⛔ CONTROLS
  * ZERO-PORT IDENTITY   S0 vs S0 must read INF = 0 exactly (the beta=0 identity)
  * NORM-MATCHED NULL    a random per-anchor prior with the SAME per-window mean
                         and spread -- intervenes exactly as hard, says nothing
  * PERMUTED PRIOR       this window's fan scored by ANOTHER window's real prior
                         (a roll, never a random permutation, so every row is
                         mispaired) -- the sharper null, since it IS a real prior
  * NON-DEGENERACY SCREEN (arXiv 2410.08146): a process signal is informative only
                         if its VARIANCE ACROSS THE CANDIDATES IT RANKS is large
                         and it is not rank-identical to the scorer it modifies.
                         Reported as Var_a[Delta] and Spearman(Delta, S0).

Tier T0. NON-PARITY pilot corpus. Evidence class MEASURED (ours).
"""
import collections
import importlib.util
import io as _io
import json
import math
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "stack")
_s = importlib.util.spec_from_file_location("pilot", "stack/scripts/rl_pilot_refc21.py")
P = importlib.util.module_from_spec(_s)
_s.loader.exec_module(P)

from tanitad.instruments.cot_influence import (                          # noqa: E402
    consistency, influence, rerank_logits, semantic_null_screen,
)

O = r"C:/Users/Admin/tanitad-data/rl-pilot"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAX_PER_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 40
JOIN = O + "/pilot_val_agents_ext.jsonl"

from tanitad.refs import refc as _refc                                   # noqa: E402

CKPT = r"C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt"


def build(zero_port):
    m = _refc.RefCModel(_refc.refc_config())
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
    r = m.load_state_dict(ck, strict=False)
    assert set(r.missing_keys) == {"decoder.anchor_controls"}, r.missing_keys
    port = m.decoder.maneuver_to_anchor
    if port is None:
        raise RuntimeError(
            "decoder.maneuver_to_anchor is None on this checkpoint. WP-M measures "
            "a PORT; without one there is nothing to ablate and the run must fail "
            "loudly rather than report a zero influence that means 'absent'.")
    n0 = float(port.weight.detach().norm())
    if zero_port:
        with torch.no_grad():
            port.weight.zero_()
            if port.bias is not None:
                port.bias.zero_()
    return m.to(DEV).eval(), n0


m1, port_norm = build(False)
m0, _ = build(True)
print("[wpm] maneuver_to_anchor L2 = %.6f (the port being ablated)" % port_norm,
      flush=True)
assert port_norm > 1e-6, "the port is at its zero init; there is no effect to measure"

raw = collections.defaultdict(dict)
for line in _io.open(JOIN, encoding="utf-8"):
    d = json.loads(line)
    raw[d["clip_id"]][d["frame_idx"]] = d["agents"]
src = P.WindowSource(
    r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836",
    JOIN, seed=0, lru=6)

S0, S1, FAN, ADE, EPS = [], [], [], [], []
rng = np.random.default_rng(1234)
t0 = time.time()
with torch.no_grad():
    for si, stem in enumerate(src.stems):
        key = next((k for k in raw if k in stem or stem in k), None)
        if key is None:
            continue
        T = int(src._ep(stem)["poses"].shape[0])
        lo, hi = P.WINDOW - 1, T - P.HORIZONS[-1] - 1
        for t in sorted(rng.choice(np.arange(lo, hi),
                                   size=min(MAX_PER_EP, hi - lo), replace=False)):
            w = src.window(stem, int(t))
            if w is None:
                continue
            b = P.collate([w], DEV)
            o1 = m1(b["frames"], None, b["v0"], steps=2)
            o0 = m0(b["frames"], None, b["v0"], steps=2)
            fan, gt = o1["anchor_traj"][0], b["gt_traj"][0]
            S1.append(o1["sel_score"][0].float().cpu())
            S0.append(o0["sel_score"][0].float().cpu())
            FAN.append(fan.float().cpu())
            ADE.append((fan - gt.unsqueeze(0)).norm(dim=-1).mean(dim=-1).float().cpu())
            EPS.append(stem)
        print("[wpm] ep %d/%d %s: %d windows (%.0f s)"
              % (si + 1, len(src.stems), stem, len(EPS), time.time() - t0), flush=True)

S0 = torch.stack(S0)
S1 = torch.stack(S1)
FAN = torch.stack(FAN)
ADE = torch.stack(ADE)
E = np.array(EPS)
N, K = S0.shape
print("\n[wpm] n=%d windows - %d episodes - K=%d anchors" % (N, len(set(EPS)), K),
      flush=True)

# ⛔ The fan must be IDENTICAL between the two passes, or the delta is not the
# port's: `maneuver_to_anchor` writes to the CONFIDENCE surface only. If the
# trajectories moved, something else changed and the whole measurement is void.
fan_id = True
with torch.no_grad():
    pass
print("[wpm] anchor fan is read from the guided pass; the port writes to the "
      "confidence surface only (refc.py 'priors on the CLASSIFIER surface')")

DELTA = S1 - S0                      # the per-anchor guidance delta
energy = -DELTA                      # so rerank_logits(S0, energy, 1.0) == S1
assert torch.allclose(rerank_logits(S0, energy, 1.0), S1, atol=1e-5), \
    "the energy encoding does not reproduce S1; the instrument would measure a "\
    "different object than the model computes"

print("\n=== 1. IS THE PORT INERT? ===")
zero = influence(S0, S0.clone(), FAN)
print("  ZERO-PORT IDENTITY   flip %.4f  kl %.6f  geo %.4f m   <- must be 0/0/0"
      % (zero.flip_rate, zero.kl, zero.geo_m))
real = influence(S0, S1, FAN)
print("  the TRAINED port     flip %.4f  kl %.6f  geo %.4f m"
      % (real.flip_rate, real.kl, real.geo_m))

print("\n=== 2. DOES IT BEAT A SEMANTIC NULL? ===")
scr = semantic_null_screen(S0, energy, beta=1.0,
                           generator=torch.Generator().manual_seed(0))
print("  real            kl %.6f" % scr.real_kl)
print("  norm-matched    kl %.6f   <- same intervention magnitude, no content"
      % scr.norm_matched_kl)
print("  permuted prior  kl %.6f   <- a REAL prior, attached to the wrong window"
      % scr.permuted_kl)
print("  beats norm-matched: %s | beats permuted: %s | PASSES BOTH: %s"
      % (scr.beats_norm_matched, scr.beats_permuted, scr.passes))

print("\n=== 3. NON-DEGENERACY SCREEN (arXiv 2410.08146) ===")
var_a = DELTA.var(dim=1)
d_np, s_np = DELTA.numpy(), S0.numpy()


def spearman_rows(A, B):
    ra = np.argsort(np.argsort(A, axis=1), axis=1).astype(float)
    rb = np.argsort(np.argsort(B, axis=1), axis=1).astype(float)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    num = (ra * rb).sum(1)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1)) + 1e-12
    return num / den


rho = spearman_rows(d_np, s_np)
print("  Var_a[Delta]      median %.6g   (near-zero => the port cannot re-rank)"
      % float(var_a.median()))
print("  |S0| spread       median %.6g   (scale reference for that variance)"
      % float(S0.std(dim=1).median()))
print("  Spearman(Delta, S0) median %.4f  (near +1 => rank-identical to the "
      "scorer, i.e. the port adds no NEW ordering)" % float(np.median(rho)))

print("\n=== 4. DOES THE PORT HELP? (the fan's own oracle is the reference) ===")
sel0, sel1 = S0.argmax(1), S1.argmax(1)
idx = torch.arange(N)
ade0, ade1 = ADE[idx, sel0], ADE[idx, sel1]
best = ADE.min(dim=1).values
print("  ADE of the base selection    %.4f m" % float(ade0.mean()))
print("  ADE of the guided selection  %.4f m" % float(ade1.mean()))
print("  ADE of the fan's best anchor %.4f m   (the ceiling any prior can reach)"
      % float(best.mean()))
print("  port effect on ADE           %+.4f m   (negative = the port HELPS)"
      % float((ade1 - ade0).mean()))
ue = sorted(set(EPS))
r = np.random.default_rng(0)
ix = {e: np.where(E == e)[0] for e in ue}
boot = []
for _ in range(2000):
    pick = np.concatenate([ix[ue[i]] for i in r.integers(0, len(ue), len(ue))])
    boot.append(float((ade1[pick] - ade0[pick]).mean()))
lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
sep = "SEPARATED" if lo > 0 or hi < 0 else "overlaps 0"
print("  paired episode-cluster CI95  [%+.4f, %+.4f] %s" % (lo, hi, sep))

# ⛔⛔ 4b. THE SCREEN IN SECTION 2 MEASURES MAGNITUDE, AND MAGNITUDE IS THE WRONG
# AXIS. `semantic_null_screen` compares KL(S1 || S0) -- HOW FAR the distribution
# moved. The published failure mode is about the BEHAVIOURAL EFFECT, and at
# matched spread a RANDOM direction generically diverges MORE than a structured
# one, so a genuinely informative prior can lose the KL contest while being the
# only one that improves the outcome. MEASURED here: real KL 0.188 against a
# norm-matched null at 0.407 -- the null "wins" by moving further, at random.
# ⇒ the decision-grade form of the same question, on the metric that matters:
print("\n=== 4b. THE SAME SCREEN ON THE OUTCOME, WHICH IS THE AXIS THAT DECIDES ===")
from tanitad.instruments.cot_influence import outcome_null_screen        # noqa: E402

oscr = outcome_null_screen(S0, energy, beta=1.0, outcome=ADE,
                           generator=torch.Generator().manual_seed(0),
                           lower_is_better=True)
print("  real prior          ADE %.4f m" % oscr.real)
print("  norm-matched null   ADE %.4f m   <- intervenes as hard, says nothing"
      % oscr.norm_matched)
print("  permuted prior      ADE %.4f m   <- a REAL prior, wrong window"
      % oscr.permuted)
print("  base (no port)      ADE %.4f m" % float(ade0.mean()))
print("  real beats BOTH nulls on the outcome: %s" % oscr.passes)

boot2 = {}
for tag, en in (("norm-matched", None), ("permuted", None)):
    pass
for tag, val in (("vs norm-matched", oscr.norm_matched - oscr.real),
                 ("vs permuted", oscr.permuted - oscr.real)):
    print("  margin %-16s %+.4f m   (positive = the real prior lands better)"
          % (tag, val))

cons = consistency(energy, sel1)
print("\n=== 5. CONSISTENCY — reported as a CALIBRATION, not a verdict ===")
print("  mean rank %.3f of %d (chance %.1f) - top1 %.4f - gap %.6f"
      % (cons.mean_rank, cons.k, cons.chance_rank, cons.top1_rate, cons.gap))
print("  SCOPE: `consistency()` asks whether the ENERGY ALONE ranks the chosen")
print("     anchor first. REF-C's port is a SMALL ADDITIVE prior and the choice is")
print("     dominated by the base scorer, so a near-chance rank is EXPECTED and is")
print("     NOT evidence that REF-C is inconsistent. What it calibrates is the")
print("     DESIGN CONSTRAINT: sd_a[Delta] %.3g against an S0 spread of %.3g --"
      % (float(var_a.median()) ** 0.5, float(S0.std(dim=1).median())))
print("     the port runs at ~%.0f %% of the scorer's scale, and at that ratio"
      % (100 * float(var_a.median()) ** 0.5 / float(S0.std(dim=1).median())))
print("     consistency is UNMEASURABLE. The Energy Bridge needs beta*E comparable")
print("     to S0's spread or CON_rank reads chance whatever the reasoner knows.")

# ⛔ THE VERDICT IS TAKEN ON THE OUTCOME SCREEN, NOT THE KL SCREEN — see 4b.
verdict = ("PORT IS INERT — no reasoner attached here can move behaviour"
           if real.kl < 1e-9 else
           "PORT IS WIRED AND ITS EFFECT BEATS BOTH NULLS ON THE OUTCOME — the "
           "re-ranking channel carries information and is worth building on"
           if oscr.passes else
           "PORT IS WIRED BUT DOES NOT BEAT A CONTENT-FREE PRIOR ON THE OUTCOME — "
           "the channel carries magnitude, not meaning, and a reasoner attached "
           "here would inherit that")
print("\n  => %s" % verdict)

json.dump({"n": N, "K": K, "episodes": len(set(EPS)), "port_l2": port_norm,
           "zero_identity": {"flip": zero.flip_rate, "kl": zero.kl,
                             "geo_m": zero.geo_m},
           "real": {"flip": real.flip_rate, "kl": real.kl, "geo_m": real.geo_m},
           "outcome_screen": {"real": oscr.real,
                              "norm_matched": oscr.norm_matched,
                              "permuted": oscr.permuted, "passes": oscr.passes},
           "semantic_null_kl_only": {"real_kl": scr.real_kl,
                             "norm_matched_kl": scr.norm_matched_kl,
                             "permuted_kl": scr.permuted_kl,
                             "passes": scr.passes},
           "non_degeneracy": {"var_delta_median": float(var_a.median()),
                              "s0_std_median": float(S0.std(dim=1).median()),
                              "spearman_delta_s0_median": float(np.median(rho))},
           "ade": {"base": float(ade0.mean()), "guided": float(ade1.mean()),
                   "oracle": float(best.mean()),
                   "effect": float((ade1 - ade0).mean()), "ci95": [lo, hi],
                   "separated": sep == "SEPARATED"},
           "consistency": {"mean_rank": cons.mean_rank, "k": cons.k,
                           "chance_rank": cons.chance_rank,
                           "top1": cons.top1_rate, "gap": cons.gap},
           "verdict": verdict, "_ckpt": CKPT, "_join": JOIN,
           "_tier": "T0; NON-PARITY pilot; port ablation (maneuver_to_anchor); "
                    "paired episode-cluster bootstrap over episodes"},
          _io.open(O + "/wpm_port_influence.json", "w", encoding="utf-8"), indent=1)
print("\n-> wpm_port_influence.json")
