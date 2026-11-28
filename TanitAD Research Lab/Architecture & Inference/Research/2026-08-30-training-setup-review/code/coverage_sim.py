"""PI Q1 — does the 30k-step run actually see all 26 h of driving?

⛔ "EPOCHS" IS THE WRONG FRAME, AND THE DIVISION THAT PRODUCES IT IS INVALID HERE.
`train_v58f_unicycle_head.make_sampler` (the v6 sampler) draws **i.i.d. WITH
REPLACEMENT** — `rng.randrange` over episodes, then `rng.randrange` over that
episode's windows (`:368-378`). It is not a shuffled epoch loop. So
`steps x batch / n_windows` is an EXPECTED-DRAWS ratio, not a coverage fraction:
with replacement, N draws from M items never touch N distinct items.

Coverage is coupon-collector: E[distinct] / M = 1 - (1 - 1/M)^N ~= 1 - e^(-N/M).

⭐ AND THE QUESTION HAS THREE DIFFERENT ANSWERS depending on the unit, which is
why a single "epoch" number cannot answer the PI:

  EPISODE coverage  — "is every clip visited?"  This is what "all 26 h" means.
  WINDOW coverage   — "is every distinct window drawn?"
  FRAME coverage    — "is every frame of driving inside some drawn window?"
                      ⭐ THE HONEST MEASURE OF "SEEING THE DRIVING": windows are
                      stride-1 and each spans window+max_h frames, so consecutive
                      windows overlap by all but one frame. Missing a window does
                      NOT mean missing the driving in it.

Evidence class: MEASURED (simulation of the ACTUAL sampler) + the episode lengths
measured from the real caches. Nothing inherited.
"""
import random

import numpy as np

FRAMES_PER_EP = 199          # MEASURED: med 199 (min 197) over 25 sampled eps,
                             # both physicalai-train-14231cd29c74 and val caches
WINDOW = 6                   # MEASURED: V6Config().predictor.window
EPS_PER_BATCH = 4            # trainer default --eps-per-batch

CORPORA = {"parity (2,376 eps)": 2376, "B1 (4,713 eps)": 4713}
# max_h is STAGE-DERIVED (train_v6_staged.py:5070 max(need, plan.maneuver_h));
# report both plausible values rather than pretending one is canonical.
MAX_H = {"max_h=20 (v4-style)": 20, "max_h=60 (plan_steps horizon)": 60}
CONFIGS = {"Thor live (batch 8)": 8, "trainer default (batch 16)": 16}
STEPS = 30000


def simulate(n_eps, win_per_ep, batch, steps, seed=0, n_frames=FRAMES_PER_EP,
             max_h=20):
    """Run the REAL sampler's draw logic and count coverage three ways."""
    rng = random.Random(seed)
    ep_seen = np.zeros(n_eps, dtype=bool)
    # ⛔ TRACK EVERY EPISODE, NOT ONE PROBE. A coverage REQUIREMENT is decided by
    # the WORST episode, not the mean — "all 26 h" fails if a handful of clips
    # are barely drawn, and a single probe episode also has high variance in its
    # own draw count (which is why the first run's window figures disagreed with
    # the analytic by up to 12 pp).
    span = WINDOW + max_h                    # frames a single window touches
    win_seen = np.zeros((n_eps, win_per_ep), dtype=bool)

    for _ in range(steps):
        chosen = [rng.randrange(n_eps) for _ in range(EPS_PER_BATCH)]
        gi = 0
        while gi < batch:
            e = chosen[gi % len(chosen)]
            t = rng.randrange(win_per_ep)
            ep_seen[e] = True
            win_seen[e, t] = True
            gi += 1

    # frame coverage per episode, derived from that episode's drawn windows
    cum = np.cumsum(win_seen, axis=1)
    frame_cov = np.zeros(n_eps)
    for e in range(n_eps):
        hits = np.zeros(n_frames, dtype=bool)
        for t in np.flatnonzero(win_seen[e]):
            hits[t:min(t + span, n_frames)] = True
        frame_cov[e] = hits.mean()
    return (ep_seen.mean(), win_seen.mean(), frame_cov.mean(),
            frame_cov.min(), np.percentile(frame_cov, 1))


print(f"sampler: i.i.d. WITH REPLACEMENT (make_sampler:368-378) · "
      f"eps_per_batch={EPS_PER_BATCH} · steps={STEPS}")
print(f"episode length MEASURED {FRAMES_PER_EP} frames · window {WINDOW}\n")

for mh_label, mh in MAX_H.items():
    wpe = FRAMES_PER_EP - WINDOW - mh
    print(f"=== {mh_label}: windows/episode = {FRAMES_PER_EP} - {WINDOW} - {mh}"
          f" = {wpe} ===")
    for cname, n_eps in CORPORA.items():
        total_w = n_eps * wpe
        for bname, batch in CONFIGS.items():
            n_draw = STEPS * batch
            ratio = n_draw / total_w
            analytic = 1.0 - np.exp(-ratio)
            ep_c, win_c, f_mean, f_min, f_p1 = simulate(n_eps, wpe, batch,
                                                        STEPS, max_h=mh)
            print(f"  {cname:<18} {bname:<24} "
                  f"draws {n_draw:,} / {total_w:,} windows = {ratio:.2f}x")
            print(f"      EPISODE {ep_c*100:6.2f}%   WINDOW {win_c*100:6.2f}% "
                  f"(analytic {analytic*100:.2f}%)   FRAME mean {f_mean*100:6.2f}% "
                  f"p1 {f_p1*100:6.2f}% MIN {f_min*100:6.2f}%")
    print()
