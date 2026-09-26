"""Why are ~29 % of halfB's windows INELIGIBLE, and is the surviving subset representative?

⚠️ THIS FILE'S TITLE ONCE READ "only 27.2 % are eligible". That figure was MINE and it was a
DENOMINATOR ERROR: the 2,882 came from a 4,000-window POOL, not from the 10,600-window grid.
The census below is the correction — **71.22 % eligible** — and the wrong number is recorded here
rather than quietly deleted, because the instrument that produced it is the right place to warn
the next reader.

``s1_pass.Corpus.eligibility`` refuses for three reasons and two questions follow:

1. **WHY.** ``eligibility`` refuses for three distinct reasons (``s1_pass.py:98-110``) — ``t0``
   (the NOW frame is the episode's first), ``future`` (the route horizon runs past the provider's
   last pose) and ``agents`` (a tick in the agent window has no join record). They are different
   failures and a single percentage hides which dominates.
2. ⛔ **WHETHER IT BIASES THE CEILING.** My saturation sweep reported the effective-n ceiling as
   **62 clips**, the episode count of halfB. That is only true if EVERY episode contributes
   eligible windows. If eligibility wipes out whole episodes, the ceiling is lower than 62 and
   every interval in the panel is wider than I said — including the bar's.

## ANSWERED (2026-09-20), and the refusals decompose EXACTLY

* ``future`` = **2,418 = 62 episodes × 39 windows each**, an exact identity: with
  ``ROUTE_TICKS`` 60 and ``W`` 8, every 199-pose episode loses precisely its last 39 windows.
  ⇒ it is a **fixed positional TAIL**, uniform across episodes, so it cannot bias the read by
  anything except position-in-episode. *(The Master Mind reached the same conclusion empirically
  — ``future`` overlaps KEPT on near-forward count and range — so arithmetic and measurement
  agree from two directions.)*
* ``agents`` = **633**, of which **264 are episodes 21 and 37** (132 each: every window in their
  eligible span), and 369 scattered.
* ``t0`` = **0** — that branch never fires on this corpus.
* 7,549 + 2,418 + 633 = 10,600 ✔

⛔ **The two wholly-ineligible episodes are NOT short.** I predicted they would be — the
arithmetic says an episode is wholly refused on ``future`` iff ``n_prov < W + ROUTE_TICKS`` = 68
— and the prediction FAILED: both carry **199 poses**, the corpus norm. They are lost to
``agents``: the join has no usable coverage across their ``AGENT_TICKS`` runs. The ceiling is 60
because of **join coverage**, not clip length.

⛔ This is a census over the FULL grid, not a sample: no randperm, no ``[:n]``. Reasons are read
from ``eligibility``'s own return value, never re-derived here — re-implementing the predicate
would measure my copy of it rather than the one the probes use.

CPU only; reads no model and touches no GPU.

Usage: python probe_eligibility.py <out.json>
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
for p in (ROOT / "stack", ROOT / "taniteval", ROOT / "taniteval" / "tools",
          ROOT / "stack" / "scripts"):
    sys.path.insert(0, str(p))

A8 = r"C:\Users\Admin\tanitad-caches\a8-occupancy-5k-20260919\run"


# ⛔ A CHECKER MUST NOT DIE ON ITS OWN OUTPUT. MEASURED 2026-09-20: three separate
# readouts crashed with a cp1252 `UnicodeEncodeError` on this box mid-print -- one of
# them after reporting "lines lost = 1" but BEFORE naming the line, i.e. it had verified
# nothing while looking like it had. Relying on the caller to export PYTHONIOENCODING is
# a habit; this is a guard. `errors="replace"` means the print degrades instead of
# raising even if the stream cannot take utf-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 -- a stream that cannot be reconfigured is not fatal
    pass


def main(argv=None) -> int:
    out_path = (sys.argv[1:] if argv is None else argv)[0]
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import s1_pass as SP

    corp = SP.Corpus(A8 + r"\config.json",
                     r"D:\Projects\TanitAD-artifacts\v2ep-eval124clean-416x1024cyl-halfB",
                     r"C:\Users\Admin\tanitad-caches\a7-imagenet-knockout-20260919\inputs\s2_labels_v8_eval.jsonl.gz",
                     r"D:\Projects\TanitAD-artifacts\b1-agent-join-3d-20260917\b1eval_agents_3d.jsonl.xz",
                     None)
    n_total = len(corp.ds.index)
    reasons = Counter()
    per_ep_total = Counter()
    per_ep_ok = Counter()
    per_ep_reason = defaultdict(Counter)
    for wi in range(n_total):
        e_i = int(corp.ds.index[wi][0])
        per_ep_total[e_i] += 1
        r = corp.eligibility(wi)          # ⛔ the predicate ITSELF, not a re-derivation
        reasons[r or "OK"] += 1
        per_ep_reason[e_i][r or "OK"] += 1
        if r is None:
            per_ep_ok[e_i] += 1

    n_eps = len(per_ep_total)
    eps_with_any = sum(1 for e in per_ep_total if per_ep_ok[e] > 0)
    eps_with_none = [e for e in per_ep_total if per_ep_ok[e] == 0]
    frac_ok = [per_ep_ok[e] / per_ep_total[e] for e in sorted(per_ep_total)]
    frac_ok_sorted = sorted(frac_ok)
    rep = {
        "_what": "eligibility census over the FULL halfB window grid (no sampling)",
        "_evidence_class": "MEASURED (ours)",
        "n_windows_total": n_total,
        "n_eligible": reasons["OK"],
        "frac_eligible": round(reasons["OK"] / max(n_total, 1), 4),
        "refusal_reasons": {k: v for k, v in reasons.most_common()},
        "refusal_share": {k: round(v / max(n_total, 1), 4) for k, v in reasons.most_common()},
        "n_episodes": n_eps,
        "n_episodes_with_at_least_one_eligible_window": eps_with_any,
        "n_episodes_with_ZERO_eligible_windows": len(eps_with_none),
        "per_episode_eligible_fraction": {
            "min": round(frac_ok_sorted[0], 4),
            "p10": round(frac_ok_sorted[max(0, int(n_eps * 0.1) - 1)], 4),
            "median": round(frac_ok_sorted[n_eps // 2], 4),
            "max": round(frac_ok_sorted[-1], 4),
        },
    }
    # ⭐ WHY are some episodes WHOLLY ineligible? `eligibility` refuses "future" when
    # `t0 + ROUTE_TICKS > n_prov - 1` with `t0 = t + W - 1`; the smallest t is 0, so an episode
    # is wholly refused on that branch iff `n_prov < W + ROUTE_TICKS`. That is an ARITHMETIC
    # prediction, so it can be checked rather than believed.
    import ddv2_rl_refcv5 as _DD
    need = int(corp.W) + int(_DD.ROUTE_TICKS)
    per_ep = []
    for e in sorted(per_ep_total):
        n_prov = int(corp.poses(e).shape[0])
        per_ep.append({"ep": e, "n_prov_poses": n_prov, "n_windows": per_ep_total[e],
                       "n_eligible": per_ep_ok[e],
                       "predicted_wholly_refused": n_prov < need,
                       "reasons": dict(per_ep_reason[e])})
    zero = [r for r in per_ep if r["n_eligible"] == 0]
    rep["route_ticks"] = int(_DD.ROUTE_TICKS)
    rep["min_poses_for_any_eligible_window"] = need
    rep["zero_eligible_episodes"] = zero
    rep["prediction_holds"] = all(r["predicted_wholly_refused"] for r in zero) and         not any(r["predicted_wholly_refused"] for r in per_ep if r["n_eligible"] > 0)
    rep["shortest_episodes_poses"] = sorted(r["n_prov_poses"] for r in per_ep)[:6]
    rep["ceiling_verdict"] = (
        "the effective-n ceiling is %d episodes (every episode contributes at least one "
        "eligible window), so the 62-clip cap I reported STANDS" % eps_with_any
        if not eps_with_none else
        "⛔ the ceiling is %d, NOT %d: %d episode(s) contribute ZERO eligible windows, so every "
        "interval in the panel is wider than the 62-clip cap implied"
        % (eps_with_any, n_eps, len(eps_with_none)))
    Path(out_path).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    print("ZZELIG-DONE ZZ", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
