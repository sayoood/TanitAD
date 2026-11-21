"""Do the ``agents`` refusals (5.97 %) differ from KEPT — and they are TWO questions, not one.

633 windows are refused for ``agents``. **264 of them are episodes 21 and 37 in their entirety.**
So "do the agents refusals differ from KEPT" conflates:

* **Q1 — the two DEAD CLIPS.** They are absent from every held-out read. The question is not
  whether their windows differ but whether LOSING THOSE CLIPS costs representativeness or only
  ``n``. ⛔ A between-episode comparison cannot answer the scattered case and a within-episode
  comparison cannot answer this one; they need different tests.
* **Q2 — the 369 SCATTERED refusals**, which sit inside otherwise-live episodes. These admit a
  **WITHIN-EPISODE** comparison against the KEPT windows of the same clips, which controls for
  clip identity — the confound that would dominate a pooled test.

⭐ Neither the positional-tail argument nor the ``future`` comparison covers these: ``agents``
is not a positional tail, so this is genuinely uncovered ground.

⛔ NO MODEL PASS. Target density comes from the BANKED census (`p3_targets_halfB.npz`:
``labelled``, ``n_raw``, ``n_gate`` for all 10,600 windows), and the refusal reason from
``eligibility`` itself. Nothing here re-derives either.

Usage: python probe_agents_refusal.py <out.json>
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
for p in (ROOT / "stack", ROOT / "taniteval", ROOT / "taniteval" / "tools",
          ROOT / "stack" / "scripts"):
    sys.path.insert(0, str(p))

A8 = r"C:\Users\Admin\tanitad-caches\a8-occupancy-5k-20260919\run"
BANK = r"C:\Users\Admin\tanitad-caches\p3-prebuild-20260920\p3_targets_v2ep-eval124clean-416x1024cyl-halfB.npz"


def _stats(name, lab, raw, gate):
    n = int(lab.size)
    return {"group": name, "n_windows": n,
            "frac_labelled": round(float(lab.mean()), 4) if n else None,
            "mean_n_raw": round(float(raw.mean()), 3) if n else None,
            "mean_n_gate": round(float(gate.mean()), 3) if n else None}


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

    z = np.load(BANK)
    lab_all, raw_all, gate_all = z["labelled"].astype(bool), z["n_raw"], z["n_gate"]
    corp = SP.Corpus(A8 + r"\config.json",
                     r"D:\Projects\TanitAD-artifacts\v2ep-eval124clean-416x1024cyl-halfB",
                     r"C:\Users\Admin\tanitad-caches\a7-imagenet-knockout-20260919\inputs\s2_labels_v8_eval.jsonl.gz",
                     r"D:\Projects\TanitAD-artifacts\b1-agent-join-3d-20260917\b1eval_agents_3d.jsonl.xz",
                     None)
    n = len(corp.ds.index)
    if n != lab_all.size:
        raise SystemExit("⛔ bank/grid mismatch: %d vs %d — refusing" % (lab_all.size, n))

    eps = np.array([int(corp.ds.index[w][0]) for w in range(n)])
    reason = np.array([(corp.eligibility(w) or "OK") for w in range(n)], dtype=object)
    per_ep_ok = Counter(eps[reason == "OK"].tolist())
    dead = sorted({int(e) for e in set(eps.tolist()) if per_ep_ok[int(e)] == 0})

    is_ag = reason == "agents"
    is_ok = reason == "OK"
    in_dead = np.isin(eps, dead)

    rep = {"_what": "do the `agents` refusals differ from KEPT — split into two questions",
           "_evidence_class": "MEASURED (ours)", "_no_model_pass": True,
           "dead_episodes": dead,
           "n_agents_total": int(is_ag.sum()),
           "n_agents_in_dead_episodes": int((is_ag & in_dead).sum()),
           "n_agents_scattered": int((is_ag & ~in_dead).sum())}

    # ---- Q1: are the DEAD CLIPS unusual, or ordinary clips we simply lose? ----------
    rep["Q1_dead_clips"] = {
        "dead": _stats("dead episodes (all windows)", lab_all[in_dead], raw_all[in_dead],
                       gate_all[in_dead]),
        "live": _stats("live episodes (all windows)", lab_all[~in_dead], raw_all[~in_dead],
                       gate_all[~in_dead]),
    }
    # ---- Q2: WITHIN-EPISODE, scattered agents-refusals vs KEPT of the SAME clips ----
    # ⛔ paired by episode: a pooled test would be confounded by which clips contribute.
    pairs = []
    for e in sorted(set(eps[is_ag & ~in_dead].tolist())):
        m_ag = is_ag & ~in_dead & (eps == e)
        m_ok = is_ok & (eps == e)
        if m_ag.sum() == 0 or m_ok.sum() == 0:
            continue
        pairs.append({"ep": int(e), "n_ag": int(m_ag.sum()), "n_ok": int(m_ok.sum()),
                      "d_frac_labelled": round(float(lab_all[m_ag].mean() - lab_all[m_ok].mean()), 4),
                      "d_mean_n_gate": round(float(gate_all[m_ag].mean() - gate_all[m_ok].mean()), 3)})
    rep["Q2_scattered_within_episode"] = {
        "n_episodes_paired": len(pairs),
        "per_episode": pairs,
        "median_d_frac_labelled": (round(float(np.median([p["d_frac_labelled"] for p in pairs])), 4)
                                   if pairs else None),
        "median_d_mean_n_gate": (round(float(np.median([p["d_mean_n_gate"] for p in pairs])), 3)
                                 if pairs else None),
        "n_episodes_where_refused_are_DENSER": sum(1 for p in pairs if p["d_mean_n_gate"] > 0),
    }
    # ---- Q3: a signal INDEPENDENT OF THE JOIN, because Q2 is circular ----------------
    # ⛔ Q2 compares target geometry between KEPT and `agents`-refused windows. But `agents`
    # refuses exactly when join records are MISSING, and both `labelled` and the target counts
    # are read FROM that join — so "the refused windows have no targets" is DEFINITIONAL, not
    # evidence. A non-circular test needs a channel the join does not touch: the ego POSES.
    W = int(corp.W)
    def ego_speed(w):
        e_i, t = corp.ds.index[w]
        t0 = t + W - 1
        pa = corp.poses(int(e_i))
        if t0 < 1 or t0 >= int(pa.shape[0]):
            return None
        return float(((pa[t0][:2, 3] - pa[t0 - 1][:2, 3]) ** 2).sum() ** 0.5)             if pa.ndim == 3 else float(((pa[t0][:2] - pa[t0 - 1][:2]) ** 2).sum() ** 0.5)
    sp_ag, sp_ok = [], []
    for w in range(n):
        if in_dead[w]:
            continue                      # Q1 handles the dead clips; they are not this question
        v = ego_speed(w)
        if v is None:
            continue
        if is_ag[w]:
            sp_ag.append(v)
        elif is_ok[w]:
            sp_ok.append(v)
    # ⛔ AN EPISODE-CLUSTERED INTERVAL, or this is a point difference and not a finding.
    # The windows are clustered by clip; a naive interval on 369 vs 7,549 would overstate.
    from taniteval.ci import episode_cluster_bootstrap as _ecb
    ag_e = [int(eps[w]) for w in range(n) if is_ag[w] and not in_dead[w]
            and ego_speed(w) is not None]
    ok_e = [int(eps[w]) for w in range(n) if is_ok[w] and ego_speed(w) is not None]
    try:
        ci_ag = _ecb(np.asarray(sp_ag, dtype=float), ag_e, n_boot=2000)
        ci_ok = _ecb(np.asarray(sp_ok, dtype=float), ok_e, n_boot=2000)
    except Exception as exc:                                  # noqa: BLE001
        ci_ag = ci_ok = {"error": "%s: %s" % (type(exc).__name__, exc)}
    import statistics as st
    rep["Q3_ego_speed_independent_of_join"] = {
        "ci_agents_episode_clustered": ci_ag,
        "ci_kept_episode_clustered": ci_ok,
        "n_episodes_agents": len(set(ag_e)), "n_episodes_kept": len(set(ok_e)),
        "_why": "Q2 is circular: `agents` and `labelled` both read the join. Poses do not.",
        "n_agents_scattered": len(sp_ag), "n_kept": len(sp_ok),
        "median_ego_step_m_agents": (round(st.median(sp_ag), 4) if sp_ag else None),
        "median_ego_step_m_kept": (round(st.median(sp_ok), 4) if sp_ok else None),
        "mean_ego_step_m_agents": (round(st.mean(sp_ag), 4) if sp_ag else None),
        "mean_ego_step_m_kept": (round(st.mean(sp_ok), 4) if sp_ok else None),
    }
    # ---- Q4: DOSE-RESPONSE. Does the join's miss rate TRACK speed, or is 2.08x a two-group
    # artefact? A monotone curve over deciles is far stronger evidence about the JOIN BUILDER
    # than any contrast of two groups, and it is what anyone reusing this join needs to know.
    # ⛔ n_episodes per bin is reported because speed CLUSTERS WITHIN CLIPS — the asymmetry that
    # made the quartile test's fast stratum the wide one. A bin count alone would hide it.
    live = [w for w in range(n) if not in_dead[w] and ego_speed(w) is not None]
    sp = np.array([ego_speed(w) for w in live], dtype=float)
    order = np.argsort(sp)
    bins, nb = [], 10
    for b in range(nb):
        idx = order[b * len(order) // nb:(b + 1) * len(order) // nb]
        ws = [live[i] for i in idx]
        if not ws:
            continue
        ep_here = {int(eps[w]) for w in ws}
        bins.append({
            "decile": b + 1,
            "speed_lo": round(float(sp[idx].min()), 4),
            "speed_hi": round(float(sp[idx].max()), 4),
            "n_windows": len(ws),
            "n_episodes": len(ep_here),
            "frac_agents_refused": round(float(np.mean([bool(is_ag[w]) for w in ws])), 4),
            "frac_unlabelled": round(float(np.mean([not lab_all[w] for w in ws])), 4),
        })
    fr = [b["frac_agents_refused"] for b in bins]
    rep["Q4_dose_response_join_miss_vs_speed"] = {
        "_why": "a monotone rise would implicate the JOIN BUILDER at speed; a flat curve would "
                "make the 2.08x a two-group artefact",
        "bins": bins,
        "frac_refused_lowest_decile": fr[0] if fr else None,
        "frac_refused_highest_decile": fr[-1] if fr else None,
        "monotone_nondecreasing": all(a <= b + 1e-9 for a, b in zip(fr, fr[1:])) if fr else None,
        "n_deciles_with_zero_refusals": sum(1 for v in fr if v == 0.0),
    }
    Path(out_path).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "Q2_scattered_within_episode"},
                     indent=1))
    q2 = rep["Q2_scattered_within_episode"]
    print("Q2 within-episode: %d episodes paired, median d_frac_labelled %s, "
          "median d_mean_n_gate %s, refused denser in %d/%d"
          % (q2["n_episodes_paired"], q2["median_d_frac_labelled"], q2["median_d_mean_n_gate"],
             q2["n_episodes_where_refused_are_DENSER"], q2["n_episodes_paired"]))
    print("ZZAGREF-DONE ZZ", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
