"""⛔ TRUNCATION ORDER: my census applied the pad to the GATE SUBSET; the trainer applies it to ALL.

`prebuild_p3_targets.census` computes ``keep_g = nearest_n(cx[g], cy[g], pad)`` — the nearest 32
*among gate-relevant targets*. The trainer does something else (`refc_v3_train._agent_item`):

    if n_raw > pad:
        order = np.argsort(np.hypot(ag[:, 0], ag[:, 1]))[:pad]   # nearest 32 of ALL targets
        ag = ag[order]

and the near-forward filter is applied LATER, at scoring. So a behind-the-ego target that is
physically close CONSUMES A SLOT that my version silently gave to a gate target.

⇒ My banked ``delivered_gate_after_pad`` (and the "halfB drops 15.7 % of gate boxes" figure I gave
the Master Mind) answer a question the trainer never asks. This probe measures BOTH orders on the
same windows so the size of the error is known rather than assumed.

⛔ It also tests the candidate I declined to assert for my 139-vs-129 cross-check gap: if the
CORRECT order reduces the bar draw's 24 windows from 139 gate targets to ~129, the pad explains
the gap; if it does not, something else does and the pad was a plausible-magnitude coincidence —
the exact trap I fell into over A7-IN-s0 this morning.

CPU only, no model pass.

Usage: python probe_pad_order.py <out.json>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
for p in (ROOT / "stack", ROOT / "taniteval", ROOT / "taniteval" / "tools",
          ROOT / "stack" / "scripts"):
    sys.path.insert(0, str(p))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

A8 = r"C:\Users\Admin\tanitad-caches\a8-occupancy-5k-20260919\run"
PAD = 32
X_FWD_M, Y_HALF_M = 60.0, 16.0


def gate_mask(cx, cy):
    return (cx >= 0.0) & (cx <= X_FWD_M) & (np.abs(cy) <= Y_HALF_M)


def main(argv=None) -> int:
    out_path = (sys.argv[1:] if argv is None else argv)[0]
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import s1_pass as SP
    from train_p8_occupancy import JoinFileReader

    corp = SP.Corpus(A8 + r"\config.json",
                     r"D:\Projects\TanitAD-artifacts\v2ep-eval124clean-416x1024cyl-halfB",
                     r"C:\Users\Admin\tanitad-caches\a7-imagenet-knockout-20260919\inputs\s2_labels_v8_eval.jsonl.gz",
                     r"D:\Projects\TanitAD-artifacts\b1-agent-join-3d-20260917\b1eval_agents_3d.jsonl.xz",
                     None)
    reader = JoinFileReader(r"D:\Projects\TanitAD-artifacts\b1-agent-join-3d-20260917\b1eval_agents_3d.jsonl.xz",
                            episode_ids=[int(e.episode_id) for e in corp.eps])
    W = int(corp.W)

    def counts(w):
        """-> (n_gate_pre, mine_gate_after_pad, TRAINER_gate_after_pad)."""
        e_i, t = corp.ds.index[w]
        ag = reader.lookup(int(corp.eps[int(e_i)].episode_id), int(t) + W - 1)
        if ag is None or ag.shape[0] == 0:
            return 0, 0, 0
        cx, cy = np.asarray(ag[:, 0], float), np.asarray(ag[:, 1], float)
        g = np.nonzero(gate_mask(cx, cy))[0]
        # MINE: nearest-PAD within the gate subset
        mine = min(int(g.size), PAD)
        # TRAINER: nearest-PAD over ALL targets, THEN filter to gate
        if cx.shape[0] > PAD:
            keep = np.argsort(np.hypot(cx, cy))[:PAD]
        else:
            keep = np.arange(cx.shape[0])
        trainer = int(gate_mask(cx[keep], cy[keep]).sum())
        return int(g.size), mine, trainer

    # ---- the BAR's own 24-window draw ------------------------------------------------
    wis = [w for w in SP.trainer_windows(corp.ds, 4000) if corp.eligibility(w) is None][:24]
    pre = mine = tr = 0
    for w in wis:
        a, b, c = counts(w)
        pre += a; mine += b; tr += c
    bar = {"n_windows": len(wis), "gate_pre_pad": pre, "mine_gate_after_pad": mine,
           "TRAINER_gate_after_pad": tr, "banked_matched_near_forward_n": 129}

    # ---- the full halfB grid ---------------------------------------------------------
    n = len(corp.ds.index)
    P = M = T = 0
    for w in range(n):
        a, b, c = counts(w)
        P += a; M += b; T += c
    full = {"n_windows": n, "gate_pre_pad": P, "mine_gate_after_pad": M,
            "TRAINER_gate_after_pad": T,
            "my_reported_drop_pct": round(100.0 * (P - M) / max(P, 1), 2),
            "TRUE_drop_pct": round(100.0 * (P - T) / max(P, 1), 2)}

    rep = {"_what": "pad-32 truncation applied to the GATE SUBSET (mine) vs ALL TARGETS (trainer)",
           "_evidence_class": "MEASURED (ours)", "pad": PAD,
           "bar_draw": bar, "full_halfB_grid": full}
    rep["pad_explains_the_139_vs_129_gap"] = (tr == 129)
    rep["verdict"] = (
        "the pad EXPLAINS the cross-check gap: correct order gives %d, banked matched is 129"
        % tr if tr == 129 else
        "⛔ the pad does NOT explain the gap: correct order gives %d, banked matched is 129 — "
        "the plausible-magnitude match was a coincidence and the cause is elsewhere" % tr)
    Path(out_path).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    print("ZZPADORD-DONE ZZ", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
