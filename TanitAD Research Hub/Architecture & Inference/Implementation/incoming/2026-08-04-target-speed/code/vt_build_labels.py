#!/usr/bin/env python3
"""D-VT1 step 1 — mint the ORACLE and the GUARDED target-speed labels on val40.

Reads a poses-only view of the canonical val40 cache (40 episodes, sha256-pinned
against `…/2026-07-26-s3-decision-grade/artifacts/manifest_EVALPOD_val40.json`)
and reproduces `rollout.collect`'s 881-window grid through
`taniteval.lead_source.window_last_indices` — the SAME function the banked lead
block used, so every row here is attachable to an already-scored dump with no
re-inference.

⛔ PARITY: read-only. No episode is re-selected; the corpus is
`physicalai-train-e438721ae894` / val `physicalai-val-0c5f7dac3b11`, untouched.

Emits, per window:
  vt_oracle   `vtarget_v2`      — reads v[l+1 : l+200]  ⛔ CONTAINS the scored horizon
  vt_guarded  `vtarget_guarded` — reads v[l+21 : l+200] ✅ DISJOINT from it
  the valid masks, the realised lookaheads, the 23-band tokens for both,
  and the scored quantities the guard is audited against
  (dv_2s, along-track 0-2 s displacement, v0 and the causal past-speed block).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "taniteval"))

from taniteval.lead_source import window_last_indices          # noqa: E402
from tanitad.lake.vocab import VTARGET_TOKENS, vtarget_band     # noqa: E402
from tanitad.lake.vtarget import (VT_GUARD_STEPS, read_window,  # noqa: E402
                                  vtarget_guarded, vtarget_v2)

MANIFEST = (REPO / "TanitAD Research Hub" / "Architecture & Inference" /
            "Implementation" / "incoming" / "2026-07-26-s3-decision-grade" /
            "artifacts" / "manifest_EVALPOD_val40.json")
#: causal past-speed window, in steps BEFORE the window origin. 0.7 s — the same
#: span `sc_train.py` gives the situation classifier, so the "what does the past
#: alone buy" baseline is not tuned to flatter the guard.
PAST_STEPS = 7


def build(view_npz: Path) -> dict:
    z = np.load(view_npz, allow_pickle=True)
    meta = json.loads(bytes(z["_meta_json"]).decode())
    man = {e["file"]: e for e in json.load(open(MANIFEST))["episodes"]}

    rows, eps = [], []
    for m in sorted(meta, key=lambda d: d["file"]):
        f = m["file"]
        poses = z[f"{Path(f).stem}__poses"].astype(np.float64)   # [T, 4] x,y,yaw,v
        sha = hashlib.sha256(
            z[f"{Path(f).stem}__poses"].astype(np.float32).tobytes()).hexdigest()
        ref = man.get(f, {})
        ok = sha == ref.get("poses_sha256")
        x, y, v = poses[:, 0], poses[:, 1], poses[:, 3]
        t_len = poses.shape[0]
        last = window_last_indices(t_len)

        vt_o, ok_o, look_o, _vs = vtarget_v2(v, last, min_lookahead=50)
        vt_g, ok_g, look_g, _ = vtarget_guarded(
            v, last, guard_steps=VT_GUARD_STEPS, min_lookahead=50)

        for i, ell in enumerate(last):
            lo, hi = read_window(int(ell), t_len, VT_GUARD_STEPS)
            scored = set(range(int(ell), int(ell) + VT_GUARD_STEPS + 1))
            # ⛔ the admissibility check, recomputed per window rather than argued
            assert not (scored & set(range(lo, hi))), f"{f} l={ell}: guard leak"
            past = v[max(0, ell - PAST_STEPS):ell + 1]
            if past.size < PAST_STEPS + 1:                       # warm-up pad
                past = np.concatenate([np.full(PAST_STEPS + 1 - past.size,
                                               past[0]), past])
            j = min(int(ell) + VT_GUARD_STEPS, t_len - 1)
            rows.append(dict(
                file=f, eid=int(m["episode_id"]), l=int(ell), T=int(t_len),
                v0=float(v[ell]), v_2s=float(v[j]),
                dv_2s=float(v[j] - v[ell]),
                along_2s=float(np.hypot(x[j] - x[ell], y[j] - y[ell])),
                vt_oracle=float(vt_o[i]), vt_oracle_valid=bool(ok_o[i]),
                vt_oracle_look=int(look_o[i]),
                vt_guarded=float(vt_g[i]), vt_guarded_valid=bool(ok_g[i]),
                vt_guarded_look=int(look_g[i]),
                band_oracle=vtarget_band(float(vt_o[i])),
                band_guarded=vtarget_band(float(vt_g[i])),
                past=[float(p) for p in past],
                read_lo=int(lo), read_hi=int(hi)))
        eps.append(dict(file=f, eid=int(m["episode_id"]), T=int(t_len),
                        n_windows=int(last.size), sha_ok=bool(ok),
                        in_manifest=bool(ref),
                        n_valid_oracle=int(ok_o.sum()),
                        n_valid_guarded=int(ok_g.sum())))

    return {
        "_what": "ORACLE vs LEAK-GUARDED target-speed labels, canonical val40 grid",
        "_parity": "read-only view of physicalai-val-0c5f7dac3b11; no re-selection",
        "_guard_steps": VT_GUARD_STEPS,
        "_guard_derivation": ("RefCConfig.trajectory.horizons[-1] == 20 and "
                              "lead_source.K_MAX == 20 and the manoeuvre label "
                              "dv = v(t+2s) - v(t) reads pose l+20"),
        "_past_steps": PAST_STEPS,
        "_bands": list(VTARGET_TOKENS),
        "n_windows": len(rows), "n_episodes": len(eps),
        "canonical_881": len(rows) == 881,
        "sha_ok_all": all(e["sha_ok"] for e in eps),
        "episodes": eps, "rows": rows,
    }


if __name__ == "__main__":
    out = build(Path(sys.argv[1]))
    dst = Path(sys.argv[2])
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out), encoding="utf-8")
    print(f"{out['n_windows']} windows / {out['n_episodes']} episodes "
          f"canonical_881={out['canonical_881']} sha_ok_all={out['sha_ok_all']}")
    print(f"valid: oracle={sum(e['n_valid_oracle'] for e in out['episodes'])} "
          f"guarded={sum(e['n_valid_guarded'] for e in out['episodes'])}")
    print(f"wrote {dst} ({dst.stat().st_size} B)")
