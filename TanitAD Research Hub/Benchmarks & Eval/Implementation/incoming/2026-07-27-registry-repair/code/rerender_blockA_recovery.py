#!/usr/bin/env python3
"""RE-RENDER (not recompute) the one blockA node whose printed interval
contradicts its printed verdict.

⛔ WHAT THIS IS NOT. It does not recompute a statistic, does not change an
estimator, does not re-run a model and does not restate a headline. It replays
the EXACT deterministic pipeline that produced the committed node — the same
per-window ``pw_*.npz`` dumps, the same ``taniteval.pseudosim.score_windows``,
the same ``taniteval.ci.paired_episode_cluster_bootstrap`` at B=2000 / seed 0 —
so that the estimator output is bit-for-bit the same object it was, and only
``ci.py``'s RENDERING (fixed at f45b100) differs.

THE CHECK THAT PROVES IT IS A RE-RENDER: every field that the rendering fix does
NOT touch (``p_delta_gt0``, ``separated``, ``n_windows``, ``n_episodes``,
``n_rows_paired``, ``reducer``, ``estimator``) must be IDENTICAL to the committed
node, and ``delta``/``lo``/``hi``/``ci95`` must be identical AT 4 dp. Anything
else is a recompute and the script fails loud.

TARGET (named in `…/2026-07-27-vtband-wiring/VTBAND_WIRING.md` §4.3):
  artifact  …/2026-07-28-tactical-action-input/artifacts/blockA/blockA_full_panel_20arm.json
  node      paired / refc_base_produced__minus__refc_base_v0on / recovery
  printed   delta 0.0  [0.0, 0.0]  separated=true  p_delta_gt0 0.9885  n=13184 / 40 ep

Run (dev box, no GPU, no pod):
  "C:/Users/Admin/venvs/tanitad/Scripts/python.exe" <this file>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "taniteval"))

from taniteval import ci as _ci          # noqa: E402
from taniteval import pseudosim as PS    # noqa: E402

HUB = REPO / "TanitAD Research Hub"
AI = HUB / "Architecture & Inference" / "Implementation" / "incoming"
BE = HUB / "Benchmarks & Eval" / "Implementation" / "incoming"

PANEL = AI / "2026-07-28-tactical-action-input" / "artifacts" / "blockA" / "blockA_full_panel_20arm.json"
PW = {
    # exactly the dirs the 20-arm combine gathered from
    "refc_base_produced": BE / "2026-07-27-pseudosim-arm-panel" / "artifacts" / "pw_refc_base_produced.npz",
    "refc_base_v0on": AI / "2026-07-28-tactical-action-input" / "artifacts" / "blockA" / "pw_refc_base_v0on.npz",
    "v1_tactical_follow": BE / "2026-07-27-pseudosim-arm-panel" / "artifacts" / "pw_v1_tactical_follow.npz",
}
ARM_A, ARM_B, METRIC = "refc_base_produced", "refc_base_v0on", "recovery"
OUT = Path(__file__).resolve().parents[1] / "raw" / "rerender_blockA_recovery.json"


# --- verbatim from `…/2026-07-27-pseudosim-arm-panel/scripts/panel_combine.py` ---
def load_pw(path):
    z = np.load(path, allow_pickle=False)
    return {"traj": torch.as_tensor(z["traj"]),
            "ref_path": torch.as_tensor(z["ref_path"]),
            "ref_yaw": torch.as_tensor(z["ref_yaw"]),
            "v0": torch.as_tensor(z["v0"]),
            "pt_dlat": torch.as_tensor(z["pt_dlat"]),
            "pt_dyaw": torch.as_tensor(z["pt_dyaw"]),
            "pt_dlon": torch.as_tensor(z["pt_dlon"]),
            "anchor": torch.as_tensor(z["anchor"]),
            "ep_i": torch.as_tensor(z["ep_i"]),
            "eid": [str(x) for x in z["eid"]]}


def key_of(pw):
    return np.stack([pw["ep_i"].numpy(), pw["anchor"].numpy(),
                     np.round(pw["pt_dlat"].numpy(), 6) * 1e3,
                     np.round(pw["pt_dyaw"].numpy(), 6) * 1e3,
                     pw["pt_dlon"].numpy()], axis=1).astype(np.int64)


def paired(a, b, eid, n_boot=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    out = _ci.paired_episode_cluster_bootstrap(a[m], b[m], list(np.asarray(eid)[m]), n_boot=n_boot)
    out["n_rows_paired"] = int(m.sum())
    return out
# --- end verbatim ---


def main() -> int:
    committed = json.loads(PANEL.read_text(encoding="utf-8"))
    ref_arm = committed["reference_arm"]
    assert ref_arm == "v1_tactical_follow", ref_arm
    old = committed["paired"][f"{ARM_A}__minus__{ARM_B}"][METRIC]

    arms = {n: load_pw(p) for n, p in PW.items()}

    # the panel's own admissibility gate, re-asserted here
    kref = key_of(arms[ref_arm])
    row_identity = {}
    for n, pw in arms.items():
        k = key_of(pw)
        ok = (k.shape == kref.shape) and bool((k == kref).all())
        row_identity[n] = {"n_rows": int(k.shape[0]), "identical_to_reference": ok}
        assert ok, f"{n} fails row identity against {ref_arm} — pairing inadmissible"
    eid = arms[ref_arm]["eid"]
    eid_same = {n: bool(arms[n]["eid"] == eid) for n in arms}
    assert all(eid_same.values()), eid_same

    vals = {n: PS.score_windows(arms[n])[METRIC] for n in (ARM_A, ARM_B)}
    new = paired(vals[ARM_A], vals[ARM_B], eid)

    # ---- the re-render proof -------------------------------------------- #
    exact = ["p_delta_gt0", "separated", "n_windows", "n_episodes",
             "n_rows_paired", "reducer", "n_boot", "estimator"]
    at4 = ["delta", "lo", "hi", "ci95"]
    mism = {k: (old.get(k), new.get(k)) for k in exact if old.get(k) != new.get(k)}
    mism |= {k: (old.get(k), round(float(new[k]), 4))
             for k in at4 if round(float(old[k]), 4) != round(float(new[k]), 4)}
    identical = not mism

    res = {
        "_what": "RE-RENDER of one committed node. The statistics are unchanged; "
                 "only ci.py's display precision differs (fix at f45b100).",
        "_evidence_class": "MEASURED (ours) — deterministic replay of the committed "
                           "per-window dumps; no model, no GPU, no pod.",
        "_estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B=2000, seed=0, "
                      "unit = val episode. overlapping_holdout_se NEVER called.",
        "_source_artifact": str(PANEL.relative_to(REPO)).replace("\\", "/"),
        "_source_node": f"paired/{ARM_A}__minus__{ARM_B}/{METRIC}",
        "_source_artifact_unmodified": True,
        "_inputs": {n: str(p.relative_to(REPO)).replace("\\", "/") for n, p in PW.items()},
        "_reference_arm": ref_arm,
        "_row_identity": row_identity,
        "_eid_identical_across_arms": eid_same,
        "committed_rendering": old,
        "rerendered": new,
        "RE_RENDER_VERIFIED_IDENTICAL": identical,
        "_verification_rule": "fields untouched by the rendering fix must be EQUAL; "
                              "delta/lo/hi/ci95 must be equal AT 4 dp. A difference in "
                              "any of them would mean this was a RECOMPUTE, not a re-render.",
        "_mismatches": mism,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("_row")}, indent=1))
    print(f"\n[rerender] wrote {OUT}")
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
