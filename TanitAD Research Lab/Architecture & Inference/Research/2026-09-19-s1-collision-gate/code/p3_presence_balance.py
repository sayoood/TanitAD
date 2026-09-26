"""P3's presence/no-object balance shift, measured THROUGH THE REAL SET LOSS.

`P3_PREBUILD.md` §5 names the risk in counts: swapping the 360° target population for the
gate-relevant one takes halfA from 18.95 to 4.63 matched targets per labelled window, and takes
the share of labelled windows with NO targets at all from 2.26 % to 24.88 %. This probe turns
those counts into the quantity that actually reaches the optimiser, by calling
``agent_slots.slot_set_loss`` itself rather than re-deriving what it does.

⛔ CONTROLS, because a probe that tunes on what it scores manufactures results:
 * **IDENTITY control** — run both arms on the SAME count sequence. They must read EXACTLY equal
   on every term; any difference is the harness, not the population.
 * **CONSTANT control** — a sequence of all-zero-target windows must read the no-information
   value: matched 0, every geometry term exactly 0.0 with n = 0, and a presence loss that is pure
   no-object.
 * The predictions are drawn ONCE from a fixed seed and reused across arms, so the only thing that
   differs between arms is the target count sequence.

Counts come from the BANKED census (`p3_targets_<cache>.npz`: `n_raw`, `n_gate`, `labelled`), i.e.
the real per-window distributions, never their means — a mean would hide the zero-target windows
that are the whole point.

Usage:
    python p3_presence_balance.py --bank <p3_targets_halfA.npz> --out <json> [--n-windows 400]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "stack"))

from tanitad.models import agent_slots as AS        # noqa: E402

N_QUERIES = 100          # the refcv6 box3d decoder's own count (`refcv6_perception.n_queries`)
PAD = 32


def _batch(counts: np.ndarray, gen: torch.Generator) -> tuple[dict, dict]:
    """One batch: `len(counts)` windows, window i carrying `counts[i]` valid targets."""
    b = len(counts)
    pred = {"box": torch.randn(b, N_QUERIES, 4, generator=gen) * 5.0,
            "yaw_vec": torch.randn(b, N_QUERIES, 2, generator=gen),
            "cls_logits": torch.randn(b, N_QUERIES, 10, generator=gen),
            "presence_logit": torch.randn(b, N_QUERIES, generator=gen),
            "occ_logit": torch.randn(b, N_QUERIES, generator=gen),
            "rates": torch.randn(b, N_QUERIES, 3, generator=gen)}
    valid = torch.zeros(b, PAD, dtype=torch.bool)
    for i, c in enumerate(counts):
        valid[i, :min(int(c), PAD)] = True
    tgt = {"box": torch.randn(b, PAD, 4, generator=gen) * 5.0,
           "yaw": torch.randn(b, PAD, generator=gen),   # tgt yaw is an ANGLE, pred is a vector
           "valid": valid,
           "cls": torch.zeros(b, PAD, dtype=torch.long),
           "occ": torch.zeros(b, PAD),
           "rates": torch.randn(b, PAD, 3, generator=gen),
           "rates_mask": valid.clone()}
    return pred, tgt


def arm(counts: np.ndarray, seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    pred, tgt = _batch(counts, gen)
    with torch.no_grad():
        out = AS.slot_set_loss(pred, tgt)
    n = out["n"]
    matched, b = int(n["matched"]), len(counts)
    # the presence term's weight mass, computed the way `slot_set_loss` builds `wgt_pres`
    pos = float(matched)
    neg = float(b * N_QUERIES - matched) * AS.NO_OBJECT_W
    return {"n_windows": b,
            "matched_total": matched, "matched_per_window": round(matched / max(b, 1), 4),
            "windows_with_zero_matched": int((np.asarray(counts) == 0).sum()),
            "frac_windows_zero_matched": round(float((np.asarray(counts) == 0).mean()), 4),
            "presence_positive_weight_share": round(pos / max(pos + neg, 1e-9), 4),
            "loss_presence": round(float(out["loss_presence"]), 6),
            "loss_centre": round(float(out["loss_centre"]), 6),
            "n_centre": int(n["centre"]), "n_rates": int(n["rates"]),
            "total": round(float(out["total"]), 6)}


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-windows", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    z = np.load(a.bank)
    lab = z["labelled"].astype(bool)
    raw, gate = z["n_raw"][lab], z["n_gate"][lab]
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(len(raw), size=min(a.n_windows, len(raw)), replace=False)
    c360, cgate = raw[pick], gate[pick]

    res = {"bank": Path(a.bank).name, "n_queries": N_QUERIES, "pad": PAD,
           "no_object_w": AS.NO_OBJECT_W, "seed": a.seed,
           "arm_360": arm(c360, a.seed), "arm_gate": arm(cgate, a.seed)}
    # ---- controls ------------------------------------------------------------------
    res["control_identity"] = {"arm_a": arm(c360, a.seed), "arm_b": arm(c360, a.seed)}
    res["control_identity"]["equal"] = (
        res["control_identity"]["arm_a"] == res["control_identity"]["arm_b"])
    zeros = arm(np.zeros(len(pick), dtype=int), a.seed)
    res["control_constant_zero_targets"] = zeros
    res["control_constant_reads_no_information_value"] = bool(
        zeros["matched_total"] == 0 and zeros["n_centre"] == 0
        and zeros["loss_centre"] == 0.0 and zeros["presence_positive_weight_share"] == 0.0)

    r3, rg = res["arm_360"], res["arm_gate"]
    res["shift"] = {
        "matched_per_window": [r3["matched_per_window"], rg["matched_per_window"]],
        "matched_ratio": round(r3["matched_per_window"] / max(rg["matched_per_window"], 1e-9), 3),
        "presence_positive_weight_share":
            [r3["presence_positive_weight_share"], rg["presence_positive_weight_share"]],
        "frac_windows_zero_matched":
            [r3["frac_windows_zero_matched"], rg["frac_windows_zero_matched"]],
        "loss_presence": [r3["loss_presence"], rg["loss_presence"]],
    }
    # ⭐ THE KNOB THIS IMPLIES, solved rather than tuned: the NO_OBJECT_W that holds the
    # gate arm's positive weight share EQUAL to the 360° arm's, so a P3 control arm can
    # separate "the population helped" from "the presence term re-balanced".
    #   share = pos / (pos + (Q - pos) * W)  ->  W = pos * (1 - share) / (share * (Q - pos))
    _share = r3["presence_positive_weight_share"]
    _pos = rg["matched_per_window"]
    res["shift"]["no_object_w_for_matched_balance"] = round(
        _pos * (1.0 - _share) / max(_share * (N_QUERIES - _pos), 1e-9), 5)
    res["shift"]["no_object_w_today"] = AS.NO_OBJECT_W
    ok = (res["control_identity"]["equal"]
          and res["control_constant_reads_no_information_value"])
    res["controls_pass"] = bool(ok)
    print(json.dumps({k: res[k] for k in
                      ("bank", "shift", "controls_pass",
                       "control_constant_reads_no_information_value")}, indent=1))
    if not ok:
        print("ZZP3BAL-CONTROLS-FAILED — the numbers above are NOT admissible ZZ", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("ZZP3BAL-WROTE %s ZZ" % a.out, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
