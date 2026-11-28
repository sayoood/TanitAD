"""E-RECON-1 — does DINOv2's 91x `lead_gap` READOUT advantage show up in REF-A's DRIVING?

THE DISCRIMINATING EXPERIMENT, AND WHY IT IS THE CHEAP ONE
----------------------------------------------------------
C104 (registry 12.1) measured, through the deployed pool on the same windows:

    lead_gap   r2   DINOv2-B/14 0.44997   vs   ours 0.00496    (91x)
    ego_v0     r2               0.71733   vs        0.05240    (13.7x)
    lead_closing r2             0.01713   vs        0.00000

REF-A is that same frozen DINOv2-B/14 inside the same 4-brain, and it drives 5.1x WORSE
than the flagship. The two results have been read as a contradiction. They are only a
contradiction if the readout rung and the driving metric are about the same quantity —
and NOBODY HAS EVER MEASURED THE DRIVING SIDE OF THE `lead_gap` RUNG.

They have not, because `taniteval.driving` REFUSES the whole family:

    driving.py:608  "no lead-agent state exists (lead_state is a None stub)"

That refusal is now STALE. `taniteval/taniteval/lead_source.py` (the `obstacle.offline`
-> `win["lead"]` wiring) landed, and a val40 lead block was built and row-verified against
these exact dumps:

    raw/val40_lead_block.npz            881 rows, LEAD 270 / NO_LEAD 551 / NO_LABEL 60
    raw/alignment_report.json           gt within 4 ulp, eid partition matches, 881 rows

`raw/ade_by_lead_state.json` already used it — for `refc-base-30k`, `flagship-30k` and `cv`.
**REF-A IS ABSENT FROM IT.** So the one arm whose encoder is the subject of C104 has never
been scored on the one stratification that could test C104's driving relevance. This script
adds it. Zero GPU: every path was already scored and banked.

PRE-REGISTRATION — BOTH OUTCOMES COMMITTED BEFORE THE RUN
----------------------------------------------------------
Primary statistic: the DIFFERENCE OF PAIRED DIFFERENCES

    D_lead    = mean_LEAD    (REF-A err - flagship err)
    D_nolead  = mean_NO_LEAD (REF-A err - flagship err)
    contrast  = D_lead - D_nolead          (paired episode-cluster bootstrap)

on `ade_0_2s`, and separately on the LONGITUDINAL family members (`speed_mae_mps`,
`long_abs_2s_m`) and the LATERAL ones (`lat_abs_2s_m`, `heading_mae_2s_deg`).

    O1  contrast < 0, CI-separated
        REF-A's deficit SHRINKS where a lead vehicle is present.
        => DINOv2's readable lead information IS partly usable by the driving stack, and
           C104's rung is measuring something the driving arm can and does exploit. The
           encoder gap is then real AND directionally relevant, and REF-A's overall loss
           must be attributed to something else (capacity / objective), not to the encoder
           lacking lead information.

    O2  contrast ~ 0, CI containing zero
        REF-A's deficit is the SAME with and without a lead in front.
        => The 91x readout advantage buys the driving stack NOTHING on the very quantity it
           was measured on. READABLE != USABLE, demonstrated on the rung itself rather than
           argued. C104 stays true as a statement about linear decodability of a frozen
           representation, and stops being evidence about driving.

    O3  contrast > 0, CI-separated
        REF-A's deficit is WORSE where a lead is present.
        => Strongest form of O2 plus a positive finding: the frozen arm degrades exactly
           where the scene is most interactive, which points at the predictor/objective
           rather than at what the encoder encodes.

⛔ NO OUTCOME OF THIS TEST CAN "PROVE C104 WRONG". C104 is a T0 linear-readout fact about a
frozen representation and this is a T0 driving-fidelity fact about a trained stack. The test
decides RELEVANCE, not correctness. That asymmetry is registered here so it cannot be
narrated away afterwards.

⚠️ POWER, registered in advance: LEAD n=270 windows in 21 episode clusters, NO_LEAD n=551 in
(counted at runtime). The bootstrap resamples EPISODES, so 21 clusters is the real n for the
LEAD arm. A null here is "not separated at n=21 clusters", NEVER "no effect exists".

⚠️ CONFOUND, registered in advance: LEAD windows are not a random subsample — they are
slower and more urban (the lead block's own `by_speed` shows lead_rate 0.72 at 0-1 m/s).
The contrast is therefore reported ALSO within speed bands, so "there is a lead" is not
silently standing in for "the ego is slow".

TIER: T0 throughout (`taniteval.driving/tier0` dumps, `rollout.collect` feeds the expert's
true future actions — rollout.py:146). NEVER quotable as driving performance.

Usage:
    PYTHONPATH=<repo>/taniteval python refa_lead_rung.py --out raw/
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from taniteval import ci as tci
from taniteval import driving as tdrv
from taniteval import lead_metrics as tlm

DT_WP_S = 0.5      # the 4 banked waypoints are 0.5 s apart (wp_steps 5/10/15/20 at 10 Hz)

ARMS = ("refa-dinov2", "flagship-30k", "refa-dynin-30k")

# Metrics carried through the lead stratification, by family.
METRICS = {
    "ade_0_2s": "TRAJECTORY",
    "long_abs_2s_m": "LONGITUDINAL",
    "speed_mae_mps": "LONGITUDINAL",
    "progress_abs_err_m": "LONGITUDINAL",
    "lat_abs_2s_m": "LATERAL",
    "heading_mae_2s_deg": "LATERAL",
    "pathgeom_crosstrack_m": "LATERAL",
}


def alignment_gate(block, dumps):
    """The lead block must attach ROW-FOR-ROW, and that is checked, not assumed.

    The banked `alignment_report.json` established this for refc/flagship. It is re-checked
    here for the REF-A dumps because a claim inherited from another package's report is
    exactly what this programme's rules forbid quoting.
    """
    ref = dumps["flagship-30k"]
    n = ref["pred"].shape[0]
    speeds_blk = np.asarray(block["speeds"], dtype=np.float64)
    out = {
        "n_block_rows": int(len(block["state"])),
        "n_dump_rows": int(n),
        "row_counts_match": bool(len(block["state"]) == n),
        # eid STRINGS differ by design (block 'ep_00000' vs dump '0'); what must match is
        # the PARTITION — same episode boundaries in the same order.
        "eid_partition_matches": None,
        "speed_max_abs_diff_mps": None,
        "speed_corr": None,
        "wp_rel_s_block": [float(x) for x in np.asarray(block["ts_rel_s"]).tolist()],
        "wp_steps_dump": list(ref["wp_steps"]),
    }
    def runs(seq):
        out_, last, c = [], None, 0
        for x in seq:
            if x != last:
                if last is not None:
                    out_.append(c)
                last, c = x, 1
            else:
                c += 1
        out_.append(c)
        return out_
    out["eid_partition_matches"] = bool(
        runs([str(x) for x in block["eid"]]) == runs([str(x) for x in ref["eid"]]))
    sd = ref["speed"].numpy().astype(np.float64)
    out["speed_max_abs_diff_mps"] = round(float(np.abs(speeds_blk - sd).max()), 6)
    out["speed_corr"] = round(float(np.corrcoef(speeds_blk, sd)[0, 1]), 8)
    out["_speed_note"] = (
        "the block recomputes v0 from the registered clip clock (realised spacing "
        "~0.1007 s, not 0.1), so a small difference is EXPECTED and is not misalignment; "
        "the partition test and the correlation are the alignment evidence")
    out["passed"] = bool(out["row_counts_match"] and out["eid_partition_matches"]
                         and out["speed_corr"] > 0.999)
    return out


def contrast(pa, pb, eid, mask_a, mask_b, n_boot=2000, seed=0):
    """(A-B on mask_a) minus (A-B on mask_b), one bootstrap, EPISODES resampled jointly.

    ⛔ NOT two separate intervals differenced — that is the quadrature error the programme
    already banned in another costume. The two strata share episodes, so the difference of
    strata means must be resampled inside ONE draw.
    """
    e = np.asarray([str(x) for x in eid])
    ia, ib = np.flatnonzero(mask_a), np.flatnonzero(mask_b)
    uniq, idx_by_ep = tci.episode_index(e)
    set_a, set_b = set(ia.tolist()), set(ib.tolist())

    def stat(sel):
        sa = [i for i in sel if i in set_a]
        sb = [i for i in sel if i in set_b]
        if not sa or not sb:
            return np.nan
        return ((pa[sa].mean() - pb[sa].mean())
                - (pa[sb].mean() - pb[sb].mean()))

    point = ((pa[ia].mean() - pb[ia].mean()) - (pa[ib].mean() - pb[ib].mean()))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_ep[p] for p in pick])
        draws.append(stat(sel))
    d = np.asarray(draws, dtype=np.float64)
    d = d[np.isfinite(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"contrast": round(float(point), 4),
            "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "p_gt0": round(float((d > 0).mean()), 4),
            "separated": bool(lo > 0 or hi < 0),
            "n_boot_finite": int(d.size),
            "n_a": int(ia.size), "n_b": int(ib.size),
            "n_episodes_a": int(len(set(e[ia].tolist()))),
            "n_episodes_b": int(len(set(e[ib].tolist()))),
            "estimator": "paired_episode_cluster_bootstrap (difference of strata deltas, "
                         "one joint draw)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    here = pathlib.Path(__file__).resolve()
    repo = next(p for p in here.parents if (p / "taniteval" / "results").is_dir())
    res = repo / "taniteval" / "results"
    blkp = (repo / "TanitAD Research Hub" / "Benchmarks & Eval" / "Implementation"
            / "incoming" / "2026-08-04-distance-keeping-arms" / "raw"
            / "val40_lead_block.npz")
    block = np.load(blkp, allow_pickle=True)
    dumps = {a: torch.load(res / f"windows_{a}.pt", map_location="cpu",
                           weights_only=False) for a in ARMS}

    align = alignment_gate(block, dumps)
    if not align["passed"]:
        print(json.dumps({"ALIGNMENT_GATE": "FAILED", **align}, indent=1))
        return 2

    state = np.asarray(block["state"]).astype(str)
    eid = [str(x) for x in dumps["flagship-30k"]["eid"]]
    pw = {a: tdrv.per_window(dumps[a]["pred"], dumps[a]["gt"]) for a in ARMS}
    masks = {s: state == s for s in ("LEAD", "NO_LEAD", "NO_LABEL")}

    out = {
        "block": "refa_reconciliation/E-RECON-1_lead_rung",
        "tier": "T0",
        "tier_note": ("teacher-forced (rollout.collect feeds the expert's true future "
                      "actions, rollout.py:146). NEVER driving performance."),
        "prereg": {"O1": "contrast < 0 separated -> readout advantage IS usable",
                   "O2": "contrast ~ 0 -> READABLE != USABLE, on the rung itself",
                   "O3": "contrast > 0 separated -> worse where the scene is interactive"},
        "alignment_gate": align,
        "lead_block": str(blkp.relative_to(repo)).replace("\\", "/"),
        "state_counts": {s: int(m.sum()) for s, m in masks.items()},
        "by_state": {},
        "contrast_LEAD_minus_NOLEAD": {},
        "distance_keeping": {},
        "within_speed_band": {},
    }

    # ---- per-state point + paired deltas, per metric family --------------------- #
    for s, m in masks.items():
        e = [x for x, k in zip(eid, m) if k]
        cell = {"n_windows": int(m.sum()), "n_episodes": len(set(e))}
        for met, fam in METRICS.items():
            red = tdrv.REDUCE.get(met, "mean")
            cell[met] = {"family": fam}
            for a in ARMS:
                cell[met][a] = tci.episode_cluster_bootstrap(
                    np.asarray(pw[a][met], np.float64)[m], e,
                    reduce=red, n_boot=args.n_boot, seed=args.seed)
            for a in ("refa-dinov2", "refa-dynin-30k"):
                cell[met][f"{a}_minus_flagship"] = tci.paired_episode_cluster_bootstrap(
                    np.asarray(pw[a][met], np.float64)[m],
                    np.asarray(pw["flagship-30k"][met], np.float64)[m],
                    e, reduce=red, n_boot=args.n_boot, seed=args.seed)
        out["by_state"][s] = cell

    # ---- THE PRIMARY STATISTIC --------------------------------------------------- #
    for a in ("refa-dinov2", "refa-dynin-30k"):
        out["contrast_LEAD_minus_NOLEAD"][a] = {
            met: contrast(np.asarray(pw[a][met], np.float64),
                          np.asarray(pw["flagship-30k"][met], np.float64),
                          eid, masks["LEAD"], masks["NO_LEAD"],
                          n_boot=args.n_boot, seed=args.seed)
            for met in METRICS}

    # ---- the actual distance-keeping family (headway / time-gap / min-TTC) -------- #
    leads = np.asarray(block["leads"], dtype=np.float64)
    lead_lens = np.asarray(block["lead_lens"], dtype=np.float64)
    speeds = np.asarray(block["speeds"], dtype=np.float64)
    dk = {}
    for a in ARMS:
        dk[a] = tlm.distance_keeping(dumps[a]["pred"].double().numpy(), leads,
                                     lead_lens, speeds, dt=DT_WP_S)
    dk["gt"] = tlm.distance_keeping(dumps["flagship-30k"]["gt"].double().numpy(),
                                    leads, lead_lens, speeds, dt=DT_WP_S)
    dk["cv"] = tlm.distance_keeping(dumps["flagship-30k"]["cv"].double().numpy(),
                                    leads, lead_lens, speeds, dt=DT_WP_S)
    out["distance_keeping"]["point"] = {
        a: {k: v for k, v in d.items() if not k.startswith("_")} for a, d in dk.items()}
    out["distance_keeping"]["paired"] = {}
    for a in ("refa-dinov2", "refa-dynin-30k", "flagship-30k"):
        out["distance_keeping"]["paired"][f"{a}_minus_gt"] = tlm.paired_distance_keeping(
            dk[a], dk["gt"], eid, names=(a, "gt"), n_boot=args.n_boot, seed=args.seed)
    out["distance_keeping"]["paired"]["refa-dinov2_minus_flagship-30k"] = (
        tlm.paired_distance_keeping(dk["refa-dinov2"], dk["flagship-30k"], eid,
                                    names=("refa-dinov2", "flagship-30k"),
                                    n_boot=args.n_boot, seed=args.seed))
    out["distance_keeping"]["_dt_note"] = (
        "dt = 0.5 s, the banked waypoint spacing. TTC scales as 1/dt through the closing "
        "rate (lead_metrics:134), so these TTCs are NOT comparable to a dense-path TTC "
        "computed at dt = 0.1 s. Headway and time-gap are dt-invariant and ARE comparable.")

    # ---- the registered confound control: contrast INSIDE speed bands ------------- #
    v0 = dumps["flagship-30k"]["speed"].numpy().astype(np.float64)
    bands = [("lo", v0 < 5.0), ("mid", (v0 >= 5.0) & (v0 < 12.0)), ("hi", v0 >= 12.0)]
    for bname, bmask in bands:
        ml, mn = masks["LEAD"] & bmask, masks["NO_LEAD"] & bmask
        cell = {"n_lead": int(ml.sum()), "n_nolead": int(mn.sum()),
                "n_ep_lead": len(set(np.asarray(eid)[ml].tolist())),
                "n_ep_nolead": len(set(np.asarray(eid)[mn].tolist()))}
        if cell["n_lead"] >= 30 and cell["n_nolead"] >= 30:
            for met in ("ade_0_2s", "speed_mae_mps"):
                cell[met] = contrast(np.asarray(pw["refa-dinov2"][met], np.float64),
                                     np.asarray(pw["flagship-30k"][met], np.float64),
                                     eid, ml, mn, n_boot=args.n_boot, seed=args.seed)
        else:
            cell["status"] = "UNPOWERED"
            cell["reason"] = ("a stratum thinner than 30 windows is reported, never "
                              "quoted (lead_metrics.MIN_STRATUM_N)")
        out["within_speed_band"][bname] = cell

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "refa_lead_rung.json").write_text(json.dumps(out, indent=1, default=str),
                                                encoding="utf-8")
    print(json.dumps({"alignment": align["passed"],
                      "state_counts": out["state_counts"],
                      "wrote": str(outdir / "refa_lead_rung.json")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
