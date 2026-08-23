"""THE DECISIVE MEASUREMENT — how many decodes reproduce the FULL fan?

Follows ``fan_width_sweep.py`` and the same pre-registration
(``PREREG_FAN_WIDTH.md``, staged blob ``1bffa9db6a6047325dceff1ef787d67ab2fd5152``).

THE MECHANISM, stated before the numbers
============================================================================
S2 is MEASURED **exactly** ADE-inert (paired Δ 0.0, both arms). That is not a
null result — read literally it says ``argmax(logits)`` over the FULL fan
**already lands inside the reachability band on every window**. If that is true,
then ANY candidate subset that CONTAINS every band survivor selects the
identical trajectory, because the argmax cannot move to a candidate that was
never going to win.

⇒ the shippable budget is not "N hypotheses"; it is **the per-window survivor
count of a band we can evaluate on the ANCHORS, before any decode runs**.

This module measures the three things that claim needs and cannot be assumed:

  A. **Selection containment.** On what fraction of windows does the full fan's
     argmax survive the ANCHOR-level band? (The anchor band is not the decoded
     band — they agree on ~96 %, so this must be measured, not inherited.)
  B. **Minimum sufficient budget** N_suff = the smallest N for which the
     anchor-band-prefix policy reproduces the full-fan SELECTION INDEX on 100 %
     of windows — a bit-exact criterion, not an ADE tie.
  C. **Paired ADE vs the FULL fan** at every rung, so a budget below N_suff is
     quoted three-sided with an interval rather than as a point.

PLUS the RETRAINED ladder, which the within-arm sweep cannot see
============================================================================
Prefix truncation says what THESE weights do at width N. It does not say what a
model TRAINED at width N does. The programme already has three arms trained at
64 / 128 / 256 on the SAME 881 windows, so the retrained ladder is free — as a
PAIRED comparison, since the windows are shared. ⚠️ It confounds width with
CAPACITY (small/base/XL differ in decoder size too) and that is stated beside
every row, not buried.

⛔ ``overlapping_holdout_se`` is never called.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "6")
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

PREREG = ("TanitAD Research Hub/Architecture & Inference/Implementation/"
          "incoming/2026-08-04-fan-width/PREREG_FAN_WIDTH.md")
PREREG_BLOB_AT_WRITE = "1bffa9db6a6047325dceff1ef787d67ab2fd5152"
ACCEL_MAX, HORIZON_S = 2.5, 2.0
N_BOOT = 2000
LADDER = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256]


def _git(repo: Path, *a: str) -> str:
    try:
        return subprocess.run(["git", *a], cwd=repo, capture_output=True,
                              text=True, timeout=90).stdout.strip()
    except Exception as exc:                                    # pragma: no cover
        return f"<git failed: {exc}>"


def candidate_ade(fan, gt):
    """VERBATIM ``refc_rerank._score_row``."""
    return torch.linalg.norm(fan - gt[:, None], dim=-1).mean(-1)


def band_prefix_idx(keep: torch.Tensor, n: int) -> torch.Tensor:
    """First ``n`` in FPS order that survive ``keep``, topped up in FPS order."""
    return torch.argsort((~keep).long(), dim=1, stable=True)[:, :n].contiguous()


def three_sided(p: dict) -> str:
    if not p["separated"]:
        return "not separated"
    return "better" if p["delta"] < 0 else "worse"


def run(banks: dict[str, str], out_dir: str, anchors_path: str,
        repo: Path) -> dict:
    t0 = time.time()
    for p in (repo / "taniteval", repo / "stack"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from taniteval import ci, four_families                      # noqa: PLC0415
    from tanitad.refs import refc_select as sl                   # noqa: PLC0415

    V = torch.load(anchors_path, map_location="cpu", weights_only=False)
    res = {
        "what": "the decisive budget: how many DECODES reproduce the full fan",
        "prereg": {"path": PREREG,
                   "staged_blob": (_git(repo, "ls-files", "-s", "--", PREREG)
                                   .split() or ["", "<not staged>"])[1],
                   "worktree_blob": _git(repo, "hash-object", PREREG),
                   "blob_at_write_time": PREREG_BLOB_AT_WRITE},
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "estimator": {"paired": "paired_episode_cluster_bootstrap",
                      "unit": "episode", "n_boot": N_BOOT,
                      "⛔": "overlapping_holdout_se is NEVER called"},
        "band": (f"v_mean in [max(0, v0-{ACCEL_MAX*HORIZON_S}), "
                 f"v0+{ACCEL_MAX*HORIZON_S}] m/s — flagship_v15's OWN clamp via "
                 f"refc_select.reachability_mask, accel_max={ACCEL_MAX}, "
                 f"horizon_s={HORIZON_S}. NOT tuned here."),
        "arms": {}, "retrained_ladder": {},
    }
    loaded = {}
    for arm, bp in banks.items():
        d = torch.load(bp, map_location="cpu", weights_only=False)
        loaded[arm] = d
        eid, fan, logits, gt = list(d["eid"]), d["fan"], d["logits"], d["gt"]
        B, NMAX = fan.shape[:2]
        de_all = candidate_ade(fan, gt)
        v0 = d["v0"].to(fan.dtype)
        sel_full = logits.argmax(1)
        de_full = de_all.gather(1, sel_full[:, None]).squeeze(1)

        # ⭐ The nested-FPS-prefix relation was VERIFIED bit-exactly
        # (xl256[:128] == base128, base128[:64] == xl256[:64] == small64,
        # maxabs 0.0), so an arm whose own buffer was not pulled takes the
        # prefix of the widest one. That is an identity, not a substitution —
        # and it is asserted, not assumed.
        if arm in V:
            anc = V[arm][:NMAX].to(fan.dtype)
        else:
            widest = max(V, key=lambda k: V[k].shape[0])
            anc = V[widest][:NMAX].to(fan.dtype)
            for other in V:
                m = min(NMAX, V[other].shape[0])
                assert torch.equal(V[widest][:m], V[other][:m]), (
                    f"nested-FPS-prefix broken between {widest} and {other} — "
                    f"§2.1 is void, STOP")
        keep_anc = sl.reachability_mask(anc[None].expand(B, -1, -1, -1)
                                        .contiguous(), v0,
                                        accel_max=ACCEL_MAX,
                                        horizon_s=HORIZON_S)
        keep_dec = sl.reachability_mask(fan, v0, accel_max=ACCEL_MAX,
                                        horizon_s=HORIZON_S)

        # -- A. SELECTION CONTAINMENT ------------------------------------- #
        in_anc = keep_anc.gather(1, sel_full[:, None]).squeeze(1)
        in_dec = keep_dec.gather(1, sel_full[:, None]).squeeze(1)
        # FPS rank of the winner among the anchor-band survivors
        order = torch.argsort((~keep_anc).long(), dim=1, stable=True)
        pos = (order == sel_full[:, None]).float().argmax(1)      # 0-based
        contain = {
            "full_fan_argmax_inside_DECODED_band_frac":
                round(float(in_dec.double().mean()), 6),
            "full_fan_argmax_inside_ANCHOR_band_frac":
                round(float(in_anc.double().mean()), 6),
            "n_windows_where_winner_fails_the_anchor_band":
                int((~in_anc).sum()),
            "budget_rank_of_the_winner_under_the_anchor_band": {
                "median": float(pos.float().median()) + 1,
                "p95": float(pos.float().quantile(0.95)) + 1,
                "max": int(pos.max()) + 1},
            "⭐": ("the DECODED-band figure is the literal content of 'S2 is "
                   "exactly ADE-inert'. The ANCHOR-band figure is the one a "
                   "pre-decode filter actually gets, and it is measured, not "
                   "inherited from the decoded one."),
        }

        # -- B. MINIMUM SUFFICIENT BUDGET (bit-exact selection identity) --- #
        n_suff = int(pos.max()) + 1
        idx_s = band_prefix_idx(keep_anc, n_suff)
        sel_s = idx_s.gather(1, logits.gather(1, idx_s).argmax(1)[:, None]
                             ).squeeze(1)
        exact = bool(torch.equal(sel_s, sel_full))
        survivors = keep_anc.sum(1)

        # -- C. PAIRED ADE vs the FULL fan at every rung ------------------- #
        rows = []
        for n in [x for x in LADDER if x <= NMAX] + ([n_suff]
                                                     if n_suff not in LADDER
                                                     else []):
            ix = band_prefix_idx(keep_anc, n)
            de_s = de_all.gather(1, ix)
            j = logits.gather(1, ix).argmax(1)
            sel_n = ix.gather(1, j[:, None]).squeeze(1)
            de_n = de_s.gather(1, j[:, None]).squeeze(1)
            p = ci.paired_episode_cluster_bootstrap(
                de_n.numpy().astype(np.float64),
                de_full.numpy().astype(np.float64), list(d["eid"]),
                n_boot=N_BOOT)
            rows.append({
                "n_decodes": n,
                "compute_saving_x": round(NMAX / n, 3),
                "selected_ade_0_2s": round(float(de_n.mean()), 6),
                "full_fan_selected_ade_0_2s": round(float(de_full.mean()), 6),
                "oracle_in_subset": round(float(de_s.min(1).values.mean()), 6),
                "selection_index_identical_to_full_fan_frac":
                    round(float((sel_n == sel_full).double().mean()), 6),
                "paired_vs_FULL_fan": p, "verdict": three_sided(p)})
        rows.sort(key=lambda r: r["n_decodes"])

        # four families at the two operating points that matter
        dt, dt_prov = four_families.infer_dt({"wp_steps": list(d["wp_steps"]),
                                              "dt_s": 0.1})
        fams = {}
        b = torch.arange(B)
        for tag, idx in (("full_fan", sel_full),
                         ("anchor_band_N_suff", sel_s)):
            f = four_families.all_families({"pred": fan[b, idx], "gt": gt,
                                            "wp_steps": list(d["wp_steps"]),
                                            "dt_s": 0.1})
            fams[tag] = f

        res["arms"][arm] = {
            "bank": str(bp), "n_windows": B, "n_anchors": NMAX,
            "n_episodes": len(set(eid)),
            "A_selection_containment": contain,
            "B_minimum_sufficient_budget": {
                "N_suff": n_suff,
                "compute_saving_x": round(NMAX / n_suff, 3),
                "selection_bit_identical_to_full_fan": exact,
                "criterion": ("the smallest N whose anchor-band prefix "
                              "reproduces the full-fan SELECTION INDEX on "
                              "100 % of windows — bit-exact, not an ADE tie"),
                "anchor_band_survivors_per_window": {
                    "median": float(survivors.float().median()),
                    "mean": round(float(survivors.double().mean()), 2),
                    "min": int(survivors.min()), "max": int(survivors.max()),
                    "p95": float(survivors.float().quantile(0.95))},
                "⚠️": ("N_suff is driven by the WORST window, so it is the "
                       "worst-case budget. The median window needs far fewer; "
                       "a variable-width fan would be cheaper still and is NOT "
                       "measured here."),
            },
            "C_budget_ladder_vs_full_fan": rows,
            "four_families": {"_dt_s": dt, "_dt_provenance": dt_prov, **fams},
        }

    # ---- THE RETRAINED LADDER (paired, same windows, capacity-confounded) -- #
    names = list(loaded)
    ref = loaded[names[0]]
    align = {}
    for a in names:
        align[a] = {
            "gt_bit_identical_to_" + names[0]:
                bool(torch.equal(loaded[a]["gt"], ref["gt"])),
            "eid_identical": bool(list(loaded[a]["eid"]) == list(ref["eid"])),
            "n_windows": int(loaded[a]["fan"].shape[0])}
    res["retrained_ladder"]["window_alignment"] = align
    if all(v["gt_bit_identical_to_" + names[0]] and v["eid_identical"]
           for v in align.values()):
        per = {}
        for a in names:
            d = loaded[a]
            de = candidate_ade(d["fan"], d["gt"])
            s = d["logits"].argmax(1)
            per[a] = {"sel": de.gather(1, s[:, None]).squeeze(1).numpy()
                      .astype(np.float64),
                      "or": de.min(1).values.numpy().astype(np.float64),
                      "n": int(d["fan"].shape[1])}
        eid = list(ref["eid"])
        pairs = []
        order = sorted(names, key=lambda a: per[a]["n"])
        for i in range(len(order) - 1):
            lo, hi = order[i], order[i + 1]
            pairs.append({
                "narrower": f"{lo} ({per[lo]['n']} anchors)",
                "wider": f"{hi} ({per[hi]['n']} anchors)",
                "paired_selected_narrow_minus_wide":
                    ci.paired_episode_cluster_bootstrap(
                        per[lo]["sel"], per[hi]["sel"], eid, n_boot=N_BOOT),
                "paired_oracle_narrow_minus_wide":
                    ci.paired_episode_cluster_bootstrap(
                        per[lo]["or"], per[hi]["or"], eid, n_boot=N_BOOT)})
        res["retrained_ladder"]["pairs"] = pairs
        res["retrained_ladder"]["points"] = {
            a: {"n_anchors": per[a]["n"],
                "selected_ade_0_2s": round(float(per[a]["sel"].mean()), 6),
                "oracle_in_fan_ade_0_2s": round(float(per[a]["or"].mean()), 6)}
            for a in order}
        res["retrained_ladder"]["⚠️_confound"] = (
            "these three arms differ in DECODER CAPACITY as well as anchor "
            "count (refc-small / -base / -xl), so a difference here is "
            "width+capacity, never width alone. It is reported because it is "
            "the only RETRAINED ladder that exists and because prefix "
            "truncation provably cannot answer the retrained question — not "
            "because the confound is small.")
    else:
        res["retrained_ladder"]["status"] = (
            "SKIPPED — the banks are not on identical windows; a paired test "
            "across them would be a category error")

    res["wall_s"] = round(time.time() - t0, 1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "reachable_budget.json"

    def _c(o):
        if isinstance(o, dict):
            return {k: _c(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_c(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return f"<ndarray {o.shape}>"
        if isinstance(o, torch.Tensor):
            return o.tolist() if o.numel() < 32 else f"<tensor {tuple(o.shape)}>"
        return o

    dest.write_text(json.dumps(_c(res), indent=1, ensure_ascii=False),
                    encoding="utf-8")
    for a, v in res["arms"].items():
        print(f"[budget] {a}: N_suff = {v['B_minimum_sufficient_budget']['N_suff']}"
              f" of {v['n_anchors']} "
              f"({v['B_minimum_sufficient_budget']['compute_saving_x']}x), "
              f"selection bit-identical = "
              f"{v['B_minimum_sufficient_budget']['selection_bit_identical_to_full_fan']}",
              flush=True)
    print(f"[budget] -> {dest} ({res['wall_s']}s)", flush=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchors", required=True)
    a = ap.parse_args(argv)
    repo = Path(a.repo)
    banks = {
        "refc-base-30k": str(repo / "taniteval/results/fan_refc-base-30k.pt"),
        "refc-xl-30k": str(repo / "taniteval/results/fan_refc-xl-30k.pt"),
        "refc-small-30k": str(repo / "TanitAD Research Hub/Benchmarks & Eval/"
                              "Implementation/incoming/2026-07-22-refc-small-30k/"
                              "fan_refc-small-30k.pt"),
    }
    run(banks, a.out, a.anchors, repo)
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
