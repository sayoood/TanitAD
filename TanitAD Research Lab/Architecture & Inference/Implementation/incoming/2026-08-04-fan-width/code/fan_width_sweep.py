"""FAN WIDTH — how many trajectory hypotheses does REF-C actually need?

Pre-registration: ``…/2026-08-04-fan-width/PREREG_FAN_WIDTH.md``, staged blob
``1bffa9db6a6047325dceff1ef787d67ab2fd5152``. Every threshold here is a
TRANSCRIPTION of §4/§5/§6 and none may be edited by a run; the module prints
both blob ids so a reader can re-run ``git hash-object`` and check.

WHY A PREFIX OF THE BANK IS AN EXACT DECODE AT WIDTH N (prereg §2)
============================================================================
1. ``furthest_point_sample`` is greedy FPS, so its chosen list is NESTED and
   ``chosen[:N]`` IS the FPS-N solution. VERIFIED bit-exactly on the buffers
   pulled from both checkpoints: ``xl256[:128] == base128`` and
   ``base128[:64] == xl256[:64] == small64``, maxabs 0.0.
2. ``CrossAttnLayer`` cross-attends the IMAGE conv map only — there is NO
   attention over the candidate axis — and every graft on these two arms
   (``maneuver_to_anchor``, an ``nn.Linear(n_man, N, bias=False)``) is a
   per-anchor row. So ``_decode`` maps candidate i to ``(conf_i, offset_i)``
   with zero dependence on the other candidates.

⇒ ``fan[:, :N]`` / ``logits[:, :N]`` are BIT-EXACTLY what the decoder emits with
its anchor buffer truncated to N rows. C-flags (§5) asserts the two
cross-candidate couplings that WOULD break this — ``_goal_along_prior``'s
across-fan z-score and ``_apply_grafts``'s group-norm clamp — are absent from
both checkpoints.

⛔ WHAT THIS DOES NOT CLAIM. It is a statement about INFERENCE at width N from
THESE weights. It says nothing about a model RETRAINED at width N.

ESTIMATOR, DECLARED BEFORE ANY NUMBER
============================================================================
``taniteval.ci.episode_cluster_bootstrap`` / ``paired_episode_cluster_bootstrap``
resampling unit = **episode**, ``n_boot = 2000``. ⛔ ``overlapping_holdout_se``
is never called: it is not a jackknife, not a valid SE, and it BIASES the point
estimate (mean-of-split-means, -6.67 % to +11.69 % over 27 arms, up to a sign
flip on paired deltas).

USAGE
============================================================================
    OMP_NUM_THREADS=6 python fan_width_sweep.py \
        --bank taniteval/results/fan_refc-xl-30k.pt \
        --anchors refc_anchor_vocab.pt --arm refc-xl-30k --out <raw dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

# torch spawns ~113 threads PER PROCESS and a multi-arm panel then makes NO
# progress while looking exactly like a deadlock (MEASURED 2026-07-27).
os.environ.setdefault("OMP_NUM_THREADS", "6")
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))


def _add_paths(repo: Path) -> None:
    for p in (repo / "taniteval", repo / "stack"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


PREREG = ("TanitAD Research Hub/Architecture & Inference/Implementation/"
          "incoming/2026-08-04-fan-width/PREREG_FAN_WIDTH.md")
PREREG_BLOB_AT_WRITE = "1bffa9db6a6047325dceff1ef787d67ab2fd5152"

#: TRANSCRIPTION of prereg §4/§5/§6. None of these may be edited by a run.
THRESHOLDS = {
    "ladder": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    "saturation": ("smallest N with paired-vs-Nmax NOT separated AND "
                   "|delta| <= sat_tol_m"),
    "sat_tol_m": 0.02,          # = PREREG_D-SEL §6.3 `free_win_m`, transcribed
    "full_rung_tol_m": 1.5e-4,  # 4-dp rounding half-width (refc_sel_probe)
    "red_flag_m": 0.10,         # beat published by more than this => leak audit
    "random_seeds": 24,
    "random_prefix_min_rungs_won": 7,   # of 9; below this the "principled
                                        # subset" premise is WITHDRAWN
    "band_fidelity_max_disagree": 0.20,
    "accel_max": 2.5, "horizon_s": 2.0,
    "registered_prediction": ("O1 SELECTOR-BOUND with N*(selected) <= 32 and "
                              "N*(oracle) 128/256 or unreached (possibly O4); "
                              "P2: R2 at wide N, R1 below ~16"),
}

#: prereg §5 C-full-rung. n travels with the number or the number is not
#: quotable: these are 881-window values.
PUBLISHED = {
    "refc-base-30k": {"selected_ade2s": 0.4728, "oracle_ade2s": 0.1914,
                      "n_anchors": 128},
    "refc-xl-30k": {"selected_ade2s": 0.4714, "oracle_ade2s": 0.1640,
                    "n_anchors": 256},
    "refc-small-30k": {"selected_ade2s": 0.5261, "oracle_ade2s": 0.2213,
                       "n_anchors": 64},
}
N_BOOT = 2000


# --------------------------------------------------------------------------- #
# provenance                                                                   #
# --------------------------------------------------------------------------- #

def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True,
                              text=True, timeout=90).stdout.strip()
    except Exception as exc:                                    # pragma: no cover
        return f"<git failed: {exc}>"


def prereg_provenance(repo: Path) -> dict:
    staged = _git(repo, "ls-files", "-s", "--", PREREG).split()
    worktree = _git(repo, "hash-object", PREREG)
    blob = staged[1] if len(staged) > 1 else "<not staged>"
    return {"path": PREREG, "staged_blob": blob, "worktree_blob": worktree,
            "blob_at_write_time": PREREG_BLOB_AT_WRITE,
            "thresholds_unmoved_since_staging": bool(blob and blob == worktree),
            "thresholds_unmoved_since_the_code_was_written":
                bool(worktree == PREREG_BLOB_AT_WRITE),
            "thresholds": THRESHOLDS}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# the quantities                                                               #
# --------------------------------------------------------------------------- #

def candidate_ade(fan: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """[B, N] per-candidate ADE@2s — VERBATIM ``refc_rerank._score_row``.

    Re-deriving it is how two definitions of the headline quantity drift apart;
    this is the one the published 0.4728 / 0.1914 were computed with.
    """
    return torch.linalg.norm(fan - gt[:, None], dim=-1).mean(-1)


def rung_block(de_sub: torch.Tensor, score_sub: torch.Tensor, eid, ci) -> dict:
    """Everything one rung reports. ``de_sub``/``score_sub`` are [B, N_rung]."""
    idx = score_sub.argmax(1)
    de_sel = de_sub.gather(1, idx[:, None]).squeeze(1)
    de_or = de_sub.min(1).values
    gap = de_sel - de_or
    two_x = (de_sel > 2 * de_or).double()
    hit = (idx == de_sub.argmin(1)).double()
    return {
        "n": int(de_sub.shape[1]),
        "selected_ade_0_2s": ci.episode_cluster_bootstrap(
            de_sel.numpy().astype(np.float64), eid, n_boot=N_BOOT),
        "oracle_in_fan_ade_0_2s": ci.episode_cluster_bootstrap(
            de_or.numpy().astype(np.float64), eid, n_boot=N_BOOT),
        "sel_gap": ci.episode_cluster_bootstrap(
            gap.numpy().astype(np.float64), eid, n_boot=N_BOOT),
        "frac_sel_2x_worse": ci.episode_cluster_bootstrap(
            two_x.numpy(), eid, n_boot=N_BOOT),
        "rank_acc": ci.episode_cluster_bootstrap(hit.numpy(), eid,
                                                 n_boot=N_BOOT),
        "_sel": de_sel, "_or": de_or, "_idx": idx,
    }


def three_sided(pair: dict) -> str:
    """better / worse / not separated — prereg §4. NEVER two-sided."""
    if not pair["separated"]:
        return "not separated"
    return "better" if pair["delta"] < 0 else "worse"


def saturation(rungs: dict, key: str, n_max: int, ci, eid, tol: float) -> dict:
    """N* = smallest N with paired-vs-Nmax NOT separated AND |delta| <= tol."""
    ref = rungs[n_max]["_sel" if key == "selected" else "_or"]
    rows, star = [], None
    for n in sorted(rungs):
        cur = rungs[n]["_sel" if key == "selected" else "_or"]
        p = ci.paired_episode_cluster_bootstrap(
            cur.numpy().astype(np.float64), ref.numpy().astype(np.float64),
            eid, n_boot=N_BOOT)
        ok = (not p["separated"]) and abs(p["delta"]) <= tol
        rows.append({"n": n, "paired_vs_Nmax": p, "verdict": three_sided(p),
                     "saturated": bool(ok)})
        if ok and star is None:
            star = n
    return {"metric": key, "N_star": star, "tol_m": tol, "n_max": n_max,
            "rungs": rows,
            "note": ("N_star None = never satisfied on this ladder, i.e. the "
                     "metric is still moving at the widest fan we have (O4)")}


# --------------------------------------------------------------------------- #
# subset rules                                                                 #
# --------------------------------------------------------------------------- #

def prefix_idx(b: int, n: int, n_max: int) -> torch.Tensor:
    return torch.arange(n).expand(b, n).contiguous()


def stride_idx(b: int, n: int, n_max: int) -> torch.Tensor:
    return torch.linspace(0, n_max - 1, n).round().long().expand(b, n).contiguous()


def random_idx(b: int, n: int, n_max: int, seed: int) -> torch.Tensor:
    """Uniform WITHOUT replacement, independently per window."""
    g = torch.Generator().manual_seed(seed)
    return torch.argsort(torch.rand(b, n_max, generator=g), dim=1)[:, :n]


def band_prefix_idx(keep: torch.Tensor, n: int) -> torch.Tensor:
    """First ``n`` candidates in FPS order that SURVIVE ``keep`` [B, N_max].

    The realisable inference policy: a fixed budget of n decodes, spent on the
    earliest surviving anchors, topped up in FPS order when fewer than n
    survive. Implemented by a stable sort on (not keep) so FPS order is
    preserved within each group.
    """
    order = torch.argsort((~keep).long(), dim=1, stable=True)
    return order[:, :n].contiguous()


# --------------------------------------------------------------------------- #
# four families                                                                #
# --------------------------------------------------------------------------- #

def families_at_rung(d: dict, fan_sub: torch.Tensor, idx: torch.Tensor,
                     four_families) -> dict:
    b = torch.arange(fan_sub.shape[0])
    win = {"pred": fan_sub[b, idx], "gt": d["gt"],
           "wp_steps": list(d["wp_steps"]), "dt_s": 0.1}
    fam = four_families.all_families(win)
    fam["_strategic_reason"] = (
        "STRATEGIC UNAVAILABLE from a fan bank, n = "
        f"{int(fan_sub.shape[0])} windows / 0 with a strategic label: "
        "refc_rerank.dump stores no route/goal label and decoded with "
        "nav_mode='follow_constant', so the route input was never exercised. "
        "A WORK ITEM, not a pass.")
    fam["_tactical_reason"] = (
        "TACTICAL: the goal/anchor-SELECTION half (rank_acc, sel_gap, "
        "frac_sel_2x_worse) IS measured at every rung and is the half fan "
        "width acts on. The manoeuvre-DECISION half needs decoded manoeuvre "
        "logits, which no fan bank stores.")
    return fam


def _clean(o):
    if isinstance(o, dict):
        # int rung keys are legitimate; only the private "_..." STRING keys are
        # dropped (they carry per-window tensors that must not reach the JSON).
        return {k: _clean(v) for k, v in o.items()
                if not (isinstance(k, str) and k.startswith("_")
                        and k not in ("_dt_s", "_dt_provenance",
                                      "_strategic_reason", "_tactical_reason"))}
    if isinstance(o, (list, tuple)):
        return [_clean(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size < 64 else f"<ndarray {o.shape}>"
    if isinstance(o, torch.Tensor):
        return o.tolist() if o.numel() < 64 else f"<tensor {tuple(o.shape)}>"
    return o


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #

def run(bank: str, out_dir: str, *, arm: str | None = None,
        anchors_path: str | None = None, repo: Path | None = None) -> dict:
    t0 = time.time()
    repo = repo or Path(__file__).resolve().parents[6]
    _add_paths(repo)
    from taniteval import ci, four_families                      # noqa: PLC0415

    bp = Path(bank)
    d = torch.load(bp, map_location="cpu", weights_only=False)
    arm = arm or bp.stem.replace("fan_", "")
    eid = list(d["eid"])
    fan, logits, gt = d["fan"], d["logits"], d["gt"]
    B, N_MAX = fan.shape[0], fan.shape[1]
    de_all = candidate_ade(fan, gt)
    ladder = [n for n in THRESHOLDS["ladder"] if n <= N_MAX]

    res: dict = {
        "what": "FAN WIDTH — the two saturation curves and the latency they buy",
        "prereg": prereg_provenance(repo),
        "arm": arm,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "estimator": {
            "point_and_interval": "episode_cluster_bootstrap",
            "paired": "paired_episode_cluster_bootstrap",
            "resampling_unit": "episode", "n_boot": N_BOOT,
            "⛔": "overlapping_holdout_se is NEVER called",
        },
        "bank": {"path": str(bp), "sha256": _sha256(bp),
                 "n_windows": B, "n_anchors": N_MAX,
                 "n_episodes": len(set(eid)),
                 "wp_steps": list(d["wp_steps"]),
                 "decode_ckpt": d.get("ckpt"),
                 "decode_ckpt_step": d.get("ckpt_step"),
                 "diffusion_steps": d.get("steps"),
                 "nav_mode": d.get("nav_mode", "follow_constant (implied)")},
        "ladder": ladder,
        "controls": {}, "curves": {}, "P2": {},
    }

    # ---- P1 rungs, PREFIX (primary) --------------------------------------- #
    rungs = {}
    for n in ladder:
        rungs[n] = rung_block(de_all[:, :n], logits[:, :n], eid, ci)
    res["curves"]["prefix"] = {n: _clean(v) for n, v in rungs.items()}

    # ---- C-monotone-oracle: nested prefixes => oracle must not increase ---- #
    mono, prev = [], None
    for n in ladder:
        cur = float(rungs[n]["_or"].mean())
        bad = prev is not None and cur > prev + 1e-9
        mono.append({"n": n, "oracle_mean": round(cur, 6), "increase": bool(bad)})
        prev = cur
    fired = any(r["increase"] for r in mono)
    res["controls"]["C-monotone-oracle"] = {
        "control": "C-monotone-oracle", "rows": mono,
        "status": ("FAIL — STOP, the prefix sets are not nested or the pipeline "
                   "is broken" if fired else "PASS"),
        "why_it_can_fire": ("oracle_in_fan is a min over a GROWING nested set, "
                            "so an increase is impossible unless the code is "
                            "wrong. This is a check on my own pipeline.")}
    if fired:
        raise SystemExit("C-monotone-oracle FAILED — see " + str(out_dir))

    # ---- C-full-rung ------------------------------------------------------- #
    pub = PUBLISHED.get(arm, {})
    got_sel = float(rungs[N_MAX]["_sel"].mean())
    got_or = float(rungs[N_MAX]["_or"].mean())
    dev_s = None if not pub else abs(got_sel - pub["selected_ade2s"])
    dev_o = None if not pub else abs(got_or - pub["oracle_ade2s"])
    tol = THRESHOLDS["full_rung_tol_m"]
    if not pub:
        st = "NOT-COMPARABLE — no published value for this arm"
    elif B != 881:
        st = f"NOT-COMPARABLE — n = {B} != the canonical 881"
    elif dev_s <= tol and dev_o <= tol:
        st = "PASS"
    else:
        st = "FAIL — the bank is not the published decode; STOP"
    res["controls"]["C-full-rung"] = {
        "control": "C-full-rung", "n_windows": B, "canonical_n_windows": 881,
        "recomputed_selected": round(got_sel, 6),
        "published_selected": pub.get("selected_ade2s"),
        "abs_dev_selected": None if dev_s is None else round(dev_s, 8),
        "recomputed_oracle": round(got_or, 6),
        "published_oracle": pub.get("oracle_ade2s"),
        "abs_dev_oracle": None if dev_o is None else round(dev_o, 8),
        "tolerance": tol,
        "direction_predicate": ("BETTER than published is as much a FAIL as "
                                "worse — a two-sided |dev| test, not one-sided"),
        "red_flag": ("beating published by > "
                     f"{THRESHOLDS['red_flag_m']} m => leak audit, do not "
                     "publish"),
        "red_flag_fired": bool(pub and (pub["selected_ade2s"] - got_sel)
                               > THRESHOLDS["red_flag_m"]),
        "status": st}
    if st.startswith("FAIL"):
        raise SystemExit("C-full-rung FAILED")

    # ---- SATURATION — the two curves, reported TOGETHER -------------------- #
    res["saturation"] = {
        "selected_ade": saturation(rungs, "selected", N_MAX, ci, eid,
                                   THRESHOLDS["sat_tol_m"]),
        "oracle_in_fan": saturation(rungs, "oracle", N_MAX, ci, eid,
                                    THRESHOLDS["sat_tol_m"]),
        "⛔": ("reporting one of these without the other is out of contract — "
               "the GAP between them is the question"),
    }

    # ---- C-random-subset (24 seeds) + C-stride ----------------------------- #
    rnd_rows, wins = [], 0
    for n in ladder:
        sels, ors = [], []
        for s in range(THRESHOLDS["random_seeds"]):
            ix = random_idx(B, n, N_MAX, 20260804 + s)
            de_s, lo_s = de_all.gather(1, ix), logits.gather(1, ix)
            j = lo_s.argmax(1)
            sels.append(de_s.gather(1, j[:, None]).squeeze(1))
            ors.append(de_s.min(1).values)
        sel_m = torch.stack(sels).mean(0)
        or_m = torch.stack(ors).mean(0)
        p_or = ci.paired_episode_cluster_bootstrap(
            rungs[n]["_or"].numpy().astype(np.float64),
            or_m.numpy().astype(np.float64), eid, n_boot=N_BOOT)
        p_sel = ci.paired_episode_cluster_bootstrap(
            rungs[n]["_sel"].numpy().astype(np.float64),
            sel_m.numpy().astype(np.float64), eid, n_boot=N_BOOT)
        prefix_wins_oracle = bool(p_or["delta"] <= 0)
        wins += int(prefix_wins_oracle)
        rnd_rows.append({
            "n": n,
            "random_mean_oracle": round(float(or_m.mean()), 6),
            "prefix_oracle": round(float(rungs[n]["_or"].mean()), 6),
            "paired_prefix_minus_random_oracle": p_or,
            "verdict_oracle": three_sided(p_or),
            "random_mean_selected": round(float(sel_m.mean()), 6),
            "prefix_selected": round(float(rungs[n]["_sel"].mean()), 6),
            "paired_prefix_minus_random_selected": p_sel,
            "verdict_selected": three_sided(p_sel),
            "prefix_wins_oracle": prefix_wins_oracle})
    need = THRESHOLDS["random_prefix_min_rungs_won"]
    res["controls"]["C-random-subset"] = {
        "control": "C-random-subset", "seeds": THRESHOLDS["random_seeds"],
        "rows": rnd_rows,
        "direction_predicate": (f"FPS prefix must be <= random on oracle at >= "
                                f"{need} of {len(ladder)} rungs"),
        "rungs_prefix_won": wins, "rungs_total": len(ladder),
        "status": ("PASS" if wins >= min(need, len(ladder)) else
                   "FIRED — the 'principled subset' premise is WITHDRAWN; the "
                   "curves must be re-read as 'any subset of this size does as "
                   "well', which is a weaker claim")}

    str_rows = []
    for n in ladder:
        ix = stride_idx(B, n, N_MAX)
        de_s, lo_s = de_all.gather(1, ix), logits.gather(1, ix)
        j = lo_s.argmax(1)
        str_rows.append({
            "n": n,
            "stride_selected": round(float(de_s.gather(1, j[:, None]).mean()), 6),
            "stride_oracle": round(float(de_s.min(1).values.mean()), 6),
            "prefix_selected": round(float(rungs[n]["_sel"].mean()), 6),
            "prefix_oracle": round(float(rungs[n]["_or"].mean()), 6)})
    res["controls"]["C-stride"] = {
        "control": "C-stride", "rows": str_rows,
        "status": "REPORTED — no verdict rides on it (prereg §5)"}

    # ---- P2 — reallocating INTO the reachable set -------------------------- #
    from tanitad.refs import refc_select as sl                    # noqa: PLC0415
    a_max, hz = THRESHOLDS["accel_max"], THRESHOLDS["horizon_s"]
    v0 = d["v0"].to(fan.dtype)
    keep_dec = sl.reachability_mask(fan, v0, accel_max=a_max, horizon_s=hz)

    keep_anc = None
    if anchors_path:
        V = torch.load(anchors_path, map_location="cpu", weights_only=False)
        anc = V[arm] if arm in V else V.get("anchors", None)
        if anc is not None:
            anc = anc[:N_MAX].to(fan.dtype)
            keep_anc = sl.reachability_mask(
                anc[None].expand(B, -1, -1, -1).contiguous(), v0,
                accel_max=a_max, horizon_s=hz)

    # C-band-fidelity — does the anchor-level band predict the decoded one?
    if keep_anc is not None:
        a_, dd = keep_anc.reshape(-1), keep_dec.reshape(-1)
        tt = int((a_ & dd).sum()); tf = int((a_ & ~dd).sum())
        ft = int((~a_ & dd).sum()); ff = int((~a_ & ~dd).sum())
        tot = a_.numel()
        admit_wrongly = tf / tot
        res["controls"]["C-band-fidelity"] = {
            "control": "C-band-fidelity",
            "confusion_over_window_x_candidate_pairs": {
                "anchor_keep_AND_decoded_keep": tt,
                "anchor_keep_BUT_decoded_reject": tf,
                "anchor_reject_BUT_decoded_keep": ft,
                "anchor_reject_AND_decoded_reject": ff, "total": tot},
            "agreement": round((tt + ff) / tot, 6),
            "frac_admitted_wrongly": round(admit_wrongly, 6),
            "anchor_band_keep_frac": round(float(keep_anc.double().mean()), 6),
            "decoded_band_keep_frac": round(float(keep_dec.double().mean()), 6),
            "per_window_survivors_anchor": {
                "median": float(keep_anc.sum(1).double().median()),
                "min": int(keep_anc.sum(1).min()),
                "max": int(keep_anc.sum(1).max())},
            "per_window_survivors_decoded": {
                "median": float(keep_dec.sum(1).double().median()),
                "min": int(keep_dec.sum(1).min()),
                "max": int(keep_dec.sum(1).max())},
            "trigger": (f"> {THRESHOLDS['band_fidelity_max_disagree']:.0%} of "
                        "pairs admitted by the anchor band but rejected by the "
                        "decoded band => P2b's saving is illusory"),
            "status": ("FIRED — P2b's compute saving is illusory, do not quote "
                       "its ADE as a shippable number"
                       if admit_wrongly > THRESHOLDS["band_fidelity_max_disagree"]
                       else "PASS")}

    p2 = {"band": (f"v_mean in [max(0, v0 - {a_max * hz}), v0 + {a_max * hz}] "
                   f"m/s — flagship_v15's OWN clamp via refc_select, NOT tuned"),
          "⛔": ("the clamp's ADE-inertness at full width is ALREADY MEASURED "
                 "(paired delta exactly 0.0) and is NOT re-reported here as a "
                 "finding. The registered question is what its freed budget "
                 "buys."),
          "P2a_information_decoded_band": [], "P2b_compute_anchor_band": []}
    for n in ladder:
        ix = band_prefix_idx(keep_dec, n)
        de_s, lo_s = de_all.gather(1, ix), logits.gather(1, ix)
        j = lo_s.argmax(1)
        sel = de_s.gather(1, j[:, None]).squeeze(1)
        pr = ci.paired_episode_cluster_bootstrap(
            sel.numpy().astype(np.float64),
            rungs[n]["_sel"].numpy().astype(np.float64), eid, n_boot=N_BOOT)
        p2["P2a_information_decoded_band"].append({
            "n": n, "reach_prefix_selected": round(float(sel.mean()), 6),
            "plain_prefix_selected": round(float(rungs[n]["_sel"].mean()), 6),
            "reach_prefix_oracle": round(float(de_s.min(1).values.mean()), 6),
            "n_survivors_used_median": float(
                torch.minimum(keep_dec.sum(1), torch.tensor(n)).double().median()),
            "paired_reach_minus_plain": pr, "verdict": three_sided(pr)})
        if keep_anc is not None:
            ixa = band_prefix_idx(keep_anc, n)
            de_a, lo_a = de_all.gather(1, ixa), logits.gather(1, ixa)
            ja = lo_a.argmax(1)
            sela = de_a.gather(1, ja[:, None]).squeeze(1)
            pa = ci.paired_episode_cluster_bootstrap(
                sela.numpy().astype(np.float64),
                rungs[n]["_sel"].numpy().astype(np.float64), eid, n_boot=N_BOOT)
            p2["P2b_compute_anchor_band"].append({
                "n": n, "anchor_band_selected": round(float(sela.mean()), 6),
                "plain_prefix_selected": round(float(rungs[n]["_sel"].mean()), 6),
                "anchor_band_oracle": round(float(de_a.min(1).values.mean()), 6),
                "paired_anchorband_minus_plain": pa, "verdict": three_sided(pa)})
    res["P2"] = p2

    # ---- ρ hygiene: restricted to REACHABLE SURVIVORS, beside a selection --- #
    # ⛔ rho over the FULL candidate axis is not a proxy for a selector.
    try:
        from scipy.stats import spearmanr                         # noqa: PLC0415
        full, surv = [], []
        lg, da = logits.numpy(), de_all.numpy()
        km = keep_dec.numpy()
        for i in range(B):
            full.append(spearmanr(lg[i], -da[i]).statistic)
            m = km[i]
            surv.append(spearmanr(lg[i][m], -da[i][m]).statistic
                        if m.sum() >= 3 else np.nan)
        full = np.array(full, dtype=np.float64)
        surv = np.array(surv, dtype=np.float64)
        ok = ~np.isnan(surv)
        res["rho_hygiene"] = {
            "rho_shipped_logits_vs_minus_ADE_FULL_axis":
                ci.episode_cluster_bootstrap(full, eid, n_boot=N_BOOT),
            "rho_RESTRICTED_to_reachable_survivors":
                ci.episode_cluster_bootstrap(
                    surv[ok], [e for e, k in zip(eid, ok) if k], n_boot=N_BOOT),
            "n_windows_with_ge3_survivors": int(ok.sum()),
            "selection_ade_beside_it": res["curves"]["prefix"][N_MAX][
                "selected_ade_0_2s"],
            "⛔": ("rho over the full candidate axis is NOT a proxy for a "
                   "selector; restricted to survivors one rho in this "
                   "programme went 0.6657 -> 0.3008 and another crossed zero. "
                   "Both are printed beside the selection ADE, always."),
        }
    except Exception as exc:                                      # pragma: no cover
        res["rho_hygiene"] = {"status": f"UNAVAILABLE — {type(exc).__name__}: {exc}"}

    # ---- FOUR FAMILIES at every rung, per family, never pooled -------------- #
    dt, dt_prov = four_families.infer_dt({"wp_steps": list(d["wp_steps"]),
                                          "dt_s": 0.1})
    fam = {"_dt_s": dt, "_dt_provenance": dt_prov, "per_rung": {}}
    for n in ladder:
        f = families_at_rung(d, fan[:, :n], rungs[n]["_idx"], four_families)
        f["tactical_goal_anchor_selection"] = {
            k: res["curves"]["prefix"][n][k]
            for k in ("rank_acc", "sel_gap", "frac_sel_2x_worse")}
        fam["per_rung"][n] = _clean(f)
    res["four_families"] = fam

    res["wall_s"] = round(time.time() - t0, 1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"fan_width_{arm}.json"
    dest.write_text(json.dumps(_clean(res), indent=1, ensure_ascii=False),
                    encoding="utf-8")
    print(f"[fanwidth] {arm}: {B} windows x {N_MAX} anchors -> {dest} "
          f"({res['wall_s']}s)", flush=True)
    print(f"[fanwidth]   N*(selected) = {res['saturation']['selected_ade']['N_star']}"
          f"   N*(oracle) = {res['saturation']['oracle_in_fan']['N_star']}",
          flush=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--anchors", default=None,
                    help="refc_anchor_vocab.pt — unlocks P2b + C-band-fidelity")
    ap.add_argument("--repo", default=None)
    a = ap.parse_args(argv)
    run(a.bank, a.out, arm=a.arm, anchors_path=a.anchors,
        repo=Path(a.repo) if a.repo else None)
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
