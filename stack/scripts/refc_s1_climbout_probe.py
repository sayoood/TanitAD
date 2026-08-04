"""E-S1-0 — IS THERE A TARGET FOR S1'S CLIMB-OUT? 0 GPU, banked fans only.

Pre-registration: ``TanitAD Research Hub/Architecture & Inference/Implementation/
incoming/2026-08-03-s1-climbout/PREREG_S1_CLIMBOUT.md`` — staged BEFORE any statistic
here was computed; the runner re-verifies the staged blob id on every arm.

WHAT THIS DECIDES, AND WHAT IT DELIBERATELY DOES NOT
============================================================================
E-SEL-0 MEASURED that REF-C's discarded refined confidence ranks 0.8372 m (base) /
0.9187 m (XL) WORSE than the shipped t=0 score — separated — while scoring 8.7x /
16.6x chance. The parent prereg's POST-HOC fifth branch reads that as "off-
distribution, not uninformative => S1 must CLIMB OUT". This probe asks the prior
question:

    before spending a GPU-day supervising that readout, is there ANY supervised
    ranker over REF-C's own DEPLOYABLE per-candidate information that beats the
    incumbent selector OUT-OF-EPISODE?

⛔ IT IS NOT A TEST OF THE SUPERVISED CLIMB-OUT, and §2 of the prereg registers
that in advance. A ranker built as any strictly-monotone function of ONE banked
scalar has the IDENTICAL argmax, so refitting on `refined_logits` alone reproduces
1.3100 exactly and refitting on `logits` alone reproduces 0.4728 exactly. Both are
CONTROLS here, not treatments. Bounding the real climb-out needs the refined pass's
QUERY FEATURES (a GPU inference dump, still 0 training); measuring it needs a
retrain.

⚠️ THE TRAP THIS PROBE IS BUILT AROUND. S3_DEPLOYABLE §3 MEASURED that rho over
the full candidate axis is NOT a proxy for a selector: rho = 0.6657 selects at
6.49 m against a shipped 0.4728, and rho = 0.9951 (`cv`) selects at 0.8149.
Mechanism: 72-74 % of the fan is outside the bounded-acceleration band and is never
selected, so a statistic over the whole axis is dominated by candidates no selector
can pick. ⇒ EVERYTHING here is restricted to the S2-reachable survivor set FIRST,
and the headline is SELECTION ADE, never rho.

THE PROTOCOL
============================================================================
A per-candidate scorer ``s_i = w . phi_i`` fit by LISTWISE SOFTMAX CROSS-ENTROPY
against the oracle index, restricted to the reachable survivors, evaluated
LEAVE-ONE-EPISODE-OUT: no window is ever scored by a model that saw its own
episode. Headline = selection ADE@2s of ``argmax_i s_i`` over the survivors, PAIRED
against the shipped selector on the same windows (episode-cluster bootstrap,
n_boot = 2000, unit = episode). ⛔ `overlapping_holdout_se` is never called.

Run (dev box or any CPU host — no checkpoint, no cache, no pod):
    OMP_NUM_THREADS=6 python scripts/refc_s1_climbout_probe.py \
        --bank ".../2026-08-03-esel-verdict/raw/fan_refined_refc-base-30k.pt" \
        --arm refc-base-30k --out ".../2026-08-03-s1-climbout/raw"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "6")

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
for _p in (_REPO / "taniteval", _REPO / "stack", _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from taniteval import ci                                          # noqa: E402
import refc_sel_probe as P                                        # noqa: E402

PREREG = ("TanitAD Research Hub/Architecture & Inference/Implementation/incoming/"
          "2026-08-03-s1-climbout/PREREG_S1_CLIMBOUT.md")

#: A TRANSCRIPTION of prereg §4. Every value is quoted from the staged blob whose
#: id this module prints; none may be edited by a run.
PREREG_THRESHOLDS = {
    "estimator": ("paired_episode_cluster_bootstrap / episode_cluster_bootstrap, "
                  "unit = episode, n_boot = 2000; overlapping_holdout_se NEVER"),
    "material_m": 0.02,          # §4, inherited from the parent's free_win_m
    "red_flag_m": 0.10,          # §4.5 stop-and-audit
    "leak_gap_frac": 0.50,       # §4.1 C-leak: in-sample - LOEO < 50 % of effect
    "S1_TARGET_EXISTS": ("ADE(B-both) - ADE(shipped) separated AND <= -0.02 m AND "
                         "LATERAL not separated worse AND C-leak gap < 50 %"),
    "S1_TARGET_MARGINAL": "separated and better but > -0.02 m",
    "S1_NOT_SEPARATED": "CI includes 0 => S1 IS DEAD AFTER ALL (§4.4)",
    "S1_ADVERSE": "separated AND worse => S1 IS DEAD AFTER ALL, more strongly",
    "LON_LEVER_LIVE": ("ADE(C-lon) - ADE(shipped) separated AND <= -0.02 m AND "
                       "the LONGITUDINAL family separated better"),
    "LON_LEVER_NULL": "CI includes 0",
    "LON_LEVER_ADVERSE": "separated AND worse",
    "S1b_HYGIENE": "|dADE| < 0.02 m or not separated (PREDICTED)",
    "registered_prediction": ("B-both S1-NOT-SEPARATED; C-lon LON-LEVER-NULL at "
                              "~55 % confidence; S1b HYGIENE; overall §4.4 fires "
                              "and S1 is reported DEAD"),
    "direction_predicate": ("EVERY trigger reads 'separated AND the delta favours "
                            "the treatment' — a trigger satisfied literally while "
                            "its controls beat the score is what happened to S3"),
    "vacuous_controls_declared": ["C-shuffled"],
}

ACCEL_MAX, HORIZON_S = 2.5, 2.0
N_BOOT = ci.DEFAULT_N_BOOT
FIT_ITERS, FIT_LR = 400, 0.25
RNG_SEED = 0


# =========================================================================== #
# provenance                                                                  #
# =========================================================================== #

def prereg_provenance() -> dict:
    staged = P._git("ls-files", "-s", "--", PREREG).split()
    worktree = P._git("hash-object", PREREG)
    blob = staged[1] if len(staged) > 1 else "<not staged>"
    return {
        "path": PREREG,
        "staged_blob": blob,
        "worktree_blob": worktree,
        "thresholds_unmoved_since_staging": bool(blob and blob == worktree),
        "how_to_verify": (f"git ls-files -s -- '{PREREG}' && "
                          f"git hash-object '{PREREG}'   # must match"),
        "thresholds": PREREG_THRESHOLDS,
        "parent": P.prereg_provenance(),
        # ⚠️ The repo mount (Google Drive) was in a whole-mount READ-FAILURE
        # state when this ran, so the prereg could not be `git add`ed BEFORE
        # measuring and `staged_blob` reads <not staged> on this host. The
        # falsifiable object is therefore the git BLOB ID computed from the exact
        # bytes the run used — a deterministic function of content, so once the
        # file is staged `git ls-files -s` MUST print the same id. Pinned in a
        # separate artifact written BEFORE the first statistic.
        "pin": _pin(),
    }


def _pin() -> dict:
    for c in (Path.cwd() / "raw" / "prereg_pin.json",
              Path.home() / "s1_climbout" / "raw" / "prereg_pin.json"):
        if c.exists():
            return json.loads(c.read_text())
    return {"status": "NO PIN FILE FOUND — provenance is UNVERIFIED"}


# =========================================================================== #
# reachability — the support EVERYTHING is restricted to, computed FIRST        #
# =========================================================================== #

def survivor_mask(d: dict) -> tuple[torch.Tensor, dict]:
    """[B, N] bool — S2's band, with the decoder's OWN empty-set fallback.

    THE re-export of ``flagship_v15.reachability_mask`` through ``refc_select``,
    i.e. the same function object the 72.08 % / oracle-survives-100 % / paired-
    delta-exactly-0.0 measurement was made with. A re-implementation here would
    silently detach the restriction from the number that justifies it.
    """
    from tanitad.refs import refc_select as sl
    m = sl.reachability_mask(d["fan"], d["v0"].to(d["fan"].dtype),
                             accel_max=ACCEL_MAX, horizon_s=HORIZON_S)
    dead = ~m.any(1)
    keep = m | dead[:, None]          # a window with no survivor keeps its fan
    tele = {
        "frac_candidates_removed": round(float(1 - m.double().mean()), 4),
        "frac_windows_with_empty_survivor_set": round(float(dead.double().mean()), 4),
        "search_space_cheaper_x": round(float(1 / max(m.double().mean(), 1e-9)), 4),
        "accel_max": ACCEL_MAX, "horizon_s": HORIZON_S,
    }
    return keep, tele


# =========================================================================== #
# deployable per-candidate features — NOTHING here may see the future          #
# =========================================================================== #

def build_features(d: dict, keep: torch.Tensor) -> dict:
    """Named [B, N] feature planes. EVERY one is computable at inference.

    ⛔ ``cons_score`` / ``cons_oracle`` are EXCLUDED on purpose: they read
    ``z_{t+5}``, the future frame. Including them would be the C6 / REF-A-I-JEPA
    class — a measurement-time input the deployed path does not have.
    """
    fan, v0 = d["fan"].double(), d["v0"].double()          # [B,N,S,2], [B]
    cv = d["cv"].double()                                  # [B,S,2]
    end = fan[:, :, -1]                                    # [B,N,2]
    along = end[..., 0]                                    # terminal along-track (m)
    cross = end[..., 1]                                    # terminal cross-track (m)
    v_impl = along / HORIZON_S                             # implied mean speed
    dv = v_impl - v0[:, None]                              # implied speed change
    band = ACCEL_MAX * HORIZON_S
    outside = (dv.abs() - band).clamp_min(0.0)             # metres/s outside S2's band
    d_cv = -torch.linalg.norm(fan - cv[:, None], dim=-1).mean(-1)   # [B,N]
    f = {
        "logits": d["logits"].double(),
        "refined_logits": d["refined_logits"].double(),
        # --- LONGITUDINAL geometry (the family the gap says must move) -------
        "along_end_m": along,
        "dv_mps": dv,
        "abs_dv_mps": dv.abs(),
        "outside_band_mps": outside,
        # --- controls / diagnostics ------------------------------------------
        "neg_cv_dist_m": d_cv,
        "abs_cross_end_m": cross.abs(),
    }
    return {k: _zrow(v, keep) for k, v in f.items()}


def _zrow(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Per-window z-score over the SURVIVORS only.

    Per-window and per-feature, so features are comparable in the fit. For a
    SINGLE feature this is a strictly monotone within-window transform, so the
    degenerate one-feature controls (§3.2 A-shipped / A-refined) still reproduce
    their incumbent argmax EXACTLY — which is what makes C-monotone a control
    that can fire rather than a formality.
    """
    m = keep.double()
    n = m.sum(1, keepdim=True).clamp_min(1.0)
    mu = (x * m).sum(1, keepdim=True) / n
    var = (((x - mu) ** 2) * m).sum(1, keepdim=True) / n
    return torch.where(keep, (x - mu) / var.sqrt().clamp_min(1e-9),
                       torch.zeros_like(x))


FEATURE_SETS = {
    "A-shipped":   ["logits"],
    "A-refined":   ["refined_logits"],
    "B-both":      ["logits", "refined_logits"],
    "C-lon":       ["along_end_m", "dv_mps", "abs_dv_mps", "outside_band_mps"],
    "D-lon+scores": ["logits", "refined_logits", "along_end_m", "dv_mps",
                     "abs_dv_mps", "outside_band_mps"],
    "E-cv":        ["neg_cv_dist_m"],
}
#: §3.2 — declared in the code, not only in prose, so a reader of the JSON knows
#: which rows are controls that CANNOT move and which are treatments.
DEGENERATE = {"A-shipped", "A-refined", "E-cv"}


# =========================================================================== #
# the listwise fit                                                             #
# =========================================================================== #

def _stack(feats: dict, names: list[str], keep: torch.Tensor) -> torch.Tensor:
    return torch.stack([feats[n] for n in names], dim=-1)          # [B,N,F]


def _fit(phi: torch.Tensor, tgt: torch.Tensor, keep: torch.Tensor,
         iters: int = FIT_ITERS, lr: float = FIT_LR) -> torch.Tensor:
    """Listwise softmax CE against the oracle index, over the survivors only.

    No bias term: a per-window constant cancels in a softmax over candidates, so
    a bias would be an unidentifiable parameter, not capacity.
    """
    w = torch.zeros(phi.shape[-1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    neg = torch.finfo(torch.float64).min / 4          # finite, so no -inf arithmetic
    for _ in range(iters):
        opt.zero_grad()
        s = (phi * w).sum(-1)
        s = torch.where(keep, s, torch.full_like(s, neg))
        loss = torch.nn.functional.cross_entropy(s, tgt)
        loss.backward()
        opt.step()
    return w.detach()


def _fit_softade(phi: torch.Tensor, de: torch.Tensor, keep: torch.Tensor,
                 iters: int = FIT_ITERS, lr: float = FIT_LR) -> torch.Tensor:
    """Minimise the EXPECTED ADE under the score's own softmax, over survivors.

    ⭐ WHY THIS EXISTS (POST-HOC — it decides no registered branch). The
    registered fit is a LISTWISE SOFTMAX CE against the oracle INDEX, which is
    exactly the objective `refc_train.loss_rcls` uses. If a ranker fit that way
    is separated WORSE than the incumbent even when the incumbent's own score is
    one of its features, the natural next question is whether the failure is
    "no information" or "wrong objective" — and those license opposite decisions.
    This is the discriminating test: the same features, the same folds, the same
    survivors, optimising the quantity actually cared about (a soft argmax ADE)
    instead of the log-likelihood of one winner among 128 near-duplicates.

    A single-feature fit is STILL a monotone transform, so the degenerate
    controls keep reproducing their incumbents exactly under this objective too.
    """
    w = torch.zeros(phi.shape[-1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    neg = torch.finfo(torch.float64).min / 4
    d0 = de.double().masked_fill(~keep, 0.0)
    for _ in range(iters):
        opt.zero_grad()
        s_ = (phi * w).sum(-1)
        s_ = torch.where(keep, s_, torch.full_like(s_, neg))
        loss = (torch.softmax(s_, dim=-1) * d0).sum(-1).mean()
        loss.backward()
        opt.step()
    return w.detach()


def loeo_scores(feats: dict, names: list[str], keep: torch.Tensor,
                de_all: torch.Tensor, eid, objective: str = "ce"
                ) -> tuple[torch.Tensor, dict]:
    """LEAVE-ONE-EPISODE-OUT scores. No window is scored by a model that saw its
    own episode — the property the whole read depends on, asserted per fold."""
    phi = _stack(feats, names, keep)
    tgt = de_all.double().masked_fill(~keep, float("inf")).argmin(1)
    uniq, by_ep = ci.episode_index(eid)
    n = phi.shape[0]
    out = torch.zeros(n, phi.shape[1], dtype=torch.float64)
    ws = []
    for e in uniq:
        te = torch.zeros(n, dtype=torch.bool)
        te[torch.as_tensor(by_ep[e], dtype=torch.long)] = True
        tr = ~te
        assert bool(te.any()) and bool(tr.any()), f"degenerate fold for episode {e}"
        assert not bool((te & tr).any())          # folds are DISJOINT, asserted
        w = (_fit(phi[tr], tgt[tr], keep[tr]) if objective == "ce"
             else _fit_softade(phi[tr], de_all[tr], keep[tr]))
        out[te] = (phi[te] * w).sum(-1)
        ws.append(w.numpy())
    w_full = (_fit(phi, tgt, keep) if objective == "ce"
              else _fit_softade(phi, de_all, keep))                    # for the C-leak in-sample row
    return out, {"objective": objective, "n_folds": len(uniq), "w_mean": np.mean(ws, 0).tolist(),
                 "w_std": np.std(ws, 0).tolist(), "w_insample": w_full.tolist(),
                 "features": names, "_w_full": w_full, "_phi": phi, "_tgt": tgt}


def argmax_over_survivors(score: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    neg = torch.finfo(torch.float64).min / 4
    return torch.where(keep, score.double(), torch.full_like(score.double(), neg)) \
        .argmax(1)


# =========================================================================== #
# rho — SECONDARY, and on the reachable subset only                            #
# =========================================================================== #

def rho_reachable(score: torch.Tensor, de_all: torch.Tensor,
                  keep: torch.Tensor, eid) -> dict:
    """Per-window Spearman on the SURVIVORS, then an episode-cluster bootstrap.

    ⚠️ Reported as a DIAGNOSTIC, never as a sizing statistic. S3_DEPLOYABLE §3
    is the measurement that forbids the latter.
    """
    s, a = score.double().numpy(), (-de_all.double()).numpy()
    k = keep.numpy()
    per = np.full(s.shape[0], np.nan)
    for i in range(s.shape[0]):
        m = k[i]
        if m.sum() < 3:
            continue
        x, y = s[i][m], a[i][m]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            continue
        rx = np.argsort(np.argsort(x)).astype(np.float64)
        ry = np.argsort(np.argsort(y)).astype(np.float64)
        per[i] = np.corrcoef(rx, ry)[0, 1]
    ok = ~np.isnan(per)
    return {"n_windows": int(ok.sum()),
            **P._boot(per[ok], [e for e, o in zip(eid, ok) if o]),
            "_note": ("SECONDARY DIAGNOSTIC on the S2-reachable subset. rho over "
                      "the full candidate axis is NOT a proxy for a selector "
                      "(S3_DEPLOYABLE §3): rho 0.6657 selects at 6.49 m.")}


# =========================================================================== #
# the run                                                                      #
# =========================================================================== #

def s1b_block(d: dict, em: dict, de_all, de_or, keep, eid, shipped) -> dict:
    """§4.3 — does scoring the EMITTED fan beat scoring its predecessor?

    Both readouts come from ONE forward (`sel_score_emitted=True` returns
    `emitted_logits` and `prefinal_logits` together), so the paired contrast is
    exact. The controls are asserted against E-SEL's bank bit-for-bit FIRST: a
    changed fan would silently re-baseline every D-SEL number.
    """
    ctl = {
        "fan_bit_identical_to_esel": bool(torch.equal(d["fan"], em["fan"])),
        "gt_bit_identical": bool(torch.equal(d["gt"], em["gt"])),
        "eid_match": list(d["eid"]) == list(em["eid"]),
        "prefinal_reproduces_esel_refined": bool(
            torch.equal(d["refined_logits"], em["prefinal_logits"])),
        "emitted_differs_from_prefinal": not bool(
            torch.equal(em["emitted_logits"], em["prefinal_logits"])),
        "can_fire": True,
    }
    i_em = argmax_over_survivors(em["emitted_logits"], keep)
    i_pre = argmax_over_survivors(em["prefinal_logits"], keep)
    b_em = P.ranker_block(de_all, de_or, i_em, eid, tag="S1b-emitted")
    b_pre = P.ranker_block(de_all, de_or, i_pre, eid, tag="S1-prefinal")
    return {
        "controls": ctl,
        "emitted": {k: v for k, v in b_em.items() if not k.startswith("_")},
        "prefinal": {k: v for k, v in b_pre.items() if not k.startswith("_")},
        "paired_emitted_minus_prefinal": P._paired(
            b_em["_per_window_ade"].numpy(), b_pre["_per_window_ade"].numpy(), eid),
        "paired_emitted_minus_shipped": P._paired(
            b_em["_per_window_ade"].numpy(),
            shipped["_per_window_ade"].numpy(), eid),
        "agreement_emitted_vs_prefinal": round(
            float((i_em == i_pre).double().mean()), 4),
        "corr_emitted_prefinal": round(float(np.corrcoef(
            em["emitted_logits"].flatten().numpy(),
            em["prefinal_logits"].flatten().numpy())[0, 1]), 4),
        "_idx": i_em,
    }


def _absolute_or_reason(d, idx, eid, blk, name) -> dict:
    try:
        return P.families_block(d, idx, eid, sel_half=blk, tag=name)
    except Exception as exc:
        return {"_tag": name,
                "status": f"UNAVAILABLE on this host — {type(exc).__name__}: {exc}",
                "n": int(d["fan"].shape[0]),
                "why_it_is_not_load_bearing": (
                    "R-2026-08-03-c: absolute four-family RATES before the dt fix "
                    "are wrong by 5x-25x; cross-arm ranks and PAIRED deltas "
                    "survive. `families.*.paired_vs_shipped` IS computed and is "
                    "what prereg §6 judges on.")}


def run(bank: str, arm: str, out_dir: str, emitted_bank: str | None = None) -> dict:
    t0 = time.time()
    d = P.load_fan(bank)
    if "refined_logits" not in d:
        raise KeyError(f"{bank} carries no refined_logits — this must be an "
                       f"AUGMENTED bank (refc_sel_dump_refined.py)")
    eid = list(d["eid"])
    de_all = P.candidate_ade(d["fan"], d["gt"])                 # [B,N]
    de_or = de_all.min(1).values
    keep, reach = survivor_mask(d)
    feats = build_features(d, keep)
    pub = P.PUBLISHED[arm]
    sets = dict(FEATURE_SETS)
    em = None
    if emitted_bank:
        em = torch.load(emitted_bank, map_location="cpu", weights_only=False)
        feats["emitted_logits"] = _zrow(em["emitted_logits"].double(), keep)
        sets["A-emitted"] = ["emitted_logits"]
        sets["B-both+emitted"] = ["logits", "refined_logits", "emitted_logits"]
        sets["D-all"] = FEATURE_SETS["D-lon+scores"] + ["emitted_logits"]
        DEGENERATE.add("A-emitted")

    # ---- controls that reproduce, run FIRST ------------------------------
    controls = {
        "C-reproduce": P.control_identity(d, de_all, arm),
        "C-oracle-floor": P.control_oracle_floor(d, de_or, arm),
        "C-raster": P.control_raster(d),
    }
    shipped_idx = d["sel"]
    shipped = P.ranker_block(de_all, de_or, shipped_idx, eid, tag="shipped")
    refined_idx = argmax_over_survivors(d["refined_logits"], keep)

    # ---- the arms ---------------------------------------------------------
    arms, rows = {}, {}
    for name, names in sets.items():
        sc, meta = loeo_scores(feats, names, keep, de_all, eid)
        idx = argmax_over_survivors(sc, keep)
        blk = P.ranker_block(de_all, de_or, idx, eid, tag=name)
        ins = argmax_over_survivors((meta["_phi"] * meta["_w_full"]).sum(-1), keep)
        ins_ade = de_all.gather(1, ins[:, None]).squeeze(1)
        rows[name] = {
            "features": names,
            "degenerate_control": name in DEGENERATE,
            "loeo": {k: v for k, v in blk.items() if not k.startswith("_")},
            "paired_vs_shipped": P._paired(blk["_per_window_ade"].numpy(),
                                           shipped["_per_window_ade"].numpy(), eid),
            "C-leak_in_sample_ade": round(float(ins_ade.mean()), 6),
            "C-leak_gap_m": round(float(ins_ade.mean()
                                        - blk["_per_window_ade"].mean()), 6),
            "rho_reachable": rho_reachable(sc, de_all, keep, eid),
            "weights": {k: v for k, v in meta.items() if not k.startswith("_")},
        }
        arms[name] = (idx, blk)

    # ---- C-monotone: the degenerate rows MUST reproduce their incumbents ---
    a_ship = de_all.gather(1, arms["A-shipped"][0][:, None]).squeeze(1).mean()
    a_ref = de_all.gather(1, arms["A-refined"][0][:, None]).squeeze(1).mean()
    ref_ade = de_all.gather(1, refined_idx[:, None]).squeeze(1).mean()
    controls["C-monotone"] = {
        "what": ("a strictly-monotone function of ONE banked scalar has the "
                 "IDENTICAL argmax, so these two rows are CONTROLS and must "
                 "reproduce their incumbent selectors exactly"),
        "A-shipped_ade": round(float(a_ship), 6),
        "shipped_ade": round(float(shipped["_per_window_ade"].mean()), 6),
        "A-refined_ade": round(float(a_ref), 6),
        "refined_argmax_ade": round(float(ref_ade), 6),
        "shipped_matches": bool(abs(float(a_ship - shipped["_per_window_ade"]
                                          .mean())) < 1e-6),
        "refined_matches": bool(abs(float(a_ref - ref_ade)) < 1e-6),
        "can_fire": True,
    }

    # ---- C-permuted-target: an INDEPENDENT per-window candidate permutation -
    g = torch.Generator().manual_seed(RNG_SEED)
    perm = torch.stack([torch.randperm(de_all.shape[1], generator=g)
                        for _ in range(de_all.shape[0])])
    # ⚠️ The FEATURES are permuted; `keep` is NOT. Permuting the mask too would
    # let the argmax land outside the true survivor set, and the comparator would
    # silently become the FULL-FAN uniform floor (14.54) instead of the
    # survivor-restricted one (~2.78) — a control judged against the wrong floor
    # is not a control. Caught on the first run of this probe.
    pf = {k: torch.gather(v, 1, perm) for k, v in feats.items()}
    psc, _ = loeo_scores(pf, FEATURE_SETS["D-lon+scores"], keep, de_all, eid)
    pidx = argmax_over_survivors(psc, keep)
    p_ade = de_all.gather(1, pidx[:, None]).squeeze(1)
    uni = de_all.masked_fill(~keep, float("nan")).nanmean(1)   # exact uniform pick
    controls["C-permuted-target"] = {
        "what": ("features permuted INDEPENDENTLY per window, then fit and "
                 "evaluated LOEO. Must land at the uniform-random floor; a score "
                 "above it means the split leaks or the protocol is broken."),
        "ade": round(float(p_ade.mean()), 4),
        "uniform_floor_over_survivors": round(float(uni.mean()), 4),
        "paired_vs_floor": P._paired(p_ade.numpy(), uni.numpy(), eid),
        "can_fire": True,
    }
    controls["C-shuffled"] = {
        "what": "permute-then-argmax is a uniform pick for ANY score",
        "status": "VACUOUS BY CONSTRUCTION — reported, never load-bearing",
        "can_fire": False,
    }

    # ---- POST-HOC: the SAME features under a SELECTION-AWARE objective -----
    soft = {}
    for name in ("A-shipped", "B-both", "C-lon", "D-lon+scores"):
        sc, meta = loeo_scores(feats, sets[name], keep, de_all, eid,
                               objective="softade")
        idx = argmax_over_survivors(sc, keep)
        blk = P.ranker_block(de_all, de_or, idx, eid, tag=f"{name}-softade")
        soft[name] = {
            "features": sets[name],
            "loeo": {k: v for k, v in blk.items() if not k.startswith("_")},
            "paired_vs_shipped": P._paired(blk["_per_window_ade"].numpy(),
                                           shipped["_per_window_ade"].numpy(), eid),
            "paired_vs_same_features_under_CE": P._paired(
                blk["_per_window_ade"].numpy(),
                arms[name][1]["_per_window_ade"].numpy(), eid),
            "weights": {k: v for k, v in meta.items() if not k.startswith("_")},
        }

    # ---- S1b: the emitted-fan readout (needs the emitted bank) -------------
    s1b = None
    if em is not None:
        s1b = s1b_block(d, em, de_all, de_or, keep, eid, shipped)
        arms["S1b-emitted"] = (s1b["_idx"],
                               P.ranker_block(de_all, de_or, s1b["_idx"], eid,
                                              tag="S1b-emitted"))

    # ---- the four families, per family, never pooled -----------------------
    fams = {}
    for name, (idx, blk) in arms.items():
        fams[name] = {
            "paired_vs_shipped": P.family_paired(d, idx, shipped_idx, eid,
                                                 tag=f"{name}-minus-shipped"),
            # ⛔ The PAIRED block above is the load-bearing one and is NEVER
            # allowed to fail: R-2026-08-03-c established that every published
            # four-family ABSOLUTE rate before the dt fix is wrong by 5x-25x
            # while cross-arm ranks and PAIRED deltas survive. So an absolute
            # block that cannot be computed on this host is recorded WITH ITS
            # REASON rather than silently dropped or allowed to kill the run.
            "absolute": _absolute_or_reason(d, idx, eid, blk, name),
        }
    fams["_ceiling_oracle_minus_shipped"] = P.family_paired(
        d, de_all.argmin(1), shipped_idx, eid, tag="oracle-minus-shipped")

    res = {
        "experiment": "E-S1-0 — is there a TARGET for S1's climb-out?",
        "arm": arm, "n_windows": int(d["fan"].shape[0]),
        "n_episodes": len(set(eid)), "n_anchors": int(d["logits"].shape[1]),
        "bank": {"path": d["_path"], "sha256": d["_sha256"],
                 "ckpt": d.get("ckpt"), "ckpt_step": d.get("ckpt_step"),
                 "steps": d.get("steps"), "nav_mode": d.get("nav_mode"),
                 "host": d.get("host")},
        "published": pub,
        "prereg": prereg_provenance(),
        "protocol": {
            "fit": ("listwise softmax CE vs the oracle index, restricted to the "
                    "S2-reachable survivors, no bias (a per-window constant "
                    "cancels in the softmax)"),
            "eval": "LEAVE-ONE-EPISODE-OUT, asserted disjoint per fold",
            "estimator": PREREG_THRESHOLDS["estimator"],
            "headline": "selection ADE@2s over the survivors, PAIRED vs shipped",
            "iters": FIT_ITERS, "lr": FIT_LR,
        },
        "reachability": reach,
        "shipped": {k: v for k, v in shipped.items() if not k.startswith("_")},
        "arms": rows,
        "controls": controls,
        "objective_diagnostic": {
            "status": ("POST-HOC. Decides NO registered branch and moved no "
                       "threshold. It exists because EVERY registered fit came "
                       "back separated ADVERSE — including one whose feature set "
                       "CONTAINS the incumbent score — which makes 'no "
                       "information' and 'wrong objective' the two live readings, "
                       "and they license opposite decisions."),
            "registered_objective": ("listwise softmax CE vs the oracle INDEX — "
                                     "the objective refc_train.loss_rcls uses"),
            "diagnostic_objective": ("expected ADE under the score's own softmax "
                                     "over survivors (a soft argmax ADE)"),
            "arms": soft},
        "s1b": ({k: v for k, v in s1b.items() if not k.startswith("_")}
                if s1b else {"status": "UNMEASURED — no --emitted-bank given",
                             "n": 0}),
        "families": fams,
        "wall_s": round(time.time() - t0, 1),
    }
    res["verdict"] = adjudicate(res)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"s1_climbout_probe_{arm}.json").write_text(
        json.dumps(P._clean(res), indent=2), encoding="utf-8")
    return res


def _sep_dir(p: dict, better_is_negative: bool = True) -> tuple[bool, bool]:
    """(separated, favours_treatment) — the DIRECTION PREDICATE prereg §4 requires.

    A trigger that reads only "separated" was satisfied LITERALLY by S3 while both
    of its controls BEAT the score. Every branch below reads both bits.
    """
    mean = p.get("delta", p.get("mean"))
    sep = bool(p.get("separated", False)) and not bool(p.get("degenerate", False))
    fav = bool(mean is not None and ((mean < 0) if better_is_negative else (mean > 0)))
    return sep, fav


def adjudicate(res: dict) -> dict:
    """§4's branches, applied MECHANICALLY. No row is chosen by hand."""
    mat = PREREG_THRESHOLDS["material_m"]
    v = {}
    for key, arm in (("B-both", "B-both"), ("C-lon", "C-lon"),
                     ("D-lon+scores", "D-lon+scores")):
        p = res["arms"][arm]["paired_vs_shipped"]
        sep, fav = _sep_dir(p)
        m = p.get("mean", p.get("delta"))
        if sep and fav and m is not None and m <= -mat:
            b = "TARGET-EXISTS" if key == "B-both" else "LEVER-LIVE"
        elif sep and fav:
            b = "TARGET-MARGINAL" if key == "B-both" else "LEVER-MARGINAL"
        elif sep and not fav:
            b = "ADVERSE"
        else:
            b = "NOT-SEPARATED"
        v[key] = {"branch": b, "delta_m": m, "separated": sep,
                  "favours_treatment": fav,
                  "leak_gap_m": res["arms"][arm]["C-leak_gap_m"]}
        if m is not None and sep and fav and m <= -PREREG_THRESHOLDS["red_flag_m"]:
            v[key]["RED_FLAG"] = ("separated better by > 0.10 m — STOP AND AUDIT "
                                  "for a protocol leak, do not publish as a win")
    if res.get("s1b", {}).get("paired_emitted_minus_prefinal"):
        p = res["s1b"]["paired_emitted_minus_prefinal"]
        sep, fav = _sep_dir(p)
        m = p.get("delta", p.get("mean"))
        mm = PREREG_THRESHOLDS["material_m"]
        v["S1b"] = {"branch": ("S1b-MATERIAL" if (sep and fav and m <= -mm)
                               else "S1b-ADVERSE" if (sep and not fav)
                               else "S1b-HYGIENE"),
                    "delta_m": m, "separated": sep, "favours_treatment": fav}
    else:
        v["S1b"] = {"branch": "S1b-UNMEASURED",
                    "reason": "no emitted bank supplied"}
    dead = (v["B-both"]["branch"] in ("NOT-SEPARATED", "ADVERSE")
            and v["C-lon"]["branch"] in ("NOT-SEPARATED", "ADVERSE"))
    v["S1_IS_DEAD_AFTER_ALL"] = {
        "fired": bool(dead),
        "rule": PREREG_THRESHOLDS["S1_NOT_SEPARATED"],
        "what_it_does_NOT_license": (
            "'supervising the refined readout cannot work'. Prereg §2 registers "
            "IN ADVANCE that a monotone re-weighting of a banked scalar cannot "
            "test the climb-out; a null bounds the FREE version only."),
    }
    v["registered_prediction"] = PREREG_THRESHOLDS["registered_prediction"]
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(P.PUBLISHED))
    ap.add_argument("--out", required=True)
    ap.add_argument("--emitted-bank", default=None,
                    help="bank from refc_s1_dump_emitted.py (S1b, §4.3)")
    a = ap.parse_args(argv)
    res = run(a.bank, a.arm, a.out, emitted_bank=a.emitted_bank)
    print(json.dumps(P._clean(res["verdict"]), indent=2), flush=True)
    return 0


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())
