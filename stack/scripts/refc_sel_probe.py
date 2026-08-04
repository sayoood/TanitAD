"""E-SEL-0 / E-SEL-1 — the 0-GPU-day probe that GATES the D-SEL retrain.

This is §9 escalation item 2 of ``Project Steering/PREREG_D-SEL_REFC_SELECTION_
SURFACE.md``, addressed to the eval/tools stream and written here because it did
not exist and nothing in §7 may run until it reports.

WHAT IT DECIDES
============================================================================
D-SEL's claim is that every MEASURED defect of REF-C is on the SELECTION
surface. §6 registers the cheapest discriminating experiment against that claim,
with BOTH outcomes and every threshold fixed in advance (prereg §6.3):

  E-SEL-0  Is the DISCARDED refined confidence a better ranker than the t=0
           classifier score we ship?  -> S1 FREE WIN / NEEDS TRAINING / DEAD
  E-SEL-1  Does the CONSEQUENCE (``law_head``) carry candidate-discriminating
           information?                -> S3 LIVE / S3 DEAD

Four negative controls run FIRST and are reported alongside (prereg §6.0):
``identity``, ``shuffled``, ``raster``, ``oracle-floor``.

⛔ THE THRESHOLDS LIVE IN THE PRE-REGISTRATION, NOT HERE. This module reads them
from :data:`PREREG_THRESHOLDS`, which is a TRANSCRIPTION of §6.3, and it prints
the prereg's staged git blob id beside every verdict so a reader can re-run
``git hash-object`` and check that the transcription's source has not moved. A
threshold that a probe may edit is not a pre-registration.

TWO RUN MODES, AND WHY THE SPLIT IS A FINDING RATHER THAN A COMPROMISE
============================================================================
``--fan <fan_refc-*.pt>`` alone is the **fan-only** mode. The banked fan dumps
(``taniteval/results/fan_refc-{base,xl}-30k.pt``, written by
``taniteval.refc_rerank.dump``) store the emitted candidate set ``fan`` [B,N,4,2]
and the SHIPPED score ``logits`` [B,N] — which is ``anchor_logits``, the t=0
classifier pass. **They do NOT store the refined confidence**, because the
decoder discarded it: that discarding IS defect D1. So E-SEL-0's *treatment* leg
needs a decoder forward and cannot come from the bank.

⭐ Its *control* leg can, EXACTLY, and this is not a workaround. C-shuffled
permutes the score along the candidate axis and takes ``argmax``; for a uniform
permutation that is a UNIFORM RANDOM CANDIDATE, whatever the score was. The
shuffled control is therefore **score-independent** — the shuffled control for
the refined confidence is the same distribution as the shuffled control for the
shipped logits — so it is fully determined by the bank and is computed here in
CLOSED FORM (per window, the exact expectation over all N! permutations is the
mean over candidates), with an empirical multi-seed draw printed beside it as a
self-test. See :func:`shuffled_control`.

``--ckpt`` + ``--data`` is the **full** mode: one forward over the canonical val
episodes recovers ``refined_logits`` and ``pooled`` (``RefCModel.forward``
returns both unconditionally post-D-SEL), which closes E-SEL-0 and E-SEL-1.

⚠️ REQUIRED IN FULL MODE — the raster (prereg §5.2, R-2026-08-02-a). REF-C was
once scored at 176x624 = 120 tokens against its trained 8x8 = 64; **XL crashed
and base returned numbers silently**. C-raster asserts the square raster before
anything is scored and REFUSES otherwise. The w120 256x640 cylindrical caches
built for the flagship are NOT admissible input to REF-C.

ESTIMATOR, DECLARED BEFORE ANY NUMBER
============================================================================
``taniteval.ci.paired_episode_cluster_bootstrap`` / ``episode_cluster_bootstrap``,
resampling unit = **episode**, ``n_boot = 2000``.
⛔ ``overlapping_holdout_se`` is never called: it is not a jackknife, it is not a
valid SE, and it BIASES the point estimate (mean-of-split-means, measured -6.67 %
to +11.69 % over 27 arms, bidirectional, up to a sign flip on paired deltas).

FOUR METRIC FAMILIES (binding, Sayed 2026-08-02)
============================================================================
Reported per family, never pooled, via ``taniteval.four_families.all_families``
— the module whose ``infer_dt`` derives the grid from ``wp_steps`` instead of
assuming 0.1 s. The banked fans carry ``wp_steps = [5,10,15,20]``, i.e. a 0.5 s
spacing, so a hard-coded 0.1 s would inflate every speed by 5x and every accel by
25x (R-2026-08-03-c). Where a family cannot be computed it is reported with its
REASON and its n — never silently dropped.

USAGE
============================================================================
    # fan-only: the controls + TACTICAL family + S2, 0 GPU, no checkpoint
    OMP_NUM_THREADS=6 python scripts/refc_sel_probe.py \\
        --fan taniteval/results/fan_refc-base-30k.pt \\
        --out ".../incoming/2026-08-03-esel-verdict/raw" \\
        --controls identity,shuffled,raster,oracle-floor

    # full: closes E-SEL-0 and E-SEL-1
    OMP_NUM_THREADS=6 python scripts/refc_sel_probe.py \\
        --fan taniteval/results/fan_refc-base-30k.pt \\
        --ckpt /root/models/refc-base-30k/ckpt.pt \\
        --data /root/valdata/physicalai-val-0c5f7dac3b11 \\
        --anchors /workspace/experiments/refc_anchors_base128.pt \\
        --out ".../incoming/2026-08-03-esel-verdict/raw"
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

# --------------------------------------------------------------------------- #
# torch spawns ~113 threads PER PROCESS and a multi-arm panel then makes NO
# progress while looking exactly like a deadlock (MEASURED 2026-07-27: 7 arms at
# GPU sm 0-6 % for 50 minutes; OMP_NUM_THREADS=6 and the same arm took 232 s).
# Set before torch does any work, and never silently override a caller's value.
# --------------------------------------------------------------------------- #
os.environ.setdefault("OMP_NUM_THREADS", "6")
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]                      # <repo>/stack/scripts/x.py
for _p in (_REPO / "taniteval", _REPO / "stack"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from taniteval import ci, four_families                        # noqa: E402

PREREG = "Project Steering/PREREG_D-SEL_REFC_SELECTION_SURFACE.md"

#: A TRANSCRIPTION of prereg §6.3. Every value here is quoted from the staged
#: blob whose id this module prints; none may be edited by a run.
PREREG_THRESHOLDS = {
    "S1_FREE_WIN": "E-SEL-0 paired dADE@2s separated AND >= 0.02 m better",
    "S1_NEEDS_TRAINING": ("E-SEL-0 not separated AND C-shuffled clearly worse "
                          "than refined (frac_sel_2x_worse gap >= 0.05)"),
    "S1_DEAD": ("E-SEL-0 not separated AND C-shuffled indistinguishable from "
                "refined"),
    "S3_LIVE": "E-SEL-1 Spearman rho separated from C-shuffled AND |rho| >= 0.10",
    "S3_DEAD": "rho CI includes the shuffled control",
    "INDETERMINATE": ("E-SEL-0 separated but < 0.02 m; tie-break on E-SEL-1 "
                      "(S3 LIVE => run S1+S2+S3+S4; S3 DEAD => ship S2+S4 only)"),
    "RED_FLAG_STOP": ("ADE separated better by > 0.10 m => stop and audit for a "
                      "leak, do not publish"),
    "registered_prediction": ("S1 NEEDS TRAINING (E-SEL-0 not separated, "
                              "C-shuffled clearly worse) and S3 LIVE at a small "
                              "rho, 0.10-0.25"),
    "free_win_m": 0.02,
    "frac_2x_gap_m": 0.05,
    "rho_min": 0.10,
    "red_flag_m": 0.10,
}

#: prereg §5.3 / §6.0 C-oracle-floor. The arm travels with the number or the
#: number is not quotable: 0.1640 / 45.4 % are REF-C-**XL**'s, not base's.
PUBLISHED = {
    "refc-base-30k": {"selected_ade2s": 0.4728, "oracle_ade2s": 0.1914,
                      "frac_sel_2x_worse": 0.4109, "sel_gap": 0.2813,
                      "n_anchors": 128,
                      "source": "taniteval/results/scaleab_refc-base-30k_vs_refc-xl-30k.json"},
    "refc-xl-30k": {"selected_ade2s": 0.4714, "oracle_ade2s": 0.1640,
                    "frac_sel_2x_worse": 0.4540, "sel_gap": 0.3075,
                    "n_anchors": 256,
                    "source": "taniteval/results/scaleab_refc-base-30k_vs_refc-xl-30k.json"},
    # ⚠️ prereg §9.3: MODEL_REGISTRY.md §4.2 cites `taniteval/results/
    # refc-small-30k.json`, WHICH DOES NOT EXIST. The artifact lives only in the
    # research-hub incoming package below — that directory is the source, not a
    # copy. `oracle_ade2s` is `selected - sel_gap` (0.5261 - 0.3048), the same
    # decomposition the other two rows publish directly.
    "refc-small-30k": {"selected_ade2s": 0.5261, "oracle_ade2s": 0.2213,
                       "frac_sel_2x_worse": 0.3825, "sel_gap": 0.3048,
                       "n_anchors": 64,
                       "source": ("TanitAD Research Hub/Benchmarks & Eval/"
                                  "Implementation/incoming/2026-07-22-refc-small-30k/"
                                  "scaleab_refc-small-30k_vs_refc-base-30k.json")},
}

#: REF-C's trained input geometry. ``taniteval/data.py`` documents the episode
#: cache as ``[T, 9, 256, 256] uint8`` (3 stacked RGB frames) and the decoder
#: cross-attends the resulting 8x8 = 64 conv-map tokens. R-2026-08-02-a is the
#: retraction that exists because this was once 176x624 -> 120 tokens.
RASTER = {"frame_hw": (256, 256), "token_grid": (8, 8), "n_tokens": 64,
          "channels": 9}

TOL_IDENTITY = 1e-6          # prereg §6.0 C-identity: any deviation > this STOPS
#: ⚠️ Half-width for comparing against a PUBLISHED value. The registry stores 4 dp,
#: so a 4-dp round already admits 5e-5 — and a value sitting exactly at the 5th
#: place (XL's oracle is 0.16395) lands on the boundary, where a 1e-6 float
#: difference between two hosts decides PASS vs FAIL. MEASURED: the Thor decode
#: gave 0.163949 against a published 0.1640, a deviation of 5.1e-5 that a 5e-5
#: gate called a FAILURE. The EXACT check is `C-identity-vs-bank` (which reads
#: 1.0000 agreement); this one is a rounding-tolerant sanity row and its
#: tolerance says so.
TOL_PUBLISHED = 1.5e-4
N_BOOT = ci.DEFAULT_N_BOOT   # 2000
SHUFFLE_SEEDS = 24           # empirical self-test beside the closed form


# =========================================================================== #
# provenance                                                                  #
# =========================================================================== #

def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=_REPO, capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception as exc:                                   # pragma: no cover
        return f"<git failed: {exc}>"


def prereg_provenance() -> dict:
    """The falsifiable "fixed in advance" object of prereg §10.1.

    A document cannot carry its own sha256, so the prereg deliberately prints
    none. The claim it DOES make is that its staged git blob id is fixed at
    staging time. We record both the indexed id and the working-tree id: if they
    differ, §6.3's thresholds moved after staging and "committed in advance" is
    void — which is exactly the property an mtime never bought (R11 refuted
    D-TAC1's mtime-based version of this claim).
    """
    staged = _git("ls-files", "-s", "--", PREREG).split()
    worktree = _git("hash-object", PREREG)
    blob = staged[1] if len(staged) > 1 else "<not staged>"
    return {
        "path": PREREG,
        "staged_blob": blob,
        "worktree_blob": worktree,
        "thresholds_unmoved_since_staging": bool(blob and blob == worktree),
        "how_to_verify": (f"git ls-files -s -- '{PREREG}' && "
                          f"git hash-object '{PREREG}'  # must match"),
        "thresholds": PREREG_THRESHOLDS,
    }


def _sha256(path: Path, limit: int = 1 << 30) -> str:
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as fh:
        while n < limit:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest()


# =========================================================================== #
# the fan bank                                                                #
# =========================================================================== #

def load_fan(path: str | Path) -> dict:
    """Load a banked fan dump and pin what it does and does NOT contain."""
    path = Path(path)
    d = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("fan", "logits", "sel", "gt", "eid", "wp_steps"):
        if k not in d:
            raise KeyError(f"{path.name} has no {k!r} — not a refc_rerank.dump "
                           f"fan bank (keys: {sorted(d)})")
    d["_path"] = str(path)
    d["_sha256"] = _sha256(path)
    d["_has_refined_logits"] = "refined_logits" in d
    d["_has_pooled"] = "pooled" in d
    return d


def candidate_ade(fan: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """[B, N] per-candidate ADE@2s — the program's own definition.

    VERBATIM from ``taniteval.refc_rerank._score_row``::

        de_all = torch.linalg.norm(fan - gt[:, None], dim=-1).mean(-1)

    Re-deriving it would be how two definitions of the headline quantity drift
    apart; this is the one the published 0.4728 / 0.1914 were computed with.
    """
    return torch.linalg.norm(fan - gt[:, None], dim=-1).mean(-1)


def _boot(per_window, eid, **kw) -> dict:
    return ci.episode_cluster_bootstrap(np.asarray(per_window, dtype=np.float64),
                                        eid, n_boot=N_BOOT, **kw)


def _paired(a, b, eid, **kw) -> dict:
    return ci.paired_episode_cluster_bootstrap(
        np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
        eid, n_boot=N_BOOT, **kw)


def ranker_block(de_all: torch.Tensor, de_or: torch.Tensor, idx: torch.Tensor,
                 eid, *, tag: str) -> dict:
    """Everything §7.1's TACTICAL family asks of one selection rule."""
    de_sel = de_all.gather(1, idx[:, None]).squeeze(1)
    gap = de_sel - de_or
    two_x = (de_sel > 2 * de_or).double()
    hit = (idx == de_all.argmin(1)).double()
    return {
        "tag": tag,
        "ade_0_2s": _boot(de_sel.numpy(), eid),
        "sel_gap": _boot(gap.numpy(), eid),
        "frac_sel_2x_worse": _boot(two_x.numpy(), eid),
        "rank_acc": _boot(hit.numpy(), eid),
        "_per_window_ade": de_sel,
        "_per_window_gap": gap,
        "_per_window_2x": two_x,
    }


# =========================================================================== #
# CONTROLS (prereg §6.0) — run FIRST, reported alongside, non-negotiable       #
# =========================================================================== #

def control_identity(d: dict, de_all: torch.Tensor, arm: str) -> dict:
    """C-identity — re-scoring the bank with every D-SEL flag off reproduces the
    published selected ADE bit-for-bit. Any deviation > 1e-6 ⇒ STOP."""
    shipped = d["logits"].argmax(1)
    idx_ok = bool(torch.equal(shipped, d["sel"]))
    ade = float(de_all.gather(1, d["sel"][:, None]).squeeze(1).mean())
    pub = PUBLISHED.get(arm, {}).get("selected_ade2s")
    dev = None if pub is None else abs(ade - pub)
    # ⛔ A MISSING PUBLISHED VALUE IS "NOT COMPARABLE", NOT "FAIL". Conflating
    # the two would make this control fire on an arm nobody published rather
    # than on the harness drift it exists to catch — a guard that cannot
    # distinguish its own inapplicability from a real failure is C13's class.
    n = int(de_all.shape[0])
    # ⛔ 0.4728 IS AN 881-WINDOW NUMBER. Comparing it to a run on a different
    # window set is not a control, it is a category error — and it would fire as
    # a harness alarm on what is really a quarantined clip. n travels with the
    # number or the number is not quotable.
    if pub is not None and n != 881:
        status = (("PASS (bit-exact half only) — n = %d != the canonical 881, so "
                   "the published %.4f is NOT-COMPARABLE here; see "
                   "C-identity-vs-bank for the episode-matched check")
                  % (n, pub)) if idx_ok else \
                 "FAIL — argmax(logits) != sel; the decode is not self-consistent"
    elif pub is None:
        status = ("PASS (bit-exact half only) — NO PUBLISHED VALUE for this arm, "
                  "so the reproduction half is NOT-COMPARABLE") if idx_ok else \
                 "FAIL — argmax(logits) != sel; the bank is not self-consistent"
    elif idx_ok and dev <= TOL_PUBLISHED:
        status = "PASS"
    else:
        status = "FAIL — STOP, the harness changed something else"
    return {
        "control": "C-identity",
        "argmax_logits_equals_sel": idx_ok,
        "recomputed_selected_ade_0_2s": round(ade, 6),
        "published_selected_ade_0_2s": pub,
        "n_windows": n,
        "canonical_n_windows": 881,
        "abs_deviation": None if dev is None else round(dev, 8),
        "tolerance": TOL_IDENTITY,
        "note": ("the published value is stored rounded to 4 dp, so the "
                 f"admissible deviation is {TOL_PUBLISHED} (4-dp rounding "
                 "half-width plus cross-host float); "
                 "the argmax==sel check is the BIT-EXACT half of this control"),
        "status": status,
    }


def control_oracle_floor(d: dict, de_or: torch.Tensor, arm: str) -> dict:
    """C-oracle-floor — the fan dump and the published numbers must agree."""
    got = float(de_or.mean())
    n = int(de_or.shape[0])
    pub = PUBLISHED.get(arm, {}).get("oracle_ade2s")
    dev = None if pub is None else abs(got - pub)
    if pub is None:
        status = "NOT-COMPARABLE — no published oracle for this arm"
    elif n != 881:
        # ⛔ Same category error as C-identity: 0.1914 is an 881-window number.
        status = (f"NOT-COMPARABLE — n = {n} != the canonical 881 (a clip was "
                  f"quarantined). The episode-matched reproduction check is "
                  f"C-identity-vs-bank; this row is informational only.")
    elif dev <= TOL_PUBLISHED:
        status = "PASS"
    else:
        status = "FAIL — the fan dump and the published numbers disagree"
    return {
        "control": "C-oracle-floor",
        "recomputed_oracle_in_fan_ade_0_2s": round(got, 6),
        "published_oracle_in_fan_ade_0_2s": pub,
        "n_windows": n, "canonical_n_windows": 881,
        "abs_deviation": None if dev is None else round(dev, 8),
        "arm_the_number_belongs_to": arm,
        "⚠️": "0.1640 / 45.4 % are REF-C-XL's; base is 0.1914 / 41.09 %",
        "status": status,
    }


def control_identity_vs_bank(d: dict, bank_path: str | Path,
                             de_all: torch.Tensor) -> dict:
    """C-identity, STRONG form — a re-decode must reproduce the banked fan.

    Run when ``--fan`` is an AUGMENTED dump produced by
    ``scripts/refc_sel_dump_refined.py``: the same weights, the same nav mode and
    the same protocol on a different host must land on the same fan. Windows are
    matched by EPISODE (the augmented dump may have fewer episodes if a clip was
    quarantined), never by position.

    ⚠️ A bitwise match across hosts is NOT expected and NOT required — Thor is
    aarch64 on a different torch build. What is required is that the SELECTION
    agrees and the trajectories agree to float tolerance. **E-SEL-0's delta never
    depends on this check**: its paired comparison is internal to one forward
    (``argmax(refined)`` vs ``argmax(anchor)`` over the SAME ``anchor_traj``), so
    a host difference cannot manufacture or hide it. This control exists to show
    the re-decode is the same experiment, not to supply the contrast.
    """
    bank_path = Path(bank_path)
    if not bank_path.exists():
        return {"control": "C-identity-vs-bank", "status": "SKIPPED — no bank",
                "bank": str(bank_path)}
    bk = torch.load(bank_path, map_location="cpu", weights_only=False)
    keep_eps = set(d["eid"])
    m = torch.tensor([e in keep_eps for e in bk["eid"]])
    if int(m.sum()) != de_all.shape[0]:
        return {"control": "C-identity-vs-bank",
                "status": "SKIPPED — window counts do not line up after the "
                          "episode match",
                "bank_windows_matched": int(m.sum()),
                "decode_windows": int(de_all.shape[0])}
    bf, bs = bk["fan"][m], bk["sel"][m]
    # ⛔ THE WINDOW-ALIGNMENT CHECK COMES FIRST, and it is GT-based. An episode
    # MASK is not an alignment proof: `data.load_frames` numbers episodes by
    # POSITION, so dropping one unreadable clip renumbers every later clip and a
    # mask on `eid` then silently compares DIFFERENT windows. MEASURED here on
    # the first attempt — selection agreement read 0.7183 and the fans looked
    # 22 m apart, both artefacts of the shift. The GT is identical for identical
    # windows and independent of the model, so it is the right invariant.
    gt_max = float((bk["gt"][m] - d["gt"]).abs().max())
    if gt_max > 1e-3:
        return {"control": "C-identity-vs-bank",
                "status": ("FAIL — WINDOWS ARE NOT ALIGNED (GT differs by "
                           f"{gt_max:.4g} m). Do NOT read any comparison against "
                           "the bank; check episode-id assignment first."),
                "max_abs_gt_difference_m": gt_max}
    per_win = (bf - d["fan"]).abs().amax(dim=(1, 2, 3))
    fan_max = float(per_win.max())
    fan_med = float(per_win.median())
    fan_p95 = float(per_win.quantile(torch.tensor(0.95)))
    sel_agree = float((bs == d["sel"]).double().mean())
    bank_de = candidate_ade(bf, bk["gt"][m]).gather(1, bs[:, None]).squeeze(1)
    dec_de = de_all.gather(1, d["sel"][:, None]).squeeze(1)
    return {
        "control": "C-identity-vs-bank",
        "bank": str(bank_path),
        "n_windows_matched": int(m.sum()),
        "n_episodes_matched": len(keep_eps),
        "max_abs_gt_difference_m": gt_max,
        "per_window_max_abs_fan_difference_m": {
            "median": round(fan_med, 8), "p95": round(fan_p95, 6),
            "max": round(fan_max, 6),
            "n_windows_above_0p01m": int((per_win > 0.01).sum())},
        "selection_agreement": round(sel_agree, 6),
        "bank_selected_ade_on_matched_windows": round(float(bank_de.mean()), 6),
        "decode_selected_ade_on_matched_windows": round(float(dec_de.mean()), 6),
        "abs_deviation": round(abs(float(bank_de.mean() - dec_de.mean())), 8),
        "note": ("cross-HOST, cross-ARCH (Thor aarch64 vs the eval pod's x86) "
                 "and cross-torch-build; bitwise identity is not expected. "
                 "E-SEL-0's paired delta is internal to ONE forward and does "
                 "not rest on this check."),
        # ⛔ JUDGE ON THE SELECTION AND THE TYPICAL WINDOW, NOT ON `max`. The fan
        # contains candidates implying up to 171.5 km/h (72-74 % are outside a
        # bounded-acceleration band), and a float difference on one of those
        # moves it metres while changing nothing that is ever selected. A `max`
        # threshold would fire on the arithmetic of a candidate the model has
        # already effectively discarded.
        "status": ("PASS" if sel_agree >= 0.99 and fan_med < 1e-2 else
                   "REVIEW — the re-decode is not reproducing the bank"),
    }


def control_raster(d: dict, cfg=None, frames_hw=None) -> dict:
    """C-raster — assert REF-C's square raster before anything is scored.

    R-2026-08-02-a: REF-C was once scored at 176x624 = 120 tokens against its
    trained 8x8 = 64. **XL crashed; base returned numbers silently.** "When two
    arms share a defect and only one crashes, the crash is the honest signal."
    """
    out = {"control": "C-raster", "required": dict(RASTER)}
    if frames_hw is None and cfg is None:
        out.update({
            "scored_a_decode": False,
            "status": "NOT-APPLICABLE (fan-only mode)",
            "reason": ("no decode is run in fan-only mode, so no raster is fed "
                       "to REF-C. The banked fan is the OUTPUT of a decode that "
                       "already ran at the trained raster on the eval pod "
                       f"(ckpt {d.get('ckpt')!r}, step {d.get('ckpt_step')!r})."),
            "⛔": ("⛔ THIS CONTROL BECOMES BINDING THE MOMENT A --ckpt DECODE "
                   "IS ADDED. The only val cache currently reachable off-pod is "
                   "the w120 256x640 CYLINDRICAL flagship cache; feeding that to "
                   "REF-C IS R-2026-08-02-a and this control must REFUSE it.")})
        return out
    got = tuple(frames_hw) if frames_hw is not None else None
    ok = got == RASTER["frame_hw"]
    out.update(scored_a_decode=True, observed_frame_hw=got,
               status="PASS" if ok else "REFUSE — raster mismatch (R-2026-08-02-a)")
    if not ok:
        raise SystemExit(
            f"C-raster REFUSES to score: frames are {got}, REF-C was trained at "
            f"{RASTER['frame_hw']} (8x8 = 64 tokens). This is R-2026-08-02-a — "
            f"XL crashed on it and BASE RETURNED NUMBERS SILENTLY.")
    return out


def shuffled_control(de_all: torch.Tensor, de_or: torch.Tensor, eid, *,
                     seeds: int = SHUFFLE_SEEDS) -> dict:
    """C-shuffled — rank on a PERMUTED score vector.

    ⭐ THE OBSERVATION THAT MAKES THIS LEG UNBLOCKED. Permuting a score along the
    candidate axis and taking ``argmax`` returns ``pi^-1(argmax s)``; for a
    uniform permutation ``pi`` that index is UNIFORM over the N candidates
    **whatever the score was**. C-shuffled is therefore SCORE-INDEPENDENT: the
    shuffled control for the discarded refined confidence has exactly the same
    distribution as the shuffled control for the shipped logits, and both are
    fully determined by the banked fan.

    So it is computed in CLOSED FORM — the exact expectation over all N!
    permutations is, per window, the mean over candidates — and an empirical
    ``seeds``-draw is reported beside it purely as a self-test of that argument.
    Rao-Blackwellising also removes the seed noise from the interval, which is
    the right thing to do for a control the §6.3 branches are read against.

    ⚠️ It is a control, not a treatment. It bounds S1 from BELOW (what a ranker
    with no information scores); it says nothing about the refined confidence
    until the refined confidence is measured.
    """
    exact_ade = de_all.mean(1)                                    # E[ADE]
    exact_gap = exact_ade - de_or
    exact_2x = (de_all > 2 * de_or[:, None]).double().mean(1)
    n = de_all.shape[1]
    emp = []
    g = torch.Generator().manual_seed(20260803)
    for _ in range(seeds):
        idx = torch.randint(0, n, (de_all.shape[0],), generator=g)
        emp.append(float(de_all.gather(1, idx[:, None]).mean()))
    return {
        "control": "C-shuffled",
        "mechanism": ("uniform permutation of the score along the candidate "
                      "axis then argmax == a UNIFORM RANDOM CANDIDATE, for any "
                      "score; hence score-independent and computable in closed "
                      "form from the bank alone"),
        "ade_0_2s": _boot(exact_ade.numpy(), eid),
        "sel_gap": _boot(exact_gap.numpy(), eid),
        "frac_sel_2x_worse": _boot(exact_2x.numpy(), eid),
        "rank_acc_exact": round(1.0 / n, 6),
        "selftest_empirical_mean_ade": round(float(np.mean(emp)), 6),
        "selftest_empirical_sd_over_seeds": round(float(np.std(emp)), 6),
        "selftest_n_seeds": seeds,
        "selftest_closed_form_mean_ade": round(float(exact_ade.mean()), 6),
        "_per_window_ade": exact_ade,
        "_per_window_gap": exact_gap,
        "_per_window_2x": exact_2x,
    }


# =========================================================================== #
# S2 — the reachability band, replicated on the arm that carries it            #
# =========================================================================== #

def reachability_block(d: dict, de_all: torch.Tensor, de_or: torch.Tensor,
                       eid, *, accel_max: float = 2.5,
                       horizon_s: float = 2.0) -> dict:
    """S2 on THIS arm's fan. D3's 72.08 % was MEASURED on REF-C-**XL**.

    The band is ``flagship_v15.reachability_mask``, re-exported through
    ``refc_select`` — the same function object the 72.08 % / oracle-survives-100 %
    / paired-delta-0.0 measurement was made with, deliberately not a copy.

    S2 is a PRECONDITION, not a win: nobody should expect it to move ADE. It is
    reported as telemetry (prereg §7's rule), and the paired delta is here only
    to confirm the inertness on the base arm too.
    """
    try:
        from tanitad.refs import refc_select as sl
        mask = sl.reachability_mask(d["fan"], d["v0"].to(d["fan"].dtype),
                                    accel_max=accel_max, horizon_s=horizon_s)
    except Exception as exc:
        return {"status": f"UNAVAILABLE — {type(exc).__name__}: {exc}"}
    dead = ~mask.any(1)
    keep = mask | dead[:, None]                    # empty-set fallback (prereg)
    scored = d["logits"].masked_fill(~keep, float("-inf"))
    idx_clipped = scored.argmax(1)
    de_clipped = de_all.gather(1, idx_clipped[:, None]).squeeze(1)
    de_shipped = de_all.gather(1, d["sel"][:, None]).squeeze(1)
    or_clipped = de_all.masked_fill(~keep, float("inf")).min(1).values
    return {
        "band": (f"v_term in [max(0, v0 - {accel_max * horizon_s}), "
                 f"v0 + {accel_max * horizon_s}] m/s; accel_max={accel_max}, "
                 f"horizon_s={horizon_s} — flagship_v15's OWN goal clamp, NOT tuned"),
        "frac_candidates_removed": round(float(1 - mask.double().mean()), 4),
        "frac_windows_with_empty_survivor_set": round(float(dead.double().mean()), 4),
        "oracle_survives_frac": round(float((or_clipped <= de_or + 1e-6)
                                            .double().mean()), 4),
        "oracle_ade_unfiltered": round(float(de_or.mean()), 4),
        "oracle_ade_after_clip": round(float(or_clipped.mean()), 4),
        "as_trained_ade": round(float(de_shipped.mean()), 4),
        "clipped_ade": round(float(de_clipped.mean()), 4),
        "search_space_cheaper_x": round(float(1 / max(mask.double().mean(), 1e-9)), 4),
        "paired_clipped_minus_as_trained": _paired(de_clipped.numpy(),
                                                   de_shipped.numpy(), eid),
        "role": ("PRECONDITION, not a win — S2 is MEASURED inert on ADE and its "
                 "value is COMPUTE (it is what lets S3 exist). Claiming an ADE "
                 "effect for it would contradict its own measurement."),
    }


# =========================================================================== #
# FOUR METRIC FAMILIES (binding) — per family, never pooled                    #
# =========================================================================== #

def per_window_family_components(pred: torch.Tensor, gt: torch.Tensor,
                                 dt: float) -> dict:
    """[n] per-window LONGITUDINAL / LATERAL components, for PAIRED intervals.

    ``four_families`` returns population scalars; the binding rule (§7.1) wants
    each family to carry a paired episode-cluster bootstrap **on the same
    windows**, which needs one number per window. Built from
    ``four_families._seq_geometry`` — the SAME geometry the aggregate block
    uses, at the SAME derived ``dt`` — rather than a second implementation, so
    the paired delta and the headline scalar can never drift apart.

    ⚠️ ``dt`` must come from :func:`four_families.infer_dt`. On the sparse
    4-waypoint view it is 0.5 s, not 0.1 s; a hard-coded 0.1 inflates speed 5x
    and accel 25x (R-2026-08-03-c). ``along_*``, ``cross_*``, heading and
    curvature are dt-invariant and survive that defect; speed and yaw-rate do not.
    """
    import math
    P = four_families._seq_geometry(pred, dt)
    G = four_families._seq_geometry(gt, dt)
    both = P["valid"] & G["valid"]
    both_pair = P["pair_valid"] & G["pair_valid"]
    dh = P["heading"] - G["heading"]
    dh = (dh + math.pi) % (2 * math.pi) - math.pi

    def _rowmean(x, m):
        m = m.to(x.dtype)
        return (x.abs() * m).sum(1) / m.sum(1).clamp_min(1e-9)

    return {
        # LONGITUDINAL
        "speed_abs_err_mps": (P["speed"] - G["speed"]).abs().mean(1),
        "speed_signed_err_mps": (P["speed"] - G["speed"]).mean(1),
        "along_abs_err_m": (P["along"] - G["along"]).abs().mean(1),
        "along_signed_err_m": (P["along"] - G["along"]).mean(1),
        # LATERAL
        "cross_abs_err_m": (P["cross"] - G["cross"]).abs().mean(1),
        "heading_abs_err_deg": _rowmean(dh, both) * 180.0 / math.pi,
        "curvature_abs_err_1pm": _rowmean(P["curvature"] - G["curvature"], both_pair),
        "yaw_rate_abs_err_degps": _rowmean(P["yaw_rate"] - G["yaw_rate"],
                                           both_pair) * 180.0 / math.pi,
    }


#: which family each per-window component belongs to. Per family, NEVER pooled.
FAMILY_OF = {
    "speed_abs_err_mps": "LONGITUDINAL", "speed_signed_err_mps": "LONGITUDINAL",
    "along_abs_err_m": "LONGITUDINAL", "along_signed_err_m": "LONGITUDINAL",
    "cross_abs_err_m": "LATERAL", "heading_abs_err_deg": "LATERAL",
    "curvature_abs_err_1pm": "LATERAL", "yaw_rate_abs_err_degps": "LATERAL",
}


def family_paired(d: dict, idx_a: torch.Tensor, idx_b: torch.Tensor, eid, *,
                  tag: str) -> dict:
    """Paired episode-cluster bootstrap of ranker A minus ranker B, PER FAMILY.

    ⛔ Per family, never pooled — a single composite hides exactly the trade-off
    this decomposition exists to see.
    """
    b = torch.arange(d["fan"].shape[0])
    dt, prov = four_families.infer_dt({"wp_steps": list(d["wp_steps"]),
                                       "dt_s": 0.1})
    A = per_window_family_components(d["fan"][b, idx_a], d["gt"], dt)
    B = per_window_family_components(d["fan"][b, idx_b], d["gt"], dt)
    out = {"_tag": tag, "_dt_s": dt, "_dt_provenance": prov,
           "LONGITUDINAL": {}, "LATERAL": {}}
    for k in A:
        out[FAMILY_OF[k]][k] = _paired(A[k].numpy(), B[k].numpy(), eid)
    out["TACTICAL"] = ("see rankers.* — rank_acc / sel_gap / frac_sel_2x_worse "
                       "are the goal/anchor-selection half and ARE computed")
    out["STRATEGIC"] = ("UNAVAILABLE — see families.*._strategic_reason "
                        "(no route/goal label in a fan bank), n = 0")
    return out


def families_block(d: dict, idx: torch.Tensor, eid=None, sel_half=None, *,
                   tag: str) -> dict:
    """``four_families.all_families`` on the trajectory THIS ranker emits."""
    b = torch.arange(d["fan"].shape[0])
    win = {"pred": d["fan"][b, idx], "gt": d["gt"],
           "wp_steps": list(d["wp_steps"]), "dt_s": 0.1}
    fam = four_families.all_families(win)
    fam["_tag"] = tag
    if sel_half is not None:
        # ⭐ The TACTICAL family is NOT wholly absent here. `four_families.
        # tactical` reports UNAVAILABLE because the fan bank carries no decoded
        # manoeuvre, which is true and stays printed — but the GOAL/ANCHOR-
        # SELECTION half named in prereg §7.1 (`rank_acc`, `sel_gap`,
        # `frac_sel_2x_worse`) is exactly what a fan bank CAN express, and it is
        # the half D-SEL exists to move. Reporting the family as a bare
        # UNAVAILABLE would understate what is measured; merging the two halves
        # into one score would overstate it. So both are carried, labelled.
        fam["tactical"] = dict(fam["tactical"])
        fam["tactical"]["goal_anchor_selection"] = {
            k: sel_half[k] for k in
            ("rank_acc", "sel_gap", "frac_sel_2x_worse") if k in sel_half}
        fam["tactical"]["status"] = (
            "PARTIAL — goal/anchor-selection half MEASURED; manoeuvre-decision "
            "half UNAVAILABLE (no decoded manoeuvre in a fan bank)")
    fam["_strategic_reason"] = (
        "STRATEGIC is UNAVAILABLE from a fan bank: refc_rerank.dump stores no "
        "route/goal label and no decoded strategic decision, and it decoded "
        "with nav_mode='follow_constant' (the historical nav_cmd=None), so the "
        "route input was never exercised. n = "
        f"{int(d['fan'].shape[0])} windows, 0 with a strategic label. This is a "
        "WORK ITEM (prereg §7.1 needs it for the S5 arm), not a pass.")
    fam["_tactical_note"] = (
        "the DECISION half of TACTICAL (selected-vs-executed manoeuvre, the "
        "full confusion) needs decoded manoeuvre logits, which the fan bank "
        "does not store. The GOAL/ANCHOR-SELECTION half — rank_acc, sel_gap, "
        "frac_sel_2x_worse — IS computed, and is the half D-SEL exists to move.")
    return fam


# =========================================================================== #
# E-SEL-0 / E-SEL-1                                                           #
# =========================================================================== #

def e_sel_0(d: dict, de_all: torch.Tensor, de_or: torch.Tensor, eid,
            shipped: dict, shuf: dict, refined_logits=None) -> dict:
    """§6.1 — is the DISCARDED refined confidence a better ranker than the
    shipped t=0 classifier score?

    ⚠️ A LOWER BOUND ON S1, NOT S1. The banked refined confidences were never
    trained as a ranker; S1's claim is that SUPERVISING them makes them one. A
    null here does not falsify S1 — it bounds how much of S1 is free. §6.3
    encodes that asymmetry, and so does the verdict text.
    """
    if refined_logits is None:
        return {
            "experiment": "E-SEL-0",
            "status": "BLOCKED — treatment leg not computable from the bank",
            "reason": ("refc_rerank.dump stores `logits` = anchor_logits (the "
                       "t=0 classifier pass) and the refined `fan`, but NOT the "
                       "refined confidence — the decoder DISCARDED it, which is "
                       "defect D1 itself. Recovering it needs a decoder forward "
                       "over the val episodes at REF-C's 256x256 raster."),
            "n_windows_treatment": 0,
            "n_windows_control": int(de_all.shape[0]),
            "control_leg": {"C-shuffled": "MEASURED — see controls.C-shuffled",
                            "shipped": "MEASURED — see rankers.shipped"},
            "what_would_unblock_it": (
                "--ckpt <refc-base-30k ckpt.pt> --data <physicalai-val-"
                "0c5f7dac3b11 episode cache at [T,9,256,256]>. "
                "RefCModel.forward returns `refined_logits` unconditionally "
                "post-D-SEL, so this is ONE forward, no training."),
            "⛔": ("no substitute statistic is reported in its place. A "
                   "different experiment presented as this one is the failure "
                   "mode the pre-registration exists to prevent."),
        }
    idx_ref = torch.as_tensor(refined_logits).argmax(1)
    ref = ranker_block(de_all, de_or, idx_ref, eid, tag="refined")
    pair = _paired(ref["_per_window_ade"].numpy(),
                   shipped["_per_window_ade"].numpy(), eid)
    gap2x = (float(shuf["frac_sel_2x_worse"]["mean"])
             - float(ref["frac_sel_2x_worse"]["mean"]))
    delta = float(pair["delta"])                   # refined - shipped; < 0 better
    sep = bool(pair["separated"])
    if sep and -delta >= PREREG_THRESHOLDS["free_win_m"]:
        verdict = "S1 FREE WIN"
    elif sep and -delta > 0:
        verdict = "INDETERMINATE"
    elif (not sep) and gap2x >= PREREG_THRESHOLDS["frac_2x_gap_m"]:
        verdict = "S1 NEEDS TRAINING"
    elif not sep:
        verdict = "S1 DEAD"
    else:
        verdict = "S1 WORSE THAN SHIPPED (not a registered branch — report it)"
    return {"experiment": "E-SEL-0", "status": "MEASURED",
            "refined": {k: v for k, v in ref.items() if not k.startswith("_")},
            "paired_refined_minus_shipped_ade": pair,
            "frac_2x_gap_shuffled_minus_refined": round(gap2x, 4),
            "verdict": verdict, "thresholds": PREREG_THRESHOLDS,
            "⚠️": ("LOWER BOUND on S1: these confidences were never trained as "
                   "a ranker. A null bounds how much of S1 is free; it does not "
                   "falsify S1.")}


def e_sel_1(d: dict, de_all, eid, cons_score=None) -> dict:
    """§6.2 — does the CONSEQUENCE carry candidate-discriminating information?

    Spearman rho between ``-||law_head([pooled, fan_i]) - z_{t+5}||^2`` and the
    candidate's true ADE, against C-shuffled (the same statistic on a permuted
    candidate axis). If rho is not separated from the shuffled control, S3 is a
    mechanism without information and must not consume a GPU-day.
    """
    if cons_score is None:
        return {
            "experiment": "E-SEL-1",
            "status": "BLOCKED — not computable from the bank",
            "reason": ("needs (a) `pooled` per window, (b) REF-C's trained "
                       "`law_head`, and (c) z_{t+5} — the no_grad encoding of "
                       "the frame 5 steps ahead. refc_rerank.dump stores none "
                       "of the three; all require a checkpoint AND the val "
                       "episode cache at REF-C's 256x256 raster."),
            "n_windows": 0,
            "note": ("law_head is [pooled, traj] -> pooled_{t+5} and is a "
                     "single MLP evaluation per candidate, so this is cheap "
                     "ONCE the two inputs exist — it is an access problem, not "
                     "a compute one."),
            "⛔": ("imagine_probes is NOT a fallback: "
                   "IMAGINATION_HAS_CANDIDATE_AXIS = False (32 tokens identical "
                   "for all 256 candidates) and REF-C's law_head cannot be "
                   "iterated. The prereg refuses that port from source."),
        }
    from scipy.stats import spearmanr                           # noqa: PLC0415
    per_win = np.array([spearmanr(cons_score[i], -de_all[i].numpy()).statistic
                        for i in range(de_all.shape[0])], dtype=np.float64)
    rng = np.random.default_rng(20260803)
    shuf = np.array([spearmanr(rng.permutation(cons_score[i]),
                               -de_all[i].numpy()).statistic
                     for i in range(de_all.shape[0])], dtype=np.float64)
    boot = _boot(per_win, eid)
    pair = _paired(per_win, shuf, eid)
    rho = float(boot["mean"])
    live = bool(pair["separated"]) and abs(rho) >= PREREG_THRESHOLDS["rho_min"]
    return {"experiment": "E-SEL-1", "status": "MEASURED",
            "spearman_rho": boot, "paired_rho_minus_shuffled": pair,
            "verdict": "S3 LIVE" if live else "S3 DEAD",
            "⛔_if_dead": ("drop S3 from the arm and say so; do NOT reframe it "
                          "as 'needs training to emerge'")}


# =========================================================================== #
# driver                                                                      #
# =========================================================================== #

def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not k.startswith("_per_window")}
    if isinstance(o, (list, tuple)):
        return [_clean(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, torch.Tensor):
        return o.tolist() if o.numel() < 64 else f"<tensor {tuple(o.shape)}>"
    return o


def run(fan_path: str, out_dir: str, *, arm: str | None = None,
        ckpt: str | None = None, data: str | None = None,
        controls: str = "identity,shuffled,raster,oracle-floor") -> dict:
    t0 = time.time()
    want = {c.strip() for c in controls.split(",") if c.strip()}
    d = load_fan(fan_path)
    arm = arm or Path(fan_path).stem.replace("fan_", "")
    eid = list(d["eid"])
    de_all = candidate_ade(d["fan"], d["gt"])
    de_or = de_all.min(1).values

    res: dict = {
        "what": "E-SEL-0 / E-SEL-1 — the 0-GPU-day probe that gates the D-SEL retrain",
        "prereg": prereg_provenance(),
        "arm": arm,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "estimator": {
            "point_and_interval": "episode_cluster_bootstrap",
            "paired": "paired_episode_cluster_bootstrap",
            "resampling_unit": "episode",
            "n_boot": N_BOOT,
            "⛔": ("overlapping_holdout_se is NEVER called — it is not a "
                   "jackknife, not a valid SE, and it biases the POINT "
                   "estimate bidirectionally (mean-of-split-means)"),
        },
        "fan_bank": {
            "path": d["_path"], "sha256": d["_sha256"],
            "n_windows": int(d["fan"].shape[0]),
            "n_anchors": int(d["fan"].shape[1]),
            "n_episodes": len(set(eid)),
            "wp_steps": list(d["wp_steps"]),
            "decode_ckpt": d.get("ckpt"), "decode_ckpt_step": d.get("ckpt_step"),
            "diffusion_steps": d.get("steps"),
            "nav_mode": d.get("nav_mode", "follow_constant (implied)"),
            "stores_refined_logits": d["_has_refined_logits"],
            "stores_pooled": d["_has_pooled"],
        },
        "run_mode": "full" if (ckpt and data) else "fan-only",
        "controls": {}, "rankers": {}, "families": {},
    }

    # ---- controls FIRST (prereg §6.0), non-negotiable ---------------------- #
    if "raster" in want:
        res["controls"]["C-raster"] = control_raster(
            d, frames_hw=tuple(d["raster"]) if "raster" in d else None)
    if "identity" in want and d["_has_refined_logits"]:
        res["controls"]["C-identity-vs-bank"] = control_identity_vs_bank(
            d, _REPO / "taniteval" / "results" / f"fan_{arm}.pt", de_all)
    if "identity" in want:
        c = control_identity(d, de_all, arm)
        res["controls"]["C-identity"] = c
        if c["status"].startswith("FAIL"):
            raise SystemExit(f"C-identity FAILED: {json.dumps(c, indent=1)}")
    if "oracle-floor" in want:
        c = control_oracle_floor(d, de_or, arm)
        res["controls"]["C-oracle-floor"] = c
        if c["status"].startswith("FAIL"):
            raise SystemExit(f"C-oracle-floor FAILED: {json.dumps(c, indent=1)}")
    shuf = shuffled_control(de_all, de_or, eid) if "shuffled" in want else None
    if shuf is not None:
        res["controls"]["C-shuffled"] = shuf

    # ---- the rankers the bank can express --------------------------------- #
    shipped = ranker_block(de_all, de_or, d["sel"], eid, tag="shipped (t=0 classifier)")
    oracle = ranker_block(de_all, de_or, de_all.argmin(1), eid, tag="oracle-in-fan")
    res["rankers"]["shipped"] = shipped
    res["rankers"]["oracle"] = oracle
    if shuf is not None:
        res["rankers"]["shuffled"] = shuf
        res["rankers"]["paired_shipped_minus_shuffled_ade"] = _paired(
            shipped["_per_window_ade"].numpy(), shuf["_per_window_ade"].numpy(), eid)
        res["rankers"]["paired_shipped_minus_shuffled_frac2x"] = _paired(
            shipped["_per_window_2x"].numpy(), shuf["_per_window_2x"].numpy(), eid)
        res["rankers"]["headroom_shipped_vs_oracle"] = _paired(
            shipped["_per_window_ade"].numpy(), oracle["_per_window_ade"].numpy(), eid)

    # ---- S2 telemetry on THIS arm ----------------------------------------- #
    refined = d.get("refined_logits")
    res["S2_reachability"] = reachability_block(d, de_all, de_or, eid)

    # ---- S1 x S2 — the combination `dsel-nocons` actually trains ----------- #
    # ⭐ NOT a new experiment: prereg §7's `dsel-nocons` arm is S1+S2+S4, so
    # "does the reachability band rescue the refined ranker?" is a question the
    # registered arm asks and the bank can answer for free. It matters because
    # 72-74 % of the fan is unflyable: a ranker that is worse than the shipped
    # one MIGHT be worse only because it prefers those, in which case S2 removes
    # the failure mode rather than merely paying for compute.
    if refined is not None:
        try:
            from tanitad.refs import refc_select as sl
            mask = sl.reachability_mask(d["fan"], d["v0"].to(d["fan"].dtype),
                                        accel_max=2.5, horizon_s=2.0)
            keep = mask | (~mask.any(1))[:, None]
            r2 = ranker_block(
                de_all, de_or,
                torch.as_tensor(refined).masked_fill(~keep, float("-inf")).argmax(1),
                eid, tag="refined + S2 reachability band")
            s2 = ranker_block(
                de_all, de_or,
                d["logits"].masked_fill(~keep, float("-inf")).argmax(1),
                eid, tag="shipped + S2 reachability band")
            res["rankers"]["refined_plus_reach"] = r2
            res["rankers"]["shipped_plus_reach"] = s2
            res["rankers"]["paired_refinedreach_minus_shipped_ade"] = _paired(
                r2["_per_window_ade"].numpy(),
                shipped["_per_window_ade"].numpy(), eid)
            bare = ranker_block(de_all, de_or,
                                torch.as_tensor(refined).argmax(1), eid,
                                tag="refined (no band)")
            res["rankers"]["paired_refinedreach_minus_refined_ade"] = _paired(
                r2["_per_window_ade"].numpy(),
                bare["_per_window_ade"].numpy(), eid)
        except Exception as exc:                               # pragma: no cover
            res["rankers"]["refined_plus_reach"] = {
                "status": f"UNAVAILABLE — {type(exc).__name__}: {exc}"}

    # ---- four families ----------------------------------------------------- #
    idx_or = de_all.argmin(1)
    res["families"]["shipped"] = families_block(d, d["sel"], eid, shipped,
                                                tag="shipped")
    res["families"]["oracle"] = families_block(d, idx_or, eid, oracle,
                                               tag="oracle-in-fan")
    # ⭐ THE FOUR-FAMILY DECOMPOSITION OF THE SELECTION GAP. What would a PERFECT
    # reranker of this exact fan buy, per family? It is the pre-computed UPPER
    # BOUND on anything S1 can deliver, on the same windows, paired — and it is
    # the table prereg §7.1 would judge the retrain against.
    res["families"]["paired_oracle_minus_shipped"] = family_paired(
        d, idx_or, d["sel"], eid, tag="oracle-in-fan MINUS shipped")
    if shuf is not None:
        res["families"]["paired_shipped_minus_shuffled"] = family_paired(
            d, d["sel"], torch.randint(0, de_all.shape[1], (de_all.shape[0],),
                                       generator=torch.Generator().manual_seed(20260803)),
            eid, tag="shipped MINUS one shuffled draw (control)")

    # ---- the experiments --------------------------------------------------- #
    res["E-SEL-0"] = e_sel_0(d, de_all, de_or, eid, shipped, shuf, refined)
    res["E-SEL-1"] = e_sel_1(d, de_all, eid, d.get("cons_score"))

    res["wall_s"] = round(time.time() - t0, 1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"esel_probe_{arm}.json"
    dest.write_text(json.dumps(_clean(res), indent=1, ensure_ascii=False),
                    encoding="utf-8")
    print(f"[esel] {arm}: {res['fan_bank']['n_windows']} windows x "
          f"{res['fan_bank']['n_anchors']} anchors -> {dest} "
          f"({res['wall_s']}s)", flush=True)
    print(f"[esel]   E-SEL-0 {res['E-SEL-0']['status']} · "
          f"E-SEL-1 {res['E-SEL-1']['status']}", flush=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fan", required=True, help="banked fan_refc-*.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--ckpt", default=None,
                    help="REF-C checkpoint; unlocks E-SEL-0's treatment leg")
    ap.add_argument("--data", default=None,
                    help="val episode cache at REF-C's [T,9,256,256] raster")
    ap.add_argument("--controls", default="identity,shuffled,raster,oracle-floor")
    a = ap.parse_args(argv)
    if bool(a.ckpt) ^ bool(a.data):
        ap.error("--ckpt and --data must be given together: a decode needs both "
                 "the weights and the val episodes at REF-C's raster")
    run(a.fan, a.out, arm=a.arm, ckpt=a.ckpt, data=a.data, controls=a.controls)
    return 0


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())
