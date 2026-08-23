"""RL1 — build the ladder TARGETS on the canonical 881 val40 windows, gated.

JOB 2 of the REF-C architecture review: the C104 readout ladder has never been
run on REF-C's encoder — the one encoder attached to the programme's
demonstrated vision-grounded driving (REF-C evaluates with ``nav_cmd=None``).
The banked REF-C latents (`…/2026-08-04-lambda-findability/raw/
latents_refc-{base,xl}-30k-ep.pt`) carry the VISION-ONLY ``pooled`` /
``pooled_seq`` surfaces on exactly these 881 windows, so the probe needs NO
checkpoint, NO frames and NO new inference — but it needs TARGETS, and this
file builds them with the identity gates that make the rows trustworthy.

⛔ EVAL TIER: T0-DIAGNOSTIC. A frozen-latent linear readout is a world-model /
representation diagnostic and never driving performance (EVAL_DOCTRINE.md).

WINDOW IDENTITY — asserted, never assumed (every gate below fails loud):
  G1  the poses-only val40 view matches `manifest_EVALPOD_val40.json` by
      per-episode sha256 over the poses bytes, 40/40.
  G2  the banked latent dump's per-window ``eid`` partition is 22 windows on 39
      episodes + 23 on one, total 881, and equals the rebuilt
      `lead_source.window_last_indices` grid per episode.
  G3  the 2026-08-04 lead block (`val40_lead_block.npz`) row-aligns with the
      latent dump: eid partition equal, ``speeds`` vs the dump's ``v0`` within
      5 mm/s (they are independently derived — egomotion vs actions channel).
  G4  the dump's own ``controls_vs_bank`` flags (fan/gt/sel/v0 bit-identity
      against the canonical fan bank) are all True and instrument_fail is [].
  G5  the lead re-derived HERE (select_lead_causal at the registered t0) agrees
      with the banked block: state agreement and |Δgap| reported; the run
      REFUSES if state agreement < 99 % or median |Δgap| > 1 cm.

TARGET DEFINITIONS (source of each, and the one declared deviation):
  ego_v0          the dump's banked ``v0`` (the model's own input channel) —
                  the ladder's sanity-anchor rung, LL semantics unchanged.
  lead_present    banked block ``state == 'LEAD'`` (lead_source three-state;
                  NO_LABEL windows are excluded from the mask, never counted
                  as clear road).
  lead_gap        banked block ``gap0_m`` (= along − size_x/2, the
                  lead_state_gate convention, rig origin to rear face).
  lead_closing    −v_rel_x of the selected lead AT t0, instantaneous:
                  lead world velocity from the two nearest DISTINCT cuboid
                  samples bracketing t0 (fallback: nearest distinct pair
                  within ±``CLOSING_STALE_S``), ego world velocity from the
                  egomotion vx/vy interpolated at t0, both projected on the
                  ego heading at t0. This matches ll1_ladder's
                  "−v_rel_x of the GT lead" semantics; the ladder computed it
                  from the join's banked per-track rates, we compute it from
                  the same obstacle.offline samples directly. DECLARED
                  deviation: none in quantity, only in plumbing.
                  ⚠️ C123: lead_closing failed K1 on every un-planted arm
                  including C104's own — expect degeneracy, stamp it, never
                  quote it bare.
  n_agents_grid   count of DISTINCT obstacle tracks (all label classes) with a
                  cuboid sample within ±``AGENT_TOL_S`` of t0 whose rig-frame
                  centre lies in the sp1 grid (0 < cx <= 60, |cy| <= 16 —
                  `sp1_cache_latents.py:242`'s keep rule, re-stated).
                  DECLARED reimplementation: sp1 counted rows of the banked
                  join at the reference frame; we sample obstacle.offline
                  directly at t0. Same grid, same class-agnostic rule.
  n_agents_any    the same count with no grid filter (the "whole frame"
                  analogue of the ladder's n_agents_all; PhysicalAI cuboids
                  have no image-plane footprint here, so "in frame" is not
                  computable from the labels — stated, not fudged).

Provenance of every input is written into the gates JSON. NO episode is
selected, added, removed or re-ordered; the banked window set is read verbatim.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[5]
for _p in (_REPO / "taniteval", _REPO / "stack", _REPO / "stack" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lead_state_gate import VEHICLE_CLASSES, quaternion_yaw  # noqa: E402
from taniteval.lead_source import (LEAD, NO_LABEL, NO_LEAD,  # noqa: E402
                                   register_poses_to_time, select_lead_causal,
                                   window_last_indices)

AGENT_TOL_S = 0.15          # nearest-sample tolerance for the agent count
CLOSING_STALE_S = 0.50      # max separation of the two samples the lead
#                             velocity is differenced over (lead_source's
#                             MAX_STALE_S, restated)
GRID_X_FWD_M = 60.0         # sp1_cache_latents GRID (x_fwd_m, y_half_m)
GRID_Y_HALF_M = 16.0


def _p(*a):
    print(*a, flush=True)


def sha256_poses(t: torch.Tensor) -> str:
    return hashlib.sha256(np.ascontiguousarray(t.numpy()).tobytes()).hexdigest()


def _member(z: zipfile.ZipFile, clip: str):
    for n in z.namelist():
        if n.endswith(".parquet") and n.rsplit("/", 1)[-1].startswith(clip):
            return n
    return None


def _read_parquet(zpath: Path, clip: str):
    if not zpath.exists():
        return None
    with zipfile.ZipFile(zpath) as z:
        n = _member(z, clip)
        if n is None:
            return None
        return pd.read_parquet(io.BytesIO(z.read(n)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses-view", required=True,
                    help="canonical val40 poses-only view (ep_000*.pt)")
    ap.add_argument("--manifest", required=True,
                    help="manifest_EVALPOD_val40.json (sha256 per episode)")
    ap.add_argument("--latents", required=True,
                    help="latents_refc-base-30k-ep.pt (the eid/v0 authority)")
    ap.add_argument("--latents-xl", default=None,
                    help="optional XL dump; v0/eid cross-checked if given")
    ap.add_argument("--lead-block", required=True,
                    help="val40_lead_block.npz (2026-08-04, row-aligned)")
    ap.add_argument("--data", required=True,
                    help="physicalai data root (r0/phase0_selection.parquet + "
                         "labels/{egomotion,obstacle.offline})")
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-gates", required=True)
    a = ap.parse_args(argv)
    t_start = time.time()

    gates: dict = {"_evidence_class":
                   "MEASURED (ours; label join on the canonical banked windows)",
                   "eval_tier": "T0-DIAGNOSTIC",
                   "inputs": {k: str(getattr(a, k)) for k in
                              ("poses_view", "manifest", "latents",
                               "lead_block", "data")}}

    # ---- G1: poses view == canonical manifest, 40/40 ----------------------
    man = json.loads(Path(a.manifest).read_text("utf-8"))
    view = Path(a.poses_view)
    eps, sha_fail = [], []
    for e in man["episodes"]:
        d = torch.load(view / e["file"], map_location="cpu",
                       weights_only=False)
        if sha256_poses(d["poses"]) != e["poses_sha256"]:
            sha_fail.append(e["file"])
        eps.append({"file": e["file"], "episode_id": int(d["episode_id"]),
                    "poses": d["poses"].numpy().astype(np.float64)})
    gates["G1_poses_sha256"] = {"n": len(eps), "fail": sha_fail}
    if sha_fail:
        raise SystemExit(f"[rl1] G1 FAILED — poses drift: {sha_fail}")

    # ---- the latent dump: the eid/v0 authority ------------------------------
    lat = torch.load(a.latents, map_location="cpu", weights_only=False)
    eid = np.asarray(lat["eid"], dtype=np.int64)
    v0 = lat["v0"].numpy().astype(np.float64)
    gt = lat["gt"].numpy().astype(np.float64)              # [881, 4, 2]
    n = eid.size
    cvb = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
           for k, v in lat.get("controls_vs_bank", {}).items()}
    gates["G4_dump_controls_vs_bank"] = cvb
    gates["G4_instrument_fail"] = list(lat.get("instrument_fail", []))
    bad = [k for k, v in cvb.items() if isinstance(v, bool) and not v]
    if bad or gates["G4_instrument_fail"]:
        raise SystemExit(f"[rl1] G4 FAILED — dump not clean: {bad}, "
                         f"{gates['G4_instrument_fail']}")
    if a.latents_xl:
        lx = torch.load(a.latents_xl, map_location="cpu", weights_only=False)
        same = (np.array_equal(np.asarray(lx["eid"]), eid)
                and float(np.abs(lx["v0"].numpy() - v0).max()) == 0.0)
        gates["G4b_xl_dump_same_rows"] = bool(same)
        if not same:
            raise SystemExit("[rl1] G4b FAILED — XL dump rows differ from base")

    # ---- G2: eid partition == rebuilt window grid ---------------------------
    counts = np.bincount(eid, minlength=len(eps))
    rebuilt = np.array([window_last_indices(e["poses"].shape[0]).size
                        for e in eps])
    gates["G2_eid_partition"] = {
        "dump_counts": counts.tolist(), "rebuilt_counts": rebuilt.tolist(),
        "equal": bool((counts == rebuilt).all()), "total": int(counts.sum())}
    if not gates["G2_eid_partition"]["equal"] or counts.sum() != n:
        raise SystemExit("[rl1] G2 FAILED — window grid mismatch")

    # ---- G3: lead block row-aligns ------------------------------------------
    blk = np.load(a.lead_block, allow_pickle=True)
    blk_eid = np.array([int(s.split("_")[1]) for s in blk["eid"]])
    dv = float(np.abs(blk["speeds"] - v0).max())
    gates["G3_block_alignment"] = {
        "eid_equal": bool((blk_eid == eid).all()),
        "speeds_vs_dump_v0_max_abs": dv, "tol": 0.005}
    if not gates["G3_block_alignment"]["eid_equal"] or dv > 0.005:
        raise SystemExit(f"[rl1] G3 FAILED — block misaligned (max dv {dv})")
    state = blk["state"].astype(str)
    gap0 = blk["gap0_m"].astype(np.float64)

    # ---- the clip mapping (episode_id -> UUID prefix -> selection row) ------
    sel = pd.read_parquet(Path(a.data) / "r0" / "phase0_selection.parquet")
    ids = sel["clip_id"].astype(str).tolist()
    chunk_of = dict(zip(sel["clip_id"].astype(str), sel["chunk"].astype(int)))
    for e in eps:
        pref = int(e["episode_id"]).to_bytes(4, "big").decode("ascii",
                                                              "replace")
        m = [c for c in ids if c.startswith(pref)]
        if len(m) != 1:
            raise SystemExit(f"[rl1] clip mapping ambiguous for "
                             f"{e['file']}: {len(m)} matches")
        e["clip"], e["chunk"] = m[0], chunk_of[m[0]]
    gates["clip_mapping"] = {e["file"]: e["clip"] for e in eps}

    # ---- the join: per episode, per window ----------------------------------
    y_closing = np.full(n, np.nan)
    ok_closing = np.zeros(n, dtype=bool)
    y_nag_grid = np.full(n, np.nan)
    y_nag_any = np.full(n, np.nan)
    ok_nag = np.zeros(n, dtype=bool)
    my_state = np.array(["NO_LABEL"] * n, dtype=object)
    my_gap = np.full(n, np.nan)
    reg_residuals, grid_dts = [], []
    row0 = 0
    for i, e in enumerate(eps):
        poses = e["poses"]
        last = window_last_indices(poses.shape[0])
        n_win = last.size
        rows = slice(row0, row0 + n_win)
        row0 += n_win
        ego_df = _read_parquet(
            Path(a.data) / "labels" / "egomotion"
            / f"egomotion.chunk_{e['chunk']:04d}.zip", e["clip"])
        obs_df = _read_parquet(
            Path(a.data) / "labels" / "obstacle.offline"
            / f"obstacle.offline.chunk_{e['chunk']:04d}.zip", e["clip"])
        if ego_df is None:
            _p(f"  [rl1] ep{i:05d}: NO egomotion — rows stay NO_LABEL")
            continue
        t = ego_df["timestamp"].to_numpy(np.float64) / 1e6
        o = np.argsort(t)
        g = lambda c: ego_df[c].to_numpy(np.float64)[o]     # noqa: E731
        ego_t = t[o]
        ego_x, ego_y = g("x"), g("y")
        ego_vx, ego_vy = g("vx"), g("vy")
        ego_yaw = np.unwrap(quaternion_yaw(g("qx"), g("qy"), g("qz"),
                                           g("qw")))
        reg = register_poses_to_time(poses[:, :2], ego_t, ego_x, ego_y)
        reg_residuals.append(float(reg["residual_m"]["median"]))
        grid_dts.append(float(reg["grid_dt_s"]))
        t0s = np.asarray(reg["t_s"])[last]

        if obs_df is None or obs_df.empty:
            _p(f"  [rl1] ep{i:05d}: no obstacle.offline — NO_LABEL")
            continue
        tcol = ("timestamp_us" if "timestamp_us" in obs_df.columns
                else "timestamp")
        ot = obs_df[tcol].to_numpy(np.float64) / 1e6
        oo = np.argsort(ot)
        og = lambda c: obs_df[c].to_numpy()[oo]             # noqa: E731
        ot = ot[oo]
        otrk = og("track_id").astype(str)
        ocx = og("center_x").astype(np.float64)
        ocy = og("center_y").astype(np.float64)
        osx = og("size_x").astype(np.float64)
        oveh = np.isin(og("label_class").astype(str),
                       np.asarray(VEHICLE_CLASSES))
        span_lo, span_hi = float(ot.min()), float(ot.max())

        for w, t0 in enumerate(t0s):
            ridx = rows.start + w
            if not (span_lo <= t0 <= span_hi):
                continue                                  # stays NO_LABEL
            trk, gap, sx = select_lead_causal(ot, otrk, ocx, ocy, osx, oveh,
                                              float(t0))
            my_state[ridx] = LEAD if trk is not None else NO_LEAD
            my_gap[ridx] = gap
            # ---- n_agents at t0 (all classes) -----------------------------
            near = np.abs(ot - t0) <= AGENT_TOL_S
            n_any = n_grid = 0
            if near.any():
                k = np.flatnonzero(near)
                bytrk: dict = {}
                for j in k:                       # nearest sample per track
                    prev = bytrk.get(otrk[j])
                    if prev is None or (abs(ot[j] - t0)
                                        < abs(ot[prev] - t0)):
                        bytrk[otrk[j]] = j
                n_any = len(bytrk)
                n_grid = sum(1 for j in bytrk.values()
                             if 0.0 < ocx[j] <= GRID_X_FWD_M
                             and abs(ocy[j]) <= GRID_Y_HALF_M)
            y_nag_any[ridx] = float(n_any)
            y_nag_grid[ridx] = float(n_grid)
            ok_nag[ridx] = True
            # ---- instantaneous closing of the selected lead ----------------
            if trk is None:
                continue
            m = otrk == trk
            tt, xx, yy = ot[m], ocx[m], ocy[m]
            if tt.size < 2:
                continue
            jr = int(np.searchsorted(tt, t0, side="right"))
            jl = jr - 1
            if jl < 0:
                jl, jr = 0, 1
            elif jr >= tt.size:
                jl, jr = tt.size - 2, tt.size - 1
            if tt[jr] - tt[jl] <= 1e-6 or tt[jr] - tt[jl] > CLOSING_STALE_S:
                continue
            # lead world positions at its own two sample times
            def _world(j):
                ex = np.interp(tt[j], ego_t, ego_x)
                ey = np.interp(tt[j], ego_t, ego_y)
                yaw = np.interp(tt[j], ego_t, ego_yaw)
                c, s = np.cos(yaw), np.sin(yaw)
                return (ex + xx[j] * c - yy[j] * s,
                        ey + xx[j] * s + yy[j] * c)
            (wx0, wy0), (wx1, wy1) = _world(jl), _world(jr)
            dt = tt[jr] - tt[jl]
            vlx, vly = (wx1 - wx0) / dt, (wy1 - wy0) / dt
            vex = np.interp(t0, ego_t, ego_vx)
            vey = np.interp(t0, ego_t, ego_vy)
            yaw0 = float(np.interp(t0, ego_t, ego_yaw))
            closing = -((vlx - vex) * np.cos(yaw0)
                        + (vly - vey) * np.sin(yaw0))
            y_closing[ridx] = float(closing)
            ok_closing[ridx] = True

    # ---- G5: the re-derived lead agrees with the banked block ---------------
    agree = (my_state.astype(str) == state)
    both_lead = (my_state.astype(str) == LEAD) & (state == LEAD)
    dgap = np.abs(my_gap[both_lead] - gap0[both_lead])
    gates["G5_lead_rederivation"] = {
        "state_agreement": float(agree.mean()),
        "n_both_lead": int(both_lead.sum()),
        "gap_abs_diff_median_m": (float(np.median(dgap)) if dgap.size
                                  else None),
        "gap_abs_diff_max_m": float(dgap.max()) if dgap.size else None,
        "registration_residual_median_m": float(np.median(reg_residuals)),
        "grid_dt_s_range": [float(np.min(grid_dts)),
                            float(np.max(grid_dts))]}
    if agree.mean() < 0.99 or (dgap.size and np.median(dgap) > 0.01):
        raise SystemExit(f"[rl1] G5 FAILED — lead re-derivation drifts: "
                         f"{gates['G5_lead_rederivation']}")

    # ---- assemble targets ---------------------------------------------------
    lead_mask = state == LEAD
    labeled = state != NO_LABEL
    tgt = {
        "ego_v0": (v0, np.ones(n, dtype=bool)),
        "lead_present": (lead_mask.astype(np.float64), labeled),
        "lead_gap": (np.where(np.isfinite(gap0), gap0, 0.0), lead_mask
                     & np.isfinite(gap0)),
        "lead_closing": (np.where(ok_closing, y_closing, 0.0),
                         lead_mask & ok_closing),
        "n_agents_grid": (np.where(ok_nag, y_nag_grid, 0.0), ok_nag),
        "n_agents_any": (np.where(ok_nag, y_nag_any, 0.0), ok_nag),
    }
    out = {"eid": eid, "v0": v0, "state": state.astype("U8"), "gt": gt}
    summary = {}
    for k, (y, ok) in tgt.items():
        out[f"y_{k}"], out[f"ok_{k}"] = y, ok
        yv = y[ok]
        summary[k] = {"n": int(ok.sum()),
                      "mean": float(yv.mean()) if ok.any() else None,
                      "sd": float(yv.std()) if ok.any() else None,
                      "min": float(yv.min()) if ok.any() else None,
                      "max": float(yv.max()) if ok.any() else None}
    gates["targets_summary"] = summary
    gates["state_counts"] = {s: int((state == s).sum())
                             for s in (LEAD, NO_LEAD, NO_LABEL)}
    gates["definitions"] = {
        "lead_closing": "-v_rel_x at t0; lead world velocity from the two "
                        "nearest distinct cuboid samples bracketing t0 "
                        f"(max separation {CLOSING_STALE_S}s), ego velocity "
                        "from egomotion vx/vy at t0, projected on ego heading "
                        "at t0; +ve = closing",
        "n_agents_grid": f"distinct tracks, ANY class, nearest sample within "
                         f"±{AGENT_TOL_S}s of t0, 0<cx<={GRID_X_FWD_M} & "
                         f"|cy|<={GRID_Y_HALF_M} (sp1 grid)",
        "n_agents_any": "same, no grid filter",
        "lead_gap": "banked val40_lead_block gap0_m (along - size_x/2)",
        "lead_present": "banked block state=='LEAD'; NO_LABEL excluded",
        "ego_v0": "banked latent-dump v0 (the model's own input channel)"}
    gates["wall_s"] = round(time.time() - t_start, 1)

    np.savez_compressed(a.out_npz, **out)
    Path(a.out_gates).write_text(json.dumps(gates, indent=1), "utf-8")
    _p(f"[rl1] targets -> {a.out_npz}")
    _p(f"[rl1] gates   -> {a.out_gates}")
    _p(json.dumps({k: v["n"] for k, v in summary.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
