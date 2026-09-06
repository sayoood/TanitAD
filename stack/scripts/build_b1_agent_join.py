"""B1 EVAL `obstacle.offline` -> agent JOIN, keyed on (clip_id, RAW v2ep frame).

WP-6 (agent conditioning) needs agents on the corpus the v7 arms actually
evaluate: the **v7.2 EVAL split of B1**. The existing joins do not serve it —
MEASURED 2026-09-06, and the fractions are the whole point:

    union(train2400 join, val40 join, lead130 subset) covers  199 / 4,719  B1 clips
    train2400_agents.jsonl.xz          n B1 clips =  193 / 4,719   (TRAIN corpus)
    val40_agents.jsonl                 n B1 clips =    6 /    39
    EVAL141 (what refav1 scores on)    n joined   =   11 /   141

So 130 of the 141 B1-EVAL clips had NO agent join at all. That is the WP-6 blocker.

⛔⛔ THE INDEX SPACE IS THE TRAP, AND IT IS WHY THIS IS A SEPARATE BUILDER.
``build_obstacle_join.py`` emits ``frame_idx`` in EPISODE index space — the
POST-n_stack-trim provider index ``i - (n_stack - 1)``. The B1 EVAL lead block
(``b1_eval_lead_block.npz``, 29,556 rows / 147 clips) emits ``frame`` in the RAW
v2ep index ``i in [0, n_target)``, because the refav1 loader reads RAW poses with
no trim (``build_lead_block_b1.py`` docstring, FRAME INDEX SPACE). With n_stack 3
the two spaces differ by 2 frames -- about 0.2 s, ~2.7 m of lead displacement at
13.6 m/s. They are NOT interchangeable, and a same-named ``frame_idx`` would
invite exactly the silent mis-join this programme has already paid for twice.

  => this builder emits ``frame`` for the RAW index AND ``frame_idx`` for the
     post-trim one, in the SAME line, because the corpus has two consumers that
     disagree (see join_clip_raw). The RAW index never travels under the name
     ``frame_idx``, so a reader that only skims the schema cannot confuse them.
     ⚠️ This paragraph read "emits ``frame``, NEVER ``frame_idx``" until
     2026-09-06 and CONTRADICTED the code, which has emitted both since the
     first build -- a docstring that would have talked a reader out of the key
     the training-side consumer actually needs.

ALIGNMENT IS PROVEN, NOT ASSERTED. The banked lead block is ground truth for
``(clip_id, frame) -> t0_s`` and ego speed, so every build cross-checks against
it AND runs the deliberate mis-join(+1) control that must separate:

    MEASURED 2026-09-06 (6 clips, probe): TRUE max|dt| 6.05e-04 s vs
    MIS+1 max|dt| 1.013e-01 s  =>  167.5x separation; and the v2ep records'
    ``poses[:, 3]`` reproduces the block's ``speeds`` to max|dv| 1.10e-05 m/s.

A run whose control does NOT separate REFUSES to write. An unproven join is
worse than no join: it silently conditions the model on the wrong agents.

LINE SCHEMA (one line per LABELLED RAW frame)::

    {"clip_id": str, "frame": int, "t_s": float,
     "agents": [{"cx": f, "cy": f, "yaw": f, "l": f, "w": f, "occ": 0|1,
                 "track_id": str, "cls": str}]}

Geometry, visibility and the NO_LABEL rule are IMPORTED from
``build_obstacle_join`` -- never re-implemented, so the two joins cannot drift:

  * ``cx/cy/yaw`` in the per-frame EGO frame (+x fwd, +y LEFT), ``l = size_x``,
    ``w = size_y``; composition rig@sample -> world -> ego@frame.
  * ``occ`` 0 = agent centre inside the 120 deg front-camera field, 1 = outside
    while the track continues. It IS ``bev_raster.fov_mask`` at agent-centre
    granularity (P4_PREDICATE_IDENTITY) -- a field mask, not an independent label.
  * An ABSENT (clip, frame) line is NO_LABEL. An EMPTY ``agents`` list IS a
    label: labelled clear. The two must never be conflated.

Usage (dev box, CPU only, no GPU -- the A40 is running refcv5 and is untouched)::

  set PYTHONPATH=<repo>/stack
  python stack/scripts/build_b1_agent_join.py ^
      --eps-dir      C:/Users/Admin/tanitad-data/refav1-eval141/eps ^
      --ego-dir      C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo ^
      --obstacle-dir C:/Users/Admin/tanitad-data/physicalai/labels/obstacle_offline_b1eval ^
      --lead-block   "<repo>/TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz" ^
      --out          <out>/b1eval_agents.jsonl.xz

Writes ``<out>.meta.json`` (provenance + per-clip stats + the alignment proof)
and ends with one ``B1_JOIN_DONE {...}`` line. Every print is ASCII: this box is
cp1252 and a non-ASCII print has already truncated a banked artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parents[1]
for _p in (str(_SCRIPTS), str(_SCRIPTS.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _bootstrap_taniteval() -> None:
    """``taniteval`` sits beside ``stack/`` in the repo; pods also set PYTHONPATH."""
    try:
        import taniteval  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    for cand in (_REPO / "taniteval", _REPO.parent / "taniteval"):
        if (cand / "taniteval").is_dir() or (cand / "__init__.py").exists():
            if str(cand) not in sys.path:
                sys.path.append(str(cand))
            return


#: default key name. NOT ``frame_idx`` -- see the module docstring.
FRAME_KEY = "frame"

#: ⛔ THE GATE IS THE **SPEED** AGREEMENT, NOT THE TIME AGREEMENT.
#:
#: Both this join and the lead block carry the ego speed at the SAME frame index
#: (v2ep ``poses[:, 3]`` and the block's ``speeds``), and both are egomotion
#: interpolated at that frame -- so their agreement is the label-free proof that
#: the join hit the right clip AND the right frame (``refav1_arm.py`` :2077-2080).
#: Crucially it is INDEPENDENT of ``register_poses_to_time``'s fit, which is what
#: makes it the right instrument for a KEY-correctness question.
#:
#: Value adopted VERBATIM from ``refav1_arm.py:238`` -- the programme's own proven
#: constant for exactly this test, not a threshold invented here.
LEAD_SPEED_TOL_MPS = 1e-3

#: the mis-join(+1) control must beat the true join by at least this factor.
MIN_MISJOIN_SEPARATION = 20.0

#: ⛔ THE **TRAIN** GATE, AND WHY IT IS A DIFFERENT STATISTIC FROM THE EVAL ONE.
#:
#: The EVAL gate compares two INDEPENDENTLY PRODUCED speed arrays indexed by the
#: same frame (v2ep ``poses[:,3]`` and the banked block's ``speeds``). On B1
#: TRAIN neither exists: there is no lead block, and the poses are BUILT on the
#: camera grid, so ``poses[:,3]`` IS the reference's own speeds. Re-sampling one
#: side at the registered times to manufacture a difference does not rescue it --
#: MEASURED 2026-09-06 on the 141 EVAL clips, that variant reads true 1.191e-01
#: m/s against a mis-join(+1) of 2.134e-02 m/s, i.e. the CONTROL IS SMALLER THAN
#: THE TRUE VALUE (separation 0x). It had stopped measuring key correctness and
#: started measuring registration PRECISION -- the exact confusion the EVAL
#: package diagnosed for the 5e-3 s time gate, re-introduced in a new costume.
#:
#: ⭐ So the TRAIN gate asks the key-correctness question in ITS OWN UNITS: how
#: far, IN FRAMES, the join's time base (``register_poses_to_time``, POSITION)
#: sits from the corpus's own camera timestamp grid. A frame index is an
#: integer, so the decision boundary is 0.5 frames; these thresholds put a 2x
#: margin on each side of it, and the mis-join(+1) control must read ~1.0 frames
#: -- a KNOWN value, not merely a bigger one.
FRAME_ERR_MAX_TRUE = 0.25
FRAME_ERR_MIN_MISJOIN = 0.75

#: ⛔ THE GATE IS PER-CLIP, AND THE CORPUS BOUND IS WHAT STOPS THAT BEING A
#: GOALPOST MOVE.
#:
#: MEASURED 2026-09-06 on the first full B1 TRAIN build: over **4,434** probed
#: clips the frame error is median **0.0032**, p95 **0.0103**, p99 **0.0317**
#: frames -- and **ONE** clip (``065482b3``) reads **1.0506** frames with its
#: mis-join(+1) control at **1.04e-01 s**, i.e. ``mis/true = 0.981``: shifting
#: that clip by one frame is as good as not shifting it, which is what a
#: genuinely one-frame-off clip looks like. A global max over the corpus is
#: dominated by that single clip, so the build REFUSED all 4,433 good ones.
#:
#: ⭐ Refusing the corpus protects nothing -- the offender would simply be
#: dropped -- so the gate is applied PER CLIP: a failing clip is EXCLUDED and
#: NAMED with its numbers, exactly as a clip with no obstacle member is. This is
#: strictly STRICTER than the global form, which would have passed a corpus
#: containing a clip at 0.24 frames.
#:
#: ⚠️ And because "exclude the failures" is one edit away from "loosen the
#: gate", the corpus-level refusal stays: if more than this FRACTION of probed
#: clips fail, the build refuses, because a systematic one-frame defect would
#: hit many clips at once and must never be absorbed as "a few bad ones".
#: 0.005 is ~22 clips at this corpus size, ~200x the one offender MEASURED.
MAX_ALIGNMENT_EXCLUDED_FRAC = 0.005

#: ⚠️ REPORTED, NOT GATED -- and this distinction was MEASURED, not assumed.
#: A first cut gated on |t_s - block t0_s| <= 5e-3 s and REFUSED the 141-clip
#: build on ONE clip (fe764229, 1.04e-02 s). Diagnosis: that clip registers on
#: **23 probes** where healthy clips use 96, so its fitted time OFFSET is less
#: precise -- the error is a smooth ~7 ms bias across all frames, not a ~0.1 s
#: frame step. Its speed agreement is 2.25e-05 m/s, i.e. 44x INSIDE the proven
#: tolerance above, and ``lead_source``'s own documented accuracy is "worst
#: 25.9 ms over 500 clips". The 5e-3 s figure was invented here and was measuring
#: REGISTRATION PRECISION while the question asked was KEY CORRECTNESS.
MAX_TRUE_DT_S_REPORTED = 25.9e-3


# ============================================================================
# inputs
# ============================================================================
def read_raw_poses(v2ep_path: Path) -> tuple[str, np.ndarray, int]:
    """RAW poses ``[T, >=4]`` of one v2ep record -- NO n_stack trim.

    This is the one line that separates this builder from
    ``build_obstacle_join.corpus_first_clips``, which returns ``poses[n_stack-1:]``.
    Column 3 is the ego speed, and it is what the lead block's ``speeds`` holds --
    which is what makes the alignment proof label-free.
    """
    import torch
    d = torch.load(v2ep_path, map_location="cpu", weights_only=False, mmap=True)
    poses = d["poses"].float().numpy().astype(np.float64)
    cid = str(d.get("clip_id") or v2ep_path.name.split(".v2ep")[0])
    return cid, poses, int(d.get("n_stack", 0))


#: the ego clock. ``EgoTrack`` divides ``timestamp`` by 1e6, and ``signals_at``
#: interpolates against the RAW column -- so a camera grid in any other unit
#: would silently mis-scale the query. Refused rather than assumed.
EGO_CLOCK_UNIT = 1e6

_LEAD_BLOCK_MOD = None


def _lead_block_mod():
    """``taniteval/tools/build_lead_block_b1.py``, loaded BY PATH.

    ⚠️ ``import taniteval.tools.build_lead_block_b1`` does NOT work in this repo
    and the error does not say why: the OUTER ``taniteval/`` is a plain
    directory holding ``tools/``, while ``import taniteval`` resolves to the
    INNER package ``taniteval/taniteval/`` (the one with ``__init__.py``). So
    ``taniteval.tools`` is a ModuleNotFoundError even with the path bootstrapped
    -- the namespace-shadow trap this programme has already paid for. MEASURED
    here: 141/141 clips skipped with that message, caught only because the empty
    -artifact guard refused to publish the resulting zeros file.

    Loading by path keeps ONE source of truth for ``episode_grid`` /
    ``episode_poses`` -- the same functions the lead block was built with -- so
    the two builders cannot drift.
    """
    global _LEAD_BLOCK_MOD
    if _LEAD_BLOCK_MOD is None:
        import importlib.util

        p = _REPO / "taniteval" / "tools" / "build_lead_block_b1.py"
        if not p.is_file():
            raise SystemExit("[b1join] cannot find %s" % p)
        spec = importlib.util.spec_from_file_location("_b1_lead_block_src", str(p))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _LEAD_BLOCK_MOD = m
    return _LEAD_BLOCK_MOD


def read_reconstructed_poses(ts_path: Path, ego_df, n_stack: int):
    """RAW poses ``[T, 4]`` rebuilt from the camera timestamp grid + egomotion.

    ⭐ THIS IS WHAT MAKES A B1 **TRAIN** BUILD POSSIBLE AT ALL. ``read_raw_poses``
    needs ``*.v2ep.pt``; the B1 TRAIN epcache is ~161 GB (34 MB/episode x 4,713)
    and lives on Thor, not on this box -- while the only thing the join actually
    reads out of a 34 MB record is a ``[T, 4]`` pose array of ~6 KB.

    ⛔ EQUIVALENCE IS MEASURED, NOT ASSUMED. The banked grid cross-check
    (``.../2026-09-02-b1-eval-lead-block/raw/grid_crosscheck.json``, 20/20 clips)
    reports ``max_dxy_m 0.0``, ``max_dyaw_deg 0.0``, ``max_dv_mps 0.0`` -- the
    reconstruction is **float32-IDENTICAL** to the banked v2ep poses, not merely
    close, and ``stack/tests/test_refav1_lead_block.py`` pins it. The two paths
    are the same two functions (``episode_grid`` -> ``episode_poses``) that
    ``physicalai.build_episode`` itself uses.

    Returns ``(poses, t_grid_s, unit)``. ``n_stack`` is passed through because a
    reconstructed clip carries no record to read it from -- it is an OPERATOR
    input here, and it is recorded in the sidecar as such.
    """
    import pandas as pd
    _m = _lead_block_mod()
    episode_grid, episode_poses = _m.episode_grid, _m.episode_poses

    ts_df = pd.read_parquet(ts_path)
    tcol = next(c for c in ts_df.columns if "time" in c.lower())
    t_query, unit, _n_target = episode_grid(ts_df[tcol].to_numpy(np.float64))
    if float(unit) != EGO_CLOCK_UNIT:
        raise ValueError(
            "%s: camera clock unit %r != ego clock %r -- the grid would be "
            "queried against a different clock than the ego track was fitted "
            "in. REFUSING rather than mis-scaling silently."
            % (ts_path.name, unit, EGO_CLOCK_UNIT))
    poses = np.asarray(episode_poses(ego_df, t_query), dtype=np.float64)
    return poses, np.asarray(t_query, dtype=np.float64) / unit, float(unit)


def grid_reference(clip_id: str, ego_df, t_grid_s: np.ndarray) -> dict:
    """The alignment gate's reference when NO lead block exists for the corpus.

    ⭐ Same construction as ``b1_eval_lead_block.npz``'s ``t0_s`` / ``speeds``
    (``build_lead_block_b1`` main: ``episode_grid`` -> ``episode_poses``) -- the
    EVAL gate READ those two arrays out of a banked npz; this recomputes them
    from the same two inputs. The gate statistic, its tolerance and its
    mis-join(+1) control are unchanged.

    ⛔ WHY THE GATE IS NOT A TAUTOLOGY HERE, WHICH IS THE WHOLE RISK.
    The join's own frame->time map comes from ``register_poses_to_time``, which
    identifies time from POSITION against the egomotion track. This reference's
    comes from the CAMERA TIMESTAMP GRID. Two different time bases: the banked
    ``grid_crosscheck.json`` measures them 9.2e-05 - 6.0e-04 s apart, i.e.
    genuinely non-zero. Had the gate instead compared the grid against itself it
    would read exactly 0.0 with infinite separation and would be measuring
    nothing -- the failure mode this programme has already paid for four times
    in one afternoon on a ridge probe.
    """
    episode_poses = _lead_block_mod().episode_poses

    v_grid = np.asarray(episode_poses(ego_df, t_grid_s * EGO_CLOCK_UNIT),
                        dtype=np.float64)[:, 3]
    return {str(clip_id): {int(i): (float(t_grid_s[i]), float(v_grid[i]))
                           for i in range(len(t_grid_s))}}


def poses_at_registered_time(ego_df, t_s: np.ndarray) -> np.ndarray:
    """``[T, 4]`` ego signals sampled at the REGISTERED times ``t_s``.

    Column 3 is the speed the gate compares against the grid reference. Passing
    the grid-built poses instead would compare the grid with itself.
    """
    episode_poses = _lead_block_mod().episode_poses

    return np.asarray(episode_poses(ego_df, np.asarray(t_s, dtype=np.float64)
                                    * EGO_CLOCK_UNIT), dtype=np.float64)


def load_parquet(path: Path):
    import pandas as pd
    return pd.read_parquet(path)


def load_lead_block(path: Path | None) -> dict | None:
    """``(clip_id, frame) -> (t0_s, speed)`` from the banked B1 EVAL lead block."""
    if path is None:
        return None
    if not Path(path).is_file():
        print("[b1join] WARN lead block not found at %s -- alignment proof "
              "DISABLED (the join will be written UNPROVEN)" % path, flush=True)
        return None
    blk = np.load(str(path), allow_pickle=False)
    need = ("clip_id", "frame", "t0_s", "speeds")
    miss = [k for k in need if k not in blk.files]
    if miss:
        print("[b1join] WARN lead block lacks %s -- alignment proof DISABLED"
              % (miss,), flush=True)
        return None
    out: dict[str, dict[int, tuple[float, float]]] = {}
    for c, f, t, s in zip(blk["clip_id"], blk["frame"], blk["t0_s"], blk["speeds"]):
        out.setdefault(str(c), {})[int(f)] = (float(t), float(s))
    return out


# ============================================================================
# the join (geometry imported, never re-implemented)
# ============================================================================
def join_clip_raw(clip_id, poses, ego, obs, *, tol_s, hfov_deg, n_stack=0,
                  t_s_override=None):
    """One clip -> records keyed on the RAW frame index + stats.

    Mirrors ``build_obstacle_join.join_clip`` exactly, with two deliberate
    differences: ``poses`` is the RAW (untrimmed) array, and the emitted key is
    ``frame`` (the raw index ``i``) rather than ``frame_idx``.

    ``t_s_override``: the clip time of each RAW frame, supplied instead of
    recovered. Used ONLY for clips where content registration is impossible
    because the ego barely moves -- ``register_poses_to_time`` identifies time
    by POSITION, so a clip that does not move carries no positional signal. The
    lead block's ``t0_s`` is a valid substitute because it comes from the
    corpus's own timestamp grid rather than from position.

    ⚠️ For an overridden clip the time-agreement statistic against the block is
    CIRCULAR (it is the block's own number) and is excluded from the reported
    dt. The SPEED gate stays non-circular and still governs: it compares the
    v2ep record's ``poses[:, 3]`` with the block's ``speeds``, two independently
    produced arrays, at the same frame index.
    """
    from build_obstacle_join import (clip_tracks, visibility_occ,
                                     world_agents_at)
    from tanitad.data.bev_raster import ego_frame_agents
    _bootstrap_taniteval()
    from taniteval import lead_source as ls

    poses = np.asarray(poses, dtype=np.float64)
    if t_s_override is not None:
        t_s = np.asarray(t_s_override, dtype=np.float64)
        reg = {"a": float("nan"), "b": float("nan"),
               "residual_m": None, "n_inlier": 0, "n_probe": 0,
               "time_source": "lead_block_t0_s"}
    else:
        reg = ls.register_poses_to_time(poses[:, :2], ego.t, ego.x, ego.y)
        reg = dict(reg)
        reg["time_source"] = "register_poses_to_time"
        t_s = np.asarray(reg["t_s"], dtype=np.float64)
    tracks = clip_tracks(obs)
    ot = np.asarray(obs["timestamp_us"], dtype=np.float64) / 1e6
    if ot.size == 0:
        raise ValueError("%s: obstacle parquet has 0 rows" % clip_id)
    span = (float(ot.min()), float(ot.max()))

    # ⭐ BOTH index spaces are emitted, because THIS CORPUS HAS TWO CONSUMERS
    # and they disagree:
    #   `frame`     RAW v2ep index      -> refav1 / b1_eval_lead_block adapter
    #   `frame_idx` post-n_stack-trim   -> train_p8_occupancy.JoinFileReader,
    #                                      which refc_v3_train.py --agent-join uses
    # A reader ignores keys it does not know (build_obstacle_join's roundtrip
    # test pins this), so one artifact serves both and NEITHER has to guess.
    # frame_idx is negative for the first (n_stack - 1) frames: those keys are
    # simply never looked up, and dropping the rows instead would lose raw
    # frames the lead-block consumer legitimately wants.
    trim = max(int(n_stack) - 1, 0)
    records = []
    n_boxes = n_vis = 0
    cls_hist: dict[str, int] = {}
    for i in range(poses.shape[0]):
        ti = float(t_s[i])
        if not (span[0] - tol_s <= ti <= span[1] + tol_s):
            continue                                   # NO_LABEL: no line at all
        ag_w, tids, clss = world_agents_at(tracks, ti, ego, tol_s)
        ag_e = ego_frame_agents(ag_w, poses[i, :3])
        occ = visibility_occ(ag_e, hfov_deg)
        agents = [{"cx": round(float(a[0]), 4), "cy": round(float(a[1]), 4),
                   "yaw": round(float(a[2]), 5),
                   "l": round(float(a[3]), 3), "w": round(float(a[4]), 3),
                   "occ": int(o), "track_id": t, "cls": c}
                  for a, o, t, c in zip(ag_e, occ, tids, clss)]
        for c in clss:
            cls_hist[str(c)] = cls_hist.get(str(c), 0) + 1
        records.append({"clip_id": str(clip_id), FRAME_KEY: int(i),
                        "frame_idx": int(i) - trim,
                        "t_s": round(ti, 4), "agents": agents})
        n_boxes += len(agents)
        n_vis += int((occ == 0).sum())

    stats = {"clip_id": str(clip_id), "n_frames": int(poses.shape[0]),
             "n_labelled": len(records), "n_agent_boxes": n_boxes,
             "n_visible_boxes": n_vis,
             "visible_frac": round(n_vis / n_boxes, 4) if n_boxes else None,
             "n_tracks": len(tracks),
             "label_span_s": [round(span[0], 3), round(span[1], 3)],
             "cls_hist": cls_hist,
             "time_source": reg["time_source"],
             "registration": {"a": round(float(reg["a"]), 6),
                              "b": round(float(reg["b"]), 6),
                              "residual_m": reg["residual_m"],
                              "n_inlier": reg["n_inlier"],
                              "n_probe": reg["n_probe"]}}
    return records, stats, t_s


# ============================================================================
# alignment proof: the TRUE join vs the deliberate mis-join(+1) control
# ============================================================================
def _block_time_base(block, clip_id, n_poses):
    """The block's ``t0_s`` for every RAW frame of ``clip_id``, or None.

    Returns None unless the block covers frames ``0..n_poses-1`` contiguously —
    a partial cover would silently shift the tail.
    """
    if block is None or clip_id not in block:
        return None
    fb = block[clip_id]
    if not all(i in fb for i in range(n_poses)):
        return None
    return np.array([fb[i][0] for i in range(n_poses)], dtype=np.float64)


def alignment_probe(clip_id, t_s, poses, block) -> dict | None:
    """Compare the recovered per-RAW-frame time against the banked block.

    Returns the TRUE and MIS+1 discrepancies. The control exists because a small
    TRUE residual alone proves nothing: it must be small *relative to* what a
    one-frame error would look like on this same clip.
    """
    if block is None or clip_id not in block:
        return None
    fb = block[clip_id]
    frames = sorted(fb)
    if len(frames) < 3:
        return None
    t_blk = np.array([fb[f][0] for f in frames], dtype=np.float64)
    s_blk = np.array([fb[f][1] for f in frames], dtype=np.float64)
    idx = np.array(frames, dtype=np.int64)
    ok = idx < len(t_s)
    idx, t_blk, s_blk = idx[ok], t_blk[ok], s_blk[ok]
    if idx.size < 3:
        return None
    d_true = np.abs(t_s[idx] - t_blk)
    # mis-join(+1): my frame i read against the block's frame i+1
    ok2 = (idx + 1) < len(t_s)
    d_mis = np.abs(t_s[idx[ok2] + 1] - t_blk[ok2])

    # ---- the GATE statistic: ego speed at the SAME frame index ---------------
    # TRUE:  poses[i, 3]   vs block speeds[i]
    # MIS+1: poses[i+1, 3] vs block speeds[i]   <- the control, same statistic
    if poses.shape[1] > 3:
        d_spd = np.abs(poses[idx, 3] - s_blk)
        ok3 = (idx + 1) < poses.shape[0]
        d_spd_mis = (np.abs(poses[idx[ok3] + 1, 3] - s_blk[ok3])
                     if ok3.any() else np.array([np.nan]))
    else:
        d_spd = d_spd_mis = np.array([np.nan])
    # ---- the FRAME-UNIT statistic: how far, in frames, the join's time base
    # sits from the reference grid. This is the gate when no second,
    # independently produced speed array exists (see FRAME_ERR_* below).
    dt_grid = float(np.median(np.diff(t_blk))) if t_blk.size > 1 else float("nan")
    if np.isfinite(dt_grid) and dt_grid > 0:
        f_true = float(d_true.max() / dt_grid)
        f_mis = float(d_mis.min() / dt_grid) if d_mis.size else float("nan")
    else:
        f_true = f_mis = float("nan")
    return {"n": int(idx.size),
            "true_max_dt_s": float(d_true.max()),
            "misjoin1_max_dt_s": float(d_mis.max()) if d_mis.size else float("nan"),
            "grid_dt_s": dt_grid,
            "true_max_err_frames": f_true,
            "misjoin1_min_err_frames": f_mis,
            "speed_max_dv_mps": float(np.nanmax(d_spd)),
            "speed_misjoin1_max_dv_mps": float(np.nanmax(d_spd_mis))}


# ============================================================================
# output
# ============================================================================
def open_out(path, compressed: bool):
    """⚠️ ``compressed`` is EXPLICIT, never sniffed from ``path``.

    The build writes to ``<out>.part`` and renames on success. Sniffing ``.xz``
    off the temp name yields False (``.xz.part``) while the reader sniffs the
    FINAL name and yields True -- so the file is written plain and read as xz.
    MEASURED here on the first smoke run: ``LZMAError: Input format not
    supported by decoder``, caught only because the content assertion re-reads
    what was actually written.
    """
    p = str(path)
    if compressed:
        return lzma.open(p, "wt", encoding="utf-8", preset=6)
    return open(p, "w", encoding="utf-8")


def md5_of(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_content(out_path, n_expect_lines, compressed: bool) -> dict:
    """CONTENT assertion on the WRITTEN bytes -- never on the builder's own counters.

    A pre-allocated file of zeros, a truncated write and a successful build are
    indistinguishable by size and exit code; they are not indistinguishable by
    re-reading and requiring the numbers to be non-degenerate.
    """
    n_lines = n_boxes = n_vis = 0
    clips = set()
    frames_seen = set()
    sx = sy = 0.0
    op = str(out_path)
    fh = lzma.open(op, "rt", encoding="utf-8") if compressed \
        else open(op, "r", encoding="utf-8")
    with fh:
        for line in fh:
            rec = json.loads(line)
            n_lines += 1
            clips.add(rec["clip_id"])
            frames_seen.add((rec["clip_id"], rec[FRAME_KEY]))
            for a in rec["agents"]:
                n_boxes += 1
                n_vis += 1 if a["occ"] == 0 else 0
                sx += abs(a["cx"])
                sy += abs(a["cy"])
    if n_lines != n_expect_lines:
        raise SystemExit("[b1join] REFUSING: wrote %d lines, expected %d"
                         % (n_lines, n_expect_lines))
    if n_lines == 0 or n_boxes == 0:
        raise SystemExit("[b1join] REFUSING: the written join is EMPTY "
                         "(%d lines, %d boxes) -- a zeros artifact" % (n_lines, n_boxes))
    if len(frames_seen) != n_lines:
        raise SystemExit("[b1join] REFUSING: duplicate (clip_id, %s) keys -- "
                         "the join would be ambiguous" % FRAME_KEY)
    mean_ax, mean_ay = sx / n_boxes, sy / n_boxes
    if not (np.isfinite(mean_ax) and np.isfinite(mean_ay)) or mean_ax == 0.0:
        raise SystemExit("[b1join] REFUSING: agent coordinates are degenerate "
                         "(mean|cx|=%r mean|cy|=%r)" % (mean_ax, mean_ay))
    return {"n_lines": n_lines, "n_clips": len(clips), "n_agent_boxes": n_boxes,
            "n_visible_boxes": n_vis,
            "visible_frac": round(n_vis / n_boxes, 4),
            "mean_abs_cx_m": round(mean_ax, 4), "mean_abs_cy_m": round(mean_ay, 4),
            "unique_keys": len(frames_seen)}


# ============================================================================
def build_args(argv=None):
    ap = argparse.ArgumentParser(
        description="B1 EVAL obstacle.offline -> agent join, keyed on "
                    "(clip_id, RAW v2ep frame) to match b1_eval_lead_block.npz")
    ap.add_argument("--pose-source", choices=("v2ep", "reconstruct"),
                    default="v2ep",
                    help="v2ep: read poses from *.v2ep.pt (needs --eps-dir). "
                         "reconstruct: rebuild them from --ts-dir + --ego-dir "
                         "(float32-identical, see read_reconstructed_poses) -- "
                         "the only route for B1 TRAIN, whose epcache is ~161 GB "
                         "and lives on Thor")
    ap.add_argument("--eps-dir", default=None,
                    help="dir of *.v2ep.pt RAW episode records (pose source; "
                         "required for --pose-source v2ep)")
    ap.add_argument("--ts-dir", default=None,
                    help="dir of {clip}.timestamps.parquet (required for "
                         "--pose-source reconstruct)")
    ap.add_argument("--n-stack", type=int, default=None,
                    help="n_stack for the post-trim `frame_idx` key. Read from "
                         "each record in v2ep mode; an OPERATOR input in "
                         "reconstruct mode, and recorded as such in the sidecar")
    ap.add_argument("--ego-dir", required=True,
                    help="dir of per-clip egomotion parquets")
    ap.add_argument("--obstacle-dir", required=True,
                    help="dir of per-clip obstacle.offline parquets")
    ap.add_argument("--lead-block", default=None,
                    help="b1_eval_lead_block.npz -- ground truth for the "
                         "alignment proof; omitting it writes an UNPROVEN join")
    ap.add_argument("--out", required=True, help="output jsonl or jsonl.xz")
    ap.add_argument("--clips", default=None,
                    help="optional json/txt list restricting the clip set")
    ap.add_argument("--tol-s", type=float, default=None,
                    help="per-track match tolerance [s] (default: "
                         "bev_raster.DEFAULT_TOL_S)")
    ap.add_argument("--hfov-deg", type=float, default=None,
                    help="front-camera field for the occ flag (default 120)")
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: only the first N clips")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    from build_obstacle_join import (DEFAULT_TOL_S, EgoTrack,
                                     HFOV_DEG_DEFAULT)
    a = build_args(argv)
    tol_s = DEFAULT_TOL_S if a.tol_s is None else a.tol_s
    hfov = HFOV_DEG_DEFAULT if a.hfov_deg is None else a.hfov_deg

    ego_dir = Path(a.ego_dir)
    obs_dir = Path(a.obstacle_dir)
    recon = a.pose_source == "reconstruct"
    ts_dir = Path(a.ts_dir) if a.ts_dir else None
    if recon:
        if ts_dir is None or not a.clips or a.n_stack is None:
            raise SystemExit("[b1join] --pose-source reconstruct needs --ts-dir, "
                             "--clips and --n-stack (a reconstructed clip has no "
                             "record to read n_stack from)")
    elif not a.eps_dir:
        raise SystemExit("[b1join] --pose-source v2ep needs --eps-dir")

    want = None
    if a.clips:
        raw = Path(a.clips).read_text(encoding="utf-8")
        try:
            j = json.loads(raw)
            want = set(j if isinstance(j, list)
                       else next(v for v in j.values() if isinstance(v, list)))
        except Exception:
            want = set(x.strip() for x in raw.split() if x.strip())

    if recon:
        # the clip list IS the corpus definition here; a missing timestamps
        # parquet is named, never silently dropped to a shorter corpus.
        cids = sorted(want)
        missing_ts = [c for c in cids
                      if not (ts_dir / (c + ".timestamps.parquet")).is_file()]
        if missing_ts:
            raise SystemExit("[b1join] %d/%d clips have no timestamps parquet "
                             "under %s (first: %s)"
                             % (len(missing_ts), len(cids), ts_dir, missing_ts[:3]))
        files = [ts_dir / (c + ".timestamps.parquet") for c in cids]
    else:
        eps = Path(a.eps_dir)
        files = sorted(eps.glob("*.v2ep.pt"))
        if not files:
            raise SystemExit("[b1join] no *.v2ep.pt under %s" % eps)
        if want is not None:
            files = [f for f in files if f.name.split(".v2ep")[0] in want]
    if a.limit:
        files = files[:a.limit]

    block = load_lead_block(Path(a.lead_block) if a.lead_block else None)
    print("[b1join] clips=%d  tol_s=%.4f  hfov=%.1f  pose_source=%s  "
          "lead_block=%s" % (len(files), tol_s, hfov, a.pose_source,
                             "yes" if block else
                             ("per-clip GRID reference" if recon
                              else "NO (unproven)")), flush=True)

    t0 = time.time()
    # `alignment_failed` is initialised EMPTY rather than created on first use,
    # so the sidecar always carries the key and "no clip was excluded" is a
    # POSITIVE statement in the artifact instead of an absent field a reader has
    # to interpret.
    per_clip, probes, skipped = [], [], {"no_obstacle": [], "no_egomotion": [],
                                         "error": [], "alignment_failed": []}
    n_lines = 0
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    compressed = str(outp).endswith(".xz")
    tmp = outp.with_name(outp.name + ".part")

    with open_out(tmp, compressed) as fh:
        for j, f in enumerate(files):
            cid = (f.name.split(".timestamps")[0] if recon
                   else f.name.split(".v2ep")[0])
            op = obs_dir / (cid + ".parquet")
            gp = ego_dir / (cid + ".parquet")
            if not op.is_file():
                skipped["no_obstacle"].append(cid)
                print("  [%d/%d] %s SKIP no obstacle parquet" % (j + 1, len(files), cid[:8]),
                      flush=True)
                continue
            if not gp.is_file():
                skipped["no_egomotion"].append(cid)
                print("  [%d/%d] %s SKIP no egomotion parquet" % (j + 1, len(files), cid[:8]),
                      flush=True)
                continue
            try:
                ego_df = load_parquet(gp)
                clip_ref, gate_poses = block, None
                if recon:
                    poses, t_grid_s, _u = read_reconstructed_poses(
                        f, ego_df, a.n_stack)
                    n_stack = int(a.n_stack)
                    # the gate's reference for a corpus with no banked block
                    clip_ref = grid_reference(cid, ego_df, t_grid_s)
                else:
                    _cid, poses, n_stack = read_raw_poses(f)
                ego = EgoTrack(ego_df)
                obs = load_parquet(op)
                try:
                    recs, st, t_s = join_clip_raw(cid, poses, ego, obs,
                                                  tol_s=tol_s, hfov_deg=hfov,
                                                  n_stack=n_stack)
                except Exception as e:                              # noqa: BLE001
                    # A clip that does not MOVE carries no positional signal, so
                    # registration by content cannot identify its time. That is a
                    # correct refusal, not a bug -- but it is recoverable when the
                    # block (built off the timestamp grid) covers the clip.
                    ov = _block_time_base(clip_ref, cid, poses.shape[0])
                    if ov is None or "egistration" not in repr(e):
                        raise
                    recs, st, t_s = join_clip_raw(cid, poses, ego, obs,
                                                  tol_s=tol_s, hfov_deg=hfov,
                                                  n_stack=n_stack, t_s_override=ov)
                    print("  [%d/%d] %s registration impossible (%s) -> time base "
                          "from %s" % (j + 1, len(files), cid[:8], repr(e)[:60],
                                       "camera timestamp grid" if recon
                                       else "lead block"), flush=True)
            except Exception as e:                                  # noqa: BLE001
                skipped["error"].append([cid, repr(e)[:200]])
                print("  [%d/%d] %s SKIP %r" % (j + 1, len(files), cid[:8], e),
                      flush=True)
                continue
            st["n_stack"] = n_stack
            if recon:
                # The speed columns are REPORTED, never gated, in this mode:
                # `poses` was built on the grid, so `poses[:,3]` IS the
                # reference's own speeds (an identity), while sampling at the
                # registered times measures registration PRECISION rather than
                # key correctness (MEASURED: control smaller than true; see
                # FRAME_ERR_*). The gate is the frame-unit statistic instead.
                gate_poses = poses_at_registered_time(ego_df, t_s)
            pr = alignment_probe(cid, t_s, gate_poses if recon else poses,
                                 clip_ref)
            if pr:
                # a reference-sourced time base makes the TIME comparison
                # circular; the SPEED gate stays independent and still governs.
                pr["time_circular"] = (st["time_source"] == "lead_block_t0_s")
                st["alignment"] = pr
                probes.append(pr)
                # ---- the PER-CLIP gate (reconstruct mode) -------------------
                # A clip whose time base is a frame off must not be written at
                # all: it would condition the head on the WRONG agents, which is
                # worse than having no label for it. It is EXCLUDED and NAMED,
                # exactly as a clip with no obstacle member is -- never silently
                # kept, and never a reason to refuse the 4,433 clips that are
                # right. The corpus-level bound below is what keeps this from
                # becoming a way to absorb a systematic defect.
                if recon and not (
                        pr["true_max_err_frames"] <= FRAME_ERR_MAX_TRUE
                        and pr["misjoin1_min_err_frames"] >= FRAME_ERR_MIN_MISJOIN):
                    skipped.setdefault("alignment_failed", []).append(
                        {"clip_id": cid,
                         "true_max_err_frames": pr["true_max_err_frames"],
                         "misjoin1_min_err_frames": pr["misjoin1_min_err_frames"],
                         "true_max_dt_s": pr["true_max_dt_s"],
                         "misjoin1_max_dt_s": pr["misjoin1_max_dt_s"],
                         "grid_dt_s": pr["grid_dt_s"],
                         "n_stack": n_stack,
                         "reason": "frame gate: the join's time base is not on "
                                   "the corpus's own frame grid"})
                    probes.pop()          # excluded clips do not score the corpus
                    print("  [%d/%d] %s EXCLUDED by the frame gate "
                          "(true %.4f frames, misjoin+1 %.4f)"
                          % (j + 1, len(files), cid[:8],
                             pr["true_max_err_frames"],
                             pr["misjoin1_min_err_frames"]), flush=True)
                    continue
            for rec in recs:
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            n_lines += len(recs)
            per_clip.append(st)
            print("  [%d/%d] %s n_raw=%d labelled=%d boxes=%d vis=%.3f%s"
                  % (j + 1, len(files), cid[:8], st["n_frames"], st["n_labelled"],
                     st["n_agent_boxes"], st["visible_frac"] or 0.0,
                     ("  dt=%.2e mis=%.2e" % (pr["true_max_dt_s"],
                                              pr["misjoin1_max_dt_s"])) if pr else ""),
                  flush=True)

    # ---- the alignment VERDICT, before anything is published ----------------
    proof = None
    if probes:
        # GATE: ego-speed agreement at the same frame index (key correctness).
        dv = max(p["speed_max_dv_mps"] for p in probes)
        dv_mis = min(p["speed_misjoin1_max_dv_mps"] for p in probes)
        sep = (dv_mis / dv) if dv > 0 else float("inf")
        # the frame-unit statistic (the gate in reconstruct mode)
        fe = max(p["true_max_err_frames"] for p in probes)
        fe_mis = min(p["misjoin1_min_err_frames"] for p in probes)
        fsep = (fe_mis / fe) if fe > 0 else float("inf")
        # REPORTED: time agreement (registration precision, not key correctness).
        # Circular clips (block-sourced time base) are EXCLUDED -- comparing the
        # block against itself would report a flattering 0.0 and mean nothing.
        noncirc = [p for p in probes if not p.get("time_circular")]
        tr = max((p["true_max_dt_s"] for p in noncirc), default=float("nan"))
        ms = max((p["misjoin1_max_dt_s"] for p in noncirc), default=float("nan"))
        worst = tr
        n_excl = len(skipped.get("alignment_failed", []))
        n_probed_total = len(probes) + n_excl
        excl_frac = (n_excl / n_probed_total) if n_probed_total else 0.0
        if recon:
            # every WRITTEN clip already passed the per-clip gate, so `fe` /
            # `fe_mis` here are the worst SURVIVING values; the corpus question
            # is whether too many clips had to be excluded.
            passed = bool(fe <= FRAME_ERR_MAX_TRUE
                          and fe_mis >= FRAME_ERR_MIN_MISJOIN
                          and excl_frac <= MAX_ALIGNMENT_EXCLUDED_FRAC)
        else:
            passed = bool(dv <= LEAD_SPEED_TOL_MPS
                          and sep >= MIN_MISJOIN_SEPARATION)
        proof = {
            "n_clips_probed": len(probes),
            "gate": ({
                "statistic": "max |t_s[i] - t_grid[i]| / dt_grid  -- the "
                             "frame-index error IN FRAMES, at the SAME "
                             "(clip_id, frame)",
                "why": "a frame index is an INTEGER, so the decision boundary "
                       "is 0.5 frames. The two sides come from two different "
                       "time bases: register_poses_to_time recovers time from "
                       "POSITION, the reference from the CAMERA TIMESTAMP GRID.",
                "reference_source": "per-clip camera timestamp grid "
                                    "(episode_grid -> episode_poses), the SAME "
                                    "construction as b1_eval_lead_block.npz's "
                                    "t0_s/speeds -- recomputed rather than read "
                                    "from a banked npz, because B1 TRAIN has no "
                                    "lead block",
                "why_not_the_eval_speed_gate":
                    "the EVAL gate needs TWO independently produced speed "
                    "arrays indexed by the same frame (v2ep poses[:,3] and the "
                    "block's speeds). Neither exists on B1 TRAIN: the poses are "
                    "BUILT on the grid, so poses[:,3] IS the reference's own "
                    "speeds. Re-sampling one side at the registered times to "
                    "manufacture a difference was TRIED and MEASURED on the 141 "
                    "EVAL clips: true 1.191e-01 m/s vs mis-join(+1) 2.134e-02 "
                    "m/s -- the CONTROL READ SMALLER THAN THE TRUE VALUE. It "
                    "had become a registration-PRECISION statistic, the same "
                    "confusion the EVAL package diagnosed for its 5e-3 s time "
                    "gate.",
                "true_max_err_frames": fe,
                "misjoin1_min_err_frames": fe_mis,
                "max_true_allowed_frames": FRAME_ERR_MAX_TRUE,
                "min_misjoin_required_frames": FRAME_ERR_MIN_MISJOIN,
                "control_expected_value_frames": 1.0,
                "applied": "PER CLIP -- a failing clip is EXCLUDED and NAMED in "
                           "skipped.alignment_failed, never written. The values "
                           "here are the worst among the clips that were KEPT.",
                "n_clips_excluded_by_the_gate": n_excl,
                "n_clips_probed_total": n_probed_total,
                "excluded_frac": round(excl_frac, 6),
                "max_excluded_frac_allowed": MAX_ALIGNMENT_EXCLUDED_FRAC,
                "why_a_corpus_bound": "a systematic one-frame defect would hit "
                                      "many clips at once; the bound stops "
                                      "'exclude the failures' from absorbing "
                                      "it as 'a few bad ones'",
                "separation_x": round(fsep, 1),
                "grid_dt_s_median": float(np.median(
                    [p["grid_dt_s"] for p in probes])),
                "speed_reported_only": {
                    "true_max_dv_mps": dv, "misjoin1_min_dv_mps": dv_mis,
                    "note": "REPORTED, NOT GATED in reconstruct mode -- see "
                            "why_not_the_eval_speed_gate"},
                "passed": passed} if recon else {
                "statistic": "max |v2ep poses[:,3] - block speeds| at the "
                             "SAME (clip_id, frame)",
                "why": "both are egomotion interpolated at that frame, so "
                       "agreement proves the join hit the right clip AND "
                       "frame; INDEPENDENT of the registration fit",
                "reference_source": "b1_eval_lead_block.npz",
                "tol_mps": LEAD_SPEED_TOL_MPS,
                "tol_source": "refav1_arm.py:238 LEAD_SPEED_TOL_MPS",
                "true_max_dv_mps": dv,
                "misjoin1_min_dv_mps": dv_mis,
                "separation_x": round(sep, 1),
                "min_separation_required": MIN_MISJOIN_SEPARATION,
                "passed": passed}),
            "reported_only": {
                "statistic": "max |recovered t_s - block t0_s|",
                "true_max_dt_s": tr, "misjoin1_max_dt_s": ms,
                "worst_clip_dt_s": worst,
                "n_clips_noncircular": len(noncirc),
                "n_clips_circular_excluded": len(probes) - len(noncirc),
                "instrument_documented_worst_s": MAX_TRUE_DT_S_REPORTED,
                "note": "registration PRECISION, not key correctness -- a clip "
                        "with few registration probes carries a smooth time "
                        "bias while its frame index is exactly right"},
            "key": "(clip_id, %s) with %s = RAW v2ep index" % (FRAME_KEY, FRAME_KEY),
            "control": "mis-join(+1): frame i+1 read against block frame i, on "
                       "the SAME statistic as the gate",
            "passed": passed}
        if recon:
            print("[b1join] ALIGNMENT GATE frames (KEPT clips) true_max=%.4f "
                  "(allowed <= %.2f)  misjoin+1 min=%.4f (required >= %.2f, "
                  "expected ~1.0)  sep=%.1fx  -> %s"
                  % (fe, FRAME_ERR_MAX_TRUE, fe_mis, FRAME_ERR_MIN_MISJOIN,
                     fsep, "PASS" if passed else "FAIL"), flush=True)
            print("[b1join] per-clip gate EXCLUDED %d of %d probed clips "
                  "(%.4f%%, bound %.2f%%)"
                  % (n_excl, n_probed_total, 100.0 * excl_frac,
                     100.0 * MAX_ALIGNMENT_EXCLUDED_FRAC), flush=True)
            print("[b1join] reported-only: speed true_max_dv=%.3e m/s  "
                  "misjoin+1 min=%.3e m/s (NOT gated in reconstruct mode)"
                  % (dv, dv_mis), flush=True)
        else:
            print("[b1join] ALIGNMENT GATE speed true_max_dv=%.3e m/s (tol %.1e)"
                  "  misjoin+1 min=%.3e m/s  sep=%.0fx  -> %s"
                  % (dv, LEAD_SPEED_TOL_MPS, dv_mis, sep,
                     "PASS" if passed else "FAIL"), flush=True)
        print("[b1join] reported-only: time true_max_dt=%.3e s  misjoin+1=%.3e s "
              "(instrument worst %.1e s)" % (tr, ms, MAX_TRUE_DT_S_REPORTED), flush=True)
        if not passed:
            tmp.unlink(missing_ok=True)
            if recon:
                raise SystemExit(
                    "[b1join] REFUSING TO WRITE: frame gate on the KEPT clips "
                    "true %.4f (allowed %.2f), mis-join(+1) %.4f (required "
                    "%.2f); EXCLUDED %d of %d probed = %.4f%% (bound %.2f%%). "
                    "An unproven join silently conditions on the wrong agents."
                    % (fe, FRAME_ERR_MAX_TRUE, fe_mis, FRAME_ERR_MIN_MISJOIN,
                       n_excl, n_probed_total, 100.0 * excl_frac,
                       100.0 * MAX_ALIGNMENT_EXCLUDED_FRAC))
            raise SystemExit(
                "[b1join] REFUSING TO WRITE: the speed gate failed (true %.3e "
                "m/s vs tol %.1e; mis-join(+1) %.3e m/s, %.0fx). An unproven "
                "join silently conditions on the wrong agents."
                % (dv, LEAD_SPEED_TOL_MPS, dv_mis, sep))
    else:
        print("[b1join] WARN no alignment probe ran -- the join is UNPROVEN", flush=True)

    os.replace(tmp, outp)
    content = assert_content(outp, n_lines, compressed)
    md5 = md5_of(outp)
    wall = round(time.time() - t0, 1)

    meta = {
        "task": ("B1 %s obstacle.offline -> agent join (WP-6 agent conditioning)"
                 % ("TRAIN" if recon else "EVAL")),
        "_evidence_class": "MEASURED (ours; artifact = the jsonl + this meta)",
        "builder": "stack/scripts/build_b1_agent_join.py",
        "corpus": {
            "name": "B1 v7.2 %s split" % ("TRAIN" if recon else "EVAL"),
            "n_clips_offered": len(files),
            "pose_source": a.pose_source,
            "pose_source_note":
                ("poses rebuilt from {clip}.timestamps.parquet + egomotion "
                 "(episode_grid -> episode_poses). MEASURED float32-IDENTICAL "
                 "to the banked v2ep poses on 20/20 clips (max_dxy_m 0.0, "
                 "max_dyaw_deg 0.0, max_dv_mps 0.0; "
                 ".../2026-09-02-b1-eval-lead-block/raw/grid_crosscheck.json), "
                 "so the ~161 GB B1 TRAIN epcache on Thor is not needed."
                 if recon else "poses read from *.v2ep.pt records"),
            "eps_dir": str(a.eps_dir) if a.eps_dir else None,
            "ts_dir": str(ts_dir) if ts_dir else None,
            "n_stack": int(a.n_stack) if recon else None,
            "n_stack_source": ("OPERATOR (--n-stack); a reconstructed clip "
                               "carries no record to read it from"
                               if recon else "per-record d['n_stack']"),
            "note": "B1 = the 4,719-clip v7 corpus (4,713 built after the parity "
                    "gate drops 6 val40 clips); the v7.2 release splits it "
                    "4,572 TRAIN / 147 EVAL; 141 of the 147 have pixels in the "
                    "built B1 epcache and are what refav1 scores on. TRAIN is "
                    "r0_selection (4,719) minus those 147 = exactly 4,572.",
        },
        "key": {
            "fields": ["clip_id", FRAME_KEY],
            "frame_index_space": "RAW v2ep index i in [0, n_target) -- NO n_stack "
                                 "trim; identical to b1_eval_lead_block.npz's "
                                 "`frame`, and DIFFERENT from build_obstacle_join's "
                                 "post-trim `frame_idx` (i - (n_stack - 1))",
            "why_renamed": "the key is `frame`, not `frame_idx`, so the two index "
                           "spaces cannot be silently confused",
        },
        "args": {"pose_source": a.pose_source,
                 "eps_dir": str(a.eps_dir) if a.eps_dir else None,
                 "ts_dir": str(ts_dir) if ts_dir else None,
                 "n_stack": a.n_stack, "clips": a.clips,
                 "ego_dir": str(ego_dir),
                 "obstacle_dir": str(obs_dir), "lead_block": a.lead_block,
                 "out": str(outp), "tol_s": tol_s, "hfov_deg": hfov,
                 "argv": sys.argv[1:]},
        "summary": {"n_clips_joined": len(per_clip), "n_clips_offered": len(files),
                    "n_lines": n_lines, "wall_s": wall, "md5": md5,
                    # legacy key, kept for readers that already parse it. The
                    # AUTHORITATIVE declaration is the top-level `digest_scope`
                    # block attached below via join_meta.attach; "plain" here
                    # was never a member of ARTIFACT_SCOPES.
                    "digest_scope": ("compressed" if str(outp).endswith(".xz")
                                     else "decompressed"),
                    "skipped": skipped},
        "content_assertion": content,
        "alignment_proof": proof,
        "conventions": {
            "frame": "per-frame EGO frame, +x fwd +y LEFT (refb_labels.ego_frame "
                     "via bev_raster.ego_frame_agents)",
            "composition": "rig@sample -> world at the sample's OWN timestamp -> "
                           "ego@frame",
            "time": "recovered per clip by lead_source.register_poses_to_time over "
                    "the RAW pose array (grid ~0.1007 s, not 0.1)",
            "NO_LABEL": "absent (clip, frame) line -- outside the obstacle span; an "
                        "EMPTY agents list IS a label (labelled clear)",
            "occ": "0 = agent centre inside the 120 deg front-camera field, 1 = "
                   "outside while the track continues; IS bev_raster.fov_mask at "
                   "agent-centre granularity (P4_PREDICATE_IDENTITY), not an "
                   "independent occlusion label",
            "sizes": "l = size_x, w = size_y",
        },
        "per_clip": per_clip,
    }
    # ⛔ DECLARE THE DIGEST SCOPE THE WAY THE CONSUMER READS IT, OR THE JOIN IS
    # UNUSABLE. MEASURED 2026-09-06: `refc_v3_train.py --agent-join-verify auto`
    # calls `join_meta.read_digest_scope`, which requires a TOP-LEVEL
    # `digest_scope` BLOCK (a dict of algo/digest/scope/filename) and REFUSES a
    # sidecar that declares nothing -- deliberately, with no fallback. This
    # builder wrote only `summary.digest_scope`, a STRING nested inside
    # `summary`, which that reader never looks at: the train run would have died
    # at startup with JoinDigestScopeMissing on a perfectly good join.
    # ⚠️ And the legacy string's vocabulary was wrong too -- it emitted "plain"
    # for a non-.xz output, while ARTIFACT_SCOPES is ("compressed",
    # "decompressed"), so even a reader that found it would have raised on the
    # value. Both are fixed here; the legacy key is LEFT IN PLACE (additive,
    # nothing that reads these sidecars today changes behaviour).
    scope = "compressed" if str(outp).endswith(".xz") else "decompressed"
    try:
        from tanitad.data import join_meta as _jm

        meta = _jm.attach(meta, md5, scope=scope, filename=outp.name,
                          algo="md5", declared_by="build_b1_agent_join",
                          note="digest taken over the artifact as written")
        print("[b1join] digest_scope declared: md5(%s of %s) = %s"
              % (scope, outp.name, md5), flush=True)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "[b1join] REFUSING: could not attach the digest-scope declaration "
            "(%r). A sidecar without it is refused by refc_v3_train.py "
            "--agent-join-verify auto, so publishing one would ship a join "
            "that cannot be loaded." % e)

    mp = Path(str(outp) + ".meta.json")
    mp.write_text(json.dumps(meta, indent=1), encoding="utf-8")

    print("B1_JOIN_DONE %s" % json.dumps(
        {"out": str(outp), "meta": str(mp), "md5": md5,
         "n_clips": content["n_clips"], "n_lines": content["n_lines"],
         "n_agent_boxes": content["n_agent_boxes"],
         "visible_frac": content["visible_frac"],
         "alignment_passed": (proof or {}).get("passed"),
         "wall_s": wall}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
