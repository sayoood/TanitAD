#!/usr/bin/env python3
"""refcv5_preflight.py -- can refcv5 start a REAL arm on the parity corpus?

Zero-GPU. Every check is a POSITIVE assertion, and a check that cannot read
its subject reports **INCONCLUSIVE and is counted as a FAILURE, never a pass**
(CLAUDE.md: on this mount an empty result is indistinguishable from a failed
query).

WHY THIS EXISTS -- three MEASURED incidents, all the same shape
---------------------------------------------------------------
* `t1_eval.py` rolled both arms over all 40 episodes (~11 min/arm) and died in
  `analyze()` on a missing import. The expensive part had succeeded.
* `--agents head --w-agent 1.0` TRAINED, CONVERGED, wrote a checkpoint and
  stamped `w_agent: 1.0` while the detector was never supervised, because the
  synthetic corpus carried no join.
* `--agent-w-project 0.2` was stamped into config.json while
  `model._rig_camera` was never assigned, so the term computed nothing (M18).

And one this tool found on its first run:

* `--agent-w-ground` computes a TAUTOLOGY. `ground_range_prior` projects a
  foot at rig z=0 and back-projects onto z=ROAD_PLANE_Z_M=0.0 -- exact
  inverses -- so it reads 2.6e-08 with a gradient of 8.7e-11. A flag guard
  cannot see that; only a GRADIENT probe can.

=> The rule this file enforces: **a weight that reaches config.json must reach
a loss with a non-zero gradient, and every count must come with a same-breath
control that reads non-zero.**

Usage
-----
    python stack/scripts/refcv5_preflight.py \\
        [--v2-cache <dir of *.v2ep.pt>] [--agent-join <join.jsonl.xz>] \\
        [--v7-labels <v72.jsonl.gz>] [--anchors <anchors.pt>] \\
        [--rig-extrinsics <extrinsics.json>] [--smoke] [--json out.json]

Every data argument is optional; an ABSENT input is reported INCONCLUSIVE
(= a failure), never skipped, because "we did not check" and "it is fine" must
not look the same in the exit code.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STACK = HERE.parent
REPO = STACK.parent
if str(STACK) not in sys.path:
    sys.path.insert(0, str(STACK))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# the parity corpus, from the code that owns the fact -- never retyped here
PARITY_HINTS = ("e438721ae894", "f09e44db")

RESULTS: list[dict] = []


def record(name, ok, detail, extra=None):
    """ok: True PASS / False FAIL / None INCONCLUSIVE (counted as a failure)."""
    state = "PASS" if ok is True else ("INCONCLUSIVE" if ok is None else "FAIL")
    RESULTS.append({"state": state, "check": name, "detail": str(detail),
                    **(extra or {})})
    print("  [%-12s] %s\n                 %s" % (state, name, detail),
          flush=True)


def _read(path, tries=20):
    """Retrying read. On this mount an OSError is not evidence of absence."""
    last = None
    for _ in range(tries):
        try:
            with io.open(path, "rb") as fh:
                b = fh.read()
            if b:
                return b, None
            last = "empty read"
        except OSError as ex:
            last = "%s: %s" % (type(ex).__name__, ex)
        time.sleep(0.2)
    return None, last


# ======================================================================== A --
# A. THE CODE: does every seam the trainer stamps actually compute?
# =========================================================================== #

def check_imports():
    """The preflight import probe. A missing optional module must fail in
    seconds, not after the expensive part."""
    mods = ["tanitad.refs.refc_agents", "tanitad.refs.refc_sampler",
            "tanitad.refs.refc_selector", "tanitad.refs.refc_selector_targets",
            "tanitad.refs.refc_v3", "tanitad.refs.refc",
            "tanitad.models.agent_slots", "tanitad.data.rig_projection",
            "tanitad.data.bev_raster", "tanitad.data.join_meta",
            "tanitad.data.v2_dataset", "taniteval.ci"]
    bad = []
    for m in mods:
        try:
            __import__(m)
        except Exception as ex:                               # noqa: BLE001
            bad.append("%s (%s: %s)" % (m, type(ex).__name__, ex))
    record("refcv5 modules import", not bad,
           "%d/%d import" % (len(mods) - len(bad), len(mods))
           + ("; FAILED: " + "; ".join(bad) if bad else ""))
    try:
        import refc_v3_train                                  # noqa: F401
        record("trainer imports", True, "refc_v3_train loaded from %s"
               % refc_v3_train.__file__)
        return refc_v3_train
    except Exception as ex:                                   # noqa: BLE001
        record("trainer imports", False, "%s: %s" % (type(ex).__name__, ex))
        return None


def check_class_enum():
    """The vocabulary has exactly ONE spelling, and it is the label side's. A
    first draft hand-wrote `bicycle/motorcycle/train_or_tram_car`, none of
    which exist in the corpus."""
    try:
        from tanitad.data.bev_raster import ALL_CLASSES
        from tanitad.refs.refc_agents import AGENT_CLASSES, N_AGENT_CLASSES
    except Exception as ex:                                   # noqa: BLE001
        record("class enum is the corpus enum", None,
               "could not read the enums: %s" % ex)
        return
    same = tuple(AGENT_CLASSES) == tuple(ALL_CLASSES)
    forbidden = [c for c in ("bicycle", "motorcycle", "train_or_tram_car")
                 if c in AGENT_CLASSES]
    record("class enum is the corpus enum",
           bool(same and not forbidden and N_AGENT_CLASSES == 10),
           "n=%d identical=%s fabricated=%s; control: 'automobile' in enum=%s"
           % (N_AGENT_CLASSES, same, forbidden or "none",
              "automobile" in AGENT_CLASSES))


def check_query_budget(t):
    """`--agent-queries` must cover the TRAIN max (94), not val40's (24)."""
    if t is None:
        record("query budget covers the train corpus", None, "no trainer")
        return
    try:
        from tanitad.models.agent_slots import N_QUERIES_DEFAULT
        n = int(t.build_parser().get_default("agent_queries"))
    except Exception as ex:                                   # noqa: BLE001
        record("query budget covers the train corpus", None, str(ex))
        return
    # ⛔ A CONSTANT WITH TWO SPELLINGS DOES NOT MOVE WHEN YOU CHANGE ONE.
    # Until 2026-09-05 `train_v6_staged.py` carried TWO hardcoded 16s that
    # never read N_QUERIES_DEFAULT -- the `getattr` fallback AND the
    # `--n-slot-queries` argparse default -- so correcting the constant
    # alone would have left the v6 trainer at 16 while every audit reported
    # the new value. That is the `advect` precedent: two implementations of
    # one number.
    # ⚠️ THE FIRST VERSION OF THIS CHECK ONLY KNEW THE `getattr` FORM and
    # was blind to the argparse one, because an underscore grep does not
    # find a dashed flag. Both spellings are scanned below, and each must
    # resolve to the SYMBOL rather than to any literal.
    import re
    dup = []
    try:
        src = (STACK / "scripts" / "train_v6_staged.py").read_text(
            encoding="utf-8", errors="replace")
        # SAME-BREATH CONTROL: a scan that finds nothing in a file it could
        # not READ is indistinguishable from a clean file.
        ctl = src.count("n_slot_queries") + src.count("n-slot-queries")
        if ctl == 0 or len(src) < 100_000:
            dup.append("INCONCLUSIVE: read %d bytes of train_v6_staged.py "
                       "and found %d n_slot_queries mentions -- suspect the "
                       "read, not the file" % (len(src), ctl))
        else:
            sites = {
                "getattr fallback": re.findall(
                    r"""getattr\(\s*a\s*,\s*["']n_slot_queries["']\s*,"""
                    r"""\s*([^),]+?)\s*\)""", src),
                "--n-slot-queries default": re.findall(
                    r"""["']--n-slot-queries["'][^)]*?default\s*=\s*"""
                    r"""([A-Za-z_0-9.]+)""", src)}
            for name, found in sites.items():
                if not found:
                    dup.append("INCONCLUSIVE: 0 matches for the %s while "
                               "%d mentions read -- suspect the pattern"
                               % (name, ctl))
                elif set(found) != {"N_QUERIES_DEFAULT"}:
                    dup.append("%s is %s, not N_QUERIES_DEFAULT (control: "
                               "%d mentions read)"
                               % (name, sorted(set(found)), ctl))
    except OSError as ex:
        dup.append("INCONCLUSIVE: could not read train_v6_staged.py (%s)" % ex)
    up_ok = int(N_QUERIES_DEFAULT) >= 94
    record("query budget covers the train corpus",
           n >= 94 and up_ok and not dup,
           "--agent-queries default %d (train max 94, val40 max 24). "
           "Upstream N_QUERIES_DEFAULT is %d (ruled M17; 16 was REFUTED at "
           "11.17 pct of boxes dropped, nearest sacrificed target 7.3 m) "
           "and covers the train max: %s. refcv5 is NOT exposed either way "
           "(build_agent_head always passes n_queries explicitly). Second "
           "spellings found: %s"
           % (n, N_QUERIES_DEFAULT, up_ok, dup or "none"),
           {"agent_queries": n, "upstream_default": int(N_QUERIES_DEFAULT),
            "upstream_covers_train_max": bool(up_ok),
            "duplicate_spellings": dup})


def check_gradient_reachability():
    """*** THE CHECK THAT CATCHES A STAMPED WEIGHT THAT TRAINS NOTHING. ***

    For every agent-side loss term the trainer can weight, drive it on
    synthetic boxes and MEASURE the parameter gradient. A term whose gradient
    is indistinguishable from zero is a term whose weight is decoration.

    Carries its own positive control: if EVERY term reads flat the probe is
    broken and the result is INCONCLUSIVE, not a finding.
    """
    try:
        import torch
        from tanitad.data import calib
        from tanitad.data.rig_projection import RigCamera
        from tanitad.refs import refc_agents as ra
    except Exception as ex:                                   # noqa: BLE001
        record("every weighted term has a gradient", None, str(ex))
        return
    cam = RigCamera.nominal(calib.PHYSICALAI_WIDE120_256x640, height_m=1.5)
    g = torch.Generator().manual_seed(0)
    B, N = 4, 16

    def boxes():
        cx = torch.rand(B, N, generator=torch.Generator().manual_seed(0)) \
            * 50.0 + 5.0
        cy = (torch.rand(B, N, generator=torch.Generator().manual_seed(1))
              * 2.0 - 1.0) * 14.0
        return torch.stack([cx, cy, torch.full_like(cx, 4.5),
                            torch.full_like(cx, 1.9)], -1).requires_grad_(True)

    idx = torch.arange(N)
    match = {"rows": [idx] * B, "cols": [idx] * B,
             "n_dropped": [0] * B, "n_target": [N] * B}
    grads = {}
    b1 = boxes()
    pj = ra.monocular_projection_loss(b1, boxes().detach() + 1.0, cam, match)
    grads["agent_w_project"] = (
        float(torch.autograd.grad(pj["loss"], b1)[0].abs().max())
        if pj["n"] else 0.0, int(pj["n"]))
    b2 = boxes()
    gp = ra.ground_range_prior(b2, cam)
    grads["agent_w_ground"] = (
        float(torch.autograd.grad(gp["loss"], b2)[0].abs().max())
        if gp["n"] and gp["loss"].requires_grad else 0.0, int(gp["n"]))
    floor = 1e-7
    live = {k: v for k, v in grads.items() if v[0] > floor}
    dead = {k: v for k, v in grads.items() if v[0] <= floor}
    if not live:
        record("every weighted term has a gradient", None,
               "EVERY term read a flat gradient -- the probe itself is "
               "suspect, so this is INCONCLUSIVE (= a failure), not a "
               "finding. %s" % grads)
        return
    detail = "; ".join("%s grad=%.3e n=%d" % (k, v[0], v[1])
                       for k, v in sorted(grads.items()))
    record("every weighted term has a gradient", not dead,
           detail + (" | DEAD: %s -- the weight would be stamped into "
                     "config.json and add exactly 0 to the total"
                     % sorted(dead) if dead else "")
           + " | control: %d of %d terms are live" % (len(live), len(grads)),
           {"grads": {k: v[0] for k, v in grads.items()},
            "dead_terms": sorted(dead)})


def check_seam_refusals(t):
    """Each guard must FAIL its defect. A guard never shown to fire is a
    guard nobody has tested."""
    if t is None:
        record("the seam guards refuse their defects", None, "no trainer")
        return
    from tanitad.refs import refc_v3 as v3
    cases = [
        ("--agents head with --w-agent 0",
         ["--agents", "head", "--w-agent", "0"]),
        ("--agents off with --w-agent 1",
         ["--agents", "off", "--w-agent", "1.0"]),
        ("--agent-w-project with no camera",
         ["--agents", "oracle", "--agent-w-project", "0.2"]),
        ("--sampler ddim with --w-u0 0",
         ["--sampler", "ddim", "--w-u0", "0"]),
    ]
    fired, missed = [], []
    for name, extra in cases:
        try:
            a = t.build_parser().parse_args(
                ["--out", "x", "--arm", "hier", "--image-hw", "256", "640",
                 *extra])
            t._pin_trainer_cfg(v3.refc_v3_sized_config("tiny", hier=True), a)
            missed.append(name)
        except SystemExit:
            fired.append(name)
        except Exception as ex:                               # noqa: BLE001
            missed.append("%s (%s)" % (name, type(ex).__name__))
    # the same-breath CONTROL: a legal config must NOT refuse
    ctl_ok = True
    try:
        a = t.build_parser().parse_args(
            ["--out", "x", "--arm", "hier", "--image-hw", "256", "640"])
        t._pin_trainer_cfg(v3.refc_v3_sized_config("tiny", hier=True), a)
    except SystemExit:
        ctl_ok = False
    if not ctl_ok:
        record("the seam guards refuse their defects", None,
               "the CONTROL (a plain legal config) also refused -- the probe "
               "is measuring something other than the guards")
        return
    record("the seam guards refuse their defects", not missed,
           "%d/%d guards fired; control (legal config) passed"
           % (len(fired), len(cases))
           + ("; DID NOT FIRE: %s" % missed if missed else ""))


def check_knob_closure(t):
    """A run whose record cannot state its own weights must refuse to start,
    and the assertion must be DERIVED from argparse so it cannot rot."""
    if t is None:
        record("every knob reaches config.json", None, "no trainer")
        return
    try:
        dests = t.agent_knob_dests(t.build_parser())
        a = t.build_parser().parse_args(["--out", "x", "--arm", "hier"])
        stamp = t.agent_knob_stamp(a)
        leaves = json.dumps(stamp)
        missing = [d for d in dests if d not in leaves]
    except Exception as ex:                                   # noqa: BLE001
        record("every knob reaches config.json", None,
               "%s: %s" % (type(ex).__name__, ex))
        return
    record("every knob reaches config.json",
           bool(dests) and not missing,
           "%d knobs derived from argparse, %d missing from the stamp%s; "
           "control: n_knobs > 12 = %s"
           % (len(dests), len(missing),
              (" (%s)" % missing) if missing else "", len(dests) > 12),
           {"n_knobs": len(dests)})


def check_camera_scope(t, extrinsics):
    """A per-clip mount pose is retraction class C28. The record must say
    which scope it used, and a per-clip table must be resolvable per row."""
    if t is None:
        record("the camera scope is stated", None, "no trainer")
        return None
    from tanitad.refs import refc_v3 as v3
    try:
        a = t.build_parser().parse_args(["--out", "x", "--arm", "hier"])
        _, stamp = t._build_rig_camera(
            t._pin_trainer_cfg(v3.refc_v3_sized_config("tiny", hier=True), a),
            a)
        has_key = "mount_pose_scope" in stamp
    except Exception as ex:                                   # noqa: BLE001
        record("the camera scope is stated", None, str(ex))
        return None
    if not extrinsics:
        record("the camera scope is stated", None,
               "no --rig-extrinsics given, so the PER-CLIP path is UNCHECKED. "
               "The default stamp carries mount_pose_scope=%s (key present: "
               "%s). A parity arm training the projection term needs a "
               "per-clip table -- one camera for the corpus is C28."
               % (stamp.get("mount_pose_scope"), has_key))
        return None
    try:
        a = t.build_parser().parse_args(
            ["--out", "x", "--arm", "hier", "--image-hw", "256", "640",
             "--agents", "oracle", "--agent-rig-camera", "extrinsics",
             "--agent-rig-extrinsics", str(extrinsics),
             "--agent-w-project", "0.2"])
        cam, stamp = t._build_rig_camera(
            t._pin_trainer_cfg(v3.refc_v3_sized_config("tiny", hier=True), a),
            a)
    except SystemExit as ex:
        record("the camera scope is stated", False,
               "the extrinsics file was REFUSED: %s" % ex)
        return None
    from tanitad.refs import refc_agents as ra
    per_clip = isinstance(cam, ra.RigCameraBank)
    record("the camera scope is stated", bool(stamp.get("mount_pose_scope")),
           "mount_pose_scope=%s n_clips=%s per_clip_bank=%s"
           % (stamp.get("mount_pose_scope"), stamp.get("n_clips"), per_clip),
           {"mount_pose_scope": stamp.get("mount_pose_scope"),
            "n_clips": stamp.get("n_clips")})
    return cam if per_clip else None


# ======================================================================== B --
# B. THE DATA: is there a parity corpus, and does the join reach it?
# =========================================================================== #

def check_v2_cache(v2):
    """Is there a v2 cache with STABLE ids, and is it the PARITY corpus?"""
    if not v2:
        record("v2 cache present with stable ids", None,
               "no --v2-cache given. Parity is "
               "physicalai-train-e438721ae894 (2,376 eps, skip-hash "
               "f09e44db); without a cache no real arm can start.")
        return None
    p = Path(v2)
    if not p.is_dir():
        record("v2 cache present with stable ids", False,
               "%s is not a directory" % p)
        return None
    eps = sorted(p.glob("*.v2ep.pt"))
    ctl = sorted(p.glob("*"))                       # same-breath control
    if not ctl:
        record("v2 cache present with stable ids", None,
               "the directory listed ZERO entries -- indistinguishable from "
               "a failed read on this mount")
        return None
    parity = any(h in str(p) for h in PARITY_HINTS)
    man = p / "_v2manifest.pt"
    record("v2 cache present with stable ids", bool(eps) and parity,
           "%d *.v2ep.pt (control: %d entries in the dir), manifest %s, "
           "path names the parity key: %s"
           % (len(eps), len(ctl), "present" if man.exists() else "ABSENT",
              parity),
           {"n_v2ep": len(eps), "parity_in_path": parity})
    try:
        from tanitad.data.v2_dataset import stable_episode_id
        ids = {stable_episode_id(f.name.split(".")[0]): f.name for f in eps}
        record("episode ids are collision-free", len(ids) == len(eps),
               "%d distinct stable ids for %d files (control: a 16-bit legacy "
               "key collides on 34/2308 train clips)" % (len(ids), len(eps)))
        return ids
    except Exception as ex:                                   # noqa: BLE001
        record("episode ids are collision-free", None, str(ex))
        return None


def check_clip_ids(clip_ids, cached):
    """The corpus clip list, verified against the committed parity digest.

    ⭐ Coverage must be measured against the CORPUS, not against the slice of
    it that is cached on this box. `--v2-cache` on a dev box held 96 of 2,400
    parity episodes; a camera table covering all 96 would have read 100 %.
    """
    if not clip_ids:
        record("the corpus clip list is the PARITY set", None,
               "no --clip-ids given, so coverage below is measured against "
               "whatever the local cache holds (%d episodes) -- which is a "
               "denominator, not the corpus" % len(cached or {}))
        return None
    p = Path(clip_ids)
    b, err = _read(p)
    if b is None:
        record("the corpus clip list is the PARITY set", None,
               "could not read %s: %s" % (p, err))
        return None
    try:
        import hashlib
        from tanitad.data import parity as pa
        from tanitad.data.v2_dataset import stable_episode_id
        ids = sorted(json.loads(b.decode("utf-8")))
        dig = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
        ent = pa.manifest_entry("physicalai-train-e438721ae894") or {}
        want = (ent.get("clip_membership") or {}).get("clip_id_sha256_sorted")
    except Exception as ex:                                   # noqa: BLE001
        record("the corpus clip list is the PARITY set", None, str(ex))
        return None
    ok = (want is not None and dig == want)
    record("the corpus clip list is the PARITY set", ok,
           "%d clip_ids, sha256 %s vs committed %s -> %s (control: the "
           "manifest names %s episodes after the 24-clip skip)"
           % (len(ids), dig[:16], (want or "NONE")[:16],
              "MATCH" if ok else "MISMATCH", ent.get("episode_count")),
           {"n_clips": len(ids), "parity_digest_match": ok})
    return {stable_episode_id(c): c for c in ids} if ok else None


def check_join(join, ep_ids):
    """Is the join JOINABLE to that cache, through the STABLE id?"""
    if not join:
        record("agent join joins to the cache", None,
               "no --agent-join given; --agents head REFUSES without it, so "
               "E-AGT-HEAD cannot train")
        return
    p = Path(join)
    if not p.exists():
        record("agent join joins to the cache", False, "%s missing" % p)
        return
    try:
        sys.path.insert(0, str(HERE))
        from train_p8_occupancy import JoinFileReader
    except Exception as ex:                                   # noqa: BLE001
        record("agent join joins to the cache", None,
               "JoinFileReader unavailable: %s" % ex)
        return
    try:
        r = JoinFileReader(str(p))
    except Exception as ex:                                   # noqa: BLE001
        record("agent join joins to the cache", False,
               "%s: %s" % (type(ex).__name__, ex))
        return
    n_rec = int(getattr(r, "n_records", 0))
    n_clip = int(getattr(r, "n_clips", 0))
    if n_rec == 0:
        record("agent join joins to the cache", None,
               "the reader returned 0 records -- INCONCLUSIVE, not empty")
        return
    if not ep_ids:
        record("agent join joins to the cache", None,
               "join reads %d records / %d clips, but there is no v2 cache "
               "to join it AGAINST -- coverage is UNCHECKED"
               % (n_rec, n_clip), {"n_records": n_rec, "n_clips": n_clip})
        return
    stable = sum(1 for i in ep_ids if i in getattr(r, "_clip_of_uid", {}))
    legacy = sum(1 for i in ep_ids
                 if i not in getattr(r, "_clip_of_uid", {})
                 and i in getattr(r, "_clip_of_legacy", {}))
    frac = stable / max(len(ep_ids), 1)
    record("agent join joins to the cache", frac > 0.95,
           "%d/%d episodes join on the STABLE id (%.4f); %d only on the "
           "COLLIDING 16-bit legacy key; join carries %d records / %d clips"
           % (stable, len(ep_ids), frac, legacy, n_rec, n_clip),
           {"stable_join_frac": frac, "n_legacy_only": legacy})


def check_join_digest(join):
    """A recorded md5 carries the ARTIFACT it covers, or it is a number: the
    two joins' sidecars digest DIFFERENT artifacts (.xz vs .jsonl)."""
    if not join:
        record("join digest declares its scope", None, "no --agent-join")
        return
    side = Path(str(join) + ".meta.json")
    if not side.exists():
        record("join digest declares its scope", None,
               "no sidecar at %s -- the digest scope is UNDECLARED, and a "
               "checker inherited from the other join REFUSES a good file"
               % side.name)
        return
    try:
        from tanitad.data import join_meta as jm
        meta = json.loads(side.read_text(encoding="utf-8"))
        scope = jm.declared_scope(meta) if hasattr(jm, "declared_scope") \
            else meta.get("digest_scope")
    except Exception as ex:                                   # noqa: BLE001
        record("join digest declares its scope", None, str(ex))
        return
    record("join digest declares its scope", scope is not None,
           "declared scope: %r" % (scope,))


def check_v7_labels(labels, ep_ids):
    """The trainer REFUSES below 50 % v7.2 coverage. On parity it is 7.92 %."""
    if not labels:
        record("v7.2 tactical labels cover the corpus", None,
               "no --v7-labels given. The trainer refuses --v7-labels below "
               "50 % coverage; MEASURED on parity it is 7.92 %, so a parity "
               "arm either drops --nav-from-v7 and the 8x8 vocabulary, or "
               "waits for a parity v7.2 build.")
        return
    p = Path(labels)
    if not p.exists():
        record("v7.2 tactical labels cover the corpus", False, "%s missing" % p)
        return
    try:
        import gzip
        from tanitad.data.v2_dataset import stable_episode_id
        opener = gzip.open if str(p).endswith(".gz") else open
        sids = set()
        with opener(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                cid = json.loads(line).get("clip_id")
                if cid:
                    sids.add(stable_episode_id(str(cid)))
    except Exception as ex:                                   # noqa: BLE001
        record("v7.2 tactical labels cover the corpus", None, str(ex))
        return
    if not sids:
        record("v7.2 tactical labels cover the corpus", None,
               "0 clip_ids read -- INCONCLUSIVE, not absence")
        return
    if not ep_ids:
        record("v7.2 tactical labels cover the corpus", None,
               "%d labelled clips, but no v2 cache to measure coverage "
               "against" % len(sids))
        return
    hit = sum(1 for i in ep_ids if i in sids)
    frac = hit / max(len(ep_ids), 1)
    record("v7.2 tactical labels cover the corpus", frac >= 0.50,
           "%d/%d episodes labelled (%.4f); the trainer's floor is 0.50 "
           "(control: %d labelled clips read from the file)"
           % (hit, len(ep_ids), frac, len(sids)),
           {"v7_coverage": frac})


def check_camera_coverage(bank, ep_ids):
    if bank is None:
        record("per-clip cameras cover the corpus", None,
               "no per-clip camera bank was built, so coverage is UNCHECKED")
        return
    if not ep_ids:
        record("per-clip cameras cover the corpus", None,
               "no v2 cache episode ids to measure coverage against")
        return
    cov = bank.coverage(list(ep_ids))
    record("per-clip cameras cover the corpus", cov["n_missing"] == 0,
           "%d/%d episodes have their own camera (%.4f); bank holds %d clips"
           % (cov["n_covered"], cov["n"], cov["frac"], len(bank)),
           {"camera_coverage": cov["frac"]})


def check_anchors(anchors):
    """An artifact that does not name its UNITS is inadmissible: the same
    column read as curvature instead of lateral acceleration gives 396 g."""
    if not anchors:
        record("anchor vocabulary declares its units", None,
               "no --anchors given. A run without one silently falls back to "
               "the SYNTHETIC default, whose oracle-in-vocabulary ADE is "
               "1.0882 m against 0.3796 m -- the ceiling below the floor.")
        return
    p = Path(anchors)
    if not p.exists():
        record("anchor vocabulary declares its units", False, "%s missing" % p)
        return
    try:
        import torch
        d = torch.load(str(p), map_location="cpu", weights_only=False)
        from tanitad.refs import anchor_meta as am
        meta = am.read_meta(d) if hasattr(am, "read_meta") else None
        units = (meta or {}).get("control_units") if isinstance(meta, dict) \
            else (d.get("control_units") if isinstance(d, dict) else None)
        has_ctrl = isinstance(d, dict) and d.get("controls") is not None
    except Exception as ex:                                   # noqa: BLE001
        record("anchor vocabulary declares its units", None, str(ex))
        return
    ok = (units is not None) or not has_ctrl
    record("anchor vocabulary declares its units", ok,
           "controls present: %s; declared control_units: %r "
           "(a controls-carrying file that declares nothing must be REFUSED "
           "unless --anchor-control-units is passed by name)"
           % (has_ctrl, units))


# ======================================================================== C --
# C. END TO END: does a 2-step run write a config that states every weight?
# =========================================================================== #

def check_smoke(t):
    out = Path(tempfile.mkdtemp(prefix="refcv5_pf_"))
    cmd = [sys.executable, str(HERE / "refc_v3_train.py"),
           "--arm", "hier", "--out", str(out), "--smoke", "--steps", "2",
           "--device", "cpu", "--synth-episodes", "4",
           "--agents", "oracle", "--agent-queries", "8"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="6",
               PYTHONPATH=str(STACK) + os.pathsep + str(REPO / "taniteval"))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                           env=env, errors="replace")
    except Exception as ex:                                   # noqa: BLE001
        record("a 2-step run states every weight", None,
               "the smoke could not run: %s" % ex)
        return
    cfg_p = out / "config.json"
    if r.returncode != 0 or not cfg_p.exists():
        record("a 2-step run states every weight", False,
               "exit %d; config.json %s; tail: %s"
               % (r.returncode, "written" if cfg_p.exists() else "ABSENT",
                  (r.stderr or r.stdout or "")[-400:].replace("\n", " | ")))
        return
    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    seams = cfg.get("seams", {})
    knobs = seams.get("agent_knobs")
    cam = seams.get("agent_rig_camera", {})
    need = ["w_agent", "w_u0", "agent_queries", "agent_rig_camera"]
    miss = [k for k in need if knobs is None or k not in json.dumps(knobs)]
    record("a 2-step run states every weight",
           bool(knobs) and not miss and bool(cam),
           "config.json: %d knobs stamped, camera scope %r, missing %s "
           "(control: seams block has %d keys)"
           % (len(knobs or {}), cam.get("mount_pose_scope"), miss or "none",
              len(seams)),
           {"smoke_out": str(out), "n_knobs_stamped": len(knobs or {})})


# =========================================================================== #

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--v2-cache")
    ap.add_argument("--clip-ids",
                    help="JSON list of the corpus clip_ids. The HONEST "
                         "coverage denominator: a local v2 cache may hold "
                         "a fraction of the corpus, and a coverage number "
                         "over that fraction is a scope error.")
    ap.add_argument("--agent-join")
    ap.add_argument("--v7-labels")
    ap.add_argument("--anchors")
    ap.add_argument("--rig-extrinsics")
    ap.add_argument("--smoke", action="store_true",
                    help="also run a 2-step CPU training and read its "
                         "config.json back (~1-2 min)")
    ap.add_argument("--json", help="write the full result table here")
    a = ap.parse_args(argv)

    print("== refcv5 preflight ==  repo %s" % REPO, flush=True)
    print("-- A. the code ------------------------------------------------")
    t = check_imports()
    check_class_enum()
    check_query_budget(t)
    check_gradient_reachability()
    check_seam_refusals(t)
    check_knob_closure(t)
    bank = check_camera_scope(t, a.rig_extrinsics)
    print("-- B. the data ------------------------------------------------")
    ep_ids = check_v2_cache(a.v2_cache)
    corpus_ids = check_clip_ids(a.clip_ids, ep_ids)
    ep_ids = corpus_ids or ep_ids
    check_join(a.agent_join, ep_ids)
    check_join_digest(a.agent_join)
    check_v7_labels(a.v7_labels, ep_ids)
    check_camera_coverage(bank, ep_ids)
    check_anchors(a.anchors)
    if a.smoke:
        print("-- C. end to end ----------------------------------------------")
        check_smoke(t)
    else:
        record("a 2-step run states every weight", None,
               "--smoke not passed, so the end-to-end record was NOT checked")

    n_pass = sum(1 for r in RESULTS if r["state"] == "PASS")
    n_fail = sum(1 for r in RESULTS if r["state"] == "FAIL")
    n_inc = sum(1 for r in RESULTS if r["state"] == "INCONCLUSIVE")
    print("\n== %d PASS  %d FAIL  %d INCONCLUSIVE (inconclusive counts as a "
          "failure) ==" % (n_pass, n_fail, n_inc), flush=True)
    verdict = "GO" if (n_fail + n_inc) == 0 else "NO-GO"
    print("== VERDICT: %s ==" % verdict, flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"verdict": verdict, "n_pass": n_pass, "n_fail": n_fail,
             "n_inconclusive": n_inc, "checks": RESULTS}, indent=2),
            encoding="utf-8")
        print("wrote %s" % a.json, flush=True)
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
