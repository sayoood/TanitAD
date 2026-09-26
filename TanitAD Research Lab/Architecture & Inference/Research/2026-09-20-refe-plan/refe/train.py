"""REFe training loop, with an OVERFIT TEST as its control.

THE CONTROL, and why it is the right one for a training loop. A loop can run, print a falling
average, and still be broken -- the optimizer holding no trainable tensors, the graph detached, the
target mis-shaped. The discriminating check is whether it can DRIVE THE LOSS TO ~0 ON A HANDFUL OF
SAMPLES. A correct loop memorises 8 tuples easily; a broken one plateaus. It also ships a
deliberate-regression arm (`--break-optimizer`) that hands the optimizer an empty parameter set: if
the overfit test still "passes" under that, the test is inert and proves nothing.

IMAGES. `--synthetic` substitutes deterministic noise images keyed by the tuple index, for validating
the MECHANICS before the bulk CAM_F0 fetch lands. It is deliberately loud: any run started with it
prints that its loss numbers are about the loop and not about driving.

Usage:
  python train.py --overfit --synthetic              # the control, ~1 min on the 4060
  python train.py --overfit --synthetic --break-optimizer   # must FAIL
  python train.py --targets <dir> --images <root> --steps 2000
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import BACKBONES, REFe, REFeConfig, param_report, wta_loss  # noqa: E402
import ckpt_io  # noqa: E402


CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")


class FrameStore:
    """Read a frame from ANY of the rig's cameras, from a loose file OR a per-(log, camera) zip.

    exFAT on D: uses 1 MiB allocation units, so 26,060 loose 204 KB JPEGs cost 25 GB of disk for
    5.08 GB of data (MEASURED, 5.0x). Containers remove that because the loss is per FILE. Handles
    are cached per log; a zip read is a seek plus a copy, so this is not slower in practice.

    ⛔ TWO FOUR-CAMERA DEFECTS FIXED HERE 2026-09-21 (Review 5, D8).

    1. The container path was hardcoded to `{log_name}_CAM_F0.zip`, so under four cameras every
       L0/R0/B0 frame missed its container.
    2. Worse, and the reason this was a silent failure rather than a slow one: on `KeyError` it
       **returned None instead of falling through** to the loose-file path below. So the mere
       PRESENCE of a `_CAM_F0.zip` at the root made every non-front frame unreadable **even when
       its loose file was sitting right there** — a container for one camera disabling the other
       three. The container lookup is now a *try*, never a verdict.
    """

    def __init__(self, root: str | None):
        self.root = root
        self._zips: dict[str, object] = {}
        self._members: dict[str, set] = {}          # container -> member names, read once
        self._containers: set | None = None         # containers present at root, listed ONCE
        self._index: dict[str, str] | None = None   # basename -> path of every loose frame, ONCE

    def container_set(self) -> set:
        """The `<log>_<CAM>.zip` containers at the root, from ONE directory listing.

        Before 2026-09-23 every frame probed four container paths with `os.path.exists`, i.e. 16
        stats per 4-camera sample for containers that, on the pod's loose layout, never exist.
        """
        if self._containers is None:
            try:
                self._containers = ({n for n in os.listdir(self.root) if n.endswith(".zip")}
                                    if self.root and os.path.isdir(self.root) else set())
            except OSError:
                self._containers = set()
        return self._containers

    def index(self) -> dict:
        """basename -> path for every loose frame under the root, built ONCE by one walk.

        ⛔⛔ THIS REPLACES A RECURSIVE GLOB THAT RAN PER IMAGE (R21, 2026-09-23). OpenScene's
        `navtrain_current_N.tgz` extracts as `navtrain_current_N/<log>/<CAM>/<hash>.jpg`, while a
        row names `<log>/<CAM>/<hash>.jpg` -- so the direct path always MISSED on the pod and every
        read fell through to `glob(root/**/name)`: a walk of the whole ~400 K-file pixel tree, four
        times per sample, for ~4.4 M sample-passes. Nothing errors -- the run is just slower by a
        factor set by the tree size. The member name is a per-image 64-bit hash, so a basename
        index is exact regardless of how deep the extraction put it.
        """
        if self._index is None:
            idx: dict[str, str] = {}
            if self.root and os.path.isdir(self.root):
                for dp, _dn, fns in os.walk(self.root):
                    for fn in fns:
                        if fn.endswith((".jpg", ".jpeg", ".png")):
                            idx.setdefault(fn, os.path.join(dp, fn))
            self._index = idx
        return self._index

    def _member_set(self, zp: str) -> set:
        if zp not in self._members:
            z = self._open(zp)
            try:
                self._members[zp] = set(z.namelist()) if z is not None else set()
            except Exception:
                self._members[zp] = set()
        return self._members[zp]

    def resolvable(self, image_ref: str, log_name: str) -> bool:
        """Would `read` find this frame? Answered from the listings -- never a per-frame stat."""
        if "::" in image_ref:
            zp, member = image_ref.split("::", 1)
            if member in self._member_set(zp):
                return True
        name = os.path.basename(image_ref.split("::", 1)[-1])
        if self.root:
            have = self.container_set()
            for c in CAMERAS:
                fn = f"{log_name}_{c}.zip"
                if fn in have and name in self._member_set(os.path.join(self.root, fn)):
                    return True
            return name in self.index()
        return os.path.isabs(image_ref) and os.path.exists(image_ref)

    @staticmethod
    def camera_of(image_ref: str) -> str | None:
        for c in CAMERAS:
            if c in image_ref:
                return c
        return None

    def _open(self, zp: str):
        import zipfile
        z = self._zips.get(zp)
        if z is None and os.path.exists(zp):
            try:
                z = self._zips[zp] = zipfile.ZipFile(zp)
            except Exception:
                return None
        return z

    def read(self, image_ref: str, log_name: str) -> bytes | None:
        # build_targets.index_images emits "<container.zip>::<member>" for containerised frames
        if "::" in image_ref:
            zp, member = image_ref.split("::", 1)
            z = self._open(zp)
            if z is not None:
                try:
                    return z.read(member)
                except KeyError:
                    pass                      # fall through rather than declare the frame missing
        name = os.path.basename(image_ref)
        if self.root:
            cam = self.camera_of(image_ref)
            have = self.container_set()
            # try this frame's own camera first, then every other channel's container: the member
            # name is a per-image hash, so a hit in the "wrong" container is still the right image.
            for c in ([cam] if cam else []) + [c for c in CAMERAS if c != cam]:
                if f"{log_name}_{c}.zip" not in have:
                    continue                  # listed once at start: absent containers cost nothing
                z = self._open(os.path.join(self.root, f"{log_name}_{c}.zip"))
                if z is None:
                    continue
                try:
                    return z.read(name)
                except KeyError:
                    continue                  # ⛔ was `return None` -- see the docstring
        p = image_ref if os.path.isabs(image_ref) else os.path.join(
            self.root or "", *image_ref.split("/"))
        if os.path.exists(p):
            with open(p, "rb") as f:
                return f.read()
        q = self.index().get(name) if self.root else None
        if q is not None:
            with open(q, "rb") as f:
                return f.read()
        return None


KMAX = 16          # padding width for the per-frame candidate set (the set is 8-9 today)

# ⛔ WHY THE CANDIDATE NAME HAS TO SURVIVE INTO THE LOOP. `assign_d`, the distance from a banked
# candidate to the model proposal it is attached to, has a FLOOR THAT IS NOT ZERO and is set by the
# candidate set, not by model quality: a competent planner will never propose the `over-curb` path,
# so that row stays far away however well training goes. A single pooled mean therefore cannot
# distinguish "the supervision is landing" from "the model is bad", and would read as a permanent
# failure either way. Per-candidate is the only reading that separates them: `teacher` and `lat+-2`
# SHOULD converge, `over-curb` and `stopped` SHOULD NOT.
CAND_NAMES = ("teacher", "lat-4", "lat-2", "lat+2", "lat+4",
              "lon x0.5", "lon x1.5", "stopped", "jerky", "reverse", "over-curb")
CAND_INDEX = {n: i for i, n in enumerate(CAND_NAMES)}


def _scorer_files(path) -> list:
    """The scorer bank's files: every `scorer_targets*.jsonl` of a directory, or one named file."""
    if not path:
        return []
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "scorer_targets*.jsonl")))
    return [path] if os.path.isfile(path) else []


# ⭐ GROWING BANK (2026-09-24; PI: "try to parallelize training and data prep"). With --grow the run
# starts while the augmentation search and the scorer are still producing rows, and RE-READS the bank
# at every epoch boundary. What one epoch reads is fixed by a BYTE SNAPSHOT of every bank file taken
# at that boundary and stored in the checkpoint: a resume inside the epoch rebuilds the identical
# bank, and rows appended after the snapshot simply belong to a later epoch. The bank files are
# append-only (code/grow_assemble.py), which is what makes a byte length a complete description.
def bank_snapshot(targets_dir: str, scorer_path) -> dict:
    snap = {}
    for q in sorted(Path(targets_dir).glob("targets_rank*.jsonl")):
        snap["t:" + q.name] = q.stat().st_size
    for q in _scorer_files(scorer_path):
        snap["s:" + os.path.basename(q)] = os.path.getsize(q)
    return snap


def read_upto(path, nbytes: int) -> list:
    """The complete lines in the first `nbytes` of `path`; a trailing partial line is not a row."""
    size = os.path.getsize(path)
    if size < nbytes:
        raise SystemExit(f"bank file {path} is SHORTER ({size:,} B) than its snapshot ({nbytes:,} B): "
                         f"it was rewritten, so it is no longer the bank this run trained on")
    with open(path, "rb") as f:
        buf = f.read(nbytes)
    cut = buf.rfind(b"\n")
    return [] if cut < 0 else [l for l in buf[:cut + 1].decode("utf-8").splitlines() if l.strip()]


class ScorerBank:
    """The six-component PDM targets, keyed by (log_name, step). Built by build_scorer_targets.py.

    ⛔ THE APPROXIMATION, STATED RATHER THAN HIDDEN. DriveZero scores the STUDENT'S OWN 64
    proposals online; we precompute targets for a fixed candidate set and attach each candidate's
    target to the model proposal NEAREST to it. The supervision is therefore only as good as that
    assignment, so the training loop REPORTS the mean assignment distance every log line. A large
    distance means the scorer is being taught about trajectories the model does not produce.

    ⛔ AND AN INERT COMPONENT IS WORSE THAN A MISSING ONE, so this class measures the per-component
    variance across the whole bank at load time and names any component that never varies. A
    constant target trains a constant bias and looks exactly like a component that has been
    learned. MEASURED 2026-09-20: `goal_reaching` is constant across every candidate including a
    stationary ego, which is why this check exists.
    """

    # ⛔ THE OLD TUPLE WAS ("off_road", "collision", "ttc", "comfort", "goal_reaching",
    # "center_line") AND TWO OF ITS SLOTS WERE MIS-MAPPED, which is worse than being wrong: the
    # names read as the paper's components, so nobody re-checked them.
    #   * `goal_reaching` stood in for PROGRESS. It is a TERMINAL indicator of coming within 3.0 m
    #     of a goal point >= 60 m ahead, so over a 4.0 s candidate it fires for whichever travels
    #     furthest -- it REWARDED OVERSPEED (lon x1.5 45/74 vs teacher 13/74).
    #   * `center_line` stood in for DRIVING-DIRECTION COMPLIANCE. It is lateral deviation with
    #     heading explicitly unused, while the real DDC detector writes categories 3 and 4 into
    #     `OffRoad.info` -- which we never read, so DDC supervision was ZERO.
    # These are the paper's own six, in its own words.
    COMPONENTS = ("collision", "drivable_area", "progress", "ttc", "comfort", "ddc")

    def __init__(self, path: str, sizes: dict | None = None):
        self.by: dict = {}
        self.n_rows = 0
        self.files: list = []
        self.n_legacy_norank = 0
        # a DIRECTORY loads every shard: parallel builder passes at different frame offsets each
        # write their own file, and a bank assembled from shards must not depend on which one you
        # happened to name.
        files = _scorer_files(path)
        if not files:
            return
        if sizes is not None:                   # --grow: exactly the epoch snapshot's files + bytes
            files = [f for f in files if "s:" + os.path.basename(f) in sizes]
        self.files = files
        for fpath in files:
          with open(fpath, encoding="utf-8") as fh:
            src = (read_upto(fpath, sizes["s:" + os.path.basename(fpath)]) if sizes is not None
                   else fh)
            for line in src:
                r = json.loads(line)
                # ⛔ THE KEY MUST CARRY THE TOKEN. `log_name` is the DRIVE LOG, and one drive log
                # holds SEVERAL scenarios: MEASURED 2026-09-20, 50 of 154 (log, step) keys collided,
                # one log carrying 3 distinct tokens and another 2. Keyed without the token, a
                # frame's candidates were silently pooled with a DIFFERENT scenario's candidates at
                # the same step index, and the scorer was supervised on trajectories belonging to
                # another scene. ⭐ It was caught by ARITHMETIC, not by reading the code: the
                # per-candidate report showed `teacher` with n = 837 against only 605 covered
                # samples, which is impossible when each sample contributes one teacher row. It also
                # inflated the teacher's assignment distance to 5.73 m while the winner-takes-all
                # distance to the same trajectory was 0.57.
                # ⛔ AND THE RANK. The paper requires the navigation command, the teacher
                # trajectory and the proposal scores to be MUTUALLY CONSISTENT -- all three from
                # the SAME augmented route. A key without the rank hands a rank-1 training tuple
                # the rank-0 proposal scores, which is a silent mislabelling that no loss curve can
                # show. MEASURED 2026-09-20 by the conformance review: 540 of 1,080 covered tuples
                # were exactly that, while the two routes' goals differed in 1,064/1,091 tuples
                # (median 3.80 m, max 28.84 m) and their teacher targets by a median of 2.418 m.
                # ⚠️ A row with no `rank` field predates the fix. The earlier version of this
                # comment said the rank-0 assumption "is asserted in the builder's stats file" --
                # ⛔ and NO CODE READS THAT FILE, and it is not rank-suffixed either, so the
                # assertion existed only in prose. That is the same shape as an allow-list entry
                # nobody re-reads. A legacy row is now COUNTED and reported at load time, so the
                # assumption is visible in the run rather than buried in a comment.
                k = (r["log_name"], r.get("token", ""), int(r["step"]), int(r.get("rank", 0)))
                e = self.by.setdefault(k, {"traj": [], "tgt": [], "name": []})
                if "rank" not in r:
                    self.n_legacy_norank += 1
                e["traj"].append(r["traj"])
                e["tgt"].append(self.components(r["targets"]))
                e["name"].append(r["candidate"])
                self.n_rows += 1

    @staticmethod
    def components(t: dict) -> list:
        """PDM convention: 1.0 = NO violation, 0.0 = violation. Soft values stay soft."""
        def g(k, d=0.0):
            v = t.get(k, d)
            if v is None:
                return d
            try:
                return float(v)
            except Exception:
                return d
        clip = lambda x: min(max(x, 0.0), 1.0)          # noqa: E731
        # ⛔ `OffRoad.info` IS A CATEGORY, NOT A FLAG: 0 none / 1 off-road / 2 solid-line crossing
        # / 3 wrong-way / 4 severe wrong-way. The old code did `1 - clip(info)`, which collapses
        # 2, 3 and 4 onto the same value as 1 and silently destroys the distinction. Harmless while
        # only {0, 1} occur in the bank, and live the moment a richer scene appears. Compare the
        # category explicitly instead.
        # the two questions are now derived per PREFIX in score_proposals and aggregated as
        # booleans, so this consumer no longer re-reads a max-aggregated nominal category
        dac_viol = g("dac.violation", None)
        ep = g("progress.ep", float("nan"))
        return [
            1.0 - clip(g("collision.NuPlanCollision.info")),        # collision avoidance
            (1.0 - clip(dac_viol)) if dac_viol is not None                 # drivable-area
            else (0.0 if int(round(g("off_road.OffRoad.info"))) == 1 else 1.0),
            clip(ep) if ep == ep else 1.0,                          # progress (EP, teacher-relative)
            clip(g("ttc.NuPlanTTC.ttc_reward", 1.0)),               # time to collision
            clip(g("comfort.Comfort.reward", 1.0)),                 # comfort
            1.0 - clip(g("ddc.violation")),                         # driving-direction compliance
        ]

    def variance_report(self):
        """Can each component RANK PROPOSALS? That is a WITHIN-FRAME question.

        ⛔ THIS GUARD WAS WRONG AND A DOC QUOTED IT AS PROOF. The first version pooled every row in
        the bank and reported one standard deviation per component. A component that is CONSTANT
        across the candidates of every single frame, and merely differs between frames, then shows
        a healthy pooled spread and passes -- while being completely unable to rank anything.
        MEASURED 2026-09-20 by the conformance review: `comfort` varies across candidates in
        **0 of 540 frames** and the pooled guard passed it at sd 0.1111, which `REFE_MODEL.md` then
        cited as evidence that all six components discriminate.
        ⭐ The scorer's job is to choose among the candidates OF ONE FRAME, so the only measurement
        that answers the question is the spread WITHIN a frame. This reports the fraction of frames
        in which each component varies at all, and names anything below 5 % as unable to rank.
        Same family as the four self-referential checks of 2026-09-07: a check whose question is
        narrower than the claim hung on it.
        """
        if not self.by:
            return []
        import statistics
        n_frames = len(self.by)
        out = []
        for i, name in enumerate(self.COMPONENTS):
            vals, n_vary = [], 0
            for e in self.by.values():
                col = [row[i] for row in e["tgt"]]
                vals += col
                if len(col) > 1 and (max(col) - min(col)) > 1e-9:
                    n_vary += 1
            frac = n_vary / max(n_frames, 1)
            pooled = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            out.append((name, sum(vals) / len(vals), pooled, frac, frac < 0.05))
        return out

    def get(self, log_name: str, token: str, step: int, rank: int = 0):
        e = self.by.get((log_name, token, int(step), int(rank)))
        if e is None:
            return None
        return e["traj"], e["tgt"], e["name"]


class TargetBank(Dataset):
    def __init__(self, target_dir: str, images_root: str | None, cfg: REFeConfig,
                 synthetic: bool = False, limit: int | None = None, spread: bool = False, scorer=None,
                 max_missing_frac: float = 0.001, calib: str | None = None,
                 sizes: dict | None = None):
        self.scorer = scorer
        self.rows = []
        loaded = sorted(Path(target_dir).glob("targets_rank*.jsonl"))
        if sizes is not None:                   # --grow: exactly the epoch snapshot's files + bytes
            loaded = [p for p in loaded if "t:" + p.name in sizes]
        for p in loaded:
            if sizes is not None:
                self.rows += [json.loads(l) for l in read_upto(p, sizes["t:" + p.name])]
                continue
            with open(p, encoding="utf-8") as f:
                self.rows += [json.loads(l) for l in f]
        # ⛔⛔ A TARGET FILE LEFT IN THE DIRECTORY BUT NOT LOADED IS A SILENTLY DROPPED ARM.
        # MEASURED 2026-09-23: the diversity search writes `targets_aug.jsonl`, which the glob above
        # never matched -- so the goal augmentation (the lever the paper credits with +0.80 PDMS)
        # would have been dropped while training ran clean on rank 0 alone. Refuse instead.
        # against the FULL rank glob, not `loaded`: under --grow a rank file that appeared after the
        # snapshot is simply not in this epoch -- it is not a stray
        stray = sorted(set(Path(target_dir).glob("targets_*.jsonl"))
                       - set(Path(target_dir).glob("targets_rank*.jsonl")))
        if stray:
            raise SystemExit(f"TargetBank: {[s.name for s in stray]} sit in {target_dir} but do not "
                             f"match targets_rank*.jsonl and would be SILENTLY DROPPED. Rename them "
                             f"(augmented rows carry rank 1 -> targets_rank1.jsonl).")
        by_rank = {}
        for r in self.rows:
            by_rank[int(r.get("rank", 0))] = by_rank.get(int(r.get("rank", 0)), 0) + 1
        print(f"  targets: {len(self.rows):,} rows from {[p.name for p in loaded]}  "
              f"per rank {dict(sorted(by_rank.items()))}")
        if limit:
            # MEASURED 2026-09-20: taking the FIRST `limit` rows takes CONSECUTIVE frames of ONE
            # scenario, 0.1 s apart. Their pairwise L2 is 7.40 against 294.29 for independent noise
            # -- 39.8x closer -- so a synthetic arm gets a perfect per-index lookup key while the
            # real arm must map near-identical inputs to targets differing by ~1 m. That makes the
            # overfit control WEAK and makes real-vs-synthetic look like an image-difficulty result
            # when it is a sampling artifact. `spread` takes an even stride across the whole bank,
            # which draws from different scenarios.
            if spread and len(self.rows) > limit:
                step = len(self.rows) // limit
                self.rows = [self.rows[i * step] for i in range(limit)]
            else:
                self.rows = self.rows[:limit]
        self.images_root = images_root
        self.store = FrameStore(images_root)
        self.cfg = cfg
        self.synthetic = synthetic
        self.n_dropped_missing = 0
        # ⛔⛔ RESOLVE EVERY FRAME AT LOAD, NOT ON DAY 20 (R21). `__getitem__` raises on a missing
        # frame; under a supervisor that is a CRASH LOOP -- the resume restores the exact data
        # position, re-reads the same tuple and dies again until the restart budget is gone. A few
        # absent frames are dropped HERE, counted and named; a fraction above `max_missing_frac`
        # is a broken JOIN (wrong root, wrong split, wrong layout) and is refused outright.
        if not synthetic and images_root and self.rows:
            t_res = time.time()
            keep, miss, first_bad = [], [], []
            for r in self.rows:
                refs = r["image"] if isinstance(r["image"], list) else [r["image"]]
                bad = [x for x in refs if not self.store.resolvable(x, r.get("log_name", ""))]
                (miss if bad else keep).append(r)
                if bad and len(first_bad) < 3:
                    first_bad.append(bad[0])          # name the frame that is ACTUALLY missing
            frac = len(miss) / max(len(self.rows), 1)
            print(f"  images: {len(keep):,}/{len(self.rows):,} tuples resolve ALL cameras under "
                  f"{images_root}  ({len(self.store.index()):,} loose frames indexed, "
                  f"{len(self.store.container_set())} containers, {time.time() - t_res:.1f} s)")
            if miss:
                ex = first_bad
                if frac > max_missing_frac:
                    raise SystemExit(
                        f"TargetBank: {len(miss):,} of {len(self.rows):,} tuples "
                        f"({100 * frac:.2f} %) name a frame that does not resolve under "
                        f"{images_root} -- above the {100 * max_missing_frac:.2f} % limit, so this "
                        f"is a broken JOIN, not a few absent frames. e.g. {ex}")
                print(f"  *** DROPPED {len(miss):,} tuples ({100 * frac:.3f} %) with an "
                      f"unresolvable frame, e.g. {ex}")
                self.n_dropped_missing = len(miss)
                self.rows = keep
        # ⭐ PER-SAMPLE CALIBRATION (R22). PETR -- the paper's cited 3D position embedding --
        # builds each sample's embedding from ITS OWN rig. Without a table every tuple is lifted
        # with the one baked vehicle (MEASURED exact on 3.2 % of local logs, rotation off by up to
        # 2.67 deg). A tuple whose log has no calibration is dropped and counted like a missing
        # frame -- never silently given the baked rig, which would re-open the defect per tuple.
        self.calib_table = None
        self.n_dropped_calib = 0
        if calib and self.rows:
            import calib_table as CT
            self.calib_table = CT.load(calib)
            keep, miss = [], []
            for r in self.rows:
                v = CT.vectors(self.calib_table, r.get("log_name", ""), cfg.cameras)
                if v is None:
                    miss.append(r)
                else:
                    r["_calib"] = v
                    keep.append(r)
            frac = len(miss) / max(len(self.rows), 1)
            n_rigs = len({tuple(map(tuple, r["_calib"])) for r in keep})
            print(f"  calib: {len(keep):,}/{len(self.rows):,} tuples carry their own rig "
                  f"({n_rigs} distinct rigs) <- {calib}")
            if miss:
                ex = sorted({m.get("log_name") for m in miss})[:3]
                if frac > max_missing_frac:
                    raise SystemExit(f"TargetBank: {len(miss):,} tuples ({100 * frac:.2f} %) have no "
                                     f"calibration for their log in {calib} -- above the "
                                     f"{100 * max_missing_frac:.2f} % limit. e.g. {ex}")
                print(f"  *** DROPPED {len(miss):,} tuples with no calibration, e.g. {ex}")
                self.n_dropped_calib = len(miss)
                self.rows = keep
        if not self.rows:
            raise RuntimeError(f"no tuples under {target_dir}")
        # ⭐ SCENES (R24). Rows sharing (log_name, token, step) are ONE scene: its rank-0 target plus
        # any goal-augmented twins. The key is the scorer bank's key without the rank, so it means
        # the same frame for the navtrain source (token = frame, step = 0) and the older val14 one
        # (token = scenario, step = frame). Members are ordered by rank: member 0 = logged intent.
        idx: dict = {}
        for i, r in enumerate(self.rows):
            idx.setdefault((r.get("log_name", ""), r.get("token", ""), int(r.get("step", 0))),
                           []).append(i)
        self.scene_groups = [sorted(v, key=lambda i: int(self.rows[i].get("rank", 0)))
                             for v in idx.values()]
        # ⛔ a scene holding the SAME rank twice is a duplicated row (an overlapping shard merge),
        # not a twin -- it would silently double one target's weight. The pipeline guarantees
        # uniqueness (resume keys on the same tuple), so a violation is a defect: refuse it.
        dup = [g for g in self.scene_groups
               if len({int(self.rows[i].get("rank", 0)) for i in g}) != len(g)]
        if dup:
            r0 = self.rows[dup[0][0]]
            raise SystemExit(f"TargetBank: {len(dup):,} scenes hold the same rank twice (duplicated "
                             f"rows), e.g. {r0.get('log_name')} {r0.get('token')} step "
                             f"{r0.get('step')}. Fix the bank merge; do not train on it.")
        sizes: dict = {}
        for g in self.scene_groups:
            sizes[len(g)] = sizes.get(len(g), 0) + 1
        print(f"  scenes: {len(self.scene_groups):,} over {len(self.rows):,} tuples  (rows per "
              f"scene: {', '.join(f'{k} x {v:,}' for k, v in sorted(sizes.items()))})")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        c = self.cfg
        if self.synthetic:
            g = torch.Generator().manual_seed(i)          # deterministic per index
            img = (torch.randn(c.n_cameras, 3, c.img_h, c.img_w, generator=g)
                   if c.n_cameras > 1 else torch.randn(3, c.img_h, c.img_w, generator=g))
        else:
            import cv2
            # ⭐ FOUR CAMERAS. A tuple's "image" is a LIST of paths, one per camera, in the order
            # `build_targets.CAMERAS` records. A legacy single-path row still loads, and the
            # camera-count guard in REFe.forward is what stops a 1-camera tuple being trained
            # through a 4-camera model by accident.
            paths = r["image"] if isinstance(r["image"], list) else [r["image"]]
            frames = []
            for pth in paths:
                blob = self.store.read(pth, r.get("log_name", ""))
                if blob is None:
                    raise FileNotFoundError(f"{pth} (log {r.get('log_name')})")
                im = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
                if im is None:
                    raise ValueError(f"undecodable frame: {pth}")
                im = cv2.resize(im, (c.img_w, c.img_h))[:, :, ::-1].astype(np.float32) / 255.0
                frames.append(torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))))
            img = torch.stack(frames) if len(frames) > 1 else frames[0]
        # the per-frame candidate set, PADDED to KMAX with a mask. A frame with no scorer targets
        # returns an all-zero mask and contributes NOTHING to the scorer loss -- which is why the
        # loop prints coverage: a silently empty bank would just look like a small loss.
        ct = torch.zeros(KMAX, c.horizon_steps, 2, dtype=torch.float32)
        cg = torch.zeros(KMAX, 6, dtype=torch.float32)
        cm = torch.zeros(KMAX, dtype=torch.float32)
        ci = torch.full((KMAX,), -1, dtype=torch.long)
        got = (self.scorer.get(r.get("log_name", ""), r.get("token", ""), r.get("step", -1),
                               int(r.get("rank", 0)))
               if self.scorer else None)
        if got is not None:
            trs, tgs, nms = got
            for j, (tr, tg, nm) in enumerate(zip(trs, tgs, nms)):
                if j >= KMAX:
                    break
                ci[j] = CAND_INDEX.get(nm, -1)
                a = torch.tensor(tr, dtype=torch.float32)[:c.horizon_steps, :2]
                ct[j, :a.shape[0]] = a
                cg[j] = torch.tensor(tg, dtype=torch.float32)
                cm[j] = 1.0
        # ⛔ REFUSE A BANK WHOSE EGO WIDTH IS NOT THE MODEL'S, AND SAY WHICH BANK.
        # `goal` two lines below has always been SLICED to the config, so a goal mismatch passed
        # silently; `ego` was neither sliced nor checked, so an `ego_dim` move showed up as
        # `mat1 and mat2 shapes cannot be multiplied (2x44 and 45x256)` -- a matmul error 3 frames
        # deep that names neither the bank nor the field. ⚠️ Deliberately NOT padded or truncated:
        # a 7-D kinematics vector and an 8-D one with a trailing pad are different corpora, and
        # quietly reshaping one into the other is how a model trains on a column of zeros.
        ego_t = torch.tensor(r["ego"], dtype=torch.float32)
        if ego_t.numel() != c.ego_dim:
            raise RuntimeError(
                f"target bank has {ego_t.numel()}-D ego but the model is configured for "
                f"{c.ego_dim}-D (log {r.get('log_name','?')} step {r.get('step','?')}). "
                f"Rebuild the bank with build_targets.py, or set REFeConfig.ego_dim to match. "
                f"These two must move together -- see diag_consumer_conformance.py.")
        # float64 on purpose: the rig is a cache KEY in the model, and float32 would round every
        # vehicle's numbers differently from the table's -- same geometry, different key
        cal = (torch.tensor(r["_calib"], dtype=torch.float64) if "_calib" in r
               else torch.zeros(0, dtype=torch.float64))
        return (img,
                ego_t,
                torch.tensor(r["goal"], dtype=torch.float32)[:2 * c.n_goal_points],
                torch.tensor(r["traj"], dtype=torch.float32),
                ct, cg, cm, ci, cal)


class EpochSampler(torch.utils.data.Sampler):
    """A seeded permutation PER EPOCH that can start mid-epoch.

    ⛔ The DataLoader's own `shuffle=True` draws its permutation from the GLOBAL torch RNG at the
    start of each epoch, and a resumed process cannot rewind to "sample 41,216 of epoch 7" -- it
    would replay the epoch from its start, training twice on what it had already seen and never
    on the rest. Seeding by (seed, epoch) makes every epoch's order a pure function of two
    integers, so a checkpoint only has to record (epoch, position).
    """

    def __init__(self, n: int, seed: int, shuffle: bool = True):
        self.n, self.seed, self.shuffle = int(n), int(seed), bool(shuffle)
        self.epoch, self.start = 0, 0

    def set_epoch(self, epoch: int, start: int = 0) -> None:
        self.epoch, self.start = int(epoch), int(start)

    def order(self) -> list:
        if not self.shuffle:
            return list(range(self.n))
        g = torch.Generator()
        g.manual_seed(self.seed * 1_000_003 + self.epoch)
        return torch.randperm(self.n, generator=g).tolist()

    def __iter__(self):
        return iter(self.order()[self.start:])

    def __len__(self) -> int:
        return max(self.n - self.start, 0)


class SceneEpochSampler(torch.utils.data.Sampler):
    """ONE ROW PER SCENE PER EPOCH -- the paper's unit (PI decision 2026-09-23, R24).

    DriveZero counts its training data in SCENES ("100K navtrain + 237K SimScale scenes", Table
    A12) and trains "25 epochs". Our bank stores each goal-augmented target as an extra ROW beside
    its scene's rank-0 row (~1.71 rows per scene at tau 0.3), so an epoch over ROWS spends 1.71x
    the paper's budget -- ~64-69 A40-days instead of ~37-41 -- and visits scenes that have an
    augmented twin twice as often as the ~29 % that have none.

    Every epoch visits every scene exactly once, in a seeded order, using ONE of its rows: member
    `(epoch + offset[scene]) mod k`, with `offset` a seeded per-scene constant. Over E epochs each
    member of a k-row scene is therefore used floor/ceil(E/k) times (balanced WITHIN the scene),
    and inside any single epoch the members are MIXED across scenes -- ⛔ not "rank 0 on even
    epochs, rank 1 on odd", which would make whole epochs single-goal. Pure function of
    (seed, epoch), so a checkpoint's (epoch, position) still resumes exactly.
    """

    def __init__(self, groups: list, seed: int, shuffle: bool = True):
        self.groups = [list(g) for g in groups]
        self.n = len(self.groups)
        self.seed, self.shuffle = int(seed), bool(shuffle)
        self.epoch, self.start = 0, 0
        g0 = torch.Generator()
        g0.manual_seed(self.seed * 7_919 + 104_729)
        self.offset = torch.randint(0, 1 << 30, (max(self.n, 1),), generator=g0).tolist()

    def set_epoch(self, epoch: int, start: int = 0) -> None:
        self.epoch, self.start = int(epoch), int(start)

    def order(self) -> list:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed * 1_000_003 + self.epoch)
            scenes = torch.randperm(self.n, generator=g).tolist()
        else:
            scenes = range(self.n)
        return [self.groups[q][(self.epoch + self.offset[q]) % len(self.groups[q])]
                for q in scenes]

    def __iter__(self):
        return iter(self.order()[self.start:])

    def __len__(self) -> int:
        return max(self.n - self.start, 0)


# Arguments that DEFINE a run. A resume that changes any of them is not a resume -- it is a new
# run wearing the old one's step counter -- so it is refused, not reconciled. Paths are recorded
# but not compared (a bank moved to another disk is the same bank); the bank's CONTENT is.
IDENTITY_ARGS = ("backbone", "synthetic", "n_cameras", "undistort", "detach_scorer_context",
                 "batch", "accum", "lr", "weight_decay", "cosine", "epochs", "steps", "score_w",
                 "seed", "spread", "overfit", "overfit_n", "epoch_unit", "amp", "tf32",
                 "fused_adam", "compile", "grow", "grow_scenes")
# what a GROWING bank changes by design; each epoch's own snapshot is in the checkpoint instead
GROW_EXEMPT = ("n_tuples", "n_scenes", "per_rank", "targets_sha256", "scorer_sha256")


def _digest_files(paths) -> str:
    h = hashlib.sha256()
    for q in paths:
        h.update(os.path.basename(str(q)).encode("utf-8"))
        with open(q, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
    return h.hexdigest()


def run_identity(a, ds, scorer_bank, frozen_sha256) -> dict:
    """What a resume must match: the defining arguments, the bank's content, and the trunk."""
    tfiles = sorted(Path(a.targets).glob("targets_rank*.jsonl"))
    per_rank: dict = {}
    for r in ds.rows:
        k = str(int(r.get("rank", 0)))
        per_rank[k] = per_rank.get(k, 0) + 1
    grow = bool(getattr(a, "grow", False))
    return {"args": {k: getattr(a, k, None) for k in IDENTITY_ARGS},
            "n_tuples": len(ds), "n_scenes": len(ds.scene_groups), "per_rank": per_rank,
            # not hashed under --grow: the files are still being appended to (GROW_EXEMPT)
            "targets_sha256": None if grow else _digest_files(tfiles),
            "scorer_sha256": (None if grow else
                              _digest_files(getattr(scorer_bank, "files", []) or [])),
            "calib_sha256": _digest_files([a.calib]) if a.calib else None,
            "frozen_sha256": frozen_sha256}


def _identity_diff(old: dict, new: dict) -> list:
    out = []
    for k in IDENTITY_ARGS:
        if old["args"].get(k) != new["args"].get(k):
            out.append(f"--{k.replace('_', '-')}: {old['args'].get(k)!r} -> {new['args'].get(k)!r}")
    grow = bool(new["args"].get("grow")) and bool(old["args"].get("grow"))
    for k in ("n_tuples", "n_scenes", "per_rank", "targets_sha256", "scorer_sha256",
              "calib_sha256", "frozen_sha256"):
        if grow and k in GROW_EXEMPT:
            continue
        if old.get(k) != new.get(k):
            ov, nv = old.get(k), new.get(k)
            if isinstance(ov, str) and len(ov) == 64:
                ov, nv = ov[:12], (nv or "")[:12]
            out.append(f"{k}: {ov!r} -> {nv!r}")
    return out


def _rng_state() -> dict:
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(), "python": random.getstate()}


def _set_rng_state(st: dict) -> None:
    torch.set_rng_state(st["torch"])
    if st.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append_jsonl(path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def run(a) -> int:
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    # ⛔⛔ A TRAINING RUN THAT SAVES NOTHING IS REFUSED. Until 2026-09-23 (R20) this trainer had no
    # checkpoint, no final model and no resume: the planned ~60-day ViT-L run would have ended with
    # its weights in process memory only, and any pod restart would have cost all of it. Controls
    # and smoke runs (<= 500 steps, or --overfit) may still run without a run directory.
    if a.resume and not a.out:
        print("  REFUSING: --resume needs --out (the run directory that holds ckpt_last.pt).")
        return 4
    if not a.out and not a.overfit and (a.epochs > 0 or a.steps > 500):
        print("  REFUSING TO TRAIN WITHOUT --out: this run would save NO checkpoint and NO model. "
              "Pass --out <run dir> (and --resume under a supervisor).")
        return 4
    # ⭐ SPEED FLAGS (2026-09-23, first REFe pod; PI: training must fit <= 15 days). MEASURED on the
    # A40, ViT-L, 4 cameras, the real step (refe/bench_speed.py): FP32 batch 4 1.282 s/sample, TF32
    # 1.007, bf16 autocast 0.366 -- 3.5x. ⚠️ Table A12 says "Training precision FP32": --amp bf16 is a
    # DECLARED DEPARTURE, recorded in config.json and in the resume identity, and validated for
    # numerics against FP32 (refe/diag_amp.py) before it is used for a run.
    if a.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if a.amp != "none" or a.tf32 or a.fused_adam or a.workers or a.compile:
        print(f"  SPEED: amp={a.amp} tf32={a.tf32} compile={a.compile} fused_adam={a.fused_adam} "
              f"workers={a.workers}"
              + ("   (DEPARTURE from the paper's FP32)" if a.amp != "none" or a.tf32 else ""))
    cfg = REFeConfig.for_backbone(a.backbone)
    # ⭐ EVERY DEPARTURE IS A FLAG, AND EVERY FLAG IS PRINTED. A run that silently differs from the
    # default is how an ablation becomes an unexplained result three weeks later.
    if a.n_cameras is not None:
        cfg.n_cameras = a.n_cameras
        cfg.cameras = tuple(cfg.cameras[:a.n_cameras])
        print(f"  ABLATION: rig reduced to {a.n_cameras} camera(s) {cfg.cameras}")
    if not a.undistort:
        cfg.undistort = False
        print("  ABLATION: ideal-pinhole frustum (lens distortion NOT inverted)")
    if not a.detach_scorer_context:
        cfg.detach_scorer_context = False
        print("  ABLATION: the PDM loss reaches the backbone LoRA (REFe's extra detach removed)")
    torch.manual_seed(a.seed)
    model = REFe(cfg)

    # ⛔⛔ LOAD THE PRETRAINED TRUNK. UNTIL 2026-09-23 THIS FILE DID NOT, AND NOTHING NOTICED.
    # `REFe(cfg)` builds the backbone at RANDOM init; `load_dinov3.map_into_backbone` is the only
    # thing that fills it, and the sole importer of that module was `diag_rope.py`. The trainer
    # would therefore have trained a FROZEN RANDOM TRUNK -- the premise of the method inverted --
    # while the loss still fell, because the LoRA adapters and both heads learn regardless.
    # MEASURED on ViT-S: the mapper reports loaded=186 missing=0 unused=0 and **186 backbone
    # tensors MOVE**; `cls_token` and `reg_token` read all-zeros before it runs. ⚠️ The banked
    # overfit control (1.9316 -> 0.3730) would have passed identically -- overfitting 8 scenarios
    # does not need good features -- and `diag_rope` stayed green because IT loads the weights
    # itself. Advisory class A sitting inside class F.
    if a.weights != "none":
        import load_dinov3 as LD
        wdir = a.weights or os.path.join(LD.BACKBONE_ROOT, BACKBONES[a.backbone]["weights"])
        sd = LD.load_state(wdir)
        loaded, missing, unused = LD.map_into_backbone(model.backbone, sd)
        print(f"  trunk: loaded {len(loaded)} tensors from {wdir} "
              f"(missing {len(missing)}, unused {len(unused)})")
        # ⛔ A LOAD THAT SILENTLY NO-OPS IS THE SAME DEFECT WEARING A PRINT STATEMENT. Refuse.
        if not loaded:
            print("  REFUSING TO TRAIN: the checkpoint mapped 0 tensors into the backbone.")
            return 3
        if missing or unused:
            print(f"  REFUSING TO TRAIN: incomplete mapping -- missing={missing[:4]} "
                  f"unused={unused[:4]}. A partially-loaded trunk is not the published backbone.")
            return 3
    else:
        print("  *** --weights none: FROZEN RANDOM TRUNK. An ABLATION, not a reproduction; no "
              "number from it may be compared with the paper. ***")
    # fingerprint the FROZEN trunk once, on CPU, before the move: it is what makes a partial
    # checkpoint safe to complete (ckpt_io.load_partial refuses a different trunk).
    frozen_sha = ckpt_io.frozen_fingerprint(model) if a.out else None
    model = model.to(dev)
    if a.compile and dev == "cuda":
        # ⭐ nn.Module.compile() compiles IN PLACE: parameter names and state_dict keys are unchanged,
        # so checkpoints, resume and planner.py's strict load are unaffected. (`torch.compile(module)`
        # would return a wrapper whose keys all gain an `_orig_mod.` prefix.) MEASURED on the A40:
        # bf16 + compiled backbone 0.220 s/sample vs 0.366 eager (1.66x); ~2 min compile, once per shape.
        model.backbone.compile()
    rep = param_report(model)
    print(f"  REFe {a.backbone} {rep['total']/1e6:.2f} M total / {rep['trainable']/1e6:.2f} M "
          f"trainable ({rep['pct']:.2f} %)  device={dev}")
    # WTA gives gradient to ONE proposal per SAMPLE, so at most `batch` of M proposals are touched
    # per step. Printed because it is a property of the RUN, not of the model, and it is what
    # decides whether the 64 proposals specialise at all. Their training used batch 256.
    print(f"  WTA reach: at most {a.batch} of {cfg.n_proposals} proposals updated per step "
          f"({100*a.batch/cfg.n_proposals:.1f} %)")
    if a.synthetic:
        print("  *** SYNTHETIC IMAGES: this run validates the LOOP MECHANICS ONLY.")
        print("  *** Its loss says nothing about driving and must not be quoted as a result.")

    grow_snap = None
    if a.grow:
        if a.epochs and not a.grow_scenes:
            print("  REFUSING: --grow with --epochs needs --grow-scenes, the FINAL scene count the "
                  "schedule is sized for -- the bank is still growing, so its size now is not it.")
            return 4
        ck0 = Path(a.out) / "ckpt_last.pt" if a.out else None
        if a.resume and ck0 is not None and ck0.exists():
            # resuming INSIDE an epoch: that epoch's bank is the one it started with, byte for byte
            grow_snap = torch.load(ck0, map_location="cpu", weights_only=False)["state"].get(
                "bank_snapshot")
            if grow_snap is not None and os.environ.get("REFE_GROW_MUTATE_FRESH_SNAPSHOT") == "1":
                print("  *** DELIBERATE REGRESSION: resume IGNORES the checkpoint's bank snapshot")
                grow_snap = None
        src_ck = grow_snap is not None
        if grow_snap is None:
            grow_snap = bank_snapshot(a.targets, a.scorer_targets)
        print(f"  GROW: bank snapshot {len(grow_snap)} files, {sum(grow_snap.values()):,} bytes"
              f"{' (the checkpointed epoch snapshot)' if src_ck else ' (taken now)'}")
    scorer_bank = ScorerBank(a.scorer_targets, sizes=grow_snap)
    if scorer_bank.n_rows:
        print(f"  scorer bank: {scorer_bank.n_rows:,} rows over "
              f"{len(scorer_bank.by):,} frames  <- {a.scorer_targets}")
        if scorer_bank.n_legacy_norank:
            print(f"  *** {scorer_bank.n_legacy_norank:,} rows carry NO `rank` field and are being "
                  f"read as rank 0.")
            print(f"  *** Every rank-1 training tuple therefore MISSES, and coverage halves "
                  f"SILENTLY. Rebuild the bank with --rank before quoting a scorer number.")
        print(f"      {'component':14s} {'mean':>6s} {'pooled sd':>10s} {'varies in':>10s}"
              f"   <- 'varies in' is the fraction of FRAMES whose candidates differ at all")
        for name, mean, sd, frac, inert in scorer_bank.variance_report():
            flag = "  *** CANNOT RANK: constant within every frame" if inert else ""
            print(f"      {name:14s} {mean:6.3f} {sd:10.3f} {100*frac:9.1f} %{flag}")
    else:
        print("  scorer bank: NONE -- the scorer falls back to the zero PLACEHOLDER and "
              "proposal selection stays arbitrary")
    ds = TargetBank(a.targets, a.images, cfg, synthetic=a.synthetic,
                    limit=a.overfit_n if a.overfit else None, spread=a.spread,
                    scorer=scorer_bank if scorer_bank.n_rows else None,
                    max_missing_frac=a.max_missing_images, calib=a.calib, sizes=grow_snap)
    if not a.calib:
        print("  *** NO --calib: every tuple is lifted with the ONE baked rig (pre-R22 behaviour). "
              "An ABLATION -- PETR, the paper's cited embedding, uses each sample's own. ***")
    if a.epoch_unit == "scenes":
        sampler = SceneEpochSampler(ds.scene_groups, a.seed, shuffle=not a.overfit)
    else:
        sampler = EpochSampler(len(ds), a.seed, shuffle=not a.overfit)
    epoch_len = sampler.n                    # what ONE epoch passes over: scenes, or rows
    # workers decode JPEGs while the GPU runs; the ORDER still comes from the sampler in this process,
    # so resume stays exact. ⚠️ each worker is a fork holding the banks (~8 GiB at full scale,
    # copy-on-write pages get copied as rows are touched) -- keep workers small under a RAM cap.
    # under --grow the dataset is REPLACED at every epoch boundary, so its workers must not persist
    dl = DataLoader(ds, batch_size=a.batch, sampler=sampler, num_workers=a.workers,
                    drop_last=False, pin_memory=(dev == "cuda"),
                    persistent_workers=a.workers > 0 and not a.grow,
                    prefetch_factor=(4 if a.workers > 0 else None))
    print(f"  bank: {len(ds):,} tuples in {len(ds.scene_groups):,} scenes   epoch = one pass over "
          f"{epoch_len:,} {a.epoch_unit}   batch {a.batch}   "
          f"{'OVERFIT CONTROL on ' + str(a.overfit_n) + ' tuples' if a.overfit else 'training'}")
    # PAPER: "We train DriveZero for 25 EPOCHS with a batch size of 256." This trainer counts
    # STEPS, so until 2026-09-20 the paper's schedule could not be stated in its own unit at all
    # and the cosine T_max had nothing to span. Deriving steps from epochs here -- BEFORE the
    # optimiser is built -- is what makes "decayed to zero over the whole run" mean the run.
    if a.epochs:
        import math as _m
        # ⛔⛔ THIS DIVIDED BY `a.batch` ALONE UNTIL 2026-09-21, AND GRADIENT ACCUMULATION TURNED IT
        # INTO A SILENT 64x OVERRUN. `step` counts OPTIMISER steps; one optimiser step consumes
        # `batch * accum` SAMPLES. Dividing by the micro-batch therefore asked for `accum` times
        # too many steps, and the run consumed `accum` epochs for every epoch requested.
        # MEASURED: `--epochs 2` printed the IDENTICAL step count at `--accum 1` and `--accum 4`
        # while consuming 2 vs 8 epochs of data -- the two runs are indistinguishable from the log.
        # ⚠️ The documented recipe `--batch 4 --accum 64 --epochs 25` would have run **1,601
        # epochs**, and every wall-clock estimate derived from it was 64x optimistic.
        eff = max(a.batch, 1) * max(a.accum, 1)
        # --grow: the schedule is sized for the FINAL bank; epochs early in the run are shorter
        sched_len = a.grow_scenes if a.grow else epoch_len
        per_epoch = max(_m.ceil(sched_len / eff), 1)
        a.steps = a.epochs * per_epoch
        print(f"  schedule from EPOCHS: {a.epochs} x {per_epoch} optimiser steps/epoch "
              f"= {a.steps:,} steps  (effective batch {eff} = {a.batch} x {a.accum}; "
              f"paper: 25 epochs at batch 256)")
        print(f"    -> {a.steps * eff:,} sample-passes = {a.steps * eff / max(sched_len, 1):.2f} "
              f"epochs of {sched_len:,} {a.epoch_unit}   (must equal --epochs, or the derivation "
              f"is wrong){'  [--grow: sized for --grow-scenes, not the bank now]' if a.grow else ''}")

    params = [p for p in model.parameters() if p.requires_grad]
    if a.break_optimizer:
        print("  *** DELIBERATE REGRESSION: optimizer given an EMPTY parameter set.")
        print("  *** If the overfit control still PASSES below, the control is inert.")
        params = [torch.nn.Parameter(torch.zeros(1, device=dev))]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay,
                            fused=bool(a.fused_adam and dev == "cuda"))
    # PAPER: "AdamW and an initial learning rate of 2 x 10^-4 DECAYED TO ZERO VIA COSINE
    # ANNEALING". Until 2026-09-20 this trainer had NO scheduler at all and a default lr of 3e-4 --
    # an UNINTENDED divergence on both counts, fixed on PI instruction. eta_min is 0.0 because
    # "decayed to zero" is the paper's words, not a rounding of some small floor.
    sched = None
    if a.cosine:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.steps, 1), eta_min=0.0)
    print(f"  optimiser: AdamW lr {a.lr:g} wd {a.weight_decay:g}  schedule: "
          f"{'cosine -> 0 over ' + str(a.steps) + ' steps' if sched else 'NONE (--no-cosine)'}")

    first, last, t0 = None, None, time.time()
    step, n_cov, n_seen, assign_d = 0, 0, 0, float("nan")
    per_cand = [[0.0, 0] for _ in CAND_NAMES]
    # ⛔ A WHOLE-RUN MEAN CANNOT ANSWER THE QUESTION THIS TABLE IS FOR. The accumulator above spans
    # every step INCLUDING the untrained ones -- MEASURED 2026-09-20, step 0 read 11.89 m and the
    # last steps read ~5.4 m, so a run-average of 3.54 m for `teacher` is a blend of "the model was
    # bad at the start" and "the supervision is mis-assigned". Those are exactly the two things the
    # per-candidate report exists to separate, so the run-average silently re-conflates them.
    # ⇒ a TRAILING window, accumulated only over the last quarter of the run, is the number that
    # answers "where is the supervision landing NOW". Both are printed; neither is quoted alone.
    per_cand_late = [[0.0, 0] for _ in CAND_NAMES]
    late_from = int(a.steps * 0.75)
    n_uncovered = 0
    micro = 0
    # ⭐ THE CONSTANT SCORE NORMALISER, derived from the BANK rather than from any micro-batch, so
    # gradient accumulation is exactly equivalent to one large batch (see the note at `l_score`).
    # Mean covered candidates per SAMPLE across the whole training set, floored at 1 so an
    # uncovered bank cannot divide by zero.
    cov_norm = 1.0
    if scorer_bank is not None and getattr(scorer_bank, "by", None) and len(ds) > 0:
        # the bank keys COVERED frames; dividing the total banked candidates by the FULL dataset
        # size is exactly the mean covered candidates per sample, which is what the loss needs.
        cov_norm = max(sum(len(e["traj"]) for e in scorer_bank.by.values()) / float(len(ds)), 1e-3)
    print(f"  score-loss normaliser: {cov_norm:.4f} covered candidates/sample x batch "
          f"(CONSTANT, so --accum is exactly equivalent to one large batch)")
    opt.zero_grad(set_to_none=True)          # the window is opened here, then after every step
    if a.accum > 1:
        print(f"  gradient accumulation: {a.accum} micro-batches of {a.batch} "
              f"= effective batch {a.accum * a.batch}  (paper: 256)")

    # ⭐ RUN DIRECTORY. config.json (what ran) · metrics.jsonl (every logged step) · ckpt_last.pt
    # (resume: partial model + optimiser + schedule + data position + RNG) · snap_epochNNN.pt
    # (partial weights per epoch, for learning-curve evals) · model_final.pt (FULL, what
    # planner.py loads) · summary.json {"done": true} LAST -- the supervisor's off-switch.
    epoch, pos, elapsed0 = 0, 0, 0.0
    run_dir = Path(a.out) if a.out else None
    ident = None
    meta = {}
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        ident = run_identity(a, ds, scorer_bank, frozen_sha)
        if a.weights == "none":
            wdir_used = "none"
        else:
            import load_dinov3 as _LD
            wdir_used = a.weights or os.path.join(_LD.BACKBONE_ROOT,
                                                  BACKBONES[a.backbone]["weights"])
        meta = {"backbone": a.backbone, "weights_dir": wdir_used, "frozen_sha256": frozen_sha,
                "per_sample_calib": bool(a.calib),
                "n_cameras": cfg.n_cameras, "cameras": list(cfg.cameras)}
        ck_path = run_dir / "ckpt_last.pt"
        summ = run_dir / "summary.json"
        if summ.exists():
            try:
                done = bool(json.load(open(summ, encoding="utf-8")).get("done"))
            except Exception:
                done = False
            if done:
                print(f"  REFUSING: {summ} says this run is FINISHED (done=true). A supervisor "
                      f"relaunching it would overwrite a completed model. Use a new --out.")
                print("TRAIN_ALREADY_DONE")
                return 0
        if ck_path.exists() and not a.resume:
            print(f"  REFUSING: {ck_path} exists. Pass --resume to continue that run, or choose a "
                  f"new --out -- starting over would silently overwrite it.")
            return 4
        if a.resume and ck_path.exists():
            st = torch.load(ck_path, map_location="cpu", weights_only=False)
            diff = _identity_diff(st["identity"], ident)
            if diff:
                print("  REFUSING TO RESUME -- the run's identity changed:")
                for d_ in diff:
                    print(f"      {d_}")
                return 4
            ckpt_io.load_partial(model, st["model_partial"], st["meta"].get("frozen_sha256"))
            opt.load_state_dict(st["opt"])
            if sched is not None and st.get("sched") is not None:
                sched.load_state_dict(st["sched"])
            S = st["state"]
            step, micro, epoch, pos = S["step"], S["micro"], S["epoch"], S["pos"]
            first, last, elapsed0 = S["first"], S["last"], S["elapsed_s"]
            n_cov, n_seen, n_uncovered = S["n_cov"], S["n_seen"], S["n_uncovered"]
            per_cand, per_cand_late = S["per_cand"], S["per_cand_late"]
            _set_rng_state(st["rng"])
            print(f"  RESUMED {ck_path}: step {step:,}/{a.steps:,}  epoch {epoch}  sample "
                  f"{pos:,}/{len(ds):,}  (saved {st.get('saved_at', '?')})")
            _append_jsonl(run_dir / "metrics.jsonl",
                          {"event": "resume", "step": step, "epoch": epoch, "pos": pos,
                           "at": _now()})
        else:
            if a.resume:
                print(f"  --resume: no {ck_path.name} yet -- starting fresh")

            def _src(f):
                q = Path(__file__).resolve().parent / f
                return ckpt_io.sha256_file(q) if q.exists() else None
            cfg_out = {"argv": sys.argv,
                       "args": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                                    else str(v)) for k, v in vars(a).items()},
                       "identity": ident, "meta": meta,
                       "params": {"total": rep["total"], "trainable": rep["trainable"]},
                       "torch": torch.__version__,
                       "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                       "source_sha256": {f: _src(f) for f in
                                         ("train.py", "model.py", "load_dinov3.py", "ckpt_io.py")},
                       "started_at": _now()}
            with open(run_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg_out, f, indent=1)
            _append_jsonl(run_dir / "metrics.jsonl", {"event": "start", "at": _now()})

    def _state():
        return {"step": step, "micro": micro, "epoch": epoch, "pos": pos, "first": first,
                "last": last, "elapsed_s": time.time() - t0, "n_cov": n_cov, "n_seen": n_seen,
                "n_uncovered": n_uncovered, "per_cand": per_cand, "per_cand_late": per_cand_late,
                "bank_snapshot": grow_snap}

    def _bank_event():
        pr: dict = {}
        for r_ in ds.rows:
            pr[str(int(r_.get("rank", 0)))] = pr.get(str(int(r_.get("rank", 0))), 0) + 1
        return {"event": "bank", "epoch": epoch, "n_tuples": len(ds),
                "n_scenes": len(ds.scene_groups), "per_rank": pr,
                "scorer_frames": len(getattr(scorer_bank, "by", {}) or {}),
                "cov_norm": cov_norm, "snapshot_bytes": sum(grow_snap.values()) if grow_snap else None,
                "at": _now()}

    def _rebuild(snap):
        """--grow: the next epoch's bank, from a fresh snapshot (same construction as above)."""
        sb = ScorerBank(a.scorer_targets, sizes=snap)
        d = TargetBank(a.targets, a.images, cfg, synthetic=a.synthetic,
                       limit=a.overfit_n if a.overfit else None, spread=a.spread,
                       scorer=sb if sb.n_rows else None,
                       max_missing_frac=a.max_missing_images, calib=a.calib, sizes=snap)
        smp = (SceneEpochSampler(d.scene_groups, a.seed, shuffle=not a.overfit)
               if a.epoch_unit == "scenes" else EpochSampler(len(d), a.seed, shuffle=not a.overfit))
        ld = DataLoader(d, batch_size=a.batch, sampler=smp, num_workers=a.workers,
                        drop_last=False, pin_memory=(dev == "cuda"), persistent_workers=False,
                        prefetch_factor=(4 if a.workers > 0 else None))
        cn = 1.0
        if getattr(sb, "by", None) and len(d) > 0:
            cn = max(sum(len(e["traj"]) for e in sb.by.values()) / float(len(d)), 1e-3)
        return sb, d, smp, ld, cn

    def _save_ckpt():
        nbytes = ckpt_io.atomic_save(
            {"format": ckpt_io.FORMAT_PARTIAL, "model_partial": ckpt_io.partial_state(model),
             "meta": meta, "opt": opt.state_dict(),
             "sched": sched.state_dict() if sched is not None else None,
             "state": _state(), "rng": _rng_state(), "identity": ident, "saved_at": _now()},
            run_dir / "ckpt_last.pt")
        _append_jsonl(run_dir / "metrics.jsonl",
                      {"event": "ckpt", "step": step, "epoch": epoch, "pos": pos,
                       "bytes": nbytes, "at": _now()})

    t0 = time.time() - elapsed0          # printed elapsed continues across resumes
    last_ck = time.time()
    if a.grow and run_dir is not None:
        _append_jsonl(run_dir / "metrics.jsonl", _bank_event())
    while step < a.steps:
        sampler.set_epoch(epoch, start=pos)
        for img, ego, goal, tgt, cand_xy, cand_tg, cand_m, cand_id, cal in dl:
            img, ego, goal, tgt, cand_xy, cand_tg, cand_m, cand_id = (
                x.to(dev, non_blocking=True)
                for x in (img, ego, goal, tgt, cand_xy, cand_tg, cand_m, cand_id))
            # the rig stays on the CPU in float64: the model uses it as a cache key, not a tensor op
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(a.amp == "bf16" and dev == "cuda")):
                traj, score = model(img, ego, goal, calib=cal if cal.numel() else None)
            traj, score = traj.float(), score.float()      # every loss and assignment in FP32
            l_traj, idx = wta_loss(traj, tgt)
            n_cand = float(cand_m.sum())
            if n_cand > 0:
                # assign each banked candidate to the NEAREST model proposal and supervise THAT
                # proposal's six components. `assign_d` is printed because it BOUNDS the
                # approximation: a large distance means the scorer is being taught about
                # trajectories this model does not produce.
                d = ((traj[:, :, :, :2].unsqueeze(1) - cand_xy.unsqueeze(2)) ** 2)                     .sum(-1).mean(-1)                       # [B, K, M]
                near = d.argmin(-1)                          # [B, K]
                sel = torch.gather(score, 1,
                                   near.unsqueeze(-1).expand(-1, -1, score.shape[-1]))
                per = F.binary_cross_entropy_with_logits(sel, cand_tg, reduction="none").mean(-1)
                # ⛔⛔ NORMALISING BY THIS MICRO-BATCH'S OWN COVERED COUNT BREAKS ACCUMULATION.
                # Under `--accum N` the window's gradient should equal the gradient of the same
                # loss on the full effective batch, i.e. `sum_i(S_i) / sum_i(n_i)`. Dividing each
                # micro-batch by its OWN `n_i` and then by `accum` gives `mean_i(S_i / n_i)`, which
                # is only equal when every `n_i` is the same -- and coverage is ~19 %, so they
                # never are. MEASURED by the sixth review: the accumulated score term came out at
                # **0.7500** of its correct value (3 of 4 micro-batches covered, each scaled 1/4).
                # ⭐ THE FIX IS A CONSTANT DENOMINATOR. With a normaliser that does not depend on
                # the micro-batch, `sum_i(S_i / C) / accum == sum_i(S_i) / (C * accum)` exactly
                # matches the full-batch form, and a poorly-covered micro-batch correctly
                # contributes LESS gradient instead of being silently upweighted to parity.
                # `cov_norm` is the bank's mean covered candidates per SAMPLE (computed once at
                # load), times the batch -- so the loss keeps the scale it had before this change
                # and `--score-w` does not need retuning.
                denom = max(cov_norm * float(img.shape[0]), 1.0)
                l_score = (per * cand_m).sum() / denom * a.score_w
                assign_d = float(((d.min(-1).values.clamp(min=0).sqrt() * cand_m).sum()
                                  / cand_m.sum().clamp(min=1.0)))
                n_cov += int((cand_m.sum(1) > 0).sum())
                dmin = d.min(-1).values.clamp(min=0).sqrt()        # [B, K]
                for nm_i in range(len(CAND_NAMES)):
                    sel_m = (cand_id == nm_i) & (cand_m > 0)
                    if bool(sel_m.any()):
                        tot_i, cnt_i = float(dmin[sel_m].sum()), int(sel_m.sum())
                        per_cand[nm_i][0] += tot_i
                        per_cand[nm_i][1] += cnt_i
                        if step >= late_from:
                            per_cand_late[nm_i][0] += tot_i
                            per_cand_late[nm_i][1] += cnt_i
            else:
                # ⛔ A ZERO TARGET IS NOT A NEUTRAL FALLBACK -- it teaches the scorer that EVERY
                # proposal violates EVERYTHING, which is worse than no supervision at all. The
                # original placeholder did exactly that, and it was defensible only while no real
                # targets existed. Now that they do, an uncovered batch contributes NOTHING and is
                # COUNTED, so the fraction of steps with no supervision is visible instead of being
                # quietly filled with a wrong label.
                # MEASURED 2026-09-20: at 25 % frame coverage and batch 8, P(no covered sample in a
                # batch) = 0.75^8 = 10 %, so a tenth of all steps were being taught the wrong thing.
                l_score = score.sum() * 0.0          # keeps the graph, contributes no gradient
                n_uncovered += 1
                assign_d = float("nan")
            n_seen += int(img.shape[0])
            pos += int(img.shape[0])
            # ⭐ GRADIENT ACCUMULATION. `step` counts OPTIMISER steps, never micro-batches, so
            # `--steps` and the cosine schedule's `T_max` keep meaning the same thing whatever
            # `--accum` is. Dividing by accum makes the accumulated gradient the MEAN over the
            # effective batch, which is what a single large batch would have produced; clipping
            # then sees the real global norm rather than one micro-batch's.
            loss = (l_traj + l_score) / a.accum
            loss.backward()
            micro += 1
            if micro % a.accum:
                continue                     # still filling the window: no step, no step count
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if sched is not None:
                sched.step()
            first = first if first is not None else float(l_traj)
            last = float(l_traj)
            if step % a.log_every == 0:
                uniq = len(set(idx.tolist()))
                print(f"    step {step:5d}  traj_L1 {float(l_traj):8.4f}  score {float(l_score):7.4f}  "
                      f"winners {uniq}/{img.shape[0]}  assign_d {assign_d:6.2f} m  "
                      f"scorer_cov {100.0*n_cov/max(n_seen,1):5.1f} %  "
                      f"lr {opt.param_groups[0]['lr']:.2e}  {time.time()-t0:6.1f}s")
                if run_dir is not None:
                    _append_jsonl(run_dir / "metrics.jsonl", {
                        "step": step, "epoch": epoch, "pos": pos, "traj_L1": float(l_traj),
                        "score": float(l_score), "winners": uniq, "batch": int(img.shape[0]),
                        "assign_d": assign_d, "scorer_cov": n_cov / max(n_seen, 1),
                        "lr": opt.param_groups[0]["lr"], "elapsed_s": time.time() - t0,
                        "samples": n_seen,
                        "cuda_max_mem_gb": (torch.cuda.max_memory_allocated() / 2**30
                                            if dev == "cuda" else None)})
            step += 1
            # checkpoint only HERE, right after an optimiser step: every consumed sample is then
            # in the weights, so (epoch, pos) is exact and no half-filled window is lost or doubled
            if run_dir is not None and step < a.steps and (
                    time.time() - last_ck >= a.ckpt_every_min * 60.0
                    or (a.ckpt_every_steps and step % a.ckpt_every_steps == 0)):
                _save_ckpt()
                last_ck = time.time()
                if a.halt_after_steps and step >= a.halt_after_steps:
                    print(f"  TEST HALT after the checkpoint at step {step} (--halt-after-steps)")
                    print("TRAIN_HALTED")
                    return 7
            if step >= a.steps:
                break
        else:
            # the loader ran dry without a break: an EPOCH completed
            epoch += 1
            pos = 0
            if run_dir is not None:
                ckpt_io.atomic_save({"format": ckpt_io.FORMAT_PARTIAL,
                                     "model_partial": ckpt_io.partial_state(model),
                                     "meta": dict(meta, epoch=epoch, step=step), "saved_at": _now()},
                                    run_dir / f"snap_epoch{epoch:03d}.pt")
                _append_jsonl(run_dir / "metrics.jsonl",
                              {"event": "epoch", "epoch": epoch, "step": step, "at": _now()})
            if a.grow and step < a.steps:
                # the next epoch reads the bank AS IT IS NOW -- a new snapshot, a new sampler
                grow_snap = bank_snapshot(a.targets, a.scorer_targets)
                scorer_bank, ds, sampler, dl, cov_norm = _rebuild(grow_snap)
                print(f"  GROW epoch {epoch}: {len(ds):,} tuples in {len(ds.scene_groups):,} scenes, "
                      f"scorer frames {len(scorer_bank.by):,}, cov_norm {cov_norm:.4f}")
                if run_dir is not None:
                    _append_jsonl(run_dir / "metrics.jsonl", _bank_event())
            continue
        break

    print("")
    print(f"  scorer coverage: {n_cov:,} of {n_seen:,} samples carried PDM targets "
          f"({100.0*n_cov/max(n_seen,1):.1f} %)")
    print(f"  batches with NO scorer supervision at all: {n_uncovered} "
          f"(they contribute zero gradient, not a zero target)")
    if n_cov > 0:
        print("  assignment distance per candidate (the floor is set by the CANDIDATE SET, not by")
        print("  model quality -- teacher/lat should converge, over-curb/stopped should not):")
        print(f"      {'candidate':10s} {'n':>6s} {'whole run':>12s} "
              f"{'last 25 %':>12s}   (the trailing column is the one that answers the question)")
        for i, nm in enumerate(CAND_NAMES):
            tot, cnt = per_cand[i]
            ltot, lcnt = per_cand_late[i]
            if cnt:
                late = f"{ltot/lcnt:10.2f} m" if lcnt else f"{'n/a':>12s}"
                print(f"      {nm:10s} {cnt:6d} {tot/cnt:10.2f} m {late}")
    if n_cov == 0:
        print("  *** THE SCORER SAW NO REAL TARGETS. It trained against the zero placeholder and")
        print("  *** proposal selection at inference is still arbitrary. Do not report a selection")
        print("  *** result from this run.")
    drop = 100.0 * (1 - last / max(first, 1e-9))
    print(f"\n  traj L1: first {first:.4f} -> last {last:.4f}   ({drop:.1f} % reduction)")
    if a.overfit:
        ok = last < a.overfit_thresh
        print(f"  OVERFIT CONTROL: final L1 {last:.4f} < {a.overfit_thresh} ? "
              f"{'PASS -- the loop can learn' if ok else 'FAIL -- the loop cannot memorise 8 tuples'}")
        print("TRAIN_CONTROL_OK" if ok else "TRAIN_CONTROL_FAILED")
        return 0 if ok else 1
    if run_dir is not None:
        # order matters: resume state first, then the model, then the done-marker LAST, so a kill
        # anywhere in here leaves a run that a relaunch completes rather than one that lies
        _save_ckpt()
        fin = run_dir / "model_final.pt"
        ckpt_io.atomic_save({"format": ckpt_io.FORMAT_FULL, "model": ckpt_io.full_state(model),
                             "meta": dict(meta, step=step, epoch=epoch, finished_at=_now())},
                            fin)
        fin_sha = ckpt_io.sha256_file(fin)
        summary = {"done": True, "steps": step, "epochs_completed": epoch,
                   "samples_seen": n_seen, "first_traj_L1": first, "last_traj_L1": last,
                   "scorer_cov": n_cov / max(n_seen, 1), "model_final": fin.name,
                   "model_final_sha256": fin_sha, "finished_at": _now()}
        tmp = run_dir / "summary.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
        os.replace(tmp, run_dir / "summary.json")
        print(f"  saved {fin} (sha256 {fin_sha[:12]}) and summary.json (done=true)")
    print("TRAIN_DONE")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vitl16", choices=sorted(BACKBONES))
    ap.add_argument("--weights", default="",
                    help="pretrained DINOv3 dir; default is derived from --backbone. "
                         "'none' = frozen RANDOM trunk (an ablation, never a reproduction).")
    # ⛔ THE DEFAULT MOVED 2026-09-21. `refe_targets` is the SINGLE-CAMERA, 8-D-ego bank and is now
    # incompatible with the model by construction; `refe_targets_4cam` is the rebuild. Leaving the
    # old path as the default would have made every no-argument run fail on a bank nobody meant to
    # use -- and before the ego guard existed, fail as an unreadable matmul error.
    ap.add_argument("--targets", default="D:/Projects/TanitAD/data/refe_targets_4cam")
    ap.add_argument("--scorer-targets",
                    default="D:/Projects/TanitAD/data/refe_scorer_targets_full",
                    help="six-component PDM targets from build_scorer_targets.py; absent => "
                         "the zero placeholder, and selection at inference stays arbitrary")
    ap.add_argument("--images", default=None)
    ap.add_argument("--calib", default=None,
                    help="calib_table.json (refe/calib_table.py): each tuple's OWN camera rig for "
                         "the PETR position embedding. Absent -> the one baked rig (an ablation).")
    ap.add_argument("--max-missing-images", type=float, default=0.001,
                    help="fraction of tuples allowed to name an unresolvable frame; they are "
                         "DROPPED and counted. Above it the load is REFUSED as a broken join.")
    ap.add_argument("--grow", action="store_true",
                    help="GROWING BANK: re-read --targets/--scorer-targets at every epoch boundary "
                         "(up to a byte snapshot kept in the checkpoint, so resume stays exact) -- "
                         "training runs while the augmentation search and the scorer still produce "
                         "rows. The files must be append-only (code/grow_assemble.py).")
    ap.add_argument("--grow-scenes", type=int, default=0,
                    help="with --grow --epochs: the FINAL scene count the schedule is sized for")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--overfit-n", type=int, default=8)
    ap.add_argument("--overfit-thresh", type=float, default=0.5)
    ap.add_argument("--spread", action="store_true",
                    help="sample the overfit tuples ACROSS the bank instead of the first N "
                         "consecutive frames -- see TargetBank for why the default is weak")
    ap.add_argument("--break-optimizer", action="store_true")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    # ⛔ ACCUMULATION IS NOT A SPEED KNOB HERE -- IT DECIDES WHETHER THE 64 PROPOSALS EXIST.
    # The WTA objective gives gradient to exactly ONE proposal per SAMPLE, so a step at batch 4
    # touches at most 4 of 64 and the rest never move. MEASURED: at batch 2 the winner was 1/2 on
    # EVERY logged step -- both samples picking the same proposal. Their batch was 256, data
    # parallel over 16 H20s; on one A40 the same effective batch is `--batch 4 --accum 64`.
    # ⚠️ It does NOT reproduce batch 256 in every respect: the forward statistics are still those
    # of the micro-batch. It reproduces the GRADIENT, which is the part WTA coverage depends on.
    ap.add_argument("--accum", type=int, default=1,
                    help="micro-batches per optimiser step; effective batch = batch x accum")
    # a single-camera ablation is legitimate (it is what every result before 2026-09-20 used) and
    # must not require a source edit -- the camera guard's own message now points here.
    ap.add_argument("--n-cameras", type=int, default=None,
                    help="override the rig size, e.g. 1 for the single-front-camera ablation")
    ap.add_argument("--no-undistort", dest="undistort", action="store_false",
                    help="ablation: unproject with an IDEAL pinhole, as before 2026-09-21")
    ap.add_argument("--scorer-sees-trunk", dest="detach_scorer_context", action="store_false",
                    help="ablation: let the PDM loss reach the backbone LoRA (paper does not say)")
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="PAPER: initial 2e-4, decayed to zero via cosine annealing")
    ap.add_argument("--weight-decay", type=float, default=0.01,
                    help="PAPER Table A12: 0.01. Was 1e-4 here -- a 100x error, read as "
                         "'not stated' by an earlier review and refuted by reading the table.")
    ap.add_argument("--no-cosine", dest="cosine", action="store_false",
                    help="ablation only -- the paper's schedule is cosine to zero")
    ap.set_defaults(cosine=True)
    ap.add_argument("--epoch-unit", choices=("scenes", "rows"), default="scenes",
                    help="what ONE epoch passes over. 'scenes' (default, the paper's unit -- Table "
                         "A12 counts its data in scenes; PI 2026-09-23, R24): every scene once per "
                         "epoch, its goal-augmented targets taking turns across epochs. 'rows': "
                         "every stored target every epoch -- 1.71x the paper's budget at tau 0.3.")
    ap.add_argument("--epochs", type=int, default=0,
                    help="PAPER: 25 epochs at batch 256. When set, --steps is DERIVED as "
                         "epochs * ceil(epoch length/(batch*accum)) -- the epoch length being the "
                         "scene count under --epoch-unit scenes -- so the paper's schedule is stated "
                         "in its own unit; the cosine T_max then spans the whole run.")
    ap.add_argument("--score-w", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--amp", choices=("none", "bf16"), default="none",
                    help="bf16 autocast of the forward (losses stay FP32). MEASURED 3.5x on the A40. "
                         "A DEPARTURE from the paper's FP32 -- recorded, and part of the resume identity")
    ap.add_argument("--tf32", action="store_true",
                    help="TF32 tensor-core matmuls for FP32 math (A40: 1.27x at FP32)")
    ap.add_argument("--fused-adam", action="store_true", help="torch's fused AdamW kernel")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the frozen-trunk backbone in place (keys unchanged). A40: "
                         "bf16 0.366 -> 0.220 s/sample")
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers decoding JPEGs off the critical path (order unchanged)")
    ap.add_argument("--out", default=None,
                    help="RUN DIRECTORY: config.json, metrics.jsonl, ckpt_last.pt, "
                         "snap_epochNNN.pt, model_final.pt, summary.json. REQUIRED for any real "
                         "run (> 500 steps or --epochs); without it nothing is saved.")
    ap.add_argument("--resume", action="store_true",
                    help="continue <out>/ckpt_last.pt if present (start fresh if absent). Refused "
                         "if the run's identity (defining args, bank content, trunk) changed.")
    ap.add_argument("--ckpt-every-min", type=float, default=20.0,
                    help="wall-clock checkpoint cadence, taken at optimiser-step boundaries")
    ap.add_argument("--ckpt-every-steps", type=int, default=0,
                    help="additionally checkpoint every N optimiser steps (0 = off)")
    ap.add_argument("--halt-after-steps", type=int, default=0,
                    help="TEST ONLY: exit right after the checkpoint at this step, to prove that "
                         "halt + --resume is bit-identical to an uninterrupted run")
    sys.exit(run(ap.parse_args()))


if __name__ == "__main__":
    main()
