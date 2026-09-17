#!/usr/bin/env python3
"""T-B — the LEAD MASK, and its area-matched random off-agent twin.

⛔ WHAT THIS IS. An IMAGE-SPACE intervention applied to every frame the encoder sees,
inside the eval dataset, so the rest of `refcv3_arm.py`'s forward is bit-identical to
the banked roll. It is NOT a second copy of the harness: `run_mask_roll.py` imports the
real `refcv3_arm.py` and replaces exactly one factory.

THE FRAME ALGEBRA, read from the code rather than assumed:
  * `comma2k19.stack_frames`: provider row p carries channel groups c = 0,1,2 showing
    RAW frames p, p+1, p+2 -- "oldest frame first, current frame in the LAST 3 channels".
  * `v2_dataset.py:36-38`: the provider stores `poses[n_stack-1:]`, so provider row p IS
    raw frame p + (n_stack - 1) = p + 2.
  ==> a window starting at provider row t spans provider rows t..t+W-1, and sub-frame
  (j, c) of that window shows RAW frame t + j + c. The window ORIGIN (t0 = t+W-1) is
  raw frame t + W + 1, which is the key the lead block and the agent join are joined on.

THE LEAD. The window origin's row of `b1_eval_lead_block.npz`: `has_lead` and
`lead_track_id`, restricted to `gap0_m <= --gap-max` (30 m, the registered support).
The SAME track id is then looked up in the agent join at every raw frame of the stack,
so the mask follows the lead through the stack instead of smearing one pose over it.

THE CUBOID AND THE ~1 m z OFFSET. The agent join carries (cx, cy, yaw, l = size_x,
w = size_y) in the per-frame EGO frame and NO z or height -- `build_b1_agent_join.py:53`
keeps sizes x/y only. ⭐ That is the SAFE side of `RETR-2026-09-13-SAM3MAP-A2-GHOST-GROUND`:
"PhysicalAI's tracked boxes carry z about 1 m under the LiDAR ground; a display mask
built from z - h/2 blanked the road in front of every vehicle and the PI asked why."
The box is therefore built GROUND-STANDING -- bottom face at rig z = 0, top at z = H(class)
-- which is the correction, not the defect. H(class) is ESTIMATED (typical vehicle
heights) and every arm uses the same table, so it cancels in lead-minus-random.

THE PROJECTION is the programme's own: `rig_projection.RigCamera` with the per-clip
front-wide extrinsics of `refcv5v2_final/extrinsics141.json` into
`calib.PHYSICALAI_WIDE120_256x640` (cylindrical, f_ref 305.577). The 8 cuboid corners are
projected and their axis-aligned pixel bbox is filled with the frame mean -- the same
fill the registered `frames_blind` ablation uses, so the two arms differ only in support.

THE RANDOM TWIN is the SAME bbox translated along the column axis by ONE column shift
`delta` DRAWN ONCE PER WINDOW (the pre-registration's wording: "drawn per window"), to a
placement that overlaps NO labelled agent's projected bbox (dilated by
`--agent-margin-px`) at the window's ORIGIN frame. ⭐ ONE DRAW PER WINDOW, NOT PER
SUB-FRAME: an independently re-drawn patch would jump between the 24 sub-frames of the
D-015 stack and add APPARENT MOTION the lead mask does not have -- the contrast would then
confound "the lead" with "temporal coherence". Translation preserves the area EXACTLY;
a shift that would leave the frame is slid minimally back inside (still a translation,
still exact area) and counted as `rand_clamped`. The draw is seeded by
(clip_sha12, raw_origin, mask_seed) so the arm is reproducible and order-independent.

MODES: `lead` | `rand` | `zero`. ⛔ `zero` runs the ENTIRE code path -- join lookup,
projection, bbox -- and fills NOTHING: it is the registered bit-identity control, and it
is only a control because it is the same code.
"""
from __future__ import annotations

import hashlib
import json
import lzma
import os

import numpy as np
import torch

#: ESTIMATED typical vehicle heights (m), used only to give the ground-standing cuboid a
#: top face. Identical in every arm, so it cancels in lead-minus-random.
CLASS_H_M = {
    "automobile": 1.55, "heavy_truck": 3.80, "trailer": 4.00, "bus": 3.20,
    "other_vehicle": 2.20, "rider": 1.80, "person": 1.75, "protruding_object": 1.00,
}
DEFAULT_H_M = 1.80


def sha12(clip_id: str) -> str:
    return hashlib.sha256(clip_id.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
def load_agents(path: str, keep_clips: set[str], max_range_m: float = 80.0) -> dict:
    """(clip_id, raw_frame) -> list of (track_id, cx, cy, yaw, l, w, cls).

    Only the clips of this corpus and only |centre| <= max_range_m are kept -- the join
    carries 905,512 boxes out to >100 m and the mask never reaches past 30 m.
    """
    out: dict[tuple[str, int], list] = {}
    n_lines = n_box = 0
    with lzma.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            cid = d["clip_id"]
            if cid not in keep_clips:
                continue
            n_lines += 1
            rows = []
            for a in d["agents"]:
                cx, cy = float(a["cx"]), float(a["cy"])
                if abs(cx) > max_range_m or abs(cy) > max_range_m:
                    continue
                rows.append((str(a["track_id"]), cx, cy, float(a["yaw"]),
                             float(a["l"]), float(a["w"]), str(a.get("cls", ""))))
            n_box += len(rows)
            out[(cid, int(d["frame"]))] = rows
    out["_stats"] = {"n_lines": n_lines, "n_boxes_kept": n_box,
                     "n_clips": len(keep_clips)}
    return out


def load_leads(path: str, keep_clips: set[str]) -> dict:
    """(clip_id, raw_frame) -> (has_lead, lead_track_id, gap0_m, state)."""
    z = np.load(path, allow_pickle=True)
    cid = z["clip_id"].astype(str)
    fr = z["frame"].astype(np.int64)
    hl = z["has_lead"].astype(bool)
    tid = z["lead_track_id"].astype(str)
    gap = z["gap0_m"].astype(np.float64)
    st = z["state"].astype(str)
    out = {}
    for i in range(len(cid)):
        if cid[i] not in keep_clips:
            continue
        out[(cid[i], int(fr[i]))] = (bool(hl[i]), tid[i], float(gap[i]), st[i])
    return out


# --------------------------------------------------------------------------- #
class Projector:
    """Per-clip RigCamera into the deployed 256x640 cylindrical frame."""

    def __init__(self, extrinsics_json: str):
        from tanitad.data.calib import PHYSICALAI_WIDE120_256x640 as FR
        from tanitad.data.physicalai import FrontWideExtrinsics
        from tanitad.data.rig_projection import RigCamera
        self.FR = FR
        with open(extrinsics_json, encoding="utf-8") as fh:
            raw = json.load(fh)
        self.cams = {}
        for cid, e in raw.items():
            ex = FrontWideExtrinsics(qx=e["qx"], qy=e["qy"], qz=e["qz"], qw=e["qw"],
                                     x=e["x"], y=e["y"], z=e["z"])
            R = torch.as_tensor(ex.rotation_cam_to_vehicle(), dtype=torch.float64)
            self.cams[cid] = RigCamera(
                R_cam_to_rig=R,
                t_cam_in_rig=torch.tensor([ex.x, ex.y, ex.z], dtype=torch.float64),
                frame=FR)

    def bbox(self, cid: str, cx: float, cy: float, yaw: float, l: float, w: float,
             h: float):
        """Ground-standing cuboid -> (c0, c1, r0, r1) integer pixel bbox, or None."""
        cam = self.cams.get(cid)
        if cam is None:
            return None
        ca, sa = np.cos(yaw), np.sin(yaw)
        hl, hw = l / 2.0, w / 2.0
        pts = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                px = cx + sx * hl * ca - sy * hw * sa
                py = cy + sx * hl * sa + sy * hw * ca
                pts.append((px, py, 0.0))
                pts.append((px, py, h))
        col, row, val = cam.project(torch.tensor(pts, dtype=torch.float64))
        v = val.numpy().astype(bool)
        if int(v.sum()) < 2:
            return None
        c = col.numpy()[v]
        r = row.numpy()[v]
        # cylindrical wrap guard: a box wider than half the frame is not a 30 m lead
        if float(c.max() - c.min()) > self.FR.width / 2.0:
            return None
        c0 = int(np.floor(c.min())); c1 = int(np.ceil(c.max()))
        r0 = int(np.floor(r.min())); r1 = int(np.ceil(r.max()))
        c0 = max(c0, 0); r0 = max(r0, 0)
        c1 = min(c1, self.FR.width - 1); r1 = min(r1, self.FR.height - 1)
        if c1 <= c0 or r1 <= r0:
            return None
        return (c0, c1, r0, r1)


# --------------------------------------------------------------------------- #
class LeadMasker:
    """The stateful masker the patched dataset calls once per window."""

    def __init__(self, mode: str, agents: dict, leads: dict, proj: Projector,
                 n_stack: int = 3, gap_max_m: float = 30.0,
                 agent_margin_px: int = 8, seed: int = 0):
        assert mode in ("lead", "rand", "zero")
        self.mode = mode
        self.agents = agents
        self.leads = leads
        self.proj = proj
        self.k = n_stack - 1
        self.gap_max_m = float(gap_max_m)
        self.margin = int(agent_margin_px)
        self.seed = int(seed)
        #: (clip_id, raw_frame) -> blocked column intervals / the lead's bbox. A
        #: window's 24 sub-frames and the stride-5 overlap between windows re-ask
        #: for the same raw frames dozens of times; without the memo the random
        #: arm re-projects every labelled agent each time.
        self._memo_blocked: dict = {}
        self._memo_bbox: dict = {}
        self.stats = {"windows": 0, "windows_masked": 0, "windows_no_lead": 0,
                      "windows_lead_far": 0, "windows_no_join": 0,
                      "subframes": 0, "subframes_masked": 0,
                      "rand_fallback": 0, "rand_clamped": 0,
                      "rand_hits_agent_offorigin": 0,
                      "px_masked": 0, "px_total": 0,
                      "area_lead_px": [], "area_rand_px": []}

    # -- the single entry point ------------------------------------------- #
    def mask_window(self, clip_id: str, t_provider: int, frames: torch.Tensor):
        """frames [W, 3*n_stack, H, W] uint8 -> the same tensor, masked in place-safe copy."""
        self.stats["windows"] += 1
        W = int(frames.shape[0])
        raw_origin = t_provider + W - 1 + self.k
        lead = self.leads.get((clip_id, raw_origin))
        if lead is None or not lead[0] or not lead[1]:
            self.stats["windows_no_lead"] += 1
            return frames
        _, tid, gap, _state = lead
        if not (gap == gap and gap <= self.gap_max_m):
            self.stats["windows_lead_far"] += 1
            return frames
        out = frames.clone()
        fill = int(round(float(frames.float().mean())))
        # ⛔ NOT `hash()`: Python randomises string hashing per process, which would
        # make the random twin un-reproducible across runs of the same arm.
        _k = f"{sha12(clip_id)}|{int(raw_origin)}|{self.seed}".encode()
        rng = np.random.default_rng(
            int.from_bytes(hashlib.sha256(_k).digest()[:8], "little"))
        # ⭐ ONE column shift for the whole window, drawn at the ORIGIN sub-frame.
        delta = None
        if self.mode == "rand":
            bb0 = self._lead_bbox(clip_id, raw_origin, tid)
            if bb0 is None:
                self.stats["rand_fallback"] += 1
                return frames
            delta = self._draw_delta(clip_id, raw_origin, bb0, rng)
            if delta is None:
                self.stats["rand_fallback"] += 1
                return frames
        n_sub_masked = 0
        for j in range(W):
            for c in range(self.k + 1):
                raw = t_provider + j + c
                self.stats["subframes"] += 1
                rows = self.agents.get((clip_id, raw))
                if rows is None:
                    self.stats["windows_no_join"] += 1
                    continue
                tgt = [r for r in rows if r[0] == tid]
                if not tgt:
                    continue
                _, cx, cy, yaw, l, w, cls = tgt[0]
                if cx <= 0.0 or (cx * cx + cy * cy) ** 0.5 > self.gap_max_m + 15.0:
                    continue
                # ⛔ THE TRACK ID IS PART OF THE KEY. Two windows can share a raw
                # frame while their window ORIGINS name DIFFERENT leads; a
                # (clip, frame) memo would then mask the wrong vehicle.
                mk = (clip_id, raw, tid)
                if mk in self._memo_bbox:
                    bb = self._memo_bbox[mk]
                else:
                    bb = self.proj.bbox(clip_id, cx, cy, yaw, l, w,
                                        CLASS_H_M.get(cls, DEFAULT_H_M))
                    self._memo_bbox[mk] = bb
                if bb is None:
                    continue
                c0, c1, r0, r1 = bb
                area = (c1 - c0 + 1) * (r1 - r0 + 1)
                self.stats["area_lead_px"].append(int(area))
                if self.mode == "rand":
                    Wf = self.proj.FR.width
                    nc0, nc1 = c0 + delta, c1 + delta
                    if nc0 < 0:
                        nc1 -= nc0; nc0 = 0; self.stats["rand_clamped"] += 1
                    elif nc1 > Wf - 1:
                        nc0 -= nc1 - (Wf - 1); nc1 = Wf - 1
                        self.stats["rand_clamped"] += 1
                    if nc0 < 0:
                        continue
                    for (ba, bb_) in self._blocked(clip_id, raw, rows):
                        if not (nc1 < ba or nc0 > bb_):
                            self.stats["rand_hits_agent_offorigin"] += 1
                            break
                    c0, c1 = int(nc0), int(nc1)
                    self.stats["area_rand_px"].append(
                        int((c1 - c0 + 1) * (r1 - r0 + 1)))
                if self.mode != "zero":
                    out[j, 3 * c:3 * (c + 1), r0:r1 + 1, c0:c1 + 1] = fill
                    self.stats["px_masked"] += area
                self.stats["px_total"] += int(out.shape[-1] * out.shape[-2])
                self.stats["subframes_masked"] += 1
                n_sub_masked += 1
        if n_sub_masked:
            self.stats["windows_masked"] += 1
        return out

    # -- the area-matched off-agent placement ------------------------------ #
    def _lead_bbox(self, clip_id, raw, tid):
        rows = self.agents.get((clip_id, raw))
        if rows is None:
            return None
        tgt = [r for r in rows if r[0] == tid]
        if not tgt:
            return None
        _, cx, cy, yaw, l, w, cls = tgt[0]
        if cx <= 0.0:
            return None
        mk = (clip_id, raw, tid)
        if mk not in self._memo_bbox:
            self._memo_bbox[mk] = self.proj.bbox(
                clip_id, cx, cy, yaw, l, w, CLASS_H_M.get(cls, DEFAULT_H_M))
        return self._memo_bbox[mk]

    def _blocked(self, clip_id, raw, rows):
        """Column intervals of EVERY labelled agent ahead, dilated by the margin."""
        mk = (clip_id, raw)
        b = self._memo_blocked.get(mk)
        if b is None:
            b = []
            for (_tid, cx, cy, yaw, l, w, cls) in rows:
                if cx <= 0.0:
                    continue
                bb = self.proj.bbox(clip_id, cx, cy, yaw, l, w,
                                    CLASS_H_M.get(cls, DEFAULT_H_M))
                if bb is not None:
                    b.append((bb[0] - self.margin, bb[1] + self.margin))
            self._memo_blocked[mk] = b
        return b

    def _draw_delta(self, clip_id, raw_origin, bb0, rng):
        """ONE column shift for the whole window: an off-agent placement at the
        ORIGIN frame, expressed as a translation so the area is preserved exactly."""
        c0, c1, _r0, _r1 = bb0
        wpx = c1 - c0
        Wf = self.proj.FR.width
        blocked = self._blocked(clip_id, raw_origin,
                                self.agents.get((clip_id, raw_origin)) or [])
        for _ in range(128):
            nc0 = int(rng.integers(0, max(Wf - wpx - 1, 1)))
            nc1 = nc0 + wpx
            if nc1 > Wf - 1:
                continue
            if any(not (nc1 < a or nc0 > b) for (a, b) in blocked):
                continue
            return int(nc0 - c0)
        return None

    def summary(self) -> dict:
        s = dict(self.stats)
        for k in ("area_lead_px", "area_rand_px"):
            v = np.asarray(s.pop(k), dtype=np.float64)
            s[k + "_n"] = int(v.size)
            s[k + "_mean"] = float(v.mean()) if v.size else None
            s[k + "_median"] = float(np.median(v)) if v.size else None
            s[k + "_p95"] = float(np.percentile(v, 95)) if v.size else None
        s["mode"] = self.mode
        s["gap_max_m"] = self.gap_max_m
        s["class_heights_m"] = CLASS_H_M
        s["masked_pixel_frac_of_masked_subframes"] = (
            s["px_masked"] / s["px_total"] if s["px_total"] else None)
        return s
