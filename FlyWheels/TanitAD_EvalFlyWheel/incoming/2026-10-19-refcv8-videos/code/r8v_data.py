"""r8v_data.py -- read-only data layer for the refcv8 eval videos (CPU only, no GPU, no repo writes).

Everything the renderer needs per eval139 episode, assembled from BANKED artifacts and keyed by the SHORT episode index (`clip_index`) --
clip UUIDs exist only in memory (programme clip-id rule).

SOURCES (all read-only; every path is a module constant so a later checkpoint is a one-argument change)
  dumps    <ARCH>/<tree>/{panel_s0,s6_s0,dump_s0}/ep###.npz, dump_s0/decisions/ep###.npz, dump_s0/refcv6_extras.npz     (battery roll, inference seed 0)
  prior    DEPLOY8_POS_<tree30k>_s0.npz  S2_prior8 / S2_pos8                        (30k EMA ONLY: the shipped re-timed plan; the prior is model-free)
           ⭐ a PER-FRAME tree (*_pf, 2026-10-20) never defaults to that clip-fed file: its RT layer is the REGISTERED per-frame table's
           DEPLOY8_POS_<tree>_s0.npz (pf_lane d8, provenance-checked by resolve_deploy_pf) or it is OFF with an on-screen note.
  nav      dump_s0/manifest.json nav_feed.mode + decisions sidecar (nav_cmd = the FED token; nav_cmd_clip / r8_nav_fed / r8_nav_known on a pf tree),
           read by r8v_feed.feed_from_decisions -- NEVER re-derived by a separate lookup (R57).
  frames   <KIT6>/data/refcv6-b1-416x1024-eval139/<clip>.v2ep.pt  (PNG, 416x1024, CYLINDRICAL f_ref 488.924, n_stack 3, 10 Hz)
  extrinsic <KIT8>/data/refcv8_train4572_eval139_extrinsics.json (per-clip cam->rig quaternion + mount x,y,z)
  map GT   <KIT6>/data/sam3_gt_v3_eval/<sha12>.sam3mapgt.npz  cart_frac [T,9,200,120] (rig frame, 0.5 m cells, x 0..100 m, y -30..+30 m)   137/139 clips
  agents   eval139_agents3d.pkl from extract_eval_agents_r8.py (B1 3-D agent join, ego frame per raw frame)

INDEX CONVENTION (MEASURED 2026-10-19, ep000): decisions/ep_poses [199,4] == v2ep poses[2:]; `ws` is the STACKED ROW of the window origin t0
(ep_poses[ws] == pose_last); the camera frame shown at row j is RAW frame j + n_stack - 1 = j + 2; the map GT and the agent join are keyed on the RAW frame.
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import os
import re
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r8v_feed as FEED  # noqa: E402  (numpy only at import)

ARCH = Path("C:/Users/Admin/refcv8_eval_archive/battery")
KIT6 = Path("E:/refcv6_eval_kit")
KIT8 = Path("E:/refcv8_eval_kit")
BATT_PKG = Path("C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery")
DEPLOY8_30K = BATT_PKG / "raw/retime_deploy/DEPLOY8_POS_step30000ema_unc_s0.npz"
# ---- the PER-FRAME re-timed (RT) layer (SHIP-R8 AMENDMENT 3 (ii)); every path overridable on the CLI
PF_HOLD = Path("C:/Users/Admin/qland/hold_r8pf")
PF_REGISTERED = PF_HOLD / "REGISTERED_PF_TABLE.sha256"            # 64 hex == sha256(FROZEN_GAINS_RETIME_train_pf.json), written when the table is frozen
PF_PKG = PF_HOLD / "stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final/pfpkg"   # pf_lane --pkg (chain_final_pf_r8.sh PFPKG)
PF_RS_TABLE_NAME = "refcv8_retime_deploy_gains_pf.json"          # == pf_lane.RS_TABLE_NAME
CLIP_FED_FROZEN_SHA256 = "b208c9c1e4236075de107ef49eaef39c1dddddba257f6e594bf6b84f8a24b1ee"   # LITERAL: the clip-fed frozen train table (FORWARD_DEPLOY8_*_unc.json)
CLIP_FED_RS_TABLE_SHA256 = "f18359be021dd400f4865cdcb3ca053bb8c0910b54c0611ba09d44da9a97219e"  # LITERAL: its runtime stage table (DEPLOY8_POS_*_unc meta)
AUTO = "auto"
EVAL_CACHE = KIT6 / "data/refcv6-b1-416x1024-eval139"
MAP_GT_DIR = KIT6 / "data/sam3_gt_v3_eval"
EXTR_JSON = KIT8 / "data/refcv8_train4572_eval139_extrinsics.json"
AGENTS_PKL = KIT8 / "videos/_cache/eval139_agents3d.pkl"

N_STACK = 3
K = N_STACK - 1
DT = 0.1                                   # native rate 10 Hz (poses: dx per row == v * 0.1 s, MEASURED ep000)
SLOT_S8 = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)   # model slot instants (deploy8 meta: slot_instants_s)
S6_SLOTS = (1, 3, 4, 5, 6, 7)              # indices into the 8 slots -> 1..6 s
FRAME_F_REF = 488.92398517830253           # cylindrical, 416x1024 (== 305.577 * 1.6)
FRAME_H, FRAME_W = 416, 1024

LAT_NAMES = ("LANE_KEEP", "LANE_CHANGE_L", "LANE_CHANGE_R", "ABORT_LC", "NUDGE_L", "NUDGE_R", "TURN_L", "TURN_R")
LON_NAMES = ("FOLLOW", "CRUISE", "YIELD_MERGE", "BRAKE_TO", "CREEP", "HOLD", "ADAPT_SPEED_FOR_CURVE", "ACCELERATE")
NAV_NAMES = ("FOLLOW_ROAD", "TURN_L", "TURN_R")   # NAV_COMMAND_TOKENS order (vocab_v7); values {0,1,2} MEASURED in decisions.nav_cmd


def sha12_of(clip_id: str) -> str:
    return hashlib.sha256(clip_id.encode()).hexdigest()[:12]


def softmax(x):
    x = np.asarray(x, np.float64)
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_registered(path) -> str:
    """64 lower-case hex or '' (an empty / short string is INCONCLUSIVE, never a match) -- the pf_lane.read_registered contract."""
    try:
        s = "".join(c for c in Path(path).read_text(encoding="utf-8-sig").split()[0] if c in "0123456789abcdefABCDEF").lower()
    except Exception:      # noqa: BLE001
        return ""
    return s if len(s) == 64 else ""


def resolve_deploy_pf(tree: str, deploy_npz=AUTO, registered_file=PF_REGISTERED, pf_pkg=PF_PKG) -> dict:
    """The RT layer of a PER-FRAME tree: ON only when every link of the provenance chain is the REGISTERED per-frame table, else OFF with the reason.

    chain (each a positive assertion; any missing / unreadable link reads OFF, never ON):
      registered   REGISTERED_PF_TABLE.sha256 holds 64 hex (the per-frame re-fitted FROZEN_GAINS_RETIME_train_pf.json, frozen 2026-10-20)
      npz          DEPLOY8_POS_<tree>_s0.npz (default: <pf_pkg>/raw/retime_deploy/, written by pf_lane.py d8) exists, meta_json.tree == tree
      stage table  meta_json.table_sha256 == sha256(<npz dir>/refcv8_retime_deploy_gains_pf.json) and that file's source_sha256 == registered
      forward      FORWARD_DEPLOY8_<tree>.json (same dir) frozen_gains_sha256 == registered
      not clip-fed none of the above equals the clip-fed literals b208c9c1... / f18359be...
    """
    off = {"on": False, "path": None, "label": "OFF", "provenance": {}}
    reg = read_registered(registered_file)
    if not reg:
        return {**off, "note": f"RT layer OFF: no registered per-frame table ({Path(registered_file).name} absent / not 64 hex)"}
    if reg in (CLIP_FED_FROZEN_SHA256, CLIP_FED_RS_TABLE_SHA256):
        return {**off, "note": "RT layer OFF: the 'registered' sha256 is the CLIP-FED table (b208c9c1...)"}
    p = Path(pf_pkg) / "raw/retime_deploy" / f"DEPLOY8_POS_{tree}_s0.npz" if deploy_npz in (None, AUTO) else Path(deploy_npz)
    if not p.exists():
        return {**off, "note": f"RT layer OFF: per-frame deploy npz not built yet ({p.name})"}
    try:
        z = np.load(p, allow_pickle=True)
        meta = json.loads(str(z["meta_json"]))
        rs = p.parent / PF_RS_TABLE_NAME
        rs_sha = _sha256_file(rs) if rs.exists() else ""
        rs_src = (json.load(open(rs, encoding="utf-8")).get("source_sha256") or "") if rs.exists() else ""
        fw = p.parent / f"FORWARD_DEPLOY8_{tree}.json"
        fw_fz = (json.load(open(fw, encoding="utf-8")).get("frozen_gains_sha256") or "") if fw.exists() else ""
    except Exception as exc:      # noqa: BLE001
        return {**off, "note": f"RT layer OFF: provenance unreadable ({type(exc).__name__})"}
    prov = {"npz": str(p), "meta_tree": meta.get("tree"), "meta_table_sha256": meta.get("table_sha256"), "rs_table_sha256": rs_sha,
            "rs_source_sha256": rs_src, "forward_frozen_gains_sha256": fw_fz, "registered": reg}
    bad = []
    if meta.get("tree") != tree:
        bad.append(f"npz tree {meta.get('tree')!r} != {tree!r}")
    if len(rs_sha) != 64 or meta.get("table_sha256") != rs_sha:
        bad.append("stage table sha mismatch / absent")
    if rs_src != reg:
        bad.append("stage table source != registered")
    if fw_fz != reg:
        bad.append("FORWARD_DEPLOY8 frozen sha != registered")
    if {meta.get("table_sha256"), rs_src, fw_fz} & {CLIP_FED_FROZEN_SHA256, CLIP_FED_RS_TABLE_SHA256}:
        bad.append("clip-fed table b208c9c1/f18359be in the chain")
    if bad:
        return {**off, "provenance": prov, "note": "RT layer OFF: " + "; ".join(bad)}
    return {"on": True, "path": str(p), "label": f"pf table {reg[:8]}", "provenance": prov, "note": ""}


class EvalBank:
    """One checkpoint's banked roll (`tree`) + the model-free / baseline layers.

    deploy_npz: AUTO (default) = per-clip tree -> the 30k EMA file (legacy, labelled); per-frame tree -> resolve_deploy_pf (registered pf table or OFF).
    An explicit path on a per-frame tree is accepted only if it passes resolve_deploy_pf's provenance chain."""

    def __init__(self, tree="step40000ema_unc", deploy_npz=AUTO, load_agents=True, arch_root=ARCH, pf_registered=PF_REGISTERED, pf_pkg=PF_PKG):
        self.tree = tree
        self.root = Path(arch_root) / tree
        self.feed = FEED.tree_feed(self.root, tree)          # SystemExit if the tree's name and its manifest's nav_feed disagree
        self.feed_mode = self.feed["mode"]
        self.rt = {"on": None, "note": "", "label": ""}
        if self.feed_mode == FEED.NAV_FEED_PF:
            self.rt = resolve_deploy_pf(tree, deploy_npz, pf_registered, pf_pkg)
            deploy_npz = self.rt["path"]
            print(f"[r8v_data] {tree}: nav feed {self.feed_mode}; {self.rt['note'] or 'RT layer ON (' + self.rt['label'] + ')'}", flush=True)
        elif deploy_npz in (None, AUTO):
            deploy_npz = DEPLOY8_30K
        self.rt_note = self.rt.get("note", "")
        summ = json.load(open(self.root / "battery_summary.json", encoding="utf-8"))
        self.summary = summ
        self.ckpt_step = summ.get("step")
        self.ckpt_md5 = summ.get("ckpt_md5")
        self.sha12_by_index = {int(k): v for k, v in summ["stages"]["roll_s0"]["extras"]["clip_sha12_by_index"].items()}
        self._v2ep = {}
        for f in os.listdir(EVAL_CACHE):
            if f.endswith(".v2ep.pt"):
                cid = f.split(".")[0]
                self._v2ep[sha12_of(cid)] = str(EVAL_CACHE / f)
        self._extr = json.load(open(EXTR_JSON, encoding="utf-8"))
        self._extr_by_sha = {sha12_of(k): v for k, v in self._extr.items()}
        self.ex = np.load(self.root / "dump_s0/refcv6_extras.npz")
        self._ex_key = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(self.ex["clip_index"], self.ex["ws"]))}
        self.dep = np.load(deploy_npz, allow_pickle=True) if deploy_npz and Path(deploy_npz).exists() else None
        m = re.search(r"step(\d+)(ema)?", Path(deploy_npz).name) if deploy_npz else None
        self.deploy_label = (f"{int(m.group(1)) / 1000:g}k {'EMA' if m.group(2) else 'raw'}" if m else "n/a")
        if self.feed_mode == FEED.NAV_FEED_PF:
            self.deploy_label = (f"{self.deploy_label}, {self.rt['label']}" if self.rt["on"] else "RT OFF")
        if self.dep is None and not self.rt_note:
            self.rt_note = f"RT layer OFF: deploy npz {deploy_npz} absent"
        mt = re.search(r"step(\d+)(ema)?", tree)
        self.ema = bool(mt and mt.group(2)) or "ckpt_ema_" in str(summ.get("ckpt", "")).replace("\\", "/")   # a tree name without "stepNNNNNema" (smoke)
        self.step_k = (int(mt.group(1)) / 1000) if mt else (self.ckpt_step or 0) / 1000
        self.name_label = f"refcv8 {'EMA' if self.ema else 'raw'}-{self.step_k:g}k"
        self._dep_rows = {}
        if self.dep is not None:
            cl, ws = self.dep["S2_clip"], self.dep["S2_ws"]
            for i, (c, w) in enumerate(zip(cl, ws)):
                self._dep_rows[(int(c), int(w))] = i
        self.agents = None
        if load_agents and AGENTS_PKL.exists():
            import pickle
            self.agents = pickle.load(open(AGENTS_PKL, "rb"))["agents"]

    # ------------------------------------------------------------------ identity helpers
    def indices(self):
        return sorted(self.sha12_by_index)

    def sha12(self, ci):
        return self.sha12_by_index[int(ci)]

    def v2ep_path(self, ci):
        return self._v2ep[self.sha12(ci)]

    def extrinsics(self, ci):
        return self._extr_by_sha[self.sha12(ci)]

    def map_gt_path(self, ci):
        p = MAP_GT_DIR / f"{self.sha12(ci)}.sam3mapgt.npz"
        return p if p.exists() else None

    # ------------------------------------------------------------------ per-episode banked layers
    def episode(self, ci):
        """Every banked per-window array of episode `ci` (window axis n == 35 on a 199-row clip, stride 5, ws 7..177)."""
        ci = int(ci)
        f = f"ep{ci:03d}.npz"
        pan = np.load(self.root / "panel_s0" / f)
        dec = np.load(self.root / "dump_s0/decisions" / f)
        s6 = np.load(self.root / "s6_s0" / f)
        s6d = np.load(self.root / "s6_s0/decisions" / f)
        assert int(pan["clip_index"][0]) == ci, "panel file index != clip_index"
        ws = pan["ws"].astype(int)
        assert np.array_equal(ws, dec["ws"]), "panel ws != decisions ws"
        rows = [self._ex_key[(ci, int(w))] for w in ws]
        ex = self.ex
        fd = FEED.feed_from_decisions(dec, self.feed_mode)     # the FED nav as the roll WROTE it (sidecar), never a lookup
        assert np.array_equal(fd["ws"], ws), "feed ws != panel ws"
        e = {
            "nav_feed": self.feed_mode, "nav_fed": fd["nav_cmd_fed"], "nav_clip": fd["nav_cmd_clip"],
            "r8_nav_fed": fd["r8_nav"], "r8_nav_known": fd["r8_nav_known"],
            "ci": ci, "ws": ws, "poses": dec["ep_poses"].astype(np.float64),         # [199, 4] x, y, yaw, v  (world / clip-local)
            "v0": pan["v0"].astype(np.float64),
            "g4": pan["g"].astype(np.float64),                                         # GT ego-frame @ 0.5/1/1.5/2 s
            "os4": pan["os"].astype(np.float64), "r7_4": pan["b_refcv7_50400_s0"].astype(np.float64),
            "ha0_ext4": pan["ha0_ext"].astype(np.float64),
            "plan8": dec["plan_full_nav_true"].astype(np.float64),                    # refcv8 8-slot plan, ego frame @ t0
            "gt_future_world": dec["gt_future_ext"].astype(np.float64),               # [n, 60, 4] world x, y, yaw, v @ t0+0.1..6.0 s
            "gt_future_valid": dec["gt_future_valid_ext"].astype(bool),
            "pose_last": dec["pose_last"].astype(np.float64),
            "nav_cmd": fd["nav_cmd_fed"].astype(int), "nav_valid": dec["nav_valid"].astype(bool),     # nav_cmd == the FED token (pf: per-frame)
            "sel_idx": dec["sel_idx"].astype(int), "sel_agrees_oracle": dec["sel_agrees_oracle"].astype(float),
            "lat_label": ex["lat_v7"][rows].astype(int), "lon_label": ex["lon_v7"][rows].astype(int),
            "lat_logits": ex["nav_true.tacv6_lat_logits"][rows].astype(np.float64),
            "lon_logits": ex["nav_true.tacv6_lon_logits"][rows].astype(np.float64),
            "lat_logits_z": ex["nav_true.lat_logits_tac"][rows].astype(np.float64),
            "lon_logits_z": ex["nav_true.lon_logits_tac"][rows].astype(np.float64),
            "v_max_ms": ex["v_max_ms"][rows].astype(np.float64),
            "gt_max_speed_6s": ex["gt_max_speed_6s"][rows].astype(np.float64),
            "traj_extras8": ex["nav_true.traj"][rows].astype(np.float64),
        }
        assert np.array_equal(e["traj_extras8"], e["plan8"].astype(np.float32).astype(np.float64)) or \
            np.abs(e["traj_extras8"] - e["plan8"]).max() < 1e-4, "extras traj != decisions plan_full (window order drift)"
        # S6 windows (ws <= 137): 6 slots @ 1..6 s
        e["s6_ws"] = s6["ws"].astype(int)
        e["s6_g"] = s6["g"].astype(np.float64)
        e["s6_os"] = s6["os"].astype(np.float64)
        e["s6_r7"] = s6["b_refcv7_50400_s0"].astype(np.float64)
        e["s6_v0"] = s6["v0"].astype(np.float64)
        e["s6_gt_v6"] = s6d["gt_future_ext"][:, 59, 3].astype(np.float64)
        e["s6_gt_valid"] = s6d["gt_future_valid_ext"].min(axis=1) >= 1.0
        # model-free prior + shipped re-timed plan (30k EMA deploy npz), same (clip, ws) key
        if self.dep is not None:
            r = [self._dep_rows[(ci, int(w))] for w in ws]
            e["prior8"] = self.dep["S2_prior8"][r].astype(np.float64)
            e["ship8"] = self.dep["S2_pos8"][r].astype(np.float64)
            e["ship_plan8_30k"] = self.dep["S2_plan8"][r].astype(np.float64)
        return e

    # ------------------------------------------------------------------ frames / map / agents
    def frames(self, ci):
        """Decode every RAW frame of the clip (PNG) -> uint8 [R, 416, 1024, 3]; also returns the raw poses [R, 4]."""
        import torch
        import torchvision.io as tvio
        d = torch.load(self.v2ep_path(ci), map_location="cpu", weights_only=False)
        assert str(d.get("codec", "jpeg")) == "png" and int(d["n_stack"]) == N_STACK
        fr = d["frame"]
        assert fr["projection"] == "cylindrical" and abs(fr["f_ref"] - FRAME_F_REF) < 1e-6 and (fr["height"], fr["width"]) == (FRAME_H, FRAME_W)
        offs = np.concatenate([[0], np.cumsum(d["jpeg_len"].numpy().astype(np.int64))])
        buf = d["jpeg_buf"]
        out = np.empty((len(d["jpeg_len"]), FRAME_H, FRAME_W, 3), np.uint8)
        for i in range(len(d["jpeg_len"])):
            im = tvio.decode_png(buf[int(offs[i]):int(offs[i + 1])], mode=tvio.ImageReadMode.RGB)
            out[i] = im.permute(1, 2, 0).numpy()
        return out, d["poses"].numpy().astype(np.float64)

    def map_gt(self, ci):
        """(cart_frac [T,9,200,120] uint8, T_world_rig [T,4,4]) or None.  RAW-frame indexed.  x 0..100 m (rows), y -30..+30 m (cols; col 0 = RIGHT)."""
        p = self.map_gt_path(ci)
        if p is None:
            return None
        z = np.load(p, allow_pickle=True)
        m = json.loads(str(z["meta_json"]))
        assert m["schema"] == "tanitad.sam3_map_gt/3" and m["source"]["clip_sha12"] == self.sha12(ci)
        assert tuple(m["cartesian"]["shape"]) == (200, 120) and m["channels"][8] == "not seen"
        return z["cart_frac"], z["T_world_rig"]

    def agents_at(self, ci, raw_frame):
        """(array [n,8] cx,cy,yaw,l,w,occ,cz,h ; cls list) at a RAW frame, or None when that (clip, frame) is NO_LABEL."""
        if self.agents is None:
            return None
        return self.agents.get(int(ci), {}).get(int(raw_frame))


# ---------------------------------------------------------------------------------------------- geometry helpers (numpy, SE(2))
def to_ego(pts_world_xy, pose):
    """world xy [N,2] -> ego frame of `pose` (x, y, yaw, v)."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    d = np.asarray(pts_world_xy, np.float64) - np.asarray(pose[:2], np.float64)
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)


def to_world(pts_ego_xy, pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    p = np.asarray(pts_ego_xy, np.float64)
    return np.stack([pose[0] + c * p[:, 0] - s * p[:, 1], pose[1] + s * p[:, 0] + c * p[:, 1]], axis=1)


def reframe(pts_ego_xy, pose_from, pose_to):
    """Re-express points given in the ego frame of `pose_from` in the ego frame of `pose_to` (a HELD plan seen from a later frame)."""
    return to_ego(to_world(pts_ego_xy, pose_from), pose_to)


def plan_speeds(P8, slot_s=SLOT_S8):
    """Plan-implied mean speed per slot interval [8]: |P_k - P_{k-1}| / dt_k with P_{-1} = origin."""
    P = np.asarray(P8, np.float64)
    prev = np.vstack([[0.0, 0.0], P[:-1]])
    dts = np.diff(np.concatenate([[0.0], np.asarray(slot_s, np.float64)]))
    return np.linalg.norm(P - prev, axis=1) / dts


# ---------------------------------------------------------------------------------------------- the battery's cruise false-brake definition (IMPORTED semantics, see cruise_brake_retime_r8.py / s6_lon_decomp.py)
def gt_dv_class(v0, v6):
    """s6_lon_decomp.partitions 'dvclass', first matching rule wins (the GT-cruise mask is class == 'cruising')."""
    dv = v6 - v0
    if v0 < 1.0 and v6 >= 2.0:
        return "start_from_stop"
    if v0 >= 2.0 and v6 < 1.0:
        return "stopping"
    if v0 < 2.0 and v6 < 2.0:
        return "slow_both"
    if dv >= 1.5:
        return "accelerating"
    if dv <= -1.5:
        return "decelerating"
    return "cruising"


def plan_lon_class_one(P6, v0, dt=1.0, thr=1.5):
    """s6_lon_decomp.plan_lon_class for ONE window: speed over the plan's LAST second (|P6 - P5| / dt) minus the measured v0; <= -1.5 -> 'decel'."""
    P = np.asarray(P6, np.float64)
    v_end = float(np.linalg.norm(P[-1] - P[-2]) / dt)
    dv = v_end - v0
    return ("decel" if dv <= -thr else "accel" if dv >= thr else "cruise"), v_end


def false_brake(plan8, v0, v6_gt, gt_valid):
    """(flag | None, gt_dv_class | None, plan_class, plan_end_speed).  flag = GT class 'cruising' AND plan class 'decel'
    (== the battery's `false_decel_on_dvcruise`); None when the GT 6 s speed does not exist (window runs past the clip end)."""
    pc, vend = plan_lon_class_one(np.asarray(plan8)[list(S6_SLOTS)], v0)
    if not gt_valid:
        return None, None, pc, vend
    gc = gt_dv_class(v0, v6_gt)
    return (gc == "cruising" and pc == "decel"), gc, pc, vend
