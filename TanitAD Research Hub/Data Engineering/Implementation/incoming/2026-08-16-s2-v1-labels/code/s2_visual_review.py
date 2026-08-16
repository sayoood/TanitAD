"""S2 v1 VISUAL review sheet — self-contained HTML (frames + BEV, data URIs).

The text-only REVIEW_SHEET.md is not reviewable: the PI must judge, per clip,
whether the emitted strategic label matches what the video shows. This builds
`review/VISUAL_REVIEW.html`: per clip (a) 6 camera frames sampled across the
clip (t0 highlighted), (b) a BEV plot from the bridged ego npz — trajectory
colored by speed, heading-up ego frame at t0, the emitted goal drawn (distance
arcs for STOP_AT/TURN args, lateral-offset arrow for LANE_TARGET), axes in
metres, (c) the emitted g_str + a_str with args/provenance/VLM-disagreement
flags, (d) a verdict row (correct / wrong / unsure + note, localStorage-backed,
one-click JSON export).

Coverage (deterministic, `s2rev_pull.py` selection, stratified by decision
value): ALL 19 LANE_TARGET (the 0/19-corroboration gate) · 6 TURN spot-checks
incl. the u-turn and both directions · 5 STOP_AT · the ROUTE_TO disposition
(remaps + the abstain) · the 4 excluded val records · 4 FOLLOW controls.

Frame sources (stated per clip, absence-with-reason, never silent):
  aug120  Sayood/tanitad-ph0-aug120 bridged_w120train_2400/videos/<cid>.mp4 —
          pose-aligned BY CONSTRUCTION (v2_to_pilot: frame k == pose k, 10 Hz,
          channel slice [-3:]); frames via ffmpeg select-by-frame-number.
  val     Sayood/tanitad-physicalai-w120-256x640cyl .v2ep.pt — raw JPEG bytes
          sliced straight out of `jpeg_buf` (pose i == raw frame i+n_stack-1,
          the exact v2_dataset alignment; png payloads transcoded via PIL).

CPU-only; token read in place from Keys.txt (never printed); output stays
< 40 MB by construction (640x256 JPEGs ~q70 + ~55 KB BEV PNGs).
"""
from __future__ import annotations

import base64
import html as _html
import io
import json
import os
import subprocess
import sys

import numpy as np

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
PKG = os.path.join(REPO, "TanitAD Research Hub", "Data Engineering",
                   "Implementation", "incoming", "2026-08-16-s2-v1-labels")
SP = (r"C:\Users\Admin\AppData\Local\Temp\claude"
      r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
      r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
AST = os.path.join(SP, "s2rev_assets")
EGO = os.path.join(SP, "s2_ego")
OUT = os.path.join(PKG, "review", "VISUAL_REVIEW.html")
T0_IDX = 80
FPS = 10.0

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.collections import LineCollection    # noqa: E402


# --------------------------------------------------------------------------- #
# data                                                                         #
# --------------------------------------------------------------------------- #
def load_jsonl(p: str) -> dict[str, dict]:
    out = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        out[r["clip_id"]] = r
    return out


LAB_AUG = load_jsonl(os.path.join(PKG, "labels", "s2_labels_aug120.jsonl"))
EA_AUG = load_jsonl(os.path.join(PKG, "labels", "engine_a_aug120.jsonl"))
EA_VAL = load_jsonl(os.path.join(PKG, "labels", "engine_a_w120val.jsonl"))
ROWS = {r["clip_id"]: r for r in json.load(
    open(os.path.join(PKG, "raw", "review_rows_aug120.json"),
         encoding="utf-8"))}
EXCLUDED = {e["clip_id"]: e for e in json.load(
    open(os.path.join(PKG, "labels", "s2_excluded_w120val.json"),
         encoding="utf-8"))}
SEL = json.load(open(os.path.join(AST, "selection.json"), encoding="utf-8"))


def frame_indices(T: int) -> list[int]:
    """6 sample indices across the clip; t0 always exact."""
    idx = sorted({max(0, min(T - 1, i)) for i in
                  (int(0.10 * T), int(0.28 * T), min(T0_IDX, T - 1),
                   int(0.55 * T), int(0.72 * T), int(0.88 * T))})
    return idx


# --------------------------------------------------------------------------- #
# frames                                                                       #
# --------------------------------------------------------------------------- #
def frames_from_mp4(cid: str, idxs: list[int]) -> list[bytes] | None:
    mp4 = os.path.join(AST, "mp4", f"{cid}.mp4")
    if not os.path.exists(mp4):
        return None
    tmp = os.path.join(AST, "jpg_tmp")
    os.makedirs(tmp, exist_ok=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    expr = "+".join(f"eq(n\\,{i})" for i in idxs)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", mp4,
           "-vf", f"select='{expr}'", "-fps_mode", "vfr",
           "-q:v", "5", os.path.join(tmp, "f_%02d.jpg")]
    pr = subprocess.run(cmd, capture_output=True)
    outs = sorted(os.listdir(tmp))
    if pr.returncode != 0 or len(outs) != len(idxs):
        raise RuntimeError(f"ffmpeg {cid}: rc={pr.returncode} "
                           f"got {len(outs)}/{len(idxs)} frames "
                           f"{pr.stderr.decode()[:200]}")
    return [open(os.path.join(tmp, f), "rb").read() for f in outs]


def frames_from_v2ep(cid: str, idxs: list[int]) -> list[bytes] | None:
    """Slice raw JPEG bytes out of the v2ep payload (pose i = raw i+k)."""
    p = os.path.join(AST, "v2ep", f"{cid}.v2ep.pt")
    if not os.path.exists(p):
        return None
    import torch
    d = torch.load(p, map_location="cpu", weights_only=False)
    lens = d["jpeg_len"].to(torch.int64)
    offs = torch.cat([torch.zeros(1, dtype=torch.int64),
                      torch.cumsum(lens, 0)])
    buf = d["jpeg_buf"]
    k = int(d.get("n_stack", 1)) - 1
    codec = str(d.get("codec", "jpeg"))
    out = []
    for i in idxs:
        ri = min(i + k, len(lens) - 1)
        raw = bytes(buf[int(offs[ri]):int(offs[ri + 1])].numpy())
        if codec == "png":                        # transcode for size
            from PIL import Image
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            bio = io.BytesIO()
            im.save(bio, "JPEG", quality=70)
            raw = bio.getvalue()
        out.append(raw)
    return out


# --------------------------------------------------------------------------- #
# BEV                                                                          #
# --------------------------------------------------------------------------- #
def ego_frame_xy(poses: np.ndarray, t0: int) -> tuple[np.ndarray, np.ndarray]:
    """World -> ego@t0 (x fwd, y LEFT — refb_labels.ego_frame convention)."""
    dxy = poses[:, :2] - poses[t0, :2]
    yaw0 = poses[t0, 2]
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    return dxy[:, 0] * c - dxy[:, 1] * s, dxy[:, 0] * s + dxy[:, 1] * c


def arc_from_t0(poses: np.ndarray, t0: int) -> np.ndarray:
    seg = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
    a = np.zeros(len(poses))
    a[t0 + 1:] = np.cumsum(seg[t0:])
    return a


def point_at_arc(ex, ey, arc, t0, a):
    """Interpolated ego-frame point at arc distance `a` beyond t0."""
    fa = arc[t0:]
    if a <= 0:
        return ex[t0], ey[t0], True
    if a >= fa[-1]:
        return ex[-1], ey[-1], False
    j = int(np.searchsorted(fa, a))
    j0, j1 = t0 + j - 1, t0 + j
    w = (a - arc[j0]) / max(arc[j1] - arc[j0], 1e-9)
    return (ex[j0] + w * (ex[j1] - ex[j0]),
            ey[j0] + w * (ey[j1] - ey[j0]), True)


def gated_lc_event(ea: dict) -> dict | None:
    """Mirror of s2_derive._gated_lc_event, for display only."""
    for ev in (ea.get("lane_change_events") or []):
        if ev.get("token") in ("lc_left", "lc_right") \
                and abs(float(ev.get("lat_m") or 0.0)) >= 3.0:
            return ev
    return None


def bev_png(cid: str, poses: np.ndarray, lab: dict | None, ea: dict,
            fidx: list[int]) -> tuple[bytes, bool]:
    """Render the BEV. Returns (png, is_wide). LANE_TARGET gets a second,
    zoomed panel with a t0-parallel 3.5 m lane grid — at full-path scale a
    one-lane shift is sub-pixel, which is exactly the ambiguity S1 must
    resolve."""
    T = len(poses)
    t0 = min(T0_IDX, T - 1)
    ex, ey = ego_frame_xy(poses, t0)
    arc = arc_from_t0(poses, t0)
    v = poses[:, 3]
    tok = (lab or {}).get("g_str", {}).get("token")
    g = (lab or {}).get("g_str", {})
    is_lt = tok == "LANE_TARGET"

    if is_lt:
        fig, (ax, axz) = plt.subplots(
            1, 2, figsize=(8.4, 4.9), dpi=105,
            gridspec_kw={"width_ratios": [1.0, 1.0]})
    else:
        fig, ax = plt.subplots(figsize=(4.9, 5.4), dpi=105)
        axz = None

    def draw(a, legend: bool, colorbar: bool):
        a.plot(ey[:t0 + 1], ex[:t0 + 1], color="0.72", lw=1.6, zorder=2,
               label="past (0..t0)")
        pts = np.stack([ey, ex], axis=1)[t0:]
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lcoll = LineCollection(
            segs, cmap="viridis",
            norm=plt.Normalize(0.0, max(float(v.max()), 1.0)),
            lw=2.6, zorder=3)
        lcoll.set_array(0.5 * (v[t0:-1] + v[t0 + 1:]))
        a.add_collection(lcoll)
        if colorbar:
            cb = fig.colorbar(lcoll, ax=a, fraction=0.042, pad=0.03)
            cb.set_label("speed m/s", fontsize=7)
            cb.ax.tick_params(labelsize=6)
        a2i, b2i = max(0, t0 - 20), min(T - 1, t0 + 20)
        a.plot(ey[a2i:b2i + 1], ex[a2i:b2i + 1], color="#ff9d2e", lw=7,
               alpha=0.28, zorder=1, solid_capstyle="round",
               label="valid ±2 s")
        stopped = v < 0.5
        if stopped.any():
            a.plot(ey[stopped], ex[stopped], "x", color="#d62728", ms=4,
                   mew=1.1, zorder=4, label="v<0.5 m/s")
        for n, i in enumerate(fidx, 1):
            a.plot(ey[i], ex[i], "o", color="white", mec="#333", ms=9,
                   zorder=6)
            a.annotate(str(n), (ey[i], ex[i]), ha="center", va="center",
                       fontsize=6.2, zorder=7,
                       color="#b03000" if i == t0 else "#333")
        a.plot(0, 0, marker=(3, 0, 0), ms=13, color="#111", zorder=8)
        a.set_aspect("equal", adjustable="datalim")
        a.grid(True, lw=0.3, alpha=0.5)
        a.tick_params(labelsize=6.5)
        a.set_xlabel("lateral m   (+ = LEFT of t0 heading)", fontsize=7)
        if legend:
            a.set_ylabel("forward m (along t0 heading)", fontsize=7)
            a.legend(loc="best", fontsize=5.8, framealpha=0.85)

    draw(ax, legend=True, colorbar=not is_lt)

    # ---- goal annotation (main axis) --------------------------------------
    note = None
    if tok in ("TURN_LEFT", "TURN_RIGHT", "STOP_AT") and g.get("arg_mask",
                                                              [0])[0]:
        d = float(g["args"][0])
        gx, gy, on = point_at_arc(ex, ey, arc, t0, d)
        col = "#d62728" if tok == "STOP_AT" else "#7b2d8b"
        th = np.linspace(0, 2 * np.pi, 100)
        ax.plot(d * np.sin(th), d * np.cos(th), ls=(0, (4, 3)), lw=1.0,
                color=col, alpha=0.55, zorder=2)
        ax.plot(gy, gx, marker=("8" if tok == "STOP_AT" else "*"),
                ms=13 if tok != "STOP_AT" else 10, color=col, zorder=9,
                mec="white", mew=0.8)
        what = "stop point" if tok == "STOP_AT" else "maneuver start"
        note = f"{what} @ {d:.1f} m arc" + ("" if on else " (end of clip)")
    elif is_lt:
        dirn = +1.0 if float(g["args"][0]) > 0 else -1.0
        onset = float(g["args"][1]) if g["arg_mask"][1] else 0.0
        ev = gated_lc_event(ea)
        lat = float(ev["lat_m"]) if ev else None
        t_a = float(ev.get("t_start_s") or 0.0) if ev else 0.0
        t_b = float(ev.get("t_end_s") or 4.0) if ev else 4.0
        # zoom panel: a SHORT forward strip after the onset (before road
        # curvature dominates the t0-parallel frame), lane grid for scale.
        # Lateral axis is deliberately stretched (aspect auto): a 3.5 m
        # shift must be legible; the grid carries the true scale.
        gx0, _, _ = point_at_arc(ex, ey, arc, t0, onset)
        f_lo, f_hi = gx0 - 15.0, gx0 + 65.0
        m = (ex >= f_lo) & (ex <= f_hi)
        ys = ey[m] if m.any() else ey
        draw(axz, legend=False, colorbar=False)
        for k in (-5.25, -1.75, 1.75, 5.25):
            axz.axvline(k, ls=(0, (5, 4)), lw=0.8, color="#8a6fa0",
                        alpha=0.55, zorder=0)
        gx, gy, _ = point_at_arc(ex, ey, arc, t0, onset)
        for a_ in (ax, axz):
            a_.annotate("", xytext=(gy, gx), xy=(gy + dirn * 3.5, gx),
                        arrowprops=dict(arrowstyle="-|>", lw=2.2,
                                        color="#7b2d8b"), zorder=9)
        latspan = max(7.0, float(np.abs(ys).max()) + 1.5)
        axz.set_aspect("auto")
        axz.set_xlim(latspan, -latspan)            # +left renders left
        axz.set_ylim(f_lo, f_hi)
        axz.set_title(
            f"zoom −15…+65 m fwd · lat axis stretched · "
            f"lc t0+[{t_a:.0f},{t_b:.0f}] s · grid = 3.5 m lanes",
            fontsize=6.6)
        note = (f"lane {'LEFT' if dirn > 0 else 'RIGHT'} "
                f"(lat {lat:+.2f} m), onset @ {onset:.0f} m arc")
    elif tok in ("TURN_LEFT", "TURN_RIGHT"):
        note = "turn direction only — route fired without a distance"

    ax.invert_xaxis()                       # +y (left) renders on the LEFT
    ttl = f"{tok or 'excluded'}"
    if note:
        ttl += f" — {note}"
    if is_lt:
        fig.suptitle(ttl, fontsize=8, y=0.995)
        ax.set_title("full clip", fontsize=7)
    else:
        ax.set_title(ttl, fontsize=7.6)
    bio = io.BytesIO()
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.97) if is_lt else None)
    fig.savefig(bio, format="png")
    plt.close(fig)
    return bio.getvalue(), is_lt


# --------------------------------------------------------------------------- #
# card text                                                                    #
# --------------------------------------------------------------------------- #
def esc(x) -> str:
    return _html.escape(str(x))


def fmt_args(blk: dict) -> str:
    names = ("arg0", "arg1", "arg2", "arg3", "within_m", "by_time_s",
             "at_arc_m", "hold_for_s")
    parts = [f"{names[i]}={blk['args'][i]:g}"
             for i in range(8) if blk["arg_mask"][i]]
    return ", ".join(parts) if parts else "no args"


def check_text(lab: dict, ea: dict) -> str:
    g = lab["g_str"]
    tok = g["token"]
    d = g["args"][0] if g["arg_mask"][0] else None
    if tok in ("TURN_LEFT", "TURN_RIGHT"):
        side = "LEFT" if tok == "TURN_LEFT" else "RIGHT"
        if "u_turn" in str(g["sources"]):
            return (f"video should show a U-TURN (extreme left, dyaw≈171°) "
                    f"starting ~{d:.0f} m ahead — labeled TURN_LEFT by the "
                    f"u-turn⊂left convention. WRONG if it is a roundabout.")
        w = (f"video should show a {side} turn starting ~{d:.1f} m ahead of "
             f"the t0 frame (frame 3)." if d is not None else
             f"video should show a {side} turn/curve commitment — the route "
             f"fired with NO measured distance (dist=None).")
        return w + " WRONG if the road merely curves and no junction/turn " \
                   "maneuver happens."
    if tok == "STOP_AT":
        dd = f"~{d:.1f} m ahead" if d is not None else \
            "at an unmeasured distance (profile-level stop)"
        return (f"ego should come to a full stop {dd} (red light / queue / "
                f"stop sign visible). WRONG if it merely slows or never "
                f"stops.")
    if tok == "LANE_TARGET":
        side = "LEFT" if float(g["args"][0]) > 0 else "RIGHT"
        return (f"ego should deliberately change ONE LANE to the {side} "
                f"(onset near the marked arc). WRONG if the lateral shift is "
                f"a road curve, a bend drift, or a widening lane — that is "
                f"the gentle-curve-drift failure this section exists to "
                f"catch.")
    if tok == "FOLLOW_MAIN_ROAD":
        return ("NEGATIVE CONTROL: no turn, no stop, no lane change — plain "
                "corridor following. WRONG if any junction turn, stop, or "
                "lane change is clearly visible (a default hiding a "
                "maneuver = false negative).")
    if tok == "NONE_ABSTAIN":
        return (f"abstained: {esc(lab['g_str'].get('reason') or '')} — "
                f"confirm that neither a junction turn nor a confident "
                f"FOLLOW is clearly right.")
    return ""


def corro_chips(lab: dict) -> str:
    g = lab["g_str"]
    corr = g.get("corroboration") or {}
    chips = []
    vg = corr.get("vlm_goal_kind")
    if corr.get("remapped_from_route_to"):
        chips.append('<span class="chip remap">ROUTE_TO remap</span>')
    if vg:
        ag = corr.get("agrees")
        cls, mark = (("agree", "✓") if ag else
                     (("disagree", "✗") if ag is False else ("na", "·")))
        chips.append(f'<span class="chip {cls}">VLM: {esc(vg)} {mark}</span>')
    else:
        chips.append('<span class="chip na">VLM: no goal</span>')
    a = lab["a_str"]
    verbs = (a.get("corroboration") or {}).get("vlm_verbs") or []
    for vb in verbs[:3]:
        gm = vb.get("geometry")
        cls = "agree" if gm == "ok" else "disagree"
        chips.append(f'<span class="chip {cls}">vlm verb '
                     f'{esc(vb.get("verb"))}({esc(vb.get("direction"))}) '
                     f'{esc(gm)}</span>')
    chips.append(f'<span class="chip prov">prov: {esc(g["provenance"])}'
                 f' · conf {g.get("confidence")}</span>')
    return "".join(chips)


def ea_line(ea: dict) -> str:
    r = ea.get("route") or {}
    sp = ea.get("speed_profile") or {}
    ev = []
    for e in (ea.get("lane_change_events") or [])[:2]:
        ev.append(f"{e['token']} lat={e.get('lat_m')} m "
                  f"@{e.get('arc_from_t0_m')} m")
    for e in (ea.get("speed_events") or [])[:2]:
        sd = e.get("stop_dist_m")
        ev.append(f"{e['token']}{f' stop@{sd} m' if sd is not None else ''}")
    dist = r.get("dist_m")
    return (f"route <b>{esc(r.get('token'))}</b>"
            f"{'' if r.get('token_valid') else ' (invalid)'}"
            f" · dyaw {r.get('maneuver_dyaw_rad')} rad"
            f" · dist {dist if dist is None else round(float(dist), 1)} m"
            f" · v_t0 {sp.get('v_t0_ms')} m/s · net_dv {sp.get('net_dv_ms')}"
            f" m/s · stops {sp.get('stops')}"
            + (f" · events: {esc('; '.join(ev))}" if ev else ""))


# --------------------------------------------------------------------------- #
# sections                                                                     #
# --------------------------------------------------------------------------- #
SECTIONS = [
    ("S1_LANE_TARGET", "S1 · LANE_TARGET — all 19 (the gate)",
     "LANE_TARGET has ZERO VLM corroboration (0/19). Either the VLM misses "
     "realized lane changes (plausible: its turn recall was 2/29) or "
     "gentle-curve drift survives the follow+|lat|≥3.0 m gate. Your verdicts "
     "decide: mostly CORRECT → keep the family; mostly WRONG → one-constant "
     "re-emit (raise LC_MIN_LAT_M or require VLM/Alpamayo corroboration), "
     "7 min. Each card shows the measured lateral displacement."),
    ("S2_TURN", "S2 · TURN spot-checks (both directions + the u-turn)",
     "Ratifies the direction conventions the whole label set uses: dyaw>0 = "
     "LEFT = +1 (ego frame +y = left), u-turn⊂TURN_LEFT, and the "
     "dist-to-maneuver arg. If a TURN_LEFT card visibly turns right, the "
     "sign convention is inverted program-wide — that is what these 6 clips "
     "rule out. Includes dist=None edge cases."),
    ("S3_STOP_AT", "S3 · STOP_AT (distance-arg sanity)",
     "First-ever STOP_AT emission. Verdicts decide whether the stop-distance "
     "args (lonmode stop events / profile stops) are trustworthy as training "
     "signal — the stop point is drawn on the BEV and the speed coloring "
     "must go to zero at it."),
    ("S4_ROUTE_TO_DISPOSITION", "S4 · ROUTE_TO disposition (remaps + abstain)",
     "ROUTE_TO is gated (G1 closed 0/31, validator-refused). The 31 VLM "
     "route_to claims were remapped to their geometry token (30) or "
     "abstained (1). Verdicts ratify the remap-or-abstain policy: a remap "
     "card whose video contradicts its geometry token would reopen the "
     "disposition; a clean sweep keeps the gate closed."),
    ("S5_EXCLUDED_VAL", "S5 · the 4 excluded val records",
     "Excluded as triple-empty (VLM/SAM3/Alpamayo absent, ego_state null) — "
     "their fused NONE_ABSTAIN was a default-of-absence, not a judgement. "
     "Confirm they are garbage records (exclusion right), and note their "
     "Engine A geometry still exists — recoverable on the standing val600 "
     "re-fuse decision."),
    ("S6_FOLLOW_CONTROLS", "S6 · FOLLOW_MAIN_ROAD negative controls",
     "The default token (98/201). A FOLLOW card whose video shows a turn / "
     "stop / lane change is a FALSE NEGATIVE of the geometry gates — the "
     "miss direction the positive sections cannot see. Includes the one "
     "merge-backed FOLLOW and a no-valid-route default."),
]

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1c2733;--mut:#5b6b7b;--line:#dde3ea;
--acc:#7b2d8b;--ok:#0d7a3f;--bad:#b3261e;--warn:#a35b00}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 "Segoe UI",system-ui,
sans-serif;background:var(--bg);color:var(--ink);padding:18px}
h1{font-size:21px;margin:0 0 6px}h2{font-size:17px;margin:34px 0 4px;
border-top:3px solid var(--acc);padding-top:12px}
.sub{color:var(--mut);font-size:12.5px;margin:2px 0 10px}
.decision{background:#f3e9f7;border-left:4px solid var(--acc);padding:8px
 12px;font-size:12.8px;margin:8px 0 14px;border-radius:0 6px 6px 0}
.decision b{color:var(--acc)}
.clip{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:12px 14px;margin:12px 0;box-shadow:0 1px 2px rgba(20,30,40,.05)}
.clip-head{display:flex;flex-wrap:wrap;align-items:center;gap:8px}
.idx{background:var(--ink);color:#fff;border-radius:6px;padding:2px 8px;
font-size:12px;font-weight:600}
code{font:12px Consolas,monospace;color:var(--mut)}
.label-line{font-size:14px;margin:7px 0 2px}
.label-line b{font-size:15px}
.tok{color:var(--acc)}
.meta{font-size:12.3px;color:var(--mut);margin:2px 0}
.check{background:#fff8e6;border-left:4px solid var(--warn);padding:6px 10px;
font-size:12.8px;margin:8px 0;border-radius:0 6px 6px 0}
.chip{display:inline-block;font-size:11px;border-radius:20px;padding:1px 9px;
margin:1px 3px 1px 0;border:1px solid var(--line);background:#f2f4f7}
.chip.agree{background:#e7f5ec;color:var(--ok);border-color:#bfe3cd}
.chip.disagree{background:#fdeceb;color:var(--bad);border-color:#f3c1be}
.chip.remap{background:#efe3f5;color:var(--acc);border-color:#d9c2e4;
font-weight:600}
.chip.na{color:var(--mut)}
.chip.prov{background:#eef2f6}
.media{display:flex;flex-wrap:wrap;gap:10px;margin-top:9px;align-items:
flex-start}
.frames{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;flex:1 1
 560px;min-width:340px}
.fr{position:relative}
.fr img{width:100%;display:block;border-radius:5px}
.fr.t0 img{outline:3px solid #ff9d2e}
.fr .t{position:absolute;left:4px;bottom:4px;background:rgba(10,14,18,.72);
color:#fff;font-size:10.5px;padding:0 6px;border-radius:4px}
.fr.t0 .t{background:#c96a00}
.bev{flex:0 1 430px;min-width:320px}
.bev.wide{flex:1 1 640px;min-width:480px}
.bev img{width:100%;border:1px solid var(--line);border-radius:6px}
.noframes{flex:1 1 560px;min-width:340px;background:#f1f3f5;border:1px dashed
 var(--line);border-radius:6px;padding:20px;color:var(--mut);font-size:12.5px}
.verdict{margin-left:auto;display:flex;gap:6px;align-items:center;
font-size:12.5px}
.verdict label{border:1px solid var(--line);border-radius:6px;padding:2px
 8px;cursor:pointer;background:#fbfcfd}
.verdict input{vertical-align:-1px;margin:0 3px 0 0}
.verdict label:has(input[value=correct]:checked){background:#e7f5ec;
border-color:var(--ok)}
.verdict label:has(input[value=wrong]:checked){background:#fdeceb;
border-color:var(--bad)}
.verdict label:has(input[value=unsure]:checked){background:#fff3da;
border-color:var(--warn)}
.note{border:1px solid var(--line);border-radius:6px;padding:3px 8px;
font-size:12px;width:170px}
#exportbox{width:100%;height:130px;font:11px Consolas,monospace;margin-top:6px}
.btn{background:var(--acc);color:#fff;border:0;border-radius:7px;padding:7px
 14px;font-size:13px;cursor:pointer}
.summary{font-size:12.5px;color:var(--mut);margin-left:10px}
.hdr{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:14px 16px}
.hdr table{border-collapse:collapse;font-size:12.5px;margin-top:6px}
.hdr td,.hdr th{border:1px solid var(--line);padding:3px 9px;text-align:left}
.small{font-size:11.5px;color:var(--mut)}
"""

JS = """
const KEY='s2rev_v1_';
function save(cid){
 const v=document.querySelector(`input[name="v_${cid}"]:checked`);
 const n=document.getElementById(`n_${cid}`);
 localStorage.setItem(KEY+cid,JSON.stringify({v:v?v.value:null,note:n.value}));
 refresh();}
function refresh(){
 let c=0,w=0,u=0,t=0;
 document.querySelectorAll('.clip[data-clip]').forEach(el=>{t++;
  const s=localStorage.getItem(KEY+el.dataset.clip);
  if(!s)return; const o=JSON.parse(s);
  if(o.v==='correct')c++;else if(o.v==='wrong')w++;
  else if(o.v==='unsure')u++;});
 document.getElementById('sum').textContent=
  `${c+w+u}/${t} judged — ${c} correct · ${w} wrong · ${u} unsure`;}
function doExport(){
 const out={_sheet:'S2 v1 visual review aug120', _exported:new Date()
  .toISOString(), verdicts:{}};
 document.querySelectorAll('.clip[data-clip]').forEach(el=>{
  const s=localStorage.getItem(KEY+el.dataset.clip);
  out.verdicts[el.dataset.clip]=Object.assign(
   {section:el.dataset.sec,label:el.dataset.tok},
   s?JSON.parse(s):{v:null,note:''});});
 const box=document.getElementById('exportbox');
 box.style.display='block';box.value=JSON.stringify(out,null,1);
 box.select();try{document.execCommand('copy');}catch(e){}}
window.addEventListener('DOMContentLoaded',()=>{
 document.querySelectorAll('.clip[data-clip]').forEach(el=>{
  const cid=el.dataset.clip,s=localStorage.getItem(KEY+cid);
  if(s){const o=JSON.parse(s);
   if(o.v){const r=el.querySelector(`input[value="${o.v}"]`);if(r)r.checked=1;}
   el.querySelector('.note').value=o.note||'';}
  el.querySelectorAll('input[type=radio]').forEach(r=>
   r.addEventListener('change',()=>save(cid)));
  el.querySelector('.note').addEventListener('input',()=>save(cid));});
 refresh();});
"""


def b64img(data: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def clip_card(sec: str, n: int, cid: str, split: str) -> str:
    is_val = split == "w120val"
    lab = LAB_AUG.get(cid)
    ea = (EA_VAL if is_val else EA_AUG)[cid]["engine_a"]
    row = ROWS.get(cid, {})
    npz = np.load(os.path.join(EGO, split, f"{cid}.npz"))
    poses = np.asarray(npz["poses"], dtype=np.float64)
    T = len(poses)
    fidx = frame_indices(T)

    # frames -----------------------------------------------------------------
    reason = None
    try:
        frames = (frames_from_v2ep(cid, fidx) if is_val
                  else frames_from_mp4(cid, fidx))
        if frames is None:
            reason = ("no mp4 in Sayood/tanitad-ph0-aug120 "
                      "bridged_w120train_2400/videos/ for this clip"
                      if not is_val else
                      "v2ep.pt shard not pulled from "
                      "Sayood/tanitad-physicalai-w120-256x640cyl")
    except Exception as e:                                  # noqa: BLE001
        frames, reason = None, f"{type(e).__name__}: {e}"

    if frames:
        cells = []
        for k, (i, jpg) in enumerate(zip(fidx, frames), 1):
            t0c = " t0" if i == min(T0_IDX, T - 1) else ""
            tt = f"{k} · t={i / FPS:.1f}s" + (" (t0)" if t0c else "")
            cells.append(f'<div class="fr{t0c}"><img loading="lazy" '
                         f'src="{b64img(jpg)}" alt="f{k}">'
                         f'<span class="t">{tt}</span></div>')
        media_frames = f'<div class="frames">{"".join(cells)}</div>'
    else:
        media_frames = (f'<div class="noframes">frames unavailable — '
                        f'{esc(reason)}<br>BEV (from the ego npz) is still '
                        f'authoritative for geometry.</div>')

    bev_bytes, bev_wide = bev_png(cid, poses, lab, ea, fidx)
    bev = b64img(bev_bytes, "image/png")
    bev_cls = "bev wide" if bev_wide else "bev"

    # text -------------------------------------------------------------------
    if lab:
        g, a = lab["g_str"], lab["a_str"]
        tok = g["token"]
        label_line = (
            f'g_str <b class="tok">{esc(tok)}</b> ({esc(fmt_args(g))}) '
            f'&nbsp;·&nbsp; a_str <b>{esc(a["token"])}</b> '
            f'({esc(fmt_args(a))})')
        src = ", ".join(g["sources"]) + " / " + ", ".join(a["sources"])
        chips = corro_chips(lab)
        check = check_text(lab, ea)
    else:                                   # excluded val record
        exc = EXCLUDED[cid]
        tok = "EXCLUDED"
        label_line = ('<b class="tok">EXCLUDED from the S2 label set</b> '
                      '(no g_str/a_str emitted)')
        src = "exclusion list: labels/s2_excluded_w120val.json"
        chips = ('<span class="chip disagree">triple-empty: VLM+SAM3+'
                 'Alpamayo absent, ego_state null</span>')
        check = (f"confirm this record deserved exclusion — "
                 f"{esc(exc['reason'])} Note: {esc(exc['note'])}")

    scen = row.get("scenario")
    scen_html = (f'<div class="meta">scene: {esc(scen)}</div>' if scen else "")

    return f"""
<div class="clip" data-clip="{cid}" data-sec="{sec}" data-tok="{esc(tok)}">
 <div class="clip-head"><span class="idx">{sec.split('_')[0]}.{n:02d}</span>
  <code>{cid}</code>{'<span class="chip na">val split</span>' if is_val
                     else ''}
  <div class="verdict">
   <label><input type="radio" name="v_{cid}" value="correct">correct</label>
   <label><input type="radio" name="v_{cid}" value="wrong">wrong</label>
   <label><input type="radio" name="v_{cid}" value="unsure">unsure</label>
   <input class="note" id="n_{cid}" type="text" placeholder="note…">
  </div></div>
 <div class="label-line">{label_line}</div>
 <div class="meta">{chips}</div>
 <div class="meta">Engine A (hindsight): {ea_line(ea)}</div>
 <div class="meta small">sources: {esc(src)}</div>
 {scen_html}
 <div class="check"><b>CHECK:</b> {check}</div>
 <div class="media">{media_frames}
  <div class="{bev_cls}"><img loading="lazy" src="{bev}" alt="BEV"></div>
 </div>
</div>"""


def main() -> int:
    parts = []
    n_total = 0
    n_noframes = 0
    for key, title, decision in SECTIONS:
        cids = SEL[key]
        cards = []
        for i, cid in enumerate(cids, 1):
            split = "w120val" if key == "S5_EXCLUDED_VAL" else "aug120"
            card = clip_card(key, i, cid, split)
            if "frames unavailable" in card:
                n_noframes += 1
            cards.append(card)
            n_total += 1
            print(f"[{key}] {i}/{len(cids)} {cid}", flush=True)
        parts.append(f'<section id="{key}"><h2>{esc(title)} '
                     f'<span class="summary">n={len(cids)}</span></h2>'
                     f'<div class="decision"><b>DECISION THIS FEEDS:</b> '
                     f'{esc(decision)}</div>{"".join(cards)}</section>')

    counts = " · ".join(f"{k.split('_')[0]} n={len(SEL[k])}" for k, _, _
                        in SECTIONS)
    head = f"""
<div class="hdr"><h1>S2 v1 strategic labels — visual review (aug120)</h1>
<div class="sub">One card per clip: camera frames (t0 orange, numbered dots
on the BEV are the same frames) + BEV in metres from the bridged ego npz
(heading-up at t0, <b>+lateral = LEFT</b>, future path colored by speed,
orange band = the ±2 s validity window) + the emitted g_str/a_str with args
and provenance. Judge each card: does the label match what the video shows?
Verdicts persist in your browser (localStorage); press
<b>Export verdicts</b> when done — it copies a JSON to the clipboard.</div>
<div class="sub">{n_total} clips: {counts}. Labels:
<code>labels/s2_labels_aug120.jsonl</code> (201) — this sheet covers the
decision-bearing strata, not all 201. Frame sources: aug120 =
pose-aligned bridge mp4s (Sayood/tanitad-ph0-aug120, frame k = pose k,
10 Hz); val = raw JPEGs from the .v2ep.pt shards
(Sayood/tanitad-physicalai-w120-256x640cyl).</div>
<button class="btn" onclick="doExport()">Export verdicts (JSON →
clipboard)</button><span class="summary" id="sum"></span>
<textarea id="exportbox" style="display:none"></textarea></div>"""

    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width,'
           f'initial-scale=1"><title>S2 v1 visual review — aug120</title>'
           f'<style>{CSS}</style></head><body>{head}{"".join(parts)}'
           f'<script>{JS}</script></body></html>')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(doc)
    mb = os.path.getsize(OUT) / 1e6
    print(f"WROTE {OUT}  {mb:.1f} MB  clips={n_total} "
          f"noframes={n_noframes}", flush=True)
    assert mb < 40, "over the 40 MB budget"
    return 0


if __name__ == "__main__":
    sys.exit(main())
