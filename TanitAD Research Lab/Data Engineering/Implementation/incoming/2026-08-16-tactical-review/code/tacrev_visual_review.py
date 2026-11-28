#!/usr/bin/env python3
"""Build the TACTICAL visual review sheet — real frames, per-axis, per-leg.

⛔ WHY THIS EXISTS. `TACTICAL_LABEL_VALIDATION.md` §5.2 item 1: *"No human has
looked at a single tactical label. This is the single largest gap."* Every
number in that document is machine-vs-machine agreement. The strategic sheet
that preceded this one caught a ≈78 % error rate on a whole label class, and
the PI's feedback on the FIRST attempt was that a text-only artifact is not
reviewable — *"I dont see any clips or visual elements"*. ⇒ **Frames are
mandatory.** This script refuses to emit a card without them.

WHAT IS SHOWN, and why each choice is forced by a measurement:

  * **HORIZON 2.0 s.** MEASURED (`…/2026-08-16-tactical-labels/raw/
    a4_horizon_sweep.json`): κ vs Alpamayo peaks at 2.0 s on BOTH axes —
    LON 0.188 → **0.3655**, LAT 0.290 → **0.4694** — and 2.0 s is the v6
    tactical band (`v6.py:161`, g_tac = the 2–6 s layer). The production
    reading (full 11.8 s window) sits in the WORST region of that surface. A
    label without its horizon is not reviewable, so the horizon is printed on
    the sheet and the frames span t0−1.0 s → t0+2.0 s.
  * **LAT and LON SEPARATELY, never pooled** — the four-metric-families rule.
    Each axis carries its own verdict control, because a clip can be right on
    one axis and wrong on the other and one button cannot say so.
  * **THREE LEGS SIDE BY SIDE** so the PI can see WHICH leg is wrong, not
    merely that something is. ⚠️ The sheet states, on every card, that the
    VLM and ego legs are **NOT independent** (MEASURED 201/201: the VLM
    prompt carries the ego `motion`/`turning` fields the ego voter reads).
  * **The Alpamayo `cot` reasoning**, verbatim — the PI asked about reasoning
    specifically, and it is the only way to judge whether the rationale
    matches the frames.

⛔ The VLM leg here is the **POST-FIX** mapping, imported live from
`stack/scripts/ph1_fuse.py`. The banked corpus still carries the pre-fix
tokens; it has NOT been re-fused (escalated separately). The sheet says so.

Usage (CPU only, no GPU, no network):
  python tacrev_visual_review.py --out <html> --meta <json> [--max-clips N]
"""
from __future__ import annotations

import argparse
import base64
import collections
import glob
import html
import io
import json
import os
import subprocess
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]
_STACK = Path(__file__).resolve().parents[6] / "stack"
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

import ph1_fuse  # noqa: E402

SP = (r"C:\Users\Admin\AppData\Local\Temp\claude"
      r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
      r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
FFMPEG = (r"C:\Users\Admin\AppData\Local\Microsoft\WinGet\Packages"
          r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
          r"\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe")
HUB = r"C:\Users\Admin\.cache\huggingface\hub\datasets--Sayood--tanitad-ph0-aug120"
INC = _PKG.parent

T0_IDX = 80
FPS = 10.0
#: ⭐ THE READ HORIZON. Both axes' κ peak here; it is the v6 tactical band.
H_S = 2.0
H_IDX = T0_IDX + int(round(H_S * FPS))
#: the thresholds that maximised κ at this horizon (a4 sweep, best cell)
DV_THRESH = 0.75          # m/s
DYAW_THRESH = 0.05        # rad
#: frames: one second of run-up, then the whole 2 s tactical window
FRAME_IDXS = [T0_IDX - 10, T0_IDX, T0_IDX + 5, T0_IDX + 10,
              T0_IDX + 15, T0_IDX + 20]

# --------------------------------------------------------------------------- #
# 3-BAND PROJECTIONS — FOR STRATIFICATION ONLY.                                 #
# ⚠️ They DISCARD severity (Sharp Steer and Steer collapse), so any agreement   #
# figure computed on them is an UPPER BOUND. They decide which clips are shown, #
# never what the card claims: each card prints every leg's NATIVE opinion.      #
# --------------------------------------------------------------------------- #
ALP_LAT3 = {"go straight": "straight", "steer left": "left",
            "steer right": "right", "sharp steer left": "left",
            "sharp steer right": "right", "reverse left": "left",
            "reverse right": "right"}
ALP_LON3 = {"gentle deceleration": "decelerate", "strong deceleration":
            "decelerate", "stop": "decelerate", "maintain speed": "maintain",
            "gentle acceleration": "accelerate", "strong acceleration":
            "accelerate", "reverse": None}
V6_LAT3 = {"LANE_KEEP": "straight", "LANE_CHANGE_L": "left",
           "LANE_CHANGE_R": "right", "NUDGE_L": "left", "NUDGE_R": "right",
           "ABORT_LC": None}
V6_LON3 = {"BRAKE_TO": "decelerate", "CREEP": "decelerate",
           "HOLD": "decelerate", "YIELD_MERGE": "decelerate",
           "CRUISE": "maintain", "FOLLOW": "maintain"}


def b64img(data: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def find_mp4s() -> dict:
    """Every locally-present clip video. ⚠️ ABSENCE FOUND AT ONE LOCATION IS
    NOT ABSENCE — three roots are probed, and the HF cache is one of them."""
    out: dict[str, str] = {}
    for pat in (os.path.join(SP, "s2rev_assets", "mp4", "*.mp4"),
                os.path.join(SP, "sam3fix_assets", "videos", "*.mp4"),
                os.path.join(HUB, "snapshots", "*", "bridged_w120train_2400",
                             "videos", "*.mp4")):
        for p in glob.glob(pat):
            out.setdefault(os.path.basename(p)[:-4], p)
    return out


def frames_from_mp4(mp4: str, idxs: list[int]) -> list[bytes] | None:
    tmp = os.path.join(SP, "tacrev_tmp")
    os.makedirs(tmp, exist_ok=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    expr = "+".join(f"eq(n\\,{i})" for i in idxs)
    cmd = [FFMPEG, "-y", "-v", "error", "-i", mp4, "-vf", f"select='{expr}'",
           "-fps_mode", "vfr", "-q:v", "5", os.path.join(tmp, "f_%02d.jpg")]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except Exception:                                            # noqa: BLE001
        return None
    got = sorted(glob.glob(os.path.join(tmp, "f_*.jpg")))
    if len(got) != len(idxs):
        return None
    return [open(p, "rb").read() for p in got]


def bev_png(poses) -> bytes:
    """Metric BEV of the 2 s tactical window in the ego frame at t0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x0, y0, yaw0 = poses[T0_IDX, 0], poses[T0_IDX, 1], poses[T0_IDX, 2]
    dx, dy = poses[:, 0] - x0, poses[:, 1] - y0
    fwd = dx * np.cos(yaw0) + dy * np.sin(yaw0)
    lft = -dx * np.sin(yaw0) + dy * np.cos(yaw0)
    lo, hi = max(0, T0_IDX - 40), min(len(poses) - 1, T0_IDX + 60)
    fig, ax = plt.subplots(figsize=(2.5, 3.1), dpi=105)
    ax.plot(-lft[lo:hi], fwd[lo:hi], color="#c9ccd1", lw=1.4, zorder=1)
    h = min(H_IDX, len(poses) - 1)
    ax.plot(-lft[T0_IDX:h + 1], fwd[T0_IDX:h + 1], color="#1f6feb", lw=2.8,
            zorder=3, solid_capstyle="round")
    ax.scatter([0], [0], s=52, c="#111", marker="^", zorder=4)
    ax.scatter([-lft[h]], [fwd[h]], s=46, c="#1f6feb", zorder=4)
    ax.annotate(f"+{H_S:.1f}s", (-lft[h], fwd[h]), textcoords="offset points",
                xytext=(5, 2), fontsize=7, color="#1f6feb")
    ax.axhline(0, color="#e5e7eb", lw=0.8, zorder=0)
    ax.axvline(0, color="#e5e7eb", lw=0.8, zorder=0)
    span = max(12.0, float(np.nanmax(np.abs(fwd[T0_IDX:h + 1]))) * 1.25)
    ax.set_xlim(-span * 0.6, span * 0.6)
    ax.set_ylim(-span * 0.18, span)
    ax.set_xlabel("← left    lateral m    right →", fontsize=6.5)
    ax.set_ylabel("forward m", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.set_title("BEV · ego frame @ t0", fontsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(pad=0.3)
    bio = io.BytesIO()
    fig.savefig(bio, format="png")
    plt.close(fig)
    return bio.getvalue()


def ego_at_horizon(poses) -> dict:
    """The ego GEOMETRY leg, read at H = 2.0 s.

    ⚠️ The yaw sign convention is MEASURED, not assumed: over the 201 clips
    with poses, mean Δyaw@2.0s is **+0.1963** for Alpamayo `Sharp Steer Left`
    and **−0.5371** for `Sharp Steer Right` (`Go Straight` −0.0006).
    ⇒ POSITIVE Δyaw = LEFT.
    """
    h = min(H_IDX, len(poses) - 1)
    dv = float(poses[h, 3] - poses[T0_IDX, 3])
    dyaw = float(poses[h, 2] - poses[T0_IDX, 2])
    lat3 = ("left" if dyaw >= DYAW_THRESH else
            "right" if dyaw <= -DYAW_THRESH else "straight")
    lon3 = ("accelerate" if dv >= DV_THRESH else
            "decelerate" if dv <= -DV_THRESH else "maintain")
    return {"dv": dv, "dyaw": dyaw, "lat3": lat3, "lon3": lon3,
            "v_t0": float(poses[T0_IDX, 3]), "v_h": float(poses[h, 3])}


def agreement(a, b, c) -> str:
    """Stratum of three opinions on one axis. `None` = the leg was SILENT and
    is counted separately — silence is never folded into disagreement."""
    spoke = [x for x in (a, b, c) if x]
    if len(spoke) < 2:
        return "insufficient"
    u = set(spoke)
    if len(u) == 1:
        return "unanimous" if len(spoke) == 3 else "agree2"
    if len(spoke) == 3 and len(u) == 3:
        return "three_way_split"
    return "two_of_three"


def main() -> int:
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_PKG / "review" /
                                         "TACTICAL_VISUAL_REVIEW.html"))
    ap.add_argument("--meta", default=str(_PKG / "raw" /
                                          "tacrev_selection.json"))
    ap.add_argument("--max-clips", type=int, default=44)
    a = ap.parse_args()

    tl = INC / "2026-08-16-tactical-labels" / "raw"
    a1 = {}
    for line in open(tl / "a1_alpamayo_taxonomy_per_clip.jsonl",
                     encoding="utf-8"):
        r = json.loads(line)
        a1[r["clip_id"]] = r
    v2raw = json.load(open(os.path.join(SP, "aug120", "merged",
                                        "ph0_v2.json"), encoding="utf-8"))
    v2raw = v2raw if isinstance(v2raw, list) else v2raw.get("clips", v2raw)
    v2_by = {r["clip_id"]: r for r in v2raw if isinstance(r, dict)}
    mp4s = find_mp4s()

    # ------------------------------------------------------------------ #
    # build one row per candidate clip: three legs, both axes, at H=2.0 s #
    # ------------------------------------------------------------------ #
    rows = []
    for cid, v2 in sorted(v2_by.items()):
        npz = os.path.join(SP, "s2_ego", "aug120", f"{cid}.npz")
        if cid not in mp4s or not os.path.exists(npz) or cid not in a1:
            continue
        poses = np.load(npz)["poses"]
        if len(poses) <= T0_IDX:
            continue
        ego = ego_at_horizon(poses)
        alp = a1[cid]
        alp_lane = (alp.get("lane") or "").strip()
        alp_lat_raw = (alp.get("lateral") or "").strip()
        alp_lon_raw = (alp.get("longitudinal") or "").strip()
        # ---- Alpamayo through the POST-FIX explicit mapping -------------
        alp_lat_v6, alp_lon_v6, alp_notes = ph1_fuse.map_alpamayo_axes(
            {"lane": alp_lane or None, "longitudinal": alp_lon_raw or None})
        # ---- VLM through the POST-FIX explicit mapping ------------------
        vlm_acts, vlm_lat_v6, vlm_lon_v6 = [], None, None
        for act in ((v2.get("symbols") or {}).get("actions") or []):
            lat_t, lon_t, _n = ph1_fuse.map_vlm_action(act.get("verb"),
                                                       act.get("direction"))
            vlm_acts.append({"verb": act.get("verb"),
                             "direction": act.get("direction"),
                             "lat": lat_t, "lon": lon_t})
            vlm_lat_v6 = vlm_lat_v6 or lat_t
            vlm_lon_v6 = vlm_lon_v6 or lon_t
        egost = v2.get("ego_state") or {}
        rows.append({
            "clip_id": cid, "mp4": mp4s[cid], "npz": npz,
            "alp_lane": alp_lane, "alp_lat_raw": alp_lat_raw,
            "alp_lon_raw": alp_lon_raw, "cot": alp.get("cot") or "",
            "alp_lat_v6": alp_lat_v6, "alp_lon_v6": alp_lon_v6,
            "alp_notes": alp_notes,
            "alp_lat3": ALP_LAT3.get(alp_lat_raw.lower()),
            "alp_lon3": ALP_LON3.get(alp_lon_raw.lower()),
            "vlm_acts": vlm_acts, "vlm_lat_v6": vlm_lat_v6,
            "vlm_lon_v6": vlm_lon_v6,
            "vlm_lat3": V6_LAT3.get(vlm_lat_v6),
            "vlm_lon3": V6_LON3.get(vlm_lon_v6),
            "ego": ego,
            "ego_turning": egost.get("turning"),
            "ego_motion": egost.get("motion"),
            "ego_prompt_mode": v2.get("_ego_prompt_mode"),
            "scene": v2.get("scene") or {},
        })
    for r in rows:
        r["lat_state"] = agreement(r["alp_lat3"], r["vlm_lat3"],
                                   r["ego"]["lat3"])
        r["lon_state"] = agreement(r["alp_lon3"], r["vlm_lon3"],
                                   r["ego"]["lon3"])
        if r["alp_lane"] in ("Turn Left", "Turn Right"):
            r["stratum"] = "D_UNREPRESENTABLE"
        elif not r["alp_lane"] or not r["alp_lat_raw"]:
            r["stratum"] = "E_STOP_ROW_ONE_AXIS"
        elif "three_way_split" in (r["lat_state"], r["lon_state"]):
            r["stratum"] = "C_THREE_WAY_SPLIT"
        elif "two_of_three" in (r["lat_state"], r["lon_state"]):
            r["stratum"] = "B_TWO_OF_THREE"
        elif "unanimous" in (r["lat_state"], r["lon_state"]):
            r["stratum"] = "A_UNANIMOUS"
        else:
            r["stratum"] = "F_INSUFFICIENT"

    # ---- stratified, DETERMINISTIC selection (sorted, no RNG) ----------
    quota = {"D_UNREPRESENTABLE": 6, "E_STOP_ROW_ONE_AXIS": 4,
             "C_THREE_WAY_SPLIT": 12, "B_TWO_OF_THREE": 18,
             "A_UNANIMOUS": 12, "F_INSUFFICIENT": 2}
    by_s: dict[str, list] = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: r["clip_id"]):
        by_s[r["stratum"]].append(r)
    sel: list = []
    for s in ("D_UNREPRESENTABLE", "E_STOP_ROW_ONE_AXIS", "C_THREE_WAY_SPLIT",
              "B_TWO_OF_THREE", "A_UNANIMOUS", "F_INSUFFICIENT"):
        sel.extend(by_s[s][:quota[s]])
    sel = sel[:a.max_clips]

    # ---- render --------------------------------------------------------
    cards, ok, failed = [], 0, []
    for r in sel:
        frames = frames_from_mp4(r["mp4"], FRAME_IDXS)
        if not frames:
            failed.append(r["clip_id"])
            continue                      # ⛔ never emit a card without frames
        poses = np.load(r["npz"])["poses"]
        cards.append(render_card(r, frames, bev_png(poses)))
        ok += 1
        print(f"[tacrev] {ok}/{len(sel)} {r['clip_id']}", flush=True)

    counts = collections.Counter(x["stratum"] for x in sel)
    pool = collections.Counter(x["stratum"] for x in rows)
    hdr = header_html(len(rows), len(sel), ok, counts, pool, failed)
    doc = HTML_SHELL.replace("__BODY__", hdr + "\n".join(cards))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(doc)
    mb = os.path.getsize(a.out) / 1e6
    assert mb < 40, f"over the 40 MB budget ({mb:.1f})"
    meta = {
        "_evidence_class": "MEASURED (ours) for every leg value; the PI "
                           "verdicts this sheet collects are the first HUMAN "
                           "adjudication of any tactical label.",
        "horizon_s": H_S, "t0_idx": T0_IDX, "fps": FPS,
        "dv_thresh_ms": DV_THRESH, "dyaw_thresh_rad": DYAW_THRESH,
        "frame_idxs": FRAME_IDXS,
        "n_candidate_pool": len(rows),
        "_pool_bound": "the 201 aug120 clips are ALL that have our w120 "
                       "video; of those the pool here is the subset with a "
                       "LOCAL mp4. The other 4,528 Alpamayo clips have NO "
                       "video anywhere local or on HF — a visual sheet for "
                       "them is impossible (TACTICAL_LABEL_VALIDATION.md §5.5).",
        "pool_by_stratum": dict(pool),
        "n_selected": len(sel), "n_rendered": ok,
        "selected_by_stratum": dict(counts),
        "render_failures": failed,
        "html_mb": round(mb, 2),
        "clips": [{k: v for k, v in r.items()
                   if k not in ("mp4", "npz")} for r in sel],
    }
    json.dump(meta, open(a.meta, "w", encoding="utf-8"), indent=1)
    print(f"TACREV_DONE rendered={ok} strata={dict(counts)} {mb:.2f} MB "
          f"-> {a.out}")
    return 0


# --------------------------------------------------------------------------- #
# rendering                                                                     #
# --------------------------------------------------------------------------- #
def _leg(name: str, native: str, v6: str | None, band: str | None,
         warn: str = "") -> str:
    e = html.escape
    tok = (f'<span class="tok">{e(v6)}</span>' if v6 else
           '<span class="tok none">— no token</span>')
    return (f'<div class="leg{" dep" if warn else ""}">'
            f'<div class="lname">{e(name)}{f" <sup>{e(warn)}</sup>" if warn else ""}</div>'
            f'<div class="lnat">{e(native or "silent")}</div>'
            f'{tok}<div class="lband">{e(band or "—")}</div></div>')


def render_card(r: dict, frames: list[bytes], bev: bytes) -> str:
    e = html.escape
    cid = r["clip_id"]
    times = [f"{(i - T0_IDX) / FPS:+.1f}s" for i in FRAME_IDXS]
    imgs = "".join(
        f'<figure class="{"key" if i in (T0_IDX, H_IDX) else ""}">'
        f'<img src="{b64img(b)}" alt="frame {t}">'
        f'<figcaption>{e(t)}{" · t0" if i == T0_IDX else ""}'
        f'{" · READ" if i == H_IDX else ""}</figcaption></figure>'
        for b, i, t in zip(frames, FRAME_IDXS, times))
    ego = r["ego"]
    lat = ('<div class="axis"><h4>LATERAL <span class="st">'
           f'{e(r["lat_state"])}</span></h4><div class="legs">'
           + _leg("Alpamayo", r["alp_lane"] or "—", r["alp_lat_v6"],
                  r["alp_lat3"])
           + _leg("VLM", ", ".join(f'{x["verb"]}/{x["direction"]}'
                                   for x in r["vlm_acts"]) or "—",
                  r["vlm_lat_v6"], r["vlm_lat3"], warn="not independent")
           + _leg("ego @2.0s", f'Δyaw {ego["dyaw"]:+.3f} rad '
                  f'(turning={r["ego_turning"]})', None, ego["lat3"],
                  warn="not independent")
           + "</div></div>")
    lon = ('<div class="axis"><h4>LONGITUDINAL <span class="st">'
           f'{e(r["lon_state"])}</span></h4><div class="legs">'
           + _leg("Alpamayo", r["alp_lon_raw"] or "—", r["alp_lon_v6"],
                  r["alp_lon3"])
           + _leg("VLM", ", ".join(f'{x["verb"]}/{x["direction"]}'
                                   for x in r["vlm_acts"]) or "—",
                  r["vlm_lon_v6"], r["vlm_lon3"], warn="not independent")
           + _leg("ego @2.0s", f'Δv {ego["dv"]:+.2f} m/s '
                  f'({ego["v_t0"]:.1f}→{ego["v_h"]:.1f})', None, ego["lon3"],
                  warn="not independent")
           + "</div></div>")
    notes = r.get("alp_notes") or {}
    gaps = "".join(f'<li><b>{e(k)}</b>: {e(v)}</li>' for k, v in notes.items())
    sc = r["scene"]
    return f"""
<section class="clip" data-clip="{e(cid)}" data-stratum="{e(r['stratum'])}"
         data-lat="{e(r['alp_lane'] or '')}" data-lon="{e(r['alp_lon_raw'] or '')}"
         data-lattok="{e(r['alp_lat_v6'] or '')}"
         data-lontok="{e(r['alp_lon_v6'] or '')}">
  <header><h3>{e(cid)}</h3>
    <span class="badge b{e(r['stratum'][0])}">{e(r['stratum'])}</span>
    <span class="meta">{e(sc.get('illumination','?'))} ·
      {e(sc.get('weather','?'))} · {e(sc.get('road_type','?'))} ·
      v(t0)={ego['v_t0']:.1f} m/s</span></header>
  <div class="strip">{imgs}<figure class="bev">
    <img src="{b64img(bev,'image/png')}" alt="BEV">
    <figcaption>path t0 → +2.0s</figcaption></figure></div>
  <div class="cot"><b>Alpamayo reasoning (<code>cot</code>)</b>
    <q>{e(r['cot'])}</q></div>
  <div class="axes">{lat}{lon}</div>
  {f'<ul class="gaps">{gaps}</ul>' if gaps else ''}
  <div class="verdicts">
    <div class="vgroup"><span class="vlab">LATERAL label correct?</span>
      {"".join(f'<label><input type="radio" name="lat_{cid}" value="{v}">{v}</label>' for v in ("correct","wrong","unsure"))}</div>
    <div class="vgroup"><span class="vlab">LONGITUDINAL label correct?</span>
      {"".join(f'<label><input type="radio" name="lon_{cid}" value="{v}">{v}</label>' for v in ("correct","wrong","unsure"))}</div>
    <input class="note" id="n_{e(cid)}" placeholder="note — which leg is wrong, and why?">
  </div>
</section>"""


def header_html(pool_n, sel_n, ok, counts, pool, failed) -> str:
    e = html.escape
    strat = "".join(
        f"<tr><td><code>{e(k)}</code></td><td>{pool.get(k,0)}</td>"
        f"<td><b>{counts.get(k,0)}</b></td><td>{e(STRAT_WHY[k])}</td></tr>"
        for k in ("D_UNREPRESENTABLE", "E_STOP_ROW_ONE_AXIS",
                  "C_THREE_WAY_SPLIT", "B_TWO_OF_THREE", "A_UNANIMOUS",
                  "F_INSUFFICIENT"))
    fail = (f'<p class="warn">⚠️ {len(failed)} clip(s) dropped for missing '
            f'frames: {e(", ".join(failed))}</p>' if failed else "")
    return f"""
<h1>Tactical label review — LAT and LON, read at the 2.0 s horizon</h1>
<p class="lede"><b>No human has ever reviewed a tactical label.</b> Every
number in <code>TACTICAL_LABEL_VALIDATION.md</code> is machine-vs-machine
agreement; this sheet is the first human adjudication. Judge each axis
<b>separately</b> — a clip can be right laterally and wrong longitudinally,
and one verdict cannot say so.</p>
<div class="box">
<h2>What you are looking at</h2>
<ul>
<li><b>Horizon = 2.0 s.</b> Frames run <code>t0−1.0s → t0+2.0s</code>; the
  <b>t0</b> and <b>READ</b> frames are outlined. MEASURED: κ against Alpamayo
  peaks at 2.0 s on <i>both</i> axes (LON 0.188→<b>0.3655</b>,
  LAT 0.290→<b>0.4694</b>), and 2.0 s is the v6 tactical band. The production
  reading used the full 11.8 s window — the worst region of that surface.</li>
<li><b>Three legs, side by side</b>, each showing its NATIVE output, the v6
  token it maps to, and the 3-band projection used only for grouping.</li>
<li>⛔ <b>The VLM and ego legs are NOT independent</b> (marked
  <sup>not independent</sup>). MEASURED 201/201: the VLM's prompt contains the
  ego <code>motion</code>/<code>turning</code> fields the ego voter reads.
  VLM↔ego κ <b>0.7608</b> vs Alpamayo↔VLM <b>0.1717</b>. If those two agree
  and Alpamayo differs, that is <i>one source agreeing with itself</i> — not
  two votes.</li>
<li><b>Alpamayo is a TEACHER, not ground truth.</b> PhysicalAI-AV is listed as
  its training data; the overlap is UNRESOLVED.</li>
<li>The VLM tokens here are the <b>POST-FIX</b> mapping. The banked corpus
  still holds the pre-fix ones — it has not been re-fused.</li>
</ul>
<p class="warn">⭐ <b>One thing to look for specifically — Alpamayo's
<code>Stop</code> class is a STATE, not an ACTION.</b> MEASURED on the 8
<code>Stop</code> clips with poses: mean <b>v(t0) = 0.51 m/s</b> — the ego is
<i>already stopped</i> — and by the 2 s horizon it has <b>accelerated
+2.44 m/s</b> (0.51 → 2.95). Two of the eight <code>cot</code> strings say
verbatim <i>"Resume speed from stop since the traffic light turns green"</i>:
the label says <code>Stop</code> while its own reasoning says the ego is
<b>launching</b>. ⇒ A naive <code>Stop → BRAKE_TO</code> mapping would be
wrong on most of them. This is why the longitudinal axis is left
<code>REASON_REQUIRED</code> rather than typed by magnitude. <b>Please
adjudicate whether "Stop" is an acceptable label for these frames.</b></p>
<h2>The sample — {sel_n} selected of a {pool_n}-clip pool, {ok} rendered</h2>
<p>⚠️ A sheet of only clean cases would flatter the corpus; a sheet of only
conflicts would libel it. Both are here, labelled.</p>
<table class="strata"><tr><th>stratum</th><th>in pool</th><th>shown</th>
<th>why it is here</th></tr>{strat}</table>
<p class="warn">⚠️ <b>Pool bound:</b> only the <b>201</b> aug120 clips have our
w120 video at all, and this pool is the subset with a local copy. The other
<b>4,528</b> Alpamayo-labelled clips have no video — they cannot be reviewed
visually by any means available today.</p>{fail}
</div>
<div class="exportbar"><button onclick="doExport()">Export verdicts JSON</button>
<span id="prog"></span><textarea id="exportbox"></textarea></div>"""


STRAT_WHY = {
    "D_UNREPRESENTABLE": "Alpamayo says Turn Left/Right — ⛔ v6 has NO such "
                         "lateral action token. 186 clips corpus-wide have "
                         "nowhere to go. Shown so the gap is visible.",
    "E_STOP_ROW_ONE_AXIS": "a STOPPED vehicle emits ONE axis only; the missing "
                           "axis is NOT-APPLICABLE, never a value to impute.",
    "C_THREE_WAY_SPLIT": "all three legs speak and all three differ — the "
                         "hardest cases, and where a label is least trustworthy.",
    "B_TWO_OF_THREE": "exactly two agree. ⚠️ If the two are VLM+ego this is "
                      "the ECHO, not corroboration.",
    "A_UNANIMOUS": "all three agree. Included so the sheet cannot be accused "
                   "of showing only failures — but unanimity among "
                   "non-independent legs is weaker than it looks.",
    "F_INSUFFICIENT": "fewer than two legs spoke — an availability fact, not "
                      "a disagreement.",
}

HTML_SHELL = """<!doctype html><html><head><meta charset="utf-8">
<title>TanitAD — tactical label review (2.0 s horizon)</title>
<style>
:root{--ok:#1a7f37;--bad:#cf222e;--mid:#9a6700;--line:#d0d7de;--ink:#1f2328}
*{box-sizing:border-box}
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;color:var(--ink);
 margin:0;padding:22px;background:#f6f8fa;max-width:1500px}
h1{font-size:23px;margin:0 0 6px} h2{font-size:15px;margin:16px 0 6px}
.lede{font-size:15px;max-width:1000px}
.box{background:#fff;border:1px solid var(--line);border-radius:8px;
 padding:14px 18px;margin:14px 0 22px}
.box ul{margin:6px 0;padding-left:20px} .box li{margin:4px 0}
.warn{color:#7d4e00;background:#fff8e5;border-left:3px solid #d4a72c;
 padding:7px 10px;margin:8px 0}
table.strata{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px}
table.strata th,table.strata td{border:1px solid var(--line);padding:5px 8px;
 text-align:left;vertical-align:top}
table.strata th{background:#f6f8fa}
.exportbar{position:sticky;top:0;z-index:9;background:#f6f8fa;padding:8px 0;
 border-bottom:1px solid var(--line)}
.exportbar button{font-size:14px;padding:7px 14px;cursor:pointer;
 border:1px solid var(--line);border-radius:6px;background:#fff}
#exportbox{display:none;width:100%;height:170px;margin-top:8px;
 font-family:ui-monospace,Consolas,monospace;font-size:11px}
#prog{margin-left:12px;color:#57606a}
section.clip{background:#fff;border:1px solid var(--line);border-radius:8px;
 padding:14px;margin:0 0 20px}
section.clip header{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
section.clip h3{margin:0;font:600 13px ui-monospace,Consolas,monospace}
.badge{font-size:11px;padding:2px 7px;border-radius:10px;background:#eaeef2}
.badge.bD{background:#ffe9e9;color:#8b1a1a}.badge.bC{background:#fff1e0;color:#8a4b00}
.badge.bB{background:#fff8e5;color:#7d4e00}.badge.bA{background:#e7f5ec;color:#0f5132}
.meta{font-size:12px;color:#57606a}
.strip{display:flex;gap:6px;margin:10px 0;overflow-x:auto;padding-bottom:4px}
.strip figure{margin:0;flex:0 0 auto}
.strip img{height:132px;display:block;border-radius:4px;border:2px solid transparent}
.strip figure.key img{border-color:#1f6feb}
.strip figcaption{font-size:10.5px;color:#57606a;text-align:center;margin-top:2px}
.strip figure.bev img{height:132px;border:1px solid var(--line);background:#fff}
.cot{background:#f6f8fa;border-left:3px solid #1f6feb;padding:7px 11px;
 margin:6px 0;font-size:13px}
.cot q{display:block;margin-top:2px;font-style:italic}
.axes{display:flex;gap:14px;flex-wrap:wrap;margin-top:10px}
.axis{flex:1 1 400px;border:1px solid var(--line);border-radius:6px;padding:9px}
.axis h4{margin:0 0 7px;font-size:12.5px;letter-spacing:.04em}
.st{font-weight:400;color:#57606a;font-size:11.5px;margin-left:6px}
.legs{display:flex;gap:7px}
.leg{flex:1;background:#fafbfc;border:1px solid var(--line);border-radius:5px;
 padding:6px 8px;font-size:12px}
.leg.dep{border-style:dashed;border-color:#d4a72c}
.lname{font-weight:600;font-size:11px;text-transform:uppercase;color:#57606a}
.lname sup{font-weight:400;text-transform:none;color:#9a6700;font-size:9px}
.lnat{margin:2px 0;font-size:11.5px}
.tok{display:inline-block;font:600 11px ui-monospace,Consolas,monospace;
 background:#ddf4ff;padding:1px 5px;border-radius:3px}
.tok.none{background:#ffebe9;color:#8b1a1a;font-weight:400}
.lband{font-size:10.5px;color:#57606a;margin-top:3px}
.gaps{margin:8px 0 0;padding-left:20px;font-size:12px;color:#7d4e00}
.verdicts{display:flex;gap:16px;align-items:center;flex-wrap:wrap;
 margin-top:11px;padding-top:10px;border-top:1px dashed var(--line)}
.vlab{font-size:11.5px;font-weight:600;color:#57606a;margin-right:6px}
.vgroup label{margin-right:8px;font-size:12.5px;cursor:pointer;
 border:1px solid var(--line);border-radius:5px;padding:3px 8px}
.vgroup label:has(input[value=correct]:checked){background:#e7f5ec;border-color:var(--ok)}
.vgroup label:has(input[value=wrong]:checked){background:#ffebe9;border-color:var(--bad)}
.vgroup label:has(input[value=unsure]:checked){background:#fff8e5;border-color:var(--mid)}
.note{flex:1 1 300px;padding:5px 8px;border:1px solid var(--line);border-radius:5px}
</style></head><body>__BODY__
<script>
const KEY='tacrev_v1_';
function save(cid){
 const g=n=>{const el=document.querySelector(`input[name="${n}_${cid}"]:checked`);
  return el?el.value:null;};
 localStorage.setItem(KEY+cid,JSON.stringify({lat:g('lat'),lon:g('lon'),
  note:(document.getElementById('n_'+cid)||{}).value||''}));
 prog();}
function prog(){let d=0;const all=document.querySelectorAll('.clip[data-clip]');
 all.forEach(el=>{const s=localStorage.getItem(KEY+el.dataset.clip);
  if(s){const o=JSON.parse(s);if(o.lat||o.lon)d++;}});
 document.getElementById('prog').textContent=d+' / '+all.length+' clips judged';}
document.querySelectorAll('.clip[data-clip]').forEach(el=>{
 const cid=el.dataset.clip;
 const s=localStorage.getItem(KEY+cid);
 if(s){const o=JSON.parse(s);
  ['lat','lon'].forEach(ax=>{if(o[ax]){const r=el.querySelector(
   `input[name="${ax}_${cid}"][value="${o[ax]}"]`);if(r)r.checked=true;}});
  const n=document.getElementById('n_'+cid);if(n&&o.note)n.value=o.note;}
 el.querySelectorAll('input').forEach(i=>{
  i.addEventListener('change',()=>save(cid));
  i.addEventListener('input',()=>save(cid));});});
function doExport(){
 const out={_sheet:'TanitAD tactical label review (LAT/LON @ 2.0 s horizon)',
  _horizon_s:2.0,_exported:new Date().toISOString(),
  _note:'lat and lon are judged SEPARATELY; null = UNREVIEWED, not agreement',
  verdicts:{}};
 document.querySelectorAll('.clip[data-clip]').forEach(el=>{
  const s=localStorage.getItem(KEY+el.dataset.clip);
  out.verdicts[el.dataset.clip]=Object.assign(
   {stratum:el.dataset.stratum,
    alp_lane_raw:el.dataset.lat||null,alp_lon_raw:el.dataset.lon||null,
    alp_lat_v6:el.dataset.lattok||null,alp_lon_v6:el.dataset.lontok||null},
   s?JSON.parse(s):{lat:null,lon:null,note:''});});
 const box=document.getElementById('exportbox');
 box.style.display='block';box.value=JSON.stringify(out,null,1);
 box.select();try{document.execCommand('copy');}catch(e){}}
prog();
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
