"""The failure gallery: pick the worst scenes, build the render job, run it in the NavSim venv, embed it.

Split of work (the brief's "design the renderer so the map-dependent part runs there"):
* HERE (TanitAD venv): which scenes, which plans, the decoded manoeuvre (the programme's OWN canonical
  labeller — ``four_families.maneuver_kinematics`` + ``refc_tactical.factor_from_kinematics``, never a
  second gate), the failing sub-metrics, the overlay text, and the check that the PNGs really exist;
* THERE (NavSim venv, py3.9): the nuPlan map, the agent boxes and the camera frame — ``navsim_gallery.py``.

SELECTION RULE (printed in the report, so it can be argued with): stage-2 scenes the arm answered
ITSELF, ranked by (1) the arm's per-scene ``score`` ascending, (2) the arm's deficit against the BEST
floor on that scene ascending — the worst failures where standing still or constant velocity was fine —
(3) the token, for determinism; with at most ``max_per_log`` scenes per log so one bad log cannot fill
the gallery.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

from .charts import fmt
from .metrics import EPDMS_V2, failing_submetrics
from .palette import GALLERY
from .svg import esc

SELECTION_RULE = ("stage-2 scenes the arm answered itself, ranked by the arm's per-scene `score` "
                  "ascending, then by its deficit against the best floor on that scene, then by token; "
                  "at most {max_per_log} per log")


def _stage_view(row: dict, stage: int) -> dict:
    suf = "_stage_one" if stage == 1 else "_stage_two"
    out = {k: row.get(c + suf) for k, c in EPDMS_V2.columns.items()}
    out["score"] = row.get("score")
    return out


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def _decode_manoeuvre(poses, dt: float) -> str:
    """The programme's canonical factored labeller — NOT a second manoeuvre gate."""
    try:
        import torch
        from taniteval.four_families import maneuver_kinematics
        from tanitad.refs.refc_tactical import LAT_CLASSES, LON_CLASSES, factor_from_kinematics
    except Exception as e:                                          # noqa: BLE001
        return f"UNAVAILABLE ({type(e).__name__}: {str(e)[:60]}) — the canonical labeller is not importable"
    wp = torch.tensor([[[float(p[0]), float(p[1])] for p in poses]], dtype=torch.float32)
    dyaw, dv, v0, v1, _prov = maneuver_kinematics(wp, dt)
    lat, lon = factor_from_kinematics(dyaw, dv, v0, v1)
    return (f"{LAT_CLASSES[int(lat[0])]} · {LON_CLASSES[int(lon[0])]} "
            f"(Δyaw {float(dyaw[0]):+.3f} rad, Δv {float(dv[0]):+.2f} m/s over the plan)")


def select_scenes(run, n: int = 8, max_per_log: int = 2) -> dict:
    """-> {status, scenes: [...], reason?} using the run's own per-token CSVs and scenes.json."""
    from .adapt import read_devkit_csv
    if run.scenes is None:
        return {"status": "UNAVAILABLE", "reason": "scenes.json is absent from the run directory — the "
                                                   "gallery needs token -> scene pickle / frame bank / map"}
    arm = run.primary_arm
    csv = run.file((run.arm(arm).get("files") or {}).get("scores"))
    if csv is None:
        return {"status": "UNAVAILABLE", "reason": f"no per-token CSV for the primary arm {arm}"}
    rows = {a: read_devkit_csv(run.file((run.arm(a).get("files") or {}).get("scores")))[0]
            for a in run.arm_names if run.file((run.arm(a).get("files") or {}).get("scores"))}
    plans = {}
    for a in run.arm_names:
        p = run.file((run.arm(a).get("files") or {}).get("plans"))
        if p is not None:
            import numpy as np
            z = np.load(p, allow_pickle=False)
            plans[a] = {"token": [str(t) for t in z["token"]], "poses": z["poses"],
                        "source": [str(s) for s in z["source"]],
                        "sampling": [float(x) for x in z["sampling"]]}
    if arm not in plans:
        return {"status": "UNAVAILABLE", "reason": f"no plans/{arm}.npz in the run directory (W1: emit the "
                                                   f"per-arm plan poses to enable the gallery)"}
    tok_meta = run.scenes["tokens"]
    own = {t: i for i, (t, s) in enumerate(zip(plans[arm]["token"], plans[arm]["source"]))
           if s not in ("cv_standin",) and tok_meta.get(t, {}).get("stage") == 2}
    floors = [f for f in run.floors if f in rows]
    cands = []
    for t in own:
        sc = rows[arm].get(t, {}).get("score")
        if not _finite(sc):
            continue
        fl = [rows[f][t]["score"] for f in floors if _finite(rows[f].get(t, {}).get("score"))]
        best = max(fl) if fl else None
        cands.append((sc, (sc - best) if best is not None else 0.0, t))
    cands.sort(key=lambda c: (c[0], c[1], c[2]))
    picked, per_log = [], {}
    for sc, deficit, t in cands:
        lg = tok_meta[t]["log"]
        if per_log.get(lg, 0) >= max_per_log:
            continue
        per_log[lg] = per_log.get(lg, 0) + 1
        picked.append(t)
        if len(picked) >= n:
            break
    return {"status": "OK", "tokens": picked, "rows": rows, "plans": plans, "arm": arm, "floors": floors,
            "rule": SELECTION_RULE.format(max_per_log=max_per_log), "n_candidates": len(cands)}


def build_job(run, sel: dict, out_dir: Path) -> dict:
    tok_meta = run.scenes["tokens"]
    bank = Path(run.scenes.get("frame_bank", ""))
    pickles = Path(run.scenes.get("synthetic_scene_pickles", ""))
    arm, rows, plans = sel["arm"], sel["rows"], sel["plans"]
    dt = plans[arm]["sampling"][1] if len(plans[arm]["sampling"]) > 1 else 0.5
    series_order = [arm] + [f for f in ("CV", "STOP", "ECHO") if f in plans and f != arm]
    colors = {arm: GALLERY["plan"], "CV": GALLERY["cv"], "STOP": GALLERY["stop"], "ECHO": "#4a3aa7"}
    scenes, skipped = [], []
    for rank, t in enumerate(sel["tokens"], 1):
        meta = tok_meta[t]
        pkl = pickles / (meta.get("pickle") or f"{meta.get('scene_token')}.pkl")
        npy = bank / "frames" / f"{meta.get('scene_token')}.npy"
        if not pkl.is_file() or not npy.is_file():
            skipped.append({"token": t, "reason": f"missing input: {'pickle ' + str(pkl) if not pkl.is_file() else ''}"
                                                  f"{' frames ' + str(npy) if not npy.is_file() else ''}"})
            continue
        series = []
        for a in series_order:
            idx = plans[a]["token"].index(t) if t in plans[a]["token"] else None
            if idx is None:
                continue
            p = plans[a]["poses"][idx]
            if not all(_finite(float(x)) for x in p.reshape(-1)):
                continue
            moved = float(max(abs(p[:, 0]).max(), abs(p[:, 1]).max()))
            series.append({"name": a, "label": run.label(a) + (" — does not move" if moved < 0.05 else ""),
                           "poses": [[float(x) for x in row] for row in p] if moved >= 0.05 else [],
                           "color": colors.get(a, "#898781"), "width": 2.4 if a == arm else 2.0,
                           "style": "-" if a == arm else ("--" if a == "CV" else ":"),
                           "marker": "o" if a == arm else "."})
        r2 = _stage_view(rows[arm][t], 2)
        fails = failing_submetrics(r2, EPDMS_V2)
        fail_txt = " · ".join(f"{k}={fmt(v, 'f2') if k not in EPDMS_V2.multipliers else fmt(v, 'f1')}"
                              f"{' (zero ⇒ scene = 0)' if kind == 'zero-multiplier' else ''}"
                              for k, v, kind in fails) or "none — every sub-metric at 1.0"
        others = " | ".join(f"{run.label(a).split(' · ')[0]} {fmt(rows[a][t]['score'], 'f4')}"
                            for a in ([arm] + sel["floors"])
                            if _finite(rows.get(a, {}).get(t, {}).get("score")))
        pa = plans[arm]["poses"][plans[arm]["token"].index(t)]
        travel = float((((pa[1:, :2] - pa[:-1, :2]) ** 2).sum(-1) ** 0.5).sum() + (pa[0, 0] ** 2 + pa[0, 1] ** 2) ** 0.5)
        cmd = meta.get("command", "?")
        fed = "yes" if any("driving_command" in d for d in (run.arm(arm).get("declared_inputs") or [])) else \
            "no — withheld from this arm"
        overlay = [
            f"log {meta.get('log')} · stage {meta.get('stage')} ({meta.get('frame_type')}) · map {meta.get('map_name')}",
            f"v0 {meta.get('v0', float('nan')):.2f} m/s · NavSim command at t0: {cmd} · fed to this arm: {fed}",
            f"decoded manoeuvre of the plan: {_decode_manoeuvre(pa, dt)}",
            f"plan travels {travel:.1f} m in {dt * len(pa):.1f} s",
            f"score  {others}",
            f"failing sub-metrics ({run.label(arm).split(' · ')[0]}): {fail_txt}",
        ]
        notes = ["human: n/a — a synthetic stage-2 scene carries no logged future (num_future_frames = 0)"
                 if meta.get("num_future_frames") in (0, None) else "",
                 "a plan that does not move is drawn as a square at the ego origin",
                 f"sub-metrics read from scores/{arm}.csv (stage-2 columns), token {t}"]
        scenes.append({"token": t, "scene_token": meta.get("scene_token"), "pickle": str(pkl),
                       "map_name": meta.get("map_name"), "frames_npy": str(npy), "frame_index": None,
                       "title": f"worst #{rank} · token {t} · {run.label(arm)} scored "
                                f"{fmt(rows[arm][t]['score'], 'f4')}",
                       "overlay": overlay, "notes": [n for n in notes if n], "series": series})
    for s in scenes:
        s.pop("frame_index")
    job = {"out_dir": str(out_dir), "maps_root": run.scenes.get("maps_root", "C:/Users/Admin/navsim-crun/data/maps"),
           "devkit": "", "frame": {"height": 256, "width": 640, "f_ref": 305.5774907364391},
           "style": {"plan": GALLERY["plan"], "ink": GALLERY["ink"], "ink2": GALLERY["ink2"],
                     "surface": GALLERY["surface"], "agent_fill": "#cfcfc9", "ego_fill": "#e8e8e4"},
           "scenes": scenes}
    return {"job": job, "skipped": skipped, "dt": dt}


def build_gallery(run_dir, *, n: int = 8, max_per_log: int = 2, navsim_python: str, devkit: str,
                  quiet: bool = False) -> dict:
    from .contract import load_run
    run = load_run(run_dir)
    out_dir = Path(run_dir) / "report" / "fig" / "gallery"
    sel = select_scenes(run, n=n, max_per_log=max_per_log)
    if sel["status"] != "OK":
        return {"status": "UNAVAILABLE", "reason": sel["reason"]}
    if not sel["tokens"]:
        return {"status": "UNAVAILABLE", "reason": "no stage-2 scene of the primary arm to show"}
    built = build_job(run, sel, out_dir)
    job = built["job"]
    job["devkit"] = devkit
    if not job["scenes"]:
        return {"status": "UNAVAILABLE", "reason": "no selected scene has both its scene pickle and its frame "
                                                   f"bank entry: {built['skipped'][:2]}"}
    py = Path(navsim_python)
    if not py.is_file():
        return {"status": "UNAVAILABLE", "reason": f"the NavSim venv python is not at {py} — the map/BEV half "
                                                   f"of the gallery runs there (py3.9 + devkit + nuPlan maps)",
                "selection_rule": sel["rule"]}
    out_dir.mkdir(parents=True, exist_ok=True)
    job_path = out_dir / "job.json"
    job_path.write_text(json.dumps(job, indent=1), encoding="utf-8")
    script = Path(__file__).with_name("navsim_gallery.py")
    cmd = [str(py), str(script), "--job", str(job_path)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return {"status": "FAILED", "reason": "the NavSim-venv renderer timed out after 1800 s",
                "selection_rule": sel["rule"]}
    (out_dir / "render.log").write_text((p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or ""), encoding="utf-8")
    man_p = out_dir / "manifest.json"
    if not man_p.is_file():
        tail = ((p.stderr or p.stdout or "").strip().splitlines() or ["no output"])[-1][:300]
        return {"status": "FAILED", "reason": f"the NavSim-venv renderer wrote no manifest (rc {p.returncode}): {tail}",
                "selection_rule": sel["rule"], "log": str(out_dir / "render.log")}
    man = json.loads(man_p.read_text(encoding="utf-8"))
    ok = [s for s in man["scenes"] if s.get("ok") and Path(s["png"]).is_file() and Path(s["png"]).stat().st_size > 5000]
    res = {"status": "RENDERED" if ok else "FAILED", "n_rendered": len(ok), "n_requested": len(job["scenes"]),
           "scenes": man["scenes"], "ok_tokens": [s["token"] for s in ok], "manifest": str(man_p),
           "selection_rule": sel["rule"], "n_candidates": sel["n_candidates"], "skipped": built["skipped"],
           "job": str(job_path), "dir": str(out_dir), "renderer": {"python": man.get("python"), "cmd": cmd},
           "overlays": {s["token"]: s for s in job["scenes"]}}
    if not ok:
        res["reason"] = f"the renderer produced no usable PNG (rc {p.returncode}); see {out_dir / 'render.log'}"
    elif len(ok) < len(job["scenes"]):
        res["reason"] = f"{len(job['scenes']) - len(ok)} of {len(job['scenes'])} scenes failed to render"
        res["status"] = "PARTIAL"
    if not quiet:
        print(f"[gallery] {res['status']}: {len(ok)}/{len(job['scenes'])} scenes -> {out_dir}")
    return res


def gallery_html(run, gal: dict) -> tuple:
    """-> (section html, one-line note for the provenance table)."""
    head = ('<h2 id="gallery">7 · Failure gallery — the worst scenes</h2>')
    if gal.get("status") not in ("RENDERED", "PARTIAL"):
        note = f"{gal.get('status')}: {gal.get('reason')}"
        return (head + f'<div class="banner critical"><span class="ic">⛔</span><b>NOT RENDERED</b> — {esc(note)}'
                f'</div><p class="lede">The gallery is a deliverable of this report, not an extra: when it cannot '
                f'be drawn the reason is printed here and the report exits non-zero.</p>', note)
    rule = gal.get("selection_rule", "")
    lead = (f'<p class="lede">{esc(rule)} — {gal.get("n_rendered")} of {gal.get("n_candidates")} candidate scenes. '
            f'Left: the nuPlan map (lanes, drivable area, agent boxes) drawn by the NavSim devkit\'s own '
            f'<code>navsim.visualization</code>, with our plan against the floors. Right: the stitched 3-camera '
            f'256×640 cylindrical frame the model actually saw, with the same plans projected through the frame '
            f'bank\'s own ray model. Below each: command, decoded manoeuvre and the failing sub-metrics.</p>')
    figs = []
    for s in gal["scenes"]:
        if not s.get("ok"):
            figs.append(f'<div class="banner serious"><span class="ic">▲</span> scene <code>{esc(s["token"])}</code> '
                        f'failed to render: {esc(s.get("error") or "unknown")}</div>')
            continue
        ov = (gal.get("overlays") or {}).get(s["token"], {})
        rel = "fig/gallery/" + Path(s["png"]).name
        cap = " · ".join(esc(x) for x in ov.get("overlay", [])[:2])
        figs.append(f'<figure><img src="{esc(rel)}" alt="{esc(ov.get("title", s["token"]))}" loading="lazy">'
                    f'<figcaption><b>{esc(ov.get("title", ""))}</b><br>{cap}</figcaption></figure>')
    skipped = gal.get("skipped") or []
    skip_html = ("".join(f'<li><code>{esc(x["token"])}</code>: {esc(x["reason"])}</li>' for x in skipped))
    return (head + lead + f'<div class="card gallery">{"".join(figs)}'
            + (f'<p class="sub">skipped (inputs missing):</p><ul class="sub">{skip_html}</ul>' if skip_html else "")
            + f'<p class="sub">renderer: <code>{esc(json.dumps(gal.get("renderer", {}).get("cmd", []))[:200])}</code> '
              f'(python {esc(gal.get("renderer", {}).get("python"))}) · manifest '
              f'<code>{esc(Path(gal.get("manifest", "")).name)}</code></p></div>',
            f"{gal.get('status')}: {gal.get('n_rendered')}/{gal.get('n_requested')} scenes")


_ = sys
