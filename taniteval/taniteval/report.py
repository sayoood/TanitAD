"""TanitEval — report generator. Renders the full dashboard HTML from the
results directory (benchmark JSONs, A/B JSONs, window tensors -> BEV SVGs,
profile JSONs). The dashboard is a build artifact of the app."""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/taniteval")
from taniteval import rollout, viz  # noqa: E402
from taniteval.registry import EXTERNAL  # noqa: E402

CSS = """
:root{--bg:#0a0e13;--panel:#111820;--panel2:#0d141b;--ink:#e9edf3;--ink2:#a3afbf;
--ink3:#6b7789;--line:#1e2833;--acc:#2dd4bf;--good:#34d399;--warn:#fbbf24;
--crit:#f87171;--frz:#a78bfa;--mono:ui-monospace,'JetBrains Mono',Menlo,monospace;}
@media(prefers-color-scheme:light){:root{--bg:#f6f7f9;--panel:#fff;--panel2:#f0f2f5;
--ink:#0e141b;--ink2:#475264;--ink3:#7c8798;--line:#dfe3e9;--acc:#0d9488;
--good:#0f9d58;--warn:#c07a08;--crit:#c8442f;--frz:#6d5ae0;}}
:root[data-theme="light"]{--bg:#f6f7f9;--panel:#fff;--panel2:#f0f2f5;--ink:#0e141b;
--ink2:#475264;--ink3:#7c8798;--line:#dfe3e9;--acc:#0d9488;--good:#0f9d58;
--warn:#c07a08;--crit:#c8442f;--frz:#6d5ae0;}
:root[data-theme="dark"]{--bg:#0a0e13;--panel:#111820;--panel2:#0d141b;--ink:#e9edf3;
--ink2:#a3afbf;--ink3:#6b7789;--line:#1e2833;--acc:#2dd4bf;--good:#34d399;
--warn:#fbbf24;--crit:#f87171;--frz:#a78bfa;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 Inter,-apple-system,'Segoe UI',sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:32px 20px 70px}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
h1{font-size:22px;margin:0;letter-spacing:-.02em}h2{font-size:16px;margin:0 0 4px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.13em;
text-transform:uppercase;color:var(--acc);margin:34px 0 10px;display:flex;
align-items:center;gap:10px}.eyebrow::after{content:"";flex:1;height:1px;background:var(--line)}
.sub{color:var(--ink3);font-size:12.5px;font-family:var(--mono)}
.lede{color:var(--ink3);font-size:13px;margin:0 0 14px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
table{width:100%;border-collapse:collapse;font-size:13px}
th{font-family:var(--mono);font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;
color:var(--ink3);text-align:left;padding:11px 13px;border-bottom:1px solid var(--line);
font-weight:500;white-space:nowrap}
td{padding:11px 13px;border-bottom:1px solid var(--panel2);vertical-align:middle}
tr:last-child td{border-bottom:0}tr:hover td{background:var(--panel2)}
td.r,th.r{text-align:right}.big{font-family:var(--mono);font-size:15px;font-weight:600}
.mname{font-weight:620;font-size:13.5px}.meta{font-family:var(--mono);font-size:10.5px;color:var(--ink3)}
.pill{display:inline-block;font-family:var(--mono);font-size:10px;padding:2px 8px;
border-radius:12px;border:1px solid var(--line);color:var(--ink3);white-space:nowrap}
.pill.good{color:var(--good);border-color:color-mix(in srgb,var(--good) 40%,transparent)}
.pill.warn{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 40%,transparent)}
.pill.crit{color:var(--crit);border-color:color-mix(in srgb,var(--crit) 40%,transparent)}
.gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px;
padding:14px}.gal svg{width:100%;height:auto;display:block}
.note{font-size:12.5px;color:var(--ink3);background:var(--panel2);
border:1px solid var(--line);border-left:3px solid var(--warn);
border-radius:0 8px 8px 0;padding:10px 14px;margin-top:12px}
.foot{margin-top:44px;padding-top:16px;border-top:1px solid var(--line);
font-family:var(--mono);font-size:10.5px;color:var(--ink3);line-height:1.8}
.hdr{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;
flex-wrap:wrap;border-bottom:1px solid var(--line);padding-bottom:18px}
.overx{overflow-x:auto}
"""


def _fmt(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def _lb_rows(results):
    rows = []
    order = sorted(results.items(),
                   key=lambda kv: kv[1]["heldout"]["model"]["ade_0_2s"]["mean"])
    for key, r in order:
        hm, hc = r["heldout"]["model"], r["heldout"]["cv"]
        ratio = hm["ade_0_2s"]["mean"] / max(hc["ade_0_2s"]["mean"], 1e-9)
        cls = "good" if ratio < 1 else "warn" if ratio < 3 else "crit"
        m = r.get("model", {})
        rows.append(f"""<tr>
<td><div class="mname">{m.get('name', key)}</div>
<div class="meta">{m.get('encoder', '')} · step {r.get('ckpt_step')}</div></td>
<td class="r"><span class="big">{_fmt(hm['ade_0_2s']['mean'])}</span>
<span class="meta">±{_fmt(hm['ade_0_2s']['ci95'])}</span></td>
<td class="r mono">{_fmt(hm['fde@2s']['mean'])}</td>
<td class="r mono">{_fmt(hm['rmse']['mean'])}</td>
<td class="r mono">{_fmt(hm['miss_rate@2m']['mean'], 3)}</td>
<td class="r mono">{_fmt(hm['tms_openloop']['mean'], 3)}</td>
<td class="r"><span class="pill {cls}">{ratio:.1f}× CV</span></td>
<td class="r mono">{r['n_windows']}</td></tr>""")
    return "\n".join(rows)


def _strata_rows(results):
    rows = []
    for key, r in sorted(results.items()):
        for strat, lab in (("by_curvature", "curv"), ("by_speed", "speed")):
            for name, v in r.get(strat, {}).items():
                rows.append(
                    f"<tr><td class='mono'>{key}</td><td class='mono'>{lab}:"
                    f"{name}</td><td class='r mono'>{_fmt(v['model_ade@1s'])}"
                    f"</td><td class='r mono'>{_fmt(v['cv_ade@1s'])}</td>"
                    f"<td class='r mono'>{_fmt(v['model_ade@2s'])}</td>"
                    f"<td class='r mono'>{v['n']}</td></tr>")
    return "\n".join(rows)


def _diag_html(res_dir):
    """Kinematic FLOOR / ego-status + latent CEILING / skill_score per arm
    (P0 FLEET DIRECTIVE item 1). Reads results/diag_<key>.json."""
    rows = []
    for f in sorted(res_dir.glob("diag_*.json")):
        d = json.loads(f.read_text())
        dg = d.get("diagnostic")
        if not dg:
            continue
        m = d.get("model", {})
        model = dg["model_ade_0_2s"]
        floor = dg["kinematic_floor"]["best_of_3_ade_0_2s"]
        skill = model / max(floor, 1e-9)
        scls = "good" if skill < 1 else "warn" if skill <= 3 else "crit"
        ego, lat = dg.get("ego_status_ceiling"), dg.get("latent_ceiling")
        if ego:
            bc = ego["ridge_beats_ctrv"]
            egotxt = (f"{_fmt(ego['held_out_ade_0_2s'])} "
                      f"<span class='pill {'warn' if bc else 'good'}'>"
                      f"{'beats CTRV' if bc else '≤CTRV: keep bar'}</span>")
        else:
            egotxt = "—"
        lattxt = _fmt(lat["held_out_ade_0_2s"]) if lat else "—"
        bs = dg["skill_score"]["by_speed"]

        def sk(k):
            v = bs.get(k, {}).get("skill_vs_floor")
            return f"{v}" if v is not None else "—"
        spd = f"lo {sk('low')} · md {sk('med')} · hi {sk('high')}"
        st = dg["falsifiers"]["skill_on_straights"]
        fcls = "warn" if st.get("near_trivial_competitive") else "good"
        ftxt = ("near-trivial on straights" if st.get("near_trivial_competitive")
                else "clears trivial bar")
        lc = (" · <span class='pill warn'>low-conf</span>"
              if dg.get("low_confidence") else "")
        rows.append(f"""<tr>
<td><div class="mname">{m.get('name', d.get('model'))}</div>
<div class="meta">step {d.get('ckpt_step')} · {dg['n_windows']} win{lc}</div></td>
<td class="r mono">{_fmt(model)}</td>
<td class="r mono">{_fmt(floor)}</td>
<td class="r">{egotxt}</td>
<td class="r mono">{lattxt}</td>
<td class="r"><span class="pill {scls}">{skill:.2f}×</span>
<div class="meta">{spd}</div></td>
<td><span class="pill {fcls}">{ftxt}</span></td></tr>""")
    if not rows:
        return "<tr><td colspan=7 class='meta'>no diagnostic panels yet</td></tr>"
    return "\n".join(rows)


def _plan_html(res_dir):
    """Planning panel: route-from-vision vs the majority-class base rate, and
    behavior (maneuver) decodability vs chance. Reads results/plan_<key>.json."""
    rows = []
    for f in sorted(res_dir.glob("plan_*.json")):
        d = json.loads(f.read_text())
        if d.get("skipped"):
            continue
        m = d.get("model", {})
        s = d.get("strategic", {})
        b = d.get("behavior_decodability", {})
        rskill = s.get("route_skill_vs_chance")
        rcls = "good" if (rskill or 0) > 0.05 else "warn"
        beats = b.get("beats_chance")
        bcls = "good" if beats else "warn"
        rows.append(f"""<tr>
<td><div class="mname">{m.get('name', d.get('model'))}</div>
<div class="meta">step {d.get('ckpt_step')} · {b.get('n_windows', '—')} win</div></td>
<td class="r mono">{_fmt(s.get('route_acc_follow'))}</td>
<td class="r mono">{_fmt(s.get('majority_route_base_rate'))}</td>
<td class="r"><span class="pill {rcls}">{_fmt(rskill) if rskill is not None
    else '—'}</span></td>
<td class="r mono">{_fmt(b.get('maneuver_balanced_accuracy'))}</td>
<td class="r mono">{_fmt(b.get('chance_balacc'))}</td>
<td><span class="pill {bcls}">{'decodable' if beats else '~ chance'}</span>
<div class="meta">{d.get('verdict', '')}</div></td></tr>""")
    if not rows:
        return "<tr><td colspan=7 class='meta'>no planning panels yet</td></tr>"
    return "\n".join(rows)


def _imag_html(res_dir):
    rows = []
    for f in sorted(res_dir.glob("imag_*.json")):
        d = json.loads(f.read_text())
        m = d.get("model", {})
        lf = d.get("latent_fidelity")
        vu, im = d["vision_use_pct"], d["imagination_pct"]
        # semantic colour: more vision/imagination = better (green)
        vcls = "good" if vu >= 8 else "warn" if vu >= 3 else "crit"
        icls = "good" if im >= 8 else "warn" if im >= 3 else "crit"
        a = d["ade"]
        rows.append(f"""<tr>
<td><div class="mname">{m.get('name', d.get('model'))}</div>
<div class="meta">{m.get('encoder', '')} · step {d.get('ckpt_step')}</div></td>
<td class="r"><span class="pill {vcls}">{vu:+.1f}%</span></td>
<td class="r"><span class="pill {icls}">{im:+.1f}%</span></td>
<td class="r mono">{lf['real'] if lf else '—'}
<span class="meta">{'Δ'+format(lf['vision_gain'],'+.3f') if lf else ''}</span></td>
<td class="r mono">{a['A']:.3f} / {a['B']:.3f} / {a['D']:.3f} / {a['E']:.3f}</td>
<td class="meta">{d.get('verdict', '')}</td></tr>""")
    if not rows:
        return "<tr><td colspan=6 class='meta'>no imagination panels yet</td></tr>"
    return "\n".join(rows)


def _hier_html(res_dir):
    """Hierarchy panel (H26): per-seam load-bearing verdict for each 4-brain arm."""
    rows = []
    for f in sorted(res_dir.glob("hier_*.json")):
        d = json.loads(f.read_text())
        if "seam_nav_to_strategic" not in d:      # skipped / non-4-brain arm
            continue
        m = d.get("model", {})
        nav, ctx = d["seam_nav_to_strategic"], d["seam_ctx_to_tactical"]
        itn = d["seam_intent_to_operative"]
        con = d["consistency"]["maneuver_vs_trajectory"]
        h18 = d["h18_grounded_vs_ungrounded"]
        th = d["thesis_read"]["A_conditioning_helps_conditioned_layer"]

        def pill(txt, cls):
            return f"<span class='pill {cls}'>{txt}</span>"
        nav_c, nav_t = (("good", "vision route") if nav["vision_route_beats_majority"]
                        else ("warn", "cmd echo only"))
        ctx_c, ctx_t = (("good", "load-bearing") if ctx["load_bearing"]
                        else ("crit", "decorative"))
        if itn.get("harmful_if_engaged"):
            int_c, int_t = "crit", "harmful if engaged"
        elif itn["load_bearing"]:
            int_c, int_t = "good", "load-bearing"
        else:
            int_c, int_t = "warn", "decorative"
        kap = con["kappa"] or 0
        kcls = "good" if kap >= 0.4 else "warn" if kap >= 0.2 else "crit"
        n_lb = th["n_of_3_seams_beneficial"]
        vcls = "good" if n_lb >= 2 else "crit"
        gd = ctx["goal_latent_cos"]["delta_real_vs_mean"]["mean"]
        rows.append(f"""<tr>
<td><div class="mname">{m.get('name', d.get('model'))}</div>
<div class="meta">step {d.get('ckpt_step')} · {d['n_windows']} win</div></td>
<td>{pill(nav_t, nav_c)}<div class="meta">route {nav['route_acc_follow']}→{nav['route_acc_nav']} vs maj {nav['majority_straight_rate']}</div></td>
<td>{pill(ctx_t, ctx_c)}<div class="meta">goal-cos Δ{gd:+.4f} (man/wp n.s.)</div></td>
<td>{pill(int_t, int_c)}<div class="meta">lat cos {itn['latent_cos']['real']}/{itn['latent_cos']['none']} real/free</div></td>
<td class="r"><span class="pill {kcls}">κ {con['kappa']}</span><div class="meta">agree {con['agreement']['mean']}</div></td>
<td class="r mono">{h18['grounded_op_rollout_ade_2s']} / {h18['ungrounded_tactical_head_ade_2s']}<div class="meta">grnd / head m</div></td>
<td><span class="pill {vcls}">{n_lb}/3 seams</span><div class="meta">{th['verdict'].split('—')[0].strip()}</div></td></tr>""")
    if not rows:
        return "<tr><td colspan=7 class='meta'>no hierarchy panels yet</td></tr>"
    return "\n".join(rows)


def _ab_html(res_dir):
    out = []
    for f in sorted(res_dir.glob("ab_*.json")):
        r = json.loads(f.read_text())
        cls = "good" if r["significant"] else "warn"
        out.append(f"""<tr><td class="mono">{r['a']}</td>
<td class="mono">{r['b']}</td>
<td class="r mono">{_fmt(r['ade_a'])} / {_fmt(r['ade_b'])}</td>
<td class="r mono">{r['win_rate_b']:.1%}</td>
<td class="r mono">{r['mean_delta_m']:+.3f} [{r['delta_ci95'][0]:+.3f},{r['delta_ci95'][1]:+.3f}]</td>
<td class="r"><span class="pill {cls}">{r['verdict']}</span></td></tr>""")
    return "\n".join(out) or "<tr><td colspan=6 class='meta'>no A/B runs</td></tr>"


def _galleries(res_dir, results, k_each=1):
    blocks = []
    for key in sorted(results):
        wp = res_dir / f"windows_{key}.pt"
        if not wp.exists():
            continue
        data = rollout.load_windows(wp)
        svgs = viz.gallery(data, key, k_each=k_each)
        blocks.append(f"<h2 style='margin:16px 0 6px'>{key}</h2>"
                      f"<div class='panel'><div class='gal'>"
                      + "".join(svgs) + "</div></div>")
    return "\n".join(blocks)


def _profile_html(res_dir):
    p = res_dir / "profile.json"
    fp = res_dir / "forward_profile.json"
    rows = []
    prof = json.loads(p.read_text()) if p.exists() else {}
    fwd = json.loads(fp.read_text()) if fp.exists() else {}
    for key, d in prof.items():
        pr = d.get("profile", {})
        f = fwd.get(key, {})
        enc = f.get("encode_trained_vit") or f.get("adapter") or {}
        rows.append(
            f"<tr><td class='mono'>{key}</td>"
            f"<td class='r mono'>{pr.get('trained_params_m', '—')}M</td>"
            f"<td class='r mono'>{pr.get('frozen_encoder_params_m', '—')}M</td>"
            f"<td class='r mono'>{pr.get('total_inference_params_m', '—')}M</td>"
            f"<td class='r mono'>{enc.get('latency_ms', '—')} ms</td>"
            f"<td class='r mono'>{enc.get('peak_vram_mb', '—')} MB</td></tr>")
    frozen = {k: v for k, v in fwd.items() if k.startswith("frozen_")}
    frows = "".join(
        f"<tr><td class='mono'>{k[7:]}</td><td class='r mono'>"
        f"{v.get('latency_ms', '—')} ms</td><td class='r mono'>"
        f"{v.get('throughput_fps', '—')} fps</td><td class='r mono'>"
        f"{v.get('peak_vram_mb', '—')} MB</td></tr>"
        for k, v in frozen.items() if isinstance(v, dict))
    return "\n".join(rows), frows


def _gen_html(res_dir):
    """Genuine-prediction / anticipation panel: does the model beat CTRV MOST on
    high-divergence windows, and does that advantage COLLAPSE without vision?
    Reads results/gen_<key>.json."""
    rows = []
    for f in sorted(res_dir.glob("gen_*.json")):
        d = json.loads(f.read_text())
        if d.get("skipped"):
            continue
        m = d.get("model", {})
        A = d.get("A_ctrv_divergence", {})
        hl = d.get("B_vision_ablation", {}).get("headline", {})
        Dd = d.get("D_latent_decodability", {})
        F = d.get("F_cosmos_counterfactual", {})
        adv = A.get("high_div_advantage_m")
        beat = A.get("high_div_beats_ctrv_frac")
        ve = hl.get("high_div_vision_effect_m")
        ci = hl.get("high_div_vision_effect_ci95") or [None, None]
        vb = hl.get("anticipation_is_vision_based")
        vcls = "good" if vb else "warn"
        vtxt = "vision-based" if vb else "inconclusive"
        latg = Dd.get("latent_minus_ego_r2")
        dcls = "good" if (latg or 0) > 0.03 else "warn"
        road = F.get("road_ahead_vs_periphery_ratio")
        if F.get("skipped"):
            fcls, ftxt = "warn", "n/a (features)"
        else:
            fcls = "good" if F.get("reads_road_ahead") else "crit"
            ftxt = f"{road}×" if road is not None else "—"
        cist = (f"[{ci[0]:+.2f},{ci[1]:+.2f}]" if ci and ci[0] is not None else "")
        rows.append(f"""<tr>
<td><div class="mname">{m.get('name', d.get('model'))}</div>
<div class="meta">step {d.get('ckpt_step')} · {d.get('n_windows', '—')} win</div></td>
<td class="r mono">{_fmt(adv)}<span class="meta"> · {_fmt(beat, 2)} beat</span></td>
<td class="r mono">{_fmt(ve)} <span class="meta">{cist}</span></td>
<td class="r"><span class="pill {vcls}">{vtxt}</span></td>
<td class="r"><span class="pill {dcls}">{_fmt(latg, 3)}</span></td>
<td class="r"><span class="pill {fcls}">{ftxt}</span></td></tr>""")
    if not rows:
        return "<tr><td colspan=6 class='meta'>no generalization panels yet</td></tr>"
    return "\n".join(rows)


def _pathspeed_html(res_dir):
    """Decoupled longitudinal/lateral planning-quality panel (adf2). Reads
    results/pathspeed_<key>.json — the high-speed stratum long-vs-lat split."""
    rows = []
    for f in sorted(res_dir.glob("pathspeed_*.json")):
        d = json.loads(f.read_text())
        if d.get("skipped"):
            continue
        m = d.get("model", {})
        hl = d.get("headline", {})
        fast = d.get("strata", {}).get("fast_top10pct_speed", {}).get("read", {})
        comp = hl.get("high_speed_loss_is", "—")
        lf = hl.get("high_speed_long_frac_of_2s_sqerr")
        ccls = ("crit" if comp == "longitudinal" else
                "good" if comp == "lateral" else "warn")
        mad = fast.get("model_ade_2s_m")
        cad = fast.get("ctrv_ade_2s_m")
        adcls = "crit" if (mad or 0) > (cad or 1e9) else "good"
        lon = fast.get("model_long_rmse_2s_m")
        lat = fast.get("model_lat_rmse_2s_m")
        sb = hl.get("high_speed_speed_bias_mps")
        ovlf = hl.get("overall_long_frac_of_2s_sqerr")
        rows.append(f"""<tr>
<td><div class="mname">{m.get('name', d.get('model'))}</div>
<div class="meta">step {d.get('ckpt_step')} · {d.get('n_windows', '—')} win</div></td>
<td class="r"><span class="pill {ccls}">{comp}</span>
<span class="meta"> · {_fmt(lf, 2)} long</span></td>
<td class="r mono"><span class="pill {adcls}">{_fmt(mad)}</span>
<span class="meta"> vs CTRV {_fmt(cad)}</span></td>
<td class="r mono">{_fmt(lon)} / {_fmt(lat)}</td>
<td class="r mono">{_fmt(sb)}</td>
<td class="r mono">{_fmt(ovlf, 2)}</td></tr>""")
    if not rows:
        return "<tr><td colspan=6 class='meta'>no pathspeed panels yet</td></tr>"
    return "\n".join(rows)


def build(res_dir=Path("/root/taniteval/results"),
          out=Path("/root/taniteval/results/dashboard.html")):
    results = {}
    for f in res_dir.glob("*.json"):
        if f.name.startswith(("ab_", "golden", "profile", "forward_", "run_",
                              "diag_", "plan_", "imag_", "hier_", "gen_",
                              "pathspeed_", "driving_")):
            continue
        d = json.loads(f.read_text())
        if "heldout" in d:
            results[f.stem] = d
    reg = res_dir / "regression_status.txt"
    regtxt = reg.read_text().strip() if reg.exists() else "not run"
    prof_rows, frozen_rows = _profile_html(res_dir)
    try:                                   # efficiency panel (default axis)
        from taniteval import efficiency
        eff_rows = efficiency.panel_rows(res_dir)
    except Exception as _e:                # never let the panel break the report
        eff_rows = (f"<tr><td colspan='10' class='mono'>efficiency panel "
                    f"unavailable: {type(_e).__name__}: {str(_e)[:100]}</td></tr>")
    try:                                   # driving-capability panel (default axis)
        from taniteval import driving
        drv_rows = driving.panel_rows(res_dir)
    except Exception as _e:                # never let the panel break the report
        drv_rows = (f"<tr><td colspan='9' class='mono'>driving panel "
                    f"unavailable: {type(_e).__name__}: {str(_e)[:100]}</td></tr>")
    if not drv_rows:
        drv_rows = ("<tr><td colspan='9' class='meta'>no driving panels yet — "
                    "run <span class='mono'>python -m taniteval.runner "
                    "driving-all</span> (CPU-only, no GPU)</td></tr>")
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    ext = "".join(f"<tr><td>{e['name']}</td><td class='mono'>{e['bench']}</td>"
                  f"<td class='r mono'>{e['metric']} {e['value']}</td>"
                  f"<td class='meta'>{e['src']}</td></tr>" for e in EXTERNAL)
    html = f"""<title>TanitEval</title>
<style>{CSS}</style>
<div class="wrap">
<div class="hdr"><div><h1>TanitEval <span class="pill">v0.2</span></h1>
<div class="sub">AD world-model evaluation · benchmarks · profiling · A/B · regression</div></div>
<div class="sub" style="text-align:right">generated {now}<br>
held-out val 0c5f7dac3b11 · eval pod A40<br>regression: <b>{regtxt}</b></div></div>

<div class="eyebrow">01 · Leaderboard — fresh runs (open-loop, weak claim per 2605.00066)</div>
<div class="panel overx"><table>
<thead><tr><th>model</th><th class="r">ADE 0–2s</th><th class="r">FDE@2s</th>
<th class="r">RMSE</th><th class="r">miss@2m</th><th class="r">TMS</th>
<th class="r">vs CV</th><th class="r">windows</th></tr></thead>
<tbody>{_lb_rows(results)}</tbody></table></div>
<div class="note">All metrics computed fresh by <span class="mono">taniteval.runner</span>
on the same 40 held-out episodes; ±CI95 from 8 overlapping random 20% holdouts
(DEPRECATED estimator — not a jackknife; episode-cluster bootstrap supersedes it).
TMS = hub smoothness metric on the predicted path (→1 smooth).</div>

<div class="eyebrow">01b · Kinematic floor / ego-status ceiling / skill_score (top program risk G1)</div>
<div class="panel overx"><table>
<thead><tr><th>model</th><th class="r">model ADE</th><th class="r">floor (best-of-3)</th>
<th class="r">ego-status ceiling</th><th class="r">latent ceiling</th>
<th class="r">skill vs floor</th><th>straights</th></tr></thead>
<tbody>{_diag_html(res_dir)}</tbody></table></div>
<div class="note"><b>floor</b> = per-window best of {{constant-velocity, go-straight,
CTRV}} — the trivial kinematic predictor a non-trivial model must beat.
<b>ego-status ceiling</b> = held-out ridge on ego kinematics only (AD-MLP repro,
NO perception); if it does not beat hand-coded CTRV there is no learned shortcut —
keep CTRV as the bar (falsifier). <b>latent ceiling</b> = best linear readout of
the world-model latent (the model rollout is <i>action-privileged</i>, ceilings/
floor are not). <b>skill vs floor</b> = model_L2 / floor (＜1 beats floor); the
sub-line is speed-gated (lo/md/hi). Computed by
<span class="mono">taniteval.bench.diagnostic</span> on diag_&lt;arm&gt;.json.</div>

<div class="eyebrow">02 · Imagination panel — does the model USE vision, or just integrate actions?</div>
<div class="panel overx"><table>
<thead><tr><th>model</th><th class="r">vision use</th><th class="r">imagination</th>
<th class="r">latent fidelity</th><th class="r">ADE A/B/D/E</th><th>verdict</th></tr></thead>
<tbody>{_imag_html(res_dir)}</tbody></table></div>
<div class="note">Grounded ADE is measured WITH true future actions, which over-determine
the pose — this panel isolates vision. <b>vision use</b> = ADE cost of removing scene
content (actions kept); <b>imagination</b> = ADE that real vision recovers when actions
are <i>withheld</i>; <b>latent fidelity</b> = cos(imagined latent, true future latent), Δ vs
vision-ablated. Higher = more genuine visual world-modelling. ADE cols: A real+trueActions /
B meanVision+trueActions / D real+noActions / E meanVision+noActions.</div>

<div class="eyebrow">02b · Hierarchy (H26) — is the operative→tactical→strategic conditioning load-bearing?</div>
<div class="panel overx"><table>
<thead><tr><th>model</th><th>nav→strategic</th><th>ctx→tactical</th>
<th>intent→operative</th><th class="r">man~traj</th><th class="r">H18 grnd/head</th>
<th>read</th></tr></thead>
<tbody>{_hier_html(res_dir)}</tbody></table></div>
<div class="note">Cross-layer ablation: replace each FiLM cond (weights fixed) with a
mean/zero control and measure the downstream delta (8 overlapping random holdouts,
DEPRECATED — not a jackknife). A seam is
<b>load-bearing</b> only if the real upstream signal beats its control, CI-separated.
<b>nav→strategic</b> is load-bearing by construction (the command propagates to the route
head); the honest test is whether the follow head beats the majority-straight baseline.
<b>intent→operative</b> is measured on the intent-conditioned JEPA latent (its true regime —
the grounded pose rollout is intent-free by design). <b>man~traj κ</b> = Cohen's kappa
between the tactical maneuver direction and the operative rolled-trajectory heading
(same 2 s timescale). <b>H18</b> = grounded operative-rollout endpoint vs the ungrounded
tactical waypoint head, both vs GT @2 s (m).</div>

<div class="eyebrow">02c · Planning — route-from-vision vs chance · behavior decodability</div>
<div class="panel overx"><table>
<thead><tr><th>model</th><th class="r">route follow</th><th class="r">route base rate</th>
<th class="r">route skill</th><th class="r">maneuver bal-acc</th><th class="r">chance</th>
<th>read</th></tr></thead>
<tbody>{_plan_html(res_dir)}</tbody></table></div>
<div class="note"><b>route follow</b> = route (L/S/R) predicted from VISION (nav=follow)
vs <b>route base rate</b> = the majority-class fraction (chance); <b>route skill</b> =
follow − base (＞0 the model reads route from the scene above always-guess-majority).
<b>maneuver bal-acc</b> = balanced accuracy of a class-weighted linear probe decoding
the maneuver from the latent (eval_behavior instrument; raw accuracy is meaningless
under lane-keep imbalance) vs <b>chance</b> = 1/5. From
<span class="mono">taniteval.planning</span> on plan_&lt;arm&gt;.json.</div>

<div class="eyebrow">02d · Genuine prediction / anticipation — beats CTRV on high-divergence windows, and does it survive vision-ablation?</div>
<div class="panel overx"><table>
<thead><tr><th>model</th><th class="r">hi-div adv vs CTRV</th><th class="r">vision effect [CI95]</th><th class="r">anticipation</th><th class="r">latent−ego R² (scene)</th><th class="r">occlusion road/periph</th></tr></thead>
<tbody>{_gen_html(res_dir)}</tbody></table></div>
<div class="note"><b>hi-div adv vs CTRV</b> = meters the model beats the constant-turn-rate extrapolation by on the top-quartile-divergence windows (an upcoming maneuver the current dynamics do not contain); ·beat = fraction of those windows it beats CTRV. <b>vision effect</b> = advantage(real vision) − advantage(scene mean-replaced), paired cluster-bootstrap CI; >0 CI-separated => the anticipation is READ FROM THE SCENE (the causal clincher). <b>latent−ego R²</b> = upcoming road-curvature decodable from the latent above an ego-kinematics-only probe (scene encoded). <b>occlusion road/periph</b> = predicted-path shift when the road-ahead pixels are occluded vs the periphery, dynamics held fixed (Cosmos-counterfactual proxy). From <span class="mono">taniteval.generalization</span> on gen_&lt;arm&gt;.json.</div>

<div class="eyebrow">02e · Decoupled longitudinal / lateral planning quality — where the high-speed loss lives (TF++/PerlAD path-speed split)</div>
<div class="panel overx"><table>
<thead><tr><th>model</th><th class="r">hi-speed loss is</th>
<th class="r">model ADE@2s (hi-speed)</th><th class="r">long / lat RMSE@2s</th>
<th class="r">speed bias</th><th class="r">overall long-frac</th></tr></thead>
<tbody>{_pathspeed_html(res_dir)}</tbody></table></div>
<div class="note">ADE conflates <b>longitudinal</b> (along-track / planned-speed) and <b>lateral</b>
(cross-track / path-geometry) error. This panel decouples them: residuals are projected onto
the GT path tangent/normal per horizon (frenet), and a separate <b>arc-length-resampled
fixed-distance</b> cross-track isolates path shape from speed (refbpatch idea). <b>hi-speed loss
is</b> = the dominant component on the top-decile-speed windows (long-frac = along-track share
of the 2 s squared-error; ≥0.6 ⇒ longitudinal). <b>speed bias</b> = mean planned − GT speed
(m/s; &lt;0 under-predicts). At the highest speeds trivial CTRV is near-perfect, so a large
along-track RMSE here is the model over-integrating a small speed error over the long horizon.
From <span class="mono">taniteval.pathspeed</span> on pathspeed_&lt;arm&gt;.json.</div>

<div class="eyebrow">03 · A/B — paired per-window, bootstrap 10k</div>
<div class="panel overx"><table>
<thead><tr><th>A</th><th>B</th><th class="r">ADE A / B</th><th class="r">B win-rate</th>
<th class="r">Δ mean [CI95]</th><th class="r">verdict</th></tr></thead>
<tbody>{_ab_html(res_dir)}</tbody></table></div>

<div class="eyebrow">04 · Compute & memory (measured)</div>
<div class="panel overx"><table>
<thead><tr><th>model</th><th class="r">trained</th><th class="r">+frozen</th>
<th class="r">=infer</th><th class="r">enc lat/8</th><th class="r">peak VRAM</th></tr></thead>
<tbody>{prof_rows}</tbody></table></div>
<div class="panel overx" style="margin-top:10px"><table>
<thead><tr><th>frozen encoder</th><th class="r">latency/8</th>
<th class="r">throughput</th><th class="r">peak VRAM</th></tr></thead>
<tbody>{frozen_rows}</tbody></table></div>

<div class="eyebrow">04b · Inference efficiency — one planning step (window → 4 waypoints)</div>
<div class="panel overx"><table>
<thead><tr><th>model</th><th class="r">p50</th><th class="r">p95 / p99</th>
<th>where the budget goes</th><th class="r">GFLOPs</th><th class="r">peak VRAM</th>
<th class="r">params</th><th class="r">ADE@2s</th><th class="r">10 Hz budget</th>
<th class="r">precision</th></tr></thead>
<tbody>{eff_rows}</tbody></table></div>
<div class="note"><b>The deployment axis.</b> Wall-clock for ONE forward planning
step at <b>batch 1</b> (the deployment case), ≥200 warmed iterations, per-iteration
CUDA events, <span class="mono">torch.cuda.synchronize()</span> bracketed, warmup
discarded. Precision is applied <b>identically to every arm and recorded</b> —
letting it drift between arms is the classic way to publish a fake 2× speedup.
Host→device copy and uint8→float are excluded (reported separately as
<span class="mono">input_prep</span>). <b>where the budget goes</b> is the
architectural read: a grounded world-model arm pays for <i>20 sequential</i>
predictor steps, an anchored-diffusion arm pays for a <i>parallel</i> anchor fan
plus its truncated-denoise passes. GFLOPs are profiler-derived
(<span class="mono">FlopCounterMode</span>: conv/matmul/SDPA only — a lower
bound, elementwise and norm work excluded). <b>10 Hz budget</b> = p99 as a share
of 100 ms. Frozen-encoder (REF-A) rows EXCLUDE the external DINOv2/I-JEPA forward
— never compare them to a pixels-in arm unadjusted. From
<span class="mono">taniteval.efficiency</span> on eff_&lt;arm&gt;.json (and inline
in every <span class="mono">results/&lt;arm&gt;.json</span>).</div>

<div class="eyebrow">04c · Driving capability (TanitEval v2, tier 0) — what a single ADE column hides</div>
<div class="panel overx"><table>
<thead><tr><th>model</th><th class="r">ADE 0–2s [CI95]</th>
<th class="r">along / cross @2s</th><th class="r">speed MAE</th>
<th class="r">cruise Δ vs hold-v0</th><th class="r">heading on straights</th>
<th class="r">κ-sign</th><th class="r">tick p50</th>
<th class="r">where the win lives</th></tr></thead>
<tbody>{drv_rows}</tbody></table></div>
<div class="note"><b>ADE is a component, not the verdict.</b> Every cell here is
an <b>episode-cluster bootstrap</b> over the val episodes; every win/tie/LOST tag
is a <b>paired</b> episode-cluster test against a trivial floor, and it is
<b>three-way on purpose</b> — a separated interval that favours the FLOOR means
the trivial baseline beat the model, which a sep/tie rendering would have shown
as a win. The deprecated <span class="mono">overlapping_holdout_se</span>
(1.28–2.06× too narrow) is <i>refused</i> by this block, not merely discouraged.
<b>along / cross</b> = the Frenet split of the 2 s residual on the GT tangent —
flagship v1's entire CI-separated advantage over CV is <i>cross-track</i>
(+0.772 [+0.417, +1.191]) while along-track is <i>not separated</i> (+0.254
[−0.028, +0.530]). <b>speed MAE</b> is measured against <b>hold-v0</b> (go
straight at the observed entry speed) — the strongest trivial longitudinal
floor, and the one VTARGET provably loses to at 2 s. <b>cruise Δ</b> is L1
CRUISE-QUALITY on the longitudinally steady windows: <i>every arm in the
program is separated AGAINST hold-v0 here</i> while winning on brake/accel —
the two point in opposite directions and ADE averages them away.
<b>heading on straights</b> is T3 (CV scores 1.399°). <b>κ-sign</b> = curvature
SIGN agreement; the curvature magnitude is refused at this resolution (measured
24× the signal). <b>tick p50</b> is the same measurement as panel 04b.
Kinematic strata are <b>signatures, not scenarios</b> — no map, no agents, no
scenario ground truth exist, and a 2 s window cannot see a 5–20 s intersection.
Open-loop, weak claim (arXiv:2605.00066). From
<span class="mono">taniteval.driving</span> on driving_&lt;arm&gt;.json (and
inline in every <span class="mono">results/&lt;arm&gt;.json</span>); spec:
<span class="mono">TANITEVAL_V2_METRIC_SUITE.md</span>.</div>

<div class="eyebrow">05 · Strata — where the error lives (ade@1s / @2s)</div>
<div class="panel overx"><table>
<thead><tr><th>model</th><th>stratum</th><th class="r">model@1s</th>
<th class="r">cv@1s</th><th class="r">model@2s</th><th class="r">n</th></tr></thead>
<tbody>{_strata_rows(results)}</tbody></table></div>

<div class="eyebrow">06 · BEV examples — best / median / worst</div>
{_galleries(res_dir, results)}

<div class="eyebrow">07 · External context (closed-loop, separate scale)</div>
<div class="panel overx"><table><thead><tr><th>method</th><th>bench</th>
<th class="r">score</th><th>source</th></tr></thead><tbody>{ext}</tbody></table></div>

<div class="foot">TanitEval v0.2 — package: registry · loaders · data · rollout ·
bench · ab · viz · runner · regression · report<br>
protocol: grounded operative rollout under true actions → step-readout → SE(2);
metrics from the hub suite (tanitad_metrics.py) + gate ADE conventions<br>
open-loop ⊥ closed-loop (arXiv:2605.00066) — closed-loop arbiters (LAL/OKRI/LOPS/CNCE)
pending the CARLA substrate</div></div>"""
    out.write_text(html, encoding="utf-8")
    return str(out)
