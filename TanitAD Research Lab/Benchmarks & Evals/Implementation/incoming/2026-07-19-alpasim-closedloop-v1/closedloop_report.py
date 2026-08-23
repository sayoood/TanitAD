"""Render results/CLOSEDLOOP_REPORT.md from the closedloop_<arm>.json files.

Standalone: python3 closedloop_report.py  (reads /root/taniteval/results).
Headline = flagship-30k (v1); also tabulates flagship-speed + flagship-nospeed."""
import json
from pathlib import Path

RES = Path("/root/taniteval/results")
ARMS = ["flagship-30k", "flagship-speed", "flagship-nospeed"]
HZ = [("0.5s", "de@0.5s", "ade@0.5s"), ("1s", "de@1s", "ade@1s"),
      ("1.5s", "de@1.5s", "ade@1.5s"), ("2s", "de@2s", "ade@2s")]


def load(arm):
    p = RES / f"closedloop_{arm}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def m(d, *ks):
    for k in ks:
        d = d[k]
    return d


def fnum(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main():
    data = {a: load(a) for a in ARMS}
    v1 = data["flagship-30k"]
    L = []
    W = L.append
    W("# Closed-Loop Evaluation — TanitAD Flagship (imagination-in-the-loop)\n")
    W("**Date:** 2026-07-19 · **Pod:** tanitad-eval (A40) · **Corpus:** "
      "PhysicalAI-AV val (in-distribution) · **Renderer:** NONE.\n")
    W("NO-RENDERER closed loop: AlpaSim's NuRec renderer is unrunnable on this pod "
      "(unprivileged container, seccomp blocks user namespaces). With only a front "
      "camera + ego poses (no HD map / agent boxes) the honest closed-loop test is "
      "**imagination-in-the-loop drift + log-relative stability**, NOT a collision / "
      "drivable-area PDM. The flagship world model is used as its OWN neural "
      "simulator.\n")
    W("**Loop (per 0.1 s tick, 2 s horizon):** encode a real window -> latent z0; "
      "then (a) PLAN 2 s ego waypoints via the trained strategic->tactical hierarchy "
      "on the CURRENT (imagined) latent; (b) derive control a_k=(steer,accel) from "
      "the 0.5 s waypoint (pure-pursuit + speed P-controller); (c) IMAGINE "
      "z_{k+1}=operative_predictor(z_k, a_k) — the model consumes its OWN prediction "
      "(intent-free, the deployed regime); (d) DRIVE the ego pose by a kinematic "
      "bicycle under a_k. Re-plan every tick (receding horizon).\n")
    W("Two closed-loop paths are reported: **closed_bike** (kinematic integration of "
      "the executed controls — the HEADLINE) and **closed_grnd** (the operative "
      "predictor's own metric step-readout on the imagined roll — apples-to-apples "
      "with the teacher-forced open-loop). Baselines: **open_grnd** = the gate "
      "grounded rollout under TRUE actions; **open_bike** = bicycle under TRUE "
      "actions (the kinematic-fidelity FLOOR); **cv** = constant velocity.\n")

    if v1 is None:
        W("\n**flagship-30k results missing — run `python3 -m taniteval.closedloop "
          "--arm flagship-30k` first.**\n")
        (RES / "CLOSEDLOOP_REPORT.md").write_text("\n".join(L))
        print("[report] v1 missing; wrote stub")
        return

    s = v1["summary"]
    W(f"\n## Headline — flagship-30k (v1 FINAL, step {v1['ckpt_step']}, "
      f"n={v1['n_windows']} windows / {v1['n_episodes']} episodes)\n")
    W(f"- **Closed-loop ADE@2s = {fnum(s['closed_bike_ade@2s'])} m** "
      f"(±{fnum(s['closed_bike_ade@2s_ci95'])}), "
      f"**FDE@2s = {fnum(s['closed_bike_fde@2s'])} m** (closed_bike, headline).")
    W(f"- Teacher-forced open-loop grounded ADE@2s = "
      f"{fnum(s['open_grnd_ade@2s'])} m; bicycle kinematic floor (true actions) = "
      f"{fnum(s['open_bike_ade@2s_kinematic_floor'])} m; CV baseline = "
      f"{fnum(s['cv_ade@2s'])} m.")
    W(f"- **Compounding delta (closed − open) @2s: grounded = "
      f"+{fnum(s['closed_minus_open_grnd_de@2s'])} m, bicycle = "
      f"+{fnum(s['closed_minus_open_bike_de@2s'])} m** (point error).")
    W(f"- **Divergence rate (closed_bike drift > 5 m @2s) = "
      f"{s['divergence_rate_gt5m@2s']*100:.1f}%.**")
    hl = s["high_vs_low_speed_closed_bike_ade@2s"]
    W(f"- Speed-stratified closed_bike ADE@2s: low={fnum(hl[0])} m, "
      f"high={fnum(hl[1])} m.\n")

    # per-horizon table (v1)
    ho = v1["closedloop_ade_fde"]["heldout"]
    W("### Per-horizon (flagship-30k) — ADE (cumulative) / point-error, metres\n")
    W("| horizon | closed_bike ADE | closed_bike pt | closed_grnd ADE | "
      "open_grnd ADE | open_bike floor | CV |")
    W("|--|--|--|--|--|--|--|")
    for name, de_k, ade_k in HZ:
        W(f"| {name} | {fnum(m(ho,'closed_bike',ade_k,'mean'))} | "
          f"{fnum(m(ho,'closed_bike',de_k,'mean'))} | "
          f"{fnum(m(ho,'closed_grnd',ade_k,'mean'))} | "
          f"{fnum(m(ho,'open_grnd',ade_k,'mean'))} | "
          f"{fnum(m(ho,'open_bike',ade_k,'mean'))} | "
          f"{fnum(m(ho,'cv',ade_k,'mean'))} |")

    # compounding
    cg = v1["compounding_error_grounded"]
    cbk = v1["compounding_error_bicycle"]
    W("\n## Compounding error (closed − open, per-horizon point error ± CI95)\n")
    W("| horizon | grounded Δ (m) | bicycle Δ (m) |")
    W("|--|--|--|")
    for name, _, _ in HZ:
        g = cg[f"delta@{name}"]
        b = cbk[f"delta@{name}"]
        W(f"| {name} | +{fnum(g['mean'])} ±{fnum(g['ci95'])} | "
          f"+{fnum(b['mean'])} ±{fnum(b['ci95'])} |")
    W(f"\n_Grounded Δ caveat: {cg['_caveat']}_\n")

    # stability
    st = v1["stability"]
    cf = st["comfort"]
    ld = st["lateral_deviation_growth_m"]
    W("## Stability / comfort (closed-loop executed controls)\n")
    W(f"- Divergence (>5 m @2s): "
      f"{st['divergence_rate_gt5m@2s']['mean']*100:.1f}% "
      f"(±{st['divergence_rate_gt5m@2s']['ci95']*100:.1f} pts).")
    W(f"- Lateral-deviation growth vs GT: "
      + ", ".join(f"{k}={fnum(v)} m" for k, v in ld.items())
      + " (drift is longitudinal-dominated — the known high-speed weakness).")
    W(f"- Comfort: mean|accel|={fnum(cf['mean_abs_accel_mps2'])} m/s², "
      f"mean|jerk|={fnum(cf['mean_abs_jerk_mps3'])} m/s³, "
      f"mean|lat_accel|={fnum(cf['mean_abs_lat_accel_mps2'])} m/s²; "
      f"{cf['frac_steps_exceed_lon_comfort']*100:.0f}% of steps exceed the "
      f"{cf['bounds']['a_lon']} m/s² longitudinal comfort bound, "
      f"{cf['frac_steps_exceed_jerk_comfort']*100:.0f}% exceed jerk "
      f"(noisy longitudinal command from the tactical head; lateral is smooth).\n")

    # speed strata
    bs = v1["speed_stratified"]["by_speed"]
    thr = v1["speed_stratified"]["thresholds_mps"]
    W(f"## Speed-stratified closed-loop drift (tertiles at {thr} m/s)\n")
    W("| stratum | mean speed | closed_bike ADE@2s | closed−open grnd Δ@2s | "
      "divergence | n |")
    W("|--|--|--|--|--|--|")
    for lab in ("low", "med", "high"):
        if lab not in bs:
            continue
        r = bs[lab]
        W(f"| {lab} | {fnum(r['mean_speed_mps'])} | "
          f"{fnum(r['closed_bike_ade@2s'])} | "
          f"+{fnum(r['closed_minus_open_grnd_ade@2s'])} | "
          f"{r['divergence_rate_gt5m@2s']*100:.1f}% | {r['n']} |")

    # arm comparison
    W("\n## Arm comparison — does the speed channel help closed-loop stability?\n")
    W("| arm | ckpt | speed-in | closed_bike ADE@2s | FDE@2s | closed−open grnd "
      "Δ@2s | divergence | open_grnd (ref) |")
    W("|--|--|--|--|--|--|--|--|")
    for a in ARMS:
        d = data[a]
        if d is None:
            W(f"| {a} | — | — | (not run) | | | | |")
            continue
        if d.get("skipped"):
            W(f"| {a} | — | — | skipped: {d['skipped'][:40]} | | | | |")
            continue
        ss = d["summary"]
        W(f"| {a} | {d['ckpt_step']} | "
          f"{'yes' if d['model'].get('speed_input') else 'NO'} | "
          f"{fnum(ss['closed_bike_ade@2s'])} ±{fnum(ss['closed_bike_ade@2s_ci95'])} | "
          f"{fnum(ss['closed_bike_fde@2s'])} | "
          f"+{fnum(ss['closed_minus_open_grnd_de@2s'])} | "
          f"{ss['divergence_rate_gt5m@2s']*100:.1f}% | "
          f"{fnum(ss['open_grnd_ade@2s'])} |")
    W("\n_REF-B is a direct planner (no operative latent predictor + metric "
      "step-readout), so it is architecturally incompatible with this "
      "imagination-in-the-loop harness and is not tabulated._\n")

    # caveats
    W("## Honest caveats\n")
    for c in v1["limitations"]:
        W(f"- {c}")
    W("\n**Re-run:** `python3 -m taniteval.closedloop --arm flagship-30k "
      "[--episodes 40]` (or `--all-flagships`); then `python3 closedloop_report.py`.")

    (RES / "CLOSEDLOOP_REPORT.md").write_text("\n".join(L))
    print(f"[report] wrote {RES/'CLOSEDLOOP_REPORT.md'} "
          f"({len(L)} lines; arms: "
          f"{[a for a in ARMS if data[a] and not data[a].get('skipped')]})")


if __name__ == "__main__":
    main()
