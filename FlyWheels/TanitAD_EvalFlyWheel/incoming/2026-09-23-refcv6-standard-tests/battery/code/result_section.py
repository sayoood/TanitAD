"""Digest of ONE battery tag for RESULT.md, both inference seeds side by side. Every number is copied
from a JSON the runner / post_tag wrote (nothing recomputed). Usage: python result_section.py <tag_dir>"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_battery as SB  # noqa: E402

KEY_PAIRS = [("os_minus_ha0ext", "os − ha0_ext (echo)"), ("os_minus_ha", "os − ha (hold)"),
             ("os_minus_ha0__CV", "os − ha0 (CV)"), ("os_minus_refcv4b", "os − refcv4b"),
             ("os_minus_refcv5v2_s0", "os − refcv5-v2 s0"), ("navzero_minus_os", "nav withheld − os"),
             ("navshuf_minus_os", "nav shuffled − os"), ("vmaxzero_minus_os", "max-speed withheld − os")]
COLS = [("ADE", "ade_m"), ("longitudinal", "LON_speed_mae_mps"), ("longitudinal", "LON_along_mae_m"),
        ("lateral", "LAT_heading_mae_deg"), ("lateral", "LAT_yaw_rate_mae_radps_valid"),
        ("lateral", "LAT_cross_mae_m"), ("tactical", "TAC_traj_lat_correct"),
        ("tactical", "TAC_traj_lon_correct")]
DIGEST_ARMS = ["os", "os_navzero", "os_vmaxzero", "b_refcv4b", "b_refcv5v2_s0", "ha", "ha0", "ha0_ext",
               "oracle_sel"]


def J(p: Path):
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def cell(x):
    if not x or "delta" not in x:
        return "—"
    return f"{x['delta']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]{' **sep**' if x['separated'] else ''}"


def paired_digest(cp: dict) -> list:
    out = ["| cell (b − a) | ADE m | speed MAE | along MAE | heading ° | yaw-rate rad/s (A3) | cross-track | "
           "traj lat correct | traj lon correct |", "|---|---|---|---|---|---|---|---|---|"]
    for nm, lab in KEY_PAIRS:
        fam = ((cp["pairs"].get(nm) or {}).get("families")) or {}
        out.append(f"| {lab} | " + " | ".join(cell((fam.get(f) or {}).get(m)) for f, m in COLS) + " |")
    out.append(f"\nn = {cp['n_windows']} windows / {cp['n_episodes']} episodes; paired episode-cluster bootstrap "
               f"(n_boot 2000, seed 0, cluster = clip); **sep** = CI excludes 0")
    return out


def main():
    d = Path(sys.argv[1])
    s = J(d / "battery_summary.json")
    seeds = [k for k in (0, 1) if (d / f"analysis_s{k}.json").exists()]
    g0 = (s.get("stages") or {}).get("g0") or {}
    L = [f"### {s['tag']}: step {s['step']}, ckpt md5 `{s['ckpt_md5']}`, SPEC sha at run `{str(s['spec_sha256'])[:12]}…`",
         "",
         "**Tier and estimator.** T1 = self-action open loop; T0 = `oracle_sel` only. Paired episode-cluster "
         "bootstrap. One training seed. Every separated cell answers only the EPISODE question; the INFERENCE question "
         "is answered by the two-seed replicate below.",
         "",
         "⛔ **Run-defect stamp:** F3 detach-only, F4 on the last layer only; tactical labels ~0.37 s early. No "
         "weakness here is attributed to the registered design.",
         "",
         f"**Gate.** G0 as registered **{g0.get('G0_as_registered')}**, G0-A1 **{g0.get('G0_A1')}**, "
         f"G0-A2 (operative) **{g0.get('G0_A2')}**.",
         f"* A2 max wrapper rel: {g0.get('A2_max_wrapper_rel')}.",
         f"* Reasons: A1 {g0.get('G0_A1_reasons')}; A2 {g0.get('G0_A2_reasons')}.",
         ""]
    bars = s.get("bars") or []
    if bars:
        L += ["| bar | statement | seed 0 | seed 1 | verdict |", "|---|---|---|---|---|"]
        for b in bars:
            per = b.get("per_inference_seed") or {}
            c = lambda k: (cell(per.get(k) or per.get(str(k))) if (per.get(k) or per.get(str(k))) else "—")  # noqa: E731
            L.append(f"| {b['id']} | {b['statement']} | {c(0)} | {c(1)} | **{b['verdict']}** |")
        b1 = next((b for b in bars if b["id"] == "BAR-R6-1"), None)
        if b1:
            v = b1["verdict"]
            L += ["", f"⭐ **Primary bar BAR-R6-1 = {v}.** At this checkpoint refcv6 is "
                  + ("**PROVEN** to beat the echo control at 0–2 s at both inference seeds."
                     if v == "PASS" else "**NOT PROVEN** to beat the echo control at 0–2 s." if v == "FAIL"
                     else f"not evaluable ({v}).")]
    rep = J(d / "inference_seed_replicate.json")
    if rep:
        L += ["", f"**Inference-seed replicate** (os seed 0 − os seed 1, same windows): ADE "
              f"{cell(rep['families']['ADE']['ade_m'])}; max |path diff| {rep['max_abs_path_diff_m']:.4f} m."]
    for k in seeds:
        cp = J(d / f"cross_paired_s{k}.json")
        if cp:
            L += ["", f"**Key paired cells, S2 (0–2 s), inference seed {k}**", ""] + paired_digest(cp)
    an0 = J(d / f"analysis_s{seeds[0]}.json") if seeds else None
    if an0:
        keep = SB.ARM_ORDER
        SB.ARM_ORDER = [a for a in keep if a in DIGEST_ARMS]
        L += ["", f"**Levels, S2, inference seed {seeds[0]}** ({an0['n_windows']} windows / {an0['n_episodes']} eps)", "",
              SB.families_table(an0), "", "**Distance keeping (LONGITUDINAL)**", "", SB.dk_table(an0)]
        SB.ARM_ORDER = keep
    for k in seeds:
        t = J(d / f"tactical_v6_s{k}.json")
        if not t:
            continue
        if k == seeds[0]:
            L += ["", f"**TACTICAL, inference seed {k}.** Declared heads use the trainer's label clock (stamp above).",
                  "", SB.tactical_tables(t)]
        else:
            ln = [f"{h.upper()}: v6 decoder acc {SB.ci((t[h].get('v6_behaviour_decoder') or {}).get('acc'))} "
                  f"κ {SB.f((t[h].get('v6_behaviour_decoder') or {}).get('kappa'))}" for h in ("lat", "lon")]
            L += ["", f"**TACTICAL, inference seed {k}:** " + "; ".join(ln)]
    L += ["", "**STRATEGIC: NOT APPLICABLE, n = 0.** The strategic layer is OFF (`--no-strategic`); its route CE is "
          "gated off in training, so the route head is untrained and is not scored."]
    s6r = ["| seed | os ADE 1–6 s [CI] | os − ha0_ext | os − refcv4b | os − refcv5-v2 s0 |", "|---|---|---|---|---|"]
    any6 = False
    for k in seeds:
        a6, c6 = J(d / f"analysis_s6_s{k}.json"), J(d / f"cross_paired_s6_s{k}.json")
        if not (a6 and c6):
            continue
        any6 = True
        pc = lambda nm: cell((((c6["pairs"].get(nm) or {}).get("families") or {}).get("ADE") or {}).get("ade_m"))  # noqa: E731
        s6r.append(f"| {k} | {SB.ci(a6['arms']['os']['intervals']['metrics']['ade_dense_m'])} | "
                   f"{pc('os_minus_ha0ext')} | {pc('os_minus_refcv4b')} | {pc('os_minus_refcv5v2_s0')} |")
    if any6:
        L += ["", "**S6: the 6 s horizon, ADE 1–6 s** (BAR-R6-5 reads the os − ha0_ext column)", ""] + s6r
    for k in seeds:
        acc = J(d / f"acceptance_s{k}.json")
        if not acc:
            continue
        tf, ob = acc.get("tflip") or {}, acc.get("obedience") or {}
        m = tf.get("measured") or {}
        L += ["", f"**Frozen acceptance instruments, inference seed {k}.**",
              f"* **T-FLIP: {tf.get('verdict')}.** follows_FED {(m.get('follows_FED_command') or {}).get('mean')} "
              f"CI {(m.get('follows_FED_command') or {}).get('ci')}, against a bar of 0.50 (refcv5-v2: 0.205).",
              f"  * true − shuffled: {(m.get('paired_true_minus_shuffled') or {}).get('delta')} "
              f"CI {(m.get('paired_true_minus_shuffled') or {}).get('ci')}, against a bar of 0.38 (refcv5-v2: 0.099).",
              f"  * n = {(m.get('follows_FED_command') or {}).get('n_windows')} windows.",
              f"* **OBEDIENCE: {ob.get('verdict')}.** Obeys {(ob.get('obeys') or {}).get('mean')}; "
              f"{ob.get('rows_with_no_compliant_candidate')} of {ob.get('n_windows')} rows have NO compliant candidate. "
              "The bar is structurally unsatisfiable on this population (§3 F8)."]
    for k in seeds:
        vg = J(d / f"void_gates_s{k}.json")
        if vg:
            L.append(f"* VOID gates, seed {k}: STOP {vg['stop']['pass']}, ha0 {vg['ha0']['pass']}, "
                     f"profiles {vg.get('profiles')}")
    for f_ in ("LEVERS.md", "EPS0.md"):
        p = d / "levers" / f_
        if p.exists():
            L += ["", p.read_text(encoding="utf-8")]
    L += ["", f"Full tables: `raw/{s['tag']}/TABLES_s0.md`" + (", `TABLES_s1.md`" if len(seeds) > 1 else "") + "."]
    print("\n".join(L))


if __name__ == "__main__":
    main()
