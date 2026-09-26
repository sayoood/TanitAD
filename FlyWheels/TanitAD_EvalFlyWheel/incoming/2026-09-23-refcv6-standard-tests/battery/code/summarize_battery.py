"""Render a battery tag directory (raw/<tag>/) as markdown tables. Every number is copied from a JSON
the runner wrote; nothing is recomputed here. Usage: python summarize_battery.py <tag_dir> [seed]"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ARM_ORDER = ["os", "os_navzero", "os_navshuf", "os_navflip", "os_vmaxzero", "b_refcv4b",
             "b_refcv5v2_s0", "b_refcv5v2_s1", "ha", "ha0", "ha0_ext", "stop", "oracle_sel"]
LABEL = {"os": "**refcv6 `os`**", "os_navzero": "refcv6 nav withheld", "os_navshuf": "refcv6 nav shuffled",
         "os_navflip": "refcv6 nav flipped", "os_vmaxzero": "refcv6 max-speed withheld",
         "b_refcv4b": "refcv4b `os` (banked)", "b_refcv5v2_s0": "refcv5-v2 `os` seed 0 (banked)",
         "b_refcv5v2_s1": "refcv5-v2 `os` seed 1 (banked)", "ha": "`ha` hold-action",
         "ha0": "`ha0` constant velocity", "ha0_ext": "`ha0_ext` echo", "stop": "`stop` STOP",
         "oracle_sel": "`oracle_sel` (T0 ceiling)"}


SWITCH_STEP = 34500      # PI "stop now, resume with fixes" (A16), resumed on 82c2331 at this step


def run_stamp(step: int) -> str:
    """The Master Mind's per-checkpoint stamp (2026-09-26): which experiment a checkpoint belongs to."""
    if step <= SWITCH_STEP:
        return ("⛔ **Run-defect stamp (pre-switch checkpoint, step ≤ 34,500):** F3 detach-only, F4 on the last "
                "layer only; tactical labels ~0.37 s early (D-REFCV6-F3-WHITELIST, D-REFCV6-LABEL-CLOCK; audit "
                "92337fa6). No weakness below may be attributed to the REGISTERED design while these hold. TACTICAL "
                "is read under BOTH label clocks (SPEC A5); this checkpoint's PRIMARY clock is the OLD one it was "
                "trained on. Physical-unit rates use dt = 0.5 s for a true 0.5033 s (row = 0.100667 s), identically "
                "for every arm and baseline.")
    return ("⛔ **Run stamp (post-switch checkpoint, step > 34,500):** hybrid: F3 cascade loss + true label clock "
            "from step 34,500 (resumed on 82c2331; F4 unchanged). A comparison with a pre-switch checkpoint MIXES "
            "training time with the fix and is never attributed to the fix alone. TACTICAL is read under BOTH label "
            "clocks (SPEC A5); this checkpoint's PRIMARY clock is the CORRECTED one. Physical-unit rates use "
            "dt = 0.5 s for a true 0.5033 s (row = 0.100667 s), identically for every arm and baseline.")



def f(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, dict):
        # a metric REFUSED by the instrument (e.g. heading of a path that never moves): its status
        # and n, never 0.0 and never a silent blank
        return f"{v.get('status', 'REFUSED')} (n={v.get('n')})"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def ci(c, nd=4):
    if not c:
        return "—"
    return f"{f(c.get('mean', c.get('delta')), nd)} [{f(c.get('lo'), nd)}, {f(c.get('hi'), nd)}]"


def families_table(an: dict, ade_label: str = "ADE 0–2 s") -> str:
    rows = [f"| arm | tier | {ade_label} m [CI] | FDE m | speed MAE m/s | tgt-speed acc@0.5 | along MAE m | "
            "heading ° | curvature 1/m | yaw-rate °/s | cross-track m | lat κ (traj) | lon κ (traj) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARM_ORDER:
        blk = an["arms"].get(a)
        if not blk:
            continue
        ff = blk["four_families"]
        iv = blk["intervals"]["metrics"]
        lo, la, ta = ff.get("longitudinal", {}), ff.get("lateral", {}), ff.get("tactical", {})
        tsa = (lo.get("target_speed_acc") or {}).get("within_0.5_mps")
        latk = (ta.get("lateral_decision") or {}).get("kappa")
        lonk = (ta.get("longitudinal_decision") or {}).get("kappa")
        rows.append(f"| {LABEL.get(a, a)} | {blk['tier']} | {ci(iv.get('ade_dense_m'))} | "
                    f"{f((iv.get('fde_last_m') or {}).get('mean'))} | {f(lo.get('speed_mae_mps'))} | "
                    f"{f(tsa)} | {f(lo.get('along_mae_m'))} | {f(la.get('heading_mae_deg'))} | "
                    f"{f(la.get('curvature_mae_1pm'), 6)} | {f(la.get('yaw_rate_mae_degps'))} | "
                    f"{f(la.get('cross_mae_m'))} | {f(latk)} | {f(lonk)} |")
    return "\n".join(rows)


def dk_table(an: dict) -> str:
    dk = ((an.get("refcv3") or {}).get("distance_keeping") or {})
    per = dk.get("per_arm") or {}
    out = [f"lead block: `{os.path.basename(str(dk.get('block')))}`, status {dk.get('status')}, "
           f"instants {((per.get('os') or {}).get('instants_s'))} s",
           "", "| arm | status | n with lead (eps) | min headway m [CI] | min time-gap s [CI] (n) | "
               "min TTC s [CI] | n closing (rest censored at 30 s) |",
           "|---|---|---|---|---|---|---|"]
    for a in ARM_ORDER:
        b = per.get(a)
        if not b:
            continue
        c = b.get("ci") or {}
        hw, tg, tt = (c.get("headway_min_m") or {}, c.get("time_gap_min_s") or {},
                      c.get("min_ttc_s") or {})
        out.append(f"| {LABEL.get(a, a)} | {b.get('status')} | {b.get('n')} ({hw.get('n_episodes')}) | "
                   f"{ci(hw)} | {ci(tg)} ({b.get('n_time_gap')}) | {ci(tt)} | {b.get('n_closing')} |")
    return "\n".join(out)


YAW_VALID = "LAT_yaw_rate_mae_radps_valid"      # SPEC A3 (refcv6_panel.YAW_VALID)


def paired_table(cp: dict, names=None) -> str:
    a3 = any(YAW_VALID in ((b.get("families") or {}).get("lateral") or {})
             for b in cp["pairs"].values())
    yaw_hdr = ("yaw-rate rad/s (valid steps, A3)" if a3 else
               "yaw-rate rad/s (⚠ UNMASKED shared cell, DEFECTIVE per SPEC A3 — do not quote)")
    out = [f"| cell (b − a) | ADE m [CI] sep | FDE m | speed MAE | along MAE | heading ° | {yaw_hdr} | "
           "cross-track | traj lat correct | traj lon correct |", "|---|---|---|---|---|---|---|---|---|---|"]
    for nm, blk in cp["pairs"].items():
        if names and nm not in names:
            continue
        if "families" not in blk:
            out.append(f"| {nm} | {blk.get('status')} {blk.get('missing')} | | | | | | | | |")
            continue
        fam = blk["families"]

        def c(fk, mk):
            x = (fam.get(fk) or {}).get(mk)
            if not x or "delta" not in x:
                return "—"
            s = "**sep**" if x.get("separated") else "ns"
            return f"{f(x['delta'])} [{f(x['lo'])}, {f(x['hi'])}] {s}"
        yaw = c('lateral', YAW_VALID) if a3 else c('lateral', 'LAT_yaw_rate_mae_radps')
        out.append(f"| {blk['direction']} | {c('ADE', 'ade_m')} | {c('ADE', 'fde_m')} | "
                   f"{c('longitudinal', 'LON_speed_mae_mps')} | {c('longitudinal', 'LON_along_mae_m')} | "
                   f"{c('lateral', 'LAT_heading_mae_deg')} | {yaw} | "
                   f"{c('lateral', 'LAT_cross_mae_m')} | {c('tactical', 'TAC_traj_lat_correct')} | "
                   f"{c('tactical', 'TAC_traj_lon_correct')} |")
    n = cp.get("n_windows"), cp.get("n_episodes")
    return ("\n".join(out) + f"\n\nn = {n[0]} windows / {n[1]} episodes · estimator: paired episode-cluster bootstrap"
            + ("" if a3 else "\n\n⚠ This panel predates SPEC A3: its yaw-rate column is the shared UNMASKED "
               "cell (scored on steps with no path tangent) and must not be quoted."))


def tactical_tables(t: dict) -> str:
    out = []
    for head in ("lat", "lon"):
        b = t[head]
        out.append(f"**{head.upper()}** (v8 labels, in-band windows n = {b['n_in_band']} / "
                   f"{b['n_episodes']} eps; classes {b['classes']})")
        out.append("")
        out.append("| surface | acc [CI] | κ | majority-class rate |")
        out.append("|---|---|---|---|")
        for s in ("v6_behaviour_decoder", "z_tac_v7_heads", "v6_behaviour_decoder_NAVZERO"):
            x = b.get(s) or {}
            out.append(f"| {s} | {ci(x.get('acc'))} | {f(x.get('kappa'))} | {f(x.get('majority_class_rate'))} |")
        out.append("")
    g = t["goal_22"]
    out.append(f"**22-token goal selection** (per class, nav true; floor n_pos {g['scoreability_floor_n_pos']})")
    out.append("")
    out.append("| token | n_pos / n_neg | eps | AUROC | AP | prevalence | P@0.5 | R@0.5 | AUROC nav-zero | status |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    pt = g["per_class"].get("nav_true", {})
    pz = g["per_class"].get("nav_zero", {})
    for tk in g["tokens"]:
        x = pt.get(tk) or {}
        z = pz.get(tk) or {}
        out.append(f"| {tk} | {x.get('n_pos')} / {x.get('n_neg')} | {x.get('n_episodes')} | {f(x.get('auroc'))} | "
                   f"{f(x.get('ap'))} | {f(x.get('prevalence'))} | {f(x.get('precision@0.5'))} | "
                   f"{f(x.get('recall@0.5'))} | {f(z.get('auroc'))} | {x.get('status')} |")
    return "\n".join(out)


def main():
    d = Path(sys.argv[1])
    seed = sys.argv[2] if len(sys.argv) > 2 else "0"
    s = json.load(open(d / "battery_summary.json", encoding="utf-8"))
    print(f"## {s['tag']} — ckpt md5 `{s['ckpt_md5']}`, step {s['step']}, SPEC sha256 `{s['spec_sha256'][:16]}…`\n")
    print("**Tier:** every number is T1 = self-action OPEN loop (UNRULED for an action-free model) except "
          "`oracle_sel` (T0). Not closed loop, not driving. **Estimator:** full-set pooled means; paired "
          "episode-cluster bootstrap (n_boot 2000, seed 0, cluster = clip). One training seed: every "
          "separated cell answers the EPISODE question only; the INFERENCE question is answered by the "
          "seed replicate.\n")
    print(run_stamp(int(s["step"])) + "\n")
    g0 = (s.get("stages") or {}).get("g0") or {}
    print("### Gate\n")
    print(f"* G0 as registered: **{g0.get('G0_as_registered')}**; G0-A1: **{g0.get('G0_A1')}**; "
          f"G0-A2 (operative): **{g0.get('G0_A2')}**; A2 wrapper max rel by condition: "
          f"{g0.get('A2_max_wrapper_rel')}; A1 reasons {g0.get('G0_A1_reasons')}; "
          f"A2 reasons {g0.get('G0_A2_reasons')}")
    for k in sorted(s.get("stages") or {}):
        v = s["stages"][k]
        if k.startswith("void_gates_s"):
            print(f"* {k}: STOP {v['stop']['pass']}, ha0 {v['ha0']['pass']}, profiles {v['profiles']}")
        elif k.startswith("panel_s"):
            print(f"* {k} pairing gates (g + ha/ha0/ha0_ext bit-identical vs every baseline): "
                  f"{v.get('void_gates_3_4')}")
        elif k.startswith("roll_s"):
            print(f"* {k}: {v.get('n_windows')} windows / {v.get('n_episodes')} eps, skipped "
                  f"{v.get('skipped')}, device {v.get('device')}, wall {v.get('wall_s')} s, peak "
                  f"{v.get('cuda_max_memory_allocated_gib')} GiB, frame memo {v.get('frame_memo')}")
        elif k.startswith("echo_gate_s"):
            print(f"* {k} (echo_gate GATE 1 only; margins INHERITED from refcv5_compare): "
                  f"{json.dumps((v or {}).get('per_reference'))[:600]}")
        elif k.startswith("criteria_s"):
            print(f"* {k}: {json.dumps(v)}")
    if s.get("inference_seed_replicate"):
        print(f"* inference-seed replicate (os seed A − seed B, same windows): "
              f"{json.dumps(s['inference_seed_replicate'])}")
    print(f"* parent_cuda_initialized: {s.get('parent_cuda_initialized')}\n")
    an = json.load(open(d / f"analysis_s{seed}.json", encoding="utf-8"))
    print(f"### Four families + ADE, S2 ({an['n_windows']} windows / {an['n_episodes']} episodes), inference seed {seed}\n")
    print(families_table(an))
    print("\n### Distance keeping\n")
    print(dk_table(an))
    cp = json.load(open(d / f"cross_paired_s{seed}.json", encoding="utf-8"))
    print(f"\n### Paired cells, S2, inference seed {seed}\n")
    print(paired_table(cp))
    t = json.load(open(d / f"tactical_v6_s{seed}.json", encoding="utf-8"))
    print(f"\n### TACTICAL — declared heads and the 22-token goal set, inference seed {seed}\n")
    print(tactical_tables(t))
    s6 = d / f"cross_paired_s6_s{seed}.json"
    if s6.exists():
        print(f"\n### S6 — the 6 s horizon, inference seed {seed}\n")
        a6 = json.load(open(d / f"analysis_s6_s{seed}.json", encoding="utf-8"))
        print(families_table(a6, ade_label="ADE 1–6 s"))
        print()
        print(paired_table(json.load(open(s6, encoding="utf-8"))))
    acc = d / f"acceptance_s{seed}.json"
    if acc.exists():
        print("\n### Frozen refcv6 acceptance instruments\n")
        print("```json\n" + json.dumps(json.load(open(acc, encoding="utf-8")), indent=1, default=str)[:3000] + "\n```")
    print("\n### Bars\n")
    for b in s.get("bars", []):
        print(f"* **{b['id']}** — {b['statement']} — **{b['verdict']}** — "
              + "; ".join(f"seed {k}: {v.get('delta')} [{v.get('lo')}, {v.get('hi')}]"
                          for k, v in (b.get("per_inference_seed") or {}).items()))


if __name__ == "__main__":
    main()
