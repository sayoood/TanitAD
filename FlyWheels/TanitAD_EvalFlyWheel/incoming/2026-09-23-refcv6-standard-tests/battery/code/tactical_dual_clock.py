"""SPEC A5: the TACTICAL family of one battery tag under BOTH label clocks (CPU, no forward pass).

usage: python tactical_dual_clock.py <tag_dir> [--tables <label_tables.npz>]

For every rolled inference seed: `refcv6_panel.tactical_v6` with the OLD and the CORRECTED label
table (built and controlled by `label_clock_table.py`; its record must read PASS). Writes
`tactical_v6_s<k>_clock-old.json`, `..._clock-corrected.json` and `TACTICAL_CLOCKS.md` in the tag.
PRIMARY clock: OLD for a pre-switch checkpoint (step <= 34,500), CORRECTED for a post-switch one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refcv6_panel as P  # noqa: E402
import summarize_battery as SB  # noqa: E402

DEFAULT_TABLES = HERE.parent / "raw" / "label_clock" / "label_tables.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag_dir")
    ap.add_argument("--tables", default=str(DEFAULT_TABLES))
    a = ap.parse_args()
    root = Path(a.tag_dir)
    tables = Path(a.tables)
    rec = json.load(open(tables.parent / "label_clock_record.json", encoding="utf-8"))
    if rec.get("verdict") != "PASS":
        raise SystemExit(f"[clocks] the label tables' controls read {rec.get('verdict')} -- refusing (SPEC A5)")
    s = json.load(open(root / "battery_summary.json", encoding="utf-8"))
    step = int(s["step"])
    primary = "old" if step <= SB.SWITCH_STEP else "corrected"
    lc = rec.get("label_clock_corrected_build") or {}
    dt_src = (f"CORRECTED = grid_start + (t + w - 1 + n_stack - 1) * dt; {lc.get('n_from_sidecar')}/"
              f"{lc.get('n_clips')} eval clips on the sidecar clock (dt median {lc.get('dt_s_median')}), "
              f"{lc.get('n_nominal_dt_grid_start_0')} on nominal dt 0.1 s with grid_start 0, "
              f"{lc.get('n_pose_dt_grid_start_0')} on pose dt; OLD = (t + w - 1) * 0.1 s")
    seeds = sorted(int(p.name.split("_s")[-1]) for p in root.glob("panel_s*"))
    out = {"step": step, "primary_clock": primary, "clock_note": dt_src,
           "tables": str(tables), "tables_sha256": rec.get("label_tables_sha256"), "per_seed": {}}
    md = [f"### TACTICAL under both label clocks (SPEC A5): {s['tag']}, step {step}. PRIMARY clock: "
          f"**{primary.upper()}**", "", f"Clock and dt source: {dt_src}.", "",
          f"Tier T1 (UNRULED for an action-free model). Declared-head accuracy [episode-cluster CI], κ full-set; "
          f"in-band n per clock. Label tables sha256 `{str(rec.get('label_tables_sha256'))[:12]}…`; controls C1–C4 PASS "
          f"(`raw/label_clock/label_clock_record.json`).", ""]
    for k in seeds:
        man = json.load(open(root / f"panel_s{k}" / "manifest.json", encoding="utf-8"))
        clip_eid = {int(e["episode_index"]): int(e["file_index"]) for e in man["episodes"]}
        ex = str(Path(s["stages"][f"roll_s{k}"]["dump_dir"]) / "refcv6_extras.npz")
        res = {}
        for clock in ("old", "corrected"):
            t = P.tactical_v6(ex, clip_eid, labels=P.labels_for_extras(ex, str(tables), clock))
            t["primary"] = clock == primary
            json.dump(t, open(root / f"tactical_v6_s{k}_clock-{clock}.json", "w", encoding="utf-8"),
                      indent=1, default=str)
            res[clock] = t
        out["per_seed"][k] = {c: {h: {"n_in_band": res[c][h]["n_in_band"],
                                      "v6_acc": (res[c][h]["v6_behaviour_decoder"] or {}).get("acc"),
                                      "v6_kappa": (res[c][h]["v6_behaviour_decoder"] or {}).get("kappa"),
                                      "ztac_kappa": (res[c][h].get("z_tac_v7_heads") or {}).get("kappa")}
                                  for h in ("lat", "lon")} for c in res}
        md += [f"**Inference seed {k}**", "",
               "| head | surface | OLD clock: acc [CI] · κ (n in band) | CORRECTED clock: acc [CI] · κ (n in band) |",
               "|---|---|---|---|"]
        for h in ("lat", "lon"):
            for surf in ("v6_behaviour_decoder", "z_tac_v7_heads", "v6_behaviour_decoder_NAVZERO"):
                cells = []
                for c in ("old", "corrected"):
                    x = res[c][h].get(surf) or {}
                    cells.append(f"{SB.ci(x.get('acc'))} · {SB.f(x.get('kappa'))} ({res[c][h]['n_in_band']})"
                                 + (" ⭐" if c == primary else ""))
                md.append(f"| {h.upper()} | {surf} | {cells[0]} | {cells[1]} |")
        g_old = res["old"]["goal_22"]["per_class"].get("nav_true", {})
        g_new = res["corrected"]["goal_22"]["per_class"].get("nav_true", {})
        md += ["", "| goal token (nav true) | OLD: n_pos / AUROC / status | CORRECTED: n_pos / AUROC / status |",
               "|---|---|---|"]
        for tk in res["old"]["goal_22"]["tokens"]:
            o, n = g_old.get(tk) or {}, g_new.get(tk) or {}
            md.append(f"| {tk} | {o.get('n_pos')} / {SB.f(o.get('auroc'))} / {o.get('status')} | "
                      f"{n.get('n_pos')} / {SB.f(n.get('auroc'))} / {n.get('status')} |")
        md.append("")
    md.append("⭐ = the PRIMARY clock for this checkpoint. A tactical comparison across step 34,500 must show both "
              "clocks, and it mixes training time with the fix.")
    json.dump(out, open(root / "tactical_clocks.json", "w", encoding="utf-8"), indent=1, default=str)
    (root / "TACTICAL_CLOCKS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:30]))


if __name__ == "__main__":
    main()
