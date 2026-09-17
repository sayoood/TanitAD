"""The cold start's T1 position, all four families — free information from the BASE roll.

⛔ Binding (Sayed 2026-08-02): LONGITUDINAL + LATERAL + TACTICAL + STRATEGIC in ADDITION
to ADE, per family, never pooled. ADE alone is an incomplete eval.
"""
import json, pathlib
D = pathlib.Path("C:/Users/Admin/tanitad-caches/ddv2rl-l1-20260917/t1_base.json")
d = json.loads(D.read_text(encoding="utf-8"))
fp = d["refcv3"]["families_paired"]
BLOCKS = {"paired_os_minus_ha0": "vs the hold-action floor (ha0)",
          "paired_os_minus_ha0ext": "vs the EXTENDED do-nothing baseline (ha0_ext) -- THE BAR",
          "paired_os_minus_navzero": "vs itself with the nav SIGNAL REMOVED -- what nav is worth",
          "paired_os_minus_navshuf": "vs itself with nav SHUFFLED -- pairing broken, marginal kept"}
out = {"_what": "refcv5-v2 COLD START at T1, all four metric families, from the BASE T1 roll.",
       "_tier": "T1 (self-action OPEN loop -- never 'closed loop')",
       "_evidence_class": "MEASURED (ours)",
       "_n": {"windows": d["n_windows"], "episodes": d["n_episodes"]},
       "_estimator": "paired_episode_cluster_bootstrap, n_boot 2000, seed 0",
       "blocks": {}}
for key, title in BLOCKS.items():
    blk = fp.get(key)
    if not blk: continue
    print(f"\n=== {key} — {title}")
    fam = blk.get("families", {})
    rec = {}
    for fname in sorted(fam):
        for mname, m in sorted(fam[fname].items()):
            if not isinstance(m, dict) or "delta" not in m: continue
            rec[f"{fname}.{mname}"] = {k: m.get(k) for k in ("delta","lo","hi","separated")}
            flag = "  <== SEPARATED" if m.get("separated") else ""
            print(f"   {fname:12s} {mname:24s} {m['delta']:+9.4f} "
                  f"[{m['lo']:+8.4f}, {m['hi']:+8.4f}]{flag}")
    out["blocks"][key] = {"_title": title, "metrics": rec}
pathlib.Path("C:/Users/Admin/qland/pkgrl/raw/t1_base_four_families.json").write_text(
    json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
print("\nwrote raw/t1_base_four_families.json")
