"""Per-arm table of the early G-MAP-OVERFIT record (read-only). Prints markdown + a JSON line."""
import json
import sys

rec = json.load(open(sys.argv[1], encoding="utf-8"))
res = rec["results"]
cls = ["nocls", "drivable", "sidewalk", "lane", "crosswalk", "arrow", "edge", "hatched"]
bars = {"nocls": .85, "drivable": .85, "sidewalk": .85}
print("binding:", rec.get("binding"), "| launch_commit:", rec.get("launch_commit"),
      "| gpu_shared_with:", rec.get("gpu_shared_with"))
print("decision_rule:", rec.get("decision_rule"), "| weights sha256:",
      (rec.get("class_weights") or {}).get("sha256"))
for arm, r in res.items():
    f, s0 = r["final"], r["step0"]
    print(f"\n### {arm}  (s/step {r.get('s_per_step')}, peak {r.get('peak_mem_GiB')} GiB, "
          f"finite {r.get('loss_finite_every_step')}, trunk_s8_abs_change "
          f"{r.get('trunk_s8_abs_change'):.4g})")
    print("| class | bar | IoU (declared) | IoU (raw) | n | CE final / step 0 |")
    print("|---|---:|---:|---:|---:|---:|")
    for c in cls:
        bar = bars.get(c, .5)
        ce0, ce1 = s0["ce_mean"].get(c), f["ce_mean"].get(c)
        ratio = (ce1 / ce0) if (ce0 and ce1 is not None) else None
        iou, iour = f["iou"].get(c), f["iou_raw"].get(c)
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
        print(f"| {c} | {bar} | {fmt(iou)} | {fmt(iour)} | {f['n'].get(c)} | {fmt(ratio)} |")
v = rec["verdict"]
print("\nVERDICT:", json.dumps(v, default=str)[:3000])
print("\nearly_run:", json.dumps(rec.get("early_run", {}), default=str)[:600])
