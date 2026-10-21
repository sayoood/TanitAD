"""tables_md.py -- markdown tables for RESULT.md from the box / map read JSONs.  python tables_md.py --box b.json --map m.json --out-dir d"""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmt import cell, dcell, f


def box_tables(B):
    A, Bn = B["arms"]["a"], B["arms"]["b"]
    C = B["ap_auroc_cells"]
    L = []
    L.append(f"| metric (pooled over {B['n_windows_common']} windows / {B['n_clips']} episodes) | {A} | {Bn} | paired {A} - {Bn} |")
    L.append("|---|---|---|---|")
    names = [("ap0p5_all", "box AP@0.5 m (all classes)"), ("ap1_all", "box AP@1 m"), ("ap2_all", "box AP@2 m"), ("ap4_all", "box AP@4 m"),
             ("map0p5", "class-mean AP@0.5 m"), ("map1", "class-mean AP@1 m"), ("map2", "class-mean AP@2 m"), ("map4", "class-mean AP@4 m"),
             ("ap2_band0", "AP@2 m, 0-20 m"), ("ap2_band1", "AP@2 m, 20-40 m"), ("ap2_band2", "AP@2 m, 40-60 m"),
             ("auroc_matched", "AUROC (matched slots)"), ("auroc_objectness", "AUROC (objectness, 2 m)")]
    for k, lab in names:
        c = C[k]
        L.append(f"| {lab} | {cell(c[A])} | {cell(c[Bn])} | {dcell(c['paired_a_minus_b'])} |")
    cls = B["classes"]
    L.append("")
    L.append(f"| per-class AP@2 m | {A} | {Bn} | paired |")
    L.append("|---|---|---|---|")
    for i, nm in enumerate(cls):
        c = C[f"ap2_cls{i}"]
        L.append(f"| {nm} | {cell(c[A])} | {cell(c[Bn])} | {dcell(c['paired_a_minus_b'])} |")
    return "\n".join(L)


def census_table(B, vn):
    A, Bn = B["arms"]["a"], B["arms"]["b"]
    V = B["census_variants"][vn]
    L = [f"gate / NMS: `{vn}` -- {A}: gate {V['gate']['a']:.4f}, NMS {V['nms']['a']}; {Bn}: gate {V['gate']['b']:.4f}, NMS {V['nms']['b']}", "",
         f"| metric | {A} | {Bn} | paired |", "|---|---|---|---|"]
    for k, lab in (("rec", "recall"), ("prec", "precision"), ("f1", "F1"), ("conf_ratio", "conf_ratio (n_conf / n_pos)"),
                   ("boxes_per_object", "boxes per detected object"), ("frac_objects_ge2", "frac. objects with >=2 boxes"),
                   ("cls_acc_tp", "class accuracy on TPs")):
        c = V["cells"][k]
        L.append(f"| {lab} | {cell(c[A])} | {cell(c[Bn])} | {dcell(c['paired_a_minus_b'])} |")
    cnt = V["counts"]
    L.append(f"| counts (tp / n_conf / n_pos / detected objects) | {int(cnt[A]['tp'])} / {int(cnt[A]['n_conf'])} / {int(cnt[A]['n_pos'])} / {int(cnt[A]['n_detected_objects'])} "
             f"| {int(cnt[Bn]['tp'])} / {int(cnt[Bn]['n_conf'])} / {int(cnt[Bn]['n_pos'])} / {int(cnt[Bn]['n_detected_objects'])} | |")
    r = V["recall_by_class_at_gate"]
    L.append("")
    L.append("recall by class at this gate: " + "; ".join(f"{nm} {r[A][i]:.3f} / {r[Bn][i]:.3f}" for i, nm in enumerate(B["classes"])) + f"  ({A} / {Bn})")
    return "\n".join(L)


def map_tables(M):
    A, Bn = M["arms"]["a"], M["arms"]["b"]
    S = M["summary"]["arms"]
    P = M.get("paired_a_minus_b", {})
    bands = M["bands"]
    names = ["nocls", "drivable", "lane", "crosswalk", "arrow", "edge", "hatched", "sidewalk"]
    # the class names inside the table come from MH.CLASS_NAMES; read them from the keys
    cls_names = []
    for k in S[A]:
        cn = k.split("|")[0]
        if cn not in cls_names:
            cls_names.append(cn)
    L = [f"10 cm map IoU per class x distance band, {M['n_windows']} windows / {M['n_clips']} episodes (cells with GT; n_gt = {A} GT cells)", ""]
    L.append(f"| class | band | n_gt | {A} IoU | {Bn} IoU | paired {A} - {Bn} |")
    L.append("|---|---|---|---|---|---|")
    for cn in cls_names:
        for b in bands:
            ka = f"{cn}|{b}|iou"
            ca, cb = S[A].get(ka), S[Bn].get(ka)
            if ca is None or ca.get("mean") is None:
                continue
            ng = S[A].get(f"{cn}|{b}|n_gt")
            pc = P.get(ka)
            L.append(f"| {cn} | {b} | {ng} | {cell(ca)} | {cell(cb) if cb and cb.get('mean') is not None else 'n/a'} | {dcell(pc) if pc else 'n/a'} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box")
    ap.add_argument("--map")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.box:
        B = json.load(open(a.box))
        if "ap0p5_all" in B["ap_auroc_cells"]:
            open(f"{a.out_dir}/tbl_box_ap.md", "w", encoding="utf-8").write(box_tables(B))
        for vn in B["census_variants"]:
            open(f"{a.out_dir}/tbl_census_{vn}.md", "w", encoding="utf-8").write(census_table(B, vn))
    if a.map:
        M = json.load(open(a.map))
        open(f"{a.out_dir}/tbl_map.md", "w", encoding="utf-8").write(map_tables(M))


if __name__ == "__main__":
    main()
