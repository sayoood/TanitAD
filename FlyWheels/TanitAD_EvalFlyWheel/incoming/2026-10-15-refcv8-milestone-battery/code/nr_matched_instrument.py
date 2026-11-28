"""nr_matched_instrument.py -- BAR-R8-NR re-read on the MATCHED instrument (refcv8 `os` with the max-speed ceiling filter OFF), seed 0.  CPU only.

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' \
      python nr_matched_instrument.py probe                       # task 1: where is the filter-OFF plan, per surface
      python nr_matched_instrument.py read --tree step10500_unc   # task 2: variants + controls + the matched read, one tree
      python nr_matched_instrument.py assemble                    # task 3: the 8-tree side-by-side + verdict-change table

WHY (AMENDMENT A-R7M-1).  The banked refcv7-50,400 dumps that every refcv8 paired read compares against were rolled with the max-speed ceiling
filter INERT; refcv8's batteries were rolled with it ACTING (E9's final re-rank applies `ceil_keep`; it changes the selected anchor on ~6-7 %
of the windows).  So BAR-R8-NR compared two different instruments.  refcv8's seed-0 dumps carry the SAME window rolled with the filter OFF
(`dump_s0/refcv6_extras.npz :: filter_off.traj`, [N, 8, 2], same DDIM draw), so the matched read needs no GPU.

HOW (no metric is re-derived).  For each tree and surface a COPY of the banked panel directory is written to a scratch dir whose `os` arm is
replaced by the filter-OFF plan (rows matched on (episode_index, ws), slots = the panel manifest's own `grid.slots`); then
`pair_read_r8.read_surface` -- the SAME function that produced the banked PAIRED_READ -- reads it: `refav1_arm._components`, four_families
geometry, the SPEC section 5 cells, `taniteval.ci.paired_episode_cluster_bootstrap` (B 2000, seed 0, 95 % percentile), refcv8 minus
refcv7-50,400 (`b_refcv7_50400_s0`, the panel's own embedded arm).

CONTROLS (pre-registered here, before any matched number was looked at):
  (A) IDENTITY.  The same pipeline with `os` substituted BY ITSELF (A1: the panel's own `os` rewritten through the writer; A2: `os` taken from
      the extras' `nav_true.traj` through the SAME row-matching / slot-selection / dtype path the OFF arm uses) must reproduce every banked
      seed-0 row of the banked PAIRED_READ EXACTLY (delta, lo, hi, n_windows, n_episodes, separated, n_dropped_nonfinite; |mean| differences
      reported).  If (A) fails the tool exits non-zero and writes NO matched number.
  (B) MISPAIRING.  The OFF plans rolled by 35 windows (about one episode block: B_ep) must change EVERY NR headline cell's delta and raise the
      S2 ADE of refcv8 by >= 0.05 m; rolled by one window (B_win) must change >= 1 NR headline delta.  A matched read that survives a
      mispairing unchanged was not reading the OFF arm.
ESTIMATOR / QUESTION: episode-cluster bootstrap over the 139 eval episodes, models FIXED -- "another draw of EPISODES" only; NOT another
training run, NOT another inference run.  TIER: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).  Inference seed 0.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ARCHIVE = "C:/Users/Admin/refcv8_eval_archive/battery"
SCRATCH = "C:/Users/Admin/refcv8_eval_archive/nr_matched_scratch"
OUT = HERE.parent / "raw" / "nr_matched_instrument"
TREES = ["step5000_unc", "step5000ema_unc", "step10500_unc", "step10500ema_unc",
         "step15000_unc", "step15000ema_unc", "step20000_unc", "step20000ema_unc"]
# later milestones (2026-10-18): R8_NR_EXTRA_TREES=step30000_unc,step30000ema_unc adds them to every stage
TREES += [t for t in os.environ.get("R8_NR_EXTRA_TREES", "").split(",") if t and t not in TREES]
SEED = 0
REF_ARM = "b_refcv7_50400_s0"
PANEL = {"S2": "panel_s0", "S6": "s6_s0"}
BASELINE_DUMP = {"S2": "dump_s0", "S6": "s6_s0"}
TIER = "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)"
QUESTION = ("episode-cluster bootstrap over the 139 eval episodes, models FIXED: answers 'another draw of EPISODES' only; "
            "NOT another training run, NOT another inference run")
N_BOOT = 2000
MIS_EP_ROLL = 35          # windows: about one episode block (episode 0 has 35 windows on S2)
MIS_EP_ADE_RISE_MIN = 0.05  # m, pre-registered


# ------------------------------------------------------------------------------------------ helpers
def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jdump(obj, p):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)


def load_extras(dump_dir: str) -> dict:
    z = np.load(os.path.join(dump_dir, "refcv6_extras.npz"))
    need = ["clip_index", "ws", "nav_true.traj", "nav_true.sel_idx"]
    miss = [k for k in need if k not in z.files]
    if miss:
        raise SystemExit(f"[nr_matched] {dump_dir}: extras lack {miss}")
    ex = {k: z[k] for k in need}
    ex["has_filter_off"] = "filter_off.traj" in z.files
    if ex["has_filter_off"]:
        ex["filter_off.traj"] = z["filter_off.traj"]
        ex["filter_off.sel_idx"] = z["filter_off.sel_idx"]
    ex["key"] = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(ex["clip_index"], ex["ws"]))}
    if len(ex["key"]) != len(ex["ws"]):
        raise SystemExit(f"[nr_matched] {dump_dir}: duplicate (clip_index, ws) rows in the extras")
    return ex


def panel_layout(panel_dir: str):
    man = jload(os.path.join(panel_dir, "manifest.json"))
    files = sorted(glob.glob(os.path.join(panel_dir, "ep*.npz")))
    ep_of = {int(e["file_index"]): int(e["episode_index"]) for e in man["episodes"]}
    return man, files, ep_of


def fi_of(f: str) -> int:
    return int(os.path.basename(f)[2:5])


def rows_of(ex: dict, ep_of: dict, f: str, ws) -> list:
    rows = [ex["key"].get((ep_of[fi_of(f)], int(w))) for w in ws]
    if any(r is None for r in rows):
        raise SystemExit(f"[nr_matched] {f}: {sum(r is None for r in rows)} panel windows are absent from the extras (refusing to pair)")
    return rows


def plans_from_extras(ex: dict, ep_of: dict, files: list, key: str, slots: list) -> list:
    out = []
    for f in files:
        ws = np.load(f)["ws"]
        out.append(ex[key][rows_of(ex, ep_of, f, ws)][:, slots].astype(np.float32))
    return out


def own_os(files: list) -> list:
    return [np.load(f)["os"].astype(np.float32) for f in files]


def roll_plans(plans: list, shift: int) -> list:
    """window i receives the plan of window i + shift (cyclic over the concatenated window axis), shapes preserved."""
    sizes = [len(p) for p in plans]
    cat = np.roll(np.concatenate(plans, axis=0), -shift, axis=0)
    out, a = [], 0
    for n in sizes:
        out.append(cat[a:a + n])
        a += n
    return out


def write_variant(src_dir: str, dst_dir: str, files: list, new_os: list) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copyfile(os.path.join(src_dir, "manifest.json"), os.path.join(dst_dir, "manifest.json"))
    for old in glob.glob(os.path.join(dst_dir, "ep*.npz")):
        os.remove(old)
    for f, new in zip(files, new_os):
        z = dict(np.load(f))
        if z["os"].shape != new.shape:
            raise SystemExit(f"[nr_matched] {f}: os shape {z['os'].shape} vs replacement {new.shape}")
        z["os"] = new
        np.savez(os.path.join(dst_dir, os.path.basename(f)), **z)


def max_abs(a_list: list, b_list: list) -> float:
    return float(max(np.abs(a.astype(np.float64) - b.astype(np.float64)).max() for a, b in zip(a_list, b_list)))


# ------------------------------------------------------------------------------------------ task 1: probe
def probe_tree(tree: str, r7base: str) -> dict:
    root = os.path.join(ARCHIVE, tree)
    ex = load_extras(os.path.join(root, "dump_s0"))
    res = {"tree": tree, "extras_has_filter_off": ex["has_filter_off"], "n_extras_rows": int(len(ex["ws"]))}
    ex1 = np.load(os.path.join(root, "dump_s1", "refcv6_extras.npz"))
    res["seed1_extras_has_filter_off"] = "filter_off.traj" in ex1.files
    res["seed1_extras_keys"] = len(ex1.files)
    if not ex["has_filter_off"]:
        res["status"] = "NO filter_off arm in the seed-0 extras: the matched read is impossible from banked data"
        return res
    for surf in ("S2", "S6"):
        pdir = os.path.join(root, PANEL[surf])
        man, files, ep_of = panel_layout(pdir)
        slots = list(man["grid"]["slots"])
        s = {"panel_dir": pdir, "grid_slots": slots, "n_panel_arms_in_ep000": sorted(np.load(files[0]).files)}
        s["filter_off_arm_in_panel_ep000"] = "os_filteroff" in s["n_panel_arms_in_ep000"]
        dz = np.load(os.path.join(pdir, "decisions", os.path.basename(files[0])))
        s["decisions_plan_full_keys"] = sorted(k for k in dz.files if k.startswith("plan_full"))
        s["filter_off_in_decisions"] = any("filter_off" in k or "filteroff" in k for k in dz.files)
        on = plans_from_extras(ex, ep_of, files, "nav_true.traj", slots)
        off = plans_from_extras(ex, ep_of, files, "filter_off.traj", slots)
        own = own_os(files)
        s["n_windows"] = int(sum(len(a) for a in on))
        s["panel_os_vs_extras_nav_true_slots_max_abs"] = max_abs(own, on)
        # decisions plan_full_nav_true (all 8 slots) vs the extras' nav_true.traj
        d_full = []
        for f in files:
            z = np.load(os.path.join(pdir, "decisions", os.path.basename(f)))
            ws = np.load(f)["ws"]
            d_full.append(np.abs(z["plan_full_nav_true"].astype(np.float64)
                                 - ex["nav_true.traj"][rows_of(ex, ep_of, f, ws)].astype(np.float64)).max())
        s["decisions_plan_full_nav_true_vs_extras_nav_true_traj_max_abs"] = float(max(d_full))
        if surf == "S2" and s["filter_off_arm_in_panel_ep000"]:
            pf = [np.load(f)["os_filteroff"].astype(np.float32) for f in files]
            s["panel_os_filteroff_vs_extras_filter_off_max_abs"] = max_abs(pf, off)
        sel_on, sel_off, same_plan_diff = [], [], 0.0
        n_same_sel = n_diff_sel_plan_equal = 0
        for f, a_on, a_off in zip(files, on, off):
            ws = np.load(f)["ws"]
            rr = rows_of(ex, ep_of, f, ws)
            so, sf = ex["nav_true.sel_idx"][rr], ex["filter_off.sel_idx"][rr]
            sel_on.append(so)
            sel_off.append(sf)
            same = so == sf
            n_same_sel += int(same.sum())
            if same.any():
                same_plan_diff = max(same_plan_diff, float(np.abs(a_on[same].astype(np.float64) - a_off[same].astype(np.float64)).max()))
            n_diff_sel_plan_equal += int(((~same) & np.all(a_on == a_off, axis=(1, 2))).sum())
        so, sf = np.concatenate(sel_on), np.concatenate(sel_off)
        s["n_windows_filter_changes_selected_anchor"] = int((so != sf).sum())
        s["frac_windows_filter_changes_selected_anchor"] = float((so != sf).mean())
        s["max_abs_on_vs_off_plan_where_selected_anchor_equal"] = same_plan_diff
        s["n_windows_selection_changed_but_slot_plan_identical"] = n_diff_sel_plan_equal
        s["max_abs_on_vs_off_plan_all_windows"] = max_abs(on, off)
        res[surf] = s
    # the refcv7-50,400 side: is its ceiling really inert?  (INHERITED from KNOWN_VALUE_CONTROL.md -> re-verified here from the banked dump)
    ex7 = load_extras(os.path.join(r7base, "dump_s0"))
    if ex7["has_filter_off"]:
        res["refcv7_50400_dump_s0_selection_changed_by_filter"] = int((ex7["nav_true.sel_idx"] != ex7["filter_off.sel_idx"]).sum())
        res["refcv7_50400_dump_s0_plan_on_vs_off_max_abs"] = float(np.abs(ex7["nav_true.traj"].astype(np.float64)
                                                                          - ex7["filter_off.traj"].astype(np.float64)).max())
        res["refcv7_50400_dump_s0_n_windows"] = int(len(ex7["ws"]))
    else:
        res["refcv7_50400_dump_s0_filter_off"] = "ABSENT (cannot re-verify inertness from this dump)"
    return res


# ------------------------------------------------------------------------------------------ task 2: the matched read
def compare_to_banked(read: dict, banked_seed: dict) -> dict:
    mism, n_cmp, mean_diff = [], 0, 0.0
    for key, brow in banked_seed["rows"].items():
        for which in ("os_minus_refcv7_50400", "os_minus_ha0_ext"):
            b, n = brow.get(which), read["rows"].get(key, {}).get(which)
            if b is None or "delta" not in b:
                continue
            if n is None or "delta" not in n:
                mism.append({"metric": key, "cell": which, "why": "absent in the re-read"})
                continue
            n_cmp += 1
            for fld in ("delta", "lo", "hi", "n_windows", "n_episodes", "separated", "n_dropped_nonfinite"):
                if b.get(fld) != n.get(fld):
                    mism.append({"metric": key, "cell": which, "field": fld, "banked": b.get(fld), "reread": n.get(fld)})
            for fld in ("mean_new", "mean_ref"):
                mean_diff = max(mean_diff, abs(float(b[fld]) - float(n[fld])))
    return {"n_cells_compared": n_cmp, "n_mismatched_fields": len(mism), "mismatches": mism[:20],
            "max_abs_mean_diff": mean_diff, "exact": (n_cmp > 0 and not mism)}


def nr_seed0(PR, reads_by_surf: dict) -> tuple:
    per = {}
    for fam, key, label, unit, lower, surfs in PR.METRICS:
        for surf in surfs:
            c = reads_by_surf[surf]["rows"][key]["os_minus_refcv7_50400"]
            sw = bool(c["lo"] > 0)
            per[f"{surf}:{key}"] = {"family": fam, "label": label, "unit": unit, "delta": c["delta"], "lo": c["lo"], "hi": c["hi"],
                                    "mean_refcv8": c["mean_new"], "mean_refcv7": c["mean_ref"], "n_windows": c["n_windows"],
                                    "n_episodes": c["n_episodes"], "separated_worse": sw,
                                    "verdict": "FAIL (separated worse, seed 0)" if sw else "PASS (not separated worse, seed 0)"}
    fam_of = {k: v["family"] for k, v in per.items()}
    return per, PR.rollup_NR({k: {"verdict": v["verdict"]} for k, v in per.items()}, fam_of)


def headline_deltas(PR, read_by_surf: dict) -> dict:
    out = {}
    for fam, key, label, unit, lower, surfs in PR.METRICS:
        for surf in surfs:
            out[f"{surf}:{key}"] = read_by_surf[surf]["rows"][key]["os_minus_refcv7_50400"]["delta"]
    return out


def read_tree(tree: str, n_boot: int, force: bool) -> int:
    out_json = OUT / "trees" / f"{tree}.json"
    if out_json.exists() and not force:
        print(f"[{tree}] already read ({out_json}); --force to redo")
        return 0
    import r8_battery_patch as BP
    L, P = BP.apply()
    ra = P.RR.ra3().ra
    sha12 = L.sha12
    import pair_read_r8 as PR
    t0 = time.time()
    root = os.path.join(ARCHIVE, tree)
    banked = jload(os.path.join(root, f"PAIRED_READ_{tree}.json"))
    r7base = BP.R7_BASE_ROOT
    ex = load_extras(os.path.join(root, "dump_s0"))
    if not ex["has_filter_off"]:
        raise SystemExit(f"[{tree}] no filter_off arm in the seed-0 extras")
    work = os.path.join(SCRATCH, tree)
    res = {"tool": "nr_matched_instrument.py", "tree": tree, "tier": TIER, "estimator": QUESTION, "inference_seed": SEED,
           "n_boot": n_boot, "bootstrap_seed": PR.SEED, "reference_arm": REF_ARM, "generated": time.strftime("%FT%T"),
           "banked_read": {"file": os.path.join(root, f"PAIRED_READ_{tree}.json"), "finished": banked.get("finished"),
                           "label": banked.get("label")},
           "surfaces": {}, "controls": {}}
    reads_off, reads_on = {}, {}
    cells_shift = {}
    ctlA_ok = True
    for surf in ("S2", "S6"):
        pdir = os.path.join(root, PANEL[surf])
        man, files, ep_of = panel_layout(pdir)
        slots = list(man["grid"]["slots"])
        bw = BP.windows_of(os.path.join(r7base, BASELINE_DUMP[surf]))
        bseed = banked["surfaces"][surf]["seeds"][str(SEED)]
        wd = os.path.join(work, surf)
        on_x = plans_from_extras(ex, ep_of, files, "nav_true.traj", slots)
        off = plans_from_extras(ex, ep_of, files, "filter_off.traj", slots)
        own = own_os(files)
        ctl = {"os_vs_extras_nav_true_slots_max_abs": max_abs(own, on_x)}
        # A1: os substituted by itself (writer / reader round trip)
        write_variant(pdir, wd, files, own)
        rA1 = PR.read_surface(ra, sha12, wd, REF_ARM, n_boot, bw)
        ctl["A1_identity_panel_os"] = compare_to_banked(rA1, bseed)
        # A2: os substituted by the extras' nav_true.traj through the OFF arm's own matching path
        write_variant(pdir, wd, files, on_x)
        rA2 = PR.read_surface(ra, sha12, wd, REF_ARM, n_boot, bw)
        ctl["A2_identity_extras_nav_true"] = compare_to_banked(rA2, bseed)
        ctlA_ok = ctlA_ok and ctl["A1_identity_panel_os"]["exact"] and ctl["A2_identity_extras_nav_true"]["exact"]
        reads_on[surf] = rA1
        res["surfaces"][surf] = {"grid_slots": slots, "n_windows": rA1["n_windows"], "n_episodes": rA1["n_episodes"],
                                 "windows_check_vs_refcv7": rA1.get("windows_check")}
        res["controls"][surf] = ctl
        if not ctlA_ok:
            res["controls"]["A_identity_PASS"] = False
            jdump(res, str(OUT / "trees" / f"{tree}.CONTROL_A_FAILED.json"))
            print(f"[{tree}] CONTROL (A) FAILED on {surf}: {json.dumps(ctl, default=str)[:1500]}")
            return 3
        # the matched read
        write_variant(pdir, wd, files, off)
        rOFF = PR.read_surface(ra, sha12, wd, REF_ARM, n_boot, bw)
        reads_off[surf] = rOFF
        # paired OFF - ON on refcv8's own metric (same windows, same bootstrap draws, same cell function)
        man_o, A_o, eid_o, keys_o = PR.load_panel_keyed(wd, sha12)
        man_n, A_n, eid_n, keys_n = PR.load_panel_keyed(pdir, sha12)
        if keys_o != keys_n:
            raise SystemExit(f"[{tree}] {surf}: OFF panel and banked panel window order differ")
        dt = float(man_o["grid"]["dt_s"])
        c_off = PR.components(ra, A_o["os"], A_o["g"], dt)
        c_on = PR.components(ra, A_n["os"], A_n["g"], dt)
        cells_shift[surf] = {key: PR.paired_cell(c_off[key], c_on[key], eid_o, n_boot) for _f, key, *_r in PR.METRICS}
        # B: mispairings
        cB = {}
        for name, shift in (("B_ep", MIS_EP_ROLL), ("B_win", 1)):
            write_variant(pdir, wd, files, roll_plans(off, shift))
            cB[name] = PR.read_surface(ra, sha12, wd, REF_ARM, n_boot, bw)
        res["surfaces"][surf]["read_OFF"] = rOFF
        res["surfaces"][surf]["read_ON_rewritten"] = {"rows_os_minus_refcv7_50400": {k: v.get("os_minus_refcv7_50400") for k, v in rA1["rows"].items()}}
        res["surfaces"][surf]["paired_OFF_minus_ON_refcv8_own_metric"] = cells_shift[surf]
        res["surfaces"][surf]["mispaired_rows"] = {n: {k: v.get("os_minus_refcv7_50400") for k, v in x["rows"].items()} for n, x in cB.items()}
        print(f"[{tree}] {surf} done {time.time() - t0:.0f}s", flush=True)
        res.setdefault("_tmp", {})[surf] = cB
    res["controls"]["A_identity_PASS"] = bool(ctlA_ok)
    # control B verdicts (pre-registered criteria)
    hd_off = headline_deltas(PR, reads_off)
    ade_off = reads_off["S2"]["rows"]["ade_m"]["os_minus_refcv7_50400"]["mean_new"]
    cb = {}
    for name in ("B_ep", "B_win"):
        hd = headline_deltas(PR, {s: res["_tmp"][s][name] for s in ("S2", "S6")})
        changed = {k: (hd[k] != hd_off[k]) for k in hd_off}
        ade_m = res["_tmp"]["S2"][name]["rows"]["ade_m"]["os_minus_refcv7_50400"]["mean_new"]
        cb[name] = {"n_headline_cells": len(changed), "n_changed": int(sum(changed.values())), "changed": changed,
                    "s2_ade_refcv8_mean_matched": ade_off, "s2_ade_refcv8_mean_mispaired": ade_m, "s2_ade_rise": ade_m - ade_off}
    cb["B_ep"]["PASS"] = bool(cb["B_ep"]["n_changed"] == cb["B_ep"]["n_headline_cells"] and cb["B_ep"]["s2_ade_rise"] >= MIS_EP_ADE_RISE_MIN)
    cb["B_win"]["PASS"] = bool(cb["B_win"]["n_changed"] >= 1)
    res["controls"]["B_mispairing"] = cb
    res["controls"]["B_PASS"] = bool(cb["B_ep"]["PASS"] and cb["B_win"]["PASS"])
    del res["_tmp"]
    # NR, seed 0 only: ON (the banked rows) and OFF (matched)
    per_on, fam_on = nr_seed0(PR, {s: banked["surfaces"][s]["seeds"][str(SEED)] for s in ("S2", "S6")})
    per_off, fam_off = nr_seed0(PR, reads_off)
    res["NR_seed0"] = {"ON_banked": {"per_metric": per_on, "per_family": fam_on},
                       "OFF_matched": {"per_metric": per_off, "per_family": fam_off},
                       "banked_both_seed_per_family": banked["bars"]["BAR-R8-NR"]["per_family"],
                       "rule": "error metric, refcv8 - refcv7-50,400: FAIL iff CI lower > 0 (separated worse); seed 0 only -- seed 1 has no "
                               "filter_off arm (os-only roll), so a both-seed matched verdict is NOT EVALUABLE from banked data"}
    # TACTICAL (no NR headline in pair_read_r8.METRICS): the rows, both instruments
    res["TACTICAL_rows"] = {surf: {key: {"ON": reads_on[surf]["rows"][key]["os_minus_refcv7_50400"],
                                         "OFF": reads_off[surf]["rows"][key]["os_minus_refcv7_50400"]}
                                   for fam, key, *_r in PR.METRICS if fam == "TACTICAL"} for surf in ("S2", "S6")}
    res["wall_s"] = round(time.time() - t0, 1)
    ok = res["controls"]["A_identity_PASS"] and res["controls"]["B_PASS"]
    res["controls"]["ALL_PASS"] = bool(ok)
    jdump(res, str(out_json))
    print(f"[{tree}] WROTE {out_json}  A={res['controls']['A_identity_PASS']} B={res['controls']['B_PASS']}  {res['wall_s']}s", flush=True)
    return 0 if ok else 4


# ------------------------------------------------------------------------------------------ task 3: assemble
def fmt_cell(c: dict) -> str:
    return f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"


def assemble() -> int:
    trees = {}
    missing = []
    for t in TREES:
        p = OUT / "trees" / f"{t}.json"
        if not p.exists():
            missing.append(t)
            continue
        trees[t] = jload(str(p))
    if missing:
        raise SystemExit(f"[assemble] per-tree reads missing: {missing}")
    bad = [t for t, r in trees.items() if not r["controls"].get("ALL_PASS")]
    if bad:
        raise SystemExit(f"[assemble] controls did not all pass for {bad}: refusing to assemble")
    probe = jload(str(OUT / "S6_FILTER_OFF_PROBE.json")) if (OUT / "S6_FILTER_OFF_PROBE.json").exists() else None
    fams = ["ADE", "LONGITUDINAL", "LATERAL"]
    summary = {"tool": "nr_matched_instrument.py assemble", "generated": time.strftime("%FT%T"), "tier": TIER, "estimator": QUESTION,
               "inference_seed": SEED, "n_boot": N_BOOT,
               "instrument": ("ON = refcv8 `os` as banked (max-speed ceiling filter ACTING, as every refcv8 battery was rolled); "
                              "OFF = the SAME window, same DDIM draw, ceiling filter OFF (`filter_off.traj`) = the instrument the banked "
                              "refcv7-50,400 reference was rolled with (its filter was INERT: final selection == filter-off, re-verified in the probe)"),
               "per_tree": {}, "verdict_changes": [], "controls": {t: {"A": r["controls"]["A_identity_PASS"], "B": r["controls"]["B_PASS"]} for t, r in trees.items()}}
    for t, r in trees.items():
        nr = r["NR_seed0"]
        entry = {"banked_both_seed": nr["banked_both_seed_per_family"], "ON_seed0": nr["ON_banked"]["per_family"],
                 "OFF_seed0": nr["OFF_matched"]["per_family"], "families": {}, "headline_rows": {}}
        for f in fams:
            on0, off0, both = nr["ON_banked"]["per_family"].get(f), nr["OFF_matched"]["per_family"].get(f), nr["banked_both_seed_per_family"].get(f)
            entry["families"][f] = {"banked_both_seeds": both, "ON_seed0": on0, "OFF_seed0_matched": off0,
                                    "seed0_verdict_changes_on_to_off": (on0 != off0)}
            if on0 != off0:
                summary["verdict_changes"].append({"tree": t, "family": f, "ON_seed0": on0, "OFF_seed0": off0, "banked_both_seeds": both})
        for k, v_on in nr["ON_banked"]["per_metric"].items():
            v_off = nr["OFF_matched"]["per_metric"][k]
            surf, key = k.split(":")
            sh = r["surfaces"][surf]["paired_OFF_minus_ON_refcv8_own_metric"][key]
            entry["headline_rows"][k] = {"family": v_on["family"], "ON": {x: v_on[x] for x in ("delta", "lo", "hi", "separated_worse", "verdict")},
                                         "OFF": {x: v_off[x] for x in ("delta", "lo", "hi", "separated_worse", "verdict")},
                                         "shift_delta_OFF_minus_ON": round(v_off["delta"] - v_on["delta"], 4),
                                         "paired_OFF_minus_ON_refcv8_own_metric": {x: sh.get(x) for x in ("delta", "lo", "hi", "separated", "mean_new", "mean_ref")},
                                         "n_windows": v_on["n_windows"], "n_episodes": v_on["n_episodes"],
                                         "refcv8_mean_ON": v_on["mean_refcv8"], "refcv8_mean_OFF": v_off["mean_refcv8"], "refcv7_mean": v_on["mean_refcv7"]}
        # MIXED both-seed reading: seed 0 on the matched (OFF) instrument + seed 1 as banked (ON; seed 1 has no filter_off arm).  This is NOT a
        # matched verdict; it shows what the both-seed bar can and cannot say.  Seed-1 rows come straight from the banked PAIRED_READ.
        import pair_read_r8 as PR
        banked = jload(os.path.join(ARCHIVE, t, f"PAIRED_READ_{t}.json"))
        mixed = {}
        for k, h in entry["headline_rows"].items():
            surf, key = k.split(":")
            c1 = banked["surfaces"][surf]["seeds"]["1"]["rows"][key]["os_minus_refcv7_50400"]
            h["seed1_ON_banked"] = {x: c1[x] for x in ("delta", "lo", "hi")}
            h["seed1_ON_banked"]["separated_worse"] = bool(c1["lo"] > 0)
            fail = bool(h["OFF"]["separated_worse"] or h["seed1_ON_banked"]["separated_worse"])
            mixed[k] = {"verdict": "FAIL (separated worse at seed 0 OFF or seed 1 ON)" if fail else "PASS", "family": h["family"]}
            h["mixed_verdict"] = mixed[k]["verdict"].split(" ")[0]
            h["banked_both_seed_verdict"] = ("FAIL" if (h["ON"]["separated_worse"] or h["seed1_ON_banked"]["separated_worse"]) else "PASS")
            h["borderline_lo_le_2e-4"] = bool(abs(h["ON"]["lo"]) <= 2e-4 or abs(h["OFF"]["lo"]) <= 2e-4)
        recon = PR.rollup_NR({k: {"verdict": h["banked_both_seed_verdict"]} for k, h in entry["headline_rows"].items()},
                             {k: h["family"] for k, h in entry["headline_rows"].items()})
        # the rollup strings name the failing metrics; compare the PASS/FAIL set per family against the banked rollup
        for f in fams:
            if (recon.get(f, "").split(":")[0]) != (banked["bars"]["BAR-R8-NR"]["per_family"].get(f, "").split(":")[0]) or                     sorted(recon.get(f, "").split(": ")[-1].split(", ")) != sorted(banked["bars"]["BAR-R8-NR"]["per_family"].get(f, "").split(": ")[-1].split(", ")):
                raise SystemExit(f"[assemble] {t} {f}: reconstructed banked both-seed verdict {recon.get(f)!r} != banked {banked['bars']['BAR-R8-NR']['per_family'].get(f)!r}")
        entry["banked_both_seed_reconstruction_equals_banked"] = True
        entry["MIXED_seed0_OFF_plus_seed1_ON_per_family"] = PR.rollup_NR(mixed, {k: v["family"] for k, v in mixed.items()})
        for f in fams:
            entry["families"][f]["mixed_seed0_OFF_seed1_ON"] = entry["MIXED_seed0_OFF_plus_seed1_ON_per_family"].get(f)
        entry["TACTICAL_rows"] = r["TACTICAL_rows"]
        summary["per_tree"][t] = entry
    summary["metric_level_changes_seed0"] = [
        {"tree": t, "metric": k, "family": h["family"], "ON": h["ON"]["verdict"].split(" ")[0], "OFF": h["OFF"]["verdict"].split(" ")[0],
         "ON_cell": [h["ON"]["delta"], h["ON"]["lo"], h["ON"]["hi"]], "OFF_cell": [h["OFF"]["delta"], h["OFF"]["lo"], h["OFF"]["hi"]],
         "banked_both_seed_verdict": h["banked_both_seed_verdict"], "mixed_verdict": h["mixed_verdict"]}
        for t, e in summary["per_tree"].items() for k, h in e["headline_rows"].items()
        if h["ON"]["verdict"].split(" ")[0] != h["OFF"]["verdict"].split(" ")[0]]
    shifts = [h["shift_delta_OFF_minus_ON"] for e in summary["per_tree"].values() for h in e["headline_rows"].values()]
    summary["shift_sign_summary"] = {"n_rows": len(shifts), "n_negative(OFF better than ON)": int(sum(x < 0 for x in shifts)),
                                     "n_zero": int(sum(x == 0 for x in shifts)), "n_positive": int(sum(x > 0 for x in shifts))}
    jdump(summary, str(OUT / "NR_MATCHED_INSTRUMENT.json"))
    open(str(OUT / "NR_MATCHED_INSTRUMENT.md"), "w", encoding="utf-8").write(render_md(summary, trees, probe))
    print("WROTE", OUT / "NR_MATCHED_INSTRUMENT.json", OUT / "NR_MATCHED_INSTRUMENT.md")
    return 0


def lat_survival(summary: dict) -> list:
    rows = []
    for t in ("step10500_unc", "step10500ema_unc", "step15000_unc", "step15000ema_unc", "step20000_unc", "step20000ema_unc"):
        e = summary["per_tree"][t]
        for key in ("S2:LAT_heading_mae_deg", "S2:LAT_curvature_mae_1pm_valid", "S2:LAT_yaw_rate_mae_radps_valid"):
            h = e["headline_rows"][key]
            rows.append((t, key.split(":")[1], h["ON"], h["OFF"]))
    return rows


def render_md(S: dict, trees: dict, probe) -> str:
    L = ["# BAR-R8-NR on the MATCHED instrument (refcv8 ceiling filter OFF), inference seed 0",
         "",
         f"* tier: **{S['tier']}**; estimator: {S['estimator']}; B {S['n_boot']}, bootstrap seed 0, 95 % percentile; inference seed **0 only**.",
         f"* instrument: {S['instrument']}.",
         "* reference: `b_refcv7_50400_s0` (the banked refcv7-50,400 `os`, embedded in every panel). Rows are refcv8 minus refcv7; error metrics, negative = refcv8 better.",
         "* NR rule (pair_read_r8.bar_NR_metric): a headline is FAIL iff the paired CI **lower bound > 0** (separated worse). The banked verdict uses BOTH inference seeds; the matched one can only use seed 0 "
         "(seed-1 refcv8 rolls are os-only: no `filter_off` arm). So three columns are shown: banked both-seed, banked seed-0-only (ON), matched seed-0-only (OFF). The instrument effect is **ON seed 0 vs OFF seed 0**.",
         "* A separated CI from one-seed arms is necessary, not sufficient (H-ESTIM-SEED-1); the inference-seed floor is not re-estimated here. Single training seed; EARLY reads of a 48,852-step run.",
         "", "## Controls (all mandatory; per tree)", "",
         "| tree | (A) identity: `os`->`os` reproduces banked rows exactly (S2 / S6; A1 panel os, A2 extras nav_true) | (B) mispairing changes rows |", "|---|---|---|"]
    for t, r in trees.items():
        c = r["controls"]
        a = []
        for surf in ("S2", "S6"):
            a1, a2 = c[surf]["A1_identity_panel_os"], c[surf]["A2_identity_extras_nav_true"]
            a.append(f"{surf}: A1 {a1['n_cells_compared']} cells, {a1['n_mismatched_fields']} mismatch, max|mean diff| {a1['max_abs_mean_diff']:.1e}; "
                     f"A2 {a2['n_cells_compared']} cells, {a2['n_mismatched_fields']} mismatch, {a2['max_abs_mean_diff']:.1e}")
        b = c["B_mispairing"]
        L.append(f"| {t} | {' ; '.join(a)} | B_ep {b['B_ep']['n_changed']}/{b['B_ep']['n_headline_cells']} cells changed, S2 ADE +{b['B_ep']['s2_ade_rise']:.3f} m "
                 f"({'PASS' if b['B_ep']['PASS'] else 'FAIL'}); B_win {b['B_win']['n_changed']}/{b['B_win']['n_headline_cells']} ({'PASS' if b['B_win']['PASS'] else 'FAIL'}) |")
    L += ["", "## Per-family NR verdicts: banked (ceiling ON) vs matched (ceiling OFF)", "",
          "| tree | family | banked, both seeds (ON) | seed 0 only, ON | seed 0 only, OFF (matched) | changes ON->OFF at seed 0? | MIXED: seed 0 OFF + seed 1 ON (not matched) |", "|---|---|---|---|---|---|---|"]
    for t, e in S["per_tree"].items():
        for f in ("ADE", "LONGITUDINAL", "LATERAL"):
            x = e["families"][f]
            L.append(f"| {t} | {f} | {x['banked_both_seeds']} | {x['ON_seed0']} | {x['OFF_seed0_matched']} | {'**YES**' if x['seed0_verdict_changes_on_to_off'] else 'no'} | {x['mixed_seed0_OFF_seed1_ON']} |")
    L += ["", "TACTICAL: `pair_read_r8.METRICS` carries NO NR headline for the TACTICAL family (TAC_* rows have an empty headline-surface tuple), so BAR-R8-NR has no TACTICAL verdict "
          "under either instrument (NOT EVALUABLE as coded, not a pass); the TAC rows are shown below.",
          "", "### Verdict changes at seed 0 (ON -> OFF)", ""]
    if S["verdict_changes"]:
        L += ["| tree | family | seed 0 ON | seed 0 OFF | banked both-seed |", "|---|---|---|---|---|"]
        for v in S["verdict_changes"]:
            L.append(f"| {v['tree']} | {v['family']} | {v['ON_seed0']} | {v['OFF_seed0']} | {v['banked_both_seeds']} |")
    else:
        L.append("None: no per-family seed-0 NR verdict changes between the ceiling-ON and ceiling-OFF instrument.")
    L += ["", "### Metric-level headline verdict changes at seed 0 (ON -> OFF)", ""]
    if S["metric_level_changes_seed0"]:
        L += ["| tree | metric | ON seed 0 delta [CI] | verdict | OFF seed 0 delta [CI] | verdict | banked both-seed | mixed (s0 OFF + s1 ON) |", "|---|---|---|---|---|---|---|---|"]
        for v in S["metric_level_changes_seed0"]:
            o, f_ = v["ON_cell"], v["OFF_cell"]
            L.append(f"| {v['tree']} | {v['metric']} | {o[0]:+.4f} [{o[1]:+.4f}, {o[2]:+.4f}] | {v['ON']} | {f_[0]:+.4f} [{f_[1]:+.4f}, {f_[2]:+.4f}] | {v['OFF']} | {v['banked_both_seed_verdict']} | {v['mixed_verdict']} |")
    else:
        L.append("None.")
    ss = S["shift_sign_summary"]
    L += ["", f"Sign of the instrument shift (OFF delta minus ON delta, all {ss['n_rows']} NR headline rows over 8 trees): negative {ss['n_negative(OFF better than ON)']}, zero {ss['n_zero']}, positive {ss['n_positive']} "
          "(negative = removing the ceiling filter moves refcv8 closer to / better than refcv7-50,400).", ""]
    L += ["", "## LATERAL heading / curvature / yaw-rate rows at 10.5k-20k, S2 0-2 s (does the FAIL survive?)", "",
          "| tree | metric | ON seed 0: delta [CI] | verdict | OFF seed 0: delta [CI] | verdict | shift (OFF-ON) |", "|---|---|---|---|---|---|---|"]
    for t in ("step10500_unc", "step10500ema_unc", "step15000_unc", "step15000ema_unc", "step20000_unc", "step20000ema_unc"):
        for key in ("S2:LAT_heading_mae_deg", "S2:LAT_curvature_mae_1pm_valid", "S2:LAT_yaw_rate_mae_radps_valid"):
            h = S["per_tree"][t]["headline_rows"][key]
            L.append(f"| {t} | {key.split(':')[1]} | {fmt_cell(h['ON'])} | {h['ON']['verdict'].split(' ')[0]} | {fmt_cell(h['OFF'])} | {h['OFF']['verdict'].split(' ')[0]} | {h['shift_delta_OFF_minus_ON']:+.4f} |")
    L += ["", "## All NR headline rows, per tree (refcv8 - refcv7-50,400; ON = banked, OFF = matched; seed 0)", "",
          "| tree | family | surface:metric | refcv8 mean ON / OFF | refcv7 mean | ON delta [CI] | OFF delta [CI] | shift | paired OFF-ON on refcv8 [CI] | ON verdict | OFF verdict | n win / ep |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t, e in S["per_tree"].items():
        for k, h in e["headline_rows"].items():
            p = h["paired_OFF_minus_ON_refcv8_own_metric"]
            pc = f"{p['delta']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]{' SEP' if p['separated'] else ''}" if p.get("delta") is not None else "n/a"
            L.append(f"| {t} | {h['family']} | {k} | {h['refcv8_mean_ON']:.5g} / {h['refcv8_mean_OFF']:.5g} | {h['refcv7_mean']:.5g} | {fmt_cell(h['ON'])} | {fmt_cell(h['OFF'])} | "
                     f"{h['shift_delta_OFF_minus_ON']:+.4f} | {pc} | {h['ON']['verdict'].split(' ')[0]} | {h['OFF']['verdict'].split(' ')[0]} | {h['n_windows']}/{h['n_episodes']} |")
    L += ["", "## TACTICAL rows (trajectory-derived decisions; no NR headline), S2 and S6, ON vs OFF (refcv8 - refcv7-50,400; higher is better)", "",
          "| tree | surface | metric | ON delta [CI] | sep. | OFF delta [CI] | sep. |", "|---|---|---|---|---|---|---|"]
    for t, e in S["per_tree"].items():
        for surf, rows in e["TACTICAL_rows"].items():
            for key, v in rows.items():
                L.append(f"| {t} | {surf} | {key} | {fmt_cell(v['ON'])} | {v['ON']['separated']} | {fmt_cell(v['OFF'])} | {v['OFF']['separated']} |")
    if probe:
        L += ["", "## Where the filter-OFF plan lives (probe: `S6_FILTER_OFF_PROBE.json`)", "",
              "* **S2** (`panel_s0`, slots 0-3 = 0.5-2.0 s): the panel carries an `os_filteroff` arm AND `dump_s0/refcv6_extras.npz :: filter_off.traj` [N, 8, 2] holds the full plan. "
              "This tool reads the extras (not the panel arm) and uses the panel arm only as a cross-check (max abs difference below).",
              "* **S6** (`s6_s0`, slots [1,3,4,5,6,7] = 1-6 s): the S6 panel and its `decisions/` carry NO filter-off arm (only `plan_full_nav_{true,shuffled,flipped,zero}`), and there is no S6 extras file. "
              "The S6 filter-OFF plan is DERIVED: `filter_off.traj[row][:, [1,3,4,5,6,7]]` of the S2 extras, rows matched on (episode_index, ws). The ON-side of that derivation is verified bit-exact "
              "(S6 `os` == `nav_true.traj[:, slots]`, and `decisions/plan_full_nav_true` == `nav_true.traj`, max abs 0.0 in all 8 trees); the OFF side uses the same row map and slot map but "
              "has never been rolled on its own (UNVERIFIED beyond that identity).",
              "* **Seed 1**: `dump_s1/refcv6_extras.npz` has 20 keys and NO `filter_off.*` (os-only roll): no matched seed-1 read is possible from banked data.",
              "* **refcv7-50,400 side**: its dump_s0 extras read `nav_true.sel_idx != filter_off.sel_idx` in 0 of 4,754 windows and ON-vs-OFF plan max abs 0.0 (re-verified here; the filter was inert).",
              "", "| tree | S2 windows where the filter changes the selected anchor | S6 windows (of 3,668) | S2 panel `os_filteroff` vs extras max abs | S6 `os` vs extras `nav_true` slots max abs | ON==OFF plan where the anchor is unchanged (max abs) |",
              "|---|---|---|---|---|---|"]
        for t, r in probe["trees"].items():
            a, b = r["S2"], r["S6"]
            L.append(f"| {t} | {a['n_windows_filter_changes_selected_anchor']} / {a['n_windows']} ({a['frac_windows_filter_changes_selected_anchor']:.3f}) | "
                     f"{b['n_windows_filter_changes_selected_anchor']} ({b['frac_windows_filter_changes_selected_anchor']:.3f}) | {a.get('panel_os_filteroff_vs_extras_filter_off_max_abs')} | "
                     f"{b['panel_os_vs_extras_nav_true_slots_max_abs']} | {max(a['max_abs_on_vs_off_plan_where_selected_anchor_equal'], b['max_abs_on_vs_off_plan_where_selected_anchor_equal'])} |")
        L += ["", "Independent numpy cross-check of the matched ADE (no torch, no pair_read_r8): `ADE_CROSSCHECK.json` (worst |mean diff| 1e-8 m over 8 trees x 2 surfaces)."]
    L += ["", "Cells whose CI lower bound is within 2e-4 of zero are flagged `borderline_lo_le_2e-4` in the JSON; `taniteval.ci` renders bounds at adaptive precision so the "
          "published sign equals the unrounded test."]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("probe", "read", "assemble"))
    ap.add_argument("--tree", default=None, help="one of %s (read stage)" % TREES)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[nr_matched] set CUDA_VISIBLE_DEVICES=-1 (a GPU battery is running; this tool is CPU only)")
    if a.stage == "probe":
        import r8_battery_patch as BP
        res = {"tool": "nr_matched_instrument.py probe", "generated": time.strftime("%FT%T"), "trees": {}}
        for t in TREES:
            res["trees"][t] = probe_tree(t, BP.R7_BASE_ROOT)
            print(t, "probed", flush=True)
        jdump(res, str(OUT / "S6_FILTER_OFF_PROBE.json"))
        print("WROTE", OUT / "S6_FILTER_OFF_PROBE.json")
        return 0
    if a.stage == "read":
        if a.tree not in TREES:
            raise SystemExit(f"--tree must be one of {TREES}")
        return read_tree(a.tree, a.n_boot, a.force)
    return assemble()


if __name__ == "__main__":
    sys.exit(main())
