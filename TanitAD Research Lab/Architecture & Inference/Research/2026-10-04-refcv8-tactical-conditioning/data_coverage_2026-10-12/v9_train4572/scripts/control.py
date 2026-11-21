"""CONTROL: new v9 train release (4,572 clips) vs the OLD release (4,369 clips), by clip (sid), row-for-row, every array.
Also: clip-id coverage, eval139 disjointness, ADD coverage.  Writes a JSON verdict + prints a text report.

usage: control.py <old.npz> <new.npz> <out.json>
"""
import sys, json
import numpy as np, torch

OLD, NEW = sys.argv[1], sys.argv[2]
OUTJ = sys.argv[3]
A = "C:/Users/Admin/tanitad-data/add203/"
sys.path.insert(0, "C:/Users/Admin/qland/hold_v9train4572/work/patched")
import build_v9_labels as B

zo = np.load(OLD, allow_pickle=False)
zn = np.load(NEW, allow_pickle=False)
rep = {"old": OLD, "new": NEW}
fo, fn = sorted(zo.files), sorted(zn.files)
rep["fields_old"], rep["fields_new"] = len(fo), len(fn)
rep["field_sets_equal"] = fo == fn
rep["fields_only_old"] = sorted(set(fo) - set(fn))
rep["fields_only_new"] = sorted(set(fn) - set(fo))
rep["dtype_differs"] = [f for f in fo if f in zn.files and (zo[f].dtype != zn[f].dtype)]

# manifests (clip universe)
mn = torch.load(A + "in/train4572_v2manifest.pt", map_location="cpu", weights_only=False)
me = torch.load(A + "in/eval139_v2manifest.pt", map_location="cpu", weights_only=False)
cn = [str(c) for c in mn["clip_id"]]
ce = [str(c) for c in me["clip_id"]]
add = json.load(open("add_ids.json"))["add"]
sid_man = np.array([B.stable_episode_id(c) for c in cn])
s12_man = np.array([B.sha12(c) for c in cn])
sid_eval = {B.stable_episode_id(c) for c in ce}
s12_eval = {B.sha12(c) for c in ce}
sid_add = np.array([B.stable_episode_id(c) for c in add])
rep["n_clips_new"] = int(len(zn["clip__sid"]))
rep["n_clips_old"] = int(len(zo["clip__sid"]))
rep["new_sid_set_equals_manifest4572"] = (set(map(int, zn["clip__sid"])) == set(map(int, sid_man))) and len(set(zn["clip__sid"])) == len(zn["clip__sid"])
rep["new_sha12_set_equals_manifest4572"] = set(zn["clip__sha12"].astype(str)) == set(s12_man)
rep["n_eval139_sid_in_new"] = int(len(set(map(int, zn["clip__sid"])) & sid_eval))
rep["n_eval139_sha12_in_new"] = int(len(set(zn["clip__sha12"].astype(str)) & s12_eval))
rep["n_eval139"] = len(sid_eval)
rep["old_sids_subset_of_new"] = set(map(int, zo["clip__sid"])) <= set(map(int, zn["clip__sid"]))
rep["add_sids_in_new"] = int(len(set(map(int, sid_add)) & set(map(int, zn["clip__sid"]))))
rep["add_sids_in_old"] = int(len(set(map(int, sid_add)) & set(map(int, zo["clip__sid"]))))

# row index maps by sid
so, ro0, rn = zo["clip__sid"].astype(np.int64), zo["clip__row0"].astype(np.int64), zo["clip__n_rows"].astype(np.int64)
sn, nr0, nn = zn["clip__sid"].astype(np.int64), zn["clip__row0"].astype(np.int64), zn["clip__n_rows"].astype(np.int64)
pos_new = {int(s): i for i, s in enumerate(sn)}
rep["old_row0_contiguous"] = bool(np.all(ro0[1:] == (ro0 + rn)[:-1]) and ro0[0] == 0 and (ro0[-1] + rn[-1]) == len(zo["row__k"]))
rep["new_row0_contiguous"] = bool(np.all(nr0[1:] == (nr0 + nn)[:-1]) and nr0[0] == 0 and (nr0[-1] + nn[-1]) == len(zn["row__k"]))
rep["n_rows_old"], rep["n_rows_new"] = int(len(zo["row__k"])), int(len(zn["row__k"]))
assert rep["old_sids_subset_of_new"]
npos = np.array([pos_new[int(s)] for s in so])
rep["old_clip_n_rows_equal_new"] = bool(np.array_equal(rn, nn[npos]))
rep["new_clip_order_equals_manifest_order"] = bool(np.array_equal(sn, sid_man))
idx_new = np.concatenate([np.arange(nr0[p], nr0[p] + nn[p]) for p in npos])
assert len(idx_new) == len(zo["row__k"])
clip_of_old_row = np.repeat(np.arange(len(so)), rn)


def row_mismatch(a, b):
    """per-row BITWISE mismatch mask (NaN payloads included)."""
    ab = np.ascontiguousarray(a).view(np.uint8).reshape(len(a), -1)
    bb = np.ascontiguousarray(b).view(np.uint8).reshape(len(b), -1)
    return (ab != bb).any(axis=1)


fieldres = {}
differing_clip = np.zeros(len(so), bool)
for f in fo:
    if not f.startswith("row__"):
        continue
    a, b = zo[f], zn[f][idx_new]
    mism = row_mismatch(a, b)
    valeq = bool(np.array_equal(a, b, equal_nan=True)) if a.dtype.kind == "f" else bool(np.array_equal(a, b))
    nbad = int(mism.sum())
    fieldres[f] = {"rows_bitwise_differ": nbad, "value_equal_nan_eq": valeq}
    if nbad:
        differing_clip[np.unique(clip_of_old_row[mism])] = True
        if a.dtype.kind == "f":
            m2 = ~((a == b) | (np.isnan(a) & np.isnan(b)))
            fieldres[f]["rows_value_differ"] = int(m2.sum())
            if m2.any():
                fieldres[f]["max_abs_diff"] = float(np.nanmax(np.abs(a[m2] - b[m2])))
for f in fo:
    if f.startswith("clip__") and f != "clip__row0":
        a, b = zo[f], zn[f][npos]
        if a.dtype.kind == "f":
            bad = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        else:
            bad = a != b
        fieldres[f] = {"clips_differ": int(bad.sum())}
        differing_clip |= bad
rep["fields_compared_row"] = sum(1 for f in fieldres if f.startswith("row__"))
rep["fields_compared_clip"] = sum(1 for f in fieldres if f.startswith("clip__"))
rep["n_old_clips_compared"] = int(len(so))
rep["n_old_rows_compared"] = int(len(idx_new))
rep["n_old_clips_differing"] = int(differing_clip.sum())
rep["fields_with_any_difference"] = sorted(f for f, r in fieldres.items() if (r.get("rows_bitwise_differ", 0) or r.get("clips_differ", 0)))
rep["field_results"] = fieldres

# ADD coverage
is_add_clip = np.isin(sn, sid_add)
rep["n_add_clips"] = int(is_add_clip.sum())
rep["n_add_rows"] = int(nn[is_add_clip].sum())
add_rows = np.concatenate([np.arange(nr0[i], nr0[i] + nn[i]) for i in np.nonzero(is_add_clip)[0]])
cov = {}
for f in fn:
    if not f.startswith("row__"):
        continue
    for tag, rows in (("add", add_rows), ("old", idx_new)):
        v = zn[f][rows]
        if v.dtype.kind == "f":
            cov.setdefault(f, {})[tag] = {"nonNaN": round(float(np.mean(~np.isnan(v))), 4)}
        else:
            cov.setdefault(f, {})[tag] = {"IGNORE": round(float(np.mean(v == B.IGNORE)), 4) if v.dtype.kind == "i" else None,
                                          "eq0": round(float(np.mean(v == 0)), 4), "gt0": round(float(np.mean(v > 0)), 4)}
rep["coverage"] = cov
cl = {}
for f in ("clip__has_record", "clip__has_lead", "clip__has_lc", "clip__n_turns", "clip__n_announced", "clip__suppressed"):
    v = zn[f]
    cl[f] = {"add_sum": int(v[is_add_clip].sum()), "add_n": int(is_add_clip.sum()),
             "old_sum": int(v[~is_add_clip].sum()), "old_n": int((~is_add_clip).sum())}
rep["clip_level_add_vs_old"] = cl
rep["clock_src_counts_add"] = {int(k): int((zn["clip__clock_src"][is_add_clip] == k).sum()) for k in np.unique(zn["clip__clock_src"])}
rep["clock_src_counts_old_in_new"] = {int(k): int((zn["clip__clock_src"][~is_add_clip] == k).sum()) for k in np.unique(zn["clip__clock_src"])}
rep["now_s_all_finite_new"] = bool(np.isfinite(zn["row__now_s"]).all())
rep["verdict"] = "PASS" if (rep["n_old_clips_differing"] == 0 and rep["field_sets_equal"] and not rep["dtype_differs"]
                            and rep["n_old_clips_compared"] == 4369 and rep["n_eval139_sid_in_new"] == 0
                            and rep["n_eval139_sha12_in_new"] == 0 and rep["new_sid_set_equals_manifest4572"]
                            and rep["add_sids_in_new"] == 203 and rep["add_sids_in_old"] == 0) else "FAIL"
json.dump(rep, open(OUTJ, "w"), indent=1, default=str)


def P(*a):
    print(*a, flush=True)


P("=== CONTROL v9 train: new(4,572) vs old(4,369) ===")
P("old:", OLD)
P("new:", NEW)
P("fields old/new:", rep["fields_old"], rep["fields_new"], "| identical field sets:", rep["field_sets_equal"], "| dtype differs:", rep["dtype_differs"])
P("n_clips old/new:", rep["n_clips_old"], rep["n_clips_new"], "| rows old/new:", rep["n_rows_old"], rep["n_rows_new"])
P("new sid set == manifest4572 sids:", rep["new_sid_set_equals_manifest4572"], "| sha12 set equal:", rep["new_sha12_set_equals_manifest4572"],
  "| new clip order == manifest order:", rep["new_clip_order_equals_manifest_order"])
P("eval139 clips in new release: sid %d / sha12 %d (of %d eval139)" % (rep["n_eval139_sid_in_new"], rep["n_eval139_sha12_in_new"], rep["n_eval139"]))
P("old sids subset of new:", rep["old_sids_subset_of_new"], "| ADD sids in new/old:", rep["add_sids_in_new"], rep["add_sids_in_old"])
P("row0 contiguity old/new:", rep["old_row0_contiguous"], rep["new_row0_contiguous"], "| per-clip n_rows equal:", rep["old_clip_n_rows_equal_new"])
P("COMPARED: %d old clips, %d rows, %d row fields + %d clip fields (bitwise, by sid)" % (
    rep["n_old_clips_compared"], rep["n_old_rows_compared"], rep["fields_compared_row"], rep["fields_compared_clip"]))
P("DIFFERING old clips:", rep["n_old_clips_differing"], "| fields with any difference:", rep["fields_with_any_difference"])
P("CONTROL VERDICT:", rep["verdict"])
P("ADD: %d clips, %d rows" % (rep["n_add_clips"], rep["n_add_rows"]))
for f, d in cl.items():
    P("  ", f, d)
P("clock_src (0 sidecar,1 pose_dt,2 nominal) ADD:", rep["clock_src_counts_add"], "| old-in-new:", rep["clock_src_counts_old_in_new"])
P("now_s finite over all new rows:", rep["now_s_all_finite_new"])
P("--- ADD vs old row-level coverage (selected) ---")
for f in ("row__lead_valid", "row__lat_cls_a", "row__lat_cls_b", "row__lon_cls", "row__lon_v7id", "row__lat_v7id_a",
          "row__nav_args_valid", "row__nav_token", "row__rc_A30_valid", "row__rc_A50_valid", "row__rcH_A50_valid",
          "row__lc_measurable", "row__lc_frac", "row__lc_d_cross_m", "row__goal_w", "row__goal_y", "row__h_obs_s",
          "row__lon_v_target_ms", "row__lead_gap_m", "row__now_s", "row__speed_n2_kmh"):
    if f in cov:
        P("  %-24s add=%s  old=%s" % (f, cov[f]["add"], cov[f]["old"]))
