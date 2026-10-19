"""Compare Thor md5 vs local md5 for the refcv8 eval-side inputs -> kit_checks/eval_inputs_md5_compare.json"""
import json, re, collections
KC = "E:/refcv8_eval_kit/kit_checks/"
def read(p, strip_local=False):
    d = {}
    for ln in open(p, encoding="utf-8"):
        m = re.match(r"^([0-9a-f]{32})\s+\*?(\S+)", ln.strip())
        if m:
            d[m.group(2)] = m.group(1)
    return d
thor = read(KC + "thor_md5.txt")
loc_raw = read(KC + "local_md5.txt")
# thor: v9 path is absolute; normalise to the data-relative key used locally
thor = {("refcv8_v9labels/release/" + k.rsplit("/", 1)[1] if k.startswith("/home/nvidia/refcv8_v9labels/release/") else k): v for k, v in thor.items()}
loc = loc_raw
done_local = "ZZLOCALDONEZZ" in open(KC + "local_md5.txt", encoding="utf-8").read()
done_thor = "ZZTHORDONEZZ" in open(KC + "thor_md5.txt", encoding="utf-8").read()
def group(k):
    if k.startswith("refcv6-b1-416x1024-eval139/"): return "refcv6-b1-416x1024-eval139 (eval cache)"
    if k.startswith("sam3_gt_v3/"): return "sam3_gt_v3 (137 eval-clip files)"
    if k.startswith("refcv8_v9labels/"): return "v9 eval release"
    if k.startswith("refcv8/") or k.startswith("refcv8_"): return "NEW refcv8 file: " + k
    return k
rows = []
only_thor = sorted(k for k in thor if k not in loc)
only_loc = sorted(k for k in loc if k not in thor)
by = collections.OrderedDict()
n_differ = 0; differ = []
for k in sorted(set(thor) & set(loc)):
    same = thor[k] == loc[k]
    g = group(k)
    by.setdefault(g, [0, 0]); by[g][0] += 1; by[g][1] += int(same)
    if not same:
        n_differ += 1; differ.append({"file": k, "thor": thor[k], "local": loc[k]})
res = {
  "thor_n": len(thor), "local_n": len(loc), "compared": len(set(thor) & set(loc)),
  "identical": sum(v[1] for v in by.values()), "n_differ": n_differ, "differ": differ,
  "thor_only": only_thor, "local_only": only_loc,
  "by_group_n_compared_n_identical": by,
  "thor_job_done_marker": done_thor, "local_job_done_marker": done_local,
  "expected_v9_eval_md5_from_argv": "6b5c7f207cffc3b7eb3cd527fd433599",
  "v9_eval_thor_md5": thor.get("refcv8_v9labels/release/v9_labels_eval139.npz"),
  "v9_eval_local_md5": loc.get("refcv8_v9labels/release/v9_labels_eval139.npz"),
  "local_roots": {"matched files (reused, NOT copied)": "E:/refcv6_eval_kit/data (sam3_gt_v3 -> sam3_gt_v3_eval)",
                  "new/different files (copied by read-only scp)": "E:/refcv8_eval_kit/data"},
  "thor_root": "/home/nvidia/data (+ /home/nvidia/refcv8_v9labels/release for the v9 eval release)",
  "method": "md5sum on Thor (nice -n 19 ionice -c3, read-only, started 2026-10-15 ~21:35 Berlin) vs hashlib md5 on the dev box; eval139 = every file of the dir; sam3_gt_v3 restricted to the 137 eval-clip files the kit carries (names from E:/refcv7_eval_kit/kit_checks/sam3_v3_eval_names.txt). Entry 'sam3_gt_v3/' is a directory artefact of the list builder (Thor: 'Is a directory'; local: PermissionError) and is not a file.",
}
bad = [k for k in only_thor if k not in ("sam3_gt_v3/", "refcv8_v9labels/release/v9_labels_eval139.manifest.json")]
res["verdict"] = ("ALL IDENTICAL" if (done_thor and done_local and n_differ == 0 and not bad and
                  res["v9_eval_thor_md5"] == res["expected_v9_eval_md5_from_argv"] == res["v9_eval_local_md5"])
                  else "NOT CLEAN (see differ / thor_only / flags)")
res["unexpected_thor_only"] = bad
json.dump(res, open(KC + "eval_inputs_md5_compare.json", "w", encoding="utf-8"), indent=1)
print(json.dumps({k: res[k] for k in ("thor_n","local_n","compared","identical","n_differ","verdict","unexpected_thor_only","thor_job_done_marker","local_job_done_marker")}, indent=1))
print(json.dumps(by, indent=1))
