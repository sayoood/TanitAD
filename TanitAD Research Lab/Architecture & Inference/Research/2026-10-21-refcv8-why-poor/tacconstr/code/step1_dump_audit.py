"""STEP 1 (positive-assertion audit): do the banked refcv8 per-frame battery dumps hold the tactical CONSTRAINT head
outputs?  Reads EVERY npz in each tree; counts files actually READ (never infers absence from a failed read); each
file also asserts a same-breath CONTROL key that MUST be present (lat_pred_nav_true / ws), so an unreadable file is
counted as unreadable, not as 'no constraint keys'.  CPU only, read-only.
"""
import glob, json, os, re, sys
import numpy as np

TREES = {
    "step48852ema_pf/dump_s0": "C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/dump_s0",
    "step48852ema_pf/dump_s1": "C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/dump_s1",
    "step40000ema_pf/dump_s0": "C:/Users/Admin/refcv8_pf/battery40k_pf/step40000ema_pf/dump_s0",
    "step40000ema_pf/dump_s1": "C:/Users/Admin/refcv8_pf/battery40k_pf/step40000ema_pf/dump_s1",
}
# a key is 'constraint-like' if it names a constraint head / v9 constraint vector / probe-level constraint
PAT = re.compile(r"(cons|constraint|v9_|lat_c\b|lon_c\b|speed_goal|psi_term|prog6|v_end|onset|r8_(?!nav))", re.I)
out = {"patterns": PAT.pattern, "trees": {}}
for name, root in TREES.items():
    rec = {"root": root}
    for sub, glob_pat in (("episode", "ep*.npz"), ("decisions", "decisions/ep*.npz")):
        files = sorted(glob.glob(os.path.join(root, glob_pat)))
        keys, nread, nerr, ctrl_ok = {}, 0, 0, 0
        for f in files:
            try:
                z = np.load(f, allow_pickle=True)
                ks = list(z.files)
                nread += 1
                for k in ks:
                    keys[k] = keys.get(k, 0) + 1
                ctrl = "lat_pred_nav_true" if sub == "decisions" else "ws"
                ctrl_ok += int(ctrl in ks)
            except Exception as e:  # noqa
                nerr += 1
        hits = sorted(k for k in keys if PAT.search(k))
        rec[sub] = {"n_files_listed": len(files), "n_files_read": nread, "n_read_errors": nerr,
                    "n_files_with_control_key": ctrl_ok, "n_distinct_keys": len(keys),
                    "constraint_like_keys": hits, "all_keys": sorted(keys)}
    out["trees"][name] = rec
# the NAMED constraint outputs the model emits (refc_v3.py:1998-2005, refcv6_tactical.py:642-645) -- none may be in any dump
EMITTED = ["v9_lat_c", "v9_lon_c", "v9_speed", "r8_v9_lat_c", "r8_v9_lon_c", "r8_v9_speed", "cons_lat", "cons_lon",
           "cons_lat8", "cons_lon8", "cons_raw", "cons_valid", "cons_score"]
allkeys = set()
for r in out["trees"].values():
    for sub in ("episode", "decisions"):
        allkeys |= set(r[sub]["all_keys"])
out["emitted_constraint_names_searched"] = EMITTED
out["emitted_names_found_in_any_dump"] = sorted(set(EMITTED) & allkeys)
out["any_constraint_like_key_in_any_dump"] = sorted({k for r in out["trees"].values() for sub in ("episode", "decisions") for k in r[sub]["constraint_like_keys"]})
tot_read = sum(r[s]["n_files_read"] for r in out["trees"].values() for s in ("episode", "decisions"))
tot_ctrl = sum(r[s]["n_files_with_control_key"] for r in out["trees"].values() for s in ("episode", "decisions"))
out["total_files_read"], out["total_files_with_control_key"] = tot_read, tot_ctrl
out["verdict"] = ("CONSTRAINTS ARE NOT IN THE DUMPS" if (tot_read == tot_ctrl and tot_read > 0 and not out["any_constraint_like_key_in_any_dump"])
                  else "INCONCLUSIVE (unreadable files or control keys missing)")
os.makedirs("C:/Users/Admin/refcv8_pf/tacconstr", exist_ok=True)
json.dump(out, open("C:/Users/Admin/refcv8_pf/tacconstr/STEP1_DUMP_AUDIT.json", "w"), indent=1)
print(out["verdict"], "| files read", tot_read, "| with control key", tot_ctrl)
for n, r in out["trees"].items():
    print(n, {s: (r[s]["n_files_read"], r[s]["n_read_errors"], len(r[s]["constraint_like_keys"])) for s in ("episode", "decisions")})
print("constraint-like keys anywhere:", out["any_constraint_like_key_in_any_dump"])
print("emitted names found:", out["emitted_names_found_in_any_dump"])
