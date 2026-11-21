import os, sys, json, re, time
sys.path.insert(0, 'C:/Users/Admin/par18/s0probe')
from common_s0 import *
logs, tok_log, by = s0()
L = set(logs)
DATA = 'D:/Projects/TanitAD/data'
ROOTS = {"refe_proxy/pixels": DATA + "/refe_proxy/pixels", "refe_m5/pixels": DATA + "/refe_m5/pixels", "nuplan-camera/mini": DATA + "/nuplan-camera/mini",
         "nuplan-camera/scenarios": DATA + "/nuplan-camera/scenarios", "nuplan-camera/scenarios4": DATA + "/nuplan-camera/scenarios4", "nuplan-camera/probe": DATA + "/nuplan-camera/probe",
         "refe_navtrain": DATA + "/refe_navtrain", "refe_navtrain10": DATA + "/refe_navtrain10", "refe_navtest": DATA + "/refe_navtest", "refe_m5r": DATA + "/refe_m5r",
         "refe_consistent": DATA + "/refe_consistent", "heldout_frames": "C:/Users/Admin/p2heldout_work/frames_heldout", "videos": DATA + "/videos"}
pat = re.compile(r'(20\d\d\.\d\d\.\d\d\.\d\d\.\d\d\.\d\d_veh-\d+_\d+_\d+)')
res = {}
for name, root in ROOTS.items():
    t0 = time.time()
    found = set()
    n_ent = 0
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            it = list(os.scandir(d))
        except Exception:
            continue
        for e in it:
            n_ent += 1
            for m in pat.findall(e.name):
                if m in L:
                    found.add(m)
            if e.is_dir(follow_symlinks=False) and depth < 3:
                stack.append((e.path, depth + 1))
    res[name] = {'exists': os.path.isdir(root), 'entries_scanned_depth<=3': n_ent, 'S0_logs_named': len(found), 'seconds': round(time.time() - t0, 1)}
    print(name, res[name])
json.dump(res, open('C:/Users/Admin/par18/s0probe/roots13.json', 'w'), indent=1)
