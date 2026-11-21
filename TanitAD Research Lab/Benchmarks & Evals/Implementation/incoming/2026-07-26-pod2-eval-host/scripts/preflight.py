"""pod2 eval-host PREFLIGHT.

Three standing checks + the sys.path audit that the eval-pod incident earned:
the eval pod was 62 % stale with corridor.py entirely MISSING, and a SECOND
stale tree (/root/TanitAD/stack) was hard-coded by every taniteval submodule via
sys.path.insert -- so probing only the obvious tree declared the pod clean when
it wasn't. This enumerates EVERY tree on sys.path after the decision-grade
imports have run, and verifies each against the repo md5 manifests.

Checks:
  P0  sys.path audit  -- every entry, every loaded taniteval/tanitad module,
                         every OTHER stack/taniteval tree present on the box.
  P1  corridor.py     -- present AND EXERCISED (not merely importable).
  P2  lateral.py      -- paired_cross_track must stamp horizon_provenance and
                         horizon_s = 2.0 on the SPARSE (4-knot) surface.
                         0.4 s means stale code (step * DT, the 5x bug).
  P3  v1 = 0.4271     -- reproduced by the runner, not quoted.  (run separately)
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/preflight_pod2.json"
MAN = {"taniteval": "/root/md5_pod2_taniteval.json",
       "stack": "/root/md5_pod2_stack.json"}
VERIFIED_ROOTS = {"/root/taniteval": "taniteval", "/root/TanitAD/stack": "stack"}

res = {"host": os.uname().nodename, "python": sys.version.split()[0],
       "cwd": os.getcwd(), "argv0_dir": os.path.dirname(os.path.abspath(__file__)),
       "env": {k: os.environ.get(k) for k in
               ("PYTHONPATH", "TANITEVAL_STACK_OVERRIDE", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}}

# ------------------------------------------------------------------ P0 imports
import taniteval                                                    # noqa: E402
from taniteval import (bench, ci, closedloop, corridor, data,       # noqa: E402
                       hierarchy, lateral, pathspeed, registry, rollout, runner)
import tanitad                                                       # noqa: E402
from tanitad.data import parity                                      # noqa: E402

# --- every entry on sys.path, classified
def tree_kind(p):
    k = []
    if os.path.isfile(os.path.join(p, "tanitad", "__init__.py")):
        k.append("HOSTS tanitad")
    if os.path.isfile(os.path.join(p, "taniteval", "__init__.py")):
        k.append("HOSTS taniteval")
    return k


path_rows = []
for p in sys.path:
    ap = os.path.abspath(p) if p else os.path.abspath(os.getcwd())
    row = {"entry": p, "abspath": ap, "exists": os.path.isdir(ap),
           "hosts": tree_kind(ap) if os.path.isdir(ap) else [],
           "verified_root": None, "verified": None}
    for vr, lab in VERIFIED_ROOTS.items():
        if ap == vr or ap.startswith(vr + os.sep):
            row["verified_root"] = vr
    path_rows.append(row)
res["P0_sys_path"] = path_rows
res["P0_importable_trees_on_path"] = [
    r for r in path_rows if r["hosts"]]

# --- re-verify each verified root against its manifest, NOW (files can change)
def verify_tree(root, manifest):
    man = json.loads(Path(manifest).read_text())["files"]
    bad, missing, extra = [], [], []
    seen = set()
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in
                 ("__pycache__", ".git", ".pytest_cache", ".mypy_cache")]
        for f in fn:
            if f.endswith(".pyc"):
                continue
            full = os.path.join(dp, f)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            seen.add(rel)
            if rel not in man:
                extra.append(rel)
                continue
            h = hashlib.md5()
            with open(full, "rb") as fh:
                for b in iter(lambda: fh.read(1 << 20), b""):
                    h.update(b)
            if h.hexdigest() != man[rel]:
                bad.append(rel)
    missing = sorted(set(man) - seen)
    return {"root": root, "n_manifest": len(man), "n_seen": len(seen),
            "n_mismatch": len(bad), "mismatch": bad[:20],
            "n_missing": len(missing), "missing": missing[:20],
            "n_extra": len(extra), "extra": extra[:20],
            "clean": not bad and not missing}


res["P0_tree_verification"] = {r: verify_tree(r, MAN[l])
                               for r, l in VERIFIED_ROOTS.items()}

# --- every loaded taniteval.*/tanitad.* module and where it actually came from
mods = {}
for name, m in sorted(sys.modules.items()):
    if not (name == "taniteval" or name.startswith("taniteval.")
            or name == "tanitad" or name.startswith("tanitad.")):
        continue
    f = getattr(m, "__file__", None)
    if not f:
        continue
    fa = os.path.abspath(f)
    root = next((vr for vr in VERIFIED_ROOTS
                 if fa == vr or fa.startswith(vr + os.sep)), None)
    mods[name] = {"file": fa, "from_verified_root": root}
res["P0_loaded_modules"] = mods
res["P0_modules_off_verified_trees"] = {k: v for k, v in mods.items()
                                        if v["from_verified_root"] is None}

# --- OTHER stack/taniteval trees present on this box (absence at one location
#     is not absence: find them all, then prove none of them is on the path)
try:
    found = subprocess.run(
        ["bash", "-lc",
         "find / -maxdepth 6 \\( -path /proc -o -path /sys -o -path /dev \\) "
         "-prune -o -type f -name '__init__.py' "
         "\\( -path '*/tanitad/__init__.py' -o -path '*/taniteval/__init__.py' \\) "
         "-print 2>/dev/null | head -60"],
        capture_output=True, text=True, timeout=600).stdout.split()
except Exception as ex:                                              # noqa: BLE001
    found = [f"FIND_FAILED:{ex}"]
onpath = {r["abspath"] for r in path_rows if r["exists"]}
res["P0_all_trees_on_box"] = [
    {"init": f,
     "tree_root": os.path.dirname(os.path.dirname(os.path.abspath(f))),
     "on_sys_path": os.path.dirname(os.path.dirname(os.path.abspath(f))) in onpath,
     "is_verified_root": os.path.dirname(os.path.dirname(os.path.abspath(f)))
                          in VERIFIED_ROOTS}
    for f in found if not f.startswith("FIND_FAILED")]
res["P0_verdict"] = (
    "PASS" if (all(v["clean"] for v in res["P0_tree_verification"].values())
               and not res["P0_modules_off_verified_trees"]
               and not [t for t in res["P0_all_trees_on_box"]
                        if t["on_sys_path"] and not t["is_verified_root"]])
    else "FAIL")

# --------------------------------------------------------------- P1 corridor
import torch                                                         # noqa: E402

p1 = {"module_file": corridor.__file__,
      "constants": {"CORRIDOR_HALFWIDTH_M": corridor.CORRIDOR_HALFWIDTH_M,
                    "CORRIDOR_GRID_M": list(corridor.CORRIDOR_GRID_M),
                    "JUNCTION_DEG": corridor.JUNCTION_DEG}}
try:
    # EXERCISE it: 4 episodes x 3 windows, a dense path that departs the corridor
    g = torch.zeros(12, 30, 2)
    g[:, :, 0] = torch.linspace(0, 30, 30)          # straight ahead
    p = g.clone()
    p[:6, :, 1] = torch.linspace(0, 4.0, 30)        # half the windows drift out
    eid = [f"ep_{i//3:03d}" for i in range(12)]
    lat_abs = corridor.cross_track_from_paths(p, g)   # already |XTE| numpy [N,K]
    blk = corridor.corridor_block(lat_abs, eid, n_boot=200)
    head = torch.zeros(12); head[:6] = 25.0
    spd = torch.full((12,), 12.0)
    st = corridor.stratified(lat_abs, eid, head, spd, n_boot=200)
    p1.update(exercised=True,
              lat_abs_shape=list(lat_abs.shape),
              horizon_seconds_185=corridor.horizon_seconds(185),
              horizon_seconds_grid={str(k): corridor.horizon_seconds(k)
                                    for k in (20, 60, 90, 120, 150, 185)},
              horizon_ceiling_T205=corridor.horizon_ceiling(205),
              horizon_ceiling_T198=corridor.horizon_ceiling(198),
              block_keys=sorted(str(k) for k in blk.keys()),
              summary=corridor.summarise(blk),
              strata_keys=sorted(str(k) for k in st.keys()),
              full_block=blk)
    p1["verdict"] = "PASS"
except Exception as ex:                                              # noqa: BLE001
    p1.update(exercised=False, error=f"{type(ex).__name__}: {ex}",
              verdict="FAIL")
res["P1_corridor"] = p1

# ---------------------------------------------------------------- P2 lateral
p2 = {"module_file": lateral.__file__}
try:
    N = 40
    eid = [f"ep_{i//4:03d}" for i in range(N)]
    gt = torch.zeros(N, 4, 2)
    gt[:, :, 0] = torch.tensor([2.5, 5.0, 7.5, 10.0])     # 4 knots, 0.5 s apart
    pa = gt.clone(); pa[:, :, 1] = 0.10
    pb = gt.clone(); pb[:, :, 1] = 0.40
    d_last = lateral.paired_cross_track(pa, pb, gt, eid, step=4, n_boot=200)
    d_expl = lateral.paired_cross_track(pa, pb, gt, eid, step=4, n_boot=200,
                                        knot_dt=0.5)
    sparse = lateral.from_sparse_windows(
        {"eid": eid, "pred": pa, "gt": gt, "speed": None}, n_boot=200)
    p2.update(
        horizon_s_step4=d_last.get("horizon_s"),
        horizon_provenance=d_last.get("horizon_provenance"),
        n_knots=d_last.get("n_knots"),
        horizon_s_explicit_knot_dt=d_expl.get("horizon_s"),
        horizon_provenance_explicit=d_expl.get("horizon_provenance"),
        sparse_surface=sparse.get("surface"),
        sparse_dt_s=sparse.get("dt_s"),
        sparse_horizon_s=sparse.get("horizon_s"),
        estimator=d_last.get("estimator") or d_last.get("method"),
        delta_keys=sorted(d_last.keys()))
    ok = (abs(float(d_last["horizon_s"]) - 2.0) < 1e-9
          and d_last.get("horizon_provenance") == "inferred_from_knot_count"
          and abs(float(d_expl["horizon_s"]) - 2.0) < 1e-9
          and d_expl.get("horizon_provenance") == "explicit")
    p2["stale_signature_0p4s"] = abs(float(d_last["horizon_s"]) - 0.4) < 1e-9
    p2["verdict"] = "PASS" if ok else "FAIL"
except Exception as ex:                                              # noqa: BLE001
    p2.update(error=f"{type(ex).__name__}: {ex}", verdict="FAIL")
res["P2_lateral"] = p2

# ------------------------------------------------------- val deployment probe
try:
    files = data.list_val_episodes(f"/root/valdata/{data.CLEAN_VAL}", 40)
    rec = data.last_val_parity()
    res["P3_val_chokepoint"] = {"n_files": len(files),
                                "first": os.path.basename(str(files[0])),
                                "last": os.path.basename(str(files[-1])),
                                "provenance": rec,
                                "verdict": "PASS" if len(files) == 40 else "FAIL"}
except Exception as ex:                                              # noqa: BLE001
    res["P3_val_chokepoint"] = {"error": f"{type(ex).__name__}: {ex}",
                                "verdict": "FAIL"}

Path(OUT).write_text(json.dumps(res, indent=1, default=str))
print(f"P0 sys.path/tree audit : {res['P0_verdict']}")
for r, v in res["P0_tree_verification"].items():
    print(f"   {r}: seen={v['n_seen']}/{v['n_manifest']} mismatch={v['n_mismatch']} "
          f"missing={v['n_missing']} extra={v['n_extra']} clean={v['clean']}")
print(f"   modules off verified trees: {list(res['P0_modules_off_verified_trees'])}")
print(f"   other trees on box: "
      f"{[(t['tree_root'], t['on_sys_path']) for t in res['P0_all_trees_on_box']]}")
print(f"P1 corridor EXERCISED  : {res['P1_corridor']['verdict']}  "
      f"horizon_seconds(185)={res['P1_corridor'].get('horizon_seconds_185')}")
print(f"P2 lateral horizon     : {res['P2_lateral']['verdict']}  "
      f"horizon_s={res['P2_lateral'].get('horizon_s_step4')} "
      f"prov={res['P2_lateral'].get('horizon_provenance')} "
      f"(0.4 s => stale: {res['P2_lateral'].get('stale_signature_0p4s')})")
print(f"P3 val chokepoint      : {res['P3_val_chokepoint']['verdict']} "
      f"n={res['P3_val_chokepoint'].get('n_files')}")
print(f"-> {OUT}")
