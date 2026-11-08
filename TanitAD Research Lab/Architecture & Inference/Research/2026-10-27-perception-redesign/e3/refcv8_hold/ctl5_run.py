"""ctl5_run.py -- MM 4a3d05, 2026-11-08: close S6C-HOLD's real-data control WITHOUT a GPU roll.
Rebuilds the `gt` of the 5 rasters that are 0 bytes in F:/refcv9_mapsel/rast_s0 from the 10 cm map labels (the roll's own source: mapsel_roll.py:208 `gt` = map_fine = FineMapGTStore.frames_for_windows,
refc_v3_train.py:4252), into a SCRATCH overlay; F:/refcv9_mapsel is not written. Positive control: the same rebuild on neighbouring and random windows must be byte-equal to their banked `gt`.
The own planes of the 5 are zeros: mapsel_score pass 2 reads only `gt` and the trace's plan for the BASE off-drivable rate. Then s6c_hold.py --subset all, unchanged, with mapsel_score.rast_file redirected for those 5 only,
must reproduce MAPSEL_RESULT.json BASE off_drivable EXACTLY (float equality). Argv: <export root> <output json>."""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
EXP = Path(sys.argv[1]); OUTJ = Path(sys.argv[2])
MC = EXP / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-25-refcv9-mapsel/code"
L2 = EXP / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-25-refcv9-l2/code"
BANKED = EXP / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-25-refcv9-mapsel/result/s0/MAPSEL_RESULT.json"
SPLIT = next(EXP.rglob("DEV_HOLD_SPLIT.json"))
TREE = Path("C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58")
TRACE = Path("F:/refcv9_mapsel/trace_s0.npz"); RAST = Path("F:/refcv9_mapsel/rast_s0")
OVER = EXP.parent / "overlay"; OVER.mkdir(parents=True, exist_ok=True)
BAD = [(10, 82), (10, 87), (10, 97), (10, 102), (42, 139)]
for p in (TREE / "stack", EXP / "stack/ops/r9e3_read", EXP / "stack/ops/r9e2_read", EXP / "taniteval", MC, L2):
    sys.path.insert(0, str(p))
import numpy as np  # noqa: E402
import tanitad  # noqa: E402
assert Path(tanitad.__file__).as_posix().startswith(TREE.as_posix()), tanitad.__file__
from tanitad.data import semantic_map_gt_fine as F  # noqa: E402
import mapsel_data as D  # noqa: E402

z = np.load(TRACE, allow_pickle=False)
meta = json.loads(str(z["meta_json"]))
uuids = D.clip_uuids()[0]
cfg = json.loads(Path("E:/refcv8_eval_kit/ckpt/config.json").read_text(encoding="utf-8"))
args = cfg.get("args", cfg)
x, y = args.get("map_hires_x_max_m"), args.get("map_hires_y_half_m")
ext = F.MapExtent(float(F.EXTENT_REFCV7.x_max_m if x is None else x), float(F.EXTENT_REFCV7.y_half_m if y is None else y))
store = F.FineMapGTStore(Path(meta["map_gt_root"]), max_open=4, extent=ext)
ns = int(meta["n_stack"])


def regen(e, w):
    return np.asarray(store.frames_for_windows(uuids[int(e)], [int(w)], n_stack=ns).codes[0], np.uint8)


def rf(e, w):
    return RAST / f"e{int(e):03d}_w{int(w):04d}.npz"


idx = [(int(a), int(b)) for a, b, g in zip(z["e_i"], z["ws"], z["gt_label"]) if bool(g)]
rng = random.Random(0)
pos = [(10, 77), (10, 92), (10, 107), (42, 134), (42, 144)] + rng.sample([k for k in idx if k not in BAD], 25)
checked = []
for e, w in pos:
    if not rf(e, w).exists() or rf(e, w).stat().st_size == 0:
        continue
    b = np.load(rf(e, w))["gt"]
    r = regen(e, w)
    eq = b.shape == r.shape and bool((b == r).all())
    checked.append({"e": e, "w": w, "byte_equal": eq})
    if not eq:
        raise SystemExit(f"[ctl5] POSITIVE CONTROL FAILED at e{e} w{w}: shape {b.shape} vs {r.shape}, cells differ {int((b != r).sum()) if b.shape == r.shape else 'n/a'}")
print(f"[ctl5] positive control: {len(checked)} windows byte-equal", flush=True)
assert len(checked) >= 25

tmpl = np.load(rf(10, 77))
rebuilt = []
for e, w in BAD:
    if rf(e, w).stat().st_size != 0:
        raise SystemExit(f"[ctl5] {rf(e, w)} is no longer 0 bytes: stop and look")
    arrs = {k: np.zeros_like(tmpl[k]) for k in tmpl.keys()}
    arrs["gt"] = regen(e, w)
    assert arrs["gt"].shape == tmpl["gt"].shape
    np.savez_compressed(OVER / rf(e, w).name, **arrs)
    rebuilt.append({"e": e, "w": w, "gt_sha256": hashlib.sha256(arrs["gt"].tobytes()).hexdigest(), "gt_seen_frac": float((arrs["gt"] != 255).mean())})
print("[ctl5] rebuilt", rebuilt, flush=True)

import s6c_hold as S  # noqa: E402
orig_run = S.run_scorer
hits = {}


def run_scorer(M, *a, **k):
    orig_rf = M.rast_file

    def over_rf(rdir, e, w):
        key = (int(e), int(w))
        if key in BAD:
            hits[key] = hits.get(key, 0) + 1
            return OVER / rf(e, w).name
        return orig_rf(rdir, e, w)
    M.rast_file = over_rf
    try:
        return orig_run(M, *a, **k)
    finally:
        M.rast_file = orig_rf


S.run_scorer = run_scorer
OUT = EXP.parent / "out_all"; OUT.mkdir(exist_ok=True)
RO = EXP.parent / "read_all_overlay.json"
rc = S.main(["--mapsel-code", str(MC), "--l2-code", str(L2), "--split", str(SPLIT), "--subset", "all", "--seed", "0", "--trace", str(TRACE), "--rast", str(RAST), "--out", str(OUT), "--read-out", str(RO)])
assert rc == 0
read = json.loads(RO.read_text(encoding="utf-8"))
ctl = S.real_data_control(read, BANKED)
res = {"what": "S6C-HOLD real-data control, closed without a GPU roll: ALL windows, the 5 zero-byte rasters' gt rebuilt from the 10 cm labels in a scratch overlay",
       "tier": "T1 self-action open loop on logged eval139 frames", "evidence_class": "MEASURED", "positive_control_byte_equal": checked, "n_positive": len(checked), "rebuilt": rebuilt,
       "overlay_redirect_hits": {f"{a}_{b}": c for (a, b), c in hits.items()}, "real_data_control": ctl, "read_all_rate": read["rate"], "read_all_n": read["n"], "script": Path(__file__).name,
       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "s6c_hold_sha256": hashlib.sha256((EXP / "stack/ops/r9e3_read/s6c_hold.py").read_bytes()).hexdigest()}
OUTJ.write_text(json.dumps(res, indent=1), encoding="utf-8")
print("[ctl5] RESULT", json.dumps(ctl), flush=True)
print("ZZCTL5-" + ("PASS" if ctl["pass"] else "FAIL") + "ZZ", flush=True)
