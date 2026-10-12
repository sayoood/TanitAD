#!/usr/bin/env python3
"""MEASURE the absolute tolerance of `refe/rl_fan_dump.py --compare-atol` on the dev-box GPU (RTX 4060).  Never assumed; every arm below is dumped with the
CLI dumper itself (`--compare-sets <A40 banked step-10,075 sets> --compare-atol`), so the path that produces the verdict is the path the pod chain will run.

    python run_calibration.py --phase prep|dump|analyse|all [--only arm1,arm2] [--work <dir>] [--tol-candidate 0.5]
    Python: the dev box's driverl-eval venv (torch + cuda + cv2).  Inputs: the identity instrument's assets (eval/rl_fan_identity.py).

PRE-REGISTRATION (written 2026-10-12 BEFORE any arm was dumped; the rule is applied by `analyse`, not by hand afterwards)
----------------------------------------------------------------------------------------------------------------------
Sample: 48 keys, a deterministic stride (every 28th) over the 1,365 keys that have BOTH a dev-box image set and a banked A40 step-10,075 set (`identity/keys.tsv`).
Reference: `sets_10075_proxy.jsonl` (the A40's banked sets).  Delta of a set = max |new - ref| over every x, y (metres) and yaw (radians) value, ONE number.

NULL arms (a correct dumper; they define the floor and must read WITHIN the tolerance):
    N0_b4   the new dumper, batch 4                                  (dev box vs the A40: the cross-GPU floor on this sample)
    N1_b1   batch 1                                                  (batch-grouping floor, same GPU)
    N2_b2   batch 2
    N_full  the identity run's 1,365-key dump (CPU re-analysis, no GPU)      and      N_b1 / N_b2 / N_b4 / N_b4shift / N_b4w2: its 24-key invariance dumps (CPU)
  floor_max = the largest per-set delta over the NULL arms above.
INFORMATIONAL null (reported, NOT part of the floor): N3_jpeg = every image decoded and re-encoded at JPEG q95 with no other change (benign pixel noise).

DECISION regression arms (a deliberate defect of the kind a fresh pod could have; each must be CAUGHT):
    (a) a WRONG calibration table:   A1_calib_other_vehicle   each log's 4-camera rig replaced by a DIFFERENT vehicle's rig (real rig, wrong vehicle)
                                     A2_calib_half_scale      fx, fy, cx, cy x 0.5 (resized-pixel intrinsics where native-pixel ones are expected)
    (b) CORRUPTED images:            B1_img_hflip             all 4 cameras mirrored left-right          (re-encoded at q95, as N3)
                                     B2_img_bgr_rgb           channel order swapped on all 4 cameras     (re-encoded at q95)
    (c) the WRONG checkpoint:        C1_head_noise10          model_final.pt with traj_head.2 (the 60x256 output layer, 15,420 params) + N(0, (0.1 std(W))^2), seed 20261012
                                     C2_snap_epoch021         ANOTHER banked checkpoint: snapshots/snap_epoch021.pt (step 7,348, 2,727 steps before the final)
SENSITIVITY arms (reported, not part of the decision; they map the detection limit):
    S_focal105 (fx, fy x 1.05)   S_yaw3 (rig rotated 3 deg about the ego z axis)   S_head_noise1 (1 % noise)   S_snap027 (step 9,763, 312 steps early)
    S_wrong_scene (each key sees the NEXT key's four images)   S_cam_reversed (camera order reversed)

DECISION RULE:   tol = 2 x floor_max, rounded UP to a multiple of 0.125   (2 bf16 ulps at the top of the measured range: the floor is quantised at ulp multiples).
    ACCEPT tol  iff  (i) every NULL arm has 0 sets beyond tol,
                     (ii) every DECISION arm has max delta >= 10 x tol  AND  >= max(5, 10 % of the sample) sets beyond tol   ("clearly non-zero").
    Otherwise NO TOLERANCE IS CHOSEN: `verdict.json` names each arm that does not separate and the largest tol that WOULD still separate it.
A refused arm (exit 4: C2's checkpoint step has no reference at step 10,075) is judged from the offline delta with ckpt_step relabelled, and its CLI exit is reported next to it.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(PKG, "refe"))
import rl_fan_dump as RF  # noqa: E402

FINAL = "D:/Projects/TanitAD-artifacts/refe-final-2026-10-01"
SNAPS = FINAL + "/snapshots"
PIXELS = "D:/Projects/TanitAD/data/refe_proxy/pixels"
MODEL_FINAL_SHA = "4246a320be71ad0e315f8f298b7dadab3dbbba2c24f4fc30706e0129f5057e66"
STEP = 10075
N_KEYS, STRIDE = 48, 28
TOL_CANDIDATE = 0.5                      # 2 x the 0.25 floor already MEASURED by the identity run; the rule below re-derives it from THIS run's nulls
SEED = 20261012
GRID = [0.25, 0.3125, 0.375, 0.5, 0.75, 1.0, 2.0, 5.0]

ARMS = [   # name, kind, variant
    ("N0_b4", "null", {"batch": 4}),
    ("N1_b1", "null", {"batch": 1}),
    ("N2_b2", "null", {"batch": 2}),
    ("N3_jpeg", "info_null", {"pix": "jpeg"}),
    ("A1_calib_other_vehicle", "decision", {"calib": "other_vehicle"}),
    ("A2_calib_half_scale", "decision", {"calib": "half_scale"}),
    ("B1_img_hflip", "decision", {"pix": "hflip"}),
    ("B2_img_bgr_rgb", "decision", {"pix": "bgr_rgb"}),
    ("C1_head_noise10", "decision", {"ckpt": "head_noise10"}),
    ("C2_snap_epoch021", "decision", {"ckpt": "snap021"}),
    ("S_focal105", "info", {"calib": "focal105"}),
    ("S_yaw3", "info", {"calib": "yaw3"}),
    ("S_head_noise1", "info", {"ckpt": "head_noise1"}),
    ("S_snap027", "info", {"ckpt": "snap027"}),
    ("S_wrong_scene", "info", {"bank": "wrong_scene"}),
    ("S_cam_reversed", "info", {"bank": "cam_reversed"}),
]


def sha256_file(p):
    return RF.sha256_bytes(p)


def wr_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)


# ------------------------------------------------------------------------------------------------------------------ prep (CPU)
def pick_keys(W):
    rows = [ln.rstrip("\n").split("\t") for ln in open(f"{W}/identity/keys.tsv", encoding="utf-8")][1:]
    sample = rows[::STRIDE][:N_KEYS]
    assert len(sample) == N_KEYS, len(sample)
    return [(t, l) for t, l in sample]


def prep(W, A):
    os.makedirs(A, exist_ok=True)
    keys = pick_keys(W)
    kp = f"{A}/keys48.tsv"
    with open(kp, "w", encoding="utf-8", newline="\n") as f:
        f.write("token\tlog\n" + "".join(f"{t}\t{l}\n" for t, l in keys))
    kset = {(l, t) for t, l in keys}
    rows = [json.loads(ln) for ln in open(f"{W}/identity/bank/targets_rank0.jsonl", encoding="utf-8")]
    rows = [r for r in rows if (r["log_name"], r["token"]) in kset]
    assert len(rows) == N_KEYS, len(rows)
    rows.sort(key=lambda r: (r["log_name"], r["token"]))

    def write_bank(name, rs):
        d = f"{A}/bank_{name}"
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/targets_rank0.jsonl", "w", encoding="utf-8", newline="\n") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rs))

    write_bank("base", rows)
    nxt = [dict(r, image=rows[(i + 1) % len(rows)]["image"]) for i, r in enumerate(rows)]
    write_bank("wrong_scene", nxt)
    write_bank("cam_reversed", [dict(r, image=list(reversed(r["image"]))) for r in rows])

    # ---- pixel variants: only the 48 x 4 frames the sample reads, decoded -> transformed -> re-encoded (q95) into zips of their own.  The originals are never touched.
    import cv2
    import numpy as np

    def make_pix(name, fn):
        root = f"{A}/pix_{name}"
        if os.path.isdir(root) and len([x for x in os.listdir(root) if x.endswith(".zip")]) >= 4:
            return
        os.makedirs(root, exist_ok=True)
        outs = {}
        for r in rows:
            for ref in r["image"]:
                cam = ref.split("/")[1]
                zname = f"{r['log_name']}_{cam}.zip"
                base = os.path.basename(ref)
                with zipfile.ZipFile(f"{PIXELS}/{zname}") as zin:
                    blob = zin.read(base)
                im = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
                assert im is not None, ref
                ok, enc = cv2.imencode(".jpg", np.ascontiguousarray(fn(im)), [cv2.IMWRITE_JPEG_QUALITY, 95])
                assert ok
                z = outs.get(zname)
                if z is None:
                    z = outs[zname] = zipfile.ZipFile(f"{root}/{zname}", "w", zipfile.ZIP_STORED)
                if base not in z.namelist():
                    z.writestr(base, enc.tobytes())
        for z in outs.values():
            z.close()

    make_pix("jpeg", lambda im: im)
    make_pix("hflip", lambda im: cv2.flip(im, 1))
    make_pix("bgr_rgb", lambda im: im[:, :, ::-1])

    # ---- calibration variants (copies in the work dir; the original table is never edited)
    base = json.load(open(f"{W}/bank_full/calib_table.json", encoding="utf-8"))
    FI = base["fields"]
    ix = {f: i for i, f in enumerate(FI)}

    def save_calib(name, tab):
        wr_json(f"{A}/calib_{name}.json", tab)

    def vehicle_of(log):
        return log.split("_")[1]

    cams = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")
    t = copy.deepcopy(base)
    logs = sorted(base["logs"])
    vehs = sorted({vehicle_of(l) for l in logs})
    changed = same = 0
    for lg in logs:
        v = vehicle_of(lg)
        dv = vehs[(vehs.index(v) + 1) % len(vehs)]                       # the next vehicle, cyclically
        donor = next(d for d in logs if vehicle_of(d) == dv and any(base["logs"][d][c] != base["logs"][lg][c] for c in cams))
        t["logs"][lg] = copy.deepcopy(base["logs"][donor])
        changed += 1
    t["source"] = "ATOL CALIBRATION ARM A1: every log carries the NEXT vehicle's rig"
    save_calib("other_vehicle", t)

    def scaled(name, fields, k):
        tt = copy.deepcopy(base)
        for lg in tt["logs"]:
            for c in tt["logs"][lg]:
                for f in fields:
                    tt["logs"][lg][c][ix[f]] *= k
        tt["source"] = f"ATOL CALIBRATION ARM {name}: {fields} x {k}"
        save_calib(name, tt)
    scaled("half_scale", ("fx", "fy", "cx", "cy"), 0.5)
    scaled("focal105", ("fx", "fy"), 1.05)

    th = math.radians(3.0)
    qz = (math.cos(th / 2), 0.0, 0.0, math.sin(th / 2))                    # rotation about the ego z axis, (w, x, y, z)
    c_, s_ = math.cos(th), math.sin(th)

    def qmul(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)
    ty = copy.deepcopy(base)
    for lg in ty["logs"]:
        for c in ty["logs"][lg]:
            v = ty["logs"][lg][c]
            x, y = v[ix["tx"]], v[ix["ty"]]
            v[ix["tx"]], v[ix["ty"]] = c_ * x - s_ * y, s_ * x + c_ * y
            q = qmul(qz, (v[ix["qw"]], v[ix["qx"]], v[ix["qy"]], v[ix["qz"]]))
            v[ix["qw"]], v[ix["qx"]], v[ix["qy"]], v[ix["qz"]] = q
    ty["source"] = "ATOL CALIBRATION ARM S_yaw3: rig rotated 3 deg about ego z"
    save_calib("yaw3", ty)

    # ---- checkpoint variants
    import torch

    def head_noise(name, rel):
        out = f"{A}/ckpt_{name}.pt"
        if os.path.exists(out):
            return
        sd = torch.load(f"{FINAL}/model_final.pt", map_location="cpu", weights_only=False)
        w = sd["model"]["traj_head.2.weight"]
        g = torch.Generator().manual_seed(SEED)
        sd["model"]["traj_head.2.weight"] = w + torch.randn(w.shape, generator=g, dtype=w.dtype) * (rel * float(w.float().std()))
        torch.save(sd, out)
    head_noise("head_noise10", 0.10)
    head_noise("head_noise1", 0.01)
    man = {"keys48_sha256": sha256_file(kp), "n_keys": len(keys), "stride": STRIDE, "vehicles_in_table": vehs, "calib_other_vehicle_logs_changed": changed,
           "model_final_sha256": sha256_file(f"{FINAL}/model_final.pt"), "proxy_sets_sha256": sha256_file(f"{W}/sets_10075_proxy.jsonl"),
           "calib_original_sha256": sha256_file(f"{W}/bank_full/calib_table.json"), "pixels_original": PIXELS,
           "variants": {f: sha256_file(f"{A}/{f}") for f in sorted(os.listdir(A)) if f.startswith(("calib_", "ckpt_")) and os.path.isfile(f"{A}/{f}")}}
    assert man["model_final_sha256"] == MODEL_FINAL_SHA
    wr_json(f"{A}/prep_manifest.json", man)
    print("prep done", json.dumps(man)[:400])


# ------------------------------------------------------------------------------------------------------------------ dump (GPU)
def arm_inputs(name, var, A, W):
    ckpt = f"{FINAL}/model_final.pt"
    pin = True
    if var.get("ckpt") in ("head_noise10", "head_noise1"):
        ckpt, pin = f"{A}/ckpt_{var['ckpt']}.pt", False
    elif var.get("ckpt") == "snap021":
        ckpt, pin = f"{SNAPS}/snap_epoch021.pt", False
    elif var.get("ckpt") == "snap027":
        ckpt, pin = f"{SNAPS}/snap_epoch027.pt", False
    bank = f"{A}/bank_{var.get('bank', 'base')}"
    pix = f"{A}/pix_{var['pix']}" if "pix" in var else PIXELS
    calib = f"{A}/calib_{var['calib']}.json" if "calib" in var else f"{W}/bank_full/calib_table.json"
    return ckpt, pin, bank, pix, calib, int(var.get("batch", 4))


def dump(W, A, only, tol, py_dumper):
    waiter = f"{W}/wait_idle.py"
    env = dict(os.environ, PYTHONIOENCODING="utf-8", TMP=f"{W}/tmp", TEMP=f"{W}/tmp")
    os.makedirs(f"{W}/tmp", exist_ok=True)
    od = f"{A}/runs"
    os.makedirs(od, exist_ok=True)
    for name, kind, var in ARMS:
        if only and name not in only:
            continue
        out = f"{od}/{name}"
        mf = f"{out}/rl_fan_dump_cal_s0of1.manifest.json"
        if os.path.exists(mf) and json.load(open(mf)).get("finished") and os.path.exists(f"{out}/cli.json"):
            print(f"[skip] {name}: finished", flush=True)
            continue
        ckpt, pin, bank, pix, calib, batch = arm_inputs(name, var, A, W)
        # THE SHARED GPU: block until it is genuinely idle (wait_idle exits 0), and proceed only on 0.
        rc = subprocess.call([sys.executable, waiter])
        if rc != 0:
            print(f"[abort] wait_idle exited {rc}: the GPU is not provably idle; not running {name}", flush=True)
            sys.exit(7)
        cmd = [py_dumper, os.path.join(PKG, "refe", "rl_fan_dump.py"), "--ckpt", ckpt, *(["--expect-ckpt-sha256", MODEL_FINAL_SHA] if pin else []),
               "--keys", f"{A}/keys48.tsv", "--targets", bank, "--images", pix, "--calib", calib, "--out", out, "--tag", "cal", "--batch", str(batch),
               "--compare-sets", f"{W}/sets_10075_proxy.jsonl", "--compare-atol", str(tol), "--compare-min-matched", "40"]
        os.makedirs(out, exist_ok=True)
        t0 = time.time()
        with open(f"{out}/cli.log", "w", encoding="utf-8") as lf:
            r = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        wr_json(f"{out}/cli.json", {"arm": name, "kind": kind, "exit": r, "wall_sec": round(time.time() - t0, 1), "argv_tail": cmd[2:], "atol_candidate": tol})
        print(f"[{name}] exit {r} in {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------------------------------ analyse (CPU)
def relabel_step(src, dst, step):
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8", newline="\n") as g:
        for ln in f:
            r = json.loads(ln)
            r["ckpt_step"] = step
            g.write(json.dumps(r) + "\n")


def load_sets(path, step):
    """{key: (traj, yaw)} of the onpolicy_set lines of this ckpt_step (compare_sets' own reader, minus the comparison)."""
    out = {}
    marker = f'"ckpt_step": {int(step)}'.encode()
    with open(path, "rb") as f:
        for ln in f:
            if marker not in ln:
                continue
            r = json.loads(ln)
            if r.get("kind") != "onpolicy_set" or int(r["ckpt_step"]) != int(step):
                continue
            out.setdefault(RF.key_of(r), (r["traj"], r["yaw"]))
    return out


def xy_moments(a, b):
    """(sum |d|, sum d^2, n, n_nonzero) over every x / y value of two traj arrays (a shape mismatch is the caller's problem: the max-delta path already reads it as inf)."""
    sa = sq = 0.0
    n = nnz = 0
    for pa, pb in zip(a[0], b[0]):
        for qa, qb in zip(pa, pb):
            for u, w in zip(qa, qb):
                d = u - w
                sa += abs(d)
                sq += d * d
                n += 1
                nnz += d != 0.0
    return sa, sq, n, nnz


def arm_deltas(path, step, ref):
    """{key: (max |delta|, d_xy, d_yaw, mean |delta| over x/y, sum |d|, sum d^2, n, n_nonzero)}"""
    new = load_sets(path, step)
    d = {}
    for k, v in new.items():
        if k in ref:
            dxy, dyaw = RF.set_delta_strict(v, ref[k])
            sa, sq, n, nnz = xy_moments(v, ref[k])
            d[k] = (max(dxy, dyaw), dxy, dyaw, (sa / n if n else 0.0), sa, sq, n, nnz)
    return d


def summarise(d):
    ds = sorted(x[0] for x in d.values())
    q = {"p50": RF.quantile_nearest_rank(ds, 0.5), "p90": RF.quantile_nearest_rank(ds, 0.9), "p99": RF.quantile_nearest_rank(ds, 0.99), "max": ds[-1] if ds else 0.0}
    n = sum(x[6] for x in d.values())
    sets_mean = sorted(x[3] for x in d.values())
    return {"matched": len(ds), "bit_identical": sum(1 for x in ds if x == 0.0), "max_abs_delta": q["max"], "max_abs_delta_xy": max((x[1] for x in d.values()), default=0.0),
            "max_abs_delta_yaw": max((x[2] for x in d.values()), default=0.0), "delta_quantiles": q,
            "beyond_at": {str(g): sum(1 for x in ds if x > g) for g in GRID},
            # ---- distribution-wide statistics (informational: what a gate with MORE POWER than the per-set max could use)
            "xy_values": n, "xy_mean_abs": (sum(x[4] for x in d.values()) / n if n else 0.0), "xy_rms": (math.sqrt(sum(x[5] for x in d.values()) / n) if n else 0.0),
            "xy_frac_nonzero": (sum(x[7] for x in d.values()) / n if n else 0.0),
            "per_set_mean_abs_xy": {"p50": RF.quantile_nearest_rank(sets_mean, 0.5), "p90": RF.quantile_nearest_rank(sets_mean, 0.9), "max": sets_mean[-1] if sets_mean else 0.0}}


def beyond(d, tol):
    return sum(1 for x in d.values() if x[0] > tol)


def analyse(W, A, out_dir, tol_cand):
    proxy = f"{W}/sets_10075_proxy.jsonl"
    ref = load_sets(proxy, STEP)
    arms, deltas = {}, {}
    idn = f"{W}/identity"
    # ---- NULL arms from the identity run's dumps (CPU re-analysis, same estimator)
    for tag, path in (("N_full", f"{idn}/new_full/onpolicy_rlfan_full_s0of1.jsonl"), ("N_b1", f"{idn}/new_b1/onpolicy_rlfan_b1_s0of1.jsonl"),
                      ("N_b2", f"{idn}/new_b2/onpolicy_rlfan_b2_s0of1.jsonl"), ("N_b4", f"{idn}/new_b4/onpolicy_rlfan_b4_s0of1.jsonl"),
                      ("N_b4shift", f"{idn}/new_b4shift/onpolicy_rlfan_b4shift_s0of1.jsonl"), ("N_b4w2", f"{idn}/new_b4w2/onpolicy_rlfan_b4w2_s0of1.jsonl")):
        if os.path.exists(path):
            deltas[tag] = arm_deltas(path, STEP, ref)
            arms[tag] = dict(kind="null", source="identity run dump, CPU re-analysis", **summarise(deltas[tag]))
    # ---- this run's arms
    for name, kind, var in ARMS:
        out = f"{A}/runs/{name}"
        f = f"{out}/onpolicy_rlfan_cal_s0of1.jsonl"
        if not os.path.exists(f) or not os.path.exists(f"{out}/cli.json"):
            arms[name] = {"kind": kind, "missing": True}
            continue
        cli = json.load(open(f"{out}/cli.json"))
        man = json.load(open(f"{out}/rl_fan_dump_cal_s0of1.manifest.json"))
        step = int(man["ckpt_step"])
        path = f
        if step != STEP:
            path = f"{out}/relabelled_step{STEP}.jsonl"
            relabel_step(f, path, STEP)
        deltas[name] = arm_deltas(path, STEP, ref)
        cc = man.get("compare_sets") or {}
        arms[name] = dict(kind=kind, cli_exit=cli["exit"], ckpt_step=step, gpu=man.get("gpu"), torch=man.get("torch"), s_per_set_total=man.get("s_per_set_total"),
                          ckpt_sha256=man.get("ckpt_sha256"), cli_compare={k: cc.get(k) for k in ("atol", "reference_keys_matched", "within_atol", "beyond_atol", "max_abs_delta", "delta_quantiles")},
                          **summarise(deltas[name]))
        if step == STEP and cc:                              # the CLI's own numbers must equal this analysis' (two code paths, one answer)
            arms[name]["cli_vs_analysis_max_delta_equal"] = bool(abs(cc["max_abs_delta"] - arms[name]["max_abs_delta"]) <= 1e-12)
    nulls = {n: a for n, a in arms.items() if a.get("kind") == "null" and not a.get("missing")}
    floor_max = max(a["max_abs_delta"] for a in nulls.values())
    tol = math.ceil(2 * floor_max / 0.125 - 1e-12) * 0.125
    need_beyond = max(5, math.ceil(0.10 * N_KEYS))
    reasons, dec = [], {}
    for n, a in nulls.items():
        b = beyond(deltas[n], tol)
        if b != 0:
            reasons.append(f"null arm {n}: {b} sets beyond tol {tol}")
    for n, kind, var in ARMS:
        if kind != "decision":
            continue
        a = arms[n]
        if a.get("missing"):
            reasons.append(f"decision arm {n}: NOT RUN")
            continue
        b = beyond(deltas[n], tol)
        ok = a["max_abs_delta"] >= 10 * tol and b >= need_beyond
        dec[n] = {"max_abs_delta": a["max_abs_delta"], "ratio_to_tol": a["max_abs_delta"] / tol, "beyond_at_tol": b, "need_beyond": need_beyond, "separates": bool(ok)}
        if not ok:
            reasons.append(f"decision arm {n} does NOT separate at tol {tol}: max delta {a['max_abs_delta']:.4g} ({a['max_abs_delta'] / tol:.2f}x tol; need 10x), {b} sets beyond "
                           f"(need >= {need_beyond}); the largest tol at which it would still separate on max delta is {a['max_abs_delta'] / 10.0:.4g}")
    info = {n: {"max_abs_delta": arms[n]["max_abs_delta"], "ratio_to_tol": arms[n]["max_abs_delta"] / tol, "beyond_at_tol": beyond(deltas[n], tol), "caught_at_tol": beyond(deltas[n], tol) > 0}
            for n, kind, var in ARMS if kind in ("info", "info_null") and not arms[n].get("missing")}
    verdict = {"rule": "tol = 2 x floor_max rounded up to a multiple of 0.125; accept iff all null arms 0 beyond AND every decision arm max delta >= 10 x tol AND >= max(5, 10 %) sets beyond",
               "floor_max_over_null_arms": floor_max, "floor_arms": {n: a["max_abs_delta"] for n, a in nulls.items()}, "tol_candidate_used_in_cli_runs": tol_cand,
               "tol_by_rule": tol, "decision_arms": dec, "informational_arms": info, "failures": reasons, "CHOSEN_ATOL": (tol if not reasons else None)}
    wr_json(f"{out_dir}/arms_summary.json", arms)
    wr_json(f"{out_dir}/verdict.json", verdict)
    print(json.dumps(verdict, indent=1))
    lines = ["| arm | kind | CLI exit | sets matched | max delta | p50 | p90 | p99 | beyond@0.25 | @0.375 | @0.5 | @1 | @5 | max / 0.5 | x/y RMS delta | worst set's mean abs x/y delta |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, a in arms.items():
        if a.get("missing"):
            lines.append(f"| {n} | {a['kind']} | not run | | | | | | | | | | | | | |")
            continue
        q, b = a["delta_quantiles"], a["beyond_at"]
        lines.append(f"| {n} | {a['kind']} | {a.get('cli_exit', 'n/a (CPU)')} | {a['matched']} | {a['max_abs_delta']:.4g} | {q['p50']:.4g} | {q['p90']:.4g} | {q['p99']:.4g} | "
                     f"{b['0.25']} | {b['0.375']} | {b['0.5']} | {b['1.0']} | {b['5.0']} | {a['max_abs_delta'] / 0.5:.1f}x | {a['xy_rms']:.4g} | {a['per_set_mean_abs_xy']['max']:.4g} |")
    open(f"{out_dir}/RESULTS_TABLE.md", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["prep", "dump", "analyse", "all"])
    ap.add_argument("--work", default="D:/Projects/TanitAD-artifacts/_rlfan_work")
    ap.add_argument("--only", default="")
    ap.add_argument("--tol-candidate", type=float, default=TOL_CANDIDATE)
    ap.add_argument("--py-dumper", default="C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe")
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    W = a.work.replace("\\", "/")
    A = f"{W}/atol_cal"
    only = {x for x in a.only.split(",") if x}
    if a.phase in ("prep", "all"):
        prep(W, A)
    if a.phase in ("dump", "all"):
        dump(W, A, only, a.tol_candidate, a.py_dumper)
    if a.phase in ("analyse", "all"):
        analyse(W, A, a.out, a.tol_candidate)


if __name__ == "__main__":
    main()
