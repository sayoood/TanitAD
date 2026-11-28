#!/usr/bin/env python3
"""Proof that PATCH_train.diff / PATCH_gate.diff wire L_field (refe/field_loss.py) into refe/train.py and the launch gate, and that nothing else moves.

It APPLIES THE PATCHES ITSELF, to temporary copies of the CURRENT originals (`<pkg>/refe`, `code`, `splits`), so anyone can re-run it:

    python test_field_wiring.py                 # everything (CPU, minutes: ~25 tiny ViT-S trainer runs)
    python test_field_wiring.py --pkg <root>    # a package root other than the one this file sits in (<root>/refe, <root>/code, <root>/splits)
    python test_field_wiring.py --keep          # keep the temp workspace (printed)
    python test_field_wiring.py --part train    # only the trainer's proofs (a, b, c); --part gate: only the gate's

Prints `ZZTEST_PASS <n>` on success. Nothing here skips: a check that cannot run FAILS.

The setup is SELF-MADE (no dev-box data): a 6-scene synthetic bank (2 with a rank-1 twin) with a curving logged path per scene, and a label store written by the real
`aux_label_store.write_shard` whose SDF is a 2.5 m corridor around that scene's own logged path, so the labels are a function of the frame, are NOT symmetric in y, and one scene
(log5) has no label at all (valid = False).  The model is the real REFe, ViT-S, one camera, random frozen trunk (`--weights none`), synthetic images, CPU.

  (a) DEFAULT PATH BIT-IDENTICAL   the patched trainer with no field flag, with `--w-field 0.0`, and with `--w-field 0.0 --field-labels` (the loader delivering the label, the term
                                   off) gives exactly the losses, gradients (before AND after the clip), parameters, metric rows and loader items of the UNPATCHED trainer: torch.equal
                                   everywhere, on the fixed-scorer path AND the --labeller online (deferred-backward) path, plus a halt + resume of an UNPATCHED checkpoint under the
                                   patched trainer.  Controls: unpatched vs unpatched must be identical (the runs are deterministic) and `--w-field 1.0` must DIFFER (the comparison can see a change).
  (b) GRADIENT REACH               an L_field-only backward reaches traj_head / dec / queries / scene_proj / reg_compress / the LoRA and is None (no autograd path) on score_q_mlp,
                                   score_dec, score_head; inside the trainer, `--w-field 0` reads exactly 0.0 on every parameter and `--w-field W` is non-zero where it should be.
  (c) MUTATIONS THAT MUST GO RED   one line of the patched trainer is edited in a copy and the proofs must fail: the multiplier wired off by a factor (0.5 w), the term computed but not added
                                   (and added detached), the label y-flipped (caught by the alignment check, QC `aux_label_store.target_probe`), the valid mask dropped, the margin
                                   hard-coded; each on the fixed path, and the first two on the online path too.
  (d) THE GATE                     PATCH_gate.diff: the registry, the `refe_field` / `refe_field_ctrl` profiles and their refusals, G-DVB's observed rows (the margin and interp that
                                   `field_loss` actually RECEIVED, and the multiplier measured as the gradient that arrives at the term), G-LIVE's `field_L`, the SUITE, and the
                                   deliberate-regression arms; `test_launch_gate.py --static` and `test_field_loss.py` are run on the patched copy.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N = 0


def ok(cond, msg=""):
    global N
    assert cond, msg
    N += 1


def expect_red(label, thunk):
    """A mutation arm: the thunk must RAISE (some proof must fail); returns the message.  If it does not, the arm is inert and that is the failure."""
    try:
        thunk()
    except AssertionError as e:
        ok(True)
        print(f"    RED (as required): {label}: {str(e)[:160]}")
        return str(e)
    raise AssertionError(f"MUTATION ESCAPED: {label} -- no proof went red")


# ======================================================================================================================
# CHILD PROCESSES (each tree is imported fresh, in its own interpreter: `train`, `model`, ... exist in both trees under the same names)
# ======================================================================================================================
def _child_run(refe: str, outpt: str, pin: str | None, zero_traj: bool, argv: list) -> None:
    import io
    sys.path.insert(0, refe)
    import torch
    import train as T
    if pin:
        import aux_drivable_head as AUX
        sha, n = pin.split(":")
        AUX.PINNED_STORE_SHA256, AUX.PINNED_STORE_KEYS = sha, int(n)
    def dig(t):
        """sha256 of a tensor's dtype, shape and BYTES (or None): equal digests == bit-identical, stricter than torch.equal (which reads -0.0 == 0.0)."""
        if t is None:
            return None
        t = t.detach().contiguous().cpu()
        return hashlib.sha256(str(t.dtype).encode() + str(tuple(t.shape)).encode() + t.numpy().tobytes()).hexdigest()

    def groups_of(named: dict) -> dict:
        g = {}
        for n, t in named.items():
            top = n.split(".")[0]
            gn = "backbone.lora" if (top == "backbone" and n.rsplit(".", 1)[-1] in ("A", "B")) else top
            r = g.setdefault(gn, {"abs_sum": 0.0, "n": 0, "n_none": 0})
            r["n"] += 1
            if t is None:
                r["n_none"] += 1
            else:
                r["abs_sum"] += float(t.double().abs().sum())
        return g
    rec = {"names": None, "l_traj": [], "tgt": [], "pre": [], "pre_groups": [], "post": [], "fld": [], "fld_hook": [], "params": None, "rc": None, "metrics": [],
           "has_field_loss": hasattr(T, "field_loss")}
    holder = {}
    Orig = T.REFe

    class Rec(Orig):
        def __init__(self, cfg):
            super().__init__(cfg)
            holder["m"] = self

    T.REFe = Rec

    def names_of():
        return {id(p): n for n, p in holder["m"].named_parameters()}

    clip0 = torch.nn.utils.clip_grad_norm_

    def clip_rec(params, *a, **k):
        params = list(params)
        nm = names_of()
        gr = {nm[id(p)]: (None if p.grad is None else p.grad.detach().clone()) for p in params}
        rec["pre"].append({n: dig(t) for n, t in gr.items()})
        rec["pre_groups"].append(groups_of(gr))
        return clip0(params, *a, **k)

    torch.nn.utils.clip_grad_norm_ = clip_rec

    def step_pre(opt, a_, k_):
        nm = names_of()
        rec["post"].append({nm[id(p)]: dig(p.grad) for g in opt.param_groups for p in g["params"]})

    import importlib
    h = importlib.import_module("torch.optim.optimizer").register_optimizer_step_pre_hook(step_pre)
    wta0 = T.wta_loss

    def wta_rec(traj, target, *a, **k):
        r = wta0(traj, target, *a, **k)
        rec["l_traj"].append(dig(r[0]))
        rec["tgt"].append(target.detach().clone())
        if zero_traj:                                    # L_field-ONLY gradients inside the real trainer: the trajectory loss keeps its graph and contributes exactly 0
            r = (r[0] * 0.0,) + tuple(r[1:])
        return r

    T.wta_loss = wta_rec
    if hasattr(T, "field_loss"):
        fl0 = T.field_loss

        def fl_rec(traj, sdf, *a, **k):
            r = fl0(traj, sdf, *a, **k)
            out = r[0] if isinstance(r, tuple) else r
            rec["fld"].append({"traj": traj.detach().clone(), "sdf": sdf.detach().clone(), "valid": None if k.get("valid") is None else k["valid"].detach().clone(),
                               "margin": k.get("margin"), "interp": k.get("interp"), "value": out.detach().clone()})
            if out.requires_grad:                        # the gradient that ARRIVES at the term: d(total loss)/d(L_field) = w_field / accum, measured at the graph
                out.register_hook(lambda g: rec["fld_hook"].append(float(g)))
            return r

        T.field_loss = fl_rec
    a = T.build_parser().parse_args(argv)
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            rec["rc"] = T.run(a)
    except SystemExit as e:                              # TargetBank's refusals
        rec["rc"] = f"SystemExit: {e}"
    except Exception as e:                               # noqa: BLE001
        import traceback
        rec["rc"] = f"EXC: {traceback.format_exc()[-1500:]}"
    h.remove()
    rec["log"] = log.getvalue()[-40000:]
    if "m" in holder:
        rec["params"] = {n: dig(p) for n, p in holder["m"].named_parameters()}
        rec["names"] = [n for n, _ in holder["m"].named_parameters()]
    if a.out and (Path(a.out) / "metrics.jsonl").exists():
        rec["metrics"] = [json.loads(l) for l in (Path(a.out) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    torch.save(rec, outpt)


def _child_items(refe: str, outpt: str, pin: str | None, argv: list) -> None:
    """The loader's own output: dataset items and the default-collated first batch, for the tree under test."""
    sys.path.insert(0, refe)
    import torch
    import train as T
    from model import REFeConfig
    if pin:
        import aux_drivable_head as AUX
        sha, n = pin.split(":")
        AUX.PINNED_STORE_SHA256, AUX.PINNED_STORE_KEYS = sha, int(n)
    a = T.build_parser().parse_args(argv)
    cfg = REFeConfig.for_backbone("vits16")
    cfg.n_cameras, cfg.cameras = 1, cfg.cameras[:1]
    kw = {}
    if a.field_labels if hasattr(a, "field_labels") else False:
        kw = {"field_labels": a.field_labels, "field_shuffle_seed": a.field_shuffle_seed}
    ds = T.TargetBank(a.targets, None, cfg, synthetic=True, online=(a.labeller == "online"), **kw)
    def pack(x):
        t = x.detach().contiguous().cpu()
        return {"d": hashlib.sha256(str(t.dtype).encode() + str(tuple(t.shape)).encode() + t.numpy().tobytes()).hexdigest(), "dtype": t.dtype, "shape": tuple(t.shape),
                "t": t if t.numel() <= 40000 else None}
    items = [[pack(x) for x in ds[i]] for i in range(len(ds))]
    batch = [pack(x) for x in next(iter(torch.utils.data.DataLoader(ds, batch_size=3, shuffle=False)))]
    torch.save({"items": items, "batch": batch, "rows": [(r.get("log_name"), r.get("token"), r.get("rank")) for r in ds.rows]}, outpt)


def _child_gate(refe: str, outpt: str, pin: str, spec_json: str) -> None:
    """The launch gate's own instruments (gate_checks.instrumented_run -> check_dvb / check_live) on the REAL patched trainer, tiny scale, with and without each regression arm."""
    sys.path.insert(0, refe)
    import torch
    import aux_drivable_head as AUX
    import gate_checks as C
    import launch_gate as L
    sha, n = pin.split(":")
    AUX.PINNED_STORE_SHA256, AUX.PINNED_STORE_KEYS = sha, int(n)
    spec = json.loads(spec_json)
    prof = {**L.PROFILES["refe_rehearsal"], "live_terms": ("traj_L1", "score", "field_L"), "field_min_coverage": 0.8, "forbidden_flags": ()}
    out = {}

    def go(tag, argv, arm=None, profile=prof):
        work = Path(spec["work"]) / tag
        bld, smk = L.derived_argvs(profile, argv, work)
        b = C.instrumented_run(bld, mode="build", work=work / "b", arm=arm)
        s_ = C.instrumented_run(smk, mode="smoke", work=work / "s", arm=arm)
        dvb = C.check_dvb(argv, b, s_, profile, arm=arm)
        live = C.check_live(argv, s_, profile, arm=arm)
        strict = C.check_dvb(argv, b, s_, {**profile, "field_min_coverage": 0.999}, arm=arm)
        out[tag] = {"dvb": {"status": dvb["status"], "reasons": dvb["reasons"], "rows": [r for r in dvb["details"].get("rows", []) if r["dest"].startswith(("w_field", "field_"))],
                            "n_rows": dvb["details"].get("observed_rows")},
                    "strict_dvb": {"status": strict["status"], "reasons": strict["reasons"]},
                    "live": {"status": live["status"], "reasons": live["reasons"], "field_reach": live["details"].get("field_reach"),
                             "inext": live["details"].get("field_inext_logged"), "viol": live["details"].get("field_viol_logged")},
                    "smoke_rc": s_.rc, "build_rc": b.rc, "smoke_args_w": s_.args.w_field if s_.args is not None else None}
    go("clean", spec["argv"])
    go("weight_off", spec["argv"], arm="field_weight_off")
    go("term_detached", spec["argv"], arm="field_term_detached")
    go("weight_zero", spec["argv"], arm="field_weight_zero")
    go("w0_but_live_term_declared", spec["argv_w0"])
    torch.save(out, outpt)


def _child_wrlin(refe: str, outpt: str, pin: str, argv_json: str) -> None:
    """weight_rule's measurement: for every candidate the gradient of (w * L_field) taken by a REAL backward equals w * (the w = 1 gradient), BIT FOR BIT."""
    sys.path.insert(0, refe)
    sys.path.insert(0, str(HERE))
    import torch
    import weight_rule as WR
    from torch.utils.data import default_collate
    T, a, model, ds, dev = WR.build_instrumented(Path(refe), json.loads(argv_json), pin)
    head = [p for p in model.traj_head.parameters() if p.requires_grad]
    batch = default_collate([ds[i] for i in range(2)])
    img, ego, goal, tgt = (x.to(dev) for x in batch[:4])
    sdf, valid = batch[-2].to(dev), batch[-1].to(dev)
    traj = model(img, ego, goal)[0].float()
    res = {}
    lf1 = T.field_loss(traj, sdf, margin=a.field_margin, valid=valid, interp=a.field_interp)
    g1 = torch.autograd.grad(lf1, head, retain_graph=True, allow_unused=True)
    for w in WR.CANDIDATES:
        gw = torch.autograd.grad(lf1 * w, head, retain_graph=True, allow_unused=True)
        res[w] = all(torch.equal(x * w, y) for x, y in zip(g1, gw) if x is not None) and any(x is not None and float(x.abs().sum()) > 0 for x in g1)
    torch.save({"bitwise_linear": res}, outpt)


def _child_reach(refe: str, outpt: str, pin: str | None, store: str) -> None:
    """L_field-ONLY backward through the real model: which parameter groups receive gradient and which have NO autograd path."""
    sys.path.insert(0, refe)
    import torch
    import aux_drivable_head as AUX
    from field_loss import field_loss
    from model import REFe, REFeConfig
    sha, n = (pin or ":0").split(":")
    cfg = REFeConfig.for_backbone("vits16")
    cfg.n_cameras, cfg.cameras = 1, cfg.cameras[:1]
    torch.manual_seed(0)
    m = REFe(cfg)
    src = AUX.AuxLabelSource(store, pinned_sha256=sha, pinned_keys=int(n))
    keys = sorted(src.all_keys())[:2]
    sdf, valid = src.labels(keys)
    g = torch.Generator().manual_seed(3)
    img = torch.randn(2, 1, 3, cfg.img_h, cfg.img_w, generator=g) * 0.1
    ego = torch.randn(2, cfg.ego_dim, generator=g) * 0.1
    goal = torch.tensor([[30.0, 0.0, 60.0, 0.0]] * 2)
    out = {}
    for tag, w in (("w1", 1.0), ("w0", 0.0)):
        m.zero_grad(set_to_none=True)
        traj, score = m(img, ego, goal)[:2]
        loss = field_loss(traj.float(), sdf, valid=valid) * w
        loss.backward()
        grp = {}
        for nme, p in m.named_parameters():
            top = nme.split(".")[0]
            gname = "backbone.lora" if (top == "backbone" and nme.rsplit(".", 1)[-1] in ("A", "B")) else top
            r = grp.setdefault(gname, {"n": 0, "n_none": 0, "abs_sum": 0.0, "n_req": 0})
            r["n"] += 1
            r["n_req"] += int(p.requires_grad)
            if p.grad is None:
                r["n_none"] += 1
            else:
                r["abs_sum"] += float(p.grad.double().abs().sum())
        out[tag] = {"loss": float(loss), "groups": grp}
    torch.save(out, outpt)


# ======================================================================================================================
# the self-made tiny setup
# ======================================================================================================================
GYS = [0.0, 7.0, -7.0, 3.5, -3.5, 0.0]               # lateral displacement at 4 s per scene; scene 5 has NO label


def make_bank(root: Path) -> list:
    rng = np.random.default_rng(5)
    d = root / "bank"
    d.mkdir(parents=True, exist_ok=True)
    rows0, rows1 = [], []
    for s, gy in enumerate(GYS):
        t = np.arange(1, 21)
        x = 1.6 * t
        y = gy * (t / 20.0) ** 2
        yaw = np.arctan2(np.gradient(y), np.gradient(x))
        base = {"image": [f"log{s}/CAM_F0/a{s}.jpg"], "cameras": ["CAM_F0"], "ego": [float(v) for v in rng.normal(0, 0.3, 7)],
                "goal": [30.0, gy, 60.0, 2 * gy], "traj": [[float(x[i]), float(y[i]), float(yaw[i])] for i in range(20)],
                "step": 0, "token": f"tok{s:04x}", "log_name": f"log{s}", "scenario_type": "t", "source": "fixture"}
        rows0.append({**base, "rank": 0})
        if s < 2:
            rows1.append({**base, "rank": 1, "goal": [30.0, 0.8 * gy, 60.0, 1.6 * gy], "traj": [[float(x[i]), float(0.8 * y[i]), float(0.8 * yaw[i])] for i in range(20)],
                          "aug": {"kind": "lane_rank", "value": 1, "divergence_m": 2.0}})
    for nm, rows in (("targets_rank0.jsonl", rows0), ("targets_rank1.jsonl", rows1)):
        (d / nm).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    return rows0


def corridor_sdf_i8(traj, half_width: float = 2.5):
    """int8 SDF [160, 128] (+ inside, 0.1 m unit, index [ix, iy]) of the corridor of half-width `half_width` around the logged path (origin + 20 poses), in the label frame."""
    import aux_label_store as ALS
    pts = np.vstack([[0.0, 0.0], np.asarray(traj)[:, :2]])
    dense = np.vstack([pts[i] + (pts[i + 1] - pts[i]) * np.linspace(0, 1, 17)[:, None] for i in range(len(pts) - 1)])
    gx = ALS.X0 + (np.arange(ALS.NX) + 0.5) * ALS.RES
    gy = ALS.Y0 + (np.arange(ALS.NY) + 0.5) * ALS.RES
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    cells = np.stack([X.ravel(), Y.ravel()], axis=1)
    dist = np.sqrt(((cells[:, None, :] - dense[None, :, :]) ** 2).sum(-1)).min(1).reshape(ALS.NX, ALS.NY)
    return ALS.quantize_sdf(half_width - dist)


def make_store(root: Path, rows0: list, refe: str) -> dict:
    """A REAL store on disk (real shard writer, real manifest arithmetic): scenes 0-4 labelled, scene 5 absent."""
    sys.path.insert(0, refe)
    import aux_label_store as ALS
    store = root / "store"
    files, n = {}, 0
    for r in rows0:
        if r["log_name"] == "log5":
            continue
        sdf = corridor_sdf_i8(r["traj"])
        nbytes, sha = ALS.write_shard(store / "shards" / f"{r['log_name']}.aux1", r["log_name"], "tiny",
                                      [{"t": r["token"], "blob": ALS.encode_sdf(sdf), "s": "bank", "p": [0.0, 0.0, 0.0], "q": None}])
        files[f"shards/{r['log_name']}.aux1"] = {"sha256": sha, "bytes": nbytes}
        n += 1
    man = {"schema": ALS.SCHEMA, "raster": ALS.raster_spec(), "counts": {"stored_keys": n}, "leakage": {"token_intersection": 0, "log_intersection": 0}, "files": files}
    man["store_sha256"] = hashlib.sha256("\n".join(f"{k}:{v['sha256']}" for k, v in sorted(files.items())).encode()).hexdigest()
    (store / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return {"store": str(store), "pin": f"{man['store_sha256']}:{n}", "sha": man["store_sha256"], "n": n}


# ======================================================================================================================
# the workspace: current originals + the patches
# ======================================================================================================================
def copy_pkg(pkg: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ("refe", "code", "splits"):
        shutil.copytree(pkg / sub, dst / sub, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pt", "*.safetensors"))


def apply_patch(root: Path, diff: Path) -> str:
    r = subprocess.run(["patch", "-p1", "--no-backup-if-mismatch", "-i", str(diff)], cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, f"patch {diff.name} did not apply cleanly:\n{r.stdout}\n{r.stderr}"
    return r.stdout


def run_child(mode: str, refe: Path, outpt: Path, argv=(), *, pin=None, store=None, zero_traj=False, spec=None) -> dict:
    import torch
    cmd = [sys.executable, str(HERE / "test_field_wiring.py"), "--child", mode, "--refe", str(refe), "--out", str(outpt)]
    if pin:
        cmd += ["--pin", pin]
    if store:
        cmd += ["--store", store]
    if spec:
        cmd += ["--spec", spec]
    if zero_traj:
        cmd += ["--zero-traj"]
    cmd += ["--"] + list(argv)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONHASHSEED="0")
    env.pop("TANITAD_GATE_ARM", None)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(refe))
    assert r.returncode == 0 and outpt.exists(), f"child {mode} failed rc={r.returncode}\n{r.stdout[-1500:]}\n{r.stderr[-2500:]}"
    rec = torch.load(outpt, map_location="cpu", weights_only=False)
    outpt.unlink()
    return rec


# ======================================================================================================================
# comparisons
# ======================================================================================================================
def eq(a, b) -> bool:
    """digests (strings) or None: bit-identical iff equal."""
    return a == b


def diff_runs(A: dict, B: dict, keys=("traj_L1", "score", "pos_L1", "yaw_L1")) -> list:
    """Every way run B differs from run A: losses per micro-batch, gradients before and after the clip, final parameters, logged metrics.  [] = bit-identical."""
    out = []
    if A["rc"] != 0 or B["rc"] != 0:
        return [f"rc {A['rc']!r} / {B['rc']!r}"]
    if len(A["l_traj"]) != len(B["l_traj"]) or not all(eq(x, y) for x, y in zip(A["l_traj"], B["l_traj"])):
        out.append("l_traj per micro-batch")
    for which in ("pre", "post"):
        if len(A[which]) != len(B[which]):
            out.append(f"{which}-clip grad step count")
            continue
        for s, (ga, gb) in enumerate(zip(A[which], B[which])):
            if set(ga) != set(gb):
                out.append(f"{which}-clip grad names step {s}")
                continue
            bad = [k for k in ga if not eq(ga[k], gb[k])]
            if bad:
                out.append(f"{which}-clip grad step {s}: {len(bad)} tensors differ, e.g. {bad[:2]}")
    bad = [k for k in A["params"] if not eq(A["params"][k], B["params"].get(k))]
    if set(A["params"]) != set(B["params"]) or bad:
        out.append(f"final parameters: {len(bad)} differ, e.g. {bad[:2]}")
    ra = [r for r in A["metrics"] if "traj_L1" in r]
    rb = [r for r in B["metrics"] if "traj_L1" in r]
    if len(ra) != len(rb) or any(x[k] != y[k] for x, y in zip(ra, rb) for k in keys):
        out.append("logged step rows")
    return out


def group_sums(rec: dict, step: int = 0) -> dict:
    """per parameter group: the |grad| sum, the tensor count and the number with NO gradient (None), before the clip, from the child."""
    return rec["pre_groups"][step]


# ======================================================================================================================
# the trainer's proofs
# ======================================================================================================================
BASE = ["--backbone", "vits16", "--weights", "none", "--n-cameras", "1", "--synthetic", "--cpu", "--batch", "2", "--accum", "2", "--steps", "2",
        "--lr", "5e-5", "--log-every", "1", "--ckpt-every-min", "999"]


def argv_fixed(W: Path, out: str, extra=()):
    return BASE + ["--targets", str(W / "data" / "bank"), "--scorer-targets", str(W / "data" / "bank_none"), "--out", out] + list(extra)


def argv_online(W: Path, out: str, extra=()):
    return BASE + ["--targets", str(W / "data" / "bank"), "--scorer-targets", str(W / "data" / "bank_none"), "--yaw-wrap", "off", "--nav-mode", "command",
                   "--labeller", "online", "--label-backend", "synthetic", "--label-workers", "2", "--scorer-sees-trunk", "--out", out] + list(extra)


def field_flags(S: dict, w: float, *, margin=None, interp=None, shuffle=None):
    f = ["--w-field", repr(w), "--field-labels", S["store"]]
    if margin is not None:
        f += ["--field-margin", repr(margin)]
    if interp is not None:
        f += ["--field-interp", interp]
    if shuffle is not None:
        f += ["--field-shuffle-seed", str(shuffle)]
    return f


class Env:
    """The workspace and a cache of child runs."""

    def __init__(self, W: Path, orig: Path, new: Path, S: dict):
        self.W, self.orig, self.new, self.S = W, orig, new, S
        self._runs: dict = {}
        self.k = 0

    def run(self, tree: Path, tag: str, argv: list, *, zero_traj=False, mode="run", cache=True, pin=None) -> dict:
        key = (str(tree), tag)
        if cache and key in self._runs:
            return self._runs[key]
        self.k += 1
        outd = self.W / "runs" / f"{self.k:02d}_{tag}"
        argv = [t if t != "@OUT@" else str(outd) for t in argv]
        rec = run_child(mode, tree / "refe", self.W / "runs" / f"{self.k:02d}_{tag}.pt", argv, pin=pin or self.S["pin"], zero_traj=zero_traj)
        if cache:
            self._runs[key] = rec
        return rec


def part_a(E: Env) -> None:
    import torch
    print("(a) DEFAULT PATH BIT-IDENTICAL")
    F = lambda extra=(): argv_fixed(E.W, "@OUT@", extra)  # noqa: E731
    O = lambda extra=(): argv_online(E.W, "@OUT@", extra)  # noqa: E731
    S = E.S
    A1 = E.run(E.orig, "orig_fixed_1", F())
    A2 = E.run(E.orig, "orig_fixed_2", F(), cache=False)
    ok(A1["rc"] == 0 and len(A1["pre"]) == 2 and len(A1["l_traj"]) == 4, (A1["rc"], A1["log"][-400:]))
    ok(diff_runs(A1, A2) == [], diff_runs(A1, A2))          # CONTROL: the unpatched trainer is deterministic, so a difference below is the patch's
    ok(not A1["has_field_loss"])
    B0 = E.run(E.new, "new_fixed_default", F())
    ok(B0["has_field_loss"])
    ok(diff_runs(A1, B0) == [], diff_runs(A1, B0))
    ok(diff_runs(A1, E.run(E.new, "new_fixed_w0", F(["--w-field", "0.0"]))) == [])
    D0 = E.run(E.new, "new_fixed_w0_labels", F(field_flags(S, 0.0)))
    ok(diff_runs(A1, D0) == [], diff_runs(A1, D0))          # the label delivered, the term off: nothing may move
    ok(not D0["fld"] and not any("field_L" in r for r in D0["metrics"]), "w_field 0 must not compute the term")
    # POSITIVE CONTROL: the comparison can see a change
    P1 = E.run(E.new, "new_fixed_w1", F(field_flags(S, 1.0)))
    ok(P1["rc"] == 0 and len(P1["fld"]) == 4, (P1["rc"], P1["log"][-500:]))
    d = diff_runs(A1, P1)
    ok(d and any("final parameters" in x for x in d) and any("grad" in x for x in d), d)
    # the label source travels to DataLoader WORKERS (spawn on Windows pickles the dataset): --workers 2 reproduces --workers 0 bit for bit, with the term on
    P2w = E.run(E.new, "new_fixed_w1_workers2", F(field_flags(S, 1.0) + ["--workers", "2"]))
    ok(P2w["rc"] == 0 and len(P2w["fld"]) == 4 and diff_runs(P1, P2w) == [], (P2w["rc"], P2w["log"][-300:], diff_runs(P1, P2w)))
    # the online (deferred-backward) path
    AO1 = E.run(E.orig, "orig_online", O())
    AO2 = E.run(E.orig, "orig_online_2", O(), cache=False)
    ok(AO1["rc"] == 0 and diff_runs(AO1, AO2) == [], (AO1["rc"], AO1["log"][-500:], diff_runs(AO1, AO2)))
    ok(diff_runs(AO1, E.run(E.new, "new_online_default", O())) == [])
    ok(diff_runs(AO1, E.run(E.new, "new_online_w0_labels", O(field_flags(S, 0.0)))) == [])
    PO = E.run(E.new, "new_online_w1", O(field_flags(S, 1.0)))
    ok(PO["rc"] == 0 and len(PO["fld"]) == 4 and diff_runs(AO1, PO), PO["rc"])
    # the loader: the item tuple and the collated batch are byte-for-byte what they were
    ia = run_child("items", E.orig / "refe", E.W / "runs" / "items_orig.pt", F(), pin=S["pin"])
    ib = run_child("items", E.new / "refe", E.W / "runs" / "items_new.pt", F(), pin=S["pin"])
    same = lambda x, y: x["d"] == y["d"]  # noqa: E731
    ok(len(ia["items"]) == len(ib["items"]) == 8 and all(len(x) == 9 and len(y) == 9 and all(same(p, q) for p, q in zip(x, y)) for x, y in zip(ia["items"], ib["items"])))
    ok(len(ia["batch"]) == len(ib["batch"]) == 9 and all(same(p, q) for p, q in zip(ia["batch"], ib["batch"])))
    ioa = run_child("items", E.orig / "refe", E.W / "runs" / "items_orig_on.pt", O(), pin=S["pin"])
    iob = run_child("items", E.new / "refe", E.W / "runs" / "items_new_on.pt", O(), pin=S["pin"])
    ok(len(ioa["items"][0]) == len(iob["items"][0]) == 10 and all(same(p, q) for x, y in zip(ioa["items"], iob["items"]) for p, q in zip(x, y)))
    # with the labels the items carry exactly two more elements, the LAST TWO, and the first nine are unchanged
    il = run_child("items", E.new / "refe", E.W / "runs" / "items_new_lab.pt", F(field_flags(S, 1.0)), pin=S["pin"])
    ok(all(len(x) == 11 and all(same(p, q) for p, q in zip(x[:9], y)) for x, y in zip(il["items"], ia["items"])))
    ok(all(x[9]["dtype"] == torch.int8 and x[9]["shape"] == (160, 128) and x[10]["dtype"] == torch.bool and x[10]["shape"] == () for x in il["items"]))
    ok(il["batch"][9]["dtype"] == torch.int8 and il["batch"][9]["shape"] == (3, 160, 128) and il["batch"][10]["dtype"] == torch.bool and il["batch"][10]["shape"] == (3,))
    ilo = run_child("items", E.new / "refe", E.W / "runs" / "items_new_lab_on.pt", O(field_flags(S, 1.0)), pin=S["pin"])
    ok(all(len(x) == 12 and same(x[9], y[9]) for x, y in zip(ilo["items"], ioa["items"])), "online: the row index stays at position 9, the label pair is LAST")
    # valid: the unlabelled scene (log5) is False, every other scene True; twins share their frame's label
    val = {r: bool(x[10]["t"]) for r, x in zip(il["rows"], il["items"])}
    ok(all(v == (r[0] != "log5") for r, v in val.items()), val)
    twins = [x for r, x in zip(il["rows"], il["items"]) if r[0] == "log0"]
    ok(len(twins) == 2 and same(twins[0][9], twins[1][9]), "goal twins share their frame's label")
    # an OLD checkpoint resumes under the patched trainer: halt after step 1 (UNPATCHED), resume (PATCHED), compare with the uninterrupted UNPATCHED run
    out_r = E.W / "runs" / "resume_x"
    h = run_child("run", E.orig / "refe", E.W / "runs" / "resume_orig_halt.pt", argv_fixed(E.W, str(out_r), ["--halt-after-steps", "1", "--ckpt-every-steps", "1"]), pin=S["pin"])
    ok((out_r / "ckpt_last.pt").exists(), h["log"][-500:])
    ident_old = torch.load(out_r / "ckpt_last.pt", map_location="cpu", weights_only=False)["identity"]
    ok("w_field" not in ident_old["args"] and "field_store_sha256" not in ident_old)       # an old checkpoint: none of the new keys
    r2 = run_child("run", E.new / "refe", E.W / "runs" / "resume_new.pt", argv_fixed(E.W, str(out_r), ["--resume"]), pin=S["pin"])
    ok(r2["rc"] == 0 and "RESUMED" in r2["log"], r2["log"][-800:])
    ok(A1["params"] == r2["params"], "halt (unpatched) + resume (patched) == the uninterrupted unpatched run")
    # ... and a resume that CHANGES the term is refused (the identity records it), as is one that changes the store's content
    out_f = E.W / "runs" / "resume_f"
    run_child("run", E.new / "refe", E.W / "runs" / "resume_f1.pt", argv_fixed(E.W, str(out_f), ["--halt-after-steps", "1", "--ckpt-every-steps", "1"] + field_flags(S, 1.0)), pin=S["pin"])
    r3 = run_child("run", E.new / "refe", E.W / "runs" / "resume_f2.pt", argv_fixed(E.W, str(out_f), ["--resume"] + field_flags(S, 2.0)), pin=S["pin"])
    ok(r3["rc"] == 4 and "--w-field: 1.0 -> 2.0" in r3["log"], (r3["rc"], r3["log"][-400:]))
    r4 = run_child("run", E.new / "refe", E.W / "runs" / "resume_f3.pt", argv_fixed(E.W, str(out_f), ["--resume"] + field_flags(S, 1.0)), pin=S["pin"])
    ok(r4["rc"] == 0 and "RESUMED" in r4["log"], r4["log"][-400:])
    ok(r4["params"] == P1["params"], "halt + resume of a FIELD run == the uninterrupted field run (G-CKPT's claim, with the term on)")


def part_refusals(E: Env) -> None:
    print("refusals and the shuffle control")
    S = E.S
    F = lambda extra=(): argv_fixed(E.W, "@OUT@", extra)  # noqa: E731
    r = E.run(E.new, "ref_nolabels", F(["--w-field", "1.0"]))
    ok(r["rc"] == 4 and "needs --field-labels" in r["log"], (r["rc"], r["log"][-300:]))
    r = E.run(E.new, "ref_shuffle_nolabels", F(["--field-shuffle-seed", "1"]))
    ok(r["rc"] == 4 and "--field-shuffle-seed needs --field-labels" in r["log"])
    r = E.run(E.new, "ref_negative", F(["--w-field", "-1.0", "--field-labels", S["store"]]))
    ok(r["rc"] == 4 and "must be a finite number >= 0" in r["log"])
    r = E.run(E.new, "ref_nan", F(["--w-field", "nan", "--field-labels", S["store"]]))
    ok(r["rc"] == 4)
    # a store that is not the pinned one is refused by the reader, as a SystemExit naming the flag
    import torch
    bad = E.run(E.new, "ref_badpin", F(field_flags(S, 1.0)), pin="0" * 64 + ":5")
    ok(isinstance(bad["rc"], str) and "SystemExit" in bad["rc"] and "is refused" in bad["rc"] and "not the pinned" in bad["rc"], bad["rc"])
    # the label-shuffled control: every delivered label is the label of ANOTHER key, no key keeps its own, the multiset is preserved
    ia = run_child("items", E.new / "refe", E.W / "runs" / "items_sh_plain.pt", F(field_flags(S, 1.0)), pin=S["pin"])
    ib = run_child("items", E.new / "refe", E.W / "runs" / "items_sh.pt", F(field_flags(S, 1.0, shuffle=20261011)), pin=S["pin"])
    pa = {r: x[9] for r, x in zip(ia["rows"], ia["items"]) if r[2] == 0 and r[0] != "log5"}
    pb = {r: x for r, x in zip(ib["rows"], ib["items"]) if r[2] == 0}
    own = [r for r in pa if pb[r][9]["d"] == pa[r]["d"]]
    ok(own == [], f"keys that kept their own label under the derangement: {own}")
    ok(all(bool(pb[r][10]["t"]) == (r[0] != "log5") for r in pb), "a frame whose key is in the store draws SOME label under the derangement; a frame outside the store has none (no KeyError)")
    ok(sorted(pb[r][9]["d"] for r in pb if r[0] != "log5") == sorted(pa[r]["d"] for r in pa), "the multiset of delivered labels is the store's own (a derangement permutes, it does not invent)")


def part_b(E: Env) -> None:
    import torch
    print("(b) GRADIENT REACH")
    S = E.S
    reach = run_child("reach", E.new / "refe", E.W / "runs" / "reach.pt", pin=S["pin"], store=S["store"])
    g1, g0 = reach["w1"]["groups"], reach["w0"]["groups"]
    ok(reach["w1"]["loss"] > 0.0, reach["w1"]["loss"])
    for grp in ("traj_head", "dec", "queries", "scene_proj", "reg_compress", "backbone.lora"):
        ok(g1[grp]["abs_sum"] > 0.0 and g1[grp]["n_none"] == 0, (grp, g1[grp]))
    for grp in ("score_q_mlp", "score_dec", "score_head"):
        ok(g1[grp]["abs_sum"] == 0.0 and g1[grp]["n_none"] == g1[grp]["n"], (grp, g1[grp]))        # NO autograd path at all (grad is None), not merely a zero
    ok(g1["backbone"]["abs_sum"] == 0.0 if "backbone" in g1 else True)
    # the zero control of the probe itself: weight 0 reads exactly 0.0 on EVERY parameter that has a path (and None elsewhere)
    ok(all(v["abs_sum"] == 0.0 for v in g0.values()), {k: v for k, v in g0.items() if v["abs_sum"] != 0.0})
    # inside the real trainer: L_field-only gradients (trajectory loss zeroed, --score-w 0), non-zero where it should be, exactly zero control at w = 0
    F = lambda extra=(): argv_fixed(E.W, "@OUT@", extra) + ["--score-w", "0.0"]  # noqa: E731
    on = E.run(E.new, "reach_trainer_w4", F(field_flags(S, 4.0)), zero_traj=True)
    off = E.run(E.new, "reach_trainer_w0", F(field_flags(S, 0.0)), zero_traj=True)
    ok(on["rc"] == 0 and off["rc"] == 0, (on["rc"], on["log"][-400:], off["rc"]))
    gon, goff = group_sums(on), group_sums(off)
    for grp in ("traj_head", "dec", "queries", "scene_proj", "reg_compress", "backbone.lora"):
        ok(gon[grp]["abs_sum"] > 0.0, (grp, gon[grp]))
    for grp in ("score_q_mlp", "score_dec", "score_head"):
        ok(gon[grp]["abs_sum"] == 0.0, (grp, gon[grp]))
    ok(all(v["abs_sum"] == 0.0 for v in goff.values()), {k: v for k, v in goff.items() if v["abs_sum"] != 0.0})
    ok(len(off["fld"]) == 0, "--w-field 0 must not even call the term")
    # the multiplier, measured at the graph: the gradient arriving at L_field is w / accum on every micro-batch
    h = on["fld_hook"]
    ok(len(h) == 4 and all(abs(v * 2 - 4.0) <= 1e-6 * 4.0 for v in h), h)
    # the logged telemetry is the term's own value, and the recomputation from the recorded inputs reproduces it bitwise
    sys.path.insert(0, str(E.orig / "refe"))
    rows = [r for r in on["metrics"] if "field_L" in r]
    ok(len(rows) == 2 and all(r["field_w"] == 4.0 and math.isfinite(r["field_L"]) and 0.0 <= r["field_viol"] <= 1.0 and 0.0 <= r["field_inext"] <= 1.0 for r in rows), rows)
    ok(all(r["field_L"] > 0.0 for r in rows), "an untrained fan violates a corridor: field_L > 0 at step 0")


def recompute_field_L(E: Env, rec: dict, margin=0.25, interp="5hz") -> list:
    """field_loss recomputed (by the ORIGINAL module, in this process) from the traj / sdf / valid the trainer passed, to be compared with the logged field_L of the step's last micro-batch."""
    import importlib
    import torch
    for m in ("field_loss",):
        sys.modules.pop(m, None)
    sys.path.insert(0, str(E.orig / "refe"))
    FL = importlib.import_module("field_loss")
    vals = [float(FL.field_loss(f["traj"], f["sdf"], margin=margin, valid=f["valid"], interp=interp)) for f in rec["fld"]]
    sys.path.remove(str(E.orig / "refe"))
    return vals


def alignment(rec: dict, E: Env) -> float:
    """QC (aux_label_store.target_probe): the share of the logged target path's points that fall inside the label the loader delivered with them, over the labelled samples."""
    sys.path.insert(0, str(E.orig / "refe"))
    import aux_label_store as ALS
    ins = tot = 0
    for f, tgt in zip(rec["fld"], rec["tgt"]):
        for b in range(tgt.shape[0]):
            if f["valid"] is not None and not bool(f["valid"][b]):
                continue
            i, n = ALS.target_probe(f["sdf"][b].numpy(), tgt[b, :, :2].numpy())
            ins, tot = ins + i, tot + n
    sys.path.remove(str(E.orig / "refe"))
    return ins / max(tot, 1)


def proofs_for_mutation(E: Env, tree: Path, tag: str, *, online: bool, w: float = 4.0) -> None:
    """The proofs a mutation must trip.  Raises AssertionError at the first one that fails."""
    S = E.S
    base = argv_online if online else argv_fixed
    on = E.run(tree, f"{tag}_w", base(E.W, "@OUT@", field_flags(S, w) + ["--score-w", "0.0"]), zero_traj=True, cache=False)
    assert on["rc"] == 0, on["log"][-600:]
    n_micro = 4
    # 1. the term reaches the graph, with the registered multiplier
    assert len(on["fld_hook"]) == n_micro, f"the gradient reached L_field on {len(on['fld_hook'])} of {n_micro} micro-batches (the term was computed but never entered the loss)"
    assert all(abs(v * 2 - w) <= 1e-6 * w for v in on["fld_hook"]), f"the multiplier arriving at L_field is {[v * 2 for v in on['fld_hook']]}, not w_field = {w}"
    # 2. and the parameters feel it
    g = group_sums(on)
    assert g["traj_head"]["abs_sum"] > 0.0, "L_field-only gradient on traj_head is zero"
    # 3. what the term RECEIVED is what was declared (before anything that could mask a missing valid mask)
    assert all(f["margin"] == 0.25 and f["interp"] == "5hz" for f in on["fld"]), f"field_loss received margin/interp {[(f['margin'], f['interp']) for f in on['fld']][:1]}"
    assert any(f["valid"] is not None and not bool(f["valid"].all()) for f in on["fld"]) and all(f["valid"] is not None for f in on["fld"]), \
        "the valid mask was not passed (or no unlabelled sample was drawn)"
    # 4. what the loader delivers is aligned with the logged path, and the telemetry is the term's own value
    al = alignment(on, E)
    assert al >= 0.95, f"alignment {al:.3f} < 0.95: the delivered label does not contain the logged path (misaligned / flipped label)"
    rows = [r for r in on["metrics"] if "field_L" in r]
    rec = recompute_field_L(E, on)
    assert len(rows) == 2 and all(r["field_L"] == float(v) for r, v in zip(rows, (rec[1], rec[3]))), f"logged field_L {[r['field_L'] for r in rows]} != recomputed {rec}"


def mutate(src: Path, dst: Path, old: str, new: str, count: int = 1) -> None:
    s = src.read_text(encoding="utf-8")
    assert s.count(old) == count, f"mutation anchor appears {s.count(old)}x, expected {count}: {old!r}"
    dst.write_text(s.replace(old, new), encoding="utf-8", newline="\n")


def part_c(E: Env) -> None:
    print("(c) MUTATIONS THAT MUST GO RED")
    # the clean tree passes every proof (so a RED below is the mutation's)
    proofs_for_mutation(E, E.new, "clean_fixed", online=False)
    proofs_for_mutation(E, E.new, "clean_online", online=True)
    ok(True)
    muts = [
        ("multiplier off by a factor (0.5 w), fixed path", False, "loss = loss + a.w_field * l_field", "loss = loss + 0.5 * a.w_field * l_field"),
        ("term computed but NOT added, fixed path", False, "loss = loss + a.w_field * l_field", "loss = loss + 0.0 * a.w_field"),
        ("term added DETACHED, fixed path", False, "loss = loss + a.w_field * l_field", "loss = loss + a.w_field * l_field.detach()"),
        ("multiplier off by a factor (0.5 w), online path", True, "tot_ = tot_ + a.w_field * P[\"l_field\"]", "tot_ = tot_ + 0.5 * a.w_field * P[\"l_field\"]"),
        ("term computed but NOT added, online path", True, "tot_ = tot_ + a.w_field * P[\"l_field\"]", "tot_ = tot_ + 0.0 * a.w_field"),
        ("label Y-FLIPPED in the loader", False, "item = item + (lab[0], vld[0])", "item = item + (lab[0].flip(1), vld[0])"),
        ("valid mask dropped", False, "valid=valid_t, interp=a.field_interp", "valid=None, interp=a.field_interp"),
        ("margin hard-coded", False, "margin=a.field_margin, valid=valid_t", "margin=0.5, valid=valid_t"),
        ("interp ignored", False, "interp=a.field_interp, return_parts=True", "interp='navsim2hz', return_parts=True"),
    ]
    for i, (label, online, old, new) in enumerate(muts):
        mroot = E.W / f"mut{i}"
        shutil.copytree(E.new / "refe", mroot / "refe", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        mutate(E.new / "refe" / "train.py", mroot / "refe" / "train.py", old, new)
        expect_red(label, lambda mroot=mroot, i=i, online=online: proofs_for_mutation(E, mroot, f"mut{i}", online=online))
        shutil.rmtree(mroot, ignore_errors=True)


def run_py(refe: Path, script: str, *args, timeout=3000) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop("TANITAD_GATE_ARM", None)
    return subprocess.run([sys.executable, script, *args], cwd=str(refe), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def part_d(E: Env) -> None:
    print("(d) THE GATE")
    S = E.S
    import re
    # the existing static suite, before and after, and the module's own tests, on the PATCHED copy
    before = run_py(E.orig / "refe", "test_launch_gate.py", "--static", "--no-banked")
    m0 = re.search(r"ZZLG_STATIC_PASS (\d+)", before.stdout)
    ok(before.returncode == 0 and m0, before.stdout[-800:] + before.stderr[-800:])
    after = run_py(E.new / "refe", "test_launch_gate.py", "--static", "--no-banked")
    m1 = re.search(r"ZZLG_STATIC_PASS (\d+)", after.stdout)
    ok(after.returncode == 0 and m1, after.stdout[-800:] + after.stderr[-800:])
    n0, n1 = int(m0.group(1)), int(m1.group(1))
    ok(n1 > n0, (n0, n1))
    print(f"    ZZLG_STATIC_PASS before {n0}  after {n1}  (+{n1 - n0})")
    fl = run_py(E.new / "refe", "test_field_loss.py")
    m2 = re.search(r"ZZTEST_PASS (\d+)", fl.stdout)
    ok(fl.returncode == 0 and m2 and int(m2.group(1)) == 42, fl.stdout[-600:] + fl.stderr[-600:])
    print(f"    test_field_loss.py: ZZTEST_PASS {m2.group(1)}")
    # the gate's own instruments on the REAL patched trainer (tiny scale), clean and with each deliberate-regression arm
    argv = argv_online(E.W, str(E.W / "gate_launch_out"), field_flags(S, 4.0))
    argv_w0 = argv_online(E.W, str(E.W / "gate_launch_out0"), field_flags(S, 0.0))
    spec = json.dumps({"argv": argv, "argv_w0": argv_w0, "work": str(E.W / "gate_work")})
    g = run_child("gate", E.new / "refe", E.W / "runs" / "gate.pt", pin=S["pin"], spec=spec)
    c = g["clean"]
    ok(c["build_rc"] == "STOPPED_AT_BUILD" and c["smoke_rc"] == 0, c)
    ok(c["dvb"]["status"] == "PASS" and c["dvb"]["reasons"] == [], c["dvb"]["reasons"])
    rows = c["dvb"]["rows"]
    ok(len(rows) >= 13 and all(r["ok"] for r in rows), [r for r in rows if not r["ok"]])
    ok(any("measured at the graph" in r["how"] and r["ok"] for r in rows) and any("actually received" in r["how"] for r in rows))
    ok(c["live"]["status"] == "PASS" and c["live"]["reasons"] == [], c["live"]["reasons"])
    fr = c["live"]["field_reach"]
    ok(fr["ran"] and fr["labelled"] > 0 and all(fr["declared"]["groups"][k]["abs_sum"] > 0 for k in ("traj_head", "dec", "queries", "scene_proj", "reg_compress", "backbone.lora"))
       and all(fr["declared"]["groups"][k]["n_none"] == fr["declared"]["groups"][k]["n"] for k in ("score_q_mlp", "score_dec", "score_head"))
       and all(v["abs_sum"] == 0.0 for v in fr["zero"]["groups"].values()))
    ok(c["live"]["inext"] and all(0.0 <= v <= 1.0 for v in c["live"]["inext"]) and c["live"]["viol"])
    # NEGATIVE CONTROL: the registered 99.9 % coverage threshold refuses this tiny bank (7 of 8 rows have a label)
    ok(c["strict_dvb"]["status"] == "FAIL" and any("coverage" in x for x in c["strict_dvb"]["reasons"]), c["strict_dvb"])
    # the arms: each makes ITS check FAIL, with the reason that names the fault
    wo = g["weight_off"]
    ok(wo["dvb"]["status"] == "FAIL" and any("multiplier measured at the graph" in x for x in wo["dvb"]["reasons"]), wo["dvb"]["reasons"])
    td = g["term_detached"]
    ok(td["dvb"]["status"] == "FAIL" and any("multiplier measured at the graph" in x and "0 of 4" in x for x in td["dvb"]["reasons"]), td["dvb"]["reasons"])
    wz = g["weight_zero"]
    ok(wz["live"]["status"] == "FAIL" and any("EXACTLY zero" in x and "traj_head" in x for x in wz["live"]["reasons"]), wz["live"]["reasons"])
    z = g["w0_but_live_term_declared"]
    ok(z["live"]["status"] == "FAIL" and any("field_L" in x and "ABSENT" in x for x in z["live"]["reasons"]), z["live"]["reasons"])
    # a downstream arm does not leak into another check: the weight-off arm still passes G-LIVE's reach (the fault is in the loss, not in the graph of the term)
    ok(wo["live"]["status"] == "PASS" and td["live"]["status"] == "PASS")


def part_e(E: Env) -> None:
    import torch
    print("(e) THE WEIGHT RULE")
    S = E.S
    sys.path.insert(0, str(HERE))
    import weight_rule as WR
    # the committed rule, as a pure function, at its boundaries
    cand = WR.CANDIDATES
    lin = lambda k: {w: k * w for w in cand}  # noqa: E731
    ok(WR.decide(lin(0.1))["w_field"] == 4.0 and WR.decide(lin(0.1))["status"] == "SELECT")            # r(2) = 0.2 < 0.25 <= r(4) = 0.4
    ok(WR.decide(lin(0.25))["w_field"] == 1.0)                                                            # r(1) = 0.25 EXACTLY: >= selects
    ok(WR.decide(lin(0.2499999999))["w_field"] == 2.0)
    ok(WR.decide(lin(0.25 / 64))["w_field"] == 64.0 and WR.decide(lin(0.25 / 64))["status"] == "SELECT")  # r(64) = 0.25 exactly
    r = lin(0.25 / 64 * 0.999)
    ok(WR.decide(r)["status"] == "K2" and WR.decide(r)["w_field"] is None)                                # r(64) < 0.25: not launched
    ok(WR.decide(lin(0.0))["status"] == "K2")                                                              # a clean fan: the loss is exactly 0
    r = lin(4.0 + 1e-9)
    ok(WR.decide(r)["status"] == "K2b" and WR.decide(r)["w_field"] is None)                                # r(1/4) = 1.00000000025 > 1.0
    ok(WR.decide(lin(4.0))["status"] == "SELECT" and WR.decide(lin(4.0))["w_field"] == 0.25)               # r(1/4) = 1.0 EXACTLY is not > 1.0
    ok(WR.decide({**lin(1.0), 4.0: float("nan")})["status"] == "ERROR" and WR.decide({**lin(1.0), 4.0: -1.0})["status"] == "ERROR")
    try:
        WR.decide({1.0: 1.0})
        ok(False)
    except ValueError:
        ok(True)
    # the candidates, the seed and the thresholds are the prereg's
    ok(cand == (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0) and WR.SEED == 20261012 and WR.N_MICRO == 8 and WR.R_TARGET == 0.25 and WR.R_MAX_LOW == 1.0)
    # the tool on the tiny bank (TEST PIN: its result is stamped NOT A DECISION)
    targv = ["--backbone", "vits16", "--weights", "none", "--n-cameras", "1", "--synthetic", "--cpu", "--batch", "1", "--accum", "2", "--steps", "2", "--targets",
             str(E.W / "data" / "bank"), "--scorer-targets", str(E.W / "data" / "bank_none"), "--field-labels", S["store"], "--w-field", "4", "--resume"]
    outj = E.W / "wr1.json"

    def tool(args, out):
        r_ = subprocess.run([sys.executable, str(HERE / "weight_rule.py"), "--refe", str(E.new / "refe"), "--args", args, "--out", str(out), "--test-pin", S["pin"]],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        return r_
    t1 = tool(" ".join(targv), outj)
    ok(t1.returncode in (0, 5) and outj.exists(), t1.stdout[-800:] + t1.stderr[-1500:])
    print("    " + t1.stdout.strip().replace("\n", "\n    "))
    R = json.loads(outj.read_text(encoding="utf-8"))
    ok(R["NOT_A_DECISION"] is True and R["seed"] == 20261012 and R["n_micro"] == 8 and R["ignored_resume"] is True)
    ok(not (Path(tempfile.gettempdir()) / "weight_rule_unused_run_dir").exists(), "the tool wrote a run directory")
    ids = [tuple(i) for b in R["micro_batch_ids"] for i in b]
    g = torch.Generator().manual_seed(20261012)
    rows_order = [(r["log_name"], r["token"], r["rank"]) for r in [json.loads(l) for n_ in ("targets_rank0.jsonl", "targets_rank1.jsonl")
                                                                    for l in (E.W / "data" / "bank" / n_).read_text(encoding="utf-8").splitlines()]]
    want = [rows_order[i] for i in torch.randperm(len(rows_order), generator=g).tolist()[:8]]
    ok(ids == want and len(set(ids)) == 8, (ids, want))                                                    # the seeded draw, recomputed independently
    rw = R["r_by_w"]
    ok(all(abs(rw[repr(w)] - w * R["pooled"]["r1"]) <= 1e-12 * max(1.0, w * R["pooled"]["r1"]) for w in cand))
    ok(R["rule"] == WR.decide({w: rw[repr(w)] for w in cand}) and (t1.returncode == 0) == (R["rule"]["status"] == "SELECT"))
    ok(R["pooled"]["norm_grad_traj_loss"] > 0.0 and len(R["per_micro_batch"]) == 8 and R["head_params"] > 0)
    outj2 = E.W / "wr2.json"
    tool(" ".join(targv), outj2)
    R2 = json.loads(outj2.read_text(encoding="utf-8"))
    ok(R2["pooled"] == R["pooled"] and R2["micro_batch_ids"] == R["micro_batch_ids"] and R2["rule"] == R["rule"], "the measurement is reproducible")
    # no label store: refused (nothing to measure against)
    t3 = tool(" ".join(t for i, t in enumerate(targv) if t not in ("--field-labels", S["store"])), E.W / "wr3.json")
    ok(t3.returncode != 0 and not (E.W / "wr3.json").exists() and "no --field-labels" in (t3.stdout + t3.stderr))
    # too few rows for 8 micro-batches of --batch 2: refused, not padded
    t4 = tool(" ".join(targv).replace("--batch 1", "--batch 2"), E.W / "wr4.json")
    ok(t4.returncode != 0 and "fewer than" in (t4.stdout + t4.stderr), t4.stdout[-300:] + t4.stderr[-300:])
    # an all-road label: L_field is exactly 0 -> r = 0 at every w -> K2 (exit 5), never a weight
    sys.path.insert(0, str(E.orig / "refe"))
    import aux_label_store as ALS
    road_root = E.W / "road"
    road_root.mkdir()
    (E.W / "data").joinpath("rows0.json").write_text("[]")
    rows0 = [json.loads(l) for l in (E.W / "data" / "bank" / "targets_rank0.jsonl").read_text(encoding="utf-8").splitlines()]
    files, nkeys = {}, 0
    for r_ in rows0:
        if r_["log_name"] == "log5":
            continue
        nb, sha = ALS.write_shard(road_root / "shards" / f"{r_['log_name']}.aux1", r_["log_name"], "tiny",
                                  [{"t": r_["token"], "blob": ALS.encode_sdf(np.full((ALS.NX, ALS.NY), 127, dtype=np.int8)), "s": "bank", "p": [0, 0, 0], "q": None}])
        files[f"shards/{r_['log_name']}.aux1"] = {"sha256": sha, "bytes": nb}
        nkeys += 1
    man = {"schema": ALS.SCHEMA, "raster": ALS.raster_spec(), "counts": {"stored_keys": nkeys}, "leakage": {"token_intersection": 0, "log_intersection": 0}, "files": files}
    man["store_sha256"] = hashlib.sha256("\n".join(f"{k}:{v['sha256']}" for k, v in sorted(files.items())).encode()).hexdigest()
    (road_root / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    outk = E.W / "wr_k2.json"
    tk = subprocess.run([sys.executable, str(HERE / "weight_rule.py"), "--refe", str(E.new / "refe"), "--args",
                         " ".join(targv).replace(S["store"], str(road_root)), "--out", str(outk), "--test-pin", f"{man['store_sha256']}:{nkeys}"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    RK = json.loads(outk.read_text(encoding="utf-8")) if outk.exists() else {}
    ok(tk.returncode == 5 and RK.get("rule", {}).get("status") == "K2" and RK["rule"]["w_field"] is None and all(v == 0.0 for v in RK["r_by_w"].values()), tk.stdout[-600:] + tk.stderr[-600:])
    # the measurement itself: w * grad == grad of (w * L_field), bit for bit, at every candidate (the powers of two are exact)
    lin_ = run_child("wrlin", E.new / "refe", E.W / "runs" / "wrlin.pt", pin=S["pin"], spec=json.dumps(targv[:-1]))
    ok(all(lin_["bitwise_linear"].values()) and len(lin_["bitwise_linear"]) == 9, lin_)


# ======================================================================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", choices=("run", "items", "reach", "gate", "wrlin"))
    ap.add_argument("--refe")
    ap.add_argument("--out")
    ap.add_argument("--pin")
    ap.add_argument("--store")
    ap.add_argument("--spec")
    ap.add_argument("--zero-traj", action="store_true")
    ap.add_argument("--pkg", default=str(HERE.parents[1]))
    ap.add_argument("--part", choices=("train", "gate", "all"), default="all")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--only", action="append", choices=("a", "refusals", "b", "c"), help="run only these trainer parts (debugging; the full proof is the default)")
    ap.add_argument("argv", nargs="*")
    a, rest = ap.parse_known_args()
    if a.child:
        argv = a.argv + [x for x in rest if x != "--"]
        if a.child == "run":
            _child_run(a.refe, a.out, a.pin, a.zero_traj, argv)
        elif a.child == "items":
            _child_items(a.refe, a.out, a.pin, argv)
        elif a.child == "gate":
            _child_gate(a.refe, a.out, a.pin, a.spec)
        elif a.child == "wrlin":
            _child_wrlin(a.refe, a.out, a.pin, a.spec)
        else:
            _child_reach(a.refe, a.out, a.pin, a.store)
        return 0
    pkg = Path(a.pkg).resolve()
    W = Path(tempfile.mkdtemp(prefix="fieldwire_"))
    print(f"workspace {W}")
    try:
        orig, new = W / "orig", W / "new"
        copy_pkg(pkg, orig)
        copy_pkg(pkg, new)
        for d in ("PATCH_train.diff", "PATCH_gate.diff"):
            if (HERE / d).exists():
                print(f"  patch -p1 < {d}: " + apply_patch(new, HERE / d).strip().replace("\n", " | "))
        (W / "runs").mkdir()
        (W / "data").mkdir()
        rows0 = make_bank(W / "data")
        S = make_store(W / "data", rows0, str(orig / "refe"))
        E = Env(W, orig, new, S)
        if a.part in ("train", "all"):
            for nm, fn in (("a", part_a), ("refusals", part_refusals), ("b", part_b), ("c", part_c)):
                if not a.only or nm in a.only:
                    fn(E)
        if a.part in ("gate", "all"):
            if a.part == "gate":
                (W / "runs").mkdir(exist_ok=True)
            part_d(E)
            part_e(E)
        print(f"ZZTEST_PASS {N}")
        return 0
    finally:
        if a.keep:
            print(f"kept {W}")
        else:
            shutil.rmtree(W, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
