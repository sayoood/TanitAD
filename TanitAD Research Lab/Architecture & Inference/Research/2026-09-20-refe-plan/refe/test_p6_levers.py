#!/usr/bin/env python3
"""P6 reachability proofs (SPEC_P6_PAPER_FAITHFUL.md section 2; CLAUDE.md "built, tested and unreachable from its caller").

Every new lever is traced flag -> model attribute -> forward -> loss term -> backward, with (a) a NON-ZERO gradient on the parameters it reaches, (b) a
control that must read EXACTLY 0.0, (c) a MUTATION: the emitting line is deleted in a copy of the source and the check must go RED.

    python test_p6_levers.py --fast        # data-free + tiny-trainer checks (self-made synthetic bank, ViT-S, 1 camera, CPU); no mutations. G-SUITE runs this.
    python test_p6_levers.py               # --fast, then every mutation (copies refe/*.py to a temp dir, edits one line, runs ONE check there: must exit != 0)
    python test_p6_levers.py --check NAME  # one check (the mutation harness's entry point)
Prints `ZZP6_PASS <n>`.  A skip is not a pass: nothing here skips.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402

N = 0


def ok(cond, msg=""):
    global N
    assert cond, msg
    N += 1


# ------------------------------------------------------------------------------------------------ a self-made tiny bank (no dev-box data needed)
def make_bank(root: Path, n_scenes: int = 4) -> str:
    """6 tuples in 4 scenes (2 with a rank-1 twin), 4-D goals, a lane-rank `aug` on the twins. Synthetic images (`--synthetic`)."""
    rng = np.random.default_rng(5)
    d = root / "bank"
    d.mkdir(parents=True, exist_ok=True)
    rows0, rows1 = [], []
    for s in range(n_scenes):
        gy = [0.0, 9.0, -9.0, 1.0][s % 4]                                   # straight, left, right, straight
        base = {"image": [f"log{s}/CAM_F0/a{s}.jpg"], "cameras": ["CAM_F0"], "ego": [float(x) for x in rng.normal(0, 0.3, 7)],
                "goal": [30.0, gy, 60.0, 2 * gy], "traj": [[float(0.7 * (t + 1)), float(gy * (t + 1) / 80.0), float(0.02 * s)] for t in range(20)],
                "step": 0, "token": f"tok{s:04x}", "log_name": f"log{s}", "scenario_type": "t", "source": "fixture"}
        rows0.append({**base, "rank": 0})
        if s < 2:
            rows1.append({**base, "rank": 1, "goal": [30.0, -gy, 60.0, -2 * gy], "aug": {"kind": "lane_rank", "value": 1, "divergence_m": 2.0}})
    for nm, rows in (("targets_rank0.jsonl", rows0), ("targets_rank1.jsonl", rows1)):
        (d / nm).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    return str(d)


def argv_p6(bank: str, out: str, *, steps: int = 2, accum: int = 2, extra=(), yaw="off", sees_trunk=True) -> list:
    a = ["--backbone", "vits16", "--weights", "none", "--n-cameras", "1", "--synthetic", "--cpu", "--targets", bank, "--scorer-targets", bank + "_none",
         "--yaw-wrap", yaw, "--nav-mode", "command", "--labeller", "online", "--label-backend", "synthetic", "--label-workers", "2",
         "--batch", "1", "--accum", str(accum), "--steps", str(steps), "--lr", "2e-4", "--log-every", "1", "--ckpt-every-min", "999", "--out", out]
    if sees_trunk:
        a.append("--scorer-sees-trunk")
    return a + list(extra)


def argv_default(bank: str, out: str, extra=()) -> list:
    return ["--backbone", "vits16", "--weights", "none", "--n-cameras", "1", "--synthetic", "--cpu", "--targets", bank, "--scorer-targets", bank + "_none", "--batch", "1",
            "--accum", "2", "--steps", "3", "--lr", "5e-5", "--log-every", "1", "--ckpt-every-min", "999", "--out", out] + list(extra)


def run_sub(argv, env=None) -> subprocess.CompletedProcess:
    e = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONHASHSEED="0")
    e.pop("TANITAD_GATE_ARM", None)
    e.update(env or {})
    return subprocess.run([sys.executable, str(HERE / "train.py")] + argv, cwd=str(HERE), env=e, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")


def rows_of(out: str) -> list:
    return [json.loads(l) for l in (Path(out) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]


def digest_model(path: str) -> str:
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().contiguous().numpy().tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ unit checks (no trainer)
def wta_old(traj, target, yaw_w=0.1):                    # VERBATIM the pre-P6 model.wta_loss body
    pos = (traj[..., :2] - target[..., :2].unsqueeze(1)).abs().sum(-1).mean(-1)
    idx = pos.argmin(dim=1)
    ar = torch.arange(pos.shape[0], device=pos.device)
    l_pos = pos[ar, idx].mean()
    dy = traj[ar, idx, :, 2] - target[..., 2]
    dy = (dy + math.pi) % (2 * math.pi) - math.pi
    return l_pos + yaw_w * dy.abs().mean(), idx


def chk_wta_default_identity():
    from model import wta_loss
    g = torch.Generator().manual_seed(1)
    for _ in range(5):
        traj = torch.randn(3, 8, 20, 3, generator=g) * 4
        tgt = torch.randn(3, 20, 3, generator=g) * 4
        a, ia = wta_loss(traj, tgt)
        b, ib = wta_old(traj, tgt)
        ok(torch.equal(a, b) and torch.equal(ia, ib))
        c, ic, lp, ly = wta_loss(traj, tgt, yaw_wrap=True, parts=True)
        ok(torch.equal(c, b) and torch.equal(ic, ib) and torch.equal(c, lp + 0.1 * ly))


def chk_yaw_analytic():
    """D5: error 2*pi - eps => unwrapped yaw L1 = 2*pi - eps (loss (2*pi-eps)*0.1); wrapped = eps. yaw_w = 0 => the yaw-channel gradient is EXACTLY 0.0."""
    from model import wta_loss
    eps = 0.25
    traj = torch.zeros(1, 2, 4, 3, requires_grad=True)
    tgt = torch.zeros(1, 4, 3)
    with torch.no_grad():
        traj[:, :, :, 2] = 2 * math.pi - eps
    _, _, lp_off, ly_off = wta_loss(traj, tgt, yaw_wrap=False, parts=True)
    _, _, lp_on, ly_on = wta_loss(traj, tgt, yaw_wrap=True, parts=True)
    ok(abs(float(ly_off) - (2 * math.pi - eps)) < 1e-5, float(ly_off))
    ok(abs(float(ly_on) - eps) < 1e-5, float(ly_on))                           # the wrapped value: the mutation arm of 'off' reads THIS, and must FAIL the line above
    tot, *_ = wta_loss(traj, tgt, yaw_w=0.1, yaw_wrap=False, parts=True)
    ok(abs(float(tot) - 0.1 * (2 * math.pi - eps)) < 1e-5)
    traj.grad = None
    wta_loss(traj, tgt, yaw_w=0.0, yaw_wrap=False)[0].backward()
    ok(float(traj.grad[..., 2].abs().sum()) == 0.0, "yaw_w=0 must leave the yaw channel's gradient EXACTLY 0.0")
    traj.grad = None
    wta_loss(traj, tgt, yaw_w=0.1, yaw_wrap=False)[0].backward()
    ok(float(traj.grad[..., 2].abs().sum()) > 0.0, "yaw_w=0.1 must reach the yaw channel")


def chk_nav_rule():
    import nav_command as NC
    ok(NC.goal_to_command([30.0, 0.5, 60.0, 1.0]) == NC.STRAIGHT and NC.goal_to_command([30.0, 9.0, 60, 18]) == NC.LEFT
       and NC.goal_to_command([30.0, -9.0, 60, -18]) == NC.RIGHT and NC.goal_to_command([float("nan"), 0.0]) == NC.UNKNOWN)
    ok(NC.command_of_row({"goal": [30.0, 9.0], "command": 2}) == NC.RIGHT, "a row's own (NAVSIM) command must win over the goal rule")
    ok(NC.command_of_row({"goal": [30.0, 9.0]}) == NC.LEFT and NC.command_source({"goal": [1, 1]}) == "goal_rule" and NC.command_source({"command": 1}) == "row")
    ok(NC.from_driving_command([0, 1, 0, 0]) == NC.STRAIGHT and NC.from_driving_command([0, 0, 0, 0]) == NC.UNKNOWN and NC.onehot(2) == [0.0, 0.0, 1.0, 0.0])
    try:
        NC.command_of_row({"goal": [1, 1], "command": 9})
        ok(False)
    except ValueError:
        ok(True)


def _tiny_model(nav_mode):
    from model import REFe, REFeConfig
    cfg = REFeConfig.for_backbone("vits16", nav_mode=nav_mode)
    cfg.n_cameras, cfg.cameras = 1, cfg.cameras[:1]
    cfg.detach_scorer_context = False
    torch.manual_seed(0)
    return REFe(cfg)


def chk_command_model_reach():
    """D6: command mode builds `cmd_embed` and NO goal encoder; the command reaches the output; non-zero gradient on cmd_embed; a zero loss reads exactly 0.0."""
    m = _tiny_model("command")
    ok(hasattr(m, "cmd_embed") and not hasattr(m, "goal_freqs"), "command mode must build cmd_embed and NOT the goal encoder")
    mg = _tiny_model("goal")
    ok(hasattr(mg, "goal_freqs") and not hasattr(mg, "cmd_embed"))
    img = torch.randn(1, 1, 3, 512, 960) * 0.1
    ego = torch.randn(1, 7) * 0.1
    left, right = torch.tensor([[1.0, 0, 0, 0]]), torch.tensor([[0, 0, 1.0, 0]])
    t_l, s_l = m(img, ego, left)
    t_r, s_r = m(img, ego, right)
    ok(float((t_l - t_r).abs().max()) > 0.0, "the command must change the proposals")
    for tag, scale in (("zero", 0.0), ("real", 1.0)):
        m.zero_grad(set_to_none=True)
        t, s = m(img, ego, left)
        ((t.sum() + s.sum()) * scale).backward()
        g = {n: float(p.grad.double().abs().sum()) for n, p in m.named_parameters() if p.grad is not None and n.startswith(("cmd_embed", "ego_enc"))}
        if tag == "zero":
            ok(g and all(v == 0.0 for v in g.values()), g)                         # the probe's own control: exactly 0.0
        else:
            ok(g.get("cmd_embed.weight", 0.0) > 0.0, g)
    try:
        m(img, ego, torch.zeros(1, 4 + 1))
        ok(False)
    except ValueError:
        ok(True)


def chk_ckpt_meta_nav_mode():
    import ckpt_io
    from model import REFe, REFeConfig
    td = Path(tempfile.mkdtemp())
    m = _tiny_model("command")
    p = td / "m.pt"
    ckpt_io.atomic_save({"format": ckpt_io.FORMAT_FULL, "model": ckpt_io.full_state(m), "meta": {"backbone": "vits16", "nav_mode": "command"}}, p)
    base = REFeConfig.for_backbone("vits16")
    base.n_cameras, base.cameras = 1, base.cameras[:1]
    pc = ckpt_io.config_for_checkpoint(base, str(p))
    ok(pc.nav_mode == "command" and base.nav_mode == "goal")
    pm = REFe(pc)
    pm.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["model"], strict=True)
    mg = _tiny_model("goal")
    p2 = td / "g.pt"
    ckpt_io.atomic_save({"format": ckpt_io.FORMAT_FULL, "model": ckpt_io.full_state(mg), "meta": {"backbone": "vits16"}}, p2)
    ok(ckpt_io.config_for_checkpoint(base, str(p2)).nav_mode == "goal")
    shutil.rmtree(td, ignore_errors=True)


def chk_identity_resume():
    import train as T
    new = {"args": {k: T.IDENTITY_DEFAULTS.get(k, 1) for k in T.IDENTITY_ARGS}}
    old = {"args": {k: v for k, v in new["args"].items() if k not in T.IDENTITY_DEFAULTS}}      # a checkpoint written before P6 lacks the new keys
    old.update({k: None for k in ("n_tuples", "n_scenes", "per_rank", "targets_sha256", "scorer_sha256", "calib_sha256", "frozen_sha256")})
    new.update({k: None for k in ("n_tuples", "n_scenes", "per_rank", "targets_sha256", "scorer_sha256", "calib_sha256", "frozen_sha256")})
    ok(T._identity_diff(old, new) == [], T._identity_diff(old, new))
    new2 = copy.deepcopy(new)
    new2["args"]["nav_mode"] = "command"
    ok(any("--nav-mode" in d for d in T._identity_diff(old, new2)), "a resume across nav_mode must be refused")


def chk_profile_p6():
    import launch_gate as L
    good = ("--backbone vitl16 --targets /w/t --images /w/p --calib /w/c.json --metric-cache /w/mc --scorer-sees-trunk --yaw-wrap off --nav-mode command "
            "--labeller online --epochs 25 --batch 4 --accum 64 --lr 2e-4 --weight-decay 0.01 --amp bf16 --score-w 0.1 --out /w/run")
    prof = L.PROFILES["refe_p6"]
    ok(L.profile_refusals(prof, L.parse_args_string(good)) == [], L.profile_refusals(prof, L.parse_args_string(good)))
    rs_on = L.profile_refusals(prof, L.parse_args_string(good.replace("--yaw-wrap off", "--yaw-wrap on")))
    ok(rs_on and any("requires --yaw-wrap off" in r for r in rs_on), ("PI ruling P6-Y (2026-10-11): unwrapped yaw; --yaw-wrap on is a value mismatch", rs_on))
    for bad, frag in ((good.replace("--yaw-wrap off ", ""), "--yaw-wrap"), (good.replace("--labeller online", "--labeller bank"), "--labeller"),
                      (good.replace("--nav-mode command", "--nav-mode goal"), "--nav-mode"), (good + " --grow", "FORBIDS"),
                      (good + " --label-backend synthetic", "FORBIDS"), (good.replace("--epochs 25", "--epochs 24"), "--epochs"),
                      (good.replace("--scorer-sees-trunk ", ""), "--scorer-sees-trunk")):
        rs = L.profile_refusals(prof, L.parse_args_string(bad))
        ok(rs and any(frag in r for r in rs), (frag, rs))
    ok(prof["expect_schedule"]["steps"] == 25 * math.ceil(103037 / 256), "the registered step count must be the trainer's own definition")


# ------------------------------------------------------------------------------------------------ trainer-level checks
def chk_trainer_reach_online():
    """flag -> built object -> loss -> backward, observed IN the real train.run: yaw_wrap reaches the loss, the labeller is built, every P6 term is logged and finite,
    every trainable group (incl. cmd_embed) has a non-zero gradient, lag is 0, the step-0 bank event is written, and the tuple the loader serves carries the command."""
    import gate_checks as C
    td = Path(tempfile.mkdtemp())
    bank = make_bank(td)
    for yaw, want in (("off", False), ("on", True)):
        cap = C.instrumented_run(argv_p6(bank, str(td / f"o_{yaw}"), yaw=yaw), mode="smoke", work=td / f"w_{yaw}")
        ok(cap.rc == 0, (cap.rc, cap.error))
        ok(cap.wta_yaw_wrap is want, (yaw, cap.wta_yaw_wrap))
    cap = C.instrumented_run(argv_p6(bank, str(td / "o_main")), mode="smoke", work=td / "w_main")
    ok(cap.rc == 0 and cap.banks.get("OnlineLabeller") and cap.banks["OnlineLabeller"][0]["kwargs"]["backend"] == "synthetic")
    ok(cap.cfg.nav_mode == "command" and hasattr(cap.model, "cmd_embed") and not hasattr(cap.model, "goal_freqs"))
    steps = [r for r in cap.metrics if "traj_L1" in r]
    ok(len(steps) == 2)
    terms = ("traj_L1", "score", "pos_L1", "yaw_L1", "bce_nc", "bce_dac", "bce_ep", "bce_ttc", "bce_c", "bce_ddc")
    for r in steps:
        ok(all(t in r and math.isfinite(r[t]) for t in terms), r)
        mean_bce = sum(r[f"bce_{c}"] for c in ("nc", "dac", "ep", "ttc", "c", "ddc")) / 6.0
        ok(abs(r["score"] - 0.1 * mean_bce) < 1e-6 * max(1.0, abs(r["score"])) + 1e-7, (r["score"], 0.1 * mean_bce))      # score == score_w * mean(bce_*)
        ok(abs(r["traj_L1"] - (r["pos_L1"] + 0.1 * r["yaw_L1"])) < 1e-6)
    grads = {g: max(rec[g]["abs_sum"] for rec in cap.grads) for g in cap.grads[0]}
    for g in ("backbone.lora", "registers", "queries", "pos3d_mlp", "reg_compress", "scene_proj", "ego_enc", "cmd_embed", "dec", "traj_head", "score_dec",
              "score_q_mlp", "score_head"):
        ok(grads.get(g, 0.0) > 0.0, (g, grads))
    ok("goal_freqs" not in grads or grads["goal_freqs"] == 0.0)
    ev = [r for r in cap.metrics if r.get("event") == "labeller"]
    ok(ev and all(set(e["lag_hist"]) == {"0"} for e in ev) and all(e["failed"] == 0 and e["no_parts"] == 0 for e in ev), ev[:1])
    bk = [r for r in cap.metrics if r.get("event") == "bank"]
    ok(bk and bk[0]["n_tuples"] == 6 and bk[0]["n_scenes"] == 4 and bk[0]["labeller"] == "online" and bk[0]["command_sources"] == {"goal_rule": 6}, bk[:1])
    shutil.rmtree(td, ignore_errors=True)


def _score_only_grads(td, bank, tag, *, sees_trunk=True, env=None, patch_bce=False):
    """Run the trainer with the TRAJECTORY term's gradient removed (wta_loss on detached proposals), so every gradient observed is the SCORE term's."""
    import gate_checks as C
    import train as T
    import torch.nn.functional as F
    real = T.wta_loss

    def score_only(traj, target, *a, **k):
        l, idx, lp, ly = real(traj.detach(), target, *a, **{**k, "parts": True})
        return (l, idx, lp, ly) if k.get("parts") else (l, idx)
    old_env = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    orig_bce = F.binary_cross_entropy_with_logits
    T.wta_loss = score_only
    if patch_bce:
        F.binary_cross_entropy_with_logits = lambda x, *a, **k: x * 0.0
    try:
        cap = C.instrumented_run(argv_p6(bank, str(td / f"o_{tag}"), sees_trunk=sees_trunk), mode="smoke", work=td / f"w_{tag}")
    finally:
        T.wta_loss = real
        F.binary_cross_entropy_with_logits = orig_bce
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    ok(cap.rc == 0, (tag, cap.rc, cap.error))
    return {g: max(rec[g]["abs_sum"] for rec in cap.grads) for g in cap.grads[0]}


def chk_score_only_grad():
    """D1 (b) / D2 / D3 from the TRAINER: with the trajectory term's gradient removed, the score loss alone gives a non-zero gradient on the scorer heads (and cmd_embed's
    consumers via the context) and on the backbone LoRA when the context is un-detached; `traj_head` reads EXACTLY 0.0 (candidates detached); the LoRA reads
    EXACTLY 0.0 when the context is detached; with the labeller OFF (no labels) the scorer reads EXACTLY 0.0."""
    td = Path(tempfile.mkdtemp())
    bank = make_bank(td)
    on = _score_only_grads(td, bank, "on")
    for g in ("score_dec", "score_q_mlp", "score_head", "backbone.lora"):
        ok(on[g] > 0.0, (g, on))
    ok(on["traj_head"] == 0.0 and on["dec"] == 0.0, on)                                    # D2: the candidates are detached
    det = _score_only_grads(td, bank, "detached", sees_trunk=False)
    ok(det["backbone.lora"] == 0.0 and det["score_head"] > 0.0, det)                       # D3: the control
    off = _score_only_grads(td, bank, "off", env={"REFE_LAB_MUTATE": "no_labels"})
    ok(all(off[g] == 0.0 for g in ("score_dec", "score_q_mlp", "score_head", "backbone.lora", "traj_head")), off)       # labeller off: EXACTLY 0.0
    zero_bce = _score_only_grads(td, bank, "zbce", patch_bce=True)
    ok(zero_bce["score_head"] == 0.0, "the BCE term dropped must read exactly 0.0 (the gate's zero_score arm)")
    shutil.rmtree(td, ignore_errors=True)


def chk_deferred_bit_identity():
    """D4: the gradient of the DEFERRED backward over one accumulation window is bit-identical to the undeferred one given the same labels (same accumulation
    order). Replays the first window from the recorded inputs, weights and labels. Also: the replay with the labels shifted by one micro-batch must DIFFER."""
    import torch.nn.functional as F
    import online_labeller as OLm
    import train as T
    from model import REFe, wta_loss
    td = Path(tempfile.mkdtemp())
    bank = make_bank(td)
    rec = {"inputs": [], "tgts": [], "labels": [], "props": []}
    first = {}
    OLsub, OLcol = OLm.OnlineLabeller.submit, OLm.OnlineLabeller.collect
    wta0 = T.wta_loss
    Orig = T.REFe
    import torch.optim.optimizer as opt_mod

    class Rec(Orig):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.register_forward_pre_hook(lambda m, a, kw: rec["inputs"].append((a[0].detach().clone(), a[1].detach().clone(), a[2].detach().clone(),
                                                                                 kw.get("calib"))), with_kwargs=True)
            first["model"] = self

    def sub(self, stamp, step, metas, props):
        rec["props"].append(props.copy())
        return OLsub(self, stamp, step, metas, props)

    def col(self, h, step_now):
        r = OLcol(self, h, step_now)
        rec["labels"].append((r[0].copy(), r[1].copy()))
        return r

    def wta(traj, tgt, *a, **k):
        rec["tgts"].append(tgt.detach().clone())
        return wta0(traj, tgt, *a, **k)

    def pre(optimizer, _a, _k):
        if "grads" not in first and first.get("model") is not None:
            m = first["model"]
            first["grads"] = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
            first["w0"] = {n: p.detach().clone() for n, p in m.named_parameters()}
    h = opt_mod.register_optimizer_step_pre_hook(pre)
    T.REFe, T.wta_loss = Rec, wta
    OLm.OnlineLabeller.submit, OLm.OnlineLabeller.collect = sub, col
    try:
        ap = T.build_parser()
        a = ap.parse_args(argv_p6(bank, str(td / "o"), steps=1, accum=2))
        rc = T.run(a)
    finally:
        h.remove()
        T.REFe, T.wta_loss = Orig, wta0
        OLm.OnlineLabeller.submit, OLm.OnlineLabeller.collect = OLsub, OLcol
    ok(rc == 0 and "grads" in first and len(rec["labels"]) == 2)
    # the labels the trainer consumed == the offline function of the proposals it labelled (D1 (a), identity)
    for props, (lab, msk) in zip(rec["props"], rec["labels"]):
        ok(all(np.array_equal(lab[i], OLm.synthetic_labels(props[i], salt=0)) for i in range(props.shape[0])))
    m = first["model"]
    ref = Orig(m.cfg)
    with torch.no_grad():
        for n, p in ref.named_parameters():
            p.copy_(first["w0"][n])

    own = []

    def replay(label_order):
        ref.zero_grad(set_to_none=True)
        own.clear()
        for k in range(2):
            img, ego, goal, cal = rec["inputs"][k]
            traj, score = ref(img, ego, goal, calib=cal)
            own.append(traj.detach().double().numpy())
            l_traj, _ = wta_loss(traj.float(), rec["tgts"][k], yaw_wrap=False)
            lab, msk = rec["labels"][label_order[k]]
            bce = F.binary_cross_entropy_with_logits(score.float(), torch.from_numpy(lab), reduction="none")
            l_score = (bce.mean(-1) * torch.from_numpy(msk)[:, None]).sum() / max(64.0 * img.shape[0], 1.0) * 0.1
            ((l_traj + l_score) / 2).backward()
        torch.nn.utils.clip_grad_norm_([p for p in ref.parameters() if p.requires_grad], 1.0)          # the trainer clips before opt.step()
        return {n: p.grad.detach().clone() for n, p in ref.named_parameters() if p.grad is not None}
    g_ref = replay([0, 1])
    # D1 (a) from the MODEL's own proposals: the labels the trainer consumed are the offline labels of the proposals this model actually produced
    for k in range(2):
        ok(all(np.array_equal(rec["labels"][k][0][i], OLm.synthetic_labels(own[k][i], salt=0)) for i in range(own[k].shape[0])),
           "the consumed labels are not the labels of the model's own proposals")
    same = set(g_ref) == set(first["grads"]) and all(torch.equal(g_ref[n], first["grads"][n]) for n in g_ref)
    ok(same, "deferred gradient != undeferred gradient")
    g_shift = replay([1, 0])
    ok(not all(torch.equal(g_shift[n], first["grads"][n]) for n in g_shift if n in first["grads"]), "shifted labels must change the gradient (the check can go RED)")
    shutil.rmtree(td, ignore_errors=True)


def chk_stale_label_refused():
    """D1 (c): feeding one-step-stale proposals to the labeller must be REFUSED (proposal identity), not silently trained on."""
    td = Path(tempfile.mkdtemp())
    bank = make_bank(td)
    good = run_sub(argv_p6(bank, str(td / "g"), steps=2, accum=2))
    ok(good.returncode == 0, good.stdout[-600:] + good.stderr[-600:])
    bad = run_sub(argv_p6(bank, str(td / "b"), steps=2, accum=2), env={"REFE_LAB_MUTATE": "stale"})
    ok(bad.returncode != 0 and "IDENTITY VIOLATED" in (bad.stdout + bad.stderr), (bad.returncode, (bad.stdout + bad.stderr)[-400:]))
    shutil.rmtree(td, ignore_errors=True)


def chk_online_conflicts():
    import train as T
    td = Path(tempfile.mkdtemp())
    bank = make_bank(td)
    for extra in (["--grow"], ["--pdm-only"], ["--scorer-mode", "onpolicy"]):
        a = T.build_parser().parse_args(argv_p6(bank, str(td / "x"), extra=extra))
        ok(T.run(a) == 4)
    a = T.build_parser().parse_args([t for t in argv_p6(bank, str(td / "y")) if t != "synthetic"][:0] + ["--labeller", "online", "--cpu", "--synthetic", "--targets", bank])
    ok(T.run(a) == 4, "the pdm backend without --metric-cache must be refused")
    shutil.rmtree(td, ignore_errors=True)


def chk_default_flags_inert():
    """The new flags at their defaults are inert: a default-argv run == a run that names every new flag at its default value (losses AND the final model, bit for bit)."""
    td = Path(tempfile.mkdtemp())
    bank = make_bank(td)
    a = run_sub(argv_default(bank, str(td / "a")))
    b = run_sub(argv_default(bank, str(td / "b"), extra=["--yaw-wrap", "on", "--nav-mode", "goal", "--labeller", "bank"]))
    ok(a.returncode == 0 and b.returncode == 0, (a.stdout[-300:], b.stdout[-300:]))
    la = [(r["traj_L1"], r["score"]) for r in rows_of(str(td / "a")) if "traj_L1" in r]
    lb = [(r["traj_L1"], r["score"]) for r in rows_of(str(td / "b")) if "traj_L1" in r]
    ok(la == lb and len(la) == 3, (la, lb))
    ok(digest_model(str(td / "a" / "model_final.pt")) == digest_model(str(td / "b" / "model_final.pt")))
    ok(not any(k.startswith("bce_") for r in rows_of(str(td / "a")) for k in r), "the default path must not grow the online-labeller columns")
    shutil.rmtree(td, ignore_errors=True)


def chk_gate_judges():
    """G-LIVE-LAB / G-LIVE-MEM / G-EGR judges (pod-run checks): fixture-driven, each with its regression arm."""
    import gate_checks as C
    lab = {"identity_ok": True, "n_sets": 64, "identity_equal_sets": 64, "stale_refused": True, "throughput_sets_s": 9.6, "workers": 12}
    ev = [{"event": "labeller", "sets": 8, "labelled": 8, "lag_hist": {"0": 8}, "stall_frac": 0.30},
          {"event": "labeller", "sets": 8, "labelled": 8, "lag_hist": {"0": 8}, "stall_frac": 0.01}]
    ok(C.judge_lab(lab, ev, 12) == [], C.judge_lab(lab, ev, 12))
    for bad, frag in (({**lab, "identity_equal_sets": 63, "identity_ok": False}, "identity"), ({**lab, "stale_refused": False}, "stale"),
                      ({**lab, "throughput_sets_s": 4.3}, "throughput"), ({**lab, "workers": 2}, "launch argv"), ({**lab, "n_sets": 32}, "identity")):
        r = C.judge_lab(bad, ev, 12)
        ok(r and any(frag in x for x in r), (frag, r))
    ok(any("stall" in x for x in C.judge_lab(lab, [ev[0], {**ev[1], "stall_frac": 0.2}], 12)))
    ok(any("lag" in x for x in C.judge_lab(lab, [{**ev[0], "lag_hist": {"0": 4, "3": 4}}, ev[1]], 12)))
    ok(any("no `labeller` event" in x for x in C.judge_lab(lab, [], 12)))
    ok(C.judge_lab(lab, ev, 12, arm="lab_two_workers"), "a 2-worker pool must FAIL the throughput bar")
    rows = [{"cuda_max_mem_gb": 31.0, "traj_L1": 1.0, "samples": 800, "batch": 4}]
    ok(C.judge_mem(rows) == [] and C.judge_mem(rows, arm="mem_over_budget"))
    ok(C.judge_mem([{**rows[0], "cuda_max_mem_gb": 47.0}]) and C.judge_mem([{**rows[0], "cuda_max_mem_gb": None}]) and C.judge_mem([{**rows[0], "samples": 40}]))
    egr = {"pass": True, "main": {"ok": True}, "regression": {"refused_by_server": True}}
    ok(C.judge_egr(egr, 228234301, 228234301) == [] and C.judge_egr(egr, 228234301, 228234301, arm="egr_wrong_size"))
    ok(C.judge_egr({**egr, "pass": False}) and C.judge_egr({**egr, "regression": {"refused_by_server": False}}) and C.judge_egr(egr, 10, 11))


def chk_gate_p6_rows():
    """The P6 levers' G-DVB rows, G-CLOCK bank check and G-LIVE terms/groups, read off the REAL built objects (BUILD + SMOKE captures), with regression arms."""
    import gate_checks as C
    import launch_gate as L
    td = Path(tempfile.mkdtemp())
    bank = make_bank(td)
    base = argv_p6(bank, str(td / "o"), steps=2, accum=1)
    b_argv = C._set(C._drop(base, "--steps", 1), "--epochs", "1")
    prof = {**L.PROFILES["refe_p6"], "forbidden_flags": (), "required_flags": {}, "provisional_flags": (),
            "expect_schedule": {"epochs": 1, "steps": 4, "epoch_len": 4, "n_tuples": 6}}
    b = C.instrumented_run(b_argv, mode="build", work=td / "wb")
    s = C.instrumented_run(base, mode="smoke", work=td / "ws")
    ok(b.rc == "STOPPED_AT_BUILD" and s.rc == 0, (b.rc, b.error, s.rc, s.error))
    L._free(b)                                  # run_checks frees the BUILD model before the checks run: the rows must not depend on it (a real bug found by the P2 full test)
    ev = C.check_dvb(b_argv, b, s, prof)
    ok(ev["status"] == "PASS", ev["reasons"])
    dests = {r["dest"] for r in ev["details"]["rows"]}
    ok({"labeller", "label_backend", "label_workers", "label_timeout_s", "label_python", "metric_cache", "nav_mode", "yaw_wrap"} <= dests, sorted(dests))
    ok(all(r["ok"] for r in ev["details"]["rows"]))
    b2 = C.instrumented_run(b_argv, mode="build", work=td / "wb2", arm="lever_ignored")                # the model quietly ignores --scorer-sees-trunk
    ev2 = C.check_dvb(b_argv, b2, s, prof, arm="lever_ignored")
    ok(ev2["status"] == "FAIL" and "declared False != built True" in " ".join(ev2["reasons"]), ev2["reasons"])
    ok(C.check_clock(b_argv, b, prof)["status"] == "PASS", C.check_clock(b_argv, b, prof)["reasons"])
    ok(C.check_clock(b_argv, b, {**prof, "expect_schedule": {**prof["expect_schedule"], "n_tuples": 189888}})["status"] == "FAIL")   # a bank of the wrong size
    ok(C.check_clock(b_argv, b, prof, arm="clock_wrong_epochs")["status"] == "FAIL")
    live = C.check_live(base, s, prof)
    ok(live["status"] == "PASS", live["reasons"])
    ok("cmd_embed" in live["details"]["groups"] and live["details"]["groups"]["cmd_embed"]["max_abs_sum"] > 0.0)
    base0 = argv_p6(bank, str(td / "o_zero"), steps=2, accum=1)              # a FRESH --out (a finished run directory is refused by the trainer)
    s0 = C.instrumented_run(base0, mode="smoke", work=td / "ws0", arm="zero_score")
    lv0 = C.check_live(base0, s0, prof, arm="zero_score")
    ok(lv0["status"] == "FAIL" and "EXACTLY zero" in " ".join(lv0["reasons"]), lv0["reasons"])
    shutil.rmtree(td, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ the mutation harness
CHECKS = {"wta_default_identity": chk_wta_default_identity, "yaw_analytic": chk_yaw_analytic, "nav_rule": chk_nav_rule,
          "command_model_reach": chk_command_model_reach, "ckpt_meta_nav_mode": chk_ckpt_meta_nav_mode, "identity_resume": chk_identity_resume,
          "profile_p6": chk_profile_p6, "online_conflicts": chk_online_conflicts, "default_flags_inert": chk_default_flags_inert,
          "trainer_reach_online": chk_trainer_reach_online, "score_only_grad": chk_score_only_grad, "deferred_bit_identity": chk_deferred_bit_identity,
          "stale_label_refused": chk_stale_label_refused, "gate_judges": chk_gate_judges, "gate_p6_rows": chk_gate_p6_rows}

# (file, old, new, the check that must go RED).  Each deletes/neutralises ONE emitting line.
MUTATIONS = [
    ("model.py", "        if yaw_wrap:                                         # P6 / D5", "        if True:                                             # P6 / D5", "yaw_analytic"),
    ("model.py", "ego_tok = self.ego_enc(torch.cat([ego, self.cmd_embed(goal.to(ego.dtype))], -1)).unsqueeze(1)",
     "ego_tok = self.ego_enc(torch.cat([ego, 0.0 * self.cmd_embed(goal.to(ego.dtype)).detach()], -1)).unsqueeze(1)", "command_model_reach"),
    ("train.py", 'yaw_wrap=(a.yaw_wrap == "on"), parts=True)', "yaw_wrap=True, parts=True)", "trainer_reach_online"),
    ("train.py", '"bce_ttc": bce_comp[3], ', '', "trainer_reach_online"),
    ("train.py", "tot_ = P[\"l_traj\"] + l_score\n", "tot_ = P[\"l_traj\"] + 0.0 * l_score\n", "score_only_grad"),   # anchor moved by 134be4f (field-loss wiring split the backward line); R40
    ("train.py", 'cfg.nav_mode = a.nav_mode', 'pass', "trainer_reach_online"),
    ("train.py", "lab.submit(micro, step, metas, props_np)", "lab.submit(micro, step, metas, props_np + 0.01)", "deferred_bit_identity"),
    ("online_labeller.py", 'if r["sha"] != h["shas"][i]:', 'if False:', "stale_label_refused"),
    ("ckpt_io.py", '    if nav and nav != c.nav_mode:\n        c.nav_mode = nav\n', '', "ckpt_meta_nav_mode"),
    ("nav_command.py", "    if ang > GOAL_RULE_BEARING_RAD:\n        return LEFT", "    if ang > GOAL_RULE_BEARING_RAD:\n        return RIGHT", "nav_rule"),
    ("launch_gate.py", '"--yaw-wrap": "off", ', '', "profile_p6"),
]


def run_mutations() -> None:
    for fn, old, new, check in MUTATIONS:
        td = Path(tempfile.mkdtemp(prefix="p6mut_"))
        for f in HERE.glob("*.py"):
            shutil.copyfile(f, td / f.name)
        src = (td / fn).read_text(encoding="utf-8")
        ok(src.count(old) == 1, f"mutation target not found exactly once in {fn}: {old[:60]!r}")
        (td / fn).write_text(src.replace(old, new), encoding="utf-8", newline="\n")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        r = subprocess.run([sys.executable, str(td / "test_p6_levers.py"), "--check", check], cwd=str(td), env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        ok(r.returncode != 0, f"MUTATION SURVIVED: {fn}: {old[:60]!r} -> check {check} still passed")
        print(f"  mutation RED as required: {fn} [{old[:48].strip()}] -> {check}", flush=True)
        shutil.rmtree(td, ignore_errors=True)


def main() -> int:
    if "--check" in sys.argv:
        name = sys.argv[sys.argv.index("--check") + 1]
        CHECKS[name]()
        print(f"CHECK_OK {name} {N}")
        return 0
    for name, fn in CHECKS.items():
        fn()
        print(f"  {name}: ok ({N})", flush=True)
    if "--fast" not in sys.argv:
        run_mutations()
    print(f"ZZP6_PASS {N}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
