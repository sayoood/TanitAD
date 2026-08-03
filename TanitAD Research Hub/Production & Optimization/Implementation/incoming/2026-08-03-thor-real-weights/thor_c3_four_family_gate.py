"""Thor P6 / runbook O1 — THE FOUR-FAMILY ACCURACY GATE ON THE OPTIMISED PIPELINE, REAL WEIGHTS.

The runbook's §2 says, in its own words, that its precision table is *"an ARCHITECTURE READ … NOT a
deployment precision gate"*, and that the four-family gate on real windows with a trained
checkpoint **has not run**. This is that run.

⛔ BINDING (Sayed 2026-08-02): every eval reports LONGITUDINAL + LATERAL + TACTICAL + STRATEGIC in
ADDITION to ADE, per-family, each with the paired episode-cluster bootstrap. ADE alone is
INCOMPLETE.

TWO ARMS, IDENTICAL WINDOWS, ONE PROCESS
  A  fp32 eager                                          — the reference the deployment must match
  B  bf16-autocast encoder + TensorRT-fp16 predictor      — the shipped optimisation

Pre-registered falsifiers, both outcomes committed in advance:
  F-ADE       paired dADE@2s CI excludes 0 AND |dADE| > 0.02 m   (the programme's own §7.10 bar)
              => fp16 changes the trajectory materially and does not ship as-is.
  F-LON/LAT   any longitudinal or lateral family metric's paired CI excludes 0 with a delta
              beyond its own CI half-width on the fp32 arm => that family degrades.
  F-TACTICAL  manoeuvre decision agreement between arms < 95.3 % (the inherited bar)
              => the optimisation silently changes decisions and does not ship.
  F-STRATEGIC route decision agreement < 95.3 %  => same.

⚠️ A NEGATIVE IS A RESULT. If nothing fires, the reading is "fp16 is deployment-safe on this
checkpoint at this geometry" — not "quantisation is always safe".

⚠️ SCOPE, stated up front: the encoder here is v1's 256x256 SQUARE raster because that is the
raster flagship-v1-speedjerk was TRAINED at. Scoring it at 176x624 would violate the parity rule.
The v5f arm on Thor is step 1000 and is a latency/numerics control only.
"""
import json
import os
import subprocess
import sys
import time

for _p in ("~/TanitAD/stack", "~/TanitAD/stack/scripts", "~/TanitAD/taniteval"):
    sys.path.insert(0, os.path.expanduser(_p))
sys.path.insert(0, "/usr/lib/python3.12/dist-packages")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEV = "cuda"
OUT_JSON = os.path.expanduser("~/thor_c3_four_family_gate.json")
WORK = os.path.expanduser("~/trt_c3")
os.makedirs(WORK, exist_ok=True)
V1_CKPT = os.path.expanduser("~/models/flagship-v1-speedjerk/ckpt.pt")
V1_VAL = os.path.expanduser("~/valdata/physicalai-val-0c5f7dac3b11")
WIN, K_MAX, GOAL_H, STRIDE, BATCH = 8, 20, 20, 8, 8
AGREE_BAR = 0.953

OUT = {"purpose": "runbook O1 — four-family accuracy gate on the TRT-fp16 + bf16 pipeline, "
                  "REAL trained weights, REAL held-out windows",
       "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
       "arms": {"A": "fp32 eager (reference)",
                "B": "bf16-autocast encoder + TensorRT-fp16 predictor (shipped optimisation)"},
       "binding_rule": "Sayed 2026-08-02 — four families ADDED to ADE, per-family, paired "
                       "episode-cluster bootstrap"}
import tensorrt as trt  # noqa: E402
OUT["trt_version"] = trt.__version__
OUT["thor_repo_sha"] = subprocess.run(
    ["git", "-C", os.path.expanduser("~/TanitAD"), "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True).stdout.strip()


def bank(tag=""):
    with open(OUT_JSON, "w") as f:
        json.dump(OUT, f, indent=1, default=str)
    print(f"[bank] {tag}", flush=True)


# ===================================================================== model + data
from tanitad.config import flagship4b_config                   # noqa: E402
from tanitad.models.fourbrain import WorldModel                # noqa: E402
from tanitad.models.metric_dynamics import (HierarchicalGrounding,  # noqa: E402
                                            rollout_decode)
from driving_diagnostic import (WP_STEPS, baseline_waypoints,  # noqa: E402
                                gt_ego_waypoints, net_heading_change_deg)
import refb_labels as rl                                       # noqa: E402
from taniteval import data as TD                               # noqa: E402
from taniteval import rollout as RO                            # noqa: E402
from taniteval import four_families as FF                      # noqa: E402
from taniteval import ci as CI                                 # noqa: E402

cfg = flagship4b_config()
object.__setattr__(cfg.predictor, "action_dim", 3)
if cfg.tactical_pred is not None:
    object.__setattr__(cfg.tactical_pred, "action_dim", 3)
object.__setattr__(cfg.encoder, "grad_checkpoint", False)
model = WorldModel(cfg)
grounding = HierarchicalGrounding(model.state_dim)
ck = torch.load(V1_CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"])          # STRICT — real trained weights
grounding.load_state_dict(ck["grounding"])  # STRICT
model = model.to(DEV).eval()
grounding = grounding.to(DEV).eval()
step_readout = grounding.step["op"]
OUT["model"] = {"ckpt": V1_CKPT, "step": int(ck.get("step", -1)),
                "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                "raster": f"{cfg.encoder.image_size}x{cfg.encoder.image_size} SQUARE (v1's "
                          f"TRAINED raster — parity rule)",
                "has_tactical_policy": model.tactical_policy is not None,
                "has_strategic_policy": model.strategic_policy is not None}
print(OUT["model"], flush=True)

import glob as _glob  # noqa: E402
_bad = []
for _f in sorted(_glob.glob(os.path.join(V1_VAL, "ep_*.pt"))):
    try:
        _ = torch.load(_f, map_location="cpu", weights_only=True, mmap=True)["frames_u8"].shape
    except Exception as _e:
        _bad.append(os.path.basename(_f))
files = [f for f in TD.list_val_episodes(V1_VAL, n=40, allow_partial=True)
         if os.path.basename(str(f)) not in set(_bad)]
eps = TD.load_frames(files)
OUT["val"] = {"dir": V1_VAL, "n_episodes": len(eps), "excluded_corrupt": _bad,
              "parity": TD.last_val_parity(),
              "decision_grade_absolute": not _bad,
              "note": ("the PAIRED delta is on identical windows in both arms and is unaffected "
                       "by the excluded clip; only ABSOLUTE levels lose the 40/40 stamp")}
print(OUT["val"], flush=True)
bank("setup")

# ===================================================================== TRT predictor (dynamic B)
class PredWrap(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, states, actions):
        return self.p(states, actions)[1]


class TRTPredictor:
    """Drop-in for ``model.predictor``: returns (None, z_next) so ``rollout_decode`` is untouched.
    ⛔ Binds by NAME, never by index."""

    def __init__(self, plan_path):
        with open(plan_path, "rb") as f:
            self.engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(
                f.read())
        assert self.engine is not None
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        for n in ("states", "actions", "z_next"):
            assert n in names, f"engine IO {names} lacks {n!r}"

    def __call__(self, states, actions):
        st = states.detach().contiguous().float()
        ac = actions.detach().contiguous().float()
        self.ctx.set_input_shape("states", tuple(st.shape))
        self.ctx.set_input_shape("actions", tuple(ac.shape))
        out = torch.empty(tuple(self.ctx.get_tensor_shape("z_next")),
                          device=DEV, dtype=torch.float32)
        self.ctx.set_tensor_address("states", st.data_ptr())
        self.ctx.set_tensor_address("actions", ac.data_ptr())
        self.ctx.set_tensor_address("z_next", out.data_ptr())
        assert self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.current_stream().synchronize()
        return (None, out)


S, A = model.state_dim, cfg.predictor.action_dim
onnx_p, plan_p = f"{WORK}/pred_dyn.onnx", f"{WORK}/pred_dyn_fp16.plan"
torch.backends.mha.set_fastpath_enabled(False)      # runbook §4: cheap insurance, kept
st0 = torch.randn(BATCH, WIN, S, device=DEV)
ac0 = torch.randn(BATCH, WIN, A, device=DEV)
torch.onnx.export(PredWrap(model.predictor).eval(), (st0, ac0), onnx_p,
                  input_names=["states", "actions"], output_names=["z_next"],
                  dynamic_axes={"states": {0: "B"}, "actions": {0: "B"}, "z_next": {0: "B"}},
                  opset_version=17, dynamo=False)
t0 = time.perf_counter()
r = subprocess.run(["/usr/src/tensorrt/bin/trtexec", f"--onnx={onnx_p}",
                    f"--saveEngine={plan_p}", "--fp16", "--skipInference",
                    f"--minShapes=states:1x{WIN}x{S},actions:1x{WIN}x{A}",
                    f"--optShapes=states:{BATCH}x{WIN}x{S},actions:{BATCH}x{WIN}x{A}",
                    f"--maxShapes=states:{BATCH}x{WIN}x{S},actions:{BATCH}x{WIN}x{A}"],
                   capture_output=True, text=True, timeout=3600)
assert os.path.exists(plan_p), (r.stderr or r.stdout)[-800:]
OUT["engine"] = {"plan": plan_p, "build_s": round(time.perf_counter() - t0, 1),
                 "MB": round(os.path.getsize(plan_p) / 1e6, 1),
                 "dynamic_batch": f"1..{BATCH}", "fastpath": "OFF (runbook §4)"}
trt_pred = TRTPredictor(plan_p)
print("engine", OUT["engine"], flush=True)

# ⛔ NEGATIVE CONTROL BEFORE ANY SCORE: prove the engine is not silently returning the eager
# result (a wired-through test would make every delta trivially 0 and the gate meaningless).
with torch.no_grad():
    _a = model.predictor(st0, ac0)[1]
    _b = trt_pred(st0, ac0)[1]
    _c = trt_pred(st0, ac0 * 0)[1]
OUT["negative_control"] = {
    "engine_vs_eager_rel_err": round(float((_a - _b).norm() / _a.norm()), 8),
    "engine_responds_to_action_change_rel": round(float((_b - _c).norm() / _b.norm()), 6),
    "discriminates": bool(float((_b - _c).norm() / _b.norm()) > 1e-3),
    "why": ("an engine that ignored its inputs, or one aliased to the eager tensor, would make "
            "every family delta 0 and the gate would 'pass' vacuously")}
assert OUT["negative_control"]["discriminates"], "engine does not respond to its inputs"
print("negative_control", OUT["negative_control"], flush=True)
bank("engine")


# ===================================================================== the scored pass
@torch.no_grad()
def collect_arm(arm: str):
    """arm 'A' = fp32 eager everywhere; arm 'B' = bf16 encoder/heads + TRT-fp16 predictor."""
    bf16 = (arm == "B")
    pred_fn = trt_pred if bf16 else model.predictor
    P, G, CV, EID, SPD, HDG = [], [], [], [], [], []
    MP, MT, RF, RN, RT, VAL = [], [], [], [], [], []
    t0 = time.perf_counter()
    for ei, ep in enumerate(eps):
        T = min(ep.feats.shape[0], ep.actions.shape[0], ep.poses.shape[0])
        starts = list(range(0, T - WIN - K_MAX, STRIDE))
        for i in range(0, len(starts), BATCH):
            ch = starts[i:i + BATCH]
            last = torch.tensor([t + WIN - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(ep.feats[t:t + WIN]) for t in ch]).to(DEV)
            fw = fw.float().div_(255.0)
            aw = torch.stack([ep.actions[t:t + WIN] for t in ch]).to(DEV)
            fa = torch.stack([ep.actions[t + WIN:t + WIN + K_MAX] for t in ch]).to(DEV)
            aw, fa = RO.append_ego(aw, fa, ep.poses, last, True, False, False, DEV)

            if bf16:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    states = model.encode_window(fw)
                states = states.float()
            else:
                states = model.encode_window(fw)

            wp_full, _ = rollout_decode(pred_fn, states, aw, fa, step_readout, K_MAX)
            P.append(wp_full.index_select(1, torch.tensor([k - 1 for k in WP_STEPS],
                                                          device=DEV)).cpu().float())
            G.append(gt_ego_waypoints(ep.poses, last))
            CV.append(baseline_waypoints(ep.poses, last)["constant_velocity"])
            EID.extend([ep.episode_id] * len(ch))
            SPD.append(ep.poses[last, 3])
            HDG.append(net_heading_change_deg(ep.poses, last))

            # ---- TACTICAL + STRATEGIC decisions, under the SAME arm precision ----
            navs, rts, valids = [], [], []
            for t in ch:
                cmd, valid = rl.nav_command(ep.poses, t + WIN - 1)
                navs.append(cmd)
                rts.append(rl.route_target(cmd))
                valids.append(bool(valid))
            nav = torch.tensor(navs, device=DEV)
            follow = torch.zeros(len(ch), dtype=torch.long, device=DEV)
            ctxmgr = (torch.autocast("cuda", dtype=torch.bfloat16) if bf16
                      else torch.autocast("cuda", enabled=False))
            with ctxmgr:
                sf = model.strategic_policy(states, follow)
                sn = model.strategic_policy(states, nav)
                tacf = model.tactical_policy(states, sf["ctx"])
            MP += tacf["maneuver_logits"].float().argmax(-1).cpu().tolist()
            RF += sf["route_logits"].float().argmax(-1).cpu().tolist()
            RN += sn["route_logits"].float().argmax(-1).cpu().tolist()
            RT += rts
            VAL += valids
            fut = torch.stack([torch.as_tensor(ep.poses[t + WIN:t + WIN + GOAL_H])
                               for t in ch]).to(DEV).float()
            pl = ep.poses[last].to(DEV).float()
            MT += rl.classify_maneuver(pl[:, 2], fut[:, GOAL_H - 1, 2],
                                       pl[:, 3], fut[:, GOAL_H - 1, 3]).long().cpu().tolist()
        if ei % 10 == 0:
            print(f"  [{arm}] ep {ei}/{len(eps)} n={len(EID)} "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
    return {"pred": torch.cat(P), "gt": torch.cat(G).float(), "cv": torch.cat(CV).float(),
            "eid": EID, "speed": torch.cat(SPD).float(), "head_deg": torch.cat(HDG).float(),
            "wp_steps": list(WP_STEPS),
            "maneuver_pred": MP, "maneuver_gt": MT,
            "route_pred": RF, "route_pred_nav": RN, "route_gt": RT, "route_valid": VAL,
            "wall_s": round(time.perf_counter() - t0, 1)}


print("=== ARM A: fp32 eager ===", flush=True)
WA = collect_arm("A")
print("=== ARM B: bf16 encoder + TRT-fp16 predictor ===", flush=True)
WB = collect_arm("B")
OUT["n_windows"] = len(WA["eid"])
OUT["wall_s"] = {"A": WA["wall_s"], "B": WB["wall_s"]}
assert len(WA["eid"]) == len(WB["eid"]), "arms are not on identical windows"
bank("collected")


# ===================================================================== metrics
def ade_per_window(w, upto=None):
    """Per-window mean waypoint displacement (m). upto=None -> all 4 wp (0-2 s)."""
    d = (w["pred"] - w["gt"]).norm(dim=-1)          # [N, 4]
    if upto is not None:
        d = d[:, :upto]
    return d.mean(-1).numpy()


def seq(w):
    return FF._seq_geometry(w["pred"].float()), FF._seq_geometry(w["gt"].float())


def per_window_family(w):
    """Per-window scalars for each family, so the PAIRED bootstrap can run on them."""
    P, G = seq(w)
    import math
    dh = P["heading"] - G["heading"]
    dh = (dh + math.pi) % (2 * math.pi) - math.pi
    both = P["valid"] & G["valid"]
    bp = P["pair_valid"] & G["pair_valid"]

    def _m(x, m):
        num = (x * m).sum(-1)
        den = m.sum(-1).clamp(min=1)
        return (num / den).numpy()
    return {
        "speed_mae_mps": (P["speed"] - G["speed"]).abs().mean(-1).numpy(),
        "speed_bias_mps": (P["speed"] - G["speed"]).mean(-1).numpy(),
        "along_mae_m": (P["along"] - G["along"]).abs().mean(-1).numpy(),
        "accel_mae_mps2": (P["accel"] - G["accel"]).abs().mean(-1).numpy(),
        "heading_mae_deg": _m(dh.abs().rad2deg(), both),
        "yaw_rate_mae_degps": _m((P["yaw_rate"] - G["yaw_rate"]).abs().rad2deg(), bp),
        "curvature_mae_1pm": _m((P["curvature"] - G["curvature"]).abs(), bp),
        "cross_mae_m": (P["cross"] - G["cross"]).abs().mean(-1).numpy(),
    }


eid = WA["eid"]
res = {"ADE": {}, "LONGITUDINAL": {}, "LATERAL": {}, "TACTICAL": {}, "STRATEGIC": {}}

# ---- ADE (still reported — it is ONE ROW of four families, never "the result") ----
for label, upto in (("ade_0_2s", None), ("ade_0_5s", 1), ("ade_1s", 2), ("ade_2s_final", None)):
    a, b = ade_per_window(WA, upto), ade_per_window(WB, upto)
    res["ADE"][label] = {
        "A_fp32": CI.episode_cluster_bootstrap(a, eid),
        "B_opt": CI.episode_cluster_bootstrap(b, eid),
        "paired_delta_B_minus_A": CI.paired_episode_cluster_bootstrap(b, a, eid)}
# the exact horizon the §7.10 falsifier is written against: the 2 s waypoint alone
d2A = (WA["pred"] - WA["gt"]).norm(dim=-1)[:, -1].numpy()
d2B = (WB["pred"] - WB["gt"]).norm(dim=-1)[:, -1].numpy()
res["ADE"]["fde_2s"] = {"A_fp32": CI.episode_cluster_bootstrap(d2A, eid),
                        "B_opt": CI.episode_cluster_bootstrap(d2B, eid),
                        "paired_delta_B_minus_A": CI.paired_episode_cluster_bootstrap(
                            d2B, d2A, eid)}

# ---- LONGITUDINAL + LATERAL: full family block per arm + paired per-metric CI ----
famA, famB = FF.all_families(WA), FF.all_families(WB)
res["LONGITUDINAL"]["A_fp32"] = famA["longitudinal"]
res["LONGITUDINAL"]["B_opt"] = famB["longitudinal"]
res["LATERAL"]["A_fp32"] = famA["lateral"]
res["LATERAL"]["B_opt"] = famB["lateral"]
pwA, pwB = per_window_family(WA), per_window_family(WB)
LON_KEYS = ("speed_mae_mps", "speed_bias_mps", "along_mae_m", "accel_mae_mps2")
LAT_KEYS = ("heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm", "cross_mae_m")
res["LONGITUDINAL"]["paired_delta_B_minus_A"] = {
    k: CI.paired_episode_cluster_bootstrap(pwB[k], pwA[k], eid) for k in LON_KEYS}
res["LATERAL"]["paired_delta_B_minus_A"] = {
    k: CI.paired_episode_cluster_bootstrap(pwB[k], pwA[k], eid) for k in LAT_KEYS}

# ---- TACTICAL: manoeuvre decision vs GT, per arm, PLUS arm-to-arm decision agreement ----
try:
    from tanitad.refs.refb import MANEUVER_CLASSES as _MC
    MC = list(_MC)
except Exception:
    MC = None
mA = torch.tensor(WA["maneuver_pred"])
mB = torch.tensor(WB["maneuver_pred"])
mT = torch.tensor(WA["maneuver_gt"])
res["TACTICAL"]["A_fp32"] = FF._decision_family(
    {"maneuver_pred": mA, "maneuver_gt": mT}, "tactical", "maneuver_pred", "maneuver_gt", MC)
res["TACTICAL"]["B_opt"] = FF._decision_family(
    {"maneuver_pred": mB, "maneuver_gt": mT}, "tactical", "maneuver_pred", "maneuver_gt", MC)
agree_t = (mA == mB).float().numpy()
corrA_t = (mA == mT).float().numpy()
corrB_t = (mB == mT).float().numpy()
res["TACTICAL"]["decision_agreement_A_vs_B"] = CI.episode_cluster_bootstrap(agree_t, eid)
res["TACTICAL"]["paired_delta_accuracy_B_minus_A"] = CI.paired_episode_cluster_bootstrap(
    corrB_t, corrA_t, eid)
res["TACTICAL"]["n_flipped_decisions"] = int((mA != mB).sum())

# ---- STRATEGIC: route decision (vision-only 'follow' is the SKILL test; nav is privileged) ----
try:
    from tanitad.refs.refb import ROUTE_CLASSES as _RC
    RC = list(_RC)
except Exception:
    RC = None
rA = torch.tensor(WA["route_pred"])
rB = torch.tensor(WB["route_pred"])
rNA = torch.tensor(WA["route_pred_nav"])
rNB = torch.tensor(WB["route_pred_nav"])
rT = torch.tensor(WA["route_gt"])
vmask = torch.tensor(WA["route_valid"], dtype=torch.bool)
res["STRATEGIC"]["A_fp32_follow_visiononly"] = FF._decision_family(
    {"route_pred": rA, "route_gt": rT}, "strategic", "route_pred", "route_gt", RC)
res["STRATEGIC"]["B_opt_follow_visiononly"] = FF._decision_family(
    {"route_pred": rB, "route_gt": rT}, "strategic", "route_pred", "route_gt", RC)
agree_s = (rA == rB).float().numpy()
res["STRATEGIC"]["decision_agreement_A_vs_B_follow"] = CI.episode_cluster_bootstrap(agree_s, eid)
res["STRATEGIC"]["decision_agreement_A_vs_B_nav"] = CI.episode_cluster_bootstrap(
    (rNA == rNB).float().numpy(), eid)
res["STRATEGIC"]["paired_delta_accuracy_B_minus_A"] = CI.paired_episode_cluster_bootstrap(
    (rB == rT).float().numpy(), (rA == rT).float().numpy(), eid)
res["STRATEGIC"]["majority_straight_rate"] = round(
    float((rT == 1).float().mean()), 4)
res["STRATEGIC"]["n_flipped_decisions"] = int((rA != rB).sum())
res["STRATEGIC"]["route_label_valid_frac"] = round(float(vmask.float().mean()), 4)
res["STRATEGIC"]["⛔_void_by_construction"] = (
    "GATE_PROTOCOL §0.7 — route_acc_nav is PRIVILEGED (the answer is an input). The skill read is "
    "the vision-only 'follow' column against majority_straight_rate. This gate's PRIMARY question "
    "is nevertheless AGREEMENT between precisions, which is unaffected by that caveat.")

# ---- distance keeping: honest UNAVAILABLE with reason + n (binding rule #5) ----
res["LONGITUDINAL"]["distance_keeping"] = famA["longitudinal"]["distance_keeping"]
OUT["FOUR_FAMILIES"] = res
bank("families")

# ===================================================================== verdicts
dade = res["ADE"]["ade_0_2s"]["paired_delta_B_minus_A"]
ta = res["TACTICAL"]["decision_agreement_A_vs_B"]["mean"]
sa = res["STRATEGIC"]["decision_agreement_A_vs_B_follow"]["mean"]
lon_fire = [k for k, v in res["LONGITUDINAL"]["paired_delta_B_minus_A"].items()
            if v["separated"] and not v.get("degenerate")]
lat_fire = [k for k, v in res["LATERAL"]["paired_delta_B_minus_A"].items()
            if v["separated"] and not v.get("degenerate")]
OUT["VERDICT"] = {
    "F_ADE": {"bar": "|paired dADE@0-2s| > 0.02 m AND CI excludes 0",
              "delta_m": dade["delta"], "ci": [dade["lo"], dade["hi"]],
              "separated": dade["separated"], "degenerate": dade.get("degenerate", False),
              "fires": bool(dade["separated"] and abs(dade["delta"]) > 0.02
                            and not dade.get("degenerate"))},
    "F_LONGITUDINAL": {"bar": "any longitudinal metric's paired CI excludes 0 (non-degenerate)",
                       "separated_metrics": lon_fire, "fires": bool(lon_fire)},
    "F_LATERAL": {"bar": "any lateral metric's paired CI excludes 0 (non-degenerate)",
                  "separated_metrics": lat_fire, "fires": bool(lat_fire)},
    "F_TACTICAL": {"bar": f"manoeuvre decision agreement < {AGREE_BAR}",
                   "agreement": ta, "n_flipped": res["TACTICAL"]["n_flipped_decisions"],
                   "fires": bool(ta < AGREE_BAR)},
    "F_STRATEGIC": {"bar": f"route decision agreement < {AGREE_BAR}",
                    "agreement": sa, "n_flipped": res["STRATEGIC"]["n_flipped_decisions"],
                    "fires": bool(sa < AGREE_BAR)}}
fired = [k for k, v in OUT["VERDICT"].items() if v.get("fires")]
OUT["VERDICT"]["ANY_FIRED"] = fired
OUT["VERDICT"]["READING"] = (
    ("⛔ %s fired — the optimised pipeline is NOT a drop-in for fp32 on this checkpoint" % fired)
    if fired else
    "✅ no falsifier fired: on flagship-v1-speedjerk @256px, bf16 encoder + TRT-fp16 predictor is "
    "deployment-equivalent to fp32 across ADE and all four families, at this n. This is a "
    "CHECKPOINT-SPECIFIC and GEOMETRY-SPECIFIC pass, not a general licence for quantisation.")
OUT["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
bank("DONE")
print(json.dumps(OUT["VERDICT"], indent=1), flush=True)

# window dumps so this is re-analysable without the GPU (banked, per the corpus rule)
torch.save({"A": {k: v for k, v in WA.items()}, "B": {k: v for k, v in WB.items()}},
           os.path.expanduser("~/thor_c3_windows.pt"))
print("=== DONE ===", flush=True)
