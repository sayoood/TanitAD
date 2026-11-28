"""lead_target_check.py <tree root> -- controls of the lead on the TARGET build (card PREREG_V8F_PROOF_ADD19), independent of the checks of the build agent.  CPU.

c1 FUTTOK, analytic value: with the weight of tok_head set to 0 and a random bias b, the term must equal the mean over the starts s in (0, 2, 4) of
   mean((b - LN(feats[:, s+1 : s+K+1]))^2), computed here directly from the token grid (tolerance 1e-6). The same formula on the frames shifted by one
   (s : s+K) must NOT match (the control can see an off-by-one). The token grid of c1 gives every frame its own block of features,
   because on i.i.d. random tokens the shifted formula differs by 2e-5 only (first run of this control, bar set too tight by the lead).
c2 FUTTOK, frame alignment: changing the tokens of frame 0 or of frame 15 (never a target with K 10 and 16 frames) leaves the term exactly unchanged (0.0);
   changing frame 14 or frame 1 changes it.
c3 EGOV, one spelling: the third channel of scale_actions equals speed / WM_SPEED_SCALE exactly, and the transition j -> j+1 carries the speed of frame j;
   with the lever off scale_actions ignores the speed and returns the two channels of the base tree.
Exit 0 all pass, 2 a control failed.
"""
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
import torch.nn.functional as F

tree = sys.argv[1]
import tanitad  # noqa: E402

assert os.path.abspath(tanitad.__file__).lower().startswith(os.path.abspath(tree).lower()), tanitad.__file__
from tanitad.models import v8f_mini as M  # noqa: E402

K, B = 10, 2
res = {"tanitad_file": tanitad.__file__}


def wm(pred, fk=0, fw=0.0, act="acc_yaw", seed=0):
    cfg = M.V8FMiniConfig(wm="b", hmax=16, wm_state="grid", wm_l1_pred=pred, wm_l1_futtok_k=fk, w_wm_futtok=fw, wm_l1_act=act, d=32, ff=64, heads=4, n_q=2,
                          depth_s=1, depth_t=1, depth_o=1, nb=6)
    torch.manual_seed(seed)
    return cfg, M.WMAux(cfg)


def inputs(cfg, seed=5):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, 16, cfg.n_tok, cfg.d, generator=g), torch.randn(B, 16, 2, generator=g), torch.rand(B, 16, generator=g) * 20.0,
            torch.randn(B, 16, cfg.n_tok, cfg.d_feat, generator=g).half())


ok = True
for pred in ("flat", "cell"):
    cfg, w = wm(pred, fk=K, fw=1.0)
    w.eval()
    adapt, act, spd, feats = inputs(cfg)
    with torch.no_grad():
        w.tok_head.weight.zero_()
        w.tok_head.bias.copy_(torch.randn(w.tok_head.bias.shape, generator=torch.Generator().manual_seed(9)))
        fc = (0.1 * feats.float()).clone()                              # c1 token grid: every frame j carries its own block of features (+5.0), so a shift by one frame is visible
        for j in range(16):
            fc[:, j, :, 64 * j:64 * (j + 1)] += 5.0
        fc = fc.half()
        term = float(w(adapt, act, feats=fc)["futtok/L1"])
        tgt = F.layer_norm(fc.float(), (fc.shape[-1],))
        b = w.tok_head.bias
        direct = float(torch.stack([(b - tgt[:, s + 1:s + K + 1]).pow(2).mean() for s in (0, 2, 4)]).mean())
        shifted = float(torch.stack([(b - tgt[:, s:s + K]).pow(2).mean() for s in (0, 2, 4)]).mean())
        c1 = {"term": term, "direct": direct, "abs_diff": abs(term - direct), "shifted_by_one": shifted, "abs_diff_shifted": abs(term - shifted),
              "pass": bool(abs(term - direct) <= 1e-6 and abs(term - shifted) > 1e-3)}
        # c2: frame alignment with the trained-shape head (random weight again)
        cfg2, w2 = wm(pred, fk=K, fw=1.0, seed=1)
        w2.eval()
        base = float(w2(adapt, act, feats=feats)["futtok/L1"])
        d = {}
        for fr in (0, 1, 14, 15):
            f2 = feats.clone()
            f2[:, fr] = torch.randn(f2[:, fr].shape, generator=torch.Generator().manual_seed(100 + fr)).half()
            d[str(fr)] = abs(float(w2(adapt, act, feats=f2)["futtok/L1"]) - base)
        c2 = {"abs_change_by_frame": d, "pass": bool(d["0"] == 0.0 and d["15"] == 0.0 and d["1"] > 1e-6 and d["14"] > 1e-6)}
    res["futtok_" + pred] = {"c1_analytic_value": c1, "c2_frame_alignment": c2}
    ok &= c1["pass"] and c2["pass"]

# c3 EGOV
cfg, w = wm("cell", act="speed")
adapt, act, spd, feats = inputs(cfg)
a3 = w.scale_actions(act, spd)
cfg0, w0 = wm("cell")
a2 = w0.scale_actions(act)
a2s = w0.scale_actions(act, spd)
used = w.scale_actions(act, spd)[:, :-1]
c3 = {"shape_on": list(a3.shape), "shape_off": list(a2.shape), "scale": float(M.WM_SPEED_SCALE),
      "third_channel_max_abs_diff": float((a3[..., 2] - spd / M.WM_SPEED_SCALE).abs().max()),
      "first_two_equal_to_base": bool(torch.equal(a3[..., :2], a2)), "off_ignores_speed": bool(torch.equal(a2s, a2)),
      "transition_j_carries_speed_of_frame_j": bool(torch.equal(used[:, :, 2], (spd / M.WM_SPEED_SCALE)[:, :-1]))}
c3["pass"] = bool(c3["shape_on"][-1] == 3 and c3["shape_off"][-1] == 2 and c3["third_channel_max_abs_diff"] == 0.0 and c3["first_two_equal_to_base"]
                  and c3["off_ignores_speed"] and c3["transition_j_carries_speed_of_frame_j"] and c3["scale"] == 10.0)
try:
    w.scale_actions(act)
    c3["on_without_speed_raises"] = False
except Exception as e:  # noqa: BLE001
    c3["on_without_speed_raises"] = True
    c3["error_type"] = type(e).__name__
c3["pass"] = bool(c3["pass"] and c3["on_without_speed_raises"])
res["egov"] = {"c3_one_spelling": c3}
ok &= c3["pass"]
res["all_pass"] = bool(ok)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lead_target_check.json")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1))
sys.exit(0 if ok else 2)
