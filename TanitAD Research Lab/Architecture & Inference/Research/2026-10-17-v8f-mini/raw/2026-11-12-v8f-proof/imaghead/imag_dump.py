"""imag_dump.py: the POD-SIDE READ HOOK of the tool `imaghead` (card PREREG_V8F_PROOF_ADD36, section 3). Read-only for the model; CPU or cuda.

For every eval window with a lead label at t (window frame 4 = t, S0 = 4 of the 16-frame window; frame f = bank step g0 - 4 + f) it writes the outputs of the
supervised lead head (LeadSupHead of the patched tree; presence logit, gap [m], closing speed [m/s]) on
  * the present states (frames 0..4), the true future states at 1.0 s (frames 5..9) and 2.0 s (frames 10..14);
  * rolled-out states: the stack's own rollout (HJEPAStack.rollout, level 1, context = the predictor's own, started from the state of frame 4, 10 steps of 0.2 s, the
    context growing by its own predictions) under the logged ego actions and under a constant acceleration offset d added to every step (yaw rate unchanged);
    steps 1..5 are read at 1.0 s, steps 6..10 at 2.0 s. Offsets: 0 (through the offset code path), -2, -1, +1 m/s2;
  * the same rolled-out responses on a COPY of the model whose action embedding (the predictor's action input, level 1) has all weights and biases set to zero (control c4).
The acceleration channel of the level-1 action is acc / WM_ACC_SCALE (v8f_mini.py:201, used at WMAux.scale_actions v8f_mini.py:947); the offset d [m/s2] is therefore
added to the SCALED channel as d / WM_ACC_SCALE (in the units the predictor was trained with).

Input (all of it small, no network):
  --ckpt CKPT            checkpoint trained with --lead-sup on (grid state, level-1 action width 2)
  --tree TREE            the patched tree (contains stack/); tanitad is imported from it and its path asserted
  --windows NPZ          window list in the layout of LATENTS_CTX_<ARM>_R1.npz: eid, g0, (t, w_fold), act_scaled [n, 10, 2] = scaled actions of transitions 4..13
  --pack NPZ             the arm-independent eval pack (leadsup/read/eval_pack.npz): eid, g0, V [n,16] ego speed, G15 [n,15] label gap, pres0, same order as --windows
  --states FILE          eval state file of extract_states.py (Z float16 [N, D], clip_idx, t, clips); window frame f = (clip eid, t = g0 - 4 + f)
  --window-states FILE   or: npz[key Z1] [n, 16, D] already assembled in the order of --windows (tests, other machines)
Output: --out NPZ and --out.json (checkpoint sha256, argv, counts, file:line facts, the control numbers c1, c2, c4 measured at dump time).
Tier T0 (a world-model diagnostic on logged frames): never driving performance."""
import argparse
import copy
import hashlib
import json
import os
import sys
import time

import numpy as np

OFFSETS = (0.0, -2.0, -1.0, 1.0)       # constant acceleration offsets [m/s2], d = 0 goes through the offset code path
S0, K, FRAMES = 4, 10, 16
STEPS_T = {"1.0": (0, 5), "2.0": (5, 10)}   # rollout steps (0-based slice) the head reads at 1.0 s and 2.0 s
FACTS = {   # file:line facts of the patched tree (re-asserted against the source by facts_check)
    "wm_acc_scale": ("stack/tanitad/models/v8f_mini.py", 201, "WM_ACC_SCALE = 2.0"),
    "scale_actions_acc": ("stack/tanitad/models/v8f_mini.py", 947, "act[..., 0] / WM_ACC_SCALE"),
    "lead_head": ("stack/tanitad/models/v8f_mini.py", 814, "class LeadSupHead"),
    "rollout": ("stack/tanitad/models/hjepa.py", 919, "def rollout(self, level"),
    "act_embed": ("stack/tanitad/models/hjepa.py", 650, "self.act_embed = nn.Sequential(nn.Linear(c1.action_dim"),
    "roll_pairs": ("stack/tanitad/models/hjepa.py", 1, "def roll_pairs"),   # line looked up by text
}


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def facts_check(tree):
    """The file:line facts, each verified by reading the line of the tree (text must be on the stated line; roll_pairs is located by text)."""
    out = {}
    for k, (rel, ln, txt) in FACTS.items():
        with open(os.path.join(tree, rel), encoding="utf-8") as fh:
            lines = fh.read().split("\n")
        if k == "roll_pairs":
            ln = next(i + 1 for i, s in enumerate(lines) if txt in s)
        out[k] = {"file": rel, "line": ln, "text": lines[ln - 1].strip()[:140], "ok": bool(txt in lines[ln - 1])}
    return out


def import_stack(tree):
    tree = os.path.abspath(tree).replace(os.sep, "/")
    sp = tree + "/stack"
    if sp not in sys.path:
        sys.path.insert(0, sp)
    import torch
    import tanitad
    got = os.path.abspath(tanitad.__file__).replace(os.sep, "/")
    if not got.lower().startswith(sp.lower() + "/"):
        raise SystemExit(f"REFUSED: tanitad imported from {got}, expected under {sp}")
    from tanitad.models import v8f_mini as M
    return torch, M, tree


def load_model(torch, M, ckpt, require_head=True):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    if require_head and cfg.lead_sup != "on":
        raise SystemExit(f"REFUSED: {ckpt} was trained with lead_sup {cfg.lead_sup!r}: no lead head to read")
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    if model.wm is None or cfg.wm_state != "grid":
        raise SystemExit("REFUSED: needs the world-model auxiliary with a grid state")
    st = model.wm.stack
    if int(st.cfg[0].action_dim) != 2:
        raise SystemExit(f"REFUSED: level-1 action width {st.cfg[0].action_dim} (this tool reads acc_yaw arms; the window file carries 2 scaled channels)")
    return cfg, model, st, ck


def predictor_context(st):
    return int(getattr(st.predictors[0], "context", 0) or 3)       # as ctx_roll.py


def offset_actions(A, d_phys, acc_scale):
    """A [n, K, 2] scaled actions -> the same with the acceleration channel + d / acc_scale (d in m/s2; yaw channel untouched). d = 0 runs the same code (adds 0.0)."""
    B = A.clone()
    B[..., 0] = B[..., 0] + float(d_phys) / float(acc_scale)
    return B


def head_raw(model, Z5):
    """Z5 [n, 5, D] states -> [n, 3] raw head output at the end frame of the five (presence logit, gap / 50 m, closing speed / 10 m/s): the trainer's own units."""
    return model.wm.lead_head(Z5.float())[:, 0]


def head_phys(model, M, Z5):
    """Z5 [n, 5, D] states -> [n, 3] = (presence logit, gap [m], closing speed [m/s]) at the end frame of the five."""
    return torch_stack_scale(head_raw(model, Z5), M)


def torch_stack_scale(o, M):
    import torch
    return torch.stack([o[:, 0], o[:, 1] * M.LEAD_SUP_GAP_SCALE_M, o[:, 2] * M.LEAD_SUP_CLOS_SCALE_MS], 1)


def roll_states(torch, st, z0, A, ctx):
    """The stack's own rollout (HJEPAStack.rollout, level 1) from z0 [n, D] under scaled actions A [n, K, 2] -> [n, K, D]."""
    with torch.no_grad():
        return st.rollout(1, z0, A, context=ctx)


def head_reads_rolled(torch, model, M, R):
    """R [n, 10, D] rolled-out states -> [n, 2, 3]: the head at 1.0 s (steps 1..5) and 2.0 s (steps 6..10)."""
    return torch.stack([head_phys(model, M, R[:, a:b]) for a, b in STEPS_T.values()], 1)


def zero_action_input(torch, model):
    """Control c4: a deep copy whose action embedding of level 1 (the predictor's action input) is all zeros (weights and biases)."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        for p in m.wm.stack.act_embed.parameters():
            p.zero_()
    return m.eval()


def compute_reads(torch, M, model, st, Z, A, offsets=OFFSETS, chunk=256, zero_model=None):
    """Z [n, 16, D] float32 states, A [n, 10, 2] scaled actions -> dict of arrays (physical units: logit, gap m, closing m/s on the last axis).
    present [n,3], true1 [n,3], true2 [n,3]; roll_logged [n,2,3]; roll_off [n,len(offsets),2,3]; the z_* versions on the zero-action-input copy."""
    ctx = predictor_context(st)
    acc_scale = M.WM_ACC_SCALE
    n = Z.shape[0]
    out = {k: [] for k in ("present", "true1", "true2", "roll_logged", "roll_off")}
    if zero_model is not None:
        out.update({"z_roll_logged": [], "z_roll_off": []})
    with torch.no_grad():
        for i in range(0, n, chunk):
            z, a = Z[i:i + chunk], A[i:i + chunk]
            out["present"].append(head_phys(model, M, z[:, 0:5]))
            out["true1"].append(head_phys(model, M, z[:, 5:10]))
            out["true2"].append(head_phys(model, M, z[:, 10:15]))
            for tag, mdl, stk in (("", model, st),) + ((("z_", zero_model, zero_model.wm.stack),) if zero_model is not None else ()):
                R = roll_states(torch, stk, z[:, S0], a, ctx)
                out[tag + "roll_logged"].append(head_reads_rolled(torch, mdl, M, R))
                ro = []
                for d in offsets:
                    Rd = roll_states(torch, stk, z[:, S0], offset_actions(a, d, acc_scale), ctx)
                    ro.append(head_reads_rolled(torch, mdl, M, Rd))
                out[tag + "roll_off"].append(torch.stack(ro, 1))
    return {k: torch.cat(v).numpy().astype(np.float32) for k, v in out.items()}


def control_c2(torch, M, model, st, Z, A, n_check=48):
    """c2 as an equality of two code paths on the SAME model and states.
    (a) the tool's rollout (HJEPAStack.rollout, chunked, no gradient) against the trainer's own rollout core (HJEPAStack.roll_pairs, the start s = 4 of the 16-frame window,
        the embedded actions of the full window) -> max abs difference of the rolled-out states.
    (b) the head on the true states (the tool's 5-frame slices) against the head on the whole 16-frame window sequence, as the trainer's WMAux.forward feeds it (end frames 4, 9, 14);
    compared in the trainer's output units (gap / 50, closing / 10); the same difference in metres (x50) is reported beside it and is not the bar."""
    ctx = predictor_context(st)
    m = min(n_check, Z.shape[0])
    z, a = Z[:m], A[:m]
    with torch.no_grad():
        tool = roll_states(torch, st, z[:, S0], a, ctx)
        afull = torch.zeros(m, FRAMES - 1, 2)
        afull[:, S0:S0 + K] = a
        pairs, _, starts = st.roll_pairs(z, st.act_embed(afull), K)
        ref = pairs[starts.index(S0)]
        d_roll = float((tool - ref).abs().max())
        seq = model.wm.lead_head(z)                                   # [m, 16 - hist + 1, 3], the trainer's call on the window sequence
        hist = model.wm.lead_head.hist
        d_head = d_head_m = 0.0
        for e in (4, 9, 14):
            mine_raw = head_raw(model, z[:, e - hist + 1:e + 1])          # the tool's 5-frame slice, in the trainer's output units (presence logit, gap / 50, closing / 10)
            theirs_raw = seq[:, e - (hist - 1)]
            d_head = max(d_head, float((mine_raw - theirs_raw).abs().max()))
            d_head_m = max(d_head_m, float((head_phys(model, M, z[:, e - hist + 1:e + 1]) - torch_stack_scale(theirs_raw, M)).abs().max()))
    return {"rollout_vs_roll_pairs_max_abs": d_roll, "head_vs_sequence_max_abs": d_head, "head_vs_sequence_max_abs_physical_units": d_head_m, "tol": 1e-5, "n_windows": m, "pass": bool(d_roll <= 1e-5 and d_head <= 1e-5)}


def banked_roll_check(torch, st, Z, A, banked_npz, n_max=2000):
    """Optional: the rolled-out state at step 10 against a banked LATENTS_CTX_<ARM>_R3g.npz pred_2s (same windows), max abs difference (the banked file is float32)."""
    ctx = predictor_context(st)
    b = np.load(banked_npz, allow_pickle=True)
    m = min(Z.shape[0], len(b["pred_2s"]), n_max)
    with torch.no_grad():
        r = roll_states(torch, st, Z[:m, S0], A[:m], ctx)[:, K - 1].numpy()
    return {"max_abs": float(np.abs(r - np.asarray(b["pred_2s"][:m], np.float32)).max()), "n": int(m), "file": banked_npz.replace(os.sep, "/"), "tol": 1e-5}


def assemble_states(torch, st_file, eid, g0):
    z = np.load(st_file, allow_pickle=True)
    names = np.asarray(z["clips"]).astype(str)
    key = {(names[int(c)], int(t)): i for i, (c, t) in enumerate(zip(z["clip_idx"], z["t"]))}
    rows = np.full((len(eid), FRAMES), -1, np.int64)
    for j, (e, g) in enumerate(zip(eid, g0)):
        for f in range(FRAMES):
            rows[j, f] = key.get((str(e), int(g) - S0 + f), -1)
    if (rows < 0).any():
        raise SystemExit(f"REFUSED: {(rows < 0).any(1).sum()} windows have frames missing in the state file")
    return torch.as_tensor(np.asarray(z["Z"])[rows]).float()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--pack", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--states")
    g.add_argument("--window-states")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--banked-roll", default=None, help="optional LATENTS_CTX_<ARM>_R3g.npz of the same checkpoint: step-10 state compared (c2, banked)")
    ap.add_argument("--max-windows", type=int, default=0, help="tests: use the first N selected windows")
    a = ap.parse_args(argv)
    t0 = time.time()
    torch, M, tree = import_stack(a.tree)
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    cfg, model, st, ck = load_model(torch, M, a.ckpt)
    W = np.load(a.windows, allow_pickle=True)
    P = np.load(a.pack, allow_pickle=True)
    eid, g0 = W["eid"].astype(str), W["g0"].astype(np.int64)
    if not (np.array_equal(eid, P["eid"].astype(str)) and np.array_equal(g0, P["g0"].astype(np.int64))):
        raise SystemExit("REFUSED: the window list and the pack do not hold the same windows in the same order")
    A_all = torch.as_tensor(np.asarray(W["act_scaled"], np.float32))
    if A_all.shape[1:] != (K, 2):
        raise SystemExit(f"REFUSED: act_scaled {tuple(A_all.shape)}, expected [n, {K}, 2]")
    if a.states:
        Z_all = assemble_states(torch, a.states, eid, g0)
        in_path, in_kind = a.states, "states"
    else:
        zz = np.load(a.window_states, allow_pickle=True)
        Z_all = torch.as_tensor(np.asarray(zz["Z1"])).float()
        in_path, in_kind = a.window_states, "window-states"
    if Z_all.shape[0] != len(eid) or Z_all.shape[1] != FRAMES:
        raise SystemExit(f"REFUSED: states {tuple(Z_all.shape)} for {len(eid)} windows")
    G15 = np.asarray(P["G15"], np.float64)
    sel = np.flatnonzero(np.isfinite(G15[:, S0]))
    if a.max_windows:
        sel = sel[:a.max_windows]
    Z, A = Z_all[sel].to(dev), A_all[sel].to(dev)
    model, st = model.to(dev), model.wm.stack
    zero_model = zero_action_input(torch, model)
    reads = compute_reads(torch, M, model, st, Z, A, chunk=a.chunk, zero_model=zero_model)
    c2 = control_c2(torch, M, model, st, Z, A)
    if a.banked_roll:
        c2["banked"] = banked_roll_check(torch, st, Z, A, a.banked_roll)
        c2["pass"] = bool(c2["pass"] and c2["banked"]["max_abs"] <= 1e-5)
    c1 = float(np.abs(reads["roll_off"][:, OFFSETS.index(0.0)] - reads["roll_logged"]).max())
    c4 = {"max_abs_response_zero_model": float(max(np.abs(reads["z_roll_off"][:, i] - reads["z_roll_logged"]).max() for i in range(len(OFFSETS)))),
          "max_abs_response_real_model_nonzero_offsets": float(max(np.abs(reads["roll_off"][:, i] - reads["roll_logged"]).max() for i in range(len(OFFSETS)) if OFFSETS[i] != 0.0))}
    V = np.asarray(P["V"], np.float64)
    res = dict(eid=eid[sel], g0=g0[sel], row=sel.astype(np.int64), t=np.asarray(W["t"])[sel] if "t" in W.files else np.full(len(sel), -1), fold=np.asarray(P["fold"])[sel],
               v_t=V[sel, S0], gap_t=G15[sel, S0], gap_t1=G15[sel, 9], gap_t2=G15[sel, 14], pres_t=np.asarray(P["pres0"], np.float64)[sel],
               offsets=np.asarray(OFFSETS, np.float64), **reads)
    meta = {"ckpt": a.ckpt.replace(os.sep, "/"), "ckpt_sha256": sha256_file(a.ckpt), "argv": list(argv if argv is not None else sys.argv[1:]), "tree": tree,
            "tanitad_file": __import__("tanitad").__file__.replace(os.sep, "/"), "cfg_lead": {k: getattr(cfg, k) for k in ("lead_sup", "w_lead_sup", "lead_hist", "lead_sup_detach")},
            "cfg_wm": {k: str(getattr(cfg, k)) for k in ("wm_state", "wm_l1_residual", "wm_l1_roll_k", "wm_l1_act", "wm_l1_pred")},
            "n_windows_total": int(len(eid)), "n_windows_dumped": int(len(sel)), "predictor_context": predictor_context(st), "wm_acc_scale": M.WM_ACC_SCALE,
            "lead_gap_scale_m": M.LEAD_SUP_GAP_SCALE_M, "lead_clos_scale_ms": M.LEAD_SUP_CLOS_SCALE_MS, "offsets_m_s2": list(OFFSETS),
            "offset_rule": "d [m/s2] added to the acceleration channel of the SCALED level-1 action as d / WM_ACC_SCALE at every step; yaw channel unchanged",
            "inputs": {"windows": a.windows.replace(os.sep, "/"), "windows_sha256": sha256_file(a.windows), "pack": a.pack.replace(os.sep, "/"), "pack_sha256": sha256_file(a.pack),
                       in_kind: in_path.replace(os.sep, "/"), in_kind + "_sha256": sha256_file(in_path)},
            "facts": facts_check(tree), "controls_at_dump": {"c1_offset0_max_abs_response": c1, "c2": c2, "c4": c4}, "device": a.device, "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **res, ckpt_sha256=meta["ckpt_sha256"])
    with open(a.out + ".json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=1)
    print(f"imag_dump: {len(sel)} of {len(eid)} windows -> {a.out}; c1 {c1}; c2 {c2['pass']} (roll {c2['rollout_vs_roll_pairs_max_abs']:.2e}, head {c2['head_vs_sequence_max_abs']:.2e}); c4 zero-model {c4['max_abs_response_zero_model']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
