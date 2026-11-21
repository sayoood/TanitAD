r"""leadsup_read.py -- the LEADSUP lead read, section 3a of card PREREG_V8F_PROOF_ADD24 (definitions LEADSUP_READ_DEFS.md).
Tier T0: a world-model diagnostic on logged frames, never driving performance.  CPU only.  Reads files; touches no pod, no box 2, no git.

  python leadsup_read.py pack  [--out-dir DIR]                         # one-off: freeze the arm-independent windows, labels and train-probe truth
  python leadsup_read.py run --arms LS_s0,LS_s1,LSH_s0,LSH_s1,LSD_s0,LSD_s1 --in-dir DIR --out-dir DIR [--B 2000]

Per arm (files in --in-dir, written by the pod steps of run_leadsup_read.sh):
  <ARM>_eval.npz(+.json)  <ARM>_train.npz(+.json)       level-1 states (extract_states.py)
  <ARM>_eval_lead.npz(+.json, +.head.npz)  <ARM>_train_lead.npz(+.json, +.head.npz)   pod_lead_apply.py outputs and head tensors
On the 429 scored windows (48 eval episodes): M1 closing speed at S0, M2 gap at S0 beside the one-frame probe, M3 slot (rule R3 of card ADD20 with the head's
gap and closing speed), M4 presence AUROC over all 1,948 windows; the same four on the train probe clips; paired M1 differences between arm types.  Controls k1..k4
of the card (k5..k7 belong to the lead) plus input checks; no name is written unless every control passes.
STRATEGIC: PAUSED (PI item 25); LONGITUDINAL, LATERAL, TACTICAL: not measured by this read.
"""
import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/")
ROOT = "F:/TanitAD-scratch/v8proof"
for _p in (ROOT + "/spatial/leadhead", ROOT + "/spatial/objslot"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import leadhead_s0 as H  # noqa: E402

O = H.O
RP = H.RP
TIER = O.TIER
FAMILIES = O.FAMILIES
S0F = O.S0F
DT = O.DT
K = 10                                   # 2.0 s
GAP_SCALE_M = 50.0                       # LEAD_SUP_GAP_SCALE_M of the patched model
CLOS_SCALE_MS = 10.0                     # LEAD_SUP_CLOS_SCALE_MS
ARMS = ("LS_s0", "LS_s1", "LSH_s0", "LSH_s1", "LSD_s0", "LSD_s1")
TYPES = ("LS", "LSH", "LSD")
BANKED_OBJSLOT = ROOT + "/spatial/objslot/RESULT_OBJSLOT_S0.json"
BANKED_RB_DIR = O.CT + "/rb"
JOIN = "C:/Users/Admin/lorapod/ship2/join_train.npz"
EXPECT = {"windows": 1948, "scored": 429, "episodes_scored": 48, "train_clips": 1200, "train_rows": 121166}
TOL_K2 = 1e-4
BAR_K3 = 0.02
HEAD_KEYS = ("frame_w", "frame_b", "mlp0_w", "mlp0_b", "mlp2_w", "mlp2_b")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def jdump(o, path):
    O.jdump(o, path)


# ============================================================================================ the head forward (dev box, independent of the pod code)
def load_head(path):
    z = np.load(path, allow_pickle=False)
    for k in HEAD_KEYS:
        if k not in z.files:
            raise SystemExit("REFUSED: head file %s lacks %s" % (path, k))
    T = {k: np.asarray(z[k], np.float32) for k in HEAD_KEYS}
    T["hist"] = int(T["mlp0_w"].shape[1] // T["frame_w"].shape[0])
    return T


def head_forward(T, X):
    """X [n, hist, D] float -> raw head output [n, 3] = (presence logit, gap / 50 m, closing speed / 10 m/s). Per-frame linear, concatenation oldest first,
    linear, GELU, linear: the arithmetic of LeadSupHead built from the six tensors only."""
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        e = F.linear(x, torch.from_numpy(T["frame_w"]), torch.from_numpy(T["frame_b"]))
        w = e.reshape(e.shape[0], -1)
        h = F.gelu(F.linear(w, torch.from_numpy(T["mlp0_w"]), torch.from_numpy(T["mlp0_b"])))
        return F.linear(h, torch.from_numpy(T["mlp2_w"]), torch.from_numpy(T["mlp2_b"])).numpy()


def head_out_units(raw):
    return {"presence_logit": raw[:, 0].astype(np.float64), "gap_m": raw[:, 1].astype(np.float64) * GAP_SCALE_M, "closing_ms": raw[:, 2].astype(np.float64) * CLOS_SCALE_MS}


def window_outputs(T, Z15, end=S0F):
    """Head outputs of the windows at the end frame `end` (S0 = 4): the head reads frames end-hist+1 .. end of the window array and nothing else."""
    h = T["hist"]
    if end - h + 1 < 0 or end >= Z15.shape[1]:
        raise ValueError("end frame %d outside the window" % end)
    return head_out_units(head_forward(T, Z15[:, end - h + 1:end + 1]))


def row_outputs(T, Z, idx5, chunk=20000):
    """Head outputs for every state row that has a history (idx5 [n, hist], -1 = none): per-frame embeddings once, then the read-out. NaN where no history."""
    import torch
    import torch.nn.functional as F
    n = len(Z)
    ok = (idx5 >= 0).all(1)
    out = np.full((n, 3), np.nan, np.float32)
    with torch.no_grad():
        E = np.empty((n, T["frame_w"].shape[0]), np.float32)
        W, b = torch.from_numpy(T["frame_w"]), torch.from_numpy(T["frame_b"])
        for i in range(0, n, chunk):
            E[i:i + chunk] = F.linear(torch.from_numpy(np.asarray(Z[i:i + chunk], np.float32)), W, b).numpy()
        rows = np.flatnonzero(ok)
        for i in range(0, len(rows), chunk):
            r = rows[i:i + chunk]
            w = torch.from_numpy(E[idx5[r]].reshape(len(r), -1))
            h = F.gelu(F.linear(w, torch.from_numpy(T["mlp0_w"]), torch.from_numpy(T["mlp0_b"])))
            out[r] = F.linear(h, torch.from_numpy(T["mlp2_w"]), torch.from_numpy(T["mlp2_b"])).numpy()
    return out, ok


# ============================================================================================ states -> windows
def state_key(st):
    names = np.asarray(st["clips"]).astype(str)
    return {(names[int(c)], int(t)): i for i, (c, t) in enumerate(zip(st["clip_idx"], st["t"]))}


def assemble_windows(Z, key, eid, g0, frames=15):
    """Z15 [n, 15, D] float16: the states of window frame f = bank step g0 - 4 + f of the eval state file; rows [n, 15] (-1 where the state file has no such frame;
    those frames are zero-filled and are only ever used as noise targets of control k4)."""
    n = len(eid)
    rows = np.full((n, frames), -1, np.int64)
    for j, (e, g) in enumerate(zip(eid, g0)):
        for f in range(frames):
            rows[j, f] = key.get((str(e), int(g) - S0F + f), -1)
    Z15 = np.zeros((n, frames, Z.shape[1]), Z.dtype)
    m = rows >= 0
    Z15[m] = Z[rows[m]]
    return Z15, rows


# ============================================================================================ packs (arm-independent truth)
def build_eval_pack(out_path):
    CT = O.CT
    L = np.load(CT + "/LATENTS_CTX_G_s0_R1.npz", allow_pickle=True)
    eid, g0, fold = L["eid"].astype(str), L["g0"].astype(np.int64), L["w_fold"].astype(np.int64)
    G15, miss = O.load_label_frames(eid, g0)
    S = np.load(O.V_ST, allow_pickle=True)
    key = {(str(e), int(t)): i for i, (e, t) in enumerate(zip(S["eid"].astype(str), S["t"]))}
    idx = np.array([key.get((str(e), int(g) + 11), -1) for e, g in zip(eid, g0)])
    assert (idx >= 0).all(), "windows without a speed row"
    V = np.asarray(S["V"], np.float64)[idx]
    assert np.isfinite(V).all()
    B = np.load(O.BANKP)
    bb, bg = B["beid"].astype(str), B["bg"].astype(np.int64)
    pos = {(e, int(g)): i for i, (e, g) in enumerate(zip(bb, bg))}
    pres0 = np.array([float(B["lead_present"][pos[(e, int(g))]]) if (e, int(g)) in pos else np.nan for e, g in zip(eid, g0)])
    ref = json.load(open(BANKED_OBJSLOT, encoding="utf-8"))["label"]["10"]["rules"]["R2"]["skill_over_copy"]["skill"]
    meta = {"sources": {"windows": CT + "/LATENTS_CTX_G_s0_R1.npz", "speed": O.V_ST, "bank": O.BANKP, "k1_reference": BANKED_OBJSLOT},
            "sha256": {"windows": sha256_file(CT + "/LATENTS_CTX_G_s0_R1.npz"), "speed": sha256_file(O.V_ST), "bank": sha256_file(O.BANKP)},
            "k1_reference_r2_skill": float(ref), "label_frames_without_bank_row": int(miss)}
    np.savez(out_path, eid=eid, g0=g0, fold=fold, V=V, G15=G15, pres0=pres0, bank_beid=bb, bank_bg=bg, bank_okE=np.isfinite(B["E"]).all(1),
             bank_gap=B["lead_gap"].astype(np.float64), bank_pres=B["lead_present"].astype(np.float64), meta=json.dumps(meta))
    return meta


def build_train_pack(out_path):
    if ROOT + "/spatial/leadhead_b" not in sys.path:
        sys.path.insert(0, ROOT + "/spatial/leadhead_b")
    import leadhead_b as HB
    M = HB.load_probe_meta()
    z = np.load(JOIN)
    jclip, off = z["clip"].astype(str), z["off"].astype(np.int64)
    lv = np.asarray(z["lead_valid"]).astype(np.int64)
    jc = {c: i for i, c in enumerate(jclip)}
    clips = list(M["uep"])
    ji = np.array([jc[c] for c in clips])
    jidx = off[ji[M["epi"]]] + M["bg"]
    pres = (lv[jidx] == 1).astype(np.float64)
    keep = M["keep"]
    assert bool(keep.all()), "rows with a non-finite kinematic vector exist: the probe-set layout differs from the one this pack assumes"
    meta = {"join": JOIN, "join_sha256": sha256_file(JOIN), "n_rows": int(len(pres)), "n_clips": len(clips), "gap_rule": "lead_valid == 1, finite, 0 <= gap <= 60 m (leadhead_b)",
            "ego_speed": "kin[:, 0] of the probe-set shards (leadhead_b M3)"}
    np.savez(out_path, clips=np.array(clips), clip_idx=M["epi"].astype(np.int64), t=M["bg"].astype(np.int64), gap=M["gap"].astype(np.float64), pres=pres,
             v=M["kin"][:, 0].astype(np.float64), meta=json.dumps(meta))
    return meta


def load_pack(path):
    z = np.load(path, allow_pickle=False)
    P = {k: z[k] for k in z.files if k != "meta"}
    P["meta"] = json.loads(str(z["meta"]))
    return P


# ============================================================================================ context of the windows (pure)
def window_ctx(P, B):
    """Everything arm-independent that the window reads need (same objects as leadhead_s0 / leadhead_b): label closing speed, masks, episodes, the bootstrap draw."""
    eid, g0, fold, G15, V = P["eid"].astype(str), P["g0"], P["fold"], P["G15"], P["V"]
    sc, cnt = O.scored_mask(G15, K)
    c_lab, hok = H.label_c(G15)
    valid = sc & hok
    ep10, n10 = O.epi_of(eid, sc)
    C10 = RP.boot_counts(n10, B, O.SEED)
    ep_all, n_all = O.epi_of(eid, np.ones(len(eid), bool))
    C_all = RP.boot_counts(n_all, B, O.SEED)
    return {"eid": eid, "g0": g0, "fold": fold, "G15": G15, "V": V, "sc": sc, "cnt": cnt, "c_lab": c_lab, "hok": hok, "valid": valid, "y": G15[:, S0F + K],
            "ep10": ep10, "n10": n10, "C10": C10, "ep_all": ep_all, "n_all": n_all, "C_all": C_all, "pres0": P["pres0"], "B": B}


def fid_one(y, p, valid, ep, n_ep, C):
    """R2 and RMSE of p against y on the valid rows with the episode-cluster interval (the estimator of leadhead_s0.fidelity, one target)."""
    lv, _ = O.level_row(y, p, valid, ep, n_ep, C)
    rm = H.rmse_row(y, p, valid, ep, n_ep, C)
    return {"r2": lv["r2"], "r2_ci95": lv["ci95"], "n": lv["n"], "rmse": rm["rmse"], "rmse_ci95": rm["ci95"]}


def boot_auc_chunked(ep, y, s, C, chunk=100):
    out = []
    for i in range(0, len(C), chunk):
        out.append(RP.boot_auc(ep, y, s, C[i:i + chunk]))
    return np.concatenate(out)


def auroc_row(y, s, ep, n_ep, C):
    m = np.isfinite(y) & np.isfinite(s) & (ep >= 0)
    pt = RP.auc_pooled(y, s)
    b = boot_auc_chunked(ep, y, s, C)
    return {"auroc": pt, "ci95": RP.ci(b), "n": int(m.sum()), "n_pos": int((y[m] == 1).sum()), "n_neg": int((y[m] == 0).sum())}


def banked_compare_file(PV, y, sc, ep, n_ep, C, eid, g0, path):
    """Level R2 of the rules of the head against the banked rollout read (the `p_i` of an OOF_RB_*_R1.npz), paired over episodes (leadhead_s0.banked_compare with a file path)."""
    bz = np.load(path, allow_pickle=True)
    if not (np.array_equal(bz["w_eid"].astype(str), eid) and np.array_equal(bz["w_g0"], g0)):
        raise SystemExit("REFUSED: banked rollout file %s lists other windows" % path)
    pi = bz["p_i"]
    out = {}
    for rr in ("R3", "R2", "R1"):
        pa = PV["pred"][rr]
        m2 = sc & np.isfinite(PV["pred"]["R3"]) & np.isfinite(pi) & np.isfinite(pa) & (ep >= 0)
        Sa = RP.ep_stats_r2(ep[m2], y[m2], pa[m2], n_ep)
        Sb = RP.ep_stats_r2(ep[m2], y[m2], pi[m2], n_ep)
        dd = RP.diff_row(RP.boot_r2(Sa, C), RP.boot_r2(Sb, C), RP.r2_pooled(y[m2], pa[m2]), RP.r2_pooled(y[m2], pi[m2]))
        dd["level_r2_rule"] = float(RP.r2_pooled(y[m2], pa[m2]))
        dd["level_r2_banked_rollout"] = float(RP.r2_pooled(y[m2], pi[m2]))
        dd["n"] = int(m2.sum())
        out[rr] = dd
    return out


def banked_path(arm, banked_dir):
    """The banked rollout read of the arm type: the arm's own file when it exists, else the base arm G of the same seed (the only banked rollout of this encoder and gate)."""
    own = "%s/OOF_RB_%s_R1.npz" % (banked_dir, arm)
    if os.path.exists(own):
        return own, arm
    seed = arm.rsplit("_", 1)[-1]
    g = "%s/OOF_RB_G_%s_R1.npz" % (banked_dir, seed)
    if os.path.exists(g):
        return g, "G_" + seed
    return None, None


def bias_corrected_copy(g0, y, sc, fold, nfold=5):
    """copy_bc = head gap at S0 + b, b = mean of (target - head gap at S0) over the scored windows of the OTHER folds (the folds R3 is cross-fitted on); never the scored fold itself."""
    out = np.full(len(g0), np.nan)
    fin = sc & np.isfinite(g0) & np.isfinite(y)
    for f in range(nfold):
        tr = fin & (fold != f)
        te = np.isfinite(g0) & (fold == f)
        if tr.sum() >= 1 and te.any():
            out[te] = g0[te] + float(np.mean((y - g0)[tr]))
    return out


def slot_skill_rows(ctx, PV, g_h):
    """R3 skill over the bias-corrected copy (GATED, decision M3-BC) and over the raw copy (the card's form, printed, names nothing)."""
    cbc = bias_corrected_copy(g_h, ctx["y"], ctx["sc"], ctx["fold"])
    m = ctx["sc"] & np.isfinite(PV["pred"]["R3"]) & np.isfinite(cbc)
    row, _ = O.skill_row(ctx["y"], PV["pred"]["R3"], cbc, m, ctx["ep10"], ctx["n10"], ctx["C10"])
    return row


def eval_read(ctx, out, probe_gap=None, banked=None):
    """M1..M4 of one arm on the windows from the head's outputs at S0 (dict presence_logit, gap_m, closing_ms of length n_windows)."""
    g_h, c_h, pl = out["gap_m"], out["closing_ms"], out["presence_logit"]
    G15, valid, ep, n, C = ctx["G15"], ctx["valid"], ctx["ep10"], ctx["n10"], ctx["C10"]
    R = {"M1": fid_one(ctx["c_lab"], c_h, valid, ep, n, C), "M2": {"head": fid_one(G15[:, S0F], g_h, valid, ep, n, C)}}
    if probe_gap is not None:
        R["M2"]["probe_one_frame"] = fid_one(G15[:, S0F], probe_gap, valid, ep, n, C)
        m = valid & np.isfinite(probe_gap)
        Sa = RP.ep_stats_r2(ep[m], G15[m, S0F], g_h[m], n)
        Sb = RP.ep_stats_r2(ep[m], G15[m, S0F], probe_gap[m], n)
        R["M2"]["head_minus_probe_r2"] = RP.diff_row(RP.boot_r2(Sa, C), RP.boot_r2(Sb, C), RP.r2_pooled(G15[m, S0F], g_h[m]), RP.r2_pooled(G15[m, S0F], probe_gap[m]))
    PV = H.head_pipeline(g_h, c_h, ctx["V"], K, ctx["y"], ctx["sc"], ctx["fold"])
    tb, _ = O.table(PV, ep, n, C)
    R["M3"] = {"rules": tb, "coefs": PV["coefs"], "n_scored": int(ctx["sc"].sum())}
    R["M3"]["rules"]["R3"]["skill_over_copy_bc"] = slot_skill_rows(ctx, PV, g_h)
    sc_ = ctx["sc"] & np.isfinite(g_h) & np.isfinite(ctx["G15"][:, S0F])
    R["M3"]["gap_offset_m"] = float(np.mean(g_h[sc_] - ctx["G15"][sc_, S0F]))
    R["M3"]["banked"] = banked_compare_file(PV, ctx["y"], ctx["sc"], ep, n, C, ctx["eid"], ctx["g0"], banked) if banked else None
    R["M4"] = auroc_row(ctx["pres0"], pl, ctx["ep_all"], ctx["n_all"], ctx["C_all"])
    return R, PV


# ============================================================================================ train probe read
def train_read(T, Ztr, pack, B, chunk=100):
    """M1..M4 of the head applied to the train probe clips (held in for the model and the head). Labels as leadhead_b builds them (H.build_fit_sets on the probe-set layout)."""
    clips = np.asarray(pack["clips"]).astype(str)
    names = clips[pack["clip_idx"]]
    bg = pack["t"].astype(np.int64)
    FS = H.build_fit_sets(names, bg, pack["gap"], pack["v"])
    idx5 = FS["idx5"]
    raw, ok5 = row_outputs(T, Ztr, idx5)
    out = head_out_units(raw)
    ep = pack["clip_idx"].astype(np.int64)
    n_ep = len(clips)
    C = RP.boot_counts(n_ep, B, O.SEED)
    R = {"M1": fid_one(FS["c"], out["closing_ms"], FS["c_ok"], ep, n_ep, C), "M2": {"head": fid_one(FS["gap"], out["gap_m"], FS["gap_ok"], ep, n_ep, C)}}
    # M3: rows with the 5-frame history and a gap label on every frame t .. t+10 without a jump (dir_ok); ego speeds t .. t+10 from the probe-set kinematics
    idxF = H.history_index(names, bg, range(-4, 11))
    Vw = H.gather(pack["v"], idxF)
    scr = FS["dir_ok"] & np.isfinite(Vw[:, S0F:]).all(1)
    y = np.where(scr, FS["gap"] + FS["delta"], np.nan)
    fold = RP.make_folds(names)
    ep3, n3 = O.epi_of(names, scr)
    C3 = RP.boot_counts(n3, B, O.SEED)
    Vw = np.nan_to_num(Vw, nan=0.0)
    PV = H.head_pipeline(out["gap_m"], out["closing_ms"], Vw, K, y, scr, fold)
    tb, _ = O.table(PV, ep3, n3, C3)
    R["M3"] = {"rules": tb, "coefs": PV["coefs"], "n_scored": int(scr.sum()), "n_clips_scored": int(n3), "banked": None}
    cbc = bias_corrected_copy(out["gap_m"], y, scr, fold)
    mm = scr & np.isfinite(PV["pred"]["R3"]) & np.isfinite(cbc)
    R["M3"]["rules"]["R3"]["skill_over_copy_bc"] = O.skill_row(y, PV["pred"]["R3"], cbc, mm, ep3, n3, C3)[0]
    R["M3"]["gap_offset_m"] = float(np.mean(out["gap_m"][scr] - FS["gap"][scr]))
    pres = np.where(ok5, pack["pres"], np.nan)
    R["M4"] = auroc_row(pres, np.where(ok5, out["presence_logit"], np.nan), ep, n_ep, C)
    R["counts"] = {"rows": int(len(names)), "rows_with_history": int(ok5.sum()), "gap_rows": int(FS["gap_ok"].sum()), "closing_rows": int(FS["c_ok"].sum()),
                   "scored_rows_M3": int(scr.sum()), "clips": int(n_ep)}
    return R, out, ok5


# ============================================================================================ controls
def control_k1(ctx, ref):
    c = H.control_h6(ctx["G15"], ctx["V"], ctx["fold"], ref)
    c["card_value"] = 0.750
    c["pass"] = bool(c["pass"] and abs(c["r2_skill_this_path"] - 0.750) <= 5e-4)
    return c


def control_k2(T, Ztr_rows, idx5, pod, rows_S0=None):
    """The head applied here to the pulled states equals the pod's outputs. Compared on EVERY row with a history; the windows' S0 rows are counted apart (card: at least 64 windows)."""
    raw, ok = row_outputs(T, Ztr_rows, idx5)
    dev = head_out_units(np.where(np.isfinite(raw), raw, np.nan))
    same_mask = bool(np.array_equal(ok, np.asarray(pod["has_history"], bool)))
    both = ok & np.isfinite(pod["gap_m"]) & np.isfinite(pod["closing_ms"])
    dg = float(np.nanmax(np.abs(dev["gap_m"][both] - pod["gap_m"][both]))) if both.any() else float("nan")
    dc = float(np.nanmax(np.abs(dev["closing_ms"][both] - pod["closing_ms"][both]))) if both.any() else float("nan")
    dp = float(np.nanmax(np.abs(dev["presence_logit"][both] - pod["presence_logit"][both]))) if both.any() else float("nan")
    if rows_S0 is not None:
        r0 = np.asarray(rows_S0)
        nw = int(both[r0[r0 >= 0]].sum())
    else:
        nw = int(both.sum())
    ok_all = bool(same_mask and both.sum() > 0 and dg <= TOL_K2 and dc <= TOL_K2 and nw >= 64)
    return {"pass": ok_all, "max_abs_gap_m": dg, "max_abs_closing_ms": dc, "max_abs_presence_logit": dp, "tol": TOL_K2, "n_rows_compared": int(both.sum()),
            "n_windows_compared": nw, "has_history_equal": same_mask}


def scored_permutation(ctx, seed):
    """Identity outside the scored windows; the states of the scored windows permuted among themselves (the windows the rule path reads)."""
    n = len(ctx["sc"])
    perm = np.arange(n)
    idx = np.flatnonzero(ctx["sc"])
    perm[idx] = idx[np.random.default_rng(seed).permutation(len(idx))]
    return perm


def _shuffle_numbers(T, Z15, ctx, perm):
    out = window_outputs(T, Z15[perm])
    Rr, _ = eval_read(ctx, out)
    v = ctx["valid"]
    return {"closing_r2": Rr["M1"]["r2"], "r3_skill": Rr["M3"]["rules"]["R3"]["skill_over_copy_bc"]["skill"], "r3_skill_raw_copy": Rr["M3"]["rules"]["R3"]["skill_over_copy"]["skill"],
            "gap_mean_minus_label_mean_m": float(np.nanmean(out["gap_m"][v]) - np.nanmean(ctx["G15"][v, S0F]))}


def control_k3(T, Z15, ctx, perm=None, seed=0):
    """States permuted across windows (deliberate-regression arm of the read): closing-speed R2 and rule R3 skill at most +0.02.
    The permutation is taken among the scored windows (decision K3, LEADSUP_READ_DEFS.md): a permutation over all windows moves the states of windows without a lead onto scored
    windows, and the intercept of R3 then removes the resulting offset of the gap, which is a property of R3 and not a leak. The permutation over all windows is reported beside, not gated."""
    n = len(Z15)
    if perm is None:
        perm = scored_permutation(ctx, seed)
    g = _shuffle_numbers(T, Z15, ctx, perm)
    allp = _shuffle_numbers(T, Z15, ctx, np.random.default_rng(seed).permutation(n))
    extra = []
    for sd in (seed + 1, seed + 2, seed + 3):                           # information only: the spread of the R3 skill over other scored-window permutations
        o2 = window_outputs(T, Z15[scored_permutation(ctx, sd)])
        P2 = H.head_pipeline(o2["gap_m"], o2["closing_ms"], ctx["V"], K, ctx["y"], ctx["sc"], ctx["fold"])
        cb2 = bias_corrected_copy(o2["gap_m"], ctx["y"], ctx["sc"], ctx["fold"])
        extra.append(O.skill_point(ctx["y"], P2["pred"]["R3"], cb2, ctx["sc"] & np.isfinite(P2["pred"]["R3"]) & np.isfinite(cb2)))
    return {"pass": bool(g["closing_r2"] <= BAR_K3 and g["r3_skill"] <= BAR_K3), "closing_r2_shuffled": g["closing_r2"], "r3_skill_shuffled": g["r3_skill"], "r3_skill_shuffled_raw_copy_info_only": g["r3_skill_raw_copy"], "bar": BAR_K3,
            "n_windows_permuted": int((perm != np.arange(n)).sum()), "permutation_is_identity": bool(np.array_equal(perm, np.arange(n))), "seed": seed,
            "shuffled_head_gap_mean_minus_label_gap_mean_m": g["gap_mean_minus_label_mean_m"], "info_only_r3_skill_other_permutations": extra,
            "info_only_permutation_over_all_windows": allp}


def control_k4(T, Z15, ctx, end=S0F, seed=3):
    """Every state after S0 replaced by noise: no head output and no rule output changes."""
    base = window_outputs(T, Z15, end=S0F)
    Zm = Z15.copy()
    rng = np.random.default_rng(seed)
    Zm[:, S0F + 1:, :] = (rng.standard_normal(Zm[:, S0F + 1:, :].shape) * 5).astype(Zm.dtype)
    out = window_outputs(T, Zm, end=end)                               # end != S0F is the deliberate regression: a head that peeks at a later frame
    dh = max(float(np.max(np.abs(out[k] - base[k]))) for k in ("presence_logit", "gap_m", "closing_ms"))
    P0 = H.head_pipeline(base["gap_m"], base["closing_ms"], ctx["V"], K, ctx["y"], ctx["sc"], ctx["fold"])
    P1 = H.head_pipeline(out["gap_m"], out["closing_ms"], ctx["V"], K, ctx["y"], ctx["sc"], ctx["fold"])
    dr = max(float(np.nanmax(np.abs(P0["pred"][r] - P1["pred"][r]))) for r in ("R1", "R2", "R3"))
    return {"pass": bool(dh == 0.0 and dr == 0.0), "max_abs_change_of_head_outputs": dh, "max_abs_change_of_rule_outputs": dr, "end_frame_read": int(end)}


# ============================================================================================ outcomes (card section 5)
def _sep_pos(row):
    return bool(row["ci95"][0] > 0)


def closing_name(rows):
    """rows: M1 dicts of both seeds. SUP-READS-CLOSING if R2 above 0 with an interval that excludes 0 at both seeds."""
    if len(rows) < 2:
        return "not named (needs both seeds)"
    return "SUP-READS-CLOSING" if all(r["r2"] > 0 and r["r2_ci95"][0] > 0 for r in rows) else "SUP-CLOSING-FLAT"


def slot_names(m3rows):
    """m3rows: M3 dicts of both seeds. SUP-SLOT-IMAGINES if the R3 skill over the head's copy is above 0 with an interval excluding 0 at both seeds; plus
    SUP-SLOT-BEATS-ROLLOUT if the level R2 minus the banked rollout read is above 0 with an interval excluding 0 at both seeds (when a banked read exists)."""
    if len(m3rows) < 2:
        return ["not named (needs both seeds)"]
    im = all(r["rules"]["R3"]["skill_over_copy_bc"]["separated_pos"] for r in m3rows)
    if not im:
        return ["SUP-SLOT-FLAT"]
    names = ["SUP-SLOT-IMAGINES"]
    if all(r.get("banked") for r in m3rows):
        if all(r["banked"]["R3"]["separated_pos"] for r in m3rows):
            names.append("SUP-SLOT-BEATS-ROLLOUT")
    else:
        names.append("(SUP-SLOT-BEATS-ROLLOUT not evaluable: no banked rollout read for this arm type)")
    return names


def gradient_name(d0, d1, floor):
    """d0, d1: paired M1 R2 difference rows (diff, ci95, separated_pos / separated_neg) of the two seeds, a minus the detached arm; floor = |M1(LSD_s0) - M1(LSD_s1)|."""
    mean = 0.5 * (d0["diff"] + d1["diff"])
    if d0["diff"] > 0 and d1["diff"] > 0 and d0["separated_pos"] and d1["separated_pos"] and mean > floor:
        return "SUP-GRADIENT-HELPS"
    if d0["diff"] < 0 and d1["diff"] < 0 and d0["separated_neg"] and d1["separated_neg"] and -mean > floor:
        return "SUP-GRADIENT-HURTS"
    return "SUP-GRADIENT-EQUAL"


def reading_with_earlier(cl):
    """Card section 5, last bullet, from this card's own closing names per arm type (the precondition 'flat in cards ADD21 and ADD23' is the lead's to confirm)."""
    d, ls, lh = cl.get("LSD"), cl.get("LS"), cl.get("LSH")
    if None in (d, ls, lh) or any(str(x).startswith("not named") for x in (d, ls, lh)):
        return "not named (needs all three arm types with both seeds)"
    if d == "SUP-READS-CLOSING":
        return "SUP-READS-CLOSING at LSD: the state of G held the motion and the earlier heads were the limit (if cards ADD21 and ADD23 were flat; the lead confirms that precondition)"
    if ls == "SUP-READS-CLOSING" or lh == "SUP-READS-CLOSING":
        return "flat at LSD and reading at LS or LSH: the state holds the motion only when asked during training"
    return "flat in all three: the pooled 4 by 10 grid of frozen camera tokens does not carry the motion of the lead at this precision"


def pair_m1(ctx, ca, cb):
    """Paired difference of the M1 R2 of two arms (a minus b) on the same windows and the same episode draw."""
    m = ctx["valid"] & np.isfinite(ca) & np.isfinite(cb) & np.isfinite(ctx["c_lab"]) & (ctx["ep10"] >= 0)
    y = ctx["c_lab"]
    ep, n, C = ctx["ep10"], ctx["n10"], ctx["C10"]
    Sa = RP.ep_stats_r2(ep[m], y[m], ca[m], n)
    Sb = RP.ep_stats_r2(ep[m], y[m], cb[m], n)
    d = RP.diff_row(RP.boot_r2(Sa, C), RP.boot_r2(Sb, C), RP.r2_pooled(y[m], ca[m]), RP.r2_pooled(y[m], cb[m]))
    d["n"] = int(m.sum())
    return d


def name_outcomes(per_arm, paired, floor_by_seed_pair):
    """per_arm[arm] = eval read dict; paired[(a, b, seed)] = pair_m1 row; floor = |M1(LSD_s0) - M1(LSD_s1)|. Only arm types with both seeds are named."""
    o = {"named": True, "per_type": {}, "gradient": {}}
    for t in TYPES:
        seeds = [a for a in ("%s_s0" % t, "%s_s1" % t) if a in per_arm]
        if len(seeds) < 2:
            o["per_type"][t] = {"closing": "not named (needs both seeds; have %s)" % seeds, "slot": ["not named (needs both seeds; have %s)" % seeds]}
            continue
        o["per_type"][t] = {"closing": closing_name([per_arm[a]["M1"] for a in seeds]), "slot": slot_names([per_arm[a]["M3"] for a in seeds])}
    for t in ("LS", "LSH"):
        k0, k1 = (t, "LSD", "s0"), (t, "LSD", "s1")
        if k0 in paired and k1 in paired and floor_by_seed_pair is not None:
            o["gradient"]["%s against LSD" % t] = gradient_name(paired[k0], paired[k1], floor_by_seed_pair)
        else:
            o["gradient"]["%s against LSD" % t] = "not named (needs both seeds of both arm types)"
    o["reading_with_earlier_cards"] = reading_with_earlier({t: o["per_type"][t]["closing"] for t in TYPES})
    o["cost"] = "named by the lead's T1-kin cost read (q3_read.py, controls k5..k7), not by this reader"
    return o


# ============================================================================================ inputs of one arm
def arm_files(in_dir, arm, states_dir=None, alias=None):
    sd = states_dir or in_dir
    sname = (alias or {}).get(arm, arm)
    f = {"eval_states": "%s/%s_eval.npz" % (sd, sname), "train_states": "%s/%s_train.npz" % (sd, sname)}
    for sp in ("eval", "train"):
        f[sp + "_lead"] = "%s/%s_%s_lead.npz" % (in_dir, arm, sp)
    return f


def check_records(files, plumbing=False, do_train=True):
    """Every file against the sha256 of its record: states vs out_sha256 of `.npz.json`; lead outputs and head tensors vs `<out>.json`. Returns (control dict, ckpt shas)."""
    rows, shas = {}, {}
    ok = True
    names = ("eval_states", "train_states", "eval_lead", "train_lead") if do_train else ("eval_states", "eval_lead")
    for nm in names:
        p = files[nm]
        rp = p + ".json"
        if not os.path.exists(p) or not os.path.exists(rp):
            rows[nm] = "MISSING file or record"
            ok = False
            continue
        rec = json.load(open(rp, encoding="utf-8"))
        want = rec.get("out_sha256")
        got = sha256_file(p)
        good = bool(want == got)
        row = {"sha256": got, "record_sha256": want, "match": good}
        if nm.endswith("_lead"):
            hp = p + ".head.npz"
            hg = sha256_file(hp) if os.path.exists(hp) else None
            row["head_sha256"] = hg
            row["head_match"] = bool(hg is not None and hg == rec.get("head_sha256"))
            good = good and row["head_match"]
            row["states_sha256_in_record"] = rec.get("states_sha256")
        shas[nm] = rec.get("ckpt_sha256")
        row["ckpt_sha256"] = rec.get("ckpt_sha256")
        row["pass"] = good
        ok = ok and good
        rows[nm] = row
    cks = {v for v in shas.values() if v}
    lead_ck = shas.get("eval_lead")
    rows["ckpt_sha256_identical_across_the_four_files"] = bool(len(cks) == 1 and all(shas.get(k) for k in names))
    if plumbing:
        rows["ckpt_identity_not_required_in_a_plumbing_run"] = True
    else:
        ok = ok and rows["ckpt_sha256_identical_across_the_four_files"]
    # the state file the pod applied the head to is the one pulled here
    for sp in (("eval", "train") if do_train else ("eval",)):
        r = rows.get(sp + "_lead")
        s = rows.get(sp + "_states")
        if isinstance(r, dict) and isinstance(s, dict):
            r["applied_to_pulled_states"] = bool(r.get("states_sha256_in_record") == s.get("sha256"))
            ok = ok and r["applied_to_pulled_states"]
    rows["pass"] = bool(ok)
    return rows, (list(cks)[0] if len(cks) == 1 else lead_ck)


def load_states(path):
    z = np.load(path, allow_pickle=False)
    return {"Z": z["Z"], "clip_idx": z["clip_idx"], "t": z["t"], "clips": z["clips"]}


def probe_reference(st, P, ctx, fit_fn=None):
    """The one-frame probe of the same arm (the level probe of the r-b read, ADD20): per fold fitted on the eval-bank rows of the other folds, applied to the S0 state."""
    if fit_fn is None:
        import s0_read as S0
        fit_fn = S0.fit_probe
        seed0 = S0.SEED
    else:
        seed0 = 0
    key = state_key(st)
    bb, bg = P["bank_beid"].astype(str), P["bank_bg"].astype(np.int64)
    okE = P["bank_okE"]
    idx = np.array([key.get((e, int(g)), -1) for e, g in zip(bb, bg)])
    rows = np.flatnonzero(okE & (idx >= 0))
    beid = bb[rows]
    Sb = st["Z"][idx[rows]].astype(np.float64)
    gap = P["bank_gap"][rows]
    uep = sorted(set(beid))
    ui = {e: i for i, e in enumerate(uep)}
    epi = np.array([ui[e] for e in beid])
    folds_b = RP.make_folds(beid)
    VG = np.full(len(ctx["eid"]), np.nan)
    s0rows = np.array([key.get((str(e), int(g)), -1) for e, g in zip(ctx["eid"], ctx["g0"])])
    Z0 = st["Z"][np.where(s0rows >= 0, s0rows, 0)].astype(np.float64)
    for f in range(5):
        tr = np.flatnonzero(folds_b != f)
        vtr = tr[np.isfinite(gap[tr])]
        fit = fit_fn(Sb[vtr], gap[vtr], epi[vtr], seed0 + f)
        te = np.flatnonzero((ctx["fold"] == f) & (s0rows >= 0))
        if len(te):
            VG[te] = fit(Z0[te])
    return VG


# ============================================================================================ run
def run(arms, in_dir, out_dir, B, pack_dir, states_dir=None, alias=None, banked_dir=BANKED_RB_DIR, do_train=True, do_probe=True, plumbing=False, probe_fit=None):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    P = load_pack(pack_dir + "/eval_pack.npz")
    PT = load_pack(pack_dir + "/train_pack.npz") if do_train else None
    ctx = window_ctx(P, B)
    res = {"tier": TIER, "families": FAMILIES, "card": "PREREG_V8F_PROOF_ADD24 section 3a", "B": B, "plumbing": bool(plumbing), "arms": {}, "controls": {}, "paired": {},
           "pack_meta": P["meta"], "counts": {"windows": int(len(ctx["eid"])), "scored": int(ctx["sc"].sum()), "label_history_ok": int(ctx["valid"].sum()),
                                              "episodes_scored": int(ctx["n10"]), "episodes_all": int(ctx["n_all"]), "scored_detail": ctx["cnt"]}}
    ctrl = {}
    ctrl["k1_label_path"] = control_k1(ctx, P["meta"]["k1_reference_r2_skill"])
    ctrl["a2_pack_counts"] = {"pass": bool(ctx["cnt"]["windows"] == EXPECT["windows"] and ctx["cnt"]["scored"] == EXPECT["scored"] and ctx["n10"] == EXPECT["episodes_scored"]),
                              "expected": {k: EXPECT[k] for k in ("windows", "scored", "episodes_scored")}, "got": {"windows": ctx["cnt"]["windows"], "scored": ctx["cnt"]["scored"], "episodes_scored": ctx["n10"]}}
    outs = {}
    for arm in arms:
        log("arm", arm)
        f = arm_files(in_dir, arm, states_dir, alias)
        recs, ck = check_records(f, plumbing, do_train)
        ctrl["a1_records|%s" % arm] = recs
        if not recs["pass"]:
            log(arm, "input records failed; arm not read")
            res["arms"][arm] = {"files": f, "ckpt_sha256": ck, "read": False}
            continue
        Th_e = load_head(f["eval_lead"] + ".head.npz")
        if do_train:
            Th_t = load_head(f["train_lead"] + ".head.npz")
            same_head = all(np.array_equal(Th_e[k], Th_t[k]) for k in HEAD_KEYS) and Th_e["hist"] == Th_t["hist"]
            ctrl["a3_same_head_eval_train|%s" % arm] = {"pass": bool(same_head)}
        st = load_states(f["eval_states"])
        key = state_key(st)
        Z15, rows = assemble_windows(st["Z"], key, ctx["eid"], ctx["g0"])
        missing_hist = int((rows[:, :S0F + 1] < 0).sum())
        finite = bool(np.isfinite(Z15[:, :S0F + 1].astype(np.float32)).all())
        ctrl["a4_window_states|%s" % arm] = {"pass": bool(missing_hist == 0 and finite), "window_frames_0to4_without_state": missing_hist, "window_states_0to4_finite": finite,
                                             "future_frames_without_state": int((rows[:, S0F + 1:] < 0).sum())}
        idx_e, _ = _history_idx(st, Th_e["hist"])
        ze = np.load(f["eval_lead"], allow_pickle=False)
        pod_e = {k: ze[k] for k in ("presence_logit", "gap_m", "closing_ms", "has_history")}
        ctrl["k2_head_forward_eval|%s" % arm] = control_k2(Th_e, st["Z"], idx_e, pod_e, rows[:, S0F])
        out = window_outputs(Th_e, Z15)
        banked, bsrc = banked_path(arm, banked_dir)
        probe = probe_reference(st, P, ctx, probe_fit) if do_probe else None
        R, PV = eval_read(ctx, out, probe, banked)
        R["banked_source"] = bsrc
        R["ckpt_sha256"] = ck
        R["files"] = f
        R["read"] = True
        ctrl["k3_shuffled_states|%s" % arm] = control_k3(Th_e, Z15, ctx)
        ctrl["k4_no_future_input|%s" % arm] = control_k4(Th_e, Z15, ctx)
        if do_train:
            stt = load_states(f["train_states"])
            lay = {"clips_equal": bool(list(map(str, stt["clips"])) == list(map(str, PT["clips"]))), "clip_idx_equal": bool(np.array_equal(stt["clip_idx"].astype(np.int64), PT["clip_idx"])),
                   "t_equal": bool(np.array_equal(stt["t"].astype(np.int64), PT["t"])), "n_rows": int(len(stt["Z"]))}
            lay["pass"] = bool(lay["clips_equal"] and lay["clip_idx_equal"] and lay["t_equal"])
            ctrl["a5_train_layout|%s" % arm] = lay
            if lay["pass"]:
                idx_t, _ = _history_idx(stt, Th_t["hist"])
                zt = np.load(f["train_lead"], allow_pickle=False)
                pod_t = {k: zt[k] for k in ("presence_logit", "gap_m", "closing_ms", "has_history")}
                rows_ok = np.flatnonzero((idx_t >= 0).all(1))
                ctrl["k2_head_forward_train|%s" % arm] = control_k2(Th_t, stt["Z"], idx_t, pod_t, rows_ok)
                Rt, _, _ = train_read(Th_t, stt["Z"], PT, B)
                R["train"] = Rt
            del stt
        res["arms"][arm] = R
        outs[arm] = out
        del st, Z15
    # paired M1 differences between arm types of the same seed, and the floor from the two LSD seeds
    for s in ("s0", "s1"):
        for a, b in (("LS", "LSD"), ("LSH", "LSD"), ("LSH", "LS")):
            ka, kb = "%s_%s" % (a, s), "%s_%s" % (b, s)
            if ka in outs and kb in outs:
                res["paired"]["%s minus %s|%s" % (a, b, s)] = pair_m1(ctx, outs[ka]["closing_ms"], outs[kb]["closing_ms"])
    floor = None
    if "LSD_s0" in res["arms"] and "LSD_s1" in res["arms"] and res["arms"]["LSD_s0"].get("read") and res["arms"]["LSD_s1"].get("read"):
        floor = abs(res["arms"]["LSD_s0"]["M1"]["r2"] - res["arms"]["LSD_s1"]["M1"]["r2"])
    res["floor_lsd_seeds_abs_diff_m1_r2"] = floor
    res["controls"] = ctrl
    flat = _flatten_controls(ctrl)
    res["controls_pass"] = bool(all(v for v in flat.values()))
    res["controls_failed"] = [k for k, v in flat.items() if not v]
    all_read = all(res["arms"].get(a, {}).get("read") for a in arms)
    pairs = {}
    for k, v in res["paired"].items():
        lhs, seed = k.split("|")
        a, b = [x.strip() for x in lhs.split("minus")]
        pairs[(a, b, seed)] = v
    if plumbing:
        res["outcomes"] = {"named": False, "reason": "plumbing run: a random head stands in for an arm; no number of this run is a result"}
    elif not res["controls_pass"]:
        res["outcomes"] = {"named": False, "reason": "a control failed (%s); no outcome is named" % ", ".join(res["controls_failed"])}
    elif not all_read:
        res["outcomes"] = {"named": False, "reason": "an arm was not read"}
    else:
        res["outcomes"] = name_outcomes({a: res["arms"][a] for a in arms}, pairs, floor)
    res["elapsed_s"] = round(time.time() - t0, 1)
    jdump(res, os.path.join(out_dir, "RESULT_LEADSUP_T0.json"))
    write_verdict(os.path.join(out_dir, "VERDICT_LEADSUP_T0.md"), res, arms)
    log("controls", "PASS" if res["controls_pass"] else "FAILED %s" % res["controls_failed"], "elapsed %.0f s" % res["elapsed_s"])
    return 0 if res["controls_pass"] else 2


def _history_idx(st, hist):
    key = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(st["clip_idx"], st["t"]))}
    n = len(st["t"])
    idx = np.full((n, hist), -1, np.int64)
    for j in range(hist):
        off = hist - 1 - j
        idx[:, j] = [key.get((int(c), int(t) - off), -1) for c, t in zip(st["clip_idx"], st["t"])]
    ok = (idx >= 0).all(1)
    idx[~ok] = -1
    return idx, ok


def _flatten_controls(ctrl):
    """name -> bool for every control and every sub-check that carries a pass flag."""
    out = {}
    for k, v in ctrl.items():
        if isinstance(v, dict) and "pass" in v:
            out[k] = bool(v["pass"])
        elif isinstance(v, dict):
            out[k] = all(bool(x["pass"]) for x in v.values() if isinstance(x, dict) and "pass" in x)
    return out


# ============================================================================================ writing
def _ci(c):
    return "[%+.3f, %+.3f]" % (c[0], c[1])


def _row(r, unit=""):
    return "R2 %+.3f %s; RMSE %.2f%s %s; n=%d" % (r["r2"], _ci(r["r2_ci95"]), r["rmse"], unit, _ci(r["rmse_ci95"]), r["n"])


def _sk(row):
    return "%+.3f %s n=%d%s" % (row["skill"], _ci(row["ci95"]), row["n"], " (separated above 0)" if row["separated_pos"] else (" (separated below 0)" if row.get("separated_neg") else ""))


def _block(R, label):
    L = []
    L.append("- M1 closing speed at S0 (%s): %s" % (label, _row(R["M1"], " m/s")))
    L.append("- M2 gap at S0, head: %s" % _row(R["M2"]["head"], " m"))
    if "probe_one_frame" in R["M2"]:
        d = R["M2"]["head_minus_probe_r2"]
        L.append("- M2 gap at S0, one-frame probe of the same arm: %s; head minus probe in R2 %+.3f %s" % (_row(R["M2"]["probe_one_frame"], " m"), d["diff"], _ci(d["ci95"])))
    for r in ("R1", "R2", "R3"):
        row = R["M3"]["rules"][r]
        if r == "R3":
            L.append("- M3 R3 skill over the BIAS-CORRECTED copy of the same head (gated; names the slot): %s" % _sk(row["skill_over_copy_bc"]))
        L.append("- M3 %s: skill over the raw copy of the same head (card form, names nothing) %s; level R2 %+.3f %s" % (r, _sk(row["skill_over_copy"]), row["level_r2"]["r2"], _ci(row["level_r2"]["ci95"])))
    L.append("- M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): %+.2f m" % R["M3"]["gap_offset_m"]) if "gap_offset_m" in R["M3"] else None
    L.append("- M3 R0 (head copy): level R2 %+.3f %s (n scored %d)" % (R["M3"]["rules"]["R0"]["level_r2"]["r2"], _ci(R["M3"]["rules"]["R0"]["level_r2"]["ci95"]), R["M3"]["n_scored"]))
    b = R["M3"].get("banked")
    if b:
        d = b["R3"]
        L.append("- M3 R3 level R2 %+.3f minus banked rollout read %+.3f: %+.3f %s%s (n=%d)" % (d["level_r2_rule"], d["level_r2_banked_rollout"], d["diff"], _ci(d["ci95"]),
                 " (separated above 0)" if d["separated_pos"] else (" (separated below 0)" if d["separated_neg"] else ""), d["n"]))
    else:
        L.append("- M3 difference to the banked rollout read: not available (no banked rollout file for this arm type)")
    m4 = R["M4"]
    L.append("- M4 presence AUROC at S0: %.3f %s; n=%d (positives %d, negatives %d); names nothing" % (m4["auroc"], _ci(m4["ci95"]), m4["n"], m4["n_pos"], m4["n_neg"]))
    return L


def write_verdict(path, res, arms):
    L = ["# VERDICT LEADSUP T0 (card PREREG_V8F_PROOF_ADD24, section 3a)", ""]
    if res["plumbing"]:
        L += ["**PLUMBING RUN: a random head stands in for an arm (tag PLUMB). It proves that the window assembly, labels, estimator and controls execute on the real layout. No number below is a result.**", ""]
    L += ["Tier " + TIER + ". " + FAMILIES + ".", "Evidence class: MEASURED once the arms exist. Estimator: episode-cluster bootstrap, B %d, rng 0 (episodes of the scored windows; clips on the train probe set)." % res["B"],
          "Definitions and every reader decision: LEADSUP_READ_DEFS.md.", "", "## Controls (first)", ""]
    for k, v in res["controls"].items():
        if isinstance(v, dict) and "pass" in v:
            L.append("- %s: %s :: %s" % (k, "PASS" if v["pass"] else "FAIL", json.dumps({a: b for a, b in v.items() if a != "pass"}, default=str)))
        else:
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    L.append("- %s / %s: %s :: %s" % (k, kk, "PASS" if vv.get("pass") else "FAIL", json.dumps({a: b for a, b in vv.items() if a != "pass"}, default=str)))
                else:
                    L.append("- %s / %s: %s" % (k, kk, vv))
    L += ["- k5 (trained, not only built), k6 (checkpoint identity with the T1-kin dump), k7 (q3_read controls): the lead's, not run here.",
          "- controls_pass: %s" % res["controls_pass"], ""]
    if not res["controls_pass"]:
        L += ["**A control failed (%s). No outcome is named.**" % ", ".join(res["controls_failed"]), ""]
    c = res["counts"]
    L += ["## Names (card section 5; both seeds needed)", ""]
    o = res["outcomes"]
    if not o.get("named"):
        L.append("No outcome is named: %s." % o["reason"])
    else:
        for t in TYPES:
            d = o["per_type"][t]
            L.append("- Arm type %s: closing **%s**; slot **%s**" % (t, d["closing"], " + ".join(d["slot"])))
        for k, v in o["gradient"].items():
            L.append("- Gradient question, %s (paired over episodes, floor |M1(LSD_s0) - M1(LSD_s1)| = %s): **%s**" % (k, "%.3f" % res["floor_lsd_seeds_abs_diff_m1_r2"] if res["floor_lsd_seeds_abs_diff_m1_r2"] is not None else "n/a", v))
        L.append("- Reading with the earlier cards: %s" % o["reading_with_earlier_cards"])
        L.append("- Cost: %s" % o["cost"])
    L += ["", "Four families: LONGITUDINAL, LATERAL, TACTICAL not measured by this read; STRATEGIC: PAUSED (PI item 25).", "",
          "## Windows", "", "Windows %d; scored %d (episodes %d); with the label history rule %d; episodes over all windows %d." % (c["windows"], c["scored"], c["episodes_scored"], c["label_history_ok"], c["episodes_all"]), ""]
    L += ["## Per arm, eval windows (T0)", ""]
    for arm in arms:
        R = res["arms"].get(arm, {})
        L += ["### " + arm, ""]
        if not R.get("read"):
            L += ["Not read (input records failed or files missing).", ""]
            continue
        L += ["checkpoint sha256 %s" % R["ckpt_sha256"], ""] + _block(R, "label closing speed") + [""]
    if any("train" in res["arms"].get(a, {}) for a in arms):
        L += ["## Per arm, train probe clips (held in for the model and the head; reported only, names nothing)", ""]
        for arm in arms:
            R = res["arms"].get(arm, {}).get("train")
            if not R:
                continue
            L += ["### " + arm, "", "rows %(rows)d, with a 5-frame history %(rows_with_history)d, gap rows %(gap_rows)d, closing rows %(closing_rows)d, scored rows (M3) %(scored_rows_M3)d, clips %(clips)d" % R["counts"], ""] + _block(R, "train labels") + [""]
    L += ["## Paired differences of M1 (R2, episode-cluster bootstrap of the difference; floor above)", "", "| pair | seed | diff | 95% interval | n |", "|---|---|---|---|---|"]
    for k, v in res["paired"].items():
        lhs, seed = k.split("|")
        L.append("| %s | %s | %+.3f | %s | %d |" % (lhs, seed, v["diff"], _ci(v["ci95"]), v["n"]))
    L += ["", "## Limits (card section 7 and the reader)", "",
          "- Two training runs per arm type: a name from two seeds is stated with that limit; the interval answers the question about episodes, not about training runs. The gradient question carries its floor from the two LSD seeds.",
          "- One corpus, one evaluation set: 429 scored windows of 48 episodes. Labels come from the offline obstacle tracks; the eval presence label is the bank lead_present, the train presence label is lead_valid (LEADSUP_READ_DEFS.md).",
          "- The states are the float16 state files of the extraction, not the float32 states the head saw in training; the pod and this box apply the head to the same float16 states (control k2).",
          "- The one-frame probe is fitted on eval-bank rows (cross-fitted by episode); the head is fitted on the train clips. The two are different protocols and are reported beside, not as a paired test of the same fit.",
          "- R3 coefficients are cross-fitted over eval folds; the ego speeds of the rule are the logged ones (the action input of a rollout); no goal input, nothing of the situation classifier.",
          "- The cost read (T1-kin) and controls k5 to k7 belong to the lead.", ""]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--out-dir", default=HERE)
    r = sub.add_parser("run")
    r.add_argument("--arms", default=",".join(ARMS))
    r.add_argument("--in-dir", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--pack-dir", default=HERE)
    r.add_argument("--states-dir", default=None)
    r.add_argument("--alias", default="", help="TAG=STATEFILE_NAME,... (plumbing: PLUMB=G_s0)")
    r.add_argument("--banked-dir", default=BANKED_RB_DIR)
    r.add_argument("--B", type=int, default=2000)
    r.add_argument("--no-train", action="store_true")
    r.add_argument("--no-probe", action="store_true")
    r.add_argument("--plumbing", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "pack":
        m1 = build_eval_pack(a.out_dir + "/eval_pack.npz")
        m2 = build_train_pack(a.out_dir + "/train_pack.npz")
        print(json.dumps({"eval": m1, "train": m2}, indent=1))
        return 0
    alias = dict(x.split("=") for x in a.alias.split(",") if x)
    return run([x for x in a.arms.split(",") if x], a.in_dir, a.out_dir, a.B, a.pack_dir, a.states_dir, alias, a.banked_dir, not a.no_train, not a.no_probe, a.plumbing)


if __name__ == "__main__":
    sys.exit(main())
