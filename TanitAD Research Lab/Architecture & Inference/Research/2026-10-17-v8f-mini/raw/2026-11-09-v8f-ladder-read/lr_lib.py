"""PREREG_V8F_LADDER read (T0): per-level reach read of the ladder world models, IMAG1B's instruments extended to each level's own horizons.
Tier T0 on every number: a world-model diagnostic (latent-space prediction of the encoded future), never driving performance.

Design (IMPL_CHOICES_LADDER_READ.md, frozen before any arm statistic):
  * data   = the 141 eval clips of the M2b cache (101 frames, 0.2 s), pooled frozen-adapter latents ``zmean`` -> ``wm.z1_from_mean`` -> ``stack.encode`` over the WHOLE clip;
  * anchors of level l = its states j (frame j * cum_stride_l); Copy-Last = the state at the anchor; Oracle = the encoded state j + k; Predicted = ``stack.rollout`` under the logged actions;
  * IA-0 skill = 1 - sum|pred - oracle|^2 / sum|copy - oracle|^2 (episode-moment form, IMAG1B's ``Mom``), two-part rule: episode-cluster bootstrap interval excludes 0 on the claimed side AND
    the point lies beyond the 97.5th / 2.5th percentile of the same statistic over R = 199 draws in which the ANSWER (the encoded future) is deranged inside the clip (the IA-0 reference of IMAG1B: no probe is involved);
  * IA-0 health = IMAG1B's MRR at k = 1 (query the predicted latent, answer the window's own encoded future among the clip's anchors) and RankMe, same two-part rule.
Beside, no bar: IA-1 capture, IA-2 probe transfer, IA-3 action audit, TSI, cross-level slope (ridge probes, episode-cluster bootstrap intervals only).
"""
import hashlib
import json
import math
import os
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
THEORY = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(THEORY, "imag1b"))
CODE_ROOT = os.environ.get("LADDER_READ_CODE_ROOT", "F:/TanitAD-scratch/ladder_read/tree")
for _p in (os.path.join(CODE_ROOT, "stack"), os.path.join(CODE_ROOT, "taniteval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as TF  # noqa: E402

import tanitad  # noqa: E402
assert os.path.abspath(tanitad.__file__).replace("\\", "/").lower().startswith(os.path.abspath(CODE_ROOT).replace("\\", "/").lower()), tanitad.__file__
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import imag1b_lib as IL  # noqa: E402  (Mom, boot_weights, mrr_mom, rankme, sep_above, cosine, ...)
import imag1b_probes as PB  # noqa: E402

ART = "E:/Projects/TanitAD-artifacts"
SF = f"{ART}/sfresh_inputs"
PULLS = os.environ.get("LADDER_READ_PULLS", "C:/Users/Admin/t2v8f_pulls")
SCR = "F:/TanitAD-scratch/ladder_read"
LABELS = os.path.join(THEORY, "imag1", "labels_imag1_eval141.npz")
DT = 0.2                                              # one cached frame, seconds
HORIZONS = {1: (1.0, 2.0), 2: (3.0, 6.0), 3: (6.0, 9.0, 12.0, 15.0, 18.0)}      # seconds per level (the card); 15 and 18 s are beside
BAND_END = {1: 2.0, 2: 6.0, 3: 12.0}                  # the deciding horizon of each level
R_REF, B_BOOT = 199, 2000
MIN_FREE_GB = 6.0
FAST, SLOW = ("accel", "yaw_rate", "lead_closing"), ("speed", "lead_present", "lon7", "agents30")
CONTRAST = {"lonp": (0, +1.0), "lonm": (0, -1.0), "latp": (1, +0.25), "latm": (1, -0.25)}
TARGETS = IL.TARGETS
IA2_TARGETS = ("speed", "lead_gap", "yaw_rate")


class LadderMismatch(RuntimeError):
    """The BUILT model does not match the run's argv."""


# ------------------------------------------------------------------------------------------------ memory guard (IMAG1B's gate)
def free_gb():
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30


def memory_guard(min_gb=MIN_FREE_GB):
    f = free_gb()
    if f < min_gb:
        raise MemoryError(f"free physical memory {f:.2f} GB is below the {min_gb} GB guard")
    return f


# ------------------------------------------------------------------------------------------------ run argv, model load, BUILT-vs-argv
def parse_argv(text):
    """argv.txt -> {flag: [values]} (flags without values -> [])."""
    toks, out, cur = text.split(), {}, None
    for t in toks:
        if t.startswith("--"):
            cur = t[2:]
            out[cur] = []
        elif cur is not None:
            out[cur].append(t)
    return out


def load_arm(run_dir):
    """Model of one banked ladder run + its parsed argv. Frozen adapter and WM come from the run's own ckpt."""
    argv = parse_argv(open(os.path.join(run_dir, "argv.txt"), encoding="utf-8").read())
    ck = torch.load(os.path.join(run_dir, "ckpt_last.pt"), map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    r = model.load_state_dict(ck["model"], strict=False)
    assert not r.missing_keys and set(r.unexpected_keys) <= {"hacc_w"}, (r.missing_keys, r.unexpected_keys)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, argv, ck


def built_values(model, n_frames=91):
    """Everything is READ FROM THE BUILT MODEL (never the config that built it): strides three ways, history windows / dilation / causal pad, the free-run k and its anchor count, the context built."""
    wm = model.wm
    bs, bk, bl = wm.built_strides(), wm.built_k(), wm.built_ladder(n_frames)
    return {"strides": bs["stride"], "macro_stride": bs["macro_stride"], "macro_shape_stride": bs["macro_shape_stride"], "hist_dilation": bs["dilation"],
            "hist_k": bk["k"], "hist_dilation_k": bk["dilation"], "causal_pad": bk["causal_pad"], "trained_k": bl["k"], "anchors": bl["anchors"], "ctx_built": bl["ctx_built"],
            "ctx_passed": bl["ctx_passed"], "cum_stride": bl["cum_stride"], "w_in_loss": bl["w_in_loss"], "states_91": bl["states"]}


def _ints(argv, key, n):
    v = argv.get(key)
    if v is None or len(v) != n:
        raise LadderMismatch(f"argv has no usable --{key} ({v})")
    return [int(x) for x in v]


def check_built_vs_argv(built, argv):
    """Raise LadderMismatch (all findings at once) unless the BUILT values equal the argv: strides (three readers), trained k, free-run anchor stride, context, history K."""
    bad = []
    st, k, rs, ctx = _ints(argv, "wm-strides", 2), _ints(argv, "wm-rollout-k", 3), _ints(argv, "wm-rollout-stride", 3), _ints(argv, "wm-ctx", 3)
    hk = _ints(argv, "wm-k", 1)[0]
    if list(built["strides"]) != st:
        bad.append(f"stride: built {built['strides']} vs argv {st}")
    if list(built["macro_stride"]) != st or list(built["macro_shape_stride"]) != st:
        bad.append(f"macro stride / its parameter-shape stride: built {built['macro_stride']} / {built['macro_shape_stride']} vs argv {st}")
    if list(built["cum_stride"]) != [1, st[0], st[0] * st[1]]:
        bad.append(f"cumulative stride: built {built['cum_stride']} vs argv {[1, st[0], st[0] * st[1]]}")
    want_k = [x if x > 1 else 0 for x in k]                       # built_rollout counts predictor calls inside the free run; 0 = no free-run term was built
    if list(built["trained_k"]) != want_k:
        bad.append(f"trained k (free-run predictor calls): built {built['trained_k']} vs argv {k} (expected {want_k})")
    for lv in range(3):
        if k[lv] > 1:
            n_states = built["states_91"][lv]
            want_anch = len(range(0, n_states - 1, rs[lv]))
            if built["anchors"][lv] != want_anch:
                bad.append(f"free-run anchors L{lv + 1}: built {built['anchors'][lv]} vs argv stride {rs[lv]} on {n_states} states = {want_anch}")
    if list(built["ctx_built"]) != ctx:
        bad.append(f"context (predictor parameter shape): built {built['ctx_built']} vs argv {ctx}")
    if list(built["hist_k"]) != [hk, hk, hk]:
        bad.append(f"history K: built {built['hist_k']} vs argv {hk}")
    if hk > 1 and list(built["hist_dilation_k"]) != [1, st[0], st[1]]:
        bad.append(f"history dilation: built {built['hist_dilation_k']} vs argv strides {[1] + st}")
    if hk > 1 and not all(built["causal_pad"]):
        bad.append(f"causal pad not built: {built['causal_pad']}")
    if bad:
        raise LadderMismatch("; ".join(bad))
    return {"strides": st, "trained_k": k, "rollout_stride": rs, "ctx": ctx, "hist_k": hk}


def adapter_sha(model):
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(k for k in sd if k.startswith("adapter.") or k == "pos"):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode() + str(tuple(t.shape)).encode() + str(t.dtype).encode() + t.numpy().tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ eval clips: frame latents, actions, labels
@torch.no_grad()
def zmean_of(model, feats16, ch=16):
    """The frozen adapter's mean token per frame (zcache.py's definition, chunks of 16): feats16 [T, n_tok, d_feat] -> [T, d]."""
    out = []
    for s in range(0, feats16.shape[0], ch):
        f = torch.as_tensor(feats16[s:s + ch]).float()[None]
        out.append((model.adapter(f) + model.pos[None, None]).mean(2)[0])
    return torch.cat(out, 0).float()


def list_eval_clips():
    d = f"{SF}/cache/eval/clips"
    return sorted(f[:-3] for f in os.listdir(d) if f.endswith(".pt"))


def build_eval_frames(model, out_path):
    """zmean [T, d], act [T, 2] (the loader's channels: nan_to_num(a), ch1 = measured yaw rate), ok [T] for every eval clip, with the adapter hash. Saved as one npz (resumable cache)."""
    tracks = D.load_native_tracks(f"{SF}/inputs/native_tracks.pt")
    store = D.ClipStore(f"{SF}/cache", "eval", lru=2, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    clips = list_eval_clips()
    zs, acts, oks = [], [], []
    for c in clips:
        d = store.get(c)
        zs.append(zmean_of(model, d["feats"]).numpy())
        acts.append(np.stack([np.nan_to_num(d["a"]), d["ch1"]], -1).astype(np.float32))
        ok = np.isfinite(d["p10f"]).all(1)
        ok[0] = True
        oks.append(ok)
    T = np.array([len(z) for z in zs])
    np.savez(out_path, clips=np.array(clips), T=T, zmean=np.concatenate(zs), act=np.concatenate(acts), ok=np.concatenate(oks), adapter_sha=adapter_sha(model))
    return out_path


def load_eval_frames(path):
    z = np.load(path, allow_pickle=False)
    T = z["T"]
    cut = np.cumsum(T)[:-1]
    return {"clips": [str(c) for c in z["clips"]], "T": [int(t) for t in T], "zmean": np.split(z["zmean"], cut), "act": np.split(z["act"], cut), "ok": np.split(z["ok"], cut),
            "adapter_sha": str(z["adapter_sha"])}


def build_label_frames(clips, T):
    """{target: [n_clips, max T]} NaN except at the bank frames the IMAG1 label file holds (episode, cache step); ``T`` the per-clip lengths."""
    lab = np.load(LABELS)
    idx = {c: i for i, c in enumerate(clips)}
    Y = {q: np.full((len(clips), max(T)), np.nan) for q in TARGETS}
    for b, (e, g) in enumerate(zip(lab["beid"], lab["bg"])):
        i = idx.get(str(e))
        if i is not None and 0 <= int(g) < T[i]:
            for q in TARGETS:
                Y[q][i, int(g)] = lab[q][b]
    return Y


# ------------------------------------------------------------------------------------------------ the rollout cells (one code path for windows and whole clips)
@torch.no_grad()
def roll_cells(stack, level, z, act, anchors, kmax, ctx):
    """Rollouts from each anchor state of one level under the logged (planning) actions, ``ctx`` states of history: z [B, T, D], act [B, T-1, A] (raw at level 1, macro embeddings above),
    anchors list of state indices -> [B, n_anchor, kmax, D]. Steps past the sequence use zero padding; the callers mask them (validity by k)."""
    idx = torch.as_tensor(list(anchors), dtype=torch.long)
    pad = TF.pad(act, (0, 0, 0, kmax))
    pos = idx[:, None] + torch.arange(kmax)[None]
    return stack.rollout(level, z[:, idx], pad[:, pos], context=ctx)


@torch.no_grad()
def encode_clip(model, zmean, act, contrast=None):
    """One clip (zmean [T, d], act [T, 2] raw) -> (z list per level [1, T_l, D], planning actions per level [1, T_l-1, A])."""
    wm = model.wm
    a = torch.as_tensor(act).clone()
    if contrast is not None:
        ch, dv = contrast
        a[..., ch] += dv
    z1 = wm.z1_from_mean(torch.as_tensor(zmean)[None])
    a_sc = wm.scale_actions(a[None])[:, :-1]
    enc = wm.stack.encode(z1, a_sc)
    plan = [a_sc] + [enc.act[l] for l in (1, 2)]
    return enc.z, plan


def k_of(level, h_s, cum):
    """Steps of level ``level`` nearest to ``h_s`` seconds (half rounds up) and whether h_s is exactly on the level's grid."""
    frames = h_s / DT
    k = int(math.floor(frames / cum + 0.5 + 1e-9))
    return max(k, 1), bool(abs(frames / cum - round(frames / cum)) < 1e-9)


def horizon_plan(cum):
    """{level: [(h_s, k, actual_s, on_grid)]} for a run's cumulative strides."""
    out = {}
    for lv in (1, 2, 3):
        rows = []
        for h in HORIZONS[lv]:
            k, on = k_of(lv, h, cum[lv - 1])
            rows.append((h, k, k * cum[lv - 1] * DT, on))
        out[lv] = rows
    return out


# ------------------------------------------------------------------------------------------------ IA-0 statistics (probe-free)
def sq_err(a, b):
    return ((a.astype(np.float64) - b.astype(np.float64)) ** 2).sum(-1)


def skill_mom(pred, copy, ora, pos, n_ep):
    """Episode moments of skill = sum(e_copy - e_pred) / sum(e_copy): 0 = Copy-Last, 1 = perfect."""
    ep, ec = sq_err(pred, ora), sq_err(copy, ora)
    A = np.bincount(pos, weights=ec - ep, minlength=n_ep)[:, None]
    Dn = np.bincount(pos, weights=ec, minlength=n_ep)[:, None]
    return IL.Mom(A, Dn, None, len(pos), len(np.unique(pos)))


def derange_idx(groups, n, rng):
    """Global answer index per anchor: each clip's answers deranged among its own anchors (IMAG1B's ``mrr_mom`` derangement); a clip with fewer than 2 anchors keeps its own (and is excluded by the caller)."""
    ans = np.arange(n)
    for g in groups:
        m = len(g)
        if m < 2:
            continue
        perm = rng.permutation(m)
        a = np.empty(m, int)
        a[perm] = np.roll(perm, 1)
        ans[g] = g[a]
    return ans


def skill_stats(pred, copy, ora, pos, W, key, R=R_REF):
    """IA-0 skill row: point, bootstrap interval, the derangement reference (R draws; the same statistic with the answer deranged inside the clip, BOTH terms recomputed), the two-part flags.
    Only clips with >= 2 anchors enter (the reference needs a derangement); the excluded anchors are counted."""
    n_ep = W.shape[1]
    cnt = np.bincount(pos, minlength=n_ep)
    keep = cnt[pos] >= 2
    out = {"n_anchor_all": int(len(pos)), "n_excluded_lt2_per_clip": int((~keep).sum())}
    if keep.sum() == 0:
        out.update({"readable": False, "n": 0})
        return out
    pred, copy, ora, pos = pred[keep], copy[keep], ora[keep], pos[keep]
    m = skill_mom(pred, copy, ora, pos, n_ep)
    row = m.row(W)
    order = np.argsort(pos, kind="stable")
    assert (order == np.arange(len(pos))).all(), "anchors must be sorted by clip"
    groups = [np.flatnonzero(pos == e) for e in np.unique(pos)]
    p32, c32, o32 = pred.astype(np.float32), copy.astype(np.float32), ora.astype(np.float32)
    ref = np.empty(R)
    for r in range(R):
        rng = np.random.default_rng([int(key[0]), int(key[1]), r + 1])
        ans = derange_idx(groups, len(pos), rng)
        oa = o32[ans]
        ec = ((c32 - oa) ** 2).sum(-1).sum()
        ep = ((p32 - oa) ** 2).sum(-1).sum()
        ref[r] = (ec - ep) / ec
    hi, lo = float(np.percentile(ref, 97.5)), float(np.percentile(ref, 2.5))
    pt = row["point"]
    row.update({"thr_hi": hi, "thr_lo": lo, "ref_mean": float(ref.mean()), "ref_sd": float(ref.std()), "above": IL.sep_above(row, hi), "below": IL.sep_below(row, lo),
                "rank_p": float((1 + (ref >= pt).sum()) / (R + 1)), "readable": True, "n_anchor_all": out["n_anchor_all"], "n_excluded_lt2_per_clip": out["n_excluded_lt2_per_clip"]})
    return row


def health_stats(pred1, ora1, pos, W, level, R=R_REF):
    """IA-0 health at k = 1: MRR of the encoded future among the clip's anchors (query = predicted latent) against exact chance, with IMAG1B's derangement reference, + RankMe."""
    n_ep = W.shape[1]
    groups_all = [np.flatnonzero(pos == e) for e in range(n_ep)]
    ctx = type("C", (), {})()
    ctx.groups, ctx.qpos, ctx.n_ep = groups_all, pos, n_ep
    m, mrr, chance = IL.mrr_mom(ctx, pred1, ora1, rng=None)
    row = m.row(W)
    ref = np.array([IL.mrr_mom(ctx, pred1, ora1, rng=np.random.default_rng([level, 7, s]))[0].point() for s in range(1, R + 1)])
    thr = float(np.percentile(ref, 97.5))
    row.update({"mrr": mrr, "chance": chance, "thr97": thr, "rank_p": float((1 + (ref >= row["point"]).sum()) / (R + 1)), "separated_two_part": IL.sep_above(row, thr),
                "rankme": {"pred_k1": IL.rankme(pred1), "oracle_k1": IL.rankme(ora1)}})
    return row


# ------------------------------------------------------------------------------------------------ the read of one arm
def gather(model, fr, ctx_modes=(None, 1)):
    """Per level the anchor cells for every horizon (and k = 1 for health): arrays indexed by clip. Returns dict[(level, k, ctx_tag)] -> {pred, copy, ora, pos, frame, ok} and the encoded states per level
    (for the probes). ``ctx_modes``: None = the model's BUILT context (the trained free-run form), 1 = the planning form (IMAG1's)."""
    wm = model.wm
    cum = [1, int(wm.stack.cfg[1].stride), int(wm.stack.cfg[1].stride) * int(wm.stack.cfg[2].stride)]
    built_ctx = [int(p.pos.shape[0]) for p in wm.stack.predictors]
    plan = horizon_plan(cum)
    ks = {lv: sorted({k for _, k, _, _ in plan[lv]} | {1}) for lv in (1, 2, 3)}
    nclip = len(fr["clips"])
    cells, states = {}, {lv: [] for lv in (1, 2, 3)}
    acc = {}
    for ci in range(nclip):
        z, pl = encode_clip(model, fr["zmean"][ci], fr["act"][ci])
        ok = fr["ok"][ci]
        for lv in (1, 2, 3):
            zl, al = z[lv - 1], pl[lv - 1]
            n_st = zl.shape[1]
            states[lv].append(zl[0].numpy())
            kmax = max(ks[lv])
            for cm in ctx_modes:
                ctx = built_ctx[lv - 1] if cm is None else cm
                tag = "built" if cm is None else f"ctx{cm}"
                roll = roll_cells(wm.stack, lv, zl, al, list(range(0, n_st - 1)), kmax, ctx)[0].numpy()        # [n_st-1, kmax, D]
                for k in ks[lv]:
                    j = np.arange(0, n_st - k)
                    if len(j) == 0:
                        continue
                    fr0, fr1 = j * cum[lv - 1], (j + k) * cum[lv - 1]
                    okc = np.array([ok[a:b + 1].all() for a, b in zip(fr0, fr1)])
                    d = acc.setdefault((lv, k, tag), {"pred": [], "copy": [], "ora": [], "pos": [], "frame": [], "ok": [], "n_masked_end": 0, "n_states": n_st})
                    d["pred"].append(roll[j, k - 1])
                    d["copy"].append(zl[0, j].numpy())
                    d["ora"].append(zl[0, j + k].numpy())
                    d["pos"].append(np.full(len(j), ci))
                    d["frame"].append(fr0)
                    d["ok"].append(okc)
                    d["n_masked_end"] += min(k, n_st)
    for key, d in acc.items():
        cells[key] = {k_: (np.concatenate(v) if isinstance(v, list) else v) for k_, v in d.items()}
    flat = {}
    for lv, v in states.items():
        flat[lv] = {"X": np.concatenate(v), "pos": np.concatenate([np.full(len(a), i) for i, a in enumerate(v)]), "frame": np.concatenate([np.arange(len(a)) * cum[lv - 1] for a in v])}
    return cells, flat, cum, plan, built_ctx


def check_cells_against_states(cells, states, cum):
    """Control 2b: the Copy-Last row of every cell must BE the encoded state at the anchor frame and the Oracle the encoded state k steps later (read independently from the flat per-level states), bit for bit.
    A Copy-Last reference shifted by one step, or an Oracle off by one, fails here. Returns the number of cells checked; raises RuntimeError on the first mismatch."""
    n = 0
    for (lv, k, tag), c in cells.items():
        st = states[lv]
        lut = {(int(p), int(f)): i for i, (p, f) in enumerate(zip(st["pos"], st["frame"]))}
        ia = np.array([lut[(int(p), int(f))] for p, f in zip(c["pos"], c["frame"])])
        ib = np.array([lut[(int(p), int(f) + k * cum[lv - 1])] for p, f in zip(c["pos"], c["frame"])])
        if not (np.array_equal(c["copy"], st["X"][ia]) and np.array_equal(c["ora"], st["X"][ib])):
            raise RuntimeError(f"cell (level {lv}, k {k}, {tag}): Copy-Last / Oracle are not the encoded states at the anchor / anchor + k steps")
        n += 1
    return n


def ia0_arm(cells, plan, cum, W, ctx_tag="built", smoke=False):
    """IA-0 skill rows at every horizon of every level + health, for one arm/seed."""
    out = {}
    R = 19 if smoke else R_REF
    for lv in (1, 2, 3):
        rows = {}
        for h, k, actual, on in plan[lv]:
            c = cells.get((lv, k, ctx_tag))
            if c is None:
                rows[f"{h:g}"] = {"k": k, "actual_s": actual, "on_grid": on, "readable": False, "reason": "no anchor holds this reach", "n_masked_clips_steps": None}
                continue
            sel = c["ok"]
            n_bad = int((~sel).sum())
            r = skill_stats(c["pred"][sel], c["copy"][sel], c["ora"][sel], c["pos"][sel], W, (lv, k), R=R)
            r.update({"k": k, "actual_s": actual, "on_grid": on, "n_not_ok_frames_dropped": n_bad, "n_masked_end_of_clip": int(c["n_masked_end"]), "band_end": bool(abs(h - BAND_END[lv]) < 1e-9)})
            rows[f"{h:g}"] = r
        c1 = cells[(lv, 1, ctx_tag)]
        sel = c1["ok"]
        health = health_stats(c1["pred"][sel], c1["ora"][sel], c1["pos"][sel], W, lv, R=R)
        out[str(lv)] = {"rows": rows, "health": health, "step_s": cum[lv - 1] * DT}
    return out


def beside_copy_selfcheck(cells, W, ctx_tag="built"):
    """A Copy-Last row against itself must read exactly zero skill (every level, every k): returns the max |point|, |lo|, |hi|."""
    worst = 0.0
    for (lv, k, tag), c in cells.items():
        if tag != ctx_tag:
            continue
        m = skill_mom(c["copy"], c["copy"], c["ora"], c["pos"], W.shape[1])
        r = m.row(W)
        worst = max(worst, abs(r["point"]), abs(r["lo"]), abs(r["hi"]))
    return worst


# ------------------------------------------------------------------------------------------------ outcome labels (the card)
def level_passes(ia0_levels):
    """{level: {'beats': bool|None, 'healthy': bool}} for one run. beats = skill at the band end separated above Copy-Last by the two-part rule."""
    out = {}
    for lv in ("1", "2", "3"):
        be = [r for r in ia0_levels[lv]["rows"].values() if r.get("band_end")]
        r = be[0] if be else {}
        out[lv] = {"beats": bool(r.get("above")) if r.get("readable") else None, "worse": bool(r.get("below")) if r.get("readable") else None,
                   "healthy": bool(ia0_levels[lv]["health"]["separated_two_part"]), "skill": r.get("point"), "lo": r.get("lo"), "hi": r.get("hi")}
    return out


def band_outcome(per_seed):
    """per_seed: {seed: level_passes}. A level counts for the ARM only if it beats at EVERY seed (both seeds required); healthy likewise.
    BAND-MET = all three beat and all three healthy; BAND-PARTIAL = at least one beats, not (all beat and all healthy) -> names the levels; BAND-MISS = none beats."""
    levels = ("1", "2", "3")
    beats = {lv: all(p[lv]["beats"] is True for p in per_seed.values()) for lv in levels}
    healthy = {lv: all(p[lv]["healthy"] for p in per_seed.values()) for lv in levels}
    n = sum(beats.values())
    if n == 3 and all(healthy.values()):
        label = "BAND-MET"
    elif n == 0:
        label = "BAND-MISS"
    else:
        label = "BAND-PARTIAL"
    note = ""
    if n == 3 and not all(healthy.values()):
        note = "all three levels beat Copy-Last but health fails at L" + ",L".join(lv for lv in levels if not healthy[lv]) + " (a case the card's BAND-MET / PARTIAL wording does not name; listed as PARTIAL)"
    return {"label": label, "levels_beating": [lv for lv in levels if beats[lv]], "levels_not_beating": [lv for lv in levels if not beats[lv]], "healthy": healthy, "note": note}


# ------------------------------------------------------------------------------------------------ beside modules (no bar; bootstrap intervals only)
def fold_of(e):
    return int(hashlib.sha256(e.encode()).hexdigest()[:8], 16) % 2


def _mom_cont(y, p, valid, pos, n_ep, ymean_by_row):
    return IL.mom_cont(y, p, valid, pos, n_ep, ymean_by_row)


def fit_level_probes(st, Y, clips):
    """Ridge / logistic probes on ENCODED states of one level (``st`` = gather's flat dict: X, pos, frame), per target and fold (fit on the other fold's episodes). Returns {(q, fold): probe}."""
    fold = np.array([fold_of(c) for c in clips])[st["pos"]]
    probes = {}
    for q in TARGETS:
        y = Y[q][st["pos"], st["frame"]]
        for k in (0, 1):
            m = (fold != k) & np.isfinite(y)
            if m.sum() < 30 or (IL.KIND[q] != "cont" and len(np.unique(y[m])) < 2):
                continue
            probes[(q, k)] = PB.fit_lin(IL.KIND[q], st["X"][m].astype(np.float64), y[m])
    return probes


def decode_states(probes, q, X, fold_rows):
    out = np.full(len(X), np.nan)
    for k in (0, 1):
        m = fold_rows == k
        if m.any() and (q, k) in probes:
            out[m] = PB.predict_lin(probes[(q, k)], X[m].astype(np.float64))
    return out


def _score_mom(q, y, p, valid, pos, n_ep, rowfold, fbase):
    if IL.KIND[q] == "cont":
        ym = np.array([y[valid & (rowfold == k)].mean() if (valid & (rowfold == k)).any() else 0.0 for k in (0, 1)])
        return IL.mom_cont(y, p, valid, pos, n_ep, ym[rowfold])
    base = np.where(rowfold == 0, fbase[(q, 0)], fbase[(q, 1)])
    return IL.mom_cls(y, p, base, valid, pos, n_ep, IL.CLASSES[q])


def beside_arm(model, fr, Y, states, cells, cum, plan, W, ctx_tag="built", smoke=False):
    # states: gather's flat per-level dicts
    """IA-1 (capture), IA-2 (probe transfer on speed / lead gap / yaw rate), IA-3 (action audit), TSI and the cross-level slope. Ridge / logistic probes, episode-cluster bootstrap intervals only (no bar, no reference)."""
    clips = fr["clips"]
    n_ep = len(clips)
    out = {"ia1": {}, "ia2": {}, "ia3": {}, "tsi": {}, "slope": {}}
    PB.CONV["fits"] = 0
    PB.CONV["not_converged"] = 0
    fold = np.array([fold_of(c) for c in clips])
    fb = {}
    probes_by_level = {}
    for lv in (1, 2, 3):
        st = states[lv]
        probes = fit_level_probes(st, Y, clips)
        probes_by_level[lv] = probes
        pos = st["pos"]
        rf = fold[pos]
        X = st["X"]
        sc = {}
        for q in TARGETS:
            y = Y[q][pos, st["frame"]]
            valid = np.isfinite(y)
            for k in (0, 1):
                fb[(q, k)] = PB.majority(y[valid & (rf != k)]) if IL.KIND[q] != "cont" and (valid & (rf != k)).any() else np.nan
            if not any((q, k) in probes for k in (0, 1)):
                out["ia1"][f"{lv}|{q}"] = {"readable": False, "n": 0}
                continue
            p = decode_states(probes, q, X, rf)
            m = _score_mom(q, y, p, valid & np.isfinite(p), pos, n_ep, rf, fb)
            row = m.row(W)
            row["score_is"] = "1-NMSE" if IL.KIND[q] == "cont" else "balanced-accuracy minus majority"
            out["ia1"][f"{lv}|{q}"] = row
            sc[q] = (m.point(), m.ba_point() if IL.KIND[q] != "cont" else None)
        fast = [1 - sc[q][0] for q in FAST if q in sc]
        slow = [(1 - sc[q][0]) if IL.KIND[q] == "cont" else (1 - sc[q][1]) for q in SLOW if q in sc]
        out["tsi"][str(lv)] = {"tsi": (float(np.mean(fast)) - float(np.mean(slow))) if fast and slow else None, "n_fast": len(fast), "n_slow": len(slow)}
    # IA-2: probes of each level decode predicted / Copy-Last / Oracle at the horizons
    for lv in (1, 2, 3):
        for h, k, actual, on in plan[lv]:
            c = cells.get((lv, k, ctx_tag))
            if c is None:
                continue
            sel = c["ok"]
            pos, rf = c["pos"][sel], fold[c["pos"][sel]]
            tf = c["frame"][sel] + k * cum[lv - 1]
            for q in IA2_TARGETS:
                y = Y[q][c["pos"][sel], tf]
                valid = np.isfinite(y)
                if valid.sum() < 30 or not any((q, kk) in probes_by_level[lv] for kk in (0, 1)):
                    out["ia2"][f"{lv}|{h:g}|{q}"] = {"readable": False, "n": int(valid.sum())}
                    continue
                moms = {}
                for nm in ("pred", "copy", "ora"):
                    p = decode_states(probes_by_level[lv], q, c[nm][sel], rf)
                    moms[nm] = _score_mom(q, y, p, valid & np.isfinite(p), pos, n_ep, rf, fb)
                row = {nm: moms[nm].row(W) for nm in moms}
                row["pred_minus_copy"] = IL.Mom(moms["pred"].A - moms["copy"].A, moms["pred"].D, None, moms["pred"].n, moms["pred"].nep).row(W)
                row.update({"k": k, "actual_s": actual, "on_grid": on, "n": int(valid.sum())})
                out["ia2"][f"{lv}|{h:g}|{q}"] = row
    # IA-3: contrasted rollouts, decoded speed / yaw rate (speed probe for lon, yaw probe for lat), gain and the 2 x 2 Jacobian cosine
    out["ia3"] = ia3_arm(model, fr, probes_by_level, fold, cum, plan, W, smoke)
    out["slope"] = slope_arm(model, fr, probes_by_level, fold, cum, plan, W)
    out["conv"] = dict(PB.CONV)
    return out


def _slope_vec(x, y, pos, n_ep, W):
    """OLS slope of y on x with the episode-cluster bootstrap (per-episode sums)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, pos = x[ok], y[ok], pos[ok]
    S = {n: np.bincount(pos, weights=v, minlength=n_ep) for n, v in (("n", np.ones_like(x)), ("x", x), ("y", y), ("xx", x * x), ("xy", x * y))}

    def sl(w):
        n, sx, sy, sxx, sxy = (w @ S[k] for k in ("n", "x", "y", "xx", "xy"))
        den = n * sxx - sx ** 2
        return np.where(den > 0, (n * sxy - sx * sy) / np.where(den > 0, den, 1.0), np.nan)
    pt = float(sl(np.ones((1, n_ep)))[0])
    v = sl(W)
    lo, hi = np.nanpercentile(v, [2.5, 97.5]) if np.isfinite(v).any() else (np.nan, np.nan)
    return {"point": pt, "lo": float(lo), "hi": float(hi), "n": int(ok.sum()), "nep": int(len(np.unique(pos)))}


def slope_arm(model, fr, probes_by_level, fold, cum, plan, W):
    """Cross-level consistency, no bar: decoded change of speed over the same real horizon read from level l (k_l steps) and level l+1 (k_{l+1} steps) at anchors on level l+1's grid (the level l state at the same frame).
    Horizon 6.0 s for L1 vs L2; L2 vs L3 at 6.0 s when both levels hold it on their grid, else 12.0 s. Slope of the coarser level's decoded change on the finer one's."""
    out = {}
    pairs = []
    for (a, b) in ((1, 2), (2, 3)):
        for h in (6.0, 12.0):
            ka, oka = k_of(a, h, cum[a - 1])
            kb, okb = k_of(b, h, cum[b - 1])
            if oka and okb:
                pairs.append((a, b, h, ka, kb))
                break
    wm = model.wm
    n_ep = len(fr["clips"])
    built_ctx = [int(p.pos.shape[0]) for p in wm.stack.predictors]
    for a, b, h, ka, kb in pairs:
        xs, ys, ps = [], [], []
        for ci in range(n_ep):
            z, pl = encode_clip(model, fr["zmean"][ci], fr["act"][ci])
            nb = z[b - 1].shape[1]
            jb = np.arange(0, nb - kb)
            if len(jb) == 0:
                continue
            ja = jb * (cum[b - 1] // cum[a - 1])                      # level a's state at the same frame
            if ja.max() + ka > z[a - 1].shape[1] - 1:
                keep = ja + ka <= z[a - 1].shape[1] - 1
                jb, ja = jb[keep], ja[keep]
                if len(jb) == 0:
                    continue
            ra = roll_cells(wm.stack, a, z[a - 1], pl[a - 1], list(ja), ka, built_ctx[a - 1])[0, :, ka - 1].numpy()
            rb = roll_cells(wm.stack, b, z[b - 1], pl[b - 1], list(jb), kb, built_ctx[b - 1])[0, :, kb - 1].numpy()
            za, zb = z[a - 1][0, ja].numpy(), z[b - 1][0, jb].numpy()
            rf = np.full(len(jb), fold[ci])
            da = decode_states(probes_by_level[a], "speed", ra, rf) - decode_states(probes_by_level[a], "speed", za, rf)
            db = decode_states(probes_by_level[b], "speed", rb, rf) - decode_states(probes_by_level[b], "speed", zb, rf)
            xs.append(da)
            ys.append(db)
            ps.append(np.full(len(jb), ci))
        if xs:
            r = _slope_vec(np.concatenate(xs), np.concatenate(ys), np.concatenate(ps), n_ep, W)
            r.update({"levels": [a, b], "horizon_s": h, "k": [ka, kb]})
            out[f"L{a}vsL{b}"] = r
    return out


def ia3_arm(model, fr, probes_by_level, fold, cum, plan, W, smoke=False):
    wm = model.wm
    n_ep = len(fr["clips"])
    built_ctx = [int(p.pos.shape[0]) for p in wm.stack.predictors]
    ks = {lv: sorted({k for _, k, _, _ in plan[lv]}) for lv in (1, 2, 3)}
    acc = {}                                                    # (lv, k, tag) -> list over clips of {cn: decoded (speed, yaw) arrays}
    for ci in range(n_ep):
        for cn, ctr in CONTRAST.items():
            z, pl = encode_clip(model, fr["zmean"][ci], fr["act"][ci], contrast=ctr)
            for lv in (1, 2, 3):
                zl, al = z[lv - 1], pl[lv - 1]
                n_st = zl.shape[1]
                kmax = max(ks[lv])
                roll = roll_cells(wm.stack, lv, zl, al, list(range(0, n_st - 1)), kmax, built_ctx[lv - 1])[0].numpy()
                for k in ks[lv]:
                    j = np.arange(0, n_st - k)
                    if len(j) == 0:
                        continue
                    rf = np.full(len(j), fold[ci])
                    dq = {tq: decode_states(probes_by_level[lv], tq, roll[j, k - 1], rf) for tq in ("speed", "yaw_rate")}
                    acc.setdefault((lv, k), {}).setdefault(ci, {})[cn] = dq
    out = {}
    for (lv, k), per in acc.items():
        h = k * cum[lv - 1] * DT
        A = {n: np.zeros(n_ep) for n in ("s_lon", "s_lat", "y_lon", "y_lat")}
        cnt = np.zeros(n_ep)
        for ci, d in per.items():
            if len(d) < 4:
                continue
            cnt[ci] = len(d["lonp"]["speed"])
            A["s_lon"][ci] = np.nansum(d["lonp"]["speed"] - d["lonm"]["speed"])
            A["s_lat"][ci] = np.nansum(d["latp"]["speed"] - d["latm"]["speed"])
            A["y_lon"][ci] = np.nansum(d["lonp"]["yaw_rate"] - d["lonm"]["yaw_rate"])
            A["y_lat"][ci] = np.nansum(d["latp"]["yaw_rate"] - d["latm"]["yaw_rate"])
        dl, dt_ = IL.DELTA["lon"], IL.DELTA["lat"]
        den = {"s_lon": cnt * dl, "s_lat": cnt * dt_, "y_lon": cnt * dl, "y_lat": cnt * dt_}
        J = {n: IL.Mom(A[n][:, None], den[n][:, None], None, int(cnt.sum()), int((cnt > 0).sum())) for n in A}
        gl = IL.Mom(A["s_lon"][:, None], (cnt * dl * h)[:, None], None, int(cnt.sum()), int((cnt > 0).sum()))      # speed effect / physical (delta_a x h)
        gt = IL.Mom(A["y_lat"][:, None], (cnt * dt_ * 1.0)[:, None], None, int(cnt.sum()), int((cnt > 0).sum()))   # yaw effect / physical (delta yaw rate)
        Jv = IL.jac_boot(J, W)
        cos_b = IL.cosine(Jv, h)
        cos_pt = IL.cosine({n: np.array(J[n].point()) for n in J}, h)
        lo, hi = np.nanpercentile(cos_b, [2.5, 97.5])
        out[f"{lv}|{k}"] = {"h_s": h, "gain_lon": gl.row(W), "gain_lat": gt.row(W), "cos": {"point": float(cos_pt), "lo": float(lo), "hi": float(hi)}, "n": int(cnt.sum())}
    return out


def encoded_states_ok(model, fr):
    """Sanity: the encode of every clip gives the states the horizon plan assumes (states per level)."""
    z, _ = encode_clip(model, fr["zmean"][0], fr["act"][0])
    return [int(t.shape[1]) for t in z]
