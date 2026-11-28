"""stats_pass.py -- stream the eval cache clip by clip (scp from box 2, READ ONLY), run the hooked forward of BOTH training seeds on every
eval window, bank one chunk per clip, delete the clip.  Footprint on C: = a few clips (21 MB each) + the chunks (~200 MB in total).

Per window and seed it banks: all cross-attention weights [n, heads, n_q, S] of every (level, layer), the model outputs, and the CONTROL numbers
  * hooked vs unhooked traj60 max|d| (per batch),
  * hooked traj30 / ADE6 vs the BANKED eval npz (eval_M2b_s{0,1}-kind-own.npz) max|d|,
and for seed 0 the Adebayo-style model-randomisation control (one level's Reader re-initialised, downstream inputs frozen at the trained values).

  python stats_pass.py [--limit N]        (resumable: a clip whose chunk exists is skipped)
"""
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import threading
import queue
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attn_extract import *  # noqa: E402,F401,F403
from scipy.stats import rankdata  # noqa: E402

STREAM = f"{WORK}/cache_stream"
CHUNKS = f"{WORK}/stats/chunks"
REMOTE = "tanitad-box2:/D:/tanitad/cache_full/eval/clips"
SEEDS = ("M2b_s0", "M2b_s1")
BANKED = {s: np.load(f"{WORK}/banked/eval_{s}-kind-own.npz") for s in SEEDS}


def fetch(clip: str, dst_dir: str, tries: int = 4) -> str:
    dst = f"{dst_dir}/{clip}.pt"
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        return dst
    for k in range(tries):
        r = subprocess.run(["scp", "-q", f"{REMOTE}/{clip}.pt", dst + ".part"], capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(dst + ".part"):
            os.replace(dst + ".part", dst)
            return dst
        time.sleep(2 + 3 * k)
    raise RuntimeError(f"scp failed for {clip}: {r.stderr[-300:]}")


def random_reader(model, level: str, seed: int = 1234):
    """A deep copy of `model` whose `read_<level>` is a FRESH Reader (q, decoder, norm re-initialised by the constructor), heads untouched."""
    m2 = copy.deepcopy(model)
    old = getattr(m2, READER[level])
    torch.manual_seed(seed)
    new = M.Reader(m2.cfg.d, m2.cfg.n_q, m2.cfg.heads, m2.cfg.ff, len(old.dec.layers), old.uses_scene)
    setattr(m2, READER[level], new.eval())
    return m2


def frozen_downstream(m2, level: str, trained_out: dict):
    """Make the RANDOMISED level run (so its attention is captured) but hand the trained output to the levels below it, so that every
    other level sees bit-identical inputs.  S -> returns the trained checkpoints; T -> the trained tactical dict; O -> nothing below."""
    if level == "S":
        orig = m2.run_s
        m2.run_s = lambda *a, **k: (orig(*a, **k), trained_out["ckpt"])[1]
    elif level == "T":
        orig = m2.run_t
        keys = ("goal_logits", "lat_logits", "lon_logits", "cons_std", "goal_prob", "cons_pred")
        m2.run_t = lambda *a, **k: (orig(*a, **k), {kk: trained_out[kk] for kk in keys})[1]


def scene_map(attn: dict, level: str):
    """[n, 4, 40] scene attention, mean over layers, heads, queries (the plotted aggregation)."""
    return level_maps(attn, level, "mean")["scene"]


def spearman_rows(a: np.ndarray, b: np.ndarray):
    ra, rb = rankdata(a, axis=1), rankdata(b, axis=1)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))


def process_clip(clip: str, models: dict, ew: EvalWorld, bs: int = 24):
    idx = ew.clip_windows(clip)
    chunk = {"win": idx}
    ctl = {}
    for sname, model in models.items():
        outs, atts = [], {}
        dmax = dnw = dw = 0.0
        for b0 in range(0, len(idx), bs):
            ii = idx[b0:b0 + bs]
            b = ew.batch(ii)
            o0 = forward_plain(model, b)
            o1, att = forward_tapped(model, b)                        # mode='exact': the real forward, untouched
            dmax = max(dmax, float((o0["traj60"] - o1["traj60"]).abs().max()))
            o3, att3 = forward_tapped(model, b, mode="need_weights")    # torch's own weights: weights agreement + the (<= 1 ulp) alt-path deviation
            dnw = max(dnw, float((o0["traj60"] - o3["traj60"]).abs().max()))
            dw = max(dw, max(float((att[k] - att3[k]).abs().max()) for k in att))
            outs.append({k: o1[k].float().numpy() for k in ("traj30", "traj60", "ckpt", "goal_prob", "lat_logits", "lon_logits", "cons_pred", "code")}
                        | {"a_t": b["a_t"].numpy(), "a_s": b["a_s"].numpy(), "ck_now": M.transform_ckpt(o1["ckpt"], b["pose_s"], b["pose_now"]).numpy(),
                           "pad_frac": (~b["frame_ok"][:, -4:]).float().mean(1).numpy()})
            for k, v in att.items():
                atts.setdefault(k, []).append(v.numpy())
            # ---- randomisation control (seed 0 only)
            if sname == SEEDS[0]:
                for lv in LEVELS:
                    m2 = random_reader(model, lv)
                    frozen_downstream(m2, lv, o1)
                    o2, att2 = forward_tapped(m2, b)
                    r = ctl.setdefault(lv, {"rho_all": [], "rho_new": [], "dother": [], "dtraj": [], "cond_trained": [], "cond_random": []})
                    sm_t, sm_r = scene_map(att, lv).reshape(len(ii), -1).numpy(), scene_map(att2, lv).reshape(len(ii), -1).numpy()
                    r["rho_all"].append(spearman_rows(sm_t, sm_r))
                    r["rho_new"].append(spearman_rows(sm_t[:, -NTOK:], sm_r[:, -NTOK:]))
                    r["cond_trained"].append(level_maps(att, lv)["cond"].sum(1).numpy())
                    r["cond_random"].append(level_maps(att2, lv)["cond"].sum(1).numpy())
                    dd = 0.0
                    for k in att:
                        if k[0] != lv:
                            dd = max(dd, float((att[k] - att2[k]).abs().max()))
                    r["dother"].append(np.full(len(ii), dd, np.float32))
                    r["dtraj"].append(np.full(len(ii), float((o1["traj60"] - o2["traj60"]).abs().max()), np.float32))
        for k in outs[0]:
            chunk[f"{sname}__{k}"] = np.concatenate([o[k] for o in outs])
        for k, v in atts.items():
            chunk[f"{sname}__att_{k[0]}{k[1]}"] = np.concatenate(v)
        bk = BANKED[sname]
        pred = chunk[f"{sname}__traj30"]
        chunk[f"{sname}__ade6"] = ade6_of(pred, ew.X["gt"][idx])
        chunk[f"{sname}__ctl_hook_vs_plain_traj60"] = np.full(len(idx), dmax, np.float32)
        chunk[f"{sname}__ctl_needweights_vs_plain_traj60"] = np.full(len(idx), dnw, np.float32)
        chunk[f"{sname}__ctl_explicit_vs_needweights_attn"] = np.full(len(idx), dw, np.float32)
        chunk[f"{sname}__ctl_vs_banked_pred"] = np.abs(pred - bk["pred"][idx]).reshape(len(idx), -1).max(1)
        chunk[f"{sname}__ctl_vs_banked_ade6"] = np.abs(chunk[f"{sname}__ade6"] - bk["ade6"][idx])
    for lv, r in ctl.items():
        for k, v in r.items():
            chunk[f"rand_{lv}__{k}"] = np.concatenate(v)
    return chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clips", default="")
    a = ap.parse_args()
    os.makedirs(f"{STREAM}/eval/clips", exist_ok=True)
    os.makedirs(CHUNKS, exist_ok=True)
    models = {s: load_model(s)[0] for s in SEEDS}
    cfg = load_model(SEEDS[0])[1]
    ew = EvalWorld(cfg, lru=2, cache=STREAM)
    clips = list(dict.fromkeys(ew.win_clip.tolist()))
    if a.clips:
        clips = [c for c in clips if c in a.clips.split(",")]
    todo = [c for c in clips if not os.path.exists(f"{CHUNKS}/{c}.npz")]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(clips)} clips, {len(todo)} todo", flush=True)
    q: queue.Queue = queue.Queue(maxsize=2)

    def producer():
        for c in todo:
            q.put((c, fetch(c, f"{STREAM}/eval/clips")))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    t0 = time.time()
    n = 0
    while True:
        it = q.get()
        if it is None:
            break
        clip, path = it
        ew.store._c.clear()
        ch = process_clip(clip, models, ew)
        np.savez(f"{CHUNKS}/{clip}.npz", **ch)
        os.remove(path)
        n += 1
        print(f"[{n}/{len(todo)}] {clip} {len(ch['win'])} windows  {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
