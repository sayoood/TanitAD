"""brake_locus_reroll_r8.py -- the CPU re-roll that settles what the banked dumps cannot (BRAKE_LOCUS_REROLL_DESIGN.md phases A and B + AMENDMENT 3's hedge arms).
NEW file.  `force_lon_reroll_r8.py` is IMPORTED (loader constants, per-window sampler seed, apportion, jdump), never edited and never copied wholesale; the roll
preamble mirrors its `stage_roll`.  CPU ONLY (CUDA_VISIBLE_DEVICES=-1; never touches devbox_gpu.lock).

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=6 R8_NO_POSE_SYNC=1 PYTHONIOENCODING=utf-8 \
    PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' \
      python -X utf8 brake_locus_reroll_r8.py select                    # banked tree -> plan (numpy only)
      python -X utf8 brake_locus_reroll_r8.py roll --chunk A-003        # ONE child process = one chunk = one model load; resumable
      python -X utf8 brake_locus_reroll_r8.py roll --smoke 2            # 2-window smoke of EVERY arm type into tag 'smoke'
      python -X utf8 brake_locus_reroll_r8.py status                    # artifact read only
      python -X utf8 brake_locus_reroll_r8.py next                      # next pending chunk id in priority order
      python -X utf8 brake_locus_reroll_r8.py smoke-report              # shapes / selected==emitted / bit-equality vs force_lon / timing

STAGES (priority order; every stage is chunked by WHOLE EPISODES, ~20 min / chunk, one model load per chunk):
  A   (phase A, S6 part) native N on FB (all 493 banked r8 false-brake S6 windows) u CT (250 seeded GT-cruise non-false-brake) u DC (100 seeded GT-decel);
      N_s1 (second inference draw, window_seed(INFER_SEED+1)) on the ALONG windows among them.
  BH  (phase B, FIRST: AMENDMENT 3 priority) hedge arms H1 / H2 + the N2 floor on FB250 u CT100 u DC100.
  BF  F2 (force CRUISE) on FB250; Fbrake (force BRAKE_TO, the positive control) on CT100.
  BS  S0a (gamma_prog := 0), S0b (all four r8_sel scalars := 0) on FB250 u CT100; S0id / S0x controls on a seeded 60.
  AL  (phase A, S2 part, LAST: costliest, and its expected outcome is stated in advance in AMENDMENT 1 A3) N + N_s1 on the remaining ALONG-RAND windows
      (5 per episode of the S2 compliant windows, 690 in all, 150 of them already in stage A).
  Order A, BH, BF, BS, AL = value per CPU-hour under the 2026-10-20 15:00 stop (E needs only A; the hedge arms only BH); a complete run is the same set.
  A stage whose arm list has no N SKIPS the base forward (the proxy returns the last arm's output to the dump flow): N for those windows is stage A's.
Every arm banks the SAME fields (DESIGN section 1): refined fan [149,8,2] fp16, sel_score, sel_score_v3, lon6 / lat3 / kind tags, cons_raw / cons_valid,
reach_keep, the unrefined base bank [117,8,2], the residual prior path, sel_idx / sel_idx_base, the emitted plan, the allocation (r8_hyp_alloc).
Sidecars: <SCR>/bank_<tag>/<chunk>/part*.npz + progress.json (the progress file is the truth; an npz not listed in it is an orphan and is redone).

TIER: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).  ESTIMATOR: paired episode-cluster bootstrap (the analysis).
R6 (design section 5): the stochastic DDIM on CPU is NOT bit-equal to the GPU banked dumps; the controls are re-roll vs re-roll (N vs force_lon's N,
S0id vs N) plus the N2 floor.  The seeds of PREREG_BRAKE_LOCUS.md and AMENDMENTs 1, 2, 3 are verified before anything runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PKG = HERE.parent
OUT = PKG / "raw" / "brake_reroll"
HEAD = PKG / "raw" / "refcv9_headroom"
DEP = PKG / "raw" / "retime_deploy"
SCR = Path(os.environ.get("BLR_SCR", "E:/refcv8_eval_kit/brake_locus_scratch"))
ARCH = "C:/Users/Admin/refcv8_eval_archive/battery/step30000ema_unc"
TREE = "step30000ema_unc"

import force_lon_reroll_r8 as FL  # noqa: E402  (imported, never edited)

TIER = FL.TIER
CKPT, CONFIG = FL.CKPT, FL.CONFIG
SEL_SEED = 20261020                      # window sampling seed, sealed in AMENDMENT 3 before any number existed
ALONG_SEED = 20261019                    # the RAND-candidate draw seed of AMENDMENT 1 A3 (analysis)
N_FB_USED, N_CT, N_CT_B, N_DC, N_FB_B = 493, 250, 100, 100, 250
ALONG_PER_EP = 5
N_S0CTL = 60
N2_OFF = FL.N2_SEED_OFFSET
LIT = {"n_cruise": 1856, "n_fb": 493, "n_ct_pop": 1363}      # typed from L2_HEADROOM.json / the original prereg C1 (0.265625 * 1856 = 493)
STAGES = ("A", "BH", "BF", "BS", "AL")
ARM_SPECS = {                            # name -> (sampler seed offset, kind, arg)
    "N": (0, "native", None), "N_s1": (1, "native", None), "N2": (N2_OFF, "native", None),
    "H1": (0, "hedge", "H1"), "H2": (0, "hedge", "H2"),
    "F2": (0, "force", FL.CRUISE6), "Fbrake": (0, "force", FL.BRAKE6),
    "S0a": (0, "s0", "a"), "S0b": (0, "s0", "b"), "S0id": (0, "s0", "id"), "S0x": (0, "s0", "x"),
}
COST_S = {"N": 14.6}                     # seconds per forward, MEASURED in the 2-window smoke (2026-10-19, CPU busy with the 40k battery): base N 14.6, every extra arm ~7.5 (6.9-7.8); a stage WITHOUT arm N skips the base forward
CHUNK_TARGET_S = 1300.0
PREREG_FILES = ("PREREG_BRAKE_LOCUS", "PREREG_BRAKE_LOCUS_AMENDMENT1", "PREREG_BRAKE_LOCUS_AMENDMENT2", "PREREG_BRAKE_LOCUS_AMENDMENT3", "PREREG_BRAKE_LOCUS_AMENDMENT4")
S6_SLOTS = [1, 3, 4, 5, 6, 7]


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_seals() -> dict:
    """REFUSE unless the sealed prereg and all three amendments match their .sha256 (AMENDMENT 3: 'the roll and its analysis REFUSE to run')."""
    out = {}
    for nm in PREREG_FILES:
        sealed = (HEAD / f"{nm}.sha256").read_text(encoding="utf-8").strip()
        actual = sha256_file(HEAD / f"{nm}.md")
        if sealed != actual:
            raise SystemExit(f"[brake_reroll] REFUSED: {nm} sealed {sealed} != file {actual}")
        out[nm] = actual[:12]
    return out


def sha12(clip: str) -> str:
    return hashlib.sha256(str(clip).encode()).hexdigest()[:12]


def est_cost(arms) -> float:
    return sum(COST_S.get(a, 7.5) for a in arms)


# ====================================================================================================== stage: select
def pick(mask, n, ep, rng) -> list:
    """<= n rows of `mask`, proportional over episodes (largest remainder, never above an episode's own count), shuffled within episode."""
    by = {}
    for i in np.nonzero(mask)[0]:
        by.setdefault(int(ep[i]), []).append(int(i))
    if n >= sum(len(v) for v in by.values()):
        return sorted(i for v in by.values() for i in v)
    take = FL.apportion({e: len(v) for e, v in by.items()}, n, rng)
    sel = []
    for e in sorted(by):
        idx = list(by[e])
        rng.shuffle(idx)
        sel += idx[:take[e]]
    return sorted(sel)


def build_chunks(wins: dict, stage_tasks: dict) -> list:
    """stage_tasks[stage] = {key: [arms]}.  Chunks hold WHOLE episodes, ordered by (episode, ws), cumulative ESTIMATED cost >= CHUNK_TARGET_S closes a chunk."""
    chunks = []
    for st in STAGES:
        tasks = stage_tasks.get(st) or {}
        by_ep = {}
        for k in sorted(tasks, key=lambda k: (wins[k]["ep_index"], wins[k]["ws"])):
            by_ep.setdefault(wins[k]["ep_index"], []).append(k)
        cur, cost, n = [], 0.0, 0
        for ep in sorted(by_ep):
            for k in by_ep[ep]:
                cur.append([k, list(tasks[k])])
                cost += est_cost(tasks[k])
            if cost >= CHUNK_TARGET_S:
                chunks.append({"id": f"{st}-{n:03d}", "stage": st, "tasks": cur, "est_s": cost})
                cur, cost, n = [], 0.0, n + 1
        if cur:
            chunks.append({"id": f"{st}-{n:03d}", "stage": st, "tasks": cur, "est_s": cost})
    return chunks


def stage_select(a) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[brake_reroll] set CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    seals = check_seals()
    import brake_locus_r8 as BL
    import cruise_brake_retime_r8 as CB
    T = BL.load_tree(f"{BL.ARCHIVE}/{BL.TREE}", 0, True)
    z = np.load(DEP / f"DEPLOY8_POS_{TREE}_s0.npz", allow_pickle=True)
    comp = np.asarray(z["S6_compliant"], bool)
    # ---- alignment controls: the deploy npz is the SAME window list as the banked tree (clip int == episode_index)
    assert np.array_equal(np.asarray(z["S6_ws"]), T["ws"]) and np.array_equal(np.asarray(z["S6_clip"]), T["ep_index"]), "S6 window list != deploy npz"
    ep6, ws6 = T["ep_index"], T["ws"]
    gtc = CB.plan_class_thr(T["g"], T["v0"], 1.0, 1.5)
    c8 = CB.plan_class_thr(T["os"], T["v0"], 1.0, 1.5)
    m = comp & (gtc == "cruise")
    fb = m & (c8 == "decel")
    ctp = m & ~fb
    dcp = comp & (gtc == "decel")
    assert int(m.sum()) == LIT["n_cruise"] and int(fb.sum()) == LIT["n_fb"] and int(ctp.sum()) == LIT["n_ct_pop"], f"literals: {m.sum()} {fb.sum()} {ctp.sum()}"
    rng = np.random.default_rng(SEL_SEED)
    fb_rows = sorted(np.nonzero(fb)[0].tolist())
    ct_rows = pick(ctp, N_CT, ep6, rng)
    dc_rows = pick(dcp, N_DC, ep6, rng)
    fbm = np.zeros(len(ws6), bool)
    fbm[fb_rows] = True
    fb_b = pick(fbm, N_FB_B, ep6, rng)
    ctm = np.zeros(len(ws6), bool)
    ctm[ct_rows] = True
    ct_b = pick(ctm, N_CT_B, ep6, rng)
    # ---- clip ids: the S2 dump manifest maps episode_index -> clip_id (brake_locus_r8.join_extras does the same); S6 manifest must agree
    man = FL.jload(f"{ARCH}/dump_s0/manifest.json")
    cid = {int(e["episode_index"]): str(e["clip_id"]) for e in man["episodes"]}
    assert all(cid[int(e)] == str(c) for e, c in zip(ep6[:2000], T["clip"][:2000])), "clip ids: S6 manifest != S2 dump manifest"
    # ---- ALONG-RAND sample: 5 per episode of the S2 compliant windows (A1 A3)
    s2c = np.asarray(z["S2_compliant"], bool)
    ep2, ws2 = np.asarray(z["S2_clip"]), np.asarray(z["S2_ws"])
    along_rows = []
    for e in sorted(set(ep2[s2c].tolist())):
        idx = np.nonzero(s2c & (ep2 == e))[0]
        idx = list(idx)
        rng.shuffle(idx)
        along_rows += sorted(int(i) for i in idx[:ALONG_PER_EP])
    # ---- windows
    wins = {}

    def add(clip, ws, ep, setname, row6=-1, row2=-1):
        k = f"{clip}|{int(ws)}"
        w = wins.setdefault(k, {"key": k, "clip": clip, "ws": int(ws), "ep_index": int(ep), "sets": [], "row6": -1, "row2": -1})
        if setname not in w["sets"]:
            w["sets"].append(setname)
        w["row6"] = max(w["row6"], row6)
        w["row2"] = max(w["row2"], row2)
        return k
    s2key = {(int(e), int(w)): i for i, (e, w) in enumerate(zip(ep2, ws2))}
    for name, rows in (("FB", fb_rows), ("CT", ct_rows), ("DC", dc_rows)):
        for i in rows:
            add(str(T["clip"][i]), ws6[i], ep6[i], name, row6=int(i), row2=s2key.get((int(ep6[i]), int(ws6[i])), -1))
    for name, rows in (("FB250", fb_b), ("CT100", ct_b)):
        for i in rows:
            add(str(T["clip"][i]), ws6[i], ep6[i], name, row6=int(i))
    for i in dc_rows:
        add(str(T["clip"][i]), ws6[i], ep6[i], "DC100", row6=int(i))
    for i in along_rows:
        add(cid[int(ep2[i])], ws2[i], ep2[i], "ALONG", row2=int(i))
    n6_in_s2 = sum(1 for w in wins.values() if w["row6"] >= 0 and w["row2"] >= 0)
    n6 = sum(1 for w in wins.values() if w["row6"] >= 0)
    # ---- stage tasks
    S = lambda name: [k for k, w in wins.items() if name in w["sets"]]
    s6sets = ("FB", "CT", "DC")
    tasks = {"A": {k: (["N", "N_s1"] if "ALONG" in w["sets"] else ["N"]) for k, w in wins.items() if any(s in w["sets"] for s in s6sets)},
             "AL": {k: ["N", "N_s1"] for k, w in wins.items() if not any(s in w["sets"] for s in s6sets)}}
    bh = {k: ["H1", "H2", "N2"] for k, w in wins.items() if any(s in w["sets"] for s in ("FB250", "CT100", "DC100"))}
    tasks["BH"] = bh
    tasks["BF"] = {**{k: ["F2"] for k in S("FB250")}, **{k: ["Fbrake"] for k in S("CT100")}}
    fbct = S("FB250") + S("CT100")
    ctl = set()
    rr = np.random.default_rng(SEL_SEED + 1)
    for grp in (S("FB250"), S("CT100")):
        gk = sorted(grp)
        rr.shuffle(gk)
        ctl |= set(gk[:N_S0CTL // 2])
    tasks["BS"] = {k: (["S0a", "S0b"] + (["S0id", "S0x"] if k in ctl else [])) for k in fbct}
    chunks = build_chunks(wins, tasks)
    SCR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {"sel_seed": SEL_SEED, "seals": seals, "windows": wins, "chunks": chunks, "stage_order": list(STAGES), "s0_control_keys": sorted(ctl),
            "along_seed": ALONG_SEED}
    FL.jdump(plan, SCR / "plan_raw.json")
    pub = {"sel_seed": SEL_SEED, "seals": seals, "tier": TIER,
           "windows": [{"clip_sha12": sha12(w["clip"]), "ws": w["ws"], "ep_index": w["ep_index"], "sets": w["sets"]} for w in wins.values()],
           "chunks": [{"id": c["id"], "stage": c["stage"], "n_windows": len(c["tasks"]), "est_s": round(c["est_s"]), "arms": sorted({x for _k, ar in c["tasks"] for x in ar})} for c in chunks]}
    FL.jdump(pub, OUT / "plan_sha12.json")
    st = {"tool": "brake_locus_reroll_r8.py select", "generated": time.strftime("%FT%T%z"), "n_windows_unique": len(wins), "n_S6_windows": n6, "n_S6_also_S2": n6_in_s2,
          "sets": {s: len(S(s)) for s in ("FB", "CT", "DC", "FB250", "CT100", "DC100", "ALONG")}, "n_S0_controls": len(ctl),
          "chunks": {s: sum(1 for c in chunks if c["stage"] == s) for s in STAGES},
          "ESTIMATED_cpu_hours": {s: round(sum(c["est_s"] for c in chunks if c["stage"] == s) / 3600.0, 2) for s in STAGES},
          "basis": "COST_S: native ~10 s, each extra arm ~5 s (force_lon HANDOFF, ESTIMATED not measured here)",
          "literals_checked": LIT, "note": "raw clip ids are in plan_raw.json under E: only; plan_sha12.json is sha12-keyed"}
    st["ESTIMATED_cpu_hours"]["total"] = round(sum(st["ESTIMATED_cpu_hours"].values()), 2)
    FL.jdump(st, OUT / "select_stats.json")
    print(json.dumps(st, indent=1))
    print("ZZBLRSELECTOKZZ")
    return 0


# ====================================================================================================== bank sidecars (chunked npz + resumable progress)
def bank_dir(tag, chunk) -> Path:
    return SCR / f"bank_{tag}" / chunk


def load_progress(tag, chunk) -> dict:
    p = bank_dir(tag, chunk) / "progress.json"
    return FL.jload(p) if p.exists() else {"parts": [], "done": {}}


def flush_part(tag, chunk, prog, buf) -> None:
    """buf = [(key, rec)].  npz first (tmp + replace), progress.json second: a kill in between leaves an orphan npz that is ignored and redone."""
    if not buf:
        return
    d = bank_dir(tag, chunk)
    d.mkdir(parents=True, exist_ok=True)
    k = len(prog["parts"])
    Z = {"keys": np.asarray([key for key, _ in buf]), "win__head_logits": np.asarray([r["win"]["head_logits"] for _, r in buf], np.float32),
         "win__own6": np.asarray([r["win"]["own6"] for _, r in buf], np.int8)}
    arms = sorted({a for _, r in buf for a in r["arms"]})
    for a in arms:
        recs = [(key, r["arms"][a]) for key, r in buf if a in r["arms"]]
        Z[f"{a}__keys"] = np.asarray([key for key, _ in recs])
        for f in recs[0][1]:
            Z[f"{a}__{f}"] = np.asarray([rc[f] for _, rc in recs])
    fn = f"part{k:04d}.npz"
    tmp = d / (fn + ".tmp.npz")
    np.savez(tmp, **Z)
    os.replace(tmp, d / fn)
    prog["parts"].append({"file": fn, "n": len(buf), "arms": arms})
    for key, r in buf:
        prog["done"][key] = {"part": fn, "arms": sorted(r["arms"])}
    FL.jdump(prog, d / "progress.json")


def load_bank(tag, chunks=None) -> dict:
    """-> {arm: {field: array [n, ...], 'keys': array}} concatenated over every chunk listed in the progress files (orphan npz files are ignored)."""
    out = {}
    root = SCR / f"bank_{tag}"
    for cd in sorted(root.iterdir()) if root.exists() else []:
        if chunks and cd.name not in chunks:
            continue
        pj = cd / "progress.json"
        if not pj.exists():
            continue
        for part in FL.jload(pj)["parts"]:
            z = np.load(cd / part["file"], allow_pickle=False)
            for a in part["arms"]:
                tgt = out.setdefault(a, {})
                for f in [k[len(a) + 2:] for k in z.files if k.startswith(a + "__")]:
                    tgt.setdefault(f, []).append(z[f"{a}__{f}"])
    return {a: {f: np.concatenate(v) for f, v in fs.items()} for a, fs in out.items()}


# ====================================================================================================== roll (ONE chunk per process)
def row0(o, k, torch):
    t = o[k]
    if k == "r8_kind":
        return t.detach().cpu().numpy()
    return t[0].detach().float().cpu().numpy() if t.is_floating_point() else t[0].detach().cpu().numpy()


FIELD_KEYS = (("fan", "anchor_traj", np.float16), ("sel_score", "sel_score", np.float32), ("sel_score_v3", "sel_score_v3", np.float32),
              ("lon6", "r8_lon6", np.int8), ("lat3", "r8_lat3", np.int8), ("kind", "r8_kind", np.int8), ("cons_raw", "r8_cons_raw", np.float32),
              ("cons_valid", "r8_cons_valid", np.float32), ("reach_keep", "reach_keep", np.int8), ("bank", "anchor_bank", np.float16),
              ("prior_path", "residual_prior_path", np.float32), ("hyp_alloc", "r8_hyp_alloc", np.int8), ("emit_keep", "r8_emit_keep", np.int8),
              ("ceil_keep", "ceil_keep", np.int8))


def capture(o, torch, wall) -> dict:
    r = {"traj": row0(o, "traj", torch).astype(np.float32), "sel_idx": np.int32(int(o["sel_idx"][0])), "wall_s": np.float32(wall)}
    r["sel_idx_base"] = np.int32(int(o["sel_idx_base"][0])) if "sel_idx_base" in o else np.int32(-1)
    for name, key, dt in FIELD_KEYS:
        if key in o and torch.is_tensor(o[key]):
            r[name] = row0(o, key, torch).astype(dt)
    return r


class Hedge:
    """The allocation hedge patch (AMENDMENT 3): `refcv8_conditioning.allocate` is wrapped in THIS process only; identity unless a mode is set for one forward."""
    mode = None
    calls = 0
    forced = None

    @classmethod
    def install(cls, torch):
        import tanitad.refs.refcv8_conditioning as C8
        if getattr(C8.allocate, "_brake_locus_wrapped", False):
            return
        assert C8.LON6[FL.CRUISE6] == "CRUISE" and C8.LON6_V7_IDS[FL.CRUISE6] == 1, "CRUISE index drifted"
        orig = C8.allocate

        def allocate(logp_joint, m_total, top_k=4, min_each=2, force=None, force_min=4):
            mode = cls.mode
            if mode is None:
                return orig(logp_joint, m_total, top_k, min_each, force=force, force_min=force_min)
            cls.calls += 1
            lj = logp_joint.detach().to(torch.float64)
            if mode == "H2":                                              # posterior ** 0.5, renormalised
                return orig(torch.log_softmax(0.5 * lj, dim=-1), m_total, top_k, min_each, force=force, force_min=force_min)
            if mode == "H1":                                              # training's force + force_min 4, GT replaced by (own argmax lat, CRUISE)
                if force is not None:
                    raise RuntimeError("[brake_reroll] H1 refuses: a force is already set on this forward")
                b = lj.shape[0]
                lat = torch.logsumexp(lj.reshape(b, C8.N_LAT3, C8.N_LON6), dim=-1).argmax(-1)
                f = lat * C8.N_LON6 + FL.CRUISE6
                cls.forced = [int(x) for x in f.tolist()]
                return orig(logp_joint, m_total, top_k, min_each, force=f, force_min=4)
            raise RuntimeError(f"unknown hedge mode {mode}")
        allocate._brake_locus_wrapped = True
        C8.allocate = allocate


def r8_sel_module(model):
    mods = [m for n, m in model.named_modules() if n.endswith("r8_sel") and hasattr(m, "gamma_prog")]
    if len(mods) != 1:
        raise RuntimeError(f"[brake_reroll] expected exactly one r8_sel module, found {len(mods)}")
    return mods[0]


def stage_roll(a) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[brake_reroll] set CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    if os.environ.get("R8_NO_POSE_SYNC", "") not in ("1", "true", "yes"):
        raise SystemExit("[brake_reroll] R8_NO_POSE_SYNC=1 is required (the banked tree is the STANDALONE refcv7-comparable clock)")
    seals = check_seals()
    if a.smoke:
        tag, chunk_id = ("smoke" if a.tag == "full" else a.tag), "SMOKE"
        plan_w, tasks = smoke_tasks(a.smoke, a.smoke_arms.split(",") if a.smoke_arms else None)
    else:
        tag, chunk_id = a.tag, a.chunk
        plan = FL.jload(SCR / "plan_raw.json")
        plan_w = plan["windows"]
        ch = [c for c in plan["chunks"] if c["id"] == chunk_id]
        if len(ch) != 1:
            raise SystemExit(f"[brake_reroll] unknown chunk {chunk_id}")
        tasks = ch[0]["tasks"]
    prog = load_progress(tag, chunk_id)
    todo = [(k, ar) for k, ar in tasks if k not in prog["done"]]
    print(f"ZZBLRSTARTZZ chunk={chunk_id} tag={tag} tasks={len(tasks)} done_already={len(tasks) - len(todo)} todo={len(todo)} t={time.strftime('%FT%T')}", flush=True)
    if not todo:
        print("ZZBLRDONEZZ nothing to do", flush=True)
        return 0
    import r8_battery_patch as BP
    BP.apply_child()
    import roll_seed_r7 as RS                          # noqa: F401
    import run_battery_r7 as RB
    import r7_roll as RR
    import torch
    L = RR.L
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
    _orig = L.build_model

    def _fp32(*aa, **kk):                              # exactly roll_seed_r7's --smoke-cpu-fp32-trunk (as force_lon)
        m, c, ar, r = _orig(*aa, **kk)
        for mod in m.modules():
            lv = getattr(mod, "memory_levers", None)
            if isinstance(lv, dict) and "bf16" in lv:
                lv["bf16"] = False
                lv["channels_last"] = False
        r["smoke_note"] = "trunk bf16/NHWC OFF (CPU fp32 trunk)"
        return m, c, ar, r
    L.build_model = _fp32
    from tanitad.refs import refcv8_conditioning as C8
    assert tuple(C8.LON6) == FL.LON6 and tuple(C8.LON6_V7_IDS) == FL.LON6_V7_IDS, "LON6 mapping drifted"
    Hedge.install(torch)
    W = 8
    clip_ids = RB._clip_ids_by_index(RB.KIT_EVAL)
    ci = {c: i for i, c in enumerate(clip_ids)}
    windows = {}
    for k, _ar in todo:
        windows.setdefault(plan_w[k]["clip"], []).append(int(plan_w[k]["ws"]))
    windows = {c: sorted(v) for c, v in windows.items()}
    st = RR.install_patches(config_path=CONFIG, windows=windows, frame_memo=True, obedience=False, vmaxzero=False, with_navflip=True, filteroff=False)
    dump_dir = SCR / f"dump_{tag}_{chunk_id}"
    dump_dir.mkdir(parents=True, exist_ok=True)
    by_key = {}
    for k, ar in todo:
        w = plan_w[k]
        by_key[(ci[w["clip"]], int(w["ws"]) - (W - 1))] = (k, ar)
    state = {"n_done": 0, "n_todo": len(todo), "t0": time.time(), "buf": []}
    _Base = RR.Refcv6Proxy

    class RerollProxy(_Base):
        def __call__(self, frames, nav_cmd=None, v0=None, steps=0, ego_state=None, **kw):
            d = self.__dict__
            if nav_cmd is None:
                raise RuntimeError("[brake_reroll] unexpected nav-null call (the roll is configured --no-navzero)")
            key = tuple(self.ctx.index)
            if key not in by_key:
                raise RuntimeError(f"[brake_reroll] window {key} has no task")
            wkey, arms = by_key[key]
            model, n, dev = d["model"], int(frames.shape[0]), frames.device
            t_w = time.time()
            rec = {"win": {}, "arms": {}}
            Hedge.mode = None
            out = None
            rec["win"] = {"head_logits": np.full(8, np.nan), "own6": -1}
            if "N" in arms:
                torch.manual_seed(FL.window_seed(FL.INFER_SEED, key[0], key[1]))
                out = _Base.__call__(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **dict(kw))   # ARM N (wrapper installed, mode None)
                ex = d["extras"][key]["nav_true"]
                lg = np.asarray(ex["tacv6_lon_logits"], dtype=np.float64)
                rec["win"] = {"head_logits": lg, "own6": int(np.argmax(lg[list(FL.LON6_V7_IDS)]))}
                rec["arms"]["N"] = capture(out, torch, time.time() - t_w)
            for arm in arms:
                if arm == "N":
                    continue
                off, kind, arg = ARM_SPECS[arm]
                kw2 = dict(kw)
                kw2.update(self._inputs(n, dev))           # ego_poses is one-shot: re-supplied for every call
                torch.manual_seed(FL.window_seed(FL.INFER_SEED + off, key[0], key[1]))
                t_a = time.time()
                extra = {}
                saved = None
                try:
                    if kind == "force":
                        model.set_r8_force(lon6=int(arg))
                    elif kind == "hedge":
                        Hedge.mode, Hedge.calls, Hedge.forced = arg, 0, None
                    elif kind == "s0":
                        sel = r8_sel_module(model)
                        names = ("beta_lat", "beta_lon", "gamma_prog", "gamma_head")
                        saved = {nm: getattr(sel, nm).detach().clone() for nm in names}
                        extra["s0_before"] = np.asarray([float(saved[nm]) for nm in names], np.float64)
                        with torch.no_grad():
                            if arg == "a":
                                sel.gamma_prog.zero_()
                            elif arg == "b":
                                for nm in names:
                                    getattr(sel, nm).zero_()
                            elif arg == "x":
                                for nm in names:
                                    getattr(sel, nm).mul_(10.0)
                        extra["s0_during"] = np.asarray([float(getattr(sel, nm)) for nm in names], np.float64)
                    with torch.no_grad():
                        o = model(frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **kw2)
                finally:
                    Hedge.mode = None
                    if saved is not None:
                        with torch.no_grad():
                            for nm, v in saved.items():
                                getattr(sel, nm).copy_(v)
                        if not all(bool(torch.equal(getattr(sel, nm).detach(), v)) for nm, v in saved.items()):
                            raise RuntimeError("[brake_reroll] r8_sel scalars were NOT restored")
                if kind == "force" and getattr(model, "_r8_force", None) is not None:
                    raise RuntimeError(f"[brake_reroll] arm {arm}: the force was NOT consumed by the forward (window {key})")
                if kind == "hedge" and Hedge.calls < 1:
                    raise RuntimeError(f"[brake_reroll] arm {arm}: the allocation wrapper was NOT called (window {key})")
                r = capture(o, torch, time.time() - t_a)
                if kind == "force":
                    r["forced6"] = np.int32(int(arg))
                if kind == "hedge":
                    r["hedge_calls"] = np.int32(Hedge.calls)
                    if arg == "H1":
                        r["hedge_forced_joint"] = np.int32(Hedge.forced[0])
                r.update(extra)
                rec["arms"][arm] = r
                if out is None:
                    out = o                                    # no base forward in this stage: the dump flow gets the first arm's output (the dump is unused)
            rec["wall_s"] = time.time() - t_w
            state["buf"].append((wkey, rec))
            state["n_done"] += 1
            if len(state["buf"]) >= int(a.save_every) or state["n_done"] == state["n_todo"]:
                flush_part(tag, chunk_id, prog, state["buf"])
                state["buf"] = []
                el = time.time() - state["t0"]
                print(f"ZZBLRPROGZZ {chunk_id} {state['n_done']}/{state['n_todo']} elapsed={el:.0f}s eta={el / state['n_done'] * (state['n_todo'] - state['n_done']):.0f}s "
                      f"last_window_wall={rec['wall_s']:.1f}s", flush=True)
            return out

    RR.Refcv6Proxy = RerollProxy
    argv = ["--ckpt", CKPT, "--config", CONFIG, "--episodes", RB.KIT_EVAL, "--labels", RB.KIT_LABELS, "--nav-source", "v72", "--grid", "2s", "--action-units", "steer",
            "--window-stride", "1", "--with-oracle-sel", "--dump-dir", str(dump_dir), "--dump-only", "--out", str(dump_dir / "_unused.json"), "--device", "cpu",
            "--infer-seed", str(FL.INFER_SEED), "--seed", "0", "--arm", "refcv7", "--lru", "8", "--with-navflip", "--no-navzero"]
    R = RR.ra3()
    t0 = time.time()
    R.main(argv)
    if state["buf"]:
        flush_part(tag, chunk_id, prog, state["buf"])
        state["buf"] = []
    FL.jdump({"tool": "brake_locus_reroll_r8.py roll", "chunk": chunk_id, "tag": tag, "ckpt": CKPT, "device": "cpu", "trunk": "fp32 (bf16/NHWC off)", "omp": os.environ.get("OMP_NUM_THREADS"),
              "n_windows_done": len(prog["done"]), "wall_s": round(time.time() - t0, 1), "seals": seals, "script_sha256": sha256_file(__file__),
              "finished": time.strftime("%FT%T")}, bank_dir(tag, chunk_id) / "meta.json")
    print(f"ZZBLRDONEZZ chunk={chunk_id} done={len(prog['done'])} wall={time.time() - t0:.0f}s", flush=True)
    return 0


def smoke_tasks(n: int, arms=None):
    """The first n windows of force_lon's banked roll (so N / F2 / N2 can be compared bit-for-bit) with EVERY arm type, a window dict in plan form."""
    plan = FL.jload("E:/refcv8_eval_kit/force_lon_scratch/plan_raw.json")["entries"]
    fl = FL.jload("E:/refcv8_eval_kit/force_lon_scratch/roll_full.json")["windows"]
    pick_e = [e for e in plan if f"{e['clip']}|{e['ws']}" in fl and e["set"] == "FB" and "F2" in fl[f"{e['clip']}|{e['ws']}"]["arms"]][:n]
    wins, tasks = {}, []
    for e in pick_e:
        k = f"{e['clip']}|{e['ws']}"
        wins[k] = {"key": k, "clip": e["clip"], "ws": int(e["ws"]), "ep_index": int(e["ep_index"]), "sets": ["SMOKE"]}
        tasks.append([k, arms or ["N", "N_s1", "N2", "F2", "Fbrake", "H1", "H2", "S0a", "S0b", "S0id", "S0x"]])
    return wins, tasks


# ====================================================================================================== status / next
def chunk_state(plan, tag="full") -> list:
    rows = []
    for c in plan["chunks"]:
        prog = load_progress(tag, c["id"])
        n = sum(1 for k, _ in c["tasks"] if k in prog["done"])
        rows.append({"id": c["id"], "stage": c["stage"], "n": len(c["tasks"]), "done": n, "est_s": c["est_s"]})
    return rows


def stage_status(a) -> int:
    plan = FL.jload(SCR / "plan_raw.json")
    rows = chunk_state(plan, a.tag)
    for s in STAGES:
        rs = [r for r in rows if r["stage"] == s]
        print(f"{s}: chunks {sum(1 for r in rs if r['done'] == r['n'])}/{len(rs)} complete, windows {sum(r['done'] for r in rs)}/{sum(r['n'] for r in rs)}")
    nd, nt = sum(r["done"] for r in rows), sum(r["n"] for r in rows)
    print(f"ZZBLRSTATUSZZ windows_tasks_done={nd}/{nt} chunks_complete={sum(1 for r in rows if r['done'] == r['n'])}/{len(rows)}")
    return 0


def stage_next(a) -> int:
    plan = FL.jload(SCR / "plan_raw.json")
    skip = set(a.skip.split(",")) if a.skip else set()
    for r in chunk_state(plan, a.tag):
        if r["done"] < r["n"] and r["id"] not in skip:
            print(f"ZZBLRNEXTZZ {r['id']} remaining={r['n'] - r['done']}")
            return 0
    print("ZZBLRNEXTZZ NONE")
    return 0

# ====================================================================================================== smoke report
def smoke_report() -> int:
    B = load_bank("smoke")
    if not B:
        raise SystemExit("[brake_reroll] no smoke bank: run `roll --smoke 2` first")
    flr = FL.jload("E:/refcv8_eval_kit/force_lon_scratch/roll_full.json")["windows"]
    R = {"tool": "brake_locus_reroll_r8.py smoke-report", "generated": time.strftime("%FT%T%z"), "arms": {}, "checks": {}}
    for a, fs in B.items():
        R["arms"][a] = {"n_windows": int(len(fs["keys"])), "shapes": {f: list(v.shape[1:]) for f, v in fs.items() if f != "keys"},
                        "dtypes": {f: str(v.dtype) for f, v in fs.items() if f != "keys"}, "mean_wall_s": float(np.mean(fs["wall_s"]))}
    keys = [str(k) for k in B["N"]["keys"]]
    idx = {a: {str(k): i for i, k in enumerate(B[a]["keys"])} for a in B}

    def g(a, f, k):
        return B[a][f][idx[a][k]]
    ck = {}
    # (1) the selected candidate of the re-rolled fan == the re-rolled emitted plan; and sel_idx == argmax of the final blended score over reach survivors
    sel_err, argmax_ok = [], []
    for a in B:
        for k in [str(x) for x in B[a]["keys"]]:
            fan, tj, si = g(a, "fan", k).astype(np.float64), g(a, "traj", k).astype(np.float64), int(g(a, "sel_idx", k))
            sel_err.append((a, float(np.abs(fan[si] - tj).max()), float(np.abs(tj).max())))
            sc = g(a, "sel_score_v3", k).astype(np.float64) if "sel_score_v3" in B[a] else g(a, "sel_score", k).astype(np.float64)
            rk = g(a, "reach_keep", k).astype(bool)
            argmax_ok.append((a, bool(int(np.argmax(np.where(rk, sc, -np.inf))) == si)))
    tol = lambda mx: 6e-4 * max(1.0, mx) + 1e-3                                # fp16 storage of the fan (11-bit mantissa)
    ck["selected_candidate_equals_emitted_plan"] = {"max_abs_diff_m": max(e for _a, e, _m in sel_err), "fp16_tolerance_rule": "6e-4 * max(1,|plan|max) + 1e-3 m",
                                                    "PASS": bool(all(e <= tol(m) for _a, e, m in sel_err)), "n_checked": len(sel_err)}
    ck["sel_idx_equals_argmax_blended_over_reach_keep"] = {"PASS": bool(all(ok for _a, ok in argmax_ok)), "per_arm_fail": sorted({a for a, ok in argmax_ok if not ok}), "n_checked": len(argmax_ok)}
    # (2) bit-equality with force_lon's banked arms on the same windows (same CPU, same code path, same per-window seed)
    eq = {}
    for a, fa in (("N", "N"), ("F2", "F2"), ("N2", "N2")):
        d = []
        for k in keys:
            r = flr.get(k, {}).get("arms", {}).get(fa)
            if r is not None and k in idx[a]:
                d.append(float(np.abs(np.asarray(r["traj"], np.float64) - g(a, "traj", k).astype(np.float64)).max()))
        eq[a] = {"n_compared": len(d), "max_abs_diff_m_vs_force_lon": (max(d) if d else None), "note": "force_lon ran OMP 8, this roll OMP " + str(os.environ.get("OMP_NUM_THREADS")) + " (fp32 reduction order may differ)"}
    ck["vs_force_lon_banked_arms"] = eq
    # (3) interventions
    s0 = {}
    for a in ("S0id", "S0a", "S0b", "S0x"):
        if a in B:
            s0[a] = {"max_abs_diff_traj_vs_N_m": float(max(np.abs(g(a, "traj", k).astype(np.float64) - g("N", "traj", k).astype(np.float64)).max() for k in keys)),
                     "sel_idx_changed_vs_N": int(sum(int(g(a, "sel_idx", k)) != int(g("N", "sel_idx", k)) for k in keys)), "n": len(keys),
                     "scalars_before": g(a, "s0_before", keys[0]).tolist(), "scalars_during": g(a, "s0_during", keys[0]).tolist()}
    ck["S0_arms"] = s0
    hed = {}
    for a in ("N", "H1", "H2"):
        if a in B:
            hed[a] = {"alloc_hist_by_joint_window0": np.bincount(g(a, "hyp_alloc", keys[0]).astype(np.int64), minlength=18).tolist()}
    if "H1" in B:
        hed["H1"]["forced_joint_window0"] = int(g("H1", "hedge_forced_joint", keys[0]))
        hed["H1"]["forced_joint_count_in_alloc_ge_4_all_windows"] = bool(all(int((g("H1", "hyp_alloc", k).astype(np.int64) == int(g("H1", "hedge_forced_joint", k))).sum()) >= 4 for k in keys))
    for a in ("H1", "H2", "F2", "Fbrake", "N2", "N_s1"):
        if a in B:
            hed.setdefault(a, {})["traj_changed_vs_N_windows"] = int(sum(float(np.abs(g(a, "traj", k).astype(np.float64) - g("N", "traj", k).astype(np.float64)).max()) > 1e-4 for k in keys))
    ck["hedge_and_force_arms"] = hed
    R["checks"] = ck
    R["fields_missing_in_N"] = [f for f in [n for n, _k, _d in FIELD_KEYS] if f not in B["N"]]
    R["per_window_wall_s"] = {"N_incl_base_forward_mean": R["arms"]["N"]["mean_wall_s"], "extra_arm_mean_wall_s": {a: R["arms"][a]["mean_wall_s"] for a in R["arms"] if a != "N"}}
    OUT.mkdir(parents=True, exist_ok=True)
    FL.jdump(R, OUT / "SMOKE_REPORT.json")
    print(json.dumps(R, indent=1))
    ok = ck["selected_candidate_equals_emitted_plan"]["PASS"] and ck["sel_idx_equals_argmax_blended_over_reach_keep"]["PASS"] and not R["fields_missing_in_N"]
    print("ZZBLRSMOKEZZ", "PASS" if ok else "FAIL")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    r = sub.add_parser("roll")
    r.add_argument("--chunk", default="")
    r.add_argument("--tag", default="full")
    r.add_argument("--smoke", type=int, default=0)
    r.add_argument("--smoke-arms", default="")
    r.add_argument("--save-every", type=int, default=10)
    s = sub.add_parser("status")
    s.add_argument("--tag", default="full")
    n = sub.add_parser("next")
    n.add_argument("--tag", default="full")
    n.add_argument("--skip", default="")
    sub.add_parser("smoke-report")
    a = ap.parse_args()
    if a.cmd == "select":
        return stage_select(a)
    if a.cmd == "roll":
        if not a.smoke and not a.chunk:
            raise SystemExit("roll needs --chunk ID or --smoke N")
        return stage_roll(a)
    if a.cmd == "status":
        return stage_status(a)
    if a.cmd == "next":
        return stage_next(a)
    if a.cmd == "smoke-report":
        return smoke_report()
    return 2


if __name__ == "__main__":
    sys.exit(main())
