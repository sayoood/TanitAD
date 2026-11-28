#!/usr/bin/env python3
"""ONE forward pass over navtest -> the scorer logits of SEVERAL fine-tuned scorer arms (+ a mandatory control).

Why: a full navtest pass of refe_navtest_seam.py is ~15,640 s (MEASURED, F:/TanitAD_p6read_work/CHAIN.log line 13). The scorer arms differ from the
P6 model ONLY in the three SCORER modules (scorer_finetune.py: SCORER = score_q_mlp, score_dec, score_head), so the trunk and the proposals are
computed once and every arm's scorer is applied to the SAME inputs.

How (no re-implementation of the model): the planner's own `infer` runs once per token (same loader, same input construction, same Amendment 7 repair,
same driving_command hop as eval/refe_navtest_seam.py). Two forward hooks record exactly what REFe.forward hands its scorer: the input of
`score_q_mlp` (the detached, flattened proposals) and the context `score_dec[0]` cross-attends to. Each arm's three modules (deep copies of the base
modules, loaded from the arm checkpoint) are then applied to those two tensors with the SAME call sequence REFe.forward uses. The CONTROL arm
`P6_control` is the base checkpoint's own scorer through that identical path.

Control (mandatory, fail-fast, checked per token and again at the end):
  (1) INTERNAL: control logits == the planner's own `_score` from the same forward pass, exactly (max |diff| must be 0.0).
  (2) REFERENCE: proposals and logits == proposals_p6final.npz (a seam run with --dump-proposals), exactly, on every token processed.
      The reference was produced ON GPU. A CPU smoke cannot match a GPU run bit for bit, so `--ref-tol X` (default 0.0 = exact) relaxes ONLY this
      comparison and is printed with the verdict; the GPU run must use the default.
Prints ZZARMPASS_OK only when the control passes, else ZZARMPASS_FAIL <reason> (exit 1).

Non-scorer guard: every state key OUTSIDE the three SCORER modules of every arm checkpoint must be bit-equal to the base checkpoint, else the tool
refuses naming the first differing key (an arm that moved the trunk would not be a scorer arm and its logits would not share this forward pass).

Resumable: every --chunk tokens one part file (all arms + control + proposals) goes to OUT/_progress; a restart skips done tokens. The final per-arm
files are written atomically. Output exists + different tokens/checkpoint sha => refuse.

  python arm_logits_pass.py --base-ckpt F:/TanitAD_p6read_work/ckpt/run/model_final.pt --arm S1E=<...>/model_sft_B.pt --out OUT \
      --frames E:/Projects/TanitAD/data/refe_navtest/frames --export <navtest_inputs.json.gz> --tokens <navtest_full_tokens.json> \
      --db-dir E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test [--limit N] [--device cuda|cpu]
(needs the seam's environment: see chain_arm_pass.py, which builds it the way chain_p6.py does and takes the GPU lock.)
"""
from __future__ import annotations

import argparse
import copy
import glob
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

SCORER = ("score_q_mlp", "score_dec", "score_head")      # = scorer_finetune.py line 91
N_OUT = 6
CONTROL = "P6_control"
FORMAT_FULL = "refe-full-v1"                             # = ckpt_io.FORMAT_FULL
DEFAULT_SEAM_DIR = "F:/TanitAD_p6read_work/evalcode/eval"
DEFAULT_REF = "F:/TanitAD_p6read_work/pass2/proposals_p6final.npz"


class ArmRefused(Exception):
    pass


# ---------------------------------------------------------------------------------------------------- checkpoints
def sha256_file(path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_ckpt(path):
    """-> (state dict, meta). Memory-mapped (the box is RAM-bound); FULL format only."""
    try:
        sd = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        sd = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(sd, dict) and "model" in sd):
        raise ArmRefused(f"{path}: not a full checkpoint (no 'model' key)")
    fmt = sd.get("format")
    if fmt is not None and fmt != FORMAT_FULL:
        raise ArmRefused(f"{path}: format {fmt!r}, expected {FORMAT_FULL!r}")
    return sd["model"], (sd.get("meta") or {})


def is_scorer_key(k: str) -> bool:
    return k.split(".", 1)[0] in SCORER


def _bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return torch.equal(a.contiguous(), b.contiguous())


def verify_arm(name: str, base_state: dict, arm_state: dict, arm_meta: dict) -> None:
    """Refuse (ArmRefused, naming the first offending key) unless the arm differs from the base ONLY inside the SCORER modules."""
    bk, ak = list(base_state), set(arm_state)
    for k in bk:
        if k not in ak:
            raise ArmRefused(f"arm {name}: key {k} is missing from the arm checkpoint")
    extra = sorted(ak - set(bk))
    if extra:
        raise ArmRefused(f"arm {name}: key {extra[0]} is not in the base checkpoint")
    for k in bk:
        if is_scorer_key(k):
            continue
        if not _bit_equal(base_state[k], arm_state[k]):
            raise ArmRefused(f"arm {name}: non-scorer key {k} differs from the base checkpoint (not a scorer-only arm)")
    for k in bk:
        if is_scorer_key(k) and (arm_state[k].shape != base_state[k].shape):
            raise ArmRefused(f"arm {name}: scorer key {k} has shape {tuple(arm_state[k].shape)}, base {tuple(base_state[k].shape)}")
    n_out = int(arm_state["score_head.weight"].shape[0])
    declared = arm_meta.get("n_score_components", n_out)
    if n_out != N_OUT or int(declared) != N_OUT:
        raise ArmRefused(f"arm {name}: output count {n_out} (meta {declared}), this tool handles exactly {N_OUT}")


class ScorerArm:
    """Three SCORER modules (deep copies of `model`'s own), optionally loaded from `arm_state`; the call sequence is REFe.forward's."""

    def __init__(self, name: str, model, arm_state: dict | None = None):
        self.name = name
        self.mods = {m: copy.deepcopy(getattr(model, m)).eval() for m in SCORER}
        if arm_state is not None:
            for m in SCORER:
                sub = {k[len(m) + 1:]: v for k, v in arm_state.items() if k.startswith(m + ".")}
                self.mods[m].load_state_dict(sub, strict=True)

    @torch.no_grad()
    def __call__(self, x, ctx):
        s = self.mods["score_q_mlp"](x)
        for blk in self.mods["score_dec"]:
            s = blk(s, ctx)
        return self.mods["score_head"](s)


class Tap:
    """Records what REFe.forward hands its scorer: score_q_mlp's input and the context score_dec[0] reads."""

    def __init__(self, model):
        self.x = self.ctx = None
        self._h = [model.score_q_mlp.register_forward_pre_hook(lambda m, a: setattr(self, "x", a[0])),
                   model.score_dec[0].register_forward_pre_hook(lambda m, a: setattr(self, "ctx", a[1]))]

    def clear(self):
        self.x = self.ctx = None

    def get(self):
        if self.x is None or self.ctx is None:
            raise RuntimeError("the scorer hooks did not fire during the planner's forward pass")
        return self.x, self.ctx


def process_token(planner, ego, img, arms: dict, tap: Tap, to_navsim, ctl_name: str = CONTROL):
    """One trunk+proposal forward (the planner's own `infer`), then every arm's scorer on the SAME tensors.
    -> (proposals [M,8,3] f32, {arm: logits [M,6] f32}, internal control diff, pick, timing dict)"""
    tap.clear()
    t0 = time.time()
    traj, score, k = planner.infer(ego, img)
    t1 = time.time()
    x, ctx = tap.get()
    logits = {n: a(x, ctx)[0].float().cpu().numpy().astype(np.float32) for n, a in arms.items()}
    t2 = time.time()
    own = score.float().cpu().numpy().astype(np.float32)
    diff = float(np.abs(logits[ctl_name] - own).max())
    props = np.stack([to_navsim(planner.executed(traj, j).float().cpu().numpy())
                      for j in range(traj.shape[0])]).astype(np.float32)
    t3 = time.time()
    return props, logits, diff, int(k), {"forward_s": t1 - t0, "arms_s": t2 - t1, "props_s": t3 - t2}


# ---------------------------------------------------------------------------------------------------- reference control
class Reference:
    def __init__(self, path):
        d = np.load(path)
        self.row = {str(t): i for i, t in enumerate(d["token"])}
        self.props, self.logits = d["proposals"], d["logits"]

    def compare(self, tokens, props, logits, tol: float = 0.0):
        """-> dict(ok, n, max_prop_diff, max_logit_diff, props_bit_equal, reason)"""
        missing = [t for t in tokens if str(t) not in self.row]
        if missing:
            return dict(ok=False, n=len(tokens), reason=f"token {missing[0]} is not in the reference", max_prop_diff=None, max_logit_diff=None,
                        props_bit_equal=False)
        idx = [self.row[str(t)] for t in tokens]
        rp, rl = self.props[idx], self.logits[idx]
        pd = float(np.abs(props.astype(np.float64) - rp).max()) if len(tokens) else 0.0
        ld = float(np.abs(logits.astype(np.float64) - rl).max()) if len(tokens) else 0.0
        peq = bool(np.array_equal(props, rp))
        ok = (peq if tol == 0.0 else pd <= tol) and ld <= tol
        reason = "" if ok else f"control vs reference: proposals max|diff| {pd:.3e} (bit-equal {peq}), logits max|diff| {ld:.3e}, tol {tol}"
        return dict(ok=ok, n=len(tokens), reason=reason, max_prop_diff=pd, max_logit_diff=ld, props_bit_equal=peq)


# ---------------------------------------------------------------------------------------------------- resumable store
def tokens_sha(tokens) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, tokens))).encode()).hexdigest()


class Store:
    def __init__(self, out: str, tokens, base_sha: str, arm_shas: dict, chunk: int = 100):
        self.out, self.chunk = out, chunk
        self.dir = os.path.join(out, "_progress")
        self.meta = {"tokens_sha": tokens_sha(tokens), "n_tokens": len(tokens), "base_sha": base_sha, "arms": dict(sorted(arm_shas.items()))}
        self.buf: list = []
        self.n_parts = 0
        self.done: set = set()
        self._check_outputs()
        os.makedirs(self.dir, exist_ok=True)
        mp = os.path.join(self.dir, "meta.json")
        if os.path.exists(mp):
            old = json.load(open(mp, encoding="utf-8"))
            if old != self.meta:
                raise ArmRefused(f"{mp} exists for a different run (tokens/checkpoint shas differ): {old} vs {self.meta}")
        else:
            self._atomic_text(mp, json.dumps(self.meta, indent=1))
        for p in sorted(glob.glob(os.path.join(self.dir, "part_*.npz"))):
            self.n_parts = max(self.n_parts, int(os.path.basename(p)[5:11]) + 1)
            self.done |= set(map(str, np.load(p)["token"]))

    @staticmethod
    def _atomic_text(path, text):
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(path + ".tmp", path)

    def out_path(self, name):
        return os.path.join(self.out, name, "e3_logits.npz")

    def _check_outputs(self):
        for name in list(self.meta["arms"]) + [CONTROL]:
            p = self.out_path(name)
            if not os.path.exists(p):
                continue
            d = np.load(p)
            want_sha = self.meta["base_sha"] if name == CONTROL else self.meta["arms"][name]
            if "tokens_sha" not in d.files or str(d["tokens_sha"]) != self.meta["tokens_sha"] or str(d["ckpt_sha"]) != want_sha:
                raise ArmRefused(f"{p} exists with different tokens or checkpoint sha; refusing to overwrite")

    def add(self, tok, props, logits, diff, pick):
        self.buf.append((str(tok), props, logits, diff, pick))
        self.done.add(str(tok))
        return len(self.buf) >= self.chunk

    def flush(self):
        """write the buffered tokens as one part; returns the buffer's contents (for the per-chunk reference check) and clears it"""
        if not self.buf:
            return None
        names = list(self.buf[0][2])
        d = dict(token=np.array([b[0] for b in self.buf]), proposals=np.stack([b[1] for b in self.buf]),
                 internal_diff=np.array([b[3] for b in self.buf]), pick=np.array([b[4] for b in self.buf], dtype=np.int64))
        for n in names:
            d["L__" + n] = np.stack([b[2][n] for b in self.buf])
        p = os.path.join(self.dir, f"part_{self.n_parts:06d}.npz")
        np.savez(p + ".tmp.npz", **d)
        os.replace(p + ".tmp.npz", p)
        self.n_parts += 1
        self.buf = []
        return d

    def load_all(self):
        parts = [np.load(p) for p in sorted(glob.glob(os.path.join(self.dir, "part_*.npz")))]
        if not parts:
            raise ArmRefused("no progress parts")
        out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
        return out

    def finalize(self, tokens_expected):
        a = self.load_all()
        got = set(map(str, a["token"]))
        miss = set(map(str, tokens_expected)) - got
        if miss:
            raise ArmRefused(f"{len(miss)} requested tokens have no row (e.g. {sorted(miss)[0]})")
        if len(got) != len(a["token"]):
            raise ArmRefused("duplicate tokens in the progress parts")
        for n in [CONTROL] + list(self.meta["arms"]):
            d = os.path.join(self.out, n)
            os.makedirs(d, exist_ok=True)
            p = self.out_path(n)
            np.savez(p + ".tmp.npz", token=a["token"], logits=a["L__" + n], proposals=a["proposals"],
                     tokens_sha=np.array(self.meta["tokens_sha"]),
                     ckpt_sha=np.array(self.meta["base_sha"] if n == CONTROL else self.meta["arms"][n]),
                     base_sha=np.array(self.meta["base_sha"]), arm=np.array(n))
            os.replace(p + ".tmp.npz", p)
        return a


# ---------------------------------------------------------------------------------------------------- the pass
def run_pass(planner, arms: dict, items_factory, tokens, store: Store, ref: Reference, to_navsim, ref_tol: float = 0.0, log=print, chunk_check=True):
    """Loops over (tok, ego, img) from `items_factory(remaining_tokens)`. Returns (verdict str, report dict). Fails fast on the control."""
    tap = Tap(planner.model)
    remaining = [t for t in tokens if str(t) not in store.done]
    log(f"  {len(tokens):,} tokens; {len(store.done):,} done in the progress dir; {len(remaining):,} to run; arms {list(arms)}")
    tm = {"n": 0, "frame_s": 0.0, "forward_s": 0.0, "arms_s": 0.0, "props_s": 0.0}
    worst_internal = 0.0
    ref_cmp = {"n": 0, "max_prop_diff": 0.0, "max_logit_diff": 0.0, "props_bit_equal": True}

    def check_chunk(d):
        if d is None or not chunk_check:
            return None
        r = ref.compare(list(d["token"]), d["proposals"], d["L__" + CONTROL], ref_tol)
        ref_cmp["n"] += r["n"]
        if r["max_prop_diff"] is not None:
            ref_cmp["max_prop_diff"] = max(ref_cmp["max_prop_diff"], r["max_prop_diff"])
            ref_cmp["max_logit_diff"] = max(ref_cmp["max_logit_diff"], r["max_logit_diff"])
        ref_cmp["props_bit_equal"] &= r["props_bit_equal"]
        return None if r["ok"] else r["reason"]

    t_wait = time.time()
    for tok, ego, img in items_factory(remaining):
        tm["frame_s"] += time.time() - t_wait
        props, logits, diff, k, t = process_token(planner, ego, img, arms, tap, to_navsim)
        worst_internal = max(worst_internal, diff)
        if diff != 0.0:
            return f"ZZARMPASS_FAIL internal control: token {tok} control logits differ from the planner's own _score by {diff:.3e} (must be exactly 0.0)", {}
        tm["n"] += 1
        for key in ("forward_s", "arms_s", "props_s"):
            tm[key] += t[key]
        if store.add(tok, props, logits, diff, k):
            bad = check_chunk(store.flush())
            if bad:
                return "ZZARMPASS_FAIL " + bad, {}
            log(f"    {len(store.done):,}/{len(tokens):,} tokens banked")
        t_wait = time.time()
    bad = check_chunk(store.flush())
    if bad:
        return "ZZARMPASS_FAIL " + bad, {}
    a = store.finalize(tokens)
    # the end check, over EVERY token of the final control file (includes tokens banked by an earlier run)
    full = ref.compare(list(a["token"]), a["proposals"], a["L__" + CONTROL], ref_tol)
    rep = {"tokens": len(a["token"]), "arms": list(arms), "internal_control_max_abs_diff": worst_internal, "reference_control": full,
           "ref_tol": ref_tol, "timing_this_run": {**tm, **({"per_token_frame_s": tm["frame_s"] / tm["n"], "per_token_forward_s": tm["forward_s"] / tm["n"],
                                                             "per_token_arms_s": tm["arms_s"] / tm["n"], "per_token_props_s": tm["props_s"] / tm["n"]} if tm["n"] else {})}}
    if not full["ok"]:
        return "ZZARMPASS_FAIL " + full["reason"], rep
    return "ZZARMPASS_OK", rep


# ---------------------------------------------------------------------------------------------------- real-data wiring
def load_seam(seam_dir):
    p = os.path.join(seam_dir, "refe_navtest_seam.py")
    spec = importlib.util.spec_from_file_location("refe_navtest_seam", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["refe_navtest_seam"] = m
    spec.loader.exec_module(m)
    return m


def free_ram_gb():
    import ctypes

    class M(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong)] + \
                   [(c, ctypes.c_ulonglong) for c in "abcde"]
    m = M()
    m.l = ctypes.sizeof(M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.avail / 2 ** 30


def parse_arms(items):
    arms = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--arm wants NAME=path, got {it!r}")
        n, p = it.split("=", 1)
        if n in arms or n == CONTROL or not n or any(c in n for c in "/\\:"):
            raise SystemExit(f"bad or duplicate arm name {n!r}")
        arms[n] = p
    return arms


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--arm", action="append", help="NAME=path/to/model_sft_B.pt (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default="E:/Projects/TanitAD/data/refe_navtest/frames")
    ap.add_argument("--db-dir", default="E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test")
    ap.add_argument("--export", required=True)
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke: only the first N tokens of the (export-ordered, filtered) token list")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone", default="vitl16")
    ap.add_argument("--seam-dir", default=DEFAULT_SEAM_DIR)
    ap.add_argument("--ref", default=DEFAULT_REF)
    ap.add_argument("--ref-tol", type=float, default=0.0, help="0.0 = exact (GPU run). >0 ONLY for a CPU smoke against the GPU-made reference")
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--min-free-gb", type=float, default=5.0, help="wait (never push) while free RAM is below this; 0 = off")
    a = ap.parse_args(argv)
    arm_paths = parse_arms(a.arm)
    torch.set_num_threads(max(1, min(a.threads, 4)) if a.device == "cpu" else a.threads)
    while a.min_free_gb > 0 and free_ram_gb() < a.min_free_gb:
        print(f"  free RAM {free_ram_gb():.1f} GB < {a.min_free_gb}; waiting", flush=True)
        time.sleep(60)

    # 1 checkpoints: shas, then the non-scorer guard for every arm BEFORE any forward run
    base_sha = sha256_file(a.base_ckpt)
    base_state, base_meta = load_ckpt(a.base_ckpt)
    if int(base_state["score_head.weight"].shape[0]) != N_OUT:
        print(f"ZZARMPASS_FAIL base checkpoint has {base_state['score_head.weight'].shape[0]} outputs")
        return 1
    arm_shas, arm_states = {}, {}
    try:
        for n, p in arm_paths.items():
            st, meta = load_ckpt(p)
            verify_arm(n, base_state, st, meta)
            arm_shas[n] = sha256_file(p)
            arm_states[n] = st
        exp = json.load(gzip.open(a.export, "rt", encoding="utf-8"))["tokens"]
        toks = list(exp)
        if a.tokens:
            sub = json.load(open(a.tokens, encoding="utf-8"))
            want = set(sub["tokens"] if isinstance(sub, dict) else sub)
            toks = [t for t in toks if t in want]
        if a.limit:
            toks = toks[:a.limit]
        store = Store(a.out, toks, base_sha, arm_shas, a.chunk)
    except ArmRefused as e:
        print(f"ZZARMPASS_FAIL refused: {e}")
        return 1
    ref = Reference(a.ref)
    print(f"  base {base_sha[:12]}; arms { {n: s[:12] for n, s in arm_shas.items()} }; non-scorer keys bit-equal for every arm; "
          f"tokens {len(toks):,}", flush=True)

    # 2 the planner = the seam's own construction
    seam = load_seam(a.seam_dir)
    import navtrain_scenarios as NS
    from planner import REFePlanner
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
    planner = REFePlanner(checkpoint=a.base_ckpt, images_root=a.frames, db_dir=a.db_dir, backbone=a.backbone, device=a.device, select="best",
                          rule=None, repair_last_heading=None, sanitize_goal=None, goal_fix=None)
    print(f"  planner: trained={planner.trained} per_sample_calib={planner.per_sample_calib} device={planner.device} select={planner.select}", flush=True)
    arms = {CONTROL: ScorerArm(CONTROL, planner.model, None)}
    for n, st in arm_states.items():
        arms[n] = ScorerArm(n, planner.model, st)
    del arm_states
    by_log_all: dict = {}
    for t in toks:
        by_log_all.setdefault(exp[t]["log_name"], []).append(t)

    def items_factory(remaining):
        rem = set(remaining)
        for log, lt in sorted(by_log_all.items()):
            lt = [t for t in lt if t in rem]
            if not lt:
                continue
            db = os.path.join(a.db_dir, f"{log}.db")
            for sc in NS.build_scenarios_for_log(db, lt, history_rows=1, future_rows=80):
                tok = sc._initial_lidar_token
                planner._scenario = sc
                planner.initialize(PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(),
                                                         mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
                ego = sc.get_ego_state_at_iteration(0)
                img = planner._image_for(ego)
                if img is None:
                    raise RuntimeError(f"token {tok}: no frames: {planner.frames.miss_reason}")
                seam.set_driving_command(planner, tok, exp[tok])
                yield tok, ego, img

    t0 = time.time()
    verdict, rep = run_pass(planner, arms, items_factory, toks, store, ref, seam.to_navsim, a.ref_tol, log=lambda m, **k: print(m, **k))
    rep["seconds_this_run"] = round(time.time() - t0, 1)
    rep.update(base_ckpt=a.base_ckpt, base_sha=base_sha, arm_shas=arm_shas, device=str(planner.device), tokens_sha=tokens_sha(toks))
    os.makedirs(a.out, exist_ok=True)
    json.dump(rep, open(os.path.join(a.out, "arm_pass_report.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep, indent=1))
    if a.ref_tol > 0:
        print(f"  NOTE: reference comparison used --ref-tol {a.ref_tol} (cross-device smoke); the internal control was exact 0.0")
    print(verdict)
    return 0 if verdict == "ZZARMPASS_OK" else 1


if __name__ == "__main__":
    sys.exit(main())
