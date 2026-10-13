#!/usr/bin/env python3
"""Tests for the auxiliary drivable-area head's wiring into refe/scorer_finetune.py (the S1 host).  CPU only, no pod, no checkpoint, no GPU, no label store.
Run:  PYTHONIOENCODING=utf-8 python refe/test_aux_s1_wiring.py   ->   ZZTEST_PASS <n>   (prints the mutation table).

WHAT IS PROVEN, AND HOW (CLAUDE.md "built, tested, and unreachable from its caller"):
  * DEFAULT OFF IS BIT-IDENTICAL.  The pre-edit trainer is stored verbatim (refe/fixtures/scorer_finetune_pre_aux.py.txt, sha256 pinned below).  The REAL main() (preflight,
    then the run) of that source and of the edited one are executed on CPU with the same stubs as test_rl_epo.py -- plain, --epo, --epo + a 7-output base -- and every log row
    (minus wall-clock fields and paths), every saved scorer tensor and scorer_states_last.pt are compared BIT FOR BIT.  Controls: the baseline run twice reads EXACTLY equal, and a
    1e-4 perturbation of the learning rate reads DIFFERENT, so a green comparison is a property of the code and not of a blind comparison.
  * THE HEAD IS WIRED, not merely importable: with --aux-drv the REAL main() runs against a stub trunk whose context is the 256-wide tensor the real head reads, and a stub
    label source that serves a synthetic SDF per (log, token).  Checked end to end: the flag reaches the loss (aux_bce / aux_huber logged, head parameters move, the head's own
    Adam state exists), the held-out perception read equals an independent recompute row for row, and the data event carries BUILT facts.
  * H1 DOES NOTHING TO THE SCORER: a whole H1 run leaves arm A, arm B, the base and every logged selection metric bit-identical to a head-less run of the same seed.
    H2 starts identical (step 0) and then differs (the feed is reachable).  w_aux = 0 reads exactly zero gradient.
  * Each lever is also broken IN THE SOURCE (the snippet must occur exactly once; a vacuous mutation proves nothing) and the check that owns it must go RED; three runtime switches
    (REFE_AUX_MUTATE_TAP / _DROP_LOSS / _DETACH_HEAD) must turn the preflight's G12 RED and withhold PREFLIGHT_PASS.json.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import types
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aux_drivable_head as AUXREAL  # noqa: E402
import test_rl_epo as TR  # noqa: E402

SRC_PATH = HERE / "scorer_finetune.py"
BASELINE_PATH = HERE / "fixtures" / "scorer_finetune_pre_aux.py.txt"
BASELINE_SHA256 = "b777d7292fb9973b1bd379f4330988b06c4cba378c78e94369062d5883a6febd"          # scorer_finetune.py at git blob fec6174d, before the aux wiring
VOLATILE = {"at", "s_per_update", "hours", "s_per_microbatch", "est_hours_per_epoch", "gpu_mem_gb", "model_A", "model_B", "path"}      # wall clock and file paths
LAB = "D:/stub/label_store"


# ================================================================================================ stubs for the ON path (a 256-wide trunk, a synthetic label store)
def synth_label(log: str, token: str) -> np.ndarray:
    """a deterministic int8 SDF [160, 128] (0.1 m unit, + inside, index [ix, iy]) for a key: a straight corridor of random half-width and offset"""
    rng = np.random.default_rng(zlib.crc32(f"{log}|{token}".encode()))
    half, c = rng.uniform(3.0, 9.0), rng.uniform(-6.0, 6.0)
    ys = -32.0 + (np.arange(128) + 0.5) * 0.5
    q = np.clip(np.round((half - np.abs(ys - c)) / 0.1), -127, 127).astype(np.int8)
    return np.broadcast_to(q[None, :], (160, 128)).copy()


def synth_valid(log: str, token: str) -> bool:
    return zlib.crc32(f"v|{log}|{token}".encode()) % 10 != 0                      # ~10 % of the frames carry no label


class StubSource:
    """AUX.AuxLabelSource without a store on disk: same surface (labels / key_map / manifest / shuffle_seed), synthetic labels, and a record of every key asked for"""
    CALLS: list = []

    def __init__(self, store, pinned_sha256=None, *, verify_shards=True, max_logs=None, shuffle_seed=None, pinned_keys=None):
        self.store, self.shuffle_seed, self.max_logs = store, shuffle_seed, max_logs
        self.manifest = {"store_sha256": AUXREAL.PINNED_STORE_SHA256, "counts": {"stored_keys": AUXREAL.PINNED_STORE_KEYS}}
        self._km = None

    def all_keys(self):
        used = sorted((f"T{i // 4}", f"tok{i}") for i in range(100)) + sorted((f"H{i // 4}", f"tok{i}") for i in range(12))
        pad = [(f"Z{i // 64}", f"pad{i}") for i in range(AUXREAL.PINNED_STORE_KEYS - len(used))]               # the real store has 103,139 keys: the shuffle must cover ALL of them (the gate pins it)
        return used + pad

    def key_map(self):
        if self._km is None:
            ks = self.all_keys()
            self._km = dict(zip(ks, AUXREAL.derangement(ks, self.shuffle_seed)))
        return self._km

    def labels(self, keys):
        StubSource.CALLS.append(list(keys))
        out = np.zeros((len(keys), 160, 128), dtype=np.int8)
        valid = np.zeros(len(keys), dtype=bool)
        for i, (log, tok) in enumerate(keys):
            if self.shuffle_seed is not None:
                log, tok = self.key_map()[(log, tok)]
            if synth_valid(log, tok):
                out[i], valid[i] = synth_label(log, tok), True
        return torch.from_numpy(out), torch.from_numpy(valid)


class StubBlock256(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(256, 256)

    def forward(self, x, ctx):
        return torch.tanh(self.lin(x + ctx.mean(1, keepdim=True)))


class Stub256REFe(nn.Module):
    """REFe's scoring surface with the real names/shapes the head and the feed touch: scene_proj called twice (scene_ctx, then visual_ctx), a 60-wide indexable
    score_q_mlp, score_dec blocks reading a 256-wide context, score_head."""

    def __init__(self, cfg):
        super().__init__()
        self.trunk, self.scene_proj, self.traj_head = nn.Linear(3, 256), nn.Linear(256, 256), nn.Linear(256, 2)
        self.score_q_mlp = nn.Sequential(nn.Linear(60, 256), nn.ReLU(), nn.Linear(256, 256))
        self.score_dec = nn.Sequential(StubBlock256())
        self.score_head = nn.Linear(256, cfg.n_score_components)

    def score_trajectories(self, trajs, sctx):
        s = self.score_q_mlp(trajs.flatten(2))
        for blk in self.score_dec:
            s = blk(s, sctx)
        return self.score_head(s)

    def context(self, img):
        visual = self.trunk(img) * 0.05                              # a small context, as the real head's init sees: no common-mode logit shift, so the initial BCE sits at ln 2
        scene_ctx = self.scene_proj(visual[:, :2])
        visual_ctx = self.scene_proj(visual)
        return scene_ctx, visual_ctx

    def forward(self, img, ego, goal, calib=None, score_extra=None):
        scene_ctx, visual_ctx = self.context(img)
        sctx = visual_ctx.detach()                                  # the real scoring decoder reads the context DETACHED; detach() shares the storage
        return self.traj_head(scene_ctx.mean(1)), None, self.score_trajectories(score_extra, sctx)


class StubTargets256(torch.utils.data.Dataset):
    def __init__(self, targets, images, cfg, calib=None, onpolicy=None):
        self.onpolicy = onpolicy
        tr, ho = TR.StubBank("train", 8, 4), TR.StubBank("held", 8, 4)
        self.rows = [{"log_name": k[0], "token": k[1], "step": k[2], "rank": k[3]} for k in list(tr.by) + list(ho.by)]

    def __len__(self):
        return len(self.rows)

    def sample_inputs(self, i):
        g = torch.Generator().manual_seed(1000 + i)
        # held-out rows (index >= 100) carry a LARGE image, so the untrained head's output depends strongly on WHICH sample it is: a head output paired with another
        # sample's label then changes the held-out metrics (with a near-constant output a misaligned pairing would be invisible).  Train rows stay small (initial BCE ~ ln 2).
        return torch.randn(8, 3, generator=g) * (1.0 if i < 100 else 20.0), torch.randn(8, 20, 3, generator=g) * 2.0

    def __getitem__(self, i):
        r = self.rows[i]
        e = self.onpolicy.by[(r["log_name"], r["token"], int(r["step"]), int(r["rank"]))]
        img, cxy = self.sample_inputs(i)
        return (img, torch.zeros(4), torch.zeros(2), torch.zeros(1), cxy, torch.as_tensor(e[2]), torch.ones(8), torch.full((8,), 5), torch.zeros(0))


def smoke_aux(src: str, td: str, *, tag: str, extra=(), preflight=True, run=True, env=None, max_updates=25, accum=2) -> dict:
    """the REAL main() of `src` on CPU against the 256-wide stub trunk and the synthetic label store -> {codes, rows, tensors, files}"""
    import ckpt_io as real_ckpt_io
    csrc = (src.replace('dev = "cuda"', 'dev = "cpu"').replace("pin_memory=True", "pin_memory=False").replace("num_workers=2", "num_workers=0")
            .replace("n=10000, seed=20260927", "n=40, seed=20260927").replace('torch.autocast("cuda", dtype=torch.bfloat16)', 'torch.autocast("cpu", enabled=False)'))
    sm = TR.load(csrc, f"aux_smoke_{tag}")

    class StubCkptIO:
        FORMAT_FULL = "full"
        config_for_checkpoint = staticmethod(real_ckpt_io.config_for_checkpoint)

        @staticmethod
        def load_for_inference(model, path, map_location="cpu", backbone=None, meta_out=None):
            sd = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(sd["model"], strict=True)
            if meta_out is not None:
                meta_out.update(sd.get("meta", {}) or {})
            return "full"

        @staticmethod
        def trainable_names(model):
            return {n for n, p in model.named_parameters() if p.requires_grad}

        @staticmethod
        def atomic_save(obj, path):
            torch.save(obj, path)
    stub_aux = types.ModuleType("aux_stub")
    stub_aux.__dict__.update(vars(AUXREAL))
    stub_aux.AuxLabelSource = StubSource
    sm.REFe, sm.REFeConfig, sm.ckpt_io, sm.AUX = Stub256REFe, TR.StubCfgFactory, StubCkptIO, stub_aux
    sm.T = types.SimpleNamespace(OnPolicyBank=TR.StubBank, TargetBank=StubTargets256)
    torch.manual_seed(0)
    ck = os.path.join(td, "ck.pt")
    torch.save({"format": "full", "model": Stub256REFe(types.SimpleNamespace(n_score_components=6)).state_dict(), "meta": {}}, ck)
    out = os.path.join(td, "out")
    with open(os.path.join(td, "ho_logs.txt"), "w") as fh:
        fh.write(" ".join(f"H{i}" for i in range(4)))
    args = ["--ckpt", ck, "--targets", "t", "--images", "i", "--calib", "c", "--onpolicy", "train", "--heldout", "held", "--heldout-logs", os.path.join(td, "ho_logs.txt"),
            "--out", out, "--batch", "2", "--accum", str(accum), "--workers", "0", "--max-updates", str(max_updates), "--eval-every", str(max_updates), "--warmup", "2",
            "--lr", "1e-3", "--b-mode", "listnet", "--lam", "0.3", "--pdm-labels-train", "d", "--pdm-labels-heldout", "d", "--require-pdm", *extra]
    saved_argv, saved_cuda = sys.argv, (torch.cuda.synchronize, torch.cuda.max_memory_allocated)
    torch.cuda.synchronize, torch.cuda.max_memory_allocated = (lambda *a, **k: None), (lambda *a, **k: 0)
    saved_env = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    StubSource.CALLS = []
    res: dict = {"out": out, "td": td, "argv": list(args)}
    try:
        for phase, want in (("pre", preflight), ("run", run)):
            if not want:
                continue
            sys.argv = ["scorer_finetune.py"] + args + (["--preflight"] if phase == "pre" else [])
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    res["rc_" + phase] = sm.main()
            except Exception as e:                                                   # noqa: BLE001 -- a crash IS a result (a mutated caller)
                res["rc_" + phase] = f"EXC {type(e).__name__}: {str(e)[:160]}"
            res["out_" + phase] = buf.getvalue()
    finally:
        sys.argv = saved_argv
        torch.cuda.synchronize, torch.cuda.max_memory_allocated = saved_cuda
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    res["calls"] = [k for c in StubSource.CALLS for k in c]
    return harvest(res, out)


def harvest(res: dict, out: str) -> dict:
    rows = {}
    for nm in ("preflight.jsonl", "sft.jsonl"):
        p = os.path.join(out, nm)
        rows[nm] = [json.loads(x) for x in open(p, encoding="utf-8")] if os.path.exists(p) else []
    res["rows"] = rows
    pf = os.path.join(out, "PREFLIGHT_PASS.json")
    res["pass_json"] = json.load(open(pf)) if os.path.exists(pf) else None
    for arm in ("A", "B"):
        p = os.path.join(out, f"model_sft_{arm}.pt")
        res[arm] = torch.load(p, weights_only=False) if os.path.exists(p) else None
    for nm, key in (("aux_drv.pt", "aux"), ("scorer_states_last.pt", "states")):
        p = os.path.join(out, nm)
        res[key] = torch.load(p, weights_only=False) if os.path.exists(p) else None
    res["out_exists"] = os.path.isdir(out)
    return res


def from_tr(r: dict, td: str) -> dict:
    """TR.smoke's result -> the shape harvest() produces (its out dir is <td>/out)"""
    return harvest({"rc_pre": r.get("pre"), "rc_run": r.get("run"), "td": td}, os.path.join(td, "out"))


# ================================================================================================ comparisons
def norm_row(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in VOLATILE}


def rows_of(res: dict, event: str, log="sft.jsonl") -> list:
    return [norm_row(r) for r in res["rows"][log] if r.get("event") == event]


def tensors_equal(x, y) -> bool:
    if x is None or y is None:
        return x is None and y is None
    kx = [k for k in x["model"] if k.split(".")[0] in ("score_q_mlp", "score_dec", "score_head")]
    return bool(kx) and set(kx) == {k for k in y["model"] if k.split(".")[0] in ("score_q_mlp", "score_dec", "score_head")} and all(torch.equal(x["model"][k], y["model"][k]) for k in kx)


def states_equal(x, y) -> bool:
    if x is None or y is None:
        return x is None and y is None
    return x["update"] == y["update"] and all(torch.equal(x[arm][m][k], y[arm][m][k]) for arm in ("A", "B") for m in x[arm] for k in x[arm][m])


def identical_runs(r0: dict, r1: dict) -> list:
    """the names of everything that differs between two runs of the SAME configuration (empty = bit-identical)"""
    bad = []
    if (r0.get("rc_pre"), r0.get("rc_run")) != (r1.get("rc_pre"), r1.get("rc_run")):
        bad.append("exit codes")
    for log in ("preflight.jsonl", "sft.jsonl"):
        if [norm_row(r) for r in r0["rows"][log]] != [norm_row(r) for r in r1["rows"][log]]:
            bad.append(f"{log} rows")
    for arm in ("A", "B"):
        if not tensors_equal(r0[arm], r1[arm]):
            bad.append(f"model_sft_{arm} scorer tensors")
    if not states_equal(r0["states"], r1["states"]):
        bad.append("scorer_states_last.pt")
    if (r0["pass_json"] is None) != (r1["pass_json"] is None):
        bad.append("PREFLIGHT_PASS.json presence")
    return bad


def scorer_identical(r0: dict, r1: dict) -> list:
    """what a head-less run and an H1 run must share: the scorer arms' tensors and every logged selection / loss number (the head adds EXTRA rows and one data block)"""
    bad = []
    for arm in ("A", "B"):
        if not tensors_equal(r0[arm], r1[arm]):
            bad.append(arm)
    s1 = {k: v for k, v in (r1["states"] or {}).items() if k != "AUX"} or None
    if not states_equal(r0["states"], s1):
        bad.append("scorer_states_last")
    for ev, keep in (("train", ("update", "of", "bceA", "lossB", "listnet", "lr")), ("eval", None), ("start", None)):
        a_, b_ = rows_of(r0, ev), rows_of(r1, ev)
        if keep:
            a_, b_ = [{k: r[k] for k in keep} for r in a_], [{k: r[k] for k in keep} for r in b_]
        if a_ != b_:
            bad.append(f"{ev} rows")
    return bad


def gate(res: dict, name="G12_aux_drv_live_and_identity"):
    pre = [r for r in res["rows"]["preflight.jsonl"] if r.get("event") == "preflight"]
    return pre[-1]["gates"].get(name) if pre else None


def data_event(res: dict, log="preflight.jsonl"):
    d = [r for r in res["rows"][log] if r.get("event") == "data"]
    return d[-1] if d else None


# ================================================================================================ the suite
H1_ARGS = ["--aux-drv", "--w-aux-drv", "0.1", "--aux-drv-labels", LAB, "--aux-drv-lr", "2e-4"]
H2_ARGS = H1_ARGS + ["--aux-drv-feed", "on"]
CACHE: dict = {}


def cached(key, fn):
    if key not in CACHE:
        CACHE[key] = fn()
    return CACHE[key]


def tmp():
    return tempfile.mkdtemp(prefix="auxs1_", dir=os.environ.get("TMP") or None)


def get_N():
    """the head-less run of the PRE-EDIT trainer on the 256-wide stub (preflight + run): the comparator of every head-on run"""
    return cached(("N",), lambda: smoke_aux(BASELINE_PATH.read_text(encoding="utf-8"), tmp(), tag="N_base"))


def suite(src: str, only=None) -> list:
    """Every check against the source `src` (or only the checks whose name contains one of `only`).  Returns the list of FAILING check names (empty = green)."""
    bad: list = []
    tag = f"{abs(hash(src)) % 10**7}"
    S = TR.load(src, f"aux_s1_mod_{tag}")
    memo: dict = {}

    def mem(key, fn):
        if key not in memo:
            memo[key] = fn()
        return memo[key]

    def t(name, fn):
        if only and not any(o in name for o in only):
            return
        try:
            ok = bool(fn())
        except Exception as e:                                                       # noqa: BLE001 -- an exception IS a failed check
            ok = False
            print(f"   [{name}] raised {type(e).__name__}: {str(e)[:240]}")
        if not ok:
            bad.append(name)

    def pre(name, extra, env=None):
        return mem(("pre", name), lambda: smoke_aux(src, tmp(), tag=f"{name}_{tag}", extra=extra, preflight=True, run=False, env=env))

    def run(name, extra):
        return mem(("run", name), lambda: smoke_aux(src, tmp(), tag=f"{name}_{tag}", extra=extra))

    ap = S.build_parser()
    base_cli = "--ckpt c --targets t --images i --calib k --onpolicy o --heldout h --heldout-logs l --out x".split()
    ns0 = ap.parse_args(base_cli)

    # ------------------------------------------------------------------ unit
    t("unit/every_aux_flag_defaults_OFF", lambda: (ns0.aux_drv is False and ns0.w_aux_drv == 0.0 and ns0.aux_drv_feed == "off" and ns0.aux_drv_labels == ""
                                                    and ns0.aux_drv_shuffle_seed is None and ns0.aux_drv_lr is None and ns0.aux_drv_tap == "visual" and ns0.aux_drv_depth == 2))
    t("unit/default_argv_has_no_refusal", lambda: S.AUX.aux_refusals(ns0) == [] and S.aux_s1_refusals(ns0) == [])

    def P(extra):
        return ap.parse_args(base_cli + extra.split())
    ok_ns = P(f"--aux-drv --w-aux-drv 0.1 --aux-drv-labels {LAB} --aux-drv-lr 2e-4")
    t("unit/named_head_has_no_refusal", lambda: S.AUX.aux_refusals(ok_ns) == [] and S.aux_s1_refusals(ok_ns) == [])
    for nm, extra, frag, who in (("lr_without_switch", "--aux-drv-lr 2e-4", "--aux-drv-lr is set but --aux-drv is not", "s1"),
                                 ("weight_without_switch", "--w-aux-drv 0.1", "w-aux-drv is set but --aux-drv is not", "aux"),
                                 ("feed_without_switch", "--aux-drv-feed on", "feed is set but --aux-drv is not", "aux"),
                                 ("switch_without_lr", f"--aux-drv --w-aux-drv 0.1 --aux-drv-labels {LAB}", "needs --aux-drv-lr", "s1"),
                                 ("zero_lr", f"--aux-drv --w-aux-drv 0.1 --aux-drv-labels {LAB} --aux-drv-lr 0", "needs --aux-drv-lr", "s1"),
                                 ("switch_without_labels", "--aux-drv --w-aux-drv 0.1 --aux-drv-lr 2e-4", "needs --aux-drv-labels", "aux"),
                                 ("scene_tap", f"--aux-drv --aux-drv-labels {LAB} --aux-drv-lr 2e-4 --aux-drv-tap scene", "scene is not built", "s1"),
                                 ("shuffle_with_feed", f"--aux-drv --aux-drv-labels {LAB} --aux-drv-lr 2e-4 --aux-drv-feed on --aux-drv-shuffle-seed 7", "shuffle-seed", "aux")):
        ns = P(extra)
        t(f"unit/refused/{nm}", lambda ns=ns, frag=frag, who=who: any(frag in m for m in (S.aux_s1_refusals(ns) if who == "s1" else S.AUX.aux_refusals(ns))))

    class FakeDs:
        rows = [{"log_name": "T0", "token": "a"}] * 3

        def __len__(self):
            return 3

        def __getitem__(self, i):
            return (torch.zeros(1),) * 9
    t("unit/make_train_set_OFF_is_the_stock_subset", lambda: type(S.make_train_set(FakeDs(), [0, 1], False)) is torch.utils.data.Subset
      and type(S.make_train_set(FakeDs(), [0, 1], True)) is S.IndexedSubset)
    fd = FakeDs()
    fd.aux_src = StubSource(LAB)
    t("unit/make_train_set_with_a_source_is_AuxSubset", lambda: type(S.make_train_set(fd, [0, 1], False)) is S.AuxSubset and S.make_train_set(fd, [0], True).with_rows is True)
    t("unit/split_aux_is_a_noop_when_off", lambda: S.split_aux(("x",) * 9, False) == (("x",) * 9, None))
    def peel_pairs_each_body_with_its_own_tail():
        bs = [tuple(torch.full((2,), float(k)) for _ in range(9)) + (torch.full((2, 160, 128), k, dtype=torch.int8), torch.tensor([True, k % 2 == 0])) for k in range(5)]
        ld = S.AuxPeelLoader(bs)
        seen = []
        for _epoch in range(2):                                              # a second pass starts a fresh generator: the tail must follow it too
            for body in ld:
                seen.append((len(body), float(body[0][0]), int(ld.tail[0][0, 0, 0]), bool(ld.tail[1][1])))
        return len(ld) == 5 and seen == [(9, float(k), k, k % 2 == 0) for k in range(5)] * 2
    t("unit/AuxPeelLoader_pairs_each_batch_body_with_ITS_OWN_tail_across_epochs", peel_pairs_each_body_with_its_own_tail)
    t("unit/split_aux_peels_the_last_two", lambda: S.split_aux(tuple(range(11)), True) == (tuple(range(9)), (9, 10)) and S.split_aux(tuple(range(12)), True)[0] == tuple(range(10)))

    def labels_follow_the_row():
        ds = StubTargets256(None, None, None, onpolicy=TR.StubBank("train", 8, 4))
        ds.rows = ds.rows[:20]
        for epo in (False, True):
            sub = S.AuxSubset(ds, [5, 3, 11], StubSource(LAB), epo)
            for j, i in enumerate([5, 3, 11]):
                it, r = sub[j], ds.rows[i]
                lab, valid = it[-2], it[-1]
                if synth_valid(r["log_name"], r["token"]):
                    if not (bool(valid) and torch.equal(lab, torch.from_numpy(synth_label(r["log_name"], r["token"])))):
                        return False
                elif bool(valid) or bool(lab.any()):
                    return False
                if len(it) != 9 + int(epo) + 2 or (epo and it[9] != i):
                    return False
            if len(sub.__getitems__([0, 1, 2])) != 3:
                return False
        return True
    t("unit/AuxSubset_serves_the_label_of_ITS_row_keyed_by_log_token_only", labels_follow_the_row)

    def eval_acc_equals_head_metrics():
        g = torch.Generator().manual_seed(3)
        lg, sd = torch.randn(10, 40, 32, generator=g), torch.randn(10, 40, 32, generator=g) * 3
        lab = torch.stack([torch.from_numpy(synth_label("T0", f"t{i}")) for i in range(10)])
        valid = torch.tensor([True] * 8 + [False, True])
        acc = S.AuxEvalAcc()
        for s in range(0, 10, 4):
            acc.add(lg[s:s + 4], sd[s:s + 4], lab[s:s + 4], valid[s:s + 4])
        r, h = acc.result(), AUXREAL.head_metrics(lg, sd, lab, valid)
        return (math.isclose(r["iou"], h["iou"], rel_tol=1e-9) and math.isclose(r["acc"], h["acc"], rel_tol=1e-9) and math.isclose(r["sdf_mae_m"], h["sdf_mae_m"], rel_tol=1e-5)
                and r["n_cells"] == h["n_cells"] and r["labelled"] == 9 and r["frames"] == 10)
    t("unit/AuxEvalAcc_chunked_equals_head_metrics_whole", eval_acc_equals_head_metrics)
    t("unit/scorer_backward_line_is_byte_identical_and_unique", lambda: src.count("((lA + lB) / a.accum).backward()") == 1)

    # ------------------------------------------------------------------ off: the default-OFF path is bit-identical to the pre-edit trainer
    base_src = BASELINE_PATH.read_text(encoding="utf-8")
    t("off/baseline_fixture_is_the_pinned_blob", lambda: hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest() == BASELINE_SHA256)

    def base_run(nm, kw):
        def go():
            td = tmp()
            return from_tr(TR.smoke(base_src, td, tag=f"b_{nm}", **kw), td)
        return cached(("base", nm), go)
    for nm, kw in (("plain", dict(epo=False)), ("epo", dict(epo=True)), ("epo_native7", dict(epo=True, native7=True))):
        def chk(kw=kw, nm=nm):
            rb = base_run(nm, kw)
            td = tmp()
            rn = from_tr(TR.smoke(src, td, tag=f"n_{nm}_{tag}", **kw), td)
            d = identical_runs(rb, rn)
            if d:
                print(f"   off/{nm}: differs in {d}")
            return not d and rb["rc_run"] == 0 and rn["rc_run"] == 0 and rn["rc_pre"] == 0 and len(rn["rows"]["sft.jsonl"]) > 5 and rn["A"] is not None
        t(f"off/default_OFF_is_bit_identical_to_pre_edit_main/{nm}", chk)

    def control_same_source_twice():
        td = tmp()
        r2 = from_tr(TR.smoke(base_src, td, tag=f"b2_{tag}", epo=False), td)
        return not identical_runs(base_run("plain", dict(epo=False)), r2)
    t("off/control_the_baseline_twice_reads_EXACTLY_equal", control_same_source_twice)

    def control_perturbation_is_seen():
        pert = base_src.replace('gr["lr"] = lr_at(upd)', 'gr["lr"] = lr_at(upd) * 1.0001')
        assert pert != base_src
        td = tmp()
        rp = from_tr(TR.smoke(pert, td, tag=f"p_{tag}", epo=False), td)
        return bool(identical_runs(base_run("plain", dict(epo=False)), rp))
    t("off/control_a_1e-4_lr_perturbation_reads_DIFFERENT", control_perturbation_is_seen)

    def refusal_precedes_makedirs():
        sm = TR.load(src.replace('dev = "cuda"', 'dev = "cpu"'), f"aux_ref_{tag}")
        out = os.path.join(tmp(), "never")
        sv, buf = sys.argv, io.StringIO()
        sys.argv = ["scorer_finetune.py", "--ckpt", "c", "--targets", "t", "--images", "i", "--calib", "c", "--onpolicy", "o", "--heldout", "h", "--heldout-logs", "l",
                    "--out", out, "--aux-drv-lr", "2e-4"]
        try:
            with contextlib.redirect_stdout(buf):
                rc = sm.main()
        finally:
            sys.argv = sv
        return rc == 4 and "REFUSED: --aux-drv-lr is set but --aux-drv is not" in buf.getvalue() and not os.path.exists(out)
    t("off/a_refused_command_line_exits_4_before_makedirs", refusal_precedes_makedirs)

    # ------------------------------------------------------------------ pre: the preflight with the head on (G12, the data event's built facts, the regression switches)
    def h1pre():
        return pre("h1pre", H1_ARGS)

    def blk_of(r):
        return (data_event(r) or {}).get("aux_drv") or {}
    t("pre/H1_preflight_passes_and_writes_PASS", lambda: h1pre()["rc_pre"] == 0 and h1pre()["pass_json"] is not None and "ZZSFT_PREFLIGHT_PASS" in h1pre()["out_pre"])
    t("pre/G12_present_and_ok", lambda: gate(h1pre()) is not None and gate(h1pre())["ok"] is True)
    t("pre/G12_head_is_live_in_every_submodule_at_the_declared_weight", lambda: len(gate(h1pre())["per_group_declared"]) >= 5
      and all(v > 0 and math.isfinite(v) for v in gate(h1pre())["per_group_declared"].values()))
    t("pre/G12_weight0_control_reads_EXACTLY_zero_on_the_same_groups", lambda: set(gate(h1pre())["per_group_w0"]) == set(gate(h1pre())["per_group_declared"])
      and all(v == 0.0 for v in gate(h1pre())["per_group_w0"].values()))
    t("pre/G12_tap_is_visual_ctx_storage_not_scene_ctx", lambda: gate(h1pre())["tap_is_visual_ctx"] is True and gate(h1pre())["tap_is_scene_ctx"] is False
      and gate(h1pre())["scene_proj_calls"] == 2)
    t("pre/G12_no_autograd_path_from_score_to_head", lambda: gate(h1pre())["score_to_head_grads_all_none"] is True)
    t("pre/G12_H1_scorer_bit_identical_after_2_steps_with_a_deterministic_reference",
      lambda: all(gate(h1pre())[k] is True for k in ("scorer_params_bit_identical_2steps", "reference_replay_identical", "identity_probe_reads_one_ulp")))
    t("pre/G12_every_head_parameter_holds_Adam_state", lambda: gate(h1pre())["adam_head_params"] > 20 and gate(h1pre())["adam_head_params_with_state"] == gate(h1pre())["adam_head_params"])
    t("pre/G12_initial_bce_near_ln2", lambda: gate(h1pre())["bce_near_ln2"] is True)
    t("pre/data_event_carries_BUILT_aux_facts", lambda: (lambda b: b.get("depth") == 2 and b.get("tap") == "visual" and b.get("feed") == "off" and b.get("w_aux") == 0.1
                                                       and b.get("lr") == 2e-4 and b.get("store_sha256") == AUXREAL.PINNED_STORE_SHA256 and b.get("frames") == AUXREAL.PINNED_STORE_KEYS
                                                       and b.get("scorer_in_width") == 60 and b.get("feed_built") is False and b.get("cell") == 2.0 and b.get("nx") == 40
                                                       and b.get("ny") == 32 and b.get("head_params_in_optimiser") == b.get("n_params") == 2440450)(blk_of(h1pre())))
    t("pre/head_less_run_has_no_aux_block_and_no_G12", lambda: data_event(get_N(), "preflight.jsonl") is not None and "aux_drv" not in data_event(get_N(), "preflight.jsonl")
      and gate(get_N()) is None)
    t("pre/depth_flag_reaches_the_built_head", lambda: (lambda r: r["rc_pre"] == 0 and blk_of(r).get("depth") == 1 and blk_of(r).get("head_params_in_optimiser") < 2440450)(
        pre("d1", H1_ARGS + ["--aux-drv-depth", "1"])))
    t("pre/weight_0_arm_reads_all_gradients_exactly_zero_and_passes", lambda: (lambda r: r["rc_pre"] == 0 and gate(r)["ok"] and all(v == 0.0 for v in gate(r)["per_group_declared"].values())
                                                                           and blk_of(r).get("w_aux") == 0.0)(pre("w0", ["--aux-drv", "--w-aux-drv", "0", "--aux-drv-labels", LAB, "--aux-drv-lr", "2e-4"])))
    t("pre/shuffle_arm_builds_a_derangement_of_all_keys", lambda: (lambda r: r["rc_pre"] == 0 and blk_of(r).get("shuffle_seed") == 7 and blk_of(r).get("derangement_fixed_points") == 0
                                                                 and blk_of(r).get("shuffle_keys") == AUXREAL.PINNED_STORE_KEYS)(pre("sh", H1_ARGS + ["--aux-drv-shuffle-seed", "7"])))
    for nm, env_k, tap in (("tap_copy", "REFE_AUX_MUTATE_TAP", True), ("loss_dropped", "REFE_AUX_MUTATE_DROP_LOSS", False), ("head_detached", "REFE_AUX_MUTATE_DETACH_HEAD", False)):
        def sw(nm=nm, env_k=env_k, tap=tap):
            rm = pre("mut_" + nm, H1_ARGS, env={env_k: "1"})
            gm = gate(rm)
            red = (rm["rc_pre"] == 3 and gm is not None and gm["ok"] is False and rm["pass_json"] is None and "ZZSFT_PREFLIGHT_FAIL" in rm["out_pre"] and env_k in gm["mutate"])
            return red and (gm["tap_is_visual_ctx"] is False if tap else all(v == 0.0 for v in gm["per_group_declared"].values()))
        t(f"pre/regression_switch_{env_k}_turns_G12_RED_and_withholds_PASS", sw)

    # ------------------------------------------------------------------ h1: a whole head-on run, feed off
    def h1():
        return run("h1", H1_ARGS)

    def init_head(depth=2):
        return AUXREAL.build_head(AUXREAL.AuxDrivableConfig(tap="visual", depth=depth, feed="off", w_aux=0.1, seed=0)).state_dict()
    t("h1/run_completes", lambda: h1()["rc_pre"] == 0 and h1()["rc_run"] == 0 and h1()["A"] is not None and h1()["aux"] is not None)
    t("h1/aux_terms_are_logged_finite_at_the_declared_weight", lambda: rows_of(h1(), "train") and all(math.isfinite(x["aux_bce"]) and math.isfinite(x["aux_huber"]) and x["w_aux"] == 0.1
                                                                                                    for x in rows_of(h1(), "train")))
    t("h1/scorer_arms_bit_identical_to_a_head_less_run_of_the_same_seed", lambda: not scorer_identical(get_N(), h1()))
    t("h1/head_parameters_moved_during_training", lambda: h1()["aux"] is not None and sum(1 for k, v in h1()["aux"]["head"].items() if not torch.equal(v, init_head()[k])) >= 20)
    t("h1/head_has_its_own_Adam_state_in_aux_drv_pt", lambda: h1()["aux"] is not None and len(h1()["aux"]["opt"]["state"]) == len(h1()["aux"]["head"]) > 20)
    t("h1/scorer_states_last_carries_the_head", lambda: h1()["states"] is not None and "AUX" in h1()["states"])
    t("h1/head_lr_follows_the_scorer_schedule_at_the_head_peak", lambda: rows_of(h1(), "train") and all(abs(x["aux_lr"] - 1e-3 * 0.5 * (1 + math.cos(math.pi * (24 - 2) / (25 - 2))) * (2e-4 / 1e-3)) <= 1e-12 for x in rows_of(h1(), "train")))
    t("h1/aux_eval_logged_at_step0_and_at_the_last_update", lambda: [x["tag"] for x in rows_of(h1(), "aux_eval")] == ["step0", "update25"] and all(x["labelled"] > 0 for x in rows_of(h1(), "aux_eval")))
    t("h1/meta_aux_drv_in_model_sft_B_while_the_scorer_keys_are_unchanged", lambda: h1()["B"]["meta"].get("aux_drv", {}).get("store_sha256") == AUXREAL.PINNED_STORE_SHA256
      and set(h1()["B"]["model"]) == set(get_N()["B"]["model"]))

    def eval_matches_independent_recompute():
        """the held-out read of the LAST evaluation == the saved head run independently over the same held-out sets and the same labels"""
        r1 = h1()
        m = Stub256REFe(types.SimpleNamespace(n_score_components=6))
        m.load_state_dict(torch.load(os.path.join(r1["td"], "ck.pt"), weights_only=False)["model"])
        head = AUXREAL.build_head(AUXREAL.AuxDrivableConfig(tap="visual", depth=2, feed="off", w_aux=0.1, seed=0))
        head.load_state_dict(r1["aux"]["head"])
        ds = StubTargets256(None, None, None, onpolicy=TR.StubBank("held", 8, 4))
        want_bank = TR.StubBank("held", 8, 4)
        idx = [i for i, r in enumerate(ds.rows) if (r["log_name"], r["token"], r["step"], r["rank"]) in want_bank.by]
        acc, per = S.AuxEvalAcc(), {}
        with torch.no_grad():
            for i in idx:
                img, _ = ds.sample_inputs(i)
                lg, sd = head(m.context(img[None])[1])
                lab, valid = StubSource(LAB).labels([(ds.rows[i]["log_name"], ds.rows[i]["token"])])
                acc.add(lg, sd, lab, valid)
                per.setdefault(ds.rows[i]["log_name"], S.AuxEvalAcc()).add(lg, sd, lab, valid)
        got, want = rows_of(r1, "aux_eval")[-1], acc.result()

        def close(x, y):
            return x["n_cells"] == y["n_cells"] and x["labelled"] == y["labelled"] and x["frames"] == y["frames"] and all(abs(x[k] - y[k]) < 1e-3 for k in ("iou", "acc", "sdf_mae_m"))
        return (close(got, want) and want["frames"] == len(idx) and set(got["per_log"]) == set(per) and all(close(got["per_log"][k], v.result()) for k, v in per.items())
                and 0.05 < want["iou"] < 0.95)
    t("h1/held_out_perception_read_equals_an_independent_recompute_row_for_row", eval_matches_independent_recompute)
    t("h1/labels_were_requested_by_log_and_token_pairs_only", lambda: h1()["calls"] and all(isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], str) for k in h1()["calls"]))

    # ------------------------------------------------------------------ h2: head + feed
    def h2():
        return run("h2", H2_ARGS)
    t("h2/preflight_and_run_complete", lambda: h2()["rc_pre"] == 0 and h2()["rc_run"] == 0 and h2()["aux"] is not None)
    t("h2/G12_feed_fields_ok", lambda: (lambda g: g and g["ok"] and g["feed_zero_init"] and g["feed_step0_equal_base"] and g["feed_probe_reads_difference"] and g["feed_proj_grad_abs"] > 0.0)(gate(h2())))
    t("h2/data_event_scorer_width_63_and_feed_on", lambda: (lambda b: b.get("feed") == "on" and b.get("feed_built") is True and b.get("scorer_in_width") == 63)(blk_of(h2())))
    t("h2/step0_evaluation_is_the_base_scorer_exactly", lambda: (lambda a_, b_: bool(a_) and a_ == b_)([x for x in rows_of(get_N(), "eval") if x["tag"] == "step0"],
                                                                                                  [x for x in rows_of(h2(), "eval") if x["tag"] == "step0"]))
    t("h2/arm_A_untouched_by_the_feed", lambda: tensors_equal(h2()["A"], get_N()["A"]))
    t("h2/arm_B_differs_from_the_head_less_run_so_the_feed_is_reachable", lambda: h2()["B"] is not None and not tensors_equal(h2()["B"], get_N()["B"]))
    t("h2/feed_projection_left_zero", lambda: h2()["aux"] is not None and h2()["aux"]["feed"] is not None and float(h2()["aux"]["feed"]["proj.weight"].abs().max()) > 0.0)
    t("h2/feed_projection_trains_in_the_scorer_optimiser_not_the_head_optimiser", lambda: h2()["aux"] is not None and len(h2()["aux"]["opt"]["state"]) == len(h2()["aux"]["head"]))

    # ------------------------------------------------------------------ gate: the trainer's REAL preflight output through the launch gate's own consumers (G-DVB, G-PREFLIGHT)
    # The static tests in test_launch_gate.py feed the gate fixtures; these feed it what main() actually wrote, so a built-vs-declared drift or a G12 the consumer would accept
    # but the trainer cannot produce shows up HERE, in the trainer's own suite.  Both consumers re-judge raw fields: a mutated run whose verdict and `ok` are forged to PASS is still refused.
    import gate_checks as GC
    import launch_gate as LG
    PRF = LG.PROFILES["refe_aux_s1"]

    def pf_path(r):
        return Path(r["out"]) / "preflight.jsonl"

    def dvb(r, arm_name, drop_forbidden=()):
        prof = {**PRF, "declared_arm": arm_name, "forbidden_flags": tuple(f for f in PRF["forbidden_flags"] if f not in drop_forbidden)}
        return GC.check_dvb_scorer(r["argv"], GC.load_scorer_preflight(pf_path(r)), prof)

    def consume(r, arm_name):
        return LG.consume_scorer_preflight(pf_path(r), r["argv"], {**PRF, "declared_arm": arm_name})

    def forged(r, name, mut):
        """a copy of the run's preflight.jsonl with the verdict and every gate's own `ok` forced to PASS/True (and `mut` applied to the G12 dict): only the RAW fields can still refuse it"""
        rows = [dict(x) for x in r["rows"]["preflight.jsonl"]]
        for x in rows:
            if x.get("event") == "preflight":
                x["verdict"] = "PASS"
                x["gates"] = {k: dict(v) for k, v in x["gates"].items()}
                for k in x["gates"]:
                    x["gates"][k]["ok"] = True
                g12k = "G12_aux_drv_live_and_identity"
                if g12k in x["gates"]:
                    mut(x["gates"][g12k])
        p = Path(tmp()) / name
        p.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
        return LG.consume_scorer_preflight(p, r["argv"], {**PRF, "declared_arm": "H1-s0"})

    def rows_n(ev):
        return ev["details"]["observed_rows"]
    t("gate/G_DVB_passes_on_the_real_H1_data_event_and_observes_the_head", lambda: (lambda ev: ev["status"] == "PASS" and rows_n(ev) == rows_n(dvb(get_N(), "N-s0")) + 15
                                                                                  and ev["details"]["unknown_flags"] == [] and ev["details"]["stale_registry"] == [])(dvb(h1pre(), "H1-s0")))
    t("gate/G_DVB_passes_on_the_real_H2_data_event_scorer_width_63", lambda: (lambda ev: ev["status"] == "PASS" and any(
        r_["flag"] == "--aux-drv-feed" and r_["built"] == 63 for r_ in ev["details"]["rows"]))(dvb(pre("h2pre", H2_ARGS), "H2-s0")))
    t("gate/G_DVB_passes_on_the_real_shuffle_data_event_all_keys_deranged", lambda: (lambda r: dvb(r, "H1-s0", ("--aux-drv-shuffle-seed",))["status"] == "PASS"
                                                                                  and blk_of(r).get("shuffle_keys") == AUXREAL.PINNED_STORE_KEYS)(pre("sh", H1_ARGS + ["--aux-drv-shuffle-seed", "7"])))
    t("gate/G_DVB_passes_on_the_PRE_EDIT_trainers_head_less_data_event", lambda: (lambda ev: ev["status"] == "PASS" and rows_n(ev) == 4)(dvb(get_N(), "N-s0")))
    t("gate/G_DVB_refuses_an_H1_argv_offered_the_head_less_trainers_data_event", lambda: (lambda ev: ev["status"] == "FAIL" and "--aux-drv: declared True != built False" in " ".join(ev["reasons"]))(
        GC.check_dvb_scorer(h1pre()["argv"], GC.load_scorer_preflight(pf_path(get_N())), {**PRF, "declared_arm": "H1-s0"})))
    t("gate/G_DVB_refuses_a_head_less_argv_offered_an_H1_data_event", lambda: (lambda ev: ev["status"] == "FAIL" and "--aux-drv: declared False != built True" in " ".join(ev["reasons"]))(
        GC.check_dvb_scorer(get_N()["argv"], GC.load_scorer_preflight(pf_path(h1pre())), {**PRF, "declared_arm": "N-s0"})))
    t("gate/consumer_accepts_the_real_H1_preflight_and_obliges_G12", lambda: (lambda ev: ev["status"] == "PASS" and "G12" in ev["details"]["required"])(consume(h1pre(), "H1-s0")))
    t("gate/consumer_accepts_the_real_H2_preflight", lambda: consume(pre("h2pre", H2_ARGS), "H2-s0")["status"] == "PASS")
    t("gate/consumer_accepts_the_PRE_EDIT_head_less_preflight_without_G12", lambda: (lambda ev: ev["status"] == "PASS" and "G12" not in ev["details"]["required"])(consume(get_N(), "N-s0")))

    def g12_dropped():
        rows = [dict(x) for x in h1pre()["rows"]["preflight.jsonl"]]
        for x in rows:
            if x.get("event") == "preflight":
                x["gates"] = {k: v for k, v in x["gates"].items() if not k.startswith("G12_")}
        p = Path(tmp()) / "nog12.jsonl"
        p.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
        ev = LG.consume_scorer_preflight(p, h1pre()["argv"], {**PRF, "declared_arm": "H1-s0"})
        return ev["status"] == "FAIL" and "ABSENT: ['G12']" in " ".join(ev["reasons"])
    t("gate/consumer_refuses_the_real_H1_preflight_with_G12_removed", g12_dropped)
    for nm, env_k in (("tap_copy", "REFE_AUX_MUTATE_TAP"), ("loss_dropped", "REFE_AUX_MUTATE_DROP_LOSS"), ("head_detached", "REFE_AUX_MUTATE_DETACH_HEAD")):
        def sw_gate(nm=nm, env_k=env_k):
            rm = pre("mut_" + nm, H1_ARGS, env={env_k: "1"})
            honest = consume(rm, "H1-s0")                                                   # the trainer's own verdict (FAIL) is refused ...
            lie = forged(rm, f"forged_{nm}.jsonl", lambda g: None)                          # ... and so is a forgery that forces verdict and every `ok` to PASS: the raw fields refuse it
            return honest["status"] == "FAIL" and lie["status"] == "FAIL" and any(("regression switch" in r_ or "non-zero" in r_ or "visual_ctx" in r_) for r_ in lie["reasons"])
        t(f"gate/consumer_refuses_the_real_preflight_of_{env_k}_even_with_verdict_and_ok_forged", sw_gate)
    t("gate/consumer_re_judges_a_forged_clean_real_G12_as_still_PASS", lambda: forged(h1pre(), "forged_clean.jsonl", lambda g: None)["status"] == "PASS")      # control: forging a CLEAN run changes nothing
    t("gate/consumer_refuses_the_real_G12_with_one_raw_field_broken", lambda: all(forged(h1pre(), f"broken_{i}.jsonl", m)["status"] == "FAIL" for i, m in enumerate((
        lambda g: g["per_group_declared"].update(q=0.0), lambda g: g["per_group_w0"].update(q=1e-12), lambda g: g.update(tap_is_visual_ctx=False),
        lambda g: g.update(score_to_head_grads_all_none=False), lambda g: g.update(scorer_params_bit_identical_2steps=False), lambda g: g.update(adam_head_params_with_state=0),
        lambda g: g.update(bce_declared=1.2), lambda g: g.update(w_aux=0.2), lambda g: g.update(labelled_frames_in_batch=0)))))
    return bad


# ------------------------------------------------------------------------------------------------ mutations: (name, snippet, replacement, the checks that own it)
MUTATIONS = [
    ("loop_drops_the_aux_backward", "a_bce, a_hub = aux_backward(dl.tail, a.accum)", "a_bce, a_hub = 0.0, 0.0", ("h1/head_parameters_moved",)),
    ("peel_loader_not_installed", "dl = AuxPeelLoader(dl)", "dl = dl", ("h1/run_completes",)),
    ("peel_loader_tail_goes_stale", "            body, self.tail = split_aux(b, True)\n",
     "            body, _t = split_aux(b, True)\n            self.tail = self.tail if self.tail is not None else _t\n", ("unit/AuxPeelLoader",)),
    ("aux_backward_drops_the_term", 'if os.environ.get("REFE_AUX_MUTATE_DROP_LOSS") != "1":', "if False:", ("h1/head_parameters_moved", "pre/G12_present_and_ok")),
    ("head_optimiser_never_steps", "                optH.step()\n", "                pass\n", ("h1/head_parameters_moved",)),
    ("head_lr_ignores_aux_drv_lr", 'gr["lr"] = lr_at(upd) * lr_ratio', 'gr["lr"] = lr_at(upd)', ("h1/head_lr_follows",)),
    ("head_params_added_to_the_scorer_optimiser", "paramsB = list(armB.parameters()) + (list(aux.feed.parameters()) if (aux is not None and aux.feed is not None) else [])",
     "paramsB = list(armB.parameters()) + (list(aux.feed.parameters()) if (aux is not None and aux.feed is not None) else []) + (list(aux.head.parameters()) if aux is not None else [])",
     ("h1/scorer_arms_bit_identical", "h1/head_has_its_own_Adam_state")),
    ("dataset_hook_removed", "        ds.aux_src = aux_src ", "        pass  ", ("h1/run_completes",)),
    ("aux_eval_rows_misaligned", "rows = [ds.rows[i] for i in ho_idx]\n        n = sum(", "rows = [ds.rows[i] for i in ho_idx[::-1]]\n        n = sum(", ("h1/held_out_perception_read",)),
    ("fwd_ignores_the_feed", "else aux.score(armB, cxy, sctx, sdf)", "else armB(cxy, sctx)", ("h2/arm_B_differs", "h2/feed_projection_left_zero")),
    ("feed_projection_not_in_the_scorer_optimiser", "(list(aux.feed.parameters()) if (aux is not None and aux.feed is not None) else [])", "[]", ("h2/feed_projection_left_zero",)),
    ("labels_keyed_by_the_wrong_row", 'lab, valid = self.src.labels([(r.get("log_name", ""), r.get("token", ""))])',
     'lab, valid = self.src.labels([(self.dataset.rows[0].get("log_name", ""), self.dataset.rows[0].get("token", ""))])', ("unit/AuxSubset_serves",)),
    ("labels_keyed_with_step_and_rank", 'lab, valid = self.src.labels([(r.get("log_name", ""), r.get("token", ""))])',
     'lab, valid = self.src.labels([(r.get("log_name", ""), r.get("token", "") + str(r.get("step", 0)))])', ("unit/AuxSubset_serves",)),
    ("head_optimiser_lr_is_the_scorers", "optH = torch.optim.AdamW(list(aux.head.parameters()), lr=a.aux_drv_lr, weight_decay=0.01)",
     "optH = torch.optim.AdamW(list(aux.head.parameters()), lr=a.lr, weight_decay=0.01)", ("pre/data_event_carries_BUILT", "gate/G_DVB_passes_on_the_real_H1")),
    ("head_built_with_the_default_depth", "depth=a.aux_drv_depth,", "depth=2,", ("pre/depth_flag_reaches", "gate/G_DVB_passes_on_the_real_shuffle")),
    ("w_aux_not_passed_to_the_head", "w_aux=a.w_aux_drv,", "w_aux=0.0,", ("pre/data_event_carries_BUILT", "pre/G12_present_and_ok", "gate/G_DVB_passes_on_the_real_H1")),
    ("g12_tap_check_bypassed", 'g["tap_is_visual_ctx"] = bool(len(outs) >= 2 and AUX.tap_is(cap["aux_ctx"], outs[1]))', 'g["tap_is_visual_ctx"] = True', ("pre/regression_switch_REFE_AUX_MUTATE_TAP", "gate/consumer_refuses_the_real_preflight_of_REFE_AUX_MUTATE_TAP")),
    ("g12_zero_control_never_zeroes_the_weight", '            if mode == "w0":\n                aux.w_aux = 0.0\n', '            if mode == "w0":\n                pass\n',
     ("pre/G12_weight0_control", "pre/G12_present_and_ok", "gate/consumer_accepts_the_real_H1")),
    ("g12_not_in_gates", '            gates["G12_aux_drv_live_and_identity"] = g12_probe(batch, aux_lv)\n', "            pass\n", ("pre/G12_present_and_ok", "gate/consumer_accepts_the_real_H1")),
]


def main() -> int:
    src = SRC_PATH.read_text(encoding="utf-8")
    n = 0
    print("unmutated suite ...", flush=True)
    real = suite(src)
    assert not real, f"the unmutated scorer_finetune.py FAILS its aux wiring checks: {real}"
    n += 1
    print(f"{'mutation':46s} caught-by (failing checks)")
    for i, (name, old, new, owners) in enumerate(MUTATIONS):
        cnt = src.count(old)
        assert cnt == 1, f"mutation {name}: the snippet occurs {cnt} times (must be exactly 1) -- a vacuous mutation proves nothing"
        msrc = src.replace(old, new)
        try:
            compile(msrc, str(SRC_PATH), "exec")
        except SyntaxError as e:
            raise AssertionError(f"mutation {name} does not compile: {e}")
        try:
            failing = suite(msrc, only=owners)
        except Exception as e:                                                       # noqa: BLE001 -- a mutated caller that breaks the harness is RED, and said so
            failing = [f"suite setup raised {type(e).__name__}: {str(e)[:70]}"]
        assert failing, f"mutation {name} stayed GREEN on {owners}: the suite cannot see it"
        print(f"{name:46s} {', '.join(failing[:3])}", flush=True)
        n += 1
    print(f"ZZTEST_PASS {n}  (1 unmutated suite + {len(MUTATIONS)} RED mutations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
