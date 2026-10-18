#!/usr/bin/env python3
"""Tests for `--dump-scores` in refe/scorer_finetune.py (RL-S1 priority 3).  Run: python refe/test_dump_scores.py  ->  ZZTEST_PASS <n>   (prints the mutation table).

CPU only, no checkpoint, no pod.  `evaluate()` lives inside main(), so -- as refe/test_rl_epo.py does -- it is AST-extracted from the SOURCE TEXT of
scorer_finetune.py and exec'd against a tiny fixture (stub loader, stub `fwd` whose scores are a known function of the row index).  So a mutation of the real
caller is seen by every check.  What is proven:
  * KEYING: row n of each .npz is the set at ho_idx[n] (log, token, step, rank), its ckpt_step is the held-out bank's, and proposal m of the row is the m-th
    proposal of that set -- on 37 sets (batches of 16, 16, 5) in a permuted order, for the base / A / B arms, each carrying its own scores.
  * THE PAIR: the dump is read by the REAL `eval/rl_val_eval.load_scores` and `run()`; the PDMS of each arm's pick equals the hand-derived value, so a dump that
    is mis-keyed or mis-ordered cannot reproduce it.
  * DEFAULT OFF = BIT-IDENTICAL: the parser default is "", nothing is written, evaluate() with the dump statements removed is AST-identical to the pre-edit
    evaluate() embedded below, and its returned numbers equal the pre-edit function's exactly -- with the dump ON they are equal too (observational).
  * the file layout equals `rl_val_eval.write_scores_npz` array for array.
  * the REAL main() (test_rl_epo's CPU smoke harness): the flag, the files, and bit-identical weights / logged numbers with the dump on and off; the PDMS of each
    arm's pick recomputed from the dump equals the number the trainer itself logged.
Each mutation edits scorer_finetune.py's SOURCE (the snippet must occur exactly once) and the suite must then report >= 1 failing check.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import types

import numpy as np
import torch
from torch.utils.data import Subset

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(HERE, "scorer_finetune.py")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))
import rl_reward  # noqa: E402
import rl_val_eval as RV  # noqa: E402

# ---- BASELINE: the pre-edit evaluate(), sliced verbatim (ast.unparse) from the index blob of scorer_finetune.py before this change.  Do not edit.
BASE_EVALUATE = r'''@torch.no_grad()
def evaluate(tag, limit=None):
    idx = ho_idx[:limit] if limit else ho_idx
    dlh = DataLoader(Subset(ds_ho, idx), batch_size=16, shuffle=False, num_workers=2)
    res = {k: {'pick': [], 'rand': [], 'best': [], 'zero': [], 'auc_nc': [], 'auc_dac': [], 'auc_ddc': [], 'onc': [], 'pick_x': [], 'onc_x': [], 'zero_x': [], 'lk': [], 'pick_l': [], 'zero_l': [], 'onc_l': [], 'lk_l': [], 'auc_lane': []} for k in ('base', 'A', 'B')}
    lk_rows = [ho.lane_keep.get(key(ds.rows[i])) for i in idx]
    toks, logs_ = ([], [])
    j = 0
    for batch in dlh:
        sx0, sxA, sxB, tg = fwd(batch)
        tlane = tg[..., 6] if tg.shape[-1] == 7 else None
        tg = tg[..., :6]
        ta = agg_v1(tg)
        lkb = [lk_rows[j + bi] for bi in range(tg.shape[0])]
        for k, sx in (('base', sx0), ('A', sxA), ('B', sxB)):
            p = torch.sigmoid(sx)
            ix = agg_v1(p).argmax(1, keepdim=True)
            pick = ta.gather(1, ix).squeeze(1)
            res[k]['pick'] += pick.tolist()
            res[k]['rand'] += ta.mean(1).tolist()
            res[k]['best'] += ta.max(1).values.tolist()
            res[k]['zero'] += (pick == 0).float().tolist()
            res[k]['onc'] += (tg[..., 5].gather(1, ix).squeeze(1) < 1).float().tolist()
            ixx = (agg_v1(p) * p[..., 5]).argmax(1, keepdim=True)
            pkx = ta.gather(1, ixx).squeeze(1)
            res[k]['pick_x'] += pkx.tolist()
            res[k]['zero_x'] += (pkx == 0).float().tolist()
            res[k]['onc_x'] += (tg[..., 5].gather(1, ixx).squeeze(1) < 1).float().tolist()

            def lk_of(ixs):
                return [float(lkb[bi][int(ixs[bi])]) if lkb[bi] is not None else float('nan') for bi in range(len(lkb))]
            res[k]['lk'] += lk_of(ix.squeeze(1).tolist())
            if p.shape[-1] == 7:
                ixl = (agg_v1(p) * p[..., 6]).argmax(1, keepdim=True)
                pkl = ta.gather(1, ixl).squeeze(1)
                res[k]['pick_l'] += pkl.tolist()
                res[k]['zero_l'] += (pkl == 0).float().tolist()
                res[k]['onc_l'] += (tg[..., 5].gather(1, ixl).squeeze(1) < 1).float().tolist()
                res[k]['lk_l'] += lk_of(ixl.squeeze(1).tolist())
                if tlane is not None:
                    for bi in range(p.shape[0]):
                        if bool((tlane[bi] >= 0).all()):
                            v = set_auc(p[bi, :, 6].cpu(), tlane[bi].cpu())
                            if v is not None:
                                res[k]['auc_lane'].append(v)
            for bi in range(p.shape[0]):
                for comp, nm in ((0, 'auc_nc'), (1, 'auc_dac'), (5, 'auc_ddc')):
                    v = set_auc(p[bi, :, comp].cpu(), tg[bi, :, comp].cpu())
                    if v is not None:
                        res[k][nm].append(v)
        for _ in range(tg.shape[0]):
            r = ds.rows[idx[j]]
            j += 1
            toks.append(f"{r.get('log_name')}|{r.get('token')}|{r.get('step')}|{r.get('rank', 0)}")
            logs_.append(r.get('log_name'))
    out = {'event': 'eval', 'tag': tag, 'n': len(toks)}
    for k, v in res.items():
        pk, rd, bs = (np.mean(v['pick']), np.mean(v['rand']), np.mean(v['best']))
        out[k] = {'pick': 100 * pk, 'random': 100 * rd, 'best': 100 * bs, 'skill': (pk - rd) / max(bs - rd, 1e-09), 'zero_picks': int(sum(v['zero'])), 'auc_nc': float(np.mean(v['auc_nc'])) if v['auc_nc'] else None, 'auc_dac': float(np.mean(v['auc_dac'])) if v['auc_dac'] else None, 'auc_ddc': float(np.mean(v['auc_ddc'])) if v['auc_ddc'] else None, 'onc_picks': int(sum(v['onc'])), 'x_ddc': {'pick': 100 * float(np.mean(v['pick_x'])), 'zero_picks': int(sum(v['zero_x'])), 'onc_picks': int(sum(v['onc_x']))}, 'lk10_picks': int(np.nansum(v['lk'])) if any((x == x for x in v['lk'])) else None}
        if v['pick_l']:
            out[k]['x_lane'] = {'pick': 100 * float(np.mean(v['pick_l'])), 'zero_picks': int(sum(v['zero_l'])), 'onc_picks': int(sum(v['onc_l'])), 'lk10_picks': int(np.nansum(v['lk_l'])) if any((x == x for x in v['lk_l'])) else None, 'auc_lane_vs_teacher': float(np.mean(v['auc_lane'])) if v['auc_lane'] else None}
    tl = tl_map = dict(zip(toks, logs_))
    for k in ('A', 'B'):
        d = {t: 100 * (res[k]['pick'][i] - res['base']['pick'][i]) for i, t in enumerate(toks)}
        mu, lo, hi = boot_logs(d, tl)
        out[k]['pick_minus_base'] = {'mean': mu, 'ci95': [lo, hi], 'estimator': 'log-cluster bootstrap over the held-out logs, 10,000, seed 20260927'}
        dx = {t: 100 * (res[k]['pick_x'][i] - res['base']['pick'][i]) for i, t in enumerate(toks)}
        mu, lo, hi = boot_logs(dx, tl)
        out[k]['x_ddc']['pick_minus_base_v1'] = {'mean': mu, 'ci95': [lo, hi]}
        do = {t: 100 * (res[k]['onc_x'][i] - res['base']['onc'][i]) for i, t in enumerate(toks)}
        mu, lo, hi = boot_logs(do, tl)
        out[k]['x_ddc']['onc_pp_minus_base_v1'] = {'mean': mu, 'ci95': [lo, hi]}
    for k in ('A', 'B'):
        if res[k]['pick_l']:
            dl_ = {t: 100 * (res[k]['pick_l'][i] - res['base']['pick'][i]) for i, t in enumerate(toks)}
            mu, lo, hi = boot_logs(dl_, tl_map)
            out[k]['x_lane']['pick_minus_base_v1'] = {'mean': mu, 'ci95': [lo, hi]}
            if all((x == x for x in res[k]['lk_l'])) and all((x == x for x in res['base']['lk'])):
                dk = {t: 100 * (res[k]['lk_l'][i] - res['base']['lk'][i]) for i, t in enumerate(toks)}
                mu, lo, hi = boot_logs(dk, tl_map)
                out[k]['x_lane']['lk10_pp_minus_base_v1'] = {'mean': mu, 'ci95': [lo, hi]}
    dx = {t: 100 * (res['base']['pick_x'][i] - res['base']['pick'][i]) for i, t in enumerate(toks)}
    mu, lo, hi = boot_logs(dx, tl)
    out['base']['x_ddc']['pick_minus_base_v1'] = {'mean': mu, 'ci95': [lo, hi]}
    do = {t: 100 * (res['base']['onc_x'][i] - res['base']['onc'][i]) for i, t in enumerate(toks)}
    mu, lo, hi = boot_logs(do, tl)
    out['base']['x_ddc']['onc_pp_minus_base_v1'] = {'mean': mu, 'ci95': [lo, hi]}
    log(out)
    return out'''
BASE_DESTS = ('ckpt', 'backbone', 'targets', 'images', 'calib', 'onpolicy', 'heldout', 'heldout_logs', 'out', 'batch', 'accum', 'lr', 'warmup', 'epochs', 'lam', 'tau_t', 'eval_every', 'workers', 'seed', 'preflight', 'max_updates', 'lane_labels_train', 'lane_labels_heldout', 'b_mode', 'b_compw', 'pdm_labels_train', 'pdm_labels_heldout', 'require_pdm', 'teacher_lane_train', 'teacher_lane_heldout', 'lane_head', 'lane_weight', 'tau_s', 'epo', 'epo_temp', 'epo_beta', 'epo_ramp', 'epo_lam', 'epo_lam_d', 'epo_w_prog', 'epo_k', 'epo_lam_k', 'epo_l', 'epo_lam_l', 'epo_terms_train', 'epo_terms_heldout')

N, M, C = 37, 4, 7                                                  # 37 sets -> loader batches 16, 16, 5; 4 proposals; 7 outputs (the lane head)


def load(src, name):
    m = types.ModuleType(name)
    m.__file__ = SRC_PATH
    sys.modules[name] = m
    exec(compile(src, SRC_PATH, "exec"), m.__dict__)
    return m


def nested_fn(src, name="evaluate"):
    t = ast.parse(src)
    main = next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    return next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == name)


class StripDump(ast.NodeTransformer):
    """Remove the dump's statements: `sdump = ...` and every `if sdump is not None:` block."""

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "sdump":
            return None
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        t = node.test
        if isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "sdump":
            return None
        return node


class FakeLoader:
    def __init__(self, subset, batch_size=1, shuffle=False, num_workers=0):
        self.items = [subset[k] for k in range(len(subset))]
        self.bs = batch_size

    def __iter__(self):
        for i in range(0, len(self.items), self.bs):
            yield torch.stack(self.items[i:i + self.bs])


class StubDS:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.tensor(i)


def row(i):
    return {"log_name": f"L{i % 7:02d}", "token": f"tok{i:03d}", "step": 0, "rank": i % 2}


def ckpt_of(i):
    return 10075 if i % 3 else 9000


def exp_scores(r, m, c, arm):
    """The stub scorers' logit of row r, proposal m, output c, arm 0/1/2 (base/A/B): exactly representable in float32."""
    return (r * 64 + m * 8 + c) / 8 + 1000.0 * arm


def make_ns(src, a_dump, ho_idx, n_out, fwd=None, cols=C):
    mod = load(src, "sf_mut")
    ds = types.SimpleNamespace(rows=[row(i) for i in range(n_out)])
    ho = types.SimpleNamespace(lane_keep={}, by={mod.key(row(i)): (ckpt_of(i), None, None, (ckpt_of(i), 3), False) for i in range(n_out)})
    logs = []

    def default_fwd(batch):
        r = batch.to(torch.float64).view(-1, 1, 1)
        mm = torch.arange(M, dtype=torch.float64).view(1, M, 1)
        cc = torch.arange(cols, dtype=torch.float64).view(1, 1, cols)
        sx = [((r * 64 + mm * 8 + cc) / 8 + 1000.0 * arm).float() for arm in range(3)]
        tg = torch.zeros(batch.shape[0], M, cols)
        tg[..., :6] = (mm + 1) / (M + 1)
        if cols == 7:
            tg[..., 6] = -1.0
        return sx[0], sx[1], sx[2], tg

    ns = dict(mod.__dict__)
    ns.update(a=types.SimpleNamespace(dump_scores=a_dump), ds=ds, ho=ho, ds_ho=StubDS(n_out), ho_idx=list(ho_idx), fwd=fwd or default_fwd,
              DataLoader=FakeLoader, Subset=Subset, log=logs.append, boot_logs=lambda d, l, n=10000, seed=0: (float(np.mean(list(d.values()))), 0.0, 0.0))
    return mod, ns, logs


def loadz(path):
    """np.load that CLOSES the file (a TemporaryDirectory cannot be removed on Windows while an NpzFile holds it open)."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def build_eval(text, ns):
    exec(compile(text, "<evaluate>", "exec"), ns)
    return ns["evaluate"]


def jdump(o):
    return json.dumps(o, sort_keys=True, default=str)


# ------------------------------------------------------------------------------------------------------------------ the checks
def suite(src):
    out = []

    def check(name, ok, detail=""):
        out.append((name, bool(ok), detail))

    def guarded(name, fn):
        try:
            fn()
        except Exception as e:                                                                 # noqa: BLE001
            check(name, False, f"raised {type(e).__name__}: {str(e)[:140]}")

    ho_idx = [(i * 17) % N for i in range(N)]                                                  # a permutation of 0..36 (17 is coprime to 37)
    new_text = ast.unparse(nested_fn(src))

    # ---- parser
    def t_parser():
        mod = load(src, "sf_parser")
        ap = mod.build_parser()
        dests = {act.dest for act in ap._actions if act.dest != "help"}
        ns_ = ap.parse_args(["--ckpt", "c", "--targets", "t", "--images", "i", "--calib", "k", "--onpolicy", "o", "--heldout", "h", "--heldout-logs", "l", "--out", "x"])
        check("parser: --dump-scores exists and DEFAULTS TO OFF ('')", getattr(ns_, "dump_scores", None) == "", str(getattr(ns_, "dump_scores", "<absent>")))
        # the auxiliary drivable-area head (a LATER, separately tested edit: test_aux_s1_wiring.py) adds eight default-OFF dests; they are not this edit's and are named, not waved through
        aux_dests = {"aux_drv", "w_aux_drv", "aux_drv_tap", "aux_drv_depth", "aux_drv_feed", "aux_drv_labels", "aux_drv_shuffle_seed", "aux_drv_lr"}
        # the R0c EPO anchor (a LATER, separately tested edit: test_rl_epo.py / test_rl_r0c.py, f64a952) adds ONE dest; its default "base" is every pre-R0c arm bit for bit (R41)
        epo_dests = {"epo_ref"} | {"epo_con", "epo_con_eps", "epo_con_lr", "epo_con_max", "epo_con_lam0"}          # SAFEOBJ (raw/2026-10-17-safety-objective): five more default-OFF dests, ALL-or-none like the others; "off" is every pre-SAFEOBJ arm bit for bit
        epo_dests |= {"epo_con_temp"}          # PICKCON (raw/2026-10-18-so-pickcon): ONE more dest, the constraint's own softmax temperature (EPO_CON_DESTS); default None = it INHERITS --epo-temp, every pre-PICKCON arm bit for bit
        check("parser: exactly ONE new dest from the dump edit (dump_scores); every pre-edit dest is still there",
              dests - set(BASE_DESTS) - aux_dests - epo_dests == {"dump_scores"} and set(BASE_DESTS) <= dests and (dests & aux_dests) in (set(), aux_dests)
              and (dests & epo_dests) in (set(), epo_dests) and getattr(ns_, "epo_ref", "base") == "base" and getattr(ns_, "epo_con", "off") == "off"
              and getattr(ns_, "epo_con_temp", None) is None, str(sorted(dests - set(BASE_DESTS))))
    guarded("parser", t_parser)

    # ---- OFF: structurally identical to the pre-edit evaluate()
    def t_off_ast():
        stripped = StripDump().visit(ast.parse(new_text))
        base_ast = ast.parse(BASE_EVALUATE)
        check("OFF: evaluate() minus the dump's statements is AST-IDENTICAL to the pre-edit evaluate()", ast.dump(stripped) == ast.dump(base_ast))
        n_guarded = sum(1 for n in ast.walk(ast.parse(new_text)) if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name) and n.test.left.id == "sdump")
        check("OFF: the dump touches evaluate() in exactly 2 guarded blocks (collect, write) and 1 creation", n_guarded == 2, str(n_guarded))
    guarded("off_ast", t_off_ast)

    # ---- OFF / ON numerics and keying
    def t_run():
        with tempfile.TemporaryDirectory() as d:
            _, ns_base, logs_base = make_ns(src, "", ho_idx, N)
            res_base = build_eval(BASE_EVALUATE, ns_base)("step0")
            _, ns_off, logs_off = make_ns(src, "", ho_idx, N)
            res_off = build_eval(new_text, ns_off)("step0")
            check("OFF: returns EXACTLY the pre-edit function's numbers, and logs the same rows", jdump(res_off) == jdump(res_base) and jdump(logs_off) == jdump(logs_base) and len(logs_base) == 1)
            check("OFF: nothing is written (no score_dump event)", all(r.get("event") != "score_dump" for r in logs_off))
            dd = os.path.join(d, "dump")
            _, ns_on, logs_on = make_ns(src, dd, ho_idx, N)
            res_on = build_eval(new_text, ns_on)("step0")
            check("ON: the returned evaluation is UNCHANGED (the dump is observational)", jdump(res_on) == jdump(res_base))
            ev = [r for r in logs_on if r.get("event") == "score_dump"]
            check("ON: one score_dump log event naming 3 files and 37 sets", len(ev) == 1 and ev[0]["n_sets"] == N and set(ev[0]["files"]) == {"base", "A", "B"}, str(ev)[:200])
            files = {arm: os.path.join(dd, f"scores_step0_{arm}.npz") for arm in ("base", "A", "B")}
            check("ON: exactly the 3 files scores_step0_{base,A,B}.npz exist", all(os.path.exists(p) for p in files.values()) and sorted(os.listdir(dd)) == sorted(os.path.basename(p) for p in files.values()), str(sorted(os.listdir(dd))))
            for ai, arm in enumerate(("base", "A", "B")):
                z = loadz(files[arm])
                keys_ok = all((str(z["log"][n]), str(z["token"][n]), int(z["step"][n]), int(z["rank"][n])) == (row(ho_idx[n])["log_name"], row(ho_idx[n])["token"], 0, row(ho_idx[n])["rank"]) for n in range(len(z["token"])))
                ck_ok = all(int(z["ckpt_step"][n]) == ckpt_of(ho_idx[n]) for n in range(len(z["token"])))
                sc = z["scores"]
                sc_ok = sc.shape == (N, M, C) and all(sc[n, m, c] == exp_scores(ho_idx[n], m, c, ai) for n in range(N) for m in range(M) for c in range(C))
                check(f"{arm}: row n is the set at ho_idx[n] (log, token, step, rank) -- across the 16/16/5 batch boundaries", keys_ok and len(z["token"]) == N)
                check(f"{arm}: ckpt_step per row is the held-out bank's own", ck_ok)
                check(f"{arm}: scores[n, m, c] is THAT set's proposal m, output c, of THIS arm (float64, exact)", sc_ok and sc.dtype == np.float64, f"{sc.shape} {sc.dtype}")
                check(f"{arm}: space is 'logit' (the evaluator applies the sigmoid)", str(z["space"]) == "logit")
            # the file layout equals the evaluator's own writer, array for array
            ks = [(row(ho_idx[n])["log_name"], row(ho_idx[n])["token"], 0, row(ho_idx[n])["rank"]) for n in range(N)]
            ref = os.path.join(d, "ref.npz")
            RV.write_scores_npz(ref, ks, [ckpt_of(ho_idx[n]) for n in range(N)], loadz(files["B"])["scores"], "logit")
            za, zb = loadz(files["B"]), loadz(ref)
            check("layout: identical to rl_val_eval.write_scores_npz (same arrays, dtypes, values)",
                  sorted(za) == sorted(zb) and all(za[f].dtype == zb[f].dtype and np.array_equal(za[f], zb[f]) for f in za), str(sorted(za)))
            loaded = RV.load_scores(files["B"])
            k0 = ks[5]
            check("the REAL rl_val_eval.load_scores reads it: 37 keys, no duplicate, right ckpt and [M, C] block", len(loaded) == N and loaded[k0]["ckpt"] == ckpt_of(ho_idx[5]) and loaded[k0]["scores"].shape == (M, C))
            # limit: the first 10 of ho_idx
            dl = os.path.join(d, "lim")
            _, ns_l, _ = make_ns(src, dl, ho_idx, N)
            build_eval(new_text, ns_l)("lim", 10)
            zl = loadz(os.path.join(dl, "scores_lim_B.npz"))
            check("limit=10: the dump holds exactly the first 10 evaluated sets, in order", len(zl["token"]) == 10 and [str(t) for t in zl["token"]] == [row(ho_idx[n])["token"] for n in range(10)])
            # an unsafe tag is made file-name safe
            _, ns_t, _ = make_ns(src, os.path.join(d, "tg"), ho_idx, N)
            build_eval(new_text, ns_t)("update 400/x")
            check("tag with a space and a slash -> a safe file name", os.path.exists(os.path.join(d, "tg", "scores_update_400_x_B.npz")), str(os.listdir(os.path.join(d, "tg"))))
    guarded("run", t_run)

    # ---- THE PAIR end to end: dump -> real evaluator -> hand-derived PDMS of each arm's pick
    def t_pair():
        S, Mx = 6, 4
        best = [(3 * s) % 4 for s in range(S)]                       # s=0..5 -> 0, 3, 2, 1, 0, 3
        base_pick = [best[s] if s % 3 == 0 else (best[s] + 1) % 4 for s in range(S)]        # base is right on s = 0 and 3 only
        hi = list(range(S))

        def fwd(batch):
            b = batch.shape[0]
            sx = [torch.full((b, Mx, 6), -5.0) for _ in range(3)]
            for bi in range(b):
                s = int(batch[bi])
                sx[0][bi, base_pick[s], :] = 5.0                     # base
                sx[1][bi, base_pick[s], :] = 5.0                     # A = control: same picks as base
                sx[2][bi, best[s], :] = 5.0                          # B always picks the best
            return sx[0], sx[1], sx[2], torch.zeros(b, Mx, 6)

        with tempfile.TemporaryDirectory() as d:
            _, ns, _ = make_ns(src, os.path.join(d, "p"), hi, S, fwd=fwd, cols=6)
            build_eval(new_text, ns)("pair")
            arms = {"C0": RV.load_scores(os.path.join(d, "p", "scores_pair_base.npz")), "R0": RV.load_scores(os.path.join(d, "p", "scores_pair_B.npz"))}
            pdm = {}
            for s in range(S):
                nc = [1.0 if m == best[s] else 0.0 for m in range(Mx)]
                tg = np.stack([nc, [1.0] * Mx, [1.0] * Mx, [1.0] * Mx, [1.0] * Mx, [1.0] * Mx], -1)
                pdm[(row(s)["log_name"], row(s)["token"], 0, row(s)["rank"])] = {"ckpt": ckpt_of(s), "tg": tg}
            res = RV.run(arms, pdm, None, None, "navsim_v1", rl_reward.RewardCfg(lam_d=0.0), n_boot=200, seed=1)
            # PDMS(pick) = NC * (5 + 5 + 2) / 12 = NC: base is right on 2 of 6 sets -> 100 * 2/6 ; B on all 6 -> 100
            check("pair: base PDMS_x100 = 100 * 2/6 (hand-derived)", abs(res["arms"]["C0"]["PDMS_x100"] - 100 * 2 / 6) < 1e-9, str(res["arms"]["C0"]["PDMS_x100"]))
            check("pair: B PDMS_x100 = 100 and 0 zero picks; base has 4 zero picks (hand-derived)", abs(res["arms"]["R0"]["PDMS_x100"] - 100.0) < 1e-9 and res["arms"]["R0"]["zero_picks"] == 0 and res["arms"]["C0"]["zero_picks"] == 4,
                  str((res["arms"]["R0"]["PDMS_x100"], res["arms"]["R0"]["zero_picks"], res["arms"]["C0"]["zero_picks"])))
            check("pair: all 6 sets used, none stale or missing", res["eligibility"]["sets_used"] == S and res["eligibility"]["stale_or_mismatched_ckpt"] == 0 and res["eligibility"]["missing_pdm"] == 0, str(res["eligibility"]))
            check("pair: the evaluator's own self-pair and swap checks pass on the dumped arms", res["selfcheck"]["selfpair"]["ok"] and res["selfcheck"]["swap"]["ok"])
    guarded("pair", t_pair)

    # ---- the REAL main() (preflight, then train, on CPU with test_rl_epo's stub model / banks): the flag, evaluate(), the saved weights
    def t_main_smoke():
        import test_rl_epo as TE
        snippet = 'ap.add_argument("--dump-scores", default="",'
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as td_off, tempfile.TemporaryDirectory() as td_on:
            try:
                os.chdir(td_off)                                                   # a mutant whose default is a relative dir must not litter the repo
                r_off = TE.smoke(src, td_off, epo=False, tag="dsoff")
                dd = os.path.join(td_on, "dump").replace("\\", "/")
                src_on = src.replace(snippet, f'ap.add_argument("--dump-scores", default={dd!r},') if src.count(snippet) == 1 else src
                os.chdir(td_on)
                r_on = TE.smoke(src_on, td_on, epo=False, tag="dson")
            finally:
                os.chdir(cwd)
            check("main smoke: preflight and run both exit 0, with and without the dump", (r_off["pre"], r_off["run"], r_on["pre"], r_on["run"]) == (0, 0, 0, 0), str((r_off.get("pre"), r_off.get("run"), r_on.get("pre"), r_on.get("run"))))
            files_off = [f for _, _, fs in os.walk(td_off) for f in fs if f.startswith("scores_")]
            files_on = sorted(f for f in os.listdir(dd)) if os.path.isdir(dd) else []
            want = sorted(f"scores_{t}_{a}.npz" for t in ("step0", "update25") for a in ("base", "A", "B"))
            check("main smoke OFF: no scores_*.npz anywhere and no score_dump event", not files_off and all(x.get("event") != "score_dump" for x in r_off["rows"]["sft.jsonl"]), str(files_off))
            check("main smoke ON: exactly scores_{step0,update25}_{base,A,B}.npz", files_on == want, str(files_on))

            def evals(r):
                return json.dumps([{k: v for k, v in x.items() if k != "at"} for x in r["rows"]["sft.jsonl"] if x.get("event") == "eval"], sort_keys=True)

            def losses(r):
                return json.dumps([{k: x[k] for k in ("bceA", "listnet", "lossB", "update") if k in x} for x in r["rows"]["sft.jsonl"] if x.get("event") == "train"], sort_keys=True)
            check("main smoke: every logged eval number is IDENTICAL with the dump on and off", evals(r_on) == evals(r_off) and evals(r_off) != "[]")
            check("main smoke: every logged training loss is IDENTICAL with the dump on and off", losses(r_on) == losses(r_off) and losses(r_off) != "[]")
            same_w = all(torch.equal(r_on[a][k], r_off[a][k]) for a in ("A", "B") for k in r_off[a])
            check("main smoke: the saved A and B scorer weights are BIT-IDENTICAL with the dump on and off", same_w)
            if len(files_on) == 6:
                z = {(t, a): loadz(os.path.join(dd, f"scores_{t}_{a}.npz")) for t in ("step0", "update25") for a in ("base", "A", "B")}
                bank = TE.StubBank("held", 8, 4)
                keys = [(f"H{i // 4}", f"tok{i}", i, 0) for i in range(TE.N_HELD)]
                zz = z[("step0", "B")]
                check("main smoke: keys are the held-out sets in evaluation order, ckpt_step 5, scores [12, 8, 6] float64 logit",
                      [(str(zz["log"][n]), str(zz["token"][n]), int(zz["step"][n]), int(zz["rank"][n])) for n in range(TE.N_HELD)] == keys and set(zz["ckpt_step"].tolist()) == {5}
                      and zz["scores"].shape == (TE.N_HELD, 8, 6) and zz["scores"].dtype == np.float64 and str(zz["space"]) == "logit")
                check("main smoke: at step 0 base == A == B exactly (the trainer's own invariant, read off the dump)",
                      np.array_equal(z[("step0", "base")]["scores"], z[("step0", "A")]["scores"]) and np.array_equal(z[("step0", "base")]["scores"], z[("step0", "B")]["scores"]))
                check("main smoke: after training A and B moved off base (arm B = arm A here: --lam 0, no EPO) and the frozen base did not move",
                      not np.array_equal(z[("update25", "base")]["scores"], z[("update25", "B")]["scores"]) and not np.array_equal(z[("update25", "base")]["scores"], z[("update25", "A")]["scores"])
                      and np.array_equal(z[("update25", "base")]["scores"], z[("step0", "base")]["scores"]))
                ev = {x["tag"]: x for x in r_on["rows"]["sft.jsonl"] if x.get("event") == "eval"}
                ok_all, worst = True, 0.0
                for tag in ("step0", "update25"):
                    for arm, nm in (("base", "base"), ("A", "A"), ("B", "B")):
                        vals = []
                        for n, k in enumerate(keys):
                            i, _ = RV.select(z[(tag, arm)]["scores"][n], "navsim_v1", "logit")
                            s = rl_reward.split_components(bank.by[k][2])
                            vals.append(float(rl_reward.pdms_v1(s["nc"], s["dac"], s["ep"], s["ttc"], s["c"])[i]))
                        got, logged = 100 * float(np.mean(vals)), ev[tag][nm]["pick"]
                        worst = max(worst, abs(got - logged))
                        ok_all = ok_all and abs(got - logged) < 1e-3
                check("main smoke: the PDMS of each arm's pick RECOMPUTED from the dump (rl_val_eval.select + the labels) equals the number evaluate() logged, all 6 (tag, arm)", ok_all, f"max diff {worst:.2e}")
    guarded("main_smoke", t_main_smoke)

    # ---- the writer on its own
    def t_unit():
        mod = load(src, "sf_unit")
        sd = mod.ScoreDump()
        try:
            sd.add([("l", "t", 0, 0)], [1, 2], torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), torch.zeros(1, 2, 3))
            check("ScoreDump.add refuses len(keys) != len(ckpts)", False)
        except ValueError:
            check("ScoreDump.add refuses len(keys) != len(ckpts)", True)
        try:
            sd.add([("l", "t", 0, 0)], [1], torch.zeros(2, 2, 3), torch.zeros(1, 2, 3), torch.zeros(1, 2, 3))
            check("ScoreDump.add refuses an arm whose row count differs from the keys", False)
        except ValueError:
            check("ScoreDump.add refuses an arm whose row count differs from the keys", True)
    guarded("unit", t_unit)
    return out


# ------------------------------------------------------------------------------------------------------------------ mutations
MUTATIONS = [
    ("token_column_reversed", "keying", "token=np.asarray([k[1] for k in self.keys])", "token=np.asarray([k[1] for k in reversed(self.keys)])"),
    ("proposal_axis_flipped", "keying", "scores=np.concatenate(self.chunks[arm], 0)", "scores=np.concatenate(self.chunks[arm], 0)[:, ::-1]"),
    ("arms_swapped", "keying", "for arm, sx in zip(self.ARMS, (sx0, sxA, sxB)):", "for arm, sx in zip(self.ARMS, (sxB, sxA, sx0)):"),
    ("ckpt_step_zero", "keying", "[ho.by[k_][0] for k_ in bkeys]", "[0 for k_ in bkeys]"),
    ("row_offset_forgotten", "keying", "key(ds.rows[idx[j + bi]])", "key(ds.rows[idx[bi]])"),
    ("last_batch_keys_truncated", "keying", "self.keys += [tuple(k) for k in keys]", "self.keys += [tuple(k) for k in keys][:-1]"),
    ("sigmoid_written_as_logit", "format", "self.chunks[arm].append(sx.detach().float().cpu().numpy().astype(np.float64))", "self.chunks[arm].append(torch.sigmoid(sx.detach().float()).cpu().numpy().astype(np.float64))"),
    ("space_prob", "format", 'space=np.asarray("logit")', 'space=np.asarray("prob")'),
    ("float32_scores", "format", "self.chunks[arm].append(sx.detach().float().cpu().numpy().astype(np.float64))", "self.chunks[arm].append(sx.detach().float().cpu().numpy().astype(np.float32))"),
    ("step_column_dropped", "format", "step=np.asarray([k[2] for k in self.keys], np.int64), ", ""),
    ("dump_alters_the_scores", "off", "self.chunks[arm].append(sx.detach().float().cpu().numpy().astype(np.float64))", "self.chunks[arm].append(sx.sigmoid_().detach().float().cpu().numpy().astype(np.float64))"),
    ("dump_always_on", "off", "sdump = ScoreDump() if a.dump_scores else None", "sdump = ScoreDump()"),
    ("flag_default_on", "off", 'ap.add_argument("--dump-scores", default="",', 'ap.add_argument("--dump-scores", default="scores_out",'),
    ("unguarded_edit_in_evaluate", "off", "            sx0, sxA, sxB, tg = fwd(batch)\n            if sdump is not None:", "            sx0, sxA, sxB, tg = fwd(batch)\n            tg = tg.clone()\n            if sdump is not None:"),
]


def main() -> int:
    src = open(SRC_PATH, encoding="utf-8").read()
    base = suite(src)
    bad = [(n, d) for n, ok, d in base if not ok]
    for n, d in bad:
        print(f"  FAIL {n}  {d}")
    assert not bad, f"the unmutated source FAILS its own checks: {[n for n, _ in bad]}"
    print(f"unmutated: {len(base)} checks green")
    print(f"{'mutation':32s} {'class':8s} caught-by")
    for name, cls, old, new in MUTATIONS:
        cnt = src.count(old)
        assert cnt == 1, f"mutation {name}: the snippet occurs {cnt} times (must be exactly 1) -- a vacuous mutation proves nothing"
        msrc = src.replace(old, new)
        compile(msrc, SRC_PATH, "exec")
        failing = [n for n, ok, _ in suite(msrc) if not ok]
        print(f"{name:32s} {cls:8s} {len(failing)}  {failing[0][:80] if failing else ''}")
        assert failing, f"mutation {name} ({cls}) stayed GREEN: the suite cannot see it"
    need = {"keying", "format", "off"}
    assert {c for _, c, _, _ in MUTATIONS} >= need
    print(f"ZZTEST_PASS {1 + len(MUTATIONS) + 1}  (1 unmutated suite of {len(base)} checks + {len(MUTATIONS)} RED mutations + coverage)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
