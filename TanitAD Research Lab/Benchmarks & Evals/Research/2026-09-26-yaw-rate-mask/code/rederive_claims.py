"""D-YAWMASK-1 -- re-derive every landed claim that quotes a PAIRED yaw-rate cell made by
`refav1_arm._components`, where the dumps still exist on the dev box. 0 GPU.

For each claim record, ONE procedure, two runs:
  1. REPRODUCTION CONTROL -- the TIP refav1_arm.py (the cell as it was): every re-computed cell
     must equal the banked artifact's cell (delta / lo / hi / separated / n). If it does not,
     the record's re-derivation is INADMISSIBLE and says so (the dump or code drifted).
  2. FIXED -- the same computation with only refav1_arm.py swapped; every NON-yaw cell must be
     unchanged vs step 1, and the yaw cells are reported before -> after.

Handlers (each the record's OWN instrument, never a re-derivation of it):
  families_paired  refcv3_arm / refav1_arm / egodrop records -> `_paired_families` in-process,
                   loading the dump exactly as the analyzer does (g/arm [..., :2], eid = file stem)
  t1_blocks        the refcv6-rl-stage-a `t1_base_four_families.json` flattening of the same
  wbank            the withheld-bank panel (panel_score.py `paired_vs_A0`) from its per-arm npz
  paired_openloop  taniteval/tools/paired_openloop.py CLI, argv rebuilt from the record
  paired_delta     taniteval/tools/refav1_paired_delta.py (== the research copies, same blob) CLI
  stratified       taniteval/tools/stratified_openloop.py CLI

usage: python rederive_claims.py --work C:/Users/Admin/ym26/claims --bank <pkg>/raw/claims [--only id,id]
Trees: C:/Users/Admin/ym26/te_tip/taniteval and te_fix/taniteval differ ONLY in refav1_arm.py
(blob-asserted at start); stack = C:/Users/Admin/ym26/tree_m26/stack (tip).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sanitize as S  # noqa: E402

GIT_DIR = "C:/Users/Admin/tanitad-push/.git"
BRANCH = "agent/arch-inf-20260803"
PY = sys.executable
YM = Path("C:/Users/Admin/ym26")
STACK = YM / "tree_m26" / "stack"
TE = {"TIP": YM / "te_tip" / "taniteval", "FIXED": YM / "te_fix" / "taniteval"}
BLOB = {"TIP": "c7013107ab1353c5102fd806d29cd967216b7a3a",
        "FIXED": "963d98e6c3f1a00df8599c9ea8f39cb538e26e23"}
YAW = "LAT_yaw_rate_mae_radps"
FIELDS = ("delta", "lo", "hi", "separated", "n_windows", "n_episodes", "n_dropped_nonfinite")
AI = "TanitAD Research Lab/Architecture & Inference/Research/"
DO = "TanitAD Research Lab/Deployment & Optimization/Research/"
WP56 = "C:/Users/Admin/_wp56/dump/refcv3_40284_dump"
RL17 = "C:/Users/Admin/tanitad-caches/ddv2rl-l1-20260917"

CLAIMS = [
    # id, handler, banked record (repo path), params
    ("refcv3_40284_arm", "families_paired", "taniteval/results/refcv3-40284-openloop.ARM.json",
     {"key": "refcv3", "dump": WP56}),
    ("refav1_step1000", "families_paired",
     AI + "2026-09-03-refav1-step1000-read/raw/refav1_t1_step1000.json",
     {"key": "refav1", "dump": "C:/Users/Admin/refav1_eval_slice/t1_dump"}),
    ("refav1_21109", "families_paired", "taniteval/results/refav1-21109-openloop.json",
     {"key": "refav1", "dump": "C:/Users/Admin/refav1_eval_slice/thor/full"}),
    ("egodrop_9500_2s", "families_paired",
     AI + "2026-09-05-refcv4b-egodrop-burden/raw/EGODROP_9500.json",
     {"key": "four_families_2s", "dump": "C:/Users/Admin/refcv4b_egodrop/dump_refcv4b step 9500_2s",
      "dt": 0.5}),
    ("egodrop_5000_2s", "families_paired",
     AI + "2026-09-05-refcv4b-egodrop-burden/raw/EGODROP_5000.json",
     {"key": "four_families_2s", "dump": "C:/Users/Admin/refcv4b_egodrop/dump_refcv4b step 5000_2s",
      "dt": 0.5}),
    ("t1_base_rl17", "t1_blocks", AI + "2026-09-17-refcv6-rl-stage-a/raw/t1_base_four_families.json",
     {"dump": RL17 + "/t1_base_dump",
      "pairs": {"paired_os_minus_ha0": ("ha0", "os"), "paired_os_minus_ha0ext": ("ha0_ext", "os"),
                "paired_os_minus_navzero": ("os_navzero", "os"),
                "paired_os_minus_navshuf": ("os_navshuf", "os")}}),
    ("wbank_panel", "wbank", AI + "2026-09-05-withheld-bank-panel/raw/panel_report.json",
     {"dumps": "C:/Users/Admin/run_wbank/score_dumps"}),
    ("veto_s0", "paired_openloop", DO + "2026-09-05-veto-only-fan-safety/raw/run/paired_s0-veto200_vs_base.json", {}),
    ("veto_s1", "paired_openloop", DO + "2026-09-05-veto-only-fan-safety/raw/run/paired_s1-veto200_vs_base.json", {}),
    ("rl_readiness", "paired_openloop", DO + "2026-09-05-refc-rl-readiness/raw/run/paired_rl_vs_base.json", {}),
    ("l1_rl_s0", "paired_openloop", AI + "2026-09-17-refcv6-rl-stage-a/raw/paired_l1-rl-s0__vs__base.json", {}),
    ("l1_rl_s1", "paired_openloop", AI + "2026-09-17-refcv6-rl-stage-a/raw/paired_l1-rl-s1__vs__base.json", {}),
    ("pd_phase1", "paired_delta", AI + "2026-09-05-refav1-surface-sweep/raw/paired_delta_phase1.json", {}),
    ("pd_all", "paired_delta", AI + "2026-09-05-refav1-cost-geometry/raw/pd_all.json", {}),
    ("pd_kamm", "paired_delta", AI + "2026-09-05-refav1-cost-geometry/raw/pd_kamm.json", {}),
    ("pd_seedfloor", "paired_delta", AI + "2026-09-05-refav1-cost-geometry/raw/pd_seedfloor.json", {}),
    ("pd_fact", "paired_delta", AI + "2026-09-05-refav1-cost-geometry/raw/pd_fact.json", {}),
    ("pd_comb", "paired_delta", AI + "2026-09-05-refav1-cost-geometry/raw/pd_comb.json", {}),
    ("pd_lonshift", "paired_delta", AI + "2026-09-05-refav1-longitudinal/raw/pd_lonshift.json", {}),
    ("pd_devbox_lonshift", "paired_delta",
     AI + "2026-09-05-refav1-lonshift-t1/raw/devbox/pd_devbox_lonshift.json", {}),
    ("strat_40284", "stratified", "taniteval/results/refcv3-40284-stratified.json", {"dump": WP56}),
]


# --------------------------------------------------------------------------- #
def git(*args, binary=False):
    r = subprocess.run(["git", f"--git-dir={GIT_DIR}", *args], capture_output=True)
    if r.returncode:
        raise RuntimeError(f"git {args} rc={r.returncode}: {r.stderr.decode('utf-8', 'replace')[:300]}")
    return r.stdout if binary else r.stdout.decode("utf-8").strip()


def banked(path):
    return json.loads(git("show", f"{BRANCH}:{path}"))


def blob_of(p: Path) -> str:
    b = git("hash-object", "--no-filters", str(p))
    if len(b) != 40:
        raise RuntimeError(f"INCONCLUSIVE blob for {p}")
    return b


def env():
    e = dict(os.environ)
    e.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
              "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    return e


_MODS = {}


def ra(which):
    if which not in _MODS:
        for p in (str(STACK), str(TE[which])):
            if p not in sys.path:
                sys.path.insert(0, p)
        spec = importlib.util.spec_from_file_location(f"ra_{which}",
                                                      str(TE[which] / "tools" / "refav1_arm.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _MODS[which] = m
    return _MODS[which]


def f(c):
    return tuple((c or {}).get(k) for k in FIELDS)


def cmp_cells(a: dict, b: dict, skip=None, only=None):
    keys = sorted(set(a) | set(b), key=str)
    diffs = []
    n = 0
    for k in keys:
        mk = k[-1]
        if skip and mk == skip:
            continue
        if only and mk not in only:
            continue
        n += 1
        if a.get(k) != b.get(k):
            diffs.append(f"{k}: {a.get(k)} -> {b.get(k)}")
    return {"n_cells_compared": n, "n_differences": len(diffs), "differences": diffs[:25]}


# --------------------------------------------------------------------------- #
def load_dump(d, arms):
    files = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    if not files:
        raise FileNotFoundError(f"no ep*.npz under {d}")
    G, P, eid = [], {a: [] for a in arms}, []
    for fp in files:
        with np.load(fp) as z:
            G.append(z["g"][..., :2].astype(np.float64))
            eid += [os.path.splitext(os.path.basename(fp))[0]] * z["g"].shape[0]
            for a in arms:
                P[a].append(z[a][..., :2].astype(np.float64))
    return np.concatenate(G), {a: np.concatenate(v) for a, v in P.items()}, eid, len(files)


def fp_cells(blocks: dict) -> dict:
    out = {}
    for nm, blk in blocks.items():
        for fam, ms in ((blk or {}).get("families") or {}).items():
            for mk, c in (ms or {}).items():
                if isinstance(c, dict):
                    out[(nm, fam, mk)] = f(c)
    return out


def h_families_paired(rec, p):
    blocks = rec
    for k in p["key"].split("."):
        blocks = blocks[k]
    blocks = blocks["families_paired"]
    pairs = {}
    for nm, blk in blocks.items():
        y, x = [s.strip() for s in blk["direction"].split(" - ")]
        pairs[nm] = (x, y)
    arms = sorted({a for xy in pairs.values() for a in xy})
    dt = float(p.get("dt") or rec.get("dt_s") or rec.get("grid", {}).get("dt_s"))
    G, P, eid, n_files = load_dump(p["dump"], arms)
    n_boot = next((c.get("n_boot") for blk in blocks.values()
                   for ms in (blk.get("families") or {}).values() for c in ms.values()
                   if isinstance(c, dict) and c.get("n_boot")), 2000)
    tiers = {a: "T1" for a in arms}
    out = {"dt_s": dt, "n_windows": int(len(eid)), "n_files": n_files, "n_boot": n_boot,
           "pairs": {nm: list(xy) for nm, xy in pairs.items()}}
    res = {}
    for which in ("TIP", "FIXED"):
        m = ra(which)
        comps = {a: m._components(P[a], G, dt) for a in arms}
        res[which] = {nm: m._paired_families(comps, x, y, eid, tiers, n_boot, 0)
                      for nm, (x, y) in pairs.items()}
    return out, fp_cells(blocks), fp_cells(res["TIP"]), fp_cells(res["FIXED"])


def h_t1_blocks(rec, p):
    blocks = rec["blocks"]
    pairs = p["pairs"]
    arms = sorted({a for xy in pairs.values() for a in xy})
    man = json.load(open(os.path.join(p["dump"], "manifest.json"), encoding="utf-8"))
    dt = float(man["grid"]["dt_s"])
    G, P, eid, n_files = load_dump(p["dump"], arms)

    def flat(blks):
        o = {}
        for nm, blk in blks.items():
            for key, c in (blk.get("metrics") or {}).items():
                fam, mk = key.split(".", 1)
                o[(nm, fam, mk)] = tuple(c.get(k) for k in ("delta", "lo", "hi", "separated"))
        return o

    res = {}
    for which in ("TIP", "FIXED"):
        m = ra(which)
        comps = {a: m._components(P[a], G, dt) for a in arms}
        fpb = {nm: m._paired_families(comps, x, y, eid, {a: "T1" for a in arms}, 2000, 0)
               for nm, (x, y) in pairs.items()}
        o = {}
        for nm, blk in fpb.items():
            for fam, ms in blk["families"].items():
                for mk, c in ms.items():
                    o[(nm, fam, mk)] = tuple(c.get(k) for k in ("delta", "lo", "hi", "separated"))
        res[which] = (o, fpb)
    fixed_n = {nm: {k: blk["families"]["lateral"][YAW].get(k) for k in ("n_windows", "n_dropped_nonfinite")}
               for nm, blk in res["FIXED"][1].items()}
    return ({"dt_s": dt, "n_windows": int(len(eid)), "n_files": n_files,
             "fixed_yaw_n": fixed_n}, flat(blocks), res["TIP"][0], res["FIXED"][0])


def h_wbank(rec, p):
    SL2, SL6 = [0, 1, 2, 3], [1, 3, 4, 5, 6, 7]
    d = Path(p["dumps"])
    Z = {a: np.load(d / f"{a}.npz", allow_pickle=True) for a in rec["arms"]
         if (d / f"{a}.npz").exists()}
    a0n = "A0_fixed"
    eid = Z[a0n]["eid"]
    wp = Z[a0n]["wp"]
    for a, z in Z.items():
        assert np.array_equal(z["eid"], eid) and np.array_equal(z["wp"], wp), a
    from taniteval import ci as CI  # noqa: E402  (tip ci, on sys.path via ra())

    def paired(a, b):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 8:
            return {"status": "UNAVAILABLE", "n_windows": int(ok.sum())}
        e = [x for x, k in zip(eid, ok) if k]
        r = CI.paired_episode_cluster_bootstrap(a[ok], b[ok], e, n_boot=2000, seed=0)
        r["n_dropped_nonfinite"] = int((~ok).sum())
        return r

    def cells_of(pv_by_arm):
        o = {}
        for arm, pv in pv_by_arm.items():
            for tag, blk in pv.items():
                for sec in ("plan_2s", "plan_6s"):
                    for mk, c in (blk.get(sec) or {}).items():
                        o[(arm, tag, sec, mk)] = f(c)
        return o

    bank = {arm: r["paired_vs_A0"] for arm, r in rec["arms"].items() if "paired_vs_A0" in r}
    res = {}
    for which in ("TIP", "FIXED"):
        m = ra(which)
        comps = {}
        for arm, z in Z.items():
            tags = sorted({k[:-5] for k in z.files if k.endswith("_traj")})
            for tag in tags:
                P = z[f"{tag}_traj"]
                comps[(arm, tag)] = (m._components(P[:, SL2], wp[:, SL2], 0.5),
                                     m._components(P[:, SL6], wp[:, SL6], 1.0))
        pv_all = {}
        for arm in bank:
            pv_all[arm] = {}
            for tag in bank[arm]:
                base_tag = bank[arm][tag]["vs_A0_regime"]
                c2, c6 = comps[(arm, tag)]
                a2, a6 = comps[(a0n, base_tag)]
                pv_all[arm][tag] = {"plan_2s": {mk: paired(c2[mk], a2[mk]) for mk in c2},
                                    "plan_6s": {mk: paired(c6[mk], a6[mk]) for mk in c2}}
        res[which] = pv_all
    return ({"n_windows": int(len(eid)), "n_episodes": int(len(set(eid.tolist()))),
             "arms": sorted(bank)}, cells_of(bank), cells_of(res["TIP"]), cells_of(res["FIXED"]))


def _run_cli(which, argv, out_json, timeout=3600):
    e = env()
    e["PYTHONPATH"] = f"{STACK.as_posix()};{TE[which].as_posix()}"
    if out_json.exists():
        out_json.unlink()
    t0 = time.time()
    r = subprocess.run([PY, "-u", *argv], capture_output=True, text=True, env=e,
                       encoding="utf-8", errors="replace", timeout=timeout, cwd=str(YM))
    out_json.with_suffix(".log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr,
                                            encoding="utf-8")
    if not out_json.exists():
        raise RuntimeError(f"{which}: the tool wrote NO JSON (rc {r.returncode}); "
                           f"tail: {r.stderr[-600:]}")
    return round(time.time() - t0, 1)


def po_cells(rec):
    o = {}
    for sec in ("absolute_pooled_full_set", "margins_over_floor"):
        for arm, ms in (rec.get(sec) or {}).items():
            for mk, c in (ms or {}).items():
                if isinstance(c, dict):
                    o[(sec, arm, mk)] = tuple((c or {}).get(k) for k in ("mean",) + FIELDS)
    for mk, c in (rec.get("cross_model_difference_of_margins") or {}).items():
        if isinstance(c, dict):
            o[("cross", "cross", mk)] = tuple((c or {}).get(k) for k in ("mean",) + FIELDS)
    return o


def h_paired_openloop(rec, p, work: Path, cid):
    A, B = rec["inputs"]["A"], rec["inputs"]["B"]
    for d in (A["dump"], B["dump"]):
        if not glob.glob(os.path.join(d, "ep*.npz")):
            raise FileNotFoundError(f"dump absent: {d}")
    argv = ["--a-dump", A["dump"], "--a-name", A["name"], "--a-arm", rec["_a_arm"],
            "--b-dump", B["dump"], "--b-name", B["name"], "--b-arm", rec["_b_arm"],
            "--floor", rec["_floor"], "--n-boot", str(rec["estimator"]["n_boot"]),
            "--seed", str(rec["estimator"]["seed"])]
    for x in rec.get("_a_extra") or []:
        argv += ["--a-extra", x]
    for x in rec.get("_b_extra") or []:
        argv += ["--b-extra", x]
    outs, walls = {}, {}
    for which in ("TIP", "FIXED"):
        oj = work / f"{cid}.{which}.json"
        walls[which] = _run_cli(which, [str(TE[which] / "tools" / "paired_openloop.py"), *argv,
                                        "--out", str(oj), "--md", str(oj.with_suffix(".md"))], oj)
        outs[which] = json.loads(oj.read_text(encoding="utf-8"))
    return ({"argv": argv, "wall_s": walls, "void": {w: outs[w].get("void") for w in outs}},
            po_cells(rec), po_cells(outs["TIP"]), po_cells(outs["FIXED"]))


def pd_cells(rec, metrics):
    o = {}
    for pr, d in (rec.get("paired") or {}).items():
        for mk, c in (d or {}).items():
            if mk in metrics and isinstance(c, dict):
                o[(pr, mk)] = f(c)
    return o


def h_paired_delta(rec, p, work: Path, cid):
    arm = rec["arm"]
    names = list(rec["dumps"])
    argv = []
    for nm in names:
        path = rec["dumps"][nm]["path"]
        if not glob.glob(os.path.join(path, "ep*.npz")):
            raise FileNotFoundError(f"dump absent: {path}")
        argv += ["--dump", f"{nm}={path}"]
    ref = names[0]
    for key in rec["paired"]:
        y, x = [s.strip() for s in key.split(" - ")]
        if x.endswith(f".{arm}") and y.endswith(f".{arm}") and not (x == y == f"{ref}.{arm}"):
            argv += ["--pair", f"{y[:-len(arm) - 1]}-{x[:-len(arm) - 1]}"]
    argv += ["--arm", arm, "--n-boot", str(rec.get("n_boot", 2000)), "--seed", str(rec.get("seed", 0)),
             "--stack", str(STACK)]
    metrics = set(ra("TIP")._FAMILY_OF)
    outs, walls = {}, {}
    for which in ("TIP", "FIXED"):
        oj = work / f"{cid}.{which}.json"
        walls[which] = _run_cli(which, [str(TE[which] / "tools" / "refav1_paired_delta.py"), *argv,
                                        "--taniteval", str(TE[which]), "--out", str(oj),
                                        "--md", str(oj.with_suffix(".md"))], oj)
        outs[which] = json.loads(oj.read_text(encoding="utf-8"))
    return ({"argv": argv, "wall_s": walls,
             "known_value_control": {w: outs[w].get("known_value_control", {}).get("PASS")
                                     for w in outs}},
            pd_cells(rec, metrics), pd_cells(outs["TIP"], metrics), pd_cells(outs["FIXED"], metrics))


def st_cells(rec):
    o = {}
    for con, blk in (rec.get("per_stratum") or {}).items():
        for cell, row in (blk.get("cells") or {}).items():
            for fam, ms in (row.get("families") or {}).items():
                for mk, c in (ms or {}).items():
                    if isinstance(c, dict):
                        o[(con, cell, fam, mk)] = tuple(c.get(k) for k in FIELDS + ("verdict",))
    return o


def h_stratified(rec, p, work: Path, cid):
    argv = ["--dump", p["dump"], "--n-boot", str(rec["estimator"]["n_boot"]),
            "--seed", str(rec["estimator"]["seed"]), "--tag", rec["tag"],
            "--dt", str(rec["dt_s"])]
    outs, walls = {}, {}
    for which in ("TIP", "FIXED"):
        oj = work / f"{cid}.{which}.json"
        walls[which] = _run_cli(which, [str(TE[which] / "tools" / "stratified_openloop.py"), *argv,
                                        "--out", str(oj)], oj)
        outs[which] = json.loads(oj.read_text(encoding="utf-8"))
    return ({"argv": argv, "wall_s": walls}, st_cells(rec), st_cells(outs["TIP"]),
            st_cells(outs["FIXED"]))


# --------------------------------------------------------------------------- #
def yaw_view(cells: dict) -> dict:
    return {" | ".join(str(x) for x in k): v for k, v in cells.items() if YAW in k}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--only", default=None)
    a = ap.parse_args(argv)
    work, bank = Path(a.work), Path(a.bank)
    work.mkdir(parents=True, exist_ok=True)
    bank.mkdir(parents=True, exist_ok=True)
    for w in ("TIP", "FIXED"):
        got = blob_of(TE[w] / "tools" / "refav1_arm.py")
        if got != BLOB[w]:
            raise SystemExit(f"{w} tree carries refav1_arm blob {got}, expected {BLOB[w]}")
    only = set(a.only.split(",")) if a.only else None
    summary = {}
    for cid, handler, path, params in CLAIMS:
        if only and cid not in only:
            continue
        t0 = time.time()
        r = {"id": cid, "handler": handler, "banked_record": path}
        try:
            rec = banked(path)
            r["banked_blob"] = git("rev-parse", f"{BRANCH}:{path}")
            if handler == "families_paired":
                meta, cb, ct, cf = h_families_paired(rec, params)
            elif handler == "t1_blocks":
                meta, cb, ct, cf = h_t1_blocks(rec, params)
            elif handler == "wbank":
                meta, cb, ct, cf = h_wbank(rec, params)
            elif handler == "paired_openloop":
                meta, cb, ct, cf = h_paired_openloop(rec, params, work, cid)
            elif handler == "paired_delta":
                meta, cb, ct, cf = h_paired_delta(rec, params, work, cid)
            elif handler == "stratified":
                meta, cb, ct, cf = h_stratified(rec, params, work, cid)
            else:
                raise ValueError(handler)
            rep = cmp_cells(cb, ct)
            nonyaw = cmp_cells(ct, cf, skip=YAW)
            r.update({"meta": meta,
                      "reproduction_vs_banked": rep,
                      "reproduces_exactly": rep["n_differences"] == 0,
                      "fixed_vs_tip_non_yaw": nonyaw,
                      "yaw_banked": yaw_view(cb), "yaw_tip_rerun": yaw_view(ct),
                      "yaw_fixed": yaw_view(cf)})
            r["n_yaw_cells"] = sum(1 for k in cb if YAW in k)
            r["status"] = "OK"
        except FileNotFoundError as ex:
            r.update({"status": "UNVERIFIABLE", "reason": f"dump missing: {ex}"})
        except Exception as ex:  # noqa: BLE001
            r.update({"status": "ERROR", "reason": f"{type(ex).__name__}: {ex}",
                      "trace": traceback.format_exc()[-2000:]})
        r["wall_s"] = round(time.time() - t0, 1)
        (work / f"{cid}.result.json").write_text(json.dumps(r, indent=1, default=str),
                                                 encoding="utf-8")
        S.sanitize_json(work / f"{cid}.result.json", bank / f"{cid}.result.json")
        summary[cid] = {"status": r["status"], "reproduces_exactly": r.get("reproduces_exactly"),
                        "repro_diffs": (r.get("reproduction_vs_banked") or {}).get("n_differences"),
                        "repro_cells": (r.get("reproduction_vs_banked") or {}).get("n_cells_compared"),
                        "non_yaw_diffs_after_fix": (r.get("fixed_vs_tip_non_yaw") or {}).get("n_differences"),
                        "reason": r.get("reason"), "wall_s": r["wall_s"]}
        print(f"[claims] {cid}: {json.dumps(summary[cid], default=str)}", flush=True)
    sp = bank / "claims_summary.json"
    old = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    old.update(summary)
    sp.write_text(json.dumps(old, indent=1, default=str), encoding="utf-8")
    sc = S.scan(bank)
    print(f"[claims] bank scan: {sc['n_files']} files, {sc['n_uuid_hits']} UUID hits, "
          f"{sc['n_unread']} unread", flush=True)
    if sc["n_uuid_hits"] or sc["n_unread"]:
        raise SystemExit(f"bank scan FAILED: {sc}")


if __name__ == "__main__":
    main()
