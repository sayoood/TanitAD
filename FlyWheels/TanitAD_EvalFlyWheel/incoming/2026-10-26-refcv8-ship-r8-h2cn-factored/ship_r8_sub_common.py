"""ship_r8_sub_common.py -- shared pieces of the SHIP-R8 AMENDMENT 4 (H2cn stage) / AMENDMENT 5 (factored decode) tools.  NEW file (EvalFlyWheel 2026-10-26).

* the amendment seals (every tool refuses to run when an amendment's sha256 differs from its .sha256 seal),
* the frozen H2cn stage file (write / load / verify / predict) -- the networks are EXACTLY the ones `l1_fit_score.h2n_fit` returned (taken out of its
  closure, never refitted here), and the predictor reproduces that closure's arithmetic (`F.mlp_inputs` -> net, float64, no_grad),
* the fresh-clip disjointness assertion (fresh split vs eval139 and train4572), from INDEPENDENT sources, with a mutation arm that must read RED.
No raw clip id is ever printed or written: sets are compared as sha256 digests; the only names written are counts.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np

WORK = Path("C:/Users/Admin/refcv8_pf/ship_r8_sub")
AMENDMENTS = {"4": WORK / "SHIP_R8_AMENDMENT_4_H2CN.md", "5": WORK / "SHIP_R8_AMENDMENT_5_FACTORED.md"}
# code this package builds on (imported, never edited)
T_TREE = Path("C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58")
R8_CODE = Path("C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code")
PF_HOLD = Path("C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final")
L1_CODE = Path("C:/Users/Admin/refcv8_pf/refcv9_l1/code")
L1C_CODE = Path("C:/Users/Admin/refcv8_pf/refcv9_l1c/code")
# the frozen stage
STAGE_PT = WORK / "H2CN_STAGE_FROZEN.pt"
STAGE_SHA = WORK / "H2CN_STAGE_FROZEN.sha256"
STAGE_SCHEMA = "tanitad.ship_r8_sub.h2cn_stage/1"
HIDDEN = 128                                   # F.mlp_train: Linear(d_in, 128) -> GELU -> Linear(128, d_out), float64 (LITERAL there)
SLOTS = {"S2": [0, 1, 2, 3], "S6": [1, 3, 4, 5, 6, 7]}
INSTANTS = {"S2": [0.5, 1.0, 1.5, 2.0], "S6": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
# disjointness sources (each pinned)
FRESH_IDS = Path("F:/l1c/select/final_ids_fresh.json")
FRESH_TREE_ROOT, FRESH_TREE = Path("F:/l1c/tree"), "step48852ema_fresh"
V9_EVAL = Path("E:/refcv8_eval_kit/data/refcv8_v9labels/release/v9_labels_eval139.npz")
V9_EVAL_MD5 = "6b5c7f207cffc3b7eb3cd527fd433599"          # refcv8 config argv --r8-v9-eval-md5
V9_TRAIN = Path("E:/refcv8_eval_kit/data/refcv8_v9labels/release_train4572/v9_labels_train.npz")
V9_TRAIN_MD5 = "83e30bd3791f0767707eade5f4671759"         # refcv8 config argv --r8-v9-md5
EXTR = Path("E:/refcv8_eval_kit/data/refcv8_train4572_eval139_extrinsics.json")   # refcv8 config argv --agent-rig-extrinsics (4,711 = 4,572 + 139 clip ids)
EVAL_TREE_DUMP_JSON = Path("C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/dump_s0/refcv6_extras.npz.json")
DIGESTS = T_TREE / "stack/tanitad/data"


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def md5_file(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def hex64(s: str) -> str:
    """first token, hex only, lower case; '' unless exactly 64 hex chars (an empty or short string is never a match)."""
    try:
        t = "".join(c for c in str(s).split()[0] if c in "0123456789abcdefABCDEF").lower()
    except Exception:          # noqa: BLE001
        return ""
    return t if len(t) == 64 else ""


def read_seal(p: Path) -> str:
    try:
        return hex64(Path(p).read_text(encoding="utf-8-sig"))
    except Exception:          # noqa: BLE001
        return ""


def check_amendment_seals() -> dict:
    """REFUSES (SystemExit) unless both amendments hash to their seals (both sides 64 hex)."""
    out = {}
    for k, md in AMENDMENTS.items():
        got = sha256_file(md) if md.is_file() else ""
        want = read_seal(md.with_suffix(".sha256"))
        if len(got) != 64 or len(want) != 64:
            raise SystemExit(f"[ship_r8_sub] INCONCLUSIVE seal for amendment {k}: file '{got}' seal '{want}' -- refusing")
        if got != want:
            raise SystemExit(f"[ship_r8_sub] amendment {k} ({md.name}) does not match its seal ({got} != {want}) -- refusing")
        out[k] = {"file": str(md), "sha256": got, "seal_matches": True}
    return out


AMENDMENT_4B = WORK / "SHIP_R8_AMENDMENT_4B_H2CN_CAP.md"


def check_amendment_4b_seal() -> dict:
    """Amendment 4b's seal (H2CN_CAP8).  Kept OUT of AMENDMENTS on purpose: check_amendment_seals() is recorded in every Amendment 4 / 5 output, and adding a
    key there would change those records.  REFUSES (SystemExit) unless the file hashes to its seal (both sides 64 hex); an unreadable seal is INCONCLUSIVE."""
    got = sha256_file(AMENDMENT_4B) if AMENDMENT_4B.is_file() else ""
    want = read_seal(AMENDMENT_4B.with_suffix(".sha256"))
    if len(got) != 64 or len(want) != 64:
        raise SystemExit(f"[ship_r8_sub] INCONCLUSIVE seal for amendment 4b: file '{got}' seal '{want}' -- refusing")
    if got != want:
        raise SystemExit(f"[ship_r8_sub] amendment 4b ({AMENDMENT_4B.name}) does not match its seal ({got} != {want}) -- refusing")
    return {"file": str(AMENDMENT_4B), "sha256": got, "seal_matches": True}


def setup_paths() -> None:
    """sys.path as the registered lanes use it (chain_l1c.sh cpu(), chain_final_pf_r8.sh lane()): launch tree stack + taniteval + battery code."""
    os.environ.setdefault("R8_LAUNCH_TREE", str(T_TREE))
    for p in (str(L1C_CODE), str(L1_CODE), str(R8_CODE), str(T_TREE / "taniteval"), str(T_TREE / "stack")):
        if p not in sys.path:
            sys.path.insert(0, p)


# ------------------------------------------------------------------------------------------------ fresh-clip disjointness
def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def fresh_disjointness(fit_sha12: dict | None = None, inject_eval_clip: bool = False) -> dict:
    """The fit's clips are disjoint from eval139 and train4572.  Sources (independent of each other and of the fit):
      fresh   = sha256 of the 300 raw ids in final_ids_fresh.json  (cross-checked == the fresh tree dump's clip_sha12_by_index)
      eval139 = v9_labels_eval139.npz clip__sha12 (md5 pinned by the refcv8 config)  (cross-checked == the eval139 battery dump's clip_sha12_by_index)
      train   = v9_labels_train.npz clip__sha12 (md5 pinned by the refcv8 config, 4,572)
      union   = sha256 of the 4,711 raw ids of the refcv8 train4572+eval139 extrinsics file (full 64-hex compare; cross-checked: its sha12 set == train | eval139)
    `inject_eval_clip` is the deliberate regression: one eval139 digest is put into the fresh set -> the check MUST read RED."""
    r = {"sources": {}}
    ids = json.load(open(FRESH_IDS, encoding="utf-8"))
    fresh = {_sha(x["clip_id"]) for x in ids}
    dump_fresh12 = set(json.load(open(FRESH_TREE_ROOT / FRESH_TREE / "dump_s0" / "refcv6_extras.npz.json", encoding="utf-8"))["clip_sha12_by_index"].values())
    m_ev, m_tr = md5_file(V9_EVAL), md5_file(V9_TRAIN)
    ev12 = set(str(x) for x in np.load(V9_EVAL, allow_pickle=False)["clip__sha12"])
    tr12 = set(str(x) for x in np.load(V9_TRAIN, allow_pickle=False)["clip__sha12"])
    dump_ev12 = set(json.load(open(EVAL_TREE_DUMP_JSON, encoding="utf-8"))["clip_sha12_by_index"].values())
    union = {_sha(k) for k in json.load(open(EXTR, encoding="utf-8")).keys()}
    dig = {n: set(json.load(open(DIGESTS / f"{n}_clip_digests.json", encoding="utf-8"))["clip_id_digests"]) for n in ("parity_train", "deployed_val40", "v72_eval")}
    fresh0 = set(fresh)
    if inject_eval_clip:
        fresh = set(fresh) | {next(iter(sorted(u for u in union if u[:12] in ev12)))}
    f12 = {h[:12] for h in fresh}
    r["sources"] = {"fresh_ids": {"file": str(FRESH_IDS), "sha256": sha256_file(FRESH_IDS), "n": len(ids)},
                    "v9_eval139": {"file": str(V9_EVAL), "md5": m_ev, "md5_pinned": V9_EVAL_MD5, "n": len(ev12)},
                    "v9_train4572": {"file": str(V9_TRAIN), "md5": m_tr, "md5_pinned": V9_TRAIN_MD5, "n": len(tr12)},
                    "extrinsics_train4572_eval139": {"file": str(EXTR), "sha256": sha256_file(EXTR), "n": len(union)},
                    "eval_tree_dump_json": {"file": str(EVAL_TREE_DUMP_JSON), "n": len(dump_ev12)},
                    "fresh_tree_dump_json": {"file": str(FRESH_TREE_ROOT / FRESH_TREE / "dump_s0/refcv6_extras.npz.json"), "n": len(dump_fresh12)},
                    "programme_digest_lists": {n: len(v) for n, v in dig.items()}}
    c = {"v9_md5s_pinned": bool(m_ev == V9_EVAL_MD5 and m_tr == V9_TRAIN_MD5),
         "counts": bool(len(ids) == 300 and len(fresh) == 300 + int(inject_eval_clip) and len(ev12) == 139 and len(tr12) == 4572 and len(union) == 4711),
         "xcheck_fresh_ids_eq_fresh_dump": bool({h[:12] for h in fresh0} == dump_fresh12),
         "xcheck_v9_eval_eq_eval_dump": bool(ev12 == dump_ev12),
         "xcheck_extrinsics_sha12_eq_train_union_eval": bool({h[:12] for h in union} == (tr12 | ev12)),
         "xcheck_train_eval_disjoint": bool(not (tr12 & ev12))}
    inter = {"fresh_full_x_extrinsics_full": len(fresh & union), "fresh12_x_eval139": len(f12 & ev12), "fresh12_x_train4572": len(f12 & tr12),
             "fresh12_x_eval_dump": len(f12 & dump_ev12), **{f"fresh_full_x_{n}": len(fresh & v) for n, v in dig.items()}}
    if fit_sha12 is not None:
        for surf, s12 in fit_sha12.items():
            s12 = set(str(x) for x in s12)
            inter[f"fit_rows_{surf}_x_train4572_or_eval139"] = len(s12 & (tr12 | ev12))
            c[f"fit_rows_{surf}_subset_of_fresh"] = bool(s12 <= dump_fresh12)
            r[f"fit_rows_{surf}_n_clips"] = len(s12)
    r["consistency"] = c
    r["intersections"] = inter
    r["disjoint"] = bool(all(v == 0 for v in inter.values()))
    r["PASS"] = bool(r["disjoint"] and all(c.values()))
    r["mutation_injected_one_eval139_clip"] = bool(inject_eval_clip)
    return r


# ------------------------------------------------------------------------------------------------ the frozen stage
def net_and_stats_of(pred_closure):
    """The fitted network and its standardisation stats, taken OUT OF `F.h2n_fit`'s returned closure (no refit, no copy of the training code)."""
    fv = dict(zip(pred_closure.__code__.co_freevars, (c.cell_contents for c in pred_closure.__closure__)))
    return fv["net"], fv["st"]


def make_net(d_in: int, d_out: int):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(d_in, HIDDEN), torch.nn.GELU(), torch.nn.Linear(HIDDEN, d_out)).double()


def stage_payload(surf_fits: dict, provenance: dict) -> dict:
    """surf_fits[surf] = {"net", "st", "info", "n_fresh", "n_fresh_episodes", "n_fresh_clips"}.  Deterministic content only (no timestamps)."""
    import torch
    out = {"schema": STAGE_SCHEMA, "surfaces": {}, "provenance": provenance,
           "use": "s_k = arc(prior8)[slot] + f_surf(X, Z)_j ; X = progress_scalar.feats(win, lim) (10 cols, col 0 = 1), Z = z_tac (512) ; "
                  "f(X, Z) = net(concat[(X[:,1:]-xmu)/xsd, (Z-zmu)/zsd]) float64 (l1_fit_score.mlp_inputs)"}
    for surf, d in surf_fits.items():
        sd = {k: v.detach().clone().contiguous() for k, v in d["net"].state_dict().items()}
        out["surfaces"][surf] = {"state_dict": sd, "st": {k: torch.from_numpy(np.ascontiguousarray(np.asarray(d["st"][k], np.float64))) for k in ("xmu", "xsd", "zmu", "zsd")},
                                 "wd": float(d["info"]["wd"]), "epochs": int(d["info"]["epochs"]), "cv_l1_sum": float(d["info"]["cv_l1_sum"]),
                                 "n_params": int(d["info"]["n_params"]), "d_in": int(sd["0.weight"].shape[1]), "d_out": int(sd["2.weight"].shape[0]),
                                 "hidden": HIDDEN, "slots": list(SLOTS[surf]), "instants_s": list(INSTANTS[surf]),
                                 "target": "arc(G) - arc(HP) at the surface's slots (HP = ha0_ext_pose prior), fresh compliant rows",
                                 "n_fresh_rows": int(d["n_fresh"]), "n_fresh_episodes": int(d["n_fresh_episodes"]), "n_fresh_clips": int(d["n_fresh_clips"]),
                                 "fit": "l1_fit_score.h2n_fit(X, Z, H, eid, seed=0): wd in (1e-4, 1e-2, 1e-1) x epochs <= 200 by 5-fold EPISODE CV (L1 loss), refit on all fresh rows"}
    return out


def load_stage(path: Path = STAGE_PT, registered_sha: str | None = None) -> tuple:
    """Load the frozen stage.  With `registered_sha`, REFUSES unless sha256(file) equals it (both 64 hex).  Returns (preds, blob, sha)."""
    import torch
    got = sha256_file(path)
    if registered_sha is not None:
        reg = hex64(registered_sha)
        if len(got) != 64 or len(reg) != 64:
            raise SystemExit(f"[ship_r8_sub] INCONCLUSIVE stage sha256: file '{got}' registered '{reg}' -- refusing")
        if got != reg:
            raise SystemExit(f"[ship_r8_sub] the stage file hashes to {got}, the REGISTERED stage is {reg} -- refusing")
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if blob.get("schema") != STAGE_SCHEMA:
        raise SystemExit(f"[ship_r8_sub] stage schema {blob.get('schema')!r} != {STAGE_SCHEMA}")
    setup_paths()
    import l1_fit_score as F
    preds = {}
    for surf, d in blob["surfaces"].items():
        net = make_net(d["d_in"], d["d_out"])
        net.load_state_dict(d["state_dict"], strict=True)
        net.eval()
        st = {k: d["st"][k].numpy().astype(np.float64) for k in ("xmu", "xsd", "zmu", "zsd")}

        def pred(Xe, Ze, _net=net, _st=st):
            with torch.no_grad():
                return _net(torch.from_numpy(F.mlp_inputs(Xe, Ze, _st))).numpy()
        preds[surf] = pred
    return preds, blob, got
