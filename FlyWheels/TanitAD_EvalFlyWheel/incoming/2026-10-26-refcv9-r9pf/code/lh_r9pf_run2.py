"""lh_r9pf_run2.py -- PREREG_R9PF_LH (6de4e4bb..., sealed 16:02:32) RUN 2: R9-PF's registered speed profiles driven by LONHEAD-C's frozen clean-split
decision head LH, offline composition on the BANKED fb47a58 paths, CPU only.  NEW file (EvalFlyWheel run 2, 2026-10-26).
Precondition (s2): R9-PF's K-OFF (result/R9PF_KEQ.json) AND K-EQ under AMENDMENT2 (result/R9PF_KEQ_AMEND2.json, 865c9569...) PASSED -- the run-1
VOID record result/R9PF_LH_VERDICT.{json,md} is left untouched; this run writes result/R9PF_LH_VERDICT_RUN2.{json,md}.  Without the precondition it
writes a PROVISIONAL record to scratch only.

Arms (per R9 profile fit seed k = 0/1/2; the head = heads/R9PF-TL_fs{k}.pt, i.e. R9PF's fitted profiles with the trunk_lon decision code path):
  R9LH-ARG   LH's 6 LON6 logits placed in their v7 columns -> r9pf_common.compose(..., lon_logits_v7) -> refcv9_progress.trunk_lon_regime (the
             IDENTICAL map R9PF-TL uses: refcv8_conditioning.LON6_V7_IDS columns, argmax, refcv9_progress.LON6_TO_REGIME) -- nothing re-typed.
  R9LH-SOFT  p(regime) = sum of LH's LON6 softmax over refcv9_progress.LON6_TO_REGIME groups; dS = sum_r p(r) dS[r]; s_t = monotone(s_P + dS);
             traj = retime(plan8, s_t) -- R9-PF's exploratory SOFT decode (x_attrib_r9pf.py) with LH's posterior in place of the head's own.
  R9DER-ARG  as R9LH-ARG with DER (LONHEAD-C's deranged control head) -- K-DER.
Controls: registrations re-hash; K-MAP (R9LH-ARG with the trunk lon head's LON6 logits == R9PF-TL's banked plans + decisions, bit for bit, both
inference seeds, every fit seed); K-SOFT1 (one-hot posterior == R9LH-ARG bit for bit); K-DER; K-LABELS (LH / DER argmax == LONHEAD's banked label
files); K-KHEAD; K-Z (LH input z_tac == the composition's z_tac bit for bit); K-CELL (the alpha-parameterised cell == pair_read_r8.paired_cell at
alpha 0.05); K-SCORE (the K-MAP plans scored here reproduce R9PF_PA_BARS.json's R9PF-TL cells); K-BESIDE-PINNED; K-ALIGN; K-EXC; K-ZERO.
Bars (S6 compliant unless stated; paired vs the reference on the SAME bootstrap draw; B 2000 seed 0; both inference seeds): PA-1a, PA-1b, PA-2
(D11 conjunction), PA-G, PA-R; every REQUIRED separation on the 97.5 % interval, every "not separated above / worse" clause on the 95 % interval.
Tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9pf_common as RC  # noqa: E402

TREE = "step48852ema_pf"
LONHEAD = Path("C:/Users/Admin/refcv8_pf/refcv9_lonhead")
LANDED = Path("C:/Users/Admin/refcv8_pf/refcv9_hsgate/out_mdfix/arms/h2cn_mdfix")
LANDED_MAN = Path("C:/Users/Admin/refcv8_pf/refcv9_hsgate/MANIFEST_MDFIX.sha256")
BATT = Path("C:/Users/Admin/refcv8_eval_archive/battery") / TREE
BESIDE = {"H8": "H2CN_DEPLOY8", "FX": "FACTORED_DEPLOY8", "D8": "RETIME_DEPLOY8"}
PKG_IN = RC.SCRATCH / "arms"
PKG_OUT = RC.SCRATCH / "arms_lh_run2"
SEALS = {"PREREG_R9PF_LH.md": "6de4e4bb0e8abfe0ada03e3e0e401faea9996857ed80744547b923c4f9f2bd59",
         "PREREG_REFCV9_INMODEL_PA_AMENDMENT2.md": "865c95692383f78a0c6355003c6d1b0ec63413a5816827babacb28cc5674cc3d"}
PA_BARS_SHA = "497a551d72ab9a92f6bed7ab56cc4d1156a0fe48d6cb55514ef3df9fd51bcad6"
VOID_FILES = {"R9PF_KEQ.json": "d1de8a28b421379659b24bfa38787e30b3f8ce0452268db367d90faf7b281da7",
              "R9PF_PA_VERDICT.json": "8c6f3c4602aebc30c12e5a0ed70ba24f37b54a51bf804c483b937c6dcc3bdf65",
              "R9PF_LH_VERDICT.json": "ebc82e313519f57f62b1499e825d3df91ee18cc16f91037c3634010f84dbdee4",
              "R9PF_LH_VERDICT.md": "41f247e761a70c97140bbf568102c4b3e943b123273d41159c7f16a0e92f5d62"}
LH_PINS = {"LH_FROZEN.pt": "6fd3ea1e3700a15b6506176bc855355d0cc1eefb3d87d390584e791d98e1d8f6",
           "DER_FROZEN.pt": "08d1085d8d4407f2c6113c07be0fed04616f373cd102fe678c3a52d3d46c1f7e"}
A_SEP, A_NS = 0.025, 0.05            # 97.5 % for required separations; 95 % for "not separated above / worse"
TIER = "T1 self-action OPEN LOOP on logged eval139 frames, not driving performance"
EST = ("paired episode-cluster bootstrap over EPISODES (taniteval.ci.paired_episode_cluster_bootstrap, B 2000, bootstrap seed 0, windows finite on "
       "BOTH sides), models fixed -- blind to training and inference variance; ONE refcv8 trunk seed; ONE LH fit; R9 profile fit seeds 0/1/2 = the "
       "training-variance read; inference seeds 0 / 1 reported separately, never pooled")
ARMS = ("R9LH-ARG", "R9LH-SOFT", "R9DER-ARG")


def seal(name: str, want: str) -> dict:
    p = RC.WORK / name
    got = RC.sha256_file(p) if p.is_file() else ""
    sf = p.with_suffix(".sha256")
    s = RC.hex64(sf.read_text(encoding="utf-8-sig")) if sf.is_file() else ""
    if len(got) != 64 or len(s) != 64 or not (got == s == want):
        raise SystemExit(f"[lh2] seal {name}: INCONCLUSIVE / mismatch ({got} / {s} / {want}) -- refusing")
    return {"file": str(p), "sha256": got}


def pinfile(path: Path, root: Path | None = None, rel_ok: bool = False) -> dict:
    pins, bad = {}, []
    for ln in path.read_text(encoding="utf-8-sig").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        w, p = ln.split(None, 1)
        p = p.lstrip("*").strip()
        fp = (root / p) if (rel_ok and root is not None and not Path(p).is_absolute()) else Path(p)
        want, got = RC.hex64(w), (RC.sha256_file(fp) if fp.is_file() else "")
        pins[p] = want
        if len(want) != 64 or len(got) != 64 or want != got:
            bad.append({"path": p, "pinned": want, "now": got})
    if bad or not pins:
        raise SystemExit(f"[lh2] pin file {path}: {bad} -- refusing")
    return {"file": str(path), "sha256": RC.sha256_file(path), "n_pins": len(pins), "ALL_MATCH": True, "pins": pins}


def sep_below(c) -> bool:
    return bool(c.get("separated") and c.get("hi") is not None and c["hi"] < 0)


def sep_above(c) -> bool:
    return bool(c.get("separated") and c.get("lo") is not None and c["lo"] > 0)


def fmt(c) -> str:
    if not c or c.get("delta") is None:
        return "n/a"
    if not all(k in c for k in ("lo", "hi", "mean_new", "mean_ref", "n_windows", "n_episodes")):
        return json.dumps({k: c[k] for k in ("delta", "lo", "hi", "separated", "n_windows") if k in c}, default=str)
    return (f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" + (" SEP" if c.get("separated") else "")
            + f" ({c['mean_new']:.4f} vs {c['mean_ref']:.4f}, n={c['n_windows']}/{c['n_episodes']} ep)")


KEEP = ("delta", "lo", "hi", "separated", "n_windows", "n_episodes", "mean_new", "mean_ref", "n_dropped_nonfinite", "status")


def pcell(PR, new, ref, eid, alpha: float) -> dict:
    """pair_read_r8.paired_cell with the interval level exposed (same estimator, same B / seed, same finite-on-both-sides drop)."""
    from taniteval import ci
    new, ref = np.asarray(new, np.float64), np.asarray(ref, np.float64)
    keep = np.isfinite(new) & np.isfinite(ref)
    if keep.sum() == 0:
        return {"status": "REFUSED"}
    e = [x for x, k in zip(eid, keep) if k]
    r = ci.paired_episode_cluster_bootstrap(new[keep], ref[keep], e, n_boot=PR.N_BOOT, seed=PR.SEED, alpha=alpha)
    r["n_dropped_nonfinite"] = int((~keep).sum())
    r["mean_new"], r["mean_ref"] = float(new[keep].mean()), float(ref[keep].mean())
    out = {k: (float(r[k]) if isinstance(r.get(k), (float, np.floating)) else r.get(k)) for k in KEEP if k in r}
    out["ci_level"] = 1.0 - alpha
    return out


def manifest_sha(rel: str) -> str:
    for ln in LANDED_MAN.read_text(encoding="utf-8").splitlines():
        p = ln.split(None, 1)
        if len(p) == 2 and p[1].lstrip("*").strip() == rel:
            return RC.hex64(p[0])
    return ""


def main() -> int:
    t0 = time.time()
    prereg = RC.check_prereg()
    amend1 = RC.check_amendment1()
    fixpin = RC.check_fix_pin()
    seals = {k: seal(k, v) for k, v in SEALS.items()}
    void_ok = {k: RC.sha256_file(RC.RESULT / k) == v for k, v in VOID_FILES.items()}
    if not all(void_ok.values()):
        raise SystemExit(f"[lh2] a run-1 VOID record changed: {void_ok} -- refusing")
    reg_r9 = pinfile(RC.WORK / "REGISTERED_R9PF_PA.sha256")
    reg_lh = pinfile(LONHEAD / "REGISTERED_LH.sha256", root=LONHEAD, rel_ok=True)
    for k, v in LH_PINS.items():
        if reg_lh["pins"].get(k) != v:
            raise SystemExit(f"[lh2] REGISTERED_LH.sha256 does not pin {k} = {v} -- refusing")
    pa_bars_sha = RC.sha256_file(RC.RESULT / "R9PF_PA_BARS.json")
    if pa_bars_sha != PA_BARS_SHA:
        raise SystemExit(f"[lh2] R9PF_PA_BARS.json {pa_bars_sha} != {PA_BARS_SHA} -- refusing")
    # ---- precondition: K-OFF (run 1) + K-EQ under AMENDMENT2
    keq1 = json.loads((RC.RESULT / "R9PF_KEQ.json").read_text(encoding="utf-8"))
    pa2 = RC.RESULT / "R9PF_KEQ_AMEND2.json"
    keqa2 = json.loads(pa2.read_text(encoding="utf-8")) if pa2.is_file() else None
    precond = {"K-OFF_run1": bool(keq1["K-OFF"]["PASS"]), "K-EQ_run1_as_written": bool(keq1["K-EQ"]["PASS"]),
               "K-EQ_AMEND2": bool(keqa2 is not None and keqa2["K-EQ_AMEND2"]["PASS"]),
               "K-EQ_AMEND2_file": (None if keqa2 is None else {"file": str(pa2), "sha256": RC.sha256_file(pa2)})}
    precond["satisfied_in_the_AMENDMENT2_labelled_sense"] = bool(precond["K-OFF_run1"] and precond["K-EQ_AMEND2"])

    RC.setup_paths()
    sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
    sys.path.insert(0, str(LONHEAD / "code"))
    import torch
    import l2_common as L2
    import compliant_reread_r8 as CR
    import clamp_posthoc_r8 as CP
    import l1_fit_score as F
    import lh_fit as LF
    from tanitad.refs import refcv8_conditioning as r8c
    st = L2.stack()
    PR, ra = st["PR"], st["ra"]
    METRICS = [(m[0], m[1], bool(m[4])) for m in PR.METRICS]
    rc = F.RT.tools_load()["rc"]
    V_cmp, vrec = CP.load_vendored()
    R = RC.r9p()
    ctl = {}
    # ---- the map, READ from the R9PF-TL code path (refcv9_progress.trunk_lon_regime's own constants), never re-typed
    ids6 = tuple(int(x) for x in r8c.LON6_V7_IDS)
    map6 = tuple(int(x) for x in R.LON6_TO_REGIME)
    ctl["K-MAP-SOURCE"] = {"LON6_V7_IDS_from": "tanitad.refs.refcv8_conditioning.LON6_V7_IDS (the columns trunk_lon_regime index_selects)",
                           "LON6_TO_REGIME_from": "refcv9_progress.LON6_TO_REGIME (trunk_lon_regime's lookup)", "ids6": ids6, "map6": map6,
                           "LH_class_order": tuple(int(x) for x in LF.LON6_V7),
                           "PASS": bool(tuple(int(x) for x in LF.LON6_V7) == ids6 and len(map6) == 6)}
    kh = json.loads((LONHEAD / "result" / "K_HEAD.json").read_text(encoding="utf-8"))
    ctl["K-KHEAD"] = {"PASS": bool(kh.get("go") is True and kh["controls"]["ALL_PASS"] is True), "file": str(LONHEAD / "result" / "K_HEAD.json"),
                      "sha256": RC.sha256_file(LONHEAD / "result" / "K_HEAD.json")}
    LH = LF.load_frozen(LONHEAD / "LH_FROZEN.pt", LH_PINS["LH_FROZEN.pt"])
    DER = LF.load_frozen(LONHEAD / "DER_FROZEN.pt", LH_PINS["DER_FROZEN.pt"])
    labels = {t: np.load(LONHEAD / "labels" / f"labels_{t}.npz", allow_pickle=False) for t in ("LHS", "DER")}
    heads = {}
    for k in RC.FIT_SEEDS:
        for nm in (f"R9PF-TL_fs{k}", f"R9PF_fs{k}"):
            hp = RC.HEADS / f"{nm}.pt"
            heads[nm] = RC.load_head(hp, reg_r9["pins"][hp.as_posix()])
        a_sd, b_sd = heads[f"R9PF-TL_fs{k}"][0].state_dict(), heads[f"R9PF_fs{k}"][0].state_dict()
        ctl[f"K-SAMEPROFILES|fs{k}"] = {"rule": "R9PF-TL_fs{k} and R9PF_fs{k} carry the SAME fitted head (D4) -- state dicts bit-equal",
                                        "decision_TL": heads[f"R9PF-TL_fs{k}"][1]["decision"],
                                        "PASS": bool(heads[f"R9PF-TL_fs{k}"][1]["decision"] == "trunk_lon" and a_sd.keys() == b_sd.keys()
                                                     and all(torch.equal(a_sd[x], b_sd[x]) for x in a_sd))}
    # ---- beside packages pinned to the landed manifest
    pins = {}
    for kk, nm in BESIDE.items():
        for s in (0, 1):
            rel = f"out_mdfix/arms/h2cn_mdfix/{nm}/DEPLOY8_POS_{TREE}_s{s}.npz"
            got, want = RC.sha256_file(LANDED / nm / f"DEPLOY8_POS_{TREE}_s{s}.npz"), manifest_sha(rel)
            pins[f"{nm}|s{s}"] = {"sha256": got, "manifest": want, "match": bool(len(got) == 64 and got == want)}
    ctl["K-BESIDE-PINNED"] = {"PASS": all(v["match"] for v in pins.values()), "pins": pins}
    pa_bars = json.loads((RC.RESULT / "R9PF_PA_BARS.json").read_text(encoding="utf-8"))
    Rc, dec_stats, pkgs = {}, {}, {}
    PKG_OUT.mkdir(parents=True, exist_ok=True)
    for s in (0, 1):
        E = L2.load_eval_pf(s)
        I = np.load(PKG_IN / f"EVAL_INPUTS_{TREE}_s{s}.npz", allow_pickle=False)
        eid6, comp6 = list(E["eid"]), np.asarray(E["compliant"], bool)
        B = {kk: np.load(LANDED / nm / f"DEPLOY8_POS_{TREE}_s{s}.npz", allow_pickle=False) for kk, nm in BESIDE.items()}
        n2 = len(I["S2_ws"])
        k2 = {(int(c), int(w)): j for j, (c, w) in enumerate(zip(I["S2_clip"], I["S2_ws"]))}
        j6 = np.array([k2.get((int(c), int(w)), -1) for c, w in zip(I["S6_clip"], I["S6_ws"])])
        al = {"S6_inputs_eq_E": bool(np.array_equal(I["S6_ws"], E["ws"]) and np.array_equal(I["S6_clip"], E["clip"])),
              "S6_rows_in_S2": bool((j6 >= 0).all()),
              **{f"beside_{kk}": bool(np.array_equal(z["S6_ws"], E["ws"]) and np.array_equal(z["S2_ws"], I["S2_ws"])
                                      and np.array_equal(z["S2_clip"], I["S2_clip"])) for kk, z in B.items()}}
        # trunk lon logits (R9PF-TL's decision input), joined exactly as a3_compose.py
        ex = np.load(BATT / f"dump_s{s}" / "refcv6_extras.npz", allow_pickle=False)
        exk = {(int(c), int(w)): j for j, (c, w) in enumerate(zip(ex["clip_index"], ex["ws"]))}
        je = np.array([exk.get((int(c), int(w)), -1) for c, w in zip(I["S2_clip"], I["S2_ws"])])
        al["extras_all_joined"] = bool((je >= 0).all())
        lon_v7 = ex["nav_true.tacv6_lon_logits"][je].astype(np.float32)
        del ex
        ctl[f"K-ALIGN|s{s}"] = {"PASS": all(al.values()), "detail": al}
        # LH / DER inputs: the L1 eval bank (z_tac + fed v0), exactly as LONHEAD's k_head.py / build_labels.py
        zb = F.load_bank(RC.EVAL_BANK / f"ztac_eval_s{s}.npz")
        jz = np.array([zb["idx"].get((int(c), int(w)), -1) for c, w in zip(I["S2_clip"], I["S2_ws"])])
        Zl, v0l = zb["z"][jz], np.asarray(zb["v0_in"], np.float64)[jz]
        ctl[f"K-Z|s{s}"] = {"rule": "LH's z_tac (L1 bank, float64) == the composition's S2_Z bit for bit (as float32); every S2 row joined",
                            "PASS": bool((jz >= 0).all() and np.array_equal(Zl.astype(np.float32), I["S2_Z"].astype(np.float32))),
                            "max_abs": float(np.abs(Zl - I["S2_Z"].astype(np.float64)).max())}
        lg = {"LH": LF.predict_logits6(LH, Zl, v0l), "DER": LF.predict_logits6(DER, Zl, v0l)}
        ids = np.asarray(ids6, np.int64)
        for t, lab in (("LH", "LHS"), ("DER", "DER")):
            L = labels[lab]
            kl = {(int(c), int(w)): int(v) for c, w, v in zip(L["clip_index"], L["ws"], L["lon_v7"])}
            want = np.array([kl.get((int(c), int(w)), -999) for c, w in zip(I["S2_clip"], I["S2_ws"])])
            got = ids[lg[t].argmax(1)]
            ctl[f"K-LABELS|{t}|s{s}"] = {"rule": f"{t} argmax (float64 logits, LON6 -> v7) == LONHEAD labels_{lab}.npz on every S2 window",
                                         "n": int(n2), "n_agree": int((want == got).sum()), "PASS": bool((want == got).all() and len(kl) == n2)}
        # the regime posterior (SOFT): LON6 softmax summed over the map's groups
        def post3(l6):
            z = l6 - l6.max(1, keepdims=True)
            p6 = np.exp(z) / np.exp(z).sum(1, keepdims=True)
            P = np.zeros((len(l6), 3), np.float64)
            for c, r in enumerate(map6):
                P[:, r] += p6[:, c]
            return P

        def embed(l6):
            a = np.zeros((n2, lon_v7.shape[1]), np.float32)
            a[:, ids] = np.asarray(l6, np.float32)
            return a

        plan2, Z2, X2, P2 = I["S2_plan_battery"], I["S2_Z"], I["S2_x9"], I["S2_prior8"]
        for k in RC.FIT_SEEDS:
            h, blob, sh = heads[f"R9PF-TL_fs{k}"]
            ref = np.load(PKG_IN / f"R9PF-TL_fs{k}" / f"R9_POS_{TREE}_s{s}.npz", allow_pickle=False)
            # K-MAP: the R9LH-ARG function with the TRUNK lon head's LON6 logits == R9PF-TL's banked composition, bit for bit
            o_map = RC.compose(h, blob, Z2, X2, P2, plan2, embed(lon_v7[:, ids]))
            o_full = RC.compose(h, blob, Z2, X2, P2, plan2, lon_v7)
            ctl[f"K-MAP|fs{k}|s{s}"] = {"PASS": bool(np.array_equal(o_map["traj"].astype(np.float32), ref["S2_pos8"])
                                                     and np.array_equal(o_map["decision"], ref["S2_decision"])
                                                     and np.array_equal(o_full["traj"].astype(np.float32), ref["S2_pos8"])),
                                        "n": int(n2), "banked": str(PKG_IN / f"R9PF-TL_fs{k}" / f"R9_POS_{TREE}_s{s}.npz")}
            plans = {}
            o_lh = RC.compose(h, blob, Z2, X2, P2, plan2, embed(lg["LH"]))
            o_der = RC.compose(h, blob, Z2, X2, P2, plan2, embed(lg["DER"]))
            plans["R9LH-ARG"] = (o_lh["traj"].astype(np.float32), o_lh["decision"])
            plans["R9DER-ARG"] = (o_der["traj"].astype(np.float32), o_der["decision"])
            for t, o in (("LH", o_lh), ("DER", o_der)):
                want = np.asarray(map6, np.int64)[lg[t].argmax(1)]
                ctl.setdefault(f"K-ARGMAP|{t}|s{s}", {"rule": "the composed decision (float32 path) == map(float64 argmax)", "per_fit_seed": {}})
                ctl[f"K-ARGMAP|{t}|s{s}"]["per_fit_seed"][f"fs{k}"] = int((o["decision"] == want).sum())
                ctl[f"K-ARGMAP|{t}|s{s}"]["PASS"] = all(v == n2 for v in ctl[f"K-ARGMAP|{t}|s{s}"]["per_fit_seed"].values())
            prof, s_p = torch.from_numpy(o_lh["profiles"]), torch.from_numpy(o_lh["s_prior"])
            pl = torch.from_numpy(np.ascontiguousarray(plan2, np.float32))

            def soft_plan(P3):
                with torch.no_grad():
                    ds = (torch.from_numpy(np.ascontiguousarray(P3, np.float32))[:, :, None] * prof).sum(1)
                    return R.retime(pl, R.monotone_progress(s_p + ds)).to(torch.float32).numpy()
            P3 = post3(lg["LH"])
            soft = soft_plan(P3)
            plans["R9LH-SOFT"] = (soft, P3.argmax(1).astype(np.int64))
            oh = np.eye(3, dtype=np.float32)[o_lh["decision"]]
            ctl[f"K-SOFT1|fs{k}|s{s}"] = {"PASS": bool(np.array_equal(soft_plan(oh), plans["R9LH-ARG"][0])), "n": int(n2)}
            for a, (p8, dec) in plans.items():
                tag = f"{a}_fs{k}"
                pos2, pos6 = p8[:, RC.SLOTS["S2"]], p8[j6][:, RC.SLOTS["S6"]]
                pkgs.setdefault(tag, {})[f"s{s}"] = {"S2_pos8": p8, "S2_pos": pos2, "S6_pos": pos6, "S2_decision": dec, "S6_decision": dec[j6]}
                d = PKG_OUT / tag
                d.mkdir(parents=True, exist_ok=True)
                meta = {"tool": "lh_r9pf_run2.py", "arm": a, "fit_seed": k, "head_file": f"R9PF-TL_fs{k}.pt", "head_sha256": sh, "inference_seed": s,
                        "tier": TIER, "path": "the battery tree's banked fb47a58 plan (AMENDMENT1 change 4)", "slots": RC.SLOTS}
                np.savez(d / f"R9_POS_{TREE}_s{s}.npz", meta_json=np.asarray(json.dumps(meta)), S2_pos8=p8, S2_pos=pos2, S6_pos=pos6,
                         S2_decision=dec, S6_decision=dec[j6], S2_clip=I["S2_clip"], S2_ws=I["S2_ws"], S6_clip=I["S6_clip"], S6_ws=I["S6_ws"])
            pkgs[f"KMAP_TL_fs{k}"] = {**pkgs.get(f"KMAP_TL_fs{k}", {}),
                                      f"s{s}": {"S2_pos8": o_map["traj"].astype(np.float32), "S2_pos": o_map["traj"].astype(np.float32)[:, RC.SLOTS["S2"]],
                                                "S6_pos": o_map["traj"].astype(np.float32)[j6][:, RC.SLOTS["S6"]]}}
        # ---- scoring (as a3_bars.py)
        surf = {"S2": {"G": I["S2_G"].astype(np.float32), "dt": float(I["S2_dt"]), "eid": list(I["S2_eid"]), "mask": I["S2_compliant"].astype(bool),
                       "HP": I["S2_HP"].astype(np.float32)},
                "S6": {"G": E["g"].astype(np.float32), "dt": float(E["dt"]), "eid": eid6, "mask": comp6, "HP": I["S6_HP"].astype(np.float32)}}
        arms_pos = {t: {sf: v[f"s{s}"][f"{sf}_pos"] for sf in ("S2", "S6")} for t, v in pkgs.items()}
        arms_pos.update({kk: {sf: z[f"{sf}_pos"] for sf in ("S2", "S6")} for kk, z in B.items()})
        arms_pos["HP"] = {sf: surf[sf]["HP"] for sf in ("S2", "S6")}
        comps = {a: {sf: PR.components(ra, np.asarray(p[sf], np.float32), surf[sf]["G"], surf[sf]["dt"]) for sf in ("S2", "S6")} for a, p in arms_pos.items()}
        ind = {a: L2.lon_indicators(np.asarray(p["S6"], np.float32), E, comp6) for a, p in arms_pos.items() if a != "HP"}
        fdI, mdI = {a: v[0] for a, v in ind.items()}, {a: v[1] for a, v in ind.items()}
        ctl[f"K-ZERO|s{s}"] = {"PASS": bool(pcell(PR, mdI["H8"], mdI["H8"], eid6, A_NS)["delta"] == 0.0)}
        c_ref = PR.paired_cell(np.asarray(mdI["H8"], np.float64), np.asarray(mdI["FX"], np.float64), eid6)
        c_me = pcell(PR, mdI["H8"], mdI["FX"], eid6, 0.05)
        ctl[f"K-CELL|s{s}"] = {"rule": "pcell(alpha 0.05) == pair_read_r8.paired_cell on H8-vs-FX MD", "PASS": all(c_me[x] == c_ref[x] for x in ("delta", "lo", "hi", "separated"))}
        lim2 = I["S2_lim"].astype(np.float32)
        exc = {}
        for a in list(pkgs) + list(B):
            p8 = pkgs[a][f"s{s}"]["S2_pos8"] if a in pkgs else B[a]["S2_pos8"]
            pm = CR.pmax(V_cmp, np.asarray(p8, np.float32), [int(hh) for hh in rc["horizons"]], rc["tick_s"])
            exc[a] = (np.isfinite(lim2) & ((pm - lim2).astype(np.float32) > 0)).astype(np.float64)
        ctl[f"K-EXC|s{s}"] = {"PASS": bool(int(exc["H8"].sum()) == int(pa_bars["cells"]["H8"][f"s{s}"]["EXC_n"])), "mine": int(exc["H8"].sum()),
                              "registered_bars": pa_bars["cells"]["H8"][f"s{s}"]["EXC_n"]}
        eid2 = surf["S2"]["eid"]
        r_star, _dv = RC.regime_independent(I["S6_G"][:, 4], I["S6_G"][:, 5], I["S6_v0"])
        for t in pkgs:
            r = Rc.setdefault(t, {}).setdefault(f"s{s}", {})
            for lvl, al_ in (("95", A_NS), ("975", A_SEP)):
                r[f"MD_vs_H8_{lvl}"], r[f"FD_vs_H8_{lvl}"] = pcell(PR, mdI[t], mdI["H8"], eid6, al_), pcell(PR, fdI[t], fdI["H8"], eid6, al_)
                r[f"MD_vs_FX_{lvl}"], r[f"FD_vs_FX_{lvl}"] = pcell(PR, mdI[t], mdI["FX"], eid6, al_), pcell(PR, fdI[t], fdI["FX"], eid6, al_)
                for sf in ("S2", "S6"):
                    m = surf[sf]["mask"]
                    em = [e for e, kk in zip(surf[sf]["eid"], m) if kk]
                    r[f"ADE_{sf}_vs_D8_{lvl}"] = pcell(PR, comps[t][sf]["ade_m"][m], comps["D8"][sf]["ade_m"][m], em, al_)
                    r[f"ADE_{sf}_vs_HP_{lvl}"] = pcell(PR, comps[t][sf]["ade_m"][m], comps["HP"][sf]["ade_m"][m], em, al_)
            r["MD_rate"], r["FD_rate"] = float(np.nanmean(mdI[t])), float(np.nanmean(fdI[t]))
            r["MD_n"], r["FD_n"] = int(np.isfinite(mdI[t]).sum()), int(np.isfinite(fdI[t]).sum())
            if t.startswith("KMAP"):
                continue
            for sf in ("S2", "S6"):
                m = surf[sf]["mask"]
                em = [e for e, kk in zip(surf[sf]["eid"], m) if kk]
                r[f"fam_{sf}_vs_H8_95"] = {k_: {**pcell(PR, comps[t][sf][k_][m], comps["H8"][sf][k_][m], em, A_NS), "family": fam, "lower_is_better": lib}
                                           for fam, k_, lib in METRICS if k_ in comps[t][sf]}
            r["EXC_vs_H8_95"] = pcell(PR, exc[t], exc["H8"], eid2, A_NS)
            r["EXC_rate"], r["EXC_n"] = float(exc[t].mean()), int(exc[t].sum())
            dz = pkgs[t][f"s{s}"]["S6_decision"]
            r["TACTICAL_decision_vs_GT_regime_S6_compliant"] = {
                "rows_GT_regime_cols_decision (0 decel, 1 cruise, 2 accel)": [[int(((r_star == i) & (dz == jj) & comp6).sum()) for jj in range(3)] for i in range(3)],
                "accuracy": float(((r_star == dz) & comp6).sum() / comp6.sum()),
                "decel_recall": float(((r_star == 0) & (dz == 0) & comp6).sum() / max(((r_star == 0) & comp6).sum(), 1)),
                "false_decel_on_GT_cruise": float(((r_star == 1) & (dz == 0) & comp6).sum() / max(((r_star == 1) & comp6).sum(), 1)),
                "n_S6_compliant": int(comp6.sum()), "note": "for R9LH-SOFT the 'decision' is the posterior's argmax (the plan uses the full posterior)"}
        # K-SCORE: the K-MAP plans scored here reproduce the registered bars' R9PF-TL cells (95 %, 4 dp)
        for k in RC.FIT_SEEDS:
            mine, reg = Rc[f"KMAP_TL_fs{k}"][f"s{s}"], pa_bars["cells"][f"R9PF-TL_fs{k}"][f"s{s}"]
            ok = all(abs(mine[f"{x}_95"][q] - reg[x][q]) < 1e-12 for x in ("MD_vs_H8", "FD_vs_H8", "MD_vs_FX", "FD_vs_FX", "ADE_S2_vs_D8", "ADE_S6_vs_HP")
                     for q in ("delta", "lo", "hi"))
            ctl[f"K-SCORE|fs{k}|s{s}"] = {"PASS": bool(ok and abs(mine["MD_rate"] - reg["MD_rate"]) < 1e-12)}
        dec_stats[f"s{s}"] = {"n_S2": int(n2), "n_S6": int(len(j6)), "n_S6_compliant": int(comp6.sum()), "n_S2_compliant": int(surf["S2"]["mask"].sum()),
                              "LH_regime_counts_S2": np.bincount(np.asarray(map6)[lg["LH"].argmax(1)], minlength=3).tolist(),
                              "DER_regime_counts_S2": np.bincount(np.asarray(map6)[lg["DER"].argmax(1)], minlength=3).tolist(),
                              "trunk_regime_counts_S2": np.bincount(np.asarray(map6)[lon_v7[:, ids].argmax(1)], minlength=3).tolist()}
        print(f"[lh2] seed {s} done ({time.time() - t0:.0f} s)", flush=True)
    # ---------------------------------------------------------------- bars
    def pa1a(t, s, alpha_sep="975"):
        r = Rc[t][f"s{s}"]
        return bool(sep_below(r[f"MD_vs_H8_{alpha_sep}"]) and not sep_above(r["FD_vs_H8_95"]))

    def pa1b(t, s):
        r = Rc[t][f"s{s}"]
        return bool(sep_below(r["FD_vs_FX_975"]) and not sep_above(r["MD_vs_FX_95"]))

    def pa2(t, s, lvl="975"):
        r = Rc[t][f"s{s}"]
        return {"vs_RETIME": bool(sep_below(r[f"ADE_S2_vs_D8_{lvl}"]) and sep_below(r[f"ADE_S6_vs_D8_{lvl}"])),
                "E1_E3_vs_ha0_ext_pose": bool(sep_below(r[f"ADE_S2_vs_HP_{lvl}"]) and sep_below(r[f"ADE_S6_vs_HP_{lvl}"]))}

    def pag(t):
        worse = []
        for s in (0, 1):
            r = Rc[t][f"s{s}"]
            for sf in ("S2", "S6"):
                for k_, c in r[f"fam_{sf}_vs_H8_95"].items():
                    if (c["lower_is_better"] and sep_above(c)) or (not c["lower_is_better"] and sep_below(c)):
                        worse.append(f"s{s}|{sf}|{c['family']}|{k_}: {fmt(c)}")
            if sep_above(r["EXC_vs_H8_95"]):
                worse.append(f"s{s}|exceedance (full plan, all windows): {fmt(r['EXC_vs_H8_95'])}")
        return worse

    bars = {}
    for a in ARMS:
        per_fs = {}
        for k in RC.FIT_SEEDS:
            t = f"{a}_fs{k}"
            p2 = {f"s{s}": pa2(t, s) for s in (0, 1)}
            w = pag(t)
            per_fs[f"fs{k}"] = {"PA-1a_per_seed": {f"s{s}": pa1a(t, s) for s in (0, 1)}, "PA-1b_per_seed": {f"s{s}": pa1b(t, s) for s in (0, 1)},
                                "PA-2_per_seed": p2, "PA-2_per_seed_at_95_beside": {f"s{s}": pa2(t, s, "95") for s in (0, 1)},
                                "PA-G_worse_rows": w, "PA-1a_at_95_beside": {f"s{s}": pa1a(t, s, "95") for s in (0, 1)}}
            v = per_fs[f"fs{k}"]
            v["PA-1a"] = all(v["PA-1a_per_seed"].values())
            v["PA-1b"] = all(v["PA-1b_per_seed"].values())
            v["PA-2"] = all(x["vs_RETIME"] and x["E1_E3_vs_ha0_ext_pose"] for x in p2.values())
            v["PA-G"] = not w
        p = per_fs["fs0"]
        pa_r = {fs: {x: per_fs[fs][x] for x in ("PA-1a", "PA-1b", "PA-2")} for fs in per_fs}
        pa_r_holds = all(all(v.values()) for v in pa_r.values())
        pag_all = all(per_fs[fs]["PA-G"] for fs in per_fs)
        arm_pass = bool(p["PA-1a"] and p["PA-1b"] and p["PA-2"] and pag_all and pa_r_holds)
        bars[a] = {"PA-1a": p["PA-1a"], "PA-1b": p["PA-1b"], "PA-2": p["PA-2"], "PA-G": pag_all, "PA-G_fs0_only_beside": p["PA-G"],
                   "PA-R": pa_r_holds, "PA-R_detail": pa_r, "PASS": arm_pass, "per_fit_seed": per_fs,
                   "class": ("PASS" if arm_pass else ("PARTIAL" if p["PA-1a"] != p["PA-1b"] else "FAIL"))}
    der = bars["R9DER-ARG"]["per_fit_seed"]
    der_975 = {fs: v["PA-1a"] for fs, v in der.items()}
    der_95 = {fs: all(v["PA-1a_at_95_beside"].values()) for fs, v in der.items()}
    ctl["K-DER"] = {"rule": "R9DER-ARG must NOT pass PA-1a at ANY fit seed (both inference seeds, as the bar defines); stricter: also not at the 95 % "
                            "separation reading", "PA-1a_97.5": der_975, "PA-1a_95_stricter": der_95,
                    "PASS": not any(der_975.values()) and not any(der_95.values())}
    ctl["K-PRECONDITION"] = {"PASS": precond["satisfied_in_the_AMENDMENT2_labelled_sense"], **precond}
    failed = [k for k, v in ctl.items() if isinstance(v, dict) and "PASS" in v and not v["PASS"]]
    if failed:
        word = "VOID"
    elif any(bars[a]["PASS"] for a in ("R9LH-ARG", "R9LH-SOFT")):
        word = "PASS"
    elif any(bars[a]["class"] == "PARTIAL" for a in ("R9LH-ARG", "R9LH-SOFT")):
        word = "PARTIAL"
    else:
        word = "FAIL"
    outcome = {"VOID": "VOID -- a control failed (named in failed_controls); no reading of the bars.",
               "PASS": "s5 PASS (either arm): the first post-hoc composition beyond the MD / FD frontier: (i) refcv8 candidate -- a SHIP-R8 amendment "
                       "needs a PI go; (ii) refcv9 design = R9 profiles + a clean-split decision head (the Phase-B request carries this number).",
               "PARTIAL": "s5 PA-1a PASS, PA-1b FAIL (or the reverse): a new operating point, not a frontier move; next = a decision head with richer "
                          "supervision (the regime itself as the target on the fresh split, LH-architecture), pre-registered first.",
               "FAIL": "s5 Both FAIL: the decision information reachable from frozen z_tac (AUC ~0.73) is not enough for this transfer; the lever is "
                       "training the decision INTO the trunk (Phase B, PI compute, D-R9-INMODEL Q3)."}[word]
    # ---- beside references (read, never re-scored): R9PF / R9PF-TL / FX / D8 / H8 (registered bars), ORACLE (exploratory attribution), LHS_FACT8
    xa = json.loads((RC.RESULT / "R9PF_X_ATTRIB.json").read_text(encoding="utf-8"))
    lv = json.loads((LONHEAD / "result" / "LONHEAD_VERDICT.json").read_text(encoding="utf-8"))
    bes = {}
    for s in (0, 1):
        b = {}
        for t in [f"R9PF_fs{k}" for k in RC.FIT_SEEDS] + [f"R9PF-TL_fs{k}" for k in RC.FIT_SEEDS] + ["FX", "D8"]:
            c = pa_bars["cells"][t][f"s{s}"]
            b[t] = {"MD_rate": c["MD_rate"], "FD_rate": c["FD_rate"], "EXC_rate": c.get("EXC_rate"), "MD_vs_H8_95": fmt(c["MD_vs_H8"]),
                    "FD_vs_H8_95": fmt(c["FD_vs_H8"])}
        b["H2CN_DEPLOY8"] = pa_bars["cells"]["H8"][f"s{s}"]
        b["ORACLE_EXPLORATORY_inadmissible"] = {f"fs{k}": {x: (fmt(v) if isinstance(v, dict) else v) for x, v in xa["cells"][f"R9PF_fs{k}"]["ORACLE"][f"s{s}"].items()
                                                            if x in ("MD_rate", "FD_rate", "MD_vs_H8", "FD_vs_H8", "ADE_S6_vs_H8")} for k in RC.FIT_SEEDS}
        ps = lv["per_seed"][str(s)]
        b["LHS_FACT8"] = {"rates_S6_compliant": ps["rates_S6_compliant"].get("LHS_FACT8"),
                          "MD_vs_H8": fmt(ps["cells"]["F2_MD_LHS_vs_H8"]), "FD_vs_H8": fmt(ps["cells"]["F2_FD_LHS_vs_H8"])}
        bes[f"s{s}"] = b
    out = {"tool": "lh_r9pf_run2.py", "script_sha256": RC.sha256_file(__file__), "prereg": seals["PREREG_R9PF_LH.md"], "parent_prereg": prereg,
           "amendment1": amend1, "amendment2": seals["PREREG_REFCV9_INMODEL_PA_AMENDMENT2.md"], "fix_pin": fixpin,
           "registrations": {"REGISTERED_R9PF_PA": {k: v for k, v in reg_r9.items() if k != "pins"}, "REGISTERED_LH": reg_lh},
           "label": "R9PF_LH run 2 -- precondition satisfied in the AMENDMENT2 labelled sense (K-EQ post-failure, device-matched)",
           "word": word, "committed_outcome_s5": outcome, "tier": TIER, "estimator": EST,
           "interval_rule": "required separations (PA-1a MD vs H8, PA-1b FD vs FX, PA-2 four ADE cells) on the 97.5 % interval; 'not separated above / "
                            "worse' clauses (PA-1a FD, PA-1b MD, PA-G family rows + exceedance) on the 95 % interval; both from the SAME bootstrap draw",
           "bars": bars, "controls": ctl, "failed_controls": failed, "decision_stats": dec_stats, "cells": Rc, "beside_references": bes,
           "families_note": "ADE / LONGITUDINAL / LATERAL / TACTICAL = pair_read_r8.METRICS (14 rows) vs H2CN_DEPLOY8 on S2 and S6 compliant; TACTICAL "
                            "also = the regime decision vs the GT regime (S6 compliant confusion). STRATEGIC n/a (refcv8 runs --no-strategic: no strategic "
                            "head to score; n = 0). Distance-keeping (headway / TTC) n/a: no lead-agent join in the eval139 panel.",
           "packages": str(PKG_OUT), "vendored_selection": vrec, "wall_s": round(time.time() - t0, 1), "generated": time.strftime("%FT%T%z")}
    dest = (RC.RESULT / "R9PF_LH_VERDICT_RUN2.json") if precond["satisfied_in_the_AMENDMENT2_labelled_sense"] else (RC.SCRATCH / "R9PF_LH_RUN2_PROVISIONAL_NO_KEQ.json")
    RC.jdump(out, dest)
    print(f"[lh2] word {word} failed={failed} -> {dest}  ZZLH2_{word}ZZ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
