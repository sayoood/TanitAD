"""OPTB read chain helpers (computed evidence, never typed). Subcommands: evidence | delta | bandmet | manifest.  Run through optb_route.py where a model is built (delta); the others need no model.
Exit codes of `evidence`: 0 ok (controls recorded, they are DATA: a false c2_c3_hold is an input to the label), 3 integrity (missing file, sha256 differs from RUN.done), 4 C5 / weights identity / init identity failed."""
import argparse, hashlib, json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.environ.get("OPTB_PKG", "C:/Users/Admin/optb_pkg")
LADDER_REF = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/ladder_read/results"
SCORE = "F:/TanitAD-scratch/actrsp/lonplan/score"
READ = "F:/TanitAD-scratch/actrsp/optb/read"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_checker():
    sys.path.insert(0, PKG)
    import optb_check as C
    return C


def cmd_evidence(a):
    C = load_checker()
    import torch
    out = {"tag": a.tag, "bank": a.bank, "problems": [], "dry": bool(a.dry)}
    lever = a.lever or C.TABLE[a.tag]["lever"]
    init = a.init if a.init is not None else C.TABLE[a.tag]["init"]
    out["lever"], out["init_chassis"] = lever, init
    rd = os.path.join(a.bank, "RUN.done")
    try:
        ev = json.load(open(rd, encoding="utf-8"))["evidence"]
    except Exception as e:
        out["problems"].append(f"RUN.done unreadable: {e}")
        return finish(out, a, 3)
    out["sha"] = {}
    for f, k in (("ckpt_last.pt", "ckpt_sha256"), ("summary.json", "summary_sha256"), ("train_log.jsonl", "train_log_sha256")):
        p = os.path.join(a.bank, f)
        if not os.path.isfile(p):
            out["problems"].append(f"{f} missing")
            continue
        got = sha256_file(p)
        out["sha"][f] = {"got": got, "want": ev.get(k), "equal": got == ev.get(k)}
        if got != ev.get(k):
            out["problems"].append(f"{f}: sha256 {got[:12]} != RUN.done {str(ev.get(k))[:12]}")
    bm = os.path.join(a.bank, "BANK_MANIFEST.json")
    if os.path.isfile(bm):
        files = json.load(open(bm, encoding="utf-8"))["files"]
        for f, v in files.items():
            p = os.path.join(a.bank, f)
            if not os.path.isfile(p) or sha256_file(p) != v["sha256"]:
                out["problems"].append(f"BANK_MANIFEST: {f} missing or sha256 differs")
    else:
        out["problems"].append("BANK_MANIFEST.json missing")
    if out["problems"]:
        return finish(out, a, 3)
    # ---- C2 (log identity) and C3 (step-0 identity, RSP arms), both read from the banked files
    gev, gp = C.gamma_log_problems(a.tag if not a.lever else ("FTRSP_s0" if lever == "rsp" else "FTNULL_s0"), os.path.join(a.bank, "train_log.jsonl"))
    out["c2"] = {"evidence": gev, "problems": gp}
    sc = json.load(open(os.path.join(a.bank, "selfcheck.json"), encoding="utf-8"))
    live = sc.get("live") if isinstance(sc.get("live"), dict) else {}
    if lever == "rsp":
        lf = sc.get("live_first") if isinstance(sc.get("live_first"), dict) else None
        cp = C.c3_problems(live, lf) if lf and "wm/rsp/L1" in lf else ["C3: live_first / wm/rsp/L1 missing from the banked selfcheck"]
    else:
        cp = [] if ("c3" not in live and "live_first" not in sc) else ["a NULL arm carries live.c3 / live_first"]
    out["c3"] = {"problems": cp, "c3": live.get("c3")}
    out["c2_hold"], out["c3_hold"] = not gp, not cp
    out["c2_c3_hold"] = out["c2_hold"] and out["c3_hold"]
    # ---- C5 + start-state identity, from the banked checkpoint itself
    init_p = os.path.join(a.init_dir, f"ladder_R35_s{init}", "ckpt_last.pt")
    if not os.path.isfile(init_p) or sha256_file(init_p) != C.INIT_SHA[init]:
        out["problems"].append(f"init checkpoint {init_p} missing or sha256 != INIT_SHA[{init}]")
        return finish(out, a, 4)
    A = torch.load(init_p, map_location="cpu", weights_only=False)
    B = torch.load(os.path.join(a.bank, "ckpt_last.pt"), map_location="cpu", weights_only=False)
    wev, wp = C.weights_identity(A["model"], B["model"])
    aev, ap_ = C.adam_state_problems(B)
    want = a.expect_adam if a.expect_adam is not None else C.N_WM_PARAMS - C.N_PROJ_TENSORS   # MM ruling 2026-11-18 (MM_RULINGS_POSTCHECK.md): 129 = 133 wm.* parameters - 4 wm.proj; bound to the checker's constants
    if not (aev.get("n_group_params") == want and aev.get("n_with_state") == want):
        ap_.append(f"C5: {aev.get('n_with_state')!r} of {aev.get('n_group_params')!r} optimiser tensors have Adam state, wanted {want}")
    pk = sorted(k for k in A["model"] if k.startswith("wm.proj."))
    pd = [k for k in pk if k not in B["model"] or not C.same_bits(A["model"][k], B["model"][k])]
    if len(pk) != C.N_PROJ_TENSORS:
        ap_.append(f"init holds {len(pk)} wm.proj tensors, wanted {C.N_PROJ_TENSORS}")
    if pd:
        ap_.append(f"wm.proj moved: {len(pd)} of {len(pk)} differ from the init checkpoint")
    out["c5"] = {"weights_identity": wev, "adam": aev, "wm_proj_tensors": len(pk), "wm_proj_differ": len(pd), "problems": wp + ap_}
    out["c5_hold"] = not (wp or ap_)
    if wp or ap_:
        out["problems"] += ["C5/identity: " + "; ".join(wp + ap_)[:500]]
        return finish(out, a, 4 if not a.dry else 0)
    return finish(out, a, 0)


def finish(out, a, rc):
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: out.get(k) for k in ("tag", "c2_hold", "c3_hold", "c2_c3_hold", "c5_hold", "problems")}))
    return rc


def cmd_delta(a):
    """delta in WM units of a +/-1.0 m/s^2 acceleration contrast, read from the arm's own WM scale_actions (frozen action scale). Prints one JSON line; run through optb_route.py."""
    import torch
    sys.path.insert(0, READ)
    import ia3pk_read as I
    wm, info, _ = I.load_ckpt(a.ckpt)
    z = torch.zeros(1, 1, 2)
    o = torch.zeros(1, 1, 2); o[..., 0] = 1.0
    d = (wm.scale_actions(o) - wm.scale_actions(z))[..., 0].item()
    d2 = (wm.scale_actions(2 * o) - wm.scale_actions(z))[..., 0].item()
    if not (abs(d2 - 2 * d) < 1e-6 * max(1.0, abs(d)) and d > 0):
        raise SystemExit(f"scale_actions is not an affine map on channel 0 ({d}, {d2}): the 1.0 m/s^2 contrast cannot be expressed as one delta")
    print(json.dumps({"delta_wm_1ms2": d, "term_delta_wm": I.DELTA_WM}))


def met(p):
    return bool(p["beats"]) and bool(p["healthy"]) and not bool(p["worse"])


def cmd_bandmet(a):
    """BAND-MET lost (card HARM; verdict_actsens.py:87-91 rule): R35 met the level (beats and healthy and not worse) and the arm does not. ia_read.json `passes` of the arm vs the banked ladder_read arm_R35_s{k}.json `passes`."""
    arm = json.load(open(a.ia, encoding="utf-8"))["passes"]
    ref = json.load(open(os.path.join(LADDER_REF, f"arm_R35_s{a.chassis}.json"), encoding="utf-8"))["passes"]
    lost = {lv: {"r35": ref[lv], "arm": arm[lv]} for lv in ("1", "2", "3") if met(ref[lv]) and not met(arm[lv])}
    res = {"tag": a.tag, "chassis": a.chassis, "r35_met": {lv: met(ref[lv]) for lv in ref}, "arm_met": {lv: met(arm[lv]) for lv in arm}, "lost_levels": sorted(lost), "band_met_lost": bool(lost), "detail": lost,
           "rule": "R35 met the level and the arm does not (verdict_actsens.py:87-91)"}
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: res[k] for k in ("tag", "lost_levels", "band_met_lost")}))


def cmd_manifest(a):
    """Assemble the optb_decide.py manifest from what exists under --root (plans/<tag>/, evidence/<tag>.json, bandmet/<tag>.json). controls.c2_c3_hold = all arms with evidence hold (null when none);
    band_met_lost per chassis = any arm of that chassis lost a level (null when no arm of that chassis has a read)."""
    sys.path.insert(0, PKG)
    C = load_checker()
    R = a.root
    arms, ev, bm = {}, {}, {}
    for t in C.RUN_ORDER:
        p = lambda s: os.path.join(R, "plans", t, f"plans_hwm_oracle_s{s}.npz")
        if os.path.isfile(os.path.join(R, "plans", t, "DONE")) and os.path.isfile(p(0)) and os.path.isfile(p(1)):
            arms[t] = {"s0": p(0).replace("\\", "/"), "s1": p(1).replace("\\", "/")}
        e = os.path.join(R, "evidence", f"{t}.json")
        if os.path.isfile(e):
            ev[t] = json.load(open(e, encoding="utf-8"))
        b = os.path.join(R, "bandmet", f"{t}.json")
        if os.path.isfile(b):
            bm[t] = json.load(open(b, encoding="utf-8"))
    base = {f"R35_s{k}": {"s0": f"{SCORE}/A0_s{k}/plans_hwm_oracle_s0.npz", "s1": f"{SCORE}/A0_s{k}/plans_hwm_oracle_s1.npz"} for k in (0, 1)}
    c2c3 = (all(v["c2_c3_hold"] for v in ev.values()) if ev else None)
    lost = {}
    for k in (0, 1):
        ts = [t for t in bm if C.TABLE[t]["init"] == k]
        lost[str(k)] = (any(bm[t]["band_met_lost"] for t in ts) if ts else None)
    mech = os.path.join(R, "mech.json")
    man = {"smoke": False, "n_boot": 2000, "windows": f"{SCORE}/A0_s0/windows.npz", "arms": arms, "baselines": base, "mech": mech.replace("\\", "/") if os.path.isfile(mech) else None,
           "controls": {"c2_c3_hold": c2c3, "band_met_lost": lost},
           "controls_detail": {t: {"c2_hold": ev[t].get("c2_hold"), "c3_hold": ev[t].get("c3_hold"), "c5_hold": ev[t].get("c5_hold"), "band_met_lost": bm.get(t, {}).get("lost_levels")} for t in ev},
           "note": "assembled by readchain_tools.py manifest; computed from banked files, nothing typed"}
    json.dump(man, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({"arms_planned": sorted(arms), "evidence": sorted(ev), "bandmet": sorted(bm), "controls": man["controls"]}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evidence")
    e.add_argument("--tag", required=True); e.add_argument("--bank", required=True); e.add_argument("--out", required=True)
    e.add_argument("--init-dir", default="C:/Users/Admin/t2v8f_pulls"); e.add_argument("--expect-adam", type=int, default=None)
    e.add_argument("--lever", default=None, choices=[None, "rsp", "null"]); e.add_argument("--init", type=int, default=None); e.add_argument("--dry", action="store_true")
    d = sub.add_parser("delta"); d.add_argument("--ckpt", required=True)
    b = sub.add_parser("bandmet"); b.add_argument("--ia", required=True); b.add_argument("--chassis", type=int, required=True); b.add_argument("--tag", required=True); b.add_argument("--out", required=True)
    m = sub.add_parser("manifest"); m.add_argument("--root", required=True); m.add_argument("--out", required=True)
    a = ap.parse_args()
    rc = {"evidence": cmd_evidence, "delta": cmd_delta, "bandmet": cmd_bandmet, "manifest": cmd_manifest}[a.cmd](a)
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())
