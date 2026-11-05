#!/usr/bin/env python3
"""llr_precheck.py -- the pulled-file controls of the LLR read chain, in seconds, BEFORE any rollout (raw/LLR_PRECHECK.json; exit 0 = PASS, 8 = a control failed, 5 = a file is absent).

Per seed s in {0, 1}, run dir  <rdb>/M2b_s{s}_recft_LLR/  and marker  <rdb>/llr_markers/RUN_LLR_{s}.done :
  KP1  ckpt_last.pt, SHA256SUMS.txt, argv.json exist (REFUSED when the checkpoint is absent: no LLR checkpoint, no read; exit 5)
  KP2  sha256(ckpt_last.pt) == the SHA256SUMS.txt line == the box-2 marker's evidence.ckpt_sha256 (a marker without one is refused)
  KP3  argv audit: the argv equals CLR's argv (<rdb>/M2b_s{s}_recft_CLR/argv.json) plus EXACTLY `--lead-head on --w-lead W_LEAD --aux-cond token` (``--out`` ignored); W_LEAD and WIRING from a2_prereg/FROZEN_LLR.txt
       (the same at both seeds), `--seed s`, `--w-level` / `--level-head` / every `--recover-*` token / `--init-ckpt` = CLR's, nothing removed, nothing changed
  KP4  the checkpoint's own cfg: level_head on, w_level = CLR's, lead_head on, w_lead = W_LEAD, aux_cond = the frozen wiring, step == the argv's --steps; the state dict carries lvl_head.*, lead_head.* and (token) e_aux.*,
       and NO e_level.* / e_lead.* (the additive path is not built under the token wiring); under `add` the reverse
  KP5  REPORTED (not a gate): sum |e_aux.weight| per input column (level, presence, gap) and sum |lead_head| of the trained checkpoint
  KP6  the box-2 marker (and the run's summary.json): recover_n > 0; level_head / w_level / lead_head / w_lead / aux_cond must MATCH where present (an absent key is recorded, the cfg check above decides)
  KP7  the attribution base = CLR's own checkpoint: M2b_s{s}_recft_CLR/ckpt_last.pt sha256 == its SHA256SUMS.txt line == clr_markers/RUN_CLR_{s}.done evidence.ckpt_sha256 == clr_read/raw/CLR_PRECHECK.json's (the checkpoint the
       banked CLR arrays came from); cfg level_head on, lead_head off.  Recorded as clr_ckpt_sha256
  KP8  the launch-gate token llr_markers/gate_token_LLR_{s}.json parses, its binding.commit == llr_launch/landed_tip.txt, w_lead / aux_cond match
No GPU; torch reads the checkpoint dicts only (CPU).   usage: python -I llr_precheck.py [--rdb E:/Projects/TanitAD-artifacts/v8f-recover] [--frozen-llr <file>] [--raw <dir>] [--no-ckpt-load]"""
import argparse
import importlib.util
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
DL = "E:/Projects/TanitAD-artifacts/dagland/motion_c"
RDB = "E:/Projects/TanitAD-artifacts/v8f-recover"
FROZEN_LLR = f"{DL}/a2_prereg/FROZEN_LLR.txt"
LANDED_TIP = f"{DL}/llr_launch/landed_tip.txt"
CLR_PRECHECK_JSON = f"{DL}/clr_read/raw/CLR_PRECHECK.json"
CP = f"{DL}/clr_read/code/clr_precheck.py"
CP_SHA = "11f15ec6b6f8155dbd65d2ee1f6215a0ea30e2a0d90c5368ba48101298437e49"      # the CLR pre-check (parse_argv, sha256_file, _same_num are reused); an edit of it is a refusal
LEAD_NEW = ("--lead-head", "--w-lead", "--aux-cond")


def _load_cp():
    spec = importlib.util.spec_from_file_location("clr_precheck_base", CP)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cp = _load_cp()
sha256_file, parse_argv, _same_num = cp.sha256_file, cp.parse_argv, cp._same_num


def read_frozen_llr(path):
    """FROZEN_LLR.txt KEY=VALUE lines; WIRING in (add, token) and W_LEAD a positive number are required.  Returns {WIRING, W_LEAD} as written."""
    d = {}
    for line in open(path, encoding="utf-8"):
        t = line.strip()
        if t and not t.startswith("#") and "=" in t:
            k, v = t.split("=", 1)
            d[k.strip()] = v.strip()
    if d.get("WIRING") not in ("add", "token"):
        raise ValueError(f"FROZEN_LLR.txt WIRING {d.get('WIRING')!r} is not add|token")
    if not float(d.get("W_LEAD", "nan")) > 0:
        raise ValueError(f"FROZEN_LLR.txt W_LEAD {d.get('W_LEAD')!r} is not a positive number")
    return {"WIRING": d["WIRING"], "W_LEAD": d["W_LEAD"]}


def argv_audit(llr_tokens, clr_tokens, s, w_lead, wiring):
    """LLR's argv = CLR's argv + `--lead-head on --w-lead W_LEAD --aux-cond WIRING`, nothing else (``--out`` ignored)."""
    a, b = parse_argv(llr_tokens), parse_argv(clr_tokens)
    x, r = {k: v for k, v in a.items() if k != "--out"}, {k: v for k, v in b.items() if k != "--out"}
    added, removed = {k: x[k] for k in x if k not in r}, sorted(k for k in r if k not in x)
    changed = {k: (r[k], x[k]) for k in x if k in r and r[k] != x[k]}
    c = {"added_exactly_the_three_lead_tokens": bool(set(added) == set(LEAD_NEW) and added.get("--lead-head") == "on"), "nothing_removed": not removed, "nothing_changed": not changed,
         "w_lead_is_the_frozen_W_LEAD": bool(added.get("--w-lead") is not None and _same_num(added.get("--w-lead"), w_lead)), "aux_cond_is_the_frozen_wiring": added.get("--aux-cond") == wiring,
         "seed_is_s": a.get("--seed") == str(s), "init_ckpt_is_CLRs": a.get("--init-ckpt") == b.get("--init-ckpt"), "level_tokens_are_CLRs": bool(a.get("--level-head") == "on" and a.get("--level-head") == b.get("--level-head") and a.get("--w-level") == b.get("--w-level"))}
    rec = lambda d: {k: v for k, v in d.items() if k.startswith("--recover")}
    p = str(rec(a).get("--recover-p", "0"))
    c["recovery_lever_is_CLRs"] = bool(rec(a) == rec(b) and p.replace(".", "", 1).isdigit() and float(p) > 0)
    c["diff"] = {"added": added, "removed": removed, "changed": {k: list(v) for k, v in changed.items()}}
    c["PASS"] = all(v for k, v in c.items() if k != "diff")
    return c


def sums_line(d, name="ckpt_last.pt"):
    return [l.split()[0] for l in open(f"{d}/SHA256SUMS.txt", encoding="utf-8") if l.strip().split()[-1].lstrip("*").endswith(name)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdb", default=RDB)
    ap.add_argument("--frozen-llr", default=FROZEN_LLR)
    ap.add_argument("--landed-tip", default=LANDED_TIP)
    ap.add_argument("--clr-precheck", default=CLR_PRECHECK_JSON)
    ap.add_argument("--raw", default=f"{PK}/raw")
    ap.add_argument("--no-ckpt-load", action="store_true", help="skip KP4/KP5/KP7's cfg (the checkpoint dicts are not opened)")
    a = ap.parse_args(argv)
    os.makedirs(a.raw, exist_ok=True)
    res = {"card": "LLR read chain pre-check (PREREG_LEVELHEAD_A2.md + A2_RULING_1.md + FROZEN_LLR.txt)", "rdb": a.rdb, "seeds": {}, "failures": [], "clock": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if cp.sha256_file(CP) != CP_SHA:
        res["failures"].append(f"the CLR pre-check {CP} is not the pinned file")
    try:
        fz = read_frozen_llr(a.frozen_llr)
    except Exception as e:
        res["failures"].append(f"FROZEN_LLR: {type(e).__name__}: {e}")
        fz = {"WIRING": None, "W_LEAD": None}
    wl, wiring = fz["W_LEAD"], fz["WIRING"]
    res["w_lead"], res["aux_cond"], res["frozen_llr_file"] = wl, wiring, a.frozen_llr
    tip = open(a.landed_tip, encoding="utf-8").read().split()[0] if os.path.isfile(a.landed_tip) else None
    res["landed_tip"] = tip
    try:
        clr_pc = json.load(open(a.clr_precheck, encoding="utf-8"))
    except Exception as e:
        clr_pc = None
        res["failures"].append(f"the CLR pre-check record {a.clr_precheck}: {type(e).__name__}: {e}")
    absent = False
    for s in (0, 1):
        k, d = f"s{s}", f"{a.rdb}/M2b_s{s}_recft_LLR"
        r = {"run_dir": d}
        for f in ("ckpt_last.pt", "SHA256SUMS.txt", "argv.json"):
            if not os.path.isfile(f"{d}/{f}"):
                res["failures"].append(f"{k}: {d}/{f} is absent")
                absent = True
        mk = f"{a.rdb}/llr_markers/RUN_LLR_{s}.done"
        if not os.path.isfile(mk):
            res["failures"].append(f"{k}: box-2 marker {mk} is absent")
            absent = True
        evd = {}
        if os.path.isfile(f"{d}/ckpt_last.pt") and os.path.isfile(f"{d}/SHA256SUMS.txt"):
            got, sums = sha256_file(f"{d}/ckpt_last.pt"), sums_line(d)
            r["ckpt_sha256"], r["sha256sums_line"] = got, sums[0] if sums else None
            if not sums or got != sums[0]:
                res["failures"].append(f"{k}: ckpt sha256 {got} != SHA256SUMS.txt {sums[0] if sums else None} (a partial copy?)")
            if os.path.isfile(mk):
                evd = json.load(open(mk, encoding="utf-8")).get("evidence") or {}
                ev = evd.get("ckpt_sha256")
                r["marker_ckpt_sha256"] = ev
                if not ev:
                    res["failures"].append(f"{k}: RUN_LLR_{s}.done carries no evidence.ckpt_sha256")
                elif ev != got:
                    res["failures"].append(f"{k}: ckpt sha256 {got} != box-2 marker {ev}")
        sm = {}
        if os.path.isfile(f"{d}/summary.json"):
            sm = json.load(open(f"{d}/summary.json", encoding="utf-8"))
        r["marker_and_summary_keys"] = {}
        for key, want in (("level_head", "on"), ("lead_head", "on"), ("aux_cond", wiring), ("w_lead", wl)):
            for src_name, src in (("marker", evd), ("summary", sm)):
                if key in src:
                    ok = _same_num(src[key], want) if key == "w_lead" else src[key] == want
                    r["marker_and_summary_keys"][f"{src_name}.{key}"] = {"value": src[key], "ok": bool(ok)}
                    if not ok:
                        res["failures"].append(f"{k}: {src_name} {key} {src[key]!r} is not {want!r}")
        rn = evd.get("recover_n", sm.get("recover_n"))
        r["recover_n"] = rn
        if not (isinstance(rn, (int, float)) and not isinstance(rn, bool) and rn > 0):
            res["failures"].append(f"{k}: recover_n {rn!r} (marker evidence / summary.json) is not > 0 (the recovery lever perturbed no sample)")
        cd = f"{a.rdb}/M2b_s{s}_recft_CLR"
        if os.path.isfile(f"{d}/argv.json") and os.path.isfile(f"{cd}/argv.json"):
            lt = json.load(open(f"{d}/argv.json", encoding="utf-8"))["argv"]
            ct = json.load(open(f"{cd}/argv.json", encoding="utf-8"))["argv"]
            r["argv_audit"] = argv_audit(lt, ct, s, wl, wiring)
            if not r["argv_audit"]["PASS"]:
                res["failures"].append(f"{k}: argv audit failed: {r['argv_audit']}")
            r["steps_argv"] = parse_argv(lt).get("--steps")
            r["w_level_argv"] = parse_argv(ct).get("--w-level")
        if os.path.isfile(f"{d}/ckpt_last.pt") and not a.no_ckpt_load:
            import torch
            ck = torch.load(f"{d}/ckpt_last.pt", map_location="cpu", weights_only=False)
            cfg, sd = ck["cfg"], ck["model"]
            pre = lambda *p: sorted(x for x in sd if x.startswith(p))
            lv, ld, ea, el, ee = pre("lvl_head."), pre("lead_head."), pre("e_aux."), pre("e_level."), pre("e_lead.")
            r.update({"cfg_level_head": cfg.get("level_head"), "cfg_w_level": cfg.get("w_level"), "cfg_lead_head": cfg.get("lead_head"), "cfg_w_lead": cfg.get("w_lead"), "cfg_aux_cond": cfg.get("aux_cond"), "step": ck.get("step"),
                      "keys": {"lvl_head": len(lv), "lead_head": len(ld), "e_aux": len(ea), "e_level": len(el), "e_lead": len(ee)}})
            if "e_aux.weight" in sd:
                wt = sd["e_aux.weight"].float().abs().sum(0)
                r["e_aux_abs_sum"] = float(wt.sum())
                r["e_aux_abs_sum_per_input_column_level_presence_gap"] = [float(x) for x in wt]
            r["lead_head_abs_sum"] = float(sum(sd[x].abs().sum() for x in ld))
            if cfg.get("level_head") != "on" or (r.get("w_level_argv") is not None and not _same_num(cfg.get("w_level"), r["w_level_argv"])):
                res["failures"].append(f"{k}: cfg level_head {cfg.get('level_head')!r} / w_level {cfg.get('w_level')!r} is not on / CLR's {r.get('w_level_argv')!r}")
            if cfg.get("lead_head") != "on" or (wl is not None and not _same_num(cfg.get("w_lead"), wl)):
                res["failures"].append(f"{k}: cfg lead_head {cfg.get('lead_head')!r} / w_lead {cfg.get('w_lead')!r} is not on / {wl!r}")
            if cfg.get("aux_cond") != wiring:
                res["failures"].append(f"{k}: cfg aux_cond {cfg.get('aux_cond')!r} is not the frozen wiring {wiring!r}")
            if not lv or not ld:
                res["failures"].append(f"{k}: the state dict lacks lvl_head.* ({len(lv)}) or lead_head.* ({len(ld)}) keys")
            if wiring == "token" and (not ea or el or ee):
                res["failures"].append(f"{k}: token wiring needs e_aux.* and no e_level.* / e_lead.* keys (e_aux {len(ea)}, e_level {len(el)}, e_lead {len(ee)})")
            if wiring == "add" and (ea or not el or not ee):
                res["failures"].append(f"{k}: add wiring needs e_level.* and e_lead.* and no e_aux.* keys (e_aux {len(ea)}, e_level {len(el)}, e_lead {len(ee)})")
            if r.get("steps_argv") is not None and str(ck.get("step")) != str(r["steps_argv"]):
                res["failures"].append(f"{k}: checkpoint step {ck.get('step')} != argv --steps {r['steps_argv']}")
        gt = f"{a.rdb}/llr_markers/gate_token_LLR_{s}.json"
        if not os.path.isfile(gt):
            res["failures"].append(f"{k}: launch-gate token {gt} is absent")
            absent = True
        else:
            try:
                b = json.load(open(gt, encoding="utf-8"))["binding"]
                r["gate_token_binding"] = {q: b.get(q) for q in ("commit", "w_lead", "aux_cond")}
                if tip is not None and b.get("commit") != tip:
                    res["failures"].append(f"{k}: gate token commit {b.get('commit')} != llr_launch/landed_tip.txt {tip}")
                if not _same_num(b.get("w_lead"), wl) or b.get("aux_cond") != wiring:
                    res["failures"].append(f"{k}: gate token w_lead / aux_cond {b.get('w_lead')!r} / {b.get('aux_cond')!r} != {wl!r} / {wiring!r}")
            except Exception as e:
                res["failures"].append(f"{k}: gate token {gt} unreadable: {type(e).__name__}: {e}")
        # KP7: the attribution base = CLR's own checkpoint (the banked CLR arrays' checkpoint)
        if os.path.isfile(f"{cd}/ckpt_last.pt") and os.path.isfile(f"{cd}/SHA256SUMS.txt"):
            gc, sc = sha256_file(f"{cd}/ckpt_last.pt"), sums_line(cd)
            r["clr_ckpt_sha256"] = gc
            if not sc or gc != sc[0]:
                res["failures"].append(f"{k}: CLR ckpt sha256 {gc} != its SHA256SUMS.txt {sc[0] if sc else None}")
            cmk = f"{a.rdb}/clr_markers/RUN_CLR_{s}.done"
            cev = (json.load(open(cmk, encoding="utf-8")).get("evidence") or {}).get("ckpt_sha256") if os.path.isfile(cmk) else None
            if cev != gc:
                res["failures"].append(f"{k}: CLR ckpt sha256 {gc} != clr_markers evidence {cev}")
            if clr_pc is not None and clr_pc["seeds"][k].get("ckpt_sha256") != gc:
                res["failures"].append(f"{k}: CLR ckpt sha256 {gc} != the CLR read's pre-check record {clr_pc['seeds'][k].get('ckpt_sha256')} (the banked CLR arrays came from another file)")
            if not a.no_ckpt_load:
                import torch
                bk = torch.load(f"{cd}/ckpt_last.pt", map_location="cpu", weights_only=False)["cfg"]
                r["clr_cfg_level_head"], r["clr_cfg_lead_head"] = bk.get("level_head"), bk.get("lead_head", "off")
                if bk.get("level_head") != "on" or bk.get("lead_head", "off") == "on":
                    res["failures"].append(f"{k}: the attribution base CLR must be level on, lead off (cfg {bk.get('level_head')!r} / {bk.get('lead_head', 'off')!r})")
        else:
            res["failures"].append(f"{k}: the attribution base {cd}/ckpt_last.pt or SHA256SUMS.txt is absent")
            absent = True
        res["seeds"][k] = r
    res["PASS"] = not res["failures"]
    json.dump(res, open(f"{a.raw}/LLR_PRECHECK.json", "w", encoding="utf-8", newline="\n"), indent=1)
    for f in res["failures"]:
        print("PRECHECK FAIL:", f)
    print("PRECHECK PASS" if res["PASS"] else "PRECHECK FAIL")
    return 0 if res["PASS"] else (5 if absent else 8)


if __name__ == "__main__":
    sys.exit(main())
