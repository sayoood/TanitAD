#!/usr/bin/env python3
"""PREREG_V8F_OACC box-2 stages that are neither training nor the standard eval: the w_acc RULE and the HX weight FIT. CPU, forward only (no gradient, no training).

  python v8f_mini_eval_oacc.py wacc --ckpt <M2b_s0 ckpt_last.pt> --expect-sha256 <sha> --fit-clips <fit_clips.json> --cache-dir <train cache> --join-train <join_train.npz> --tracks <native_tracks.pt> \
        --banked-fit-windows <fit_windows_M2b_s0.npz> --out w_acc.json --out-txt w_acc.txt
  python v8f_mini_eval_oacc.py wacc --from-npz <fit_windows_M2b_s0.npz> --out w_acc.json --out-txt w_acc.txt          (OFFLINE: the rule on HBLEND's banked B_s0 outputs, no model)
  python v8f_mini_eval_oacc.py hx   --ckpt <X ckpt_last.pt> --expect-sha256 <sha> --fit-clips ... --cache-dir ... --join-train ... --tracks ... --tag M2b-oacc_s0 --out weights_M2b-oacc_s0.json

THE RULE (PREREG_V8F_OACC, "w_acc is fixed by a rule, computed once before launch"):     w_acc = 0.5 * L_traj(B_s0) / L_acc(B_s0)
Both terms are evaluated with B's s0 checkpoint on the TRAINING windows of HBLEND's fit set (141 clips, k = 32, 9,582 windows; ``fit_clips.json``), by the checkpoint's own held outputs
(``forward_infer``: p_gt 0, T's own code, the eval's own age schedule -- HBLEND's convention, so the arrays ARE ``fit_windows_M2b_s0.npz``'s).

DECLARED INTERPRETATIONS (the PREREG names the rule and not these; each is fixed HERE, before any OACC number exists, and recorded in w_acc.json):
  R1  L_traj = the trainer's OWN trajectory contribution to its total, ``w_traj * (ade / 3 + 0.5 * fde / 6)`` (``v8f_mini.V8FMini.loss``), mean over the fit windows, with ``ade`` / ``fde`` from
      ``v8f_mini.traj_terms`` -- the very function the loss calls. (ADE alone would be a different number: ``alt_readings_not_used`` carries it, deciding nothing.)
  R2  L_acc = ``v8f_mini.accel_term`` at UNIT weight on the same arrays: the term the trainer logs under ``T["acc"]``. So w_acc * L_acc = 0.5 * L_traj: the term enters B's loss at HALF the trajectory term.
  R3  the value is rounded to 6 significant digits ONCE; that number is in the JSON, in the .txt the queue reads, and in X's argv, so the argv, the JSON and the built config are one value.
Numerics: float32 torch, the trainer's own functions. A cross-check against the banked HBLEND arrays (``--banked-fit-windows``) must agree to 1e-3 relative (the model route is CPU on box 2, the banked
arrays were made CPU on the dev box): a larger gap means the fit set or the checkpoint is not HBLEND's, and the stage REFUSES (rc 2).

THE HX FIT (``hx``): PREREG "HX is X with HBLEND's per-step weights re-fitted on X's own training windows (the deployed form)". The closed form, the clip to [0, 1] and the isotonic (PAV) step are
HBLEND's own ``fit_weights`` (imported from ``raw/2026-10-25-hblend/v8f_hblend.py``, never re-implemented); only the checkpoint changes. The output json carries HBLEND's keys (``w_along_by_step`` /
``w_cross_by_step``), which ``code/v8f_oacc_read.py`` reads."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PKG = os.path.join(REPO, "TanitAD Research Lab", "Architecture & Inference", "Research", "2026-10-17-v8f-mini")
HBLEND_PY = os.path.join(PKG, "raw", "2026-10-25-hblend", "v8f_hblend.py")
FRAC = 0.5                      # PREREG: w_acc = 0.5 * L_traj / L_acc
SIG_DIGITS = 6                  # R3
CROSS_RTOL = 1e-3               # model route vs the banked arrays


def refuse(msg):
    print(f"OACC EVAL REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def round_sig(x: float, n: int = SIG_DIGITS) -> float:
    """R3: ``x`` to ``n`` significant digits, as the float that ``repr`` of the printed text parses back to."""
    return float(f"{x:.{n}g}")


def load_hblend():
    """HBLEND's module, loaded by path (it is a research-package file, not a package member). Its import side effects are harmless: it defaults CUDA off and appends two dev-box scratch directories to sys.path."""
    spec = importlib.util.spec_from_file_location("v8f_hblend_for_oacc", HBLEND_PY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------------------------------ the rule (pure)
def rule_terms(traj30, gt, gok, w_traj: float = 1.0) -> dict:
    """``traj30`` / ``gt`` [N, 30, 2] (ego frame at NOW), ``gok`` [N] -> the rule's terms, from the TRAINER's own functions. ``L_traj`` = ``w_traj * (ade / 3 + 0.5 * fde / 6)`` (R1); ``L_acc`` = accel_term (R2)."""
    import torch
    from tanitad.models import v8f_mini as M
    t = torch.as_tensor(np.asarray(traj30), dtype=torch.float32)
    g = torch.as_tensor(np.asarray(gt), dtype=torch.float32)
    k = torch.as_tensor(np.asarray(gok), dtype=torch.float32)
    ade, fde = M.traj_terms(t, g, k)
    acc = M.accel_term(t, g, k)
    ade, fde, acc = float(ade), float(fde), float(acc)
    return {"L_ade": ade, "L_fde": fde, "L_traj": float(w_traj) * (ade / 3.0 + 0.5 * fde / 6.0), "L_acc": acc, "n_windows": int(k.sum().item()), "w_traj": float(w_traj)}


def w_acc_from_terms(L_traj: float, L_acc: float, frac: float = FRAC) -> dict:
    """The rule itself: ``w_acc = frac * L_traj / L_acc``. Refuses a non-positive or non-finite term (a rule that divides by zero is not a rule)."""
    if not (math.isfinite(L_traj) and math.isfinite(L_acc) and L_traj > 0.0 and L_acc > 0.0):
        refuse(f"the rule needs L_traj > 0 and L_acc > 0, both finite: got L_traj {L_traj}, L_acc {L_acc}")
    raw = frac * L_traj / L_acc
    return {"w_acc_raw": raw, "w_acc": round_sig(raw)}


def rule_record(R: dict, frac: float = FRAC) -> dict:
    v = w_acc_from_terms(R["L_traj"], R["L_acc"], frac)
    alt = w_acc_from_terms(R["w_traj"] * R["L_ade"], R["L_acc"], frac)
    return {"schema": "v8f-oacc-wacc-v1", "prereg": "PREREG_V8F_OACC.md", "rule": "w_acc = 0.5 * L_traj(B_s0) / L_acc(B_s0), both evaluated with B's s0 checkpoint on the training windows of HBLEND's fit set",
            "frac": frac, "L_traj_definition": "w_traj * (ade / 3 + 0.5 * fde / 6): the trainer's own trajectory contribution (R1); mean over the fit windows",
            "L_acc_definition": "v8f_mini.accel_term at unit weight on the same windows (R2)", **R, **v,
            "check_w_acc_x_L_acc_over_L_traj": v["w_acc"] * R["L_acc"] / R["L_traj"],
            "alt_readings_not_used": {"L_traj_ade_only": R["w_traj"] * R["L_ade"], "w_acc_if_L_traj_were_ade_only": alt["w_acc"],
                                      "note": "R1 is declared before any OACC number; the ADE-only reading is shown so the choice is visible, and decides nothing"}}


# ------------------------------------------------------------------------------------------------ the forward over the fit set
def load_ckpt_model(path, expect_sha=None):
    import torch
    from tanitad.models import v8f_mini as M
    if expect_sha is not None:
        got = sha256_file(path)
        if got != expect_sha:
            refuse(f"checkpoint {path} has sha256 {got}, expected {expect_sha}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    return model.eval(), cfg, ck


def forward_fit_set(model, cfg, ck, fit_clips, cache_dir, join_train, tracks, batch=64, threads=0, log=print):
    """HBLEND's ``cmd_fit`` loop, parameterised: the TRAINER's own window builder (``train_windows``) over ``fit_clips`` and the checkpoint's held outputs (``forward_infer``) on the eval's age schedule.
    -> dict of arrays traj30, p30, gt, gt_ok, ci, t, a_t, a_s."""
    import torch
    from tanitad.data import v8f_mini_data as D
    if threads:
        torch.set_num_threads(threads)
    J = D.load_join(join_train)
    tr = D.load_native_tracks(tracks) if cfg.kin_mode == "yaw" else None
    store = D.ClipStore(cache_dir, "train", lru=8, grid=ck.get("grid", "pool4x10"), kin_mode=cfg.kin_mode, tracks=tr)
    wins = D.train_windows(store, J, list(fit_clips), hist=cfg.hist, start=cfg.window_start)
    ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    N = len(wins)
    log(f"[oacc fit-set] clips {len(fit_clips)} windows {N} (the trainer's own builder: train_windows, hist {cfg.hist}, start {ds.window_start})")
    outs = {k: [] for k in ("traj30", "p30", "gt", "gt_ok", "ci", "t", "a_t", "a_s")}
    t0 = time.time()
    with torch.no_grad():
        for b0 in range(0, N, batch):
            items = []
            for i in range(b0, min(b0 + batch, N)):
                ci, t, _, _ = wins[i]
                a_t = i % 3                              # the EVAL's own deterministic age schedule (--ages cycle), on the window's index in the list
                a_s = a_t + (i // 3) % 5
                items.append(ds.item(ci, t, a_t, a_s))
            b = D.collate(items)
            o = model.forward_infer(b)
            outs["traj30"].append(o["traj30"].float().numpy())
            outs["p30"].append(o["p10"].float().numpy())
            for k in ("gt", "gt_ok", "ci", "t", "a_t", "a_s"):
                outs[k].append(b[k].numpy())
            if (b0 // batch) % 20 == 0:
                log(f"[oacc fit-set] {b0 + len(items)}/{N}  {time.time() - t0:.0f}s")
    O = {k: np.concatenate(v) for k, v in outs.items()}
    if not O["gt_ok"].all():
        refuse("a train window of the fit set has no 6 s future (gt_ok False): the fit set is not HBLEND's")
    return O


def fit_clips_of(path, max_clips=0):
    fc = json.load(open(path, encoding="utf-8"))
    clips = fc["clips"][:max_clips] if max_clips else fc["clips"]
    return fc, clips


# ------------------------------------------------------------------------------------------------ commands
def cmd_wacc(a):
    if a.from_npz:
        z = np.load(a.from_npz)
        src = {"route": "offline: HBLEND's banked arrays", "fit_windows": os.path.basename(a.from_npz), "fit_windows_sha256": sha256_file(a.from_npz)}
        R = rule_terms(z["traj30"], z["gt"], z["gt_ok"].astype(np.float32), a.w_traj)
        rec = rule_record(R)
        rec["source"] = src
    else:
        for need in ("ckpt", "fit_clips", "cache_dir", "join_train", "tracks"):
            if getattr(a, need) is None:
                refuse(f"--{need.replace('_', '-')} is required for the model route")
        model, cfg, ck = load_ckpt_model(a.ckpt, a.expect_sha256)
        if float(cfg.w_acc) != 0.0:
            refuse(f"the checkpoint was built with w_acc {cfg.w_acc}: the rule is evaluated on B (w_acc 0)")
        fc, clips = fit_clips_of(a.fit_clips, a.max_clips)
        O = forward_fit_set(model, cfg, ck, clips, a.cache_dir, a.join_train, a.tracks, a.batch, a.threads)
        R = rule_terms(O["traj30"], O["gt"], O["gt_ok"].astype(np.float32), cfg.w_traj)
        rec = rule_record(R)
        rec["source"] = {"route": "model: B_s0's checkpoint on the fit windows (forward_infer)", "ckpt": os.path.basename(os.path.dirname(os.path.abspath(a.ckpt))) + "/" + os.path.basename(a.ckpt),
                         "ckpt_sha256": sha256_file(a.ckpt), "fit_clips": os.path.basename(a.fit_clips), "fit_clips_sha256_declared": fc.get("sha256_newline_joined"), "k": fc.get("k"),
                         "n_clips": len(clips), "n_windows": int(len(O["gt"]))}
        if a.banked_fit_windows:
            z = np.load(a.banked_fit_windows)
            Rb = rule_terms(z["traj30"], z["gt"], z["gt_ok"].astype(np.float32), cfg.w_traj)
            rel = {k: abs(R[k] - Rb[k]) / max(abs(Rb[k]), 1e-12) for k in ("L_ade", "L_fde", "L_traj", "L_acc")}
            rec["banked_crosscheck"] = {"fit_windows": os.path.basename(a.banked_fit_windows), "fit_windows_sha256": sha256_file(a.banked_fit_windows), "rtol": CROSS_RTOL,
                                        "banked": {k: Rb[k] for k in ("L_ade", "L_fde", "L_traj", "L_acc", "n_windows")}, "relative_difference": rel,
                                        "max_abs_traj30_diff_m": float(np.abs(O["traj30"] - z["traj30"]).max()) if O["traj30"].shape == z["traj30"].shape else None,
                                        "same_windows": bool(O["traj30"].shape == z["traj30"].shape and np.array_equal(O["t"], z["t"]) and np.array_equal(O["ci"], z["ci"]))}
            rec["banked_crosscheck"]["pass"] = bool(max(rel.values()) <= CROSS_RTOL and rec["banked_crosscheck"]["same_windows"])
            if not rec["banked_crosscheck"]["pass"]:
                json.dump(rec, open(a.out, "w", encoding="utf-8"), indent=1)
                refuse(f"the model route disagrees with HBLEND's banked arrays: {rec['banked_crosscheck']} -- the fit set or the checkpoint is not HBLEND's")
    json.dump(rec, open(a.out, "w", encoding="utf-8"), indent=1)
    if a.out_txt:
        with open(a.out_txt, "w", encoding="ascii", newline="\n") as f:
            f.write(f"{rec['w_acc']!r}\n")
    print(f"OACC w_acc = {rec['w_acc']!r}  (L_traj {rec['L_traj']:.6f}, L_acc {rec['L_acc']:.6f}, n {rec['n_windows']}; w_acc x L_acc / L_traj = {rec['check_w_acc_x_L_acc_over_L_traj']:.6f})")
    return 0


def cmd_hx(a):
    for need in ("ckpt", "fit_clips", "cache_dir", "join_train", "tracks", "tag"):
        if getattr(a, need) is None:
            refuse(f"--{need.replace('_', '-')} is required")
    H = load_hblend()
    t0 = time.time()
    model, cfg, ck = load_ckpt_model(a.ckpt, a.expect_sha256)
    fc, clips = fit_clips_of(a.fit_clips, a.max_clips)
    O = forward_fit_set(model, cfg, ck, clips, a.cache_dir, a.join_train, a.tracks, a.batch, a.threads)
    R = H.fit_weights(O["traj30"], O["p30"], O["gt"])
    d = lambda p: np.linalg.norm(p.astype(np.float64) - O["gt"].astype(np.float64), axis=-1)      # noqa: E731
    Hf = H.apply_w(O["p30"], O["traj30"], R["w"])
    rec = {"schema": "v8f-oacc-hxfit-v1", "tag": a.tag, "ckpt_sha256": sha256_file(a.ckpt), "w_acc_built": float(cfg.w_acc), "k": fc.get("k"), "n_clips": len(clips),
           "fit_clips_sha256": fc.get("sha256_newline_joined"), "n_windows": int(len(O["gt"])),
           "fit": "HBLEND's own closed form + clip [0, 1] + isotonic (v8f_hblend.fit_weights), on THIS checkpoint's held outputs over the TRAIN windows only",
           "w_along_by_step": R["w"][:, 0].tolist(), "w_cross_by_step": R["w"][:, 1].tolist(), "raw_along": R["raw"][:, 0].tolist(), "raw_cross": R["raw"][:, 1].tolist(),
           "n_clipped_at_0": int((R["raw"] < 0).sum()), "n_clipped_at_1": int((R["raw"] > 1).sum()),
           "in_sample_train_ade6": {"tau": float(d(O["traj30"]).mean()), "P": float(d(O["p30"]).mean()), "H": float(d(Hf).mean())},
           "seconds": time.time() - t0}
    json.dump(rec, open(a.out, "w", encoding="utf-8"), indent=1)
    if a.fit_windows_out:
        np.savez_compressed(a.fit_windows_out, **O)
    print(f"OACC HX fit {a.tag}: {rec['n_windows']} windows, w_along[0,14,29] = {R['w'][0, 0]:.3f} {R['w'][14, 0]:.3f} {R['w'][29, 0]:.3f}; train ADE6 tau {rec['in_sample_train_ade6']['tau']:.4f} H {rec['in_sample_train_ade6']['H']:.4f}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("wacc", "hx"):
        p = sub.add_parser(name)
        p.add_argument("--ckpt")
        p.add_argument("--expect-sha256", dest="expect_sha256")
        p.add_argument("--fit-clips", dest="fit_clips", help="HBLEND's fit_clips.json (141 clips, k = 32)")
        p.add_argument("--cache-dir", dest="cache_dir")
        p.add_argument("--join-train", dest="join_train")
        p.add_argument("--tracks")
        p.add_argument("--batch", type=int, default=64)
        p.add_argument("--threads", type=int, default=0)
        p.add_argument("--max-clips", dest="max_clips", type=int, default=0, help="TEST INPUT: cap the fit clips")
        p.add_argument("--out", required=True)
    pw = sub.choices["wacc"]
    pw.add_argument("--from-npz", dest="from_npz", help="OFFLINE route: HBLEND's fit_windows_M2b_s0.npz (traj30, gt, gt_ok); no model")
    pw.add_argument("--banked-fit-windows", dest="banked_fit_windows", help="model route: cross-check against HBLEND's banked arrays (rel 1e-3)")
    pw.add_argument("--out-txt", dest="out_txt", help="the bare number the queue reads into X's --w-acc")
    pw.add_argument("--w-traj", dest="w_traj", type=float, default=1.0, help="OFFLINE route only: B's w_traj (the trainer default, 1.0)")
    ph = sub.choices["hx"]
    ph.add_argument("--tag", help="e.g. M2b-oacc_s0")
    ph.add_argument("--fit-windows-out", dest="fit_windows_out", help="save the fit-set arrays (npz)")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    return cmd_wacc(a) if a.mode == "wacc" else cmd_hx(a)


if __name__ == "__main__":
    sys.exit(main())
