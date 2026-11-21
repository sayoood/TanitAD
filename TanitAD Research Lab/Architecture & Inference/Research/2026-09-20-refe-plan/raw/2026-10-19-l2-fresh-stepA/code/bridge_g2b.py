#!/usr/bin/env python3
"""READ 2 (universe D) G2b BRIDGE gate.  ONE GPU phase, ~5 min: run ONLY through `launch_stepA_ctype.py run -- ...` (the dev-box GPU lock; `stepA.require_lock_for_gpu` exits 4 otherwise).  NOT RUN by the package.

WHY.  Universe D has no banked P2 fan, so G2b ("the bundle forward reproduces the banked fan, logits and pick bit for bit") cannot be the literal L1 gate there.  The ruling's replacement: P2's fan comes from the
SAME bundle code, which must FIRST reproduce the banked fan bit for bit on B tokens (IDENTITY), with a RED arm that proves the identity can see a change (SENSITIVITY).

WHAT (both arms run the bundle script ITSELF, byte-identical, through `runpy`; nothing of its forward is re-implemented here):
  IDENTITY  `<bundle-script> --sample <3 B tokens>` must exit 0 and print ZZBUNDLE_OK: its own G2b-fan gate compared the repaired raw trajectory with the banked proposals20, the logits and the pick, bit for bit.
            This wrapper ALSO records every forward's outputs (a patch on `REFePlanner.infer`, read-only) and re-compares them to the banked table itself, so the verdict does not rest on the script's own say-so.
  RED       the same run with ONE input perturbed: `REFePlanner._image_for` returns the image with one pixel of one channel of camera 0 raised by one 8-bit level (1/255).  The bundle script's own gate MUST then
            fire (exit 1, ZZBUNDLE_FAIL, bad == number of tokens) AND the independent re-comparison must read "differs" on EVERY token AND the perturbation must have been applied once per token (`fired`).
            A RED arm whose perturbation never reached the forward, or whose gate stayed silent, FAILS the bridge: the gate is then blind and the identity above certifies nothing.
PASS = IDENTITY on all tokens AND RED on all tokens.  Markers: ZZBRIDGE_IDENTITY_OK n, ZZBRIDGE_RED_OK n, ZZBRIDGE_PASS | ZZBRIDGE_FAIL <why>.  `--dry-run` (CPU, no model, no GPU) checks the inputs and prints the plan.

    <locked>  C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -u bridge_g2b.py --ckpt <p2/model_final.pt> --plan <heldout_tokens.json> --sample <bridge3_sample.json> --table <table_p2_heldout.npz> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import runpy
import sys
import time

import numpy as np

L1_CODE = os.environ.get("L1_CODE", "C:/Users/Admin/par18/l1run/code")
L1_SCRIPT = os.path.join(L1_CODE, "make_stepA_bundle.py")
L1_SCRIPT_SHA256_LF = "8e544b8676c18c537076a52243df26a761ae299397fc25ec69653eedd40aa237"        # LF sha256 of l1run/code/make_stepA_bundle.py as the B chain runs it (2026-10-19); the bridge certifies THIS script
PROPOSALS20 = "D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18/proposals.npz"
TREE_DEFAULT = "C:/Users/Admin/p2heldout/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"


class Tee(io.TextIOBase):
    def __init__(self, real):
        self.real, self.buf = real, io.StringIO()

    def write(self, s):
        self.buf.write(s)
        return self.real.write(s)

    def flush(self):
        self.real.flush()


def sha_lf(p):
    return hashlib.sha256(open(p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def tree_paths():
    tree = os.environ.get("L1_TREE", TREE_DEFAULT)
    return [os.path.join(tree, "refe"), os.path.join(tree, "eval")]


def compare_to_bank(rec: dict, bank_logits, bank_pick, bank_p20) -> dict:
    """independent bit-for-bit re-comparison of one recorded forward with the banked rows.  rec = (traj_raw [64,20,3] f32, logits [64,6] f32, pick)."""
    traj, lg, k = rec
    rep = np.array(traj, np.float32, copy=True)
    rep[:, -1, 2] = rep[:, -2, 2]                                         # planner.repair_last_heading, as make_stepA_bundle.py does
    eq_fan, eq_lg, eq_pick = bool(np.array_equal(rep, bank_p20)), bool(np.array_equal(lg, bank_logits)), bool(int(k) == int(bank_pick))
    return {"fan_equal": eq_fan, "logits_equal": eq_lg, "pick_equal": eq_pick, "all_equal": bool(eq_fan and eq_lg and eq_pick),
            "max_abs_dlogit": float(np.abs(np.asarray(lg, np.float64) - np.asarray(bank_logits, np.float64)).max()), "max_abs_dtraj": float(np.abs(rep.astype(np.float64) - bank_p20.astype(np.float64)).max())}


def run_arm(script: str, argv: list, red: bool) -> dict:
    """run the bundle script through runpy with the recording patch (and, for RED, the one-pixel perturbation).  Returns rc, the script's stdout, the recorded forwards and the perturbation counter."""
    for p in reversed(tree_paths()):
        if p not in sys.path:
            sys.path.insert(0, p)
    import planner as PLM
    state = {"fired": 0, "calls": {}}
    orig_image_for, orig_infer = PLM.REFePlanner._image_for, PLM.REFePlanner.infer

    def image_for(self, ego):
        img = orig_image_for(self, ego)
        if red and img is not None:
            img = img.clone()
            img[0, 0, 0, img.shape[-2] // 2, img.shape[-1] // 2] += 1.0 / 255.0         # ONE pixel, ONE channel, camera 0, one 8-bit level
            state["fired"] += 1
        return img

    def infer(self, ego, img):
        traj, score, k = orig_infer(self, ego, img)
        state["calls"][str(self._scenario._initial_lidar_token)] = (traj.float().cpu().numpy().astype(np.float32), score.float().cpu().numpy().astype(np.float32), int(k))
        return traj, score, k

    PLM.REFePlanner._image_for, PLM.REFePlanner.infer = image_for, infer
    tee, rc = Tee(sys.stdout), 0
    old_argv = sys.argv
    sys.argv = [script] + list(argv)
    try:
        with contextlib.redirect_stdout(tee):
            try:
                runpy.run_path(script, run_name="__main__")
            except SystemExit as e:
                rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        sys.argv = old_argv
        PLM.REFePlanner._image_for, PLM.REFePlanner.infer = orig_image_for, orig_infer
    return {"rc": int(rc), "stdout": tee.buf.getvalue(), "fired": state["fired"], "calls": state["calls"]}


def judge(identity: dict, red: dict, tokens: list, bank: dict) -> dict:
    """the pure verdict (tested without a GPU)"""
    n = len(tokens)
    out = {"n_tokens": n, "identity": {"rc": identity["rc"], "script_said_ok": "ZZBUNDLE_OK" in identity["stdout"], "per_token": {}}, "red": {"rc": red["rc"], "script_said_fail": "ZZBUNDLE_FAIL" in red["stdout"],
           "perturbation_applied": red["fired"], "per_token": {}}}
    for t in tokens:
        for arm, name in ((identity, "identity"), (red, "red")):
            out[name]["per_token"][t] = compare_to_bank(arm["calls"][t], *bank[t]) if t in arm["calls"] else {"all_equal": None, "missing": True}
    idn, rd = out["identity"], out["red"]
    idn["all_tokens_bit_equal"] = bool(all(v.get("all_equal") is True for v in idn["per_token"].values()) and len(idn["per_token"]) == n)
    idn["pass"] = bool(idn["rc"] == 0 and idn["script_said_ok"] and idn["all_tokens_bit_equal"])
    rd["all_tokens_differ"] = bool(all(v.get("all_equal") is False for v in rd["per_token"].values()) and len(rd["per_token"]) == n)
    why = []
    if rd["perturbation_applied"] != n:
        why.append(f"the perturbation was applied {rd['perturbation_applied']} times for {n} tokens (it never reached the forward on some token)")
    if not (rd["rc"] == 1 and rd["script_said_fail"]):
        why.append("the bundle script's own gate did not fire under the perturbation (exit 1 + ZZBUNDLE_FAIL expected)")
    if not rd["all_tokens_differ"]:
        why.append("a perturbed forward still equals the banked rows on some token (the bit-for-bit comparison cannot see a one-level pixel change)")
    rd["why_not_pass"] = why
    rd["pass"] = not why
    out["pass"] = bool(idn["pass"] and rd["pass"])
    return out


def load_bank(table_path: str, p20_path: str, tokens: list) -> dict:
    t = np.load(table_path, allow_pickle=True)
    ix = {str(k): i for i, k in enumerate(t["token"])}
    p = np.load(p20_path, allow_pickle=True)
    pix = {str(k): i for i, k in enumerate(p["token"])}
    P20 = p["proposals20"]
    missing = [x for x in tokens if x not in ix or x not in pix]
    if missing:
        raise SystemExit(f"{len(missing)} bridge tokens are not in the banked table / proposals20: {missing[:3]}")
    return {x: (np.asarray(t["logits"][ix[x]], np.float32), int(t["pick"][ix[x]]), np.asarray(P20[pix[x]], np.float32)) for x in tokens}


def dry_run(a, tokens) -> dict:
    """CPU, no model, no GPU: every bridge token has a banked row and a plan entry, its DB exists, and its four frames resolve with the planner's own FrameResolver"""
    plan = json.load(open(a.plan, encoding="utf-8"))
    load_bank(a.table, a.proposals20, tokens)
    for p in reversed(tree_paths()):
        sys.path.insert(0, p)
    import planner as PLM
    rec = {"script": a.bundle_script, "script_sha256_lf": sha_lf(a.bundle_script), "tokens": {}}
    for t in tokens:
        lg = plan["token_log"][t]
        db = plan["db"][lg]
        fr = PLM.FrameResolver(db_dir="", images_root=plan["token_root"][t])
        fr.db_dir = os.path.dirname(db)
        rec["tokens"][t] = {"log": lg, "db_exists": os.path.exists(db), "frame_root": plan["token_root"][t]}
        paths = fr.resolve(lg, int(plan["token_ts_us"][t])) if os.path.exists(db) else None
        rec["tokens"][t]["frames_resolve"] = bool(paths)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--proposals20", default=PROPOSALS20)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bundle-script", default=L1_SCRIPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    tokens = sorted(json.load(open(a.sample, encoding="utf-8"))["tokens"])
    same = os.path.normcase(os.path.abspath(a.bundle_script)) == os.path.normcase(os.path.abspath(L1_SCRIPT))
    if same and sha_lf(a.bundle_script) != L1_SCRIPT_SHA256_LF:
        raise SystemExit(f"REFUSED: {a.bundle_script} is not the script this bridge certifies (sha256 {sha_lf(a.bundle_script)[:12]} != {L1_SCRIPT_SHA256_LF[:12]}); pass --bundle-script for another one")
    if len(tokens) < 3:
        raise SystemExit("the bridge needs >= 3 tokens")
    os.makedirs(a.out_dir, exist_ok=True)
    if a.dry_run:
        print("ZZBRIDGE_DRYRUN", json.dumps(dry_run(a, tokens)), flush=True)
        return 0
    bank = load_bank(a.table, a.proposals20, tokens)
    t0 = time.time()
    common = ["--ckpt", a.ckpt, "--plan", a.plan, "--sample", a.sample, "--table", a.table, "--proposals20", a.proposals20, "--device", a.device]
    ident = run_arm(a.bundle_script, common + ["--out-dir", os.path.join(a.out_dir, "identity")], red=False)
    red = run_arm(a.bundle_script, common + ["--out-dir", os.path.join(a.out_dir, "red")], red=True)
    res = judge(ident, red, tokens, bank)
    res.update(script=a.bundle_script, script_sha256_lf=sha_lf(a.bundle_script), tokens=tokens, sample_sha256=hashlib.sha256(open(a.sample, "rb").read()).hexdigest(), seconds=round(time.time() - t0, 1),
               what="G2b bridge: IDENTITY of the bundle forward against the banked fan on B tokens + a one-pixel RED arm")
    json.dump(res, open(os.path.join(a.out_dir, "bridge_g2b.json"), "w"), indent=1)
    if res["identity"]["pass"]:
        print(f"ZZBRIDGE_IDENTITY_OK {len(tokens)}", flush=True)
    if res["red"]["pass"]:
        print(f"ZZBRIDGE_RED_OK {len(tokens)}", flush=True)
    if res["pass"]:
        print("ZZBRIDGE_PASS", flush=True)
        return 0
    print("ZZBRIDGE_FAIL", json.dumps({"identity_pass": res["identity"]["pass"], "red_why": res["red"]["why_not_pass"]}), flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
