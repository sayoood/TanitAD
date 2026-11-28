#!/usr/bin/env python3
"""G-BRIDGE for the navtest builder: ONE GPU phase (~3 min), run ONLY through `launch_stepA_ctype.py run -- ...` (the dev-box GPU lock).  NOT RUN by the package.

Same design as the certified `l2fresh/code/bridge_g2b.py`, with the reference swapped from the held-out fan table to P2's BANKED NAVTEST SEAM.  Both arms run `make_navtest_bundle.py` ITSELF through `runpy`:
  IDENTITY  on 3 tokens the builder must exit 0 and print ZZBUNDLE_OK (its own G-IDENT gate), AND this wrapper records every forward (a read-only patch on `REFePlanner.infer`) and re-compares to the banked seam rows
            itself, byte for byte, so the verdict does not rest on the builder's say-so.
  RED       the same run with ONE pixel of camera 0 raised by one 8-bit level: the builder's gate MUST fire (exit 1, ZZBUNDLE_FAIL) AND the independent re-comparison must read "differs" on EVERY token AND the
            perturbation must have been applied once per token.  A RED arm that never reached the forward, or whose gate stayed silent, FAILS the bridge (the identity above would certify nothing).
PASS = IDENTITY on all tokens AND RED on all tokens.  Markers ZZBRIDGE_IDENTITY_OK n, ZZBRIDGE_RED_OK n, ZZBRIDGE_PASS | ZZBRIDGE_FAIL <why>.  `judge` is pure and tested without a GPU.
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

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.environ.get("L1_TREE", "C:/Users/Admin/p2heldout/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")      # dev-box checkout of the refe model tree (NOT repo-relative: the chain sets L1_TREE to the hybrid tree)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import navpipe_core as NC  # noqa: E402
NC.add_code_paths()                                    # L1's code: the repo-relative sibling `l1_stepA` (+ its patched run_2026-10-19), L1_CODE overrides
L1_CODE = NC.L1_CODE
BUILDER = os.path.join(HERE, "make_navtest_bundle.py")


class Tee(io.TextIOBase):
    def __init__(self, real):
        self.real, self.buf = real, io.StringIO()

    def write(self, s):
        self.buf.write(s)
        return self.real.write(s)

    def flush(self):
        self.real.flush()


def compare_to_bank(rec: tuple, ref, token: str = None, seam=None) -> dict:
    """one recorded forward (traj_raw [64,20,3] f32, logits [64,6] f32, pick) against P2's banked reference (kind-specific level: seam = byte equality of to_navsim(repair(traj)[pick]); table = all 64 proposals,
    logits and pick; s0fan = logits and pick).  A bare pose array is accepted for the legacy seam form."""
    traj, lg, k = rec
    if not hasattr(ref, "equal"):
        got = NC.exec_to_poses(NC.repair_last_heading(np.asarray(traj, np.float32)[int(k)])[None], seam)[0]
        eq = bool(got.tobytes() == np.asarray(ref, np.float32).tobytes())
        return {"pose_equal": eq, "all_equal": eq, "max_abs_dpose_m": float(np.abs(got.astype(np.float64) - np.asarray(ref, np.float64)).max())}
    eq = bool(ref.equal(token, traj, lg, int(k), seam))
    return {"all_equal": eq, "kind": ref.kind}


def judge(identity: dict, red: dict, tokens: list, banked, seam=None) -> dict:
    """the pure verdict (tested without a GPU); `banked` is a reference object (NC.make_reference) or the legacy banked-seam dict"""
    if not hasattr(banked, "equal"):
        banked = NC.SeamRef.from_dict(banked)
    n = len(tokens)
    out = {"n_tokens": n, "identity": {"rc": identity["rc"], "script_said_ok": "ZZBUNDLE_OK" in identity["stdout"], "per_token": {}},
           "red": {"rc": red["rc"], "script_said_fail": "ZZBUNDLE_FAIL" in red["stdout"], "perturbation_applied": red["fired"], "per_token": {}}}
    for t in tokens:
        for arm, name in ((identity, "identity"), (red, "red")):
            out[name]["per_token"][t] = compare_to_bank(arm["calls"][t], banked, t, seam) if t in arm["calls"] else {"all_equal": None, "missing": True}
    idn, rd = out["identity"], out["red"]
    idn["all_tokens_bit_equal"] = bool(all(v.get("all_equal") is True for v in idn["per_token"].values()) and len(idn["per_token"]) == n)
    idn["pass"] = bool(idn["rc"] == 0 and idn["script_said_ok"] and idn["all_tokens_bit_equal"])
    rd["all_tokens_differ"] = bool(all(v.get("all_equal") is False for v in rd["per_token"].values()) and len(rd["per_token"]) == n)
    why = []
    if rd["perturbation_applied"] != n:                                                    # @GUARD:BR-FIRED
        why.append(f"the perturbation was applied {rd['perturbation_applied']} times for {n} tokens (it never reached the forward on some token)")
    if not (rd["rc"] == 1 and rd["script_said_fail"]):                                      # @GUARD:BR-GATE-FIRES
        why.append("the builder's own gate did not fire under the perturbation (exit 1 + ZZBUNDLE_FAIL expected)")
    if not rd["all_tokens_differ"]:                                                         # @GUARD:BR-DIFFER
        why.append("a perturbed forward still equals the banked pick on some token (the byte comparison cannot see a one-level pixel change)")
    rd["why_not_pass"] = why
    rd["pass"] = not why
    out["pass"] = bool(idn["pass"] and rd["pass"])
    return out


def run_arm(script: str, argv: list, red: bool) -> dict:
    for p in reversed([os.path.join(TREE, "refe"), os.path.join(TREE, "eval")]):
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--sample", required=True, help="{'tokens': [3 navtest tokens]}")
    ap.add_argument("--banked-seam", default=None)
    ap.add_argument("--reference-kind", default="seam", choices=("seam", "table", "s0fan"))
    ap.add_argument("--reference-path", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    ref_path = a.reference_path or a.banked_seam
    tokens = sorted(json.load(open(a.sample, encoding="utf-8"))["tokens"])
    if len(tokens) < 3:
        raise SystemExit("the bridge needs >= 3 tokens")
    banked = NC.make_reference(a.reference_kind, ref_path)
    miss = [t for t in tokens if not banked.has(t)]
    if miss:
        raise SystemExit(f"bridge tokens not in the banked {a.reference_kind} reference: {miss[:3]}")
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    common = ["--ckpt", a.ckpt, "--plan", a.plan, "--sample", a.sample, "--reference-kind", a.reference_kind, "--reference-path", ref_path, "--device", a.device]
    ident = run_arm(BUILDER, common + ["--out-dir", os.path.join(a.out_dir, "identity")], red=False)
    red = run_arm(BUILDER, common + ["--out-dir", os.path.join(a.out_dir, "red")], red=True)
    res = judge(ident, red, tokens, banked)
    res.update(script=BUILDER, script_sha256_lf=NC.sha_lf(BUILDER), tokens=tokens, sample_sha256=hashlib.sha256(open(a.sample, "rb").read()).hexdigest(), seconds=round(time.time() - t0, 1),
               what="navtest G-IDENT bridge: IDENTITY of the builder's forward against P2's banked navtest seam + a one-pixel RED arm")
    json.dump(res, open(os.path.join(a.out_dir, "bridge_navtest.json"), "w"), indent=1)
    if res["identity"]["pass"]:
        print(f"ZZBRIDGE_IDENTITY_OK {len(tokens)}", flush=True)
    if res["red"]["pass"]:
        print(f"ZZBRIDGE_RED_OK {len(tokens)}", flush=True)
    if res["pass"]:
        print("ZZBRIDGE_PASS", flush=True)
        return 0
    print("ZZBRIDGE_FAIL", json.dumps({"identity_pass": res["identity"]["pass"], "red_why": res["red"]["why_not_pass"]})[:600], flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
