#!/usr/bin/env python3
"""r9_e2_make_argv.py -- the refcv9 E2 launch argv: K0 (regenerated from refcv8's launch argv) and K2's variants (composed from K0).

  K0       refcv8's launch argv + the E2 schedule ONLY (REFCV9_DESIGN 4.4): ``regen_k0(refcv8_doc)``. The committed ``r9e2_K0.argv.json`` MUST regenerate
           byte-equal (``--check-k0``); ``base_sha256`` = sha256 of the base's ``argv`` list as ``json.dumps(argv, sort_keys=True)`` (MEASURED: this
           reproduces the committed 4758a324...; the refcv8 FILE's own sha256 does not).
  K2 ...   ``compose(k0_doc, variant, w_drivable)``: K0 + the crowned E1 levers + the C8 planner wires, composed by MAP CONFIG (one named variant each).

  variant   map head                                  levers added to K0's argv
  K2        C0's 8-code head (no banded head)         T1h bundle + W-BOX (--r9-plan-box-tokens) + W-MAP-1 critic (non-banded)
  K2M1N20   banded, near band 0-20 m (E1's M1N20)     K2's + --r9-map-banded --r9-map-banded-near-m 20 (near lift 20) + W-MAP-2 (--r9-plan-map-channels) + banded critic
  K2M3      banded + own 128-ch encoder, near 0-10 m  K2's + --r9-map-banded --r9-map-own-enc (near lift 10) + W-MAP-2 + banded critic
  K2M3N20   M3's encoder on the 0-20 m near band      K2's + --r9-map-banded --r9-map-own-enc --r9-map-banded-near-m 20 (near lift 20) + W-MAP-2 + banded critic

The T1h bundle, the banded flags, the own-encoder flag and the near-band flag are built by ``r9_e1_make_argv`` (``t1_arm_argv`` / ``map_arm_argv``) -- the SAME code
that built the E1 arms' argv, never re-typed here. Only the C8 planner wires are this file's own.

``--w-r8-drivable`` (W-MAP-1) is a MEASURED quantity: REFCV9_DESIGN 2.7 W-COST sets it so the critic's weighted BCE is ~0.1 x the trajectory loss at step 0, "fixed from the
smoke, never from eval". The trainer prints ``[v3] refcv9 W-COST at step N: critic-BCE / trajectory-loss = R`` once (refc_v3_train.py, ``plan_wires_flag.wcost_ratio``) and the
weight is ``0.1 / R`` (``w_from_ratio``). The composer therefore takes the weight as a REQUIRED argument; a file written with ``--provisional`` is NOT the argv a token may bind.

  python r9_e2_make_argv.py --variant K2 --k0 <r9e2_K0.argv.json> --w-r8-drivable 0.1 --provisional --write <r9e2_K2.argv.json>
  python r9_e2_make_argv.py --variant K2 --k0 <r9e2_K0.argv.json> --w-cost-ratio R --w-cost-source <the explore smoke's log path> --write <r9e2_K2.argv.json>
  python r9_e2_make_argv.py --check-k0 --base <refcv8-r101-s0.argv.json> --k0 <r9e2_K0.argv.json>

K2-WCOST: the argv json records WHERE its weight came from, at the top level: ``w_cost_ratio`` (R) + ``w_cost_source`` (the smoke log path) with ``--w-cost-ratio``, and
``w_cost: "provisional"`` without it. The stamp is applied by ``stamp_w_cost`` in ``main`` only -- ``variant_doc`` (and so the committed K2 files' byte-equal regeneration) and
``regen_k0`` (K0) are untouched. The GATE does not trust the stamp: G-LIVE (``launch_gate.wcost_reasons``) holds ``--w-r8-drivable`` to 0.1 / R of ITS OWN smoke.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9_e1_make_argv as E1  # noqa: E402  (the E1 arm builders: reused, never copied)

RUN_OUT = "/home/nvidia/r8ns/r9e2/runs/r9e2-%s"
INIT_FROM = "/home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt"
JOIN_MASKS = "/home/nvidia/r8ns/r9e2/launch_tree/stack/tanitad/configs/refcv8_join_label_defects_train4572.json"
E2_STEPS, WSD_DECAY, GRAD_SHARE_AT = 6000, 1000, "1000,3000,5000"

#: K0 = refcv8's launch argv with exactly these flags SET (replace in place), then these APPENDED. The text is the committed K0's own (byte-equal regeneration).
K0_SET = (("--out", [RUN_OUT % "K0"]), ("--init-from", [INIT_FROM]), ("--steps", [str(E2_STEPS)]),
          ("--grad-share-at", [GRAD_SHARE_AT]), ("--join-defect-masks", [JOIN_MASKS]))
K0_APPEND = (("--lr-schedule", ["wsd"]), ("--wsd-decay-steps", [str(WSD_DECAY)]))
K0_WHAT = ("K0 = refcv8's launch recipe + the E2 schedule ONLY (REFCV9_DESIGN 4.4): no refcv9 lever, planner + tactical built, init refcv8 EMA-48,852. "
           "K2 / K3 are composed AFTER E1 reads: see C:/Users/Admin/qland/hold_r9_c12full/HANDOFF.md (the lever slot template).")
K0_SET_LINES = [
    "--out /home/nvidia/r8ns/r9e2/runs/r9e2-K0  <- the K0 run directory",
    "--init-from /home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt  <- REFCV9_DESIGN 4.4: init refcv8 EMA-48,852 (md5 b4364029b68f29a33eeb4aebbbbd002b, bound by the profile)",
    "--steps 6000  <- E2: 6,000 steps",
    "--lr-schedule wsd  <- E2: WSD (C6)",
    "--wsd-decay-steps 1000  <- E2: decay over the last 1,000 steps",
    "--grad-share-at 1000,3000,5000  <- refcv8's 1000,16250,32500,48750 pass --steps 6000 (the trainer refuses a reading past --steps); multiples of --grad-share-every 250",
    "--join-defect-masks /home/nvidia/r8ns/r9e2/launch_tree/stack/tanitad/configs/refcv8_join_label_defects_train4572.json  <- the mask file under the E2 launch tree "
    "(the gate binds it by content; operator: confirm this tree path)"]

#: K2 variants: name -> the E1 map arm whose banded flags the variant carries (None = C0's 8-code head, no banded head).
VARIANTS = {"K2": None, "K2M1N20": "M1N20", "K2M3": "M3", "K2M3N20": "M3N20"}
T1H_ARM = "T1h"
MAP_W = 1.0                      # --w-map-hires of every banded variant (E1's M1N20 fixed 1.0; M3 / M3N20 PREREG_M3: M1's 1.0) == K0's own value
PLAN_ANCHOR = "--bev-planner-crop-m"         # the C8 wires go right after the planner crop (planner-side flags, after K0's map group)
PLAN_MAP, PLAN_BOX = "--r9-plan-map-channels", "--r9-plan-box-tokens"
CRITIC, W_CRITIC = "--r8-critic-drivable", "--w-r8-drivable"
W_COST_TARGET = 0.1              # REFCV9_DESIGN 2.7 W-COST: weighted critic BCE ~ 0.1 x the trajectory loss at step 0
K0_FORBIDDEN_PREFIXES = ("--r9-",)
K0_FORBIDDEN_EXACT = (CRITIC, W_CRITIC, "--r8-critic-banded")


def pairs(argv):
    return E1.split_groups(list(argv))


def flat(ps):
    return [t for f, v in ps for t in [f, *v]]


def base_sha256(argv) -> str:
    """K0's ``base_sha256``: sha256 of ``json.dumps(base_argv, sort_keys=True)`` (MEASURED against the committed K0)."""
    return hashlib.sha256(json.dumps(list(argv), sort_keys=True).encode("utf-8")).hexdigest()


def dump(rec: dict) -> str:
    return json.dumps(rec, indent=1) + "\n"


def write(path, rec: dict) -> None:
    Path(path).write_text(dump(rec), encoding="utf-8", newline="\n")


def regen_k0(base_doc: dict) -> dict:
    """K0 from refcv8's launch argv doc: SET the five flags in place, APPEND the two schedule flags. Nothing else moves."""
    ps = pairs(base_doc["argv"])
    names = [f for f, _ in ps]
    for f, _ in K0_SET:
        if names.count(f) != 1:
            raise SystemExit("[r9-e2-argv] refcv8's argv must carry %s exactly once (has %d)" % (f, names.count(f)))
    for f, _ in K0_APPEND:
        if f in names:
            raise SystemExit("[r9-e2-argv] refcv8's argv already carries %s" % f)
    sets = dict(K0_SET)
    argv = flat([(f, sets.get(f, v)) for f, v in ps] + [(f, v) for f, v in K0_APPEND])
    return {"schema": "r9e2-argv-v1", "arm": "r9e2_K0", "seed": 0, "steps": E2_STEPS, "argv": argv, "base_sha256": base_sha256(base_doc["argv"]),
            "derived_from": base_doc["arm"], "what": K0_WHAT, "set_lines": list(K0_SET_LINES), "profile": "refcv9-e2"}


def w_from_ratio(ratio: float) -> float:
    """W-COST: ``--w-r8-drivable = 0.1 / R`` where R = critic-BCE / trajectory-loss at the step the trainer prints (``plan_wires_flag.wcost_ratio``)."""
    if not (ratio == ratio and 0.0 < ratio < float("inf")):
        raise SystemExit("[r9-e2-argv] the W-COST ratio %r must be finite and > 0" % (ratio,))
    return float("%.4g" % (W_COST_TARGET / ratio))


def stamp_w_cost(rec: dict, ratio: float | None, source: str | None) -> dict:
    """K2-WCOST: record where ``--w-r8-drivable`` came from, at the top level of the argv json. ``ratio`` given -> ``w_cost_ratio`` (R) + ``w_cost_source`` (the smoke log path the
    W-COST line was read from; REQUIRED, a measured weight with no artifact path is not quotable); not given -> ``w_cost: "provisional"``. Mutates and returns ``rec``."""
    if ratio is None:
        if source is not None:
            raise SystemExit("[r9-e2-argv] --w-cost-source needs --w-cost-ratio (a source for a weight that is not a measurement)")
        rec["w_cost"] = "provisional"
        return rec
    if not (ratio == ratio and 0.0 < ratio < float("inf")):
        raise SystemExit("[r9-e2-argv] the W-COST ratio %r must be finite and > 0" % (ratio,))
    if not (isinstance(source, str) and source.strip()):
        raise SystemExit("[r9-e2-argv] --w-cost-ratio needs --w-cost-source <the smoke log path the W-COST line was read from>")
    rec["w_cost_ratio"] = float(ratio)
    rec["w_cost_source"] = source
    return rec


def _check_k0(k0_doc: dict) -> list:
    if k0_doc.get("arm") != "r9e2_K0" or k0_doc.get("profile") != "refcv9-e2":
        raise SystemExit("[r9-e2-argv] --k0 must be r9e2_K0 (profile refcv9-e2), got %r / %r" % (k0_doc.get("arm"), k0_doc.get("profile")))
    argv = list(k0_doc["argv"])
    have = [f for f, _ in pairs(argv)]
    bad = [f for f in have if f.startswith(K0_FORBIDDEN_PREFIXES) or f in K0_FORBIDDEN_EXACT]
    if bad:
        raise SystemExit("[r9-e2-argv] K0 already carries lever flag(s) %s -- compose K2 from the lever-free K0" % bad)
    one = {"--out": None, "--agents": "head", "--bev-source": "map_hires_pool", "--map-hires": "on", "--w-box3d": None, PLAN_ANCHOR: None,
           "--w-map-hires": None, "--map-hires-near-lift-m": None, "--map-hires-near-refine-blocks": None}
    d = dict(pairs(argv))
    for f, want in one.items():
        if have.count(f) != 1:
            raise SystemExit("[r9-e2-argv] K0 must carry %s exactly once (has %d)" % (f, have.count(f)))
        if want is not None and d[f] != [want]:
            raise SystemExit("[r9-e2-argv] K0's %s is %s, the K2 wires need %r" % (f, d[f], want))
    if not float(d["--w-box3d"][0]) > 0.0:
        raise SystemExit("[r9-e2-argv] W-BOX needs --w-box3d > 0 (K0 has %s)" % d["--w-box3d"])
    if "--wp-index" in have and d["--wp-index"] == ["on"]:
        raise SystemExit("[r9-e2-argv] W-BOX is refused with --wp-index on")
    return argv


def compose(k0_doc: dict, variant: str, w_drivable: float, out: str | None = None) -> list:
    """K0's argv + the variant's levers. Returns the argv list. Order: K0's flags verbatim; after `--map-hires-near-refine-blocks` the banded flags then the T1h bundle
    (E1's own builders), after `--bev-planner-crop-m` the C8 wires: [W-MAP-2] W-BOX critic weight."""
    if variant not in VARIANTS:
        raise SystemExit("[r9-e2-argv] variant %r not in %s" % (variant, sorted(VARIANTS)))
    if not (w_drivable == w_drivable and 0.0 < float(w_drivable) < float("inf")):
        raise SystemExit("[r9-e2-argv] --w-r8-drivable %r must be finite and > 0 (the critic would be built and never supervised)" % (w_drivable,))
    argv = _check_k0(k0_doc)
    out = E1._thor_path(out or RUN_OUT % variant, "--out")
    arm = VARIANTS[variant]
    argv = E1.t1_arm_argv(argv, T1H_ARM, out)                       # the crowned T1h bundle, right after --map-hires-near-refine-blocks
    if arm is not None:
        w = MAP_W if arm in E1.OWN_ENC_ARMS else None
        argv = E1.map_arm_argv(argv, arm, out, w=w)                 # banded [+ own-enc] [+ near-m 20], near lift 10 / 20, --w-map-hires = the arm's
    wires = ([PLAN_MAP] if arm is not None else []) + [PLAN_BOX, CRITIC, W_CRITIC, repr(float(w_drivable))]
    res = []
    for f, vals in pairs(argv):
        res += [f] + vals
        if f == PLAN_ANCHOR:
            res += wires
    return res


def variant_doc(k0_doc: dict, k0_bytes: bytes, variant: str, w_drivable: float, *, provisional: bool, w_source: str, out: str | None = None) -> dict:
    argv = compose(k0_doc, variant, w_drivable, out)
    arm = VARIANTS[variant]
    lever = {"t1h_bundle": [t for t in E1.T1_ARMS[T1H_ARM]], "w_box": PLAN_BOX, "w_map_1": [CRITIC, W_CRITIC, repr(float(w_drivable))],
             "w_map_1_critic": "banded (reads the banded head's drv / drv_sdf / unobs)" if arm else "non-banded (the 8-code head's softmax sum)",
             "w_map_2": PLAN_MAP if arm else "NOT AVAILABLE without the banded head (plan_wires_flag.pin_args: needs --r9-map-banded)",
             "map_arm": arm, "map_config": "banded head" if arm else "C0's 8-code head",
             "w_r8_drivable_source": w_source,
             "w_r8_drivable_status": "PROVISIONAL -- NOT the argv a token may bind: regenerate with --w-cost-ratio R from the K2 explore smoke's W-COST line" if provisional
                                     else "MEASURED (0.1 / R, the explore smoke's W-COST ratio)"}
    if arm is not None:
        lever["banded_flags"] = [E1.BANDED_FLAG] + ([E1.OWN_ENC_FLAG] if arm in E1.OWN_ENC_ARMS else []) + ([E1.NEAR_M_FLAG, "20"] if arm in E1.NEAR20_ARMS else [])
        lever["map_near_lift_m"] = float(E1.banded_near_lift_m(arm))
        lever["w_map_hires"] = MAP_W
        lever["pending"] = ("BUILD-LEVEL checks PENDING: the trainer refuses --r9-bev-warp-supervise-hist with --r9-map-banded today (bev_warp_flag.py:178-179; another "
                            "agent implements it). The gate's STATIC admission does not read the trainer's pin.")
    return {"schema": "r9e2-argv-v1", "arm": "r9e2_%s" % variant, "seed": 0, "steps": E2_STEPS, "argv": argv,
            "base_sha256": hashlib.sha256(k0_bytes).hexdigest(), "derived_from": "r9e2_K0", "lever": lever,
            "what": ("%s = K0 + the crowned E1 lever bundle T1h + the C8 planner wires (REFCV9_DESIGN 4.4 K2; map config: %s). Composed by stack/scripts/r9_e2_make_argv.py from "
                     "stack/ops/runs.d/r9e2_K0.argv.json -- never hand-written; base_sha256 = sha256 of the K0 FILE bytes." % (variant, lever["map_config"])),
            "profile": "refcv9-e2"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS))
    ap.add_argument("--k0", required=True, help="stack/ops/runs.d/r9e2_K0.argv.json")
    ap.add_argument("--base", help="refcv8-r101-s0.argv.json (with --check-k0)")
    ap.add_argument("--check-k0", action="store_true", help="regenerate K0 from --base and require it BYTE-EQUAL to --k0")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--w-r8-drivable", type=float, help="W-MAP-1 critic BCE weight (W-COST: 0.1 / the step-0 critic-BCE / trajectory-loss ratio)")
    g.add_argument("--w-cost-ratio", type=float, help="the explore smoke's printed W-COST ratio R; the weight written is 0.1 / R (4 significant digits)")
    ap.add_argument("--w-cost-source", default=None, help="with --w-cost-ratio: the smoke log path the W-COST line was read from (stamped as w_cost_source)")
    ap.add_argument("--provisional", action="store_true", help="mark the weight PROVISIONAL (a smoke-only argv: no W-COST reading exists yet)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--write")
    a = ap.parse_args(argv)
    k0_bytes = Path(a.k0).read_bytes()
    k0_doc = json.loads(k0_bytes.decode("utf-8"))
    if a.check_k0:
        if not a.base:
            ap.error("--check-k0 needs --base")
        got = dump(regen_k0(json.loads(Path(a.base).read_text(encoding="utf-8")))).encode("utf-8")
        if got != k0_bytes:
            raise SystemExit("[r9-e2-argv] K0 does NOT regenerate byte-equal (%d vs %d bytes)" % (len(got), len(k0_bytes)))
        print("K0 regenerates BYTE-EQUAL (%d bytes, sha256 %s)" % (len(got), hashlib.sha256(got).hexdigest()))
        return 0
    if not a.variant or not a.write or (a.w_r8_drivable is None and a.w_cost_ratio is None):
        ap.error("--variant, --write and one of --w-r8-drivable / --w-cost-ratio are required")
    if a.w_cost_ratio is not None:
        w, src = w_from_ratio(a.w_cost_ratio), "0.1 / %g (--w-cost-ratio: the K2 explore smoke's W-COST line)" % a.w_cost_ratio
    else:
        w, src = float(a.w_r8_drivable), ("operator-supplied PROVISIONAL placeholder (no W-COST reading exists yet)" if a.provisional else "operator-supplied")
    rec = variant_doc(k0_doc, k0_bytes, a.variant, w, provisional=bool(a.provisional) and a.w_cost_ratio is None, w_source=src, out=a.out)
    write(a.write, stamp_w_cost(rec, a.w_cost_ratio, a.w_cost_source))
    print("wrote %s" % a.write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
