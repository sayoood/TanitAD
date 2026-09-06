#!/usr/bin/env python3
"""Build an anchor bank, OPTIONALLY extended with two-segment candidates.

⛔⛔ **PARITY IS THE POINT OF THIS TOOL.** Adding candidates changes the fan every
banked arm was scored against, so the extension is a FLAG and the flag DEFAULTS
OFF. Without ``--two-segment`` this writes the source bank's tensors unchanged —
``[N, 2]`` controls, ``control_schedule="constant"`` — and ``--assert-parity``
proves that by sha256 rather than asserting it.

MEASURED 2026-09-06 (`raw/v0_repro.py`): rolling refcv4b's own
``core.decoder.anchor_controls`` at ``ref_speed_ms = 10.0`` through
``anchor_twoseg.roll_bank`` reproduces its stored ``core.decoder.anchors``
sha256 **51f930dc…a66df** exactly, so this builder's integrator IS the
incumbent's, demonstrated rather than claimed.

WHY THE EXTENSION EXISTS
------------------------
All 117 incumbent candidates are constant-curvature arcs — **yaw is monotone on
1.000000 of 564,291 (window, candidate) pairs** — so none is an S-shape, while
**1,297 of 18,615** windows over **79 of 141** eval clips execute a lane-change
shape. A vocabulary is a CEILING; this adds the cheapest family that contains
one. ⭐ It needs **no label change** (PI, 2026-09-06: *"don't generate labels
again"*): the anchor classifier's target is a geometric ``argmin`` of the
recorded ego path over the emitted fan, not a tactical label.

USAGE
-----
    # parity build: byte-for-byte the incumbent, and prove it
    python scripts/build_twoseg_anchors.py \
        --from-checkpoint /path/ckpt_40284_FINAL.pt --out anchors_117.pt \
        --assert-anchors-sha 51f930dc… --assert-controls-sha b072f4c0…

    # the extension
    python scripts/build_twoseg_anchors.py \
        --from-checkpoint /path/ckpt_40284_FINAL.pt --out anchors_123.pt \
        --two-segment

⛔ CONSUMER STATUS. ``refc.py::AnchoredDiffusionDecoder.roll_bank`` rolls
``anchor_controls`` as a constant and its shape checks REFUSE a ``[N, 3]`` bank
(loudly, which is correct). ``tanitad.refs.anchor_twoseg.roll_bank`` is the
drop-in reference the decoder needs; wiring it is a named hand-off.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[1], "/workspace/TanitAD/stack"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tanitad.refs import anchor_meta as am           # noqa: E402
from tanitad.refs import anchor_twoseg as ts         # noqa: E402

DEFAULT_HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)
#: the speed sweep the Kamm report is run over: a standstill, the reference
#: speed, and 36 m/s (130 km/h) — the speed at which the units incident
#: manufactured 396 g.
KAMM_SPEEDS = (0.0, 4.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 36.0)


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()
    ).hexdigest()


def _assert_sha(name: str, got: str, want: str | None) -> None:
    """⛔ POSITIVE assertion. Two empty strings compare equal and that is how a
    mount outage reports MATCH; require 64 chars on BOTH sides first."""
    if want is None:
        return
    if len(got) != 64 or len(want) != 64:
        raise SystemExit(f"INCONCLUSIVE: {name} sha256 is not 64 chars "
                         f"(got {len(got)}, want {len(want)})")
    if got != want:
        raise SystemExit(f"MISMATCH: {name} sha256 {got} != {want}")
    print(f"[src] {name} sha256 VERIFIED {got}", flush=True)


def load_source(args) -> tuple[torch.Tensor, torch.Tensor]:
    """``(anchors [N, S, 2], controls [N, 2])`` from a checkpoint or a .pt bank."""
    if args.from_checkpoint:
        sd = torch.load(args.from_checkpoint, map_location="cpu",
                        weights_only=False)
        d = sd.get("model", sd)
        ka = [k for k in d if k.endswith("decoder.anchors")]
        kc = [k for k in d if k.endswith("anchor_controls")]
        if not ka or not kc:
            raise SystemExit(
                f"{args.from_checkpoint} has no decoder.anchors / "
                f"anchor_controls (found {len(ka)} / {len(kc)}); this is not a "
                f"v0-conditioned REF-C checkpoint")
        return d[ka[0]].float(), d[kc[0]].float()
    art = am.read_anchor_artifact(args.from_anchors,
                                  cli_control_units=args.control_units)
    if art.controls is None:
        raise SystemExit(f"{args.from_anchors} is a FIXED-PATH bank (no "
                         f"`controls`); a two-segment extension needs the "
                         f"control parameterisation, not the rolled paths")
    if art.control_schedule != am.CONSTANT_SCHEDULE:
        raise SystemExit(f"{args.from_anchors} already declares "
                         f"control_schedule={art.control_schedule!r}; refusing "
                         f"to extend an already-extended bank (the family "
                         f"would be appended twice)")
    return art.anchors.float(), art.controls.float()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-checkpoint", help="a REF-C checkpoint carrying "
                     "core.decoder.anchors + core.decoder.anchor_controls")
    src.add_argument("--from-anchors", help="an anchor artifact .pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--control-units", default="alat",
                    choices=list(am.CONTROL_UNITS))
    # ⛔⛔ THE FLAG. DEFAULT OFF. Every banked arm reproduces bit-identically
    # without it, and `--assert-parity` proves that rather than asserting it.
    ap.add_argument("--two-segment", action="store_true",
                    help="APPEND the two-segment family (default OFF: the "
                         "output is the source bank, tensors unchanged)")
    ap.add_argument("--a-lat", default=",".join(
        str(x) for x in ts.DEFAULT_A_LAT_MS2),
        help="lateral accelerations m/s^2 of the new family")
    ap.add_argument("--t-split", default=",".join(
        str(x) for x in ts.DEFAULT_T_SPLIT_S),
        help="split times s. ⚠️ a split at or beyond the horizon NEVER flips "
             "and reproduces a constant arc EXACTLY -- that is the deliberate-"
             "regression arm, not a build option")
    ap.add_argument("--a-lon", default=",".join(
        str(x) for x in ts.DEFAULT_A_LON_MS2),
        help="longitudinal accelerations m/s^2 of the new family")
    ap.add_argument("--horizons", default=",".join(
        str(h) for h in DEFAULT_HORIZONS))
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--ref-speed-ms", type=float, default=10.0)
    ap.add_argument("--kappa-cap", type=float, default=0.12)
    ap.add_argument("--alat-v-floor", type=float, default=4.0)
    ap.add_argument("--mu", type=float, default=0.7,
                    help="friction coefficient the new candidates are gated on")
    ap.add_argument("--assert-anchors-sha", default=None)
    ap.add_argument("--assert-controls-sha", default=None)
    ap.add_argument("--assert-parity", action="store_true",
                    help="without --two-segment, REFUSE unless the written "
                         "tensors are sha256-identical to the source")
    args = ap.parse_args(argv)

    hz = tuple(int(x) for x in args.horizons.split(","))
    slots = [h - 1 for h in hz]
    horizon_s = max(hz) * float(args.dt)
    steps = max(hz)

    A0, C0 = load_source(args)
    sha_a0, sha_c0 = _sha(A0), _sha(C0)
    print(f"[src] anchors {tuple(A0.shape)} controls {tuple(C0.shape)}")
    print(f"[src] anchors  sha256 {sha_a0}")
    print(f"[src] controls sha256 {sha_c0}")
    _assert_sha("source anchors", sha_a0, args.assert_anchors_sha)
    _assert_sha("source controls", sha_c0, args.assert_controls_sha)
    if C0.shape[1] != 2:
        raise SystemExit(f"source controls must be [N, 2]; got "
                         f"{tuple(C0.shape)}")
    n_base = int(C0.shape[0])

    if not args.two_segment:
        # ⛔ THE PARITY PATH: the tensors are the source's, untouched. Nothing is
        # re-rolled, because re-rolling would introduce a float difference the
        # flag is supposed to exclude by construction.
        controls, anchors = C0, A0
        schedule = am.CONSTANT_SCHEDULE
        kamm = None
        print("[ext] --two-segment ABSENT: the bank is the source, unchanged")
    else:
        a_lat = [float(x) for x in args.a_lat.split(",") if x != ""]
        t_spl = [float(x) for x in args.t_split.split(",") if x != ""]
        a_lon = [float(x) for x in args.a_lon.split(",") if x != ""]
        controls = ts.extend_controls(C0, horizon_s, a_lat=a_lat,
                                      t_split_s=t_spl, a_lon=a_lon)
        schedule = am.TWO_SEGMENT_SCHEDULE
        # ⚠️ `anchors` is the bank rolled at the REFERENCE speed -- the
        # checkpoint-visible artifact the priors fall back to. Rolling the WHOLE
        # extended bank (not just the new rows) keeps one integrator responsible
        # for the whole file; the first n_base rows are asserted unchanged below.
        anchors = ts.roll_bank(controls, torch.tensor([args.ref_speed_ms]),
                               control_units=args.control_units, steps=steps,
                               slots=slots, dt=args.dt,
                               alat_v_floor=args.alat_v_floor,
                               kappa_cap=args.kappa_cap)[0]
        if _sha(anchors[:n_base].contiguous()) != sha_a0:
            raise SystemExit(
                "REFUSED: re-rolling the extended bank did not reproduce the "
                "source anchors on the first %d rows. The extension must be "
                "APPEND-ONLY -- a checkpoint trained on the incumbent names "
                "candidates by integer index." % n_base)
        print(f"[ext] first {n_base} rolled anchors BIT-IDENTICAL to source")
        print(f"[ext] {ts.describe_family(controls, n_base)}")
        # ⛔ KINEMATIC ADMISSIBILITY -- of the NEW rows, at the REALISED a_lat
        kamm = ts.kamm_report(controls[n_base:], KAMM_SPEEDS,
                              control_units=args.control_units, mu=args.mu,
                              alat_v_floor=args.alat_v_floor,
                              kappa_cap=args.kappa_cap)
        base_kamm = ts.kamm_report(C0, KAMM_SPEEDS,
                                   control_units=args.control_units,
                                   mu=args.mu,
                                   alat_v_floor=args.alat_v_floor,
                                   kappa_cap=args.kappa_cap)
        kamm["incumbent_peak_total_g"] = base_kamm["peak_total_g"]
        kamm["incumbent_n_violations"] = base_kamm["n_violations"]
        print(f"[kamm] new candidates: peak {kamm['peak_total_g']:.4f} g, "
              f"{kamm['n_violations']} / {kamm['n_candidate_speed_pairs']} "
              f"pairs over mu={args.mu} "
              f"(incumbent peak {base_kamm['peak_total_g']:.4f} g, "
              f"{base_kamm['n_violations']} violations)")
        if kamm["kappa_cap_reached"]:
            raise SystemExit(
                "REFUSED: a new candidate's derived |kappa| reaches kappa_cap "
                "%g, so the candidate the file DECLARES is not the candidate "
                "the integrator drives." % args.kappa_cap)
        if kamm["n_violations"]:
            raise SystemExit(
                "REFUSED: %d new (candidate, speed) pairs exceed mu=%g."
                % (kamm["n_violations"], args.mu))

    extra = {"source_anchors_sha256": sha_a0,
             "source_controls_sha256": sha_c0,
             "source": (args.from_checkpoint or args.from_anchors),
             "n_base_candidates": n_base,
             "two_segment": bool(args.two_segment)}
    if kamm is not None:
        extra["kamm"] = kamm
    art = am.build_anchor_artifact(
        anchors, controls, control_units=args.control_units, horizons=hz,
        dt=args.dt, ref_speed_ms=args.ref_speed_ms, kappa_cap=args.kappa_cap,
        alat_v_floor=args.alat_v_floor, control_schedule=schedule,
        builder=__file__, extra=extra)

    if args.assert_parity:
        if args.two_segment:
            raise SystemExit("--assert-parity is meaningless with "
                             "--two-segment: the bank is deliberately different")
        if art["anchors_sha256"] != sha_a0 or art["controls_sha256"] != sha_c0:
            raise SystemExit("PARITY FAILED: the written tensors differ from "
                             "the source")
        print("[parity] written tensors BIT-IDENTICAL to the source: "
              f"anchors {art['anchors_sha256']} controls "
              f"{art['controls_sha256']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(art, out)

    # READ IT BACK -- units AND schedule must resolve from the FILE, no override
    ra = am.read_anchor_artifact(str(out))
    if ra.control_units_source != "file":
        raise SystemExit("REFUSED: units did not resolve from the file "
                         f"({ra.control_units_source})")
    if ra.control_schedule != schedule:
        raise SystemExit(f"REFUSED: schedule read back as "
                         f"{ra.control_schedule!r}, wrote {schedule!r}")
    print("[read] " + am.describe(ra))
    print(json.dumps({"saved": str(out),
                      "n_anchors": int(controls.shape[0]),
                      "controls_shape": list(controls.shape),
                      "control_schedule": schedule,
                      "anchors_sha256": art["anchors_sha256"],
                      "controls_sha256": art["controls_sha256"],
                      "bytes": os.path.getsize(out)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
