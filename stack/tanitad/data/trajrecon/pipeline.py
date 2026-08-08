"""Production pipeline: a folder of Sensor Logger zips -> validated training data.

    python pipeline.py --input-dir  /path/to/zips \
                       --output-dir /path/to/dataset

Design rules
------------
**Never process blind.**  Every recording is diagnosed before any solving
happens.  A recording whose GNSS is a frozen cached fix, whose video cannot be
aligned to the sensors, or whose camera geometry is impossible is *rejected*
with a written analysis -- not silently turned into confident, fictional ground
truth.  That failure mode already exists in this data set and it is the one that
poisons a training set without anyone noticing.

**Resumable and idempotent.**  A registry records the state and input
fingerprint of every zip.  Re-running skips completed work, retries failures,
and reprocesses anything whose input changed.  Interrupting mid-run costs at
most one recording.

**Work locally, publish once.**  Everything happens in a local scratch
directory and is copied to the output at the end.  The source folder here is a
Google Drive mount that has already dropped out mid-run once; streaming 2247
JPEGs off it repeatedly is also 6x slower than local disk.

Per-recording output
--------------------
``report.json`` / ``report.md``   diagnosis, metrics, calibration, timings
``diagnosis.log``                 the pre-flight analysis, kept even on rejection
``sensors/``                      synchronised sensor streams, session level
``frames/``                       extracted frame images
``trajectory.jsonl``              one record per frame: past+future ego trajectory
``validation.png``                a single composite frame: projection + BEV + HUD
``overlay.mp4``                   the full session as video
``calibration.json``              intrinsics, mount, and how each was obtained
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trajlib import diagnose as DG
from trajlib import timesync as TS
from trajlib import viz
from trajlib.camera import calibrate_camera
from trajlib.io_sensorlogger import load_session
from trajlib.steering import AUDI_A6_ETRON, estimate_steering
from trajlib.trajectory import (ego_trajectory, estimate_trajectory,
                                to_vehicle_reference)
from trajlib.validate import validate
from trajlib.vehicle_frame import estimate_vehicle_frame

PENDING, RUNNING, DONE, REJECTED, FAILED = "PENDING", "RUNNING", "DONE", "REJECTED", "FAILED"

# Bump whenever a change could alter a recording's verdict or its outputs.  The
# registry stores it per recording, so a fixed bug automatically re-opens the
# archives it previously mis-handled -- without that, a rejection is permanent
# even after the code that caused it has been corrected.
PIPELINE_VERSION = "1.6.1-lateral"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
class Log:
    """Tee to stdout and to one or two files, with timestamps."""

    def __init__(self, *paths):
        self.files = []
        for p in paths:
            if p:
                os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
                self.files.append(open(p, "a", encoding="utf-8"))

    def __call__(self, msg="", level="INFO"):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"{ts} {level:5s} {msg}"
        print(line, flush=True)
        for f in self.files:
            f.write(line + "\n")
            f.flush()

    def block(self, title, body, level="INFO"):
        self(f"--- {title} ---", level)
        for ln in str(body).splitlines():
            self("    " + ln, level)

    def close(self):
        for f in self.files:
            f.close()


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def fingerprint(path):
    st = os.stat(path)
    return f"{st.st_size}:{int(st.st_mtime)}"


class Registry:
    """Durable per-zip state so a run can be resumed or retried."""

    def __init__(self, path):
        self.path = path
        self.data = {}
        if os.path.exists(path):
            try:
                self.data = json.load(open(path, encoding="utf-8"))
            except Exception:
                self.data = {}

    def get(self, name):
        return self.data.get(name, {})

    def set(self, name, **kw):
        self.data.setdefault(name, {}).update(kw)
        self.save()

    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1)
        os.replace(tmp, self.path)

    def should_process(self, name, fp, retry_failed=True, version=PIPELINE_VERSION):
        e = self.get(name)
        if not e:
            return True, "new"
        if e.get("fingerprint") != fp:
            return True, "input changed"
        if e.get("pipeline_version") != version:
            return True, f"pipeline {e.get('pipeline_version', 'unversioned')} -> {version}"
        if e.get("status") == DONE:
            return False, "already done"
        if e.get("status") == REJECTED:
            return False, f"rejected previously ({e.get('verdict_reason', '')})"
        if e.get("status") in (FAILED, RUNNING):
            return (retry_failed, "retrying" if retry_failed else "previously failed")
        return True, "pending"

    def summary(self):
        c = {}
        for v in self.data.values():
            c[v.get("status", "?")] = c.get(v.get("status", "?"), 0) + 1
        return c


# --------------------------------------------------------------------------- #
def find_export_root(extract_dir):
    """Sensor Logger zips sometimes nest the export one level down."""
    if os.path.exists(os.path.join(extract_dir, "Metadata.csv")):
        return extract_dir
    for root, dirs, files in os.walk(extract_dir):
        if "Metadata.csv" in files:
            return root
    return None


def write_reports(out_dir, report, diag_text):
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    with open(os.path.join(out_dir, "diagnosis.log"), "w", encoding="utf-8") as f:
        f.write(diag_text)

    d = report.get("diagnosis", {})
    lines = [f"# {report.get('recording', '?')}", "",
             f"**Status:** {report.get('status')}  ",
             f"**Verdict:** {d.get('verdict')}  ",
             f"**Processed:** {report.get('finished_utc', '-')}  ",
             f"**Duration:** {report.get('duration_s', 0):.0f} s", ""]
    fnd = d.get("findings", [])
    if fnd:
        lines += ["## Findings", "", "| level | code | message |", "|---|---|---|"]
        for x in fnd:
            lines.append(f"| {x['level']} | `{x['code']}` | {x['message']} |")
        lines.append("")
    if d.get("stats"):
        lines += ["## Measurements", "", "| quantity | value |", "|---|---|"]
        for k, v in d["stats"].items():
            lines.append(f"| {k} | {v} |")
        lines.append("")
    if report.get("outputs"):
        lines += ["## Outputs", ""]
        for k, v in report["outputs"].items():
            lines.append(f"- `{k}` — {v}")
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
def process_one(zip_path, out_dir, scratch, args, log) -> dict:
    """Run one recording end to end.  Returns the report dict."""
    t_start = time.time()
    name = os.path.splitext(os.path.basename(zip_path))[0]
    # self_calibrate writes the measured lateral offset back into args so the
    # renderers pick it up.  args outlives the recording, so without this reset
    # the next archive silently inherits the previous one's value -- Rose Ave
    # was emitted with the Pacific Coast Highway offset of -0.20 m labelled
    # "operator prior".  A stale parameter presented as this recording's is
    # precisely the failure this pipeline exists to prevent.
    args.lateral_offset = args.lateral_offset_prior
    report = dict(recording=name, zip=os.path.basename(zip_path), status=RUNNING,
                  started_utc=datetime.now(timezone.utc).isoformat(), outputs={})
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------------------------------- unpack
    work = os.path.join(scratch, name)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    log(f"unpacking {os.path.basename(zip_path)} "
        f"({os.path.getsize(zip_path) / 1e6:.0f} MB) -> local scratch")
    try:
        with zipfile.ZipFile(zip_path) as z:
            bad = z.testzip()
            if bad is not None:
                raise zipfile.BadZipFile(f"corrupt member: {bad}")
            z.extractall(work)
    except Exception as e:
        d = DG.Diagnosis().add(DG.REJECT, "CORRUPT_ARCHIVE", f"cannot unpack: {e}")
        report.update(status=REJECTED, diagnosis=d.to_dict(),
                      finished_utc=datetime.now(timezone.utc).isoformat(),
                      duration_s=time.time() - t_start)
        write_reports(out_dir, report, str(d))
        log(str(d), "ERROR")
        return report

    root = find_export_root(work)
    if root is None:
        d = DG.Diagnosis().add(DG.REJECT, "NOT_AN_EXPORT",
                               "no Metadata.csv anywhere in the archive")
        report.update(status=REJECTED, diagnosis=d.to_dict(),
                      finished_utc=datetime.now(timezone.utc).isoformat(),
                      duration_s=time.time() - t_start)
        write_reports(out_dir, report, str(d))
        log(str(d), "ERROR")
        return report

    # ------------------------------------------------------------ diagnose 1
    d_struct = DG.diagnose_export(root)
    log.block("structure", d_struct)
    if not d_struct.ok:
        report.update(status=REJECTED, diagnosis=d_struct.to_dict(),
                      finished_utc=datetime.now(timezone.utc).isoformat(),
                      duration_s=time.time() - t_start)
        write_reports(out_dir, report, str(d_struct))
        log("REJECTED at the structural check; nothing was processed", "ERROR")
        return report

    session = load_session(root)
    log(f"session {session.device}  {session.duration:.1f}s  "
        f"streams={list(session.sensors)}")
    video = TS.probe_video(session.video_path)

    d_sess = DG.diagnose_session(session, video)
    log.block("content", d_sess)
    diag = DG.merge(d_struct, d_sess)
    if not diag.ok:
        report.update(status=REJECTED, diagnosis=diag.to_dict(),
                      finished_utc=datetime.now(timezone.utc).isoformat(),
                      duration_s=time.time() - t_start)
        write_reports(out_dir, report, str(diag))
        log("REJECTED at the content check; nothing was processed", "ERROR")
        return report

    # ---------------------------------------------------------------- sync
    log("synchronising video to sensors (optical flow vs gyro)...")
    flow = TS.image_angular_rate(video, t_dur=min(args.calib_seconds, video.duration))
    sync = TS.synchronise(session, video, cam_flow=flow)
    d_sync = DG.diagnose_sync(sync)
    log.block("sync", d_sync)
    diag = DG.merge(diag, d_sync)
    if not diag.ok:
        report.update(status=REJECTED, diagnosis=diag.to_dict(),
                      finished_utc=datetime.now(timezone.utc).isoformat(),
                      duration_s=time.time() - t_start)
        write_reports(out_dir, report, str(diag))
        log("REJECTED: synchronisation could not be verified", "ERROR")
        return report

    # ---------------------------------------------------- frame & trajectory
    vframe = estimate_vehicle_frame(session)
    log(f"vehicle frame: {vframe.fwd_sign_source}, lateral_score={vframe.lateral_score:.3f}")
    traj = estimate_trajectory(session, vframe)
    val = validate(session, vframe, traj, k=5)
    log.block("validation", val)
    d_traj = DG.diagnose_trajectory(traj, val)
    log.block("trajectory", d_traj)
    diag = DG.merge(diag, d_traj)
    if not diag.ok:
        report.update(status=REJECTED, diagnosis=diag.to_dict(),
                      finished_utc=datetime.now(timezone.utc).isoformat(),
                      duration_s=time.time() - t_start)
        write_reports(out_dir, report, str(diag))
        log("REJECTED: the solved trajectory failed its checks", "ERROR")
        return report

    steer = estimate_steering(traj, {**AUDI_A6_ETRON, "wheelbase_m": args.wheelbase,
                                     "steering_ratio": args.steering_ratio},
                              max_wheel_rate_deg_s=args.max_wheel_rate)
    log.block("steering", steer.summary())

    # ------------------------------------------------------------- camera
    cam = calibrate_camera(video, session, sync, traj=traj, cam_flow=flow,
                           vframe=vframe, height_m=args.cam_height)
    cam, calib = self_calibrate(video, cam, traj, sync, args, log,
                                session=session, vframe=vframe)
    # The trajectory is about to be re-referenced to the vehicle, so the camera
    # is no longer at the origin: it sits at the mount position.  project()
    # already subtracts t_v, it simply was never told.
    cam.longitudinal_m = float(args.mount_longitudinal)
    cam.lateral_m = float(args.lateral_offset)
    d_cam = DG.diagnose_camera(cam)
    log.block("camera", d_cam)
    diag = DG.merge(diag, d_cam)
    if not diag.ok:
        report.update(status=REJECTED, diagnosis=diag.to_dict(),
                      finished_utc=datetime.now(timezone.utc).isoformat(),
                      duration_s=time.time() - t_start)
        write_reports(out_dir, report, str(diag))
        log("REJECTED: camera geometry is not usable", "ERROR")
        return report

    # ------------------------------------------------------------- outputs
    outs = emit_outputs(out_dir, session, video, sync, traj, steer, cam, args, log,
                        calib=calib)
    report["outputs"] = outs
    report["calibration_set"] = calib

    report.update(status=DONE, diagnosis=diag.to_dict(),
                  calibration=dict(fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy,
                                   yaw_deg=float(np.rad2deg(cam.yaw)),
                                   pitch_deg=float(np.rad2deg(cam.pitch)),
                                   roll_deg=float(np.rad2deg(cam.roll)),
                                   height_m=cam.height_m,
                                   lateral_offset_m=0.0,
                                   sources=cam.source),
                  sync=dict(t_video_start_s=sync.t_video_start, source=sync.source),
                  finished_utc=datetime.now(timezone.utc).isoformat(),
                  duration_s=time.time() - t_start)
    write_reports(out_dir, report, str(diag))
    shutil.rmtree(work, ignore_errors=True)
    return report


def self_calibrate(video, cam, traj, sync, args, log, session=None, vframe=None):
    """Run the full self-calibration for **this recording** and record its provenance.

    Every recording gets its own parameter set.  The mount is not assumed to be
    the same as last time: a phone gets re-seated, a cradle slips, and a stale
    calibration is indistinguishable from a correct one until the projection is
    already wrong.

    Each parameter is taken from the method that is well posed for it, which is
    not the same method for all of them:

    ==========  ==========================  ==============================================
    parameter   source                      why that one
    ==========  ==========================  ==============================================
    focal       gyro vs image yaw rate      absolute scale from a trusted rate sensor
    yaw         lane vanishing point        parallel-to-travel by construction, and
                                            invariant to feature height; the FOE is
                                            fitted over curves too and drifts sideways
    pitch       focus of expansion          direction-based; immune to flow-magnitude bias
    height      road-plane homography       t/d with a metric baseline; the FOE cannot
    roll        vertical vanishing point    the only method with lateral leverage
    lateral     ego-lane centre             unobservable from motion, but lane geometry
                                            supplies the missing external reference
    ==========  ==========================  ==============================================

    Yaw moved off the FOE after the Pacific Coast Highway recording showed the two
    disagreeing by 0.98 deg -- 0.70 m of lateral error at a 40 m look-ahead, and
    visible as a ribbon drifting off the lane toward the horizon.  The FOE is kept
    as the fallback and the disagreement is recorded.

    Where two methods overlap they are compared rather than averaged, and the
    disagreement is recorded: a cross-check that is never looked at is not a
    cross-check.  Returns ``(cam, calibration_record)``.
    """
    rec = {"per_recording": True, "parameters": {}, "cross_checks": {}}

    def put(name, value, unit, source, ci=None, note=None):
        rec["parameters"][name] = {k: v for k, v in
                                   dict(value=value, unit=unit, source=source,
                                        ci95=ci, note=note).items() if v is not None}

    put("focal_px", round(cam.fx, 1), "px", cam.source.get("intrinsics", "nominal"))
    put("hfov_deg", round(cam.hfov_deg(), 2), "deg", "derived from focal")
    put("yaw_deg", round(float(np.rad2deg(cam.yaw)), 3), "deg",
        cam.source.get("extrinsics", "nominal"))
    foe_pitch = float(np.rad2deg(cam.pitch))
    put("pitch_deg", round(foe_pitch, 3), "deg", cam.source.get("extrinsics", "nominal"))
    if "foe_vs_hough_vp" in cam.source:
        rec["cross_checks"]["foe_vs_hough_vp"] = cam.source["foe_vs_hough_vp"]

    # -- height (and an independent pitch) from the road plane -------------- #
    if args.plane_calib:
        try:
            from trajlib import ground_calib as GC
            from trajlib import plane_calib as PC
            rows = GC.collect_road_tracks(video, cam, traj, sync, gap_frames=4, max_pairs=250)
            res = PC.estimate_from_tracks(rows, cam) if rows else None
            if res is None:
                log("plane calibration produced too few usable homographies", "WARN")
                rec["cross_checks"]["plane"] = "insufficient homographies"
            else:
                log.block("road-plane homography", res.summary())
                dp = abs(res.pitch_deg - foe_pitch)
                log(f"    pitch cross-check vs FOE {foe_pitch:+.2f} deg: {dp:.2f} deg apart")
                rec["cross_checks"]["pitch_foe_vs_plane_deg"] = round(dp, 3)
                rec["cross_checks"]["plane_pairs"] = res.n_pairs
                # The decomposition returns a height whether or not it is
                # trustworthy, so the cross-check has to be acted on rather than
                # merely recorded.  Rose Ave produced 1.621 m +/-0.589 from 21
                # pairs with a pitch 9.7 deg away from the FOE -- a number that
                # is wrong in a way only the cross-check reveals.
                why = []
                if res.n_pairs < 40:
                    why.append(f"only {res.n_pairs} homographies")
                if dp > 2.0:
                    why.append(f"pitch disagrees with the FOE by {dp:.1f} deg")
                if res.height_mad_m > 0.15:
                    why.append(f"height spread +/-{res.height_mad_m:.2f} m")
                if not (0.8 <= res.height_m <= 1.8):
                    why.append(f"height {res.height_m:.2f} m is not a windscreen mount")
                if why:
                    log(f"plane height rejected: {'; '.join(why)}", "WARN")
                    rec["cross_checks"]["plane_height_rejected"] = "; ".join(why)
                else:
                    cam = PC.apply(cam, res, set_height=True, set_roll=False, set_pitch=False)
                    put("height_m", round(cam.height_m, 3), "m",
                        f"road-plane homography ({res.n_pairs} pairs)",
                        note=f"spread +/-{res.height_mad_m:.3f} m")
        except Exception as e:
            log(f"plane calibration skipped: {e}", "WARN")
    if "height_m" not in rec["parameters"]:
        put("height_m", round(cam.height_m, 3), "m", "operator default (not measured)")

    # -- roll from vertical structure --------------------------------------- #
    if args.vp_calib:
        try:
            from trajlib import vp_calib as VP
            idx = np.linspace(0, len(video.pts) - 1, args.vp_frames).astype(int)
            res = VP.estimate(video, cam, idx)
            if res is None:
                log("vertical-VP calibration found too few vertical lines", "WARN")
                rec["cross_checks"]["vertical_vp"] = "insufficient vertical lines"
            else:
                log.block("vertical vanishing point", res.summary())
                cam = VP.apply(cam, res, set_roll=True, set_pitch=False)
                put("roll_deg", round(float(np.rad2deg(cam.roll)), 3), "deg",
                    f"vertical vanishing point ({res.n_frames} frames)",
                    ci=[round(res.roll_ci_deg[0], 2), round(res.roll_ci_deg[1], 2)],
                    note="CI spanning zero means roll is not distinguishable from level")
                rec["cross_checks"]["vp_pitch_deg"] = round(res.pitch_deg, 2)
                rec["cross_checks"]["vp_pitch_note"] = (
                    "ill-conditioned for a level camera (depends on the tiny convergence "
                    "of near-parallel lines); recorded, not used")
        except Exception as e:
            log(f"vertical-VP calibration skipped: {e}", "WARN")
    if "roll_deg" not in rec["parameters"]:
        put("roll_deg", round(float(np.rad2deg(cam.roll)), 3), "deg", "assumed level (not measured)")

    # -- yaw and lateral offset from lane markings -------------------------- #
    # Last, because it consumes the pitch and height settled above.
    _lane_width_measured = None
    if args.lane_calib and session is not None and vframe is not None:
        try:
            from trajlib import lane_calib as LC
            foe_yaw = float(np.rad2deg(cam.yaw))
            res = LC.estimate(session, video, cam, vframe,
                              t_video_start_s=sync.t_video_start)
            log.block("lane markings", str(res))
            for n in res.notes:
                log(f"    {n}", "WARN")
            rec["cross_checks"]["lane_frames"] = res.n_frames
            rec["cross_checks"]["lane_segments"] = res.n_segments
            if np.isfinite(res.pitch_check_px):
                rec["cross_checks"]["pitch_foe_vs_lane_vp_px"] = round(res.pitch_check_px, 1)
                log(f"    pitch cross-check: lane VP row is {res.pitch_check_px:.1f} px "
                    f"from the FOE row")
            if res.yaw_deg is not None:
                d = res.yaw_deg - foe_yaw
                rec["cross_checks"]["yaw_foe_vs_lane_deg"] = round(d, 3)
                log(f"    yaw: FOE {foe_yaw:+.2f} deg -> lane VP {res.yaw_deg:+.2f} deg "
                    f"({d:+.2f} deg, {abs(d) * np.pi / 180 * 40:.2f} m at 40 m)")
            _lane_width_measured = (float(res.lane_width_m)
                                    if np.isfinite(res.lane_width_m) else None)
            prov = LC.apply(cam, res)
            for k, v in prov.items():
                rec["parameters"][k] = v
            if res.lateral_offset_m is not None:
                args.lateral_offset = res.lateral_offset_m
        except Exception as e:
            log(f"lane calibration skipped: {e}", "WARN")

    # -- separate focal from height using f*h and the lane width ------------ #
    # The ground plane sees h laterally and f*h longitudinally, so no amount of
    # ground self-consistency separates them -- which is how a 33% focal error
    # survived every check on the Pacific Coast Highway recording.
    lane_w_measured = _lane_width_measured
    if args.scale_calib:
        try:
            from trajlib import scale_calib as SC
            ft = sync.frame_times(video.pts)
            spd = np.interp(ft, traj.t, traj.speed)
            vals, note = SC.estimate_fh(video, np.arange(len(ft)), spd, ft,
                                        cam.horizon_v(), video.width, video.height)
            if note:
                log(f"f*h not measured: {note}", "WARN")
            res = SC.solve(vals, lane_w_measured, cam.height_m,
                           args.lane_width, cam.cx)
            log.block("scale (f vs h)", str(res))
            for n in res.notes:
                log(f"    {n}", "WARN")
            if np.isfinite(res.fh):
                rec["cross_checks"]["fh_px_m"] = round(res.fh, 1)
                rec["cross_checks"]["fh_tracks"] = res.fh_n
                rec["cross_checks"]["fh_spread"] = round(res.fh_spread, 3)
            if res.focal_px is not None:
                gyro_f = cam.fx
                rec["cross_checks"]["focal_gyro_vs_scale_px"] = round(res.focal_px - gyro_f, 1)
                log(f"    focal: gyro {gyro_f:.0f} px -> f*h/h {res.focal_px:.0f} px "
                    f"({(res.focal_px / gyro_f - 1) * 100:+.0f}%)")
                # yaw and pitch were measured as *pixel* locations (the lane VP
                # column, the FOE row) and only converted to angles through f.
                # Changing f must preserve those pixels, not the angles -- else
                # the horizon walks off the FOE row that produced it.
                _rescale = gyro_f / float(res.focal_px)
                cam.yaw = float(np.arctan(np.tan(cam.yaw) * _rescale))
                cam.pitch = float(np.arctan(np.tan(cam.pitch) * _rescale))
                cam.fx = cam.fy = float(res.focal_px)
                cam.height_m = float(res.height_m)
                put("yaw_deg", round(float(np.rad2deg(cam.yaw)), 3), "deg",
                    rec["parameters"].get("yaw_deg", {}).get("source", "?")
                    + " (angle rescaled to the corrected focal)")
                put("pitch_deg", round(float(np.rad2deg(cam.pitch)), 3), "deg",
                    rec["parameters"].get("pitch_deg", {}).get("source", "?")
                    + " (angle rescaled to the corrected focal)")
                put("focal_px", round(cam.fx, 1), "px",
                    f"f*h from {res.fh_n} road tracks / height from lane width "
                    f"(assumed {res.lane_width_used_m:.2f} m)")
                put("hfov_deg", round(cam.hfov_deg(), 2), "deg", "derived from focal")
                put("height_m", round(cam.height_m, 3), "m",
                    f"f*h / focal, lane width assumed {res.lane_width_used_m:.2f} m")
        except Exception as e:
            log(f"scale calibration skipped: {e}", "WARN")

    # Non-circular sanity gate: every other focal check compares f against
    # something derived using f.  A physical field of view does not.
    try:
        from trajlib.scale_calib import check_focal_plausible
        ok_f, hfov = check_focal_plausible(cam.fx, cam.cx)
        rec["cross_checks"]["hfov_deg"] = round(hfov, 2)
        if not ok_f:
            nominal = cam.cx / np.tan(np.deg2rad(68.0) / 2.0)
            log(f"focal {cam.fx:.0f} px implies {hfov:.1f} deg HFOV, impossible for a "
                f"phone camera -- falling back to a nominal 68 deg ({nominal:.0f} px)", "WARN")
            rec["cross_checks"]["focal_rejected"] = f"{hfov:.1f} deg HFOV out of band"
            _r = cam.fx / float(nominal)
            cam.yaw = float(np.arctan(np.tan(cam.yaw) * _r))
            cam.pitch = float(np.arctan(np.tan(cam.pitch) * _r))
            cam.fx = cam.fy = float(nominal)
            put("yaw_deg", round(float(np.rad2deg(cam.yaw)), 3), "deg",
                rec["parameters"].get("yaw_deg", {}).get("source", "?")
                + " (angle rescaled to the nominal focal)")
            put("pitch_deg", round(float(np.rad2deg(cam.pitch)), 3), "deg",
                rec["parameters"].get("pitch_deg", {}).get("source", "?")
                + " (angle rescaled to the nominal focal)")
            put("focal_px", round(cam.fx, 1), "px",
                "device-nominal 68 deg HFOV (measured focal rejected as impossible)")
            put("hfov_deg", round(cam.hfov_deg(), 2), "deg", "derived from focal")
    except Exception as e:
        log(f"focal plausibility check skipped: {e}", "WARN")

    if "yaw_deg" not in rec["parameters"] or "lane" not in rec["parameters"]["yaw_deg"].get("source", ""):
        put("yaw_deg", round(float(np.rad2deg(cam.yaw)), 3), "deg",
            cam.source.get("extrinsics", "nominal"))
    if "lateral_offset_m" not in rec["parameters"]:
        put("lateral_offset_m", args.lateral_offset, "m", "operator prior",
            note="+left. Unobservable from motion -- a sideways camera shift moves every "
                 "ground point identically at both ends of a track and cancels out of any "
                 "feature-tracking residual -- so it falls back to the operator prior when "
                 "lane markings are unusable.")
    put("mount_longitudinal_m", args.mount_longitudinal, "m", "operator prior",
        note="+forward. Phone position ahead of the vehicle reference point (rear "
             "axle). The exported trajectory is the rear axle's path, not the "
             "phone's: a point mounted this far forward swings wide through every "
             "turn, and ignoring it left up to 2.6 m of lateral error at "
             "intersections while straight-road checks still showed centimetres.")
    put("trajectory_reference", "rear axle", "-",
        f"phone re-referenced by (L={args.mount_longitudinal:.2f}, "
        f"W={args.lateral_offset:+.2f}) m via a per-point rigid transform")
    put("vehicle_width_m", args.vehicle_width, "m", "operator setting")
    return cam, rec


# --------------------------------------------------------------------------- #
def _ego_at(traj, t_ref, args):
    """Ego window at ``t_ref``, re-referenced from the phone to the vehicle.

    GNSS locates the phone; the vehicle body follows a different curve through
    every turn.  Doing the transform here means the JSONL export, the validation
    still and the video all get the same, correct path.
    """
    ego = ego_trajectory(traj, float(t_ref), t_past=args.t_past,
                         standstill_speed=args.standstill_speed,
                         t_future_standstill=args.t_future_standstill,
                         t_future=args.t_future, dt_out=args.dt_out)
    if ego is None:
        return None
    return to_vehicle_reference(ego, args.mount_longitudinal, args.lateral_offset)


def emit_outputs(out_dir, session, video, sync, traj, steer, cam, args, log, calib=None):
    """Write sensors, frames, trajectories, the validation still and the video."""
    import cv2

    outs = {}
    ft = sync.frame_times(video.pts)
    usable = np.nonzero((ft >= traj.t[0] + args.t_past) &
                        (ft <= traj.t[-1] - args.t_future))[0]
    log(f"{len(usable)} of {len(ft)} frames have a complete "
        f"[-{args.t_past:.0f}, +{args.t_future:.0f}] s window")

    # -- sensors, session level -------------------------------------------- #
    sdir = os.path.join(out_dir, "sensors")
    os.makedirs(sdir, exist_ok=True)
    for key, df in session.sensors.items():
        df.to_csv(os.path.join(sdir, f"{key}.csv"), index=False)
    meta = dict(device=session.device, epoch_ms=session.epoch_ms,
                timezone=session.timezone_name, duration_s=session.duration,
                t_video_start_s=sync.t_video_start,
                note="One file per stream on the session timebase (seconds_elapsed). "
                     "Per-frame sensor windows are deliberately NOT duplicated: the "
                     "original pipeline stored each sample ~150 times, which is what "
                     "made a 78 s recording into a 2 GB archive.")
    json.dump(meta, open(os.path.join(sdir, "meta.json"), "w"), indent=1)
    outs["sensors/"] = f"{len(session.sensors)} streams on a common timebase"

    # -- per-frame trajectory ---------------------------------------------- #
    tj = os.path.join(out_dir, "trajectory.jsonl")
    n_written = 0
    with open(tj, "w", encoding="utf-8") as f:
        for fi in usable:
            t_ref = float(ft[fi])
            ego = _ego_at(traj, t_ref, args)
            if ego is None:
                continue
            sw = float(np.interp(t_ref, steer.t, steer.wheel_deg))
            sv = bool(np.interp(t_ref, steer.t, steer.valid.astype(float)) > 0.5)
            f.write(json.dumps(dict(
                frame=int(fi), pts_s=round(float(video.pts[fi]), 6),
                t_session_s=round(t_ref, 6),
                t_utc=round(float(session.elapsed_to_utc(t_ref)), 3),
                speed_ms=round(float(ego["speed_ref"]), 4),
                steer_wheel_deg=round(sw, 2), steer_valid=sv,
                standstill=bool(ego.get("standstill", False)),
                t_future_s=round(float(ego.get("t_future_s", args.t_future)), 2),
                complete=bool(ego["complete"]),
                t=[round(float(x), 2) for x in ego["t"]],
                x=[round(float(x), 4) for x in ego["x"]],
                y=[round(float(x), 4) for x in ego["y"]],
                yaw=[round(float(x), 5) for x in ego["yaw"]],
                v=[round(float(x), 4) for x in ego["speed"]],
                pos_sigma=[round(float(x), 3) for x in ego["pos_std"]])) + "\n")
            n_written += 1
    outs["trajectory.jsonl"] = f"{n_written} frames, ego frame FLU, past+future"

    # -- frames -------------------------------------------------------------- #
    if args.frame_stride > 0:
        fdir = os.path.join(out_dir, "frames")
        os.makedirs(fdir, exist_ok=True)
        sel = usable[::args.frame_stride]
        log(f"extracting {len(sel)} frames (stride {args.frame_stride})...")
        _extract_frames(video, sel, fdir, args.frame_quality, args.frame_width)
        outs["frames/"] = f"{len(sel)} JPEGs (stride {args.frame_stride})"

    # -- validation still ---------------------------------------------------- #
    mid = usable[len(usable) // 2]
    ego = _ego_at(traj, float(ft[mid]), args)
    img = _grab_bgr(video, int(mid))
    if img is not None and ego is not None:
        panel = _compose(img, ego, cam, steer, float(ft[mid]), args)
        cv2.imwrite(os.path.join(out_dir, "validation.png"), panel)
        outs["validation.png"] = f"composite at t={ft[mid]:.1f}s"

    # -- video ---------------------------------------------------------------- #
    if not args.no_video:
        vp = os.path.join(out_dir, "overlay.mp4")
        n = _render_video(video, usable[::args.video_stride], ft, traj, steer, cam,
                          args, vp, log)
        if n:
            outs["overlay.mp4"] = f"{n} frames, projection + BEV + steering"

    cal = dict(recording=os.path.basename(out_dir),
               image=dict(width=cam.width, height=cam.height),
               K=dict(fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy),
               extrinsics=dict(yaw_deg=float(np.rad2deg(cam.yaw)),
                               pitch_deg=float(np.rad2deg(cam.pitch)),
                               roll_deg=float(np.rad2deg(cam.roll)),
                               height_m=cam.height_m,
                               # Read from the camera, not hardcoded: cam now carries
                               # the mount offsets in t_v and the projection uses them.
                               # This block used to say 0.0 because viz applied the
                               # lateral shift separately -- stale since 1.6.0, and it
                               # silently handed every downstream consumer the wrong
                               # mounting position.
                               lateral_offset_m=float(cam.lateral_m),
                               longitudinal_m=float(cam.longitudinal_m)),
               horizon_row_px=cam.horizon_v(),
               raw_sources=cam.source)
    if calib:
        cal.update(calib)
    json.dump(cal, open(os.path.join(out_dir, "calibration.json"), "w"), indent=1)
    outs["calibration.json"] = "per-recording parameter set with provenance per parameter"
    return outs


def _grab_bgr(video, index):
    import cv2
    if hasattr(video, "read_bgr"):
        return video.read_bgr(index)
    cap = cv2.VideoCapture(video.path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def _extract_frames(video, indices, out_dir, quality, width):
    """Decode once and write the wanted frames, rather than seeking per frame."""
    import cv2
    want = set(int(i) for i in indices)
    if not want:
        return

    if hasattr(video, "read_bgr"):          # already-extracted frame folder
        for i in sorted(want):
            img = video.read_bgr(i)
            if img is None:
                continue
            if width and img.shape[1] != width:
                h = int(round(img.shape[0] * width / img.shape[1] / 2) * 2)
                img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out_dir, f"{i:06d}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
        return

    cap = cv2.VideoCapture(video.path)
    i, n = 0, 0
    hi = max(want)
    while i <= hi:
        ok, img = cap.read()
        if not ok:
            break
        if i in want:
            if width and img.shape[1] != width:
                h = int(round(img.shape[0] * width / img.shape[1] / 2) * 2)
                img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out_dir, f"{i:06d}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
            n += 1
        i += 1
    cap.release()


def _compose(img, ego, cam, steer, t_ref, args):
    over = viz.draw_trajectory_on_image(img, ego, cam, vehicle_width=args.vehicle_width,
                                        lateral_offset_m=0.0)
    over = viz.draw_hud(over, ego)
    sw = float(np.interp(t_ref, steer.t, steer.wheel_deg))
    sr = float(np.interp(t_ref, steer.t, steer.wheel_rate_deg_s))
    sv = bool(np.interp(t_ref, steer.t, steer.valid.astype(float)) > 0.5)
    over = viz.draw_steering_wheel(over, sw, valid=sv, rate_deg_s=sr)
    bev = viz.draw_bev_cv(ego, size=(int(args.panel_height * 0.86), args.panel_height),
                          vehicle_width=args.vehicle_width,
                          lateral_offset_m=0.0)
    return viz.compose_panels(over, bev, height=args.panel_height)


def _render_video(video, indices, ft, traj, steer, cam, args, out_path, log):
    import cv2
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        log("ffmpeg not on PATH; skipping the video", "WARN")
        return 0
    fps = max(video.avg_fps / max(args.video_stride, 1), 1.0)
    proc, n = None, 0
    t0 = time.time()
    for k, fi in enumerate(indices):
        ego = _ego_at(traj, float(ft[fi]), args)
        img = _grab_bgr(video, int(fi))
        if ego is None or img is None:
            continue
        panel = _compose(img, ego, cam, steer, float(ft[fi]), args)
        if proc is None:
            h, w = panel.shape[:2]
            proc = subprocess.Popen(
                [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-",
                 "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
                stdin=subprocess.PIPE)
        proc.stdin.write(panel.tobytes())
        n += 1
        if k and k % 300 == 0:
            log(f"    video {k}/{len(indices)} ({time.time() - t0:.0f}s)")
    if proc is not None:
        proc.stdin.close()
        proc.wait()
    log(f"    video: {n} frames in {time.time() - t0:.0f}s -> "
        f"{os.path.getsize(out_path) / 1e6:.0f} MB" if n else "    video: nothing written")
    return n


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="folder containing Sensor Logger zips")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scratch", default=None, help="local working directory")
    ap.add_argument("--limit", type=int, default=0, help="process at most N recordings")
    ap.add_argument("--only", default=None, help="substring filter on the zip name")
    ap.add_argument("--force", action="store_true", help="reprocess even if already done")
    ap.add_argument("--no-retry-failed", action="store_true")

    # vehicle / mount
    ap.add_argument("--lateral-offset", type=float, default=-0.35,
                    help="camera offset from the vehicle centreline, +left. Default is "
                         "NEGATIVE (phone mounted right of centre). Unobservable from "
                         "motion - measure it once per mount.")
    ap.add_argument("--cam-height", type=float, default=1.17)
    ap.add_argument("--vehicle-width", type=float, default=1.8)
    ap.add_argument("--wheelbase", type=float, default=AUDI_A6_ETRON["wheelbase_m"])
    ap.add_argument("--steering-ratio", type=float, default=AUDI_A6_ETRON["steering_ratio"])
    ap.add_argument("--max-wheel-rate", type=float, default=180.0)

    # trajectory export
    ap.add_argument("--t-past", type=float, default=3.0)
    ap.add_argument("--t-future", type=float, default=5.0)
    ap.add_argument("--dt-out", type=float, default=0.1)

    # calibration
    ap.add_argument("--calib-seconds", type=float, default=40.0)
    ap.add_argument("--plane-calib", action="store_true", default=True,
                    help="measure camera height from the road-plane homography")
    ap.add_argument("--no-plane-calib", dest="plane_calib", action="store_false")
    ap.add_argument("--mount-longitudinal", type=float, default=2.1,
                    help="phone position ahead of the vehicle reference point (m). "
                         "GNSS locates the phone, so the exported trajectory is "
                         "re-referenced to the rear axle using this and "
                         "--lateral-offset. Set 0 to keep the phone as origin")
    ap.add_argument("--standstill-speed", type=float, default=0.15,
                    help="speed below which the vehicle counts as stopped (m/s). "
                         "Strict on purpose: low speed is not standstill")
    ap.add_argument("--t-future-standstill", type=float, default=1.0,
                    help="future horizon emitted while stopped (s)")
    ap.add_argument("--scale-calib", action="store_true", default=True,
                    help="separate focal from camera height via f*h and lane width")
    ap.add_argument("--no-scale-calib", dest="scale_calib", action="store_false")
    ap.add_argument("--lane-width", type=float, default=3.65,
                    help="true lane width in metres; external metric knowledge that "
                         "lets f and h be separated (US 12 ft = 3.66, DE = 3.50)")
    ap.add_argument("--lane-calib", action="store_true", default=True,
                    help="estimate mount yaw and lateral offset from lane markings")
    ap.add_argument("--no-lane-calib", dest="lane_calib", action="store_false")
    ap.add_argument("--vp-calib", action="store_true", default=True,
                    help="measure roll from the vertical vanishing point")
    ap.add_argument("--no-vp-calib", dest="vp_calib", action="store_false")
    ap.add_argument("--vp-frames", type=int, default=90)

    # rendering
    ap.add_argument("--frame-stride", type=int, default=1, help="0 disables frame export")
    ap.add_argument("--frame-quality", type=int, default=90)
    ap.add_argument("--frame-width", type=int, default=0, help="0 keeps native width")
    ap.add_argument("--video-stride", type=int, default=1)
    ap.add_argument("--panel-height", type=int, default=720)
    ap.add_argument("--crf", type=int, default=22)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    args.lateral_offset_prior = args.lateral_offset    # restored per recording

    os.makedirs(args.output_dir, exist_ok=True)
    scratch = args.scratch or os.path.join(os.environ.get("TEMP", "/tmp"), "trajpipe")
    os.makedirs(scratch, exist_ok=True)
    master = Log(os.path.join(args.output_dir, "_pipeline.log"))
    reg = Registry(os.path.join(args.output_dir, "_registry.json"))

    zips = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(".zip"))
    if args.only:
        zips = [z for z in zips if args.only in z]
    master(f"pipeline {PIPELINE_VERSION}: {len(zips)} archive(s) in {args.input_dir}")
    master(f"output -> {args.output_dir}   scratch -> {scratch}")
    master(f"lateral offset {args.lateral_offset:+.2f} m, camera height {args.cam_height:.2f} m")

    done = 0
    for i, z in enumerate(zips, 1):
        if args.limit and done >= args.limit:
            master(f"stopping at --limit {args.limit}")
            break
        zp = os.path.join(args.input_dir, z)
        fp = fingerprint(zp)
        go, why = reg.should_process(z, fp, retry_failed=not args.no_retry_failed)
        if args.force:
            go, why = True, "forced"
        master("")
        master(f"[{i}/{len(zips)}] {z}  ({why})")
        if not go:
            continue

        stem = os.path.splitext(z)[0]
        out_dir = os.path.join(args.output_dir, stem)
        os.makedirs(out_dir, exist_ok=True)
        log = Log(os.path.join(args.output_dir, "_pipeline.log"),
                  os.path.join(out_dir, "process.log"))
        reg.set(z, status=RUNNING, fingerprint=fp, pipeline_version=PIPELINE_VERSION,
                started_utc=datetime.now(timezone.utc).isoformat())
        try:
            rep = process_one(zp, out_dir, scratch, args, log)
            reg.set(z, status=rep["status"], fingerprint=fp,
                    pipeline_version=PIPELINE_VERSION,
                    verdict=rep.get("diagnosis", {}).get("verdict"),
                    verdict_reason="; ".join(
                        f["code"] for f in rep.get("diagnosis", {}).get("findings", [])
                        if f["level"] == DG.REJECT),
                    duration_s=round(rep.get("duration_s", 0), 1),
                    finished_utc=rep.get("finished_utc"),
                    metrics=rep.get("diagnosis", {}).get("stats", {}))
            master(f"    -> {rep['status']} in {rep.get('duration_s', 0):.0f}s")
        except Exception as e:
            tb = traceback.format_exc()
            log(f"UNHANDLED: {e}", "ERROR")
            log.block("traceback", tb, "ERROR")
            reg.set(z, status=FAILED, fingerprint=fp, error=str(e),
                    pipeline_version=PIPELINE_VERSION,
                    finished_utc=datetime.now(timezone.utc).isoformat())
            with open(os.path.join(out_dir, "ERROR.txt"), "w", encoding="utf-8") as f:
                f.write(tb)
            master(f"    -> FAILED: {e}", "ERROR")
        finally:
            log.close()
            done += 1

    master("")
    master(f"pipeline finished. {reg.summary()}")
    master(f"registry: {reg.path}")
    master.close()


if __name__ == "__main__":
    main()
