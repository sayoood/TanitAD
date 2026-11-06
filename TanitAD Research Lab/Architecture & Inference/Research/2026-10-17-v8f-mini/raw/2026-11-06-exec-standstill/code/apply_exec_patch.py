#!/usr/bin/env python3
"""apply_exec_patch.py -- regenerates v8f_t1_eval_exec.py byte for byte from the LF6 T1-kin tool (sha256 949c4b7a...).

The patch adds exactly two optional flags and nothing else:
  --exec-eps EPS      (default 1e-3 = EPS_M = the base executor) the executor's heading threshold, metres per 0.2 s interval (the executor's own
                      tangent norm, the quantity hermite_frames already compares with EPS_M). Below EPS the executed heading is HELD (so the
                      10 Hz yaw rate is 0); positions, speeds, replans and windows are unchanged.
  --dump-exec-ch FILE (default off) writes FILE (npz): per window, for the MODEL's rollout (cl_*) and the P10 floor's (fl_*), the executed 10 Hz
                      track rows 2*t0+1 .. 2*(t0+30) (x, y, psi, v) and the cache-step yaw rate ch1[t0+1 .. t0+30] (rad/s). Reading only.
With neither flag (or --exec-eps 1e-3) every output of the tool is unchanged bit for bit; the header gets the key exec_eps only when EPS != EPS_M.

usage:  python -I apply_exec_patch.py [--src LF6_TOOL] [--dst OUT]
"""
import argparse
import hashlib
import os
import sys

SRC_DEFAULT = "E:/Projects/TanitAD-artifacts/dagland/motion_c/lf6_result_pkg/lf6_build/tool/v8f_t1_eval_ri_lf6.py"
SRC_SHA = "949c4b7a384943015a4b520987bc9c4267dee4a2160f566aa2225208ed3f7f01"
DST_DEFAULT = "E:/Projects/TanitAD-artifacts/dagland/motion_c/exec_standstill/code/v8f_t1_eval_exec.py"

EDITS = [
    # (old, new)  -- each old string must occur EXACTLY once
    ("EPS_M = 1e-3                 # a tangent shorter than 1 mm per 0.2 s interval holds the previous heading (plan_kin_t's convention)\n",
     "EPS_M = 1e-3                 # a tangent shorter than 1 mm per 0.2 s interval holds the previous heading (plan_kin_t's convention)\n"
     "#: EXEC-STANDSTILL (PREREG_EXEC_STANDSTILL.md): the executor's heading threshold in force for this process (default EPS_M = the base executor); set by ``--exec-eps`` in cmd_run.\n"
     "EXEC_STATE = {\"eps\": EPS_M}\n"),
    ("def hermite_frames(k_prev, knots, m: int, psi0: float):\n",
     "def hermite_frames(k_prev, knots, m: int, psi0: float, eps=None):\n"),
    ("    even frame the knot itself. Heading = tangent angle (held while the tangent is < ``EPS_M``); speed = |tangent| / 0.2 s.\"\"\"\n",
     "    even frame the knot itself. Heading = tangent angle (held while the tangent is < ``eps``, default ``EPS_M``); speed = |tangent| / 0.2 s.\"\"\"\n"
     "    eps_ = EPS_M if eps is None else float(eps)\n"),
    ("            if nrm >= EPS_M:\n",
     "            if nrm >= eps_:\n"),
    ("        xy, psi, v = hermite_frames(self.track[2 * (j - 1), :2], knots, m, float(pose[2]))\n",
     "        xy, psi, v = hermite_frames(self.track[2 * (j - 1), :2], knots, m, float(pose[2]), eps=EXEC_STATE[\"eps\"])\n"),
    ("    p.add_argument(\"--dump-heads\", action=\"store_true\",",
     "    p.add_argument(\"--exec-eps\", type=float, default=EPS_M, help=\"EXEC-STANDSTILL (PREREG_EXEC_STANDSTILL.md): the executor's heading threshold in metres per 0.2 s interval (the executor's own tangent norm); below it the executed heading is HELD and the yaw rate is 0. Default 1e-3 = EPS_M = the base executor, bit for bit\")\n"
     "    p.add_argument(\"--dump-exec-ch\", default=\"\", help=\"EXEC-STANDSTILL: a FILE (npz); writes, per window, the executed 10 Hz track rows (x, y, psi, v) 2*t0+1 .. 2*(t0+30) and the cache-step yaw rate ch1[t0+1 .. t0+30] of the model's and of the P10 floor's rollout. Reading only\")\n"
     "    p.add_argument(\"--dump-heads\", action=\"store_true\","),
    ("    FREEZE_STATS.update({\"items\": 0, \"max_row_dev\": 0.0, \"n_unequal_elements\": 0})\n    ri_reset()\n",
     "    FREEZE_STATS.update({\"items\": 0, \"max_row_dev\": 0.0, \"n_unequal_elements\": 0})\n    ri_reset()\n"
     "    if not (math.isfinite(a.exec_eps) and a.exec_eps >= 0.0):\n"
     "        raise SystemExit(f\"REFUSED: --exec-eps must be a finite float >= 0, got {a.exec_eps!r}\")\n"
     "    EXEC_STATE[\"eps\"] = float(a.exec_eps)\n"
     "    exc = {k: [] for k in (\"cl_track\", \"cl_ch1\", \"fl_track\", \"fl_ch1\", \"ci\", \"t0\")}\n"),
    ("        n_clip[\"cl\"] += sum(s.clipped > 0 for s in st_m)\n",
     "        if a.dump_exec_ch:\n"
     "            for role, sts in ((\"cl\", st_m), (\"fl\", st_f)):\n"
     "                for s_ in sts:\n"
     "                    exc[role + \"_track\"].append(np.array(s_.track[2 * s_.t0 + 1:2 * (s_.t0 + H) + 1], np.float64))\n"
     "                    exc[role + \"_ch1\"].append(np.array(s_.d[\"ch1\"][s_.t0 + 1:s_.t0 + H + 1], np.float64))\n"
     "            exc[\"ci\"] += [w[0] for w in chunk]\n"
     "            exc[\"t0\"] += [w[1] for w in chunk]\n"
     "        n_clip[\"cl\"] += sum(s.clipped > 0 for s in st_m)\n"),
    ("    if a.scene == \"hold_now\":                                            # only then: every other --scene value",
     "    if a.exec_eps != EPS_M:                                              # only then: the default (the base executor) leaves the header byte-identical to the LF6 tool's\n"
     "        header[\"exec_eps\"] = {\"eps_m_per_interval\": a.exec_eps, \"base_eps\": EPS_M, \"rule\": \"hermite_frames: the executed heading follows the tangent only when its norm >= eps; below it the previous heading is held (10 Hz yaw rate 0); positions, speeds, replans, windows unchanged\",\n"
     "                               \"prereg\": \"dagland/motion_c/exec_standstill/PREREG_EXEC_STANDSTILL.md\"}\n"
     "    if a.dump_exec_ch:\n"
     "        os.makedirs(os.path.dirname(os.path.abspath(a.dump_exec_ch)), exist_ok=True)\n"
     "        np.savez(a.dump_exec_ch, **{k: np.asarray(v) for k, v in exc.items()}, exec_eps=np.float64(a.exec_eps))\n"
     "    if a.scene == \"hold_now\":                                            # only then: every other --scene value"),
]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--dst", default=DST_DEFAULT)
    a = ap.parse_args(argv)
    raw = open(a.src, "rb").read()
    got = sha256_bytes(raw)
    if got != SRC_SHA:
        print(f"REFUSED: {a.src} is {got}, not the LF6 tool {SRC_SHA}", file=sys.stderr)
        return 2
    if b"\r\n" in raw:
        print("REFUSED: the source carries CRLF endings", file=sys.stderr)
        return 2
    s = raw.decode("utf-8")
    for i, (old, new) in enumerate(EDITS):
        n = s.count(old)
        if n != 1:
            print(f"REFUSED: edit {i} matches {n} times (want exactly 1): {old[:80]!r}", file=sys.stderr)
            return 3
        s = s.replace(old, new)
    os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
    with open(a.dst, "wb") as f:
        f.write(s.encode("utf-8"))
    print("WROTE", a.dst, sha256_bytes(s.encode("utf-8")), "from", got[:8])
    return 0


if __name__ == "__main__":
    sys.exit(main())
