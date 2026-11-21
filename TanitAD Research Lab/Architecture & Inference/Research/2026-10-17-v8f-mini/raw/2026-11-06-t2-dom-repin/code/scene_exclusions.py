"""Rule R-SCN of AMENDMENT_T2_SCENE_REFUSAL.md (frozen 2167a7eb, Master Mind, 2026-11-06): which scenes a stage's renderer refused before any planner ran.

A scene is EXCLUDED iff every one of its cells in the stage is FAIL with 0 records, and each of those cells ended on one of two scene-level refusals:
  ACTOR_REFUSED     `actor placement REFUSED by its own falsifier: {json}` (actor_map.attach_actors_verified, verdict REFUSED)
  NO_DYNAMIC_LAYER  `KeyError: "no timestamps_us_ranges for layer 'dynamic_rigids'..."` (actor_map.layer_track_ranges)
Attribution: the stage's failed cells, in run order, are matched one to one with the terminal events of that pass's cell log (a refusal line, or the final
exception line of a traceback). The counts must be equal; a scene carries one reason; an ACTOR_REFUSED scene's n_tracks_json must equal the track count of its
own sequence_tracks.json, sha-checked against the stage's scene sources. Anything else is a PROBLEM, and a stage with any problem is UNATTRIBUTED (the read is
then NOT-READ).

CLI: --result STAGE*_RESULT.json --pull <dir holding rollouts/> --log deciding=<path>[@from_line] --log beside=<path>[@from_line] --sources SCENE_SOURCES*.json
     --out EXCLUSIONS.json
"""
import argparse
import glob
import hashlib
import json
import os
import re

REFUSED = "actor placement REFUSED by its own falsifier: "
NO_LAYER = re.compile(r"""^KeyError: "no timestamps_us_ranges for layer 'dynamic_rigids'""")
EXC_LINE = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Exit|Interrupt)\b")
REASONS = ("ACTOR_REFUSED", "NO_DYNAMIC_LAYER")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def terminal_events(lines):
    """-> [(reason, info)] in log order. reason in REASONS, or "OTHER:<exception line>" for any other traceback."""
    ev, in_tb = [], False
    for ln in lines:
        s = ln.rstrip("\r\n")
        if s.startswith(REFUSED):
            try:
                info = json.loads(s[len(REFUSED):])
            except ValueError:
                info = {"unparsed": s[len(REFUSED):][:200]}
            ev.append(("ACTOR_REFUSED", info))
            in_tb = False
        elif s.startswith("Traceback (most recent call last):"):
            in_tb = True
        elif in_tb and EXC_LINE.match(s):
            ev.append(("NO_DYNAMIC_LAYER", {}) if NO_LAYER.match(s) else ("OTHER:" + s[:200], {}))
            in_tb = False
    return ev


def track_count(path):
    d = json.load(open(path, encoding="utf-8"))
    return len(d[next(iter(d))]["tracks_data"]["tracks_id"])


def scene_cells(pull, scene8):
    """{"<tc>|<arm>": state} for every cell directory of the scene in the pull; state DONE, FAIL0 (FAIL with records=0/...), FAIL (other) or NONE."""
    out = {}
    for d in sorted(glob.glob(f"{pull}/rollouts/{scene8}/objects/*/*")):
        if not os.path.isdir(d):
            continue
        tc, arm = d.replace("\\", "/").split("/")[-2:]
        if os.path.exists(f"{d}/DONE"):
            st = "DONE"
        elif os.path.exists(f"{d}/FAIL"):
            st = "FAIL0" if " records=0/" in " " + open(f"{d}/FAIL", encoding="utf-8").read() else "FAIL"
        else:
            st = "NONE"
        out[f"{tc}|{arm}"] = st
    return out


def build(result, pull, logs, sources):
    """result: the stage's RESULT dict. logs: {pass: (path, from_line)} (1-based; lines before from_line belong to an earlier run). sources: SCENE_SOURCES list."""
    st = result["steps"]["rollouts"]["state"]
    problems, cell_reason = [], {}
    log_meta = {}
    for p in ("deciding", "beside"):
        failed = list(st.get(p, {}).get("failed") or [])
        if not failed:
            log_meta[p] = {"n_failed": 0}
            continue
        if p not in logs:
            problems.append(f"{p}: {len(failed)} failed cells and no cell log given")
            continue
        path, frm = logs[p]
        lines = open(path, "rb").read().decode("utf-8", errors="replace").split("\n")       # split on LF only, as `wc -l` / `sed -n` count: progress bars carry bare CRs
        ev = terminal_events(lines[frm - 1:])
        log_meta[p] = {"log": os.path.basename(path), "log_sha256": sha256(path), "from_line": frm, "n_lines": len(lines), "n_failed": len(failed), "n_events": len(ev)}
        if len(ev) != len(failed):
            problems.append(f"{p}: {len(failed)} failed cells but {len(ev)} terminal events in the log from line {frm}")
            continue
        for key, (reason, info) in zip(failed, ev):
            cell_reason[key] = (reason, info)
    by_scene = {}
    for key, (reason, info) in cell_reason.items():
        by_scene.setdefault(key.split("|")[0], []).append((key, reason, info))
    src = {}
    for s in sources:
        src.setdefault(s["scene"][:8], {"scene": s["scene"], "idx": int(s["idx"])})
        if s["rel"] == "extracted/sequence_tracks.json":
            src[s["scene"][:8]]["tracks"] = s
    excluded = {}
    for sc8, items in sorted(by_scene.items()):
        reasons = sorted({r for _, r, _ in items})
        cells = scene_cells(pull, sc8)
        rec = {"scene": src.get(sc8, {}).get("scene"), "idx": src.get(sc8, {}).get("idx"), "reason": reasons[0] if len(reasons) == 1 else reasons,
               "n_failed_cells": len(items), "n_cell_dirs": len(cells), "cell_states": sorted(set(cells.values()))}
        bad = []
        if len(reasons) != 1 or reasons[0] not in REASONS:
            bad.append(f"reasons {reasons}")
        if not cells or any(v != "FAIL0" for v in cells.values()):
            bad.append(f"not every cell is FAIL with 0 records: {dict(sorted(cells.items()))}")
        if len(items) != len(cells):
            bad.append(f"{len(items)} failed keys but {len(cells)} cell directories")
        if reasons == ["ACTOR_REFUSED"]:
            fps = sorted({json.dumps({k: i.get(k) for k in ("n_tracks_json", "n_layer_tracks", "n_mapped")}, sort_keys=True) for _, _, i in items})
            verdicts = sorted({str(i.get("verdict")) for _, _, i in items})
            rec["fingerprint"] = json.loads(fps[0]) if len(fps) == 1 else fps
            rec["falsifier_pass_frames"] = sorted({i.get("falsifier_pass_frames") for _, _, i in items}, key=str)
            if len(fps) != 1:
                bad.append(f"{len(fps)} different refusal fingerprints")
            if verdicts != ["REFUSED"]:
                bad.append(f"verdicts {verdicts}")
            t = src.get(sc8, {}).get("tracks")
            if not t or not os.path.exists(t["source"]):
                bad.append("no sequence_tracks.json in the scene sources to cross-check n_tracks_json")
            elif sha256(t["source"]) != t["sha256"]:
                bad.append(f"sequence_tracks.json sha256 differs from the sources: {t['source']}")
            else:
                n = track_count(t["source"])
                rec["tracks_file_count"] = n
                if len(fps) == 1 and rec["fingerprint"]["n_tracks_json"] != n:
                    bad.append(f"n_tracks_json {rec['fingerprint']['n_tracks_json']} != {n} tracks in its own sequence_tracks.json")
        if bad:
            rec["problems"] = bad
            problems += [f"{sc8}: {b}" for b in bad]
        excluded[sc8] = rec
    return {"rule": "R-SCN, AMENDMENT_T2_SCENE_REFUSAL.md (frozen 2167a7eb)", "stage": result.get("stage"), "manifest_sha256": result.get("manifest_sha256"),
            "logs": log_meta, "excluded": excluded, "n_excluded": len(excluded), "problems": problems,
            "verdict": "ATTRIBUTED" if not problems else "UNATTRIBUTED"}


def parse_log(arg):
    p, rest = arg.split("=", 1)
    path, frm = (rest.rsplit("@", 1) if "@" in os.path.basename(rest) else (rest, "1"))
    return p, (path, int(frm))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--pull", required=True)
    ap.add_argument("--log", action="append", default=[])
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = build(json.load(open(a.result, encoding="utf-8")), a.pull, dict(parse_log(x) for x in a.log), json.load(open(a.sources, encoding="utf-8")))
    res["inputs"] = {"result_sha256": sha256(a.result), "sources_sha256": sha256(a.sources)}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"verdict": res["verdict"], "n_excluded": res["n_excluded"], "problems": res["problems"][:5],
                      "excluded": {k: v["reason"] for k, v in res["excluded"].items()}}))


if __name__ == "__main__":
    main()
