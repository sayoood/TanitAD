"""The devkit-side AgentInput EXPORT (E2's ``export_agent_inputs.py``, promoted VERBATIM into
``devkit_side/``) — run in the NavSim venv, cached per split under ``EXP_ROOT/exports/<split>/``.

Why an export (E2, verbatim reasoning): the declared-input manifest is only evidence if the numbers
it declares are the numbers the devkit hands ANY agent; a second implementation of
``get_agent_input`` in another venv is how two "independent" checks agree on a wrong answer. The
per-token FINGERPRINT is re-computed by the seam agent on the object the scorer hands it and must
match (a consistency check; NOT a key — MEASURED: identical ego histories on distinct renders).

Cache validity is a CONTENT key (export-script blob, both split yamls' sha256, the devkit's
``dataclasses.py`` blob); a stale export is regenerated, never reused.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from . import profiles as P


def _sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def export_key(prof: "P.SplitProfile") -> dict:
    return {"export_script_blob": P.git_blob(P.EXPORT_SCRIPT),
            "scene_filter_yaml_sha256": _sha256(P.TTS / "scene_filter" / f"{prof.name}.yaml"),
            "split_yaml_sha256": _sha256(P.TTS / f"{prof.name}.yaml"),
            "devkit_dataclasses_blob": P.git_blob(P.DEVKIT / "navsim" / "common" / "dataclasses.py"),
            "syn_sensors": str(prof.syn_sensors).replace(os.sep, "/"),
            "split": prof.name}


def ensure_export(prof: "P.SplitProfile", *, root: Path | None = None, log=print) -> tuple:
    """-> (doc, record). Reuses a cached export only if its content key and sha256 match."""
    out_dir = Path(root or (P.EXP_ROOT / "exports")) / prof.name
    doc_path, done = out_dir / "navsim_agent_inputs.json", out_dir / "EXPORT_DONE.json"
    key = export_key(prof)
    if done.exists() and doc_path.exists():
        d = json.loads(done.read_text(encoding="utf-8"))
        if d.get("key") == key and d.get("sha256") == _sha256(doc_path):
            log(f"[navsim] export reused: {doc_path} (content key + sha256 match)")
            return json.loads(doc_path.read_text(encoding="utf-8")), {**d, "reused": True}
    out_dir.mkdir(parents=True, exist_ok=True)
    env = P.scorer_env(P.EXP_ROOT)
    env.update({"E2_SPLIT": prof.name, "E2_SYN_SENSORS": str(prof.syn_sensors).replace(os.sep, "/")})
    cmd = [str(P.PY), str(P.EXPORT_SCRIPT), "--out", str(out_dir), "--skip-log-windows"]
    t0 = time.time()
    lp = out_dir / "export.log"
    with open(lp, "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(out_dir)).returncode
    if rc != 0 or not doc_path.exists():
        raise P.Refusal(f"export failed rc={rc} (log {lp}): "
                        + lp.read_text(encoding="utf-8", errors="replace")[-600:])
    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    bad = []
    if doc.get("n_stage1") != prof.n_stage1 or doc.get("n_stage2") != prof.n_stage2:
        bad.append(f"counts {doc.get('n_stage1')}/{doc.get('n_stage2')} != {prof.n_stage1}/{prof.n_stage2}")
    if any(not r.get("fingerprint") for r in doc.get("tokens", {}).values()):
        bad.append("a token has no fingerprint")
    if bad:
        raise P.Refusal(f"export {doc_path} fails its checks: {bad}")
    rec = {"key": key, "sha256": _sha256(doc_path), "path": str(doc_path).replace(os.sep, "/"),
           "n_tokens": len(doc["tokens"]), "wall_s": round(time.time() - t0, 1), "cmd": cmd,
           "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    tmp = done.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    os.replace(tmp, done)
    log(f"[navsim] export built: {rec['n_tokens']} tokens in {rec['wall_s']} s")
    return doc, {**rec, "reused": False}


def compare_exports(doc: dict, ref_path: Path) -> dict:
    """Control: this export's token set + fingerprints vs a BANKED export (E2's)."""
    if not Path(ref_path).exists():
        return {"status": "UNAVAILABLE", "reason": f"reference export absent: {ref_path}", "n": 0}
    ref = json.loads(Path(ref_path).read_text(encoding="utf-8"))
    a, b = doc["tokens"], ref["tokens"]
    common = set(a) & set(b)
    fp_diff = sorted(t for t in common if a[t]["fingerprint"] != b[t]["fingerprint"])
    return {"status": "OK", "reference": str(ref_path).replace(os.sep, "/"), "same_token_set": set(a) == set(b),
            "n_common": len(common), "n_fingerprint_mismatch": len(fp_diff), "first_mismatch": fp_diff[:3],
            "identical": set(a) == set(b) and not fp_diff}
