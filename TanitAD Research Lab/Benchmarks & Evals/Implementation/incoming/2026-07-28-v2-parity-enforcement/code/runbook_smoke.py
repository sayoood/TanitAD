"""Run the v5 runbook (V2_PARITY_ENFORCEMENT.md §7) end to end on a SYNTHETIC
corpus, so the published command list is executed rather than asserted.

⚠️ A runbook that has never been run is a hypothesis. This exercises the real
``scripts/register_v2_sibling.py`` as a subprocess — verify-only, then register
+ write-manifest, then the trainer guard, then a re-registration attempt — using
the EXACT corpus key §7 tells a v5 launch to use
(``physicalai-train-e438721ae894-w120-256x640cyl``). That key deliberately
contains its parent's key, so it is also the live test of the ``corpus_key_of``
longest-match tie-break.

🔒 Synthetic clip ids and a synthetic manifest in a temp dir. The committed
``parity_manifest.json`` is NOT touched.

    python runbook_smoke.py --out raw/runbook_smoke_2026-07-27.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
STACK = REPO / "stack"
sys.path.insert(0, str(STACK))

from tanitad.data import parity                                  # noqa: E402

SRC = "synthetic-train-aaaaaaaaaaaa"
V5_KEY = f"{parity.PARITY_TRAIN_KEY}-w120-256x640cyl"
CLIPS = [f"{i:02d}cl-{chr(ord('a') + i)}" for i in range(12)]
SKIPS = [3, 7]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    steps = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        man = {"schema": parity.MANIFEST_SCHEMA, "corpora": {SRC: {
            "corpus_key": SRC, "split": "train",
            "episode_count": len(CLIPS) - len(SKIPS),
            "uid_kind": parity.EPCACHE_UID_KIND,
            "clip_membership": {
                "n_clips": len(CLIPS),
                "clip_id_sha256_sorted": parity.uid_digest(CLIPS),
                "ordered_equals_sorted": CLIPS == sorted(CLIPS),
                "decode_failures": len(SKIPS)},
            "skip_indices": SKIPS, "skip_count": len(SKIPS)}}}
        mp = root / "m.json"
        mp.write_text(json.dumps(man, indent=1), encoding="utf-8")
        cache = root / V5_KEY
        cache.mkdir(parents=True)
        for c in CLIPS:
            (cache / f"{c}{parity.V2_SUFFIX}").touch()
        (cache / "_geometry.json").write_text(json.dumps({
            "frame": {"height": 256, "width": 640, "f_ref": 305.5774907364391,
                      "projection": "cylindrical"},
            "frame_tag": "256x640f305.5775cyl", "codec": "png",
            "geometry_check": {"requested_hfov_deg": 120.0,
                               "achieved_hfov_deg": 120.0}}), encoding="utf-8")
        clips = root / "clips.txt"
        clips.write_text("\n".join(CLIPS) + "\n", encoding="utf-8")

        def run(*extra):
            return subprocess.run(
                [sys.executable, str(STACK / "scripts" / "register_v2_sibling.py"),
                 "--cache", str(cache), "--expect-clips", str(clips),
                 "--source-key", SRC, "--manifest", str(mp), *extra],
                capture_output=True, text=True, encoding="utf-8", cwd=str(STACK))

        # ---- §7 step 2a --------------------------------------------------- #
        r = run("--verify-only", "--out", str(root / "v.json"))
        v = json.loads((root / "v.json").read_text(encoding="utf-8"))
        steps.append({"step": "2a  register_v2_sibling.py --verify-only",
                      "exit": r.returncode,
                      "pass_criteria_as_published": {
                          "extra_count == 0": v["extra_count"] == 0,
                          "membership_identical OR shortfall==recorded 24": bool(
                              v["membership_identical"]
                              or v["shortfall_matches_recorded_skips"])},
                      "observed": {k: v[k] for k in
                                   ("clips_built", "clips_expected",
                                    "membership_identical", "missing_count",
                                    "extra_count",
                                    "shortfall_matches_recorded_skips", "mode")},
                      "wrote_nothing_to_the_manifest": json.loads(
                          mp.read_text(encoding="utf-8"))["corpora"].keys()
                      .__len__() == 1})

        # ---- §7 step 2c --------------------------------------------------- #
        r = run("--new-key", V5_KEY, "--write-manifest", "--out",
                str(root / "e.json"))
        m2 = json.loads(mp.read_text(encoding="utf-8"))
        ent = m2["corpora"].get(V5_KEY, {})
        steps.append({"step": "2c  register_v2_sibling.py --write-manifest",
                      "exit": r.returncode, "registered": V5_KEY in m2["corpora"],
                      "uid_kind": ent.get("uid_kind"),
                      "episode_count": ent.get("episode_count"),
                      "derived_from": (ent.get("provenance") or {}).get(
                          "derived_from"),
                      "geometry_recorded": bool(
                          (ent.get("provenance") or {}).get("geometry")),
                      "reminds_the_operator_to_stage":
                          "NOW STAGE IT" in (r.stdout or "")})

        # ---- the key resolves to the SIBLING, not the parent --------------- #
        resolved = parity.corpus_key_of(cache, mp)
        # ⚠️ the "old rule" must be simulated over the SAME candidate set
        # corpus_key_of uses — the hardcoded keys UNION the manifest's. A first
        # draft of this probe used only the manifest's corpora and therefore
        # reported the sibling for both rules, i.e. it did not test the thing it
        # printed a "reading" about.
        cand = ({parity.PARITY_TRAIN_KEY, parity.PARITY_VAL_KEY,
                 *parity.LEAKY_SPLIT_KEYS} | set(m2["corpora"]))
        s = str(cache.resolve()).replace("\\", "/")
        legacy = next((k for k in sorted(cand) if k in s), None)
        steps.append({"step": "corpus_key_of on the published key",
                      "resolved": resolved, "is_the_sibling": resolved == V5_KEY,
                      "old_lexicographic_rule_would_have_resolved": legacy,
                      "old_rule_was_wrong": legacy != V5_KEY,
                      "candidates_matching_this_path": sorted(
                          k for k in cand if k in s),
                      "reading": "the runbook key contains its parent's key on "
                                 "purpose; without the longest-match tie-break "
                                 "the guard reads a registered wide cache as "
                                 "the RAW parent corpus (uid_kind "
                                 "epcache_basename) and refuses it for the "
                                 "wrong reason."})

        # ---- §7 step 4: the trainer guard --------------------------------- #
        rec = parity.assert_v2_parity_cache(cache, label="v5-launch",
                                            require=True, manifest_path=mp)
        steps.append({"step": "4  train_flagship4b --require-parity (guard only)",
                      "parity": rec["parity"], "corpus_key": rec["corpus_key"],
                      "episodes_loaded": rec["episodes_loaded"],
                      "content_check": rec["content_check"]})

        # ---- re-registration must be refused ------------------------------ #
        r = run("--new-key", V5_KEY, "--write-manifest")
        steps.append({"step": "re-registration WITHOUT --force",
                      "exit": r.returncode, "refused": r.returncode != 0,
                      "reason": (r.stderr or r.stdout).strip().splitlines()[-1]
                      [:160] if (r.stderr or r.stdout) else ""})

    out = {"what": "V2_PARITY_ENFORCEMENT.md §7 runbook, executed end to end on "
                   "a synthetic corpus with the published key",
           "when": date.today().isoformat(),
           "evidence_class": "MEASURED (subprocess runs of the real "
                             "scripts/register_v2_sibling.py; committed "
                             "parity_manifest.json untouched)",
           "v5_key_under_test": V5_KEY,
           "steps": steps}
    ok = (steps[0]["exit"] == 0 and all(steps[0]["pass_criteria_as_published"].values())
          and steps[1]["registered"] and steps[2]["is_the_sibling"]
          and steps[3]["parity"] and steps[4]["refused"])
    out["ALL_PASS"] = ok
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False),
                           encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
