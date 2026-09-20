"""Emit the token -> |v0| join key for navhard's synthetic scenes, and evaluate the
PRE-REGISTERED fast-start threshold on it.

⛔ ORDERING, which is the whole point: this is run BEFORE any navhard score has been seen by
the author. The threshold below is derived from the SCORING RULE, not from the score
distribution and not from the speed distribution either — see THRESHOLD. The speed
distribution is used only to report the resulting n, after the number is fixed.

/!\ Two Windows obstacles, recorded so nobody rediscovers them: the pickles carry
pathlib.PosixPath (CPython refuses to instantiate it here; peek.py maps it to PurePosixPath,
same string, instantiable everywhere) and they need `nuplan`, which lives only in the navsim
venv at C:/Users/Admin/navsim-crun/venv.

/!\ The archive holds TWO .pkl directories; only synthetic_scene_pickles carries ego_status.
Filtering on ".pkl" alone previously produced 2,731 phantom "unreadable" scenes.
"""
from __future__ import annotations

import io
import json
import pathlib
import sys
import tarfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from peek import WinUnpickler  # noqa: E402

TAR = ("C:/Users/Admin/navsim/data/navsim-v2/"
       "navsim_v2.2_navhard_two_stage_scene_pickles.tar.gz")
HERE = pathlib.Path(__file__).resolve().parent

# ── THRESHOLD, FIXED BEFORE ANY SCORE IS SEEN ────────────────────────────────
# Derived from the CLAUSE, not from any distribution. Both constants are READ FROM THE
# INSTALLED devkit, not remembered:
#   CLAUSE   pdm_scorer.py:232-237 — when the best MASKED (rule-compliant) progress is NOT
#            greater than the threshold, `normalized_progress = np.ones(...)`, i.e. ego
#            progress is 1 for EVERY proposal. 237 is the line that does it.
#   5.0 m    progress_distance_threshold, config/pdm_scoring/scorer/pdm_scorer.yaml:20
#            (same value as the dataclass default, pdm_scorer.py:58)
#   4 s      time_horizon, config/common/agent/{constant_velocity,ego_status_mlp,human,
#            transfuser}_agent.yaml:7 — all four agree
# A vehicle merely MAINTAINING 5.0 m/s covers 20 m in 4 s — 4x the clause distance — so the
# clause cannot fire on progress grounds for any plan that roughly holds speed.
# ⛔ 5.0 was chosen for that reason and not because of where it falls in navhard's
#    distribution. Its n is REPORTED below, never used to pick it.
# ⚠️ The span was carried as `231-236` in four landed docs and `232-237` in a fifth. 231 is a
#    comment and 236 is the bare `else:`; the cited range EXCLUDED line 237, the assignment the
#    claim is about. Corrected here after reading the file.
THRESHOLD_MS = 5.0
HORIZON_S = 4.0


def stream_rows() -> list:
    rows = []
    with tarfile.open(TAR, "r:gz") as tf:
        for m in tf:
            if not (m.isfile() and "/synthetic_scene_pickles/" in m.name
                    and m.name.endswith(".pkl")):
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            try:
                o = WinUnpickler(io.BytesIO(f.read())).load()
                tok = o["scene_metadata"]["scene_token"]
                v = o["frames"][-1]["ego_status"]["ego_velocity"]
                rows.append((tok, round((float(v[0]) ** 2 + float(v[1]) ** 2) ** 0.5, 6)))
            except Exception:
                pass
            if len(rows) % 1000 == 0 and rows:
                print(f"  {len(rows)} scenes", flush=True)
    return rows


def main() -> int:
    csv = HERE / "navhard_token_v0.csv"
    # --from-csv regenerates the JSON from the banked table (e.g. after a wording fix) WITHOUT
    # re-streaming, so the numbers cannot drift between the table and its description.
    if "--from-csv" in sys.argv and csv.exists():
        rows = [(a, float(b)) for a, b in
                (ln.split(",") for ln in csv.read_text().splitlines()[1:])]
    else:
        rows = stream_rows()

    assert rows, "control: zero scenes read"
    assert len({t for t, _ in rows}) == len(rows), "control: scene tokens are not unique"

    fast = [t for t, s in rows if s >= THRESHOLD_MS]
    slow = [t for t, s in rows if s < THRESHOLD_MS]
    covered = THRESHOLD_MS * HORIZON_S

    csv.write_bytes(
        ("scene_token,v0_ms\n" + "\n".join(f"{t},{s}" for t, s in rows)).encode())

    meta = {
        "_what": "pre-registered fast-start stratum for H-NAVHARD-STOP-1, and its join key",
        "_evidence_class": "MEASURED (ours) — streamed from the archive, nothing extracted",
        "_ordering": ("FIXED BEFORE ANY navhard SCORE WAS SEEN by the author. The threshold is "
                      "derived from the scoring rule, not from the speed or score distribution."),
        "threshold_ms": THRESHOLD_MS, "horizon_s": HORIZON_S,
        "_why_this_threshold": (
            f"pdm_scorer.py:232-237 sets normalized_progress = ones(...) — EP 1 for EVERY "
            f"proposal — when the best MASKED (rule-compliant) progress is not greater than "
            f"progress_distance_threshold = 5.0 m (pdm_scorer.yaml:20). Over the "
            f"{HORIZON_S:g} s agent time_horizon (agent yamls:7), merely MAINTAINING "
            f"{THRESHOLD_MS:g} m/s covers {covered:g} m = {covered/5:g}x the clause distance, so "
            f"the clause cannot fire on progress grounds for a plan that roughly holds speed."),
        "_citation_correction": (
            "the clause was carried as pdm_scorer.py:231-236 in four landed docs and 232-237 in "
            "a fifth. 231 is a comment, 236 is the bare `else:`, and the cited range EXCLUDED "
            "line 237 — the assignment the claim is about. 232-237 is correct; both constants "
            "were re-read from the installed devkit rather than carried forward."),
        "n_total": len(rows), "n_fast": len(fast), "n_slow": len(slow),
        "frac_fast": round(len(fast) / len(rows), 4),
        "join_key": "scene_token -> navhard_token_v0.csv",
        "_prediction": ("on the FAST stratum the <= 5 m EP clause's firing fraction must collapse "
                        "toward zero, and STOP must LOSE to A1. If the clause still fires often, "
                        "the threshold did not do what it was derived to do; if it collapses and "
                        "STOP still wins, the clause is NOT what makes stopping win."),
        "_scope": ("this stratifies navhard's synthetic scenes. Which of them a given bench run "
                   "actually scores is that run's business; the join is by scene_token."),
    }
    print(json.dumps(meta, indent=1))
    (HERE / "navhard_stratum.json").write_bytes(json.dumps(meta, indent=1).encode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
