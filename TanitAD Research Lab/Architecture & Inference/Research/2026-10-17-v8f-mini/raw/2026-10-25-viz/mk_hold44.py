"""Build qland/hold_v8f_report_viz: the post-doc report (report/2026-10-25/: Markdown, the published HTML page, figures, the per-level attention video),
   the attention instrument (raw/2026-10-25-viz/), PREREG_V8F_SFRESH (before any data), and the register: H-MINI-STRAT-1 (the banked
   numbers fail clause 1, comparability unresolved), H-V8F-SFRESH-1 (new), D-V8F-ATTN-1 (new).   python mk_hold44.py <remote tip sha>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
A2, REP, VIZ = S + "grid_a2/", S + "report/", S + "viz/out/"
H = "C:/Users/Admin/qland/hold_v8f_report_viz"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RD, VD = f"{PK}/report/2026-10-25/", f"{PK}/raw/2026-10-25-viz/"
BIN = (".png", ".mp4", ".npz", ".pt")


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    assert len(rel) <= 220, rel
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def stage(src, rel):
    assert show(TIP, rel) is None, f"{rel} already exists on the tip"
    b = open(src, "rb").read()
    assert b, src
    put("stage", rel, b if rel.endswith(BIN) else b.replace(b"\r\n", b"\n"))


for n in ("hold_v8f_report_viz", "landed_hold_v8f_report_viz"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
n_rep = 0
for f in sorted(os.listdir(REP)):
    if os.path.isfile(REP + f) and (f.endswith((".md", ".html", ".png", ".mp4")) or f == "make_figures.py"):
        stage(REP + f, RD + f); n_rep += 1
stage(REP + "data/SHA256SUMS.txt", RD + "data/SHA256SUMS.txt")   # the figure inputs are banked JSONs already in the repo; their hashes pin them
n_viz = 0
for root, dirs, files in os.walk(VIZ):
    for f in files:
        p = os.path.join(root, f)
        rel = os.path.relpath(p, VIZ).replace("\\", "/")
        if rel in ("v8f_hierarchy_attention.mp4", "fig_attention_levels.png", "fig_attention_stats.png") or rel.startswith("_test"):
            continue                                  # the video and both figures are banked once, under report/2026-10-25/
        assert os.path.getsize(p) < 20_000_000, rel
        stage(p, VD + rel); n_viz += 1
stage(A2 + "mk_hold44.py", VD + "mk_hold44.py")
stage(A2 + "PREREG_V8F_SFRESH.md", "Project Steering/PREREG_V8F_SFRESH.md")
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-MINI-STRAT-1** |")]
assert len(ix) == 1 and lines[ix[0]].endswith(" |")
lines[ix[0]] = lines[ix[0]][:-2] + (
    " **2026-10-25: clause 1 FAILS on the banked numbers, but comparability is UNRESOLVED.** MEASURED, `raw/full/eval/compare.json`, cross-checked"
    " by the report draft: at 50 m, learned S − the nav-only kinematic baseline = +3.37 [+2.14, +4.53] m (M1) and +3.28 [+2.06, +4.47] m (M2b),"
    " S worse, separated at both seeds. At 80 m it is ≈ +0.2 to +0.4 m, not separated. Clause 2 holds (nav shuffle costs +4.25 m at 50 m)."
    " S is scored at its HELD tick (up to 1.2 s stale) and the baseline is not, so the zero-training discriminating read is pre-registered as"
    " `PREREG_V8F_SFRESH` (`H-V8F-SFRESH-1`). |")
hb = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-HBLEND-1** (2026-10-25) |")]
assert len(hb) == 1
assert not any(x.startswith(("| **H-V8F-SFRESH-1**", "| **D-V8F-ATTN-1**")) for x in lines)
sf = ("| **H-V8F-SFRESH-1** (2026-10-25) | `H-MINI-STRAT-1`'s clause-1 failure (S worse than the nav-only kinematic baseline at 50 m) is an"
      " artefact of scoring S at its held tick: under `--ages fresh`, S − baseline at 50 m is ≤ 0 at both seeds, and fresh − cycle is < 0"
      " and separated | **PRE-REGISTERED, NOT RUN (2026-10-25).** `Project Steering/PREREG_V8F_SFRESH.md`. Eval-only on the banked M2b"
      " s0 / s1, zero training, under `PI-V8F-2026-10-25b`. Outcomes: STALENESS ARTEFACT / S GENUINELY WORSE (the nav-only layer ships"
      " goes to the PI, with an architecture arm) / NOT ESTABLISHED. |")
at = ("| **D-V8F-ATTN-1** (2026-10-25) | What each v8F-mini level attends to (cross-attention of the S, T and O Readers; mean over heads,"
      " queries and layers) | **MEASURED, T0 open loop, M2b s0 / s1, all 2,399 windows; episode-cluster bootstrap (episode draws only);"
      " `…/2026-10-17-v8f-mini/raw/2026-10-25-viz/` (`VIZ_REPORT.md`, `fig_attention_stats.json`, `controls.json`).** Controls passed:"
      " the hooked forward is bit-identical (max|Δ| 0.0) and reproduces the banked ADE6 (2.1753 / 2.0546); the frames align (cached step"
      " = best match 404/404, PSNR ≥ 48.6 dB); in the model-randomisation test the randomised level decorrelates (ρ S +0.08, T +0.03,"
      " O −0.09) while the others stay unchanged (0.0). **Seed-robust:** S and T put 97–100 % of their attention on scene tokens, and O"
      " puts 76 % / 93 % on its conditioning tokens (the upstream plan, ages, limit, ego); effective tokens are S ≈ 6, T ≈ 8, O ≈ 15"
      " of 160. **NOT seed-robust, so not claimed:** which cond token O relies on, the far-row share, and any level-specific"
      " spatial map. Cross-level correlation is as high as cross-seed correlation. The worst window (ADE6 15.95 m) is a strategic route"
      " error propagating down (6.41 m with the GT code). Attention is not explanation. Video and report:"
      " `…/2026-10-17-v8f-mini/report/2026-10-25/`. |")
lines[hb[0] + 1:hb[0] + 1] = [sf, at]
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("v8F-mini post-doc report + per-level attention video banked; PREREG_V8F_SFRESH (before any data); register STRAT-1 / SFRESH / ATTN\n\n"
       "- report/2026-10-25/: V8F_SCIENTIFIC_REPORT.md (method, mathematics, every result with estimator and tier, refutations,\n"
       "  next levers), the published HTML page, 10 figures, v8f_hierarchy_attention.mp4 (80 s; S/T/O cross-attention on 4 clips\n"
       "  chosen by a rule fixed before rendering). PI deliverables of PI-V8F-2026-10-25b.\n"
       "- raw/2026-10-25-viz/: the attention instrument + controls (hooked forward bit-identical, frames aligned 404/404,\n"
       "  model-randomisation passed) and the aggregate statistics.\n"
       "- PREREG_V8F_SFRESH: the zero-training read of whether STRAT-1 (S worse than nav-only kinematics at 50 m, +3.3 m sep.)\n"
       "  is a stale-tick artefact (--ages fresh).\n"
       "- Register: H-MINI-STRAT-1 note, H-V8F-SFRESH-1, D-V8F-ATTN-1.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok", n_rep, "report files,", n_viz, "viz files")
