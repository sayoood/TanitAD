"""Generate the two Colab notebooks from cell sources — the dev-side builder.

Why a builder instead of hand-edited .ipynb JSON: the CPU smoke driver
(`colab/smoke_run.py`) executes the SHIPPED .ipynb's code cells, so the cells
must stay pure Python (no ! or % magics — pip installs and the Drive mount are
ordinary guarded Python). Regenerate after editing:

    python colab/nb_build.py

Both notebooks are thin wrappers over `colab/s2_lab_lib.py`; the vocabulary
comes only from `colab/s2_schema.py` (the PROVISIONAL one-file swap point).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _cell(kind: str, src: str, i: int, nb: str) -> dict:
    c = {"cell_type": kind, "id": f"{nb}-{i:02d}", "metadata": {},
         "source": src.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def build(path: Path, cells: list[tuple[str, str]], nb_id: str) -> None:
    nb = {
        "cells": [_cell(k, s, i, nb_id) for i, (k, s) in enumerate(cells)],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {path} ({len(cells)} cells)")


# =========================================================================== #
# SAM3_BACKFILL_115                                                            #
# =========================================================================== #
BACKFILL: list[tuple[str, str]] = []

BACKFILL.append(("markdown", """
# SAM3 backfill — the 115 SAM3-absent aug120 clips

**What this is.** `aug120_pipeline.py` never passed `--n` to `ph0_sam3.py`
(default 4), so **115 of 201** aug120 clips have **no SAM3 record** — every
`batch_*/sam3/sam3.json` holds exactly 4 clips while `SAM3_RC=0` read as full
coverage. Root cause + accounting: `TanitAD Research Hub/Data Engineering/
Implementation/incoming/2026-08-15-aug120-fusion/AUG120_FUSION_RESULT.md` §3.
This notebook re-runs the SAM3 leg for exactly those clips on the free-Colab
T4 (~30 GPU-min), **banking per clip** so a session death costs one clip,
never the run.

**Deliberately the first workload:** small, known shape — it validates the
whole loop (auth → pull → GPU → bank → far-side verify → resume) before the
label lab spends GPU time.

**Rules built in:** the gap is derived from the fused **records'** own
`perception.absent` marker (count records, not files — C18), cross-checked
against two more far-side sources; every push is far-side verified by byte
round-trip; restart = re-run all cells (the far-side listing resumes).

Smoke mode (`S2_SMOKE=1`): CPU-only plumbing test — stubs the GPU leg, banks
to `Sayood/tanitad-s2-lab/smoke/` instead of the production label repo.
"""))

BACKFILL.append(("code", """
# --- parameters -------------------------------------------------------------
import os
SMOKE = os.environ.get('S2_SMOKE', '0') == '1'   # CPU plumbing test (stub GPU)
N_LIMIT = int(os.environ.get('S2_N', '0')) or None      # None = the whole gap
BATCH = int(os.environ.get('S2_BATCH', '12'))    # shards per pull (~36 MB each)
FRAME_STRIDE = 8                                  # ph0_sam3 default
GAP_LIMIT = int(os.environ.get('S2_GAP_LIMIT', '0')) or None
if SMOKE and GAP_LIMIT is None:
    GAP_LIMIT = 12                                # smoke: first K records only
if SMOKE and N_LIMIT is None:
    N_LIMIT = 1
print(f'SMOKE={SMOKE} N_LIMIT={N_LIMIT} BATCH={BATCH} GAP_LIMIT={GAP_LIMIT}')
"""))

BACKFILL.append(("code", """
# --- Drive mount + repo imports (this repo IS the PI's Drive) ---------------
import json, sys, time
from pathlib import Path
try:
    import s2_lab_lib as L
except ImportError:                    # bare Colab: mount Drive, then import
    from google.colab import drive
    drive.mount('/content/drive')
    sys.path.insert(0, '/content/drive/MyDrive/SayBouBase/raw/Projects/'
                       'TanitAD/colab')
    import s2_lab_lib as L
ROOT = L.add_stack_paths()
print('repo root:', ROOT)
L.pip_install_colab(SMOKE)             # no-op off Colab / in smoke
import s2_schema
print('schema:', s2_schema.SCHEMA_VERSION,
      '| v6 drift:', s2_schema.check_v6_drift())
"""))

BACKFILL.append(("code", """
# --- auth + bank target -----------------------------------------------------
# Token: Colab Secret HF_TOKEN (key icon, left sidebar) — never printed.
api = L.hf_api()
WORK = Path('/content/backfill') if L.in_colab() else \\
    ROOT / 'colab' / '_smoke_work' / 'backfill'
WORK.mkdir(parents=True, exist_ok=True)
# smoke banks to the lab repo's smoke/ prefix, NEVER the production label repo
BANK_REPO = L.DS_LAB if SMOKE else L.DS_LABELS
BANK_PREFIX = (L.SMOKE_PREFIX + 'sam3_backfill/') if SMOKE else \\
    L.BACKFILL_PREFIX
L.ensure_repo(api, BANK_REPO)
print(f'banking to {BANK_REPO}/{BANK_PREFIX} (far-side verified per clip)')
"""))

BACKFILL.append(("code", """
# --- the gap, from the RECORDS (C18), cross-checked -------------------------
gap = L.derive_sam3_gap(api, limit=GAP_LIMIT)
L.cross_check_gap(api, gap, partial=GAP_LIMIT is not None)
if GAP_LIMIT is None:
    L.check_gap_fixture(gap, ROOT)     # loud diff vs the banked dev-box list
todo_all = gap['absent']
print(f'SAM3-absent clips: {len(todo_all)} '
      f'(of {gap["n_records_checked"]} records checked)')
"""))

BACKFILL.append(("code", """
# --- resume: find what is done, then continue -------------------------------
done = L.done_set(api, BANK_REPO, BANK_PREFIX)
todo = [c for c in todo_all if c not in done]
if N_LIMIT:
    todo = todo[:N_LIMIT]
print(f'far side already holds {len(done)} -> this run: {len(todo)} clips')
"""))

BACKFILL.append(("code", """
# --- v2 records (the B3 sign boxes SAM3 cross-checks) -----------------------
v2_by = L.load_v2_records(api, set(todo)) if todo else {}
"""))

BACKFILL.append(("code", """
# --- video locations + Alpamayo join (real runs only) -----------------------
loc, REC_PQ = {}, None
if todo and not SMOKE:
    loc = L.w120_locations(api)
    missing_video = [c for c in todo if c not in loc]
    assert not missing_video, (f'{len(missing_video)} gap clips lack w120 '
                               f'shards: {missing_video[:3]}')
    REC_PQ = str(WORK / 'records.parquet')
    if not Path(REC_PQ).exists():
        import shutil
        shutil.copyfile(L.hf_download(L.DS_ALP, 'records.parquet'), REC_PQ)
    print(f'w120 shards located for all {len(todo)} clips')
"""))

BACKFILL.append(("code", """
# --- the SAM3 leg: pull batch -> bridge -> detect -> BANK PER CLIP ----------
import shutil
proc = None
if todo and not SMOKE:
    proc, _meta = L.load_sam3()
    L.gpu_mem_report('sam3 load')
t_start, n_banked = time.time(), 0
for b0 in range(0, len(todo), BATCH):
    batch = todo[b0:b0 + BATCH]
    bwork = WORK / f'b{b0:05d}'
    if SMOKE:
        frames_by = {c: L.stub_frames() for c in batch}
    else:
        L.bridge_batch(batch, loc, REC_PQ, bwork)   # pulls + DELETES shards
        import ph0_pilot
        frames_by = {c: ph0_pilot.sample_clip_frames(
            str(bwork / 'videos' / f'{c}.mp4'), t0_s=8.0)[0] for c in batch}
    for cid in batch:
        rec = (L.stub_sam3_record(cid) if SMOKE else
               L.sam3_leg(proc, frames_by[cid], v2_by[cid],
                          frame_stride=FRAME_STRIDE))
        # ⛔ the count is EXPLICIT, always — the --n default of 4 is the
        # measured root cause of this very gap (AUG120_FUSION_RESULT.md §3)
        rec['_n_explicit'] = len(batch)
        sz = L.bank_json(api, BANK_REPO, f'{BANK_PREFIX}{cid}.json', rec)
        n_banked += 1
        print(f'[bank] {n_banked}/{len(todo)} {cid[:8]} {sz} B '
              'far-side-verified', flush=True)
    L.gpu_mem_report(f'after batch b{b0:05d}')
    shutil.rmtree(bwork, ignore_errors=True)
print(f'BANKED {n_banked} clips in {time.time() - t_start:.0f}s')
"""))

BACKFILL.append(("code", """
# --- final accounting + run manifest + the escalation -----------------------
final_done = L.done_set(api, BANK_REPO, BANK_PREFIX, verify_sample=False)
resid = [c for c in todo_all if c not in final_done]
L.run_manifest(api, BANK_REPO, BANK_PREFIX, 'sam3-backfill', {
    'smoke': SMOKE, 'todo_this_run': len(todo), 'banked_this_run': n_banked,
    'far_side_done_now': len(final_done), 'residual_gap': len(resid),
    'frame_stride': FRAME_STRIDE,
    'n_rule': 'clip count explicit always (aug120 --n root cause)',
    'evidence_class': 'SMOKE-STUB' if SMOKE else 'MEASURED'})
print(f'far side now holds {len(final_done)} · residual gap {len(resid)}')
print('NEXT (escalated here, not buried): re-fuse the backfilled clips — '
      'the 115 fused records still carry perception.absent and must be '
      're-emitted. Owner: aug120-fusion package '
      '(AUG120_FUSION_RESULT.md §9 items 1-2, fuser resumes per clip).')
print('BACKFILL_DONE')
"""))


# =========================================================================== #
# STRATEGIC_LABEL_LAB                                                          #
# =========================================================================== #
LAB: list[tuple[str, str]] = []

LAB.append(("markdown", """
# Strategic label lab — VLM + SAM3 + ego → `g_str` / `a_str`

**Purpose (PI):** *"review if the pipeline is extracting the right
information and optimizing it until the correct strategic vocabulary is
generated in the right format, then we can scale."*

Per clip, three legs run **sequentially** (the 16 GB T4 fits them one at a
time, never together), then fuse:

| leg | engine | provenance tag |
|---|---|---|
| VLM | Qwen3.5-9B, **unsloth 4-bit** (~7–8 GB), grammar-constrained B1–B4 calls from `ph0_v2.py` | `vlm` |
| SAM3 | `facebook/sam3` text-prompted concepts (~3 GB), `ph0_sam3.py` | `sam3` |
| ego | engine-A geometry + spine (`ph0_pilot.py`, `ph1_fuse.py`), yaw vote with `strategic_gt.py` thresholds | `ego` (privileged: labels-only) |

Fusion is `ph1_fuse.py`'s own 2-of-3 voting; the S2 tokens come ONLY from
`colab/s2_schema.py` (**PROVISIONAL** — swaps in one file when the S2-gap
agent's `S2_STRATEGIC_GAP.md` lands). Every token carries **per-token
provenance** for the S-S gate's goal-provenance audit. Labels may use ego;
the goal fields never carry situation-classifier output (asserted per
record).

The **review sheet** at the end renders frames + tokens + provenance +
corroborations per clip — that is the artifact to judge the pipeline by.
Banked per clip to `Sayood/tanitad-s2-lab/` with a run manifest; restart =
re-run all cells (far-side resume).

**Prompt iteration:** the B1–B4 prompts live in `stack/scripts/ph0_v2.py`
(`P_B1…P_B4`) on this same Drive — edit there (from any box), wait for Drive
sync, then `import importlib, ph0_v2; importlib.reload(ph0_v2)` and re-run
the VLM-leg cell. That is the optimise loop.
"""))

LAB.append(("code", """
# --- parameters -------------------------------------------------------------
import os
SMOKE = os.environ.get('S2_SMOKE', '0') == '1'
N = int(os.environ.get('S2_N', '1' if SMOKE else '4'))   # clips this run
CLIP_IDS = [c for c in os.environ.get('S2_CLIPS', '').split(',') if c]
ALLOW_FALLBACK = os.environ.get('S2_ALLOW_FALLBACK', '0') == '1'
EGO_IN_PROMPT = 'past'      # ph0_v2 production setting (speed-redacted to B2)
print(f'SMOKE={SMOKE} N={N} CLIP_IDS={CLIP_IDS or "(auto: sam3-covered)"} '
      f'ALLOW_FALLBACK={ALLOW_FALLBACK}')
"""))

LAB.append(("code", """
# --- Drive mount + repo imports ---------------------------------------------
import json, sys, time
from pathlib import Path
try:
    import s2_lab_lib as L
except ImportError:
    from google.colab import drive
    drive.mount('/content/drive')
    sys.path.insert(0, '/content/drive/MyDrive/SayBouBase/raw/Projects/'
                       'TanitAD/colab')
    import s2_lab_lib as L
ROOT = L.add_stack_paths()
print('repo root:', ROOT)
L.pip_install_colab(SMOKE)
import s2_schema
print('schema:', s2_schema.SCHEMA_VERSION,
      '| v6 drift:', s2_schema.check_v6_drift())
"""))

LAB.append(("code", """
# --- auth + bank target -----------------------------------------------------
api = L.hf_api()
WORK = Path('/content/lab') if L.in_colab() else \\
    ROOT / 'colab' / '_smoke_work' / 'lab'
WORK.mkdir(parents=True, exist_ok=True)
BANK_REPO = L.DS_LAB
BANK_PREFIX = (L.SMOKE_PREFIX + 'lab/') if SMOKE else L.LAB_PREFIX
L.ensure_repo(api, BANK_REPO)
print(f'banking to {BANK_REPO}/{BANK_PREFIX}')
"""))

LAB.append(("code", """
# --- clip selection + resume ------------------------------------------------
# Default pool: the SAM3-COVERED fused clips, so all three legs have ground
# to compare against. Override with S2_CLIPS=<id,id,...>.
ls = json.load(open(L.hf_download(L.DS_LABELS,
                                  L.FUSED_PREFIX + '_label_sources.json')))
covered = sorted(c for c, s in ls['sources'].items() if s.get('sam3'))
pool = CLIP_IDS or covered
done = L.done_set(api, BANK_REPO, BANK_PREFIX, suffix='.s2.json')
clips = [c for c in pool if c not in done][:N]
print(f'pool {len(pool)} · far-side done {len(done)} · this run {len(clips)}')
for c in clips:
    print('  ', c)
"""))

LAB.append(("code", """
# --- inputs: ego npz (always) + video frames + Alpamayo ---------------------
inputs, alp_by = {}, {}
if clips and SMOKE:
    alp_by = {c: L.stub_alpamayo(c) for c in clips}
elif clips:
    REC_PQ = str(WORK / 'records.parquet')
    if not Path(REC_PQ).exists():
        import shutil
        shutil.copyfile(L.hf_download(L.DS_ALP, 'records.parquet'), REC_PQ)
    alp_all = L.load_alpamayo(REC_PQ)
    alp_by = {c: alp_all.get(c) for c in clips}
    loc = L.w120_locations(api)
for cid in clips:
    ego_p = L.hf_download(L.DS_LABELS, f'{L.EGO_PREFIX}{cid}.npz')
    if SMOKE:
        frames, n_past = L.stub_frames(), 16
    else:
        cw = WORK / cid[:8]
        L.bridge_batch([cid], loc, REC_PQ, cw)
        import ph0_pilot
        frames, _t, n_past = ph0_pilot.sample_clip_frames(
            str(cw / 'videos' / f'{cid}.mp4'), t0_s=8.0)
    inputs[cid] = {'ego_npz': ego_p, 'frames': frames, 'n_past': n_past}
print(f'inputs ready for {len(inputs)} clips '
      f'(alpamayo present: {sum(1 for c in inputs if alp_by.get(c))})')
"""))

LAB.append(("code", """
# --- leg 1 (0-GPU): ego geometry — engine A + spine + yaw vote --------------
ego_by = {c: L.ego_leg(inputs[c]['ego_npz']) for c in inputs}
for c, e in ego_by.items():
    v = e['g_str_vote']
    print(f'{c[:8]} ego vote {v["token"]} {v.get("args") or ""} '
          f'(net dyaw {v["net_dyaw_deg_from_t0"]}°) · spine '
          + json.dumps((e['spine'] or {}).get('speed_profile', {}))[:110])
"""))

LAB.append(("code", """
# --- leg 2: VLM (unsloth 4-bit Qwen3.5-9B), grammar-constrained B1–B4 -------
# Resolution is at RUNTIME from a candidate list, newest first; what loaded
# is PRINTED; an older-generation substitute needs ALLOW_FALLBACK=True.
resolved = L.resolve_vlm_model(api, allow_fallback=ALLOW_FALLBACK)
if SMOKE:
    print('[vlm] SMOKE — resolver ran for real (above); records are stubs')
    v2_by = {c: L.stub_vlm_record(c) for c in inputs}
else:
    vlm = L.load_vlm(resolved['model_id'])
    L.gpu_mem_report('vlm load')
    v2_by = {}
    for cid in inputs:
        rec = L.vlm_leg(vlm, inputs[cid]['frames'], inputs[cid]['n_past'],
                        ego_by[cid]['engine_a'], ego_by[cid]['ego_state'],
                        EGO_IN_PROMPT)
        rec['clip_id'] = cid
        v2_by[cid] = rec
        print(f'[vlm] {cid[:8]} all_valid={rec.get("_all_valid")} '
              f'goal={((rec.get("symbols") or {}).get("goal_kind"))}')
    L.gpu_mem_report('vlm leg total')
    L.free_leg(vlm)                    # sequential legs: free before SAM3
    L.gpu_mem_report('after vlm free')
"""))

LAB.append(("code", """
# --- leg 3: SAM3 (text-prompted concepts + B3 box cross-check) --------------
if SMOKE:
    sam3_by = {c: L.stub_sam3_record(c) for c in inputs}
else:
    proc, _m = L.load_sam3()
    L.gpu_mem_report('sam3 load')
    sam3_by = {}
    for cid in inputs:
        sam3_by[cid] = L.sam3_leg(proc, inputs[cid]['frames'], v2_by[cid])
        h = sam3_by[cid]['per_concept_hits']
        print(f'[sam3] {cid[:8]} ' + (', '.join(
            f'{k}:{v}' for k, v in h.items() if v) or
            'no detections (valid abstention)'))
    L.gpu_mem_report('sam3 leg total')
    L.free_leg(proc)
    L.gpu_mem_report('after sam3 free')
"""))

LAB.append(("code", """
# --- fuse (ph1_fuse 2-of-3 voting) -> S2 record -> BANK PER CLIP ------------
entries = []
for cid in inputs:
    fused = L.fuse_one(v2_by[cid], sam3_by.get(cid), inputs[cid]['ego_npz'],
                       alp_by.get(cid))
    s2 = L.to_s2(fused, ego_extra=ego_by[cid])
    sz = L.bank_json(api, BANK_REPO, f'{BANK_PREFIX}{cid}.s2.json',
                     {**s2, '_fused': fused, '_smoke': SMOKE})
    entries.append({'s2': s2, 'fused': fused,
                    'frames': inputs[cid]['frames'], 'smoke': SMOKE})
    g = s2['g_str']
    print(f'[bank] {cid[:8]} {sz} B far-side-verified')
    print(f'   g_str {g["token"]} {g["args"] or ""} prov={g["provenance"]} '
          f'conf={g["confidence"]}')
    print('   a_str', [(a['token'], a['args'], a['provenance'])
                       for a in s2['a_str']] or '(none)')
"""))

LAB.append(("code", """
# --- the REVIEW SHEET: frames + tokens + provenance + corroborations --------
html = L.review_sheet_html(
    entries, title=f'S2 label lab — {time.strftime("%Y-%m-%d %H:%M")} '
                   f'({"SMOKE" if SMOKE else "T4"})')
sheet_path = str(WORK / 'review_sheet.html')
L.show_html(html, sheet_path)          # renders inline in Colab + saves
api.upload_file(path_or_fileobj=sheet_path,
                path_in_repo=f'{BANK_PREFIX}_sheets/'
                             f'{time.strftime("%Y%m%d-%H%M%S")}.html',
                repo_id=BANK_REPO, repo_type='dataset')
print('sheet banked beside the records')
"""))

LAB.append(("code", """
# --- run manifest + resume proof --------------------------------------------
man_rf = L.run_manifest(api, BANK_REPO, BANK_PREFIX, 'label-lab', {
    'smoke': SMOKE, 'n_clips': len(entries),
    'clips': [e['s2']['clip_id'] for e in entries],
    'vlm_resolved': resolved, 'schema_version': s2_schema.SCHEMA_VERSION,
    'ego_in_prompt': EGO_IN_PROMPT,
    'evidence_class': 'SMOKE-STUB' if SMOKE else 'MEASURED'})
done2 = L.done_set(api, BANK_REPO, BANK_PREFIX, suffix='.s2.json')
for e in entries:
    assert e['s2']['clip_id'] in done2, \\
        f'banked clip {e["s2"]["clip_id"]} MISSING from far-side done-set!'
print(f'resume check: far side holds {len(done2)} records; this run\\'s '
      'clips all present -> a session death now costs nothing')
print(f'manifest: {man_rf}')
print('LAB_DONE')
"""))


if __name__ == "__main__":
    build(HERE / "SAM3_BACKFILL_115.ipynb", BACKFILL, "bf")
    build(HERE / "STRATEGIC_LABEL_LAB.ipynb", LAB, "lab")
    sys.exit(0)
