import json, os, sys, csv, collections, hashlib, time
sys.path.insert(0, 'C:/Users/Admin/par18/s0probe')
from common_s0 import *
W = 'C:/Users/Admin/par18/s0probe/'
logs, tok_log, by = s0()
rows = {r['log']: r for r in csv.DictReader(open(SR + '/run/plan/ranked_fit_logs.tsv'), delimiter='\t')}
raw = json.load(open(W + 'probe_s0_raw.json'))['logs']
ctx = json.load(open(W + 'ctx_cov.json'))
tgz = json.load(open(W + 'tgz_cov.json'))
ver = json.load(open(W + 'verify_bytes.json'))
bankv = json.load(open(W + 'verify_bank.json'))
bankm = json.load(open(W + 'verify_bank_manifest.json'))
mc = json.load(open(W + 'mc_overlap.json'))
tokscan = json.load(open(W + 'token_scan.json'))
roots13 = json.load(open(W + 'roots13.json'))
plans = json.load(open(W + 'plans_cov.json'))
logdirs = json.load(open(W + 'logdir_overlap.json'))
ctx_logs = collections.Counter(tok_log[t] for t in ctx['tokens_done'])
# pod shard + staged bytes per log (S0 stage manifests: measured pod file sizes)
shard = {}
B = SR + '/run/batches'
for u in os.listdir(B):
    p = os.path.join(B, u, 'copy_list_frames.tsv')
    if os.path.exists(p):
        for l in open(p, encoding='utf-8'):
            q = l.rstrip('\n').split('\t')
            if len(q) >= 3:
                shard.setdefault(q[2].split('/')[0], set()).add(q[1].rsplit('/', 1)[-1])
SZ = collections.defaultdict(int)
SN = collections.defaultdict(int)
for u in os.listdir(B):
    p = os.path.join(B, u, 'copy_manifest_frames.tsv')
    if os.path.exists(p):
        for l in open(p, encoding='utf-8'):
            q = l.rstrip('\n').split('\t')
            if len(q) >= 2 and q[1].isdigit():
                SZ[q[0].split('/')[0]] += int(q[1])
                SN[q[0].split('/')[0]] += 1
perlog = []
gap_logs = []
for lg in logs:
    r = raw[lg]
    nt = r['n_tokens']
    res = r.get('tokens_all4cams_resolved', 0)
    d0 = r['db_paths'][0] if r['db_paths'] else None
    rec = {'log': lg, 'rank': int(rows[lg]['rank']), 'fold': int(rows[lg]['fold']), 'n_tokens': nt, 'n_sets': int(rows[lg]['n_fit_sets']),
           'db_local': bool(d0), 'db_path': d0['path'] if d0 else None, 'db_bytes': d0['bytes'] if d0 else None, 'db_tokens_found': r.get('db_tokens_found'), 'map_version': r.get('map_version'),
           'tokens_with_4cams_local': res, 'frames_complete': res == nt, 'frames_via': r.get('tokens_via_root', {}), 'needed_frames': r.get('n_needed_frames'),
           'ctx_tokens_cached': ctx_logs.get(lg, 0), 'pod_pixel_shard': sorted(shard.get(lg, [])), 'pod_frames_bytes_S0_manifest': SZ.get(lg), 'pod_frames_n_S0_manifest': SN.get(lg),
           'in_openscene_probe_tgz25': lg in set(tgz['S0_logs_in_shard'])}
    perlog.append(rec)
    if res < nt:
        gap_logs.append(rec)
miss_tok = sum(g['n_tokens'] - g['tokens_with_4cams_local'] for g in gap_logs)
miss_frames_known = sum(raw[g['log']]['n_missing_frames'] for g in gap_logs)
miss_bytes_measured = sum(raw[g['log']]['missing_frames_bytes_known'] for g in gap_logs)
miss_known_n = sum(raw[g['log']]['missing_frames_with_known_size'] for g in gap_logs)
# estimate from present logs (the 1,966 tokens staged by sh5, sizes from the sha list)
ff_bytes = ver['frames_full']['listed_bytes']
ff_tokens = ver['frames_full']['listed'] // 4
per_tok = ff_bytes / ff_tokens
est_frames_bytes = per_tok * miss_tok
# tgz25 recoverable
tgz_missing = tgz['S0_missing_logs_in_shard']
tgz_tokens = sum(len(by[l]) for l in tgz_missing)
tgz_bytes = sum(SZ.get(l, 0) for l in tgz_missing)
ctx_mib = 3.75
out = {
    'what': 'S0 search-bank inputs: where each input for the 117 P2-S0 logs lives (probe of 2026-10-20, read-only, CPU)',
    'mirror_tip_sha': open(W + 'SHA.txt').read().strip(),
    'S0': {'logs': 117, 'tokens': 8922, 'sets': 15808, 'ranked_fit_logs_sha256': 'a61275037a3d2b06903dc7e1ad0f6f99fb740f668a6ec7488fdef6fe901ce9b1',
           'ranked_fit_logs_path': SR + '/run/plan/ranked_fit_logs.tsv', 'master_plan_path': SR + '/run/plan/master_plan.json',
           'selection': 'first_fraction<=0.10 rows of ranked_fit_logs.tsv (1,167 rows total; 117 selected)'},
    'builder_claim_checked': {'claim': 'dev box has DB and frames for only 22 of 117 S0 logs, and no plan json or labelled P2 fan on the S0 tokens',
                              'verdict': {'DB': 'FALSE NOW: 117 of 117 DBs local (22/25 was the pre-S0 availability.json snapshot of 2026-10-18; the same 76 DBs S0 had staged and deleted were re-staged 2026-10-19 by the pod-stage stream (par18/podstage) and now sit in splits/)',
                                          'frames': 'PARTLY TRUE: 41 of 117 logs / 2,782 of 8,922 tokens have all 4 cameras locally (not 22); 76 logs / 6,140 tokens have none',
                                          'plan_json': 'FALSE: master_plan.json + 23 per-unit batch_tokens.json (8,922 tokens) + sh5 plan_full (2,782) exist',
                                          'labelled_P2_fan': 'TRUE for P2 own-fan trajectories+labels (not banked, --lean). FALSE as stated: P2 own-fan logits+pick for 8,922 tokens and P2 logits+labels on 15,808 stored fans ARE banked',
                                          'source_of_22': 'srttc_s0/run/plan/NOTES.md and plan/availability.json (pre-S0 snapshot: 25 logs with DB, 38 with 4 zips, 22 with both)'}},
    'inputs': {
        'a_frames_4cams': {'logs_complete': 41, 'tokens_complete': 2782, 'logs_none': 76, 'tokens_none': 6140, 'logs_partial': 0,
                           'tokens_via_root_for_complete_logs': 'sh5_ctx/frames_full 1,954; refe_proxy/pixels 714; refe_m5/pixels 102; nuplan-camera/mini 90; nuplan-camera/scenarios4 12 (a token can draw cameras from >1 root)',
                           'resolver': 'planner.FrameResolver semantics re-implemented read-only: nearest image row per camera within 120 ms, jpg basename must be present in a root',
                           'roots_with_usable_S0_frames': {
                               'D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19/frames_full': {'logs': 41, 'files': 7864, 'bytes': ff_bytes, 'verified': ver['frames_full']},
                               'D:/Projects/TanitAD/data/refe_proxy/pixels': {'logs_with_zips': 38, 'note': 'zip per (log, camera); a stratified sample of frames'},
                               'D:/Projects/TanitAD/data/refe_m5/pixels': {'logs': 15}, 'D:/Projects/TanitAD/data/nuplan-camera/mini': {'logs': 3, 'note': 'CAM_F0 zip only per log'},
                               'D:/Projects/TanitAD/data/nuplan-camera/scenarios4': {'logs': 1}},
                           'roots_with_only_fragments': {'D:/TanitAstraResearch/data/e024_* (anchors, sequence stages, report images), e023_sequences': 'a few jpgs for 12-15 S0 logs; 573 frames of the 24,560 needed by the 76 missing logs; no complete token; anchors jpgs byte-identical to proxy zips (66/66)'},
                           'roots_excluded': {'_rlfan_work/atol_cal{,2}/pix_{jpeg,hflip,bgr_rgb}': 'calibration variants (re-encoded/flipped/BGR), not original frames, 2-4 S0 logs each', 'srttc_s0_stage/frames': '0 files (deleted after banking)'},
                           'openscene_probe_tgz': {'path': 'D:/Projects/TanitAD/data/openscene_probe/navtrain_current_25.tgz', 'bytes': 4496159137, 'S0_logs_in_shard': tgz['S0_logs_in_shard'],
                                                   'S0_missing_logs_in_shard': tgz_missing, 'tokens_recoverable_locally': tgz_tokens, 'frames_needed_present': tgz['present_total'], 'frames_needed': tgz['needed_total'],
                                                   'note': 'gzip tar, sequential extract only; all frames the 2 gap logs need are listed in members.txt'}},
        'b_db': {'logs_local': 117, 'logs_missing': 0, 'bytes_local_total': sum(p['db_bytes'] for p in perlog),
                 'tokens_found_in_lidar_pc': sum(p['db_tokens_found'] for p in perlog),
                 'locations': {'D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits (flat)': 99 + 2 - 0, 'D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval': 15 + 1,
                               'extra copies': 'dblinks/driverl_val14 (3 logs, link copies)'},
                 'restaged_76_by_podstage_stream': ver['dbs_76'], 'maps': 'D:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0 (4 cities, map.gpkg 186-524 MB each) + nuplan-maps-v1.1.zip',
                 'map_versions_of_S0_logs': dict(collections.Counter(p['map_version'] for p in perlog)),
                 'navsim_metric_cache_local': {'navtrain_S0_tokens_covered': 0, 'other_caches_found': mc,
                                               'note': 'no local navtrain metric_cache; the repo labeller (label_stepB.py -> label_perturbations.label_plans -> navsim_pdm_targets.metric_cache_parts) rebuilds the cache from DB + maps, so DB + maps suffice'}},
        'c_p2_bank': {'bank_dir': 'C:/Users/Admin/srttc_s0/work/bank', 'files': 69, 'units': 23, 'manifest_verified': bankm, 'fan_tokens': bankv['fan_tokens'], 'sets': bankv['sets_rows'], 'logs': bankv['bank_logs'],
                      'sets.npz': 'log, token, step, rank, ckpt_step, feat[S,64,256]fp16 (score_head input), logits[S,64,6] (P2 scorer on the STORED fan), top8, labels6[S,64,6] (NC,DAC,EP,TTC,C,DDC); NO trajectories',
                      'fan.npz': 'token, fingerprint, logits[n,64,6] and pick[n] of P2 OWN fan for 8,922 tokens; proposals/proposals20 are EMPTY (--lean); NO labels',
                      'stored_fan_trajectories': {'path': SR + '/work/sets/<log>.npz', 'stores': 117, 'sets': bankv['work_sets_rows'], 'joined_to_bank_by_key': bankv['work_sets_joined_to_bank'], 'traj3': '[S,64,20,3]',
                                                  'note': 'these are on-policy fans of EARLIER checkpoints (1,281 step/rank/ckpt_step combos), NOT P2 own fan; single copy on C:'},
                      'pod_text_copies_of_stored_sets_and_labels': {'sets_pod/sets/onpolicy_selected.jsonl': 16946, 'sets_pod/pdm/pdm_selected.jsonl': 15808, 'path': SR + '/work/sets_pod'},
                      'derived_copy': 'D:/Projects/TanitAD-artifacts/headrow_probe_2026-10-19/oof_proposals_f10.npz (15,808 sets: P2 logits, labels6, top8, pick, OOF head rows; no trajectories)',
                      'used_by': ['EP-RELIAB-PICKS', 'EP-ADMISSION-R1/R2 (ep_admission.py --p2-bank)', 'P6-READ-PREP (READY_P6_READ.md)', 'PICKRULE-AUDIT-NC', 'HEADROW-PROBE', 'FASTNEG-CENSUS'],
                      'p2_checkpoint': {'path': 'D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt', 'bytes': bankv['ckpt_bytes'], 'sha256': bankv['ckpt_sha256'], 'matches_S0_pin_ea31db2296210b6a': bankv['ckpt_sha256'].startswith('ea31db2296210b6a')},
                      'not_found_anywhere': 'labelled P2-own-fan trajectories on S0 tokens; scanned: p2_full_table (12,146 tokens, 0 in S0), p2_heldout_table_full (0), p2_zero_tail_table (0), dgdump (4 tokens, not S0 tokens)',
                      'token_scan': tokscan},
        'd_ctx_cache': {'path': 'D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19/ctx_local/ctx.npy', 'shape': ctx['ctx_shape'], 'dtype': ctx['ctx_dtype'], 'tokens': ctx['n_manifest_tokens'], 'tokens_done': ctx['n_done'],
                        'logs': ctx['n_logs_done'], 'ckpt': 'p2/model_final.pt', 'c_fid': 'frac_within_tol 1.0 on 1,444 sets (maxabs 0.0)', 'bytes': 3208642688, 'ctx_sha256': 'f758cb47b720041084e632e4a9612fc94d1b337a285cf20c045e7dc3c11da212',
                        'tokens_without_ctx': 8922 - ctx['n_done'], 'other_caches_checked': {'refe_trunk_cache/d7c59f4f (epoch-15 snapshot, 1,600+300 navtrain keys, NOT P2 final)': 'wrong checkpoint; not usable',
                                                                                           'l2_offprop/feats.npz': '400 Step-A tokens, 0 in S0', 'p2_heldout_table*': '0 in S0'}},
        'plan_json': {'master_plan': SR + '/run/plan/master_plan.json', 'per_unit_plans': SR + '/run/batches/<unit>/batch_tokens.json (23 units, union = 8,922 tokens)', 'sh5_plan_full': 'D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19/plan_full/plan.json (2,782 tokens, current token_root)',
                      'caveat': 'token_root and db inside the per-unit plans point at the deleted S0 staging dirs; only sh5 plan_full carries current roots', 'units': plans['units']},
    },
    'probes_run': [
        'P01 srttc_s0/run/plan/availability.json + NOTES.md (source of the "22"): pre-S0 snapshot',
        'P02 find *.db named like nuPlan logs under D:/Projects, D:/tanitad, D:/hold_temporal, D:/TanitAstra*, D:/Archive, D:/thor_archive, D:/qland_work_archive_sept, D:/tmp_fixtree, D:/lgs, D:/t6, D:/refcv*_kit and C:/Users/Admin: 3,130 DBs, all 117 S0 logs found',
        'P03 sqlite open (read-only) of each S0 DB, 8,922/8,922 tokens found in lidar_pc; camera/image tables read; 76 re-staged DBs size-equal, 12 smallest sha256-equal, all 76 size+sha256-equal to the S0 stage manifests',
        'P04 13 local frame roots (READY.md sec I list) name-scanned to depth 3 for S0 log names (roots13.json): hits only in refe_proxy/pixels (38 logs), refe_m5/pixels (15), nuplan-camera/mini (3), scenarios4 (1); 0 in scenarios, probe, refe_navtrain, refe_navtrain10, refe_navtest, refe_m5r, refe_consistent, heldout_frames, videos',
        'P05 zip/dir membership resolution of every S0 token x 4 cameras against 18 roots incl. sh5 frames_full and Astra image dirs',
        'P06 find log-named dirs and *_CAM_*.zip on D: roots and C:/Users/Admin (5,080 hits) grouped by parent: found sh5 frames_full (41/41 S0), Astra e023/e024 dirs (fragments), _rlfan_work calibration copies',
        'P07 sha256: 400 random frames_full jpgs vs frames_full_sha256.txt (0 mismatch); all 7,864 present with equal size',
        'P08 openscene_probe/members.txt: shard 25 holds 4 S0 logs, 2 of them gap logs, every needed frame listed',
        'P09 metric cache search by name on D: and C:/Users/Admin: Astra caches (5 dirs), devbox-C warmup, navsim-crun navtest/navhard: S0 token overlap 22/3/4/0/0 (Astra e024/e025/e026/e023/e031), 0 S0 logs elsewhere; no *.pkl.xz navtrain cache',
        'P10 bank: 69 files sha256 vs MANIFEST (0 bad); 8,922 fan tokens, 15,808 sets, 117 logs; proposals arrays empty in all 23 fan.npz; work/sets 117 stores / 15,808 sets all joined by key',
        'P11 token scan of p2_full_table, p2_heldout_table_full, p2_zero_tail_table, dgdump, headrow oof, sh5 inputs, l2_offprop feats for S0 tokens',
        'P12 sh5 ctx_local: 816 tokens (38 logs) all done, C-FID pass; refe_trunk_cache is the epoch-15 snapshot (wrong ckpt)',
        'P13 repo text at the mirror tip: pod paths (git grep navtrain_pixels / navtrain_dbs / onpolicy/sets / train_pdm / metric_cache)',
        'NOT DONE: ssh to any pod (forbidden), GPU use, any download; pod-side existence of the 76 logs was NOT re-checked (INHERITED from the S0 stage manifests, 2026-10-18)'],
    'gap': {
        'frames': {'logs_missing': len(gap_logs), 'tokens_missing': miss_tok, 'frames_needed_4cam': 4 * miss_tok, 'frames_absent_locally': miss_frames_known,
                   'bytes_MEASURED_pod_file_sizes_from_S0_stage_manifests': miss_bytes_measured, 'frames_with_known_size': miss_known_n,
                   'bytes_ESTIMATED_from_present_logs': round(est_frames_bytes), 'estimate_basis': f'{ff_bytes} B / {ff_tokens} tokens = {per_tok:.0f} B per token over the 41 present sh5 logs (1,966 staged tokens), x {miss_tok} tokens',
                   'recoverable_without_network': {'logs': tgz_missing, 'tokens': tgz_tokens, 'bytes': tgz_bytes, 'how': 'sequential gunzip of openscene_probe/navtrain_current_25.tgz (4.5 GB), extract 548 jpgs'},
                   'pod_source_paths': '/workspace/data/navtrain_pixels/navtrain_current_N/<log>/<CAM>/<hash>.jpg (N per log in per_log.pod_pixel_shard)'},
        'db': {'logs_missing': 0, 'bytes': 0, 'if_lost_refetch_bytes': ver['dbs_76']['listed_bytes'], 'pod_path': '/workspace/data/navtrain_dbs/<log>.db'},
        'maps': {'missing': 0},
        'p2_own_fan_with_labels': {'exists_for_tokens': 0, 'of': 8922, 'to_create': 'one P2 forward per token (needs frames or ctx+decoder) + PDM labelling of 64 plans per token (DB + maps); S0 dump rate MEASURED 14,173 s for 8,922 tokens (3.94 h GPU, build_candidate_bank.md) => ~1.2 h for the 2,782 tokens that have frames (ESTIMATED, scaling)'},
        'ctx_cache': {'tokens_missing': 8922 - ctx['n_done'], 'bytes_ESTIMATED_if_built': round((8922 - ctx['n_done']) * ctx_mib * 1024 * 1024), 'basis': '3.75 MiB per token (ctx.npy 3,208,642,688 B / 816 tokens + header)', 'kind': 'DERIVED by a GPU forward, not fetchable'},
        'disk_free_now': {'C:': '4.9 GB', 'D:': '0 GB (exFAT, full)', 'consequence': 'the 4.9 GB frame gap does not fit on D:; C: has ~4.9 GB'}},
    'pod_paths_from_repo_text': {
        'frames': '/workspace/data/navtrain_pixels/navtrain_current_N/<log>/<CAM>/<hash>.jpg', 'dbs': '/workspace/data/navtrain_dbs/<log>.db', 'stored_on_policy_sets': '/workspace/data/refe_navtrain/onpolicy/sets',
        'pdm_labels_of_sets': '/workspace/data/refe_sft2/train_pdm (pdm_onpolicy_*.jsonl)', 'navsim_metric_cache': '/workspace/data/refe_navtrain/metric_cache/<log>/<token>.pkl.xz',
        'source_in_repo': 'srttc_s0 code (srttc_s0_stage.py, srttc_s0_podsets.py) staged at raw/2026-10-18-sel-rank/ and raw/2026-10-16-p6-d7-pdm-route-rank/HANDOFF.md, p6_pod.sh:32'},
    'per_log': perlog,
}
out['inputs']['plan_json']['in_repo_tip'] = {'dir': 'TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-sel-rank/raw', 'master_plan.json': 's0/plan/master_plan.json (sha256 c3590a81..., equal to local)',
    'ranked_fit_logs.tsv': 'a2_subsample/ranked_fit_logs.tsv (sha256 a6127503..., equal to local)', 'also': 'S0_SUMMARY.json, 23 batch audit.json, 19 copy_manifest_{frames,dbs}.tsv, availability.json (the pre-S0 snapshot behind the 22), NOTES.md',
    'NOT_in_repo': 'bank *.sets.npz / *.fan.npz, work/sets/*.npz, batch_tokens.json, sh5 ctx.npy, frames_full'}
out['inputs']['c_p2_bank']['copies'] = 'single copy on C: (S0_HANDOFF.md); no *.npz of the bank in the repo or on D: (name search for 3 unit ids on D:/Projects, D:/tanitad, D:/Archive, C:/Users/Admin found only batch dirs)'
json.dump(out, open(W + 's0_inputs.json', 'w'), indent=1)
print('wrote s0_inputs.json', os.path.getsize(W + 's0_inputs.json'))
print('gap logs', len(gap_logs), 'missing tokens', miss_tok, 'frames absent', miss_frames_known, 'bytes measured', miss_bytes_measured, 'est', round(est_frames_bytes), 'tgz recover tokens', tgz_tokens, tgz_bytes, tgz_missing)
sh = collections.Counter(s for g in gap_logs for s in g['pod_pixel_shard'])
print('pod shards of gap logs', len(sh), sorted(sh.items())[:40])
print('db local bytes', out['inputs']['b_db']['bytes_local_total'], 'map versions', out['inputs']['b_db']['map_versions_of_S0_logs'])
