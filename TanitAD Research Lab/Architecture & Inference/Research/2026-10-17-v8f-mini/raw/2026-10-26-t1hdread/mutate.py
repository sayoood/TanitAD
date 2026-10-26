"""Deliberate-regression checks OF THE READER'S OWN CONTROLS (smoke data): each mutation must turn its control RED. Not part of the PREREG."""
import sys, os, shutil, glob, copy, json
import numpy as np
SC = 'C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/sfresh'
W = 'E:/Projects/TanitAD-artifacts/v8f-t1heads-run/tool_81aa3e36'
PK = 'TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw'
argv = ['--mode', 'smoke', '--tool-root', W, '--eval-cache', SC + '/cache', '--tracks', SC + '/inputs/native_tracks.pt', '--eval-grid', f'{W}/{PK}/eval_grid_windows.json',
        '--eval-join', f'{W}/{PK}/join_eval141.npz', '--banked-dir', 'C:/Users/Admin/t1hdread/banked', '--out', 'x.json']
sys.path.insert(0, 'C:/Users/Admin/t1hdread')
import t1hd_read as R
a = R.parse(argv); R.setup_imports(a.tool_root)
ctx = R.Ctx(a)
TK, DD, REF = 'C:/Users/Admin/t1heads/smoke/ticks/ticks_SM.npz', 'C:/Users/Admin/t1heads/smoke/heads/dump_SM', 'C:/Users/Admin/t1heads/smoke/off/dump_SM'
tt = R.load_ticks(TK)
out = {}
# baseline: all green
base = R.analyze_run(ctx, tt, a, 'base', 200, DD)
out['baseline'] = {'K2': R.k2_frames(tt)['PASS'], 'K3': R.k3_tick0(base, tt)['PASS'], 'K7': R.k7_bit_exact(DD, 'banked', 's0', REF)['PASS'],
                   'gt_frame_median_tick1_m': base['gt_frame']['tick1']['median_m']}
# K2: the dumped window-origin checkpoints moved by 2e-4 m
m = copy.deepcopy(tt); m['s_ckpt_w0'] = m['s_ckpt_w0'] + np.float32(2e-4)
out['K2_mutant_s_ckpt_w0_plus_2e-4'] = R.k2_frames(m)['PASS']
# K3: tick-0 s_pose moved by 1e-3 m
m = copy.deepcopy(tt); m['s_pose'][:, 0, 0] += np.float32(1e-3)
out['K3_mutant_s_pose0_plus_1e-3'] = R.k3_tick0(R.analyze_run(ctx, m, a, 'm', 200, DD), m)['PASS']
# K7: one value of one episode's cl flipped in a COPY of the dump dir
tmp = 'C:/Users/Admin/t1hdread/mut_tmp/dump'
shutil.rmtree('C:/Users/Admin/t1hdread/mut_tmp', ignore_errors=True); shutil.copytree(DD, tmp)
f = sorted(glob.glob(tmp + '/ep*.npz'))[3]; z = dict(np.load(f)); z['cl'] = z['cl'].copy(); z['cl'][0, 5, 0] = np.nextafter(z['cl'][0, 5, 0], np.inf); np.savez(f, **z)
out['K7_mutant_one_ulp_in_cl'] = R.k7_bit_exact(tmp, 'banked', 's0', REF)['PASS']
shutil.rmtree('C:/Users/Admin/t1hdread/mut_tmp', ignore_errors=True)
# K1: an estimator that does not return exactly 0 for (a, a)
orig = R.PB
R.PAIRS.append(('x', np.arange(10.0), np.arange(10.0), np.arange(10), 'mean'))
R.PB = lambda a_, b_, eid, **kw: dict(orig(a_, b_, eid, **kw), delta=1e-4)
out['K1_mutant_delta_1e-4'] = R.k1_identity(50)['PASS']
R.PB = orig; R.PAIRS.clear()
# GT frame: the label read as if it were already in the window-origin frame (no transform from r' to t0)
orig_to = R.to_w0
R.to_w0 = lambda ck, src, dst: orig_to(ck, dst if np.asarray(ck).ndim == 4 else src, dst)
mm = R.analyze_run(ctx, tt, a, 'm', 200, DD)
R.to_w0 = orig_to
out['GT_frame_mutant_no_transform'] = {'median_tick1_m': mm['gt_frame']['tick1']['median_m'], 'median_tick2_m': mm['gt_frame']['tick2']['median_m'],
                                       'baseline_median_tick1_m': base['gt_frame']['tick1']['median_m'], 'baseline_median_tick2_m': base['gt_frame']['tick2']['median_m']}
# CTRV curvature: sign flipped must move the tick-0 CTRV error on a turning window (the comparator is wired to the executed track)
pe0 = np.nanmean(base['err']['CTRV']['pos_err_m'][:, 0, 1])
kap = base['kappa']
ctrv_f, _, _ = R.ctrv_straight(-kap, tt['s_pose'].astype(np.float64), tt['pose_w0'])
ef = R.C.checkpoint_errors(ctrv_f, base['gt_w0'], base['gt_ok'])['pos_err_m']
out['CTRV_kappa_sign_flip_tick0_pos50'] = {'baseline': float(pe0), 'mutant': float(np.nanmean(ef[:, 0, 1]))}
json.dump(out, open('C:/Users/Admin/t1hdread/MUTATION_CHECKS.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
