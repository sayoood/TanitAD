"""SCALE TEST on the complete first-launch causal-s1 ticks file: controls K2 / K3 / X1 and timing ONLY. No outcome row is computed or printed."""
import sys, time, json
sys.path.insert(0, 'C:/Users/Admin/t1hdread')
SC = 'C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/sfresh'
W = 'E:/Projects/TanitAD-artifacts/v8f-t1heads-run/tool_81aa3e36'
PK = 'TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw'
import t1hd_read as R
a = R.parse(['--mode', 'real', '--tool-root', W, '--eval-cache', SC + '/cache', '--tracks', SC + '/inputs/native_tracks.pt', '--eval-grid', f'{W}/{PK}/eval_grid_windows.json',
             '--eval-join', f'{W}/{PK}/join_eval141.npz', '--banked-dir', 'C:/Users/Admin/t1hdread/banked', '--out', 'x.json'])
R.setup_imports(a.tool_root)
t = time.time(); ctx = R.Ctx(a); print('ctx', round(time.time() - t), 's')
tag = 'HACC_s1_m10_causal_heads'
tt = R.load_ticks(f'C:/Users/Admin/t1hd_run/ticks/ticks_{tag}.npz')
t = time.time(); Rr = R.analyze_run(ctx, tt, a, 'scale', 2000, f'C:/Users/Admin/t1hd_run/dump_{tag}'); print('analyze_run', round(time.time() - t), 's for', tt['N'], 'windows')
print('N == grid:', tt['N'] == len(ctx.wins_grid), [(int(c), int(t0)) for c, t0 in zip(tt['ci'], tt['t0'])] == [(int(w[0]), int(w[1])) for w in ctx.wins_grid])
print('K2', json.dumps(R.k2_frames(tt)))
print('K3', json.dumps(R.k3_tick0(Rr, tt)))
print('X1', json.dumps(Rr['x1']))
print('kappa frames min', int(Rr['kappa_frames'].min()), 'zero-frame ticks', int((Rr['kappa_frames'] == 0).sum()))
