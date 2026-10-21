import json, sys
s = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))['box']['refcv7']
cells = s['ap_auroc_cells']
A = s['arms']['a']
pairs = [('ap0p5_all','eval_box3d_det_ap0p5_all_all'),('ap1_all','eval_box3d_det_ap1_all_all'),('ap2_all','eval_box3d_det_ap2_all_all'),('ap4_all','eval_box3d_det_ap4_all_all'),
 ('map0p5','eval_box3d_det_map0p5_all'),('map1','eval_box3d_det_map1_all'),('map2','eval_box3d_det_map2_all'),('map4','eval_box3d_det_map4_all'),
 ('ap2_band0','eval_box3d_det_ap2_all_0_20'),('ap2_band1','eval_box3d_det_ap2_all_20_40'),('ap2_band2','eval_box3d_det_ap2_all_40_60'),
 ('auroc_matched','eval_box3d_auroc_matched'),('auroc_objectness','eval_box3d_auroc_objectness')]
mx = 0
for k, kb in pairs:
    m = cells[k][A]['mean']; r = b[kb]; mx = max(mx, abs(m-r))
    print(f"{k:18s} mine {m:.9f} banked {r:.9f} diff {abs(m-r):.2e}  CI [{cells[k][A]['lo']:.4f},{cells[k][A]['hi']:.4f}]")
print('max abs diff', mx)
cls = s['classes']
for c in range(10):
    print(cls[c], cells[f'ap2_cls{c}'][A]['mean'], b[f'eval_box3d_det_ap2_{cls[c]}_all'])
v = s['census_variants']['declared_gate_0.5']['cells']
print('rec@0.5', v['rec'][A]['mean'], b['eval_box3d_rec@gate'], 'prec', v['prec'][A]['mean'], b['eval_box3d_prec@gate'], 'cr', v['conf_ratio'][A]['mean'], b['eval_box3d_conf_ratio'], 'accTP', v['cls_acc_tp'][A]['mean'], b['eval_box3d_cls_acc_tp'])
print('centre err p50', s['centre_err_p50_matched'], b['eval_box3d_centre_err_p50'])
print({k:v['pass'] for k,v in s['controls'].items()})
print(s['census_variants']['declared_gate_0.5']['recall_by_class_at_gate'][A])
