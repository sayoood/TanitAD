import json,sys
ref=json.load(open('F:/TanitAD-scratch/rollhist/read/out/read/hw/RESULT_HEADWAY.json'))['pairs']
new=json.load(open(sys.argv[1]+'/RESULT_HEADWAY.json'))['pairs']
ok=True;out={}
for t in ['I10RH_R0KHb20_minus_I10B0_R0KHb20','I10RH_R0C10XHb20_minus_I10B0_R0C10XHb20']:
    for cell in ['U','C','F']:
        r=ref[t]['cells'][cell];n=new[t]['cells'][cell]
        for k in ['D','A_lo','A_hi','d_s']:
            same=(r[k]==n[k]); ok&=same
            out[t+'/'+cell+'/'+k]=(r[k],n[k],same)
print(ok)
for k,v in out.items():
    if '/U/' in k: print(k,v)
json.dump({'pass':ok,'cmp':out},open('R0_CHECK.json','w'),indent=1)
