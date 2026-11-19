import io
p="F:/TanitAD-scratch/v8proof/imagplan_q/pod_mut_q.sh"
s=io.open(p,encoding="utf-8",newline="").read()
o="  all) ARGS="
assert s.count(o)==1
s=s.replace(o,'  after) ARGS="--control-only"; OUT=$D/pod_out_qplan.json;;   # control after the mutations, into the record of the full run\n'+o)
s=s.replace("<control-only|all|ID[,ID]>","<control-only|all|after|ID[,ID]>")
io.open(p,"w",encoding="utf-8",newline="").write(s)
