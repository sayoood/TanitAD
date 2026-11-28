p='F:/TanitAD-scratch/v8proof/paper/v9/frag_level.tex'
t=open(p,encoding='utf8').read()
t=t.replace('{\textwidth}', '{'+chr(92)+'textwidth}')
open(p,'w',encoding='utf8',newline='').write(t)
print(t.count(chr(9)))
