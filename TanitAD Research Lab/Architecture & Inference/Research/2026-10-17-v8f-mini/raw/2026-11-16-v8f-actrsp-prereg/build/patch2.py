import sys
p=sys.argv[1]
s=open(p,encoding='utf-8',newline='').read()
old="""        def wrapped(level, z0, actions, *, context=1):
            flag["lv"] = level
"""
new="""        def wrapped(level, z0, actions, *, context=1):
            if actions.dim() == 5:                                                   # PREREG_V8F_ACTRSP: the action-response term's own call (actions (contrast, B, anchor, K, A)) is not the ladder's free run: not counted, not recorded (:meth:`built_rsp` reads it)
                return orig(level, z0, actions, context=context)
            flag["lv"] = level
"""
assert s.count(old)==1
s=s.replace(old,new)
open(p,'w',encoding='utf-8',newline='').write(s)
