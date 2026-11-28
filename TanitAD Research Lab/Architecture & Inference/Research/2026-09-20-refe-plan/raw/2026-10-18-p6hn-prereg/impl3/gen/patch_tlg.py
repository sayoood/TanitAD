p = "test_launch_gate.py"
s = open(p, encoding="utf-8").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)
rep('''        if k.startswith(("G8_", "G12_", "G13_")):                  # r0 carries no lane labels / no --aux-drv / no --epo-con: G8 / G12 / G13 are not obliged, so such a flag in the fixture is simply not written''',
    '''        if k.startswith(("G8_", "G12_", "G13_", "G14_", "G15_")):                  # r0 carries no lane labels / no --aux-drv / no --epo-con / no --hn-bank / no --ctx-cache: G8 / G12 / G13 / G14 / G15 are not obliged, so such a flag in the fixture is simply not written''')
blk = open("C:/Users/Admin/hn34/work/hn_static_block.py", encoding="utf-8").read()
rep('''def full() -> None:''', blk + '''

def full() -> None:''')
rep('''    aux_s1_static()
    field_static()''', '''    aux_s1_static()
    hnscreen_static()
    field_static()''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched", len(s))
