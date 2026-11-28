"""Insert a keyword into matching calls (text-level, from AST end positions). Usage: inject_kw.py file name kw value [recv_filter]"""
import ast, sys
def inject(path, callee, kw, value, cond):
    raw = open(path, 'rb').read().decode('utf-8')
    assert '\r' not in raw
    tree = ast.parse(raw)
    lines = raw.split('\n')
    # absolute offsets in characters
    starts = [0]
    for ln in lines[:-1]:
        starts.append(starts[-1] + len(ln) + 1)
    def abs_pos(lineno, col_bytes):
        l = lines[lineno - 1]
        col = len(l.encode('utf-8')[:col_bytes].decode('utf-8'))
        return starts[lineno - 1] + col
    edits = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call): continue
        f = n.func
        nm = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
        if nm != callee: continue
        kws = {k.arg for k in n.keywords}
        if kw in kws or not cond(n, kws): continue
        close = abs_pos(n.end_lineno, n.end_col_offset) - 1
        assert raw[close] == ')', (path, n.lineno, raw[close])
        q = close - 1
        while q >= 0 and raw[q] in ' \t\n': q -= 1
        c = raw[q]
        gap = raw[q + 1:close]
        item = f'{kw}={value!r}'.replace("'", '"')
        if c == ',':
            ins = ('' if gap else ' ') + item
        elif c == '(':
            ins = item
        else:
            ins = ', ' + item if '\n' not in gap else ', ' + item
        edits.append((close, ins))
    for pos, ins in sorted(edits, reverse=True):
        raw = raw[:pos] + ins + raw[pos:]
    ast.parse(raw)
    open(path, 'w', encoding='utf-8', newline='').write(raw)
    return len(edits)
if __name__ == '__main__':
    path, callee, kw, value = sys.argv[1:5]
    mode = sys.argv[5] if len(sys.argv) > 5 else 'any'
    conds = {'any': lambda n, k: True,
             'v0': lambda n, k: 'v0' in k,
             'ctx': lambda n, k: isinstance(n.func, ast.Attribute) and ast.unparse(n.func.value) == 'ctx'}
    print(path, callee, inject(path, callee, kw, value, conds[mode]))
