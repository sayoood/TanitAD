import pathlib
P = pathlib.Path("C:/Users/Admin/dagread/scratch/direct_run.py")
s = P.read_text(encoding="utf-8")
old = s[s.index("class Cap:"):s.index("def call(")]
new = '''class Cap:
    def __init__(self):
        self.buf, self.ebuf, self.ctx, self.ctx2 = io.StringIO(), io.StringIO(), None, None
    def start(self):
        self.ctx = contextlib.redirect_stdout(self.buf); self.ctx.__enter__()
        self.ctx2 = contextlib.redirect_stderr(self.ebuf); self.ctx2.__enter__()
    def stop(self):
        if self.ctx2: self.ctx2.__exit__(None, None, None); self.ctx2 = None
        if self.ctx: self.ctx.__exit__(None, None, None); self.ctx = None
    def readouterr(self):
        v, e = self.buf.getvalue(), self.ebuf.getvalue()
        for b in (self.buf, self.ebuf): b.seek(0); b.truncate()
        return type("R", (), {"out": v, "err": e})()


'''
s = s.replace(old, new)
P.write_text(s, encoding="utf-8", newline="\n")
