"""Unit test (plain unittest, no navsim needed) of the --metric-cache-root fix in transp_score.py:
flag parses; default (empty) still routes to PT.metric_cache_parts; a missing cache file raises naming the path (no DB fallback).
Source-level checks, because main() needs navsim + the nuPlan DB; the real end-to-end check is check_cli_cachefix.py."""
import ast, os, re, tempfile, unittest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transp_score.py")


def _load():
    text = open(SRC, encoding="utf-8").read()
    return text, ast.parse(text)


def _fn(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


class CacheFlag(unittest.TestCase):
    def test_flag_declared_default_empty(self):
        _, tree = _load()
        hit = [c for c in ast.walk(_fn(tree, "main")) if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "add_argument"
               and c.args and getattr(c.args[0], "value", None) == "--metric-cache-root"]
        self.assertEqual(len(hit), 1)
        kw = {k.arg: k.value for k in hit[0].keywords}
        self.assertEqual(kw["default"].value, "")

    def test_default_route_still_metric_cache_parts(self):
        text, _ = _load()
        m = re.findall(r"parts = load_cache_parts\(a\.metric_cache_root, lg, tok\) if a\.metric_cache_root else PT\.metric_cache_parts\(sc\)", text)
        self.assertEqual(len(m), 1)
        self.assertEqual(text.count("PT.metric_cache_parts("), 1)      # no other DB route, and it is the else-branch of the flag

    def test_missing_cache_raises_naming_path(self):
        _, tree = _load()
        ns = {"os": os}
        fn = _fn(tree, "load_cache_parts")
        exec(compile(ast.Module(body=[fn], type_ignores=[]), SRC, "exec"), ns)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError) as cm:
                ns["load_cache_parts"](d, "LOGX", "tok123")
            exp = os.path.join(d, "LOGX", "unknown", "tok123", "metric_cache.pkl")
            self.assertIn(exp, str(cm.exception))


if __name__ == "__main__":
    unittest.main()
