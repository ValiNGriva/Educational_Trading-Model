"""Executes streamlit_app.py against a minimal fake `streamlit` so the UI logic is regression-tested
without a browser (and without Streamlit installed).  Run:  python -m unittest discover -s tests -v"""
import datetime as dt
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class _Ctx:
    """Context manager / layout container; any widget call on it is delegated to the fake `st`."""

    def __init__(self, fake):
        self._fake = fake

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._fake, name)


class FakeStreamlit(types.ModuleType):
    """Widgets return their defaults, layout helpers return context managers, everything else is a no-op."""

    def __init__(self, state):
        super().__init__("streamlit")
        self.session_state = state
        self.calls = []
        self.sidebar = _Ctx(self)

    # decorators
    def _cache(self, *a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda fn: fn

    cache_resource = cache_data = _cache

    # widgets
    def slider(self, label, min_value=None, max_value=None, value=None, step=None, **k):
        return value if value is not None else min_value

    def selectbox(self, label, options, index=0, **k):
        return list(options)[index]

    def radio(self, label, options, index=0, **k):
        return list(options)[index]

    def multiselect(self, label, options, default=None, **k):
        return list(default or [])

    def checkbox(self, label, value=False, **k):
        return value

    def text_input(self, label, value="", **k):
        return value

    def text_area(self, label, value="", **k):
        return value

    def date_input(self, label, value=None, **k):
        return value

    def form_submit_button(self, *a, **k):
        return False

    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [_Ctx(self) for _ in range(n)]

    def tabs(self, labels):
        return [_Ctx(self) for _ in labels]

    def form(self, *a, **k):
        return _Ctx(self)

    expander = spinner = form

    def stop(self):
        raise SystemExit

    def __getattr__(self, name):          # metric, image, dataframe, markdown, write, ...
        if name.startswith("__"):
            raise AttributeError(name)

        def f(*a, **k):
            self.calls.append((name, a, k))
        return f


def run_app(params):
    state = {"params": params} if params else {}
    fake = FakeStreamlit(state)
    saved = sys.modules.get("streamlit")
    sys.modules["streamlit"] = fake
    try:
        code = (ROOT / "streamlit_app.py").read_text()
        try:
            exec(compile(code, "streamlit_app.py", "exec"), {"__name__": "__main__", "__file__": str(ROOT / "streamlit_app.py")})
        except SystemExit:
            pass
    finally:
        if saved is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = saved
    return fake


BASE = dict(source="synthetic", ticker="SPY", start=dt.date(2010, 1, 1), end=dt.date(2025, 12, 31), up=1.0, down=1.0,
            linked=True, train_end=dt.date(2020, 12, 31), val_end=dt.date(2022, 12, 31), fast=True)


class TestApp(unittest.TestCase):
    def test_default_settings_render_all_tabs(self):
        fake = run_app(dict(BASE))
        kinds = [c[0] for c in fake.calls]
        self.assertGreater(kinds.count("image"), 15)       # EDA + modeling + backtest figures
        self.assertGreater(kinds.count("dataframe"), 10)
        self.assertIn("download_button", kinds)
        self.assertFalse(any(c[0] == "error" for c in fake.calls), [c for c in fake.calls if c[0] == "error"])

    def test_max_threshold_shows_message_not_crash(self):
        fake = run_app({**BASE, "up": 20.0, "down": 20.0})
        self.assertTrue(any(c[0] == "warning" and "Lower the thresholds" in str(c[1]) for c in fake.calls))
        self.assertGreater([c[0] for c in fake.calls].count("image"), 5)   # EDA still renders

    def test_asymmetric_thresholds_binary_class_case(self):
        fake = run_app({**BASE, "up": 1.0, "down": 20.0})                   # Sell class vanishes
        self.assertFalse(any(c[0] == "error" for c in fake.calls))


if __name__ == "__main__":
    unittest.main()
