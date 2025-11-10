import importlib


def test_import_fixray():
    module = importlib.import_module("fixray")
    assert module is not None
