"""pytest entry point: the same checks as `python -m physprobe selftest`."""
from physprobe import selftest


def test_planted_structure():
    selftest._results.clear()
    selftest.part_a()
    assert all(ok for _, ok in selftest._results), [n for n, ok in selftest._results if not ok]


def test_pipeline_on_tiny_model():
    selftest._results.clear()
    selftest.part_b()
    assert all(ok for _, ok in selftest._results), [n for n, ok in selftest._results if not ok]
