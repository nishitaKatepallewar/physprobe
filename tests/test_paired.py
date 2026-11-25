import numpy as np

from physprobe import probes


def test_paired_accuracy_counts_pairs_within_sets():
    s = np.array([0.9, 0.1, 0.5, 0.5, 2.0, 3.0])
    y = np.array([1, 0, 1, 0, 1, 0])
    sets = np.array([0, 0, 1, 1, 2, 2])
    # set 0: win, set 1: tie (one half), set 2: loss
    assert probes.paired_accuracy(s, y, sets) == 0.5


def test_paired_scoring_removes_scene_offsets():
    """Large per-scene offsets hide a small label effect from absolute accuracy, but the
    matched-set comparison sees it."""
    rng = np.random.default_rng(0)
    n_scene, d = 60, 40
    X, y, scene = [], [], []
    for s in range(n_scene):
        base = rng.normal(size=d) * 4.0
        for lab in (1, 1, 0, 0):
            X.append(base + rng.normal(size=d) + 1.0 * (2 * lab - 1) * np.eye(d)[0])
            y.append(lab)
            scene.append(s)
    X, scene = np.array(X), np.array(scene)
    t = probes.Target("possible", "binary", np.array(y, float)[:, None])
    df = probes.cv_probe(X, t, scene, pairs=scene, solver="ridge")
    assert df.paired_accuracy.mean() > df.accuracy.mean() + 0.1
    assert df.paired_accuracy.mean() > 0.75
