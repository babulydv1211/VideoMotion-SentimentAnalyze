import numpy as np

from trained_spatial_pillar import ROOT_CLASS_NAMES, _root_order


def test_trained_spatial_adapter_reorders_checkpoint_class_order():
    # Checkpoint order is Positive, Neutral, Negative; root app order is the
    # displayed Negative, Neutral, Positive order.
    actual = _root_order(np.array([0.20, 0.30, 0.50], dtype=np.float32))
    assert ROOT_CLASS_NAMES == ("Negative", "Neutral", "Positive")
    np.testing.assert_allclose(actual, [0.50, 0.30, 0.20])
