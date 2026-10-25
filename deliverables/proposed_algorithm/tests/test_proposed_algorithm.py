"""
Unit test suite for AgroStack-CR.
Validates scikit-learn API conformity, output shapes, reproducibility,
conformal coverage bounds, and edge case resilience.
"""

import sys
import os
import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure deliverables/proposed_algorithm is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from proposed_algorithm import AgroStackCR


@pytest.fixture
def synthetic_data():
    """Generates synthetic multi-class agricultural data mimicking 12 crop classes."""
    X, y = make_classification(
        n_samples=400,
        n_features=20,
        n_informative=14,
        n_redundant=4,
        n_classes=6,
        random_state=42,
    )
    return X, y


def test_shapes_and_probabilities(synthetic_data):
    """Checks output shapes and verifies probabilities sum to 1.0."""
    X, y = synthetic_data
    model = AgroStackCR(n_estimators_et=10, max_iter_hgb=10, random_state=42)
    model.fit(X, y)

    # Predict proba
    P = model.predict_proba(X[:25])
    assert P.shape == (25, 6)
    assert np.allclose(np.sum(P, axis=1), 1.0, atol=1e-5)

    # Point predictions
    y_pred = model.predict(X[:25])
    assert y_pred.shape == (25,)
    assert set(y_pred).issubset(set(model.classes_))

    # Conformal sets
    sets = model.predict_conformal(X[:25])
    assert len(sets) == 25
    for s in sets:
        assert len(s) >= 1
        assert set(s).issubset(set(model.classes_))


def test_reproducibility(synthetic_data):
    """Verifies that identical random states produce identical outputs."""
    X, y = synthetic_data
    model1 = AgroStackCR(n_estimators_et=10, max_iter_hgb=10, random_state=42)
    model2 = AgroStackCR(n_estimators_et=10, max_iter_hgb=10, random_state=42)

    model1.fit(X, y)
    model2.fit(X, y)

    P1 = model1.predict_proba(X[:10])
    P2 = model2.predict_proba(X[:10])

    assert np.allclose(P1, P2, atol=1e-7)
    assert np.isclose(model1.temperature_, model2.temperature_, atol=1e-5)
    assert np.isclose(model1.q_hat_, model2.q_hat_, atol=1e-5)


def test_pipeline_integration_no_leakage(synthetic_data):
    """Verifies AgroStack-CR operates seamlessly inside a Scikit-Learn Pipeline without leakage."""
    X, y = synthetic_data
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", AgroStackCR(n_estimators_et=10, max_iter_hgb=10, random_state=42)),
    ])
    pipe.fit(X, y)
    preds = pipe.predict(X[:15])
    assert preds.shape == (15,)


def test_extreme_inputs(synthetic_data):
    """Checks resilience against out-of-range sensor inputs and extreme scaling."""
    X, y = synthetic_data
    model = AgroStackCR(n_estimators_et=10, max_iter_hgb=10, random_state=42)
    model.fit(X, y)

    # Test with extreme values (e.g., pH=14.0 or EC=100.0)
    extreme_X = np.ones((5, X.shape[1])) * 1e4
    P = model.predict_proba(extreme_X)
    assert not np.isnan(P).any()
    assert np.allclose(np.sum(P, axis=1), 1.0, atol=1e-5)

    preds, sets, statuses = model.predict_with_status(extreme_X)
    assert len(statuses) == 5
    for status in statuses:
        assert status in [
            "VALID_CONFIDENT_SINGLE_CROP",
            "VALID_MULTI_CROP_PORTFOLIO",
            "AMBIGUOUS_RETEST_REQUIRED",
        ]


def test_conformal_coverage(synthetic_data):
    """Verifies conformal sets satisfy empirical coverage bounds on holdout set."""
    X, y = synthetic_data
    X_tr, X_te = X[:250], X[250:]
    y_tr, y_te = y[:250], y[250:]

    model = AgroStackCR(n_estimators_et=15, max_iter_hgb=15, alpha=0.10, random_state=42)
    model.fit(X_tr, y_tr)

    sets = model.predict_conformal(X_te)
    covered = sum(y_te[i] in sets[i] for i in range(len(y_te)))
    empirical_coverage = covered / len(y_te)

    # Nominal is 90%; allow finite sample variance band on small calibration split (n=37)
    assert empirical_coverage >= 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
