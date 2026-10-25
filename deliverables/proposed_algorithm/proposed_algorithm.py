"""
================================================================================
Proposed Algorithm: AgroStack-CR
(Cost-Sensitive Agronomic Stacking Ensemble with Conformal Risk Rejection)
================================================================================
Author: Final Year B.Tech Computer Science Project (MAKAUT)
Research Study: Soil Fertility, Crop Suitability & Fertilizer Recommendation
                in West Bengal, India
License: Open Academic Research (Reproducible)
================================================================================
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AgroStackCR")


class AgroStackCR(BaseEstimator, ClassifierMixin):
    """
    AgroStack-CR: Cost-Sensitive Agronomic Stacking Ensemble with Conformal Risk Rejection.

    Parameters
    ----------
    n_estimators_et : int, default=100
        Number of trees in the Extra Trees base estimator.
    max_iter_hgb : int, default=100
        Maximum boosting iterations for HistGradientBoosting base estimator.
    C_logistic : float, default=1.0
        Inverse regularization strength for multinomial Logistic Regression base estimator.
    meta_C : float, default=1.0
        Inverse regularization strength for the Ridge meta-learner blender.
    C_FP : float, default=4.0
        Agronomic penalty cost for False Positive crop recommendation (Type I error).
    C_FN : float, default=1.0
        Agronomic cost for False Negative / missed crop recommendation (Type II error).
    alpha : float, default=0.10
        Conformal significance level for prediction sets (target coverage = 1 - alpha = 0.90).
    max_set_size : int, default=4
        Maximum cardinality of conformal prediction set before triggering rejection flag.
    calibration_size : float, default=0.15
        Fraction of training data reserved for temperature calibration and conformal quantile estimation.
    random_state : int, default=42
        Deterministic random seed for full reproducibility.
    """

    def __init__(
        self,
        n_estimators_et: int = 100,
        max_iter_hgb: int = 100,
        C_logistic: float = 1.0,
        meta_C: float = 1.0,
        C_FP: float = 4.0,
        C_FN: float = 1.0,
        alpha: float = 0.10,
        max_set_size: int = 4,
        calibration_size: float = 0.15,
        random_state: int = 42,
    ) -> None:
        self.n_estimators_et = n_estimators_et
        self.max_iter_hgb = max_iter_hgb
        self.C_logistic = C_logistic
        self.meta_C = meta_C
        self.C_FP = C_FP
        self.C_FN = C_FN
        self.alpha = alpha
        self.max_set_size = max_set_size
        self.calibration_size = calibration_size
        self.random_state = random_state

        # Runtime state attributes populated in fit()
        self.classes_: Optional[np.ndarray] = None
        self.K_: Optional[int] = None
        self.temperature_: float = 1.0
        self.q_hat_: float = 1.0
        self.base_models_: Dict[str, Any] = {}
        self.meta_learner_: Optional[LogisticRegression] = None
        self.is_fitted_: bool = False

    def _init_base_models(self) -> Dict[str, Any]:
        """Instantiates fresh instances of the three diverse base estimators."""
        return {
            "et": ExtraTreesClassifier(
                n_estimators=self.n_estimators_et,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "hgb": HistGradientBoostingClassifier(
                max_iter=self.max_iter_hgb,
                random_state=self.random_state,
            ),
            "lr": LogisticRegression(
                C=self.C_logistic,
                max_iter=1000,
                solver="lbfgs",
                random_state=self.random_state,
            ),
        }

    def fit(self, X: Any, y: Any) -> "AgroStackCR":
        """
        Fit AgroStack-CR using out-of-fold cross-validated stacking,
        temperature probability calibration, and conformal quantile estimation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : object
            Returns the fitted estimator.
        """
        X_arr, y_arr = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
        self.classes_ = np.unique(y_arr)
        self.K_ = len(self.classes_)

        if self.K_ < 2:
            raise ValueError(f"AgroStack-CR requires at least 2 distinct classes, found {self.K_}.")

        n_samples = len(y_arr)
        logger.info(
            f"Fitting AgroStack-CR on {n_samples} samples across {self.K_} crop classes..."
        )

        # Map string classes to continuous zero-indexed integers if necessary
        self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        self._idx_to_class = {i: c for i, c in enumerate(self.classes_)}
        y_encoded = np.array([self._class_to_idx[val] for val in y_arr], dtype=np.int64)

        # Partition into stacking split and post-hoc calibration split
        X_stack, X_cal, y_stack, y_cal = train_test_split(
            X_arr,
            y_encoded,
            test_size=self.calibration_size,
            stratify=y_encoded,
            random_state=self.random_state,
        )
        n_stack = len(y_stack)
        n_cal = len(y_cal)

        # ------------------------------------------------------------------
        # Step 1: Out-of-Fold Stacking Feature Generation
        # ------------------------------------------------------------------
        meta_features_stack = np.zeros((n_stack, 3 * self.K_), dtype=np.float64)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_stack, y_stack)):
            X_tr, y_tr = X_stack[train_idx], y_stack[train_idx]
            X_va = X_stack[val_idx]

            fold_models = self._init_base_models()
            col_offset = 0
            for name, model in fold_models.items():
                model.fit(X_tr, y_tr)
                p_val = model.predict_proba(X_va)
                meta_features_stack[val_idx, col_offset : col_offset + self.K_] = p_val
                col_offset += self.K_

        # ------------------------------------------------------------------
        # Step 2: Fit Final Base Models and Meta-Learner
        # ------------------------------------------------------------------
        self.base_models_ = self._init_base_models()
        for name, model in self.base_models_.items():
            model.fit(X_stack, y_stack)

        self.meta_learner_ = LogisticRegression(
            C=self.meta_C,
            max_iter=1000,
            solver="lbfgs",
            random_state=self.random_state,
        )
        self.meta_learner_.fit(meta_features_stack, y_stack)

        # ------------------------------------------------------------------
        # Step 3: Temperature Scaling Probability Calibration
        # ------------------------------------------------------------------
        meta_features_cal = self._generate_meta_features(X_cal)
        raw_logits_cal = self.meta_learner_.decision_function(meta_features_cal)
        if raw_logits_cal.ndim == 1:
            raw_logits_cal = np.vstack([-raw_logits_cal, raw_logits_cal]).T

        def _nll_objective(temp_param: np.ndarray) -> float:
            T = temp_param[0]
            scaled_logits = raw_logits_cal / T
            scaled_probs = softmax(scaled_logits, axis=1)
            # Clip probabilities for numerical stability
            scaled_probs = np.clip(scaled_probs, 1e-12, 1.0 - 1e-12)
            nll = -np.mean(np.log(scaled_probs[np.arange(n_cal), y_cal]))
            return float(nll)

        opt_res = minimize(
            _nll_objective,
            x0=np.array([1.0]),
            bounds=[(0.1, 10.0)],
            method="L-BFGS-B",
        )
        self.temperature_ = float(opt_res.x[0]) if opt_res.success else 1.0
        logger.info(f"Optimal Temperature Scaling T* = {self.temperature_:.4f}")

        # ------------------------------------------------------------------
        # Step 4: Split-Conformal Quantile Estimation
        # ------------------------------------------------------------------
        calibrated_probs_cal = softmax(raw_logits_cal / self.temperature_, axis=1)
        true_probs = calibrated_probs_cal[np.arange(n_cal), y_cal]
        non_conformity_scores = 1.0 - true_probs

        # Finite-sample adjusted quantile formula
        q_target = np.ceil((n_cal + 1) * (1.0 - self.alpha)) / n_cal
        q_target = float(np.clip(q_target, 0.0, 1.0))
        self.q_hat_ = float(np.quantile(non_conformity_scores, q_target, method="higher"))
        logger.info(
            f"Conformal Quantile q_hat (1-alpha={1.0 - self.alpha:.2f}) = {self.q_hat_:.4f}"
        )

        self.is_fitted_ = True
        return self

    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Helper function to concatenate base model probability outputs."""
        n = X.shape[0]
        meta_Z = np.zeros((n, 3 * self.K_), dtype=np.float64)
        col_offset = 0
        for name, model in self.base_models_.items():
            p = model.predict_proba(X)
            meta_Z[:, col_offset : col_offset + self.K_] = p
            col_offset += self.K_
        return meta_Z

    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Returns strictly temperature-calibrated posterior probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input queries.

        Returns
        -------
        P : ndarray of shape (n_samples, n_classes)
            Calibrated posterior probabilities summing to 1.0 per row.
        """
        check_is_fitted(self, ["is_fitted_", "meta_learner_"])
        X_arr = check_array(X, accept_sparse=False, dtype=np.float64)
        meta_Z = self._generate_meta_features(X_arr)
        raw_logits = self.meta_learner_.decision_function(meta_Z)
        if raw_logits.ndim == 1:
            raw_logits = np.vstack([-raw_logits, raw_logits]).T
        calibrated_logits = raw_logits / self.temperature_
        return softmax(calibrated_logits, axis=1)

    def predict(self, X: Any) -> np.ndarray:
        """
        Bayesian Minimum Expected Cost (MEC) Point Prediction.

        Decodes the crop recommendation that minimizes expected farmer risk:
            y_hat = argmin_i sum_j L(i, j) * P(y=j | x)
        where L(i, j) = C_FP for i != j, and 0 for i == j.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Class labels of predicted crops.
        """
        P = self.predict_proba(X)
        n_samples = P.shape[0]

        # Construct Agronomic Loss Matrix L in R^{K x K}
        L = np.full((self.K_, self.K_), fill_value=self.C_FP, dtype=np.float64)
        np.fill_diagonal(L, 0.0)

        # Expected risk R = P @ L.T
        risk_matrix = np.matmul(P, L.T)  # shape: [n_samples, K]
        best_indices = np.argmin(risk_matrix, axis=1)
        return np.array([self._idx_to_class[idx] for idx in best_indices])

    def predict_conformal(self, X: Any) -> List[np.ndarray]:
        """
        Generates distribution-free conformal prediction sets with >= (1 - alpha)
        marginal ground-truth coverage guarantee.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        prediction_sets : list of ndarray
            List where each element contains the array of co-suitable crop labels.
        """
        P = self.predict_proba(X)
        inclusion_threshold = 1.0 - self.q_hat_
        sets = []
        for i in range(P.shape[0]):
            valid_indices = np.where(P[i, :] >= inclusion_threshold)[0]
            if len(valid_indices) == 0:
                # Fallback to argmax if set is empty
                valid_indices = np.array([np.argmax(P[i, :])])
            crop_labels = np.array([self._idx_to_class[idx] for idx in valid_indices])
            sets.append(crop_labels)
        return sets

    def predict_with_status(
        self, X: Any
    ) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
        """
        Integrated advisory output providing point prediction, conformal set,
        and selective agronomic rejection status.

        Status Codes:
        - VALID_CONFIDENT_SINGLE_CROP: Clear agronomic profile, singleton recommendation.
        - VALID_MULTI_CROP_PORTFOLIO: Multiple co-viable crops within 90% conformal set.
        - AMBIGUOUS_RETEST_REQUIRED: Ambiguous soil profile flagged for field verification.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        point_preds : ndarray
        conformal_sets : list of ndarray
        statuses : list of str
        """
        P = self.predict_proba(X)
        point_preds = self.predict(X)
        conformal_sets = self.predict_conformal(X)
        threshold = 1.0 - self.q_hat_

        statuses = []
        for i in range(P.shape[0]):
            max_prob = np.max(P[i, :])
            set_size = len(conformal_sets[i])

            if max_prob < threshold or set_size > self.max_set_size:
                statuses.append("AMBIGUOUS_RETEST_REQUIRED")
            elif set_size > 1:
                statuses.append("VALID_MULTI_CROP_PORTFOLIO")
            else:
                statuses.append("VALID_CONFIDENT_SINGLE_CROP")

        return point_preds, conformal_sets, statuses

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            "n_estimators_et": self.n_estimators_et,
            "max_iter_hgb": self.max_iter_hgb,
            "C_logistic": self.C_logistic,
            "meta_C": self.meta_C,
            "C_FP": self.C_FP,
            "C_FN": self.C_FN,
            "alpha": self.alpha,
            "max_set_size": self.max_set_size,
            "calibration_size": self.calibration_size,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "AgroStackCR":
        """Set the parameters of this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for estimator AgroStackCR.")
        return self
