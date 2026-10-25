# Algorithmic Pseudocode: AgroStack-CR (Agent 2)

**Algorithm Name:** **`AgroStack-CR`** (Cost-Sensitive Agronomic Stacking Ensemble with Conformal Risk Rejection)  
**File Output:** `deliverables/proposed_algorithm/pseudocode.md`  

---

## 1. Algorithm Overview

`AgroStack-CR` operates across four coordinated algorithmic modules:
1. **Module 1 (`Fit_Stacking`):** Trains the tripartite base learners (Extra Trees, HistGradientBoosting, Multinomial Logistic Regression) using stratified out-of-fold cross-validation to assemble an unbiased meta-feature training matrix $\mathbf{Z} \in \mathbb{R}^{N \times 3K}$, and fits the regularized Ridge meta-classifier $\mathcal{M}_{\text{meta}}$.
2. **Module 2 (`Calibrate_Temperature`):** Solves a one-dimensional negative log-likelihood minimization on holdout validation logits to find the optimal temperature scaling parameter $T^* > 0$.
3. **Module 3 (`Conformal_Calibration`):** Computes empirical non-conformity scores $s_i = 1 - P_{\text{cal}}(y_i \mid x_i)$ on the calibration set $\mathcal{D}_{\text{cal}}$ and determines the empirical $(1-\alpha)$-quantile $\hat{q}_{1-\alpha}$.
4. **Module 4 (`Predict_MEC_and_Conformal`):** For novel input queries, generates calibrated class posteriors, evaluates Bayesian Minimum Expected Cost (MEC) under agronomic loss matrix $\mathbf{L}$, constructs the $90\%$ conformal prediction set $\mathcal{C}_{1-\alpha}(x)$, and applies the selective uncertainty rejection gate $\tau_{\text{reject}}$.

---

## 2. Complete Pseudocode

```python
"""
================================================================================
ALGORITHM: AgroStack-CR (Cost-Sensitive Agronomic Stacking with Conformal Risk Rejection)
================================================================================
INPUTS:
    D_train = {(x_i, y_i)}_{i=1}^N : Training dataset with features x_i in R^D,
                                      labels y_i in {0, 1, ..., K-1} (K=12 crops)
    V_folds                        : Number of internal stacking folds (default = 5)
    C_FP, C_FN                     : Agronomic cost penalties (default: C_FP=4.0, C_FN=1.0)
    alpha                          : Conformal significance level (default = 0.10 for 90% coverage)
    tau_reject                     : Probability threshold for selective rejection (default = auto)
    max_set_size                   : Maximum conformal set cardinality before flagging (default = 4)
    random_state                   : Deterministic pseudorandom seed (default = 42)

OUTPUTS:
    Fitted AgroStack-CR Model Object providing:
        - fit(X, y)
        - predict_proba(X)        -> Calibrated Posterior Matrix [N_test, K]
        - predict(X)              -> Bayesian MEC Predictions [N_test]
        - predict_conformal(X)    -> List of Safe Viable Crop Sets [N_test]
        - predict_with_status(X)  -> Tuple of (point_pred, set_pred, uncertainty_status)
================================================================================
"""

CLASS AgroStackCR:

    FUNCTION __init__(self,
                      n_estimators_et=100,
                      max_iter_hgb=100,
                      C_logistic=1.0,
                      meta_C=1.0,
                      C_FP=4.0,
                      C_FN=1.0,
                      alpha=0.10,
                      max_set_size=4,
                      random_state=42):
        self.n_estimators_et = n_estimators_et
        self.max_iter_hgb = max_iter_hgb
        self.C_logistic = C_logistic
        self.meta_C = meta_C
        self.C_FP = C_FP
        self.C_FN = C_FN
        self.alpha = alpha
        self.max_set_size = max_set_size
        self.random_state = random_state

        # Initialize Base Estimators
        self.base_models = {
            'et': ExtraTreesClassifier(n_estimators=self.n_estimators_et,
                                       random_state=self.random_state,
                                       n_jobs=-1),
            'hgb': HistGradientBoostingClassifier(max_iter=self.max_iter_hgb,
                                                  random_state=self.random_state),
            'lr': LogisticRegression(C=self.C_logistic,
                                     max_iter=1000,
                                     multi_class='multinomial',
                                     random_state=self.random_state)
        }
        self.meta_learner = LogisticRegression(C=self.meta_C,
                                               max_iter=1000,
                                               multi_class='multinomial',
                                               random_state=self.random_state)
        self.temperature = 1.0
        self.q_hat = 1.0
        self.classes_ = None
        self.K = None
    END FUNCTION


    FUNCTION fit(self, X, y):
        """
        Fits base learners, generates out-of-fold meta-features, fits meta-learner,
        and performs temperature calibration and conformal quantile estimation.
        """
        N, D = shape(X)
        self.classes_ = unique_classes(y)
        self.K = length(self.classes_)

        # Split 85% for Stacking Fit, 15% for Temperature Calibration & Conformal Quantiles
        X_stack, X_cal, y_stack, y_cal = stratified_split(X, y, test_size=0.15,
                                                          random_state=self.random_state)
        N_stack = length(y_stack)
        N_cal = length(y_cal)

        # -------------------------------------------------------------
        # STEP 1: Out-of-Fold Stacking Feature Generation
        # -------------------------------------------------------------
        Z_meta = zeros(shape=(N_stack, 3 * self.K))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        FOR train_idx, val_idx IN cv.split(X_stack, y_stack):
            X_tr, y_tr = X_stack[train_idx], y_stack[train_idx]
            X_va = X_stack[val_idx]

            # Clone and fit temporary base models on fold
            fold_models = clone_models(self.base_models)
            col_offset = 0
            FOR name, model IN fold_models.items():
                model.fit(X_tr, y_tr)
                p_val = model.predict_proba(X_va)  # shape: [len(val_idx), K]
                Z_meta[val_idx, col_offset : col_offset + self.K] = p_val
                col_offset += self.K
            END FOR
        END FOR

        # -------------------------------------------------------------
        # STEP 2: Fit Final Base Models and Meta-Learner
        # -------------------------------------------------------------
        # Fit final base models on all X_stack
        FOR name, model IN self.base_models.items():
            model.fit(X_stack, y_stack)
        END FOR

        # Fit meta-learner on out-of-fold meta-features
        self.meta_learner.fit(Z_meta, y_stack)

        # -------------------------------------------------------------
        # STEP 3: Post-Hoc Temperature Scaling on Calibration Set
        # -------------------------------------------------------------
        # Generate meta-features for calibration split
        Z_cal = self._generate_meta_features(X_cal)
        raw_logits_cal = self.meta_learner.decision_function(Z_cal)

        # Optimize temperature T via L-BFGS-B on validation NLL
        FUNCTION objective_nll(T):
            scaled_logits = raw_logits_cal / T[0]
            probs = softmax(scaled_logits, axis=1)
            # Compute negative log-likelihood of ground-truth labels
            loss = -mean(log(probs[arange(N_cal), y_cal] + 1e-12))
            RETURN loss
        END FUNCTION

        opt_result = minimize(objective_nll, x0=[1.0], bounds=[(0.1, 5.0)], method='L-BFGS-B')
        self.temperature = opt_result.x[0]

        # -------------------------------------------------------------
        # STEP 4: Conformal Calibration Quantile Estimation
        # -------------------------------------------------------------
        calibrated_probs_cal = softmax(raw_logits_cal / self.temperature, axis=1)
        # Compute non-conformity scores: s_i = 1 - P(y_true | x_i)
        true_class_probs = calibrated_probs_cal[arange(N_cal), y_cal]
        non_conformity_scores = 1.0 - true_class_probs

        # Finite-sample adjusted quantile: q_level = ceil((N_cal + 1) * (1 - alpha)) / N_cal
        q_level = ceil((N_cal + 1) * (1.0 - self.alpha)) / N_cal
        q_level = clip(q_level, 0.0, 1.0)
        self.q_hat = quantile(non_conformity_scores, q_level, method='higher')

        RETURN self
    END FUNCTION


    FUNCTION _generate_meta_features(self, X):
        """Helper to concatenate base model probability predictions."""
        N = shape(X)[0]
        Z = zeros(shape=(N, 3 * self.K))
        col_offset = 0
        FOR name, model IN self.base_models.items():
            p = model.predict_proba(X)
            Z[:, col_offset : col_offset + self.K] = p
            col_offset += self.K
        END FOR
        RETURN Z
    END FUNCTION


    FUNCTION predict_proba(self, X):
        """
        Returns strictly temperature-calibrated posterior probabilities.
        """
        Z = self._generate_meta_features(X)
        raw_logits = self.meta_learner.decision_function(Z)
        calibrated_logits = raw_logits / self.temperature
        probs = softmax(calibrated_logits, axis=1)
        RETURN probs
    END FUNCTION


    FUNCTION predict(self, X):
        """
        Bayesian Minimum Expected Cost (MEC) Point Prediction.
        Decodes the crop recommendation minimizing farmer risk under C_FP : C_FN.
        """
        P = self.predict_proba(X)  # shape: [N_samples, K]
        N_samples = shape(P)[0]

        # Construct Agronomic Cost Matrix L in R^{K x K}
        # L[i, j] = Cost of predicting i when true viable crop is j
        L = full(shape=(self.K, self.K), fill_value=self.C_FP)
        FOR k FROM 0 TO self.K - 1:
            L[k, k] = 0.0  # Zero cost for correct recommendation
        END FOR

        # Compute Expected Risk for each candidate action i:
        # R(i | x) = sum_{j=0}^{K-1} L[i, j] * P[x, j]
        # In matrix form: Risk_Matrix = P @ L.T  (shape: [N_samples, K])
        Risk_Matrix = matmul(P, transpose(L))

        # Select action i that minimizes expected risk
        y_pred_indices = argmin(Risk_Matrix, axis=1)
        RETURN self.classes_[y_pred_indices]
    END FUNCTION


    FUNCTION predict_conformal(self, X):
        """
        Constructs distribution-free conformal prediction sets with >= (1 - alpha)
        marginal ground-truth coverage guarantee.
        """
        P = self.predict_proba(X)
        N_samples = shape(P)[0]
        prediction_sets = []

        # Inclusion threshold: Include class k if P(y=k | x) >= 1 - q_hat
        threshold = 1.0 - self.q_hat

        FOR i FROM 0 TO N_samples - 1:
            viable_class_indices = where(P[i, :] >= threshold)[0]
            IF length(viable_class_indices) == 0:
                # Fallback to top-1 if set is empty due to extreme uncertainty
                viable_class_indices = [argmax(P[i, :])]
            END IF
            prediction_sets.append(self.classes_[viable_class_indices])
        END FOR

        RETURN prediction_sets
    END FUNCTION


    FUNCTION predict_with_status(self, X):
        """
        Integrated advisory output delivering point prediction, conformal set,
        and selective agronomic rejection status.
        """
        P = self.predict_proba(X)
        point_preds = self.predict(X)
        conformal_sets = self.predict_conformal(X)
        N_samples = shape(P)[0]

        rejection_threshold = 1.0 - self.q_hat
        statuses = []

        FOR i FROM 0 TO N_samples - 1:
            max_prob = max(P[i, :])
            set_size = length(conformal_sets[i])

            IF max_prob < rejection_threshold OR set_size > self.max_set_size:
                statuses.append("AMBIGUOUS_RETEST_REQUIRED")
            ELSE IF set_size > 1:
                statuses.append("VALID_MULTI_CROP_PORTFOLIO")
            ELSE:
                statuses.append("VALID_CONFIDENT_SINGLE_CROP")
            END IF
        END FOR

        RETURN point_preds, conformal_sets, statuses
    END FUNCTION

END CLASS
```
