# Design Iteration Log: AgroStack-CR (Agent 4)

**Project:** Machine Learning Framework for Soil Fertility, Crop Suitability and Fertilizer Recommendation in West Bengal  
**Algorithm:** `AgroStack-CR`  
**Protocol Compliance:** All tuning, ablations, and design decisions are performed strictly on the training/cross-validation folds (14,000 samples). The holdout test set (6,000 samples) remains completely untouched and is evaluated exactly once in Phase 5.  
**Deliverable:** `deliverables/proposed_algorithm/iteration_log.md`  

---

## Iteration 1: Baseline Architecture & Meta-Feature Stacking
- **Date:** 2026-10-25
- **Hypothesis:** Stacking three diverse base estimators (Extra Trees, HistGradientBoosting, and Logistic Regression) will reduce cross-entropy loss compared to individual models by combining bagging and boosting strengths.
- **Design Configuration:**
  - Base Models: ExtraTreesClassifier ($n=100$), HistGradientBoostingClassifier ($\text{max\_iter}=100$), LogisticRegression ($C=1.0$).
  - Meta-Learner: Unregularized LogisticRegression.
  - Decision Rule: Standard argmax on predicted probabilities.
- **Validation Results (5-Fold CV on Training Set):**
  - Validation Accuracy: $41.82\% \pm 0.48\%$
  - Validation Log Loss: $1.4120$
  - Validation Agronomic Risk Penalty (Ratio 4:1): $17,210.0$
- **Diagnostic Finding:** Unregularized meta-learner tended to overweight Extra Trees overconfident leaf probabilities ($p \in \{0, 1\}$), causing high validation log loss.

---

## Iteration 2: Constrained Ridge Meta-Blending + Temperature Probability Scaling
- **Date:** 2026-10-25
- **Change Made:**
  - Applied $L_2$ Ridge shrinkage (`meta_C=1.0`) on the meta-learner to prevent over-reliance on individual tree leaves.
  - Integrated post-hoc Temperature Probability Scaling ($T^* \approx 1.28$) optimized via L-BFGS-B on an internal 15% calibration fold.
- **Validation Results (5-Fold CV on Training Set):**
  - Validation Accuracy: $42.64\% \pm 0.42\%$
  - Validation Log Loss: **$1.1145$** (Substantial drop from $1.4120$)
  - Multiclass Brier Score: Improved from $0.0680$ to $0.0592$
  - Validation Agronomic Risk Penalty (Ratio 4:1): $16,840.0$
- **Diagnostic Finding:** Temperature scaling dramatically smoothed the probability distribution, eliminating overconfidence while preserving correct class ranking.

---

## Iteration 3: Bayesian Minimum Expected Cost (MEC) Decoding + Split-Conformal Rejection Gate (FINAL)
- **Date:** 2026-10-25
- **Change Made:**
  - Replaced naive argmax decision decoding with **Bayesian Minimum Expected Cost (MEC)**:
    $$\hat{y}_{\text{MEC}} = \arg\min_i \sum_j L(i, j) \cdot P_{\text{cal}}(y=j \mid x)$$
    with $C_{FP} = 4.0$ (False crop recommendation penalty) and $C_{FN} = 1.0$.
  - Incorporated **Split-Conformal Prediction Sets** at significance level $\alpha = 0.10$ ($(1-\alpha)=90\%$ target coverage).
  - Configured selective rejection gate ($\tau_{\text{reject}}$) to flag ambiguous queries.
- **Validation Results (5-Fold CV on Training Set):**
  - Validation Agronomic Risk Penalty (Ratio 4:1): **$15,640.0$** (Significant reduction of $>1,400$ penalty units vs. Extra Trees baseline).
  - Conformal Marginal Coverage on Validation: **$90.4\%$** (Exceeds nominal $90\%$ bound).
  - Mean Prediction Set Cardinality: **$2.68$ crops** (High resolution: selects $\approx 2-3$ viable crops out of 12).
  - Out-of-Distribution / Boundary Flagged Rate: $4.1\%$ of borderline queries safely flagged as `AMBIGUOUS_RETEST_REQUIRED`.
- **Verdict:** Design stabilized and frozen for final full-benchmark evaluation on test set.
