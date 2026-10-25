# Proposed Algorithm Design: AgroStack-CR (Agent 2)

**Project:** Machine Learning Framework for Soil Fertility, Crop Suitability and Fertilizer Recommendation in West Bengal  
**Algorithm Name:** **`AgroStack-CR`** (Cost-Sensitive Agronomic Stacking Ensemble with Conformal Risk Rejection)  
**Selected Architectural Paradigm:** Tripartite Orthogonal Stacking + Bayesian Minimum Expected Cost Decoding + Split-Conformal Uncertainty Rejection Gate  
**Output Deliverable:** `deliverables/proposed_algorithm/algorithm_design.md`  

---

## 1. Candidate Architectural Designs Evaluated

To resolve the 4 core empirical weaknesses documented in `diagnosis_report.md`, we formulated and evaluated three distinct candidate designs:

### Candidate 1: MODE-Stack (Multi-Objective Differential Evolution Feature-Stacking)
- **Concept:** Uses Differential Evolution (DE) to jointly optimize continuous feature-selection masks across the 12 soil parameters and tune stacking blender weights, minimizing both cross-entropy loss and feature acquisition cost.
- **Strengths:** Automated pruning of noisy, non-predictive trace micronutrients ($Zn, B, Fe, Cu, Mn$).
- **Weaknesses:** Highly computationally expensive during cross-validation; does not inherently output calibrated probabilities or distribution-free confidence sets for farmers facing multi-crop ambiguity.

### Candidate 2: H-SFI-Tree (Hierarchical Soil Fertility Constrained Multi-Task Classifier)
- **Concept:** A sequential two-stage pipeline where Stage 1 classifies Soil Fertility Index (SFI) into categorical fertility tiers (Low, Medium, High), and Stage 2 predicts crop suitability conditional on the predicted SFI tier.
- **Strengths:** Enforces strict agronomic hierarchy matching extension manual workflows.
- **Weaknesses:** **Catastrophic cascading error propagation.** If Stage 1 misclassifies SFI tier due to boundary noise, Stage 2 is hopelessly misdirected. Does not address the 4:1 asymmetric agronomic penalty.

### Candidate 3 (CHOSEN CHAMPION): `AgroStack-CR` (Cost-Sensitive Agronomic Stacking Ensemble with Conformal Risk Rejection)
- **Concept & Justification:** Combines the strengths of diverse model families (bagging, boosting, and linear regularized models) through out-of-fold cross-validated meta-learning, applies post-hoc temperature probability calibration, implements Bayesian Minimum Expected Cost decoding under asymmetric 4:1 risk, and equips the output with Split-Conformal Prediction Sets ($1-\alpha=0.90$) and a selective uncertainty rejection gate.
- **Why Chosen:** Directly solves all 4 diagnosed baseline weaknesses without cascading error, preserves scikit-learn pipeline compatibility, executes rapidly, and provides statistically guaranteed coverage intervals for agricultural decision-makers.

---

## 2. One-Paragraph Architecture Summary: `AgroStack-CR`

**`AgroStack-CR`** is a hybrid, scikit-learn-compatible machine learning framework designed specifically to overcome the $\approx 43\%$ sub-tropical Bayes error ceiling and the asymmetric financial risk of crop recommendation in West Bengal. It combines a diverse tripartite base ensemble—randomized sub-space bagging (Extra Trees) to insulate against collinear nutrient ratios, histogram-binned gradient boosting (HistGradientBoosting) to model non-linear agro-climatic thresholds, and regularized multinomial logistic regression ($L_2$) to stabilize posterior calibration—fused via an out-of-fold constrained Ridge meta-learner. Rather than forcing standard 0–1 argmax predictions, `AgroStack-CR` subjects the blended probabilities to temperature scaling and decodes predictions through a **Bayesian Minimum Expected Cost (MEC)** decision rule that penalizes disastrous false-positive crop recommendations under a customizable $C_{FP}:C_{FN}$ cost matrix (evaluated at 1:1 and 4:1). Finally, it wraps predictions in **Split-Conformal Prediction Sets** with guaranteed $90\%$ coverage and an automated **Agronomic Rejection Gate** ($\tau_{\text{reject}}$), providing farmers with a safe multi-crop portfolio and flagging borderline or out-of-distribution soil samples for field verification instead of forcing high-stakes errors.

---

## 3. High-Level System Block Diagram

```
+-------------------------------------------------------------------------------------------------+
|                                 INPUT: Soil & Agro-Climate Vector                               |
|        x in R^D (12 Soil Nutrients, 4 Climate, 2 Geospatial, Stoichiometric Ratios, SFI)       |
+-------------------------------------------------------------------------------------------------+
                                                |
                                                v
+-------------------------------------------------------------------------------------------------+
| STAGE 1: TRIPARTITE DIVERSE BASE ENSEMBLE (Trained via 5-Fold Stratified Stacking)              |
|                                                                                                 |
|   +--------------------------+  +--------------------------+  +-----------------------------+   |
|   |   Extra Trees (n=100)    |  |  HistGradientBoosting    |  |  Multinomial Logistic (L2)  |   |
|   |  - Random split thresholds| |  - Histogram binning      | |  - Generalized linear model  |   |
|   |  - Insulates collinearity|  |  - Non-linear thresholds |  |  - Well-calibrated prior    |   |
|   +--------------------------+  +--------------------------+  +-----------------------------+   |
|                 |                             |                              |                  |
|                 v                             v                              v                  |
|          p_1(x) in Delta^K             p_2(x) in Delta^K              p_3(x) in Delta^K         |
+-------------------------------------------------------------------------------------------------+
                                                |
                                                v
+-------------------------------------------------------------------------------------------------+
| STAGE 2: CONSTRAINED RIDGE META-BLENDER & TEMPERATURE CALIBRATION                               |
|   - Meta-feature matrix Z = [p_1(x), p_2(x), p_3(x)] in R^{3K}                                  |
|   - Softmax Logits: z_meta(x) = W_meta * Z(x) + b_meta                                          |
|   - Temperature-Calibrated Posterior: P_cal(y=k | x) = exp(z_k / T) / sum_j exp(z_j / T)       |
+-------------------------------------------------------------------------------------------------+
                                                |
                        +-----------------------+-----------------------+
                        |                                               |
                        v                                               v
+-----------------------------------------------+ +-----------------------------------------------+
| STAGE 3: BAYESIAN MINIMUM EXPECTED COST (MEC) | | STAGE 4: SPLIT-CONFORMAL PREDICTION & GATE    |
|   - Agronomic Cost Matrix L(i, j)             | |   - Non-conformity Score: s_i = 1 - P(y_i|x_i)|
|     L(i, j) = 4.0 if i != j (FP Crop Penalty) | |   - Conformal Quantile: q_hat_{1-alpha} (90%) |
|     L(i, j) = 0.0 if i == j                   | |   - Safe Recommendation Set:                  |
|   - Expected Risk: R(i | x) = sum_j L(i,j)*P_j| |     C_{0.90}(x) = {k : P(y=k|x) >= 1 - q_hat} |
|   - Point Decision:                           | |                                               |
|     y_hat_MEC = argmin_i R(i | x)             | |   - Conformal Rejection Gate:                 |
|                                               | |     If max_k P(k|x) < tau_reject:             |
|                                               | |       Flag "AMBIGUOUS: RETEST ADVISORY"       |
+-----------------------------------------------+ +-----------------------------------------------+
                        |                                               |
                        +-----------------------+-----------------------+
                                                |
                                                v
+-------------------------------------------------------------------------------------------------+
|                                 FINAL INTEGRATED ADVISORY OUTPUT                                |
|   1. Point Recommendation (Cost-Minimized under 4:1 farmer risk): y_hat_MEC                     |
|   2. 90% Conformal Safe Crop Set (Guaranteed Coverage): C_{0.90}(x)                             |
|   3. Calibrated Recommendation Confidence: max_k P_cal(y=k | x)                                |
|   4. Agronomic Uncertainty Status: [VALID_CONFIDENT | VALID_SET | REJECT_RETEST]                |
+-------------------------------------------------------------------------------------------------+
```

---

## 4. Mathematical Formulation

Let $\mathcal{X} \subset \mathbb{R}^D$ represent the input feature space comprising the 12 soil parameters, agro-climatic indices, stoichiometric nutrient ratios ($N:P, N:K, OC:N$), and Soil Fertility Index (SFI). Let $\mathcal{Y} = \{1, 2, \dots, K\}$ denote the discrete set of $K = 12$ crop classes.

### 4.1. Tripartite Stacking Meta-Blender
Let $M_1(x), M_2(x), M_3(x)$ denote the three base estimators (Extra Trees, HistGradientBoosting, and Multinomial Logistic Regression). During training, 5-fold stratified cross-validation produces out-of-fold probability vectors:
$$\mathbf{p}_m(x_i) \in \Delta^{K-1}, \quad m \in \{1, 2, 3\}$$
where $\Delta^{K-1}$ is the probability simplex. The concatenated meta-feature vector is:
$$\mathbf{z}_i = [\mathbf{p}_1(x_i)^\top, \mathbf{p}_2(x_i)^\top, \mathbf{p}_3(x_i)^\top]^\top \in \mathbb{R}^{3K}$$

The meta-learner optimizes regularized cross-entropy with an $L_2$ shrinkage parameter $\lambda_{\text{meta}}$:
$$\min_{\mathbf{W}, \mathbf{b}} -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K \mathbb{I}(y_i = k) \ln \sigma_k(\mathbf{W} \mathbf{z}_i + \mathbf{b}) + \frac{\lambda_{\text{meta}}}{2} \|\mathbf{W}\|_F^2$$

### 4.2. Temperature Scaling Probability Calibration
To eliminate overconfidence artifacts from tree leaves while preserving class ranking, logits $\boldsymbol{\ell}(x) = \mathbf{W}\mathbf{z}(x) + \mathbf{b}$ are calibrated via a scalar temperature $T > 0$ learned on the validation split:
$$P_{\text{cal}}(y = k \mid x) = \frac{\exp(\ell_k(x) / T)}{\sum_{j=1}^K \exp(\ell_j(x) / T)}$$
where $T^* = \arg\min_T -\sum_{i \in \mathcal{D}_{\text{val}}} \ln P_{\text{cal}}(y_i \mid x_i; T)$.

### 4.3. Bayesian Minimum Expected Cost (MEC) Decision Rule
In standard learning, predictions maximize posterior likelihood ($\hat{y} = \arg\max_k P(y=k|x)$), which implicitly assumes a symmetric 1:1 cost matrix. In West Bengal smallholder agriculture, recommending a failing crop incurs severe seed and fertilizer losses.

Let $\mathbf{L} \in \mathbb{R}^{K \times K}$ be the agronomic loss matrix, where $L(i, j)$ denotes the cost of recommending crop $i$ when true viable crop is $j$:
$$L(i, j) = \begin{cases} 0, & \text{if } i = j \\ C_{FP}, & \text{if } i \neq j \text{ (Type I: False Crop Recommendation)} \\ C_{FN}, & \text{if } i = \text{abstain/missed} \end{cases}$$
For the primary risk analysis, we define $C_{FP} = 4.0$ and $C_{FN} = 1.0$ (Cost Ratio 4:1).

The conditional expected agronomic risk of predicting crop $i$ given observation $x$ is:
$$\mathcal{R}(i \mid x) = \sum_{j=1}^K L(i, j) \cdot P_{\text{cal}}(y = j \mid x)$$

The Bayesian optimal decision $\hat{y}_{\text{MEC}}$ selects the action minimizing expected loss:
$$\hat{y}_{\text{MEC}}(x) = \arg\min_{i \in \{1, \dots, K\}} \mathcal{R}(i \mid x)$$

### 4.4. Split-Conformal Prediction Sets
To address the multi-crop physiological overlap where a single soil profile legitimately supports multiple crops, `AgroStack-CR` constructs distribution-free conformal prediction sets.

1. **Non-conformity Score:** Using a proper validation calibration fold $\mathcal{D}_{\text{cal}} = \{(x_i, y_i)\}_{i=1}^{N_{\text{cal}}}$, compute non-conformity scores:
   $$s_i = 1 - P_{\text{cal}}(y_i \mid x_i)$$
2. **Conformal Quantile:** For a user-specified confidence level $1 - \alpha$ (e.g., $1-\alpha = 0.90$), compute the empirical $(1-\alpha)(1 + 1/N_{\text{cal}})$-th quantile of $\{s_1, \dots, s_{N_{\text{cal}}}\}$, denoted $\hat{q}_{1-\alpha}$.
3. **Prediction Set:** For any novel test sample $x_{\text{new}}$, the conformal set of viable crops is:
   $$\mathcal{C}_{1-\alpha}(x_{\text{new}}) = \left\{ k \in \{1, \dots, K\} : P_{\text{cal}}(y = k \mid x_{\text{new}}) \ge 1 - \hat{q}_{1-\alpha} \right\}$$
   **Theoretical Guarantee:** Under exchangeability of training and test samples:
   $$\mathbb{P}\left( y_{\text{new}} \in \mathcal{C}_{1-\alpha}(x_{\text{new}}) \right) \ge 1 - \alpha$$

### 4.5. Agronomic Conformal Rejection Gate
If a soil sample falls in an extreme or unresolvable edaphic region, the model triggers selective abstention:
$$\text{Decision}(x) = \begin{cases} \hat{y}_{\text{MEC}}(x), & \text{if } \max_k P_{\text{cal}}(y=k \mid x) \ge \tau_{\text{reject}} \text{ and } |\mathcal{C}_{1-\alpha}(x)| \le K_{\max} \\ \text{AMBIGUOUS\_RETEST}, & \text{otherwise} \end{cases}$$
where $\tau_{\text{reject}} = 1 - \hat{q}_{1-\alpha}$ and $K_{\max} = 4$.

---

## 5. Computational Complexity Analysis

1. **Training Time Complexity:**
   - Base Model 1 (Extra Trees): $\mathcal{O}(M \cdot D' \cdot N \log N)$, where $M = 100$ trees, $D' = \sqrt{D}$. On $N=14,000$, empirically $0.74\text{s}$.
   - Base Model 2 (HistGBM): $\mathcal{O}(B \cdot N \cdot D + T_{\text{boost}} \cdot K \cdot N)$, where $B = 256$ bins. Empirically $2.23\text{s}$.
   - Base Model 3 (Logistic Regression): $\mathcal{O}(I_{\text{iter}} \cdot N \cdot D \cdot K)$. Empirically $0.48\text{s}$.
   - 5-Fold Stacking Procedure: Total training requires $5 \times$ base fittings $\approx 17.5\text{ seconds}$.
   - Meta-learner & Conformal Calibration: $\mathcal{O}(N \cdot 3K + N_{\text{cal}} \log N_{\text{cal}}) \approx 0.15\text{s}$.
   - **Total Training Complexity:** $\mathcal{O}(N \log N)$, well within consumer laptop budgets ($<20\text{ seconds}$).

2. **Inference / Prediction Time Complexity:**
   - Per-sample forward pass: Evaluating three base models takes $\approx 15\mu\text{s} + 16\mu\text{s} + 1\mu\text{s} \approx 32\mu\text{s}$.
   - MEC matrix vector product ($12 \times 12$ matrix): $\approx 1\mu\text{s}$.
   - Conformal threshold comparison: $\approx 0.5\mu\text{s}$.
   - **Total Inference Latency:** $\approx 35\mu\text{s}$ per query ($\approx 28,000$ queries/second), fully suitable for real-time mobile advisory deployments.

3. **Memory Footprint:**
   - Disk serialization size: $\approx 110\text{ MB}$ (dominated by Extra Trees estimators).
   - In-memory execution RAM: $< 250\text{ MB}$.

---

## 6. Hyper-Parameters and Search Budget

| Parameter | Type | Default Value | Search Grid / Range | Agronomic Rationale |
|---|---|---|---|---|
| `n_estimators_et` | Integer | $100$ | $[50, 100, 200]$ | Controls variance reduction in randomized tree ensemble |
| `max_iter_hgb` | Integer | $100$ | $[50, 100, 150]$ | Prevents over-boosting on noisy trace micronutrients |
| `C_logistic` | Float | $1.0$ | $[0.1, 1.0, 10.0]$ | $L_2$ regularization strength for calibrated baseline posteriors |
| `meta_C` | Float | $1.0$ | $[0.01, 0.1, 1.0, 10.0]$ | Ridge shrinkage on meta-feature weights to avoid meta-overfitting |
| `temperature_init` | Float | $1.0$ | $[0.5, 2.5]$ (Continuous via L-BFGS-B) | Post-hoc temperature calibration parameter |
| `cost_ratio` ($C_{FP}:C_{FN}$) | Float | $4.0$ | Fixed benchmark values ($1.0$, $4.0$) | Evaluates balanced vs smallholder risk-averse loss |
| `alpha_conformal` | Float | $0.10$ | $[0.05, 0.10, 0.20]$ | Guarantees $(1-\alpha) = 90\%$ marginal ground-truth coverage |
| `max_set_size_rejection` | Integer | $4$ | $[3, 4, 5]$ | Maximum cardinality before triggering ambiguous soil retest advisory |

---

## 7. Hypotheses to be Tested (Pre-Registration)

Before executing experiments on the test set, we formally register the following empirical hypotheses:

1. **Hypothesis 1 (Agronomic Risk Reduction):**
   Under Cost Ratio 4:1 ($C_{FP} = 4.0, C_{FN} = 1.0$), `AgroStack-CR` will achieve a **statistically significant reduction in total Agronomic Risk Penalty** compared to baseline Extra Trees ($17,115.0$) and HistGradientBoosting ($17,390.0$), with $p < 0.05$ via Wilcoxon signed-rank test.
2. **Hypothesis 2 (Probability Calibration Superiority):**
   `AgroStack-CR` will achieve lower multiclass Log Loss ($< 1.15$) and multiclass Brier score than raw Extra Trees ($1.1702$) and raw HistGradientBoosting ($1.5840$).
3. **Hypothesis 3 (Conformal Coverage Guarantee):**
   The conformal prediction sets $\mathcal{C}_{0.90}(x)$ generated by `AgroStack-CR` will satisfy the theoretical marginal coverage condition ($\ge 89.0\%$ empirical test coverage) while maintaining an informative mean set cardinality ($\le 3.0$ crops out of 12).
4. **Hypothesis 4 (Raw Accuracy Realism):**
   Because the $\sim 43\%$ accuracy ceiling is governed by irreducible Bayes error across overlapping sub-tropical crops, `AgroStack-CR` is **not hypothesized to magically reach $80\%-90\%$ raw 0-1 accuracy**; any modest gain in raw accuracy ($+1\%$ to $+3\%$) will be accompanied by substantial variance reduction across CV folds.
