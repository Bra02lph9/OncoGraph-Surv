# Multimodal Survival Prediction in METABRIC Using Graph Attention Networks

## Overview

This project implements an end-to-end multimodal survival-analysis pipeline for breast cancer using the METABRIC cohort.

The main objective is to predict patient mortality risk from heterogeneous clinical and molecular data while explicitly modeling relationships between patients through a graph.

Each patient is represented as:

* a node containing clinical and molecular features;
* a member of a patient–patient similarity graph;
* an observation with a survival time;
* an event indicator specifying whether death was observed or censored.

The primary model is a multimodal Graph Attention Network adapted to censored survival data. It produces a continuous, unbounded Cox risk score rather than a conventional class probability.

The notebook also includes:

* strict fold-specific preprocessing;
* clinical, gene-expression, mutation, CNA and pathway features;
* multi-view patient graph construction;
* Cox proportional-hazards loss;
* pairwise survival-ranking loss;
* nested validation inside each cross-validation fold;
* out-of-fold survival predictions;
* time-dependent AUC;
* Brier score and integrated Brier score;
* Kaplan–Meier risk stratification;
* calibration analysis;
* feature-selection stability;
* attention-weight analysis;
* MLP baselines;
* an ablation study;
* final-model serialization and reload verification.

---

# 1. Clinical question

The clinical objective is to estimate the relative mortality risk of breast cancer patients from information available in METABRIC.

For patient (i), the model receives:

[
X_i =
\left[
X_i^{\text{clinical}},
X_i^{\text{expression}},
X_i^{\text{mutation}},
X_i^{\text{CNA}},
X_i^{\text{pathway}}
\right]
]

and predicts a risk score:

[
r_i \in \mathbb{R}
]

A larger risk score means a higher predicted mortality risk.

The output is not constrained to the interval ([0,1]), because the model is trained using a Cox proportional-hazards objective:

[
h_i(t) = h_0(t)\exp(r_i)
]

where:

* (h_i(t)) is the hazard of patient (i);
* (h_0(t)) is the baseline hazard;
* (r_i) is the neural-network risk score.


The model can subsequently estimate survival probabilities at clinically relevant time horizons using the Breslow baseline-hazard estimator.

---

# 2. Dataset

## 2.1 Source

The notebook uses the Kaggle METABRIC dataset:

```text
/kaggle/input/datasets/raghadalharbi/
breast-cancer-gene-expression-profiles-metabric/
METABRIC_RNA_Mutation.csv
```

The data are supplied as a single table in which each row represents one patient.

After outcome cleaning and exclusion of invalid survival observations, the evaluated dataset contained approximately:

```text
1,903 patients
```

The exact number may vary if outcome-filtering rules or the input version are changed.

---

## 2.2 METABRIC

METABRIC stands for:

```text
Molecular Taxonomy of Breast Cancer International Consortium
```

The cohort contains breast cancer patients characterized by clinical variables and several types of molecular measurements.

The notebook treats the dataset as a multimodal table and detects or constructs the following information groups.

### Clinical variables

Examples include:

* age at diagnosis;
* type of breast surgery;
* cancer type;
* detailed cancer type;
* tumor cellularity;
* chemotherapy;
* hormone-receptor status;
* HER2 status;
* PAM50 and claudin-low subtype;
* tumor grade;
* tumor size;
* lymph-node status;
* menopausal status;
* treatment-related variables;
* Nottingham Prognostic Index or its components, depending on configuration.

### Gene-expression variables

Continuous gene-expression measurements are detected from numeric columns that are not identified as:

* clinical variables;
* outcome variables;
* mutation columns;
* CNA columns;
* metadata;
* technical columns;
* leakage-prone variables.

### Mutation variables

Mutation columns are converted into binary or numeric mutation indicators.

A mutation feature generally represents whether a gene contains a detected mutation for a particular patient.

True missing values are preserved during the initial encoding and subsequently handled by the fold-specific imputer.

### Copy-number alteration variables

CNA features represent genomic gains, losses or copy-number states.

These features are detected conservatively because gene-expression and CNA columns can sometimes have similar names.

### Pathway-level features

Pathway scores are calculated from predefined gene signatures.

Instead of relying only on thousands of individual genes, the model can use biologically meaningful aggregate representations such as:

* proliferation;
* estrogen signaling;
* HER2 signaling;
* DNA repair;
* apoptosis;
* immune response;
* epithelial–mesenchymal transition;
* PI3K–AKT–mTOR;
* MAPK signaling;
* cell cycle;
* hypoxia;
* angiogenesis.

Only genes available in the METABRIC expression table are used.

---

# 3. Survival endpoint

## 3.1 Primary endpoint

The primary endpoint is overall survival.

Two standardized columns are created:

```text
survival_time
event
```

where:

* `survival_time` is the observed follow-up time;
* `event = 1` indicates an observed death;
* `event = 0` indicates a censored patient.

Patients with invalid, missing, non-finite or non-positive survival times are excluded.

The original row identifier is retained as:

```text
original_row_index
```

This allows predictions to be traced back to their original input rows.

---

## 3.2 Cancer-specific endpoint

The notebook also attempts to detect a reliable cancer-specific event column.

When such a column is unavailable, the pipeline explicitly reports that no cancer-specific outcome was invented.

Overall survival therefore remains the primary endpoint.

A future cancer-specific extension would need to distinguish between:

* breast-cancer deaths;
* deaths from other causes;
* censored observations.

Cause-specific Cox analysis would generally censor other-cause deaths at their observed time, whereas cumulative-incidence analysis would require a competing-risk framework.

---

# 4. Leakage prevention

Survival modeling is highly sensitive to information leakage.

The notebook therefore identifies and excludes columns that directly or indirectly reveal the outcome.

Examples include columns containing terms related to:

* overall survival;
* survival status;
* death;
* deceased status;
* follow-up;
* disease-specific survival;
* recurrence-free survival;
* progression-free survival;
* event labels;
* outcome-derived metadata.

The patient identifier, source outcome columns, technical indices and metadata-only variables are also excluded from model inputs.

Most importantly, all learned preprocessing operations are fitted only on the training portion of the current fold.

This includes:

* missing-value imputation;
* variance filtering;
* scaling;
* dimensionality reduction;
* gene ranking;
* survival-based feature selection;
* mutation selection;
* CNA filtering;
* pathway transformations;
* clinical encoding.

The held-out fold is never used to select features, fit scalers or choose the best epoch.

---

# 5. Clinical-feature configurations

The notebook supports several clinical-input strategies.

## 5.1 NPI without components

```text
clinical_feature_mode = "npi_without_components"
```

This mode includes the Nottingham Prognostic Index but removes individual variables already used to calculate it.

This helps reduce redundancy.

## 5.2 Components without NPI

```text
clinical_feature_mode = "components_without_npi"
```

This mode excludes the final NPI value but retains its clinical components, such as:

* tumor size;
* tumor grade;
* lymph-node status.

This can make the model more flexible than using the fixed clinical index.

## 5.3 All features with NPI as a secondary variable

```text
clinical_feature_mode = "all_with_npi_secondary"
```

This mode retains both NPI and its component variables.

It can provide more information but may introduce redundancy and collinearity.

Treatment variables may be included or excluded using the corresponding configuration parameter.

---

# 6. Fold-specific preprocessing

For every cross-validation fold, a new preprocessor is fitted exclusively on the fold’s training patients.

The resulting preprocessing artifact contains separate transformations for each modality.

## 6.1 Clinical preprocessing

Numerical clinical variables are:

1. coerced to numeric values;
2. imputed;
3. standardized.

Categorical clinical variables are:

1. imputed;
2. one-hot encoded;
3. aligned with categories observed in the training fold.

Unknown categories in validation or held-out patients are safely ignored.

---

## 6.2 Gene-expression preprocessing

The expression pipeline can perform:

1. numeric conversion;
2. missing-value imputation;
3. variance filtering;
4. standardization;
5. survival-aware feature ranking;
6. selection of the top genes;
7. optional PCA representation.

Several gene-selection strategies are supported.

### Mutual-information selection

The relationship between a gene and survival information is estimated using a mutual-information-based score.

### Univariate Cox selection

Each gene is evaluated independently using a univariate Cox score test.

The selection table may contain:

* estimated coefficient;
* standard error;
* z statistic;
* p-value;
* feature score;
* rank;
* selected status.

### Cox elastic-net selection

When `scikit-survival` is available, a Cox elastic-net model can be used to select features through penalized survival regression.

The elastic-net approach is particularly useful when:

* there are many correlated genes;
* the number of features is large relative to the number of patients;
* sparse feature selection is desired.

Two separate expression representations may be selected:

```text
node expression genes
graph expression genes
```

Node genes are included in patient node attributes.

Graph genes are used when calculating expression-based patient similarity.

---

## 6.3 Mutation preprocessing

Mutation variables are:

1. encoded into numerical mutation indicators;
2. imputed using training-fold information;
3. filtered according to mutation frequency and missingness;
4. scored for survival relevance;
5. ranked;
6. selected.

The output retains:

* mutation name;
* selection score;
* mutation frequency;
* missing-value fraction;
* fold;
* rank.

---

## 6.4 CNA preprocessing

CNA variables are:

1. converted to numeric values;
2. imputed;
3. filtered by variance;
4. standardized.

CNA features are included in the node-feature matrix when available.

---

## 6.5 Pathway preprocessing

For each pathway, the notebook identifies the subset of signature genes available in the expression data.

A minimum number of available genes may be required before a pathway is retained.

Pathway scores can be calculated using strategies such as:

* mean standardized expression;
* a pathway-specific PCA representation.

Pathway transformations are fitted on the training fold and reused on validation and held-out patients.

The resulting pathway metadata are saved for interpretation and reproducibility.

---

# 7. Patient-node representation

After fold-specific transformations, all active modalities are concatenated into one node-feature matrix:

```text
X_node
```

Conceptually:

[
X_{node} =
[
X_{clinical}
\mid
X_{expression}
\mid
X_{mutation}
\mid
X_{CNA}
\mid
X_{pathway}
]
]

The preprocessing artifact stores the start and end indices of each modality:

```python
modality_slices = {
    "clinical": (start, end),
    "expression": (start, end),
    "mutation": (start, end),
    "cna": (start, end),
    "pathway": (start, end),
}
```

These slices allow the neural network to process each modality independently before fusion.

The node-feature names are exported to:

```text
artifacts/node_feature_names.csv
```

---

# 8. Multi-view patient graph

## 8.1 Why use a graph?

A standard MLP treats patients as independent observations.

However, biologically similar patients may share:

* clinical profiles;
* molecular expression patterns;
* pathway activity;
* tumor subtype;
* receptor status.

A graph allows each patient to receive information from similar patients.

The graph is defined as:

[
G = (V,E)
]

where:

* each node (v_i \in V) represents one patient;
* each edge ((i,j) \in E) indicates patient similarity;
* each edge contains one or more similarity attributes.

---

## 8.2 Graph views

The graph is built from three separate patient representations.

### Clinical view

```text
X_graph_clinical
```

This view measures similarity in transformed clinical space.

### Expression view

```text
X_graph_expression
```

This view measures similarity using selected graph-specific expression genes.

### Pathway view

```text
X_graph_pathway
```

This view measures similarity using pathway activity scores.

Each view is normalized row-wise, and cosine similarity is used.

---

## 8.3 K-nearest-neighbor graph

For every active view, a K-nearest-neighbor search identifies the most similar patients.

Self-neighbors are removed at this stage.

The number of neighbors is controlled by:

```text
knn_neighbors
```

Possible edge-symmetry strategies include:

### Union graph

An undirected connection is retained if at least one direction exists.

### Mutual graph

A connection is retained only when both patients select each other as neighbors.

The mutual strategy creates a more conservative graph.

---

## 8.4 Graph-construction modes

### Weighted union

```text
graph_mode = "weighted_union"
```

An edge may be retained when it appears in at least one active view.

This produces a denser graph.

### Multi-view consensus

```text
graph_mode = "multi_view_consensus"
```

An edge is retained only when a minimum number of views agree.

For example:

```text
minimum_agreeing_views = 2
```

requires an edge to appear in at least two of:

* clinical view;
* expression view;
* pathway view.

This aims to reduce noisy patient relationships.

### Single-view graph

The notebook can also construct graphs from only one modality:

```text
clinical_knn
expression_knn
pathway_knn
```

These modes are useful for graph ablation experiments.

---

## 8.5 Edge attributes

The graph uses multidimensional edge attributes instead of a single scalar weight.

Potential edge features include:

```text
clinical_similarity
expression_similarity
pathway_similarity
same_pam50
same_er_status
same_her2_status
```

The first three are cosine similarities.

The status features equal one when both connected patients share the same known category and zero otherwise.

A weighted global similarity is also calculated using:

```text
alpha_clinical
alpha_expression
alpha_pathway
```

Edges below the configured minimum similarity threshold may be removed.

---

## 8.6 Self-loops

Exactly one self-loop is added for every patient.

Self-loops allow a patient to retain its own information during graph message passing.

The notebook verifies that:

```text
each node has exactly one self-loop
```

and that edge attributes and edge indices remain aligned.

---

## 8.7 Outcome independence

Survival events are never used to create graph edges.

The `event` vector is supplied only after graph construction to calculate a descriptive event-homophily statistic.

This avoids directly leaking the outcome into the graph topology.

The graph-construction process and fold-specific matrices are generated before the SurvivalGAT is trained.

---

# 9. SurvivalGAT architecture

## 9.1 Design motivation

The original GAT improved over a clinical-only model but remained weaker than the full-feature MLP.

This suggested that:

* multimodal patient features contained strong prognostic information;
* forcing all information through message passing could degrade part of that signal;
* some patient-neighbor relationships were useful;
* other graph connections were potentially noisy.

The updated architecture therefore combines:

1. a strong direct multimodal branch;
2. a residual graph-attention branch;
3. a patient-specific gate controlling graph contribution;
4. a final learned fusion.

---

## 9.2 Modality-specific encoders

Each modality is processed by its own encoder:

[
z_i^{(m)} = f_m(X_i^{(m)})
]

where (m) may correspond to:

* clinical;
* expression;
* mutation;
* CNA;
* pathway.

Each encoder includes:

* layer normalization;
* linear projection;
* nonlinear activation;
* dropout;
* a second linear transformation.

This prevents a very large expression block from automatically dominating smaller clinical or pathway blocks.

---

## 9.3 Learnable modality fusion

The model contains learnable modality logits.

They are converted to normalized weights using softmax:

[
\alpha_m =
\frac{\exp(w_m)}
{\sum_k \exp(w_k)}
]

The fused patient representation is:

[
z_i =
\sum_m \alpha_m z_i^{(m)}
]

This allows the model to learn the global importance of each modality.

---

## 9.4 Direct non-graph branch

A residual MLP branch processes the fused patient representation without using neighbors.

This branch preserves the strong individual-level signal identified by the full-feature MLP baseline.

Conceptually:

[
z_i^{direct} = z_i + MLP(z_i)
]

It produces a direct Cox risk contribution:

[
r_i^{direct}
]

---

## 9.5 Edge encoder

Raw multidimensional edge attributes are passed through an edge encoder.

This allows the model to transform:

* clinical similarity;
* expression similarity;
* pathway similarity;
* receptor agreement;
* subtype agreement;

into a representation better suited to attention computation.

---

## 9.6 Graph-attention layers

The graph branch contains two edge-aware GAT layers.

For patient (i), a GAT layer aggregates information from its neighborhood:

[
h_i' =
\sum_{j \in \mathcal{N}(i)}
\alpha_{ij} W h_j
]

where (\alpha_{ij}) is a learned attention coefficient.

The attention coefficient depends on:

* the source-patient representation;
* the target-patient representation;
* the edge attributes.

Multiple attention heads allow the model to learn different forms of patient relationships.

The attention heads are combined without expanding the final hidden dimension.

---

## 9.7 Residual graph blocks

Each GAT layer is followed by:

* dropout;
* residual addition;
* layer normalization;
* feed-forward transformation;
* another residual addition.

Conceptually:

[
h_i^{(1)} =
LayerNorm(
z_i^{direct}
+
GAT_1(z_i^{direct})
)
]

[
h_i^{(1,ffn)} =
h_i^{(1)}
+
FFN_1(h_i^{(1)})
]

The same logic is repeated for the second GAT block.

Residual connections help:

* retain the patient’s original information;
* stabilize optimization;
* reduce oversmoothing;
* train a deeper model.

---

## 9.8 Patient-specific graph gate

Not all patients benefit equally from neighbor information.

The model therefore learns a patient-specific graph gate:

[
g_i =
\sigma(
MLP[
z_i^{direct}
\Vert
h_i^{graph}
]
)
]

where:

* (\sigma) is the sigmoid function;
* (\Vert) denotes concatenation.

The graph contribution becomes:

[
\tilde{h}_i^{graph}
===================

g_i
\cdot
s_{graph}
\cdot
h_i^{graph}
]

where (s_{graph}) is a learnable global graph scale.

This allows the model to:

* strongly use graph information for some patients;
* reduce it for others;
* nearly ignore noisy neighbors when necessary.

---

## 9.9 Risk output

The final risk combines:

* direct individual-level risk;
* graph-specific risk;
* fused direct and graph representations.

The output remains an unbounded scalar:

[
r_i \in \mathbb{R}
]

A larger score corresponds to higher predicted hazard.

No sigmoid is applied because this is not a binary-classification model.

---

# 10. Improved SurvivalMLP baseline

The project includes an MLP that uses the same multimodal patient-node matrix without graph message passing.

The improved MLP contains:

* input layer normalization;
* nonlinear projection;
* patient-specific feature gating;
* two residual feed-forward blocks;
* dropout;
* a bottleneck risk head;
* an unbounded Cox output.

The MLP is important because it answers the following question:

> Does the graph improve survival prediction beyond the information already present in patient features?

This comparison is essential. A graph model should not be considered superior merely because it is more complex.

---

# 11. Survival loss

## 11.1 Cox partial-likelihood loss

The primary objective is the negative Cox partial log-likelihood.

For each patient (i) with an observed event:

[
\mathcal{L}_{Cox}
=================

*

\sum_{i:E_i=1}
\left[
r_i
---

\log
\left(
\sum_{j:T_j \geq T_i}
\exp(r_j)
\right)
\right]
]

The implementation handles tied event times using a Breslow-style treatment.

Only patients in the training mask contribute to the training loss.

Validation and held-out patients never contribute gradients.

---

## 11.2 Pairwise ranking loss

An optional survival-ranking loss is added.

A valid pair ((i,j)) is created when:

* patient (i) experienced an observed event;
* (T_i < T_j).

The desired ordering is:

[
r_i > r_j
]

The ranking loss uses a smooth softplus penalty:

[
\mathcal{L}_{rank}
==================

softplus(-(r_i-r_j))
]

The total loss is:

[
\mathcal{L}
===========

\mathcal{L}*{Cox}
+
\lambda*{rank}
\mathcal{L}_{rank}
]

The ranking component is generated only from training-fold patients.

---

# 12. Training procedure

The model is optimized using AdamW.

Main training mechanisms include:

* weight decay;
* dropout;
* gradient clipping;
* mixed-precision training when a compatible GPU is available;
* automatic CPU fallback;
* learning-rate reduction on plateau;
* early stopping;
* best-checkpoint restoration.

The learning-rate scheduler monitors either:

```text
validation C-index
```

or:

```text
validation loss
```

depending on configuration.

The preferred checkpoint metric is validation C-index.

Training history records:

* epoch;
* total training loss;
* Cox loss;
* ranking loss;
* training C-index;
* validation Cox loss;
* validation C-index;
* learning rate;
* gradient norm.

The best model is selected only from internal validation performance.

---

# 13. Cross-validation protocol

## 13.1 Outer cross-validation

The notebook uses stratified K-fold cross-validation.

The stratification combines:

* event status;
* quantile-based survival-time groups.

When a combined stratum contains too few observations, the pipeline falls back to event-only stratification.

For each outer fold:

* the held-out partition is used only for final fold evaluation;
* the remaining patients form the outer training set.

---

## 13.2 Internal validation

The outer training set is split again into:

* training patients;
* validation patients.

The split uses survival-aware strata when possible.

The validation set is used for:

* early stopping;
* checkpoint selection;
* learning-rate scheduling.

The held-out fold is never used for these decisions.

---

## 13.3 Fold-specific fitting

Inside every fold:

1. fit the preprocessor on training patients;
2. select genes using training patients;
3. select mutations using training patients;
4. fit clinical encoders using training patients;
5. fit scalers and imputers using training patients;
6. transform all graph nodes with the fitted preprocessor;
7. construct the graph from transformed multimodal similarities;
8. train using only the training mask;
9. select the checkpoint using only the validation mask;
10. evaluate once on the held-out mask.

The fold execution produces risk scores and survival probabilities for held-out patients after training the model with training-only preprocessing.

---

## 13.4 Strict out-of-fold predictions

Predictions from all held-out folds are concatenated.

The notebook verifies that:

* every patient receives exactly one OOF prediction;
* no patient appears twice;
* no patient is missing;
* row order is recoverable;
* survival times match the source analysis table;
* event indicators match the source analysis table;
* risk scores are finite.

These predictions are saved to:

```text
cross_validation/cross_validation_predictions.csv
```

The OOF predictions provide the least biased estimate of internal generalization performance.

---

# 14. Survival-probability estimation

The neural network directly outputs a risk score, not a survival probability.

To obtain survival probabilities:

1. estimate the baseline cumulative hazard from training patients using Breslow;
2. use the same training-risk offset for numerical stability;
3. evaluate the cumulative baseline hazard at each requested horizon;
4. combine it with each patient’s relative risk.

The survival probability is:

[
S_i(t)
======

\exp
\left[
-H_0(t)\exp(r_i)
\right]
]

Typical configured horizons include:

```text
60 months
120 months
```

These correspond approximately to:

* 5-year survival;
* 10-year survival.

The same risk centering derived from the training partition is applied to held-out patients to maintain mathematical consistency.

---

# 15. Evaluation metrics

## 15.1 Concordance index

The C-index evaluates whether higher predicted risk corresponds to earlier observed events.

[
C =
P(r_i > r_j \mid T_i < T_j)
]

Interpretation:

```text
0.50  random ranking
0.60  weak-to-moderate discrimination
0.65  useful prognostic signal
0.70  good discrimination
0.75+ strong discrimination
```

The C-index is the main evaluation metric.

---

## 15.2 Time-dependent AUC

Time-dependent AUC evaluates discrimination at a fixed horizon.

The notebook evaluates horizons such as:

```text
AUC at 60 months
AUC at 120 months
```

These metrics account for censored data using inverse-probability-of-censoring weighting when supported.

Some folds may not support a particular horizon if the estimated censoring-survival function reaches zero before that horizon.

Such failures are recorded rather than silently replaced.

---

## 15.3 Brier score

The Brier score evaluates the squared error of predicted survival probabilities.

Lower values are better.

The notebook calculates:

* Brier score at 60 months;
* Brier score at 120 months;
* integrated Brier score.

The integrated Brier score summarizes prediction error across time.

---

## 15.4 Kaplan–Meier risk stratification

OOF patients are divided into:

* low-risk group;
* high-risk group;

using the median OOF risk score.

The notebook reports:

* number of patients per group;
* number of events per group;
* log-rank statistic;
* log-rank p-value;
* high-risk versus low-risk hazard ratio;
* 95% confidence interval;
* Cox p-value.

A Kaplan–Meier plot is saved as:

```text
kaplan_meier_oof_risk_groups.png
```

The median threshold is used for descriptive stratification and is not presented as a validated clinical decision threshold.

---

## 15.5 Calibration

Patients are divided into quantile groups according to predicted survival probability.

For each group, the notebook compares:

* mean predicted survival;
* Kaplan–Meier observed survival;
* 95% confidence interval;
* absolute calibration error;
* number of patients;
* number of events.

Calibration plots are generated at configured horizons:

```text
calibration_60m.png
calibration_120m.png
```

The notebook also exports:

```text
calibration_summary.csv
```

with mean, weighted and maximum absolute calibration errors.

---

# 16. Ablation study

The notebook compares three models.

## 16.1 Clinical-only MLP

Uses only clinical node features.

Purpose:

> Determine the prognostic information available from conventional clinical variables.

## 16.2 Full-feature MLP

Uses all multimodal node features but no graph.

Purpose:

> Measure the value of multimodal integration without patient–patient message passing.

## 16.3 Full SurvivalGAT

Uses all multimodal node features and the multi-view graph.

Purpose:

> Determine whether graph relationships improve prediction beyond the multimodal feature matrix.

---

# 17. Results

## 17.1 Original SurvivalGAT

The earlier SurvivalGAT achieved:

```text
Overall strict OOF C-index: 0.6734
```

Approximate fold-level summary:

```text
Mean held-out C-index: 0.6725 ± 0.0229
Best fold:            approximately 0.700
Weakest fold:         approximately 0.639
```

Time-dependent evaluation from that run showed approximately:

```text
AUC at 60 months:  0.665
AUC at 120 months: 0.695
Integrated Brier:  0.087
```

These results demonstrated a real prognostic signal with moderate-to-good discrimination.

---

## 17.2 Updated residual gated SurvivalGAT

After introducing:

* a direct multimodal branch;
* two residual GAT blocks;
* patient-specific graph gating;
* learnable global graph scaling;
* edge-attribute encoding;
* improved direct/graph fusion;

the model achieved:

```text
Overall strict OOF C-index: 0.6783
```

Compared with the previous GAT:

```text
Previous GAT: 0.6734
Updated GAT:  0.6783
Absolute gain: +0.0049
```

The improvement is modest but positive.

It suggests that preserving the strong non-graph patient representation and using graph information as a gated residual correction is more effective than forcing all prognostic information through graph message passing.

---

## 17.3 Ablation results

The earlier ablation run produced approximately:

| Model                |      Mean C-index | AUC 60 months | AUC 120 months | Integrated Brier |
| -------------------- | ----------------: | ------------: | -------------: | ---------------: |
| Clinical-only MLP    |     0.614 ± 0.026 |         0.554 |          0.625 |            0.091 |
| Original SurvivalGAT |     0.672 ± 0.023 |         0.665 |          0.695 |            0.087 |
| Full-feature MLP     | **0.701 ± 0.023** |     **0.697** |      **0.724** |        **0.086** |

The main findings were:

1. Multimodal molecular data substantially improved performance compared with clinical features alone.
2. The graph model substantially outperformed the clinical-only MLP.
3. The full-feature MLP remained stronger than the original GAT.
4. The graph topology or message-passing mechanism was therefore a likely limiting factor.
5. The updated gated GAT narrowed the gap slightly by increasing the OOF C-index to `0.6783`.

The improved residual MLP architecture must be re-evaluated in the same cross-validation protocol before reporting a new definitive MLP result.

---

# 18. Scientific interpretation

The results support several conclusions.

## 18.1 Multimodal information is useful

The large improvement from the clinical-only MLP to the full-feature models indicates that:

* expression data;
* mutation features;
* CNA information;
* pathway activity;

contain prognostic information not captured by clinical variables alone.

## 18.2 The graph contains some useful signal

The SurvivalGAT consistently performs better than the clinical-only model.

Therefore, the combined multimodal representation and graph-aware architecture learn a meaningful survival signal.

## 18.3 The current graph is not yet optimal

Because the full-feature MLP performs better than the GAT, the current graph does not yet add enough reliable information to compensate for the complexity of message passing.

Potential reasons include:

* noisy patient connections;
* inappropriate K value;
* graph density that is too high or too low;
* insufficient agreement between views;
* similarity features that do not reflect survival-relevant biology;
* oversmoothing;
* graph heterophily;
* weak transferability of neighbor information;
* subtype-specific relationships not captured by a single global graph.

## 18.4 Graph gating helps

The increase from `0.6734` to `0.6783` supports the decision to make graph information optional and patient dependent.

Some patients may benefit from their molecular neighbors, while others may be better predicted from their own feature profile.

---

# 19. Feature-selection stability

Gene, graph-gene and mutation selections are aggregated across folds.

For every feature, the notebook calculates:

* number of folds in which it was selected;
* selection frequency;
* mean score;
* score standard deviation;
* mean rank;
* median rank;
* stable-feature indicator.

A feature is considered stable when:

```text
selection_frequency >= stable_feature_frequency
```

Outputs include:

```text
gene_selection_stability.csv
graph_gene_selection_stability.csv
mutation_selection_stability.csv
```

These features are described as:

```text
candidate prognostic features
```

They must not automatically be interpreted as:

* causal drivers;
* validated biomarkers;
* treatment targets.

Survival association does not establish causality.

---

# 20. Attention analysis

The notebook can extract attention coefficients from the GAT.

For each edge, the attention table may contain:

* source node;
* target node;
* source patient identifier;
* target patient identifier;
* self-loop status;
* mean attention weight;
* maximum attention across heads;
* attention standard deviation;
* head-specific attention weights;
* source and target events;
* source and target risk scores;
* risk-score difference;
* edge similarities;
* receptor or subtype matches.

Attention weights describe model information flow.

They do not establish:

* biological causality;
* biomarker importance;
* clinical influence;
* treatment response.

A high-risk attention subgraph is generated by:

1. selecting patients with the largest predicted risks;
2. selecting their strongest non-self attention edges;
3. constructing a directed NetworkX graph;
4. coloring nodes by predicted risk;
5. scaling edges by attention.

Output:

```text
high_risk_attention_subgraph.png
```

---

# 21. Final-model training

After cross-validation, a final train/validation split is created.

The final workflow is:

1. fit the preprocessor on the final training subset;
2. transform all patients;
3. construct the final multi-view graph;
4. train the SurvivalGAT on the final training mask;
5. select the checkpoint using final validation C-index;
6. calculate risk scores for all patients;
7. estimate survival probabilities;
8. save the model and all preprocessing artifacts.

The final validation result is not a substitute for the strict OOF estimate.

The OOF C-index should remain the main reported internal-validation performance.

---

# 22. Saved outputs

The pipeline creates several output directories.

A typical structure is:

```text
survival_gat_outputs/
├── checkpoints/
├── cross_validation/
├── artifacts/
├── plots/
├── predictions.csv
├── risk_scores.csv
├── training_history.csv
├── ablation_results.csv
├── ablation_summary.csv
├── calibration_summary.csv
├── kaplan_meier_results.csv
├── attention_weights.csv
├── final_graph_diagnostics.csv
└── artifact_manifest.csv
```

---

## 22.1 Cross-validation files

```text
cross_validation_metrics.csv
```

Fold-level evaluation metrics.

```text
cross_validation_predictions.csv
```

Strict held-out prediction for every patient.

```text
cross_validation_summary.csv
```

Mean, standard deviation, minimum and maximum metrics.

```text
cross_validation_oof_audit.json
```

Verification that every patient was predicted exactly once.

```text
cross_validation_selected_genes.csv
cross_validation_selected_graph_genes.csv
cross_validation_selected_mutations.csv
```

Fold-specific selected molecular features.

```text
cross_validation_graph_diagnostics.csv
```

Fold-specific graph properties.

---

## 22.2 Stability files

```text
gene_selection_stability.csv
graph_gene_selection_stability.csv
mutation_selection_stability.csv
```

Selection consistency across cross-validation folds.

---

## 22.3 Prediction files

```text
predictions.csv
```

Contains:

* row index;
* original row index;
* patient identifier;
* survival time;
* event;
* risk score;
* train/validation label;
* survival probability at each requested horizon.

```text
risk_scores.csv
```

Compact table with patient-level risk estimates.

---

## 22.4 Model artifacts

```text
final_survival_gat.pt
```

Contains:

* trained state dictionary;
* number of node features;
* hidden dimension;
* attention-head count;
* dropout;
* edge dimension;
* edge-feature names;
* modality slices;
* graph residual scale;
* best epoch;
* validation C-index;
* risk horizons;
* training-risk offset;
* configuration.

```text
final_preprocessor.joblib
preprocessors.joblib
```

Contain fitted clinical, expression, mutation, CNA and pathway transformations.

---

## 22.5 Graph artifacts

```text
edge_index.npy
edge_attr.npy
```

Final graph connectivity and multidimensional edge attributes.

```text
reference_X_graph.npy
reference_X_graph_clinical.npy
reference_X_graph_expression.npy
reference_X_graph_pathway.npy
```

Reference graph-input matrices.

```text
final_graph_diagnostics.csv
```

Contains information such as:

* number of nodes;
* number of edges;
* edge dimension;
* graph density;
* mean degree;
* isolated nodes;
* similarity statistics;
* descriptive event homophily.

---

## 22.6 Node and biological metadata

```text
node_feature_names.csv
graph_feature_names.csv
edge_feature_names.csv
selected_genes.csv
selected_graph_genes.csv
selected_mutations.csv
pathway_scores_metadata.csv
patient_metadata.csv
```

These files document the exact representation used by the final model.

---

## 22.7 Survival-calibration artifacts

```text
breslow_event_times.npy
breslow_cumulative_hazard.npy
breslow_risk_offset.npy
```

These are required to convert future risk scores into survival probabilities using the trained baseline hazard.

---

# 23. Reload-consistency test

The notebook performs a complete reload test.

It reloads:

* the preprocessor;
* node features;
* edge index;
* edge attributes;
* model checkpoint;
* reference risk scores.

The SurvivalGAT architecture is reconstructed using checkpoint metadata.

Predictions from the reloaded model are compared against saved reference predictions using numerical tolerances.

The test passes only when:

```python
np.allclose(
    reloaded_risk_scores,
    reference_risk_scores,
    rtol=1e-5,
    atol=1e-6,
)
```

This confirms that the saved artifacts are sufficient to reproduce model risk scores.

---

# 24. GPU compatibility

The notebook checks whether the available CUDA device is compatible with the installed PyTorch build.

A real tensor operation is attempted before enabling GPU execution.

If the GPU architecture is incompatible, the pipeline falls back to CPU rather than failing during training.

This is particularly relevant to Kaggle Tesla P100 sessions when the installed PyTorch build no longer supports CUDA compute capability 6.0.

Mixed-precision training is enabled only when a compatible CUDA device is available.

---

# 25. Reproducibility

The notebook controls random seeds for:

* Python;
* NumPy;
* PyTorch;
* CUDA, when available;
* cross-validation;
* train/validation splitting;
* graph visualization.

The configuration and software versions are exported.

Reproducibility is improved by saving:

* fold predictions;
* selected features;
* graph diagnostics;
* preprocessing objects;
* model parameters;
* risk-offset information;
* Breslow baseline hazard;
* complete artifact manifest.

Exact bitwise reproducibility may still depend on:

* CUDA kernels;
* PyTorch version;
* PyTorch Geometric version;
* hardware;
* nondeterministic sparse operations.

---

# 26. Installation

The notebook uses libraries including:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
networkx
joblib
torch
torch-geometric
lifelines
scikit-survival
```

`torch-geometric` is required for the graph model.

`lifelines` is used for:

* Kaplan–Meier estimation;
* log-rank testing;
* Cox hazard-ratio estimation;
* calibration.

`scikit-survival` is optional and is used when available for:

* Cox elastic-net selection;
* time-dependent AUC;
* Brier score;
* integrated Brier score.

---

# 27. Running the notebook

## Step 1 — Add the dataset

Attach the Kaggle dataset containing:

```text
METABRIC_RNA_Mutation.csv
```

## Step 2 — Select the accelerator

A CPU runtime is supported.

A compatible GPU can reduce training time.

## Step 3 — Run cells in order

The notebook defines dependent classes and functions progressively.

Important components must be defined before use:

1. imports and package checks;
2. device selection;
3. configuration;
4. dataset loading;
5. leakage detection;
6. feature detection;
7. fold-specific preprocessing;
8. multi-view graph construction;
9. graph data and Cox loss;
10. SurvivalGAT and metrics;
11. training and fold execution;
12. cross-validation;
13. ablation;
14. calibration and interpretation;
15. final training and export;
16. reload test;
17. artifact manifest.

Running cells out of order can leave an older function or architecture active in memory.

Restarting the kernel and using “Run All” is recommended after major architectural changes.

---

# 28. Main configuration groups

Important parameters include:

## Cross-validation

```python
CONFIG["cv_folds"]
CONFIG["validation_size_within_train"]
CONFIG["seed"]
CONFIG["run_cross_validation"]
```

## Feature selection

```python
CONFIG["gene_selection_method"]
CONFIG["top_k_node_genes"]
CONFIG["top_k_graph_genes"]
CONFIG["top_k_gene_candidates"]
CONFIG["cox_elastic_net_alpha"]
CONFIG["cox_elastic_net_l1_ratio"]
CONFIG["stable_feature_frequency"]
```

## Graph

```python
CONFIG["graph_mode"]
CONFIG["knn_neighbors"]
CONFIG["knn_graph_mode"]
CONFIG["minimum_agreeing_views"]
CONFIG["minimum_edge_similarity"]
CONFIG["alpha_clinical"]
CONFIG["alpha_expression"]
CONFIG["alpha_pathway"]
```

## Edge attributes

```python
CONFIG["include_same_pam50_edge_feature"]
CONFIG["include_same_er_edge_feature"]
CONFIG["include_same_her2_edge_feature"]
CONFIG["use_multidimensional_edge_attr"]
```

## Model

```python
CONFIG["hidden_channels"]
CONFIG["gat_heads"]
CONFIG["dropout"]
CONFIG["initial_residual_scale"]
```

## Optimization

```python
CONFIG["learning_rate"]
CONFIG["weight_decay"]
CONFIG["gradient_clip_norm"]
CONFIG["patience"]
CONFIG["max_epochs"]
CONFIG["cv_max_epochs"]
CONFIG["checkpoint_metric"]
```

## Survival loss

```python
CONFIG["survival_loss_type"]
CONFIG["ranking_loss_weight"]
```

## Evaluation

```python
CONFIG["risk_horizons_months"]
CONFIG["run_ablation_study"]
```

---

# 29. Limitations

## 29.1 Internal validation only

Cross-validation estimates internal generalization within METABRIC.

It does not establish external clinical validity.

An independent cohort is required.

Possible external cohorts include other breast-cancer datasets with compatible:

* clinical definitions;
* expression genes;
* survival endpoints;
* mutation formats;
* measurement platforms.

---

## 29.2 Platform shift

Gene-expression values may differ across:

* microarray platforms;
* RNA-seq;
* normalization procedures;
* laboratories;
* batches.

External validation requires careful gene matching and normalization.

---

## 29.3 Graph transductivity

The current graph may include all patients as nodes while training loss is restricted to training nodes.

The held-out outcomes are not used, but held-out covariates may influence graph topology in a transductive setting.

This should be reported clearly.

For a fully inductive clinical deployment, a new patient should be connected to a fixed training/reference graph without rebuilding the complete model using future cohort information.

---

## 29.4 No causal interpretation

Selected genes, mutations, pathways and attention edges are associative.

They are not automatically:

* causal;
* mechanistically validated;
* clinically actionable;
* treatment-predictive.

Experimental and external clinical validation are necessary.

---

## 29.5 Hyperparameter optimization

Repeatedly modifying the architecture after inspecting cross-validation results can indirectly overfit the research process to METABRIC.

A final model specification should therefore be frozen and evaluated on an untouched external cohort.

---

## 29.6 Time-dependent metric limitations

IPCW-based metrics may fail at late horizons when censoring survival approaches zero.

Evaluation horizons must remain within a statistically supported follow-up interval.

---

## 29.7 Clinical threshold

The median risk score creates balanced Kaplan–Meier groups but is not a clinically validated cutoff.

A clinical threshold would require:

* independent training;
* decision-curve analysis;
* sensitivity/specificity assessment;
* external validation;
* prospective evaluation.

---

# 30. Recommended next experiments

## 30.1 Graph-topology optimization

Evaluate using the same folds:

```text
weighted_union
multi_view_consensus
clinical_knn
expression_knn
pathway_knn
```

Test a limited predefined grid of:

* K-neighbor values;
* similarity thresholds;
* graph weights;
* minimum agreeing views;
* union versus mutual KNN.

Selection must use validation folds only.

---

## 30.2 Stronger graph ablation

Evaluate:

* no graph;
* clinical-only graph;
* expression-only graph;
* pathway-only graph;
* clinical + expression;
* clinical + pathway;
* expression + pathway;
* all views.

This identifies which relationships contribute useful signal.

---

## 30.3 Graph regularization

Potential additions include:

* stochastic edge dropout;
* graph-layer dropout;
* consistency regularization;
* attention entropy control;
* neighbor-sampling robustness;
* reduced graph depth.

---

## 30.4 Transfer learning

A promising strategy is:

1. train the strong full-feature SurvivalMLP;
2. copy its encoder weights into the direct branch of SurvivalGAT;
3. freeze the encoder;
4. train the graph layers and graph gate;
5. progressively unfreeze the encoder;
6. fine-tune with a smaller learning rate.

A stronger external transfer-learning strategy could pretrain a transcriptomic encoder on a larger independent breast-cancer or pan-cancer cohort.

Any supervised adaptation must remain fold specific.

---

## 30.5 External validation

The final scientific objective should be:

```text
Train and tune on METABRIC
            ↓
Freeze preprocessing and model
            ↓
Evaluate on an independent cohort
```

External validation should report:

* C-index;
* time-dependent AUC;
* Brier score;
* calibration;
* Kaplan–Meier separation;
* subgroup performance;
* missing-feature handling;
* dataset shift.

---

# 31. Recommended reporting statement

A balanced scientific summary is:

> We developed a leakage-aware multimodal survival-analysis pipeline for breast cancer using clinical, gene-expression, mutation, copy-number and pathway features from METABRIC. Patients were represented as nodes in a multi-view similarity graph constructed from clinical, expression and pathway spaces. An edge-aware residual Graph Attention Network was trained using Cox partial-likelihood and pairwise ranking losses. Strict out-of-fold evaluation produced an overall C-index of 0.6783 for the updated SurvivalGAT. Ablation experiments showed that multimodal molecular information substantially improved prediction compared with a clinical-only MLP, while the full-feature MLP remained the strongest baseline in the earlier experiment. These results indicate that the graph contains useful but currently suboptimal information. External validation is required before any clinical interpretation or deployment.

---

# 32. Current project status

```text
Dataset preprocessing:              Implemented
Outcome cleaning:                   Implemented
Leakage detection:                  Implemented
Clinical encoding:                  Implemented
Expression selection:               Implemented
Mutation selection:                 Implemented
CNA preprocessing:                  Implemented
Pathway scoring:                    Implemented
Multi-view graph:                   Implemented
Multidimensional edge attributes:   Implemented
Residual gated SurvivalGAT:         Implemented
Cox loss:                           Implemented
Ranking loss:                       Implemented
Nested validation:                  Implemented
Strict OOF prediction:              Implemented
Time-dependent metrics:             Implemented
Kaplan–Meier analysis:              Implemented
Calibration:                        Implemented
Feature stability:                  Implemented
Attention extraction:               Implemented
Ablation study:                     Implemented
Model serialization:                Implemented
Reload consistency test:            Implemented
External validation:                Not yet performed
Transfer learning:                  Proposed future extension
Clinical deployment:                Not supported
```

---

# 33. Final conclusion

This notebook is not only a GAT training script.

It is a complete multimodal survival-research framework that combines:

* clinical epidemiology;
* genomic feature engineering;
* pathway biology;
* graph construction;
* graph neural networks;
* censoring-aware loss functions;
* strict cross-validation;
* survival calibration;
* interpretable outputs;
* reproducible artifact export.

The updated SurvivalGAT achieved:

```text
Strict OOF C-index = 0.6783
```

This represents a real, stable prognostic signal and a small improvement over the previous GAT architecture.

The ablation study demonstrates that multimodal molecular information is valuable. However, the graph does not yet outperform the strongest full-feature MLP baseline.

The scientifically correct conclusion is therefore:

> The current patient graph provides additional modeling capacity and interpretable patient–patient information flow, but further optimization and external validation are needed before claiming that graph message passing improves survival prediction beyond a strong multimodal non-graph model.
