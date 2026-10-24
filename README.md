Leakage-Safe Feature Selection for ID3 and k-NN: Measuring Performance Inflation and Feature Stability on High-Dimensional Genomic Data

Research Area: Machine Learning · Feature Selection · Data Leakage · High-Dimensional Classification

Project Type: Reproducible Research

⸻

Abstract

Feature selection is an important step in machine learning, particularly when classification problems involve a large number of features and relatively few samples. However, performing feature selection before cross-validation can introduce data leakage, allowing information from evaluation samples to influence the selected features and potentially inflate reported predictive performance.

This research investigates the impact of feature-selection data leakage on classification performance by comparing leakage-safe and intentionally leaky evaluation pipelines. The experimental framework considers ID3-style decision trees and k-nearest neighbors (k-NN), with feature-selection methods such as the ANOVA F-test and Mutual Information.

The study aims to examine differences in predictive performance between the evaluation pipelines and investigate feature-selection stability where supported by the experimental methodology. The implementation is intended to support transparent experimentation and reproducible evaluation in high-dimensional classification settings.

Note: The abstract describes the research objective and methodology. Specific findings will be reported after the experimental results have been verified.

⸻

1. Research Motivation

In machine learning, feature selection identifies the variables most relevant to a prediction task. When the number of features is very large relative to the number of available samples, selecting informative features becomes particularly important.

A potential methodological problem arises when feature selection is performed on the entire dataset before splitting the data into training and evaluation folds. Information from evaluation samples can influence feature selection, resulting in an overly optimistic estimate of model performance.

This project investigates the effect of that methodological error and demonstrates why feature selection should be incorporated correctly into the model evaluation pipeline.

⸻

2. Research Objectives

The objectives of this study are:

* To compare leakage-safe and intentionally leaky feature-selection pipelines.
* To investigate how feature-selection leakage affects classification performance.
* To evaluate ID3-style decision trees and k-nearest neighbors (k-NN).
* To examine the behavior of ANOVA F-test and Mutual Information feature selection.
* To investigate feature-selection stability, if supported by the implemented experiments.
* To provide a reproducible experimental framework for studying data leakage in high-dimensional classification.

⸻

3. Methodology

The experimental framework is designed around two evaluation approaches.

3.1 Leakage-Safe Feature Selection

Feature selection is fitted exclusively on the training portion of each cross-validation fold. The selected features are then used to train the classifier and evaluate its performance on the corresponding held-out fold.

This approach prevents the held-out fold from influencing feature selection.

3.2 Intentionally Leaky Feature Selection

Feature selection is performed before cross-validation using information from the full dataset. The selected features are subsequently used in the cross-validation experiments.

This approach serves as a methodological control to investigate how using evaluation-set information during feature selection can affect reported performance. It is not a recommended model-evaluation procedure.

3.3 Feature-Selection Methods

The study considers the following methods, subject to the final implementation:

* ANOVA F-test: Evaluates feature relevance using an analysis-of-variance statistic.
* Mutual Information: Estimates the statistical dependence between individual features and the target variable.

3.4 Classification Algorithms

* ID3-style Decision Tree: A decision-tree classifier based on information gain, as implemented in this project.
* k-Nearest Neighbors (k-NN): An instance-based classification algorithm that predicts a class using nearby training samples.

The implementation details, hyperparameters, and preprocessing procedures will be documented alongside the experimental code.

⸻

4. Experimental Evaluation

The experiments are designed to compare the two feature-selection pipelines across the implemented feature-selection methods and classifiers.

Evaluation may include:

* Accuracy
* Precision
* Recall
* F1-score
* Variation in performance across repeated evaluations
* Feature-selection stability, if implemented

The final report will contain the actual experimental results, including appropriate summary statistics and visualizations.

No numerical findings are asserted in this README until they have been verified against the experimental outputs.

⸻

5. Repository Structure

The repository will contain the implementation and supporting research materials.

feature-selection-leakage-study/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
├── experiments/
├── notebooks/
├── results/
└── docs/

The final directory structure will reflect the actual files included in the repository.

⸻

6. Reproducibility

The project aims to make its experiments reproducible by documenting:

* Software dependencies and versions
* Dataset provenance and access instructions
* Preprocessing procedures
* Feature-selection settings
* Classifier configurations
* Cross-validation strategy
* Random seeds, where applicable
* Experimental commands and evaluation metrics

Reproduction instructions will be added after the execution scripts and environment configuration have been finalized.

⸻

7. Dataset and Data Usage

Dataset provenance, access conditions, and applicable usage terms must be established before the dataset is redistributed or used in a public demonstration.

This repository will document the dataset source and provide appropriate access instructions when those details have been verified.

The repository will not redistribute data without the necessary authorization.

⸻

8. Research Limitations

The conclusions of this study must be interpreted in the context of the datasets, algorithms, preprocessing procedures, and validation strategies actually evaluated.

Experiments on a limited number of datasets do not establish that the same performance effects will occur in every machine-learning setting.

The distinction between leakage-safe and intentionally leaky feature selection is intended to demonstrate a methodological issue, not to suggest that the leaky procedure is appropriate for real-world evaluation.

⸻

9. Research Status

This repository documents the implementation and experimental investigation associated with the research project.

Experimental results, reproducibility instructions, and supporting research materials will be added or updated as they are verified.

⸻

10. Citation

If you use or discuss this implementation, please cite the repository and the associated research work when a formal citation becomes available.

The final citation information will be provided after the authorship details and publication status have been established.

⸻

11. Contact and Collaboration

Researchers and students interested in feature selection, machine-learning evaluation, and reproducible experimentation are welcome to explore the project and discuss its methodology.

For questions, suggestions, or research collaboration, please use the repository’s GitHub Issues or Discussions features when available.
