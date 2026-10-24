"""Regenerate the Phase 7 paired analysis from saved fold-level results.

The analysis is intentionally descriptive. The bootstrap intervals resample
the 15 paired fold rows and are not inferential confidence intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MANIFEST_FILE = PROJECT_ROOT / "configs" / "outer_fold_manifest.csv"

DEFAULT_ID3_RESULTS = TABLES_DIR / "shared_3x5_id3_results.csv"
DEFAULT_ID3_SUMMARY = TABLES_DIR / "shared_3x5_id3_summary.csv"
DEFAULT_KNN_RESULTS = (
    TABLES_DIR / "shared_3x5_knn_phase6_corrected_results.csv"
)
DEFAULT_KNN_SUMMARY = (
    TABLES_DIR / "shared_3x5_knn_phase6_corrected_summary.csv"
)

BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 2_026_1024
INTERVAL_NOTE = (
    "Descriptive percentile bootstrap over paired fold rows; "
    "repeated-CV dependence prevents inferential interpretation."
)
METRICS = ["Macro-F1", "Balanced accuracy"]
FOLD_METRICS = ["Balanced accuracy", "Macro-F1"]
SELECTOR_ORDER = ["ANOVA", "MI"]
CLASSIFIER_ORDER = ["ID3", "k-NN"]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def _require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    description: str,
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(
            f"{description} is missing required columns: {', '.join(missing)}"
        )


def _canonicalize_results(
    frame: pd.DataFrame,
    classifier: str,
) -> pd.DataFrame:
    if classifier == "ID3":
        _require_columns(
            frame,
            {
                "selector",
                "selection_mode",
                "repeat",
                "fold",
                "global_fold",
                "features_selected",
                "n_train",
                "n_test",
                "macro_f1",
                "balanced_accuracy",
            },
            "ID3 results",
        )
        selector_map = {"ANOVA": "ANOVA", "MI": "MI"}
        canonical = frame.copy()
        canonical["selector"] = canonical["selector"].map(selector_map)
    elif classifier == "k-NN":
        _require_columns(
            frame,
            {
                "selector",
                "selection_mode",
                "repeat",
                "fold",
                "global_fold",
                "features_selected",
                "n_train",
                "n_test",
                "macro_f1",
                "balanced_accuracy",
            },
            "corrected k-NN results",
        )
        selector_map = {
            "anova": "ANOVA",
            "mutual_information": "MI",
        }
        canonical = frame.copy()
        canonical["selector"] = canonical["selector"].map(selector_map)
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")

    if canonical["selector"].isna().any():
        bad = sorted(frame.loc[canonical["selector"].isna(), "selector"].unique())
        raise ValueError(
            f"{classifier} results contain unknown selector values: {bad}"
        )

    canonical["classifier"] = classifier
    canonical["selector_key"] = canonical["selector"].map(
        {"ANOVA": "anova", "MI": "mutual_information"}
    )
    canonical["metric_values"] = list(
        zip(canonical["macro_f1"], canonical["balanced_accuracy"])
    )
    return canonical


def validate_model_results(frame: pd.DataFrame, classifier: str) -> None:
    """Validate the shared 3x5 fold-level result contract."""

    expected_keys = {
        (selector, mode, fold)
        for selector in SELECTOR_ORDER
        for mode in ("safe", "leaky")
        for fold in range(1, 16)
    }
    actual_keys = set(
        zip(frame["selector"], frame["selection_mode"], frame["global_fold"])
    )
    if actual_keys != expected_keys:
        raise ValueError(
            f"{classifier} results must contain exactly 60 unique "
            "selector/mode/global_fold rows."
        )
    if frame[["selector", "selection_mode", "global_fold"]].duplicated().any():
        raise ValueError(f"{classifier} results contain duplicate fold rows.")
    if not frame["repeat"].between(1, 5).all():
        raise ValueError(f"{classifier} repeat values must be in 1..5.")
    if not frame["fold"].between(1, 3).all():
        raise ValueError(f"{classifier} fold values must be in 1..3.")
    if not frame["n_train"].eq(48).all() or not frame["n_test"].eq(24).all():
        raise ValueError(f"{classifier} results must use 48/24 train/test rows.")
    if not frame["features_selected"].eq(50).all():
        raise ValueError(f"{classifier} results must select exactly 50 features.")
    for column in ("macro_f1", "balanced_accuracy"):
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not ((0 <= values) & (values <= 1)).all():
            raise ValueError(f"{classifier} {column} values must be finite in [0, 1].")

    metadata = frame[
        ["global_fold", "repeat", "fold", "n_train", "n_test"]
    ].drop_duplicates()
    if len(metadata) != 15:
        raise ValueError(f"{classifier} has inconsistent fold metadata.")
    expected_metadata = pd.DataFrame(
        {
            "global_fold": np.arange(1, 16),
            "repeat": np.repeat(np.arange(1, 6), 3),
            "fold": np.tile(np.arange(1, 4), 5),
            "n_train": 48,
            "n_test": 24,
        }
    )
    merged = expected_metadata.merge(
        metadata,
        on=["global_fold", "repeat", "fold", "n_train", "n_test"],
        how="outer",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"{classifier} fold metadata does not match 3x5 protocol.")


def validate_source_summary(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    classifier: str,
) -> None:
    """Check the model's existing summary against its fold-level results."""

    _require_columns(
        summary,
        {
            "selector",
            "selection_mode",
            "folds",
            "macro_f1_mean",
            "macro_f1_std",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
        },
        f"{classifier} summary",
    )
    expected = (
        results.groupby(["selector", "selection_mode"], sort=True)
        .agg(
            folds=("macro_f1", "count"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
        )
        .reset_index()
    )
    actual = summary.copy()
    actual["selector"] = actual["selector"].replace(
        {"anova": "ANOVA", "mutual_information": "MI"}
    )
    actual = actual[
        [
            "selector",
            "selection_mode",
            "folds",
            "macro_f1_mean",
            "macro_f1_std",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
        ]
    ].sort_values(["selector", "selection_mode"]).reset_index(drop=True)
    expected = expected.sort_values(["selector", "selection_mode"]).reset_index(
        drop=True
    )
    if not np.allclose(
        actual.iloc[:, 2:].to_numpy(dtype=float),
        expected.iloc[:, 2:].to_numpy(dtype=float),
        rtol=0,
        atol=1e-14,
    ):
        raise ValueError(f"{classifier} summary does not match fold-level results.")


def _bootstrap_interval(
    differences: np.ndarray,
    rng: np.random.Generator,
    resamples: int,
) -> tuple[float, float]:
    samples = rng.choice(
        differences,
        size=(resamples, len(differences)),
        replace=True,
    )
    means = samples.mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _difference_summary(
    differences: np.ndarray,
    rng: np.random.Generator,
    resamples: int,
) -> dict[str, float | int]:
    low, high = _bootstrap_interval(differences, rng, resamples)
    return {
        "n_paired_folds": len(differences),
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "difference_std": float(np.std(differences, ddof=1)),
        "bootstrap_95_percentile_low": low,
        "bootstrap_95_percentile_high": high,
        "positive_folds": int(np.sum(differences > 0)),
        "zero_folds": int(np.sum(differences == 0)),
        "negative_folds": int(np.sum(differences < 0)),
    }


def _metric_value(row: pd.Series, metric: str) -> float:
    return float(row["macro_f1"] if metric == "Macro-F1" else row["balanced_accuracy"])


def _paired_rows(
    id3: pd.DataFrame,
    knn: pd.DataFrame,
    rng: np.random.Generator,
    resamples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for classifier, frame in (("ID3", id3), ("k-NN", knn)):
        for selector in SELECTOR_ORDER:
            subset = frame[frame["selector"] == selector]
            for metric in METRICS:
                safe = subset[subset["selection_mode"] == "safe"].set_index(
                    "global_fold"
                )
                leaky = subset[subset["selection_mode"] == "leaky"].set_index(
                    "global_fold"
                )
                folds = sorted(set(safe.index) & set(leaky.index))
                differences = np.array(
                    [
                        _metric_value(leaky.loc[fold], metric)
                        - _metric_value(safe.loc[fold], metric)
                        for fold in folds
                    ]
                )
                interval_low, interval_high = _bootstrap_interval(
                    differences, rng, resamples
                )
                for fold in folds:
                    safe_row = safe.loc[fold]
                    leaky_row = leaky.loc[fold]
                    rows.append(
                        {
                            "classifier": classifier,
                            "selector": selector,
                            "selector_key": safe_row["selector_key"],
                            "repeat": int(safe_row["repeat"]),
                            "fold": int(safe_row["fold"]),
                            "global_fold": int(fold),
                            "metric": metric,
                            "safe_value": _metric_value(safe_row, metric),
                            "leaky_value": _metric_value(leaky_row, metric),
                            "leaky_minus_safe": float(
                                _metric_value(leaky_row, metric)
                                - _metric_value(safe_row, metric)
                            ),
                            "difference_definition": "leaky_minus_safe",
                            "bootstrap_95_percentile_low": interval_low,
                            "bootstrap_95_percentile_high": interval_high,
                        }
                    )
                summary = _difference_summary_no_bootstrap(differences)
                summary["bootstrap_95_percentile_low"] = interval_low
                summary["bootstrap_95_percentile_high"] = interval_high
                summary.update(
                    {
                        "classifier": classifier,
                        "selector": selector,
                        "selector_key": safe_row["selector_key"],
                        "metric": metric,
                        "safe_mean": float(
                            np.mean(
                                [
                                    _metric_value(safe.loc[fold], metric)
                                    for fold in folds
                                ]
                            )
                        ),
                        "leaky_mean": float(
                            np.mean(
                                [
                                    _metric_value(leaky.loc[fold], metric)
                                    for fold in folds
                                ]
                            )
                        ),
                        "difference_definition": "leaky_minus_safe",
                        "interval_note": INTERVAL_NOTE,
                    }
                )
                summaries.append(summary)

    # _difference_summary above would consume a second bootstrap stream. The
    # saved Phase 7 stream consumes exactly one interval per comparison, so
    # correct the output by calculating descriptive fields directly.
    summary_columns = [
        "classifier",
        "selector",
        "selector_key",
        "metric",
        "n_paired_folds",
        "safe_mean",
        "leaky_mean",
        "mean_difference",
        "median_difference",
        "difference_std",
        "bootstrap_95_percentile_low",
        "bootstrap_95_percentile_high",
        "positive_folds",
        "zero_folds",
        "negative_folds",
        "difference_definition",
        "interval_note",
    ]
    rows_frame = pd.DataFrame(rows)
    rows_frame["metric"] = pd.Categorical(
        rows_frame["metric"], categories=FOLD_METRICS, ordered=True
    )
    rows_frame = rows_frame.sort_values(
        ["classifier", "selector", "metric", "global_fold"]
    ).reset_index(drop=True)
    rows_frame["metric"] = rows_frame["metric"].astype(str)
    summary_frame = pd.DataFrame(summaries)[summary_columns]
    summary_frame["metric"] = pd.Categorical(
        summary_frame["metric"], categories=METRICS, ordered=True
    )
    summary_frame = summary_frame.sort_values(
        ["classifier", "selector", "metric"]
    ).reset_index(drop=True)
    summary_frame["metric"] = summary_frame["metric"].astype(str)
    return rows_frame, summary_frame


def _model_rows(
    id3: pd.DataFrame,
    knn: pd.DataFrame,
    rng: np.random.Generator,
    resamples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for selector in SELECTOR_ORDER:
        id3_safe = id3[
            (id3["selector"] == selector) & (id3["selection_mode"] == "safe")
        ].set_index("global_fold")
        knn_safe = knn[
            (knn["selector"] == selector) & (knn["selection_mode"] == "safe")
        ].set_index("global_fold")
        for metric in METRICS:
            folds = sorted(set(id3_safe.index) & set(knn_safe.index))
            differences = np.array(
                [
                    _metric_value(knn_safe.loc[fold], metric)
                    - _metric_value(id3_safe.loc[fold], metric)
                    for fold in folds
                ]
            )
            interval_low, interval_high = _bootstrap_interval(
                differences, rng, resamples
            )
            for fold in folds:
                rows.append(
                    {
                        "selector": selector,
                        "selector_key": id3_safe.loc[fold]["selector_key"],
                        "repeat": int(id3_safe.loc[fold]["repeat"]),
                        "fold": int(id3_safe.loc[fold]["fold"]),
                        "global_fold": int(fold),
                        "metric": metric,
                        "id3_safe_value": _metric_value(id3_safe.loc[fold], metric),
                        "knn_safe_value": _metric_value(knn_safe.loc[fold], metric),
                        "knn_minus_id3": float(differences[folds.index(fold)]),
                        "difference_definition": "kNN_safe_minus_ID3_safe",
                        "bootstrap_95_percentile_low": interval_low,
                        "bootstrap_95_percentile_high": interval_high,
                    }
                )
            stats = _difference_summary_no_bootstrap(differences)
            stats.update(
                {
                    "selector": selector,
                    "selector_key": id3_safe.iloc[0]["selector_key"],
                    "metric": metric,
                    "id3_safe_mean": float(
                        np.mean([_metric_value(id3_safe.loc[f], metric) for f in folds])
                    ),
                    "knn_safe_mean": float(
                        np.mean([_metric_value(knn_safe.loc[f], metric) for f in folds])
                    ),
                    "difference_definition": "kNN_safe_minus_ID3_safe",
                    "interval_note": INTERVAL_NOTE,
                    "bootstrap_95_percentile_low": interval_low,
                    "bootstrap_95_percentile_high": interval_high,
                    "knn_higher_folds": stats.pop("positive_folds"),
                    "equal_folds": stats.pop("zero_folds"),
                    "knn_lower_folds": stats.pop("negative_folds"),
                }
            )
            summaries.append(stats)
    columns = [
        "selector",
        "selector_key",
        "metric",
        "n_paired_folds",
        "id3_safe_mean",
        "knn_safe_mean",
        "mean_difference",
        "median_difference",
        "difference_std",
        "bootstrap_95_percentile_low",
        "bootstrap_95_percentile_high",
        "knn_higher_folds",
        "equal_folds",
        "knn_lower_folds",
        "difference_definition",
        "interval_note",
    ]
    rows_frame = pd.DataFrame(rows)
    rows_frame["metric"] = pd.Categorical(
        rows_frame["metric"], categories=FOLD_METRICS, ordered=True
    )
    rows_frame = rows_frame.sort_values(
        ["selector", "metric", "global_fold"]
    ).reset_index(drop=True)
    rows_frame["metric"] = rows_frame["metric"].astype(str)
    summary_frame = pd.DataFrame(summaries)[columns]
    summary_frame["metric"] = pd.Categorical(
        summary_frame["metric"], categories=METRICS, ordered=True
    )
    summary_frame = summary_frame.sort_values(
        ["selector", "metric"]
    ).reset_index(drop=True)
    summary_frame["metric"] = summary_frame["metric"].astype(str)
    return rows_frame, summary_frame


def _difference_summary_no_bootstrap(differences: np.ndarray) -> dict[str, object]:
    return {
        "n_paired_folds": len(differences),
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "difference_std": float(np.std(differences, ddof=1)),
        "positive_folds": int(np.sum(differences > 0)),
        "zero_folds": int(np.sum(differences == 0)),
        "negative_folds": int(np.sum(differences < 0)),
    }


def _selector_rows(
    leakage: pd.DataFrame,
    rng: np.random.Generator,
    resamples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for classifier in CLASSIFIER_ORDER:
        frame = leakage[leakage["classifier"] == classifier]
        for metric in METRICS:
            differences_by_selector = {}
            for selector in SELECTOR_ORDER:
                subset = frame[frame["selector"] == selector]
                safe = subset[subset["selection_mode"] == "safe"].set_index(
                    "global_fold"
                )
                leaky = subset[subset["selection_mode"] == "leaky"].set_index(
                    "global_fold"
                )
                differences_by_selector[selector] = {
                    fold: _metric_value(leaky.loc[fold], metric)
                    - _metric_value(safe.loc[fold], metric)
                    for fold in sorted(set(safe.index) & set(leaky.index))
                }
            folds = sorted(
                set(differences_by_selector["ANOVA"])
                & set(differences_by_selector["MI"])
            )
            anova = np.array([differences_by_selector["ANOVA"][f] for f in folds])
            mi = np.array([differences_by_selector["MI"][f] for f in folds])
            differences = mi - anova
            interval_low, interval_high = _bootstrap_interval(
                differences, rng, resamples
            )
            for fold, anova_value, mi_value, difference in zip(
                folds, anova, mi, differences
            ):
                row = frame[frame["global_fold"] == fold].iloc[0]
                rows.append(
                    {
                        "classifier": classifier,
                        "repeat": int(row["repeat"]),
                        "fold": int(row["fold"]),
                        "global_fold": int(fold),
                        "metric": metric,
                        "anova_leakage_difference": float(anova_value),
                        "mi_leakage_difference": float(mi_value),
                        "mi_minus_anova": float(difference),
                        "difference_definition": (
                            "MI_leakage_difference_minus_ANOVA_leakage_difference"
                        ),
                        "bootstrap_95_percentile_low": interval_low,
                        "bootstrap_95_percentile_high": interval_high,
                    }
                )
            stats = _difference_summary_no_bootstrap(differences)
            summaries.append(
                {
                    "classifier": classifier,
                    "metric": metric,
                    "n_paired_folds": stats["n_paired_folds"],
                    "anova_leakage_mean": float(np.mean(anova)),
                    "mi_leakage_mean": float(np.mean(mi)),
                    "mi_minus_anova_mean_difference": stats["mean_difference"],
                    "mi_minus_anova_median_difference": stats["median_difference"],
                    "difference_std": stats["difference_std"],
                    "bootstrap_95_percentile_low": interval_low,
                    "bootstrap_95_percentile_high": interval_high,
                    "mi_higher_folds": stats["positive_folds"],
                    "equal_folds": stats["zero_folds"],
                    "mi_lower_folds": stats["negative_folds"],
                    "difference_definition": (
                        "MI_leakage_difference_minus_ANOVA_leakage_difference"
                    ),
                    "interval_note": INTERVAL_NOTE,
                }
            )
    columns = [
        "classifier",
        "metric",
        "n_paired_folds",
        "anova_leakage_mean",
        "mi_leakage_mean",
        "mi_minus_anova_mean_difference",
        "mi_minus_anova_median_difference",
        "difference_std",
        "bootstrap_95_percentile_low",
        "bootstrap_95_percentile_high",
        "mi_higher_folds",
        "equal_folds",
        "mi_lower_folds",
        "difference_definition",
        "interval_note",
    ]
    rows_frame = pd.DataFrame(rows)
    rows_frame["metric"] = pd.Categorical(
        rows_frame["metric"], categories=FOLD_METRICS, ordered=True
    )
    rows_frame = rows_frame.sort_values(
        ["classifier", "metric", "global_fold"]
    ).reset_index(drop=True)
    rows_frame["metric"] = rows_frame["metric"].astype(str)
    summary_frame = pd.DataFrame(summaries)[columns]
    summary_frame["metric"] = pd.Categorical(
        summary_frame["metric"], categories=METRICS, ordered=True
    )
    summary_frame = summary_frame.sort_values(
        ["classifier", "metric"]
    ).reset_index(drop=True)
    summary_frame["metric"] = summary_frame["metric"].astype(str)
    return rows_frame, summary_frame


def _write_figures(
    results: pd.DataFrame,
    leakage: pd.DataFrame,
    model: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    safe_leaky = results.groupby(
        ["classifier", "selector", "selection_mode"], sort=False
    )[["macro_f1", "balanced_accuracy"]].mean().reset_index()
    labels = [
        f"{row.classifier}\n{row.selector}"
        for row in safe_leaky[
            ["classifier", "selector"]
        ].drop_duplicates().itertuples()
    ]
    x = np.arange(len(labels))
    width = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, column, title in zip(
        axes,
        ["macro_f1", "balanced_accuracy"],
        ["Macro-F1", "Balanced accuracy"],
    ):
        for offset, mode in enumerate(["safe", "leaky"]):
            values = []
            for classifier, selector in [
                ("ID3", "ANOVA"),
                ("ID3", "MI"),
                ("k-NN", "ANOVA"),
                ("k-NN", "MI"),
            ]:
                values.append(
                    safe_leaky[
                        (safe_leaky.classifier == classifier)
                        & (safe_leaky.selector == selector)
                        & (safe_leaky.selection_mode == mode)
                    ][column].iloc[0]
                )
            axis.bar(x + (offset - 0.5) * width, values, width, label=mode)
        axis.set_title(title)
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Mean test-fold score")
        axis.set_xticks(x, labels)
        axis.legend()
    path = output_dir / "phase7_safe_leaky_performance.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for index, (classifier, selector) in enumerate(
        [
            ("ID3", "ANOVA"),
            ("ID3", "MI"),
            ("k-NN", "ANOVA"),
            ("k-NN", "MI"),
        ]
    ):
        subset = leakage[
            (leakage.classifier == classifier)
            & (leakage.selector == selector)
            & (leakage.metric == "Macro-F1")
        ].sort_values("global_fold")
        axis.plot(
            subset.global_fold,
            subset.leaky_minus_safe,
            marker="o",
            linewidth=1,
            label=f"{classifier} + {selector}",
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("Fold-level leakage differences (leaky minus safe)")
    axis.set_xlabel("Shared outer fold")
    axis.set_ylabel("Macro-F1 difference")
    axis.legend(ncol=2)
    path = output_dir / "phase7_fold_leakage_differences.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    model_macro = model[model.metric == "Macro-F1"]
    for index, selector in enumerate(SELECTOR_ORDER):
        subset = model_macro[model_macro.selector == selector].sort_values(
            "global_fold"
        )
        axis.plot(
            subset.global_fold,
            subset.knn_minus_id3,
            marker="o",
            linewidth=1,
            label=selector,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("Safe-condition classifier differences (k-NN minus ID3)")
    axis.set_xlabel("Shared outer fold")
    axis.set_ylabel("Macro-F1 difference")
    axis.legend()
    path = output_dir / "phase7_safe_classifier_differences.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)
    return paths


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def run_analysis(
    *,
    id3_results: Path = DEFAULT_ID3_RESULTS,
    id3_summary: Path = DEFAULT_ID3_SUMMARY,
    knn_results: Path = DEFAULT_KNN_RESULTS,
    knn_summary: Path = DEFAULT_KNN_SUMMARY,
    output_dir: Path = TABLES_DIR,
    figure_dir: Path = FIGURES_DIR,
    manifest_file: Path = MANIFEST_FILE,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    overwrite: bool = False,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    """Run the maintained Phase 7 analysis and return generated paths."""

    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive.")
    for path, description in (
        (id3_results, "ID3 results"),
        (id3_summary, "ID3 summary"),
        (knn_results, "corrected Phase 6 k-NN results"),
        (knn_summary, "corrected Phase 6 k-NN summary"),
        (manifest_file, "outer-fold manifest"),
    ):
        require_file(path, description)
    if "phase6_corrected" not in knn_results.name:
        raise ValueError(
            "The k-NN input must be the corrected Phase 6 file; "
            f"received {knn_results.name}."
        )

    output_paths = [
        output_dir / name
        for name in (
            "phase7_paired_differences.csv",
            "phase7_paired_differences_summary.csv",
            "phase7_model_comparison.csv",
            "phase7_model_comparison_summary.csv",
            "phase7_selector_comparison.csv",
            "phase7_selector_comparison_summary.csv",
        )
    ]
    figure_paths = [
        figure_dir / name
        for name in (
            "phase7_safe_leaky_performance.png",
            "phase7_fold_leakage_differences.png",
            "phase7_safe_classifier_differences.png",
        )
    ]
    if not overwrite:
        existing = [path for path in output_paths + figure_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing Phase 7 artifacts. "
                "Back them up and rerun with --overwrite. Existing: "
                + ", ".join(map(str, existing))
            )

    id3_raw = pd.read_csv(id3_results)
    knn_raw = pd.read_csv(knn_results)
    id3 = _canonicalize_results(id3_raw, "ID3")
    knn = _canonicalize_results(knn_raw, "k-NN")
    validate_model_results(id3, "ID3")
    validate_model_results(knn, "corrected k-NN")
    validate_source_summary(
        id3,
        pd.read_csv(id3_summary),
        "ID3",
    )
    validate_source_summary(
        knn,
        pd.read_csv(knn_summary),
        "corrected k-NN",
    )
    id3_folds = set(
        map(
            tuple,
            id3[
                ["global_fold", "repeat", "fold", "n_train", "n_test"]
            ].drop_duplicates().to_numpy(),
        )
    )
    knn_folds = set(
        map(
            tuple,
            knn[
                ["global_fold", "repeat", "fold", "n_train", "n_test"]
            ].drop_duplicates().to_numpy(),
        )
    )
    if id3_folds != knn_folds:
        raise ValueError("ID3 and corrected k-NN fold metadata do not match.")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(bootstrap_seed)
    leakage_input = pd.concat([id3, knn], ignore_index=True)
    leakage, leakage_summary = _paired_rows(
        id3, knn, rng, bootstrap_resamples
    )
    model, model_summary = _model_rows(
        id3, knn, rng, bootstrap_resamples
    )
    selector, selector_summary = _selector_rows(
        leakage_input, rng, bootstrap_resamples
    )
    _write_frame(leakage, output_paths[0])
    _write_frame(leakage_summary, output_paths[1])
    _write_frame(model, output_paths[2])
    _write_frame(model_summary, output_paths[3])
    _write_frame(selector, output_paths[4])
    _write_frame(selector_summary, output_paths[5])
    generated_figures = _write_figures(
        leakage_input, leakage, model, figure_dir
    )

    return {
        "tables": output_paths,
        "figures": generated_figures,
        "inputs": [
            id3_results,
            id3_summary,
            knn_results,
            knn_summary,
            manifest_file,
        ],
    }


def _package_versions() -> dict[str, str]:
    names = ["numpy", "pandas", "matplotlib", "scikit-learn", "scipy"]
    return {
        name: importlib.metadata.version(name)
        for name in names
        if importlib.metadata.version(name)
    }


def write_manifest(
    path: Path,
    command: list[str],
    inputs: Iterable[Path],
    outputs: Iterable[Path],
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "command": command,
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_resamples": bootstrap_resamples,
        "interval_note": INTERVAL_NOTE,
        "inputs": [
            {"path": str(item), "sha256": sha256_file(item)}
            for item in inputs
        ],
        "outputs": [
            {"path": str(item), "sha256": sha256_file(item)}
            for item in outputs
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate descriptive Phase 7 analysis artifacts."
    )
    parser.add_argument("--id3-results", type=Path, default=DEFAULT_ID3_RESULTS)
    parser.add_argument("--id3-summary", type=Path, default=DEFAULT_ID3_SUMMARY)
    parser.add_argument("--knn-results", type=Path, default=DEFAULT_KNN_RESULTS)
    parser.add_argument("--knn-summary", type=Path, default=DEFAULT_KNN_SUMMARY)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_FILE)
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "phase9b_reproducibility_manifest.json",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing Phase 7 tables and figures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generated = run_analysis(
        id3_results=args.id3_results,
        id3_summary=args.id3_summary,
        knn_results=args.knn_results,
        knn_summary=args.knn_summary,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        manifest_file=args.manifest,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        overwrite=args.overwrite,
    )
    manifest_outputs = generated["tables"] + generated["figures"]
    write_manifest(
        args.manifest_output,
        [sys.executable, *sys.argv],
        generated["inputs"],
        manifest_outputs,
        args.bootstrap_seed,
        args.bootstrap_resamples,
    )
    print("Generated Phase 7 tables:")
    for path in generated["tables"]:
        print(f"  {path}")
    print("Generated Phase 7 figures:")
    for path in generated["figures"]:
        print(f"  {path}")
    print(f"Reproducibility manifest: {args.manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
