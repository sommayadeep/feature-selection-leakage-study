"""Create reproducible figures from verified research result tables."""

from __future__ import annotations

import argparse
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

PHASE7_PAIRED = TABLES_DIR / "phase7_paired_differences.csv"
PHASE7_MODEL = TABLES_DIR / "phase7_model_comparison.csv"
FEATURE_STABILITY = TABLES_DIR / "feature_stability.csv"

EXPECTED_CLASSIFIERS = ["ID3", "k-NN"]
EXPECTED_SELECTORS = ["ANOVA", "MI"]
EXPECTED_METRICS = ["Macro-F1", "Balanced accuracy"]


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    description: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(
            f"{description} is missing required columns: {', '.join(missing)}"
        )


def _require_no_missing(
    frame: pd.DataFrame,
    columns: Iterable[str],
    description: str,
) -> None:
    missing = [
        column
        for column in columns
        if frame[column].isna().any()
    ]
    if missing:
        raise ValueError(
            f"{description} contains missing values in: {', '.join(missing)}"
        )


def _validate_numeric(
    frame: pd.DataFrame,
    columns: Iterable[str],
    description: str,
) -> None:
    for column in columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"{description} column '{column}' must be numeric.")


def _validate_fold_keys(
    frame: pd.DataFrame,
    keys: list[str],
    expected_keys: set[tuple[object, ...]],
    description: str,
) -> None:
    if frame[keys].duplicated().any():
        raise ValueError(f"{description} contains duplicate fold rows.")
    actual_keys = set(frame[keys].itertuples(index=False, name=None))
    if actual_keys != expected_keys:
        raise ValueError(
            f"{description} must contain every expected configuration/fold "
            "combination exactly once."
        )


def load_phase7_paired_results(path: Path = PHASE7_PAIRED) -> pd.DataFrame:
    """Load and validate the shared 3x5 paired leakage results."""

    if not path.is_file():
        raise FileNotFoundError(f"Phase 7 paired results not found: {path}")
    frame = pd.read_csv(path)
    required = {
        "classifier",
        "selector",
        "repeat",
        "fold",
        "global_fold",
        "metric",
        "safe_value",
        "leaky_value",
        "leaky_minus_safe",
        "bootstrap_95_percentile_low",
        "bootstrap_95_percentile_high",
    }
    _require_columns(frame, required, "Phase 7 paired results")
    _require_no_missing(frame, required, "Phase 7 paired results")
    _validate_numeric(
        frame,
        required - {"classifier", "selector", "metric"},
        "Phase 7 paired results",
    )
    numeric_columns = required - {"classifier", "selector", "metric"}
    if not np.isfinite(frame[list(numeric_columns)].to_numpy(dtype=float)).all():
        raise ValueError("Phase 7 paired results must contain finite numeric values.")

    if set(frame["classifier"]) != set(EXPECTED_CLASSIFIERS):
        raise ValueError("Phase 7 paired results must contain ID3 and k-NN.")
    if set(frame["selector"]) != set(EXPECTED_SELECTORS):
        raise ValueError("Phase 7 paired results must contain ANOVA and MI.")
    if set(frame["metric"]) != set(EXPECTED_METRICS):
        raise ValueError(
            "Phase 7 paired results must contain Macro-F1 and balanced accuracy."
        )
    expected_paired_keys = {
        (classifier, selector, repeat, fold, (repeat - 1) * 3 + fold, metric)
        for classifier in EXPECTED_CLASSIFIERS
        for selector in EXPECTED_SELECTORS
        for repeat in range(1, 6)
        for fold in range(1, 4)
        for metric in EXPECTED_METRICS
    }
    _validate_fold_keys(
        frame,
        ["classifier", "selector", "repeat", "fold", "global_fold", "metric"],
        expected_paired_keys,
        "Phase 7 paired results",
    )
    if not frame["safe_value"].between(0, 1).all():
        raise ValueError("Safe metric values must be between 0 and 1.")
    if not frame["leaky_value"].between(0, 1).all():
        raise ValueError("Leaky metric values must be between 0 and 1.")
    if not np.allclose(
        frame["leaky_value"] - frame["safe_value"],
        frame["leaky_minus_safe"],
    ):
        raise ValueError("Leaky-minus-safe values do not match source metrics.")
    return frame


def load_phase7_model_results(path: Path = PHASE7_MODEL) -> pd.DataFrame:
    """Load and validate the shared 3x5 safe model-comparison results."""

    if not path.is_file():
        raise FileNotFoundError(f"Phase 7 model results not found: {path}")
    frame = pd.read_csv(path)
    required = {
        "selector",
        "repeat",
        "fold",
        "global_fold",
        "metric",
        "id3_safe_value",
        "knn_safe_value",
        "knn_minus_id3",
        "bootstrap_95_percentile_low",
        "bootstrap_95_percentile_high",
    }
    _require_columns(frame, required, "Phase 7 model comparison")
    _require_no_missing(frame, required, "Phase 7 model comparison")
    _validate_numeric(
        frame,
        required - {"selector", "metric"},
        "Phase 7 model comparison",
    )
    numeric_columns = required - {"selector", "metric"}
    if not np.isfinite(frame[list(numeric_columns)].to_numpy(dtype=float)).all():
        raise ValueError("Phase 7 model comparison must contain finite numeric values.")
    if set(frame["selector"]) != set(EXPECTED_SELECTORS):
        raise ValueError("Phase 7 model comparison must contain ANOVA and MI.")
    if set(frame["metric"]) != set(EXPECTED_METRICS):
        raise ValueError(
            "Phase 7 model comparison must contain Macro-F1 and balanced accuracy."
        )
    expected_model_keys = {
        (selector, repeat, fold, (repeat - 1) * 3 + fold, metric)
        for selector in EXPECTED_SELECTORS
        for repeat in range(1, 6)
        for fold in range(1, 4)
        for metric in EXPECTED_METRICS
    }
    _validate_fold_keys(
        frame,
        ["selector", "repeat", "fold", "global_fold", "metric"],
        expected_model_keys,
        "Phase 7 model comparison",
    )
    if not frame["id3_safe_value"].between(0, 1).all():
        raise ValueError("ID3 metric values must be between 0 and 1.")
    if not frame["knn_safe_value"].between(0, 1).all():
        raise ValueError("k-NN metric values must be between 0 and 1.")
    if not np.allclose(
        frame["knn_safe_value"] - frame["id3_safe_value"],
        frame["knn_minus_id3"],
    ):
        raise ValueError("k-NN-minus-ID3 values do not match source metrics.")
    return frame


def load_feature_stability(path: Path = FEATURE_STABILITY) -> pd.DataFrame:
    """Load and validate the standalone feature-stability measurements."""

    if not path.is_file():
        raise FileNotFoundError(f"Feature stability results not found: {path}")
    frame = pd.read_csv(path)
    required = {
        "selector",
        "repeats",
        "features_per_repeat",
        "number_of_pairs",
        "mean_jaccard",
        "std_jaccard",
    }
    _require_columns(frame, required, "Feature stability results")
    _require_no_missing(frame, required, "Feature stability results")
    _validate_numeric(
        frame,
        required - {"selector"},
        "Feature stability results",
    )
    numeric_columns = required - {"selector"}
    if not np.isfinite(frame[list(numeric_columns)].to_numpy(dtype=float)).all():
        raise ValueError("Feature stability results must contain finite numeric values.")
    if frame["selector"].duplicated().any():
        raise ValueError("Feature stability results contain duplicate selectors.")
    if set(frame["selector"]) != {"anova", "mutual_information"}:
        raise ValueError(
            "Feature stability results must contain ANOVA and mutual information."
        )
    if not frame["mean_jaccard"].between(0, 1).all():
        raise ValueError("Mean Jaccard values must be between 0 and 1.")
    if (frame["std_jaccard"] < 0).any():
        raise ValueError("Jaccard standard deviations cannot be negative.")
    return frame


def summarize_leakage(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate paired leakage differences while preserving source intervals."""

    grouped = (
        frame.groupby(["classifier", "selector", "metric"], sort=False)
        .agg(
            mean_difference=("leaky_minus_safe", "mean"),
            interval_low=("bootstrap_95_percentile_low", "first"),
            interval_high=("bootstrap_95_percentile_high", "first"),
            n_folds=("global_fold", "nunique"),
        )
        .reset_index()
    )
    if (grouped["n_folds"] != 15).any():
        raise ValueError("Each leakage comparison must have 15 paired folds.")
    if (grouped["interval_high"] < grouped["interval_low"]).any():
        raise ValueError("Leakage interval bounds are not ordered.")
    return grouped


def summarize_model_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate safe k-NN-minus-ID3 differences with source intervals."""

    grouped = (
        frame.groupby(["selector", "metric"], sort=False)
        .agg(
            mean_difference=("knn_minus_id3", "mean"),
            interval_low=("bootstrap_95_percentile_low", "first"),
            interval_high=("bootstrap_95_percentile_high", "first"),
            n_folds=("global_fold", "nunique"),
        )
        .reset_index()
    )
    if (grouped["n_folds"] != 15).any():
        raise ValueError("Each model comparison must have 15 paired folds.")
    if (grouped["interval_high"] < grouped["interval_low"]).any():
        raise ValueError("Model-comparison interval bounds are not ordered.")
    return grouped


def _save_figure(figure: plt.Figure, path: Path, overwrite: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Figure already exists: {path}. Use --overwrite to replace it."
        )
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_interval_bars(
    axis: plt.Axes,
    labels: list[str],
    values: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    color: str,
) -> None:
    errors = np.vstack([values - lows, highs - values])
    if np.any(errors < -1e-12):
        raise ValueError("An interval does not contain its plotted mean.")
    axis.bar(
        np.arange(len(labels)),
        values,
        yerr=np.maximum(errors, 0),
        capsize=4,
        color=color,
        edgecolor="black",
        linewidth=0.6,
    )
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.grid(axis="y", alpha=0.25)


def plot_leakage_comparison(
    frame: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
    overwrite: bool = False,
) -> Path:
    """Plot descriptive leaky-minus-safe effects for the shared protocol."""

    summary = summarize_leakage(frame)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    labels = [
        f"{row.classifier}\n{row.selector}"
        for row in summary.itertuples()
    ]
    for axis, metric in zip(axes, EXPECTED_METRICS):
        selected = summary[summary["metric"] == metric]
        selected = selected.set_index(["classifier", "selector"]).loc[
            [(classifier, selector)
             for classifier in EXPECTED_CLASSIFIERS
             for selector in EXPECTED_SELECTORS]
        ].reset_index()
        metric_labels = [
            f"{row.classifier}\n{row.selector}"
            for row in selected.itertuples()
        ]
        values = selected["mean_difference"].to_numpy()
        _plot_interval_bars(
            axis,
            metric_labels,
            values,
            selected["interval_low"].to_numpy(),
            selected["interval_high"].to_numpy(),
            "#c44e52",
        )
        axis.set_title(metric)
        axis.set_xlabel("Classifier and selector")
        axis.set_ylabel("Leaky minus safe score")
    figure.suptitle(
        "Feature-selection leakage effect under the shared 3×5 protocol",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.01,
        "Bars are mean paired differences across 15 folds; error bars are "
        "descriptive percentile bootstrap intervals, not inferential CIs.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    return _save_figure(
        figure,
        output_dir / "leakage_comparison.png",
        overwrite,
    )


def plot_model_comparison(
    frame: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
    overwrite: bool = False,
) -> Path:
    """Plot safe k-NN-minus-ID3 differences under the shared protocol."""

    summary = summarize_model_comparison(frame)
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 4.8), sharey=True)
    for axis, metric in zip(axes, EXPECTED_METRICS):
        selected = summary[summary["metric"] == metric].set_index(
            "selector"
        ).loc[EXPECTED_SELECTORS].reset_index()
        values = selected["mean_difference"].to_numpy()
        _plot_interval_bars(
            axis,
            EXPECTED_SELECTORS,
            values,
            selected["interval_low"].to_numpy(),
            selected["interval_high"].to_numpy(),
            "#4c72b0",
        )
        axis.set_title(metric)
        axis.set_xlabel("Feature selector")
        axis.set_ylabel("Safe k-NN minus safe ID3")
    figure.suptitle(
        "Safe-condition classifier differences under the shared 3×5 protocol",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.01,
        "Positive values favor k-NN; intervals are descriptive percentile "
        "bootstrap summaries over paired folds.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    return _save_figure(
        figure,
        output_dir / "model_comparison.png",
        overwrite,
    )


def plot_feature_stability(
    frame: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
    overwrite: bool = False,
) -> Path:
    """Plot mean pairwise Jaccard stability for each selector."""

    order = ["anova", "mutual_information"]
    selected = frame.set_index("selector").loc[order].reset_index()
    labels = ["ANOVA", "Mutual information"]
    figure, axis = plt.subplots(figsize=(6.5, 4.8))
    axis.bar(
        labels,
        selected["mean_jaccard"],
        yerr=selected["std_jaccard"],
        capsize=4,
        color="#55a868",
        edgecolor="black",
        linewidth=0.6,
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean pairwise Jaccard similarity")
    axis.set_xlabel("Feature selector")
    axis.set_title("Feature-selection stability")
    axis.grid(axis="y", alpha=0.25)
    figure.text(
        0.5,
        0.01,
        "Bars show mean Jaccard similarity; error bars show the reported "
        "standard deviation across feature-set pairs.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.96))
    return _save_figure(
        figure,
        output_dir / "feature_stability.png",
        overwrite,
    )


def plot_all(
    output_dir: Path = FIGURES_DIR,
    overwrite: bool = False,
) -> list[Path]:
    """Validate source tables and create all supported figures."""

    paired = load_phase7_paired_results()
    model = load_phase7_model_results()
    stability = load_feature_stability()
    return [
        plot_leakage_comparison(paired, output_dir, overwrite),
        plot_model_comparison(model, output_dir, overwrite),
        plot_feature_stability(stability, output_dir, overwrite),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot supported research results without rerunning experiments."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing figures with the same names.",
    )
    args = parser.parse_args()

    for path in plot_all(args.output_dir, args.overwrite):
        print(f"Generated {path}")


if __name__ == "__main__":
    main()
