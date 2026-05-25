import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "llm_runs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "plots"

def infer_model_name(path: Path) -> str:
    name = path.stem

    match = re.match(r"results_(.*?)_", name)
    if match:
        return match.group(1)

    return path.stem


def load_results(input_dir: Path) -> pd.DataFrame:
    csv_paths = sorted(input_dir.rglob("results_*_all.csv"))

    if not csv_paths:
        raise FileNotFoundError(
            f"No '*_all.csv' files found recursively in {input_dir}. "
            "Run model evaluation first or check the input directory."
        )

    frames = []

    for path in csv_paths:
        frame = pd.read_csv(path)

        if "model_key" not in frame.columns:
            frame["model_key"] = infer_model_name(path)

        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def compute_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for model, model_df in df.groupby("model_key"):
        label_mask = model_df["task"].eq("classification")
        recommendation_mask = model_df["task"].eq("recommendation")
        decision_mask = model_df["task"].eq("decision_answering")

        label_flips = as_bool(model_df.get("label_flip", pd.Series(False, index=model_df.index)))
        recommendation_flips = as_bool(
            model_df.get("recommendation_flip", pd.Series(False, index=model_df.index))
        )
        choice_flips = as_bool(model_df.get("choice_flip", pd.Series(False, index=model_df.index)))

        any_decision_flip = label_flips | recommendation_flips | choice_flips

        confidence_shift = safe_numeric(model_df["confidence_shift"])
        reasoning_similarity = safe_numeric(model_df["reasoning_similarity"])
        reasoning_length_shift = safe_numeric(model_df["reasoning_length_shift"])

        rows.append(
            {
                "model": model,
                "total_pairs": len(model_df),
                "classification_pairs": int(label_mask.sum()),
                "recommendation_pairs": int(recommendation_mask.sum()),
                "decision_answering_pairs": int(decision_mask.sum()),
                "label_flips": int(label_flips.sum()),
                "label_flip_rate": label_flips[label_mask].mean() if label_mask.sum() else 0,
                "recommendation_flips": int(recommendation_flips.sum()),
                "recommendation_flip_rate": (
                    recommendation_flips[recommendation_mask].mean()
                    if recommendation_mask.sum()
                    else 0
                ),
                "choice_flips": int(choice_flips.sum()),
                "choice_flip_rate": choice_flips[decision_mask].mean()
                if decision_mask.sum()
                else 0,
                "any_decision_flips": int(any_decision_flip.sum()),
                "any_decision_flip_rate": any_decision_flip.mean(),
                "confidence_shift_mean": confidence_shift.mean(),
                "confidence_shift_max": confidence_shift.max(),
                "reasoning_similarity_mean": reasoning_similarity.mean(),
                "reasoning_similarity_median": reasoning_similarity.median(),
                "reasoning_similarity_min": reasoning_similarity.min(),
                "reasoning_similarity_lt_080": int((reasoning_similarity < 0.80).sum()),
                "reasoning_similarity_lt_065": int((reasoning_similarity < 0.65).sum()),
                "reasoning_length_shift_mean": reasoning_length_shift.mean(),
                "reasoning_length_shift_max": reasoning_length_shift.max(),
            }
        )

    return pd.DataFrame(rows)


def compute_task_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (model, task), group in df.groupby(["model_key", "task"]):
        label_flips = as_bool(group.get("label_flip", pd.Series(False, index=group.index)))
        recommendation_flips = as_bool(
            group.get("recommendation_flip", pd.Series(False, index=group.index))
        )
        choice_flips = as_bool(group.get("choice_flip", pd.Series(False, index=group.index)))
        ranking_changed = as_bool(group.get("ranking_changed", pd.Series(False, index=group.index)))

        rows.append(
            {
                "model": model,
                "task": task,
                "pairs": len(group),
                "label_flips": int(label_flips.sum()),
                "recommendation_flips": int(recommendation_flips.sum()),
                "choice_flips": int(choice_flips.sum()),
                "ranking_changed": int(ranking_changed.sum()),
                "confidence_shift_mean": safe_numeric(group["confidence_shift"]).mean(),
                "reasoning_similarity_mean": safe_numeric(group["reasoning_similarity"]).mean(),
                "reasoning_similarity_median": safe_numeric(group["reasoning_similarity"]).median(),
            }
        )

    return pd.DataFrame(rows)


def compute_bias_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (model, bias_axis), group in df.groupby(["model_key", "bias_axis"]):
        label_flips = as_bool(group.get("label_flip", pd.Series(False, index=group.index)))
        recommendation_flips = as_bool(
            group.get("recommendation_flip", pd.Series(False, index=group.index))
        )
        choice_flips = as_bool(group.get("choice_flip", pd.Series(False, index=group.index)))

        any_decision_flip = label_flips | recommendation_flips | choice_flips

        rows.append(
            {
                "model": model,
                "bias_axis": bias_axis,
                "pairs": len(group),
                "any_decision_flips": int(any_decision_flip.sum()),
                "any_decision_flip_rate": any_decision_flip.mean(),
                "reasoning_similarity_mean": safe_numeric(group["reasoning_similarity"]).mean(),
                "reasoning_similarity_lt_080": int(
                    (safe_numeric(group["reasoning_similarity"]) < 0.80).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def compute_subtask_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (model, subtask), group in df.groupby(["model_key", "subtask"]):
        label_flips = as_bool(group.get("label_flip", pd.Series(False, index=group.index)))
        recommendation_flips = as_bool(
            group.get("recommendation_flip", pd.Series(False, index=group.index))
        )
        choice_flips = as_bool(group.get("choice_flip", pd.Series(False, index=group.index)))

        any_decision_flip = label_flips | recommendation_flips | choice_flips

        rows.append(
            {
                "model": model,
                "subtask": subtask,
                "pairs": len(group),
                "any_decision_flips": int(any_decision_flip.sum()),
                "any_decision_flip_rate": any_decision_flip.mean(),
                "reasoning_similarity_mean": safe_numeric(group["reasoning_similarity"]).mean(),
                "reasoning_similarity_lt_080": int(
                    (safe_numeric(group["reasoning_similarity"]) < 0.80).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def save_bar_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    output_path: Path,
    rotation: int = 0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_data = data.copy()
    plot_data[x] = plot_data[x].astype(str)

    plt.figure(figsize=(10, 6))
    plt.bar(plot_data[x], plot_data[y])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right" if rotation else "center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_grouped_bar_plot(
    data: pd.DataFrame,
    index_col: str,
    value_cols: list[str],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_data = data.set_index(index_col)[value_cols]

    plt.figure(figsize=(10, 6))
    plot_data.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_reasoning_histograms(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for model, group in df.groupby("model_key"):
        values = safe_numeric(group["reasoning_similarity"]).dropna()

        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=20)
        plt.title(f"Reasoning similarity distribution - {model}")
        plt.xlabel("Reasoning similarity")
        plt.ylabel("Number of pairs")
        plt.tight_layout()
        plt.savefig(output_dir / f"reasoning_similarity_hist_{model}.png", dpi=200)
        plt.close()


def save_confidence_shift_histograms(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for model, group in df.groupby("model_key"):
        values = safe_numeric(group["confidence_shift"]).dropna()

        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=range(0, int(values.max()) + 2))
        plt.title(f"Confidence shift distribution - {model}")
        plt.xlabel("Confidence shift")
        plt.ylabel("Number of pairs")
        plt.tight_layout()
        plt.savefig(output_dir / f"confidence_shift_hist_{model}.png", dpi=200)
        plt.close()


def save_heatmap(
    pivot: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 7))
    plt.imshow(pivot.values, aspect="auto")
    plt.title(title)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label="Value")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_decision_flip_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    label_flips = as_bool(df.get("label_flip", pd.Series(False, index=df.index)))
    recommendation_flips = as_bool(
        df.get("recommendation_flip", pd.Series(False, index=df.index))
    )
    choice_flips = as_bool(df.get("choice_flip", pd.Series(False, index=df.index)))

    df = df.copy()
    df["any_decision_flip"] = label_flips | recommendation_flips | choice_flips

    grouped = (
        df.groupby(["bias_axis", "model_key"])["any_decision_flip"]
        .mean()
        .reset_index()
    )

    pivot = grouped.pivot(
        index="bias_axis",
        columns="model_key",
        values="any_decision_flip",
    ).fillna(0)

    save_heatmap(
        pivot=pivot,
        title="Decision flip rate by bias axis and model",
        output_path=output_dir / "heatmap_decision_flip_rate_by_bias_axis.png",
    )


def save_reasoning_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        df.groupby(["bias_axis", "model_key"])["reasoning_similarity"]
        .apply(lambda s: safe_numeric(s).mean())
        .reset_index()
    )

    pivot = grouped.pivot(
        index="bias_axis",
        columns="model_key",
        values="reasoning_similarity",
    ).fillna(0)

    save_heatmap(
        pivot=pivot,
        title="Mean reasoning similarity by bias axis and model",
        output_path=output_dir / "heatmap_reasoning_similarity_by_bias_axis.png",
    )


def save_top_cases(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    label_flips = as_bool(df.get("label_flip", pd.Series(False, index=df.index)))
    recommendation_flips = as_bool(
        df.get("recommendation_flip", pd.Series(False, index=df.index))
    )
    choice_flips = as_bool(df.get("choice_flip", pd.Series(False, index=df.index)))

    df = df.copy()
    df["any_decision_flip"] = label_flips | recommendation_flips | choice_flips
    df["reasoning_similarity"] = safe_numeric(df["reasoning_similarity"])
    df["confidence_shift"] = safe_numeric(df["confidence_shift"])

    selected_columns = [
        "model_key",
        "pair_id",
        "task",
        "subtask",
        "bias_axis",
        "replacement_group",
        "original_term",
        "counterfactual_term",
        "label_flip",
        "recommendation_flip",
        "choice_flip",
        "confidence_shift",
        "reasoning_similarity",
        "reasoning_length_shift",
        "original_reason",
        "counterfactual_reason",
    ]

    selected_columns = [column for column in selected_columns if column in df.columns]

    decision_flip_cases = df[df["any_decision_flip"]].copy()
    decision_flip_cases[selected_columns].to_csv(
        output_dir / "cases_decision_flips.csv",
        index=False,
    )

    low_reasoning_cases = (
        df.sort_values("reasoning_similarity", ascending=True)
        .head(100)
        .copy()
    )
    low_reasoning_cases[selected_columns].to_csv(
        output_dir / "cases_lowest_reasoning_similarity.csv",
        index=False,
    )

    stable_output_low_reasoning = df[
        (~df["any_decision_flip"])
        & (df["reasoning_similarity"] < 0.80)
    ].copy()

    stable_output_low_reasoning[selected_columns].to_csv(
        output_dir / "cases_stable_output_low_reasoning_similarity.csv",
        index=False,
    )


def generate_plots(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_summary = compute_model_summary(df)
    task_summary = compute_task_summary(df)
    bias_summary = compute_bias_summary(df)
    subtask_summary = compute_subtask_summary(df)

    save_table(model_summary, output_dir / "summary_by_model.csv")
    save_table(task_summary, output_dir / "summary_by_task.csv")
    save_table(bias_summary, output_dir / "summary_by_bias_axis.csv")
    save_table(subtask_summary, output_dir / "summary_by_subtask.csv")

    save_bar_plot(
        data=model_summary,
        x="model",
        y="any_decision_flip_rate",
        title="Decision flip rate by model",
        ylabel="Decision flip rate",
        output_path=output_dir / "decision_flip_rate_by_model.png",
    )

    save_grouped_bar_plot(
        data=model_summary,
        index_col="model",
        value_cols=[
            "label_flip_rate",
            "recommendation_flip_rate",
            "choice_flip_rate",
        ],
        title="Task-specific flip rates by model",
        ylabel="Flip rate",
        output_path=output_dir / "task_flip_rates_by_model.png",
    )

    save_bar_plot(
        data=model_summary,
        x="model",
        y="reasoning_similarity_mean",
        title="Mean reasoning similarity by model",
        ylabel="Mean reasoning similarity",
        output_path=output_dir / "mean_reasoning_similarity_by_model.png",
    )

    save_bar_plot(
        data=model_summary,
        x="model",
        y="reasoning_similarity_lt_080",
        title="Pairs with reasoning similarity below 0.80 by model",
        ylabel="Number of pairs",
        output_path=output_dir / "reasoning_similarity_lt_080_by_model.png",
    )

    save_bar_plot(
        data=model_summary,
        x="model",
        y="confidence_shift_mean",
        title="Mean confidence shift by model",
        ylabel="Mean confidence shift",
        output_path=output_dir / "mean_confidence_shift_by_model.png",
    )

    save_reasoning_histograms(
        df=df,
        output_dir=output_dir,
    )

    save_confidence_shift_histograms(
        df=df,
        output_dir=output_dir,
    )

    save_decision_flip_heatmap(
        df=df,
        output_dir=output_dir,
    )

    save_reasoning_heatmap(
        df=df,
        output_dir=output_dir,
    )

    save_top_cases(
        df=df,
        output_dir=output_dir,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate plots and summary tables from SE4AI evaluation CSV files."
    )

    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing results_*_all.csv files.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where plots and summary CSV files are saved.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    df = load_results(input_dir)

    generate_plots(
        df=df,
        output_dir=output_dir,
    )

    print(f"Loaded {len(df)} rows from {input_dir}")
    print(f"Saved plots and tables to {output_dir}")


if __name__ == "__main__":
    main()