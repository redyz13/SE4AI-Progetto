import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RUN_A_DIR = Path("outputs/llm_runs")
DEFAULT_RUN_B_DIR = Path("outputs/llm_run2")
DEFAULT_OUTPUT_DIR = Path("outputs/run_comparison")

PLOT_FORMAT = "pdf"

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def plot_path(output_dir: Path, name: str) -> Path:
    return output_dir / f"{name}.{PLOT_FORMAT}"


def infer_model_name(path: Path) -> str:
    name = path.stem
    match = re.match(r"results_(.*?)_", name)
    if match:
        return match.group(1)
    return path.parent.name.lower()


def as_bool(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    return series.astype(str).str.lower().eq("true")


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_run_results(run_dir: Path, run_name: str) -> pd.DataFrame:
    csv_paths = sorted(run_dir.rglob("results_*_all.csv"))

    if not csv_paths:
        raise FileNotFoundError(
            f"No results_*_all.csv files found recursively in {run_dir}"
        )

    frames = []

    for path in csv_paths:
        frame = pd.read_csv(path)

        if "model_key" not in frame.columns:
            frame["model_key"] = infer_model_name(path)

        frame["run"] = run_name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    label_flip = as_bool(df.get("label_flip", pd.Series(False, index=df.index)))
    recommendation_flip = as_bool(
        df.get("recommendation_flip", pd.Series(False, index=df.index))
    )
    choice_flip = as_bool(df.get("choice_flip", pd.Series(False, index=df.index)))
    ranking_changed = as_bool(
        df.get("ranking_changed", pd.Series(False, index=df.index))
    )

    df["label_flip_bool"] = label_flip
    df["recommendation_flip_bool"] = recommendation_flip
    df["choice_flip_bool"] = choice_flip
    df["ranking_changed_bool"] = ranking_changed
    df["any_decision_flip"] = label_flip | recommendation_flip | choice_flip

    if "confidence_shift" in df.columns:
        df["confidence_shift_num"] = safe_numeric(df["confidence_shift"])

    if "reasoning_similarity" in df.columns:
        df["reasoning_similarity_num"] = safe_numeric(df["reasoning_similarity"])

    if "reasoning_length_shift" in df.columns:
        df["reasoning_length_shift_num"] = safe_numeric(df["reasoning_length_shift"])

    return df


def compute_summary_by_model(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (run, model), group in df.groupby(["run", "model_key"]):
        classification = group["task"].eq("classification")
        recommendation = group["task"].eq("recommendation")
        decision_answering = group["task"].eq("decision_answering")

        rows.append(
            {
                "run": run,
                "model": model,
                "total_pairs": len(group),
                "classification_pairs": int(classification.sum()),
                "recommendation_pairs": int(recommendation.sum()),
                "decision_answering_pairs": int(decision_answering.sum()),
                "label_flips": int(group["label_flip_bool"].sum()),
                "label_flip_rate": (
                    group.loc[classification, "label_flip_bool"].mean()
                    if classification.sum()
                    else 0
                ),
                "recommendation_flips": int(group["recommendation_flip_bool"].sum()),
                "recommendation_flip_rate": (
                    group.loc[recommendation, "recommendation_flip_bool"].mean()
                    if recommendation.sum()
                    else 0
                ),
                "choice_flips": int(group["choice_flip_bool"].sum()),
                "choice_flip_rate": (
                    group.loc[decision_answering, "choice_flip_bool"].mean()
                    if decision_answering.sum()
                    else 0
                ),
                "ranking_changed": int(group["ranking_changed_bool"].sum()),
                "ranking_changed_rate": (
                    group.loc[recommendation, "ranking_changed_bool"].mean()
                    if recommendation.sum()
                    else 0
                ),
                "any_decision_flips": int(group["any_decision_flip"].sum()),
                "any_decision_flip_rate": group["any_decision_flip"].mean(),
                "confidence_shift_mean": group["confidence_shift_num"].mean(),
                "confidence_shift_max": group["confidence_shift_num"].max(),
                "reasoning_similarity_mean": group["reasoning_similarity_num"].mean(),
                "reasoning_similarity_median": group[
                    "reasoning_similarity_num"
                ].median(),
                "reasoning_similarity_min": group["reasoning_similarity_num"].min(),
                "reasoning_similarity_lt_080": int(
                    (group["reasoning_similarity_num"] < 0.80).sum()
                ),
                "reasoning_similarity_lt_065": int(
                    (group["reasoning_similarity_num"] < 0.65).sum()
                ),
                "reasoning_length_shift_mean": group[
                    "reasoning_length_shift_num"
                ].mean(),
                "reasoning_length_shift_max": group[
                    "reasoning_length_shift_num"
                ].max(),
            }
        )

    return pd.DataFrame(rows)


def compute_group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []

    for keys, group in df.groupby(["run", "model_key", group_col]):
        run, model, group_value = keys

        rows.append(
            {
                "run": run,
                "model": model,
                group_col: group_value,
                "pairs": len(group),
                "label_flips": int(group["label_flip_bool"].sum()),
                "recommendation_flips": int(group["recommendation_flip_bool"].sum()),
                "choice_flips": int(group["choice_flip_bool"].sum()),
                "ranking_changed": int(group["ranking_changed_bool"].sum()),
                "any_decision_flips": int(group["any_decision_flip"].sum()),
                "any_decision_flip_rate": group["any_decision_flip"].mean(),
                "confidence_shift_mean": group["confidence_shift_num"].mean(),
                "reasoning_similarity_mean": group["reasoning_similarity_num"].mean(),
                "reasoning_similarity_median": group[
                    "reasoning_similarity_num"
                ].median(),
                "reasoning_similarity_lt_080": int(
                    (group["reasoning_similarity_num"] < 0.80).sum()
                ),
                "reasoning_similarity_lt_065": int(
                    (group["reasoning_similarity_num"] < 0.65).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def compute_delta(
    summary: pd.DataFrame,
    id_cols: list[str],
    run_a: str,
    run_b: str,
) -> pd.DataFrame:
    a = summary[summary["run"] == run_a].copy()
    b = summary[summary["run"] == run_b].copy()

    a = a.drop(columns=["run"])
    b = b.drop(columns=["run"])

    merged = a.merge(
        b,
        on=id_cols,
        how="outer",
        suffixes=(f"_{run_a}", f"_{run_b}"),
    )

    metric_cols = [
        col
        for col in a.columns
        if col not in id_cols and pd.api.types.is_numeric_dtype(a[col])
    ]

    for col in metric_cols:
        col_a = f"{col}_{run_a}"
        col_b = f"{col}_{run_b}"

        if col_a in merged.columns and col_b in merged.columns:
            merged[f"{col}_delta"] = merged[col_b] - merged[col_a]

    return merged


def compute_pair_level_comparison(
    run_a_df: pd.DataFrame,
    run_b_df: pd.DataFrame,
    run_a: str,
    run_b: str,
) -> pd.DataFrame:
    key_cols = ["model_key", "pair_id"]

    selected_cols = [
        "model_key",
        "pair_id",
        "scenario_id",
        "task",
        "subtask",
        "bias_axis",
        "replacement_group",
        "original_term",
        "counterfactual_term",
        "original_label",
        "counterfactual_label",
        "original_recommended_option",
        "counterfactual_recommended_option",
        "original_ranking",
        "counterfactual_ranking",
        "original_selected_subject",
        "counterfactual_selected_subject",
        "label_flip_bool",
        "recommendation_flip_bool",
        "choice_flip_bool",
        "ranking_changed_bool",
        "any_decision_flip",
        "confidence_shift_num",
        "reasoning_similarity_num",
        "reasoning_length_shift_num",
        "original_reason",
        "counterfactual_reason",
    ]

    selected_cols = [col for col in selected_cols if col in run_a_df.columns]

    a = run_a_df[selected_cols].copy()
    b = run_b_df[selected_cols].copy()

    merged = a.merge(
        b,
        on=key_cols,
        how="outer",
        suffixes=(f"_{run_a}", f"_{run_b}"),
        indicator=True,
    )

    for col in [
        "label_flip_bool",
        "recommendation_flip_bool",
        "choice_flip_bool",
        "ranking_changed_bool",
        "any_decision_flip",
    ]:
        col_a = f"{col}_{run_a}"
        col_b = f"{col}_{run_b}"

        if col_a in merged.columns and col_b in merged.columns:
            merged[f"{col}_changed"] = merged[col_a] != merged[col_b]

    if (
        f"reasoning_similarity_num_{run_a}" in merged.columns
        and f"reasoning_similarity_num_{run_b}" in merged.columns
    ):
        merged["reasoning_similarity_delta"] = (
            merged[f"reasoning_similarity_num_{run_b}"]
            - merged[f"reasoning_similarity_num_{run_a}"]
        )

    if (
        f"confidence_shift_num_{run_a}" in merged.columns
        and f"confidence_shift_num_{run_b}" in merged.columns
    ):
        merged["confidence_shift_delta"] = (
            merged[f"confidence_shift_num_{run_b}"]
            - merged[f"confidence_shift_num_{run_a}"]
        )

    return merged


def save_bar_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    path: Path,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(data[x].astype(str), data[y])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_grouped_run_plot(
    summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    path: Path,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    pivot = summary.pivot(index="model", columns="run", values=metric).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_delta_plot(
    delta: pd.DataFrame,
    id_col: str,
    metric_delta: str,
    title: str,
    ylabel: str,
    path: Path,
    rotation: int = 0,
):
    if metric_delta not in delta.columns:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    plot_data = delta.copy()
    plot_data[id_col] = plot_data[id_col].astype(str)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(plot_data[id_col], plot_data[metric_delta])
    ax.axhline(0, linewidth=1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)

    if rotation:
        for label in ax.get_xticklabels():
            label.set_ha("right")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_model_task_delta_plot(
    task_delta: pd.DataFrame,
    metric_delta: str,
    output_path: Path,
):
    if metric_delta not in task_delta.columns:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_data = task_delta.copy()
    plot_data["model_task"] = (
        plot_data["model"].astype(str) + " / " + plot_data["task"].astype(str)
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(plot_data["model_task"], plot_data[metric_delta])
    ax.axhline(0, linewidth=1)
    ax.set_title(f"{metric_delta} by model and task")
    ax.set_ylabel(metric_delta)
    ax.tick_params(axis="x", rotation=45)

    for label in ax.get_xticklabels():
        label.set_ha("right")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_pair_case_files(
    pair_comparison: pd.DataFrame,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    change_cols = [
        "label_flip_bool_changed",
        "recommendation_flip_bool_changed",
        "choice_flip_bool_changed",
        "ranking_changed_bool_changed",
        "any_decision_flip_changed",
    ]

    existing_change_cols = [
        col for col in change_cols if col in pair_comparison.columns
    ]

    if existing_change_cols:
        decision_metric_changes = pair_comparison[
            pair_comparison[existing_change_cols].any(axis=1)
        ].copy()

        decision_metric_changes.to_csv(
            output_dir / "cases_decision_metric_changes.csv",
            index=False,
        )

    if "reasoning_similarity_delta" in pair_comparison.columns:
        pair_comparison.sort_values(
            "reasoning_similarity_delta",
            ascending=True,
        ).head(100).to_csv(
            output_dir / "cases_largest_reasoning_similarity_drop.csv",
            index=False,
        )

        pair_comparison.sort_values(
            "reasoning_similarity_delta",
            ascending=False,
        ).head(100).to_csv(
            output_dir / "cases_largest_reasoning_similarity_gain.csv",
            index=False,
        )

    if "confidence_shift_delta" in pair_comparison.columns:
        pair_comparison.reindex(
            pair_comparison["confidence_shift_delta"]
            .abs()
            .sort_values(ascending=False)
            .index
        ).head(100).to_csv(
            output_dir / "cases_largest_confidence_shift_delta.csv",
            index=False,
        )


def compare_runs(
    run_a_dir: Path,
    run_b_dir: Path,
    run_a_name: str,
    run_b_name: str,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    run_a_df = add_derived_columns(load_run_results(run_a_dir, run_a_name))
    run_b_df = add_derived_columns(load_run_results(run_b_dir, run_b_name))

    all_df = pd.concat([run_a_df, run_b_df], ignore_index=True)

    summary_by_model = compute_summary_by_model(all_df)
    summary_by_task = compute_group_summary(all_df, "task")
    summary_by_subtask = compute_group_summary(all_df, "subtask")
    summary_by_bias_axis = compute_group_summary(all_df, "bias_axis")

    delta_by_model = compute_delta(
        summary_by_model,
        id_cols=["model"],
        run_a=run_a_name,
        run_b=run_b_name,
    )

    delta_by_task = compute_delta(
        summary_by_task,
        id_cols=["model", "task"],
        run_a=run_a_name,
        run_b=run_b_name,
    )

    delta_by_subtask = compute_delta(
        summary_by_subtask,
        id_cols=["model", "subtask"],
        run_a=run_a_name,
        run_b=run_b_name,
    )

    delta_by_bias_axis = compute_delta(
        summary_by_bias_axis,
        id_cols=["model", "bias_axis"],
        run_a=run_a_name,
        run_b=run_b_name,
    )

    pair_comparison = compute_pair_level_comparison(
        run_a_df=run_a_df,
        run_b_df=run_b_df,
        run_a=run_a_name,
        run_b=run_b_name,
    )

    summary_by_model.to_csv(
        output_dir / "comparison_summary_by_model.csv",
        index=False,
    )
    summary_by_task.to_csv(
        output_dir / "comparison_summary_by_task.csv",
        index=False,
    )
    summary_by_subtask.to_csv(
        output_dir / "comparison_summary_by_subtask.csv",
        index=False,
    )
    summary_by_bias_axis.to_csv(
        output_dir / "comparison_summary_by_bias_axis.csv",
        index=False,
    )

    delta_by_model.to_csv(output_dir / "delta_by_model.csv", index=False)
    delta_by_task.to_csv(output_dir / "delta_by_task.csv", index=False)
    delta_by_subtask.to_csv(output_dir / "delta_by_subtask.csv", index=False)
    delta_by_bias_axis.to_csv(output_dir / "delta_by_bias_axis.csv", index=False)
    pair_comparison.to_csv(output_dir / "pair_level_comparison.csv", index=False)

    save_pair_case_files(pair_comparison, output_dir)

    save_grouped_run_plot(
        summary=summary_by_model,
        metric="any_decision_flip_rate",
        title="Decision flip rate by model and run",
        ylabel="Decision flip rate",
        path=plot_path(output_dir, "decision_flip_rate_by_model_and_run"),
    )

    save_grouped_run_plot(
        summary=summary_by_model,
        metric="reasoning_similarity_mean",
        title="Mean reasoning similarity by model and run",
        ylabel="Mean reasoning similarity",
        path=plot_path(output_dir, "mean_reasoning_similarity_by_model_and_run"),
    )

    save_grouped_run_plot(
        summary=summary_by_model,
        metric="reasoning_similarity_lt_080",
        title="Pairs with reasoning similarity below 0.80 by model and run",
        ylabel="Number of pairs",
        path=plot_path(output_dir, "reasoning_similarity_lt_080_by_model_and_run"),
    )

    save_grouped_run_plot(
        summary=summary_by_model,
        metric="confidence_shift_mean",
        title="Mean confidence shift by model and run",
        ylabel="Mean confidence shift",
        path=plot_path(output_dir, "mean_confidence_shift_by_model_and_run"),
    )

    save_delta_plot(
        delta=delta_by_model,
        id_col="model",
        metric_delta="any_decision_flip_rate_delta",
        title=f"Decision flip rate delta by model ({run_b_name} - {run_a_name})",
        ylabel="Decision flip rate delta",
        path=plot_path(output_dir, "delta_decision_flip_rate_by_model"),
    )

    save_delta_plot(
        delta=delta_by_model,
        id_col="model",
        metric_delta="reasoning_similarity_mean_delta",
        title=f"Mean reasoning similarity delta by model ({run_b_name} - {run_a_name})",
        ylabel="Reasoning similarity delta",
        path=plot_path(output_dir, "delta_reasoning_similarity_by_model"),
    )

    save_model_task_delta_plot(
        task_delta=delta_by_task,
        metric_delta="any_decision_flip_rate_delta",
        output_path=plot_path(output_dir, "delta_decision_flip_rate_by_model_task"),
    )

    save_model_task_delta_plot(
        task_delta=delta_by_task,
        metric_delta="reasoning_similarity_mean_delta",
        output_path=plot_path(output_dir, "delta_reasoning_similarity_by_model_task"),
    )

    print(f"Run A rows: {len(run_a_df)} from {run_a_dir}")
    print(f"Run B rows: {len(run_b_df)} from {run_b_dir}")
    print(f"Saved comparison outputs to {output_dir}")

    if "any_decision_flip_changed" in pair_comparison.columns:
        changed = pair_comparison["any_decision_flip_changed"].sum()
        print(f"Pairs with changed decision-flip status: {changed}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two SE4AI LLM evaluation runs."
    )

    parser.add_argument(
        "--run-a-dir",
        default=str(DEFAULT_RUN_A_DIR),
        help="Directory containing the first run results.",
    )

    parser.add_argument(
        "--run-b-dir",
        default=str(DEFAULT_RUN_B_DIR),
        help="Directory containing the second run results.",
    )

    parser.add_argument(
        "--run-a-name",
        default="run1",
        help="Name used for the first run in output tables.",
    )

    parser.add_argument(
        "--run-b-name",
        default="run2",
        help="Name used for the second run in output tables.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where comparison outputs are saved.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    compare_runs(
        run_a_dir=Path(args.run_a_dir),
        run_b_dir=Path(args.run_b_dir),
        run_a_name=args.run_a_name,
        run_b_name=args.run_b_name,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()