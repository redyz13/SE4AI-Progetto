import gc
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import torch

from llm.inference import run_prompt
from llm.models import load_model


def load_prompt_pairs(input_path: Path) -> list[dict]:
    rows = []

    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def extract_json(text: str) -> dict:
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)

        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    return {
        "label": None,
        "confidence": None,
        "reason": text,
    }


def compute_confidence_shift(original_output: dict, counterfactual_output: dict):
    original_confidence = original_output.get("confidence")
    counterfactual_confidence = counterfactual_output.get("confidence")

    if isinstance(original_confidence, int) and isinstance(counterfactual_confidence, int):
        return abs(original_confidence - counterfactual_confidence)

    return None


def normalize_reason_text(text) -> str:
    if not isinstance(text, str):
        return ""

    return " ".join(text.lower().strip().split())


def compute_reasoning_metrics(
    original_output: dict,
    counterfactual_output: dict,
) -> dict:
    original_reason = normalize_reason_text(original_output.get("reason"))
    counterfactual_reason = normalize_reason_text(counterfactual_output.get("reason"))

    if not original_reason or not counterfactual_reason:
        return {
            "reasoning_changed": None,
            "reasoning_similarity": None,
            "reasoning_length_shift": None,
        }

    reasoning_similarity = SequenceMatcher(
        None,
        original_reason,
        counterfactual_reason,
    ).ratio()

    return {
        "reasoning_changed": original_reason != counterfactual_reason,
        "reasoning_similarity": round(reasoning_similarity, 4),
        "reasoning_length_shift": abs(
            len(original_reason.split()) - len(counterfactual_reason.split())
        ),
    }


def compute_ranking_instability(
    original_ranking: list,
    counterfactual_ranking: list,
):
    if len(original_ranking) != len(counterfactual_ranking):
        return None

    original_positions = {
        item: index
        for index, item in enumerate(original_ranking)
    }

    counterfactual_positions = {
        item: index
        for index, item in enumerate(counterfactual_ranking)
    }

    if set(original_positions.keys()) != set(counterfactual_positions.keys()):
        return None

    total_distance = 0

    for item in original_positions:
        total_distance += abs(
            original_positions[item] - counterfactual_positions[item]
        )

    return total_distance


def compute_task_metrics(
    task: str,
    original_output: dict,
    counterfactual_output: dict,
) -> dict:
    confidence_shift = compute_confidence_shift(
        original_output=original_output,
        counterfactual_output=counterfactual_output,
    )

    reasoning_metrics = compute_reasoning_metrics(
        original_output=original_output,
        counterfactual_output=counterfactual_output,
    )

    if task == "classification":
        original_label = original_output.get("label")
        counterfactual_label = counterfactual_output.get("label")

        label_flip = None
        if original_label is not None and counterfactual_label is not None:
            label_flip = original_label != counterfactual_label

        return {
            "label_flip": label_flip,
            "confidence_shift": confidence_shift,
            **reasoning_metrics,
        }

    if task == "recommendation":
        original_recommended_option = original_output.get("recommended_option")
        counterfactual_recommended_option = counterfactual_output.get("recommended_option")

        recommendation_flip = None
        if (
            original_recommended_option is not None
            and counterfactual_recommended_option is not None
        ):
            recommendation_flip = (
                original_recommended_option != counterfactual_recommended_option
            )

        original_ranking = original_output.get("ranking")
        counterfactual_ranking = counterfactual_output.get("ranking")

        ranking_changed = None
        ranking_instability = None

        if isinstance(original_ranking, list) and isinstance(counterfactual_ranking, list):
            ranking_changed = original_ranking != counterfactual_ranking
            ranking_instability = compute_ranking_instability(
                original_ranking=original_ranking,
                counterfactual_ranking=counterfactual_ranking,
            )

        return {
            "recommendation_flip": recommendation_flip,
            "ranking_changed": ranking_changed,
            "ranking_instability": ranking_instability,
            "confidence_shift": confidence_shift,
            **reasoning_metrics,
        }

    if task == "decision_answering":
        original_selected_subject = original_output.get("selected_subject")
        counterfactual_selected_subject = counterfactual_output.get("selected_subject")

        choice_flip = None
        if original_selected_subject is not None and counterfactual_selected_subject is not None:
            choice_flip = original_selected_subject != counterfactual_selected_subject

        return {
            "choice_flip": choice_flip,
            "confidence_shift": confidence_shift,
            **reasoning_metrics,
        }

    return {
        "confidence_shift": confidence_shift,
        **reasoning_metrics,
    }


def build_result_row(
    model_key: str,
    pair: dict,
    original_text: str,
    counterfactual_text: str,
    original_output: dict,
    counterfactual_output: dict,
    metrics: dict,
) -> dict:
    task = pair["task"]

    base_row = {
        "model_key": model_key,
        "pair_id": pair.get("pair_id"),
        "scenario_id": pair.get("scenario_id"),
        "task": task,
        "subtask": pair["subtask"],
        "bias_axis": pair["bias_axis"],
        "replacement_group": pair["replacement_group"],
        "original_value": pair.get("original_value"),
        "counterfactual_value": pair.get("counterfactual_value"),
        "original_term": pair["original_term"],
        "counterfactual_term": pair["counterfactual_term"],
    }

    reasoning_row = {
        "reasoning_changed": metrics.get("reasoning_changed"),
        "reasoning_similarity": metrics.get("reasoning_similarity"),
        "reasoning_length_shift": metrics.get("reasoning_length_shift"),
    }

    if task == "classification":
        task_row = {
            "original_label": original_output.get("label"),
            "counterfactual_label": counterfactual_output.get("label"),
            "original_confidence": original_output.get("confidence"),
            "counterfactual_confidence": counterfactual_output.get("confidence"),
            "label_flip": metrics.get("label_flip"),
            "confidence_shift": metrics.get("confidence_shift"),
            **reasoning_row,
        }

    elif task == "recommendation":
        task_row = {
            "original_recommended_option": original_output.get("recommended_option"),
            "counterfactual_recommended_option": counterfactual_output.get("recommended_option"),
            "original_ranking": original_output.get("ranking"),
            "counterfactual_ranking": counterfactual_output.get("ranking"),
            "recommendation_flip": metrics.get("recommendation_flip"),
            "ranking_changed": metrics.get("ranking_changed"),
            "ranking_instability": metrics.get("ranking_instability"),
            "original_confidence": original_output.get("confidence"),
            "counterfactual_confidence": counterfactual_output.get("confidence"),
            "confidence_shift": metrics.get("confidence_shift"),
            **reasoning_row,
        }

    elif task == "decision_answering":
        task_row = {
            "original_selected_subject": original_output.get("selected_subject"),
            "counterfactual_selected_subject": counterfactual_output.get("selected_subject"),
            "choice_flip": metrics.get("choice_flip"),
            "original_confidence": original_output.get("confidence"),
            "counterfactual_confidence": counterfactual_output.get("confidence"),
            "confidence_shift": metrics.get("confidence_shift"),
            **reasoning_row,
        }

    else:
        task_row = {
            "original_confidence": original_output.get("confidence"),
            "counterfactual_confidence": counterfactual_output.get("confidence"),
            "confidence_shift": metrics.get("confidence_shift"),
            **reasoning_row,
        }

    common_output_row = {
        "original_reason": original_output.get("reason"),
        "counterfactual_reason": counterfactual_output.get("reason"),
        "original_raw_output": original_text,
        "counterfactual_raw_output": counterfactual_text,
    }

    return {
        **base_row,
        **task_row,
        **common_output_row,
    }


def save_results(
    results: list[dict],
    output_dir: Path,
    model_key: str,
    model_id: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_model_name = model_id.replace("/", "__")

    all_output_path = output_dir / f"results_{model_key}_{safe_model_name}_all.csv"

    df = pd.DataFrame(results)
    df.to_csv(all_output_path, index=False)

    output_paths = {
        "all": all_output_path,
    }

    task_column_sets = {
        "classification": [
            "model_key",
            "pair_id",
            "scenario_id",
            "task",
            "subtask",
            "bias_axis",
            "replacement_group",
            "original_value",
            "counterfactual_value",
            "original_term",
            "counterfactual_term",
            "original_label",
            "counterfactual_label",
            "original_confidence",
            "counterfactual_confidence",
            "label_flip",
            "confidence_shift",
            "reasoning_changed",
            "reasoning_similarity",
            "reasoning_length_shift",
            "original_reason",
            "counterfactual_reason",
            "original_raw_output",
            "counterfactual_raw_output",
        ],
        "recommendation": [
            "model_key",
            "pair_id",
            "scenario_id",
            "task",
            "subtask",
            "bias_axis",
            "replacement_group",
            "original_value",
            "counterfactual_value",
            "original_term",
            "counterfactual_term",
            "original_recommended_option",
            "counterfactual_recommended_option",
            "original_ranking",
            "counterfactual_ranking",
            "recommendation_flip",
            "ranking_changed",
            "ranking_instability",
            "original_confidence",
            "counterfactual_confidence",
            "confidence_shift",
            "reasoning_changed",
            "reasoning_similarity",
            "reasoning_length_shift",
            "original_reason",
            "counterfactual_reason",
            "original_raw_output",
            "counterfactual_raw_output",
        ],
        "decision_answering": [
            "model_key",
            "pair_id",
            "scenario_id",
            "task",
            "subtask",
            "bias_axis",
            "replacement_group",
            "original_value",
            "counterfactual_value",
            "original_term",
            "counterfactual_term",
            "original_selected_subject",
            "counterfactual_selected_subject",
            "choice_flip",
            "original_confidence",
            "counterfactual_confidence",
            "confidence_shift",
            "reasoning_changed",
            "reasoning_similarity",
            "reasoning_length_shift",
            "original_reason",
            "counterfactual_reason",
            "original_raw_output",
            "counterfactual_raw_output",
        ],
    }

    for task, columns in task_column_sets.items():
        task_df = df[df["task"] == task].copy()

        if task_df.empty:
            continue

        available_columns = [
            column
            for column in columns
            if column in task_df.columns
        ]

        task_output_path = output_dir / f"results_{model_key}_{safe_model_name}_{task}.csv"
        task_df[available_columns].to_csv(task_output_path, index=False)

        output_paths[task] = task_output_path

    return output_paths


def run_evaluation(
    model_key: str,
    model_id: str,
    project_root: Path,
    input_path: Path,
    output_dir: Path,
    max_new_tokens: int = 180,
    quantization: str = "none",
) -> Path:
    tokenizer, model, _ = load_model(
        model_id=model_id,
        quantization=quantization,
    )

    prompt_pairs = load_prompt_pairs(input_path)

    results = []

    for index, pair in enumerate(prompt_pairs, start=1):
        print(f"\nRunning pair {index}/{len(prompt_pairs)}")
        print(f"{pair['original_term']} -> {pair['counterfactual_term']}")

        original_text = run_prompt(
            tokenizer=tokenizer,
            model=model,
            prompt=pair["original_prompt"],
            max_new_tokens=max_new_tokens,
        )

        counterfactual_text = run_prompt(
            tokenizer=tokenizer,
            model=model,
            prompt=pair["counterfactual_prompt"],
            max_new_tokens=max_new_tokens,
        )

        original_output = extract_json(original_text)
        counterfactual_output = extract_json(counterfactual_text)

        metrics = compute_task_metrics(
            task=pair["task"],
            original_output=original_output,
            counterfactual_output=counterfactual_output,
        )

        row = build_result_row(
            model_key=model_key,
            pair=pair,
            original_text=original_text,
            counterfactual_text=counterfactual_text,
            original_output=original_output,
            counterfactual_output=counterfactual_output,
            metrics=metrics,
        )

        results.append(row)

        print("Original:", original_output)
        print("Counterfactual:", counterfactual_output)
        print("Metrics:", metrics)

    output_paths = save_results(
        results=results,
        output_dir=output_dir,
        model_key=model_key,
        model_id=model_id,
    )

    del model
    del tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nSaved results:")
    for name, path in output_paths.items():
        print(f"- {name}: {path}")

    return output_paths["all"]