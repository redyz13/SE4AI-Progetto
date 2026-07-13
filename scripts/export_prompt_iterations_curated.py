import argparse
import csv
import subprocess
from collections import defaultdict
from pathlib import Path


TRACKED_PATHS = [
    "data/prompts",
    "scripts/generate_base_scenarios.py",
    "scripts/generate_prompt_pairs.py",
    "src/prompts",
]


ITERATIONS = [
    {
        "id": "initial_single_task_template",
        "title": "Initial single-task prompt template",
        "change": (
            "The first prompt template was introduced for a single classification "
            "sub-task, mainly hiring screening."
        ),
        "motivation": (
            "Start from a simple controlled setting before extending the framework "
            "to multiple task families and sub-tasks."
        ),
    },
    {
        "id": "structured_prompt_pair_generation",
        "title": "Structured prompt-pair generation",
        "change": (
            "Prompt generation was moved to a structured pipeline combining "
            "templates, neutral scenarios, task instructions, and sensitive sentences."
        ),
        "motivation": (
            "Ensure that original and counterfactual prompts differ only in the "
            "controlled sensitive-attribute replacement."
        ),
    },
    {
        "id": "full_subtask_template_coverage",
        "title": "Full task/sub-task template coverage",
        "change": (
            "Task-specific templates were extended to all classification, "
            "recommendation, and decision-answering sub-tasks."
        ),
        "motivation": (
            "Support a broader evaluation design covering different LLM decision "
            "contexts while keeping outputs structured and comparable."
        ),
    },
    {
        "id": "subtask_specific_scenario_generation",
        "title": "Sub-task-specific neutral scenario generation",
        "change": (
            "Separate scenario-generation prompts were introduced for each sub-task."
        ),
        "motivation": (
            "Generate more realistic neutral scenarios tailored to each sub-task, "
            "instead of relying on one generic scenario-generation prompt."
        ),
    },
    {
        "id": "scenario_generation_export_refinement",
        "title": "Scenario generation and export refinement",
        "change": (
            "Scenario generation was refined and exported as structured JSONL records "
            "containing scenario_id, task, subtask, scenario_text, and task_instruction."
        ),
        "motivation": (
            "Make base scenarios reusable and traceable during counterfactual prompt-pair "
            "generation."
        ),
    },
    {
        "id": "proxy_name_rendering_correction",
        "title": "Proxy-name rendering correction",
        "change": (
            "Proxy names were handled as contextual names rather than explicit group "
            "descriptors."
        ),
        "motivation": (
            "Avoid unintentionally making the sensitive attribute explicit when the "
            "replacement value is a proxy name, such as a culturally associated name."
        ),
    },
    {
        "id": "neutral_contextual_labels",
        "title": "Neutral contextual labels in final prompts",
        "change": (
            "Explicit labels such as 'Sensitive attribute' were replaced with neutral "
            "contextual labels such as 'Additional applicant information' or "
            "'Additional candidate information'."
        ),
        "motivation": (
            "Reduce fairness-aware priming and make the sensitive information appear "
            "as ordinary contextual information while preserving the counterfactual setup."
        ),
    },
]


def run_git(repo_root: Path, args: list[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root)] + args,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def safe_run_git(repo_root: Path, args: list[str]) -> str:
    try:
        return run_git(repo_root, args)
    except subprocess.CalledProcessError:
        return ""


def get_commits(repo_root: Path) -> list[str]:
    output = safe_run_git(
        repo_root,
        ["rev-list", "--reverse", "HEAD", "--"] + TRACKED_PATHS,
    )
    if not output:
        return []
    return output.splitlines()


def get_commit_metadata(repo_root: Path, commit_hash: str) -> dict:
    output = run_git(
        repo_root,
        [
            "show",
            "-s",
            "--date=short",
            "--format=%H%x09%h%x09%ad%x09%s",
            commit_hash,
        ],
    )

    full_hash, short_hash, date, subject = output.split("\t", 3)

    return {
        "full_hash": full_hash,
        "short_hash": short_hash,
        "date": date,
        "subject": subject,
    }


def get_changed_files(repo_root: Path, commit_hash: str) -> list[str]:
    output = safe_run_git(
        repo_root,
        [
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            commit_hash,
            "--",
        ]
        + TRACKED_PATHS,
    )

    if not output:
        return []

    return output.splitlines()


def get_diff(repo_root: Path, commit_hash: str) -> str:
    return safe_run_git(
        repo_root,
        [
            "show",
            "--format=",
            "--unified=2",
            commit_hash,
            "--",
        ]
        + TRACKED_PATHS,
    )


def lower_join(*items) -> str:
    parts = []

    for item in items:
        if isinstance(item, list):
            parts.extend(item)
        else:
            parts.append(str(item))

    return " ".join(parts).lower()


def is_evaluation_template(path: str) -> bool:
    return (
        path.startswith("data/prompts/classification/")
        or path.startswith("data/prompts/recommendation/")
        or path.startswith("data/prompts/decision_answering/")
    ) and "scenario_generation" not in path


def is_scenario_generation_prompt(path: str) -> bool:
    return path.startswith("data/prompts/scenario_generation/")


def get_template_families(files: list[str]) -> set[str]:
    families = set()

    for file in files:
        if file.startswith("data/prompts/classification/"):
            families.add("classification")
        elif file.startswith("data/prompts/recommendation/"):
            families.add("recommendation")
        elif file.startswith("data/prompts/decision_answering/"):
            families.add("decision_answering")

    return families


def match_iteration_ids(subject: str, files: list[str], diff: str) -> list[str]:
    text = lower_join(subject, files, diff)

    matches = []

    evaluation_templates = [
        file for file in files if is_evaluation_template(file)
    ]

    scenario_generation_files = [
        file for file in files if is_scenario_generation_prompt(file)
    ]

    template_families = get_template_families(files)

    if (
        any("hiring_screening.txt" in file for file in files)
        and len(evaluation_templates) <= 2
        and "scenario_generation" not in text
    ):
        matches.append("initial_single_task_template")

    if (
        "generate_prompt_pairs.py" in text
        or "src/prompts" in text
        or "prompt pair" in text
        or "prompt_pairs" in text
    ):
        matches.append("structured_prompt_pair_generation")

    if (
        len(evaluation_templates) >= 4
        or len(template_families) >= 2
        or "all subtask" in text
        or "all sub-task" in text
        or "templates" in text
    ):
        matches.append("full_subtask_template_coverage")

    if (
        scenario_generation_files
        or "scenario_generation" in text
        or "generate_base_scenarios.py" in text
    ):
        matches.append("subtask_specific_scenario_generation")

    if (
        "base_scenarios" in text
        or "jsonl" in text
        or "scenario_id" in text
        or "task_instruction" in text
        or "generator_model" in text
        or "export" in text
    ):
        matches.append("scenario_generation_export_refinement")

    if (
        "proxy" in text
        or "proxy name" in text
        or "jamal" in text
        or "james" in text
        or "name rendering" in text
        or "descriptor" in text
    ):
        matches.append("proxy_name_rendering_correction")

    if (
        "sensitive attribute" in text
        or "additional applicant information" in text
        or "additional candidate information" in text
        or "additional student information" in text
        or "neutral label" in text
        or "contextual label" in text
    ):
        matches.append("neutral_contextual_labels")

    return list(dict.fromkeys(matches))


def collect_evidence(repo_root: Path) -> dict[str, list[dict]]:
    evidence = defaultdict(list)

    for commit_hash in get_commits(repo_root):
        metadata = get_commit_metadata(repo_root, commit_hash)
        files = get_changed_files(repo_root, commit_hash)
        diff = get_diff(repo_root, commit_hash)

        matched_ids = match_iteration_ids(
            subject=metadata["subject"],
            files=files,
            diff=diff,
        )

        for iteration_id in matched_ids:
            evidence[iteration_id].append(
                {
                    "date": metadata["date"],
                    "short_hash": metadata["short_hash"],
                    "full_hash": metadata["full_hash"],
                    "subject": metadata["subject"],
                    "files": files,
                }
            )

    return evidence


def summarize_dates(rows: list[dict]) -> str:
    if not rows:
        return "No matching Git evidence detected automatically."

    dates = sorted({row["date"] for row in rows})

    if len(dates) == 1:
        return dates[0]

    return f"{dates[0]} to {dates[-1]}"


def summarize_files(rows: list[dict], max_files: int = 8) -> str:
    files = []

    for row in rows:
        files.extend(row["files"])

    unique_files = sorted(set(files))

    if not unique_files:
        return "-"

    visible = unique_files[:max_files]
    rendered = "<br>".join(f"`{file}`" for file in visible)

    if len(unique_files) > max_files:
        rendered += f"<br>... and {len(unique_files) - max_files} more"

    return rendered


def summarize_evidence(rows: list[dict], include_hashes: bool) -> str:
    if not rows:
        return "No matching Git evidence detected automatically."

    date_summary = summarize_dates(rows)
    text = f"{len(rows)} prompt-related commit(s), {date_summary}"

    if include_hashes:
        commits = "<br>".join(
            f"`{row['short_hash']}`: {row['subject']}" for row in rows[:5]
        )

        if len(rows) > 5:
            commits += f"<br>... and {len(rows) - 5} more"

        text += f"<br>{commits}"

    return text


def write_markdown(
    output_md: Path,
    evidence: dict[str, list[dict]],
    include_hashes: bool,
    include_details: bool,
) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)

    with output_md.open("w", encoding="utf-8") as f:
        f.write("# Prompt Iteration History\n\n")

        f.write(
            "This document summarizes the main prompt-related iterations that led "
            "to the final neutral scenario-generation prompts and task-specific "
            "evaluation templates used in the paper.\n\n"
        )

        f.write(
            "The summary was reconstructed from the Git history of prompt files "
            "and prompt-generation scripts. It reports the main methodological "
            "iterations rather than every minor edit.\n\n"
        )

        f.write("## Tracked files\n\n")

        for path in TRACKED_PATHS:
            f.write(f"- `{path}`\n")

        f.write("\n## Iteration summary\n\n")

        f.write(
            "| Iteration | Change | Motivation | Evidence from Git history | Affected files |\n"
        )
        f.write(
            "|---|---|---|---|---|\n"
        )

        for index, iteration in enumerate(ITERATIONS, start=1):
            rows = evidence.get(iteration["id"], [])

            f.write(
                f"| {index}. {iteration['title']} "
                f"| {iteration['change']} "
                f"| {iteration['motivation']} "
                f"| {summarize_evidence(rows, include_hashes)} "
                f"| {summarize_files(rows)} |\n"
            )

        f.write("\n## Final prompt organization\n\n")

        f.write(
            "The final prompts used in the experiments are stored under "
            "`data/prompts/`.\n\n"
        )

        f.write(
            "- Neutral scenario-generation prompts are stored in "
            "`data/prompts/scenario_generation/`.\n"
        )
        f.write(
            "- Task-specific evaluation templates are stored in "
            "`data/prompts/classification/`, `data/prompts/recommendation/`, "
            "and `data/prompts/decision_answering/`.\n"
        )
        f.write(
            "- Neutral base scenarios are stored in "
            "`data/generated/base_scenarios.jsonl`.\n"
        )
        f.write(
            "- Original/counterfactual prompt pairs are stored in "
            "`data/generated/prompt_pairs.jsonl`.\n"
        )

        f.write("\n## Note\n\n")

        f.write(
            "The final version separates neutral task information from sensitive "
            "attribute information. The neutral scenario is generated first and "
            "stored as structured JSONL. Sensitive information is inserted only "
            "afterwards through controlled original/counterfactual replacements.\n"
        )

        if include_details:
            f.write("\n## Detailed Git evidence\n\n")

            for index, iteration in enumerate(ITERATIONS, start=1):
                rows = evidence.get(iteration["id"], [])

                f.write(f"### {index}. {iteration['title']}\n\n")

                if not rows:
                    f.write(
                        "No matching Git evidence was detected automatically for this iteration.\n\n"
                    )
                    continue

                for row in rows:
                    commit_label = (
                        f"`{row['short_hash']}`"
                        if include_hashes
                        else "commit"
                    )

                    f.write(
                        f"- {row['date']} — {commit_label} — {row['subject']}\n"
                    )

                    for file in sorted(set(row["files"])):
                        f.write(f"  - `{file}`\n")

                f.write("\n")


def write_csv(output_csv: Path, evidence: dict[str, list[dict]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iteration_id",
                "iteration_title",
                "date",
                "short_hash",
                "subject",
                "changed_files",
            ],
        )

        writer.writeheader()

        for iteration in ITERATIONS:
            rows = evidence.get(iteration["id"], [])

            for row in rows:
                writer.writerow(
                    {
                        "iteration_id": iteration["id"],
                        "iteration_title": iteration["title"],
                        "date": row["date"],
                        "short_hash": row["short_hash"],
                        "subject": row["subject"],
                        "changed_files": "; ".join(sorted(set(row["files"]))),
                    }
                )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a curated prompt-iteration Markdown file from Git history."
    )

    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to the Git repository root.",
    )

    parser.add_argument(
        "--output-md",
        default="docs/prompt_iterations.md",
        help="Path of the Markdown file to generate.",
    )

    parser.add_argument(
        "--output-csv",
        default="docs/prompt_iterations_evidence.csv",
        help="Path of the CSV evidence file to generate.",
    )

    parser.add_argument(
        "--include-commit-hashes",
        action="store_true",
        help="Include short commit hashes in the Markdown output.",
    )

    parser.add_argument(
        "--no-details",
        action="store_true",
        help="Do not include the detailed Git evidence section.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()

    evidence = collect_evidence(repo_root)

    write_markdown(
        output_md=Path(args.output_md),
        evidence=evidence,
        include_hashes=args.include_commit_hashes,
        include_details=not args.no_details,
    )

    write_csv(
        output_csv=Path(args.output_csv),
        evidence=evidence,
    )

    total_matches = sum(len(rows) for rows in evidence.values())

    print(f"Prompt-iteration evidence entries: {total_matches}")
    print(f"Markdown written to: {args.output_md}")
    print(f"CSV written to: {args.output_csv}")


if __name__ == "__main__":
    main()