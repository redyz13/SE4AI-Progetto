# 🐈‍⬛‍️ SE4AI — Counterfactual Fairness Testing for Generative AI 🦇

## 📌 Overview

This repository contains a framework for **counterfactual fairness testing of generative language models**, developed as part of a **SE4AI course**.

The project evaluates whether Large Language Models change their behavior when a prompt is modified only with respect to a sensitive attribute. Starting from neutral task scenarios, the framework generates controlled original/counterfactual prompt pairs, runs multiple LLMs on the same inputs, and compares their outputs through task-specific stability metrics.

The framework currently supports three task families:

- Classification
- Recommendation
- Decision answering

The goal is to provide a repeatable process for detecting and analyzing fairness-related instabilities in generative AI systems.

---

## 🎯 Goals

The project aims to:

- Identify relevant sensitive attributes from existing fairness datasets and benchmarks;
- Organize tasks, subtasks, bias axes, and replacement values into a structured taxonomy;
- Generate neutral scenarios and controlled counterfactual prompt pairs;
- Evaluate multiple generative language models on the same prompt pairs;
- Compare outputs using decision-level, ranking-level, confidence-level, and reasoning-level metrics;
- Analyze the reproducibility of results across independent evaluation runs;
- Support quantitative analysis through CSV outputs and plots, as well as manual audit of relevant cases.

The focus is on **local counterfactual stability**: if two prompts differ only in a sensitive attribute that is irrelevant to the task, the model should ideally produce equivalent task-relevant outputs.

---

## 🧠 Research Context

Large Language Models are increasingly used in classification, recommendation, and decision-support contexts. In these settings, sensitive attributes such as gender, race, ethnicity, nationality, religion, disability, age, socioeconomic status, sexual orientation, physical appearance, or language background may affect model outputs in undesirable ways.

Traditional fairness evaluation often relies on static benchmarks or group-level metrics. This project adopts a **counterfactual testing** perspective: a model is evaluated on pairs of prompts that are semantically equivalent except for one controlled sensitive attribute change.

The project is inspired by research on:

- Counterfactual fairness
- Bias detection in language models
- Behavioral testing of NLP systems
- Prompt-based fairness evaluation
- Invariance testing
- Explanation-level sensitivity in LLM outputs

---

## 🧪 Methodology

The framework follows a pipeline based on controlled prompt generation and output comparison.

```text
Taxonomy construction
        ↓
Neutral scenario generation
        ↓
Counterfactual prompt-pair generation
        ↓
LLM evaluation
        ↓
Metrics, plots, summaries, audit cases, and run comparison
```

### 1️⃣ Taxonomy Construction

The taxonomy defines task types, subtasks, sensitive bias axes, replacement groups, and original/counterfactual terms.

It was built by analyzing and consolidating bias-related resources and datasets, including **CrowS-Pairs**, **BBQ**, and **HolisticBias**.

The final taxonomy files are stored in:

```text
outputs/taxonomy/
```

The main files are:

```text
final_task_taxonomy.json
final_replacement_taxonomy.json
final_taxonomy_light.json
final_taxonomy_extended.json
```

---

### 2️⃣ Scenario Generation

Neutral base scenarios are generated for each supported subtask. These scenarios describe the task context without directly inserting sensitive attributes.

Scenario generation templates are stored in:

```text
data/prompts/scenario_generation/
```

Generated scenarios are saved in dedicated folders:

```text
data/generated/
data/generated2/
```

Each folder contains:

```text
base_scenarios.jsonl
prompt_pairs.jsonl
```

Example command:

```bash
python scripts/generate_base_scenarios.py \
  --scenarios-per-subtask 3 \
  --output data/generated/base_scenarios.jsonl
```

---

### 3️⃣ Counterfactual Prompt-Pair Generation

Prompt pairs are generated from the neutral scenarios, the taxonomy, and task-specific prompt templates.

Each pair contains:

- Original prompt
- Counterfactual prompt

The two prompts differ only in the inserted sensitive attribute value.

Prompt templates are organized by task in:

```text
data/prompts/classification/
data/prompts/recommendation/
data/prompts/decision_answering/
```

Example command:

```bash
python scripts/generate_prompt_pairs.py \
  --scenarios data/generated/base_scenarios.jsonl \
  --output data/generated/prompt_pairs.jsonl \
  --max-pairs-per-group 1
```

The prompt templates use neutral contextual labels such as:

```text
Additional candidate information:
Additional learner information:
Additional information about Candidate A:
```

instead of explicitly marking the inserted information as a `Sensitive attribute`. This preserves the counterfactual structure while making the prompt less explicitly fairness-aware.

---

### 4️⃣ Model Evaluation

The framework evaluates multiple LLMs on the same prompt pairs.

Current target models are:

```python
MODEL_REGISTRY = {
    "qwen": "Qwen/Qwen3-8B",
    "mistral": "mistralai/Ministral-8B-Instruct-2410",
    "llama": "NousResearch/Meta-Llama-3.1-8B-Instruct",
}
```

Evaluation is performed with:

```text
scripts/run_llm_eval.py
```

Example command:

```bash
python scripts/run_llm_eval.py \
  --model qwen \
  --input data/generated/prompt_pairs.jsonl \
  --max-new-tokens 120 \
  --output-dir outputs/llm_runs \
  --quantization 8bit
```

Repeat the evaluation command with `--model mistral` and `--model llama` to evaluate all target models.

The full pipeline can also be executed with:

```text
scripts/run_pipeline.py
```

Example command:

```bash
python scripts/run_pipeline.py \
  --models qwen mistral llama \
  --scenarios-per-subtask 3 \
  --max-pairs-per-group 1 \
  --max-new-tokens 120 \
  --quantization 8bit
```

To reuse existing scenarios and prompt pairs:

```bash
python scripts/run_pipeline.py \
  --models qwen mistral llama \
  --skip-scenario-generation \
  --skip-prompt-generation \
  --max-new-tokens 120 \
  --quantization 8bit
```

---

## 📊 Metrics

The framework computes task-specific metrics and shared stability metrics.

For **classification** tasks:

```text
label_flip
confidence_shift
reasoning_changed
reasoning_similarity
reasoning_length_shift
```

For **recommendation** tasks:

```text
recommendation_flip
ranking_changed
ranking_instability
confidence_shift
reasoning_changed
reasoning_similarity
reasoning_length_shift
```

For **decision answering** tasks:

```text
choice_flip
confidence_shift
reasoning_changed
reasoning_similarity
reasoning_length_shift
```

Decision-level metrics measure whether the final task output changes between the original and counterfactual prompt.

Ranking-level metrics evaluate whether the order of recommended options changes.

Confidence-level metrics measure changes in the model's self-reported confidence.

Reasoning-level metrics compare the generated explanations and are used as auxiliary audit signals. They help identify cases where the final decision remains stable but the explanation changes substantially.

---

## 📁 Outputs

Evaluation results are stored in dedicated run folders.

First evaluation run:

```text
outputs/llm_runs/
```

Second evaluation run:

```text
outputs/llm_run2/
```

Both folders follow the same structure, with one subfolder per model:

```text
outputs/llm_runs/Qwen/
outputs/llm_runs/Mistral/
outputs/llm_runs/LLama/

outputs/llm_run2/Qwen/
outputs/llm_run2/Mistral/
outputs/llm_run2/LLama/
```

Each model produces:

```text
results_<model>_<model_id>_all.csv
results_<model>_<model_id>_classification.csv
results_<model>_<model_id>_recommendation.csv
results_<model>_<model_id>_decision_answering.csv
```

---

## 📈 Plotting and Analysis

Summary tables, plots, heatmaps, and audit files are generated with:

```text
scripts/summary_plots.py
```

For the first run:

```bash
python scripts/summary_plots.py \
  --input-dir outputs/llm_runs \
  --output-dir outputs/plots
```

For the second run:

```bash
python scripts/summary_plots.py \
  --input-dir outputs/llm_run2 \
  --output-dir outputs/plots2
```

The generated artifacts include aggregate summaries, stability plots, reasoning-similarity plots, confidence-shift plots, heatmaps, and audit CSV files for decision flips and low-reasoning-similarity cases.

```text
outputs/plots/  -> Analysis artifacts for run 1
outputs/plots2/ -> Analysis artifacts for run 2
```

---

## 🔁 Run Comparison

Two independent evaluation runs can be compared with:

```text
scripts/compare_runs.py
```

Example command:

```bash
python scripts/compare_runs.py \
  --run-a-dir outputs/llm_runs \
  --run-b-dir outputs/llm_run2 \
  --run-a-name run1 \
  --run-b-name run2 \
  --output-dir outputs/run_comparison
```

The comparison outputs are stored in:

```text
outputs/run_comparison/
```

They include aggregate deltas, pair-level comparisons, plots, and audit CSV files useful for checking run-to-run reproducibility.

Decision-level deltas are the primary signal for run-to-run stability. Reasoning-similarity deltas are auxiliary and mainly support manual audit of cases where explanations become more or less stable across runs.

---

## 🚀 Quick Start

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate neutral scenarios:

```bash
python scripts/generate_base_scenarios.py \
  --scenarios-per-subtask 3 \
  --output data/generated/base_scenarios.jsonl
```

Generate counterfactual prompt pairs:

```bash
python scripts/generate_prompt_pairs.py \
  --scenarios data/generated/base_scenarios.jsonl \
  --output data/generated/prompt_pairs.jsonl \
  --max-pairs-per-group 1
```

Run model evaluation:

```bash
python scripts/run_llm_eval.py \
  --model qwen \
  --input data/generated/prompt_pairs.jsonl \
  --max-new-tokens 120 \
  --output-dir outputs/llm_runs \
  --quantization 8bit
```

Repeat the evaluation command with `--model mistral` and `--model llama` to evaluate all target models.

Generate summaries and plots:

```bash
python scripts/summary_plots.py \
  --input-dir outputs/llm_runs \
  --output-dir outputs/plots
```

Compare two runs:

```bash
python scripts/compare_runs.py \
  --run-a-dir outputs/llm_runs \
  --run-b-dir outputs/llm_run2 \
  --run-a-name run1 \
  --run-b-name run2 \
  --output-dir outputs/run_comparison
```

---

## ⚠️ Notes and Limitations

This framework evaluates **counterfactual stability**, not classical group fairness.

It does not estimate group-level metrics such as demographic parity, equalized odds, or equal opportunity, because the generated benchmark does not contain population-level ground-truth distributions.

A decision flip indicates a candidate counterfactual instability and should be manually inspected. Reasoning metrics are also auxiliary: a low reasoning similarity may indicate explanation-level sensitivity, but it can also result from harmless paraphrasing.

Run comparison helps evaluate reproducibility across independent executions, but differences between runs may still be affected by generation variability, decoding behavior, quantization, and model-level nondeterminism.

Quantized inference may slightly affect generated outputs. For comparability, the same quantization setup should be used across evaluated models whenever possible.

---

## 🛠️ Repository Status

The framework is functional and supports scenario generation, prompt-pair generation, multi-model evaluation, metric computation, result visualization, and comparison between independent runs.

Further work may include expanding the taxonomy, increasing scenario diversity, improving semantic reasoning comparison, and testing mitigation strategies.

---

## 👥 Credits

This project is developed as part of a SE4AI course.

The repository represents an effort to apply software engineering principles to the evaluation and analysis of fairness-related behavior in generative AI systems.