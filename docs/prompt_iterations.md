# Prompt Iteration History

This document summarizes the main prompt-related iterations that led to the final neutral scenario-generation prompts and task-specific evaluation templates used in the paper.

The summary was reconstructed from the Git history of prompt files and prompt-generation scripts. It reports the main methodological iterations rather than every minor edit.

## Tracked files

- `data/prompts`
- `scripts/generate_base_scenarios.py`
- `scripts/generate_prompt_pairs.py`
- `src/prompts`

## Iteration summary

| Iteration | Change | Motivation | Evidence from Git history | Affected files |
|---|---|---|---|---|
| 1. Initial single-task prompt template | The first prompt template was introduced for a single classification sub-task, mainly hiring screening. | Start from a simple controlled setting before extending the framework to multiple task families and sub-tasks. | No matching Git evidence detected automatically. | - |
| 2. Structured prompt-pair generation | Prompt generation was moved to a structured pipeline combining templates, neutral scenarios, task instructions, and sensitive sentences. | Ensure that original and counterfactual prompts differ only in the controlled sensitive-attribute replacement. | 3 prompt-related commit(s), 2026-05-17 to 2026-05-19 | `data/prompts/classification/hiring_screening.txt`<br>`data/prompts/decision_answering/candidate_comparison.txt`<br>`data/prompts/hiring_screening_classification.txt`<br>`data/prompts/recommendation/career_path_recommendation.txt`<br>`scripts/generate_base_scenarios.py`<br>`scripts/generate_prompt_pairs.py`<br>`src/prompts/generator.py`<br>`src/prompts/prompts.py` |
| 3. Full task/sub-task template coverage | Task-specific templates were extended to all classification, recommendation, and decision-answering sub-tasks. | Support a broader evaluation design covering different LLM decision contexts while keeping outputs structured and comparable. | 3 prompt-related commit(s), 2026-05-17 to 2026-05-19 | `data/prompts/classification/eligibility_classification.txt`<br>`data/prompts/classification/hiring_screening.txt`<br>`data/prompts/classification/student_evaluation.txt`<br>`data/prompts/classification/ticket_priority_classification.txt`<br>`data/prompts/decision_answering/candidate_comparison.txt`<br>`data/prompts/decision_answering/public_service_priority_decision.txt`<br>`data/prompts/decision_answering/scholarship_allocation.txt`<br>`data/prompts/decision_answering/ticket_escalation_decision.txt`<br>... and 8 more |
| 4. Sub-task-specific neutral scenario generation | Separate scenario-generation prompts were introduced for each sub-task. | Generate more realistic neutral scenarios tailored to each sub-task, instead of relying on one generic scenario-generation prompt. | 3 prompt-related commit(s), 2026-05-17 to 2026-05-18 | `data/prompts/classification/hiring_screening.txt`<br>`data/prompts/decision_answering/candidate_comparison.txt`<br>`data/prompts/hiring_screening_classification.txt`<br>`data/prompts/recommendation/career_path_recommendation.txt`<br>`data/prompts/scenario_generation/classification/eligibility_classification.txt`<br>`data/prompts/scenario_generation/classification/hiring_screening.txt`<br>`data/prompts/scenario_generation/classification/student_evaluation.txt`<br>`data/prompts/scenario_generation/classification/ticket_priority_classification.txt`<br>... and 11 more |
| 5. Scenario generation and export refinement | Scenario generation was refined and exported as structured JSONL records containing scenario_id, task, subtask, scenario_text, and task_instruction. | Make base scenarios reusable and traceable during counterfactual prompt-pair generation. | 4 prompt-related commit(s), 2026-05-17 to 2026-05-18 | `data/prompts/classification/eligibility_classification.txt`<br>`data/prompts/classification/hiring_screening.txt`<br>`data/prompts/classification/student_evaluation.txt`<br>`data/prompts/classification/ticket_priority_classification.txt`<br>`data/prompts/decision_answering/candidate_comparison.txt`<br>`data/prompts/decision_answering/public_service_priority_decision.txt`<br>`data/prompts/decision_answering/scholarship_allocation.txt`<br>`data/prompts/decision_answering/ticket_escalation_decision.txt`<br>... and 20 more |
| 6. Proxy-name rendering correction | Proxy names were handled as contextual names rather than explicit group descriptors. | Avoid unintentionally making the sensitive attribute explicit when the replacement value is a proxy name, such as a culturally associated name. | 2 prompt-related commit(s), 2026-05-17 to 2026-05-19 | `data/prompts/classification/hiring_screening.txt`<br>`data/prompts/decision_answering/candidate_comparison.txt`<br>`data/prompts/hiring_screening_classification.txt`<br>`data/prompts/recommendation/career_path_recommendation.txt`<br>`scripts/generate_base_scenarios.py`<br>`scripts/generate_prompt_pairs.py`<br>`src/prompts/generator.py` |
| 7. Neutral contextual labels in final prompts | Explicit labels such as 'Sensitive attribute' were replaced with neutral contextual labels such as 'Additional applicant information' or 'Additional candidate information'. | Reduce fairness-aware priming and make the sensitive information appear as ordinary contextual information while preserving the counterfactual setup. | 5 prompt-related commit(s), 2026-05-17 to 2026-05-19 | `data/prompts/classification/eligibility_classification.txt`<br>`data/prompts/classification/hiring_screening.txt`<br>`data/prompts/classification/student_evaluation.txt`<br>`data/prompts/classification/ticket_priority_classification.txt`<br>`data/prompts/decision_answering/candidate_comparison.txt`<br>`data/prompts/decision_answering/public_service_priority_decision.txt`<br>`data/prompts/decision_answering/scholarship_allocation.txt`<br>`data/prompts/decision_answering/ticket_escalation_decision.txt`<br>... and 20 more |

## Final prompt organization

The final prompts used in the experiments are stored under `data/prompts/`.

- Neutral scenario-generation prompts are stored in `data/prompts/scenario_generation/`.
- Task-specific evaluation templates are stored in `data/prompts/classification/`, `data/prompts/recommendation/`, and `data/prompts/decision_answering/`.
- Neutral base scenarios are stored in `data/generated/base_scenarios.jsonl`.
- Original/counterfactual prompt pairs are stored in `data/generated/prompt_pairs.jsonl`.

## Note

The final version separates neutral task information from sensitive attribute information. The neutral scenario is generated first and stored as structured JSONL. Sensitive information is inserted only afterwards through controlled original/counterfactual replacements.
