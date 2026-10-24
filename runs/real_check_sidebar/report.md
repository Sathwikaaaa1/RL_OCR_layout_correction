# Final real-data sidebar check: selection stopped

Only 3 unseen pages satisfy the frozen rule, below the minimum of 80. Stopped before downloads, OCR or model inference. User decision is required before any rule change.

Preregistration committed and pushed before selection: `f9949a4`. Selection seed: 20261024. Target: 100; minimum eligible pool: 80.

On the earlier 150-page set no method beat native. This new set has NOT been evaluated.

No rule was relaxed, no page was substituted, and no new-page model run was performed. This is a selection feasibility report, not a negative or positive model result.

## Eligibility funnel

| Stage | Pages |
|---|---:|
| all_annotations | 1651 |
| unseen | 1501 |
| unseen_english | 605 |
| excluded_previous_images | 150 |
| allowed_document_type | 178 |
| allowed_layout | 178 |
| at_least_three_ordered_text_blocks | 169 |
| qualifying_unique_pages | 3 |

## All document types

| Attribute | Pages |
|---|---:|
| book | 276 |
| academic_literature | 215 |
| note | 118 |
| PPT2PDF | 253 |
| colorful_textbook | 159 |
| historical_document | 5 |
| magazine | 149 |
| exam_paper | 193 |
| research_report | 132 |
| newspaper | 151 |

## All layout types

| Attribute | Pages |
|---|---:|
| single_column | 887 |
| other_layout | 372 |
| double_column | 184 |
| 1andmore_column | 155 |
| three_column | 53 |

## Qualifying document types

| Attribute | Pages |
|---|---:|
| magazine | 2 |
| newspaper | 1 |
| colorful_textbook | 0 |
| research_report | 0 |

## Qualifying layout types

| Attribute | Pages |
|---|---:|
| single_column | 0 |
| double_column | 0 |
| three_column | 0 |
| 1andmore_column | 0 |
| other_layout | 3 |

Qualifying annotation indices and geometric witnesses are saved in selection.json. No final page sample was selected. Reproduce counts with `python scripts/select_sidebar_real_check.py`. The next action requires explicit user direction; do not relax eligibility or evaluate this undersized pool automatically.
