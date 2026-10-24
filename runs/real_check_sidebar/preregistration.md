# Final OmniDocBench sidebar-style real-data evaluation preregistration

## Status and hypothesis

This document is committed before selection by the geometric proxy, any new
image download, OCR, or model inference. This is a final, single-run external
evaluation motivated by earlier synthetic results, not a claim that previous
real-data results were positive. On the earlier 150-page real set no method
beat native Tesseract. That set is excluded in its entirety here.

Hypothesis: page-level PPO is at least as good as native and better on
sidebar-style pages; gated GAT and sequential PPO may lose. No parameter,
threshold, checkpoint, normalizer, eligibility rule or seed may be selected
or adjusted using results from these new pages. Negative results are reported.

## Frozen selection rule

Dataset: opendatalab/OmniDocBench, dataset revision
`aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec`, using the existing complete
`data/omnidocbench/OmniDocBench.json` annotation file. Exclude image paths
listed in `data/omnidocbench/selected_pages.json` (all earlier 150 pages).
Eligibility depends only on annotation metadata and ground-truth geometry,
never OCR, native correctness, tau, model outputs or model confidence.

- Language: `english`, as in the earlier evaluation and English OCR setup.
- Document type (`page_attribute.data_source`): `magazine`, `newspaper`,
  `colorful_textbook`, or `research_report`.
- Layout (`page_attribute.layout`): `single_column`, `double_column`,
  `three_column`, `1andmore_column`, or `other_layout`. All five recorded
  layouts are allowed; a layout label alone is not evidence of a sidebar.
- At least three nonignored ordered text annotations under the existing
  `rlocr.omnidocbench.text_blocks` definition.
- At least one qualifying sidebar/main pair of distinct, nonignored
  `text_block` annotations with numeric reading-order values. Bboxes are
  axis-aligned bounds of each annotation polygon, normalized to page size.
  The sidebar block must contain at least 40 non-whitespace text characters,
  have width <=0.35 of page width and height >=0.02 of page height, and have
  horizontal center <=0.25 or >=0.75 of page width. The main block must
  contain at least 100 non-whitespace characters, have width >=0.40 of page
  width, and be at least 1/0.70 times the sidebar width. Their horizontal
  interiors must not overlap, the horizontal gap must be <=0.15 page width,
  and vertical overlap must be >=0.50 of the shorter block's height.

The proxy is a geometric heuristic, not a verified sidebar label. Do not
manually retain or discard qualifying pages after looking at their content.
Print full-dataset document/layout counts, sequential eligibility counts,
and qualifying document/layout counts, including zero cells where relevant.

Target: 100 distinct pages. Minimum qualifying pool: 80 pages. Selection
seed: `20261024`; shuffle qualifying pages in annotation-index order using
Python 3.11 `random.Random(seed).shuffle`, then take the first min(100,N).
If N<80, report counts and stop for user direction: no rule relaxation,
downloads, OCR, model inference or substitute sample. If the fixed selected
images cannot fit the storage budget, stop rather than silently replacing
pages. No replacement of unscoreable pages after inference.

Download only individually selected images, resumably. Count selected images,
manifests and OCR/alignment caches plus the shared 42 MB annotation JSON toward
a 200,000,000-byte working-subset budget; do not duplicate the shared JSON.
Reserve 10,000,000 bytes for caches/metadata in preflight. Earlier unrelated
image downloads are not part of this new subset's budget. Never clone the
repository or download the images folder wholesale.

## Frozen evaluation and outcomes

Reuse existing English Tesseract, alignment criteria, full synthetic GAT
`runs/native_aware_gat_full.pt`, native prior 4.0, gate thresholds from
`runs/ppo_gate/thresholds.json`, and saved three-seed sequential PPO v2 and
page-level PPO checkpoints and normalizers. Seeds: 1337, 4242, 90210.
No retraining, tuning, checkpoint selection, threshold changes or repeated
evaluation in response to results. Cache OCR once; run each frozen method
once per predeclared input setting. Store orders and report from those caches.

Primary setting: all cached OCR lines, no 96-line cap. Secondary sensitivity
setting: the first 96 cached OCR lines for every method, including Native
and XY-cut. Use unchanged conservative text alignment: >=50% line-area
overlap; exclude lines with >=15% overlap with multiple eligible text blocks
or tied annotation orders. Require at least three matched text blocks and
three unambiguous lines to compute page tau. Keep all selected pages in
the report with counts, coverage and exclusion reasons. These are aligned
reading-order application metrics, not official OmniDocBench scores.

Primary comparison: page-level PPO minus Native mean page Kendall tau on
the all-lines setting. Average the three seed taus within each page before
paired page bootstrap (10,000 resamples, seed 314159). Report point estimate,
95% percentile CI, two-sided paired sign-flip p-value (10,000 draws, same
seed), and better/worse/same counts. The zero-margin claim "at least as good"
requires a CI lower bound >=0, not merely failure to reject a loss; evidence
of superiority requires a CI lower bound >0. Report uncertainty either way.

Secondary comparisons: page-level PPO versus gated GAT and versus sequential
PPO v2 (seed mean); XY-cut is the nonlearned control. Also report every method
versus Native and gated GAT. Per-seed and capped results, document-type
breakdowns, native-correct/native-wrong strata (explicitly GT-conditioned),
switch fractions and worst losses are descriptive/exploratory. No subgroup
result replaces the primary outcome. Distinguish actual per-seed switched
orders from any-seed-union diagnostics; a seed-mean tau has no output order.

## Artifacts and audit trail

Save selection manifest, frozen checkpoint/source hashes and selection counts.
Save `runs/real_check_sidebar/report.md`, `report.json`, and `per_page.csv`.
The preregistration commit hash must be printed before selection/evaluation
and included in the final report. Commit this preregistration separately
before any new-page model run; make one subsequent report/code commit and
push both. Do not commit downloaded images or corpus annotations.
