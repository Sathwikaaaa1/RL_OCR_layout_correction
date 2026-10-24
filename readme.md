# RL-Based Post-OCR Document Correction

Data-layer prototype for correcting document structure from OCR output only. This phase creates deterministic synthetic DocLayNet-style layouts, OCR caches, and alignment labels. The model, Gymnasium environment, and PPO training are intentionally out of scope.

## Requirements

- Python 3.11
- A local Tesseract OCR installation; this project does not install system software.
- CPU-only execution is supported.

Create and activate a plain virtual environment, then install the pinned Python dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Tesseract is not on `PATH`, set `TESSERACT_CMD` or set `ocr.tesseract_cmd` in `config/default.yaml`. On Windows, for example:

```powershell
$env:TESSERACT_CMD = 'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## Commands

Generate synthetic pages and ground truth:

```powershell
python scripts/generate_synthetic.py --config config/default.yaml --num-pages 20
```

Run OCR and write the line-level cache:

```powershell
python scripts/run_ocr_cache.py --config config/default.yaml
```

Build the complete dataset (generation, OCR, alignment, and deterministic splits):

```powershell
python scripts/build_dataset.py --config config/default.yaml --num-pages 20
```

Visualize one aligned page:

```powershell
python scripts/visualize_sample.py --config config/default.yaml --page-id page_0000
```

Run tests:

```powershell
python -m pytest -q
```

Run the heuristic reading-order baseline:

```powershell
python scripts/run_baseline.py --aligned data/synthetic/aligned
```

The next-phase API is available as `rlocr.env.OCRCorrectionEnv` and
`rlocr.model.OCRGATEncoder`. The environment exposes four actions—retain,
merge, split, and link—and the GAT consumes node features plus an
`edge_index` tensor. These are intentionally small CPU prototypes; PPO and
learned reward shaping are available through the self-contained PPO trainer:

```powershell
python scripts/train_ppo.py --aligned data/synthetic/aligned --max-pages 20 --epochs 4 --output runs/ppo_policy.pt
```

The trainer derives structural action targets from alignment labels, applies
clipped PPO with GAE, and uses the GAT encoder for per-line policy/value
predictions. Rewards are currently shaped against those derived labels; this
is a training baseline, not a production reward model. Train/test selection
uses `data/synthetic/splits.json`:

```powershell
python scripts/train_ppo.py --split train --max-pages 30 --epochs 4 --output runs/ppo_train.pt
python scripts/evaluate_policy.py --split test --checkpoint runs/ppo_train.pt --max-pages 10
```

Training uses inverse-frequency behavior-cloning warm-starts before PPO so
rare Split/Link labels are not overwhelmed by dominant actions. Adjust the
warm-start duration with `--bc-epochs`.

The masked warm-start experiment reached action accuracy `0.8058` on the
10-page test subset, with macro action F1 `0.2200`. Accuracy is therefore
mostly reflecting the dominant Merge class; macro F1 is the more informative
metric for future policy improvements.

Link targets are now derived per line from native-vs-ground-truth order, and
the environment adds a Kendall-tau delta reward after each action. The first
20-page retraining run remained at corrected tau `0.9953` versus native tau
`0.9953`, so the policy has not demonstrated an order improvement yet.

Link-priority targets and a 90-page test evaluation were also checked. The
test split contains only four genuine native-order failures; corrected tau
remained `0.99855`, equal to native tau, and macro action F1 was `0.2271`.
This confirms that harder, stratified multi-column/sidebar examples are needed
before policy gains can be measured reliably.

Added `config/hard_eval.yaml` for a separate 40-page hard corpus: 10 pages
each of two-column, three-column, sidebar, and table layouts. Its native OCR
order disagreement is `22.5%` with mean Kendall tau `0.8843`. Training on its
28-page train split and evaluating six held-out pages currently gives policy
action accuracy `0.8279`, but corrected tau `0.8759`, equal to native tau.

An explicit pairwise-order GAT scorer is available via
`scripts/train_pairwise.py`. It predicts whether each line precedes every
other line and ranks from those scores. On the hard corpus it reached Kendall
tau `0.8060` after 4 epochs and `0.8564` after 20 epochs, still below native
`0.8759`; the implementation is valid but needs better features or training.

Stage 1 hard corpus reproduction:

```powershell
$env:TESSERACT_CMD = "F:\Tesseract-OCR\tesseract.exe"
python scripts/build_dataset.py --config config/hard_large.yaml --num-pages 800
python scripts/report_native_metrics.py --aligned data/synthetic_hard_large/aligned --gt data/synthetic_hard_large/gt
```

An audit found that the synthetic column generator had been interleaving
column blocks in `reading_order` instead of using column-major order. After
fixing that ground-truth bug and rebuilding the caches, the 800-page hard
corpus has mean native Kendall tau `0.917` and `31.0%` disagreement. The
corrected 600-page normal corpus has mean tau `0.989` and `12.5%` disagreement;
the old `0.215` result must not be used.

Stage 2 diagnostics rebuilt labels with adjacent same-block Merge pairs,
valid-line filtering, significant-overlap Split labels, and pairwise order
labels. The oracle decoder passes with Kendall tau `1.0` across 800 pages, and
the custom Kendall implementation matches SciPy. A recursive XY-cut reference
reaches mean tau `0.856` on the hard set, below native OCR because narrow
gutters and overlapping sidebars remain difficult. A pair-feature MLP reaches
pairwise accuracy `0.9983` and mean tau `0.8702` on 20 pages. The corrected
pairwise GAT reaches pairwise accuracy `0.9986` and mean tau `0.9521` after
200 epochs on the same 20 pages, passing the `0.95` mean-tau overfit gate but
with a worst-page tau of `0.7037`. Held-out policy improvement is not yet
demonstrated; Stage 3 remains blocked pending a clean validation evaluation.

Evaluation now reconstructs an action-conditioned reading order and reports
Kendall tau plus merge/split F1. The current held-out result is policy action
accuracy `0.8058`, corrected Kendall tau `0.9953`, native Kendall tau `0.9953`,
merge F1 `0.5122`, and split F1 `0.4000`. The equal Kendall scores mean this
checkpoint has not yet improved reading order; the next model iteration should
learn a link policy that changes ordering rather than only proxy action labels.

The current 30-page training experiment reached mean train reward `-11.767`.
On a 10-page held-out subset, derived action accuracy was `0.8058`; the
heuristic reading-order accuracy was `0.6339`. These are not the same metric,
so this is evidence that the PPO loop runs and learns proxy labels, not proof
that reading order improves. A proper next iteration must construct corrected
orders from actions and score structural IoU/Kendall improvement directly.

Generated PDFs, images, OCR data, and run outputs are intentionally ignored by Git. The synthetic image directory is removed after OCR unless `--keep-images` is supplied.

## Native-aware pairwise baseline

`src/rlocr/order_features.py` provides OCR-only pair features including geometry,
estimated same-column status, Tesseract block equality, native rank, native rank
difference, and a native-order prior. `scripts/gradient_boosted_baseline.py`
trains a scikit-learn `HistGradientBoostingClassifier`, tunes the prior on the
validation split, and reports native, XY-cut, and learned Kendall tau with
page-bootstrap confidence intervals, template breakdowns, unambiguous-line
metrics, pairwise accuracy, and decoder-noise probes.

Example full-split command (CPU-friendly sampling still includes every training
page):

```powershell
python scripts/gradient_boosted_baseline.py --aligned data/synthetic_hard_large/aligned --splits data/synthetic_hard_large/splits.json --split test --pairs-per-page 64 --max-iter 5 --output runs/gradient_boosted_baseline_test.json
```

The one-page smoke path completed successfully. The full 560-page fit was
attempted, but HistGradientBoosting exceeded the practical CPU/resource budget
in this environment before producing held-out metrics; no baseline result is
claimed yet. The existing held-out reference remains native tau `0.9282`,
XY-cut tau `0.8581`, and pairwise-GAT tau `0.8765`.

## Project status

Phase 0 scaffold, the Phase 1/2 data layer, baseline/Gymnasium/GAT prototypes,
and evaluation-focused PPO experiments are complete. The corrected 600-page
normal run has mean native Kendall tau `0.989` and `12.5%` disagreement.
Generated data and checkpoints are ignored by Git.

## Native-aware gated evaluation

Raw GAT inference now has one canonical implementation in
`src/rlocr/gat_inference.py`. The earlier gated report's raw tau of `0.90811`
was incorrect: that script used probability-Borda decoding and omitted the
checkpoint's tuned native prior of `4.0`. Re-evaluation produces test tau
`0.9286172431`, exactly matching the trainer's saved result. The corrected
120-page gated report is in `runs/gated_decoder/`; its validation-selected
gated tau is `0.933911`, and the per-page best of native/raw GAT is `0.934357`.

The larger sidebar-heavy corpus uses seed `271828` and can be built or resumed
with eight process-parallel Tesseract workers:

```powershell
python scripts/build_dataset.py --config config/sidebar_large.yaml --num-pages 1500 --workers 8
```

It contains 1,500 pages: 756 sidebar pages and 124 pages for each of
header/footer, heading/figure, single-column, table, three-column, and
two-column layouts. Native mean tau is `0.828959`, disagreement is `48.73%`,
and sidebar tau is `0.662065`. OCR output is cached atomically per page and the
temporary image is deleted only after that page succeeds.

Five-fold grouped cross-validation over the original 800 hard pages plus the
new 1,500 pages is resumable at both epoch and fold boundaries:

```powershell
python scripts/cross_validate_gat.py --output-dir runs/gat_cross_validation --max-epochs 2 --patience 1 --threads 1 --bootstrap 10000
```

Every page appears in exactly one out-of-fold test set. Each fold reserves 15%
of its non-test pages for early stopping and selection of the native prior and
gate thresholds. Pooled results over 2,300 pages are native `0.859682`, raw GAT
`0.866001`, gated GAT `0.885913`, and best-of(native, raw GAT) `0.879174`.
The gated-minus-native mean difference is `0.026231` with paired bootstrap
95% CI `[0.022489, 0.030106]` and sign-flip `p=0.000100`. On sidebar pages,
gating improves tau from `0.665882` to `0.732817`.

The requested best-of oracle selects a whole native or raw-GAT order per page.
It is not a mathematical upper bound for the pair-level gated decoder, which
can construct a third hybrid order; this is why gated tau can exceed that
diagnostic. Full tables, per-page predictions, fold checkpoints, selected
thresholds, and the progress log are under `runs/gat_cross_validation/`.

Diagnostics of these frozen out-of-fold predictions (no retraining):

```powershell
python scripts/diagnose_gated_gat.py
```

This saves `runs/diagnostics/report.md` and `report.json`, plus native/ground
truth/gated order visualizations of the ten worst pages in
`runs/viz_worse_pages/`. Statistics reuse the saved predictions; only the ten
visualized orders are reconstructed from their frozen fold checkpoints and
checked against the saved scores. Bootstrap uses 10,000 paired page resamples
with seed 271828. Reweighted intervals hold template proportions fixed and
resample within templates.

Gating harms 158 of 1,302 native-correct pages, with mean loss `0.013105`
conditional on changing them; 156 of those pages are two- or three-column.
All five folds improve. Mean gated-minus-native tau is `+0.010959` on the old
800-page corpus (95% CI `[0.006708, 0.015562]`) and `+0.034376` on the new
1,500-page corpus (`[0.029292, 0.039770]`). With equal template weights, the
gain is `+0.007922` (`[0.006700, 0.009149]`); matching the old template mix
gives `+0.013941` (`[0.011795, 0.016123]`). These intervals condition on the
saved models; they do not capture retraining uncertainty.

## PPO native override policy

Run the CPU benchmark or the complete three-seed experiment:

```powershell
python scripts/train_ppo_gate.py --benchmark-only
python scripts/train_ppo_gate.py --steps 200000 --seeds 1337 4242 90210
```

The first run recovers and caches each page's pair probabilities from its
saved held-out GAT fold. GAT weights remain frozen. PPO uses the shared
native-prior scores at prior `4.0`, a fixed 70/15/15 page split (seed 314159),
and a two-layer 64-unit MLP. Its two actions choose native's next unplaced OCR
line or GAT's highest-scoring unplaced line; matching proposals are automatic.
The existing OCR lines are the ordering units (no new block aggregation).
Stable-baselines3 is not installed in this environment, so the implementation
uses PyTorch PPO with GAE, clipping, and bootstrapping at rollout boundaries.

Each seed's best checkpoint is selected only on PPO-validation mean tau,
breaking ties by fewer overrides. The comparison gate's thresholds are also
selected only on PPO-validation. All seed selections finish before PPO-test
evaluation. Reports, per-page results, probability-cache provenance, selected
checkpoints, histories, and reward/validation-tau curves are saved under
`runs/ppo_gate/`.

Rewards follow the requested dense placement reward, override penalty, and
terminal tau improvement. Always following native has exactly zero terminal
improvement reward; its full episode return still includes dense rewards.
Tests verify this distinction, exact native retention, automatic placements,
ground-truth-free observations, override rewards, and GAE boundaries.

Each page's GAT output is held out of its own GAT fitting and model selection.
However, the original GAT folds overlap in their training data: this second
PPO split is conditional on the frozen OOF experiment, not a fully nested
independent end-to-end test. In particular, GAT fits producing PPO-training
features can include labels of pages subsequently assigned to PPO-test.

The three-seed run completed in `237.81` seconds after cache recovery, with
200,704 decision steps per seed. On the common 345-page PPO-test split, native
tau is `0.867968`, raw GAT (fixed prior 4.0) is `0.872964`, and the gate tuned
on PPO-validation is `0.891667` (T_pair=0.65, T_page=3). PPO's selected policies
average `0.870033 +/- 0.002920` across seeds, with native-correct tau
`0.996323 +/- 0.005200`; the acceptance rule therefore reports **PPO rejected**.
The mean-seed PPO-minus-gated difference is `-0.021634`, paired-bootstrap
95% CI `[-0.030248, -0.013766]`, sign-flip p=`0.000100`.

Validation selection included the initial policy checkpoint. Seed 1337 selects
step 40,960 and exactly retains native on the test pages. Seeds 4242 and 90210
select step-zero checkpoints because PPO training never improves their
validation selection scores. Thus the small aggregate gain over native is
not evidence of learned PPO improvement. The report records selected steps
and this limitation explicitly; the existing gated decoder remains the
stronger result from this experiment.

## Final PPO override attempt (v2)

```powershell
python scripts/train_ppo_gate_v2.py --diagnostics-only
python scripts/train_ppo_gate_v2.py --steps 200000 --seeds 1337 4242 90210
```

This reuses `runs/ppo_gate/oof_probabilities.pt`, its exact saved page splits,
and its validation-selected gate thresholds. It performs no GAT inference or
retraining. Reports are saved under `runs/ppo_gate_v2/`, including
`diagnostics.md`, `diagnostics.json`, `report.md`, `report.json`, teacher
decision logs, imitation checkpoints, fine-tuned/scratch checkpoints, and
training curves. The script benchmarks CPU throughput and prints a time
estimate before imitation and full PPO training.

The observation keeps the previous 13 features and adds four precomputed page
features: confident native/GAT pair disagreement count (`log1p(count)/5`),
maximum opposite-direction probability margin above 0.5, GAT/native order
tau, and native/XY-cut tau. Reward is now only `-0.005` per override plus
terminal `20 * (predicted_tau - native_tau)`. PPO uses normalized advantages,
learning rate `5e-5`, clipping `0.1`, and entropy coefficient `0.01`.
Imitation fits plain cross-entropy on gated-teacher choices from PPO-train
pages for 30 fixed epochs. Each of three seeds runs PPO from imitation and
from scratch; validation tau selects checkpoints, breaking ties by fewer
overrides. Fine-tuning can retain imitation; scratch selects trained
checkpoints only. The new acceptance rule compares fine-tuned mean-seed tau
against gated tau minus 0.002 and native-correct tau against gated
native-correct tau minus 0.001.

Diagnostics expose an action-space limitation: full pairwise-gated orders
can choose a next line that neither binary proposal offers. Teacher labels
therefore project the full gated order by choosing the earlier-ranked
available proposal, and the report explicitly measures that projection gap.
On the native-following PPO-test trajectory, 17.88% of decision points have
a correct GAT proposal, 77.29% a correct native proposal, and 4.84% neither.
The requested greedy per-decision oracle scores `0.885505`; it is not a
global tau ceiling. An exact dynamic program over unions of native/GAT
prefixes finds the actual best reachable tau `0.889527`. This is below the
mean-tau acceptance cutoff `0.889667` on these fixed test pages, so even a
perfect binary policy cannot meet that requirement. The full gated decoder
scores `0.891667`; its projected teacher scores `0.879938`, with different
orders on 94 of 345 test pages. These descriptive test diagnostics do not
feed fitting, hyperparameter selection, or checkpoint selection.

The completed v2 experiment took `532.21` seconds (8.87 minutes), including
diagnostics and six PPO runs of 200,704 steps. Mean test tau across three
seeds is imitation `0.879940 +/- 0.000005`, fine-tuned PPO
`0.881485 +/- 0.000264`, and scratch PPO `0.881821 +/- 0.000209`. All three
fine-tuned seeds select PPO-improved validation checkpoints over imitation.
Fine-tuned native-correct tau is `0.999578 +/- 0.000094`, better than the
gated decoder's `0.997336`, but overall tau remains below the acceptance
threshold: **PPO rejected**. Fine-tuned minus gated tau is `-0.010182`, with
paired-bootstrap 95% CI `[-0.016144, -0.004667]` and sign-flip p=`0.000600`.
PPO now demonstrates learned gains over native (`+0.013517`), while the
existing gated decoder remains the stronger overall method.

## Page-level PPO selection over the full gated decoder

```powershell
python scripts/train_ppo_page_gate.py
```

This cheap contextual-bandit experiment chooses once per page: native order
or the complete gated decoder order. It reuses the frozen probability cache,
the exact previous PPO splits, and the same validation-tuned gate thresholds.
Six OCR/model-only page features describe confident disagreement count and
margin, gated/native and native/XY-cut tau, line count, and narrow edge-line
fraction. Feature standardization is fitted on PPO-train only. Three fixed
seeds train a small two-layer MLP with PPO; validation alone selects trained
checkpoints, with always selecting gated order as an explicit fallback.

The complete run took `68.81` seconds. All three seeds selected learned PPO
checkpoints. On the same 345 test pages, native tau is `0.867968`, gated GAT
is `0.891667`, best-of-page oracle is `0.896744`, and page-level PPO is
`0.893463 +/- 0.000067`. The PPO-minus-gated improvement is `+0.001796`,
paired-bootstrap 95% CI `[0.000757, 0.003027]`, sign-flip p=`0.000200`:
**page-level PPO improved** under the specified CI rule. Native-correct tau
improves from gated GAT's `0.997336` to `0.998887`; it is not perfect retention.
PPO changes an average of 66.33 pages versus native, instead of gated GAT's
114, retaining native on roughly 48 pages the gate would otherwise change.

Full per-seed metrics, page predictions, 10,000-resample paired intervals,
checkpoint selections, histories, feature normalization, and provenance are
saved in `runs/ppo_page_gate/report.md` and `report.json`. These results remain
conditional on the reused frozen OOF experiment and its overlapping original
GAT training folds.

## Frozen-GAT RETAIN/REORDER policy and structural-label audit

Binary ordering actions now use the names **RETAIN = 0** and **REORDER = 1**.
This naming change preserves numeric IDs, old network defaults, and checkpoint
tensor layouts. RETAIN takes native's next unplaced line; REORDER takes the
GAT proposal (or the full gated order in the separate page-level environment).
The older four-action structural prototype retains its separate action schema.

```powershell
python scripts/cache_gat_embeddings.py
python scripts/train_frozen_gat_policy.py --benchmark-only
python scripts/train_frozen_gat_policy.py --steps 200000
python scripts/audit_merge_split.py
```

The encoder cache extracts each page's 32-dimensional node embeddings and
mean-pooled graph embedding using its saved held-out GAT fold, verifies fold
membership and checkpoint hashes, and aligns embeddings to cached OCR lines.
No GAT training occurs. A new 104-dimensional observation concatenates two
candidate embeddings, the pooled embedding, four page lookahead features,
placed fraction, and the last decision (none/RETAIN/REORDER). Only a 2x64 MLP
head trains, with train-only embedding/feature standardization. The exact
previous PPO-train/val/test splits are reused.

Reward remains terminal `20 * (tau - native_tau)` minus `0.005` per REORDER
attempt, with an extra `0.05` penalty for invalid edits. Placed lines are
masked. A REORDER skipping more than one OCR-inferred column, while native
stays in the current column, is rejected and executes RETAIN to ensure
progress. This is a combined embedding-and-validity-rule experiment, not a
pure encoder ablation. The report includes the complete state/actions/reward/
episode summary, masking rules, invalid counts, and a constrained greedy
per-decision oracle.

The six-run experiment completed in `403.66` seconds (6.73 minutes), with
200,704 steps per run. Test tau is PPO v2 `0.881485 +/- 0.000264`, frozen-GAT
scratch `0.880866 +/- 0.000173`, and frozen-GAT imitation-start
`0.880373 +/- 0.000319`. Imitation-start minus PPO v2 is `-0.001112`, paired
95% CI `[-0.002351, 0.000036]`, sign-flip p=`0.071393`. **GAT-policy PPO did
not improve**, so PPO v2 remains the main sequential-policy result. The
separate page-level PPO result remains available. Full metrics, per-seed
selections, cached embeddings, paired tests, and RETAIN/REORDER plots are in
`runs/ppo_gat_policy/`.

The read-only audit of all 2,300 full aligned pages found 75,670 merge-label
pairs, positive merge labels on every page, and 7,921 split-flagged lines on
1,798 pages. A minimal adjacent-native-line MERGE action is feasible: 73,678
positive pairs (97.37%) are directly adjacent in native order. The audit
defines eligible-pair merge F1, a cross-column penalty, and the necessary
always-MERGE baseline. These labels also cover ordinary paragraph line
aggregation and overlap-based ambiguity, so counts must not be presented as
verified OCR fragmentation/gluing defects. No MERGE or SPLIT action was
implemented. Findings and per-template counts are saved under
`runs/merge_split_audit/`.

## Small OmniDocBench real-data check (no training)

```powershell
python -m pip install -r requirements.txt
# Set TESSERACT_CMD to your existing Tesseract executable if it is not on PATH.
python scripts/prepare_omnidocbench.py
python scripts/evaluate_omnidocbench.py --workers 4
```

The downloader uses `huggingface_hub` with a pinned dataset revision and fetches
only the 42.21 MB annotation JSON and at most 150 explicitly selected images.
It never clones the dataset or downloads the whole images folder. The seeded
English subset contains 57 double-column, 32 three-column, and 61 mixed-column
pages, each with at least three ordered nonignored text blocks. No explicit
sidebar labels are available. `data/omnidocbench/selected_pages.json` records
the allowlist, attributes, sizes, seed and dataset revision. Downloads resume
without refetching completed images, target 200 MB including annotations,
and stop before reaching the 300 MB hard limit. Unicode filenames work on
Windows. All dataset files and caches remain ignored by Git.

Evaluation caches parallel English Tesseract OCR and runs the saved full
native-aware GAT (prior 4.0), fixed synthetic-validation gate, and all three
saved sequential PPO v2 and page-level PPO policies, without training,
real-data tuning, or real-data checkpoint selection. Native and XY-cut are
the baselines. Reports are `runs/omnidocbench/report.md` and `report.json`.
The existing synthetic checkpoints must be available locally. During a slow
download, `evaluate_omnidocbench.py --ocr-only` may cache available images
without producing partial-subset scores. See [the evaluation protocol](docs/real_data_protocol.md)
for alignment, coverage, the 96-line deployment cap, and metric limitations.

The completed subset occupies **185.01 MB** including OCR/alignment caches.
Of 150 selected pages, 149 are scoreable and one has insufficient aligned
blocks within the deployment cap. Mean line tau: Native `0.976917`, XY-cut
`0.936602`, raw GAT `0.951276`, gated GAT `0.952759`, sequential PPO v2
`0.958138 +/- 0.000666`, page-level PPO `0.972937 +/- 0.000405` (three seeds).
Page-level PPO minus native is `-0.003979`, paired 95% CI
`[-0.009896, 0.000577]`, sign-flip p=`0.131587`: **no demonstrated real-data
improvement over native**. Gated GAT is significantly worse than native
(`-0.024158`, CI `[-0.031328, -0.017788]`). These conditional application
scores cover 9,839 of 12,082 input OCR lines on scored pages; 65 pages exceed
the 96-line input cap. Block tau, all per-seed/per-layout metrics, exclusions,
and checkpoint hashes are in the reports. The final cached-OCR inference
and reporting invocation took about 52 seconds; this excludes earlier OCR
and network downloads. All 50 tests pass. No real-data fitting was performed.

### Stratified real-data analysis and cap audit

```powershell
python scripts/evaluate_omnidocbench.py --cached-ocr-only --max-lines 0 --output runs/real_check/stratified/uncapped_inference
python scripts/stratify_real_check.py
```

The original 96-line setting truncates **every method**, including Native and
the XY-cut control, before scoring aligned text. It is a pipeline/training-size
constant, not an architecture limit. An inference-only all-lines run (maximum
213 OCR lines) preserves all checkpoints, thresholds, normalizers and original
results; no OCR, training, tuning or selection is repeated. It scores all 150
pages: Native `0.976384`, XY-cut `0.956493`, raw GAT `0.949155`, gated GAT
`0.951870`, sequential PPO v2 seed mean `0.956465`, page-level PPO seed mean
`0.972972`. Larger graphs do not close the gap. A third comparison restricts
the larger-graph predictions to the original 149 pages' scoring lines, so
expanded coverage is not confused with a same-support gain.

`runs/real_check/stratified/` contains per-page CSV/JSON tables (including
explicit exclusions), complete bootstrap stratum tables, switched-page
diagnostics, nine plots, `report.md` and `report.json`. Strata use annotation
layout, pre-cap line count, native correctness, and native/XY-cut agreement;
no learned outputs define groups. Native-wrong is a ground-truth-conditioned
diagnostic, not a deployable gate. XY-cut is the control. Three PPO seeds are
averaged within page before the fixed-seed 10,000-resample paired bootstrap;
actual per-seed switch counts are also shown.

Native error rate is `22.82%` on original capped scoring versus `25.33%` on
all-lines scoring, `32.50%` on the original 560-page GAT training split, and
`43.39%` on the pooled 2,300-page OOF corpus. The report also isolates the PPO
training split (native error rate `43.73%`) and each synthetic corpus. These are descriptive distribution
comparisons with different template mixes and alignment/input filtering.
Gated GAT worsens 79/115 native-correct pages in the original scoring;
seed-mean page PPO worsens 9/115. Harm persists in both line-count groups.
All 55 tests pass, including paired-bootstrap toy cases and GAT inference on
more than 96 nodes. The all-lines frozen inference took 89 seconds; the
saved-prediction analysis and plots take about 25-36 seconds on this CPU.

### PPO template analysis and synthetic-to-real decision quality

```powershell
python scripts/analyze_ppo_policies.py
```

This saved-prediction-only analysis takes about 6 seconds, with no training,
tuning, OCR or inference. Outputs are `runs/ppo_analysis/report.md`,
`report.json` and `analysis_plan.json`. It verifies the identical 345-page
synthetic PPO-test split and shared baselines, reports every observed template
and each PPO seed plus the within-page seed mean, and uses 10,000 paired page
bootstrap resamples and two-sided sign flips with seed 314159. Template
p-values receive Holm correction separately per method/reference family.
The two policy-family primary tests and two secondary tests also receive
their own Holm corrections. This is a pre-specified **retrospective** analysis
of already inspected data, not a new independent confirmatory holdout.

Primary comparison (143 synthetic sidebar pages), PPO minus native:
sequential PPO v2 `+0.034157`, CI `[0.023552, 0.045481]`; page-level PPO
`+0.061586`, CI `[0.040602, 0.084181]`. Secondary comparison (all 345 synthetic
test pages), PPO minus gated GAT: sequential `-0.010182`, CI
`[-0.016144, -0.004667]`; page-level `+0.001796`, CI
`[0.000757, 0.003027]`. Per-seed rows and other template cells are exploratory.

Exploratory real all-lines comparisons against gated GAT: sequential PPO
`+0.004595`, CI `[0.002575, 0.006856]`; page-level PPO `+0.021102`, CI
`[0.017035, 0.025847]`. Neither beats native. Among actual page-policy switches,
synthetic improvement fractions are 76.1%-78.5% across seeds, versus
12.5%-16.7% on real pages. Decision tables distinguish any tau improvement
from full restoration to tau=1, report broken native-correct pages, and show
both net improved-minus-worsened and net fully-fixed-minus-broken counts.

### Final unseen sidebar real-data preregistration: selection stopped

Preregistration was committed and pushed as `f9949a4` before geometric
selection or any new-page download/OCR/inference. It fixes English language,
four document types, a geometric narrow-edge-block/wide-main-block proxy,
seed 20261024, a target of 100, and a minimum qualifying pool of 80. All
earlier 150 images are excluded. The zero-margin primary comparison and
frozen all-lines/capped evaluation plan are recorded in
`runs/real_check_sidebar/preregistration.md`.

```powershell
python scripts/select_sidebar_real_check.py
```

Annotation-only selection found 1,501 unseen pages, 605 English pages, 178
pages in the declared document types, and 169 with at least three ordered
text blocks. **Only three pages match the preregistered geometric proxy:**
two magazine and one newspaper page, all `other_layout`; none qualify from
colorful textbooks or research reports. The run stopped as required, with
no final sample, downloads, OCR, model inference, tuning or rule relaxation.
`runs/real_check_sidebar/report.md`, `report.json` and `selection.json` save
the count audit and geometric witnesses. These are selection feasibility
findings, not model performance results. Explicit user direction is required
before changing the rule or evaluating an undersized pool.

### User-authorized three-page exploratory follow-up

After the minimum-pool stop, the user explicitly authorized evaluating the
three unseen candidates and comparing with earlier pages of the same types.
This follow-up does **not** replace or count as the preregistered 80-100-page
final evaluation. The preregistration and stop reports remain unchanged.

```powershell
python scripts/evaluate_sidebar_followup.py --prepare
python scripts/evaluate_omnidocbench.py --root data/omnidocbench_sidebar_followup --annotations data/omnidocbench/OmniDocBench.json --max-lines 0 --output runs/real_check_sidebar/followup/new_inference --workers 3
python scripts/evaluate_sidebar_followup.py
```

Set `TESSERACT_CMD` if necessary. Only three images are downloaded; the
annotation JSON is shared without copying. All-lines OCR/inference took 56
seconds, including a 424-line newspaper page, with unchanged checkpoints,
gate thresholds, prior and normalizers. No model was fitted or tuned.

Mean tau on the three unseen pages: Native `0.964912`, XY-cut `0.964905`,
raw GAT `0.896730`, gated GAT `0.871477`, sequential PPO v2 three-seed mean
`0.893430`, page-level PPO mean `0.915569`. Page-level PPO minus native is
`-0.049343`, paired 95% CI `[-0.140351, 0.000000]`; no superiority or
zero-margin noninferiority is demonstrated by this tiny sample.

The seen comparison reuses all 34 earlier magazine/newspaper pages (12/22),
not a score-selected subset: Native and page-level PPO both `0.981516`,
XY-cut `0.958153`, raw GAT `0.964551`, gated GAT `0.972370`, sequential PPO
`0.973412`. None of these 34 seen pages satisfy the frozen sidebar proxy,
so this is a document-type comparison, not a matched sidebar-style cohort.
Separate new, seen and explicitly exploratory combined results, per-type
and per-page scores, individual seeds, CIs and switch diagnostics are saved
under `runs/real_check_sidebar/followup/` in `report.md`, `report.json` and
`per_page.csv`. All 61 tests pass.
