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
