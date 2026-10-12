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

The 800-page hard corpus uses a new seed and has 100% native-order
disagreement, mean Kendall tau `0.215`, and disk usage of `52.59 MB`.

Stage 2 diagnostics rebuilt labels with adjacent same-block Merge pairs,
valid-line filtering, significant-overlap Split labels, and pairwise order
labels. The oracle decoder passed with Kendall tau `1.0` across 800 pages, and
the custom Kendall implementation matched SciPy. The required neural
overfit gate failed: the pairwise GAT reached only `0.4642` mean tau on the
same 20 pages after 100 epochs. Stage 3 is intentionally blocked until this
model/feature failure is fixed.

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

## Project status

Phase 0 scaffold, the Phase 1/2 data layer, the baseline/Gymnasium/GAT phase, and the evaluation-focused PPO phase are complete. The 600-page run completed with mean Kendall tau `0.990`, native-order disagreement `8.2%`, and data usage of about `22.14 MB`. The baseline reports mean order accuracy `0.7066` over 600 pages. Generated data and checkpoints are ignored by Git.
