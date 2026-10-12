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

The current 30-page training experiment reached mean train reward `-11.767`.
On a 10-page held-out subset, derived action accuracy was `0.8058`; the
heuristic reading-order accuracy was `0.6339`. These are not the same metric,
so this is evidence that the PPO loop runs and learns proxy labels, not proof
that reading order improves. A proper next iteration must construct corrected
orders from actions and score structural IoU/Kendall improvement directly.

Generated PDFs, images, OCR data, and run outputs are intentionally ignored by Git. The synthetic image directory is removed after OCR unless `--keep-images` is supplied.

## Project status

Phase 0 scaffold, the Phase 1/2 data layer, the baseline/Gymnasium/GAT phase, and the evaluation-focused PPO phase are complete. The 600-page run completed with mean Kendall tau `0.990`, native-order disagreement `8.2%`, and data usage of about `22.14 MB`. The baseline reports mean order accuracy `0.7066` over 600 pages. Generated data and checkpoints are ignored by Git.
