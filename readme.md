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

Generated PDFs, images, OCR data, and run outputs are intentionally ignored by Git. The synthetic image directory is removed after OCR unless `--keep-images` is supplied.

## Project status

Phase 0 scaffold: configuration, dependency pins, repository hygiene, and reproducible setup instructions.
