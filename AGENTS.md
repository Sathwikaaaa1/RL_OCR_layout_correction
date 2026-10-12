# Project Brief

## PROJECT
RL-Based Post-OCR Document Correction. OCR engines (Tesseract etc.) read characters well but break document STRUCTURE: they merge columns, split paragraphs, and scramble reading order. We keep the OCR engine and train a small RL agent that fixes structure using only OCR output (bounding boxes, text, confidence), never raw pixels.
- State: spatial graph, one node per OCR box (normalized bbox, text features, confidence), kNN edges, encoded by a small GAT.
- Actions: Merge, Split, Link (reading order), Retain.
- Reward (later): improvement in structural IoU and reading-order accuracy vs ground truth; penalties for merging across columns or splitting a sentence.
- Training (later): PPO in a custom Gymnasium env.

## HARD CONSTRAINTS
- CPU only. Low disk usage. Everything must be free and open source. No paid APIs, no cloud accounts, no Docker.
- Windows/macOS/Linux compatible: use pathlib, no hardcoded separators. Tesseract path configurable via env var TESSERACT_CMD or config.
- Use a plain venv (.venv). Pin versions in requirements.txt. Python 3.11.
- Do not install system-level software. If something is missing (for example the tesseract binary), stop and tell me exactly what to install.

## SCOPE OF THIS TASK (data layer only, synthetic data)
Do NOT build the model, RL environment, or PPO yet. Build only:

### Phase 0: Scaffold
- Create the folder structure: config/, src/rlocr/{synth,ocr,data}/, scripts/, tests/, docs/, data/, runs/.
- requirements.txt with only what this phase needs: reportlab (or fpdf2), PyMuPDF, opencv-python-headless, pytesseract, numpy, Pillow, faker, pyyaml, tqdm, pytest, matplotlib.
- .gitignore: .venv, __pycache__, data/, runs/, *.png, *.jpg, *.pdf, .env, and large files. Only code, configs, docs, and tiny sample outputs (put 3 examples in docs/samples/) are committed.
- config/default.yaml: seed, number of pages, DPI (default 150), split ratios (70/15/15), templates to use, OCR language, paths.
- README.md with setup steps and how to run each script.

### Phase 1: Synthetic page generator (src/rlocr/synth, scripts/generate_synthetic.py)
Generate DocLayNet-style pages (call them "DocLayNet-style layouts", not DocLayNet). Page templates:
1. single column body text
2. two-column
3. three-column
4. header + footer + page number
5. sidebar / pull-quote beside body text
6. heading + body + figure placeholder with caption
7. simple table (grid with text cells)
8. mixed layouts combining the above
Randomize: margins, column gaps, font family/size from reportlab built-in fonts, paragraph counts and lengths, heading levels. Text comes from Faker (offline, free); no downloads.
For every page, write ground truth JSON (data/synthetic/gt/{page_id}.json):
- page_id, width, height, template
- blocks: list of {id, type (heading|paragraph|header|footer|sidebar|caption|table_cell|figure), bbox [x0,y0,x1,y1] in pixels, text, column_index}
- reading_order: ordered list of block ids (true logical reading order, e.g. column 1 top to bottom then column 2; sidebars and headers/footers placed sensibly)
Ground truth must be exact by construction. Render to PDF then to PNG with PyMuPDF at the configured DPI. Optional mild degradations (slight blur, noise, +/-1 degree rotation) controlled by config.

Images go to data/synthetic/images_tmp/ and are DELETED after OCR to save disk (flag --keep-images to retain).

### Phase 2: OCR + alignment cache (src/rlocr/ocr, scripts/run_ocr_cache.py, scripts/build_dataset.py)
- Run Tesseract once per page via pytesseract.image_to_data; build LINE-level boxes by grouping on (block_num, par_num, line_num). Keep bbox, text, mean confidence, and Tesseract's own block/par ids and native reading order. Save to data/synthetic/ocr/{page_id}.json. Make the OCR runner an interface so a second engine (RapidOCR) can be added later without refactoring.
- Alignment: map every OCR line to a ground-truth block using bbox overlap (IoU or intersection over line area) with text-similarity (rapidfuzz) as a tie-breaker. Produce data/synthetic/aligned/{page_id}.json containing: ocr_lines, gt_blocks, ocr_to_gt mapping (a line may map to one block, or none), and derived structural labels:
  * merge_pairs: OCR lines that belong to the same GT block but were separated
  * split_needed: OCR lines whose box spans more than one GT block (for example columns glued together)
  * gt_order vs tesseract_order (reading order sequences over OCR lines)
  * simple stats: how many pages/lines have each error type
- Deterministic splits (train/val/test by seed) written to data/synthetic/splits.json.
- scripts/visualize_sample.py: draws OCR boxes (colored by aligned GT block) and GT reading order arrows for a page and saves a PNG to runs/viz/. Use it to sanity check alignment.
- scripts/build_dataset.py: one command that runs generate -> OCR -> align -> splits, with --num-pages and resumable (skip pages already done).

### Success criteria for this task
- First run a 20-page smoke test and report seconds per page for each stage so I can estimate time for 600 pages. Then run 600 pages total (default) unless that looks impractical on CPU; tell me the estimate before running if it exceeds roughly 45 minutes.
- Report error statistics: percentage of pages where Tesseract's native reading order disagrees with ground truth (Kendall tau mean), and counts of merge/split cases per template. Tesseract should visibly fail on multi-column, sidebar, and table pages. If it never fails, flag this and make the layouts harder.
- pytest tests for: generator determinism with a fixed seed, GT bbox validity (inside page, non-negative area), alignment correctness on a hand-built toy example, and JSON schema loading.
- Total disk used by data/ reported at the end.

## GIT / GITHUB WORKFLOW (important)
- Remote: <REPO_URL>. If this folder is not a git repo, run git init, set branch to main, and add the remote origin.
- Commit and push after EVERY milestone (end of Phase 0, Phase 1, Phase 2, and after tests pass), not only at the end. Use small, conventional commits (for example "feat(synth): add two-column template", "test: add alignment toy case").
- Push to main after each commit. Never force-push. Never commit data/, images, .venv, secrets, or files over 5 MB.
- Update README.md at each milestone so it always reflects what exists and how to run it.
- If a push fails (auth, remote rejects), stop and tell me the exact error. Do not work around it.

## WORKING RULES
- Make reasonable assumptions and state them briefly. if blocked, ask the user.
- Keep code simple, typed, and commented where non-obvious. No unnecessary dependencies.
- Run the code and tests yourself before claiming something works; show actual output.
- At the end give me: a summary of what was built, the exact commands to reproduce, timing and error statistics, disk usage, anything that failed or looks risky, and what the next task should be (baseline + Gymnasium env + GAT).
- Do not start the next task. Stop after this scope.

Start from phase 0

## Status

The initial data pipeline, baseline/Gymnasium/GAT/PPO prototypes, hard 40-page corpus, and pairwise GAT scorer are implemented. Stage 1 of the gated continuation is complete: the 800-page hard corpus has 100% native-order disagreement and mean Kendall tau 0.215, with 52.59 MB on disk. The latest verified state is 12 passing tests before Stage 1 changes; native OCR still outperforms the learned order models. Stage 2 is the active clean-label diagnostics gate.
