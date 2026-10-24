# Small real-data reading-order check

The dataset revision is pinned in `scripts/prepare_omnidocbench.py`. Only
`OmniDocBench.json` and an image allowlist are downloaded, never a repository
clone or image-folder snapshot. Page attributes are taken from
`page_info.page_attribute`; image names come from `page_info.image_path`.
Selection is seeded (1337), English only, double/three/mixed columns or an
explicit sidebar label, with at least three nonignored ordered text blocks.
There is no explicit sidebar category in the current release. A seeded
shuffle selects at most 150 pages without consulting OCR or model scores.
The default 200 MB decimal budget includes annotations; 1 MB is reserved for
manifest/download metadata. A 300 MB pre-download guard stops further fetches.

The frozen evaluation uses the saved full synthetic native-aware GAT, native
prior 4.0, saved synthetic-validation gate thresholds, and all three saved
page-level PPO and sequential PPO v2 seeds. No model, threshold, normalizer,
or seed is selected using real-data results. GAT input is OCR only, capped at
the checkpoint's 96 lines, without annotation-based input filtering.
Sequential rollout receives dummy native ranks for its unused reward API,
not real annotation ranks. Tesseract OCR caches are resumable and images are
retained. `--ocr-only` may cache available images during downloads, but never
produces partial-subset evaluation scores.

Primary evaluation maps OCR lines to annotated text blocks with at least
50% line-area overlap; text similarity only breaks geometric ties. Lines
overlapping multiple text blocks by at least 15%, and tied block-order
annotations, are excluded. Boxes are scaled from annotation to image size.
Paragraph line order is derived top-to-bottom. Pages need at least three
mapped blocks and lines; excluded pages and coverage are reported. Complementary
block tau deduplicates aligned annotation IDs by first occurrence, avoiding
the within-paragraph ordering assumption. Both metrics cover only observable,
aligned text and do not measure OCR recognition, tables, math, or merge/split
quality. These are application metrics, not official OmniDocBench scores.

Report mean page tau, native-correct/native-wrong subsets, per-layout results,
paired page bootstrap intervals and sign-flip tests (10,000 resamples, seed
314159). Seed means average within page before resampling, not additional
independent pages. Report truncation, alignment coverage, exclusions, hashes,
and disk usage alongside performance. Interpret domain-transfer performance
subject to those limitations; do not change the protocol after seeing scores.

Schema source: [OmniDocBench dataset card](https://huggingface.co/datasets/opendatalab/OmniDocBench).
