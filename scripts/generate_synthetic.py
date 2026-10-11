from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlocr.common import load_config
from rlocr.synth.generator import generate_pages

parser = argparse.ArgumentParser(); parser.add_argument("--config", default="config/default.yaml"); parser.add_argument("--num-pages", type=int); parser.add_argument("--keep-images", action="store_true")
args = parser.parse_args(); root = Path(__file__).resolve().parents[1]; generate_pages(load_config(root / args.config), root, args.num_pages, args.keep_images)
