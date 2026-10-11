from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from rlocr.common import ensure_dirs, load_config, read_json

parser = argparse.ArgumentParser(); parser.add_argument("--config", default="config/default.yaml"); parser.add_argument("--page-id", required=True); args = parser.parse_args(); root = Path(__file__).resolve().parents[1]; out = ensure_dirs(load_config(root / args.config), root); gt = read_json(out["gt"] / f"{args.page_id}.json"); aligned = read_json(out["aligned"] / f"{args.page_id}.json"); fig, ax = plt.subplots(figsize=(8.5, 11)); ax.set_xlim(0, gt["width"]); ax.set_ylim(gt["height"], 0)
for line in aligned["ocr_lines"]:
    x0,y0,x1,y1=line["bbox"]; ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color="tab:blue")); ax.text(x0,y0,str(line["native_order"]),fontsize=6)
for i, block_id in enumerate(gt["reading_order"][:-1]):
    a=next(b for b in gt["blocks"] if b["id"]==block_id); b=next(b for b in gt["blocks"] if b["id"]==gt["reading_order"][i+1]); ax.annotate("", ((b["bbox"][0]+b["bbox"][2])/2,(b["bbox"][1]+b["bbox"][3])/2), ((a["bbox"][0]+a["bbox"][2])/2,(a["bbox"][1]+a["bbox"][3])/2), arrowprops={"arrowstyle":"->","color":"red","lw":.7})
fig.savefig(out["viz"] / f"{args.page_id}.png", dpi=120, bbox_inches="tight"); plt.close(fig)
