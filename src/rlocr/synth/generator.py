from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import fitz
import cv2
import numpy as np
from faker import Faker
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from rlocr.common import ensure_dirs, pt_bbox_to_px, write_json


FONTS = ["Helvetica", "Times-Roman", "Courier"]
PAGE_W, PAGE_H = letter


def _text(rng: random.Random, fake: Faker, words: int = 18) -> str:
    return fake.sentence(nb_words=words, variable_nb_words=False).rstrip(".") + "."


def _add_block(blocks: list[dict[str, Any]], block_id: str, kind: str, bbox: tuple[float, float, float, float], text: str, column: int, height: float, dpi: int) -> None:
    blocks.append({"id": block_id, "type": kind, "bbox": pt_bbox_to_px(bbox, PAGE_H, dpi), "text": text, "column_index": column, "_pt_bbox": bbox, "_height": height})


def _draw_wrapped(c: canvas.Canvas, text: str, x: float, y: float, width: float, font: str, size: float, leading: float) -> tuple[float, float]:
    c.setFont(font, size)
    words, line, lines = text.split(), "", []
    for word in words:
        trial = f"{line} {word}".strip()
        if c.stringWidth(trial, font, size) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    for index, value in enumerate(lines):
        c.drawString(x, y - index * leading, value)
    return y - len(lines) * leading, len(lines) * leading


def _column_page(c: canvas.Canvas, rng: random.Random, fake: Faker, template: str, dpi: int, page_id: str, hard: bool = False) -> tuple[list[dict[str, Any]], list[str]]:
    blocks: list[dict[str, Any]] = []
    order: list[str] = []
    margin = rng.uniform(34, 46) if hard else rng.uniform(42, 60)
    gap = rng.uniform(7, 13) if hard else rng.uniform(16, 28)
    cols = {"single_column": 1, "two_column": 2, "three_column": 3}.get(template, 1)
    col_w = (PAGE_W - 2 * margin - (cols - 1) * gap) / cols
    top = PAGE_H - margin
    for col in range(cols):
        x = margin + col * (col_w + gap)
        y = top
        for paragraph in range(rng.randint(4, 7)):
            if paragraph == 0 and col == 0:
                text, kind, size = fake.sentence(nb_words=rng.randint(4, 8)), "heading", rng.uniform(15, 19)
                text = text.rstrip(".")
            else:
                text, kind, size = _text(rng, fake, rng.randint(38, 58) if hard else rng.randint(24, 42)), "paragraph", rng.uniform(8.5, 10.5)
            font = rng.choice(FONTS)
            next_y, used = _draw_wrapped(c, text, x, y, col_w, font, size, size * 1.25)
            bbox = (x, next_y - 2, x + col_w, y + 4)
            block_id = f"{page_id}_b{len(blocks):03d}"
            _add_block(blocks, block_id, kind, bbox, text, col, used, dpi)
            order.insert(len(order) // 2, block_id)
            y = next_y - rng.uniform(8, 16)
            if y < 70:
                break
    return blocks, order


def _special_page(c: canvas.Canvas, rng: random.Random, fake: Faker, template: str, dpi: int, page_id: str, hard: bool = False) -> tuple[list[dict[str, Any]], list[str]]:
    blocks, order = _column_page(c, rng, fake, "single_column", dpi, page_id, hard)
    if template == "header_footer":
        header = fake.sentence(nb_words=6)
        c.setFont("Helvetica-Bold", 11); c.drawString(50, PAGE_H - 35, header)
        _add_block(blocks, f"{page_id}_header", "header", (50, PAGE_H - 45, PAGE_W - 50, PAGE_H - 22), header, -1, 12, dpi)
        footer = f"Page {rng.randint(1, 99)}"
        c.setFont("Helvetica", 8); c.drawCentredString(PAGE_W / 2, 24, footer)
        _add_block(blocks, f"{page_id}_footer", "footer", (PAGE_W / 2 - 30, 18, PAGE_W / 2 + 30, 30), footer, -1, 10, dpi)
        order = [blocks[-2]["id"]] + order + [blocks[-1]["id"]]
    elif template == "sidebar":
        text = fake.sentence(nb_words=12)
        x, y, w, h = (48, PAGE_H - 230, 110, 170) if hard else (PAGE_W - 150, PAGE_H - 230, 95, 150)
        c.setFillColorRGB(.92, .92, .96); c.rect(x, y, w, h, fill=1, stroke=1); c.setFillColorRGB(0, 0, 0)
        _draw_wrapped(c, text, x + 8, y + h - 18, w - 16, "Helvetica-Oblique", 9, 12)
        block_id = f"{page_id}_sidebar"
        _add_block(blocks, block_id, "sidebar", (x, y, x + w, y + h), text, 1, h, dpi)
        if hard:
            order.append(block_id)
        else:
            order.insert(min(2, len(order)), block_id)
    elif template == "heading_body_figure":
        x, y, w, h = 330, 155, 210, 125
        c.setFillColorRGB(.85, .85, .85); c.rect(x, y, w, h, fill=1, stroke=1); c.setFillColorRGB(0, 0, 0)
        figure = f"{page_id} figure"
        _add_block(blocks, f"{page_id}_figure", "figure", (x, y, x + w, y + h), figure, 1, h, dpi)
        caption = "Figure: " + fake.sentence(nb_words=7)
        c.setFont("Helvetica-Oblique", 8); c.drawString(x, y - 14, caption)
        _add_block(blocks, f"{page_id}_caption", "caption", (x, y - 18, x + w, y - 5), caption, 1, 10, dpi)
        order.extend([f"{page_id}_figure", f"{page_id}_caption"])
    elif template == "table":
        x, y, w, h = 50, 150, 510, 150
        rows, columns = 5, 4
        c.setFont("Helvetica", 8)
        for row in range(rows):
            for col in range(columns):
                cell_w, cell_h = w / columns, h / rows
                bx, by = x + col * cell_w, y + h - (row + 1) * cell_h
                c.rect(bx, by, cell_w, cell_h)
                text = fake.word() + " " + fake.word()
                c.drawString(bx + 4, by + cell_h / 2, text)
                block_id = f"{page_id}_cell{row}_{col}"
                _add_block(blocks, block_id, "table_cell", (bx, by, bx + cell_w, by + cell_h), text, 0, cell_h, dpi)
                order.append(block_id)
    return blocks, order


def generate_pages(config: dict[str, Any], root: Path, count: int | None = None, keep_images: bool = False) -> list[str]:
    output = ensure_dirs(config, root)
    rng = random.Random(int(config.get("seed", 1337)))
    fake = Faker(); fake.seed_instance(int(config.get("seed", 1337)))
    dpi = int(config.get("dpi", 150)); templates = config.get("templates", [])
    total = count or int(config.get("num_pages", 600)); page_ids = []
    for index in range(total):
        page_id = f"page_{index:04d}"; page_ids.append(page_id)
        template = templates[index % len(templates)] if templates else "single_column"
        if template == "mixed":
            template = rng.choice(["two_column", "sidebar", "heading_body_figure", "table"])
        pdf_path = output["pdf"] / f"{page_id}.pdf"
        image_path = output["images_tmp"] / f"{page_id}.png"
        gt_path = output["gt"] / f"{page_id}.json"
        if gt_path.exists() and image_path.exists():
            continue
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        c.setTitle(page_id)
        if template in {"single_column", "two_column", "three_column"}:
            blocks, order = _column_page(c, rng, fake, template, dpi, page_id, bool(config.get("hard_mode", False)))
        else:
            blocks, order = _special_page(c, rng, fake, template, dpi, page_id, bool(config.get("hard_mode", False)))
        c.save()
        doc = fitz.open(str(pdf_path)); pix = doc[0].get_pixmap(dpi=dpi, alpha=False); pix.save(str(image_path)); doc.close()
        degradation = config.get("degradation", {})
        if degradation.get("enabled", False):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            sigma = float(degradation.get("blur_sigma", 0.0))
            if sigma > 0:
                image = cv2.GaussianBlur(image, (0, 0), sigma)
            noise = float(degradation.get("noise_std", 0.0))
            if noise > 0:
                rng_noise = np.random.default_rng(int(config.get("seed", 1337)) + index)
                image = np.clip(image.astype(np.float32) + rng_noise.normal(0, noise, image.shape), 0, 255).astype(np.uint8)
            rotation = float(degradation.get("rotation_degrees", 0.0))
            if rotation > 0:
                rng_angle = random.Random(int(config.get("seed", 1337)) + index)
                angle = rng_angle.uniform(-rotation, rotation)
                matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1.0)
                image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderValue=255)
            cv2.imwrite(str(image_path), image)
        for block in blocks:
            block.pop("_pt_bbox", None); block.pop("_height", None)
        write_json(gt_path, {"page_id": page_id, "width": round(PAGE_W * dpi / 72), "height": round(PAGE_H * dpi / 72), "template": template, "blocks": blocks, "reading_order": order})
        if not keep_images:
            # Images are retained until OCR consumes them; the orchestrator removes them afterward.
            pass
    return page_ids
