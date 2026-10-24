"""Extract held-out-fold GAT embeddings once, without encoder training."""

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.env import OCRCorrectionEnv
from rlocr.frozen_gat_policy import inferred_columns
from rlocr.gat_inference import gat_inference_from_logits
from rlocr.model import PairwiseOrderModel
from rlocr.ppo import spatial_edges
from train_ppo_gate_v2 import load_experiment


def cache_embeddings(splits, args, provenance):
    pages = [page for group in splits.values() for page in group]
    metadata = {}
    for fold in sorted({page["fold"] for page in pages}):
        model_path = args.cv_dir / f"fold_{fold}.pt"
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if model_hash != provenance["gat_cache_provenance"]["checkpoints"][str(fold)]:
            raise ValueError("Saved GAT checkpoint differs from the probability cache source")
        expected = {"checkpoint_sha256": model_hash, "probability_cache_sha256": provenance["cache_sha256"], "version": 1}
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        fold_pages = [page for page in pages if page["fold"] == fold]
        for page in fold_pages:
            if page["key"] not in checkpoint["test_keys"] or page["key"] in checkpoint["train_keys"] or page["key"] in checkpoint["validation_keys"]:
                raise ValueError(f"Embedding leakage: {page['key']}")
        target = args.output / "embeddings" / f"fold_{fold}.pt"
        if target.exists():
            saved = torch.load(target, map_location="cpu", weights_only=False)
            if saved["provenance"] != expected:
                raise ValueError("Embedding cache provenance mismatch")
            records = saved["records"]
        else:
            model = PairwiseOrderModel(hidden_dim=checkpoint["args"]["hidden_dim"])
            model.load_state_dict(checkpoint["model"])
            model.eval().requires_grad_(False)
            records = {}
            for index, page in enumerate(fold_pages):
                lines = page["lines"]
                env = OCRCorrectionEnv({"ocr_lines": lines}, max_lines=checkpoint["args"]["max_lines"])
                observation, _ = env.reset()
                features = torch.from_numpy(observation[:len(lines)]).float()
                edges = spatial_edges(observation[:len(lines)], k=4)
                with torch.inference_mode():
                    nodes = model.encoder(features, edges).numpy().copy()
                    if index == 0:
                        # Use the original scalar-sigmoid inference path;
                        # vectorized sigmoid can differ by a float32 ULP.
                        inference = gat_inference_from_logits(model(features, edges), lines, native_prior=4.0)
                        if any(probability != page["probabilities"][i, j] for (i, j), probability in inference.probabilities.items()):
                            raise ValueError("Reconstructed GAT input no longer matches cached probabilities")
                records[page["key"]] = {"node": nodes, "pool": nodes.mean(axis=0), "line_ids": [line["id"] for line in lines]}
                if (index + 1) % 200 == 0:
                    print(f"embedding fold={fold} cached={index+1}/{len(fold_pages)}", flush=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"provenance": expected, "records": records}, target)
        for page in fold_pages:
            record = records[page["key"]]
            if record["line_ids"] != [line["id"] for line in page["lines"]]:
                raise ValueError("Embedding-to-OCR line alignment changed")
            page["node_embedding_raw"] = record["node"]
            page["pooled_embedding_raw"] = record["pool"]
        metadata[str(fold)] = {**expected, "pages": len(fold_pages), "path": target.as_posix()}
        print(f"embedding fold={fold} ready={len(fold_pages)} frozen=True", flush=True)
    return metadata


def normalize_embeddings(splits):
    nodes = np.concatenate([page["node_embedding_raw"] for page in splits["train"]])
    pools = np.stack([page["pooled_embedding_raw"] for page in splits["train"]])
    lookahead = np.stack([page["lookahead"] for page in splits["train"]])
    normalization = {}
    for name, values in (("node", nodes), ("pool", pools), ("lookahead", lookahead)):
        mean, std = values.mean(axis=0), values.std(axis=0).clip(1e-6)
        normalization[name] = {"mean": mean.tolist(), "std": std.tolist()}
        for group in splits.values():
            for page in group:
                source = page["node_embedding_raw"] if name == "node" else page["pooled_embedding_raw"] if name == "pool" else page["lookahead"]
                destination = "node_embedding" if name == "node" else "pooled_embedding" if name == "pool" else "lookahead_normalized"
                page[destination] = ((source - mean) / std).astype(np.float32)
    for group in splits.values():
        for page in group:
            page["inferred_columns"] = inferred_columns(page)
    return {"fit_split": "PPO-train only", **normalization}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=Path("runs/ppo_gate"))
    parser.add_argument("--cv-dir", type=Path, default=Path("runs/gat_cross_validation"))
    parser.add_argument("--output", type=Path, default=Path("runs/ppo_gat_policy"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    splits, _, provenance = load_experiment(args)
    cache_embeddings(splits, args, provenance)
