from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.env import OCRCorrectionEnv
from rlocr.ppo import GATActorCritic, PPOConfig, PPOTrainer, collect_episode, spatial_edges

parser = argparse.ArgumentParser(); parser.add_argument("--aligned", default="data/synthetic/aligned"); parser.add_argument("--splits", default="data/synthetic/splits.json"); parser.add_argument("--split", default="train"); parser.add_argument("--epochs", type=int, default=2); parser.add_argument("--max-pages", type=int, default=20); parser.add_argument("--max-lines", type=int, default=128); parser.add_argument("--output", default="runs/ppo_policy.pt"); args = parser.parse_args()
split_data = json.loads(Path(args.splits).read_text(encoding="utf-8")); selected = set(split_data.get(args.split, [])); all_paths = sorted(Path(args.aligned).glob("*.json")); paths = [path for path in all_paths if path.stem in selected][:args.max_pages] if selected else all_paths[:args.max_pages]; device = torch.device("cpu"); model = GATActorCritic(); trainer = PPOTrainer(model, PPOConfig(epochs=args.epochs)); rewards = []
for path in paths:
    page = json.loads(path.read_text(encoding="utf-8")); env = OCRCorrectionEnv(page, max_lines=args.max_lines); initial, _ = env.reset(); edge_index = spatial_edges(initial, device); states, nodes, actions, old_log_probs, episode_rewards, total = collect_episode(env, model, edge_index, device); observations = torch.stack(states); trainer.update(observations, torch.tensor(nodes), torch.stack(actions), torch.stack(old_log_probs), torch.tensor(episode_rewards), torch.tensor([False] * (len(episode_rewards) - 1) + [True]), edge_index); rewards.append(total)
Path(args.output).parent.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), args.output); print(f"pages={len(paths)} mean_episode_reward={sum(rewards)/max(1,len(rewards)):.3f} output={args.output}")
