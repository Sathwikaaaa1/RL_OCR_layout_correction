from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.env import OCRCorrectionEnv
from rlocr.ppo import GATActorCritic, PPOConfig, PPOTrainer, collect_episode, ring_edges

parser = argparse.ArgumentParser(); parser.add_argument("--aligned", default="data/synthetic/aligned"); parser.add_argument("--epochs", type=int, default=2); parser.add_argument("--max-pages", type=int, default=20); parser.add_argument("--max-lines", type=int, default=128); parser.add_argument("--output", default="runs/ppo_policy.pt"); args = parser.parse_args()
paths = sorted(Path(args.aligned).glob("*.json"))[:args.max_pages]; device = torch.device("cpu"); model = GATActorCritic(); trainer = PPOTrainer(model, PPOConfig(epochs=args.epochs)); edge_index = ring_edges(args.max_lines, device); rewards = []
for path in paths:
    page = json.loads(path.read_text(encoding="utf-8")); env = OCRCorrectionEnv(page, max_lines=args.max_lines); states, nodes, actions, old_log_probs, episode_rewards, total = collect_episode(env, model, edge_index, device); observations = torch.stack(states); trainer.update(observations, torch.tensor(nodes), torch.stack(actions), torch.stack(old_log_probs), torch.tensor(episode_rewards), torch.tensor([False] * (len(episode_rewards) - 1) + [True]), edge_index); rewards.append(total)
Path(args.output).parent.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), args.output); print(f"pages={len(paths)} mean_episode_reward={sum(rewards)/max(1,len(rewards)):.3f} output={args.output}")
