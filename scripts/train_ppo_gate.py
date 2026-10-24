"""Train a small binary PPO gate on frozen, held-out-fold GAT probabilities."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.common import write_json
from rlocr.evaluation import xy_cut_order
from rlocr.gat_data import load_gat_record
from rlocr.gat_inference import infer_gat
from rlocr.gated_decoder import gated_order, paired_significance
from rlocr.model import PairwiseOrderModel
from rlocr.order_actions import ACTION_NAMES
from rlocr.order_features import pair_probability_scores
from rlocr.ppo import PPOConfig
from rlocr.ppo_gate import OBSERVATION_NAMES, OverrideActorCritic, OverrideNativeEnv, gae, index_tau, ppo_update


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probability_dict(page):
    matrix = page["probabilities"]
    return {(i, j): float(matrix[i, j]) for i in range(len(matrix)) for j in range(i + 1, len(matrix))}


def load_pages(args) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cv = args.cv_dir
    provenance = {"cv_report": digest(cv / "report.json"), "checkpoints": {str(i): digest(cv / f"fold_{i}.pt") for i in range(5)}, "native_prior": 4.0, "cache_version": 1}
    cache = args.output_dir / "oof_probabilities.pt"
    if cache.exists():
        saved = torch.load(cache, map_location="cpu", weights_only=False)
        if saved["provenance"] != provenance:
            raise ValueError("OOF cache provenance changed; choose a new output directory")
        return saved["pages"], provenance
    rows = json.loads((cv / "report.json").read_text(encoding="utf-8"))["rows"]
    pages = []
    roots = {"hard800": args.hard_root, "sidebar1500": args.sidebar_root}
    for fold in range(5):
        checkpoint = torch.load(cv / f"fold_{fold}.pt", map_location="cpu", weights_only=False)
        model = PairwiseOrderModel(hidden_dim=checkpoint["args"]["hidden_dim"])
        model.load_state_dict(checkpoint["model"])
        model.eval()
        for row in rows:
            if row["fold"] != fold:
                continue
            key = row["key"]
            if key not in checkpoint["test_keys"] or key in checkpoint["train_keys"] or key in checkpoint["validation_keys"]:
                raise ValueError(f"Page not held out of GAT fitting and selection: {key}")
            root = roots[key.split("/")[0]]
            record = load_gat_record(root / "aligned" / f"{row['page_id']}.json", root / "gt" / f"{row['page_id']}.json", checkpoint["args"]["max_lines"], key)
            lines = record["lines"]
            inference = infer_gat(model, record["features"], record["edges"], lines, native_prior=4.0)
            ids = {line["id"]: i for i, line in enumerate(lines)}
            gt_order = [ids[x] for x in record["page"]["gt_order"] if x in ids]
            gt_rank = np.argsort(gt_order)
            native = [ids[x] for x in record["native_order"]]
            raw = [ids[x] for x in inference.order]
            matrix = np.full((len(lines), len(lines)), 0.5, dtype=np.float64)
            for (i, j), probability in inference.probabilities.items():
                matrix[i, j], matrix[j, i] = probability, 1.0 - probability
            scores = pair_probability_scores(lines, inference.probabilities, native_prior=4.0)
            native_tau = index_tau(native, gt_rank)
            if not np.isclose(native_tau, row["native"]["tau"], atol=1e-12):
                raise ValueError(f"Native reference mismatch: {key}")
            gt_page = json.loads((root / "gt" / f"{row['page_id']}.json").read_text(encoding="utf-8"))
            xy = [ids[x] for x in xy_cut_order(lines)]
            pages.append({"key": key, "fold": fold, "template": row["template"], "lines": lines,
                          "native": native, "raw": raw, "native_rank": np.argsort(native), "gt_rank": gt_rank,
                          "native_tau": native_tau, "probabilities": matrix, "scores": scores,
                          "xy_tau": index_tau(native, np.argsort(xy)),
                          "widths": np.asarray([(line["bbox"][2] - line["bbox"][0]) / gt_page["width"] for line in lines]),
                          "x_centers": np.asarray([(line["bbox"][0] + line["bbox"][2]) / (2 * gt_page["width"]) for line in lines])})
            if len(pages) % 250 == 0:
                print(f"recovered_oof_probabilities={len(pages)}/{len(rows)}", flush=True)
    if len(pages) != 2300 or len({page["key"] for page in pages}) != 2300:
        raise ValueError("Expected exactly 2300 unique OOF pages")
    torch.save({"pages": pages, "provenance": provenance}, cache)
    return pages, provenance


def split_pages(pages, seed):
    ordered = sorted(pages, key=lambda page: page["key"])
    indices = np.random.default_rng(seed).permutation(len(ordered))
    train_end = int(0.70 * len(ordered))
    val_end = train_end + int(0.15 * len(ordered))
    return {name: [ordered[int(i)] for i in values] for name, values in (
        ("train", indices[:train_end]), ("val", indices[train_end:val_end]), ("test", indices[val_end:]))}


def metric(page, order, overrides=0, reward=None):
    tau = index_tau(order, page["gt_rank"])
    difference = tau - page["native_tau"]
    return {"key": page["key"], "template": page["template"], "tau": tau, "native_tau": page["native_tau"],
            "native_correct": page["native_tau"] >= 0.999999, "changed": list(order) != page["native"],
            "difference": difference, "overrides": overrides, "reward": reward}


def evaluate_policy(model, pages):
    envs = [OverrideNativeEnv(page) for page in pages]
    observations = []
    active = []
    for i, env in enumerate(envs):
        observation, info = env.reset()
        if not info["automatic_terminal"]:
            active.append(i)
            observations.append(observation)
    while active:
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(np.asarray(observations)))
            actions = logits.argmax(dim=1).tolist()
        next_active, next_observations = [], []
        for i, action in zip(active, actions):
            observation, _, terminated, _, _ = envs[i].step(action)
            if not terminated:
                next_active.append(i)
                next_observations.append(observation)
        active, observations = next_active, next_observations
    return [metric(page, env.order, env.overrides, env.episode_return) for page, env in zip(pages, envs)]


def tune_gate(pages):
    # Gate output for a fixed pair threshold is independent of T_page after
    # activation, so compute each page/threshold once rather than nine times.
    best = None
    for pair_threshold in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 1.01):
        candidates = []
        for page in pages:
            order, disagreements, changed = gated_order(page["lines"], probability_dict(page), pair_threshold, -1)
            ids = {line["id"]: i for i, line in enumerate(page["lines"])}
            candidates.append((index_tau([ids[x] for x in order], page["gt_rank"]), disagreements, changed))
        for page_threshold in (0, 1, 2, 3, 5, 8, 10, 15, 20):
            tau = np.mean([value if count > page_threshold else page["native_tau"] for page, (value, count, _) in zip(pages, candidates)])
            changed = sum(is_changed and count > page_threshold for _, count, is_changed in candidates)
            score = (float(tau), -changed, -pair_threshold, -page_threshold)
            if best is None or score > best[0]:
                best = (score, pair_threshold, page_threshold)
    return {"pair_threshold": best[1], "page_threshold": best[2], "validation_tau": best[0][0], "validation_changed_pages": -best[0][1]}


def baseline_rows(pages, thresholds):
    results = {"native": [], "raw_gat": [], "gated_gat": []}
    for page in pages:
        order, _, _ = gated_order(page["lines"], probability_dict(page), thresholds["pair_threshold"], thresholds["page_threshold"])
        ids = {line["id"]: i for i, line in enumerate(page["lines"])}
        for method, prediction in (("native", page["native"]), ("raw_gat", page["raw"]), ("gated_gat", [ids[x] for x in order])):
            results[method].append(metric(page, prediction))
    return results


class Rollout:
    def __init__(self, pages, seed, count=16, env_class=OverrideNativeEnv):
        self.pages = pages
        self.rng = np.random.default_rng(seed)
        self.count = count
        self.env_class = env_class
        self.envs = []
        self.observations = []
        # Pages with no proposal disagreement have no trainable actions.
        self.decision_pages = [page for page in pages if page["native"] != page["raw"]]
        if not self.decision_pages:
            raise ValueError("Training corpus has no decisions")
        for _ in range(count):
            env, observation = self.new_episode()
            self.envs.append(env)
            self.observations.append(observation)

    def new_episode(self):
        page = self.decision_pages[int(self.rng.integers(len(self.decision_pages)))]
        env = self.env_class(page)
        observation, _ = env.reset()
        return env, observation

    def collect(self, model, horizon, config):
        observations, actions, log_probs, values, rewards, dones = [], [], [], [], [], []
        action_masks = []
        completed = []
        for _ in range(horizon):
            state = torch.from_numpy(np.asarray(self.observations))
            with torch.no_grad():
                logits, value = model(state)
                if hasattr(self.envs[0], "action_masks"):
                    mask = torch.from_numpy(np.asarray([env.action_masks() for env in self.envs]))
                    logits = logits.masked_fill(~mask, -1e9)
                    action_masks.append(mask)
                distribution = Categorical(logits=logits)
                action = distribution.sample()
                log_prob = distribution.log_prob(action)
            step_rewards, step_dones = [], []
            for i, action_value in enumerate(action.tolist()):
                observation, reward, terminated, _, info = self.envs[i].step(action_value)
                step_rewards.append(reward)
                step_dones.append(float(terminated))
                if terminated:
                    completed.append((info["episode_return"], index_tau(self.envs[i].order, self.envs[i].page["gt_rank"])))
                    self.envs[i], observation = self.new_episode()
                self.observations[i] = observation
            observations.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor(step_rewards))
            dones.append(torch.tensor(step_dones))
        with torch.no_grad():
            _, bootstrap = model(torch.from_numpy(np.asarray(self.observations)))
        advantages, returns = gae(torch.stack(rewards), torch.stack(dones), torch.stack(values), bootstrap, config)
        self.last_action_masks = torch.cat(action_masks) if action_masks else None
        return (torch.cat(observations), torch.cat(actions), torch.cat(log_probs), advantages.flatten(), returns.flatten()), completed


def train_seed(seed, splits, args, benchmark=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = OverrideActorCritic()
    config = PPOConfig(epochs=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    rollout = Rollout(splits["train"], seed)
    total_steps = args.benchmark_steps if benchmark else args.steps
    best = None
    history = []
    completed = []
    started = time.perf_counter()
    next_evaluation = 0
    steps = 0
    while True:
        if not benchmark and (steps >= next_evaluation or steps >= total_steps):
            validation = evaluate_policy(model, splits["val"])
            tau = float(np.mean([row["tau"] for row in validation]))
            overrides = sum(row["overrides"] for row in validation)
            score = (tau, -overrides)
            if best is None or score > best:
                best = score
                torch.save({"model": model.state_dict(), "seed": seed, "steps": steps, "validation_tau": tau,
                            "validation_overrides": overrides, "observation_names": OBSERVATION_NAMES,
                            "config": vars(config), "native_prior": 4.0}, args.output_dir / f"seed_{seed}.pt")
            history.append({"steps": steps, "validation_tau": tau, "validation_overrides": overrides,
                            "mean_train_reward": float(np.mean([x[0] for x in completed])) if completed else None,
                            "mean_train_tau": float(np.mean([x[1] for x in completed])) if completed else None,
                            "seconds": time.perf_counter() - started})
            completed = []
            write_json(args.output_dir / f"seed_{seed}_history.json", history)
            print(f"seed={seed} steps={steps} val_tau={tau:.6f} val_overrides={overrides} seconds={time.perf_counter()-started:.1f}", flush=True)
            next_evaluation = steps + args.eval_every
        if steps >= total_steps:
            break
        batch, episodes = rollout.collect(model, horizon=128, config=config)
        ppo_update(model, optimizer, *batch, config, generator)
        completed.extend(episodes)
        steps += rollout.count * 128
    elapsed = time.perf_counter() - started
    if benchmark:
        return {"steps": steps, "seconds": elapsed, "steps_per_second": steps / elapsed}
    selected = torch.load(args.output_dir / f"seed_{seed}.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model"])
    return model, history, selected


def summarize(rows):
    correct = [row for row in rows if row["native_correct"]]
    wrong = [row for row in rows if not row["native_correct"]]
    mean = lambda group: float(np.mean([row["tau"] for row in group])) if group else None
    return {"pages": len(rows), "mean_tau": mean(rows), "native_correct_tau": mean(correct), "native_wrong_tau": mean(wrong),
            "per_template_tau": {name: mean([row for row in rows if row["template"] == name]) for name in sorted({row["template"] for row in rows})},
            "pages_changed": sum(row["changed"] for row in rows), "better": sum(row["difference"] > 1e-9 for row in rows),
            "worse": sum(row["difference"] < -1e-9 for row in rows), "same": sum(abs(row["difference"]) <= 1e-9 for row in rows)}


def across_seed_summary(values):
    result = {}
    for field in values[0]:
        if field == "per_template_tau":
            result[field] = {name: {"mean": float(np.mean([x[field][name] for x in values])), "std": float(np.std([x[field][name] for x in values]))} for name in values[0][field]}
        else:
            result[field] = {"mean": float(np.mean([x[field] for x in values])), "std": float(np.std([x[field] for x in values]))}
    return result


def write_reports(report, output):
    report["action_names"] = ACTION_NAMES
    lines = ["# PPO override gate", "", report["acceptance_message"], "", report["methodology"], "",
             "| Seed | Method | Mean tau | Native-correct | Native-wrong | Changed | Better | Worse | Same |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for seed, result in report["seeds"].items():
        for method, summary in result["table"].items():
            lines.append(f"| {seed} | {method} | {summary['mean_tau']:.6f} | {summary['native_correct_tau']:.6f} | {summary['native_wrong_tau']:.6f} | {summary['pages_changed']} | {summary['better']} | {summary['worse']} | {summary['same']} |")
    lines += ["", "## Mean +/- population standard deviation across seeds", "", "| Method | Mean tau | Native-correct tau | Native-wrong tau | Pages changed | Better | Worse | Same |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda value: f"{value['mean']:.6f} +/- {value['std']:.6f}"
    for method, value in report["across_seeds"].items():
        lines.append("| " + method + " | " + " | ".join(fmt(value[field]) for field in ("mean_tau", "native_correct_tau", "native_wrong_tau", "pages_changed", "better", "worse", "same")) + " |")
    lines += ["", "## Per-template tau", "", "| Seed | Method | Template | Tau |", "|---|---|---|---:|"]
    for seed, result in report["seeds"].items():
        for method, value in result["table"].items():
            for name, tau in value["per_template_tau"].items():
                lines.append(f"| {seed} | {method} | {name} | {tau:.6f} |")
    for method, value in report["across_seeds"].items():
        for name, tau in value["per_template_tau"].items():
            lines.append(f"| mean +/- std | {method} | {name} | {fmt(tau)} |")
    lines += ["", "## Paired significance", "", "| Seed / aggregate | Comparison | Mean delta | Bootstrap 95% CI | Sign-flip p |", "|---|---|---:|---|---:|"]
    for seed, comparisons in report["significance"].items():
        for name, value in comparisons.items():
            difference = value["difference"]
            lines.append(f"| {seed} | {name} | {difference['mean']:+.6f} | [{difference['low']:+.6f}, {difference['high']:+.6f}] | {value['permutation_p_value']:.6f} |")
    lines += ["", "## Validation-selected checkpoints", "", "| Seed | Selected decision steps | Validation tau | Validation overrides |", "|---|---:|---:|---:|"]
    for seed, value in report["seeds"].items():
        lines.append(f"| {seed} | {value['selected_steps']} | {value['validation_tau']:.6f} | {value['validation_overrides']} |")
    initial_selected = [seed for seed, value in report["seeds"].items() if value["selected_steps"] == 0]
    if initial_selected:
        lines += ["", "The initial (step-zero) policy was included in validation selection. For seeds " + ", ".join(initial_selected) + ", PPO training did not improve on that initial checkpoint's validation selection score; their reported selected policies must not be interpreted as learned improvements."]
    lines += ["", "## Validation-selected settings", "", "```json", json.dumps(report["thresholds"], indent=2), "```", "",
              "## Curves", "", "![Reward and tau against steps](training_curves.png)", "",
              "Mean-seed significance first averages PPO tau across seeds per test page, then resamples pages. It does not treat 3 x test pages as independent. Intervals condition on fitted models.", "",
              "Always RETAIN has zero terminal tau-improvement reward. Its full return includes dense rewards, as specified. Automatic placements also receive dense rewards and use no policy steps.", ""]
    write_json(output / "report.json", report)
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_curves(histories, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for seed, history in histories.items():
        reward_rows = [row for row in history if row["mean_train_reward"] is not None]
        axes[0].plot([row["steps"] for row in reward_rows], [row["mean_train_reward"] for row in reward_rows], label=str(seed))
        axes[1].plot([row["steps"] for row in history], [row["validation_tau"] for row in history], label=f"{seed} val")
    axes[0].set(xlabel="Policy decision steps", ylabel="Mean completed training episode return")
    axes[1].set(xlabel="Policy decision steps", ylabel="Validation mean Kendall tau")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-dir", type=Path, default=Path("runs/gat_cross_validation"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ppo_gate"))
    parser.add_argument("--hard-root", type=Path, default=Path("data/synthetic_hard_large"))
    parser.add_argument("--sidebar-root", type=Path, default=Path("data/synthetic_sidebar_large"))
    parser.add_argument("--steps", type=int, default=200_000, help="Policy decision steps per seed")
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337, 4242, 90210])
    parser.add_argument("--split-seed", type=int, default=314159)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--benchmark-steps", type=int, default=8192)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--render-only", action="store_true", help="Render existing report JSON without inference or training")
    args = parser.parse_args()
    if args.render_only:
        saved_report = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
        write_reports(saved_report, args.output_dir)
        return
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pages, provenance = load_pages(args)
    splits = split_pages(pages, args.split_seed)
    split_keys = {name: {page["key"] for page in group} for name, group in splits.items()}
    if any(split_keys[left] & split_keys[right] for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("PPO train/validation/test page overlap")
    write_json(args.output_dir / "splits.json", {"seed": args.split_seed, **{key: [page["key"] for page in value] for key, value in splits.items()}})
    benchmark = train_seed(args.seeds[0], splits, args, benchmark=True)
    probe_model = OverrideActorCritic()
    val_started = time.perf_counter()
    evaluate_policy(probe_model, splits["val"])
    validation_seconds = time.perf_counter() - val_started
    estimate = len(args.seeds) * (args.steps / benchmark["steps_per_second"] + (args.steps / args.eval_every + 1) * validation_seconds)
    timing = {"benchmark": benchmark, "validation_seconds": validation_seconds, "estimated_training_seconds": estimate, "cache_and_benchmark_seconds": time.perf_counter() - started}
    write_json(args.output_dir / "timing.json", timing)
    print(f"CPU estimate before full training: {estimate/60:.2f} minutes for {len(args.seeds)} seeds x {args.steps} steps, plus gate tuning/report. Measured {benchmark['steps_per_second']:.0f} steps/s. GAT weights frozen.", flush=True)
    if args.benchmark_only:
        return
    thresholds = tune_gate(splits["val"])
    write_json(args.output_dir / "thresholds.json", thresholds)
    print(f"PPO-val gate thresholds: {thresholds}", flush=True)
    trained = {}
    histories = {}
    selected = {}
    for seed in args.seeds:
        trained[seed], histories[seed], selected[seed] = train_seed(seed, splits, args)
    # First access to PPO-test outcomes is after every seed's selection finishes.
    baselines = baseline_rows(splits["test"], thresholds)
    seed_results, significance, ppo_rows = {}, {}, {}
    native_values = [row["tau"] for row in baselines["native"]]
    gated_values = [row["tau"] for row in baselines["gated_gat"]]
    for seed in args.seeds:
        rows = {**baselines, "ppo_gate": evaluate_policy(trained[seed], splits["test"])}
        ppo_rows[seed] = rows["ppo_gate"]
        seed_results[str(seed)] = {"selected_steps": selected[seed]["steps"], "validation_tau": selected[seed]["validation_tau"],
                                   "validation_overrides": selected[seed]["validation_overrides"],
                                   "table": {method: summarize(value) for method, value in rows.items()}, "rows": rows}
        values = [row["tau"] for row in rows["ppo_gate"]]
        significance[str(seed)] = {"PPO - native": paired_significance(values, native_values, args.bootstrap, args.split_seed),
                                   "PPO - gated": paired_significance(values, gated_values, args.bootstrap, args.split_seed)}
    average_values = np.mean([[row["tau"] for row in ppo_rows[seed]] for seed in args.seeds], axis=0).tolist()
    significance["mean_seed_per_page"] = {"PPO - native": paired_significance(average_values, native_values, args.bootstrap, args.split_seed),
                                         "PPO - gated": paired_significance(average_values, gated_values, args.bootstrap, args.split_seed)}
    across = {method: across_seed_summary([value["table"][method] for value in seed_results.values()]) for method in ("native", "raw_gat", "gated_gat", "ppo_gate")}
    accepted = across["ppo_gate"]["mean_tau"]["mean"] >= across["gated_gat"]["mean_tau"]["mean"] - 0.002 and across["ppo_gate"]["native_correct_tau"]["mean"] >= 0.999
    report = {"acceptance_message": "PPO accepted" if accepted else "PPO rejected", "accepted": accepted,
              "acceptance_rule": "mean-seed PPO tau >= gated tau - 0.002 and mean-seed PPO native-correct tau >= 0.999",
              "methodology": "Frozen GAT; native prior 4.0. Each page's probabilities come only from its saved held-out GAT fold. A fixed page split (seed 314159) gives 1610 PPO-train / 345 PPO-val / 345 PPO-test. Gate thresholds and each PPO checkpoint are selected on PPO-val only. PPO uses a 2x64 MLP, GAE, clipped updates and 200k decision steps per seed (defaults). Existing OCR lines are the ordering units. Scores are canonical logit-Borda page scores; proposals choose the highest-scoring unplaced line. Candidate probabilities are softmax of scores divided by page line count; margin is candidate score minus best other remaining score in that scale. Saved GAT folds were originally trained on overlapping sets: while every page's own GAT prediction is held out, this secondary split is not a fully nested independent end-to-end test, because GAT fits for PPO-train pages can include PPO-test labels. Treat results as conditional on this frozen OOF experiment.",
              "provenance": provenance, "observation_names": OBSERVATION_NAMES, "split_seed": args.split_seed,
              "split_counts": {key: len(value) for key, value in splits.items()}, "thresholds": thresholds,
              "steps_per_seed": args.steps, "seeds": seed_results, "across_seeds": across, "significance": significance,
              "bootstrap_samples": args.bootstrap, "bootstrap_seed": args.split_seed, "histories": histories,
              "timing": {**timing, "total_seconds": time.perf_counter() - started}}
    plot_curves(histories, args.output_dir)
    write_reports(report, args.output_dir)
    print(report["acceptance_message"], flush=True)
    print(json.dumps({"across_seeds": across, "total_seconds": report["timing"]["total_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
