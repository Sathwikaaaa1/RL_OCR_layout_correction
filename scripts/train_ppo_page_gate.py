"""Cheap batched PPO contextual bandit over native versus fixed gated pages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.common import write_json
from rlocr.gated_decoder import paired_significance
from rlocr.ppo import PPOConfig
from rlocr.ppo_gate import OverrideActorCritic, index_tau, ppo_update
from rlocr.ppo_page_gate import PAGE_FEATURE_NAMES, page_gate_features
from train_ppo_gate import across_seed_summary
from train_ppo_gate_v2 import load_experiment


def page_arrays(pages, threshold):
    return {
        "features": np.stack([page_gate_features(page, threshold) for page in pages]),
        "native_tau": np.asarray([page["native_tau"] for page in pages]),
        "gated_tau": np.asarray([index_tau(page["gated"], page["gt_rank"]) for page in pages]),
        "different_orders": np.asarray([page["gated"] != page["native"] for page in pages]),
    }


def actions_for(model, features):
    with torch.no_grad():
        return model(features)[0].argmax(-1).numpy()


def validation_score(actions, arrays):
    return float(np.where(actions, arrays["gated_tau"], arrays["native_tau"]).mean())


def train_seed(seed, arrays, features, args, benchmark=False):
    torch.manual_seed(seed)
    model = OverrideActorCritic(len(PAGE_FEATURE_NAMES))
    config = PPOConfig(epochs=4, clip_ratio=0.2, entropy_coef=0.01, learning_rate=3e-4, gamma=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    rewards_by_page = torch.tensor(arrays["train"]["gated_tau"] - arrays["train"]["native_tau"], dtype=torch.float32)
    # This named constant-action fallback is evaluated only on validation.
    best_tau = float(arrays["val"]["gated_tau"].mean())
    selected = {"kind": "always_gated", "steps": 0, "validation_tau": best_tau}
    selected_state = None
    history = []
    steps, next_evaluation = 0, 0
    limit = 4096 if benchmark else args.steps
    started = time.perf_counter()
    rewards = []
    while True:
        if not benchmark and (steps >= next_evaluation or steps >= limit):
            validation_actions = actions_for(model, features["val"])
            tau = validation_score(validation_actions, arrays["val"])
            # Prefer the named fallback on ties. Random initialization is
            # logged, but is not an eligible learned checkpoint.
            if steps > 0 and tau > best_tau + 1e-12:
                best_tau = tau
                selected_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                selected = {"kind": "ppo", "steps": steps, "validation_tau": tau}
            history.append({"steps": steps, "validation_tau": tau,
                "validation_action1_pages": int(validation_actions.sum()),
                "mean_train_reward": float(np.mean(rewards)) if rewards else None,
                "seconds": time.perf_counter() - started})
            rewards = []
            write_json(args.output / f"seed_{seed}_history.json", history)
            print(f"seed={seed} steps={steps} val_tau={tau:.6f} selected={selected['kind']} seconds={time.perf_counter()-started:.1f}", flush=True)
            next_evaluation = steps + args.eval_every
        if steps >= limit:
            break
        indices = torch.randint(len(features["train"]), (1024,), generator=generator)
        observations = features["train"][indices]
        with torch.no_grad():
            logits, values = model(observations)
            distribution = Categorical(logits=logits)
            actions = distribution.sample()
            old_log_probs = distribution.log_prob(actions)
            # Every sample is a separate terminal one-decision episode.
            reward = rewards_by_page[indices] * actions.float()
            advantages = reward - values
            returns = reward
        ppo_update(model, optimizer, observations, actions, old_log_probs, advantages, returns, config, generator)
        rewards.append(float(reward.mean()))
        steps += len(indices)
    if benchmark:
        return steps / (time.perf_counter() - started)
    checkpoint = {**selected, "model": selected_state, "seed": seed, "config": vars(config),
                  "observation_names": PAGE_FEATURE_NAMES, "normalization": args.normalization}
    torch.save(checkpoint, args.output / f"seed_{seed}.pt")
    if selected_state is not None:
        model.load_state_dict(selected_state)
    return model, selected, history


def metric_rows(pages, actions):
    rows = []
    for page, action in zip(pages, actions):
        order = page["gated"] if action else page["native"]
        tau = index_tau(order, page["gt_rank"])
        delta = tau - page["native_tau"]
        rows.append({"key": page["key"], "template": page["template"], "action": int(action),
                     "tau": tau, "native_tau": page["native_tau"], "native_correct": page["native_tau"] >= 0.999999,
                     "changed": order != page["native"], "difference": delta,
                     "retained_native_instead_of_gated": not action and page["gated"] != page["native"]})
    return rows


def summarize(rows):
    mean = lambda group: float(np.mean([row["tau"] for row in group])) if group else None
    return {"pages": len(rows), "mean_tau": mean(rows),
            "native_correct_tau": mean([row for row in rows if row["native_correct"]]),
            "native_wrong_tau": mean([row for row in rows if not row["native_correct"]]),
            "pages_switched": sum(row["changed"] for row in rows),
            "action1_pages": sum(row["action"] for row in rows),
            "retained_native_instead_of_gated": sum(row["retained_native_instead_of_gated"] for row in rows),
            "better": sum(row["difference"] > 1e-9 for row in rows),
            "worse": sum(row["difference"] < -1e-9 for row in rows),
            "same": sum(abs(row["difference"]) <= 1e-9 for row in rows),
            "per_template_tau": {name: mean([row for row in rows if row["template"] == name]) for name in sorted({row["template"] for row in rows})}}


def write_report(report, output):
    lines = ["# Page-level PPO gate", "", report["outcome"], "", report["methodology"], "",
             "## Mean +/- population std across seeds", "", "| Method | Mean tau | Native-correct | Native-wrong | Pages switched | Better | Worse | Same |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda value: f"{value['mean']:.6f} +/- {value['std']:.6f}"
    for name, values in report["across_seeds"].items():
        lines.append("| " + name + " | " + " | ".join(fmt(values[key]) for key in ("mean_tau", "native_correct_tau", "native_wrong_tau", "pages_switched", "better", "worse", "same")) + " |")
    lines += ["", "Pages switched means output order differs from native. Retained-native-instead-of-gated counts output orders that differ from the full gated baseline. Selecting action 1 on an unchanged page does not count as a switched page.", "",
              "## Per-seed test results", "", "| Seed | Method | Tau | Native-correct | Native-wrong | Switched | Retained native vs gated | Better | Worse | Same |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for seed, value in report["seeds"].items():
        for method, stats in value["table"].items():
            lines.append(f"| {seed} | {method} | {stats['mean_tau']:.6f} | {stats['native_correct_tau']:.6f} | {stats['native_wrong_tau']:.6f} | {stats['pages_switched']} | {stats['retained_native_instead_of_gated']} | {stats['better']} | {stats['worse']} | {stats['same']} |")
    lines += ["", "## Checkpoint selection", "", "| Seed | Selected | Steps | Validation tau |", "|---|---|---:|---:|"]
    for seed, choice in report["selection"].items():
        lines.append(f"| {seed} | {choice['kind']} | {choice['steps']} | {choice['validation_tau']:.6f} |")
    lines += ["", "## Paired inference", "", "10,000 paired page bootstrap resamples and sign flips, fixed seed 314159. Seed-mean comparisons average tau within each test page before resampling; repeated seeds are not treated as extra independent pages.", "",
              "| Method / seed | Versus | Mean difference | 95% CI | Sign-flip p |", "|---|---|---:|---|---:|"]
    for name, comparisons in report["significance"].items():
        for baseline, value in comparisons.items():
            difference = value["difference"]
            lines.append(f"| {name} | {baseline} | {difference['mean']:+.6f} | [{difference['low']:+.6f}, {difference['high']:+.6f}] | {value['permutation_p_value']:.6f} |")
    lines += ["", "## Features and protocol", "", report["feature_definitions"], "",
              "The same frozen OOF experiment is reused. Each page's GAT prediction is held out of its own GAT fit, but original GAT training folds overlap; this secondary PPO split is not a fully nested independent end-to-end evaluation. Intervals condition on fitted models.", "",
              "Per-template metrics, complete per-page results, histories, source hashes, and normalizer statistics are in report.json.", ""]
    write_json(output / "report.json", report)
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=Path("runs/ppo_gate"))
    parser.add_argument("--output", type=Path, default=Path("runs/ppo_page_gate"))
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337, 4242, 90210])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    splits, thresholds, provenance = load_experiment(args)
    arrays = {name: page_arrays(pages, thresholds["pair_threshold"]) for name, pages in splits.items()}
    mean = arrays["train"]["features"].mean(axis=0)
    std = arrays["train"]["features"].std(axis=0).clip(1e-6)
    features = {name: torch.from_numpy(((value["features"] - mean) / std).astype(np.float32)) for name, value in arrays.items()}
    args.normalization = {"mean": mean.tolist(), "std": std.tolist(), "fit_split": "train"}
    speed = train_seed(args.seeds[0], arrays, features, args, benchmark=True)
    estimate = len(args.seeds) * args.steps / speed
    print(f"CPU training estimate before full runs: {estimate:.1f} seconds for {len(args.seeds)} seeds x {args.steps} page episodes ({speed:.0f} episodes/s), plus paired reporting. No GAT fitting or inference.", flush=True)
    trained, selection, histories = {}, {}, {}
    for seed in args.seeds:
        trained[seed], selection[str(seed)], histories[str(seed)] = train_seed(seed, arrays, features, args)
    # All fitting and selection finishes before any test-policy evaluation.
    test = arrays["test"]
    baseline_actions = {"native": np.zeros(len(splits["test"]), dtype=np.int64),
                        "gated_gat": np.ones(len(splits["test"]), dtype=np.int64),
                        "best_of_page_oracle": (test["gated_tau"] > test["native_tau"]).astype(np.int64)}
    baselines = {name: metric_rows(splits["test"], actions) for name, actions in baseline_actions.items()}
    seeds, significance = {}, {}
    native_values = test["native_tau"].tolist()
    gated_values = test["gated_tau"].tolist()
    for seed in args.seeds:
        actions = np.ones(len(splits["test"]), dtype=np.int64) if selection[str(seed)]["kind"] == "always_gated" else actions_for(trained[seed], features["test"])
        rows = {**baselines, "page_level_ppo": metric_rows(splits["test"], actions)}
        seeds[str(seed)] = {"table": {name: summarize(values) for name, values in rows.items()}, "rows": rows}
        for method, values in rows.items():
            tau = [row["tau"] for row in values]
            significance[f"{method}/{seed}"] = {"native": paired_significance(tau, native_values, 10_000, 314159),
                                                "gated_gat": paired_significance(tau, gated_values, 10_000, 314159)}
    across = {}
    for method in (*baselines, "page_level_ppo"):
        across[method] = across_seed_summary([seeds[str(seed)]["table"][method] for seed in args.seeds])
        per_page = np.mean([[row["tau"] for row in seeds[str(seed)]["rows"][method]] for seed in args.seeds], axis=0).tolist()
        significance[f"{method}/mean_seed"] = {"native": paired_significance(per_page, native_values, 10_000, 314159),
                                               "gated_gat": paired_significance(per_page, gated_values, 10_000, 314159)}
    improved = significance["page_level_ppo/mean_seed"]["gated_gat"]["difference"]["low"] > 0
    report = {"outcome": "page-level PPO improved" if improved else "No significant page-level PPO improvement over gated GAT", "improved": improved,
              "improvement_rule": "Print the improvement label only when the mean-seed paired bootstrap 95% CI versus gated GAT has lower bound > 0.",
              "methodology": "One decision per page: action 0 returns native, action 1 returns the full fixed gated decoder order. Reward is chosen tau minus native tau. Cached held-out-fold pair probabilities, exact original 1610/345/345 PPO page splits, and saved validation-tuned gate thresholds are reused. Three seeds train a two-layer 64-unit MLP with clipped PPO and normalized advantages; independent terminal one-step episodes need no temporal bootstrapping. Observations are standardized using PPO-train statistics only. Checkpoint selection uses PPO-validation mean tau only. Always-action-1 is the explicit fallback; ties retain the fallback or earlier best checkpoint. Random initialization is logged but not selected. All seeds finish selection before final PPO-test scoring.",
              "feature_definitions": "Six features: log1p(confident native/GAT disagreement count)/5; maximum opposite-direction probability minus 0.5 among confident disagreements (0 when none); gated/native order tau; native/XY-cut tau; log1p(OCR-line count)/5; fraction of narrow edge lines. Narrow means width <=25% of page width, edge means center <=20% or >=80% of width. These geometry constants are fixed before training. Ordering units are existing OCR lines. No ground-truth or template labels enter observations.",
              "feature_names": PAGE_FEATURE_NAMES, "normalization": args.normalization, "provenance": provenance,
              "thresholds": thresholds, "selection": selection, "seeds": seeds, "across_seeds": across,
              "significance": significance, "histories": histories, "seed_values": args.seeds,
              "bootstrap_samples": 10_000, "bootstrap_seed": 314159,
              "timing": {"benchmark_episodes_per_second": speed, "estimated_training_seconds": estimate, "total_seconds": time.perf_counter()-started}}
    write_report(report, args.output)
    print(report["outcome"], flush=True)
    print(json.dumps({"means": {name: value["mean_tau"] for name, value in across.items()},
                      "native_correct": across["page_level_ppo"]["native_correct_tau"],
                      "selection": selection, "versus_gated": {key: value for key, value in significance["page_level_ppo/mean_seed"]["gated_gat"].items() if key != "differences"},
                      "seconds": report["timing"]["total_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
