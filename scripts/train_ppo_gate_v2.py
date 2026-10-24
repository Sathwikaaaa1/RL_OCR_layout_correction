"""Final PPO override experiment: frozen cache, diagnostics, imitation, ablation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.common import write_json
from rlocr.gated_decoder import gated_order, paired_significance
from rlocr.ppo import PPOConfig
from rlocr.order_actions import ACTION_NAMES
from rlocr.ppo_gate import OverrideActorCritic, OverrideNativeEnv, index_tau, ppo_update
from rlocr.ppo_gate_v2 import (V2_OBSERVATION_NAMES, OverrideNativeV2Env, fit_imitation,
                               gated_teacher_action, lookahead_features, reachable_tau_oracle)
from train_ppo_gate import Rollout, across_seed_summary, metric, probability_dict, summarize


def load_experiment(args):
    cache_path = args.previous / "oof_probabilities.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    split_path = args.previous / "splits.json"
    split_ids = json.loads(split_path.read_text(encoding="utf-8"))
    thresholds = json.loads((args.previous / "thresholds.json").read_text(encoding="utf-8"))
    pages = cache["pages"]
    by_key = {page["key"]: page for page in pages}
    splits = {name: [by_key[key] for key in split_ids[name]] for name in ("train", "val", "test")}
    flat = [page["key"] for group in splits.values() for page in group]
    if len(flat) != 2300 or len(set(flat)) != 2300 or set(flat) != set(by_key):
        raise ValueError("Original PPO splits must cover 2300 unique pages without overlap")
    for page in pages:
        ids = {line["id"]: i for i, line in enumerate(page["lines"])}
        order, _, _ = gated_order(page["lines"], probability_dict(page), thresholds["pair_threshold"], thresholds["page_threshold"])
        page["gated"] = [ids[key] for key in order]
        page["gated_rank"] = np.argsort(page["gated"])
        page["lookahead"] = lookahead_features(page, thresholds["pair_threshold"])
    provenance = {"cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                  "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
                  "gat_cache_provenance": cache["provenance"]}
    write_json(args.output / "splits.json", split_ids)
    return splits, thresholds, provenance


def placement_overrides(page, order):
    remaining = set(page["native"])
    overrides = 0
    for chosen in order:
        native = next(i for i in page["native"] if i in remaining)
        overrides += int(chosen != native)
        remaining.remove(chosen)
    return overrides


def result_row(page, order, decisions=None):
    row = metric(page, order, overrides=placement_overrides(page, order))
    row.update({"placements": len(order), "decisions": decisions, "order": list(order)})
    return row


def evaluate(model, pages):
    envs = [OverrideNativeV2Env(page) for page in pages]
    active, observations = [], []
    for i, env in enumerate(envs):
        observation, info = env.reset()
        if not info["automatic_terminal"]:
            active.append(i)
            observations.append(observation)
    while active:
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(np.asarray(observations)))
            actions = logits.argmax(-1).tolist()
        following, states = [], []
        for i, action in zip(active, actions):
            observation, _, done, _, _ = envs[i].step(action)
            if not done:
                following.append(i)
                states.append(observation)
        active, observations = following, states
    return [result_row(page, env.order, env.disagreements) for page, env in zip(pages, envs)]


def trajectory(page, chooser, events=None, path="", env_class=OverrideNativeV2Env):
    env = env_class(page)
    observation, info = env.reset()
    done = info["automatic_terminal"]
    while not done:
        native, gat = env._proposals()
        true_next = int(np.argmin(np.where(env.remaining, page["gt_rank"], np.inf)))
        category = "gat_correct" if gat == true_next else "native_correct" if native == true_next else "neither"
        if events is not None:
            events.append({"path": path, "key": page["key"], "template": page["template"],
                           "native_status": "native_correct" if page["native_tau"] >= 0.999999 else "native_wrong",
                           "native_candidate": native, "gat_candidate": gat, "true_next": true_next,
                           "category": category})
        action = chooser(env, observation, category)
        observation, _, done, _, _ = env.step(int(action))
    return result_row(page, env.order, env.disagreements)


def event_fractions(events):
    categories = ("gat_correct", "native_correct", "neither")
    groups = {"all": events}
    for field in ("template", "native_status"):
        for value in sorted({event[field] for event in events}):
            groups[f"{field}/{value}"] = [event for event in events if event[field] == value]
    for template in sorted({event["template"] for event in events}):
        for status in ("native_correct", "native_wrong"):
            groups[f"template_status/{template}/{status}"] = [event for event in events if event["template"] == template and event["native_status"] == status]
    return {name: {"decisions": len(group), **{category: sum(event["category"] == category for event in group) / len(group) if group else None for category in categories}} for name, group in groups.items()}


def override_summary(rows):
    overrides = sum(row["overrides"] for row in rows)
    placements = sum(row["placements"] for row in rows)
    decisions = sum(row["decisions"] for row in rows) if all(row["decisions"] is not None for row in rows) else None
    return {"overrides": overrides, "placements": placements, "decisions": decisions,
            "fraction_of_placements": overrides / placements if placements else 0.0,
            "fraction_of_decisions": overrides / decisions if decisions else None}


def diagnostics(splits, args):
    pages = splits["test"]
    all_events = []
    oracle_rows, teacher_rows = [], []
    exact_rows = []
    for page in pages:
        trajectory(page, lambda *_: 0, all_events, "native_reference")
        oracle_rows.append(trajectory(page, lambda _e, _o, category: int(category == "gat_correct"), all_events, "greedy_oracle"))
        teacher_rows.append(trajectory(page, lambda env, *_: gated_teacher_action(env), all_events, "projected_gated_teacher"))
        exact = reachable_tau_oracle(page)
        exact_rows.append(result_row(page, exact["order"]))
    previous = {}
    old_report = json.loads((args.previous / "report.json").read_text(encoding="utf-8"))
    for seed in args.seeds:
        saved = torch.load(args.previous / f"seed_{seed}.pt", map_location="cpu", weights_only=False)
        model = OverrideActorCritic()
        model.load_state_dict(saved["model"])
        model.eval()
        def choose(_env, observation, _category):
            with torch.no_grad():
                return int(model(torch.from_numpy(observation).unsqueeze(0))[0].argmax())
        rows = [trajectory(page, choose, all_events, f"previous_ppo_{seed}", env_class=OverrideNativeEnv) for page in pages]
        saved_rows = {row["key"]: row for row in old_report["seeds"][str(seed)]["rows"]["ppo_gate"]}
        if any(abs(row["tau"] - saved_rows[row["key"]]["tau"]) > 1e-12 or row["overrides"] != saved_rows[row["key"]]["overrides"] for row in rows):
            raise ValueError("Previous PPO policy no longer reproduces saved results")
        previous[str(seed)] = override_summary(rows)
    gated_rows = [result_row(page, page["gated"]) for page in pages]
    report = {
        "reference_path": "native_reference", "paths": {path: event_fractions([event for event in all_events if event["path"] == path]) for path in sorted({event["path"] for event in all_events})},
        "greedy_oracle": summary(oracle_rows), "exact_reachable_oracle": summary(exact_rows),
        "gated": summary(gated_rows), "projected_gated_teacher": summary(teacher_rows),
        "gated_pages_not_reproduced_by_projection": sum(row["order"] != page["gated"] for row, page in zip(teacher_rows, pages)),
        "override_rates": {"gated_decoder": override_summary(gated_rows), "projected_gated_teacher": override_summary(teacher_rows), **{f"previous_ppo_{seed}": value for seed, value in previous.items()}},
        "events": all_events,
    }
    write_json(args.output / "diagnostics.json", report)
    lines = ["# PPO v2 proposal diagnostics", "", "No training. Exact v1 PPO-test pages and cached held-out-fold probabilities, prior 4.0. Proposal fractions depend on the decoding trajectory. The primary table follows native throughout; other paths are reported separately, including cross-tabs by template and native status.", ""]
    for path, groups in report["paths"].items():
        lines += [f"## Path: {path}", "", "| Group | Decisions | GAT true-next fraction | Native true-next fraction | Neither fraction |", "|---|---:|---:|---:|---:|"]
        for name, value in groups.items():
            fmt = lambda field: f"{value[field]:.6f}" if value[field] is not None else "-"
            lines.append(f"| {name} | {value['decisions']} | {fmt('gat_correct')} | {fmt('native_correct')} | {fmt('neither')} |")
    lines += ["", "## Oracles and action-space limitations", "", f"Greedy per-decision oracle tau: {report['greedy_oracle']['mean_tau']:.6f}.", f"Exact binary-action reachable tau ceiling (dynamic programming): {report['exact_reachable_oracle']['mean_tau']:.6f}.", f"Full gated decoder tau: {report['gated']['mean_tau']:.6f}; projected gated teacher tau: {report['projected_gated_teacher']['mean_tau']:.6f}.", f"Projection differs from the full gated order on {report['gated_pages_not_reproduced_by_projection']} / {len(pages)} test pages.", "", "The greedy oracle chooses the true-next proposal if available, otherwise native, as requested. It need not maximize final tau, so the exact reachable oracle is the actual ceiling. The gated decoder can emit a third permutation outside the binary action space. Teacher labels choose whichever available proposal occurs earlier in its full gated order; this reproduces that order exactly whenever it is reachable.", "", "## Override fractions", "", "| Method | Overrides | Placements | Fraction of placements | Decision steps | Fraction of decision steps |", "|---|---:|---:|---:|---:|---:|"]
    for name, value in report["override_rates"].items():
        lines.append(f"| {name} | {value['overrides']} | {value['placements']} | {value['fraction_of_placements']:.6f} | {value['decisions']} | {value['fraction_of_decisions']} |")
    lines += ["", "An override is placing a line other than native's earliest remaining line. Fraction of placements uses all placed lines, which is comparable for arbitrary full gated orders and binary policies. Fraction of decision steps applies only to binary trajectories; full gated orders do not always have such a trajectory. Test diagnostics are descriptive and are not used for training, threshold selection, or checkpoint selection.", ""]
    (args.output / "diagnostics.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"reference_proposal_fractions": report["paths"]["native_reference"]["all"], "greedy_oracle_tau": report["greedy_oracle"]["mean_tau"], "exact_reachable_ceiling": report["exact_reachable_oracle"]["mean_tau"], "projection_mismatched_pages": report["gated_pages_not_reproduced_by_projection"], "override_rates": report["override_rates"]}, indent=2), flush=True)
    return oracle_rows, report


def teacher_dataset(pages, output):
    observations, actions, keys, decision_log = [], [], [], []
    for page in pages:
        env = OverrideNativeV2Env(page)
        observation, info = env.reset()
        done = info["automatic_terminal"]
        while not done:
            a, b = env._proposals()
            action = gated_teacher_action(env)
            observations.append(observation)
            actions.append(action)
            keys.append(page["key"])
            decision_log.append({"key": page["key"], "native_candidate": a, "gat_candidate": b, "action": action})
            observation, _, done, _, _ = env.step(action)
    array = np.asarray(observations)
    labels = np.asarray(actions, dtype=np.int64)
    np.savez_compressed(output / "imitation_choices.npz", observations=array, actions=labels, page_keys=np.asarray(keys))
    write_json(output / "imitation_choices.json", decision_log)
    return torch.from_numpy(array), torch.from_numpy(labels)


def summary(rows):
    return {**summarize(rows), **override_summary(rows)}


def training(seed, mode, initial, splits, args, benchmark=False):
    torch.manual_seed(seed)
    model = OverrideActorCritic(len(V2_OBSERVATION_NAMES))
    if initial is not None:
        model.load_state_dict(initial)
    config = PPOConfig(epochs=4, clip_ratio=0.1, entropy_coef=0.01, learning_rate=5e-5, gamma=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    rollout = Rollout(splits["train"], seed, env_class=OverrideNativeV2Env)
    limit = 4096 if benchmark else args.steps
    steps, next_evaluation = 0, 0
    history, completed = [], []
    best = None
    path = args.output / f"{mode}_seed_{seed}.pt"
    started = time.perf_counter()
    while True:
        if not benchmark and (steps >= next_evaluation or steps >= limit):
            validation = evaluate(model, splits["val"])
            tau = float(np.mean([row["tau"] for row in validation]))
            overrides = sum(row["overrides"] for row in validation)
            score = (tau, -overrides)
            # The imitation initialization is an allowed candidate. Scratch
            # selection uses trained checkpoints only; step zero is diagnostic.
            if (steps > 0 or initial is not None) and (best is None or score > best):
                best = score
                torch.save({"model": model.state_dict(), "seed": seed, "mode": mode, "steps": steps,
                            "selected_kind": "imitation" if steps == 0 else "ppo",
                            "validation_tau": tau, "validation_overrides": overrides,
                            "config": vars(config), "observation_names": V2_OBSERVATION_NAMES}, path)
            history.append({"steps": steps, "validation_tau": tau, "validation_overrides": overrides,
                            "reward": float(np.mean([value[0] for value in completed])) if completed else None,
                            "train_tau": float(np.mean([value[1] for value in completed])) if completed else None,
                            "seconds": time.perf_counter() - started})
            completed = []
            next_evaluation = steps + args.eval_every
            write_json(args.output / f"{mode}_seed_{seed}_history.json", history)
            print(f"{mode} seed={seed} steps={steps} val_tau={tau:.6f} overrides={overrides} seconds={time.perf_counter()-started:.1f}", flush=True)
        if steps >= limit:
            break
        batch, episodes = rollout.collect(model, 128, config)
        ppo_update(model, optimizer, *batch, config, generator)
        completed.extend(episodes)
        steps += 2048
    if benchmark:
        return steps / (time.perf_counter() - started)
    selected = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model"])
    return model, history, {key: value for key, value in selected.items() if key != "model"}


def write_report(report, output):
    report["action_names"] = ACTION_NAMES
    cutoff = report["across_seeds"]["gated_gat"]["mean_tau"]["mean"] - 0.002
    ceiling = report["exact_reachable_ceiling"]["mean_tau"]
    report["action_space_limit"] = {"exact_reachable_ceiling": ceiling, "acceptance_tau_cutoff": cutoff,
                                    "acceptance_tau_requirement_impossible": ceiling < cutoff,
                                    "shortfall_to_acceptance_cutoff": cutoff - ceiling}
    for seed, branches in report["selection"].items():
        selected = branches["fine_tuned"]
        initial_tau = report["histories"][f"fine_tuned/{seed}"][0]["validation_tau"]
        selected["imitation_validation_tau"] = initial_tau
        selected["validation_tau_gain_over_imitation"] = selected["validation_tau"] - initial_tau
        selected["selection_description"] = ("imitation retained" if selected["steps"] == 0 else
            "PPO improved validation tau" if selected["validation_tau"] > initial_tau else "PPO tied tau with fewer overrides")
    lines = ["# Final PPO override experiment", "", report["acceptance_message"], "", report["acceptance_rule"], "", report["methodology"], "",
             "## Exact action-space ceiling", "", f"The exact best reachable test tau is {ceiling:.9f}; the acceptance mean-tau cutoff is {cutoff:.9f}."]
    if ceiling < cutoff:
        lines += [f"The binary native-or-GAT proposals cannot meet the mean-tau acceptance requirement, even with perfect ground-truth decisions: the exact ceiling falls short by {cutoff-ceiling:.9f}. This is an action-space limit, not evidence that more PPO training would close the gap."]
    lines += ["", "## Aggregate across seeds", "", "| Method | Mean tau +/- std | Native-correct | Native-wrong | Overrides | Better | Worse | Same |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda value: f"{value['mean']:.6f} +/- {value['std']:.6f}"
    for method, value in report["across_seeds"].items():
        lines.append("| " + method + " | " + " | ".join(fmt(value[field]) for field in ("mean_tau", "native_correct_tau", "native_wrong_tau", "overrides", "better", "worse", "same")) + " |")
    lines += ["", "## Per-seed results", "", "| Seed | Method | Tau | Native-correct | Native-wrong | Overrides | Changed pages | Better | Worse | Same |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for seed, methods in report["seeds"].items():
        for method, value in methods["table"].items():
            lines.append(f"| {seed} | {method} | {value['mean_tau']:.6f} | {value['native_correct_tau']:.6f} | {value['native_wrong_tau']:.6f} | {value['overrides']} | {value['pages_changed']} | {value['better']} | {value['worse']} | {value['same']} |")
    lines += ["", "## Selected checkpoints", "", "| Seed | Branch | Selection | Steps | Validation tau | Validation overrides |", "|---|---|---|---:|---:|---:|"]
    for seed, value in report["selection"].items():
        for mode, selected in value.items():
            lines.append(f"| {seed} | {mode} | {selected.get('selection_description', selected['selected_kind'])} | {selected['steps']} | {selected['validation_tau']:.6f} | {selected['validation_overrides']} |")
    lines += ["", "## Paired significance", "", "Seed-mean results average tau per page across seeds before resampling pages; seed replications are not treated as independent pages. Intervals condition on fitted models.", "", "| Method / seed | Comparison | Mean delta | Bootstrap 95% CI | Sign-flip p |", "|---|---|---:|---|---:|"]
    for name, comparisons in report["significance"].items():
        for comparison, value in comparisons.items():
            diff = value["difference"]
            lines.append(f"| {name} | {comparison} | {diff['mean']:+.6f} | [{diff['low']:+.6f}, {diff['high']:+.6f}] | {value['permutation_p_value']:.6f} |")
    lines += ["", "## Per-template tau", "", "| Method | Template | Mean +/- std |", "|---|---|---:|"]
    for method, value in report["across_seeds"].items():
        for template, tau in value["per_template_tau"].items():
            lines.append(f"| {method} | {template} | {fmt(tau)} |")
    lines += ["", "Per-seed template metrics and complete test predictions are included in report.json.", "", "## Curves", "", "![Reward and validation tau](training_curves.png)", "", "[Proposal diagnostics](diagnostics.md)", ""]
    write_json(output / "report.json", report)
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def curves(histories, output, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if title:
        fig.suptitle(title)
    for name, history in histories.items():
        rewards = [row for row in history if row["reward"] is not None]
        axes[0].plot([row["steps"] for row in rewards], [row["reward"] for row in rewards], label=name)
        axes[1].plot([row["steps"] for row in history], [row["validation_tau"] for row in history], label=name)
    axes[0].set(xlabel="PPO decision steps", ylabel="Mean training episode return")
    axes[1].set(xlabel="PPO decision steps", ylabel="Validation mean tau")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=Path("runs/ppo_gate"))
    parser.add_argument("--output", type=Path, default=Path("runs/ppo_gate_v2"))
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337, 4242, 90210])
    parser.add_argument("--imitation-epochs", type=int, default=30)
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--render-only", action="store_true", help="Refresh Markdown from saved results without training or selection")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        write_report(json.loads((args.output / "report.json").read_text(encoding="utf-8")), args.output)
        return
    started = time.perf_counter()
    splits, thresholds, provenance = load_experiment(args)
    oracle_rows, diagnostic_report = diagnostics(splits, args)
    if args.diagnostics_only:
        return
    observations, actions = teacher_dataset(splits["train"], args.output)
    speed = training(args.seeds[0], "benchmark", None, splits, args, benchmark=True)
    probe = OverrideActorCritic(len(V2_OBSERVATION_NAMES))
    val_started = time.perf_counter()
    evaluate(probe, splits["val"])
    val_seconds = time.perf_counter() - val_started
    estimate = 2 * len(args.seeds) * (args.steps / speed + (args.steps / args.eval_every + 1) * val_seconds) + 30
    print(f"CPU estimate BEFORE imitation/full PPO training: {estimate/60:.2f} minutes for {2*len(args.seeds)} runs x {args.steps} steps plus imitation. {speed:.0f} steps/s. Fixed lr=5e-5, clip=0.1, entropy=0.01; no further hyperparameter search.", flush=True)
    write_json(args.output / "timing.json", {"estimated_training_seconds": estimate, "benchmark_steps_per_second": speed, "preparation_seconds": time.perf_counter()-started})
    models, histories, selection, imitation_history = {}, {}, {}, {}
    for seed in args.seeds:
        torch.manual_seed(seed)
        imitation = OverrideActorCritic(len(V2_OBSERVATION_NAMES))
        imitation_history[str(seed)] = fit_imitation(imitation, observations, actions, seed=seed, epochs=args.imitation_epochs)
        torch.save({"model": imitation.state_dict(), "seed": seed, "kind": "imitation", "observation_names": V2_OBSERVATION_NAMES,
                    "epochs": args.imitation_epochs, "training_keys": sorted({page["key"] for page in splits["train"]})}, args.output / f"imitation_seed_{seed}.pt")
        models[(seed, "imitation")] = imitation
        selection[str(seed)] = {}
        for mode in ("fine_tuned", "scratch"):
            model, history, selected = training(seed, mode, imitation.state_dict() if mode == "fine_tuned" else None, splits, args)
            models[(seed, mode)] = model
            histories[f"{mode}/{seed}"] = history
            selection[str(seed)][mode] = selected
    # Test diagnostics above are never fed to any fitting or selection path.
    # Final candidate evaluation occurs only after every selection finishes.
    baseline = {name: [result_row(page, page[field]) for page in splits["test"]] for name, field in (("native", "native"), ("raw_gat", "raw"), ("gated_gat", "gated"))}
    baseline["per_decision_oracle"] = oracle_rows
    seeds, significance = {}, {}
    native_values = [row["tau"] for row in baseline["native"]]
    gated_values = [row["tau"] for row in baseline["gated_gat"]]
    for seed in args.seeds:
        rows = {**baseline, **{mode: evaluate(models[(seed, mode)], splits["test"]) for mode in ("imitation", "fine_tuned", "scratch")}}
        seeds[str(seed)] = {"table": {method: summary(value) for method, value in rows.items()}, "rows": rows}
        for method, values in rows.items():
            significance[f"{method}/{seed}"] = {
                "method - native": paired_significance([row["tau"] for row in values], native_values, 10_000, 314159),
                "method - gated": paired_significance([row["tau"] for row in values], gated_values, 10_000, 314159)}
    methods = tuple(baseline) + ("imitation", "fine_tuned", "scratch")
    across = {}
    # Null decision-count denominators for arbitrary full permutations are
    # retained in per-seed tables, omitted only from numeric seed aggregation.
    for method in methods:
        numeric = [{key: value for key, value in seeds[str(seed)]["table"][method].items() if key not in ("decisions", "fraction_of_decisions")} for seed in args.seeds]
        across[method] = across_seed_summary(numeric)
        means = np.mean([[row["tau"] for row in seeds[str(seed)]["rows"][method]] for seed in args.seeds], axis=0).tolist()
        significance[f"{method}/mean_seed"] = {"method - native": paired_significance(means, native_values, 10_000, 314159),
                                                "method - gated": paired_significance(means, gated_values, 10_000, 314159)}
    accepted = across["fine_tuned"]["mean_tau"]["mean"] >= across["gated_gat"]["mean_tau"]["mean"] - 0.002 and across["fine_tuned"]["native_correct_tau"]["mean"] >= across["gated_gat"]["native_correct_tau"]["mean"] - 0.001
    improved = significance["fine_tuned/mean_seed"]["method - gated"]["difference"]["low"] > 0
    message = ("PPO accepted" if accepted else "PPO rejected") + ("; PPO improved" if improved else "")
    report = {"acceptance_message": message, "accepted": accepted, "improved": improved,
              "acceptance_rule": '"PPO accepted" iff fine-tuned mean-seed tau >= gated tau - 0.002 AND fine-tuned native-correct tau >= gated native-correct tau - 0.001. Add "PPO improved" iff paired 95% CI for fine-tuned minus gated has lower bound > 0.',
              "methodology": "Exact frozen v1 cache and original 1610/345/345 page splits; prior 4.0, saved validation-tuned gate thresholds. No GAT inference or training. All 13 previous observation features plus four precomputed OCR/model-only lookahead features. Reward=-0.005 per override plus terminal 20*(tau-native_tau), with normalized advantages. Imitation uses fixed-epoch training-only cross-entropy on the gated teacher's projected binary choices; action-space projection and oracle limits are detailed in diagnostics.md. Fine-tuning can retain imitation at step zero; scratch selects trained checkpoints only. All checkpoint selection uses PPO-validation tau with fewer overrides as tie-break; no test-driven hyperparameter search. Seed std is population std. Original overlapping GAT fold training sets mean this secondary PPO split remains conditional on the frozen OOF experiment, not a fully nested independent end-to-end evaluation.",
              "provenance": provenance, "thresholds": thresholds, "observation_names": V2_OBSERVATION_NAMES,
              "seed_values": args.seeds, "split_counts": {key: len(value) for key, value in splits.items()},
              "seeds": seeds, "across_seeds": across, "selection": selection, "histories": histories,
              "imitation_history": imitation_history, "significance": significance, "bootstrap_samples": 10_000,
              "bootstrap_seed": 314159, "exact_reachable_ceiling": diagnostic_report["exact_reachable_oracle"],
              "timing": {"total_seconds": time.perf_counter()-started, "estimated_training_seconds": estimate}}
    curves(histories, args.output)
    write_report(report, args.output)
    print(message, flush=True)
    print(json.dumps({"means": {method: value["mean_tau"] for method, value in across.items()}, "selection": selection, "seconds": report["timing"]["total_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
