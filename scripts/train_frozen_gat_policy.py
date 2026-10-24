"""Frozen out-of-fold GAT encoder plus a trainable RETAIN/REORDER PPO head."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.common import write_json
from rlocr.frozen_gat_policy import FrozenGATPolicyEnv, mask_logits
from rlocr.gated_decoder import paired_significance
from rlocr.order_actions import ACTION_NAMES, RETAIN, REORDER
from rlocr.ppo import PPOConfig
from rlocr.ppo_gate import OverrideActorCritic, ppo_update
from rlocr.ppo_gate_v2 import fit_imitation, gated_teacher_action
from cache_gat_embeddings import cache_embeddings, normalize_embeddings
from label_ordering_actions import label_existing_reports
from train_ppo_gate import Rollout, across_seed_summary
from train_ppo_gate_v2 import load_experiment, result_row, summary as previous_summary, curves


def evaluate(model, pages):
    envs = [FrozenGATPolicyEnv(page) for page in pages]
    active, observations = [], []
    for i, env in enumerate(envs):
        observation, info = env.reset()
        if not info["automatic_terminal"]:
            active.append(i)
            observations.append(observation)
    while active:
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(np.asarray(observations)))
            masks = torch.from_numpy(np.asarray([envs[i].action_masks() for i in active]))
            actions = mask_logits(logits, masks).argmax(-1).tolist()
        following, states = [], []
        for i, action in zip(active, actions):
            observation, _, done, _, _ = envs[i].step(action)
            if not done:
                following.append(i)
                states.append(observation)
        active, observations = following, states
    return [{**result_row(page, env.order, env.disagreements), "invalid_actions": env.invalid_actions,
             "reorder_attempts": env.reorder_attempts} for page, env in zip(pages, envs)]


def teacher_data(pages, output):
    observations, actions, keys = [], [], []
    rejected_teacher_reorders = 0
    for page in pages:
        env = FrozenGATPolicyEnv(page)
        observation, info = env.reset()
        done = info["automatic_terminal"]
        while not done:
            action = gated_teacher_action(env)
            if action == REORDER and env.invalid_reorder_jump(*env._proposals()):
                action = RETAIN
                rejected_teacher_reorders += 1
            if not env.action_masks()[int(action)]:
                action = RETAIN
            observations.append(observation)
            actions.append(int(action))
            keys.append(page["key"])
            observation, _, done, _, _ = env.step(int(action))
    data = np.asarray(observations)
    target = np.asarray(actions, dtype=np.int64)
    np.savez_compressed(output / "imitation_choices.npz", observations=data, actions=target, page_keys=np.asarray(keys))
    return torch.from_numpy(data), torch.from_numpy(target), rejected_teacher_reorders


def training(seed, mode, initial, splits, args, dimension, benchmark=False):
    torch.manual_seed(seed)
    model = OverrideActorCritic(dimension)
    if initial is not None:
        model.load_state_dict(initial)
    config = PPOConfig(epochs=4, clip_ratio=0.1, entropy_coef=0.01, learning_rate=5e-5, gamma=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    rollout = Rollout(splits["train"], seed, env_class=FrozenGATPolicyEnv)
    steps, next_evaluation, best = 0, 0, None
    history, completed = [], []
    limit = 4096 if benchmark else args.steps
    checkpoint_path = args.output / f"{mode}_seed_{seed}.pt"
    started = time.perf_counter()
    while True:
        if not benchmark and (steps >= next_evaluation or steps >= limit):
            rows = evaluate(model, splits["val"])
            tau = float(np.mean([row["tau"] for row in rows]))
            overrides = sum(row["overrides"] for row in rows)
            score = (tau, -overrides)
            if (steps > 0 or initial is not None) and (best is None or score > best):
                best = score
                torch.save({"model": model.state_dict(), "input_dim": dimension, "seed": seed,
                            "mode": mode, "steps": steps, "selected_kind": "imitation" if steps == 0 else "ppo",
                            "validation_tau": tau, "validation_overrides": overrides,
                            "action_names": ACTION_NAMES, "config": vars(config),
                            "encoder_frozen": True, "normalization": args.normalization}, checkpoint_path)
            history.append({"steps": steps, "validation_tau": tau, "validation_overrides": overrides,
                            "validation_invalid_actions": sum(row["invalid_actions"] for row in rows),
                            "reward": float(np.mean([value[0] for value in completed])) if completed else None,
                            "train_tau": float(np.mean([value[1] for value in completed])) if completed else None,
                            "seconds": time.perf_counter()-started})
            completed = []
            next_evaluation = steps + args.eval_every
            write_json(args.output / f"{mode}_seed_{seed}_history.json", history)
            print(f"{mode} seed={seed} steps={steps} val_tau={tau:.6f} REORDER={overrides} invalid={history[-1]['validation_invalid_actions']} seconds={time.perf_counter()-started:.1f}", flush=True)
        if steps >= limit:
            break
        batch, episodes = rollout.collect(model, 128, config)
        ppo_update(model, optimizer, *batch, config, generator, action_masks=rollout.last_action_masks)
        completed.extend(episodes)
        steps += 2048
    if benchmark:
        return steps / (time.perf_counter()-started)
    selected = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model"])
    metadata = {key: value for key, value in selected.items() if key not in ("model", "normalization")}
    return model, history, metadata


def summarize(rows):
    stats = previous_summary(rows)
    stats.update({"invalid_actions": sum(row.get("invalid_actions", 0) for row in rows),
                  "reorder_attempts": sum(row.get("reorder_attempts", row["overrides"]) for row in rows)})
    return stats


def constrained_oracle(pages):
    """True-next greedy oracle under this environment's validity rule."""
    rows = []
    for page in pages:
        env = FrozenGATPolicyEnv(page)
        _, info = env.reset()
        done = info["automatic_terminal"]
        while not done:
            native, gat = env._proposals()
            true_next = int(np.argmin(np.where(env.remaining, page["gt_rank"], np.inf)))
            action = int(gat == true_next and not env.invalid_reorder_jump(native, gat))
            _, _, done, _, _ = env.step(action)
        rows.append({**result_row(page, env.order, env.disagreements), "invalid_actions": env.invalid_actions,
                     "reorder_attempts": env.reorder_attempts})
    return rows


def refresh_oracle(report, pages):
    """Refresh only oracle diagnostics; never refit or reselect any policy."""
    values = constrained_oracle(pages)
    for seed, result in report["seeds"].items():
        result["rows"]["per_decision_oracle"] = values
        result["table"]["per_decision_oracle"] = summarize(values)
        report["significance"][f"per_decision_oracle/{seed}"] = {
            baseline: paired_significance([row["tau"] for row in values],
                [row["tau"] for row in result["rows"][baseline]], 10_000, 314159)
            for baseline in ("native", "gated_gat", "ppo_v2")}
    numeric = [{key: value for key, value in result["table"]["per_decision_oracle"].items()
                if key not in ("decisions", "fraction_of_decisions")} for result in report["seeds"].values()]
    report["across_seeds"]["per_decision_oracle"] = across_seed_summary(numeric)
    report["significance"]["per_decision_oracle/mean_seed"] = {
        baseline: paired_significance([row["tau"] for row in values],
            np.mean([[row["tau"] for row in result["rows"][baseline]] for result in report["seeds"].values()], axis=0).tolist(),
            10_000, 314159) for baseline in ("native", "gated_gat", "ppo_v2")}
    report["oracle_definition"] = "Greedy true-next oracle respecting the current column-jump restriction; otherwise RETAIN. This is not asserted to be a global tau ceiling."


def write_report(report, output):
    report["methodology"] = report["methodology"].replace(
        "All test evaluation and inference are after selection.",
        "Candidate policy test scoring occurs after selection; frozen encoder feature extraction includes all splits.")
    lines = ["# Frozen-GAT RETAIN/REORDER PPO policy", "", report["outcome"], "",
             f"Main sequential policy result: `{report['main_result']}`. Primary comparison is the imitation-start branch, fixed before test evaluation; scratch is an ablation.", "", "## MDP summary", ""]
    for name, value in report["mdp"].items():
        lines += [f"**{name}:** {value}", ""]
    lines += ["## Protocol", "", report["methodology"], "", report.get("oracle_definition", ""), "",
              "The observation and validity-rule changes are evaluated together; this is not a pure embedding-only ablation.", "",
              "## Mean +/- population std across seeds", "",
              "| Method | Mean tau | Native-correct | Native-wrong | REORDER placements | Better | Worse | Same | Invalid attempts |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda value: f"{value['mean']:.6f} +/- {value['std']:.6f}"
    for method, value in report["across_seeds"].items():
        lines.append("| " + method + " | " + " | ".join(fmt(value[key]) for key in ("mean_tau", "native_correct_tau", "native_wrong_tau", "overrides", "better", "worse", "same", "invalid_actions")) + " |")
    lines += ["", "REORDER placements count actual non-native placements; invalid REORDER attempts are rejected and execute RETAIN. Historical methods did not have the new invalid-action rule, so their invalid count is not an audit of old trajectories.", "", "## Per-seed results", "",
              "| Seed | Method | Tau | Native-correct | Native-wrong | REORDER placements | Better | Worse | Same |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for seed, value in report["seeds"].items():
        for method, stats in value["table"].items():
            lines.append(f"| {seed} | {method} | {stats['mean_tau']:.6f} | {stats['native_correct_tau']:.6f} | {stats['native_wrong_tau']:.6f} | {stats['overrides']} | {stats['better']} | {stats['worse']} | {stats['same']} |")
    lines += ["", "## Validation checkpoint selection", "", "| Seed | Branch | Selected | Steps | Validation tau |", "|---|---|---|---:|---:|"]
    for seed, branches in report["selection"].items():
        for mode, selected in branches.items():
            lines.append(f"| {seed} | {mode} | {selected['selected_kind']} | {selected['steps']} | {selected['validation_tau']:.6f} |")
    lines += ["", "## Paired inference", "", "10,000 paired page bootstrap resamples and sign flips, fixed seed 314159. Seed means are computed within each page before resampling; seed replications are not extra independent pages.", "",
              "| Method / seed | Versus | Mean difference | 95% CI | Sign-flip p |", "|---|---|---:|---|---:|"]
    for method, comparisons in report["significance"].items():
        for baseline, value in comparisons.items():
            difference = value["difference"]
            lines.append(f"| {method} | {baseline} | {difference['mean']:+.6f} | [{difference['low']:+.6f}, {difference['high']:+.6f}] | {value['permutation_p_value']:.6f} |")
    lines += ["", "## Per-template tau", "", "| Method | Template | Mean +/- std |", "|---|---|---:|"]
    for method, value in report["across_seeds"].items():
        for template, tau in value["per_template_tau"].items():
            lines.append(f"| {method} | {template} | {fmt(tau)} |")
    lines += ["", "## Training plots", "", "![Reward and validation tau](training_curves.png)", "",
              "Numeric action IDs remain 0=RETAIN and 1=REORDER. Legacy class defaults, checkpoint tensor names and dimensions are preserved. Old policy checkpoints remain loadable with their original input dimensions.", "",
              "The encoder folds were fitted independently and their representation coordinates need not align across folds. Observations use a compressed state summary and do not expose the complete placed-line mask to the MLP. The original overlapping GAT training folds also make this secondary PPO split conditional on the frozen OOF experiment, not a fully nested independent end-to-end evaluation.", "",
              "[Read-only merge/split audit](../merge_split_audit/report.md)", ""]
    write_json(output / "report.json", report)
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=Path("runs/ppo_gate"))
    parser.add_argument("--v2-report", type=Path, default=Path("runs/ppo_gate_v2/report.json"))
    parser.add_argument("--cv-dir", type=Path, default=Path("runs/gat_cross_validation"))
    parser.add_argument("--output", type=Path, default=Path("runs/ppo_gat_policy"))
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337, 4242, 90210])
    parser.add_argument("--imitation-epochs", type=int, default=30)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--render-only", action="store_true", help="Refresh report/plots and constrained oracle without training or selection")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    label_existing_reports()
    splits, thresholds, provenance = load_experiment(args)
    embedding_metadata = cache_embeddings(splits, args, provenance)
    args.normalization = normalize_embeddings(splits)
    if args.render_only:
        report = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
        refresh_oracle(report, splits["test"])
        curves(report["histories"], args.output, title="Frozen GAT policy: RETAIN / REORDER")
        write_report(report, args.output)
        return
    dimension = FrozenGATPolicyEnv(splits["train"][0]).observation_space.shape[0]
    observations, actions, rejected_teacher = teacher_data(splits["train"], args.output)
    speed = training(args.seeds[0], "benchmark", None, splits, args, dimension, benchmark=True)
    probe = OverrideActorCritic(dimension)
    validation_started = time.perf_counter()
    evaluate(probe, splits["val"])
    validation_seconds = time.perf_counter()-validation_started
    estimate = 2*len(args.seeds)*(args.steps/speed + (args.steps/args.eval_every+1)*validation_seconds) + 30
    timing = {"estimated_training_seconds": estimate, "benchmark_steps_per_second": speed,
              "preparation_seconds": time.perf_counter()-started}
    write_json(args.output / "timing.json", timing)
    print(f"CPU estimate BEFORE full training: {estimate/60:.2f} minutes for six runs of {args.steps} steps plus imitation. Preparation {timing['preparation_seconds']:.1f}s; encoder frozen; head input {dimension}.", flush=True)
    if args.benchmark_only:
        return
    models, selection, histories = {}, {}, {}
    for seed in args.seeds:
        torch.manual_seed(seed)
        imitation = OverrideActorCritic(dimension)
        fit_imitation(imitation, observations, actions, seed=seed, epochs=args.imitation_epochs)
        torch.save({"model": imitation.state_dict(), "input_dim": dimension, "kind": "imitation",
                    "seed": seed, "action_names": ACTION_NAMES, "normalization": args.normalization}, args.output / f"imitation_seed_{seed}.pt")
        selection[str(seed)] = {}
        for mode in ("scratch", "imitation_start"):
            model, history, selected = training(seed, mode, imitation.state_dict() if mode == "imitation_start" else None,
                                                splits, args, dimension)
            models[(seed, mode)] = model
            selection[str(seed)][mode] = selected
            histories[f"{mode}/{seed}"] = history
    previous = json.loads(args.v2_report.read_text(encoding="utf-8"))
    # Only after all candidate selections finish, consume frozen test rows.
    methods = ("native", "raw_gat", "gated_gat", "per_decision_oracle", "ppo_v2", "gat_scratch", "gat_imitation_start")
    seeds, significance = {}, {}
    for seed in args.seeds:
        previous_rows = previous["seeds"][str(seed)]["rows"]
        rows = {name: previous_rows[name] for name in ("native", "raw_gat", "gated_gat", "per_decision_oracle")}
        rows["ppo_v2"] = previous_rows["fine_tuned"]
        expected = [page["key"] for page in splits["test"]]
        if any([row["key"] for row in values] != expected for values in rows.values()):
            raise ValueError("Historical PPO v2 test page ordering/split mismatch")
        rows["gat_scratch"] = evaluate(models[(seed, "scratch")], splits["test"])
        rows["gat_imitation_start"] = evaluate(models[(seed, "imitation_start")], splits["test"])
        seeds[str(seed)] = {"table": {method: summarize(values) for method, values in rows.items()}, "rows": rows}
        for method, values in rows.items():
            significance[f"{method}/{seed}"] = {baseline: paired_significance(
                [row["tau"] for row in values], [row["tau"] for row in rows[baseline]], 10_000, 314159)
                for baseline in ("native", "gated_gat", "ppo_v2")}
    across = {}
    for method in methods:
        numeric = [{key: value for key, value in seeds[str(seed)]["table"][method].items()
                    if key not in ("decisions", "fraction_of_decisions")} for seed in args.seeds]
        across[method] = across_seed_summary(numeric)
        means = np.mean([[row["tau"] for row in seeds[str(seed)]["rows"][method]] for seed in args.seeds], axis=0).tolist()
        significance[f"{method}/mean_seed"] = {baseline: paired_significance(means,
            np.mean([[row["tau"] for row in seeds[str(seed)]["rows"][baseline]] for seed in args.seeds], axis=0).tolist(),
            10_000, 314159) for baseline in ("native", "gated_gat", "ppo_v2")}
    improved = significance["gat_imitation_start/mean_seed"]["ppo_v2"]["difference"]["low"] > 0
    report = {"outcome": "GAT-policy PPO improved" if improved else "GAT-policy PPO did not improve; keep PPO v2 as the main sequential result",
              "improved": improved, "main_result": "gat_imitation_start" if improved else "ppo_v2",
              "mdp": {
                  "State": f"Latent state is the page graph, placed order and remaining-line mask. Policy observation ({dimension} floats): native/REORDER candidate embeddings, mean-pooled page embedding, four v2 page lookahead features, placed fraction, and one-hot last decision (none/RETAIN/REORDER). Embedding and lookahead standardization is fitted only on PPO-train.",
                  "Actions": "RETAIN=0 places native's earliest remaining OCR line. REORDER=1 places the highest-scoring remaining GAT line at prior 4.0. Agreeing proposals are automatic. Already-placed lines are masked before proposal choice and policy sampling. Numeric IDs and legacy checkpoint layouts are unchanged.",
                  "Reward": "20*(terminal tau - native tau) - 0.005 per REORDER attempt - 0.05 per invalid attempt; no dense placement bonus. A rejected attempt executes RETAIN to ensure progress.",
                  "Invalid edit": "A placed-line choice is invalid and masked. A REORDER skipping >1 inferred left-edge column is invalid when native remains in the column of the last placed line. OCR-only column clusters use consecutive left-edge gaps >max(3.5% page width, twice median line height). No GT column labels enter inference.",
                  "Episode": "One page. Decision steps occur only for differing proposals. Episode ends when every evaluated OCR line is placed exactly once; ordering units remain OCR lines, not newly aggregated GT blocks."},
              "methodology": "Frozen original held-out-fold GAT encoders, cached node/mean pooled embeddings and unchanged cached probabilities. Exact previous PPO page splits and validation-tuned gate reused. Only a 2x64 PPO MLP head is trained: lr=5e-5, clip=0.1, entropy=0.01, normalized advantages, GAE. Three fixed seeds per scratch/imitation branch. Imitation refits the head on PPO-train gated-teacher projected choices in the new observation space; invalid teacher jumps are relabeled RETAIN. Original v2 imitation input dimensions differ and are not silently reused. Validation mean tau selects checkpoints, ties by fewer actual REORDER placements; imitation-start can retain its warm start, scratch uses trained checkpoints only. All test evaluation and inference are after selection. Previous PPO v2 is the saved sequential fine-tuned v2 branch, not the separate newer page-level gate.",
              "action_names": ACTION_NAMES, "encoder_frozen": True, "embedding_cache": embedding_metadata,
              "normalization": args.normalization, "provenance": provenance, "thresholds": thresholds,
              "rejected_teacher_reorders": rejected_teacher, "selection": selection, "seeds": seeds,
              "across_seeds": across, "significance": significance, "histories": histories,
              "bootstrap_samples": 10_000, "bootstrap_seed": 314159,
              "timing": {**timing, "total_seconds": time.perf_counter()-started}}
    refresh_oracle(report, splits["test"])
    curves(histories, args.output, title="Frozen GAT policy: RETAIN / REORDER")
    write_report(report, args.output)
    print(report["outcome"], flush=True)
    print(json.dumps({"means": {method: value["mean_tau"] for method, value in across.items()},
                      "versus_v2": {key: value for key, value in significance["gat_imitation_start/mean_seed"]["ppo_v2"].items() if key != "differences"},
                      "seconds": report["timing"]["total_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
