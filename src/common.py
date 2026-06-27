from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

from src.environments.base_env import board


ACTION_NAMES = ["up", "right", "down", "left"]


try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - exercised only in minimal envs
    def _tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(kwargs.get("total", 0))


def progress(iterable: Iterable, **kwargs):
    return _tqdm(iterable, **kwargs)


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str
    seed: int
    workers: int
    output_dir: str
    search_games: int
    search_depth: int
    regret_boards: int
    planning_boards: int
    planning_rollouts: list[int]
    planning_stability_runs: int
    planning_eval_games: int
    max_game_steps: int | None
    q_episodes: int
    q_eval_games: int
    q_bias_episode: int
    q_bias_samples: int
    q_td_window: int


def default_config(mode: str, seed: int, workers: int | None, output_dir: str) -> ExperimentConfig:
    cpu_workers = max(1, (os.cpu_count() or 2) - 1)
    if mode == "smoke":
        return ExperimentConfig(
            mode=mode,
            seed=seed,
            workers=workers or 1,
            output_dir=output_dir,
            search_games=1,
            search_depth=3,
            regret_boards=1,
            planning_boards=3,
            planning_rollouts=[5, 10],
            planning_stability_runs=3,
            planning_eval_games=1,
            max_game_steps=3,
            q_episodes=8,
            q_eval_games=1,
            q_bias_episode=6,
            q_bias_samples=2,
            q_td_window=10,
        )

    return ExperimentConfig(
        mode=mode,
        seed=seed,
        workers=workers or cpu_workers,
        output_dir=output_dir,
        search_games=100,
        search_depth=3,
        regret_boards=500,
        planning_boards=10,
        # planning_rollouts=[200, 500, 1000, 2000],
        planning_rollouts=[200, 500],
        planning_stability_runs=50,
        planning_eval_games=20,
        # max_game_steps=None,
        # q_episodes=100_000,
        # q_eval_games=10,
        # q_bias_episode=90_000,
        # q_bias_samples=100,
        # q_td_window=1000,
        max_game_steps=None,
        q_episodes=25_000,
        q_eval_games=100,
        q_bias_episode=90_000,
        q_bias_samples=100,
        q_td_window=1000,
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--seed", type=int, default=181)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output-dir", default=os.path.join("models","phrase_1","eval_results"))

def add_common_args_2(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--seed", type=int, default=181)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output-dir", default=os.path.join("models","phrase_2", "eval_results"))


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return default_config(args.mode, args.seed, args.workers, args.output_dir)


def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def max_tile_value(b: board) -> int:
    return (1 << max(b.at(i) for i in range(16))) & ~1


def empty_count(b: board) -> int:
    return sum(1 for i in range(16) if b.at(i) == 0)


def legal_actions(b: board) -> list[int]:
    actions = []
    for action in range(4):
        trial = board(b.raw)
        if trial.move(action) != -1:
            actions.append(action)
    return actions


def apply_action(raw: int, action: int) -> tuple[int, int]:
    trial = board(raw)
    reward = trial.move(action)
    return trial.raw, reward


def popup_with_rng(b: board, rng: random.Random, p4: float = 0.1) -> None:
    spaces = [i for i in range(16) if b.at(i) == 0]
    if not spaces:
        return
    b.set(rng.choice(spaces), 2 if rng.random() < p4 else 1)


def spawn_initial_board(rng: random.Random) -> board:
    b = board()
    popup_with_rng(b, rng)
    popup_with_rng(b, rng)
    return b


def play_policy(
    seed: int,
    choose_action: Callable[[board], int],
    p4: float = 0.1,
    max_steps: int | None = None,
) -> dict[str, float]:
    rng = random.Random(seed)
    random.seed(seed)
    b = spawn_initial_board(rng)
    score, steps, decision_time = 0, 0, []
    while max_steps is None or steps < max_steps:
        start = time.perf_counter()
        action = choose_action(b)
        decision_time.append(time.perf_counter() - start)
        next_b = board(b.raw)
        reward = next_b.move(action)
        if reward == -1:
            break
        score += reward
        steps += 1
        popup_with_rng(next_b, rng, p4=p4)
        b = next_b
    return {
        "score": float(score),
        "steps": float(steps),
        "max_tile": float(max_tile_value(b)),
        "time_per_step_ms": 1000.0 * safe_mean(decision_time),
    }


def safe_mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def safe_std(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    mu = safe_mean(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


def entropy_from_counts(counts: dict[Any, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count:
            p = count / total
            entropy -= p * math.log(p)
    return entropy

def summarize_games(records: list[dict[str, float]]) -> dict[str, float]:
    max_tiles = [r["max_tile"] for r in records]
    total_games = max(1, len(max_tiles))
    return {
        "average_score": safe_mean(r["score"] for r in records),
        "average_steps": safe_mean(r["steps"] for r in records),
        "time_per_step_ms": safe_mean(r["time_per_step_ms"] for r in records),
        "rate_1024": sum(1 for t in max_tiles if t >= 1024) / total_games,
        "rate_2048": sum(1 for t in max_tiles if t >= 2048) / total_games,
        "rate_4096": sum(1 for t in max_tiles if t >= 4096) / total_games,
        "rate_8192": sum(1 for t in max_tiles if t >= 8192) / total_games,   # 【新增】
        "rate_16384": sum(1 for t in max_tiles if t >= 16384) / total_games, # 【新增】
        "rate_32768": sum(1 for t in max_tiles if t >= 32768) / total_games, # 【新增】
    }

def generate_pressure_boards(count: int, seed: int, max_empty: int = 3) -> list[int]:
    board.lookup.init()
    rng = random.Random(seed)
    boards: list[int] = []
    attempts = 0
    while len(boards) < count and attempts < count * 200:
        attempts += 1
        b = spawn_initial_board(rng)
        for _ in range(1200):
            actions = legal_actions(b)
            if not actions:
                break
            action = rng.choice(actions)
            reward = b.move(action)
            if reward == -1:
                break
            popup_with_rng(b, rng)
            if empty_count(b) <= max_empty:
                boards.append(b.raw)
                break
    while len(boards) < count:
        boards.append(make_fallback_pressure_board(seed + len(boards)).raw)
    return boards


def make_fallback_pressure_board(seed: int) -> board:
    rng = random.Random(seed)
    values = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 0]
    rng.shuffle(values)
    b = board()
    for i, value in enumerate(values):
        b.set(i, value)
    if not legal_actions(b):
        b.set(15, 0)
        b.set(14, 1)
    return b


def write_result_bundle(
    output_dir: str,
    prefix: str,
    config: ExperimentConfig,
    summary_rows: list[dict[str, Any]],
    details: dict[str, Any] | None = None,
) -> dict[str, str]:
    ensure_output_dir(output_dir)
    stamp = timestamp()
    base = os.path.join(output_dir, f"{prefix}_{config.mode}_{stamp}")
    csv_path, json_path, md_path = f"{base}.csv", f"{base}.json", f"{base}.md"
    if summary_rows:
        fieldnames: list[str] = []
        for row in summary_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("")

    payload = {
        "config": asdict(config),
        "summary": summary_rows,
        "details": details or {},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {prefix} results ({config.mode})\n\n")
        f.write(f"- seed: `{config.seed}`\n")
        f.write(f"- workers: `{config.workers}`\n\n")
        if summary_rows:
            headers = []
            for row in summary_rows:
                for key in row.keys():
                    if key not in headers:
                        headers.append(key)
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join("---" for _ in headers) + " |\n")
            for row in summary_rows:
                f.write("| " + " | ".join(format_cell(row.get(h, "")) for h in headers) + " |\n")
    return {"csv": csv_path, "json": json_path, "md": md_path}


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
