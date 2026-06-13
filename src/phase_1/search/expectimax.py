import math
from collections.abc import Callable

from src.environments.base_env import board


class GreedyAgent:
    """One-ply baseline: evaluate the board immediately after each legal slide."""

    def __init__(self, value_func: Callable[[board], float]):
        self.value_func = value_func

    def get_best_action(self, b: board, max_depth: int = 1) -> tuple[int, float, float]:
        best_action, best_value = -1, -math.inf
        total, unique_afterstates = 0, set()
        for action in range(4):
            after = board(b.raw)
            reward = after.move(action)
            if reward == -1:
                continue
            total += 1
            unique_afterstates.add(after.raw)
            value = reward + self.value_func(after)
            if value > best_value:
                best_action, best_value = action, value

        if best_action == -1:
            return 0, 0.0, 0.0
        compression_ratio = len(unique_afterstates) / max(1, total)
        return best_action, compression_ratio, float(total)


class ExpectimaxAgent:
    """
    Expectimax with independent switches for tree topology and leaf evaluator input.

    ``max_depth`` is counted in semantic plies from the root decision:
    player action -> environment spawn -> player action is depth 3.
    ``use_afterstate`` controls the recorded/search topology.
    ``leaf_mode`` controls what the evaluator sees at the frontier:
    ``afterstate`` evaluates the board immediately after a slide, while ``state``
    averages over the following random tile spawn and evaluates the resulting
    ordinary states.
    """

    def __init__(
        self,
        use_afterstate: bool = False,
        value_func: Callable[[board], float] | None = None,
        leaf_mode: str = "state",
        prune_threshold: float = 0.0,
    ):
        if value_func is None:
            raise ValueError("value_func is required")
        if leaf_mode not in {"state", "afterstate"}:
            raise ValueError("leaf_mode must be 'state' or 'afterstate'")

        self.use_afterstate = use_afterstate
        self.value_func = value_func
        self.leaf_mode = leaf_mode
        self.prune_threshold = prune_threshold

        self.transposition_table: dict[tuple, float] = {}
        self.total_nodes_expanded = 0
        self.total_metric_visits = 0
        self.unique_metric_keys: set[tuple | int] = set()

    def get_best_action(self, b: board, max_depth: int = 3) -> tuple[int, float, float]:
        self.transposition_table.clear()
        self.total_nodes_expanded = 0
        self.total_metric_visits = 0
        self.unique_metric_keys.clear()

        best_value, best_action = -math.inf, -1
        depth = max(1, max_depth)
        for action in range(4):
            after = board(b.raw)
            reward = after.move(action)
            if reward == -1:
                continue
            self._record_decision_child(b.raw, action, after.raw)
            value = reward + self._after_action_value(after.raw, depth - 1, 1.0)
            if value > best_value:
                best_value, best_action = value, action

        if best_action == -1:
            return 0, 0.0, 0.0

        compression_ratio = len(self.unique_metric_keys) / max(1, self.total_metric_visits)
        b_eff = self.total_nodes_expanded ** (1.0 / depth) if self.total_nodes_expanded else 0.0
        return best_action, compression_ratio, b_eff

    def _record_decision_child(self, state_raw: int, action: int, after_raw: int) -> None:
        self.total_metric_visits += 1
        if self.use_afterstate:
            self.unique_metric_keys.add(after_raw)
        else:
            self.unique_metric_keys.add((state_raw, action))

    def _after_action_value(self, after_raw: int, plies_remaining: int, current_prob: float) -> float:
        self.total_nodes_expanded += 1
        if plies_remaining <= 0 or current_prob < self.prune_threshold:
            return self._leaf_value_after_action(after_raw)

        if self.use_afterstate:
            key = ("after", after_raw, plies_remaining, self.leaf_mode)
            if key in self.transposition_table:
                return self.transposition_table[key]
            value = self._chance_value(after_raw, plies_remaining, current_prob)
            self.transposition_table[key] = value
            return value

        return self._chance_value(after_raw, plies_remaining, current_prob)

    def _chance_value(self, after_raw: int, plies_remaining: int, current_prob: float) -> float:
        self.total_nodes_expanded += 1
        chance_key = ("chance", after_raw, plies_remaining, self.leaf_mode, self.use_afterstate)
        if chance_key in self.transposition_table:
            return self.transposition_table[chance_key]

        after = board(after_raw)
        empties = [i for i in range(16) if after.at(i) == 0]
        if not empties:
            value = self.value_func(after)
            self.transposition_table[chance_key] = value
            return value

        weight = 1.0 / len(empties)
        expected_value = 0.0
        for pos in empties:
            for tile, tile_prob in ((1, 0.9), (2, 0.1)):
                spawned = board(after_raw)
                spawned.set(pos, tile)
                expected_value += weight * tile_prob * self._state_value(
                    spawned.raw,
                    plies_remaining - 1,
                    current_prob * weight * tile_prob,
                )

        self.transposition_table[chance_key] = expected_value
        return expected_value

    def _state_value(self, state_raw: int, plies_remaining: int, current_prob: float) -> float:
        self.total_nodes_expanded += 1
        if plies_remaining <= 0 or current_prob < self.prune_threshold:
            return self.value_func(board(state_raw))

        state_key = ("state", state_raw, plies_remaining, self.leaf_mode, self.use_afterstate)
        if state_key in self.transposition_table:
            return self.transposition_table[state_key]

        best_value, has_action = -math.inf, False
        for action in range(4):
            after = board(state_raw)
            reward = after.move(action)
            if reward == -1:
                continue
            has_action = True
            self._record_decision_child(state_raw, action, after.raw)
            value = reward + self._after_action_value(after.raw, plies_remaining - 1, current_prob)
            if value > best_value:
                best_value = value

        value = best_value if has_action else self.value_func(board(state_raw))
        self.transposition_table[state_key] = value
        return value

    def _leaf_value_after_action(self, after_raw: int) -> float:
        if self.leaf_mode == "afterstate":
            return self.value_func(board(after_raw))

        after = board(after_raw)
        empties = [i for i in range(16) if after.at(i) == 0]
        if not empties:
            return self.value_func(after)

        weight = 1.0 / len(empties)
        expected_value = 0.0
        for pos in empties:
            for tile, tile_prob in ((1, 0.9), (2, 0.1)):
                spawned = board(after_raw)
                spawned.set(pos, tile)
                self.total_nodes_expanded += 1
                expected_value += weight * tile_prob * self.value_func(spawned)
        return expected_value
