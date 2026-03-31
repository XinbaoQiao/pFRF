from __future__ import annotations

DEFAULT_WARMUP_ACC_FLOOR = 0.10


class EarlyStopper:
    def __init__(self, patience_rounds: int, min_delta: float, warmup_rounds: int):
        self.patience_rounds = int(patience_rounds)
        self.min_delta = float(min_delta)
        self.warmup_rounds = int(warmup_rounds)
        self.best_acc = float("-inf")
        self.best_round = -1
        self.bad_rounds = 0

    def update(self, round_id: int, acc: float, chance_acc: float | None = None) -> dict:
        # Keep warmup active while accuracy is still clearly undertrained.
        # By default, any top-1 below 10% is treated as warmup even if chance is lower.
        warmup_threshold = max(float(chance_acc) if chance_acc is not None else 0.0, float(DEFAULT_WARMUP_ACC_FLOOR))
        below_chance = bool(float(acc) < warmup_threshold)
        improved = acc > (self.best_acc + self.min_delta)
        if improved:
            self.best_acc = float(acc)
            self.best_round = int(round_id)
            self.bad_rounds = 0
        else:
            if round_id >= self.warmup_rounds and not below_chance:
                self.bad_rounds += 1
        is_converged = (
            round_id >= self.warmup_rounds
            and not below_chance
            and self.bad_rounds >= self.patience_rounds
        )
        return {
            "improved": improved,
            "is_converged": is_converged,
            "best_acc": float(self.best_acc),
            "best_round": int(self.best_round),
            "bad_rounds": int(self.bad_rounds),
            "below_chance": below_chance,
            "warmup_threshold": float(warmup_threshold),
        }
