from __future__ import annotations


class EarlyStopper:
    def __init__(self, patience_rounds: int, min_delta: float, warmup_rounds: int):
        self.patience_rounds = int(patience_rounds)
        self.min_delta = float(min_delta)
        self.warmup_rounds = int(warmup_rounds)
        self.best_acc = float("-inf")
        self.best_round = -1
        self.bad_rounds = 0

    def update(self, round_id: int, acc: float, chance_acc: float | None = None) -> dict:
        # Adaptive warmup: if accuracy is still below chance level (1 / num_classes),
        # keep warmup active and do not count bad rounds.
        below_chance = bool(chance_acc is not None and float(acc) < float(chance_acc))
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
        }
