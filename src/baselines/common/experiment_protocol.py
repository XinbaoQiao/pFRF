from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EarlyStopConfig:
    patience_rounds: int = 10
    min_delta: float = 1e-4
    warmup_rounds: int = 10


@dataclass
class FederatedProtocol:
    seeds: list[int] = field(default_factory=lambda: [3407, 3408, 3409, 3410, 3411])
    max_rounds: int = 500
    local_epochs: int = 1
    local_batch_size: int = 64
    local_lr: float = 1e-3
    num_clients: int = 100
    partition: str = "iid"
    dirichlet_alpha: float = 0.01
    dirichlet_balance: bool = True
    dirichlet_min_size: int = 1
    shard_per_client: int = 2
    classes_per_client: int = 2
    early_stop: EarlyStopConfig = field(default_factory=EarlyStopConfig)
    profile_flops: bool = True
    profile_comm: bool = True


@dataclass
class CentralizedProtocol:
    seeds: list[int] = field(default_factory=lambda: [3407, 3408, 3409, 3410, 3411])
    batch_size: int = 256
    epochs: int = 100
    lr: float = 1e-3
    profile_flops: bool = True
