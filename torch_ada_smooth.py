from typing import Optional, Callable

import torch
from torch.optim import Optimizer


class AdaSmooth(Optimizer):
    def __init__(
        self,
        params,
        learning_rate: float,
        epsilon: float,
        fast_decay: float,
        slow_decay: float,
    ):
        if learning_rate < 0.0:
            raise ValueError(
                f"Invalid learning rate: {learning_rate}. Should be >= 0.0."
            )
        if epsilon < 0.0:
            raise ValueError(f"Invalid epsilon: {epsilon}. Should be >= 0.0.")
        if fast_decay < 0.0:
            raise ValueError(f"Invalid fast_decay: {fast_decay}. Should be >= 0.0.")
        if slow_decay < 0.0:
            raise ValueError(f"Invalid slow_decay: {slow_decay}. Should be >= 0.0.")

        defaults = {
            "lr": learning_rate,
            "eps": epsilon,
            "fast_decay": fast_decay,
            "slow_decay": slow_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                var = p.data
                grad = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["sum_delta"] = torch.zeros_like(var)
                    state["sum_abs_delta"] = torch.zeros_like(var)
                    state["norm"] = torch.zeros_like(grad)

                sum_delta, sum_abs_delta, norm = (
                    state["sum_delta"],
                    state["sum_abs_delta"],
                    state["norm"],
                )
                state["step"] += 1

                er = torch.zeros_like(sum_delta)
                mask = sum_abs_delta != 0
                er[mask] = torch.abs(sum_delta)[mask] / sum_abs_delta[mask]

                smoothing = (group["slow_decay"] - group["fast_decay"]) * er + (
                    1 - group["slow_decay"]
                )
                smoothing_sqr = smoothing ** 2

                norm = smoothing_sqr * grad ** 2 + (1 - smoothing_sqr) * norm

                delta = -group["lr"] * grad / torch.sqrt(norm + group["eps"])
                sum_delta.add_(delta)
                sum_abs_delta.add_(abs(delta))

                var.add_(delta)
        return loss
