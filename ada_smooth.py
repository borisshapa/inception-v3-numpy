import numpy as np


def ada_smooth_optimizer(variables, gradients, config, state):
    state.setdefault("sum_delta", {})
    state.setdefault("sum_abs_delta", {})
    state.setdefault("norm", {})
    state.setdefault("t", 0)
    state["t"] += 1

    for k in ["learning_rate", "epsilon", "fast_decay", "slow_decay"]:
        assert k in config, config.keys()

    assert (
        config["slow_decay"] > config["fast_decay"]
    ), f"slow decay: {config['slow_decay']}, fast decay: {config['fast_decay']}"

    var_index = 0
    for var, grad in zip(variables, gradients):
        sum_delta = state["sum_delta"].setdefault(var_index, np.zeros_like(var))
        sum_abs_delta = state["sum_abs_delta"].setdefault(var_index, np.zeros_like(var))
        norm_term = state["norm"].setdefault(var_index, np.zeros_like(grad))

        er = np.divide(
            abs(sum_delta), sum_abs_delta, out=np.zeros_like(sum_delta), where=sum_abs_delta != 0
        )
        smoothing = (config["slow_decay"] - config["fast_decay"]) * er + (
            1 - config["slow_decay"]
        )
        smoothing_sqr = smoothing ** 2

        np.add(
            smoothing_sqr * grad ** 2,
            (1 - smoothing_sqr) * norm_term,
            out=norm_term,
        )

        delta = -config["learning_rate"] * grad / np.sqrt(norm_term + config["epsilon"])

        np.add(sum_delta, delta, out=sum_delta)
        np.add(sum_abs_delta, abs(delta), out=sum_abs_delta)

        var += delta

        assert sum_delta is state["sum_delta"].get(var_index)
        assert sum_abs_delta is state["sum_abs_delta"].get(var_index)
        assert norm_term is state["norm"].get(var_index)
        var_index += 1
