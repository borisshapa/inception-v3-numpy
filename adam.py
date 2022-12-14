import numpy as np


def adam_optimizer(variables, gradients, config, state):
    # 'variables' and 'gradients' have complex structure, accumulated_grads will be stored in a simpler one
    state.setdefault("m", {})  # first moment vars
    state.setdefault("v", {})  # second moment vars
    state.setdefault("t", 0)  # timestamp
    state["t"] += 1
    for k in ["learning_rate", "beta1", "beta2", "epsilon"]:
        assert k in config, config.keys()

    var_index = 0
    lr_t = (
        config["learning_rate"]
        * np.sqrt(1 - config["beta2"] ** state["t"])
        / (1 - config["beta1"] ** state["t"])
    )
    for current_var, current_grad in zip(variables, gradients):
        var_first_moment = state["m"].setdefault(var_index, np.zeros_like(current_grad))
        var_second_moment = state["v"].setdefault(
            var_index, np.zeros_like(current_grad)
        )

        np.add(
            config["beta1"] * var_first_moment,
            (1 - config["beta1"]) * current_grad,
            out=var_first_moment,
        )
        np.add(
            config["beta2"] * var_second_moment,
            (1 - config["beta2"]) * current_grad ** 2,
            out=var_second_moment,
        )
        delta = (
            -lr_t * var_first_moment / (np.sqrt(var_second_moment) + config["epsilon"])
        )
        current_var += delta

        assert var_first_moment is state["m"].get(var_index)
        assert var_second_moment is state["v"].get(var_index)
        var_index += 1
