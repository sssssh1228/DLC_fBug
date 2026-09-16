# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Live dict mutated during VariableBuilder iteration
# title      : Concurrent OrderedDict realization
# sibling    : collections.OrderedDict passed as a positional argument

import sys
import threading
import time
from collections import OrderedDict

import torch

# Sibling construct: collections.OrderedDict positional argument.
sys.setswitchinterval(1e-6)
state = OrderedDict((f"key_{i}", i) for i in range(20000))
state["payload"] = torch.tensor(3.0)


def fn(mapping, x):
    return mapping["payload"] + x


def invoke_while_mutating(mapping, callback):
    started = threading.Event()
    stop = threading.Event()

    def mutate():
        value = 0
        while not stop.is_set():
            mapping["transient"] = value
            started.set()
            time.sleep(0)
            mapping.pop("transient", None)
            time.sleep(0)
            value += 1

    worker = threading.Thread(target=mutate, daemon=True)
    worker.start()
    assert started.wait(timeout=2.0)
    try:
        return callback()
    finally:
        stop.set()
        worker.join(timeout=2.0)
        assert not worker.is_alive()


x = torch.tensor(4.0)
expected = fn(state, x)
compiled_fn = torch.compile(fn, backend="eager")
actual = invoke_while_mutating(state, lambda: compiled_fn(state, x))
torch.testing.assert_close(actual, expected)
