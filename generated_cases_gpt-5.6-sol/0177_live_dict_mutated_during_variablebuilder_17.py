# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Live dict mutated during VariableBuilder iteration
# title      : Concurrent defaultdict closure realization
# sibling    : collections.defaultdict captured in a closure cell

import sys
import threading
import time
from collections import defaultdict

import torch

# Sibling construct: collections.defaultdict captured by a closure.
sys.setswitchinterval(1e-6)
state = defaultdict(int)
state.update((f"key_{i}", i) for i in range(20000))
state["payload"] = torch.tensor(5.0)


def make_fn(mapping):
    def fn(x):
        return mapping["payload"] * x

    return fn


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


fn = make_fn(state)
x = torch.tensor(2.0)
expected = fn(x)
compiled_fn = torch.compile(fn, backend="eager")
actual = invoke_while_mutating(state, lambda: compiled_fn(x))
torch.testing.assert_close(actual, expected)
