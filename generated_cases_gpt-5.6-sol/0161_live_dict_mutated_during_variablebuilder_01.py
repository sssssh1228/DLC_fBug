# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Live dict mutated during VariableBuilder iteration
# title      : Concurrent OrderedDict mutation
# sibling    : collections.OrderedDict

import sys
import threading
import time
from collections import OrderedDict

import torch

# Sibling under test: collections.OrderedDict mutation during variable construction.
state = OrderedDict((f"item_{i}", i) for i in range(512))
state["anchor"] = 3
x = torch.tensor([1.0, 2.0])


def fn(mapping, value):
    return value + mapping["anchor"]


eager = fn(state, x)
compiled_fn = torch.compile(fn, backend="eager")
started = threading.Event()
stop = threading.Event()
sys.setswitchinterval(1e-5)


def mutate():
    state["volatile"] = -1
    started.set()
    while not stop.is_set():
        time.sleep(0.0001)
        state.pop("volatile", None)
        time.sleep(0.0001)
        state["volatile"] = -1
    state.pop("volatile", None)


worker = threading.Thread(target=mutate, daemon=True)
worker.start()
started.wait()
try:
    compiled = compiled_fn(state, x)
finally:
    stop.set()
    worker.join(timeout=5)

assert not worker.is_alive()
assert torch.equal(eager, compiled), (eager, compiled)
