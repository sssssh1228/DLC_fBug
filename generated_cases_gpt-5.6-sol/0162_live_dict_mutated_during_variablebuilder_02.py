# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Live dict mutated during VariableBuilder iteration
# title      : Concurrent defaultdict mutation
# sibling    : collections.defaultdict

import sys
import threading
import time
from collections import defaultdict

import torch

# Sibling under test: collections.defaultdict mutation through its default factory.
state = defaultdict(int, {f"item_{i}": i for i in range(512)})
state["anchor"] = 4
x = torch.tensor([2.0, 5.0])


def fn(mapping, value):
    return value * mapping["anchor"]


eager = fn(state, x)
compiled_fn = torch.compile(fn, backend="eager")
started = threading.Event()
stop = threading.Event()
sys.setswitchinterval(1e-5)


def mutate():
    state["volatile"] += 1
    started.set()
    while not stop.is_set():
        time.sleep(0.0001)
        state.pop("volatile", None)
        time.sleep(0.0001)
        state["volatile"] += 1
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
