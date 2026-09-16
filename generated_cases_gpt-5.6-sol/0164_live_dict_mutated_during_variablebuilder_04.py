# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Live dict mutated during VariableBuilder iteration
# title      : Concurrent function namespace mutation
# sibling    : function __dict__

import sys
import threading
import time

import torch

# Sibling under test: a Python function's writable attribute dictionary.
def holder():
    return None


for i in range(512):
    setattr(holder, f"item_{i}", i)
holder.anchor = 6
namespace = holder.__dict__
x = torch.tensor([12.0, 18.0])


def fn(mapping, value):
    return value / mapping["anchor"]


eager = fn(namespace, x)
compiled_fn = torch.compile(fn, backend="eager")
started = threading.Event()
stop = threading.Event()
sys.setswitchinterval(1e-5)


def mutate():
    holder.volatile = -1
    started.set()
    while not stop.is_set():
        time.sleep(0.0001)
        del holder.volatile
        time.sleep(0.0001)
        holder.volatile = -1
    if hasattr(holder, "volatile"):
        del holder.volatile


worker = threading.Thread(target=mutate, daemon=True)
worker.start()
started.wait()
try:
    compiled = compiled_fn(namespace, x)
finally:
    stop.set()
    worker.join(timeout=5)

assert not worker.is_alive()
assert torch.equal(eager, compiled), (eager, compiled)
