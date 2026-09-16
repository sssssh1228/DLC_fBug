# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Live dict mutated during VariableBuilder iteration
# title      : Concurrent class mappingproxy mutation
# sibling    : class namespace mappingproxy

import sys
import threading
import time

import torch

# Sibling under test: a class namespace exposed through types.MappingProxyType.
attributes = {f"item_{i}": i for i in range(512)}
attributes["anchor"] = 7
Namespace = type("Namespace", (), attributes)
namespace = Namespace.__dict__
x = torch.tensor([14.0, 21.0])


def fn(mapping, value):
    return value + mapping["anchor"]


eager = fn(namespace, x)
compiled_fn = torch.compile(fn, backend="eager")
started = threading.Event()
stop = threading.Event()
sys.setswitchinterval(1e-5)


def mutate():
    setattr(Namespace, "volatile", -1)
    started.set()
    while not stop.is_set():
        time.sleep(0.0001)
        delattr(Namespace, "volatile")
        time.sleep(0.0001)
        setattr(Namespace, "volatile", -1)
    if hasattr(Namespace, "volatile"):
        delattr(Namespace, "volatile")


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
