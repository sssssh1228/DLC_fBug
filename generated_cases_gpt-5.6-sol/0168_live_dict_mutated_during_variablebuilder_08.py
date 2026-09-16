# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Live dict mutated during VariableBuilder iteration
# title      : Class mapping proxy mutation
# sibling    : class __dict__ mappingproxy backed by a mutable namespace

import threading
import time

import torch

# Sibling construct: class namespace exposed through a mappingproxy.
class Namespace:
    anchor = 9


for i in range(4096):
    setattr(Namespace, f"item_{i}", i)
namespace = vars(Namespace)


def fn(mapping):
    return torch.tensor(mapping["anchor"] - 4)


eager = fn(namespace)
stop = threading.Event()
started = threading.Event()


def mutate():
    while not stop.is_set():
        setattr(Namespace, "dynamo_noise", 1)
        started.set()
        time.sleep(0.001)
        delattr(Namespace, "dynamo_noise")
        time.sleep(0.001)


thread = threading.Thread(target=mutate, daemon=True)
thread.start()
assert started.wait(timeout=5)
try:
    compiled = torch.compile(fn, backend="eager")
    actual = compiled(namespace)
finally:
    stop.set()
    thread.join(timeout=5)

assert torch.equal(eager, actual), (eager, actual)
