# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Live dict mutated during VariableBuilder iteration
# title      : Ordered dictionary mutation
# sibling    : collections.OrderedDict

import threading
import time
from collections import OrderedDict

import torch

# Sibling construct: insertion-ordered mutable mapping (OrderedDict).
namespace = OrderedDict((f"item_{i}", i) for i in range(4096))
namespace["anchor"] = 7


def fn(mapping):
    return torch.tensor(mapping["anchor"] ** 2)


eager = fn(namespace)
stop = threading.Event()
started = threading.Event()


def mutate():
    while not stop.is_set():
        namespace["dynamo_noise"] = 1
        started.set()
        time.sleep(0.001)
        namespace.pop("dynamo_noise", None)
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
