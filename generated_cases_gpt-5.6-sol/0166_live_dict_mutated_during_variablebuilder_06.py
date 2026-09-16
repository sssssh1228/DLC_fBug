# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Live dict mutated during VariableBuilder iteration
# title      : Function attribute namespace mutation
# sibling    : function object __dict__

import threading
import time

import torch

# Sibling construct: function object attribute namespace (__dict__).
def holder():
    pass


holder.anchor = 41
for i in range(4096):
    setattr(holder, f"item_{i}", i)
namespace = holder.__dict__


def fn(mapping):
    return torch.tensor(mapping["anchor"]) + 1


eager = fn(namespace)
stop = threading.Event()
started = threading.Event()


def mutate():
    while not stop.is_set():
        holder.dynamo_noise = 1
        started.set()
        time.sleep(0.001)
        del holder.dynamo_noise
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
