# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Live dict mutated during VariableBuilder iteration
# title      : Concurrent mapping proxy realization
# sibling    : types.MappingProxyType live view over a mutable backing dict

import sys
import threading
import time
from types import MappingProxyType

import torch

# Sibling construct: MappingProxyType live view of a mutable dict.
sys.setswitchinterval(1e-6)
backing = {f"key_{i}": i for i in range(20000)}
backing["payload"] = torch.tensor(11.0)
view = MappingProxyType(backing)


def fn(x, *, options):
    return options["payload"] - x


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
expected = fn(x, options=view)
compiled_fn = torch.compile(fn, backend="eager")
actual = invoke_while_mutating(
    backing, lambda: compiled_fn(x, options=view)
)
torch.testing.assert_close(actual, expected)
