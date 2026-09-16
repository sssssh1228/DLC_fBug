# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Live dict mutated during VariableBuilder iteration
# title      : Concurrent function attribute namespace
# sibling    : user-defined function __dict__ accessed from compiled code

import sys
import threading
import time

import torch

# Sibling construct: user-defined function attribute dictionary.
sys.setswitchinterval(1e-6)


def carrier():
    pass


carrier.__dict__.update((f"key_{i}", i) for i in range(20000))
carrier.payload = torch.tensor(7.0)


def fn(x):
    return carrier.__dict__["payload"] + x.square()


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


x = torch.tensor(3.0)
expected = fn(x)
compiled_fn = torch.compile(fn, backend="eager")
actual = invoke_while_mutating(carrier.__dict__, lambda: compiled_fn(x))
torch.testing.assert_close(actual, expected)
