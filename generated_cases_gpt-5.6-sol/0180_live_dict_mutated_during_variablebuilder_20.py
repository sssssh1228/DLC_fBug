# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Live dict mutated during VariableBuilder iteration
# title      : Concurrent callable instance namespace
# sibling    : callable user object __dict__ realized through self

import sys
import threading
import time

import torch

# Sibling construct: callable instance attribute dictionary.
sys.setswitchinterval(1e-6)


class Reader:
    def __init__(self):
        self.__dict__.update((f"key_{i}", i) for i in range(20000))
        self.payload = torch.tensor(13.0)

    def __call__(self, x):
        return self.__dict__["payload"] / x


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


reader = Reader()
x = torch.tensor(2.0)
expected = reader(x)
compiled_reader = torch.compile(reader, backend="eager")
actual = invoke_while_mutating(reader.__dict__, lambda: compiled_reader(x))
torch.testing.assert_close(actual, expected)
