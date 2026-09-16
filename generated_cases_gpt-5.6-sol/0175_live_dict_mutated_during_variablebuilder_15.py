# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Live dict mutated during VariableBuilder iteration
# title      : OrderedDict mutation
# sibling    : collections.OrderedDict mutable mapping

import collections
import threading
import time
import torch

# Sibling construct: collections.OrderedDict mutable mapping.
table = collections.OrderedDict((f"item_{i}", i) for i in range(10000))
table["stable"] = 19


def fn(x):
    return x / table["stable"]


def mutate(add, value):
    if add:
        table["transient"] = value
    else:
        table.pop("transient", None)


def run_while_mutating(call):
    stop = threading.Event()
    ready = threading.Event()

    def worker():
        value = 0
        while not stop.is_set():
            mutate(True, value)
            ready.set()
            time.sleep(0)
            mutate(False, value)
            time.sleep(0)
            value += 1

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    assert ready.wait(timeout=5)
    try:
        return call()
    finally:
        stop.set()
        thread.join(timeout=5)
        assert not thread.is_alive()


expected = run_while_mutating(lambda: fn(torch.tensor(38.0)))
compiled = torch.compile(fn, backend="eager")
actual = run_while_mutating(lambda: compiled(torch.tensor(38.0)))
torch.testing.assert_close(actual, expected)
