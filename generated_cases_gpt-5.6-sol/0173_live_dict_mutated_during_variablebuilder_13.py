# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Live dict mutated during VariableBuilder iteration
# title      : Instance namespace mutation
# sibling    : user object instance __dict__

import threading
import time
import torch

# Sibling construct: user object instance __dict__.
class Box:
    pass


box = Box()
box.stable = 13
for i in range(5000):
    setattr(box, f"field_{i}", i)


def fn(x):
    return x - box.stable


def mutate(add, value):
    if add:
        box.transient = value
    elif hasattr(box, "transient"):
        del box.transient


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


expected = run_while_mutating(lambda: fn(torch.tensor(20)))
compiled = torch.compile(fn, backend="eager")
actual = run_while_mutating(lambda: compiled(torch.tensor(20)))
torch.testing.assert_close(actual, expected)
