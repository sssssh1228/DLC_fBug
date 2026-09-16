# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Live dict mutated during VariableBuilder iteration
# title      : Closure-captured dictionary mutation
# sibling    : closure-captured plain dict

import threading
import time
import torch

# Sibling construct: closure-captured plain dict.
def make_function():
    table = {"stable": 7, **{f"item_{i}": i for i in range(10000)}}

    def fn(x):
        return x + table["stable"]

    def mutate(add, value):
        if add:
            table["transient"] = value
        else:
            table.pop("transient", None)

    return fn, mutate


def run_while_mutating(call, mutate):
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


fn, mutate = make_function()
expected = run_while_mutating(lambda: fn(torch.tensor(5)), mutate)
compiled = torch.compile(fn, backend="eager")
actual = run_while_mutating(lambda: compiled(torch.tensor(5)), mutate)
torch.testing.assert_close(actual, expected)
