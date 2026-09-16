# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Missing or incorrect BuiltinVariable handler for Python builtin semantics
# title      : Callable sentinel iterator
# sibling    : two-argument iter(callable, sentinel) termination semantics

import torch

# Sibling under test: iter(callable, sentinel) stopping before the sentinel.
def fn(x):
    values = [x + 1, x + 2, x + 3, None]
    position = 0

    def produce():
        nonlocal position
        value = values[position]
        position += 1
        return value

    total = torch.zeros_like(x)
    for value in iter(produce, None):
        total = total + value
    return total, position


x = torch.tensor([2.0, -1.0])
eager = fn(x)
compiled = torch.compile(fn, backend="eager", fullgraph=True)(x)
assert torch.equal(compiled[0], eager[0]), (compiled, eager)
assert compiled[1] == eager[1], (compiled, eager)
