# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Iterator objects modeled as UserDefinedObjectVariable lacking iterator protocol
# title      : Range Iterator With Next Default
# sibling    : range_iterator and two-argument next

import torch

# Sibling under test: range_iterator and two-argument next.
def fn(it, x):
    values = [next(it, -1), next(it, -1), next(it, -1)]
    return x + values[0] + values[1] * 2 + values[2] * 3


def main():
    x = torch.tensor(4)
    eager = fn(iter(range(6, 10)), x)
    compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
    compiled = compiled_fn(iter(range(6, 10)), x)
    torch.testing.assert_close(compiled, eager)


if __name__ == "__main__":
    main()
