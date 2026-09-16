# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Iterator objects modeled as UserDefinedObjectVariable lacking iterator protocol
# title      : Map Iterator Consecutive Next
# sibling    : map iterator __next__

import torch

# Sibling under test: map iterator __next__.
def fn(it, x):
    first = next(it)
    second = next(it)
    return x * first + second


def make_iterator():
    return map(lambda value: value * value, [2, 3, 4])


def main():
    x = torch.tensor(5)
    eager = fn(make_iterator(), x)
    compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
    compiled = compiled_fn(make_iterator(), x)
    torch.testing.assert_close(compiled, eager)


if __name__ == "__main__":
    main()
