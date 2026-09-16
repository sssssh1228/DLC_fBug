# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Iterator objects modeled as UserDefinedObjectVariable lacking iterator protocol
# title      : Filter Iterator Skips Rejected Values
# sibling    : filter iterator predicate-driven __next__

import torch

# Sibling under test: filter iterator predicate-driven __next__.
def fn(it, x):
    first_match = next(iter(it))
    second_match = next(it)
    return x + first_match - second_match


def make_iterator():
    return filter(lambda value: value % 2 == 0, [1, 2, 3, 4, 5])


def main():
    x = torch.tensor(10)
    eager = fn(make_iterator(), x)
    compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
    compiled = compiled_fn(make_iterator(), x)
    torch.testing.assert_close(compiled, eager)


if __name__ == "__main__":
    main()
