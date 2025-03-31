#!/usr/bin/env python
from sys import stdin
from functools import partial
import matplotlib.pyplot as plt
print = partial(print, flush=True)
from datetime import datetime


def main():
    start = datetime.now()

    steps = []
    losses = []
    for line in stdin:
        if not "differential_evolution" in line:
            print(line, end="")
            continue

        left, right = line.split("=")
        loss = float(right.strip())
        left = left[len("differential_evolution step "):]
        step = int(left.split(":")[0].strip())

        print(datetime.now() - start, step, loss)
        steps.append(step)
        losses.append(loss)

    plt.plot(steps, losses)
    plt.gca().set_yscale("log")
    plt.show()

if __name__ == '__main__':
    main()
