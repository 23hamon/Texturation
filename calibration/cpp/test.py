#!/usr/bin/env python3
import numpy as np
import global_refract_loss


def main():
    a = np.zeros((2, 3, 3))
    res = global_refract_loss.test_func(a)
    print(res)


if __name__ == '__main__':
    main()
