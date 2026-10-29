"""Suma de vectores en GPU con Numba-CUDA, midiendo H2D, kernel y D2H por separado."""
import time
import numpy as np
from numba import cuda


@cuda.jit
def kernel_suma(a, b, c):
    i = cuda.grid(1)
    if i < c.size:
        c[i] = a[i] + b[i]


def suma_cuda(a, b, threads=256):
    n = a.size
    cuda.synchronize()
    t0 = time.perf_counter()
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.device_array_like(a)
    cuda.synchronize()
    t_h2d = time.perf_counter() - t0

    blocks = (n + threads - 1) // threads
    t0 = time.perf_counter()
    kernel_suma[blocks, threads](d_a, d_b, d_c)
    cuda.synchronize()
    t_kernel = time.perf_counter() - t0

    t0 = time.perf_counter()
    c = d_c.copy_to_host()
    cuda.synchronize()
    t_d2h = time.perf_counter() - t0
    return c, t_h2d, t_kernel, t_d2h, blocks
