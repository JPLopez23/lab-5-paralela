"""Suma de vectores C[i] = A[i] + B[i] - version secuencial (CPU, Python puro sobre arreglos NumPy)."""
import time
import numpy as np


def generar_datos(n, seed=42):
    rng = np.random.default_rng(seed)
    return rng.random(n, dtype=np.float32), rng.random(n, dtype=np.float32)


def suma_secuencial(a, b):
    c = np.empty_like(a)
    for i in range(a.shape[0]):
        c[i] = a[i] + b[i]
    return c


if __name__ == "__main__":
    n = 10_000_000
    a, b = generar_datos(n)
    t0 = time.perf_counter()
    c = suma_secuencial(a, b)
    t = time.perf_counter() - t0
    print(f"N={n} tiempo CPU secuencial: {t:.4f} s")
