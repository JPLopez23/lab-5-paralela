"""Compara CPU secuencial vs CUDA y guarda resultados (json) en reporte/."""
import json
import time
import numpy as np
from numba import cuda
from suma_secuencial import generar_datos, suma_secuencial
from suma_cuda import suma_cuda, kernel_suma

N = 10_000_000
REPS = 5

a, b = generar_datos(N)

# warm-up (compilacion JIT del kernel, no se cuenta)
suma_cuda(a[:1024].copy(), b[:1024].copy())

t0 = time.perf_counter()
c_cpu = suma_secuencial(a, b)
t_cpu = time.perf_counter() - t0

runs = [suma_cuda(a, b) for _ in range(REPS)]
c_gpu = runs[-1][0]
assert np.allclose(c_cpu, c_gpu) and np.array_equal(c_gpu, a + b), "resultados distintos"

h2d = float(np.median([r[1] for r in runs]))
ker = float(np.median([r[2] for r in runs]))
d2h = float(np.median([r[3] for r in runs]))
total = h2d + ker + d2h

# referencia: NumPy vectorizado (solo informativo)
t0 = time.perf_counter(); a + b; t_np = time.perf_counter() - t0

# barrido de tamanos de bloque
sweep = {}
for th in (32, 64, 128, 256, 512, 1024):
    rs = [suma_cuda(a, b, th) for _ in range(3)]
    sweep[th] = float(np.median([r[2] for r in rs]))

res = dict(n=N, bytes_vector=a.nbytes, bytes_total=a.nbytes * 3, gpu=cuda.get_current_device().name,
           blocks=runs[0][4], t_cpu=t_cpu, t_numpy=t_np, h2d=h2d, kernel=ker, d2h=d2h, t_gpu_total=total,
           speedup=t_cpu / total, speedup_kernel=t_cpu / ker, sweep=sweep)
json.dump(res, open("reporte/resultados.json", "w"), indent=2)
print(json.dumps(res, indent=2))
