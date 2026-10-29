# Lab 5 – Suma de vectores con CUDA (Numba-CUDA)

- `suma_secuencial.py`: versión secuencial en CPU.
- `suma_cuda.py`: kernel CUDA y medición de H2D / kernel / D2H.
- `benchmark.py`: compara ambas y guarda `reporte/resultados.json`.
- `generar_figuras.py`: diagramas y gráficas del reporte.

Instalación: `pip install "numba-cuda[cu13]" numpy matplotlib`. Ejecutar: `python benchmark.py`.
