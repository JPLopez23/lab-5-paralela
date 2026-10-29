"""Genera diagramas y graficas del reporte en reporte/img/ (a partir de reporte/resultados.json)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

r = json.load(open("reporte/resultados.json"))


def flow(path, steps, title):
    fig, ax = plt.subplots(figsize=(5, 1.4 * len(steps)))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2 * len(steps)); ax.axis("off"); ax.set_title(title)
    for k, (kind, txt) in enumerate(steps):
        y = 2 * (len(steps) - k) - 1
        if kind == "dec":
            ax.add_patch(Polygon([[5, y + .9], [9.3, y], [5, y - .9], [.7, y]], fc="#fff3c4", ec="k"))
        elif kind == "io":
            ax.add_patch(Polygon([[1.5, y + .6], [9.5, y + .6], [8.5, y - .6], [.5, y - .6]], fc="#d6ecff", ec="k"))
        else:
            ax.add_patch(FancyBboxPatch((1, y - .6), 8, 1.2, boxstyle="round,pad=0.05" if kind == "box" else "round,pad=0.3",
                                        fc="#d8f5d0" if kind == "box" else "#eeeeee", ec="k"))
        ax.text(5, y, txt, ha="center", va="center", fontsize=8)
        if k < len(steps) - 1:
            ax.annotate("", xy=(5, y - 1.3 if kind != "dec" else y - 1.35), xytext=(5, y - .9 if kind == "dec" else y - .65),
                        arrowprops=dict(arrowstyle="->"))
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


flow("reporte/img/flujo_secuencial.png", [
    ("end", "Inicio"),
    ("io", "Entrada: vectores A, B (N elementos float32)"),
    ("box", "C = vector vacio de tamano N;  i = 0"),
    ("dec", "i < N ?   (No -> salir)"),
    ("box", "C[i] = A[i] + B[i];   i = i + 1\n(vuelve a la decision)"),
    ("io", "Salida: vector C"),
    ("end", "Fin"),
], "Diagrama de flujo - suma secuencial (CPU)")

flow("reporte/img/flujo_cuda.png", [
    ("end", "Inicio (Host)"),
    ("io", "Generar A, B en RAM"),
    ("box", "cuda.to_device(A), cuda.to_device(B)  (H2D)\ndevice_array_like -> d_C"),
    ("box", f"blocks = ceil(N/256) = {r['blocks']}, threads = 256"),
    ("box", "kernel_suma[blocks, threads]\nhilo i = cuda.grid(1)"),
    ("dec", "i < N ?"),
    ("box", "d_C[i] = d_A[i] + d_B[i]\n(si no: el hilo no hace nada)"),
    ("box", "cuda.synchronize(); d_C.copy_to_host()  (D2H)"),
    ("end", "Fin: C en el Host"),
], "Diagrama de flujo - suma en CUDA")

# Mapeo hilos -> elementos
fig, ax = plt.subplots(figsize=(8, 2.6)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4)
for b in range(3):
    ax.add_patch(plt.Rectangle((.3 + b * 3.9, 1.6), 3.6, 1.5, fc="#d6ecff", ec="k"))
    ax.text(2.1 + b * 3.9, 2.85, f"Bloque {b} (256 hilos)", ha="center", fontsize=8)
    for t, lab in enumerate(["hilo 0", "hilo 1", "...", "hilo 255"]):
        ax.add_patch(plt.Rectangle((.5 + b * 3.9 + t * .85, 1.8), .8, .6, fc="#d8f5d0", ec="k"))
        ax.text(.9 + b * 3.9 + t * .85, 2.1, lab, ha="center", va="center", fontsize=6)
    ax.text(2.1 + b * 3.9, 1.2, f"i = {b}*256 + threadIdx.x\n-> C[{b*256}..{b*256+255}]", ha="center", fontsize=7)
ax.set_title("Grid 1D: cada hilo suma un par A[i]+B[i]")
fig.savefig("reporte/img/grid_hilos.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# Tiempos
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].bar(["CPU secuencial", "CUDA total"], [r["t_cpu"], r["t_gpu_total"]], color=["#c0504d", "#4f81bd"])
axs[0].set_ylabel("segundos"); axs[0].set_title(f"Tiempo total (speedup = {r['speedup']:.1f}x)")
parts = [r["h2d"], r["kernel"], r["d2h"]]
axs[1].bar(["H2D", "Kernel", "D2H"], [p * 1e3 for p in parts], color=["#f79646", "#4f81bd", "#9bbb59"])
axs[1].set_ylabel("ms"); axs[1].set_title("Desglose de CUDA")
for i, p in enumerate(parts): axs[1].text(i, p * 1e3, f"{p*1e3:.2f}", ha="center", va="bottom")
fig.tight_layout(); fig.savefig("reporte/img/tiempos.png", dpi=150); plt.close(fig)
