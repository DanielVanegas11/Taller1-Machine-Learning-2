from typing import List, Tuple
import numpy as np

def matrix() -> Tuple[dict, np.ndarray]:
    cols = int(input("Ingrese el número de columnas: "))
    rows = int(input("Ingrese el número de filas: "))
    
    mat = np.random.rand(rows, cols)
    eigenvalues, eigenvectors = None, None
    if mat.shape[0] == mat.shape[1]:
        eigenvalues, eigenvectors = np.linalg.eig(mat)
    rank = np.linalg.matrix_rank(mat)
    trace = np.trace(mat)
    det = np.linalg.det(mat) if mat.shape[0] == mat.shape[1] else None
    
    if det is not None and np.isfinite(det) and det != 0:
        invertible = True
        mensaje = "La matriz es invertible"
    elif det is None:
        invertible = False
        mensaje = "La matriz no es cuadrada, no se puede calcular el determinante ni hablar de su invertibilidad"
    else:
        invertible = False
        mensaje = "La matriz no es invertible"
    
    if eigenvalues is not None:
        eigenvalues = np.real(eigenvalues).tolist()
        eigenvectors = np.real(eigenvectors).tolist()
    
    return {
        "Creacion de matrix": f"La matriz tiene {cols} columnas y {rows} filas",
        "Rank": f"El rango de la matriz es {rank}",
        "Traza": f"La traza de la matriz es {trace}",
        "Determinante": f"El determinante de la matriz es {det}" if det is not None else "La matriz no es cuadrada, no se puede calcular el determinante",
        "Invertible": mensaje,
        "Eigenvalues": eigenvalues,
        "Eigenvectors": eigenvectors,
        "Matriz": mat.tolist()
    }, mat

resultados, matriz = matrix()
print(resultados)
print(matriz)