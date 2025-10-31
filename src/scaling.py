import numpy as np

def _safe_max(arr):
    arr = np.asarray(arr, float)
    m = np.nanmax(arr)
    return 1.0 if not np.isfinite(m) or m == 0 else float(m)

def _safe_p95(arr):
    arr = np.asarray(arr, float)
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return 1.0
    p = np.percentile(vals, 95)
    return 1.0 if p == 0 else float(p)

def scale_xy(x, y, method_x: str = "p95", scale_y: bool = True):
    """
    Devuelve x', y', sx, sy según:
      x' = x / sx
      y' = y / sy (si scale_y, si no, sy=1)
    method_x: 'p95' o 'max'
    """
    if method_x not in {"p95", "max"}:
        method_x = "p95"

    if method_x == "p95":
        sx = _safe_p95(x)
    else:
        sx = _safe_max(x)

    if scale_y:
        sy = _safe_max(y)
    else:
        sy = 1.0

    x_s = np.asarray(x, float) / sx
    y_s = np.asarray(y, float) / sy
    return x_s, y_s, float(sx), float(sy)

def scale_params(eq_key: str, a, b, c, sx: float, sy: float):
    """
    Convierte parámetros de escala original -> escala (x', y').
    """
    a_p = a / sy
    if eq_key == "abc_old":
        b_p = b * (sx ** c)
    else:  # abc_new
        b_p = b / sx
    c_p = c
    return {"a": float(a_p), "b": float(b_p), "c": float(c_p)}

def unscale_params(eq_key: str, a_p, b_p, c_p, sx: float, sy: float):
    """
    Convierte parámetros de escala (x', y') -> escala original (x, y).
    """
    a = a_p * sy
    if eq_key == "abc_old":
        b = b_p / (sx ** c_p)
    else:  # abc_new
        b = b_p * sx
    c = c_p
    return {"a": float(a), "b": float(b), "c": float(c)}
