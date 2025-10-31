import pandas as pd
import numpy as np

def _coerce_xy(df: pd.DataFrame) -> pd.DataFrame | None:
    """Intenta extraer las dos primeras columnas numéricas como x,y.
    Ignora filas no numéricas; retorna None si no hay suficientes datos."""
    # Si vienen con encabezados, simplemente tomamos las dos primeras columnas
    # y convertimos a numéricos (coercing), luego dropna.
    if df.shape[1] < 2:
        return None

    # Tomar las dos primeras columnas
    tmp = df.iloc[:, :2].copy()
    tmp.columns = ["x", "y"]
    tmp["x"] = pd.to_numeric(tmp["x"], errors="coerce")
    tmp["y"] = pd.to_numeric(tmp["y"], errors="coerce")
    tmp = tmp.dropna(subset=["x", "y"])
    if tmp.empty:
        return None

    # Ordenar por x por conveniencia
    tmp = tmp.sort_values("x").reset_index(drop=True)
    return tmp

def load_excel_curves(file) -> dict:
    """Lee un Excel (bytes o path) y devuelve dict {sheet_name: DataFrame(x,y)}"""
    xl = pd.ExcelFile(file)
    curves = {}
    for sheet in xl.sheet_names:
        df_raw = xl.parse(sheet_name=sheet, header=0)
        df = _coerce_xy(df_raw)
        if df is not None and len(df) >= 3:
            curves[sheet] = df
    return curves
