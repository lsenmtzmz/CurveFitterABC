import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data import load_excel_curves
from src.models import EQUATIONS, equation_name_map
from src.fitting import fit_curve_iterative
from src.plotting import plot_curve_comparison

st.set_page_config(page_title="Curve Fitter ABC", layout="wide")

st.title("Curve Fitter ABC")
st.caption("Ajuste de parámetros A, B, C para modelos ABC (Old/New) con múltiples curvas.")

# ---------------------------
# Estado de sesión (persistencia)
# ---------------------------
if "curves" not in st.session_state:
    st.session_state.curves = None
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "fitted_series" not in st.session_state:
    st.session_state.fitted_series = None
if "curve_to_plot" not in st.session_state:
    st.session_state.curve_to_plot = None

# ---------------------------
# Cache de carga de Excel
# ---------------------------
@st.cache_data
def load_excel_curves_cached(file_bytes: bytes) -> dict:
    """Carga y normaliza el Excel desde bytes y devuelve dict {sheet: DataFrame(x,y)}."""
    # Acepta bytes para que sea cacheable; internamente pandas maneja el objeto.
    return load_excel_curves(io.BytesIO(file_bytes))

# ---------------------------
# Sidebar: configuración
# ---------------------------
with st.sidebar:
    st.header("Configuración")

    eq_display = st.selectbox(
        "Ecuación a utilizar",
        options=list(equation_name_map.keys()),
        index=0,
        help="Selecciona la forma funcional del modelo ABC.",
        key="eq_select",
    )
    eq_key = equation_name_map[eq_display]  # "abc_old" o "abc_new"
    model = EQUATIONS[eq_key]

    st.subheader("Parámetros iniciales (opcional)")
    a0 = st.number_input("A inicial", value=1.0, step=0.1, format="%.2f", key="a0")
    b0 = st.number_input("B inicial", value=1.0, step=0.1, format="%.2f", key="b0")
    # ✅ Opción 1: defaults negativos para C
    c0 = st.number_input("C inicial", value=-1.0, step=0.1, format="%.2f", key="c0")

    st.subheader("Límites (bounds)")
    a_min = st.number_input("A min", value=1e-12, step=0.1, format="%.2f", key="a_min")
    a_max = st.number_input("A max", value=1e15, step=0.1, format="%.2f", key="a_max")
    b_min = st.number_input("B min", value=1e-12, step=0.1, format="%.2f", key="b_min")
    b_max = st.number_input("B max", value=1e15, step=0.1, format="%.2f", key="b_max")
    # ✅ Opción 1: bounds negativos para C
    c_min = st.number_input("C min", value=-10.0, step=0.1, format="%.2f", key="c_min")
    c_max = st.number_input("C max", value=-0.3, step=0.1, format="%.2f", key="c_max")

    st.subheader("Criterios de ajuste")
    target_r2 = st.number_input(
        "R² objetivo", value=0.999, min_value=0.0, max_value=0.999999, step=0.0001, format="%.6f", key="target_r2"
    )
    max_iter = st.slider("Iteraciones máximas", min_value=1, max_value=20, value=10, help="Número máximo de reintentos por curva.", key="max_iter")
    jitter_pct = st.slider("Jitter inicial (%)", min_value=0, max_value=50, value=20, help="Magnitud de aleatoriedad para reintentos.", key="jitter_pct")
    random_seed = st.number_input("Random seed", value=42, step=1, help="Para reproducibilidad.", key="seed")

# ---------------------------
# 1) Carga de archivo Excel
# ---------------------------
st.markdown("### 1) Carga de archivo Excel")
uploaded = st.file_uploader(
    "Sube un Excel con múltiples hojas. Cada hoja debe tener en la primera columna el **Spend (X)** y en la segunda el **Revenue (Y)**. Puede tener encabezados.",
    type=["xlsx", "xls"],
    key="uploader",
)

if uploaded is not None:
    try:
        curves = load_excel_curves_cached(uploaded.getvalue())
        st.session_state.curves = curves  # ✅ persistir
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
        st.stop()

    if not st.session_state.curves:
        st.error("No se encontraron hojas válidas con dos columnas numéricas.")
        st.stop()

    st.success(f"Se cargaron {len(st.session_state.curves)} hoja(s).")
    st.write("Hojas detectadas:", list(st.session_state.curves.keys()))
else:
    st.info("Sube un archivo Excel para comenzar.")

# ---------------------------
# 2) Ajuste
# ---------------------------
st.markdown("### 2) Ejecutar ajuste")
run_fit = st.button("Ajustar todas las curvas", key="fit_all_btn")

if run_fit and st.session_state.curves is not None:
    results_rows = []
    fitted_series = {}

    for curve_name, df in st.session_state.curves.items():
        x = df["x"].to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=float)

        fit_out = fit_curve_iterative(
            x=x,
            y=y,
            model=model,
            init_params={"a": a0, "b": b0, "c": c0},
            bounds={"a": (a_min, a_max), "b": (b_min, b_max), "c": (c_min, c_max)},
            target_r2=target_r2,
            max_iter=max_iter,
            jitter_pct=jitter_pct,
            random_seed=int(random_seed),
            method="least_squares",
            loss="soft_l1",
        )

        # ---------------------------
        # Aceptar éxito por R² objetivo
        # ---------------------------
        if fit_out["r2"] >= target_r2:
            fit_out["success"] = True
            fit_out["message"] = f"Objetivo alcanzado R² ({fit_out['r2']:.6f})."
        
        elif fit_out["message"] == "`xtol` termination condition is satisfied.":
            fit_out["success"] = False
            fit_out["message"] = "El optimizador convergio sin alcanzar el Objetivo R²"
            
        elif fit_out["message"] == "`ftol` termination condition is satisfied.":
            fit_out["success"] = False
            fit_out["message"] = "El optimizador no convergio"

        elif fit_out["message"] == "Both `ftol` and `xtol` termination conditions are satisfied.":
            fit_out["success"] = False
            fit_out["message"] = "El optimizador no convergio"

        elif fit_out["message"].find("Fit aborted")!=-1:
            fit_out["message"] = "Abortado: Limite de evaluaciones permitido"
            
        results_rows.append({
            "curve": curve_name,
            "equation": eq_display,
            "A": fit_out["params"]["a"],
            "B": fit_out["params"]["b"],
            "C": fit_out["params"]["c"],
            "R2": fit_out["r2"],
            "success": fit_out["success"],
            "nfev": fit_out["nfev"],
            "message": fit_out["message"],
        })

        y_hat = model(x, **fit_out["params"])
        fitted_series[curve_name] = pd.DataFrame({"x": x, "y": y, "y_hat": y_hat}).sort_values("x")

    results_df = pd.DataFrame(results_rows).sort_values(["success", "R2", "curve"], ascending=[False, False, True])
    
    # ------- Cálculo de B′ por ecuación -------
    if eq_key == "abc_old":
        # B' = B^(-1/C)
        bprime = np.power(results_df["B"].astype(float), -1.0 / results_df["C"].astype(float))
    else:  # abc_new
        # B' = 1 / B^C
        bprime = 1.0 / np.power(results_df["B"].astype(float), results_df["C"].astype(float))

    results_df.insert(results_df.columns.get_loc("B") + 1, "B'", bprime)

    # ✅ Persistir resultados y series
    st.session_state.results_df = results_df
    st.session_state.fitted_series = fitted_series

    # ✅ Selección por defecto para el gráfico
    if not st.session_state.curve_to_plot and len(fitted_series) > 0:
        st.session_state.curve_to_plot = list(fitted_series.keys())[0]

# ---------------------------
# 3) Resultados (persistentes)
# ---------------------------
if st.session_state.results_df is not None:
    st.markdown("### 3) Resultados")
    st.dataframe(st.session_state.results_df, use_container_width=True)

    st.markdown("#### Descargar resultados (Excel)")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        st.session_state.results_df.to_excel(writer, sheet_name="results", index=False)
    st.download_button(
        label="Descargar resultados.xlsx",
        data=buffer.getvalue(),
        file_name="abc_fit_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_results_btn",
    )

    # ---------------------------
    # 4) Visualización
    # ---------------------------
    st.markdown("### 4) Visualización")

    options = list(st.session_state.fitted_series.keys())
    # índice que respeta la última selección
    if st.session_state.curve_to_plot in options:
        default_index = options.index(st.session_state.curve_to_plot)
    else:
        default_index = 0

    curve_to_plot = st.selectbox(
        "Selecciona una curva para graficar",
        options=options,
        index=default_index,
        key="curve_to_plot",  # ✅ mantiene selección
    )

    dfp = st.session_state.fitted_series[curve_to_plot]
    row = st.session_state.results_df.loc[st.session_state.results_df["curve"] == curve_to_plot].iloc[0]
    params_str = f"A={row['A']:.6g}, B={row['B']:.6g}, C={row['C']:.6g}, R²={row['R2']:.6f}"
    fig = plot_curve_comparison(
        x=dfp["x"].values,
        y=dfp["y"].values,
        y_hat=dfp["y_hat"].values,
        title=f"{curve_to_plot} — {row['equation']} ({params_str})"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Aún no hay resultados. Sube un Excel y presiona **Ajustar todas las curvas**.")
