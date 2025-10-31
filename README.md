# Curve Fitter ABC

Aplicación web en **Streamlit** para ajustar parámetros **A, B, C** de modelos ABC a partir de curvas de respuesta (Spend → Revenue) provistas en un archivo Excel con **múltiples hojas** (una hoja por curva).

## Modelos

- **ABC Old**: `revenue = a / (1 + b * spend^c)`
- **ABC New**: `revenue = a / (1 + (spend/b)^c)`

## Requisitos

- **Anaconda / Miniconda** instalado.
- **Python 3.10** (recomendado).  
- Ver `requirements.txt` para las dependencias.

## Instalación con conda (Anaconda/Miniconda)

> **Sugerencia:** instala primero los paquetes numéricos pesados con conda y luego el resto con pip.

### 1) Crear y activar el entorno

```bash
conda create -n curve-abc python=3.10 -y
conda activate curve-abc
```

### 2) Instalar dependencias base con conda (recomendado)

```bash
conda install -c conda-forge numpy pandas scipy openpyxl xlsxwriter -y
```

### 3) Instalar el resto con pip

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
### 4) Ejecución

```bash
conda activate curve-abc
streamlit run app.py
```

## Arquitectura

```bash
curve-fitter-abc/
├── app.py                       # UI de Streamlit: inputs, botones, orquestación, descargas
├── requirements.txt             # Dependencias para instalación (pip)
├── README.md                    # Este documento
└── src/
    ├── data.py                  # Carga/normalización de Excel → dict{nombre_hoja: DataFrame[x,y]}
    ├── models.py                # Definición de ecuaciones ABC y mapa de nombres
    ├── fitting.py               # Ajuste con lmfit + estrategia iterativa (jitter, reintentos, bounds)
    ├── evaluation.py            # Métricas (R²) y utilidades de evaluación
    └── plotting.py              # Gráficos comparativos con Plotly (original vs. ajustada)
```

## Uso

### 1) Cargar Excel: Un archivo con múltiples hojas. Cada hoja:

- Columna 1: Spend (X)

- Columna 2: Revenue (Y)

Puede tener encabezado; la app toma las dos primeras columnas y hace coerce a numérico.

### 2) Seleccionar ecuación (ABC Old o ABC New).

### 3) (Opcional) Definir valores iniciales A, B, C y límites (bounds).

### 4) Ajustar criterios:

- R² objetivo (por defecto 0.999).

- Iteraciones máximas (por defecto 10).

- Jitter inicial (%) y Random seed.

### 5) Presiona “Ajustar todas las curvas”.

### 6) Revisa la tabla de resultados (A, B, C, R², éxito, etc.).

### 7) Descarga Excel de resultados.

### 8) Usa el desplegable para graficar la curva seleccionada (original vs. ajustada) con Plotly.