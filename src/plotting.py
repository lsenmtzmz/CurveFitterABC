import numpy as np
import plotly.graph_objects as go

def plot_curve_comparison(x, y, y_hat, title="Curva"):
    x = np.asarray(x)
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Original"))
    fig.add_trace(go.Scatter(x=x, y=y_hat, mode="lines", name="Ajustada"))
    fig.update_layout(
        title=title,
        xaxis_title="Spend (X)",
        yaxis_title="Revenue (Y)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x"
    )
    return fig
