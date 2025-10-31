import numpy as np
from lmfit import Parameters, Minimizer
from .evaluation import r2_score

def _make_params(init_params, bounds):
    p = Parameters()
    for name in ["a", "b", "c"]:
        val = float(init_params.get(name, 1.0))
        (lo, hi) = bounds.get(name, (1e-12, 1e12))
        p.add(name, value=val, min=lo, max=hi)
    return p

def _residuals(params, x, y, model):
    a = params["a"].value
    b = params["b"].value
    c = params["c"].value
    y_hat = model(x, a=a, b=b, c=c)
    return y_hat - y  # residuales (modelo - observado) para least_squares

def _jitter_params(init_params, jitter_pct, rng):
    """Aplica jitter porcentual simétrico a A,B,C."""
    out = {}
    for k, v in init_params.items():
        if v == 0:
            # si 0, usa pequeño valor aleatorio
            scale = 1.0
        else:
            scale = abs(v)
        jitter = 1.0 + (rng.uniform(-jitter_pct, jitter_pct) / 100.0)
        out[k] = max(1e-12, v * jitter) if scale > 0 else v + rng.uniform(1e-6, 1e-3)
    return out

def fit_once(x, y, model, init_params, bounds, method="least_squares", loss="soft_l1"):
    params = _make_params(init_params, bounds)
    # ✅ FIX: el primer argumento posicional es la función de residuales
    minimizer = Minimizer(
        _residuals,
        params,
        fcn_args=(x, y, model),
        nan_policy="omit",
    )
    result = minimizer.minimize(
        method=method,
        loss=loss,
        max_nfev=20000, # ✅ FIX: Fit aborted: number of function evaluations > 8000
        xtol=1e-12,   # más estricto (por defecto ~1e-8)
        ftol=1e-12,   # opcional: tolerancia en la función objetivo
        gtol=1e-12    # opcional: tolerancia en gradiente
    ) 
    a = result.params["a"].value
    b = result.params["b"].value
    c = result.params["c"].value
    y_hat = model(x, a=a, b=b, c=c)
    r2 = r2_score(y, y_hat)
    out = {
        "params": {"a": a, "b": b, "c": c},
        "r2": float(r2),
        "success": bool(result.success),
        "nfev": int(getattr(result, "nfev", 0)),
        "message": str(result.message),
    }
    return out

def fit_curve_iterative(
    x, y, model,
    init_params, bounds,
    target_r2=0.999, max_iter=10, jitter_pct=20, random_seed=42,
    method="least_squares", loss="soft_l1",
):
    rng = np.random.default_rng(int(random_seed))

    # Iteración 0: con los iniciales provistos
    best = fit_once(x, y, model, init_params, bounds, method=method, loss=loss)

    # Si ya cumple, retornamos
    if best["r2"] >= target_r2 and best["success"]:
        return best

    current = dict(init_params)
    for _ in range(1, max_iter):
        # Estrategia: jitter sobre los mejores params hasta ahora
        current = _jitter_params(best["params"], jitter_pct, rng)
        cand = fit_once(x, y, model, current, bounds, method=method, loss=loss)
        if (cand["r2"] > best["r2"]) or (cand["r2"] == best["r2"] and cand["success"] and not best["success"]):
            best = cand
        if best["r2"] >= target_r2 and best["success"]:
            break

    return best
