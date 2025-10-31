import numpy as np

def abc_old(spend, a, b, c):
    # revenue = a / (1 + b*(spend**c))
    spend = np.asarray(spend, dtype=float)
    return a / (1.0 + b * np.power(spend, c))

def abc_new(spend, a, b, c):
    # revenue = a / (1 + (spend/b)**c)
    spend = np.asarray(spend, dtype=float)
    return a / (1.0 + np.power(spend / b, c))

EQUATIONS = {
    "abc_old": abc_old,
    "abc_new": abc_new,
}

equation_name_map = {
    "ABC Old (a / [1 + b * spend^c])": "abc_old",
    "ABC New (a / [1 + (spend/b)^c])": "abc_new",
}
