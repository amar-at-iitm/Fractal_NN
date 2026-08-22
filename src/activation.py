import numpy as np
from src.alpha_fractal_function import (
    alpha_fractalize_second_derivative,
    alpha_fractalize_first_derivative,
    alpha_fractalize,
    pointwise_fractal
)

# Global cache to hold pre-computed fractal activation functions and their derivatives
FRACTAL_CACHE = {}

def set_fractal_activation(store, store_d):
    """Sets the stored fractal dictionary and its derivative for 'alpha_fractal' activation."""
    global FRACTAL_CACHE
    FRACTAL_CACHE["f"] = store
    FRACTAL_CACHE["f_d"] = store_d

def check_and_auto_set_fractal_cache():
    """Auto-populates FRACTAL_CACHE from __main__ if store and store_d are defined in global notebook scope."""
    if "f" not in FRACTAL_CACHE:
        import sys
        main_mod = sys.modules.get('__main__')
        if hasattr(main_mod, 'store') and hasattr(main_mod, 'store_d'):
            set_fractal_activation(getattr(main_mod, 'store'), getattr(main_mod, 'store_d'))

def activation_fn(z, fn):
    """Applies the activation function."""
    if isinstance(fn, tuple):  # (store, store_d)
        store = fn[0]
        p = store["partition"]
        z_clipped = np.clip(z, p[0], p[-1])
        return pointwise_fractal(z_clipped, store)
    elif isinstance(fn, dict):  # direct store dictionary
        p = fn["partition"]
        z_clipped = np.clip(z, p[0], p[-1])
        return pointwise_fractal(z_clipped, fn)
    elif isinstance(fn, str):
        fn_lower = fn.lower()
        if fn_lower == "relu":
            return np.maximum(0, z)
        elif fn_lower == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif fn_lower == "tanh":
            return np.tanh(z)
        elif fn_lower == "softmax":
            return softmax(z) 
        elif fn_lower == "alpha_fractal": 
            check_and_auto_set_fractal_cache()
            if "f" not in FRACTAL_CACHE:
                raise ValueError("Fractal function not set. Please run cell defining 'store' and 'store_d' or call set_fractal_activation(store, store_d) first.")
            store = FRACTAL_CACHE["f"]
            p = store["partition"]
            z_clipped = np.clip(z, p[0], p[-1])
            return pointwise_fractal(z_clipped, store)
    
    raise ValueError(f"Unsupported activation function: {fn}")
    

def activation_derivative(z, fn):
    """Computes the derivative of the activation function."""
    if isinstance(fn, tuple):  # (store, store_d)
        store_d = fn[1]
        p = store_d["partition"]
        z_clipped = np.clip(z, p[0], p[-1])
        return pointwise_fractal(z_clipped, store_d)
    elif isinstance(fn, dict):  # direct store_d dictionary
        p = fn["partition"]
        z_clipped = np.clip(z, p[0], p[-1])
        return pointwise_fractal(z_clipped, fn)
    elif isinstance(fn, str):
        fn_lower = fn.lower()
        if fn_lower == "relu":
            return (z > 0).astype(float)
        elif fn_lower == "sigmoid":
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)
        elif fn_lower == "tanh":
            return 1 - np.tanh(z) ** 2
        elif fn_lower == "alpha_fractal":
            check_and_auto_set_fractal_cache()
            if "f_d" not in FRACTAL_CACHE:
                raise ValueError("Fractal derivative not set. Please run cell defining 'store' and 'store_d' or call set_fractal_activation(store, store_d) first.")
            store_d = FRACTAL_CACHE["f_d"]
            p = store_d["partition"]
            z_clipped = np.clip(z, p[0], p[-1])
            return pointwise_fractal(z_clipped, store_d)
    
    raise ValueError(f"Unsupported activation function: {fn}")


def softmax(z):
    """Applies the softmax function."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stability trick
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


