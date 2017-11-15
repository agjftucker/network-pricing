import json
import numpy as np

def load_coefficients():
    with open('pseudospectral.json') as f:
        coeff = json.load(f)
    return {k : np.array(v) for k, v in coeff}
