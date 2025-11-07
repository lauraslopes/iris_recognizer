import numpy as np

def hamming_distance(code_query, code_database):
    nm = code_query.shape[0] * code_query.shape[1]
    xor = np.logical_xor(code_query, code_database).sum()
    return xor / float(nm)
    