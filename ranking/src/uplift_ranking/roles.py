"""Roles of rows, resampling groups and folds (Sections 2.1, 5.4 and 5.5).

A row's role depends only on the 32 bits of its float32 features and a salt. Rows with the same
bits share a hash, hence a role, a fold and a resampling group.
"""

import numpy as np

ROLE_SALT = 20261115
FOLD_SALT = 20261116
GOLDEN = 0x9E3779B97F4A7C15
TEST, VALIDATION, TRAINING = 0, 1, 2
# (test below, validation below) for u; Lazada's test rows are its test file.
THRESHOLDS = {"criteo": (0.30, 0.40), "lazada": (0.0, 0.10), "hillstrom": (0.40, 0.50)}


def feature_bits(x):
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)


def unit_hash(x, salt):
    """u = (h >> 11) / 2^53 with h = mix(h XOR (b_j + j * GOLDEN)) over the features j, h starting at the salt."""
    bits = feature_bits(x)
    h = np.full(len(bits), salt, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(bits.shape[1]):
            h ^= bits[:, j].astype(np.uint64) + np.uint64(j) * np.uint64(GOLDEN)
            h ^= h >> np.uint64(30)
            h *= np.uint64(0xBF58476D1CE4E5B9)
            h ^= h >> np.uint64(27)
            h *= np.uint64(0x94D049BB133111EB)
            h ^= h >> np.uint64(31)
    return (h >> np.uint64(11)).astype(np.float64) / 2.0**53


def roles(x, dataset):
    """TEST, VALIDATION or TRAINING per row (for Lazada, of its training file)."""
    test, validation = THRESHOLDS[dataset]
    u = unit_hash(x, ROLE_SALT)
    return np.where(u < test, TEST, np.where(u < validation, VALIDATION, TRAINING)).astype(np.int8)


def first_fold(x, seed):
    """Fold 1 of a fit with seed s: the rows with u < 0.5 under the salt 20261116 + s."""
    return unit_hash(x, FOLD_SALT + seed) < 0.5


def groups(x):
    """Resampling groups: rows with the same feature bits, numbered by their first row.

    Identical feature vectors (Sections 2.1 and 5.5) are compared by their float32 bits, as in the hash,
    so -0.0 and 0.0 differ."""
    bits = feature_bits(x)
    rows = bits.view(np.dtype((np.void, 4 * bits.shape[1]))).ravel()
    _, first, inverse = np.unique(rows, return_index=True, return_inverse=True)
    number = np.empty(len(first), dtype=np.int64)
    number[np.argsort(first)] = np.arange(len(first))
    return number[inverse.ravel()]
