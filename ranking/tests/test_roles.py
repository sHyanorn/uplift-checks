import struct

import numpy as np
import pandas as pd
import pytest

from uplift_ranking import roles

M = 2**64


def reference_u(row, salt):
    """Section 2.1 on Python integers modulo 2^64."""
    h = salt
    for j, value in enumerate(row):
        b = struct.unpack("<I", struct.pack("<f", value))[0]
        h ^= (b + j * roles.GOLDEN) % M
        h ^= h >> 30
        h = h * 0xBF58476D1CE4E5B9 % M
        h ^= h >> 27
        h = h * 0x94D049BB133111EB % M
        h ^= h >> 31
    return (h >> 11) / 2**53


def awkward(n, d, seed):
    """float32 features with zeros of both signs, subnormals, extremes and repeated rows."""
    rng = np.random.default_rng(seed)
    special = np.array([0.0, -0.0, 1e-45, -1e-45, 1e-40, 1.1754942e-38, np.finfo(np.float32).max,
                        -np.finfo(np.float32).max, 1.0, -1.0, 12.616365], dtype=np.float32)
    x = rng.normal(scale=10, size=(n, d)).astype(np.float32)
    mask = rng.random((n, d)) < 0.3
    x[mask] = rng.choice(special, mask.sum())
    x[n // 2 :: 7] = x[: len(x[n // 2 :: 7])]
    return x


@pytest.mark.parametrize("salt", [roles.ROLE_SALT, roles.FOLD_SALT + 3])
def test_hash_matches_integer_arithmetic(salt):
    x = awkward(400, 12, 0)
    expected = np.array([reference_u(row.tolist(), salt) for row in x])
    assert np.array_equal(roles.unit_hash(x, salt), expected)


def test_signed_zero_is_a_different_feature_vector():
    x = np.array([[0.0, 1.0], [-0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    u = roles.unit_hash(x, roles.ROLE_SALT)
    assert u[0] == u[2] != u[1]
    assert list(roles.groups(x)) == [0, 1, 0]


def test_identical_feature_vectors_share_role_fold_and_group():
    x = awkward(30000, 8, 1)
    group = roles.groups(x)
    assert len(np.unique(group)) < len(x)
    for labels in (roles.roles(x, "criteo"), roles.roles(x, "hillstrom"), roles.first_fold(x, 2)):
        frame = pd.DataFrame({"group": group, "label": labels})
        assert (frame.groupby("group")["label"].nunique() == 1).all()


def test_groups_are_numbered_by_first_row():
    a, b, c = [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]  # sorted by their bytes: b, a, c
    assert list(roles.groups(np.array([c, b, c, a, b, b], dtype=np.float32))) == [0, 1, 0, 2, 1, 1]


def test_roles_follow_the_thresholds():
    x = np.random.default_rng(3).normal(size=(200000, 12)).astype(np.float32)
    u = roles.unit_hash(x, roles.ROLE_SALT)
    for dataset, (test, validation) in roles.THRESHOLDS.items():
        r = roles.roles(x, dataset)
        assert np.array_equal(r == roles.TEST, u < test)
        assert np.array_equal(r == roles.VALIDATION, (u >= test) & (u < validation))
        assert np.mean(r == roles.TEST) == pytest.approx(test, abs=0.004)
        assert np.mean(r == roles.VALIDATION) == pytest.approx(validation - test, abs=0.004)


def test_folds_depend_on_the_seed():
    x = np.random.default_rng(4).normal(size=(50000, 6)).astype(np.float32)
    f0, f1 = roles.first_fold(x, 0), roles.first_fold(x, 1)
    assert np.array_equal(f0, roles.unit_hash(x, 20261116) < 0.5)
    assert f0.mean() == pytest.approx(0.5, abs=0.01)
    assert 0.45 < np.mean(f0 == f1) < 0.55


def test_csv_text_reads_to_the_float32_of_its_double(tmp_path):
    """pandas' C engine with dtype float32 gives, bit for bit, float32(float(text)), including text just
    above a float32 midpoint, where a direct rounding to float32 would round up."""
    texts = [
        "1.000000059604644775390625",  # midpoint of 1 and the next float32
        "1.0000000596046447753906251",  # just above it
        "1.0000000596046447753906249",  # just below it
        "12.616365", "-0.0", "0", "3.4028234663852886e38", "1.401298464324817e-45", "7.006492321624085e-46",
    ]
    rng = np.random.default_rng(6)
    texts += [f"{v:.6f}" for v in rng.normal(scale=20, size=200)]
    path = tmp_path / "f.csv"
    path.write_text("f0\n" + "\n".join(texts) + "\n")
    read = pd.read_csv(path, dtype={"f0": np.float32}, engine="c")["f0"].to_numpy()
    expected = np.array([np.float32(float(t)) for t in texts])
    assert np.array_equal(read.view(np.uint32), expected.view(np.uint32))
    assert read[1] == np.float32(1.0)  # double rounding; a direct rounding would give the next float32
