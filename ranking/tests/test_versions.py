import pytest

from uplift_ranking import versions


def test_installed_versions_are_the_registered_ones():
    found = versions.check()
    assert {name: found[name] for name in versions.PINNED} == versions.PINNED
    assert found["python"].startswith("3.13.")


@pytest.mark.parametrize("name", sorted(versions.PINNED))
def test_any_other_version_stops(name):
    found = dict(versions.PINNED, **{name: versions.PINNED[name] + ".post1"})
    with pytest.raises(versions.VersionMismatch, match=f"^{name} "):
        versions.check(found, python=versions.PYTHON)


def test_a_cuda_build_of_torch_stops():
    with pytest.raises(versions.VersionMismatch, match="^torch 2.15.0 "):
        versions.check(dict(versions.PINNED, torch="2.15.0"), python=versions.PYTHON)


def test_another_python_stops():
    with pytest.raises(versions.VersionMismatch, match="^Python 3.12 "):
        versions.check(dict(versions.PINNED), python=(3, 12))
