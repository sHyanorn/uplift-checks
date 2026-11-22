import numpy as np
import pytest
import torch
from scipy.special import expit

from uplift_ranking import deep, learners
from uplift_ranking.synthetic import rct

SETTINGS = dict(rank=32, batch=256, lr=3e-3, weight_decay=1e-4, alpha=0.5, standardise=True, max_epochs=4)


@pytest.fixture(scope="module")
def data():
    train, valid = rct(20000, 1, treated=0.3), rct(4000, 2, treated=0.3)
    return (train.x, train.w, train.y), (valid.x, valid.w, valid.y), valid.tau


def bce(p, t):
    return -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))


def batch(n=64, d=5, seed=0):
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.normal(size=(n, d)).astype(np.float32))
    return x, torch.from_numpy(rng.integers(0, 2, n).astype(np.float32)), torch.from_numpy(rng.integers(0, 2, n).astype(np.float32))


def test_descn_loss_is_the_sum_of_its_terms():
    torch.manual_seed(0)
    model = deep.DESCN(5, 16)
    x, w, y = batch()
    lp, l1, l0, tau = (z.detach().double().numpy() for z in model.heads(x))
    w_, y_ = w.double().numpy(), y.double().numpy()
    pi, mu1, mu0 = expit(lp), expit(l1), expit(l0)
    cross = np.where(w_ > 0, -(y_ * np.log(expit(l0 + tau)) + (1 - y_) * np.log(1 - expit(l0 + tau))),
                     -(y_ * np.log(expit(l1 - tau)) + (1 - y_) * np.log(1 - expit(l1 - tau)))).mean()
    expected = bce(pi, w_) + bce(pi * mu1, w_ * y_) + bce((1 - pi) * mu0, (1 - w_) * y_) + 0.3 * cross
    assert model.loss(x, w, y, 0.3).item() == pytest.approx(expected, rel=1e-5)


def test_efin_constraint_is_trained_against_the_inverted_treatment():
    torch.manual_seed(0)
    model = deep.EFIN(5, 16)
    with torch.no_grad():
        model.constraint.bias.fill_(2.0)  # predictions far from 1/2, so the two labels give different losses
    x, w, y = batch()
    _, _, aware = model.parts(x)
    logit = model.constraint(aware)[:, 0].detach().double().numpy()
    extra = (model.loss(x, w, y, 0.7) - model.loss(x, w, y, 0.0)).item()
    assert extra == pytest.approx(0.7 * bce(expit(logit), 1 - w.double().numpy()), rel=1e-4)
    assert extra != pytest.approx(0.7 * bce(expit(logit), w.double().numpy()), rel=1e-2)


def test_tarnet_factual_loss_uses_the_head_of_the_observed_arm():
    torch.manual_seed(0)
    model = deep.TARNet(5, 16)
    x, w, y = batch()
    control, treated = (z.detach().double().numpy() for z in model.logits(x))
    expected = bce(expit(np.where(w.numpy() > 0, treated, control)), y.double().numpy())
    assert model.loss(x, w, y, 0.0).item() == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("model", list(deep.MODELS))
def test_effects_track_the_true_effect_and_repeat_exactly(model, data):
    train, valid, tau = data
    first = deep.fit(model, train, valid, seed=0, **SETTINGS)
    effect = first.effect(valid[0])
    assert np.corrcoef(effect, tau)[0, 1] > 0.5
    assert np.array_equal(effect, deep.fit(model, train, valid, seed=0, **SETTINGS).effect(valid[0]))
    assert not np.array_equal(effect, deep.fit(model, train, valid, seed=1, **SETTINGS).effect(valid[0]))
    assert np.allclose(effect, first.effect(valid[0], chunk=777), rtol=0, atol=1e-6)
    assert first.losses[first.epochs - 1] == min(first.losses)
    assert first.factual_loss(*valid) == pytest.approx(min(first.losses), rel=1e-12)


def test_initial_weights_and_batch_order_draw_from_different_streams():
    for seed in range(5):
        init = torch.Generator().manual_seed(seed)  # the stream torch.manual_seed(seed) gives the initial weights
        assert not torch.equal(torch.randperm(1000, generator=deep.batch_order(seed)), torch.randperm(1000, generator=init))


def test_efin_blocks_do_not_change_predictions(data):
    train, valid, _ = data
    fitted = deep.fit("EFIN", train, valid, seed=0, **dict(SETTINGS, max_epochs=1))
    effect = fitted.effect(valid[0])
    fitted.module.block = 333
    assert np.allclose(effect, fitted.effect(valid[0]), rtol=0, atol=1e-6)


def test_early_stopping_waits_five_epochs_and_restores_the_best(monkeypatch, data):
    train, valid, _ = data
    losses, states = iter([0.5, 0.4, 0.45, 0.41, 0.42, 0.43, 0.44, 0.39, 0.3]), []

    def loss(self, *args):
        states.append({k: v.clone() for k, v in self.module.state_dict().items()})
        return next(losses)

    monkeypatch.setattr(deep.Fitted, "factual_loss", loss)
    fitted = deep.fit("TARNet", train, valid, seed=0, **dict(SETTINGS, max_epochs=20, batch=4096))
    assert fitted.losses == [0.5, 0.4, 0.45, 0.41, 0.42, 0.43, 0.44] and fitted.epochs == 2
    final = fitted.module.state_dict()
    assert all(torch.equal(final[k], states[1][k]) for k in final)
    assert not all(torch.equal(final[k], states[-1][k]) for k in final)


def test_standardisation_uses_training_moments(data):
    train, valid, _ = data
    fitted = deep.fit("TARNet", train, valid, seed=0, **dict(SETTINGS, max_epochs=1, batch=4096))
    assert np.allclose(fitted.mean, train[0].mean(axis=0), atol=1e-6)
    assert np.allclose(fitted.inputs(train[0]).numpy().std(axis=0), 1, atol=1e-3)
    raw = deep.fit("TARNet", train, valid, seed=0, **dict(SETTINGS, max_epochs=1, batch=4096, standardise=False))
    assert np.array_equal(raw.inputs(valid[0]).numpy(), valid[0])


@pytest.mark.parametrize("model", list(deep.MODELS))
def test_a_diverged_fit_ends_with_r_loss_inf(model, data):
    """Section 11: a non-finite validation prediction is recorded as R-loss inf, whatever the model."""
    train, valid, _ = data
    fitted = deep.fit(model, train, valid, seed=0, **dict(SETTINGS, lr=1e30, max_epochs=2))
    assert all(np.isnan(loss) for loss in fitted.losses)
    assert learners.r_loss(fitted.effect(valid[0]), valid[2], valid[1], 0.2, 0.3) == float("inf")
