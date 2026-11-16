"""TARNet, DESCN and EFIN in torch, written from the papers' descriptions (Section 6).

`rank` is h. Every loss is a mean over the rows of a batch. Training uses AdamW, a seeded batch
order and early stopping on the validation factual loss; predictions are made in chunks of 50,000
rows, and EFIN evaluates a chunk in blocks of rows to bound the memory of its attention.

A fit's seed s seeds torch's global generator, which draws the initial weights, and the batch order
is drawn from a torch.Generator seeded with numpy.random.SeedSequence([20261115, 6, s]), so the two
draw from different streams (Sections 6 and 11).

Choices Section 6 leaves open: the validation factual loss is the binary cross-entropy of the
output of each row's arm; every model's effect is sigmoid(treated logit) - sigmoid(control logit);
early stopping restores the weights of the best epoch; a feature with standard deviation 0 is
divided by 1 when inputs are standardised.
"""

import copy
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import expit
from torch import nn
from torch.nn import functional as F

CHUNK = 50_000
THREADS = 4


def mlp(sizes):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ELU()]
    return nn.Sequential(*layers)


def head(h):
    """Two ELU layers of width h/2 and a logit."""
    return nn.Sequential(mlp([h, h // 2, h // 2]), nn.Linear(h // 2, 1))


class TARNet(nn.Module):
    """Shalit et al. (2017): a shared representation of three ELU layers and one head per arm."""

    def __init__(self, d, h):
        super().__init__()
        self.shared = mlp([d, h, h, h])
        self.control, self.treated = head(h), head(h)

    def logits(self, x):
        r = self.shared(x)
        return self.control(r)[:, 0], self.treated(r)[:, 0]

    def loss(self, x, w, y, alpha):
        return factual_loss(*self.logits(x), w, y)


class DESCN(nn.Module):
    """Zhong et al. (2022), Eq. 7: entire-space losses with weight 1 and the cross losses with weight alpha.

    Each row's cross loss is that of its arm, averaged over the batch; the effect uses the two response
    heads, not the propensity (Section 6)."""

    def __init__(self, d, h):
        super().__init__()
        self.shared = mlp([d, h, h, h])
        self.propensity, self.treated, self.control, self.pseudo = head(h), head(h), head(h), head(h)

    def heads(self, x):
        r = self.shared(x)
        return [part(r)[:, 0] for part in (self.propensity, self.treated, self.control, self.pseudo)]

    def logits(self, x):
        _, treated, control, _ = self.heads(x)
        return control, treated

    def loss(self, x, w, y, alpha):
        propensity, treated, control, pseudo = self.heads(x)
        pi, mu1, mu0 = torch.sigmoid(propensity), torch.sigmoid(treated), torch.sigmoid(control)
        entire_space = (
            F.binary_cross_entropy(pi, w)
            + F.binary_cross_entropy(pi * mu1, w * y)
            + F.binary_cross_entropy((1 - pi) * mu0, (1 - w) * y)
        )
        cross = torch.where(
            w > 0,
            F.binary_cross_entropy_with_logits(control + pseudo, y, reduction="none"),
            F.binary_cross_entropy_with_logits(treated - pseudo, y, reduction="none"),
        ).mean()
        return entire_space + alpha * cross


class EFIN(nn.Module):
    """Liu et al. (2023): feature embeddings w_j x_j + b_j, self-attention for the control response,
    attention of a treatment embedding for the uplift, and an intervention constraint on the inverted
    treatment label. One attention head; the MLPs have hidden widths h and h/2.

    Choices beyond the paper's description (Sections 6 and 13): ELU activations; the control MLP takes
    the flattened attention output (d h values), the uplift MLP the treatment-aware representation (h
    values); embedding weights and the treatment embedding start from N(0, 0.1^2), biases from 0; a
    prediction chunk is evaluated in blocks of 4,096 rows."""

    block = 4096

    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.scale, self.shift = nn.Parameter(0.1 * torch.randn(d, h)), nn.Parameter(torch.zeros(d, h))
        self.query, self.key, self.value = (nn.Linear(h, h, bias=False) for _ in range(3))
        self.control = nn.Sequential(mlp([d * h, h, h // 2]), nn.Linear(h // 2, 1))
        self.treatment = nn.Parameter(0.1 * torch.randn(h))
        self.t_query, self.t_key, self.t_value = (nn.Linear(h, h, bias=False) for _ in range(3))
        self.uplift = nn.Sequential(mlp([h, h, h // 2]), nn.Linear(h // 2, 1))
        self.constraint = nn.Linear(h, 1)

    def parts(self, x):
        e = x[:, :, None] * self.scale + self.shift
        q, k = self.query(e), self.key(e)
        weights = torch.softmax(q @ k.transpose(1, 2) / self.h**0.5, dim=-1)
        control = self.control((weights @ self.value(e)).flatten(1))[:, 0]
        t_weights = torch.softmax(self.t_key(e) @ self.t_query(self.treatment) / self.h**0.5, dim=-1)
        aware = (t_weights[:, :, None] * self.t_value(e)).sum(1)
        return control, self.uplift(aware)[:, 0], aware

    def logits(self, x):
        out = [self.parts(x[i : i + self.block])[:2] for i in range(0, len(x), self.block)]
        control, uplift = (torch.cat(z) for z in zip(*out))
        return control, control + uplift

    def loss(self, x, w, y, alpha):
        control, uplift, aware = self.parts(x)
        constraint = F.binary_cross_entropy_with_logits(self.constraint(aware)[:, 0], 1 - w)
        return F.binary_cross_entropy_with_logits(control + w * uplift, y) + alpha * constraint


MODELS = {"TARNet": TARNet, "DESCN": DESCN, "EFIN": EFIN}


def factual_loss(control, treated, w, y):
    return F.binary_cross_entropy_with_logits(torch.where(w > 0, treated, control), y)


@dataclass
class Fitted:
    module: nn.Module
    mean: np.ndarray
    sd: np.ndarray
    epochs: int  # epoch of the lowest validation factual loss, 1-based
    losses: list  # validation factual loss after each epoch

    def inputs(self, x):
        return torch.from_numpy(((np.asarray(x, dtype=np.float32) - self.mean) / self.sd).astype(np.float32))

    @torch.no_grad()
    def logits(self, x, chunk=CHUNK):
        self.module.eval()
        out = [self.module.logits(self.inputs(x[i : i + chunk])) for i in range(0, len(x), chunk)]
        return [torch.cat(z).double().numpy() for z in zip(*out)]

    def effect(self, x, chunk=CHUNK):
        control, treated = self.logits(x, chunk)
        return expit(treated) - expit(control)

    def factual_loss(self, x, w, y):
        control, treated = (torch.from_numpy(z) for z in self.logits(x))
        return float(factual_loss(control, treated, torch.from_numpy(np.asarray(w, dtype=np.float64)), torch.from_numpy(np.asarray(y, dtype=np.float64))))


def batch_order(seed):
    """The generator of a fit's batch order (see the module docstring)."""
    state = np.random.SeedSequence([20261115, 6, seed]).generate_state(1, np.uint64)[0]
    return torch.Generator().manual_seed(int(state))


def configure():
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)


def fit(model, train, valid, *, rank, batch, lr, weight_decay, alpha, standardise, seed, max_epochs=20, patience=5):
    """Fit one deep model. `train` and `valid` are (x, w, y); returns the weights of the best epoch."""
    configure()
    x, w, y = train
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32) if standardise else np.zeros(x.shape[1], np.float32)
    sd = x.std(axis=0, dtype=np.float64).astype(np.float32) if standardise else np.ones(x.shape[1], np.float32)
    sd[sd == 0] = 1
    torch.manual_seed(seed)
    fitted = Fitted(MODELS[model](x.shape[1], rank), mean, sd, 0, [])
    x_t = fitted.inputs(x)
    w_t, y_t = torch.from_numpy(w.astype(np.float32)), torch.from_numpy(y.astype(np.float32))
    optimiser = torch.optim.AdamW(fitted.module.parameters(), lr=lr, weight_decay=weight_decay)
    order = batch_order(seed)
    best = (np.inf, None)
    for epoch in range(1, max_epochs + 1):
        fitted.module.train()
        for rows in torch.randperm(len(x_t), generator=order).split(batch):
            optimiser.zero_grad()
            fitted.module.loss(x_t[rows], w_t[rows], y_t[rows], alpha).backward()
            optimiser.step()
        fitted.losses.append(fitted.factual_loss(*valid))
        if fitted.losses[-1] < best[0]:
            best, fitted.epochs = (fitted.losses[-1], copy.deepcopy(fitted.module.state_dict())), epoch
        elif epoch - fitted.epochs >= patience:
            break
    if best[1] is not None:  # every epoch non-finite: keep the last weights; the R-loss is then inf
        fitted.module.load_state_dict(best[1])
    return fitted
