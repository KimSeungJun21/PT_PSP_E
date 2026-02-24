import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNormal(nn.Module):
    def __init__(self, in_features, units):
        super(DenseNormal, self).__init__()
        self.units = int(units)
        self.linear = nn.Linear(in_features, 2 * self.units)

    def forward(self, x):
        output = self.linear(x)
        # tf.split(output, 2, axis=-1) -> torch.split(output, self.units, dim=-1)
        mu, logsigma = torch.split(output, self.units, dim=-1)
        sigma = F.softplus(logsigma) + 1e-6
        return torch.cat([mu, sigma], dim=-1)


class DenseNormalGamma(nn.Module):
    def __init__(self, in_features, units):
        super(DenseNormalGamma, self).__init__()
        self.units = int(units)
        self.linear = nn.Linear(in_features, 4 * self.units)

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x):
        output = self.linear(x)
        mu, logv, logalpha, logbeta = torch.split(output, self.units, dim=-1)
        v = self.evidence(logv) + 1e-6
        alpha = self.evidence(logalpha) + 1 + 1e-6
        beta = self.evidence(logbeta) + 1e-6
        return torch.cat([mu, v, alpha, beta], dim=-1)


class DenseDirichlet(nn.Module):
    def __init__(self, in_features, units):
        super(DenseDirichlet, self).__init__()
        self.units = int(units)
        self.linear = nn.Linear(in_features, self.units)

    def forward(self, x):
        output = self.linear(x)
        evidence = torch.exp(output)
        alpha = evidence + 1
        prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
        return torch.cat([alpha, prob], dim=-1)


class DenseSigmoid(nn.Module):
    def __init__(self, in_features, units):
        super(DenseSigmoid, self).__init__()
        self.units = int(units)
        self.linear = nn.Linear(in_features, self.units)

    def forward(self, x):
        logits = self.linear(x)
        prob = torch.sigmoid(logits)
        # Keras 버전처럼 리스트로 반환하거나, 필요에 따라 concat/stack 하세요.
        return logits, prob