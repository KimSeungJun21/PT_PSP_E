import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def MSE(y, y_, reduce=True):
    # 첫 번째 차원(배치)을 제외한 나머지 차원들에 대해 평균
    mse = torch.mean((y - y_)**2, dim=list(range(1, len(y.shape))))
    return torch.mean(mse) if reduce else mse

def RMSE(y, y_):
    return torch.sqrt(torch.mean((y - y_)**2))

def Gaussian_NLL(y, mu, sigma, reduce=True):
    # log(sigma) + 0.5*log(2*pi) + 0.5*((y-mu)/sigma)**2
    logprob = -torch.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * ((y - mu) / sigma)**2
    loss = torch.mean(-logprob, dim=list(range(1, len(y.shape))))
    return torch.mean(loss) if reduce else loss

def Gaussian_NLL_logvar(y, mu, logvar, reduce=True):
    # logvar를 사용하여 수치적 안정성을 높인 NLL
    log_likelihood = 0.5 * (
        -torch.exp(-logvar) * (mu - y)**2 - np.log(2 * np.pi) - logvar
    )
    loss = torch.mean(-log_likelihood, dim=list(range(1, len(y.shape))))
    return torch.mean(loss) if reduce else loss

def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    twoBlambda = 2 * beta * (1 + v)

    nll = 0.5 * torch.log(np.pi / v) \
        - alpha * torch.log(twoBlambda) \
        + (alpha + 0.5) * torch.log(v * (y - gamma)**2 + twoBlambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)

    return torch.mean(nll) if reduce else nll

def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    # 두 Normal-Gamma 분포 사이의 KL Divergence
    KL = 0.5 * (a1 - 1) / b1 * (v2 * (mu2 - mu1)**2) \
        + 0.5 * v2 / v1 \
        - 0.5 * torch.log(torch.abs(v2) / torch.abs(v1)) \
        - 0.5 + a2 * torch.log(b1 / b2) \
        - (torch.lgamma(a1) - torch.lgamma(a2)) \
        + (a1 - a2) * torch.digamma(a1) \
        - (b1 - b2) * a1 / b1
    return KL

# ✅ 수정됨: reduce(bool) 대신 reduction(str)을 받도록 변경하여 'none' 지원
def NIG_NLL(y, gamma, v, alpha, beta, reduction='mean'):
    twoBlambda = 2 * beta * (1 + v)

    nll = 0.5 * torch.log(np.pi / v) \
        - alpha * torch.log(twoBlambda) \
        + (alpha + 0.5) * torch.log(v * (y - gamma)**2 + twoBlambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)

    if reduction == 'mean':
        return torch.mean(nll)
    elif reduction == 'sum':
        return torch.sum(nll)
    else: # 'none'
        return nll

# ✅ 수정됨: reduction(str) 지원 추가
def NIG_Reg(y, gamma, v, alpha, beta, omega=0.01, reduction='mean', kl=False):
    error = torch.abs(y - gamma)

    if kl:
        # 특정 타겟 분포와의 KL를 규제항으로 사용
        kl_val = KL_NIG(gamma, v, alpha, beta, gamma, torch.tensor(omega).to(y.device), 1 + omega, beta)
        reg = error * kl_val
    else:
        # Evidence 기반의 규제항
        evi = 2 * v + alpha
        reg = error * evi

    if reduction == 'mean':
        return torch.mean(reg)
    elif reduction == 'sum':
        return torch.sum(reg)
    else: # 'none'
        return reg

# ✅ 수정됨: 가중치(pos_weight) 적용 버전으로 통합
def EvidentialRegressionLoss(y_true, evidential_output, coeff=1.0, pos_weight=1.0):
    """
    Args:
        y_true: 실제 라벨 (B, N) 또는 (B, N, 1)
        evidential_output: 모델 출력 (B, N, 4) -> [gamma, v, alpha, beta]
        coeff: Regularization 항의 계수
        pos_weight: 0이 아닌(유효한 흡착) 데이터에 부여할 가중치. 
                    (1.0이면 가중치 없음, >1.0이면 0이 아닌 데이터 중요도 증가)
    """
    # 이전에 만든 DenseNormalGamma 레이어의 출력을 4개로 쪼개기
    units = evidential_output.shape[-1] // 4
    gamma, v, alpha, beta = torch.split(evidential_output, units, dim=-1)
    
    # 차원 맞추기 (B, N) -> (B, N, 1)
    if y_true.dim() == evidential_output.dim() - 1:
        y_true = y_true.unsqueeze(-1)

    # 1. 개별 포인트별 Loss 계산 (Reduction='none'으로 호출)
    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta, reduction='none')
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta, reduction='none')
    
    total_pointwise_loss = loss_nll + coeff * loss_reg

    # 2. 가중치 마스크 생성 (Data Imbalance 해결)
    # y_true > 1e-4 (부동소수점 고려 0초과) 인 지점에 pos_weight 부여
    weight_mask = torch.ones_like(y_true)
    if pos_weight != 1.0:
        weight_mask[y_true > 1e-4] = pos_weight
    
    # 3. 가중치 적용 후 평균 계산
    weighted_loss = (total_pointwise_loss * weight_mask).mean()
    
    return weighted_loss

def kl_beta_to_uniform(alpha_beta, eps=1e-8):
    """
    KL( Beta(alpha, beta) || Beta(1, 1) )

    Args:
        alpha_beta (torch.Tensor): (B, 2) 텐서. [:, 0]는 alpha, [:, 1]는 beta
    """
    # 안정성을 위해 작은 값 추가
    alpha_beta = alpha_beta + eps
    
    # alpha와 beta 분리
    alpha = alpha_beta[:, 0].unsqueeze(1) # (B, 1)
    beta = alpha_beta[:, 1].unsqueeze(1)  # (B, 1)
    
    # alpha0 = alpha + beta (K=2)
    alpha0 = alpha_beta.sum(dim=1, keepdim=True)  # (B, 1)

    # 1. log Beta Function 계산 (log B(alpha, beta))
    # log B(alpha, beta) = lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)
    logB_alpha_beta = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha0)
    logB_alpha_beta = logB_alpha_beta.squeeze(1) # (B,)

    # 2. 다이감마 함수 (Digamma function, ψ)를 사용한 KL 발산 계산
    # KL = logB(1) - logB(alpha, beta) + Σ (α_k - 1)(ψ(α_k) - ψ(α0))

    # 첫 번째 항: logB(1) - logB(alpha, beta)
    kl_part1 = -logB_alpha_beta # (B,)

    # 두 번째 항: (α - 1)(ψ(α) - ψ(α+β)) + (β - 1)(ψ(β) - ψ(α+β))
    
    # ψ(α) - ψ(α+β)
    digamma_alpha = torch.digamma(alpha) - torch.digamma(alpha0) 
    # ψ(β) - ψ(α+β)
    digamma_beta = torch.digamma(beta) - torch.digamma(alpha0) 
    
    # (α - 1) * (ψ(α) - ψ(α+β))
    term_alpha = (alpha - 1.0) * digamma_alpha 
    # (β - 1) * (ψ(β) - ψ(α+β))
    term_beta = (beta - 1.0) * digamma_beta 
    
    kl_part2 = (term_alpha + term_beta).squeeze(1) # (B,)
    
    kl = kl_part1 + kl_part2

    return kl


def evidential_loss(alpha, beta, y, lam=0.2,eps=1e-8):
    # y: (B,) in {0,1}
    # S = alpha + beta
    # p = alpha / (S + 1e-8)
    y = y.view(-1)
    if y.dtype != torch.long:
        y = y.long()
    
    # 1. alpha와 beta를 하나의 (B, 2) 텐서로 만듭니다.
    alpha_beta = torch.stack([alpha, beta], dim=1) # (B, 2) 텐서 생성
    alpha_beta = alpha_beta.clamp_min(eps)         # alpha > 0 보장

    #alpha_beta = torch.stack([alpha, beta], dim=1)
    S = alpha_beta.sum(dim=1) # (B,)

    #alpha_y = alpha_beta[torch.arange(y.size(0), device=y.device), y]
    alpha_y = torch.where(y == 1, alpha, beta)
    #alpha = alpha.clamp_min(eps)  # Dirichlet는 >0 필요

    uce = torch.digamma(S) - torch.digamma(alpha_y)
    reg = kl_beta_to_uniform(alpha_beta)          # ← Uniform prior로 끌어당김
    # print('S:',S)
    # print('P:',p)
    batch_loss = uce + lam * reg
    return batch_loss.mean()


def weighted_mse_loss(pred, target, weight_val=2.0):
    # 2. 그 상태에서 MSE 계산
    raw_loss = (pred - target) ** 2
    # 3. 가중치 적
    weights = torch.where(target > 0.001, weight_val, 1.0)
    return (raw_loss * weights).mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = self.bce(inputs, targets)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
            
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class AleatoricLoss(nn.Module):

    def __init__(self, is_log_sigma, res_loss='l2', nb_samples=10):
        super().__init__()
        self.is_log_sigma = is_log_sigma
        self.nb_samples = nb_samples
        self.ignore_index=255
        self.res_loss = res_loss

    def forward(self, logits, sigma, target):
        if logits.dim() > target.dim():
            logits = logits.squeeze(-1)
        if sigma.dim() > target.dim():
            sigma = sigma.squeeze(-1)
        
        if self.res_loss == 'l2':
            loss1 = torch.mul(torch.exp(-sigma), F.mse_loss(logits, target, reduction='none'))
        elif self.res_loss == 'l1':
            loss1 = torch.mul(torch.exp(-sigma), F.l1_loss(logits, target, reduction='none'))
        else:
            raise Exception("Invalid residual loss")
        loss2 = sigma
        loss = (0.5 * (loss1 + loss2)).mean()
        return loss
