import os,sys
import random
import torch
import numpy as np
import wandb  # ✅ wandb 추가

os.environ.pop("BOOST_ROOT", None)
sys.path.insert(0, "/home/kimseungjun/task/PointTransformer/Pointcept")
sys.path.insert(0, "/home/kimseungjun/task/PointTransformer/My_point_transformer")

from model_utils.data_loader import PT_data_loader, unified_collate_fn
from models.PT3_model import PointTransformerV3

from functools import partial


from torch.utils.data import random_split,DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
device = torch.device("cuda:0")
import time

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

class Config:
    data_dir: str = '/home/kimseungjun/datasets/My_PT_data/PT_data'
    label_key: str = "label"
    val_ratio: float = 0.2
    batch_size: int = 128
    hidden: int = 64
    epochs: int = 30
    lr: float = 1e-4
    seed: int = 22
    save_dir: str = "./runs_PT"
    pos_weight: float = 1.0
    wd: float = 1e-4
    # ✅ wandb 설정 추가
    wandb_project: str = "cmes-PT"
    wandb_entity: str = None  # None이면 기본 사용자
    wandb_name: str = "cmes-PT-experiment"
# ---- Early Stopping ----
    patience: int = 50        # 개선 없음을 몇 epoch까지 허용할지
    min_delta_loss: float = 1e-4   # '개선'으로 인정할 최소 향상(accuracy 기준)
    stop_on_big_drop: bool = False  # 큰 폭 하락 시 즉시 중단할지
    drop_delta: float = 0.05        # 큰 폭 하락 정의(예: 5%p 하락)
    min_lr: float = 1e-6     # 최소 학습률
    t_max: int = 300   

CFG = Config()

# ====== 유틸 ======
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def move_to_device(d, device):
    for k, v in list(d.items()):
        if torch.is_tensor(v):
            d[k] = v.to(device, non_blocking=True)
def time_stamp():
        t_local = time.localtime()
        t_str = f"| {t_local.tm_hour:>2}:{t_local.tm_min:>2}:{t_local.tm_sec:>2} |"
        return t_str



# ====== 학습/평가 ======
def accuracy_from_logits(logits, y):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()
    return (preds == y).float().mean().item()

def accuracy_from_prob(p, y):
    preds = (p > 0.5).long()
    return (preds == y).float().mean().item()

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

    # log B(uniform) = log B(1, 1) = log( (Gamma(1)Gamma(1)) / Gamma(2) ) = log(1/1) = 0
    logB_uniform = 0.0

    # 2. 다이감마 함수 (Digamma function, ψ)를 사용한 KL 발산 계산
    # KL = logB(1) - logB(alpha, beta) + Σ (α_k - 1)(ψ(α_k) - ψ(α0))
    
    # 두 항으로 나눠서 계산
    
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



def train_loop():
    data_path = '/home/kimseungjun/datasets/My_PT_data/PT_data'
    batch_size = 4
    max_iter = 64000
    lr = 0.001
    max_epoch = 300
    data=PT_data_loader(data_path, split='train', use_color=False)
    data_size=len(data)
    train_size=int(data_size*0.8)
    valid_size=data_size-train_size
    train_data,valid_data = random_split(data,[train_size,valid_size])
    train_data_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=partial(unified_collate_fn, mix_prob=0.0),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
    test_data_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=partial(unified_collate_fn, mix_prob=0.0),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

    PT_model = PointTransformerV3()
    PT_model.to(device)
    pos_weight = torch.tensor([1.0], device=device)

    loss_model = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    set_seed(CFG.seed)
    os.makedirs(CFG.save_dir, exist_ok=True)
    # ✅ wandb 초기화
    wandb.init(
        project=CFG.wandb_project,
        entity=CFG.wandb_entity,
        name=CFG.wandb_name,
        config={
            "data_dir": CFG.data_dir,
            "batch_size": CFG.batch_size,
            "epochs": CFG.epochs,
            "learning_rate": CFG.lr,
            "weight_decay": CFG.wd,
            "pos_weight": CFG.pos_weight,
            "patience": CFG.patience,
            "val_ratio": CFG.val_ratio,
            "seed": CFG.seed
        }
    )
    wandb.define_metric("epoch")                      # epoch을 축으로
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*",   step_metric="epoch")

    PT_model.apply(inplace_relu)
    # optimizer = torch.optim.Adam(
    #         PT_model.parameters(),
    #         lr=3e-4,
    #         betas=(0.9, 0.999),
    #         eps=1e-08,
    #         weight_decay=1e-4
    #     )
    optimizer = optim.AdamW(PT_model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.t_max, eta_min=CFG.min_lr)

    count =0
    global_step = 0
    for e in range(max_epoch):
        PT_model.train()
        total_correct_n = 0
        total_data_len = 0
        epoch_train_losses = []   # <-- 추가
        epoch_train_accs   = []   # <-- 추가

        for i, (batch) in enumerate(train_data_loader):
            iter_range = e*len(train_data_loader) + i
            #label = batch.pop("label").to(device).float()      # (B,) float
            label = batch.pop("label").to(device).long()      # (B,) long
            inputs = batch                                     # {'coord','feat','offset', ...}

            # 입력 텐서 전부 같은 디바이스로    
            move_to_device(inputs, device)                  # {'coord','feat','offset',...}
            optimizer.zero_grad(set_to_none=True)
        
            alpha, beta, p = PT_model(inputs)
            
#            target = label.float().unsqueeze(1)
            target = label.view(-1) 
            evi_loss = evidential_loss(alpha=alpha, beta=beta, y=target, lam=0.1)


            # 2) 그래디언트 실제로 생기는지
            #optimizer.zero_grad()
            evi_loss.backward()

            pred_label = (p > 0.5).long()
            n_correct = (pred_label == label).sum()
            # pred_label = (p > 0.5).long()  
            # n_correct = (pred_label == label.float()).sum()

            batch_acc = (n_correct.float() / len(target)).item()
            epoch_train_losses.append(evi_loss.item())
            epoch_train_accs.append(batch_acc)

            #loss.backward()
            optimizer.step()
            stt = (e+1) * len(train_data_loader)
            
            total_correct_n +=n_correct
            total_data_len += len(target)
            accuracy = total_correct_n/total_data_len
            if count % 10 == 0:
                times = time_stamp()
                result_s = "***"*20 + " " + times + "\n"
                print(result_s)
                print(f'Train Epoch: {e} / iter.{iter_range}')
                print('epoch loss',evi_loss.item())
                print('epoch accuracy',accuracy.item())
                correct_mask = (pred_label == label)
                wrong_mask   = ~correct_mask

                S = alpha + beta

                print("correct S mean:", S[correct_mask].mean().item())
                print("wrong   S mean:", S[wrong_mask].mean().item())
                            
            count+=1
            global_step = 0
        wandb.log({
            "epoch": e,
            "train/epoch_loss": float(np.mean(epoch_train_losses)),
            "train/epoch_acc": float(np.mean(epoch_train_accs)),
            }, step=e)


        val_correct = 0
        val_data_len = 0
        epoch_val_losses = []
        epoch_val_accs   = []
        with torch.no_grad():
            PT_model.eval()
            val_total = 0
            val_correct = 0
            val_loss_sum = 0.0
            val_label_pos = 0
            val_pred_pos  = 0
            for i, (batch) in enumerate(test_data_loader):
                label = batch.pop("label").to(device).float()      # (B,) float
                inputs = batch                                     # {'coord','feat','offset', ...}

                # 입력 텐서 전부 같은 디바이스로    
                move_to_device(inputs, device)                  # {'coord','feat','offset',...}
                optimizer.zero_grad(set_to_none=True)

                alpha, beta, p = PT_model(inputs)
                target = label.float().unsqueeze(1)
                evi_loss = evidential_loss(alpha=alpha, beta=beta, y=target, lam=0.1)
                #loss = loss_model(out, target)
                
                #pred_label = (probs > 0.5).long()
                pred_label = (p > 0.5).long()
                corr  = (pred_label == label.float()).sum().item()
                B = target.numel()     
                val_pred_pos  += pred_label.sum().item()
                val_label_pos += label.long().sum().item()
                val_correct += corr
                val_total   += B
                val_loss_sum += evi_loss.item() * B
            
            val_loss_epoch = val_loss_sum / val_total
            val_acc_epoch  = val_correct / val_total

            wandb.log({
                "val/loss": val_loss_epoch,
                "val/acc":  val_acc_epoch,
                "val/pred_pos": val_pred_pos,
                "val/label_pos": val_label_pos,
                "epoch": e,
            }, step=e)

            times = time_stamp()
            result_s = "***"*20 + " " + times + "\n"
            print(result_s)
            print(f'Validation Epoch: {e} / iter.{iter_range}')
            print('val loss', val_loss_epoch)
            print('val accuracy', val_acc_epoch)
            print(f"[VAL] #pred=1: {val_pred_pos} | #pred=0: {val_total - val_pred_pos}")
            print(f"[VAL] #label=1: {val_label_pos} | #label=0: {val_total - val_label_pos}")
        #torch.save(PT_model.state_dict(), "save_point_net_"+str(e)+".pth")
        scheduler.step()

    wandb.finish()
    save_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(save_dir, exist_ok=True)

    
if __name__ == "__main__":
    train_loop()
