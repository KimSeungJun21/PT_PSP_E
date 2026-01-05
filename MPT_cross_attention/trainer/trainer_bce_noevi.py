import os,sys
import random
import torch
import numpy as np
import wandb  # ✅ wandb 추가

os.environ.pop("BOOST_ROOT", None)

sys.path.insert(0, "/home/kimseungjun/task/PointTransformer/Pointcept")
sys.path.insert(0, "/home/kimseungjun/task/PointTransformer/MPT_cross_attention")


from model_utils.data_loader_suctionnet import PT_data_loader, unified_collate_fn
from models.PT3_model import PointTransformerV3
from models.My_model import PTCrossATT
from functools import partial

from torch.utils.data import random_split,DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
device = torch.device("cuda:0")
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CV_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

class Config:
    data_dir: str = '/home/kimseungjun/datasets/graspnet_data/suctionnet'
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

from collections.abc import Mapping, Sequence
def to_device(data, device):
    """
    딕셔너리나 리스트가 중첩되어 있어도 끝까지 파고들어서 
    모든 Tensor를 device로 옮겨주는 함수
    """
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    elif isinstance(data, Mapping):  # 딕셔너리인 경우 (scene, target, label 등)
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)): # 리스트인 경우
        return [to_device(d, device) for d in data]
    else:
        return data

# ====== 학습/평가 ======
# def accuracy_from_logits(logits, y):
#     probs = torch.sigmoid(logits)
#     preds = (probs > 0.5).long()
#     return (preds == y).float().mean().item()

def accuracy_from_logits(logits, y):
    # logits: (B, 2), y: (B,)
    preds = torch.argmax(logits, dim=1)
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
    data_path = '/home/kimseungjun/datasets/graspnet_data/suctionnet'
    batch_size = CFG.batch_size
    max_iter = 64000
    lr = 0.001
    max_epoch = 300
    data=PT_data_loader(data_path, use_color=True)
    data_size=len(data)
    train_size=int(data_size*0.8)
    valid_size=data_size-train_size
    train_data,valid_data = random_split(data,[train_size,valid_size])
    train_data_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=5,
            collate_fn=unified_collate_fn,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
    test_data_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=5,
            collate_fn=unified_collate_fn,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

    My_suction_model = PTCrossATT()

    #PT_model = PointTransformerV3()
    My_suction_model.to(device)

    criterion_score = nn.MSELoss()        # 점수용 (Regression)
    criterion_coll = nn.BCEWithLogitsLoss() # 충돌용 (Binary Classification)

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

    My_suction_model.apply(inplace_relu)
    # optimizer = torch.optim.Adam(
    #         PT_model.parameters(),
    #         lr=3e-4,
    #         betas=(0.9, 0.999),
    #         eps=1e-08,
    #         weight_decay=1e-4
    #     )
    optimizer = optim.AdamW(My_suction_model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.t_max, eta_min=CFG.min_lr)

    count =0
    global_step = 0
    for e in range(max_epoch):
        My_suction_model.train()
# 에폭별 통계 저장용 리스트
        epoch_total_loss = []
        epoch_score_loss = []
        epoch_coll_loss  = []
        epoch_coll_acc   = []
        epoch_score_auc = []
        epoch_score_ap   = []  # [추가] AP 저장용 리스트
        for i, (batch) in enumerate(train_data_loader):
            batch = to_device(batch, device)
            iter_range = e*len(train_data_loader) + i
            label = batch.pop("label")
            pick_feature = torch.cat([label['coord'], label['normal']], dim=-1)
            #1,512,6
            
            # 입력 텐서 전부 같은 디바이스로    
            optimizer.zero_grad(set_to_none=True)
        

            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            # 라벨 GPU 이동 및 차원 맞추기
            label_score = label['score'].float().cuda()         # (B, N)
            label_coll  = label['collision'].float().cuda()     # (B, N)
            
            pred_score_logit, pred_coll_logit = My_suction_model(inputs,pick_feature) # 출력 형태: (B, 2)

            # (1) Score Loss (MSE)
            # Sigmoid를 통과시켜 0~1로 맞춘 뒤 MSE 계산 (혹은 Logit 자체로 계산)
            pred_score = torch.sigmoid(pred_score_logit) 
            loss_score = criterion_score(pred_score.squeeze(-1), label_score)
            
            # (2) Collision Loss (BCE)
            # BCEWithLogitsLoss는 내부에서 Sigmoid를 처리하므로 Logit 그대로 넣음
            loss_collision = criterion_coll(pred_coll_logit.squeeze(-1), label_coll)

            total_loss = (1.0 * loss_score) + (1.0 * loss_collision)
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()

            pred_coll_prob = torch.sigmoid(pred_coll_logit.squeeze(-1))
            # 확률 0.5 이상이면 충돌(1), 아니면 안전(0)으로 판단
            pred_coll_binary = (pred_coll_prob > 0.5).float()
            
            # 맞은 개수 / 전체 개수
            correct_mask = (pred_coll_binary == label_coll)
            acc_coll = correct_mask.float().mean().item()

            # (3) Score AUC 계산
            # 정답(label_score)이 실수값이므로, 0.5 기준으로 0/1로 변환해야 AUC 계산 가능
            np_label_score = label_score.detach().cpu().numpy().flatten()
            np_pred_score  = torch.sigmoid(pred_score_logit).detach().cpu().numpy().flatten()
            np_label_score_binary = (np_label_score > 0.5).astype(int)

            try:
            # 정답(0/1) vs 예측값(0~1 확률) 비교
                auc_score = roc_auc_score(np_label_score_binary, np_pred_score)
                ap_score = average_precision_score(np_label_score_binary, np_pred_score)
            except ValueError:
                auc_score = 0.5
                ap_score = 0.0



            epoch_total_loss.append(total_loss.item())
            epoch_score_loss.append(loss_score.item())
            epoch_coll_loss.append(loss_collision.item())
            epoch_coll_acc.append(acc_coll)
            epoch_score_auc.append(auc_score) # 리스트 선언 필요
            epoch_score_ap.append(ap_score)    # [추가] 리스트에 추가
            
            #loss.backward()
            optimizer.step()
            stt = (e+1) * len(train_data_loader)

            if count % 10 == 0:
                times = time_stamp() # (함수 정의되어 있다고 가정)
                print(f"[{times}] Epoch: {e} | Iter: {iter_range}")
                print(f"  Total Loss: {total_loss.item():.4f}")
                print(f"  - Score Loss: {loss_score.item():.4f}")
                print(f"  - Coll Loss : {loss_collision.item():.4f}")
                print(f"  Collision Acc: {acc_coll * 100:.2f}%")
                print(f"Epoch {e+1} | AP: {np.mean(epoch_score_ap):.4f} | AUC: {np.mean(epoch_score_auc):.4f}")
                print("-" * 30)
                            
            count+=1
            global_step = 0
        
        wandb.log({
            "epoch": e,
            "train/total_loss": np.mean(epoch_total_loss),
            "train/score_loss": np.mean(epoch_score_loss),     # Score는 MSE가 얼마나 주는지 봄
            "train/coll_loss":  np.mean(epoch_coll_loss),
            "train/coll_acc":   np.mean(epoch_coll_acc),       # 충돌 예측 정확도
            "lr": optimizer.param_groups[0]['lr']
        }, step=e)


        val_correct = 0
        val_epoch_total_loss = []
        val_epoch_score_loss = []
        val_epoch_coll_loss  = []
        val_epoch_coll_acc   = []
        val_epoch_score_auc  = []
        val_epoch_score_ap   = []  # [추가] AP 저장용 리스트

        with torch.no_grad():
            My_suction_model.eval()

            for i, (batch) in enumerate(test_data_loader):
                batch = to_device(batch, device)
                val_iter_range = e*len(test_data_loader) + i

                val_label = batch.pop("label")
                val_pick_feature = torch.cat([val_label['coord'], val_label['normal']], dim=-1)
                #1,512,6
                
                # 입력 텐서 전부 같은 디바이스로    
                optimizer.zero_grad(set_to_none=True)

                val_inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                # 라벨 GPU 이동 및 차원 맞추기
                val_label_score = val_label['score'].float().cuda()         # (B, N)
                val_label_coll  = val_label['collision'].float().cuda()     # (B, N)
            
                val_pred_score_logit, val_pred_coll_logit = My_suction_model(val_inputs,val_pick_feature) # 출력 형태: (B, 2)
                # (1) Score Loss (MSE)
                # Sigmoid를 통과시켜 0~1로 맞춘 뒤 MSE 계산 (혹은 Logit 자체로 계산)
                val_pred_score = torch.sigmoid(val_pred_score_logit) 
                val_loss_score = criterion_score(val_pred_score.squeeze(-1), val_label_score)
                
                # (2) Collision Loss (BCE)
                # BCEWithLogitsLoss는 내부에서 Sigmoid를 처리하므로 Logit 그대로 넣음
                val_loss_collision = criterion_coll(val_pred_coll_logit.squeeze(-1), val_label_coll)

                val_total_loss = (1.0 * val_loss_score) + (1.0 * val_loss_collision)
                
                val_pred_coll_prob = torch.sigmoid(val_pred_coll_logit.squeeze(-1))
                # 확률 0.5 이상이면 충돌(1), 아니면 안전(0)으로 판단
                val_pred_coll_binary = (val_pred_coll_prob > 0.5).float()
                
                # 맞은 개수 / 전체 개수
                val_correct_mask = (val_pred_coll_binary == val_label_coll)
                val_acc_coll = val_correct_mask.float().mean().item()

                # (3) Score AUC 계산
                # 정답(label_score)이 실수값이므로, 0.5 기준으로 0/1로 변환해야 AUC 계산 가능
                val_np_label_score = val_label_score.detach().cpu().numpy().flatten()
                val_np_pred_score  = torch.sigmoid(val_pred_score_logit).detach().cpu().numpy().flatten()
                val_np_label_score_binary = (val_np_label_score > 0.5).astype(int)

                try:
                # 정답(0/1) vs 예측값(0~1 확률) 비교
                    val_auc_score = roc_auc_score(val_np_label_score_binary, val_np_pred_score)
                    val_ap_score = average_precision_score(val_np_label_score_binary, val_np_pred_score)

                except ValueError:
                    val_auc_score = 0.5
                    val_ap_score = 0.0


                # 5. 리스트에 저장
                val_epoch_total_loss.append(val_total_loss.item())
                val_epoch_score_loss.append(val_loss_score.item())
                val_epoch_coll_loss.append(val_loss_collision.item())
                val_epoch_coll_acc.append(val_acc_coll)
                val_epoch_score_auc.append(val_auc_score)
                val_epoch_score_ap.append(val_ap_score)    # [추가] 리스트에 추가

            mean_val_loss = np.mean(val_epoch_total_loss)
            mean_val_acc  = np.mean(val_epoch_coll_acc)
            mean_val_score_auc = np.mean(val_epoch_score_auc)
            mean_val_score_ap = np.mean(val_epoch_score_ap)


            wandb.log({
                "val/loss": mean_val_loss,
                "val/score_loss": np.mean(val_epoch_score_loss),
                "val/coll_loss": np.mean(val_epoch_coll_loss),
                "val/coll_acc":  mean_val_acc,
                "val/score_ap": mean_val_score_ap,
                "val/score_auc": mean_val_score_auc,
                "epoch": e,
            }, step=e)

            times = time_stamp()
            print(f"[{times}] Validation Epoch: {e}")
            print(f"  Val Loss      : {mean_val_loss:.4f}")
            print(f"  Val Coll Acc  : {mean_val_acc * 100:.2f}%")
            print(f"  Val Score AUC : {mean_val_score_auc:.4f}")
            print(f"  Val Score AP : {mean_val_score_ap:.4f}")
            print("*" * 60)
        #torch.save(PT_model.state_dict(), "save_point_net_"+str(e)+".pth")
        scheduler.step()

    wandb.finish()
    save_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(save_dir, exist_ok=True)

    
if __name__ == "__main__":
    train_loop()
