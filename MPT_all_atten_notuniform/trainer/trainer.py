import os,sys
import random
import torch
import numpy as np
import wandb  # ✅ wandb 추가
from pathlib import Path
current_file_path = Path(__file__).resolve()
path = Path(current_file_path)
os.environ.pop("BOOST_ROOT", None)
sys.path.insert(0, "/home/kimseungjun/task/PointTransformer/Pointcept")
#sys.path.insert(0, "/home/kimseungjun/task/PointTransformer/MPT_cross_attention_0119")
work_path = str(path.parent.parent)
sys.path.insert(0, work_path)

from model_utils.loss_utils import EvidentialRegressionLoss, FocalLoss, AleatoricLoss
from model_utils.data_loader_suctionnet import PT_dataset, unified_collate_fn
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

import logging
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
    data_dir: str = '/home/kimseungjun/task/PointTransformer/dataset/suctionnet'
    label_key: str = "label"
    val_ratio: float = 0.2
    batch_size: int = 8
    hidden: int = 64
    epochs: int = 40
    lr: float = 1e-4
    seed: int = 22
    save_dir: str = "./runs_PT"
    pos_weight: float = 1.1
    wd: float = 1e-2
    # ✅ wandb 설정 추가
    wandb_project: str = "PT-Picking"
    wandb_entity: str = None  # None이면 기본 사용자
    wandb_name: str = "MPT-custommodel"
# ---- Early Stopping ----
    patience: int = 50        # 개선 없음을 몇 epoch까지 허용할지
    min_delta_loss: float = 1e-4   # '개선'으로 인정할 최소 향상(accuracy 기준)
    stop_on_big_drop: bool = False  # 큰 폭 하락 시 즉시 중단할지
    drop_delta: float = 0.05        # 큰 폭 하락 정의(예: 5%p 하락)
    min_lr: float = 1e-6     # 최소 학습률
    t_max: int = 40   

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


def train_loop(log):
    logger = log
    data_path = '/home/kimseungjun/task/PointTransformer/dataset/suctionnet'
    batch_size = CFG.batch_size
    max_epoch = CFG.epochs
    best_val_loss = float('inf')  # 최소 Loss 추적용 (무한대로 초기화)

    train_data=PT_dataset(data_path,camera='realsense', use_color=False,log = logger)
    valid_data=PT_dataset(data_path,split='test_seen',camera='realsense', use_color=False,log=logger)
    train_data_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=6,
            collate_fn=unified_collate_fn,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=1,
        )
    
    test_data_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=6,
            collate_fn=unified_collate_fn,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=1,
        )

    My_suction_model = PTCrossATT()

    #PT_model = PointTransformerV3()
    My_suction_model.to(device)
    #aleatoric_criterion = AleatoricLoss(is_log_sigma=False, res_loss='l1', nb_samples=10)

    #evidence_loss = #coeff=1e-2

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
    optimizer = optim.AdamW(My_suction_model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.t_max, eta_min=CFG.min_lr)

    count =0
    global_step = 0
    ANNEALING_EPOCHS = 10
    MAX_KL_COEFF = 1e-1
    for e in range(max_epoch):
        My_suction_model.train()
        # 에폭별 통계 저장용 리스트
        epoch_total_loss = []
        epoch_score_loss = []
        epoch_coll_loss  = []
        epoch_coll_acc   = []
        epoch_score_auc = []
        epoch_score_ap_20   = []  # [추가] AP 저장용 리스트
        epoch_score_ap_40   = []
        epoch_score_ap_60   = []
        epoch_score_ap_80   = []
        k = 50
        epoch_aleatoric_uncertainty = []
        epoch_epistemic_uncertainty = []

        if e < ANNEALING_EPOCHS:
            current_coeff = MAX_KL_COEFF * (e / ANNEALING_EPOCHS)
        else:
            current_coeff = MAX_KL_COEFF

        for i, (batch) in enumerate(train_data_loader):
            
            batch = to_device(batch, device)
            iter_range = e*len(train_data_loader) + i
            label = batch.pop("label")
            pick_feature = torch.cat([label['coord'], label['normal']], dim=-1)
            
            # 입력 텐서 전부 같은 디바이스로    
            optimizer.zero_grad(set_to_none=True)
            inputs = batch

            # 라벨 GPU 이동 및 차원 맞추기
            seal_label_score = label['seal_score'].float().cuda()         # (B, N)
            wrench_label_score = label['wrench_score'].float().cuda()
            #label_coll  = label['collision'].float().cuda()     # (B, N)
            label_score = seal_label_score * wrench_label_score
            
            pred_score_logit  = My_suction_model(inputs,pick_feature) # 출력 형태: (B, 2)
            loss_score = EvidentialRegressionLoss(label_score, pred_score_logit, coeff=current_coeff,pos_weight=CFG.pos_weight)
            
            units = pred_score_logit.shape[-1] // 4
            gamma, v, alpha, beta = torch.split(pred_score_logit, units, dim=-1)
            # 1. 알레아토리 불확실성 (데이터 노이즈)
            aleatoric = beta / (alpha - 1)

            # 2. 에피스테믹 불확실성 (모델의 확신도 - 지식 부족)
            epistemic = beta / (v * (alpha - 1))
            #loss_score = weighted_mse_loss(pred_score_logit.squeeze(-1), label_score, weight_val=2.0)


            epoch_aleatoric_uncertainty.append(aleatoric.mean().item())
            epoch_epistemic_uncertainty.append(epistemic.mean().item())

            

            total_loss = (1.0 * loss_score) #+ (1.0 * loss_collision)
            total_loss.backward()

            gamma = pred_score_logit[:, :, 0]
            topk_scores, topk_indices = torch.topk(gamma, k=k, dim=1, largest=True, sorted=True)
            
            topk_labels = torch.gather(label_score, dim=1, index=topk_indices)

            num_success_20 = (topk_labels >= 0.2).float().sum(dim=1) # (B, )
            precision_at_k_20 = (num_success_20 / k).mean().item()

            num_success_40 = (topk_labels >= 0.4).float().sum(dim=1) # (B, )
            precision_at_k_40 = (num_success_40 / k).mean().item()

            num_success_60 = (topk_labels >= 0.6).float().sum(dim=1) # (B, )
            precision_at_k_60 = (num_success_60 / k).mean().item()

            num_success_80 = (topk_labels >= 0.8).float().sum(dim=1) # (B, )
            precision_at_k_80 = (num_success_80 / k).mean().item()

            epoch_total_loss.append(total_loss.item())
            epoch_score_loss.append(loss_score.item())
            # epoch_coll_loss.append(loss_collision.item())
            # epoch_coll_acc.append(acc_coll)
            #epoch_score_auc.append(auc_score) # 리스트 선언 필요
            epoch_score_ap_20.append(precision_at_k_20)    # [추가] 리스트에 추가
            epoch_score_ap_40.append(precision_at_k_40)
            epoch_score_ap_60.append(precision_at_k_60)
            epoch_score_ap_80.append(precision_at_k_80)

            #loss.backward()
            optimizer.step()
            stt = (e+1) * len(train_data_loader)

            if count % 100 == 0:
                times = time_stamp() # (함수 정의되어 있다고 가정)
                print(f"[{times}] Epoch: {e} | Iter: {iter_range}")
                print(f"  Total Loss: {total_loss.item():.4f}")
                print(f"  - Score Loss: {loss_score.item():.4f}")
                #print(f"  - Coll Loss : {loss_collision.item():.4f}")
                #print(f"  Collision Acc: {acc_coll * 100:.2f}%")
                print(f"  Current AP 0.2: {precision_at_k_20:.4f}")
                print(f"  Current AP 0.4: {precision_at_k_40:.4f}")
                print(f"  Current AP 0.6: {precision_at_k_60:.4f}")
                print(f"  Current AP 0.8: {precision_at_k_80:.4f}")
                print(f"  Uncertainty (Current Batch):")
                print(f"    - Aleatoric (Data Noise): {aleatoric.mean().item():.4f}")
                print(f"    - Epistemic (Model Conf): {epistemic.mean().item():.4f}")

                print(f"Epoch {e+1} | AP 0.2: {np.mean(epoch_score_ap_20):.4f}")
                print(f"Epoch {e+1} | AP 0.4: {np.mean(epoch_score_ap_40):.4f}")
                print(f"Epoch {e+1} | AP 0.6: {np.mean(epoch_score_ap_60):.4f}")
                print(f"Epoch {e+1} | AP 0.8: {np.mean(epoch_score_ap_80):.4f}")
                # [추가] 이번 에폭 전체의 평균 불확실성
                print(f"  Epoch {e+1} Avg Uncertainty:")
                print(f"    - Avg Aleatoric: {np.mean(epoch_aleatoric_uncertainty):.4f}")
                print(f"    - Avg Epistemic: {np.mean(epoch_epistemic_uncertainty):.4f}")

                print("-" * 30)
                            
            count+=1
            global_step = 0
        
        train_ap_20 = np.mean(epoch_score_ap_20)
        train_ap_40 = np.mean(epoch_score_ap_40)
        train_ap_60 = np.mean(epoch_score_ap_60)
        train_ap_80 = np.mean(epoch_score_ap_80)

        wandb.log({
            "epoch": e,
            "train/total_loss": np.mean(epoch_total_loss),
            "train/score_loss": np.mean(epoch_score_loss),
            "train/AP@0.2": train_ap_20,
            "train/AP@0.4": train_ap_40,
            "train/AP@0.6": train_ap_60,
            "train/AP@0.8": train_ap_80,
            "lr": optimizer.param_groups[0]['lr'],
        }, step=e)

        val_correct = 0
        val_epoch_total_loss = []
        val_epoch_score_loss = []
        val_epoch_coll_loss  = []

        val_epoch_score_ap_20   = []  # [추가] AP 저장용 리스트
        val_epoch_score_ap_40   = []
        val_epoch_score_ap_60   = []
        val_epoch_score_ap_80   = []
        val_epoch_aleatoric_uncertainty = []
        val_epoch_epistemic_uncertainty = []
        with torch.no_grad():
            My_suction_model.eval()

            for i, (batch) in enumerate(test_data_loader):
                batch = to_device(batch, device)
                val_label = batch.pop("label")
                val_pick_feature = torch.cat([val_label['coord'], val_label['normal']], dim=-1)
                #1,512,6
                val_inputs = batch
                # 라벨 GPU 이동 및 차원 맞추기
                val_seal_label_score = val_label['seal_score'].float().cuda()         # (B, N)
                val_wrench_label_score = val_label['wrench_score'].float().cuda()
                #label_coll  = label['collision'].float().cuda()     # (B, N)
                val_label_score = val_seal_label_score * val_wrench_label_score

                val_pred_score_logit = My_suction_model(val_inputs,val_pick_feature) # 출력 형태: (B, 2)
                val_loss_score = EvidentialRegressionLoss(val_label_score, val_pred_score_logit, coeff=1e-2,pos_weight=CFG.pos_weight)

                val_units = val_pred_score_logit.shape[-1] // 4
                val_gamma, val_v, val_alpha, val_beta = torch.split(val_pred_score_logit, val_units, dim=-1)
                # 1. 알레아토리 불확실성 (데이터 노이즈)
                val_aleatoric = val_beta / (val_alpha - 1)
                # 2. 에피스테믹 불확실성 (모델의 확신도 - 지식 부족)
                val_epistemic = val_beta / (val_v * (val_alpha - 1))
                #loss_score = weighted_mse_loss(pred_score_logit.squeeze(-1), label_score, weight_val=2.0)

                val_epoch_aleatoric_uncertainty.append(val_aleatoric.mean().item())
                val_epoch_epistemic_uncertainty.append(val_epistemic.mean().item())


                val_total_loss = (1.0 * val_loss_score) #+ (1.0 * val_loss_collision)

                val_gamma = val_pred_score_logit[:, :, 0]

                val_topk_scores, val_topk_indices = torch.topk(val_gamma, k=k, dim=1, largest=True, sorted=True)
                val_topk_indices = val_topk_indices.squeeze(-1)
                val_topk_labels = torch.gather(val_label_score, dim=1, index=val_topk_indices)

                val_num_success_20 = (val_topk_labels >= 0.2).float().sum(dim=1) # (B, )
                val_precision_at_k_20 = (val_num_success_20 / k).mean().item()

                val_num_success_40 = (val_topk_labels >= 0.4).float().sum(dim=1) # (B, )
                val_precision_at_k_40 = (val_num_success_40 / k).mean().item()

                val_num_success_60 = (val_topk_labels >= 0.6).float().sum(dim=1) # (B, )
                val_precision_at_k_60 = (val_num_success_60 / k).mean().item()

                val_num_success_80 = (val_topk_labels >= 0.8).float().sum(dim=1) # (B, )
                val_precision_at_k_80 = (val_num_success_80 / k).mean().item()

                # epoch_coll_acc.append(acc_coll)
                #epoch_score_auc.append(auc_score) # 리스트 선언 필요
                val_epoch_score_ap_20.append(val_precision_at_k_20)    # [추가] 리스트에 추가
                val_epoch_score_ap_40.append(val_precision_at_k_40)
                val_epoch_score_ap_60.append(val_precision_at_k_60)
                val_epoch_score_ap_80.append(val_precision_at_k_80)


                # 5. 리스트에 저장
                val_epoch_total_loss.append(val_total_loss.item())
                val_epoch_score_loss.append(val_loss_score.item())

            mean_val_loss = np.mean(val_epoch_total_loss)
            val_ap_20 = np.mean(val_epoch_score_ap_20)
            val_ap_40 = np.mean(val_epoch_score_ap_40)
            val_ap_60 = np.mean(val_epoch_score_ap_60)
            val_ap_80 = np.mean(val_epoch_score_ap_80)

            wandb.log({
                "val/loss": mean_val_loss,
                "val/score_loss": np.mean(val_epoch_score_loss),
                "val/AP@0.2": val_ap_20,
                "val/AP@0.4": val_ap_40,
                "val/AP@0.6": val_ap_60,
                "val/AP@0.8": val_ap_80,
                "epoch": e,
            }, step=e)

            times = time_stamp()
            print(f"[{times}] Validation Epoch: {e}")
            print(f"  Total val Loss: {np.mean(val_epoch_score_loss)}")
            print(f"  AP@0.2: {val_ap_20:.4f}")
            print(f"  AP@0.4: {val_ap_40:.4f}")
            print(f"  AP@0.6: {val_ap_60:.4f}")
            print(f"  AP@0.8: {val_ap_80:.4f}")
            print(f"  Epoch {e+1} Avg Uncertainty:")
            print(f"    - Avg Aleatoric: {np.mean(val_epoch_aleatoric_uncertainty):.4f}")
            print(f"    - Avg Epistemic: {np.mean(val_epoch_epistemic_uncertainty):.4f}")
            print("*" * 60)
        
        checkpoint = {
            'epoch': e,
            'model_state_dict': My_suction_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': mean_val_loss,
        }

        # 1. 최신 모델 저장 (언제든 이어 학습할 수 있도록)
        latest_path = os.path.join(CFG.save_dir, "latest_model.pth")
        torch.save(checkpoint, latest_path)

        # 2. 베스트 모델 저장 (Validation Loss 기준 최소값 갱신 시)
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_path = os.path.join(CFG.save_dir, "best_model_loss.pth")
            torch.save(checkpoint, best_path)
            print(f"⭐ Best Model Saved! Loss: {best_val_loss:.4f} (Epoch {e})")
        
        
        #torch.save(PT_model.state_dict(), "save_point_net_"+str(e)+".pth")
        scheduler.step()

    wandb.finish()
    save_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(save_dir, exist_ok=True)

    
if __name__ == "__main__":
    log_file_path = os.path.join(work_path,'log')

    log_filename = os.path.join(log_file_path,f"planning.log")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            #logging.StreamHandler()
        ]
    )
    logging.getLogger().handlers[0].flush()
    logging.info("✅ 로그가 정상적으로 기록됩니다.")
    train_loop(log = logging)
