import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import scipy.io as scio
import time
import pickle
from pathlib import Path
current_file_path = Path(__file__).resolve()
path = Path(current_file_path)
work_path = str(path.parent.parent)
sys.path.insert(0, work_path)

# Add paths for MPT model
os.environ.pop("BOOST_ROOT", None)
from sklearn.preprocessing import MinMaxScaler
# Pointcept 라이브러리 경로 설정
pointcept_path = os.environ.get('POINTCEPT_PATH', None)
if pointcept_path is None:
    # 프로젝트 내 Pointcept-main 디렉토리 찾기
    #base_dir = os.path.dirname(os.path.abspath(__file__))
    #pointcept_path = os.path.join(base_dir, 'MPT_cross_attention_0119', 'Pointcept-main')
    
    # Pointcept-main이 없으면 기본 경로 시도
    #if not os.path.exists(pointcept_path):
    pointcept_path = "/home/kimseungjun/task/PointTransformer/Pointcept"

if os.path.exists(pointcept_path):
    sys.path.insert(0, pointcept_path)
    print(f"Pointcept path: {pointcept_path}")
else:
    print(f"Warning: Pointcept path not found: {pointcept_path}")
    print("Please set POINTCEPT_PATH environment variable or ensure Pointcept-main is in MPT_cross_attention/")
#from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask
from models.My_model import PTCrossATT
from model_utils.data_loader_suctionnet import (
    CenterShift, NormalizeColor, ToTensor, Collect, GridSample, SphereCrop
)
from model_utils.loader_utils import (CenterShift,NormalizeColor,ToTensor,Collect,GridSample,SphereCrop,
                                        CameraInfo, create_point_cloud_from_depth_image,get_workspace_mask)

from model_utils.data_loader_suctionnet import PT_dataset, unified_collate_fn
import matplotlib.pyplot as plt
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)

minimum_num_pt = 50
num_pt = 1024
width = 1280
height = 720
suction_height = 0.1
suction_radius = 0.01
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
default_path = os.path.join(BASE_DIR, 'runs_PT', 'latest_model.pth')


parser = argparse.ArgumentParser()
parser.add_argument('--split', default='test_novel', help='dataset split [default: test_seen]')
parser.add_argument('--camera', default='realsense', help='camera to use [default: realsense]')
parser.add_argument('--dump_dir', default='figue_evidence', help='where to save')
parser.add_argument('--gpu_id', default='0', help='GPU index')
parser.add_argument('--checkpoint_path', default=default_path, help='path to checkpoint')
parser.add_argument('--seg_model', default='uois', help='segmentation model [uois/uoais]')
parser.add_argument('--dataset_root', default='/home/kimseungjun/datasets/graspnet_data/suctionnet', help='where dataset is')
parser.add_argument('--voxel_size', type=float, default=0.005, help='voxel size for point cloud processing')
cfgs = parser.parse_args()
print(cfgs)

device = torch.device("cuda:{}".format(cfgs.gpu_id) if torch.cuda.is_available() else "cpu")
split = cfgs.split
camera = cfgs.camera
dataset_root = cfgs.dataset_root
checkpoint_path = cfgs.checkpoint_path
seg_model = cfgs.seg_model
voxel_size = cfgs.voxel_size

if seg_model not in ['uoais', 'uois']:
    raise ValueError('unsupported segmentation model: ' + seg_model)

dump_dir = os.path.join('experiment', cfgs.dump_dir)
torch.cuda.set_device(device)
#suction_dir = os.path.join(dump_dir, split, 'scene_%04d'%scene_idx, camera, 'suction')
os.makedirs(dump_dir, exist_ok=True)

score_evidence_path = os.path.join(dump_dir,'score_evi')
os.makedirs(score_evidence_path, exist_ok=True)

score_evidence_path2 = os.path.join(dump_dir,'score_evi2')
os.makedirs(score_evidence_path2, exist_ok=True)

score_uncertainty_path = os.path.join(dump_dir,'uncertainty')
os.makedirs(score_uncertainty_path, exist_ok=True)

pred_vs_gt_path = os.path.join(dump_dir,'pred_vs_gt')
os.makedirs(pred_vs_gt_path, exist_ok=True)
# Load MPT model
print(f"Loading model from {checkpoint_path}")
net = PTCrossATT(hidden_dim=512, nhead=4)
net.to(device)

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        net.load_state_dict(checkpoint['state_dict'])
    else:
        net.load_state_dict(checkpoint)
    print("Model loaded successfully")
else:
    print(f"Warning: Checkpoint not found at {checkpoint_path}, using untrained model")
net.eval()

def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

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
def analyze_top_k(df, score_col='Pred_Score', k=50):
    """
    특정 점수(score_col)를 기준으로 상위 k개를 뽑았을 때의 정밀도 분석
    """
    # 1. 모델 예측 기준 상위 K개 (score_col을 기준으로 정렬!)
    top_k_pred_idx = set(df.nlargest(k, score_col).index)
    
    # 2. 실제 정답(GT) 기준 상위 K개 (이건 항상 GT_Score 기준)
    top_k_gt_idx = set(df.nlargest(k, 'GT_Score').index)
    
    # 3. 교집합(Intersection) 계산
    common_points = top_k_pred_idx.intersection(top_k_gt_idx)
    intersection_count = len(common_points)
    precision_at_k = intersection_count / k if k > 0 else 0
    
    # 4. 모델이 뽑은 Top K들의 실제 GT 점수 평균
    # (보정된 점수로 뽑았더라도, 평가는 '실제 GT 점수'로 해야 함)
    avg_gt_of_pred_top_k = df.loc[list(top_k_pred_idx), 'GT_Score'].mean()
    
    return {
        'precision': precision_at_k,
        'common_count': intersection_count,
        'model_topk_gt_avg': avg_gt_of_pred_top_k
    }
def get_top_k_precision(df, score_col, gt_col='GT_Score', k=50):
    if len(df) < k: return 0.0
    # 모델이 선택한 Top-K
    top_k_pred = set(df.nlargest(k, score_col).index)
    # 실제 정답(GT) Top-K (gt_col 인자 사용)
    top_k_gt = set(df.nlargest(k, gt_col).index)
    return len(top_k_pred.intersection(top_k_gt)) / k

def get_tiered_top_k_precision(df, raw_score_col, calib_score_col, gt_col, k_initial=200, k_final=50):
    if len(df) < k_initial: 
        # 데이터가 200개보다 적으면 그냥 보정 점수 기준으로 k_final개 추출
        return get_top_k_precision(df, calib_score_col, gt_col=gt_col, k=k_final)
    
    # 1단계: 원본 점수(Pred_Score) 기준 상위 200개 추출
    df_top_initial = df.nlargest(k_initial, raw_score_col)
    
    # 2단계: 그 200개 안에서 보정 점수(calib_score_col) 기준 상위 50개 추출
    top_k_pred_idx = set(df_top_initial.nlargest(k_final, calib_score_col).index)
    
    # 실제 정답(GT) Top-50 (전체 데이터 기준)
    top_k_gt_idx = set(df.nlargest(k_final, gt_col).index)
    
    return len(top_k_pred_idx.intersection(top_k_gt_idx)) / k_final

    
def draw_comparison_plot(data, title, filename):
    comp_save_path = os.path.join(dump_dir, 'calibration_comparison')
    os.makedirs(comp_save_path, exist_ok=True) # 폴더가 없으면 생성 (있으면 무시)
    plt.figure(figsize=(20, 8))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=data, x='Batch_ID', y='Precision', hue='Method', palette='Set2')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50% Target')
    plt.title(title, fontsize=18, fontweight='bold')
    plt.ylim(0, 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(comp_save_path, filename), dpi=150)
    plt.close()

def save_bar_chart(df, gt_type, filename,target_k):
    comp_save_path = os.path.join(dump_dir, 'calibration_comparison')
    os.makedirs(comp_save_path, exist_ok=True) # 폴더가 없으면 생성 (있으면 무시)

    plt.figure(figsize=(20, 8))
    sns.set_style("whitegrid")
    
    # 해당 GT Type만 필터링 (Standard or Safe)
    plot_data = df[df['Type'] == gt_type]
    
    ax = sns.barplot(data=plot_data, x='Batch_ID', y='Precision', hue='Method', palette='Set2')
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)

    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50% Target')
    plt.title(f'Top-{target_k} Precision: {gt_type} (Scene {scene_idx})', fontsize=18, fontweight='bold')
    plt.ylabel(f'Precision @ {target_k}', fontsize=14)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.ylim(0, 1.15)
    plt.tight_layout()
    
    plt.savefig(os.path.join(comp_save_path, filename), dpi=150)
    plt.close()

##########
TOTAL_SCENE_NUM = 190
from tqdm import tqdm
import open3d as o3d
import seaborn as sns
import pandas as pd
def data_analyze(scene_idx,valid_data):
    for anno_idx in range(256):
        samples,pick_suction_points,pick_normal_points = valid_data.analyze_data(scene_idx,anno_idx)
    
        chunk_size = 1  # or 5
        all_pred_scores = []
        all_pred_collision = []
        all_GT_scores = []
        all_GT_collision = []
        batch_stats = []
        start = 0
        for chunk in chunk_list(samples, chunk_size):
            batch_data = unified_collate_fn(chunk)
            batch = to_device(batch_data, device)
            label = batch.pop("label")
            pick_feature = torch.cat([label['coord'], label['normal']], dim=-1)
            inputs = batch
            seal_label_score = label['seal_score'].float().cuda()         # (B, N)
            wrench_label_score = label['wrench_score'].float().cuda()
            collision_label = label.get('collision', torch.zeros_like(seal_label_score)).float().to(device)  # (B, N)

            with torch.no_grad():
#                pred_score_logit,collision_score = net(inputs,pick_feature)
                pred_score_logit = net(inputs,pick_feature)

            all_pred_scores.append(pred_score_logit.detach().cpu())
#            all_pred_collision.append(collision_score.detach().cpu())
            gt_score_batch = (seal_label_score * wrench_label_score).detach().cpu()
            all_GT_scores.append(gt_score_batch)
            all_GT_collision.append(collision_label.detach().cpu())
            

# 현재 배치의 예측 점수(gamma) 추출
            units = pred_score_logit.shape[-1] // 4
            gamma, v, alpha, beta = torch.split(pred_score_logit, units, dim=-1)            
            gamma_vals = gamma.detach().cpu().numpy().flatten()
            eps = 1e-6
            
            # 값들을 numpy로 변환
            gamma_vals = gamma.detach().cpu().numpy().flatten()
            # Aleatoric: beta / (alpha - 1)
            ale_vals = (beta / (alpha - 1 + eps)).detach().cpu().numpy().flatten()
            # Epistemic: beta / (v * (alpha - 1))
            epi_vals = (beta / (v * (alpha - 1 + eps) + eps)).detach().cpu().numpy().flatten()
            
            # 배치별 데이터 저장 (Uncertainty 키값 추가)
            gt_score_batch = (seal_label_score * wrench_label_score).detach().cpu()
            gt_vals = gt_score_batch.numpy().flatten() # GT 값 추출
            coll_vals = collision_label.detach().cpu().numpy().flatten()
            # 배치별 데이터 저장 (GT_Score 추가)
            for g, e, a, gt, c in zip(gamma_vals, epi_vals, ale_vals, gt_vals, coll_vals):
    #            for g, e, a, gt in zip(gamma_vals, epi_vals, ale_vals, gt_vals):
                batch_stats.append({
                    'Batch_ID': f'B{start:02d}', 
                    'Pred_Score': g,
                    'GT_Score': gt,
                    'Collision': c,  # <--- 충돌 점수 추가
                    'Epistemic': e,
                    'Aleatoric': a
                })
            
            # # 배치별 데이터 저장 (Batch ID와 예측값)
            # for val in gamma_vals:
            #     batch_stats.append({'Batch_ID': f'B{start:02d}', 'Pred_Score': val})
            start +=1

            torch.cuda.empty_cache()  # optional but helpful

        pred_score_logit = torch.cat(all_pred_scores, dim=0)   # (B_total, N, 1)
        #pred_collision_logit = torch.cat(all_pred_collision, dim=0)
        GT_score_logit = torch.cat(all_GT_scores, dim=0)   # (B_total, N, 1)
        GT_collision_logit = torch.cat(all_GT_collision, dim=0)  

        units = pred_score_logit.shape[-1] // 4
        gamma, v, alpha, beta = torch.split(pred_score_logit, units, dim=-1)

        total_evidence = 2*v+alpha
        ######################################3
        """
        점수를 얼마나 정교하게 예측했는지가 evidence와 관련이 있는가?(not collision)
        """
        gamma_np = gamma.view(-1, 1).cpu().numpy()
        evidence_np = total_evidence.view(-1, 1).cpu().numpy()

        gt_score_np = GT_score_logit.view(-1, 1).cpu().numpy()
        GT_collision_np = GT_collision_logit.view(-1, 1).cpu().numpy()

        # 2. Min-Max 스케일러 생성 및 적용
        scaler = MinMaxScaler()

        # 점수 정규화 (보통 이미 0~1 사이라면 생략 가능)
        gamma_norm = scaler.fit_transform(gamma_np)

        # Evidence 정규화 (값이 크므로 필수!)
        # 로그를 먼저 취하고 정규화하는 것을 권장합니다.
        evidence_log = np.log1p(evidence_np)
        evidence_norm = scaler.fit_transform(evidence_log)

        # 3. 정규화된 데이터로 상관관계 분석
        #correlation = np.corrcoef(gamma_norm.flatten(), evidence_norm.flatten())[0, 1]
        #print(f"정규화 후 상관계수: {correlation:.4f}")

                # 데이터 준비 (정규화된 값 가정)
        error = np.abs(gamma_np - gt_score_np) # 예측 오차
        correlation = np.corrcoef(error.flatten(), evidence_norm.flatten())[0, 1]

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=evidence_norm.flatten(), y=error.flatten(), alpha=0.1, s=10)

        # 추세선(Regression line) 추가 - 데이터의 흐름을 한눈에 보여줍니다.
        sns.regplot(x=evidence_norm.flatten(), y=error.flatten(), scatter=False, color='red')

        plt.xlabel('Normalized Evidence (Confidence)')
        plt.ylabel('Prediction Error (|Pred - GT|)')
        plt.title(f'Scatter Plot: Evidence vs Error (Corr: {correlation:.4f})')
        plt.grid(True, linestyle='--', alpha=0.5)
        file_name = f'analysis_scene{scene_idx:03d}_anno{anno_idx:03d}.png'
        plt.savefig(os.path.join(score_evidence_path, file_name), dpi=150) # dpi를 조절해 화질 설정 가능
        plt.close()
        #plt.show()
        print(f'[Saved] {file_name} (Corr: {correlation:.4f})')

        ##############################################################################
        plt.figure(figsize=(10, 6))

        # x축: 예측 점수(gamma), y축: Evidence
        sns.scatterplot(x=gamma_norm.flatten(), y=evidence_norm.flatten(), alpha=0.1, s=10, color='purple')

        # 경향성을 보기 위한 2차식(Order=2) 추세선 추가 (U자형 곡선 확인용)
        sns.regplot(x=gamma_norm.flatten(), y=evidence_norm.flatten(), 
                    scatter=False, order=2, color='green', line_kws={'label': 'Trend Line'})

        plt.xlabel('Predicted Score (Gamma)')
        plt.ylabel('Normalized Evidence (Confidence)')
        plt.title(f'Scene: {scene_idx}, Anno: {anno_idx} | Score vs Evidence')
        plt.grid(True, linestyle='--', alpha=0.5)

        # 파일 저장 (이름 구분)
        file_name_score = f'score_vs_evid_scene{scene_idx:03d}_anno{anno_idx:03d}.png'
        plt.savefig(os.path.join(score_evidence_path2, file_name_score), dpi=150)
        plt.close()

        ################################################################
        eps = 1e-6
        # 1. Aleatoric (데이터의 통계적 노이즈 - 예: 물체의 반사, 센서 노이즈)
        aleatoric_np = (beta / (alpha - 1 + eps)).view(-1, 1).cpu().numpy()

        # 2. Epistemic (모델의 지식 부족 - 예: 처음 보는 물체, 학습 부족)
        epistemic_np = (beta / (v * (alpha - 1 + eps) + eps)).view(-1, 1).cpu().numpy()

        # 3. Total Uncertainty
        total_unc_np = aleatoric_np + epistemic_np

        # 분석을 위해 MinMaxScaler 적용 (동일한 0~1 스케일로 비교)
        aleatoric_norm = scaler.fit_transform(np.log1p(aleatoric_np))
        epistemic_norm = scaler.fit_transform(np.log1p(epistemic_np))
        total_unc_norm = scaler.fit_transform(np.log1p(total_unc_np))


        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        uncertainties = [
            (aleatoric_norm, 'Aleatoric (Data Noise)', 'orange'),
            (epistemic_norm, 'Epistemic (Model Knowledge)', 'blue'),
            (total_unc_norm, 'Total Uncertainty', 'red')
        ]

        for i, (unc_data, title, color) in enumerate(uncertainties):
            ax = axes[i]
            # 산점도
            ax.scatter(gamma_norm.flatten(), unc_data.flatten(), alpha=0.05, s=5, color=color)
            # 추세선 (2차식으로 U자 곡선 확인)
            sns.regplot(x=gamma_norm.flatten(), y=unc_data.flatten(), 
                        scatter=False, order=2, ax=ax, color='black', line_kws={'linewidth': 2})
            
            ax.set_xlabel('Predicted Score (Gamma)')
            ax.set_ylabel('Normalized Uncertainty')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Scene: {scene_idx}, Anno: {anno_idx} | Uncertainty Analysis', fontsize=16)
        plt.tight_layout()

        # 저장
        file_name_unc = f'uncertainty_analysis_scene{scene_idx:03d}_anno{anno_idx:03d}.png'
        plt.savefig(os.path.join(score_uncertainty_path, file_name_unc), dpi=150)
        plt.close()

        ######################################################################3
        masked_gt_score = gt_score_np * (1 - GT_collision_np)
        
        pred_flat = gamma_np.flatten()
        gt_masked_flat = (gt_score_np * (1 - GT_collision_np)).flatten()
        coll_flat = GT_collision_np.flatten()

        # 충돌 여부에 따른 마스크 생성 (0.5 기준으로 안전/위험 분리)
        mask_safe = coll_flat < 0.5
        mask_coll = coll_flat >= 0.5

        plt.figure(figsize=(10, 8))

        # 2. 안전 지점(Safe) 먼저 그리기: 투명도를 낮춰서(0.15) 배경처럼 만듦
        plt.scatter(
            pred_flat[mask_safe], 
            gt_masked_flat[mask_safe], 
            c='dodgerblue', 
            label='Safe (Normal)', 
            alpha=0.15, 
            s=15, 
            edgecolor='none'
        )

        # 3. 충돌 지점(Collision) 나중에 그리기: 불투명(1.0)하고 빨간색 'X'로 강조
        # 이 부분이 나중에 호출되므로 무조건 파란색 점 위에 찍힙니다.
        if np.any(mask_coll): # 충돌 데이터가 있을 때만 실행
            plt.scatter(
                pred_flat[mask_coll], 
                gt_masked_flat[mask_coll], 
                c='red', 
                marker='x', 
                label='Collision (Danger)', 
                alpha=1.0, 
                s=40, 
                linewidths=1.5
            )

        # 4. 가이드라인 및 서식 설정
        plt.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=0.6, label='Ideal (y=x)')
        
        plt.xlabel('Predicted Score (Gamma)')
        plt.ylabel('Masked GT Score (GT * (1-Collision))')
        plt.title(f'Scene: {scene_idx}, Anno: {anno_idx} | Collision Analysis (Forced Red)')
        plt.legend(loc='upper left', frameon=True)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 1.02)

        # 5. 저장
        file_name_final = f'final_masked_scene{scene_idx:03d}_anno{anno_idx:03d}.png'
        plt.savefig(os.path.join(pred_vs_gt_path, file_name_final), dpi=150)
        plt.close() # 메모리 해제 필수

        print(f'[Saved] {file_name_final} with {np.sum(mask_coll)} collision')
        ##########################################################################3
        # --- 라벨 기준 분포 분석 시작 ---
        dist_path = os.path.join(dump_dir, 'label_distribution')
        os.makedirs(dist_path, exist_ok=True)

        # 데이터 카테고리 분류 (라벨 기준)
        # 1. 실제 충돌 지점 (Collision GT == 1)
        # 2. 실제 비흡착 지점 (Score GT == 0 & Collision == 0)
        # 3. 실제 흡착 가능 지점 (Score GT > 0 & Collision == 0)
        
        is_collision = (GT_collision_logit.flatten() > 0.5).cpu().numpy()
        is_zero_score = (GT_score_logit.flatten() == 0).cpu().numpy()
        is_suctionable = (~is_collision & ~is_zero_score)
        is_pure_zero = (~is_collision & is_zero_score)

        # 시각화를 위한 데이터프레임 생성
        plot_df = pd.DataFrame({
            'Predicted_Score': gamma_np.flatten(),
            'Category': 'None'
        })
        plot_df.loc[is_collision, 'Category'] = 'Actual Collision'
        plot_df.loc[is_pure_zero, 'Category'] = 'Actual Score 0 (Safe)'
        plot_df.loc[is_suctionable, 'Category'] = 'Actual Positive Score'

        # 1. 히스토그램 (예측값의 분포 확인)
        plt.figure(figsize=(12, 6))
        for cat in plot_df['Category'].unique():
            subset = plot_df[plot_df['Category'] == cat]
            sns.histplot(subset['Predicted_Score'], label=cat, kde=True, element="step", alpha=0.3)
        
        plt.title(f'Scene {scene_idx} Anno {anno_idx}: Prediction Distribution by GT Category')
        plt.xlabel('Model Predicted Score (Gamma)')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(os.path.join(dist_path, f'dist_hist_scene{scene_idx:03d}_{anno_idx:03d}.png'))
        plt.close()

        # 2. 바이올린 플롯 (각 카테고리별 예측값의 밀도 비교)
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Category', y='Predicted_Score', data=plot_df, inner="quart")
        plt.title(f'Scene {scene_idx} Anno {anno_idx}: Prediction Confidence by GT')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(dist_path, f'dist_violin_scene{scene_idx:03d}_{anno_idx:03d}.png'))
        plt.close()
        #########################3
        """
        순수 점수 상관관계 분석 (Collision 배제)
        """
        pure_score_corr_path = os.path.join(dump_dir, 'pure_score_correlation')
        os.makedirs(pure_score_corr_path, exist_ok=True)

        # 1. 데이터 준비 (numpy flat)
        gt_flat = gt_score_np.flatten()
        pred_flat = gamma_np.flatten()

        # 2. 상관계수 계산
        pure_correlation = np.corrcoef(gt_flat, pred_flat)[0, 1]

        # 3. 시각화
        plt.figure(figsize=(8, 8))
        # 산점도 (투명도를 주어 밀도 확인)
        plt.scatter(gt_flat, pred_flat, alpha=0.1, s=10, color='teal')

        # 추세선 (y=x 대각선)
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Ideal (y=x)')

        # 회귀선 (실제 데이터 경향)gt_flat
        sns.regplot(x=pred_flat, y=gt_flat, scatter=False, color='darkblue', label='Regression Line')

        plt.ylabel('Ground Truth Score (Pure)')
        plt.xlabel('Predicted Score (Gamma)')
        plt.title(f'Pure Score Correlation: {pure_correlation:.4f}\n(Scene {scene_idx}, Anno {anno_idx})')
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 저장
        file_name_pure = f'pure_corr_scene{scene_idx:03d}_anno{anno_idx:03d}.png'
        plt.savefig(os.path.join(pure_score_corr_path, file_name_pure), dpi=150)
        plt.close()

        print(f'[Saved] {file_name_pure} (Corr: {pure_correlation:.4f})')

        ##############################################################################
        """
        예측 점수(Gamma) 기준 구간별(0.2 단위) 상관관계 분석
        """
        bin_path = os.path.join(dump_dir, 'score_bins_analysis')
        os.makedirs(bin_path, exist_ok=True)

        # 1. 구간(Bin) 설정
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']

        # 데이터 준비
        pred_f = gamma_norm.flatten()  # 0~1 정규화된 예측값
        gt_f = gt_score_np.flatten()
        epi_f = epistemic_norm.flatten()
        ale_f = aleatoric_norm.flatten()

        fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
        fig.suptitle(f'Scene {scene_idx} Anno {anno_idx}: Analysis by Predicted Score Intervals', fontsize=20)

        for i in range(len(bins)-1):
            # 해당 구간에 속하는 인덱스 추출
            mask = (pred_f >= bins[i]) & (pred_f < bins[i+1])
            
            if np.sum(mask) == 0:
                axes[i].set_title(f'Bin {labels[i]}\n(No Data)')
                continue

            # 구간 데이터 추출
            curr_gt = gt_f[mask]
            curr_pred = pred_f[mask]
            curr_epi = epi_f[mask]
            
            # 산점도: 색상은 Epistemic Uncertainty로 지정
            sc = axes[i].scatter(curr_pred, curr_gt, c=curr_epi, cmap='viridis', alpha=0.3, s=15)
            
            # 구간 내 통계치 계산
            avg_error = np.mean(np.abs(curr_gt - curr_pred))
            avg_epi = np.mean(curr_epi)
            
            axes[i].set_title(f'Bin {labels[i]}\nPoints: {np.sum(mask)}\nAvg Err: {avg_error:.3f}')
            axes[i].set_ylabel('GT Score')
            if i == 0: axes[i].set_xlabel('Predicted Score')
            
            # 기준선
            axes[i].plot([0, 1], [0, 1], 'r--', alpha=0.5)
            axes[i].set_xlim(-0.05, 1.05)
            axes[i].set_ylim(-0.05, 1.05)
            axes[i].grid(True, alpha=0.2)

        # 컬러바 추가 (오른쪽 끝에 하나만)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
        fig.colorbar(sc, cax=cbar_ax, label='Epistemic Uncertainty')

        # 저장
        file_name_bin = f'bins_analysis_scene{scene_idx:03d}_anno{anno_idx:03d}.png'
        plt.savefig(os.path.join(bin_path, file_name_bin), dpi=150, bbox_inches='tight')
        plt.close()

        print(f'[Saved] {file_name_bin} (Binned Analysis)')

        ###########################################################     
        batch_df = pd.DataFrame(batch_stats)
        # target_unc = 'Epistemic'
        
        # # 1. FacetGrid 설정 (제목 크기를 줄이기 위해 margin_titles나 despine 활용 가능)
        g = sns.FacetGrid(batch_df, col="Batch_ID", col_wrap=5, height=3.5)

        batch_detail_path = os.path.join(dump_dir, 'batch_individual_plots', f'scene_{scene_idx:03d}_anno_{anno_idx:03d}')
        os.makedirs(batch_detail_path, exist_ok=True)

        #from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

        # 전체 데이터에 대해 정규화된 Uncertainty 계산
        batch_df['Epi_norm'] = scaler.fit_transform(batch_df[['Epistemic']])
        batch_df['Ale_norm'] = scaler.fit_transform(batch_df[['Aleatoric']])
        batch_df['Total_Sum'] = batch_df['Epistemic'] + batch_df['Aleatoric']
        batch_df['Total_Unc'] = scaler.fit_transform(batch_df[['Total_Sum']])

        # --- 보정 점수 계산 (Calibration) ---
        batch_df['Score_Epi_Calib'] = batch_df['Pred_Score'] * (1 - batch_df['Epi_norm'])
        batch_df['Score_Ale_Calib'] = batch_df['Pred_Score'] * (1 - batch_df['Ale_norm'])
        batch_df['Score_Total_Calib'] = batch_df['Pred_Score'] * (1 - batch_df['Total_Unc'])
        batch_df['GT_Score_Safe'] = batch_df['GT_Score'] * (1 - batch_df['Collision'])
        batch_results = []
        comp_results = []
        comp_results_safe = []

        epi_threshold = batch_df['Epistemic'].quantile(0.9)

        # 상위 10%에 해당하면 1, 아니면 0인 마스크 생성
        # (1 - mask)를 곱하면 상위 10% 데이터의 점수는 강제로 0이 됩니다.
        batch_df['Epi_Hard_Mask'] = (batch_df['Epistemic'] >= epi_threshold).astype(float)
        batch_df['Score_Epi_Hard_Calib'] = batch_df['Pred_Score'] * (1 - batch_df['Epi_Hard_Mask'])

        # 분석할 k값 설정 (예: 50)
        target_k = 50
        for b_id in batch_df['Batch_ID'].unique():
            df_b = batch_df[batch_df['Batch_ID'] == b_id]
            
            # 레이아웃 수정: 2행 3열 (상단 산점도 3개, 하단 산점도 1개 + 에러 히스토그램 1개 + 표 1개)
            fig = plt.figure(figsize=(22, 12))
            gs = fig.add_gridspec(2, 3)
            
            # 각 그래프 설정 (컬럼명, 제목, 위치, 색상)
            plot_configs = [
                ('Pred_Score', 'Raw Score', gs[0, 0], 'gray'),
                ('Score_Epi_Calib', 'Epi-Soft Calib', gs[0, 1], 'purple'),
                ('Score_Epi_Hard_Calib', 'Epi-Hard Calib', gs[0, 2], 'orange'),
                ('Score_Total_Calib', 'Total Calibrated', gs[1, 0], 'darkred')
            ]

            for col, title, pos, color in plot_configs:
                if col not in df_b.columns: continue
                ax = fig.add_subplot(pos)
                ax.scatter(df_b[col], df_b['GT_Score'], c=color, alpha=0.4, s=20)
                ax.plot([0, 1], [0, 1], 'r--', alpha=0.5) # 대각선 기준선
                
                corr = df_b[[col, 'GT_Score']].corr().iloc[0, 1]
                ax.set_title(f'[{title}]\nCorr: {corr:.3f}', fontsize=13, fontweight='bold')
                ax.grid(True, linestyle=':', alpha=0.5)

            # (B) 하단 중앙: 에러 분포 히스토그램 (루프 밖에서 한 번만 실행)
            ax_hist = fig.add_subplot(gs[1, 1])
            raw_err = np.abs(df_b['Pred_Score'] - df_b['GT_Score'])
            hard_err = np.abs(df_b['Score_Epi_Hard_Calib'] - df_b['GT_Score'])
            
            sns.histplot(raw_err, color="gray", label="Raw Error", kde=True, ax=ax_hist, alpha=0.3)
            sns.histplot(hard_err, color="orange", label="Epi-Hard Error", kde=True, ax=ax_hist, alpha=0.5)
            ax_hist.set_title("Error Distribution: Raw vs Hard Filter", fontsize=13, fontweight='bold')
            ax_hist.legend()
            ax_table = fig.add_subplot(gs[1, 2])
            ax_table.axis('off')

            stats_data = [
                ["Metric", "Raw", "Epi-Soft", "Epi-Hard", "Total"],
                ["Mean Err", 
                 f"{np.mean(raw_err):.3f}",
                 f"{np.mean(np.abs(df_b['Score_Epi_Calib']-df_b['GT_Score'])):.3f}",
                 f"{np.mean(hard_err):.3f}",
                 f"{np.mean(np.abs(df_b['Score_Total_Calib']-df_b['GT_Score'])):.3f}"],
                ["Corr", 
                 f"{df_b[['Pred_Score', 'GT_Score']].corr().iloc[0,1]:.3f}",
                 f"{df_b[['Score_Epi_Calib', 'GT_Score']].corr().iloc[0,1]:.3f}",
                 f"{df_b[['Score_Epi_Hard_Calib', 'GT_Score']].corr().iloc[0,1]:.3f}",
                 f"{df_b[['Score_Total_Calib', 'GT_Score']].corr().iloc[0,1]:.3f}"]
            ]
            # --- 성능 비교 표 ---
            ax_table = fig.add_subplot(gs[1, 2])
            ax_table.axis('off')
            # (기존 stats_data 및 table 생성 코드 그대로 사용)
            table = ax_table.table(cellText=stats_data, loc='center', cellLoc='center')
            table.set_fontsize(12)
            table.scale(1, 2.5)
            
            plt.suptitle(f'Calibration Detailed Analysis: {b_id} (Scene {scene_idx})', fontsize=20, y=0.98)
            plt.tight_layout()
            
            save_name = f'calibration_{b_id}.png'
            plt.savefig(os.path.join(batch_detail_path, save_name), dpi=150)
            plt.close()


            # (A) 원본 점수(Pred_Score)로 분석
            # 4가지 케이스에 대해 각각 Precision 계산
            method_configs = [
                ('Original', 'Pred_Score', False),
                ('Epi-Soft', 'Score_Epi_Calib', False),
                ('Epi-Hard', 'Score_Epi_Hard_Calib', False), # 이전 대화에서 언급된 Hard 보정
                ('Tiered-Epi', 'Score_Epi_Calib', True),     # <--- 200개 중 50개 뽑기 추가
                ('Total-Calib', 'Score_Total_Calib', False)
            ]
            
            for label, col_name, is_tiered in method_configs:
                if col_name in df_b.columns:
                    # 1. Standard GT 기준
                    if is_tiered:
                        acc_std = get_tiered_top_k_precision(df_b, 'Pred_Score', col_name, gt_col='GT_Score', k_initial=200, k_final=target_k)
                    else:
                        acc_std = get_top_k_precision(df_b, col_name, gt_col='GT_Score', k=target_k)
                    
                    comp_results.append({'Batch_ID': b_id, 'Method': label, 'Type': 'Standard', 'Precision': acc_std})
                    
                    # 2. Safe GT 기준 (충돌 고려)
                    if is_tiered:
                        acc_safe = get_tiered_top_k_precision(df_b, 'Pred_Score', col_name, gt_col='GT_Score_Safe', k_initial=200, k_final=target_k)
                    else:
                        acc_safe = get_top_k_precision(df_b, col_name, gt_col='GT_Score_Safe', k=target_k)
                        
                    comp_results.append({'Batch_ID': b_id, 'Method': label, 'Type': 'Safe', 'Precision': acc_safe})

        comp_df = pd.DataFrame(comp_results)


        diag_path = os.path.join(dump_dir, 'uncertainty_diagnostic')

        # 3. 폴더가 없으면 새로 만들기
        os.makedirs(diag_path, exist_ok=True)

        # 4. 해당 경로에 파일 저장
        plt.savefig(os.path.join(diag_path, f'diag_scatter_scene{scene_idx:03d}_{anno_idx}.png'))

                # 두 가지 버전의 리포트 생성
        save_bar_chart(comp_df, 'Standard', f'compare_std_scene{scene_idx:03d}_{anno_idx}.png',target_k=target_k)
        #save_bar_chart(comp_df, 'Safe', f'compare_safe_scene{scene_idx:03d}_{anno_idx}.png',target_k=target_k)


        # # ==========================================================
        # # --- 2. Uncertainty 진단 (Global Diagnostics) ---
        # # ==========================================================
        # # (A) Scatter: Error vs Uncertainty (모델이 자기 오차를 아는가?)
        # plt.figure(figsize=(14, 6))
        # error = np.abs(batch_df['Pred_Score'] - batch_df['GT_Score'])

        # # Epistemic과 Aleatoric을 동시에 확인 (서브플롯)
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # sns.regplot(x=batch_df['Epistemic'], y=error, ax=ax1, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
        # ax1.set_title('Error vs Epistemic (Model Knowledge)')

        # sns.regplot(x=batch_df['Aleatoric'], y=error, ax=ax2, scatter_kws={'alpha':0.3, 'color':'orange'}, line_kws={'color':'red'})
        # ax2.set_title('Error vs Aleatoric (Data Noise)')

        # plt.suptitle(f'Uncertainty-Error Correlation (Scene {scene_idx})', fontsize=16)
        # os.makedirs(diag_path, exist_ok=True)
        # plt.savefig(os.path.join(diag_path, f'diag_scatter_scene{scene_idx:03d}_{anno_idx}.png'))
        # plt.close()


        # # (B) Sparsification Plot: "불확실한 놈 버리기" 효과
        # # ----------------------------------------------------------
        # # 불확실성(Epistemic) 기준으로 전체 데이터 정렬
        # sorted_df = batch_df.sort_values(by='Epistemic')
        # fractions = np.linspace(0.1, 1.0, 10) # 10%부터 100%까지
        # filter_results = []

        # for frac in fractions:
        #     cutoff = int(len(sorted_df) * frac)
        #     if cutoff < 5: continue
            
        #     temp_df = sorted_df.iloc[:cutoff] # 불확실성이 낮은(확실한) 상위 frac%만 유지
        #     mean_err = np.abs(temp_df['Pred_Score'] - temp_df['GT_Score']).mean()
        #     filter_results.append({'Fraction': frac, 'Mean Error': mean_err})

        # plt.figure(figsize=(10, 6))
        # res_df = pd.DataFrame(filter_results)
        # sns.lineplot(data=res_df, x='Fraction', y='Mean Error', marker='o', color='green', linewidth=2)

        # plt.gca().invert_xaxis() # 오른쪽으로 갈수록 데이터를 많이 버림(확실한 놈만 남음)
        # plt.title('Sparsification Plot: Error reduction by Filtering', fontsize=14)
        # plt.xlabel('Fraction of Data Kept (1.0 = All, 0.2 = Only the Most Confident)')
        # plt.grid(True, alpha=0.3)
        # plt.savefig(os.path.join(diag_path, f'diag_filtering_scene{scene_idx:03d}_{anno_idx}.png'))
        # plt.close()

        print(f"🔎 [Analysis Complete] Scene {scene_idx} diagnostics saved to {diag_path}")

if __name__ == "__main__":
    scene_list = []
    valid_data=PT_dataset(dataset_root,split=split,camera='realsense', use_color=False)

    if split == 'test':
        for i in range(100, 190):
            scene_list.append(i)
    elif split == 'test_seen':
        for i in range(100, 130):
            scene_list.append(i)
    elif split == 'test_similar':
        for i in range(130, 160):
            scene_list.append(i)
    elif split == 'test_novel':
        for i in range(160, 190):
            scene_list.append(i)
    else:
        print('invalid split')
        exit(1)

    for scene_idx in scene_list:
        print(f"Processing scene {scene_idx}...")
        data_analyze(scene_idx,valid_data)

