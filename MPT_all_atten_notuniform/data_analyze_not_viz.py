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
def get_top_k_precision(df, score_col, gt_col, k=50):
    """일반적인 Top-K Precision 계산"""
    if len(df) < k: return 0.0
    top_k_pred = set(df.nlargest(k, score_col).index)
    top_k_gt = set(df.nlargest(k, gt_col).index)
    return len(top_k_pred.intersection(top_k_gt)) / k

def get_tiered_top_k_precision(df, raw_col, calib_col, gt_col, k_initial=200, k_final=50):
    """200개 선별 후 50개 보정 선정 (Tiered Filtering)"""
    if len(df) < k_initial:
        return get_top_k_precision(df, calib_col, gt_col, k=k_final)
    # 1. 원본 점수로 200개 선점
    top_200_df = df.nlargest(k_initial, raw_col)
    # 2. 그 안에서 보정 점수로 50개 최종 선정
    top_50_pred = set(top_200_df.nlargest(k_final, calib_col).index)
    top_50_gt = set(df.nlargest(k_final, gt_col).index)
    return len(top_50_pred.intersection(top_50_gt)) / k_final

##########
TOTAL_SCENE_NUM = 190
from tqdm import tqdm
import open3d as o3d
import seaborn as sns
import pandas as pd

def data_analyze(scene_idx, valid_data):
    # 결과를 저장할 리스트
    comp_results = []
    all_anno_results = []
    # 저장 경로 설정
    analysis_save_path = os.path.join(dump_dir, 'summary_analysis')
    os.makedirs(analysis_save_path, exist_ok=True)
    # 256개의 Annotation 중 샘플로 1개(또는 루프) 분석
    for anno_idx in range(256): # 시연을 위해 0번만 수행 (필요시 range(256))
        samples, _, _ = valid_data.analyze_data(scene_idx, anno_idx)
        
        chunk_size = 4
        batch_stats = []
        anno_stats = []
        for start_idx, chunk in enumerate(chunk_list(samples, chunk_size)):
            batch_data = unified_collate_fn(chunk)
            batch = to_device(batch_data, device)
            label = batch.pop("label")
            pick_feature = torch.cat([label['coord'], label['normal']], dim=-1)
            
            # GT 계산
            seal = label['seal_score'].float().to(device)
            wrench = label['wrench_score'].float().to(device)
            coll = label.get('collision', torch.zeros_like(seal)).float().to(device)
            gt_score = (seal * wrench).detach().cpu().numpy().flatten()
            coll_np = coll.detach().cpu().numpy().flatten()

            with torch.no_grad():
                pred_logit = net(batch, pick_feature)

            # EDL 파라미터 분리 및 불확실성 계산
            units = pred_logit.shape[-1] // 4
            gamma, v, alpha, beta = torch.split(pred_logit, units, dim=-1)
            eps = 1e-6
            
            gamma_np = gamma.detach().cpu().numpy().flatten()
            ale_np = (beta / (alpha - 1 + eps)).detach().cpu().numpy().flatten()
            epi_np = (beta / (v * (alpha - 1 + eps) + eps)).detach().cpu().numpy().flatten()
            
            # 한 Chunk 내의 모든 포인트를 수집
            for i in range(len(gamma_np)):
                anno_stats.append({
                    'Pred_Score': gamma_np[i],
                    'Epistemic': epi_np[i],
                    'Aleatoric': ale_np[i],
                    'GT_Score': gt_score[i],
                    'Collision': coll_np[i]
                })
        
        # 2. 한 Annotation이 끝난 시점에 데이터프레임 변환 (통합 분석)
        df_anno = pd.DataFrame(anno_stats)
        df_anno['GT_Safe'] = df_anno['GT_Score'] * (1 - df_anno['Collision'])
        
        # 정규화 (이 Annotation 내에서 상대적 위치)
        scaler = MinMaxScaler()
        df_anno[['Epi_norm', 'Ale_norm']] = scaler.fit_transform(df_anno[['Epistemic', 'Aleatoric']])
        df_anno['Total_Sum'] = df_anno['Epistemic'] + df_anno['Aleatoric']
        df_anno['Total_Unc'] = scaler.fit_transform(df_anno[['Total_Sum']])
        
        # 보정 점수 계산
        df_anno['Score_Epi'] = df_anno['Pred_Score'] * (1 - df_anno['Epi_norm'])
        df_anno['Score_Ale'] = df_anno['Pred_Score'] * (1 - df_anno['Ale_norm'])
        df_anno['Score_Total'] = df_anno['Pred_Score'] * (1 - df_anno['Total_Unc'])
        
        target_k = 50
        q_90 = df_anno['Epistemic'].quantile(0.9)
        df_epi_hard = df_anno[df_anno['Epistemic'] < q_90]
        q_ale_90 = df_anno['Aleatoric'].quantile(0.9)
        df_ale_hard = df_anno[df_anno['Aleatoric'] < q_ale_90]
        q_total_90 = df_anno['Total_Unc'].quantile(0.9)
        df_total_hard = df_anno[df_anno['Total_Unc'] < q_total_90]

        results = {
            'Original': get_top_k_precision(df_anno, 'Pred_Score', 'GT_Safe', k=target_k),
            'Epi-Soft': get_top_k_precision(df_anno, 'Score_Epi', 'GT_Safe', k=target_k),
            'Ale-Soft': get_top_k_precision(df_anno, 'Score_Ale', 'GT_Safe', k=target_k),
            'Total-Soft': get_top_k_precision(df_anno, 'Score_Total', 'GT_Safe', k=target_k),
            'Epi-Hard': get_top_k_precision(df_epi_hard, 'Pred_Score', 'GT_Safe', k=target_k),
            'Tiered-Epi': get_tiered_top_k_precision(df_anno, 'Pred_Score', 'Score_Epi', 'GT_Safe', 200, 50),
            'Ale-Hard': get_top_k_precision(df_ale_hard, 'Pred_Score', 'GT_Safe', k=target_k),
            'Tiered-ale': get_tiered_top_k_precision(df_anno, 'Pred_Score', 'Score_Ale', 'GT_Safe', 200, 50),
            'Total-Hard': get_top_k_precision(df_total_hard, 'Pred_Score', 'GT_Safe', k=target_k),
            'Tiered-total': get_tiered_top_k_precision(df_anno, 'Pred_Score', 'Score_Total', 'GT_Safe', 200, 50),
        }

        for m_name, val in results.items():
            all_anno_results.append({
                'Scene': scene_idx, 
                'Anno': anno_idx, 
                'Method': m_name, 
                'Precision': val
            })

# --- 2. [Bar Chart 시각화] Scene 전체 결과 종합 ---
    final_df = pd.DataFrame(all_anno_results)
    
    if not final_df.empty:
# 1. [Scene-wise Anno Trend] 추가
        plt.figure(figsize=(15, 6))
        sns.lineplot(data=final_df, x='Anno', y='Precision', hue='Method')
        plt.title(f'Scene {scene_idx}: Precision Trend Across Annotations', fontsize=14)
        plt.grid(True, alpha=0.3); plt.ylim(-0.05, 1.05)
        plt.savefig(os.path.join(analysis_save_path, f'scene_{scene_idx:03d}_anno_trend.png'))
        plt.close()

        # 2. [Scene-wise Bar Chart]
        plt.figure(figsize=(15, 8))
        ax = sns.barplot(data=final_df, x='Method', y='Precision', palette='Set2', capsize=.1, errorbar='sd')
        for container in ax.containers: ax.bar_label(container, fmt='%.3f', padding=3, fontweight='bold')
        plt.title(f'Scene {scene_idx} Performance (Mean & SD)', fontsize=16)
        plt.savefig(os.path.join(analysis_save_path, f'scene_{scene_idx:03d}_summary.png'), dpi=150)
        plt.close()

    return final_df




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

    final_dfs = []
    for scene_idx in scene_list:
        print(f"Processing scene {scene_idx}...")
        scene_data = data_analyze(scene_idx,valid_data)
        scene_ap = scene_data.groupby('Method')['Precision'].mean().reset_index()
        scene_ap['Scene_ID'] = scene_idx
        final_dfs.append(scene_ap)

# --- [전체 Scene에 대한 AP 추이] ---
    global_df = pd.concat(final_dfs, ignore_index=True)
    plt.figure(figsize=(18, 7))
    sns.lineplot(data=global_df, x='Scene_ID', y='Precision', hue='Method', marker='o')
    plt.title(f'Global Performance Trend: Average Precision per Scene ({split})', fontsize=16)
    plt.grid(True, alpha=0.3); plt.ylim(-0.05, 1.05)
    global_trend_path = os.path.join(dump_dir, 'summary_analysis', 'global_scene_trend.png')
    plt.savefig(global_trend_path, dpi=150); plt.close()

    # --- [최종 결과 요약 및 표 시각화] ---
    # 1. 콘솔 출력
    final_ap_summary = global_df.groupby('Method')['Precision'].mean().sort_values(ascending=False).reset_index()
    final_ap_summary.columns = ['Method', 'mAP (Mean of Scenes)']
    print("\n" + "="*45)
    print(f" FINAL EVALUATION RESULTS (Split: {split})")
    print("="*45)
    print(final_ap_summary.to_string(index=False))
    print("="*45)

    # 2. 표 형태의 이미지 저장
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    tbl = ax.table(cellText=final_ap_summary.values, colLabels=final_ap_summary.columns, 
                   cellLoc='center', loc='center', colColours=["#f2f2f2"]*2)
    tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1.2, 2)
    plt.title(f'Final mAP Summary ({split})', fontsize=15, pad=20)
    table_path = os.path.join(dump_dir, 'summary_analysis', 'final_ap_table.png')
    plt.savefig(table_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"✅ 완료! 표 이미지는 {table_path}에 저장되었습니다.")