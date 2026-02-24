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
parser.add_argument('--split', default='test_similar', help='dataset split [default: test_seen]')
parser.add_argument('--camera', default='realsense', help='camera to use [default: realsense]')
parser.add_argument('--dump_dir', default='mpt_evidence', help='where to save')
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

def _format_tensor_info(tensor, max_samples=3):
    """Format tensor information for debugging (shape + first few samples)"""
    if tensor is None or isinstance(tensor, str):
        return str(tensor)
    if isinstance(tensor, (torch.Tensor, np.ndarray)):
        shape = tensor.shape
        if len(tensor) > 0:
            samples = tensor[:max_samples] if len(tensor) > max_samples else tensor
            return f"{shape} (first {min(max_samples, len(tensor))}: {samples.tolist() if len(samples) <= max_samples else '...'})"
        else:
            return f"{shape} (empty)"
    return str(tensor)

def _get_shape(data_dict, key):
    """Helper function to get shape safely"""
    val = data_dict.get(key)
    if val is None:
        return 'N/A'
    if isinstance(val, (np.ndarray, torch.Tensor)):
        return val.shape
    return type(val).__name__

import pandas as pd
def visualization(scene_idx,anno_idx,gamma,evidence_uncertainty):
    raw_scores_np = torch.clamp(gamma.squeeze(-1), 0, 1).detach().cpu().numpy().flatten()
    unc_np = evidence_uncertainty.detach().cpu().numpy().flatten()

    # 2. 보정 점수 계산 (LCB 방식 or 곱하기 방식)
    # 여기서는 곱하기 방식 사용 (원하시는 방식대로 수정 가능)
    calibrated_scores_np = raw_scores_np * (1 - unc_np)

    # 3. 데이터프레임 생성
    df = pd.DataFrame({
        'Raw_Score': raw_scores_np,
        'Cal_Score': calibrated_scores_np,
        'Uncertainty': unc_np
    })

    # 4. 구간(Bin) 설정 (0.5부터 1.0까지 0.1 단위)
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['0.0-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']

    # 5. 각 데이터가 어느 구간에 속하는지 라벨링
    # (1) Raw Score 기준 구간
    df['Bin_Raw'] = pd.cut(df['Raw_Score'], bins=bins, labels=labels, include_lowest=True)
    # (2) Calibrated Score 기준 구간
    df['Bin_Cal'] = pd.cut(df['Cal_Score'], bins=bins, labels=labels, include_lowest=True)

    # 6. 통계 집계 (Pivot Table)
    # Raw 기준 통계
    stats_raw = df.groupby('Bin_Raw', observed=False).agg(
        Count_Raw=('Uncertainty', 'count'),
        Avg_Unc_Raw=('Uncertainty', 'mean')
    )

    # Calibrated 기준 통계
    stats_cal = df.groupby('Bin_Cal', observed=False).agg(
        Count_Cal=('Uncertainty', 'count'),
        Avg_Unc_Cal=('Uncertainty', 'mean')
    )

    # 7. 두 표 합치기 (Join)
    comparison_table = stats_raw.join(stats_cal)

    # 8. 보기 좋게 출력
    print(f"\n=== [Scene {scene_idx} / Anno {anno_idx}] Score Calibration Effect ===")
    print(f"{'Score Range':<12} | {'[BEFORE] Raw Score':^22} | {'[AFTER] Calibrated Score':^22}")
    print(f"{'':<12} | {'Count':>8}   {'Avg Unc':>10} | {'Count':>8}   {'Avg Unc':>10}")
    print("-" * 75)

    # 높은 점수 구간부터 출력 (역순)
    for label in labels[::-1]:
        row = comparison_table.loc[label]
        
        # Raw Data
        cnt_raw = int(row['Count_Raw'])
        unc_raw = row['Avg_Unc_Raw']
        unc_raw_str = f"{unc_raw:.4f}" if cnt_raw > 0 else "-"
        
        # Calibrated Data
        cnt_cal = int(row['Count_Cal'])
        unc_cal = row['Avg_Unc_Cal']
        unc_cal_str = f"{unc_cal:.4f}" if cnt_cal > 0 else "-"
        
        # 변화량 표시 (화살표)
        # 개수가 줄었다면 상위권에서 탈락했다는 뜻
        print(f"{label:<12} | {cnt_raw:8d}   {unc_raw_str:>10} | {cnt_cal:8d}   {unc_cal_str:>10}")

    print("-" * 75)
    print("Tip: [AFTER] 쪽의 상위 구간(0.9-1.0) Avg Unc가 현저히 낮아야 정상입니다.")



def inference(scene_idx,valid_data):
    infer_time_list = []
    for anno_idx in range(256):
        samples,pick_suction_points,pick_normal_points = valid_data.test_inference(scene_idx,anno_idx)
        
        chunk_size = 4  # or 5
        all_scores = []
        all_sigmas = []
        all_collision = []
        start = 0
        for chunk in chunk_list(samples, chunk_size):
            batch_data = unified_collate_fn(chunk)
            batch = to_device(batch_data, device)
            label = batch.pop("label")
            pick_feature = torch.cat([label['coord'], label['normal']], dim=-1)
            inputs = batch
            with torch.no_grad():
                pred_score_logit = net(inputs,pick_feature)
            #all_collision.append(collision_score.detach().cpu())
            all_scores.append(pred_score_logit.detach().cpu())
            #all_sigmas.append(pred_sigma_logit.detach().cpu())

            torch.cuda.empty_cache()  # optional but helpful


        pred_score_logit = torch.cat(all_scores, dim=0)   # (B_total, N, 1)
        #pred_collision_logit = torch.cat(all_collision, dim=0)  
        #pred_sigma_logit = torch.cat(all_sigmas, dim=0)

        units = pred_score_logit.shape[-1] // 4
        gamma, v, alpha, beta = torch.split(pred_score_logit, units, dim=-1)
        #evidence_uncertainty = units/(v+alpha)
        
        # # # 1. 알레아토리 불확실성 (데이터 노이즈)
        aleatoric = beta / (alpha - 1)

        # # # 2. 에피스테믹 불확실성 (모델의 확신도 - 지식 부족)
        epistemic = beta / (v * (alpha - 1))


        total_uncertainty = (aleatoric+epistemic)

        uncertainty = (total_uncertainty - total_uncertainty.min(dim=1, keepdim=True)[0]) \
            / (total_uncertainty.max(dim=1, keepdim=True)[0] - total_uncertainty.min(dim=1, keepdim=True)[0])

        #evidence_uncertainty = evidence_uncertainty.squeeze(-1)
        evidence_uncertainty = uncertainty.squeeze(-1)

        #scores_flat = pred_score_logit.squeeze(-1)
        scores = pred_score_logit[:, :, 0]
        #pred_coll = pred_collision_logit.squeeze(-1)
        #scores = np_pred_score.squeeze(-1)c
        total_score = scores #*(pred_coll > 0.5).float()
        topk_scores, topk_indices = torch.topk(total_score*(1-evidence_uncertainty), k=200, dim=1, largest=True, sorted=True)
        top_suction_indices = topk_indices.detach().cpu().numpy()

        if torch.is_tensor(pick_normal_points):
            pick_normal_points = pick_normal_points.numpy()
        if torch.is_tensor(pick_suction_points):
            pick_suction_points = pick_suction_points.numpy()

        B = scores.shape[0]
        batch_idx = np.arange(B)[:, None]
        # top_scores, top_indices = torch.topk(
        #     scores, k=top_k, dim=1, largest=True, sorted=True
        # )

        suction_arr = np.concatenate([
            scores[batch_idx, top_suction_indices][..., np.newaxis],      # (B, K, 1)
            pick_normal_points[batch_idx, top_suction_indices] ,
            pick_suction_points[batch_idx, top_suction_indices],          # (B, K, 3)
        ], axis=-1)

        suction_arr = suction_arr.reshape(-1, 7)

        if len(suction_arr)>0:
            suction_dir = os.path.join(dump_dir, split, 'scene_%04d'%scene_idx, camera, 'suction')
            os.makedirs(suction_dir, exist_ok=True)
            print('Saving:', suction_dir+'/%04d.npz'%anno_idx)
            np.savez(suction_dir+'/%04d.npz'%anno_idx, suction_arr)
        else:
            print(f"No suctions generated for scene {scene_idx}, annotation {anno_idx}")



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
        inference(scene_idx,valid_data)

