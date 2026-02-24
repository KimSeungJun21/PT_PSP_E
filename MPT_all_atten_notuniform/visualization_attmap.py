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

import open3d as o3d
import numpy as np
import matplotlib
matplotlib.use('Agg')  # <--- 이 줄이 핵심입니다. GUI를 아예 사용하지 않겠다고 선언함
import matplotlib.pyplot as plt

import argparse


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

minimum_num_pt = 50
num_pt = 1024
width = 1280
height = 720
suction_height = 0.1
suction_radius = 0.01
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#default_path = os.path.join(BASE_DIR, 'runs_PT', 'latest_model.pth')
default_path = '/home/kimseungjun/task/PointTransformer/MPT_custommodel/runs_PT/best_model_loss.pth'

parser = argparse.ArgumentParser()
parser.add_argument('--split', default='test_seen', help='dataset split [default: test_seen]')
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

def save_attention_pcd(input_dict, attn_results, filename="attn_vis.ply", batch_idx=0, pick_idx=10):
    weights, indices, _ = attn_results
    
    # 1. Scene 데이터 (전체)
    scene_xyz = input_dict['scene']['coord'].cpu().numpy()
    
    # 2. Target 데이터 (따로 띄울 용도)
    target_coord = input_dict['target']['coord']
    target_batch = input_dict['target']['batch']
    target_mask = (target_batch == batch_idx)
    target_xyz = target_coord[target_mask].cpu().numpy()
    
    # [핵심] 여기서 Offset을 줍니다! 
    # Scene과 겹치지 않게 X축으로 2미터 밀어냅니다.
    target_xyz_offset = target_xyz + np.array([0.0, 0, 0]) 

    # 3. Attention 가중치 및 인덱스
    pick_weights = weights[batch_idx, pick_idx].detach().cpu().numpy().flatten()
    neighbor_indices = indices[batch_idx, pick_idx].detach().cpu().numpy().flatten().astype(np.int64)

    # 4. 색상 설정
    # (1) Scene 색상: 기본 어두운 회색 + Attention 무지개색
    scene_colors = np.ones((scene_xyz.shape[0], 3)) * 0.15
    norm_w = (pick_weights - pick_weights.min()) / (pick_weights.max() - pick_weights.min() + 1e-8)
    
    # [추가] 가중치 대비 강조: 가중치가 낮은 애들은 확 죽이고 높은 애들만 살리고 싶을 때 사용
    # norm_w = np.power(norm_w, 2) # 제곱을 하면 상위 가중치만 더 밝게 빛납니다.

    # 빨간색 농도 조절 (Linear Interpolation)
    # 배경색(0.15)에서 빨간색(1.0) 사이를 가중치(norm_w)에 따라 섞음
    red_intensity = 0.15 + (1.0 - 0.15) * norm_w
    
    # 색상 적용: R 채널은 intensity만큼, G/B 채널은 가중치가 높을수록 0(순수 빨강)에 가깝게
    attn_colors = np.zeros((len(norm_w), 3))
    attn_colors[:, 0] = red_intensity  # Red 채널
    attn_colors[:, 1] = 0.15 * (1 - norm_w)  # Green 채널 (어두운 회색에서 0으로)
    attn_colors[:, 2] = 0.15 * (1 - norm_w)  # Blue 채널 (어두운 회색에서 0으로)

    scene_colors[neighbor_indices] = attn_colors
    
    # 5. Picking Point (가장 중요한 지점) - 완전히 흰색으로 표시해서 눈에 띄게 함
    scene_colors[neighbor_indices[0]] = [0, 1, 0]
    # (2) Target 색상: 구분을 위해 밝은 보라색
    target_colors = np.ones((target_xyz.shape[0], 3)) * np.array([0.6, 0.2, 0.8])

    # 5. 데이터 결합 (Scene + Offset된 Target)
    combined_xyz = np.concatenate([scene_xyz, target_xyz_offset], axis=0)
    combined_colors = np.concatenate([scene_colors, target_colors], axis=0)

    # 6. 저장
    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(combined_xyz)
    full_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    
    o3d.io.write_point_cloud(filename, full_pcd)
    print(f"--> Saved: {filename} (Target is offset by 2.0m on X-axis)")


# def save_attention_pcd(input_dict, attn_results, filename="attn_vis.ply", batch_idx=0, pick_idx=10):
#     weights, indices, _ = attn_results
    
#     # 1. 좌표 추출
#     scene_coord = input_dict['scene']['coord']
#     batch_idx_tensor = input_dict['scene']['batch']
#     batch_mask = (batch_idx_tensor == batch_idx)
#     scene_xyz = scene_coord[batch_mask].cpu().numpy()
    
#     # 2. 가중치 및 인덱스 (1차원으로 강제 고정)
#     pick_weights = weights[batch_idx, pick_idx].detach().cpu().numpy().reshape(-1)
#     neighbor_indices = indices[batch_idx, pick_idx].detach().cpu().numpy().reshape(-1).astype(int)

#     # [검증] 가중치 값 확인 (터미널 출력)
#     print(f"Weight - Max: {pick_weights.max():.4f}, Min: {pick_weights.min():.4f}, Mean: {pick_weights.mean():.4f}")

#     # 3. 컬러 맵 생성 (Jet: 빨강이 높은 가중치, 파랑이 낮은 가중치)
#     cmap = plt.get_cmap('jet')
#     # 정규화 (최소~최대 차이가 작으면 제곱하여 대비를 키움)
#     diff = pick_weights.max() - pick_weights.min() + 1e-8
#     norm_w = (pick_weights - pick_weights.min()) / diff
    
#     # 대비 강조 (가중치가 높은 점만 더 빨갛게)
#     norm_w = np.power(norm_w, 2) 
    
#     colors = cmap(norm_w)[:, :3] # RGB만 추출

#     # 4. Point Cloud 생성
#     full_pcd = o3d.geometry.PointCloud()
#     full_pcd.points = o3d.utility.Vector3dVector(scene_xyz)
    
#     # [수정] 배경을 어두운 회색으로 변경 (유색 점들이 잘 보이게)
#     full_pcd.paint_uniform_color([0.2, 0.2, 0.2]) 
    
#     full_colors = np.asarray(full_pcd.colors)
    
#     # 5. 색상 입히기
#     try:
#         # 이웃 점들 색칠
#         full_colors[neighbor_indices] = colors
        
#         # [추가] '중심점(Pick Point)' 자체를 확인하기 위해 아주 밝은 녹색이나 흰색으로 표시
#         # 만약 pick_idx가 scene 내 인덱스라면 아래 코드 활성화
#         # full_colors[pick_idx] = [0, 1, 0] # Bright Green
        
#     except IndexError:
#         print("Error: 인덱스가 Scene 범위를 벗어났습니다. 인덱스 체계를 확인하세요.")

#     full_pcd.colors = o3d.utility.Vector3dVector(full_colors)

#     # 6. 저장
#     o3d.io.write_point_cloud(filename, full_pcd)
#     print(f"--> [SUCCESS] {filename} saved with high contrast.")



def visualize_attention(input_dict, attn_results, batch_idx=0, pick_idx=42):
    """
    batch_idx: 시각화할 배치 번호
    pick_idx: 512개의 pick point 중 어떤 점의 attention을 볼 것인가
    """
    weights, indices, neighbor_feats = attn_results
    
    # 1. 전체 Scene 포인트 준비
    scene_coord = input_dict['scene']['coord']
    batch_mask = (input_dict['scene']['batch'] == batch_idx)
    scene_xyz = scene_coord[batch_mask].cpu().numpy()
    
    # 2. 특정 Pick Point의 이웃 인덱스와 가중치 가져오기
    # weights: (B, N_pick, K) -> (K,)
    pick_weights = weights[batch_idx, pick_idx].detach().cpu().numpy()
    # indices: (B, N_pick, K) -> (K,)
    neighbor_indices = indices[batch_idx, pick_idx].detach().cpu().numpy()
    
    # 3. 컬러 맵 생성 (가중치가 높을수록 붉은색)
    cmap = plt.get_cmap('jet')
    # 가중치 정규화 (0~1)
    norm_weights = (pick_weights - pick_weights.min()) / (pick_weights.max() - pick_weights.min() + 1e-8)
    colors = cmap(norm_weights)[:, :3] # RGBA -> RGB

    # 4. Open3D 객체 생성
    # 전체 씬은 회색으로
    pcd_scene = o3d.geometry.PointCloud()
    pcd_scene.points = o3d.utility.Vector3dVector(scene_xyz)
    pcd_scene.paint_uniform_color([0.8, 0.8, 0.8]) 

    # Attention을 받는 이웃점들만 추출
    neighbor_xyz = scene_xyz[neighbor_indices]
    pcd_attn = o3d.geometry.PointCloud()
    pcd_attn.points = o3d.utility.Vector3dVector(neighbor_xyz)
    pcd_attn.colors = o3d.utility.Vector3dVector(colors)

    # 5. 시각화 (전체 씬 + 강조된 Attention 포인트)
    o3d.visualization.draw_geometries([pcd_scene, pcd_attn])

def move_to_device(d, device):
    for k, v in list(d.items()):
        if torch.is_tensor(v):
            d[k] = v.to(device, non_blocking=True)

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

##########
TOTAL_SCENE_NUM = 190
from tqdm import tqdm
import open3d as o3d
import seaborn as sns
import pandas as pd

def main(scene_idx,valid_data):
    for anno_idx in range(256):
        pcd_data,pick_suction_points,pick_normal_points = valid_data.analyze_data(scene_idx,anno_idx)
        
        chunk_size = 1  # or 5
        all_pred_scores = []
        all_pred_collision = []
        all_GT_scores = []
        all_GT_collision = []
        batch_stats = []
        start = 0
#        for chunk in chunk_list(pcd_data, chunk_size):
        for i, chunk in enumerate(chunk_list(pcd_data, chunk_size)):
            batch_data = unified_collate_fn(chunk)
            batch = to_device(batch_data, device)
            label = batch.pop("label")
            pick_feature = torch.cat([label['coord'], label['normal']], dim=-1)
            inputs = batch
            seal_label_score = label['seal_score'].float().cuda()         # (B, N)
            wrench_label_score = label['wrench_score'].float().cuda()
            collision_label = label.get('collision', torch.zeros_like(seal_label_score)).float().to(device)  # (B, N)

            with torch.no_grad():
                output, (attn_weights, indices, neighbor_tensor) = net.visual_forward(inputs,pick_feature)
            
            if i == 0: # 첫 번째 청크만 저장 (너무 많이 생성 방지)
                save_filename = f"scene_{scene_idx}_anno_{anno_idx}_attn.ply"
                save_attention_pcd(
                    input_dict=inputs, 
                    attn_results=(attn_weights, indices, neighbor_tensor),
                    filename=save_filename,
                    batch_idx=0, 
                    pick_idx=10 # 10번째 suction candidate의 attention 확인
                )
            print(1)




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
    elif split=='train':
        for i in range(100):
            scene_list.append(i)
    else:
        print('invalid split')
        exit(1)

    for scene_idx in scene_list:
        print(f"Processing scene {scene_idx}...")
        main(scene_idx,valid_data)
