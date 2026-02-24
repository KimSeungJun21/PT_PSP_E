import os,sys
import numpy as np
import pickle
import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, random_split
import os, json, glob, random
os.environ.pop("BOOST_ROOT", None)
from pathlib import Path
current_file_path = Path(__file__).resolve()
path = Path(current_file_path)
work_path = str(path.parent.parent)
sys.path.insert(0, work_path)

# Pointcept 라이브러리 경로 설정
pointcept_path = os.environ.get('POINTCEPT_PATH', None)
if pointcept_path is None:
    # 프로젝트 내 Pointcept-main 디렉토리 찾기
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 현재 파일이 model_utils/ 안에 있으므로, 상위 디렉토리의 상위 디렉토리에서 Pointcept-main을 찾음
    pointcept_path = os.path.join(os.path.dirname(os.path.dirname(base_dir)), 'Pointcept-main')
    
    # Pointcept-main이 없으면 기본 경로 시도
    if not os.path.exists(pointcept_path):
        pointcept_path = "/home/kimseungjun/task/PointTransformer/Pointcept"

if os.path.exists(pointcept_path):
    sys.path.insert(0, pointcept_path)

from model_utils.loader_utils import (CenterShift,NormalizeColor,ToTensor,Collect,GridSample,SphereCrop,CenterShiftGrid,
                                        CameraInfo, create_point_cloud_from_depth_image,get_workspace_mask,get_target_surrounding_mask,
                                        RandomScale,RandomJitter,RandomDropout,RandomRotate)
from functools import partial
#from pointcept.datasets.transform import Compose, TRANSFORMS

from utils.logger import get_root_logger

import os
from PIL import Image
import scipy.io as scio

import torch
sys.path.append('/home/kimseungjun/task/PointTransformer/suctionnetAPI')

import open3d as o3d
import cv2
import copy
import time
from tqdm import tqdm  # 상단에 추가
import psutil, os

process = psutil.Process(os.getpid())

def mem_mb():
    return process.memory_info().rss / 1024**2

#log = get_root_logger(log_file="trainer/train.log", log_level=logging.DEBUG, file_mode="a")  # "w"면 매번 새로
log = get_root_logger(file_mode="a")  # "w"면 매번 새로

import os
# 이 설정이 없으면 Open3D와 PyTorch DataLoader가 싸우다가 SegFault 남
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CV_NUM_THREADS"] = "1"

def safe_torch_load(fn, map_location="cpu"):
    """
    Load a torch serialized file with a clearer error message for corrupted files.

    The original crash was ``ValueError: buffer size does not match array size``
    coming from ``torch.load`` when a ``.pth`` file was partially written.  The
    wrapped loader attempts a CPU load first and, on failure, re-raises with the
    file path and size so the caller knows which sample needs to be regenerated.
    """

    try:
        return torch.load(fn, map_location=map_location)
    except ValueError as exc:
        try:
            file_size = os.stat(fn).st_size
        except OSError:
            file_size = None

        size_info = f" ({file_size} bytes)" if file_size is not None else ""
        raise ValueError(
            f"Failed to load corrupted sample '{fn}'{size_info}: {exc}. "
            "Please regenerate or re-download the dataset file."
        ) from exc



import torch, numpy as np
from collections.abc import Mapping

def to_f32(x):
    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=torch.float32)
    return torch.as_tensor(x, dtype=torch.float32).contiguous()

def to_i32(x):
    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=torch.int32)
    return torch.as_tensor(x, dtype=torch.int32).contiguous()

def to_long_1d(x):
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x).view(-1).to(torch.long)
    else:
        t = torch.as_tensor(x).view(-1).to(torch.long)
    return t.contiguous()

def safe_torch_load(fn, map_location="cpu"):
    """
    Load a torch serialized file with a clearer error message for corrupted files.

    The original crash was ``ValueError: buffer size does not match array size``
    coming from ``torch.load`` when a ``.pth`` file was partially written.  The
    wrapped loader attempts a CPU load first and, on failure, re-raises with the
    file path and size so the caller knows which sample needs to be regenerated.
    """

    try:
        return torch.load(fn, map_location=map_location)
    except ValueError as exc:
        try:
            file_size = os.stat(fn).st_size
        except OSError:
            file_size = None

        size_info = f" ({file_size} bytes)" if file_size is not None else ""
        raise ValueError(
            f"Failed to load corrupted sample '{fn}'{size_info}: {exc}. "
            "Please regenerate or re-download the dataset file."
        ) from exc

def transform_point_cloud(cloud, transform, format='4x4'):
    """ Transform points to new coordinates with transformation matrix.

        Input:
            cloud: [np.ndarray, (N,3), np.float32]
                points in original coordinates
            transform: [np.ndarray, (3,3)/(3,4)/(4,4), np.float32]
                transformation matrix, could be rotation only or rotation+translation
            format: [string, '3x3'/'3x4'/'4x4']
                the shape of transformation matrix
                '3x3' --> rotation matrix
                '3x4'/'4x4' --> rotation matrix + translation matrix

        Output:
            cloud_transformed: [np.ndarray, (N,3), np.float32]
                points in new coordinates
    """
    if not (format == '3x3' or format == '4x4' or format == '3x4'):
        raise ValueError('Unknown transformation format, only support \'3x3\' or \'4x4\' or \'3x4\'.')
    if format == '3x3':
        cloud_transformed = np.dot(transform, cloud.T).T
    elif format == '4x4' or format == '3x4':
        ones = np.ones(cloud.shape[0])[:, np.newaxis]
        cloud_ = np.concatenate([cloud, ones], axis=1)
        cloud_transformed = np.dot(transform, cloud_.T).T
        cloud_transformed = cloud_transformed[:, :3]
    return cloud_transformed

k = 15.6
g = 9.8
radius = 0.01
wrench_thre = k * radius * np.pi

def batch_viewpoint_to_matrix(batch_viewpoint):
    """
    입력된 방향 벡터(batch_viewpoint)를 로컬 좌표계의 X축으로 변환하고,
    Gram-Schmidt 과정을 통해 직교하는 Y, Z축을 생성하여 회전 행렬을 반환합니다.
    
    Args:
        batch_viewpoint (np.array): (N, 3) 형태의 방향 벡터 (Suction Normal)
        
    Returns:
        matrix (np.array): (N, 3, 3) 형태의 회전 행렬
                           Col 0: Approach direction (X axis)
                           Col 1: Orthogonal axis (Y axis)
                           Col 2: Orthogonal axis (Z axis)
    """
    # 1. 입력 벡터 정규화 (이것이 로컬 X축이 됨)
    axis_x = batch_viewpoint / np.linalg.norm(batch_viewpoint, axis=1, keepdims=True)

    # 2. 임의의 기준 벡터 생성 (X축과 평행하지 않은 벡터를 선택)
    # 대부분의 경우 [1, 0, 0]을 사용하지만, X축이 [1, 0, 0]과 평행할 경우 [0, 1, 0] 사용
    # 여기서는 간단히 [0, 1, 0]을 업벡터로 가정하고 특이점 처리를 함
    up_vector = np.array([0.0, 1.0, 0.0])
    up_vectors = np.tile(up_vector[np.newaxis, :], (axis_x.shape[0], 1))
    
    # 만약 입력 벡터가 up_vector와 너무 평행하면(내적값 절대값이 1에 가까우면), 다른 축([0, 0, 1])을 사용
    # (실제 구현에서는 대부분의 Grasping normal이 수직 위를 향하지 않는다고 가정하거나, 
    #  안전을 위해 모든 벡터에 대해 교차검증을 수행합니다. 아래는 일반적인 구현입니다.)
    
    # 3. Z축 계산 (X cross UP) -> X와 UP에 수직인 벡터
    axis_z = np.cross(axis_x, up_vectors)
    # 크기가 0에 가까운 경우(평행한 경우) 처리 로직이 필요할 수 있으나, 
    # 일반적인 Suction 상황에서는 axis_z 정규화만 수행
    norm_z = np.linalg.norm(axis_z, axis=1, keepdims=True)
    # 혹시 0이 될 경우를 대비해 1e-6 더해줌
    axis_z = axis_z / (norm_z + 1e-8)

    # 4. Y축 계산 (Z cross X) -> Z와 X에 수직인 벡터 (이미 직교함)
    axis_y = np.cross(axis_z, axis_x)
    
    # 5. 행렬로 합치기 (Column stack: [X, Y, Z])
    # shape: (N, 3, 3)
    # coord 계산시 matmul(vector, matrix) 형태를 취하므로 
    # matrix의 열(column)들이 각 좌표계의 기저 벡터가 되어야 함.
    matrix = np.stack((axis_x, axis_y, axis_z), axis=2)

    return matrix


def batch_get_wrench_score(suction_points, directions, center, g_direction):
    gravity = g_direction * g

    suction_axis = batch_viewpoint_to_matrix(directions)
    bs = suction_axis.shape[0]

    suction2center = (center[np.newaxis, :] - suction_points)[:, np.newaxis, :]
    coord = np.matmul(suction2center, suction_axis)

    gravity_proj = np.matmul(
        np.tile(gravity[np.newaxis, :], (bs, 1, 1)), suction_axis)

    torque_y = gravity_proj[:, 0, 0] * coord[:, 0, 2] - \
        gravity_proj[:, 0, 2] * coord[:, 0, 0]
    torque_z = -gravity_proj[:, 0, 0] * coord[:, 0, 1] + gravity_proj[:, 0, 1] * coord[:, 0, 0]

    torque_max = np.maximum(np.abs(torque_z), np.abs(torque_y))
    score = 1 - np.minimum(torque_max / wrench_thre, 1)

    return score

def get_visible_pcd(pcd, camera_location=[0, 0, 0], voxel_size=0.005):
    """
    안전장치와 속도 최적화가 추가된 버전
    """
    # [안전장치 1] 입력 데이터가 비어있으면 바로 리턴 (SegFault 방지 핵심!)
    if pcd is None or len(pcd.points) < 10:
        return pcd

    # [속도 최적화] 점이 수만 개면 HPR 연산이 느리고 메모리를 많이 먹음 -> 다운샘플링
    # voxel_size=0.005 (5mm) 정도면 형상은 유지하면서 점 개수는 1/10로 줌
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # [안전장치 2] 다운샘플링 했더니 점이 다 사라진 경우 체크
    if len(pcd_down.points) < 10:
        return pcd # 원본 반환 (혹은 pcd_down 반환)

    # 1. 파라미터 설정
    try:
        # get_max_bound 계산 중 에러 방지
        curr_min = pcd_down.get_min_bound()
        curr_max = pcd_down.get_max_bound()
        diameter = np.linalg.norm(curr_max - curr_min)
        
        # [안전장치 3] 지름이 0인 경우 (점이 한 곳에 뭉침) -> HPR 계산 불가
        if diameter < 1e-6:
            return pcd_down

        radius = diameter * 100

        # 2. Hidden Point Removal 수행
        # 여기서 Qhull 라이브러리가 도는데, 점이 꼬여있으면 가끔 터짐
        _, pt_map = pcd_down.hidden_point_removal(camera_location, radius)
    
    except Exception as e:
        # C++ 레벨 에러나 연산 에러 시, 그냥 다운샘플된 전체 데이터 사용
        # print(f"[Warning] HPR Failed: {e}. Using all points.") 
        return pcd_down

    # 3. 보이는 점만 선택
    visible_pcd = pcd_down.select_by_index(pt_map)
    return visible_pcd

#@profile
def unified_collate_fn(batch):
    # batch: [( {"scene": d_s, "target": d_t}, query_dict ), ... ] 형태
    items, labels = zip(*batch)
    
    scene_dicts = [it["scene"] for it in items]
    target_dicts = [it["target"] for it in items]
    #@profile
    def collate_ptv3(dicts):
        """PTv3 포맷용 Collate"""
        coords_list, feats_list, grid_coords_list, lens = [], [], [], []
        
        for d in dicts:
            coords_list.append(torch.as_tensor(d["coord"]).float())
            feats_list.append(torch.as_tensor(d["feat"]).float())
            lens.append(d["coord"].shape[0])
            if "grid_coord" in d:
                grid_coords_list.append(torch.as_tensor(d["grid_coord"]))

        offset = torch.cumsum(torch.tensor(lens, dtype=torch.long), dim=0)
        
        out = {
            "coord": torch.cat(coords_list, dim=0),
            "feat": torch.cat(feats_list, dim=0),
            "offset": offset,
        }

        if grid_coords_list:
            grid_coords = torch.cat(grid_coords_list, dim=0)
            grid_coords -= grid_coords.min(dim=0)[0] 
            out["grid_coord"] = grid_coords.to(torch.int32)
            
            batch_ids = []
            for i, l in enumerate(lens):
                batch_ids.append(torch.full((l,), i, dtype=torch.int64))
            out["batch"] = torch.cat(batch_ids, dim=0)
        
        return out

    # Scene & Target Collate
    scene_batch = collate_ptv3(scene_dicts)
    target_batch = collate_ptv3(target_dicts)

    # [수정] Query(Label) Collate
    # 리스트의 딕셔너리들을 하나의 딕셔너리로 합칩니다 (Stacking)
    # 예: batch_coord -> (B, N, 3)
    query_batch = {}
    if labels:
        first_label = labels[0]
        for key in first_label.keys():
            # numpy array들을 tensor로 변환 후 stack
            query_batch[key] = torch.stack([torch.as_tensor(d[key]) for d in labels])

    return {
        "scene": scene_batch,
        "target": target_batch,
        "label": query_batch  # 딕셔너리 형태 유지
    }



class PT_dataset(Dataset):

    NORMALIZATION_DEBUG_SAMPLES = 5

    def __init__(self, root, camera='kinect', split='train', input_size=(480, 480), use_color: bool = False,log=None):
        #self.logger = log
        
        self.root = root
        self.data_path = []
        self.label = None
        self.use_color = use_color
        self._normalization_logged = 0
        self.remove_outlier = True
        self.num_points = 1024
        self.minimum_num_pt = 50
        scene_path  = os.path.join(self.root,'scenes')
        self.camera=camera
        #self.sn = SuctionNet(root=self.root, camera=self.camera)
        #self.voxel_size = 0.002
        self.bins = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        if split == 'train':
            self.sceneIds = list(range(100))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]
        self.grid_size = 0.006
        self.colorpath = []
        self.depthpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        self.suctionnesspath = []
        self.normalpath = []
        self.segmask_path = []
        for x in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
            for img_num in range(256):
                self.colorpath.append(os.path.join(
                    root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4) + '.png'))
                self.depthpath.append(os.path.join(
                    root, 'scenes', x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                self.labelpath.append(os.path.join(
                    root, 'scenes', x, camera, 'label', str(img_num).zfill(4) + '.png'))
                self.metapath.append(os.path.join(
                    root, 'scenes', x, camera, 'meta', str(img_num).zfill(4) + '.mat'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)
                self.normalpath.append(os.path.join(root, 'normals', x, camera, str(img_num).zfill(4) + '.npy'))
                self.suctionnesspath.append(
                        os.path.join(root, 'suction', x, camera, str(img_num).zfill(4) + '.npz'))
                if split !='train':
                    self.segmask_path.append(os.path.join(root,'uoais_mask',x,camera,str(img_num).zfill(4) + '.png'))
        
        self.outlier = 0.05
        ###########################################
        sample_keys = ["coord", "normal"]
        if self.use_color:
            sample_keys.append("color")
        self.CenterShift1 = CenterShift(apply_z=True)
        self.gridsample = GridSample(grid_size=self.grid_size, # 0.05에서 다시 정밀하게 조정 (필요시)
                hash_type="fnv",
                mode=split,
                return_grid_coord=True,
                keys=tuple(sample_keys),)
        self.spherecrop = SphereCrop(point_max=30000, mode="random")
        #self.centershiftgrid = CenterShiftGrid()
        self.CenterShift2 = CenterShift(apply_z=False)
        self.normalizecolors=NormalizeColor()
        self.totensor = ToTensor()
        self.collect = Collect( 
                keys=("coord", "grid_coord"),
                feat_keys=("normal",) + (("color",) if self.use_color else ()),)

        #RandomScale,RandomJitter,RandomDropout,RandomRotate)
        self.randomdropout = RandomDropout(dropout_ratio=0.2, dropout_application_ratio=0.2)
        self.randomsrotate_x = RandomRotate(angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5)
        self.randomsrotate_y = RandomRotate(angle=[-0.02, 0.02], axis="x", p=0.5)
        self.randomsrotate_z = RandomRotate(angle=[-0.02, 0.02], axis="y", p=0.5)
        self.randomscale = RandomScale(scale=[0.95, 1.05])
        self.randomjitter = RandomJitter(sigma=0.005, clip=0.02)


    def __len__(self):
        return len(self.depthpath)
        #return self.cumulative_sizes[-1] if len(self.cumulative_sizes) > 0 else 0
    def __getitem__(self, index):
        #return self._get_item(index)
        # 최대 재시도 횟수 설정 (무한 루프 방지)
        max_retries = 100
        
        for _ in range(max_retries):
            try:
                return self._get_item(index)
            
            # [수정] ValueError -> Exception (모든 에러를 다 잡도록 변경)
            except Exception as e:
                # [로그 출력] 어떤 에러인지, 몇 번 인덱스인지 확인용 (필요 없으면 주석)
                # print(f"[Warning] Skipping index {index} due to error: {e}")
                # bad_scene, bad_ann, _ = self.data_list[index]
                # print(f"🚨 Corrupted Data! Scene: {bad_scene}, Ann: {bad_ann}")
                # print(f"Error: {e}")
                # 현재 인덱스가 불량(파일 깨짐 등)하므로, 전체 데이터셋 중 랜덤한 다른 인덱스로 교체
                index = np.random.randint(0, len(self.depthpath))
        
        # 운이 너무 나빠서 100번 연속 불량 데이터가 걸리면 에러 발생
        raise RuntimeError(f"Failed to fetch a valid sample after {max_retries} retries.")

    
    def _imread_or_fail(self, path, flags):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        img = cv2.imread(path, flags)
        if img is None:
            raise IOError(f"Failed to read image (None): {path}")
        return img
    
    #@profile
# @profile
    def _get_item(self, index):
        t_start = time.time()

        # 1. 파일 로드 구간
        t0 = time.time()
        color = self._imread_or_fail(self.colorpath[index], cv2.IMREAD_UNCHANGED)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = color.astype(np.float32)

        depth = self._imread_or_fail(self.depthpath[index], cv2.IMREAD_UNCHANGED).astype(np.float32)
        # seg 로딩 중복 제거 (기존 코드 살림)
        seg = self._imread_or_fail(self.labelpath[index], cv2.IMREAD_UNCHANGED).astype(np.float32)

        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        normal = np.load(self.normalpath[index])
        suctionness = np.load(self.suctionnesspath[index])
        seal_score = suctionness['seal_score']
        t_load = time.time() - t0

        # 2. 전처리 및 카메라 정보 준비
        t0 = time.time()
        t_prep_start = time.time()

        # 1. Collision 및 기본 변수 세팅
        t_sub = time.time()
        if 'collision' in suctionness.keys():
            collision = suctionness['collision']
            if len(collision.shape) == 3:
                collision = collision.reshape(-1, collision.shape[-1])[:, 0] if collision.shape[-1] > 0 else collision.reshape(-1)
            elif len(collision.shape) == 2:
                collision = collision.flatten()
        else:
            if len(seal_score.shape) == 3:
                collision = np.zeros(seal_score.shape[0] * seal_score.shape[1], dtype=bool)
            elif len(seal_score.shape) == 2:
                collision = np.zeros(seal_score.shape[0], dtype=bool)
            else:
                collision = np.zeros(seal_score.shape[0], dtype=bool)
        t_collision = time.time() - t_sub

        # 2. Meta 데이터 추출
        t_sub = time.time()
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e), scene)
        t_meta = time.time() - t_sub

        # 3. Point Cloud 생성 (병목 1순위 의심)
        t_sub = time.time()
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        H, W = cloud.shape[:2]
        t_pcd_create = time.time() - t_sub
        
        # 4. 포즈 파일 로드 (I/O 병목 의심)
        t_sub = time.time()
        depth_mask = (depth > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            camera_pose = np.dot(align_mat, camera_poses[self.frameid[index]])
            t_pose_load = time.time() - t_sub

            # 5. Workspace 마스크 생성 (병목 2순위 의심)
            t_sub = time.time()
            workspace_mask = get_workspace_mask(cloud, seg, trans=camera_pose, outlier=0.02)
            mask = (depth_mask & workspace_mask)
            t_work_mask = time.time() - t_sub
        else:
            t_pose_load = 0.0
            t_work_mask = 0.0
            mask = depth_mask
            
        t_prep_total = time.time() - t_prep_start

        if index % 10 == 0:
            print(f"  [Prep Detail] Col: {t_collision:.4f}s | Meta: {t_meta:.4f}s | PCD_Gen: {t_pcd_create:.4f}s | PoseLoad: {t_pose_load:.4f}s | WorkMask: {t_work_mask:.4f}s | Total Prep: {t_prep_total:.4f}s")
        t_prep = time.time() - t0

        # 3. 마스킹 및 유효성 필터링
        t0 = time.time()
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]
        normal_masked = normal
        
        valid_mask = depth_mask[mask] < 3000
        cloud_masked  = cloud_masked[valid_mask]
        color_masked  = color_masked[valid_mask]
        seg_masked    = seg_masked[valid_mask]
        normal_masked = normal_masked[valid_mask]
        seal_score = seal_score[valid_mask]
        
        if len(collision.shape) == 2 and collision.shape[:2] == (H, W):
            collision = collision.reshape(-1)[mask][valid_mask]
        elif len(collision.shape) == 1:
            if collision.shape[0] == H * W:
                collision = collision.reshape(H, W)[mask][valid_mask]
            elif collision.shape[0] == mask.sum():
                collision = collision[valid_mask]
            else:
                collision = collision[:valid_mask.sum()] if collision.shape[0] >= valid_mask.sum() else np.pad(collision, (0, valid_mask.sum() - collision.shape[0]), constant_values=False)
        else:
            collision = np.zeros(valid_mask.sum(), dtype=bool)
        t_masking = time.time() - t0

        # 4. 객체 선택 및 랜덤 샘플링 (단순화됨)
        t0 = time.time()
        while 1:
            choose_idx = np.random.choice(np.arange(len(obj_idxs)))
            inst_mask = seg_masked == obj_idxs[choose_idx]
            inst_mask_len = inst_mask.sum()
            if inst_mask_len > self.minimum_num_pt:
                break
                
        if inst_mask_len >= self.num_points:
            idxs = np.random.choice(inst_mask_len, self.num_points, replace=False)
        else:
            idxs1 = np.arange(inst_mask_len)
            idxs2 = np.random.choice(inst_mask_len, self.num_points - inst_mask_len, replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        t_sampling = time.time() - t0

        # 5. 후처리 및 Wrench 계산
        t0 = time.time()
# 5. 후처리 및 Wrench 계산 상세 분석
        t_post_start = time.time()

        # 5-1. Surrounding Mask 생성 (가장 의심되는 구간)
        t_sub = time.time()
        target_sur_mask = get_target_surrounding_mask(
            cloud=cloud, seg=seg, obj_id=obj_idxs[choose_idx], 
            trans=camera_pose, outlier=self.outlier
        )
        target_sur_mask = target_sur_mask[mask][valid_mask]
        t_sur_mask = time.time() - t_sub

        # 5-2. 기본 인덱싱
        t_sub = time.time()
        inst_cloud = cloud_masked[inst_mask][idxs]
        #inst_color = color_masked[inst_mask][idxs]
        inst_normals = normal_masked[inst_mask][idxs]
        inst_seal_score = seal_score[inst_mask][idxs][:, 0]
        inst_collision = collision[inst_mask][idxs]
        
        target_object_pcd = cloud_masked[inst_mask]
        target_object_color = color_masked[inst_mask]
        target_object_normal = normal_masked[inst_mask]
        t_indexing = time.time() - t_sub

        # 5-3. Target Object Outlier 제거 (Percentile 연산 포함)
        t_sub = time.time()
        if len(target_object_pcd) > 0:
            center = target_object_pcd.mean(axis=0)
            target_dist = np.linalg.norm(target_object_pcd - center, axis=1)
            r = np.percentile(target_dist, 97) # 이 연산이 무거울 수 있음
            mask_outlier = target_dist < r
            if mask_outlier.sum() > 0:
                target_object_pcd = target_object_pcd[mask_outlier]
                target_object_color = target_object_color[mask_outlier]
                target_object_normal = target_object_normal[mask_outlier]
        t_outlier = time.time() - t_sub

        # 5-4. Wrench Score 계산 (핵심 물리 연산)
        t_sub = time.time()
        obj_pose = np.transpose(poses, (2, 0, 1))[choose_idx]
        inst_wrench_score = self.get_wrench_score(inst_cloud, inst_normals, obj_pose, camera_pose)
        t_wrench = time.time() - t_sub

        # 5-5. Surrounding Data 추출
        t_sub = time.time()
        cloud_sur  = cloud_masked[target_sur_mask]
        color_sur  = color_masked[target_sur_mask]
        normal_sur = normal_masked[target_sur_mask]
        t_sur_extract = time.time() - t_sub

        t_post = time.time() - t_post_start

        # 상세 로그 출력 (마찬가지로 10개마다)
        if index % 10 == 0:
            print(f"  [Post Detail] SurMask: {t_sur_mask:.3f}s | Indexing: {t_indexing:.3f}s | Outlier: {t_outlier:.3f}s | Wrench: {t_wrench:.3f}s | SurExtract: {t_sur_extract:.3f}s")
        t_post = time.time() - t0

        # 6. Transform 및 데이터 구성
        t0 = time.time()
        raw_scene_data_dict = {"coord": cloud_sur.astype(np.float32), "normal": normal_sur.astype(np.float32), "color": color_sur.astype(np.float32)}
        raw_target_data_dict = {"coord": target_object_pcd.astype(np.float32), "normal": target_object_normal.astype(np.float32), "color": target_object_color.astype(np.float32)}
        raw_query_data_dict = {
            "coord": inst_cloud.astype(np.float32), 
            "normal": inst_normals.astype(np.float32), 
            "seal_score": inst_seal_score.astype(np.float32),
            "wrench_score": inst_wrench_score.astype(np.float32),
            "collision": inst_collision.astype(np.float32)
        }
        
        scene_data_dict, target_data_dict, query_data_dict = self.transform(scene_input=raw_scene_data_dict, target_input=raw_target_data_dict, pick_input=raw_query_data_dict)
        t_trans = time.time() - t0

        t_total = time.time() - t_start

        if index % 10 == 0:
            print(f"\n[Timer Index {index}] Load: {t_load:.3f}s | Prep: {t_prep:.3f}s | Mask: {t_masking:.3f}s | Sample: {t_sampling:.3f}s | Post: {t_post:.3f}s | Trans: {t_trans:.3f}s | Total: {t_total:.3f}s")

        return {"scene": scene_data_dict, "target": target_data_dict}, query_data_dict
    
    #@profile
    def get_wrench_score(self, obj_points, obj_normals, obj_pose, camera_pose):
        inst_center = obj_pose[:3, 3]
        g_direction = np.array([[0, 0, -1]], dtype=np.float32)
        g_direction = transform_point_cloud(g_direction, np.linalg.inv(camera_pose), '4x4')
        g_direction = g_direction / np.linalg.norm(g_direction)
        wrench_score = batch_get_wrench_score(obj_points, obj_normals, inst_center, g_direction)
        return wrench_score
    
    #@profile
    def transform(self,scene_input,target_input,pick_input,mode = 'train'):
        sample_keys = ["coord", "normal"]
        if self.use_color:
            sample_keys.append("color")
        data_dict,shift1 = self.CenterShift1(scene_input)

        data_dict = self.gridsample(data_dict)
        data_dict = self.spherecrop(data_dict)
        data_dict,shift2 = self.CenterShift2(data_dict)
        #data_dict = self.centershiftgrid(data_dict)
        if self.use_color:
            data_dict = self.normalizecolors(data_dict)
        data_dict = self.totensor(data_dict)
        data_dict = self.collect(data_dict)

        target_input["coord"] -= shift1
        target_data_dict = self.gridsample(target_input)
        target_data_dict = self.spherecrop(target_data_dict)
        target_data_dict["coord"] -= shift2
        target_data_dict = self.totensor(target_data_dict)
        target_data_dict = self.collect(target_data_dict)

        ######pick
        pick_input["coord"] -= shift1

        pick_input["coord"] = pick_input["coord"].astype(np.float32)
        if "normal" in pick_input:
            pick_input["normal"] = pick_input["normal"].astype(np.float32)


        return data_dict, target_data_dict, pick_input

    def test_inference(self,input_index,ann):
        find_scene_id = f'scene_0{input_index}'
        scene_index = self.sceneIds.index(find_scene_id)

        index = scene_index*256 + ann
        num_pt = 1024

        color = self._imread_or_fail(self.colorpath[index], cv2.IMREAD_UNCHANGED)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB) # 여기서 순서를 뒤집어줍니다.
        color = color.astype(np.float32)

        depth = self._imread_or_fail(self.depthpath[index], cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth = depth.astype(np.float32)

        seg   = self._imread_or_fail(self.labelpath[index], cv2.IMREAD_UNCHANGED).astype(np.float32)
        seg = seg.astype(np.float32)

        net_seg   = self._imread_or_fail(self.segmask_path[index], cv2.IMREAD_UNCHANGED).astype(np.float32)
        net_seg = net_seg.astype(np.float32)


        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        normal = np.load(self.normalpath[index])
        suctionness = np.load(self.suctionnesspath[index])
        seal_score = suctionness['seal_score']
        
        # ✅ Collision 정보 로드 (test_inference에서도 필요)
        if 'collision' in suctionness.keys():
            collision = suctionness['collision']
            if len(collision.shape) == 3:
                collision = collision.reshape(-1, collision.shape[-1])[:, 0] if collision.shape[-1] > 0 else collision.reshape(-1)
            elif len(collision.shape) == 2:
                collision = collision.flatten()
        else:
            # Collision 정보가 없으면 모두 False로 설정
            if len(seal_score.shape) == 3:
                collision = np.zeros(seal_score.shape[0] * seal_score.shape[1], dtype=bool)
            elif len(seal_score.shape) == 2:
                collision = np.zeros(seal_score.shape[0], dtype=bool)
            else:
                collision = np.zeros(seal_score.shape[0], dtype=bool)
        
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        
        # ✅ H, W 정의 (collision 마스킹에 필요)
        H, W = cloud.shape[:2]
        
        depth_mask = (depth > 0)
        #seg_mask = (seg > 0)
        
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            camera_pose = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=camera_pose, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask

        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = net_seg[mask]
        normal_masked = normal
        
        valid_mask = depth_mask[mask] < 3000
        
        cloud_masked  = cloud_masked[valid_mask]
        color_masked  = color_masked[valid_mask]
        seg_masked    = seg_masked[valid_mask]
        normal_masked = normal_masked[valid_mask]
        #depth_masked  = depth_masked[valid_mask]
        seal_score = seal_score[valid_mask]
        
        # ✅ Collision 정보도 마스킹 적용 (test_inference)
        if len(collision.shape) == 2 and collision.shape[:2] == (H, W):
            collision = collision.reshape(-1)[mask][valid_mask]
        elif len(collision.shape) == 1:
            if collision.shape[0] == H * W:
                collision = collision.reshape(H, W)[mask][valid_mask]
            elif collision.shape[0] == mask.sum():
                # collision이 이미 mask가 적용된 상태
                collision = collision[valid_mask]
            else:
                # 예외 처리
                if collision.shape[0] > valid_mask.sum():
                    collision = collision[:valid_mask.sum()]
                else:
                    padding = np.zeros(valid_mask.sum() - collision.shape[0], dtype=collision.dtype)
                    collision = np.concatenate([collision, padding])
        else:
            collision = np.zeros(valid_mask.sum(), dtype=bool)

        #######여기까지가 전체 scene pcd들
        #target과 그 주위 정보를 어떻게 crop할것인가?
        #time_4 = time.time()
        
        
        # Process each object instance
        label_pick_point = []
        label_pick_normal = []

        samples = []
        seg_idxs = np.unique(net_seg)
        #print(f"seg_idxs: {seg_idxs}")
        #for index,obj_idx in enumerate(obj_idxs):
        for obj_idx in seg_idxs:
            if obj_idx == 0:
                #print("check_1")
                continue
        
            inst_mask = seg_masked == obj_idx
            inst_mask_len = inst_mask.sum()
            if inst_mask_len < self.minimum_num_pt:
                continue
            #target 물체 pcd 뽑기
            if inst_mask_len >= num_pt:
                idxs = np.random.choice(inst_mask_len, num_pt, replace=False)
            else:
                idxs1 = np.arange(inst_mask_len)
                idxs2 = np.random.choice(inst_mask_len, num_pt - inst_mask_len, replace=True)
                idxs = np.concatenate([idxs1, idxs2], axis=0)

            #scene pcd
            target_sur_mask = get_target_surrounding_mask(cloud=cloud,seg=net_seg,obj_id = obj_idx, trans=camera_pose,outlier=self.outlier)
            target_sur_mask = target_sur_mask[mask]
            target_sur_mask = target_sur_mask[valid_mask]

            #target object pcd
            target_object_pcd = cloud_masked[inst_mask]
            target_object_color = color_masked[inst_mask]
            target_object_normal = normal_masked[inst_mask]

            center = target_object_pcd.mean(axis=0)
            target_dist = np.linalg.norm(target_object_pcd - center, axis=1)
            r = np.percentile(target_dist, 97)  # 상위 3% 제거
            mask_outlier = target_dist < r
            if mask_outlier.sum() > 0:
                target_object_pcd = target_object_pcd[mask_outlier]
                target_object_color = target_object_color[mask_outlier]
                target_object_normal = target_object_normal[mask_outlier]
            else:
                # 필터링했더니 다 사라지면, 그냥 원본(foreground)을 씀
                pass

            ######pick feature
            inst_cloud = cloud_masked[inst_mask][idxs]
            #inst_color = color_masked[inst_mask][idxs]
            inst_normals = normal_masked[inst_mask][idxs]
            #inst_seal_score = seal_score[inst_mask][idxs]
            #inst_seal_score = inst_seal_score[:, 0]
            
            # ✅ Collision 정보도 샘플링 (test_inference)
            inst_collision = collision[inst_mask][idxs]

            #obj_pose = np.transpose(poses, (2, 0, 1))[index]
            # if self.augment:
            #     inst_cloud, obj_pose_list = self.augment_data(inst_cloud, [obj_pose])
            # else:
            #obj_pose_list = [obj_pose]
            #inst_wrench_score = self.get_wrench_score(inst_cloud, inst_normals, obj_pose_list[0], camera_pose)


            cloud_sur  = cloud_masked[target_sur_mask]
            color_sur  = color_masked[target_sur_mask]
            #seg_sur    = seg_masked[target_sur_mask]
            normal_sur = normal_masked[target_sur_mask]
            #seal_sur   = seal_score[target_sur_mask]


            raw_scene_data_dict = {
                "coord": cloud_sur.astype(np.float32),
                "normal": normal_sur.astype(np.float32), # 색상 없으면 0이나 normal 사용
                "color": color_sur.astype(np.float32),    # Regression Target (Score)
            }

            #scene_data_dict,shift1,shift2 = self.scene_transform(input=scene_data_dict)

            target_points = np.asarray(target_object_pcd)
            target_colors = np.asarray(target_object_color)
            #target_colors = np.asarray(target_model.colors)
            target_normals = np.asarray(target_object_normal)

            raw_target_data_dict = {
                "coord": target_points.astype(np.float32),
                "normal": target_normals.astype(np.float32), # 색상 없으면 0이나 normal 사용
                "color": target_colors.astype(np.float32), 
            }

            #target_data_dict = self.target_transform(input=target_data_dict,shift1=shift1,shift2=shift2)

            #labels_coord = inst_cloud - shift1 - shift2
            #label_color = inst_color/ 127.5 - 1
            raw_query_data_dict = {
                        "coord": inst_cloud.astype(np.float32),   # 학습할 위치 (t)
                        #'corlor': label_color.astype(np.float32),
                        "normal": inst_normals.astype(np.float32), # 학습할 방향 (Rotated Normal)
                        #"seal_score": inst_seal_score.astype(np.float32),   # 정답 점수
                        #"wrench_score": inst_wrench_score.astype(np.float32)
                        "collision": inst_collision.astype(np.float32)  # ✅ Collision 정보 (test_inference에서도 포함)
                    }
            scene_data_dict,target_data_dict,query_data_dict = self.transform(scene_input=raw_scene_data_dict,
                                                                    target_input=raw_target_data_dict,
                                                                    pick_input=raw_query_data_dict)

            samples.append((
                {"scene": scene_data_dict, "target": target_data_dict},
                query_data_dict
            ))
            label_pick_point.append(inst_cloud.astype(np.float32))
            label_pick_normal.append(inst_normals.astype(np.float32))
        
        #batch = unified_collate_fn(samples)
                # 리스트 → 텐서
        label_pick_point = torch.from_numpy(
            np.stack(label_pick_point, axis=0)
        ).float()        # (B, 1024, 3)

        label_pick_normal = torch.from_numpy(
            np.stack(label_pick_normal, axis=0)
        ).float()


        return samples, label_pick_point, label_pick_normal
        #이게 b가 object 갯수,

    def analyze_data(self,input_index,ann):
        find_scene_id = f'scene_0{input_index}'
        scene_index = self.sceneIds.index(find_scene_id)

        index = scene_index*256 + ann
        num_pt = 1024
        
        color = self._imread_or_fail(self.colorpath[index], cv2.IMREAD_UNCHANGED)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB) # 여기서 순서를 뒤집어줍니다.
        color = color.astype(np.float32)

        depth = self._imread_or_fail(self.depthpath[index], cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth = depth.astype(np.float32)

        seg   = self._imread_or_fail(self.labelpath[index], cv2.IMREAD_UNCHANGED).astype(np.float32)
        seg = seg.astype(np.float32)
        
        # net_seg   = self._imread_or_fail(self.segmask_path[index], cv2.IMREAD_UNCHANGED).astype(np.float32)
        # net_seg = net_seg.astype(np.float32)


        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        normal = np.load(self.normalpath[index])
        suctionness = np.load(self.suctionnesspath[index])
        seal_score = suctionness['seal_score']
        
        if 'collision' in suctionness.keys():
            collision = suctionness['collision']
            if len(collision.shape) == 3:
                collision = collision.reshape(-1, collision.shape[-1])[:, 0] if collision.shape[-1] > 0 else collision.reshape(-1)
            elif len(collision.shape) == 2:
                collision = collision.flatten()
        else:
            # Collision 정보가 없으면 모두 False로 설정
            if len(seal_score.shape) == 3:
                collision = np.zeros(seal_score.shape[0] * seal_score.shape[1], dtype=bool)
            elif len(seal_score.shape) == 2:
                collision = np.zeros(seal_score.shape[0], dtype=bool)
            else:
                collision = np.zeros(seal_score.shape[0], dtype=bool)

        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        H, W = cloud.shape[:2]
        depth_mask = (depth > 0)
        #seg_mask = (seg > 0)

        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            camera_pose = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=camera_pose, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask

        if len(seal_score.shape) == 3 and seal_score.shape[:2] == (H, W):
            seal_score = seal_score.reshape(-1, seal_score.shape[-1])[mask]
        elif len(seal_score.shape) == 2:
            if seal_score.shape[0] == H * W:
                seal_score_reshaped = seal_score.reshape(H, W, seal_score.shape[1])
                seal_score = seal_score_reshaped[mask]
            elif seal_score.shape[0] == mask.sum():
                # seal_score가 이미 mask가 적용된 상태
                pass
            else:
                # 예외 처리
                if seal_score.shape[0] > mask.sum():
                    seal_score = seal_score[:mask.sum()]
                else:
                    padding = np.zeros((mask.sum() - seal_score.shape[0], seal_score.shape[1]), dtype=seal_score.dtype)
                    seal_score = np.concatenate([seal_score, padding], axis=0)
        elif len(seal_score.shape) == 1:
            if seal_score.shape[0] == H * W:
                seal_score_reshaped = seal_score.reshape(H, W)
                seal_score = seal_score_reshaped[mask]
            elif seal_score.shape[0] == mask.sum():
                # seal_score가 이미 mask가 적용된 상태
                pass
            else:
                # 예외 처리
                if seal_score.shape[0] > mask.sum():
                    seal_score = seal_score[:mask.sum()]
                else:
                    padding = np.zeros(mask.sum() - seal_score.shape[0], dtype=seal_score.dtype)
                    seal_score = np.concatenate([seal_score, padding])

        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]
        normal_masked = normal
        
        valid_mask = depth_mask[mask] < 3000
        
        cloud_masked  = cloud_masked[valid_mask]
        color_masked  = color_masked[valid_mask]
        seg_masked    = seg_masked[valid_mask]
        normal_masked = normal_masked[valid_mask]
        #depth_masked  = depth_masked[valid_mask]
        seal_score = seal_score[valid_mask]
        if len(collision.shape) == 2 and collision.shape[:2] == (H, W):
            collision = collision.reshape(-1)[mask][valid_mask]
        elif len(collision.shape) == 1:
            if collision.shape[0] == H * W:
                collision = collision.reshape(H, W)[mask][valid_mask]
            elif collision.shape[0] == mask.sum():
                collision = collision[valid_mask]
            else:
                # 이미 valid_mask가 적용된 상태
                collision = collision[:valid_mask.sum()] if collision.shape[0] >= valid_mask.sum() else np.pad(collision, (0, valid_mask.sum() - collision.shape[0]), constant_values=False)
        else:
            collision = np.zeros(valid_mask.sum(), dtype=bool)
        
        # Process each object instance
        label_pick_point = []
        label_pick_normal = []

        samples = []
        seg_idxs = np.unique(seg)
        for choose_idx, obj_idx in enumerate(obj_idxs):
            # if obj_idx == 0:
            #     continue
        
            inst_mask = seg_masked == obj_idx
            inst_mask_len = inst_mask.sum()
            if inst_mask_len < self.minimum_num_pt:
                continue
            #target 물체 pcd 뽑기
            if inst_mask_len >= num_pt:
                idxs = np.random.choice(inst_mask_len, num_pt, replace=False)
            else:
                idxs1 = np.arange(inst_mask_len)
                idxs2 = np.random.choice(inst_mask_len, num_pt - inst_mask_len, replace=True)
                idxs = np.concatenate([idxs1, idxs2], axis=0)

            #scene pcd
            target_sur_mask = get_target_surrounding_mask(cloud=cloud,seg=seg,obj_id = obj_idx, trans=camera_pose,outlier=self.outlier)
            target_sur_mask = target_sur_mask[mask]
            target_sur_mask = target_sur_mask[valid_mask]


            #target object pcd
            target_object_pcd = cloud_masked[inst_mask]
            target_object_color = color_masked[inst_mask]
            target_object_normal = normal_masked[inst_mask]

            center = target_object_pcd.mean(axis=0)
            target_dist = np.linalg.norm(target_object_pcd - center, axis=1)
            r = np.percentile(target_dist, 97)  # 상위 3% 제거
            mask_outlier = target_dist < r
            if mask_outlier.sum() > 0:
                target_object_pcd = target_object_pcd[mask_outlier]
                target_object_color = target_object_color[mask_outlier]
                target_object_normal = target_object_normal[mask_outlier]
            else:
                # 필터링했더니 다 사라지면, 그냥 원본(foreground)을 씀
                pass

            ######pick feature
            inst_cloud = cloud_masked[inst_mask][idxs]
            #inst_color = color_masked[inst_mask][idxs]
            inst_normals = normal_masked[inst_mask][idxs]
            inst_seal_score = seal_score[inst_mask][idxs]
            inst_collision = collision[inst_mask][idxs]

            inst_seal_score = inst_seal_score[:, 0]

            target_object_pcd = cloud_masked[inst_mask]
            target_object_color = color_masked[inst_mask]
            target_object_normal = normal_masked[inst_mask]
            center = target_object_pcd.mean(axis=0)
            target_dist = np.linalg.norm(target_object_pcd - center, axis=1)
            r = np.percentile(target_dist, 97)  # 상위 3% 제거
            mask_outlier = target_dist < r
            if mask_outlier.sum() > 0:
                target_object_pcd = target_object_pcd[mask_outlier]
                target_object_color = target_object_color[mask_outlier]
                target_object_normal = target_object_normal[mask_outlier]

            else:
                # 필터링했더니 다 사라지면, 그냥 원본(foreground)을 씀
                pass

            obj_pose = np.transpose(poses, (2, 0, 1))[choose_idx]
            obj_pose_list = [obj_pose]
            inst_wrench_score = self.get_wrench_score(inst_cloud, inst_normals, obj_pose_list[0], camera_pose)
            

            cloud_sur  = cloud_masked[target_sur_mask]
            color_sur  = color_masked[target_sur_mask]
            #seg_sur    = seg_masked[target_sur_mask]
            normal_sur = normal_masked[target_sur_mask]
            #seal_sur   = seal_score[target_sur_mask]

            raw_scene_data_dict = {
                "coord": cloud_sur.astype(np.float32),
                "normal": normal_sur.astype(np.float32), # 색상 없으면 0이나 normal 사용
                "color": color_sur.astype(np.float32),    # Regression Target (Score)
            }

            #scene_data_dict,shift1,shift2 = self.scene_transform(input=scene_data_dict)

            target_points = np.asarray(target_object_pcd)
            target_colors = np.asarray(target_object_color)
            #target_colors = np.asarray(target_model.colors)
            target_normals = np.asarray(target_object_normal)

            raw_target_data_dict = {
                "coord": target_points.astype(np.float32),
                "normal": target_normals.astype(np.float32), # 색상 없으면 0이나 normal 사용
                "color": target_colors.astype(np.float32), 
            }

            raw_query_data_dict = {
                        "coord": inst_cloud.astype(np.float32),   # 학습할 위치 (t)
                        #'corlor': label_color.astype(np.float32),
                        "normal": inst_normals.astype(np.float32), # 학습할 방향 (Rotated Normal)
                        "seal_score": inst_seal_score.astype(np.float32),   # 정답 점수
                        "wrench_score": inst_wrench_score.astype(np.float32),
                        "collision": inst_collision.astype(np.float32)
                    }
            scene_data_dict,target_data_dict,query_data_dict = self.transform(scene_input=raw_scene_data_dict,
                                                                    target_input=raw_target_data_dict,
                                                                    pick_input=raw_query_data_dict)

            samples.append((
                {"scene": scene_data_dict, "target": target_data_dict},
                query_data_dict
            ))
            label_pick_point.append(inst_cloud.astype(np.float32))
            label_pick_normal.append(inst_normals.astype(np.float32))
        
        #batch = unified_collate_fn(samples)
                # 리스트 → 텐서
        label_pick_point = torch.from_numpy(
            np.stack(label_pick_point, axis=0)
        ).float()        # (B, 1024, 3)

        label_pick_normal = torch.from_numpy(
            np.stack(label_pick_normal, axis=0)
        ).float()


        return samples, label_pick_point, label_pick_normal


from torch.utils.data import Subset
import argparse
if __name__ == '__main__':
    from torch.utils.data import DataLoader, random_split   
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx', type=int, default=0,
                        help='subset start index')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='number of samples to iterate')
    parser.add_argument('--mode', type=str, default='test_seen',
                    help='train mode')
    parser.add_argument('--obj_idx', type=int, default=0,
                        help='subset start index')
    args = parser.parse_args()
    
    
    suction_file_path = '/home/kimseungjun/task/PointTransformer/dataset/suctionnet'
    #suction_file_path = '/home/kimseungjun/datasets/graspnet_data/suctionnet'
    work_path = '/home/kimseungjun/datasets/My_PT_data/PT_data'
    split_mode = args.mode
    #split_mode = 'test_seen'
    obj_idx = args.obj_idx
    data = PT_dataset(suction_file_path,split=split_mode,camera='realsense', use_color=True)
    
    start = args.start_idx
    end = min(start + args.num_samples, len(data))


    subset_indices = list(range(start, end))
    subset_data = Subset(data, subset_indices)

    # ⭐ 반드시 unified_collate_fn을 인자로 넣어줘야 합니다.
    train_loader = DataLoader(
        subset_data, 
        batch_size=8,         # 1보다 큰 값도 테스트해 보세요
        shuffle=False, 
        num_workers=0,
        collate_fn=unified_collate_fn  # 이 부분이 빠지면 list가 반환됩니다.
    )
    data.test_inference(110,110)
    print("Start Training Loop Check...")
    start_origin_time = time.time()
    for batch in tqdm(train_loader):
        #pass
        start_time = time.time()
        # # print('load time')
        print(start_time-start_origin_time)
        # # # 구조 체크
        # # scene_feat = batch['scene']['feat']    
        # # target_feat = batch['target']['feat']  
        # # scene_offset = batch['scene']['offset']
        
        # # # 라벨 체크 (Dictionary 형태)
        # # label_coord = batch['label']['coord']
        # # label_seal_score = batch['label']['seal_score']
        # # label_wrench_score = batch['label']['wrench_score']

        # # print(f"\n[Check] Scene Feat Shape: {scene_feat.shape}") 
        # # print(f"[Check] Target Feat Shape: {target_feat.shape}")
        # # print(f"[Check] Label Coord Shape: {label_coord.shape}") # (B, 256, 3) 또는 (B*256, 3) 예상
        # # print(f"[Check] Label Score Shape: {label_seal_score.shape}")
        # # print(f"[Check] Label Score Shape: {label_wrench_score.shape}")
        # # return_time = time.time()
        # # print('---------------loader time--------------')
        # # print(return_time-start_time)
        start_origin_time = time.time()
