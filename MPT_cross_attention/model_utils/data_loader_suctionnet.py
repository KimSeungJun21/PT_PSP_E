import os,sys
import numpy as np
import pickle
import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, random_split
from dataclasses import dataclass, field
import os, json, glob, random
os.environ.pop("BOOST_ROOT", None)
sys.path.insert(0, "/home/kimseungjun/task/PointTransformer/Pointcept")
sys.path.insert(0, "/home/kimseungjun/task/PointTransformer/MPT_cross_attention")

from model_utils.loader_utils import CenterShift,NormalizeColor,ToTensor,Collect,GridSample,SphereCrop
from functools import partial
from pointcept.datasets.transform import Compose, TRANSFORMS

from utils.logger import get_root_logger

import logging, os

from suctionnetAPI import SuctionNet
import torch
sys.path.append('/home/kimseungjun/task/PointTransformer/suctionnetAPI')
from suctionnetAPI.utils.utils import (
    generate_scene_model, 
    plot_sucker_collision, 
    transform_points, 
    parse_posevector, 
    create_table_cloud, 
    get_model_suctions
)
import open3d as o3d
from suctionnetAPI.utils.rotation import (
    viewpoint_to_matrix
)
import cv2
from suctionnetAPI.utils.xmlhandler import (
    xmlReader
)



from tqdm import tqdm  # 상단에 추가


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


def build_transform(include_color: bool = False):
    sample_keys = ["coord", "normal", "segment"]
    if include_color:
        sample_keys.append("color")

    t = [
        # 1. 초기 중심 이동
        dict(type="CenterShift", apply_z=True),
        
        # 2. GridSample (먼저 해서 격자 구조를 먼저 잡음)
        dict(
            type="GridSample",
            grid_size=0.05, # 0.05에서 다시 정밀하게 조정 (필요시)
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
            keys=tuple(sample_keys),
        ),
        
        # 3. SphereCrop (포인트 수 제한)
        # SphereCrop이 keys를 안 받으므로 coord, segment(semantic_gt) 위주로 작동함
        dict(type="SphereCrop", point_max=30000, mode="random"),
        
        # 4. 마무리 정규화
        dict(type="CenterShift", apply_z=False),
        *( [dict(type="NormalizeColor")] if include_color else [] ),
        dict(type="ToTensor"),
        
        # 5. 수집
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "segment"),
            feat_keys=("normal",) + (("color",) if include_color else ()),
        ),
    ]
    return t


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

# def get_visible_pcd(pcd, camera_location=[0, 0, 0]):
#     """
#     pcd: open3d.geometry.PointCloud 객체
#     camera_location: 카메라의 위치 (보통 [0, 0, 0])
#     """
#     # 1. 가시성 체크를 위한 파라미터 설정
#     # 점구름의 전체 크기(대각선 길이)를 계산하여 적절한 radius 설정
#     diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    
#     # radius가 너무 작으면 구멍이 뚫리고, 너무 크면 뒷면이 안 가려집니다. 
#     # 보통 diameter의 100배 정도가 적당합니다.
#     radius = diameter * 100 
    
#     # 2. Hidden Point Removal 수행 (현재 점들 중 카메라에서 보이는 점의 인덱스 반환)
#     # _, pt_map = pcd.hidden_point_removal(camera_location, radius)
    
#     try:
#         # 가려진 점 제거 시도
#         _, pt_map = pcd.hidden_point_removal(camera_location, radius)
#     except RuntimeError:
#         # Qhull 에러가 나면 그냥 모든 점이 보인다고 가정하고 넘어감 (혹은 해당 데이터 스킵)
#         print(f"[Warning] Qhull error ignored at index. Using all points.")
#         pt_map = list(range(len(pcd.points)))  # 모든 점 선택


#     # 3. 보이는 점들만 선택하여 반환
#     visible_pcd = pcd.select_by_index(pt_map)
#     return visible_pcd

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

def unified_collate_fn(batch):
    # batch: [( {"scene": d_s, "target": d_t}, query_dict ), ... ] 형태
    items, labels = zip(*batch)
    
    scene_dicts = [it["scene"] for it in items]
    target_dicts = [it["target"] for it in items]

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

def visualize_arrows(sucker_params, scene_pcd=None):
    geometries = []
    if scene_pcd is not None:
        geometries.append(scene_pcd)

    for param in sucker_params:
        x, y, z = param[0], param[1], param[2]
        nx, ny, nz = param[3], param[4], param[5]
        
        # 화살표 생성 (start_point -> end_point)
        # 0.05는 화살표 길이 (적절히 조절)
        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.002, cone_radius=0.005, cylinder_height=0.03, cone_height=0.01)
        
        # 화살표는 기본적으로 Z축을 향해 서 있음 -> Normal 방향으로 회전 필요
        t = np.array([x, y, z])
        normal = np.array([nx, ny, nz])
        
        R = viewpoint_to_matrix(normal) # Normal 방향으로 정렬하는 회전 행렬
        
        # 변환 적용
        arrow.rotate(R, center=[0,0,0])
        arrow.translate(t)
        
        arrow.paint_uniform_color([1, 0, 0]) # 빨간색 화살표
        geometries.append(arrow)

    o3d.visualization.draw_geometries(geometries, window_name="Normal Vectors Check")


def visualize_sucker_params(sucker_params, scene_pcd=None):
    """
    sucker_params: List of [x, y, z, nx, ny, nz, radius, height]
    """
    geometries = []
    
    # 배경 포인트 클라우드가 있다면 추가
    if scene_pcd is not None:
        geometries.append(scene_pcd)

    for param in sucker_params:
        # 1. 데이터 언패킹
        x, y, z = param[0], param[1], param[2]       # 위치 (t)
        nx, ny, nz = param[3], param[4], param[5]    # 회전된 노멀 (Rotated Normal)
        radius = param[6]
        height = param[7]
        
        t = np.array([x, y, z])
        normal = np.array([nx, ny, nz])

        # 2. 회전 행렬(R) 복원
        # 저장해둔 '회전된 노멀'을 기준으로 다시 자세를 잡습니다.
        R = viewpoint_to_matrix(normal)

        # 3. 원기둥 생성 (기본: Z축 정렬)
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
        vertices = np.asarray(cylinder.vertices)

        # =========================================================
        # [핵심 수정] 사용자님의 기존 create_mesh_cylinder_detection 로직 복원
        # =========================================================
        # 1) 축 변경: Z축(높이)을 X축으로 변경 (Open3D Cylinder는 Z축이 높이임)
        # vertices[:, 2]가 높이였는데, 이걸 0번 인덱스(X)로 보냄
        vertices = vertices[:, [2, 1, 0]] 
        
        # 2) 오프셋: 높이의 절반만큼 이동 (Tip을 원점에 맞추기 위함으로 추정)
        vertices[:, 0] += height / 2
        
        # 3) 회전 및 이동 적용
        # (N, 3) = (R @ (N, 3).T).T + t
        vertices = np.dot(R, vertices.T).T + t
        
        # 좌표 업데이트
        cylinder.vertices = o3d.utility.Vector3dVector(vertices)
        
        # 색상 (파란색)
        cylinder.paint_uniform_color([0, 0, 1]) 
        
        geometries.append(cylinder)

    # 시각화 실행
    o3d.visualization.draw_geometries(geometries, window_name="Corrected Sucker Visualizer", width=1024, height=768)


class PT_data_loader(Dataset):

    NORMALIZATION_DEBUG_SAMPLES = 5

    def __init__(self, root, camera='kinect', split='train', input_size=(480, 480), use_color: bool = False):
        self.root = root
        self.data_path = []
        self.label = None
        self.use_color = use_color
        self.transform = Compose(build_transform(include_color=self.use_color))
        self._normalization_logged = 0

        scene_path  = os.path.join(self.root,'scenes')
        scene_id_list = os.listdir(scene_path)
        self.camera='kinect'

        self.data_list = []
        scene_pbar = tqdm(scene_id_list, desc="Overall Scenes", unit="scene")
        self.sn = SuctionNet(root=self.root, camera=self.camera)
        
        for si in scene_pbar:
            scene_id = int(si.split('_')[-1])
            scene_kinect_path = os.path.join(scene_path,si,self.camera)
            annotation_list = os.listdir(os.path.join(scene_kinect_path,'annotations'))
            for ann_l in annotation_list:
                ann = int(ann_l.replace('.xml',''))
                #model_list, obj_list, pose_list = generate_scene_model(suction_file_path, si, ann, return_poses=True, camera=camera, align=True)
                
                ################################
                scene_reader = xmlReader(os.path.join(self.root, 'scenes', si, self.camera, 'annotations', '%04d.xml'%ann))
                posevectors = scene_reader.getposevectorlist()
                obj_list = [parse_posevector(pv)[0] for pv in posevectors]
                ################################
                for obj_i in range(len(obj_list)):

                    self.data_list.append((si, ann_l,obj_i))
                    
        ###########################################
        sample_keys = ["coord", "normal", "segment"]
        if self.use_color:
            sample_keys.append("color")
        self.CenterShift1 = CenterShift(apply_z=True)
        self.gridsample = GridSample(grid_size=0.05, # 0.05에서 다시 정밀하게 조정 (필요시)
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=tuple(sample_keys),)
        self.spherecrop = SphereCrop(point_max=30000, mode="random")
        self.CenterShift2 = CenterShift(apply_z=False)
        self.normalizecolors=NormalizeColor()
        self.totensor = ToTensor()
        self.collect = Collect( 
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("normal",) + (("color",) if self.use_color else ()),)


    def __len__(self):
        return len(self.data_list)
        #return self.cumulative_sizes[-1] if len(self.cumulative_sizes) > 0 else 0

    def __getitem__(self, index):

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
                index = np.random.randint(0, len(self.data_list))
        
        # 운이 너무 나빠서 100번 연속 불량 데이터가 걸리면 에러 발생
        raise RuntimeError(f"Failed to fetch a valid sample after {max_retries} retries.")

#        return self._get_item(index)

    def _get_item(self, index):
        scene_idx,annotation_id,object_index = self.data_list[index]
        scene_id = int(scene_idx.split('_')[-1])
        ann = int(annotation_id.replace('.xml',''))
        total_scene_pcd = self.sn.loadScenePointCloud(sceneId=scene_id, camera=self.camera, annId=ann, format='open3d')
        align_mat = np.load(os.path.join(self.root, 'scenes', scene_idx, self.camera, 'cam0_wrt_table.npy'))
        
        camera_poses = np.load(os.path.join(self.root, 'scenes', scene_idx, self.camera, 'camera_poses.npy'.format(self.camera)))
        camera_pose = camera_poses[ann]
        real_camera_pose = np.matmul(align_mat, camera_pose)
        cam_pos_in_aligned_space = np.linalg.inv(real_camera_pose)[:3, 3]
        total_scene_pcd.transform(real_camera_pose)
        
        plane_model, inliers = total_scene_pcd.segment_plane(distance_threshold=0.015,
                                             ransac_n=3,
                                             num_iterations=1000)
        table = total_scene_pcd.select_by_index(inliers)
        table.paint_uniform_color([0.2, 0.2, 0.2])

        [a, b, c, d] = plane_model # 평면 방정식: ax + by + cz + d = 0

        # 물체 주변 영역에만 그리드 생성 (예: -0.5 ~ 0.5 범위)
        grid_size = 0.01
        x = np.arange(-0.5, 0.5, grid_size)
        y = np.arange(-0.5, 0.5, grid_size)
        gx, gy = np.meshgrid(x, y)

        # 평면 방정식에 맞춰 z값 계산 (z = -(ax + by + d) / c)
        gz = -(a * gx + b * gy + d) / c -0.01

        # 가상 포인트들을 Open3D 객체로 변환
        points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        table = o3d.geometry.PointCloud()
        table.points = o3d.utility.Vector3dVector(points)
        table.paint_uniform_color([0.1, 0.1, 0.1])
        table_normals = np.tile(np.array([a, b, c]), (points.shape[0], 1))
        
        norm_magnitude = np.linalg.norm(table_normals, axis=1, keepdims=True)
        table_normals = table_normals / (norm_magnitude + 1e-8)
        
        table.normals = o3d.utility.Vector3dVector(table_normals)

        #print('여기가 문제인가?')
        model_list, obj_list, pose_list = generate_scene_model(self.root, scene_idx, ann, return_poses=True, camera=self.camera, align=True)

        collision_dir = os.path.join(self.root, 'suction_collision_label')
            
        collision_dump = np.load(os.path.join(collision_dir, '{:04d}_collision.npz'.format(scene_id)))
        
        radius = 0.01
        height = 0.1
        
        
        full_scene_pcd = o3d.geometry.PointCloud()
        full_scene_pcd += table
        visible_model_list = []

        for m in model_list:
            v_pcd = get_visible_pcd(m, camera_location=cam_pos_in_aligned_space)
            # 시각적 구분을 위해 랜덤 색상을 입힐 수도 있습니다.
            # v_pcd.paint_uniform_color(np.random.rand(3)) 
            visible_model_list.append(v_pcd)
            full_scene_pcd += v_pcd

        obj_idx = obj_list[object_index]
        trans = pose_list[object_index]
        target_object_pcd = visible_model_list[object_index]


        target_points = np.asarray(target_object_pcd.points).copy()
        target_normals = np.asarray(target_object_pcd.normals).copy()
        #target_colors = np.asarray(target_model.colors)
        target_colors = np.asarray(target_object_pcd.colors).copy() if target_object_pcd.has_colors() else np.zeros_like(target_object_pcd)
        

        seal_dir = os.path.join(self.root, 'seal_label')
        sampled_points, normals, scores, _ = get_model_suctions('%s/%03d_seal.npz'%(seal_dir, obj_idx))
        collisions = collision_dump['arr_{}'.format(object_index)]


        full_scene_points = np.asarray(full_scene_pcd.points).copy()
        full_normals = np.asarray(full_scene_pcd.normals).copy()
        full_colors = np.asarray(full_scene_pcd.colors).copy()

        scene_data_dict = {
            "coord": full_scene_points.astype(np.float32),
            "normal": full_normals.astype(np.float32), # 색상 없으면 0이나 normal 사용
            "color": full_colors.astype(np.float32),    # Regression Target (Score)
            "segment": np.zeros(len(full_normals), dtype=np.int32)
        }

        target_data_dict = {
            "coord": target_points.astype(np.float32),
            "normal": target_normals.astype(np.float32), # 색상 없으면 0이나 normal 사용
            "color": target_colors.astype(np.float32), 
            "segment": np.zeros(len(target_points), dtype=np.int32)
        }



        scene_data_dict,shift1,shift2 = self.scene_transform(input=scene_data_dict)
        target_data_dict = self.target_transform(input=target_data_dict,shift1=shift1,shift2=shift2)

        # 몇 개 뽑을지 설정 (예: Positive 128개 + Negative 128개 = 총 256개)
        num_sample = 256

        R_obj = trans[:3, :3] # 물체의 회전
        t_obj = trans[:3, 3]  # 물체의 이동

        raw_labels_coord = (sampled_points @ R_obj.T) + t_obj
        labels_normal = (normals @ R_obj.T)
        label_collision = collisions
        label_score = scores


        # 입력 데이터에서 뺀 값(centroid)을 똑같이 빼줍니다.

        pos_indices = np.where((label_score > 0.4) & (label_collision == 0))[0]
        neg_indices = np.where((label_score <= 0.4) | (label_collision == 1))[0]

        # 정답 라벨도 똑같이 이동
        labels_coord = raw_labels_coord - shift1 - shift2


        # Positive 샘플링 (개수가 모자라면 있는 만큼만)
        if len(pos_indices) > 0:
            replace = len(pos_indices) < num_sample
            sample_pos = np.random.choice(pos_indices, num_sample, replace=replace)
        else:
            raise ValueError(f"[Error] No positive samples found in {scene_idx}, Annotation: {ann}")
            sample_pos = np.array([], dtype=int)

        # Negative 샘플링
        if len(neg_indices) > 0:
            replace = len(neg_indices) < num_sample
            sample_neg = np.random.choice(neg_indices, num_sample, replace=replace)
        else:
            raise ValueError(f"[Error] No positive samples found in {scene_idx}, Annotation: {ann}")
            sample_neg = np.array([], dtype=int)

        # 인덱스 합치고 섞기
        query_idx = np.concatenate([sample_pos, sample_neg])
        np.random.shuffle(query_idx)


        # query_data_dict = {
        #             "coord": labels_coord.astype(np.float32),   # 학습할 위치 (t)
        #             "normal": labels_normal.astype(np.float32), # 학습할 방향 (Rotated Normal)
        #             "score": label_score.astype(np.float32),   # 정답 점수
        #             "collision": label_collision.astype(np.int64) # 정답 충돌여부
        #         }

        query_data_dict = {
                    "coord": labels_coord[query_idx].astype(np.float32),   # 학습할 위치 (t)
                    "normal": labels_normal[query_idx].astype(np.float32), # 학습할 방향 (Rotated Normal)
                    "score": label_score[query_idx].astype(np.float32),   # 정답 점수
                    "collision": label_collision[query_idx].astype(np.int64) # 정답 충돌여부
                }



        # for point_ind in range(len(sampled_points)):
        #     target_point = sampled_points[point_ind]
        #     normal = normals[point_ind]
        #     score = scores[point_ind]
        #     collision = collisions[point_ind]
        #     R = viewpoint_to_matrix(normal)
        #     t = transform_points(target_point[np.newaxis,:], trans).squeeze()
        #     R = np.dot(trans[:3,:3], R)
            
        #     rotated_normal = np.dot(trans[:3,:3], normal)
            
        #     #sucker_params.append([target_point[0],target_point[1],target_point[2],normal[0],normal[1],normal[2],radius, height])
        #     sucker_score.append(score)
        #     collision_list.append(collision)
        #     labels.append(([target_point[0],target_point[1],target_point[2],normal[0],normal[1],normal[2],radius, height],collision,score) )
        #     sucker = plot_sucker_collision(R, t, collision, radius, height)
        #     suckers.append(sucker)
        #     sucker_params.append([t[0], t[1], t[2], rotated_normal[0], rotated_normal[1], rotated_normal[2], radius, height])
        #     # o3d.visualization.draw_geometries([table, *visible_model_list], width=1536, height=864) #total scene pcd
        #     # o3d.visualization.draw_geometries([target_object_pcd, *suckers], #target pcd
        #     #                   window_name=f"Object {obj_idx} Only",
        #     #                   width=1536, height=864)
        #     #visualize_arrows(sucker_params, full_scene_pcd)
        #     #visualize_sucker_params(sucker_params, full_scene_pcd)
        #     #o3d.visualization.draw_geometries([table, *visible_model_list, *suckers], width=1536, height=864)
            


        return {"scene": scene_data_dict, "target": target_data_dict}, query_data_dict
   


    def scene_transform(self, input = None):
        sample_keys = ["coord", "normal", "segment"]
        if self.use_color:
            sample_keys.append("color")

        data_dict,shift1 = self.CenterShift1(input)
        data_dict = self.gridsample(data_dict)
        data_dict = self.spherecrop(data_dict)
        data_dict,shift2 = self.CenterShift2(data_dict)
        if self.use_color:
            data_dict = self.normalizecolors(data_dict)
        data_dict = self.totensor(data_dict)
        data_dict = self.collect(data_dict)

        return data_dict,shift1,shift2


    def target_transform(self,include_color: bool = False, input = None, shift1=None,shift2=None):
        sample_keys = ["coord", "normal", "segment"]
        if include_color:
            sample_keys.append("color")
        #scene_CenterShift1 = CenterShift(apply_z=True)

        input["coord"] -= shift1
        
        data_dict = self.gridsample(input)
        data_dict = self.spherecrop(data_dict)
        data_dict['coord'] -= shift2
        if include_color:
            data_dict = self.normalizecolors(data_dict)
        data_dict = self.totensor(data_dict)
        data_dict = self.collect(data_dict)

        return data_dict




if __name__ == '__main__':
    from torch.utils.data import DataLoader, random_split   
    suction_file_path = '/home/kimseungjun/datasets/graspnet_data/suctionnet'
    work_path = '/home/kimseungjun/datasets/My_PT_data/PT_data'
    data = PT_data_loader(suction_file_path, use_color=True)
    
    dataset_size = len(data)
    train_size = int(dataset_size * 0.8)
    val_size = dataset_size - train_size
    train_data, _ = random_split(data, [train_size, val_size])

    # ⭐ 반드시 unified_collate_fn을 인자로 넣어줘야 합니다.
    train_loader = DataLoader(
        train_data, 
        batch_size=4,         # 1보다 큰 값도 테스트해 보세요
        shuffle=True, 
        collate_fn=unified_collate_fn  # 이 부분이 빠지면 list가 반환됩니다.
    )

    print("Start Training Loop Check...")
    for batch in tqdm(train_loader):
        # 구조 체크
        scene_feat = batch['scene']['feat']    
        target_feat = batch['target']['feat']  
        scene_offset = batch['scene']['offset']
        
        # 라벨 체크 (Dictionary 형태)
        label_coord = batch['label']['coord']
        label_score = batch['label']['score']

        print(f"\n[Check] Scene Feat Shape: {scene_feat.shape}") 
        print(f"[Check] Target Feat Shape: {target_feat.shape}")
        print(f"[Check] Label Coord Shape: {label_coord.shape}") # (B, 256, 3) 또는 (B*256, 3) 예상
        print(f"[Check] Label Score Shape: {label_score.shape}")