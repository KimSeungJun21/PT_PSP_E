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
sys.path.insert(0, "/home/kimseungjun/task/PointTransformer/My_point_transformer")

from functools import partial
from pointcept.datasets.transform import Compose, TRANSFORMS

from utils.logger import get_root_logger

import logging, os

#log = get_root_logger(log_file="trainer/train.log", log_level=logging.DEBUG, file_mode="a")  # "w"면 매번 새로
log = get_root_logger(file_mode="a")  # "w"면 매번 새로


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
    sample_keys = ["coord", "normal", "segment", "instance"]
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


# def build_transform(include_color: bool = False):
#     # GridSample에서 "같이 샘플링"할 키들
#     grid_keys = ["coord", "normal","semantic_gt", "target_mask"]
#     if include_color:
#         grid_keys.append("color")   # ⭐ color도 같이 GridSample로 인덱싱돼야 shape 안 깨짐

#     # Collect에서 "최종 dict에 남길" 키들
#     collect_keys = ["coord", "grid_coord", "semantic_gt", "target_mask"]
#     # seg_feat를 feat로만 쓰고 dict에서는 버려도 되면 아래 줄은 빼도 됨

#     feat_keys = ["normal",'target_mask']
#     if include_color:
#         feat_keys.append("color")

#     t = [
#         dict(type="CenterShift", apply_z=True),
#         dict(
#             type="GridSample",
#             grid_size=0.02,
#             hash_type="fnv",
#             mode="train",
#             return_grid_coord=True,
#             keys=tuple(grid_keys),
#         ),
#         dict(type="CenterShift", apply_z=True),
#         *( [dict(type="NormalizeColor")] if include_color else [] ),
#         dict(type="ToTensor"),
#         dict(
#             type="Collect",
#             keys=tuple(collect_keys),
#             feat_keys=tuple(feat_keys),
#         ),
#     ]
#     return t



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

# def unified_collate_fn(batch):
#     # (data_dict, label) 튜플 분리
#     dicts, labels = zip(*batch)

#     coords_list, feats_list, lens = [], [], []
#     fn_list = []
#     sem_list = []      # semantic_gt (Ni,)
#     target_list = []   # target_mask (Ni,)
#     lab_list = []      # sample-level label (B,)

#     for d in dicts:
#         # 좌표 및 특징 (색상 또는 노멀)
#         coord = torch.as_tensor(d["coord"]).float()
#         # feat는 color와 normal을 합쳐서 사용할 수도 있습니다.
#         feat = torch.as_tensor(d.get("color", d.get("normal"))).float()
        
#         coords_list.append(coord)
#         feats_list.append(feat)
#         lens.append(coord.shape[0])
#         fn_list.append(d.get('data_fn', ""))

#         # Point-wise Labels (핵심 수정: Ni 크기의 마스크들을 리스트에 추가)
#         if "semantic_gt" in d:
#             sem_list.append(torch.as_tensor(d["semantic_gt"]).long())
#         if "target_mask" in d:
#             target_list.append(torch.as_tensor(d["target_mask"]).long())

#     # Concat (N1+N2+..., C)
#     coords = torch.cat(coords_list, dim=0)
#     feats = torch.cat(feats_list, dim=0)
#     offset = torch.cumsum(torch.tensor(lens, dtype=torch.long), dim=0)

#     out = {
#         "coord": coords, 
#         "feat": feats, 
#         "offset": offset,
#         "data_path": fn_list
#     }

#     # 리스트가 비어있지 않을 때만 concat하여 추가
#     if sem_list:
#         out["semantic_gt"] = torch.cat(sem_list, dim=0)
#     if target_list:
#         out["target_mask"] = torch.cat(target_list, dim=0)
    
#     # Sample-level Label (B,)
#     out["label"] = torch.tensor(labels, dtype=torch.long)

#     return out
def unified_collate_fn(batch):
    # batch: [( {"scene": d_s, "target": d_t}, label ), ... ] 형태
    items, labels = zip(*batch)
    
    # scene과 target 데이터를 분리
    scene_dicts = [it["scene"] for it in items]
    target_dicts = [it["target"] for it in items]

    def collate_ptv3(dicts):
        """PTv3 포맷에 맞게 딕셔너리 리스트를 하나로 합치는 헬퍼 함수"""
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
            grid_coords -= grid_coords.min(dim=0)[0] # 음수 방지
            out["grid_coord"] = grid_coords.to(torch.int32)
            
            # batch index 생성
            batch_ids = []
            for i, l in enumerate(lens):
                batch_ids.append(torch.full((l,), i, dtype=torch.int64))
            out["batch"] = torch.cat(batch_ids, dim=0)
        
        return out

    # Scene과 Target 각각에 대해 합치기 수행
    scene_batch = collate_ptv3(scene_dicts)
    target_batch = collate_ptv3(target_dicts)

    # 최종 결과 구성
    return {
        "scene": scene_batch,
        "target": target_batch,
        "label": torch.tensor(labels, dtype=torch.long)
    }

# def unified_collate_fn(batch, mix_prob=0.0, **kwargs):
#     dicts, labels = zip(*batch)
#     coords_list, feats_list, lens = [], [], []
    
#     for d in dicts:
#         coords_list.append(torch.as_tensor(d["coord"]).float())
#         feats_list.append(torch.as_tensor(d["feat"]).float())
#         lens.append(d["coord"].shape[0])

#     # 기본값 설정
#     grid_size = dicts[0].get("grid_size", 0.02)
#     offset = torch.cumsum(torch.tensor(lens, dtype=torch.long), dim=0)

#     out = {
#         "coord": torch.cat(coords_list, dim=0),
#         "feat": torch.cat(feats_list, dim=0),
#         "offset": offset,
#         "grid_size": grid_size,
#         "label": torch.tensor(labels, dtype=torch.long)
#     }
    
#     # ⭐ grid_coord 처리: 타입과 범위를 엄격하게 제한
#     if "grid_coord" in dicts[0]:
#         grid_coords = torch.cat([torch.as_tensor(d["grid_coord"]) for d in dicts], dim=0)
        
#         # 1. 최소값을 0으로 맞춰 음수 방지
#         grid_coords -= grid_coords.min(dim=0)[0]
        
#         # 2. 타입 변환 (Long보다 Int32가 힐베르트 연산에 더 안정적일 수 있음)
#         out["grid_coord"] = grid_coords.to(torch.int32)
        
#         # 3. 추가 안전장치: batch 정보 명시
#         batch_ids = []
#         for i, l in enumerate(lens):
#             batch_ids.append(torch.full((l,), i, dtype=torch.int64))
#         out["batch"] = torch.cat(batch_ids, dim=0)

#     return out

class PT_data_loader(Dataset):

    NORMALIZATION_DEBUG_SAMPLES = 5

    def __init__(self, root, split='train', process_data=False, use_color: bool = False):
        self.root = root
        self.data_path = []
        self.label = None
        self.use_color = use_color
        self.transform = Compose(build_transform(include_color=self.use_color))
        self._normalization_logged = 0

        if split == 'train':
            data_file_path = os.path.join(self.root, 'train')
            data_file_list = os.listdir(data_file_path)
            for data_file in data_file_list:
                if data_file.endswith('.pth'):
                    path = os.path.join(data_file_path, data_file)
                    self.data_path.append(path)
                # elif data_file.endswith('.json'):
                #     path = os.path.join(data_file_path, data_file)
                #     with open(path,'r') as f:
                #         labels = json.load(f)
                #     self.label = labels

        if split == 'test':
            data_file_path = os.path.join(self.root, 'test')
            data_file_list = os.listdir(data_file_path)
            for data_file in data_file_list:
                if data_file.endswith('.pth'):
                    path = os.path.join(data_file_path, data_file)
                    self.data_path.append(path)
                elif data_file.endswith('.json'):
                    path = os.path.join(data_file_path, data_file)
                    with open(path,'r') as f:
                        labels = json.load(f)
                    self.label = labels




    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        return self._get_item(index)

    def _get_item(self, index):
        max_retry = 10  # 실패 대비 재시도 횟수 상향
        for attempt in range(max_retry):
            fn = self.data_path[index]
            try:
                # 1. 파일 로드
                datas = safe_torch_load(fn, map_location="cpu")
                
                # 2. 전처리 (NaN 제거 등)
                coord = datas['coord']
                normal = datas['normal']
                semantic_gt = datas['semantic_gt'].reshape(-1, 1)
                target_mask = datas['target_mask'].reshape(-1, 1)
                
                mask = ~np.isnan(coord).any(axis=1)
                if not np.any(mask): # 모든 포인트가 NaN인 경우 대비
                    raise ValueError("All points are NaN")

                data_dict = dict(
                    coord=coord[mask],
                    normal=normal[mask],
                    segment=semantic_gt[mask],
                    instance=target_mask[mask],
                )

                if self.use_color and "color" in datas:
                    data_dict["color"] = datas["color"][mask]

                # 3. Transform 적용 (GridSample, SphereCrop 등)
                target_indices = np.where(data_dict["instance"].flatten() == 1)[0]
                if len(target_indices) < 10: # 타겟 포인트가 너무 적으면 에러 처리
                    raise ValueError(f"Target object has too few points: {len(target_indices)}")
                target_dict = dict(
                    coord=data_dict["coord"][target_indices].copy(),
                    normal=data_dict["normal"][target_indices].copy(),
                    segment=data_dict["segment"][target_indices].copy(),
                    instance=data_dict["instance"][target_indices].copy(), # ⭐ 이 줄을 추가하세요!
                )                
                if "color" in data_dict:
                    target_dict["color"] = data_dict["color"][target_indices].copy()

                data_dict = self.transform(data_dict)
                target_dict = self.transform(target_dict)
                # 4. ⭐ 핵심 체크: 포인트 개수가 너무 적으면 학습 불가 (PTv3 최소 기준)
                # 힐베르트 인코딩 에러를 피하기 위해 최소 32개 이상의 포인트 권장
                if data_dict["coord"].shape[0] < 32:
                    raise ValueError(f"Too few points after transform: {data_dict['coord'].shape[0]}")

                # 성공 시 결과 반환
                data_dict['data_fn'] = fn
                raw_label = datas['label']
                label = int(raw_label) if isinstance(raw_label, (bool, np.bool_)) else int(raw_label)
                return {"scene": data_dict, "target": target_dict}, label

            except Exception as e:
                # 에러 발생 시 로그를 남기고 다른 인덱스로 재시도
                log.warning(f"[Retry {attempt+1}/{max_retry}] Skipping {fn} due to: {e}")
                index = random.randrange(len(self.data_path))
                continue

        raise RuntimeError(f"Exceeded {max_retry} attempts to load a valid sample.")




if __name__ == '__main__':
    from torch.utils.data import DataLoader, random_split
    
    work_path = '/home/kimseungjun/datasets/My_PT_data/PT_data'
    data = PT_data_loader(work_path, split='train', use_color=False)
    
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

    for batch in tqdm(train_loader):
            # 구조가 바뀌었으므로 접근 방식도 변경
            scene_feat = batch['scene']['feat']    
            target_feat = batch['target']['feat']  
            scene_offset = batch['scene']['offset']
            label = batch['label']

            print(f"\n[Check] Scene Feat: {scene_feat.shape}") 
            print(f"[Check] Target Feat: {target_feat.shape}")
            print(f"[Check] Scene Offset: {scene_offset}")
        #break # 하나만 확인하고 중단