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



# transform=[
#             dict(type="CenterShift", apply_z=True),
#             dict(
#                 type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
#             ),
#             # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
#             dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
#             dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
#             dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
#             dict(type="RandomScale", scale=[0.9, 1.1]),
#             # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
#             dict(type="RandomFlip", p=0.5),
#             dict(type="RandomJitter", sigma=0.005, clip=0.02),
#             # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
#             dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
#             dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
#             dict(type="ChromaticJitter", p=0.95, std=0.05),
#             # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
#             # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
#             dict(
#                 type="GridSample",
#                 grid_size=0.02,
#                 hash_type="fnv",
#                 mode="train",
#                 return_grid_coord=True,
#                 keys = ('coord',)
#             ),
#             dict(type="SphereCrop", sample_rate=0.6, mode="random"),
#             dict(type="SphereCrop", point_max=204800, mode="random"),
#             dict(type="CenterShift", apply_z=False),
#             dict(type="NormalizeColor"),
#             # dict(type="ShufflePoint"),
#             dict(type="ToTensor"),
#             dict(
#                 type="Collect",
#                 keys=("coord", "grid_coord", "segment"),
#                 feat_keys=("coord", ),
#             ),
#         ]


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
    # GridSample에서 "같이 샘플링"할 키들
    grid_keys = ["coord", "segment", "seg_feat", "instance"]
    if include_color:
        grid_keys.append("color")   # ⭐ color도 같이 GridSample로 인덱싱돼야 shape 안 깨짐

    # Collect에서 "최종 dict에 남길" 키들
    collect_keys = ["coord", "grid_coord", "segment", "instance"]
    # seg_feat를 feat로만 쓰고 dict에서는 버려도 되면 아래 줄은 빼도 됨
    collect_keys.append("seg_feat")  # ⭐ 디버깅/분석용으로 남기고 싶으면 유지

    feat_keys = ["normal"]
    if include_color:
        feat_keys.append("color")

    t = [
        dict(type="CenterShift", apply_z=True),
        dict(
            type="GridSample",
            grid_size=0.02,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
            keys=tuple(grid_keys),
        ),
        dict(type="CenterShift", apply_z=True),
        *( [dict(type="NormalizeColor")] if include_color else [] ),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=tuple(collect_keys),
            feat_keys=tuple(feat_keys),
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


def unified_collate_fn(batch, mix_prob=0.0):
    """
    지원:
      - [dict, ...]
      - [(dict, label), ...]
      - dict 내부에 'label' 키 포함
    출력:
      coord:(∑N,3), feat:(∑N,C), offset:(B,)
      + grid_coord:(∑N,3) [있으면]
      + grid_size: float or tensor [있으면, 그대로 전달(보통 스칼라)]
      + segment:(∑N,) [있으면]  # per-point semantic
      + instance:(∑N,) [있으면]
      + label:(B,) [있으면]
    """
    first = batch[0]
    # (dict, label) 튜플 케이스
    if isinstance(first, tuple) and isinstance(first[0], Mapping):
        dicts, labels = zip(*batch)
    else:
        dicts, labels = batch, None

    coords_list, feats_list, lens = [], [], []
    fn_list = []
    grid_list = []     # grid_coord 모으기
    sem_list = []      # per-point 'segment'
    inst_list = []     # per-point 'instance'
    lab_list  = []     # sample-level label
    grid_size = None   # 스칼라/텐서면 그대로 유지 (샘플마다 동일 가정)

    for d in dicts:
        # 좌표/특징
        coord = to_f32(d["coord"])                            # (Ni,3)
        feat  = to_f32(d.get("feat", d.get("color")))         # (Ni,C)
        coords_list.append(coord)
        feats_list.append(feat)
        lens.append(coord.shape[0])
        fn_list.append(d['data_fn'])

        # grid_coord / grid_size
        if "grid_coord" in d:
            grid_list.append(to_i32(d["grid_coord"]))         # (Ni,3) int32
        if grid_size is None and "grid_size" in d:
            # 스칼라라면 그대로 보관 (텐서라도 괜찮음)
            grid_size = d["grid_size"]

        # per-point 라벨
        if "segment" in d:
            sem_list.append(to_long_1d(d["segment"]))
        if "semantic_gt" in d:  # 방어적으로도 지원
            sem_list.append(to_long_1d(d["semantic_gt"]))
        if "instance" in d:
            inst_list.append(to_long_1d(d["instance"]))

        # sample-level 라벨(dict 내부)
        if "label" in d:
            lab_list.append(int(d["label"]))

    # concat & offset
    coords = torch.cat(coords_list, dim=0)
    feats  = torch.cat(feats_list,  dim=0)
    lens   = torch.tensor(lens, dtype=torch.long)
    offset = torch.cumsum(lens, dim=0)


    out = {"coord": coords, "feat": feats, "offset": offset,'data_path':fn_list}

    if grid_list:
        out["grid_coord"] = torch.cat(grid_list, dim=0)
    if grid_size is not None:
        out["grid_size"] = grid_size

    if sem_list:
        out["segment"] = torch.cat(sem_list, dim=0)
    if inst_list:
        out["instance"] = torch.cat(inst_list, dim=0)

    # sample-level label
    if labels is not None:
        out["label"] = torch.tensor([int(l) for l in labels], dtype=torch.long)
    elif lab_list:
        out["label"] = torch.tensor(lab_list, dtype=torch.long)

    return out

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
                elif data_file.endswith('.json'):
                    path = os.path.join(data_file_path, data_file)
                    with open(path,'r') as f:
                        labels = json.load(f)
                    self.label = labels

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
        # fn = self.data_path[index]
        # #datas = torch.load(fn, map_location="cuda:0")
        # datas = safe_torch_load(fn, map_location="cpu")

        max_retry = 5
        for attempt in range(max_retry):
            fn = self.data_path[index]
            try:
                datas = safe_torch_load(fn, map_location="cpu")
                break
            except ValueError as exc:
                log.warning(
                    "Skipping corrupted sample %s (attempt %d/%d): %s",
                    fn,
                    attempt + 1,
                    max_retry,
                    exc,
                )
                index = random.randrange(len(self.data_path))
        else:
            raise RuntimeError(
                f"Exceeded {max_retry} attempts to load a valid sample (last: '{fn}')"
            )


        if "semantic_gt" in datas.keys():
            segment = datas["semantic_gt"].reshape([-1])
            target_id = max(segment)
            seg_feat = (segment == target_id).astype(np.float32).reshape(-1, 1)
            #seg_feat = segment.astype(np.float32).reshape(-1, 1)

        if "instance_gt" in datas.keys():
            instance = datas["instance_gt"].reshape([-1])


        if not self.use_color and "color" in datas:
            datas.pop("color", None)


        data_dict = dict(
            coord=datas['coord'],
            segment=segment,
            seg_feat=seg_feat,
            instance=instance,
        )

        if self.use_color and "color" in datas:
            data_dict["color"] = datas["color"]

        #log.info(f"datas {fn}")


        data_dict = self.transform(data_dict)
        if self._normalization_logged < self.NORMALIZATION_DEBUG_SAMPLES:
            coord = data_dict.get("coord")
            color = data_dict.get("color") if self.use_color else None
            # coord normalization check (CenterShift twice -> mean near 0)
            if coord is not None and hasattr(coord, "mean"):
                coord_mean = coord.mean(dim=0)
                coord_std = coord.std(dim=0)
                # log.info(
                #     "[norm-check] coord mean=%s std=%s for %s",
                #     coord_mean.tolist(),
                #     coord_std.tolist(),
                #     fn,
                # )
            if color is not None and hasattr(color, "amin"):
                # log.info(
                #     "[norm-check] color min=%.3f max=%.3f (expected 0-1) for %s",
                #     float(color.amin()),
                #     float(color.amax()),
                #     fn,
                # )
                pass
            self._normalization_logged += 1


        data_dict['data_fn'] = fn
        fn_fixed = fn.replace("/PT_data/train/", "/PT_data/")
        
        f_bn = os.path.basename(fn)

        label = self.label[f_bn]
        
        return data_dict,label

        # max_retry = 10
        # for _ in range(max_retry):
        #     fn = self.data_path[index]
        #     datas = torch.load(fn, map_location="cpu")

        #     # --- sanitize (전) ---
        #     coord = np.asarray(datas["coord"], dtype=np.float32)
        #     color = np.asarray(datas["color"], dtype=np.float32)

        #     # NaN/Inf 제거
        #     mask = np.isfinite(coord).all(axis=1)
        #     coord = coord[mask]
        #     color = color[mask]

        #     segment = datas.get("semantic_gt", None)
        #     if segment is not None:
        #         segment = np.asarray(segment).reshape(-1)[mask]
        #     instance = datas.get("instance_gt", None)
        #     if instance is not None:
        #         instance = np.asarray(instance).reshape(-1)[mask]

        #     # 비거나 너무 적으면 재시도
        #     if coord.shape[0] < 16:  # 최소 포인트 수 임계값 (필요시 조정)
        #         index = random.randrange(len(self.data_path))
        #         continue

        #     data_dict = {"coord": coord, "color": color}
        #     if segment is not None: data_dict["segment"] = segment
        #     if instance is not None: data_dict["instance"] = instance

        #     try:
        #         data_dict = self.transform(data_dict)
        #     except Exception as e:
        #         # 변환 중 에러도 재시도
        #         index = random.randrange(len(self.data_path))
        #         continue

        #     # --- sanitize (후) ---
        #     coord_t = data_dict.get("coord", None)
        #     if coord_t is None:
        #         index = random.randrange(len(self.data_path))
        #         continue

        #     # numpy / torch 모두 대응
        #     num_pts = (coord_t.shape[0] if isinstance(coord_t, np.ndarray)
        #             else (coord_t.size(0) if hasattr(coord_t, "size") else 0))
        #     if num_pts == 0:
        #         index = random.randrange(len(self.data_path))
        #         continue

        #     # 라벨
        #     #fn_fixed = fn.replace("/PT_data/train/", "/PT_data/")
        #     label = self.label[fn] if self.label is not None else None
        #     return (data_dict, label) if label is not None else data_dict

        # # 여러 번 실패하면 건너뛰도록 IndexError
        # raise IndexError("too many empty/invalid samples encountered")




if __name__ == '__main__':
    import torch
    path=os.getcwd()
    print(path)
    # work_path=os.path.join(path,'data')  # 기존 경로
    work_path='/home/kimseungjun/cmes_planner/PT_data'  # 실제 사용하는 경로로 수정
    data = PT_data_loader(work_path, split='train')
    dataset_size = len(data)
    train_size = int(dataset_size * 0.8)
    validation_size = dataset_size-train_size

    train_data,valid_data=random_split(data,[train_size,validation_size])
    DataLoader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
    for point, label in tqdm(DataLoader):
        print(point.shape)
        print(label.shape)