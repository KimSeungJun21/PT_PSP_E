import numpy as np
import random
import numbers
import scipy
import scipy.ndimage
import scipy.interpolate
import scipy.stats
import numpy as np
import torch
import copy
from collections.abc import Sequence, Mapping
import numba
from memory_profiler import profile


class RandomScale(object):
    def __init__(self, scale=None, anisotropic=False):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.anisotropic = anisotropic

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            scale = np.random.uniform(
                self.scale[0], self.scale[1], 3 if self.anisotropic else 1
            )
            data_dict["coord"] *= scale
        return data_dict,scale


class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            jitter = np.clip(
                self.sigma * np.random.randn(data_dict["coord"].shape[0], 3),
                -self.clip,
                self.clip,
            )
            data_dict["coord"] += jitter
        return data_dict,jitter


class RandomDropout(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data_dict):
        if random.random() < self.dropout_application_ratio:
            n = len(data_dict["coord"])
            idx = np.random.choice(n, int(n * (1 - self.dropout_ratio)), replace=False)
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx = np.unique(np.append(idx, data_dict["sampled_index"]))
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx])[0]
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx]
            if "normal" in data_dict.keys():
                data_dict["normal"] = data_dict["normal"][idx]
            if "strength" in data_dict.keys():
                data_dict["strength"] = data_dict["strength"][idx]
            if "segment" in data_dict.keys():
                data_dict["segment"] = data_dict["segment"][idx]
            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][idx]
        return data_dict


class RandomRotate(object):
    def __init__(self, angle=None, center=None, axis="z", always_apply=False, p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict, np.zeros(3), np.eye(3)
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict,center,rot_t

class CenterShiftGrid(object):
    """
    Grid Coordinate의 최소값을 빼서 (0,0,0) 근처로 당겨주는 클래스.
    Octree Depth 에러를 방지하기 위해 필수적입니다.
    """
    #@profile
    def __call__(self, data_dict):
        if "grid_coord" in data_dict:
            # grid_coord는 정수이므로 min을 빼서 0으로 맞춤
            data_dict["grid_coord"] -= data_dict["grid_coord"].min(axis=0)
        return data_dict

class CenterShift(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z
    #@profile
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            x_min, y_min, z_min = data_dict["coord"].min(axis=0)
            x_max, y_max, _ = data_dict["coord"].max(axis=0)
            if self.apply_z:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
            else:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
            data_dict["coord"] -= shift
        return data_dict,shift

class NormalizeColor(object):
    #@profile
    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"] / 127.5 - 1
        return data_dict

class ToTensor(object):
    #@profile
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            # note that str is also a kind of sequence, judgement should before sequence
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        elif isinstance(data, Mapping):
            result = {sub_key: self(item) for sub_key, item in data.items()}
            return result
        elif isinstance(data, Sequence):
            result = [self(item) for item in data]
            return result
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")

class Collect(object):
    def __init__(self, keys, offset_keys_dict=None, **kwargs):
        """
        e.g. Collect(keys=[coord], feat_keys=[coord, color])
        """
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="coord")
        self.keys = keys
        self.offset_keys = offset_keys_dict
        self.kwargs = kwargs
    #@profile
    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            data[key] = data_dict[key]
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        return data


class GridSample(object):
    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        keys=("coord", "color"),
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
    ):
        if mode == 'train':
            pass
        else:
            mode = 'test'        
        
        
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement
    #@profile
    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == "train":  # train mode
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx_unique = np.unique(
                    np.append(idx_unique, data_dict["sampled_index"])
                )
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_unique])[0]
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                displacement = (
                    scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = np.sum(
                        displacement * data_dict["normal"], axis=-1, keepdims=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
            for key in self.keys:
                data_dict[key] = data_dict[key][idx_unique]
            return data_dict

        elif self.mode == "test":  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = dict(index=idx_part)
                if self.return_inverse:
                    data_dict["inverse"] = np.zeros_like(inverse)
                    data_dict["inverse"][idx_sort] = inverse
                if self.return_grid_coord:
                    data_part["grid_coord"] = grid_coord[idx_part]
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                if self.return_displacement:
                    displacement = (
                        scaled_coord - grid_coord - 0.5
                    )  # [0, 1] -> [-0.5, 0.5] displacement to center
                    if self.project_displacement:
                        displacement = np.sum(
                            displacement * data_dict["normal"], axis=-1, keepdims=True
                        )
                    data_dict["displacement"] = displacement[idx_part]
                for key in data_dict.keys():
                    if key in self.keys:
                        data_part[key] = data_dict[key][idx_part]
                    else:
                        data_part[key] = data_dict[key]
                data_part_list.append(data_part)
            if len(data_part_list) > 0:
                return data_part_list[0] 
            else:
                return data_dict
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr



class SphereCrop(object):
    def __init__(self, point_max=80000, sample_rate=None, mode="random"):
        self.point_max = point_max
        self.sample_rate = sample_rate
        assert mode in ["random", "center", "all"]
        self.mode = mode
    #@profile
    def __call__(self, data_dict):
        point_max = (
            int(self.sample_rate * data_dict["coord"].shape[0])
            if self.sample_rate is not None
            else self.point_max
        )

        assert "coord" in data_dict.keys()
        if self.mode == "all":
            # TODO: Optimize
            if "index" not in data_dict.keys():
                data_dict["index"] = np.arange(data_dict["coord"].shape[0])
            data_part_list = []
            # coord_list, color_list, dist2_list, idx_list, offset_list = [], [], [], [], []
            if data_dict["coord"].shape[0] > point_max:
                coord_p, idx_uni = np.random.rand(
                    data_dict["coord"].shape[0]
                ) * 1e-3, np.array([])
                while idx_uni.size != data_dict["index"].shape[0]:
                    init_idx = np.argmin(coord_p)
                    dist2 = np.sum(
                        np.power(data_dict["coord"] - data_dict["coord"][init_idx], 2),
                        1,
                    )
                    idx_crop = np.argsort(dist2)[:point_max]

                    data_crop_dict = dict()
                    if "coord" in data_dict.keys():
                        data_crop_dict["coord"] = data_dict["coord"][idx_crop]
                    if "grid_coord" in data_dict.keys():
                        data_crop_dict["grid_coord"] = data_dict["grid_coord"][idx_crop]
                    if "normal" in data_dict.keys():
                        data_crop_dict["normal"] = data_dict["normal"][idx_crop]
                    if "color" in data_dict.keys():
                        data_crop_dict["color"] = data_dict["color"][idx_crop]
                    if "displacement" in data_dict.keys():
                        data_crop_dict["displacement"] = data_dict["displacement"][
                            idx_crop
                        ]
                    if "strength" in data_dict.keys():
                        data_crop_dict["strength"] = data_dict["strength"][idx_crop]
                    data_crop_dict["weight"] = dist2[idx_crop]
                    data_crop_dict["index"] = data_dict["index"][idx_crop]
                    data_part_list.append(data_crop_dict)

                    delta = np.square(
                        1 - data_crop_dict["weight"] / np.max(data_crop_dict["weight"])
                    )
                    coord_p[idx_crop] += delta
                    idx_uni = np.unique(
                        np.concatenate((idx_uni, data_crop_dict["index"]))
                    )
            else:
                data_crop_dict = data_dict.copy()
                data_crop_dict["weight"] = np.zeros(data_dict["coord"].shape[0])
                data_crop_dict["index"] = data_dict["index"]
                data_part_list.append(data_crop_dict)
            return data_part_list
        # mode is "random" or "center"
        elif data_dict["coord"].shape[0] > point_max:
            if self.mode == "random":
                center = data_dict["coord"][
                    np.random.randint(data_dict["coord"].shape[0])
                ]
            elif self.mode == "center":
                center = data_dict["coord"][data_dict["coord"].shape[0] // 2]
            else:
                raise NotImplementedError
            idx_crop = np.argsort(np.sum(np.square(data_dict["coord"] - center), 1))[
                :point_max
            ]
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx_crop]
            if "origin_coord" in data_dict.keys():
                data_dict["origin_coord"] = data_dict["origin_coord"][idx_crop]
            if "grid_coord" in data_dict.keys():
                data_dict["grid_coord"] = data_dict["grid_coord"][idx_crop]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx_crop]
            if "normal" in data_dict.keys():
                data_dict["normal"] = data_dict["normal"][idx_crop]
            if "segment" in data_dict.keys():
                data_dict["segment"] = data_dict["segment"][idx_crop]
            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][idx_crop]
            if "displacement" in data_dict.keys():
                data_dict["displacement"] = data_dict["displacement"][idx_crop]
            if "strength" in data_dict.keys():
                data_dict["strength"] = data_dict["strength"][idx_crop]
        return data_dict
    

class CameraInfo():
    """ Camera intrisics for point cloud creation. """
    #@profile
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


#@profile
def create_point_cloud_from_depth_image(depth, camera, organized=True):
    """ Generate point cloud using depth image only.

        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            camera: [CameraInfo]
                camera intrinsics
            organized: bool
                whether to keep the cloud in image shape (H,W,3)

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    assert (depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud



#@profile
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

@numba.njit(cache=True)
def _get_workspace_mask_njit(cloud_flat, seg_flat, trans, outlier):
    n = cloud_flat.shape[0]
    mask = np.zeros(n, dtype=np.bool_)

    # 1. Bounding Box 초기화
    xmin, ymin, zmin = np.inf, np.inf, np.inf
    xmax, ymax, zmax = -np.inf, -np.inf, -np.inf

    # 2. 변환 행렬 캐싱 (반복문 안에서 연산 최소화)
    if trans is not None:
        r00, r01, r02, tx = trans[0,0], trans[0,1], trans[0,2], trans[0,3]
        r10, r11, r12, ty = trans[1,0], trans[1,1], trans[1,2], trans[1,3]
        r20, r21, r22, tz = trans[2,0], trans[2,1], trans[2,2], trans[2,3]

    # 3. Foreground의 Bounding Box 계산 (변환과 동시에)
    has_fg = False
    for i in range(n):
        if seg_flat[i] > 0:
            has_fg = True
            x, y, z = cloud_flat[i, 0], cloud_flat[i, 1], cloud_flat[i, 2]
            
            if trans is not None:
                px = r00*x + r01*y + r02*z + tx
                py = r10*x + r11*y + r12*z + ty
                pz = r20*x + r21*y + r22*z + tz
            else:
                px, py, pz = x, y, z

            if px < xmin: xmin = px
            if px > xmax: xmax = px
            if py < ymin: ymin = py
            if py > ymax: ymax = py
            if pz < zmin: zmin = pz
            if pz > zmax: zmax = pz

    if not has_fg:
        return mask

    # Outlier 여유분 추가
    xmin -= outlier
    xmax += outlier
    ymin -= outlier
    ymax += outlier
    zmin -= outlier
    zmax += outlier

    # 4. 전체 포인트 마스킹 (메모리 할당 없이 한 번에 평가)
    for i in range(n):
        x, y, z = cloud_flat[i, 0], cloud_flat[i, 1], cloud_flat[i, 2]
        
        if trans is not None:
            px = r00*x + r01*y + r02*z + tx
            py = r10*x + r11*y + r12*z + ty
            pz = r20*x + r21*y + r22*z + tz
        else:
            px, py, pz = x, y, z

        if (xmin < px < xmax) and (ymin < py < ymax) and (zmin < pz < zmax):
            mask[i] = True

    return mask

# 기존 코드와 호환성을 유지하기 위한 Wrapper 함수
def get_workspace_mask(cloud, seg, trans=None, outlier=0.0):
    h, w, _ = cloud.shape
    cloud_flat = cloud.reshape(-1, 3)
    seg_flat = seg.reshape(-1)
    
    # Numba 엔진 태우기
    mask_flat = _get_workspace_mask_njit(cloud_flat, seg_flat, trans, float(outlier))
    
    return mask_flat.reshape(h, w)

# #@profile
# def get_workspace_mask(cloud, seg, trans=None, outlier=0):
#     """ Keep points in workspace as input.
#         Workspace here refers to the space with objects.

#         Input:
#             cloud: [np.ndarray, (H,W,3), np.float32]
#                 scene point cloud
#             seg: [np.ndarray, (H,W,), np.uint8]
#                 segmantation label of scene points
#             trans: [np.ndarray, (4,4), np.float32]
#                 transformation matrix for scene points, default: None.
#             organized: [bool]
#                 whether to keep the cloud in image shape (H,W,3)
#             outlier: [float]
#                 if the distance between a point and workspace is greater than outlier, the point will be removed

#         Output:
#             workspace_mask: [np.ndarray, (H,W)/(H*W,), np.bool]
#                 mask to indicate whether scene points are in workspace
#     """
#     h, w, _ = cloud.shape
#     cloud = cloud.reshape([h * w, 3])
#     seg = seg.reshape(h * w)
#     if trans is not None:
#         cloud = transform_point_cloud_4x4_njit(cloud, trans)
#     foreground = cloud[seg > 0]
#     xmin, ymin, zmin = foreground.min(axis=0)
#     xmax, ymax, zmax = foreground.max(axis=0)
#     mask_x = ((cloud[:, 0] > xmin - outlier) & (cloud[:, 0] < xmax + outlier))
#     mask_y = ((cloud[:, 1] > ymin - outlier) & (cloud[:, 1] < ymax + outlier))
#     mask_z = ((cloud[:, 2] > zmin - outlier) & (cloud[:, 2] < zmax + outlier))
#     workspace_mask = (mask_x & mask_y & mask_z)
#     workspace_mask = workspace_mask.reshape([h, w])

#     return workspace_mask


import open3d as o3d
import numpy as np
import copy

def visualize_masked_cloud(
    cloud,
    mask,
    color=[1.0, 0.0, 0.0],
    title="Masked Cloud",
    base_cloud=None,
    base_color=[0.6, 0.6, 0.6]
):
    """
    cloud: (N,3)
    mask:  (N,) bool
    """

    geoms = []

    if base_cloud is not None:
        base_pcd = o3d.geometry.PointCloud()
        base_pcd.points = o3d.utility.Vector3dVector(base_cloud.astype(np.float64))
        base_pcd.paint_uniform_color(base_color)
        geoms.append(base_pcd)

    masked_pcd = o3d.geometry.PointCloud()
    masked_pcd.points = o3d.utility.Vector3dVector(
        cloud[mask].astype(np.float64)
    )
    masked_pcd.paint_uniform_color(color)

    geoms.append(masked_pcd)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geoms.append(coord)

    o3d.visualization.draw_geometries(geoms, window_name=title)

# def get_target_surrounding_mask(cloud, seg, obj_id ,trans=None, outlier=0, debug=False):
#     """ Keep points in workspace as input.
#         Workspace here refers to the space with objects.

#         Input:
#             cloud: [np.ndarray, (H,W,3), np.float32]
#                 scene point cloud
#             seg: [np.ndarray, (H,W,), np.uint8]
#                 segmantation label of scene points
#             trans: [np.ndarray, (4,4), np.float32]
#                 transformation matrix for scene points, default: None.
#             organized: [bool]
#                 whether to keep the cloud in image shape (H,W,3)
#             outlier: [float]
#                 if the distance between a point and workspace is greater than outlier, the point will be removed

#         Output:
#             workspace_mask: [np.ndarray, (H,W)/(H*W,), np.bool]
#                 mask to indicate whether scene points are in workspace
#     """
#     h, w, _ = cloud.shape
#     cloud = cloud.reshape([h * w, 3])
#     seg = seg.reshape(h * w)
#     if trans is not None:
#         cloud = transform_point_cloud_4x4_njit(cloud, trans)
#     foreground = cloud[seg == obj_id]
#     center = foreground.mean(axis=0)
#     dist = np.linalg.norm(foreground - center, axis=1)

#     r = np.percentile(dist, 97)  # 상위 5% 제거
#     foreground_filtered = foreground[dist < r]
#     if foreground_filtered.shape[0] > 0:
#             foreground = foreground_filtered
#     else:
#         # 필터링했더니 다 사라지면, 그냥 원본(foreground)을 씀
#         pass

#     xmin, ymin, zmin = foreground.min(axis=0)
#     xmax, ymax, zmax = foreground.max(axis=0)
#     mask_x = ((cloud[:, 0] > xmin - outlier) & (cloud[:, 0] < xmax + outlier))
#     mask_y = ((cloud[:, 1] > ymin - outlier) & (cloud[:, 1] < ymax + outlier))
#     mask_z = ((cloud[:, 2] > zmin - outlier) & (cloud[:, 2] < zmax + outlier))
#     #mask_z = cloud[:, 2]
#     workspace_mask = (mask_x & mask_y & mask_z)
#     #workspace_mask = (mask_x & mask_y)

#     workspace_mask = workspace_mask.reshape([h, w])

#     if debug:
#         foreground_mask = (seg == obj_id)
#         visualize_masked_cloud(
#             cloud,
#             foreground_mask,
#             color=[1.0, 0.0, 0.0],
#             title="Target Foreground",
#             base_cloud=cloud
#         )
#         # visualize_masked_cloud(cloud, workspace_mask,
#         #                     title="Final Surrounding Mask",
#         #                     base_cloud=cloud)
#     return workspace_mask

# @numba.njit(cache=True)
# def transform_point_cloud_4x4_njit(cloud, transform):
#     mat_3x4 = transform[:3, :]
#     mat_3x4_T_contig = np.ascontiguousarray(mat_3x4.T)
#     ones = np.ones((cloud.shape[0], 1), dtype=cloud.dtype)
#     cloud_homogeneous = np.concatenate((cloud, ones), axis=1)
#     cloud_transformed = np.dot(cloud_homogeneous, mat_3x4_T_contig)
#     return cloud_transformed

@numba.njit(cache=True)
def _apply_surrounding_bbox_mask_njit(cloud_flat, trans, bounds):
    n = cloud_flat.shape[0]
    mask = np.zeros(n, dtype=np.bool_)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    
    if trans is not None:
        r00, r01, r02, tx = trans[0,0], trans[0,1], trans[0,2], trans[0,3]
        r10, r11, r12, ty = trans[1,0], trans[1,1], trans[1,2], trans[1,3]
        r20, r21, r22, tz = trans[2,0], trans[2,1], trans[2,2], trans[2,3]

    for i in range(n):
        x, y, z = cloud_flat[i, 0], cloud_flat[i, 1], cloud_flat[i, 2]
        
        if trans is not None:
            px = r00*x + r01*y + r02*z + tx
            py = r10*x + r11*y + r12*z + ty
            pz = r20*x + r21*y + r22*z + tz
        else:
            px, py, pz = x, y, z

        if (px > xmin) and (px < xmax) and (py > ymin) and (py < ymax) and (pz > zmin) and (pz < zmax):
            mask[i] = True
            
    return mask

def get_target_surrounding_mask(cloud, seg, obj_id, trans=None, outlier=0.0, debug=False):
    h, w, _ = cloud.shape
    cloud_flat = cloud.reshape(-1, 3)
    seg_flat = seg.reshape(-1)
    
    # 1. 대상 객체(Foreground)만 먼저 추출
    fg_mask = (seg_flat == obj_id)
    fg_points = cloud_flat[fg_mask]
    
    if len(fg_points) == 0:
        return np.zeros((h, w), dtype=bool)
        
    # 2. Foreground 포인트에만 Trans 적용
    if trans is not None:
        fg_points = (fg_points @ trans[:3, :3].T) + trans[:3, 3]
        
    # 3. Outlier 필터링 최적화 (버그 수정됨 ✅)
    center = fg_points.mean(axis=0)
    dist = np.linalg.norm(fg_points - center, axis=1)
    
    if len(dist) > 1000:
        # np.random.choice 대신 전체 길이에서 대략 1000개가 되도록 등간격 추출
        step = len(dist) // 1000
        sample_dist = dist[::step] 
        r = np.percentile(sample_dist, 97)
    else:
        r = np.percentile(dist, 97)

    fg_filtered = fg_points[dist < r]
    if len(fg_filtered) == 0:
        fg_filtered = fg_points

    # 4. Bounding Box 계산 및 Numba 함수 호출
    xmin, ymin, zmin = fg_filtered.min(axis=0) - outlier
    xmax, ymax, zmax = fg_filtered.max(axis=0) + outlier
    bounds = np.array([xmin, xmax, ymin, ymax, zmin, zmax], dtype=np.float32)

    # 5. 메모리 할당 없는 C언어급 마스킹 연산 (_apply_surrounding_bbox_mask_njit는 그대로 사용)
    mask_flat = _apply_surrounding_bbox_mask_njit(cloud_flat, trans, bounds)
    
    workspace_mask = mask_flat.reshape(h, w)

    return workspace_mask
