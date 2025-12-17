import numpy as np
import torch
import open3d as o3d

# -------------------------
# utils
# -------------------------
def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def normalize_color(c):
    """color가 0~255 또는 0~1이 섞여있어도 0~1로 통일"""
    c = to_numpy(c)
    if c is None:
        return None
    c = c.astype(np.float32)
    if c.ndim == 2 and c.shape[1] >= 3:
        c = c[:, :3]
    else:
        return None
    if c.max() > 1.5:  # 대충 255 스케일 판단
        c = c / 255.0
    c = np.clip(c, 0.0, 1.0)
    return c

def label_to_color(labels, seed=0):
    """
    정수 라벨(N,) -> 랜덤 고정 팔레트 RGB(N,3) (0~1)
    - -1 같은 invalid는 회색으로
    """
    labels = to_numpy(labels).reshape(-1)
    labels = labels.astype(np.int64)

    colors = np.zeros((labels.shape[0], 3), dtype=np.float32)
    invalid = labels < 0

    uniq = np.unique(labels[~invalid])
    rng = np.random.RandomState(seed)

    # 라벨 -> 색 매핑
    lut = {}
    for u in uniq:
        lut[int(u)] = rng.rand(3).astype(np.float32)

    for i, v in enumerate(labels):
        if v < 0:
            colors[i] = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        else:
            colors[i] = lut[int(v)]
    return colors

def build_o3d_pcd(coords, colors=None, voxel_size=None, max_points=None, seed=0):
    coords = to_numpy(coords).astype(np.float32)
    if coords.ndim != 2 or coords.shape[1] < 3:
        raise ValueError(f"coord shape이 이상함: {coords.shape}")
    pts = coords[:, :3]

    # 다운샘플(랜덤)
    if max_points is not None and pts.shape[0] > max_points:
        rng = np.random.RandomState(seed)
        idx = rng.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
        if colors is not None:
            colors = colors[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

    if colors is not None:
        colors = colors.astype(np.float32)
        if colors.shape[0] != pts.shape[0]:
            raise ValueError(f"colors/points 개수 불일치: {colors.shape[0]} vs {pts.shape[0]}")
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    # voxel downsample (선택)
    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))

    return pcd

# -------------------------
# main visualizer
# -------------------------
def vis_datas_open3d(
    datas: dict,
    mode="semantic",          # "rgb" | "semantic" | "instance"
    max_points=500_000,
    voxel_size=None,
    seed=0,
    semantic_filter=None,     # 예: [1, 5, 7]
    instance_filter=None,     # 예: [3, 10]
    show_axes=True,
):
    """
    datas keys (너 코드 기준):
      - coord: (N,3 or N,>=3)
      - semantic_gt: (N,) or (N,1) or reshape 가능
      - instance_gt: (N,) ...
      - color: (N,3) 0~255 or 0~1
    """
    if "coord" not in datas:
        raise KeyError("datas에 'coord'가 없음")

    coord = datas["coord"]

    # 라벨들
    sem = datas.get("semantic_gt", None)
    ins = datas.get("instance_gt", None)
    col = datas.get("color", None)

    sem = to_numpy(sem).reshape(-1) if sem is not None else None
    ins = to_numpy(ins).reshape(-1) if ins is not None else None
    col = normalize_color(col)

    # 필터링 마스크
    N = to_numpy(coord).shape[0]
    mask = np.ones((N,), dtype=bool)

    if semantic_filter is not None:
        if sem is None:
            raise ValueError("semantic_filter를 썼는데 semantic_gt가 없음")
        semantic_filter = set(map(int, semantic_filter))
        mask &= np.isin(sem.astype(np.int64), list(semantic_filter))

    if instance_filter is not None:
        if ins is None:
            raise ValueError("instance_filter를 썼는데 instance_gt가 없음")
        instance_filter = set(map(int, instance_filter))
        mask &= np.isin(ins.astype(np.int64), list(instance_filter))

    # mask 적용
    coord_np = to_numpy(coord)
    coord_np = coord_np[mask]
    if sem is not None:
        sem = sem[mask]
    if ins is not None:
        ins = ins[mask]
    if col is not None:
        col = col[mask]

    # 모드별 색 구성
    if mode == "rgb":
        if col is None:
            raise ValueError("mode='rgb'인데 datas에 color가 없음")
        colors = col

    elif mode == "semantic":
        if sem is None:
            raise ValueError("mode='semantic'인데 datas에 semantic_gt가 없음")
        colors = label_to_color(sem, seed=seed)

    elif mode == "instance":
        if ins is None:
            raise ValueError("mode='instance'인데 datas에 instance_gt가 없음")
        colors = label_to_color(ins, seed=seed)

    else:
        raise ValueError("mode는 'rgb' | 'semantic' | 'instance' 중 하나")

    pcd = build_o3d_pcd(
        coords=coord_np,
        colors=colors,
        voxel_size=voxel_size,
        max_points=max_points,
        seed=seed,
    )

    geoms = [pcd]
    if show_axes:
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))

    o3d.visualization.draw_geometries(geoms)



def vis_pth_sample(
    pth_path,
    safe_torch_load=None,  # 네가 쓰는 safe_torch_load 넣어도 되고, None이면 torch.load 사용
    mode="semantic",
    **kwargs
):
    if safe_torch_load is None:
        datas = torch.load(pth_path, map_location="cpu")
    else:
        datas = safe_torch_load(pth_path, map_location="cpu")

    # 키 이름 너 코드에 맞춰 통일
    # (혹시 semantic_gt/instance_gt가 없고 segment/instance로 되어있으면 자동 대응)
    if "semantic_gt" not in datas and "segment" in datas:
        datas["semantic_gt"] = datas["segment"]
    if "instance_gt" not in datas and "instance" in datas:
        datas["instance_gt"] = datas["instance"]

    vis_datas_open3d(datas, mode=mode, **kwargs)

# 사용 예:
vis_pth_sample("/home/kimseungjun/datasets/My_PT_data/PT_data/train/0000012632_20250918_152808.pth",
               mode="semantic", max_points=300000, voxel_size=0.01)

