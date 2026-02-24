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

@numba.njit(cache=True)
def transform_point_cloud_fast(cloud, transform):
    """
    Numba optimized point cloud transformation.
    Auto-detects format based on transform.shape.
    """
    num_points = cloud.shape[0]
    out = np.empty((num_points, 3), dtype=cloud.dtype)
    
    # Transform shape checking (compile-time constant check optimization in Numba)
    r_idx = transform.shape[0]
    c_idx = transform.shape[1]

    # Case 1: Rotation Only (3x3)
    if r_idx == 3 and c_idx == 3:
        for i in range(num_points):
            # Unrolled matrix multiplication for speed
            x = cloud[i, 0]
            y = cloud[i, 1]
            z = cloud[i, 2]
            
            out[i, 0] = transform[0, 0] * x + transform[0, 1] * y + transform[0, 2] * z
            out[i, 1] = transform[1, 0] * x + transform[1, 1] * y + transform[1, 2] * z
            out[i, 2] = transform[2, 0] * x + transform[2, 1] * y + transform[2, 2] * z

    # Case 2: Rotation + Translation (3x4 or 4x4)
    elif (r_idx == 3 and c_idx == 4) or (r_idx == 4 and c_idx == 4):
        for i in range(num_points):
            x = cloud[i, 0]
            y = cloud[i, 1]
            z = cloud[i, 2]
            
            # Rotation + Translation (avoiding homogenous coordinate creation)
            out[i, 0] = (transform[0, 0] * x + transform[0, 1] * y + transform[0, 2] * z) + transform[0, 3]
            out[i, 1] = (transform[1, 0] * x + transform[1, 1] * y + transform[1, 2] * z) + transform[1, 3]
            out[i, 2] = (transform[2, 0] * x + transform[2, 1] * y + transform[2, 2] * z) + transform[2, 3]
            
    return out


@numba.njit(fastmath=True, cache=True)
def batch_get_wrench_score(suction_points, directions, center, g_direction, g, wrench_thre):
    """
    Numba optimized version of batch_get_wrench_score.
    메모리 할당 없이 루프 한 번에 모든 계산을 끝냅니다.
    """
    num_points = suction_points.shape[0]
    scores = np.empty(num_points, dtype=np.float32)
    
    # 중력 벡터 미리 계산
    gravity_vec_x = g_direction[0] * g
    gravity_vec_y = g_direction[1] * g
    gravity_vec_z = g_direction[2] * g

    # Up Vector (임시 변수)
    up_x, up_y, up_z = 0.0, 1.0, 0.0

    for i in range(num_points):
        # 1. Directions Normalization (X axis)
        dx = directions[i, 0]
        dy = directions[i, 1]
        dz = directions[i, 2]
        
        norm_x = (dx*dx + dy*dy + dz*dz)**0.5
        if norm_x > 1e-8:
            axis_x_0 = dx / norm_x
            axis_x_1 = dy / norm_x
            axis_x_2 = dz / norm_x
        else:
            axis_x_0, axis_x_1, axis_x_2 = 1.0, 0.0, 0.0

        # 2. Handle Singularity for Up Vector
        # 만약 방향 벡터가 수직(Y축)에 가까우면 Up Vector를 X축(1,0,0)으로 변경
        if abs(axis_x_1) > 0.99:
            up_x, up_y, up_z = 1.0, 0.0, 0.0
        else:
            up_x, up_y, up_z = 0.0, 1.0, 0.0

        # 3. Z axis calculation (Cross Product: X cross UP)
        # axis_z = cross(axis_x, up)
        axis_z_0 = axis_x_1 * up_z - axis_x_2 * up_y
        axis_z_1 = axis_x_2 * up_x - axis_x_0 * up_z
        axis_z_2 = axis_x_0 * up_y - axis_x_1 * up_x
        
        # Normalize Z
        norm_z = (axis_z_0**2 + axis_z_1**2 + axis_z_2**2)**0.5
        if norm_z > 1e-8:
            axis_z_0 /= norm_z
            axis_z_1 /= norm_z
            axis_z_2 /= norm_z
            
        # 4. Y axis calculation (Cross Product: Z cross X) - 이미 직교하므로 정규화 불필요
        axis_y_0 = axis_z_1 * axis_x_2 - axis_z_2 * axis_x_1
        axis_y_1 = axis_z_2 * axis_x_0 - axis_z_0 * axis_x_2
        axis_y_2 = axis_z_0 * axis_x_1 - axis_z_1 * axis_x_0

        # --- 좌표 변환 행렬을 만들지 않고 바로 내적(Projection) 수행 ---
        
        # Vector from Suction Point to Center of Mass
        # vec = center - suction_points[i]
        vec_x = center[0] - suction_points[i, 0]
        vec_y = center[1] - suction_points[i, 1]
        vec_z = center[2] - suction_points[i, 2]

        # Project Vec to Local Frame (coord)
        # coord_x = dot(vec, axis_x)
        # coord_y = dot(vec, axis_y)
        # coord_z = dot(vec, axis_z)
        coord_local_0 = vec_x * axis_x_0 + vec_y * axis_x_1 + vec_z * axis_x_2
        coord_local_1 = vec_x * axis_y_0 + vec_y * axis_y_1 + vec_z * axis_y_2
        coord_local_2 = vec_x * axis_z_0 + vec_y * axis_z_1 + vec_z * axis_z_2

        # Project Gravity to Local Frame (gravity_proj)
        # grav_local = dot(gravity, axis_x/y/z)
        grav_local_0 = gravity_vec_x * axis_x_0 + gravity_vec_y * axis_x_1 + gravity_vec_z * axis_x_2
        grav_local_1 = gravity_vec_x * axis_y_0 + gravity_vec_y * axis_y_1 + gravity_vec_z * axis_y_2
        # grav_local_2는 torque 식에서 쓰이지 않으면 계산 생략 가능하지만, torque_y 식에 쓰임
        grav_local_2 = gravity_vec_x * axis_z_0 + gravity_vec_y * axis_z_1 + gravity_vec_z * axis_z_2

        # 5. Torque Calculation
        # torque_y = G_x * P_z - G_z * P_x
        t_y = grav_local_0 * coord_local_2 - grav_local_2 * coord_local_0
        
        # torque_z = -G_x * P_y + G_y * P_x
        t_z = -grav_local_0 * coord_local_1 + grav_local_1 * coord_local_0

        # 6. Score Calculation
        t_max = max(abs(t_z), abs(t_y))
        
        # Clip and store
        val = t_max / wrench_thre
        if val > 1.0:
            val = 1.0
        scores[i] = 1.0 - val

    return scores