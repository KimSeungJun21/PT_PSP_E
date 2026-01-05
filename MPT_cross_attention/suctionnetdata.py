
import os,sys
import suctionnetAPI
import open3d as o3d
import numpy as np
from suctionnetAPI import SuctionNet

sys.path.append('/home/kimseungjun/task/PointTransformer/suctionnetAPI')
from suctionnetAPI.utils.utils import (
    generate_scene_model, 
    plot_sucker_collision, 
    transform_points, 
    parse_posevector, 
    create_table_cloud, 
    get_model_suctions,
    #loadScenePointCloud
)

from suctionnetAPI.utils.rotation import (
    viewpoint_to_matrix
)


root = '/home/kimseungjun/datasets/graspnet_data/suctionnet' 
scene_idx = 0
anno_idx = 0
camera='kinect'
visu_num_each=5
'''
**Input:**

- scene_idx: int of the scene index.

- anno_idx: int of the annotation index.

- camera: string of the camera type, 'realsense' or 'kinect'.

- visu_num_each: int of the number of suctions to viualize on each object'.

**Output:**

- No output but the 3D visualization of the scene and collision labels will show up.
'''

scene_name = 'scene_%04d' % scene_idx

camera_poses = np.load(os.path.join(root, 'scenes', scene_name, camera, 'camera_poses.npy'.format(camera)))
camera_pose = camera_poses[anno_idx]
dataset_root = '/home/kimseungjun/datasets/graspnet_data/suctionnet'

intrinsics = np.load(os.path.join(root, 'scenes', scene_name, camera, 'camK.npy'))

align_mat = np.load(os.path.join(dataset_root, 'scenes', scene_name, camera, 'cam0_wrt_table.npy'))
camera_pose = align_mat.dot(camera_poses[anno_idx])

#camera_pose = np.matmul(camera_poses[anno_idx], align_mat)
#camera_pose = np.matmul(np.linalg.inv(align_mat), camera_poses[anno_idx])
T_world_to_cam = np.linalg.inv(camera_pose)
model_list, obj_list, pose_list = generate_scene_model(root, scene_name, anno_idx, return_poses=True, camera=camera, align=False)

sn = SuctionNet(root=dataset_root, camera='kinect')

scene_pcd = sn.loadScenePointCloud(sceneId=scene_idx, camera='kinect', annId=anno_idx, format='open3d')

scene_pcd.transform(T_world_to_cam)

scene_pcd = scene_pcd.voxel_down_sample(voxel_size=0.005)

plane_model, inliers = scene_pcd.segment_plane(distance_threshold=0.015,
                                             ransac_n=3,
                                             num_iterations=1000)
table = scene_pcd.select_by_index(inliers)
table.paint_uniform_color([0.2, 0.2, 0.2]) # 시각화를 위해 어두운 회색으로 변경

[a, b, c, d] = plane_model # 평면 방정식: ax + by + cz + d = 0

# 물체 주변 영역에만 그리드 생성 (예: -0.5 ~ 0.5 범위)
grid_size = 0.01
x = np.arange(-0.5, 0.5, grid_size)
y = np.arange(-0.5, 0.5, grid_size)
gx, gy = np.meshgrid(x, y)

# 평면 방정식에 맞춰 z값 계산 (z = -(ax + by + d) / c)
gz = -(a * gx + b * gy + d) / c -0.02

# 가상 포인트들을 Open3D 객체로 변환
points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
table = o3d.geometry.PointCloud()
table.points = o3d.utility.Vector3dVector(points)
table.paint_uniform_color([0.1, 0.1, 0.1])


collision_dir = os.path.join(root, 'suction_collision_label')
collision_dump = np.load(os.path.join(collision_dir, '{:04d}_collision.npz'.format(scene_idx)))

radius = 0.01
height = 0.1

num_obj = len(obj_list)

for model in model_list:
    model.transform(T_world_to_cam)

for obj_i in range(len(obj_list)):
    suckers = []
    print('Checking ' + str(obj_i+1) + ' / ' + str(num_obj))
    obj_idx = obj_list[obj_i]
    trans = pose_list[obj_i]

    trans_world = pose_list[obj_i]
    trans_cam = np.dot(T_world_to_cam, trans_world)

    seal_dir = os.path.join(root, 'seal_label')
    sampled_points, normals, _, _ = get_model_suctions('%s/%03d_seal.npz'%(seal_dir, obj_idx))
    collisions = collision_dump['arr_{}'.format(obj_i)]

    point_inds = np.random.choice(sampled_points.shape[0], visu_num_each)
    np.random.shuffle(point_inds)
    
    sucker_params = []

    for point_ind in point_inds:
        target_point = sampled_points[point_ind]
        normal = normals[point_ind]
        # score = scores[point_ind]
        collision = collisions[point_ind]

        R = viewpoint_to_matrix(normal)
        t = transform_points(target_point[np.newaxis,:], trans_cam).squeeze()
        #t = transform_points(target_point[np.newaxis,:], trans).squeeze()
        #R = np.dot(trans[:3,:3], R)
        R = np.dot(trans_cam[:3,:3], R)
        sucker = plot_sucker_collision(R, t, collision, radius, height)
        suckers.append(sucker)
        sucker_params.append([target_point[0],target_point[1],target_point[2],normal[0],normal[1],normal[2],radius, height])
        
    o3d.visualization.draw_geometries([table, *model_list, *suckers], width=1536, height=864)